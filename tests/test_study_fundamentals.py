import numpy as np
import pandas as pd
import pytest

from earningsiq.study.events import attach_earnings_releases
from earningsiq.study.features import LLM_TONE_PROMPT, MDNA_SOURCE
from earningsiq.study.fundamentals import attach_surprises, quarterly_values, reported_periods, seasonal_surprise
from earningsiq.study.press_releases import extract_press_release


def fact(start, end, value, filed):
    return {"start": start, "end": end, "val": value, "filed": filed}


def test_reported_periods_keeps_latest_filed_value_and_prefers_first_tag():
    us_gaap = {
        "NetIncomeLoss": {"units": {"USD": [
            fact("2023-01-01", "2023-03-31", 100, "2023-05-01"),
            fact("2023-01-01", "2023-03-31", 110, "2024-05-01"),   # restated a year later
        ]}},
        "ProfitLoss": {"units": {"USD": [
            fact("2023-01-01", "2023-03-31", 999, "2024-06-01"),
            fact("2023-04-01", "2023-06-30", 120, "2023-08-01"),   # only this tag has Q2
        ]}},
    }

    periods = reported_periods(us_gaap, ["NetIncomeLoss", "ProfitLoss"]).set_index("end")["value"]

    assert periods[pd.Timestamp("2023-03-31")] == 110
    assert periods[pd.Timestamp("2023-06-30")] == 120


def test_fourth_quarter_is_full_year_minus_nine_months():
    periods = pd.DataFrame({
        "start": pd.to_datetime(["2023-01-01", "2023-04-01", "2023-07-01", "2023-01-01", "2023-01-01"]),
        "end": pd.to_datetime(["2023-03-31", "2023-06-30", "2023-09-30", "2023-09-30", "2023-12-31"]),
        "value": [10.0, 20.0, 30.0, 60.0, 100.0],   # Q1, Q2, Q3, nine months, full year
    })

    quarters = quarterly_values(periods)

    assert quarters.tolist() == [10.0, 20.0, 30.0, 40.0]
    assert quarters.index[-1] == pd.Timestamp("2023-12-31")


def test_seasonal_surprise_compares_with_the_same_quarter_a_year_earlier():
    ends = pd.date_range("2019-03-31", periods=16, freq="QE")
    # steady year-over-year growth of +4, +5, +4, +5 ... then a jump in the last quarter
    values = [100, 100, 100, 100, 104, 105, 104, 105, 108, 110, 108, 110, 112, 115, 112, 135]
    quarters = pd.Series(values, index=ends, dtype=float)

    result = seasonal_surprise(quarters)

    assert result["surprise"].iloc[:8].isna().all()            # needs a year plus four past changes
    assert result["growth"].iloc[4] == pytest.approx(0.04)
    assert abs(result["surprise"].iloc[-2]) < 3                # an ordinary quarter
    assert result["surprise"].iloc[-1] > 20                    # +23 where +4 or +5 is usual


def test_attach_surprises_matches_fiscal_period_within_tolerance():
    events = pd.DataFrame({"ticker": ["AAA", "AAA", "BBB"], "period_end": pd.to_datetime(["2024-03-30", "2024-09-30", "2024-03-31"])})
    surprises = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "period_end": pd.to_datetime(["2024-03-31", "2024-06-30"]),
        "income_surprise": [1.5, -0.5],
    })

    matched = attach_surprises(events, surprises)["income_surprise"]

    assert matched[0] == 1.5
    assert np.isnan(matched[1]) and np.isnan(matched[2])


def test_extract_press_release_prefers_exhibit_99_1():
    submission = (
        b"<DOCUMENT>\n<TYPE>8-K\n<TEXT><html><body><p>Cover page</p></body></html></TEXT>\n</DOCUMENT>\n"
        b"<DOCUMENT>\n<TYPE>EX-99.2\n<TEXT><html><body><p>Slides</p></body></html></TEXT>\n</DOCUMENT>\n"
        b"<DOCUMENT>\n<TYPE>EX-99.1\n<SEQUENCE>2\n<TEXT><html><body><p>Revenue grew 10%.</p></body></html></TEXT>\n</DOCUMENT>\n"
    )

    assert extract_press_release(submission) == "Revenue grew 10%."
    assert extract_press_release(b"<DOCUMENT>\n<TYPE>8-K\n<TEXT>x</TEXT>\n</DOCUMENT>") == ""


def test_release_match_carries_accession_and_cik():
    events = pd.DataFrame({"ticker": ["AAA"], "filing_accepted": pd.to_datetime(["2024-05-03 06:00"])})
    releases = pd.DataFrame({
        "ticker": ["AAA", "AAA"],
        "cik": [42, 42],
        "release_accession": ["old", "new"],
        "release_accepted": pd.to_datetime(["2024-02-01 16:30", "2024-05-02 16:30"]),
    })

    matched = attach_earnings_releases(events, releases).iloc[0]

    assert matched["release_accession"] == "new" and matched["cik"] == 42


def test_filing_prompt_wording_is_unchanged():
    # The cached MD&A scores were produced with exactly this sentence.
    prompt = LLM_TONE_PROMPT.format(source=MDNA_SOURCE.format(filing_type="10-Q"), text="x")

    assert prompt.startswith(
        "Below is an excerpt from the Management's Discussion and Analysis section of a company's 10-Q report.\n"
    )
