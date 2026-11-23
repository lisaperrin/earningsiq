"""Earnings surprises from the numbers companies report to the SEC.

Without analyst forecasts, the expectation for a quarter is built from the
company's own history: the same quarter one year earlier plus the company's
usual year-over-year change (a "seasonal random walk with drift"). The
surprise is how far the actual change departs from the usual one, scaled by
how much that change varies:

    change   = value - value one year earlier
    surprise = (change - mean of the previous 8 changes) / std of the previous 8 changes

A company that grows steadily therefore has no surprise; +2 means the quarter
beat its usual year-over-year change by two of its usual swings.
"""

import numpy as np
import pandas as pd

from ..utils.logger import log
from .events import _sec_get, load_cik_map

COMPANY_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

# XBRL tags, in order of preference. Companies switch tags over the years.
INCOME_TAGS = ["NetIncomeLoss", "ProfitLoss"]
REVENUE_TAGS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "SalesRevenueNet",
    "RevenuesNetOfInterestExpense",
]

QUARTER_DAYS = (80, 100)
YEAR_TOLERANCE_DAYS = 10    # fiscal quarters end on slightly different dates each year
HISTORY_QUARTERS = 8
MIN_HISTORY_QUARTERS = 4


def reported_periods(us_gaap: dict, tags: list[str]) -> pd.DataFrame:
    """All reported values of a measure as (start, end, value), one row per period.

    A period is often reported again in later filings (as the prior-year
    comparison); the most recently filed value is kept. Where several tags
    cover a period, the first tag in `tags` wins.
    """
    frames = []
    for priority, tag in enumerate(tags):
        units = us_gaap.get(tag, {}).get("units", {}).get("USD")
        if not units:
            continue
        frame = pd.DataFrame(units)
        if "start" not in frame:
            continue
        frame = frame.dropna(subset=["start"])
        frames.append(pd.DataFrame({
            "start": pd.to_datetime(frame["start"]),
            "end": pd.to_datetime(frame["end"]),
            "value": frame["val"].astype(float),
            "filed": pd.to_datetime(frame["filed"]),
            "priority": priority,
        }))

    if not frames:
        return pd.DataFrame(columns=["start", "end", "value"])

    periods = pd.concat(frames).sort_values(["priority", "filed"], ascending=[True, False])
    return periods.drop_duplicates(["start", "end"])[["start", "end", "value"]].reset_index(drop=True)


def quarterly_values(periods: pd.DataFrame) -> pd.Series:
    """Value of each single quarter, indexed by the quarter's end date.

    Three-month figures are used where reported. The fourth quarter is only
    reported as part of the full year, so it is derived as the full-year value
    minus the nine-month value with the same start date.
    """
    periods = periods.assign(days=(periods["end"] - periods["start"]).dt.days)
    is_quarter = periods["days"].between(*QUARTER_DAYS)
    quarters = periods[is_quarter].set_index("end")["value"]
    quarters = quarters[~quarters.index.duplicated()]

    derived = {}
    longer = periods[~is_quarter]
    for start, group in longer.groupby("start"):
        group = group.sort_values("end")
        for (_, earlier), (_, later) in zip(group.iloc[:-1].iterrows(), group.iloc[1:].iterrows()):
            gap = (later["end"] - earlier["end"]).days
            if QUARTER_DAYS[0] <= gap <= QUARTER_DAYS[1] and later["end"] not in quarters.index:
                derived[later["end"]] = later["value"] - earlier["value"]

    return pd.concat([quarters, pd.Series(derived, dtype=float)]).sort_index()


def seasonal_surprise(quarters: pd.Series) -> pd.DataFrame:
    """Year-over-year change of each quarter and its standardized surprise."""
    ends = quarters.index
    changes = pd.Series(np.nan, index=ends)
    growth = pd.Series(np.nan, index=ends)

    for end in ends:
        distance = np.abs((ends - (end - pd.Timedelta(days=365))).days)
        if len(distance) and distance.min() <= YEAR_TOLERANCE_DAYS:
            year_ago = quarters.iloc[int(distance.argmin())]
            changes[end] = quarters[end] - year_ago
            growth[end] = (quarters[end] - year_ago) / abs(year_ago) if year_ago else np.nan

    # The company's own past changes, excluding the current quarter
    past = changes.shift(1).rolling(HISTORY_QUARTERS, min_periods=MIN_HISTORY_QUARTERS)
    surprise = (changes - past.mean()) / past.std().replace(0, np.nan)
    return pd.DataFrame({"value": quarters, "growth": growth, "surprise": surprise})


def fetch_surprises(tickers: list[str]) -> pd.DataFrame:
    """Quarterly net income and revenue surprises for each company (rows: ticker, period_end)."""
    cik_by_ticker = load_cik_map()
    frames = []

    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        if cik is None:
            log.warning(f"No CIK found for {ticker}")
            continue

        us_gaap = _sec_get(COMPANY_FACTS_URL.format(cik=cik))["facts"]["us-gaap"]
        income = seasonal_surprise(quarterly_values(reported_periods(us_gaap, INCOME_TAGS)))
        revenue = seasonal_surprise(quarterly_values(reported_periods(us_gaap, REVENUE_TAGS)))

        frame = income.add_prefix("income_").join(revenue.add_prefix("revenue_"), how="outer")
        frame.index.name = "period_end"
        frame = frame.reset_index()
        frame.insert(0, "ticker", ticker)
        frames.append(frame)
        log.info(f"{ticker}: {income['surprise'].notna().sum()} income and {revenue['surprise'].notna().sum()} revenue surprises")

    return pd.concat(frames, ignore_index=True)


def attach_surprises(events: pd.DataFrame, surprises: pd.DataFrame, tolerance_days: int = 5) -> pd.DataFrame:
    """Add to each event the surprise of the fiscal period it reports on."""
    events = events.copy()
    columns = [column for column in surprises.columns if column not in ("ticker", "period_end")]
    matched = {column: [] for column in columns}

    by_ticker = {ticker: group.set_index("period_end") for ticker, group in surprises.groupby("ticker")}
    for ticker, period_end in zip(events["ticker"], events["period_end"]):
        company = by_ticker.get(ticker)
        row = None
        if company is not None and pd.notna(period_end):
            distance = np.abs((company.index - period_end).days)
            if distance.min() <= tolerance_days:
                row = company.iloc[int(distance.argmin())]
        for column in columns:
            matched[column].append(row[column] if row is not None else np.nan)

    for column in columns:
        events[column] = matched[column]
    return events


def fetch_analyst_surprises(tickers: list[str], limit: int = 60) -> pd.DataFrame:
    """Reported EPS against the analyst consensus before the release, from Yahoo Finance.

    This is the surprise relative to what the market expected, which the
    filings themselves cannot provide. Rows: ticker, earnings_date,
    eps_estimate, eps_reported, analyst_surprise (in percent of the estimate).
    """
    import yfinance as yf

    frames = []
    for ticker in tickers:
        try:
            dates = yf.Ticker(ticker).get_earnings_dates(limit=limit)
        except Exception as e:
            log.warning(f"No analyst estimates for {ticker}: {e}")
            continue
        if dates is None or dates.empty:
            log.warning(f"No analyst estimates for {ticker}")
            continue

        dates = dates.dropna(subset=["EPS Estimate", "Reported EPS"])
        frames.append(pd.DataFrame({
            "ticker": ticker,
            "earnings_date": pd.DatetimeIndex(dates.index).tz_localize(None).normalize(),
            "eps_estimate": dates["EPS Estimate"].to_numpy(),
            "eps_reported": dates["Reported EPS"].to_numpy(),
            "analyst_surprise": dates["Surprise(%)"].to_numpy(),
        }))
        log.info(f"{ticker}: {len(dates)} quarters with analyst estimates")

    return pd.concat(frames, ignore_index=True)


def attach_analyst_surprises(events: pd.DataFrame, analyst: pd.DataFrame, tolerance_days: int = 3) -> pd.DataFrame:
    """Add the analyst surprise of the earnings release each event belongs to."""
    events = events.copy()
    columns = ["eps_estimate", "eps_reported", "analyst_surprise"]
    matched = {column: [] for column in columns}

    by_ticker = {ticker: group.set_index("earnings_date") for ticker, group in analyst.groupby("ticker")}
    for ticker, released in zip(events["ticker"], events["release_accepted"]):
        company = by_ticker.get(ticker)
        row = None
        if company is not None and pd.notna(released):
            distance = np.abs((company.index - released.normalize()).days)
            if distance.min() <= tolerance_days:
                row = company.iloc[int(distance.argmin())]
        for column in columns:
            matched[column].append(row[column] if row is not None else np.nan)

    for column in columns:
        events[column] = matched[column]
    return events
