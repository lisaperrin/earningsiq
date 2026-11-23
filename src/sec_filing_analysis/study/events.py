"""When did each filing, and the earnings release that preceded it, become public?

Timing matters at the level of hours: most filings reach EDGAR after the
market has closed, so the first price that can react is the next session's.
"""

from datetime import time
from pathlib import Path
from typing import Optional
import re
import time as time_module

import pandas as pd
import requests

from ..config import settings
from ..data.preprocessor import resolve_filing_path
from ..utils.logger import log

MARKET_CLOSE = time(16, 0)  # US Eastern, the time zone of EDGAR timestamps

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/"
SEC_REQUEST_PAUSE = 0.15  # seconds; EDGAR allows 10 requests per second

# An 8-K with item 2.02 ("Results of Operations and Financial Condition") is
# the earnings release.
EARNINGS_ITEM = "2.02"

# Quarterly filings are about 90 days apart, so an earnings release more than
# 50 days before a filing belongs to the previous quarter.
MAX_RELEASE_LEAD = pd.Timedelta(days=50)
MAX_RELEASE_LAG = pd.Timedelta(days=1)


def read_submission_header(file_path: Path) -> dict:
    """Acceptance time (US Eastern) and fiscal period end of a filing, from its header."""
    try:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            header = f.read(3000)
    except OSError as e:
        log.warning(f"Cannot read {file_path}: {e}")
        return {"filing_accepted": None, "period_end": None}

    accepted = re.search(r"<ACCEPTANCE-DATETIME>(\d{14})", header)
    period = re.search(r"CONFORMED PERIOD OF REPORT:\s*(\d{8})", header)
    return {
        "filing_accepted": pd.to_datetime(accepted.group(1), format="%Y%m%d%H%M%S") if accepted else None,
        "period_end": pd.to_datetime(period.group(1), format="%Y%m%d") if period else None,
    }


def read_acceptance_time(file_path: Path) -> Optional[pd.Timestamp]:
    """Time EDGAR accepted a filing (US Eastern), from the submission header."""
    return read_submission_header(file_path)["filing_accepted"]


def event_session(accepted: pd.Timestamp, sessions: pd.DatetimeIndex) -> Optional[pd.Timestamp]:
    """First trading session whose closing price can reflect the news.

    News accepted before the close counts for that session; news accepted at
    or after the close, or on a non-trading day, counts for the next one.
    """
    if pd.isna(accepted):
        return None

    day = accepted.normalize()
    if accepted.time() >= MARKET_CLOSE:
        day += pd.Timedelta(days=1)

    position = sessions.searchsorted(day)
    return sessions[position] if position < len(sessions) else None


def build_filing_events(filing_index: pd.DataFrame) -> pd.DataFrame:
    """One row per filing with the time it became public."""
    rows = []
    for _, row in filing_index.iterrows():
        path = resolve_filing_path(row["file_path"])
        rows.append({
            "ticker": row["ticker"],
            "filing_type": row["filing_type"],
            "accession": path.parent.name,
            "filing_date": row["filing_date"],
            **read_submission_header(path),
        })

    events = pd.DataFrame(rows)
    missing = events["filing_accepted"].isna().sum()
    if missing:
        log.warning(f"{missing} filings have no acceptance time and are dropped")
    return events.dropna(subset=["filing_accepted"]).reset_index(drop=True)


def sec_get(url: str) -> requests.Response:
    """GET from SEC EDGAR with the required contact header and rate limit."""
    if "@" not in settings.edgar_user_agent:
        raise ValueError("Set EDGAR_USER_AGENT to your e-mail address (see .env.example).")

    response = requests.get(
        url,
        headers={"User-Agent": f"sec-filing-analysis {settings.edgar_user_agent}"},
        timeout=60,
    )
    response.raise_for_status()
    time_module.sleep(SEC_REQUEST_PAUSE)
    return response


def _sec_get(url: str) -> dict:
    return sec_get(url).json()


def load_cik_map() -> dict[str, int]:
    """Ticker -> SEC company number (CIK)."""
    return {entry["ticker"]: entry["cik_str"] for entry in _sec_get(SEC_TICKER_MAP_URL).values()}


def fetch_earnings_releases(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Earnings releases (8-K, item 2.02) of each company from EDGAR."""
    cik_by_ticker = load_cik_map()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    rows = []

    for ticker in tickers:
        cik = cik_by_ticker.get(ticker)
        if cik is None:
            log.warning(f"No CIK found for {ticker}")
            continue

        submissions = _sec_get(f"{SEC_SUBMISSIONS_URL}CIK{cik:010d}.json")
        blocks = [submissions["filings"]["recent"]]

        # Older filings are split over additional files
        for extra in submissions["filings"].get("files", []):
            if pd.Timestamp(extra["filingTo"]) >= start_ts and pd.Timestamp(extra["filingFrom"]) <= end_ts:
                blocks.append(_sec_get(SEC_SUBMISSIONS_URL + extra["name"]))

        count = 0
        for block in blocks:
            for form, items, accepted, accession in zip(
                block["form"], block["items"], block["acceptanceDateTime"], block["accessionNumber"]
            ):
                if form == "8-K" and EARNINGS_ITEM in items.split(","):
                    rows.append({"ticker": ticker, "cik": cik, "release_accession": accession, "release_accepted": accepted})
                    count += 1
        log.info(f"{ticker}: {count} earnings releases")

    releases = pd.DataFrame(rows)
    # The API reports UTC; submission headers and market hours are US Eastern.
    releases["release_accepted"] = (
        pd.to_datetime(releases["release_accepted"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.tz_localize(None)
    )
    releases = releases[releases["release_accepted"].between(start_ts, end_ts)]
    return releases.drop_duplicates().sort_values(["ticker", "release_accepted"]).reset_index(drop=True)


def attach_earnings_releases(events: pd.DataFrame, releases: pd.DataFrame) -> pd.DataFrame:
    """Add to each filing the earnings release it reports on, if one is found."""
    matched = []
    for _, event in events.iterrows():
        accepted = event["filing_accepted"]
        candidates = releases[
            (releases["ticker"] == event["ticker"])
            & (releases["release_accepted"] >= accepted - MAX_RELEASE_LEAD)
            & (releases["release_accepted"] <= accepted + MAX_RELEASE_LAG)
        ]
        matched.append(candidates.loc[candidates["release_accepted"].idxmax()] if len(candidates) else None)

    events = events.copy()
    events["release_accepted"] = [match["release_accepted"] if match is not None else pd.NaT for match in matched]
    for column in ("release_accession", "cik"):
        if column in releases:
            events[column] = [match[column] if match is not None else None for match in matched]
    return events
