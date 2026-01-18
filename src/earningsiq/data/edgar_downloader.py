from pathlib import Path
from typing import Optional
from dataclasses import dataclass
from sec_edgar_downloader import Downloader
import pandas as pd
from ..config import settings
from ..utils.logger import log


@dataclass
class TickerInfo:
    ticker: str
    cik: Optional[str] = None
    company_name: Optional[str] = None


class EdgarDownloader:
    def __init__(self, user_agent: str = None, output_dir: Path = None):
        self.user_agent = user_agent or settings.edgar_user_agent
        self.output_dir = output_dir or settings.raw_data_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.downloader = Downloader(company_name="EarningsIQ", email_address=self.user_agent, download_folder=str(self.output_dir))

    def download_filings(
        self,
        ticker: str,
        filing_type: str = "10-Q",
        after: str = None,
        before: str = None,
        limit: int = None,
    ) -> int:
        try:
            after_date = after or f"{settings.edgar_start_year}-01-01"
            before_date = before or f"{settings.edgar_end_year}-12-31"

            log.info(f"Downloading {filing_type} for {ticker} from {after_date} to {before_date}")

            count = self.downloader.get(
                filing_type,
                ticker,
                after=after_date,
                before=before_date,
                limit=limit,
                download_details=True
            )

            log.info(f"Downloaded {count} {filing_type} filings for {ticker}")
            return count

        except Exception as e:
            log.error(f"Failed to download {filing_type} for {ticker}: {e}")
            return 0

    def download_bulk(
        self,
        tickers: list[str],
        filing_types: list[str] = None,
        after: str = None,
        before: str = None,
    ) -> dict[str, int]:
        filing_types = filing_types or settings.edgar_filing_types
        results = {}

        for ticker in tickers:
            ticker_results = {}
            for filing_type in filing_types:
                count = self.download_filings(ticker, filing_type, after, before)
                ticker_results[filing_type] = count
            results[ticker] = ticker_results

        return results

    def get_filing_paths(self, ticker: str, filing_type: str) -> list[Path]:
        filing_dir = self.output_dir / "sec-edgar-filings" / ticker / filing_type
        if not filing_dir.exists():
            return []

        filing_paths = []
        for submission_dir in filing_dir.iterdir():
            if submission_dir.is_dir():
                for file in submission_dir.glob("*.txt"):
                    if "full-submission" in file.name:
                        filing_paths.append(file)

        return sorted(filing_paths)

    def _extract_filing_date(self, file_path: Path) -> Optional[str]:
        try:
            content = file_path.read_text(encoding="utf-8", errors="ignore")
            for line in content.split("\n")[:50]:
                if "FILED AS OF DATE:" in line:
                    date_str = line.split("FILED AS OF DATE:")[1].strip()
                    return date_str[:8]
        except Exception as e:
            log.warning(f"Failed to extract filing date from {file_path}: {e}")
        return None

    def create_filing_index(self, tickers: list[str]) -> pd.DataFrame:
        records = []

        for ticker in tickers:
            for filing_type in settings.edgar_filing_types:
                paths = self.get_filing_paths(ticker, filing_type)
                for path in paths:
                    filing_date = self._extract_filing_date(path)
                    records.append({
                        "ticker": ticker,
                        "filing_type": filing_type,
                        "filing_date": filing_date,
                        "file_path": str(path)
                    })

        df = pd.DataFrame(records)
        if not df.empty:
            df["filing_date"] = pd.to_datetime(df["filing_date"], errors="coerce")
            df = df.sort_values(["ticker", "filing_date"]).reset_index(drop=True)

        index_path = settings.processed_data_dir / "filing_index.parquet"
        settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
        df.to_parquet(index_path, index=False)

        log.info(f"Created filing index with {len(df)} entries at {index_path}")
        return df


def load_sp500_tickers() -> list[str]:
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        sp500_table = tables[0]
        tickers = sp500_table["Symbol"].str.replace(".", "-").tolist()
        log.info(f"Loaded {len(tickers)} S&P 500 tickers")
        return tickers
    except Exception as e:
        log.error(f"Failed to load S&P 500 tickers: {e}")
        return []


def load_sample_tickers() -> list[str]:
    return ["AAPL", "MSFT", "GOOGL", "NVDA", "TSLA", "META", "AMZN", "JPM", "V", "WMT"]
