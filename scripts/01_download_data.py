"""
01_download_data.py - Download 10-K and 10-Q filings from SEC EDGAR

Usage:
    python scripts/01_download_data.py              # first 5 sample companies (about 2 GB)
    python scripts/01_download_data.py --all        # all 20 sample companies (about 8 GB)
    python scripts/01_download_data.py KO PFE       # specific tickers

Companies that are already on disk are skipped; pass --force to download
them again. The filing index always covers everything on disk.
"""

import sys
from pathlib import Path
import argparse

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.data.edgar_downloader import EdgarDownloader, load_sample_tickers
from earningsiq.utils.logger import log


def main():
    parser = argparse.ArgumentParser(description="Download SEC filings")
    parser.add_argument("tickers", nargs="*", help="Tickers to download (default: first 5 sample companies)")
    parser.add_argument("--all", action="store_true", help="Download all 20 sample companies")
    parser.add_argument("--force", action="store_true", help="Download again even if already on disk")
    args = parser.parse_args()

    if args.tickers:
        tickers = [ticker.upper() for ticker in args.tickers]
    elif args.all:
        tickers = load_sample_tickers()
    else:
        tickers = load_sample_tickers()[:5]

    downloader = EdgarDownloader()

    on_disk = set(downloader.downloaded_tickers())
    to_download = tickers if args.force else [t for t in tickers if t not in on_disk]
    skipped = [t for t in tickers if t not in to_download]
    if skipped:
        log.info(f"Already on disk, skipping: {', '.join(skipped)}")

    if to_download:
        log.info(f"Downloading filings for: {', '.join(to_download)}")
        results = downloader.download_bulk(to_download)
        for ticker, counts in results.items():
            log.info(f"{ticker}: {counts}")

    filing_index = downloader.create_filing_index(downloader.downloaded_tickers())
    log.info(f"Created index with {len(filing_index)} filings")

    print(filing_index.groupby(["ticker", "filing_type"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
