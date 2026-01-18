import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.data.edgar_downloader import EdgarDownloader, load_sample_tickers
from earningsiq.utils.logger import log


def main():
    tickers = load_sample_tickers()[:5]

    log.info(f"Downloading filings for: {', '.join(tickers)}")

    downloader = EdgarDownloader()
    results = downloader.download_bulk(tickers)

    for ticker, counts in results.items():
        log.info(f"{ticker}: {counts}")

    filing_index = downloader.create_filing_index(tickers)
    log.info(f"Created index with {len(filing_index)} filings")

    print(filing_index.head())


if __name__ == "__main__":
    main()
