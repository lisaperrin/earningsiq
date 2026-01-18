import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.data.price_data import PriceDataFetcher
from earningsiq.config import settings
from earningsiq.utils.logger import log


def main():
    filing_index_path = settings.processed_data_dir / "filing_index.parquet"

    if not filing_index_path.exists():
        log.error("Filing index not found. Run 01_download_data.py first")
        return

    filing_index = pd.read_parquet(filing_index_path)
    filing_index = filing_index.head(10)

    log.info(f"Analyzing price reactions for {len(filing_index)} filings")

    price_fetcher = PriceDataFetcher()
    price_reactions = price_fetcher.build_price_reaction_dataset(filing_index)

    log.info(f"Built price reaction dataset: {len(price_reactions)} entries")

    if not price_reactions.empty:
        print("\nPrice Reactions Summary:")
        print(price_reactions.describe())

        print("\nTop 5 Positive Reactions:")
        print(price_reactions.nlargest(5, 'price_change_pct')[['ticker', 'earnings_date', 'price_change_pct']])

        print("\nTop 5 Negative Reactions:")
        print(price_reactions.nsmallest(5, 'price_change_pct')[['ticker', 'earnings_date', 'price_change_pct']])


if __name__ == "__main__":
    main()
