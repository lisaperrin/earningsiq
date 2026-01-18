import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.data.preprocessor import DocumentProcessor
from earningsiq.config import settings
from earningsiq.utils.logger import log


def main():
    filing_index_path = settings.processed_data_dir / "filing_index.parquet"

    if not filing_index_path.exists():
        log.error("Filing index not found. Run 01_download_data.py first")
        return

    filing_index = pd.read_parquet(filing_index_path)
    log.info(f"Loaded {len(filing_index)} filings")

    processor = DocumentProcessor()
    chunked_docs = processor.process_batch(filing_index)

    log.info(f"Processed {len(chunked_docs)} chunks")
    print(chunked_docs.head())


if __name__ == "__main__":
    main()
