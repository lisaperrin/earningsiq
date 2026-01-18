import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.rag.vector_store import VectorStore
from earningsiq.config import settings
from earningsiq.utils.logger import log


def main():
    chunked_docs_path = settings.processed_data_dir / "chunked_documents.parquet"

    if not chunked_docs_path.exists():
        log.error("Chunked documents not found. Run 02_process_documents.py first")
        return

    documents_df = pd.read_parquet(chunked_docs_path)
    log.info(f"Loaded {len(documents_df)} document chunks")

    vector_store = VectorStore()
    vector_store.create_collection(reset=True)

    vector_store.add_documents(documents_df, batch_size=100)

    stats = vector_store.get_collection_stats()
    log.info(f"Vector index stats: {stats}")


if __name__ == "__main__":
    main()
