from pathlib import Path
from typing import Optional, List, Dict
import pandas as pd
import chromadb
from chromadb.config import Settings as ChromaSettings
from sentence_transformers import SentenceTransformer
from ..config import settings
from ..utils.logger import log


class EmbeddingModel:
    def __init__(self, model_name: str = None):
        self.model_name = model_name or settings.embedding_model
        log.info(f"Loading embedding model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name)
        self.dimension = settings.embedding_dim

    def encode(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        return embeddings.tolist()

    def encode_single(self, text: str) -> List[float]:
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


class VectorStore:
    def __init__(self, collection_name: str = None, persist_directory: Path = None):
        self.collection_name = collection_name or settings.vector_collection_name
        self.persist_directory = persist_directory or settings.vector_db_path
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )

        self.embedding_model = EmbeddingModel()
        self.collection = None

    def create_collection(self, reset: bool = False):
        if reset:
            try:
                self.client.delete_collection(name=self.collection_name)
                log.info(f"Deleted existing collection: {self.collection_name}")
            except:
                pass

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Financial earnings reports embeddings"}
        )

        log.info(f"Created/loaded collection: {self.collection_name}")
        return self.collection

    def add_documents(
        self,
        documents_df: pd.DataFrame,
        batch_size: int = 100
    ):
        if self.collection is None:
            self.create_collection()

        total_docs = len(documents_df)
        log.info(f"Adding {total_docs} documents to vector store")

        for i in range(0, total_docs, batch_size):
            batch = documents_df.iloc[i:i + batch_size]

            texts = batch["text"].tolist()
            embeddings = self.embedding_model.encode(texts, batch_size=32)

            ids = [f"{row['ticker']}_{row['filing_type']}_{row['filing_date']}_{row['section']}_{row['chunk_id']}"
                   for _, row in batch.iterrows()]

            metadatas = [
                {
                    "ticker": str(row["ticker"]),
                    "filing_type": str(row["filing_type"]),
                    "filing_date": str(row["filing_date"]),
                    "section": str(row["section"]),
                    "chunk_id": int(row["chunk_id"]),
                }
                for _, row in batch.iterrows()
            ]

            self.collection.add(
                embeddings=embeddings,
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )

            log.info(f"Added batch {i // batch_size + 1}/{(total_docs + batch_size - 1) // batch_size}")

        log.info(f"Successfully added {total_docs} documents to vector store")

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter_dict: Optional[Dict] = None
    ) -> Dict:
        if self.collection is None:
            self.create_collection()

        query_embedding = self.embedding_model.encode_single(query_text)

        # ChromaDB requires $and operator for multiple filter conditions
        where_clause = None
        if filter_dict:
            if len(filter_dict) == 1:
                where_clause = filter_dict
            else:
                # Multiple conditions need $and wrapper
                where_clause = {
                    "$and": [{k: v} for k, v in filter_dict.items()]
                }

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause,
            include=["documents", "metadatas", "distances"]
        )

        return results

    def query_by_ticker(
        self,
        query_text: str,
        ticker: str,
        n_results: int = 5
    ) -> Dict:
        return self.query(
            query_text,
            n_results=n_results,
            filter_dict={"ticker": ticker}
        )

    def query_by_section(
        self,
        query_text: str,
        section: str,
        n_results: int = 5
    ) -> Dict:
        return self.query(
            query_text,
            n_results=n_results,
            filter_dict={"section": section}
        )

    def get_collection_stats(self) -> Dict:
        if self.collection is None:
            self.create_collection()

        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "total_documents": count,
            "persist_directory": str(self.persist_directory)
        }

    def delete_collection(self):
        try:
            self.client.delete_collection(name=self.collection_name)
            log.info(f"Deleted collection: {self.collection_name}")
        except Exception as e:
            log.error(f"Failed to delete collection: {e}")


class DocumentRetriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None,
        section: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[Dict]:
        filter_dict = {}
        if ticker:
            filter_dict["ticker"] = ticker
        if section:
            filter_dict["section"] = section

        results = self.vector_store.query(
            query,
            n_results=top_k * 3 if year else top_k,  # Fetch more if filtering by year
            filter_dict=filter_dict if filter_dict else None
        )

        # Post-filter by year if specified (ChromaDB doesn't support partial string match)
        if year and results["documents"]:
            filtered_docs = []
            filtered_meta = []
            filtered_dist = []
            for i in range(len(results["documents"][0])):
                filing_date = results["metadatas"][0][i].get("filing_date", "")
                if str(year) in filing_date:
                    filtered_docs.append(results["documents"][0][i])
                    filtered_meta.append(results["metadatas"][0][i])
                    filtered_dist.append(results["distances"][0][i])
            results = {
                "documents": [filtered_docs[:top_k]],
                "metadatas": [filtered_meta[:top_k]],
                "distances": [filtered_dist[:top_k]]
            }

        retrieved_docs = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                doc = {
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
                retrieved_docs.append(doc)

        return retrieved_docs

    def retrieve_context(
        self,
        query: str,
        top_k: int = 5,
        ticker: Optional[str] = None
    ) -> str:
        docs = self.retrieve(query, top_k, ticker)

        context_parts = []
        for i, doc in enumerate(docs):
            meta = doc["metadata"]
            context_parts.append(
                f"[Document {i+1}]\n"
                f"Ticker: {meta['ticker']}\n"
                f"Filing: {meta['filing_type']}\n"
                f"Date: {meta['filing_date']}\n"
                f"Section: {meta['section']}\n"
                f"Content: {doc['text']}\n"
            )

        return "\n---\n".join(context_parts)
