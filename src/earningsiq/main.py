from pathlib import Path
from typing import Optional
import pandas as pd
from .config import settings
from .data.edgar_downloader import EdgarDownloader, load_sample_tickers
from .data.preprocessor import DocumentProcessor
from .data.price_data import PriceDataFetcher
from .rag.vector_store import VectorStore, DocumentRetriever
from .rag.query_engine import RAGQueryEngine, QueryRouter
from .agents.multi_agent import MultiAgentOrchestrator
from .models.fine_tuning import LoRAFineTuner, FinancialDatasetPreparator
from .evaluation.metrics import ComprehensiveEvaluator
from .utils.logger import log


class EarningsIQ:
    def __init__(self, llm_model_name: Optional[str] = None):
        self.llm_model_name = llm_model_name or settings.llm_model
        self.vector_store = None
        self.rag_engine = None
        self.query_router = None
        self.multi_agent = None
        self.price_fetcher = PriceDataFetcher()

    def setup(self):
        log.info("Setting up EarningsIQ system")

        self.vector_store = VectorStore()
        self.vector_store.create_collection()

        self.rag_engine = RAGQueryEngine(self.vector_store, self.llm_model_name)
        self.query_router = QueryRouter(self.rag_engine)
        self.multi_agent = MultiAgentOrchestrator(self.rag_engine, self.price_fetcher)

        log.info("EarningsIQ system ready")

    def download_data(self, tickers: list[str] = None, use_sample: bool = True):
        if use_sample and tickers is None:
            tickers = load_sample_tickers()

        log.info(f"Downloading data for {len(tickers)} tickers")

        downloader = EdgarDownloader()
        results = downloader.download_bulk(tickers)

        filing_index = downloader.create_filing_index(tickers)

        log.info(f"Downloaded {len(filing_index)} filings")
        return filing_index

    def process_documents(self, filing_index: pd.DataFrame, use_parallel: bool = True):
        log.info("Processing documents")

        processor = DocumentProcessor()

        if use_parallel:
            chunked_docs = processor.process_batch_parallel(filing_index, n_partitions=4)
        else:
            chunked_docs = processor.process_batch(filing_index)

        log.info(f"Processed {len(chunked_docs)} document chunks")
        return chunked_docs

    def build_vector_index(self, documents_df: pd.DataFrame):
        log.info("Building vector index")

        if self.vector_store is None:
            self.vector_store = VectorStore()
            self.vector_store.create_collection(reset=True)

        self.vector_store.add_documents(documents_df, batch_size=100)

        stats = self.vector_store.get_collection_stats()
        log.info(f"Vector index built: {stats}")

    def query(self, question: str, ticker: Optional[str] = None, use_multi_agent: bool = False, year: Optional[int] = None):
        if self.rag_engine is None:
            self.setup()

        if use_multi_agent and self.multi_agent:
            result = self.multi_agent.execute_query(question, ticker, year=year)
        else:
            result = self.query_router.route_query(question, ticker, year=year)

        return result

    def fine_tune_model(self, use_finqa: bool = True, max_samples: int = 1000):
        log.info("Starting model fine-tuning")

        preparator = FinancialDatasetPreparator()

        if use_finqa:
            dataset = preparator.load_finqa_dataset()
            train_dataset = preparator.prepare_training_data(dataset, max_samples)
        else:
            log.error("Custom dataset not implemented")
            return

        finetuner = LoRAFineTuner()
        finetuner.setup_model()
        finetuner.train(train_dataset, num_epochs=3, batch_size=4)

        log.info("Fine-tuning completed")

    def build_price_correlation_dataset(self, filing_index: pd.DataFrame):
        log.info("Building price correlation dataset")

        price_reactions = self.price_fetcher.build_price_reaction_dataset(filing_index)

        log.info(f"Built price correlation dataset: {len(price_reactions)} entries")
        return price_reactions

    def evaluate(self, test_queries: list[dict]):
        log.info("Running evaluation")

        evaluator = ComprehensiveEvaluator()

        predictions = []
        references = []

        for query_data in test_queries:
            question = query_data.get("question")
            expected_answer = query_data.get("answer")
            ticker = query_data.get("ticker")

            result = self.query(question, ticker)
            predictions.append(result.get("answer", ""))
            references.append(expected_answer)

        eval_results = evaluator.run_full_evaluation(predictions, references)
        report = evaluator.generate_report(eval_results)

        log.info("Evaluation completed")
        print(report)

        return eval_results


def run_data_pipeline(sample_only: bool = True):
    system = EarningsIQ()

    if sample_only:
        tickers = load_sample_tickers()[:3]
    else:
        tickers = load_sample_tickers()

    filing_index = system.download_data(tickers, use_sample=True)

    chunked_docs = system.process_documents(filing_index, use_parallel=False)

    system.build_vector_index(chunked_docs)

    price_data = system.build_price_correlation_dataset(filing_index)

    log.info("Data pipeline completed")
    return system


def run_query_demo(system: EarningsIQ):
    system.setup()

    queries = [
        {"question": "What revenue guidance did AAPL provide?", "ticker": "AAPL"},
        {"question": "What are the main risk factors for NVDA?", "ticker": "NVDA"},
        {"question": "Why did TSLA stock move after earnings?", "ticker": "TSLA"},
    ]

    for query in queries:
        log.info(f"Query: {query['question']}")
        result = system.query(query["question"], query.get("ticker"))
        print(f"\nQ: {query['question']}")
        print(f"A: {result.get('answer', 'No answer available')}\n")
        print("-" * 80)


def run_multi_agent_demo(system: EarningsIQ):
    system.setup()

    result = system.multi_agent.execute_query(
        "Analyze NVDA's latest earnings and explain the stock price reaction",
        ticker="NVDA"
    )

    print("\nMulti-Agent Analysis:")
    print("=" * 80)
    print(f"Query: {result['query']}")
    print(f"Sub-tasks executed: {result['sub_tasks']}")
    print(f"\nFinal Synthesis:\n{result['final_synthesis']['synthesis']}")


if __name__ == "__main__":
    log.info("Starting EarningsIQ")

    system = run_data_pipeline(sample_only=True)

    run_query_demo(system)
