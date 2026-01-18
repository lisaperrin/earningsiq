import sys
from pathlib import Path
import time
import argparse
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.main import EarningsIQ
from earningsiq.evaluation.metrics import (
    ComprehensiveEvaluator,
    RetrievalEvaluator,
    BaselineComparator,
    PerformanceMetrics,
)
from earningsiq.utils.logger import log


# Test queries with expected reference answers (ground truth)
TEST_CASES = [
    {
        "query": "What were Apple's key financial results and how did net sales perform?",
        "ticker": "AAPL",
        "year": 2024,
        "reference": "Apple reported net sales with iPhone increasing, Services growing, and Mac showing growth. Products gross margin improved. The company saw varying performance across segments with some declines in iPad and Wearables.",
        "expected_entities": ["revenue", "margin", "profit"],
    },
    {
        "query": "What are NVIDIA's main risk factors?",
        "ticker": "NVDA",
        "year": 2024,
        "reference": "NVIDIA faces supply chain risks including long manufacturing lead times and uncertain component availability. Competition from established players poses challenges. Regulatory risks from complex global laws and export controls to certain countries. Cybersecurity vulnerabilities and data privacy concerns. Geopolitical tensions affecting operations.",
        "expected_entities": ["risk"],
    },
    {
        "query": "How did Microsoft's cloud business perform?",
        "ticker": "MSFT",
        "year": 2024,
        "reference": "Microsoft's Intelligent Cloud segment showed revenue growth driven by Azure and other cloud services. The company continues to invest in AI capabilities and cloud infrastructure.",
        "expected_entities": ["revenue", "growth"],
    },
    {
        "query": "What were Tesla's key operational highlights?",
        "ticker": "TSLA",
        "year": 2024,
        "reference": "Tesla reported on vehicle deliveries, production capacity, and energy business performance. The company discussed manufacturing efficiency and expansion plans.",
        "expected_entities": ["revenue", "margin"],
    },
    {
        "query": "What risks does Google face from competition?",
        "ticker": "GOOGL",
        "year": 2024,
        "reference": "Google faces competition in search, advertising, cloud computing, and AI. The company must maintain innovation while addressing regulatory scrutiny and antitrust concerns.",
        "expected_entities": ["risk"],
    },
]


def run_evaluation(system: EarningsIQ, num_cases: int = None):
    """Run comprehensive evaluation of the RAG system.

    Args:
        system: The EarningsIQ system to evaluate
        num_cases: Number of test cases to run (None = all)
    """
    print("\n" + "=" * 80)
    print("RAG SYSTEM EVALUATION")
    print("=" * 80)

    evaluator = ComprehensiveEvaluator()
    retrieval_eval = RetrievalEvaluator()
    baseline_comp = BaselineComparator()
    perf_metrics = PerformanceMetrics()

    predictions = []
    references = []
    query_times = []
    retrieved_doc_counts = []

    test_cases = TEST_CASES[:num_cases] if num_cases else TEST_CASES
    print(f"\nRunning {len(test_cases)} test queries...\n")

    for i, test_case in enumerate(test_cases):
        query = test_case["query"]
        ticker = test_case["ticker"]
        year = test_case["year"]
        reference = test_case["reference"]

        print(f"[{i+1}/{len(test_cases)}] {ticker}: {query[:50]}...")

        # Time the query
        start_time = time.time()
        result = system.query(query, ticker=ticker, year=year)
        query_time = time.time() - start_time

        answer = result.get("answer", "")
        sources = result.get("sources", [])

        predictions.append(answer)
        references.append(reference)
        query_times.append(query_time)
        retrieved_doc_counts.append(len(sources))

        perf_metrics.record_query_time(f"query_{i}", query_time, "llm_system")

        print(f"    Time: {query_time:.2f}s | Docs: {len(sources)} | Answer length: {len(answer)} chars")

    # Calculate metrics
    print("\n" + "-" * 80)
    print("EVALUATION RESULTS")
    print("-" * 80)

    # Run comprehensive evaluation
    eval_results = evaluator.run_full_evaluation(predictions, references)

    # Print ROUGE scores
    rouge_result = eval_results.get("rouge")
    if rouge_result and rouge_result.details:
        print("\n[ROUGE Scores - Answer Quality]")
        print(f"    ROUGE-1: {rouge_result.details.get('rouge1', 0):.4f}")
        print(f"    ROUGE-2: {rouge_result.details.get('rouge2', 0):.4f}")
        print(f"    ROUGE-L: {rouge_result.details.get('rougeL', 0):.4f}")

    # Print entity extraction scores
    entity_result = eval_results.get("entity_extraction")
    if entity_result and entity_result.details:
        print("\n[Entity Extraction - Financial Terms]")
        print(f"    Precision: {entity_result.details.get('precision', 0):.4f}")
        print(f"    Recall:    {entity_result.details.get('recall', 0):.4f}")
        print(f"    F1 Score:  {entity_result.details.get('f1', 0):.4f}")

    # Performance metrics
    print("\n[Performance Metrics]")
    print(f"    Avg Query Time:    {sum(query_times) / len(query_times):.2f}s")
    print(f"    Min Query Time:    {min(query_times):.2f}s")
    print(f"    Max Query Time:    {max(query_times):.2f}s")
    print(f"    Avg Docs Retrieved: {sum(retrieved_doc_counts) / len(retrieved_doc_counts):.1f}")

    # Retrieval quality (based on whether answers contain expected info)
    print("\n[Retrieval Quality]")
    answers_with_content = sum(1 for p in predictions if len(p) > 100 and "context" not in p.lower()[:50])
    print(f"    Queries with substantive answers: {answers_with_content}/{len(predictions)}")

    # Summary metrics
    print("\n" + "=" * 80)
    print("SUMMARY METRICS")
    print("=" * 80)

    summary = {
        "rouge_l": rouge_result.details.get("rougeL", 0) if rouge_result else 0,
        "entity_f1": entity_result.details.get("f1", 0) if entity_result else 0,
        "avg_query_time_s": sum(query_times) / len(query_times),
        "avg_docs_retrieved": sum(retrieved_doc_counts) / len(retrieved_doc_counts),
        "substantive_answer_rate": answers_with_content / len(predictions),
    }

    print(f"\n    ROUGE-L Score:           {summary['rouge_l']:.4f}")
    print(f"    Entity Extraction F1:    {summary['entity_f1']:.4f}")
    print(f"    Avg Query Time:          {summary['avg_query_time_s']:.2f}s")
    print(f"    Avg Docs Retrieved:      {summary['avg_docs_retrieved']:.1f}")
    print(f"    Substantive Answer Rate: {summary['substantive_answer_rate']:.1%}")

    # Save results
    results_df = pd.DataFrame({
        "query": [tc["query"] for tc in test_cases],
        "ticker": [tc["ticker"] for tc in test_cases],
        "prediction": predictions,
        "reference": references,
        "query_time_s": query_times,
        "docs_retrieved": retrieved_doc_counts,
    })

    output_path = Path(__file__).parent.parent / "data" / "evaluation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nDetailed results saved to: {output_path}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Evaluate the EarningsIQ RAG system")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation with only 2 test cases (much faster)"
    )
    parser.add_argument(
        "-n", "--num-cases",
        type=int,
        default=None,
        help="Number of test cases to run (default: all 5)"
    )
    args = parser.parse_args()

    num_cases = 2 if args.quick else args.num_cases

    log.info("Initializing EarningsIQ system for evaluation...")
    system = EarningsIQ()
    system.setup()

    stats = system.vector_store.get_collection_stats()
    log.info(f"Vector store ready: {stats}")

    summary = run_evaluation(system, num_cases=num_cases)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
