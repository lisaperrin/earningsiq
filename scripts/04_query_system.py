import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.main import EarningsIQ
from earningsiq.utils.logger import log


def run_standard_queries(system: EarningsIQ):
    """Run standard RAG queries (retrieval + LLM generation)."""
    print("\n" + "=" * 80)
    print("STANDARD RAG QUERIES")
    print("=" * 80)

    queries = [
        {"question": "What were Apple's key financial results and how did net sales perform?", "ticker": "AAPL", "year": 2024},
        {"question": "What are NVIDIA's main risk factors?", "ticker": "NVDA", "year": 2024},
    ]

    for q in queries:
        log.info(f"Query: {q['question']} (Ticker: {q['ticker']}, Year: {q.get('year')})")

        result = system.query(q["question"], ticker=q["ticker"], year=q.get("year"))

        print(f"\n{'-'*80}")
        print(f"Q: {q['question']} [{q['ticker']}]")
        print(f"{'-'*80}")
        print(f"A: {result.get('answer', 'No answer available')}")

        if result.get("sources"):
            print(f"\nSources: {len(result['sources'])} documents retrieved")


def run_multi_agent_queries(system: EarningsIQ):
    """Run multi-agent queries for complex analysis."""
    print("\n" + "=" * 80)
    print("MULTI-AGENT QUERIES")
    print("=" * 80)

    queries = [
        {
            "question": "What drove NVIDIA's revenue growth and what are the key business segments?",
            "ticker": "NVDA",
            "year": 2024,
        },
    ]

    for q in queries:
        log.info(f"Multi-agent query: {q['question']} (Ticker: {q['ticker']}, Year: {q.get('year')})")

        result = system.query(q["question"], ticker=q["ticker"], use_multi_agent=True, year=q.get("year"))

        print(f"\n{'-'*80}")
        print(f"Q: {q['question']} [{q['ticker']}]")
        print(f"{'-'*80}")

        print(f"\nAgents executed: {', '.join(result.get('sub_tasks', []))}")

        for response in result.get("agent_responses", []):
            agent_name = response.agent_role.value
            if response.success and response.result:
                if hasattr(response.result, "get"):
                    answer = response.result.get("answer", response.result.get("fundamentals", ""))
                else:
                    answer = str(response.result)
                print(f"\n[{agent_name}]")
                print(f"{str(answer)[:300]}..." if len(str(answer)) > 300 else answer)

        synthesis = result.get("final_synthesis", {})
        if synthesis:
            print(f"\n[SYNTHESIS]")
            print(synthesis.get("synthesis", "No synthesis available"))


def main():
    log.info("Initializing EarningsIQ system...")
    system = EarningsIQ()
    system.setup()

    stats = system.vector_store.get_collection_stats()
    log.info(f"Vector store ready: {stats}")

    run_standard_queries(system)

    run_multi_agent_queries(system)

    print("\n" + "=" * 80)
    print("Query demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
