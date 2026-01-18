# EarningsIQ: Financial Report Analysis with LLMs

LLM-powered system for analyzing SEC earnings reports and predicting market reactions using RAG, multi-agent orchestration, and LoRA fine-tuning.

## Quick Start

```bash
pip install -r requirements.txt
pip install -e .

python scripts/01_download_data.py
python scripts/02_process_documents.py
python scripts/03_build_vector_index.py
python scripts/04_query_system.py
python scripts/06_evaluate_system.py --quick
python scripts/07_finetune_model.py --quick
```

## Features

- **SEC EDGAR Integration**: Download and process 10-K/10-Q filings
- **RAG-Based Q&A**: Semantic search over financial reports with LLM-generated answers
- **Multi-Agent System**: Specialized agents for financial, risk, and price analysis
- **Price Correlation**: yfinance integration for earnings-price reaction analysis
- **LoRA Fine-tuning**: Optional model fine-tuning on FinQA dataset

## Architecture

```
SEC Filings → Dask Processing → ChromaDB Embeddings → RAG Engine → LLM → Insights
                                                            ↓
                                                    Multi-Agent System
                                                            ↓
                                          Price Analysis + Correlation
```

### Multi-Agent Query Flow

For complex queries, the system decomposes requests into specialized sub-tasks:

```
User Query
    ↓
QueryDecomposerAgent → Creates sub-tasks based on keywords
    ↓
Parallel Execution:
├── FinancialAnalystAgent → RAG query on MD&A section (top_k=7)
├── RiskAnalystAgent → RAG query on RISK_FACTORS section (top_k=5)
└── PriceAnalystAgent → Stock data via yfinance
    ↓
SynthesisAgent → Combines all responses into unified answer
```

**LLM Options:** TinyLlama (default), Qwen 1.5B, Phi-3 3.8B, or Mistral 7B

## Example Usage

```python
from earningsiq.main import EarningsIQ

system = EarningsIQ()
system.setup()

# Standard RAG query
result = system.query(
    "What revenue guidance did AAPL provide?",
    ticker="AAPL"
)
print(result["answer"])

# Multi-agent query for complex analysis
result = system.query(
    "Analyze NVDA's latest earnings and explain the stock price reaction",
    ticker="NVDA",
    use_multi_agent=True
)
print(result["final_synthesis"]["synthesis"])
```

## Evaluation Metrics

The system includes comprehensive evaluation across multiple dimensions:

| Metric | Description | Target |
|--------|-------------|--------|
| **ROUGE-L** | Text similarity to gold standard answers | > 0.40 |
| **ROUGE-1** | Unigram overlap with references | > 0.45 |
| **ROUGE-2** | Bigram overlap with references | > 0.25 |
| **Entity F1** | Financial entity extraction accuracy | > 0.70 |
| **Precision@5** | Relevant docs in top 5 results | > 0.60 |
| **Recall@5** | Coverage of relevant documents | > 0.50 |
| **MRR** | Mean Reciprocal Rank for retrieval | > 0.65 |
| **Sentiment-Price Correlation** | Pearson correlation between text sentiment and price movement | Significant (p < 0.05) |

### Running Evaluation

```python
from earningsiq.main import EarningsIQ

system = EarningsIQ()
system.setup()

test_queries = [
    {"question": "What was Apple's revenue guidance?", "answer": "...", "ticker": "AAPL"},
    {"question": "What risks does NVIDIA face?", "answer": "...", "ticker": "NVDA"},
]

eval_results = system.evaluate(test_queries)
```

### Baseline Comparison

The system compares LLM-powered retrieval against keyword-based baselines:
- **Keyword Search**: TF-IDF style word overlap scoring
- **LLM System**: Semantic embedding similarity + LLM answer generation

Improvements tracked: Precision gain, Recall gain, Time reduction

## Project Structure

```
earningsiq/
├── data/           # EDGAR downloader, preprocessor, price data
├── rag/            # Vector store (ChromaDB), query engine
├── models/         # PEFT/LoRA fine-tuning
├── agents/         # Multi-agent orchestration
├── evaluation/     # Metrics (ROUGE, F1, correlation)
└── utils/          # Logging, config
```

## Documentation

- [USAGE.md](USAGE.md) - Complete usage guide
- [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md) - Architecture details

## Technology Stack

| Category | Technologies |
|----------|--------------|
| **Big Data** | Polars, ChromaDB |
| **LLM** | Hugging Face Transformers, sentence-transformers |
| **Data Sources** | sec-edgar-downloader, yfinance, FinQA |
| **Evaluation** | rouge-score, scikit-learn, scipy |


**Model Selection by RAM:**
| RAM | Recommended Model |
|-----|-------------------|
| 4-8GB | TinyLlama-1.1B |
| 8-16GB | Qwen-1.5B |
| 16GB | Phi-3-mini-4k |
| 16GB+ w/ GPU | Mistral-7B |


