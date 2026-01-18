# EarningsIQ Usage Guide

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

**Note:** First run will download the LLM model from Hugging Face (size varies by model selected).

## Quick Start

### 1. Download SEC Filings

```bash
python scripts/01_download_data.py
```

Downloads 10-Q and 10-K filings for sample tickers (AAPL, MSFT, NVDA, etc.).

### 2. Process Documents

```bash
python scripts/02_process_documents.py
```

Extracts sections (MD&A, Risk Factors) and chunks documents for embedding.

### 3. Build Vector Index

```bash
python scripts/03_build_vector_index.py
```

Creates ChromaDB vector store with document embeddings using bge-small-en-v1.5.

### 4. Query the System

```bash
python scripts/04_query_system.py
```

Runs sample queries using RAG retrieval.

### 5. Analyze Price Correlations

```bash
python scripts/05_price_correlation.py
```

Calculates post-earnings price reactions using yfinance data.

### 6. Evaluate the System

```bash
python scripts/06_evaluate_system.py --quick   # Fast (2 test cases)
python scripts/06_evaluate_system.py           # Full (5 test cases)
```

Runs ROUGE and F1 evaluation metrics with baseline comparison.

### 7. Fine-Tune Model with LoRA

```bash
python scripts/07_finetune_model.py --quick    # Fast demo (50 samples, 1 epoch)
python scripts/07_finetune_model.py            # Full (500 samples, 3 epochs)
```

Demonstrates LoRA fine-tuning on FinQA dataset with before/after comparison.

## Programmatic Usage

```python
from earningsiq.main import EarningsIQ

system = EarningsIQ()
system.setup()

result = system.query(
    "What revenue guidance did AAPL provide?",
    ticker="AAPL"
)

print(result["answer"])
```

## Multi-Agent Query

```python
from earningsiq.main import EarningsIQ

system = EarningsIQ()
system.setup()

result = system.query(
    "Analyze NVDA's latest earnings and explain stock price reaction",
    ticker="NVDA",
    use_multi_agent=True
)

print(result["final_synthesis"])
```

## Fine-Tuning

```python
from earningsiq.models.fine_tuning import LoRAFineTuner, FinancialDatasetPreparator

preparator = FinancialDatasetPreparator()
dataset = preparator.load_finqa_dataset()
train_data = preparator.prepare_training_data(dataset, max_samples=1000)

finetuner = LoRAFineTuner(model_name="mistralai/Mistral-7B-Instruct-v0.3")
finetuner.setup_model()
finetuner.train(train_data, num_epochs=3, batch_size=4)
```

## Project Structure

```
earningsiq/
├── data/           # Data ingestion and preprocessing
├── rag/            # RAG system and vector store
├── models/         # Fine-tuning pipeline
├── agents/         # Multi-agent orchestration
├── evaluation/     # Metrics and evaluation
└── utils/          # Logging and utilities
```

## Configuration

Edit `src/earningsiq/config.py` or create `.env` file:

```
EDGAR_USER_AGENT=your_email@example.com
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
```

Supported models: TinyLlama-1.1B, Qwen-1.5B, Phi-3-mini-4k, Mistral-7B.
