# EarningsIQ

Question answering over SEC 10-K and 10-Q filings with retrieval-augmented generation (RAG) and small language models that run locally. Capstone project for a university Big Data course (January 2026).

## What it does

1. Downloads 10-K and 10-Q filings from SEC EDGAR (AAPL, MSFT, GOOGL, NVDA, TSLA, 2019–2024: 120 filings).
2. Extracts the main sections (Business, Risk Factors, MD&A, Financial Statements) and splits them into about 11,000 chunks.
3. Embeds the chunks with `bge-small-en-v1.5` into ChromaDB.
4. Answers questions with a local LLM (Qwen2.5-1.5B by default) from the retrieved chunks, filtered by ticker, section and year. Keyword rules route a question to the matching section.
5. Reports stock price moves around each filing date (yfinance).
6. Fine-tunes a small model with LoRA on a financial Q&A dataset.

## Quick start

Python 3.9+; a CUDA GPU is strongly recommended.

```bash
pip install -e .
cp .env.example .env        # set EDGAR_USER_AGENT to your e-mail address

python scripts/01_download_data.py        # about 2 GB
python scripts/02_process_documents.py
python scripts/03_build_vector_index.py
python scripts/04_query_system.py
python scripts/05_price_correlation.py
python scripts/06_evaluate_system.py --quick
python scripts/07_finetune_model.py --quick
```

```python
from earningsiq.main import EarningsIQ

system = EarningsIQ()
system.setup()

result = system.query("What are NVIDIA's main risk factors?", ticker="NVDA", year=2024)
print(result["answer"])
```

## Results and limits

- Evaluation is a smoke test: five hand-written questions, no labelled retrieval data. On an RTX 3060 Ti a question takes about a minute.
- The LoRA demo (TinyLlama, 50 samples, 1 epoch) lowered ROUGE-L from 0.36 to 0.28 on 10 test examples; too small a run to conclude anything.
- The project does not predict market reactions. It measures price changes around filing dates only.
- The "multi-agent" mode is rule-based: keywords choose which section queries run, and their answers are joined without a further LLM call.

The original course proposal is in [docs/PROPOSAL.md](docs/PROPOSAL.md).

## License

[MIT](LICENSE)
