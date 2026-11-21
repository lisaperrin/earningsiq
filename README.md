# EarningsIQ

Question answering over SEC 10-K and 10-Q filings with retrieval-augmented generation (RAG) and small language models that run locally.

This was my capstone project for a university Big Data course (January 2026). It is a working small-scale pipeline, not a product: it covers five companies, runs on a single consumer GPU, and its evaluation is a smoke test rather than a benchmark. The [original proposal](docs/PROPOSAL.md) aimed much higher; the [Limitations](#limitations) section says plainly what was and was not built.

## What it does

1. **Download** 10-K and 10-Q filings from SEC EDGAR.
2. **Parse** each filing into its main sections (Business, Risk Factors, MD&A, Financial Statements) and split them into chunks of at most 512 tokens.
3. **Embed** the chunks with `BAAI/bge-small-en-v1.5` and store them in ChromaDB.
4. **Answer questions**: retrieve the closest chunks, optionally filtered by ticker, section and filing year, and let a local LLM answer from them.
5. **Route questions** to the matching filing section with keyword rules, and combine several section-specific answers for broader questions.
6. **Measure price moves** around each filing date with yfinance.
7. **Fine-tune** a small model with LoRA on a financial Q&A dataset and compare ROUGE before and after.

```
SEC EDGAR ──> parse sections ──> chunk (≤512 tokens) ──> embed ──> ChromaDB
                                                                      │
question ──> keyword router ──> retrieve top-k chunks ──> local LLM ──> answer
```

## Corpus

| | |
|---|---|
| Companies | AAPL, MSFT, GOOGL, NVDA, TSLA |
| Period | 2019–2024 |
| Filings | 120 (30 10-K, 90 10-Q), about 2 GB raw |
| Chunks | 10,987 (Financial Statements 5,350, MD&A 2,726, Risk Factors 2,393, Business 518) |

## Quick start

Requires Python 3.9+. A CUDA GPU is strongly recommended; the numbers below were measured on an RTX 3060 Ti (8 GB) with 16 GB RAM.

```bash
pip install -e .
cp .env.example .env        # then set EDGAR_USER_AGENT to your e-mail address

python scripts/01_download_data.py        # download filings (about 2 GB)
python scripts/02_process_documents.py    # extract sections, chunk
python scripts/03_build_vector_index.py   # embed into ChromaDB
python scripts/04_query_system.py         # sample questions
python scripts/05_price_correlation.py    # price moves around filing dates
python scripts/06_evaluate_system.py --quick   # 2 test questions (omit --quick for 5)
python scripts/07_finetune_model.py --quick    # LoRA demo: 50 samples, 1 epoch
```

SEC EDGAR asks automated clients to identify themselves, so the download fails until `EDGAR_USER_AGENT` is set. The first query downloads the language model from Hugging Face.

### From Python

```python
from earningsiq.main import EarningsIQ

system = EarningsIQ()
system.setup()

result = system.query("What are NVIDIA's main risk factors?", ticker="NVDA", year=2024)
print(result["answer"])
print(len(result["sources"]), "chunks used")

# Several section-specific answers combined into one
result = system.query(
    "What drove NVDA's revenue, and what risks does it face?",
    ticker="NVDA",
    use_multi_agent=True,
)
print(result["final_synthesis"]["synthesis"])
```

### Models

| Setting (`LLM_MODEL`) | Model | Context window |
|---|---|---|
| `auto` (default) | Qwen2.5-1.5B if there is ≥3 GB VRAM or ≥10 GB RAM, otherwise TinyLlama | |
| `qwen` | `Qwen/Qwen2.5-1.5B-Instruct` | 32,768 |
| `tinyllama` | `TinyLlama/TinyLlama-1.1B-Chat-v1.0` | 2,048 |
| `phi3` | `microsoft/Phi-3-mini-4k-instruct` | 4,096 |
| `mistral` | `mistralai/Mistral-7B-Instruct-v0.3` | 8,192 |

Retrieval returns up to 10 chunks. Models with a small context window only receive as many of the best-ranked chunks as fit, so TinyLlama answers from about two chunks. All development and evaluation used Qwen.

## How questions are routed

The routing is rule-based. No LLM plans or coordinates anything.

- **Standard query.** Keywords in the question pick a section filter: risk words select Risk Factors, financial words (revenue, earnings, margin, ...) select MD&A, anything else searches all sections.
- **Multi-agent query.** Keywords decide which of three components run, one after another:
  - the *financial analyst* runs a RAG query on MD&A,
  - the *risk analyst* runs a RAG query on Risk Factors,
  - the *price analyst* fetches trailing P/E and sector from yfinance.

  The "synthesis" step joins their outputs under headings. It does not make another LLM call.

## Results

### Section extraction

Checked against all 120 downloaded filings. The original parser mostly returned the wrong text for MD&A (cover-page text, the share-buyback item, XBRL tags); this was found and fixed after the course ended.

| | Original parser | Current parser |
|---|---|---|
| Filings with an MD&A section | 64 of 120, mostly wrong text | 120 of 120 |
| Sections cut off at 50,000 characters | 61% | none |
| Chunks longer than the embedding model's 512-token limit | 93% | none |
| Chunks in total | 4,544 | 10,987 |

### Answer quality and fine-tuning

These runs are from January 2026 and predate the parser fixes above. They have not been re-run, so read them as a record of what was measured, not as the current state.

**RAG answers** (script 06 `--quick`, 2 questions, Qwen2.5-1.5B):

| ROUGE-1 | ROUGE-2 | ROUGE-L | Time per question |
|---|---|---|---|
| 0.218 | 0.071 | 0.145 | 56–82 s |

The answers were on topic and cited figures from the filings, but they run to 1,400–2,500 characters against two-sentence reference answers, so ROUGE is low by construction. Two questions with hand-written references do not measure accuracy.

**LoRA fine-tuning** (script 07 `--quick`: TinyLlama, 50 training samples, 1 epoch, 10 test examples, `virattt/financial-qa-10K`):

| | Before | After |
|---|---|---|
| ROUGE-1 | 0.369 | 0.283 |
| ROUGE-2 | 0.272 | 0.211 |
| ROUGE-L | 0.359 | 0.275 |

Fine-tuning made the scores worse. With 50 samples and 10 test examples this says little either way, and the run had two flaws that have since been fixed: the training loss included padding tokens, and evaluation used sampled rather than greedy decoding.

## Limitations

- **Scale.** The proposal targeted 100,000 filings. The project processes 120, on one machine, with pandas. Nothing here is distributed.
- **No prediction.** The project does not predict market reactions. Script 05 only reports the price change around each filing date, which is not the earnings announcement date, and no sentiment–price correlation is computed.
- **Evaluation.** Five hand-written test questions, no labelled relevance data. The "keyword overlap F1" checks whether the same ten financial terms appear in answer and reference, not whether the numbers agree.
- **Unused code.** Retrieval metrics (precision@k, recall@k, MRR), a keyword-search baseline, a sentiment–price correlation and a prompt-chaining class exist in the code but no script runs them.
- **Parsing gaps.** NVIDIA's 10-Ks keep their financial statements in Item 15, so no Financial Statements section is extracted for them. 10-Q risk sections that only say "no material changes" are skipped.
- **Retrieval.** The year filter uses the filing year, applied after retrieval. The ChromaDB collection uses its default distance metric rather than the cosine similarity the embedding model was trained for.
- **Fine-tuning.** The fine-tuned adapter is not used by the query pipeline.

## Project layout

```
src/earningsiq/
├── config.py            settings (overridable via .env)
├── main.py              EarningsIQ facade used by the scripts
├── data/
│   ├── edgar_downloader.py   download filings, build the filing index
│   ├── preprocessor.py       section extraction and token-based chunking
│   └── price_data.py         yfinance prices and fundamentals
├── rag/
│   ├── vector_store.py       embeddings and ChromaDB
│   └── query_engine.py       model loading, RAG query, keyword router
├── agents/multi_agent.py     rule-based multi-step queries
├── models/fine_tuning.py     LoRA fine-tuning
└── evaluation/metrics.py     ROUGE and keyword-overlap scores
scripts/                 numbered pipeline steps 01–07
tests/                   unit tests for parsing and chunking
docs/PROPOSAL.md         original course proposal
```

Downloaded filings, the vector index, models and logs live under `data/` and `logs/` and are not committed.

## Tests

```bash
pip install -e ".[dev]"
pytest
```

The tests cover section extraction and chunking. The model-dependent parts (retrieval, generation, fine-tuning) have no automated tests.

## License

[MIT](LICENSE)
