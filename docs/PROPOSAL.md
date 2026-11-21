# EarningsIQ: LLM-Powered Financial Report Analysis & Market Reaction Predictor

## Capstone Project Proposal

**Course:** Big Data
**Student ID:** 0256130201
**Date:** January 18, 2026

---

## 1. Problem Statement

### Business Problem
Investment analysts and portfolio managers manually read hundreds of pages of quarterly earnings reports (10-Q) and annual reports (10-K) for each company they track. With 5,000+ public companies filing reports quarterly, this creates an insurmountable data analysis bottleneck.


### Technical Challenge
Process up to 100,000+ SEC filings (10M+ pages of unstructured text) to extract actionable insights and try to predict market reactions using advanced LLM techniques at scale.

### Quantifiable Impact Goal
- Reduce time-to-insight from 50 hours (manual) to <2 hours (automated) per 100 reports
- Achieve >0.85 F1 score on financial entity extraction (vs 0.65 baseline)
- Demonstrate >0.35 correlation between sentiment and price movement (vs 0.12 keyword baseline)

---

## 2. Dataset Description

### Primary Data Sources

**SEC EDGAR Database**
- Source: sec.gov public API
- Volume: 100,000 filings (2019-2024)
- Coverage: S&P 500 + Russell 2000 companies
- Filing Types: 10-Q (quarterly), 10-K (annual)
- Size: ~10 million pages, ~50GB unstructured text
- Key Sections: MD&A, Risk Factors, Financial Statements

**Market Data**
- Source: yfinance Python library
- Coverage: Earnings dates, daily prices, fundamentals
- Window: 5-day price windows around earnings dates
- Metrics: Close price, volume, P/E ratio, sector data

**Fine-Tuning Dataset**
- Source: FinQA dataset (Hugging Face)
- Size: 9,000+ expert-annotated financial Q&A pairs
- Purpose: PEFT fine-tuning for financial reasoning

### Data Characteristics
- Unstructured text: Management discussion, risk narratives
- Temporal: 5 years of quarterly data
- Multi-modal: Text + numerical financial data
- Public domain: No privacy concerns

---

## 3. Methodology

### Architecture Overview

```
[SEC Filings] → [Preprocessing] → [Vector DB] → [RAG Engine] → [LLM] → [Insights]
                                                      ↓
                                              [Multi-Agent System]
                                                      ↓
[yfinance] → [Price Data] → [Correlation Analysis] → [Predictions]
```

### Phase 1: Data Pipeline (Objective 2: Scalability)

**Ingestion:**
- sec-edgar-downloader: Batch download 100K filings
- Polars: Fast dataframe operations for large-scale processing

**Preprocessing:**
- BeautifulSoup + lxml: Extract text from HTML filings
- Section detection: Regex-based extraction of MD&A, Risk Factors
- Chunking: 512 tokens with 100-token overlap → 20M+ chunks

**Storage:**
- Parquet: Compressed metadata storage
- ChromaDB: Vector embeddings for 20M+ chunks

### Phase 2: RAG System (Objective 1: Technical Implementation)

**Embedding:**
- Model: bge-small-en-v1.5 (384 dimensions)
- Batch processing: 10K chunks at a time
- ChromaDB persistent storage

**Retrieval:**
- Semantic search with metadata filtering (ticker, section, date)
- Top-k retrieval (k=5-10)
- Context window: 8K tokens

**Generation:**
- LLMs: TinyLlama-1.1B, Qwen-1.5B, Phi-3-mini-4k, Mistral-7B (selectable)
- Framework: Hugging Face Transformers
- Inference: GPU-accelerated with FP16, optional 4-bit quantization

### Phase 3: Advanced Techniques (Objective 3)

**PEFT Fine-Tuning:**
- Method: LoRA (rank=16, alpha=32)
- Dataset: Financial Q&A from Hugging Face (7K+ samples)
- Trainable parameters: <1% of model
- Optional: 4-bit quantization with bitsandbytes
- Baseline comparison: Zero-shot vs Fine-tuned with ROUGE metrics

**Advanced Prompt Chaining:**
1. Query Router: Classifies query type (guidance/risk/financial)
2. Section Specialist: Routes to appropriate filing section
3. Entity Extractor: Pulls numerical metrics
4. Sentiment Analyzer: Analyzes management tone
5. Synthesis Agent: Combines results

### Phase 4: Multi-Agent System (Objective 1)

**Agents:**
- Query Decomposer: Breaks complex queries into sub-tasks
- Financial Analyst: Analyzes MD&A sections
- Risk Analyst: Extracts risk factors
- Price Analyst: Fetches market data via yfinance
- Synthesis Agent: Combines agent outputs

**Workflow:**
```
User Query → Decomposer → [Agent Pool] → Synthesis → Final Answer
```

### Phase 5: Price Correlation Analysis

**Pipeline:**
1. Extract earnings dates from yfinance
2. Calculate price change (-2 to +2 days around earnings)
3. Extract sentiment from report text
4. Correlate sentiment score with price movement
5. Classification: Predict up/down movement

---

## 4. Evaluation & Expected Outcomes (Objective 5)

### Metrics

**Summarization Quality:**
- Metric: ROUGE-L score
- Baseline: N/A (no prior summarization)
- Target: >0.60 ROUGE-L

**Entity Extraction:**
- Metric: F1 score
- Baseline: Regex extraction = 0.65
- Target: LLM extraction >0.85
- Entities: Revenue, EPS, guidance, risk factors

**Sentiment-Price Correlation:**
- Metric: Pearson correlation coefficient
- Baseline: Keyword sentiment = 0.12
- Target: LLM sentiment >0.35

**Retrieval Quality:**
- Metric: Precision@5, Recall@5
- Baseline: Keyword search
- Target: >30% improvement over baseline

### Evaluation Dataset
- Test set
- Ground truth: Labeled Q&A pairs

---

## 5. Ethical Considerations & Bias Analysis (Objective 4)

### Identified Biases

**1. Large-Cap Bias**
- Issue: S&P 500 companies have more analyst coverage, cleaner data
- Mitigation: Include Russell 2000 small-caps, stratify sampling by market cap

**2. Survivorship Bias**
- Issue: Bankrupt/delisted companies removed from datasets
- Mitigation: Document limitation, note that model trained on "surviving" companies

**3. Temporal Bias**
- Issue: Reporting language evolves (pre-COVID vs post-COVID)
- Mitigation: Train on recent data (2019-2024), include pandemic period


### Privacy & Security
- Data: Public SEC filings (no PII)
- Compliance: Fair use of publicly available data

---

## 6. Technology Stack

**Big Data Tools:**
- Polars: Fast dataframe operations for large-scale processing
- ChromaDB: Vector database for 20M+ embeddings
- Parquet: Efficient columnar storage

**LLM Stack:**
- Multiple models: TinyLlama-1.1B, Qwen-1.5B, Phi-3-mini-4k, Mistral-7B
- Hugging Face Transformers: Model loading and inference
- PEFT + LoRA: Parameter-efficient fine-tuning

**Embeddings:**
- sentence-transformers (bge-small-en-v1.5)

**Data Processing:**
- sec-edgar-downloader: SEC filings
- yfinance: Market data
- BeautifulSoup + lxml: HTML text extraction

**Evaluation:**
- ROUGE: Summarization quality
- scikit-learn: Classification metrics (F1, precision, recall)
- scipy: Correlation analysis

---

## 8. Expected Deliverables

### GitHub Repository Structure
```
Big-Data-Capstone/
├── src/earningsiq/
│   ├── data/          # EDGAR downloader, preprocessing
│   ├── rag/           # Vector store, query engine
│   ├── models/        # Fine-tuning pipeline
│   ├── agents/        # Multi-agent system
│   └── evaluation/    # Metrics framework
├── scripts/           # Pipeline scripts
├── data/             # Processed data (gitignored)
├── requirements.txt
├── setup.py
└── USAGE.md
```

---

## 9. References

- SEC EDGAR: https://www.sec.gov/edgar
- Hugging Face Transformers: https://huggingface.co/docs/transformers/
- PEFT: https://huggingface.co/docs/peft/
- ChromaDB: https://docs.trychroma.com/
- sentence-transformers: https://www.sbert.net/
