# EarningsIQ Project Structure

## Complete Implementation Overview

### Core Modules

#### 1. Data Pipeline (`src/earningsiq/data/`)

**edgar_downloader.py**
- EdgarDownloader: Downloads SEC filings via sec-edgar-downloader
- load_sp500_tickers(): Fetches S&P 500 ticker list
- load_sample_tickers(): Sample tickers for testing
- Creates filing index in Parquet format

**preprocessor.py**
- SecFilingParser: Extracts text from HTML/PDF filings
- DocumentChunker: Chunks documents (512 tokens, 100 overlap)
- DocumentProcessor: End-to-end processing pipeline

**price_data.py**
- PriceDataFetcher: yfinance integration
- get_stock_data(): Historical price data
- calculate_price_reaction(): Post-earnings price movements
- get_fundamentals(): Company fundamentals (P/E, margins, etc.)

#### 2. RAG System (`src/earningsiq/rag/`)

**vector_store.py**
- EmbeddingModel: bge-small-en-v1.5 wrapper
- VectorStore: ChromaDB integration
- DocumentRetriever: Semantic search with metadata filtering

**query_engine.py**
- LLMManager: Multi-model support (TinyLlama, Qwen, Phi-3, Mistral) via Hugging Face Transformers
- RAGQueryEngine: Retrieval + generation pipeline
- PromptChainEngine: Multi-step reasoning
- QueryRouter: Intelligent query routing by type

#### 3. Model Training (`src/earningsiq/models/`)

**fine_tuning.py**
- FinancialDatasetPreparator: FinQA dataset loader
- LoRAFineTuner: PEFT with LoRA (rank=16, alpha=32)
- ModelComparator: Baseline vs fine-tuned evaluation

#### 4. Multi-Agent System (`src/earningsiq/agents/`)

**multi_agent.py**
- QueryDecomposerAgent: Breaks down complex queries
- FinancialAnalystAgent: MD&A analysis
- RiskAnalystAgent: Risk factor extraction
- PriceAnalystAgent: Market data integration
- SynthesisAgent: Combines agent outputs
- MultiAgentOrchestrator: Coordinates all agents

#### 5. Evaluation (`src/earningsiq/evaluation/`)

**metrics.py**
- ROUGEEvaluator: Summarization quality (ROUGE-1/2/L)
- EntityExtractionEvaluator: F1 for financial entities
- SentimentCorrelationEvaluator: Sentiment-price correlation
- RetrievalEvaluator: Precision@k, Recall@k, MRR
- BaselineComparator: LLM vs keyword search
- PerformanceMetrics: Time-to-insight tracking

### Execution Scripts (`scripts/`)

1. **01_download_data.py** - Download SEC filings
2. **02_process_documents.py** - Extract and chunk documents
3. **03_build_vector_index.py** - Build ChromaDB index
4. **04_query_system.py** - Run sample queries
5. **05_price_correlation.py** - Analyze price reactions
6. **06_evaluate_system.py** - Run evaluation metrics (ROUGE, F1)
7. **07_finetune_model.py** - Fine-tune with LoRA and compare before/after

### Main Application (`src/earningsiq/main.py`)

**EarningsIQ Class**
- setup(): Initialize all components
- download_data(): Batch download filings
- process_documents(): Parallel document processing
- build_vector_index(): Create embeddings
- query(): Execute queries (single-agent or multi-agent)
- fine_tune_model(): Run PEFT training
- evaluate(): Run benchmark suite

**Demo Functions**
- run_data_pipeline(): End-to-end data setup
- run_query_demo(): Sample query execution
- run_multi_agent_demo(): Multi-agent analysis

### Data Flow

```
1. Data Ingestion
   SEC EDGAR → edgar_downloader.py → data/raw/

2. Preprocessing
   data/raw/ → preprocessor.py → data/processed/chunked_documents.parquet

3. Embedding
   chunked_documents.parquet → vector_store.py → data/chromadb/

4. Price Data
   yfinance API → price_data.py → data/processed/price_reactions.parquet

5. Query Execution
   User Query → query_engine.py → VectorStore → LLM → Answer

6. Multi-Agent
   Complex Query → multi_agent.py → [Agent Pool] → Synthesis → Answer
```



