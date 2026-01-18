from typing import Optional, List, Dict
from pathlib import Path
import torch
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import pandas as pd
from ..config import settings
from ..utils.logger import log

try:
    from llama_index.core import Settings
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    LLAMA_INDEX_AVAILABLE = True
except ImportError:
    LLAMA_INDEX_AVAILABLE = False
    Settings = None
    HuggingFaceEmbedding = None


def auto_select_model() -> tuple[str, int]:
    """Auto-select best model based on available RAM and GPU.

    Returns:
        tuple: (model_name, context_window)
    """
    ram_gb = psutil.virtual_memory().total / (1024 ** 3)
    has_gpu = torch.cuda.is_available()
    gpu_mem_gb = 0

    if has_gpu:
        try:
            gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except:
            pass

    log.info(f"System: {ram_gb:.1f}GB RAM, GPU: {has_gpu} ({gpu_mem_gb:.1f}GB VRAM)")

    models = [
        ("Qwen/Qwen2.5-1.5B-Instruct", 6, 3, 32768),
        ("mistralai/Mistral-7B-Instruct-v0.3", 24, 14, 8192),
        ("microsoft/Phi-3-mini-4k-instruct", 12, 8, 4096),
        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", 4, 2, 2048),
    ]

    for model_name, min_ram, min_gpu, context in models:
        # Check if we have enough resources
        if has_gpu and gpu_mem_gb >= min_gpu:
            log.info(f"Selected {model_name} (context: {context} tokens) - GPU mode")
            return model_name, context
        elif ram_gb >= min_ram + 4:  # Extra headroom for CPU mode
            log.info(f"Selected {model_name} (context: {context} tokens) - CPU mode")
            return model_name, context

    # Ultimate fallback
    log.warning("Low resources detected, using TinyLlama (limited context)")
    return "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 2048


class LLMManager:
    SUPPORTED_MODELS = {
        "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        "phi3": "microsoft/Phi-3-mini-4k-instruct",
        "mistral": "mistralai/Mistral-7B-Instruct-v0.3",
        "qwen": "Qwen/Qwen2.5-1.5B-Instruct",
    }

    MODEL_CONTEXT_WINDOWS = {
        "TinyLlama/TinyLlama-1.1B-Chat-v1.0": 2048,
        "microsoft/Phi-3-mini-4k-instruct": 4096,
        "mistralai/Mistral-7B-Instruct-v0.3": 8192,
        "Qwen/Qwen2.5-1.5B-Instruct": 32768,
    }

    def __init__(self, model_name: str = None):
        if model_name is None or model_name == "auto":
            self.model_name, self.context_window = auto_select_model()
        else:
            self.model_name = model_name
            if self.model_name in self.SUPPORTED_MODELS:
                self.model_name = self.SUPPORTED_MODELS[self.model_name]
            self.context_window = self.MODEL_CONTEXT_WINDOWS.get(self.model_name, 2048)

        self.tokenizer = None
        self.model = None
        self.pipeline = None

        log.info(f"Initializing LLM: {self.model_name} (context: {self.context_window})")
        self._load_model()

    def _load_model(self):
        try:
            log.info(f"Loading model {self.model_name}...")
            log.info(f"This may take several minutes on first run (downloading model)...")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            use_gpu = torch.cuda.is_available()

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if use_gpu else torch.float32,
                device_map="auto" if use_gpu else "cpu",
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            self.pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=True,
                top_p=0.95,
                pad_token_id=self.tokenizer.pad_token_id,
            )

            device = "GPU" if use_gpu else "CPU"
            log.info(f"Model loaded successfully on {device}")
            log.info(f"Model size: {sum(p.numel() for p in self.model.parameters()) / 1e9:.2f}B parameters")

        except Exception as e:
            log.error(f"Failed to load model {self.model_name}: {e}")

            if "TinyLlama" not in self.model_name:
                log.warning("Falling back to TinyLlama (smallest model)...")
                try:
                    self.model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    if self.tokenizer.pad_token is None:
                        self.tokenizer.pad_token = self.tokenizer.eos_token

                    self.model = AutoModelForCausalLM.from_pretrained(
                        self.model_name,
                        torch_dtype=torch.float32,
                        device_map="cpu",
                        low_cpu_mem_usage=True
                    )

                    self.pipeline = pipeline(
                        "text-generation",
                        model=self.model,
                        tokenizer=self.tokenizer,
                        max_new_tokens=512,
                        temperature=0.1,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

                    log.info(f"Fallback model {self.model_name} loaded successfully")

                except Exception as fallback_error:
                    log.error(f"Failed to load fallback model: {fallback_error}")
                    self.pipeline = None
            else:
                self.pipeline = None

    def generate(self, prompt: str, max_length: int = 512) -> str:
        if self.pipeline is None:
            log.error("LLM not loaded")
            return "Error: LLM model not available"

        try:
            messages = [{"role": "user", "content": prompt}]

            if hasattr(self.tokenizer, 'apply_chat_template'):
                formatted_prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                formatted_prompt = prompt

            response = self.pipeline(
                formatted_prompt,
                max_new_tokens=max_length,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

            generated_text = response[0]['generated_text']

            if isinstance(generated_text, str):
                if formatted_prompt in generated_text:
                    answer = generated_text[len(formatted_prompt):].strip()
                else:
                    answer = generated_text.strip()
            else:
                answer = str(generated_text)

            return answer

        except Exception as e:
            log.error(f"Generation failed: {e}")
            return f"Error during generation: {str(e)}"


class RAGQueryEngine:
    FINANCIAL_QA_TEMPLATE = """You are a financial analyst expert. Use the following context from SEC filings to answer the question.

Context:
{context}

Question: {question}

Instructions:
- Provide a clear, accurate answer based on the context
- Cite specific information from the filings
- If the context doesn't contain enough information, state that clearly
- Focus on quantifiable data and specific statements

Answer:"""

    def __init__(self, vector_store, llm_model_name: Optional[str] = None):
        self.vector_store = vector_store
        self.llm_manager = LLMManager(llm_model_name)

        self._setup_llama_index()

    def _setup_llama_index(self):
        if LLAMA_INDEX_AVAILABLE and Settings is not None:
            try:
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=settings.embedding_model
                )
                Settings.chunk_size = settings.chunk_size
                Settings.chunk_overlap = settings.chunk_overlap
            except Exception as e:
                log.warning(f"Could not configure llama-index settings: {e}")
        else:
            log.info("LlamaIndex not available, skipping configuration")

    def query(
        self,
        question: str,
        ticker: Optional[str] = None,
        section: Optional[str] = None,
        year: Optional[int] = None,
        top_k: int = 10
    ) -> Dict[str, any]:
        filter_dict = {}
        if ticker:
            filter_dict["ticker"] = ticker
        if section:
            filter_dict["section"] = section

        fetch_k = top_k * 10 if year else top_k

        results = self.vector_store.query(
            question,
            n_results=fetch_k,
            filter_dict=filter_dict if filter_dict else None
        )

        if year and results["documents"] and results["documents"][0]:
            filtered = {"documents": [[]], "metadatas": [[]], "distances": [[]]}
            for i in range(len(results["documents"][0])):
                filing_date = results["metadatas"][0][i].get("filing_date", "")
                if str(year) in filing_date:
                    filtered["documents"][0].append(results["documents"][0][i])
                    filtered["metadatas"][0].append(results["metadatas"][0][i])
                    filtered["distances"][0].append(results["distances"][0][i])
                    if len(filtered["documents"][0]) >= top_k:
                        break
            pre_filter_count = len(results["documents"][0])
            post_filter_count = len(filtered["documents"][0])
            if post_filter_count == 0:
                log.warning(f"Year filter removed all {pre_filter_count} results (year={year})")
            else:
                log.info(f"Year filter: {pre_filter_count} -> {post_filter_count} documents (year={year})")
            results = filtered

        context_parts = []
        source_docs = []

        doc_count = len(results["documents"][0]) if results["documents"] and results["documents"][0] else 0
        if doc_count == 0:
            log.warning(f"No documents retrieved for query: {question[:50]}...")
        else:
            log.info(f"Retrieved {doc_count} documents for query")

        if results["documents"]:
            for i in range(len(results["documents"][0])):
                doc_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i]

                context_parts.append(
                    f"[Source {i+1}] {metadata['ticker']} - {metadata['filing_type']} "
                    f"({metadata['filing_date']}) - {metadata['section']}:\n{doc_text}"
                )

                source_docs.append({
                    "text": doc_text,
                    "metadata": metadata,
                    "distance": results["distances"][0][i]
                })

        context = "\n\n".join(context_parts)

        prompt = self.FINANCIAL_QA_TEMPLATE.format(
            context=context,
            question=question
        )

        answer = self.llm_manager.generate(prompt)

        return {
            "question": question,
            "answer": answer,
            "sources": source_docs,
            "context": context
        }

    def batch_query(self, questions: List[str], **kwargs) -> List[Dict]:
        results = []
        for question in questions:
            result = self.query(question, **kwargs)
            results.append(result)
        return results


class PromptChainEngine:
    def __init__(self, rag_engine: RAGQueryEngine):
        self.rag_engine = rag_engine

    def extract_guidance(self, ticker: str, filing_date: str) -> Dict:
        questions = [
            f"What revenue guidance did {ticker} provide?",
            f"What EPS guidance did {ticker} provide?",
            f"What are the key metrics {ticker} mentioned for future performance?"
        ]

        results = []
        for q in questions:
            result = self.rag_engine.query(q, ticker=ticker)
            results.append(result)

        return {
            "ticker": ticker,
            "filing_date": filing_date,
            "guidance_analysis": results
        }

    def extract_risks(self, ticker: str, filing_date: str) -> Dict:
        questions = [
            f"What are the main risk factors mentioned by {ticker}?",
            f"What operational risks does {ticker} face?",
            f"What market or competitive risks did {ticker} identify?"
        ]

        results = []
        for q in questions:
            result = self.rag_engine.query(q, ticker=ticker, section="RISK_FACTORS")
            results.append(result)

        return {
            "ticker": ticker,
            "filing_date": filing_date,
            "risk_analysis": results
        }

    def sentiment_analysis(self, ticker: str, filing_date: str) -> Dict:
        question = (
            f"Analyze the overall tone and sentiment of {ticker}'s management discussion. "
            f"Is it optimistic, cautious, or pessimistic? Provide specific examples."
        )

        result = self.rag_engine.query(question, ticker=ticker, section="MD&A")

        return {
            "ticker": ticker,
            "filing_date": filing_date,
            "sentiment": result
        }

    def compare_quarters(self, ticker: str, date1: str, date2: str) -> Dict:
        question = (
            f"Compare {ticker}'s performance and outlook between {date1} and {date2}. "
            f"What changed in terms of revenue, risks, and guidance?"
        )

        result = self.rag_engine.query(question, ticker=ticker, top_k=10)

        return {
            "ticker": ticker,
            "comparison": f"{date1} vs {date2}",
            "analysis": result
        }


class QueryRouter:
    QUERY_TYPES = {
        "guidance": ["guidance", "forecast", "outlook", "expect", "project"],
        "risk": ["risk", "concern", "challenge", "threat", "uncertainty"],
        "financial": ["revenue", "earnings", "profit", "margin", "cash flow"],
        "sentiment": ["sentiment", "tone", "optimistic", "pessimistic", "outlook"],
    }

    def __init__(self, rag_engine: RAGQueryEngine):
        self.rag_engine = rag_engine
        self.chain_engine = PromptChainEngine(rag_engine)

    def classify_query(self, query: str) -> str:
        query_lower = query.lower()

        for query_type, keywords in self.QUERY_TYPES.items():
            if any(keyword in query_lower for keyword in keywords):
                return query_type

        return "general"

    def route_query(self, query: str, ticker: Optional[str] = None, **kwargs) -> Dict:
        query_type = self.classify_query(query)

        if query_type == "risk":
            kwargs["section"] = "RISK_FACTORS"
        elif query_type == "financial":
            kwargs["section"] = "MD&A"

        result = self.rag_engine.query(query, ticker=ticker, **kwargs)
        result["query_type"] = query_type

        return result
