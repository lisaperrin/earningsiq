"""Measures of what a filing says and how much it changed.

Tone of the MD&A section, measured three ways:
  kw_tone       positive vs. negative keyword counts (the simple baseline)
  finbert_tone  FinBERT, a BERT model trained on financial sentiment, per sentence
  llm_tone      the project's LLM (Qwen 1.5B), asked per chunk

Change against the company's previous filing of the same type:
  d_*_tone      change in each tone measure
  mdna_change   1 - cosine similarity of the MD&A embeddings
  risk_change   1 - cosine similarity of the Risk Factors embeddings
"""

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch

from ..config import settings
from ..evaluation.metrics import SentimentCorrelationEvaluator
from ..rag.query_engine import LLMManager
from ..utils.logger import log

TONE_SECTION = "MD&A"
CHANGE_SECTIONS = {"MD&A": "mdna_change", "RISK_FACTORS": "risk_change"}
TONE_FEATURES = ["kw_tone", "finbert_tone", "llm_tone"]

# An MD&A with fewer prose sentences than this is not a real MD&A: for filings
# without standard item headings the parser only captures a table of contents.
MIN_TONE_SENTENCES = 30

FINBERT_MODEL = "ProsusAI/finbert"
LLM_MODEL = LLMManager.SUPPORTED_MODELS["qwen"]

LLM_TONE_PROMPT = """Below is an excerpt from the Management's Discussion and Analysis section of a company's {filing_type} report.

\"\"\"
{text}
\"\"\"

Is the tone of this excerpt about the company's business performance and outlook positive, negative or neutral? Answer with exactly one word: positive, negative or neutral."""

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def split_sentences(chunks: list[str]) -> list[str]:
    """Prose sentences of a section, given its overlapping chunks in order.

    Chunk borders cut sentences in two. Because consecutive chunks overlap, a
    sentence cut at the end of one chunk is complete in the next, so the
    border fragments are dropped. Table rows and number-heavy lines are not
    prose and are dropped as well.
    """
    sentences, seen = [], set()

    for position, chunk in enumerate(chunks):
        parts = _SENTENCE_BREAK.split(chunk)
        if position > 0:
            parts = parts[1:]
        if position < len(chunks) - 1:
            parts = parts[:-1]

        for sentence in parts:
            sentence = sentence.strip()
            if sentence in seen or "|" in sentence:
                continue
            n_words = len(sentence.split())
            letters = sum(character.isalpha() for character in sentence)
            if 8 <= n_words <= 80 and letters / len(sentence) >= 0.75:
                seen.add(sentence)
                sentences.append(sentence)

    return sentences


def keyword_tone(sentences: list[str]) -> float:
    """(positive - negative) / (positive + negative) keyword counts, in [-1, 1]."""
    lexicon = SentimentCorrelationEvaluator()
    text = " ".join(sentences).lower()

    def count(words: list[str]) -> int:
        return sum(len(re.findall(rf"\b{re.escape(word)}", text)) for word in words)

    positive, negative = count(lexicon.positive_words), count(lexicon.negative_words)
    total = positive + negative
    return (positive - negative) / total if total else np.nan


def finbert_tone(sentences_by_filing: dict[str, list[str]], batch_size: int = 128) -> dict[str, float]:
    """Mean of P(positive) - P(negative) over a filing's sentences."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL).to(device).eval()
    label_index = {label.lower(): index for index, label in model.config.id2label.items()}

    owners, sentences = [], []
    for accession, filing_sentences in sentences_by_filing.items():
        owners.extend([accession] * len(filing_sentences))
        sentences.extend(filing_sentences)
    log.info(f"FinBERT: scoring {len(sentences)} sentences from {len(sentences_by_filing)} filings on {device}")

    # Similar lengths in a batch keep padding, and therefore runtime, low
    order = np.argsort([len(sentence) for sentence in sentences])
    scores = np.empty(len(sentences))

    with torch.no_grad():
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            inputs = tokenizer(
                [sentences[i] for i in batch],
                padding=True, truncation=True, max_length=128, return_tensors="pt",
            ).to(device)
            probabilities = torch.softmax(model(**inputs).logits, dim=-1)
            tone = probabilities[:, label_index["positive"]] - probabilities[:, label_index["negative"]]
            scores[batch] = tone.float().cpu().numpy()

    del model
    torch.cuda.empty_cache()
    return pd.Series(scores).groupby(owners).mean().to_dict()


def llm_tone(chunks: pd.DataFrame, batch_size: int = 8, max_chars: int = 3500) -> dict[str, float]:
    """Mean of P(positive) - P(negative) over a filing's MD&A chunks.

    The LLM is not asked to generate text. For each chunk the probabilities it
    assigns to "positive", "negative" and "neutral" as the first word of its
    answer are read off directly, which is fast and gives a graded score.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        log.warning("No GPU: LLM tone scoring on CPU will take hours")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL)
    tokenizer.padding_side = "left"  # the answer position must be the last token
    model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL, dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device).eval()

    # First token of each answer word, with and without a capital letter
    answer_tokens = {
        label: sorted({tokenizer.encode(word, add_special_tokens=False)[0] for word in (label, label.capitalize())})
        for label in ("positive", "negative", "neutral")
    }
    all_tokens = [token for tokens in answer_tokens.values() for token in tokens]
    if len(set(all_tokens)) != len(all_tokens):
        raise RuntimeError(f"Answer words share a first token: {answer_tokens}")

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": LLM_TONE_PROMPT.format(filing_type=row.filing_type, text=row.text[:max_chars])}],
            tokenize=False, add_generation_prompt=True,
        )
        for row in chunks.itertuples()
    ]
    log.info(f"LLM tone: scoring {len(prompts)} chunks with {LLM_MODEL} on {device}")

    order = np.argsort([len(prompt) for prompt in prompts])
    scores = np.empty(len(prompts))

    with torch.no_grad():
        for done, start in enumerate(range(0, len(order), batch_size)):
            batch = order[start:start + batch_size]
            inputs = tokenizer([prompts[i] for i in batch], padding=True, return_tensors="pt").to(device)
            next_token = torch.log_softmax(model(**inputs).logits[:, -1, :].float(), dim=-1)
            label_logp = torch.stack(
                [torch.logsumexp(next_token[:, answer_tokens[label]], dim=-1) for label in ("positive", "negative", "neutral")],
                dim=-1,
            )
            label_p = torch.softmax(label_logp, dim=-1)
            scores[batch] = (label_p[:, 0] - label_p[:, 1]).cpu().numpy()
            if done % 100 == 0:
                log.info(f"LLM tone: {start + len(batch)}/{len(prompts)} chunks")

    del model
    torch.cuda.empty_cache()
    return pd.Series(scores).groupby(chunks["accession"].to_numpy()).mean().to_dict()


def section_embeddings(chunks: pd.DataFrame, batch_size: int = 64) -> pd.DataFrame:
    """Mean chunk embedding per filing and section (rows: accession, section, embedding)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(settings.embedding_model)
    log.info(f"Embedding {len(chunks)} chunks for the change measures")
    vectors = model.encode(
        chunks["text"].tolist(), batch_size=batch_size, normalize_embeddings=True,
        show_progress_bar=False, convert_to_numpy=True,
    )

    rows = []
    keys = chunks[["accession", "section"]].reset_index(drop=True)
    for (accession, section), group in keys.groupby(["accession", "section"]):
        rows.append({"accession": accession, "section": section, "embedding": vectors[group.index].mean(axis=0)})

    del model
    torch.cuda.empty_cache()
    return pd.DataFrame(rows)


def _cosine_distance(a, b) -> float:
    if a is None or b is None or np.ndim(a) == 0 or np.ndim(b) == 0:
        return np.nan
    return float(1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def add_change_features(features: pd.DataFrame, embeddings: pd.DataFrame) -> pd.DataFrame:
    """Compare each filing with the company's previous filing of the same type.

    `features` needs ticker, filing_type, filing_accepted, accession and the
    tone columns. The first filing of each company and type has no
    predecessor, so its change features are NaN.
    """
    features = features.sort_values(["ticker", "filing_type", "filing_accepted"]).reset_index(drop=True)
    previous = features.groupby(["ticker", "filing_type"])

    for tone in TONE_FEATURES:
        if tone in features:
            features[f"d_{tone}"] = features[tone] - previous[tone].shift(1)

    previous_accession = previous["accession"].shift(1)
    for section, column in CHANGE_SECTIONS.items():
        by_accession = embeddings[embeddings["section"] == section].set_index("accession")["embedding"].to_dict()
        features[column] = [
            _cosine_distance(by_accession.get(current), by_accession.get(prior))
            for current, prior in zip(features["accession"], previous_accession)
        ]

    return features


def accession_of(metadata: dict) -> str:
    """EDGAR accession number of the filing a chunk came from."""
    return Path(metadata["file_path"]).parent.name


def build_text_features(chunks: pd.DataFrame, events: pd.DataFrame, use_llm: bool = True) -> pd.DataFrame:
    """Text features for every filing in `events` (one row per accession)."""
    chunks = chunks.copy()
    chunks["accession"] = chunks["metadata"].map(accession_of)
    chunks = chunks[chunks["accession"].isin(events["accession"])].sort_values(["accession", "section", "chunk_id"])

    tone_chunks = chunks[chunks["section"] == TONE_SECTION].reset_index(drop=True)
    sentences = {
        accession: split_sentences(group["text"].tolist())
        for accession, group in tone_chunks.groupby("accession")
    }

    features = events[["ticker", "filing_type", "accession", "filing_accepted"]].copy()
    features["n_sentences"] = features["accession"].map({k: len(v) for k, v in sentences.items()}).fillna(0).astype(int)

    usable = {accession for accession, found in sentences.items() if len(found) >= MIN_TONE_SENTENCES}
    log.info(f"{len(usable)} of {len(features)} filings have a usable MD&A section")
    sentences = {accession: sentences[accession] for accession in usable}
    tone_chunks = tone_chunks[tone_chunks["accession"].isin(usable)].reset_index(drop=True)

    features["kw_tone"] = features["accession"].map({k: keyword_tone(v) for k, v in sentences.items()})
    features["finbert_tone"] = features["accession"].map(finbert_tone(sentences))
    if use_llm:
        features["llm_tone"] = features["accession"].map(llm_tone(tone_chunks))

    change_chunks = chunks[
        ((chunks["section"] == TONE_SECTION) & chunks["accession"].isin(usable))
        | ((chunks["section"] != TONE_SECTION) & chunks["section"].isin(CHANGE_SECTIONS))
    ].reset_index(drop=True)
    return add_change_features(features, section_embeddings(change_chunks))
