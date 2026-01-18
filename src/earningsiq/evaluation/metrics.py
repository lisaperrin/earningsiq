from typing import List, Dict, Optional
import re
from dataclasses import dataclass
import pandas as pd
import numpy as np
from rouge_score import rouge_scorer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from scipy.stats import pearsonr, spearmanr
from ..utils.logger import log


@dataclass
class EvaluationResult:
    metric_name: str
    score: float
    baseline_score: Optional[float] = None
    improvement: Optional[float] = None
    details: Optional[Dict] = None


class ROUGEEvaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    def evaluate(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        if len(predictions) != len(references):
            log.error("Predictions and references must have same length")
            return {}

        rouge1_scores = []
        rouge2_scores = []
        rougeL_scores = []

        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        return {
            'rouge1': np.mean(rouge1_scores),
            'rouge2': np.mean(rouge2_scores),
            'rougeL': np.mean(rougeL_scores),
        }


class EntityExtractionEvaluator:
    FINANCIAL_ENTITIES = [
        'revenue', 'earnings', 'eps', 'ebitda', 'margin', 'profit',
        'guidance', 'forecast', 'outlook', 'cash flow'
    ]

    def __init__(self):
        pass

    def extract_entities(self, text: str) -> List[str]:
        text_lower = text.lower()
        found_entities = []

        for entity in self.FINANCIAL_ENTITIES:
            if entity in text_lower:
                found_entities.append(entity)

        return found_entities

    def extract_numerical_entities(self, text: str) -> List[Dict]:
        patterns = [
            r'\$[\d,]+\.?\d*[MBK]?',
            r'\d+\.?\d*%',
            r'\d+\.?\d*[MBK]?\s*(?:dollars|revenue|earnings|profit)',
        ]

        entities = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities.extend(matches)

        return entities

    def evaluate(
        self,
        predictions: List[str],
        ground_truth: List[str]
    ) -> Dict[str, float]:
        pred_entities_list = [set(self.extract_entities(pred)) for pred in predictions]
        true_entities_list = [set(self.extract_entities(truth)) for truth in ground_truth]

        precision_scores = []
        recall_scores = []
        f1_scores = []

        for pred_entities, true_entities in zip(pred_entities_list, true_entities_list):
            if len(pred_entities) == 0 and len(true_entities) == 0:
                precision_scores.append(1.0)
                recall_scores.append(1.0)
                f1_scores.append(1.0)
            elif len(pred_entities) == 0:
                precision_scores.append(0.0)
                recall_scores.append(0.0)
                f1_scores.append(0.0)
            else:
                tp = len(pred_entities & true_entities)
                precision = tp / len(pred_entities) if len(pred_entities) > 0 else 0
                recall = tp / len(true_entities) if len(true_entities) > 0 else 0
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)

        return {
            'precision': np.mean(precision_scores),
            'recall': np.mean(recall_scores),
            'f1': np.mean(f1_scores),
        }


class SentimentCorrelationEvaluator:
    def __init__(self):
        self.positive_words = ['growth', 'increase', 'strong', 'improved', 'positive', 'success', 'gain']
        self.negative_words = ['decline', 'decrease', 'weak', 'concern', 'risk', 'challenge', 'loss']

    def calculate_sentiment_score(self, text: str) -> float:
        text_lower = text.lower()

        positive_count = sum(1 for word in self.positive_words if word in text_lower)
        negative_count = sum(1 for word in self.negative_words if word in text_lower)

        if positive_count + negative_count == 0:
            return 0.0

        sentiment = (positive_count - negative_count) / (positive_count + negative_count)
        return sentiment

    def evaluate_correlation(
        self,
        texts: List[str],
        price_changes: List[float]
    ) -> Dict[str, float]:
        sentiment_scores = [self.calculate_sentiment_score(text) for text in texts]

        if len(sentiment_scores) != len(price_changes):
            log.error("Texts and price changes must have same length")
            return {}

        pearson_corr, pearson_p = pearsonr(sentiment_scores, price_changes)
        spearman_corr, spearman_p = spearmanr(sentiment_scores, price_changes)

        return {
            'pearson_correlation': pearson_corr,
            'pearson_p_value': pearson_p,
            'spearman_correlation': spearman_corr,
            'spearman_p_value': spearman_p,
        }


class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_mrr(self, relevant_ranks: List[int]) -> float:
        reciprocal_ranks = [1.0 / rank if rank > 0 else 0.0 for rank in relevant_ranks]
        return np.mean(reciprocal_ranks)

    def calculate_precision_at_k(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k: int = 5
    ) -> float:
        precisions = []

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_k = set(retrieved[:k])
            relevant_set = set(relevant)
            precision = len(retrieved_k & relevant_set) / k if k > 0 else 0
            precisions.append(precision)

        return np.mean(precisions)

    def calculate_recall_at_k(
        self,
        retrieved_docs: List[List[str]],
        relevant_docs: List[List[str]],
        k: int = 5
    ) -> float:
        recalls = []

        for retrieved, relevant in zip(retrieved_docs, relevant_docs):
            retrieved_k = set(retrieved[:k])
            relevant_set = set(relevant)
            recall = len(retrieved_k & relevant_set) / len(relevant_set) if len(relevant_set) > 0 else 0
            recalls.append(recall)

        return np.mean(recalls)


class BaselineComparator:
    def __init__(self):
        self.keyword_patterns = {
            'revenue': r'revenue|sales',
            'earnings': r'earnings|profit|income',
            'guidance': r'guidance|forecast|outlook|expect',
            'risk': r'risk|concern|challenge|uncertainty',
        }

    def keyword_search(self, query: str, documents: List[str], top_k: int = 5) -> List[str]:
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_docs = []
        for doc in documents:
            doc_lower = doc.lower()
            doc_words = set(doc_lower.split())

            overlap = len(query_words & doc_words)
            scored_docs.append((doc, overlap))

        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return [doc for doc, score in scored_docs[:top_k]]

    def compare_retrieval(
        self,
        llm_retrieved: List[List[str]],
        keyword_retrieved: List[List[str]],
        ground_truth: List[List[str]]
    ) -> Dict[str, Dict]:
        retrieval_eval = RetrievalEvaluator()

        llm_precision = retrieval_eval.calculate_precision_at_k(llm_retrieved, ground_truth, k=5)
        llm_recall = retrieval_eval.calculate_recall_at_k(llm_retrieved, ground_truth, k=5)

        keyword_precision = retrieval_eval.calculate_precision_at_k(keyword_retrieved, ground_truth, k=5)
        keyword_recall = retrieval_eval.calculate_recall_at_k(keyword_retrieved, ground_truth, k=5)

        return {
            'llm_system': {
                'precision@5': llm_precision,
                'recall@5': llm_recall,
            },
            'keyword_baseline': {
                'precision@5': keyword_precision,
                'recall@5': keyword_recall,
            },
            'improvement': {
                'precision': llm_precision - keyword_precision,
                'recall': llm_recall - keyword_recall,
            }
        }


class PerformanceMetrics:
    def __init__(self):
        self.metrics = []

    def record_query_time(self, query_id: str, time_seconds: float, method: str):
        self.metrics.append({
            'query_id': query_id,
            'time_seconds': time_seconds,
            'method': method
        })

    def calculate_time_reduction(self) -> Dict[str, float]:
        df = pd.DataFrame(self.metrics)

        if df.empty:
            return {}

        grouped = df.groupby('method')['time_seconds'].mean()

        if 'llm_system' in grouped and 'baseline' in grouped:
            reduction = (grouped['baseline'] - grouped['llm_system']) / grouped['baseline'] * 100
            return {
                'llm_avg_time': grouped['llm_system'],
                'baseline_avg_time': grouped['baseline'],
                'time_reduction_pct': reduction
            }

        return {}


class ComprehensiveEvaluator:
    def __init__(self):
        self.rouge_eval = ROUGEEvaluator()
        self.entity_eval = EntityExtractionEvaluator()
        self.sentiment_eval = SentimentCorrelationEvaluator()
        self.retrieval_eval = RetrievalEvaluator()
        self.baseline_comp = BaselineComparator()

    def run_full_evaluation(
        self,
        predictions: List[str],
        references: List[str],
        price_changes: Optional[List[float]] = None
    ) -> Dict[str, EvaluationResult]:
        results = {}

        rouge_scores = self.rouge_eval.evaluate(predictions, references)
        results['rouge'] = EvaluationResult(
            metric_name='ROUGE-L',
            score=rouge_scores.get('rougeL', 0.0),
            details=rouge_scores
        )

        entity_scores = self.entity_eval.evaluate(predictions, references)
        results['entity_extraction'] = EvaluationResult(
            metric_name='Entity Extraction F1',
            score=entity_scores.get('f1', 0.0),
            details=entity_scores
        )

        if price_changes:
            sentiment_corr = self.sentiment_eval.evaluate_correlation(predictions, price_changes)
            results['sentiment_correlation'] = EvaluationResult(
                metric_name='Sentiment-Price Correlation',
                score=sentiment_corr.get('pearson_correlation', 0.0),
                details=sentiment_corr
            )

        return results

    def generate_report(self, results: Dict[str, EvaluationResult]) -> str:
        report_lines = ["Evaluation Report", "=" * 50]

        for metric_name, result in results.items():
            report_lines.append(f"\n{result.metric_name}:")
            report_lines.append(f"  Score: {result.score:.4f}")

            if result.baseline_score:
                report_lines.append(f"  Baseline: {result.baseline_score:.4f}")
                report_lines.append(f"  Improvement: {result.improvement:.4f}")

            if result.details:
                report_lines.append("  Details:")
                for key, value in result.details.items():
                    report_lines.append(f"    {key}: {value:.4f}")

        return "\n".join(report_lines)
