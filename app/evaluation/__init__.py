"""Evaluation package."""

from app.evaluation.benchmark_dataset import (
    BENCHMARK_GENERATION_CASES,
    BENCHMARK_RETRIEVAL_CASES,
    BENCHMARK_STYLE_CASES,
)
from app.evaluation.metrics import (
    citation_completeness,
    faithfulness_score,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from app.evaluation.runner import BenchmarkEvaluationRunner

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "faithfulness_score",
    "citation_completeness",
    "BenchmarkEvaluationRunner",
    "BENCHMARK_RETRIEVAL_CASES",
    "BENCHMARK_GENERATION_CASES",
    "BENCHMARK_STYLE_CASES",
]
