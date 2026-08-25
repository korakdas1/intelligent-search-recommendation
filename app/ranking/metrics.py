"""Ranking metrics. Canonical implementation lives in app.evaluation.metrics."""

from app.evaluation.metrics import (
    K_VALUES,
    METRIC_VERSION,
    macro_average,
    mean_reciprocal_rank,
    metric_bundle,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    unique_preserve_order,
)

__all__ = [
    "K_VALUES",
    "METRIC_VERSION",
    "macro_average",
    "mean_reciprocal_rank",
    "metric_bundle",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "unique_preserve_order",
]
