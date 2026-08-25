"""Shared ranking metrics. Formulas match docs/EVALUATION_PLAN.md.

Not human relevance. Duplicate ranked IDs keep the first occurrence.
Missing targets contribute 0 (including NDCG when IDCG is 0).
"""

from __future__ import annotations

from collections.abc import Sequence
from math import log2

from app.evaluation.constants import K_VALUES, METRIC_VERSION

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


def unique_preserve_order(ranked_ids: Sequence[str]) -> list[str]:
    """Keep the first occurrence of each product id. Later duplicates are dropped."""

    seen: set[str] = set()
    ordered: list[str] = []
    for item in ranked_ids:
        product_id = str(item)
        if product_id in seen:
            continue
        seen.add(product_id)
        ordered.append(product_id)
    return ordered


def _relevant_set(relevant_ids: Sequence[str]) -> set[str]:
    return {str(item) for item in relevant_ids}


def recall_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = _relevant_set(relevant_ids)
    if not relevant or k < 1:
        return 0.0
    ranked = unique_preserve_order(ranked_ids)[:k]
    hit = set(ranked) & relevant
    return len(hit) / len(relevant)


def precision_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    if k < 1:
        return 0.0
    relevant = _relevant_set(relevant_ids)
    ranked = unique_preserve_order(ranked_ids)[:k]
    hit = set(ranked) & relevant
    return len(hit) / k


def mean_reciprocal_rank(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> float:
    relevant = _relevant_set(relevant_ids)
    for rank, product_id in enumerate(unique_preserve_order(ranked_ids), start=1):
        if product_id in relevant:
            return 1.0 / rank
    return 0.0


def _dcg(gains: Sequence[float]) -> float:
    return sum(gain / log2(index + 1) for index, gain in enumerate(gains, start=1))


def ndcg_at_k(ranked_ids: Sequence[str], relevant_ids: Sequence[str], k: int) -> float:
    relevant = _relevant_set(relevant_ids)
    if k < 1:
        return 0.0
    ranked = unique_preserve_order(ranked_ids)[:k]
    gains = [1.0 if product_id in relevant else 0.0 for product_id in ranked]
    if len(gains) < k:
        gains.extend([0.0] * (k - len(gains)))
    ideal_hits = min(len(relevant), k)
    ideal = [1.0] * ideal_hits + [0.0] * (k - ideal_hits)
    ideal_dcg = _dcg(ideal)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg(gains) / ideal_dcg


def metric_bundle(ranked_ids: Sequence[str], relevant_ids: Sequence[str]) -> dict[str, float]:
    ranked = unique_preserve_order(ranked_ids)
    bundle = {"mrr": mean_reciprocal_rank(ranked, relevant_ids)}
    for k in K_VALUES:
        bundle[f"recall@{k}"] = recall_at_k(ranked, relevant_ids, k)
        bundle[f"precision@{k}"] = precision_at_k(ranked, relevant_ids, k)
        bundle[f"ndcg@{k}"] = ndcg_at_k(ranked, relevant_ids, k)
    return bundle


def macro_average(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: float(sum(row[key] for row in rows) / len(rows)) for key in keys}
