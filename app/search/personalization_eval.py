"""Offline helpers for synthetic personalized-search evaluation. Not imported by the API."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from app.ranking.labels import make_title_query
from app.ranking.metrics import macro_average, metric_bundle
from app.ranking.query_ids import make_personalized_query_id
from app.recommendations.evaluation import history_size_bin
from app.recommendations.hybrid_offline import mean_profile
from app.search.personalization_constants import QUERY_GENERATOR_VERSION, QUERY_SOURCE
from app.search.personalization_rerank import rerank_candidates, validate_signal_weights
from app.search.personalization_types import BaselineCandidate, SignalWeights


@dataclass(frozen=True, slots=True)
class PreparedQuery:
    user_id: str
    target_product_id: str
    query_id: str
    query: str
    history_product_ids: tuple[str, ...]
    history_size: int
    usable: bool
    skip_reason: str | None = None


def signal_grid() -> list[tuple[str, SignalWeights]]:
    return [
        ("content_only", validate_signal_weights(1.0, 0.0)),
        ("cf_only", validate_signal_weights(0.0, 1.0)),
        ("c70_cf30", validate_signal_weights(0.70, 0.30)),
        ("c80_cf20", validate_signal_weights(0.80, 0.20)),
        ("c60_cf40", validate_signal_weights(0.60, 0.40)),
    ]


def gamma_grid() -> tuple[float, ...]:
    return (0.05, 0.10, 0.20, 0.30)


def prepare_query(
    *,
    user_id: str,
    target_product_id: str,
    title: str | None,
    history_product_ids: Sequence[str],
) -> PreparedQuery:
    if target_product_id in set(history_product_ids):
        raise AssertionError("held-out target leaked into personalization history")
    if not title or not str(title).strip():
        return PreparedQuery(
            user_id=user_id,
            target_product_id=target_product_id,
            query_id="",
            query="",
            history_product_ids=tuple(history_product_ids),
            history_size=len(history_product_ids),
            usable=False,
            skip_reason="missing_title",
        )
    generated = make_title_query(target_product_id, title)
    if generated is None or not generated.query.strip():
        return PreparedQuery(
            user_id=user_id,
            target_product_id=target_product_id,
            query_id="",
            query="",
            history_product_ids=tuple(history_product_ids),
            history_size=len(history_product_ids),
            usable=False,
            skip_reason="blank_synthetic_query",
        )
    query_id = make_personalized_query_id(
        user_id,
        target_product_id,
        QUERY_GENERATOR_VERSION,
        generated.query,
    )
    return PreparedQuery(
        user_id=user_id,
        target_product_id=target_product_id,
        query_id=query_id,
        query=generated.query,
        history_product_ids=tuple(history_product_ids),
        history_size=len(history_product_ids),
        usable=True,
        skip_reason=None,
    )


def content_scores_from_matrix(
    *,
    history_ids: Sequence[str],
    candidate_ids: Sequence[str],
    embeddings: np.ndarray,
    id_to_row: Mapping[str, int],
) -> dict[str, float] | None:
    history_rows = np.array(
        [id_to_row[product_id] for product_id in history_ids if product_id in id_to_row],
        dtype=np.int64,
    )
    profile = mean_profile(embeddings, history_rows)
    if profile is None:
        return None
    scores: dict[str, float] = {}
    for product_id in candidate_ids:
        row = id_to_row.get(str(product_id))
        if row is None:
            continue
        scores[str(product_id)] = float(np.dot(embeddings[int(row)], profile))
    return scores


def cf_scores_from_model(
    *,
    user_id: str,
    candidate_ids: Sequence[str],
    user_to_index: Mapping[str, int],
    product_to_index: Mapping[str, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
) -> dict[str, float] | None:
    user_index = user_to_index.get(str(user_id))
    if user_index is None:
        return None
    user_vec = user_factors[int(user_index)]
    scores: dict[str, float] = {}
    for product_id in candidate_ids:
        item_index = product_to_index.get(str(product_id))
        if item_index is None:
            continue
        scores[str(product_id)] = float(np.dot(user_vec, item_factors[int(item_index)]))
    return scores


def movement_stats(
    baseline_ids: Sequence[str],
    personalized_ids: Sequence[str],
    *,
    target_id: str | None = None,
) -> dict[str, float]:
    base_rank = {product_id: index for index, product_id in enumerate(baseline_ids, start=1)}
    pers_rank = {product_id: index for index, product_id in enumerate(personalized_ids, start=1)}
    movements = [abs(pers_rank[product_id] - base_rank[product_id]) for product_id in baseline_ids]
    promotions = [
        base_rank[product_id] - pers_rank[product_id]
        for product_id in baseline_ids
        if pers_rank[product_id] < base_rank[product_id]
    ]
    top10_base = set(baseline_ids[:10])
    top10_pers = set(personalized_ids[:10])
    overlap = len(top10_base & top10_pers)
    union = len(top10_base | top10_pers) or 1
    target_base = base_rank.get(str(target_id)) if target_id else None
    target_pers = pers_rank.get(str(target_id)) if target_id else None
    improved = 0.0
    worsened = 0.0
    if target_base is not None and target_pers is not None:
        improved = 1.0 if target_pers < target_base else 0.0
        worsened = 1.0 if target_pers > target_base else 0.0
    ordered = sorted(movements)
    n = len(ordered)
    median = float(ordered[n // 2]) if n else 0.0
    p95 = float(ordered[min(n - 1, int(round(0.95 * (n - 1))))]) if n else 0.0
    return {
        "top1_changed": 1.0 if baseline_ids[:1] != personalized_ids[:1] else 0.0,
        "mean_abs_rank_movement": float(sum(movements) / n) if n else 0.0,
        "median_abs_rank_movement": median,
        "p95_abs_rank_movement": p95,
        "max_promotion": float(max(promotions) if promotions else 0.0),
        "top10_overlap": float(overlap),
        "top10_jaccard": overlap / union,
        "target_improved": improved,
        "target_worsened": worsened,
        "target_baseline_rank": float(target_base or 0),
        "target_personalized_rank": float(target_pers or 0),
    }


def evaluate_ranking_pair(
    baseline_ids: Sequence[str],
    personalized_ids: Sequence[str],
    target_id: str,
    *,
    covered: bool,
) -> dict[str, Any]:
    relevant = [target_id]
    baseline_metrics = metric_bundle(baseline_ids, relevant)
    personalized_metrics = metric_bundle(personalized_ids, relevant)
    stats = movement_stats(baseline_ids, personalized_ids, target_id=target_id)
    return {
        "covered": covered,
        "baseline": baseline_metrics,
        "personalized": personalized_metrics,
        "movement": stats,
        "history_bin": None,
    }


def summarize_eval_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    baseline = macro_average([row["baseline"] for row in rows])
    personalized = macro_average([row["personalized"] for row in rows])
    movement = macro_average([row["movement"] for row in rows])
    covered_n = sum(1 for row in rows if row["covered"])
    return {
        "n": len(rows),
        "candidate_coverage": covered_n / len(rows),
        "covered_n": covered_n,
        "baseline": baseline,
        "personalized": personalized,
        "movement": movement,
    }


def segment_by_history(rows: Sequence[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get("history_bin") or "unknown")].append(row)
    return {name: summarize_eval_rows(items) for name, items in sorted(groups.items())}


def attach_history_bin(row: dict[str, Any], history_size: int) -> dict[str, Any]:
    payload = dict(row)
    payload["history_bin"] = history_size_bin(history_size)
    payload["history_size"] = history_size
    return payload


def rerank_with_config(
    baseline: Sequence[BaselineCandidate],
    *,
    content_raw: Mapping[str, float] | None,
    cf_raw: Mapping[str, float] | None,
    weights: SignalWeights,
    gamma: float,
) -> list[str]:
    outcome = rerank_candidates(
        baseline,
        content_raw=content_raw,
        cf_raw=cf_raw,
        weights=weights,
        gamma=gamma,
    )
    return [row.product_id for row in outcome.candidates]


def select_inner_winner(comparison: Mapping[str, Mapping[str, float]]) -> str:
    """Highest inner NDCG@10; ties prefer smaller gamma then content-heavier configs.

    External test is never consulted.
    """

    def key(name: str) -> tuple[float, float, float, float, float]:
        row = comparison[name]
        gamma = float(row.get("gamma", 1.0))
        content_w = float(row.get("content_weight", 0.0))
        return (
            float(row.get("ndcg@10", 0.0)),
            float(row.get("recall@10", 0.0)),
            float(row.get("mrr", 0.0)),
            -gamma,
            content_w,
        )

    return sorted(comparison, key=key, reverse=True)[0]


def query_source_name() -> str:
    return QUERY_SOURCE


def hybrid_baseline_candidates(
    session,
    *,
    query: str,
    fusion_method: str,
    candidate_k: int,
    top_k: int = 20,
):
    from app.search.hybrid import HYBRID_SOURCE, collect_hybrid_fused

    collected = collect_hybrid_fused(
        session,
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,
        candidate_k=candidate_k,
    )
    baseline = [
        BaselineCandidate(
            product_id=row.product_id,
            baseline_score=float(row.fusion_score),
            baseline_rank=index,
            source=HYBRID_SOURCE,
        )
        for index, row in enumerate(collected.fused, start=1)
    ]
    return baseline, collected
