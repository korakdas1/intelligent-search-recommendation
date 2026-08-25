"""Recommendation fusion: RRF and per-request min-max weighted scores.

Lives in ``app/recommendations/``. Search fusion classes are not reused.
Generic math helpers (RRF contribution, min-max) may be imported.

Raw cosine, CF dot product, and interaction counts are never added together.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.recommendations.hybrid_constants import (
    CANDIDATE_K_MAX,
    CANDIDATE_K_MIN,
    CANDIDATE_K_MULTIPLIER,
    CHANNEL_CF,
    CHANNEL_CONTENT,
    CHANNEL_POPULARITY,
    DEFAULT_RRF_K,
    HYBRID_CHANNELS,
    WEIGHT_SUM_TOLERANCE,
)
from app.recommendations.hybrid_types import ChannelRanking, ChannelWeights, HybridCandidate
from app.recommendations.ranking import ScoredItem, stable_rank
from app.search.fusion import minmax_normalize, rrf_contribution


def resolve_hybrid_candidate_k(
    top_k: int,
    *,
    configured: int | None = None,
    minimum: int = CANDIDATE_K_MIN,
    maximum: int = CANDIDATE_K_MAX,
    multiplier: int = CANDIDATE_K_MULTIPLIER,
) -> int:
    """Internal pool depth: ``min(max, max(min, top_k * multiplier))``, at least ``top_k``."""

    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    if maximum < 1:
        raise ValueError("candidate_k maximum must be at least 1")
    if configured is not None:
        if configured < 1:
            raise ValueError("candidate_k must be at least 1")
        return min(maximum, max(top_k, configured))
    formula = max(minimum, top_k * multiplier)
    return min(maximum, max(top_k, formula))


def validate_channel_weights(
    content: float,
    cf: float,
    popularity: float,
    *,
    tolerance: float = WEIGHT_SUM_TOLERANCE,
) -> ChannelWeights:
    values = (float(content), float(cf), float(popularity))
    if any(value < 0.0 for value in values):
        raise ValueError("channel weights must be >= 0")
    total = sum(values)
    if abs(total - 1.0) > tolerance:
        raise ValueError("channel weights must sum to 1")
    return ChannelWeights(content=float(content), cf=float(cf), popularity=float(popularity))


def renormalize_channel_weights(
    weights: ChannelWeights | Mapping[str, float],
    available: Sequence[str],
    *,
    tolerance: float = WEIGHT_SUM_TOLERANCE,
) -> dict[str, float]:
    """Renormalize configured betas over whole available channels only.

    A candidate missing from an available channel is not a reason to renormalize.
    """

    configured = weights.as_dict() if isinstance(weights, ChannelWeights) else dict(weights)
    allowed = {str(name) for name in available if str(name) in configured}
    subset = {name: float(configured[name]) for name in HYBRID_CHANNELS if name in allowed}
    total = sum(subset.values())
    if total <= 0.0:
        raise ValueError("no positive weight remains among available channels")
    scaled = {name: value / total for name, value in subset.items()}
    if abs(sum(scaled.values()) - 1.0) > tolerance:
        raise ValueError("renormalized weights must sum to 1")
    return scaled


def union_recommendation_channels(
    content: ChannelRanking | Sequence[ScoredItem] | None,
    cf: ChannelRanking | Sequence[ScoredItem] | None,
    popularity: ChannelRanking | Sequence[ScoredItem] | None,
) -> dict[str, HybridCandidate]:
    """Union by product_id. Intersection is forbidden. One row per product."""

    merged: dict[str, HybridCandidate] = {}
    _merge_channel(merged, CHANNEL_CONTENT, content)
    _merge_channel(merged, CHANNEL_CF, cf)
    _merge_channel(merged, CHANNEL_POPULARITY, popularity)
    return merged


def fuse_recommendation_rrf(
    rows: Mapping[str, HybridCandidate],
    *,
    k0: int = DEFAULT_RRF_K,
) -> list[HybridCandidate]:
    """Rank fusion. A missing channel/candidate contributes 0. No n-channel rescale."""

    fused: list[HybridCandidate] = []
    for row in rows.values():
        score = (
            rrf_contribution(row.content_rank, k0=k0)
            + rrf_contribution(row.cf_rank, k0=k0)
            + rrf_contribution(row.popularity_rank, k0=k0)
        )
        fused.append(_with_score(row, score))
    return stable_hybrid_order(fused)


def fuse_recommendation_weighted(
    rows: Mapping[str, HybridCandidate],
    weights: ChannelWeights | Mapping[str, float],
    *,
    available_channels: Sequence[str],
) -> tuple[list[HybridCandidate], dict[str, float]]:
    """Per-request min-max per available channel, then weighted sum.

    Missing candidate in an available channel contributes 0.
    An unavailable channel is dropped and remaining weights are renormalized.
    Equal channel scores become 1.0. Outputs are finite.
    """

    effective = renormalize_channel_weights(weights, available_channels)
    content_norm = _channel_minmax(rows, CHANNEL_CONTENT) if CHANNEL_CONTENT in effective else {}
    cf_norm = _channel_minmax(rows, CHANNEL_CF) if CHANNEL_CF in effective else {}
    pop_norm = _channel_minmax(rows, CHANNEL_POPULARITY) if CHANNEL_POPULARITY in effective else {}
    fused: list[HybridCandidate] = []
    for row in rows.values():
        score = (
            effective.get(CHANNEL_CONTENT, 0.0) * content_norm.get(row.product_id, 0.0)
            + effective.get(CHANNEL_CF, 0.0) * cf_norm.get(row.product_id, 0.0)
            + effective.get(CHANNEL_POPULARITY, 0.0) * pop_norm.get(row.product_id, 0.0)
        )
        if score != score or score in {float("inf"), float("-inf")}:
            raise ValueError("hybrid weighted score must be finite")
        fused.append(_with_score(row, float(score)))
    return stable_hybrid_order(fused), effective


def stable_hybrid_order(rows: Sequence[HybridCandidate]) -> list[HybridCandidate]:
    return sorted(rows, key=lambda row: (-row.fused_score, row.product_id))


def _merge_channel(
    merged: dict[str, HybridCandidate],
    name: str,
    ranking: ChannelRanking | Sequence[ScoredItem] | None,
) -> None:
    if ranking is None:
        return
    if isinstance(ranking, ChannelRanking):
        if not ranking.available:
            return
        items = ranking.items
    else:
        items = tuple(ranking)
    ranked_items = stable_rank(list(items), top_k=max(len(items), 1)) if items else []
    for rank, item in enumerate(ranked_items, start=1):
        product_id = str(item.product_id)
        row = merged.get(product_id)
        if row is None:
            row = HybridCandidate(product_id=product_id)
            merged[product_id] = row
        if name == CHANNEL_CONTENT:
            row.content_present = True
            row.content_rank = rank
            row.content_raw_score = float(item.score)
        elif name == CHANNEL_CF:
            row.cf_present = True
            row.cf_rank = rank
            row.cf_raw_score = float(item.score)
        elif name == CHANNEL_POPULARITY:
            row.popularity_present = True
            row.popularity_rank = rank
            row.popularity_raw_score = float(item.score)
        else:
            raise ValueError(f"unknown recommendation channel {name!r}")
        if name not in row.sources:
            row.sources = (*row.sources, name)


def _channel_minmax(rows: Mapping[str, HybridCandidate], channel: str) -> dict[str, float]:
    ids: list[str] = []
    scores: list[float] = []
    for row in rows.values():
        raw = _raw_score(row, channel)
        if raw is None:
            continue
        ids.append(row.product_id)
        scores.append(float(raw))
    if not ids:
        return {}
    normalized = minmax_normalize(scores)
    return dict(zip(ids, (float(value) for value in normalized), strict=True))


def _raw_score(row: HybridCandidate, channel: str) -> float | None:
    if channel == CHANNEL_CONTENT:
        return row.content_raw_score if row.content_present else None
    if channel == CHANNEL_CF:
        return row.cf_raw_score if row.cf_present else None
    if channel == CHANNEL_POPULARITY:
        return row.popularity_raw_score if row.popularity_present else None
    raise ValueError(f"unknown recommendation channel {channel!r}")


def _with_score(row: HybridCandidate, score: float) -> HybridCandidate:
    return HybridCandidate(
        product_id=row.product_id,
        fused_score=float(score),
        content_present=row.content_present,
        content_rank=row.content_rank,
        content_raw_score=row.content_raw_score,
        cf_present=row.cf_present,
        cf_rank=row.cf_rank,
        cf_raw_score=row.cf_raw_score,
        popularity_present=row.popularity_present,
        popularity_rank=row.popularity_rank,
        popularity_raw_score=row.popularity_raw_score,
        sources=row.sources,
    )
