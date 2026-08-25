"""Offline hybrid ranking helpers. Evaluation-only; does not change serving."""

from __future__ import annotations

from collections.abc import Sequence

from app.recommendations.hybrid_constants import CHANNEL_CF, CHANNEL_CONTENT, CHANNEL_POPULARITY
from app.recommendations.hybrid_fusion import (
    fuse_recommendation_rrf,
    fuse_recommendation_weighted,
    union_recommendation_channels,
    validate_channel_weights,
)
from app.recommendations.hybrid_types import ChannelWeights, HybridCandidate
from app.recommendations.ranking import ScoredItem

WEIGHT_GRID: tuple[tuple[str, float, float, float], ...] = (
    ("equal", 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0),
    ("c50_cf30_p20", 0.50, 0.30, 0.20),
    ("c60_cf30_p10", 0.60, 0.30, 0.10),
    ("c60_cf20_p20", 0.60, 0.20, 0.20),
    ("c70_cf20_p10", 0.70, 0.20, 0.10),
    ("c50_cf40_p10", 0.50, 0.40, 0.10),
)


def rank_hidden_in_fused(hidden_id: str, ranked: Sequence[HybridCandidate]) -> int | None:
    target = str(hidden_id)
    for index, row in enumerate(ranked, start=1):
        if row.product_id == target:
            return index
    return None


def fuse_channel_lists(
    content: Sequence[ScoredItem] | None,
    cf: Sequence[ScoredItem] | None,
    popularity: Sequence[ScoredItem] | None,
    *,
    fusion_method: str,
    rrf_k: int,
    weights: ChannelWeights | None = None,
    available_channels: Sequence[str] | None = None,
) -> list[HybridCandidate]:
    union = union_recommendation_channels(content, cf, popularity)
    used = list(available_channels or _infer_available(content, cf, popularity))
    if fusion_method == "weighted":
        if weights is None:
            raise ValueError("weighted fusion requires channel weights")
        ranked, _effective = fuse_recommendation_weighted(union, weights, available_channels=used)
        return ranked
    return fuse_recommendation_rrf(union, k0=rrf_k)


def coverage_flags(
    hidden_id: str,
    content: Sequence[ScoredItem] | None,
    cf: Sequence[ScoredItem] | None,
    popularity: Sequence[ScoredItem] | None,
) -> dict[str, bool]:
    hidden = str(hidden_id)
    content_ids = {item.product_id for item in content or ()}
    cf_ids = {item.product_id for item in cf or ()}
    pop_ids = {item.product_id for item in popularity or ()}
    return {
        "content_candidate": hidden in content_ids,
        "cf_candidate": hidden in cf_ids,
        "popularity_candidate": hidden in pop_ids,
        "union_candidate": hidden in (content_ids | cf_ids | pop_ids),
    }


def grid_weights() -> list[tuple[str, ChannelWeights]]:
    return [(name, validate_channel_weights(c, cf, p)) for name, c, cf, p in WEIGHT_GRID]


def _infer_available(
    content: Sequence[ScoredItem] | None,
    cf: Sequence[ScoredItem] | None,
    popularity: Sequence[ScoredItem] | None,
) -> list[str]:
    used: list[str] = []
    if content is not None:
        used.append(CHANNEL_CONTENT)
    if cf is not None:
        used.append(CHANNEL_CF)
    if popularity is not None:
        used.append(CHANNEL_POPULARITY)
    return used
