"""Hybrid fusion: RRF and min-max weighted scores.

Raw ``ts_rank_cd`` and cosine are never added together.
"""

from __future__ import annotations

from app.search.candidates import FusedCandidate, stable_fusion_order

DEFAULT_RRF_K = 60
DEFAULT_KEYWORD_WEIGHT = 0.5


def rrf_contribution(rank: int | None, *, k0: int = DEFAULT_RRF_K) -> float:
    if rank is None:
        return 0.0
    if rank < 1:
        raise ValueError("rank must be 1-based")
    if k0 < 0:
        raise ValueError("RRF k must be >= 0")
    return 1.0 / (k0 + rank)


def fuse_rrf(
    rows: dict[str, FusedCandidate],
    *,
    k0: int = DEFAULT_RRF_K,
) -> list[FusedCandidate]:
    fused: list[FusedCandidate] = []
    for row in rows.values():
        score = rrf_contribution(row.keyword_rank, k0=k0) + rrf_contribution(
            row.semantic_rank, k0=k0
        )
        fused.append(
            FusedCandidate(
                product_id=row.product_id,
                fusion_score=score,
                keyword_rank=row.keyword_rank,
                keyword_score=row.keyword_score,
                semantic_rank=row.semantic_rank,
                semantic_score=row.semantic_score,
                sources=row.sources,
            )
        )
    return stable_fusion_order(fused)


def minmax_normalize(scores: list[float]) -> list[float]:
    """Per-query min-max. Equal scores become 1.0. Empty list stays empty."""

    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [1.0 for _ in scores]
    span = high - low
    return [(value - low) / span for value in scores]


def validate_keyword_weight(alpha: float) -> float:
    if alpha < 0.0 or alpha > 1.0:
        raise ValueError("keyword weight alpha must be between 0 and 1 inclusive")
    return float(alpha)


def fuse_weighted(
    rows: dict[str, FusedCandidate],
    *,
    alpha: float = DEFAULT_KEYWORD_WEIGHT,
) -> list[FusedCandidate]:
    """Weighted sum of per-list min-max scores. Missing side contributes 0."""

    alpha = validate_keyword_weight(alpha)
    keyword_ids = [row.product_id for row in rows.values() if row.keyword_score is not None]
    semantic_ids = [row.product_id for row in rows.values() if row.semantic_score is not None]
    keyword_norm = dict(
        zip(
            keyword_ids,
            minmax_normalize(
                [rows[pid].keyword_score for pid in keyword_ids if rows[pid].keyword_score is not None]
            ),
            strict=True,
        )
    )
    semantic_norm = dict(
        zip(
            semantic_ids,
            minmax_normalize(
                [
                    rows[pid].semantic_score
                    for pid in semantic_ids
                    if rows[pid].semantic_score is not None
                ]
            ),
            strict=True,
        )
    )
    fused: list[FusedCandidate] = []
    for row in rows.values():
        k_norm = keyword_norm.get(row.product_id, 0.0)
        s_norm = semantic_norm.get(row.product_id, 0.0)
        score = alpha * k_norm + (1.0 - alpha) * s_norm
        fused.append(
            FusedCandidate(
                product_id=row.product_id,
                fusion_score=score,
                keyword_rank=row.keyword_rank,
                keyword_score=row.keyword_score,
                semantic_rank=row.semantic_rank,
                semantic_score=row.semantic_score,
                sources=row.sources,
            )
        )
    return stable_fusion_order(fused)
