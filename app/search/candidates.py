"""Shared retrieval-candidate objects and union helpers.

These are retrieval metadata only. They are not ranking features.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_CANDIDATE_K_MIN = 100
DEFAULT_CANDIDATE_K_MAX = 500
DEFAULT_CANDIDATE_K_MULTIPLIER = 5


@dataclass(frozen=True, slots=True)
class RetrievalCandidate:
    product_id: str
    rank: int
    score: float
    source: str


@dataclass
class FusedCandidate:
    product_id: str
    fusion_score: float
    keyword_rank: int | None = None
    keyword_score: float | None = None
    semantic_rank: int | None = None
    semantic_score: float | None = None
    sources: tuple[str, ...] = ()


def resolve_candidate_k(
    top_k: int,
    *,
    configured: int | None = None,
    minimum: int = DEFAULT_CANDIDATE_K_MIN,
    maximum: int = DEFAULT_CANDIDATE_K_MAX,
    multiplier: int = DEFAULT_CANDIDATE_K_MULTIPLIER,
) -> int:
    """Return an internal candidate depth: >= top_k and <= maximum."""

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


def union_candidates(
    keyword: list[RetrievalCandidate],
    semantic: list[RetrievalCandidate],
) -> dict[str, FusedCandidate]:
    """Merge lists by product_id. One row per product; both ranks kept when present."""

    merged: dict[str, FusedCandidate] = {}
    for candidate in keyword:
        row = merged.get(candidate.product_id)
        if row is None:
            merged[candidate.product_id] = FusedCandidate(
                product_id=candidate.product_id,
                fusion_score=0.0,
                keyword_rank=candidate.rank,
                keyword_score=candidate.score,
                sources=("keyword",),
            )
        else:
            row.keyword_rank = candidate.rank
            row.keyword_score = candidate.score
            if "keyword" not in row.sources:
                row.sources = (*row.sources, "keyword")
    for candidate in semantic:
        row = merged.get(candidate.product_id)
        if row is None:
            merged[candidate.product_id] = FusedCandidate(
                product_id=candidate.product_id,
                fusion_score=0.0,
                semantic_rank=candidate.rank,
                semantic_score=candidate.score,
                sources=("semantic",),
            )
        else:
            row.semantic_rank = candidate.rank
            row.semantic_score = candidate.score
            if "semantic" not in row.sources:
                row.sources = (*row.sources, "semantic")
    return merged


def stable_fusion_order(rows: list[FusedCandidate]) -> list[FusedCandidate]:
    return sorted(rows, key=lambda row: (-row.fusion_score, row.product_id))
