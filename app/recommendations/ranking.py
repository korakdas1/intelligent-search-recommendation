"""Deterministic ranking helpers shared by content and popularity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredItem:
    product_id: str
    score: float


def stable_rank(items: list[ScoredItem], *, top_k: int) -> list[ScoredItem]:
    """Sort score DESC, product_id ASC. Does not rely on FAISS tie order."""

    ordered = sorted(items, key=lambda item: (-item.score, item.product_id))
    unique: list[ScoredItem] = []
    seen: set[str] = set()
    for item in ordered:
        if item.product_id in seen:
            continue
        seen.add(item.product_id)
        unique.append(item)
        if len(unique) >= top_k:
            break
    return unique


def rank_of_item(
    hidden_id: str,
    hidden_score: float,
    *,
    other_ids: list[str],
    other_scores: list[float],
) -> int | None:
    """1-based rank among ``other`` plus the hidden item, score DESC, id ASC."""

    better = 0
    for product_id, score in zip(other_ids, other_scores, strict=True):
        if product_id == hidden_id:
            continue
        if score > hidden_score or (score == hidden_score and product_id < hidden_id):
            better += 1
    return better + 1
