"""Internal recommendation candidates without metadata formatting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from sqlalchemy.orm import Session

from app.recommendations.content import content_items_for_profile, user_content_profile
from app.recommendations.hybrid_constants import CHANNEL_CF, CHANNEL_CONTENT, CHANNEL_POPULARITY
from app.recommendations.hybrid_types import ChannelRanking
from app.recommendations.popularity import popular_products
from app.recommendations.ranking import ScoredItem, stable_rank
from app.search.runtime import SemanticRuntime


def content_candidates_for_profile(
    runtime: SemanticRuntime,
    seen_product_ids: Sequence[str],
    *,
    top_k: int,
) -> tuple[list[ScoredItem], list[str]]:
    """Return content-rec-v1 neighbors and the usable history ids. Empty if no profile."""

    profile, usable = user_content_profile(runtime, seen_product_ids)
    if profile is None or not usable:
        return [], []
    exclude = set(str(item) for item in seen_product_ids)
    return content_items_for_profile(runtime, profile, exclude=exclude, top_k=top_k), usable


def cf_candidates_for_user(
    runtime: object,
    user_id: str,
    *,
    seen_product_ids: Sequence[str],
    top_k: int,
) -> list[ScoredItem] | None:
    """BPR-MF candidate list. ``None`` if the user has no CF embedding."""

    user_index = runtime.user_to_index.get(user_id)
    if user_index is None:
        return None
    seen = {str(item) for item in seen_product_ids}
    with torch.no_grad():
        index = torch.tensor([user_index], dtype=torch.long)
        scores = runtime.model.score_items(index).detach().cpu().numpy()[0]
    scored: list[ScoredItem] = []
    for product_id, score in zip(runtime.product_ids, scores, strict=True):
        if product_id in seen:
            continue
        scored.append(ScoredItem(product_id=str(product_id), score=float(score)))
    if not scored:
        return []
    return stable_rank(scored, top_k=top_k)


def popularity_candidates(
    session: Session,
    *,
    top_k: int,
    seen_product_ids: Sequence[str],
    before=None,
    interaction_ids: Sequence[int] | None = None,
    counts: dict[str, int] | None = None,
) -> list[ScoredItem]:
    """Train-safe when ``interaction_ids`` or ``counts`` are provided; else current DB counts."""

    exclude = {str(item) for item in seen_product_ids}
    if counts is not None:
        from app.recommendations.popularity import rank_popularity_counts

        return rank_popularity_counts(counts, exclude=exclude, top_k=top_k)
    return popular_products(
        session,
        top_k=top_k,
        before=before,
        interaction_ids=interaction_ids,
        exclude=exclude,
    )


def ranking_from_items(name: str, items: Sequence[ScoredItem] | None, *, available: bool) -> ChannelRanking:
    if not available or items is None:
        return ChannelRanking(name=name, available=False, items=())
    return ChannelRanking(name=name, available=True, items=tuple(items))


def channel_rankings(
    *,
    content_items: Sequence[ScoredItem] | None,
    content_available: bool,
    cf_items: Sequence[ScoredItem] | None,
    cf_available: bool,
    popularity_items: Sequence[ScoredItem] | None,
    popularity_available: bool,
) -> tuple[ChannelRanking, ChannelRanking, ChannelRanking]:
    return (
        ranking_from_items(CHANNEL_CONTENT, content_items, available=content_available),
        ranking_from_items(CHANNEL_CF, cf_items, available=cf_available),
        ranking_from_items(CHANNEL_POPULARITY, popularity_items, available=popularity_available),
    )


def topk_ids_from_scores(
    product_ids: Sequence[str] | np.ndarray,
    scores: np.ndarray,
    *,
    exclude: set[str],
    top_k: int,
    exclude_indices: np.ndarray | None = None,
) -> list[ScoredItem]:
    """Rank finite, non-excluded scores DESC, id ASC, including cutoff ties.

    Used by offline evaluation. Nonempty ``exclude_indices`` takes precedence
    over ``exclude``; returned scores retain their float64 working values.
    """

    if top_k < 1 or len(product_ids) == 0:
        return []
    ids = np.asarray(product_ids)
    working = np.asarray(scores, dtype=np.float64)
    if working.shape[0] != ids.shape[0]:
        raise ValueError("product_ids and scores length mismatch")
    masked = np.array(working, dtype=np.float64, copy=True)
    if exclude_indices is not None and exclude_indices.size:
        masked[exclude_indices] = -np.inf
    elif exclude:
        keep = np.array([str(product_id) not in exclude for product_id in ids], dtype=bool)
        masked = np.where(keep, masked, -np.inf)
    valid_indices = np.flatnonzero(np.isfinite(masked))
    take = min(int(top_k), len(valid_indices))
    if take < 1:
        return []
    pool = valid_indices
    if take < len(valid_indices):
        valid_scores = masked[valid_indices]
        cutoff = np.partition(valid_scores, len(valid_scores) - take)[len(valid_scores) - take]
        better = valid_indices[valid_scores > cutoff]
        tied = valid_indices[valid_scores == cutoff]
        # Select boundary ties by id before truncating; partition alone picks
        # an arbitrary subset. Only the tie group and selected pool are sorted.
        tied = tied[np.argsort(ids[tied])[: take - len(better)]]
        pool = np.concatenate((better, tied))
    ordered = pool[np.lexsort((ids[pool], -masked[pool]))]
    return [ScoredItem(product_id=str(ids[index]), score=float(working[index])) for index in ordered]
