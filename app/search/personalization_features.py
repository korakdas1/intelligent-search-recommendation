"""personalization-features-v1: user affinity on QUERY candidates only.

Does not modify rank-features-v1. Does not retrieve extra products.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from app.recommendations.content import user_content_profile
from app.recommendations.vectors import embeddings_for_products
from app.search.fusion import minmax_normalize
from app.search.personalization_constants import SIGNAL_CF, SIGNAL_CONTENT
from app.search.runtime import SemanticRuntime


def finite_or_zero(value: float) -> float:
    number = float(value)
    if not np.isfinite(number):
        return 0.0
    return number


def content_affinity_for_candidates(
    runtime: SemanticRuntime,
    history_product_ids: Sequence[str],
    candidate_ids: Sequence[str],
) -> tuple[bool, dict[str, float], list[str]]:
    """Cosine between content-rec-v1 mean profile and each search candidate.

    Returns (channel_available, raw_scores_for_candidates_with_vectors, kept_history_ids).
    Missing candidate embeddings are omitted; callers treat them as contribution 0.
    """

    profile, kept = user_content_profile(runtime, history_product_ids)
    if profile is None:
        return False, {}, []
    present_ids, matrix = embeddings_for_products(runtime, candidate_ids)
    scores: dict[str, float] = {}
    if present_ids:
        dots = np.asarray(matrix @ profile, dtype=np.float32).reshape(-1)
        for product_id, score in zip(present_ids, dots, strict=True):
            scores[str(product_id)] = finite_or_zero(float(score))
    return True, scores, kept


def cf_affinity_for_candidates(
    *,
    user_id: str,
    candidate_ids: Sequence[str],
    user_to_index: Mapping[str, int],
    product_to_index: Mapping[str, int],
    score_fn,
) -> tuple[bool, dict[str, float]]:
    """Dot-product CF affinity for search candidates. Does not retrieve extra items.

    ``score_fn(user_index, item_indices) -> sequence[float]``.
    User absent from the model → channel unavailable.
    Candidate absent from the model → omitted (contribution 0 later).
    """

    user_index = user_to_index.get(str(user_id))
    if user_index is None:
        return False, {}
    present_ids: list[str] = []
    item_indices: list[int] = []
    for product_id in candidate_ids:
        item_index = product_to_index.get(str(product_id))
        if item_index is None:
            continue
        present_ids.append(str(product_id))
        item_indices.append(int(item_index))
    if not present_ids:
        return True, {}
    raw = score_fn(int(user_index), item_indices)
    scores = {
        product_id: finite_or_zero(float(score))
        for product_id, score in zip(present_ids, raw, strict=True)
    }
    return True, scores


def brand_affinity_for_candidates(
    history_brands: Sequence[str | None],
    candidate_brands: Mapping[str, str | None],
    candidate_ids: Sequence[str],
) -> tuple[bool, dict[str, float]]:
    """Diagnostic brand share. Not part of personalized-search-v1 unless selected.

    brand_affinity(u,p) = unique history products of candidate brand / unique history
    products that have a brand. History products without brand are excluded from the
    denominator. Missing candidate brand → omitted (contribution 0).
    """

    counts: dict[str, int] = {}
    for brand in history_brands:
        if not brand:
            continue
        key = str(brand)
        counts[key] = counts.get(key, 0) + 1
    total = sum(counts.values())
    if total < 1:
        return False, {}
    scores: dict[str, float] = {}
    for product_id in candidate_ids:
        brand = candidate_brands.get(str(product_id))
        if not brand:
            continue
        scores[str(product_id)] = finite_or_zero(counts.get(str(brand), 0) / total)
    return True, scores


def minmax_channel(
    candidate_ids: Sequence[str],
    raw_scores: Mapping[str, float],
    *,
    channel_available: bool,
) -> tuple[list[float], list[int]]:
    """Per-query min-max over candidates that have the signal.

    Unavailable channel: all scores 0, availability flags 0.
    Available channel, missing candidate: score 0, availability 0 for that row.
    Equal present scores become 1.0 via ``minmax_normalize``.
    """

    n = len(candidate_ids)
    if not channel_available:
        return [0.0] * n, [0] * n
    present_ids = [product_id for product_id in candidate_ids if product_id in raw_scores]
    if not present_ids:
        return [0.0] * n, [0] * n
    present_norm = dict(
        zip(
            present_ids,
            minmax_normalize(
                [finite_or_zero(float(raw_scores[product_id])) for product_id in present_ids]
            ),
            strict=True,
        )
    )
    values: list[float] = []
    flags: list[int] = []
    for product_id in candidate_ids:
        if product_id in present_norm:
            values.append(finite_or_zero(present_norm[product_id]))
            flags.append(1)
        else:
            values.append(0.0)
            flags.append(0)
    return values, flags


def query_relevance_scores(baseline_scores: Sequence[float], baseline_ranks: Sequence[int]) -> tuple[list[float], bool]:
    """Normalize base search scores. Degenerate equal scores use rank-based Q.

    Rank-based fallback: Q_i = (n - rank_i + 1) / n so the last candidate is
    not 0, but preference cannot become an unrestricted ranking.
    """

    scores = [finite_or_zero(float(score)) for score in baseline_scores]
    ranks = [int(rank) for rank in baseline_ranks]
    if not scores:
        return [], False
    if len(scores) != len(ranks):
        raise ValueError("baseline scores and ranks must have the same length")
    low = min(scores)
    high = max(scores)
    if high != low:
        return [finite_or_zero(value) for value in minmax_normalize(scores)], False
    n = len(ranks)
    if n == 1:
        return [1.0], True
    return [finite_or_zero((n - rank + 1) / n) for rank in ranks], True


def channel_name(content_available: bool, cf_available: bool) -> tuple[str, ...]:
    used: list[str] = []
    if content_available:
        used.append(SIGNAL_CONTENT)
    if cf_available:
        used.append(SIGNAL_CF)
    return tuple(used)
