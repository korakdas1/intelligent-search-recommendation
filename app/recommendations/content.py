"""Item-item and user-profile content recommendations (content-rec-v1).

Uses stored MiniLM product vectors and IndexFlatIP. No text encoder.
No collaborative filtering. No RankNet.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.recommendations.constants import SEARCH_MARGIN
from app.recommendations.ranking import ScoredItem, stable_rank
from app.recommendations.vectors import embeddings_for_products, mean_profile
from app.search.faiss_index import search_index
from app.search.runtime import SemanticRuntime


def similar_items(
    runtime: SemanticRuntime,
    product_id: str,
    *,
    top_k: int,
) -> list[ScoredItem]:
    """Nearest catalog neighbors excluding the source product."""

    query = runtime.embedding_for_product(product_id)
    if query is None:
        return []
    return neighbors_excluding(
        runtime,
        query,
        exclude={product_id},
        top_k=top_k,
    )


def user_content_profile(
    runtime: SemanticRuntime,
    product_ids: Sequence[str],
) -> tuple[np.ndarray | None, list[str]]:
    """Mean of unique historical embeddings. Duplicate IDs contribute once."""

    kept, matrix = embeddings_for_products(runtime, product_ids)
    return mean_profile(matrix), kept


def content_items_for_profile(
    runtime: SemanticRuntime,
    profile: np.ndarray,
    *,
    exclude: set[str],
    top_k: int,
) -> list[ScoredItem]:
    return neighbors_excluding(runtime, profile, exclude=exclude, top_k=top_k)


def neighbors_excluding(
    runtime: SemanticRuntime,
    query: np.ndarray,
    *,
    exclude: set[str],
    top_k: int,
) -> list[ScoredItem]:
    if top_k < 1 or runtime.ntotal < 1:
        return []
    needed = top_k + len(exclude) + SEARCH_MARGIN
    fetch_k = min(runtime.ntotal, max(needed, top_k + 1))
    while True:
        scores, indices = search_index(runtime.index, query, fetch_k)
        candidates: list[ScoredItem] = []
        for score, row in zip(scores[0], indices[0], strict=True):
            row_i = int(row)
            if row_i < 0:
                continue
            product_id = runtime.row_to_product_id(row_i)
            if product_id is None or product_id in exclude:
                continue
            candidates.append(ScoredItem(product_id=product_id, score=float(score)))
        ranked = stable_rank(candidates, top_k=top_k)
        if len(ranked) >= top_k or fetch_k >= runtime.ntotal:
            return ranked
        fetch_k = min(runtime.ntotal, fetch_k * 2)
