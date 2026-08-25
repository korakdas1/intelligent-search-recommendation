"""Shared ranking feature extractor. No labels, no ranker, no training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import log1p

import numpy as np

from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_INDEX, FEATURE_VERSION
from app.ranking.product_view import ProductView
from app.ranking.text_features import (
    all_query_tokens_present,
    exact_phrase_in,
    normalize_phrase,
    overlap_count,
    overlap_ratio,
    tokenize,
)
from app.search.candidates import FusedCandidate
from app.search.fusion import DEFAULT_KEYWORD_WEIGHT, DEFAULT_RRF_K, fuse_weighted, rrf_contribution


@dataclass(frozen=True, slots=True)
class FeatureBatch:
    product_ids: tuple[str, ...]
    values: np.ndarray
    feature_version: str = FEATURE_VERSION


def extract_features(
    query: str,
    candidates: Sequence[FusedCandidate],
    products: Mapping[str, ProductView],
    *,
    candidate_k: int,
    rrf_k: int = DEFAULT_RRF_K,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> FeatureBatch:
    """Return a (n, d) float32 matrix aligned with ``candidates`` order.

    Does not query the database. Does not embed labels.
    """

    if candidate_k < 1:
        raise ValueError("candidate_k must be at least 1")
    product_ids = [row.product_id for row in candidates]
    if len(product_ids) != len(set(product_ids)):
        raise ValueError("candidate product_ids must be unique")
    missing = [pid for pid in product_ids if pid not in products]
    if missing:
        raise KeyError(f"products missing for candidates: {missing[:8]}")

    weighted_map = {
        row.product_id: row.fusion_score
        for row in fuse_weighted({item.product_id: item for item in candidates}, alpha=keyword_weight)
    }

    matrix = np.zeros((len(candidates), FEATURE_COUNT), dtype=np.float32)
    for index, candidate in enumerate(candidates):
        product = products[candidate.product_id]
        matrix[index] = _row(
            query,
            candidate,
            product,
            candidate_k=candidate_k,
            rrf_k=rrf_k,
            weighted_score=float(weighted_map.get(candidate.product_id, 0.0)),
        )
    return FeatureBatch(product_ids=tuple(product_ids), values=matrix, feature_version=FEATURE_VERSION)


def _row(
    query: str,
    candidate: FusedCandidate,
    product: ProductView,
    *,
    candidate_k: int,
    rrf_k: int,
    weighted_score: float,
) -> np.ndarray:
    keyword_present = 1.0 if candidate.keyword_rank is not None else 0.0
    semantic_present = 1.0 if candidate.semantic_rank is not None else 0.0
    keyword_score = float(candidate.keyword_score) if candidate.keyword_score is not None else 0.0
    semantic_score = float(candidate.semantic_score) if candidate.semantic_score is not None else 0.0
    keyword_rr = (1.0 / candidate.keyword_rank) if candidate.keyword_rank else 0.0
    semantic_rr = (1.0 / candidate.semantic_rank) if candidate.semantic_rank else 0.0
    keyword_rank_fraction = (
        float(candidate.keyword_rank) / candidate_k if candidate.keyword_rank else 0.0
    )
    semantic_rank_fraction = (
        float(candidate.semantic_rank) / candidate_k if candidate.semantic_rank else 0.0
    )
    rrf_score = rrf_contribution(candidate.keyword_rank, k0=rrf_k) + rrf_contribution(
        candidate.semantic_rank, k0=rrf_k
    )

    brand = product.brand
    description = product.description
    price = product.price
    price_ok = price is not None and price > 0.0
    rating = product.average_rating

    values = np.zeros(FEATURE_COUNT, dtype=np.float32)
    values[FEATURE_INDEX["keyword_present"]] = keyword_present
    values[FEATURE_INDEX["semantic_present"]] = semantic_present
    values[FEATURE_INDEX["retrieval_source_count"]] = keyword_present + semantic_present
    values[FEATURE_INDEX["keyword_score"]] = keyword_score
    values[FEATURE_INDEX["semantic_score"]] = semantic_score
    values[FEATURE_INDEX["keyword_rr"]] = keyword_rr
    values[FEATURE_INDEX["semantic_rr"]] = semantic_rr
    values[FEATURE_INDEX["keyword_rank_fraction"]] = keyword_rank_fraction
    values[FEATURE_INDEX["semantic_rank_fraction"]] = semantic_rank_fraction
    values[FEATURE_INDEX["rrf_score"]] = rrf_score
    values[FEATURE_INDEX["weighted_score"]] = weighted_score
    values[FEATURE_INDEX["title_exact_phrase"]] = exact_phrase_in(query, product.title)
    values[FEATURE_INDEX["title_all_query_tokens"]] = all_query_tokens_present(query, product.title)
    values[FEATURE_INDEX["title_overlap_count"]] = overlap_count(query, product.title)
    values[FEATURE_INDEX["title_overlap_ratio"]] = overlap_ratio(query, product.title)
    values[FEATURE_INDEX["brand_present"]] = 1.0 if brand else 0.0
    values[FEATURE_INDEX["brand_exact"]] = (
        1.0 if brand and normalize_phrase(brand) in normalize_phrase(query) else 0.0
    )
    values[FEATURE_INDEX["brand_overlap"]] = overlap_ratio(query, brand)
    values[FEATURE_INDEX["description_missing"]] = 0.0 if description else 1.0
    values[FEATURE_INDEX["title_token_count"]] = float(len(tokenize(product.title)))
    values[FEATURE_INDEX["description_token_count"]] = float(len(tokenize(description)))
    values[FEATURE_INDEX["average_rating"]] = float(rating) if rating is not None else 0.0
    values[FEATURE_INDEX["log_rating_count"]] = (
        float(log1p(product.rating_count))
        if product.rating_count is not None and product.rating_count >= 0
        else 0.0
    )
    values[FEATURE_INDEX["price_missing"]] = 0.0 if price_ok else 1.0
    values[FEATURE_INDEX["log_price"]] = float(log1p(price)) if price_ok else 0.0
    return values
