"""Hybrid keyword + semantic candidate fusion. No LTR or personalization."""

from __future__ import annotations

import logging
from decimal import Decimal

from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.repositories.products import get_products_by_ids
from app.db.repositories.search import keyword_candidates, log_search_event
from app.schemas.search import SearchResponse, SearchResultItem
from app.search.candidates import FusedCandidate, resolve_candidate_k, union_candidates
from app.search.fusion import fuse_rrf, fuse_weighted
from app.search.runtime import SemanticRuntime, require_semantic_runtime
from app.search.semantic import semantic_candidates
from app.search.types import SearchHit

logger = logging.getLogger(__name__)

HYBRID_MODE = "hybrid"
HYBRID_SOURCE = "hybrid"
FUSION_RRF = "rrf"
FUSION_WEIGHTED = "weighted"


@dataclass(frozen=True, slots=True)
class HybridCollection:
    fused: list[FusedCandidate]
    depth: int
    method: str
    alpha: float
    keyword_count: int
    semantic_count: int
    union_count: int
    runtime: SemanticRuntime
    rrf_k: int


def collect_hybrid_fused(
    session: Session,
    *,
    query: str,
    top_k: int,
    fusion_method: str,
    keyword_weight: float | None = None,
    candidate_k: int | None = None,
) -> HybridCollection:
    """Return the full hybrid union, fused and ordered. Never logs. Never truncates."""

    settings = get_settings()
    method = fusion_method.strip().lower()
    if method not in {FUSION_RRF, FUSION_WEIGHTED}:
        raise ValueError(f"unsupported fusion_method: {fusion_method!r}")

    depth = resolve_candidate_k(
        top_k,
        configured=candidate_k if candidate_k is not None else settings.hybrid_candidate_k,
        maximum=settings.hybrid_candidate_k_max,
    )
    runtime = require_semantic_runtime(session)
    keyword = keyword_candidates(session, query, depth)
    semantic = semantic_candidates(session, query=query, top_k=depth, runtime=runtime)
    merged = union_candidates(keyword, semantic)

    alpha = settings.hybrid_keyword_weight if keyword_weight is None else keyword_weight
    if method == FUSION_RRF:
        fused = fuse_rrf(merged, k0=settings.hybrid_rrf_k)
    else:
        fused = fuse_weighted(merged, alpha=alpha)
    return HybridCollection(
        fused=fused,
        depth=depth,
        method=method,
        alpha=alpha,
        keyword_count=len(keyword),
        semantic_count=len(semantic),
        union_count=len(merged),
        runtime=runtime,
        rrf_k=settings.hybrid_rrf_k,
    )


def hybrid_search(
    session: Session,
    *,
    query: str,
    top_k: int,
    fusion_method: str,
    log: bool = True,
    keyword_weight: float | None = None,
    candidate_k: int | None = None,
) -> SearchResponse:
    """Fuse keyword and semantic candidate lists. Semantic unavailability is fatal."""

    collected = collect_hybrid_fused(
        session,
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,
        keyword_weight=keyword_weight,
        candidate_k=candidate_k,
    )
    chosen = collected.fused[:top_k]
    products = get_products_by_ids(session, [row.product_id for row in chosen])
    hits: list[SearchHit] = []
    for row in chosen:
        product = products.get(row.product_id)
        if product is None:
            logger.warning("hybrid hit %s missing from catalog; skipping", row.product_id)
            continue
        hits.append(
            SearchHit(
                product_id=product.product_id,
                title=product.title,
                brand=product.brand,
                category=product.category,
                price=product.price if isinstance(product.price, Decimal) else product.price,
                score=row.fusion_score,
                source=HYBRID_SOURCE,
            )
        )

    if log:
        log_search_event(
            session,
            query_text=query,
            top_k=top_k,
            hits=hits,
            metadata={
                key: value
                for key, value in {
                    "retrieval_mode": HYBRID_MODE,
                    "fusion_method": collected.method,
                    "candidate_k": collected.depth,
                    "keyword_weight": collected.alpha if collected.method == FUSION_WEIGHTED else None,
                    "rrf_k": collected.rrf_k if collected.method == FUSION_RRF else None,
                    "embedding_model": collected.runtime.model_name,
                    "semantic_artifact_version": collected.runtime.artifact_version,
                    "index_backend": collected.runtime.backend,
                    "keyword_candidate_count": collected.keyword_count,
                    "semantic_candidate_count": collected.semantic_count,
                    "union_candidate_count": collected.union_count,
                }.items()
                if value is not None
            },
            source=HYBRID_SOURCE,
        )

    results = [
        SearchResultItem(
            rank=index,
            product_id=hit.product_id,
            title=hit.title,
            brand=hit.brand,
            category=hit.category,
            price=hit.price,
            score=hit.score,
            source=hit.source,
        )
        for index, hit in enumerate(hits, start=1)
    ]
    return SearchResponse(
        query=query,
        retrieval_mode=HYBRID_MODE,
        fusion_method=collected.method,
        top_k=top_k,
        results=results,
    )
