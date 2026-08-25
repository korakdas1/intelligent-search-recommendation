"""Optional RankNet reranking of the hybrid candidate union. Does not retrieve."""

from __future__ import annotations

import logging
from decimal import Decimal

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.repositories.products import get_products_by_ids
from app.db.repositories.search import log_search_event
from app.ranking.constants import LTR_SOURCE, RERANK_LTR
from app.ranking.features import extract_features
from app.ranking.inference import score_feature_matrix
from app.ranking.product_view import ProductView
from app.ranking.retrieval import collect_fused_candidates
from app.ranking.runtime import require_ltr_runtime
from app.ranking.validate import validate_feature_batch
from app.schemas.search import SearchResponse, SearchResultItem
from app.search.hybrid import FUSION_RRF, FUSION_WEIGHTED, HYBRID_MODE
from app.search.personalization_types import BaselineCandidate
from app.search.runtime import require_semantic_runtime
from app.search.types import SearchHit

logger = logging.getLogger(__name__)


def _resolve_fusion_method(fusion_method: str | None) -> str:
    method = (fusion_method or get_settings().hybrid_fusion_method).strip().lower()
    if method not in {FUSION_RRF, FUSION_WEIGHTED}:
        raise ValueError(f"unsupported fusion_method: {fusion_method!r}")
    return method


def score_ltr_union(
    session: Session,
    *,
    query: str,
    top_k: int,
    fusion_method: str,
    candidate_k: int | None = None,
) -> tuple[list[BaselineCandidate], int]:
    """RankNet scores for the full hybrid union. Never logs. Never truncates."""

    settings = get_settings()
    _resolve_fusion_method(fusion_method)
    runtime = require_ltr_runtime()
    fused, depth = collect_fused_candidates(
        session,
        query,
        candidate_k=candidate_k if candidate_k is not None else settings.hybrid_candidate_k,
        top_k=top_k,
    )
    if not fused:
        return [], depth
    products = get_products_by_ids(session, [row.product_id for row in fused])
    views = {pid: ProductView.from_product(item) for pid, item in products.items()}
    batch = validate_feature_batch(
        extract_features(
            query,
            fused,
            views,
            candidate_k=depth,
            rrf_k=settings.hybrid_rrf_k,
            keyword_weight=settings.hybrid_keyword_weight,
        )
    )
    scores = score_feature_matrix(runtime.model, runtime.scaler, batch.values)
    scored = list(zip(batch.product_ids, scores, strict=True))
    scored.sort(key=lambda item: (-float(item[1]), item[0]))
    baseline = [
        BaselineCandidate(
            product_id=product_id,
            baseline_score=float(score),
            baseline_rank=index,
            source=LTR_SOURCE,
        )
        for index, (product_id, score) in enumerate(scored, start=1)
    ]
    return baseline, depth


def ltr_search(
    session: Session,
    *,
    query: str,
    top_k: int,
    fusion_method: str,
    log: bool = True,
    candidate_k: int | None = None,
) -> SearchResponse:
    """Rerank the full hybrid union with RankNet. Candidate IDs stay the union set."""

    settings = get_settings()
    method = _resolve_fusion_method(fusion_method)
    runtime = require_ltr_runtime()
    semantic = require_semantic_runtime(session)
    fused, depth = collect_fused_candidates(
        session,
        query,
        candidate_k=candidate_k if candidate_k is not None else settings.hybrid_candidate_k,
        top_k=top_k,
    )
    if not fused:
        if log:
            log_search_event(
                session,
                query_text=query,
                top_k=top_k,
                hits=[],
                metadata=_ltr_metadata(
                    method=method,
                    depth=depth,
                    runtime=runtime,
                    semantic=semantic,
                    keyword_weight=settings.hybrid_keyword_weight,
                    union_count=0,
                ),
                source=LTR_SOURCE,
            )
        return SearchResponse(
            query=query,
            retrieval_mode=HYBRID_MODE,
            fusion_method=method,
            rerank_mode=RERANK_LTR,
            top_k=top_k,
            results=[],
        )

    products = get_products_by_ids(session, [row.product_id for row in fused])
    views = {pid: ProductView.from_product(item) for pid, item in products.items()}
    batch = validate_feature_batch(
        extract_features(
            query,
            fused,
            views,
            candidate_k=depth,
            rrf_k=settings.hybrid_rrf_k,
            keyword_weight=settings.hybrid_keyword_weight,
        )
    )
    scores = score_feature_matrix(runtime.model, runtime.scaler, batch.values)
    scored = list(zip(batch.product_ids, scores, strict=True))
    scored.sort(key=lambda item: (-float(item[1]), item[0]))
    chosen = scored[:top_k]
    hits: list[SearchHit] = []
    for product_id, score in chosen:
        product = products.get(product_id)
        if product is None:
            logger.warning("LTR hit %s missing from catalog; skipping", product_id)
            continue
        hits.append(
            SearchHit(
                product_id=product.product_id,
                title=product.title,
                brand=product.brand,
                category=product.category,
                price=product.price if isinstance(product.price, Decimal) else product.price,
                score=float(score),
                source=LTR_SOURCE,
            )
        )

    if log:
        log_search_event(
            session,
            query_text=query,
            top_k=top_k,
            hits=hits,
            metadata=_ltr_metadata(
                method=method,
                depth=depth,
                runtime=runtime,
                semantic=semantic,
                keyword_weight=settings.hybrid_keyword_weight,
                union_count=len(fused),
            ),
            source=LTR_SOURCE,
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
        fusion_method=method,
        rerank_mode=RERANK_LTR,
        top_k=top_k,
        results=results,
    )


def _ltr_metadata(*, method, depth, runtime, semantic, keyword_weight, union_count) -> dict:
    payload = {
        "retrieval_mode": HYBRID_MODE,
        "rerank_mode": RERANK_LTR,
        "fusion_method": method,
        "candidate_k": depth,
        "union_candidate_count": union_count,
        "ranker_version": runtime.model_version,
        "feature_version": runtime.feature_version,
        "semantic_artifact_version": semantic.artifact_version,
        "embedding_model": semantic.model_name,
        "index_backend": semantic.backend,
        "rrf_k": get_settings().hybrid_rrf_k if method == "rrf" else None,
        "keyword_weight": keyword_weight if method == "weighted" else None,
    }
    return {key: value for key, value in payload.items() if value is not None}
