"""Orchestrate keyword, semantic, or hybrid search."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.repositories.search import log_keyword_search, search_products
from app.schemas.search import SearchResponse, SearchResultItem
from app.search.document import RETRIEVAL_MODE
from app.search.hybrid import FUSION_RRF, FUSION_WEIGHTED, hybrid_search
from app.search.semantic import semantic_search
from app.search.types import SearchHit


def hits_to_response(query: str, top_k: int, hits: list[SearchHit]) -> SearchResponse:
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
        retrieval_mode=RETRIEVAL_MODE,
        fusion_method=None,
        rerank_mode="none",
        top_k=top_k,
        results=results,
    )


def keyword_search(
    session: Session,
    *,
    query: str,
    top_k: int,
    log: bool = True,
) -> SearchResponse:
    hits = search_products(session, query, top_k)
    if log:
        log_keyword_search(session, query_text=query, top_k=top_k, hits=hits)
    return hits_to_response(query, top_k, hits)


def resolve_fusion_method(fusion_method: str | None) -> str:
    method = (fusion_method or get_settings().hybrid_fusion_method).strip().lower()
    if method not in {FUSION_RRF, FUSION_WEIGHTED}:
        raise ValueError(f"unsupported fusion_method: {fusion_method!r}")
    return method


def run_search(
    session: Session,
    *,
    query: str,
    top_k: int,
    retrieval_mode: str = RETRIEVAL_MODE,
    fusion_method: str | None = None,
    rerank_mode: str = "none",
    personalization_mode: str = "none",
    user_id: str | None = None,
    log: bool = True,
    candidate_k: int | None = None,
) -> SearchResponse:
    mode = (rerank_mode or "none").strip().lower()
    if mode not in {"none", "ltr"}:
        raise ValueError(f"unsupported rerank_mode: {rerank_mode!r}")
    personalization = (personalization_mode or "none").strip().lower()
    if personalization not in {"none", "bounded"}:
        raise ValueError(f"unsupported personalization_mode: {personalization_mode!r}")
    if personalization == "bounded":
        if not user_id:
            raise ValueError("user_id is required when personalization_mode=bounded")
        if retrieval_mode != "hybrid":
            raise ValueError("personalization_mode=bounded requires retrieval_mode=hybrid")
        from app.search.personalization_service import personalized_search

        return personalized_search(
            session,
            query=query,
            top_k=top_k,
            user_id=user_id,
            fusion_method=resolve_fusion_method(fusion_method),
            rerank_mode=mode,
            log=log,
            candidate_k=candidate_k,
        )
    if mode == "ltr":
        if retrieval_mode != "hybrid":
            raise ValueError("rerank_mode=ltr requires retrieval_mode=hybrid")
        from app.search.ltr import ltr_search

        return ltr_search(
            session,
            query=query,
            top_k=top_k,
            fusion_method=resolve_fusion_method(fusion_method),
            log=log,
            candidate_k=candidate_k,
        )
    if retrieval_mode == "semantic":
        return semantic_search(session, query=query, top_k=top_k, log=log)
    if retrieval_mode == "hybrid":
        return hybrid_search(
            session,
            query=query,
            top_k=top_k,
            fusion_method=resolve_fusion_method(fusion_method),
            log=log,
            candidate_k=candidate_k,
        )
    return keyword_search(session, query=query, top_k=top_k, log=log)
