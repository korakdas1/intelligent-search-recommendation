"""Semantic retrieval: encode query → FAISS → batch product metadata.

Keyword SQL is not used here. Hybrid fusion calls ``semantic_candidates``.
"""

from __future__ import annotations

import logging
import time
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.repositories.products import get_products_by_ids
from app.db.repositories.search import log_search_event
from app.embeddings.constants import SEMANTIC_RETRIEVAL_MODE, SEMANTIC_SOURCE
from app.embeddings.encoder import SentenceTransformerEncoder, resolve_device
from app.embeddings.normalize import l2_normalize
from app.schemas.search import SearchResponse, SearchResultItem
from app.search.candidates import RetrievalCandidate
from app.search.faiss_index import search_index
from app.search.runtime import SemanticRuntime, require_semantic_runtime
from app.search.types import SearchHit

logger = logging.getLogger(__name__)


def semantic_candidates(
    session: Session,
    *,
    query: str,
    top_k: int,
    runtime: SemanticRuntime | None = None,
) -> list[RetrievalCandidate]:
    """Return FAISS neighbors without metadata fetch or search logging."""

    runtime = runtime or require_semantic_runtime(session)
    if runtime.ntotal < 1 or top_k < 1:
        return []
    encoder = _ensure_encoder(runtime)
    query_vector = l2_normalize(encoder.encode([query], batch_size=1, show_progress=False))
    fetch_k = min(max(top_k * 2, top_k), runtime.ntotal)
    scores, indices = search_index(
        runtime.index,
        query_vector,
        fetch_k,
        ef_search=_ef_search_for(runtime),
    )
    ranked = _stable_hits(runtime, scores[0], indices[0], top_k)
    return [
        RetrievalCandidate(
            product_id=hit.product_id,
            rank=index,
            score=hit.score,
            source=SEMANTIC_SOURCE,
        )
        for index, hit in enumerate(ranked, start=1)
    ]


def semantic_search(
    session: Session,
    *,
    query: str,
    top_k: int,
    log: bool = True,
    runtime: SemanticRuntime | None = None,
) -> SearchResponse:
    runtime = runtime or require_semantic_runtime(session)
    encoder = _ensure_encoder(runtime)
    started = time.perf_counter()
    query_vector = encoder.encode([query], batch_size=1, show_progress=False)
    encode_ms = (time.perf_counter() - started) * 1000
    query_vector = l2_normalize(query_vector)

    fetch_k = min(max(top_k * 2, top_k), runtime.ntotal or top_k)
    started = time.perf_counter()
    scores, indices = search_index(
        runtime.index,
        query_vector,
        fetch_k,
        ef_search=_ef_search_for(runtime),
    )
    search_ms = (time.perf_counter() - started) * 1000

    ranked = _stable_hits(runtime, scores[0], indices[0], top_k)
    product_ids = [hit.product_id for hit in ranked]
    started = time.perf_counter()
    products = get_products_by_ids(session, product_ids)
    fetch_ms = (time.perf_counter() - started) * 1000

    hits: list[SearchHit] = []
    for candidate in ranked:
        product = products.get(candidate.product_id)
        if product is None:
            logger.warning("semantic hit %s missing from catalog; skipping", candidate.product_id)
            continue
        hits.append(
            SearchHit(
                product_id=product.product_id,
                title=product.title,
                brand=product.brand,
                category=product.category,
                price=product.price if isinstance(product.price, Decimal) else product.price,
                score=candidate.score,
                source=SEMANTIC_SOURCE,
            )
        )
        if len(hits) >= top_k:
            break

    if log:
        log_search_event(
            session,
            query_text=query,
            top_k=top_k,
            hits=hits,
            metadata={
                "retrieval_mode": SEMANTIC_RETRIEVAL_MODE,
                "embedding_model": runtime.model_name,
                "semantic_artifact_version": runtime.artifact_version,
                "index_backend": runtime.backend,
                "encode_ms": round(encode_ms, 3),
                "faiss_ms": round(search_ms, 3),
                "metadata_fetch_ms": round(fetch_ms, 3),
            },
            source=SEMANTIC_SOURCE,
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
        retrieval_mode=SEMANTIC_RETRIEVAL_MODE,
        top_k=top_k,
        results=results,
    )


def _ensure_encoder(runtime: SemanticRuntime) -> SentenceTransformerEncoder | object:
    if runtime.encoder is not None:
        return runtime.encoder
    from app.core.config import get_settings

    settings = get_settings()
    try:
        device = resolve_device(settings.semantic_device)
    except RuntimeError:
        logger.warning("requested CUDA unavailable; falling back to CPU for encoding")
        device = "cpu"
    runtime.encoder = SentenceTransformerEncoder(
        runtime.model_name,
        device=device,
        model_revision=runtime.embedding_manifest.get("model_revision"),
        model_license=runtime.embedding_manifest.get("model_license"),
        embedding_dim=runtime.embedding_dim,
    )
    return runtime.encoder


def _ef_search_for(runtime: SemanticRuntime) -> int | None:
    if runtime.backend != "hnsw":
        return None
    from app.core.config import get_settings

    return get_settings().semantic_hnsw_ef_search


def _stable_hits(
    runtime: SemanticRuntime,
    scores: object,
    indices: object,
    top_k: int,
) -> list[SearchHit]:
    candidates: list[tuple[float, str]] = []
    for score, row in zip(scores, indices, strict=True):
        row_i = int(row)
        if row_i < 0:
            continue
        product_id = runtime.row_to_product_id(row_i)
        if product_id is None:
            continue
        candidates.append((float(score), product_id))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    unique: list[SearchHit] = []
    seen: set[str] = set()
    for score, product_id in candidates:
        if product_id in seen:
            continue
        seen.add(product_id)
        unique.append(
            SearchHit(
                product_id=product_id,
                title="",
                brand=None,
                category=None,
                price=None,
                score=score,
                source=SEMANTIC_SOURCE,
            )
        )
        if len(unique) >= top_k:
            break
    return unique
