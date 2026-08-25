"""PostgreSQL full-text keyword retrieval."""

from __future__ import annotations

from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.events import DATASET_VERSION_FULL
from app.models.dataset import DatasetVersion
from app.models.product import Product
from app.models.search import SearchEvent, SearchEventResult
from app.search.candidates import RetrievalCandidate
from app.search.document import PARSER_NAME, RANKER_NAME, RETRIEVAL_MODE, SEARCH_CONFIG
from app.search.types import SearchHit


def parse_websearch_query(query_text: str):
    """Return a parameterized ``websearch_to_tsquery`` clause."""

    return func.websearch_to_tsquery(SEARCH_CONFIG, query_text)


def tsquery_has_lexemes(session: Session, query_text: str) -> bool:
    """False when English FTS strips the query to an empty tsquery."""

    node_count = session.scalar(select(func.numnode(parse_websearch_query(query_text))))
    return bool(node_count and int(node_count) > 0)


def keyword_candidates(session: Session, query_text: str, top_k: int) -> list[RetrievalCandidate]:
    """Return lexical product_id/score/rank only. Never logs."""

    if top_k < 1:
        return []
    if not tsquery_has_lexemes(session, query_text):
        return []

    tsquery = parse_websearch_query(query_text)
    rank = func.ts_rank_cd(Product.search_document, tsquery)
    stmt = (
        select(Product.product_id, rank.label("score"))
        .where(Product.search_document.op("@@")(tsquery))
        .order_by(rank.desc(), Product.product_id.asc())
        .limit(top_k)
    )
    candidates: list[RetrievalCandidate] = []
    for position, row in enumerate(session.execute(stmt), start=1):
        score = float(row.score) if row.score is not None else 0.0
        candidates.append(
            RetrievalCandidate(
                product_id=row.product_id,
                rank=position,
                score=score,
                source=RETRIEVAL_MODE,
            )
        )
    return candidates


def search_products(session: Session, query_text: str, top_k: int) -> list[SearchHit]:
    """Return lexically ranked products. Empty tsquery → no rows, no error."""

    if top_k < 1:
        return []
    if not tsquery_has_lexemes(session, query_text):
        return []

    tsquery = parse_websearch_query(query_text)
    rank = func.ts_rank_cd(Product.search_document, tsquery)
    stmt = (
        select(
            Product.product_id,
            Product.title,
            Product.brand,
            Product.category,
            Product.price,
            rank.label("score"),
        )
        .where(Product.search_document.op("@@")(tsquery))
        .order_by(rank.desc(), Product.product_id.asc())
        .limit(top_k)
    )
    hits: list[SearchHit] = []
    for row in session.execute(stmt):
        score = float(row.score) if row.score is not None else 0.0
        hits.append(
            SearchHit(
                product_id=row.product_id,
                title=row.title,
                brand=row.brand,
                category=row.category,
                price=row.price if isinstance(row.price, Decimal) else row.price,
                score=score,
                source=RETRIEVAL_MODE,
            )
        )
    return hits


def resolve_dataset_version(session: Session) -> str | None:
    if session.get(DatasetVersion, DATASET_VERSION_FULL) is not None:
        return DATASET_VERSION_FULL
    return session.scalar(select(DatasetVersion.dataset_version).limit(1))


def log_keyword_search(
    session: Session,
    *,
    query_text: str,
    top_k: int,
    hits: list[SearchHit],
) -> int:
    """Persist the issued query and retrieved rows. Not relevance labels."""

    return log_search_event(
        session,
        query_text=query_text,
        top_k=top_k,
        hits=hits,
        metadata={
            "retrieval_mode": RETRIEVAL_MODE,
            "parser": PARSER_NAME,
            "ranker": RANKER_NAME,
        },
        source=RETRIEVAL_MODE,
    )


def log_search_event(
    session: Session,
    *,
    query_text: str,
    top_k: int,
    hits: list[SearchHit],
    metadata: dict,
    source: str,
    user_id: str | None = None,
) -> int:
    """Persist an issued query. Ranked rows are observed retrievals, not labels."""

    event = SearchEvent(
        user_id=user_id,
        query_text=query_text,
        normalized_query=" ".join(query_text.split()).lower(),
        top_k=top_k,
        dataset_version=resolve_dataset_version(session),
        label_class="observed",
        metadata_=metadata,
    )
    session.add(event)
    session.flush()
    for rank_position, hit in enumerate(hits, start=1):
        session.add(
            SearchEventResult(
                search_event_id=event.search_event_id,
                product_id=hit.product_id,
                rank=rank_position,
                score=hit.score,
                source=source,
            )
        )
    return event.search_event_id
