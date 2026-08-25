"""Historical interaction-count popularity. Not real-time trending."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from app.models.interaction import Interaction
from app.recommendations.ranking import ScoredItem, stable_rank


def rank_popularity_counts(
    counts: dict[str, int],
    *,
    exclude: set[str] | None = None,
    top_k: int,
) -> list[ScoredItem]:
    blocked = exclude or set()
    items = [
        ScoredItem(product_id=product_id, score=float(count))
        for product_id, count in counts.items()
        if product_id not in blocked
    ]
    return stable_rank(items, top_k=top_k)


def popularity_counts(
    session: Session,
    *,
    before: datetime | None = None,
    interaction_ids: Sequence[int] | None = None,
) -> dict[str, int]:
    """Count interactions. Evaluation must pass train-only ids or a cutoff."""

    stmt: Select[tuple[str, int]] = select(
        Interaction.product_id,
        func.count().label("cnt"),
    )
    if before is not None:
        stmt = stmt.where(Interaction.occurred_at < before)
    if interaction_ids is not None:
        if not interaction_ids:
            return {}
        stmt = stmt.where(Interaction.interaction_id.in_(list(interaction_ids)))
    stmt = stmt.group_by(Interaction.product_id)
    rows = session.execute(stmt).all()
    return {str(product_id): int(count) for product_id, count in rows}


def popular_products(
    session: Session,
    *,
    top_k: int,
    before: datetime | None = None,
    interaction_ids: Sequence[int] | None = None,
    exclude: set[str] | None = None,
) -> list[ScoredItem]:
    blocked = exclude or set()
    fetch = top_k + len(blocked)
    stmt: Select[tuple[str, int]] = select(
        Interaction.product_id,
        func.count().label("cnt"),
    )
    if before is not None:
        stmt = stmt.where(Interaction.occurred_at < before)
    if interaction_ids is not None:
        if not interaction_ids:
            return []
        stmt = stmt.where(Interaction.interaction_id.in_(list(interaction_ids)))
    stmt = (
        stmt.group_by(Interaction.product_id)
        .order_by(func.count().desc(), Interaction.product_id.asc())
        .limit(fetch)
    )
    counts = {str(product_id): int(count) for product_id, count in session.execute(stmt)}
    return rank_popularity_counts(counts, exclude=blocked, top_k=top_k)
