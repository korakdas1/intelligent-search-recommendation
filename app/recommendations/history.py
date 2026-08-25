"""User-history helpers for content profiles. Serving uses full known history."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.interaction import Interaction


def list_user_interactions(
    session: Session,
    user_id: str,
    *,
    before: datetime | None = None,
) -> list[Interaction]:
    """All events for a user. ``before`` is a strict cutoff (as-of)."""

    stmt = select(Interaction).where(Interaction.user_id == user_id)
    if before is not None:
        stmt = stmt.where(Interaction.occurred_at < before)
    stmt = stmt.order_by(Interaction.occurred_at.asc(), Interaction.interaction_id.asc())
    return list(session.scalars(stmt))


def unique_product_ids(interactions: Sequence[Interaction]) -> list[str]:
    """Deduplicate by product_id. One product contributes one embedding."""

    seen: dict[str, None] = {}
    for row in interactions:
        seen.setdefault(str(row.product_id), None)
    return list(seen.keys())


def last_occurrence_by_product(interactions: Sequence[Interaction]) -> dict[str, datetime]:
    latest: dict[str, datetime] = {}
    for row in interactions:
        product_id = str(row.product_id)
        previous = latest.get(product_id)
        if previous is None or row.occurred_at >= previous:
            latest[product_id] = row.occurred_at
    return latest
