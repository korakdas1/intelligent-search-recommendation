"""Interaction data access."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.events import EVENT_TYPE_SET
from app.db.repositories.products import product_exists
from app.db.repositories.users import upsert_user_for_event
from app.models.interaction import Interaction


class InteractionError(ValueError):
    """Invalid interaction payload or missing catalog reference."""


def get_user_history(session: Session, user_id: str, *, limit: int = 50) -> list[Interaction]:
    stmt = (
        select(Interaction)
        .where(Interaction.user_id == user_id)
        .order_by(Interaction.occurred_at.desc())
        .limit(limit)
    )
    return list(session.scalars(stmt))


def find_duplicate(
    session: Session,
    *,
    user_id: str,
    product_id: str,
    event_type: str,
    occurred_at: datetime,
) -> Interaction | None:
    stmt = select(Interaction).where(
        Interaction.user_id == user_id,
        Interaction.product_id == product_id,
        Interaction.event_type == event_type,
        Interaction.occurred_at == occurred_at,
    )
    return session.scalars(stmt).first()


def record_interaction(
    session: Session,
    *,
    user_id: str,
    product_id: str,
    event_type: str,
    event_value: float | None = None,
    occurred_at: datetime | None = None,
    metadata: dict[str, Any] | None = None,
    dataset_version: str | None = None,
) -> tuple[Interaction, bool]:
    """Persist an interaction. Returns ``(row, created)``.

    Duplicate ``(user, product, event_type, occurred_at)`` rows are not
    inserted; the existing row is returned with ``created=False``.
    """

    if event_type not in EVENT_TYPE_SET:
        raise InteractionError(f"invalid event_type: {event_type}")
    if not user_id or not product_id:
        raise InteractionError("user_id and product_id are required")
    if not product_exists(session, product_id):
        raise InteractionError("product_id does not exist")

    when = occurred_at or datetime.now(tz=UTC)
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)

    existing = find_duplicate(
        session,
        user_id=user_id,
        product_id=product_id,
        event_type=event_type,
        occurred_at=when,
    )
    if existing is not None:
        return existing, False

    upsert_user_for_event(session, user_id, occurred_at=when, dataset_version=dataset_version)
    row = Interaction(
        user_id=user_id,
        product_id=product_id,
        event_type=event_type,
        event_value=event_value,
        occurred_at=when,
        dataset_version=dataset_version,
        metadata_=metadata or {},
    )
    session.add(row)
    session.flush()
    return row, True
