"""User data access."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy.orm import Session

from app.models.user import User


def get_user(session: Session, user_id: str) -> User | None:
    return session.get(User, user_id)


def upsert_user_for_event(
    session: Session,
    user_id: str,
    *,
    occurred_at: datetime,
    dataset_version: str | None = None,
) -> User:
    """Create a user or widen first/last seen around ``occurred_at``.

    Incoming timestamps never push ``first_seen_at`` later or
    ``last_seen_at`` earlier.
    """

    user = session.get(User, user_id)
    if user is None:
        user = User(
            user_id=user_id,
            dataset_version=dataset_version,
            first_seen_at=occurred_at,
            last_seen_at=occurred_at,
            metadata_={},
        )
        session.add(user)
        # Flush before any interaction INSERT. SQLAlchemy's unit of work does
        # not order pending rows by matching scalar FK values unless a
        # relationship() is configured, so the users row must exist first.
        session.flush()
        return user
    if user.first_seen_at is None or occurred_at < user.first_seen_at:
        user.first_seen_at = occurred_at
    if user.last_seen_at is None or occurred_at > user.last_seen_at:
        user.last_seen_at = occurred_at
    return user
