"""ORM model for user–item events."""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Numeric,
    Text,
    UniqueConstraint,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.events import EVENT_TYPES
from app.db.base import Base

_EVENT_SQL = ", ".join(f"'{name}'" for name in EVENT_TYPES)


class Interaction(Base):
    __tablename__ = "interactions"
    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "product_id",
            "event_type",
            "occurred_at",
            name="uq_interactions_user_product_type_time",
        ),
        CheckConstraint(f"event_type IN ({_EVENT_SQL})", name="ck_interactions_event_type"),
        Index("ix_interactions_user_occurred", "user_id", "occurred_at"),
        Index("ix_interactions_product_event", "product_id", "event_type"),
        Index("ix_interactions_occurred_at", "occurred_at"),
    )

    interaction_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(Text, ForeignKey("users.user_id"), nullable=False)
    product_id: Mapped[str] = mapped_column(Text, ForeignKey("products.product_id"), nullable=False)
    event_type: Mapped[str] = mapped_column(Text, nullable=False)
    event_value: Mapped[Decimal | None] = mapped_column(Numeric, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    dataset_version: Mapped[str | None] = mapped_column(
        Text,
        ForeignKey("dataset_versions.dataset_version"),
        nullable=True,
    )
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default=text("'{}'::jsonb"),
    )
