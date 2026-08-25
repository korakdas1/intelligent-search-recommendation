"""Search-log tables. Records issued queries, not relevance labels."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class SearchEvent(Base):
    __tablename__ = "search_events"
    __table_args__ = (Index("ix_search_events_user_occurred", "user_id", "occurred_at"),)

    search_event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    user_id: Mapped[str | None] = mapped_column(Text, ForeignKey("users.user_id"), nullable=True)
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    top_k: Mapped[int | None] = mapped_column(Integer, nullable=True)
    occurred_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
    )
    clicked_product_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("products.product_id"), nullable=True
    )
    converted_product_id: Mapped[str | None] = mapped_column(
        Text, ForeignKey("products.product_id"), nullable=True
    )
    dataset_version: Mapped[str | None] = mapped_column(
        Text, ForeignKey("dataset_versions.dataset_version"), nullable=True
    )
    label_class: Mapped[str | None] = mapped_column(Text, nullable=True)
    metadata_: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default="{}",
    )


class SearchEventResult(Base):
    __tablename__ = "search_event_results"
    __table_args__ = (
        CheckConstraint("rank >= 1", name="ck_search_event_results_rank"),
        Index("ix_search_event_results_event_rank", "search_event_id", "rank"),
    )

    search_event_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("search_events.search_event_id"),
        primary_key=True,
    )
    product_id: Mapped[str] = mapped_column(
        Text,
        ForeignKey("products.product_id"),
        primary_key=True,
    )
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source: Mapped[str | None] = mapped_column(Text, nullable=True)
