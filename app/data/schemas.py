"""Dataset-neutral normalized records for catalog ingest.

These are file/dataframe representations for later PostgreSQL ingest.
They are not database models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True, slots=True)
class NormalizedProduct:
    product_id: str
    title: str
    description: str | None
    category: str | None
    subcategory: str | None
    brand: str | None
    store: str | None
    price: float | None
    average_rating: float | None
    rating_count: int | None
    searchable_text: str
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NormalizedInteraction:
    user_id: str
    product_id: str
    event_type: str
    event_value: float | None
    occurred_at: datetime
    source_asin: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


PRODUCT_COLUMNS = [
    "product_id",
    "title",
    "description",
    "category",
    "subcategory",
    "brand",
    "store",
    "price",
    "average_rating",
    "rating_count",
    "searchable_text",
    "extras_json",
]

INTERACTION_COLUMNS = [
    "user_id",
    "product_id",
    "source_asin",
    "event_type",
    "event_value",
    "occurred_at",
    "verified_purchase",
    "helpful_vote",
    "extras_json",
]
