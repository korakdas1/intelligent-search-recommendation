"""Keyword or semantic search result objects."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal


@dataclass(frozen=True, slots=True)
class SearchHit:
    product_id: str
    title: str
    brand: str | None
    category: str | None
    price: Decimal | None
    score: float
    source: str = "keyword"
