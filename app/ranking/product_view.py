"""Product snapshot used by the ranking feature extractor. No SQL here."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    return float(value)


@dataclass(frozen=True, slots=True)
class ProductView:
    product_id: str
    title: str
    brand: str | None = None
    category: str | None = None
    description: str | None = None
    price: float | None = None
    average_rating: float | None = None
    rating_count: int | None = None

    @classmethod
    def from_product(cls, product: Any) -> ProductView:
        rating_count = product.rating_count
        return cls(
            product_id=product.product_id,
            title=product.title or "",
            brand=_optional_str(product.brand),
            category=_optional_str(product.category),
            description=_optional_str(product.description),
            price=_optional_float(product.price),
            average_rating=_optional_float(product.average_rating),
            rating_count=int(rating_count) if rating_count is not None else None,
        )
