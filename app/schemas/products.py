"""Product API schemas. Exact ID lookup only — not search."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class ProductResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    product_id: str
    title: str
    description: str | None = None
    category: str | None = None
    subcategory: str | None = None
    brand: str | None = None
    store: str | None = None
    price: Decimal | None = None
    currency: str
    average_rating: Decimal | None = None
    rating_count: int | None = None


class ErrorResponse(BaseModel):
    detail: str = Field(..., examples=["Product not found"])
