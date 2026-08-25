"""Recommendation API schemas. Embeddings are never returned."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field


class RecommendationItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int
    product_id: str
    title: str
    brand: str | None = None
    category: str | None = None
    price: Decimal | None = None
    score: float
    source: str


class RecommendationResponse(BaseModel):
    recommendation_mode: str
    content_rec_version: str | None = None
    cf_model_version: str | None = None
    hybrid_version: str | None = None
    fusion_method: str | None = None
    channels_used: list[str] | None = None
    fallback_reason: str | None = None
    product_id: str | None = None
    user_id: str | None = None
    history_items: int | None = None
    reason: str | None = None
    top_k: int
    results: list[RecommendationItem]


class RecommendationQuery(BaseModel):
    top_k: int = Field(default=10, ge=1, le=100)
