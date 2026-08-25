"""Search API schemas. Keyword, semantic, or hybrid retrieval."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=512)
    top_k: int = Field(default=10, ge=1, le=100)
    retrieval_mode: Literal["keyword", "semantic", "hybrid"] = "keyword"
    fusion_method: Literal["rrf", "weighted"] | None = None
    rerank_mode: Literal["none", "ltr"] = "none"
    personalization_mode: Literal["none", "bounded"] = "none"
    user_id: str | None = Field(default=None, max_length=256)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("query must not be blank")
        return stripped

    @field_validator("user_id")
    @classmethod
    def user_id_must_not_be_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            raise ValueError("user_id must not be blank")
        return stripped

    @model_validator(mode="after")
    def fusion_method_only_for_hybrid(self) -> Self:
        if self.retrieval_mode != "hybrid" and self.fusion_method is not None:
            raise ValueError("fusion_method is only valid when retrieval_mode is hybrid")
        if self.rerank_mode == "ltr" and self.retrieval_mode != "hybrid":
            raise ValueError("rerank_mode=ltr requires retrieval_mode=hybrid")
        if self.personalization_mode == "bounded":
            if self.user_id is None:
                raise ValueError("user_id is required when personalization_mode=bounded")
            if self.retrieval_mode != "hybrid":
                raise ValueError("personalization_mode=bounded requires retrieval_mode=hybrid")
        if self.personalization_mode == "none" and self.user_id is not None:
            raise ValueError("user_id requires personalization_mode=bounded")
        return self


class SearchResultItem(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rank: int
    product_id: str
    title: str
    brand: str | None = None
    category: str | None = None
    price: Decimal | None = None
    score: float
    source: str = "keyword"


class SearchResponse(BaseModel):
    query: str
    retrieval_mode: str = "keyword"
    fusion_method: str | None = None
    rerank_mode: str = "none"
    personalization_mode: str = "none"
    personalization_version: str | None = None
    personalization_applied: bool = False
    personalization_reason: str | None = None
    personalization_signals: list[str] = Field(default_factory=list)
    user_id: str | None = None
    top_k: int
    results: list[SearchResultItem]
