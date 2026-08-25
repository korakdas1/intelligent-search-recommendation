"""Interaction API schemas."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_serializer

EventType = Literal["view", "click", "add_to_cart", "purchase", "review"]


class InteractionCreateRequest(BaseModel):
    user_id: str = Field(..., min_length=1, max_length=256)
    product_id: str = Field(..., min_length=1, max_length=64)
    event_type: EventType
    event_value: Decimal | None = None
    occurred_at: datetime | None = None
    metadata: dict[str, Any] | None = None


class InteractionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    interaction_id: int
    user_id: str
    product_id: str
    event_type: str
    event_value: Decimal | None = None
    occurred_at: datetime
    created: bool

    @field_serializer("occurred_at")
    def serialize_occurred_at(self, value: datetime) -> str:
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
