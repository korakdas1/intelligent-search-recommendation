"""Amazon Reviews 2023 adapter.

Source-specific field names stay here. Normalized output is dataset-neutral.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from app.data.schemas import NormalizedInteraction, NormalizedProduct
from app.data.text import build_searchable_text

CANONICAL_PRODUCT_ID_FIELD = "parent_asin"
EVENT_TYPE_REVIEW = "review"
TIMESTAMP_SOURCE_FIELD = "timestamp"
TIMESTAMP_UNIT = "unix_milliseconds"

_MISSING_STRINGS = {"", "none", "null", "nan", "n/a"}


class NormalizationError(ValueError):
    """A required field is missing or invalid."""


def _clean_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and value != value:
        return None
    text = str(value).strip()
    if text.lower() in _MISSING_STRINGS:
        return None
    return text


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        cleaned = _clean_str(value)
        return [cleaned] if cleaned else []
    if isinstance(value, list):
        out: list[str] = []
        for item in value:
            cleaned = _clean_str(item)
            if cleaned:
                out.append(cleaned)
        return out
    return []


def _join_text(parts: Iterable[str]) -> str | None:
    joined = " ".join(part.strip() for part in parts if part and part.strip())
    return joined or None


def parse_details(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() in _MISSING_STRINGS:
            return {}
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def parse_price(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value != value:
            return None
        return float(value)
    cleaned = _clean_str(value)
    if cleaned is None:
        return None
    cleaned = cleaned.replace("$", "").replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def parse_rating(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        rating = float(value)
    except (TypeError, ValueError):
        return None
    if rating != rating:
        return None
    if rating < 1.0 or rating > 5.0:
        return None
    return rating


def parse_timestamp(value: Any) -> datetime:
    """Convert Amazon review timestamps to UTC.

    Official All_Beauty examples use millisecond Unix time (e.g. 1588687728923).
    Values below 10^11 are treated as seconds.
    """

    if value is None or isinstance(value, bool):
        raise NormalizationError("timestamp is missing")
    try:
        raw = float(value)
    except (TypeError, ValueError) as exc:
        raise NormalizationError(f"timestamp is not numeric: {value!r}") from exc
    if raw != raw or raw <= 0:
        raise NormalizationError(f"timestamp is invalid: {value!r}")
    seconds = raw / 1000.0 if raw >= 1e11 else raw
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError) as exc:
        raise NormalizationError(f"timestamp out of range: {value!r}") from exc


def extract_brand(details: Mapping[str, Any], store: str | None) -> str | None:
    for key in ("Brand", "brand"):
        brand = _clean_str(details.get(key))
        if brand:
            return brand
    return None


def normalize_product(record: Mapping[str, Any]) -> NormalizedProduct:
    product_id = _clean_str(record.get("parent_asin"))
    if not product_id:
        raise NormalizationError("parent_asin is required")
    title = _clean_str(record.get("title"))
    if not title:
        raise NormalizationError("title is required")

    details = parse_details(record.get("details"))
    store = _clean_str(record.get("store"))
    brand = extract_brand(details, store)
    category = _clean_str(record.get("main_category"))
    categories = _as_text_list(record.get("categories"))
    subcategory = None
    if categories:
        subcategory = next((item for item in categories if item != category), categories[0])

    features = _join_text(_as_text_list(record.get("features")))
    description = _join_text(_as_text_list(record.get("description")))
    searchable = build_searchable_text(
        title=title,
        brand=brand,
        category=category,
        features=features,
        description=description,
    )
    extras = {
        "source_parent_asin": product_id,
        "categories": categories,
        "has_features": bool(features),
        "has_images": bool(record.get("images")),
        "details_keys": sorted(str(key) for key in details.keys()),
    }
    rating_count = record.get("rating_number")
    try:
        rating_count_int = int(rating_count) if rating_count is not None else None
    except (TypeError, ValueError):
        rating_count_int = None

    return NormalizedProduct(
        product_id=product_id,
        title=title,
        description=description,
        category=category,
        subcategory=subcategory,
        brand=brand,
        store=store,
        price=parse_price(record.get("price")),
        average_rating=parse_rating(record.get("average_rating"))
        if record.get("average_rating") is not None
        else None,
        rating_count=rating_count_int,
        searchable_text=searchable,
        extras=extras,
    )


def normalize_interaction(record: Mapping[str, Any]) -> NormalizedInteraction:
    user_id = _clean_str(record.get("user_id"))
    if not user_id:
        raise NormalizationError("user_id is required")
    product_id = _clean_str(record.get("parent_asin"))
    if not product_id:
        raise NormalizationError("parent_asin is required")
    occurred_at = parse_timestamp(record.get("timestamp"))
    extras = {
        "review_title": _clean_str(record.get("title")),
        "has_review_text": bool(_clean_str(record.get("text"))),
    }
    helpful = record.get("helpful_vote", record.get("helpful_votes"))
    try:
        helpful_int = int(helpful) if helpful is not None else None
    except (TypeError, ValueError):
        helpful_int = None
        extras["helpful_vote_raw"] = helpful
    verified = record.get("verified_purchase")
    if isinstance(verified, str):
        verified = verified.strip().lower() in {"true", "1", "yes"}
    elif verified is not None:
        verified = bool(verified)

    return NormalizedInteraction(
        user_id=user_id,
        product_id=product_id,
        event_type=EVENT_TYPE_REVIEW,
        event_value=parse_rating(record.get("rating")),
        occurred_at=occurred_at,
        source_asin=_clean_str(record.get("asin")),
        extras={
            **extras,
            "verified_purchase": verified,
            "helpful_vote": helpful_int,
        },
    )


def product_to_row(product: NormalizedProduct) -> dict[str, Any]:
    return {
        "product_id": product.product_id,
        "title": product.title,
        "description": product.description,
        "category": product.category,
        "subcategory": product.subcategory,
        "brand": product.brand,
        "store": product.store,
        "price": product.price,
        "average_rating": product.average_rating,
        "rating_count": product.rating_count,
        "searchable_text": product.searchable_text,
        "extras_json": json.dumps(product.extras, ensure_ascii=False, sort_keys=True),
    }


def interaction_to_row(interaction: NormalizedInteraction) -> dict[str, Any]:
    extras = dict(interaction.extras)
    return {
        "user_id": interaction.user_id,
        "product_id": interaction.product_id,
        "source_asin": interaction.source_asin,
        "event_type": interaction.event_type,
        "event_value": interaction.event_value,
        "occurred_at": interaction.occurred_at,
        "verified_purchase": extras.pop("verified_purchase", None),
        "helpful_vote": extras.pop("helpful_vote", None),
        "extras_json": json.dumps(extras, ensure_ascii=False, sort_keys=True),
    }
