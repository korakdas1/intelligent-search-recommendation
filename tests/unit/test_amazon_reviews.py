"""Amazon adapter tests. Fixtures are synthetic; no network or real dataset."""

from datetime import UTC

import pytest

from app.data.amazon_reviews import (
    NormalizationError,
    normalize_interaction,
    normalize_product,
    parse_price,
    parse_timestamp,
)
from app.data.text import TEXT_FIELD_ORDER, build_searchable_text


def test_timestamp_milliseconds() -> None:
    dt = parse_timestamp(1588687728923)
    assert dt.tzinfo is UTC
    assert dt.year == 2020


def test_timestamp_seconds() -> None:
    dt = parse_timestamp(1588687728)
    assert dt.year == 2020


def test_timestamp_missing() -> None:
    with pytest.raises(NormalizationError):
        parse_timestamp(None)


def test_price_none_string() -> None:
    assert parse_price("None") is None
    assert parse_price(None) is None
    assert parse_price(6.99) == 6.99


def test_normalize_product_brand_from_details_json() -> None:
    product = normalize_product(
        {
            "parent_asin": "B00TEST",
            "title": "Test Cream",
            "main_category": "All Beauty",
            "categories": ["All Beauty", "Skin Care"],
            "description": ["A gentle cream."],
            "features": ["Fragrance free"],
            "price": "None",
            "store": "Acme Store",
            "details": '{"Brand": "Acme", "Size": "2 oz"}',
        }
    )
    assert product.product_id == "B00TEST"
    assert product.brand == "Acme"
    assert product.store == "Acme Store"
    assert product.price is None
    assert product.subcategory == "Skin Care"
    assert "Test Cream" in product.searchable_text
    assert "Acme" in product.searchable_text


def test_normalize_product_requires_title() -> None:
    with pytest.raises(NormalizationError):
        normalize_product({"parent_asin": "B00TEST", "title": ""})


def test_normalize_interaction_is_review_not_purchase() -> None:
    interaction = normalize_interaction(
        {
            "user_id": "U1",
            "parent_asin": "B00TEST",
            "asin": "B00CHILD",
            "rating": 5.0,
            "timestamp": 1588687728923,
            "verified_purchase": True,
            "helpful_vote": 2,
            "title": "Great",
            "text": "Loved it",
        }
    )
    assert interaction.event_type == "review"
    assert interaction.event_value == 5.0
    assert interaction.extras["verified_purchase"] is True
    assert interaction.source_asin == "B00CHILD"


def test_invalid_rating_returns_none_on_value() -> None:
    interaction = normalize_interaction(
        {
            "user_id": "U1",
            "parent_asin": "B00TEST",
            "rating": 9.0,
            "timestamp": 1588687728923,
        }
    )
    assert interaction.event_value is None


def test_searchable_text_field_order() -> None:
    assert TEXT_FIELD_ORDER == ("title", "brand", "category", "features", "description")
    text = build_searchable_text(
        title="Title",
        brand=None,
        category="Cat",
        features="Feat",
        description="Desc",
    )
    assert text.split("\n") == ["Title", "Cat", "Feat", "Desc"]
