"""Semantic product-text construction."""

from app.embeddings.text import build_semantic_text


def test_labeled_fields_in_stable_order() -> None:
    text = build_semantic_text(
        title="Leather Cream",
        brand="Howard",
        category="All Beauty",
        subcategory=None,
        description="Restores dry leather.",
    )
    assert text == (
        "Title: Leather Cream\n"
        "Brand: Howard\n"
        "Category: All Beauty\n"
        "Description: Restores dry leather."
    )


def test_missing_fields_omitted() -> None:
    text = build_semantic_text(title="Soap", brand=None, category=None, description=None)
    assert text == "Title: Soap"
    assert "Brand:" not in text


def test_store_is_not_a_brand_parameter() -> None:
    text = build_semantic_text(title="Oil", brand=None)
    assert "store" not in text.lower()
    assert "Brand:" not in text


def test_whitespace_collapsed() -> None:
    text = build_semantic_text(title="  Foo   Bar  ")
    assert text == "Title: Foo Bar"
