"""Deterministic product text for semantic embeddings.

Version: ``semantic-text-v1``.

Uses catalog fields already in PostgreSQL. Missing fields are omitted.
Brand is never filled from ``store`` (D-031). Review text, ratings, image
URLs, and interaction history are never included.
"""

from __future__ import annotations

from app.embeddings.constants import MAX_FIELD_CHARS, SEMANTIC_TEXT_VERSION

SEMANTIC_TEXT_SCHEMA = SEMANTIC_TEXT_VERSION
FIELD_LABELS = (
    ("title", "Title"),
    ("brand", "Brand"),
    ("category", "Category"),
    ("subcategory", "Subcategory"),
    ("description", "Description"),
)


def _clip(value: str, limit: int = MAX_FIELD_CHARS) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def build_semantic_text(
    *,
    title: str | None,
    brand: str | None = None,
    category: str | None = None,
    subcategory: str | None = None,
    description: str | None = None,
) -> str:
    """Build labeled semantic text. Empty fields are omitted; store is unused."""

    values = {
        "title": title,
        "brand": brand,
        "category": category,
        "subcategory": subcategory,
        "description": description,
    }
    parts: list[str] = []
    for field, label in FIELD_LABELS:
        raw = values[field]
        if raw is None:
            continue
        clipped = _clip(str(raw))
        if clipped:
            parts.append(f"{label}: {clipped}")
    return "\n".join(parts)
