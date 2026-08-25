"""Deterministic searchable product text. No embeddings."""

from __future__ import annotations


TEXT_FIELD_ORDER = ("title", "brand", "category", "features", "description")
MAX_FIELD_CHARS = 4000
SEPARATOR = "\n"


def _clip(value: str, limit: int = MAX_FIELD_CHARS) -> str:
    text = " ".join(value.split())
    if len(text) <= limit:
        return text
    return text[:limit].rstrip()


def build_searchable_text(
    *,
    title: str | None,
    brand: str | None = None,
    category: str | None = None,
    features: str | None = None,
    description: str | None = None,
) -> str:
    """Join available fields in a fixed order. Empty fields are omitted."""

    values = {
        "title": title,
        "brand": brand,
        "category": category,
        "features": features,
        "description": description,
    }
    parts: list[str] = []
    for name in TEXT_FIELD_ORDER:
        raw = values[name]
        if raw is None:
            continue
        clipped = _clip(str(raw))
        if clipped:
            parts.append(clipped)
    return SEPARATOR.join(parts)
