"""Deterministic synthetic search queries. Not human relevance labels."""

from __future__ import annotations

from dataclasses import dataclass

from app.ranking.constants import LABEL_CLASS, QUERY_SOURCE_ATTRIBUTE, QUERY_SOURCE_TITLE
from app.ranking.query_ids import make_ltr_query_id
from app.ranking.text_features import tokenize, unique_tokens

# Catalog noise and units. Not PostgreSQL stopwords; documented LTR-only filter.
UNINFORMATIVE_TOKENS = frozenset(
    {
        "a",
        "an",
        "and",
        "by",
        "ct",
        "count",
        "fl",
        "for",
        "from",
        "in",
        "inch",
        "inches",
        "jar",
        "jars",
        "kit",
        "lb",
        "lbs",
        "ml",
        "new",
        "of",
        "on",
        "or",
        "ounce",
        "ounces",
        "oz",
        "pack",
        "pc",
        "pcs",
        "piece",
        "pieces",
        "set",
        "size",
        "sizes",
        "the",
        "to",
        "unit",
        "units",
        "with",
    }
)


@dataclass(frozen=True, slots=True)
class SyntheticQuery:
    query_id: str
    query: str
    query_source: str
    label_class: str
    source_product_id: str
    relevant_ids: tuple[str, ...]


def _has_digit(token: str) -> bool:
    return any(char.isdigit() for char in token)


def informative_title_tokens(title: str) -> list[str]:
    """Unique title tokens with units, stopwords, and SKU-like tokens removed."""

    kept: list[str] = []
    seen: set[str] = set()
    for token in tokenize(title):
        if token in UNINFORMATIVE_TOKENS or len(token) < 2 or _has_digit(token):
            continue
        if token not in seen:
            seen.add(token)
            kept.append(token)
    return kept


def title_query_from_tokens(tokens: list[str]) -> str | None:
    """Last two informative tokens, preserving order. None if fewer than two remain."""

    if len(tokens) < 2:
        return None
    return " ".join(tokens[-2:])


def make_title_query(product_id: str, title: str) -> SyntheticQuery | None:
    query_text = title_query_from_tokens(informative_title_tokens(title))
    if query_text is None:
        return None
    return SyntheticQuery(
        query_id=make_ltr_query_id(QUERY_SOURCE_TITLE, product_id, query_text),
        query=query_text,
        query_source=QUERY_SOURCE_TITLE,
        label_class=LABEL_CLASS,
        source_product_id=product_id,
        relevant_ids=(product_id,),
    )


def make_attribute_query(product_id: str, title: str, brand: str | None) -> SyntheticQuery | None:
    """Brand first token + last informative title token. Source product is the only positive.

    Catalog-wide brand+type expansion is not used: many brands span unrelated SKUs.
    """

    if not brand or not brand.strip():
        return None
    brand_tokens = unique_tokens(brand)
    if not brand_tokens:
        return None
    title_tokens = [token for token in informative_title_tokens(title) if token not in set(brand_tokens)]
    if not title_tokens:
        return None
    type_token = title_tokens[-1]
    query_text = f"{brand_tokens[0]} {type_token}"
    title_query = title_query_from_tokens(informative_title_tokens(title))
    if title_query is not None and query_text == title_query:
        return None
    return SyntheticQuery(
        query_id=make_ltr_query_id(QUERY_SOURCE_ATTRIBUTE, product_id, query_text),
        query=query_text,
        query_source=QUERY_SOURCE_ATTRIBUTE,
        label_class=LABEL_CLASS,
        source_product_id=product_id,
        relevant_ids=(product_id,),
    )
