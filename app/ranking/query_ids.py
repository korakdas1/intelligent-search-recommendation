"""Deterministic query IDs for ranking-feature rows and LTR groups."""

from __future__ import annotations

import hashlib

from app.ranking.text_features import normalize_phrase


def make_query_id(query: str, source: str) -> str:
    payload = f"{source}\n{normalize_phrase(query)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def make_ltr_query_id(query_source: str, source_product_id: str, query: str) -> str:
    payload = f"{query_source}\n{source_product_id}\n{normalize_phrase(query)}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]


def make_personalized_query_id(
    user_id: str,
    target_product_id: str,
    query_generator_version: str,
    query: str,
) -> str:
    payload = (
        f"{user_id}\n{target_product_id}\n{query_generator_version}\n{normalize_phrase(query)}"
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:16]
