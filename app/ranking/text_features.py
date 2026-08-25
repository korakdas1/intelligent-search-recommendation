"""Lightweight deterministic tokenizer for ranking lexical features.

These tokens are NOT PostgreSQL FTS lexemes. No stemming, synonyms, or typo correction.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[0-9a-z]+", flags=re.UNICODE)
_WHITESPACE = re.compile(r"\s+")


def normalize_phrase(text: str | None) -> str:
    """Casefold and collapse whitespace. Empty/None → empty string."""

    if not text:
        return ""
    return _WHITESPACE.sub(" ", text.casefold()).strip()


def tokenize(text: str | None) -> list[str]:
    """Alphanumeric tokens after Unicode casefold. Order preserved; repeats kept."""

    if not text:
        return []
    return _TOKEN.findall(text.casefold())


def unique_tokens(text: str | None) -> list[str]:
    """Unique tokens in first-seen order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for token in tokenize(text):
        if token not in seen:
            seen.add(token)
            ordered.append(token)
    return ordered


def exact_phrase_in(query: str, haystack: str | None) -> float:
    phrase = normalize_phrase(query)
    if not phrase:
        return 0.0
    return 1.0 if phrase in normalize_phrase(haystack) else 0.0


def overlap_count(query: str, haystack: str | None) -> float:
    query_set = set(unique_tokens(query))
    if not query_set:
        return 0.0
    hay_set = set(unique_tokens(haystack))
    return float(len(query_set & hay_set))


def overlap_ratio(query: str, haystack: str | None) -> float:
    query_set = set(unique_tokens(query))
    if not query_set:
        return 0.0
    hay_set = set(unique_tokens(haystack))
    return float(len(query_set & hay_set) / len(query_set))


def all_query_tokens_present(query: str, haystack: str | None) -> float:
    query_set = set(unique_tokens(query))
    if not query_set:
        return 0.0
    hay_set = set(unique_tokens(haystack))
    return 1.0 if query_set <= hay_set else 0.0
