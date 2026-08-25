"""Unit tests for ranking lexical helpers. No MiniLM, no PostgreSQL."""

from app.ranking.text_features import (
    all_query_tokens_present,
    exact_phrase_in,
    normalize_phrase,
    overlap_count,
    overlap_ratio,
    tokenize,
    unique_tokens,
)


def test_tokenize_casefold_and_punctuation() -> None:
    assert tokenize("Howard Leather Conditioner 4-Pack!") == [
        "howard",
        "leather",
        "conditioner",
        "4",
        "pack",
    ]
    assert tokenize("  LEATHER   conditioner  ") == ["leather", "conditioner"]
    assert tokenize(None) == []
    assert tokenize("the") == ["the"]
    assert tokenize("xyzzyqwerty12345notaproduct") == ["xyzzyqwerty12345notaproduct"]
    assert tokenize("O'Reilly SPF/50") == ["o", "reilly", "spf", "50"]


def test_normalize_phrase() -> None:
    assert normalize_phrase("  Leather   Conditioner ") == "leather conditioner"
    assert normalize_phrase(None) == ""


def test_exact_phrase_ignores_case() -> None:
    title = "Howard Leather Conditioner 4-Pack"
    assert exact_phrase_in("leather conditioner", title) == 1.0
    assert exact_phrase_in("LEATHER CONDITIONER", title) == 1.0
    assert exact_phrase_in("howard leather", title) == 1.0
    assert exact_phrase_in("leather oil", title) == 0.0
    assert exact_phrase_in("   ", title) == 0.0


def test_all_tokens_need_not_be_contiguous() -> None:
    title = "Howard LC0008 Leather Conditioner"
    assert all_query_tokens_present("howard leather", title) == 1.0
    assert exact_phrase_in("howard leather", title) == 0.0
    assert all_query_tokens_present("howard soap", title) == 0.0
    assert all_query_tokens_present("the", "other words") == 0.0


def test_overlap_ratio_no_divide_by_zero() -> None:
    assert overlap_ratio("   ", "Leather") == 0.0
    assert overlap_count("leather conditioner", "Howard Leather Conditioner") == 2.0
    assert overlap_ratio("leather conditioner", "Howard Leather Conditioner") == 1.0
    assert unique_tokens("leather leather oil") == ["leather", "oil"]
