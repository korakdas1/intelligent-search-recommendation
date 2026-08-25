"""Deterministic RRF and weighted-fusion tests. No PostgreSQL or MiniLM."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.search.candidates import (
    FusedCandidate,
    RetrievalCandidate,
    resolve_candidate_k,
    union_candidates,
)
from app.search.fusion import fuse_rrf, fuse_weighted, minmax_normalize, validate_keyword_weight


def _lists() -> tuple[list[RetrievalCandidate], list[RetrievalCandidate]]:
    keyword = [
        RetrievalCandidate("A", 1, 3.0, "keyword"),
        RetrievalCandidate("B", 2, 2.0, "keyword"),
        RetrievalCandidate("C", 3, 1.0, "keyword"),
    ]
    semantic = [
        RetrievalCandidate("C", 1, 0.9, "semantic"),
        RetrievalCandidate("A", 2, 0.6, "semantic"),
        RetrievalCandidate("D", 3, 0.3, "semantic"),
    ]
    return keyword, semantic


def test_rrf_exact_scores_and_order() -> None:
    keyword, semantic = _lists()
    fused = fuse_rrf(union_candidates(keyword, semantic), k0=60)
    by_id = {row.product_id: row.fusion_score for row in fused}
    assert by_id["A"] == pytest.approx(1 / 61 + 1 / 62)
    assert by_id["B"] == pytest.approx(1 / 62)
    assert by_id["C"] == pytest.approx(1 / 63 + 1 / 61)
    assert by_id["D"] == pytest.approx(1 / 63)
    assert [row.product_id for row in fused] == ["A", "C", "B", "D"]


def test_rrf_missing_candidate_still_scores() -> None:
    keyword = [RetrievalCandidate("ONLYK", 1, 1.0, "keyword")]
    semantic = [RetrievalCandidate("ONLYS", 1, 0.8, "semantic")]
    fused = fuse_rrf(union_candidates(keyword, semantic), k0=60)
    by_id = {row.product_id: row for row in fused}
    assert by_id["ONLYK"].fusion_score == pytest.approx(1 / 61)
    assert by_id["ONLYS"].fusion_score == pytest.approx(1 / 61)
    assert by_id["ONLYK"].semantic_rank is None
    assert by_id["ONLYS"].keyword_rank is None


def test_union_deduplicates_product_id() -> None:
    keyword, semantic = _lists()
    merged = union_candidates(keyword, semantic)
    assert set(merged) == {"A", "B", "C", "D"}
    assert merged["A"].keyword_rank == 1
    assert merged["A"].semantic_rank == 2
    assert merged["B"].semantic_rank is None
    assert merged["D"].keyword_rank is None


def test_minmax_normalize_and_equal_scores() -> None:
    assert minmax_normalize([1.0, 3.0, 5.0]) == [0.0, 0.5, 1.0]
    assert minmax_normalize([2.0, 2.0, 2.0]) == [1.0, 1.0, 1.0]
    assert minmax_normalize([]) == []


def test_weighted_both_and_missing_sides() -> None:
    keyword, semantic = _lists()
    fused = fuse_weighted(union_candidates(keyword, semantic), alpha=0.5)
    by_id = {row.product_id: row.fusion_score for row in fused}
    assert by_id["A"] == pytest.approx(0.75)
    assert by_id["B"] == pytest.approx(0.25)
    assert by_id["C"] == pytest.approx(0.5)
    assert by_id["D"] == pytest.approx(0.0)
    assert [row.product_id for row in fused] == ["A", "C", "B", "D"]


def test_weighted_alpha_one_is_normalized_keyword() -> None:
    keyword, semantic = _lists()
    fused = fuse_weighted(union_candidates(keyword, semantic), alpha=1.0)
    assert [row.product_id for row in fused] == ["A", "B", "C", "D"]
    assert fused[0].fusion_score == pytest.approx(1.0)
    assert fused[1].fusion_score == pytest.approx(0.5)
    assert fused[2].fusion_score == pytest.approx(0.0)
    assert fused[3].fusion_score == pytest.approx(0.0)


def test_weighted_alpha_zero_is_normalized_semantic() -> None:
    keyword, semantic = _lists()
    fused = fuse_weighted(union_candidates(keyword, semantic), alpha=0.0)
    assert [row.product_id for row in fused] == ["C", "A", "B", "D"]
    assert fused[0].fusion_score == pytest.approx(1.0)
    assert fused[1].fusion_score == pytest.approx(0.5)


def test_weighted_tie_breaks_by_product_id() -> None:
    rows = {
        "B": FusedCandidate("B", 0.0, keyword_rank=1, keyword_score=1.0),
        "A": FusedCandidate("A", 0.0, keyword_rank=2, keyword_score=1.0),
    }
    fused = fuse_weighted(rows, alpha=1.0)
    assert [row.product_id for row in fused] == ["A", "B"]
    assert fused[0].fusion_score == fused[1].fusion_score == pytest.approx(1.0)


def test_invalid_alpha() -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        validate_keyword_weight(-0.01)
    with pytest.raises(ValueError, match="between 0 and 1"):
        fuse_weighted({}, alpha=1.1)
    with pytest.raises(ValidationError):
        Settings(_env_file=None, hybrid_keyword_weight=1.5)


def test_candidate_k_bounds() -> None:
    assert resolve_candidate_k(10) == 100
    assert resolve_candidate_k(30) == 150
    assert resolve_candidate_k(100) == 500
    assert resolve_candidate_k(1) == 100
    assert resolve_candidate_k(80, maximum=200) == 200
    assert resolve_candidate_k(10, configured=250) == 250
    assert resolve_candidate_k(80, configured=40) == 80
    with pytest.raises(ValueError):
        resolve_candidate_k(0)
    with pytest.raises(ValueError):
        resolve_candidate_k(10, configured=0)
