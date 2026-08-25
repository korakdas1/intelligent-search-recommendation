"""Deterministic ranking-feature extractor tests. No MiniLM, no PostgreSQL."""

from __future__ import annotations

import json
from math import log1p
from pathlib import Path

import numpy as np
import pytest

from app.ranking.feature_schema import (
    FEATURE_COUNT,
    FEATURE_INDEX,
    FEATURE_NAMES,
    FEATURE_VERSION,
    feature_name_at,
    schema_document,
)
from app.ranking.features import extract_features
from app.ranking.io import feature_frame, read_feature_parquet, write_feature_parquet
from app.ranking.product_view import ProductView
from app.ranking.query_ids import make_query_id
from app.ranking.validate import FeatureValidationError, validate_feature_batch
from app.search.candidates import FusedCandidate
from app.search.fusion import fuse_weighted, rrf_contribution

ROOT = Path(__file__).resolve().parents[2]


def _product(**overrides: object) -> ProductView:
    data = dict(
        product_id="A",
        title="Howard Leather Conditioner",
        brand="Howard",
        category="All Beauty",
        description="A leather care cream.",
        price=12.5,
        average_rating=4.5,
        rating_count=10,
    )
    data.update(overrides)
    return ProductView(**data)  # type: ignore[arg-type]


def _candidate(**overrides: object) -> FusedCandidate:
    data = dict(
        product_id="A",
        fusion_score=0.0,
        keyword_rank=1,
        keyword_score=1.4,
        semantic_rank=3,
        semantic_score=0.6,
        sources=("keyword", "semantic"),
    )
    data.update(overrides)
    return FusedCandidate(**data)  # type: ignore[arg-type]


def test_schema_order_and_version() -> None:
    assert FEATURE_VERSION == "rank-features-v1"
    assert FEATURE_COUNT == 25
    assert FEATURE_NAMES[0] == "keyword_present"
    assert FEATURE_NAMES[9] == "rrf_score"
    assert FEATURE_NAMES[-1] == "log_price"
    assert feature_name_at(7) == "keyword_rank_fraction"
    assert len(set(FEATURE_NAMES)) == FEATURE_COUNT
    with pytest.raises(IndexError):
        feature_name_at(FEATURE_COUNT)


def test_committed_schema_matches_code() -> None:
    path = ROOT / "docs/ranking_feature_schema.json"
    assert json.loads(path.read_text(encoding="utf-8")) == schema_document()


def test_controlled_fixture_leather_conditioner() -> None:
    query = "leather conditioner"
    candidate = _candidate()
    product = _product()
    batch = extract_features(
        query,
        [candidate],
        {"A": product},
        candidate_k=100,
        rrf_k=60,
        keyword_weight=0.5,
    )
    validate_feature_batch(batch)
    row = batch.values[0]
    assert batch.product_ids == ("A",)
    assert row.dtype == np.float32
    assert row[FEATURE_INDEX["keyword_present"]] == 1.0
    assert row[FEATURE_INDEX["semantic_present"]] == 1.0
    assert row[FEATURE_INDEX["retrieval_source_count"]] == 2.0
    assert row[FEATURE_INDEX["keyword_score"]] == pytest.approx(1.4)
    assert row[FEATURE_INDEX["semantic_score"]] == pytest.approx(0.6)
    assert row[FEATURE_INDEX["keyword_rr"]] == pytest.approx(1.0)
    assert row[FEATURE_INDEX["semantic_rr"]] == pytest.approx(1 / 3)
    assert row[FEATURE_INDEX["keyword_rank_fraction"]] == pytest.approx(0.01)
    assert row[FEATURE_INDEX["semantic_rank_fraction"]] == pytest.approx(0.03)
    expected_rrf = rrf_contribution(1, k0=60) + rrf_contribution(3, k0=60)
    assert row[FEATURE_INDEX["rrf_score"]] == pytest.approx(expected_rrf)
    assert row[FEATURE_INDEX["weighted_score"]] == pytest.approx(1.0)
    assert row[FEATURE_INDEX["title_exact_phrase"]] == 1.0
    assert row[FEATURE_INDEX["title_all_query_tokens"]] == 1.0
    assert row[FEATURE_INDEX["title_overlap_count"]] == 2.0
    assert row[FEATURE_INDEX["title_overlap_ratio"]] == 1.0
    assert row[FEATURE_INDEX["brand_present"]] == 1.0
    assert row[FEATURE_INDEX["brand_exact"]] == 0.0
    assert row[FEATURE_INDEX["brand_overlap"]] == 0.0
    assert row[FEATURE_INDEX["description_missing"]] == 0.0
    assert row[FEATURE_INDEX["title_token_count"]] == 3.0
    assert row[FEATURE_INDEX["average_rating"]] == pytest.approx(4.5)
    assert row[FEATURE_INDEX["log_rating_count"]] == pytest.approx(log1p(10))
    assert row[FEATURE_INDEX["price_missing"]] == 0.0
    assert row[FEATURE_INDEX["log_price"]] == pytest.approx(log1p(12.5))
    assert np.isfinite(row).all()


def test_missing_product_fields() -> None:
    product = _product(
        brand=None,
        description=None,
        price=None,
        average_rating=None,
        rating_count=None,
        category=None,
    )
    batch = extract_features(
        "leather",
        [_candidate(semantic_rank=None, semantic_score=None, sources=("keyword",))],
        {"A": product},
        candidate_k=100,
    )
    row = batch.values[0]
    assert row[FEATURE_INDEX["brand_present"]] == 0.0
    assert row[FEATURE_INDEX["brand_exact"]] == 0.0
    assert row[FEATURE_INDEX["brand_overlap"]] == 0.0
    assert row[FEATURE_INDEX["description_missing"]] == 1.0
    assert row[FEATURE_INDEX["description_token_count"]] == 0.0
    assert row[FEATURE_INDEX["price_missing"]] == 1.0
    assert row[FEATURE_INDEX["log_price"]] == 0.0
    assert row[FEATURE_INDEX["average_rating"]] == 0.0
    assert row[FEATURE_INDEX["log_rating_count"]] == 0.0
    assert np.isfinite(row).all()


def test_keyword_only_and_semantic_only() -> None:
    products = {"K": _product(product_id="K"), "S": _product(product_id="S", title="Other")}
    keyword_only = _candidate(
        product_id="K", semantic_rank=None, semantic_score=None, sources=("keyword",)
    )
    semantic_only = _candidate(
        product_id="S",
        keyword_rank=None,
        keyword_score=None,
        semantic_rank=1,
        semantic_score=0.8,
        sources=("semantic",),
    )
    batch = extract_features(
        "leather conditioner",
        [keyword_only, semantic_only],
        products,
        candidate_k=50,
    )
    k_row, s_row = batch.values
    assert k_row[FEATURE_INDEX["keyword_present"]] == 1.0
    assert k_row[FEATURE_INDEX["semantic_present"]] == 0.0
    assert k_row[FEATURE_INDEX["semantic_score"]] == 0.0
    assert k_row[FEATURE_INDEX["semantic_rr"]] == 0.0
    assert s_row[FEATURE_INDEX["keyword_present"]] == 0.0
    assert s_row[FEATURE_INDEX["semantic_present"]] == 1.0
    assert s_row[FEATURE_INDEX["keyword_score"]] == 0.0
    assert s_row[FEATURE_INDEX["rrf_score"]] == pytest.approx(rrf_contribution(1, k0=60))


def test_weighted_reuses_phase6_minmax() -> None:
    a = _candidate(product_id="A", keyword_score=3.0, semantic_score=0.2, semantic_rank=2)
    b = _candidate(
        product_id="B",
        keyword_rank=2,
        keyword_score=1.0,
        semantic_rank=1,
        semantic_score=0.9,
    )
    fused = {row.product_id: row for row in (a, b)}
    expected = {row.product_id: row.fusion_score for row in fuse_weighted(fused, alpha=0.5)}
    batch = extract_features(
        "q",
        [a, b],
        {"A": _product(product_id="A"), "B": _product(product_id="B", title="B")},
        candidate_k=10,
    )
    assert batch.values[0, FEATURE_INDEX["weighted_score"]] == pytest.approx(expected["A"])
    assert batch.values[1, FEATURE_INDEX["weighted_score"]] == pytest.approx(expected["B"])
    assert batch.product_ids == ("A", "B")


def test_preserves_candidate_order() -> None:
    rows = [
        _candidate(product_id="C", keyword_rank=3, keyword_score=0.1),
        _candidate(product_id="A", keyword_rank=1, keyword_score=1.0),
        _candidate(product_id="B", keyword_rank=2, keyword_score=0.5),
    ]
    products = {
        "A": _product(product_id="A"),
        "B": _product(product_id="B"),
        "C": _product(product_id="C"),
    }
    batch = extract_features("leather", rows, products, candidate_k=10)
    assert batch.product_ids == ("C", "A", "B")


def test_query_edge_cases_are_finite() -> None:
    product = _product()
    candidate = _candidate()
    for query in ("LEATHER CONDITIONER", "leather, conditioner!", '"leather"', "the", "   xyzzy   ", "\t"):
        batch = extract_features(query.strip() or "x", [candidate], {"A": product}, candidate_k=10)
        assert np.isfinite(batch.values).all()


def test_validation_rejects_nan_and_wrong_shape() -> None:
    batch = extract_features("q", [_candidate()], {"A": _product()}, candidate_k=10)
    validate_feature_batch(batch)
    bad = batch.values.copy()
    bad[0, 0] = np.nan
    with pytest.raises(FeatureValidationError, match="NaN"):
        validate_feature_batch(
            type(batch)(product_ids=batch.product_ids, values=bad, feature_version=FEATURE_VERSION)
        )


def test_parquet_roundtrip(tmp_path: Path) -> None:
    batch = extract_features("leather conditioner", [_candidate()], {"A": _product()}, candidate_k=100)
    frame = feature_frame(
        query_id=make_query_id("leather conditioner", "issued"),
        query="leather conditioner",
        query_source="issued",
        candidate_rank_start=1,
        batch=batch,
    )
    path = tmp_path / "features.parquet"
    write_feature_parquet(path, frame)
    loaded = read_feature_parquet(path)
    assert len(loaded) == 1
    assert list(loaded["product_id"]) == ["A"]
    assert loaded["feature_version"].iloc[0] == FEATURE_VERSION
    for name in FEATURE_NAMES:
        assert loaded[name].iloc[0] == pytest.approx(float(batch.values[0, FEATURE_INDEX[name]]))


def test_missing_product_raises() -> None:
    with pytest.raises(KeyError):
        extract_features("q", [_candidate()], {}, candidate_k=10)
