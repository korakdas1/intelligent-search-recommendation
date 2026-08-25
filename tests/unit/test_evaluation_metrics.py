"""Hand-built metric tests. No PostgreSQL, MiniLM, or evaluation dumps."""

from __future__ import annotations

import math

import pytest

from app.evaluation.metrics import (
    K_VALUES,
    macro_average,
    mean_reciprocal_rank,
    metric_bundle,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    unique_preserve_order,
)


def test_perfect_rank_one() -> None:
    ranked = ["A", "B", "C"]
    relevant = ["A"]
    assert recall_at_k(ranked, relevant, 5) == 1.0
    assert precision_at_k(ranked, relevant, 5) == pytest.approx(1.0 / 5)
    assert mean_reciprocal_rank(ranked, relevant) == 1.0
    assert ndcg_at_k(ranked, relevant, 5) == pytest.approx(1.0)


def test_hit_at_rank_two() -> None:
    ranked = ["B", "A"]
    relevant = ["A"]
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert precision_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert mean_reciprocal_rank(ranked, relevant) == pytest.approx(0.5)
    expected_dcg = 1.0 / math.log2(3)
    expected_idcg = 1.0 / math.log2(2)
    assert ndcg_at_k(ranked, relevant, 2) == pytest.approx(expected_dcg / expected_idcg)


def test_miss_is_zero() -> None:
    ranked = ["B", "C"]
    relevant = ["A"]
    assert recall_at_k(ranked, relevant, 10) == 0.0
    assert precision_at_k(ranked, relevant, 10) == 0.0
    assert mean_reciprocal_rank(ranked, relevant) == 0.0
    assert ndcg_at_k(ranked, relevant, 10) == 0.0


def test_multiple_relevant_items() -> None:
    ranked = ["A", "X", "B"]
    relevant = ["A", "B"]
    assert recall_at_k(ranked, relevant, 3) == 1.0
    assert precision_at_k(ranked, relevant, 3) == pytest.approx(2.0 / 3)
    assert mean_reciprocal_rank(ranked, relevant) == 1.0
    dcg = 1.0 / math.log2(2) + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, relevant, 3) == pytest.approx(dcg / idcg)


def test_result_shorter_than_k() -> None:
    ranked = ["A"]
    relevant = ["A"]
    assert recall_at_k(ranked, relevant, 10) == 1.0
    assert precision_at_k(ranked, relevant, 10) == pytest.approx(0.1)
    assert mean_reciprocal_rank(ranked, relevant) == 1.0
    assert ndcg_at_k(ranked, relevant, 10) == pytest.approx(1.0)
    assert math.isfinite(ndcg_at_k(ranked, relevant, 10))


def test_duplicate_ids_keep_first_occurrence() -> None:
    ranked = ["A", "A", "B"]
    assert unique_preserve_order(ranked) == ["A", "B"]
    relevant = ["A"]
    assert recall_at_k(ranked, relevant, 2) == 1.0
    assert precision_at_k(ranked, relevant, 2) == pytest.approx(0.5)
    assert mean_reciprocal_rank(["Z", "A", "A"], relevant) == pytest.approx(0.5)


def test_k_equals_one() -> None:
    assert recall_at_k(["A", "B"], ["A"], 1) == 1.0
    assert precision_at_k(["A", "B"], ["A"], 1) == 1.0
    assert ndcg_at_k(["A"], ["A"], 1) == pytest.approx(1.0)
    assert recall_at_k(["B"], ["A"], 1) == 0.0


def test_empty_result_list() -> None:
    relevant = ["A"]
    assert recall_at_k([], relevant, 10) == 0.0
    assert precision_at_k([], relevant, 10) == 0.0
    assert mean_reciprocal_rank([], relevant) == 0.0
    assert ndcg_at_k([], relevant, 10) == 0.0
    bundle = metric_bundle([], relevant)
    assert bundle["mrr"] == 0.0
    assert all(math.isfinite(value) for value in bundle.values())


def test_binary_ndcg_and_no_relevant() -> None:
    assert ndcg_at_k(["A", "B"], [], 5) == 0.0
    assert recall_at_k(["A"], [], 5) == 0.0
    assert math.isfinite(ndcg_at_k(["A"], ["B"], 1))


def test_macro_average_is_deterministic() -> None:
    rows = [
        metric_bundle(["A"], ["A"]),
        metric_bundle(["B", "A"], ["A"]),
    ]
    averaged = macro_average(rows)
    assert averaged["mrr"] == pytest.approx((1.0 + 0.5) / 2)
    assert averaged["recall@5"] == pytest.approx(1.0)
    again = macro_average(rows)
    assert again == averaged
    assert macro_average([]) == {}
    assert K_VALUES == (5, 10, 20)


def test_one_positive_precision_is_hit_over_k() -> None:
    hit = metric_bundle(["T", "X"], ["T"])
    miss = metric_bundle(["X", "Y"], ["T"])
    assert hit["precision@10"] == pytest.approx(0.1)
    assert miss["precision@10"] == 0.0
    assert hit["recall@10"] == 1.0
    assert miss["recall@10"] == 0.0
