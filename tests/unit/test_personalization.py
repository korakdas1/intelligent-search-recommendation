"""personalized-search-v1 unit tests. No PostgreSQL, MiniLM, or CF training."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.search.fusion import minmax_normalize
from app.search.personalization_constants import GAMMA_MAX, PERSONALIZATION_VERSION
from app.search.personalization_features import (
    brand_affinity_for_candidates,
    cf_affinity_for_candidates,
    minmax_channel,
    query_relevance_scores,
)
from app.search.personalization_policy import load_personalization_policy
from app.search.personalization_rerank import (
    gated_final_score,
    inject_forbidden,
    renormalize_signal_weights,
    rerank_candidates,
    validate_gamma,
    validate_signal_weights,
)
from app.search.personalization_types import BaselineCandidate
from app.schemas.search import SearchRequest
from pydantic import ValidationError


def _base(ids_scores: list[tuple[str, float]]) -> list[BaselineCandidate]:
    return [
        BaselineCandidate(product_id=product_id, baseline_score=score, baseline_rank=index, source="hybrid")
        for index, (product_id, score) in enumerate(ids_scores, start=1)
    ]


def test_candidate_identity_preserved() -> None:
    baseline = _base([("A", 1.0), ("B", 0.5), ("C", 0.1)])
    outcome = rerank_candidates(
        baseline,
        content_raw={"A": 0.1, "B": 0.9, "C": 0.2},
        cf_raw={"A": 1.0, "B": -1.0, "C": 0.0},
        weights=validate_signal_weights(0.7, 0.3),
        gamma=0.30,
    )
    ids = [row.product_id for row in outcome.candidates]
    assert set(ids) == {"A", "B", "C"}
    assert len(ids) == 3


def test_no_candidate_injection() -> None:
    baseline = _base([("SHAMPOO1", 0.9), ("SHAMPOO2", 0.8)])
    outcome = inject_forbidden(baseline, "BEARD_OIL")
    assert "BEARD_OIL" not in {row.product_id for row in outcome.candidates}
    assert set(row.product_id for row in outcome.candidates) == {"SHAMPOO1", "SHAMPOO2"}


def test_gamma_cap_rejected() -> None:
    with pytest.raises(ValueError, match="gamma must be <= 0.3"):
        validate_gamma(0.31, version=PERSONALIZATION_VERSION)


def test_gamma_zero_reproduces_baseline_order() -> None:
    baseline = _base([("A", 5.0), ("B", 3.0), ("C", 1.0)])
    outcome = rerank_candidates(
        baseline,
        content_raw={"A": 0.0, "B": 1.0, "C": 0.5},
        cf_raw={"A": 0.0, "B": 1.0, "C": 0.5},
        weights=validate_signal_weights(0.7, 0.3),
        gamma=0.0,
    )
    assert [row.product_id for row in outcome.candidates] == ["A", "B", "C"]


def test_weak_query_cannot_dominate() -> None:
    baseline = _base([("A", 10.0), ("B", 0.01)])
    outcome = rerank_candidates(
        baseline,
        content_raw={"A": 0.0, "B": 1.0},
        cf_raw=None,
        weights=validate_signal_weights(1.0, 0.0),
        gamma=0.30,
    )
    by_id = {row.product_id: row for row in outcome.candidates}
    assert by_id["A"].query_relevance == pytest.approx(1.0)
    assert by_id["B"].query_relevance == pytest.approx(0.0)
    assert by_id["B"].preference == pytest.approx(1.0)
    assert by_id["A"].final_score > by_id["B"].final_score
    assert outcome.candidates[0].product_id == "A"


def test_cf_affinity_dot_product() -> None:
    def score_fn(user_index: int, item_indices: list[int]) -> list[float]:
        user = np.array([1.0, 0.0])
        items = {0: np.array([0.5, 0.0]), 1: np.array([0.0, 2.0])}
        del user_index
        return [float(user @ items[index]) for index in item_indices]

    available, scores = cf_affinity_for_candidates(
        user_id="U1",
        candidate_ids=["P0", "P1", "P_COLD"],
        user_to_index={"U1": 0},
        product_to_index={"P0": 0, "P1": 1},
        score_fn=score_fn,
    )
    assert available is True
    assert scores["P0"] == pytest.approx(0.5)
    assert scores["P1"] == pytest.approx(0.0)
    assert "P_COLD" not in scores


def test_cf_cold_candidate_contributes_zero() -> None:
    baseline = _base([("WARM", 0.8), ("COLD", 0.7)])
    outcome = rerank_candidates(
        baseline,
        content_raw={"WARM": 0.2, "COLD": 0.2},
        cf_raw={"WARM": 5.0},
        weights=validate_signal_weights(0.5, 0.5),
        gamma=0.20,
    )
    by_id = {row.product_id: row for row in outcome.candidates}
    assert by_id["COLD"].cf_available == 0
    assert by_id["COLD"].cf_affinity == pytest.approx(0.0)
    assert "COLD" in {row.product_id for row in outcome.candidates}


def test_user_outside_cf_uses_content() -> None:
    baseline = _base([("A", 0.6), ("B", 0.4)])
    outcome = rerank_candidates(
        baseline,
        content_raw={"A": 0.1, "B": 0.9},
        cf_raw=None,
        weights=validate_signal_weights(0.7, 0.3),
        gamma=0.20,
    )
    assert outcome.signals_used == ("content",)
    assert outcome.effective_weights["content"] == pytest.approx(1.0)
    assert "cf" not in outcome.effective_weights
    assert outcome.applied is True


def test_cold_user_returns_baseline() -> None:
    baseline = _base([("A", 0.9), ("B", 0.2)])
    outcome = rerank_candidates(
        baseline,
        content_raw=None,
        cf_raw=None,
        weights=validate_signal_weights(0.7, 0.3),
        gamma=0.20,
    )
    assert outcome.applied is False
    assert outcome.reason == "no_personalized_history"
    assert [row.product_id for row in outcome.candidates] == ["A", "B"]


def test_missing_cf_artifact_degrades_to_content() -> None:
    baseline = _base([("A", 1.0), ("B", 0.5)])
    outcome = rerank_candidates(
        baseline,
        content_raw={"A": 0.0, "B": 1.0},
        cf_raw=None,
        weights=validate_signal_weights(0.7, 0.3),
        gamma=0.10,
    )
    assert "cf" not in outcome.signals_used
    assert all(math.isfinite(row.final_score) for row in outcome.candidates)


def test_normalization_negative_cf_and_equal_content() -> None:
    values, flags = minmax_channel(
        ["NEG", "POS", "MISS"],
        {"NEG": -5.0, "POS": -1.0},
        channel_available=True,
    )
    assert values[0] == pytest.approx(0.0)
    assert values[1] == pytest.approx(1.0)
    assert values[2] == pytest.approx(0.0)
    assert flags == [1, 1, 0]
    equal, equal_flags = minmax_channel(["A", "B"], {"A": 0.4, "B": 0.4}, channel_available=True)
    assert equal == [1.0, 1.0]
    assert equal_flags == [1, 1]
    unavailable, unavailable_flags = minmax_channel(["A"], {}, channel_available=False)
    assert unavailable == [0.0]
    assert unavailable_flags == [0]


def test_query_relevance_degenerate_uses_rank() -> None:
    scores, degenerate = query_relevance_scores([2.0, 2.0, 2.0], [1, 2, 3])
    assert degenerate is True
    assert scores[0] == pytest.approx(1.0)
    assert scores[-1] == pytest.approx(1 / 3)
    assert all(math.isfinite(value) for value in scores)
    one, _ = query_relevance_scores([7.0], [1])
    assert one == [1.0]


def test_weight_renormalization_when_cf_unavailable() -> None:
    weights = validate_signal_weights(0.7, 0.3)
    effective = renormalize_signal_weights(weights, ["content"])
    assert effective["content"] == pytest.approx(1.0)
    assert "cf" not in effective


def test_cf_only_policy_falls_back_to_content() -> None:
    weights = validate_signal_weights(0.0, 1.0)
    effective = renormalize_signal_weights(weights, ["content"])
    assert effective["content"] == pytest.approx(1.0)
    assert "cf" not in effective


def test_tie_break_is_deterministic() -> None:
    baseline = _base([("B", 1.0), ("A", 1.0)])
    first = rerank_candidates(
        baseline,
        content_raw={"A": 0.5, "B": 0.5},
        cf_raw=None,
        weights=validate_signal_weights(1.0, 0.0),
        gamma=0.10,
    )
    second = rerank_candidates(
        baseline,
        content_raw={"A": 0.5, "B": 0.5},
        cf_raw=None,
        weights=validate_signal_weights(1.0, 0.0),
        gamma=0.10,
    )
    assert [row.product_id for row in first.candidates] == [row.product_id for row in second.candidates]


def test_gated_formula() -> None:
    assert gated_final_score(0.5, 1.0, 0.2) == pytest.approx(0.5 * 1.2)
    assert gated_final_score(0.0, 1.0, 0.3) == pytest.approx(0.0)


def test_cf_user_absent() -> None:
    available, scores = cf_affinity_for_candidates(
        user_id="NOPE",
        candidate_ids=["P0"],
        user_to_index={"U1": 0},
        product_to_index={"P0": 0},
        score_fn=lambda _u, items: [1.0] * len(items),
    )
    assert available is False
    assert scores == {}


def test_brand_affinity_diagnostic() -> None:
    available, scores = brand_affinity_for_candidates(
        ["Acme", "Acme", "Howard"],
        {"P1": "Acme", "P2": "Other", "P3": None},
        ["P1", "P2", "P3"],
    )
    assert available is True
    assert scores["P1"] == pytest.approx(2 / 3)
    assert scores["P2"] == pytest.approx(0.0)
    assert "P3" not in scores


def test_minmax_helper_equal_is_one() -> None:
    assert minmax_normalize([3.0, 3.0]) == [1.0, 1.0]


def test_request_requires_user_id_for_bounded() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="baby shampoo", retrieval_mode="hybrid", personalization_mode="bounded")
    parsed = SearchRequest(
        query="baby shampoo",
        retrieval_mode="hybrid",
        personalization_mode="bounded",
        user_id=" U1 ",
    )
    assert parsed.user_id == "U1"


def test_user_id_without_personalization_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="cream", user_id="U1")


def test_keyword_personalization_rejected() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(
            query="cream",
            retrieval_mode="keyword",
            personalization_mode="bounded",
            user_id="U1",
        )


def test_default_search_request_unchanged() -> None:
    parsed = SearchRequest(query="leather")
    assert parsed.personalization_mode == "none"
    assert parsed.user_id is None
    assert parsed.retrieval_mode == "keyword"


def test_gamma_max_constant() -> None:
    assert GAMMA_MAX == pytest.approx(0.30)
    policy = load_personalization_policy()
    assert policy.personalization_version == PERSONALIZATION_VERSION
    assert policy.gamma <= GAMMA_MAX


def test_all_outputs_finite() -> None:
    baseline = _base([("ONLY", 0.4)])
    outcome = rerank_candidates(
        baseline,
        content_raw={"ONLY": float("nan")},
        cf_raw={"ONLY": float("-inf")},
        weights=validate_signal_weights(0.6, 0.4),
        gamma=0.05,
    )
    row = outcome.candidates[0]
    assert math.isfinite(row.final_score)
    assert math.isfinite(row.preference)
    assert math.isfinite(row.query_relevance)
