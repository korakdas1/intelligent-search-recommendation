"""hybrid-rec-v1 unit tests. No PostgreSQL, MiniLM, or CF training."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.recommendations.hybrid_candidates import topk_ids_from_scores
from app.recommendations.hybrid_constants import (
    CHANNEL_CF,
    CHANNEL_CONTENT,
    CHANNEL_POPULARITY,
    DEFAULT_RRF_K,
    REASON_NO_PERSONALIZED_HISTORY,
)
from app.recommendations.hybrid_fusion import (
    fuse_recommendation_rrf,
    fuse_recommendation_weighted,
    renormalize_channel_weights,
    resolve_hybrid_candidate_k,
    union_recommendation_channels,
    validate_channel_weights,
)
from app.recommendations.hybrid_policy import resolve_channel_availability
from app.recommendations.ranking import ScoredItem
from app.recommendations.service import validate_user_method
from app.search.fusion import rrf_contribution


def _scored(ids_scores: list[tuple[str, float]]) -> list[ScoredItem]:
    return [ScoredItem(product_id=product_id, score=score) for product_id, score in ids_scores]


def _reference_topk(product_ids, scores, *, exclude: set[str], top_k: int) -> list[ScoredItem]:
    if top_k < 1:
        return []
    items = [
        ScoredItem(str(product_id), float(score))
        for product_id, score in zip(product_ids, np.asarray(scores, dtype=np.float64), strict=True)
        if str(product_id) not in exclude and np.isfinite(score)
    ]
    return sorted(items, key=lambda item: (-item.score, item.product_id))[:top_k]


def test_offline_topk_cutoff_tie_chooses_lexicographically_first_ids() -> None:
    ids = ["P30", "P28", "P27", "P24", "P19", "P16", "P06", "P02"]
    scores = np.ones(len(ids))
    expected = _scored([(product_id, 1.0) for product_id in ["P02", "P06", "P16", "P19", "P24"]])
    assert _reference_topk(ids, scores, exclude=set(), top_k=5) == expected
    assert topk_ids_from_scores(ids, scores, exclude=set(), top_k=5) == expected


@pytest.mark.parametrize("top_k", [1, 3, 5, 8, 12])
@pytest.mark.parametrize(
    "scores",
    [
        [1, 1, 1, 1, 1, 1, 1, 1],
        [3, 3, 3, 2, 1, 0, -1, -2],
        [3, 2, 2, 2, 2, 2, 1, 0],
        [3, 3, 2, 2, 1, 1, 0, 0],
    ],
    ids=["all-equal", "tie-ends-at-k", "tie-spans-k", "multiple-groups"],
)
def test_offline_topk_ties_are_independent_of_input_order(scores, top_k: int) -> None:
    ids = np.array(["P08", "P02", "P07", "P01", "P05", "P03", "P06", "P04"])
    values = np.asarray(scores, dtype=np.float64)
    expected = _reference_topk(ids, values, exclude=set(), top_k=top_k)
    for order in (np.arange(8), np.arange(8)[::-1], np.random.default_rng(42).permutation(8)):
        assert topk_ids_from_scores(ids[order], values[order], exclude=set(), top_k=top_k) == expected


@pytest.mark.parametrize("exclude", [{"A", "B"}, {"HIGH"}, {"HIGH", "A", "B"}])
@pytest.mark.parametrize("top_k", [1, 3, 20])
def test_offline_topk_exclusion_paths_refill_ties_identically(exclude: set[str], top_k: int) -> None:
    ids = np.array(["HIGH", "E", "C", "A", "D", "B", "LOW"])
    scores = np.array([9, 1, 1, 1, 1, 1, -1], dtype=np.float64)
    indices = np.array([index for index, product_id in enumerate(ids) if product_id in exclude])
    expected = _reference_topk(ids, scores, exclude=exclude, top_k=top_k)
    assert topk_ids_from_scores(ids, scores, exclude=exclude, top_k=top_k) == expected
    assert topk_ids_from_scores(ids, scores, exclude=set(), exclude_indices=indices, top_k=top_k) == expected
    assert topk_ids_from_scores(ids, scores, exclude=exclude, exclude_indices=indices, top_k=top_k) == expected


def test_offline_topk_retains_exclusion_indices_precedence_and_empty_fallback() -> None:
    ids = ["C", "B", "A"]
    scores = np.ones(3)
    assert topk_ids_from_scores(
        ids, scores, exclude={"A"}, exclude_indices=np.array([0]), top_k=2,
    ) == _scored([("A", 1.0), ("B", 1.0)])
    assert topk_ids_from_scores(
        ids, scores, exclude={"A"}, exclude_indices=np.array([], dtype=np.int64), top_k=2,
    ) == _scored([("B", 1.0), ("C", 1.0)])


@pytest.mark.parametrize("top_k", [1, 3, 5, 20])
def test_offline_topk_only_finite_scores_preserving_float64_values(top_k: int) -> None:
    ids = ["NAN", "POS_INF", "NEG_INF", "ZERO_B", "NEG_B", "ZERO_A", "NEG_A", "PRECISE"]
    scores = np.array([np.nan, np.inf, -np.inf, 0.0, -2.5, -0.0, -2.5, 1.0000000000000002])
    original = scores.copy()
    result = topk_ids_from_scores(ids, scores, exclude=set(), top_k=top_k)
    assert result == _reference_topk(ids, scores, exclude=set(), top_k=top_k)
    assert result[0].score == 1.0000000000000002
    np.testing.assert_array_equal(scores, original)


@pytest.mark.parametrize("top_k", [0, -1])
def test_offline_topk_nonpositive_k(top_k: int) -> None:
    assert topk_ids_from_scores(["A"], np.array([1.0]), exclude=set(), top_k=top_k) == []


def test_offline_topk_empty_or_no_valid_candidates() -> None:
    assert topk_ids_from_scores([], np.array([]), exclude=set(), top_k=5) == []
    assert topk_ids_from_scores(["A"], np.ones(1), exclude={"A"}, top_k=5) == []
    assert topk_ids_from_scores(
        ["A", "B", "C"], np.array([np.nan, np.inf, -np.inf]), exclude=set(), top_k=5,
    ) == []


def test_offline_topk_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="product_ids and scores length mismatch"):
        topk_ids_from_scores(["A"], np.ones(2), exclude=set(), top_k=1)


@pytest.mark.parametrize("seed", [0, 7, 19, 42, 2026])
def test_offline_topk_randomized_reference_parity(seed: int) -> None:
    rng = np.random.default_rng(seed)
    for count in (1, 8, 37, 128):
        ids = np.array([f"P{index:04d}" for index in rng.permutation(count)])
        scores = rng.choice([-3.5, -1.0, 0.0, 0.25, 1.0, 2.0], size=count)
        indices = np.flatnonzero(rng.random(count) < 0.25)
        exclude = set(ids[indices])
        original = scores.copy()
        for top_k in (0, 1, 3, count // 2, count, count + 5):
            expected = _reference_topk(ids, scores, exclude=exclude, top_k=top_k)
            assert topk_ids_from_scores(ids, scores, exclude=exclude, top_k=top_k) == expected
            assert topk_ids_from_scores(
                ids, scores, exclude=set(), exclude_indices=indices, top_k=top_k,
            ) == expected
        np.testing.assert_array_equal(scores, original)


def _prompt_lists() -> tuple[list[ScoredItem], list[ScoredItem], list[ScoredItem]]:
    content = _scored([("A", 0.9), ("B", 0.8), ("C", 0.7)])
    cf = _scored([("C", 4.0), ("A", 2.0), ("D", -1.0)])
    popularity = _scored([("A", 1900.0), ("D", 50.0), ("E", 10.0)])
    return content, cf, popularity


def test_rrf_prompt_fixture_exact_scores_and_order() -> None:
    content, cf, popularity = _prompt_lists()
    fused = fuse_recommendation_rrf(union_recommendation_channels(content, cf, popularity), k0=60)
    by_id = {row.product_id: row for row in fused}
    assert by_id["A"].fused_score == pytest.approx(1 / 61 + 1 / 62 + 1 / 61)
    assert by_id["B"].fused_score == pytest.approx(1 / 62)
    assert by_id["C"].fused_score == pytest.approx(1 / 63 + 1 / 61)
    assert by_id["D"].fused_score == pytest.approx(1 / 63 + 1 / 62)
    assert by_id["E"].fused_score == pytest.approx(1 / 63)
    assert [row.product_id for row in fused] == ["A", "C", "D", "B", "E"]
    assert by_id["A"].content_rank == 1
    assert by_id["A"].cf_rank == 2
    assert by_id["A"].popularity_rank == 1
    assert by_id["B"].cf_rank is None
    assert by_id["E"].content_rank is None


def test_rrf_ignores_raw_score_magnitudes() -> None:
    content = _scored([("A", 0.01), ("B", 0.009)])
    cf = _scored([("A", 1000.0), ("B", 999.0)])
    popularity = _scored([("A", 2.0), ("B", 1.0)])
    fused = fuse_recommendation_rrf(union_recommendation_channels(content, cf, popularity), k0=60)
    by_id = {row.product_id: row.fused_score for row in fused}
    assert by_id["A"] == pytest.approx(1 / 61 + 1 / 61 + 1 / 61)
    assert by_id["B"] == pytest.approx(1 / 62 + 1 / 62 + 1 / 62)
    assert fused[0].product_id == "A"


def test_rrf_one_two_three_channels_and_missing() -> None:
    only = fuse_recommendation_rrf(union_recommendation_channels(_scored([("A", 1.0)]), None, None), k0=60)
    assert only[0].fused_score == pytest.approx(1 / 61)
    two = fuse_recommendation_rrf(
        union_recommendation_channels(_scored([("A", 1.0)]), _scored([("B", 1.0)]), None),
        k0=60,
    )
    assert {row.product_id for row in two} == {"A", "B"}
    assert all(row.fused_score == pytest.approx(1 / 61) for row in two)


def test_union_keeps_channel_exclusive_items() -> None:
    content = _scored([("A", 1.0)])
    cf = _scored([("B", 2.0)])
    popularity = _scored([("C", 9.0)])
    merged = union_recommendation_channels(content, cf, popularity)
    assert set(merged) == {"A", "B", "C"}
    assert merged["A"].content_present and not merged["A"].cf_present
    assert merged["B"].cf_present and not merged["B"].content_present
    assert merged["C"].popularity_present and not merged["C"].content_present


def test_union_deduplicates_and_keeps_all_contributions() -> None:
    content = _scored([("X", 0.5)])
    cf = _scored([("X", 3.0)])
    popularity = _scored([("X", 12.0)])
    merged = union_recommendation_channels(content, cf, popularity)
    assert list(merged) == ["X"]
    row = merged["X"]
    assert row.content_raw_score == 0.5
    assert row.cf_raw_score == 3.0
    assert row.popularity_raw_score == 12.0
    assert row.sources == (CHANNEL_CONTENT, CHANNEL_CF, CHANNEL_POPULARITY)


def test_union_does_not_restrict_to_cf_universe() -> None:
    content = _scored([("COLD", 0.8), ("WARM", 0.2)])
    cf = _scored([("WARM", 1.0)])
    popularity = _scored([("POP", 4.0)])
    merged = union_recommendation_channels(content, cf, popularity)
    assert "COLD" in merged
    assert merged["COLD"].cf_present is False
    fused = fuse_recommendation_rrf(merged, k0=60)
    assert "COLD" in {row.product_id for row in fused}


def test_weighted_minmax_and_missing_candidate() -> None:
    content, cf, popularity = _prompt_lists()
    weights = validate_channel_weights(1 / 3, 1 / 3, 1 / 3)
    fused, effective = fuse_recommendation_weighted(
        union_recommendation_channels(content, cf, popularity),
        weights,
        available_channels=(CHANNEL_CONTENT, CHANNEL_CF, CHANNEL_POPULARITY),
    )
    assert effective[CHANNEL_CONTENT] == pytest.approx(1 / 3)
    by_id = {row.product_id: row.fused_score for row in fused}
    # content minmax: C=0, B=0.5, A=1; CF: D=0, A=(2-(-1))/5=0.6, C=1; pop: E=0, D≈0.021, A=1
    assert by_id["B"] == pytest.approx((1 / 3) * 0.5)
    assert by_id["E"] == pytest.approx(0.0)
    assert all(math.isfinite(score) for score in by_id.values())
    assert fused[0].product_id == "A"


def test_minmax_one_candidate_equal_scores_negative_cf() -> None:
    content = _scored([("ONLY", 0.4)])
    cf = _scored([("NEG", -5.0), ("POS", -1.0)])
    popularity = _scored([("P1", 7.0), ("P2", 7.0)])
    fused, _effective = fuse_recommendation_weighted(
        union_recommendation_channels(content, cf, popularity),
        validate_channel_weights(0.5, 0.3, 0.2),
        available_channels=(CHANNEL_CONTENT, CHANNEL_CF, CHANNEL_POPULARITY),
    )
    by_id = {row.product_id: row for row in fused}
    assert by_id["ONLY"].fused_score == pytest.approx(0.5)
    assert by_id["NEG"].fused_score == pytest.approx(0.0)
    assert by_id["POS"].fused_score == pytest.approx(0.3)
    assert by_id["P1"].fused_score == pytest.approx(0.2)
    assert by_id["P2"].fused_score == pytest.approx(0.2)
    assert [row.product_id for row in fused if row.fused_score == pytest.approx(0.2)] == ["P1", "P2"]


def test_weight_validation() -> None:
    with pytest.raises(ValueError, match=">= 0"):
        validate_channel_weights(-0.1, 0.6, 0.5)
    with pytest.raises(ValueError, match="sum to 1"):
        validate_channel_weights(0.5, 0.5, 0.5)
    assert validate_channel_weights(0.6, 0.3, 0.1).content == 0.6


def test_unavailable_channel_renormalizes_not_missing_candidate() -> None:
    scaled = renormalize_channel_weights(
        validate_channel_weights(0.6, 0.3, 0.1),
        available=(CHANNEL_CONTENT, CHANNEL_POPULARITY),
    )
    assert scaled[CHANNEL_CONTENT] == pytest.approx(0.6 / 0.7)
    assert scaled[CHANNEL_POPULARITY] == pytest.approx(0.1 / 0.7)
    assert CHANNEL_CF not in scaled
    assert sum(scaled.values()) == pytest.approx(1.0)

    content = _scored([("A", 1.0), ("B", 0.0)])
    popularity = _scored([("A", 10.0), ("C", 0.0)])
    fused, effective = fuse_recommendation_weighted(
        union_recommendation_channels(content, None, popularity),
        validate_channel_weights(0.6, 0.3, 0.1),
        available_channels=(CHANNEL_CONTENT, CHANNEL_POPULARITY),
    )
    assert CHANNEL_CF not in effective
    by_id = {row.product_id: row.fused_score for row in fused}
    # B is missing from popularity: pop contribution 0, not a channel drop.
    assert by_id["B"] == pytest.approx((0.6 / 0.7) * 0.0)
    assert by_id["C"] == pytest.approx((0.1 / 0.7) * 0.0)
    assert by_id["A"] == pytest.approx(1.0)


def test_weighted_does_not_add_raw_incompatible_scores() -> None:
    content = _scored([("A", 0.72), ("B", 0.10)])
    cf = _scored([("A", 2.1), ("B", 2.0)])
    popularity = _scored([("A", 1.0), ("B", 1900.0)])
    fused, _effective = fuse_recommendation_weighted(
        union_recommendation_channels(content, cf, popularity),
        validate_channel_weights(0.5, 0.3, 0.2),
        available_channels=(CHANNEL_CONTENT, CHANNEL_CF, CHANNEL_POPULARITY),
    )
    by_id = {row.product_id: row.fused_score for row in fused}
    raw_a = 0.72 + 2.1 + 1.0
    raw_b = 0.10 + 2.0 + 1900.0
    assert raw_b > raw_a
    assert by_id["A"] > by_id["B"]


def test_deterministic_tie_product_id_asc() -> None:
    content = _scored([("B", 1.0)])
    cf = _scored([("A", 9.0)])
    fused = fuse_recommendation_rrf(union_recommendation_channels(content, cf, None), k0=60)
    assert fused[0].fused_score == pytest.approx(fused[1].fused_score)
    assert [row.product_id for row in fused] == ["A", "B"]


def test_candidate_k_policy() -> None:
    assert resolve_hybrid_candidate_k(10) == 100
    assert resolve_hybrid_candidate_k(30) == 150
    assert resolve_hybrid_candidate_k(200) == 500
    assert resolve_hybrid_candidate_k(1, configured=80) == 80
    with pytest.raises(ValueError):
        resolve_hybrid_candidate_k(0)


def test_fallback_policies() -> None:
    warm = resolve_channel_availability(
        content_usable_history=True,
        semantic_artifact_available=True,
        cf_artifact_available=True,
        cf_user_in_model=True,
    )
    assert warm.channels_used == (CHANNEL_CONTENT, CHANNEL_CF, CHANNEL_POPULARITY)
    assert warm.fallback_reason is None

    outside_cf = resolve_channel_availability(
        content_usable_history=True,
        semantic_artifact_available=True,
        cf_artifact_available=True,
        cf_user_in_model=False,
    )
    assert outside_cf.channels_used == (CHANNEL_CONTENT, CHANNEL_POPULARITY)
    assert outside_cf.fallback_reason is None

    cold = resolve_channel_availability(
        content_usable_history=False,
        semantic_artifact_available=True,
        cf_artifact_available=True,
        cf_user_in_model=False,
    )
    assert cold.channels_used == (CHANNEL_POPULARITY,)
    assert cold.fallback_reason == REASON_NO_PERSONALIZED_HISTORY

    missing_cf = resolve_channel_availability(
        content_usable_history=True,
        semantic_artifact_available=True,
        cf_artifact_available=False,
        cf_user_in_model=False,
    )
    assert missing_cf.channels_used == (CHANNEL_CONTENT, CHANNEL_POPULARITY)
    assert missing_cf.fallback_reason == "cf_artifact_unavailable"

    missing_content = resolve_channel_availability(
        content_usable_history=True,
        semantic_artifact_available=False,
        cf_artifact_available=True,
        cf_user_in_model=True,
    )
    assert missing_content.channels_used == (CHANNEL_CF, CHANNEL_POPULARITY)
    assert missing_content.fallback_reason == "content_artifact_unavailable"

    unknown_not_handled_here = resolve_channel_availability(
        content_usable_history=False,
        semantic_artifact_available=False,
        cf_artifact_available=False,
        cf_user_in_model=False,
        popularity_available=False,
    )
    assert unknown_not_handled_here.channels_used == ()


def test_user_method_allows_hybrid_keeps_default_content() -> None:
    assert validate_user_method("hybrid") == "hybrid"
    assert validate_user_method("content") == "content"
    assert validate_user_method("cf") == "cf"
    assert validate_user_method("") == "content"
    with pytest.raises(ValueError):
        validate_user_method("als")


def test_rrf_contribution_helper_matches_formula() -> None:
    assert rrf_contribution(1, k0=DEFAULT_RRF_K) == pytest.approx(1 / 61)
    assert rrf_contribution(None) == 0.0


def test_hybrid_tune_protocol_reuses_cf_tune_and_keeps_hidden_out() -> None:
    from app.recommendations.cf_data import EvalUserRecord, RecsysEvalSplit, hidden_pairs_checksum
    from app.recommendations.cf_tune import assert_tune_leakage_free, build_cf_tune_split
    from app.recommendations.hybrid_constants import TUNE_PROTOCOL
    from app.recommendations.hybrid_eval import coverage_flags, fuse_channel_lists, rank_hidden_in_fused

    split = RecsysEvalSplit(
        evaluation_version="fixture",
        users=(
            EvalUserRecord("u1", ("p1", "p2", "p3"), "p9"),
            EvalUserRecord("u2", ("p1",), "p8"),
        ),
        two_core_pairs=6,
        evaluation_users=2,
        hidden_checksum=hidden_pairs_checksum([("u1", "p9"), ("u2", "p8")]),
        catalog_size=10,
    )
    tune = build_cf_tune_split(split, protocol=TUNE_PROTOCOL)
    assert_tune_leakage_free(split, tune)
    assert tune.protocol == TUNE_PROTOCOL
    assert ("u1", "p3") not in set(tune.inner_train_pairs)
    assert "p3" not in tune.validation_users[0].inner_train_product_ids
    assert "p9" not in tune.validation_users[0].inner_train_product_ids

    hidden = "p3"
    content = _scored([(hidden, 0.9), ("Z", 0.1)])
    cf = _scored([("Y", 1.0)])
    popularity = _scored([("X", 4.0)])
    flags = coverage_flags(hidden, content, cf, popularity)
    assert flags["content_candidate"] is True
    assert flags["cf_candidate"] is False
    assert flags["union_candidate"] is True
    ranked = fuse_channel_lists(content, cf, popularity, fusion_method="rrf", rrf_k=60)
    assert rank_hidden_in_fused(hidden, ranked) is not None
    assert hidden not in set(tune.validation_users[0].inner_train_product_ids)
