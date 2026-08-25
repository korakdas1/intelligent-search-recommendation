"""Content-rec-v1 unit tests. No MiniLM, no full catalog, no training."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest
from pydantic import ValidationError

from app.embeddings.normalize import l2_normalize
from app.recommendations.content import similar_items, user_content_profile
from app.recommendations.evaluation import (
    history_size_bin,
    iterative_two_core,
    leave_last_product_split,
    rank_to_metrics,
)
from app.recommendations.history import unique_product_ids
from app.recommendations.popularity import rank_popularity_counts
from app.recommendations.ranking import rank_of_item
from app.recommendations.service import validate_user_method
from app.recommendations.vectors import mean_profile
from app.schemas.recommendations import RecommendationQuery
from app.search.faiss_index import build_flat_ip_index
from app.search.runtime import SemanticRuntime


def _runtime(matrix: np.ndarray, ids: list[str]) -> SemanticRuntime:
    vectors = l2_normalize(np.asarray(matrix, dtype=np.float32))
    return SemanticRuntime(
        artifact_version="test",
        dataset_version="test",
        model_name="fake",
        embedding_dim=vectors.shape[1],
        product_ids=np.array(ids, dtype="U16"),
        index=build_flat_ip_index(vectors),
        backend="flat",
        embedding_manifest={},
        index_manifest={},
        skip_catalog_count_check=True,
    )


def test_similar_excludes_source() -> None:
    runtime = _runtime(np.eye(3, dtype=np.float32), ["A", "B", "C"])
    results = similar_items(runtime, "A", top_k=5)
    assert "A" not in {item.product_id for item in results}
    assert {item.product_id for item in results} == {"B", "C"}


def test_similar_nearest_neighbor() -> None:
    docs = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.99, 0.01, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    runtime = _runtime(docs, ["SRC", "NEAR", "FAR"])
    ranked = similar_items(runtime, "SRC", top_k=1)
    assert ranked[0].product_id == "NEAR"


def test_profile_is_normalized_mean() -> None:
    a = l2_normalize(np.array([1.0, 0.0], dtype=np.float32))
    b = l2_normalize(np.array([0.0, 1.0], dtype=np.float32))
    expected = mean_profile(np.stack([a, b]))
    runtime = _runtime(np.stack([a, b, np.array([0.0, 1.0], dtype=np.float32)]), ["A", "B", "C"])
    profile, kept = user_content_profile(runtime, ["A", "B"])
    assert kept == ["A", "B"]
    assert profile is not None
    assert np.allclose(profile, expected, atol=1e-5)


def test_history_dedup_one_embedding() -> None:
    class Row:
        def __init__(self, product_id: str) -> None:
            self.product_id = product_id

    ids = unique_product_ids([Row("A"), Row("A"), Row("B")])  # type: ignore[list-item]
    assert ids == ["A", "B"]
    runtime = _runtime(np.eye(2, dtype=np.float32), ["A", "B"])
    _profile, kept = user_content_profile(runtime, ["A", "A", "B"])
    assert kept == ["A", "B"]


def test_seen_items_excluded_from_user_recs() -> None:
    runtime = _runtime(np.eye(3, dtype=np.float32), ["A", "B", "C"])
    profile, kept = user_content_profile(runtime, ["A"])
    assert profile is not None
    from app.recommendations.content import content_items_for_profile

    ranked = content_items_for_profile(runtime, profile, exclude=set(kept), top_k=5)
    assert "A" not in {item.product_id for item in ranked}


def test_popularity_counts_and_ties() -> None:
    ranked = rank_popularity_counts({"B": 3, "A": 3, "C": 1}, top_k=3)
    assert [item.product_id for item in ranked] == ["A", "B", "C"]
    ranked = rank_popularity_counts({"X": 5, "Y": 2}, top_k=1)
    assert ranked[0].product_id == "X"


def test_train_safe_popularity_ignores_held_out_counts() -> None:
    train = {"A": 1, "B": 1}
    # Product X is popular only in the held-out slice.
    full = {"A": 1, "B": 1, "X": 50}
    train_ranked = rank_popularity_counts(train, top_k=10)
    full_ranked = rank_popularity_counts(full, top_k=10)
    assert [item.product_id for item in train_ranked][0] in {"A", "B"}
    assert full_ranked[0].product_id == "X"
    assert "X" not in {item.product_id for item in train_ranked}


def test_leave_last_out_hides_latest_product() -> None:
    t1 = datetime(2020, 1, 1, tzinfo=UTC)
    t2 = datetime(2020, 2, 1, tzinfo=UTC)
    t3 = datetime(2020, 3, 1, tzinfo=UTC)
    users = {"u1": [("A", t1), ("A", t2), ("C", t3), ("B", t2)]}
    split = leave_last_product_split(users)
    assert len(split) == 1
    assert split[0].hidden_product_id == "C"
    assert set(split[0].train_product_ids) == {"A", "B"}
    assert "C" not in split[0].train_product_ids


def test_hidden_item_not_in_seen_filter() -> None:
    seen = {"A", "B"}
    hidden = "C"
    rank = rank_of_item(
        hidden,
        0.5,
        other_ids=["A", "B", "D"],
        other_scores=[0.9, 0.8, 0.4],
    )
    # D is worse; A/B would beat C but they are still "other". For evaluation
    # callers must omit seen ids. This test documents the helper.
    assert rank == 3
    rank_eval = rank_of_item(hidden, 0.5, other_ids=["D"], other_scores=[0.4])
    assert rank_eval == 1


def test_two_core_drops_degree_one() -> None:
    pairs = [
        ("u1", "p1"),
        ("u1", "p2"),
        ("u2", "p1"),
        ("u2", "p2"),
        ("u3", "p3"),
    ]
    core = iterative_two_core(pairs)
    assert ("u3", "p3") not in core
    assert core == {("u1", "p1"), ("u1", "p2"), ("u2", "p1"), ("u2", "p2")}


def test_rank_to_metrics_hit_at_one() -> None:
    metrics = rank_to_metrics(1, 100)
    assert metrics["mrr"] == 1.0
    assert metrics["recall@5"] == 1.0
    assert metrics["ndcg@5"] == 1.0


def test_rank_to_metrics_miss() -> None:
    metrics = rank_to_metrics(None, 100)
    assert metrics["mrr"] == 0.0
    assert metrics["recall@10"] == 0.0


def test_history_size_bins() -> None:
    assert history_size_bin(1) == "train_size=1"
    assert history_size_bin(3) == "train_size=2-4"
    assert history_size_bin(9) == "train_size>=5"


def test_top_k_bounds() -> None:
    with pytest.raises(ValidationError):
        RecommendationQuery(top_k=0)
    with pytest.raises(ValidationError):
        RecommendationQuery(top_k=101)
    assert RecommendationQuery(top_k=1).top_k == 1
    assert RecommendationQuery(top_k=100).top_k == 100


def test_user_method_allows_hybrid_and_cf() -> None:
    assert validate_user_method("hybrid") == "hybrid"
    assert validate_user_method("cf") == "cf"
    assert validate_user_method("content") == "content"
    assert validate_user_method("CONTENT") == "content"
