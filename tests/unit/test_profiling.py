"""Density and k-core tests on tiny synthetic frames."""

import pandas as pd

from app.data.profile_report import decide_cf_viability
from app.data.profiling import (
    density_report,
    interaction_density,
    iterative_kcore,
    join_coverage,
    temporal_holdout_users,
)
from app.data.sampling import build_development_sample


def _interactions() -> pd.DataFrame:
    rows = [
        ("u1", "p1", "2020-01-01T00:00:00+00:00"),
        ("u1", "p2", "2020-02-01T00:00:00+00:00"),
        ("u2", "p1", "2020-01-15T00:00:00+00:00"),
        ("u3", "p3", "2020-03-01T00:00:00+00:00"),
        ("u1", "p1", "2020-04-01T00:00:00+00:00"),
    ]
    return pd.DataFrame(rows, columns=["user_id", "product_id", "occurred_at"])


def test_density_known_values() -> None:
    stats = interaction_density(n_users=2, n_items=4, n_unique_pairs=3)
    assert stats["density"] == 3 / 8
    assert abs(stats["sparsity"] - 0.625) < 1e-12


def test_density_report_counts() -> None:
    report = density_report(_interactions())
    assert report["interaction_rows"] == 5
    assert report["unique_user_item_pairs"] == 4
    assert report["users"] == 3
    assert report["items"] == 3
    assert report["users_at_least"]["2"]["count"] == 1
    assert report["users_at_least"]["3"]["count"] == 0


def test_kcore_2_2_drops_singletons() -> None:
    pairs = _interactions()[["user_id", "product_id"]].drop_duplicates()
    core = iterative_kcore(pairs, 2, 2)
    # u3-p3 is a leaf; after it drops, remaining degrees fall below 2.
    assert core.empty


def test_temporal_holdout() -> None:
    stats = temporal_holdout_users(_interactions(), min_events=2)
    assert stats["eligible_users"] == 1
    assert stats["all_users"] == 3


def test_decide_cf_viability_two_core_is_b() -> None:
    profile = {
        "interactions": {
            "users_at_least": {"2": {"count": 40000}},
            "per_user": {"median": 1.0},
        },
        "kcore": [
            {"min_user": 2, "min_item": 2, "unique_pairs": 50000, "users": 20000},
            {"min_user": 5, "min_item": 5, "unique_pairs": 2000, "users": 200},
        ],
    }
    decision = decide_cf_viability(profile)
    assert decision["code"] == "B"
    assert "filtering" in decision["label"].lower()
    products = pd.DataFrame({"product_id": ["p1", "p2"]})
    interactions = pd.DataFrame(
        {"user_id": ["u1", "u2"], "product_id": ["p1", "p9"]}
    )
    stats = join_coverage(products, interactions)
    assert stats["unique_metadata_ids"] == 2
    assert stats["unique_interaction_item_ids"] == 2
    assert stats["ids_in_both"] == 1
    assert stats["interaction_ids_without_metadata"] == 1
    assert stats["metadata_ids_without_interactions"] == 1
    assert stats["interaction_rows_with_metadata"] == 1


def test_sample_is_deterministic() -> None:
    products = pd.DataFrame(
        {
            "product_id": ["p1", "p2", "p3", "p4"],
            "title": ["a", "b", "c", "d"],
        }
    )
    interactions = _interactions()
    first_p, first_i = build_development_sample(
        products, interactions, seed=42, repeat_users=10, singleton_users=10, extra_products=1
    )
    second_p, second_i = build_development_sample(
        products, interactions, seed=42, repeat_users=10, singleton_users=10, extra_products=1
    )
    assert list(first_p["product_id"]) == list(second_p["product_id"])
    assert list(first_i["user_id"]) == list(second_i["user_id"])
    assert "p1" in set(first_p["product_id"])
    assert "u1" in set(first_i["user_id"])
