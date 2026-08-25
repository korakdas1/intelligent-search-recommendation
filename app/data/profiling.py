"""Dataset-neutral quality, join, density, and k-core statistics."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd


USER_THRESHOLDS = (2, 3, 5, 10)
ITEM_THRESHOLDS = (2, 3, 5, 10)


def _pct(part: int, whole: int) -> float:
    if whole == 0:
        return 0.0
    return 100.0 * part / whole


def threshold_counts(counts: pd.Series, thresholds: Sequence[int]) -> dict[str, dict[str, float | int]]:
    total = int(len(counts))
    out: dict[str, dict[str, float | int]] = {}
    for threshold in thresholds:
        n = int((counts >= threshold).sum())
        out[str(threshold)] = {"count": n, "percent": round(_pct(n, total), 4)}
    return out


def interaction_density(
    n_users: int,
    n_items: int,
    n_unique_pairs: int,
) -> dict[str, float | int]:
    denominator = n_users * n_items
    density = (n_unique_pairs / denominator) if denominator else 0.0
    return {
        "users": n_users,
        "items": n_items,
        "unique_pairs": n_unique_pairs,
        "density": density,
        "sparsity": 1.0 - density,
        "density_scientific": f"{density:.6e}",
        "sparsity_percent": f"{(1.0 - density) * 100:.6f}",
    }


def unique_pairs(interactions: pd.DataFrame) -> pd.DataFrame:
    return interactions[["user_id", "product_id"]].drop_duplicates()


def user_item_counts(pairs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    user_counts = pairs.groupby("user_id").size()
    item_counts = pairs.groupby("product_id").size()
    return user_counts, item_counts


def summarize_counts(counts: pd.Series) -> dict[str, float | int]:
    if counts.empty:
        return {"n": 0, "mean": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    return {
        "n": int(len(counts)),
        "mean": float(counts.mean()),
        "median": float(counts.median()),
        "p90": float(counts.quantile(0.90)),
        "p99": float(counts.quantile(0.99)),
        "max": int(counts.max()),
    }


def density_report(interactions: pd.DataFrame) -> dict[str, object]:
    pairs = unique_pairs(interactions)
    user_counts, item_counts = user_item_counts(pairs)
    n_users = int(user_counts.shape[0])
    n_items = int(item_counts.shape[0])
    n_pairs = int(len(pairs))
    return {
        "interaction_rows": int(len(interactions)),
        "unique_user_item_pairs": n_pairs,
        "users": n_users,
        "items": n_items,
        "per_user": summarize_counts(user_counts),
        "per_item": summarize_counts(item_counts),
        "users_at_least": threshold_counts(user_counts, USER_THRESHOLDS),
        "items_at_least": threshold_counts(item_counts, ITEM_THRESHOLDS),
        "matrix": interaction_density(n_users, n_items, n_pairs),
    }


def iterative_kcore(
    pairs: pd.DataFrame,
    min_user: int,
    min_item: int,
) -> pd.DataFrame:
    """Drop users/items below thresholds until the remaining graph is stable."""

    current = pairs[["user_id", "product_id"]].drop_duplicates().copy()
    while True:
        if current.empty:
            return current
        user_ok = current.groupby("user_id")["product_id"].transform("size") >= min_user
        item_ok = current.groupby("product_id")["user_id"].transform("size") >= min_item
        nxt = current[user_ok & item_ok]
        if len(nxt) == len(current):
            return nxt.reset_index(drop=True)
        current = nxt


def kcore_report(
    interactions: pd.DataFrame,
    configs: Iterable[tuple[int, int]],
) -> list[dict[str, object]]:
    base_pairs = unique_pairs(interactions)
    base_n = int(len(base_pairs))
    rows: list[dict[str, object]] = []
    for min_user, min_item in configs:
        filtered = iterative_kcore(base_pairs, min_user, min_item)
        user_counts, item_counts = user_item_counts(filtered) if not filtered.empty else (
            pd.Series(dtype=int),
            pd.Series(dtype=int),
        )
        n_pairs = int(len(filtered))
        n_users = int(user_counts.shape[0])
        n_items = int(item_counts.shape[0])
        rows.append(
            {
                "min_user": min_user,
                "min_item": min_item,
                "users": n_users,
                "items": n_items,
                "unique_pairs": n_pairs,
                "pct_pairs_retained": round(_pct(n_pairs, base_n), 4),
                "per_user": summarize_counts(user_counts),
                "per_item": summarize_counts(item_counts),
                "matrix": interaction_density(n_users, n_items, n_pairs),
            }
        )
    return rows


def temporal_holdout_users(interactions: pd.DataFrame, min_events: int = 2) -> dict[str, int | float]:
    """Users who have at least ``min_events`` timestamped events (train + later)."""

    ordered = interactions.sort_values(["user_id", "occurred_at"])
    sizes = ordered.groupby("user_id").size()
    eligible = int((sizes >= min_events).sum())
    n_users = int(sizes.shape[0])
    return {
        "min_events": min_events,
        "eligible_users": eligible,
        "all_users": n_users,
        "percent": round(_pct(eligible, n_users), 4),
    }


def join_coverage(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
) -> dict[str, int | float]:
    product_ids = set(products["product_id"].astype(str))
    interaction_ids = set(interactions["product_id"].astype(str))
    matched = product_ids & interaction_ids
    n_int = int(len(interactions))
    resolved = int(interactions["product_id"].astype(str).isin(product_ids).sum())
    return {
        "unique_metadata_ids": len(product_ids),
        "unique_interaction_item_ids": len(interaction_ids),
        "ids_in_both": len(matched),
        "interaction_ids_without_metadata": len(interaction_ids - product_ids),
        "metadata_ids_without_interactions": len(product_ids - interaction_ids),
        "interaction_rows": n_int,
        "interaction_rows_with_metadata": resolved,
        "interaction_row_match_percent": round(_pct(resolved, n_int), 4),
    }
