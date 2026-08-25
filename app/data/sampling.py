"""Deterministic relationship-aware development samples."""

from __future__ import annotations

import pandas as pd


def build_development_sample(
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    *,
    seed: int = 42,
    repeat_users: int = 200,
    singleton_users: int = 100,
    extra_products: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep users, their interactions, and the products those interactions need.

    Repeat users are sampled first so the sample is useful for later CF tests.
    Extra catalog products are added so missing-interaction items still appear.
    """

    rng = seed
    pair_counts = interactions.groupby("user_id")["product_id"].nunique()
    repeats = pair_counts[pair_counts >= 2].index.to_series(name="user_id")
    singles = pair_counts[pair_counts == 1].index.to_series(name="user_id")

    n_repeat = min(repeat_users, int(len(repeats)))
    n_single = min(singleton_users, int(len(singles)))
    chosen_repeats = repeats.sample(n=n_repeat, random_state=rng) if n_repeat else repeats
    chosen_singles = singles.sample(n=n_single, random_state=rng + 1) if n_single else singles
    chosen_users = pd.Index(chosen_repeats).union(pd.Index(chosen_singles))

    sample_interactions = interactions[interactions["user_id"].isin(chosen_users)].copy()
    needed_products = set(sample_interactions["product_id"].astype(str))

    remaining = products[~products["product_id"].astype(str).isin(needed_products)]
    n_extra = min(extra_products, int(len(remaining)))
    extra = remaining.sample(n=n_extra, random_state=rng + 2) if n_extra else remaining.iloc[0:0]
    keep_ids = needed_products | set(extra["product_id"].astype(str))
    sample_products = products[products["product_id"].astype(str).isin(keep_ids)].copy()
    return sample_products.reset_index(drop=True), sample_interactions.reset_index(drop=True)
