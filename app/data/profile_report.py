"""Write the density/quality profile from processed Parquet."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.prepare import coverage
from app.data.profiling import (
    density_report,
    join_coverage,
    kcore_report,
    temporal_holdout_users,
    unique_pairs,
)

KCORE_CONFIGS = ((1, 1), (2, 1), (2, 2), (3, 2), (3, 3), (5, 2), (5, 5))


def _rating_distribution(interactions: pd.DataFrame) -> dict[str, int]:
    if interactions.empty:
        return {}
    counts = interactions["event_value"].value_counts().sort_index()
    return {str(float(k)): int(v) for k, v in counts.items()}


def build_profile(products: pd.DataFrame, interactions: pd.DataFrame) -> dict[str, Any]:
    density = density_report(interactions)
    pairs = unique_pairs(interactions)
    repeated_user_item = int(len(interactions) - len(pairs))
    ts_min = interactions["occurred_at"].min() if len(interactions) else None
    ts_max = interactions["occurred_at"].max() if len(interactions) else None
    return {
        "generated_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "products": {
            "rows": int(len(products)),
            "unique_ids": int(products["product_id"].nunique()) if len(products) else 0,
            "coverage": {
                "title": coverage(products["title"]) if len(products) else {},
                "description": coverage(products["description"]) if len(products) else {},
                "category": coverage(products["category"]) if len(products) else {},
                "subcategory": coverage(products["subcategory"]) if len(products) else {},
                "brand": coverage(products["brand"]) if len(products) else {},
                "store": coverage(products["store"]) if len(products) else {},
                "price": coverage(products["price"]) if len(products) else {},
                "searchable_text": coverage(products["searchable_text"]) if len(products) else {},
            },
            "searchable_text_chars": {
                "mean": float(products["searchable_text"].str.len().mean()) if len(products) else 0.0,
                "median": float(products["searchable_text"].str.len().median()) if len(products) else 0.0,
                "p90": float(products["searchable_text"].str.len().quantile(0.9)) if len(products) else 0.0,
            },
        },
        "interactions": {
            **density,
            "repeated_user_item_extra_rows": repeated_user_item,
            "rating_distribution": _rating_distribution(interactions),
            "verified_purchase_true": int(interactions["verified_purchase"].fillna(False).astype(bool).sum())
            if len(interactions)
            else 0,
            "timestamp_min": ts_min.isoformat() if ts_min is not None else None,
            "timestamp_max": ts_max.isoformat() if ts_max is not None else None,
        },
        "join": join_coverage(products, interactions),
        "kcore": kcore_report(interactions, KCORE_CONFIGS),
        "temporal_holdout": {
            "min_2_events": temporal_holdout_users(interactions, 2),
            "min_3_events": temporal_holdout_users(interactions, 3),
        },
    }


def decide_cf_viability(profile: dict[str, Any]) -> dict[str, str]:
    """Map measured density onto D-020 outcomes A / B / C."""

    users_ge2 = int(profile["interactions"]["users_at_least"]["2"]["count"])
    kcores = { (row["min_user"], row["min_item"]): row for row in profile["kcore"] }
    core22 = kcores.get((2, 2), {})
    core55 = kcores.get((5, 5), {})
    pairs_22 = int(core22.get("unique_pairs") or 0)
    users_22 = int(core22.get("users") or 0)
    pairs_55 = int(core55.get("unique_pairs") or 0)

    if users_ge2 < 1000 or pairs_22 < 5000:
        verdict = "C"
        label = "Not suitable"
        detail = (
            "Too few users with repeated history (or a 2-core that is too small) "
            "for a meaningful collaborative-filtering experiment."
        )
    elif pairs_55 >= 20000 and int(core55.get("users") or 0) >= 2000:
        verdict = "B"
        label = "Suitable with controlled filtering"
        detail = (
            "Raw All_Beauty is sparse, but a 5-core (or similar) retains a usable "
            "repeated-history subset."
        )
    elif pairs_22 >= 10000 and users_22 >= 1500:
        verdict = "B"
        label = "Suitable with controlled filtering"
        detail = (
            "Raw All_Beauty is sparse. A light 2-core retains enough repeated "
            "user-item history for a constrained CF experiment. A 5-core is optional "
            "and may be small."
        )
    else:
        verdict = "C"
        label = "Not suitable"
        detail = "Even after light filtering, repeated history remains too weak."

    # Outcome A would require raw mean/median history that is already CF-ready.
    per_user_median = float(profile["interactions"]["per_user"]["median"])
    if per_user_median >= 3 and users_ge2 >= 10000:
        verdict = "A"
        label = "Suitable"
        detail = "Unfiltered All_Beauty already has substantial repeated user history."

    return {"code": verdict, "label": label, "detail": detail}
