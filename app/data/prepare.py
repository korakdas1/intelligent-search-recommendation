"""Prepare normalized Parquet tables from Amazon Reviews 2023 JSONL."""

from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.data.amazon_reviews import (
    NormalizationError,
    interaction_to_row,
    normalize_interaction,
    normalize_product,
    product_to_row,
)
from app.data.io import iter_jsonl, sha256_file, write_parquet
from app.data.sampling import build_development_sample

PREPROCESS_VERSION = "phase2-v1"

META_PROFILE_FIELDS = (
    "parent_asin",
    "title",
    "description",
    "features",
    "categories",
    "main_category",
    "price",
    "store",
    "details",
    "average_rating",
    "rating_number",
    "images",
    "bought_together",
    "subtitle",
    "author",
)
REVIEW_PROFILE_FIELDS = (
    "user_id",
    "asin",
    "parent_asin",
    "rating",
    "title",
    "text",
    "timestamp",
    "sort_timestamp",
    "verified_purchase",
    "helpful_vote",
    "helpful_votes",
    "images",
)


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return value == value
    if isinstance(value, str):
        return value.strip().lower() not in {"", "none", "null", "nan", "n/a"}
    if isinstance(value, (list, dict, tuple, set)):
        return len(value) > 0
    return True


def _value_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "str"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "dict"
    return type(value).__name__


def update_raw_field_stats(
    stats: dict[str, Any],
    record: dict[str, Any],
    fields: tuple[str, ...],
) -> None:
    stats["rows"] = int(stats.get("rows", 0)) + 1
    stats.setdefault("all_keys", set()).update(record.keys())
    type_samples: dict[str, list[str]] = stats.setdefault("value_types", {})
    for field_name in fields:
        bucket = stats.setdefault("fields", {}).setdefault(
            field_name, {"present": 0, "missing": 0}
        )
        value = record.get(field_name)
        if field_name in record and _is_present(value):
            bucket["present"] += 1
        else:
            bucket["missing"] += 1
        observed = type_samples.setdefault(field_name, [])
        type_name = _value_type_name(value) if field_name in record else "absent"
        if type_name not in observed and len(observed) < 6:
            observed.append(type_name)


def finalize_raw_field_stats(stats: dict[str, Any]) -> dict[str, Any]:
    rows = int(stats.get("rows", 0))
    fields_out: dict[str, Any] = {}
    for name, bucket in stats.get("fields", {}).items():
        present = int(bucket["present"])
        missing = int(bucket["missing"])
        fields_out[name] = {
            "present": present,
            "missing": missing,
            "missing_percent": round(100.0 * missing / rows, 4) if rows else 0.0,
            "value_types": stats.get("value_types", {}).get(name, []),
        }
    return {
        "rows": rows,
        "all_keys": sorted(stats.get("all_keys", [])),
        "fields": fields_out,
    }


def normalize_products(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    seen: dict[str, int] = {}
    raw_stats: dict[str, Any] = {}
    raw_rows = 0
    for record in iter_jsonl(path):
        raw_rows += 1
        update_raw_field_stats(raw_stats, record, META_PROFILE_FIELDS)
        if raw_rows % 50_000 == 0:
            print(f"products scanned: {raw_rows}", flush=True)
        try:
            product = normalize_product(record)
        except NormalizationError as exc:
            rejects[str(exc)] += 1
            continue
        if product.product_id in seen:
            rejects["duplicate_parent_asin"] += 1
            continue
        seen[product.product_id] = 1
        rows.append(product_to_row(product))
    rejects["raw_rows"] = raw_rows
    frame = pd.DataFrame(rows)
    return frame, dict(rejects), finalize_raw_field_stats(raw_stats)


def normalize_interactions(
    path: Path,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rejects: Counter[str] = Counter()
    exact_keys: set[tuple[str, str, str, str]] = set()
    raw_stats: dict[str, Any] = {}
    raw_rows = 0
    for record in iter_jsonl(path):
        raw_rows += 1
        update_raw_field_stats(raw_stats, record, REVIEW_PROFILE_FIELDS)
        if raw_rows % 100_000 == 0:
            print(f"reviews scanned: {raw_rows}", flush=True)
        try:
            interaction = normalize_interaction(record)
        except NormalizationError as exc:
            rejects[str(exc)] += 1
            continue
        if interaction.event_value is None:
            rejects["invalid_or_missing_rating"] += 1
            continue
        key = (
            interaction.user_id,
            interaction.product_id,
            interaction.event_type,
            interaction.occurred_at.isoformat(),
        )
        if key in exact_keys:
            rejects["exact_duplicate_user_item_time"] += 1
            continue
        exact_keys.add(key)
        rows.append(interaction_to_row(interaction))
    rejects["raw_rows"] = raw_rows
    frame = pd.DataFrame(rows)
    if not frame.empty and not pd.api.types.is_datetime64_any_dtype(frame["occurred_at"]):
        frame["occurred_at"] = pd.to_datetime(frame["occurred_at"], utc=True, format="ISO8601")
    return frame, dict(rejects), finalize_raw_field_stats(raw_stats)


def coverage(series: pd.Series) -> dict[str, float | int]:
    n = int(len(series))
    nonempty = series.notna() & (series.astype(str).str.strip() != "")
    present = int(nonempty.sum())
    return {
        "n": n,
        "present": present,
        "missing": n - present,
        "missing_percent": round(100.0 * (n - present) / n, 4) if n else 0.0,
    }


def build_manifest(
    *,
    category: str,
    meta_path: Path,
    review_path: Path,
    products: pd.DataFrame,
    interactions: pd.DataFrame,
    product_rejects: dict[str, int],
    interaction_rejects: dict[str, int],
    product_raw_stats: dict[str, Any],
    review_raw_stats: dict[str, Any],
    sample_seed: int,
    sample_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "preprocess_version": PREPROCESS_VERSION,
        "source": {
            "dataset": "Amazon Reviews 2023",
            "repo_id": "McAuley-Lab/Amazon-Reviews-2023",
            "category": category,
            "metadata_file": meta_path.name,
            "reviews_file": review_path.name,
            "source_site": "https://amazon-reviews-2023.github.io/main.html",
            "huggingface": "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023",
            "citation": "Hou et al., Bridging Language and Items for Retrieval and Recommendation, arXiv:2403.03952 (2024)",
            "license_note": "McAuley Lab: no SPDX license assigned; made available primarily for research.",
            "acquired_at_utc": datetime.now(tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        },
        "checksums": {
            "metadata_sha256": sha256_file(meta_path),
            "reviews_sha256": sha256_file(review_path),
            "metadata_bytes": meta_path.stat().st_size,
            "reviews_bytes": review_path.stat().st_size,
        },
        "canonical_product_id": "parent_asin",
        "timestamp_field": "timestamp",
        "timestamp_unit": "unix_milliseconds",
        "event_type": "review",
        "verified_purchase": "stored as interaction metadata; not duplicated as a purchase event",
        "raw_field_stats": {"metadata": product_raw_stats, "reviews": review_raw_stats},
        "rejects": {"products": product_rejects, "interactions": interaction_rejects},
        "processed": {
            "products": int(len(products)),
            "interactions": int(len(interactions)),
            "users": int(interactions["user_id"].nunique()) if len(interactions) else 0,
            "interaction_items": int(interactions["product_id"].nunique()) if len(interactions) else 0,
        },
        "coverage": {
            "title": coverage(products["title"]) if len(products) else {},
            "description": coverage(products["description"]) if len(products) else {},
            "category": coverage(products["category"]) if len(products) else {},
            "brand": coverage(products["brand"]) if len(products) else {},
            "store": coverage(products["store"]) if len(products) else {},
            "price": coverage(products["price"]) if len(products) else {},
            "searchable_text": coverage(products["searchable_text"]) if len(products) else {},
        },
        "sample": {"seed": sample_seed, **sample_counts},
    }


def prepare(
    category: str,
    raw_dir: Path,
    processed_dir: Path,
    sample_dir: Path,
    sample_seed: int = 42,
) -> dict[str, Any]:
    meta_path = raw_dir / f"meta_{category}.jsonl"
    review_path = raw_dir / f"{category}.jsonl"
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    if not review_path.exists():
        raise FileNotFoundError(review_path)

    products, product_rejects, product_raw_stats = normalize_products(meta_path)
    interactions, interaction_rejects, review_raw_stats = normalize_interactions(review_path)

    processed_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    products_path = processed_dir / "products.parquet"
    interactions_path = processed_dir / "interactions.parquet"
    write_parquet(products, products_path)
    write_parquet(interactions, interactions_path)

    sample_products, sample_interactions = build_development_sample(
        products, interactions, seed=sample_seed
    )
    write_parquet(sample_products, sample_dir / "products.parquet")
    write_parquet(sample_interactions, sample_dir / "interactions.parquet")

    sample_counts = {
        "products": int(len(sample_products)),
        "interactions": int(len(sample_interactions)),
        "users": int(sample_interactions["user_id"].nunique()) if len(sample_interactions) else 0,
    }
    manifest = build_manifest(
        category=category,
        meta_path=meta_path,
        review_path=review_path,
        products=products,
        interactions=interactions,
        product_rejects=product_rejects,
        interaction_rejects=interaction_rejects,
        product_raw_stats=product_raw_stats,
        review_raw_stats=review_raw_stats,
        sample_seed=sample_seed,
        sample_counts=sample_counts,
    )
    manifest_path = processed_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    docs_manifest = Path("docs/dataset_manifest.json")
    public = {
        key: manifest[key]
        for key in (
            "preprocess_version",
            "source",
            "canonical_product_id",
            "timestamp_field",
            "timestamp_unit",
            "event_type",
            "verified_purchase",
            "processed",
            "coverage",
            "sample",
            "checksums",
            "rejects",
            "raw_field_stats",
        )
    }
    docs_manifest.write_text(json.dumps(public, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
