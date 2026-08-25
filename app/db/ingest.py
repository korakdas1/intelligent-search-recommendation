"""Bulk ingest of processed Parquet into PostgreSQL.

Does not re-parse Amazon JSONL. Orphan interactions (product_id missing
from the product table) are skipped and counted, not inserted.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.events import DATASET_VERSION_FULL, DEFAULT_CURRENCY
from app.models.dataset import DatasetVersion
from app.models.interaction import Interaction
from app.models.product import Product
from app.models.user import User

BATCH_SIZE = 1000


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _parse_json_cell(value: Any) -> dict[str, Any]:
    if value is None or (isinstance(value, float) and value != value):
        return {}
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text_value = value.strip()
        if not text_value:
            return {}
        try:
            parsed = json.loads(text_value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _batched(rows: Sequence[dict[str, Any]], size: int) -> Iterable[Sequence[dict[str, Any]]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def _execute_batches(session: Session, statement_factory, rows: Sequence[dict[str, Any]]) -> None:
    for batch in _batched(rows, BATCH_SIZE):
        session.execute(statement_factory(list(batch)))


def load_manifest(processed_dir: Path) -> dict[str, Any]:
    path = processed_dir / "dataset_manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    docs = Path("docs/dataset_manifest.json")
    if docs.exists():
        return json.loads(docs.read_text(encoding="utf-8"))
    return {}


def register_dataset_version(
    session: Session,
    *,
    dataset_version: str,
    manifest: dict[str, Any],
    row_counts: dict[str, Any],
) -> None:
    source = manifest.get("source", {})
    checksums = manifest.get("checksums", {})
    values = {
        "dataset_version": dataset_version,
        "source_name": source.get("dataset", "Amazon Reviews 2023"),
        "source_slice": source.get("category", "All_Beauty"),
        "source_url": source.get("huggingface") or source.get("source_site"),
        "license_note": source.get("license_note"),
        "row_counts": row_counts,
        "checksums": checksums,
        "created_at": _now_utc(),
    }
    stmt = insert(DatasetVersion).values(values)
    stmt = stmt.on_conflict_do_update(
        index_elements=["dataset_version"],
        set_={
            "source_name": stmt.excluded.source_name,
            "source_slice": stmt.excluded.source_slice,
            "source_url": stmt.excluded.source_url,
            "license_note": stmt.excluded.license_note,
            "row_counts": stmt.excluded.row_counts,
            "checksums": stmt.excluded.checksums,
        },
    )
    session.execute(stmt)


def product_rows_from_frame(frame: pd.DataFrame, dataset_version: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    now = _now_utc()
    for record in frame.itertuples(index=False):
        extras = _parse_json_cell(getattr(record, "extras_json", None))
        rows.append(
            {
                "product_id": str(record.product_id),
                "dataset_version": dataset_version,
                "title": record.title,
                "description": None if pd.isna(record.description) else record.description,
                "category": None if pd.isna(record.category) else record.category,
                "subcategory": None if pd.isna(record.subcategory) else record.subcategory,
                "brand": None if pd.isna(record.brand) else record.brand,
                "store": None if pd.isna(record.store) else record.store,
                "price": None if pd.isna(record.price) else record.price,
                "currency": DEFAULT_CURRENCY,
                "average_rating": None if pd.isna(record.average_rating) else record.average_rating,
                "rating_count": None if pd.isna(record.rating_count) else int(record.rating_count),
                "metadata": extras,
                "created_at": now,
                "updated_at": now,
            }
        )
    return rows


def upsert_products(session: Session, rows: Sequence[dict[str, Any]]) -> None:
    table = Product.__table__

    def factory(batch: list[dict[str, Any]]):
        stmt = insert(table).values(batch)
        update_cols = {
            "dataset_version": stmt.excluded.dataset_version,
            "title": stmt.excluded.title,
            "description": stmt.excluded.description,
            "category": stmt.excluded.category,
            "subcategory": stmt.excluded.subcategory,
            "brand": stmt.excluded.brand,
            "store": stmt.excluded.store,
            "price": stmt.excluded.price,
            "currency": stmt.excluded.currency,
            "average_rating": stmt.excluded.average_rating,
            "rating_count": stmt.excluded.rating_count,
            "metadata": stmt.excluded["metadata"],
            "updated_at": stmt.excluded.updated_at,
        }
        return stmt.on_conflict_do_update(index_elements=["product_id"], set_=update_cols)

    _execute_batches(session, factory, rows)


def upsert_users(session: Session, rows: Sequence[dict[str, Any]]) -> None:
    table = User.__table__

    def factory(batch: list[dict[str, Any]]):
        stmt = insert(table).values(batch)
        return stmt.on_conflict_do_update(
            index_elements=["user_id"],
            set_={
                "dataset_version": stmt.excluded.dataset_version,
                "first_seen_at": func.least(table.c.first_seen_at, stmt.excluded.first_seen_at),
                "last_seen_at": func.greatest(table.c.last_seen_at, stmt.excluded.last_seen_at),
            },
        )

    _execute_batches(session, factory, rows)


def insert_interactions(session: Session, rows: Sequence[dict[str, Any]]) -> None:
    table = Interaction.__table__

    def factory(batch: list[dict[str, Any]]):
        stmt = insert(table).values(batch)
        return stmt.on_conflict_do_nothing(
            index_elements=["user_id", "product_id", "event_type", "occurred_at"]
        )

    _execute_batches(session, factory, rows)


def table_count(session: Session, model) -> int:
    return int(session.scalar(select(func.count()).select_from(model)) or 0)


def ingest_parquet(
    engine: Engine,
    processed_dir: Path,
    *,
    dataset_version: str = DATASET_VERSION_FULL,
    report_dir: Path | None = None,
) -> dict[str, Any]:
    products_path = processed_dir / "products.parquet"
    interactions_path = processed_dir / "interactions.parquet"
    if not products_path.exists():
        raise FileNotFoundError(products_path)
    if not interactions_path.exists():
        raise FileNotFoundError(interactions_path)

    started = time.perf_counter()
    manifest = load_manifest(processed_dir)
    products = pd.read_parquet(
        products_path,
        columns=[
            "product_id",
            "title",
            "description",
            "category",
            "subcategory",
            "brand",
            "store",
            "price",
            "average_rating",
            "rating_count",
            "extras_json",
        ],
    )
    interactions = pd.read_parquet(
        interactions_path,
        columns=[
            "user_id",
            "product_id",
            "source_asin",
            "event_type",
            "event_value",
            "occurred_at",
            "verified_purchase",
            "helpful_vote",
            "extras_json",
        ],
    )

    product_ids = set(products["product_id"].astype(str))
    orphan_mask = ~interactions["product_id"].astype(str).isin(product_ids)
    orphan_frame = interactions.loc[orphan_mask]
    valid = interactions.loc[~orphan_mask]
    orphan_product_ids = sorted(set(orphan_frame["product_id"].astype(str)))

    user_bounds = (
        valid.groupby("user_id", sort=False)["occurred_at"].agg(["min", "max"]).reset_index()
        if len(valid)
        else pd.DataFrame(columns=["user_id", "min", "max"])
    )
    user_rows = [
        {
            "user_id": str(row.user_id),
            "dataset_version": dataset_version,
            "first_seen_at": row.min.to_pydatetime() if hasattr(row.min, "to_pydatetime") else row.min,
            "last_seen_at": row.max.to_pydatetime() if hasattr(row.max, "to_pydatetime") else row.max,
            "metadata": {},
        }
        for row in user_bounds.itertuples(index=False)
    ]

    product_payload = product_rows_from_frame(products, dataset_version)
    interaction_payload: list[dict[str, Any]] = []
    for record in valid.itertuples(index=False):
        extras = _parse_json_cell(getattr(record, "extras_json", None))
        occurred = record.occurred_at
        if hasattr(occurred, "to_pydatetime"):
            occurred = occurred.to_pydatetime()
        extras["source_asin"] = None if pd.isna(record.source_asin) else str(record.source_asin)
        extras["verified_purchase"] = bool(record.verified_purchase) if not pd.isna(record.verified_purchase) else None
        extras["helpful_vote"] = None if pd.isna(record.helpful_vote) else int(record.helpful_vote)
        interaction_payload.append(
            {
                "user_id": str(record.user_id),
                "product_id": str(record.product_id),
                "event_type": record.event_type,
                "event_value": None if pd.isna(record.event_value) else record.event_value,
                "occurred_at": occurred,
                "dataset_version": dataset_version,
                "metadata": extras,
            }
        )

    timings: dict[str, float] = {}
    factory = sessionmaker(bind=engine, autoflush=False, future=True)
    with factory() as session:
        t0 = time.perf_counter()
        register_dataset_version(
            session,
            dataset_version=dataset_version,
            manifest=manifest,
            row_counts={},
        )
        session.flush()
        t1 = time.perf_counter()
        upsert_products(session, product_payload)
        timings["products_seconds"] = round(time.perf_counter() - t1, 3)
        t2 = time.perf_counter()
        upsert_users(session, user_rows)
        timings["users_seconds"] = round(time.perf_counter() - t2, 3)
        t3 = time.perf_counter()
        insert_interactions(session, interaction_payload)
        timings["interactions_seconds"] = round(time.perf_counter() - t3, 3)

        products_count = table_count(session, Product)
        users_count = table_count(session, User)
        interactions_count = table_count(session, Interaction)
        register_dataset_version(
            session,
            dataset_version=dataset_version,
            manifest=manifest,
            row_counts={
                "products": products_count,
                "users": users_count,
                "interactions": interactions_count,
                "orphan_interaction_rows": int(len(orphan_frame)),
                "orphan_product_ids": len(orphan_product_ids),
            },
        )
        session.commit()
        timings["dataset_version_seconds"] = round(t1 - t0, 3)

    elapsed = round(time.perf_counter() - started, 3)
    timings["total_seconds"] = elapsed
    report = {
        "dataset_version": dataset_version,
        "processed_dir": str(processed_dir),
        "source_manifest": {
            "preprocess_version": manifest.get("preprocess_version"),
            "checksums": manifest.get("checksums", {}),
        },
        "products_read": int(len(products)),
        "products_upserted": int(len(product_payload)),
        "users_derived": int(len(user_rows)),
        "interactions_read": int(len(interactions)),
        "interactions_valid": int(len(valid)),
        "interactions_inserted_attempted": int(len(interaction_payload)),
        "orphan_interaction_rows": int(len(orphan_frame)),
        "orphan_product_ids": len(orphan_product_ids),
        "orphan_product_id_list": orphan_product_ids,
        "final_table_counts": {
            "dataset_versions": 1,
            "products": products_count,
            "users": users_count,
            "interactions": interactions_count,
        },
        "elapsed": timings,
        "generated_at_utc": _now_utc().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "idempotency": "ON CONFLICT DO UPDATE for products/users/dataset_versions; DO NOTHING for interactions",
        "orphan_policy": "skip_at_ingest",
    }
    if report_dir is not None:
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "report.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        (report_dir / "orphans.json").write_text(
            json.dumps(
                {
                    "orphan_product_ids": orphan_product_ids,
                    "orphan_interaction_rows": int(len(orphan_frame)),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return report
