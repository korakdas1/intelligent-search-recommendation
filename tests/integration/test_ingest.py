"""Parquet ingest idempotency and orphan skipping."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.ingest import ingest_parquet
from app.models.interaction import Interaction
from app.models.product import Product
from app.models.user import User

pytestmark = pytest.mark.postgres


def _write_fixture(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    products = pd.DataFrame(
        [
            {
                "product_id": "P1",
                "title": "One",
                "description": "desc",
                "category": "All Beauty",
                "subcategory": None,
                "brand": "Acme",
                "store": "Acme Store",
                "price": 1.5,
                "average_rating": 4.0,
                "rating_count": 3,
                "extras_json": "{}",
            }
        ]
    )
    interactions = pd.DataFrame(
        [
            {
                "user_id": "U1",
                "product_id": "P1",
                "source_asin": "C1",
                "event_type": "review",
                "event_value": 5.0,
                "occurred_at": datetime(2020, 1, 1, tzinfo=UTC),
                "verified_purchase": True,
                "helpful_vote": 0,
                "extras_json": "{}",
            },
            {
                "user_id": "U2",
                "product_id": "ORPHAN",
                "source_asin": "CX",
                "event_type": "review",
                "event_value": 1.0,
                "occurred_at": datetime(2020, 2, 1, tzinfo=UTC),
                "verified_purchase": False,
                "helpful_vote": 0,
                "extras_json": "{}",
            },
        ]
    )
    products.to_parquet(directory / "products.parquet", index=False)
    interactions.to_parquet(directory / "interactions.parquet", index=False)


def test_ingest_skips_orphans_and_is_idempotent(postgres_engine: Engine, db_session: Session, tmp_path: Path) -> None:
    del db_session
    _write_fixture(tmp_path)
    report1 = ingest_parquet(
        postgres_engine,
        tmp_path,
        dataset_version="fixture_v1",
        report_dir=tmp_path / "report1",
    )
    assert report1["products_read"] == 1
    assert report1["orphan_interaction_rows"] == 1
    assert report1["orphan_product_ids"] == 1
    assert "ORPHAN" in report1["orphan_product_id_list"]
    factory = sessionmaker(bind=postgres_engine, future=True)
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Product)) == 1
        assert session.scalar(select(func.count()).select_from(User)) == 1
        assert session.scalar(select(func.count()).select_from(Interaction)) == 1
        user = session.get(User, "U2")
        assert user is None

    report2 = ingest_parquet(
        postgres_engine,
        tmp_path,
        dataset_version="fixture_v1",
        report_dir=tmp_path / "report2",
    )
    with factory() as session:
        assert session.scalar(select(func.count()).select_from(Interaction)) == 1
        assert session.scalar(select(func.count()).select_from(Product)) == 1
    assert report2["final_table_counts"]["interactions"] == 1
