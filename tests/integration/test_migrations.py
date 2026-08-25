"""Alembic upgrade / downgrade against the test database."""

from __future__ import annotations

from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

pytestmark = pytest.mark.postgres

ROOT = Path(__file__).resolve().parents[2]


def test_upgrade_creates_expected_tables(postgres_engine: Engine) -> None:
    inspector = inspect(postgres_engine)
    tables = set(inspector.get_table_names())
    assert {
        "dataset_versions",
        "products",
        "users",
        "interactions",
        "search_events",
        "search_event_results",
        "artifact_versions",
        "alembic_version",
    } <= tables
    indexes = {idx["name"] for idx in inspector.get_indexes("interactions")}
    assert "ix_interactions_user_occurred" in indexes
    assert "ix_interactions_product_event" in indexes
    assert "ix_interactions_occurred_at" in indexes
    product_indexes = {idx["name"] for idx in inspector.get_indexes("products")}
    assert "ix_products_category" in product_indexes
    assert "ix_products_brand" in product_indexes
    assert "ix_products_price" in product_indexes
    assert "ix_products_search_document" in product_indexes
    assert "search_document" in {col["name"] for col in inspector.get_columns("products")}


def test_downgrade_and_reupgrade(postgres_engine: Engine) -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    command.downgrade(cfg, "base")
    inspector = inspect(postgres_engine)
    assert "products" not in inspector.get_table_names()
    command.upgrade(cfg, "head")
    inspector = inspect(postgres_engine)
    assert "products" in inspector.get_table_names()
    with postgres_engine.connect() as connection:
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        assert version == "20260822_0002"


def test_keyword_search_column_downgrade_and_reupgrade(postgres_engine: Engine) -> None:
    cfg = Config(str(ROOT / "alembic.ini"))
    command.downgrade(cfg, "20260822_0001")
    inspector = inspect(postgres_engine)
    assert "search_document" not in {col["name"] for col in inspector.get_columns("products")}
    product_indexes = {idx["name"] for idx in inspector.get_indexes("products")}
    assert "ix_products_search_document" not in product_indexes
    command.upgrade(cfg, "head")
    inspector = inspect(postgres_engine)
    assert "search_document" in {col["name"] for col in inspector.get_columns("products")}
    product_indexes = {idx["name"] for idx in inspector.get_indexes("products")}
    assert "ix_products_search_document" in product_indexes
    with postgres_engine.connect() as connection:
        version = connection.execute(text("SELECT version_num FROM alembic_version")).scalar_one()
        assert version == "20260822_0002"
        using = connection.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'products' AND indexname = 'ix_products_search_document'"
            )
        ).scalar_one()
        assert "using gin" in using.lower()
