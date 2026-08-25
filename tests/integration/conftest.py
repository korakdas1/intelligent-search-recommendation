"""PostgreSQL integration fixtures.

Destructive operations run only against a database whose name contains
``test``. The development catalog is never truncated here.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, clear_settings_cache
from app.models import Base
from app.db.session import reset_engine
from app.main import app

ROOT = Path(__file__).resolve().parents[2]


def _test_url() -> str:
    settings = Settings()
    return settings.resolved_test_database_url()


def _database_name(url: str) -> str:
    return url.rsplit("/", 1)[-1].split("?")[0]


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    url = _test_url()
    name = _database_name(url)
    if "test" not in name.lower():
        raise RuntimeError(f"Refusing to run integration tests against database {name!r}")

    previous_database_url = os.environ.get("DATABASE_URL")
    previous_app_env = os.environ.get("APP_ENV")
    os.environ["DATABASE_URL"] = url
    os.environ["APP_ENV"] = "test"
    clear_settings_cache()
    reset_engine()

    engine = create_engine(url, future=True, pool_pre_ping=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            connection.commit()
    except Exception as exc:
        engine.dispose()
        pytest.skip(
            "PostgreSQL test database is not reachable. "
            f"Expected {name} at TEST_DATABASE_URL / POSTGRES_TEST_DB. "
            f"Original error: {exc}"
        )

    cfg = Config(str(ROOT / "alembic.ini"))
    command.upgrade(cfg, "head")
    yield engine
    engine.dispose()
    reset_engine()
    clear_settings_cache()
    if previous_database_url is None:
        os.environ.pop("DATABASE_URL", None)
    else:
        os.environ["DATABASE_URL"] = previous_database_url
    if previous_app_env is None:
        os.environ.pop("APP_ENV", None)
    else:
        os.environ["APP_ENV"] = previous_app_env


@pytest.fixture
def db_session(postgres_engine: Engine) -> Iterator[Session]:
    factory = sessionmaker(bind=postgres_engine, autoflush=False, future=True)
    session = factory()
    try:
        for table in reversed(Base.metadata.sorted_tables):
            session.execute(table.delete())
        session.commit()
        yield session
        session.rollback()
    finally:
        session.close()


@pytest.fixture
def client(postgres_engine: Engine, db_session: Session) -> Iterator[TestClient]:
    del db_session
    reset_engine()
    with TestClient(app) as test_client:
        yield test_client
