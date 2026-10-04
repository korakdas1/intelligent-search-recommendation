"""PostgreSQL integration fixtures.

The URL and connected database must exactly match ``POSTGRES_TEST_DB``,
which must differ from ``POSTGRES_DB``, before migrations or table cleanup.
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
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, clear_settings_cache
from app.models import Base
from app.db.session import reset_engine
from app.main import app

ROOT = Path(__file__).resolve().parents[2]


def _test_url() -> str:
    settings = Settings()
    url = settings.resolved_test_database_url()
    parsed = make_url(url)
    expected = settings.postgres_test_db
    if (
        parsed.get_backend_name() != "postgresql"
        or not expected
        or expected == settings.postgres_db
        or parsed.database != expected
    ):
        raise RuntimeError(
            "Refusing to run integration tests: the PostgreSQL database must "
            "exactly match POSTGRES_TEST_DB and differ from POSTGRES_DB."
        )
    return url


@pytest.fixture(scope="session")
def postgres_engine() -> Iterator[Engine]:
    url = _test_url()
    name = make_url(url).database
    engine = None
    try:
        with pytest.MonkeyPatch.context() as environment:
            environment.setenv("DATABASE_URL", url)
            environment.setenv("APP_ENV", "test")
            clear_settings_cache()
            reset_engine()
            engine = create_engine(url, future=True, pool_pre_ping=True)
            try:
                connection = engine.connect()
            except Exception:
                message = (
                    "PostgreSQL test database is not reachable. "
                    f"Expected {name} at TEST_DATABASE_URL / POSTGRES_TEST_DB."
                )
                if os.environ.get("REQUIRE_POSTGRES_TESTS") == "1":
                    pytest.fail(message, pytrace=False)
                pytest.skip(message)

            with connection:
                # Also catch connection options that override the URL database.
                actual = connection.scalar(text("SELECT current_database()"))
                if actual != name:
                    raise RuntimeError(
                        "Refusing to run integration tests: connected database "
                        "does not match POSTGRES_TEST_DB."
                    )

            cfg = Config(str(ROOT / "alembic.ini"))
            command.upgrade(cfg, "head")
            yield engine
    finally:
        # The environment context has restored its values before caches reset.
        try:
            if engine is not None:
                engine.dispose()
        finally:
            try:
                reset_engine()
            finally:
                clear_settings_cache()


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
