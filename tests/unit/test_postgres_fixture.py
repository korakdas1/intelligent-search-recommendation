"""Database identity and fixture cleanup regressions; no PostgreSQL required."""

import os
from unittest.mock import MagicMock

import pytest

from app.core.config import Settings, clear_settings_cache, get_settings
from app.db import session as database_session
from tests.integration import conftest as integration

TEST_DB = "intelligent_search_recommendation_test"
TEST_URL = f"postgresql+psycopg://test:password@localhost:5432/{TEST_DB}"


@pytest.mark.parametrize(
    "database",
    [
        "intelligent_search_recommendation",
        "production",
        "production_test_backup",
        "contest",
        f"{TEST_DB}_archive",
        TEST_DB.upper(),
        "",
    ],
)
def test_rejects_database_other_than_exact_expected_name(monkeypatch, database):
    settings = Settings(
        _env_file=None,
        postgres_db="intelligent_search_recommendation",
        postgres_test_db=TEST_DB,
        TEST_DATABASE_URL=f"postgresql+psycopg://test:password@localhost/{database}",
    )
    monkeypatch.setattr(integration, "Settings", lambda: settings)
    with pytest.raises(RuntimeError, match="exactly match POSTGRES_TEST_DB"):
        integration._test_url()


@pytest.mark.parametrize("override", [None, TEST_URL, TEST_URL + "?connect_timeout=2"])
def test_accepts_exact_test_database(monkeypatch, override):
    settings = Settings(
        _env_file=None,
        postgres_db="intelligent_search_recommendation",
        postgres_test_db=TEST_DB,
        TEST_DATABASE_URL=override,
    )
    monkeypatch.setattr(integration, "Settings", lambda: settings)
    assert integration._test_url() == settings.resolved_test_database_url()


def test_accepts_explicit_custom_test_database(monkeypatch):
    url = "postgresql+psycopg://test:password@localhost/isolated_ci"
    settings = Settings(
        _env_file=None,
        postgres_db="development",
        postgres_test_db="isolated_ci",
        TEST_DATABASE_URL=url,
    )
    monkeypatch.setattr(integration, "Settings", lambda: settings)
    assert integration._test_url() == url


def test_rejects_test_database_equal_to_development_database(monkeypatch):
    settings = Settings(
        _env_file=None,
        postgres_db=TEST_DB,
        postgres_test_db=TEST_DB,
        TEST_DATABASE_URL=TEST_URL,
    )
    monkeypatch.setattr(integration, "Settings", lambda: settings)
    with pytest.raises(RuntimeError, match="differ from POSTGRES_DB"):
        integration._test_url()


@pytest.mark.parametrize("previous_environment", [False, True])
@pytest.mark.parametrize(
    "stage",
    ["create", "local_skip", "ci_failure", "migration", "wrong_database", "success", "dispose"],
)
def test_fixture_restores_environment_and_caches(monkeypatch, previous_environment, stage):
    previous = {
        "DATABASE_URL": "postgresql+psycopg://old:password@localhost/development",
        "APP_ENV": "development",
    }
    for key, value in previous.items():
        if previous_environment:
            monkeypatch.setenv(key, value)
        else:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("REQUIRE_POSTGRES_TESTS", "1" if stage == "ci_failure" else "0")
    monkeypatch.setattr(integration, "_test_url", lambda: TEST_URL)

    engine = MagicMock()
    connection = engine.connect.return_value
    connection.scalar.return_value = "production" if stage == "wrong_database" else TEST_DB

    def create_engine(*args, **kwargs):
        assert os.environ["DATABASE_URL"] == TEST_URL
        assert os.environ["APP_ENV"] == "test"
        assert get_settings().resolved_database_url() == TEST_URL
        # Populate the application's lazy caches without opening a connection.
        database_session.get_session_factory()
        if stage == "create":
            raise RuntimeError("engine setup failed")
        return engine

    monkeypatch.setattr(integration, "create_engine", create_engine)
    upgrade = MagicMock()
    monkeypatch.setattr(integration.command, "upgrade", upgrade)
    if stage in {"local_skip", "ci_failure"}:
        engine.connect.side_effect = RuntimeError("connection unavailable")
    elif stage == "migration":
        upgrade.side_effect = RuntimeError("migration failed")
    elif stage == "dispose":
        engine.dispose.side_effect = RuntimeError("dispose failed")

    clear_settings_cache()
    get_settings()  # A cached pre-fixture configuration must also be replaced.
    fixture = integration.postgres_engine.__wrapped__()
    try:
        if stage in {"success", "dispose"}:
            assert next(fixture) is engine
            if stage == "dispose":
                with pytest.raises(RuntimeError, match="dispose failed"):
                    fixture.close()
            else:
                fixture.close()
        else:
            expected_error = {
                "local_skip": pytest.skip.Exception,
                "ci_failure": pytest.fail.Exception,
            }.get(stage, RuntimeError)
            with pytest.raises(expected_error):
                next(fixture)

        for key, value in previous.items():
            assert os.environ.get(key) == (value if previous_environment else None)
        assert get_settings.cache_info().currsize == 0
        assert database_session._engine is None
        assert database_session._session_factory is None
        if stage != "create":
            engine.dispose.assert_called_once()
        if stage in {"create", "local_skip", "ci_failure", "wrong_database"}:
            upgrade.assert_not_called()
        else:
            assert upgrade.call_args.args[1] == "head"
    finally:
        fixture.close()
        database_session.reset_engine()
        clear_settings_cache()
