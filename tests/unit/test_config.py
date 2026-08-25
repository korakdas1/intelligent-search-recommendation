"""Configuration loading tests."""

import pytest

from app.core.config import Settings, clear_settings_cache, get_settings


def test_defaults_load(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APP_ENV", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.app_env == "development"
    assert settings.log_level == "info"
    assert settings.random_seed == 42
    assert settings.app_host == "127.0.0.1"
    assert settings.app_port == 8000
    assert settings.service_id == "intelligent-search-recommendation-engine"
    assert settings.postgres_db == "intelligent_search_recommendation"
    assert "postgresql+psycopg://" in settings.resolved_database_url()
    assert settings.postgres_password not in "public logs"
    assert "change-me-locally" not in repr(settings)
    assert "intelligent_search_recommendation_test" in settings.resolved_test_database_url()
    assert settings.semantic_index_type == "flat"
    assert settings.semantic_model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert "all-MiniLM-L6-v2" in settings.semantic_artifact_version
    assert settings.semantic_index_is_required() is True
    assert settings.hybrid_fusion_method == "rrf"
    assert settings.hybrid_keyword_weight == 0.5
    assert settings.hybrid_rrf_k == 60
    assert settings.hybrid_candidate_k is None
    assert settings.hybrid_candidate_k_max == 500


def test_database_url_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/other")
    settings = Settings(_env_file=None)
    assert settings.resolved_database_url() == "postgresql+psycopg://u:p@localhost:5432/other"


def test_test_database_url_does_not_follow_dev_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost:5432/intelligent_search_recommendation")
    monkeypatch.delenv("TEST_DATABASE_URL", raising=False)
    settings = Settings(_env_file=None)
    assert "intelligent_search_recommendation_test" in settings.resolved_test_database_url()


def test_log_level_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = Settings(_env_file=None)
    assert settings.log_level == "debug"


def test_app_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "test")
    settings = Settings(_env_file=None)
    assert settings.app_env == "test"


def test_invalid_hybrid_weight_rejected() -> None:
    with pytest.raises(Exception):
        Settings(_env_file=None, hybrid_keyword_weight=-0.1)


def test_get_settings_cache_can_be_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    clear_settings_cache()
    monkeypatch.setenv("LOG_LEVEL", "warning")
    settings = get_settings()
    assert settings.log_level == "warning"
    clear_settings_cache()
