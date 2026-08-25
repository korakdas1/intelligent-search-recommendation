"""Environment-driven application settings.

PostgreSQL is configured via ``DATABASE_URL`` or ``POSTGRES_*`` parts.
Passwords are never logged by this module.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.events import DATASET_VERSION_FULL
from app.embeddings.constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_HNSW_EF_CONSTRUCTION,
    DEFAULT_HNSW_EF_SEARCH,
    DEFAULT_HNSW_M,
    DEFAULT_INDEX_BACKEND,
    EMBEDDING_MODEL_NAME,
    SEMANTIC_TEXT_VERSION,
)
from app.embeddings.versioning import build_artifact_version


class Settings(BaseSettings):
    """Local-development defaults. Override with environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "Intelligent Search & Recommendation Engine"
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "info"
    random_seed: int = 42
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    service_id: str = "intelligent-search-recommendation-engine"

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "intelligent_search_recommendation"
    postgres_test_db: str = "intelligent_search_recommendation_test"
    postgres_user: str = "isre"
    postgres_password: str = Field(default="change-me-locally", repr=False)
    database_url: str | None = Field(default=None, validation_alias="DATABASE_URL", repr=False)
    test_database_url: str | None = Field(
        default=None, validation_alias="TEST_DATABASE_URL", repr=False
    )

    artifacts_root: str = "artifacts"
    semantic_model_name: str = EMBEDDING_MODEL_NAME
    semantic_text_version: str = SEMANTIC_TEXT_VERSION
    semantic_artifact_version: str = Field(
        default_factory=lambda: build_artifact_version(
            dataset_version=DATASET_VERSION_FULL,
            model_name=EMBEDDING_MODEL_NAME,
            text_version=SEMANTIC_TEXT_VERSION,
        )
    )
    semantic_index_type: Literal["flat", "hnsw"] = DEFAULT_INDEX_BACKEND
    semantic_device: Literal["auto", "cpu", "cuda"] = "auto"
    semantic_batch_size: int = DEFAULT_BATCH_SIZE
    semantic_hnsw_m: int = DEFAULT_HNSW_M
    semantic_hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION
    semantic_hnsw_ef_search: int = DEFAULT_HNSW_EF_SEARCH
    semantic_require_index: bool | None = None

    hybrid_candidate_k: int | None = None
    hybrid_candidate_k_max: int = 500
    hybrid_rrf_k: int = 60
    hybrid_keyword_weight: float = 0.5
    hybrid_fusion_method: Literal["rrf", "weighted"] = "rrf"
    ltr_ranker_version: str = "ranknet-v1"
    ltr_ranker_dir: str | None = None
    ltr_require_ranker: bool | None = None
    cf_model_version: str = "bpr-mf-v1"
    cf_model_dir: str | None = None
    cf_require_model: bool | None = None

    @field_validator("semantic_index_type", "hybrid_fusion_method", mode="before")
    @classmethod
    def _normalize_index_type(cls, value: object) -> object:
        if isinstance(value, str):
            return value.strip().lower()
        return value

    @field_validator("ltr_ranker_dir", "cf_model_dir", mode="before")
    @classmethod
    def _empty_optional_dir(cls, value: object) -> object:
        if value == "" or value is None:
            return None
        return value
    @classmethod
    def _empty_candidate_k(cls, value: object) -> object:
        if value == "" or value is None:
            return None
        return value

    @field_validator("hybrid_candidate_k")
    @classmethod
    def _validate_candidate_k(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("HYBRID_CANDIDATE_K must be at least 1")
        return value

    @field_validator("hybrid_keyword_weight")
    @classmethod
    def _validate_hybrid_weight(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("HYBRID_KEYWORD_WEIGHT must be between 0 and 1 inclusive")
        return value

    @field_validator("hybrid_rrf_k")
    @classmethod
    def _validate_rrf_k(cls, value: int) -> int:
        if value < 0:
            raise ValueError("HYBRID_RRF_K must be >= 0")
        return value

    @field_validator("hybrid_candidate_k_max")
    @classmethod
    def _validate_candidate_k_max(cls, value: int) -> int:
        if value < 1:
            raise ValueError("HYBRID_CANDIDATE_K_MAX must be at least 1")
        return value

    def semantic_index_is_required(self) -> bool:
        """Development/production require artifacts; pytest does not unless forced."""

        if self.semantic_require_index is not None:
            return self.semantic_require_index
        return self.app_env != "test"

    def ltr_ranker_is_required(self) -> bool:
        """Optional by default. Set LTR_REQUIRE_RANKER=true to fail /ready without a ranker."""

        if self.ltr_require_ranker is not None:
            return self.ltr_require_ranker
        return False

    def cf_model_is_required(self) -> bool:
        """Optional by default. Set CF_REQUIRE_MODEL=true to fail /ready without BPR-MF."""

        if self.cf_require_model is not None:
            return self.cf_require_model
        return False

    def resolved_database_url(self) -> str:
        """SQLAlchemy URL for the development database."""

        if self.database_url:
            return self.database_url
        return self._build_url(self.postgres_db)

    def resolved_test_database_url(self) -> str:
        """SQLAlchemy URL for the isolated test database.

        Never fall back to ``DATABASE_URL`` (the development catalog).
        Tests must use ``TEST_DATABASE_URL`` or ``POSTGRES_TEST_DB``.
        """

        if self.test_database_url:
            return self.test_database_url
        return self._build_url(self.postgres_test_db)

    def _build_url(self, database: str) -> str:
        user = quote_plus(self.postgres_user)
        password = quote_plus(self.postgres_password)
        host = self.postgres_host
        port = self.postgres_port
        return f"postgresql+psycopg://{user}:{password}@{host}:{port}/{database}"


@lru_cache
def get_settings() -> Settings:
    """Return cached settings. Clear the cache in tests after env changes."""

    return Settings()


def clear_settings_cache() -> None:
    """Drop the cached Settings instance."""

    get_settings.cache_clear()
