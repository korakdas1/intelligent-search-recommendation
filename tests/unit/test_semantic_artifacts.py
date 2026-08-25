"""Artifact manifest and stale-index detection. Offline fixture vectors only."""

from pathlib import Path

import numpy as np
import pytest

from app.core.config import Settings, clear_settings_cache
from app.embeddings.artifacts import ArtifactIncompatibleError, ArtifactUnavailableError, validate_manifest
from app.search.runtime import load_semantic_runtime, reset_semantic_runtime
from tests.semantic_support import write_fixture_artifacts


def test_validate_manifest_requires_normalized() -> None:
    payload = {
        "artifact_version": "v",
        "dataset_version": "d",
        "product_count": 1,
        "model_name": "fake-encoder",
        "embedding_dimension": 4,
        "dtype": "float32",
        "normalized": False,
        "semantic_text_version": "semantic-text-v1",
        "faiss_metric": "inner_product",
    }
    with pytest.raises(ArtifactIncompatibleError):
        validate_manifest(payload)


def test_load_runtime_and_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["B", "A", "C"]
    embeddings = np.eye(3, 4, dtype=np.float32)
    write_fixture_artifacts(
        tmp_path,
        artifact_version="fix-v1",
        dataset_version="ds-v1",
        product_ids=ids,
        embeddings=embeddings,
        model_name="fake-encoder",
    )
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SEMANTIC_ARTIFACT_VERSION", "fix-v1")
    monkeypatch.setenv("SEMANTIC_MODEL_NAME", "fake-encoder")
    monkeypatch.setenv("SEMANTIC_INDEX_TYPE", "flat")
    clear_settings_cache()
    reset_semantic_runtime()
    settings = Settings(_env_file=None)
    runtime = load_semantic_runtime(settings, skip_catalog_count_check=True)
    assert runtime.ntotal == 3
    assert runtime.row_to_product_id(0) == "B"
    assert runtime.row_to_product_id(1) == "A"
    assert len(runtime.product_ids) == runtime.index.ntotal
    reset_semantic_runtime()
    clear_settings_cache()


def test_missing_artifacts_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path / "empty"))
    monkeypatch.setenv("SEMANTIC_ARTIFACT_VERSION", "missing")
    monkeypatch.setenv("SEMANTIC_MODEL_NAME", "fake-encoder")
    clear_settings_cache()
    settings = Settings(_env_file=None)
    with pytest.raises(ArtifactUnavailableError):
        load_semantic_runtime(settings)
    clear_settings_cache()


def test_incompatible_dimension(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["P1", "P2"]
    embeddings = np.eye(2, 4, dtype=np.float32)
    write_fixture_artifacts(
        tmp_path,
        artifact_version="fix-v2",
        dataset_version="ds-v1",
        product_ids=ids,
        embeddings=embeddings,
        model_name="fake-encoder",
    )
    manifest = tmp_path / "embeddings" / "fix-v2" / "manifest.json"
    text = manifest.read_text(encoding="utf-8").replace('"embedding_dimension": 4', '"embedding_dimension": 99')
    manifest.write_text(text, encoding="utf-8")
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SEMANTIC_ARTIFACT_VERSION", "fix-v2")
    monkeypatch.setenv("SEMANTIC_MODEL_NAME", "fake-encoder")
    monkeypatch.setenv("SEMANTIC_INDEX_TYPE", "flat")
    clear_settings_cache()
    settings = Settings(_env_file=None)
    with pytest.raises(ArtifactIncompatibleError):
        load_semantic_runtime(settings)
    clear_settings_cache()


def test_hnsw_backend_loads_hnsw(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ids = ["P1", "P2", "P3"]
    embeddings = np.eye(3, 4, dtype=np.float32)
    write_fixture_artifacts(
        tmp_path,
        artifact_version="fix-hnsw",
        dataset_version="ds-v1",
        product_ids=ids,
        embeddings=embeddings,
        model_name="fake-encoder",
    )
    monkeypatch.setenv("ARTIFACTS_ROOT", str(tmp_path))
    monkeypatch.setenv("SEMANTIC_ARTIFACT_VERSION", "fix-hnsw")
    monkeypatch.setenv("SEMANTIC_MODEL_NAME", "fake-encoder")
    monkeypatch.setenv("SEMANTIC_INDEX_TYPE", "hnsw")
    clear_settings_cache()
    settings = Settings(_env_file=None)
    runtime = load_semantic_runtime(settings, skip_catalog_count_check=True)
    assert runtime.backend == "hnsw"
    assert hasattr(runtime.index, "hnsw")
    reset_semantic_runtime()
    clear_settings_cache()
