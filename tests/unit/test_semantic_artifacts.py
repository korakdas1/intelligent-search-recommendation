"""Artifact manifest and stale-index detection. Offline fixture vectors only."""

from pathlib import Path
import json
from unittest.mock import Mock

import faiss
import numpy as np
import pytest

from app.core.config import Settings, clear_settings_cache
from app.embeddings.artifacts import ArtifactIncompatibleError, ArtifactUnavailableError, validate_manifest
from app.embeddings.checksums import sha256_file
from app.embeddings.io import save_product_ids
from app.search import runtime as runtime_module
from app.search.faiss_index import save_index
from app.search.runtime import load_semantic_runtime, reset_semantic_runtime
from tests.semantic_support import write_fixture_artifacts


def test_validate_manifest_requires_normalized() -> None:
    payload = {
        "artifact_version": "v",
        "dataset_version": "d",
        "product_count": 1,
        "model_name": "fake-encoder",
        "model_revision": None,
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


@pytest.fixture
def bundle(tmp_path):
    write_fixture_artifacts(
        tmp_path, artifact_version="provenance", dataset_version="fixture",
        product_ids=["P1", "P2"], embeddings=np.eye(2, 4, dtype=np.float32),
        semantic_catalog_sha256="a" * 64,
    )
    return Settings(
        _env_file=None, artifacts_root=str(tmp_path), semantic_artifact_version="provenance",
        semantic_model_name="fake-encoder", semantic_index_type="flat",
    )


def _path(settings, kind, filename="manifest.json"):
    return Path(settings.artifacts_root) / kind / settings.semantic_artifact_version / filename


def _edit_manifest(settings, kind, edit):
    path = _path(settings, kind)
    manifest = json.loads(path.read_text())
    edit(manifest)
    path.write_text(json.dumps(manifest))


def _legacy(settings):
    for kind in ("embeddings", "indexes"):
        _edit_manifest(settings, kind, lambda m: m.pop("semantic_catalog_sha256"))


def _assert_rejected_before_faiss(settings, monkeypatch, match):
    read = Mock(side_effect=AssertionError("FAISS must not deserialize rejected bytes"))
    monkeypatch.setattr(runtime_module, "load_index", read)
    with pytest.raises(ArtifactIncompatibleError, match=match):
        load_semantic_runtime(settings)
    read.assert_not_called()


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_strong_bundle_loads_without_reading_unused_embeddings(bundle, backend):
    bundle.semantic_index_type = backend
    _path(bundle, "embeddings", "embeddings.npy").unlink()
    runtime = load_semantic_runtime(bundle)
    assert runtime.ntotal == 2
    assert runtime.embedding_manifest["semantic_catalog_sha256"] == "a" * 64
    assert not runtime._catalog_verified


def test_same_length_mapping_changes_rejected_before_faiss(bundle, monkeypatch):
    save_product_ids(_path(bundle, "embeddings", "product_ids.npy"), ["P2", "P1"])
    _assert_rejected_before_faiss(bundle, monkeypatch, "product_ids.npy checksum mismatch")


def test_cross_manifest_mapping_checksum_mismatch(bundle, monkeypatch):
    _edit_manifest(bundle, "indexes", lambda m: m["checksums"].update({"product_ids.npy": "b" * 64}))
    _assert_rejected_before_faiss(bundle, monkeypatch, "embedding/index product_ids.npy checksum")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_selected_index_corruption_rejected_before_faiss(bundle, monkeypatch, backend, legacy):
    bundle.semantic_index_type = backend
    if legacy:
        _legacy(bundle)
    with _path(bundle, "indexes", f"{backend}.faiss").open("ab") as handle:
        handle.write(b"corrupt")
    _assert_rejected_before_faiss(bundle, monkeypatch, f"{backend}.faiss checksum mismatch")


@pytest.mark.parametrize("legacy", [False, True])
@pytest.mark.parametrize("kind,filename", [
    ("embeddings", "product_ids.npy"), ("indexes", "product_ids.npy"),
    ("indexes", "flat.faiss"), ("indexes", "hnsw.faiss"),
])
def test_required_checksums_cannot_be_omitted(bundle, monkeypatch, legacy, kind, filename):
    if legacy:
        _legacy(bundle)
    if filename == "hnsw.faiss":
        bundle.semantic_index_type = "hnsw"
    _edit_manifest(bundle, kind, lambda m: m["checksums"].pop(filename))
    _assert_rejected_before_faiss(bundle, monkeypatch, filename)


@pytest.mark.parametrize("field,value", [
    ("artifact_version", "other"), ("dataset_version", "other"), ("product_count", 3),
    ("model_name", "other"), ("model_revision", "different-revision"),
    ("embedding_dimension", 8), ("dtype", "float64"), ("normalized", False),
    ("semantic_text_version", "other"), ("faiss_metric", "L2"),
    ("semantic_catalog_sha256", "b" * 64),
])
def test_cross_manifest_identity_rejected(bundle, monkeypatch, field, value):
    _edit_manifest(bundle, "indexes", lambda m: m.update({field: value}))
    _assert_rejected_before_faiss(bundle, monkeypatch, field)


@pytest.mark.parametrize("value", [None, "", "not-a-sha256"])
def test_invalid_fingerprint_is_not_treated_as_legacy(bundle, value):
    for kind in ("embeddings", "indexes"):
        _edit_manifest(bundle, kind, lambda m: m.update(semantic_catalog_sha256=value))
    with pytest.raises(ArtifactIncompatibleError, match="semantic_catalog_sha256"):
        load_semantic_runtime(bundle)


@pytest.mark.parametrize("kind", ["embeddings", "indexes"])
def test_partial_fingerprint_is_not_treated_as_legacy(bundle, kind):
    _edit_manifest(bundle, kind, lambda m: m.pop("semantic_catalog_sha256"))
    with pytest.raises(ArtifactIncompatibleError, match="semantic_catalog_sha256"):
        load_semantic_runtime(bundle)


def _replace_index(settings, index):
    filename = f"{settings.semantic_index_type}.faiss"
    path = _path(settings, "indexes", filename)
    save_index(index, path)
    _edit_manifest(settings, "indexes", lambda m: m["checksums"].update({filename: sha256_file(path)}))


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_actual_backend_rejected_even_with_valid_checksum(bundle, backend):
    bundle.semantic_index_type = backend
    index = faiss.IndexHNSWFlat(4, 8, faiss.METRIC_INNER_PRODUCT) if backend == "flat" else faiss.IndexFlatIP(4)
    index.add(np.eye(2, 4, dtype=np.float32))
    _replace_index(bundle, index)
    with pytest.raises(ArtifactIncompatibleError, match="actual FAISS type"):
        load_semantic_runtime(bundle)


@pytest.mark.parametrize("backend", ["flat", "hnsw"])
def test_actual_metric_rejected_even_with_valid_checksum(bundle, backend):
    bundle.semantic_index_type = backend
    index = faiss.IndexFlatL2(4) if backend == "flat" else faiss.IndexHNSWFlat(4, 8, faiss.METRIC_L2)
    index.add(np.eye(2, 4, dtype=np.float32))
    _replace_index(bundle, index)
    with pytest.raises(ArtifactIncompatibleError, match="actual FAISS metric"):
        load_semantic_runtime(bundle)


@pytest.mark.parametrize("dimension,count,match", [(8, 2, "dimension"), (4, 1, "ntotal")])
def test_actual_index_shape_matches_manifests_and_mapping(bundle, dimension, count, match):
    index = faiss.IndexFlatIP(dimension)
    index.add(np.eye(count, dimension, dtype=np.float32))
    _replace_index(bundle, index)
    with pytest.raises(ArtifactIncompatibleError, match=match):
        load_semantic_runtime(bundle)


def test_both_manifest_counts_must_match_mapping(bundle, monkeypatch):
    for kind in ("embeddings", "indexes"):
        _edit_manifest(bundle, kind, lambda m: m.update(product_count=3))
    _assert_rejected_before_faiss(bundle, monkeypatch, "product_count")


def test_legacy_load_warns_and_keeps_count_check_without_rewriting(bundle, caplog, monkeypatch):
    # Alembic's logging setup can disable existing loggers earlier in a full run.
    monkeypatch.setattr(runtime_module.logger, "disabled", False)
    caplog.set_level("WARNING", logger=runtime_module.__name__)
    _legacy(bundle)
    paths = [_path(bundle, kind) for kind in ("embeddings", "indexes")]
    before = [path.read_bytes() for path in paths]
    runtime = load_semantic_runtime(bundle)
    assert "full semantic catalog identity cannot be verified" in caplog.text
    assert "rebuild recommended" in caplog.text
    assert "semantic_catalog_sha256" not in runtime.embedding_manifest
    session = Mock()
    session.scalar.return_value = 2
    runtime_module.ensure_runtime_matches_catalog(session, runtime)
    runtime_module.ensure_runtime_matches_catalog(session, runtime)
    assert session.scalar.call_count == 2
    assert not runtime._catalog_verified
    session.scalar.return_value = 3
    with pytest.raises(ArtifactIncompatibleError, match="rebuild semantic artifacts"):
        runtime_module.ensure_runtime_matches_catalog(session, runtime)
    assert before == [path.read_bytes() for path in paths]


def test_successful_fingerprint_check_cached_until_runtime_reload(bundle, monkeypatch):
    session = Mock()
    session.scalar.return_value = 2
    fingerprint = Mock(return_value="a" * 64)
    monkeypatch.setattr(runtime_module, "semantic_catalog_fingerprint", fingerprint)
    monkeypatch.setattr(runtime_module, "get_settings", lambda: bundle)
    reset_semantic_runtime()
    try:
        first = runtime_module.require_semantic_runtime(session)
        assert runtime_module.require_semantic_runtime(session) is first
        fingerprint.assert_called_once_with(session, dataset_version="fixture")
        assert session.scalar.call_count == 1
        reset_semantic_runtime()
        assert runtime_module.require_semantic_runtime(session) is not first
        assert fingerprint.call_count == 2
    finally:
        reset_semantic_runtime()


def test_failed_catalog_verification_is_not_cached(bundle, monkeypatch):
    runtime = load_semantic_runtime(bundle)
    session = Mock()
    session.scalar.return_value = 2
    fingerprint = Mock(return_value="b" * 64)
    monkeypatch.setattr(runtime_module, "semantic_catalog_fingerprint", fingerprint)
    for _ in range(2):
        with pytest.raises(ArtifactIncompatibleError, match="semantic_catalog_sha256.*rebuild"):
            runtime_module.ensure_runtime_matches_catalog(session, runtime)
    assert fingerprint.call_count == 2
    assert not runtime._catalog_verified


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
