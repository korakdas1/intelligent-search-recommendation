"""Tiny semantic builds and catalog provenance on the isolated test database."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.core.config import Settings
from app.db.repositories.products import iter_products_ordered, semantic_catalog_fingerprint
from app.embeddings.artifacts import ArtifactIncompatibleError
from app.embeddings.encoder import FakeEncoder
from app.embeddings.provenance import SemanticCatalogFingerprint
from app.embeddings.text import build_semantic_text
from app.models.dataset import DatasetVersion
from app.models.product import Product
from app.search import runtime as runtime_module
from scripts import build_semantic_index as build
from tests.semantic_support import write_fixture_artifacts

pytestmark = pytest.mark.postgres


@pytest.fixture
def catalog(db_session):
    db_session.add(DatasetVersion(dataset_version="provenance-fixture", source_name="fixture", row_counts={}, checksums={}))
    db_session.flush()
    # Insertion order deliberately differs from C-collation product-ID order.
    rows = [
        Product(product_id=pid, dataset_version="provenance-fixture", title=f"Cream {pid}",
                brand="Acme", category="Beauty", subcategory="Face", description="Soft")
        for pid in ("ä", "a", "A")
    ]
    db_session.add_all(rows)
    db_session.commit()
    return rows


def _settings(tmp_path):
    return Settings(_env_file=None, artifacts_root=str(tmp_path), semantic_artifact_version="tiny-provenance",
                    semantic_model_name="fake-encoder", semantic_index_type="flat")


def _catalog_runtime(tmp_path, db_session):
    rows = iter_products_ordered(db_session, dataset_version="provenance-fixture")
    write_fixture_artifacts(
        tmp_path, artifact_version="tiny-provenance", dataset_version="provenance-fixture",
        product_ids=[row.product_id for row in rows], embeddings=np.eye(3, 4, dtype=np.float32),
        semantic_catalog_sha256=semantic_catalog_fingerprint(db_session, dataset_version="provenance-fixture"),
    )
    return runtime_module.load_semantic_runtime(_settings(tmp_path))


@pytest.mark.parametrize("field", ["product_id", "title", "brand", "category", "subcategory", "description"])
def test_same_count_catalog_changes_rejected(tmp_path, db_session, catalog, field):
    runtime = _catalog_runtime(tmp_path, db_session)
    setattr(catalog[0], field, "changed")
    db_session.commit()
    with pytest.raises(ArtifactIncompatibleError, match="semantic_catalog_sha256.*rebuild"):
        runtime_module.ensure_runtime_matches_catalog(db_session, runtime)
    assert not runtime._catalog_verified


def test_catalog_order_and_successful_scan_cache(tmp_path, db_session, catalog, monkeypatch):
    expected = SemanticCatalogFingerprint()
    for row in sorted(catalog, key=lambda row: row.product_id.encode("utf-8")):
        expected.update(row.product_id, build_semantic_text(
            title=row.title, brand=row.brand, category=row.category,
            subcategory=row.subcategory, description=row.description,
        ))
    assert semantic_catalog_fingerprint(db_session, dataset_version="provenance-fixture") == expected.hexdigest()
    runtime = _catalog_runtime(tmp_path, db_session)
    spy = Mock(wraps=semantic_catalog_fingerprint)
    monkeypatch.setattr(runtime_module, "semantic_catalog_fingerprint", spy)
    runtime_module.set_semantic_runtime(runtime)
    try:
        assert runtime_module.require_semantic_runtime(db_session) is runtime
        assert runtime_module.require_semantic_runtime(db_session) is runtime
        spy.assert_called_once()
        # Reload starts a new validation lifetime and detects later catalog edits.
        catalog[0].title = "Changed after verification"
        db_session.commit()
        runtime_module.reset_semantic_runtime()
        reloaded = runtime_module.load_semantic_runtime(_settings(tmp_path))
        with pytest.raises(ArtifactIncompatibleError, match="semantic_catalog_sha256"):
            runtime_module.ensure_runtime_matches_catalog(db_session, reloaded)
        assert spy.call_count == 2
    finally:
        runtime_module.reset_semantic_runtime()


@pytest.mark.parametrize("revision", [None, "0123456789abcdef"])
def test_tiny_build_records_actual_requested_revision_and_fingerprint(tmp_path, db_session, catalog, monkeypatch, revision):
    requested = []

    class RecordingEncoder(FakeEncoder):
        def __init__(self, model_name, *, device, model_revision=None, model_license=None):
            super().__init__(dim=4, model_name=model_name)
            self.device = device
            self.model_revision = model_revision
            self.model_license = model_license
            self._model = SimpleNamespace(max_seq_length=32, model_card_data=SimpleNamespace(base_model_revision="do-not-infer"))
            requested.append((device, model_revision))

        def encode(self, texts, **kwargs):
            if self.device == "cuda":
                raise RuntimeError("fake GPU failure")
            return super().encode(texts, **kwargs)

    monkeypatch.setattr(build, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(build, "SentenceTransformerEncoder", RecordingEncoder)
    monkeypatch.setattr(build, "resolve_device", lambda choice: "cuda")
    monkeypatch.setattr(build, "_model_card_facts", lambda name: {
        "model_revision": revision, "model_license": "fixture", "max_seq_length": None,
        "model_card_url": "https://example.invalid/fixture",
    })
    assert build.main([
        "--dataset-version", "provenance-fixture", "--artifact-version", "tiny-provenance",
        "--model-name", "fake-encoder", "--fetch-size", "1", "--no-register",
    ]) == 0
    assert requested == [("cuda", revision), ("cpu", revision)]
    for backend in ("flat", "hnsw"):
        settings = _settings(tmp_path)
        settings.semantic_index_type = backend
        runtime = runtime_module.load_semantic_runtime(settings)
        assert runtime.embedding_manifest["model_revision"] == revision
        assert runtime.index_manifest["model_revision"] == revision
        assert runtime.product_ids.tolist() == ["A", "a", "ä"]
        assert runtime.embedding_manifest["semantic_catalog_sha256"] == semantic_catalog_fingerprint(
            db_session, dataset_version="provenance-fixture",
        )
        runtime_module.ensure_runtime_matches_catalog(db_session, runtime)
        assert runtime._catalog_verified
