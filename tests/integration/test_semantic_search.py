"""Semantic search against the PostgreSQL test database. No MiniLM download."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, clear_settings_cache
from app.db.repositories.artifacts import upsert_artifact_version
from app.db.repositories.products import get_products_by_ids
from app.embeddings.encoder import FakeEncoder
from app.embeddings.normalize import l2_normalize
from app.models.artifact import ArtifactVersion
from app.models.dataset import DatasetVersion
from app.models.product import Product
from app.models.search import SearchEvent, SearchEventResult
from app.search.faiss_index import build_flat_ip_index
from app.search.runtime import SemanticRuntime, reset_semantic_runtime, set_semantic_runtime

pytestmark = pytest.mark.postgres


class ScriptedEncoder:
    model_name = "fake-encoder"
    embedding_dim = 4
    device = "cpu"

    def encode(self, texts, *, batch_size: int = 32, show_progress: bool = False):
        del batch_size, show_progress
        rows = []
        for text in texts:
            if "target-b" in text:
                rows.append([0.0, 1.0, 0.0, 0.0])
            elif "target-c" in text:
                rows.append([0.0, 0.0, 1.0, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0, 0.0])
        return l2_normalize(np.array(rows, dtype=np.float32))


def _seed(session: Session) -> None:
    session.add(
        DatasetVersion(
            dataset_version="search_v1",
            source_name="fixture",
            row_counts={},
            checksums={},
        )
    )
    session.add_all(
        [
            Product(
                product_id="PAAA",
                dataset_version="search_v1",
                title="Alpha Cream",
                category="All Beauty",
                brand="Acme",
                price=Decimal("1.00"),
                currency="USD",
            ),
            Product(
                product_id="PBBB",
                dataset_version="search_v1",
                title="Beta Leather Balm",
                category="All Beauty",
                brand="Howard",
                price=Decimal("2.00"),
                currency="USD",
            ),
            Product(
                product_id="PCCC",
                dataset_version="search_v1",
                title="Gamma Soap",
                category="All Beauty",
                currency="USD",
            ),
        ]
    )
    session.commit()


def _install_runtime() -> SemanticRuntime:
    embeddings = l2_normalize(np.eye(3, 4, dtype=np.float32))
    index = build_flat_ip_index(embeddings)
    runtime = SemanticRuntime(
        artifact_version="test-semantic",
        dataset_version="search_v1",
        model_name="fake-encoder",
        embedding_dim=4,
        product_ids=np.array(["PAAA", "PBBB", "PCCC"], dtype="U32"),
        index=index,
        backend="flat",
        embedding_manifest={"model_revision": "test"},
        index_manifest={},
        encoder=ScriptedEncoder(),
        skip_catalog_count_check=False,
    )
    set_semantic_runtime(runtime)
    return runtime


def test_semantic_search_ranks_and_logs(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    _install_runtime()
    try:
        response = client.post(
            "/search",
            json={"query": "target-b query", "top_k": 3, "retrieval_mode": "semantic"},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["retrieval_mode"] == "semantic"
        ids = [row["product_id"] for row in body["results"]]
        assert ids[0] == "PBBB"
        assert body["results"][0]["source"] == "semantic"
        assert body["results"][0]["title"] == "Beta Leather Balm"
        scores = [row["score"] for row in body["results"]]
        assert scores == sorted(scores, reverse=True)
        db_session.expire_all()
        event = db_session.scalar(select(SearchEvent).order_by(SearchEvent.search_event_id.desc()))
        assert event is not None
        assert event.label_class == "observed"
        assert event.metadata_["retrieval_mode"] == "semantic"
        assert event.metadata_["embedding_model"] == "fake-encoder"
        assert event.metadata_["index_backend"] == "flat"
        rows = list(
            db_session.scalars(
                select(SearchEventResult).where(
                    SearchEventResult.search_event_id == event.search_event_id
                )
            )
        )
        assert rows
        assert all(row.source == "semantic" for row in rows)
    finally:
        reset_semantic_runtime()


def test_keyword_mode_still_default(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    implicit = client.post("/search", json={"query": "alpha", "top_k": 5})
    explicit = client.post(
        "/search", json={"query": "alpha", "top_k": 5, "retrieval_mode": "keyword"}
    )
    assert implicit.status_code == 200
    assert implicit.json()["retrieval_mode"] == "keyword"
    assert implicit.json()["results"][0]["source"] == "keyword"
    assert implicit.json()["results"] == explicit.json()["results"]


def test_batch_product_fetch_preserves_caller_order(db_session: Session) -> None:
    _seed(db_session)
    fetched = get_products_by_ids(db_session, ["PCCC", "PAAA", "missing"])
    assert set(fetched) == {"PCCC", "PAAA"}
    ordered = [fetched[pid].title for pid in ["PCCC", "PAAA"] if pid in fetched]
    assert ordered == ["Gamma Soap", "Alpha Cream"]
    count = db_session.scalar(select(func.count()).select_from(Product))
    assert count == 3


def test_artifact_registry_upsert(db_session: Session) -> None:
    db_session.add(
        DatasetVersion(
            dataset_version="search_v1",
            source_name="fixture",
            row_counts={},
            checksums={},
        )
    )
    db_session.commit()
    upsert_artifact_version(
        db_session,
        artifact_id="embeddings:test",
        artifact_type="embedding_model",
        version="test",
        dataset_version="search_v1",
        embedding_model_name="fake-encoder",
        embedding_dim=4,
        metric="inner_product",
        path="artifacts/embeddings/test",
        metadata={"normalized": True},
    )
    db_session.commit()
    row = db_session.get(ArtifactVersion, "embeddings:test")
    assert row is not None
    assert row.embedding_dim == 4
    assert row.metadata_["normalized"] is True


def test_ready_503_when_semantic_required_and_missing(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    del db_session
    monkeypatch.setenv("SEMANTIC_REQUIRE_INDEX", "true")
    monkeypatch.setenv("SEMANTIC_ARTIFACT_VERSION", "does-not-exist")
    monkeypatch.setenv("ARTIFACTS_ROOT", "/tmp/isre-missing-semantic-artifacts")
    clear_settings_cache()
    reset_semantic_runtime()
    try:
        health = client.get("/health")
        assert health.status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 503
        assert ready.json()["checks"]["database"] == "ok"
        assert ready.json()["checks"]["semantic_index"] == "unavailable"
    finally:
        monkeypatch.delenv("SEMANTIC_REQUIRE_INDEX", raising=False)
        clear_settings_cache()
        reset_semantic_runtime()


def test_fake_encoder_is_offline() -> None:
    encoder = FakeEncoder(dim=8)
    vectors = encoder.encode(["hello", "hello"])
    assert vectors.shape == (2, 8)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    assert np.allclose(vectors[0], vectors[1])
