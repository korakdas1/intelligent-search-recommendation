"""Hybrid fusion against the PostgreSQL test database. No MiniLM download."""

from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import clear_settings_cache
from app.db.repositories.search import keyword_candidates
from app.embeddings.normalize import l2_normalize
from app.models.dataset import DatasetVersion
from app.models.product import Product
from app.models.search import SearchEvent, SearchEventResult
from app.search.faiss_index import build_flat_ip_index
from app.search.hybrid import hybrid_search
from app.search.runtime import SemanticRuntime, reset_semantic_runtime, set_semantic_runtime
from app.search.semantic import semantic_candidates
from app.search.service import keyword_search, run_search

pytestmark = pytest.mark.postgres


class ScriptedEncoder:
    model_name = "fake-encoder"
    embedding_dim = 4
    device = "cpu"

    def encode(self, texts, *, batch_size: int = 32, show_progress: bool = False):
        del batch_size, show_progress
        rows = []
        for text in texts:
            if "leather" in text.lower() or "target-b" in text:
                rows.append([0.0, 1.0, 0.0, 0.0])
            elif "soap" in text.lower() or "target-c" in text:
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


def _event_count(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(SearchEvent)) or 0)


def test_hybrid_rrf_and_weighted_api(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    _install_runtime()
    try:
        rrf = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "fusion_method": "rrf",
            },
        )
        weighted = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "fusion_method": "weighted",
            },
        )
        assert rrf.status_code == 200
        assert weighted.status_code == 200
        rrf_body = rrf.json()
        weighted_body = weighted.json()
        assert rrf_body["retrieval_mode"] == "hybrid"
        assert rrf_body["fusion_method"] == "rrf"
        assert weighted_body["fusion_method"] == "weighted"
        assert rrf_body["results"]
        assert all(row["source"] == "hybrid" for row in rrf_body["results"])
        assert rrf_body["results"][0]["product_id"] == "PBBB"
        assert rrf_body == client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "fusion_method": "rrf",
            },
        ).json()
    finally:
        reset_semantic_runtime()


def test_hybrid_logs_one_event(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    _install_runtime()
    before = _event_count(db_session)
    try:
        response = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "fusion_method": "rrf",
            },
        )
        assert response.status_code == 200
        db_session.expire_all()
        assert _event_count(db_session) == before + 1
        event = db_session.scalar(select(SearchEvent).order_by(SearchEvent.search_event_id.desc()))
        assert event is not None
        assert event.label_class == "observed"
        assert event.metadata_["retrieval_mode"] == "hybrid"
        assert event.metadata_["fusion_method"] == "rrf"
        assert event.metadata_["candidate_k"] == 100
        assert "keyword_weight" not in event.metadata_
        rows = list(
            db_session.scalars(
                select(SearchEventResult).where(
                    SearchEventResult.search_event_id == event.search_event_id
                )
            )
        )
        assert rows
        assert all(row.source == "hybrid" for row in rows)
        sources = set(db_session.scalars(select(SearchEventResult.source)))
        assert "keyword" not in sources
        assert "semantic" not in sources
    finally:
        reset_semantic_runtime()


def test_component_paths_do_not_log(db_session: Session) -> None:
    _seed(db_session)
    _install_runtime()
    before = _event_count(db_session)
    try:
        keyword = keyword_candidates(db_session, "leather", 10)
        semantic = semantic_candidates(db_session, query="leather", top_k=10)
        silent = run_search(
            db_session,
            query="leather",
            top_k=3,
            retrieval_mode="hybrid",
            fusion_method="weighted",
            log=False,
        )
        assert keyword
        assert semantic
        assert silent.results
        assert _event_count(db_session) == before
    finally:
        reset_semantic_runtime()


def test_hybrid_empty_keyword_uses_semantic(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    _install_runtime()
    try:
        response = client.post(
            "/search",
            json={
                "query": "the",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "fusion_method": "rrf",
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["results"]
        assert all(row["source"] == "hybrid" for row in body["results"])
        keyword = keyword_candidates(db_session, "the", 10)
        assert keyword == []
    finally:
        reset_semantic_runtime()


def _empty_runtime() -> SemanticRuntime:
    return SemanticRuntime(
        artifact_version="test-semantic",
        dataset_version="search_v1",
        model_name="fake-encoder",
        embedding_dim=4,
        product_ids=np.array([], dtype="U32"),
        index=build_flat_ip_index(np.zeros((0, 4), dtype=np.float32)),
        backend="flat",
        embedding_manifest={"model_revision": "test"},
        index_manifest={},
        encoder=ScriptedEncoder(),
        skip_catalog_count_check=True,
    )


def test_hybrid_keyword_only_when_semantic_empty(db_session: Session) -> None:
    _seed(db_session)
    set_semantic_runtime(_empty_runtime())
    try:
        response = hybrid_search(
            db_session,
            query="leather",
            top_k=3,
            fusion_method="rrf",
            log=False,
        )
        assert [row.product_id for row in response.results] == ["PBBB"]
        assert response.results[0].source == "hybrid"
    finally:
        reset_semantic_runtime()


def test_hybrid_both_empty(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    set_semantic_runtime(_empty_runtime())
    try:
        response = hybrid_search(
            db_session,
            query="xyzzyqwerty12345notaproduct",
            top_k=5,
            fusion_method="rrf",
            log=False,
            candidate_k=3,
        )
        keyword = keyword_candidates(db_session, "xyzzyqwerty12345notaproduct", 5)
        assert keyword == []
        assert semantic_candidates(db_session, query="xyzzyqwerty12345notaproduct", top_k=5) == []
        assert response.results == []
        http = client.post(
            "/search",
            json={
                "query": "xyzzyqwerty12345notaproduct",
                "top_k": 5,
                "retrieval_mode": "hybrid",
                "fusion_method": "weighted",
            },
        )
        assert http.status_code == 200
        assert http.json()["results"] == []
    finally:
        reset_semantic_runtime()


def test_hybrid_omitted_fusion_uses_provisional_default(
    client: TestClient, db_session: Session
) -> None:
    _seed(db_session)
    _install_runtime()
    try:
        response = client.post(
            "/search",
            json={"query": "leather", "top_k": 3, "retrieval_mode": "hybrid"},
        )
        assert response.status_code == 200
        assert response.json()["fusion_method"] == "rrf"
    finally:
        reset_semantic_runtime()


def test_hybrid_unavailable_is_explicit_503(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(db_session)
    reset_semantic_runtime()
    monkeypatch.setenv("ARTIFACTS_ROOT", "/tmp/isre-missing-hybrid-artifacts")
    monkeypatch.setenv("SEMANTIC_ARTIFACT_VERSION", "does-not-exist")
    monkeypatch.setenv("SEMANTIC_REQUIRE_INDEX", "true")
    clear_settings_cache()
    try:
        response = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "fusion_method": "rrf",
            },
        )
        assert response.status_code == 503
        assert response.json()["detail"]["status"] == "semantic_unavailable"
    finally:
        monkeypatch.delenv("SEMANTIC_REQUIRE_INDEX", raising=False)
        clear_settings_cache()
        reset_semantic_runtime()


def test_keyword_baseline_unchanged(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    _install_runtime()
    try:
        implicit = client.post("/search", json={"query": "leather", "top_k": 5})
        explicit = client.post(
            "/search", json={"query": "leather", "top_k": 5, "retrieval_mode": "keyword"}
        )
        assert implicit.status_code == 200
        assert implicit.json()["retrieval_mode"] == "keyword"
        assert implicit.json()["fusion_method"] is None
        assert implicit.json()["results"] == explicit.json()["results"]
        assert implicit.json()["results"][0]["product_id"] == "PBBB"
        assert implicit.json()["results"][0]["source"] == "keyword"
        logged = keyword_search(db_session, query="leather", top_k=5, log=False)
        assert logged.results[0].product_id == "PBBB"
    finally:
        reset_semantic_runtime()
