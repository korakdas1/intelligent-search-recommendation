"""Optional LTR rerank against the PostgreSQL test catalog. No MiniLM download."""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import clear_settings_cache
from app.embeddings.normalize import l2_normalize
from app.models.dataset import DatasetVersion
from app.models.product import Product
from app.models.search import SearchEvent, SearchEventResult
from app.ranking.artifacts import save_ranker_bundle
from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_VERSION
from app.ranking.preprocessing import fit_scaler
from app.ranking.ranker import RankNetMLP
from app.ranking.retrieval import collect_fused_candidates
from app.ranking.runtime import reset_ltr_runtime
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


def _install_ranker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scaler = fit_scaler(np.zeros((8, FEATURE_COUNT), dtype=np.float32) + 0.1)
    model = RankNetMLP(hidden_sizes=(8,), dropout=0.0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    directory = tmp_path / "ranknet-test"
    save_ranker_bundle(
        directory,
        model=model,
        scaler=scaler,
        config={
            "model_version": "ranknet-test",
            "feature_version": FEATURE_VERSION,
            "input_dim": FEATURE_COUNT,
            "hidden_sizes": [8],
            "dropout": 0.0,
        },
        manifest={"model_version": "ranknet-test"},
    )
    monkeypatch.setenv("LTR_RANKER_DIR", str(directory))
    clear_settings_cache()
    reset_ltr_runtime()


def _event_count(session: Session) -> int:
    return int(session.scalar(select(func.count()).select_from(SearchEvent)) or 0)


def test_ltr_reranks_same_candidates(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(db_session)
    _install_runtime()
    _install_ranker(tmp_path, monkeypatch)
    try:
        hybrid = client.post(
            "/search",
            json={"query": "leather", "top_k": 3, "retrieval_mode": "hybrid", "fusion_method": "rrf"},
        )
        ltr = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 2,
                "retrieval_mode": "hybrid",
                "fusion_method": "rrf",
                "rerank_mode": "ltr",
            },
        )
        assert hybrid.status_code == 200
        assert ltr.status_code == 200
        body = ltr.json()
        assert body["rerank_mode"] == "ltr"
        assert body["retrieval_mode"] == "hybrid"
        assert len(body["results"]) == 2
        assert all(row["source"] == "hybrid_ltr" for row in body["results"])
        fused, _ = collect_fused_candidates(db_session, "leather", candidate_k=100, top_k=3)
        union = {row.product_id for row in fused}
        ltr_ids = {row["product_id"] for row in body["results"]}
        assert ltr_ids <= union
        hybrid_ids = {row["product_id"] for row in hybrid.json()["results"]}
        assert ltr_ids <= hybrid_ids | union
    finally:
        reset_semantic_runtime()
        reset_ltr_runtime()
        monkeypatch.delenv("LTR_RANKER_DIR", raising=False)
        clear_settings_cache()


def test_ltr_logs_one_event(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(db_session)
    _install_runtime()
    _install_ranker(tmp_path, monkeypatch)
    before = _event_count(db_session)
    try:
        response = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "rerank_mode": "ltr",
            },
        )
        assert response.status_code == 200
        db_session.expire_all()
        assert _event_count(db_session) == before + 1
        event = db_session.scalar(select(SearchEvent).order_by(SearchEvent.search_event_id.desc()))
        assert event is not None
        assert event.label_class == "observed"
        assert event.metadata_["rerank_mode"] == "ltr"
        assert event.metadata_["retrieval_mode"] == "hybrid"
        rows = list(
            db_session.scalars(
                select(SearchEventResult).where(
                    SearchEventResult.search_event_id == event.search_event_id
                )
            )
        )
        assert rows
        assert all(row.source == "hybrid_ltr" for row in rows)
    finally:
        reset_semantic_runtime()
        reset_ltr_runtime()
        monkeypatch.delenv("LTR_RANKER_DIR", raising=False)
        clear_settings_cache()


def test_ltr_missing_artifact_is_explicit_503(
    client: TestClient, db_session: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(db_session)
    _install_runtime()
    monkeypatch.setenv("LTR_RANKER_DIR", "/tmp/isre-missing-ranker")
    clear_settings_cache()
    reset_ltr_runtime()
    try:
        response = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "rerank_mode": "ltr",
            },
        )
        assert response.status_code == 503
        assert response.json()["detail"]["status"] == "ranker_unavailable"
    finally:
        reset_semantic_runtime()
        reset_ltr_runtime()
        monkeypatch.delenv("LTR_RANKER_DIR", raising=False)
        clear_settings_cache()


def test_hybrid_without_rerank_unchanged(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(db_session)
    _install_runtime()
    _install_ranker(tmp_path, monkeypatch)
    try:
        first = client.post(
            "/search",
            json={"query": "leather", "top_k": 3, "retrieval_mode": "hybrid", "fusion_method": "rrf"},
        )
        second = client.post(
            "/search",
            json={
                "query": "leather",
                "top_k": 3,
                "retrieval_mode": "hybrid",
                "fusion_method": "rrf",
                "rerank_mode": "none",
            },
        )
        assert first.json()["results"] == second.json()["results"]
        assert first.json()["rerank_mode"] == "none"
        assert all(row["source"] == "hybrid" for row in first.json()["results"])
    finally:
        reset_semantic_runtime()
        reset_ltr_runtime()
        monkeypatch.delenv("LTR_RANKER_DIR", raising=False)
        clear_settings_cache()
