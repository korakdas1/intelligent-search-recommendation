"""Bounded personalized search HTTP tests. Tiny fixtures; no MiniLM download."""

from __future__ import annotations

from datetime import UTC, datetime
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
from app.models.interaction import Interaction
from app.models.product import Product
from app.models.search import SearchEvent, SearchEventResult
from app.models.user import User
from app.recommendations.cf_artifacts import save_cf_bundle
from app.recommendations.cf_data import mapping_checksum
from app.recommendations.cf_model import BPRMatrixFactorization
from app.recommendations.cf_runtime import reset_cf_runtime
from app.search.faiss_index import build_flat_ip_index
from app.search.runtime import SemanticRuntime, reset_semantic_runtime, set_semantic_runtime

pytestmark = pytest.mark.postgres


class ScriptedEncoder:
    model_name = "fake-encoder"
    embedding_dim = 3
    device = "cpu"

    def encode(self, texts, *, batch_size: int = 32, show_progress: bool = False):
        del batch_size, show_progress
        rows = []
        for text in texts:
            lowered = text.lower()
            if "leather" in lowered or "balm" in lowered:
                rows.append([0.9, 0.1, 0.0])
            elif "soap" in lowered:
                rows.append([0.0, 0.0, 1.0])
            elif "shampoo" in lowered:
                rows.append([0.2, 0.8, 0.0])
            else:
                rows.append([1.0, 0.0, 0.0])
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
            Product(
                product_id="PSHAM",
                dataset_version="search_v1",
                title="Baby Shampoo Wash",
                category="All Beauty",
                brand="Gentle",
                currency="USD",
            ),
        ]
    )
    session.add_all(
        [
            User(user_id="U1", dataset_version="search_v1"),
            User(user_id="U2", dataset_version="search_v1"),
            User(user_id="UEMPTY", dataset_version="search_v1"),
            User(user_id="UNCF", dataset_version="search_v1"),
        ]
    )
    session.flush()
    session.add_all(
        [
            Interaction(
                user_id="U1",
                product_id="PBBB",
                event_type="review",
                occurred_at=datetime(2020, 1, 1, tzinfo=UTC),
                dataset_version="search_v1",
            ),
            Interaction(
                user_id="U2",
                product_id="PCCC",
                event_type="review",
                occurred_at=datetime(2021, 1, 1, tzinfo=UTC),
                dataset_version="search_v1",
            ),
            Interaction(
                user_id="UNCF",
                product_id="PAAA",
                event_type="review",
                occurred_at=datetime(2021, 2, 1, tzinfo=UTC),
                dataset_version="search_v1",
            ),
        ]
    )
    session.commit()


def _install_runtime() -> SemanticRuntime:
    embeddings = l2_normalize(
        np.array(
            [
                [1.0, 0.0, 0.0],
                [0.9, 0.1, 0.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.8, 0.0],
            ],
            dtype=np.float32,
        )
    )
    runtime = SemanticRuntime(
        artifact_version="p12-test",
        dataset_version="search_v1",
        model_name="fake-encoder",
        embedding_dim=3,
        product_ids=np.array(["PAAA", "PBBB", "PCCC", "PSHAM"], dtype="U8"),
        index=build_flat_ip_index(embeddings),
        backend="flat",
        embedding_manifest={},
        index_manifest={},
        encoder=ScriptedEncoder(),
        skip_catalog_count_check=True,
    )
    set_semantic_runtime(runtime)
    return runtime


def _write_cf_artifact(directory: Path) -> None:
    model = BPRMatrixFactorization(2, 3, 4)
    with torch.no_grad():
        model.user_embedding.weight.zero_()
        model.item_embedding.weight.zero_()
        model.user_embedding.weight[0, 0] = 1.0  # U1
        model.item_embedding.weight[0, 0] = 0.1  # PAAA
        model.item_embedding.weight[1, 0] = 5.0  # PBBB
        model.item_embedding.weight[2, 0] = 0.2  # PCCC
        model.user_embedding.weight[1, 1] = 1.0  # U2
        model.item_embedding.weight[2, 1] = 4.0
    users = ["U1", "U2"]
    items = ["PAAA", "PBBB", "PCCC"]
    user_c, item_c = mapping_checksum(users, items)
    save_cf_bundle(
        directory,
        model=model,
        user_ids=users,
        product_ids=items,
        config={
            "model_version": "bpr-mf-test",
            "model_type": "bpr_mf",
            "embedding_dim": 4,
            "n_users": 2,
            "n_items": 3,
        },
        manifest={
            "model_version": "bpr-mf-test",
            "user_mapping_checksum": user_c,
            "item_mapping_checksum": item_c,
        },
    )


@pytest.fixture
def personalized_env(db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _seed(db_session)
    _install_runtime()
    artifact = tmp_path / "cf"
    _write_cf_artifact(artifact)
    monkeypatch.setenv("CF_MODEL_DIR", str(artifact))
    clear_settings_cache()
    reset_cf_runtime()
    yield db_session
    reset_cf_runtime()
    reset_semantic_runtime()
    clear_settings_cache()


def _payload(user_id: str, query: str = "leather", top_k: int = 10) -> dict:
    return {
        "query": query,
        "top_k": top_k,
        "retrieval_mode": "hybrid",
        "fusion_method": "rrf",
        "personalization_mode": "bounded",
        "user_id": user_id,
    }


def test_unknown_user_404(client: TestClient, personalized_env: Session) -> None:
    del personalized_env
    response = client.post("/search", json=_payload("NOPE"))
    assert response.status_code == 404
    assert client.get("/recommendations/user/NOPE?method=content").status_code == 404


def test_cold_user_matches_baseline(client: TestClient, personalized_env: Session) -> None:
    del personalized_env
    baseline = client.post(
        "/search",
        json={"query": "leather", "top_k": 5, "retrieval_mode": "hybrid", "fusion_method": "rrf"},
    )
    personalized = client.post("/search", json=_payload("UEMPTY", top_k=5))
    assert personalized.status_code == 200
    body = personalized.json()
    assert body["personalization_applied"] is False
    assert body["personalization_reason"] == "no_personalized_history"
    assert [row["product_id"] for row in body["results"]] == [
        row["product_id"] for row in baseline.json()["results"]
    ]


def test_user_outside_cf_still_personalizes(client: TestClient, personalized_env: Session) -> None:
    del personalized_env
    response = client.post("/search", json=_payload("UNCF"))
    assert response.status_code == 200
    body = response.json()
    assert body["personalization_applied"] is True
    assert body["personalization_signals"] == ["content"]


def test_same_query_two_users_differ(client: TestClient, personalized_env: Session) -> None:
    del personalized_env
    first = client.post("/search", json=_payload("U1"))
    second = client.post("/search", json=_payload("U2"))
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["personalization_applied"] is True
    assert second.json()["personalization_applied"] is True
    assert {row["product_id"] for row in first.json()["results"]} == {
        row["product_id"] for row in second.json()["results"]
    }


def test_candidate_set_equals_baseline(client: TestClient, personalized_env: Session) -> None:
    del personalized_env
    baseline = client.post(
        "/search",
        json={"query": "leather", "top_k": 10, "retrieval_mode": "hybrid", "fusion_method": "rrf"},
    )
    personalized = client.post("/search", json=_payload("U1"))
    assert set(row["product_id"] for row in personalized.json()["results"]) == set(
        row["product_id"] for row in baseline.json()["results"]
    )
    assert len(personalized.json()["results"]) == len(baseline.json()["results"])


def test_beard_product_not_injected_on_shampoo_query(
    client: TestClient, personalized_env: Session
) -> None:
    del personalized_env
    response = client.post("/search", json=_payload("U1", query="baby shampoo", top_k=10))
    assert response.status_code == 200
    ids = [row["product_id"] for row in response.json()["results"]]
    baseline = client.post(
        "/search",
        json={
            "query": "baby shampoo",
            "top_k": 10,
            "retrieval_mode": "hybrid",
            "fusion_method": "rrf",
        },
    )
    assert set(ids) == set(row["product_id"] for row in baseline.json()["results"])
    assert "PBBB" not in ids or "PBBB" in [
        row["product_id"] for row in baseline.json()["results"]
    ]


def test_one_search_event_and_user_id_logged(
    client: TestClient, personalized_env: Session
) -> None:
    session = personalized_env
    before = session.scalar(select(func.count()).select_from(SearchEvent)) or 0
    response = client.post("/search", json=_payload("U1", top_k=3))
    assert response.status_code == 200
    session.expire_all()
    after = session.scalar(select(func.count()).select_from(SearchEvent)) or 0
    assert after == before + 1
    event = session.scalar(select(SearchEvent).order_by(SearchEvent.search_event_id.desc()))
    assert event is not None
    assert event.user_id == "U1"
    assert event.metadata_["personalization_mode"] == "bounded"
    rows = list(
        session.scalars(
            select(SearchEventResult).where(SearchEventResult.search_event_id == event.search_event_id)
        )
    )
    logged_ids = [row.product_id for row in sorted(rows, key=lambda item: item.rank)]
    assert logged_ids == [item["product_id"] for item in response.json()["results"]]
    sources = set(session.scalars(select(SearchEventResult.source)))
    assert "keyword" not in sources
    assert "semantic" not in sources


def test_nonpersonalized_log_still_anonymous(
    client: TestClient, personalized_env: Session
) -> None:
    session = personalized_env
    response = client.post(
        "/search",
        json={"query": "leather", "top_k": 3, "retrieval_mode": "hybrid", "fusion_method": "rrf"},
    )
    assert response.status_code == 200
    session.expire_all()
    event = session.scalar(select(SearchEvent).order_by(SearchEvent.search_event_id.desc()))
    assert event is not None
    assert event.user_id is None
    assert event.metadata_.get("personalization_mode") is None


def test_keyword_regression_unchanged(client: TestClient, personalized_env: Session) -> None:
    del personalized_env
    response = client.post("/search", json={"query": "leather", "top_k": 5})
    assert response.status_code == 200
    assert response.json()["retrieval_mode"] == "keyword"
    assert response.json()["personalization_mode"] == "none"
    assert response.json()["results"][0]["source"] == "keyword"


def test_recommendation_unaffected(client: TestClient, personalized_env: Session) -> None:
    del personalized_env
    content = client.get("/recommendations/user/U1?method=content&top_k=3")
    assert content.status_code == 200
    hybrid = client.get("/recommendations/user/U1?method=hybrid&top_k=3")
    assert hybrid.status_code == 200
    cf = client.get("/recommendations/user/U1?method=cf&top_k=3")
    assert cf.status_code == 200
    trending = client.get("/recommendations/trending?top_k=3")
    assert trending.status_code == 200


def test_missing_cf_dir_still_content(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed(db_session)
    _install_runtime()
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path / "missing-cf"))
    clear_settings_cache()
    reset_cf_runtime()
    try:
        response = client.post("/search", json=_payload("U1"))
        assert response.status_code == 200
        assert response.json()["personalization_applied"] is True
        assert "cf" not in response.json()["personalization_signals"]
        assert "content" in response.json()["personalization_signals"]
    finally:
        reset_cf_runtime()
        reset_semantic_runtime()
        clear_settings_cache()
