"""hybrid-rec-v1 HTTP tests. Tiny fixtures; no MiniLM download or CF training."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.core.config import clear_settings_cache
from app.embeddings.normalize import l2_normalize
from app.models.dataset import DatasetVersion
from app.models.interaction import Interaction
from app.models.product import Product
from app.models.user import User
from app.recommendations.cf_artifacts import save_cf_bundle
from app.recommendations.cf_data import mapping_checksum
from app.recommendations.cf_model import BPRMatrixFactorization
from app.recommendations.cf_runtime import reset_cf_runtime
from app.recommendations.hybrid_constants import (
    REASON_CF_ARTIFACT_UNAVAILABLE,
    REASON_CONTENT_ARTIFACT_UNAVAILABLE,
    REASON_NO_PERSONALIZED_HISTORY,
    SOURCE_HYBRID,
    USER_HYBRID_MODE,
)
from app.search.faiss_index import build_flat_ip_index
from app.search.runtime import SemanticRuntime, reset_semantic_runtime, set_semantic_runtime

pytestmark = pytest.mark.postgres


def _seed_catalog(session: Session) -> None:
    session.add(
        DatasetVersion(
            dataset_version="rec_v1",
            source_name="fixture",
            row_counts={},
            checksums={},
        )
    )
    session.add_all(
        [
            Product(
                product_id="RA",
                dataset_version="rec_v1",
                title="Alpha Cream",
                brand="Acme",
                category="All Beauty",
                price=Decimal("1.00"),
                currency="USD",
            ),
            Product(
                product_id="RB",
                dataset_version="rec_v1",
                title="Beta Leather Balm",
                brand="Howard",
                category="All Beauty",
                price=Decimal("2.00"),
                currency="USD",
            ),
            Product(
                product_id="RC",
                dataset_version="rec_v1",
                title="Gamma Soap",
                category="All Beauty",
                currency="USD",
            ),
            Product(
                product_id="RD",
                dataset_version="rec_v1",
                title="Delta Cold Item",
                category="All Beauty",
                currency="USD",
            ),
        ]
    )
    session.add_all(
        [
            User(user_id="U1", dataset_version="rec_v1"),
            User(user_id="U2", dataset_version="rec_v1"),
            User(user_id="U3", dataset_version="rec_v1"),
            User(user_id="UEMPTY", dataset_version="rec_v1"),
        ]
    )
    session.flush()
    session.add_all(
        [
            Interaction(
                user_id="U1",
                product_id="RA",
                event_type="review",
                occurred_at=datetime(2020, 1, 1, tzinfo=UTC),
                dataset_version="rec_v1",
            ),
            Interaction(
                user_id="U1",
                product_id="RB",
                event_type="review",
                occurred_at=datetime(2020, 2, 1, tzinfo=UTC),
                dataset_version="rec_v1",
            ),
            Interaction(
                user_id="U2",
                product_id="RC",
                event_type="review",
                occurred_at=datetime(2021, 1, 1, tzinfo=UTC),
                dataset_version="rec_v1",
            ),
            Interaction(
                user_id="U3",
                product_id="RA",
                event_type="review",
                occurred_at=datetime(2021, 3, 1, tzinfo=UTC),
                dataset_version="rec_v1",
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
                [0.7, 0.7, 0.0],
            ],
            dtype=np.float32,
        )
    )
    runtime = SemanticRuntime(
        artifact_version="rec-test",
        dataset_version="rec_v1",
        model_name="fake-encoder",
        embedding_dim=3,
        product_ids=np.array(["RA", "RB", "RC", "RD"], dtype="U8"),
        index=build_flat_ip_index(embeddings),
        backend="flat",
        embedding_manifest={},
        index_manifest={},
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
        model.item_embedding.weight[0, 0] = 0.1  # RA
        model.item_embedding.weight[1, 0] = 0.2  # RB
        model.item_embedding.weight[2, 0] = 5.0  # RC
        model.user_embedding.weight[1, 1] = 1.0  # U2
        model.item_embedding.weight[0, 1] = 3.0
    users = ["U1", "U2"]
    items = ["RA", "RB", "RC"]
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


@pytest.fixture(autouse=True)
def _reset_runtimes() -> None:
    reset_semantic_runtime()
    reset_cf_runtime()
    yield
    reset_semantic_runtime()
    reset_cf_runtime()
    clear_settings_cache()


def test_hybrid_unknown_user_404(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    assert client.get("/recommendations/user/NOPE?method=hybrid").status_code == 404


def test_hybrid_omitted_method_stays_content(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    response = client.get("/recommendations/user/U1?top_k=5")
    assert response.status_code == 200
    assert response.json()["recommendation_mode"] == "user_content"


def test_hybrid_warm_user_uses_three_channels(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    response = client.get("/recommendations/user/U1?method=hybrid&top_k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_mode"] == USER_HYBRID_MODE
    assert body["hybrid_version"] == "hybrid-rec-v1"
    assert body["channels_used"] == ["content", "cf", "popularity"]
    assert body["fallback_reason"] is None
    ids = [row["product_id"] for row in body["results"]]
    assert "RA" not in ids
    assert "RB" not in ids
    assert ids
    assert all(row["source"] == SOURCE_HYBRID for row in body["results"])
    assert all(math_isfinite(row["score"]) for row in body["results"])


def math_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def test_hybrid_non_cf_user_falls_back_to_content_popularity(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    cf = client.get("/recommendations/user/U3?method=cf")
    assert cf.status_code == 200
    assert cf.json()["reason"] == "cf_user_unavailable"
    assert cf.json()["results"] == []
    hybrid = client.get("/recommendations/user/U3?method=hybrid&top_k=5")
    assert hybrid.status_code == 200
    body = hybrid.json()
    assert body["channels_used"] == ["content", "popularity"]
    assert body["results"]
    assert body["recommendation_mode"] == USER_HYBRID_MODE
    assert "cf_user_unavailable" not in str(body)


def test_hybrid_no_history_falls_back_to_popularity(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    response = client.get("/recommendations/user/UEMPTY?method=hybrid&top_k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["channels_used"] == ["popularity"]
    assert body["fallback_reason"] == REASON_NO_PERSONALIZED_HISTORY
    assert body["results"]
    assert body["results"][0]["source"] == SOURCE_HYBRID
    ids = [row["product_id"] for row in body["results"]]
    assert ids[0] in {"RA", "RB", "RC", "RD"}


def test_hybrid_missing_cf_artifact_degrades(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path / "missing"))
    clear_settings_cache()
    reset_cf_runtime()
    strict = client.get("/recommendations/user/U1?method=cf")
    assert strict.status_code == 503
    hybrid = client.get("/recommendations/user/U1?method=hybrid&top_k=5")
    assert hybrid.status_code == 200
    body = hybrid.json()
    assert body["channels_used"] == ["content", "popularity"]
    assert body["fallback_reason"] == REASON_CF_ARTIFACT_UNAVAILABLE
    assert body["results"]


def test_hybrid_missing_semantic_degrades_to_cf_popularity(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    hybrid = client.get("/recommendations/user/U1?method=hybrid&top_k=5")
    assert hybrid.status_code == 200
    body = hybrid.json()
    assert body["channels_used"] == ["cf", "popularity"]
    assert body["fallback_reason"] == REASON_CONTENT_ARTIFACT_UNAVAILABLE
    ids = [row["product_id"] for row in body["results"]]
    assert "RA" not in ids
    assert "RB" not in ids
    assert "RC" in ids


def test_content_and_cf_methods_stay_strict(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    content = client.get("/recommendations/user/U1?method=content&top_k=5")
    assert content.status_code == 200
    assert content.json()["recommendation_mode"] == "user_content"
    assert content.json()["results"][0]["source"] == "content_user"
    cf = client.get("/recommendations/user/U1?method=cf&top_k=5")
    assert cf.status_code == 200
    assert cf.json()["recommendation_mode"] == "user_cf"
    assert cf.json()["results"][0]["source"] == "cf"
    trending = client.get("/recommendations/trending?top_k=3")
    assert trending.json()["recommendation_mode"] == "popularity"
    similar = client.get("/recommendations/similar/RA?top_k=2")
    assert similar.json()["recommendation_mode"] == "similar_content"


def test_hybrid_cf_cold_item_eligible_through_content(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    response = client.get("/recommendations/user/U1?method=hybrid&top_k=5")
    ids = [row["product_id"] for row in response.json()["results"]]
    assert "RD" in ids
    cf = client.get("/recommendations/user/U1?method=cf&top_k=5")
    assert "RD" not in [row["product_id"] for row in cf.json()["results"]]
