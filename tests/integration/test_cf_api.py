"""CF recommendation HTTP tests. Tiny fixture model; no full-catalog training."""

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
        ]
    )
    session.add_all(
        [
            User(user_id="U1", dataset_version="rec_v1"),
            User(user_id="U2", dataset_version="rec_v1"),
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
            ],
            dtype=np.float32,
        )
    )
    runtime = SemanticRuntime(
        artifact_version="rec-test",
        dataset_version="rec_v1",
        model_name="fake-encoder",
        embedding_dim=3,
        product_ids=np.array(["RA", "RB", "RC"], dtype="U8"),
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


def test_user_method_invalid_rejected(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    assert client.get("/recommendations/user/U1?method=als").status_code == 422


def test_default_method_is_content(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    response = client.get("/recommendations/user/U1?top_k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_mode"] == "user_content"
    ids = [row["product_id"] for row in body["results"]]
    assert "RA" not in ids
    assert "RB" not in ids


def test_cf_unknown_user_404(client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _seed_catalog(db_session)
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    assert client.get("/recommendations/user/NOPE?method=cf").status_code == 404


def test_cf_known_db_user_outside_model(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    response = client.get("/recommendations/user/UEMPTY?method=cf")
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["reason"] == "cf_user_unavailable"
    assert body["recommendation_mode"] == "user_cf"


def test_cf_filters_seen_and_ranks(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    response = client.get("/recommendations/user/U1?method=cf&top_k=5")
    assert response.status_code == 200
    body = response.json()
    ids = [row["product_id"] for row in body["results"]]
    assert "RA" not in ids
    assert "RB" not in ids
    assert ids[0] == "RC"
    assert body["results"][0]["source"] == "cf"
    assert body["recommendation_mode"] == "user_cf"
    assert body["cf_model_version"] == "bpr-mf-test"


def test_cf_missing_artifact_is_503_not_content(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path / "missing"))
    clear_settings_cache()
    reset_cf_runtime()
    response = client.get("/recommendations/user/U1?method=cf")
    assert response.status_code == 503
    assert response.json()["detail"]["status"] == "cf_unavailable"
    content = client.get("/recommendations/user/U1?method=content")
    assert content.status_code == 200
    assert content.json()["recommendation_mode"] == "user_content"


def test_content_method_unchanged_with_cf_artifact(
    client: TestClient, db_session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    _write_cf_artifact(tmp_path)
    monkeypatch.setenv("CF_MODEL_DIR", str(tmp_path))
    clear_settings_cache()
    reset_cf_runtime()
    response = client.get("/recommendations/user/U1?method=content&top_k=5")
    assert response.status_code == 200
    assert response.json()["recommendation_mode"] == "user_content"
    assert response.json()["results"][0]["source"] == "content_user"
