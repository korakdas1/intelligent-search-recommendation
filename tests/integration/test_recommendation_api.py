"""Recommendation HTTP tests against PostgreSQL. No MiniLM download."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.embeddings.normalize import l2_normalize
from app.models.dataset import DatasetVersion
from app.models.interaction import Interaction
from app.models.product import Product
from app.models.user import User
from app.recommendations.evaluation import leave_last_product_split
from app.recommendations.popularity import popularity_counts
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
                product_id="RA",
                event_type="view",
                occurred_at=datetime(2020, 1, 2, tzinfo=UTC),
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
                user_id="U2",
                product_id="RC",
                event_type="review",
                occurred_at=datetime(2021, 6, 1, tzinfo=UTC),
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


@pytest.fixture(autouse=True)
def _reset_runtime() -> None:
    reset_semantic_runtime()
    yield
    reset_semantic_runtime()


def test_similar_unknown_product_404(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    response = client.get("/recommendations/similar/NOPE")
    assert response.status_code == 404


def test_similar_excludes_self_and_respects_top_k(
    client: TestClient, db_session: Session
) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    response = client.get("/recommendations/similar/RA?top_k=2")
    assert response.status_code == 200
    body = response.json()
    ids = [row["product_id"] for row in body["results"]]
    assert "RA" not in ids
    assert ids[0] == "RB"
    assert body["recommendation_mode"] == "similar_content"
    assert body["results"][0]["source"] == "content_item"
    assert "features" not in body["results"][0]


def test_similar_top_k_bounds(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    assert client.get("/recommendations/similar/RA?top_k=0").status_code == 422
    assert client.get("/recommendations/similar/RA?top_k=101").status_code == 422


def test_trending_popularity_order(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    response = client.get("/recommendations/trending?top_k=5")
    assert response.status_code == 200
    body = response.json()
    assert body["recommendation_mode"] == "popularity"
    ids = [row["product_id"] for row in body["results"]]
    assert ids[0] == "RA"
    assert body["results"][0]["source"] == "popularity"
    assert body["results"][0]["score"] == 2.0
    counts = popularity_counts(db_session)
    assert counts["RA"] == 2
    assert counts["RB"] == 1
    assert counts["RC"] == 2


def test_user_unknown_404(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    assert client.get("/recommendations/user/NOPE?method=content").status_code == 404


def test_user_no_history_empty(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    response = client.get("/recommendations/user/UEMPTY?method=content")
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["reason"] == "no_usable_content_history"


def test_user_content_filters_seen(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    _install_runtime()
    response = client.get("/recommendations/user/U1?method=content&top_k=5")
    assert response.status_code == 200
    body = response.json()
    ids = [row["product_id"] for row in body["results"]]
    assert "RA" not in ids
    assert "RB" not in ids
    assert body["history_items"] == 2
    assert body["results"][0]["source"] == "content_user"


def test_user_method_invalid_rejected(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    assert client.get("/recommendations/user/U1?method=als").status_code == 422


def test_train_safe_popularity_sql(db_session: Session) -> None:
    _seed_catalog(db_session)
    train_ids = [
        row.interaction_id
        for row in db_session.scalars(select(Interaction))
        if not (row.user_id == "U2" and row.product_id == "RC")
    ]
    # RC's events are held out; remaining RA=2, RB=1
    train_counts = popularity_counts(db_session, interaction_ids=train_ids)
    assert train_counts.get("RC", 0) == 0
    assert train_counts["RA"] == 2
    full = popularity_counts(db_session)
    assert full["RC"] == 2


def test_hidden_item_not_filtered_from_eval_candidates() -> None:
    users = {
        "U1": [
            ("RA", datetime(2020, 1, 1, tzinfo=UTC)),
            ("RB", datetime(2020, 2, 1, tzinfo=UTC)),
        ]
    }
    split = leave_last_product_split(users)
    assert split[0].hidden_product_id == "RB"
    seen = set(split[0].train_product_ids)
    assert "RB" not in seen


def test_search_still_keyword(client: TestClient, db_session: Session) -> None:
    _seed_catalog(db_session)
    response = client.post("/search", json={"query": "alpha cream", "top_k": 1})
    assert response.status_code == 200
    assert response.json()["retrieval_mode"] == "keyword"
