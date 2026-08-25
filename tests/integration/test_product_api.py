"""Product lookup and interaction HTTP API against PostgreSQL."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion
from app.models.product import Product
from app.models.user import User

pytestmark = pytest.mark.postgres


def _seed(session: Session) -> None:
    session.add(
        DatasetVersion(
            dataset_version="api_v1",
            source_name="fixture",
            row_counts={},
            checksums={},
        )
    )
    session.add(
        Product(
            product_id="B00API",
            dataset_version="api_v1",
            title="API Cream",
            category="All Beauty",
            price=Decimal("3.00"),
            currency="USD",
        )
    )
    session.commit()


def test_get_product_200(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.get("/products/B00API")
    assert response.status_code == 200
    body = response.json()
    assert body["product_id"] == "B00API"
    assert body["title"] == "API Cream"
    assert "metadata" not in body


def test_get_product_404(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.get("/products/DOES-NOT-EXIST")
    assert response.status_code == 404
    assert response.json()["detail"] == "Product not found"


def test_post_interaction_creates_user(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    when = datetime(2021, 5, 1, 8, 30, tzinfo=UTC).isoformat()
    response = client.post(
        "/interactions",
        json={
            "user_id": "NEWUSER",
            "product_id": "B00API",
            "event_type": "view",
            "occurred_at": when,
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["created"] is True
    assert body["event_type"] == "view"
    db_session.expire_all()
    user = db_session.get(User, "NEWUSER")
    assert user is not None
    assert user.first_seen_at is not None


def test_post_interaction_unknown_product(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post(
        "/interactions",
        json={"user_id": "U1", "product_id": "NOPE", "event_type": "review"},
    )
    assert response.status_code == 404


def test_post_interaction_invalid_event_type(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post(
        "/interactions",
        json={"user_id": "U1", "product_id": "B00API", "event_type": "subscribe"},
    )
    assert response.status_code == 422


def test_post_interaction_idempotent(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    payload = {
        "user_id": "U2",
        "product_id": "B00API",
        "event_type": "review",
        "event_value": 4,
        "occurred_at": datetime(2021, 1, 1, tzinfo=UTC).isoformat(),
    }
    first = client.post("/interactions", json=payload)
    second = client.post("/interactions", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["interaction_id"] == second.json()["interaction_id"]
    assert first.json()["created"] is True
    assert second.json()["created"] is False


def test_ready_with_database(client: TestClient, db_session: Session) -> None:
    del db_session
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["checks"]["database"] == "ok"
    health = client.get("/health")
    assert health.status_code == 200
