"""PostgreSQL full-text search against the dedicated test database."""

from __future__ import annotations

from decimal import Decimal

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion
from app.models.product import Product
from app.models.search import SearchEvent, SearchEventResult
from app.search.document import GIN_INDEX_NAME

pytestmark = pytest.mark.postgres


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
                product_id="TITLEMATCH",
                dataset_version="search_v1",
                title="Lexemealpha Widget",
                description="unrelated soap text",
                category="All Beauty",
                brand="Acme",
                price=Decimal("4.00"),
                currency="USD",
            ),
            Product(
                product_id="DESCMATCH",
                dataset_version="search_v1",
                title="Plain Soap Bar",
                description="contains lexemealpha only in the long description",
                category="All Beauty",
                brand="Other",
                price=Decimal("5.00"),
                currency="USD",
            ),
            Product(
                product_id="CREAMPLU",
                dataset_version="search_v1",
                title="Shea Butter Creams",
                description=None,
                category="All Beauty",
                currency="USD",
            ),
            Product(
                product_id="PAAA",
                dataset_version="search_v1",
                title="Tietokenonly Oil",
                category="All Beauty",
                currency="USD",
            ),
            Product(
                product_id="PBBB",
                dataset_version="search_v1",
                title="Tietokenonly Oil",
                category="All Beauty",
                currency="USD",
            ),
            Product(
                product_id="CASEFOLD",
                dataset_version="search_v1",
                title="LEATHER OIL",
                category="All Beauty",
                currency="USD",
            ),
        ]
    )
    session.commit()


def test_search_document_and_gin_exist(postgres_engine, db_session: Session) -> None:
    del db_session
    with postgres_engine.connect() as connection:
        columns = connection.execute(
            text(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'products' AND column_name = 'search_document'"
            )
        ).scalar_one_or_none()
        assert columns == "search_document"
        indexdef = connection.execute(
            text(
                "SELECT indexdef FROM pg_indexes "
                "WHERE tablename = 'products' AND indexname = :name"
            ),
            {"name": GIN_INDEX_NAME},
        ).scalar_one()
        assert "gin" in indexdef.lower()


def test_title_weight_outranks_description(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post("/search", json={"query": "lexemealpha", "top_k": 5})
    assert response.status_code == 200
    ids = [row["product_id"] for row in response.json()["results"]]
    assert ids[0] == "TITLEMATCH"
    assert "DESCMATCH" in ids
    assert ids.index("TITLEMATCH") < ids.index("DESCMATCH")


def test_stemming_matches_plural(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post("/search", json={"query": "cream", "top_k": 10})
    assert response.status_code == 200
    ids = [row["product_id"] for row in response.json()["results"]]
    assert "CREAMPLU" in ids


def test_case_insensitive(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post("/search", json={"query": "leather oil", "top_k": 10})
    assert response.status_code == 200
    ids = [row["product_id"] for row in response.json()["results"]]
    assert "CASEFOLD" in ids


def test_no_matches_returns_empty_list(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post("/search", json={"query": "xyzzyqwerty12345notaproduct"})
    assert response.status_code == 200
    body = response.json()
    assert body["results"] == []
    assert body["retrieval_mode"] == "keyword"


def test_stopword_only_does_not_crash(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post("/search", json={"query": "the"})
    assert response.status_code == 200
    assert response.json()["results"] == []


def test_punctuation_does_not_crash(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post(
        "/search",
        json={"query": "O'Reilly SPF/50 (cream) & more?", "top_k": 5},
    )
    assert response.status_code == 200
    assert "results" in response.json()


def test_sql_injection_looking_input_is_safe(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    payload = {"query": "lexemealpha'); DROP TABLE products; --", "top_k": 5}
    response = client.post("/search", json=payload)
    assert response.status_code == 200
    remaining = db_session.scalar(select(func.count()).select_from(Product))
    assert remaining == 6


def test_top_k_limit(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post("/search", json={"query": "tietokenonly", "top_k": 1})
    assert response.status_code == 200
    assert len(response.json()["results"]) == 1


def test_tie_order_is_deterministic(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    first = client.post("/search", json={"query": "tietokenonly", "top_k": 10})
    second = client.post("/search", json={"query": "tietokenonly", "top_k": 10})
    ids_first = [row["product_id"] for row in first.json()["results"]]
    ids_second = [row["product_id"] for row in second.json()["results"]]
    assert ids_first == ids_second
    assert ids_first[:2] == ["PAAA", "PBBB"]
    assert first.json()["results"][0]["rank"] == 1


def test_search_logging(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post("/search", json={"query": "lexemealpha", "top_k": 5})
    assert response.status_code == 200
    db_session.expire_all()
    event = db_session.scalar(select(SearchEvent).order_by(SearchEvent.search_event_id.desc()))
    assert event is not None
    assert event.query_text == "lexemealpha"
    assert event.label_class == "observed"
    assert event.user_id is None
    assert event.metadata_["retrieval_mode"] == "keyword"
    rows = list(
        db_session.scalars(
            select(SearchEventResult).where(
                SearchEventResult.search_event_id == event.search_event_id
            )
        )
    )
    assert rows
    assert all(row.source == "keyword" for row in rows)
    assert rows[0].rank == 1


def test_phase3_interaction_still_works(client: TestClient, db_session: Session) -> None:
    _seed(db_session)
    response = client.post(
        "/interactions",
        json={
            "user_id": "SEARCHUSER",
            "product_id": "TITLEMATCH",
            "event_type": "view",
        },
    )
    assert response.status_code == 200
    assert response.json()["created"] is True


def test_ready_and_health(client: TestClient, db_session: Session) -> None:
    del db_session
    ready = client.get("/ready")
    assert ready.status_code == 200
    assert ready.json()["checks"]["database"] == "ok"
    health = client.get("/health")
    assert health.status_code == 200
    assert "database" not in health.json()
