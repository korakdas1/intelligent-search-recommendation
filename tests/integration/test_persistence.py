"""ORM persistence, FK, uniqueness, JSONB, and timestamptz."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.dataset import DatasetVersion
from app.models.interaction import Interaction
from app.models.product import Product
from app.models.user import User

pytestmark = pytest.mark.postgres


def _seed_catalog(session: Session) -> None:
    session.add(
        DatasetVersion(
            dataset_version="test_v1",
            source_name="fixture",
            source_slice="unit",
            row_counts={},
            checksums={},
        )
    )
    session.add(
        Product(
            product_id="P1",
            dataset_version="test_v1",
            title="Test Cream",
            brand="Acme",
            store="Acme Store",
            price=Decimal("6.50"),
            currency="USD",
            metadata_={"has_features": True},
        )
    )
    session.add(
        User(
            user_id="U1",
            dataset_version="test_v1",
            first_seen_at=datetime(2020, 1, 1, tzinfo=UTC),
            last_seen_at=datetime(2020, 1, 2, tzinfo=UTC),
        )
    )
    session.commit()


def test_product_and_interaction_roundtrip(db_session: Session) -> None:
    _seed_catalog(db_session)
    occurred = datetime(2020, 6, 1, 12, 0, tzinfo=UTC)
    db_session.add(
        Interaction(
            user_id="U1",
            product_id="P1",
            event_type="review",
            event_value=Decimal("5.0"),
            occurred_at=occurred,
            dataset_version="test_v1",
            metadata_={"verified_purchase": True, "source_asin": "CHILD1"},
        )
    )
    db_session.commit()
    product = db_session.get(Product, "P1")
    assert product is not None
    assert product.brand == "Acme"
    assert product.store == "Acme Store"
    assert product.metadata_["has_features"] is True
    row = db_session.query(Interaction).one()
    assert row.occurred_at.tzinfo is not None
    assert row.metadata_["verified_purchase"] is True
    assert row.event_type == "review"


def test_foreign_key_rejects_unknown_product(db_session: Session) -> None:
    _seed_catalog(db_session)
    db_session.add(
        Interaction(
            user_id="U1",
            product_id="MISSING",
            event_type="review",
            occurred_at=datetime(2020, 1, 1, tzinfo=UTC),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_event_type_constraint(db_session: Session) -> None:
    _seed_catalog(db_session)
    db_session.add(
        Interaction(
            user_id="U1",
            product_id="P1",
            event_type="subscribe",
            occurred_at=datetime(2020, 1, 1, tzinfo=UTC),
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


def test_unique_user_product_type_time(db_session: Session) -> None:
    _seed_catalog(db_session)
    when = datetime(2020, 1, 1, tzinfo=UTC)
    db_session.add(
        Interaction(
            user_id="U1",
            product_id="P1",
            event_type="review",
            occurred_at=when,
        )
    )
    db_session.commit()
    db_session.add(
        Interaction(
            user_id="U1",
            product_id="P1",
            event_type="review",
            occurred_at=when,
        )
    )
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()
