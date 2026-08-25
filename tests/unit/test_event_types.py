"""Event-type validation without PostgreSQL."""

import pytest

from app.core.events import EVENT_TYPE_SET
from app.db.repositories.interactions import InteractionError, record_interaction


class _DummySession:
    pass


def test_known_event_types() -> None:
    assert EVENT_TYPE_SET == {"view", "click", "add_to_cart", "purchase", "review"}


def test_record_interaction_rejects_unknown_event() -> None:
    with pytest.raises(InteractionError, match="invalid event_type"):
        record_interaction(_DummySession(), user_id="u", product_id="p", event_type="subscribe")
