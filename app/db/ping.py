"""Lightweight database readiness probe. Does not query business tables."""

from __future__ import annotations

import logging

from sqlalchemy import text

from app.db.session import get_engine

logger = logging.getLogger(__name__)


def ping_database() -> bool:
    """Return True if ``SELECT 1`` succeeds. Never logs credentials."""

    try:
        engine = get_engine()
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        logger.warning("PostgreSQL readiness check failed")
        return False
