"""SQLAlchemy 2.x declarative base. Importing this module does not connect."""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared metadata for Alembic and the ORM."""
