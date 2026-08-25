"""Add weighted tsvector search_document and GIN index.

Revision ID: 20260822_0002
Revises: 20260822_0001
Create Date: 2026-08-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from app.search.document import GIN_INDEX_NAME, SEARCH_DOCUMENT_SQL

revision: str = "20260822_0002"
down_revision: Union[str, Sequence[str], None] = "20260822_0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "products",
        sa.Column(
            "search_document",
            postgresql.TSVECTOR(),
            sa.Computed(SEARCH_DOCUMENT_SQL, persisted=True),
            nullable=True,
        ),
    )
    op.create_index(
        GIN_INDEX_NAME,
        "products",
        ["search_document"],
        postgresql_using="gin",
    )
    op.execute(sa.text("ANALYZE products"))


def downgrade() -> None:
    op.drop_index(GIN_INDEX_NAME, table_name="products")
    op.drop_column("products", "search_document")
