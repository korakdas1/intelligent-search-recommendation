"""Initial catalog schema without full-text search.

Revision ID: 20260822_0001
Revises:
Create Date: 2026-08-22
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "20260822_0001"
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "dataset_versions",
        sa.Column("dataset_version", sa.Text(), primary_key=True),
        sa.Column("source_name", sa.Text(), nullable=False),
        sa.Column("source_slice", sa.Text(), nullable=True),
        sa.Column("source_url", sa.Text(), nullable=True),
        sa.Column("license_note", sa.Text(), nullable=True),
        sa.Column(
            "row_counts",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "checksums",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )

    op.create_table(
        "products",
        sa.Column("product_id", sa.Text(), primary_key=True),
        sa.Column(
            "dataset_version",
            sa.Text(),
            sa.ForeignKey("dataset_versions.dataset_version"),
            nullable=False,
        ),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("category", sa.Text(), nullable=True),
        sa.Column("subcategory", sa.Text(), nullable=True),
        sa.Column("brand", sa.Text(), nullable=True),
        sa.Column("store", sa.Text(), nullable=True),
        sa.Column("price", sa.Numeric(12, 2), nullable=True),
        sa.Column("currency", sa.String(length=3), nullable=False, server_default="USD"),
        sa.Column("average_rating", sa.Numeric(3, 2), nullable=True),
        sa.Column("rating_count", sa.Integer(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
    )
    op.create_index("ix_products_category", "products", ["category"])
    op.create_index("ix_products_brand", "products", ["brand"])
    op.create_index("ix_products_price", "products", ["price"])

    op.create_table(
        "users",
        sa.Column("user_id", sa.Text(), primary_key=True),
        sa.Column(
            "dataset_version",
            sa.Text(),
            sa.ForeignKey("dataset_versions.dataset_version"),
            nullable=True,
        ),
        sa.Column("first_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )

    op.create_table(
        "interactions",
        sa.Column("interaction_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.user_id"), nullable=False),
        sa.Column("product_id", sa.Text(), sa.ForeignKey("products.product_id"), nullable=False),
        sa.Column("event_type", sa.Text(), nullable=False),
        sa.Column("event_value", sa.Numeric(), nullable=True),
        sa.Column("occurred_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "dataset_version",
            sa.Text(),
            sa.ForeignKey("dataset_versions.dataset_version"),
            nullable=True,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.CheckConstraint(
            "event_type IN ('view', 'click', 'add_to_cart', 'purchase', 'review')",
            name="ck_interactions_event_type",
        ),
        sa.UniqueConstraint(
            "user_id",
            "product_id",
            "event_type",
            "occurred_at",
            name="uq_interactions_user_product_type_time",
        ),
    )
    op.create_index("ix_interactions_user_occurred", "interactions", ["user_id", "occurred_at"])
    op.create_index("ix_interactions_product_event", "interactions", ["product_id", "event_type"])
    op.create_index("ix_interactions_occurred_at", "interactions", ["occurred_at"])

    op.create_table(
        "search_events",
        sa.Column("search_event_id", sa.BigInteger(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Text(), sa.ForeignKey("users.user_id"), nullable=True),
        sa.Column("query_text", sa.Text(), nullable=False),
        sa.Column("normalized_query", sa.Text(), nullable=True),
        sa.Column("top_k", sa.Integer(), nullable=True),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column("clicked_product_id", sa.Text(), sa.ForeignKey("products.product_id"), nullable=True),
        sa.Column(
            "converted_product_id",
            sa.Text(),
            sa.ForeignKey("products.product_id"),
            nullable=True,
        ),
        sa.Column(
            "dataset_version",
            sa.Text(),
            sa.ForeignKey("dataset_versions.dataset_version"),
            nullable=True,
        ),
        sa.Column("label_class", sa.Text(), nullable=True),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.create_index("ix_search_events_user_occurred", "search_events", ["user_id", "occurred_at"])

    op.create_table(
        "search_event_results",
        sa.Column(
            "search_event_id",
            sa.BigInteger(),
            sa.ForeignKey("search_events.search_event_id"),
            primary_key=True,
        ),
        sa.Column(
            "product_id",
            sa.Text(),
            sa.ForeignKey("products.product_id"),
            primary_key=True,
        ),
        sa.Column("rank", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("source", sa.Text(), nullable=True),
        sa.CheckConstraint("rank >= 1", name="ck_search_event_results_rank"),
    )
    op.create_index(
        "ix_search_event_results_event_rank",
        "search_event_results",
        ["search_event_id", "rank"],
    )

    op.create_table(
        "artifact_versions",
        sa.Column("artifact_id", sa.Text(), primary_key=True),
        sa.Column("artifact_type", sa.Text(), nullable=True),
        sa.Column("version", sa.Text(), nullable=True),
        sa.Column(
            "dataset_version",
            sa.Text(),
            sa.ForeignKey("dataset_versions.dataset_version"),
            nullable=True,
        ),
        sa.Column("embedding_model_name", sa.Text(), nullable=True),
        sa.Column("embedding_dim", sa.Integer(), nullable=True),
        sa.Column("metric", sa.Text(), nullable=True),
        sa.Column("path", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )


def downgrade() -> None:
    op.drop_table("artifact_versions")
    op.drop_index("ix_search_event_results_event_rank", table_name="search_event_results")
    op.drop_table("search_event_results")
    op.drop_index("ix_search_events_user_occurred", table_name="search_events")
    op.drop_table("search_events")
    op.drop_index("ix_interactions_occurred_at", table_name="interactions")
    op.drop_index("ix_interactions_product_event", table_name="interactions")
    op.drop_index("ix_interactions_user_occurred", table_name="interactions")
    op.drop_table("interactions")
    op.drop_table("users")
    op.drop_index("ix_products_price", table_name="products")
    op.drop_index("ix_products_brand", table_name="products")
    op.drop_index("ix_products_category", table_name="products")
    op.drop_table("products")
    op.drop_table("dataset_versions")
