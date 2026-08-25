"""Product data access."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.product import Product


def get_product(session: Session, product_id: str) -> Product | None:
    return session.get(Product, product_id)


def product_exists(session: Session, product_id: str) -> bool:
    stmt = select(Product.product_id).where(Product.product_id == product_id)
    return session.execute(stmt).scalar_one_or_none() is not None


def get_products_by_ids(session: Session, product_ids: Sequence[str]) -> dict[str, Product]:
    """Fetch many products in one query. Caller restores ranking order."""

    if not product_ids:
        return {}
    rows = session.scalars(select(Product).where(Product.product_id.in_(list(product_ids)))).all()
    return {row.product_id: row for row in rows}


def iter_products_ordered(
    session: Session,
    *,
    dataset_version: str | None = None,
    offset: int = 0,
    limit: int = 1000,
) -> list[Product]:
    stmt = select(Product).order_by(Product.product_id.asc()).offset(offset).limit(limit)
    if dataset_version is not None:
        stmt = stmt.where(Product.dataset_version == dataset_version)
    return list(session.scalars(stmt))


def count_products(session: Session, *, dataset_version: str | None = None) -> int:
    stmt = select(func.count()).select_from(Product)
    if dataset_version is not None:
        stmt = stmt.where(Product.dataset_version == dataset_version)
    return int(session.scalar(stmt) or 0)
