"""Product data access."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.embeddings.provenance import SemanticCatalogFingerprint
from app.embeddings.text import build_semantic_text
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
    stmt = select(Product).order_by(Product.product_id.collate("C").asc()).offset(offset).limit(limit)
    if dataset_version is not None:
        stmt = stmt.where(Product.dataset_version == dataset_version)
    return list(session.scalars(stmt))


def count_products(session: Session, *, dataset_version: str | None = None) -> int:
    stmt = select(func.count()).select_from(Product)
    if dataset_version is not None:
        stmt = stmt.where(Product.dataset_version == dataset_version)
    return int(session.scalar(stmt) or 0)


def semantic_catalog_fingerprint(session: Session, *, dataset_version: str) -> str:
    """Stream only semantic inputs in the same stable order as the builder."""

    stmt = (
        select(
            Product.product_id, Product.title, Product.brand, Product.category,
            Product.subcategory, Product.description,
        )
        .where(Product.dataset_version == dataset_version)
        .order_by(Product.product_id.collate("C").asc())
        .execution_options(yield_per=1024)
    )
    fingerprint = SemanticCatalogFingerprint()
    with session.execute(stmt) as rows:
        for row in rows:
            fingerprint.update(
                row.product_id,
                build_semantic_text(
                    title=row.title, brand=row.brand, category=row.category,
                    subcategory=row.subcategory, description=row.description,
                ),
            )
    return fingerprint.hexdigest()
