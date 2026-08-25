"""Exact product lookup. Not search."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.repositories.products import get_product
from app.db.session import session_scope
from app.schemas.products import ProductResponse

router = APIRouter(tags=["products"])


@router.get(
    "/products/{product_id}",
    response_model=ProductResponse,
    responses={404: {"description": "Unknown product"}},
)
def get_product_by_id(
    product_id: str,
    session: Session = Depends(session_scope),
) -> ProductResponse:
    product = get_product(session, product_id)
    if product is None:
        raise HTTPException(status_code=404, detail="Product not found")
    return ProductResponse(
        product_id=product.product_id,
        title=product.title,
        description=product.description,
        category=product.category,
        subcategory=product.subcategory,
        brand=product.brand,
        store=product.store,
        price=product.price,
        currency=product.currency,
        average_rating=product.average_rating,
        rating_count=product.rating_count,
    )
