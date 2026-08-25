"""Recommendation HTTP routes. Separate from /search."""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.db.session import session_scope
from app.recommendations.exceptions import (
    CfUnavailableError,
    SourceProductNotFound,
    SourceProductNotInIndex,
    UserNotFound,
)
from app.recommendations.service import (
    recommend_cf_for_user,
    recommend_content_for_user,
    recommend_hybrid_for_user,
    recommend_popular,
    recommend_similar,
    validate_user_method,
)
from app.schemas.recommendations import RecommendationResponse
from app.search.exceptions import SemanticUnavailableError

router = APIRouter(prefix="/recommendations", tags=["recommendations"])


def _semantic_http(exc: SemanticUnavailableError) -> HTTPException:
    return HTTPException(
        status_code=503,
        detail={"status": "semantic_unavailable", "message": str(exc)},
    )


@router.get("/similar/{product_id}", response_model=RecommendationResponse)
def get_similar(
    product_id: str,
    top_k: int = Query(default=10, ge=1, le=100),
    session: Session = Depends(session_scope),
) -> RecommendationResponse:
    try:
        return recommend_similar(session, product_id, top_k=top_k)
    except SourceProductNotFound as exc:
        raise HTTPException(status_code=404, detail="Product not found") from exc
    except SourceProductNotInIndex as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "semantic_unavailable",
                "message": f"product {exc} missing from semantic artifact",
            },
        ) from exc
    except SemanticUnavailableError as exc:
        raise _semantic_http(exc) from exc


@router.get("/trending", response_model=RecommendationResponse)
def get_trending(
    top_k: int = Query(default=10, ge=1, le=100),
    session: Session = Depends(session_scope),
) -> RecommendationResponse:
    return recommend_popular(session, top_k=top_k)


@router.get("/user/{user_id}", response_model=RecommendationResponse)
def get_user_recommendations(
    user_id: str,
    method: str = Query(default="content"),
    top_k: int = Query(default=10, ge=1, le=100),
    session: Session = Depends(session_scope),
) -> RecommendationResponse:
    try:
        chosen = validate_user_method(method)
        if chosen == "cf":
            return recommend_cf_for_user(session, user_id, top_k=top_k)
        if chosen == "hybrid":
            return recommend_hybrid_for_user(session, user_id, top_k=top_k)
        return recommend_content_for_user(session, user_id, top_k=top_k)
    except UserNotFound as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CfUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={"status": "cf_unavailable", "message": str(exc)},
        ) from exc
    except SemanticUnavailableError as exc:
        raise _semantic_http(exc) from exc
