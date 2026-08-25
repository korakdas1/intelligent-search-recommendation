"""Product search. Keyword (default), semantic, or hybrid fusion."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import session_scope
from app.schemas.search import SearchRequest, SearchResponse
from app.search.exceptions import RankerUnavailableError, SearchUserNotFound, SemanticUnavailableError
from app.search.service import run_search

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
def post_search(
    payload: SearchRequest,
    session: Session = Depends(session_scope),
) -> SearchResponse:
    try:
        response = run_search(
            session,
            query=payload.query,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
            fusion_method=payload.fusion_method,
            rerank_mode=payload.rerank_mode,
            personalization_mode=payload.personalization_mode,
            user_id=payload.user_id,
            log=True,
        )
    except SemanticUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "semantic_unavailable",
                "message": str(exc),
            },
        ) from exc
    except RankerUnavailableError as exc:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "ranker_unavailable",
                "message": str(exc),
            },
        ) from exc
    except SearchUserNotFound as exc:
        raise HTTPException(status_code=404, detail="User not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    session.commit()
    return response
