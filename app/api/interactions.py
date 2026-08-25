"""Persist a user–item event. Not a recommendation endpoint."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.repositories.interactions import InteractionError, record_interaction
from app.db.session import session_scope
from app.schemas.interactions import InteractionCreateRequest, InteractionResponse

router = APIRouter(tags=["interactions"])


@router.post(
    "/interactions",
    response_model=InteractionResponse,
    responses={400: {"description": "Invalid event"}, 404: {"description": "Unknown product"}},
)
def post_interaction(
    payload: InteractionCreateRequest,
    session: Session = Depends(session_scope),
) -> InteractionResponse:
    try:
        row, created = record_interaction(
            session,
            user_id=payload.user_id,
            product_id=payload.product_id,
            event_type=payload.event_type,
            event_value=float(payload.event_value) if payload.event_value is not None else None,
            occurred_at=payload.occurred_at,
            metadata=payload.metadata,
        )
        session.commit()
    except InteractionError as exc:
        session.rollback()
        message = str(exc)
        status = 404 if "does not exist" in message else 400
        raise HTTPException(status_code=status, detail="Invalid interaction") from exc
    return InteractionResponse(
        interaction_id=row.interaction_id,
        user_id=row.user_id,
        product_id=row.product_id,
        event_type=row.event_type,
        event_value=row.event_value,
        occurred_at=row.occurred_at,
        created=created,
    )
