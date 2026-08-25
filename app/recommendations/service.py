"""Recommendation service: similar-item, user-content, popularity, CF, hybrid.

Search ranking is not used. RankNet is not used. method=content and method=cf
remain unfused. Hybrid fusion lives in hybrid_service.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.repositories.products import get_product, get_products_by_ids
from app.db.repositories.users import get_user
from app.recommendations.cf_constants import (
    REASON_ALL_CF_ITEMS_SEEN,
    REASON_CF_USER_UNAVAILABLE,
    SOURCE_CF,
    USER_CF_MODE,
)
from app.recommendations.constants import (
    ALLOWED_USER_METHODS,
    CONTENT_REC_VERSION,
    DEFAULT_USER_METHOD,
    POPULARITY_MODE,
    SIMILAR_MODE,
    SOURCE_CONTENT_ITEM,
    SOURCE_CONTENT_USER,
    SOURCE_POPULARITY,
    USER_CONTENT_MODE,
)
from app.recommendations.hybrid_service import recommend_hybrid_for_user
from app.recommendations.content import content_items_for_profile, similar_items, user_content_profile
from app.recommendations.exceptions import (
    SourceProductNotFound,
    SourceProductNotInIndex,
    UserNotFound,
)
from app.recommendations.history import list_user_interactions, unique_product_ids
from app.recommendations.popularity import popular_products
from app.recommendations.ranking import ScoredItem, stable_rank
from app.schemas.recommendations import RecommendationItem, RecommendationResponse
from app.search.exceptions import SemanticUnavailableError
from app.search.runtime import SemanticRuntime, require_semantic_runtime


def recommend_similar(
    session: Session,
    product_id: str,
    *,
    top_k: int,
    runtime: SemanticRuntime | None = None,
) -> RecommendationResponse:
    product = get_product(session, product_id)
    if product is None:
        raise SourceProductNotFound(product_id)
    runtime = runtime or require_semantic_runtime(session)
    if runtime.product_id_to_row(product_id) is None:
        raise SourceProductNotInIndex(product_id)
    scored = similar_items(runtime, product_id, top_k=top_k)
    results = _attach_metadata(session, scored, source=SOURCE_CONTENT_ITEM)
    return RecommendationResponse(
        recommendation_mode=SIMILAR_MODE,
        content_rec_version=CONTENT_REC_VERSION,
        product_id=product_id,
        top_k=top_k,
        results=results,
    )


def recommend_content_for_user(
    session: Session,
    user_id: str,
    *,
    top_k: int,
    before: datetime | None = None,
    seen_product_ids: Sequence[str] | None = None,
    runtime: SemanticRuntime | None = None,
) -> RecommendationResponse:
    if get_user(session, user_id) is None:
        raise UserNotFound(user_id)
    if seen_product_ids is None:
        history = list_user_interactions(session, user_id, before=before)
        seen = unique_product_ids(history)
    else:
        seen = list(dict.fromkeys(str(item) for item in seen_product_ids))
    if not seen:
        return RecommendationResponse(
            recommendation_mode=USER_CONTENT_MODE,
            content_rec_version=CONTENT_REC_VERSION,
            user_id=user_id,
            history_items=0,
            top_k=top_k,
            reason="no_usable_content_history",
            results=[],
        )
    runtime = runtime or require_semantic_runtime(session)
    profile, usable = user_content_profile(runtime, seen)
    if profile is None:
        return RecommendationResponse(
            recommendation_mode=USER_CONTENT_MODE,
            content_rec_version=CONTENT_REC_VERSION,
            user_id=user_id,
            history_items=0,
            top_k=top_k,
            reason="no_usable_content_history",
            results=[],
        )
    exclude = set(usable)
    scored = content_items_for_profile(runtime, profile, exclude=exclude, top_k=top_k)
    results = _attach_metadata(session, scored, source=SOURCE_CONTENT_USER)
    return RecommendationResponse(
        recommendation_mode=USER_CONTENT_MODE,
        content_rec_version=CONTENT_REC_VERSION,
        user_id=user_id,
        history_items=len(usable),
        top_k=top_k,
        results=results,
    )


def recommend_popular(
    session: Session,
    *,
    top_k: int,
    before: datetime | None = None,
    interaction_ids: Sequence[int] | None = None,
    exclude: set[str] | None = None,
) -> RecommendationResponse:
    scored = popular_products(
        session,
        top_k=top_k,
        before=before,
        interaction_ids=interaction_ids,
        exclude=exclude,
    )
    results = _attach_metadata(session, scored, source=SOURCE_POPULARITY)
    return RecommendationResponse(
        recommendation_mode=POPULARITY_MODE,
        top_k=top_k,
        results=results,
    )


def recommend_cf_for_user(
    session: Session,
    user_id: str,
    *,
    top_k: int,
) -> RecommendationResponse:
    """Learned BPR-MF scores. No content/popularity fallback."""

    import torch

    from app.recommendations.cf_runtime import require_cf_runtime

    if get_user(session, user_id) is None:
        raise UserNotFound(user_id)
    runtime = require_cf_runtime()
    user_index = runtime.user_to_index.get(user_id)
    if user_index is None:
        return RecommendationResponse(
            recommendation_mode=USER_CF_MODE,
            cf_model_version=runtime.model_version,
            user_id=user_id,
            top_k=top_k,
            reason=REASON_CF_USER_UNAVAILABLE,
            results=[],
        )
    history = unique_product_ids(list_user_interactions(session, user_id))
    seen = set(history)
    with torch.no_grad():
        index = torch.tensor([user_index], dtype=torch.long)
        scores = runtime.model.score_items(index).detach().cpu().numpy()[0]
    scored: list[ScoredItem] = []
    for product_id, score in zip(runtime.product_ids, scores, strict=True):
        if product_id in seen:
            continue
        scored.append(ScoredItem(product_id=str(product_id), score=float(score)))
    if not scored:
        return RecommendationResponse(
            recommendation_mode=USER_CF_MODE,
            cf_model_version=runtime.model_version,
            user_id=user_id,
            history_items=len(seen),
            top_k=top_k,
            reason=REASON_ALL_CF_ITEMS_SEEN,
            results=[],
        )
    ranked = stable_rank(scored, top_k=top_k)
    results = _attach_metadata(session, ranked, source=SOURCE_CF)
    return RecommendationResponse(
        recommendation_mode=USER_CF_MODE,
        cf_model_version=runtime.model_version,
        user_id=user_id,
        history_items=len(seen),
        top_k=top_k,
        results=results,
    )


def _attach_metadata(
    session: Session,
    scored: list[ScoredItem],
    *,
    source: str,
) -> list[RecommendationItem]:
    products = get_products_by_ids(session, [item.product_id for item in scored])
    items: list[RecommendationItem] = []
    for rank, scored_item in enumerate(scored, start=1):
        product = products.get(scored_item.product_id)
        if product is None:
            continue
        price = product.price if isinstance(product.price, Decimal) else product.price
        items.append(
            RecommendationItem(
                rank=rank,
                product_id=product.product_id,
                title=product.title,
                brand=product.brand,
                category=product.category,
                price=price,
                score=scored_item.score,
                source=source,
            )
        )
    return items


def validate_user_method(method: str) -> str:
    chosen = (method or DEFAULT_USER_METHOD).strip().lower()
    if chosen not in ALLOWED_USER_METHODS:
        raise ValueError("method must be 'content', 'cf', or 'hybrid'")
    return chosen


def require_content_runtime(session: Session) -> SemanticRuntime:
    try:
        return require_semantic_runtime(session)
    except SemanticUnavailableError:
        raise
