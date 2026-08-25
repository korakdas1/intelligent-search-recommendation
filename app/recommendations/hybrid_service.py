"""hybrid-rec-v1 serving. Owns recommendation fallbacks. Does not touch /search."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.repositories.products import get_products_by_ids
from app.db.repositories.users import get_user
from app.recommendations.exceptions import CfUnavailableError, UserNotFound
from app.recommendations.history import list_user_interactions, unique_product_ids
from app.recommendations.hybrid_candidates import (
    cf_candidates_for_user,
    channel_rankings,
    content_candidates_for_profile,
    popularity_candidates,
)
from app.recommendations.hybrid_constants import (
    CHANNEL_CF,
    CHANNEL_CONTENT,
    SOURCE_HYBRID,
    USER_HYBRID_MODE,
)
from app.recommendations.hybrid_fusion import (
    fuse_recommendation_rrf,
    fuse_recommendation_weighted,
    resolve_hybrid_candidate_k,
    union_recommendation_channels,
)
from app.recommendations.hybrid_policy import load_hybrid_policy, resolve_channel_availability
from app.recommendations.hybrid_types import HybridPolicy
from app.schemas.recommendations import RecommendationItem, RecommendationResponse
from app.search.exceptions import SemanticUnavailableError
from app.search.runtime import require_semantic_runtime


def recommend_hybrid_for_user(
    session: Session,
    user_id: str,
    *,
    top_k: int,
    policy: HybridPolicy | None = None,
    before: datetime | None = None,
    seen_product_ids: Sequence[str] | None = None,
    popularity_interaction_ids: Sequence[int] | None = None,
    popularity_counts: dict[str, int] | None = None,
) -> RecommendationResponse:
    """Fuse content + CF + popularity with explicit channel fallback.

    Unknown PostgreSQL users remain 404. Missing CF/semantic artifacts degrade
    here; they do not change method=cf or method=content.
    """

    if get_user(session, user_id) is None:
        raise UserNotFound(user_id)
    policy = policy or load_hybrid_policy()
    candidate_k = resolve_hybrid_candidate_k(
        top_k,
        minimum=policy.candidate_k_min,
        maximum=policy.candidate_k_max,
        multiplier=policy.candidate_k_multiplier,
    )
    if seen_product_ids is None:
        history = list_user_interactions(session, user_id, before=before)
        seen = unique_product_ids(history)
    else:
        seen = list(dict.fromkeys(str(item) for item in seen_product_ids))

    semantic_ok = True
    content_usable = False
    content_items = []
    try:
        runtime = require_semantic_runtime(session)
        content_items, usable = content_candidates_for_profile(runtime, seen, top_k=candidate_k)
        content_usable = bool(usable)
    except SemanticUnavailableError:
        semantic_ok = False

    cf_artifact_ok = True
    cf_user_ok = False
    cf_items = None
    cf_version: str | None = None
    try:
        from app.recommendations.cf_runtime import require_cf_runtime

        cf_runtime = require_cf_runtime()
        cf_version = cf_runtime.model_version
        cf_items = cf_candidates_for_user(
            cf_runtime,
            user_id,
            seen_product_ids=seen,
            top_k=candidate_k,
        )
        cf_user_ok = cf_items is not None
    except CfUnavailableError:
        cf_artifact_ok = False

    pop_items = popularity_candidates(
        session,
        top_k=candidate_k,
        seen_product_ids=seen,
        before=before,
        interaction_ids=popularity_interaction_ids,
        counts=popularity_counts,
    )

    availability = resolve_channel_availability(
        content_usable_history=content_usable,
        semantic_artifact_available=semantic_ok,
        cf_artifact_available=cf_artifact_ok,
        cf_user_in_model=cf_user_ok,
        popularity_available=True,
    )
    content_rank, cf_rank, pop_rank = channel_rankings(
        content_items=content_items,
        content_available=availability.content,
        cf_items=cf_items or [],
        cf_available=availability.cf,
        popularity_items=pop_items,
        popularity_available=availability.popularity,
    )
    union = union_recommendation_channels(content_rank, cf_rank, pop_rank)
    if not union:
        return _hybrid_response(
            policy,
            user_id=user_id,
            top_k=top_k,
            history_items=len(seen),
            availability=availability,
            results=[],
            cf_version=cf_version if availability.cf else None,
        )

    if policy.fusion_method == "weighted":
        ranked, _effective = fuse_recommendation_weighted(
            union,
            policy.weights,
            available_channels=availability.channels_used,
        )
    else:
        ranked = fuse_recommendation_rrf(union, k0=policy.rrf_k)

    selected = ranked[:top_k]
    results = _attach_hybrid_metadata(session, selected)
    return _hybrid_response(
        policy,
        user_id=user_id,
        top_k=top_k,
        history_items=len(seen),
        availability=availability,
        results=results,
        cf_version=cf_version if availability.cf else None,
    )


def _hybrid_response(
    policy: HybridPolicy,
    *,
    user_id: str,
    top_k: int,
    history_items: int,
    availability,
    results: list[RecommendationItem],
    cf_version: str | None,
) -> RecommendationResponse:
    return RecommendationResponse(
        recommendation_mode=USER_HYBRID_MODE,
        content_rec_version=policy.content_rec_version if CHANNEL_CONTENT in availability.channels_used else None,
        cf_model_version=cf_version or (policy.cf_model_version if CHANNEL_CF in availability.channels_used else None),
        hybrid_version=policy.hybrid_version,
        fusion_method=policy.fusion_method,
        channels_used=list(availability.channels_used),
        fallback_reason=availability.fallback_reason,
        user_id=user_id,
        history_items=history_items,
        top_k=top_k,
        results=results,
    )


def _attach_hybrid_metadata(session: Session, ranked: Sequence) -> list[RecommendationItem]:
    products = get_products_by_ids(session, [row.product_id for row in ranked])
    items: list[RecommendationItem] = []
    rank = 0
    for row in ranked:
        product = products.get(row.product_id)
        if product is None:
            continue
        rank += 1
        price = product.price if isinstance(product.price, Decimal) else product.price
        items.append(
            RecommendationItem(
                rank=rank,
                product_id=product.product_id,
                title=product.title,
                brand=product.brand,
                category=product.category,
                price=price,
                score=float(row.fused_score),
                source=SOURCE_HYBRID,
            )
        )
    return items
