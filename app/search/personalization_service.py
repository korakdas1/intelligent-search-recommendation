"""Serving: bounded personalization of hybrid/LTR search candidates.

Does not inject recommendation products. Does not train a model.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal

from sqlalchemy.orm import Session

from app.db.repositories.products import get_products_by_ids
from app.db.repositories.search import log_search_event
from app.db.repositories.users import get_user
from app.recommendations.exceptions import CfUnavailableError
from app.recommendations.history import list_user_interactions, unique_product_ids
from app.schemas.search import SearchResponse, SearchResultItem
from app.search.exceptions import SearchUserNotFound
from app.search.hybrid import HYBRID_MODE, HYBRID_SOURCE, collect_hybrid_fused
from app.search.personalization_constants import (
    PERSONALIZATION_BOUNDED,
    REASON_NO_PERSONALIZED_HISTORY,
)
from app.search.personalization_features import (
    cf_affinity_for_candidates,
    content_affinity_for_candidates,
)
from app.search.personalization_policy import load_personalization_policy
from app.search.personalization_rerank import rerank_candidates
from app.search.personalization_types import (
    BaselineCandidate,
    PersonalizationOutcome,
    PersonalizationPolicy,
)
from app.search.runtime import require_semantic_runtime
from app.search.types import SearchHit


def _cf_dot_scores(runtime, user_index: int, item_indices: list[int]) -> list[float]:
    import torch

    if not item_indices:
        return []
    with torch.no_grad():
        users = torch.tensor([user_index] * len(item_indices), dtype=torch.long)
        items = torch.tensor(item_indices, dtype=torch.long)
        return [float(value) for value in runtime.model.score(users, items).tolist()]


def cf_scores_for_search_candidates(
    user_id: str,
    candidate_ids: Sequence[str],
) -> dict[str, float] | None:
    """Return raw CF dots if the user is in the model. Artifact missing → None."""

    try:
        from app.recommendations.cf_runtime import require_cf_runtime

        runtime = require_cf_runtime()
    except CfUnavailableError:
        return None
    available, scores = cf_affinity_for_candidates(
        user_id=user_id,
        candidate_ids=candidate_ids,
        user_to_index=runtime.user_to_index,
        product_to_index=runtime.product_to_index,
        score_fn=lambda user_index, item_indices: _cf_dot_scores(runtime, user_index, item_indices),
    )
    if not available:
        return None
    return scores


def baseline_from_hybrid(
    session: Session,
    *,
    query: str,
    top_k: int,
    fusion_method: str,
    candidate_k: int | None = None,
    rerank_mode: str = "none",
) -> tuple[list[BaselineCandidate], object]:
    if rerank_mode == "ltr":
        from app.search.ltr import score_ltr_union

        rows, depth = score_ltr_union(
            session,
            query=query,
            top_k=top_k,
            fusion_method=fusion_method,
            candidate_k=candidate_k,
        )
        return rows, depth
    collected = collect_hybrid_fused(
        session,
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,
        candidate_k=candidate_k,
    )
    rows = [
        BaselineCandidate(
            product_id=row.product_id,
            baseline_score=float(row.fusion_score),
            baseline_rank=index,
            source=HYBRID_SOURCE,
        )
        for index, row in enumerate(collected.fused, start=1)
    ]
    return rows, collected


def personalize_baseline(
    session: Session,
    *,
    user_id: str,
    baseline: Sequence[BaselineCandidate],
    history_product_ids: Sequence[str] | None = None,
    before: datetime | None = None,
    policy: PersonalizationPolicy | None = None,
) -> PersonalizationOutcome:
    chosen = policy or load_personalization_policy()
    if history_product_ids is None:
        history_product_ids = unique_product_ids(
            list_user_interactions(session, user_id, before=before)
        )
    candidate_ids = [row.product_id for row in baseline]
    semantic = require_semantic_runtime(session)
    content_ok, content_scores, _kept = content_affinity_for_candidates(
        semantic,
        history_product_ids,
        candidate_ids,
    )
    content_raw = content_scores if content_ok else None
    cf_raw = cf_scores_for_search_candidates(user_id, candidate_ids)
    return rerank_candidates(
        baseline,
        content_raw=content_raw,
        cf_raw=cf_raw,
        weights=chosen.weights,
        gamma=chosen.gamma,
        version=chosen.personalization_version,
    )


def _hits_for_candidates(
    session: Session,
    candidates: Sequence,
    *,
    top_k: int,
) -> list[SearchHit]:
    chosen = list(candidates[:top_k])
    products = get_products_by_ids(session, [row.product_id for row in chosen])
    hits: list[SearchHit] = []
    for row in chosen:
        product = products.get(row.product_id)
        if product is None:
            continue
        hits.append(
            SearchHit(
                product_id=product.product_id,
                title=product.title,
                brand=product.brand,
                category=product.category,
                price=product.price if isinstance(product.price, Decimal) else product.price,
                score=float(row.final_score),
                source=row.source,
            )
        )
    return hits


def personalized_search(
    session: Session,
    *,
    query: str,
    top_k: int,
    user_id: str,
    fusion_method: str,
    rerank_mode: str = "none",
    log: bool = True,
    candidate_k: int | None = None,
    history_product_ids: Sequence[str] | None = None,
    before: datetime | None = None,
    policy: PersonalizationPolicy | None = None,
) -> SearchResponse:
    """Rerank hybrid/LTR candidates for a known user. Unknown users raise 404."""

    if get_user(session, user_id) is None:
        raise SearchUserNotFound(user_id)
    chosen_policy = policy or load_personalization_policy()
    depth = candidate_k if candidate_k is not None else chosen_policy.candidate_k
    baseline, collected = baseline_from_hybrid(
        session,
        query=query,
        top_k=top_k,
        fusion_method=fusion_method,
        candidate_k=depth,
        rerank_mode=rerank_mode,
    )
    outcome = personalize_baseline(
        session,
        user_id=user_id,
        baseline=baseline,
        history_product_ids=history_product_ids,
        before=before,
        policy=chosen_policy,
    )
    hits = _hits_for_candidates(session, outcome.candidates, top_k=top_k)
    source = hits[0].source if hits else HYBRID_SOURCE
    if log:
        metadata = {
            "retrieval_mode": HYBRID_MODE,
            "fusion_method": fusion_method,
            "rerank_mode": rerank_mode,
            "candidate_k": getattr(collected, "depth", depth) if not isinstance(collected, int) else collected,
            "personalization_mode": PERSONALIZATION_BOUNDED,
            "personalization_version": chosen_policy.personalization_version,
            "personalization_applied": outcome.applied,
            "personalization_reason": outcome.reason,
            "personalization_signals": list(outcome.signals_used),
            "gamma": chosen_policy.gamma,
        }
        if not isinstance(collected, int) and collected is not None:
            metadata["keyword_candidate_count"] = collected.keyword_count
            metadata["semantic_candidate_count"] = collected.semantic_count
            metadata["union_candidate_count"] = collected.union_count
            metadata["embedding_model"] = collected.runtime.model_name
            metadata["semantic_artifact_version"] = collected.runtime.artifact_version
            metadata["index_backend"] = collected.runtime.backend
        log_search_event(
            session,
            query_text=query,
            top_k=top_k,
            hits=hits,
            metadata={key: value for key, value in metadata.items() if value is not None},
            source=source,
            user_id=user_id,
        )
    results = [
        SearchResultItem(
            rank=index,
            product_id=hit.product_id,
            title=hit.title,
            brand=hit.brand,
            category=hit.category,
            price=hit.price,
            score=hit.score,
            source=hit.source,
        )
        for index, hit in enumerate(hits, start=1)
    ]
    return SearchResponse(
        query=query,
        retrieval_mode=HYBRID_MODE,
        fusion_method=fusion_method,
        rerank_mode=rerank_mode,
        personalization_mode=PERSONALIZATION_BOUNDED,
        personalization_version=chosen_policy.personalization_version,
        personalization_applied=outcome.applied,
        personalization_reason=outcome.reason if not outcome.applied else None,
        personalization_signals=list(outcome.signals_used),
        user_id=user_id,
        top_k=top_k,
        results=results,
    )


def cold_reason() -> str:
    return REASON_NO_PERSONALIZED_HISTORY
