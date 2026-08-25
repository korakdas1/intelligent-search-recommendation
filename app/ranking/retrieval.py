"""Collect hybrid fused candidates for feature extraction. Does not log or change serving."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.repositories.search import keyword_candidates
from app.search.candidates import FusedCandidate, resolve_candidate_k, union_candidates
from app.search.fusion import fuse_rrf
from app.search.runtime import require_semantic_runtime
from app.search.semantic import semantic_candidates


def collect_fused_candidates(
    session: Session,
    query: str,
    *,
    candidate_k: int | None = None,
    top_k: int = 10,
    rrf_k: int | None = None,
) -> tuple[list[FusedCandidate], int]:
    """Return RRF-ordered union candidates plus the resolved candidate_k.

    Weighted fusion scores are computed later inside the feature extractor so
    both hybrid fusion methods appear as features without changing /search.
    """

    settings = get_settings()
    depth = resolve_candidate_k(
        top_k,
        configured=candidate_k if candidate_k is not None else settings.hybrid_candidate_k,
        maximum=settings.hybrid_candidate_k_max,
    )
    runtime = require_semantic_runtime(session)
    keyword = keyword_candidates(session, query, depth)
    semantic = semantic_candidates(session, query=query, top_k=depth, runtime=runtime)
    merged = union_candidates(keyword, semantic)
    k0 = settings.hybrid_rrf_k if rrf_k is None else rrf_k
    fused = fuse_rrf(merged, k0=k0)
    return fused, depth
