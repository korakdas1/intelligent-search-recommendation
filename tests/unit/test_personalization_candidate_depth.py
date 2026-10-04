"""Serving-depth parity through real collection/reranking paths, without models."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from app.core.config import clear_settings_cache, get_settings
from app.ranking import retrieval
from app.schemas.search import SearchRequest
from app.search import hybrid, ltr, personalization_service
from app.search.candidates import DEFAULT_CANDIDATE_K_MAX, RetrievalCandidate, resolve_candidate_k
from app.search.personalization_policy import load_personalization_policy
from app.search.service import run_search


@pytest.fixture
def serving(monkeypatch):
    """Keep resolvers, fusion, features and personalization real; fake only I/O/scoring."""

    monkeypatch.delenv("HYBRID_CANDIDATE_K", raising=False)
    monkeypatch.setenv("HYBRID_CANDIDATE_K_MAX", str(DEFAULT_CANDIDATE_K_MAX))
    clear_settings_cache()
    trace = SimpleNamespace(depths=[], pools=[], personalization_inputs=[], outcomes=[], preferred=None)
    products = {
        f"{source}{index:04d}": SimpleNamespace(
            product_id=f"{source}{index:04d}", title="Fixture cream", brand="Fixture",
            category="Beauty", description="Synthetic product", price=None,
            average_rating=None, rating_count=None,
        )
        for source in ("K", "S") for index in range(600)
    }
    semantic_runtime = SimpleNamespace(
        model_name="fake", artifact_version="fixture", backend="flat", embedding_dim=4,
    )

    def candidates(source, depth):
        trace.depths.append((source, depth))
        return [RetrievalCandidate(f"{source}{index:04d}", index + 1, 1.0 / (index + 1),
                                   "keyword" if source == "K" else "semantic")
                for index in range(depth)]

    for module in (hybrid, retrieval):
        monkeypatch.setattr(module, "keyword_candidates", lambda session, query, depth: candidates("K", depth))
        monkeypatch.setattr(module, "semantic_candidates", lambda session, *, query, top_k, runtime: candidates("S", top_k))
    for module in (hybrid, retrieval, ltr, personalization_service):
        monkeypatch.setattr(module, "require_semantic_runtime", lambda session: semantic_runtime)
    for module in (hybrid, ltr, personalization_service):
        monkeypatch.setattr(module, "get_products_by_ids", lambda session, ids: {pid: products[pid] for pid in ids})
    ranker = SimpleNamespace(model=None, scaler=None, model_version="fake", feature_version="rank-features-v1")
    monkeypatch.setattr(ltr, "require_ltr_runtime", lambda: ranker)
    monkeypatch.setattr(ltr, "score_feature_matrix", lambda model, scaler, values: np.linspace(1.0, 0.0, len(values)))
    monkeypatch.setattr(personalization_service, "get_user", lambda session, user_id: SimpleNamespace(user_id=user_id))
    monkeypatch.setattr(personalization_service, "list_user_interactions", lambda *args, **kwargs: [])
    monkeypatch.setattr(personalization_service, "cf_scores_for_search_candidates",
                        lambda user_id, ids: {pid: float(pid == trace.preferred) for pid in ids} if user_id == "warm" else None)

    collect_hybrid = hybrid.collect_hybrid_fused
    collect_ltr = ltr.collect_fused_candidates
    personalize = personalization_service.personalize_baseline

    def capture_hybrid(*args, **kwargs):
        collected = collect_hybrid(*args, **kwargs)
        trace.pools.append([row.product_id for row in collected.fused])
        return collected

    def capture_ltr(*args, **kwargs):
        rows, depth = collect_ltr(*args, **kwargs)
        trace.pools.append([row.product_id for row in rows])
        return rows, depth

    def capture_personalization(*args, **kwargs):
        trace.personalization_inputs.append([row.product_id for row in kwargs["baseline"]])
        outcome = personalize(*args, **kwargs)
        trace.outcomes.append(outcome)
        return outcome

    monkeypatch.setattr(hybrid, "collect_hybrid_fused", capture_hybrid)
    monkeypatch.setattr(personalization_service, "collect_hybrid_fused", capture_hybrid)
    monkeypatch.setattr(ltr, "collect_fused_candidates", capture_ltr)
    monkeypatch.setattr(personalization_service, "personalize_baseline", capture_personalization)
    trace.log = Mock()
    for module in (hybrid, ltr, personalization_service):
        monkeypatch.setattr(module, "log_search_event", trace.log)
    yield trace
    clear_settings_cache()


@pytest.mark.parametrize("rerank_mode", ["none", "ltr"])
@pytest.mark.parametrize("fusion_method", ["rrf", "weighted"])
@pytest.mark.parametrize("top_k,configured,maximum,override", [
    (10, None, DEFAULT_CANDIDATE_K_MAX, None),
    (20, None, DEFAULT_CANDIDATE_K_MAX, None),
    (21, None, DEFAULT_CANDIDATE_K_MAX, None),
    (50, None, DEFAULT_CANDIDATE_K_MAX, None),
    (100, None, DEFAULT_CANDIDATE_K_MAX, None),
    (50, 160, DEFAULT_CANDIDATE_K_MAX, None),
    (50, None, 180, None),
    (50, 300, 180, None),
    (50, 160, DEFAULT_CANDIDATE_K_MAX, 230),
    (50, 160, 180, 300),
    (50, None, DEFAULT_CANDIDATE_K_MAX, 10),
])
def test_shared_depth_full_pool_and_cold_order(
    serving, monkeypatch, rerank_mode, fusion_method, top_k, configured, maximum, override,
):
    if configured is not None:
        monkeypatch.setenv("HYBRID_CANDIDATE_K", str(configured))
    monkeypatch.setenv("HYBRID_CANDIDATE_K_MAX", str(maximum))
    clear_settings_cache()
    settings = get_settings()
    expected = resolve_candidate_k(
        top_k, configured=override if override is not None else settings.hybrid_candidate_k,
        maximum=settings.hybrid_candidate_k_max,
    )
    request = dict(query="cream", top_k=top_k, retrieval_mode="hybrid",
                   fusion_method=fusion_method, rerank_mode=rerank_mode, candidate_k=override)
    baseline = run_search(None, **request)
    personalized = run_search(None, **request, personalization_mode="bounded", user_id="cold")
    assert serving.depths == [("K", expected), ("S", expected)] * 2
    assert serving.pools[0] == serving.pools[1]
    assert len(serving.pools[0]) == 2 * expected > top_k
    assert serving.personalization_inputs == [serving.pools[0]]
    assert not personalized.personalization_applied
    assert personalized.personalization_reason == "no_personalized_history"
    # Personalization's existing score normalization is separate from result order.
    assert [row.model_dump(exclude={"score"}) for row in personalized.results] == [
        row.model_dump(exclude={"score"}) for row in baseline.results
    ]
    assert len(personalized.results) == top_k
    assert [call.kwargs["metadata"]["candidate_k"] for call in serving.log.call_args_list] == [expected, expected]


@pytest.mark.parametrize("rerank_mode", ["none", "ltr"])
@pytest.mark.parametrize("fusion_method", ["rrf", "weighted"])
def test_warm_user_can_promote_from_shared_full_pool(serving, rerank_mode, fusion_method):
    request = dict(query="cream", top_k=50, retrieval_mode="hybrid",
                   fusion_method=fusion_method, rerank_mode=rerank_mode)
    baseline = run_search(None, **request)
    serving.preferred = serving.pools[0][50]  # Just outside baseline top_k.
    personalized = run_search(None, **request, personalization_mode="bounded", user_id="warm")
    assert personalized.personalization_applied
    assert serving.personalization_inputs == [serving.pools[0]]
    assert serving.pools[0] == serving.pools[1]
    assert len(serving.pools[0]) == 2 * resolve_candidate_k(50)
    assert {row.product_id for row in serving.outcomes[0].candidates} == set(serving.pools[0])
    baseline_ids = {row.product_id for row in baseline.results}
    personalized_ids = {row.product_id for row in personalized.results}
    assert serving.preferred not in baseline_ids
    assert serving.preferred in personalized_ids
    assert personalized_ids <= set(serving.pools[0])


def test_frozen_tuning_depth_remains_100(monkeypatch):
    from scripts import tune_personalized_search

    monkeypatch.setenv("HYBRID_CANDIDATE_K", "250")
    monkeypatch.setattr(sys, "argv", ["tune_personalized_search.py"])
    assert tune_personalized_search.parse_args().candidate_k == 100
    assert load_personalization_policy().candidate_k == 100
    monkeypatch.setattr(sys, "argv", ["tune_personalized_search.py", "--candidate-k", "150"])
    assert tune_personalized_search.parse_args().candidate_k == 150


def test_candidate_depth_remains_internal():
    assert "candidate_k" not in SearchRequest.model_fields
