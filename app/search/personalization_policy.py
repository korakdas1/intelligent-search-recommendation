"""Load and freeze personalized-search-v1. Configuration, not a trained model."""

from __future__ import annotations

import json
from pathlib import Path

from app.search.personalization_constants import (
    BASE_FUSION_METHOD,
    BASE_RERANK_MODE,
    BASE_RETRIEVAL_MODE,
    COLD_USER_BEHAVIOR,
    DEFAULT_CANDIDATE_K,
    DEFAULT_CF_WEIGHT,
    DEFAULT_CONTENT_WEIGHT,
    DEFAULT_GAMMA,
    EQUAL_SCORE_QUERY_FALLBACK,
    EVAL_PROTOCOL,
    FEATURE_VERSION,
    GAMMA_MAX,
    GATING_FORMULA,
    IMPLEMENTED_SIGNALS,
    NORMALIZATION,
    PERSONALIZATION_VERSION,
    QUERY_GENERATOR_VERSION,
    TIE_BREAK,
    TUNE_PROTOCOL,
)
from app.search.personalization_rerank import validate_gamma, validate_signal_weights
from app.search.personalization_types import PersonalizationPolicy

POLICY_PATH = Path(__file__).resolve().parent / "policies" / "personalized-search-v1.json"


def default_personalization_policy() -> PersonalizationPolicy:
    return PersonalizationPolicy(
        personalization_version=PERSONALIZATION_VERSION,
        feature_version=FEATURE_VERSION,
        base_retrieval_mode=BASE_RETRIEVAL_MODE,
        base_fusion_method=BASE_FUSION_METHOD,
        base_rerank_mode=BASE_RERANK_MODE,
        candidate_k=DEFAULT_CANDIDATE_K,
        signals=IMPLEMENTED_SIGNALS,
        weights=validate_signal_weights(DEFAULT_CONTENT_WEIGHT, DEFAULT_CF_WEIGHT),
        gamma=validate_gamma(DEFAULT_GAMMA),
        gamma_max=GAMMA_MAX,
        normalization=NORMALIZATION,
        equal_score_query_fallback=EQUAL_SCORE_QUERY_FALLBACK,
        gating_formula=GATING_FORMULA,
        tie_break=TIE_BREAK,
        cold_user=COLD_USER_BEHAVIOR,
        query_generator_version=QUERY_GENERATOR_VERSION,
        tune_protocol=TUNE_PROTOCOL,
        evaluation_version=EVAL_PROTOCOL,
        note="Provisional until inner-validation freeze writes personalized-search-v1.json.",
    )


def policy_from_dict(payload: dict[str, object]) -> PersonalizationPolicy:
    weights_raw = payload.get("weights") or {}
    if not isinstance(weights_raw, dict):
        raise ValueError("personalization policy weights must be an object")
    signals_raw = payload.get("signals") or list(IMPLEMENTED_SIGNALS)
    if not isinstance(signals_raw, list):
        raise ValueError("personalization policy signals must be a list")
    tie_raw = payload.get("tie_break") or list(TIE_BREAK)
    if not isinstance(tie_raw, list):
        raise ValueError("tie_break must be a list")
    version = str(payload.get("personalization_version", PERSONALIZATION_VERSION))
    gamma = validate_gamma(float(payload.get("gamma", DEFAULT_GAMMA)), version=version)
    return PersonalizationPolicy(
        personalization_version=version,
        feature_version=str(payload.get("feature_version", FEATURE_VERSION)),
        base_retrieval_mode=str(payload.get("base_retrieval_mode", BASE_RETRIEVAL_MODE)),
        base_fusion_method=str(payload.get("base_fusion_method", BASE_FUSION_METHOD)),
        base_rerank_mode=str(payload.get("base_rerank_mode", BASE_RERANK_MODE)),
        candidate_k=int(payload.get("candidate_k", DEFAULT_CANDIDATE_K)),
        signals=tuple(str(item) for item in signals_raw),
        weights=validate_signal_weights(
            float(weights_raw.get("content", DEFAULT_CONTENT_WEIGHT)),
            float(weights_raw.get("cf", DEFAULT_CF_WEIGHT)),
        ),
        gamma=gamma,
        gamma_max=float(payload.get("gamma_max", GAMMA_MAX)),
        normalization=str(payload.get("normalization", NORMALIZATION)),
        equal_score_query_fallback=str(
            payload.get("equal_score_query_fallback", EQUAL_SCORE_QUERY_FALLBACK)
        ),
        gating_formula=str(payload.get("gating_formula", GATING_FORMULA)),
        tie_break=tuple(str(item) for item in tie_raw),
        cold_user=str(payload.get("cold_user", COLD_USER_BEHAVIOR)),
        query_generator_version=str(payload.get("query_generator_version", QUERY_GENERATOR_VERSION)),
        tune_protocol=str(payload.get("tune_protocol", TUNE_PROTOCOL)),
        evaluation_version=str(payload.get("evaluation_version", EVAL_PROTOCOL)),
        created_at=str(payload["created_at"]) if payload.get("created_at") else None,
        git_commit=str(payload["git_commit"]) if payload.get("git_commit") else None,
        note=str(payload["note"]) if payload.get("note") else None,
    )


def policy_to_dict(policy: PersonalizationPolicy) -> dict[str, object]:
    return {
        "personalization_version": policy.personalization_version,
        "feature_version": policy.feature_version,
        "base_retrieval_mode": policy.base_retrieval_mode,
        "base_fusion_method": policy.base_fusion_method,
        "base_rerank_mode": policy.base_rerank_mode,
        "candidate_k": policy.candidate_k,
        "signals": list(policy.signals),
        "weights": policy.weights.as_dict(),
        "gamma": policy.gamma,
        "gamma_max": policy.gamma_max,
        "normalization": policy.normalization,
        "equal_score_query_fallback": policy.equal_score_query_fallback,
        "gating_formula": policy.gating_formula,
        "tie_break": list(policy.tie_break),
        "cold_user": policy.cold_user,
        "query_generator_version": policy.query_generator_version,
        "tune_protocol": policy.tune_protocol,
        "evaluation_version": policy.evaluation_version,
        "created_at": policy.created_at,
        "git_commit": policy.git_commit,
        "note": policy.note,
        "candidate_invariant": "same_ids_permutation_only",
        "seen_item_handling": "search_may_return_previously_interacted_items",
        "unavailable_channel": "drop_and_renormalize_or_uniform_if_remaining_weights_zero",
        "missing_candidate_signal": "contribute_zero",
        "score_semantics": "bounded_rerank_not_probability",
        "recommendation_candidates": "never_merged",
    }


def load_personalization_policy(path: Path | None = None) -> PersonalizationPolicy:
    target = Path(path) if path is not None else POLICY_PATH
    if not target.is_file():
        return default_personalization_policy()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("personalization policy must be a JSON object")
    return policy_from_dict(payload)


def write_personalization_policy(policy: PersonalizationPolicy, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else POLICY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(policy_to_dict(policy), indent=2) + "\n", encoding="utf-8")
    return target
