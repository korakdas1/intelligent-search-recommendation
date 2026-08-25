"""Load hybrid-rec-v1 and resolve explicit cold-start channel availability."""

from __future__ import annotations

import json
from pathlib import Path

from app.recommendations.hybrid_constants import (
    CANDIDATE_K_MAX,
    CANDIDATE_K_MIN,
    CANDIDATE_K_MULTIPLIER,
    CF_DEP_VERSION,
    CHANNEL_CF,
    CHANNEL_CONTENT,
    CHANNEL_POPULARITY,
    CONTENT_DEP_VERSION,
    DEFAULT_RRF_K,
    EVAL_DEP_VERSION,
    FALLBACK_POLICY_VERSION,
    HYBRID_REC_VERSION,
    POPULARITY_RULE_SERVING,
    REASON_CF_ARTIFACT_UNAVAILABLE,
    REASON_CONTENT_ARTIFACT_UNAVAILABLE,
    REASON_NO_PERSONALIZED_HISTORY,
    REASON_NO_RECOMMENDATION_CHANNEL,
    TUNE_PROTOCOL,
)
from app.recommendations.hybrid_fusion import validate_channel_weights
from app.recommendations.hybrid_types import ChannelAvailability, ChannelWeights, HybridPolicy

POLICY_PATH = Path(__file__).resolve().parent / "policies" / "hybrid-rec-v1.json"


def default_hybrid_policy() -> HybridPolicy:
    """In-code fallback used by tests if the frozen JSON is temporarily unset."""

    return HybridPolicy(
        hybrid_version=HYBRID_REC_VERSION,
        fusion_method="rrf",
        rrf_k=DEFAULT_RRF_K,
        weights=validate_channel_weights(0.6, 0.3, 0.1),
        candidate_k_min=CANDIDATE_K_MIN,
        candidate_k_max=CANDIDATE_K_MAX,
        candidate_k_multiplier=CANDIDATE_K_MULTIPLIER,
        fallback_policy_version=FALLBACK_POLICY_VERSION,
        content_rec_version=CONTENT_DEP_VERSION,
        cf_model_version=CF_DEP_VERSION,
        popularity_rule=POPULARITY_RULE_SERVING,
        evaluation_version=EVAL_DEP_VERSION,
        tune_protocol=TUNE_PROTOCOL,
        note="Provisional until inner-validation freeze writes hybrid-rec-v1.json.",
    )


def load_hybrid_policy(path: Path | None = None) -> HybridPolicy:
    target = Path(path) if path is not None else POLICY_PATH
    if not target.is_file():
        return default_hybrid_policy()
    payload = json.loads(target.read_text(encoding="utf-8"))
    return policy_from_dict(payload)


def policy_from_dict(payload: dict[str, object]) -> HybridPolicy:
    weights_raw = payload.get("weights") or {}
    if not isinstance(weights_raw, dict):
        raise ValueError("hybrid policy weights must be an object")
    weights = validate_channel_weights(
        float(weights_raw.get("content", 0.0)),
        float(weights_raw.get("cf", 0.0)),
        float(weights_raw.get("popularity", 0.0)),
    )
    method = str(payload.get("fusion_method", "rrf")).strip().lower()
    if method not in {"rrf", "weighted"}:
        raise ValueError("fusion_method must be 'rrf' or 'weighted'")
    return HybridPolicy(
        hybrid_version=str(payload.get("hybrid_version", HYBRID_REC_VERSION)),
        fusion_method=method,  # type: ignore[arg-type]
        rrf_k=int(payload.get("rrf_k", DEFAULT_RRF_K)),
        weights=weights,
        candidate_k_min=int(payload.get("candidate_k_min", CANDIDATE_K_MIN)),
        candidate_k_max=int(payload.get("candidate_k_max", CANDIDATE_K_MAX)),
        candidate_k_multiplier=int(payload.get("candidate_k_multiplier", CANDIDATE_K_MULTIPLIER)),
        fallback_policy_version=str(payload.get("fallback_policy_version", FALLBACK_POLICY_VERSION)),
        content_rec_version=str(payload.get("content_rec_version", CONTENT_DEP_VERSION)),
        cf_model_version=str(payload.get("cf_model_version", CF_DEP_VERSION)),
        popularity_rule=str(payload.get("popularity_rule", POPULARITY_RULE_SERVING)),
        evaluation_version=str(payload.get("evaluation_version", EVAL_DEP_VERSION)),
        tune_protocol=str(payload.get("tune_protocol", TUNE_PROTOCOL)),
        created_at=str(payload["created_at"]) if payload.get("created_at") else None,
        git_commit=str(payload["git_commit"]) if payload.get("git_commit") else None,
        note=str(payload["note"]) if payload.get("note") else None,
    )


def policy_to_dict(policy: HybridPolicy) -> dict[str, object]:
    return {
        "hybrid_version": policy.hybrid_version,
        "fusion_method": policy.fusion_method,
        "rrf_k": policy.rrf_k,
        "weights": policy.weights.as_dict(),
        "candidate_k_min": policy.candidate_k_min,
        "candidate_k_max": policy.candidate_k_max,
        "candidate_k_multiplier": policy.candidate_k_multiplier,
        "fallback_policy_version": policy.fallback_policy_version,
        "content_rec_version": policy.content_rec_version,
        "cf_model_version": policy.cf_model_version,
        "popularity_rule": policy.popularity_rule,
        "evaluation_version": policy.evaluation_version,
        "tune_protocol": policy.tune_protocol,
        "created_at": policy.created_at,
        "git_commit": policy.git_commit,
        "note": policy.note,
        "seen_item_handling": "exclude_current_request_history",
        "unavailable_channel": "drop_and_renormalize_weighted_or_omit_rrf_term",
        "missing_candidate": "contribute_zero",
        "candidate_union": "product_id_union_not_intersection",
        "score_semantics": "rank_fusion_or_minmax_weighted_not_probability",
    }


def write_hybrid_policy(policy: HybridPolicy, path: Path | None = None) -> Path:
    target = Path(path) if path is not None else POLICY_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(policy_to_dict(policy), indent=2) + "\n", encoding="utf-8")
    return target


def resolve_channel_availability(
    *,
    content_usable_history: bool,
    semantic_artifact_available: bool,
    cf_artifact_available: bool,
    cf_user_in_model: bool,
    popularity_available: bool = True,
) -> ChannelAvailability:
    """Explicit hybrid fallback. Direct method=content and method=cf are unchanged."""

    content = bool(semantic_artifact_available and content_usable_history)
    cf = bool(cf_artifact_available and cf_user_in_model)
    popularity = bool(popularity_available)
    used: list[str] = []
    if content:
        used.append(CHANNEL_CONTENT)
    if cf:
        used.append(CHANNEL_CF)
    if popularity:
        used.append(CHANNEL_POPULARITY)

    fallback: str | None = None
    if not used:
        fallback = REASON_NO_RECOMMENDATION_CHANNEL
    elif not content and not cf and popularity:
        if not semantic_artifact_available and not cf_artifact_available:
            fallback = REASON_CONTENT_ARTIFACT_UNAVAILABLE
        elif not cf_artifact_available and content_usable_history:
            fallback = REASON_CF_ARTIFACT_UNAVAILABLE
        elif not semantic_artifact_available and cf_user_in_model:
            fallback = REASON_CONTENT_ARTIFACT_UNAVAILABLE
        else:
            fallback = REASON_NO_PERSONALIZED_HISTORY
    elif not semantic_artifact_available and (cf or popularity):
        fallback = REASON_CONTENT_ARTIFACT_UNAVAILABLE
    elif not cf_artifact_available and (content or popularity):
        fallback = REASON_CF_ARTIFACT_UNAVAILABLE

    return ChannelAvailability(
        content=content,
        cf=cf,
        popularity=popularity,
        channels_used=tuple(used),
        fallback_reason=fallback,
        content_usable_history=bool(content_usable_history),
        cf_user_in_model=bool(cf_user_in_model),
        semantic_artifact_available=bool(semantic_artifact_available),
        cf_artifact_available=bool(cf_artifact_available),
    )


def available_channel_names(availability: ChannelAvailability) -> tuple[str, ...]:
    return availability.channels_used
