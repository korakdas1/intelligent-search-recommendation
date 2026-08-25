"""Bounded multiplicative personalization of query-constrained candidates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.search.personalization_constants import (
    GAMMA_MAX,
    PERSONALIZATION_VERSION,
    REASON_NO_PERSONALIZED_HISTORY,
    SIGNAL_CF,
    SIGNAL_CONTENT,
    WEIGHT_SUM_TOLERANCE,
)
from app.search.personalization_features import (
    channel_name,
    finite_or_zero,
    minmax_channel,
    query_relevance_scores,
)
from app.search.personalization_types import (
    BaselineCandidate,
    PersonalizationOutcome,
    PersonalizedCandidate,
    SignalWeights,
)


def validate_gamma(
    gamma: float,
    *,
    cap: float = GAMMA_MAX,
    version: str = PERSONALIZATION_VERSION,
) -> float:
    value = float(gamma)
    if value < 0.0:
        raise ValueError("gamma must be >= 0")
    if version == PERSONALIZATION_VERSION and value > cap:
        raise ValueError(f"{PERSONALIZATION_VERSION} gamma must be <= {cap}")
    if not np_isfinite(value):
        raise ValueError("gamma must be finite")
    return value


def np_isfinite(value: float) -> bool:
    return value == value and value not in {float("inf"), float("-inf")}


def validate_signal_weights(
    content: float,
    cf: float,
    *,
    tolerance: float = WEIGHT_SUM_TOLERANCE,
) -> SignalWeights:
    if content < 0.0 or cf < 0.0:
        raise ValueError("signal weights must be >= 0")
    if not np_isfinite(content) or not np_isfinite(cf):
        raise ValueError("signal weights must be finite")
    total = float(content) + float(cf)
    if abs(total - 1.0) > tolerance:
        raise ValueError("signal weights must sum to 1")
    return SignalWeights(content=float(content), cf=float(cf))


def renormalize_signal_weights(
    weights: SignalWeights | Mapping[str, float],
    available: Sequence[str],
    *,
    tolerance: float = WEIGHT_SUM_TOLERANCE,
) -> dict[str, float]:
    """Renormalize over whole available channels. Not per candidate.

    If remaining configured weights are all zero (for example CF-only policy
    when the user is outside the CF model), share mass uniformly over the
    still-available signals so content can still personalize.
    """

    configured = weights.as_dict() if isinstance(weights, SignalWeights) else dict(weights)
    allowed = [name for name in (SIGNAL_CONTENT, SIGNAL_CF) if name in {str(item) for item in available}]
    subset = {name: float(configured.get(name, 0.0)) for name in allowed}
    total = sum(subset.values())
    if not allowed:
        return {}
    if total <= tolerance:
        share = 1.0 / len(allowed)
        return {name: share for name in allowed}
    return {name: value / total for name, value in subset.items()}


def preference_scores(
    *,
    content_norm: Sequence[float] | None,
    cf_norm: Sequence[float] | None,
    weights: SignalWeights | Mapping[str, float],
    n: int,
) -> tuple[list[float], dict[str, float], tuple[str, ...]]:
    available: list[str] = []
    if content_norm is not None:
        available.append(SIGNAL_CONTENT)
    if cf_norm is not None:
        available.append(SIGNAL_CF)
    effective = renormalize_signal_weights(weights, available)
    if not effective:
        return [0.0] * n, {}, ()
    values: list[float] = []
    for index in range(n):
        score = 0.0
        if SIGNAL_CONTENT in effective and content_norm is not None:
            score += effective[SIGNAL_CONTENT] * float(content_norm[index])
        if SIGNAL_CF in effective and cf_norm is not None:
            score += effective[SIGNAL_CF] * float(cf_norm[index])
        values.append(finite_or_zero(score))
    return values, effective, tuple(effective.keys())


def gated_final_score(query_relevance: float, preference: float, gamma: float) -> float:
    q = finite_or_zero(query_relevance)
    p = min(1.0, max(0.0, finite_or_zero(preference)))
    return finite_or_zero(q * (1.0 + float(gamma) * p))


def assert_candidate_identity(baseline_ids: Sequence[str], personalized_ids: Sequence[str]) -> None:
    if len(baseline_ids) != len(personalized_ids):
        raise AssertionError("personalization changed candidate count")
    if set(baseline_ids) != set(personalized_ids):
        raise AssertionError("personalization changed candidate identity")


def rerank_candidates(
    baseline: Sequence[BaselineCandidate],
    *,
    content_raw: Mapping[str, float] | None,
    cf_raw: Mapping[str, float] | None,
    weights: SignalWeights,
    gamma: float,
    version: str = PERSONALIZATION_VERSION,
) -> PersonalizationOutcome:
    """Permute ``baseline`` only. Never inserts an ID that was not retrieved."""

    gamma = validate_gamma(gamma, version=version)
    ids = [row.product_id for row in baseline]
    if not ids:
        return PersonalizationOutcome(
            candidates=(),
            applied=False,
            reason=REASON_NO_PERSONALIZED_HISTORY,
            signals_used=(),
        )

    content_available = content_raw is not None
    cf_available = cf_raw is not None
    content_norm, content_flags = minmax_channel(ids, content_raw or {}, channel_available=content_available)
    cf_norm, cf_flags = minmax_channel(ids, cf_raw or {}, channel_available=cf_available)
    query_norm, degenerate = query_relevance_scores(
        [row.baseline_score for row in baseline],
        [row.baseline_rank for row in baseline],
    )
    content_for_pref = content_norm if content_available else None
    cf_for_pref = cf_norm if cf_available else None
    preferences, effective, signals = preference_scores(
        content_norm=content_for_pref,
        cf_norm=cf_for_pref,
        weights=weights,
        n=len(ids),
    )
    applied = bool(signals)
    reason = None if applied else REASON_NO_PERSONALIZED_HISTORY

    rows: list[PersonalizedCandidate] = []
    for index, row in enumerate(baseline):
        query_value = query_norm[index]
        preference = 0.0 if not applied else preferences[index]
        final = query_value if not applied else gated_final_score(query_value, preference, gamma)
        rows.append(
            PersonalizedCandidate(
                product_id=row.product_id,
                baseline_rank=row.baseline_rank,
                baseline_score=row.baseline_score,
                query_relevance=query_value,
                preference=preference,
                final_score=final,
                content_affinity=content_norm[index],
                content_available=content_flags[index],
                cf_affinity=cf_norm[index],
                cf_available=cf_flags[index],
                source=row.source,
            )
        )

    if not applied:
        ordered = sorted(rows, key=lambda item: (item.baseline_rank, item.product_id))
    else:
        ordered = sorted(
            rows,
            key=lambda item: (-item.final_score, item.baseline_rank, item.product_id),
        )
    assert_candidate_identity(ids, [item.product_id for item in ordered])
    return PersonalizationOutcome(
        candidates=tuple(ordered),
        applied=applied,
        reason=reason,
        signals_used=channel_name(content_available, cf_available) if applied else (),
        effective_weights=effective,
        degenerate_query_scores=degenerate,
    )


def inject_forbidden(baseline: Sequence[BaselineCandidate], extra_product_id: str) -> PersonalizationOutcome:
    """Helper for tests: rerank must never add ``extra_product_id``."""

    outcome = rerank_candidates(
        baseline,
        content_raw={extra_product_id: 1.0},
        cf_raw=None,
        weights=validate_signal_weights(1.0, 0.0),
        gamma=0.30,
    )
    if extra_product_id in {row.product_id for row in outcome.candidates}:
        raise AssertionError("personalization injected a non-candidate product")
    return outcome
