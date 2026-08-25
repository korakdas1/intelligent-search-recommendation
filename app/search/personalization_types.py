"""Types for bounded personalized search. Separate from rank-features-v1."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True, slots=True)
class SignalWeights:
    content: float
    cf: float

    def as_dict(self) -> dict[str, float]:
        return {"content": float(self.content), "cf": float(self.cf)}


@dataclass(frozen=True, slots=True)
class PersonalizationPolicy:
    personalization_version: str
    feature_version: str
    base_retrieval_mode: str
    base_fusion_method: str
    base_rerank_mode: str
    candidate_k: int
    signals: tuple[str, ...]
    weights: SignalWeights
    gamma: float
    gamma_max: float
    normalization: str
    equal_score_query_fallback: str
    gating_formula: str
    tie_break: tuple[str, ...]
    cold_user: str
    query_generator_version: str
    tune_protocol: str
    evaluation_version: str
    created_at: str | None = None
    git_commit: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class BaselineCandidate:
    product_id: str
    baseline_score: float
    baseline_rank: int
    source: str = "hybrid"


@dataclass(frozen=True, slots=True)
class PersonalizationFeatures:
    """personalization-features-v1 row. Finite values only."""

    product_id: str
    content_affinity: float
    content_available: int
    cf_affinity: float
    cf_available: int
    query_relevance: float
    preference: float
    final_score: float
    baseline_rank: int
    baseline_score: float


@dataclass
class PersonalizedCandidate:
    product_id: str
    baseline_rank: int
    baseline_score: float
    query_relevance: float
    preference: float
    final_score: float
    content_affinity: float = 0.0
    content_available: int = 0
    cf_affinity: float = 0.0
    cf_available: int = 0
    source: str = "hybrid"


@dataclass(frozen=True, slots=True)
class PersonalizationOutcome:
    candidates: tuple[PersonalizedCandidate, ...]
    applied: bool
    reason: str | None
    signals_used: tuple[str, ...]
    effective_weights: dict[str, float] = field(default_factory=dict)
    degenerate_query_scores: bool = False


PersonalizationMode = Literal["none", "bounded"]
