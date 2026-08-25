"""Hybrid recommendation candidate and policy types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from app.recommendations.hybrid_constants import (
    CHANNEL_CF,
    CHANNEL_CONTENT,
    CHANNEL_POPULARITY,
)
from app.recommendations.ranking import ScoredItem


@dataclass(frozen=True, slots=True)
class ChannelWeights:
    content: float
    cf: float
    popularity: float

    def as_dict(self) -> dict[str, float]:
        return {
            CHANNEL_CONTENT: float(self.content),
            CHANNEL_CF: float(self.cf),
            CHANNEL_POPULARITY: float(self.popularity),
        }


@dataclass(frozen=True, slots=True)
class HybridPolicy:
    hybrid_version: str
    fusion_method: Literal["rrf", "weighted"]
    rrf_k: int
    weights: ChannelWeights
    candidate_k_min: int
    candidate_k_max: int
    candidate_k_multiplier: int
    fallback_policy_version: str
    content_rec_version: str
    cf_model_version: str
    popularity_rule: str
    evaluation_version: str
    tune_protocol: str
    created_at: str | None = None
    git_commit: str | None = None
    note: str | None = None


@dataclass(frozen=True, slots=True)
class ChannelRanking:
    name: str
    available: bool
    items: tuple[ScoredItem, ...] = ()


@dataclass
class HybridCandidate:
    product_id: str
    fused_score: float = 0.0
    content_present: bool = False
    content_rank: int | None = None
    content_raw_score: float | None = None
    cf_present: bool = False
    cf_rank: int | None = None
    cf_raw_score: float | None = None
    popularity_present: bool = False
    popularity_rank: int | None = None
    popularity_raw_score: float | None = None
    sources: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ChannelAvailability:
    content: bool
    cf: bool
    popularity: bool
    channels_used: tuple[str, ...]
    fallback_reason: str | None
    content_usable_history: bool
    cf_user_in_model: bool
    semantic_artifact_available: bool
    cf_artifact_available: bool


@dataclass
class HybridFusionResult:
    ranked: list[HybridCandidate] = field(default_factory=list)
    availability: ChannelAvailability | None = None
    effective_weights: dict[str, float] | None = None
    fusion_method: str = "rrf"
    candidate_k: int = 0
