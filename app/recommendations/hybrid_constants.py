"""hybrid-rec-v1 identifiers. Deterministic policy, not a trained model."""

from __future__ import annotations

from app.recommendations.cf_constants import MODEL_VERSION as CF_MODEL_VERSION
from app.recommendations.constants import CONTENT_REC_VERSION, EVALUATION_VERSION

HYBRID_REC_VERSION = "hybrid-rec-v1"
FALLBACK_POLICY_VERSION = "cold-start-v1"
TUNE_PROTOCOL = "hybrid-rec-tune-v1"
USER_HYBRID_MODE = "user_hybrid"
SOURCE_HYBRID = "hybrid"

CHANNEL_CONTENT = "content"
CHANNEL_CF = "cf"
CHANNEL_POPULARITY = "popularity"
HYBRID_CHANNELS = (CHANNEL_CONTENT, CHANNEL_CF, CHANNEL_POPULARITY)

DEFAULT_RRF_K = 60
CANDIDATE_K_MIN = 100
CANDIDATE_K_MAX = 500
CANDIDATE_K_MULTIPLIER = 5
WEIGHT_SUM_TOLERANCE = 1e-6

POPULARITY_RULE_SERVING = "count_interactions_current_postgres"
POPULARITY_RULE_EVAL = "count_interactions_train_safe_split"

REASON_NO_PERSONALIZED_HISTORY = "no_personalized_history"
REASON_CF_ARTIFACT_UNAVAILABLE = "cf_artifact_unavailable"
REASON_CONTENT_ARTIFACT_UNAVAILABLE = "content_artifact_unavailable"
REASON_NO_RECOMMENDATION_CHANNEL = "no_recommendation_channel"

CONTENT_DEP_VERSION = CONTENT_REC_VERSION
CF_DEP_VERSION = CF_MODEL_VERSION
EVAL_DEP_VERSION = EVALUATION_VERSION
