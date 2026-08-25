"""personalized-search-v1 constants. Not a learned model."""

from __future__ import annotations

PERSONALIZATION_VERSION = "personalized-search-v1"
FEATURE_VERSION = "personalization-features-v1"
PERSONALIZATION_NONE = "none"
PERSONALIZATION_BOUNDED = "bounded"
SUPPORTED_PERSONALIZATION_MODES = (PERSONALIZATION_NONE, PERSONALIZATION_BOUNDED)

SIGNAL_CONTENT = "content"
SIGNAL_CF = "cf"
IMPLEMENTED_SIGNALS = (SIGNAL_CONTENT, SIGNAL_CF)

REASON_NO_PERSONALIZED_HISTORY = "no_personalized_history"
REASON_NOT_REQUESTED = "not_requested"

GAMMA_MAX = 0.30
DEFAULT_GAMMA = 0.10
DEFAULT_CONTENT_WEIGHT = 0.70
DEFAULT_CF_WEIGHT = 0.30
WEIGHT_SUM_TOLERANCE = 1e-6

BASE_RETRIEVAL_MODE = "hybrid"
BASE_FUSION_METHOD = "rrf"
BASE_RERANK_MODE = "none"
DEFAULT_CANDIDATE_K = 100

QUERY_GENERATOR_VERSION = "ltr-synthetic-v1"
QUERY_SOURCE = "synthetic_title"
TUNE_PROTOCOL = "personalized-search-tune-v1"
EVAL_PROTOCOL = "personalized-search-eval-v1"
EXPERIMENT_ID = "E-011"

COLD_USER_BEHAVIOR = "baseline_search"
EQUAL_SCORE_QUERY_FALLBACK = "rank_based"
GATING_FORMULA = "Q * (1 + gamma * P)"
NORMALIZATION = "minmax"
TIE_BREAK = ("final_score_desc", "baseline_rank_asc", "product_id_asc")
