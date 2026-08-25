"""BPR-MF v1 identifiers. Not a hybrid or content model."""

from __future__ import annotations

from app.recommendations.constants import EVALUATION_VERSION

MODEL_VERSION = "bpr-mf-v1"
MODEL_TYPE = "bpr_mf"
TUNE_PROTOCOL = "cf-tune-v1"
USER_CF_MODE = "user_cf"
SOURCE_CF = "cf"
REASON_CF_USER_UNAVAILABLE = "cf_user_unavailable"
REASON_ALL_CF_ITEMS_SEEN = "all_cf_items_seen"

# Authoritative recsys-eval-v1 PostgreSQL snapshot. Not the historical Parquet 22,634-user count.
EXPECTED_EVAL_USERS = 22628
EXPECTED_TWO_CORE_PAIRS = 57114
EXPECTED_HIDDEN_CHECKSUM = "3e67446402e88f65afdd3bd4ca8c8f6eb0388748c0fe8b45f3180552dbdcd3be"
CATALOG_SIZE_E008 = 112578

USER_ID_DTYPE = "U64"
PRODUCT_ID_DTYPE = "U16"

DEFAULT_EMBEDDING_DIM = 64
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 1024
DEFAULT_L2 = 1e-4
DEFAULT_MAX_EPOCHS = 100
DEFAULT_PATIENCE = 8
DEFAULT_NEGATIVES_PER_POSITIVE = 1
EMBEDDING_INIT_STD = 0.01
SELECTION_METRIC = "ndcg@10"
NEGATIVE_POLICY = "uniform_train_known_items_dynamic_seed_plus_epoch"

ALLOWED_USER_METHODS = frozenset({"content", "cf", "hybrid"})
