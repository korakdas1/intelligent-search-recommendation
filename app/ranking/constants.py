"""Learning-to-rank constants. rank-features-v1 remains frozen (D-054)."""

from __future__ import annotations

DATASET_VERSION = "ltr-synthetic-v1"
MODEL_VERSION = "ranknet-v1"
LABEL_CLASS = "synthetic"
QUERY_SOURCE_TITLE = "synthetic_title"
QUERY_SOURCE_ATTRIBUTE = "synthetic_attribute"
RERANK_NONE = "none"
RERANK_LTR = "ltr"
LTR_SOURCE = "hybrid_ltr"

DEFAULT_HIDDEN_SIZES = (64, 32)
DEFAULT_DROPOUT = 0.1
DEFAULT_LEARNING_RATE = 1e-3
DEFAULT_BATCH_SIZE = 256
DEFAULT_MAX_EPOCHS = 40
DEFAULT_PATIENCE = 5
DEFAULT_NEGATIVES_PER_POSITIVE = 8
DEFAULT_SPLIT_RATIOS = (0.70, 0.15, 0.15)
STD_FLOOR = 1e-8
