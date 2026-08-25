"""Content-recommendation strategy identifiers. Not a learned model."""

CONTENT_REC_VERSION = "content-rec-v1"
EVALUATION_VERSION = "recsys-eval-v1"
SIMILAR_MODE = "similar_content"
USER_CONTENT_MODE = "user_content"
POPULARITY_MODE = "popularity"
SOURCE_CONTENT_ITEM = "content_item"
SOURCE_CONTENT_USER = "content_user"
SOURCE_POPULARITY = "popularity"
ALLOWED_USER_METHODS = frozenset({"content", "cf", "hybrid"})
DEFAULT_USER_METHOD = "content"
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
SEARCH_MARGIN = 16
ZERO_NORM_EPS = 1e-12
