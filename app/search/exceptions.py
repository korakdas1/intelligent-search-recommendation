"""Semantic retrieval errors. Distinct from keyword/SQL failures."""


class SemanticUnavailableError(Exception):
    """Semantic artifacts are missing, stale, or incompatible."""


class RankerUnavailableError(Exception):
    """LTR ranker artifact is missing or incompatible."""


class SearchUserNotFound(Exception):
    """Personalization requested for a user absent from PostgreSQL."""
