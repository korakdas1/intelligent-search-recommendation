"""Recommendation-specific errors. Distinct from search logging."""


class RecommendationError(Exception):
    """Base recommendation failure."""


class SourceProductNotFound(RecommendationError):
    """Product is absent from PostgreSQL."""


class UserNotFound(RecommendationError):
    """User is absent from PostgreSQL."""


class SourceProductNotInIndex(RecommendationError):
    """Catalog product has no row in the loaded semantic artifact."""


class CfUnavailableError(RecommendationError):
    """CF artifact is missing, stale, or incompatible."""
