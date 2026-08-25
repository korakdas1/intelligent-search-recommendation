"""Content, popularity, BPR-MF, and hybrid-rec-v1 recommendations."""

from app.recommendations.constants import CONTENT_REC_VERSION, EVALUATION_VERSION
from app.recommendations.hybrid_constants import HYBRID_REC_VERSION
from app.recommendations.service import (
    recommend_cf_for_user,
    recommend_content_for_user,
    recommend_hybrid_for_user,
    recommend_popular,
    recommend_similar,
)

__all__ = [
    "CONTENT_REC_VERSION",
    "EVALUATION_VERSION",
    "HYBRID_REC_VERSION",
    "recommend_similar",
    "recommend_content_for_user",
    "recommend_cf_for_user",
    "recommend_hybrid_for_user",
    "recommend_popular",
]
