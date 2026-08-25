"""Validate ranking feature matrices. Fail clearly; do not drop rows."""

from __future__ import annotations

import numpy as np

from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_VERSION
from app.ranking.features import FeatureBatch


class FeatureValidationError(ValueError):
    """Feature matrix does not match rank-features-v1."""


def validate_feature_batch(batch: FeatureBatch) -> FeatureBatch:
    if batch.feature_version != FEATURE_VERSION:
        raise FeatureValidationError(
            f"feature_version {batch.feature_version!r} != {FEATURE_VERSION!r}"
        )
    values = batch.values
    if not isinstance(values, np.ndarray):
        raise FeatureValidationError("values must be a numpy ndarray")
    if values.ndim != 2:
        raise FeatureValidationError(f"values must be 2-D, got ndim={values.ndim}")
    if values.dtype != np.float32:
        raise FeatureValidationError(f"values dtype must be float32, got {values.dtype}")
    if values.shape[1] != FEATURE_COUNT:
        raise FeatureValidationError(
            f"expected {FEATURE_COUNT} features, got {values.shape[1]}"
        )
    if values.shape[0] != len(batch.product_ids):
        raise FeatureValidationError(
            f"row count {values.shape[0]} != product_id count {len(batch.product_ids)}"
        )
    if not np.isfinite(values).all():
        raise FeatureValidationError("feature matrix contains NaN or Inf")
    return batch
