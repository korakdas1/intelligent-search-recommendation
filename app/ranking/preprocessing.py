"""Train-only feature standardization. Do not fit on validation or test."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.ranking.constants import STD_FLOOR
from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_NAMES, FEATURE_NAMES_SHA256, FEATURE_VERSION


class PreprocessingError(ValueError):
    """Scaler input is incompatible or non-finite."""


class FeatureScaler:
    def __init__(
        self,
        mean: np.ndarray,
        std: np.ndarray,
        *,
        feature_version: str = FEATURE_VERSION,
        feature_names_sha256: str = FEATURE_NAMES_SHA256,
        zero_variance_indices: tuple[int, ...] = (),
    ) -> None:
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.feature_version = feature_version
        self.feature_names_sha256 = feature_names_sha256
        self.zero_variance_indices = tuple(zero_variance_indices)
        if self.mean.shape != (FEATURE_COUNT,) or self.std.shape != (FEATURE_COUNT,):
            raise PreprocessingError("scaler width must match rank-features-v1 (25)")

    def transform(self, values: np.ndarray) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
            raise PreprocessingError(f"expected (n, {FEATURE_COUNT}) features")
        if not np.isfinite(matrix).all():
            raise PreprocessingError("features contain NaN or Inf")
        scaled = (matrix - self.mean) / self.std
        if not np.isfinite(scaled).all():
            raise PreprocessingError("scaled features contain NaN or Inf")
        return scaled.astype(np.float32, copy=False)


def fit_scaler(train_values: np.ndarray) -> FeatureScaler:
    matrix = np.asarray(train_values, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
        raise PreprocessingError(f"expected (n, {FEATURE_COUNT}) training features")
    if matrix.shape[0] < 1:
        raise PreprocessingError("cannot fit scaler on empty training matrix")
    if not np.isfinite(matrix).all():
        raise PreprocessingError("training features contain NaN or Inf")
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    zero_variance = tuple(int(index) for index in np.where(std < STD_FLOOR)[0])
    std = np.where(std < STD_FLOOR, 1.0, std)
    return FeatureScaler(
        mean.astype(np.float32),
        std.astype(np.float32),
        zero_variance_indices=zero_variance,
    )


def save_scaler(path: Path, scaler: FeatureScaler) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mean=scaler.mean,
        std=scaler.std,
        feature_version=np.array(scaler.feature_version),
        feature_names_sha256=np.array(scaler.feature_names_sha256),
        zero_variance_indices=np.array(scaler.zero_variance_indices, dtype=np.int32),
        feature_names=np.array(FEATURE_NAMES),
    )


def load_scaler(path: Path) -> FeatureScaler:
    with np.load(path, allow_pickle=False) as payload:
        version = str(payload["feature_version"])
        checksum = str(payload["feature_names_sha256"])
        if version != FEATURE_VERSION:
            raise PreprocessingError(f"scaler feature_version {version} != {FEATURE_VERSION}")
        if checksum != FEATURE_NAMES_SHA256:
            raise PreprocessingError("scaler feature-name checksum mismatch")
        zero = payload["zero_variance_indices"]
        return FeatureScaler(
            payload["mean"],
            payload["std"],
            feature_version=version,
            feature_names_sha256=checksum,
            zero_variance_indices=tuple(int(index) for index in zero.tolist()),
        )
