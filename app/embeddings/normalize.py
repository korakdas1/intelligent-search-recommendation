"""L2 normalization for cosine similarity via inner product."""

from __future__ import annotations

import numpy as np

NORM_EPS = 1e-12


def l2_normalize(vectors: np.ndarray, *, eps: float = NORM_EPS) -> np.ndarray:
    """Return float32 row-normalized vectors.

    Zero (or near-zero) rows stay near-zero rather than exploding.
    """

    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / np.maximum(norms, eps)
    if squeeze:
        return normalized[0]
    return normalized.astype(np.float32, copy=False)


def row_norms(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        return np.linalg.norm(array, keepdims=True)
    return np.linalg.norm(array, axis=1)


def is_unit_normalized(vectors: np.ndarray, *, atol: float = 1e-5) -> bool:
    norms = row_norms(vectors)
    return bool(np.allclose(norms, 1.0, atol=atol))
