"""Stored MiniLM product-vector lookup. Does not load Sentence Transformers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.embeddings.normalize import l2_normalize
from app.recommendations.constants import ZERO_NORM_EPS
from app.search.runtime import SemanticRuntime


def embeddings_for_products(
    runtime: SemanticRuntime,
    product_ids: Sequence[str],
) -> tuple[list[str], np.ndarray]:
    """Return unique products that exist in the artifact, with stored vectors."""

    seen: set[str] = set()
    kept: list[str] = []
    rows: list[np.ndarray] = []
    for product_id in product_ids:
        if product_id in seen:
            continue
        seen.add(product_id)
        vector = runtime.embedding_for_product(product_id)
        if vector is None:
            continue
        kept.append(product_id)
        rows.append(vector)
    if not rows:
        return [], np.zeros((0, runtime.embedding_dim), dtype=np.float32)
    return kept, np.stack(rows, axis=0)


def mean_profile(vectors: np.ndarray) -> np.ndarray | None:
    """Unweighted mean of unique item vectors, then L2-normalize."""

    if vectors.size == 0:
        return None
    mean = np.mean(np.asarray(vectors, dtype=np.float32), axis=0)
    if not np.isfinite(mean).all():
        return None
    norm = float(np.linalg.norm(mean))
    if norm < ZERO_NORM_EPS:
        return None
    return l2_normalize(mean)
