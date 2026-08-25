"""FAISS exact and HNSW index helpers.

``IndexFlatIP`` is exact nearest-neighbor search, not ANN.
``IndexHNSWFlat`` is the true approximate index (D-019).
"""

from __future__ import annotations

from pathlib import Path

import faiss
import numpy as np

from app.embeddings.constants import ANN_INDEX_TYPE, EXACT_INDEX_TYPE, FAISS_METRIC


def build_flat_ip_index(embeddings: np.ndarray) -> faiss.Index:
    """Build an exact inner-product index over L2-normalized float32 vectors."""

    matrix = _as_float32_2d(embeddings)
    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)
    return index


def build_hnsw_ip_index(
    embeddings: np.ndarray,
    *,
    m: int,
    ef_construction: int,
    ef_search: int | None = None,
) -> faiss.Index:
    """Build a genuine HNSW ANN index using inner product on unit vectors."""

    matrix = _as_float32_2d(embeddings)
    index = faiss.IndexHNSWFlat(matrix.shape[1], m, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = ef_construction
    if ef_search is not None:
        index.hnsw.efSearch = ef_search
    index.add(matrix)
    return index


def search_index(
    index: faiss.Index,
    query: np.ndarray,
    top_k: int,
    *,
    ef_search: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (scores, indices) for a single query or a batch.

    Scores are inner products. Invalid FAISS slots are ``-1``.
    """

    queries = _as_float32_2d(query)
    if top_k < 1:
        empty_scores = np.zeros((queries.shape[0], 0), dtype=np.float32)
        empty_ids = np.zeros((queries.shape[0], 0), dtype=np.int64)
        return empty_scores, empty_ids
    k = min(top_k, max(index.ntotal, 1))
    previous = None
    if ef_search is not None and hasattr(index, "hnsw"):
        previous = index.hnsw.efSearch
        index.hnsw.efSearch = ef_search
    try:
        scores, indices = index.search(queries, k)
    finally:
        if previous is not None:
            index.hnsw.efSearch = previous
    return scores, indices


def save_index(index: faiss.Index, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_index(path: Path) -> faiss.Index:
    return faiss.read_index(str(path))


def describe_index(index: faiss.Index) -> dict[str, int | str]:
    name = type(index).__name__
    kind = EXACT_INDEX_TYPE if "FlatIP" in name and "HNSW" not in name else name
    if isinstance(index, faiss.IndexHNSWFlat) or "HNSW" in name:
        kind = ANN_INDEX_TYPE
    elif isinstance(index, faiss.IndexFlatIP) or name == "IndexFlatIP":
        kind = EXACT_INDEX_TYPE
    return {
        "type": kind,
        "metric": FAISS_METRIC,
        "ntotal": int(index.ntotal),
        "dimension": int(index.d),
    }


def _as_float32_2d(vectors: np.ndarray) -> np.ndarray:
    array = np.asarray(vectors, dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if not array.flags["C_CONTIGUOUS"]:
        array = np.ascontiguousarray(array)
    return array
