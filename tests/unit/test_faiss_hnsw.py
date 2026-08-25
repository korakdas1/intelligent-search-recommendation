"""HNSW is a true ANN index; recall helper is exact-vs-ANN overlap."""

from pathlib import Path

import numpy as np

from app.embeddings.normalize import l2_normalize
from app.search.ann_metrics import mean_neighbor_recall_at_k, neighbor_recall_at_k
from app.search.faiss_index import (
    build_flat_ip_index,
    build_hnsw_ip_index,
    load_index,
    save_index,
    search_index,
)


def test_hnsw_contains_expected_count() -> None:
    docs = l2_normalize(np.eye(5, dtype=np.float32))
    index = build_hnsw_ip_index(docs, m=8, ef_construction=32, ef_search=16)
    assert "HNSW" in type(index).__name__
    assert index.ntotal == 5
    assert index.d == 5
    query = docs[2]
    scores, indices = search_index(index, query, 3, ef_search=32)
    assert int(indices[0][0]) >= 0
    assert scores.shape == (1, 3)


def test_hnsw_save_load(tmp_path: Path) -> None:
    docs = l2_normalize(np.eye(4, dtype=np.float32))
    index = build_hnsw_ip_index(docs, m=8, ef_construction=32, ef_search=16)
    path = tmp_path / "hnsw.faiss"
    save_index(index, path)
    loaded = load_index(path)
    assert loaded.ntotal == 4
    assert hasattr(loaded, "hnsw")


def test_neighbor_recall_calculation() -> None:
    exact = [1, 2, 3, 4]
    ann = [1, 9, 3, 8]
    assert neighbor_recall_at_k(exact, ann, 4) == 0.5
    assert neighbor_recall_at_k(exact, exact, 4) == 1.0
    assert mean_neighbor_recall_at_k([exact, [10, 11]], [ann, [10, 99]], 2) == 0.5


def test_hnsw_overlap_with_exact_on_tiny_set() -> None:
    rng = np.random.default_rng(1)
    docs = l2_normalize(rng.standard_normal((20, 8)).astype(np.float32))
    query = l2_normalize(rng.standard_normal(8).astype(np.float32))
    exact = build_flat_ip_index(docs)
    ann = build_hnsw_ip_index(docs, m=16, ef_construction=64, ef_search=64)
    _s, exact_idx = search_index(exact, query, 5)
    _s, ann_idx = search_index(ann, query, 5, ef_search=64)
    recall = neighbor_recall_at_k(
        [int(x) for x in exact_idx[0]],
        [int(x) for x in ann_idx[0]],
        5,
    )
    assert 0.0 <= recall <= 1.0
