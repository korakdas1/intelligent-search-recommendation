"""Exact FAISS IndexFlatIP vs independent cosine/IP calculation."""

from pathlib import Path

import numpy as np

from app.embeddings.normalize import l2_normalize
from app.search.faiss_index import build_flat_ip_index, load_index, save_index, search_index


def _numpy_topk(docs: np.ndarray, query: np.ndarray, k: int) -> tuple[list[int], list[float]]:
    scores = docs @ query
    order = np.argsort(-scores, kind="stable")
    # stable tie-break by index (product_id proxy)
    tied = []
    for idx in order:
        tied.append((float(-scores[idx]), int(idx)))
    tied.sort(key=lambda item: (item[0], item[1]))
    top = tied[:k]
    return [item[1] for item in top], [-item[0] for item in top]


def test_flat_ip_matches_numpy_cosine() -> None:
    rng = np.random.default_rng(0)
    docs = l2_normalize(rng.standard_normal((6, 8)).astype(np.float32))
    query = l2_normalize(rng.standard_normal(8).astype(np.float32))
    index = build_flat_ip_index(docs)
    assert index.ntotal == 6
    assert index.d == 8
    scores, indices = search_index(index, query, 3)
    expected_ids, expected_scores = _numpy_topk(docs, query, 3)
    faiss_ids = [int(row) for row in indices[0]]
    # Re-apply documented tie-break: score DESC, id ASC
    combined = sorted(
        [(float(score), int(row)) for score, row in zip(scores[0], indices[0], strict=True)],
        key=lambda item: (-item[0], item[1]),
    )
    assert [item[1] for item in combined] == expected_ids
    assert np.allclose([item[0] for item in combined], expected_scores, atol=1e-5)
    assert faiss_ids  # FAISS returned neighbors


def test_flat_index_save_load(tmp_path: Path) -> None:
    docs = l2_normalize(np.eye(4, dtype=np.float32))
    index = build_flat_ip_index(docs)
    path = tmp_path / "flat.faiss"
    save_index(index, path)
    loaded = load_index(path)
    assert loaded.ntotal == 4
    assert loaded.d == 4
    query = l2_normalize(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32))
    _scores, indices = search_index(loaded, query, 1)
    assert int(indices[0][0]) == 1
