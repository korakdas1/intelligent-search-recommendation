"""Stable semantic ranking and metadata reorder helpers."""

from app.embeddings.constants import SEMANTIC_SOURCE
from app.search.runtime import SemanticRuntime
from app.search.semantic import _stable_hits
import numpy as np


def test_stable_hits_tie_break_product_id() -> None:
    runtime = SemanticRuntime(
        artifact_version="v",
        dataset_version="d",
        model_name="fake-encoder",
        embedding_dim=2,
        product_ids=np.array(["PBBB", "PAAA"], dtype="U32"),
        index=type("Idx", (), {"ntotal": 2})(),
        backend="flat",
        embedding_manifest={},
        index_manifest={},
        skip_catalog_count_check=True,
    )
    scores = np.array([0.5, 0.5], dtype=np.float32)
    indices = np.array([0, 1], dtype=np.int64)
    hits = _stable_hits(runtime, scores, indices, 2)
    assert [hit.product_id for hit in hits] == ["PAAA", "PBBB"]
    assert all(hit.source == SEMANTIC_SOURCE for hit in hits)


def test_missing_faiss_slots_skipped() -> None:
    runtime = SemanticRuntime(
        artifact_version="v",
        dataset_version="d",
        model_name="fake-encoder",
        embedding_dim=2,
        product_ids=np.array(["ONLY"], dtype="U32"),
        index=type("Idx", (), {"ntotal": 1})(),
        backend="flat",
        embedding_manifest={},
        index_manifest={},
        skip_catalog_count_check=True,
    )
    scores = np.array([0.9, 0.1], dtype=np.float32)
    indices = np.array([0, -1], dtype=np.int64)
    hits = _stable_hits(runtime, scores, indices, 5)
    assert [hit.product_id for hit in hits] == ["ONLY"]
