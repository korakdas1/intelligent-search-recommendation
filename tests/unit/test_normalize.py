"""L2 normalization helpers."""

import numpy as np

from app.embeddings.normalize import is_unit_normalized, l2_normalize, row_norms


def test_l2_normalize_unit_rows() -> None:
    vectors = np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float32)
    normalized = l2_normalize(vectors)
    assert normalized.dtype == np.float32
    assert np.allclose(row_norms(normalized), 1.0, atol=1e-6)
    assert is_unit_normalized(normalized)


def test_query_and_document_share_convention() -> None:
    docs = l2_normalize(np.array([[1.0, 1.0, 0.0]], dtype=np.float32))
    query = l2_normalize(np.array([1.0, 1.0, 0.0], dtype=np.float32))
    assert np.allclose(docs[0], query)
    assert np.isclose(float(np.dot(docs[0], query)), 1.0)
