"""ANN neighbor-recall helpers.

ANNRecall@K measures overlap with exact FAISS neighbors. It is not
information-retrieval Recall@K against human relevance labels.
"""

from __future__ import annotations

from collections.abc import Sequence


def neighbor_recall_at_k(
    exact_ids: Sequence[object],
    ann_ids: Sequence[object],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    exact_set = set(list(exact_ids)[:k])
    ann_set = set(list(ann_ids)[:k])
    if not exact_set:
        return 0.0
    return len(exact_set & ann_set) / float(k)


def mean_neighbor_recall_at_k(
    exact_lists: Sequence[Sequence[object]],
    ann_lists: Sequence[Sequence[object]],
    k: int,
) -> float:
    if len(exact_lists) != len(ann_lists):
        raise ValueError("exact and ANN result lists must be the same length")
    if not exact_lists:
        return 0.0
    total = 0.0
    for exact_ids, ann_ids in zip(exact_lists, ann_lists, strict=True):
        total += neighbor_recall_at_k(exact_ids, ann_ids, k)
    return total / len(exact_lists)
