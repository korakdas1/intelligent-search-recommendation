"""Seeded grouped train/validation/test splits. No source-product leakage."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from app.ranking.constants import DEFAULT_SPLIT_RATIOS


class SplitLeakageError(ValueError):
    """Train, validation, and test source or query IDs overlap."""


def grouped_source_split(
    source_ids: Sequence[str],
    *,
    seed: int,
    ratios: tuple[float, float, float] = DEFAULT_SPLIT_RATIOS,
) -> dict[str, str]:
    """Assign each source_product_id to train, validation, or test.

    Random grouping is allowed here because labels are synthetic, not temporal.
    Do not reuse this for later interaction/recommendation splits.
    """

    train_r, val_r, test_r = ratios
    total = train_r + val_r + test_r
    if abs(total - 1.0) > 1e-6:
        raise ValueError("split ratios must sum to 1")
    unique = sorted(set(source_ids))
    rng = np.random.default_rng(seed)
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(round(n * train_r))
    n_val = int(round(n * val_r))
    if n_train + n_val > n:
        n_val = max(0, n - n_train)
    n_test = n - n_train - n_val
    if n >= 3 and min(n_train, n_val, n_test) == 0:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val
    mapping: dict[str, str] = {}
    for index, source_id in enumerate(shuffled):
        if index < n_train:
            mapping[source_id] = "train"
        elif index < n_train + n_val:
            mapping[source_id] = "validation"
        else:
            mapping[source_id] = "test"
    return mapping


def assert_split_disjoint(
    source_by_split: dict[str, set[str]],
    query_by_split: dict[str, set[str]] | None = None,
) -> None:
    splits = ("train", "validation", "test")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        source_overlap = source_by_split.get(left, set()) & source_by_split.get(right, set())
        if source_overlap:
            raise SplitLeakageError(
                f"source_product_id overlap between {left} and {right}: {sorted(source_overlap)[:8]}"
            )
        if query_by_split is not None:
            query_overlap = query_by_split.get(left, set()) & query_by_split.get(right, set())
            if query_overlap:
                raise SplitLeakageError(
                    f"query_id overlap between {left} and {right}: {sorted(query_overlap)[:8]}"
                )
