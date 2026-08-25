"""Uniform negatives from CF train-known items. Never sample a known positive."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class UniformNegativeSampler:
    """One negative per positive, redrawn each epoch with ``seed + epoch``."""

    def __init__(self, user_positives: Sequence[set[int]], n_items: int) -> None:
        if n_items < 2:
            raise ValueError("need at least two CF items to sample a negative")
        self.user_positives = [set(items) for items in user_positives]
        self.n_items = int(n_items)
        for index, positives in enumerate(self.user_positives):
            if len(positives) >= self.n_items:
                raise ValueError(f"user {index} has no eligible negative item")

    def sample(
        self,
        user_indices: np.ndarray,
        positive_indices: np.ndarray,
        *,
        seed: int,
        epoch: int,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed) + int(epoch))
        negatives = np.empty(len(user_indices), dtype=np.int64)
        for row, (user_index, positive_index) in enumerate(zip(user_indices, positive_indices, strict=True)):
            forbidden = self.user_positives[int(user_index)]
            if int(positive_index) not in forbidden:
                raise AssertionError("positive is not in the user's train positives")
            while True:
                candidate = int(rng.integers(0, self.n_items))
                if candidate not in forbidden:
                    negatives[row] = candidate
                    break
        return negatives

    def sample_unseen_for_users(
        self,
        user_indices: np.ndarray,
        *,
        seed: int,
        epoch: int,
    ) -> np.ndarray:
        """Negatives excluding train positives. The query item need not be a train positive."""

        rng = np.random.default_rng(int(seed) + int(epoch))
        negatives = np.empty(len(user_indices), dtype=np.int64)
        for row, user_index in enumerate(user_indices):
            forbidden = self.user_positives[int(user_index)]
            while True:
                candidate = int(rng.integers(0, self.n_items))
                if candidate not in forbidden:
                    negatives[row] = candidate
                    break
        return negatives


def assert_triple_valid(
    user_index: int,
    positive_index: int,
    negative_index: int,
    user_positives: Sequence[set[int]],
) -> None:
    positives = user_positives[int(user_index)]
    if int(positive_index) not in positives:
        raise AssertionError("positive not in user train positives")
    if int(negative_index) in positives:
        raise AssertionError("negative is a known train positive")
    if int(positive_index) == int(negative_index):
        raise AssertionError("positive equals negative")
