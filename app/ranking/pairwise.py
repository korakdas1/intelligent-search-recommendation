"""RankNet pairwise loss and deterministic hard-negative pair sampling."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from app.ranking.constants import DEFAULT_NEGATIVES_PER_POSITIVE
from app.ranking.feature_schema import FEATURE_NAMES


def ranknet_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """Numerically stable RankNet: BCE-with-logits on (s_i - s_j) vs target 1.

    P(i > j) = sigmoid(s_i - s_j). Equivalent to softplus(-(s_i - s_j)) in expectation
    when targets are 1.
    """

    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative score batches must match")
    if not torch.isfinite(positive_scores).all() or not torch.isfinite(negative_scores).all():
        raise ValueError("RankNet scores contain NaN or Inf")
    logits = positive_scores - negative_scores
    target = torch.ones_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)


def pairwise_accuracy(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    return (positive_scores > negative_scores).float().mean()


def sample_hard_negative_pairs(
    group: pd.DataFrame,
    *,
    negatives_per_positive: int = DEFAULT_NEGATIVES_PER_POSITIVE,
) -> list[tuple[int, int]]:
    """Pair each positive with up to N earlier (harder) negatives in baseline order.

    ``candidate_position`` is metadata, not a model feature. Lower position is
    closer to the RRF top and is treated as a harder negative.
    """

    if "relevance" not in group.columns:
        raise ValueError("group must include relevance")
    ordered = group.reset_index(drop=True)
    positive_idx = ordered.index[ordered["relevance"] > 0].tolist()
    negative_idx = ordered.index[ordered["relevance"] <= 0].tolist()
    if not positive_idx or not negative_idx:
        return []
    if "candidate_position" in ordered.columns:
        negative_idx = sorted(negative_idx, key=lambda index: int(ordered.loc[index, "candidate_position"]))
    pairs: list[tuple[int, int]] = []
    for pos in positive_idx:
        chosen = negative_idx[:negatives_per_positive]
        for neg in chosen:
            pairs.append((int(pos), int(neg)))
    return pairs


def pair_feature_tensors(
    group: pd.DataFrame,
    pairs: Sequence[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    matrix = group.loc[:, list(FEATURE_NAMES)].to_numpy(dtype=np.float32)
    positives = np.stack([matrix[left] for left, _ in pairs], axis=0)
    negatives = np.stack([matrix[right] for _, right in pairs], axis=0)
    return positives, negatives
