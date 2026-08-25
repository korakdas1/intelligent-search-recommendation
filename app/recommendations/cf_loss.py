"""BPR ranking loss plus batch-only L2 on participating embeddings."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def bpr_ranking_loss(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> torch.Tensor:
    """Mean softplus(-(s_ui - s_uj)). Lower when the positive outranks the negative."""

    if positive_scores.shape != negative_scores.shape:
        raise ValueError("positive and negative score batches must match")
    if not torch.isfinite(positive_scores).all() or not torch.isfinite(negative_scores).all():
        raise ValueError("BPR scores contain NaN or Inf")
    return F.softplus(-(positive_scores - negative_scores)).mean()


def batch_embedding_l2(
    user_vectors: torch.Tensor,
    positive_vectors: torch.Tensor,
    negative_vectors: torch.Tensor,
) -> torch.Tensor:
    """Mean squared L2 of embeddings that participated in this batch only."""

    stacked = torch.stack(
        [
            user_vectors.pow(2).sum(dim=-1),
            positive_vectors.pow(2).sum(dim=-1),
            negative_vectors.pow(2).sum(dim=-1),
        ]
    )
    return stacked.mean()


def bpr_objective(
    positive_scores: torch.Tensor,
    negative_scores: torch.Tensor,
    *,
    user_vectors: torch.Tensor,
    positive_vectors: torch.Tensor,
    negative_vectors: torch.Tensor,
    l2: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ranking = bpr_ranking_loss(positive_scores, negative_scores)
    penalty = batch_embedding_l2(user_vectors, positive_vectors, negative_vectors)
    total = ranking + float(l2) * penalty
    return total, ranking, penalty
