"""Small feed-forward RankNet scorer. No transformer, no embeddings."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from app.ranking.constants import DEFAULT_DROPOUT, DEFAULT_HIDDEN_SIZES
from app.ranking.feature_schema import FEATURE_COUNT


class RankNetMLP(nn.Module):
    """Maps a 25-d feature row to one unbounded score."""

    def __init__(
        self,
        *,
        input_dim: int = FEATURE_COUNT,
        hidden_sizes: Sequence[int] = DEFAULT_HIDDEN_SIZES,
        dropout: float = DEFAULT_DROPOUT,
    ) -> None:
        super().__init__()
        if input_dim != FEATURE_COUNT:
            raise ValueError(f"input_dim must be {FEATURE_COUNT} for rank-features-v1")
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        layers: list[nn.Module] = []
        previous = input_dim
        for index, width in enumerate(hidden_sizes):
            layers.append(nn.Linear(previous, width))
            layers.append(nn.ReLU())
            if index == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = width
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.hidden_sizes = tuple(hidden_sizes)
        self.dropout = dropout

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(f"expected (batch, {self.input_dim}), got {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise ValueError("ranker input contains NaN or Inf")
        scores = self.network(features).squeeze(-1)
        if scores.ndim != 1:
            scores = scores.reshape(features.shape[0])
        return scores

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())
