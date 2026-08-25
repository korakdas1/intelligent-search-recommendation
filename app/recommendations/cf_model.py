"""Pure latent-factor BPR matrix factorization. No MiniLM, no MLP, no biases."""

from __future__ import annotations

import torch
from torch import nn

from app.recommendations.cf_constants import EMBEDDING_INIT_STD


class BPRMatrixFactorization(nn.Module):
    """Score(u, i) = p_u · q_i. Cold catalog items are simply absent from ``item_embedding``."""

    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        if n_users < 1 or n_items < 1:
            raise ValueError("n_users and n_items must be positive")
        if embedding_dim < 1:
            raise ValueError("embedding_dim must be positive")
        self.n_users = int(n_users)
        self.n_items = int(n_items)
        self.embedding_dim = int(embedding_dim)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_dim)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_dim)
        nn.init.normal_(self.user_embedding.weight, mean=0.0, std=EMBEDDING_INIT_STD)
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=EMBEDDING_INIT_STD)

    def score(self, user_index: torch.Tensor, item_index: torch.Tensor) -> torch.Tensor:
        users = self.user_embedding(user_index)
        items = self.item_embedding(item_index)
        return (users * items).sum(dim=-1)

    def score_items(self, user_index: torch.Tensor) -> torch.Tensor:
        """Batched scores against every trained item. Shape (batch, n_items)."""

        users = self.user_embedding(user_index)
        return users @ self.item_embedding.weight.T

    def parameter_count(self) -> int:
        return sum(int(parameter.numel()) for parameter in self.parameters())

    def embedding_parameter_counts(self) -> dict[str, int]:
        return {
            "user_embedding": int(self.user_embedding.weight.numel()),
            "item_embedding": int(self.item_embedding.weight.numel()),
            "total": self.parameter_count(),
        }
