"""Seed Python, NumPy, and PyTorch for RankNet training."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from app.core.seeds import set_random_seed


def set_ltr_seeds(seed: int) -> None:
    set_random_seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(False)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
