"""Deterministic random seeds.

Startup seeds the Python standard-library ``random`` module only.
NumPy, PyTorch, or CUDA seed helpers belong next to the libraries that use them.
"""

from __future__ import annotations

import os
import random


def set_random_seed(seed: int) -> None:
    """Seed ``random`` and record ``PYTHONHASHSEED`` for this process."""

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
