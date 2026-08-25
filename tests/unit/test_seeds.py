"""Seed helper tests. NumPy and PyTorch are not used here."""

import random

from app.core.seeds import set_random_seed


def test_set_random_seed_is_repeatable() -> None:
    set_random_seed(42)
    first = [random.random() for _ in range(5)]
    set_random_seed(42)
    second = [random.random() for _ in range(5)]
    assert first == second


def test_different_seeds_diverge() -> None:
    set_random_seed(1)
    first = random.random()
    set_random_seed(2)
    second = random.random()
    assert first != second
