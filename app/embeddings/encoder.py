"""Text encoders. Sentence Transformers is imported only when loading the real model."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from app.embeddings.normalize import l2_normalize


class TextEncoder(Protocol):
    model_name: str
    embedding_dim: int
    device: str

    def encode(self, texts: Sequence[str], *, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        """Return float32 L2-normalized embeddings with shape (n, dim)."""


class FakeEncoder:
    """Deterministic offline encoder for tests. Does not download weights."""

    def __init__(self, *, dim: int = 8, model_name: str = "fake-encoder") -> None:
        self.embedding_dim = dim
        self.model_name = model_name
        self.device = "cpu"
        self.model_revision = "test"
        self.model_license = "test-only"

    def encode(self, texts: Sequence[str], *, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        del batch_size, show_progress
        rows = np.zeros((len(texts), self.embedding_dim), dtype=np.float32)
        for index, text in enumerate(texts):
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            seed = int.from_bytes(digest[:8], "little")
            rng = np.random.default_rng(seed)
            rows[index] = rng.standard_normal(self.embedding_dim, dtype=np.float32)
        return l2_normalize(rows)


class SentenceTransformerEncoder:
    """Lazy wrapper around a local SentenceTransformer. Not imported at app start."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str,
        model_revision: str | None = None,
        model_license: str | None = None,
        embedding_dim: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self.model_revision = model_revision
        self.model_license = model_license
        self._model: Any = None
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        if self._embedding_dim is None:
            self._load()
        assert self._embedding_dim is not None
        return self._embedding_dim

    def _load(self) -> None:
        if self._model is not None:
            return
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name, device=self.device)
        dim_fn = getattr(self._model, "get_embedding_dimension", None) or getattr(
            self._model, "get_sentence_embedding_dimension"
        )
        self._embedding_dim = int(dim_fn())

    def encode(self, texts: Sequence[str], *, batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
        self._load()
        vectors = self._model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return array


def resolve_device(choice: str) -> str:
    """Return ``cpu`` or ``cuda``. ``auto`` prefers CUDA when available."""

    normalized = choice.strip().lower()
    if normalized == "cpu":
        return "cpu"
    if normalized == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("SEMANTIC_DEVICE=cuda but CUDA is not available")
        return "cuda"
    if normalized == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    raise ValueError(f"unknown device choice: {choice!r}")
