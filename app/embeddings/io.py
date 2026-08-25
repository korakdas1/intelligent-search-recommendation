"""Read/write semantic embedding and mapping files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from app.embeddings.checksums import sha256_file


def save_embeddings(path: Path, embeddings: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(embeddings, dtype=np.float32)
    np.save(path, array)


def save_product_ids(path: Path, product_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(product_ids, dtype="U32")
    np.save(path, array)


def load_product_ids(path: Path) -> np.ndarray:
    return np.load(path, allow_pickle=False)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def file_size_bytes(path: Path) -> int:
    return path.stat().st_size


def checksum_map(paths: dict[str, Path]) -> dict[str, str]:
    return {name: sha256_file(path) for name, path in paths.items() if path.is_file()}
