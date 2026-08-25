"""Artifact version strings and on-disk layout.

The persistent identity is never just ``latest``. A convenience pointer
may exist in configuration, but directories are named by the version.
"""

from __future__ import annotations

from pathlib import Path

from app.embeddings.constants import EMBEDDING_MODEL_NAME, SEMANTIC_TEXT_VERSION


def model_short_name(model_name: str = EMBEDDING_MODEL_NAME) -> str:
    return model_name.rsplit("/", 1)[-1]


def build_artifact_version(
    *,
    dataset_version: str,
    model_name: str = EMBEDDING_MODEL_NAME,
    text_version: str = SEMANTIC_TEXT_VERSION,
) -> str:
    return f"{dataset_version}__{model_short_name(model_name)}__{text_version}"


def embedding_dir(artifacts_root: Path, artifact_version: str) -> Path:
    return artifacts_root / "embeddings" / artifact_version


def index_dir(artifacts_root: Path, artifact_version: str) -> Path:
    return artifacts_root / "indexes" / artifact_version


def embedding_paths(artifacts_root: Path, artifact_version: str) -> dict[str, Path]:
    base = embedding_dir(artifacts_root, artifact_version)
    return {
        "dir": base,
        "embeddings": base / "embeddings.npy",
        "product_ids": base / "product_ids.npy",
        "manifest": base / "manifest.json",
    }


def index_paths(artifacts_root: Path, artifact_version: str) -> dict[str, Path]:
    base = index_dir(artifacts_root, artifact_version)
    return {
        "dir": base,
        "flat": base / "flat.faiss",
        "hnsw": base / "hnsw.faiss",
        "manifest": base / "manifest.json",
    }
