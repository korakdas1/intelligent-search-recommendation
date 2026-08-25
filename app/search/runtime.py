"""Load semantic FAISS artifacts once. Does not load Sentence Transformers at import."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.embeddings.artifacts import (
    ArtifactIncompatibleError,
    ArtifactUnavailableError,
    validate_manifest,
)
from app.embeddings.encoder import TextEncoder
from app.embeddings.versioning import embedding_paths, index_paths
from app.models.product import Product
from app.search.exceptions import SemanticUnavailableError
from app.search.faiss_index import load_index

logger = logging.getLogger(__name__)

_runtime: SemanticRuntime | None = None


@dataclass
class SemanticRuntime:
    artifact_version: str
    dataset_version: str
    model_name: str
    embedding_dim: int
    product_ids: np.ndarray
    index: Any
    backend: str
    embedding_manifest: dict[str, Any]
    index_manifest: dict[str, Any]
    encoder: TextEncoder | None = None
    skip_catalog_count_check: bool = False
    _id_to_row: dict[str, int] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._id_to_row = {str(product_id): index for index, product_id in enumerate(self.product_ids.tolist())}

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal)

    def row_to_product_id(self, row: int) -> str | None:
        if row < 0 or row >= len(self.product_ids):
            return None
        return str(self.product_ids[row])

    def product_id_to_row(self, product_id: str) -> int | None:
        row = self._id_to_row.get(str(product_id))
        return int(row) if row is not None else None

    def embedding_for_product(self, product_id: str) -> np.ndarray | None:
        """Return the stored L2-normalized vector. Does not encode text."""

        row = self.product_id_to_row(product_id)
        if row is None:
            return None
        vector = np.asarray(self.index.reconstruct(int(row)), dtype=np.float32)
        if vector.shape != (self.embedding_dim,) or not np.isfinite(vector).all():
            return None
        return vector


def get_semantic_runtime() -> SemanticRuntime:
    global _runtime
    if _runtime is None:
        _runtime = load_semantic_runtime()
    return _runtime


def set_semantic_runtime(runtime: SemanticRuntime | None) -> None:
    global _runtime
    _runtime = runtime


def reset_semantic_runtime() -> None:
    set_semantic_runtime(None)


def load_semantic_runtime(
    settings: Settings | None = None,
    *,
    encoder: TextEncoder | None = None,
    skip_catalog_count_check: bool = False,
) -> SemanticRuntime:
    """Load FAISS + product-ID mapping. Does not load the embedding model."""

    settings = settings or get_settings()
    version = settings.semantic_artifact_version
    artifacts_root = Path(settings.artifacts_root)
    embed_paths = embedding_paths(artifacts_root, version)
    idx_paths = index_paths(artifacts_root, version)
    backend = settings.semantic_index_type
    index_path = idx_paths["flat"] if backend == "flat" else idx_paths["hnsw"]

    required = [
        embed_paths["product_ids"],
        embed_paths["manifest"],
        idx_paths["manifest"],
        index_path,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise ArtifactUnavailableError(f"semantic artifacts missing: {missing}")

    embedding_manifest = _read_json(embed_paths["manifest"])
    index_manifest = _read_json(idx_paths["manifest"])
    expected = {
        "artifact_version": version,
        "model_name": settings.semantic_model_name,
        "semantic_text_version": settings.semantic_text_version,
        "normalized": True,
    }
    validate_manifest(embedding_manifest, expected=expected)
    if index_manifest.get("artifact_version") != version:
        raise ArtifactIncompatibleError("index manifest artifact_version mismatch")
    if index_manifest.get("dataset_version") != embedding_manifest.get("dataset_version"):
        raise ArtifactIncompatibleError("index and embedding dataset versions differ")

    product_ids = np.load(embed_paths["product_ids"], allow_pickle=False)
    if product_ids.ndim != 1:
        raise ArtifactIncompatibleError("product_ids.npy must be a 1-D array")
    index = load_index(index_path)
    if backend == "hnsw" and hasattr(index, "hnsw"):
        index.hnsw.efSearch = settings.semantic_hnsw_ef_search
    if int(index.ntotal) != int(len(product_ids)):
        raise ArtifactIncompatibleError(
            f"index ntotal={index.ntotal} does not match mapping length={len(product_ids)}"
        )
    dimension = int(embedding_manifest["embedding_dimension"])
    if int(index.d) != dimension:
        raise ArtifactIncompatibleError(f"index dimension {index.d} != manifest {dimension}")
    if settings.semantic_index_type == "hnsw" and not hasattr(index, "hnsw"):
        raise ArtifactIncompatibleError("SEMANTIC_INDEX_TYPE=hnsw but loaded index is not HNSW")
    if settings.semantic_index_type == "flat" and hasattr(index, "hnsw"):
        raise ArtifactIncompatibleError("SEMANTIC_INDEX_TYPE=flat but loaded index is HNSW")

    checksum = embedding_manifest.get("checksums", {}).get("product_ids.npy")
    if checksum:
        from app.embeddings.checksums import sha256_file

        actual = sha256_file(embed_paths["product_ids"])
        if actual != checksum:
            raise ArtifactIncompatibleError("product_ids.npy checksum mismatch")

    return SemanticRuntime(
        artifact_version=version,
        dataset_version=str(embedding_manifest["dataset_version"]),
        model_name=str(embedding_manifest["model_name"]),
        embedding_dim=dimension,
        product_ids=product_ids,
        index=index,
        backend=backend,
        embedding_manifest=embedding_manifest,
        index_manifest=index_manifest,
        encoder=encoder,
        skip_catalog_count_check=skip_catalog_count_check,
    )


def ensure_runtime_matches_catalog(session: Session, runtime: SemanticRuntime) -> None:
    if runtime.skip_catalog_count_check:
        return
    count = session.scalar(
        select(func.count())
        .select_from(Product)
        .where(Product.dataset_version == runtime.dataset_version)
    )
    if count is None:
        return
    if int(count) != runtime.ntotal:
        raise ArtifactIncompatibleError(
            f"catalog has {count} products for {runtime.dataset_version} "
            f"but index ntotal={runtime.ntotal}; rebuild semantic artifacts"
        )


def probe_semantic_artifacts(settings: Settings | None = None) -> str:
    """Return ``ok``, ``unavailable``, or ``skipped`` without loading the model.

    A successful probe caches the FAISS runtime so the first search does not
    reload the index. Sentence Transformers is still not loaded here.
    """

    settings = settings or get_settings()
    required = settings.semantic_index_is_required()
    global _runtime
    if _runtime is not None:
        return "ok"
    try:
        loaded = load_semantic_runtime(settings)
    except (ArtifactUnavailableError, ArtifactIncompatibleError, OSError) as exc:
        if not required:
            logger.info("semantic artifacts not required in this environment: %s", exc)
            return "skipped"
        logger.warning("semantic artifacts unavailable: %s", exc)
        return "unavailable"
    _runtime = loaded
    return "ok"


def require_semantic_runtime(session: Session | None = None) -> SemanticRuntime:
    try:
        runtime = get_semantic_runtime()
        if session is not None:
            ensure_runtime_matches_catalog(session, runtime)
        return runtime
    except (ArtifactUnavailableError, ArtifactIncompatibleError, OSError) as exc:
        reset_semantic_runtime()
        raise SemanticUnavailableError(str(exc)) from exc


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))
