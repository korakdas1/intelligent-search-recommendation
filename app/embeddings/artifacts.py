"""Load and validate semantic artifact manifests. Does not load models."""

from __future__ import annotations

import re
from typing import Any

from app.embeddings.constants import FAISS_METRIC, VECTOR_DTYPE

REQUIRED_MANIFEST_KEYS = (
    "artifact_version",
    "dataset_version",
    "product_count",
    "model_name",
    "model_revision",
    "embedding_dimension",
    "dtype",
    "normalized",
    "semantic_text_version",
    "faiss_metric",
)


class ArtifactError(Exception):
    """Base class for missing or incompatible semantic artifacts."""


class ArtifactUnavailableError(ArtifactError):
    """Required files are missing."""


class ArtifactIncompatibleError(ArtifactError):
    """Files exist but do not match the current catalog or configuration."""


def validate_manifest(manifest: dict[str, Any], *, expected: dict[str, Any] | None = None) -> None:
    if not isinstance(manifest, dict):
        raise ArtifactIncompatibleError("manifest must be a JSON object")
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ArtifactIncompatibleError(f"manifest missing keys: {missing}")
    if manifest.get("normalized") is not True:
        raise ArtifactIncompatibleError("manifest.normalized must be true")
    for key in ("product_count", "embedding_dimension"):
        if type(manifest[key]) is not int or manifest[key] < 1:
            raise ArtifactIncompatibleError(f"manifest {key} must be a positive integer")
    for key, value in (("dtype", VECTOR_DTYPE), ("faiss_metric", FAISS_METRIC)):
        if manifest[key] != value:
            raise ArtifactIncompatibleError(f"manifest {key} must be {value!r}")
    if expected:
        for key, value in expected.items():
            actual = manifest.get(key)
            if actual != value:
                raise ArtifactIncompatibleError(
                    f"manifest {key}={actual!r} does not match expected {value!r}"
                )


def validate_manifest_pair(
    embedding: dict[str, Any], index: dict[str, Any], *, expected: dict[str, Any]
) -> None:
    """Compare build identity, excluding timing and other incidental metadata.

    Both generations use the existing checksum structure. Fingerprint presence
    opts a bundle into stronger catalog provenance; partial/invalid presence
    must never silently downgrade it to legacy behavior.
    """

    validate_manifest(embedding, expected=expected)
    validate_manifest(index, expected=expected)
    for key in REQUIRED_MANIFEST_KEYS:
        if embedding[key] != index[key]:
            raise ArtifactIncompatibleError(f"embedding/index manifest {key} mismatch")
    key = "semantic_catalog_sha256"
    if key in embedding or key in index:
        for manifest in (embedding, index):
            if not _is_sha256(manifest.get(key)):
                raise ArtifactIncompatibleError(f"manifest {key} must be a SHA-256 hex digest")
        if embedding[key] != index[key]:
            raise ArtifactIncompatibleError(f"embedding/index manifest {key} mismatch")


def manifest_checksum(manifest: dict[str, Any], filename: str) -> str:
    """Checksums written by the existing builder are required, even for legacy.

    The supported legacy format lacks only the semantic catalog fingerprint,
    not the selected index or mapping checksums. Do not accept checksum-less
    artifacts or synthesize missing provenance during loading.
    """

    checksums = manifest.get("checksums")
    checksum = checksums.get(filename) if isinstance(checksums, dict) else None
    if not _is_sha256(checksum):
        raise ArtifactIncompatibleError(f"manifest missing/invalid {filename} checksum")
    return checksum


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None
