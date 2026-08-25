"""Load and validate semantic artifact manifests. Does not load models."""

from __future__ import annotations

from typing import Any

REQUIRED_MANIFEST_KEYS = (
    "artifact_version",
    "dataset_version",
    "product_count",
    "model_name",
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
    missing = [key for key in REQUIRED_MANIFEST_KEYS if key not in manifest]
    if missing:
        raise ArtifactIncompatibleError(f"manifest missing keys: {missing}")
    if not manifest.get("normalized"):
        raise ArtifactIncompatibleError("manifest.normalized must be true")
    if expected:
        for key, value in expected.items():
            actual = manifest.get(key)
            if actual != value:
                raise ArtifactIncompatibleError(
                    f"manifest {key}={actual!r} does not match expected {value!r}"
                )
