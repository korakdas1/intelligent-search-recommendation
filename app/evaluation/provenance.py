"""File identity and compatibility checks for search/LTR evaluation only."""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any

from app.embeddings.checksums import sha256_file
from app.ranking.constants import DATASET_VERSION, LABEL_CLASS
from app.ranking.feature_schema import FEATURE_NAMES_SHA256, FEATURE_VERSION
from app.search.faiss_index import describe_index

LTR_FILES = {"queries.json": "queries_sha256", "candidates.parquet": "candidates_sha256"}


class ProvenanceError(ValueError):
    """Declared artifact identity does not match the inputs being consumed."""


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, UnicodeError) as exc:
        raise ProvenanceError(f"invalid {path.name}") from exc
    if not isinstance(manifest, dict):
        raise ProvenanceError(f"{path.name} must be a JSON object")
    return manifest


def _digest(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdefABCDEF" for c in value):
        raise ProvenanceError(f"invalid SHA-256 for {name}")
    return value.lower()


def _validate_fields(manifest: dict[str, Any], expected: dict[str, str]) -> bool:
    for field, value in expected.items():
        if field in manifest and manifest[field] != value:
            raise ProvenanceError(f"{field} mismatch: expected {value!r}")
    return all(field in manifest for field in expected)


def logical_artifact_dir(directory: Path, *, kind: str) -> str:
    """Publish a logical name, never a machine-specific absolute path."""

    return f"{kind}/{directory.resolve().name}"


def ltr_artifact_checksums(directory: Path) -> dict[str, str]:
    """Hash final dataset bytes before the temporary build directory is published."""

    return {name: sha256_file(directory / name) for name in LTR_FILES}


def validate_ltr_dataset(
    directory: Path,
    *,
    required_files: tuple[str, ...] = ("queries.json", "candidates.parquet"),
    expected_dataset_version: str = DATASET_VERSION,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify declarations; legacy or unavailable optional inputs remain unverified.

    Evaluation requires queries, training requires candidates. Existing optional
    files are streamed and checked too, without reading parquet into memory.
    Observed hashes on a legacy bundle are not historical build-time evidence.
    """

    manifest_path = directory / "manifest.json"
    manifest = _read_manifest(manifest_path)
    fields = {
        "dataset_version": expected_dataset_version,
        "label_class": LABEL_CLASS,
        "feature_version": FEATURE_VERSION,
        "feature_names_sha256": FEATURE_NAMES_SHA256,
    }
    complete = _validate_fields(manifest, fields)
    declared = manifest.get("checksums", {})
    if not isinstance(declared, dict):
        raise ProvenanceError("LTR checksums must be a JSON object")
    provenance: dict[str, Any] = {
        "artifact_dir": logical_artifact_dir(directory, kind="ltr"),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        **{field: manifest.get(field) for field in fields},
        "source_order_policy": manifest.get("source_order_policy"),
        "declared_checksums": {},
        "file_status": {},
    }
    for name, field in LTR_FILES.items():
        expected = _digest(declared[name], name) if name in declared else None
        if expected is not None:
            provenance["declared_checksums"][name] = expected
        path = directory / name
        if not path.is_file() and name in required_files:
            raise ProvenanceError(f"missing required LTR file: {name}")
        actual = sha256_file(path) if path.is_file() else None
        if actual is not None and expected is not None and actual != expected:
            raise ProvenanceError(f"{name} checksum mismatch")
        provenance[field] = actual
        provenance["file_status"][name] = (
            "unavailable" if actual is None else "verified" if expected else "legacy_unverified"
        )
    provenance["status"] = (
        "verified" if complete and all(value == "verified" for value in provenance["file_status"].values())
        else "legacy_unverified"
    )
    if provenance["status"] != "verified":
        warnings.warn(
            "LTR provenance is legacy/unverified: exact build-time identity is incomplete; "
            "observed hashes do not establish missing historical provenance",
            RuntimeWarning,
            stacklevel=2,
        )
    return manifest, provenance


def ranker_provenance(runtime: Any, dataset: dict[str, Any]) -> dict[str, Any]:
    """Inspect the loaded directory without changing the serving artifact loader."""

    directory = Path(runtime.directory)
    manifest_path = directory / "manifest.json"
    manifest = _read_manifest(manifest_path)
    complete = _validate_fields(manifest, {
        "model_version": runtime.model_version,
        "feature_version": runtime.feature_version,
        "feature_names_sha256": FEATURE_NAMES_SHA256,
    })
    if runtime.feature_version != FEATURE_VERSION:
        raise ProvenanceError("loaded ranker feature_version mismatch")
    result: dict[str, Any] = {
        "artifact_dir": logical_artifact_dir(directory, kind="models"),
        "model_version": runtime.model_version,
        "feature_version": runtime.feature_version,
        "feature_names_sha256": manifest.get("feature_names_sha256"),
        "dataset_version": manifest.get("dataset_version"),
        "manifest_sha256": sha256_file(manifest_path) if manifest_path.is_file() else None,
        "file_status": {},
    }
    for name, declaration, field in (
        ("model_state.pt", "model_checksum", "model_state_sha256"),
        ("scaler.npz", "scaler_checksum", "scaler_sha256"),
        ("model_config.json", "model_config_checksum", "model_config_sha256"),
    ):
        path = directory / name
        expected = _digest(manifest[declaration], name) if declaration in manifest else None
        actual = sha256_file(path) if path.is_file() else None
        if expected is not None and expected != actual:
            raise ProvenanceError(f"ranker {name} checksum mismatch or missing file")
        result[field] = actual
        result["file_status"][name] = (
            "unavailable" if actual is None else "verified" if expected else "legacy_unverified"
        )
    result["status"] = (
        "verified" if complete and all(value == "verified" for value in result["file_status"].values())
        else "legacy_unverified"
    )
    link = manifest.get("ltr_dataset_provenance", {})
    if not isinstance(link, dict):
        raise ProvenanceError("ranker ltr_dataset_provenance must be a JSON object")
    result["ltr_dataset_provenance"] = {
        field: link.get(field) for field in (
            "status", "manifest_sha256", "queries_sha256", "candidates_sha256",
        )
    }
    linked = 0
    for name, field in LTR_FILES.items():
        if field not in link or link[field] is None:
            continue
        expected = _digest(link[field], f"ranker LTR {name}")
        actual = dataset[field]
        declared = dataset["declared_checksums"].get(name)
        if (actual is not None and actual != expected) or (declared is not None and declared != expected):
            raise ProvenanceError(f"ranker LTR {name} linkage mismatch")
        if actual is not None:
            linked += 1
    result["dataset_linkage_status"] = "verified" if linked == len(LTR_FILES) else "legacy_unverified"
    if result["status"] != "verified" or result["dataset_linkage_status"] != "verified":
        warnings.warn("Ranker artifact or LTR linkage is legacy/unverified", RuntimeWarning, stacklevel=2)
    return result


def semantic_provenance(runtime: Any, *, configured_backend: str) -> dict[str, Any]:
    """Describe actual FAISS state and the manifests checked by the semantic loader."""

    embedding = runtime.embedding_manifest
    index = runtime.index_manifest
    description = describe_index(runtime.index)
    catalog_hash = embedding.get("semantic_catalog_sha256")
    catalog_verified = bool(catalog_hash and runtime._catalog_verified)
    return {
        "status": "verified" if catalog_verified else "legacy_unverified",
        "artifact_version": runtime.artifact_version,
        "dataset_version": runtime.dataset_version,
        "model_name": runtime.model_name,
        "model_revision": embedding.get("model_revision"),
        "semantic_text_version": embedding.get("semantic_text_version"),
        "semantic_catalog_sha256": catalog_hash,
        "catalog_verified": catalog_verified,
        "configured_backend": configured_backend,
        "loaded_backend": runtime.backend,
        "faiss": description,
        "index_sha256": index.get("checksums", {}).get(f"{runtime.backend}.faiss"),
        "product_ids_sha256": embedding.get("checksums", {}).get("product_ids.npy"),
        "hnsw_ef_search": int(runtime.index.hnsw.efSearch) if hasattr(runtime.index, "hnsw") else None,
    }
