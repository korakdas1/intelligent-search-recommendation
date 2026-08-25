"""Save/load BPR-MF state_dict plus mappings. Never pickle the Python module."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from app.recommendations.cf_constants import (
    MODEL_TYPE,
    MODEL_VERSION,
    PRODUCT_ID_DTYPE,
    USER_ID_DTYPE,
)
from app.recommendations.cf_data import encode_id_array, mapping_checksum
from app.recommendations.cf_model import BPRMatrixFactorization


class CfArtifactError(ValueError):
    """CF artifact is missing or incompatible."""


def model_dir(root: Path, model_version: str = MODEL_VERSION) -> Path:
    return Path(root) / "models" / model_version


def save_cf_bundle(
    directory: Path,
    *,
    model: BPRMatrixFactorization,
    user_ids: list[str] | tuple[str, ...],
    product_ids: list[str] | tuple[str, ...],
    config: dict[str, Any],
    manifest: dict[str, Any],
    history: dict[str, Any] | None = None,
    tuning: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model_state.pt")
    np.save(directory / "user_ids.npy", encode_id_array(user_ids, USER_ID_DTYPE))
    np.save(directory / "product_ids.npy", encode_id_array(product_ids, PRODUCT_ID_DTYPE))
    (directory / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if history is not None:
        (directory / "training_history.json").write_text(json.dumps(history, indent=2) + "\n", encoding="utf-8")
    if tuning is not None:
        (directory / "tuning_results.json").write_text(json.dumps(tuning, indent=2) + "\n", encoding="utf-8")


def load_id_array(path: Path) -> list[str]:
    values = np.load(path, allow_pickle=False)
    return [str(item) for item in values.tolist()]


def build_model_from_config(config: dict[str, Any], n_users: int, n_items: int) -> BPRMatrixFactorization:
    model_type = str(config.get("model_type", ""))
    if model_type and model_type != MODEL_TYPE:
        raise CfArtifactError(f"model_type {model_type!r} != {MODEL_TYPE!r}")
    dim = int(config["embedding_dim"])
    if int(config.get("n_users", n_users)) != n_users:
        raise CfArtifactError("n_users does not match user mapping")
    if int(config.get("n_items", n_items)) != n_items:
        raise CfArtifactError("n_items does not match item mapping")
    return BPRMatrixFactorization(n_users=n_users, n_items=n_items, embedding_dim=dim)


def load_cf_bundle(
    directory: Path,
    *,
    map_location: str = "cpu",
) -> tuple[BPRMatrixFactorization, list[str], list[str], dict[str, Any], dict[str, Any]]:
    directory = Path(directory)
    config_path = directory / "model_config.json"
    state_path = directory / "model_state.pt"
    user_path = directory / "user_ids.npy"
    item_path = directory / "product_ids.npy"
    manifest_path = directory / "manifest.json"
    if not all(path.is_file() for path in (config_path, state_path, user_path, item_path, manifest_path)):
        raise CfArtifactError(f"incomplete CF artifact in {directory}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    user_ids = load_id_array(user_path)
    product_ids = load_id_array(item_path)
    if len(user_ids) != len(set(user_ids)):
        raise CfArtifactError("duplicate user ids in mapping")
    if len(product_ids) != len(set(product_ids)):
        raise CfArtifactError("duplicate product ids in mapping")
    stored_user, stored_item = mapping_checksum(user_ids, product_ids)
    if manifest.get("user_mapping_checksum") and manifest["user_mapping_checksum"] != stored_user:
        raise CfArtifactError("user mapping checksum mismatch")
    if manifest.get("item_mapping_checksum") and manifest["item_mapping_checksum"] != stored_item:
        raise CfArtifactError("item mapping checksum mismatch")
    model = build_model_from_config(config, n_users=len(user_ids), n_items=len(product_ids))
    state = torch.load(state_path, map_location=map_location, weights_only=True)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise CfArtifactError(f"state_dict incompatible with mappings: {exc}") from exc
    expected_user = (len(user_ids), int(config["embedding_dim"]))
    expected_item = (len(product_ids), int(config["embedding_dim"]))
    if tuple(model.user_embedding.weight.shape) != expected_user:
        raise CfArtifactError("user embedding shape mismatch")
    if tuple(model.item_embedding.weight.shape) != expected_item:
        raise CfArtifactError("item embedding shape mismatch")
    model.eval()
    return model, user_ids, product_ids, config, manifest
