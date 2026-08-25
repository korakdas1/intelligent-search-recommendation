"""Save and load RankNet state_dict + config. Do not pickle the Python model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from app.ranking.constants import MODEL_VERSION
from app.ranking.feature_schema import FEATURE_COUNT, FEATURE_NAMES_SHA256, FEATURE_VERSION
from app.ranking.preprocessing import FeatureScaler, load_scaler, save_scaler
from app.ranking.ranker import RankNetMLP


class RankerArtifactError(ValueError):
    """Ranker artifact is missing or incompatible with rank-features-v1."""


def model_dir(root: Path, model_version: str = MODEL_VERSION) -> Path:
    return Path(root) / model_version


def save_ranker_bundle(
    directory: Path,
    *,
    model: RankNetMLP,
    scaler: FeatureScaler,
    config: dict[str, Any],
    manifest: dict[str, Any],
    history: dict[str, Any] | None = None,
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model_state.pt")
    (directory / "model_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    save_scaler(directory / "scaler.npz", scaler)
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if history is not None:
        (directory / "training_history.json").write_text(
            json.dumps(history, indent=2) + "\n", encoding="utf-8"
        )


def load_model_config(directory: Path) -> dict[str, Any]:
    path = directory / "model_config.json"
    if not path.is_file():
        raise RankerArtifactError(f"missing model_config.json under {directory}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_model_from_config(config: dict[str, Any]) -> RankNetMLP:
    input_dim = int(config.get("input_dim", FEATURE_COUNT))
    if input_dim != FEATURE_COUNT:
        raise RankerArtifactError(f"input_dim {input_dim} != {FEATURE_COUNT}")
    feature_version = str(config.get("feature_version", ""))
    if feature_version and feature_version != FEATURE_VERSION:
        raise RankerArtifactError(f"feature_version {feature_version} != {FEATURE_VERSION}")
    checksum = str(config.get("feature_names_sha256", ""))
    if checksum and checksum != FEATURE_NAMES_SHA256:
        raise RankerArtifactError("feature-name checksum mismatch")
    hidden = tuple(int(width) for width in config.get("hidden_sizes", (64, 32)))
    dropout = float(config.get("dropout", 0.1))
    return RankNetMLP(input_dim=input_dim, hidden_sizes=hidden, dropout=dropout)


def load_ranker_bundle(directory: Path, *, map_location: str = "cpu") -> tuple[RankNetMLP, FeatureScaler, dict[str, Any]]:
    directory = Path(directory)
    config = load_model_config(directory)
    model = build_model_from_config(config)
    state_path = directory / "model_state.pt"
    scaler_path = directory / "scaler.npz"
    if not state_path.is_file() or not scaler_path.is_file():
        raise RankerArtifactError(f"incomplete ranker artifact in {directory}")
    state = torch.load(state_path, map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    scaler = load_scaler(scaler_path)
    if scaler.mean.shape[0] != FEATURE_COUNT:
        raise RankerArtifactError("scaler width mismatch")
    return model, scaler, config
