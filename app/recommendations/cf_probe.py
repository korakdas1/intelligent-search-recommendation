"""File-only CF artifact probe. Does not import PyTorch."""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings, get_settings
from app.recommendations.cf_constants import MODEL_VERSION


def cf_artifact_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    if settings.cf_model_dir:
        return Path(settings.cf_model_dir)
    version = settings.cf_model_version or MODEL_VERSION
    return Path(settings.artifacts_root) / "models" / version


def probe_cf_artifacts(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    path = cf_artifact_path(settings)
    required = settings.cf_model_is_required()
    present = (
        (path / "model_state.pt").is_file()
        and (path / "model_config.json").is_file()
        and (path / "user_ids.npy").is_file()
        and (path / "product_ids.npy").is_file()
        and (path / "manifest.json").is_file()
    )
    if not present:
        return "unavailable" if required else "skipped"
    return "ok"
