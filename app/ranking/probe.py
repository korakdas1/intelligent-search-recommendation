"""File-only LTR artifact probe. Does not import PyTorch."""

from __future__ import annotations

from pathlib import Path

from app.core.config import Settings, get_settings
from app.ranking.constants import MODEL_VERSION


def ranker_artifact_path(settings: Settings | None = None) -> Path:
    settings = settings or get_settings()
    if settings.ltr_ranker_dir:
        return Path(settings.ltr_ranker_dir)
    version = settings.ltr_ranker_version or MODEL_VERSION
    return Path(settings.artifacts_root) / "models" / version


def probe_ltr_artifacts(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    path = ranker_artifact_path(settings)
    required = settings.ltr_ranker_is_required()
    present = (
        (path / "model_state.pt").is_file()
        and (path / "model_config.json").is_file()
        and (path / "scaler.npz").is_file()
    )
    if not present:
        return "unavailable" if required else "skipped"
    return "ok"
