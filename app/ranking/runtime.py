"""Lazy RankNet loading for optional LTR reranking. Not imported by app.main directly."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.core.config import Settings, get_settings
from app.ranking.artifacts import RankerArtifactError, load_ranker_bundle
from app.ranking.constants import MODEL_VERSION
from app.ranking.preprocessing import FeatureScaler
from app.ranking.probe import ranker_artifact_path
from app.ranking.ranker import RankNetMLP
from app.search.exceptions import RankerUnavailableError

logger = logging.getLogger(__name__)

_runtime: LtrRuntime | None = None


@dataclass
class LtrRuntime:
    model_version: str
    feature_version: str
    model: RankNetMLP
    scaler: FeatureScaler
    config: dict[str, Any]
    directory: Path


def get_ltr_runtime() -> LtrRuntime:
    global _runtime
    if _runtime is None:
        _runtime = load_ltr_runtime()
    return _runtime


def set_ltr_runtime(runtime: LtrRuntime | None) -> None:
    global _runtime
    _runtime = runtime


def reset_ltr_runtime() -> None:
    set_ltr_runtime(None)


def load_ltr_runtime(settings: Settings | None = None) -> LtrRuntime:
    settings = settings or get_settings()
    directory = ranker_artifact_path(settings)
    try:
        model, scaler, config = load_ranker_bundle(directory)
    except RankerArtifactError as exc:
        raise RankerUnavailableError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise RankerUnavailableError(f"ranker artifact missing at {directory}") from exc
    version = str(config.get("model_version", settings.ltr_ranker_version or MODEL_VERSION))
    feature_version = str(config.get("feature_version", scaler.feature_version))
    logger.info("Loaded LTR ranker %s from %s", version, directory)
    return LtrRuntime(
        model_version=version,
        feature_version=feature_version,
        model=model,
        scaler=scaler,
        config=config,
        directory=directory,
    )


def require_ltr_runtime() -> LtrRuntime:
    try:
        return get_ltr_runtime()
    except RankerUnavailableError:
        raise
    except Exception as exc:
        raise RankerUnavailableError(str(exc)) from exc
