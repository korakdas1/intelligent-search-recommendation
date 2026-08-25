"""Lazy BPR-MF loading for method=cf. Not imported by app.main."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from app.core.config import Settings, get_settings
from app.recommendations.cf_artifacts import CfArtifactError, load_cf_bundle
from app.recommendations.cf_constants import MODEL_VERSION
from app.recommendations.cf_model import BPRMatrixFactorization
from app.recommendations.cf_probe import cf_artifact_path
from app.recommendations.exceptions import CfUnavailableError

logger = logging.getLogger(__name__)

_runtime: CfRuntime | None = None


@dataclass
class CfRuntime:
    model_version: str
    model: BPRMatrixFactorization
    user_ids: list[str]
    product_ids: list[str]
    user_to_index: dict[str, int]
    product_to_index: dict[str, int]
    config: dict[str, Any]
    manifest: dict[str, Any]
    directory: Path
    device: torch.device


def get_cf_runtime() -> CfRuntime:
    global _runtime
    if _runtime is None:
        _runtime = load_cf_runtime()
    return _runtime


def set_cf_runtime(runtime: CfRuntime | None) -> None:
    global _runtime
    _runtime = runtime


def reset_cf_runtime() -> None:
    set_cf_runtime(None)


def load_cf_runtime(settings: Settings | None = None) -> CfRuntime:
    settings = settings or get_settings()
    directory = cf_artifact_path(settings)
    try:
        model, user_ids, product_ids, config, manifest = load_cf_bundle(directory)
    except CfArtifactError as exc:
        raise CfUnavailableError(str(exc)) from exc
    except FileNotFoundError as exc:
        raise CfUnavailableError(f"CF artifact missing at {directory}") from exc
    model.eval()
    version = str(config.get("model_version", settings.cf_model_version or MODEL_VERSION))
    logger.info("Loaded CF model %s from %s", version, directory)
    return CfRuntime(
        model_version=version,
        model=model,
        user_ids=user_ids,
        product_ids=product_ids,
        user_to_index={user_id: index for index, user_id in enumerate(user_ids)},
        product_to_index={product_id: index for index, product_id in enumerate(product_ids)},
        config=config,
        manifest=manifest,
        directory=directory,
        device=torch.device("cpu"),
    )


def require_cf_runtime() -> CfRuntime:
    try:
        return get_cf_runtime()
    except CfUnavailableError:
        raise
    except Exception as exc:
        raise CfUnavailableError(str(exc)) from exc
