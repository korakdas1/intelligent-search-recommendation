"""System endpoint response models."""

from typing import Literal

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness: the HTTP process is running. No dependency checks."""

    status: Literal["ok"]
    service: str
    version: str


class ReadyResponse(BaseModel):
    """Readiness for currently implemented capabilities.

    Checks include the application process, PostgreSQL ``SELECT 1``,
    semantic artifact files, an optional LTR ranker probe, and an optional
    CF model probe. Ranker/CF are skipped unless required or a valid
    artifact is present. Health does not load models.
    """

    status: Literal["ready", "not_ready"]
    checks: dict[str, str] = Field(default_factory=dict)
