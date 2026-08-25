"""Liveness and readiness endpoints.

``GET /health`` answers: is this process running?
``GET /ready`` answers: are implemented dependencies ready?

Health does not check the database or semantic artifacts.
Readiness pings PostgreSQL and checks semantic artifact files (not the model).
"""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app import __version__
from app.core.config import get_settings
from app.db.ping import ping_database
from app.ranking.probe import probe_ltr_artifacts
from app.recommendations.cf_probe import probe_cf_artifacts
from app.schemas.system import HealthResponse, ReadyResponse
from app.search.runtime import probe_semantic_artifacts

router = APIRouter(tags=["system"])


@router.get("/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return process liveness. No expensive dependency checks."""

    settings = get_settings()
    return HealthResponse(
        status="ok",
        service=settings.service_id,
        version=__version__,
    )


@router.get("/ready", response_model=ReadyResponse, responses={503: {"model": ReadyResponse}})
def get_ready() -> ReadyResponse | JSONResponse:
    """Return readiness for PostgreSQL and semantic index artifacts."""

    database_ok = ping_database()
    semantic_state = probe_semantic_artifacts()
    ltr_state = probe_ltr_artifacts()
    cf_state = probe_cf_artifacts()
    checks = {
        "application": "ok",
        "database": "ok" if database_ok else "unavailable",
        "semantic_index": semantic_state,
        "ltr_ranker": ltr_state,
        "cf_model": cf_state,
    }
    ready = (
        database_ok
        and semantic_state in {"ok", "skipped"}
        and ltr_state in {"ok", "skipped"}
        and cf_state in {"ok", "skipped"}
    )
    if ready:
        return ReadyResponse(status="ready", checks=checks)
    return JSONResponse(
        status_code=503,
        content={"status": "not_ready", "checks": checks},
    )
