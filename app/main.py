"""FastAPI application entry point.

Run locally with:

    uvicorn app.main:app --host 127.0.0.1 --port 8000

Startup does not migrate, ingest, or open a database connection.
The embedding model and FAISS indexes are not loaded at import time.
The RankNet ranker is loaded lazily on the first LTR request.
The CF model is loaded lazily on the first method=cf request.
Recommendations load stored product vectors from the existing FAISS runtime.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.demo import router as demo_router
from app.api.interactions import router as interactions_router
from app.api.products import router as products_router
from app.api.recommendations import router as recommendations_router
from app.api.search import router as search_router
from app.api.system import router as system_router
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.core.seeds import set_random_seed

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Configure logging and the random seed. Does not ingest data."""

    settings = get_settings()
    configure_logging(settings.log_level)
    set_random_seed(settings.random_seed)
    logger.info(
        "Starting %s v%s env=%s (search + optional personalization + content/CF/hybrid recs)",
        settings.app_name,
        __version__,
        settings.app_env,
    )
    yield


def create_app() -> FastAPI:
    """Build the FastAPI application with system and catalog routes."""

    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Product catalog, search (keyword/semantic/hybrid/LTR), "
            "optional bounded personalized search, content/popularity "
            "recommendations, optional BPR-MF collaborative filtering, "
            "hybrid-rec-v1 fusion, and a local demonstration UI at /demo."
        ),
        lifespan=lifespan,
    )
    application.include_router(system_router)
    application.include_router(demo_router)
    application.include_router(products_router)
    application.include_router(interactions_router)
    application.include_router(search_router)
    application.include_router(recommendations_router)
    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.is_dir():
        application.mount("/static", StaticFiles(directory=static_dir), name="static")
    return application


app = create_app()
