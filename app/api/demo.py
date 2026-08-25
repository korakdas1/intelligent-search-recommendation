"""Local demonstration page. Serves static HTML; it does not implement retrieval."""

from pathlib import Path

from fastapi import APIRouter
from fastapi.responses import FileResponse, HTMLResponse

router = APIRouter(tags=["demo"])

_DEMO_PAGE = Path(__file__).resolve().parent.parent / "static" / "demo" / "index.html"


@router.get("/demo", response_class=HTMLResponse)
def get_demo() -> FileResponse:
    """Return the local demo interface."""

    return FileResponse(_DEMO_PAGE, media_type="text/html")
