"""FastAPI application for the NFL Prediction System.

Serves both HTML pages (via Jinja2Blocks templates) and JSON API endpoints.
All data comes from the DuckDB web cache -- no model classes are imported
(UIAP-01 compliance).

Architecture:
    - Single monolith app serving HTML + JSON (D-02)
    - DuckDB cache as sole data source (D-03)
    - Jinja2Blocks for HTMX fragment support
    - Static files served from web/static/
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from utils import get_logger

from .dependencies import DB_PATH
from .routes.api_export import router as export_router
from .routes.fragments import router as fragments_router
from .routes.health import router as health_router
from .routes.pages import router as pages_router

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Static files directory (resolved relative to project root)
# ---------------------------------------------------------------------------

STATIC_DIR = Path(__file__).resolve().parent.parent / "web" / "static"


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: check cache on startup, cleanup on shutdown."""
    if DB_PATH.exists():
        logger.info("DuckDB web cache found", path=str(DB_PATH))
    else:
        logger.warning(
            "DuckDB web cache not found -- run 'make build-cache' to populate",
            path=str(DB_PATH),
        )
    yield
    logger.info("NFL Prediction API shutting down")


# ---------------------------------------------------------------------------
# App creation
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NFL Prediction System",
    description="Pre-game NFL predictions served from DuckDB cache",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include routers -- pages_router last so its catch-all "/" doesn't shadow others
app.include_router(health_router)
app.include_router(export_router)
app.include_router(fragments_router)
app.include_router(pages_router)
