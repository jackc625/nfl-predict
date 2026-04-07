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

import threading
from contextlib import asynccontextmanager
from pathlib import Path

import duckdb
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from utils import get_logger

from . import dependencies as deps
from .routes.api_export import router as export_router
from .routes.fragments import router as fragments_router
from .routes.health import router as health_router
from .routes.pages import router as pages_router
from .services import clear_cache

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
    """Application lifespan: open shared DuckDB connection, cleanup on shutdown.

    DEPLOYMENT ASSUMPTION: this app runs under single-worker uvicorn
    (``uvicorn ... --workers 1``). Thread-safety of ``app.state.db_conn`` and
    the module-level TTLCache introduced in plan 15-02 is bounded by:
        (a) Python's GIL,
        (b) the single async event loop,
        (c) ``app.state.db_lock`` (threading.RLock) below for reconnects.
    If ``--workers N`` with N > 1 is ever enabled, the shared connection and
    the cache must be revisited (each worker would hold its own copy and any
    cross-worker coordination would require out-of-process state).
    """
    app.state.db_lock = threading.RLock()

    if deps.DB_PATH.exists():
        try:
            conn = duckdb.connect(str(deps.DB_PATH), read_only=True)
            app.state.db_conn = conn
            logger.info("DuckDB connection established", path=str(deps.DB_PATH))
        except (duckdb.Error, OSError) as exc:
            app.state.db_conn = None
            logger.error(
                "Failed to open DuckDB cache at startup",
                path=str(deps.DB_PATH),
                error=str(exc),
            )
    else:
        app.state.db_conn = None
        logger.warning(
            "DuckDB web cache not found -- run 'make build-cache' to populate",
            path=str(deps.DB_PATH),
        )

    # Plan 15-02: invalidate any TTLCache entries surviving a previous run so
    # a redeploy never serves stale data beyond the startup boundary.
    clear_cache()

    yield

    conn_to_close = getattr(app.state, "db_conn", None)
    if conn_to_close is not None:
        try:
            conn_to_close.close()
            logger.info("DuckDB connection closed")
        except (duckdb.Error, OSError) as exc:
            logger.warning("Error closing DuckDB connection", error=str(exc))


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
