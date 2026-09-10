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
        (b) ``app.state.db_lock`` (threading.RLock) below for reconnects.

    WHAT ``--workers 1`` DOES *NOT* BOUND (corrected here; the earlier version of
    this docstring cited "the single async event loop" as a bound, and code in
    ``api.dependencies`` cited this docstring in turn). It bounds the number of
    PROCESSES. It does not serialise requests, and there is no single-threaded
    event-loop bound to lean on either: every route in ``api/routes/pages.py`` is
    a sync ``def``, so Starlette dispatches all of them into the AnyIO worker
    threadpool -- 40 threads by default -- rather than running them on the loop.
    Requests therefore execute CONCURRENTLY on distinct threads, and the shared
    connection is read outside ``app.state.db_lock`` (that lock covers only the
    reconnect). The consequence is recorded at its true size on the
    ``current.close()`` in ``api.dependencies._reconnect_under_lock``; do not
    re-derive a narrower bound from the worker count.

    If ``--workers N`` with N > 1 is ever enabled, the shared connection and
    the cache must be revisited (each worker would hold its own copy and any
    cross-worker coordination would require out-of-process state). The
    cache-file identity check introduced in plan 31-20 is likewise per-process,
    so under ``--workers N`` each worker detects a cache swap independently and
    there is still no cross-worker coordination.
    """
    app.state.db_lock = threading.RLock()

    if deps.DB_PATH.exists():
        try:
            # Record WHICH file this connection was opened against (plan 31-20,
            # G-31-123a). ``api.dependencies.get_db`` compares this per request so
            # a ``populate_cache`` swap is picked up without a restart. Computed
            # through the shared public helper -- a second definition here would
            # let the opener and the detector disagree.
            #
            # READ BEFORE THE CONNECT, deliberately, and this is the widest-exposure
            # instance of that ordering: the connect here is a COLD open of a ~6 MB
            # database at process boot, which is exactly the window a Friday
            # orchestrator population overlapping a service restart lands in. A stat
            # taken after the connect would record the identity of the POST-swap file
            # against a handle holding the PRE-swap one, and ``_cache_file_changed``
            # would then compare equal forever. Reading first records a STALE identity
            # instead, which the first request corrects with one spurious reconnect.
            pre_connect_identity = deps.cache_identity(deps.DB_PATH)
            conn = duckdb.connect(str(deps.DB_PATH), read_only=True)
            app.state.db_conn = conn
            # The fresh stat is the fallback for ONE case, the same one
            # ``api.dependencies._reconnect_under_lock`` falls back for: the
            # pre-connect read was ``None`` because it landed in the writer's
            # unlink-to-rename window, and the file then appeared before the connect
            # succeeded. There is no earlier observation to reuse there, and
            # recording ``None`` would switch the detector off for the whole process
            # lifetime.
            app.state.db_identity = (
                pre_connect_identity
                if pre_connect_identity is not None
                else deps.cache_identity(deps.DB_PATH)
            )
            logger.info("DuckDB connection established", path=str(deps.DB_PATH))
        except (duckdb.Error, OSError) as exc:
            app.state.db_conn = None
            app.state.db_identity = None
            logger.error(
                "Failed to open DuckDB cache at startup",
                path=str(deps.DB_PATH),
                error=str(exc),
            )
    else:
        app.state.db_conn = None
        app.state.db_identity = None
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
