"""Shared dependencies for the FastAPI application.

Provides:
- Templates engine (Jinja2Blocks) and the ``format_datetime`` / ``format_currency``
  presentation filters
- ``DB_PATH`` constant pointing at the DuckDB web cache
- ``get_db()`` FastAPI dependency that returns the shared read-only DuckDB
  connection from ``app.state.db_conn``, reconnecting under
  ``app.state.db_lock`` if the current connection is dead or missing
- ``get_data_service()`` factory that builds a ``DataService`` bound to the
  injected connection
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import duckdb
from fastapi import Depends, Request
from jinja2_fragments.fastapi import Jinja2Blocks

from api.exceptions import ModelUnavailableError
from utils import get_logger

from .services import DataService

logger = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "web_cache.duckdb"

templates = Jinja2Blocks(directory=str(TEMPLATES_DIR))


def format_datetime(value: str | datetime | None) -> str:
    """Format an ISO timestamp or datetime to human-readable string."""
    if value is None:
        return "Unknown"
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return str(value)
    return value.strftime("%b %d, %Y %I:%M %p")


def format_currency(value: float | int | None) -> str:
    """Format a number as a whole-dollar, thousands-separated string.

    Presentation-only helper for the betting dashboard's dollar KPIs (Net
    Profit, Final Bankroll, Max Drawdown). Jinja's built-in ``format`` filter
    is printf-style and rejects the ``,`` grouping flag (``"%,.0f"`` raises
    ``ValueError``), so grouping is done here via Python's ``format`` builtin.
    Returns ``"$0"`` for ``None`` / non-numeric input rather than raising.
    """
    if value is None:
        return "$0"
    try:
        return f"${format(float(value), ',.0f')}"
    except (ValueError, TypeError):
        return "$0"


templates.env.filters["format_datetime"] = format_datetime
templates.env.filters["format_currency"] = format_currency


def _reconnect_under_lock(request: Request) -> duckdb.DuckDBPyConnection:
    """Atomically replace ``app.state.db_conn`` with a fresh read-only connection.

    Called from :func:`get_db` when the existing connection is dead or missing.
    Guarded by ``app.state.db_lock`` so concurrent requests do not race.
    """
    lock = request.app.state.db_lock
    with lock:
        # Double-check inside the lock: another request may have already
        # reconnected while we were waiting for the lock.
        current = getattr(request.app.state, "db_conn", None)
        if current is not None:
            try:
                current.execute("SELECT 1")
                return current  # another request reconnected, reuse it
            except duckdb.Error:
                pass  # fall through and replace

        if not DB_PATH.exists():
            request.app.state.db_conn = None
            raise ModelUnavailableError("DuckDB cache not available (file missing)")

        try:
            new_conn = duckdb.connect(str(DB_PATH), read_only=True)
        except (duckdb.Error, OSError) as exc:
            request.app.state.db_conn = None
            logger.error(
                "Failed to reconnect to DuckDB cache",
                path=str(DB_PATH),
                error=str(exc),
            )
            raise ModelUnavailableError(
                "DuckDB cache not available (reconnect failed)"
            ) from exc

        request.app.state.db_conn = new_conn
        logger.warning("DuckDB connection replaced under lock")
        return new_conn


def get_db(request: Request) -> duckdb.DuckDBPyConnection:
    """Return the shared DuckDB read-only connection, reconnecting if stale.

    Owned by the app (not DataService). If the current connection is None or
    fails a lightweight ``SELECT 1`` health check, atomically replaces
    ``app.state.db_conn`` under ``app.state.db_lock``.
    """
    conn = getattr(request.app.state, "db_conn", None)
    if conn is None:
        return _reconnect_under_lock(request)

    # Lightweight health check -- SELECT 1 is O(1) on DuckDB read-only.
    try:
        conn.execute("SELECT 1")
        return conn
    except duckdb.Error:
        logger.warning("DuckDB connection stale, reconnecting under lock")
        return _reconnect_under_lock(request)


def get_data_service(
    conn: duckdb.DuckDBPyConnection = Depends(get_db),
) -> DataService:
    """Create a ``DataService`` instance bound to the shared connection."""
    return DataService(conn=conn)
