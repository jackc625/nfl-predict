"""Shared dependencies for the FastAPI application.

Provides:
- Templates engine (Jinja2Blocks) and the ``format_datetime`` / ``format_currency``
  presentation filters
- ``DB_PATH`` constant pointing at the DuckDB web cache
- ``cache_identity()`` -- the on-disk identity of the cache file, so a reader can
  tell that the file underneath its open handle has been REPLACED
- ``get_db()`` FastAPI dependency that returns the shared read-only DuckDB
  connection from ``app.state.db_conn``, reconnecting under
  ``app.state.db_lock`` if the current connection is dead, missing, or pointed at
  a file that has since been swapped out from under it
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

from .services import DataService, clear_cache

logger = get_logger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent.parent / "web" / "templates"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "web_cache.duckdb"

templates = Jinja2Blocks(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Cache-file identity (plan 31-20, G-31-123a)
# ---------------------------------------------------------------------------

#: The stat fields that together identify WHICH file currently sits at a path.
CacheIdentity = tuple[int, int, int, int]


def cache_identity(path: Path) -> CacheIdentity | None:
    """Return the identity of the file currently at *path*, or None if there is none.

    ``scripts/populate_cache.py`` does not mutate the cache in place -- it builds a
    temp DB and then does ``db_path.unlink()`` followed by ``tmp_path.rename(db_path)``
    (``api.cache.populate_cache``). The PATH is re-pointed at a NEW file, and a
    read-only DuckDB handle opened against the old one keeps serving the deleted
    file's contents indefinitely without raising. This identity is how a reader
    notices that happened; there is nothing in the connection itself to ask.

    The tuple is ``(st_dev, st_ino, st_mtime_ns, st_size)``, all of which change when
    a file is replaced at the same path:

    * ``st_ino`` is the inode on POSIX and the NTFS file index on Windows. It is the
      primary discriminator, because a replaced file is a different file object even
      when its path and contents are identical.
    * ``st_mtime_ns`` and ``st_size`` are carried alongside it deliberately, so the
      tuple still discriminates on any filesystem that reports ``st_ino`` as zero or
      that recycles inode numbers -- neither is guaranteed by Python's ``os.stat``
      contract on every platform.

    Returns ``None`` -- and never raises -- when the path cannot be stat'ed, so a
    caller can distinguish "no file to compare against" from a real identity. That
    distinction is load-bearing: the writer's swap has a window between the unlink and
    the rename in which the path genuinely does not exist.

    PUBLIC on purpose. ``api.main.lifespan`` is its second caller (it opens the
    process-lifetime connection), and a private second definition there would be the
    duplicated-definition drift this repository has been bitten by before -- an opener
    that records an identity computed a different way is an opener the detector
    silently disagrees with.
    """
    try:
        stat_result = path.stat()
    except OSError:
        return None
    return (
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_mtime_ns,
        stat_result.st_size,
    )


def _cache_file_changed(request: Request) -> bool:
    """Return whether the cache file has been REPLACED since the connection was opened.

    Compares the identity recorded on ``app.state.db_identity`` when the current
    connection was opened against the identity of whatever is at :data:`DB_PATH` now.
    ``True`` only when BOTH identities exist and differ. Both absences return
    ``False`` -- meaning "do not reconnect" -- and each has its own reason:

    * **No RECORDED identity.** The connection on ``app.state`` was not opened by this
      module, so there is nothing to compare it against. Adopting the current on-disk
      identity here would be WORSE than not comparing: it would record the identity of
      a file this connection may not be pointing at, and the detector would then be
      permanently blind to that exact staleness. Both openers (this module's
      :func:`_reconnect_under_lock` and ``api.main.lifespan``) record it, so this is a
      defensive branch rather than a reachable one in production.
    * **No CURRENT on-disk identity.** The path does not exist at this instant, which
      is precisely the window between the writer's ``unlink()`` and its ``rename()``.
      Tearing down a working connection there would turn a routine swap into an error
      page. The already-open handle still reads fine, and the first request after the
      rename lands sees the new identity and reconnects then.
    """
    recorded = getattr(request.app.state, "db_identity", None)
    if recorded is None:
        return False
    current = cache_identity(DB_PATH)
    if current is None:
        return False
    return recorded != current


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

    Called from :func:`get_db` when the existing connection is missing, dead, or
    pointed at a cache file that has since been replaced. Guarded by
    ``app.state.db_lock`` so concurrent requests do not race.
    """
    lock = request.app.state.db_lock
    with lock:
        # Read the on-disk identity ONCE for the whole locked section, so the
        # double-check below and the identity recorded after the connect are
        # reasoned about against one observation rather than two.
        on_disk_identity = cache_identity(DB_PATH)

        # Double-check inside the lock: another request may have already
        # reconnected while we were waiting for the lock. It must compare
        # IDENTITY as well as liveness -- a liveness-only double-check would hand
        # back the very stale connection this call was made to replace, because
        # ``SELECT 1`` passes on a handle whose file has been unlinked.
        current = getattr(request.app.state, "db_conn", None)
        recorded_identity = getattr(request.app.state, "db_identity", None)
        if (
            current is not None
            and recorded_identity is not None
            and on_disk_identity is not None
            and recorded_identity == on_disk_identity
        ):
            try:
                current.execute("SELECT 1")
                return current  # another request reconnected, reuse it
            except duckdb.Error:
                pass  # fall through and replace

        if not DB_PATH.exists():
            request.app.state.db_conn = None
            # Clear the identity alongside the connection, so a later successful
            # reconnect is not compared against a dead file's identity.
            request.app.state.db_identity = None
            raise ModelUnavailableError("DuckDB cache not available (file missing)")

        if current is not None:
            # CLOSE BEFORE CONNECT, and the ordering is load-bearing rather than
            # tidy: DuckDB caches the database instance BY PATH within a process,
            # so a connect issued while the old connection is still alive returns
            # the SAME stale instance. That was proven live in the diagnosis
            # (.planning/debug/bets-stale-cache-recovery-noop.md, E1(D)/E4 ALT-3):
            # forcing a reconnect with the old object alive still served pre-swap
            # data, and closing it first fixed it immediately.
            #
            # ACCEPTED COST (T-31-104): a request already executing on the old
            # handle in another thread can see a closed connection and fail. That
            # is a loud failure, at most once per population run, and the
            # alternative it replaces is SILENT PERMANENT staleness. The app is
            # documented single-worker (see ``api.main.lifespan``), which bounds
            # the exposure.
            try:
                current.close()
            except (duckdb.Error, OSError) as exc:
                logger.warning(
                    "Error closing the superseded DuckDB connection",
                    path=str(DB_PATH),
                    error=str(exc),
                )
            request.app.state.db_conn = None

        try:
            new_conn = duckdb.connect(str(DB_PATH), read_only=True)
        except (duckdb.Error, OSError) as exc:
            request.app.state.db_conn = None
            request.app.state.db_identity = None
            logger.error(
                "Failed to reconnect to DuckDB cache",
                path=str(DB_PATH),
                error=str(exc),
            )
            raise ModelUnavailableError(
                "DuckDB cache not available (reconnect failed)"
            ) from exc

        request.app.state.db_conn = new_conn
        # Captured AFTER the connect, deliberately. A swap landing between the two
        # makes the RECORDED identity newer than the file actually opened, which
        # the next request detects and corrects; capturing first could record an
        # identity for a file that was never opened, which nothing would correct.
        request.app.state.db_identity = cache_identity(DB_PATH)
        # The same invalidation ``api.main.lifespan`` performs at startup, for the
        # same reason: a value memoized from the PREVIOUS file must not survive
        # the swap boundary. One invalidation point, not two.
        clear_cache()
        logger.warning("DuckDB connection replaced under lock", path=str(DB_PATH))
        return new_conn


def get_db(request: Request) -> duckdb.DuckDBPyConnection:
    """Return the shared DuckDB read-only connection, reconnecting if stale.

    Owned by the app (not DataService). Replaces ``app.state.db_conn`` under
    ``app.state.db_lock`` when the connection is missing, when the cache file at
    :data:`DB_PATH` has been REPLACED since the connection was opened, or when the
    connection fails a lightweight ``SELECT 1`` health check.
    """
    conn = getattr(request.app.state, "db_conn", None)
    if conn is None:
        return _reconnect_under_lock(request)

    # THE IDENTITY COMPARISON COMES BEFORE THE ``SELECT 1`` PROBE, because the
    # probe CANNOT fail in the case being detected: a read-only handle whose file
    # has been unlinked answers ``SELECT 1`` forever. A probe-first order would
    # return the stale connection here and never reach the comparison in the one
    # situation the comparison exists for.
    if _cache_file_changed(request):
        logger.warning(
            "DuckDB cache file replaced on disk, reconnecting under lock",
            path=str(DB_PATH),
        )
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
