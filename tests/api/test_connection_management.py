"""Tests for DuckDB connection management (OPS-05).

Covers:
- Lifespan creates a connection from a valid DB file
- Lifespan handles a missing DB file gracefully (no crash, health responds)
- Lifespan handles a corrupt/unreadable DB file gracefully (no crash on startup)
- get_db returns the app.state connection under normal operation
- get_db raises ModelUnavailableError when conn is None and the file is missing
- get_db reconnects under lock when the current connection is dead
- DataService uses the injected connection (no _connect, no _execute_with_reconnect,
  no _db_path)
"""

from __future__ import annotations

import threading
from pathlib import Path
from unittest.mock import MagicMock

import duckdb
import pytest
from fastapi.testclient import TestClient

import api.dependencies as deps
import api.routes.health as health_module
from api.dependencies import get_db
from api.exceptions import ModelUnavailableError
from api.main import app
from api.services import DataService


def _point_db_path_at(monkeypatch: pytest.MonkeyPatch, new_path: Path) -> None:
    """Point both ``api.dependencies.DB_PATH`` and ``api.routes.health.DB_PATH`` at *new_path*.

    BOTH globals, because ``api/routes/health.py`` binds the name at import and so holds its own
    reference; patching one would leave the two readers measuring different files.

    VIA ``monkeypatch``, NOT RAW ASSIGNMENT (WR-07). The previous helper assigned the module
    globals directly and returned the originals for the caller to restore in a ``finally``, which
    made restoration depend on every caller placing the call immediately before ``try:`` with
    nothing in between. Plan 31-20 added two callers that put a ``duckdb.connect`` between the
    mutation and the ``try`` -- so a connect that raised (a locked file on Windows, a leftover
    handle from a sibling test, a corrupt fixture) skipped the restore and left BOTH globals
    pointed at a ``tmp_path`` fixture for the rest of the pytest session. Every downstream test
    that reads either one -- ``/health``'s file-existence check, ``cache_file_changed``'s
    ``cache_identity(DB_PATH)`` -- then measures the wrong file, and the failures surface far from
    their cause. ``monkeypatch`` undoes itself at teardown no matter where the test died, which
    removes the whole class rather than the two instances of it.
    """
    monkeypatch.setattr(deps, "DB_PATH", new_path)
    monkeypatch.setattr(health_module, "DB_PATH", new_path)


_IDENTITY_FROM_DB_PATH = object()


def _make_request_with_state(
    db_conn: duckdb.DuckDBPyConnection | None,
    db_lock: threading.RLock | None = None,
    *,
    db_identity: object = _IDENTITY_FROM_DB_PATH,
) -> MagicMock:
    """Build a MagicMock Request whose app.state exposes the three connection fields.

    ``db_identity`` MUST be set explicitly here rather than left to the MagicMock
    (plan 31-20). An unset attribute on a MagicMock auto-creates a truthy child
    mock, which ``api.dependencies.cache_file_changed`` would read as a RECORDED
    identity that mismatches every real on-disk identity -- making every healthy
    connection look swapped and every double-check look stale. That is a property
    of the mock, not of the code under test.

    It defaults to the identity of the file currently at ``deps.DB_PATH``, which
    is exactly what both production openers record. Pass ``db_identity=None`` to
    model a connection opened OUTSIDE ``api.dependencies`` (no recorded identity),
    or a literal tuple to model a stale one.
    """
    req = MagicMock()
    req.app.state.db_conn = db_conn
    req.app.state.db_lock = db_lock or threading.RLock()
    req.app.state.db_identity = (
        deps.cache_identity(deps.DB_PATH)
        if db_identity is _IDENTITY_FROM_DB_PATH
        else db_identity
    )
    return req


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def test_lifespan_creates_connection(
    test_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the DB file exists, lifespan opens a connection into app.state."""
    _point_db_path_at(monkeypatch, test_db)
    # Clear any leftover dependency overrides from other tests so the real
    # lifespan path runs.
    app.dependency_overrides.clear()
    with TestClient(app):
        assert app.state.db_conn is not None
        # The lock must exist after lifespan ran.
        assert hasattr(app.state.db_lock, "acquire")
        # And the connection must answer SELECT 1.
        app.state.db_conn.execute("SELECT 1").fetchone()


def test_lifespan_handles_missing_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the DB file is missing, lifespan sets db_conn to None without crashing.

    The /health endpoint must still respond (200 with cache_ready=false or 503).
    """
    missing = tmp_path / "nonexistent.duckdb"
    _point_db_path_at(monkeypatch, missing)
    app.dependency_overrides.clear()
    with TestClient(app) as client:
        assert app.state.db_conn is None
        response = client.get("/health")
        # Health must not 500 even with no DB.
        assert response.status_code in (200, 503), response.text


def test_lifespan_handles_corrupt_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corrupt/unreadable DB file must not crash startup.

    Either the connect succeeds (DuckDB tolerates some garbage and treats it as
    an empty store) OR ``app.state.db_conn`` is None. Either way the app must
    start and ``/health`` must respond without 500.
    """
    corrupt = tmp_path / "corrupt.duckdb"
    corrupt.write_bytes(b"not a real duckdb file" * 100)
    _point_db_path_at(monkeypatch, corrupt)
    app.dependency_overrides.clear()
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health")
        assert response.status_code in (200, 503), response.text


# ---------------------------------------------------------------------------
# get_db dependency
# ---------------------------------------------------------------------------


def test_get_db_returns_app_state_conn_when_healthy(
    test_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """get_db returns the existing conn if the file is unchanged and SELECT 1 succeeds.

    ``deps.DB_PATH`` is pointed at ``test_db`` because the healthy path now also
    requires the RECORDED cache identity to match what is on disk (plan 31-20).
    Leaving DB_PATH on the real cache would compare the identity of one file
    against another and reconnect, which is the detector working correctly rather
    than a regression.
    """
    _point_db_path_at(monkeypatch, test_db)
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        req = _make_request_with_state(conn)
        result = get_db(req)
        assert result is conn
    finally:
        conn.close()


def test_get_db_raises_model_unavailable_when_conn_none_and_file_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When app.state.db_conn is None and the DB file does not exist, get_db raises."""
    missing = tmp_path / "no.duckdb"
    _point_db_path_at(monkeypatch, missing)
    req = _make_request_with_state(None)
    with pytest.raises(ModelUnavailableError):
        get_db(req)


def test_get_db_reconnects_when_connection_is_dead(
    test_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A closed/dead connection triggers reconnect under lock."""
    _point_db_path_at(monkeypatch, test_db)
    dead = duckdb.connect(str(test_db), read_only=True)
    dead.close()  # deliberately kill the connection

    req = _make_request_with_state(dead)
    new_conn = get_db(req)
    try:
        assert new_conn is not dead
        # Confirm the new connection is alive
        new_conn.execute("SELECT 1").fetchone()
        # Confirm app.state was updated
        assert req.app.state.db_conn is new_conn
    finally:
        new_conn.close()


def test_get_db_double_check_inside_lock_reuses_existing_conn(
    test_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If another request reconnected while we waited on the lock, reuse it.

    Drives the double-check pattern: we hand ``_reconnect_under_lock`` a
    request whose ``app.state.db_conn`` is already a healthy connection.
    The function must return that same connection without opening a new one.

    Reuse now requires the recorded identity to match the on-disk one as well as
    liveness (plan 31-20), so ``deps.DB_PATH`` is pointed at ``test_db``. That
    tightening is the point of the change: a liveness-only double-check would hand
    back the very stale connection the caller entered the lock to replace.
    """
    _point_db_path_at(monkeypatch, test_db)
    healthy = duckdb.connect(str(test_db), read_only=True)
    try:
        req = _make_request_with_state(healthy)
        # Even though we entered _reconnect_under_lock as if the conn was
        # dead, the double-check inside the lock should find it healthy.
        from api.dependencies import _reconnect_under_lock

        result = _reconnect_under_lock(req)
        assert result is healthy
    finally:
        healthy.close()


# ---------------------------------------------------------------------------
# DataService contract
# ---------------------------------------------------------------------------


def test_data_service_uses_injected_connection(test_db: Path) -> None:
    """DataService never calls self._connect or reconnects on its own."""
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        svc = DataService(conn=conn)
        # DataService must NOT expose connection-management attributes any more.
        assert not hasattr(svc, "_connect")
        assert not hasattr(svc, "_execute_with_reconnect")
        assert not hasattr(svc, "_db_path")
        # And a real query must succeed using the injected connection.
        result = svc.get_predictions()
        assert isinstance(result, list)
    finally:
        conn.close()
