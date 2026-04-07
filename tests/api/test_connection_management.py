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


def _set_db_path(new_path: Path) -> tuple[Path, Path]:
    """Point both ``api.dependencies.DB_PATH`` and ``api.routes.health.DB_PATH``
    at *new_path*. Returns the originals so the caller can restore them.
    """
    original_deps = deps.DB_PATH
    original_health = health_module.DB_PATH
    deps.DB_PATH = new_path
    health_module.DB_PATH = new_path
    return original_deps, original_health


def _restore_db_path(original_deps: Path, original_health: Path) -> None:
    """Restore the originals captured by :func:`_set_db_path`."""
    deps.DB_PATH = original_deps
    health_module.DB_PATH = original_health


def _make_request_with_state(
    db_conn: duckdb.DuckDBPyConnection | None,
    db_lock: threading.RLock | None = None,
) -> MagicMock:
    """Build a MagicMock Request whose app.state exposes db_conn and db_lock."""
    req = MagicMock()
    req.app.state.db_conn = db_conn
    req.app.state.db_lock = db_lock or threading.RLock()
    return req


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


def test_lifespan_creates_connection(test_db: Path) -> None:
    """When the DB file exists, lifespan opens a connection into app.state."""
    original_deps, original_health = _set_db_path(test_db)
    # Clear any leftover dependency overrides from other tests so the real
    # lifespan path runs.
    app.dependency_overrides.clear()
    try:
        with TestClient(app) as client:
            assert client.app.state.db_conn is not None
            # The lock must exist after lifespan ran.
            assert hasattr(client.app.state.db_lock, "acquire")
            # And the connection must answer SELECT 1.
            client.app.state.db_conn.execute("SELECT 1").fetchone()
    finally:
        _restore_db_path(original_deps, original_health)


def test_lifespan_handles_missing_db(tmp_path: Path) -> None:
    """When the DB file is missing, lifespan sets db_conn to None without crashing.

    The /health endpoint must still respond (200 with cache_ready=false or 503).
    """
    missing = tmp_path / "nonexistent.duckdb"
    original_deps, original_health = _set_db_path(missing)
    app.dependency_overrides.clear()
    try:
        with TestClient(app) as client:
            assert client.app.state.db_conn is None
            response = client.get("/health")
            # Health must not 500 even with no DB.
            assert response.status_code in (200, 503), response.text
    finally:
        _restore_db_path(original_deps, original_health)


def test_lifespan_handles_corrupt_db(tmp_path: Path) -> None:
    """A corrupt/unreadable DB file must not crash startup.

    Either the connect succeeds (DuckDB tolerates some garbage and treats it as
    an empty store) OR ``app.state.db_conn`` is None. Either way the app must
    start and ``/health`` must respond without 500.
    """
    corrupt = tmp_path / "corrupt.duckdb"
    corrupt.write_bytes(b"not a real duckdb file" * 100)
    original_deps, original_health = _set_db_path(corrupt)
    app.dependency_overrides.clear()
    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            response = client.get("/health")
            assert response.status_code in (200, 503), response.text
    finally:
        _restore_db_path(original_deps, original_health)


# ---------------------------------------------------------------------------
# get_db dependency
# ---------------------------------------------------------------------------


def test_get_db_returns_app_state_conn_when_healthy(test_db: Path) -> None:
    """get_db returns the existing conn if SELECT 1 succeeds."""
    conn = duckdb.connect(str(test_db), read_only=True)
    try:
        req = _make_request_with_state(conn)
        result = get_db(req)
        assert result is conn
    finally:
        conn.close()


def test_get_db_raises_model_unavailable_when_conn_none_and_file_missing(
    tmp_path: Path,
) -> None:
    """When app.state.db_conn is None and the DB file does not exist, get_db raises."""
    missing = tmp_path / "no.duckdb"
    original_deps, original_health = _set_db_path(missing)
    try:
        req = _make_request_with_state(None)
        with pytest.raises(ModelUnavailableError):
            get_db(req)
    finally:
        _restore_db_path(original_deps, original_health)


def test_get_db_reconnects_when_connection_is_dead(test_db: Path) -> None:
    """A closed/dead connection triggers reconnect under lock."""
    original_deps, original_health = _set_db_path(test_db)
    try:
        dead = duckdb.connect(str(test_db), read_only=True)
        dead.close()  # deliberately kill the connection

        req = _make_request_with_state(dead)
        new_conn = get_db(req)
        assert new_conn is not dead
        # Confirm the new connection is alive
        new_conn.execute("SELECT 1").fetchone()
        # Confirm app.state was updated
        assert req.app.state.db_conn is new_conn
        new_conn.close()
    finally:
        _restore_db_path(original_deps, original_health)


def test_get_db_double_check_inside_lock_reuses_existing_conn(test_db: Path) -> None:
    """If another request reconnected while we waited on the lock, reuse it.

    Drives the double-check pattern: we hand ``_reconnect_under_lock`` a
    request whose ``app.state.db_conn`` is already a healthy connection.
    The function must return that same connection without opening a new one.
    """
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
