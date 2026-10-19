"""The cache's data-format version, read by the server, and the notice a format clash turns into.

Phase 36, D-19 and SPEC R9. The server reads the cache's ``data_format_version`` when it starts
and again whenever the swap detector reconnects, and keeps the verdict on ``app.state.data_format``.
An absent or unopenable cache makes no claim (None), and leaving the lifespan clears the claim, so
the module-level ``app`` never carries a stale verdict into a later test that runs without one.

Every test here enters the ``TestClient`` as a context manager, so the real lifespan runs, and
installs no ``get_db`` override, so the real swap detector decides. ``DB_PATH`` is patched in both
``api.dependencies`` and ``api.routes.health``, as ``tests/api/test_cache_swap_recovery.py`` does,
and every cache is a copy of the shared ``test_db`` fixture under ``tmp_path``: no test touches the
production cache.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import shutil
import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import duckdb
import pytest
from fastapi.testclient import TestClient

import api.dependencies as deps
import api.routes.health as health_module
from api.cache_schema import DATA_FORMAT_VERSION, DATA_FORMAT_VERSION_KEY
from api.main import app
from api.services import clear_cache

_MATCHING = str(DATA_FORMAT_VERSION)
_AHEAD = str(DATA_FORMAT_VERSION + 1)

# The app.state fields the lifespan and the reconnect write. Reset around every test, because
# ``api.main.app`` is a module-level singleton shared with the rest of ``tests/api``.
_STATE_FIELDS = ("db_conn", "db_identity", "data_format")


def _write_cache(source: Path, dest: Path, raw_version: str | None) -> Path:
    """Copy the *source* cache to *dest* and stamp *raw_version* (None removes the stamp).

    The builder connection is closed before returning: DuckDB caches a database instance by path
    within a process, and a lingering handle would keep a pre-swap instance alive.
    """
    shutil.copyfile(source, dest)
    conn = duckdb.connect(str(dest))
    try:
        conn.execute("DELETE FROM cache_meta WHERE key = ?", [DATA_FORMAT_VERSION_KEY])
        if raw_version is not None:
            conn.execute(
                "INSERT INTO cache_meta VALUES (?, ?, ?)",
                [DATA_FORMAT_VERSION_KEY, raw_version, datetime.now(tz=UTC)],
            )
    finally:
        conn.close()
    return dest


def _swap_in(new_path: Path, db_path: Path) -> None:
    """Reproduce the writer's swap exactly: unlink, then rename."""
    db_path.unlink()
    new_path.rename(db_path)


def _reset_app_state() -> None:
    app.dependency_overrides.clear()
    for field in _STATE_FIELDS:
        if hasattr(app.state, field):
            delattr(app.state, field)
    clear_cache()


@pytest.fixture()
def served_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """The path the app serves its cache from; the test decides what (if anything) is there."""
    db_path = tmp_path / "web_cache.duckdb"
    monkeypatch.setattr(deps, "DB_PATH", db_path)
    monkeypatch.setattr(health_module, "DB_PATH", db_path)
    _reset_app_state()
    app.state.db_lock = threading.RLock()
    try:
        yield db_path
    finally:
        _reset_app_state()


# ---------------------------------------------------------------------------
# D-19: the version is read at startup and on every reconnect
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_version", "expected_matches"),
    [(_MATCHING, True), (_AHEAD, False), (None, None)],
    ids=["matching", "ahead_by_one", "no_cache_file"],
)
def test_startup_records_the_cache_format(
    test_db: Path,
    served_cache: Path,
    raw_version: str | None,
    expected_matches: bool | None,
) -> None:
    """The lifespan stores the cache's format verdict, makes no claim without a cache, and clears it."""
    if raw_version is not None:
        _write_cache(test_db, served_cache, raw_version)

    with TestClient(app, raise_server_exceptions=False):
        assert hasattr(app.state, "data_format"), (
            "the lifespan did not record the cache's data format at all"
        )
        state = app.state.data_format
        if expected_matches is None:
            assert state is None, "a missing cache file made a claim about its format"
        else:
            assert state is not None
            assert state.matches is expected_matches
            assert state.raw == raw_version

    assert getattr(app.state, "data_format", None) is None, (
        "leaving the lifespan kept a format claim on the shared app"
    )


def test_a_reconnect_rereads_the_format(
    test_db: Path, served_cache: Path, tmp_path: Path
) -> None:
    """A swapped-in cache's format is read on the reconnect, both ways, with no restart."""
    _write_cache(test_db, served_cache, _MATCHING)

    with TestClient(app, raise_server_exceptions=False) as client:
        assert app.state.data_format.matches is True

        _swap_in(_write_cache(test_db, tmp_path / "ahead.duckdb", _AHEAD), served_cache)
        client.get("/bets")
        assert app.state.data_format.matches is False, (
            "the reconnect after a swap did not re-read the cache's data format"
        )

        _swap_in(
            _write_cache(test_db, tmp_path / "back.duckdb", _MATCHING), served_cache
        )
        client.get("/bets")
        assert app.state.data_format.matches is True
