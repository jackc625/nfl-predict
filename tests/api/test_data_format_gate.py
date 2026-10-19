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

import re
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
_BEHIND = str(DATA_FORMAT_VERSION - 1)

# A game the shared test_db fixture carries, with a game_context row (so its page would render).
_FIXTURE_GAME_ID = "2024_W01_BUF@KC"

_NOTICE_TITLE = "Data format mismatch | NFL Predict"
_EXPECTED_WORDING = f"data format {DATA_FORMAT_VERSION}"

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


# ---------------------------------------------------------------------------
# SPEC R9 / D-08: a format clash answers with the standalone 503 notice
# ---------------------------------------------------------------------------


def _fixture_identifiers(cache: Path) -> set[str]:
    """Every game id and team abbreviation the fixture cache carries: cache-derived content."""
    conn = duckdb.connect(str(cache), read_only=True)
    try:
        rows = conn.execute(
            "SELECT game_id, home_team, away_team FROM predictions"
        ).fetchall()
    finally:
        conn.close()
    return {value for row in rows for value in row if value}


@pytest.mark.parametrize(
    ("raw_version", "found_wording"),
    [
        (_AHEAD, f"the data file is format {_AHEAD}"),
        (_BEHIND, f"the data file is format {_BEHIND}"),
        (None, "the data file has no format number"),
        ("abc", "abc"),
    ],
    ids=["ahead_by_one", "behind_by_one", "missing", "non_integer"],
)
def test_a_mismatch_answers_the_home_page_with_the_notice(
    test_db: Path,
    served_cache: Path,
    raw_version: str | None,
    found_wording: str,
) -> None:
    """The notice names both numbers and carries nothing read from the cache."""
    _write_cache(test_db, served_cache, raw_version)
    identifiers = _fixture_identifiers(served_cache)
    assert identifiers, (
        "the fixture carries no games, so the content check would be vacuous"
    )

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/")

    assert response.status_code == 503
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["cache-control"] == "no-store"
    body = response.text
    assert _EXPECTED_WORDING in body
    assert found_wording in body
    if raw_version == "abc":
        assert "which is not a number" in body
    assert _NOTICE_TITLE in body
    assert "Predict" in body and 'class="wordmark"' in body
    for absent in ("<nav", "<script", "cache-timestamp"):
        assert absent not in body, f"the notice carries {absent!r}"
    leaked = sorted(v for v in identifiers if re.search(rf"\b{re.escape(v)}\b", body))
    assert not leaked, f"the notice carries cache-derived content: {leaked}"


_PAGE_ROUTES = (
    "/",
    "/bets",
    "/season",
    "/track-record",
    "/how-it-works",
    f"/games/{_FIXTURE_GAME_ID}",
    "/games/NO_SUCH_GAME",
    "/performance",
    "/backtest",
    "/insights",
    "/betting",
)
_PLAIN_ROUTES = (
    "/fragments/games",
    "/fragments/performance",
    "/fragments/betting",
    "/fragments/season",
    "/api/export/csv",
    "/api/export/json",
)


@pytest.mark.parametrize("path", _PAGE_ROUTES + _PLAIN_ROUTES)
def test_every_gated_route_refuses_on_a_mismatch(
    test_db: Path, served_cache: Path, path: str
) -> None:
    """Every page, fragment and export route answers 503 under a clash (one variant: ahead)."""
    _write_cache(test_db, served_cache, _AHEAD)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get(path, follow_redirects=False)

    assert response.status_code == 503, f"{path} answered {response.status_code}"
    assert response.headers["cache-control"] == "no-store"
    if path in _PAGE_ROUTES:
        assert response.headers["content-type"].startswith("text/html")
    else:
        assert response.headers["content-type"] == "text/plain; charset=utf-8"
    assert _EXPECTED_WORDING in response.text


def test_static_and_health_keep_serving_and_a_match_renders(
    test_db: Path, served_cache: Path, tmp_path: Path
) -> None:
    """/static and /health stay up under a clash; a matching cache renders and redirects normally."""
    _write_cache(test_db, served_cache, _AHEAD)
    with TestClient(app, raise_server_exceptions=False) as client:
        static = client.get("/static/css/tailwind-compiled.css")
        assert static.status_code == 200
        health = client.get("/health")
        assert health.status_code == 200
        assert health.headers["content-type"].startswith("application/json")

    _swap_in(_write_cache(test_db, tmp_path / "match.duckdb", _MATCHING), served_cache)
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/").status_code == 200
        retired = client.get("/performance", follow_redirects=False)
        assert retired.status_code == 301
        assert retired.headers["location"] == "/track-record#seasons"


def test_a_matching_rebuild_clears_the_notice_without_a_restart(
    test_db: Path, served_cache: Path, tmp_path: Path
) -> None:
    """D-19: the gate lets the swap detector reconnect first, so a fixed cache serves at once."""
    _write_cache(test_db, served_cache, _AHEAD)

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/").status_code == 503
        _swap_in(
            _write_cache(test_db, tmp_path / "rebuilt.duckdb", _MATCHING), served_cache
        )
        assert client.get("/").status_code == 200, (
            "a rebuilt matching cache still answered the notice: the gate judged before "
            "the swap detector reconnected"
        )
