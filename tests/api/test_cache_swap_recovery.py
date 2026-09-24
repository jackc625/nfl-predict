"""The G-31-123a regression: a cache swap must be picked up WITHOUT restarting the server.

The ``/bets`` stale-cache refusal names a recovery command -- ``uv run python
scripts/populate_cache.py`` -- and until this module existed, running exactly that against a
RUNNING uvicorn provably could not clear the refusal. ``populate_cache`` does not mutate the
cache in place; it builds a temp DB and then does ``db_path.unlink()`` followed by
``tmp_path.rename(db_path)``. The app holds ONE read-only DuckDB connection for the whole
process lifetime, and its only liveness probe -- ``SELECT 1`` -- PASSES on a handle whose file
has been unlinked. The server therefore kept reading the deleted pre-swap file indefinitely:
HTTP 200, no exception, the per-week marker absent, and the page serving a refusal that named a
command that could not clear it. Diagnosed in
``.planning/debug/bets-stale-cache-recovery-noop.md``.

NOTHING IN THE SUITE EXERCISED ``api.dependencies.get_db`` BEFORE THIS MODULE. Every other
``/bets`` client installs ``app.dependency_overrides[get_db]``, which substitutes a connection
the test opened itself and so bypasses the entire code path under test here. This module
deliberately installs no override and enters the ``TestClient`` as a context manager, so the
real ``api/main.py`` lifespan opens the connection and records its identity and the real
``get_db`` decides, per request, whether that connection is still pointed at the file on disk.

Selectors (``-k``): swap_recovery, unchanged, closed_before, absent_path, records_identity,
connect_window, health

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import api.dependencies as deps
import api.routes.health as health_module
from api.cache import (
    BET_LIST_COLUMNS,
    CACHE_SCHEMA,
    GRADING_STATUS_PENDING,
    bet_list_populated_at_key,
    materialize_available_bet_weeks,
    materialize_bet_list,
    materialize_bet_week_freeze,
)
from api.main import app
from api.services import clear_cache

# ---------------------------------------------------------------------------
# Fixture constants
# ---------------------------------------------------------------------------

_SEASON = 2023
_WEEK = 1
_GAME_ID = "2023_W01_DET@KC"
_SNAPSHOT_TS = "2023-09-07T18:00:00-04:00"
_FREEZE_TS = "2023-09-08T18:00:00-04:00"
_MINUS_110 = -110.0

# The game's own lock, long past. TZ-AWARE, like every instant ``build_bet_week_schedule``
# produces: ``api.routes.pages._as_utc`` REFUSES a naive value rather than guessing a zone for
# it (CR-01), so a naive fixture would exercise a shape the real writer cannot emit.
_FREEZE = datetime(2023, 9, 8, 23, 0, 0, tzinfo=UTC)

# The pre-swap build: its population left NO bet row for the locked game, so the week's list is
# missing and refused (33.2 review C2 CR-02). This is the state the recovery command has to clear.
_OLD_POPULATED_AT = "2023-09-08T22:30:00+00:00"
# The recovery ran and loaded the row. This is the value the post-swap page must render.
_NEW_POPULATED_AT = "2023-09-09T01:15:00+00:00"

# ``cache_meta.last_updated``, which is the ONE cache value ``/health`` reports. Distinct per
# build, because two builds carrying the same value could not tell the endpoint's two possible
# answers apart. NAIVE, matching what ``api.cache.populate_cache`` writes and what the health
# schema round-trips.
_OLD_LAST_UPDATED = "2023-09-08T22:30:00"
_NEW_LAST_UPDATED = "2023-09-09T01:15:00"

_HARD_BLOCK_MESSAGE = (
    "This week&#39;s list is missing -- its locked games have no list in the cache"
)
# The live row table's stake column header -- present only when the list is actually served.
_ROW_TABLE_HEADER = "Stake (units)"
_POPULATED_AT_LABEL = "Bet list last populated"


def _live_row() -> dict[str, Any]:
    """One LIVE ``bet_list`` row carrying every field the live table renders.

    Built from ``BET_LIST_COLUMNS`` rather than a literal dict so a column added to the locked
    schema shows up here as a NULL rather than as a silently absent key.
    """
    row = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": _GAME_ID,
            "season": _SEASON,
            "week": _WEEK,
            "target": "ou",
            "bet_side": "under",
            "model_value": 42.0,
            "market_value": 47.5,
            "line": 47.5,
            "per_bet_ev": 0.0625,
            "stake_units": 1.5,
            "ev_tier": "high",
            "status": "live",
            "snapshot_ts": _SNAPSHOT_TS,
            "freeze_ts": _FREEZE_TS,
            "selected_odds": _MINUS_110,
            "flat_stake": 1.0,
            "provenance": "backtest_replay",
            "validation_type": "contaminated",
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
        }
    )
    return row


def _build_cache(
    db_path: Path,
    *,
    populated_at: str,
    last_updated: str | None = None,
    with_rows: bool = True,
) -> None:
    """Build a complete one-week cache at *db_path*, stamped with *populated_at*.

    *with_rows* False builds the pre-swap state: the schedule and the marker, but no bet row for
    the locked game -- the failed insertion the /bets refusal is for.

    *last_updated* stamps ``cache_meta.last_updated``, the ONE value ``/health`` reports and the
    only thing that distinguishes two builds from that endpoint's point of view. It defaults to
    the fixed builder instant, so the cases that only care about ``/bets`` are unaffected.

    Written with the PUBLIC ``api.cache`` writers, so the fixture is the shape production
    writes rather than a hand-rolled approximation of it. The connection is CLOSED before
    returning: DuckDB caches the database instance by path within a process, so a lingering
    builder handle would keep the pre-swap instance alive and confuse the very mechanism these
    tests measure.
    """
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        if with_rows:
            materialize_bet_list(conn, pd.DataFrame([_live_row()]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame([{"game_id": _GAME_ID, "season": _SEASON, "week": _WEEK}]),
        )
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": _GAME_ID,
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": _FREEZE,
                    }
                ]
            ),
        )
        stamped_at = datetime(2023, 9, 8, 22, 30, 0)
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            [
                [bet_list_populated_at_key(_SEASON, _WEEK), populated_at, stamped_at],
                [
                    "last_updated",
                    last_updated
                    if last_updated is not None
                    else stamped_at.isoformat(),
                    stamped_at,
                ],
            ],
        )
    finally:
        conn.close()


@pytest.fixture()
def swap_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    """A TestClient serving ``/bets`` through the REAL ``get_db``, from a swappable cache file.

    Three things make this fixture different from every other ``/bets`` client in the suite,
    and all three are load-bearing:

    * ``app.dependency_overrides`` is CLEARED and never populated, so ``api.dependencies.get_db``
      actually runs. An override would substitute a connection the test opened and the swap
      detector would never be consulted.
    * ``api.dependencies.DB_PATH`` is patched. ``api/main.py`` reads ``deps.DB_PATH`` and
      ``api/dependencies.py`` reads its own module global, so this one patch covers BOTH openers.
      ``api.routes.health.DB_PATH`` is patched alongside it, because ``/health`` binds that name
      at import and is the SECOND reader of the shared connection (WR-02).
    * The client is entered as a CONTEXT MANAGER, which is what runs the lifespan. The bare
      constructor does not, and without the lifespan no identity is ever recorded.

    ``api.main.app`` is a module-level singleton shared with the rest of ``tests/api``, so the
    two connection fields are reset in teardown.
    """
    db_path = tmp_path / "web_cache.duckdb"
    _build_cache(
        db_path,
        populated_at=_OLD_POPULATED_AT,
        last_updated=_OLD_LAST_UPDATED,
        with_rows=False,
    )

    monkeypatch.setattr(deps, "DB_PATH", db_path)
    monkeypatch.setattr(health_module, "DB_PATH", db_path)
    app.dependency_overrides.clear()
    app.state.db_lock = threading.RLock()
    clear_cache()

    try:
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client
    finally:
        app.dependency_overrides.clear()
        app.state.db_conn = None
        app.state.db_identity = None
        clear_cache()


def _swap_in(new_path: Path, db_path: Path) -> None:
    """Reproduce ``api.cache.populate_cache``'s swap exactly: unlink, then rename."""
    db_path.unlink()
    new_path.rename(db_path)


def _bets(client: TestClient) -> str:
    response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")
    assert response.status_code == 200, response.text
    return response.text


def _health_last_updated(client: TestClient) -> str | None:
    """``/health``'s reported ``last_updated``, which is what the RUNBOOK sends an operator to."""
    response = client.get("/health")
    assert response.status_code == 200, response.text
    return response.json()["last_updated"]


# ---------------------------------------------------------------------------
# The load-bearing case
# ---------------------------------------------------------------------------


def test_a_cache_swap_is_picked_up_without_restarting_the_server(
    swap_client: TestClient, tmp_path: Path
) -> None:
    """The recovery command's swap clears the refusal on the very NEXT request.

    This is the reported defect, end to end. Before the fix, step 4 returned the identical
    refusal that step 1 returned and only a process restart changed it.

    The NEW marker value is asserted explicitly, not merely the absence of the block. A
    non-None ``populated_at`` is memoized by ``api/services.py``'s TTLCache for five minutes,
    so a body carrying the NEW value is what proves the ``clear_cache()`` on reconnect is
    doing its job. A fixture whose pre-swap marker were ABSENT would cache nothing, and a
    missing ``clear_cache()`` would sail through.
    """
    db_path = tmp_path / "web_cache.duckdb"

    # 1. Pre-swap: the population left no row for the locked game, so the week is refused.
    body = _bets(swap_client)
    assert _HARD_BLOCK_MESSAGE in body, (
        "the fixture did not reach the withheld state, so the swap would prove nothing"
    )
    assert _ROW_TABLE_HEADER not in body, "the list rendered despite the hard block"

    # 2. The rebuilt cache: the same schedule, now WITH the game's row, and a newer marker.
    new_path = tmp_path / "web_cache.duckdb.tmp"
    _build_cache(new_path, populated_at=_NEW_POPULATED_AT)

    # 3. The writer's swap, byte for byte. Every connection this test opened to either file is
    #    already closed by _build_cache; the app's is the only one still open, which is the point.
    _swap_in(new_path, db_path)

    # 4. Same client, same URL, no restart.
    body = _bets(swap_client)
    assert _HARD_BLOCK_MESSAGE not in body, (
        "the running server is still serving the pre-swap file: the recovery command the "
        "refusal names still cannot clear the refusal"
    )
    assert _ROW_TABLE_HEADER in body, (
        "the block cleared but the bet list did not render"
    )
    assert _NEW_POPULATED_AT in body, (
        "the page cleared the block but rendered a stale populated-at marker: a value "
        "memoized from the pre-swap file straddled the swap"
    )
    assert _POPULATED_AT_LABEL in body


# ---------------------------------------------------------------------------
# The four controls
# ---------------------------------------------------------------------------


def test_an_unchanged_cache_file_does_not_reconnect(swap_client: TestClient) -> None:
    """Two consecutive requests against an UNCHANGED file share one connection object.

    Without this control the load-bearing test above would also pass under an implementation
    that reconnected on every single request -- a real performance regression, and one that
    would make the identity comparison pointless (it would never be consulted).
    """
    _bets(swap_client)
    first = app.state.db_conn
    _bets(swap_client)
    second = app.state.db_conn

    assert first is not None
    assert first is second, (
        "the connection was replaced although the cache file never changed: the identity "
        "check has become a per-request reconnect rather than a detector"
    )


def test_the_old_connection_is_closed_before_the_reconnect(
    swap_client: TestClient, tmp_path: Path
) -> None:
    """The superseded connection is CLOSED, which is what makes the reconnect recover at all.

    DuckDB caches the database instance BY PATH within a process, so a connect issued while the
    old connection is still alive returns the SAME stale instance -- proven live in the
    diagnosis (E1(D) / E4 ALT-3). Closing first is therefore not hygiene, it is the mechanism.

    The observed exception on a closed connection is ``duckdb.ConnectionException``
    ("Connection Error: Connection already closed!"), a subclass of ``duckdb.Error``; the
    expectation is pinned at ``duckdb.Error`` rather than widened to a bare ``Exception``, which
    would also be satisfied by an unrelated crash.
    """
    db_path = tmp_path / "web_cache.duckdb"
    _bets(swap_client)
    old_conn = app.state.db_conn
    assert old_conn is not None

    new_path = tmp_path / "web_cache.duckdb.tmp"
    _build_cache(new_path, populated_at=_NEW_POPULATED_AT)
    _swap_in(new_path, db_path)

    _bets(swap_client)
    assert app.state.db_conn is not old_conn, "no reconnect happened at all"

    with pytest.raises(duckdb.Error):
        old_conn.execute("SELECT 1")


def test_a_momentarily_absent_cache_path_keeps_serving(
    swap_client: TestClient, tmp_path: Path
) -> None:
    """The instant between the writer's unlink and its rename keeps serving the open handle.

    A swap must not be able to turn a working page into an error page. The already-open handle
    still reads fine while the path is gone, and the first request after the rename lands is the
    one that sees the new identity.
    """
    db_path = tmp_path / "web_cache.duckdb"
    _bets(swap_client)
    before = app.state.db_conn
    assert before is not None

    aside = tmp_path / "web_cache.duckdb.aside"
    db_path.rename(aside)
    try:
        response = swap_client.get(f"/bets?season={_SEASON}&week={_WEEK}")
        assert response.status_code == 200, (
            "an absent cache path turned a working page into an error page: the "
            "unlink-to-rename window is a normal part of every population run"
        )
        assert app.state.db_conn is before, (
            "a working connection was torn down while the path did not exist, so there was "
            "nothing to replace it with"
        )
    finally:
        aside.rename(db_path)


def test_the_startup_connection_records_an_identity(
    swap_client: TestClient, tmp_path: Path
) -> None:
    """The lifespan records the identity of the file it opened.

    The anti-vacuity control. A lifespan that stopped recording the identity would leave
    ``cache_file_changed`` with nothing to compare against -- it returns False on an absent
    recorded identity by design -- and every other case in this module would still pass while
    the detector was permanently switched off.
    """
    db_path = tmp_path / "web_cache.duckdb"
    assert app.state.db_identity is not None, (
        "the lifespan opened a connection without recording which file it opened: the swap "
        "detector is switched off"
    )
    assert app.state.db_identity == deps.cache_identity(db_path)


# ---------------------------------------------------------------------------
# The swap that lands INSIDE the connect window (CR-01)
# ---------------------------------------------------------------------------


def _connect_that_swaps_after_opening(
    new_path: Path, db_path: Path
) -> tuple[Any, dict[str, bool]]:
    """A ``duckdb.connect`` stand-in that lands the writer's swap INSIDE the connect window.

    It opens the real connection FIRST and only then does the ``unlink()`` + ``rename()``, which
    is the ordering that produces the defect: the returned handle holds the file that WAS at the
    path, while the path itself now holds a different one. A stand-in that swapped before opening
    would hand back a handle on the NEW file and could not distinguish the two orderings at all.

    Fires ONCE. Every later call -- the healing reconnect the fix is supposed to produce -- is a
    plain delegation, so the same patch can stay installed for the rest of the test.
    """
    real_connect = duckdb.connect
    fired = {"swapped": False}

    def connect_then_swap(*args: Any, **kwargs: Any) -> Any:
        conn = real_connect(*args, **kwargs)
        if not fired["swapped"]:
            fired["swapped"] = True
            _swap_in(new_path, db_path)
        return conn

    return connect_then_swap, fired


def test_a_swap_inside_the_reconnect_connect_window_is_still_detected(
    swap_client: TestClient, tmp_path: Path
) -> None:
    """A swap landing between the connect and the identity record must NOT blind the detector.

    THE REGRESSION THIS MODULE DID NOT COVER. All five cases above swap BETWEEN requests, so all
    five pass under an implementation that records the identity by re-stat'ing the path AFTER
    ``duckdb.connect``. Under that ordering a swap landing inside the connect window leaves the
    handle on the OLD file and the RECORDED identity describing the NEW one, so
    ``cache_file_changed`` compares EQUAL on every subsequent request and only a process restart
    ever clears it -- the exact G-31-123a symptom plan 31-20 exists to remove, reinstated by the
    plan's own last statement.

    The reachable path is two population runs, or a population overlapping a server start: the
    first swap is what triggers the reconnect, and the second lands inside it. That is the
    sequence below, with the second swap driven from a ``duckdb.connect`` stand-in so the window
    is hit deterministically rather than by racing a real writer.

    The fix records the identity read BEFORE the connect, so the recorded value is STALE rather
    than wrong-by-one-file. Staleness mismatches on the very next request and self-heals through
    one spurious reconnect; that failure direction is the whole point.
    """
    db_path = tmp_path / "web_cache.duckdb"

    # 1. Establish the connection and the recorded identity of the file it opened.
    _bets(swap_client)

    # 2. The FIRST population run's swap. This is only the trigger: it makes the next request
    #    enter the reconnect, which is where the defect lives.
    second_path = tmp_path / "web_cache.duckdb.tmp"
    _build_cache(second_path, populated_at=_OLD_POPULATED_AT, with_rows=False)
    _swap_in(second_path, db_path)
    identity_of_the_file_the_reconnect_opens = deps.cache_identity(db_path)
    assert identity_of_the_file_the_reconnect_opens is not None

    # 3. The SECOND population run's swap, armed to land inside the connect window. This one
    #    carries the row and the newer marker, so the page can prove which file it serves.
    third_path = tmp_path / "web_cache.duckdb.tmp2"
    _build_cache(third_path, populated_at=_NEW_POPULATED_AT)
    connect_then_swap, fired = _connect_that_swaps_after_opening(third_path, db_path)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(duckdb, "connect", connect_then_swap)

        # 4. The request that reconnects. Its own body is not asserted on: this request is
        #    legitimately serving the file it just opened, which the swap has already superseded.
        _bets(swap_client)
        assert fired["swapped"], (
            "the stand-in never fired, so no swap landed inside the connect window and this "
            "test would pass under either ordering"
        )

        # THE LOAD-BEARING ASSERTION. The recorded identity must describe the file this
        # connection actually OPENED, not whatever replaced it a moment later.
        assert app.state.db_identity == identity_of_the_file_the_reconnect_opens, (
            "the identity was recorded by re-stat'ing the path after the connect, so it "
            "describes a file this connection is not pointing at: the detector will compare "
            "equal forever and only a restart will clear the staleness (G-31-123a)"
        )

        # 5. The consequence, end to end: the next request notices and heals.
        body = _bets(swap_client)

    assert _HARD_BLOCK_MESSAGE not in body, (
        "the swap that landed inside the connect window was never detected, so the server is "
        "still serving a file that has been deleted from underneath it"
    )
    assert _NEW_POPULATED_AT in body, (
        "the block cleared but the page did not render the marker from the file that is "
        "actually at the path now"
    )


def test_a_swap_inside_the_lifespan_connect_window_is_still_detected(
    tmp_path: Path,
) -> None:
    """The same ordering in the lifespan, which has the WIDEST exposure of the two openers.

    ``api.main.lifespan`` opens the process-lifetime connection with a COLD ``duckdb.connect`` on
    a ~6 MB database at process boot -- the longest connect window in the app, and precisely the
    one a Friday orchestrator population overlapping a service restart lands in. A separate case
    because it is a separate opener in a separate file: fixing ``api/dependencies.py`` alone would
    leave every process start able to record an identity for a file it is not reading.

    Fixture-free rather than built on ``swap_client``, because the swap has to land during the
    lifespan's own connect -- that is, before any fixture could hand back a client.
    """
    db_path = tmp_path / "web_cache.duckdb"
    _build_cache(db_path, populated_at=_OLD_POPULATED_AT, with_rows=False)
    identity_of_the_file_the_lifespan_opens = deps.cache_identity(db_path)
    assert identity_of_the_file_the_lifespan_opens is not None

    swapped_in_path = tmp_path / "web_cache.duckdb.tmp"
    _build_cache(swapped_in_path, populated_at=_NEW_POPULATED_AT)
    connect_then_swap, fired = _connect_that_swaps_after_opening(
        swapped_in_path, db_path
    )

    app.dependency_overrides.clear()
    app.state.db_lock = threading.RLock()
    clear_cache()
    try:
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(deps, "DB_PATH", db_path)
            patched.setattr(duckdb, "connect", connect_then_swap)
            with TestClient(app, raise_server_exceptions=False) as client:
                assert fired["swapped"], (
                    "the stand-in never fired during the lifespan, so no swap landed inside "
                    "its connect window"
                )
                assert (
                    app.state.db_identity == identity_of_the_file_the_lifespan_opens
                ), (
                    "the lifespan recorded the identity of the file that REPLACED the one it "
                    "opened, so this process can never detect that its connection is stale"
                )
                body = _bets(client)
    finally:
        app.dependency_overrides.clear()
        app.state.db_conn = None
        app.state.db_identity = None
        clear_cache()

    assert _HARD_BLOCK_MESSAGE not in body, (
        "a swap landing inside the lifespan's connect window left the process permanently "
        "blind: the first request served the pre-swap file and did not reconnect"
    )
    assert _NEW_POPULATED_AT in body


# ---------------------------------------------------------------------------
# The SECOND reader of the shared connection (WR-02)
# ---------------------------------------------------------------------------


def test_health_reports_post_swap_freshness_without_any_page_request(
    swap_client: TestClient, tmp_path: Path
) -> None:
    """``/health`` must not keep reporting the DELETED pre-swap file's ``last_updated``.

    THE SECOND READER. ``/health`` reads ``app.state.db_conn`` directly and deliberately
    bypasses ``get_db`` so it can never 503 -- which meant plan 31-20 fixed the page reader and
    left the OPERATIONAL reader blind. RUNBOOK.md operation 7 and PIPELINE.md stage 8 both send
    the operator to this endpoint to confirm a rebuild took effect, so a pre-swap timestamp here
    reads as "the population did not work" and invites them to run it again.

    NO ``/bets`` REQUEST IS MADE AFTER THE SWAP, and that omission is the whole test. Any
    ``get_db``-backed request would reconnect ``app.state`` as a side effect and repair
    ``/health`` for free, which is exactly how the defect stayed invisible: it only shows up
    when ``/health`` is the FIRST thing touched after the population, which is what an operator
    following the runbook actually does.
    """
    db_path = tmp_path / "web_cache.duckdb"

    assert _health_last_updated(swap_client) == _OLD_LAST_UPDATED, (
        "the fixture did not reach the pre-swap state, so the swap below would prove nothing"
    )

    new_path = tmp_path / "web_cache.duckdb.tmp"
    _build_cache(
        new_path, populated_at=_NEW_POPULATED_AT, last_updated=_NEW_LAST_UPDATED
    )
    _swap_in(new_path, db_path)

    assert _health_last_updated(swap_client) == _NEW_LAST_UPDATED, (
        "/health is still reporting the last_updated of the file that was DELETED from under "
        "it, so the endpoint the runbook points an operator at after a rebuild tells them the "
        "rebuild did not happen"
    )


def test_health_does_not_tear_down_its_connection_in_the_swap_window(
    swap_client: TestClient, tmp_path: Path
) -> None:
    """The anti-regression control: the writer's unlink-to-rename instant must stay non-fatal.

    A freshness check that reconnected whenever it could not CONFIRM freshness would fire during
    every population run's unlink-to-rename window and make ``/health`` own connection teardown
    on a routine event. ``cache_file_changed`` returns False on an absent on-disk identity for
    exactly that reason, so nothing is torn down and the endpoint still answers 200.

    ``last_updated`` is deliberately NOT asserted here: ``/health`` gates the whole read on
    ``DB_PATH.exists()``, so an absent path reports ``cache_ready: false`` and a null timestamp.
    That is pre-existing, correct, and orthogonal -- the claim under test is that the connection
    survives.
    """
    db_path = tmp_path / "web_cache.duckdb"
    assert _health_last_updated(swap_client) == _OLD_LAST_UPDATED
    before = app.state.db_conn
    assert before is not None

    aside = tmp_path / "web_cache.duckdb.aside"
    db_path.rename(aside)
    try:
        response = swap_client.get("/health")
        assert response.status_code == 200, response.text
        assert response.json()["cache_ready"] is False
        assert app.state.db_conn is before, (
            "/health tore down a working connection during the writer's unlink-to-rename "
            "window, turning a routine part of every population run into a teardown"
        )
    finally:
        aside.rename(db_path)

    assert _health_last_updated(swap_client) == _OLD_LAST_UPDATED, (
        "the same file came back at the same path and /health no longer reads it"
    )
