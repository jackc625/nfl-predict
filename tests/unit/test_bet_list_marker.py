"""The bet-list populated-at marker: per-week, and written STRICTLY AFTER a successful insert.

WHY THE MARKER IS NOT THE GENERIC CACHE TIMESTAMP
--------------------------------------------------
``/bets`` refuses to serve a week whose bet list was populated BEFORE that week's latest per-game
line freeze. The value it compares is this marker, and D31-29 rejected the obvious alternative --
reading ``cache_meta['last_updated']`` -- for a concrete reason: that timestamp advances whenever
ANY cache table is repopulated. A run that populated predictions and then FAILED on the bet list
would look perfectly fresh, so the guard would be defeated by the exact failure it exists to catch.

Two properties make the marker immune to that, and both are asserted here:

* it is **keyed by season and week**, so populating one week cannot vouch for another; and
* it is written **strictly after a successful blob insert**, so a raise anywhere in the insert
  leaves the previous value in place and the block fires deterministically.

WHY THE WRITES MUST LAND IN THE TEMPORARY DATABASE (REVIEW-CACHE)
------------------------------------------------------------------
``populate_cache`` does NOT update the live cache. It opens a FRESH temporary DuckDB, loads its
sources into it, then runs ``db_path.unlink()`` followed by ``tmp_path.rename(db_path)``. Anything
written to the LIVE cache before a population run is therefore DELETED by it -- taking the marker
and every preserved forward row with it. ``test_every_bet_list_write_uses_the_temp_connection``
pins that structurally, over the function's own AST, because a behavioural check would only ever
prove that one fixture week happened to survive.

Selectors (``-k``): key, marker, after_insert, failure, empty, generic, temp_connection, schema.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import inspect
from datetime import UTC, datetime
from typing import Any

import duckdb
import pandas as pd
import pytest

from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_POPULATED_AT_KEY,
    CACHE_META_SCHEMA,
    CACHE_SCHEMA,
    GRADING_STATUS_PENDING,
    bet_list_populated_at_key,
    materialize_bet_list_with_marker,
    populate_cache,
)

_SEASON = 2023
_WEEK = 1
_OTHER_WEEK = 2
_POPULATED_AT = datetime(2023, 9, 8, 22, 30, 0, tzinfo=UTC)
_LATER = datetime(2023, 9, 15, 22, 30, 0, tzinfo=UTC)


def _row(
    game_id: str, *, season: int = _SEASON, week: int = _WEEK, **overrides: Any
) -> dict:
    """A minimal but COMPLETE bet-list row: every locked column present, nothing invented."""
    row: dict[str, Any] = {
        "game_id": game_id,
        "season": season,
        "week": week,
        "target": "ou",
        "bet_side": "under",
        "model_value": 41.0,
        "market_value": 45.5,
        "line": 45.5,
        "slipped_line": 45.5,
        "calibrated_p_side": 0.56,
        "per_bet_ev": 0.0712,
        "stake_units": 1.25,
        "ev_tier": "high",
        "status": "live",
        "rejection_reason": None,
        "eligibility_label": "UNDER pick",
        "snapshot_ts": "2023-09-07T18:00:00-04:00",
        "freeze_ts": "2023-09-08T18:00:00-04:00",
        "selected_odds": -110.0,
        "flat_stake": 1.0,
        "provenance": "forward",
        "validation_type": "forward_realized",
        # The 29th locked column (Phase 33, Plan 33-05 Task 3). A forward row carries its own
        # observation time, strictly before its own freeze.
        "decided_at_utc": "2023-09-08T17:45:00-04:00",
        "grading_status": GRADING_STATUS_PENDING,
        "outcome": None,
        "clv": 0.02,
        "payout_flat": None,
        "realized_units": None,
        "graded_at": None,
    }
    # Phase 34 (Plan 34-01 Task 2) widened the locked schema from 29 to 51. Was: the literal above
    # was the whole width. The 22 Phase-34 columns are NULL, as on every row the pre-ledger writer
    # produced; the drift check below still proves the row spans the locked width exactly.
    for column in BET_LIST_COLUMNS:
        row.setdefault(column, None)
    row.update(overrides)
    assert set(row) == set(BET_LIST_COLUMNS), (
        "the fixture row drifted from BET_LIST_COLUMNS; the marker tests would be exercising a "
        f"different schema. Difference: {set(row) ^ set(BET_LIST_COLUMNS)}"
    )
    return row


def _conn() -> duckdb.DuckDBPyConnection:
    """An in-memory DB carrying the ``cache_meta`` table the marker is written into."""
    conn = duckdb.connect(":memory:")
    conn.execute(CACHE_META_SCHEMA)
    return conn


def _meta(conn: duckdb.DuckDBPyConnection) -> dict[str, str]:
    return dict(conn.execute("SELECT key, value FROM cache_meta").fetchall())


# ---------------------------------------------------------------------------
# The key itself
# ---------------------------------------------------------------------------


def test_the_key_is_composed_from_the_season_and_the_week() -> None:
    """Two different weeks produce two different keys -- that is the whole point of the marker."""
    assert bet_list_populated_at_key(_SEASON, _WEEK) != bet_list_populated_at_key(
        _SEASON, _OTHER_WEEK
    )
    assert bet_list_populated_at_key(_SEASON, _WEEK) != bet_list_populated_at_key(
        _SEASON + 1, _WEEK
    )
    assert str(_SEASON) in bet_list_populated_at_key(_SEASON, _WEEK)
    assert str(_WEEK) in bet_list_populated_at_key(_SEASON, _WEEK)


def test_the_key_is_never_the_bare_generic_key() -> None:
    """A per-week marker that collided with a generic key would be the defect D31-29 rejected."""
    assert bet_list_populated_at_key(_SEASON, _WEEK) != BET_LIST_POPULATED_AT_KEY
    assert bet_list_populated_at_key(_SEASON, _WEEK) != "last_updated"
    # It is still recognisably part of the same family, so a reader of cache_meta can find it.
    assert bet_list_populated_at_key(_SEASON, _WEEK).startswith(
        BET_LIST_POPULATED_AT_KEY
    )


def test_the_key_accepts_numpy_and_python_integers_identically() -> None:
    """A season read out of a DataFrame is a numpy int; it must key the same cell as a Python int."""
    import numpy as np

    assert bet_list_populated_at_key(np.int64(_SEASON), np.int64(_WEEK)) == (
        bet_list_populated_at_key(_SEASON, _WEEK)
    )


# ---------------------------------------------------------------------------
# Written strictly AFTER a successful insert
# ---------------------------------------------------------------------------


def test_a_successful_insert_stamps_one_marker_per_week_present_in_the_frame() -> None:
    conn = _conn()
    try:
        inserted = materialize_bet_list_with_marker(
            conn,
            pd.DataFrame(
                [
                    _row("2023_W01_DET@KC"),
                    _row("2023_W01_CAR@ATL"),
                    _row("2023_W02_AAA@BBB", week=_OTHER_WEEK),
                ]
            ),
            populated_at=_POPULATED_AT,
        )
        assert inserted == 3
        meta = _meta(conn)
        assert (
            meta[bet_list_populated_at_key(_SEASON, _WEEK)] == _POPULATED_AT.isoformat()
        )
        assert (
            meta[bet_list_populated_at_key(_SEASON, _OTHER_WEEK)]
            == _POPULATED_AT.isoformat()
        )
    finally:
        conn.close()


def test_populating_one_week_leaves_another_weeks_marker_untouched() -> None:
    """The keying is what stops a run on week 2 vouching for a week 1 that was never populated."""
    conn = _conn()
    try:
        materialize_bet_list_with_marker(
            conn, pd.DataFrame([_row("2023_W01_DET@KC")]), populated_at=_POPULATED_AT
        )
        materialize_bet_list_with_marker(
            conn,
            pd.DataFrame([_row("2023_W02_AAA@BBB", week=_OTHER_WEEK)]),
            populated_at=_LATER,
        )
        meta = _meta(conn)
        assert (
            meta[bet_list_populated_at_key(_SEASON, _WEEK)] == _POPULATED_AT.isoformat()
        ), (
            "the second run advanced week 1's marker; a run that never touched a week must not "
            "vouch for its freshness"
        )
        assert (
            meta[bet_list_populated_at_key(_SEASON, _OTHER_WEEK)] == _LATER.isoformat()
        )
    finally:
        conn.close()


def test_a_failed_insert_leaves_the_marker_at_its_previous_value() -> None:
    """A raise anywhere in the insert must leave the marker UNADVANCED, so the block fires.

    The failure is induced with a grading_status outside the closed four-state vocabulary, which
    ``materialize_bet_list`` validates BEFORE writing. The marker is read before and after.
    """
    conn = _conn()
    try:
        materialize_bet_list_with_marker(
            conn, pd.DataFrame([_row("2023_W01_DET@KC")]), populated_at=_POPULATED_AT
        )
        before = _meta(conn)[bet_list_populated_at_key(_SEASON, _WEEK)]

        with pytest.raises(ValueError):
            materialize_bet_list_with_marker(
                conn,
                pd.DataFrame([_row("2023_W01_DET@KC", grading_status="settled")]),
                populated_at=_LATER,
            )

        after = _meta(conn)[bet_list_populated_at_key(_SEASON, _WEEK)]
        assert after == before == _POPULATED_AT.isoformat(), (
            "a failed insert advanced the marker; the stale-cache block would then read the "
            "failed run as fresh, which is the exact failure D31-29 designed the marker against"
        )
    finally:
        conn.close()


def test_a_missing_column_also_leaves_the_marker_unadvanced() -> None:
    """The other raise path -- a schema mismatch -- must not stamp either."""
    conn = _conn()
    try:
        materialize_bet_list_with_marker(
            conn, pd.DataFrame([_row("2023_W01_DET@KC")]), populated_at=_POPULATED_AT
        )
        short = pd.DataFrame([_row("2023_W01_DET@KC")]).drop(columns=["per_bet_ev"])
        with pytest.raises(KeyError):
            materialize_bet_list_with_marker(conn, short, populated_at=_LATER)
        assert (
            _meta(conn)[bet_list_populated_at_key(_SEASON, _WEEK)]
            == _POPULATED_AT.isoformat()
        )
    finally:
        conn.close()


def test_an_empty_frame_writes_no_marker_at_all() -> None:
    """The ZERO-ROW state the hard-block exists for: no rows, therefore nothing vouched for.

    An empty frame is NOT an error (SPEC R4 empty), so nothing raises -- but it must not stamp a
    marker either, or a week whose insertion silently produced nothing would read as populated.
    """
    conn = _conn()
    try:
        assert (
            materialize_bet_list_with_marker(
                conn,
                pd.DataFrame(columns=pd.Index(BET_LIST_COLUMNS)),
                populated_at=_POPULATED_AT,
            )
            == 0
        )
        assert _meta(conn) == {}
    finally:
        conn.close()


def test_the_stamped_value_round_trips_to_the_instant_it_was_given() -> None:
    """The stored text parses back to the same aware instant -- never a naive local wall clock."""
    conn = _conn()
    try:
        materialize_bet_list_with_marker(
            conn, pd.DataFrame([_row("2023_W01_DET@KC")]), populated_at=_POPULATED_AT
        )
        stored = _meta(conn)[bet_list_populated_at_key(_SEASON, _WEEK)]
        parsed = datetime.fromisoformat(stored)
        assert parsed.tzinfo is not None, "the marker was stamped without a timezone"
        assert parsed == _POPULATED_AT
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# The generic timestamp is NOT the marker
# ---------------------------------------------------------------------------


def test_advancing_the_generic_timestamp_does_not_advance_the_bet_list_marker() -> None:
    """Directly asserted on the two keys: they are independent cells, not one value read twice."""
    conn = _conn()
    try:
        materialize_bet_list_with_marker(
            conn, pd.DataFrame([_row("2023_W01_DET@KC")]), populated_at=_POPULATED_AT
        )
        conn.execute(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            ["last_updated", _LATER.isoformat(), _LATER],
        )
        meta = _meta(conn)
        assert meta["last_updated"] == _LATER.isoformat()
        assert (
            meta[bet_list_populated_at_key(_SEASON, _WEEK)] == _POPULATED_AT.isoformat()
        )
    finally:
        conn.close()


def test_a_population_run_that_supplies_no_bet_list_advances_only_the_generic_timestamp(
    tmp_path,
) -> None:
    """The D31-29 scenario, end to end: predictions populated, bet list absent.

    A full ``populate_cache`` with no bet-list frame writes ``last_updated`` -- the cache IS
    newly built -- and stamps NO per-week marker. Reading freshness off ``last_updated`` would
    call this cache current; reading it off the marker correctly finds nothing.
    """
    db_path = tmp_path / "no_bet_list.duckdb"
    empty = tmp_path / "empty"
    empty.mkdir()

    populate_cache(
        db_path=db_path,
        artifacts_dir=empty,
        outputs_dir=empty,
        gold_dir=empty,
        silver_dir=empty,
    )

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        meta = _meta(conn)
    finally:
        conn.close()

    assert meta.get("last_updated"), "the generic cache timestamp was not written"
    marker_keys = [key for key in meta if key.startswith(BET_LIST_POPULATED_AT_KEY)]
    assert marker_keys == [], (
        f"a run that populated no bet list stamped {marker_keys}; the stale-cache block would "
        "then read a failed bet-list population as fresh"
    )


# ---------------------------------------------------------------------------
# Structural: every bet-list write lands in the TEMPORARY database
# ---------------------------------------------------------------------------

# The writers whose target connection is load-bearing. Every one of them must receive the
# temp-build connection; a live-cache connection would be erased by the very rename that ends the
# function.
_BET_LIST_WRITERS = frozenset(
    {
        "materialize_bet_list_with_marker",
        "materialize_available_bet_weeks",
        "materialize_bet_week_freeze",
        "materialize_bet_tracker_blocks",
    }
)


def _populate_cache_ast() -> ast.FunctionDef:
    tree = ast.parse(inspect.cleandoc(inspect.getsource(populate_cache)))
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    return func


def test_every_bet_list_write_uses_the_temp_connection() -> None:
    """REVIEW-CACHE, structurally: the four writers are all handed the temp-build connection.

    Behaviour cannot prove this. A live-cache write followed by a population run leaves NO trace
    of itself -- the rows are simply gone -- so a passing round-trip on one fixture week would say
    nothing about the general case. The AST does prove it.
    """
    func = _populate_cache_ast()

    connects = [
        node
        for node in ast.walk(func)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "connect"
    ]
    assert len(connects) == 1, (
        f"populate_cache opens {len(connects)} connections; exactly one temp-build connection is "
        "the premise this guard rests on"
    )

    handles = {
        target.id
        for node in ast.walk(func)
        if isinstance(node, ast.Assign) and node.value in connects
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert len(handles) == 1, (
        f"could not identify the single connection handle; found {sorted(handles)}"
    )
    temp_handle = next(iter(handles))

    called: set[str] = set()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in _BET_LIST_WRITERS:
            continue
        called.add(node.func.id)
        assert node.args, f"{node.func.id} is called with no connection argument"
        first = node.args[0]
        assert isinstance(first, ast.Name) and first.id == temp_handle, (
            f"{node.func.id} at line {node.lineno} is not passed the temp-build connection "
            f"{temp_handle!r}; anything it writes would be destroyed by the atomic rename that "
            "ends populate_cache (REVIEW-CACHE)"
        )

    assert called == _BET_LIST_WRITERS, (
        "populate_cache does not call every bet-list writer, so this guard is partly vacuous. "
        f"Missing: {sorted(_BET_LIST_WRITERS - called)}"
    )


def test_populate_cache_never_names_the_live_cache_path_as_a_write_target() -> None:
    """``db_path`` reaches only the temp-path derivation, the unlink and the rename.

    Restated here rather than left to ``tests/api/test_cache_atomic_swap.py`` because the bet-list
    load is the FIRST thing in this function whose correctness depends on it, and a reader
    changing that load needs the constraint stated where they are working.
    """
    func = _populate_cache_ast()
    for node in ast.walk(func):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)):
            continue
        if node.func.id not in _BET_LIST_WRITERS:
            continue
        names = {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        assert "db_path" not in names, (
            f"{node.func.id} at line {node.lineno} references db_path; a bet-list write against "
            "the LIVE cache is deleted by the rename at the end of this function"
        )


# ---------------------------------------------------------------------------
# The standalone cache_meta schema agrees with the one inside CACHE_SCHEMA
# ---------------------------------------------------------------------------


def test_the_standalone_cache_meta_schema_matches_the_full_cache_schema() -> None:
    """Two spellings of one table is a drift hazard, so the two are compared column by column.

    ``CACHE_META_SCHEMA`` exists so the marker writer can run against an in-memory test DB (and
    against a partially built cache) without first creating every table. That convenience is only
    safe while the two definitions agree.
    """
    standalone = duckdb.connect(":memory:")
    full = duckdb.connect(":memory:")
    try:
        standalone.execute(CACHE_META_SCHEMA)
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                full.execute(stmt)

        def _describe(conn: duckdb.DuckDBPyConnection) -> list[tuple[str, str]]:
            return [
                (row[0], row[1])
                for row in conn.execute("DESCRIBE cache_meta").fetchall()
            ]

        assert _describe(standalone) == _describe(full)
    finally:
        standalone.close()
        full.close()
