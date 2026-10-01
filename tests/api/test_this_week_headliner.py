"""This Week's headliner area agrees with /bets about a week (Broadcast redesign, Task 6).

The landing page now leads with the week's bets. It must never describe a week differently from
/bets, so the headliner's state is not decided a second time: it is read off the same
``_build_bets_context`` /bets renders from, in pages/bets.html's own precedence. These tests build
one REAL cache per /bets state, render /bets for it, and check the headliner names the same state.
They then pin the decorated slate the game grid renders from.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from api.cache import (
    BET_LIST_COLUMNS,
    CACHE_SCHEMA,
    GRADING_STATUS_PENDING,
    PREDICTIONS_TABLE_COLUMNS,
    bet_list_populated_at_key,
    materialize_available_bet_weeks,
    materialize_bet_list,
    materialize_bet_week_freeze,
)
from api.routes.pages import (
    HEADLINER_STATES,
    _build_headliner,
    _headliner_state,
    _this_week_grid_context,
)
from api.services import DataService, clear_cache

_SEASON = 2023
_WEEK = 1
_GAME = "2023_W01_DET@KC"
_SECOND_GAME = "2023_W01_CAR@ATL"
_PAST_LOCK = datetime(2023, 9, 7, 22, 0, 0, tzinfo=UTC)
_FUTURE_LOCK = datetime(2099, 9, 7, 22, 0, 0, tzinfo=UTC)

# The text only one /bets render carries, per state. "bets" is the render that carries none.
_BETS_PAGE_MARKERS: dict[str, str] = {
    "not_built": "Bet list not built yet",
    "blocked": "its locked games have no list in the cache",
    "not_evaluated": "No game of this week has reached its lock.",
    "no_week": "No current week",
    "none_cleared": "No bets cleared the floor this week",
}


class _StubRequest:
    """The one attribute a full-page render needs from a request: ``url_for`` for static files."""

    def url_for(self, name: str, **path_params: Any) -> str:
        return f"/{name}/{path_params.get('path', '')}"


_REQUEST = cast(Request, _StubRequest())


def _rendered_state(html: str) -> str:
    """Which of the six renders a /bets page made, read off the text only that render carries."""
    present = [state for state, marker in _BETS_PAGE_MARKERS.items() if marker in html]
    assert len(present) <= 1, f"/bets rendered more than one state at once: {present}"
    return present[0] if present else "bets"


def _bet_row(status: str) -> dict[str, Any]:
    """One bet_list row for _GAME carrying the minimum /bets renders, live or suppressed."""
    row = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": _GAME,
            "season": _SEASON,
            "week": _WEEK,
            "target": "ou",
            "bet_side": "under",
            "line": 47.5,
            "status": status,
            "snapshot_ts": "2023-09-07T18:00:00-04:00",
            "freeze_ts": "2023-09-07T18:00:00-04:00",
            "provenance": "backtest_replay",
            "validation_type": "contaminated",
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
        }
    )
    if status == "live":
        row.update(
            {
                "per_bet_ev": 0.0625,
                "stake_units": 1.5,
                "ev_tier": "high",
                "selected_odds": -110.0,
                "flat_stake": 1.0,
            }
        )
    else:
        row["rejection_reason"] = "ev_below_floor"
    return row


def _prediction(game_id: str, game_date: datetime | None) -> dict[str, Any]:
    """One scheduled prediction row for *game_id*, shaped like the predictions table."""
    away, home = game_id.rsplit("_", maxsplit=1)[-1].split("@")
    row = dict.fromkeys(PREDICTIONS_TABLE_COLUMNS)
    row.update(
        game_id=game_id,
        season=_SEASON,
        week=_WEEK,
        game_date=game_date,
        home_team=home,
        away_team=away,
        status="scheduled",
        wp_prob=0.6,
        ats_prediction=3.0,
        ou_prediction=44.0,
    )
    return row


def _build_state_cache(db_path: Path, state: str) -> None:
    """Build a cache in which /bets?season=2023&week=1 makes the render named *state*.

    Each state is built the way tests/api/test_bets_page.py builds it, from the same
    materializers production calls, so the parity below is checked against real cache shapes.
    """
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            if statement.strip():
                conn.execute(statement)
        stamped = datetime(2023, 9, 7, 21, 0, 0)
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            [
                [
                    bet_list_populated_at_key(_SEASON, _WEEK),
                    "2023-09-07T21:00:00+00:00",
                    stamped,
                ],
                ["last_updated", stamped.isoformat(), stamped],
            ],
        )
        if state == "no_week":
            return
        materialize_available_bet_weeks(
            conn, pd.DataFrame([{"game_id": _GAME, "season": _SEASON, "week": _WEEK}])
        )
        lock = _FUTURE_LOCK if state == "not_evaluated" else _PAST_LOCK
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": _GAME,
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": lock,
                    }
                ]
            ),
        )
        if state == "not_built":
            conn.execute("DROP TABLE bet_list")
        elif state == "none_cleared":
            materialize_bet_list(conn, pd.DataFrame([_bet_row("suppressed")]))
        elif state == "bets":
            materialize_bet_list(conn, pd.DataFrame([_bet_row("live")]))
    finally:
        conn.close()


@contextmanager
def _serving(db_path: Path) -> Iterator[tuple[TestClient, duckdb.DuckDBPyConnection]]:
    """A TestClient serving *db_path*, and the read-only connection it serves through."""
    from api.dependencies import get_db
    from api.main import app

    clear_cache()
    conn = duckdb.connect(str(db_path), read_only=True)
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = lambda: conn
    try:
        yield TestClient(app, raise_server_exceptions=False), conn
    finally:
        app.dependency_overrides.clear()
        conn.close()


# ---------------------------------------------------------------------------
# The headliner state is the /bets render, never a second decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", HEADLINER_STATES)
def test_the_headliner_names_the_render_bets_makes_for_the_same_week(
    tmp_path: Path, state: str
) -> None:
    db_path = tmp_path / f"{state}.duckdb"
    _build_state_cache(db_path, state)

    with _serving(db_path) as (client, conn):
        bets_response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")
        bets_html = bets_response.text
        clear_cache()
        headliner = _build_headliner(DataService(conn), _SEASON, _WEEK, _REQUEST)

    # A 500 page carries none of the markers and would otherwise be classified as "bets".
    assert bets_response.status_code == 200
    assert _rendered_state(bets_html) == state, (
        f"the fixture for {state!r} did not make /bets render that state; the parity below "
        "would compare the wrong things"
    )
    assert headliner["state"] == state, (
        f"/bets renders {state!r} for this week but the This Week headliner says "
        f"{headliner['state']!r}"
    )
    assert headliner["season"] == _SEASON and headliner["week"] == _WEEK


_QUIET: dict[str, Any] = {
    "bet_list_available": True,
    "bets_blocked": False,
    "week_not_evaluated": False,
    "current_week": _WEEK,
    "bets": [],
}


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        # Every flag at once: the missing TABLE wins, as in pages/bets.html.
        (
            {
                "bet_list_available": False,
                "bets_blocked": True,
                "week_not_evaluated": True,
            },
            "not_built",
        ),
        ({"bets_blocked": True, "week_not_evaluated": True}, "blocked"),
        ({"week_not_evaluated": True, "current_week": None}, "not_evaluated"),
        ({"current_week": None, "bets": [{"game_id": _GAME}]}, "no_week"),
        ({}, "none_cleared"),
        ({"bets": [{"game_id": _GAME}]}, "bets"),
    ],
)
def test_the_state_follows_the_bets_page_precedence(
    flags: dict[str, Any], expected: str
) -> None:
    assert _headliner_state({**_QUIET, **flags}) == expected


def test_a_week_bets_would_not_show_has_no_list_of_its_own(tmp_path: Path) -> None:
    """A predictions week with no schedule row resolves to ANOTHER week on /bets.

    Its bets are not this week's, so the headliner says there is no list rather than borrowing
    the other week's.
    """
    db_path = tmp_path / "mismatch.duckdb"
    _build_state_cache(db_path, "bets")

    with _serving(db_path) as (_client, conn):
        headliner = _build_headliner(DataService(conn), _SEASON, _WEEK + 1, _REQUEST)

    assert headliner == {
        "state": "no_week",
        "partial": False,
        "bets": [],
        "season": _SEASON,
        "week": _WEEK + 1,
    }


# ---------------------------------------------------------------------------
# The decorated slate the game grid renders from
# ---------------------------------------------------------------------------


def test_the_grid_context_decorates_marks_bets_and_groups_only_a_time_order(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "grid.duckdb"
    _build_state_cache(db_path, "bets")
    games = [
        _prediction(_GAME, datetime(2023, 9, 7, 20, 20)),  # Thursday night
        _prediction(_SECOND_GAME, datetime(2023, 9, 10, 13, 0)),  # Sunday 1:00
    ]

    with _serving(db_path) as (_client, conn):
        service = DataService(conn)
        by_time = _this_week_grid_context(
            service, games, _SEASON, _WEEK, "time", _REQUEST
        )
        by_edge = _this_week_grid_context(
            service, games, _SEASON, _WEEK, "edge", _REQUEST
        )
        by_band = _this_week_grid_context(
            service, games, _SEASON, _WEEK, "confidence", _REQUEST
        )
        unknown = _this_week_grid_context(
            service, games, _SEASON, _WEEK, "zzz", _REQUEST
        )

    assert [g["bet_targets"] for g in by_time["games"]] == [["ou"], []]
    assert all("home_color" in g and "kickoff_label" in g for g in by_time["games"])
    assert "home_color" not in games[0], "the source rows were mutated"

    assert by_time["slate_groups"] is not None
    assert [len(group["games"]) for group in by_time["slate_groups"]] == [1, 1]
    assert by_edge["slate_groups"] is None
    assert by_band["slate_groups"] is None
    # get_predictions orders an unknown sort by kickoff, so it is a time order and keeps groups.
    assert unknown["slate_groups"] is not None

    assert by_time["headliner"]["state"] == "bets"
    assert by_time["headliner"]["bets"][0]["game"]["game_id"] == _GAME


# ---------------------------------------------------------------------------
# Partial weeks and the HTMX fragment
# ---------------------------------------------------------------------------


def _rebuild_freeze(db_path: Path, locks: dict[str, datetime]) -> None:
    """Replace the week's per-game locks, so a built game can sit beside a missing or pending one."""
    conn = duckdb.connect(str(db_path))
    try:
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": gid,
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": ts,
                    }
                    for gid, ts in locks.items()
                ]
            ),
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("second_lock", "expected_partial", "marker"),
    [
        (
            _PAST_LOCK,
            True,
            "CAR @ ATL",
        ),  # locked with no list: named as missing on /bets
        (
            _FUTURE_LOCK,
            True,
            "Not evaluated yet:",
        ),  # lock still ahead: the pending line
    ],
)
def test_a_partly_built_week_is_flagged_partial_as_bets_discloses_it(
    tmp_path: Path, second_lock: datetime, expected_partial: bool, marker: str
) -> None:
    db_path = tmp_path / "partial.duckdb"
    _build_state_cache(db_path, "bets")
    _rebuild_freeze(db_path, {_GAME: _PAST_LOCK, _SECOND_GAME: second_lock})

    with _serving(db_path) as (client, conn):
        bets_response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")
        clear_cache()
        headliner = _build_headliner(DataService(conn), _SEASON, _WEEK, _REQUEST)

    assert bets_response.status_code == 200
    assert marker in bets_response.text, (
        "the fixture did not make /bets disclose the partial week"
    )
    assert headliner["state"] == "bets"
    assert headliner["partial"] is expected_partial


def test_a_fully_built_week_is_not_partial(tmp_path: Path) -> None:
    db_path = tmp_path / "whole.duckdb"
    _build_state_cache(db_path, "bets")

    with _serving(db_path) as (_client, conn):
        headliner = _build_headliner(DataService(conn), _SEASON, _WEEK, _REQUEST)

    assert headliner["partial"] is False


def test_the_fragment_without_a_season_resolves_the_same_headliner_as_the_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A single-season cache draws no season <select>, so HTMX sends a week with no season."""
    from api.routes import fragments, pages

    db_path = tmp_path / "single_season.duckdb"
    _build_state_cache(db_path, "bets")
    conn = duckdb.connect(str(db_path))
    try:
        columns = ", ".join(PREDICTIONS_TABLE_COLUMNS)
        marks = ", ".join("?" for _ in PREDICTIONS_TABLE_COLUMNS)
        row = _prediction(_GAME, datetime(2023, 9, 7, 20, 20))
        conn.execute(
            f"INSERT INTO predictions ({columns}) VALUES ({marks})",
            [row[c] for c in PREDICTIONS_TABLE_COLUMNS],
        )
    finally:
        conn.close()

    seen: dict[str, dict[str, Any]] = {}

    def _spy(module: Any, name: str) -> None:
        original = module._this_week_grid_context

        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            seen[name] = original(*args, **kwargs)
            return seen[name]

        monkeypatch.setattr(module, "_this_week_grid_context", wrapper)

    _spy(pages, "page")
    _spy(fragments, "fragment")

    with _serving(db_path) as (client, _conn):
        assert client.get(f"/?week={_WEEK}").status_code == 200
        assert client.get(f"/fragments/games?week={_WEEK}").status_code == 200

    assert seen["page"]["headliner"]["state"] == "bets"
    assert seen["fragment"]["headliner"] == seen["page"]["headliner"]
    assert seen["fragment"]["games"][0]["bet_targets"] == ["ou"]
