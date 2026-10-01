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


# ---------------------------------------------------------------------------
# The rendered page: headliners first, then the slate by TV window (Task 8)
# ---------------------------------------------------------------------------

_WEEK_GAMES = [
    _prediction(_GAME, datetime(2023, 9, 7, 20, 20)),  # Thursday night
    _prediction(_SECOND_GAME, datetime(2023, 9, 10, 13, 0)),  # Sunday 1:00
]


def _insert_predictions(db_path: Path, rows: list[dict[str, Any]]) -> None:
    """Add prediction rows to a cache built by _build_state_cache, so / has games to show."""
    conn = duckdb.connect(str(db_path))
    try:
        columns = ", ".join(PREDICTIONS_TABLE_COLUMNS)
        placeholders = ", ".join("?" for _ in PREDICTIONS_TABLE_COLUMNS)
        conn.executemany(
            f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
            [[row[c] for c in PREDICTIONS_TABLE_COLUMNS] for row in rows],
        )
    finally:
        conn.close()


def _page(tmp_path: Path, state: str, path: str, *, name: str = "page") -> str:
    """Build a fresh cache for *state*, serve it, and return the body of GET *path*.

    A test that renders twice passes a distinct *name*: building onto a file that already holds
    the week's predictions would insert the same game ids again and hit the predictions table's
    primary key.
    """
    db_path = tmp_path / f"{name}_{state}.duckdb"
    _build_state_cache(db_path, state)
    _insert_predictions(db_path, _WEEK_GAMES)
    with _serving(db_path) as (client, _conn):
        response = client.get(path)
    assert response.status_code == 200
    return response.text


def test_this_week_leads_with_the_weeks_bets(tmp_path: Path) -> None:
    html = _page(tmp_path, "bets", f"/?season={_SEASON}&week={_WEEK}")

    assert 'data-headliner-state="bets"' in html
    headliners = html[html.index('id="headliners"') : html.index("data-window=")]
    assert f'data-headliner-bet="{_GAME}:ou"' in headliners
    assert "Under 47.5" in headliners, "the pick is not worded the way /bets words it"
    assert "+6.25%" in headliners and "1.50u" in headliners
    assert html.index('id="headliners"') < html.index("game-card"), (
        "the bets must come before the slate"
    )
    assert html.count("data-on-bet-list") == 1, (
        "only the bet's own game carries the flag"
    )


@pytest.mark.parametrize("state", [s for s in HEADLINER_STATES if s != "bets"])
def test_every_other_state_is_one_line_pointing_to_bets(
    tmp_path: Path, state: str
) -> None:
    html = _page(tmp_path, state, f"/?season={_SEASON}&week={_WEEK}")

    assert f'data-headliner-state="{state}"' in html
    assert "data-headliner-note" in html
    assert "data-headliner-bet" not in html
    assert 'href="/bets' in html[html.index('id="headliners"') :]


def test_a_week_change_re_renders_the_headliners_with_the_slate(tmp_path: Path) -> None:
    html = _page(tmp_path, "bets", f"/fragments/games?season={_SEASON}&week={_WEEK}")

    assert 'data-headliner-state="bets"' in html
    assert "<nav" not in html and "<html" not in html


def test_the_default_sort_groups_by_tv_window_and_a_non_time_sort_does_not(
    tmp_path: Path,
) -> None:
    grouped = _page(
        tmp_path, "bets", f"/?season={_SEASON}&week={_WEEK}", name="grouped"
    )
    assert grouped.count("data-window=") == 2, (
        "Thursday night and Sunday 1:00 are two windows"
    )

    by_edge = _page(
        tmp_path, "bets", f"/?season={_SEASON}&week={_WEEK}&sort=edge", name="by_edge"
    )
    assert "data-window=" not in by_edge
    assert by_edge.count("game-card") == grouped.count("game-card")


def test_the_page_keeps_one_old_rule_label_and_the_not_advice_note(
    tmp_path: Path,
) -> None:
    """2023 is an old-rule season: the headliners sit in the game_grid block's ONE label."""
    html = _page(tmp_path, "bets", f"/?season={_SEASON}&week={_WEEK}")

    assert html.count("data-old-rule-label") == 1
    assert "Not wagering advice" in html


def test_a_game_with_no_kickoff_time_renders_under_its_own_tag() -> None:
    """Review Focus 3: a game whose kickoff is unknown still renders, under "Time TBD"."""
    from api.dependencies import templates
    from api.presentation import decorate_game, group_games_by_window

    game = {**decorate_game(_prediction(_GAME, None)), "bet_targets": []}
    context = {
        "request": _StubRequest(),
        "cache_meta": {},
        "games": [game],
        "slate_groups": group_games_by_window([game]),
        "available_weeks": [{"season": _SEASON, "week": _WEEK}],
        "available_seasons": [_SEASON],
        "current_week": _WEEK,
        "current_season": _SEASON,
        "current_sort": "time",
        "current_path": "/",
        "week_summary": {},
        "old_rule_scope": DataService.old_rule_scope([]),
    }
    html = templates.env.get_template("pages/this_week.html").render(context)

    assert 'data-window="Time TBD"' in html
    assert html.count("game-card") == 1
    assert "Time TBD" in html.split('data-window="Time TBD"', 1)[1]


@pytest.mark.parametrize(
    ("second_lock", "partial"),
    [(_PAST_LOCK, True), (_FUTURE_LOCK, True), (None, False)],
)
def test_a_partly_built_week_says_so_and_points_to_bets(
    tmp_path: Path, second_lock: datetime | None, partial: bool
) -> None:
    db_path = tmp_path / "partial_page.duckdb"
    _build_state_cache(db_path, "bets")
    if second_lock is not None:
        _rebuild_freeze(db_path, {_GAME: _PAST_LOCK, _SECOND_GAME: second_lock})
    _insert_predictions(db_path, _WEEK_GAMES)
    with _serving(db_path) as (client, _conn):
        html = client.get(f"/?season={_SEASON}&week={_WEEK}").text

    headliners = html[html.index('id="headliners"') : html.index("data-window=")]
    assert ("data-headliner-partial" in headliners) is partial
    if partial:
        line = headliners[headliners.index("data-headliner-partial") :]
        assert (
            "only partly built" in line
            and 'href="/bets?season=2023&amp;week=1"' in line
        )
