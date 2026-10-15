"""The /bets "last week: actual vs expected" block (Plan 35-06, SPEC R5, D-08).

The block reads one stored ``bet_week_results`` row per counting class for the week before the one
the page shows, and every string it renders comes from ``api.presentation.last_week_display`` --
the one formatter the weekly email calls too, so the two surfaces cannot format differently.

Each test builds its own ``tmp_path`` cache; nothing under ``data/`` is read.

Run this module:  uv run python -m pytest tests/api/test_bets_last_week_block.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from fastapi.testclient import TestClient

from api.cache import (
    BET_LIST_COLUMNS,
    BET_WEEK_RESULTS_COLUMNS,
    CACHE_SCHEMA,
    GRADING_STATUS_PENDING,
    bet_list_populated_at_key,
    materialize_available_bet_weeks,
    materialize_bet_list,
    materialize_bet_week_freeze,
    materialize_bet_week_results,
)
from api.presentation import percent_of_stake
from api.services import clear_cache

_SEASON = 2026
_LAST_WEEK = 5
_THIS_WEEK = 6
_FORWARD = "forward_realized"
_STAMPED_AT = datetime(2026, 10, 14, 22, 30, 0)
_SNAPSHOT_TS = "2026-10-17T17:00:00-04:00"

_R5_STRINGS = (
    "2-1-1 (W-L-P)",
    "Actual return: +32.7% of stake",
    "Expected return: +5.0% of stake",
    "Not yet graded: 0",
    "Counted toward the verdict",
    "One week is far too few bets to judge the model.",
)


def _week_row(
    week: int,
    *,
    season: int = _SEASON,
    counting_class: str = _FORWARD,
    record: tuple[int, int, int] = (2, 1, 1),
    graded: int = 4,
    ungraded: int = 0,
    actual: float | None = 0.32725,
    expected: float | None = 0.05,
) -> dict[str, Any]:
    """One stored bet_week_results row; the defaults are the SPEC R5 worked example."""
    wins, losses, pushes = record
    values = [
        season,
        week,
        counting_class,
        wins,
        losses,
        pushes,
        graded,
        ungraded,
        actual,
        expected,
    ]
    return dict(zip(BET_WEEK_RESULTS_COLUMNS, values, strict=True))


def _live_bet(game_id: str, week: int) -> dict[str, Any]:
    """One live, still-pending forward bet_list row: the week has a list, so it is not blocked."""
    row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": game_id,
            "season": _SEASON,
            "week": week,
            "target": "ou",
            "bet_side": "under",
            "line": 47.5,
            "per_bet_ev": 0.0625,
            "stake_units": 1.5,
            "ev_tier": "high",
            "status": "live",
            "snapshot_ts": _SNAPSHOT_TS,
            "freeze_ts": _SNAPSHOT_TS,
            "selected_odds": -110.0,
            "flat_stake": 1.0,
            "provenance": "forward",
            "validation_type": _FORWARD,
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
        }
    )
    return row


def _client(db_path: Path) -> Iterator[TestClient]:
    from api.dependencies import get_db
    from api.main import app

    conn = duckdb.connect(str(db_path), read_only=True)
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = lambda: conn
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        conn.close()


def _serve(
    tmp_path: Path,
    name: str,
    *,
    week_rows: list[dict[str, Any]],
    page_week: int = _THIS_WEEK,
    bet_weeks: tuple[int, ...] | None = None,
    locked_at: datetime | None = None,
) -> Any:
    """Build a cache holding *week_rows* and a schedule, and return a client context manager.

    The page's week is scheduled and carries a live bet in every week of *bet_weeks* (the page's
    week by default), so it renders normally. With *locked_at* every scheduled game of the page's
    week locks then; a week whose games have locked and none has a bet row is the hard block.
    """
    clear_cache()
    db_path = tmp_path / f"{name}.duckdb"
    weeks_with_bets = (page_week,) if bet_weeks is None else bet_weeks
    schedule = [
        {"game_id": f"{_SEASON}_W{week:02d}_AAA@BBB", "season": _SEASON, "week": week}
        for week in sorted({_LAST_WEEK, page_week})
    ]
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            if statement.strip():
                conn.execute(statement.strip())
        bets = [
            _live_bet(game["game_id"], game["week"])
            for game in schedule
            if game["week"] in weeks_with_bets
        ]
        if bets:
            materialize_bet_list(conn, pd.DataFrame(bets))
        materialize_available_bet_weeks(conn, pd.DataFrame(schedule))
        materialize_bet_week_results(conn, pd.DataFrame(week_rows))
        if locked_at is not None:
            page_games = [game for game in schedule if game["week"] == page_week]
            materialize_bet_week_freeze(
                conn,
                pd.DataFrame([{**g, "game_freeze_ts": locked_at} for g in page_games]),
            )
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            [
                [
                    bet_list_populated_at_key(_SEASON, page_week),
                    _STAMPED_AT.isoformat(),
                    _STAMPED_AT,
                ],
                ["last_updated", _STAMPED_AT.isoformat(), _STAMPED_AT],
            ],
        )
    finally:
        conn.close()
    return contextmanager(_client)(db_path)


def _get(client: TestClient, week: int) -> str:
    response = client.get(f"/bets?season={_SEASON}&week={week}")
    assert response.status_code == 200
    return response.text


def _block(body: str) -> str:
    """The last-week block's markup, from its opening hook to the close of its section."""
    start = body.index("data-last-week-block")
    return body[start : body.index("</section>", start)]


# ---------------------------------------------------------------------------
# Task 1 (tracer): a stored week row reaches the top of the forward record as D-08's strings
# ---------------------------------------------------------------------------


def test_the_r5_week_renders_the_d08_strings(tmp_path: Path) -> None:
    with _serve(tmp_path, "r5", week_rows=[_week_row(_LAST_WEEK)]) as client:
        block = _block(_get(client, _THIS_WEEK))

    for text in _R5_STRINGS:
        assert text in block, f"the last-week block is missing {text!r}"
    assert f"Last week -- {_SEASON} week {_LAST_WEEK}" in block


def test_the_block_sits_above_the_season_totals(tmp_path: Path) -> None:
    with _serve(tmp_path, "placement", week_rows=[_week_row(_LAST_WEEK)]) as client:
        body = _get(client, _THIS_WEEK)

    block_at = body.index("data-last-week-block")
    assert block_at > body.rindex('data-tracker-block="backtest_replay'), (
        "the last-week block is not in the forward record, below the replay sections"
    )
    assert block_at < body.index('data-tracker-block="forward:forward_realized"'), (
        "the last-week block is not above the season-to-date forward totals"
    )


def test_percent_of_stake_formats_with_one_decimal_and_a_sign() -> None:
    assert percent_of_stake(0.32725) == "+32.7% of stake"
    assert percent_of_stake(0.05) == "+5.0% of stake"
    assert percent_of_stake(-0.1) == "-10.0% of stake"
    assert percent_of_stake(None) == "not measured"
    assert percent_of_stake(float("nan")) == "not measured"
