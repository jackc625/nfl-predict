"""Last week's actual versus expected: the per-week aggregation, its cache row and its getter.

Phase 35, plan 35-03 (SPEC R5, NOTIF-05, UIAP-01). ``backtest/bet_week_results.py`` aggregates the
graded, live-status, live-arm forward rows of each NFL week at CACHE BUILD, reusing the tracker's
correction overlay and flat-return formula. ``api/cache.py`` only persists the rows and
``DataService.get_bet_week_results`` only selects them, so ``/bets`` and the weekly email read a
stored figure and never compute one.

The R5 fixture is the SPEC's worked example: two wins paying +0.909 and +1.400, one loss (-1.0) and
one push (0.0) at a flat stake of 1, with per-bet EV 0.05 / 0.08 / 0.04 / 0.03 -- W-L-P 2-1-1,
actual return 1.309 / 4 = 0.32725 and expected return 0.20 / 4 = 0.05 per unit staked.

Run this module:  uv run python -m pytest tests/unit/test_bet_week_results.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import duckdb
import pandas as pd
import pytest

import api.services
import backtest.bet_week_results
from api.cache import (
    ARM_LIVE,
    ARM_SHADOW,
    BET_LIST_COLUMNS,
    BET_LIST_CORRECTIONS_COLUMNS,
    BET_STATUS_LIVE,
    BET_WEEK_CLASS_ORDER,
    BET_WEEK_RESULTS_COLUMNS,
    CACHE_SCHEMA,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    PROVENANCE_BACKTEST_REPLAY,
    PROVENANCE_FORWARD,
    VALIDATION_TYPE_CONTAMINATED,
    VALIDATION_TYPE_FORWARD_REALIZED,
    VALIDATION_TYPE_PRE_VERDICT,
    VERDICT_SCOPE_PRE_VERDICT,
    VERDICT_SCOPE_VERDICT,
    materialize_bet_week_results,
)
from api.services import DataService, clear_cache
from backtest.bet_tracker import TrackerBlock, aggregate_all_blocks
from backtest.bet_week_results import aggregate_bet_weeks

_PAYOUT_AT_MINUS_110 = 0.909
_PAYOUT_AT_PLUS_140 = 1.400

_DEFAULT_PAYOUT: dict[str, float | None] = {
    GRADING_STATUS_WIN: _PAYOUT_AT_MINUS_110,
    GRADING_STATUS_LOSS: -1.0,
    GRADING_STATUS_PUSH: 0.0,
    GRADING_STATUS_PENDING: None,
}
_OUTCOME: dict[str, bool | None] = {
    GRADING_STATUS_WIN: True,
    GRADING_STATUS_LOSS: False,
    GRADING_STATUS_PUSH: None,
    GRADING_STATUS_PENDING: None,
}


@pytest.fixture(autouse=True)
def _isolated_service_cache() -> Iterator[None]:
    """The DataService TTL cache is keyed without the connection: clear it around every test."""
    clear_cache()
    yield
    clear_cache()


def _row(
    game_id: str,
    grading_status: str,
    *,
    per_bet_ev: float | None = 0.05,
    payout_flat: float | None = None,
    season: int = 2026,
    week: int = 5,
    target: str = "ats",
    status: str = BET_STATUS_LIVE,
    arm: str = ARM_LIVE,
    provenance: str = PROVENANCE_FORWARD,
    validation_type: str = VALIDATION_TYPE_FORWARD_REALIZED,
    verdict_scope: str = VERDICT_SCOPE_VERDICT,
) -> dict[str, Any]:
    """One bet-list row carrying every ``BET_LIST_COLUMNS`` field."""
    row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
    payout = _DEFAULT_PAYOUT[grading_status] if payout_flat is None else payout_flat
    row.update(
        {
            "game_id": game_id,
            "season": season,
            "week": week,
            "target": target,
            "bet_side": "home",
            "line": -3.0,
            "per_bet_ev": per_bet_ev,
            "stake_units": 1.0,
            "ev_tier": "medium",
            "status": status,
            "flat_stake": 1.0,
            "provenance": provenance,
            "validation_type": validation_type,
            "arm": arm,
            "verdict_scope": verdict_scope,
            "grading_status": grading_status,
            "outcome": _OUTCOME[grading_status],
            "payout_flat": payout,
            "realized_units": payout,
        }
    )
    return row


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=pd.Index(BET_LIST_COLUMNS))


def _r5_rows(**overrides: Any) -> list[dict[str, Any]]:
    """The SPEC R5 fixture: 2 wins (+0.909, +1.400), 1 loss, 1 push, EV 0.05/0.08/0.04/0.03."""
    return [
        _row("G1", GRADING_STATUS_WIN, per_bet_ev=0.05, **overrides),
        _row(
            "G2",
            GRADING_STATUS_WIN,
            per_bet_ev=0.08,
            payout_flat=_PAYOUT_AT_PLUS_140,
            **overrides,
        ),
        _row("G3", GRADING_STATUS_LOSS, per_bet_ev=0.04, **overrides),
        _row("G4", GRADING_STATUS_PUSH, per_bet_ev=0.03, **overrides),
    ]


def _cache_connection(week_results: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """An in-memory DuckDB built from CACHE_SCHEMA with *week_results* materialized into it."""
    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    materialize_bet_week_results(conn, week_results)
    return conn


def test_the_r5_fixture_gives_2_1_1_and_the_two_returns() -> None:
    result = aggregate_bet_weeks(_frame(_r5_rows()))

    assert list(result.columns) == BET_WEEK_RESULTS_COLUMNS
    assert len(result) == 1
    row = result.iloc[0]
    assert (int(row["season"]), int(row["week"])) == (2026, 5)
    assert row["counting_class"] == VALIDATION_TYPE_FORWARD_REALIZED
    assert (int(row["wins"]), int(row["losses"]), int(row["pushes"])) == (2, 1, 1)
    assert int(row["bets_graded"]) == 4
    assert int(row["bets_ungraded"]) == 0
    assert row["actual_return_units"] == pytest.approx(0.32725)
    assert row["expected_return_units"] == pytest.approx(0.05)


def test_the_row_round_trips_through_the_cache_and_the_getter() -> None:
    conn = _cache_connection(aggregate_bet_weeks(_frame(_r5_rows())))

    rows = DataService(conn).get_bet_week_results(2026, 5)

    assert len(rows) == 1
    row = rows[0]
    assert list(row) == BET_WEEK_RESULTS_COLUMNS
    assert row["counting_class"] == VALIDATION_TYPE_FORWARD_REALIZED
    assert (row["wins"], row["losses"], row["pushes"]) == (2, 1, 1)
    assert (row["bets_graded"], row["bets_ungraded"]) == (4, 0)
    assert row["actual_return_units"] == pytest.approx(0.32725)
    assert row["expected_return_units"] == pytest.approx(0.05)


def test_a_null_ev_makes_expected_not_measured() -> None:
    rows = _r5_rows()
    rows[3]["per_bet_ev"] = None

    row = aggregate_bet_weeks(_frame(rows)).iloc[0]

    assert row["expected_return_units"] is None
    # The actual return does not depend on the EV and is still measured.
    assert row["actual_return_units"] == pytest.approx(0.32725)


def test_an_all_pending_week_stores_nothing_graded() -> None:
    frame = _frame(
        [
            _row("G1", GRADING_STATUS_PENDING),
            _row("G2", GRADING_STATUS_PENDING),
            _row("G3", GRADING_STATUS_PENDING),
        ]
    )

    result = aggregate_bet_weeks(frame)

    assert len(result) == 1
    row = result.iloc[0]
    assert (int(row["bets_graded"]), int(row["bets_ungraded"])) == (0, 3)
    assert (int(row["wins"]), int(row["losses"]), int(row["pushes"])) == (0, 0, 0)
    assert row["actual_return_units"] is None
    assert row["expected_return_units"] is None


def test_shadow_suppressed_and_replay_rows_never_count() -> None:
    frame = _frame(
        [
            _row("G1", GRADING_STATUS_WIN),
            _row("G1", GRADING_STATUS_WIN, arm=ARM_SHADOW),
            _row("G2", GRADING_STATUS_PENDING, status="suppressed"),
            _row(
                "G3",
                GRADING_STATUS_LOSS,
                provenance=PROVENANCE_BACKTEST_REPLAY,
                validation_type=VALIDATION_TYPE_CONTAMINATED,
            ),
            _row(
                "G4",
                GRADING_STATUS_LOSS,
                season=2024,
                provenance=PROVENANCE_BACKTEST_REPLAY,
                validation_type=VALIDATION_TYPE_CONTAMINATED,
            ),
        ]
    )

    result = aggregate_bet_weeks(frame)

    assert len(result) == 1
    row = result.iloc[0]
    assert (int(row["season"]), int(row["week"])) == (2026, 5)
    assert (int(row["wins"]), int(row["losses"]), int(row["pushes"])) == (1, 0, 0)
    assert (int(row["bets_graded"]), int(row["bets_ungraded"])) == (1, 0)
    assert row["actual_return_units"] == pytest.approx(_PAYOUT_AT_MINUS_110)


def test_an_in_force_correction_is_overlaid() -> None:
    frame = _frame([_row("G1", GRADING_STATUS_WIN), _row("G2", GRADING_STATUS_LOSS)])
    correction: dict[str, Any] = {
        "game_id": "G2",
        "season": 2026,
        "week": 5,
        "target": "ats",
        "arm": ARM_LIVE,
        "original_grading_status": GRADING_STATUS_LOSS,
        "corrected_grading_status": GRADING_STATUS_WIN,
        "corrected_outcome": True,
        "corrected_payout_flat": _PAYOUT_AT_MINUS_110,
        "corrected_realized_units": _PAYOUT_AT_MINUS_110,
        "realized_value": -2.5,
        "detected_at_utc": "2026-10-13T12:00:00+00:00",
        "corrected_at_utc": "2026-10-13T21:00:00+00:00",
    }
    corrections = pd.DataFrame(
        [correction], columns=pd.Index(BET_LIST_CORRECTIONS_COLUMNS)
    )

    row = aggregate_bet_weeks(frame, corrections=corrections).iloc[0]
    tracker = aggregate_all_blocks(frame, corrections=corrections)

    assert (int(row["wins"]), int(row["losses"])) == (2, 0)
    assert row["actual_return_units"] == pytest.approx(_PAYOUT_AT_MINUS_110)
    # Exactly as the tracker counts the same rows under the same correction.
    (block,) = tracker
    assert isinstance(block, TrackerBlock)
    assert (int(row["wins"]), int(row["losses"])) == (block.wins, block.losses)
    assert row["actual_return_units"] == block.flat_return_units
    # Without the correction the same rows read 1-1: the overlay is what moved it.
    uncorrected = aggregate_bet_weeks(frame).iloc[0]
    assert (int(uncorrected["wins"]), int(uncorrected["losses"])) == (1, 1)


def test_week_one_has_no_last_week_and_never_reaches_the_prior_season() -> None:
    frame = _frame(
        [
            _row("P1", GRADING_STATUS_WIN, season=2025, week=22),
            _row("N1", GRADING_STATUS_LOSS, season=2026, week=1),
        ]
    )
    conn = _cache_connection(aggregate_bet_weeks(frame))
    service = DataService(conn)

    assert service.get_bet_week_results(2026, 0) == []
    # The prior season's final week exists in the cache and is still not selected.
    assert len(service.get_bet_week_results(2025, 22)) == 1


def test_two_classes_in_one_week_are_two_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    frame = _frame(
        [
            _row("G1", GRADING_STATUS_WIN, verdict_scope=VERDICT_SCOPE_PRE_VERDICT),
            _row("G2", GRADING_STATUS_LOSS, verdict_scope=VERDICT_SCOPE_VERDICT),
        ]
    )
    result = aggregate_bet_weeks(frame)

    assert list(result["counting_class"]) == list(BET_WEEK_CLASS_ORDER)
    by_class = {str(r["counting_class"]): r for _, r in result.iterrows()}
    assert int(by_class[VALIDATION_TYPE_PRE_VERDICT]["wins"]) == 1
    assert int(by_class[VALIDATION_TYPE_PRE_VERDICT]["bets_graded"]) == 1
    assert int(by_class[VALIDATION_TYPE_FORWARD_REALIZED]["losses"]) == 1
    assert int(by_class[VALIDATION_TYPE_FORWARD_REALIZED]["bets_graded"]) == 1

    conn = _cache_connection(result)
    getter_classes = [
        r["counting_class"] for r in DataService(conn).get_bet_week_results(2026, 5)
    ]
    assert getter_classes == list(BET_WEEK_CLASS_ORDER)

    # The order follows the declared constant, not the alphabet.
    reversed_order = tuple(reversed(BET_WEEK_CLASS_ORDER))
    monkeypatch.setattr(api.services, "BET_WEEK_CLASS_ORDER", reversed_order)
    monkeypatch.setattr(
        backtest.bet_week_results, "BET_WEEK_CLASS_ORDER", reversed_order
    )
    clear_cache()

    assert list(aggregate_bet_weeks(frame)["counting_class"]) == list(reversed_order)
    getter_classes = [
        r["counting_class"] for r in DataService(conn).get_bet_week_results(2026, 5)
    ]
    assert getter_classes == list(reversed_order)


def test_a_playoff_week_selects_week_minus_one() -> None:
    frame = _frame(
        [
            _row("W1", GRADING_STATUS_WIN, week=19),
            _row("W2", GRADING_STATUS_LOSS, week=20),
        ]
    )
    conn = _cache_connection(aggregate_bet_weeks(frame))

    rows = DataService(conn).get_bet_week_results(2026, 19)

    assert len(rows) == 1
    assert (rows[0]["season"], rows[0]["week"]) == (2026, 19)
    assert (rows[0]["wins"], rows[0]["losses"]) == (1, 0)
