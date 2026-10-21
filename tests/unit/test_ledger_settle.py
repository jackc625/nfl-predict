"""Grading ledger rows from refreshed silver scores (Phase 34, Plan 34-09 Task 1; LDGR-01).

``forward_ledger.settle`` grades a pending live ledger row through the EXISTING one-way grader
(``backtest.weekly_bet_list.grade_row``) under the row's own target strategy, from realized values
computed out of silver ``games`` scores with the gold label formulas exactly. These tests drive
that against synthetic rows and a synthetic games frame under ``tmp_path`` only -- nothing here
reads or writes the repository's ``ledger/``, ``data/`` or ``outputs/`` (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import copy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from backtest.selector_strategies import default_strategies
from backtest.weekly_bet_list import grade_row
from forward_ledger.settle import (
    grading_updates,
    load_silver_games,
    realized_values_from_scores,
    regrade_row,
)
from forward_ledger.store import append_rows, apply_updates, read_entries
from forward_ledger.verify import settled_result_check
from tests.unit.test_forward_ledger_store import graded, key_of, make_row

GRADED_AT = datetime(2026, 10, 19, 21, 0, tzinfo=UTC)
GRADED_AT_TEXT = "2026-10-19T21:00:00+00:00"

# Real strategies (the production grade() rules); the fit parameters only matter for pricing,
# never for grading, so fixed synthetic values keep the tests off the gitignored run record.
STRATEGIES: dict[str, Any] = {
    strategy.target: strategy
    for strategy in default_strategies(
        ou_frozen_sd=13.5,
        ou_season_bias_by_season={},
        ats_frozen_sd=13.0,
        ats_season_bias_by_season={},
    )
}

_GRADING_SIX = (
    "grading_status",
    "outcome",
    "clv",
    "payout_flat",
    "realized_units",
    "graded_at",
)


def games_frame(*games: tuple[str, int, float | None, float | None]) -> pd.DataFrame:
    """A silver-``games``-shaped frame: ``(game_id, week, home_score, away_score)`` per game."""
    return pd.DataFrame(
        {
            "game_id": [game[0] for game in games],
            "season": [2026] * len(games),
            "week": [game[1] for game in games],
            "home_score": pd.Series([game[2] for game in games], dtype="float64"),
            "away_score": pd.Series([game[3] for game in games], dtype="float64"),
        }
    )


def ats_row(game_id: str, week: int = 6, **overrides: Any) -> dict[str, Any]:
    """A live ATS ledger row betting the home cover at a slipped -3.0, priced -110."""
    return make_row(
        game_id=game_id,
        week=week,
        target="ats",
        bet_side="home_cover",
        slipped_line=-3.0,
        **overrides,
    )


def test_realized_values_formulas() -> None:
    games = games_frame(
        ("2026_W06_KC@BUF", 6, 24.0, 17.0),
        ("2026_W06_DAL@PHI", 6, 20.0, 20.0),
        ("2026_W06_SF@SEA", 6, None, None),
    )

    realized = realized_values_from_scores(games)

    assert realized["wp"]["2026_W06_KC@BUF"] == 1.0
    assert realized["ats"]["2026_W06_KC@BUF"] == 7.0
    assert realized["ou"]["2026_W06_KC@BUF"] == 41.0
    # A tie is NOT a home win: the gold label is strict binary with ties 0.
    assert realized["wp"]["2026_W06_DAL@PHI"] == 0.0
    assert realized["ats"]["2026_W06_DAL@PHI"] == 0.0
    assert realized["ou"]["2026_W06_DAL@PHI"] == 40.0
    for target in ("wp", "ats", "ou"):
        assert "2026_W06_SF@SEA" not in realized[target]


def test_grades_only_live_pending_rows_with_results(tmp_path: Path) -> None:
    to_grade = ats_row("2026_W06_KC@BUF")
    suppressed = ats_row(
        "2026_W06_DAL@PHI", status="suppressed", rejection_reason="below_ev_floor"
    )
    settled = ats_row("2026_W06_SF@SEA")
    no_result = ats_row("2026_W06_NYJ@MIA")
    append_rows(tmp_path, [to_grade, suppressed, settled, no_result])
    apply_updates(tmp_path, grading_updates={key_of(settled): graded("win", 0.9, 1.1)})
    games = games_frame(
        ("2026_W06_KC@BUF", 6, 24.0, 17.0),
        ("2026_W06_DAL@PHI", 6, 24.0, 17.0),
        ("2026_W06_SF@SEA", 6, 3.0, 30.0),
        ("2026_W06_NYJ@MIA", 6, None, None),
    )

    updates = grading_updates(
        read_entries(tmp_path),
        STRATEGIES,
        realized_values_from_scores(games),
        GRADED_AT,
    )

    assert list(updates) == [key_of(to_grade)]


def test_grading_matches_grade_row_exactly(tmp_path: Path) -> None:
    row = ats_row("2026_W06_KC@BUF")
    append_rows(tmp_path, [row])
    entries = read_entries(tmp_path)
    realized = realized_values_from_scores(
        games_frame(("2026_W06_KC@BUF", 6, 24.0, 17.0))
    )

    update = grading_updates(entries, STRATEGIES, realized, GRADED_AT)[key_of(row)]

    stored = {**entries[0].immutable, **(entries[0].grading or {})}
    outcome = STRATEGIES["ats"].grade(
        {"bet_side": "home_cover", "slipped_line": -3.0, "_actual_total": 7.0}
    )
    expected = grade_row(stored, outcome, graded_at=GRADED_AT)
    assert update == {
        **{name: expected[name] for name in _GRADING_SIX},
        "graded_at": GRADED_AT_TEXT,
    }
    assert update["grading_status"] == "win"
    assert update["outcome"] is True
    assert update["clv"] == stored["clv"]


def test_grading_updates_apply_without_hash_change(tmp_path: Path) -> None:
    rows = [ats_row("2026_W06_KC@BUF"), ats_row("2026_W06_SF@SEA")]
    append_rows(tmp_path, rows)
    before = read_entries(tmp_path)
    realized = realized_values_from_scores(
        games_frame(
            ("2026_W06_KC@BUF", 6, 24.0, 17.0), ("2026_W06_SF@SEA", 6, 3.0, 30.0)
        )
    )

    result = apply_updates(
        tmp_path,
        grading_updates=grading_updates(before, STRATEGIES, realized, GRADED_AT),
    )

    after = read_entries(tmp_path)
    assert result.grading_changed == 2
    assert [entry.chain_hash for entry in after] == [
        entry.chain_hash for entry in before
    ]
    assert [entry.immutable for entry in after] == [entry.immutable for entry in before]
    assert [(entry.grading or {})["grading_status"] for entry in after] == [
        "win",
        "loss",
    ]

    # Idempotent: a re-run finds nothing pending, so it proposes and writes nothing.
    again = grading_updates(after, STRATEGIES, realized, GRADED_AT)
    assert again == {}
    assert apply_updates(tmp_path, grading_updates=again).wrote is False


def test_week_22_is_graded_without_a_gold_build(tmp_path: Path) -> None:
    ledger_dir = tmp_path / "ledger"
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    super_bowl = ats_row("2026_W22_KC@SF", week=22)
    append_rows(ledger_dir, [super_bowl])
    games_frame(("2026_W22_KC@SF", 22, 31.0, 20.0)).to_parquet(
        silver_dir / "games.parquet"
    )

    realized = realized_values_from_scores(load_silver_games(silver_dir))
    apply_updates(
        ledger_dir,
        grading_updates=grading_updates(
            read_entries(ledger_dir), STRATEGIES, realized, GRADED_AT
        ),
    )

    entry = read_entries(ledger_dir)[0]
    assert entry.immutable["week"] == 22
    assert (entry.grading or {})["grading_status"] == "win"
    assert not (tmp_path / "gold").exists()


def test_regrade_row_reads_a_settled_row_without_touching_it() -> None:
    settled = {
        **ats_row("2026_W06_KC@BUF"),
        "grading_status": "win",
        "outcome": True,
        "clv": 0.25,
        "payout_flat": 0.9090909090909091,
        "realized_units": 1.1363636363636365,
        "graded_at": "2026-10-19T21:00:00+00:00",
    }
    snapshot = copy.deepcopy(settled)

    values = regrade_row(
        settled, STRATEGIES["ats"], -10.0, graded_at=GRADED_AT, tied=False
    )

    reset = {
        **snapshot,
        "grading_status": "pending",
        "outcome": None,
        "payout_flat": None,
        "realized_units": None,
        "graded_at": None,
    }
    outcome = STRATEGIES["ats"].grade(
        {"bet_side": "home_cover", "slipped_line": -3.0, "_actual_total": -10.0}
    )
    expected = grade_row(reset, outcome, graded_at=GRADED_AT)
    assert values == {name: expected[name] for name in _GRADING_SIX}
    assert values["grading_status"] == "loss"
    assert values["clv"] == 0.25
    assert settled == snapshot


AWAY_TIE_GAME = "2026_W06_DAL@PHI"
HOME_TIE_GAME = "2026_W06_CHI@GB"


def tied_ledger(
    ledger_dir: Path,
) -> tuple[list[dict[str, Any]], pd.DataFrame, dict[tuple[Any, ...], dict[str, Any]]]:
    """Two tied games holding one live WP row each (away and home backed) and one ATS row.

    LEDGER_ROW_KEY lets one game hold one live WP row, so the two moneyline sides sit on two
    different tied games. Returns ``(rows, games, the grading updates the settle pass proposes)``.
    """
    away_wp = make_row(
        game_id=AWAY_TIE_GAME,
        target="wp",
        bet_side="away",
        selected_odds=120.0,
        stake_units=1.0,
    )
    home_wp = make_row(
        game_id=HOME_TIE_GAME,
        target="wp",
        bet_side="home",
        selected_odds=-140.0,
        stake_units=1.0,
    )
    ats_control = ats_row(AWAY_TIE_GAME)
    rows = [away_wp, home_wp, ats_control]
    append_rows(ledger_dir, rows)
    games = games_frame((AWAY_TIE_GAME, 6, 20.0, 20.0), (HOME_TIE_GAME, 6, 13.0, 13.0))
    updates = grading_updates(
        read_entries(ledger_dir),
        STRATEGIES,
        realized_values_from_scores(games),
        GRADED_AT,
    )
    return rows, games, updates


def test_a_tied_game_grades_a_moneyline_bet_as_a_push(tmp_path: Path) -> None:
    (away_wp, home_wp, ats_control), games, updates = tied_ledger(tmp_path)

    # Whichever side it backed, a moneyline bet on a tied game returns the stake (D-21).
    for row in (away_wp, home_wp):
        update = updates[key_of(row)]
        assert update["grading_status"] == "push"
        assert update["outcome"] is None
        assert update["payout_flat"] == 0.0
        assert update["realized_units"] == 0.0

    # The ATS control on the same tied game is graded exactly as before: margin 0 against -3.0.
    ats_entry = read_entries(tmp_path)[2]
    stored = {**ats_entry.immutable, **(ats_entry.grading or {})}
    outcome = STRATEGIES["ats"].grade(
        {"bet_side": "home_cover", "slipped_line": -3.0, "_actual_total": 0.0}
    )
    expected = grade_row(stored, outcome, graded_at=GRADED_AT)
    assert updates[key_of(ats_control)] == {
        **{name: expected[name] for name in _GRADING_SIX},
        "graded_at": GRADED_AT_TEXT,
    }

    # The WP label the frozen re-fit comparison reads is untouched: a tie still scores 0.
    realized = realized_values_from_scores(games)
    assert realized["wp"][AWAY_TIE_GAME] == 0.0
    assert realized["wp"][HOME_TIE_GAME] == 0.0


def test_every_grade_path_agrees_on_a_tied_moneyline_bet(tmp_path: Path) -> None:
    (away_wp, home_wp, _), games, updates = tied_ledger(tmp_path)
    apply_updates(tmp_path, grading_updates=updates)

    entries = read_entries(tmp_path)
    statuses = {
        key_of(entry.immutable): (entry.grading or {})["grading_status"]
        for entry in entries
    }
    assert statuses[key_of(away_wp)] == "push"
    assert statuses[key_of(home_wp)] == "push"

    # The verify CLI's settled-result check re-grades the tie exactly as the settle pass graded it.
    checked, mismatches = settled_result_check(entries, games, STRATEGIES)
    assert checked == 3
    assert mismatches == []


def test_load_silver_games_reads_the_store(tmp_path: Path) -> None:
    silver_dir = tmp_path / "silver"
    silver_dir.mkdir()
    frame = games_frame(("2026_W06_KC@BUF", 6, 24.0, 17.0))
    frame.to_parquet(silver_dir / "games.parquet")

    pd.testing.assert_frame_equal(load_silver_games(silver_dir), frame)

    absent = tmp_path / "nowhere"
    with pytest.raises(FileNotFoundError, match=r"games\.parquet"):
        load_silver_games(absent)
