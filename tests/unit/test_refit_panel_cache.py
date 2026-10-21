"""The comparison panel's raw numbers, materialized at cache build (Plan 37-18; D-12..D-14, SPEC R13).

WHAT IS PROVEN
--------------
* ``refit.comparison.panel_frames`` turns the ledger, the silver scores and the re-fit records into
  two frames: one ``refit_panel`` row per target per period (the stretch before the first re-fit,
  one row per installed generation, and a season-to-date row over every paired game since the
  first re-fit), and one ``refit_aa_status`` row per evaluated A/A week. Every expected number is
  computed BY HAND from the fixture scores.
* An installed generation is a period row from the moment it is installed, even with no completed
  pair yet: paired_games 0 and every error, gap and ``ahead`` NULL ("not measured yet", never 0.0).
* ``forward_ledger.cache_sources.read_bet_cache_sources`` carries both frames, empty with the switch
  off, and never fails because ``games.parquet`` is absent.
* ``api.cache.populate_cache`` writes both tables (always created, also when nothing is supplied),
  and neither table has a test-result column (D-13).

Every store lives under pytest's temporary directories: production ``ledger/``, ``data/`` and
``outputs/`` are never read or written, and nothing touches the network (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from api import cache_schema
from api.cache import populate_cache
from api.cache_reader import open_cache_readonly
from backtest.refit_preregistration_2026 import REFIT_SEASON
from forward_ledger import cutover
from forward_ledger.cache_sources import read_bet_cache_sources
from forward_ledger.settle import load_silver_games
from forward_ledger.store import read_entries
from refit import comparison
from refit.aa import AAWeekResult, record_week
from refit.records import append_date_event
from tests.unit.test_ledger_cache_sources import _replay_rows, _write_outputs
from tests.unit.test_refit_comparison import live_row, shadow_row, write_pairs
from tests.unit.test_refit_decision_path import (
    GENERATION_ONE_IDS,
    closed_event,
    intent_event,
)

SEASON = 2026

# The fixture games and their final scores (home, away).
SCORES: dict[str, tuple[int, int]] = {
    "2026_05_KC_BUF": (21, 17),  # live-only week (no shadow yet)
    "2026_06_KC_BUF": (24, 20),  # live-only week (no shadow yet)
    "2026_07_KC_BUF": (27, 24),  # margin +3
    "2026_08_KC_BUF": (17, 20),  # margin -3
    "2026_09_KC_BUF": (24, 22),  # margin +2, total 46, home win
    "2026_09_DAL_PHI": (10, 17),  # margin -7
}

# Each fixture game's kickoff (Eastern), so the schedule the cache build reads is well formed.
KICKOFFS_ET: dict[str, str] = {
    "2026_05_KC_BUF": "2026-10-11 13:00",
    "2026_06_KC_BUF": "2026-10-18 13:00",
    "2026_07_KC_BUF": "2026-10-25 13:00",
    "2026_08_KC_BUF": "2026-11-01 13:00",
    "2026_09_KC_BUF": "2026-11-08 13:00",
    "2026_09_DAL_PHI": "2026-11-08 16:25",
}

# The names a stored test result could carry; none may appear in either table (D-13).
TEST_RESULT_FRAGMENTS: tuple[str, ...] = (
    "p_value",
    "pvalue",
    "significant",
    "ci",
    "t_stat",
)


def games_frame() -> pd.DataFrame:
    """A silver-``games``-shaped frame holding :data:`SCORES`."""
    return pd.DataFrame(
        {
            "game_id": list(SCORES),
            "season": [SEASON] * len(SCORES),
            "week": [int(game_id[5:7]) for game_id in SCORES],
            "home_score": [home for home, _away in SCORES.values()],
            "away_score": [away for _home, away in SCORES.values()],
            "kickoff_et": pd.to_datetime(
                [KICKOFFS_ET[game_id] for game_id in SCORES]
            ).tz_localize("America/New_York"),
        }
    )


def write_silver(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    games_frame().to_parquet(directory / "games.parquet", index=False)
    return directory


def install_generation_one(ledger: Path) -> None:
    """Generation 1 installed on Tue 2026-11-03, recorded as the re-fit hook records it."""
    append_date_event(ledger, intent_event(GENERATION_ONE_IDS, "a" * 64))
    append_date_event(ledger, closed_event("ran"))


def write_fixture_ledger(ledger: Path) -> None:
    """Live-only weeks 5-6, generation-0 pairs in weeks 7-8, the install, generation-1 week 9.

    ATS, by hand (loss = |value - margin|):
      week 7 (margin +3): both arms 2.0 -> 1.0 / 1.0
      week 8 (margin -3): both arms 1.0 -> 4.0 / 4.0
      -> before the first re-fit: live 2.5, frozen 2.5, gap 0.0, even, 2 games
      week 9 KC_BUF (margin +2): live 3.0 -> 1.0, frozen 1.0 -> 1.0
      week 9 DAL_PHI (margin -7): live -3.5 -> 3.5, frozen 2.5 -> 9.5
      -> period 1: live 2.25, frozen 5.25, gap -3.0, live ahead, 2 games
    WP week 9 KC_BUF (home win): live 0.6, frozen 0.5 -> live ahead.
    O/U week 9 KC_BUF (total 46): live 40.0 -> 6.0, frozen 45.0 -> 1.0 -> frozen ahead.
    """
    write_pairs(
        ledger,
        [
            (live_row("2026_05_KC_BUF", "ats", 9.0, generation=0), None),
            (live_row("2026_06_KC_BUF", "ats", -9.0, generation=0), None),
        ],
    )
    write_pairs(
        ledger,
        [
            (
                live_row("2026_07_KC_BUF", "ats", 2.0, generation=0),
                shadow_row("2026_07_KC_BUF", "ats", 2.0),
            ),
            (
                live_row("2026_08_KC_BUF", "ats", 1.0, generation=0),
                shadow_row("2026_08_KC_BUF", "ats", 1.0),
            ),
        ],
    )
    install_generation_one(ledger)
    write_pairs(
        ledger,
        [
            (
                live_row("2026_09_KC_BUF", "ats", 3.0),
                shadow_row("2026_09_KC_BUF", "ats", 1.0),
            ),
            (
                live_row("2026_09_DAL_PHI", "ats", -3.5),
                shadow_row("2026_09_DAL_PHI", "ats", 2.5),
            ),
            (
                live_row("2026_09_KC_BUF", "wp", 0.6),
                shadow_row("2026_09_KC_BUF", "wp", 0.5),
            ),
            (
                live_row("2026_09_KC_BUF", "ou", 40.0),
                shadow_row("2026_09_KC_BUF", "ou", 45.0),
            ),
        ],
    )


def panel_row(panel: pd.DataFrame, target: str, row_kind: str, index: Any) -> pd.Series:
    """The one panel row of *target* with *row_kind* and *index* (None for season to date)."""
    mask = (panel["target"] == target) & (panel["row_kind"] == row_kind)
    if index is None:
        mask &= panel["period_index"].isna()
    else:
        mask &= panel["period_index"] == index
    rows = panel.loc[mask]
    assert len(rows) == 1, (
        f"expected one {target} {row_kind} {index} row, got {len(rows)}"
    )
    return rows.iloc[0]


@pytest.fixture
def switch_on(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", True)


def test_panel_frames_rows(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    write_fixture_ledger(ledger)

    panel, _aa = comparison.panel_frames(read_entries(ledger), games_frame(), ledger)

    assert list(panel.columns) == list(cache_schema.REFIT_PANEL_COLUMNS)
    assert set(panel["season"]) == {SEASON}

    before = panel_row(panel, "ats", "period", 0)
    assert before["period_label"] == "Before the first re-fit"
    assert (before["live_error"], before["frozen_error"]) == (2.5, 2.5)
    assert before["gap"] == 0.0
    assert before["ahead"] == "even"
    assert before["paired_games"] == 2
    assert before["unit"] == "points"

    period = panel_row(panel, "ats", "period", 1)
    assert period["period_label"] == "Period 1, from Tue Nov 3"
    assert period["period_start"] == "2026-11-03"
    assert (period["live_error"], period["frozen_error"]) == (2.25, 5.25)
    assert period["gap"] == -3.0
    assert period["ahead"] == "live"
    assert period["paired_games"] == 2

    to_date = panel_row(panel, "ats", "season_to_date", None)
    assert "since Tue Nov 3" in to_date["period_label"]
    assert to_date["period_start"] == "2026-11-03"
    # Generation >= 1 only: the generation-0 weeks 7-8 are not in it.
    assert (to_date["live_error"], to_date["frozen_error"], to_date["gap"]) == (
        2.25,
        5.25,
        -3.0,
    )
    assert to_date["paired_games"] == 2

    wp = panel_row(panel, "wp", "period", 1)
    assert wp["unit"] == "log loss"
    assert wp["live_error"] == -math.log(0.6)
    assert wp["frozen_error"] == -math.log(0.5)
    assert wp["ahead"] == "live"

    ou = panel_row(panel, "ou", "period", 1)
    assert (ou["live_error"], ou["frozen_error"], ou["gap"]) == (6.0, 1.0, 5.0)
    assert ou["ahead"] == "frozen"

    # Weeks 5-6 (no shadow row) contribute nothing: the only ATS period-0 games are weeks 7-8,
    # and no target other than ATS has a period-0 row.
    assert len(panel.loc[panel["period_index"] == 0]) == 1


def test_aa_status_rows(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    first_run = datetime(2026, 10, 28, 21, 0, tzinfo=UTC)
    committed = datetime(2026, 10, 20, 19, 41, tzinfo=UTC)
    record_week(
        ledger,
        AAWeekResult(
            season=SEASON,
            week=8,
            live_rows=42,
            twins=42,
            clean=True,
            reason="clean",
            first_mismatch=None,
            week_first_decision_run_utc=first_run,
            preregistration_committed_at=committed,
        ),
        code_commit=None,
    )
    record_week(
        ledger,
        AAWeekResult(
            season=SEASON,
            week=9,
            live_rows=45,
            twins=45,
            clean=False,
            reason="field_mismatch",
            first_mismatch={
                "game_id": "2026_09_KC_BUF",
                "target": "ats",
                "field": "per_bet_ev",
            },
            week_first_decision_run_utc=first_run,
            preregistration_committed_at=committed,
        ),
        code_commit=None,
    )

    _panel, aa = comparison.panel_frames([], games_frame(), ledger)

    assert list(aa.columns) == list(cache_schema.REFIT_AA_STATUS_COLUMNS)
    assert len(aa) == 2
    clean, unclean = aa.iloc[0], aa.iloc[1]
    assert (clean["season"], clean["week"], bool(clean["clean"])) == (SEASON, 8, True)
    assert clean["reason"] == "clean"
    assert (clean["live_rows"], clean["twins"]) == (42, 42)
    assert clean["mismatch_field"] is None
    assert isinstance(clean["evaluated_at"], str)
    assert (unclean["week"], bool(unclean["clean"])) == (9, False)
    assert unclean["reason"] == "field_mismatch"
    assert (
        unclean["mismatch_field"],
        unclean["mismatch_game"],
        unclean["mismatch_target"],
    ) == ("per_bet_ev", "2026_09_KC_BUF", "ats")


def _read_table(db_path: Path, table: str, columns: list[str]) -> list[tuple[Any, ...]]:
    conn = open_cache_readonly(db_path)
    try:
        return conn.execute(f"SELECT {', '.join(columns)} FROM {table}").fetchall()
    finally:
        conn.close()


def _as_rows(frame: pd.DataFrame, columns: list[str]) -> list[tuple[Any, ...]]:
    """The frame's values with every NULL as None, the way DuckDB hands them back."""
    return [
        tuple(None if pd.isna(value) else value for value in row)
        for row in frame[columns].itertuples(index=False)
    ]


def _empty_dirs(tmp_path: Path) -> Path:
    empty = tmp_path / "empty"
    empty.mkdir(exist_ok=True)
    return empty


def test_populate_writes_both_tables(tmp_path: Path, switch_on: None) -> None:
    ledger = tmp_path / "ledger"
    write_fixture_ledger(ledger)
    output_dir = tmp_path / "outputs"
    _write_outputs(output_dir, _replay_rows())
    sources = read_bet_cache_sources(
        output_dir, ledger_dir=ledger, silver_dir=write_silver(tmp_path / "silver")
    )
    empty = _empty_dirs(tmp_path)
    db_path = tmp_path / "web_cache.duckdb"

    populate_cache(
        db_path=db_path,
        artifacts_dir=empty,
        outputs_dir=empty,
        gold_dir=empty,
        silver_dir=empty,
        refit_panel_df=sources.refit_panel,
        refit_aa_status_df=sources.refit_aa_status,
    )

    panel_columns = list(cache_schema.REFIT_PANEL_COLUMNS)
    stored = _read_table(db_path, "refit_panel", panel_columns)
    expected = _as_rows(sources.refit_panel, panel_columns)
    assert len(stored) == len(expected) > 0
    assert sorted(stored, key=repr) == sorted(expected, key=repr)
    # The generation-1 ATS period, read back from the file.
    (ats_period,) = [
        row
        for row in stored
        if row[panel_columns.index("target")] == "ats"
        and row[panel_columns.index("period_index")] == 1
    ]
    assert ats_period[panel_columns.index("gap")] == -3.0
    assert (
        _read_table(
            db_path, "refit_aa_status", list(cache_schema.REFIT_AA_STATUS_COLUMNS)
        )
        == []
    )

    bare = tmp_path / "bare.duckdb"
    populate_cache(
        db_path=bare,
        artifacts_dir=empty,
        outputs_dir=empty,
        gold_dir=empty,
        silver_dir=empty,
    )
    assert _read_table(bare, "refit_panel", panel_columns) == []
    assert (
        _read_table(bare, "refit_aa_status", list(cache_schema.REFIT_AA_STATUS_COLUMNS))
        == []
    )


def test_no_test_columns(tmp_path: Path) -> None:
    empty = _empty_dirs(tmp_path)
    db_path = tmp_path / "web_cache.duckdb"
    populate_cache(
        db_path=db_path,
        artifacts_dir=empty,
        outputs_dir=empty,
        gold_dir=empty,
        silver_dir=empty,
    )
    conn = open_cache_readonly(db_path)
    try:
        built = [
            column
            for (column,) in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name IN ('refit_panel', 'refit_aa_status')"
            ).fetchall()
        ]
    finally:
        conn.close()
    declared = [
        *cache_schema.REFIT_PANEL_COLUMNS,
        *cache_schema.REFIT_AA_STATUS_COLUMNS,
    ]
    assert sorted(built) == sorted(declared)
    for column in declared:
        hits = [fragment for fragment in TEST_RESULT_FRAGMENTS if fragment in column]
        assert hits == [], f"column {column!r} names a test result ({hits})"


def test_cache_sources_carry_the_panel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ledger = tmp_path / "ledger"
    write_fixture_ledger(ledger)
    output_dir = tmp_path / "outputs"
    _write_outputs(output_dir, _replay_rows())
    silver = write_silver(tmp_path / "silver")

    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", True)
    sources = read_bet_cache_sources(output_dir, ledger_dir=ledger, silver_dir=silver)
    panel, aa = comparison.panel_frames(
        read_entries(ledger), load_silver_games(silver), ledger
    )
    pd.testing.assert_frame_equal(sources.refit_panel, panel)
    pd.testing.assert_frame_equal(sources.refit_aa_status, aa)

    # No games.parquet: the build degrades to an empty games frame instead of raising.
    no_silver = tmp_path / "no_silver"
    degraded = read_bet_cache_sources(
        output_dir, ledger_dir=ledger, silver_dir=no_silver
    )
    periods = degraded.refit_panel.loc[degraded.refit_panel["row_kind"] == "period"]
    assert not periods.empty
    assert set(periods["paired_games"]) == {0}

    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", False)
    off_output = tmp_path / "off_outputs"
    _write_outputs(off_output, _replay_rows())
    off = read_bet_cache_sources(off_output, ledger_dir=ledger, silver_dir=silver)
    assert off.refit_panel.empty
    assert list(off.refit_panel.columns) == list(cache_schema.REFIT_PANEL_COLUMNS)
    assert off.refit_aa_status.empty
    assert list(off.refit_aa_status.columns) == list(
        cache_schema.REFIT_AA_STATUS_COLUMNS
    )


def test_an_install_with_no_pairs_is_a_period_row(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    install_generation_one(ledger)

    panel, _aa = comparison.panel_frames([], games_frame(), ledger)

    for target in ("wp", "ats", "ou"):
        period = panel_row(panel, target, "period", 1)
        assert period["period_label"] == "Period 1, from Tue Nov 3"
        assert period["paired_games"] == 0
        for column in ("live_error", "frozen_error", "gap", "ahead"):
            assert pd.isna(period[column]), f"{target} {column} is {period[column]!r}"
        to_date = panel_row(panel, target, "season_to_date", None)
        assert to_date["paired_games"] == 0
    assert set(panel["row_kind"]) == {"period", "season_to_date"}


def test_the_panel_season_is_the_record_season() -> None:
    assert cache_schema.REFIT_PANEL_SEASON == REFIT_SEASON
