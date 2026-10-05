"""The verdict reader and the start-week computation (Phase 34, Plan 34-09 Task 3; LDGR-10, D-06, D-15).

``forward_ledger.verdict.verdict_rows`` is the only way a verdict row set is read: it refuses when no
declaration exists, keeps only the declared season, arm, scope label and week range, and counts the
outcome IN FORCE (a correction entry supersedes the row's own grade). ``compute_start_week`` derives
W from the ledger's own contents and the schedule rather than from a guess.

Synthetic ledgers, schedules and scopes only; every ledger lives under ``tmp_path`` (COLD-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from forward_ledger.verdict import (
    StartWeekUnavailableError,
    compute_start_week,
    verdict_rows,
)

from forward_ledger.canonical import ENTRY_KIND_ROW
from forward_ledger.declarations import VerdictScope, VerdictScopeUndeclaredError
from forward_ledger.store import (
    LedgerEntry,
    append_rows,
    apply_updates,
    build_entry,
    commit_changes,
    ledger_head,
    read_entries,
)
from tests.unit.test_forward_ledger_store import graded, key_of, make_row
from utils.date_utils import ET

SCOPE = VerdictScope(
    season=2026,
    start_week=6,
    end_week=22,
    includes_playoff_weeks=True,
    includes_neutral_site_games=True,
    counted_arm="live",
    outcome_rule="in_force_corrected_outcome",
    fill_convention_id="fill-v1",
    bootstrap_regime_weeks=(2, 3, 4),
)


def verdict_row(game: str, week: int, **overrides: Any) -> dict[str, Any]:
    return make_row(game_id=f"2026_W{week:02d}_{game}", week=week, **overrides)


def keys_of(frame: pd.DataFrame) -> list[tuple[Any, ...]]:
    columns = ["game_id", "season", "week", "target", "arm"]
    return [tuple(record) for record in frame[columns].itertuples(index=False)]


def test_undeclared_refuses(tmp_path: Path) -> None:
    append_rows(tmp_path, [verdict_row("KC@BUF", 6)])

    with pytest.raises(VerdictScopeUndeclaredError):
        verdict_rows(read_entries(tmp_path), None)


def test_week_22_in_no_week_23(tmp_path: Path) -> None:
    super_bowl = verdict_row("KC@SF", 22)
    week_23 = verdict_row("SYN@THETIC", 23)
    append_rows(tmp_path, [super_bowl, week_23])

    frame = verdict_rows(read_entries(tmp_path), SCOPE)

    assert keys_of(frame) == [key_of(super_bowl)]


def test_pre_verdict_and_shadow_excluded(tmp_path: Path) -> None:
    counted = verdict_row("KC@BUF", 7)
    pre_verdict = verdict_row("DAL@PHI", 7, verdict_scope="pre_verdict")
    shadow = verdict_row("KC@BUF", 7, arm="shadow")
    early = verdict_row("SF@SEA", 5)
    append_rows(tmp_path, [counted, pre_verdict, shadow, early])

    frame = verdict_rows(read_entries(tmp_path), SCOPE)

    assert keys_of(frame) == [key_of(counted)]
    assert set(frame["verdict_scope"]) == {"verdict"}
    assert set(frame["arm"]) == {"live"}


def test_in_force_outcome_counted(tmp_path: Path) -> None:
    corrected_row = verdict_row("KC@BUF", 8)
    plain_row = verdict_row("DAL@PHI", 8)
    append_rows(tmp_path, [corrected_row, plain_row])
    win = graded("win", 0.9090909090909091, 1.1363636363636365)
    apply_updates(
        tmp_path,
        grading_updates={key_of(corrected_row): win, key_of(plain_row): win},
    )
    commit_changes(
        tmp_path,
        new_corrections=[
            {
                "game_id": corrected_row["game_id"],
                "season": 2026,
                "week": 8,
                "target": "ats",
                "arm": "live",
                "original_grading_status": "win",
                "original_payout_flat": 0.9090909090909091,
                "prior_grading_status": "win",
                "prior_payout_flat": 0.9090909090909091,
                "prior_realized_units": 1.1363636363636365,
                "prior_correction_seq": None,
                "corrected_grading_status": "loss",
                "corrected_outcome": False,
                "corrected_payout_flat": -1.0,
                "corrected_realized_units": -1.25,
                "realized_value": -10.0,
                "source_dataset": "schedules",
                "source_season": 2026,
                "source_week": 9,
                "source_sequence": 2,
                "detected_at_utc": "2026-10-28T21:00:00+00:00",
                "corrected_at_utc": "2026-10-28T21:00:05+00:00",
            }
        ],
    )

    frame = verdict_rows(read_entries(tmp_path), SCOPE).set_index("game_id")

    fixed = frame.loc[corrected_row["game_id"]]
    assert fixed["grading_status"] == "loss"
    assert fixed["outcome"] is False
    assert fixed["payout_flat"] == -1.0
    assert fixed["realized_units"] == -1.25
    assert bool(fixed["corrected"]) is True
    assert fixed["original_grading_status"] == "win"

    untouched = frame.loc[plain_row["game_id"]]
    assert untouched["grading_status"] == "win"
    assert untouched["payout_flat"] == 0.9090909090909091
    assert bool(untouched["corrected"]) is False
    assert untouched["original_grading_status"] == "win"


# ---------------------------------------------------------------------------
# compute_start_week
# ---------------------------------------------------------------------------


def schedule() -> pd.DataFrame:
    """Weeks 1-22: a Thursday 20:15 ET opener each week (week 6 on Thu 2026-10-15) and a Sunday game."""
    week_6_thursday = datetime(2026, 10, 15, 20, 15, tzinfo=ET)
    records = []
    for week in range(1, 23):
        thursday = week_6_thursday + timedelta(weeks=week - 6)
        for slot, kickoff in (("THU", thursday), ("SUN", thursday + timedelta(days=3))):
            records.append(
                {
                    "game_id": f"2026_W{week:02d}_{slot}",
                    "season": 2026,
                    "week": week,
                    "kickoff_et": pd.Timestamp(kickoff).tz_convert("UTC"),
                }
            )
    return pd.DataFrame.from_records(records)


def ledger_through(
    *weeks: int, old_writer_week: int | None = None
) -> list[LedgerEntry]:
    """In-memory chained row entries for *weeks*; *old_writer_week* also gets a NULL-stamp row."""
    rows = [verdict_row("KC@BUF", week, verdict_scope="pre_verdict") for week in weeks]
    if old_writer_week is not None:
        rows.append(
            verdict_row(
                "OLD@WRITER",
                old_writer_week,
                verdict_scope="pre_verdict",
                model_artifact_id=None,
                blend_id=None,
                recipe_id=None,
                fill_convention_id=None,
                upstream_capture_key=None,
                gold_generation_key=None,
                odds_snapshot_digest=None,
                decision_snapshot_digest=None,
            )
        )
    entries: list[LedgerEntry] = []
    for row in rows:
        head, count = ledger_head(entries)
        entries.append(build_entry(head, count, ENTRY_KIND_ROW, row))
    return entries


def test_start_week_after_old_writer_rows() -> None:
    now = datetime(2026, 10, 10, 12, 0, tzinfo=ET)

    start = compute_start_week(
        ledger_through(3, 4, 5, old_writer_week=5), schedule(), now
    )

    assert start.week == 6
    assert start.first_lock_utc == datetime(2026, 10, 14, 18, 0, tzinfo=ET)
    assert start.first_decision_run_utc == datetime(2026, 10, 14, 17, 0, tzinfo=ET)


def test_start_week_skips_a_week_whose_first_run_is_too_close() -> None:
    now = datetime(2026, 10, 14, 16, 30, tzinfo=ET)

    start = compute_start_week(ledger_through(3, 4, 5), schedule(), now)

    assert start.week == 7
    assert start.first_lock_utc == datetime(2026, 10, 21, 18, 0, tzinfo=ET)


def test_start_week_after_rows_already_written() -> None:
    now = datetime(2026, 10, 10, 12, 0, tzinfo=ET)

    start = compute_start_week(ledger_through(3, 4, 5, 6), schedule(), now)

    assert start.week >= 7


def test_start_week_refuses_past_22() -> None:
    now = datetime(2026, 10, 10, 12, 0, tzinfo=ET)

    with pytest.raises(StartWeekUnavailableError):
        compute_start_week(ledger_through(3, 22), schedule(), now)
