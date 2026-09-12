"""An empty bet list is either an honest no-edge week or a broken build. Two refusals.

Phase 33, Plan 33-07 Task 1 (COLD-08, R9, T-33-32).

WHAT THIS MODULE PROVES, AND WHY IT DID NOT EXIST BEFORE
---------------------------------------------------------
BOTH refusals were already in the tree and NEITHER was tested:

* ``backtest.weekly_bet_list._load_week_schedule`` raises "no scheduled games for
  {season} week {week} in {path}" when the schedule carries nothing for the week;
* ``backtest.weekly_bet_list.build_weekly_candidates`` raises "gold matrix {path} has no
  rows for {season} week {week}" when the week IS scheduled and gold does not carry it.

So the protection against publishing a broken build as "we recommended nothing this week"
was, until now, unproven. This plan's deliverable is the PROOF, not the errors.

THE TWO ARE ADJACENT AND MUST STAY DISTINGUISHABLE
----------------------------------------------------
"No games are scheduled" (a bye-shaped week, an off-by-one on the week number, a schedule
that never ingested) and "games are scheduled and gold has no rows for them" (the gold
build did not run, or ran on last week's inputs) call for different fixes. A reader who
cannot tell them apart from the message goes to the wrong place, so
``test_the_two_refusals_are_distinguishable`` asserts each message carries a token the
other does not.

EVERY REFUSAL HERE IS ASSERTED BY RAISING IT AND READING THE MESSAGE
----------------------------------------------------------------------
Never by grepping the source. Both surviving v3.0 audit warnings exist because a test read
a file instead of checking written output, and the milestone invariant is that a criterion
asserts the EFFECT.

THE NEGATIVE CONTROL IS THE POINT OF THE WHOLE MODULE. A genuine no-edge week -- gold rows
present, edges all below the pre-registered EV floor -- must produce candidate rows with a
populated ``status`` and ``rejection_reason`` and trip NEITHER refusal. Without it these
refusals would be indistinguishable from a gate that fires on every quiet week, and the
first thing an operator does with a gate like that is stop reading it.

Run this module:  uv run pytest tests/unit/test_empty_week_refusals.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from backtest import weekly_bet_list
from backtest.weekly_bet_list import build_weekly_candidates
from tests.fixtures.decision_frame import (
    FREEZE_AT_SUNDAY,
    N_GAMES,
    SEASON,
    SUNDAY_GAMEDAY,
    WEEK,
    builder_stub_frames,
    fits_with_floor,
    mini_strategies,
)

# The week the silver schedule below genuinely does NOT carry. Distinct from WEEK so a
# test that confused the two would fail rather than coincide.
_UNSCHEDULED_WEEK = 9

# The week the gold matrix carries instead of the one being asked for.
_WRONG_GOLD_WEEK = 5


def _write_silver_games(silver_dir: Path) -> Path:
    """A one-week silver schedule in the shape ``_load_week_schedule`` reads.

    ``kickoff_et`` is the only non-obvious column: ``_with_gameday`` converts it to the
    EASTERN calendar date the per-game freeze is measured from.
    """
    silver_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "game_id": [
                f"{SEASON}_{WEEK:02d}_A{index:02d}@H{index:02d}"
                for index in range(N_GAMES)
            ],
            "season": [SEASON] * N_GAMES,
            "week": [WEEK] * N_GAMES,
            "kickoff_et": [f"{SUNDAY_GAMEDAY}T17:00:00+00:00"] * N_GAMES,
        }
    )
    path = silver_dir / "games.parquet"
    frame.to_parquet(path, index=False)
    return path


def _write_odds_snapshot(silver_dir: Path) -> Path:
    """The freeze-time odds snapshot ``build_weekly_candidates`` joins."""
    game_ids = [
        f"{SEASON}_{WEEK:02d}_A{index:02d}@H{index:02d}" for index in range(N_GAMES)
    ]
    frame = pd.DataFrame(
        {
            "game_id": game_ids,
            "snapshot_ts": [FREEZE_AT_SUNDAY] * N_GAMES,
            "ml_home": [-130.0] * N_GAMES,
            "ml_away": [110.0] * N_GAMES,
            "spread": [-2.5] * N_GAMES,
            "total": [45.0] * N_GAMES,
            "sportsbook": ["consensus"] * N_GAMES,
            "is_live": [False] * N_GAMES,
        }
    )
    path = silver_dir / "odds_snapshot.parquet"
    frame.to_parquet(path, index=False)
    return path


def _write_gold(gold_dir: Path, week: int) -> None:
    gold_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "game_id": [f"{SEASON}_{week:02d}_A00@H00"],
            "season": [SEASON],
            "week": [week],
        }
    )
    for target in ("wp", "ats", "ou"):
        frame.to_parquet(gold_dir / f"features_{target}.parquet", index=False)


# ---------------------------------------------------------------------------
# Refusal 1 -- the week is not on the schedule at all
# ---------------------------------------------------------------------------


def test_a_week_with_no_scheduled_games_is_refused_by_name(tmp_path: Path) -> None:
    """The universe's spine is the SCHEDULE (D31-19); an empty spine is refused, not published."""
    silver = tmp_path / "silver"
    _write_silver_games(silver)

    with pytest.raises(ValueError) as raised:
        weekly_bet_list._load_week_schedule(SEASON, _UNSCHEDULED_WEEK, silver)

    message = str(raised.value)
    assert "no scheduled games" in message, message
    assert str(SEASON) in message, message
    assert str(_UNSCHEDULED_WEEK) in message, message
    assert "games.parquet" in message, message


# ---------------------------------------------------------------------------
# Refusal 2 -- the week IS scheduled and gold carries no row for it
# ---------------------------------------------------------------------------


def test_a_scheduled_week_whose_gold_has_no_rows_is_refused_by_name(
    tmp_path: Path,
) -> None:
    """The adjacency that matters: games exist, and the matrix that scores them does not."""
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_silver_games(silver)
    _write_odds_snapshot(silver)
    _write_gold(gold, _WRONG_GOLD_WEEK)

    with pytest.raises(ValueError) as raised:
        build_weekly_candidates(
            SEASON,
            WEEK,
            artifacts_dir=tmp_path / "artifacts",
            gold_dir=gold,
            silver_dir=silver,
        )

    message = str(raised.value)
    assert "gold matrix" in message, message
    assert "has no rows for" in message, message
    assert "features_wp.parquet" in message, message
    assert str(SEASON) in message and str(WEEK) in message, message
    assert "'wp'" in message, message


def test_the_two_refusals_are_distinguishable(tmp_path: Path) -> None:
    """Each message carries a token the other does not, so neither test can pass on the other.

    Asserted on the RAISED text of both, in one test, rather than on two remembered
    strings: a rewording that accidentally collapsed the two messages into a common form
    would make both tests above pass against either error, and neither of them alone could
    notice.
    """
    silver = tmp_path / "silver"
    gold = tmp_path / "gold"
    _write_silver_games(silver)
    _write_odds_snapshot(silver)
    _write_gold(gold, _WRONG_GOLD_WEEK)

    with pytest.raises(ValueError) as no_schedule:
        weekly_bet_list._load_week_schedule(SEASON, _UNSCHEDULED_WEEK, silver)
    with pytest.raises(ValueError) as no_gold_rows:
        build_weekly_candidates(
            SEASON,
            WEEK,
            artifacts_dir=tmp_path / "artifacts",
            gold_dir=gold,
            silver_dir=silver,
        )

    schedule_message = str(no_schedule.value)
    gold_message = str(no_gold_rows.value)

    assert "no scheduled games" in schedule_message
    assert "no scheduled games" not in gold_message, (
        "the gold-rows refusal contains the schedule refusal's distinguishing token; a "
        "test asserting on one would pass against the other"
    )
    assert "gold matrix" in gold_message
    assert "gold matrix" not in schedule_message, (
        "the schedule refusal contains the gold refusal's distinguishing token"
    )


# ---------------------------------------------------------------------------
# The negative control -- an HONEST no-edge week trips neither
# ---------------------------------------------------------------------------


def test_a_genuine_no_edge_week_carries_statuses_and_trips_neither_refusal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gold rows present, every edge below the floor: rows with reasons, and no exception.

    This is the week the two refusals must NOT fire on, and the reason they can be trusted
    when they do. The EV floor is set above anything the fixture can price, so every
    candidate is admitted to the decision and rejected on its merits -- which is what a
    real quiet week looks like.
    """
    seam = getattr(weekly_bet_list, "build_weekly_decision_frame", None)
    assert seam is not None, (
        "backtest.weekly_bet_list defines no build_weekly_decision_frame; there is no way "
        "to read a week's decision statuses without persisting a bet row, so the "
        "no-edge-week criterion cannot be asserted at all"
    )

    candidates, schedule = builder_stub_frames()
    monkeypatch.setattr(
        weekly_bet_list,
        "build_weekly_candidates",
        lambda season, week, **_kwargs: (candidates.copy(), schedule.copy()),
    )

    frame = seam(
        SEASON,
        WEEK,
        fits=fits_with_floor(10.0),
        strategies=mini_strategies(),
        now=datetime(2026, 9, 18, 12, 0, tzinfo=UTC),
    )

    assert not frame.empty, "the no-edge week produced no candidate rows at all"
    assert frame["status"].notna().all(), "a candidate row carries no status"
    assert set(frame["status"]) == {"suppressed"}, sorted(set(frame["status"]))
    assert frame["rejection_reason"].notna().all(), (
        "a suppressed row carries no rejection_reason; an unexplained suppression is "
        "indistinguishable from a row nobody looked at"
    )
