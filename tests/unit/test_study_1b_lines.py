"""Tests for Study 1b's lines: the early capture, the day-before line and the close per game.

Every frame is built here by hand; nothing reads the live lake. Instants are written in ET
(``et``) and the module compares them in UTC, so an equality below is an equality of
instants.
"""

from __future__ import annotations

import math
from typing import cast

import pandas as pd
import pytest

from backtest.study_1b_lines import (
    LINE_COLUMNS,
    NO_CLOSE,
    NO_DAY_BEFORE,
    NO_EARLY_CAPTURE,
    NOT_BEFORE_DAY_BEFORE,
    StudyLines,
    early_captures,
    study_lines,
)

_ET = "America/New_York"

SUN_W1 = "2024_W01_ARI@BUF"
MON_W1 = "2024_W01_NYJ@SF"
TARGET = "2024_W02_LV@BAL"
DAY_BEFORE = "2024-09-13 17:55"


def et(text: str) -> pd.Timestamp:
    """*text* as an ET wall-clock instant."""
    return cast(pd.Timestamp, pd.Timestamp(text, tz=_ET))


def _games(rows: list[tuple[str, int, int, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "season": [r[1] for r in rows],
            "week": [r[2] for r in rows],
            "kickoff_et": [et(r[3]) for r in rows],
        }
    )


def _timeline(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """``odds_timeline`` rows: the spread on the timeline's own INVERTED sign."""
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "snapshot_ts": [et(r[1]) for r in rows],
            "spread": [r[2] for r in rows],
            "total": [r[3] for r in rows],
        }
    )


def _day_before(rows: list[tuple[str, str, float, float]]) -> pd.DataFrame:
    """Corpus rows: ``market_spread`` already home-positive."""
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "snapshot_ts": [et(r[1]) for r in rows],
            "market_spread": [r[2] for r in rows],
            "market_total": [r[3] for r in rows],
        }
    )


def _close(rows: list[tuple[str, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [r[0] for r in rows],
            "close_spread": [r[1] for r in rows],
            "close_total": [r[2] for r in rows],
        }
    )


def kept_row(out: StudyLines, game_id: str) -> pd.Series:
    rows = out.frame.loc[out.frame["game_id"] == game_id]
    assert len(rows) == 1, f"{game_id} is not kept exactly once"
    return rows.iloc[0]


def excluded_reason(out: StudyLines, game_id: str) -> str:
    rows = out.excluded.loc[out.excluded["game_id"] == game_id]
    assert len(rows) == 1, f"{game_id} is not excluded exactly once"
    return str(rows.iloc[0]["reason"])


def _week_two(monday_kickoff: str, captures: list[str]) -> StudyLines:
    """Week 1 = a Sunday game and a 'Monday' game; the target is week 2's Sunday game."""
    games = _games(
        [
            (SUN_W1, 2024, 1, "2024-09-08 13:00"),
            (MON_W1, 2024, 1, monday_kickoff),
            (TARGET, 2024, 2, "2024-09-15 13:00"),
        ]
    )
    timeline = _timeline([(TARGET, when, -3.0, 44.5) for when in captures])
    day_before = _day_before([(TARGET, DAY_BEFORE, 3.5, 45.0)])
    close = _close([(TARGET, 3.0, 45.5)])
    return study_lines(timeline, games, day_before, close)


def test_the_early_line_is_the_first_capture_after_the_previous_week_ended():
    # week 1: Sun 2024-09-08 13:00 and Mon 2024-09-09 20:15 (ends Tue 00:15)
    out = _week_two(
        "2024-09-09 20:15",
        ["2024-09-06 17:55", "2024-09-10 11:55", "2024-09-11 11:55", DAY_BEFORE],
    )
    row = kept_row(out, TARGET)
    assert row["early_ts"] == et("2024-09-10 11:55")
    assert row["day_before_ts"] == et(DAY_BEFORE)


def test_a_rescheduled_earlier_week_game_moves_the_early_line_later():  # Review Focus 1
    # the week-1 Monday game moved to Wed 2024-09-11 20:15 (ends Thu 00:15); the 23:55
    # capture falls after its kickoff but before its end, so it is still too early
    out = _week_two(
        "2024-09-11 20:15",
        [
            "2024-09-10 11:55",
            "2024-09-11 11:55",
            "2024-09-11 23:55",
            "2024-09-12 11:55",
            DAY_BEFORE,
        ],
    )
    assert kept_row(out, TARGET)["early_ts"] == et("2024-09-12 11:55")


def test_no_capture_before_the_day_before_line_excludes_the_game():  # Review Focus 1
    # same as above without the Thursday capture: the first eligible capture IS the
    # day-before one, so the game is never paired with the previous week's board
    out = _week_two(
        "2024-09-11 20:15", ["2024-09-10 11:55", "2024-09-11 11:55", DAY_BEFORE]
    )
    assert excluded_reason(out, TARGET) == NOT_BEFORE_DAY_BEFORE
    assert TARGET not in set(out.frame["game_id"])


def test_a_game_played_late_floors_every_later_week_not_only_the_next():
    # a week-1 game played Tue 2024-09-17 20:00 (ends Wed 00:00), after week 2 ended
    # (Mon 09-16 20:15, ends Tue 00:15): week 3's early line waits for it
    target = "2024_W03_DEN@TB"
    games = _games(
        [
            ("2024_W01_CAR@NO", 2024, 1, "2024-09-17 20:00"),
            ("2024_W02_CHI@HOU", 2024, 2, "2024-09-16 20:15"),
            (target, 2024, 3, "2024-09-22 13:00"),
        ]
    )
    timeline = _timeline(
        [
            (target, "2024-09-17 11:55", -1.0, 41.0),
            (target, "2024-09-18 11:55", -1.5, 41.5),
        ]
    )
    early = early_captures(timeline, games).set_index("game_id")
    assert early.loc[target, "early_ts"] == et("2024-09-18 11:55")


def test_no_capture_more_than_seven_days_before_kickoff_is_used():  # Review Focus 2
    target, other = "2024_W01_PIT@ATL", "2024_W01_ARI@BUF"
    games = _games(
        [
            (target, 2024, 1, "2024-09-08 13:00"),
            (other, 2024, 1, "2024-09-08 13:00"),
        ]
    )
    timeline = _timeline(
        [
            (target, "2024-08-27 11:55", -3.0, 42.0),  # 12 days out
            (target, "2024-09-03 11:55", -3.5, 42.5),
            (target, "2024-09-06 17:55", -3.5, 43.0),
            (other, "2024-08-27 11:55", 6.0, 47.0),  # its only capture: 12 days out
        ]
    )
    day_before = _day_before(
        [
            (target, "2024-09-06 17:55", 3.5, 43.0),
            (other, "2024-09-06 17:55", -6.5, 47.5),
        ]
    )
    close = _close([(target, 3.5, 43.0), (other, -6.5, 47.5)])
    out = study_lines(timeline, games, day_before, close)
    assert kept_row(out, target)["early_ts"] == et("2024-09-03 11:55")
    assert excluded_reason(out, other) == NO_EARLY_CAPTURE


def test_the_window_and_the_floor_both_admit_their_own_boundary():
    # exactly 7 days before kickoff is in the window; exactly at the floor is after it;
    # exactly at kickoff is not before kickoff
    week_one, week_two = "2024_W01_GB@PHI", "2024_W02_NYG@WAS"
    games = _games(
        [
            (week_one, 2024, 1, "2024-09-09 20:15"),  # ends Tue 2024-09-10 00:15
            (week_two, 2024, 2, "2024-09-15 13:00"),
        ]
    )
    timeline = _timeline(
        [
            (week_one, "2024-09-02 20:15", 2.0, 48.0),
            (week_two, "2024-09-10 00:15", -1.0, 43.0),
        ]
    )
    early = early_captures(timeline, games).set_index("game_id")
    assert early.loc[week_one, "early_ts"] == et("2024-09-02 20:15")
    assert early.loc[week_two, "early_ts"] == et("2024-09-10 00:15")

    at_kickoff_only = _timeline([(week_one, "2024-09-09 20:15", 2.5, 48.5)])
    assert early_captures(at_kickoff_only, games).empty


def test_a_capture_missing_one_number_is_still_the_early_line():
    # a Tuesday board with a spread but no total yet is the early line (its total stays
    # NaN for the O/U test to drop); a capture carrying neither number is skipped
    games = _games(
        [
            (SUN_W1, 2024, 1, "2024-09-08 13:00"),
            (MON_W1, 2024, 1, "2024-09-09 20:15"),
            (TARGET, 2024, 2, "2024-09-15 13:00"),
        ]
    )
    timeline = _timeline(
        [
            (SUN_W1, "2024-09-03 11:55", math.nan, math.nan),
            (SUN_W1, "2024-09-04 11:55", 1.0, 41.0),
            (TARGET, "2024-09-10 11:55", -3.0, math.nan),
            (TARGET, "2024-09-11 11:55", -3.0, 44.5),
        ]
    )
    early = early_captures(timeline, games).set_index("game_id")
    assert early.loc[TARGET, "early_ts"] == et("2024-09-10 11:55")
    assert early.loc[TARGET, "early_spread"] == 3.0
    assert math.isnan(early.loc[TARGET, "early_total"])
    assert early.loc[SUN_W1, "early_ts"] == et("2024-09-04 11:55")


def test_every_line_is_home_positive():  # Review Focus 3
    # a home favourite of 3: timeline spread -3.0, corpus market_spread +3.5, close +3.0
    out = _week_two("2024-09-09 20:15", ["2024-09-10 11:55", DAY_BEFORE])
    row = kept_row(out, TARGET)
    assert (row["early_spread"], row["day_before_spread"], row["close_spread"]) == (
        3.0,
        3.5,
        3.0,
    )
    assert (row["early_total"], row["day_before_total"], row["close_total"]) == (
        44.5,
        45.0,
        45.5,
    )
    assert tuple(out.frame.columns) == LINE_COLUMNS


def test_every_game_is_kept_or_excluded_exactly_once():
    (
        kept,
        half_close,
        no_early,
        no_day,
        not_before,
        no_close,
        blank_close,
        two_faults,
    ) = (
        "2024_W01_KEEP@ONE",
        "2024_W01_HALF@CLS",
        "2024_W01_NOEA@RLY",
        "2024_W01_NODB@DAY",
        "2024_W01_SAME@CAP",
        "2024_W01_NOCL@OSE",
        "2024_W01_BLNK@CLS",
        "2024_W01_TWOF@AULT",
    )
    ids = [
        kept,
        half_close,
        no_early,
        no_day,
        not_before,
        no_close,
        blank_close,
        two_faults,
    ]
    games = _games([(gid, 2024, 1, "2024-09-08 13:00") for gid in ids])
    timeline = _timeline(
        [
            (gid, "2024-09-03 11:55", -2.0, 44.0)
            for gid in (kept, half_close, no_day, no_close, blank_close, two_faults)
        ]
        + [(not_before, "2024-09-06 17:55", -2.0, 44.0)]
    )
    day_before = _day_before(
        [
            (gid, "2024-09-06 17:55", 2.0, 44.0)
            for gid in (kept, half_close, no_early, not_before, no_close, blank_close)
        ]
    )
    close = _close(
        [(gid, 2.0, 44.0) for gid in (kept, no_early, no_day, not_before)]
        + [(half_close, 2.0, math.nan), (blank_close, math.nan, math.nan)]
    )
    out = study_lines(timeline, games, day_before, close)

    assert sorted([*out.frame["game_id"], *out.excluded["game_id"]]) == sorted(
        games["game_id"]
    )
    # a closing row carrying one number is a close; only that target drops the game
    assert list(out.frame["game_id"]) == sorted([kept, half_close])
    assert dict(zip(out.excluded["game_id"], out.excluded["reason"], strict=True)) == {
        no_early: NO_EARLY_CAPTURE,
        no_day: NO_DAY_BEFORE,
        not_before: NOT_BEFORE_DAY_BEFORE,
        no_close: NO_CLOSE,
        blank_close: NO_CLOSE,  # a closing row carrying neither number is no close
        two_faults: NO_DAY_BEFORE,  # the first failing reason wins
    }
    assert list(out.excluded.columns) == ["game_id", "season", "week", "reason"]


def test_a_naive_capture_time_is_refused():
    games = _games([(SUN_W1, 2024, 1, "2024-09-08 13:00")])
    timeline = _timeline([(SUN_W1, "2024-09-03 11:55", -3.0, 44.5)])
    timeline["snapshot_ts"] = timeline["snapshot_ts"].dt.tz_localize(None)
    with pytest.raises(ValueError, match="naive"):
        early_captures(timeline, games)
