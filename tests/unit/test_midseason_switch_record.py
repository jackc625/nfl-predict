"""The mid-season switch record agrees with the code and the schedule it describes.

The record (``backtest/midseason_switch_2026.py`` plus the root ``MIDSEASON-SWITCH-RECORD.md``)
copies values other code also holds: the v2 recipe name, the re-fit date list in force, and the
switch week, whose deadline must fall between two NFL weeks' locks. These tests prove the copies
agree, and check RELATIONSHIPS rather than only that strings appear: the paused dates are the
list-week-8 dates after the first re-fit, the coverage ends the week before the switch, and the
deadline sits between the last lock of the week before the switch and the first lock of the switch
week on the recorded 2026 schedule, both weeks derived from ``DAY_BEFORE_RETIRED_FROM``.

RE-POINTED 2026-11-06 for MIDSEASON-SWITCH-RECORD.md section 8 (the correction to week 10): the
switch-instant test is named for the derived weeks, now week 9's last lock and week 10's first.

The silver schedule is read only, through ``forward_ledger.settle.load_silver_games``; nothing is
written. The markdown is read from the repository root by path.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import utils.game_lock as lock_rule
from backtest.midseason_switch_2026 import (
    DAY_BEFORE_RETIRED_FROM,
    INSTALLED_RECIPE,
    INVALIDATES,
    PREREGISTRATION_DEADLINE_UTC,
    R6_COVERAGE_END_WEEK,
    RECORD_PATH,
    REFIT_PAUSE_RULING,
    REFIT_PAUSED_DATES,
    V2_EARNED_WEIGHTS,
)
from backtest.refit_preregistration_2026 import REFIT_DATE_LISTS
from forward_ledger.settle import load_silver_games
from models.v2_recipe import V2_RECIPE_NAME

REPO_ROOT = Path(__file__).resolve().parents[2]
ET = ZoneInfo("America/New_York")

# The re-fit date list in force: week 8 was the first clean A/A week, and its first date ran.
LIST_IN_FORCE = 8
FIRST_REFIT_DATE = "2026-11-03"


def list_8_dates() -> tuple[str, ...]:
    """The cadence Tuesdays of the frozen date list for first clean A/A week 8, in order."""
    return tuple(listed for listed, _ in REFIT_DATE_LISTS[LIST_IN_FORCE])


def _record_text() -> str:
    return (REPO_ROOT / RECORD_PATH).read_text(encoding="utf-8")


def test_the_installed_recipe_is_the_v2_recipe() -> None:
    assert INSTALLED_RECIPE == V2_RECIPE_NAME


def test_the_paused_dates_are_list_8_dates_after_the_first_re_fit() -> None:
    assert list_8_dates()[0] == FIRST_REFIT_DATE
    after_first = tuple(
        listed for listed in list_8_dates() if listed > FIRST_REFIT_DATE
    )
    assert after_first == REFIT_PAUSED_DATES


def test_coverage_ends_the_week_before_the_switch() -> None:
    switch_week = DAY_BEFORE_RETIRED_FROM[1]
    assert switch_week == R6_COVERAGE_END_WEEK + 1


def test_every_earned_weight_is_zero() -> None:
    assert dict(V2_EARNED_WEIGHTS) == {"wp": 0.0, "ats": 0.0, "ou": 0.0}


def test_the_switch_instant_falls_between_the_week_before_and_the_switch_week() -> None:
    season, switch_week = DAY_BEFORE_RETIRED_FROM
    games = load_silver_games(REPO_ROOT / "data" / "silver")
    games = games.loc[games["season"] == season]
    locks = lock_rule.lock_frame(games)
    before = games.loc[games["week"] < switch_week, "game_id"].astype(str)
    first = games.loc[games["week"] == switch_week, "game_id"].astype(str)
    assert len(before) > 0
    assert len(first) > 0
    deadline = datetime.fromisoformat(PREREGISTRATION_DEADLINE_UTC)
    assert locks[before].max() < deadline <= locks[first].min()


def test_the_record_is_ascii_and_names_every_invalidated_sha() -> None:
    text = _record_text()
    assert text.isascii()
    assert all(sha in text for sha in INVALIDATES)


def test_the_markdown_restates_the_constants() -> None:
    """The prose half carries the values the code reads, and its anchor block names both halves."""
    text = _record_text()
    for listed in REFIT_PAUSED_DATES:
        assert listed in text, listed
    assert INSTALLED_RECIPE in text
    assert REFIT_PAUSE_RULING in text
    assert "\n## 3. The v1 calendar re-fit is paused\n" in text
    assert "preregistration_anchor_recorded_in: tests/midseason_switch_state.py" in text
    assert (
        "preregistration_paths: MIDSEASON-SWITCH-RECORD.md, "
        "backtest/midseason_switch_2026.py"
    ) in text
    local_deadline = datetime.fromisoformat(PREREGISTRATION_DEADLINE_UTC).astimezone(ET)
    assert f"preregistration_deadline: {local_deadline.isoformat()}" in text
