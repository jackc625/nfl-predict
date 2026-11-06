"""The 2026 mid-season switch record: the constants the code reads.

From 2026 week 10 the day-before betting system retires, the v1 calendar re-fit is paused, and the
repaired v2 models are installed once by a recorded owner command.

CORRECTED ONCE, 2026-11-06 (record section 8): by owner ruling the switch moved from week 13 to
week 10, so ``DAY_BEFORE_RETIRED_FROM``, ``R6_COVERAGE_END_WEEK`` and
``PREREGISTRATION_DEADLINE_UTC`` changed in a later commit touching only the two record files,
whose message INVALIDATES the original record commit in part. Every other value is unchanged.

ONE RECORD IN TWO FILES, in the shape of ``backtest/refit_preregistration_2026.py``. The root
``MIDSEASON-SWITCH-RECORD.md`` is the human-readable half: it states in plain English what changes,
why, and what earlier records this one INVALIDATES. This module is the half the code reads. Both
were landed in ONE commit that touches nothing else, and every value written in both places is the
same value.

Stated plainly because it is easy to forget later in the season: EDITING THIS FILE AFTER THE
COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. A value that is wrong here is wrong for the
rest of 2026, and the only honest repair is a NEW, visibly later commit whose message says it
INVALIDATES this one by naming its commit sha. Never an edit in place, never a quiet supersession.

THIS MODULE DOES NOT RECORD ITS OWN CONTENT HASH OR COMMIT. A file that must contain its own
whole-file hash has no fixed point. The witness lives OUTSIDE it: ``tests/midseason_switch_state.py``
records the commit, the newline-normalized sha256 of both record files and the commit time in a
LATER commit.

Constants only: standard-library imports only, NO project imports, NO I/O and NO logic. A value
that restates a fact other code also holds (the v2 recipe name in ``models/v2_recipe.py``, the
re-fit dates in ``backtest/refit_preregistration_2026.py``) is COPIED here as a literal, never
imported, so this record cannot change meaning when the code around it moves;
``tests/unit/test_midseason_switch_record.py`` proves the copies agree.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

__all__ = [
    "DAY_BEFORE_RETIRED_FROM",
    "INSTALLED_RECIPE",
    "INVALIDATES",
    "PREREGISTRATION_DEADLINE_UTC",
    "R6_COVERAGE_END_WEEK",
    "RECORD_PATH",
    "REFIT_PAUSED_DATES",
    "REFIT_PAUSE_RULING",
    "V2_EARNED_WEIGHTS",
]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

# The human-readable half of this record, at the repository root.
RECORD_PATH: str = "MIDSEASON-SWITCH-RECORD.md"

# Tue 2026-11-10 17:00 America/New_York (EST), in UTC: after week 9's last lock (Sun 2026-11-08
# 18:00 ET) and before week 10's first (Wed 2026-11-11 18:00 ET). The commit that last changed the
# record, and the code that reads it, must be on GitHub's master before this instant.
PREREGISTRATION_DEADLINE_UTC: str = "2026-11-10T22:00:00+00:00"

# The earlier records this one INVALIDATES, in part or by section (record section 6): the verdict
# test (922b010), the verdict scope (c3a3399) and the rolling re-fit (e25884d).
INVALIDATES: tuple[str, ...] = (
    "922b010b70d973a21945c15ca88067dae4149ad5",
    "c3a33993160769b2ab9a8c2cf7cd2a24ad19be10",
    "e25884da8356b9ec76970c4c14f87a797d2a32d7",
)


# ---------------------------------------------------------------------------
# The day-before system retires (record sections 1 and 5)
# ---------------------------------------------------------------------------

# (season, week): from this NFL week on, no live row is written, so no shadow twin either. The
# switch is keyed on the NFL week, never the clock.
DAY_BEFORE_RETIRED_FROM: tuple[int, int] = (2026, 10)

# The last week the February R6 coverage table reports: the week before the switch.
R6_COVERAGE_END_WEEK: int = 9


# ---------------------------------------------------------------------------
# The v1 re-fit pauses (record section 3)
# ---------------------------------------------------------------------------

# Dates 2-4 of the date list in force (first clean A/A week 8), copied from
# backtest/refit_preregistration_2026.REFIT_DATE_LISTS[8]. Each closes "not run" when it falls due;
# nothing trains.
REFIT_PAUSED_DATES: tuple[str, ...] = ("2026-12-01", "2026-12-29", "2027-01-26")

# The reason every paused date's close carries.
REFIT_PAUSE_RULING: str = (
    "MIDSEASON-SWITCH-RECORD.md section 3 (owner ruling B, 2026-11-05)"
)


# ---------------------------------------------------------------------------
# The v2 install (record section 4)
# ---------------------------------------------------------------------------

# The recipe the owner install puts in force: models/v2_recipe.V2_RECIPE_NAME.
INSTALLED_RECIPE: str = "recipe-2026-v2"

# Study 1a's earned weights against the day-before line. The install refuses unless the blend
# weights it tunes equal these; a weight of 0 means that bet type places no day-before bets.
V2_EARNED_WEIGHTS: Mapping[str, float] = MappingProxyType(
    {"wp": 0.0, "ats": 0.0, "ou": 0.0}
)
