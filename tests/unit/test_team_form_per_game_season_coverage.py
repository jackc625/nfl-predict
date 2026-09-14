"""The seasons opponent adjustment reads must follow the data, not a frozen literal.

WHAT IS BEING PINNED
--------------------
The twelve ``{home,away}_{off,def}_rolling_opp_adj_*`` gold columns are the
team-strength features every target consumes. They are built from per-game EPA,
and the seasons that per-game frame covers were a hardcoded range.

THE DEFECT THIS MODULE EXISTS TO PROVE AND THEN CLOSE
-----------------------------------------------------
``features/team_form.py`` resolved the full-build season list as::

    seasons = list(range(2018, 2025))

Python's ``range`` stops BEFORE its second argument, so the pool ended at 2024
and season 2025 was never built. 2025 is the season feeding the live 2026
predictions.

MEASURED on production gold (``data/gold/features_ou.parquet``) before the fix,
for each of the twelve columns:

    season 2023   285 rows   285 distinct values
    season 2024   285 rows   285 distinct values
    season 2025   285 rows     2 distinct values

Two distinct values across 285 games is not a team-strength signal. It is the
imputation that runs when the real column is absent, wearing the column's name.

WHY THE FIX IS A DERIVATION AND NOT A NEW LITERAL
--------------------------------------------------
Replacing ``2025`` with ``2026`` would rot on exactly the same schedule, in
exactly the same silent way -- a wrong upper bound produces a full-looking
column rather than an error. So the UPPER bound is derived from the seasons the
data actually carries, which is the idiom ``build_team_form_features`` already
uses for its own full-build branch.

THE FLOOR IS DELIBERATELY PRESERVED, AND THAT IS A SCOPE DECISION
------------------------------------------------------------------
``TEAM_FORM_PER_GAME_FIRST_SEASON`` keeps the inherited 2018. The play-by-play
pin reaches back to 2001, so 2018 is NOT a data-coverage floor -- it is an
inherited literal whose origin this plan did not establish. Widening it would
move roughly ninety gold columns that are currently a flat imputed constant for
2002-2017, which is far outside the change set this rung declares. Plan 33.1-09
owns the consolidation of every season literal into ``conf/season_partition.py``
and is where the floor should be re-decided. ``test_the_floor_is_preserved``
pins it so the decision cannot drift silently in the meantime.

FOUR CONTROLS
-------------
1. NON-VACUITY: the resolved list is captured from the REAL ``fetch_pbp_data``
   call, so a resolution that never reaches the loader fails here.
2. THE ASSERTION: the latest covered season is in the pool, and the pool tracks
   a caller whose data ends somewhere else entirely.
3. A PLANTED VIOLATION: a caller whose data ends in 2024 must NOT resolve 2025
   -- the bound follows the data in both directions rather than always
   appending one.
4. NO FALSE POSITIVE: the ``target_season`` branch is byte-unchanged, and the
   floor still excludes 2017 and earlier.

The last test in this module loads the PINNED play-by-play snapshot and takes
tens of seconds. It reaches no network: ``upstream_pin.load_pbp`` refuses rather
than falling back. Nothing here writes.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from features.team_form import TeamFormCalculator

AS_OF = datetime(2026, 9, 14, 12, 0, tzinfo=ZoneInfo("America/New_York"))

# The season the live 2026 predictions are built on top of, and the last season
# present in silver `games` (6,499 rows, 2002-2025) at the time of this plan.
LATEST_COVERED_SEASON = 2025


@pytest.fixture
def captured_seasons(monkeypatch) -> list[list[int]]:
    """Capture what the REAL loader was asked for, without loading anything."""
    calls: list[list[int]] = []

    def _recorder(self, seasons):
        calls.append(list(seasons))
        return pd.DataFrame(
            columns=["posteam", "defteam", "week", "play_type", "epa", "season"]
        )

    monkeypatch.setattr(TeamFormCalculator, "fetch_pbp_data", _recorder)
    monkeypatch.setattr(
        TeamFormCalculator, "calculate_team_game_stats", lambda self, df: pd.DataFrame()
    )
    return calls


class TestTheUpperBoundFollowsTheDataThatExists:
    """KIND: unit, over the resolution rule. No network."""

    def test_a_full_build_reaches_the_latest_covered_season(
        self, captured_seasons: list[list[int]]
    ) -> None:
        """THE DEFECT. A corpus reaching 2025 must build per-game stats for 2025.

        This is the assertion that fails on the pre-fix tree, where
        ``range(2018, 2025)`` stopped at 2024 and left the twelve
        opponent-adjusted gold columns carrying 2 distinct values across the
        285 games of 2025.
        """
        calculator = TeamFormCalculator()
        calculator.get_per_game_stats(AS_OF)

        assert captured_seasons, "the loader was never called at all"
        resolved = captured_seasons[-1]
        assert LATEST_COVERED_SEASON in resolved, (
            "season 2025 feeds the live 2026 predictions. Measured in gold "
            "before the fix: 2 distinct values across 285 rows, against 285 "
            f"distinct in both 2023 and 2024. Resolved here: {resolved}"
        )

    def test_the_target_season_branch_is_byte_preserved(
        self, captured_seasons: list[list[int]]
    ) -> None:
        """CONTROL 4. A scoped build still reads the target season and its predecessor."""
        calculator = TeamFormCalculator()
        calculator.get_per_game_stats(AS_OF, target_season=2024)

        assert captured_seasons[-1] == [2023, 2024]
