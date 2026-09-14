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

from features.team_form import TEAM_FORM_PER_GAME_FIRST_SEASON, TeamFormCalculator

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

    def test_the_bound_tracks_a_caller_whose_data_ends_somewhere_else(
        self, captured_seasons: list[list[int]]
    ) -> None:
        """The property a replacement literal could not have.

        Swapping 2025 for 2026 would satisfy the test above and rot on the same
        schedule. Only a derivation tracks a caller whose data ends in 2027.
        """
        calculator = TeamFormCalculator()
        calculator.get_per_game_stats(AS_OF, seasons=[2019, 2027])

        assert captured_seasons[-1] == [2019, 2027], (
            "the pool is the seasons the caller CARRIES, floored at the "
            "coverage floor -- not a range whose end is typed in"
        )

    def test_a_caller_whose_data_ends_in_2024_does_not_resolve_2025(
        self, captured_seasons: list[list[int]]
    ) -> None:
        """PLANTED VIOLATION: the bound follows the data in BOTH directions.

        A fix that simply appended one more season would pass the two tests
        above and would fabricate a season the caller does not have.
        """
        calculator = TeamFormCalculator()
        calculator.get_per_game_stats(AS_OF, seasons=[2023, 2024, 2024])

        assert captured_seasons[-1] == [2023, 2024]


class TestTheFloorAndTheTargetSeasonBranchAreUnchanged:
    """CONTROL 4: the two behaviours this fix must not disturb."""

    def test_the_floor_is_preserved(self, captured_seasons: list[list[int]]) -> None:
        """Seasons below the floor are dropped, and the floor is still 2018.

        Widening it would move roughly ninety gold columns that are a flat
        imputed constant for 2002-2017 -- outside this rung's declared change
        set. The floor is an INHERITED literal, not a data-coverage fact: the
        play-by-play pin reaches back to 2001. Plan 33.1-09 owns re-deciding it.
        """
        assert TEAM_FORM_PER_GAME_FIRST_SEASON == 2018

        calculator = TeamFormCalculator()
        calculator.get_per_game_stats(AS_OF, seasons=[2002, 2017, 2018, 2019])

        assert captured_seasons[-1] == [2018, 2019], (
            "a season below the floor must be dropped, not silently fetched"
        )

    def test_the_target_season_branch_is_byte_preserved(
        self, captured_seasons: list[list[int]]
    ) -> None:
        """A scoped build still reads the target season and its predecessor."""
        calculator = TeamFormCalculator()
        calculator.get_per_game_stats(AS_OF, target_season=2024)

        assert captured_seasons[-1] == [2023, 2024]

    def test_an_explicit_seasons_argument_wins_over_the_silver_fallback(
        self, captured_seasons: list[list[int]]
    ) -> None:
        """The caller already holds the games frame, so it names the pool.

        ``scripts/build_features`` reads ``feature_sources["games"]`` two lines
        before this call. Re-deriving the coverage from a second store read
        would be a second source of truth for the same fact.
        """
        calculator = TeamFormCalculator()
        calculator.get_per_game_stats(AS_OF, seasons=[2021, 2022])

        assert captured_seasons[-1] == [2021, 2022]


class TestTheFallbackRefusesRatherThanGuessing:
    """The refusal is typed to ESCAPE the caller's except tuple, and that matters."""

    def test_an_unreadable_silver_games_table_raises_a_named_refusal(
        self, monkeypatch, captured_seasons: list[list[int]]
    ) -> None:
        """And the refusal names the defect it replaced.

        ``scripts/build_features`` catches ValueError, KeyError, TypeError and
        AttributeError around this call and substitutes an EMPTY per-game
        frame, which drops the whole opponent-adjusted family from gold in
        silence. A refusal typed as any of those four would be swallowed into
        exactly the shape this fix removes.
        """
        import features.team_form as team_form_module

        def _unreadable(*args, **kwargs):
            raise FileNotFoundError("silver games is not there")

        monkeypatch.setattr(team_form_module, "load_dataframe", _unreadable)

        calculator = TeamFormCalculator()
        with pytest.raises(RuntimeError) as excinfo:
            calculator.get_per_game_stats(AS_OF)

        message = str(excinfo.value)
        assert "range(2018, 2025)" in message
        assert not isinstance(
            excinfo.value, ValueError | KeyError | TypeError | AttributeError
        ), (
            "the refusal must escape the caller's except tuple, or the build "
            "reports success with the opponent-adjusted family missing"
        )
        assert not captured_seasons, "nothing should have been fetched"

    def test_a_silver_table_with_no_season_above_the_floor_refuses(
        self, monkeypatch, captured_seasons: list[list[int]]
    ) -> None:
        """An empty pool is the defect, not a smaller answer."""
        import features.team_form as team_form_module

        monkeypatch.setattr(
            team_form_module,
            "load_dataframe",
            lambda *a, **k: pd.DataFrame({"season": [2002, 2010, 2017]}),
        )

        calculator = TeamFormCalculator()
        with pytest.raises(RuntimeError, match="carries no season at or above"):
            calculator.get_per_game_stats(AS_OF)

        assert not captured_seasons

    def test_a_games_frame_with_NO_season_column_refuses_rather_than_KeyError(
        self, monkeypatch, captured_seasons: list[list[int]]
    ) -> None:
        """Code review WR-07: the one input shape that failed OPEN.

        The two tests above cover an unreadable table and a table whose seasons
        are all below the floor. The gap between them was a table that LOADS
        FINE and has no ``season`` column: that read sat OUTSIDE the try, so it
        raised a bare ``KeyError`` -- which IS in ``scripts/build_features``'
        except tuple, so it was swallowed into a warning plus an empty per-game
        frame. A full-looking gold build with the entire opponent-adjusted
        family missing, which is the exact outcome the RuntimeError typing
        exists to prevent.
        """
        import features.team_form as team_form_module

        monkeypatch.setattr(
            team_form_module,
            "load_dataframe",
            lambda *a, **k: pd.DataFrame({"game_id": ["x"], "week": [1]}),
        )

        calculator = TeamFormCalculator()
        with pytest.raises(RuntimeError) as excinfo:
            calculator.get_per_game_stats(AS_OF)

        assert not isinstance(
            excinfo.value, ValueError | KeyError | TypeError | AttributeError
        ), (
            "a games frame with no season column must refuse OUTSIDE the "
            "caller's except tuple; a KeyError here is swallowed and the "
            "opponent-adjusted family silently disappears from gold"
        )
        assert "season" in str(excinfo.value)
        assert not captured_seasons, "nothing should have been fetched"


class TestTheLiveSeasonIsNOTCappedOut:
    """Code review WR-07(b): the reviewer asked for a ceiling. There is not one.

    A ceiling at the most recent COMPLETED season would drop the LIVE season from
    the per-game pool -- the same defect the hardcoded ``range(2018, 2025)`` had,
    rebuilt from a constant. The deliberate behaviour is that the pool follows the
    caller's data, and an uncaptured season above the pin's sealed zone stops the
    build by name rather than vanishing from it. This pins that choice so it is a
    decision on record instead of an omission.
    """

    def test_a_season_above_the_sealed_pin_stays_in_the_pool(self) -> None:
        from data.upstream_pin import SEALED_THROUGH_SEASON

        live = SEALED_THROUGH_SEASON + 1
        pool = TeamFormCalculator().per_game_seasons([2018, 2024, live])

        assert live in pool, (
            f"season {live} was dropped from the per-game pool. Silently "
            "excluding the live season is how season 2025's twelve "
            "opponent-adjusted columns came to carry two distinct values "
            "across 285 games; the correct failure is a named pin refusal."
        )
        assert pool[-1] == live


class TestTheRealPerGameFrameReachesTheLatestSeason:
    """KIND: integration against the PINNED play-by-play snapshot. Slow, no network."""

    def test_the_per_game_frame_carries_2025_rows(self) -> None:
        """The end-to-end statement: real per-game rows exist for 2025.

        The resolution tests above are about a list of integers. This one is
        about whether the per-game EPA frame the opponent adjuster consumes
        actually contains the season -- which is the fact the twelve gold
        columns depend on.
        """
        calculator = TeamFormCalculator()
        stats = calculator.get_per_game_stats(AS_OF, seasons=list(range(2018, 2026)))

        assert not stats.empty
        seasons = sorted(int(season) for season in stats["season"].unique())
        assert seasons[-1] == LATEST_COVERED_SEASON, (
            f"per-game seasons resolved: {seasons}"
        )
        assert int((stats["season"] == LATEST_COVERED_SEASON).sum()) > 0
