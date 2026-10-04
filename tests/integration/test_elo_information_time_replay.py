"""The independent Elo replay, run read-only against production silver (SPEC R4, D33.2-06).

WHAT THIS PROVES
----------------
``audit.elo_replay`` recomputes every pre-game Elo row in 2002-2025 plus the provisional
2026 rows from ONLY the results that ended at or before each game's day-before lock, and
this module asserts exact ``==`` equality against silver ``elo_game_snapshots`` on all six
compared columns. It is the CONTENT evidence that pairs with the Elo source's RULE-derived
provenance from Plan 33.2-01 (RESEARCH pitfall P3).

Exact equality is accepted as evidence only TOGETHER with two controls:

* the tied-kickoff permutation result -- the replay orders ties by ``(kickoff, game_id)``
  while the canonical chain sorts on ``kickoff_et`` alone (``scripts/build_elo.py:411``),
  so equality between the two means something only if tie order provably does not; and
* the planted post-lock result -- without it, the equality could be satisfied by a replay
  that ignores results entirely.

Any mismatch found here is a real defect to FIX at its cause (D33.2-06), never a tripwire
to register.

TIME DEPENDENCE, STATED
-----------------------
"Required to have a snapshot row" is judged at the real current instant: a completed game,
or an unplayed game whose lock has passed, must have one. That is the honest reading of the
live contract -- once a game's inputs are locked, a missing Elo row is a real defect -- and
it means this module goes red if the daily capture stops producing the provisional week.

READ-ONLY
---------
The production-reading tests opt into ``data_boundary_guard`` and
``artifacts_boundary_guard``: ``ratings.elo.save_ratings`` writing
``data/silver/elo_ratings.json`` from a read-only audit is the specific threat.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import cast

import pandas as pd
import pytest

from audit.elo_replay import (
    COMPARED_COLUMNS,
    ELO_SNAPSHOT_SOURCE,
    GAMES_SOURCE,
    REPO_ROOT,
    ReplayResult,
    duration_insensitivity_bound,
    replay,
)
from utils.game_lock import game_lock

SEASONS: tuple[int, ...] = tuple(range(2002, 2027))


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return pd.read_parquet(REPO_ROOT / GAMES_SOURCE)


@pytest.fixture(scope="module")
def snapshots() -> pd.DataFrame:
    frame = pd.read_parquet(REPO_ROOT / ELO_SNAPSHOT_SOURCE)
    # PRECONDITION: the canonical 2002-2025 chain plus provisional 2026 rows. Without them
    # there is nothing to compare against, and the run must halt rather than compare to an
    # empty frame.
    assert set(range(2002, 2026)) <= set(frame["season"]), "canonical chain missing"
    assert bool(frame.loc[frame["season"] == 2026, "is_provisional"].any()), (
        "no provisional 2026 snapshot rows: nothing live to compare"
    )
    return frame


@pytest.fixture(scope="module")
def as_of() -> datetime:
    return datetime.now(UTC)


@pytest.fixture(scope="module")
def baseline(
    games: pd.DataFrame, snapshots: pd.DataFrame, as_of: datetime
) -> ReplayResult:
    return replay(SEASONS, 4.0, games, snapshots, as_of=as_of)


def _sorted(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values("game_id").reset_index(drop=True)


class TestExactEquality:
    def test_every_row_matches_on_all_six_columns_and_writes_nothing(
        self,
        games: pd.DataFrame,
        snapshots: pd.DataFrame,
        as_of: datetime,
        data_boundary_guard,
        artifacts_boundary_guard,
    ) -> None:
        """Item 1 and item 8: zero mismatches, and the production digests do not move."""
        result = replay(SEASONS, 4.0, games, snapshots, as_of=as_of)

        assert len(result.mismatches) == 0
        assert result.mismatches.empty, result.mismatches.head(20).to_string()
        assert result.missing_snapshot_rows == (), result.missing_snapshot_rows
        assert result.orphan_snapshot_rows == (), result.orphan_snapshot_rows
        assert result.ok

    def test_the_comparison_is_not_vacuous(
        self, baseline: ReplayResult, snapshots: pd.DataFrame
    ) -> None:
        """Item 2: the evidence count is every stored row minus the reported vacuous ones."""
        in_scope = snapshots[snapshots["season"].isin(SEASONS)]
        assert baseline.compared > 0
        assert baseline.compared == len(in_scope) - baseline.vacuous
        assert len(baseline.replayed) == len(in_scope)
        assert baseline.compared_cells == (
            baseline.compared * len(COMPARED_COLUMNS) - baseline.vacuous_cells
        )

    def test_the_2026_provisional_rows_match_by_value(
        self, baseline: ReplayResult, snapshots: pd.DataFrame
    ) -> None:
        """The live rows are compared, not skipped -- whatever their physical row order."""
        provisional = set(snapshots.loc[snapshots["is_provisional"], "game_id"])
        assert provisional
        assert provisional <= set(baseline.replayed["game_id"])
        assert not set(baseline.mismatches["game_id"]) & provisional


class TestTheVacuousReport:
    def test_2002_week_one_rows_are_reported_and_excluded(
        self, baseline: ReplayResult, snapshots: pd.DataFrame
    ) -> None:
        """Item 3: the rows that agree with ANY replay are not counted as evidence."""
        week_one = snapshots[(snapshots["season"] == 2002) & (snapshots["week"] == 1)]
        assert baseline.vacuous == len(week_one) == 16
        # They are vacuous because they ARE the initial state.
        assert set(week_one["home_elo_pre"]) == {1500.0}
        assert set(week_one["away_elo_pre"]) == {1500.0}
        assert set(week_one["home_elo_uncertainty"]) == {350.0}

    def test_2002_hfa_used_is_reported_as_vacuous_cells(
        self, baseline: ReplayResult, snapshots: pd.DataFrame, games: pd.DataFrame
    ) -> None:
        season_2002 = snapshots[snapshots["season"] == 2002]
        # Since WINDOWS row 19 (dc7c34d) a neutral site carries 0.0 (the 2002 Super Bowl);
        # every other 2002 game carries the 48 init, times 0.54 when divisional.
        neutral_ids = set(games.loc[games["neutral_site"].eq(True), "game_id"])
        at_neutral = season_2002["game_id"].isin(neutral_ids)
        assert at_neutral.any()
        assert set(season_2002.loc[at_neutral, "hfa_used"]) == {0.0}
        assert set(season_2002.loc[~at_neutral, "hfa_used"]) <= {48.0, 48.0 * 0.54}
        assert baseline.vacuous_cells == len(season_2002) - baseline.vacuous


class TestTheDurationIsNotLoadBearing:
    @pytest.mark.parametrize("hours", [0.0, 8.0])
    def test_0_4_and_8_hours_are_identical(
        self,
        hours: float,
        games: pd.DataFrame,
        snapshots: pd.DataFrame,
        as_of: datetime,
        baseline: ReplayResult,
    ) -> None:
        """Item 4: 0 h and 8 h replay exactly what 4 h replays."""
        other = replay(SEASONS, hours, games, snapshots, as_of=as_of)
        pd.testing.assert_frame_equal(
            _sorted(other.replayed), _sorted(baseline.replayed), check_exact=True
        )
        assert other.mismatches.empty

    def test_the_band_is_measured_on_both_sides_of_its_edge(
        self,
        games: pd.DataFrame,
        snapshots: pd.DataFrame,
        as_of: datetime,
        baseline: ReplayResult,
    ) -> None:
        """The band is MEASURED under the new lock, not inherited from the Friday one.

        At the schedule-derived edge every value is unchanged (the boundary is inclusive);
        one second past it at least one value moves. That the edge sits far above 8 h is
        what makes the declared 4 h constant not load-bearing.
        """
        edge = duration_insensitivity_bound(games)
        assert edge > timedelta(hours=8)
        edge_hours = edge.total_seconds() / 3600
        at_edge = replay(SEASONS, edge_hours, games, snapshots, as_of=as_of)
        past_edge = replay(
            SEASONS, edge_hours + 1 / 3600, games, snapshots, as_of=as_of
        )

        pd.testing.assert_frame_equal(
            _sorted(at_edge.replayed), _sorted(baseline.replayed), check_exact=True
        )
        assert not past_edge.mismatches.empty


class TestThePositiveControl:
    def test_a_planted_post_lock_result_moves_a_rating(
        self,
        games: pd.DataFrame,
        snapshots: pd.DataFrame,
        as_of: datetime,
        baseline: ReplayResult,
    ) -> None:
        """Item 5: move a LATER result to end before a target's lock; the target must move.

        Target: the 2025 week-10 game of a team; planted: that team's next completed game,
        moved (in memory) to end one minute before the target's lock. If the replay ignored
        results, or selected them by anything but the lock, the target would not move.
        """
        season_2025 = cast(
            pd.DataFrame, games[(games["season"] == 2025) & games["home_score"].notna()]
        ).sort_values(by=["kickoff_et", "game_id"])
        target = season_2025[season_2025["week"] == 10].iloc[0]
        team = str(target["home_team"])
        later = season_2025[
            (season_2025["kickoff_et"] > target["kickoff_et"])
            & ((season_2025["home_team"] == team) | (season_2025["away_team"] == team))
        ].iloc[0]
        lock = pd.Timestamp(game_lock(target["kickoff_et"])).tz_convert("UTC")

        planted = games.copy()
        moved = planted["game_id"] == later["game_id"]
        planted.loc[moved, "kickoff_et"] = lock - timedelta(hours=4, minutes=1)

        result = replay(SEASONS, 4.0, planted, snapshots, as_of=as_of)
        before = baseline.replayed.set_index("game_id").loc[target["game_id"]]
        after = result.replayed.set_index("game_id").loc[target["game_id"]]

        assert before["home_elo_pre"] != after["home_elo_pre"]
        assert target["game_id"] in set(result.mismatches["game_id"])


class TestTiedKickoffs:
    def test_permuting_tied_kickoffs_changes_no_replayed_value(
        self,
        games: pd.DataFrame,
        snapshots: pd.DataFrame,
        as_of: datetime,
        baseline: ReplayResult,
    ) -> None:
        """Item 6: the result that licenses item 1's equality as evidence.

        Every tied group is processed in the REVERSE of its default order (input rows
        reversed and an explicit reversed tie rank). Every replayed value must be unchanged.
        """
        reversed_rank = {
            gid: rank
            for rank, gid in enumerate(
                sorted(games["game_id"].astype(str), reverse=True)
            )
        }
        permuted = replay(
            SEASONS,
            4.0,
            games.iloc[::-1].reset_index(drop=True),
            snapshots,
            as_of=as_of,
            tie_rank=reversed_rank,
        )

        assert baseline.tied_kickoff_groups > 0
        assert baseline.tied_kickoff_games > baseline.tied_kickoff_groups
        pd.testing.assert_frame_equal(
            _sorted(permuted.replayed), _sorted(baseline.replayed), check_exact=True
        )
        assert permuted.mismatches.empty
        assert "kickoff_et alone" in baseline.ordering_note


class TestTheMissingProvisionalRow:
    def test_a_2026_game_with_no_snapshot_row_is_named_and_fails(
        self, games: pd.DataFrame, snapshots: pd.DataFrame, as_of: datetime
    ) -> None:
        """Item 7: a locked 2026 game without an Elo row is reported by name, never skipped."""
        provisional = cast(pd.DataFrame, snapshots[snapshots["is_provisional"]])
        dropped = str(provisional["game_id"].iloc[0])
        lock = game_lock(games.loc[games["game_id"] == dropped, "kickoff_et"].iloc[0])
        judged_at = max(as_of, lock)
        thinned = cast(pd.DataFrame, snapshots[snapshots["game_id"] != dropped])

        result = replay(SEASONS, 4.0, games, thinned, as_of=judged_at)

        assert dropped in result.missing_snapshot_rows
        assert not result.ok
