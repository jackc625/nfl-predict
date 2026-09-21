"""The Elo replay is independent of the builder, and it recomputes what it claims to.

WHAT IS BEING GUARDED
---------------------
``audit/elo_replay.py`` is the CONTENT evidence for the Elo source's information times.
The provenance Plan 33.2-01 wrote for Elo is RULE-derived: it asserts the rule the builder
already follows. A rule checked with the same rule proves the rule, not the content
(RESEARCH pitfall P3). The replay recomputes every pre-game rating from only the results
that ended at or before that game's lock and compares it with ``==`` against silver
``elo_game_snapshots``.

That comparison is worthless if the replay reaches the thing it checks. A replay that
imported ``scripts.build_elo``'s chain would agree with it on every row and prove nothing.
So this module has two halves:

1. BEHAVIOUR on small in-memory fixtures: the replay matches the canonical chain, the
   at-lock boundary is inclusive, a missing snapshot row fails by name, a mismatch names
   its game and column, vacuous rows are excluded from the evidence count, tied kickoffs
   commute, and every permitted per-game formula actually fires.
2. AN AST SOURCE SCAN over ``audit/elo_replay.py`` across three forbidden families --
   imports, legacy table names, and forbidden ``ratings.elo`` calls -- with the four
   structural controls (non-vacuity, the assertion, a planted violation per name, and
   no false positive).

MATCH MODE FOR TABLE NAMES: EQUALITY, NEVER SUBSTRING. A table name is flagged when an
``ast.Constant`` string's VALUE EQUALS it. The replay's own docstring must name every
forbidden table and call (Plan 33.2-04 Task 1), and a module docstring is itself one
``ast.Constant``. Under substring matching the scan would flag the module it exists to
clear; under equality the docstring's value is the whole docstring and equals no name.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# The synthetic fixture: two seasons, six teams, three weeks each, with a Thursday
# game, a Monday game, two tied 13:00 ET kickoffs every week, and one unplayed
# week at the end of the second season (the provisional week).
# ---------------------------------------------------------------------------

_WEEK_SUNDAYS: dict[int, datetime] = {
    2002: datetime(2002, 9, 8),
    2003: datetime(2003, 9, 7),
}

# (week, day offset from that week's Sunday, ET hour, ET minute, home, away,
#  home score, away score). Offsets: -3 = Thursday, 0 = Sunday, +1 = Monday.
_SCHEDULE: tuple[tuple[int, int, int, int, str, str, int, int], ...] = (
    (1, 0, 13, 0, "BUF", "MIA", 24, 17),
    (1, 0, 13, 0, "NE", "NYJ", 20, 23),
    (1, 0, 16, 25, "DAL", "PHI", 31, 10),
    (2, -3, 20, 20, "MIA", "NE", 14, 13),
    (2, 0, 13, 0, "BUF", "DAL", 7, 28),
    (2, 0, 13, 0, "NYJ", "PHI", 21, 21),
    (3, 0, 13, 0, "NE", "BUF", 35, 3),
    (3, 0, 13, 0, "PHI", "MIA", 17, 20),
    (3, 1, 20, 15, "NYJ", "DAL", 27, 24),
)


def _kickoff_utc(season: int, week: int, day_offset: int, hour: int, minute: int):
    sunday = _WEEK_SUNDAYS[season] + timedelta(days=7 * (week - 1) + day_offset)
    local = datetime.combine(sunday.date(), time(hour, minute), tzinfo=ET)
    return pd.Timestamp(local).tz_convert("UTC")


def synthetic_games(*, unplayed_final_week: bool = True) -> pd.DataFrame:
    """Silver-``games``-shaped rows for 2002 and 2003.

    2003 week 3 is left UNPLAYED (null scores) when *unplayed_final_week* is True, so the
    fixture carries a provisional week exactly like the live 2026 table does.
    """
    rows: list[dict[str, object]] = []
    for season in (2002, 2003):
        for week, offset, hour, minute, home, away, hs, as_ in _SCHEDULE:
            if season == 2003:
                # Reverse the home side so 2003 is not a copy of 2002.
                home, away, hs, as_ = away, home, as_ + 3, hs
            unplayed = unplayed_final_week and season == 2003 and week == 3
            rows.append(
                {
                    "game_id": f"{season}_W{week:02d}_{away}@{home}",
                    "season": season,
                    "week": week,
                    "kickoff_et": _kickoff_utc(season, week, offset, hour, minute),
                    "home_team": home,
                    "away_team": away,
                    "home_score": float("nan") if unplayed else float(hs),
                    "away_score": float("nan") if unplayed else float(as_),
                }
            )
    frame = pd.DataFrame(rows)
    frame["kickoff_et"] = pd.to_datetime(frame["kickoff_et"], utc=True)
    return frame


def canonical_snapshots(games: pd.DataFrame) -> pd.DataFrame:
    """The CANONICAL chain's snapshots for *games* -- the thing the replay is checked against.

    Tests may import the builder; only ``audit/elo_replay.py`` may not. Running the real
    ``_process_chain`` and ``snapshot_upcoming_week`` here is what makes the fixture a
    cross-check of two independent derivations rather than of the replay against itself.
    """
    from scripts.build_elo import EloBuilder

    builder = EloBuilder()
    seasons = sorted(int(s) for s in games["season"].unique())
    rows, _ = builder._process_chain(seasons, games=games, learn_from=games)
    real = pd.DataFrame(rows)
    unplayed = games[games["home_score"].isna() | games["away_score"].isna()]
    frames = [real]
    for (season, week), _group in unplayed.groupby(["season", "week"]):
        frames.append(
            builder.snapshot_upcoming_week(int(season), int(week), games=games)
        )
    return pd.concat(frames, ignore_index=True)


AFTER_EVERYTHING = datetime(2004, 1, 1, tzinfo=UTC)


def _replay(**kwargs):
    from audit.elo_replay import replay

    kwargs.setdefault("as_of", AFTER_EVERYTHING)
    return replay(**kwargs)


# ---------------------------------------------------------------------------
# Behaviour
# ---------------------------------------------------------------------------


class TestTheReplayMatchesTheCanonicalChain:
    def test_zero_mismatches_on_every_row_including_the_provisional_week(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        result = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)

        assert result.mismatches.empty, result.mismatches.to_string()
        assert result.missing_snapshot_rows == ()
        assert result.orphan_snapshot_rows == ()
        assert result.compared > 0
        assert result.compared + result.vacuous == len(snapshots)
        assert result.ok

    def test_the_provisional_rows_are_compared_not_skipped(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        provisional_ids = set(snapshots.loc[snapshots["is_provisional"], "game_id"])
        assert len(provisional_ids) == 3
        result = _replay(seasons=[2003], games_df=games, snapshots_df=snapshots)
        assert provisional_ids <= set(result.replayed["game_id"])

    def test_vacuous_rows_are_the_first_season_week_one_rows(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        result = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)
        week_one_2002 = snapshots[
            (snapshots["season"] == 2002) & (snapshots["week"] == 1)
        ]

        assert result.vacuous == len(week_one_2002) == 3
        assert result.compared == len(snapshots) - 3
        # 2002 hfa_used is the initial value on every 2002 row: reported as vacuous CELLS
        # in the rows that stay in the evidence count.
        assert result.vacuous_cells == int((snapshots["season"] == 2002).sum()) - 3

    def test_a_season_filter_still_replays_from_the_chain_start(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        result = _replay(seasons=[2003], games_df=games, snapshots_df=snapshots)
        assert result.mismatches.empty, result.mismatches.to_string()
        assert result.vacuous == 0
        assert result.compared == int((snapshots["season"] == 2003).sum())


class TestTheLockBoundary:
    """A game ending exactly AT another game's lock is applied; one second later is not."""

    @staticmethod
    def _games_with_prior_end_at(offset: timedelta) -> pd.DataFrame:
        """DAL's week-1 game is moved so it ENDS at (DAL's week-2 lock + *offset*).

        DAL's week-1 opponent (PHI) next plays on the Sunday, after that lock, so moving
        the game changes nothing about either team's state except whether it is applied.
        """
        from utils.game_lock import game_lock

        games = synthetic_games()
        target = games["game_id"] == "2002_W02_DAL@BUF"
        prior = games["game_id"] == "2002_W01_PHI@DAL"
        lock = game_lock(games.loc[target, "kickoff_et"].iloc[0])
        end = pd.Timestamp(lock).tz_convert("UTC") + offset
        games.loc[prior, "kickoff_et"] = end - timedelta(hours=4)
        return games

    def _dal_week_two_pre(self, games: pd.DataFrame) -> float:
        result = _replay(
            seasons=[2002],
            games_df=games,
            snapshots_df=canonical_snapshots(synthetic_games()),
            duration_hours=4.0,
        )
        row = result.replayed.set_index("game_id").loc["2002_W02_DAL@BUF"]
        return float(row["away_elo_pre"])

    def test_at_lock_is_applied(self) -> None:
        at_lock = self._dal_week_two_pre(self._games_with_prior_end_at(timedelta(0)))
        reference = self._dal_week_two_pre(synthetic_games())
        assert at_lock == reference
        assert at_lock != 1500.0

    def test_one_second_after_the_lock_is_not_applied(self) -> None:
        late = self._dal_week_two_pre(
            self._games_with_prior_end_at(timedelta(seconds=1))
        )
        at_lock = self._dal_week_two_pre(self._games_with_prior_end_at(timedelta(0)))
        assert late != at_lock
        assert late == 1500.0


class TestTheFailureReports:
    def test_a_missing_snapshot_row_is_named_and_fails(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        dropped = "2003_W03_NE@BUF"
        assert dropped in set(snapshots["game_id"])
        thinned = snapshots[snapshots["game_id"] != dropped]

        result = _replay(seasons=[2003], games_df=games, snapshots_df=thinned)

        assert result.missing_snapshot_rows == (dropped,)
        assert not result.ok

    def test_an_unplayed_game_whose_lock_has_not_passed_is_not_required(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        thinned = snapshots[~snapshots["is_provisional"]]
        before_week_three = datetime(2003, 9, 15, tzinfo=UTC)

        result = _replay(
            seasons=[2003],
            games_df=games,
            snapshots_df=thinned,
            as_of=before_week_three,
        )

        assert result.missing_snapshot_rows == ()
        assert result.ok

    def test_an_orphan_snapshot_row_is_named_and_fails(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        orphan = snapshots.iloc[[5]].copy()
        orphan["game_id"] = "2003_W09_XXX@YYY"
        orphan["season"] = 2003
        planted = pd.concat([snapshots, orphan], ignore_index=True)

        result = _replay(seasons=[2003], games_df=games, snapshots_df=planted)

        assert result.orphan_snapshot_rows == ("2003_W09_XXX@YYY",)
        assert not result.ok

    def test_a_mismatch_names_the_game_the_column_and_both_values(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games).copy()
        target = snapshots["game_id"] == "2002_W03_BUF@NE"
        stored = float(snapshots.loc[target, "away_elo_uncertainty"].iloc[0])
        snapshots.loc[target, "away_elo_uncertainty"] = stored + 1e-9

        result = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)

        assert len(result.mismatches) == 1
        row = result.mismatches.iloc[0]
        assert row["game_id"] == "2002_W03_BUF@NE"
        assert row["column"] == "away_elo_uncertainty"
        assert row["replayed"] == stored
        assert row["stored"] == stored + 1e-9
        assert not result.ok

    def test_an_empty_snapshot_source_is_refused_not_compared(self) -> None:
        from audit.elo_replay import EmptySnapshotSourceError

        games = synthetic_games()
        empty = canonical_snapshots(games).iloc[0:0]
        with pytest.raises(EmptySnapshotSourceError):
            _replay(seasons=[2002], games_df=games, snapshots_df=empty)


class TestTheDeclaredConstantsAreNotLoadBearing:
    @pytest.mark.parametrize("hours", [0.0, 4.0, 8.0])
    def test_duration_does_not_change_any_replayed_value(self, hours: float) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        result = _replay(
            seasons=[2002, 2003],
            games_df=games,
            snapshots_df=snapshots,
            duration_hours=hours,
        )
        assert result.mismatches.empty, result.mismatches.to_string()

    def test_permuting_tied_kickoffs_changes_no_replayed_value(self) -> None:
        games = synthetic_games()
        snapshots = canonical_snapshots(games)
        forward = _replay(seasons=[2002, 2003], games_df=games, snapshots_df=snapshots)
        reversed_rank = {
            gid: rank for rank, gid in enumerate(sorted(games["game_id"], reverse=True))
        }
        backward = _replay(
            seasons=[2002, 2003],
            games_df=games.iloc[::-1].reset_index(drop=True),
            snapshots_df=snapshots,
            tie_rank=reversed_rank,
        )

        assert forward.tied_kickoff_groups > 0
        pd.testing.assert_frame_equal(
            forward.replayed.sort_values("game_id").reset_index(drop=True),
            backward.replayed.sort_values("game_id").reset_index(drop=True),
            check_exact=True,
        )
