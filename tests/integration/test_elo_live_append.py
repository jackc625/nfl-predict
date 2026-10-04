"""A live append writes one Elo table, equal to an independent replay, in one row order.

WHAT THIS MODULE PROVES (Plan 33.2-05, D33.2-22, SPEC R11)
----------------------------------------------------------
1. A fixture full rebuild followed by a live append for one week writes ONLY
   ``elo_game_snapshots`` -- no side table in the silver layer, no side table in the
   DuckDB -- and the appended week's values EQUAL ``audit.elo_replay.replay`` on the same
   fixture. The replay is the independent recomputation Plan 33.2-04 built: it may not
   import the builder, so agreement is two derivations agreeing, not one agreeing with
   itself.
2. The live PIPELINE STEP, driven for real against a sandbox, writes the same one table.
   That is what proves both narrowed call sites -- the step and the verb -- still fit.
3. ORCHESTRATOR-ASSIGNED: every write persists ``(season, kickoff_et, game_id)`` order in
   BOTH the parquet and the DuckDB copy, including the append that REWRITES earlier-week
   rows. That append is the production defect measured in 2026: latest-wins removed a
   provisional week-1 row and appended its real successor AFTER the provisional week-2
   rows, and ``LeakageGate.check_elo_ordering`` read 30 per-team inversions. The fixture
   reproduces that exact sequence and first proves the plain upsert WOULD invert it, so
   the ordering assertion cannot pass vacuously.

Every test runs in a ``tmp_path`` sandbox and opts into both boundary guards: a test
that proved the writer correct by writing production would have proved nothing.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from tests.fixtures.elo_sandbox import (
    make_season_games,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
    seed_sandbox_games,
)

# The replay's chain always starts in 2002, so the fixture does too: 2002 is the
# burn-in season, 2003 the live one.
BURN_IN_SEASON = 2002
LIVE_SEASON = 2003
AFTER_EVERYTHING = datetime(2004, 1, 1, tzinfo=UTC)

# Every file a live append may leave in the sandbox silver layer. ``games.parquet`` is
# the seeded input; the pointer and the generation tree are the publisher's own.
ALLOWED_SILVER_ENTRIES: frozenset[str] = frozenset(
    {
        "games.parquet",
        "elo_game_snapshots.parquet",
        "elo_generation.json",
        "elo_generations",
    }
)

pytestmark = pytest.mark.usefixtures("data_boundary_guard", "artifacts_boundary_guard")


def _corpus(*, live_graded_weeks: tuple[int, ...], live_weeks: int = 2) -> pd.DataFrame:
    return pd.concat(
        [
            make_season_games(BURN_IN_SEASON, weeks=3),
            make_season_games(
                LIVE_SEASON, weeks=live_weeks, graded_weeks=live_graded_weeks
            ),
        ],
        ignore_index=True,
    )


def _sandbox_tables() -> set[str]:
    """Every table in the SANDBOX DuckDB, read so a connection failure RAISES."""
    from data.storage import get_db_connection

    rows = (
        get_db_connection()
        .execute("SELECT table_name FROM information_schema.tables")
        .fetchall()
    )
    return {str(row[0]).lower() for row in rows}


def _canonical_order(stored: pd.DataFrame, games: pd.DataFrame) -> list[str]:
    """The game_ids of *stored* in (season, kickoff_et, game_id) order, computed here."""
    keyed = stored.merge(games[["game_id", "kickoff_et"]], on="game_id", how="left")
    assert keyed["kickoff_et"].notna().all(), "a stored row has no kickoff in games"
    keyed["kickoff_et"] = pd.to_datetime(keyed["kickoff_et"], utc=True)
    return (
        keyed.sort_values(["season", "kickoff_et", "game_id"], kind="stable")["game_id"]
        .astype(str)
        .tolist()
    )


class TestALiveAppendWritesOnlyTheSnapshotTable:
    """Full rebuild, then one week appended: one table written, equal to the replay."""

    def test_the_appended_week_equals_the_independent_replay(
        self, tmp_path, monkeypatch
    ) -> None:
        from audit.elo_replay import replay

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(sandbox, _corpus(live_graded_weeks=(1,)))
        snapshots = builder.build_all_ratings(start_season=BURN_IN_SEASON)
        builder.save_full_rebuild(snapshots, start_season=BURN_IN_SEASON)

        # Week 2 lands: the schedule is re-seeded with its scores and the live
        # season is re-derived and appended through the real verb.
        graded = _corpus(live_graded_weeks=(1, 2))
        live = sandbox_builder(sandbox, graded)
        update = live.update_current_season(season=LIVE_SEASON)
        live.save_live_append(update.season, snapshots=update.snapshots)

        silver_entries = {path.name for path in (sandbox / "silver").iterdir()}
        assert silver_entries <= ALLOWED_SILVER_ENTRIES, (
            f"the live append left {sorted(silver_entries - ALLOWED_SILVER_ENTRIES)} "
            "in the silver layer; the only Elo store is elo_game_snapshots."
        )
        tables = _sandbox_tables()
        assert tables, "the sandbox DuckDB lists no tables, so absence proves nothing"
        assert "elo_game_snapshots" in tables
        assert not tables & {
            "games_with_elo",
            "elo_rating_history",
            "elo_ratings_current",
        }

        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        week_two = stored[(stored["season"] == LIVE_SEASON) & (stored["week"] == 2)]
        assert len(week_two) == 4, f"week 2 should carry four rows, got {len(week_two)}"

        result = replay(
            [LIVE_SEASON], games_df=graded, snapshots_df=stored, as_of=AFTER_EVERYTHING
        )
        assert result.mismatches.empty, result.mismatches.to_string()
        assert result.missing_snapshot_rows == ()
        assert result.orphan_snapshot_rows == ()
        assert set(week_two["game_id"]) <= set(result.replayed["game_id"]), (
            "the appended week was not among the rows the replay compared"
        )
        assert result.ok

    def test_the_live_pipeline_step_writes_only_the_snapshot_table(
        self, tmp_path, monkeypatch
    ) -> None:
        """The real ``step_build_elo``, pointed at the sandbox, end to end."""
        import scripts.build_elo as build_elo_mod
        from pipeline.steps import step_build_elo

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        seed_sandbox_games(_corpus(live_graded_weeks=(1, 2)))
        real_builder = build_elo_mod.EloBuilder
        monkeypatch.setattr(
            build_elo_mod, "EloBuilder", lambda: real_builder(data_root=sandbox)
        )
        monkeypatch.setattr(
            build_elo_mod, "get_current_nfl_week", lambda: (LIVE_SEASON, 3)
        )

        step_build_elo()

        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert set(stored["season"]) == {LIVE_SEASON} and len(stored) == 8
        silver_entries = {path.name for path in (sandbox / "silver").iterdir()}
        assert silver_entries <= ALLOWED_SILVER_ENTRIES, sorted(silver_entries)


class TestEveryWriteLandsInCanonicalRowOrder:
    """Orchestrator-assigned: (season, kickoff_et, game_id) in parquet AND DuckDB."""

    @staticmethod
    def _provisional_then_real(tmp_path, monkeypatch):
        """Reproduce the 2026 sequence: provisional weeks 1 and 2, then week 1 lands."""
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        ungraded = _corpus(live_graded_weeks=())
        builder = sandbox_builder(sandbox, ungraded)
        rebuilt = builder.build_all_ratings(start_season=BURN_IN_SEASON)
        builder.save_full_rebuild(rebuilt, start_season=BURN_IN_SEASON)

        for week in (1, 2):
            provisional = builder.snapshot_upcoming_week(LIVE_SEASON, week)
            builder.save_live_append(LIVE_SEASON, snapshots=provisional)
        before = read_sandbox_table(sandbox, "elo_game_snapshots")

        graded = _corpus(live_graded_weeks=(1,))
        live = sandbox_builder(sandbox, graded)
        update = live.update_current_season(season=LIVE_SEASON)
        return sandbox, graded, live, update, before

    def test_the_fixture_would_invert_without_the_ordering(
        self, tmp_path, monkeypatch
    ) -> None:
        """Non-vacuity: plain latest-wins on this fixture DOES put week 1 after week 2."""
        _sandbox, graded, _live, update, before = self._provisional_then_real(
            tmp_path, monkeypatch
        )
        new = update.snapshots
        latest_wins = pd.concat(
            [before[~before["game_id"].isin(new["game_id"])], new], ignore_index=True
        )
        naive = latest_wins["game_id"].astype(str).tolist()
        assert naive != _canonical_order(latest_wins, graded), (
            "plain latest-wins already produced canonical order on this fixture, so the "
            "ordering test below would pass whether or not the writer orders anything."
        )

    def test_an_append_that_rewrites_earlier_weeks_leaves_both_stores_ordered(
        self, tmp_path, monkeypatch
    ) -> None:
        from data.storage import get_db_connection

        sandbox, graded, live, update, _before = self._provisional_then_real(
            tmp_path, monkeypatch
        )
        live.save_live_append(LIVE_SEASON, snapshots=update.snapshots)

        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        expected = _canonical_order(stored, graded)
        assert stored["game_id"].astype(str).tolist() == expected, (
            "the snapshot PARQUET is not in (season, kickoff_et, game_id) order after "
            "an append that rewrote earlier-week rows."
        )
        in_duckdb = get_db_connection().fetch_df(
            "SELECT game_id FROM elo_game_snapshots"
        )
        assert in_duckdb["game_id"].astype(str).tolist() == expected, (
            "the snapshot DUCKDB copy is not in the same order as the parquet."
        )
        # Anti-vacuity: this append really did rewrite week-1 rows over provisional ones.
        week_one = stored[(stored["season"] == LIVE_SEASON) & (stored["week"] == 1)]
        assert len(week_one) == 4 and not week_one["is_provisional"].any()

    def test_the_stored_order_passes_the_leakage_gate_ordering_check(
        self, tmp_path, monkeypatch
    ) -> None:
        """The same check the production audit test runs, over the written table."""
        from features.validation import LeakageGate

        sandbox, graded, live, update, _before = self._provisional_then_real(
            tmp_path, monkeypatch
        )
        live.save_live_append(LIVE_SEASON, snapshots=update.snapshots)

        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        merged = stored.merge(
            graded[["game_id", "kickoff_et"]], on="game_id", how="left"
        )
        home = merged[["season", "kickoff_et", "home_team"]].rename(
            columns={"kickoff_et": "game_date", "home_team": "team"}
        )
        away = merged[["season", "kickoff_et", "away_team"]].rename(
            columns={"kickoff_et": "game_date", "away_team": "team"}
        )
        LeakageGate().check_elo_ordering(pd.concat([home, away], ignore_index=False))

    def test_a_full_rebuild_is_written_in_the_same_order(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = _corpus(live_graded_weeks=(1, 2))
        builder = sandbox_builder(sandbox, games)
        rebuilt = builder.build_all_ratings(start_season=BURN_IN_SEASON)
        builder.save_full_rebuild(rebuilt.iloc[::-1], start_season=BURN_IN_SEASON)

        stored = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert stored["game_id"].astype(str).tolist() == _canonical_order(stored, games)

    def test_a_snapshot_with_no_scheduled_game_is_refused_by_name(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import EloSnapshotOrderError

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(sandbox, _corpus(live_graded_weeks=(1,)))
        update = builder.build_season_frames(LIVE_SEASON)
        orphan = update.snapshots.copy()
        orphan.loc[orphan.index[0], "game_id"] = f"{LIVE_SEASON}_W09_ARI@ATL"

        with pytest.raises(EloSnapshotOrderError) as excinfo:
            builder.save_live_append(LIVE_SEASON, snapshots=orphan)

        assert f"{LIVE_SEASON}_W09_ARI@ATL" in str(excinfo.value)
        assert read_sandbox_table(sandbox, "elo_game_snapshots").empty, (
            "the refusal half-applied: rows were written before it raised."
        )


class TestTheLiveWeekRanksTheRatingsTrainingRanked:
    """Review WR-08 (owner ruling B): the live rank/percentile equals the training-time one.

    The daily step now stores a week-start row for every unplayed game of the slate's week.
    The week's first game is played, the second is tonight's slate, the last two are later.
    Training is the same week once every game is played and the chain is rebuilt. Week-1
    margins are chosen so the slate's teams sit mid-table, where a stale rating moves a rank.
    """

    RANK_COLUMNS = (
        "home_elo_rank",
        "away_elo_rank",
        "home_elo_percentile",
        "away_elo_percentile",
    )

    @staticmethod
    def _games(*, week_two_played: int) -> pd.DataFrame:
        live = make_season_games(LIVE_SEASON, weeks=2)
        week_two = live.index[live["week"] == 2]
        live.loc[week_two[week_two_played:], ["home_score", "away_score"]] = None
        week_one = live.index[live["week"] == 1]
        live.loc[week_one, "home_score"] = [27.0, 20.0, 35.0, 0.0]
        live.loc[week_one, "away_score"] = [17.0, 17.0, 3.0, 45.0]
        return pd.concat(
            [make_season_games(BURN_IN_SEASON, weeks=3), live], ignore_index=True
        )

    def test_the_slate_ranks_equal_training_and_differ_from_a_slate_only_store(
        self, tmp_path, monkeypatch
    ) -> None:
        import scripts.build_elo as build_elo_mod
        from features.elo_features import EloFeatureBuilder
        from pipeline.steps import persist_current_season_elo

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        tonight = self._games(week_two_played=1)
        builder = sandbox_builder(sandbox, tonight)
        builder.save_full_rebuild(
            builder.build_all_ratings(start_season=BURN_IN_SEASON),
            start_season=BURN_IN_SEASON,
        )
        real_builder = build_elo_mod.EloBuilder
        monkeypatch.setattr(
            build_elo_mod, "EloBuilder", lambda: real_builder(data_root=sandbox)
        )
        monkeypatch.setattr(
            build_elo_mod, "get_current_nfl_week", lambda: (LIVE_SEASON, 2)
        )
        week_two = tonight[(tonight["season"] == LIVE_SEASON) & (tonight["week"] == 2)]
        slate = str(week_two["game_id"].iloc[1])
        persist_current_season_elo(frozenset({slate}))
        live = read_sandbox_table(sandbox, "elo_game_snapshots")

        stored_week_two = live[(live["season"] == LIVE_SEASON) & (live["week"] == 2)]
        assert set(stored_week_two["game_id"]) == set(week_two["game_id"])
        assert int(stored_week_two["is_provisional"].sum()) == 3

        played = tonight.copy()
        played[["home_score", "away_score"]] = played[
            ["home_score", "away_score"]
        ].fillna({"home_score": 20.0, "away_score": 10.0})
        seed_sandbox_games(played)
        trained = real_builder(data_root=sandbox).build_all_ratings(
            start_season=BURN_IN_SEASON
        )

        # Each week-start row equals the real pre-game row that later replaces it.
        ratings = ["game_id", "home_elo_pre", "away_elo_pre"]
        pd.testing.assert_frame_equal(
            stored_week_two[ratings].sort_values("game_id").reset_index(drop=True),
            trained.loc[
                (trained["season"] == LIVE_SEASON) & (trained["week"] == 2), ratings
            ]
            .sort_values("game_id")
            .reset_index(drop=True),
        )

        game = week_two.loc[
            week_two["game_id"] == slate,
            ["game_id", "season", "week", "home_team", "away_team"],
        ]
        ranks = EloFeatureBuilder()._add_rank_features
        live_ranks = ranks(game, live)[list(self.RANK_COLUMNS)]
        trained_ranks = ranks(game, trained)[list(self.RANK_COLUMNS)]
        pd.testing.assert_frame_equal(live_ranks, trained_ranks)

        # Non-vacuity: the store before this fix held the slate's row alone for week 2.
        slate_only = live[
            ~(
                (live["season"] == LIVE_SEASON)
                & (live["week"] == 2)
                & live["is_provisional"]
                & (live["game_id"] != slate)
            )
        ]
        stale_ranks = ranks(game, slate_only)[list(self.RANK_COLUMNS)]
        assert not stale_ranks.equals(live_ranks), (
            "a slate-only store ranks the same here, so the equality above proves nothing"
        )
