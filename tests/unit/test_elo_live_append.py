"""The live append's edge cases: an empty season, and the pending-rows counter.

R3'S ZERO-COMPLETED-GAMES CASE IS NOT A FAILURE
-----------------------------------------------
A current season with no completed games persists zero new rows and returns normally.
That case is the live 2026 cold start itself, so a guard that could not distinguish
"nothing to write" from "wrote nothing" would make the phase's own subject un-runnable.
It is asserted here WITH its negative control -- the same frame carrying a single
completed game writes exactly one row -- because "zero rows were written" is trivially
satisfied by a writer that writes nothing ever.

``pending_snapshot_rows`` IS THE INSTRUMENT, so it is tested as one: non-zero after a
computation that has not been persisted, zero after one that has.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tests.fixtures.elo_sandbox import (
    make_season_games,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
)

LIVE_SEASON = 2026


def _append(builder, season: int):
    """Compute the season and persist it through the live verb."""
    update = builder.build_season_frames(season)
    builder.save_live_append(
        season,
        snapshots=update.snapshots,
        games_with_elo=update.games_with_elo,
        rating_history=update.rating_history,
    )
    return update


class TestASeasonWithNoCompletedGames:
    """Zero rows written, nothing raised -- and the control that makes it mean something."""

    def test_zero_completed_games_writes_zero_rows_and_does_not_raise(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(LIVE_SEASON, weeks=2, graded_weeks=())
        builder = sandbox_builder(sandbox, games)

        raised: Exception | None = None
        try:
            update = _append(builder, LIVE_SEASON)
        except Exception as exc:
            raised = exc
            update = None

        assert raised is None, (
            "a season with zero completed games must not raise. The live 2026 capture "
            f"reaches the Elo step in exactly this state. Raised: {raised!r}"
        )
        assert update is not None and len(update.snapshots) == 0
        assert read_sandbox_table(sandbox, "elo_game_snapshots").empty, (
            "zero completed games must persist zero snapshot rows"
        )

    def test_the_same_frame_with_one_completed_game_writes_exactly_one_row(
        self, tmp_path, monkeypatch
    ) -> None:
        """The negative control. Without it, 'zero rows' proves only inaction."""
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(LIVE_SEASON, weeks=2, graded_weeks=())
        # Grade exactly one game, leaving the other seven ungraded.
        games.loc[games.index[0], "home_score"] = 24.0
        games.loc[games.index[0], "away_score"] = 20.0
        builder = sandbox_builder(sandbox, games)

        _append(builder, LIVE_SEASON)

        written = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert len(written) == 1, (
            f"one completed game must persist exactly one snapshot row, got "
            f"{len(written)}"
        )
        assert written.iloc[0]["game_id"] == games.iloc[0]["game_id"]

    def test_an_empty_season_still_publishes_a_complete_generation(
        self, tmp_path, monkeypatch
    ) -> None:
        """An empty season is a legitimate generation, not an incomplete one."""
        from scripts.build_elo import read_elo_generation_pointer

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = make_season_games(LIVE_SEASON, weeks=2, graded_weeks=())
        builder = sandbox_builder(sandbox, games)

        _append(builder, LIVE_SEASON)

        pointer = read_elo_generation_pointer(sandbox / "silver")
        assert pointer is not None, (
            "an empty season must still publish: the generation is complete, it simply "
            "has no rows in two of its members."
        )
        assert pointer["season"] == LIVE_SEASON
        assert pointer["mode"] == "live_append"


class TestThePendingSnapshotCounter:
    """The instrument the step's refusal reads."""

    def test_the_counter_is_non_zero_after_a_computation_that_was_not_persisted(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(sandbox, make_season_games(LIVE_SEASON, weeks=2))

        builder.update_current_season(season=LIVE_SEASON)

        assert builder.pending_snapshot_rows == 8, (
            "eight graded games were computed and none were written, so eight rows "
            f"are pending. Got {builder.pending_snapshot_rows}."
        )

    def test_the_counter_returns_to_zero_after_a_live_append(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(sandbox, make_season_games(LIVE_SEASON, weeks=2))

        update = builder.update_current_season(season=LIVE_SEASON)
        builder.save_live_append(
            update.season,
            snapshots=update.snapshots,
            games_with_elo=update.games_with_elo,
            rating_history=update.rating_history,
        )

        assert builder.pending_snapshot_rows == 0

    def test_the_counter_is_zero_for_a_season_that_computed_nothing(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(
            sandbox, make_season_games(LIVE_SEASON, weeks=2, graded_weeks=())
        )

        builder.update_current_season(season=LIVE_SEASON)

        assert builder.pending_snapshot_rows == 0, (
            "nothing was computed, so nothing is pending. Conflating this with 'wrote "
            "nothing' is what would make the cold start un-runnable."
        )


class TestTheLiveAppendReadsItsOwnDataRoot:
    """The builder's data root must bind every read AND every write it performs.

    ``EloRatingSystem.load_ratings`` defaults to the PRODUCTION silver path regardless
    of the builder's root, so a sandboxed builder would silently seed itself from the
    live ``elo_ratings.json``. That is a read, not a write, so the boundary guard would
    never see it -- and the test would quietly depend on production state.
    """

    def test_the_live_update_does_not_read_the_production_ratings_file(
        self, tmp_path, monkeypatch
    ) -> None:
        import ratings.elo as elo_mod

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = sandbox_builder(sandbox, make_season_games(LIVE_SEASON, weeks=1))

        seen: list[str] = []
        real_load = elo_mod.EloRatingSystem.load_ratings

        def _recording_load(self, filepath=None):
            seen.append(str(filepath))
            return real_load(self, filepath)

        monkeypatch.setattr(elo_mod.EloRatingSystem, "load_ratings", _recording_load)
        builder.update_current_season(season=LIVE_SEASON)

        for path in seen:
            assert str(sandbox) in path, (
                "the live update read an Elo ratings file outside its own data root "
                f"({path}). A sandboxed builder that seeds from production state is "
                "not sandboxed."
            )


@pytest.mark.parametrize("season", [2024, 2025, 2026])
def test_a_live_append_writes_only_the_season_it_names(
    tmp_path, monkeypatch, season
) -> None:
    """Whatever season is named, no other season appears in the written table."""
    sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
    games = pd.concat(
        [make_season_games(year, weeks=1) for year in (2024, 2025, 2026)],
        ignore_index=True,
    )
    builder = sandbox_builder(sandbox, games)

    _append(builder, season)

    written = read_sandbox_table(sandbox, "elo_game_snapshots")
    assert set(written["season"].unique()) == {season}, (
        f"save_live_append({season}) wrote seasons {sorted(written['season'].unique())}"
    )
