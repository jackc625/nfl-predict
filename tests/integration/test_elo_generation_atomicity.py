"""Five independently-atomic files are NOT one atomic Elo state (T-33-16d).

THE FAILURE THIS MODULE EXISTS FOR
----------------------------------
``save_results`` wrote five artifacts with five separate calls. Each individual write
was atomic; the SET was not. A crash between any two of them left a MIXED generation --
snapshots from the new build sitting beside a rating history from the old one -- and
nothing on disk said so. Every consumer downstream would have read that mixture as a
single coherent Elo state, because from the outside it is indistinguishable from one.

``publish_elo_generation`` closes that by making the five atomic TOGETHER: all five are
STAGED under one generation id, VALIDATED together, published, and only then does a
single generation pointer move in one ``os.replace``. A crash before the pointer moves
leaves the previous generation intact and named; a crash between two staged writes
leaves the live artifacts untouched, because nothing live was written at all.

WHY THE INTERRUPTION IS PARAMETERISED OVER ALL FIVE POINTS
----------------------------------------------------------
Testing one interruption point proves one interruption point. The claim being made is
about the SET, so the failure is planted between every pair in turn. A publisher that
happened to be safe at point 3 and unsafe at point 4 would pass a single-point test and
ship the defect.
"""

from __future__ import annotations

import pytest

from tests.fixtures.elo_sandbox import (
    make_season_games,
    per_season_row_digests,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
)

ROW_TABLES = ("elo_game_snapshots", "games_with_elo", "elo_rating_history")


def _live_digests(sandbox) -> dict[str, dict[str, str]]:
    """Per-season row digests of every live row table, as one comparable mapping."""
    return {
        table: per_season_row_digests(read_sandbox_table(sandbox, table))
        for table in ROW_TABLES
    }


def _seed_first_generation(sandbox, season: int = 2025, weeks: int = 2):
    """Publish generation ONE through the real live-append verb."""
    builder = sandbox_builder(sandbox, make_season_games(season, weeks=weeks))
    update = builder.update_current_season(season=season)
    builder.save_live_append(
        season,
        snapshots=update.snapshots,
        games_with_elo=update.games_with_elo,
        rating_history=update.rating_history,
    )
    return builder


def _failing_stage_writer(fail_on_call: int):
    """A stage writer that writes normally until *fail_on_call*, then raises."""
    from scripts.build_elo import default_stage_writer

    state = {"calls": 0}

    def writer(path, payload) -> None:
        state["calls"] += 1
        if state["calls"] == fail_on_call:
            raise OSError(
                f"planted staging failure on staged write #{fail_on_call} ({path.name})"
            )
        default_stage_writer(path, payload)

    return writer


class TestAnInterruptedPublishLeavesThePreviousGenerationIntact:
    """The five interruption points, each asserted on live content and on the pointer."""

    def test_the_seeded_generation_is_readable(self, tmp_path, monkeypatch) -> None:
        """Anti-vacuity: every assertion below compares against THIS state."""
        from scripts.build_elo import read_elo_generation_pointer

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        _seed_first_generation(sandbox)

        digests = _live_digests(sandbox)
        assert digests["elo_game_snapshots"], (
            "generation one wrote no snapshot rows, so an 'unchanged' assertion "
            "afterwards would compare two empty mappings and prove nothing."
        )
        pointer = read_elo_generation_pointer(sandbox / "silver")
        assert pointer is not None and pointer["generation_id"]

    @pytest.mark.parametrize("fail_on_call", [1, 2, 3, 4, 5])
    def test_a_staging_failure_never_touches_the_live_artifacts(
        self, tmp_path, monkeypatch, fail_on_call
    ) -> None:
        from scripts.build_elo import (
            new_generation_id,
            publish_elo_generation,
            read_elo_generation_pointer,
        )

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = _seed_first_generation(sandbox)
        silver = sandbox / "silver"

        before_digests = _live_digests(sandbox)
        before_pointer = read_elo_generation_pointer(silver)

        # A SECOND generation's worth of content, materially different from the first.
        wider = builder.update_current_season(season=2025)
        staged = {
            "elo_game_snapshots": wider.snapshots,
            "games_with_elo": wider.games_with_elo,
            "elo_rating_history": wider.rating_history,
            "elo_ratings_current": builder.elo_system.get_current_ratings(),
            "elo_ratings": builder.elo_system.ratings_state(),
        }

        published_live = {"called": False}

        def _publish_live() -> None:
            published_live["called"] = True

        with pytest.raises(OSError, match="planted staging failure"):
            publish_elo_generation(
                staged,
                new_generation_id(),
                silver_root=silver,
                publish_live=_publish_live,
                stage_writer=_failing_stage_writer(fail_on_call),
            )

        assert not published_live["called"], (
            "the live publish ran even though staging failed. Staging exists precisely "
            "so that a failure before validation cannot reach the live artifacts."
        )
        assert _live_digests(sandbox) == before_digests, (
            f"a staging failure at write #{fail_on_call} moved a LIVE row table. The "
            "previous generation must still read back byte-for-byte on its per-season "
            "row digests."
        )
        assert read_elo_generation_pointer(silver) == before_pointer, (
            f"the generation pointer moved after a staging failure at write "
            f"#{fail_on_call}."
        )


class TestACompletingPublishMovesThePointer:
    """The positive arm: without it, a publisher that never publishes would pass."""

    def test_a_completing_publish_advances_the_pointer_and_the_live_rows(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import read_elo_generation_pointer

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = _seed_first_generation(sandbox, weeks=2)
        silver = sandbox / "silver"

        first_pointer = read_elo_generation_pointer(silver)
        first_rows = len(read_sandbox_table(sandbox, "elo_game_snapshots"))

        # Week 3 arrives: re-derive the season and publish generation two.
        from tests.fixtures.elo_sandbox import seed_sandbox_games

        seed_sandbox_games(make_season_games(2025, weeks=3))
        update = builder.update_current_season(season=2025)
        builder.save_live_append(
            2025,
            snapshots=update.snapshots,
            games_with_elo=update.games_with_elo,
            rating_history=update.rating_history,
        )

        second_pointer = read_elo_generation_pointer(silver)
        assert second_pointer is not None
        assert second_pointer["generation_id"] != first_pointer["generation_id"], (
            "the pointer did not advance after a completing publish"
        )

        second_rows = len(read_sandbox_table(sandbox, "elo_game_snapshots"))
        assert second_rows > first_rows, (
            f"generation two wrote no new snapshot rows ({first_rows} -> "
            f"{second_rows}); 'the pointer moved' would then mean nothing."
        )

        staged_dir = silver / "elo_generations" / second_pointer["generation_id"]
        for name in (*ROW_TABLES, "elo_ratings_current"):
            assert (staged_dir / f"{name}.parquet").exists(), (
                f"{name} is missing from the published generation's staging directory"
            )
        assert (staged_dir / "elo_ratings.json").exists()


class TestAnIncompleteGenerationIsNeverPublished:
    """Four of five is not a generation, and the pointer must say so by not moving."""

    def test_a_missing_artifact_is_named_and_the_pointer_does_not_move(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import (
            EloGenerationIncompleteError,
            new_generation_id,
            publish_elo_generation,
            read_elo_generation_pointer,
        )

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = _seed_first_generation(sandbox)
        silver = sandbox / "silver"
        before_pointer = read_elo_generation_pointer(silver)

        update = builder.update_current_season(season=2025)
        staged = {
            "elo_game_snapshots": update.snapshots,
            "games_with_elo": update.games_with_elo,
            "elo_ratings_current": builder.elo_system.get_current_ratings(),
            "elo_ratings": builder.elo_system.ratings_state(),
        }  # elo_rating_history deliberately withheld

        with pytest.raises(EloGenerationIncompleteError) as excinfo:
            publish_elo_generation(
                staged,
                new_generation_id(),
                silver_root=silver,
                publish_live=lambda: None,
            )

        assert "elo_rating_history" in str(excinfo.value), (
            "EloGenerationIncompleteError must NAME the missing artifact; 'incomplete' "
            f"alone does not tell an operator what to do. Got: {excinfo.value}"
        )
        assert read_elo_generation_pointer(silver) == before_pointer, (
            "the pointer moved for a generation that was never complete"
        )
