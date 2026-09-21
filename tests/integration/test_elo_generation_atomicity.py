"""An Elo generation is published atomically, or not at all (T-33-16d).

THE FAILURE THIS MODULE EXISTS FOR
----------------------------------
``save_results`` wrote five artifacts with five separate calls. Each individual write
was atomic; the SET was not. A crash between any two of them left a MIXED generation --
snapshots from the new build sitting beside a rating history from the old one -- and
nothing on disk said so. Every consumer downstream would have read that mixture as a
single coherent Elo state, because from the outside it is indistinguishable from one.

``publish_elo_generation`` closes that by STAGING every artifact under one generation
id, VALIDATING them together, publishing, and only then moving a single generation
pointer in one ``os.replace``. A crash before the pointer moves leaves the previous
generation intact and named; a crash during staging leaves the live artifacts
untouched, because nothing live was written at all.

DELIBERATE RE-PIN (Plan 33.2-05, D33.2-22, owner ratified 2026-09-21)
---------------------------------------------------------------------
Until Plan 33.2-05 this module pinned FIVE-artifact atomicity. Four of those artifacts
-- ``games_with_elo``, ``elo_rating_history``, ``elo_ratings_current`` and the JSON Elo
state -- were side stores of a legacy per-season pass that learned no home-field
advantage, and D33.2-22 deleted the pass and the stores together. WHY this module
changed: the SET the guarantee ranges over narrowed to ``elo_game_snapshots``. WHAT did
not change: the GUARANTEE -- stage, validate, publish, one pointer move -- is asserted
exactly as before, over the set that exists. This is a re-pin, not a dropped test: the
interruption is still planted at EVERY staged write (today there is one), the set is
read from the module that defines it, and a test below refuses a generation that tries
to smuggle a deleted name back in.
"""

from __future__ import annotations

import pytest

from scripts.elo_generation import ELO_ROW_TABLES, ELO_STATE_ARTIFACTS
from tests.fixtures.elo_sandbox import (
    make_season_games,
    per_season_row_digests,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
)

# The published set, read from the DEFINING module rather than re-declared here, so this
# module cannot pin a set the publisher no longer uses.
PUBLISHED_SET: tuple[str, ...] = (*ELO_ROW_TABLES, *ELO_STATE_ARTIFACTS)

# The four names D33.2-22 deleted. A generation must never stage any of them again.
DELETED_BY_RULING: tuple[str, ...] = (
    "games_with_elo",
    "elo_rating_history",
    "elo_ratings_current",
    "elo_ratings",
)


def _live_digests(sandbox) -> dict[str, dict[str, str]]:
    """Per-season row digests of every live row table, as one comparable mapping."""
    return {
        table: per_season_row_digests(read_sandbox_table(sandbox, table))
        for table in ELO_ROW_TABLES
    }


def _seed_first_generation(sandbox, season: int = 2025, weeks: int = 2):
    """Publish generation ONE through the real live-append verb."""
    builder = sandbox_builder(sandbox, make_season_games(season, weeks=weeks))
    update = builder.build_season_frames(season)
    builder.save_live_append(season, snapshots=update.snapshots)
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


class TestThePublishedSetIsExactlyTheSnapshotTable:
    """The re-pin itself, stated where a reader of this module will see it first."""

    def test_the_set_is_one_row_table_and_no_state_artifact(self) -> None:
        assert PUBLISHED_SET == ("elo_game_snapshots",), (
            f"the Elo generation now ranges over {PUBLISHED_SET}. D33.2-22 narrowed it "
            "to the snapshot table; widening it again is a decision, not a drift."
        )

    def test_no_deleted_name_is_in_the_set(self) -> None:
        assert not set(PUBLISHED_SET) & set(DELETED_BY_RULING)


class TestAnInterruptedPublishLeavesThePreviousGenerationIntact:
    """Every interruption point, each asserted on live content and on the pointer."""

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

    @pytest.mark.parametrize("fail_on_call", range(1, len(PUBLISHED_SET) + 1))
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

        # A SECOND generation's worth of content.
        wider = builder.build_season_frames(2025)
        staged = {"elo_game_snapshots": wider.snapshots}

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
        builder.elo_system = builder.elo_system.__class__()
        update = builder.build_season_frames(2025)
        builder.save_live_append(2025, snapshots=update.snapshots)

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
        staged_files = sorted(path.name for path in staged_dir.iterdir())
        assert staged_files == [f"{name}.parquet" for name in PUBLISHED_SET], (
            f"the published generation staged {staged_files}; it must stage exactly "
            "the published set and nothing a deleted writer used to put there."
        )
        assert set(second_pointer["artifacts"]) == set(PUBLISHED_SET), (
            f"the pointer names {sorted(second_pointer['artifacts'])}, not the "
            "published set."
        )

    def test_a_deleted_name_offered_to_the_publisher_is_not_staged(
        self, tmp_path, monkeypatch
    ) -> None:
        """The publisher stages the SET, not whatever a caller hands it.

        A stale caller that still builds a ``games_with_elo`` frame must not get it
        into a generation directory, where a reader would find a table D33.2-22 says
        does not exist.
        """
        from scripts.build_elo import (
            new_generation_id,
            publish_elo_generation,
            read_elo_generation_pointer,
        )

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = _seed_first_generation(sandbox)
        silver = sandbox / "silver"
        update = builder.build_season_frames(2025)

        publish_elo_generation(
            {
                "elo_game_snapshots": update.snapshots,
                "games_with_elo": update.snapshots.copy(),
            },
            new_generation_id(),
            silver_root=silver,
            publish_live=lambda: None,
        )

        pointer = read_elo_generation_pointer(silver)
        assert pointer is not None
        staged_dir = silver / "elo_generations" / pointer["generation_id"]
        assert not (staged_dir / "games_with_elo.parquet").exists()
        assert "games_with_elo" not in pointer["artifacts"]


class TestAnIncompleteGenerationIsNeverPublished:
    """A generation missing a member is not a generation, and the pointer says so."""

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
        _seed_first_generation(sandbox)
        silver = sandbox / "silver"
        before_pointer = read_elo_generation_pointer(silver)

        staged: dict[str, object] = {}  # elo_game_snapshots deliberately withheld

        with pytest.raises(EloGenerationIncompleteError) as excinfo:
            publish_elo_generation(
                staged,
                new_generation_id(),
                silver_root=silver,
                publish_live=lambda: None,
            )

        assert "elo_game_snapshots" in str(excinfo.value), (
            "EloGenerationIncompleteError must NAME the missing artifact; 'incomplete' "
            f"alone does not tell an operator what to do. Got: {excinfo.value}"
        )
        assert read_elo_generation_pointer(silver) == before_pointer, (
            "the pointer moved for a generation that was never complete"
        )
