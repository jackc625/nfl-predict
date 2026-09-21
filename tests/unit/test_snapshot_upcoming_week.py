"""A scheduled-but-unplayed game gets a snapshot row that SAYS it is provisional.

WHY THIS MODULE EXISTS (COLD-02, D33-07)
----------------------------------------
``EloBuilder._process_chain`` ``continue``s past any game whose ``home_score`` or
``away_score`` is null (scripts/build_elo.py), so an UNPLAYED game has no snapshot row
BY CONSTRUCTION. That is correct for a rebuild and useless for a live week: COLD-01's
value-by-value join against ``elo_game_snapshots`` has no left side to join to, and the
gold LEFT JOIN therefore imputes Elo for exactly the games the system is being asked to
predict.

The two rejected alternatives are on record. An UNFLAGGED provisional row is a default
indistinguishable from a real observation -- the same class of defect as the silent
``0.0`` league-average QB quality that Phase 28 replaced with an explicitly named
``REPLACEMENT_LEVEL_QB_QUALITY``. A builder-side fallback to live ratings leaves the
join with no left side at all, which is the state this plan starts from.

So the row exists AND it is labelled. ``is_provisional`` takes the snapshot table from
11 columns to 12, and every writer sets it EXPLICITLY -- there is no null third state,
because a two-valued flag with an "unknown" is the ambiguity the flag exists to remove.

WHAT EACH TEST BELOW IS DEFENDING
---------------------------------
* The row count is one per scheduled-unplayed game, and ZERO for a week that is fully
  graded -- with all twelve columns present even when empty, because a bare empty frame
  would make every downstream ``frame["is_provisional"]`` a KeyError rather than a
  no-op.
* The flag is True on the live path and False on the canonical path, asserted on BOTH
  sides. Asserting only the True side would pass against a writer that stamped every
  row provisional.
* Replace-in-place asserts the row count AND the flipped flag. An unchanged count alone
  would also hold if nothing had been written at all.
* Read-only-ness is asserted FROM OUTSIDE, over the whole ratings mapping, including the
  cold case where a team has no rating yet: ``EloRatingSystem.predict_game`` calls
  ``get_or_create_rating``, which MUTATES ``self.ratings`` for an unseen team, so
  "reads current ratings" is a claim that has to be proven rather than assumed from the
  verb.

NOTHING HERE WRITES ``data/`` OR ``artifacts/``. Every write lands under ``tmp_path``
through ``tests/fixtures/elo_sandbox``'s two seams (the ``data.storage`` module
singletons AND ``EloBuilder(data_root=...)``, which ``upsert_silver`` resolves from
instead of the singletons).
"""

from __future__ import annotations

import pytest

from tests.fixtures.elo_sandbox import (
    make_season_games,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
)
from tests.phase33_state import (
    SNAPSHOT_COLUMN_COUNT_AFTER,
    SNAPSHOT_COLUMN_COUNT_BEFORE,
)

LIVE_SEASON = 2026

# The sandbox season is two weeks of four games. Week 1 graded, week 2 not, is the
# shape of the live 2026 capture the phase is about (mid-week-1, two games already
# scored -- tests/phase33_state.CAPTURED_SCHEDULE_GRADED_GAMES).
GAMES_PER_WEEK = 4


def _live_sandbox(monkeypatch, tmp_path, *, graded_weeks: tuple[int, ...]):
    """A two-week sandbox season with *graded_weeks* carrying scores."""
    sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
    games = make_season_games(LIVE_SEASON, weeks=2, graded_weeks=graded_weeks)
    builder = sandbox_builder(sandbox, games)
    return sandbox, games, builder


def _rating_state(elo_system) -> dict[str, tuple[float, float, int, object]]:
    """The whole ratings mapping as comparable plain values.

    Compared as VALUES rather than by identity: ``snapshot_upcoming_week`` is allowed to
    hold a reference to the builder's Elo system, and identity equality would pass even
    if every rating in it had moved.
    """
    return {
        team: (
            float(rating.rating),
            float(rating.uncertainty),
            int(rating.games_played),
            rating.season,
        )
        for team, rating in elo_system.ratings.items()
    }


class TestOneFlaggedRowPerScheduledUnplayedGame:
    """The row count, and the empty-week shape that is not a bare empty frame."""

    def test_an_unplayed_week_yields_one_row_per_scheduled_game(
        self, tmp_path, monkeypatch
    ) -> None:
        _sandbox, games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )
        builder.update_current_season(season=LIVE_SEASON)

        frame = builder.snapshot_upcoming_week(LIVE_SEASON, 2)

        expected_ids = sorted(games[games["week"] == 2]["game_id"])
        assert sorted(frame["game_id"]) == expected_ids, (
            "snapshot_upcoming_week must emit exactly one row per scheduled-unplayed "
            f"game in the week. Expected {expected_ids}, got "
            f"{sorted(frame['game_id'])}."
        )
        assert len(frame) == GAMES_PER_WEEK

    def test_a_fully_graded_week_yields_an_empty_frame_carrying_all_twelve_columns(
        self, tmp_path, monkeypatch
    ) -> None:
        """Zero rows, twelve columns. A bare empty frame would be a KeyError factory."""
        from scripts.build_elo import SNAPSHOT_COLUMNS

        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1, 2)
        )
        builder.update_current_season(season=LIVE_SEASON)

        frame = builder.snapshot_upcoming_week(LIVE_SEASON, 1)

        assert len(frame) == 0, (
            "every week-1 game is graded, so there is no provisional row to emit"
        )
        assert tuple(frame.columns) == SNAPSHOT_COLUMNS, (
            "an empty provisional frame must still carry the full snapshot shape, or "
            "every downstream reference to is_provisional becomes a KeyError on the "
            "one path where it matters most -- the week with nothing left to predict."
        )

    def test_a_week_the_schedule_does_not_contain_yields_an_empty_shaped_frame(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import SNAPSHOT_COLUMNS

        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )
        builder.update_current_season(season=LIVE_SEASON)

        frame = builder.snapshot_upcoming_week(LIVE_SEASON, 18)

        assert len(frame) == 0
        assert tuple(frame.columns) == SNAPSHOT_COLUMNS


class TestTheFlagIsSetExplicitlyOnBothPaths:
    """True on the live path, False on the canonical one. Both directions asserted."""

    def test_every_provisional_row_is_flagged_true(self, tmp_path, monkeypatch) -> None:
        from scripts.build_elo import PROVISIONAL_COLUMN

        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )
        builder.update_current_season(season=LIVE_SEASON)

        frame = builder.snapshot_upcoming_week(LIVE_SEASON, 2)

        assert len(frame) > 0
        assert bool(frame[PROVISIONAL_COLUMN].all()), (
            "every row snapshot_upcoming_week emits is provisional by definition"
        )
        assert frame[PROVISIONAL_COLUMN].dtype == bool, (
            "the flag is two-valued; an object or float dtype is how a null third "
            f"state gets in. Got {frame[PROVISIONAL_COLUMN].dtype}."
        )

    def test_every_canonical_row_is_flagged_false_and_never_left_to_a_default(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import PROVISIONAL_COLUMN

        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )

        update = builder.update_current_season(season=LIVE_SEASON)

        assert len(update.snapshots) == GAMES_PER_WEEK
        assert PROVISIONAL_COLUMN in update.snapshots.columns, (
            "the canonical builder must SET the flag, not omit it and rely on a "
            "reader's default"
        )
        assert not update.snapshots[PROVISIONAL_COLUMN].any(), (
            "a snapshot taken from a completed game is a real observation"
        )

    def test_the_full_history_builder_flags_every_row_false(
        self, tmp_path, monkeypatch
    ) -> None:
        """The backfill semantic: ``--full-rebuild`` history is all real, explicitly."""
        from scripts.build_elo import PROVISIONAL_COLUMN

        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1, 2)
        )

        snapshots = builder.build_elo_with_snapshots(start_season=LIVE_SEASON)

        assert len(snapshots) == 2 * GAMES_PER_WEEK
        assert not snapshots[PROVISIONAL_COLUMN].any()


class TestTheSnapshotColumnShape:
    """Twelve columns, stated once, in one order."""

    def test_snapshot_columns_is_the_eleven_plus_the_flag_in_that_order(self) -> None:
        from scripts.build_elo import (
            ELO_SNAPSHOT_COLUMNS,
            PROVISIONAL_COLUMN,
            SNAPSHOT_COLUMNS,
        )

        assert len(ELO_SNAPSHOT_COLUMNS) == SNAPSHOT_COLUMN_COUNT_BEFORE
        assert len(SNAPSHOT_COLUMNS) == SNAPSHOT_COLUMN_COUNT_AFTER
        assert SNAPSHOT_COLUMNS[:-1] == ELO_SNAPSHOT_COLUMNS, (
            "the flag is APPENDED; reordering the existing eleven would move every "
            "column in a table three deployed models read through"
        )
        assert SNAPSHOT_COLUMNS[-1] == PROVISIONAL_COLUMN

    def test_both_writers_build_their_frames_from_the_one_constant(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import SNAPSHOT_COLUMNS

        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )
        update = builder.update_current_season(season=LIVE_SEASON)
        provisional = builder.snapshot_upcoming_week(LIVE_SEASON, 2)

        assert tuple(update.snapshots.columns) == SNAPSHOT_COLUMNS
        assert tuple(provisional.columns) == SNAPSHOT_COLUMNS


class TestTheRealResultReplacesTheProvisionalRowInPlace:
    """Two assertions, because an unchanged count alone proves only inaction."""

    def test_the_real_row_replaces_the_provisional_one_without_growing_the_table(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import PROVISIONAL_COLUMN

        sandbox, games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )

        # Week 1 is played; persist it through the live verb.
        update = builder.update_current_season(season=LIVE_SEASON)
        builder.save_live_append(
            LIVE_SEASON,
            snapshots=update.snapshots,
        )

        # Friday: week 2 is scheduled and unplayed. Persist the provisional rows.
        provisional = builder.snapshot_upcoming_week(LIVE_SEASON, 2)
        builder.save_live_append(
            LIVE_SEASON,
            snapshots=provisional,
        )

        friday = read_sandbox_table(sandbox, "elo_game_snapshots")
        week_2_ids = sorted(games[games["week"] == 2]["game_id"])
        assert len(friday) == 2 * GAMES_PER_WEEK, (
            f"four real rows plus four provisional rows is eight, got {len(friday)}"
        )
        assert bool(
            friday[friday["game_id"].isin(week_2_ids)][PROVISIONAL_COLUMN].all()
        )

        # Monday: the results land. Re-derive the season and append it.
        graded = games.copy()
        played = graded["week"] == 2
        graded.loc[played, "home_score"] = 27.0
        graded.loc[played, "away_score"] = 17.0
        monday_builder = sandbox_builder(sandbox, graded)
        monday = monday_builder.update_current_season(season=LIVE_SEASON)
        monday_builder.save_live_append(
            LIVE_SEASON,
            snapshots=monday.snapshots,
        )

        after = read_sandbox_table(sandbox, "elo_game_snapshots")
        assert len(after) == 2 * GAMES_PER_WEEK, (
            "the real result REPLACES the provisional row on game_id; the table must "
            f"not grow. Expected {2 * GAMES_PER_WEEK} rows, got {len(after)}."
        )
        replaced = after[after["game_id"].isin(week_2_ids)]
        assert len(replaced) == GAMES_PER_WEEK
        assert not replaced[PROVISIONAL_COLUMN].any(), (
            "an unchanged row count would also hold if nothing had been written. The "
            "flag flipping to False for every affected game_id is the half of the "
            "claim that proves the replacement actually happened."
        )


class TestTheProvisionalFrameIsRefusedAsATrainingInput:
    """The row serves a live week; handing it to training is refused BY NAME.

    Scoped here to the frame this module produces. The proof that the refusal is wired
    at every trainer's gold-loading boundary -- four separate entry points, none of which
    passes through the feature builder -- lives in
    ``tests/unit/test_provisional_training_refusal.py``.
    """

    def test_the_refusal_names_every_offending_game_id(
        self, tmp_path, monkeypatch
    ) -> None:
        from features.elo_features import (
            ProvisionalSnapshotAsTrainingInputError,
            assert_no_provisional_training_rows,
        )

        _sandbox, games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )
        builder.update_current_season(season=LIVE_SEASON)
        provisional = builder.snapshot_upcoming_week(LIVE_SEASON, 2)

        with pytest.raises(ProvisionalSnapshotAsTrainingInputError) as excinfo:
            assert_no_provisional_training_rows(provisional, "train:wp")

        message = str(excinfo.value)
        for game_id in sorted(games[games["week"] == 2]["game_id"]):
            assert game_id in message, (
                f"{game_id} is provisional and the refusal did not name it. An operator "
                "reading this message has to know WHICH games to wait on. Message: "
                f"{message}"
            )
        assert "train:wp" in message, "the refusal must name the context it fired in"

    def test_a_frame_of_real_rows_passes_through_untouched(
        self, tmp_path, monkeypatch
    ) -> None:
        """The negative control. Without it the guard could simply always raise."""
        from features.elo_features import assert_no_provisional_training_rows

        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )
        update = builder.update_current_season(season=LIVE_SEASON)

        assert len(update.snapshots) == GAMES_PER_WEEK
        # Must not raise: every row came from a completed game.
        assert_no_provisional_training_rows(update.snapshots, "train:wp")


class TestSnapshotUpcomingWeekMutatesNoRatingState:
    """Asserted from outside, over the whole mapping, including the cold case."""

    def test_the_ratings_mapping_is_identical_before_and_after(
        self, tmp_path, monkeypatch
    ) -> None:
        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=(1,)
        )
        builder.update_current_season(season=LIVE_SEASON)

        before = _rating_state(builder.elo_system)
        builder.snapshot_upcoming_week(LIVE_SEASON, 2)
        after = _rating_state(builder.elo_system)

        assert after == before, (
            "snapshot_upcoming_week READS current ratings. Any movement here would "
            "mean the Friday serving call had advanced the ratings the following "
            "week's real snapshots are then computed from."
        )

    def test_an_unrated_team_is_not_created_by_taking_a_provisional_snapshot(
        self, tmp_path, monkeypatch
    ) -> None:
        """``predict_game`` -> ``get_or_create_rating`` MUTATES for an unseen team.

        So the read-only claim is tested where it can actually break: a builder whose
        Elo state is empty. If the mapping gains eight teams here, then on the live
        path a Friday serving call silently invents ratings that the next canonical
        re-derivation would not have produced.
        """
        _sandbox, _games, builder = _live_sandbox(
            monkeypatch, tmp_path, graded_weeks=()
        )

        assert builder.elo_system.ratings == {}, "the fixture starts from no ratings"

        frame = builder.snapshot_upcoming_week(LIVE_SEASON, 1)

        assert len(frame) == GAMES_PER_WEEK, (
            "a cold start still owes the week a provisional row per game"
        )
        assert builder.elo_system.ratings == {}, (
            "taking a provisional snapshot must not create a rating. Got "
            f"{sorted(builder.elo_system.ratings)}."
        )
