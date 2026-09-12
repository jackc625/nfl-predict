"""A weekly append provably leaves every prior season's rows alone (R3, T-33-14).

WHAT "BYTE-IDENTICAL" MEANS HERE, AND WHY IT CANNOT MEAN FILE BYTES
-------------------------------------------------------------------
R3 asks that a live append leave every pre-existing season's snapshot rows identical.
Read as a file sha256 that criterion is UNSATISFIABLE BY ANY WRITER: both stores this
repository uses rewrite the whole artifact on a logical no-op. ``upsert_silver``
concatenates and re-serialises the entire parquet through
``_atomic_write_parquet``; ``DuckDBConnection.create_table_from_df`` drops and recreates
the table. Neither preserves file bytes, and neither has anything to do with whether a
prior season's ROWS changed.

The repository has already MEASURED this rather than argued it:
``tests/integration/test_n01_resync_control.py``'s idempotency arm exists because a
second apply of the same data moves the store's bytes while changing no value.

So D33-08's reading is PER-SEASON ROW DIGESTS -- the digest shape
``scripts/fingerprint_gold._per_season_digests`` already uses (sort by ``game_id``,
group by ``season``, sha256, truncate), widened from one column to the whole row,
because R3's criterion is about rows. ``tests/fixtures/elo_sandbox`` owns the one
implementation so this module and the rerun-identity suite cannot digest differently.

ORDER-INVARIANCE IS ASSERTED, NOT ASSUMED. An upsert concatenates the surviving rows
with the new ones, so row order after an append is not the order before it. A digest
that depended on row order would report every append as a change and the test would
have to be weakened until it passed.
"""

from __future__ import annotations

import pandas as pd

from tests.fixtures.elo_sandbox import (
    make_season_games,
    per_season_row_digests,
    read_sandbox_table,
    redirect_storage_to_sandbox,
    sandbox_builder,
    seed_sandbox_games,
)

PRIOR_SEASONS = (2023, 2024)
APPENDED_SEASON = 2025


def _seed_two_prior_seasons(sandbox):
    """Publish 2023 and 2024 through the real live verb, one season at a time."""
    games = pd.concat(
        [
            make_season_games(year, weeks=2)
            for year in (*PRIOR_SEASONS, APPENDED_SEASON)
        ],
        ignore_index=True,
    )
    builder = sandbox_builder(sandbox, games)
    for season in PRIOR_SEASONS:
        update = builder.build_season_frames(season)
        builder.save_live_append(
            season,
            snapshots=update.snapshots,
            games_with_elo=update.games_with_elo,
            rating_history=update.rating_history,
        )
    return builder


def _append_third_season(builder):
    update = builder.build_season_frames(APPENDED_SEASON)
    builder.save_live_append(
        APPENDED_SEASON,
        snapshots=update.snapshots,
        games_with_elo=update.games_with_elo,
        rating_history=update.rating_history,
    )


class TestAPriorSeasonIsUntouchedByALiveAppend:
    """The isolation property, its anti-vacuity control, and its failure control."""

    def test_the_seeded_digests_are_non_empty(self, tmp_path, monkeypatch) -> None:
        """Anti-vacuity: two empty mappings compare equal and prove nothing."""
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        _seed_two_prior_seasons(sandbox)

        digests = per_season_row_digests(
            read_sandbox_table(sandbox, "elo_game_snapshots")
        )
        assert set(digests) == {str(year) for year in PRIOR_SEASONS}, (
            f"the seed must produce a digest for each prior season, got {digests}"
        )

    def test_every_prior_season_digest_survives_the_append(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = _seed_two_prior_seasons(sandbox)

        before = per_season_row_digests(
            read_sandbox_table(sandbox, "elo_game_snapshots")
        )
        _append_third_season(builder)
        after = per_season_row_digests(
            read_sandbox_table(sandbox, "elo_game_snapshots")
        )

        for season in PRIOR_SEASONS:
            key = str(season)
            assert after[key] == before[key], (
                f"season {season}'s snapshot rows MOVED during a {APPENDED_SEASON} "
                f"append ({before[key]} -> {after[key]}). A live append may only ever "
                "touch the season it names."
            )
        assert str(APPENDED_SEASON) in after, (
            "the appended season produced no rows, so 'the others are unchanged' is "
            "satisfied by a writer that did nothing."
        )

    def test_the_digest_is_invariant_to_row_and_column_order(
        self, tmp_path, monkeypatch
    ) -> None:
        """An upsert reorders rows; a digest that cared would report a false change."""
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        _seed_two_prior_seasons(sandbox)

        frame = read_sandbox_table(sandbox, "elo_game_snapshots")
        shuffled = frame.sample(frac=1.0, random_state=17).reset_index(drop=True)
        shuffled = shuffled[list(reversed(list(shuffled.columns)))]

        assert per_season_row_digests(shuffled) == per_season_row_digests(frame), (
            "the per-season row digest changed under a pure reordering, so every "
            "append would report as a change and this test could only pass by being "
            "weakened."
        )

    def test_a_planted_prior_season_rewrite_is_detected(
        self, tmp_path, monkeypatch
    ) -> None:
        """Fail-closed control: the comparison must be able to REPORT a change."""
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        _seed_two_prior_seasons(sandbox)

        frame = read_sandbox_table(sandbox, "elo_game_snapshots")
        before = per_season_row_digests(frame)

        tampered = frame.copy()
        first_2023 = tampered.index[tampered["season"] == PRIOR_SEASONS[0]][0]
        tampered.loc[first_2023, "home_elo_pre"] = 1234.5
        after = per_season_row_digests(tampered)

        assert after[str(PRIOR_SEASONS[0])] != before[str(PRIOR_SEASONS[0])], (
            "a changed prior-season value did not move its digest, so the isolation "
            "assertion above cannot fail and asserts nothing."
        )
        assert after[str(PRIOR_SEASONS[1])] == before[str(PRIOR_SEASONS[1])], (
            "the untouched season's digest moved too, so the digest is not per-season "
            "and the assertion cannot attribute a change to a season."
        )

    def test_the_other_two_row_tables_are_isolated_as_well(
        self, tmp_path, monkeypatch
    ) -> None:
        """The property is about ROW TABLES, not about the snapshot table alone."""
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = _seed_two_prior_seasons(sandbox)

        before = {
            table: per_season_row_digests(read_sandbox_table(sandbox, table))
            for table in ("games_with_elo", "elo_rating_history")
        }
        _append_third_season(builder)
        after = {
            table: per_season_row_digests(read_sandbox_table(sandbox, table))
            for table in ("games_with_elo", "elo_rating_history")
        }

        for table, prior_digests in before.items():
            assert prior_digests, f"{table} was empty before the append"
            for season in PRIOR_SEASONS:
                key = str(season)
                assert after[table][key] == prior_digests[key], (
                    f"{table} season {season} moved during a {APPENDED_SEASON} append"
                )


class TestASecondAppendOfTheSameSeasonIsALogicalNoOp:
    """Re-running the week must not grow, duplicate or move any season."""

    def test_a_repeat_append_leaves_every_season_digest_unchanged(
        self, tmp_path, monkeypatch
    ) -> None:
        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        builder = _seed_two_prior_seasons(sandbox)
        _append_third_season(builder)

        before = per_season_row_digests(
            read_sandbox_table(sandbox, "elo_game_snapshots")
        )
        # Re-derive the SAME season from the SAME state and append it again.
        builder.elo_system = builder.elo_system.__class__()
        seed_sandbox_games(
            pd.concat(
                [
                    make_season_games(year, weeks=2)
                    for year in (*PRIOR_SEASONS, APPENDED_SEASON)
                ],
                ignore_index=True,
            )
        )
        for season in (*PRIOR_SEASONS, APPENDED_SEASON):
            builder.build_season_frames(season)
        _append_third_season(builder)

        after = per_season_row_digests(
            read_sandbox_table(sandbox, "elo_game_snapshots")
        )
        assert set(after) == set(before)
        for season in PRIOR_SEASONS:
            assert after[str(season)] == before[str(season)], (
                f"a repeat {APPENDED_SEASON} append moved season {season}"
            )
