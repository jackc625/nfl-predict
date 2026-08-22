"""Integration tests for ingestion pipeline idempotency.

These tests verify:
- Bronze snapshots are append-only (each call creates a new file, old files untouched)
- Silver upsert is idempotent (re-running with same data produces identical output)
- Silver upsert correctly adds new rows and replaces updated rows
- get_latest_bronze_file returns the most recent snapshot
- SPEC R1 (Plan 30-08): a full GOLD rebuild on unchanged inputs reproduces gold

The Bronze and Silver classes use tmp_path for complete isolation -- no real data files
required. ``TestAFullGoldRebuildIsReproducible`` is the one exception: it judges the two
fingerprint documents a pair of full-history rebuilds produced, and skips with a
remediation command when those gitignored run records are absent. Its own docstring says
why the comparison names the per-build clock instead of hiding it.
"""

import json
import time
from pathlib import Path

import pandas as pd
import pytest

from data.storage import get_latest_bronze_file, save_bronze_snapshot, upsert_silver

# -- Helpers -------------------------------------------------------------------


def _make_games_df(game_ids: list[str], home_score: int = 21) -> pd.DataFrame:
    """Create a minimal games DataFrame suitable for storage tests."""
    records = []
    for gid in game_ids:
        records.append(
            {
                "game_id": gid,
                "season": 2024,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "home_score": home_score,
                "away_score": 17,
            }
        )
    return pd.DataFrame(records)


# -- Tests ---------------------------------------------------------------------


@pytest.mark.integration
class TestBronzeAppendOnly:
    """Verify Bronze layer append-only semantics."""

    def test_save_bronze_twice_creates_two_files(self, tmp_path):
        """Calling save_bronze_snapshot twice produces two distinct files."""
        df = _make_games_df(["2024_W01_BUF@KC"])

        path1 = save_bronze_snapshot(df, "games", 2024, 1, base_path=tmp_path)
        # Brief pause so timestamp differs (filename includes seconds)
        time.sleep(1.1)
        path2 = save_bronze_snapshot(df, "games", 2024, 1, base_path=tmp_path)

        assert path1.exists(), f"First Bronze file should exist: {path1}"
        assert path2.exists(), f"Second Bronze file should exist: {path2}"
        assert path1 != path2, "Two snapshots should produce distinct file paths"

        # Both files should have the same content
        df1 = pd.read_parquet(path1)
        df2 = pd.read_parquet(path2)
        pd.testing.assert_frame_equal(df1, df2)

    def test_bronze_files_never_modified_after_creation(self, tmp_path):
        """Existing Bronze files are never overwritten or modified."""
        df1 = _make_games_df(["2024_W01_BUF@KC"], home_score=21)

        path1 = save_bronze_snapshot(df1, "games", 2024, 1, base_path=tmp_path)
        mtime1 = path1.stat().st_mtime

        # Wait and create a second snapshot with different data
        time.sleep(1.1)
        df2 = _make_games_df(["2024_W01_BUF@KC"], home_score=35)
        path2 = save_bronze_snapshot(df2, "games", 2024, 1, base_path=tmp_path)

        # First file should not have been modified
        assert path1.stat().st_mtime == mtime1, (
            "First Bronze file modification time should be unchanged"
        )
        assert path1 != path2, "Second call should create a different file"

        # Verify two distinct files exist
        bronze_dir = tmp_path / "bronze"
        all_files = list(bronze_dir.glob("games_raw_bronze_2024_W01_*.parquet"))
        assert len(all_files) == 2, f"Expected 2 Bronze files, found {len(all_files)}"

    def test_get_latest_bronze_file_returns_newest(self, tmp_path):
        """get_latest_bronze_file returns the most recently created file."""
        df = _make_games_df(["2024_W01_BUF@KC"])

        save_bronze_snapshot(df, "games", 2024, 1, base_path=tmp_path)
        time.sleep(1.1)
        path2 = save_bronze_snapshot(df, "games", 2024, 1, base_path=tmp_path)

        latest = get_latest_bronze_file("games", 2024, 1, base_path=tmp_path)
        assert latest is not None, "Should find at least one Bronze file"
        assert latest == path2, (
            f"Expected latest file to be {path2.name}, got {latest.name}"
        )


@pytest.mark.integration
class TestSilverUpsert:
    """Verify Silver layer idempotent upsert semantics."""

    def test_upsert_silver_idempotent(self, tmp_path):
        """Upserting the same data twice produces the same number of rows."""
        df = _make_games_df(
            ["g1", "g2", "g3", "g4", "g5"],
            home_score=21,
        )

        upsert_silver(df, "games", base_path=tmp_path)
        upsert_silver(df, "games", base_path=tmp_path)

        silver_path = tmp_path / "silver" / "games.parquet"
        result = pd.read_parquet(silver_path)
        assert len(result) == 5, (
            f"Expected 5 games after idempotent upsert, got {len(result)}"
        )

    def test_upsert_silver_adds_new_games(self, tmp_path):
        """Upserting new game_ids appends them to existing data."""
        df_a = _make_games_df(["g1", "g2", "g3"])
        upsert_silver(df_a, "games", base_path=tmp_path)

        df_b = _make_games_df(["g4", "g5"])
        upsert_silver(df_b, "games", base_path=tmp_path)

        silver_path = tmp_path / "silver" / "games.parquet"
        result = pd.read_parquet(silver_path)
        assert len(result) == 5, (
            f"Expected 5 games after adding new games, got {len(result)}"
        )
        result_ids = set(result["game_id"])
        assert result_ids == {"g1", "g2", "g3", "g4", "g5"}, (
            f"Expected all 5 game_ids, got {result_ids}"
        )

    def test_upsert_silver_replaces_updated_games(self, tmp_path):
        """Upserting an existing game_id with new data replaces the old row."""
        df_a = _make_games_df(["g1"], home_score=21)
        upsert_silver(df_a, "games", base_path=tmp_path)

        df_b = _make_games_df(["g1"], home_score=28)
        upsert_silver(df_b, "games", base_path=tmp_path)

        silver_path = tmp_path / "silver" / "games.parquet"
        result = pd.read_parquet(silver_path)
        assert len(result) == 1, f"Expected 1 game after replace, got {len(result)}"
        assert result.iloc[0]["home_score"] == 28, (
            f"Expected updated home_score=28, got {result.iloc[0]['home_score']}"
        )


@pytest.mark.integration
class TestAFullGoldRebuildIsReproducible:
    """SPEC R1: re-running the rebuild on unchanged inputs reproduces gold exactly.

    THIS IS ONLY POSSIBLE BECAUSE OF THE ``replace_mode=True`` GOLD WRITE landed at rung
    3 (Plan 30-07). Under the previous append path ``save_dataframe`` merged the incoming
    frame with the existing table through ``pd.concat``, so a rebuild could neither
    narrow a schema nor be relied on to reproduce one: dropped columns returned as
    all-NaN and an ``int32`` column was silently upcast to ``float64`` by the
    concatenation. Replace mode writes the passed frame AS the table, which is what makes
    a byte comparison across two runs mean anything at all. The dependency is stated here
    so it is visible rather than inferred.

    THE ONE COLUMN THAT MUST DIFFER, AND WHY THE COMPARISON NAMES IT RATHER THAN HIDING
    IT. ``scripts/build_features.py`` stamps ``feature_timestamp = datetime.now(UTC)``
    once per build, so it takes exactly one distinct value per build and MUST move on
    every rebuild by construction. Plan 30-18 established it as its own ``build_clock``
    category in the rung judge for precisely this reason (D30-OWNER-08): counting a clock
    as a moved VALUE made rung 3's empty-changed-set criterion structurally
    unsatisfiable, and a criterion that cannot be satisfied has stopped discriminating
    between a right rebuild and a wrong one.

    The same argument applies to a literal byte comparison of two fingerprint documents,
    so this asserts the stronger and actually meaningful claim in two halves: every
    column EXCEPT the named build clock reproduces its per-season digest EXACTLY in every
    season of all three matrices, AND the set of entries that do differ is asserted to be
    EXACTLY the clock's -- so nothing else can hide behind the exclusion.

    The rebuilds themselves are operator actions: two full-history
    ``python -m scripts.build_features`` runs on unchanged inputs, with their timestamps,
    durations and gold digests recorded in the Plan 30-08 SUMMARY. This test is the
    committed judge of the two documents they produced, in the same shape as every other
    fingerprint assertion in Phase 30.
    """

    _REPO_ROOT = Path(__file__).resolve().parents[2]
    _RUNG4 = _REPO_ROOT / "outputs" / "fingerprints" / "rung4.json"
    _RERUN = _REPO_ROOT / "outputs" / "fingerprints" / "rung4_rerun.json"
    _BUILD_CLOCK = "feature_timestamp"

    def _documents(self) -> tuple[dict, dict]:
        missing = [
            str(path) for path in (self._RUNG4, self._RERUN) if not path.exists()
        ]
        if missing:
            pytest.skip(
                f"fingerprint document(s) absent: {', '.join(missing)}. These are "
                "gitignored run records (.gitignore:26). Reproduce with two consecutive "
                "`python -m scripts.build_features` runs, fingerprinting after each via "
                "`python -m scripts.fingerprint_gold --out outputs/fingerprints/<name>.json`."
            )
        return (
            json.loads(self._RUNG4.read_text(encoding="utf-8")),
            json.loads(self._RERUN.read_text(encoding="utf-8")),
        )

    def test_the_rerun_reproduces_every_non_clock_column_exactly(self):
        first, second = self._documents()

        assert set(first) == set(second)
        for matrix in sorted(first):
            before, after = first[matrix], second[matrix]
            assert after["rows"] == before["rows"], (
                f"{matrix}: a re-run on unchanged inputs changed the row count "
                f"{before['rows']} -> {after['rows']}"
            )
            assert after["width"] == before["width"], (
                f"{matrix}: a re-run on unchanged inputs changed the width "
                f"{before['width']} -> {after['width']}"
            )
            assert after["rows_per_season"] == before["rows_per_season"]

            moved = [
                column
                for column in sorted(before["columns"])
                if column != self._BUILD_CLOCK
                and before["columns"][column] != after["columns"].get(column)
            ]
            assert not moved, (
                f"{matrix}: {len(moved)} column(s) did not reproduce across two builds "
                f"on unchanged inputs -- {moved[:10]}. SPEC R1 requires a rebuild to be "
                "reproducible; a column that moves without an input moving means the "
                "build carries hidden state or a non-deterministic fit."
            )

    def test_the_only_entries_that_differ_are_the_build_clocks(self):
        """Nothing may hide behind the clock exclusion, so the diff set is pinned."""
        first, second = self._documents()

        differing = {
            (matrix, column)
            for matrix in first
            for column in first[matrix]["columns"]
            if first[matrix]["columns"][column] != second[matrix]["columns"].get(column)
        }
        assert differing == {(matrix, self._BUILD_CLOCK) for matrix in first}, (
            "the set of entries differing between two builds on unchanged inputs is "
            f"{sorted(differing)}. It must be EXACTLY the per-build clock in each "
            "matrix and nothing else."
        )

    def test_the_build_clock_did_move_so_the_comparison_is_not_vacuous(self):
        """If the clock did NOT move, the two documents describe the SAME build."""
        first, second = self._documents()

        for matrix in sorted(first):
            assert (
                first[matrix]["columns"][self._BUILD_CLOCK]
                != second[matrix]["columns"][self._BUILD_CLOCK]
            ), (
                f"{matrix}'s build clock is identical across the two documents, which "
                "means they were fingerprinted from the SAME build. A re-run comparison "
                "against itself proves nothing -- re-run the build."
            )
