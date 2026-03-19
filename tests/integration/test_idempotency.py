"""Integration tests for ingestion pipeline idempotency.

These tests verify:
- Bronze snapshots are append-only (each call creates a new file, old files untouched)
- Silver upsert is idempotent (re-running with same data produces identical output)
- Silver upsert correctly adds new rows and replaces updated rows
- get_latest_bronze_file returns the most recent snapshot

All tests use tmp_path for complete isolation -- no real data files required.
"""

import time

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
