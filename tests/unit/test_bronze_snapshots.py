"""Unit tests for Bronze snapshot append-only behavior and Silver upsert.

Tests validate that:
- save_bronze_snapshot creates timestamped files that never overwrite
- upsert_silver replaces matching game_ids and preserves non-matching rows
- get_latest_bronze_file returns the most recent file for a table/season/week
"""

import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


def _make_games_df(
    game_ids: list[str], season: int = 2024, week: int = 1
) -> pd.DataFrame:
    """Create a minimal games DataFrame for testing."""
    records = []
    for gid in game_ids:
        records.append(
            {
                "game_id": gid,
                "season": season,
                "week": week,
                "home_team": "KC",
                "away_team": "BUF",
                "home_score": 27,
                "away_score": 24,
            }
        )
    return pd.DataFrame(records)


class TestSaveBronzeSnapshot:
    """Tests for save_bronze_snapshot()."""

    def test_creates_file_with_expected_naming_pattern(self, tmp_path: Path):
        """Bronze filename follows {table}_raw_bronze_{season}_W{week:02d}_{timestamp}.parquet."""
        from data.storage import save_bronze_snapshot

        df = _make_games_df(["2024_W01_BUF@KC"])
        result = save_bronze_snapshot(
            df, "games", season=2024, week=1, base_path=tmp_path
        )

        assert result.exists()
        assert result.name.startswith("games_raw_bronze_2024_W01_")
        assert result.name.endswith(".parquet")
        assert result.parent == tmp_path / "bronze"

    def test_two_calls_create_two_separate_files(self, tmp_path: Path):
        """Calling save_bronze_snapshot twice creates two distinct files (append-only)."""
        from data.storage import save_bronze_snapshot

        df = _make_games_df(["2024_W01_BUF@KC"])

        path1 = save_bronze_snapshot(
            df, "games", season=2024, week=1, base_path=tmp_path
        )
        # Small sleep to guarantee different timestamp
        time.sleep(1.1)
        path2 = save_bronze_snapshot(
            df, "games", season=2024, week=1, base_path=tmp_path
        )

        assert path1 != path2
        assert path1.exists()
        assert path2.exists()

        bronze_files = list(
            (tmp_path / "bronze").glob("games_raw_bronze_2024_W01_*.parquet")
        )
        assert len(bronze_files) == 2

    def test_file_is_readable_as_valid_parquet(self, tmp_path: Path):
        """Bronze snapshot file is a valid Parquet file that can be read back."""
        from data.storage import save_bronze_snapshot

        df = _make_games_df(["2024_W01_BUF@KC", "2024_W01_MIA@NE"])
        path = save_bronze_snapshot(
            df, "games", season=2024, week=1, base_path=tmp_path
        )

        # Read back with pyarrow and verify content
        table = pq.read_table(path)
        loaded_df = table.to_pandas()
        assert len(loaded_df) == 2
        assert "game_id" in loaded_df.columns

    def test_week_zero_padded_in_filename(self, tmp_path: Path):
        """Week number is zero-padded to 2 digits in filename."""
        from data.storage import save_bronze_snapshot

        df = _make_games_df(["2024_W01_BUF@KC"])
        path = save_bronze_snapshot(
            df, "games", season=2024, week=1, base_path=tmp_path
        )

        assert "_W01_" in path.name


class TestUpsertSilver:
    """Tests for upsert_silver()."""

    def test_creates_new_silver_file_when_none_exists(self, tmp_path: Path):
        """upsert_silver creates a new Silver file if none exists."""
        from data.storage import upsert_silver

        df = _make_games_df(["2024_W01_BUF@KC", "2024_W01_MIA@NE"])
        path = upsert_silver(df, "games", base_path=tmp_path)

        assert path.exists()
        assert path == tmp_path / "silver" / "games.parquet"
        loaded = pd.read_parquet(path)
        assert len(loaded) == 2

    def test_replaces_matching_game_ids_keeps_non_matching(self, tmp_path: Path):
        """upsert_silver replaces rows with matching game_id and keeps non-matching rows."""
        from data.storage import upsert_silver

        # First upsert: two games
        df1 = _make_games_df(["2024_W01_BUF@KC", "2024_W01_MIA@NE"])
        upsert_silver(df1, "games", base_path=tmp_path)

        # Second upsert: update BUF@KC, add new game
        df2 = pd.DataFrame(
            [
                {
                    "game_id": "2024_W01_BUF@KC",
                    "season": 2024,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_score": 30,
                    "away_score": 21,
                },
                {
                    "game_id": "2024_W01_DAL@PHI",
                    "season": 2024,
                    "week": 1,
                    "home_team": "PHI",
                    "away_team": "DAL",
                    "home_score": 17,
                    "away_score": 14,
                },
            ]
        )
        upsert_silver(df2, "games", base_path=tmp_path)

        # Load and verify
        loaded = pd.read_parquet(tmp_path / "silver" / "games.parquet")
        assert len(loaded) == 3  # MIA@NE kept, BUF@KC replaced, DAL@PHI added

        # Verify BUF@KC was updated (score changed)
        buf_kc = loaded[loaded["game_id"] == "2024_W01_BUF@KC"]
        assert len(buf_kc) == 1
        assert buf_kc.iloc[0]["home_score"] == 30

        # Verify MIA@NE was preserved
        mia_ne = loaded[loaded["game_id"] == "2024_W01_MIA@NE"]
        assert len(mia_ne) == 1

    def test_idempotent_same_data_produces_same_silver(self, tmp_path: Path):
        """upsert_silver called twice with same data produces identical Silver output."""
        from data.storage import upsert_silver

        df = _make_games_df(["2024_W01_BUF@KC", "2024_W01_MIA@NE"])

        upsert_silver(df, "games", base_path=tmp_path)
        loaded1 = pd.read_parquet(tmp_path / "silver" / "games.parquet")

        upsert_silver(df, "games", base_path=tmp_path)
        loaded2 = pd.read_parquet(tmp_path / "silver" / "games.parquet")

        assert len(loaded1) == len(loaded2)
        assert set(loaded1["game_id"]) == set(loaded2["game_id"])

    def test_custom_key_column(self, tmp_path: Path):
        """upsert_silver works with a custom key column."""
        from data.storage import upsert_silver

        df1 = pd.DataFrame(
            [
                {"game_id": "g1", "snapshot_id": "s1", "value": 10},
                {"game_id": "g2", "snapshot_id": "s2", "value": 20},
            ]
        )
        upsert_silver(df1, "test_table", key_column="snapshot_id", base_path=tmp_path)

        df2 = pd.DataFrame(
            [
                {"game_id": "g1", "snapshot_id": "s1", "value": 99},
            ]
        )
        upsert_silver(df2, "test_table", key_column="snapshot_id", base_path=tmp_path)

        loaded = pd.read_parquet(tmp_path / "silver" / "test_table.parquet")
        assert len(loaded) == 2
        s1_row = loaded[loaded["snapshot_id"] == "s1"]
        assert s1_row.iloc[0]["value"] == 99


class TestGetLatestBronzeFile:
    """Tests for get_latest_bronze_file()."""

    def test_returns_most_recent_file(self, tmp_path: Path):
        """get_latest_bronze_file returns the latest timestamped file."""
        from data.storage import get_latest_bronze_file, save_bronze_snapshot

        df = _make_games_df(["2024_W01_BUF@KC"])

        save_bronze_snapshot(df, "games", season=2024, week=1, base_path=tmp_path)
        time.sleep(1.1)
        path2 = save_bronze_snapshot(
            df, "games", season=2024, week=1, base_path=tmp_path
        )

        latest = get_latest_bronze_file(
            "games", season=2024, week=1, base_path=tmp_path
        )
        assert latest == path2

    def test_returns_none_when_no_files_exist(self, tmp_path: Path):
        """get_latest_bronze_file returns None when no matching files exist."""
        from data.storage import get_latest_bronze_file

        # Create bronze dir but no files
        (tmp_path / "bronze").mkdir(parents=True, exist_ok=True)
        result = get_latest_bronze_file(
            "games", season=2024, week=1, base_path=tmp_path
        )
        assert result is None

    def test_filters_by_table_and_week(self, tmp_path: Path):
        """get_latest_bronze_file only returns files matching the requested table/season/week."""
        from data.storage import get_latest_bronze_file, save_bronze_snapshot

        df = _make_games_df(["2024_W01_BUF@KC"])

        # Save for week 1 and week 2
        save_bronze_snapshot(df, "games", season=2024, week=1, base_path=tmp_path)
        save_bronze_snapshot(df, "games", season=2024, week=2, base_path=tmp_path)
        save_bronze_snapshot(df, "odds", season=2024, week=1, base_path=tmp_path)

        # Should only find games week 1
        latest_games_w1 = get_latest_bronze_file(
            "games", season=2024, week=1, base_path=tmp_path
        )
        assert latest_games_w1 is not None
        assert "games_raw_bronze_2024_W01_" in latest_games_w1.name

        # Should only find games week 2
        latest_games_w2 = get_latest_bronze_file(
            "games", season=2024, week=2, base_path=tmp_path
        )
        assert latest_games_w2 is not None
        assert "games_raw_bronze_2024_W02_" in latest_games_w2.name

        # Should only find odds week 1
        latest_odds_w1 = get_latest_bronze_file(
            "odds", season=2024, week=1, base_path=tmp_path
        )
        assert latest_odds_w1 is not None
        assert "odds_raw_bronze_2024_W01_" in latest_odds_w1.name
