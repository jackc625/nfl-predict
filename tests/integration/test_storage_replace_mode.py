"""G-01: replace_mode idempotency + table-scoped partition reader (Phase 20, FIX-01, D-13).

Two behavioral regressions guarded:

1. replace_mode idempotency -- save_dataframe(replace_mode=True) called twice must
   produce a single .parquet FILE (not a partition directory tree) with exactly the
   same row count as the input, regardless of how many times it runs.  The bug class:
   pq.write_to_dataset wrote partition dirs into the shared {layer}/ root; every run
   appended a new hash-named file, silently multiplying row counts.

2. Scoped partition reader -- ParquetManager.load on a target table must NEVER return
   rows belonging to a sibling table whose partition directories land in the same
   shared layer root.  The bug class: the generic fallback scanned parent_dir (the
   shared silver/ root) for ANY name=value dir, so loading "weather_features" would
   silently return odds_snapshot's snapshot_ts= partition rows.

All tests use tmp_path + save_to_db=False for full isolation -- no real data lake
touched, no DuckDB connection needed.
"""

from __future__ import annotations

import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import ParquetManager, save_dataframe

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _weather_df(n: int = 4) -> pd.DataFrame:
    """Minimal weather_features-shaped DataFrame (no datetime cols -> no tz issues)."""
    return pd.DataFrame(
        {
            "game_id": [f"2024_W0{i + 1}_BUF@KC" for i in range(n)],
            "season": [2024] * n,
            "temp_f": [55.0 + i for i in range(n)],
            "wind_mph": [10.0 + i for i in range(n)],
        }
    )


def _odds_df(n: int = 3) -> pd.DataFrame:
    """Minimal odds_snapshot-shaped DataFrame (no datetime cols -> no tz issues)."""
    return pd.DataFrame(
        {
            "game_id": [f"2024_W0{i + 1}_NE@MIA" for i in range(n)],
            "sportsbook": ["DraftKings"] * n,
            "spread": [-3.5 + i for i in range(n)],
        }
    )


# ---------------------------------------------------------------------------
# G-01 part 1: replace_mode idempotency
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestReplaceModeIdempotency:
    """save_dataframe(replace_mode=True) produces stable cardinality + single file (G-01)."""

    def test_replace_mode_twice_has_stable_row_count(self, tmp_path, monkeypatch):
        """Writing the same frame twice via replace_mode must not multiply rows.

        Regression guard: the old partitioned-append path called pq.write_to_dataset
        into the shared silver/ root on every run, growing file count indefinitely.
        replace_mode=True must write ONE self-contained file and overwrite it on the
        second call -- row count == len(input) both before and after the second write.
        """
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(tmp_path))
        )

        df = _weather_df(4)

        # First write
        save_dataframe(
            df,
            "weather_features",
            layer="silver",
            replace_mode=True,
            save_to_db=False,
        )

        # Second write -- same data; must not add rows
        save_dataframe(
            df,
            "weather_features",
            layer="silver",
            replace_mode=True,
            save_to_db=False,
        )

        result_path = tmp_path / "silver" / "weather_features.parquet"
        assert result_path.exists() and result_path.is_file(), (
            f"replace_mode must produce a single .parquet FILE at {result_path}; "
            f"found directory or nothing -- the old partitioned-append antipattern re-emerged"
        )

        loaded = pd.read_parquet(result_path)
        assert len(loaded) == len(df), (
            f"replace_mode idempotency regression: expected {len(df)} rows after two "
            f"identical writes, got {len(loaded)} -- row multiplication detected"
        )

    def test_replace_mode_result_matches_input_rows(self, tmp_path, monkeypatch):
        """Re-loading after replace_mode must return exactly the written rows (G-01).

        Ensures the single-file write path is complete and readable, not just present.
        """
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(tmp_path))
        )

        df = _weather_df(6)
        save_dataframe(
            df,
            "weather_features",
            layer="silver",
            replace_mode=True,
            save_to_db=False,
        )
        # Write again to exercise the overwrite path
        save_dataframe(
            df,
            "weather_features",
            layer="silver",
            replace_mode=True,
            save_to_db=False,
        )

        pm = ParquetManager(str(tmp_path))
        loaded = pm.load("silver/weather_features.parquet")

        assert set(loaded["game_id"]) == set(df["game_id"]), (
            f"After replace_mode double-write, loaded game_ids differ from input. "
            f"Expected {set(df['game_id'])}, got {set(loaded['game_id'])}"
        )

    def test_replace_mode_is_single_file_not_partition_tree(
        self, tmp_path, monkeypatch
    ):
        """replace_mode must never create partition directories (G-01).

        The bug class: pq.write_to_dataset scattered season=YYYY/ subdirs under
        silver/ -- the fix must produce exactly ONE .parquet file, no subdirectories.
        """
        monkeypatch.setattr(
            storage_mod, "_parquet_manager", ParquetManager(str(tmp_path))
        )

        df = _weather_df(4)
        save_dataframe(
            df,
            "weather_features",
            layer="silver",
            replace_mode=True,
            save_to_db=False,
        )

        silver_dir = tmp_path / "silver"
        # Collect any directories that look like partitions (name=value pattern)
        partition_dirs = [
            d for d in silver_dir.iterdir() if d.is_dir() and "=" in d.name
        ]
        assert partition_dirs == [], (
            f"replace_mode created partition directories in the shared silver/ root: "
            f"{[d.name for d in partition_dirs]} -- the old antipattern is back"
        )


# ---------------------------------------------------------------------------
# G-01 part 2: scoped partition reader -- no cross-table contamination
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestScopedPartitionReader:
    """pm.load on a target table never returns sibling table's rows (G-01).

    Simulates the historical contamination: odds_snapshot's snapshot_ts= dirs land
    in the shared silver/ root alongside weather_features.parquet; loading
    weather_features must return ONLY its own rows, not odds rows.
    """

    def test_loading_target_table_never_returns_sibling_rows(self, tmp_path):
        """Cross-table contamination regression: loading weather_features must not
        include any rows from the odds_snapshot sibling partition directories (G-01).

        Bug: the old generic fallback scanned the shared silver/ root for any name=value
        dir, so weather_features load would silently ingest odds_snapshot's partition
        files and return the wrong table's data -- discovered when team_game_stats had
        127,922 duplicate rows and weather had ~1048x multiplication.
        """
        pm = ParquetManager(str(tmp_path))

        # Write the TARGET table as a single file (the correct post-fix layout)
        weather_df = _weather_df(4)
        pm.save(weather_df, "silver/weather_features.parquet")

        # Simulate the SIBLING table's old partition dirs existing in the shared root.
        # Write odds_snapshot with a partition column directly into silver/ root.
        # Use URL-encoded form (as pq.write_to_dataset would produce on Windows, where
        # colons are illegal in filenames): snapshot_ts=2024-11-15%2022%3A00%3A00
        odds_df = _odds_df(3)
        # Manually write a partition-style dir that would have existed pre-fix.
        # Use a simpler name=value dir that avoids Windows colon restriction.
        sibling_dir = tmp_path / "silver" / "season=2024"
        sibling_dir.mkdir(parents=True, exist_ok=True)
        odds_df.to_parquet(sibling_dir / "part-0.parquet", index=False)

        # Loading the TARGET table must return ONLY its own rows
        loaded = pm.load("silver/weather_features.parquet")

        loaded_ids = set(loaded["game_id"])
        sibling_ids = set(odds_df["game_id"])
        weather_ids = set(weather_df["game_id"])

        assert loaded_ids == weather_ids, (
            f"Cross-table contamination: pm.load('silver/weather_features.parquet') "
            f"returned rows from the sibling season=2024 partition directory. "
            f"Got game_ids {loaded_ids}, expected only {weather_ids}. "
            f"Sibling ids that leaked in: {loaded_ids & sibling_ids}"
        )
        assert not (loaded_ids & sibling_ids), (
            f"Sibling table rows contaminated the target table load: "
            f"sibling game_ids {loaded_ids & sibling_ids} appeared in weather_features result"
        )

    def test_loading_nonexistent_table_raises_not_returns_sibling_rows(self, tmp_path):
        """When a table has no single file AND no own subdirectory, load must raise
        loudly (not silently return a sibling's rows via the shared-root fallback).

        A loud miss is correct; a silent wrong-table read is the data corruption bug
        that FIX-01 eradicated.
        """
        pm = ParquetManager(str(tmp_path))

        # Write a sibling with partition dirs in the shared silver/ root
        sibling_dir = tmp_path / "silver" / "season=2024"
        sibling_dir.mkdir(parents=True, exist_ok=True)
        _odds_df(3).to_parquet(sibling_dir / "part-0.parquet", index=False)

        # Try to load a table that has NO single file and NO own subdir
        from data.storage import DataIngestionError

        with pytest.raises(DataIngestionError, match="not found"):
            pm.load("silver/team_game_stats.parquet")
