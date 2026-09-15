"""A silver upsert must leave the reader's DuckDB copy agreeing with the parquet it wrote.

THE DEFECT THIS GUARDS (N-01, recurring)
----------------------------------------
``load_dataframe(source="auto")`` reads DuckDB FIRST whenever the table exists there, and
``upsert_silver`` used to write the PARQUET ONLY. So every upsert into a table that also has
a DuckDB copy was invisible to the whole pipeline, silently.

It has now happened twice. Phase 30 measured it as N-01 (``games`` parquet 6,499 against
DuckDB 6,292) and healed it with a one-off re-sync, leaving the writer unchanged. Plan 33-18's
live acceptance run, attempt 1 on 2026-09-15, reproduced it: ``ingest_games`` wrote the 2026
schedule to parquet (6,771 rows) while DuckDB stayed at 6,499 with ZERO 2026 rows, the weather
ingest logged "No games found" and reported success, and the run halted at ``data_qa``. The
owner ruled on 2026-09-15 to fix it at the writer (ruling R1).

THE RULES THE FIX FOLLOWS, each pinned below:

1. A table that ALREADY HAS a DuckDB copy has that copy replaced from the parquet the upsert
   just wrote, so ``auto`` reads the same rows. Same pattern
   ``scripts/build_elo.py::_upsert_row_table`` already uses for its own tables.
2. A table with NO DuckDB copy is left without one: the reader already falls back to its
   parquet, so it is consistent, and creating mirrors nobody reads widens the blast radius.
3. Only the reader's OWN root is synced. An upsert into a different ``base_path`` writes a
   parquet ``load_dataframe`` never reads, so it must not touch the reader's DuckDB -- which
   is also what keeps every sandboxed ``upsert_silver(base_path=tmp_path)`` in this suite from
   writing the production database.
4. A sync that fails FAILS LOUDLY. The parquet is already written when DuckDB is touched, so
   a swallowed DuckDB failure would reproduce the exact silent split this module exists for.

Sandboxed throughout: both the parquet manager and the DuckDB connection are pointed at
``tmp_path``. ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import (
    DuckDBConnection,
    ParquetManager,
    load_dataframe,
    upsert_silver,
    upsert_silver_composite,
)
from utils import DataIngestionError


def _games(ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ids,
            "season": [int(i[:4]) for i in ids],
            "week": [int(i[6:8]) for i in ids],
            "created_at": [datetime(2026, 9, 15, tzinfo=UTC)] * len(ids),
        }
    )


HISTORY = ["2025_W01_DAL@PHI", "2025_W01_KC@LAC", "2025_W02_TB@ATL"]
NEW_WEEK = ["2026_W02_DET@BUF", "2026_W02_CAR@ATL"]


@pytest.fixture
def reader_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The root load_dataframe reads: its parquet manager AND its DuckDB, both sandboxed."""
    root = tmp_path / "lake"
    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(root)))
    monkeypatch.setattr(
        storage_mod, "_db_connection", DuckDBConnection(str(root / "sandbox.duckdb"))
    )
    yield root
    storage_mod._db_connection.close()


def _seed_both_copies(root: Path, frame: pd.DataFrame, table: str) -> None:
    ParquetManager(str(root)).save(frame, f"silver/{table}.parquet")
    storage_mod.get_db_connection().create_table_from_df(frame, table, "replace")


def _parquet_ids(root: Path, table: str) -> set[str]:
    return set(pd.read_parquet(root / "silver" / f"{table}.parquet")["game_id"])


class TestAnExistingDuckDBCopyIsKeptInStep:
    """Rule 1 -- today's failure."""

    def test_auto_reads_every_row_the_upsert_wrote(self, reader_root):
        _seed_both_copies(reader_root, _games(HISTORY), "games")
        upsert_silver(_games(NEW_WEEK), "games", base_path=reader_root)

        auto_ids = set(
            load_dataframe("games", layer="silver", source="auto")["game_id"]
        )
        parquet_ids = _parquet_ids(reader_root, "games")
        assert auto_ids == parquet_ids, (
            f"load_dataframe(auto) returned {len(auto_ids)} game ids against "
            f"{len(parquet_ids)} in the parquet; missing from the auto read: "
            f"{sorted(parquet_ids - auto_ids)}. auto prefers DuckDB, so those rows are "
            "invisible to every consumer."
        )

    def test_the_duckdb_copy_has_the_parquets_row_count(self, reader_root):
        _seed_both_copies(reader_root, _games(HISTORY), "games")
        upsert_silver(_games(NEW_WEEK), "games", base_path=reader_root)

        db_rows = len(load_dataframe("games", layer="silver", source="db"))
        parquet_rows = len(pd.read_parquet(reader_root / "silver" / "games.parquet"))
        assert db_rows == parquet_rows == len(HISTORY) + len(NEW_WEEK)

    def test_the_composite_upsert_keeps_its_copy_in_step_too(self, reader_root):
        ts = pd.Timestamp("2026-09-15 12:00", tz="UTC")
        seed = pd.DataFrame(
            {
                "game_id": HISTORY,
                "snapshot_ts": [ts] * len(HISTORY),
                "total": [44.5] * 3,
            }
        )
        _seed_both_copies(reader_root, seed, "odds_timeline")
        new = pd.DataFrame(
            {
                "game_id": NEW_WEEK,
                "snapshot_ts": [ts] * len(NEW_WEEK),
                "total": [41.5] * 2,
            }
        )
        upsert_silver_composite(new, "odds_timeline", base_path=reader_root)

        auto_ids = set(
            load_dataframe("odds_timeline", layer="silver", source="auto")["game_id"]
        )
        assert auto_ids == _parquet_ids(reader_root, "odds_timeline")


class TestATableWithNoDuckDBCopyStaysParquetOnly:
    """Rule 2 -- a control: consistent today, and must stay consistent without a new mirror."""

    def test_auto_reads_the_parquet(self, reader_root):
        ParquetManager(str(reader_root)).save(_games(HISTORY), "silver/weather.parquet")
        upsert_silver(_games(NEW_WEEK), "weather", base_path=reader_root)

        auto_ids = set(
            load_dataframe("weather", layer="silver", source="auto")["game_id"]
        )
        assert auto_ids == _parquet_ids(reader_root, "weather")

    def test_no_duckdb_copy_is_created(self, reader_root):
        ParquetManager(str(reader_root)).save(_games(HISTORY), "silver/weather.parquet")
        upsert_silver(_games(NEW_WEEK), "weather", base_path=reader_root)

        assert not storage_mod.get_db_connection().table_exists("weather")


class TestAForeignRootNeverTouchesTheReadersDuckDB:
    """Rule 3 -- a control: what keeps sandboxed upserts off the production database."""

    def test_an_upsert_elsewhere_leaves_the_reader_copy_unchanged(
        self, reader_root, tmp_path
    ):
        _seed_both_copies(reader_root, _games(HISTORY), "games")
        elsewhere = tmp_path / "some_other_root"
        upsert_silver(_games(NEW_WEEK), "games", base_path=elsewhere)

        db_ids = set(load_dataframe("games", layer="silver", source="db")["game_id"])
        assert db_ids == set(HISTORY), (
            "an upsert into a root load_dataframe does not read changed the reader's "
            f"DuckDB copy: {sorted(db_ids)}"
        )


class TestAFailedSyncFailsLoudly:
    """Rule 4 -- a swallowed DuckDB failure would reproduce the silent split."""

    @pytest.fixture
    def duckdb_write_fails(self, monkeypatch):
        def _refuse(self, df, table_name, if_exists="replace"):
            raise DataIngestionError("simulated DuckDB write failure (held lock)")

        monkeypatch.setattr(DuckDBConnection, "create_table_from_df", _refuse)

    def test_the_upsert_raises(self, reader_root, duckdb_write_fails):
        ParquetManager(str(reader_root)).save(_games(HISTORY), "silver/games.parquet")
        storage_mod.get_db_connection().execute(
            "CREATE TABLE games AS SELECT 'x' AS game_id", write=True
        )
        with pytest.raises(DataIngestionError, match=r"(?i)duckdb"):
            upsert_silver(_games(NEW_WEEK), "games", base_path=reader_root)

    def test_after_the_failure_auto_does_not_serve_the_stale_copy(
        self, reader_root, duckdb_write_fails
    ):
        ParquetManager(str(reader_root)).save(_games(HISTORY), "silver/games.parquet")
        storage_mod.get_db_connection().execute(
            "CREATE TABLE games AS SELECT 'x' AS game_id", write=True
        )
        with pytest.raises(DataIngestionError):
            upsert_silver(_games(NEW_WEEK), "games", base_path=reader_root)

        auto_ids = set(
            load_dataframe("games", layer="silver", source="auto")["game_id"]
        )
        assert auto_ids == _parquet_ids(reader_root, "games"), (
            "the sync failed and the stale DuckDB copy is still what auto serves; the "
            "failed sync must at least drop it so readers fall back to the parquet"
        )
