"""A later append can never re-synthesize ``created_at`` (Plan 33.2-08, threat T-33.2-08-13).

Until this plan, ``data/storage.py::_migrate_schema_for_append`` replaced the WHOLE existing
``created_at`` column with ``datetime.now(UTC)`` whenever a legacy integer or object column failed
to parse. One ordinary append into silver ``odds_snapshot`` would therefore have turned the honest
NULLs this plan writes into plausible current timestamps, with only a log warning to say so.

These tests drive the real append paths against a sandbox root, never the helper directly:
``upsert_silver`` (the keyed silver path) and ``save_dataframe`` in append mode (the path the live
odds ingest writes ``odds_snapshot`` through, and the only one that reaches the migration).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import ParquetManager, save_dataframe, upsert_silver
from utils import DataIngestionError

UTC_NS = "datetime64[ns, UTC]"


def _odds(game_ids: list[str], created_at: list) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": game_ids,
            "sportsbook": ["consensus"] * len(game_ids),
            "spread": [3.0] * len(game_ids),
            "created_at": created_at,
        }
    )


def _nanoseconds(series: pd.Series) -> list:
    """Exact per-value identity, NULLs included (NaT compares as None here, never as a time)."""
    return [None if pd.isna(v) else pd.Timestamp(v).value for v in series]


def created_at_preserved(before: pd.DataFrame, after: pd.DataFrame) -> bool:
    """True when every pre-existing row's ``created_at`` survived byte-for-byte."""
    kept = after.set_index("game_id").loc[before["game_id"], "created_at"]
    return _nanoseconds(before["created_at"]) == _nanoseconds(kept)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(tmp_path)))
    (tmp_path / "silver").mkdir()
    return tmp_path


def _existing_with_nulls() -> pd.DataFrame:
    created = pd.Series(
        [pd.Timestamp("2026-09-05 04:59:49.969295", tz="UTC"), pd.NaT, pd.NaT],
        dtype=UTC_NS,
    )
    return _odds(["2025_W01_DAL@PHI", "2018_W01_ATL@PHI", "2019_W01_GB@CHI"], created)


def _new_row() -> pd.DataFrame:
    created = pd.Series(
        [pd.Timestamp("2026-09-21 12:00:00.123456", tz="UTC")], dtype=UTC_NS
    )
    return _odds(["2026_W03_KC@BUF"], created)


class TestTheKeyedSilverPath:
    def test_an_upsert_keeps_every_existing_created_at_nulls_included(
        self, sandbox: Path
    ) -> None:
        before = _existing_with_nulls()
        upsert_silver(before, "odds_snapshot", base_path=sandbox)
        upsert_silver(_new_row(), "odds_snapshot", base_path=sandbox)
        after = pd.read_parquet(sandbox / "silver" / "odds_snapshot.parquet")
        assert len(after) == 4
        assert created_at_preserved(before, after)


class TestTheAppendPathThatReachesTheMigration:
    def _seed(self, sandbox: Path, frame: pd.DataFrame) -> Path:
        path = sandbox / "silver" / "odds_snapshot.parquet"
        frame.to_parquet(path, index=False)
        return path

    def test_an_append_keeps_every_existing_created_at_nulls_included(
        self, sandbox: Path
    ) -> None:
        before = _existing_with_nulls()
        self._seed(sandbox, before)
        save_dataframe(_new_row(), "odds_snapshot", layer="silver", save_to_db=False)
        after = pd.read_parquet(sandbox / "silver" / "odds_snapshot.parquet")
        assert len(after) == 4
        assert created_at_preserved(before, after)

    def test_an_unparseable_created_at_is_refused_by_name_and_nothing_is_written(
        self, sandbox: Path
    ) -> None:
        path = self._seed(
            sandbox, _odds(["a", "b", "c"], ["2025-09-01T00:00:00Z", "not a time", "x"])
        )
        original = path.read_bytes()
        with pytest.raises(DataIngestionError, match=r"created_at.*2 value"):
            save_dataframe(
                _new_row(), "odds_snapshot", layer="silver", save_to_db=False
            )
        assert path.read_bytes() == original, (
            "a refused append must not rewrite the table"
        )

    def test_a_legacy_season_integer_is_refused_rather_than_read_as_1970(
        self, sandbox: Path
    ) -> None:
        self._seed(sandbox, _odds(["a", "b"], [2018, 2019]))
        with pytest.raises(DataIngestionError, match="created_at"):
            save_dataframe(
                _new_row(), "odds_snapshot", layer="silver", save_to_db=False
            )

    def test_a_partially_parseable_column_keeps_what_parsed(
        self, sandbox: Path
    ) -> None:
        self._seed(sandbox, _odds(["a", "b"], ["2025-09-01T00:00:00Z", None]))
        save_dataframe(_new_row(), "odds_snapshot", layer="silver", save_to_db=False)
        after = pd.read_parquet(sandbox / "silver" / "odds_snapshot.parquet").set_index(
            "game_id"
        )
        assert after.at["a", "created_at"] == pd.Timestamp("2025-09-01", tz="UTC")
        assert pd.isna(after.at["b", "created_at"])

    def test_a_clean_table_appends_and_changes_nothing(self, sandbox: Path) -> None:
        created = pd.Series(
            [pd.Timestamp("2026-01-01 01:02:03.456789", tz="UTC")] * 2, dtype=UTC_NS
        )
        before = _odds(["a", "b"], created)
        self._seed(sandbox, before)
        save_dataframe(_new_row(), "odds_snapshot", layer="silver", save_to_db=False)
        after = pd.read_parquet(sandbox / "silver" / "odds_snapshot.parquet")
        assert len(after) == 3
        assert created_at_preserved(before, after)


class TestTheIdentityCheckCatchesTheOldBehaviour:
    def test_a_wholesale_now_replacement_is_detected(self) -> None:
        before = _existing_with_nulls()
        replaced = before.copy()
        replaced["created_at"] = pd.Timestamp.now(tz="UTC")
        assert not created_at_preserved(before, replaced)
