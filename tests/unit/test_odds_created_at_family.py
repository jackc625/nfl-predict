"""The 1970 ``created_at`` family in silver ``odds_snapshot``: one cause, one entry (Plan 33.2-08).

CAUSE, read at the source rather than inferred from the values: the first historical odds ingest
(``scripts/ingest_historical_odds.py`` as added in ``7c70abf``, line 71) wrote
``"created_at": game.get("season")`` -- the SEASON integer -- and a later append read that integer
column through ``data/storage.py::_migrate_schema_for_append``'s ``pd.to_datetime(..., unit="s")``,
so season 2018 became 2,018 seconds after the epoch: 1970-01-01 00:33:38.

The family is recorded ONCE as a ``[[family]]`` entry with its full membership enumerated, and
every listed row now carries either a real timestamp or an honest NULL. No row was dropped.
Two detector controls keep the zero from being vacuous: a planted 1970 value IS flagged, and a
valid recent capture instant is NOT.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import pytest

from scripts.repair_odds_snapshot import (
    CREATED_AT_FAMILY_NAME,
    epoch_1970_mask,
    legacy_integer_created_at_mask,
)

ODDS_PATH = Path("data/silver/odds_snapshot.parquet")
RECORD_PATH = Path("config/odds_corrections.toml")
FAMILY_FIELDS = {
    "name",
    "cause",
    "detector",
    "disposition",
    "reason",
    "affected_count",
    "affected_game_ids",
}

needs_lake = pytest.mark.skipif(
    not ODDS_PATH.exists(), reason="silver odds_snapshot not built"
)


def _frame(game_id: str, created_at: pd.Timestamp | None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [game_id],
            "created_at": pd.Series([created_at], dtype="datetime64[ns, UTC]"),
        }
    )


class TestTheDetector:
    def test_a_planted_1970_value_is_flagged_by_symptom_and_mechanism(self) -> None:
        planted = _frame("2023_W18_NYJ@NE", pd.Timestamp(2023, unit="s", tz="UTC"))
        assert epoch_1970_mask(planted).tolist() == [True]
        assert legacy_integer_created_at_mask(planted).tolist() == [True]

    def test_a_valid_recent_capture_is_not_flagged(self) -> None:
        valid = _frame(
            "2025_W01_DAL@PHI", pd.Timestamp("2026-09-05 04:59:49.969295", tz="UTC")
        )
        assert not epoch_1970_mask(valid).any()
        assert not legacy_integer_created_at_mask(valid).any()

    def test_the_mechanism_sees_an_integer_the_year_filter_cannot(self) -> None:
        # 100,000,000 seconds is 1973: the same legacy integer read, not in 1970.
        disguised = _frame(
            "2019_W01_ARI@DET", pd.Timestamp(100_000_000, unit="s", tz="UTC")
        )
        assert not epoch_1970_mask(disguised).any()
        assert legacy_integer_created_at_mask(disguised).tolist() == [True]

    def test_a_null_is_not_a_member(self) -> None:
        assert not legacy_integer_created_at_mask(
            _frame("2019_W01_ARI@DET", None)
        ).any()


@needs_lake
class TestTheLiveFamily:
    @pytest.fixture(scope="class")
    def odds(self) -> pd.DataFrame:
        return pd.read_parquet(ODDS_PATH)

    @pytest.fixture(scope="class")
    def record(self) -> dict:
        assert RECORD_PATH.exists(), f"{RECORD_PATH} is the committed record R12 reads"
        return tomllib.loads(RECORD_PATH.read_text(encoding="utf-8"))

    @pytest.fixture(scope="class")
    def family(self, record: dict) -> dict:
        families = [
            f
            for f in record.get("family", [])
            if f.get("name") == CREATED_AT_FAMILY_NAME
        ]
        assert len(families) == 1, (
            "exactly one [[family]] entry records the created_at defect"
        )
        return families[0]

    def test_zero_created_at_values_fall_in_1970(self, odds: pd.DataFrame) -> None:
        assert int(epoch_1970_mask(odds).sum()) == 0
        assert int(legacy_integer_created_at_mask(odds).sum()) == 0

    def test_the_family_entry_carries_all_seven_fields(self, family: dict) -> None:
        assert FAMILY_FIELDS - set(family) == set()

    def test_the_count_matches_the_enumerated_membership(self, family: dict) -> None:
        assert family["affected_count"] == len(family["affected_game_ids"]) > 0
        assert len(set(family["affected_game_ids"])) == len(family["affected_game_ids"])

    def test_every_member_is_in_the_table_with_an_honest_timestamp(
        self, family: dict, odds: pd.DataFrame
    ) -> None:
        ids = set(family["affected_game_ids"])
        assert ids - set(odds["game_id"]) == set()
        members = odds[odds["game_id"].isin(ids)]
        created = members["created_at"]
        assert (created.isna() | (created.dt.year != 1970)).all()

    def test_no_row_was_dropped(self, record: dict, odds: pd.DataFrame) -> None:
        measured = record["measurements"]["created_at_family"]
        assert (
            measured["table_rows_before"] == measured["table_rows_after"] == len(odds)
        )
