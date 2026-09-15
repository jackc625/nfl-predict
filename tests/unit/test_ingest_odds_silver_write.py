"""A live odds pull must land in the ONE file every consumer reads.

THE DEFECT THIS GUARDS
----------------------
``OddsDataIngester.ingest_odds`` used to save silver odds with
``partition_cols=["snapshot_ts"] if len(validated_df) > 100 else None``. Over 100
rows -- which is EVERY real week, sixteen games times the US sportsbooks -- that
routed the write to ``pq.write_to_dataset(root_path=silver/)``, which:

* leaves ``silver/odds_snapshot.parquet`` untouched, so the week's lines reach
  none of the three readers, all of which open that single file
  (``models/train.py``, ``scripts/generate_current_week_predictions.py``,
  ``api/routes/health.py``);
* rewrites the WHOLE merged table -- history included -- as new hash-named files
  under one ``snapshot_ts=`` directory per distinct snapshot, in the SHARED silver
  root. That is the cross-table contamination
  ``tests/integration/test_storage_replace_mode.py`` documents as G-01.

Measured in a sandbox on 2026-09-15 (Plan 33-18): 128 new rows left the file at
0 of 2,140 rows for 2026 and added 31 files; 96 new rows took the single-file
branch and merged, 2,140 -> 2,236. Only the row count decided which one ran.

The fix follows the precedent CR-01 / D-10 set for gold and ``25c364f`` set for the
silver builders: one self-contained file, no partition directories in a shared
layer root. It keeps the APPEND merge -- the historical rows must survive -- which
is why these tests seed a history and assert it is still there.

EVERYTHING HERE IS SANDBOXED. The parquet manager is pointed at ``tmp_path``, the
API client is a stand-in that returns a documented-shape payload, and both odds
saves already pass ``save_to_db=False``. No network, no DuckDB, no production store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import ParquetManager
from scripts.ingest_odds import OddsDataIngester

ET = ZoneInfo("America/New_York")

# The row count at and above which the old write partitioned. The fixture must sit
# ABOVE it or these tests would pass on the defective code without exercising it.
OLD_PARTITION_THRESHOLD_ROWS = 100

# Thirteen games times eight books = 104 rows: over the old threshold by a margin a
# single validation drop cannot erase.
GAMES_IN_PAYLOAD = 13
BOOKS_PER_GAME = 8

HISTORICAL_ROW_COUNT = 5

SNAPSHOT_TIME = datetime(2026, 9, 18, 18, 0, tzinfo=ET)

_TEAM_NAMES: tuple[str, ...] = (
    "Atlanta Falcons",
    "Carolina Panthers",
    "Baltimore Ravens",
    "New Orleans Saints",
    "Chicago Bears",
    "Minnesota Vikings",
    "Houston Texans",
    "Cincinnati Bengals",
    "New England Patriots",
    "Pittsburgh Steelers",
    "New York Jets",
    "Green Bay Packers",
    "Tampa Bay Buccaneers",
    "Cleveland Browns",
    "Tennessee Titans",
    "Philadelphia Eagles",
    "Denver Broncos",
    "Jacksonville Jaguars",
    "Los Angeles Chargers",
    "Las Vegas Raiders",
    "Arizona Cardinals",
    "Seattle Seahawks",
    "Dallas Cowboys",
    "Washington Commanders",
    "San Francisco 49ers",
    "Miami Dolphins",
)


class _StandInOddsClient:
    """Returns a fixed documented-shape payload. Never touches the network."""

    def __init__(self, payload: list[dict]) -> None:
        self._payload = payload

    def get_nfl_odds(self, **_kwargs) -> list[dict]:
        return self._payload

    def close(self) -> None:
        return None


def _payload() -> list[dict]:
    """Thirteen games in the Odds API v4 response shape, eight books each."""
    games = []
    for index in range(GAMES_IN_PAYLOAD):
        home = _TEAM_NAMES[2 * index]
        away = _TEAM_NAMES[2 * index + 1]
        bookmakers = [
            {
                "key": f"book{book}",
                "last_update": "2026-09-15T04:00:00Z",
                "markets": [
                    {
                        "key": "totals",
                        "outcomes": [
                            {"name": "Over", "price": -110, "point": 44.5},
                            {"name": "Under", "price": -110, "point": 44.5},
                        ],
                    },
                ],
            }
            for book in range(BOOKS_PER_GAME)
        ]
        games.append(
            {
                "id": f"event{index}",
                "commence_time": "2026-09-20T17:00:00Z",
                "home_team": home,
                "away_team": away,
                "bookmakers": bookmakers,
            }
        )
    return games


def _historical_odds() -> pd.DataFrame:
    """A 2025 history in the production file's own column types."""
    return pd.DataFrame(
        {
            "game_id": [f"2025_W01_DAL@PHI{i}" for i in range(HISTORICAL_ROW_COUNT)],
            "sportsbook": ["consensus"] * HISTORICAL_ROW_COUNT,
            "ml_home": [-425.0] * HISTORICAL_ROW_COUNT,
            "ml_away": [330.0] * HISTORICAL_ROW_COUNT,
            "spread": [8.5] * HISTORICAL_ROW_COUNT,
            "total": [47.5] * HISTORICAL_ROW_COUNT,
            "is_live": [False] * HISTORICAL_ROW_COUNT,
            "last_update": pd.Series(
                [pd.NaT] * HISTORICAL_ROW_COUNT, dtype="datetime64[ns, UTC]"
            ),
            "snapshot_ts": ["2025-08-29 22:00:00+00:00"] * HISTORICAL_ROW_COUNT,
            "created_at": pd.Series(
                [datetime(2026, 9, 5, tzinfo=UTC)] * HISTORICAL_ROW_COUNT,
                dtype="datetime64[ns, UTC]",
            ),
            "spread_ju_home": [-110.0] * HISTORICAL_ROW_COUNT,
            "spread_ju_away": [-110.0] * HISTORICAL_ROW_COUNT,
            "total_over_ju": [-110.0] * HISTORICAL_ROW_COUNT,
            "total_under_ju": [-110.0] * HISTORICAL_ROW_COUNT,
        }
    )


@pytest.fixture
def sandbox_lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A tmp data lake holding a seeded odds history; the parquet manager points at it."""
    manager = ParquetManager(str(tmp_path))
    manager.save(_historical_odds(), "silver/odds_snapshot.parquet")
    monkeypatch.setattr(storage_mod, "_parquet_manager", manager)
    return tmp_path


@pytest.fixture
def ingested(sandbox_lake: Path) -> pd.DataFrame:
    """Run the REAL ingest path once against the stand-in client."""
    ingester = OddsDataIngester.__new__(OddsDataIngester)
    ingester.api_client = _StandInOddsClient(_payload())
    ingester.sportsbook_priority = []
    return ingester.ingest_odds(season=2026, week=2, snapshot_time=SNAPSHOT_TIME)


class TestTheFixtureExercisesTheOldPartitionBranch:
    """Non-vacuity: under the threshold these tests would pass on the defective code."""

    def test_the_validated_frame_is_over_one_hundred_rows(self, ingested):
        assert len(ingested) > OLD_PARTITION_THRESHOLD_ROWS, (
            f"the ingest produced {len(ingested)} validated rows; the old write only "
            f"partitioned ABOVE {OLD_PARTITION_THRESHOLD_ROWS}, so this fixture would "
            "not exercise the branch it exists to guard"
        )


class TestAnOverThresholdPullLandsInTheSingleFile:
    """The week's lines must be readable from ``silver/odds_snapshot.parquet``."""

    def test_the_new_rows_are_in_the_single_file(self, sandbox_lake, ingested):
        stored = pd.read_parquet(sandbox_lake / "silver" / "odds_snapshot.parquet")
        new_ids = set(ingested["game_id"])
        found = stored[stored["game_id"].isin(new_ids)]
        assert len(found) == len(ingested), (
            f"silver/odds_snapshot.parquet carries {len(found)} of the {len(ingested)} "
            "rows this pull validated. Every consumer reads that one file, so rows "
            "anywhere else are rows no model, prediction or decision ever sees."
        )

    def test_the_historical_rows_survive_the_write(self, sandbox_lake, ingested):
        stored = pd.read_parquet(sandbox_lake / "silver" / "odds_snapshot.parquet")
        history = stored[stored["game_id"].str.startswith("2025_")]
        assert len(history) == HISTORICAL_ROW_COUNT, (
            f"{len(history)} of {HISTORICAL_ROW_COUNT} historical rows remain after the "
            "pull. The silver odds write is an APPEND merge; a fix that turned it into a "
            "replace would trade one lost week for every lost season."
        )

    def test_no_snapshot_ts_partition_directory_is_created(
        self, sandbox_lake, ingested
    ):
        silver = sandbox_lake / "silver"
        partition_dirs = sorted(
            entry.name
            for entry in silver.iterdir()
            if entry.is_dir() and "=" in entry.name
        )
        assert partition_dirs == [], (
            f"the odds pull created partition directories in the shared silver root: "
            f"{partition_dirs}. That is the G-01 contamination "
            "tests/integration/test_storage_replace_mode.py documents."
        )
