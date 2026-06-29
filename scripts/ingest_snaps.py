"""Player-level snap-count ingestion using nflreadpy.load_snap_counts.

Produces timestamped Bronze snapshots and upserts a player-level Silver
snap_counts table (one row per game_id + player, latest-wins by game_id).

Follows the scripts/ingest_weather.py pattern exactly: the standardized
ingestion CLI (backfill + --current), a .to_pandas() conversion at the
nflreadpy/polars boundary, canonical team normalization (hard-fail on
unknowns -- SD/STL/OAK appear in snap data back to 2013), and the reused
save_bronze_snapshot / upsert_silver storage primitives.

Coverage floor: 2013. nflreadpy.load_snap_counts returns 0 rows for 2012 and
raises for pre-2012, so backfill should start at --seasons 2013 ... .

Per D-19 this script is deploy-ready (supports both historical backfill and a
--current-week refresh) but is intentionally NOT registered in the Friday
orchestrator / make snapshot -- that wiring is deferred to deploy time
(Phase 30/31).
"""

import argparse
import sys

import nflreadpy as nfl
import pandas as pd

from data.storage import save_bronze_snapshot, upsert_silver
from utils import get_logger, log_data_operation
from utils.ingestion_args import (
    add_standard_ingestion_args,
    parse_season_week_args,
)
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# Snap-count coverage floor. load_snap_counts(2012) returns 0 rows and
# pre-2012 raises "Season must be between 2012 and 2025".
SNAP_COVERAGE_FLOOR = 2013

# Raw post-game snap value columns (the D-13 leakage targets) retained in silver.
RAW_SNAP_COLUMNS = [
    "offense_snaps",
    "offense_pct",
    "defense_snaps",
    "defense_pct",
    "st_snaps",
    "st_pct",
]

# Join / grain keys retained alongside the raw snap values. Player-level grain
# (D-14): one row per (game_id, player). game_id is the silver upsert key.
SNAP_KEY_COLUMNS = [
    "game_id",
    "pfr_player_id",
    "player",
    "position",
    "team",
    "opponent",
    "season",
    "week",
]


def ingest_snaps_season(season: int, weeks: list[int] | None = None) -> pd.DataFrame:
    """Ingest player-level snap counts for one season into Bronze + Silver.

    Args:
        season: NFL season year (>= 2013).
        weeks: Optional list of weeks to restrict to. None ingests the whole
            (regular) season.

    Returns:
        The player-level snap DataFrame upserted to silver (may be empty).
    """
    if season < SNAP_COVERAGE_FLOOR:
        logger.warning(
            "Season below snap coverage floor; skipping",
            season=season,
            floor=SNAP_COVERAGE_FLOOR,
        )
        return pd.DataFrame()

    # polars -> pandas at the loader boundary (Pitfall 1).
    raw = nfl.load_snap_counts(season).to_pandas()

    if raw.empty:
        logger.warning("No snap data returned", season=season)
        return pd.DataFrame()

    # Match team_form's regular-season convention (game_type includes playoff
    # rounds WC/DIV/CON/SB; keep REG only).
    df = raw[raw["game_type"] == "REG"].copy()

    if weeks is not None:
        df = df[df["week"].isin(weeks)].copy()

    if df.empty:
        logger.warning("No REG snap rows after filtering", season=season, weeks=weeks)
        return pd.DataFrame()

    # Canonical team normalization, hard-fail on unknowns (SD/STL/OAK present
    # back to 2013).
    df["team"] = df["team"].apply(normalize_team_abbreviation)
    df["opponent"] = df["opponent"].apply(normalize_team_abbreviation)

    # Retain only the raw snap values + join/grain keys at player-level grain.
    keep_columns = SNAP_KEY_COLUMNS + RAW_SNAP_COLUMNS
    df = df[keep_columns].copy()

    # Bronze snapshot (append-only, timestamped). save_bronze_snapshot formats
    # W{week:02d}, so a season-wide / multi-week run must pass the week=0
    # sentinel exactly as ingest_weather.py:545-546 does (review #5).
    if weeks is not None and len(weeks) == 1:
        bronze_week = weeks[0]
    else:
        bronze_week = 0
    save_bronze_snapshot(df, "snap_counts", season=season, week=bronze_week)

    # Silver upsert: game-grain latest-wins. Removing all rows for any game_id
    # in the new frame then appending replaces a whole game's player rows
    # atomically -- idempotent under re-run (review #5).
    upsert_silver(df, "snap_counts", key_column="game_id")

    log_data_operation(
        operation="ingest",
        table="snap_counts",
        rows=len(df),
        season=season,
    )

    logger.info(
        "Snap-count ingestion completed",
        season=season,
        rows=len(df),
        unique_games=df["game_id"].nunique(),
    )

    return df


def main() -> None:
    """CLI entry point for snap-count ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest player-level NFL snap counts (Bronze + Silver)"
    )
    parser = add_standard_ingestion_args(parser)
    args = parser.parse_args()

    from utils import setup_logging

    setup_logging()

    seasons, weeks = parse_season_week_args(args)

    total_rows = 0
    for season in seasons:
        df = ingest_snaps_season(season, weeks=weeks)
        total_rows += len(df)

    if total_rows == 0:
        print("No snap-count data ingested")
        return

    print(f"Successfully ingested {total_rows} snap-count rows")
    print(f"Seasons: {seasons}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Snap-count ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
