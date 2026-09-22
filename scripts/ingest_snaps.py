"""Player-level snap-count ingestion using nflreadpy.load_snap_counts.

Produces timestamped Bronze snapshots and upserts a player-level Silver
snap_counts table (one row per game_id + player, latest-wins by game_id).

Follows the scripts/ingest_weather.py pattern: the standardized ingestion CLI
(backfill + --current), a .to_pandas() conversion at the nflreadpy/polars
boundary, canonical team normalization (hard-fail on unknowns -- SD/STL/OAK
appear in snap data back to 2013), and the reused save_bronze_snapshot /
upsert_silver storage primitives.

WHAT PLAN 33.2-15 CHANGED (D33.2-16: the dead feed is fixed, not worked around)
------------------------------------------------------------------------------
* Silver ``snap_counts`` stopped at 2024 and this script ran nowhere, so a 2025
  or 2026 build had no current snap data at all (measured: 2025 weeks 3 and 15
  produced identical snap values). It is now a daily pipeline step
  (``pipeline.steps.step_ingest_snaps``) and was run for 2025-2026.
* Every batch reaches silver THROUGH the promotion gate:
  ``data.quality_gates.validate_bronze_to_silver(df, SnapCountSchema)`` sits
  between the bronze snapshot and the silver upsert, exactly where
  ``scripts/ingest_weather.py`` places it for weather. One bad row fails the
  whole batch.
* Every captured row carries ``upstream_captured_at``: the nflverse release
  asset's ``updated_at`` for the file it came from (``data.upstream_asset_stamp``),
  read on both sides of the download so the stamp is the publication instant of
  exactly the bytes fetched.
* The bronze snapshot is written with ``exclusive=True``: a second capture inside
  the same second RAISES ``FileExistsError`` instead of silently overwriting the
  first (WR-04), and silver stays latest-wins by ``game_id``, so a repeat ingest of
  unchanged upstream leaves silver byte-identical.

Coverage floor: 2013. nflreadpy.load_snap_counts returns 0 rows for 2012 and
raises for pre-2012, so backfill should start at --seasons 2013 ... .
"""

import argparse
import sys
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from data.quality_gates import validate_bronze_to_silver
from data.schemas import SnapCountSchema
from data.storage import save_bronze_snapshot, upsert_silver
from data.upstream_asset_stamp import fetch_with_stamp, season_asset_name
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

#: The nflverse release (and nflreadpy dataset) snap counts are published under.
SNAP_DATASET = "snap_counts"

#: The capture-provenance column (the same literal ``features.injury.UPSTREAM_CAPTURE_COLUMN``
#: and ``data.schemas.SnapCountSchema`` declare).
CAPTURE_COLUMN = "upstream_captured_at"

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


def _load_season(season: int) -> pd.DataFrame:
    """nflreadpy's snap counts for *season*, polars -> pandas at the boundary (Pitfall 1)."""
    return nfl.load_snap_counts(season).to_pandas()


def ingest_snaps_season(
    season: int,
    weeks: list[int] | None = None,
    *,
    loader: Callable[[int], pd.DataFrame] = _load_season,
    stamp_reader: Callable[..., datetime] | None = None,
    base_path: Path | None = None,
) -> pd.DataFrame:
    """Ingest player-level snap counts for one season into Bronze + Silver.

    Args:
        season: NFL season year (>= 2013).
        weeks: Optional list of weeks to restrict to. None ingests the whole
            (regular) season.
        loader: Downloads one season (the test seam; defaults to nflreadpy).
        stamp_reader: ``data.upstream_asset_stamp.asset_published_at``'s shape (the test
            seam).
        base_path: Data root (the test seam; defaults to the configured root).

    Returns:
        The VALIDATED player-level snap DataFrame upserted to silver (may be empty).

    Raises:
        data.upstream_asset_stamp.UpstreamStampUnavailable: the asset's publication time
            could not be read, or the asset changed during the download. Nothing is written.
        utils.exceptions.DataValidationError: any row failed ``SnapCountSchema``; nothing
            reaches silver.
    """
    if season < SNAP_COVERAGE_FLOOR:
        logger.warning(
            "Season below snap coverage floor; skipping",
            season=season,
            floor=SNAP_COVERAGE_FLOOR,
        )
        return pd.DataFrame()

    asset = season_asset_name(SNAP_DATASET, season)
    raw, stamp = fetch_with_stamp(
        asset, lambda: loader(season), stamp_reader=stamp_reader
    )

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
    df = df[SNAP_KEY_COLUMNS + RAW_SNAP_COLUMNS].copy()

    # THE CAPTURE STAMP IS PROVENANCE ONLY FOR A SNAP ROW. A snap count is a mechanical
    # record of a COMPLETED game, published within hours of it (four refreshes a day in
    # season, RESEARCH section 1). Its information time is the contributing game's END
    # instant -- features.snaps admits a prior team-game once kickoff plus the declared
    # duration is at or before the target game's lock (Plan 33.2-14's registered rule). This
    # stamp records when upstream published the FILE we hold; it is never read as the time a
    # snap count became known.
    df[CAPTURE_COLUMN] = stamp

    # Bronze snapshot (append-only, timestamped) of the frame BEFORE validation. A
    # season-wide / multi-week run passes the week=0 sentinel (review #5). exclusive=True:
    # a same-second second capture raises FileExistsError rather than overwriting (WR-04).
    if weeks is not None and len(weeks) == 1:
        bronze_week = weeks[0]
    else:
        bronze_week = 0
    save_bronze_snapshot(
        df,
        "snap_counts",
        season=season,
        week=bronze_week,
        base_path=base_path,
        exclusive=True,
    )

    # THE PROMOTION GATE (Plan 33.2-15, adjudication #2). Every row is rebuilt through
    # SnapCountSchema; one bad row fails the batch and nothing reaches silver.
    validated = validate_bronze_to_silver(df, SnapCountSchema)
    validated[CAPTURE_COLUMN] = pd.to_datetime(validated[CAPTURE_COLUMN], utc=True)

    # Silver upsert: game-grain latest-wins. Removing all rows for any game_id
    # in the new frame then appending replaces a whole game's player rows
    # atomically -- idempotent under re-run (review #5).
    upsert_silver(validated, "snap_counts", key_column="game_id", base_path=base_path)

    log_data_operation(
        operation="ingest",
        table="snap_counts",
        rows=len(validated),
        season=season,
    )

    logger.info(
        "Snap-count ingestion completed",
        season=season,
        rows=len(validated),
        unique_games=validated["game_id"].nunique(),
        upstream_captured_at=stamp.isoformat(),
    )

    return validated


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
        stamp = df[CAPTURE_COLUMN].iloc[0].isoformat() if len(df) else None
        print(f"SEASON= {season} ROWS_WRITTEN= {len(df)} STAMP= {stamp}")

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
