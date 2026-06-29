"""Injury-report ingestion using nflreadpy.load_injuries.

Produces timestamped Bronze snapshots and upserts a Silver injuries table
keyed at game grain (latest-wins by game_id).

Follows the scripts/ingest_weather.py pattern and scripts/ingest_snaps.py:
the standardized ingestion CLI (backfill + --current), a .to_pandas()
conversion at the nflreadpy/polars boundary, canonical team normalization
(hard-fail on unknowns), and the reused save_bronze_snapshot / upsert_silver
storage primitives.

load_injuries is weekly-final: exactly one row per (gsis_id, season, week).
For 2009-2024 each row carries date_modified (a UTC timestamp = when the
weekly-final row was last touched), which the Plan 28-05 InjuryBuilder uses
to fence to the Friday 6 PM ET freeze. The 2025 schema DROPPED date_modified
and ADDED season_type, so this script is schema-tolerant: it branches on
`"date_modified" in df.columns` and, when absent, carries a null date_modified
(the deploy-time builder then falls back to snapshot-time fencing -- Pitfall 4 /
D-19). It never raises KeyError on the missing column.

Injuries carry no game_id; it is derived by joining the season's silver games
on (season, week, team) so upsert_silver(..., key_column="game_id") is
idempotent at game grain.

Coverage floor: 2009.

Per D-19 this script is deploy-ready (supports both historical backfill and a
--current-week refresh) but is intentionally NOT registered in the Friday
orchestrator / make snapshot -- that wiring is deferred to deploy time
(Phase 30/31).
"""

import argparse
import sys

import nflreadpy as nfl
import pandas as pd

from data.storage import load_dataframe, save_bronze_snapshot, upsert_silver
from utils import get_logger, log_data_operation
from utils.ingestion_args import (
    add_standard_ingestion_args,
    parse_season_week_args,
)
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# Injury-report coverage floor (load_injuries docstring: since 2009).
INJURY_COVERAGE_FLOOR = 2009

# Columns retained in the silver injuries table (alongside the derived game_id).
# date_modified is kept when present (2009-2024) for the Plan 28-05 Friday fence;
# it is absent in the 2025 schema and is carried as null then (schema tolerance).
INJURY_COLUMNS = [
    "gsis_id",
    "team",
    "position",
    "full_name",
    "season",
    "week",
    "game_type",
    "report_status",
    "report_primary_injury",
    "report_secondary_injury",
    "practice_status",
    "practice_primary_injury",
    "practice_secondary_injury",
    "date_modified",
]


def _build_game_id_lookup(season: int) -> pd.DataFrame:
    """Build a (season, week, team) -> game_id lookup from the silver games table.

    Teams in the silver games table are already canonical; each team plays at
    most one regular-season game per week, so the lookup is unique per
    (season, week, team).
    """
    games = load_dataframe("games", layer="silver")
    games = games[games["season"] == season]

    home = games[["season", "week", "home_team", "game_id"]].rename(
        columns={"home_team": "team"}
    )
    away = games[["season", "week", "away_team", "game_id"]].rename(
        columns={"away_team": "team"}
    )
    lookup = pd.concat([home, away], ignore_index=True)
    return lookup.drop_duplicates(subset=["season", "week", "team"], keep="first")


def ingest_injuries_season(season: int, weeks: list[int] | None = None) -> pd.DataFrame:
    """Ingest injury reports for one season into Bronze + Silver.

    Args:
        season: NFL season year (>= 2009).
        weeks: Optional list of weeks to restrict to. None ingests the whole
            (regular) season.

    Returns:
        The game-grain injury DataFrame upserted to silver (may be empty).
    """
    if season < INJURY_COVERAGE_FLOOR:
        logger.warning(
            "Season below injury coverage floor; skipping",
            season=season,
            floor=INJURY_COVERAGE_FLOOR,
        )
        return pd.DataFrame()

    # polars -> pandas at the loader boundary (Pitfall 1).
    raw = nfl.load_injuries(season).to_pandas()

    if raw.empty:
        logger.warning("No injury data returned", season=season)
        return pd.DataFrame()

    # Match the regular-season convention used by snaps / team_form.
    df = raw[raw["game_type"] == "REG"].copy()

    if weeks is not None:
        df = df[df["week"].isin(weeks)].copy()

    if df.empty:
        logger.warning("No REG injury rows after filtering", season=season, weeks=weeks)
        return pd.DataFrame()

    # Schema tolerance (Pitfall 4): 2025 dropped date_modified. Carry it as a
    # null column so the silver schema is stable and the builder can fall back
    # to snapshot-time fencing. Never KeyError on the missing column.
    if "date_modified" not in df.columns:
        logger.warning(
            "date_modified absent (2025+ schema); carrying as null for "
            "snapshot-time fencing",
            season=season,
        )
        df["date_modified"] = pd.NaT

    # Canonical team normalization, hard-fail on unknowns.
    df["team"] = df["team"].apply(normalize_team_abbreviation)

    # Retain the relevant columns at (gsis_id, season, week) grain.
    df = df[INJURY_COLUMNS].copy()

    # Derive game_id by joining the season's silver games on (season, week, team).
    lookup = _build_game_id_lookup(season)
    df = df.merge(lookup, on=["season", "week", "team"], how="inner")

    if df.empty:
        logger.warning(
            "No injury rows matched a silver game on (season, week, team)",
            season=season,
        )
        return pd.DataFrame()

    # Bronze snapshot (append-only, timestamped). save_bronze_snapshot formats
    # W{week:02d}, so a season-wide / multi-week run must pass the week=0
    # sentinel exactly as ingest_weather.py:545-546 does (review #5).
    if weeks is not None and len(weeks) == 1:
        bronze_week = weeks[0]
    else:
        bronze_week = 0
    save_bronze_snapshot(df, "injuries", season=season, week=bronze_week)

    # Silver upsert: game-grain latest-wins -- idempotent under re-run (review #5).
    upsert_silver(df, "injuries", key_column="game_id")

    log_data_operation(
        operation="ingest",
        table="injuries",
        rows=len(df),
        season=season,
    )

    logger.info(
        "Injury ingestion completed",
        season=season,
        rows=len(df),
        unique_games=df["game_id"].nunique(),
    )

    return df


def main() -> None:
    """CLI entry point for injury ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest NFL injury reports (Bronze + Silver)"
    )
    parser = add_standard_ingestion_args(parser)
    args = parser.parse_args()

    from utils import setup_logging

    setup_logging()

    seasons, weeks = parse_season_week_args(args)

    total_rows = 0
    for season in seasons:
        df = ingest_injuries_season(season, weeks=weeks)
        total_rows += len(df)

    if total_rows == 0:
        print("No injury data ingested")
        return

    print(f"Successfully ingested {total_rows} injury rows")
    print(f"Seasons: {seasons}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("Injury ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
