"""Injury-report ingestion using nflreadpy.load_injuries.

Produces timestamped Bronze snapshots and a Silver injuries table that ACCUMULATES
captures, keyed on :data:`INJURY_CAPTURE_KEY` (33.2 review C1 CR-04 = B CR-01).

Follows the scripts/ingest_weather.py pattern and scripts/ingest_snaps.py:
the standardized ingestion CLI (backfill + --current), a .to_pandas()
conversion at the nflreadpy/polars boundary, canonical team normalization
(hard-fail on unknowns), and the reused save_bronze_snapshot / upsert_silver
storage primitives.

load_injuries is weekly-final: exactly one row per (gsis_id, season, week).
For 2009-2024 each row carries date_modified (a UTC timestamp = when the
weekly-final row was last touched), which features.injury uses as the row's
information time against each game's lock. The 2025 schema DROPPED
date_modified (RESEARCH 5.1), so this script is schema-tolerant: when the
column is absent it carries a null date_modified. It never raises KeyError on
the missing column.

WHAT PLAN 33.2-15 CHANGED (D33.2-16: the dead feed is fixed, not worked around)
------------------------------------------------------------------------------
* Silver ``injuries`` stopped at 2024 and this script ran nowhere. It is now a
  daily pipeline step (``pipeline.steps.step_ingest_injuries``) and was run for
  2025-2026.
* Every batch reaches silver THROUGH the promotion gate:
  ``validate_bronze_to_silver(df, InjurySchema)`` between the bronze snapshot and
  the silver upsert. One bad row fails the whole batch.
* Every captured row carries ``upstream_captured_at``, the nflverse release
  asset's ``updated_at`` for the file it came from -- CAPTURE PROVENANCE, read on
  both sides of the download (``data.upstream_asset_stamp.fetch_with_stamp``).
* Bronze is written with ``exclusive=True`` (a same-second collision raises rather
  than overwrites).
* SILVER ACCUMULATES CAPTURES (33.2 review C1 CR-04 = B CR-01). It used to be latest-wins
  by ``game_id``, so the nightly whole-season re-capture REPLACED every played game's rows
  -- captured before its lock -- with the same reports carrying tonight's stamp, after the
  lock. For 2025+ (no ``date_modified``) that stamp is a row's only information time, so
  every played game lost the only admissible record it had. Each capture is now its own
  rows (:data:`INJURY_CAPTURE_KEY`); an identical re-capture is idempotent, and
  ``features.injury`` reads the latest capture at or before each game's lock.
* POSTSEASON REPORTS ARE KEPT (Plan 33.2-15 extra step 6b). The ingest used to keep
  ``game_type == "REG"`` only, so every postseason game read as "no report admitted"
  although upstream publishes its reports: measured 2026-09-22, 3,544 postseason rows
  over 2009-2025, with a per-row ``date_modified`` for 2010-2024 (2009's carry none;
  2025's schema has no column; 2023 upstream carries the Wild Card round only). They are
  timed exactly like regular-season reports -- a row counts for a game only when its own
  time is at or before that game's lock -- so nothing is admitted that was not known.

Injuries carry no game_id; it is derived by joining the season's silver games
on (season, week, team).

Coverage floor: 2009.
"""

import argparse
import sys
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path

import nflreadpy as nfl
import pandas as pd

from data.quality_gates import validate_bronze_to_silver
from data.schemas import InjurySchema
from data.storage import load_dataframe, save_bronze_snapshot, upsert_silver_composite
from data.upstream_asset_stamp import fetch_with_stamp, season_asset_name
from utils import get_logger, log_data_operation
from utils.exceptions import DataValidationError
from utils.ingestion_args import (
    add_standard_ingestion_args,
    parse_season_week_args,
)
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# Injury-report coverage floor (load_injuries docstring: since 2009).
INJURY_COVERAGE_FLOOR = 2009

#: The nflverse release (and nflreadpy dataset) injury reports are published under.
INJURY_DATASET = "injuries"

#: The capture-provenance column. The SAME literal ``features.injury.UPSTREAM_CAPTURE_COLUMN``
#: declares (the builder reads it) and ``data.schemas.InjurySchema`` declares (the gate keeps
#: it); tests/unit/test_injury_capture_time_basis.py asserts the spellings are one.
CAPTURE_COLUMN = "upstream_captured_at"

#: One stored row per player report PER CAPTURE. ``date_modified`` is in the key because
#: upstream publishes more than one dated report for a player in one game week (measured
#: 2026-09-24: two 2024 week-15 pairs), and the capture stamp because each capture is kept.
INJURY_CAPTURE_KEY: tuple[str, ...] = (
    "game_id",
    "gsis_id",
    "date_modified",
    CAPTURE_COLUMN,
)

#: Every game type upstream files injury reports under: the regular season and the four
#: postseason rounds. A code outside this set is REFUSED by name rather than silently dropped.
INJURY_GAME_TYPES: tuple[str, ...] = ("REG", "WC", "DIV", "CON", "SB")

#: The postseason rounds (extra step 6b backfilled exactly these for 2009-2025).
POSTSEASON_GAME_TYPES: tuple[str, ...] = ("WC", "DIV", "CON", "SB")

# Columns retained in the silver injuries table (alongside the derived game_id).
# date_modified is kept when present (2009-2024) as each row's per-row information
# time; it is absent in the 2025 schema and is carried as null then (schema tolerance).
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


def _load_season(season: int) -> pd.DataFrame:
    """nflreadpy's injury reports for *season*, polars -> pandas at the boundary (Pitfall 1)."""
    return nfl.load_injuries(season).to_pandas()


def _build_game_id_lookup(
    season: int, games: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Build a (season, week, team) -> game_id lookup from the silver games table.

    Teams in the silver games table are already canonical; each team plays at
    most one game per week, so the lookup is unique per (season, week, team).

    Args:
        season: The season to look up.
        games: The silver games frame (the test seam); loaded from silver when ``None``.
    """
    if games is None:
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


def ingest_injuries_season(
    season: int,
    weeks: list[int] | None = None,
    *,
    loader: Callable[[int], pd.DataFrame] = _load_season,
    stamp_reader: Callable[..., datetime] | None = None,
    games: pd.DataFrame | None = None,
    base_path: Path | None = None,
    game_types: Sequence[str] = INJURY_GAME_TYPES,
) -> pd.DataFrame:
    """Ingest injury reports for one season into Bronze + Silver.

    Args:
        season: NFL season year (>= 2009).
        weeks: Optional list of weeks to restrict to. None ingests the whole
            season.
        loader: Downloads one season (the test seam; defaults to nflreadpy).
        stamp_reader: ``data.upstream_asset_stamp.asset_published_at``'s shape (the test
            seam).
        games: The silver games frame for the game_id join (the test seam).
        base_path: Data root (the test seam; defaults to the configured root).
        game_types: The game types to keep (default: all of ``INJURY_GAME_TYPES``). The
            step-6b backfill passes ``POSTSEASON_GAME_TYPES`` so it adds postseason reports
            without re-writing a single stored regular-season row.

    Returns:
        The VALIDATED game-grain injury DataFrame upserted to silver (may be empty).

    Raises:
        data.upstream_asset_stamp.UpstreamStampUnavailable: the asset's publication time
            could not be read, or the asset changed during the download. Nothing is written.
        utils.exceptions.DataValidationError: any row failed ``InjurySchema``, or upstream
            used a game type outside ``INJURY_GAME_TYPES``; nothing reaches silver.
    """
    unknown_requested = sorted(set(game_types) - set(INJURY_GAME_TYPES))
    if unknown_requested:
        msg = f"unknown injury game type(s) requested: {unknown_requested}"
        raise DataValidationError(msg)
    if season < INJURY_COVERAGE_FLOOR:
        logger.warning(
            "Season below injury coverage floor; skipping",
            season=season,
            floor=INJURY_COVERAGE_FLOOR,
        )
        return pd.DataFrame()

    asset = season_asset_name(INJURY_DATASET, season)
    raw, stamp = fetch_with_stamp(
        asset, lambda: loader(season), stamp_reader=stamp_reader
    )

    if raw.empty:
        logger.warning("No injury data returned", season=season)
        return pd.DataFrame()

    # EVERY game type upstream files, regular season AND postseason (extra step 6b): a
    # postseason game's reports are timed per row exactly like a regular-season game's, so
    # dropping them only turned real, dated reports into "no report admitted". An unknown code
    # is refused by name -- a vocabulary change upstream must not silently lose rows.
    unknown = sorted(set(raw["game_type"].dropna()) - set(INJURY_GAME_TYPES))
    if unknown:
        msg = (
            f"injuries {season} carries game type(s) {unknown} outside {INJURY_GAME_TYPES}; "
            "refusing rather than silently dropping those reports"
        )
        raise DataValidationError(msg)
    df = raw[raw["game_type"].isin(list(game_types))].copy()

    if weeks is not None:
        df = df[df["week"].isin(weeks)].copy()

    if df.empty:
        logger.warning(
            "No injury rows after filtering",
            season=season,
            weeks=weeks,
            game_types=list(game_types),
        )
        return pd.DataFrame()

    # Schema tolerance (Pitfall 4): 2025 dropped date_modified. Carry it as a null column
    # so the silver schema is stable. Never KeyError on the missing column, and never
    # fabricate a value: a null here means "upstream published no per-row time".
    if "date_modified" not in df.columns:
        logger.warning(
            "date_modified absent (2025+ schema); carrying as null -- the capture stamp "
            "is the only time these rows can carry",
            season=season,
        )
        df["date_modified"] = pd.NaT

    # Canonical team normalization, hard-fail on unknowns.
    df["team"] = df["team"].apply(normalize_team_abbreviation)

    # Retain the relevant columns at (gsis_id, season, week) grain.
    df = df[INJURY_COLUMNS].copy()

    # Derive game_id by joining the season's silver games on (season, week, team).
    lookup = _build_game_id_lookup(season, games)
    pre_join_rows = len(df)
    df = df.merge(lookup, on=["season", "week", "team"], how="inner")

    if df.empty:
        logger.warning(
            "No injury rows matched a silver game on (season, week, team)",
            season=season,
        )
        return pd.DataFrame()
    unmatched = pre_join_rows - len(df)
    if unmatched:
        # Reported, not hidden: a report whose (season, week, team) names no silver game
        # cannot be attached to one and is not stored.
        logger.warning(
            "Injury rows matched no silver game and were not stored",
            season=season,
            unmatched_rows=unmatched,
        )

    # THE CAPTURE STAMP IS PROVENANCE, AND AN INFORMATION TIME ONLY BEFORE THE LOCK (Plan
    # 33.2-15, reviews round f924749). It is a true fact about the file fetched NOW: upstream
    # published it at this instant. It is NOT the time any row of a season-wide file first
    # became known. features.injury admits a row on this stamp only where the stamp is AT OR
    # BEFORE that row's game's lock -- the forward daily-capture case. Upstream dropped
    # date_modified from 2025 and the branch above fills it with pd.NaT, so no per-row
    # historical time is recoverable from anything this ingest can see; presenting a current
    # stamp as historical would make every 2025 value post-lock. A backfill of a completed
    # season therefore admits NOTHING on this basis, and its games stay the honest unknown.
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
        "injuries",
        season=season,
        week=bronze_week,
        base_path=base_path,
        exclusive=True,
    )

    # THE PROMOTION GATE (Plan 33.2-15, adjudication #2). Every row is rebuilt through
    # InjurySchema; one bad row fails the batch and nothing reaches silver. The two time
    # columns come back as aware datetimes (or None) and are stored as UTC datetime64, never
    # as the strings ParquetManager would write for an object column.
    validated = validate_bronze_to_silver(df, InjurySchema)
    for column in ("date_modified", CAPTURE_COLUMN):
        validated[column] = pd.to_datetime(validated[column], utc=True)

    # Silver: ACCUMULATE this capture beside every earlier one (33.2 review C1 CR-04). A
    # latest-wins write by game_id replaced each played game's pre-lock capture with this
    # post-lock one; an identical re-capture (same stamp) is still idempotent.
    upsert_silver_composite(
        validated,
        "injuries",
        key_columns=list(INJURY_CAPTURE_KEY),
        base_path=base_path,
    )

    log_data_operation(
        operation="ingest",
        table="injuries",
        rows=len(validated),
        season=season,
    )

    logger.info(
        "Injury ingestion completed",
        season=season,
        rows=len(validated),
        unique_games=validated["game_id"].nunique(),
        upstream_captured_at=stamp.isoformat(),
    )

    return validated


def main() -> None:
    """CLI entry point for injury ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest NFL injury reports (Bronze + Silver)"
    )
    parser = add_standard_ingestion_args(parser)
    parser.add_argument(
        "--game-types",
        nargs="+",
        choices=INJURY_GAME_TYPES,
        default=list(INJURY_GAME_TYPES),
        help=(
            "Game types to keep (default: all). --game-types WC DIV CON SB adds postseason "
            "reports without re-writing stored regular-season rows."
        ),
    )
    args = parser.parse_args()

    from utils import setup_logging

    setup_logging()

    seasons, weeks = parse_season_week_args(args)

    total_rows = 0
    for season in seasons:
        df = ingest_injuries_season(season, weeks=weeks, game_types=args.game_types)
        total_rows += len(df)
        stamp = df[CAPTURE_COLUMN].iloc[0].isoformat() if len(df) else None
        print(f"SEASON= {season} ROWS_WRITTEN= {len(df)} STAMP= {stamp}")

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
