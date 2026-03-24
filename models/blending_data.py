"""Pre-2018 odds data ingestion from nflverse for blend weight tuning.

Provides:
- TUNING_SEASONS: Default 2010-2017 seasons used for pre-backtest tuning
- load_tuning_period_data: Load schedule + odds data from nflverse
- get_tuning_period_games: Load game info columns only (no odds)

The 2010-2017 period provides historical odds data that predates our
backtest holdout (2021-2024) and training window (2018-2020), making it
safe for tuning blend weights without leaking into model evaluation.

nflreadpy returns polars DataFrames, so we filter/select with polars
operations before converting to pandas for compatibility with the rest
of the codebase.
"""

from __future__ import annotations

import pandas as pd

try:
    import nflreadpy
except ImportError as exc:
    msg = (
        "nflreadpy is required for pre-2018 data loading. "
        "Install it with: uv add nflreadpy"
    )
    raise ImportError(msg) from exc

from utils import get_logger
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TUNING_SEASONS: list[int] = list(range(2010, 2018))
"""Default seasons for blend weight tuning (2010-2017)."""

# Column mapping from nflverse schedule to our Silver-layer-compatible schema
_NFLVERSE_TO_SILVER = {
    "spread_line": "spread",
    "total_line": "total",
    "home_moneyline": "ml_home",
    "away_moneyline": "ml_away",
}

# Columns to keep from nflverse schedule data
_KEEP_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "spread_line",
    "total_line",
    "home_moneyline",
    "away_moneyline",
]

# Game info columns (no odds data)
_GAME_INFO_COLUMNS = [
    "game_id",
    "season",
    "week",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
]

# Required odds columns that must not be NaN
_REQUIRED_ODDS_COLUMNS = ["spread", "total", "ml_home", "ml_away"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_tuning_period_data(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Load pre-2018 schedule and odds data from nflverse.

    Fetches game schedules via nflreadpy.load_schedules, filters to regular
    season games, maps columns to Silver-layer-compatible names, normalizes
    team abbreviations, and drops rows with missing odds data.

    Args:
        seasons: List of seasons to load. Defaults to TUNING_SEASONS (2010-2017).

    Returns:
        Pandas DataFrame with columns: game_id, season, week, home_team,
        away_team, home_score, away_score, spread, total, ml_home, ml_away.
    """
    if seasons is None:
        seasons = TUNING_SEASONS

    logger.info("Loading tuning period data from nflverse", seasons=seasons)

    # nflreadpy returns a polars DataFrame
    schedule_pl = nflreadpy.load_schedules(seasons=seasons)

    # Filter to regular season only
    schedule_pl = schedule_pl.filter(schedule_pl["game_type"] == "REG")

    # Select only the columns we need
    schedule_pl = schedule_pl.select(_KEEP_COLUMNS)

    # Convert to pandas for downstream compatibility
    df = schedule_pl.to_pandas()

    # Rename columns to Silver-layer schema
    df = df.rename(columns=_NFLVERSE_TO_SILVER)

    # Normalize team abbreviations
    df["home_team"] = df["home_team"].apply(normalize_team_abbreviation)
    df["away_team"] = df["away_team"].apply(normalize_team_abbreviation)

    # Track how many rows had missing odds before dropping
    n_before = len(df)
    df = df.dropna(subset=_REQUIRED_ODDS_COLUMNS)
    n_dropped = n_before - len(df)

    df = df.reset_index(drop=True)

    logger.info(
        "Tuning period data loaded",
        n_games=len(df),
        seasons=sorted(df["season"].unique().tolist()) if len(df) > 0 else [],
        n_dropped_missing_odds=n_dropped,
    )

    return df


def get_tuning_period_games(
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    """Load game info columns only (no odds data) for tuning period.

    Useful for constructing features/targets for the mini walk-forward
    without loading full odds data.

    Args:
        seasons: List of seasons to load. Defaults to TUNING_SEASONS (2010-2017).

    Returns:
        Pandas DataFrame with columns: game_id, season, week, home_team,
        away_team, home_score, away_score.
    """
    full_df = load_tuning_period_data(seasons=seasons)
    return full_df[_GAME_INFO_COLUMNS].copy()
