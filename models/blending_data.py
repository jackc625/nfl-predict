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

from pathlib import Path

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


# ---------------------------------------------------------------------------
# Noise profile extraction
# ---------------------------------------------------------------------------

MIN_NOISE_SAMPLE_COUNT = 30
"""Minimum games per week for reliable noise statistics.
Below this threshold, per-week stats are blended toward season-wide
stats to prevent overfitting to sparse data (addresses review concern
about thin sample sizes for late-season weeks)."""


def extract_noise_profile(
    baselines_dir: Path | None = None,
    min_sample_count: int = MIN_NOISE_SAMPLE_COUNT,
) -> dict[str, pd.DataFrame]:
    """Extract per-week model error distributions from backtest predictions.

    Reads prediction parquets from the baselines directory, computes
    per-game errors (model value minus closing market value), and returns
    per-week statistics (mean, std, count) for each target.

    Used by dynamic blend tuning to generate realistic synthetic predictions
    for the 2010-2017 tuning period (per D-14, D-15).

    Error definitions (all measure model-vs-closing-market deviation):
    - WP: model_prob - fair_closing_prob
    - ATS: model_spread - spread (closing line)
    - O/U: model_total - total (closing line)

    For weeks with fewer than min_sample_count games, stats are blended
    toward the season-wide mean/std to prevent overfitting to sparse data.

    Args:
        baselines_dir: Path to baselines directory containing prediction parquets.
            Defaults to Path("data/baselines/v2.0"). Configurable for testing
            and for pointing to different baseline versions.
        min_sample_count: Minimum games for reliable per-week stats.
            Below this, stats blend toward season-wide values.

    Returns:
        Dict mapping target ("wp", "ats", "ou") to DataFrame with columns:
        week (int), mean (float), std (float), count (int).

    Raises:
        FileNotFoundError: If baselines_dir does not exist or required
            parquet files are missing (per D-17: fail fast, no fallback).
    """
    if baselines_dir is None:
        baselines_dir = Path("data/baselines/v2.0")

    if not baselines_dir.exists():
        msg = (
            f"Baselines directory not found: {baselines_dir}. "
            "Run backtest first to generate baseline predictions."
        )
        raise FileNotFoundError(msg)

    profiles: dict[str, pd.DataFrame] = {}

    # Target configs: (parquet filename, model column, market column)
    target_configs = {
        "wp": ("predictions_wp.parquet", "model_prob", "fair_closing_prob"),
        "ats": ("predictions_ats.parquet", "model_spread", "spread"),
        "ou": ("predictions_ou.parquet", "model_total", "total"),
    }

    for target, (filename, model_col, market_col) in target_configs.items():
        parquet_path = baselines_dir / filename
        if not parquet_path.exists():
            msg = (
                f"{filename} not found in {baselines_dir}. "
                "Run backtest first to generate baseline predictions."
            )
            raise FileNotFoundError(msg)

        df = pd.read_parquet(parquet_path)

        # Extract week from game_id: {season}_W{week}_{away}@{home}
        # Handles both W01 (zero-padded) and W1 (unpadded) formats
        df["week"] = df["game_id"].str.extract(r"_W(\d+)_")[0].astype(int)
        # Extract season for era-aware max_week filtering
        df["season"] = df["game_id"].str.split("_").str[0].astype(int)

        # Compute error (model-vs-closing-market deviation)
        df["error"] = df[model_col] - df[market_col]

        # Filter out playoff weeks per D-05 era lookup
        max_week = df["season"].apply(lambda s: 17 if s <= 2020 else 18)
        df = df[df["week"] <= max_week]

        # Compute season-wide stats for sparse-week blending
        season_mean = float(df["error"].mean())
        season_std = float(df["error"].std())

        # Aggregate per-week stats
        stats = df.groupby("week")["error"].agg(["mean", "std", "count"]).reset_index()
        stats["count"] = stats["count"].astype(int)

        # Blend sparse weeks toward season-wide stats
        for idx in stats.index:
            count = stats.at[idx, "count"]
            if count < min_sample_count:
                blend_weight = count / min_sample_count
                stats.at[idx, "mean"] = (
                    blend_weight * stats.at[idx, "mean"]
                    + (1 - blend_weight) * season_mean
                )
                stats.at[idx, "std"] = (
                    blend_weight * stats.at[idx, "std"]
                    + (1 - blend_weight) * season_std
                )

        profiles[target] = stats

        logger.info(
            "Noise profile extracted",
            target=target,
            n_weeks=len(stats),
            total_games=int(stats["count"].sum()),
            sparse_weeks=int((stats["count"] < min_sample_count).sum()),
        )

    return profiles
