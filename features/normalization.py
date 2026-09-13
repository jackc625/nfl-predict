"""Expanding-window normalization for feature matrices.

Replaces the within-season Z-score normalization (which leaks future data)
with an expanding-window approach that only uses data available up to the
current game-week.

Key behaviors:
- Each season resets independently (no cross-season contamination)
- Week 1 uses prior_season_stats for bootstrap (no NaN)
- Uses pandas expanding() for correct cumulative statistics
- Preserves DataFrame shape (no rows dropped)
"""

import sys
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import get_logger

logger = get_logger(__name__)


def expanding_normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "season",
    sort_cols: list[str] | None = None,
    min_periods: int = 4,
    prior_season_stats: dict[str, tuple[float, float]] | None = None,
    preserve_missing_cols: Sequence[str] = (),
) -> pd.DataFrame:
    """Normalize features using expanding window within each season.

    For early weeks where the expanding window has fewer than min_periods
    data points, uses prior_season_stats (mean, std) as bootstrap values.
    If prior_season_stats is not provided or a column is not in it (e.g. the
    first data-bearing season, or a season whose prior season is degenerate /
    all-placeholder), those positions fall back to 0.0 -- the neutral z-score --
    NOT the raw value (which would leak an un-normalized magnitude, e.g. a raw
    ~1500 Elo, into the normalized column) and NOT NaN.

    Resets each season to avoid cross-season distribution contamination.

    Args:
        df: DataFrame with features to normalize.
        feature_cols: List of numeric column names to normalize.
        group_col: Column to group by for season isolation.
        sort_cols: Columns to sort by within each group. Defaults to
            [group_col, "week"].
        min_periods: Minimum number of data points required for the
            expanding window to compute std. Below this, prior_season_stats
            are used as fallback.
        prior_season_stats: Dict mapping column name to (mean, std) tuple
            from the prior season. Used as bootstrap for early weeks.
        preserve_missing_cols: Columns whose INPUT NaNs must come back out as
            NaN instead of the neutral 0.0 z-score. Defaults to empty, so every
            existing caller is byte-preserved.

            TWO CAUSES WERE COLLAPSED INTO ONE FILL, and this separates them.
            The ``fillna(0.0)`` below exists for a position whose STATISTIC was
            unavailable -- an early week with fewer than ``min_periods`` points
            and no prior-season bootstrap. Applied to a position whose VALUE was
            absent it fabricates a neutral reading for a measurement that does
            not exist. A weather observation that never arrived is the second
            case, not the first (SPEC R5, D33.1-07), and only a caller that
            NAMES a column gets the second treatment for it.

    Returns:
        DataFrame with normalized values replacing raw values in
        feature_cols. Non-feature columns are preserved unchanged.
    """
    if sort_cols is None:
        sort_cols = [group_col, "week"]

    preserve_missing = set(preserve_missing_cols)

    result = df.sort_values(sort_cols).copy()

    for season, _season_group in result.groupby(group_col):
        season_mask = result[group_col] == season
        season_idx = result.loc[season_mask].index

        for col in feature_cols:
            if col not in result.columns:
                continue

            values = result.loc[season_idx, col].copy()

            # The INPUT absence mask, captured BEFORE anything is computed.
            # It has to be taken here rather than derived afterwards: by the
            # time the fill below runs, a position that was absent and a
            # position whose statistic was unavailable are both simply NaN.
            absent_mask = values.isna() if col in preserve_missing else None

            # Compute expanding mean and std (only uses data up to current row)
            exp_mean = values.expanding(min_periods=min_periods).mean()
            exp_std = values.expanding(min_periods=min_periods).std()

            # Identify positions where expanding window has insufficient data
            insufficient_mask = exp_std.isna() | (exp_std < 1e-8)

            # Apply prior season stats as bootstrap for insufficient positions
            if prior_season_stats and col in prior_season_stats:
                prior_mean, prior_std = prior_season_stats[col]
                exp_mean = exp_mean.fillna(prior_mean)
                exp_std = exp_std.where(~insufficient_mask, prior_std)

            # Z-score normalize: (value - expanding_mean) / expanding_std
            safe_std = exp_std.clip(lower=1e-8)
            normalized = (values - exp_mean) / safe_std

            # For positions still without valid stats -- insufficient expanding
            # data AND no usable prior_season_stats for this column (the first
            # data-bearing season, or a season whose prior season is degenerate /
            # all-placeholder) -- fall back to 0.0, the neutral z-score. Returning
            # the RAW value here would leak an un-normalized magnitude (e.g. a raw
            # ~1500 Elo) into the normalized column and corrupt the model feature.
            still_missing = normalized.isna()
            if still_missing.any():
                normalized = normalized.fillna(0.0)

            # ...EXCEPT where the input VALUE was absent rather than its
            # statistic. The fill above answers "this column could not be
            # normalized here"; it must not also answer "this measurement does
            # not exist". SPEC prohibition 1: replacing an absent observation
            # with any number, under any name, is the same defect wearing a
            # better label -- and a neutral z-score is a number.
            if absent_mask is not None and absent_mask.any():
                normalized = normalized.mask(absent_mask)

            result.loc[season_idx, col] = normalized

    logger.info(
        "Expanding normalization complete",
        features=len(feature_cols),
        groups=result[group_col].nunique(),
        min_periods=min_periods,
        has_prior_stats=prior_season_stats is not None,
    )

    return result


def compute_prior_season_stats(
    df: pd.DataFrame,
    feature_cols: list[str],
    season: int,
    group_col: str = "season",
) -> dict[str, tuple[float, float]]:
    """Compute mean and std for each feature column from a specific season.

    Used to provide prior_season_stats for expanding_normalize bootstrap.

    Args:
        df: DataFrame containing historical data.
        feature_cols: List of feature columns to compute stats for.
        season: The season to compute stats from (e.g., 2023 for
            bootstrapping 2024 Week 1).
        group_col: Column containing season values.

    Returns:
        Dict mapping column name to (mean, std) tuple. Columns with
        insufficient data (all NaN) are omitted.
    """
    season_data = df[df[group_col] == season]

    if len(season_data) == 0:
        logger.warning("No data for prior season", season=season)
        return {}

    stats: dict[str, tuple[float, float]] = {}
    for col in feature_cols:
        if col not in season_data.columns:
            continue

        col_data = season_data[col].dropna()
        if len(col_data) < 2:
            continue

        col_mean = float(col_data.mean())
        col_std = float(col_data.std())

        if col_std > 1e-8:
            stats[col] = (col_mean, col_std)

    logger.info(
        "Computed prior season stats",
        season=season,
        features_with_stats=len(stats),
    )

    return stats
