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

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import get_logger

logger = get_logger(__name__)


class LockOrderedStatisticUnavailableError(ValueError):
    """A caller could not supply the per-row LOCK the statistics are ordered by.

    p332_ extra step 8c (owner ruling 2026-09-22, "Order by lock time"). A row's
    expanding mean and standard deviation are computed over its season's rows whose
    LOCK is at or before that row's lock, so the lock is not optional decoration: it
    IS the window. Falling back to week order would silently restore the leak this
    step removed -- a same-week game whose lock is later entering an earlier-locking
    game's statistic -- so a caller that cannot supply one is refused BY NAME.

    A ``ValueError`` so a caller that already handles a bad frame handles this one;
    the distinct type is what lets a test assert WHICH refusal it got.
    """


def _lock_order(locks: pd.Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(order, inverse, last_of_tie)`` for one season's *locks*.

    ``order`` sorts the season's rows by lock (a STABLE sort, so equal locks keep their
    arrival order and the result is deterministic); ``inverse`` maps a sorted position
    back to its row; ``last_of_tie`` gives, for each sorted position, the LAST sorted
    position carrying the same lock. Every row of a tie group therefore reads the
    expanding statistic taken at the END of its group -- which is what "at-lock
    information is admissible, so games sharing a lock all see each other" means.
    """
    values = locks.to_numpy()
    order = np.argsort(values, kind="stable")
    inverse = np.empty(len(order), dtype=np.int64)
    inverse[order] = np.arange(len(order), dtype=np.int64)
    sorted_locks = values[order]
    starts = np.empty(len(order), dtype=bool)
    starts[0] = True
    starts[1:] = sorted_locks[1:] != sorted_locks[:-1]
    group_start_positions = np.flatnonzero(starts)
    group_end_positions = np.append(group_start_positions[1:], len(order)) - 1
    group_id = np.cumsum(starts) - 1
    return order, inverse, group_end_positions[group_id]


def expanding_normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_col: str = "season",
    sort_cols: list[str] | None = None,
    min_periods: int = 4,
    prior_season_stats: dict[str, tuple[float, float]] | None = None,
    preserve_missing_cols: Sequence[str] = (),
    preserve_level_cols: Sequence[str] = (),
    row_locks: pd.Series | None = None,
) -> pd.DataFrame:
    """Normalize features using expanding window within each season.

    For early weeks where the expanding window has fewer than min_periods
    data points, uses prior_season_stats (mean, std) as bootstrap values.
    If prior_season_stats is not provided or a column is not in it (e.g. the
    first data-bearing season, or a season whose prior season is degenerate /
    all-placeholder), a position whose VALUE is present comes back BLANK (NaN):
    nothing could score it, and the neutral 0.0 that used to be written there
    read as "exactly average" about a value nobody could place (p332_ extra step
    8d, owner ruling 2026-09-22). A position whose value was ABSENT keeps its
    existing treatment -- the neutral 0.0 unless its column is named in
    preserve_missing_cols. The raw value is never returned in either case: it
    would leak an un-normalized magnitude, e.g. a raw ~1500 Elo, into the
    normalized column.

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
            The terminal fill below exists for a position whose STATISTIC was
            unavailable -- an early week with fewer than ``min_periods`` points
            and no prior-season bootstrap. Applied to a position whose VALUE was
            absent it fabricates a neutral reading for a measurement that does
            not exist. A weather observation that never arrived is the second
            case, not the first (SPEC R5, D33.1-07), and only a caller that
            NAMES a column gets the second treatment for it.

            SINCE p332_ EXTRA STEP 8d the FIRST case is blank too, so the two
            causes now produce the same reading for opposite reasons -- and the
            separation still matters, because a column NOT named here keeps the
            neutral 0.0 for an absent value. Naming a column is still the only
            way to say "an absent measurement stays absent".
        preserve_level_cols: Columns returned at their RECORDED LEVEL rather
            than z-scored. Defaults to empty, so every existing caller is
            byte-preserved.

            FOR A COLUMN WHOSE LEVELS ARE ITS MEANING, and for no other kind
            (Plan 33.1-07 Task 4). CR-02 already makes this argument one stage
            earlier, about winsorization: "a DISCRETE INDICATOR has no outliers
            to winsorize, and clipping one destroys the distinction it exists to
            encode". A z-score destroys it too, and in the degenerate case it
            destroys it completely -- the expanding std of a CONSTANT column is
            zero, ``safe_std`` clips to 1e-8, and every row comes back 0.0.

            That is not hypothetical. ``weather_coverage`` reached gold as a
            constant 0.0 on all 6,499 rows while silver carried a constant 1.0,
            so the column that exists to distinguish "no observation" from "mild
            weather" recorded NO OBSERVATION for 6,499 games that all had one.

            NOT A GENERAL EXEMPTION FOR INDICATORS. A binary flag that VARIES
            still carries its distinction through a monotone transform, and
            z-scoring it is this pipeline's convention. Only a caller that NAMES
            a column opts it out, and the set is expected to stay very small.
        row_locks: Each row's game LOCK, as an orderable per-row value indexed like
            *df* -- in production, UTC nanoseconds from ``utils.game_lock`` through
            ``features.point_in_time_fill``. REQUIRED.

            p332_ EXTRA STEP 8c (owner ruling 2026-09-22, "Order by lock time"). A
            row's expanding mean and standard deviation are computed over every row
            of its season whose lock is AT OR BEFORE that row's lock (D33.2-01: at
            the lock is admissible), so games sharing one lock -- a whole Sunday
            slate under its Saturday 18:00 ET lock -- all see each other and share
            ONE statistic.

            Was: the rows were ordered by ``sort_cols`` (``(season, week)``) and the
            window expanded over THAT order. Within a week the sort is unstable and
            has no tie-break, so a same-week game whose lock is LATER entered an
            earlier-locking game's statistic -- a Monday night game's day-before
            forecast, issued on the Sunday, reaching a Sunday game locked on the
            Saturday -- and two games played the same afternoon got different
            statistics depending on the arbitrary order their rows arrived in.

            NOTHING ELSE CHANGES. The row set and the row ORDER of the returned
            frame, and the meanings of ``min_periods``, the prior-season bootstrap,
            the neutral-0.0 fallback, ``preserve_missing_cols`` and
            ``preserve_level_cols``, are exactly as they were. Only WHICH rows form
            each statistic moved. ``sort_cols`` still orders the RETURNED frame.

            A caller with no lock is refused by name
            (``LockOrderedStatisticUnavailableError``): week order is not a safe
            fallback, it is the leak.

    Returns:
        DataFrame with normalized values replacing raw values in
        feature_cols. Non-feature columns are preserved unchanged.

    Raises:
        LockOrderedStatisticUnavailableError: when *row_locks* is absent, does not
            cover every row of *df*, or carries a null.
    """
    if sort_cols is None:
        sort_cols = [group_col, "week"]

    if row_locks is None:
        msg = (
            "expanding_normalize requires row_locks: each row's game lock, which is "
            "the window its expanding mean and standard deviation are taken over "
            "(p332_ extra step 8c, D33.2-01). Refusing rather than falling back to "
            "week order, which admits a same-week game whose lock is later."
        )
        raise LockOrderedStatisticUnavailableError(msg)

    preserve_missing = set(preserve_missing_cols)
    preserve_level = set(preserve_level_cols)

    result = df.sort_values(sort_cols).copy()

    missing_locks = result.index.difference(row_locks.index)
    if len(missing_locks) > 0:
        msg = (
            f"{len(missing_locks)} row(s) have no lock in row_locks, so their "
            "normalization window cannot be formed. Refusing the whole frame rather "
            "than normalizing them against week order."
        )
        raise LockOrderedStatisticUnavailableError(msg)
    locks = row_locks.reindex(result.index)
    if bool(locks.isna().any()):
        msg = (
            f"{int(locks.isna().sum())} row(s) carry a NULL lock. A game with no lock "
            "has no window; refusing rather than treating it as the earliest or the "
            "latest row of its season."
        )
        raise LockOrderedStatisticUnavailableError(msg)

    for season, _season_group in result.groupby(group_col):
        season_mask = result[group_col] == season
        season_idx = result.loc[season_mask].index
        order, inverse, last_of_tie = _lock_order(locks.loc[season_idx])

        for col in feature_cols:
            if col not in result.columns:
                continue

            values = result.loc[season_idx, col].copy()

            # A LEVEL-PRESERVED COLUMN IS NOT TRANSFORMED AT ALL, and the skip
            # sits here -- before the statistics -- rather than being undone
            # afterwards. Restoring the values after z-scoring would give the
            # same numbers, but it would also leave a reader unable to tell
            # whether the column had been normalized and then repaired. It was
            # never normalized. An absent cell stays absent for free, because
            # nothing touched it.
            if col in preserve_level:
                continue

            # The INPUT absence mask, captured BEFORE anything is computed.
            # It has to be taken here rather than derived afterwards: by the
            # time the fill below runs, a position that was absent and a
            # position whose statistic was unavailable are both simply NaN.
            absent_mask = values.isna() if col in preserve_missing else None

            # Compute expanding mean and std over the LOCK-ORDERED rows, then give
            # every row of a tie group the statistic taken at the END of its group
            # (p332_ extra step 8c). ``last_of_tie`` is what makes at-lock rows see
            # each other; ``inverse`` puts the result back on the frame's own rows,
            # so the returned row order is untouched.
            ordered = pd.Series(values.to_numpy()[order])
            exp_mean = (
                ordered.expanding(min_periods=min_periods)
                .mean()
                .to_numpy()[last_of_tie][inverse]
            )
            exp_std = (
                ordered.expanding(min_periods=min_periods)
                .std()
                .to_numpy()[last_of_tie][inverse]
            )
            exp_mean = pd.Series(exp_mean, index=values.index)
            exp_std = pd.Series(exp_std, index=values.index)

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
            # all-placeholder) -- the cell is left BLANK.
            #
            # P332_ EXTRA STEP 8d (owner ruling 2026-09-22, "LEAVE THE CELL BLANK").
            # This fill used to write 0.0, the neutral z-score, into such a position.
            # A model reads a centred 0.0 as "exactly average", so a value that WAS
            # measured left gold asserting it was perfectly ordinary -- and it read
            # 0.0 WHATEVER the input value, so the number said nothing at all.
            # Measured on step 8c's own gold: 307 cells with a PRESENT value had no
            # statistic, 46 of them in columns whose coverage flag reads TRUE.
            #
            # The ruling covers ALL of them, not only the flagged ones: whether a
            # family happens to declare coverage is not a reason to treat its
            # unscorable cells differently. The models take a blank natively --
            # XGBoost's missing branch, and the WP model's in-fold imputation with
            # its ``_was_missing`` indicator -- which is the reasoning step 7b's
            # blanks already rest on.
            #
            # TWO KINDS OF ZERO STAY DISTINGUISHABLE, which is the whole point. A
            # cell whose statistic WAS formed reads 0.0 only when its value happens
            # to equal the window mean; that is a real reading and is untouched
            # here. Only a position with NO usable statistic AND no usable
            # bootstrap -- ``exp_mean`` or ``exp_std`` still absent after the
            # prior-season fill -- is blanked.
            #
            # A CELL WHOSE INPUT VALUE WAS ABSENT KEEPS ITS EXISTING TREATMENT. The
            # ruling is about a value that EXISTS and has no statistic to be scored
            # against; an absent measurement is the separate question
            # ``preserve_missing_cols`` answers below, and double-handling it here
            # would change a behaviour nobody ruled on. Hence the ``values.notna()``
            # conjunct: it is what keeps the two causes apart, exactly as the
            # ``absent_mask`` captured above does for the other direction.
            #
            # Returning the RAW value was never an option either way: it would leak
            # an un-normalized magnitude (e.g. a raw ~1500 Elo) into the normalized
            # column and corrupt the model feature.
            still_missing = normalized.isna()
            if still_missing.any():
                unscorable = (exp_mean.isna() | exp_std.isna()) & values.notna()
                normalized = normalized.fillna(0.0).mask(unscorable)

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
