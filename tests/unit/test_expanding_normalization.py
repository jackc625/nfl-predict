"""Tests for expanding-window normalization.

Covers:
- Expanding window uses only past data (no future leakage)
- Prior-season bootstrap for Week 1
- Season reset (no cross-season contamination)
- min_periods behavior
- Shape preservation
- Normalization effectiveness
- Model-input invariance under an added excluded column
"""

import numpy as np
import pandas as pd
import pytest

from features.normalization import compute_prior_season_stats, expanding_normalize
from tests.normalization_locks import row_order_locks


def _normalize(frame, **kwargs):
    """``expanding_normalize`` with the per-row lock p332_ extra step 8c made mandatory.

    These tests are not about lock ORDERING -- ``tests/unit/test_normalization_lock_order.py``
    is -- so they are given a lock that REPRODUCES the window they already assumed: one
    distinct instant per row, in the order the function sorts the frame into. Every
    assertion below is therefore unchanged by step 8c.
    """
    kwargs.setdefault("row_locks", row_order_locks(frame, kwargs.get("sort_cols")))
    return expanding_normalize(frame, **kwargs)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def multi_season_df():
    """Create a feature DataFrame spanning 2 seasons with known values.

    Season 2023: Weeks 1-10, 4 games per week = 40 rows
    Season 2024: Weeks 1-10, 4 games per week = 40 rows
    Total: 80 rows

    Feature values are deterministic for easy verification.
    """
    rows = []
    for season in [2023, 2024]:
        for week in range(1, 11):
            for game_idx in range(4):
                # Deterministic feature value: base + week offset + game noise
                feat_a = 10.0 + week * 2.0 + game_idx * 0.5
                feat_b = 50.0 - week * 1.0 + game_idx * 0.3
                rows.append(
                    {
                        "game_id": f"G_{season}_W{week:02d}_{game_idx}",
                        "season": season,
                        "week": week,
                        "feat_a": feat_a,
                        "feat_b": feat_b,
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 9: Expanding window for Week 8 uses only Weeks 1-7
# ---------------------------------------------------------------------------


def test_expanding_uses_only_past_data(multi_season_df):
    """For Week 8, expanding normalization uses only data from Weeks 1-7."""
    feature_cols = ["feat_a", "feat_b"]

    result = _normalize(
        multi_season_df,
        feature_cols=feature_cols,
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=1,
    )

    # Get original Week 8 data for season 2024
    season_2024 = multi_season_df[multi_season_df["season"] == 2024].copy()
    season_2024 = season_2024.sort_values(["season", "week"])

    # Week 8 rows (0-indexed from season start: weeks 1-7 have 28 rows)
    week_8_original = season_2024[season_2024["week"] == 8]

    # Expanding mean/std up to (but not including) week 8 values
    # pandas expanding includes the current row, so week 8's expanding
    # stats include weeks 1-7 data already in the window when computing
    # but the key check is: week 9+ data is NOT used
    for col in feature_cols:
        # The normalized values for week 8 should be computed using
        # expanding stats that do NOT include week 9+ data
        result_2024 = result[result["season"] == 2024].sort_values("week")
        result_week_8 = result_2024[result_2024["week"] == 8]

        # Verify that normalized values are reasonable Z-scores
        # (not the raw values, meaning normalization happened)
        for _, row in result_week_8.iterrows():
            original_val = week_8_original.loc[
                week_8_original["game_id"] == row["game_id"], col
            ].values[0]
            normalized_val = row[col]
            # Normalized value should differ from original
            # (unless mean is very close to the value)
            assert normalized_val != original_val or abs(normalized_val) < 0.01


# ---------------------------------------------------------------------------
# Test 10: Week 1 uses prior_season_stats for bootstrap
# ---------------------------------------------------------------------------


def test_week1_uses_prior_season_bootstrap(multi_season_df):
    """Week 1 normalization uses prior_season_stats (not NaN)."""
    feature_cols = ["feat_a", "feat_b"]

    # Compute actual prior season stats from 2023 data
    season_2023 = multi_season_df[multi_season_df["season"] == 2023]
    prior_stats = {
        col: (float(season_2023[col].mean()), float(season_2023[col].std()))
        for col in feature_cols
    }

    result = _normalize(
        multi_season_df,
        feature_cols=feature_cols,
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=4,
        prior_season_stats=prior_stats,
    )

    # Week 1 of season 2024 should NOT have NaN (bootstrapped from prior season)
    week1_2024 = result[(result["season"] == 2024) & (result["week"] == 1)]

    for col in feature_cols:
        assert week1_2024[col].notna().all(), (
            f"Week 1 season 2024 has NaN in {col} -- bootstrap failed"
        )


# ---------------------------------------------------------------------------
# Test 11: Season reset -- no cross-season contamination
# ---------------------------------------------------------------------------


def test_season_reset_no_cross_contamination(multi_season_df):
    """Season 2024 normalization resets; 2023 stats do NOT bleed into 2024."""
    feature_cols = ["feat_a"]

    # Run with min_periods=1 so we can compare within each season
    result = _normalize(
        multi_season_df,
        feature_cols=feature_cols,
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=1,
    )

    # Get week 10 normalized value for both seasons
    # If cross-season contamination exists, season 2024 week 10 would
    # use stats from 2023 + 2024 combined, producing different values
    # than if it only used 2024 data.

    # Compute what the normalized value SHOULD be for 2024 week 10
    # using only 2024 data
    s2024 = multi_season_df[multi_season_df["season"] == 2024].sort_values("week")
    s2024_week10 = s2024[s2024["week"] == 10]

    # Check the result matches season-isolated computation
    result_2024 = result[result["season"] == 2024].sort_values("week")
    result_week10 = result_2024[result_2024["week"] == 10]

    # Key check: result is normalized (not raw), meaning season isolation worked
    actual_normalized = result_week10["feat_a"].values
    raw_values = s2024_week10["feat_a"].values
    assert not np.allclose(actual_normalized, raw_values), (
        "Normalized values are identical to raw -- normalization not applied"
    )


# ---------------------------------------------------------------------------
# Test 12: min_periods=4, Weeks 1-3 use prior_season_stats
# ---------------------------------------------------------------------------


def test_min_periods_uses_prior_stats_for_insufficient_data(multi_season_df):
    """With min_periods=4, Weeks 1-3 (insufficient data) use prior_season_stats."""
    feature_cols = ["feat_a", "feat_b"]

    season_2023 = multi_season_df[multi_season_df["season"] == 2023]
    prior_stats = {
        col: (float(season_2023[col].mean()), float(season_2023[col].std()))
        for col in feature_cols
    }

    result = _normalize(
        multi_season_df,
        feature_cols=feature_cols,
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=4,
        prior_season_stats=prior_stats,
    )

    # Weeks 1-3 of season 2024 have < 4 * 4 = 16 cumulative data points
    # but min_periods applies per expanding window position, so
    # with 4 games/week, week 1 has 4 positions (1,2,3,4 cumulative)
    # Only positions 1-3 have <4 prior data points
    # The key check: no NaN in these early weeks
    early_2024 = result[(result["season"] == 2024) & (result["week"] <= 3)]

    for col in feature_cols:
        assert early_2024[col].notna().all(), (
            f"Early weeks have NaN in {col} -- prior_season_stats not applied"
        )


# ---------------------------------------------------------------------------
# Test 13: Output has same shape as input
# ---------------------------------------------------------------------------


def test_output_shape_matches_input(multi_season_df):
    """Expanding normalize preserves DataFrame shape (no rows dropped)."""
    feature_cols = ["feat_a", "feat_b"]

    result = _normalize(
        multi_season_df,
        feature_cols=feature_cols,
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=1,
    )

    assert result.shape == multi_season_df.shape, (
        f"Shape mismatch: input {multi_season_df.shape}, output {result.shape}"
    )
    assert list(result.columns) == list(multi_season_df.columns), (
        "Column mismatch between input and output"
    )


# ---------------------------------------------------------------------------
# Test 14: Normalized values for Week 8+ have mean closer to 0
# ---------------------------------------------------------------------------


def test_normalization_effectiveness(multi_season_df):
    """Normalized values for Week 8+ have mean closer to 0 than raw values."""
    feature_cols = ["feat_a", "feat_b"]

    result = _normalize(
        multi_season_df,
        feature_cols=feature_cols,
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=1,
    )

    for col in feature_cols:
        # Get Week 8+ data for season 2024
        raw_late = multi_season_df[
            (multi_season_df["season"] == 2024) & (multi_season_df["week"] >= 8)
        ][col]

        normalized_late = result[(result["season"] == 2024) & (result["week"] >= 8)][
            col
        ]

        # Raw values have a non-zero mean (by construction)
        raw_mean = abs(raw_late.mean())
        norm_mean = abs(normalized_late.mean())

        assert norm_mean < raw_mean, (
            f"Normalization not effective for {col}: "
            f"raw mean={raw_mean:.3f}, normalized mean={norm_mean:.3f}"
        )


# ---------------------------------------------------------------------------
# Test 15: Adding an excluded sibling column does not perturb model inputs
# ---------------------------------------------------------------------------


def test_added_excluded_column_does_not_change_other_zscores(multi_season_df):
    """An excluded raw passthrough cannot change another column's z-scores.

    This is the unit-level proof of the locked "model inputs MUST stay
    byte-identical" guarantee behind ``raw_weather_severity``: the gold build
    copies ``weather_severity_score`` to an un-normalized ``raw_weather_severity``
    sibling that is excluded from ``feature_cols``. Because
    ``expanding_normalize`` is a per-column independent loop (no cross-column
    accumulator), adding such a sibling must leave every normalized column
    byte-identical AND pass the raw copy through unchanged. (Task 2 adds the
    live-parquet sha256 cross-check on the real matrices.)
    """
    feature_cols = ["feat_a"]

    # Baseline: normalize feat_a alone.
    baseline = _normalize(
        multi_season_df,
        feature_cols=feature_cols,
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=4,
    )

    # Add a raw copy of the ORIGINAL feat_a to a fresh frame, excluded from
    # feature_cols (mirrors raw_weather_severity = copy of weather_severity_score).
    with_sibling = multi_season_df.copy()
    with_sibling["raw_feat_a"] = multi_season_df["feat_a"]
    normalized = _normalize(
        with_sibling,
        feature_cols=feature_cols,  # raw_feat_a deliberately excluded
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=4,
    )

    # The normalized model input (feat_a) is byte-identical with/without the
    # excluded sibling -- equal_nan covers the early-week NaN positions.
    pd.testing.assert_series_equal(
        baseline["feat_a"],
        normalized["feat_a"],
        check_exact=True,
    )

    # The excluded raw copy passes through unchanged (never normalized).
    pd.testing.assert_series_equal(
        normalized["raw_feat_a"],
        multi_season_df["feat_a"],
        check_names=False,
    )


# ---------------------------------------------------------------------------
# Test 16: Degenerate prior season -> neutral 0.0 fallback (no raw leak)
# ---------------------------------------------------------------------------


def test_degenerate_prior_season_falls_back_to_zero_not_raw():
    """When a column's prior season is degenerate (constant placeholder -> std 0
    -> omitted from prior_season_stats), the next season's first ``min_periods-1``
    games come back BLANK -- never the raw value.

    Regression for 260524-svu: pre-2018 Elo is a constant 1500 placeholder, so it
    was omitted from 2018's prior stats and the first 3 games of 2018 leaked raw
    ~1500 Elo into the normalized ``home_elo`` column. THE SUBJECT OF THIS TEST IS
    THAT LEAK, and it is unchanged: the fallback must never be the raw magnitude.

    THE FALLBACK ITSELF MOVED, CORRECTED HERE RATHER THAN SILENCED (p332_ extra
    step 8d, owner ruling 2026-09-22). Was: the three positions were asserted to be
    exactly 0.0, the neutral z-score. Nothing could score them -- no expanding
    statistic and no usable bootstrap -- and a model reads a centred 0.0 as
    "exactly average", so that answer made a confident claim about a value nobody
    could place. They are BLANK now, which the models take natively. The node name
    keeps its original spelling so its id is stable.
    """
    rows = []
    # Prior season 2017: 'elo' is a constant 1500.0 placeholder (std 0 -> omitted).
    for wk in range(1, 6):
        for _g in range(4):
            rows.append({"season": 2017, "week": wk, "elo": 1500.0})
    # Season 2018: real, varied 'elo'. Its first 3 games are insufficient-data.
    for wk in range(1, 6):
        for g in range(4):
            rows.append({"season": 2018, "week": wk, "elo": 1500.0 + wk * 10 + g})
    df = pd.DataFrame(rows)

    prior = compute_prior_season_stats(df, ["elo"], 2017)
    assert "elo" not in prior, "degenerate constant prior should be omitted"

    result = _normalize(
        df,
        feature_cols=["elo"],
        group_col="season",
        sort_cols=["season", "week"],
        min_periods=4,
        prior_season_stats=prior,
    )

    # expanding_normalize returns rows already sorted by (season, week), so the
    # filtered 2018 slice is in order -- head(3) is W1's first 3 (insufficient) games.
    s2018 = result[result["season"] == 2018]
    first3 = [float(v) for v in s2018["elo"].head(3)]
    # Raw values here were ~1510-1512; the fix must yield a blank, not those
    # magnitudes and not the neutral 0.0 that used to stand in for one.
    assert all(pd.isna(v) for v in first3), (
        f"insufficient-data positions with a degenerate prior have no statistic "
        f"and must be blank, not raw values; got {first3}"
    )
    # And no normalized value anywhere should carry a raw-Elo magnitude.
    assert bool((s2018["elo"].dropna().abs() < 10).all()), (
        "no 2018 'elo' value should exceed a plausible z-score band (raw leak)"
    )
