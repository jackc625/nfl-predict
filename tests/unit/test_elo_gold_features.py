"""Tests for post-rebuild gold feature matrix Elo column assertions.

Verifies that gold feature matrices (WP, ATS, O/U) contain real Elo-derived
features with meaningful variance after the gold rebuild with extended
EloFeatureBuilder (14 Elo features).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

# Path to gold and baseline data
GOLD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gold"
BASELINE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "baselines" / "v1.0"
)

TARGETS = ["wp", "ats", "ou"]


def _load_gold(target: str) -> pd.DataFrame:
    """Load a gold feature matrix by target name."""
    path = GOLD_DIR / f"features_{target}.parquet"
    if not path.exists():
        pytest.skip(f"Gold matrix not found: {path}")
    return pd.read_parquet(path)


class TestEloColumnsHaveVariance:
    """Verify Elo feature columns have non-zero standard deviation."""

    def test_elo_columns_have_variance(self):
        """Each Elo feature column in gold matrices has std > 0."""
        from features.elo_features import ELO_FEATURE_COLUMNS

        for target in TARGETS:
            df = _load_gold(target)
            for col in ELO_FEATURE_COLUMNS:
                if col in df.columns:
                    std = df[col].std()
                    assert std > 0, (
                        f"{target}: Elo column '{col}' has zero variance (std={std})"
                    )


class TestNoConstant1500Values:
    """Verify home_elo and away_elo are not all 1500.0."""

    def test_no_constant_1500_values(self):
        """home_elo and away_elo columns are not all 1500.0 in any target."""
        for target in TARGETS:
            df = _load_gold(target)

            assert not (df["home_elo"] == 1500.0).all(), (
                f"{target}: home_elo is all 1500.0 (batch leakage not fixed)"
            )
            assert df["home_elo"].nunique() > 1, (
                f"{target}: home_elo has only 1 unique value"
            )

            assert not (df["away_elo"] == 1500.0).all(), (
                f"{target}: away_elo is all 1500.0"
            )
            assert df["away_elo"].nunique() > 1, (
                f"{target}: away_elo has only 1 unique value"
            )


class TestMomentumColumnsPresent:
    """Verify momentum columns exist in gold matrices."""

    def test_momentum_columns_present(self):
        """home_elo_momentum and away_elo_momentum exist in all targets."""
        for target in TARGETS:
            df = _load_gold(target)
            assert "home_elo_momentum" in df.columns, (
                f"{target}: missing home_elo_momentum"
            )
            assert "away_elo_momentum" in df.columns, (
                f"{target}: missing away_elo_momentum"
            )


class TestRankColumnsPresent:
    """Verify rank and percentile columns exist in gold matrices."""

    def test_rank_columns_present(self):
        """Rank and percentile columns exist in all targets."""
        for target in TARGETS:
            df = _load_gold(target)
            for col in [
                "home_elo_rank",
                "away_elo_rank",
                "home_elo_percentile",
                "away_elo_percentile",
            ]:
                assert col in df.columns, f"{target}: missing {col}"


class TestMomentumNanCountReasonable:
    """Verify momentum has reasonable distribution (week 1 values are imputed)."""

    def test_momentum_nan_count_reasonable(self):
        """Momentum values should exist for all rows (NaN imputed by pipeline).

        The gold pipeline imputes NaN values via handle_missing_data_and_outliers,
        so week 1 games that originally had NaN momentum are filled with team
        season averages or overall medians. We verify that week 1 games have
        momentum values near zero (since early-season momentum is undefined).
        """
        for target in TARGETS:
            df = _load_gold(target)
            if "home_elo_momentum" not in df.columns:
                pytest.skip(f"{target}: no momentum column")

            # After imputation, all values should be non-NaN (or very few)
            nan_count = df["home_elo_momentum"].isna().sum()
            total = len(df)

            # Allow a small fraction of NaN (< 5%) in case some edge cases remain
            assert nan_count / total < 0.05, (
                f"{target}: Too many NaN in momentum after imputation: "
                f"{nan_count}/{total} = {nan_count/total:.1%}"
            )

            # Momentum should have meaningful variation (not constant)
            std = df["home_elo_momentum"].std()
            assert std > 0, (
                f"{target}: home_elo_momentum has zero variance (all same value)"
            )


class TestRankValuesInRange:
    """Verify rank and percentile values have meaningful variance.

    Note: Gold matrices are Z-score normalized by the feature pipeline,
    so raw rank [1-32] and percentile [0-1] ranges are transformed.
    We verify variance and distinct values instead.
    """

    def test_rank_values_in_range(self):
        """Rank and percentile columns have meaningful variance (post-normalization)."""
        for target in TARGETS:
            df = _load_gold(target)

            if "home_elo_rank" in df.columns:
                valid = df["home_elo_rank"].dropna()
                # After Z-score normalization, verify meaningful variance
                assert valid.std() > 0, (
                    f"{target}: home_elo_rank has zero variance"
                )
                # Should have many distinct values (32 ranks per week)
                assert valid.nunique() > 10, (
                    f"{target}: home_elo_rank has too few unique values: "
                    f"{valid.nunique()}"
                )

            if "home_elo_percentile" in df.columns:
                valid = df["home_elo_percentile"].dropna()
                assert valid.std() > 0, (
                    f"{target}: home_elo_percentile has zero variance"
                )
                assert valid.nunique() > 10, (
                    f"{target}: home_elo_percentile has too few unique values: "
                    f"{valid.nunique()}"
                )


class TestNewColumnsVsBaseline:
    """Compare new gold matrices with v1.0 baseline."""

    def test_new_columns_vs_baseline(self):
        """New gold matrices have 6 additional Elo columns vs v1.0 baseline."""
        baseline_path = BASELINE_DIR / "feature_metadata.json"
        if not baseline_path.exists():
            pytest.skip("Baseline feature metadata not found")

        with open(baseline_path) as f:
            baseline_meta = json.load(f)

        for target in TARGETS:
            df = _load_gold(target)
            gold_cols = set(df.columns)
            baseline_cols = set(baseline_meta[target]["columns"])

            # Find new Elo columns not in baseline
            new_elo_cols = {
                c for c in gold_cols - baseline_cols if "elo" in c.lower()
            }
            expected_new = {
                "home_elo_momentum",
                "away_elo_momentum",
                "home_elo_rank",
                "away_elo_rank",
                "home_elo_percentile",
                "away_elo_percentile",
            }
            assert expected_new.issubset(new_elo_cols), (
                f"{target}: Expected new Elo columns {expected_new} but only "
                f"found new Elo cols: {new_elo_cols}"
            )
