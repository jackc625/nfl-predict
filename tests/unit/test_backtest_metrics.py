"""Tests for backtest metrics module (BACK-03).

Validates Brier score decomposition, per-target metric computation,
and season summary aggregation.
"""

import numpy as np
import pandas as pd
import pytest

from backtest.metrics import (
    brier_decomposition,
    compute_ats_metrics,
    compute_ou_metrics,
    compute_season_summary,
    compute_target_metrics,
    compute_wp_metrics,
)


class TestBrierDecomposition:
    """Tests for brier_decomposition()."""

    def test_brier_decomposition_components_sum(self) -> None:
        """Verify brier_score == reliability - resolution + uncertainty within 1e-10."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_prob = np.clip(y_true + rng.normal(0, 0.3, size=200), 0.01, 0.99)

        result = brier_decomposition(y_true, y_prob)
        computed_sum = result["reliability"] - result["resolution"] + result["uncertainty"]
        assert abs(result["brier_score"] - computed_sum) < 1e-10

    def test_brier_decomposition_perfect_calibration(self) -> None:
        """Predictions matching observed frequencies should have near-zero reliability."""
        # Build data where each probability bin has matching observed rate
        rng = np.random.default_rng(42)
        n_per_bin = 100
        y_true = []
        y_prob = []
        for target_prob in [0.2, 0.4, 0.6, 0.8]:
            outcomes = rng.binomial(1, target_prob, size=n_per_bin)
            y_true.extend(outcomes)
            y_prob.extend([target_prob] * n_per_bin)

        result = brier_decomposition(np.array(y_true, dtype=float), np.array(y_prob))
        # Reliability should be near zero (predictions match observed rates)
        assert result["reliability"] < 0.01

    def test_brier_decomposition_no_resolution(self) -> None:
        """All predictions = 0.5 for 50/50 outcomes should have resolution == 0."""
        rng = np.random.default_rng(42)
        n = 200
        y_true = rng.integers(0, 2, size=n).astype(float)
        y_prob = np.full(n, 0.5)

        result = brier_decomposition(y_true, y_prob)
        assert result["resolution"] == pytest.approx(0.0, abs=1e-10)

    def test_brier_decomposition_uncertainty(self) -> None:
        """For base_rate ~ 0.5, uncertainty should be ~ 0.25."""
        n = 1000
        y_true = np.array([1.0] * 500 + [0.0] * 500)
        y_prob = np.random.default_rng(42).uniform(0.1, 0.9, size=n)

        result = brier_decomposition(y_true, y_prob)
        assert result["uncertainty"] == pytest.approx(0.25, abs=0.001)

    def test_brier_decomposition_returns_all_keys(self) -> None:
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3])

        result = brier_decomposition(y_true, y_prob)
        assert "brier_score" in result
        assert "reliability" in result
        assert "resolution" in result
        assert "uncertainty" in result


class TestComputeWPMetrics:
    """Tests for compute_wp_metrics()."""

    def test_wp_metrics_has_all_keys(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_prob = np.clip(y_true * 0.6 + 0.2, 0.01, 0.99)

        result = compute_wp_metrics(y_true, y_prob)
        expected_keys = [
            "accuracy", "brier_score", "brier_reliability",
            "brier_resolution", "brier_uncertainty", "ece", "log_loss",
        ]
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"

    def test_wp_accuracy_perfect(self) -> None:
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])

        result = compute_wp_metrics(y_true, y_prob)
        assert result["accuracy"] == 1.0

    def test_wp_brier_consistency(self) -> None:
        """Brier decomposition should be consistent with overall brier_score."""
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=200).astype(float)
        y_prob = np.clip(y_true + rng.normal(0, 0.3, size=200), 0.01, 0.99)

        result = compute_wp_metrics(y_true, y_prob)
        decomposed = result["brier_reliability"] - result["brier_resolution"] + result["brier_uncertainty"]
        assert abs(result["brier_score"] - decomposed) < 1e-10


class TestComputeATSMetrics:
    """Tests for compute_ats_metrics()."""

    def test_ats_metrics_has_mae_rmse(self) -> None:
        y_true = np.array([3.0, -7.0, 14.0, -2.5])
        y_pred = np.array([5.0, -5.0, 10.0, -1.0])

        result = compute_ats_metrics(y_true, y_pred)
        assert "mae" in result
        assert "rmse" in result

    def test_ats_metrics_cover_accuracy(self) -> None:
        y_true = np.array([3.0, -7.0, 14.0, -2.5])
        y_pred = np.array([5.0, -5.0, 10.0, -1.0])
        y_covered = np.array([1, 0, 1, 0])

        result = compute_ats_metrics(y_true, y_pred, y_true_covered=y_covered)
        assert "cover_accuracy" in result

    def test_ats_metrics_no_brier(self) -> None:
        """ATS should NOT include Brier decomposition keys."""
        y_true = np.array([3.0, -7.0])
        y_pred = np.array([5.0, -5.0])

        result = compute_ats_metrics(y_true, y_pred)
        assert "brier_reliability" not in result
        assert "brier_resolution" not in result


class TestComputeOUMetrics:
    """Tests for compute_ou_metrics()."""

    def test_ou_metrics_has_mae_rmse(self) -> None:
        y_true = np.array([45.0, 52.0, 38.0, 41.0])
        y_pred = np.array([43.0, 50.0, 40.0, 42.0])

        result = compute_ou_metrics(y_true, y_pred)
        assert "mae" in result
        assert "rmse" in result

    def test_ou_metrics_over_accuracy(self) -> None:
        y_true = np.array([45.0, 52.0, 38.0, 41.0])
        y_pred = np.array([43.0, 50.0, 40.0, 42.0])
        y_went_over = np.array([1, 1, 0, 0])

        result = compute_ou_metrics(y_true, y_pred, y_true_went_over=y_went_over)
        assert "over_accuracy" in result


class TestComputeTargetMetrics:
    """Tests for compute_target_metrics() dispatch."""

    def test_dispatches_wp(self) -> None:
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=100).astype(float)
        y_pred = np.clip(y_true * 0.6 + 0.2, 0.01, 0.99)

        result = compute_target_metrics("wp", y_true, y_pred)
        assert "brier_score" in result
        assert "brier_reliability" in result
        assert "accuracy" in result

    def test_dispatches_ats(self) -> None:
        y_true = np.array([3.0, -7.0, 14.0, -2.5])
        y_pred = np.array([5.0, -5.0, 10.0, -1.0])

        result = compute_target_metrics("ats", y_true, y_pred)
        assert "mae" in result
        assert "rmse" in result
        assert "brier_score" not in result

    def test_dispatches_ou(self) -> None:
        y_true = np.array([45.0, 52.0, 38.0])
        y_pred = np.array([43.0, 50.0, 40.0])

        result = compute_target_metrics("ou", y_true, y_pred)
        assert "mae" in result
        assert "rmse" in result
        assert "brier_score" not in result

    def test_unknown_target_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown target"):
            compute_target_metrics("invalid", np.array([1.0]), np.array([0.5]))


class TestComputeSeasonSummary:
    """Tests for compute_season_summary()."""

    def test_aggregates_metrics(self) -> None:
        season_results = [
            {"season": 2021, "accuracy": 0.60, "n_games": 100},
            {"season": 2022, "accuracy": 0.65, "n_games": 120},
            {"season": 2023, "accuracy": 0.55, "n_games": 110},
        ]
        clv_df = pd.DataFrame({
            "probability_clv": [0.02, -0.01, 0.03, 0.01],
            "has_closing_odds": [True, True, True, True],
        })

        result = compute_season_summary(season_results, clv_df)
        assert "total_games" in result
        assert result["total_games"] == 330
        assert "mean_clv" in result
        assert "positive_clv_pct" in result

    def test_handles_no_clv(self) -> None:
        season_results = [
            {"season": 2021, "n_games": 100},
        ]
        result = compute_season_summary(season_results, None)
        assert result["total_games"] == 100
        assert result["mean_clv"] is None
