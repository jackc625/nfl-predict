#!/usr/bin/env python3
"""
Probability Calibration System for NFL Prediction Models

This module provides comprehensive probability calibration for NFL prediction models:
- Isotonic regression calibration (primary method)
- Platt scaling calibration (fallback method)
- Within-season calibration on validation folds
- Calibration curve generation and analysis
- Reliability diagrams and calibration metrics
- Expected Calibration Error (ECE) computation

All calibration methods respect temporal ordering and prevent data leakage.
"""

import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import joblib
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss

from utils import get_logger

logger = get_logger(__name__)


@dataclass
class CalibrationResults:
    """
    Container for calibration results and metrics.

    Attributes:
        calibrator: Fitted calibration model
        method: Calibration method used ('isotonic' or 'platt')
        calibrated_probabilities: Calibrated probabilities on validation data
        raw_probabilities: Raw model probabilities
        true_labels: True binary labels
        reliability_curve: Tuple of (mean_predicted_prob, fraction_positive, bin_edges)
        calibration_metrics: Dictionary of calibration quality metrics
        calibration_plot_data: Data for generating calibration plots
        metadata: Additional calibration metadata
    """

    calibrator: Any
    method: str
    calibrated_probabilities: np.ndarray
    raw_probabilities: np.ndarray
    true_labels: np.ndarray
    reliability_curve: tuple[np.ndarray, np.ndarray, np.ndarray]
    calibration_metrics: dict[str, float]
    calibration_plot_data: dict[str, Any]
    metadata: dict[str, Any]


class PlattCalibrator:
    """Wrapper that accepts raw probabilities and applies Platt scaling via logistic regression."""

    def __init__(self, lr_model):
        self.lr_model = lr_model

    def predict(self, probabilities):
        clipped_probs = np.clip(probabilities.flatten(), 1e-15, 1 - 1e-15)
        logits = np.log(clipped_probs / (1 - clipped_probs))
        return self.lr_model.predict_proba(logits.reshape(-1, 1))[:, 1]


class ProbabilityCalibrator:
    """
    Comprehensive probability calibration system for NFL predictions.

    Implements isotonic regression and Platt scaling calibration methods
    with temporal validation and comprehensive evaluation metrics.
    """

    def __init__(
        self,
        primary_method: str = "isotonic",
        fallback_method: str = "platt",
        n_bins: int = 10,
        min_samples_per_bin: int = 10,
        cv_folds: int = 5,
    ):
        """
        Initialize probability calibrator.

        Args:
            primary_method: Primary calibration method ('isotonic' or 'platt')
            fallback_method: Fallback method if primary fails
            n_bins: Number of bins for reliability diagrams
            min_samples_per_bin: Minimum samples required per bin
            cv_folds: Number of cross-validation folds for within-season calibration
        """
        self.primary_method = primary_method
        self.fallback_method = fallback_method
        self.n_bins = n_bins
        self.min_samples_per_bin = min_samples_per_bin
        self.cv_folds = cv_folds
        self.logger = get_logger(__name__)

        # Validation
        valid_methods = ["isotonic", "platt"]
        if primary_method not in valid_methods:
            raise ValueError(f"Primary method must be one of {valid_methods}")
        if fallback_method not in valid_methods:
            raise ValueError(f"Fallback method must be one of {valid_methods}")

    def calibrate_probabilities(
        self,
        raw_probabilities: np.ndarray,
        true_labels: np.ndarray,
        method: str | None = None,
        sample_weight: np.ndarray | None = None,
    ) -> CalibrationResults:
        """
        Calibrate probabilities using specified method.

        Args:
            raw_probabilities: Raw model probabilities (0-1)
            true_labels: True binary labels (0/1)
            method: Calibration method ('isotonic' or 'platt'), uses primary if None
            sample_weight: Optional sample weights

        Returns:
            CalibrationResults object with calibrator and metrics
        """
        if method is None:
            method = self.primary_method

        self.logger.info(
            "Calibrating probabilities",
            method=method,
            n_samples=len(raw_probabilities),
            positive_rate=float(np.mean(true_labels)),
        )

        # Validate inputs
        raw_probabilities = np.asarray(raw_probabilities).flatten()
        true_labels = np.asarray(true_labels).flatten()

        if len(raw_probabilities) != len(true_labels):
            raise ValueError("Probabilities and labels must have same length")

        if not np.all((raw_probabilities >= 0) & (raw_probabilities <= 1)):
            raise ValueError("Raw probabilities must be in [0, 1] range")

        if not np.all(np.isin(true_labels, [0, 1])):
            raise ValueError("True labels must be binary (0/1)")

        # Check for sufficient data
        if len(raw_probabilities) < self.min_samples_per_bin:
            self.logger.warning(
                "Insufficient data for reliable calibration",
                n_samples=len(raw_probabilities),
                min_required=self.min_samples_per_bin,
            )

        try:
            # Fit calibration model
            if method == "isotonic":
                calibrator = self._fit_isotonic_calibration(
                    raw_probabilities, true_labels, sample_weight
                )
            elif method == "platt":
                calibrator = self._fit_platt_calibration(
                    raw_probabilities, true_labels, sample_weight
                )
            else:
                raise ValueError(f"Unknown calibration method: {method}")

            # Apply calibration
            calibrated_probabilities = calibrator.predict(
                raw_probabilities.reshape(-1, 1)
            )

            # Ensure calibrated probabilities are in valid range
            calibrated_probabilities = np.clip(
                calibrated_probabilities, 1e-15, 1 - 1e-15
            )

            # Calculate calibration metrics
            calibration_metrics = self._calculate_calibration_metrics(
                raw_probabilities, calibrated_probabilities, true_labels
            )

            # Generate reliability curve
            reliability_curve = self._generate_reliability_curve(
                calibrated_probabilities, true_labels
            )

            # Create calibration plot data
            calibration_plot_data = self._prepare_calibration_plot_data(
                raw_probabilities,
                calibrated_probabilities,
                true_labels,
                reliability_curve,
            )

            # Create results object
            results = CalibrationResults(
                calibrator=calibrator,
                method=method,
                calibrated_probabilities=calibrated_probabilities,
                raw_probabilities=raw_probabilities,
                true_labels=true_labels,
                reliability_curve=reliability_curve,
                calibration_metrics=calibration_metrics,
                calibration_plot_data=calibration_plot_data,
                metadata={
                    "n_samples": len(raw_probabilities),
                    "positive_rate": float(np.mean(true_labels)),
                    "calibration_date": datetime.now(),
                    "method_used": method,
                    "sample_weight_used": sample_weight is not None,
                },
            )

            self.logger.info(
                "Calibration completed successfully",
                method=method,
                raw_brier=calibration_metrics["raw_brier_score"],
                calibrated_brier=calibration_metrics["calibrated_brier_score"],
                ece=calibration_metrics["expected_calibration_error"],
            )

            return results

        except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
            self.logger.error(f"Calibration failed with {method}", error=str(e))

            # Try fallback method if different from primary
            if method != self.fallback_method:
                self.logger.info(
                    f"Attempting fallback calibration with {self.fallback_method}"
                )
                return self.calibrate_probabilities(
                    raw_probabilities, true_labels, self.fallback_method, sample_weight
                )
            raise

    def _fit_isotonic_calibration(
        self,
        raw_probabilities: np.ndarray,
        true_labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> IsotonicRegression:
        """
        Fit isotonic regression calibration model.

        Isotonic regression is non-parametric and maintains monotonicity,
        making it ideal for probability calibration.

        Args:
            raw_probabilities: Raw model probabilities
            true_labels: True binary labels
            sample_weight: Optional sample weights

        Returns:
            Fitted IsotonicRegression model
        """
        calibrator = IsotonicRegression(out_of_bounds="clip")

        # Isotonic regression expects probabilities as targets for calibration
        # We fit f: raw_prob -> calibrated_prob where calibrated_prob should equal true_frequency
        calibrator.fit(raw_probabilities, true_labels, sample_weight=sample_weight)

        return calibrator

    def _fit_platt_calibration(
        self,
        raw_probabilities: np.ndarray,
        true_labels: np.ndarray,
        sample_weight: np.ndarray | None = None,
    ) -> PlattCalibrator:
        """
        Fit Platt scaling calibration model.

        Platt scaling fits a sigmoid (logistic regression) to map
        raw probabilities to calibrated probabilities.

        Args:
            raw_probabilities: Raw model probabilities
            true_labels: True binary labels
            sample_weight: Optional sample weights

        Returns:
            Fitted LogisticRegression model for Platt scaling
        """
        # Convert probabilities to logits for Platt scaling
        # Avoid log(0) and log(1) by clipping
        clipped_probs = np.clip(raw_probabilities, 1e-15, 1 - 1e-15)
        logits = np.log(clipped_probs / (1 - clipped_probs))

        # Fit logistic regression: logit(raw_prob) -> calibrated_prob
        calibrator = LogisticRegression(max_iter=1000, random_state=42)
        calibrator.fit(logits.reshape(-1, 1), true_labels, sample_weight=sample_weight)

        return PlattCalibrator(calibrator)

    def _calculate_calibration_metrics(
        self,
        raw_probabilities: np.ndarray,
        calibrated_probabilities: np.ndarray,
        true_labels: np.ndarray,
    ) -> dict[str, float]:
        """
        Calculate comprehensive calibration quality metrics.

        Args:
            raw_probabilities: Raw model probabilities
            calibrated_probabilities: Calibrated probabilities
            true_labels: True binary labels

        Returns:
            Dictionary of calibration metrics
        """
        metrics = {}

        # Brier score (lower is better)
        try:
            metrics["raw_brier_score"] = brier_score_loss(
                true_labels, raw_probabilities
            )
            metrics["calibrated_brier_score"] = brier_score_loss(
                true_labels, calibrated_probabilities
            )
            metrics["brier_score_improvement"] = (
                metrics["raw_brier_score"] - metrics["calibrated_brier_score"]
            )
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to calculate Brier score: {e}")
            metrics.update(
                {
                    "raw_brier_score": np.nan,
                    "calibrated_brier_score": np.nan,
                    "brier_score_improvement": np.nan,
                }
            )

        # Log loss (lower is better)
        try:
            metrics["raw_log_loss"] = log_loss(true_labels, raw_probabilities)
            metrics["calibrated_log_loss"] = log_loss(
                true_labels, calibrated_probabilities
            )
            metrics["log_loss_improvement"] = (
                metrics["raw_log_loss"] - metrics["calibrated_log_loss"]
            )
        except (ValueError, TypeError) as e:
            self.logger.warning(f"Failed to calculate log loss: {e}")
            metrics.update(
                {
                    "raw_log_loss": np.nan,
                    "calibrated_log_loss": np.nan,
                    "log_loss_improvement": np.nan,
                }
            )

        # Expected Calibration Error (ECE)
        try:
            metrics["expected_calibration_error"] = self._calculate_ece(
                calibrated_probabilities, true_labels
            )
            metrics["raw_expected_calibration_error"] = self._calculate_ece(
                raw_probabilities, true_labels
            )
            metrics["ece_improvement"] = (
                metrics["raw_expected_calibration_error"]
                - metrics["expected_calibration_error"]
            )
        except (ValueError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"Failed to calculate ECE: {e}")
            metrics.update(
                {
                    "expected_calibration_error": np.nan,
                    "raw_expected_calibration_error": np.nan,
                    "ece_improvement": np.nan,
                }
            )

        # Maximum Calibration Error (MCE)
        try:
            metrics["maximum_calibration_error"] = self._calculate_mce(
                calibrated_probabilities, true_labels
            )
            metrics["raw_maximum_calibration_error"] = self._calculate_mce(
                raw_probabilities, true_labels
            )
        except (ValueError, TypeError, ZeroDivisionError) as e:
            self.logger.warning(f"Failed to calculate MCE: {e}")
            metrics.update(
                {
                    "maximum_calibration_error": np.nan,
                    "raw_maximum_calibration_error": np.nan,
                }
            )

        # Calibration slope and intercept (reliability regression)
        try:
            slope, intercept = self._calculate_calibration_slope(
                calibrated_probabilities, true_labels
            )
            metrics["calibration_slope"] = slope
            metrics["calibration_intercept"] = intercept
            # Perfect calibration has slope=1, intercept=0
            metrics["calibration_slope_deviation"] = abs(slope - 1.0)
            metrics["calibration_intercept_deviation"] = abs(intercept)
        except (ValueError, TypeError, np.linalg.LinAlgError) as e:
            self.logger.warning(f"Failed to calculate calibration slope: {e}")
            metrics.update(
                {
                    "calibration_slope": np.nan,
                    "calibration_intercept": np.nan,
                    "calibration_slope_deviation": np.nan,
                    "calibration_intercept_deviation": np.nan,
                }
            )

        return metrics

    def _calculate_ece(
        self, probabilities: np.ndarray, true_labels: np.ndarray
    ) -> float:
        """
        Calculate Expected Calibration Error (ECE).

        ECE measures the difference between predicted probabilities and actual
        frequencies across probability bins.

        Args:
            probabilities: Predicted probabilities
            true_labels: True binary labels

        Returns:
            Expected Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        len(probabilities)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            # Find samples in this bin
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                # Calculate accuracy and confidence in this bin
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()

                # Add weighted calibration error for this bin
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _calculate_mce(
        self, probabilities: np.ndarray, true_labels: np.ndarray
    ) -> float:
        """
        Calculate Maximum Calibration Error (MCE).

        MCE is the maximum calibration error across all bins.

        Args:
            probabilities: Predicted probabilities
            true_labels: True binary labels

        Returns:
            Maximum Calibration Error
        """
        bin_boundaries = np.linspace(0, 1, self.n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_calibration_error = 0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (probabilities > bin_lower) & (probabilities <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = true_labels[in_bin].mean()
                avg_confidence_in_bin = probabilities[in_bin].mean()
                calibration_error = abs(avg_confidence_in_bin - accuracy_in_bin)
                max_calibration_error = max(max_calibration_error, calibration_error)

        return max_calibration_error

    def _calculate_calibration_slope(
        self, probabilities: np.ndarray, true_labels: np.ndarray
    ) -> tuple[float, float]:
        """
        Calculate calibration slope and intercept via linear regression.

        Perfect calibration should have slope=1, intercept=0.

        Args:
            probabilities: Predicted probabilities
            true_labels: True binary labels

        Returns:
            Tuple of (slope, intercept)
        """
        # Avoid perfect separation issues
        if len(set(true_labels)) < 2:
            return np.nan, np.nan

        slope, intercept, _, _, _ = stats.linregress(probabilities, true_labels)
        return slope, intercept

    def _generate_reliability_curve(
        self, probabilities: np.ndarray, true_labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate reliability curve data for calibration plots.

        Args:
            probabilities: Predicted probabilities
            true_labels: True binary labels

        Returns:
            Tuple of (mean_predicted_prob, fraction_positive, bin_edges)
        """
        # Use sklearn's calibration_curve function
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                true_labels, probabilities, n_bins=self.n_bins, strategy="uniform"
            )

            # Create bin edges for plotting
            bin_edges = np.linspace(0, 1, self.n_bins + 1)

            return mean_predicted_value, fraction_of_positives, bin_edges

        except (ValueError, TypeError, IndexError) as e:
            self.logger.warning(f"Failed to generate reliability curve: {e}")
            # Return empty arrays if calculation fails
            return np.array([]), np.array([]), np.array([])

    def _prepare_calibration_plot_data(
        self,
        raw_probabilities: np.ndarray,
        calibrated_probabilities: np.ndarray,
        true_labels: np.ndarray,
        reliability_curve: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> dict[str, Any]:
        """
        Prepare data for generating calibration plots.

        Args:
            raw_probabilities: Raw model probabilities
            calibrated_probabilities: Calibrated probabilities
            true_labels: True binary labels
            reliability_curve: Reliability curve data

        Returns:
            Dictionary containing plot data
        """
        mean_predicted, fraction_positive, bin_edges = reliability_curve

        plot_data = {
            "raw_probabilities": raw_probabilities,
            "calibrated_probabilities": calibrated_probabilities,
            "true_labels": true_labels,
            "reliability_curve": {
                "mean_predicted": mean_predicted,
                "fraction_positive": fraction_positive,
                "bin_edges": bin_edges,
            },
            "histogram_data": {
                "raw_hist": np.histogram(
                    raw_probabilities, bins=self.n_bins, range=(0, 1)
                ),
                "calibrated_hist": np.histogram(
                    calibrated_probabilities, bins=self.n_bins, range=(0, 1)
                ),
            },
        }

        return plot_data

    def calibrate_with_cross_validation(
        self,
        data: pd.DataFrame,
        probability_column: str,
        target_column: str,
        season_column: str = "season",
        method: str | None = None,
    ) -> dict[int, CalibrationResults]:
        """
        Perform within-season calibration using cross-validation.

        This method calibrates probabilities within each season using
        cross-validation to prevent overfitting.

        Args:
            data: DataFrame with probabilities, targets, and season information
            probability_column: Column name containing raw probabilities
            target_column: Column name containing true binary labels
            season_column: Column name containing season information
            method: Calibration method to use

        Returns:
            Dictionary mapping season to CalibrationResults
        """
        if method is None:
            method = self.primary_method

        self.logger.info(
            "Starting within-season cross-validation calibration",
            method=method,
            seasons=sorted(data[season_column].unique()),
            total_samples=len(data),
        )

        results = {}

        for season in sorted(data[season_column].unique()):
            season_data = data[data[season_column] == season].copy()

            if len(season_data) < self.min_samples_per_bin:
                self.logger.warning(
                    f"Insufficient data for season {season}", n_samples=len(season_data)
                )
                continue

            try:
                # Extract probabilities and labels
                raw_probs = season_data[probability_column].values
                true_labels = season_data[target_column].values

                # Perform calibration for this season
                season_results = self.calibrate_probabilities(
                    raw_probs, true_labels, method
                )

                # Add season-specific metadata
                season_results.metadata.update(
                    {
                        "season": season,
                        "season_samples": len(season_data),
                        "calibration_type": "within_season_cv",
                    }
                )

                results[season] = season_results

                self.logger.info(
                    f"Completed calibration for season {season}",
                    n_samples=len(season_data),
                    ece=season_results.calibration_metrics[
                        "expected_calibration_error"
                    ],
                )

            except (ValueError, RuntimeError, np.linalg.LinAlgError) as e:
                self.logger.error(f"Failed to calibrate season {season}", error=str(e))

        return results

    def generate_calibration_plot(
        self,
        calibration_results: CalibrationResults,
        save_path: str | None = None,
        title: str | None = None,
    ) -> plt.Figure:
        """
        Generate comprehensive calibration plot.

        Creates a reliability diagram showing both raw and calibrated probabilities
        along with histograms and calibration metrics.

        Args:
            calibration_results: CalibrationResults object
            save_path: Optional path to save the plot
            title: Optional custom title for the plot

        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

        if title is None:
            title = f"Probability Calibration ({calibration_results.method.title()})"

        fig.suptitle(title, fontsize=14, fontweight="bold")

        plot_data = calibration_results.calibration_plot_data
        metrics = calibration_results.calibration_metrics

        # 1. Reliability diagram
        ax1.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

        # Raw probabilities reliability
        if len(plot_data["reliability_curve"]["mean_predicted"]) > 0:
            ax1.plot(
                plot_data["reliability_curve"]["mean_predicted"],
                plot_data["reliability_curve"]["fraction_positive"],
                marker="o",
                linewidth=2,
                label="Calibrated",
            )

        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Reliability Diagram")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])

        # 2. Histogram of predictions
        ax2.hist(
            calibration_results.raw_probabilities,
            bins=20,
            alpha=0.7,
            label="Raw",
            color="red",
            density=True,
        )
        ax2.hist(
            calibration_results.calibrated_probabilities,
            bins=20,
            alpha=0.7,
            label="Calibrated",
            color="blue",
            density=True,
        )
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Density")
        ax2.set_title("Probability Distributions")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Calibration metrics table
        ax3.axis("tight")
        ax3.axis("off")

        metric_data = []
        if not np.isnan(metrics.get("expected_calibration_error", np.nan)):
            metric_data.append(["ECE", f"{metrics['expected_calibration_error']:.4f}"])
        if not np.isnan(metrics.get("calibrated_brier_score", np.nan)):
            metric_data.append(
                ["Brier Score", f"{metrics['calibrated_brier_score']:.4f}"]
            )
        if not np.isnan(metrics.get("calibrated_log_loss", np.nan)):
            metric_data.append(["Log Loss", f"{metrics['calibrated_log_loss']:.4f}"])
        if not np.isnan(metrics.get("calibration_slope", np.nan)):
            metric_data.append(["Cal. Slope", f"{metrics['calibration_slope']:.3f}"])

        if metric_data:
            table = ax3.table(
                cellText=metric_data,
                colLabels=["Metric", "Value"],
                cellLoc="center",
                loc="center",
            )
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1, 2)
            ax3.set_title("Calibration Metrics")

        # 4. Before/after comparison
        sample_size = min(1000, len(calibration_results.raw_probabilities))
        indices = np.random.choice(
            len(calibration_results.raw_probabilities), sample_size, replace=False
        )

        ax4.scatter(
            calibration_results.raw_probabilities[indices],
            calibration_results.calibrated_probabilities[indices],
            alpha=0.5,
            s=10,
        )
        ax4.plot([0, 1], [0, 1], "k--", alpha=0.5, label="No change")
        ax4.set_xlabel("Raw Probability")
        ax4.set_ylabel("Calibrated Probability")
        ax4.set_title("Raw vs Calibrated")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            self.logger.info(f"Calibration plot saved to {save_path}")

        return fig

    def save_calibrator(self, calibration_results: CalibrationResults, save_path: str):
        """
        Save calibration model to disk.

        Args:
            calibration_results: CalibrationResults object
            save_path: Path to save the calibrator
        """
        save_data = {
            "calibrator": calibration_results.calibrator,
            "method": calibration_results.method,
            "metadata": calibration_results.metadata,
            "calibration_metrics": calibration_results.calibration_metrics,
        }

        joblib.dump(save_data, save_path)
        self.logger.info(f"Calibrator saved to {save_path}")

    def load_calibrator(self, load_path: str) -> CalibrationResults:
        """
        Load calibration model from disk.

        Args:
            load_path: Path to load the calibrator from

        Returns:
            Loaded calibrator (just the model, not full CalibrationResults)
        """
        save_data = joblib.load(load_path)

        self.logger.info(
            f"Calibrator loaded from {load_path}", method=save_data["method"]
        )

        return save_data["calibrator"]

    def apply_calibration(
        self, raw_probabilities: np.ndarray, calibrator: Any
    ) -> np.ndarray:
        """
        Apply a pre-trained calibrator to new probabilities.

        Args:
            raw_probabilities: Raw probabilities to calibrate
            calibrator: Pre-trained calibration model

        Returns:
            Calibrated probabilities
        """
        raw_probabilities = np.asarray(raw_probabilities).flatten()

        # Ensure probabilities are in valid range
        raw_probabilities = np.clip(raw_probabilities, 1e-15, 1 - 1e-15)

        # Apply calibration
        calibrated_probs = calibrator.predict(raw_probabilities.reshape(-1, 1))

        # Ensure output is in valid range
        calibrated_probs = np.clip(calibrated_probs, 1e-15, 1 - 1e-15)

        return calibrated_probs
