#!/usr/bin/env python3
"""
Model Evaluation Framework

This module provides comprehensive evaluation metrics for NFL prediction models:
- Core metrics: LogLoss, Brier score for WP; MAE, RMSE for regression targets
- Calibration metrics: ECE, reliability diagrams, sharpness
- Betting simulation metrics: ROI, CLV, Kelly criterion analysis
- Comprehensive evaluation reporting and visualization

All metrics are designed for proper evaluation of sports betting models.
"""

import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from utils import get_logger
from utils.probability_utils import moneyline_to_probability

logger = get_logger(__name__)


@dataclass
class EvaluationMetrics:
    """
    Container for all evaluation metrics.

    Attributes:
        core_metrics: Basic performance metrics
        calibration_metrics: Probability calibration quality
        betting_metrics: Betting simulation results
        statistical_tests: Statistical significance tests
        sample_info: Information about the evaluation sample
    """

    core_metrics: dict[str, float] = field(default_factory=dict)
    calibration_metrics: dict[str, float] = field(default_factory=dict)
    betting_metrics: dict[str, float] = field(default_factory=dict)
    statistical_tests: dict[str, Any] = field(default_factory=dict)
    sample_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "core_metrics": self.core_metrics,
            "calibration_metrics": self.calibration_metrics,
            "betting_metrics": self.betting_metrics,
            "statistical_tests": self.statistical_tests,
            "sample_info": self.sample_info,
        }


@dataclass
class BettingSimulationResult:
    """
    Container for betting simulation results.

    Attributes:
        bets_placed: Number of bets placed
        win_rate: Percentage of winning bets
        roi: Return on investment
        total_profit: Total profit/loss
        avg_bet_size: Average bet size
        max_drawdown: Maximum drawdown experienced
        sharpe_ratio: Risk-adjusted return metric
        kelly_fraction: Optimal Kelly fraction used
        clv: Closing line value (if available)
        bet_history: Individual bet results
    """

    bets_placed: int
    win_rate: float
    roi: float
    total_profit: float
    avg_bet_size: float
    max_drawdown: float
    sharpe_ratio: float
    kelly_fraction: float
    clv: float | None = None
    bet_history: list[dict[str, Any]] = field(default_factory=list)


class ModelEvaluationFramework:
    """
    Comprehensive evaluation framework for NFL prediction models.

    Supports evaluation of:
    - Binary classification models (WP, ATS)
    - Regression models (point totals, margins)
    - Betting simulations with Kelly criterion
    - Model calibration and reliability
    """

    def __init__(
        self,
        n_calibration_bins: int = 10,
        confidence_level: float = 0.95,
        betting_bankroll: float = 10000.0,
        max_bet_fraction: float = 0.05,
        min_edge_threshold: float = 0.02,
    ):
        """
        Initialize evaluation framework.

        Args:
            n_calibration_bins: Number of bins for calibration analysis
            confidence_level: Confidence level for statistical tests
            betting_bankroll: Starting bankroll for betting simulation
            max_bet_fraction: Maximum fraction of bankroll to bet
            min_edge_threshold: Minimum edge required to place bet
        """
        self.n_calibration_bins = n_calibration_bins
        self.confidence_level = confidence_level
        self.betting_bankroll = betting_bankroll
        self.max_bet_fraction = max_bet_fraction
        self.min_edge_threshold = min_edge_threshold

        self.logger = get_logger(__name__)

    def evaluate_model(
        self,
        predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        market_odds: np.ndarray | None = None,
        closing_odds: np.ndarray | None = None,
        prediction_type: str = "binary",
        model_name: str = "Model",
    ) -> EvaluationMetrics:
        """
        Comprehensive model evaluation.

        Args:
            predictions: Model predictions
            actual_outcomes: True outcomes
            market_odds: Market odds at prediction time (optional)
            closing_odds: Closing odds for CLV calculation (optional)
            prediction_type: "binary" for classification, "regression" for regression
            model_name: Name of model being evaluated

        Returns:
            EvaluationMetrics object with all computed metrics
        """
        self.logger.info(
            f"Evaluating {model_name}",
            predictions=len(predictions),
            prediction_type=prediction_type,
        )

        # Input validation
        predictions, actual_outcomes = self._validate_inputs(
            predictions, actual_outcomes
        )

        # Initialize metrics container
        metrics = EvaluationMetrics()

        # Core performance metrics
        if prediction_type == "binary":
            metrics.core_metrics = self._calculate_classification_metrics(
                predictions, actual_outcomes
            )
            metrics.calibration_metrics = self._calculate_calibration_metrics(
                predictions, actual_outcomes
            )
        else:
            metrics.core_metrics = self._calculate_regression_metrics(
                predictions, actual_outcomes
            )

        # Betting simulation (if market odds available)
        if market_odds is not None:
            betting_result = self._simulate_betting(
                predictions, actual_outcomes, market_odds, closing_odds
            )
            metrics.betting_metrics = self._extract_betting_metrics(betting_result)

            # Calculate CLV if closing odds available
            if closing_odds is not None:
                metrics.betting_metrics["clv"] = self._calculate_clv(
                    predictions, market_odds, closing_odds
                )

        # Statistical significance tests
        metrics.statistical_tests = self._perform_statistical_tests(
            predictions, actual_outcomes, prediction_type
        )

        # Sample information
        metrics.sample_info = self._gather_sample_info(
            predictions, actual_outcomes, market_odds
        )

        self.logger.info(
            f"{model_name} evaluation completed",
            accuracy=metrics.core_metrics.get("accuracy"),
            log_loss=metrics.core_metrics.get("log_loss"),
            roi=metrics.betting_metrics.get("roi"),
        )

        return metrics

    def _validate_inputs(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate and clean input arrays."""
        predictions = np.asarray(predictions)
        actual_outcomes = np.asarray(actual_outcomes)

        if len(predictions) != len(actual_outcomes):
            raise ValueError("Predictions and outcomes must have same length")

        # Remove NaN values
        valid_mask = ~(np.isnan(predictions) | np.isnan(actual_outcomes))
        predictions = predictions[valid_mask]
        actual_outcomes = actual_outcomes[valid_mask]

        if len(predictions) == 0:
            raise ValueError("No valid predictions after removing NaN values")

        return predictions, actual_outcomes

    def _calculate_classification_metrics(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray
    ) -> dict[str, float]:
        """Calculate classification metrics (accuracy, log-loss, Brier score)."""
        metrics = {}

        # Ensure predictions are probabilities
        if not all(0 <= p <= 1 for p in predictions):
            self.logger.warning("Predictions outside [0,1] range detected")
            predictions = np.clip(predictions, 1e-10, 1 - 1e-10)

        # Basic metrics
        binary_predictions = (predictions >= 0.5).astype(int)
        metrics["accuracy"] = accuracy_score(actual_outcomes, binary_predictions)

        try:
            metrics["log_loss"] = log_loss(actual_outcomes, predictions)
        except ValueError:
            metrics["log_loss"] = np.nan

        try:
            metrics["brier_score"] = brier_score_loss(actual_outcomes, predictions)
        except ValueError:
            metrics["brier_score"] = np.nan

        # Additional classification metrics
        metrics["precision"] = self._safe_precision(actual_outcomes, binary_predictions)
        metrics["recall"] = self._safe_recall(actual_outcomes, binary_predictions)
        metrics["f1_score"] = self._safe_f1_score(actual_outcomes, binary_predictions)

        # Probability-based metrics
        metrics["avg_predicted_prob"] = np.mean(predictions)
        metrics["base_rate"] = np.mean(actual_outcomes)
        metrics["prob_std"] = np.std(predictions)

        return metrics

    def _calculate_regression_metrics(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray
    ) -> dict[str, float]:
        """Calculate regression metrics (MAE, RMSE, R²)."""
        metrics = {}

        # Core regression metrics
        metrics["mae"] = mean_absolute_error(actual_outcomes, predictions)
        metrics["rmse"] = np.sqrt(mean_squared_error(actual_outcomes, predictions))
        metrics["r2"] = r2_score(actual_outcomes, predictions)

        # Additional regression metrics
        errors = actual_outcomes - predictions
        metrics["mean_error"] = np.mean(errors)
        metrics["median_error"] = np.median(errors)
        metrics["error_std"] = np.std(errors)
        metrics["max_error"] = np.max(np.abs(errors))

        # Percentage-based metrics
        metrics["mape"] = np.mean(np.abs(errors / actual_outcomes)) * 100

        return metrics

    def _calculate_calibration_metrics(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray
    ) -> dict[str, float]:
        """Calculate calibration metrics (ECE, reliability)."""
        metrics = {}

        # Expected Calibration Error (ECE)
        metrics["ece"] = self._calculate_ece(predictions, actual_outcomes)

        # Maximum Calibration Error (MCE)
        metrics["mce"] = self._calculate_mce(predictions, actual_outcomes)

        # Reliability and Resolution
        reliability, resolution = self._calculate_reliability_resolution(
            predictions, actual_outcomes
        )
        metrics["reliability"] = reliability
        metrics["resolution"] = resolution

        # Sharpness (average confidence)
        metrics["sharpness"] = np.mean(np.abs(predictions - 0.5))

        # Hosmer-Lemeshow test
        metrics["hosmer_lemeshow_pvalue"] = self._hosmer_lemeshow_test(
            predictions, actual_outcomes
        )

        return metrics

    def _calculate_ece(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray
    ) -> float:
        """Calculate Expected Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        ece = 0
        len(predictions)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = actual_outcomes[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece

    def _calculate_mce(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray
    ) -> float:
        """Calculate Maximum Calibration Error."""
        bin_boundaries = np.linspace(0, 1, self.n_calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        max_error = 0

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)

            if in_bin.sum() > 0:
                accuracy_in_bin = actual_outcomes[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()
                max_error = max(max_error, abs(avg_confidence_in_bin - accuracy_in_bin))

        return max_error

    def _calculate_reliability_resolution(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray
    ) -> tuple[float, float]:
        """Calculate reliability and resolution components."""
        bin_boundaries = np.linspace(0, 1, self.n_calibration_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]

        overall_accuracy = np.mean(actual_outcomes)
        reliability = 0
        resolution = 0
        len(predictions)

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers, strict=False):
            in_bin = (predictions > bin_lower) & (predictions <= bin_upper)
            prop_in_bin = in_bin.mean()

            if prop_in_bin > 0:
                accuracy_in_bin = actual_outcomes[in_bin].mean()
                avg_confidence_in_bin = predictions[in_bin].mean()

                reliability += (
                    prop_in_bin * (avg_confidence_in_bin - accuracy_in_bin) ** 2
                )
                resolution += prop_in_bin * (accuracy_in_bin - overall_accuracy) ** 2

        return reliability, resolution

    def _hosmer_lemeshow_test(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray
    ) -> float:
        """Perform Hosmer-Lemeshow goodness-of-fit test."""
        # Sort by predicted probability
        sorted_idx = np.argsort(predictions)
        predictions_sorted = predictions[sorted_idx]
        outcomes_sorted = actual_outcomes[sorted_idx]

        # Create deciles
        n = len(predictions)
        decile_size = n // 10

        chi_square = 0

        for i in range(10):
            start_idx = i * decile_size
            if i == 9:  # Last decile includes remaining observations
                end_idx = n
            else:
                end_idx = (i + 1) * decile_size

            decile_predictions = predictions_sorted[start_idx:end_idx]
            decile_outcomes = outcomes_sorted[start_idx:end_idx]

            observed_events = np.sum(decile_outcomes)
            expected_events = np.sum(decile_predictions)

            observed_non_events = len(decile_outcomes) - observed_events
            expected_non_events = len(decile_predictions) - expected_events

            # Add to chi-square statistic
            if expected_events > 0 and expected_non_events > 0:
                chi_square += (
                    (observed_events - expected_events) ** 2
                ) / expected_events
                chi_square += (
                    (observed_non_events - expected_non_events) ** 2
                ) / expected_non_events

        # Calculate p-value (8 degrees of freedom for 10 groups - 2)
        p_value = 1 - chi2.cdf(chi_square, df=8)

        return p_value

    def _simulate_betting(
        self,
        predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        market_odds: np.ndarray,
        closing_odds: np.ndarray | None = None,
    ) -> BettingSimulationResult:
        """Simulate betting with Kelly criterion."""
        bankroll = self.betting_bankroll
        bet_history = []
        bankroll_history = [bankroll]

        for i, (pred_prob, outcome, odds) in enumerate(
            zip(predictions, actual_outcomes, market_odds, strict=False)
        ):
            # Convert odds to market probability
            market_prob = moneyline_to_probability(int(odds))

            # Calculate edge
            edge = pred_prob - market_prob

            # Only bet if edge meets threshold
            if abs(edge) >= self.min_edge_threshold:
                # Calculate Kelly fraction
                if odds > 0:  # Positive odds (underdog)
                    kelly_fraction = edge / (odds / 100)
                else:  # Negative odds (favorite)
                    kelly_fraction = edge / (100 / abs(odds))

                # Limit bet size
                bet_fraction = min(abs(kelly_fraction), self.max_bet_fraction)
                bet_amount = bankroll * bet_fraction

                # Determine bet side (home or away)
                if edge > 0:  # Bet on prediction
                    bet_on_home = pred_prob > 0.5
                else:  # Bet against prediction
                    bet_on_home = pred_prob <= 0.5

                # Calculate payout
                if (bet_on_home and outcome == 1) or (
                    not bet_on_home and outcome == 0
                ):  # Won bet on home
                    payout = (
                        bet_amount * (odds / 100)
                        if odds > 0
                        else bet_amount * (100 / abs(odds))
                    )
                else:  # Lost bet
                    payout = -bet_amount

                bankroll += payout

                bet_history.append(
                    {
                        "game_idx": i,
                        "bet_amount": bet_amount,
                        "bet_on_home": bet_on_home,
                        "edge": edge,
                        "kelly_fraction": kelly_fraction,
                        "payout": payout,
                        "bankroll_after": bankroll,
                        "market_odds": odds,
                        "predicted_prob": pred_prob,
                        "actual_outcome": outcome,
                    }
                )

            bankroll_history.append(bankroll)

        # Calculate results
        if bet_history:
            total_bet = sum(abs(bet["bet_amount"]) for bet in bet_history)
            total_profit = sum(bet["payout"] for bet in bet_history)
            winning_bets = sum(1 for bet in bet_history if bet["payout"] > 0)

            # Calculate drawdown
            peak = self.betting_bankroll
            max_drawdown = 0
            for balance in bankroll_history:
                peak = max(peak, balance)
                drawdown = (peak - balance) / peak
                max_drawdown = max(max_drawdown, drawdown)

            # Calculate Sharpe ratio (simplified)
            returns = np.diff(bankroll_history) / bankroll_history[:-1]
            if len(returns) > 1 and np.std(returns) > 0:
                sharpe_ratio = (
                    np.mean(returns) / np.std(returns) * np.sqrt(252)
                )  # Annualized
            else:
                sharpe_ratio = 0

            result = BettingSimulationResult(
                bets_placed=len(bet_history),
                win_rate=winning_bets / len(bet_history) if bet_history else 0,
                roi=total_profit / total_bet if total_bet > 0 else 0,
                total_profit=total_profit,
                avg_bet_size=total_bet / len(bet_history) if bet_history else 0,
                max_drawdown=max_drawdown,
                sharpe_ratio=sharpe_ratio,
                kelly_fraction=np.mean([bet["kelly_fraction"] for bet in bet_history]),
                bet_history=bet_history,
            )
        else:
            result = BettingSimulationResult(
                bets_placed=0,
                win_rate=0,
                roi=0,
                total_profit=0,
                avg_bet_size=0,
                max_drawdown=0,
                sharpe_ratio=0,
                kelly_fraction=0,
            )

        return result

    def _calculate_clv(
        self, predictions: np.ndarray, market_odds: np.ndarray, closing_odds: np.ndarray
    ) -> float:
        """Calculate Closing Line Value (CLV)."""
        clv_values = []

        for pred_prob, open_odds, close_odds in zip(
            predictions, market_odds, closing_odds, strict=False
        ):
            open_prob = moneyline_to_probability(int(open_odds))
            close_prob = moneyline_to_probability(int(close_odds))

            # CLV = (our edge at close) - (our edge at open)
            open_edge = pred_prob - open_prob
            close_edge = pred_prob - close_prob

            clv = close_edge - open_edge
            clv_values.append(clv)

        return np.mean(clv_values) if clv_values else 0

    def _extract_betting_metrics(
        self, betting_result: BettingSimulationResult
    ) -> dict[str, float]:
        """Extract betting metrics for storage."""
        return {
            "bets_placed": betting_result.bets_placed,
            "win_rate": betting_result.win_rate,
            "roi": betting_result.roi,
            "total_profit": betting_result.total_profit,
            "avg_bet_size": betting_result.avg_bet_size,
            "max_drawdown": betting_result.max_drawdown,
            "sharpe_ratio": betting_result.sharpe_ratio,
            "avg_kelly_fraction": betting_result.kelly_fraction,
        }

    def _perform_statistical_tests(
        self, predictions: np.ndarray, actual_outcomes: np.ndarray, prediction_type: str
    ) -> dict[str, Any]:
        """Perform statistical significance tests."""
        tests = {}

        if prediction_type == "binary":
            # Test if accuracy is significantly better than random
            accuracy = accuracy_score(actual_outcomes, (predictions >= 0.5).astype(int))
            n = len(predictions)

            # Binomial test against 50% accuracy
            from scipy.stats import binomtest

            result = binomtest(int(accuracy * n), n, 0.5, alternative="greater")
            tests["accuracy_vs_random"] = {
                "statistic": accuracy,
                "p_value": result.pvalue,
                "significant": result.pvalue < (1 - self.confidence_level),
            }

            # McNemar's test would require comparing two models

        else:
            # Test if predictions correlate with outcomes
            correlation, p_value = stats.pearsonr(predictions, actual_outcomes)
            tests["correlation"] = {
                "correlation": correlation,
                "p_value": p_value,
                "significant": p_value < (1 - self.confidence_level),
            }

        return tests

    def _gather_sample_info(
        self,
        predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        market_odds: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Gather information about the evaluation sample."""
        info = {
            "n_predictions": len(predictions),
            "prediction_date": datetime.now(),
            "min_prediction": float(np.min(predictions)),
            "max_prediction": float(np.max(predictions)),
            "mean_prediction": float(np.mean(predictions)),
            "std_prediction": float(np.std(predictions)),
        }

        if market_odds is not None:
            info["has_market_odds"] = True
            info["min_odds"] = float(np.min(market_odds))
            info["max_odds"] = float(np.max(market_odds))
        else:
            info["has_market_odds"] = False

        return info

    def generate_calibration_plot(
        self,
        predictions: np.ndarray,
        actual_outcomes: np.ndarray,
        save_path: str | None = None,
    ) -> plt.Figure:
        """Generate reliability/calibration plot."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Reliability diagram
        fraction_of_positives, mean_predicted_value = calibration_curve(
            actual_outcomes, predictions, n_bins=self.n_calibration_bins
        )

        ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
        ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
        ax1.set_xlabel("Mean Predicted Probability")
        ax1.set_ylabel("Fraction of Positives")
        ax1.set_title("Reliability Diagram")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Histogram of predictions
        ax2.hist(predictions, bins=20, alpha=0.7, edgecolor="black")
        ax2.set_xlabel("Predicted Probability")
        ax2.set_ylabel("Count")
        ax2.set_title("Distribution of Predictions")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig

    def compare_models(
        self,
        model_results: dict[str, EvaluationMetrics],
        save_path: str | None = None,
    ) -> pd.DataFrame:
        """Compare multiple models side-by-side."""
        comparison_data = []

        for model_name, metrics in model_results.items():
            row = {"Model": model_name}
            row.update(metrics.core_metrics)
            row.update(metrics.calibration_metrics)
            row.update(metrics.betting_metrics)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)

        if save_path:
            comparison_df.to_csv(save_path, index=False)

        return comparison_df

    def _safe_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate precision with safe handling of edge cases."""
        try:
            from sklearn.metrics import precision_score

            return precision_score(y_true, y_pred, zero_division=0)
        except (ValueError, TypeError, ImportError):
            return 0.0

    def _safe_recall(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate recall with safe handling of edge cases."""
        try:
            from sklearn.metrics import recall_score

            return recall_score(y_true, y_pred, zero_division=0)
        except (ValueError, TypeError, ImportError):
            return 0.0

    def _safe_f1_score(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate F1 score with safe handling of edge cases."""
        try:
            from sklearn.metrics import f1_score

            return f1_score(y_true, y_pred, zero_division=0)
        except (ValueError, TypeError, ImportError):
            return 0.0
