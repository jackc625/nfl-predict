"""
Evaluation Metrics System for NFL Prediction System.

This module provides comprehensive evaluation metrics for model performance,
calibration analysis, edge bucket analysis, and statistical significance testing
for NFL prediction models.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from scipy import stats

from utils.logging_config import get_logger

logger = get_logger(__name__)


class ModelType(Enum):
    """Types of prediction models."""

    WIN_PROBABILITY = "wp"
    AGAINST_THE_SPREAD = "ats"
    OVER_UNDER = "ou"


class MetricType(Enum):
    """Types of evaluation metrics."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CALIBRATION = "calibration"
    BETTING = "betting"


@dataclass
class ClassificationMetrics:
    """Classification performance metrics."""

    accuracy: float
    balanced_accuracy: float  # Forces consideration of class imbalance
    log_loss: float
    brier_score: float
    auc_roc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1_score: float | None = None

    # Bootstrap confidence intervals (optional)
    accuracy_ci: tuple[float, float] | None = None
    log_loss_ci: tuple[float, float] | None = None
    auc_roc_ci: tuple[float, float] | None = None

    # Sample size
    n_samples: int = 0
    n_positive: int = 0
    n_negative: int = 0


@dataclass
class RegressionMetrics:
    """Regression performance metrics."""

    mae: float  # Mean Absolute Error
    rmse: float  # Root Mean Square Error
    mse: float  # Mean Square Error
    r2: float  # R-squared

    # Additional regression metrics
    median_ae: float | None = None
    max_error: float | None = None
    mean_error: float | None = None  # Bias

    # Sample size
    n_samples: int = 0


@dataclass
class CalibrationMetrics:
    """Model calibration metrics."""

    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    ace: float  # Average Calibration Error

    # Reliability diagram data
    bin_boundaries: list[float]
    bin_accuracies: list[float]
    bin_confidences: list[float]
    bin_counts: list[int]

    # Hosmer-Lemeshow test
    hl_statistic: float | None = None
    hl_p_value: float | None = None

    # Sample size
    n_samples: int = 0
    n_bins: int = 10


@dataclass
class EdgeBucketMetrics:
    """Edge bucket analysis for betting evaluation."""

    edge_ranges: list[tuple[float, float]]
    bucket_counts: list[int]
    bucket_accuracies: list[float]
    bucket_rois: list[float]  # Return on Investment
    bucket_profits: list[float]  # Absolute profit
    bucket_hit_rates: list[float]

    # Overall betting metrics
    total_bets: int
    total_profit: float
    total_roi: float
    average_edge: float
    sharpe_ratio: float | None = None

    # Best/worst performing buckets
    best_roi_bucket: int | None = None
    worst_roi_bucket: int | None = None


@dataclass
class SignificanceTest:
    """Statistical significance test results."""

    test_name: str
    statistic: float
    p_value: float
    is_significant: bool
    confidence_level: float

    # Effect size measures
    effect_size: float | None = None
    effect_size_interpretation: str | None = None

    # Additional test details
    degrees_freedom: int | None = None
    test_details: dict[str, Any] | None = None


@dataclass
class ComprehensiveMetrics:
    """Complete evaluation metrics for a model."""

    model_type: ModelType

    # Core metrics
    classification_metrics: ClassificationMetrics | None = None
    regression_metrics: RegressionMetrics | None = None
    calibration_metrics: CalibrationMetrics | None = None
    edge_bucket_metrics: EdgeBucketMetrics | None = None

    # Significance tests
    significance_tests: list[SignificanceTest] = field(default_factory=list)

    # Metadata
    evaluation_date: datetime = field(default_factory=datetime.now)
    sample_period: tuple[datetime, datetime] | None = None
    n_total_samples: int = 0

    # Summary score
    overall_score: float | None = None


class MetricsCalculator:
    """
    Comprehensive metrics calculator for NFL prediction models.

    Provides classification, regression, calibration, and betting metrics
    with statistical significance testing.
    """

    def __init__(self, confidence_level: float = 0.95):
        """Initialize metrics calculator."""
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level

        logger.info(
            f"MetricsCalculator initialized with {confidence_level:.1%} confidence level"
        )

    def calculate_comprehensive_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_type: ModelType,
        betting_odds: np.ndarray | None = None,
        sample_weights: np.ndarray | None = None,
        bootstrap_ci: bool = False,
        bootstrap_samples: int = 1000,
    ) -> ComprehensiveMetrics:
        """
        Calculate comprehensive metrics for a model.

        Args:
            y_true: True values
            y_pred: Predicted values/probabilities
            model_type: Type of model being evaluated
            betting_odds: American odds for betting analysis (optional)
            sample_weights: Sample weights (optional)
            bootstrap_ci: Whether to calculate bootstrap confidence intervals
            bootstrap_samples: Number of bootstrap samples (default 1000)

        Returns:
            ComprehensiveMetrics object with all evaluation results
        """

        logger.info(f"Calculating comprehensive metrics for {model_type.value} model")

        # Input validation
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        if len(y_true) != len(y_pred):
            raise ValueError("y_true and y_pred must have the same length")

        if len(y_true) == 0:
            raise ValueError("Cannot calculate metrics for empty arrays")

        metrics = ComprehensiveMetrics(
            model_type=model_type, n_total_samples=len(y_true)
        )

        # Calculate appropriate metrics based on model type
        if model_type == ModelType.WIN_PROBABILITY:
            # Classification metrics for win probability
            metrics.classification_metrics = self._calculate_classification_metrics(
                y_true, y_pred, sample_weights, bootstrap_ci, bootstrap_samples
            )
            metrics.calibration_metrics = self._calculate_calibration_metrics(
                y_true, y_pred, sample_weights
            )

        elif model_type in [ModelType.AGAINST_THE_SPREAD, ModelType.OVER_UNDER]:
            # Both classification and regression metrics for ATS/OU
            # Treat as classification (cover/not cover, over/under)
            y_true_binary = (y_true > 0.5).astype(int)
            metrics.classification_metrics = self._calculate_classification_metrics(
                y_true_binary, y_pred, sample_weights, bootstrap_ci, bootstrap_samples
            )
            metrics.calibration_metrics = self._calculate_calibration_metrics(
                y_true_binary, y_pred, sample_weights
            )

            # Also calculate as regression (expected margin/total)
            if model_type == ModelType.AGAINST_THE_SPREAD:
                # Convert probabilities back to expected margins for regression analysis
                y_true_margin = (y_true - 0.5) * 20  # Rough conversion to point spread
                y_pred_margin = (y_pred - 0.5) * 20
            else:  # OVER_UNDER
                # Convert to expected total deviation
                y_true_margin = (
                    y_true - 0.5
                ) * 10  # Rough conversion to total deviation
                y_pred_margin = (y_pred - 0.5) * 10

            metrics.regression_metrics = self._calculate_regression_metrics(
                y_true_margin, y_pred_margin, sample_weights
            )

        # Calculate edge bucket analysis if odds provided
        if betting_odds is not None:
            metrics.edge_bucket_metrics = self._calculate_edge_bucket_metrics(
                y_true, y_pred, betting_odds, model_type
            )

        # Calculate significance tests
        metrics.significance_tests = self._calculate_significance_tests(
            y_true, y_pred, model_type
        )

        # Calculate overall score
        metrics.overall_score = self._calculate_overall_score(metrics)

        logger.info(f"Comprehensive metrics calculated for {len(y_true)} samples")
        return metrics

    def _calculate_classification_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: np.ndarray | None = None,
        bootstrap_ci: bool = False,
        bootstrap_samples: int = 1000,
    ) -> ClassificationMetrics:
        """Calculate classification performance metrics."""

        # Convert to binary if needed
        if y_true.dtype != bool and np.max(y_true) <= 1:
            y_true_binary = (y_true > 0.5).astype(int)
        else:
            y_true_binary = y_true.astype(int)

        # Ensure predictions are probabilities
        y_pred_prob = np.clip(y_pred, 1e-15, 1 - 1e-15)  # Avoid log(0)
        y_pred_binary = (y_pred_prob > 0.5).astype(int)

        # Basic metrics
        accuracy = np.average(y_true_binary == y_pred_binary, weights=sample_weights)

        # Balanced accuracy - average of recall for each class
        # This accounts for class imbalance
        if len(np.unique(y_true_binary)) > 1:
            # Calculate recall for each class
            true_positive_rate = np.sum(
                (y_true_binary == 1) & (y_pred_binary == 1)
            ) / np.sum(y_true_binary == 1)
            true_negative_rate = np.sum(
                (y_true_binary == 0) & (y_pred_binary == 0)
            ) / np.sum(y_true_binary == 0)
            balanced_accuracy = (true_positive_rate + true_negative_rate) / 2
        else:
            # Only one class present, fall back to regular accuracy
            balanced_accuracy = accuracy

        # Log loss (cross-entropy)
        log_loss = -np.average(
            y_true_binary * np.log(y_pred_prob)
            + (1 - y_true_binary) * np.log(1 - y_pred_prob),
            weights=sample_weights,
        )

        # Brier score
        brier_score = np.average(
            (y_pred_prob - y_true_binary) ** 2, weights=sample_weights
        )

        # AUC-ROC if we have both classes
        auc_roc = None
        if len(np.unique(y_true_binary)) > 1:
            try:
                from sklearn.metrics import roc_auc_score

                auc_roc = roc_auc_score(
                    y_true_binary, y_pred_prob, sample_weight=sample_weights
                )
            except ImportError:
                logger.warning("sklearn not available, skipping AUC calculation")

        # Precision, Recall, F1
        precision = recall = f1_score = None
        try:
            from sklearn.metrics import f1_score as f1
            from sklearn.metrics import precision_score, recall_score

            precision = precision_score(
                y_true_binary,
                y_pred_binary,
                sample_weight=sample_weights,
                zero_division=0,
            )
            recall = recall_score(
                y_true_binary,
                y_pred_binary,
                sample_weight=sample_weights,
                zero_division=0,
            )
            f1_score = f1(
                y_true_binary,
                y_pred_binary,
                sample_weight=sample_weights,
                zero_division=0,
            )
        except ImportError:
            pass

        # Bootstrap confidence intervals
        accuracy_ci = log_loss_ci = auc_roc_ci = None
        if bootstrap_ci and len(y_true) > 10:  # Minimum sample size check
            try:
                accuracy_ci, log_loss_ci, auc_roc_ci = self._calculate_bootstrap_ci(
                    y_true_binary, y_pred_prob, bootstrap_samples
                )
            except Exception as e:
                logger.warning(f"Failed to calculate bootstrap CIs: {e}")

        return ClassificationMetrics(
            accuracy=accuracy,
            balanced_accuracy=balanced_accuracy,
            log_loss=log_loss,
            brier_score=brier_score,
            auc_roc=auc_roc,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            accuracy_ci=accuracy_ci,
            log_loss_ci=log_loss_ci,
            auc_roc_ci=auc_roc_ci,
            n_samples=len(y_true),
            n_positive=np.sum(y_true_binary),
            n_negative=len(y_true_binary) - np.sum(y_true_binary),
        )

    def _calculate_regression_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: np.ndarray | None = None,
    ) -> RegressionMetrics:
        """Calculate regression performance metrics."""

        # Basic error metrics
        errors = y_true - y_pred
        abs_errors = np.abs(errors)
        squared_errors = errors**2

        mae = np.average(abs_errors, weights=sample_weights)
        mse = np.average(squared_errors, weights=sample_weights)
        rmse = np.sqrt(mse)

        # R-squared
        if sample_weights is not None:
            y_mean = np.average(y_true, weights=sample_weights)
            ss_tot = np.sum(sample_weights * (y_true - y_mean) ** 2)
            ss_res = np.sum(sample_weights * squared_errors)
        else:
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            ss_res = np.sum(squared_errors)

        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        # Additional metrics
        median_ae = np.median(abs_errors)
        max_error = np.max(abs_errors)
        mean_error = np.average(errors, weights=sample_weights)  # Bias

        return RegressionMetrics(
            mae=mae,
            rmse=rmse,
            mse=mse,
            r2=r2,
            median_ae=median_ae,
            max_error=max_error,
            mean_error=mean_error,
            n_samples=len(y_true),
        )

    def _calculate_calibration_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sample_weights: np.ndarray | None = None,
        n_bins: int = 10,
    ) -> CalibrationMetrics:
        """Calculate model calibration metrics."""

        # Convert to binary if needed
        if y_true.dtype != bool and np.max(y_true) <= 1:
            y_true_binary = (y_true > 0.5).astype(int)
        else:
            y_true_binary = y_true.astype(int)

        # Ensure predictions are probabilities
        y_pred_prob = np.clip(y_pred, 0, 1)

        # Create bins
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2

        # Assign predictions to bins
        bin_indices = np.digitize(y_pred_prob, bin_boundaries) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        # Calculate bin statistics
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []

        for i in range(n_bins):
            mask = bin_indices == i
            count = np.sum(mask)

            if count > 0:
                if sample_weights is not None:
                    bin_weight = np.sum(sample_weights[mask])
                    accuracy = (
                        np.sum(sample_weights[mask] * y_true_binary[mask]) / bin_weight
                    )
                    confidence = (
                        np.sum(sample_weights[mask] * y_pred_prob[mask]) / bin_weight
                    )
                else:
                    accuracy = np.mean(y_true_binary[mask])
                    confidence = np.mean(y_pred_prob[mask])
            else:
                accuracy = 0
                confidence = bin_centers[i]

            bin_accuracies.append(accuracy)
            bin_confidences.append(confidence)
            bin_counts.append(count)

        # Expected Calibration Error (ECE)
        total_samples = len(y_true)
        ece = sum(
            (count / total_samples) * abs(acc - conf)
            for count, acc, conf in zip(
                bin_counts, bin_accuracies, bin_confidences, strict=False
            )
            if count > 0
        )

        # Maximum Calibration Error (MCE)
        mce = (
            max(
                abs(acc - conf)
                for acc, conf, count in zip(
                    bin_accuracies, bin_confidences, bin_counts, strict=False
                )
                if count > 0
            )
            if any(count > 0 for count in bin_counts)
            else 0
        )

        # Average Calibration Error (ACE)
        valid_bins = [
            (acc, conf)
            for acc, conf, count in zip(
                bin_accuracies, bin_confidences, bin_counts, strict=False
            )
            if count > 0
        ]
        ace = (
            np.mean([abs(acc - conf) for acc, conf in valid_bins]) if valid_bins else 0
        )

        # Hosmer-Lemeshow test
        hl_statistic = hl_p_value = None
        try:
            hl_statistic, hl_p_value = self._hosmer_lemeshow_test(
                y_true_binary, y_pred_prob, n_bins
            )
        except Exception as e:
            logger.warning(f"Failed to calculate Hosmer-Lemeshow test: {e}")

        return CalibrationMetrics(
            ece=ece,
            mce=mce,
            ace=ace,
            bin_boundaries=bin_boundaries.tolist(),
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            hl_statistic=hl_statistic,
            hl_p_value=hl_p_value,
            n_samples=len(y_true),
            n_bins=n_bins,
        )

    def _calculate_edge_bucket_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        betting_odds: np.ndarray,
        model_type: ModelType,
    ) -> EdgeBucketMetrics:
        """Calculate edge bucket analysis for betting evaluation."""

        # Convert odds to implied probabilities
        implied_probs = self._odds_to_probability(betting_odds)

        # Calculate edges
        edges = y_pred - implied_probs

        # Define edge buckets
        edge_ranges = [
            (-1.0, -0.05),  # Strong negative edge
            (-0.05, -0.02),  # Moderate negative edge
            (-0.02, 0.02),  # No edge
            (0.02, 0.05),  # Moderate positive edge
            (0.05, 0.10),  # Strong positive edge
            (0.10, 1.0),  # Very strong positive edge
        ]

        bucket_counts = []
        bucket_accuracies = []
        bucket_rois = []
        bucket_profits = []
        bucket_hit_rates = []

        total_profit = 0
        total_bets = 0

        for edge_min, edge_max in edge_ranges:
            mask = (edges >= edge_min) & (edges < edge_max)
            count = np.sum(mask)

            if count > 0:
                bucket_y_true = y_true[mask]
                y_pred[mask]
                bucket_odds = betting_odds[mask]
                edges[mask]

                # Calculate accuracy (hit rate)
                if model_type == ModelType.WIN_PROBABILITY:
                    hits = (bucket_y_true > 0.5).astype(int)
                else:  # ATS or OU
                    hits = (bucket_y_true > 0.5).astype(int)

                accuracy = np.mean(hits)

                # Calculate betting returns
                profits = []
                for i in range(len(bucket_odds)):
                    if hits[i]:
                        # Win: profit = stake * (odds payout - 1)
                        if bucket_odds[i] > 0:
                            profit = bucket_odds[i] / 100  # American odds conversion
                        else:
                            profit = 100 / abs(bucket_odds[i])
                    else:
                        # Bet lost: profit equals negative stake
                        profit = -1

                    profits.append(profit)

                bucket_profit = np.sum(profits)
                bucket_roi = bucket_profit / count if count > 0 else 0

                total_profit += bucket_profit
                total_bets += count
            else:
                accuracy = 0
                bucket_roi = 0
                bucket_profit = 0

            bucket_counts.append(count)
            bucket_accuracies.append(accuracy)
            bucket_rois.append(bucket_roi)
            bucket_profits.append(bucket_profit)
            bucket_hit_rates.append(accuracy)

        # Overall metrics
        total_roi = total_profit / total_bets if total_bets > 0 else 0
        average_edge = np.mean(edges)

        # Sharpe ratio (simplified)
        if len(edges) > 1:
            edge_returns = []
            for i, _edge in enumerate(edges):
                if y_true[i] > 0.5:  # Win
                    if betting_odds[i] > 0:
                        return_val = betting_odds[i] / 100
                    else:
                        return_val = 100 / abs(betting_odds[i])
                else:  # Loss
                    return_val = -1
                edge_returns.append(return_val)

            returns_std = np.std(edge_returns)
            sharpe_ratio = np.mean(edge_returns) / returns_std if returns_std > 0 else 0
        else:
            sharpe_ratio = None

        # Best/worst buckets
        valid_rois = [
            (i, roi) for i, roi in enumerate(bucket_rois) if bucket_counts[i] > 0
        ]
        best_roi_bucket = max(valid_rois, key=lambda x: x[1])[0] if valid_rois else None
        worst_roi_bucket = (
            min(valid_rois, key=lambda x: x[1])[0] if valid_rois else None
        )

        return EdgeBucketMetrics(
            edge_ranges=edge_ranges,
            bucket_counts=bucket_counts,
            bucket_accuracies=bucket_accuracies,
            bucket_rois=bucket_rois,
            bucket_profits=bucket_profits,
            bucket_hit_rates=bucket_hit_rates,
            total_bets=total_bets,
            total_profit=total_profit,
            total_roi=total_roi,
            average_edge=average_edge,
            sharpe_ratio=sharpe_ratio,
            best_roi_bucket=best_roi_bucket,
            worst_roi_bucket=worst_roi_bucket,
        )

    def _calculate_significance_tests(
        self, y_true: np.ndarray, y_pred: np.ndarray, model_type: ModelType
    ) -> list[SignificanceTest]:
        """Calculate statistical significance tests."""

        tests = []

        # 1. Test if accuracy is significantly different from random (50%)
        if model_type == ModelType.WIN_PROBABILITY:
            y_true_binary = (y_true > 0.5).astype(int)
            y_pred_binary = (y_pred > 0.5).astype(int)

            # Binomial test for accuracy vs 50%
            n_correct = np.sum(y_true_binary == y_pred_binary)
            n_total = len(y_true_binary)

            # Two-tailed binomial test
            p_value = 2 * min(
                stats.binom.cdf(n_correct, n_total, 0.5),
                1 - stats.binom.cdf(n_correct - 1, n_total, 0.5),
            )

            accuracy = n_correct / n_total
            is_significant = p_value < self.alpha

            tests.append(
                SignificanceTest(
                    test_name="Binomial Test (Accuracy vs Random)",
                    statistic=n_correct,
                    p_value=p_value,
                    is_significant=is_significant,
                    confidence_level=self.confidence_level,
                    effect_size=abs(accuracy - 0.5),
                    test_details={
                        "n_correct": int(n_correct),
                        "n_total": int(n_total),
                        "observed_accuracy": accuracy,
                        "null_hypothesis": 0.5,
                    },
                )
            )

        # 2. Kolmogorov-Smirnov test for distribution comparison
        try:
            # Test if predictions follow expected distribution
            uniform_sample = np.random.uniform(0, 1, len(y_pred))
            ks_stat, ks_p = stats.ks_2samp(y_pred, uniform_sample)

            tests.append(
                SignificanceTest(
                    test_name="Kolmogorov-Smirnov Test (Prediction Distribution)",
                    statistic=ks_stat,
                    p_value=ks_p,
                    is_significant=ks_p < self.alpha,
                    confidence_level=self.confidence_level,
                    test_details={"distribution": "uniform"},
                )
            )
        except Exception as e:
            logger.warning(f"Failed KS test: {e}")

        # 3. McNemar's test for paired predictions (if comparing two models)
        # This would be implemented when comparing models

        return tests

    def _odds_to_probability(self, american_odds: np.ndarray) -> np.ndarray:
        """Convert American odds to implied probabilities."""

        probabilities = np.zeros_like(american_odds, dtype=float)

        # Positive odds
        positive_mask = american_odds > 0
        probabilities[positive_mask] = 100 / (american_odds[positive_mask] + 100)

        # Negative odds
        negative_mask = american_odds < 0
        probabilities[negative_mask] = -american_odds[negative_mask] / (
            -american_odds[negative_mask] + 100
        )

        return probabilities

    def _hosmer_lemeshow_test(
        self, y_true: np.ndarray, y_pred: np.ndarray, n_bins: int = 10
    ) -> tuple[float, float]:
        """Perform Hosmer-Lemeshow goodness-of-fit test."""

        # Sort by predicted probabilities
        sorted_indices = np.argsort(y_pred)
        y_true_sorted = y_true[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]

        # Divide into bins
        bin_size = len(y_true) // n_bins
        chi_square = 0

        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(y_true)

            bin_y_true = y_true_sorted[start_idx:end_idx]
            bin_y_pred = y_pred_sorted[start_idx:end_idx]

            observed = np.sum(bin_y_true)
            expected = np.sum(bin_y_pred)

            if expected > 0 and expected < len(bin_y_true):
                chi_square += ((observed - expected) ** 2) / (
                    expected * (1 - expected / len(bin_y_true))
                )

        # Chi-square test with (n_bins - 2) degrees of freedom
        df = n_bins - 2
        p_value = 1 - stats.chi2.cdf(chi_square, df)

        return chi_square, p_value

    def _calculate_overall_score(self, metrics: ComprehensiveMetrics) -> float:
        """Calculate an overall performance score."""

        score_components = []

        # Classification metrics
        if metrics.classification_metrics:
            # Use balanced accuracy instead of regular accuracy to account for class imbalance
            balanced_accuracy_score = (
                metrics.classification_metrics.balanced_accuracy * 100
            )
            score_components.append(("balanced_accuracy", balanced_accuracy_score, 0.3))

            # Log loss component (lower is better, convert to 0-100 scale)
            log_loss_score = max(0, 100 - metrics.classification_metrics.log_loss * 100)
            score_components.append(("log_loss", log_loss_score, 0.25))

            # Brier score component (lower is better, convert to 0-100 scale)
            brier_score = max(0, 100 - metrics.classification_metrics.brier_score * 100)
            score_components.append(("brier", brier_score, 0.2))

        # Calibration metrics
        if metrics.calibration_metrics:
            # ECE component (lower is better)
            ece_score = max(0, 100 - metrics.calibration_metrics.ece * 500)  # Scale ECE
            score_components.append(("ece", ece_score, 0.15))

        # Betting metrics
        if metrics.edge_bucket_metrics:
            # ROI component
            roi = metrics.edge_bucket_metrics.total_roi
            roi_score = 50 + roi * 500  # Center at 50, scale by ROI
            roi_score = max(0, min(100, roi_score))
            score_components.append(("roi", roi_score, 0.1))

        # Calculate weighted average
        if score_components:
            weighted_sum = sum(score * weight for _, score, weight in score_components)
            total_weight = sum(weight for _, _, weight in score_components)
            overall_score = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            overall_score = 0

        return overall_score

    def compare_models(
        self,
        metrics_a: ComprehensiveMetrics,
        metrics_b: ComprehensiveMetrics,
        y_true: np.ndarray,
        y_pred_a: np.ndarray,
        y_pred_b: np.ndarray,
    ) -> dict[str, Any]:
        """Compare two models statistically."""

        comparison_results = {
            "model_a_score": metrics_a.overall_score,
            "model_b_score": metrics_b.overall_score,
            "score_difference": metrics_a.overall_score - metrics_b.overall_score,
            "significance_tests": [],
        }

        # McNemar's test for paired binary predictions
        if metrics_a.classification_metrics and metrics_b.classification_metrics:
            y_true_binary = (y_true > 0.5).astype(int)
            y_pred_a_binary = (y_pred_a > 0.5).astype(int)
            y_pred_b_binary = (y_pred_b > 0.5).astype(int)

            # Create contingency table
            a_correct_b_wrong = np.sum(
                (y_true_binary == y_pred_a_binary) & (y_true_binary != y_pred_b_binary)
            )
            a_wrong_b_correct = np.sum(
                (y_true_binary != y_pred_a_binary) & (y_true_binary == y_pred_b_binary)
            )

            if a_correct_b_wrong + a_wrong_b_correct > 0:
                mcnemar_stat = (abs(a_correct_b_wrong - a_wrong_b_correct) - 1) ** 2 / (
                    a_correct_b_wrong + a_wrong_b_correct
                )
                mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)

                comparison_results["significance_tests"].append(
                    {
                        "test_name": "McNemar's Test",
                        "statistic": mcnemar_stat,
                        "p_value": mcnemar_p,
                        "is_significant": mcnemar_p < self.alpha,
                        "interpretation": "Model A significantly better"
                        if mcnemar_p < self.alpha
                        and a_correct_b_wrong > a_wrong_b_correct
                        else "Model B significantly better"
                        if mcnemar_p < self.alpha
                        and a_wrong_b_correct > a_correct_b_wrong
                        else "No significant difference",
                    }
                )

        return comparison_results

    def _calculate_bootstrap_ci(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        n_samples: int = 1000,
        confidence_level: float = 0.95,
    ) -> tuple[
        tuple[float, float] | None,
        tuple[float, float] | None,
        tuple[float, float] | None,
    ]:
        """Calculate bootstrap confidence intervals for key metrics."""

        # Ensure we have sklearn for AUC calculation
        try:
            from sklearn.metrics import roc_auc_score

            has_sklearn = True
        except ImportError:
            has_sklearn = False

        # Initialize arrays to store bootstrap results
        bootstrap_accuracies = []
        bootstrap_log_losses = []
        bootstrap_aucs = [] if has_sklearn and len(np.unique(y_true)) > 1 else None

        n_data = len(y_true)

        for _ in range(n_samples):
            # Bootstrap sample with replacement
            indices = np.random.choice(n_data, size=n_data, replace=True)
            y_true_boot = y_true[indices]
            y_pred_boot = y_pred[indices]

            # Calculate metrics for this bootstrap sample
            try:
                # Accuracy
                y_pred_binary_boot = (y_pred_boot > 0.5).astype(int)
                accuracy_boot = np.mean(y_true_boot == y_pred_binary_boot)
                bootstrap_accuracies.append(accuracy_boot)

                # Log loss (with clipping for numerical stability)
                y_pred_clipped = np.clip(y_pred_boot, 1e-15, 1 - 1e-15)
                log_loss_boot = -np.mean(
                    y_true_boot * np.log(y_pred_clipped)
                    + (1 - y_true_boot) * np.log(1 - y_pred_clipped)
                )
                bootstrap_log_losses.append(log_loss_boot)

                # AUC (if sklearn available and both classes present)
                if bootstrap_aucs is not None and len(np.unique(y_true_boot)) > 1:
                    try:
                        auc_boot = roc_auc_score(y_true_boot, y_pred_boot)
                        bootstrap_aucs.append(auc_boot)
                    except ValueError:
                        # Skip this sample if AUC calculation fails
                        pass

            except Exception:
                # Skip problematic bootstrap samples
                continue

        # Calculate confidence intervals using percentile method
        alpha = 1 - confidence_level
        lower_percentile = (alpha / 2) * 100
        upper_percentile = (1 - alpha / 2) * 100

        # Accuracy CI
        accuracy_ci = None
        if bootstrap_accuracies:
            accuracy_ci = (
                np.percentile(bootstrap_accuracies, lower_percentile),
                np.percentile(bootstrap_accuracies, upper_percentile),
            )

        # Log loss CI
        log_loss_ci = None
        if bootstrap_log_losses:
            log_loss_ci = (
                np.percentile(bootstrap_log_losses, lower_percentile),
                np.percentile(bootstrap_log_losses, upper_percentile),
            )

        # AUC CI
        auc_ci = None
        if bootstrap_aucs and len(bootstrap_aucs) > 10:  # Need sufficient samples
            auc_ci = (
                np.percentile(bootstrap_aucs, lower_percentile),
                np.percentile(bootstrap_aucs, upper_percentile),
            )

        return accuracy_ci, log_loss_ci, auc_ci


def create_metrics_summary(metrics: ComprehensiveMetrics) -> dict[str, Any]:
    """Create human-readable summary of metrics."""

    summary = {
        "model_type": metrics.model_type.value,
        "overall_score": round(metrics.overall_score, 2)
        if metrics.overall_score
        else None,
        "sample_size": metrics.n_total_samples,
    }

    # Classification metrics
    if metrics.classification_metrics:
        cm = metrics.classification_metrics
        classification_summary = {
            "accuracy": round(cm.accuracy, 4),
            "balanced_accuracy": round(cm.balanced_accuracy, 4),
            "log_loss": round(cm.log_loss, 4),
            "brier_score": round(cm.brier_score, 4),
            "auc_roc": round(cm.auc_roc, 4) if cm.auc_roc else None,
        }

        # Add confidence intervals if available
        if cm.accuracy_ci:
            classification_summary["accuracy_ci"] = [
                round(x, 4) for x in cm.accuracy_ci
            ]
        if cm.log_loss_ci:
            classification_summary["log_loss_ci"] = [
                round(x, 4) for x in cm.log_loss_ci
            ]
        if cm.auc_roc_ci:
            classification_summary["auc_roc_ci"] = [round(x, 4) for x in cm.auc_roc_ci]

        summary["classification"] = classification_summary

    # Regression metrics
    if metrics.regression_metrics:
        rm = metrics.regression_metrics
        summary["regression"] = {
            "mae": round(rm.mae, 4),
            "rmse": round(rm.rmse, 4),
            "r2": round(rm.r2, 4),
            "bias": round(rm.mean_error, 4),
        }

    # Calibration metrics
    if metrics.calibration_metrics:
        cal = metrics.calibration_metrics
        summary["calibration"] = {
            "ece": round(cal.ece, 4),
            "mce": round(cal.mce, 4),
            "ace": round(cal.ace, 4),
            "hosmer_lemeshow_p": round(cal.hl_p_value, 4) if cal.hl_p_value else None,
        }

    # Betting metrics
    if metrics.edge_bucket_metrics:
        eb = metrics.edge_bucket_metrics
        summary["betting"] = {
            "total_roi": round(eb.total_roi, 4),
            "total_profit": round(eb.total_profit, 2),
            "sharpe_ratio": round(eb.sharpe_ratio, 4) if eb.sharpe_ratio else None,
            "total_bets": eb.total_bets,
        }

    # Significance tests
    summary["significance_tests"] = [
        {
            "test": test.test_name,
            "p_value": round(test.p_value, 6),
            "significant": test.is_significant,
        }
        for test in metrics.significance_tests
    ]

    return summary
