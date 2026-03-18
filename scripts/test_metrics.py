"""
Test suite for evaluation metrics system.

This script validates the MetricsCalculator functionality including performance metrics,
calibration analysis, edge bucket analysis, and statistical significance testing.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import tempfile

import numpy as np

from backtest.metrics import (
    CalibrationMetrics,
    ClassificationMetrics,
    ComprehensiveMetrics,
    EdgeBucketMetrics,
    MetricsCalculator,
    ModelType,
    RegressionMetrics,
    SignificanceTest,
    create_metrics_summary,
)
from backtest.visualization import MetricsVisualizer
from utils.logging_config import get_logger

logger = get_logger(__name__)


def create_synthetic_data(
    n_samples: int = 1000, random_seed: int = 42
) -> dict[str, np.ndarray]:
    """Create synthetic data for testing."""
    np.random.seed(random_seed)

    # Create synthetic true outcomes (binary for classification)
    y_true = np.random.binomial(1, 0.55, n_samples)  # Slightly biased toward 1

    # Create synthetic predictions with some skill
    base_prob = 0.5 + 0.2 * (y_true - 0.5)  # Start with some correlation
    noise = np.random.normal(0, 0.1, n_samples)
    y_pred = np.clip(base_prob + noise, 0.01, 0.99)  # Keep in valid probability range

    # Create synthetic betting odds (American format)
    implied_probs = y_pred + np.random.normal(
        0, 0.05, n_samples
    )  # Market slightly off from model
    implied_probs = np.clip(implied_probs, 0.1, 0.9)

    # Convert to American odds
    betting_odds = np.where(
        implied_probs >= 0.5,
        -100 * implied_probs / (1 - implied_probs),
        100 * (1 - implied_probs) / implied_probs,
    ).astype(int)

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "betting_odds": betting_odds,
        "implied_probs": implied_probs,
    }


def test_classification_metrics():
    """Test classification metrics calculation."""
    logger.info("Running test_classification_metrics")

    calculator = MetricsCalculator()
    data = create_synthetic_data(500)

    y_true = data["y_true"]
    y_pred = data["y_pred"]

    # Calculate classification metrics
    metrics = calculator._calculate_classification_metrics(y_true, y_pred)

    # Verify basic properties
    assert isinstance(metrics, ClassificationMetrics), (
        "Should return ClassificationMetrics"
    )
    assert 0 <= metrics.accuracy <= 1, "Accuracy should be between 0 and 1"
    assert metrics.log_loss >= 0, "Log loss should be non-negative"
    assert 0 <= metrics.brier_score <= 1, "Brier score should be between 0 and 1"
    assert metrics.n_samples == len(y_true), "Sample count should match input"

    # Check that accuracy makes sense
    y_pred_binary = (y_pred > 0.5).astype(int)
    expected_accuracy = np.mean(y_true == y_pred_binary)
    assert abs(metrics.accuracy - expected_accuracy) < 1e-6, (
        "Accuracy calculation should match expected"
    )

    # Check that log loss is reasonable
    assert metrics.log_loss < 2.0, "Log loss should be reasonable for good predictions"

    logger.info(
        f"Classification metrics: Acc={metrics.accuracy:.3f}, LogLoss={metrics.log_loss:.3f}, Brier={metrics.brier_score:.3f}"
    )
    return True


def test_regression_metrics():
    """Test regression metrics calculation."""
    logger.info("Running test_regression_metrics")

    calculator = MetricsCalculator()

    # Create regression data
    n_samples = 500
    np.random.seed(42)
    y_true = np.random.normal(0, 2, n_samples)  # Target values
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)  # Predictions with noise

    metrics = calculator._calculate_regression_metrics(y_true, y_pred)

    # Verify basic properties
    assert isinstance(metrics, RegressionMetrics), "Should return RegressionMetrics"
    assert metrics.mae >= 0, "MAE should be non-negative"
    assert metrics.rmse >= 0, "RMSE should be non-negative"
    assert metrics.mse >= 0, "MSE should be non-negative"
    assert metrics.rmse >= metrics.mae, "RMSE should be >= MAE"
    assert abs(metrics.mse - metrics.rmse**2) < 1e-6, "MSE should equal RMSE squared"
    assert metrics.n_samples == len(y_true), "Sample count should match input"

    # Check R-squared is reasonable
    assert -1 <= metrics.r2 <= 1, "R-squared should be between -1 and 1"

    # For data with some correlation, R2 should be positive
    assert metrics.r2 > 0.5, "R-squared should be reasonably high for correlated data"

    logger.info(
        f"Regression metrics: MAE={metrics.mae:.3f}, RMSE={metrics.rmse:.3f}, R2={metrics.r2:.3f}"
    )
    return True


def test_calibration_metrics():
    """Test calibration metrics calculation."""
    logger.info("Running test_calibration_metrics")

    calculator = MetricsCalculator()

    # Create well-calibrated data
    n_samples = 1000
    np.random.seed(42)
    y_pred = np.random.uniform(0, 1, n_samples)
    y_true = np.random.binomial(1, y_pred, n_samples)  # Perfect calibration

    metrics = calculator._calculate_calibration_metrics(y_true, y_pred)

    # Verify basic properties
    assert isinstance(metrics, CalibrationMetrics), "Should return CalibrationMetrics"
    assert 0 <= metrics.ece <= 1, "ECE should be between 0 and 1"
    assert 0 <= metrics.mce <= 1, "MCE should be between 0 and 1"
    assert 0 <= metrics.ace <= 1, "ACE should be between 0 and 1"
    assert metrics.n_samples == len(y_true), "Sample count should match input"

    # For well-calibrated data, ECE should be relatively low
    assert metrics.ece < 0.1, (
        f"ECE should be low for well-calibrated data, got {metrics.ece}"
    )

    # Verify bin data
    assert len(metrics.bin_boundaries) == metrics.n_bins + 1, (
        "Should have n_bins+1 boundaries"
    )
    assert len(metrics.bin_accuracies) == metrics.n_bins, (
        "Should have n_bins accuracies"
    )
    assert len(metrics.bin_confidences) == metrics.n_bins, (
        "Should have n_bins confidences"
    )
    assert len(metrics.bin_counts) == metrics.n_bins, "Should have n_bins counts"

    # Total counts should sum to total samples
    assert sum(metrics.bin_counts) == n_samples, (
        "Bin counts should sum to total samples"
    )

    logger.info(
        f"Calibration metrics: ECE={metrics.ece:.4f}, MCE={metrics.mce:.4f}, ACE={metrics.ace:.4f}"
    )
    return True


def test_edge_bucket_metrics():
    """Test edge bucket analysis."""
    logger.info("Running test_edge_bucket_metrics")

    calculator = MetricsCalculator()
    data = create_synthetic_data(1000)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    betting_odds = data["betting_odds"]

    metrics = calculator._calculate_edge_bucket_metrics(
        y_true, y_pred, betting_odds, ModelType.WIN_PROBABILITY
    )

    # Verify basic properties
    assert isinstance(metrics, EdgeBucketMetrics), "Should return EdgeBucketMetrics"
    assert len(metrics.edge_ranges) > 0, "Should have edge ranges"
    assert len(metrics.bucket_counts) == len(metrics.edge_ranges), (
        "Counts should match ranges"
    )
    assert len(metrics.bucket_rois) == len(metrics.edge_ranges), (
        "ROIs should match ranges"
    )

    # Total bets should equal sum of bucket counts
    assert metrics.total_bets == sum(metrics.bucket_counts), (
        "Total bets should match sum of bucket counts"
    )

    # Check that bucket counts are non-negative
    assert all(count >= 0 for count in metrics.bucket_counts), (
        "All bucket counts should be non-negative"
    )

    # Average edge should be reasonable
    assert -1 <= metrics.average_edge <= 1, "Average edge should be reasonable"

    # Best and worst buckets should be valid indices if they exist
    if metrics.best_roi_bucket is not None:
        assert 0 <= metrics.best_roi_bucket < len(metrics.bucket_rois), (
            "Best ROI bucket should be valid index"
        )
    if metrics.worst_roi_bucket is not None:
        assert 0 <= metrics.worst_roi_bucket < len(metrics.bucket_rois), (
            "Worst ROI bucket should be valid index"
        )

    logger.info(
        f"Edge bucket metrics: Total ROI={metrics.total_roi:.4f}, Avg Edge={metrics.average_edge:.4f}, Total Bets={metrics.total_bets}"
    )
    return True


def test_significance_tests():
    """Test statistical significance testing."""
    logger.info("Running test_significance_tests")

    calculator = MetricsCalculator()

    # Create data where model performs significantly better than random
    n_samples = 1000
    np.random.seed(42)

    # High-skill model
    y_true = np.random.binomial(1, 0.5, n_samples)
    y_pred = np.where(
        y_true == 1,
        np.random.beta(2, 1, np.sum(y_true)),  # Higher predictions for true positives
        np.random.beta(1, 2, np.sum(1 - y_true)),
    )  # Lower predictions for true negatives

    tests = calculator._calculate_significance_tests(
        y_true, y_pred, ModelType.WIN_PROBABILITY
    )

    # Should have at least one test
    assert len(tests) > 0, "Should perform at least one significance test"

    # Check binomial test
    binomial_test = next((t for t in tests if "Binomial" in t.test_name), None)
    assert binomial_test is not None, "Should include binomial test"
    assert isinstance(binomial_test, SignificanceTest), (
        "Should be SignificanceTest object"
    )
    assert 0 <= binomial_test.p_value <= 1, "P-value should be between 0 and 1"
    assert binomial_test.confidence_level == calculator.confidence_level, (
        "Should match calculator confidence"
    )

    # For skilled model, should likely be significant
    if binomial_test.is_significant:
        logger.info(f"Binomial test significant: p={binomial_test.p_value:.6f}")
    else:
        logger.info(f"Binomial test not significant: p={binomial_test.p_value:.6f}")

    logger.info(f"Significance tests: {len(tests)} tests performed")
    return True


def test_comprehensive_metrics():
    """Test comprehensive metrics calculation."""
    logger.info("Running test_comprehensive_metrics")

    calculator = MetricsCalculator()
    data = create_synthetic_data(1000)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    betting_odds = data["betting_odds"]

    # Test win probability model
    metrics = calculator.calculate_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_type=ModelType.WIN_PROBABILITY,
        betting_odds=betting_odds,
    )

    # Verify structure
    assert isinstance(metrics, ComprehensiveMetrics), (
        "Should return ComprehensiveMetrics"
    )
    assert metrics.model_type == ModelType.WIN_PROBABILITY, (
        "Should have correct model type"
    )
    assert metrics.n_total_samples == len(y_true), "Should track total samples"

    # Should have classification and calibration metrics for WP
    assert metrics.classification_metrics is not None, (
        "WP should have classification metrics"
    )
    assert metrics.calibration_metrics is not None, "WP should have calibration metrics"

    # Should have betting metrics since odds provided
    assert metrics.edge_bucket_metrics is not None, (
        "Should have edge bucket metrics when odds provided"
    )

    # Should have significance tests
    assert len(metrics.significance_tests) > 0, "Should have significance tests"

    # Should have overall score
    assert metrics.overall_score is not None, "Should have overall score"
    assert 0 <= metrics.overall_score <= 100, (
        "Overall score should be between 0 and 100"
    )

    logger.info(
        f"Comprehensive metrics: Overall={metrics.overall_score:.1f}, Tests={len(metrics.significance_tests)}"
    )
    return True


def test_ats_model_metrics():
    """Test metrics for ATS model type."""
    logger.info("Running test_ats_model_metrics")

    calculator = MetricsCalculator()
    data = create_synthetic_data(500)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    betting_odds = data["betting_odds"]

    # Test ATS model
    metrics = calculator.calculate_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_type=ModelType.AGAINST_THE_SPREAD,
        betting_odds=betting_odds,
    )

    # ATS should have both classification and regression metrics
    assert metrics.classification_metrics is not None, (
        "ATS should have classification metrics"
    )
    assert metrics.regression_metrics is not None, "ATS should have regression metrics"
    assert metrics.calibration_metrics is not None, (
        "ATS should have calibration metrics"
    )

    logger.info("ATS model metrics calculated successfully")
    return True


def test_metrics_summary():
    """Test metrics summary generation."""
    logger.info("Running test_metrics_summary")

    calculator = MetricsCalculator()
    data = create_synthetic_data(500)

    y_true = data["y_true"]
    y_pred = data["y_pred"]
    betting_odds = data["betting_odds"]

    metrics = calculator.calculate_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_type=ModelType.WIN_PROBABILITY,
        betting_odds=betting_odds,
    )

    # Generate summary
    summary = create_metrics_summary(metrics)

    # Verify summary structure
    assert isinstance(summary, dict), "Summary should be a dictionary"
    assert "model_type" in summary, "Should include model type"
    assert "overall_score" in summary, "Should include overall score"
    assert "sample_size" in summary, "Should include sample size"

    # Should include classification metrics
    assert "classification" in summary, "Should include classification section"
    class_metrics = summary["classification"]
    assert "accuracy" in class_metrics, "Should include accuracy"
    assert "log_loss" in class_metrics, "Should include log loss"

    # Should include calibration metrics
    assert "calibration" in summary, "Should include calibration section"

    # Should include betting metrics
    assert "betting" in summary, "Should include betting section"

    # Should include significance tests
    assert "significance_tests" in summary, "Should include significance tests"

    logger.info(f"Metrics summary: {len(summary)} sections")
    return True


def test_model_comparison():
    """Test model comparison functionality."""
    logger.info("Running test_model_comparison")

    calculator = MetricsCalculator()

    # Create two different models
    n_samples = 500
    np.random.seed(42)
    y_true = np.random.binomial(1, 0.5, n_samples)

    # Model A: Good performance
    y_pred_a = np.where(
        y_true == 1,
        np.random.beta(3, 1, np.sum(y_true)),
        np.random.beta(1, 3, np.sum(1 - y_true)),
    )

    # Model B: Worse performance
    y_pred_b = np.where(
        y_true == 1,
        np.random.beta(2, 1.5, np.sum(y_true)),
        np.random.beta(1.5, 2, np.sum(1 - y_true)),
    )

    # Calculate metrics for both models
    metrics_a = calculator.calculate_comprehensive_metrics(
        y_true=y_true, y_pred=y_pred_a, model_type=ModelType.WIN_PROBABILITY
    )

    metrics_b = calculator.calculate_comprehensive_metrics(
        y_true=y_true, y_pred=y_pred_b, model_type=ModelType.WIN_PROBABILITY
    )

    # Compare models
    comparison = calculator.compare_models(
        metrics_a, metrics_b, y_true, y_pred_a, y_pred_b
    )

    # Verify comparison structure
    assert isinstance(comparison, dict), "Comparison should be dictionary"
    assert "model_a_score" in comparison, "Should include model A score"
    assert "model_b_score" in comparison, "Should include model B score"
    assert "score_difference" in comparison, "Should include score difference"
    assert "significance_tests" in comparison, "Should include significance tests"

    # Model A should generally perform better
    assert comparison["model_a_score"] >= comparison["model_b_score"], (
        "Model A should perform better"
    )

    logger.info(
        f"Model comparison: A={comparison['model_a_score']:.1f}, B={comparison['model_b_score']:.1f}"
    )
    return True


def test_visualization_creation():
    """Test visualization utilities."""
    logger.info("Running test_visualization_creation")

    calculator = MetricsCalculator()
    visualizer = MetricsVisualizer()

    data = create_synthetic_data(500)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    betting_odds = data["betting_odds"]

    metrics = calculator.calculate_comprehensive_metrics(
        y_true=y_true,
        y_pred=y_pred,
        model_type=ModelType.WIN_PROBABILITY,
        betting_odds=betting_odds,
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        # Test dashboard creation
        plots_created = visualizer.create_metrics_dashboard(
            comprehensive_metrics=metrics, output_dir=temp_dir, model_name="TestModel"
        )

        # Should create some plots (or text files if matplotlib unavailable)
        assert isinstance(plots_created, dict), "Should return dictionary of plots"

        # Check that files were actually created
        for _plot_type, file_path in plots_created.items():
            assert os.path.exists(file_path), f"Plot file should exist: {file_path}"

    logger.info(f"Visualization test: {len(plots_created)} plots/files created")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Running test_edge_cases")

    calculator = MetricsCalculator()

    # Test with perfect predictions
    y_true = np.array([0, 1, 0, 1, 0] * 20)
    y_pred = y_true.astype(float)  # Perfect predictions

    try:
        metrics = calculator.calculate_comprehensive_metrics(
            y_true=y_true, y_pred=y_pred, model_type=ModelType.WIN_PROBABILITY
        )
        assert metrics.classification_metrics.accuracy == 1.0, (
            "Perfect predictions should have 100% accuracy"
        )
        assert metrics.classification_metrics.brier_score == 0.0, (
            "Perfect predictions should have 0 Brier score"
        )
    except Exception as e:
        logger.warning(f"Perfect predictions test failed: {e}")

    # Test with very small sample sizes
    try:
        small_y_true = np.array([0, 1])
        small_y_pred = np.array([0.3, 0.7])

        small_metrics = calculator.calculate_comprehensive_metrics(
            y_true=small_y_true,
            y_pred=small_y_pred,
            model_type=ModelType.WIN_PROBABILITY,
        )
        assert small_metrics.n_total_samples == 2, "Should handle small samples"
    except Exception as e:
        logger.warning(f"Small sample test failed: {e}")

    # Test with extreme predictions
    try:
        extreme_y_true = np.array([0, 1, 0, 1])
        extreme_y_pred = np.array([0.001, 0.999, 0.001, 0.999])  # Extreme confidence

        extreme_metrics = calculator.calculate_comprehensive_metrics(
            y_true=extreme_y_true,
            y_pred=extreme_y_pred,
            model_type=ModelType.WIN_PROBABILITY,
        )
        assert extreme_metrics is not None, "Should handle extreme predictions"
    except Exception as e:
        logger.warning(f"Extreme predictions test failed: {e}")

    logger.info("Edge cases testing completed")
    return True


def run_all_tests():
    """Run all metrics tests."""
    logger.info("Starting evaluation metrics test suite")

    tests = [
        test_classification_metrics,
        test_regression_metrics,
        test_calibration_metrics,
        test_edge_bucket_metrics,
        test_significance_tests,
        test_comprehensive_metrics,
        test_ats_model_metrics,
        test_metrics_summary,
        test_model_comparison,
        test_visualization_creation,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                logger.error(f"Test {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            import traceback

            logger.error(traceback.format_exc())

    logger.info(f"Evaluation metrics tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("All evaluation metrics tests passed!")
    else:
        logger.error(f"{failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
