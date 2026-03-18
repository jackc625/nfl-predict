#!/usr/bin/env python3
"""
Test script for model evaluation framework.

This script validates all components of the evaluation framework including:
- Core metrics (LogLoss, Brier, MAE, RMSE)
- Calibration metrics (ECE, reliability)
- Betting simulation metrics (ROI, CLV)
- Statistical tests and comparisons
"""

import sys
import tempfile
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.evaluation import (
    ModelEvaluationFramework,
)
from utils import get_logger

logger = get_logger(__name__)


def create_synthetic_predictions(
    n_samples: int = 1000, prediction_type: str = "binary", noise_level: float = 0.1
) -> dict:
    """
    Create synthetic predictions for testing.

    Args:
        n_samples: Number of samples to generate
        prediction_type: "binary" or "regression"
        noise_level: Amount of noise to add

    Returns:
        Dictionary with predictions, outcomes, and market data
    """
    np.random.seed(42)  # For reproducible tests

    if prediction_type == "binary":
        # Generate underlying probabilities
        true_probs = np.random.beta(
            2, 2, n_samples
        )  # Beta distribution for realistic probabilities

        # Add model noise
        predicted_probs = true_probs + np.random.normal(0, noise_level, n_samples)
        predicted_probs = np.clip(
            predicted_probs, 0.01, 0.99
        )  # Clip to valid probability range

        # Generate outcomes based on true probabilities
        outcomes = np.random.binomial(1, true_probs, n_samples)

        # Generate market odds (slightly worse than true probabilities)
        market_bias = np.random.normal(
            0.05, 0.02, n_samples
        )  # Market slightly favors house
        market_probs = np.clip(true_probs + market_bias, 0.01, 0.99)

        # Convert to American odds
        def prob_to_american_odds(prob):
            if prob >= 0.5:
                return int(-100 * prob / (1 - prob))
            return int(100 * (1 - prob) / prob)

        market_odds = np.array([prob_to_american_odds(p) for p in market_probs])

        # Generate closing odds (slightly different from opening)
        closing_probs = market_probs + np.random.normal(0, 0.01, n_samples)
        closing_probs = np.clip(closing_probs, 0.01, 0.99)
        closing_odds = np.array([prob_to_american_odds(p) for p in closing_probs])

        return {
            "predictions": predicted_probs,
            "outcomes": outcomes,
            "market_odds": market_odds,
            "closing_odds": closing_odds,
            "true_probs": true_probs,
        }

    # regression
    # Generate true values
    true_values = np.random.normal(45, 15, n_samples)  # Total points around 45

    # Add model noise
    predictions = true_values + np.random.normal(0, noise_level * 10, n_samples)

    # Generate actual outcomes with some noise
    outcomes = true_values + np.random.normal(0, 8, n_samples)  # Natural game variation

    return {
        "predictions": predictions,
        "outcomes": outcomes,
        "true_values": true_values,
    }


def test_classification_metrics():
    """Test classification metrics (LogLoss, Brier, accuracy)."""
    logger.info("Testing classification metrics...")

    print("\n1. Testing Classification Metrics:")

    # Create synthetic data
    data = create_synthetic_predictions(1000, "binary", noise_level=0.1)

    # Initialize evaluator
    evaluator = ModelEvaluationFramework(
        n_calibration_bins=10, betting_bankroll=10000, min_edge_threshold=0.02
    )

    # Evaluate model
    metrics = evaluator.evaluate_model(
        predictions=data["predictions"],
        actual_outcomes=data["outcomes"],
        market_odds=data["market_odds"],
        closing_odds=data["closing_odds"],
        prediction_type="binary",
        model_name="Test Model",
    )

    print(f"  Predictions: {len(data['predictions'])}")
    print(f"  Accuracy: {metrics.core_metrics['accuracy']:.3f}")
    print(f"  Log Loss: {metrics.core_metrics['log_loss']:.3f}")
    print(f"  Brier Score: {metrics.core_metrics['brier_score']:.3f}")

    # Validate metrics
    if not (0 <= metrics.core_metrics["accuracy"] <= 1):
        print("  [ERROR] Accuracy outside valid range")
        return False

    if metrics.core_metrics["log_loss"] < 0:
        print("  [ERROR] Log Loss should be non-negative")
        return False

    if not (0 <= metrics.core_metrics["brier_score"] <= 1):
        print("  [ERROR] Brier Score outside valid range")
        return False

    # Test with perfect predictions
    print("\n  Testing with perfect predictions:")
    perfect_preds = data["true_probs"]
    perfect_metrics = evaluator.evaluate_model(
        predictions=perfect_preds,
        actual_outcomes=data["outcomes"],
        prediction_type="binary",
        model_name="Perfect Model",
    )

    print(f"  Perfect model accuracy: {perfect_metrics.core_metrics['accuracy']:.3f}")
    print(f"  Perfect model log loss: {perfect_metrics.core_metrics['log_loss']:.3f}")

    # Perfect predictions should have better metrics
    if perfect_metrics.core_metrics["accuracy"] <= metrics.core_metrics["accuracy"]:
        print("  [WARN] Perfect model should have higher accuracy")

    if perfect_metrics.core_metrics["log_loss"] >= metrics.core_metrics["log_loss"]:
        print("  [WARN] Perfect model should have lower log loss")

    print("  [OK] All classification metrics computed successfully")
    print("[PASS] Classification metrics tests passed")
    return True


def test_regression_metrics():
    """Test regression metrics (MAE, RMSE, R²)."""
    logger.info("Testing regression metrics...")

    print("\n2. Testing Regression Metrics:")

    # Create synthetic regression data
    data = create_synthetic_predictions(1000, "regression", noise_level=0.2)

    evaluator = ModelEvaluationFramework()

    # Evaluate regression model
    metrics = evaluator.evaluate_model(
        predictions=data["predictions"],
        actual_outcomes=data["outcomes"],
        prediction_type="regression",
        model_name="Regression Test",
    )

    print(f"  Predictions: {len(data['predictions'])}")
    print(f"  MAE: {metrics.core_metrics['mae']:.3f}")
    print(f"  RMSE: {metrics.core_metrics['rmse']:.3f}")
    print(f"  R²: {metrics.core_metrics['r2']:.3f}")
    print(f"  MAPE: {metrics.core_metrics['mape']:.3f}%")

    # Validate metrics
    if metrics.core_metrics["mae"] < 0:
        print("  [ERROR] MAE should be non-negative")
        return False

    if metrics.core_metrics["rmse"] < metrics.core_metrics["mae"]:
        print("  [ERROR] RMSE should be >= MAE")
        return False

    if not (-1 <= metrics.core_metrics["r2"] <= 1):
        print("  [WARN] R² outside typical range (can be valid for bad models)")

    # Test with perfect predictions
    print("\n  Testing with perfect regression predictions:")
    perfect_metrics = evaluator.evaluate_model(
        predictions=data["true_values"],
        actual_outcomes=data["outcomes"],
        prediction_type="regression",
        model_name="Perfect Regression",
    )

    print(f"  Perfect MAE: {perfect_metrics.core_metrics['mae']:.3f}")
    print(f"  Perfect R²: {perfect_metrics.core_metrics['r2']:.3f}")

    print("  [OK] All regression metrics computed successfully")
    print("[PASS] Regression metrics tests passed")
    return True


def test_calibration_metrics():
    """Test calibration metrics (ECE, reliability diagrams)."""
    logger.info("Testing calibration metrics...")

    print("\n3. Testing Calibration Metrics:")

    # Create well-calibrated predictions
    data = create_synthetic_predictions(1000, "binary", noise_level=0.05)

    evaluator = ModelEvaluationFramework(n_calibration_bins=10)

    metrics = evaluator.evaluate_model(
        predictions=data["predictions"],
        actual_outcomes=data["outcomes"],
        prediction_type="binary",
        model_name="Calibration Test",
    )

    print(f"  ECE: {metrics.calibration_metrics['ece']:.4f}")
    print(f"  MCE: {metrics.calibration_metrics['mce']:.4f}")
    print(f"  Reliability: {metrics.calibration_metrics['reliability']:.4f}")
    print(f"  Resolution: {metrics.calibration_metrics['resolution']:.4f}")
    print(f"  Sharpness: {metrics.calibration_metrics['sharpness']:.4f}")

    # Validate calibration metrics
    if not (0 <= metrics.calibration_metrics["ece"] <= 1):
        print("  [ERROR] ECE outside valid range")
        return False

    if metrics.calibration_metrics["mce"] < metrics.calibration_metrics["ece"]:
        print("  [ERROR] MCE should be >= ECE")
        return False

    # Test poorly calibrated model
    print("\n  Testing poorly calibrated predictions:")
    poorly_calibrated = np.where(
        data["predictions"] < 0.5,
        data["predictions"] * 0.5,  # Underconfident on low probs
        0.5 + (data["predictions"] - 0.5) * 1.5,
    )  # Overconfident on high probs
    poorly_calibrated = np.clip(poorly_calibrated, 0.01, 0.99)

    poor_metrics = evaluator.evaluate_model(
        predictions=poorly_calibrated,
        actual_outcomes=data["outcomes"],
        prediction_type="binary",
        model_name="Poorly Calibrated",
    )

    print(f"  Poor calibration ECE: {poor_metrics.calibration_metrics['ece']:.4f}")

    if poor_metrics.calibration_metrics["ece"] <= metrics.calibration_metrics["ece"]:
        print("  [WARN] Poorly calibrated model should have higher ECE")

    # Test calibration plot generation
    print("\n  Testing calibration plot generation:")
    with tempfile.TemporaryDirectory() as temp_dir:
        plot_path = Path(temp_dir) / "calibration_plot.png"

        try:
            fig = evaluator.generate_calibration_plot(
                data["predictions"], data["outcomes"], save_path=str(plot_path)
            )
            plt.close(fig)  # Clean up

            if plot_path.exists():
                print("  [OK] Calibration plot generated successfully")
            else:
                print("  [ERROR] Calibration plot not saved")
                return False

        except Exception as e:
            print(f"  [ERROR] Calibration plot generation failed: {e}")
            return False

    print("  [OK] All calibration metrics computed successfully")
    print("[PASS] Calibration metrics tests passed")
    return True


def test_betting_simulation():
    """Test betting simulation metrics (ROI, CLV)."""
    logger.info("Testing betting simulation...")

    print("\n4. Testing Betting Simulation:")

    # Create data with profitable opportunities
    data = create_synthetic_predictions(500, "binary", noise_level=0.08)

    evaluator = ModelEvaluationFramework(
        betting_bankroll=10000, max_bet_fraction=0.05, min_edge_threshold=0.03
    )

    metrics = evaluator.evaluate_model(
        predictions=data["predictions"],
        actual_outcomes=data["outcomes"],
        market_odds=data["market_odds"],
        closing_odds=data["closing_odds"],
        prediction_type="binary",
        model_name="Betting Test",
    )

    print(f"  Bets placed: {metrics.betting_metrics['bets_placed']}")
    print(f"  Win rate: {metrics.betting_metrics['win_rate']:.3f}")
    print(f"  ROI: {metrics.betting_metrics['roi']:.3f}")
    print(f"  Total profit: ${metrics.betting_metrics['total_profit']:.2f}")
    print(f"  Max drawdown: {metrics.betting_metrics['max_drawdown']:.3f}")
    print(f"  Sharpe ratio: {metrics.betting_metrics['sharpe_ratio']:.3f}")
    print(f"  CLV: {metrics.betting_metrics['clv']:.4f}")

    # Validate betting metrics
    if not (0 <= metrics.betting_metrics["win_rate"] <= 1):
        print("  [ERROR] Win rate outside valid range")
        return False

    if not (0 <= metrics.betting_metrics["max_drawdown"] <= 1):
        print("  [ERROR] Max drawdown outside valid range")
        return False

    if metrics.betting_metrics["bets_placed"] == 0:
        print("  [WARN] No bets placed - check edge thresholds")
    else:
        print(f"  [OK] {metrics.betting_metrics['bets_placed']} bets simulated")

    # Test with no market odds
    print("\n  Testing without market odds:")
    no_odds_metrics = evaluator.evaluate_model(
        predictions=data["predictions"],
        actual_outcomes=data["outcomes"],
        prediction_type="binary",
        model_name="No Odds Test",
    )

    if len(no_odds_metrics.betting_metrics) == 0:
        print("  [OK] No betting metrics computed without odds")
    else:
        print("  [ERROR] Betting metrics should be empty without odds")
        return False

    print("  [OK] Betting simulation completed successfully")
    print("[PASS] Betting simulation tests passed")
    return True


def test_statistical_tests():
    """Test statistical significance tests."""
    logger.info("Testing statistical tests...")

    print("\n5. Testing Statistical Tests:")

    # Create data with known accuracy
    data = create_synthetic_predictions(1000, "binary", noise_level=0.1)

    evaluator = ModelEvaluationFramework()

    metrics = evaluator.evaluate_model(
        predictions=data["predictions"],
        actual_outcomes=data["outcomes"],
        prediction_type="binary",
        model_name="Stats Test",
    )

    # Check accuracy test
    if "accuracy_vs_random" in metrics.statistical_tests:
        test_result = metrics.statistical_tests["accuracy_vs_random"]
        print("  Accuracy vs random test:")
        print(f"    Statistic: {test_result['statistic']:.3f}")
        print(f"    P-value: {test_result['p_value']:.4f}")
        print(f"    Significant: {test_result['significant']}")

        if test_result["p_value"] < 0 or test_result["p_value"] > 1:
            print("  [ERROR] P-value outside valid range")
            return False

    # Test regression correlations
    reg_data = create_synthetic_predictions(1000, "regression", noise_level=0.2)

    reg_metrics = evaluator.evaluate_model(
        predictions=reg_data["predictions"],
        actual_outcomes=reg_data["outcomes"],
        prediction_type="regression",
        model_name="Regression Stats",
    )

    if "correlation" in reg_metrics.statistical_tests:
        corr_test = reg_metrics.statistical_tests["correlation"]
        print("  Correlation test:")
        print(f"    Correlation: {corr_test['correlation']:.3f}")
        print(f"    P-value: {corr_test['p_value']:.4f}")
        print(f"    Significant: {corr_test['significant']}")

    print("  [OK] Statistical tests completed successfully")
    print("[PASS] Statistical tests passed")
    return True


def test_model_comparison():
    """Test model comparison functionality."""
    logger.info("Testing model comparison...")

    print("\n6. Testing Model Comparison:")

    # Create data
    data = create_synthetic_predictions(500, "binary", noise_level=0.1)

    evaluator = ModelEvaluationFramework()

    # Evaluate multiple models
    model1_metrics = evaluator.evaluate_model(
        predictions=data["predictions"],
        actual_outcomes=data["outcomes"],
        market_odds=data["market_odds"],
        prediction_type="binary",
        model_name="Model 1",
    )

    # Create a second model (slightly worse)
    worse_preds = data["predictions"] + np.random.normal(
        0, 0.05, len(data["predictions"])
    )
    worse_preds = np.clip(worse_preds, 0.01, 0.99)

    model2_metrics = evaluator.evaluate_model(
        predictions=worse_preds,
        actual_outcomes=data["outcomes"],
        market_odds=data["market_odds"],
        prediction_type="binary",
        model_name="Model 2",
    )

    # Compare models
    comparison_results = {"Model 1": model1_metrics, "Model 2": model2_metrics}

    comparison_df = evaluator.compare_models(comparison_results)

    print(f"  Comparison dataframe shape: {comparison_df.shape}")
    print(f"  Models compared: {list(comparison_df['Model'])}")

    # Check required columns exist
    required_cols = ["Model", "accuracy", "log_loss", "brier_score"]
    missing_cols = [col for col in required_cols if col not in comparison_df.columns]

    if missing_cols:
        print(f"  [ERROR] Missing comparison columns: {missing_cols}")
        return False

    # Test CSV export
    with tempfile.TemporaryDirectory() as temp_dir:
        csv_path = Path(temp_dir) / "model_comparison.csv"
        evaluator.compare_models(comparison_results, save_path=str(csv_path))

        if csv_path.exists():
            print("  [OK] Comparison CSV saved successfully")
        else:
            print("  [ERROR] Comparison CSV not saved")
            return False

    print("  [OK] Model comparison completed successfully")
    print("[PASS] Model comparison tests passed")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Testing edge cases...")

    print("\n7. Testing Edge Cases:")

    evaluator = ModelEvaluationFramework()

    # Test with all NaN predictions
    print("  Testing with NaN predictions:")
    try:
        nan_preds = np.full(100, np.nan)
        outcomes = np.random.binomial(1, 0.5, 100)

        evaluator.evaluate_model(nan_preds, outcomes, prediction_type="binary")
        print("  [ERROR] Should have failed with all NaN predictions")
        return False
    except ValueError:
        print("  [OK] Correctly rejected all NaN predictions")

    # Test with mismatched lengths
    print("  Testing with mismatched array lengths:")
    try:
        preds = np.random.random(100)
        outcomes = np.random.binomial(1, 0.5, 50)  # Different length

        evaluator.evaluate_model(preds, outcomes, prediction_type="binary")
        print("  [ERROR] Should have failed with mismatched lengths")
        return False
    except ValueError:
        print("  [OK] Correctly rejected mismatched lengths")

    # Test with extreme predictions
    print("  Testing with extreme predictions:")
    extreme_preds = np.array([0.0001, 0.9999] * 50)
    outcomes = np.random.binomial(1, 0.5, 100)

    try:
        metrics = evaluator.evaluate_model(
            extreme_preds, outcomes, prediction_type="binary"
        )
        print(
            f"  [OK] Handled extreme predictions (LogLoss: {metrics.core_metrics['log_loss']:.3f})"
        )
    except Exception as e:
        print(f"  [ERROR] Failed with extreme predictions: {e}")
        return False

    # Test with constant predictions
    print("  Testing with constant predictions:")
    constant_preds = np.full(100, 0.6)
    outcomes = np.random.binomial(1, 0.6, 100)

    try:
        metrics = evaluator.evaluate_model(
            constant_preds, outcomes, prediction_type="binary"
        )
        print(
            f"  [OK] Handled constant predictions (Accuracy: {metrics.core_metrics['accuracy']:.3f})"
        )
    except Exception as e:
        print(f"  [ERROR] Failed with constant predictions: {e}")
        return False

    print("  [OK] All edge cases handled correctly")
    print("[PASS] Edge case tests passed")
    return True


def main():
    """Run all evaluation framework tests."""
    print("=" * 80)
    print("MODEL EVALUATION FRAMEWORK TEST SUITE")
    print("=" * 80)

    try:
        # Run all tests
        test_results = [
            test_classification_metrics(),
            test_regression_metrics(),
            test_calibration_metrics(),
            test_betting_simulation(),
            test_statistical_tests(),
            test_model_comparison(),
            test_edge_cases(),
        ]

        # Overall test summary
        print("\n[SUMMARY] OVERALL TEST RESULTS:")
        print("-" * 50)

        test_names = [
            "Classification metrics (LogLoss, Brier, Accuracy)",
            "Regression metrics (MAE, RMSE, R²)",
            "Calibration metrics (ECE, reliability diagrams)",
            "Betting simulation metrics (ROI, CLV)",
            "Statistical significance tests",
            "Model comparison functionality",
            "Edge cases and error handling",
        ]

        all_passed = True
        for test_name, result in zip(test_names, test_results, strict=False):
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status} {test_name}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n[SUCCESS] ALL MODEL EVALUATION TESTS PASSED!")
            print("The model evaluation framework is ready for production use.")
        else:
            print("\n[FAIL] Some tests failed. Please review the output above.")
            return False

    except Exception as e:
        logger.error("Model evaluation tests failed", error=str(e))
        print(f"\n[FAIL] MODEL EVALUATION TESTS FAILED: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
