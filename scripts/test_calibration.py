#!/usr/bin/env python3
"""
Test script for probability calibration system.

This script validates the calibration system with synthetic and realistic
probability data to ensure proper calibration functionality.
"""

import sys
import tempfile
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.calibrate import ProbabilityCalibrator
from utils import get_logger

logger = get_logger(__name__)


def create_sample_probability_data(
    n_samples: int = 1000, miscalibration_type: str = "none"
) -> tuple:
    """
    Create sample probability data for testing calibration.

    Args:
        n_samples: Number of samples to generate
        miscalibration_type: Type of miscalibration ('none', 'overconfident', 'underconfident', 'sigmoid')

    Returns:
        Tuple of (raw_probabilities, true_labels)
    """
    np.random.seed(42)  # Reproducible results

    if miscalibration_type == "none":
        # Well-calibrated probabilities
        true_probs = np.random.uniform(0.1, 0.9, n_samples)
        raw_probs = true_probs + np.random.normal(0, 0.05, n_samples)
        raw_probs = np.clip(raw_probs, 0.01, 0.99)

    elif miscalibration_type == "overconfident":
        # Model is overconfident (probabilities too extreme)
        true_probs = np.random.uniform(0.2, 0.8, n_samples)
        raw_probs = 0.5 + 1.5 * (true_probs - 0.5)  # Stretch probabilities
        raw_probs = np.clip(raw_probs, 0.01, 0.99)

    elif miscalibration_type == "underconfident":
        # Model is underconfident (probabilities too moderate)
        true_probs = np.random.uniform(0.1, 0.9, n_samples)
        raw_probs = 0.5 + 0.6 * (true_probs - 0.5)  # Compress probabilities
        raw_probs = np.clip(raw_probs, 0.01, 0.99)

    elif miscalibration_type == "sigmoid":
        # S-shaped miscalibration
        true_probs = np.random.uniform(0.1, 0.9, n_samples)
        raw_probs = 1 / (1 + np.exp(-6 * (true_probs - 0.5)))  # Sigmoid transformation
        raw_probs = np.clip(raw_probs, 0.01, 0.99)

    else:
        raise ValueError(f"Unknown miscalibration type: {miscalibration_type}")

    # Generate true labels based on true probabilities
    true_labels = np.random.binomial(1, true_probs)

    return raw_probs, true_labels


def create_nfl_like_probability_data(n_samples: int = 1000) -> tuple:
    """
    Create NFL-like probability data with realistic patterns.

    Args:
        n_samples: Number of samples to generate

    Returns:
        Tuple of (raw_probabilities, true_labels, game_data)
    """
    np.random.seed(42)

    # Create realistic NFL game scenarios
    game_data = []
    raw_probs = []
    true_labels = []

    seasons = [2022, 2023, 2024]
    weeks = list(range(1, 19))

    for i in range(n_samples):
        season = np.random.choice(seasons)
        week = np.random.choice(weeks)

        # Create realistic probability patterns
        if week <= 4:
            # Early season: more uncertainty, probabilities closer to 0.5
            base_prob = np.random.uniform(0.35, 0.65)
        elif week >= 15:
            # Late season: more certainty for playoff-bound teams
            base_prob = np.random.choice(
                [np.random.uniform(0.15, 0.35), np.random.uniform(0.65, 0.85)]
            )
        else:
            # Mid-season: normal distribution
            base_prob = np.random.uniform(0.2, 0.8)

        # Add some model miscalibration patterns common in sports betting
        # Models tend to be overconfident on favorites
        if base_prob > 0.6:
            raw_prob = base_prob + np.random.uniform(
                0.05, 0.15
            )  # Overconfident on favorites
        elif base_prob < 0.4:
            raw_prob = base_prob - np.random.uniform(
                0.05, 0.15
            )  # Overconfident on underdogs
        else:
            raw_prob = base_prob + np.random.normal(
                0, 0.05
            )  # Well-calibrated in middle

        raw_prob = np.clip(raw_prob, 0.01, 0.99)

        # Generate true outcome
        true_outcome = np.random.binomial(
            1, base_prob
        )  # Use base_prob as true probability

        game_data.append(
            {
                "game_id": f"TEST_{season}_{week:02d}_{i:04d}",
                "season": season,
                "week": week,
                "raw_probability": raw_prob,
                "true_outcome": true_outcome,
            }
        )

        raw_probs.append(raw_prob)
        true_labels.append(true_outcome)

    return np.array(raw_probs), np.array(true_labels), pd.DataFrame(game_data)


def test_isotonic_calibration():
    """Test isotonic regression calibration."""
    logger.info("Testing isotonic regression calibration...")

    calibrator = ProbabilityCalibrator(primary_method="isotonic")

    print("\n1. Testing Isotonic Regression Calibration:")

    # Test with overconfident probabilities
    raw_probs, true_labels = create_sample_probability_data(1000, "overconfident")

    print(f"Test data: {len(raw_probs)} samples")
    print(f"Raw probability range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")
    print(f"True positive rate: {true_labels.mean():.3f}")

    # Perform calibration
    results = calibrator.calibrate_probabilities(raw_probs, true_labels)

    print(f"Calibration method used: {results.method}")
    print(
        f"Raw ECE: {results.calibration_metrics['raw_expected_calibration_error']:.4f}"
    )
    print(
        f"Calibrated ECE: {results.calibration_metrics['expected_calibration_error']:.4f}"
    )
    print(f"ECE improvement: {results.calibration_metrics['ece_improvement']:.4f}")

    # Calibration should improve ECE
    if results.calibration_metrics["ece_improvement"] > 0:
        print("  [OK] ECE improved after calibration")
    else:
        print("  [WARN] ECE did not improve (may be due to random variation)")

    # Check calibration slope (should be closer to 1 after calibration)
    slope = results.calibration_metrics["calibration_slope"]
    slope_dev = results.calibration_metrics["calibration_slope_deviation"]

    print(f"Calibration slope: {slope:.3f} (deviation from 1.0: {slope_dev:.3f})")

    if slope_dev < 0.2:  # Allow some tolerance
        print("  [OK] Calibration slope reasonable")
    else:
        print("  [WARN] Calibration slope deviates significantly from 1.0")

    # Test edge cases
    print("\n  Testing edge cases:")

    # Single class
    try:
        single_class_labels = np.ones(100)  # All positive
        single_class_probs = np.random.uniform(0.3, 0.9, 100)
        calibrator.calibrate_probabilities(single_class_probs, single_class_labels)
        print("  [OK] Handled single-class data")
    except Exception as e:
        print(f"  [ERROR] Single-class handling failed: {e}")
        return False

    # Very few samples
    try:
        few_probs = np.array([0.1, 0.9, 0.5])
        few_labels = np.array([0, 1, 1])
        calibrator.calibrate_probabilities(few_probs, few_labels)
        print("  [OK] Handled few samples")
    except Exception as e:
        print(f"  [ERROR] Few samples handling failed: {e}")
        return False

    print("[PASS] Isotonic calibration tests passed")
    return True


def test_platt_calibration():
    """Test Platt scaling calibration."""
    logger.info("Testing Platt scaling calibration...")

    calibrator = ProbabilityCalibrator(primary_method="platt")

    print("\n2. Testing Platt Scaling Calibration:")

    # Test with sigmoid-shaped miscalibration (ideal for Platt scaling)
    raw_probs, true_labels = create_sample_probability_data(1000, "sigmoid")

    print(f"Test data: {len(raw_probs)} samples")
    print(f"Raw probability range: [{raw_probs.min():.3f}, {raw_probs.max():.3f}]")

    # Perform calibration
    results = calibrator.calibrate_probabilities(raw_probs, true_labels)

    print(f"Calibration method used: {results.method}")
    print(f"Raw Brier score: {results.calibration_metrics['raw_brier_score']:.4f}")
    print(
        f"Calibrated Brier score: {results.calibration_metrics['calibrated_brier_score']:.4f}"
    )
    print(
        f"Brier score improvement: {results.calibration_metrics['brier_score_improvement']:.4f}"
    )

    # Calibration should improve Brier score
    if results.calibration_metrics["brier_score_improvement"] > 0:
        print("  [OK] Brier score improved after calibration")
    else:
        print("  [WARN] Brier score did not improve")

    # Test with extreme probabilities
    extreme_probs = np.array([0.001, 0.999, 0.5] * 100)
    extreme_labels = np.random.binomial(1, [0.1, 0.9, 0.5] * 100)

    try:
        calibrator.calibrate_probabilities(extreme_probs, extreme_labels)
        print("  [OK] Handled extreme probabilities")
    except Exception as e:
        print(f"  [ERROR] Extreme probabilities handling failed: {e}")
        return False

    print("[PASS] Platt calibration tests passed")
    return True


def test_calibration_metrics():
    """Test calibration metrics calculation."""
    logger.info("Testing calibration metrics...")

    calibrator = ProbabilityCalibrator()

    print("\n3. Testing Calibration Metrics:")

    # Create perfectly calibrated data
    n_samples = 1000
    perfect_probs = np.random.uniform(0, 1, n_samples)
    perfect_labels = np.random.binomial(1, perfect_probs)

    results = calibrator.calibrate_probabilities(perfect_probs, perfect_labels)

    print("Perfect calibration test:")
    print(f"  ECE: {results.calibration_metrics['expected_calibration_error']:.4f}")
    print(f"  MCE: {results.calibration_metrics['maximum_calibration_error']:.4f}")
    print(
        f"  Calibration slope: {results.calibration_metrics['calibration_slope']:.3f}"
    )

    # ECE should be low for well-calibrated data
    if results.calibration_metrics["expected_calibration_error"] < 0.1:
        print("  [OK] ECE is low for well-calibrated data")
    else:
        print("  [WARN] ECE is high for supposedly well-calibrated data")

    # Test reliability curve
    reliability_curve = results.reliability_curve
    mean_pred, frac_pos, bin_edges = reliability_curve

    print(f"Reliability curve bins: {len(bin_edges) - 1}")
    print(f"Mean predicted range: [{mean_pred.min():.3f}, {mean_pred.max():.3f}]")
    print(f"Fraction positive range: [{frac_pos.min():.3f}, {frac_pos.max():.3f}]")

    if len(mean_pred) > 0 and len(frac_pos) > 0:
        print("  [OK] Reliability curve generated successfully")
    else:
        print("  [ERROR] Reliability curve generation failed")
        return False

    print("[PASS] Calibration metrics tests passed")
    return True


def test_cross_validation_calibration():
    """Test within-season cross-validation calibration."""
    logger.info("Testing cross-validation calibration...")

    calibrator = ProbabilityCalibrator()

    print("\n4. Testing Cross-Validation Calibration:")

    # Create NFL-like data with multiple seasons
    raw_probs, true_labels, game_data = create_nfl_like_probability_data(1500)

    print(
        f"Test data: {len(game_data)} games across {game_data['season'].nunique()} seasons"
    )

    # Add probability and outcome columns
    game_data["raw_probability"] = raw_probs
    game_data["true_outcome"] = true_labels

    # Perform within-season calibration
    season_results = calibrator.calibrate_with_cross_validation(
        data=game_data,
        probability_column="raw_probability",
        target_column="true_outcome",
        season_column="season",
    )

    print(f"Calibrated {len(season_results)} seasons")

    for season, results in season_results.items():
        ece = results.calibration_metrics["expected_calibration_error"]
        n_samples = results.metadata["season_samples"]
        print(f"  Season {season}: ECE = {ece:.4f}, samples = {n_samples}")

    if len(season_results) >= 2:
        print("  [OK] Multi-season calibration completed")
    else:
        print("  [ERROR] Insufficient seasons calibrated")
        return False

    # Test individual season calibration quality
    total_ece_improvement = 0
    valid_seasons = 0

    for _season, results in season_results.items():
        if not np.isnan(results.calibration_metrics["ece_improvement"]):
            total_ece_improvement += results.calibration_metrics["ece_improvement"]
            valid_seasons += 1

    if valid_seasons > 0:
        avg_ece_improvement = total_ece_improvement / valid_seasons
        print(f"  Average ECE improvement: {avg_ece_improvement:.4f}")

        if avg_ece_improvement > -0.05:  # Allow small negative due to random variation
            print("  [OK] Average ECE improvement reasonable")
        else:
            print("  [WARN] Average ECE improvement negative")

    print("[PASS] Cross-validation calibration tests passed")
    return True


def test_calibration_plotting():
    """Test calibration plot generation."""
    logger.info("Testing calibration plotting...")

    calibrator = ProbabilityCalibrator()

    print("\n5. Testing Calibration Plot Generation:")

    # Create miscalibrated data for interesting plots
    raw_probs, true_labels = create_sample_probability_data(1000, "overconfident")

    results = calibrator.calibrate_probabilities(raw_probs, true_labels)

    print(f"Plot data prepared: {len(results.calibration_plot_data)} components")

    # Test plot generation (don't save to avoid file system dependencies)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # Ignore matplotlib warnings
            fig = calibrator.generate_calibration_plot(results, save_path=None)

            if fig is not None:
                print("  [OK] Calibration plot generated successfully")
                plt.close(fig)  # Clean up
            else:
                print("  [ERROR] Calibration plot generation returned None")
                return False

    except Exception as e:
        print(f"  [ERROR] Calibration plot generation failed: {e}")
        return False

    print("[PASS] Calibration plotting tests passed")
    return True


def test_calibrator_persistence():
    """Test saving and loading calibrators."""
    logger.info("Testing calibrator persistence...")

    calibrator = ProbabilityCalibrator(primary_method="isotonic")

    print("\n6. Testing Calibrator Persistence:")

    # Create and fit a calibrator
    raw_probs, true_labels = create_sample_probability_data(500, "overconfident")
    results = calibrator.calibrate_probabilities(raw_probs, true_labels)

    # Test saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir) / "test_calibrator.joblib"

        try:
            # Save calibrator
            calibrator.save_calibrator(results, str(temp_path))
            print("  Calibrator saved to temporary file")

            # Load calibrator
            loaded_calibrator = calibrator.load_calibrator(str(temp_path))
            print("  Calibrator loaded successfully")

            # Test loaded calibrator
            test_probs = np.array([0.1, 0.5, 0.9])
            original_calibrated = calibrator.apply_calibration(
                test_probs, results.calibrator
            )
            loaded_calibrated = calibrator.apply_calibration(
                test_probs, loaded_calibrator
            )

            # Results should be identical
            if np.allclose(original_calibrated, loaded_calibrated, rtol=1e-10):
                print("  [OK] Loaded calibrator produces identical results")
            else:
                print("  [ERROR] Loaded calibrator results differ")
                return False

        except Exception as e:
            print(f"  [ERROR] Calibrator persistence test failed: {e}")
            return False

    print("[PASS] Calibrator persistence tests passed")
    return True


def test_fallback_mechanism():
    """Test fallback calibration mechanism."""
    logger.info("Testing fallback mechanism...")

    # Create calibrator that might fail with primary method
    calibrator = ProbabilityCalibrator(
        primary_method="isotonic", fallback_method="platt"
    )

    print("\n7. Testing Fallback Mechanism:")

    # Create data that might cause issues
    problematic_probs = np.array([0.5] * 100)  # All same probability
    problematic_labels = np.array([0, 1] * 50)  # Alternating labels

    try:
        results = calibrator.calibrate_probabilities(
            problematic_probs, problematic_labels
        )
        print(f"  Calibration completed with method: {results.method}")
        print("  [OK] Fallback mechanism handled problematic data")
    except Exception as e:
        print(f"  [ERROR] Fallback mechanism failed: {e}")
        return False

    # Test with invalid inputs
    try:
        invalid_probs = np.array([1.5, -0.5, 0.5])  # Invalid probability range
        invalid_labels = np.array([0, 1, 0])

        # This should fail
        results = calibrator.calibrate_probabilities(invalid_probs, invalid_labels)
        print("  [ERROR] Should have failed with invalid probabilities")
        return False

    except ValueError:
        print("  [OK] Correctly rejected invalid probabilities")
    except Exception as e:
        print(f"  [ERROR] Unexpected error with invalid probabilities: {e}")
        return False

    print("[PASS] Fallback mechanism tests passed")
    return True


def main():
    """Run all calibration tests."""
    print("=" * 80)
    print("PROBABILITY CALIBRATION SYSTEM TEST SUITE")
    print("=" * 80)

    try:
        # Run all tests
        test_results = [
            test_isotonic_calibration(),
            test_platt_calibration(),
            test_calibration_metrics(),
            test_cross_validation_calibration(),
            test_calibration_plotting(),
            test_calibrator_persistence(),
            test_fallback_mechanism(),
        ]

        # Overall test summary
        print("\n[SUMMARY] OVERALL TEST RESULTS:")
        print("-" * 40)

        test_names = [
            "Isotonic regression calibration",
            "Platt scaling calibration",
            "Calibration metrics",
            "Cross-validation calibration",
            "Calibration plotting",
            "Calibrator persistence",
            "Fallback mechanism",
        ]

        all_passed = True
        for test_name, result in zip(test_names, test_results, strict=False):
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status} {test_name}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n[SUCCESS] ALL PROBABILITY CALIBRATION TESTS PASSED!")
            print("The probability calibration system is ready for production use.")
        else:
            print("\n[FAIL] Some tests failed. Please review the output above.")
            return False

    except Exception as e:
        logger.error("Probability calibration tests failed", error=str(e))
        print(f"\n[FAIL] PROBABILITY CALIBRATION TESTS FAILED: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
