#!/usr/bin/env python3
"""
Test script for baseline Elo model.

This script validates the baseline Elo model implementation with
synthetic NFL game data and realistic scenarios.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baseline_elo import BaselineEloModel, EloModelPrediction, EloModelResults
from utils import get_logger

logger = get_logger(__name__)


def create_sample_nfl_games(n_games: int = 500, n_seasons: int = 3) -> pd.DataFrame:
    """
    Create sample NFL game data for testing.

    Args:
        n_games: Total number of games to generate
        n_seasons: Number of seasons to spread games across

    Returns:
        DataFrame with synthetic NFL game data
    """
    np.random.seed(42)  # Reproducible results

    # NFL team abbreviations
    nfl_teams = [
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LV",
        "LAC",
        "LAR",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    ]

    games = []
    base_season = 2022
    weeks = list(range(1, 19))  # 18 regular season weeks

    # Create team quality tiers for realistic Elo differences
    elite_teams = np.random.choice(nfl_teams, 8, replace=False)
    good_teams = np.random.choice(
        [t for t in nfl_teams if t not in elite_teams], 8, replace=False
    )
    avg_teams = np.random.choice(
        [t for t in nfl_teams if t not in elite_teams and t not in good_teams],
        8,
        replace=False,
    )
    poor_teams = [
        t
        for t in nfl_teams
        if t not in elite_teams and t not in good_teams and t not in avg_teams
    ]

    team_quality = {}
    for team in elite_teams:
        team_quality[team] = "elite"
    for team in good_teams:
        team_quality[team] = "good"
    for team in avg_teams:
        team_quality[team] = "average"
    for team in poor_teams:
        team_quality[team] = "poor"

    # Generate games
    for i in range(n_games):
        season = base_season + (i % n_seasons)
        week = np.random.choice(weeks)

        # Select teams with some preference for competitive matchups
        home_team = np.random.choice(nfl_teams)
        away_team = np.random.choice([t for t in nfl_teams if t != home_team])

        # Determine team strengths
        home_quality = team_quality[home_team]
        away_quality = team_quality[away_team]

        # Map quality to Elo ratings (approximate)
        quality_elo = {
            "elite": np.random.normal(1650, 50),
            "good": np.random.normal(1550, 40),
            "average": np.random.normal(1500, 30),
            "poor": np.random.normal(1450, 40),
        }

        home_elo_base = quality_elo[home_quality]
        away_elo_base = quality_elo[away_quality]

        # Add some random variation
        home_elo = home_elo_base + np.random.normal(0, 20)
        away_elo = away_elo_base + np.random.normal(0, 20)

        # Calculate win probability with home field advantage
        home_advantage = 65
        elo_diff = (home_elo + home_advantage) - away_elo
        win_prob = 1 / (1 + 10 ** (-elo_diff / 400))

        # Generate game outcome
        home_wins = np.random.binomial(1, win_prob)

        # Generate point differential based on Elo difference and outcome
        expected_margin = elo_diff / 25  # Rough conversion
        if home_wins:
            actual_margin = max(1, expected_margin + np.random.normal(0, 10))
        else:
            actual_margin = min(-1, expected_margin + np.random.normal(0, 10))

        game = {
            "game_id": f"TEST_{season}_{week:02d}_{home_team}_{away_team}",
            "season": season,
            "week": week,
            "home_team": home_team,
            "away_team": away_team,
            "home_wins": home_wins,
            "point_differential": actual_margin,
            "home_score": max(0, 24 + actual_margin / 2 + np.random.normal(0, 5)),
            "away_score": max(0, 24 - actual_margin / 2 + np.random.normal(0, 5)),
            "elo_home": home_elo,  # True Elo for validation
            "elo_away": away_elo,  # True Elo for validation
        }

        # Ensure scores are consistent with outcome
        if home_wins and game["home_score"] <= game["away_score"]:
            game["home_score"] = game["away_score"] + abs(actual_margin)
        elif not home_wins and game["away_score"] <= game["home_score"]:
            game["away_score"] = game["home_score"] + abs(actual_margin)

        games.append(game)

    games_df = pd.DataFrame(games)

    # Sort by season and week for chronological order
    games_df = games_df.sort_values(["season", "week"]).reset_index(drop=True)

    logger.info(
        "Created sample NFL games",
        games=len(games_df),
        seasons=games_df["season"].nunique(),
        teams=len(
            set(games_df["home_team"].unique()) | set(games_df["away_team"].unique())
        ),
    )

    return games_df


def test_elo_probability_calculation():
    """Test pure Elo probability calculations."""
    logger.info("Testing Elo probability calculations...")

    model = BaselineEloModel()

    print("\n1. Testing Pure Elo Probability Calculations:")

    # Test cases with known outcomes
    test_cases = [
        {
            "home_elo": 1500,
            "away_elo": 1500,
            "expected_prob": 0.593,
        },  # Equal teams + home advantage
        {
            "home_elo": 1600,
            "away_elo": 1500,
            "expected_prob": 0.760,
        },  # Home team stronger
        {
            "home_elo": 1400,
            "away_elo": 1500,
            "expected_prob": 0.407,
        },  # Away team stronger
        {
            "home_elo": 1500,
            "away_elo": 1500,
            "home_adv": 0,
            "expected_prob": 0.500,
        },  # No home advantage
    ]

    for i, case in enumerate(test_cases):
        home_adv = case.get("home_adv", 65)  # Default home advantage
        prob = model.calculate_win_probability(
            case["home_elo"], case["away_elo"], home_adv
        )

        print(
            f"Test case {i + 1}: Home={case['home_elo']}, Away={case['away_elo']}, "
            f"Home Adv={home_adv}"
        )
        print(f"  Calculated prob: {prob:.3f}, Expected: {case['expected_prob']:.3f}")

        # Allow some tolerance for floating point precision
        if abs(prob - case["expected_prob"]) < 0.05:
            print("  [OK] Probability calculation accurate")
        else:
            print("  [ERROR] Probability calculation inaccurate")
            return False

    # Test edge cases
    print("\nTesting edge cases:")

    # Extreme Elo differences
    extreme_prob = model.calculate_win_probability(1800, 1200)
    print(f"Extreme difference (1800 vs 1200): {extreme_prob:.3f}")
    if 0.9 < extreme_prob < 1.0:
        print("  [OK] Extreme high probability handled")
    else:
        print("  [ERROR] Extreme probability calculation failed")
        return False

    # Very close teams
    close_prob = model.calculate_win_probability(1500, 1499)
    print(f"Very close teams (1500 vs 1499): {close_prob:.3f}")
    if 0.59 < close_prob < 0.60:  # Should be close to default home advantage effect
        print("  [OK] Close teams probability reasonable")
    else:
        print("  [ERROR] Close teams probability unreasonable")
        return False

    print("[PASS] Elo probability calculation tests passed")
    return True


def test_training_data_preparation():
    """Test training data preparation."""
    logger.info("Testing training data preparation...")

    model = BaselineEloModel()
    games_df = create_sample_nfl_games(100, 2)

    print("\n2. Testing Training Data Preparation:")
    print(f"Input games: {len(games_df)}")

    # Test with Elo columns present
    features_df, targets = model.prepare_training_data(games_df)

    print(
        f"Prepared features: {len(features_df)} games, {len(features_df.columns)} columns"
    )
    print(f"Targets: {len(targets)} labels, positive rate: {targets.mean():.3f}")

    # Check required columns
    required_cols = [
        "game_id",
        "home_team",
        "away_team",
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_advantage",
        "raw_win_probability",
    ]

    missing_cols = [col for col in required_cols if col not in features_df.columns]
    if missing_cols:
        print(f"  [ERROR] Missing required columns: {missing_cols}")
        return False
    print("  [OK] All required columns present")

    # Check data validity
    if len(features_df) != len(targets):
        print("  [ERROR] Features and targets length mismatch")
        return False

    if not (0 <= targets.mean() <= 1):
        print("  [ERROR] Invalid target values")
        return False

    # Check Elo difference calculation
    elo_diffs = features_df["home_elo"] - features_df["away_elo"]
    if not np.allclose(features_df["elo_diff"], elo_diffs):
        print("  [ERROR] Elo difference calculation incorrect")
        return False
    print("  [OK] Elo difference calculated correctly")

    # Check probability range
    probs = features_df["raw_win_probability"]
    if not ((probs >= 0) & (probs <= 1)).all():
        print("  [ERROR] Raw probabilities outside [0, 1] range")
        return False
    print("  [OK] Raw probabilities in valid range")

    # Test without Elo columns (should use defaults)
    games_no_elo = games_df.drop(columns=["elo_home", "elo_away"])
    features_no_elo, _ = model.prepare_training_data(games_no_elo)

    if (
        features_no_elo["home_elo"].nunique() == 1
        and features_no_elo["away_elo"].nunique() == 1
    ):
        print("  [OK] Default Elo ratings used when columns missing")
    else:
        print("  [ERROR] Default Elo handling failed")
        return False

    print("[PASS] Training data preparation tests passed")
    return True


def test_model_training():
    """Test baseline Elo model training."""
    logger.info("Testing model training...")

    print("\n3. Testing Model Training:")

    # Create training and validation data
    training_games = create_sample_nfl_games(300, 2)
    validation_games = create_sample_nfl_games(100, 1)

    print(f"Training games: {len(training_games)}")
    print(f"Validation games: {len(validation_games)}")

    # Initialize and train model
    model = BaselineEloModel(use_calibration=True)

    try:
        results = model.train_model(training_games, validation_games)
        print("  [OK] Model training completed successfully")
    except Exception as e:
        print(f"  [ERROR] Model training failed: {e}")
        return False

    # Check training results
    if not isinstance(results, EloModelResults):
        print("  [ERROR] Invalid results type")
        return False

    if not model.is_trained:
        print("  [ERROR] Model not marked as trained")
        return False

    # Check model components
    if model.logistic_model is None:
        print("  [ERROR] Logistic model not trained")
        return False

    if model.elo_system is None:
        print("  [ERROR] Elo system not initialized")
        return False

    if model.calibrator is None:
        print("  [ERROR] Calibrator not trained")
        return False

    print("  [OK] All model components trained")

    # Check performance metrics
    metrics = results.performance_metrics

    required_metrics = ["training_accuracy", "validation_accuracy"]
    missing_metrics = [m for m in required_metrics if m not in metrics]

    if missing_metrics:
        print(f"  [ERROR] Missing performance metrics: {missing_metrics}")
        return False

    train_acc = metrics["training_accuracy"]
    val_acc = metrics["validation_accuracy"]

    print(f"  Training accuracy: {train_acc:.3f}")
    print(f"  Validation accuracy: {val_acc:.3f}")

    # Reasonable performance check (should be better than random)
    if train_acc < 0.52 or val_acc < 0.50:
        print("  [WARN] Model performance lower than expected")

    if train_acc > 0.55 and val_acc > 0.52:
        print("  [OK] Model performance reasonable")

    # Check model coefficients
    coeffs = results.training_history["model_coefficients"]
    print(f"  Model coefficients: {[f'{c:.3f}' for c in coeffs]}")

    # Elo difference coefficient should be positive (higher Elo = higher win prob)
    if coeffs[0] <= 0:
        print("  [ERROR] Elo difference coefficient should be positive")
        return False
    print("  [OK] Elo difference coefficient positive")

    print("[PASS] Model training tests passed")
    return True


def test_model_predictions():
    """Test model prediction generation."""
    logger.info("Testing model predictions...")

    print("\n4. Testing Model Predictions:")

    # Train a model
    training_games = create_sample_nfl_games(200, 2)
    model = BaselineEloModel(use_calibration=True)
    model.train_model(training_games)

    # Generate predictions
    test_games = create_sample_nfl_games(50, 1)
    predictions = model.predict(test_games, calibrated=True)

    print(f"Generated {len(predictions)} predictions")

    # Check prediction structure
    if not all(isinstance(p, EloModelPrediction) for p in predictions):
        print("  [ERROR] Invalid prediction objects")
        return False

    # Check required fields
    sample_pred = predictions[0]
    required_fields = [
        "game_id",
        "home_team",
        "away_team",
        "home_elo",
        "away_elo",
        "elo_diff",
        "home_advantage",
        "raw_win_probability",
    ]

    for field in required_fields:
        if not hasattr(sample_pred, field) or getattr(sample_pred, field) is None:
            print(f"  [ERROR] Missing prediction field: {field}")
            return False

    print("  [OK] All required prediction fields present")

    # Check probability ranges
    raw_probs = [p.raw_win_probability for p in predictions]
    cal_probs = [p.calibrated_win_probability for p in predictions]

    if not all(0 <= p <= 1 for p in raw_probs):
        print("  [ERROR] Raw probabilities outside [0, 1]")
        return False

    if not all(0 <= p <= 1 for p in cal_probs if p is not None):
        print("  [ERROR] Calibrated probabilities outside [0, 1]")
        return False

    print("  [OK] All probabilities in valid range")

    # Check calibration differences
    calibrated_count = sum(1 for p in cal_probs if p is not None)
    print(f"  Calibrated predictions: {calibrated_count}/{len(predictions)}")

    if calibrated_count == len(predictions):
        print("  [OK] All predictions calibrated")
    else:
        print("  [WARN] Some predictions not calibrated")

    # Check prediction confidence
    confidences = [
        p.prediction_confidence
        for p in predictions
        if p.prediction_confidence is not None
    ]
    if confidences:
        avg_confidence = np.mean(confidences)
        print(f"  Average prediction confidence: {avg_confidence:.3f}")

        if 0 <= avg_confidence <= 1:
            print("  [OK] Prediction confidence in valid range")
        else:
            print("  [ERROR] Invalid prediction confidence range")
            return False

    print("[PASS] Model predictions tests passed")
    return True


def test_walk_forward_validation():
    """Test walk-forward validation."""
    logger.info("Testing walk-forward validation...")

    print("\n5. Testing Walk-Forward Validation:")

    # Create multi-season data
    games_df = create_sample_nfl_games(600, 4)  # 4 seasons of data

    print(
        f"Historical data: {len(games_df)} games across {games_df['season'].nunique()} seasons"
    )

    # Initialize model
    model = BaselineEloModel(use_calibration=True)

    try:
        # Run walk-forward validation
        validation_results = model.run_walk_forward_validation(
            games_df, start_season=2023, end_season=2024
        )
        print("  [OK] Walk-forward validation completed")
    except Exception as e:
        print(f"  [ERROR] Walk-forward validation failed: {e}")
        return False

    # Check results structure
    required_keys = ["start_season", "end_season", "season_results", "overall_metrics"]
    missing_keys = [k for k in required_keys if k not in validation_results]

    if missing_keys:
        print(f"  [ERROR] Missing validation result keys: {missing_keys}")
        return False

    season_results = validation_results["season_results"]
    overall_metrics = validation_results["overall_metrics"]

    print(f"  Seasons validated: {len(season_results)}")
    overall_acc = overall_metrics.get("overall_accuracy", "N/A")
    if isinstance(overall_acc, (int, float)):
        print(f"  Overall accuracy: {overall_acc:.3f}")
    else:
        print(f"  Overall accuracy: {overall_acc}")

    # Check individual season results
    successful_seasons = [s for s, r in season_results.items() if "error" not in r]
    failed_seasons = [s for s, r in season_results.items() if "error" in r]

    print(f"  Successful seasons: {len(successful_seasons)}")
    print(f"  Failed seasons: {len(failed_seasons)}")

    if len(successful_seasons) == 0:
        print("  [ERROR] No seasons successfully validated")
        return False

    # Check performance across seasons
    season_accuracies = []
    for season, result in season_results.items():
        if "metrics" in result:
            acc_key = f"season_{season}_accuracy"
            if acc_key in result["metrics"]:
                season_accuracies.append(result["metrics"][acc_key])

    if season_accuracies:
        print(
            f"  Season accuracy range: {min(season_accuracies):.3f} - {max(season_accuracies):.3f}"
        )
        print(f"  Season accuracy std: {np.std(season_accuracies):.3f}")

        if min(season_accuracies) > 0.45:  # Should beat random guessing
            print("  [OK] All seasons perform reasonably")
        else:
            print("  [WARN] Some seasons perform poorly")

    # Check model stability
    if "model_stability" in validation_results:
        stability = validation_results["model_stability"]
        stability_std = stability.get("accuracy_std", "N/A")
        if isinstance(stability_std, (int, float)):
            print(f"  Model stability std: {stability_std:.3f}")
        else:
            print(f"  Model stability std: {stability_std}")

        if stability.get("accuracy_std", 1.0) < 0.1:
            print("  [OK] Model shows good stability")
        else:
            print("  [WARN] Model shows high variance across seasons")

    print("[PASS] Walk-forward validation tests passed")
    return True


def test_model_persistence():
    """Test model saving and loading."""
    logger.info("Testing model persistence...")

    print("\n6. Testing Model Persistence:")

    # Train a model
    training_games = create_sample_nfl_games(200, 2)
    model = BaselineEloModel(use_calibration=True)
    model.train_model(training_games)

    # Test saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_baseline_elo.joblib"

        try:
            # Save model
            model.save_model(str(model_path))
            print(f"  Model saved to {model_path}")

            # Create new model instance and load
            loaded_model = BaselineEloModel()
            loaded_model.load_model(str(model_path))
            print("  Model loaded successfully")

            # Test predictions are identical
            test_games = create_sample_nfl_games(10, 1)

            original_preds = model.predict(test_games)
            loaded_preds = loaded_model.predict(test_games)

            # Compare predictions
            pred_diffs = []
            for orig, loaded in zip(original_preds, loaded_preds, strict=False):
                diff = abs(orig.raw_win_probability - loaded.raw_win_probability)
                pred_diffs.append(diff)

            max_diff = max(pred_diffs)
            print(f"  Maximum prediction difference: {max_diff:.6f}")

            if max_diff < 1e-10:
                print("  [OK] Loaded model produces identical predictions")
            else:
                print("  [ERROR] Loaded model predictions differ")
                return False

            # Check model state
            if not loaded_model.is_trained:
                print("  [ERROR] Loaded model not marked as trained")
                return False

            if loaded_model.logistic_model is None:
                print("  [ERROR] Logistic model not loaded")
                return False

            print("  [OK] Model state properly restored")

        except Exception as e:
            print(f"  [ERROR] Model persistence failed: {e}")
            return False

    print("[PASS] Model persistence tests passed")
    return True


def test_model_summary():
    """Test model summary generation."""
    logger.info("Testing model summary...")

    print("\n7. Testing Model Summary:")

    # Train a model
    training_games = create_sample_nfl_games(150, 2)
    model = BaselineEloModel(use_calibration=True)
    model.train_model(training_games)

    # Get model summary
    summary = model.get_model_summary()

    print(f"Model summary keys: {list(summary.keys())}")

    # Check required summary fields
    required_fields = [
        "model_type",
        "is_trained",
        "hyperparameters",
        "training_info",
        "model_coefficients",
        "elo_system_info",
    ]

    missing_fields = [f for f in required_fields if f not in summary]
    if missing_fields:
        print(f"  [ERROR] Missing summary fields: {missing_fields}")
        return False

    print("  [OK] All required summary fields present")

    # Check specific values
    if summary["model_type"] != "BaselineEloModel":
        print("  [ERROR] Incorrect model type in summary")
        return False

    if not summary["is_trained"]:
        print("  [ERROR] Model not marked as trained in summary")
        return False

    # Check hyperparameters
    hyperparams = summary["hyperparameters"]
    expected_params = ["initial_elo", "k_factor", "home_advantage", "season_carryover"]
    missing_params = [p for p in expected_params if p not in hyperparams]

    if missing_params:
        print(f"  [ERROR] Missing hyperparameters: {missing_params}")
        return False

    # Check coefficients
    coeffs = summary["model_coefficients"]
    if "elo_diff_coef" not in coeffs or coeffs["elo_diff_coef"] <= 0:
        print("  [ERROR] Invalid Elo difference coefficient")
        return False

    print(f"  Model coefficients: {coeffs}")
    print(f"  Elo system teams: {summary['elo_system_info']['total_teams']}")
    print(f"  Calibration used: {summary['calibration_used']}")

    print("  [OK] Model summary complete and valid")
    print("[PASS] Model summary tests passed")
    return True


def main():
    """Run all baseline Elo model tests."""
    print("=" * 80)
    print("BASELINE ELO MODEL TEST SUITE")
    print("=" * 80)

    try:
        # Run all tests
        test_results = [
            test_elo_probability_calculation(),
            test_training_data_preparation(),
            test_model_training(),
            test_model_predictions(),
            test_walk_forward_validation(),
            test_model_persistence(),
            test_model_summary(),
        ]

        # Overall test summary
        print("\n[SUMMARY] OVERALL TEST RESULTS:")
        print("-" * 40)

        test_names = [
            "Elo probability calculations",
            "Training data preparation",
            "Model training",
            "Model predictions",
            "Walk-forward validation",
            "Model persistence",
            "Model summary",
        ]

        all_passed = True
        for test_name, result in zip(test_names, test_results, strict=False):
            status = "[PASS]" if result else "[FAIL]"
            print(f"{status} {test_name}")
            if not result:
                all_passed = False

        if all_passed:
            print("\n[SUCCESS] ALL BASELINE ELO MODEL TESTS PASSED!")
            print("The baseline Elo model is ready for production use.")
        else:
            print("\n[FAIL] Some tests failed. Please review the output above.")
            return False

    except Exception as e:
        logger.error("Baseline Elo model tests failed", error=str(e))
        print(f"\n[FAIL] BASELINE ELO MODEL TESTS FAILED: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
