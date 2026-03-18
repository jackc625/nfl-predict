#!/usr/bin/env python3
"""
Test suite for Win Probability Model.

This script comprehensively tests the WP model implementation including:
- Feature preparation and cleaning
- Feature selection methods
- Hyperparameter tuning
- Model training and prediction
- Walk-forward validation
- Model persistence and loading
- Calibration integration
- Edge cases and error handling
"""

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.train_wp import WinProbabilityModel
from utils import get_logger

logger = get_logger(__name__)


def create_synthetic_nfl_games(
    n_games: int = 500, n_seasons: int = 3, n_teams: int = 32
) -> pd.DataFrame:
    """Create synthetic NFL games for testing."""
    np.random.seed(42)

    # Team names
    teams = [f"T{i:02d}" for i in range(n_teams)]
    seasons = [2022, 2023, 2024][:n_seasons]

    games = []
    game_id = 1

    for season in seasons:
        for week in range(1, 19):  # 18 weeks
            games_this_week = n_games // (n_seasons * 18)

            for _ in range(games_this_week):
                home_team = np.random.choice(teams)
                away_team = np.random.choice([t for t in teams if t != home_team])

                # Simulate realistic features
                elo_home = np.random.normal(1500, 150)
                elo_away = np.random.normal(1500, 150)
                elo_diff = elo_home - elo_away

                # Simulate home field advantage
                hfa = np.random.normal(65, 15)
                total_advantage = elo_diff + hfa

                # Calculate win probability with some noise
                win_prob = 1 / (1 + 10 ** (-total_advantage / 400))
                home_wins = np.random.binomial(1, win_prob)

                game = {
                    "game_id": f"TEST_{season}_{week:02d}_{game_id:03d}",
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_wins": home_wins,
                    # Elo features
                    "elo_home": elo_home,
                    "elo_away": elo_away,
                    "elo_diff": elo_diff,
                    # Contextual features
                    "rest_days_home": np.random.choice([3, 7, 10, 14]),
                    "rest_days_away": np.random.choice([3, 7, 10, 14]),
                    "is_playoff": int(week > 18),
                    "is_primetime": np.random.binomial(1, 0.2),
                    # Weather features
                    "temperature": np.random.normal(65, 20),
                    "wind_speed": np.random.exponential(5),
                    "precipitation": np.random.exponential(0.1),
                    # Team form features (simplified)
                    "home_epa_offense_4w": np.random.normal(0, 0.3),
                    "away_epa_offense_4w": np.random.normal(0, 0.3),
                    "home_epa_defense_4w": np.random.normal(0, 0.3),
                    "away_epa_defense_4w": np.random.normal(0, 0.3),
                    # Market features
                    "opening_spread": np.random.normal(-elo_diff / 25, 3),
                    "closing_total": np.random.normal(47, 6),
                }

                games.append(game)
                game_id += 1

    return pd.DataFrame(games)


def test_feature_preparation():
    """Test feature preparation and cleaning."""
    logger.info("Testing feature preparation...")

    print("\n1. Testing Feature Preparation:")

    # Create test data
    games_df = create_synthetic_nfl_games(100, 2, 16)

    # Initialize model
    model = WinProbabilityModel(random_state=42)

    # Test feature preparation
    features_df, targets = model.prepare_features(games_df)

    print(f"  Games input: {len(games_df)}")
    print(f"  Features output: {features_df.shape}")
    print(f"  Targets: {len(targets)}")
    print(f"  Target rate: {targets.mean():.3f}")

    # Validate outputs
    if len(features_df) != len(targets):
        print("  [ERROR] Feature and target lengths don't match")
        return False

    if not all(
        col in features_df.columns for col in ["game_id", "home_team", "away_team"]
    ):
        print("  [ERROR] Missing required ID columns")
        return False

    # Test with missing data
    print("\n  Testing with missing data:")
    games_with_na = games_df.copy()
    games_with_na.loc[0:10, "elo_home"] = np.nan
    games_with_na.loc[20:30, "temperature"] = np.nan

    try:
        features_na, _targets_na = model.prepare_features(games_with_na)
        print(f"  [OK] Handled missing data: {features_na.shape}")
    except Exception as e:
        print(f"  [ERROR] Failed with missing data: {e}")
        return False

    print("  [OK] Feature preparation tests passed")
    return True


def test_feature_selection():
    """Test feature selection methods."""
    logger.info("Testing feature selection...")

    print("\n2. Testing Feature Selection Methods:")

    games_df = create_synthetic_nfl_games(200, 2, 16)

    methods_to_test = ["none", "variance", "univariate", "model_based", "recursive"]

    for method in methods_to_test:
        print(f"\n  Testing {method} feature selection:")

        try:
            model = WinProbabilityModel(
                feature_selection_method=method, max_features=10, random_state=42
            )

            features_df, targets = model.prepare_features(games_df)
            _selector, selected_features = model.select_features(features_df, targets)

            print(
                f"    Original features: {len([c for c in features_df.columns if c not in ['game_id', 'home_team', 'away_team']])}"
            )
            print(f"    Selected features: {len(selected_features)}")
            print(f"    Feature names: {selected_features[:5]}...")

            if len(selected_features) == 0:
                print(f"    [ERROR] No features selected for {method}")
                return False

            print(f"    [OK] {method} selection completed")

        except Exception as e:
            print(f"    [ERROR] {method} selection failed: {e}")
            return False

    print("  [OK] All feature selection methods passed")
    return True


def test_hyperparameter_tuning():
    """Test hyperparameter tuning methods."""
    logger.info("Testing hyperparameter tuning...")

    print("\n3. Testing Hyperparameter Tuning:")

    games_df = create_synthetic_nfl_games(150, 2, 12)

    tuning_methods = ["none", "grid_search", "random_search"]
    reg_types = ["l1", "l2", "elasticnet"]

    for reg_type in reg_types:
        print(f"\n  Testing {reg_type} regularization:")

        for tuning_method in tuning_methods:
            try:
                model = WinProbabilityModel(
                    regularization_type=reg_type,
                    hyperparameter_tuning=tuning_method,
                    random_state=42,
                )

                features_df, targets = model.prepare_features(games_df)
                _selector, selected_features = model.select_features(
                    features_df, targets
                )
                X = features_df[selected_features].values
                params = model.tune_hyperparameters(X, targets)

                print(f"    {tuning_method}: {params}")

                if "penalty" not in params:
                    print("    [ERROR] No penalty parameter found")
                    return False

                print(f"    [OK] {tuning_method} with {reg_type}")

            except Exception as e:
                print(f"    [ERROR] {tuning_method} with {reg_type} failed: {e}")
                return False

    print("  [OK] All hyperparameter tuning methods passed")
    return True


def test_model_training():
    """Test complete model training."""
    logger.info("Testing model training...")

    print("\n4. Testing Model Training:")

    games_df = create_synthetic_nfl_games(300, 2, 16)
    train_games = games_df[games_df["season"] == 2022].copy()
    val_games = games_df[games_df["season"] == 2023].copy()

    print(f"  Training games: {len(train_games)}")
    print(f"  Validation games: {len(val_games)}")

    # Test basic training
    print("\n  Testing basic training:")
    try:
        model = WinProbabilityModel(
            feature_selection_method="recursive",
            regularization_type="l2",
            use_calibration=True,
            hyperparameter_tuning="none",
            random_state=42,
        )

        results = model.train_model(train_games, val_games)

        print(
            f"    Training accuracy: {results.performance_metrics.get('training_accuracy', 0):.3f}"
        )
        print(
            f"    Validation accuracy: {results.performance_metrics.get('validation_accuracy', 0):.3f}"
        )
        print(f"    Features used: {len(results.feature_names)}")
        print(f"    Top features: {list(results.feature_importances.keys())[:5]}")

        if not model.is_trained:
            print("    [ERROR] Model not marked as trained")
            return False

        print("    [OK] Basic training completed")

    except Exception as e:
        print(f"    [ERROR] Training failed: {e}")
        return False

    # Test with different configurations
    configurations = [
        {"feature_selection_method": "model_based", "regularization_type": "l1"},
        {"feature_selection_method": "univariate", "regularization_type": "elasticnet"},
        {"use_calibration": False, "regularization_type": "l2"},
    ]

    for i, config in enumerate(configurations):
        print(f"\n  Testing configuration {i + 1}: {config}")
        try:
            config_model = WinProbabilityModel(random_state=42, **config)
            config_results = config_model.train_model(train_games)

            acc = config_results.performance_metrics.get("training_accuracy", 0)
            print(f"    Accuracy: {acc:.3f}")
            print(f"    [OK] Configuration {i + 1} completed")

        except Exception as e:
            print(f"    [ERROR] Configuration {i + 1} failed: {e}")
            return False

    print("  [OK] All model training tests passed")
    return True


def test_predictions():
    """Test prediction generation."""
    logger.info("Testing predictions...")

    print("\n5. Testing Predictions:")

    # Train a model
    games_df = create_synthetic_nfl_games(200, 2, 16)
    train_games = games_df[games_df["season"] == 2022].copy()
    test_games = games_df[games_df["season"] == 2023].head(20).copy()

    model = WinProbabilityModel(random_state=42)
    model.train_model(train_games)

    print(f"  Training completed, testing predictions on {len(test_games)} games")

    # Test calibrated predictions
    print("\n  Testing calibrated predictions:")
    calibrated_preds = model.predict(test_games, calibrated=True)

    print(f"    Predictions generated: {len(calibrated_preds)}")

    # Validate prediction objects
    pred = calibrated_preds[0]
    required_attrs = ["game_id", "home_team", "away_team", "raw_win_probability"]

    for attr in required_attrs:
        if not hasattr(pred, attr) or getattr(pred, attr) is None:
            print(f"    [ERROR] Missing prediction attribute: {attr}")
            return False

    if not (0 <= pred.raw_win_probability <= 1):
        print(
            f"    [ERROR] Raw probability outside valid range: {pred.raw_win_probability}"
        )
        return False

    if pred.calibrated_win_probability is not None:
        if not (0 <= pred.calibrated_win_probability <= 1):
            print("    [ERROR] Calibrated probability outside valid range")
            return False

    print(f"    Sample prediction: {pred.home_team} vs {pred.away_team}")
    print(f"    Raw prob: {pred.raw_win_probability:.3f}")
    print(
        f"    Cal prob: {pred.calibrated_win_probability:.3f if pred.calibrated_win_probability else 'None'}"
    )
    print(
        f"    Confidence: {pred.prediction_confidence:.3f if pred.prediction_confidence else 'None'}"
    )

    # Test non-calibrated predictions
    print("\n  Testing non-calibrated predictions:")
    raw_preds = model.predict(test_games, calibrated=False)

    if len(raw_preds) != len(calibrated_preds):
        print("    [ERROR] Different number of raw vs calibrated predictions")
        return False

    # Check that raw predictions don't have calibrated probabilities
    if raw_preds[0].calibrated_win_probability is not None:
        print("    [ERROR] Raw prediction has calibrated probability")
        return False

    print("    [OK] Non-calibrated predictions work correctly")

    print("  [OK] All prediction tests passed")
    return True


def test_walk_forward_validation():
    """Test walk-forward validation."""
    logger.info("Testing walk-forward validation...")

    print("\n6. Testing Walk-Forward Validation:")

    # Create multi-season data
    games_df = create_synthetic_nfl_games(600, 4, 16)
    print(
        f"  Created {len(games_df)} games across {games_df['season'].nunique()} seasons"
    )

    # Test walk-forward validation
    print("\n  Running walk-forward validation:")
    try:
        model = WinProbabilityModel(
            feature_selection_method="recursive",
            max_features=10,
            hyperparameter_tuning="none",
            random_state=42,
        )

        validation_results = model.run_walk_forward_validation(
            games_df,
            start_season=games_df["season"].min()
            + 1,  # Need at least 1 season for training
            end_season=games_df["season"].max(),
        )

        print(
            f"    Seasons validated: {validation_results['overall_metrics'].get('seasons_validated', 0)}"
        )
        print(
            f"    Overall accuracy: {validation_results['overall_metrics'].get('overall_accuracy', 0):.3f}"
        )
        print(
            f"    Overall log loss: {validation_results['overall_metrics'].get('overall_log_loss', 0):.3f}"
        )

        # Check that we have results for each test season
        expected_seasons = list(
            range(games_df["season"].min() + 1, games_df["season"].max() + 1)
        )
        for season in expected_seasons:
            if season not in validation_results["season_results"]:
                print(f"    [ERROR] Missing results for season {season}")
                return False

            season_result = validation_results["season_results"][season]
            if "error" in season_result:
                print(f"    [ERROR] Season {season} failed: {season_result['error']}")
                return False

        if "model_stability" in validation_results:
            stability = validation_results["model_stability"]
            print(f"    Model stability std: {stability.get('accuracy_std', 0):.3f}")

        print("    [OK] Walk-forward validation completed successfully")

    except Exception as e:
        print(f"    [ERROR] Walk-forward validation failed: {e}")
        return False

    print("  [OK] Walk-forward validation tests passed")
    return True


def test_model_persistence():
    """Test model saving and loading."""
    logger.info("Testing model persistence...")

    print("\n7. Testing Model Persistence:")

    # Train a model
    games_df = create_synthetic_nfl_games(150, 2, 12)
    train_games = games_df[games_df["season"] == games_df["season"].min()].copy()

    original_model = WinProbabilityModel(
        feature_selection_method="recursive",
        regularization_type="elasticnet",
        use_calibration=True,
        random_state=42,
    )

    results = original_model.train_model(train_games)
    print(
        f"  Original model trained with accuracy: {results.performance_metrics.get('training_accuracy', 0):.3f}"
    )

    # Test saving and loading
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "test_wp_model.joblib"

        # Save model
        print("\n  Testing model saving:")
        try:
            original_model.save_model(str(model_path))
            if not model_path.exists():
                print("    [ERROR] Model file not created")
                return False
            print(f"    [OK] Model saved to {model_path.name}")
        except Exception as e:
            print(f"    [ERROR] Model saving failed: {e}")
            return False

        # Load model
        print("\n  Testing model loading:")
        try:
            loaded_model = WinProbabilityModel()
            loaded_model.load_model(str(model_path))

            if not loaded_model.is_trained:
                print("    [ERROR] Loaded model not marked as trained")
                return False

            print("    [OK] Model loaded successfully")
            print(f"    Features: {len(loaded_model.feature_names)}")
            print(f"    Regularization: {loaded_model.regularization_type}")

        except Exception as e:
            print(f"    [ERROR] Model loading failed: {e}")
            return False

        # Test predictions with loaded model
        print("\n  Testing loaded model predictions:")
        try:
            test_games = games_df[games_df["season"] == games_df["season"].max()].head(
                10
            )
            original_preds = original_model.predict(test_games)
            loaded_preds = loaded_model.predict(test_games)

            # Compare predictions
            for orig, loaded in zip(original_preds, loaded_preds, strict=False):
                if abs(orig.raw_win_probability - loaded.raw_win_probability) > 1e-10:
                    print(
                        "    [ERROR] Predictions differ between original and loaded model"
                    )
                    return False

            print("    [OK] Loaded model predictions match original")

        except Exception as e:
            print(f"    [ERROR] Loaded model prediction test failed: {e}")
            return False

    print("  [OK] All model persistence tests passed")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Testing edge cases...")

    print("\n8. Testing Edge Cases:")

    # Test with very small dataset
    print("\n  Testing with small dataset:")
    small_games = create_synthetic_nfl_games(20, 1, 6)
    try:
        small_model = WinProbabilityModel(
            max_features=5, hyperparameter_tuning="none", random_state=42
        )
        small_results = small_model.train_model(small_games)
        print(
            f"    [OK] Small dataset handled: {len(small_results.feature_names)} features"
        )
    except Exception as e:
        print(f"    [WARN] Small dataset failed (expected): {e}")

    # Test prediction without training
    print("\n  Testing prediction without training:")
    try:
        untrained_model = WinProbabilityModel()
        test_games = create_synthetic_nfl_games(5, 1, 4)
        untrained_model.predict(test_games)
        print("    [ERROR] Should have failed with untrained model")
        return False
    except ValueError:
        print("    [OK] Correctly rejected prediction without training")

    # Test with all same target values
    print("\n  Testing with constant target values:")
    constant_games = create_synthetic_nfl_games(50, 1, 8)
    constant_games["home_wins"] = 1  # All home wins

    try:
        constant_model = WinProbabilityModel(
            hyperparameter_tuning="none", random_state=42
        )
        constant_model.train_model(constant_games)
        print("    [OK] Handled constant targets")
    except Exception as e:
        print(f"    [WARN] Constant targets caused issues: {e}")

    # Test with missing features in prediction
    print("\n  Testing prediction with missing features:")
    games_df = create_synthetic_nfl_games(100, 2, 8)
    train_games = games_df[games_df["season"] == games_df["season"].min()].copy()
    test_games = games_df[games_df["season"] == games_df["season"].max()].copy()

    # Train model
    missing_model = WinProbabilityModel(random_state=42, hyperparameter_tuning="none")
    missing_model.train_model(train_games)

    # Remove a column from test data
    test_games_missing = test_games.drop("wind_speed", axis=1, errors="ignore")

    try:
        missing_model.predict(test_games_missing)
        print("    [OK] Handled missing features in prediction data")
    except Exception as e:
        print(f"    [WARN] Missing features caused prediction failure: {e}")

    print("  [OK] Edge case tests completed")
    return True


def test_model_summary():
    """Test model summary functionality."""
    logger.info("Testing model summary...")

    print("\n9. Testing Model Summary:")

    # Test untrained model summary
    untrained_model = WinProbabilityModel()
    untrained_summary = untrained_model.get_model_summary()

    if untrained_summary.get("is_trained", True):
        print("  [ERROR] Untrained model marked as trained")
        return False

    print("  [OK] Untrained model summary correct")

    # Test trained model summary
    games_df = create_synthetic_nfl_games(100, 1, 8)
    trained_model = WinProbabilityModel(random_state=42, hyperparameter_tuning="none")
    trained_model.train_model(games_df)

    trained_summary = trained_model.get_model_summary()

    required_keys = [
        "model_type",
        "is_trained",
        "configuration",
        "model_details",
        "training_info",
    ]
    for key in required_keys:
        if key not in trained_summary:
            print(f"  [ERROR] Missing summary key: {key}")
            return False

    print("  Model Summary:")
    print(f"    Type: {trained_summary['model_type']}")
    print(f"    Trained: {trained_summary['is_trained']}")
    print(f"    Features: {trained_summary['model_details']['features_selected']}")
    print(
        f"    Regularization: {trained_summary['configuration']['regularization_type']}"
    )
    print(f"    Calibration: {trained_summary['calibration_used']}")

    print("  [OK] Trained model summary complete")
    return True


def main():
    """Run all WP model tests."""
    print("=" * 80)
    print("WIN PROBABILITY MODEL TEST SUITE")
    print("=" * 80)

    # Suppress warnings for cleaner output
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    try:
        # Run all test functions
        test_functions = [
            ("Feature Preparation", test_feature_preparation),
            ("Feature Selection Methods", test_feature_selection),
            ("Hyperparameter Tuning", test_hyperparameter_tuning),
            ("Model Training", test_model_training),
            ("Prediction Generation", test_predictions),
            ("Walk-Forward Validation", test_walk_forward_validation),
            ("Model Persistence", test_model_persistence),
            ("Edge Cases", test_edge_cases),
            ("Model Summary", test_model_summary),
        ]

        results = []
        for test_name, test_func in test_functions:
            print(f"\n{'=' * 60}")
            print(f"RUNNING TEST: {test_name.upper()}")
            print(f"{'=' * 60}")

            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                logger.error(f"Test {test_name} crashed", error=str(e))
                print(f"\n[CRASH] {test_name} failed with exception: {e}")
                results.append((test_name, False))

        # Summary
        print("\n" + "=" * 80)
        print("WP MODEL TEST SUMMARY")
        print("=" * 80)

        passed = 0
        failed = 0

        for test_name, success in results:
            status = "[PASS]" if success else "[FAIL]"
            print(f"{status} {test_name}")
            if success:
                passed += 1
            else:
                failed += 1

        print(f"\nTotal Tests: {len(results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")

        if failed == 0:
            print("\n[SUCCESS] ALL WP MODEL TESTS PASSED!")
            print("The Win Probability model is ready for production use.")
        else:
            print(f"\n[FAILED] {failed} TEST(S) FAILED")
            print("Please review the output above and fix the issues.")

        return failed == 0

    except Exception as e:
        logger.error("WP model test suite failed", error=str(e))
        print(f"\n[FATAL] Test suite crashed: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
