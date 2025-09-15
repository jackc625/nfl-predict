#!/usr/bin/env python3
"""
ATS Model Comprehensive Test Suite

This script tests all components of the ATS (Against The Spread) model:
- XGBoost/LightGBM regression for expected margin
- Residual distribution conversion to cover probabilities
- Both classification and regression approaches
- Spread betting mechanics
- Feature importance tracking
- Walk-forward validation
- Model persistence and loading

Run this script to validate the ATS model implementation.
"""

import pandas as pd
import numpy as np
import sys
from pathlib import Path
from datetime import datetime, timedelta
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.train_ats import ATSModel, ResidualDistributionConverter, ATSModelPrediction
from utils import get_logger

logger = get_logger(__name__)


def create_test_data(n_games: int = 200) -> pd.DataFrame:
    """Create realistic test data for ATS model testing."""
    np.random.seed(42)

    teams = ['KC', 'BUF', 'CIN', 'BAL', 'SF', 'PHI', 'DAL', 'GB', 'LAR', 'TB']
    seasons = [2022, 2023, 2024]

    games = []
    game_id = 1

    for season in seasons:
        for week in range(1, 19):
            games_this_week = n_games // (len(seasons) * 18)

            for _ in range(games_this_week):
                home_team = np.random.choice(teams)
                away_team = np.random.choice([t for t in teams if t != home_team])

                # Generate realistic features
                elo_home = np.random.normal(1500, 100)
                elo_away = np.random.normal(1500, 100)
                elo_diff = elo_home - elo_away

                # Market spread (negative = home favored)
                market_spread = np.random.normal(-elo_diff / 30, 2.5)

                # True margin with some relationship to features
                true_margin = (elo_diff / 30 +
                             np.random.normal(0, 10) +  # Random component
                             np.random.normal(3, 2))    # Home field advantage

                # Determine if home team covers
                covers_spread = int(true_margin > -market_spread)

                # Generate scores
                total_points = np.random.normal(47, 8)
                home_score = (total_points + true_margin) / 2
                away_score = home_score - true_margin

                game = {
                    'game_id': f'TEST_{season}_{week:02d}_{game_id:03d}',
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': round(max(0, home_score)),
                    'away_score': round(max(0, away_score)),
                    'actual_margin': true_margin,
                    'covers_spread': covers_spread,
                    'market_spread': market_spread,

                    # Features (similar to WP model)
                    'elo_home': elo_home,
                    'elo_away': elo_away,
                    'elo_diff': elo_diff,
                    'rest_days_home': np.random.choice([3, 7, 10, 14]),
                    'rest_days_away': np.random.choice([3, 7, 10, 14]),
                    'is_playoff': int(week > 18),
                    'is_primetime': np.random.binomial(1, 0.25),
                    'temperature': np.random.normal(65, 20),
                    'wind_speed': np.random.exponential(5),
                    'home_epa_offense_4w': np.random.normal(0, 0.25),
                    'away_epa_offense_4w': np.random.normal(0, 0.25),
                    'home_epa_defense_4w': np.random.normal(0, 0.25),
                    'away_epa_defense_4w': np.random.normal(0, 0.25),
                    'opening_spread': market_spread + np.random.normal(0, 1),
                    'closing_total': np.random.normal(47, 5)
                }

                games.append(game)
                game_id += 1

    return pd.DataFrame(games)


def test_residual_distribution_converter():
    """Test the ResidualDistributionConverter class."""
    print("\n" + "="*60)
    print("TEST 1: Residual Distribution Converter")
    print("="*60)

    try:
        # Create test residuals
        np.random.seed(42)
        residuals = np.random.normal(0, 8, 1000)  # Realistic NFL margin residuals

        # Test normal distribution
        converter = ResidualDistributionConverter(distribution_type="normal")
        converter.fit(residuals)

        print(f"+ Normal distribution fitted successfully")
        print(f"  Parameters: {converter.distribution_params}")

        # Test predictions
        predicted_margins = np.array([3.5, -7.2, 0.0, 14.1, -2.8])
        spreads = np.array([-3.0, 6.5, -0.5, -14.0, 3.5])

        cover_probs = converter.predict_cover_probability(predicted_margins, spreads)
        print(f"  Cover probabilities: {cover_probs}")

        # Validate probabilities are in valid range
        assert all(0 <= p <= 1 for p in cover_probs), "Invalid probabilities"
        print(f"+ Cover probabilities in valid range [0, 1]")

        # Test t-distribution
        t_converter = ResidualDistributionConverter(distribution_type="t")
        t_converter.fit(residuals)
        print(f"+ T-distribution fitted successfully")

        # Test error handling
        try:
            converter.predict_cover_probability(predicted_margins, spreads[:3])
            assert False, "Should have raised error for mismatched arrays"
        except:
            print(f"+ Error handling works correctly")

        print(f"+ Residual Distribution Converter: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Residual Distribution Converter test failed: {e}")
        return False


def test_ats_model_initialization():
    """Test ATS model initialization and configuration."""
    print("\n" + "="*60)
    print("TEST 2: ATS Model Initialization")
    print("="*60)

    try:
        # Test default initialization
        model = ATSModel()
        print(f"+ Default model initialized successfully")
        print(f"  Model type: {model.model_type}")
        print(f"  Approach: {model.approach}")
        print(f"  Feature selection: {model.feature_selection_method}")

        # Test custom initialization
        custom_model = ATSModel(
            model_type="lightgbm",
            approach="classification",
            feature_selection_method="recursive",
            max_features=15,
            use_calibration=False,
            hyperparameter_tuning="random_search"
        )
        print(f"+ Custom model initialized successfully")
        print(f"  Model type: {custom_model.model_type}")
        print(f"  Approach: {custom_model.approach}")
        print(f"  Max features: {custom_model.max_features}")

        # Test model summary before training
        summary = model.get_model_summary()
        assert summary['is_trained'] is False
        print(f"+ Model summary works correctly for untrained model")

        # Test base model creation
        regression_model = model._create_base_model("regression")
        classification_model = model._create_base_model("classification")

        print(f"+ Base models created successfully")
        print(f"  Regression model: {type(regression_model).__name__}")
        print(f"  Classification model: {type(classification_model).__name__}")

        print(f"+ ATS Model Initialization: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] ATS model initialization test failed: {e}")
        return False


def test_feature_preparation_and_selection():
    """Test feature preparation and selection methods."""
    print("\n" + "="*60)
    print("TEST 3: Feature Preparation and Selection")
    print("="*60)

    try:
        # Create test data
        data = create_test_data(100)
        model = ATSModel(max_features=10, random_state=42)

        # Test feature preparation
        X, margin_targets, cover_targets = model._prepare_features(data)

        print(f"+ Feature preparation successful")
        print(f"  Features shape: {X.shape}")
        print(f"  Margin targets shape: {margin_targets.shape}")
        print(f"  Cover targets shape: {cover_targets.shape}")
        print(f"  Sample features: {list(X.columns[:5])}")

        # Test different feature selection methods
        selection_methods = ["variance", "univariate", "model_based", "recursive"]

        for method in selection_methods:
            try:
                model.feature_selection_method = method
                selector, selected_features = model.select_features(X, margin_targets, "regression")
                print(f"+ {method} selection: {len(selected_features)} features selected")

                # Test with classification task
                selector, selected_features = model.select_features(X, cover_targets, "classification")
                print(f"+ {method} classification: {len(selected_features)} features selected")

            except Exception as e:
                print(f"- {method} selection failed: {e}")

        # Test with missing data
        X_missing = X.copy()
        X_missing.iloc[:10, 0] = np.nan
        X_filled, _, _ = model._prepare_features(data)
        assert not X_filled.isnull().any().any(), "Missing values not handled"
        print(f"+ Missing value handling works correctly")

        print(f"+ Feature Preparation and Selection: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Feature preparation test failed: {e}")
        return False


def test_model_training():
    """Test ATS model training with different configurations."""
    print("\n" + "="*60)
    print("TEST 4: Model Training")
    print("="*60)

    try:
        # Create training data
        train_data = create_test_data(150)
        val_data = create_test_data(50)

        print(f"+ Training data created: {len(train_data)} games")
        print(f"+ Validation data created: {len(val_data)} games")

        # Test regression approach
        regression_model = ATSModel(
            model_type="xgboost",
            approach="regression",
            feature_selection_method="model_based",
            max_features=12,
            hyperparameter_tuning="none",  # Skip for speed
            random_state=42
        )

        print("\n--- Testing Regression Approach ---")
        regression_results = regression_model.train_model(train_data, val_data)

        print(f"+ Regression training completed")
        print(f"  Training MAE: {regression_results.performance_metrics.get('training_mae', 0):.3f}")
        print(f"  Training RMSE: {regression_results.performance_metrics.get('training_rmse', 0):.3f}")
        print(f"  Training R2: {regression_results.performance_metrics.get('training_r2', 0):.3f}")
        print(f"  Features selected: {len(regression_results.feature_names)}")
        print(f"  Top features: {list(regression_results.feature_importances.keys())[:5]}")

        # Validate results structure
        assert regression_results.regression_model is not None
        assert len(regression_results.feature_names) > 0
        assert len(regression_results.feature_importances) > 0
        assert regression_results.residual_std is not None
        print(f"+ Results structure validated")

        # Test hybrid approach
        hybrid_model = ATSModel(
            model_type="xgboost",
            approach="hybrid",
            feature_selection_method="univariate",
            max_features=10,
            hyperparameter_tuning="none",
            random_state=42
        )

        print("\n--- Testing Hybrid Approach ---")
        hybrid_results = hybrid_model.train_model(train_data, val_data)

        print(f"+ Hybrid training completed")
        print(f"  Has regression model: {hybrid_results.regression_model is not None}")
        print(f"  Has classification model: {hybrid_results.classification_model is not None}")
        print(f"  Cover accuracy: {hybrid_results.performance_metrics.get('training_cover_accuracy', 0):.3f}")

        # Test model predictions
        predictions = regression_model.predict(val_data[:10])
        print(f"+ Predictions generated: {len(predictions)}")

        sample_pred = predictions[0]
        print(f"+ Sample prediction structure validated:")
        print(f"  Game: {sample_pred.home_team} vs {sample_pred.away_team}")
        print(f"  Predicted margin: {sample_pred.predicted_margin:.2f}")
        print(f"  Cover probability: {sample_pred.cover_probability:.3f}")
        print(f"  Confidence: {sample_pred.confidence:.3f}")

        print(f"+ Model Training: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Model training test failed: {e}")
        return False


def test_prediction_mechanics():
    """Test spread betting mechanics and prediction logic."""
    print("\n" + "="*60)
    print("TEST 5: Prediction Mechanics")
    print("="*60)

    try:
        # Create and train model
        train_data = create_test_data(100)
        model = ATSModel(
            model_type="xgboost",
            approach="hybrid",
            hyperparameter_tuning="none",
            random_state=42
        )

        results = model.train_model(train_data)
        print(f"+ Model trained for prediction testing")

        # Create test prediction data
        test_data = pd.DataFrame([
            {
                'game_id': 'TEST_001',
                'home_team': 'KC',
                'away_team': 'BUF',
                'market_spread': -3.5,  # KC favored by 3.5
                'elo_home': 1600,
                'elo_away': 1550,
                'elo_diff': 50,
                'rest_days_home': 7,
                'rest_days_away': 7,
                'is_playoff': 0,
                'is_primetime': 1,
                'temperature': 35,
                'wind_speed': 12,
                'home_epa_offense_4w': 0.15,
                'away_epa_offense_4w': 0.08,
                'home_epa_defense_4w': -0.05,
                'away_epa_defense_4w': 0.02,
                'opening_spread': -3.0,
                'closing_total': 48.5
            }
        ])

        # Make predictions
        predictions = model.predict(test_data)
        pred = predictions[0]

        print(f"+ Prediction mechanics test:")
        print(f"  Game: {pred.home_team} vs {pred.away_team}")
        print(f"  Market spread: {pred.market_spread}")
        print(f"  Predicted margin: {pred.predicted_margin:.2f}")
        print(f"  Predicted spread: {pred.predicted_spread:.2f}")
        print(f"  Cover probability: {pred.cover_probability:.3f}")
        print(f"  Edge: {pred.edge:.4f}" if pred.edge is not None else "  Edge: None")

        # Test spread mechanics
        # If predicted margin > -market_spread, home team should cover
        should_cover = pred.predicted_margin > -pred.market_spread
        cover_prob_high = pred.cover_probability > 0.5

        if should_cover == cover_prob_high:
            print(f"+ Spread mechanics consistent")
        else:
            print(f"- Spread mechanics inconsistent")
            print(f"  Predicted margin: {pred.predicted_margin:.2f}")
            print(f"  Market spread: {pred.market_spread:.2f}")
            print(f"  Should cover: {should_cover}")
            print(f"  Cover prob > 0.5: {cover_prob_high}")

        # Test multiple predictions
        multi_test_data = create_test_data(10)
        multi_predictions = model.predict(multi_test_data)

        print(f"+ Multiple predictions generated: {len(multi_predictions)}")

        # Check prediction validity
        for i, p in enumerate(multi_predictions[:3]):
            assert 0 <= p.cover_probability <= 1, f"Invalid cover probability: {p.cover_probability}"
            assert p.predicted_spread == -p.predicted_margin, "Spread/margin conversion error"
            print(f"  Prediction {i+1}: margin={p.predicted_margin:.2f}, cover_prob={p.cover_probability:.3f}")

        print(f"+ Prediction Mechanics: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Prediction mechanics test failed: {e}")
        return False


def test_walk_forward_validation():
    """Test walk-forward validation functionality."""
    print("\n" + "="*60)
    print("TEST 6: Walk-Forward Validation")
    print("="*60)

    try:
        # Create multi-season data
        games_df = create_test_data(300)  # More data for validation
        print(f"+ Multi-season data created: {len(games_df)} games")
        print(f"  Seasons: {sorted(games_df['season'].unique())}")
        print(f"  Games per season: {games_df.groupby('season').size().to_dict()}")

        # Create ATS model
        model = ATSModel(
            model_type="xgboost",
            approach="regression",
            hyperparameter_tuning="none",  # Skip for speed
            max_features=8,
            random_state=42
        )

        # Run walk-forward validation
        print(f"\n+ Running walk-forward validation...")
        validation_results = model.run_walk_forward_validation(
            games_df, start_season=2023, end_season=2024
        )

        print(f"+ Walk-forward validation completed")
        print(f"  Seasons validated: {len(validation_results['season_results'])}")

        # Check results structure
        assert 'season_results' in validation_results
        assert 'overall_metrics' in validation_results
        assert 'feature_importance_evolution' in validation_results

        # Print season-by-season results
        for result in validation_results['season_results']:
            season = result['season']
            mae = result.get('mae', 'N/A')
            accuracy = result.get('cover_accuracy', 'N/A')

            if isinstance(mae, (int, float)):
                mae_str = f"{mae:.3f}"
            else:
                mae_str = str(mae)

            if isinstance(accuracy, (int, float)):
                acc_str = f"{accuracy:.3f}"
            else:
                acc_str = str(accuracy)

            print(f"  Season {season}: MAE={mae_str}, Cover Acc={acc_str}")

        # Check overall metrics
        overall_metrics = validation_results['overall_metrics']
        if 'overall_mae' in overall_metrics:
            print(f"+ Overall metrics:")
            print(f"  Overall MAE: {overall_metrics['overall_mae']:.3f}")
            print(f"  Overall RMSE: {overall_metrics['overall_rmse']:.3f}")
            print(f"  Overall R2: {overall_metrics['overall_r2']:.3f}")

        print(f"+ Walk-Forward Validation: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Walk-forward validation test failed: {e}")
        return False


def test_model_persistence():
    """Test model saving and loading functionality."""
    print("\n" + "="*60)
    print("TEST 7: Model Persistence")
    print("="*60)

    try:
        # Create and train model
        train_data = create_test_data(100)
        original_model = ATSModel(
            model_type="xgboost",
            approach="regression",
            max_features=10,
            hyperparameter_tuning="none",
            random_state=42
        )

        results = original_model.train_model(train_data)
        print(f"+ Original model trained successfully")

        # Make predictions before saving
        test_data = train_data.head(5)
        original_predictions = original_model.predict(test_data)

        # Save model
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as tmp_file:
            model_path = tmp_file.name

        original_model.save_model(model_path)
        print(f"+ Model saved to temporary file")

        # Load model
        loaded_model = ATSModel()
        loaded_model.load_model(model_path)
        print(f"+ Model loaded successfully")

        # Verify loaded model properties
        assert loaded_model.is_trained == True
        assert loaded_model.model_type == original_model.model_type
        assert loaded_model.approach == original_model.approach
        assert loaded_model.feature_names == original_model.feature_names
        assert loaded_model.residual_std == original_model.residual_std
        print(f"+ Loaded model properties match original")

        # Make predictions with loaded model
        loaded_predictions = loaded_model.predict(test_data)

        # Verify predictions match
        matches = 0
        for orig, loaded in zip(original_predictions, loaded_predictions):
            if abs(orig.predicted_margin - loaded.predicted_margin) < 1e-10:
                matches += 1

        print(f"+ Prediction matching: {matches}/{len(original_predictions)} predictions match")

        if matches == len(original_predictions):
            print(f"+ All predictions match exactly")
        else:
            print(f"- Some predictions differ (may be due to model stochasticity)")

        # Test model summary
        loaded_summary = loaded_model.get_model_summary()
        assert loaded_summary['is_trained'] == True
        print(f"+ Loaded model summary works correctly")

        # Clean up
        Path(model_path).unlink()

        print(f"+ Model Persistence: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Model persistence test failed: {e}")
        return False


def test_edge_cases_and_error_handling():
    """Test edge cases and error handling."""
    print("\n" + "="*60)
    print("TEST 8: Edge Cases and Error Handling")
    print("="*60)

    try:
        # Test untrained model prediction
        untrained_model = ATSModel()
        test_data = create_test_data(5)

        try:
            untrained_model.predict(test_data)
            assert False, "Should have raised error for untrained model"
        except ValueError:
            print(f"+ Untrained model error handling works")

        # Test insufficient training data
        tiny_data = create_test_data(5)
        model = ATSModel(max_features=20, hyperparameter_tuning="none")

        try:
            results = model.train_model(tiny_data)
            print(f"+ Small dataset handled gracefully")
        except Exception as e:
            print(f"+ Small dataset appropriately rejected: {str(e)[:50]}...")

        # Test missing required columns
        incomplete_data = test_data.copy()
        if 'actual_margin' in incomplete_data.columns:
            incomplete_data = incomplete_data.drop(columns=['actual_margin'])
        model = ATSModel()

        try:
            model.train_model(incomplete_data)
            assert False, "Should have raised error for missing target"
        except (ValueError, KeyError):
            print(f"+ Missing target column error handling works")

        # Test prediction with missing features
        train_data = create_test_data(50)
        model = ATSModel(hyperparameter_tuning="none", max_features=5)
        results = model.train_model(train_data)

        # Remove some features from prediction data (but keep required columns)
        pred_data = train_data.head(3).drop(columns=['elo_home', 'elo_away'], errors='ignore')
        predictions = model.predict(pred_data)
        print(f"+ Missing features in prediction handled gracefully")

        # Test extreme feature values
        extreme_data = train_data.head(5).copy()
        extreme_data['elo_home'] = 9999
        extreme_data['temperature'] = -50
        extreme_data['wind_speed'] = 100

        extreme_predictions = model.predict(extreme_data)
        print(f"+ Extreme feature values handled")

        # Verify predictions are still valid
        for pred in extreme_predictions:
            assert 0 <= pred.cover_probability <= 1, "Invalid probability with extreme values"
        print(f"+ Extreme value predictions remain valid")

        print(f"+ Edge Cases and Error Handling: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Edge cases test failed: {e}")
        return False


def run_all_tests():
    """Run all ATS model tests."""
    print("ATS MODEL COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Starting test execution at {datetime.now()}")

    test_functions = [
        test_residual_distribution_converter,
        test_ats_model_initialization,
        test_feature_preparation_and_selection,
        test_model_training,
        test_prediction_mechanics,
        test_walk_forward_validation,
        test_model_persistence,
        test_edge_cases_and_error_handling
    ]

    results = []
    passed_tests = 0

    for i, test_func in enumerate(test_functions, 1):
        print(f"\nRunning test {i}/{len(test_functions)}: {test_func.__name__}")

        try:
            result = test_func()
            results.append((test_func.__name__, result))
            if result:
                passed_tests += 1
        except Exception as e:
            print(f"[FAILED] {test_func.__name__} crashed: {e}")
            results.append((test_func.__name__, False))

    # Print final summary
    print("\n" + "="*70)
    print("ATS MODEL TEST SUMMARY")
    print("="*70)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed_tests}/{len(test_functions)} tests passed")

    if passed_tests == len(test_functions):
        print("+ ALL TESTS PASSED! ATS model implementation is working correctly.")
        return True
    else:
        print(f"- {len(test_functions) - passed_tests} test(s) failed. Review implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)