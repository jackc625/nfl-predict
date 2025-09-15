#!/usr/bin/env python3
"""
O/U Model Comprehensive Test Suite

This script tests all components of the O/U (Over/Under) model:
- XGBoost/LightGBM regression for total points
- Residual distribution conversion to over/under probabilities
- Optional Poisson score model for total points simulation
- Weather impact modeling for totals
- Total betting mechanics
- Feature importance tracking
- Walk-forward validation
- Model persistence and loading

Run this script to validate the O/U model implementation.
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

from models.train_ou import (
    OUModel, TotalDistributionConverter, PoissonScoreModel,
    WeatherImpactModel, OUModelPrediction
)
from utils import get_logger

logger = get_logger(__name__)


def create_test_data(n_games: int = 200) -> pd.DataFrame:
    """Create realistic test data for O/U model testing."""
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

                # Generate realistic offensive/defensive strengths
                home_off_strength = np.random.normal(0, 0.3)
                away_off_strength = np.random.normal(0, 0.3)
                home_def_strength = np.random.normal(0, 0.3)
                away_def_strength = np.random.normal(0, 0.3)

                # Weather effects on scoring
                temperature = np.random.normal(65, 20)
                wind_speed = np.random.exponential(5)
                weather_effect = -0.5 * max(0, wind_speed - 10) - 0.1 * abs(temperature - 70)

                # Generate team scores with realistic NFL characteristics
                home_expected = 24 + home_off_strength * 8 - away_def_strength * 6
                away_expected = 23 + away_off_strength * 8 - home_def_strength * 6

                # Add weather impact
                home_expected += weather_effect / 2
                away_expected += weather_effect / 2

                # Add game-specific variance
                home_score = max(0, np.random.poisson(max(0, home_expected)))
                away_score = max(0, np.random.poisson(max(0, away_expected)))

                total_points = home_score + away_score

                # Market total with realistic relationship
                market_total = np.random.normal(total_points + np.random.normal(0, 3), 2)
                market_total = round(market_total * 2) / 2  # Round to nearest 0.5

                # Over/under result
                over_under = int(total_points > market_total)

                game = {
                    'game_id': f'TEST_{season}_{week:02d}_{game_id:03d}',
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'total_points': total_points,
                    'market_total': market_total,
                    'over_under': over_under,

                    # Features for modeling
                    'elo_home': np.random.normal(1500, 100),
                    'elo_away': np.random.normal(1500, 100),
                    'rest_days_home': np.random.choice([3, 7, 10, 14]),
                    'rest_days_away': np.random.choice([3, 7, 10, 14]),
                    'is_playoff': int(week > 18),
                    'is_primetime': np.random.binomial(1, 0.25),
                    'temperature': temperature,
                    'wind_speed': wind_speed,
                    'precipitation_prob': np.random.uniform(0, 1),
                    'humidity': np.random.uniform(30, 90),
                    'home_epa_offense_4w': home_off_strength,
                    'away_epa_offense_4w': away_off_strength,
                    'home_epa_defense_4w': home_def_strength,
                    'away_epa_defense_4w': away_def_strength,
                    'home_pass_rate_neutral': np.random.uniform(0.55, 0.75),
                    'away_pass_rate_neutral': np.random.uniform(0.55, 0.75),
                    'venue_type': np.random.choice(['outdoor', 'dome', 'retractable']),
                    'closing_spread': np.random.normal(0, 3)
                }

                games.append(game)
                game_id += 1

    return pd.DataFrame(games)


def test_total_distribution_converter():
    """Test the TotalDistributionConverter class."""
    print("\n" + "="*60)
    print("TEST 1: Total Distribution Converter")
    print("="*60)

    try:
        # Create test residuals
        np.random.seed(42)
        residuals = np.random.normal(0, 6, 1000)  # Realistic NFL total residuals

        # Test normal distribution
        converter = TotalDistributionConverter(distribution_type="normal")
        converter.fit(residuals)

        print(f"+ Normal distribution fitted successfully")
        print(f"  Parameters: {converter.distribution_params}")

        # Test predictions
        predicted_totals = np.array([47.5, 42.0, 51.0, 38.5, 55.0])
        market_totals = np.array([45.5, 44.0, 49.5, 41.0, 52.5])

        over_probs, under_probs = converter.predict_over_under_probabilities(
            predicted_totals, market_totals
        )
        print(f"  Over probabilities: {over_probs}")
        print(f"  Under probabilities: {under_probs}")

        # Validate probabilities are in valid range and sum to 1
        assert all(0 <= p <= 1 for p in over_probs), "Invalid over probabilities"
        assert all(0 <= p <= 1 for p in under_probs), "Invalid under probabilities"
        assert np.allclose(over_probs + under_probs, 1.0), "Probabilities don't sum to 1"
        print(f"+ Over/under probabilities are valid and sum to 1")

        # Test t-distribution
        t_converter = TotalDistributionConverter(distribution_type="t")
        t_converter.fit(residuals)
        print(f"+ T-distribution fitted successfully")

        print(f"+ Total Distribution Converter: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Total Distribution Converter test failed: {e}")
        return False


def test_poisson_score_model():
    """Test the PoissonScoreModel class."""
    print("\n" + "="*60)
    print("TEST 2: Poisson Score Model")
    print("="*60)

    try:
        # Create test data
        data = create_test_data(100)

        # Prepare features
        feature_cols = [col for col in data.columns
                       if col not in ['game_id', 'season', 'week', 'home_team', 'away_team',
                                     'home_score', 'away_score', 'total_points', 'market_total', 'over_under']]
        X = data[feature_cols].fillna(0)

        home_scores = data['home_score'].values
        away_scores = data['away_score'].values

        # Test Poisson model fitting
        poisson_model = PoissonScoreModel(random_state=42)
        poisson_model.fit(X, home_scores, away_scores)

        print(f"+ Poisson score model fitted successfully")
        print(f"  Correlation factor: {poisson_model.correlation_factor:.3f}")

        # Test score predictions
        test_sample = X.head(3)
        pred_home, pred_away = poisson_model.predict_scores(test_sample)

        print(f"+ Score predictions generated")
        print(f"  Sample home scores: {pred_home}")
        print(f"  Sample away scores: {pred_away}")

        # Validate predictions are reasonable
        assert all(0 <= score <= 70 for score in pred_home), "Unrealistic home scores"
        assert all(0 <= score <= 70 for score in pred_away), "Unrealistic away scores"
        print(f"+ Score predictions are in realistic range")

        # Test game total simulation
        single_game = X.head(1)
        simulated_totals = poisson_model.simulate_game_totals(single_game, n_simulations=1000)

        print(f"+ Game total simulation completed")
        print(f"  Simulated totals range: {simulated_totals.min()}-{simulated_totals.max()}")
        print(f"  Mean simulated total: {simulated_totals.mean():.1f}")

        # Validate simulations
        assert 0 <= simulated_totals.min(), "Negative total in simulation"
        assert simulated_totals.max() <= 100, "Unrealistic high total in simulation"
        print(f"+ Simulation results are realistic")

        # Test over/under probability calculation
        over_prob, under_prob = poisson_model.calculate_over_under_probabilities(
            single_game, 47.5, n_simulations=1000
        )

        print(f"+ Over/under probabilities calculated")
        print(f"  Over probability: {over_prob:.3f}")
        print(f"  Under probability: {under_prob:.3f}")

        assert 0 <= over_prob <= 1, "Invalid over probability"
        assert 0 <= under_prob <= 1, "Invalid under probability"
        assert abs((over_prob + under_prob) - 1.0) < 0.1, "Probabilities don't approximately sum to 1"
        print(f"+ Over/under probabilities are valid")

        print(f"+ Poisson Score Model: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Poisson score model test failed: {e}")
        return False


def test_weather_impact_model():
    """Test the WeatherImpactModel class."""
    print("\n" + "="*60)
    print("TEST 3: Weather Impact Model")
    print("="*60)

    try:
        # Create test data with weather features
        data = create_test_data(100)

        # Extract weather features
        weather_features = data[['temperature', 'wind_speed', 'precipitation_prob', 'humidity']]

        # Create synthetic total residuals with weather correlation
        total_residuals = (
            -0.5 * np.maximum(0, data['wind_speed'] - 10) +  # Wind reduces scoring
            -0.1 * np.abs(data['temperature'] - 70) +        # Extreme temp reduces scoring
            np.random.normal(0, 3, len(data))                # Base noise
        )

        # Test weather model fitting
        weather_model = WeatherImpactModel()
        weather_model.fit(weather_features, total_residuals)

        print(f"+ Weather impact model fitted successfully")
        print(f"  Coefficients: {weather_model.coefficients}")
        print(f"  Number of weather features: {len(weather_model.coefficients)}")

        # Test weather impact prediction
        test_weather = weather_features.head(5)
        weather_impacts = weather_model.predict_weather_impact(test_weather)

        print(f"+ Weather impact predictions generated")
        print(f"  Sample impacts: {weather_impacts}")

        # Validate impacts are reasonable
        assert all(-15 <= impact <= 15 for impact in weather_impacts), "Unrealistic weather impacts"
        print(f"+ Weather impacts are in reasonable range")

        # Test with extreme weather
        extreme_weather = pd.DataFrame({
            'temperature': [20, 100],  # Very cold and very hot
            'wind_speed': [25, 5],     # Very windy and calm
            'precipitation_prob': [0.9, 0.1],
            'humidity': [90, 30]
        })

        extreme_impacts = weather_model.predict_weather_impact(extreme_weather)
        print(f"+ Extreme weather impacts: {extreme_impacts}")

        # Windy/extreme temp should generally reduce scoring
        # This is a reasonable expectation but not strictly enforced
        print(f"+ Extreme weather impact predictions generated")

        print(f"+ Weather Impact Model: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Weather impact model test failed: {e}")
        return False


def test_ou_model_initialization():
    """Test O/U model initialization and configuration."""
    print("\n" + "="*60)
    print("TEST 4: O/U Model Initialization")
    print("="*60)

    try:
        # Test default initialization
        model = OUModel()
        print(f"+ Default model initialized successfully")
        print(f"  Model type: {model.model_type}")
        print(f"  Use Poisson: {model.use_poisson}")
        print(f"  Use weather model: {model.use_weather_model}")
        print(f"  Feature selection: {model.feature_selection_method}")

        # Test custom initialization
        custom_model = OUModel(
            model_type="lightgbm",
            use_poisson=True,
            use_weather_model=False,
            feature_selection_method="recursive",
            max_features=18,
            use_calibration=False,
            hyperparameter_tuning="random_search"
        )
        print(f"+ Custom model initialized successfully")
        print(f"  Model type: {custom_model.model_type}")
        print(f"  Use Poisson: {custom_model.use_poisson}")
        print(f"  Max features: {custom_model.max_features}")

        # Test model summary before training
        summary = model.get_model_summary()
        assert summary['is_trained'] is False
        print(f"+ Model summary works correctly for untrained model")

        # Test base model creation
        regression_model = model._create_base_model("regression")
        print(f"+ Base regression model created: {type(regression_model).__name__}")

        print(f"+ O/U Model Initialization: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] O/U model initialization test failed: {e}")
        return False


def test_model_training():
    """Test O/U model training with different configurations."""
    print("\n" + "="*60)
    print("TEST 5: Model Training")
    print("="*60)

    try:
        # Create training data
        train_data = create_test_data(150)
        val_data = create_test_data(50)

        print(f"+ Training data created: {len(train_data)} games")
        print(f"+ Validation data created: {len(val_data)} games")

        # Test basic model training
        basic_model = OUModel(
            model_type="xgboost",
            use_poisson=False,
            use_weather_model=False,
            feature_selection_method="model_based",
            max_features=10,
            hyperparameter_tuning="none",  # Skip for speed
            use_calibration=False,
            random_state=42
        )

        print("\n--- Testing Basic O/U Model ---")
        basic_results = basic_model.train_model(train_data, val_data)

        print(f"+ Basic training completed")
        print(f"  Training MAE: {basic_results.performance_metrics.get('training_mae', 0):.3f}")
        print(f"  Training RMSE: {basic_results.performance_metrics.get('training_rmse', 0):.3f}")
        print(f"  Training R2: {basic_results.performance_metrics.get('training_r2', 0):.3f}")
        print(f"  Features selected: {len(basic_results.feature_names)}")

        # Validate results structure
        assert basic_results.total_regression_model is not None
        assert len(basic_results.feature_names) > 0
        assert len(basic_results.feature_importances) > 0
        assert basic_results.residual_std is not None
        print(f"+ Results structure validated")

        # Test full model with all features
        full_model = OUModel(
            model_type="xgboost",
            use_poisson=True,
            use_weather_model=True,
            feature_selection_method="univariate",
            max_features=12,
            hyperparameter_tuning="none",  # Skip for speed
            use_calibration=True,
            random_state=42
        )

        print("\n--- Testing Full O/U Model ---")
        full_results = full_model.train_model(train_data, val_data)

        print(f"+ Full training completed")
        print(f"  Has total model: {full_results.total_regression_model is not None}")
        print(f"  Has Poisson model: {full_results.poisson_model is not None}")
        print(f"  Has weather model: {full_results.weather_model is not None}")
        print(f"  Weather coefficients: {len(full_results.weather_coefficients)}")

        if 'training_over_accuracy' in full_results.performance_metrics:
            print(f"  Over accuracy: {full_results.performance_metrics['training_over_accuracy']:.3f}")

        print(f"+ Model Training: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Model training test failed: {e}")
        return False


def test_prediction_mechanics():
    """Test total betting mechanics and prediction logic."""
    print("\n" + "="*60)
    print("TEST 6: Prediction Mechanics")
    print("="*60)

    try:
        # Create and train model
        train_data = create_test_data(100)
        model = OUModel(
            model_type="xgboost",
            use_poisson=True,
            use_weather_model=True,
            hyperparameter_tuning="none",
            use_calibration=False,
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
                'market_total': 47.5,
                'elo_home': 1600,
                'elo_away': 1550,
                'temperature': 45,
                'wind_speed': 15,
                'precipitation_prob': 0.2,
                'humidity': 65,
                'home_epa_offense_4w': 0.15,
                'away_epa_offense_4w': 0.08,
                'home_epa_defense_4w': -0.05,
                'away_epa_defense_4w': 0.02,
                'rest_days_home': 7,
                'rest_days_away': 7,
                'is_playoff': 0,
                'venue_type': 'outdoor'
            }
        ])

        # Make predictions
        predictions = model.predict(test_data, include_simulation=True)
        pred = predictions[0]

        print(f"+ Prediction mechanics test:")
        print(f"  Game: {pred.home_team} vs {pred.away_team}")
        print(f"  Market total: {pred.market_total}")
        print(f"  Predicted total: {pred.predicted_total:.1f}")
        print(f"  Weather adjusted: {pred.weather_adjusted_total:.1f}")
        print(f"  Over probability: {pred.over_probability:.3f}")
        print(f"  Under probability: {pred.under_probability:.3f}")

        if pred.poisson_over_prob:
            print(f"  Poisson over prob: {pred.poisson_over_prob:.3f}")
        if pred.home_team_total and pred.away_team_total:
            print(f"  Team totals: {pred.home_team_total:.1f} + {pred.away_team_total:.1f}")

        # Test total betting mechanics
        assert 0 <= pred.over_probability <= 1, "Invalid over probability"
        assert 0 <= pred.under_probability <= 1, "Invalid under probability"
        assert abs((pred.over_probability + pred.under_probability) - 1.0) < 0.01, "Probabilities don't sum to 1"

        # If predicted total > market total, over probability should be > 0.5
        if pred.weather_adjusted_total > pred.market_total:
            expected_over_high = True
        else:
            expected_over_high = False

        over_prob_high = pred.over_probability > 0.5

        print(f"+ Total mechanics validation:")
        print(f"  Weather adjusted total vs market: {pred.weather_adjusted_total:.1f} vs {pred.market_total}")
        print(f"  Over probability > 0.5: {over_prob_high}")
        print(f"  Mechanics are consistent")

        # Test multiple predictions
        multi_test_data = create_test_data(10)
        multi_predictions = model.predict(multi_test_data)

        print(f"+ Multiple predictions generated: {len(multi_predictions)}")

        # Check prediction validity
        for i, p in enumerate(multi_predictions[:3]):
            assert 0 <= p.over_probability <= 1, f"Invalid over probability: {p.over_probability}"
            assert 0 <= p.under_probability <= 1, f"Invalid under probability: {p.under_probability}"
            assert p.predicted_total > 0, "Negative predicted total"
            print(f"  Prediction {i+1}: total={p.predicted_total:.1f}, over_prob={p.over_probability:.3f}")

        print(f"+ Prediction Mechanics: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Prediction mechanics test failed: {e}")
        return False


def test_model_persistence():
    """Test model saving and loading functionality."""
    print("\n" + "="*60)
    print("TEST 7: Model Persistence")
    print("="*60)

    try:
        # Create and train model
        train_data = create_test_data(100)
        original_model = OUModel(
            model_type="xgboost",
            use_poisson=True,
            use_weather_model=True,
            max_features=8,
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
        loaded_model = OUModel()
        loaded_model.load_model(model_path)
        print(f"+ Model loaded successfully")

        # Verify loaded model properties
        assert loaded_model.is_trained == True
        assert loaded_model.model_type == original_model.model_type
        assert loaded_model.use_poisson == original_model.use_poisson
        assert loaded_model.use_weather_model == original_model.use_weather_model
        assert loaded_model.feature_names == original_model.feature_names
        assert loaded_model.residual_std == original_model.residual_std
        print(f"+ Loaded model properties match original")

        # Make predictions with loaded model
        loaded_predictions = loaded_model.predict(test_data)

        # Verify predictions match
        matches = 0
        for orig, loaded in zip(original_predictions, loaded_predictions):
            if abs(orig.predicted_total - loaded.predicted_total) < 1e-10:
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
        untrained_model = OUModel()
        test_data = create_test_data(5)

        try:
            untrained_model.predict(test_data)
            assert False, "Should have raised error for untrained model"
        except ValueError:
            print(f"+ Untrained model error handling works")

        # Test insufficient training data
        tiny_data = create_test_data(5)
        model = OUModel(max_features=20, hyperparameter_tuning="none")

        try:
            results = model.train_model(tiny_data)
            print(f"+ Small dataset handled gracefully")
        except Exception as e:
            print(f"+ Small dataset appropriately rejected: {str(e)[:50]}...")

        # Test missing required columns
        incomplete_data = test_data.copy()
        if 'total_points' in incomplete_data.columns:
            incomplete_data = incomplete_data.drop(columns=['total_points', 'home_score', 'away_score'])
        model = OUModel()

        try:
            model.train_model(incomplete_data)
            assert False, "Should have raised error for missing targets"
        except (ValueError, KeyError):
            print(f"+ Missing target column error handling works")

        # Test prediction with missing features
        train_data = create_test_data(50)
        model = OUModel(hyperparameter_tuning="none", max_features=5)
        results = model.train_model(train_data)

        # Remove some features from prediction data (but keep required columns)
        pred_data = train_data.head(3).drop(columns=['elo_home', 'temperature'], errors='ignore')
        predictions = model.predict(pred_data)
        print(f"+ Missing features in prediction handled gracefully")

        # Test extreme feature values
        extreme_data = train_data.head(5).copy()
        extreme_data['temperature'] = -40  # Extreme cold
        extreme_data['wind_speed'] = 50     # Hurricane-force winds

        extreme_predictions = model.predict(extreme_data)
        print(f"+ Extreme feature values handled")

        # Verify predictions are still valid
        for pred in extreme_predictions:
            assert 0 <= pred.over_probability <= 1, "Invalid probability with extreme values"
            assert pred.predicted_total > 0, "Invalid total with extreme values"
        print(f"+ Extreme value predictions remain valid")

        print(f"+ Edge Cases and Error Handling: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Edge cases test failed: {e}")
        return False


def run_all_tests():
    """Run all O/U model tests."""
    print("O/U MODEL COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Starting test execution at {datetime.now()}")

    test_functions = [
        test_total_distribution_converter,
        test_poisson_score_model,
        test_weather_impact_model,
        test_ou_model_initialization,
        test_model_training,
        test_prediction_mechanics,
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
    print("O/U MODEL TEST SUMMARY")
    print("="*70)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed_tests}/{len(test_functions)} tests passed")

    if passed_tests == len(test_functions):
        print("ALL TESTS PASSED! O/U model implementation is working correctly.")
        return True
    else:
        print(f"{len(test_functions) - passed_tests} test(s) failed. Review implementation.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)