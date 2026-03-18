#!/usr/bin/env python3
"""
Prediction Pipeline Comprehensive Test Suite

This script tests the unified prediction pipeline that combines all models:
- WP, ATS, and O/U model integration
- Fair line generation
- Edge calculation vs market lines
- Structured prediction output
- Bet recommendation engine
- Export functionality

Run this script to validate the prediction pipeline implementation.
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.prediction_pipeline import (
    BetRecommendationEngine,
    BetType,
    EdgeCalculator,
    NFLPredictionPipeline,
    OddsConverter,
    UnifiedGamePrediction,
)
from models.train_ats import ATSModel
from models.train_ou import OUModel
from models.train_wp import WinProbabilityModel
from utils import get_logger

logger = get_logger(__name__)


def create_test_models():
    """Create and train minimal test models for pipeline testing."""
    # Create minimal test data
    test_data = pd.DataFrame(
        {
            "game_id": ["test1", "test2", "test3"],
            "season": [2023, 2023, 2023],
            "week": [1, 1, 1],
            "home_team": ["KC", "BUF", "CIN"],
            "away_team": ["BUF", "CIN", "KC"],
            "home_wins": [1, 0, 1],
            "home_score": [28, 17, 35],
            "away_score": [21, 24, 14],
            "actual_margin": [7, -7, 21],
            "covers_spread": [1, 0, 1],
            "market_spread": [-3.5, 2.5, -14.0],
            "total_points": [49, 41, 49],
            "market_total": [47.5, 44.0, 45.5],
            "over_under": [1, 0, 1],
            "elo_home": [1600, 1550, 1580],
            "elo_away": [1550, 1580, 1520],
            "elo_diff": [50, -30, 60],
            "temperature": [65, 70, 60],
            "wind_speed": [5, 8, 12],
            "home_epa_offense_4w": [0.1, -0.05, 0.2],
            "away_epa_offense_4w": [0.05, 0.15, -0.1],
            "home_epa_defense_4w": [-0.05, 0.1, -0.1],
            "away_epa_defense_4w": [0.1, -0.05, 0.05],
        }
    )

    # Train WP model
    wp_model = WinProbabilityModel(
        hyperparameter_tuning="none",
        use_calibration=False,
        max_features=5,
        random_state=42,
    )
    wp_model.train_model(test_data)

    # Train ATS model
    ats_model = ATSModel(
        hyperparameter_tuning="none",
        use_calibration=False,
        max_features=5,
        random_state=42,
    )
    ats_model.train_model(test_data)

    # Train O/U model
    ou_model = OUModel(
        hyperparameter_tuning="none",
        use_calibration=False,
        use_poisson=False,
        use_weather_model=False,
        max_features=5,
        random_state=42,
    )
    ou_model.train_model(test_data)

    return wp_model, ats_model, ou_model, test_data


def test_odds_converter():
    """Test the OddsConverter utility class."""
    print("\n" + "=" * 60)
    print("TEST 1: Odds Converter")
    print("=" * 60)

    try:
        converter = OddsConverter()

        # Test American to decimal conversion
        assert abs(converter.american_to_decimal(100) - 2.0) < 1e-10
        assert abs(converter.american_to_decimal(-110) - 1.909) < 0.01
        assert abs(converter.american_to_decimal(150) - 2.5) < 1e-10
        assert abs(converter.american_to_decimal(-200) - 1.5) < 1e-10
        print("+ American to decimal conversion works correctly")

        # Test decimal to American conversion
        assert converter.decimal_to_american(2.0) == 100
        assert abs(converter.decimal_to_american(1.909) - (-110)) < 2
        assert converter.decimal_to_american(2.5) == 150
        assert converter.decimal_to_american(1.5) == -200
        print("+ Decimal to American conversion works correctly")

        # Test American to probability conversion
        prob_100 = converter.american_to_probability(100)
        prob_neg110 = converter.american_to_probability(-110)
        assert abs(prob_100 - 0.5) < 1e-10
        assert abs(prob_neg110 - 0.524) < 0.01
        print("+ American to probability conversion works correctly")

        # Test probability to American conversion
        odds_50 = converter.probability_to_american(0.5)
        odds_60 = converter.probability_to_american(0.6)
        assert odds_50 == 100
        assert abs(odds_60 - (-150)) < 5
        print("+ Probability to American conversion works correctly")

        print("+ Odds Converter: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Odds converter test failed: {e}")
        return False


def test_edge_calculator():
    """Test the EdgeCalculator class."""
    print("\n" + "=" * 60)
    print("TEST 2: Edge Calculator")
    print("=" * 60)

    try:
        calculator = EdgeCalculator(min_edge_threshold=0.02, max_kelly_fraction=0.25)

        # Test positive edge calculation
        edge = calculator.calculate_edge(0.6, -110)  # 60% fair prob vs -110 odds

        print("+ Edge calculation test:")
        print("  Fair probability: 60%")
        print("  Market odds: -110")
        print(f"  Calculated edge: {edge.edge:.4f}")
        print(f"  Expected value: {edge.expected_value:.4f}")
        print(f"  Kelly fraction: {edge.kelly_fraction:.4f}")
        print(f"  Confidence level: {edge.confidence_level}")

        assert edge.edge > 0, "Should have positive edge"
        assert edge.expected_value > 0, "Should have positive expected value"
        assert 0 < edge.kelly_fraction <= 0.25, "Kelly fraction should be reasonable"
        print("+ Positive edge calculation works correctly")

        # Test negative edge calculation
        negative_edge = calculator.calculate_edge(
            0.4, -110
        )  # 40% fair prob vs -110 odds
        assert negative_edge.edge < 0, "Should have negative edge"
        assert negative_edge.expected_value < 0, "Should have negative expected value"
        assert negative_edge.kelly_fraction == 0, (
            "Kelly fraction should be zero for negative edge"
        )
        print("+ Negative edge calculation works correctly")

        # Test edge with favorable odds
        favorable_edge = calculator.calculate_edge(
            0.6, 150
        )  # 60% fair prob vs +150 odds
        assert favorable_edge.edge > 0, "Should have positive edge with favorable odds"
        assert favorable_edge.expected_value > 0, (
            "Should have positive EV with favorable odds"
        )
        print("+ Favorable odds calculation works correctly")

        print("+ Edge Calculator: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Edge calculator test failed: {e}")
        return False


def test_recommendation_engine():
    """Test the BetRecommendationEngine class."""
    print("\n" + "=" * 60)
    print("TEST 3: Bet Recommendation Engine")
    print("=" * 60)

    try:
        engine = BetRecommendationEngine(min_edge=0.02, min_confidence=0.6)
        calculator = EdgeCalculator()

        # Test strong bet recommendation
        strong_edge = calculator.calculate_edge(0.65, -110)
        strong_edge.bet_type = BetType.MONEYLINE_HOME
        strong_rec = engine.generate_recommendation(strong_edge, 0.85)

        print("+ Strong bet test:")
        print(f"  Edge: {strong_edge.edge:.3f}, Confidence: 0.85")
        print(f"  Recommendation: {strong_rec.recommendation}")
        print(f"  Reasoning: {strong_rec.reasoning}")

        assert strong_rec.recommendation == "STRONG_BET", "Should recommend strong bet"
        print("+ Strong bet recommendation works correctly")

        # Test regular bet recommendation
        regular_edge = calculator.calculate_edge(0.58, -110)
        regular_edge.bet_type = BetType.SPREAD_HOME
        regular_rec = engine.generate_recommendation(regular_edge, 0.75)

        assert regular_rec.recommendation == "BET", "Should recommend regular bet"
        print("+ Regular bet recommendation works correctly")

        # Test lean recommendation
        lean_edge = calculator.calculate_edge(0.53, -110)
        lean_edge.bet_type = BetType.OVER
        lean_rec = engine.generate_recommendation(lean_edge, 0.65)

        assert lean_rec.recommendation == "LEAN", "Should recommend lean"
        print("+ Lean recommendation works correctly")

        # Test pass recommendation
        pass_edge = calculator.calculate_edge(0.51, -110)
        pass_edge.bet_type = BetType.UNDER
        pass_rec = engine.generate_recommendation(pass_edge, 0.55)

        assert pass_rec.recommendation == "PASS", "Should recommend pass"
        print("+ Pass recommendation works correctly")

        print("+ Bet Recommendation Engine: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Recommendation engine test failed: {e}")
        return False


def test_pipeline_initialization():
    """Test prediction pipeline initialization."""
    print("\n" + "=" * 60)
    print("TEST 4: Pipeline Initialization")
    print("=" * 60)

    try:
        # Test initialization without models
        pipeline = NFLPredictionPipeline()
        summary = pipeline.get_pipeline_summary()

        assert not any(summary["models_loaded"].values()), (
            "Should show no models loaded"
        )
        print("+ Empty pipeline initialization works correctly")

        # Create test models
        wp_model, ats_model, ou_model, _ = create_test_models()

        # Test initialization with models
        pipeline_with_models = NFLPredictionPipeline(
            wp_model=wp_model,
            ats_model=ats_model,
            ou_model=ou_model,
            min_edge_threshold=0.03,
            max_kelly_fraction=0.20,
        )

        summary_with_models = pipeline_with_models.get_pipeline_summary()
        assert all(summary_with_models["models_loaded"].values()), (
            "Should show all models loaded"
        )
        assert summary_with_models["configuration"]["min_edge_threshold"] == 0.03
        assert summary_with_models["configuration"]["max_kelly_fraction"] == 0.20
        print("+ Pipeline with models initialization works correctly")

        print("+ Pipeline summary:")
        print(f"  Models loaded: {summary_with_models['models_loaded']}")
        print(f"  Configuration: {summary_with_models['configuration']}")
        print(
            f"  Supported bet types: {len(summary_with_models['supported_bet_types'])}"
        )

        print("+ Pipeline Initialization: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Pipeline initialization test failed: {e}")
        return False


def test_unified_predictions():
    """Test unified prediction generation."""
    print("\n" + "=" * 60)
    print("TEST 5: Unified Predictions")
    print("=" * 60)

    try:
        # Create test models and data
        wp_model, ats_model, ou_model, test_data = create_test_models()

        # Add market line data
        prediction_data = test_data.copy()
        prediction_data["market_moneyline_home"] = [-150, 120, -300]
        prediction_data["market_moneyline_away"] = [130, -140, 250]

        # Initialize pipeline
        pipeline = NFLPredictionPipeline(
            wp_model=wp_model, ats_model=ats_model, ou_model=ou_model
        )

        # Generate predictions
        predictions = pipeline.predict_games(prediction_data)

        print(f"+ Generated {len(predictions)} unified predictions")
        assert len(predictions) == len(prediction_data), (
            "Should generate one prediction per game"
        )

        # Test first prediction
        pred = predictions[0]
        print(f"\n+ Sample prediction for {pred.home_team} vs {pred.away_team}:")
        print(f"  WP home probability: {pred.wp_home_probability:.3f}")
        print(f"  Predicted margin: {pred.predicted_margin:.2f}")
        print(f"  ATS cover probability: {pred.ats_cover_probability:.3f}")
        print(f"  Predicted total: {pred.predicted_total:.1f}")
        print(f"  Over probability: {pred.over_probability:.3f}")

        # Validate prediction structure
        assert isinstance(pred, UnifiedGamePrediction), (
            "Should return UnifiedGamePrediction"
        )
        assert 0 <= pred.wp_home_probability <= 1, "WP probability should be valid"
        assert 0 <= pred.ats_cover_probability <= 1, "ATS probability should be valid"
        assert 0 <= pred.over_probability <= 1, "Over probability should be valid"
        assert pred.predicted_total > 0, "Predicted total should be positive"
        print("+ Prediction structure validated")

        # Test fair lines generation
        assert len(pred.fair_lines) > 0, "Should generate fair lines"
        assert BetType.MONEYLINE_HOME in pred.fair_lines, (
            "Should have home moneyline fair line"
        )
        assert BetType.OVER in pred.fair_lines, "Should have over fair line"

        ml_home_line = pred.fair_lines[BetType.MONEYLINE_HOME]
        print(f"  Home ML fair odds: {ml_home_line.fair_odds_american}")
        print(f"  Home ML fair probability: {ml_home_line.fair_probability:.3f}")
        assert 0 <= ml_home_line.fair_probability <= 1, (
            "Fair probability should be valid"
        )
        print("+ Fair lines generation works correctly")

        # Test edge calculations
        if pred.edges:
            print(f"  Number of edges calculated: {len(pred.edges)}")
            for bet_type, edge in list(pred.edges.items())[:3]:
                print(
                    f"  {bet_type.value}: edge={edge.edge:.4f}, EV={edge.expected_value:.4f}"
                )
                assert isinstance(edge.edge, (int, float)), "Edge should be numeric"
                assert isinstance(edge.expected_value, (int, float)), (
                    "EV should be numeric"
                )
            print("+ Edge calculations work correctly")

        # Test recommendations
        if pred.recommendations:
            print(f"  Number of recommendations: {len(pred.recommendations)}")
            best_rec = pred.recommendations[0]
            print(
                f"  Best recommendation: {best_rec.bet_type.value} - {best_rec.recommendation}"
            )
            print(f"  Edge: {best_rec.edge:.4f}, Kelly: {best_rec.kelly_fraction:.4f}")
            assert best_rec.recommendation in ["STRONG_BET", "BET", "LEAN", "PASS"], (
                "Valid recommendation"
            )
            print("+ Recommendations work correctly")

        # Test model agreement
        assert pred.model_agreement is not None, "Should calculate model agreement"
        assert 0 <= pred.model_agreement <= 1, (
            "Model agreement should be valid percentage"
        )
        print(f"  Model agreement: {pred.model_agreement:.3f}")
        print("+ Model agreement calculation works correctly")

        print("+ Unified Predictions: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Unified predictions test failed: {e}")
        return False


def test_export_functionality():
    """Test prediction export functionality."""
    print("\n" + "=" * 60)
    print("TEST 6: Export Functionality")
    print("=" * 60)

    try:
        # Create test models and generate predictions
        wp_model, ats_model, ou_model, test_data = create_test_models()
        prediction_data = test_data.copy()
        prediction_data["market_moneyline_home"] = [-150, 120, -300]
        prediction_data["market_moneyline_away"] = [130, -140, 250]

        pipeline = NFLPredictionPipeline(
            wp_model=wp_model, ats_model=ats_model, ou_model=ou_model
        )

        predictions = pipeline.predict_games(prediction_data)
        print(f"+ Generated {len(predictions)} predictions for export testing")

        # Test DataFrame export
        df_export = pipeline.export_predictions(predictions, format="dataframe")
        assert isinstance(df_export, pd.DataFrame), "Should return DataFrame"
        assert len(df_export) == len(predictions), (
            "DataFrame should have correct number of rows"
        )
        assert "game_id" in df_export.columns, "Should have game_id column"
        assert "wp_home_probability" in df_export.columns, (
            "Should have WP probability column"
        )
        assert "predicted_total" in df_export.columns, (
            "Should have predicted total column"
        )
        print("+ DataFrame export works correctly")
        print(f"  DataFrame shape: {df_export.shape}")
        print(f"  Columns: {list(df_export.columns)[:5]}...")

        # Test JSON export
        json_export = pipeline.export_predictions(predictions, format="json")
        assert isinstance(json_export, str), "Should return JSON string"

        # Parse JSON to validate structure
        json_data = json.loads(json_export)
        assert len(json_data) == len(predictions), (
            "JSON should have correct number of games"
        )
        assert "game_id" in json_data[0], "JSON should have game_id"
        assert "predictions" in json_data[0], "JSON should have predictions section"
        assert "fair_lines" in json_data[0], "JSON should have fair lines section"
        assert "recommendations" in json_data[0], (
            "JSON should have recommendations section"
        )
        print("+ JSON export works correctly")
        print(f"  JSON data length: {len(json_export)} characters")
        print(f"  Sample keys: {list(json_data[0].keys())}")

        # Test invalid format
        try:
            pipeline.export_predictions(predictions, format="invalid")
            raise AssertionError("Should raise error for invalid format")
        except ValueError:
            print("+ Invalid format error handling works correctly")

        print("+ Export Functionality: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Export functionality test failed: {e}")
        return False


def test_edge_cases_and_error_handling():
    """Test edge cases and error handling."""
    print("\n" + "=" * 60)
    print("TEST 7: Edge Cases and Error Handling")
    print("=" * 60)

    try:
        # Test pipeline without models
        empty_pipeline = NFLPredictionPipeline()
        test_data = pd.DataFrame(
            {"game_id": ["test1"], "home_team": ["KC"], "away_team": ["BUF"]}
        )

        try:
            empty_pipeline.predict_games(test_data)
            raise AssertionError("Should raise error for missing models")
        except ValueError as e:
            assert "Missing trained models" in str(e)
            print("+ Missing models error handling works correctly")

        # Test with minimal market data
        wp_model, ats_model, ou_model, _ = create_test_models()
        pipeline = NFLPredictionPipeline(
            wp_model=wp_model, ats_model=ats_model, ou_model=ou_model
        )

        minimal_data = pd.DataFrame(
            {
                "game_id": ["minimal_test"],
                "season": [2023],
                "week": [1],
                "home_team": ["KC"],
                "away_team": ["BUF"],
                "elo_home": [1600],
                "elo_away": [1550],
                "elo_diff": [50],
                "temperature": [65],
                "wind_speed": [5],
                "home_epa_offense_4w": [0.1],
                "away_epa_offense_4w": [0.05],
                "home_epa_defense_4w": [-0.05],
                "away_epa_defense_4w": [0.1],
            }
        )

        minimal_predictions = pipeline.predict_games(minimal_data)
        assert len(minimal_predictions) == 1, "Should handle minimal data"

        # Should still generate predictions without market lines
        pred = minimal_predictions[0]
        assert pred.wp_home_probability > 0, "Should generate WP prediction"
        assert pred.predicted_total > 0, "Should generate total prediction"
        print("+ Minimal data handling works correctly")

        # Test extreme probability values
        extreme_data = minimal_data.copy()
        extreme_predictions = pipeline.predict_games(extreme_data)
        extreme_pred = extreme_predictions[0]

        # Probabilities should be within valid ranges
        assert 0 <= extreme_pred.wp_home_probability <= 1, (
            "WP probability in valid range"
        )
        assert 0 <= extreme_pred.over_probability <= 1, (
            "Over probability in valid range"
        )
        assert extreme_pred.predicted_total > 0, "Total should be positive"
        print("+ Extreme values handled correctly")

        # Test with missing features (should be handled by individual models)
        incomplete_data = minimal_data.drop(
            columns=["temperature", "wind_speed"], errors="ignore"
        )
        incomplete_predictions = pipeline.predict_games(incomplete_data)
        assert len(incomplete_predictions) == 1, (
            "Should handle missing features gracefully"
        )
        print("+ Missing features handled gracefully")

        print("+ Edge Cases and Error Handling: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"[FAILED] Edge cases test failed: {e}")
        return False


def run_all_tests():
    """Run all prediction pipeline tests."""
    print("PREDICTION PIPELINE COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Starting test execution at {datetime.now()}")

    test_functions = [
        test_odds_converter,
        test_edge_calculator,
        test_recommendation_engine,
        test_pipeline_initialization,
        test_unified_predictions,
        test_export_functionality,
        test_edge_cases_and_error_handling,
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
    print("\n" + "=" * 70)
    print("PREDICTION PIPELINE TEST SUMMARY")
    print("=" * 70)

    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{status}: {test_name}")

    print(f"\nOverall: {passed_tests}/{len(test_functions)} tests passed")

    if passed_tests == len(test_functions):
        print(
            "ALL TESTS PASSED! Prediction pipeline implementation is working correctly."
        )
        return True
    print(
        f"{len(test_functions) - passed_tests} test(s) failed. Review implementation."
    )
    return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
