"""
Test suite for walk-forward backtesting system.

This script validates the WalkForwardBacktester functionality including temporal ordering,
data leakage prevention, model integration, and performance tracking.
"""

import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import pickle
import tempfile
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from backtest.walkforward import (
    BacktestConfig,
    BacktestResult,
    BacktestSummary,
    DataLeakageValidator,
    SeasonSplit,
    ValidationLevel,
    WalkForwardBacktester,
    create_default_config,
    create_strict_config,
)
from utils.exceptions import BacktestError, DataValidationError
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MockModel:
    """Mock model for testing."""

    def __init__(self, model_type: str):
        self.model_type = model_type
        self.trained = False

    def fit(self, X, y):
        self.trained = True

    def predict(self, X):
        if not self.trained:
            raise ValueError("Model not trained")
        # Return random predictions based on model type
        if self.model_type == "wp":
            return np.random.uniform(0.3, 0.7, len(X))
        if self.model_type == "ats":
            return np.random.uniform(0.4, 0.6, len(X))
        # ou
        return np.random.uniform(0.45, 0.55, len(X))


def create_mock_data_loader():
    """Create mock data loader for testing."""

    def data_loader(season: int, weeks: list[int] | None = None):
        """Mock data loader that returns synthetic NFL data."""

        # Generate game data for the season
        if weeks is None:
            weeks = list(range(1, 19))  # All weeks

        games = []
        for week in weeks:
            # 16 games per week (approximate)
            for game_id in range(16):
                game_date = datetime(season, 9, 1) + timedelta(weeks=week - 1)

                games.append(
                    {
                        "game_id": f"{season}_{week:02d}_{game_id:02d}",
                        "season": season,
                        "week": week,
                        "date": game_date,
                        "game_date": game_date,
                        "home_team": f"TEAM_{game_id % 32}",
                        "away_team": f"TEAM_{(game_id + 16) % 32}",
                        "home_score": np.random.randint(14, 35),
                        "away_score": np.random.randint(14, 35),
                        "wp_actual": np.random.uniform(0, 1),
                        "ats_actual": np.random.uniform(0, 1),
                        "ou_actual": np.random.uniform(0, 1),
                        "total_points": np.random.randint(28, 70),
                    }
                )

        df = pd.DataFrame(games)
        logger.debug(
            f"Mock data loader: {len(df)} games for season {season}, weeks {weeks}"
        )
        return df

    return data_loader


def create_mock_feature_builder():
    """Create mock feature builder for testing."""

    def feature_builder(
        data: pd.DataFrame, as_of_date: datetime, include_targets: bool = True
    ):
        """Mock feature builder that creates synthetic features."""

        # Create synthetic features
        features = data.copy()

        # Add mock features with proper timestamps
        features["feature_timestamp"] = as_of_date - timedelta(
            hours=1
        )  # Always before as_of_date
        features["elo_home"] = np.random.uniform(1400, 1600, len(features))
        features["elo_away"] = np.random.uniform(1400, 1600, len(features))
        features["rest_days_home"] = np.random.randint(4, 11, len(features))
        features["rest_days_away"] = np.random.randint(4, 11, len(features))
        features["spread_line"] = np.random.uniform(-14, 14, len(features))
        features["total_line"] = np.random.uniform(40, 55, len(features))

        # Add target columns if requested
        if include_targets:
            features["wp_target"] = features["wp_actual"]
            features["ats_target"] = features["ats_actual"]
            features["ou_target"] = features["ou_actual"]

        logger.debug(
            f"Mock feature builder: {len(features)} records with {len(features.columns)} features"
        )
        return features

    return feature_builder


def create_mock_model_trainers():
    """Create mock model trainers for testing."""

    def wp_trainer(
        features: pd.DataFrame,
        target_column: str,
        validation_split: float = 0.2,
        random_state: int = 42,
    ):
        model = MockModel("wp")
        model.fit(features.drop(columns=[target_column]), features[target_column])
        return model

    def ats_trainer(
        features: pd.DataFrame,
        target_column: str,
        validation_split: float = 0.2,
        random_state: int = 42,
    ):
        model = MockModel("ats")
        model.fit(features.drop(columns=[target_column]), features[target_column])
        return model

    def ou_trainer(
        features: pd.DataFrame,
        target_column: str,
        validation_split: float = 0.2,
        random_state: int = 42,
    ):
        model = MockModel("ou")
        model.fit(features.drop(columns=[target_column]), features[target_column])
        return model

    return {"wp": wp_trainer, "ats": ats_trainer, "ou": ou_trainer}


def create_mock_evaluator():
    """Create mock evaluator for testing."""

    def evaluator(predictions: np.ndarray, actuals: np.ndarray, model_type: str):
        """Mock evaluator that calculates basic metrics."""

        # Calculate basic metrics
        metrics = {}

        if model_type == "wp":
            # Classification metrics
            binary_preds = (predictions > 0.5).astype(int)
            binary_actuals = (actuals > 0.5).astype(int)
            metrics["accuracy"] = np.mean(binary_preds == binary_actuals)
            metrics["log_loss"] = -np.mean(
                actuals * np.log(predictions + 1e-15)
                + (1 - actuals) * np.log(1 - predictions + 1e-15)
            )
            metrics["brier_score"] = np.mean((predictions - actuals) ** 2)
        else:
            # Regression metrics
            metrics["mae"] = np.mean(np.abs(predictions - actuals))
            metrics["rmse"] = np.sqrt(np.mean((predictions - actuals) ** 2))

        return metrics

    return evaluator


def test_config_creation():
    """Test configuration creation and validation."""
    logger.info("Running test_config_creation")

    # Test default config
    default_config = create_default_config()
    assert default_config.start_season == 2018, "Default start season should be 2018"
    assert default_config.end_season == 2024, "Default end season should be 2024"
    assert default_config.validation_level == ValidationLevel.MODERATE, (
        "Default validation should be MODERATE"
    )

    # Test strict config
    strict_config = create_strict_config()
    assert strict_config.validation_level == ValidationLevel.STRICT, (
        "Strict config should have STRICT validation"
    )
    assert strict_config.checkpoint_frequency == 2, (
        "Strict config should have frequent checkpoints"
    )

    logger.info("Configuration creation validated")
    return True


def test_season_splits_generation():
    """Test season split generation logic."""
    logger.info("Running test_season_splits_generation")

    # Create temporary config
    with tempfile.TemporaryDirectory() as temp_dir:
        config = BacktestConfig(
            start_season=2018,
            end_season=2021,  # Smaller range for testing
            min_training_seasons=2,
            validation_start_week=5,
            results_path=os.path.join(temp_dir, "results"),
            models_path=os.path.join(temp_dir, "models"),
        )

        backtester = WalkForwardBacktester(config)
        splits = backtester._generate_season_splits()

        # Verify split count
        expected_seasons = 2021 - (2018 + 2) + 1  # end - (start + min_training) + 1
        assert len(splits) == expected_seasons, (
            f"Expected {expected_seasons} splits, got {len(splits)}"
        )

        # Verify first split
        first_split = splits[0]
        assert first_split.validation_season == 2020, (
            "First validation season should be 2020"
        )
        assert first_split.training_seasons == [2018, 2019], (
            "Training seasons should be [2018, 2019]"
        )
        assert first_split.validation_weeks[0] == 5, "Validation should start at week 5"

        # Verify temporal ordering
        for split in splits:
            assert all(s < split.validation_season for s in split.training_seasons), (
                "Training seasons must be before validation"
            )

    logger.info(f"Season splits generation validated: {len(splits)} splits")
    return True


def test_data_leakage_validator():
    """Test data leakage validation."""
    logger.info("Running test_data_leakage_validator")

    validator = DataLeakageValidator(strict_mode=True)

    # Test valid data (no leakage)
    valid_data = pd.DataFrame(
        {
            "game_date": [datetime(2020, 9, 1), datetime(2020, 9, 8)],
            "feature1": [1.0, 2.0],
        }
    )

    cutoff = datetime(2020, 10, 1)
    result = validator.validate_training_data(valid_data, cutoff)
    assert result, "Valid data should pass leakage validation"

    # Test invalid data (has leakage)
    invalid_data = pd.DataFrame(
        {
            "game_date": [datetime(2020, 9, 1), datetime(2020, 11, 1)],  # Future date
            "feature1": [1.0, 2.0],
        }
    )

    try:
        validator.validate_training_data(invalid_data, cutoff)
        raise AssertionError("Should have raised DataValidationError for future data")
    except DataValidationError:
        pass  # Expected

    # Test feature timestamp validation
    features_with_future = pd.DataFrame(
        {
            "feature_timestamp": [
                datetime(2020, 9, 1),
                datetime(2020, 11, 1),
            ],  # Future timestamp
            "feature1": [1.0, 2.0],
        }
    )

    prediction_date = datetime(2020, 10, 1)
    try:
        validator.validate_feature_timestamps(features_with_future, prediction_date)
        raise AssertionError(
            "Should have raised DataValidationError for future features"
        )
    except DataValidationError:
        pass  # Expected

    logger.info("Data leakage validation verified")
    return True


def test_basic_backtest_execution():
    """Test basic backtest execution with mock components."""
    logger.info("Running test_basic_backtest_execution")

    with tempfile.TemporaryDirectory() as temp_dir:
        # Create minimal config for testing
        config = BacktestConfig(
            start_season=2019,
            end_season=2020,  # Very small range
            min_training_seasons=1,
            validation_start_week=16,  # Only validate last few weeks
            validation_level=ValidationLevel.BASIC,  # Minimal validation for speed
            results_path=os.path.join(temp_dir, "results"),
            models_path=os.path.join(temp_dir, "models"),
            save_intermediate_results=False,
            model_types=["wp"],  # Only one model for speed
        )

        backtester = WalkForwardBacktester(config)

        # Create mock components
        data_loader = create_mock_data_loader()
        feature_builder = create_mock_feature_builder()
        model_trainers = {"wp": create_mock_model_trainers()["wp"]}
        evaluator = create_mock_evaluator()

        # Run backtest
        summary = backtester.run_backtest(
            data_loader=data_loader,
            feature_builder=feature_builder,
            model_trainers=model_trainers,
            evaluator=evaluator,
        )

        # Verify summary
        assert isinstance(summary, BacktestSummary), "Should return BacktestSummary"
        assert summary.total_predictions > 0, "Should have made predictions"
        assert len(summary.results_by_season) > 0, "Should have seasonal results"
        assert "wp" in summary.results_by_model, "Should have results for wp model"

        # Verify execution metadata
        assert summary.start_time is not None, "Should have start time"
        assert summary.end_time is not None, "Should have end time"
        assert summary.total_execution_time > 0, "Should have execution time"

    logger.info(
        f"Basic backtest execution verified: {summary.total_predictions} predictions"
    )
    return True


def test_temporal_ordering():
    """Test that temporal ordering is strictly maintained."""
    logger.info("Running test_temporal_ordering")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = BacktestConfig(
            start_season=2019,
            end_season=2020,
            min_training_seasons=1,
            validation_start_week=15,
            validate_temporal_order=True,
            results_path=os.path.join(temp_dir, "results"),
            models_path=os.path.join(temp_dir, "models"),
            model_types=["wp"],
        )

        backtester = WalkForwardBacktester(config)

        # Generate splits and verify temporal ordering
        splits = backtester._generate_season_splits()

        for split in splits:
            # All training seasons should be before validation season
            for training_season in split.training_seasons:
                assert training_season < split.validation_season, (
                    f"Training season {training_season} should be before validation season {split.validation_season}"
                )

            # Training weeks in current season should be before validation weeks
            if split.training_weeks_current_season:
                max_training_week = max(split.training_weeks_current_season)
                min_validation_week = min(split.validation_weeks)
                assert max_training_week < min_validation_week, (
                    "Training weeks should be before validation weeks"
                )

    logger.info("Temporal ordering validated")
    return True


def test_model_integration():
    """Test integration with model training pipeline."""
    logger.info("Running test_model_integration")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = BacktestConfig(
            start_season=2019,
            end_season=2020,
            min_training_seasons=1,
            validation_start_week=17,
            results_path=os.path.join(temp_dir, "results"),
            models_path=os.path.join(temp_dir, "models"),
            save_intermediate_results=True,
            model_types=["wp", "ats"],
        )

        backtester = WalkForwardBacktester(config)

        # Create season split for testing
        split = SeasonSplit(
            validation_season=2020,
            validation_weeks=[17, 18],
            training_seasons=[2019],
            training_weeks_current_season=[1, 2, 3, 4],
            split_date=datetime(2020, 9, 1),
        )

        # Create mock components
        data_loader = create_mock_data_loader()
        feature_builder = create_mock_feature_builder()
        model_trainers = create_mock_model_trainers()

        # Test model training
        training_data = backtester._load_training_data(split, data_loader)
        training_features = backtester._build_training_features(
            split, training_data, feature_builder
        )
        trained_models = backtester._train_models(
            split, training_features, model_trainers
        )

        # Verify models were trained
        assert len(trained_models) == 2, "Should have trained 2 models"
        assert "wp" in trained_models, "Should have wp model"
        assert "ats" in trained_models, "Should have ats model"

        # Verify models can make predictions
        for model_type, model in trained_models.items():
            assert hasattr(model, "predict"), (
                f"{model_type} model should have predict method"
            )

        # Verify model files were saved
        for model_type in ["wp", "ats"]:
            model_path = os.path.join(
                config.models_path, f"{model_type}_season_2020.pkl"
            )
            assert os.path.exists(model_path), f"Model file should exist: {model_path}"

    logger.info("Model integration validated")
    return True


def test_performance_metrics():
    """Test performance metric calculations."""
    logger.info("Running test_performance_metrics")

    # Create mock results
    results = [
        BacktestResult(
            season=2020,
            week=5,
            model_type="wp",
            predictions_made=16,
            accuracy=0.65,
            log_loss=0.45,
            brier_score=0.25,
        ),
        BacktestResult(
            season=2020,
            week=6,
            model_type="wp",
            predictions_made=16,
            accuracy=0.70,
            log_loss=0.40,
            brier_score=0.22,
        ),
        BacktestResult(
            season=2020,
            week=5,
            model_type="ats",
            predictions_made=16,
            mae=0.15,
            rmse=0.20,
        ),
        BacktestResult(
            season=2020,
            week=6,
            model_type="ats",
            predictions_made=16,
            mae=0.12,
            rmse=0.18,
        ),
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        config = BacktestConfig(results_path=temp_dir)
        backtester = WalkForwardBacktester(config)

        # Calculate overall metrics
        overall_metrics = backtester._calculate_overall_metrics(results)

        # Verify WP metrics
        assert "wp_mean_accuracy" in overall_metrics, "Should have wp mean accuracy"
        assert abs(overall_metrics["wp_mean_accuracy"] - 0.675) < 1e-6, (
            "Mean accuracy should be 0.675"
        )
        assert "wp_mean_log_loss" in overall_metrics, "Should have wp mean log loss"

        # Verify ATS metrics
        assert "ats_mean_mae" in overall_metrics, "Should have ats mean MAE"
        assert abs(overall_metrics["ats_mean_mae"] - 0.135) < 1e-6, (
            "Mean MAE should be 0.135"
        )

        # Verify overall stats
        assert overall_metrics["total_predictions"] == 64, (
            "Should have 64 total predictions"
        )
        assert overall_metrics["total_weeks"] == 2, "Should have 2 unique weeks"

    logger.info("Performance metrics calculation validated")
    return True


def test_result_persistence():
    """Test saving and loading of backtest results."""
    logger.info("Running test_result_persistence")

    with tempfile.TemporaryDirectory() as temp_dir:
        config = BacktestConfig(
            start_season=2019,
            end_season=2020,
            min_training_seasons=1,
            validation_start_week=17,
            results_path=os.path.join(temp_dir, "results"),
            models_path=os.path.join(temp_dir, "models"),
            save_intermediate_results=True,
        )

        backtester = WalkForwardBacktester(config)

        # Create mock results
        results = [
            BacktestResult(
                season=2020,
                week=17,
                model_type="wp",
                predictions_made=16,
                accuracy=0.65,
            )
        ]

        # Save checkpoint
        backtester._save_checkpoint(results, 1)

        # Verify checkpoint file
        checkpoint_path = os.path.join(
            config.results_path, "checkpoints", "checkpoint_1.json"
        )
        assert os.path.exists(checkpoint_path), "Checkpoint file should exist"

        # Create and save summary
        summary = BacktestSummary(
            config=config,
            total_seasons=1,
            total_weeks=1,
            total_predictions=16,
            results_by_season={2020: results},
            results_by_model={"wp": results},
            overall_metrics={"wp_mean_accuracy": 0.65},
            seasonal_trends={"wp_accuracy": [0.65]},
            start_time=datetime.now(),
        )

        backtester._save_final_results(summary)

        # Verify final result files
        summary_path = os.path.join(config.results_path, "backtest_summary.json")
        results_path = os.path.join(config.results_path, "detailed_results.pkl")

        assert os.path.exists(summary_path), "Summary JSON should exist"
        assert os.path.exists(results_path), "Detailed results pickle should exist"

        # Verify we can load the results
        with open(results_path, "rb") as f:
            loaded_summary = pickle.load(f)
            assert loaded_summary.total_predictions == 16, (
                "Loaded summary should match original"
            )

    logger.info("Result persistence validated")
    return True


def test_error_handling():
    """Test error handling and recovery."""
    logger.info("Running test_error_handling")

    # Test with strict validation (should fail fast)
    with tempfile.TemporaryDirectory() as temp_dir:
        strict_config = BacktestConfig(
            start_season=2019,
            end_season=2020,
            validation_level=ValidationLevel.STRICT,
            check_data_leakage=True,
            results_path=os.path.join(temp_dir, "results"),
            models_path=os.path.join(temp_dir, "models"),
        )

        backtester = WalkForwardBacktester(strict_config)

        # Create data loader that returns future data (should trigger leakage error)
        def bad_data_loader(season: int, weeks: list[int] | None = None):
            if weeks is None:
                weeks = [1, 2]

            future_date = datetime(season + 1, 1, 1)  # Future date
            return pd.DataFrame(
                {
                    "game_date": [future_date] * 10,
                    "wp_actual": np.random.random(10),
                    "ats_actual": np.random.random(10),
                    "ou_actual": np.random.random(10),
                }
            )

        # Should raise error in strict mode
        try:
            backtester.run_backtest(
                data_loader=bad_data_loader,
                feature_builder=create_mock_feature_builder(),
                model_trainers=create_mock_model_trainers(),
                evaluator=create_mock_evaluator(),
            )
            raise AssertionError("Should have raised BacktestError in strict mode")
        except (BacktestError, DataValidationError):
            pass  # Expected

    logger.info("Error handling validated")
    return True


def run_all_tests():
    """Run all walk-forward backtesting tests."""
    logger.info("Starting walk-forward backtesting test suite")

    tests = [
        test_config_creation,
        test_season_splits_generation,
        test_data_leakage_validator,
        test_basic_backtest_execution,
        test_temporal_ordering,
        test_model_integration,
        test_performance_metrics,
        test_result_persistence,
        test_error_handling,
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

    logger.info(
        f"Walk-forward backtesting tests completed: {passed} passed, {failed} failed"
    )

    if failed == 0:
        logger.info("All walk-forward backtesting tests passed!")
    else:
        logger.error(f"{failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
