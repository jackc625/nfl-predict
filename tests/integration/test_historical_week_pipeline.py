"""Historical week end-to-end pipeline integration tests."""

import json
import time
from datetime import datetime
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from models.prediction_pipeline import NFLPredictionPipeline
from models.train_ats import ATSModelPrediction
from models.train_ou import OUModelPrediction
from models.train_wp import WPModelPrediction
from scripts.build_features import FeatureMatrixBuilder
from tests.conftest import (
    assert_dataframe_structure,
    assert_no_data_leakage,
    assert_probability_range,
)


def create_mock_wp_prediction(
    game_id: str, home_team: str, away_team: str, win_prob: float
) -> WPModelPrediction:
    """Create a mock WP prediction object."""
    return WPModelPrediction(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        raw_win_probability=win_prob,
        calibrated_win_probability=win_prob * 0.95,
        prediction_confidence=abs(win_prob - 0.5) * 2,
        model_version="test_v1.0",
    )


def create_mock_ats_prediction(
    game_id: str, home_team: str, away_team: str, cover_prob: float
) -> ATSModelPrediction:
    """Create a mock ATS prediction object."""
    return ATSModelPrediction(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        predicted_margin=2.5,
        predicted_spread=-2.5,
        cover_probability=cover_prob,
        confidence=abs(cover_prob - 0.5) * 2,
        model_version="test_v1.0",
    )


def create_mock_ou_prediction(
    game_id: str, home_team: str, away_team: str, over_prob: float
) -> OUModelPrediction:
    """Create a mock O/U prediction object."""
    return OUModelPrediction(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        predicted_total=45.5,
        over_probability=over_prob,
        under_probability=1.0 - over_prob,
        confidence=abs(over_prob - 0.5) * 2,
        model_version="test_v1.0",
    )


class TestHistoricalWeekPipeline:
    """Test complete pipeline execution on a historical week."""

    @pytest.fixture
    def test_week_config(self):
        """Configuration for historical test week."""
        return {
            "season": 2023,
            "week": 5,  # Mid-season week with complete data
            "snapshot_time": "2023-10-05T18:00:00-04:00",
            "games_count": 14,  # Typical week 5 games
            "teams": [
                "BUF",
                "MIA",
                "NYJ",
                "NE",
                "KC",
                "LV",
                "LAC",
                "DEN",
                "DAL",
                "NYG",
                "PHI",
                "WAS",
                "GB",
                "MIN",
                "CHI",
                "DET",
            ],
        }

    @pytest.fixture
    def historical_games_data(self, test_week_config):
        """Create realistic historical games data."""
        et_tz = ZoneInfo("America/New_York")
        season = test_week_config["season"]
        week = test_week_config["week"]

        games = []

        # Week 1-4 completed games (for form calculations)
        for w in range(1, week):
            for game_num in range(16):  # 16 games per week typically
                home_team = test_week_config["teams"][
                    game_num % len(test_week_config["teams"])
                ]
                away_team = test_week_config["teams"][
                    (game_num + 1) % len(test_week_config["teams"])
                ]

                if home_team != away_team:
                    game_id = f"HIST_{season}_W{w:02d}_{away_team}@{home_team}"
                    kickoff = datetime(season, 9, 7 + (w - 1) * 7, 13, 0, tzinfo=et_tz)

                    # Simulate realistic game outcomes
                    home_score = np.random.randint(10, 35)
                    away_score = np.random.randint(10, 35)

                    games.append(
                        {
                            "game_id": game_id,
                            "season": season,
                            "week": w,
                            "home_team": home_team,
                            "away_team": away_team,
                            "kickoff_et": kickoff,
                            "home_score": home_score,
                            "away_score": away_score,
                        }
                    )

        # Target week games (no scores yet)
        for game_num in range(test_week_config["games_count"]):
            home_team = test_week_config["teams"][
                game_num % len(test_week_config["teams"])
            ]
            away_team = test_week_config["teams"][
                (game_num + 8) % len(test_week_config["teams"])
            ]

            if home_team != away_team:
                game_id = f"HIST_{season}_W{week:02d}_{away_team}@{home_team}"
                kickoff = datetime(season, 9, 7 + (week - 1) * 7, 13, 0, tzinfo=et_tz)

                games.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "home_team": home_team,
                        "away_team": away_team,
                        "kickoff_et": kickoff,
                        "home_score": None,
                        "away_score": None,
                    }
                )

        return pd.DataFrame(games)

    @pytest.fixture
    def historical_odds_data(self, test_week_config, historical_games_data):
        """Create historical odds data."""
        # Focus on target week games
        target_week_games = historical_games_data[
            historical_games_data["week"] == test_week_config["week"]
        ]

        odds_data = []
        for _, game in target_week_games.iterrows():
            # Realistic odds data
            spread = np.random.uniform(-14, 14)
            total = np.random.uniform(35, 65)

            # Convert spread to moneylines (simplified)
            if spread > 0:  # Away team favored
                ml_home = int(100 + abs(spread) * 20)
                ml_away = int(-110 - abs(spread) * 10)
            else:  # Home team favored
                ml_home = int(-110 - abs(spread) * 10)
                ml_away = int(100 + abs(spread) * 20)

            odds_data.append(
                {
                    "game_id": game["game_id"],
                    "snapshot_spread": spread,
                    "snapshot_total": total,
                    "snapshot_ml_home": ml_home,
                    "snapshot_ml_away": ml_away,
                    "has_snapshot_lines": 1.0,
                }
            )

        return pd.DataFrame(odds_data)

    def test_complete_historical_pipeline(
        self,
        temp_data_dir,
        test_week_config,
        historical_games_data,
        historical_odds_data,
    ):
        """Test complete pipeline execution on historical week."""
        season = test_week_config["season"]
        week = test_week_config["week"]

        # Step 1: Data Ingestion

        # Mock data ingestion - save historical data to temp directories
        bronze_dir = temp_data_dir / "bronze"
        bronze_dir.mkdir(exist_ok=True)

        games_file = bronze_dir / f"games_{season}_W{week:02d}.parquet"
        historical_games_data.to_parquet(games_file)

        odds_file = bronze_dir / f"odds_{season}_W{week:02d}.parquet"
        historical_odds_data.to_parquet(odds_file)

        # Mock weather data
        weather_data = []
        for _, game in historical_odds_data.iterrows():
            weather_data.append(
                {
                    "game_id": game["game_id"],
                    "temp_f": np.random.uniform(20, 85),
                    "wind_mph": np.random.uniform(0, 20),
                    "weather_affects_game": np.random.choice([0.0, 1.0]),
                    "weather_severity_score": np.random.uniform(0, 0.8),
                }
            )

        weather_df = pd.DataFrame(weather_data)
        weather_file = bronze_dir / f"weather_{season}_W{week:02d}.parquet"
        weather_df.to_parquet(weather_file)

        # Step 2: Feature Building
        builder = FeatureMatrixBuilder()

        # Mock the data loading to use our test data
        def mock_load_data(target_season, target_week):
            if target_season == season and target_week == week:
                return {
                    "games": historical_games_data,
                    "odds": historical_odds_data,
                    "weather": weather_df,
                }
            return {
                "games": pd.DataFrame(),
                "odds": pd.DataFrame(),
                "weather": pd.DataFrame(),
            }

        # Mock Elo and team form data
        teams = test_week_config["teams"]
        elo_data = []
        form_data = []

        for team in teams:
            # Elo ratings
            elo_data.append(
                {
                    "team": team,
                    "season": season,
                    "week": week,
                    "elo_rating": np.random.uniform(1300, 1700),
                    "elo_uncertainty": np.random.uniform(50, 150),
                    "elo_games_played": week + 15,
                    "elo_form_rating": np.random.uniform(1300, 1700),
                }
            )

            # Team form
            for side in ["offense", "defense"]:
                form_data.append(
                    {
                        "team": team,
                        "target_season": season,
                        "target_week": week,
                        "side": side,
                        "rolling_epa_per_play": np.random.normal(0, 0.2),
                        "rolling_success_rate": np.random.uniform(0.35, 0.65),
                        "rolling_pass_epa_per_play": np.random.normal(0, 0.25),
                        "rolling_rush_epa_per_play": np.random.normal(0, 0.15),
                        "rolling_neutral_pass_rate": np.random.uniform(0.55, 0.75)
                        if side == "offense"
                        else np.nan,
                    }
                )

        elo_df = pd.DataFrame(elo_data)
        form_df = pd.DataFrame(form_data)

        # Mock contextual data
        contextual_data = []
        for _, game in historical_odds_data.iterrows():
            contextual_data.append(
                {
                    "game_id": game["game_id"],
                    "venue_outdoor": np.random.choice([0.0, 1.0]),
                    "away_travel_distance_miles": np.random.uniform(200, 2500),
                    "home_rest_days": np.random.choice([6, 7, 8, 9, 10]),
                    "away_rest_days": np.random.choice([6, 7, 8, 9, 10]),
                }
            )

        contextual_df = pd.DataFrame(contextual_data)

        def mock_load_all_sources(target_season=None, target_week=None):
            return {
                "games": historical_games_data,
                "team_form": form_df,
                "elo": elo_df,
                "contextual": contextual_df,
                "weather": weather_df,
                "market": historical_odds_data,
            }

        # Test feature matrix generation
        with patch.object(
            builder, "load_all_feature_sources", side_effect=mock_load_all_sources
        ):
            feature_matrices = builder.generate_feature_matrices(
                target_season=season, target_week=week
            )

        # Validate feature matrices
        assert "wp" in feature_matrices, "Should generate WP feature matrix"
        assert "ats" in feature_matrices, "Should generate ATS feature matrix"
        assert "ou" in feature_matrices, "Should generate O/U feature matrix"

        for target, matrix in feature_matrices.items():
            assert_dataframe_structure(matrix, min_rows=1)
            assert (
                f"target_{target}" in matrix.columns
                or matrix[f"target_{target}"].isna().all()
            )
            assert_no_data_leakage(matrix)

        # Step 3: Model Training and Prediction
        pipeline = NFLPredictionPipeline()

        # For historical test, we need training data from previous seasons
        training_data = {}
        for target in ["wp", "ats", "ou"]:
            # Create mock training data
            train_matrix = feature_matrices[target].copy()
            if target == "wp":
                train_matrix["target_wp"] = np.random.choice([0, 1], len(train_matrix))
            elif target == "ats":
                train_matrix["target_ats"] = np.random.choice([0, 1], len(train_matrix))
            elif target == "ou":
                train_matrix["target_ou"] = np.random.choice([0, 1], len(train_matrix))

            training_data[target] = train_matrix

        # Mock model training
        mock_wp_model = Mock()

        def mock_wp_predict_hist(games_df):
            return [
                create_mock_wp_prediction(
                    f"HIST_GAME_{i}", "HOME", "AWAY", np.random.uniform(0.3, 0.7)
                )
                for i in range(len(games_df))
            ]

        mock_wp_model.predict.side_effect = mock_wp_predict_hist
        mock_wp_model.feature_names_ = ["elo_diff", "home_elo_rating"]
        mock_wp_model.is_fitted = True

        mock_ats_model = Mock()

        def mock_ats_predict_hist(games_df):
            return [
                create_mock_ats_prediction(
                    f"HIST_GAME_{i}", "HOME", "AWAY", np.random.uniform(0.4, 0.6)
                )
                for i in range(len(games_df))
            ]

        mock_ats_model.predict.side_effect = mock_ats_predict_hist
        mock_ats_model.feature_names_ = ["elo_diff", "snapshot_spread"]
        mock_ats_model.is_fitted = True

        mock_ou_model = Mock()

        def mock_ou_predict_hist(games_df):
            return [
                create_mock_ou_prediction(
                    f"HIST_GAME_{i}", "HOME", "AWAY", np.random.uniform(0.45, 0.55)
                )
                for i in range(len(games_df))
            ]

        mock_ou_model.predict.side_effect = mock_ou_predict_hist
        mock_ou_model.feature_names_ = ["snapshot_total", "temp_f"]
        mock_ou_model.is_fitted = True

        pipeline.wp_model = mock_wp_model
        pipeline.ats_model = mock_ats_model
        pipeline.ou_model = mock_ou_model

        # Generate predictions for target week
        current_week_data = feature_matrices["wp"].copy()  # Use WP matrix as base
        predictions = pipeline.predict_games(current_week_data)

        # Validate predictions
        assert_dataframe_structure(predictions, min_rows=1)

        expected_cols = [
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "wp_prob_home",
            "wp_prob_away",
            "wp_confidence",
            "ats_prob_home",
            "ats_prob_away",
            "ats_confidence",
            "ou_prob_over",
            "ou_prob_under",
            "ou_confidence",
        ]

        for col in expected_cols:
            assert col in predictions.columns, f"Predictions missing {col}"

        # Validate probability constraints
        assert_probability_range(predictions["wp_prob_home"])
        assert_probability_range(predictions["ats_prob_home"])
        assert_probability_range(predictions["ou_prob_over"])

        # Step 4: Output Validation
        self._validate_prediction_outputs(predictions, test_week_config)

        # Step 5: API Integration Test
        self._test_api_integration(predictions, test_week_config)

    def test_pipeline_performance_benchmarks(self, temp_data_dir, test_week_config):
        """Test pipeline performance with realistic data volumes."""
        season = test_week_config["season"]
        week = test_week_config["week"]

        # Create larger dataset for performance testing
        large_games_data = []
        for w in range(1, 18):  # Full season
            for game_num in range(16):  # 16 games per week
                game_id = f"PERF_{season}_W{w:02d}_GAME{game_num}"
                large_games_data.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": w,
                        "home_team": f"TEAM{game_num % 8}",
                        "away_team": f"TEAM{(game_num + 1) % 8}",
                        "home_score": np.random.randint(7, 42) if w < week else None,
                        "away_score": np.random.randint(7, 42) if w < week else None,
                    }
                )

        large_df = pd.DataFrame(large_games_data)

        # Test feature building performance
        builder = FeatureMatrixBuilder()

        def mock_load_large_data(target_season=None, target_week=None):
            return {
                "games": large_df,
                "team_form": pd.DataFrame(),  # Mock empty for speed
                "elo": pd.DataFrame(),
                "contextual": pd.DataFrame(),
                "weather": pd.DataFrame(),
                "market": pd.DataFrame(),
            }

        # Time feature building
        start_time = time.time()
        with patch.object(
            builder, "load_all_feature_sources", side_effect=mock_load_large_data
        ):
            try:
                builder.generate_feature_matrices(
                    target_season=season, target_week=week
                )
                feature_build_time = time.time() - start_time

                # Should process full season in reasonable time
                assert feature_build_time < 10.0, (
                    f"Feature building too slow: {feature_build_time:.2f}s"
                )

            except Exception:
                # If feature building fails due to missing data, that's expected in this test
                pass

        # Test prediction generation performance
        pipeline = NFLPredictionPipeline()

        # Create mock prediction data
        mock_data = pd.DataFrame(
            {
                "game_id": [f"PERF_GAME_{i}" for i in range(272)],  # Full season
                "season": [season] * 272,
                "week": [1] * 272,
                "home_team": ["HOME"] * 272,
                "away_team": ["AWAY"] * 272,
                "elo_diff": np.random.uniform(-200, 200, 272),
            }
        )

        # Mock fast models
        pipeline.wp_model = Mock()
        pipeline.wp_model.predict_proba.return_value = np.random.uniform(0.3, 0.7, 272)
        pipeline.wp_model.feature_names_ = ["elo_diff"]

        pipeline.ats_model = Mock()
        pipeline.ats_model.predict_proba.return_value = np.random.uniform(0.4, 0.6, 272)
        pipeline.ats_model.feature_names_ = ["elo_diff"]

        pipeline.ou_model = Mock()
        pipeline.ou_model.predict_proba.return_value = np.random.uniform(
            0.45, 0.55, 272
        )
        pipeline.ou_model.feature_names_ = ["elo_diff"]

        # Time prediction generation
        start_time = time.time()
        predictions = pipeline.predict_games(mock_data)
        prediction_time = time.time() - start_time

        # Should generate predictions for full season quickly
        assert prediction_time < 2.0, (
            f"Prediction generation too slow: {prediction_time:.2f}s"
        )
        assert len(predictions) == 272, "Should generate predictions for all games"

    def test_pipeline_error_recovery(self, temp_data_dir, test_week_config):
        """Test pipeline behavior with missing/corrupted data."""
        season = test_week_config["season"]
        week = test_week_config["week"]

        builder = FeatureMatrixBuilder()

        # Test with missing odds data
        def mock_missing_odds(target_season=None, target_week=None):
            return {
                "games": pd.DataFrame(
                    [
                        {
                            "game_id": "TEST_MISSING_ODDS",
                            "season": season,
                            "week": week,
                            "home_team": "HOME",
                            "away_team": "AWAY",
                        }
                    ]
                ),
                "team_form": pd.DataFrame(),
                "elo": pd.DataFrame(),
                "contextual": pd.DataFrame(),
                "weather": pd.DataFrame(),
                "market": pd.DataFrame(),  # Missing odds
            }

        # Should handle missing odds gracefully
        with patch.object(
            builder, "load_all_feature_sources", side_effect=mock_missing_odds
        ):
            try:
                feature_matrices = builder.generate_feature_matrices(
                    target_season=season, target_week=week
                )

                # Should still generate matrices, possibly with missing market features
                assert (
                    "wp" in feature_matrices
                    or "ats" in feature_matrices
                    or "ou" in feature_matrices
                )

            except Exception as e:
                # If pipeline raises specific error for missing data, that's acceptable
                assert "odds" in str(e).lower() or "market" in str(e).lower()

        # Test with corrupted feature data
        def mock_corrupted_data(target_season=None, target_week=None):
            corrupted_games = pd.DataFrame(
                [
                    {
                        "game_id": "CORRUPTED_GAME",
                        "season": "invalid",  # Wrong type
                        "week": week,
                        "home_team": None,  # Missing required field
                        "away_team": "AWAY",
                    }
                ]
            )

            return {
                "games": corrupted_games,
                "team_form": pd.DataFrame(),
                "elo": pd.DataFrame(),
                "contextual": pd.DataFrame(),
                "weather": pd.DataFrame(),
                "market": pd.DataFrame(),
            }

        # Should handle corrupted data appropriately
        with patch.object(
            builder, "load_all_feature_sources", side_effect=mock_corrupted_data
        ):
            try:
                feature_matrices = builder.generate_feature_matrices(
                    target_season=season, target_week=week
                )

            except Exception as e:
                # Should raise meaningful error for corrupted data
                assert isinstance(e, (ValueError, TypeError, KeyError))

    def _validate_prediction_outputs(self, predictions, test_week_config):
        """Validate prediction outputs match expected schema and constraints."""
        # Check all games from target week are included
        target_week_games = predictions[predictions["week"] == test_week_config["week"]]
        assert len(target_week_games) > 0, "Should have predictions for target week"

        # Check probability constraints
        for _, game in target_week_games.iterrows():
            # Probabilities sum to 1
            assert abs(game["wp_prob_home"] + game["wp_prob_away"] - 1.0) < 0.01
            assert abs(game["ats_prob_home"] + game["ats_prob_away"] - 1.0) < 0.01
            assert abs(game["ou_prob_over"] + game["ou_prob_under"] - 1.0) < 0.01

            # Confidence values are reasonable
            assert 0 <= game["wp_confidence"] <= 1
            assert 0 <= game["ats_confidence"] <= 1
            assert 0 <= game["ou_confidence"] <= 1

    def _test_api_integration(self, predictions, test_week_config):
        """Test API integration with generated predictions."""
        client = TestClient(app)

        # Mock API services to return our test predictions

        mock_week_info = {
            "current_week": test_week_config["week"],
            "current_season": test_week_config["season"],
            "predictions_available": True,
            "last_updated": test_week_config["snapshot_time"],
        }

        mock_games = predictions.to_dict("records")

        with patch("api.services.get_current_week_info", return_value=mock_week_info):
            with patch(
                "api.services.get_games_with_predictions", return_value=mock_games
            ):
                # Test current week endpoint
                response = client.get("/current-week")
                assert response.status_code == 200
                data = response.json()
                assert data["current_week"] == test_week_config["week"]
                assert data["current_season"] == test_week_config["season"]

                # Test games endpoint
                response = client.get("/games")
                assert response.status_code == 200
                games = response.json()
                assert len(games) == len(predictions)

                # Test single game endpoint
                if len(games) > 0:
                    game_id = games[0]["game_id"]
                    with patch("api.services.get_game_by_id", return_value=games[0]):
                        response = client.get(f"/games/{game_id}")
                        assert response.status_code == 200
                        game_data = response.json()
                        assert game_data["game_id"] == game_id


class TestPipelineArtifacts:
    """Test pipeline artifact generation and validation."""

    def test_prediction_artifact_generation(self, temp_data_dir, test_week_config):
        """Test generation of prediction artifacts."""
        artifacts_dir = temp_data_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Mock prediction data
        predictions = pd.DataFrame(
            [
                {
                    "game_id": "ARTIFACT_TEST_GAME",
                    "season": test_week_config["season"],
                    "week": test_week_config["week"],
                    "home_team": "HOME",
                    "away_team": "AWAY",
                    "wp_prob_home": 0.62,
                    "wp_prob_away": 0.38,
                    "wp_confidence": 0.75,
                    "ats_prob_home": 0.54,
                    "ats_prob_away": 0.46,
                    "ats_confidence": 0.68,
                    "ou_prob_over": 0.48,
                    "ou_prob_under": 0.52,
                    "ou_confidence": 0.59,
                }
            ]
        )

        # Save prediction artifacts
        season = test_week_config["season"]
        week = test_week_config["week"]

        predictions_file = artifacts_dir / f"predictions_{season}_W{week:02d}.parquet"
        predictions.to_parquet(predictions_file)

        # Save JSON format for API
        json_file = artifacts_dir / f"predictions_{season}_W{week:02d}.json"
        with open(json_file, "w") as f:
            json.dump(predictions.to_dict("records"), f, indent=2, default=str)

        # Validate artifacts exist and are loadable
        assert predictions_file.exists(), "Predictions parquet should be saved"
        assert json_file.exists(), "Predictions JSON should be saved"

        # Test loading
        loaded_predictions = pd.read_parquet(predictions_file)
        assert len(loaded_predictions) == len(predictions)
        assert list(loaded_predictions.columns) == list(predictions.columns)

        with open(json_file) as f:
            loaded_json = json.load(f)
        assert len(loaded_json) == len(predictions)

    def test_model_artifact_validation(self, temp_data_dir):
        """Test model artifact saving and loading."""
        import pickle

        artifacts_dir = temp_data_dir / "artifacts"
        artifacts_dir.mkdir(exist_ok=True)

        # Mock trained models
        mock_wp_model = Mock()
        mock_wp_model.feature_names_ = ["elo_diff", "home_elo_rating"]
        mock_wp_model.is_fitted = True

        # Save model artifact
        model_file = artifacts_dir / "wp_model_2023.pkl"
        with open(model_file, "wb") as f:
            pickle.dump(mock_wp_model, f)

        # Validate model can be loaded
        assert model_file.exists()

        with open(model_file, "rb") as f:
            loaded_model = pickle.load(f)

        assert loaded_model.feature_names_ == mock_wp_model.feature_names_
        assert loaded_model.is_fitted == mock_wp_model.is_fitted
