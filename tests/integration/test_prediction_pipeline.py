"""Integration tests for NFLPredictionPipeline with artifact-loaded models
(Gap 12-03-01, ACCU-05).

Verifies the rewired pipeline end-to-end:
  save minimal artifacts -> load_models -> predict_games -> valid predictions

Models are saved via save_model_artifact with fitted toy models (no real data).
All three artifacts (wp/ats/ou) are written to a tmp_path artifacts dir.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

from models.artifacts import save_model_artifact
from models.prediction_pipeline import (
    NFLPredictionPipeline,
    UnifiedGamePrediction,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _fit_wp_model():
    """Fit a minimal LogisticRegression on balanced binary data."""
    rng = np.random.RandomState(7)
    n = 40
    X = rng.randn(n, 3)
    y = np.array([i % 2 for i in range(n)])  # alternating 0/1
    model = LogisticRegression(max_iter=200, C=1.0, solver="lbfgs")
    model.fit(X, y)
    return model, ["wp_feat_a", "wp_feat_b", "wp_feat_c"]


def _fit_ats_model():
    """Fit a minimal XGBRegressor for margin prediction."""
    rng = np.random.RandomState(8)
    n = 40
    X = rng.randn(n, 3)
    y = X[:, 0] * 3.0 + rng.randn(n)
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42, verbosity=0)
    model.fit(X, y)
    return model, ["ats_feat_a", "ats_feat_b", "ats_feat_c"]


def _fit_ou_model():
    """Fit a minimal XGBRegressor for total points prediction."""
    rng = np.random.RandomState(9)
    n = 40
    X = rng.randn(n, 3)
    y = 45.0 + X[:, 0] * 5.0 + rng.randn(n)
    model = XGBRegressor(n_estimators=5, max_depth=2, random_state=42, verbosity=0)
    model.fit(X, y)
    return model, ["ou_feat_a", "ou_feat_b", "ou_feat_c"]


@pytest.fixture
def artifacts_dir(tmp_path):
    """Save minimal WP/ATS/O/U artifacts to tmp_path and return the path."""
    wp_model, wp_features = _fit_wp_model()
    save_model_artifact(
        model=wp_model,
        target="wp",
        metadata={"version": "test-wp", "residual_std": 13.5},
        feature_list=wp_features,
        artifacts_dir=tmp_path,
    )

    ats_model, ats_features = _fit_ats_model()
    save_model_artifact(
        model=ats_model,
        target="ats",
        metadata={"version": "test-ats", "residual_std": 13.5},
        feature_list=ats_features,
        artifacts_dir=tmp_path,
    )

    ou_model, ou_features = _fit_ou_model()
    save_model_artifact(
        model=ou_model,
        target="ou",
        metadata={"version": "test-ou", "residual_std": 13.0},
        feature_list=ou_features,
        artifacts_dir=tmp_path,
    )

    return tmp_path


@pytest.fixture
def games_df():
    """Synthetic games DataFrame with columns matching all three feature lists."""
    rng = np.random.RandomState(42)
    n = 4
    return pd.DataFrame(
        {
            "game_id": [f"2024_01_T{i}_T{i + 1}" for i in range(n)],
            "home_team": ["KC", "BAL", "SF", "DAL"],
            "away_team": ["BUF", "MIA", "PHI", "NYG"],
            # WP features
            "wp_feat_a": rng.randn(n),
            "wp_feat_b": rng.randn(n),
            "wp_feat_c": rng.randn(n),
            # ATS features
            "ats_feat_a": rng.randn(n),
            "ats_feat_b": rng.randn(n),
            "ats_feat_c": rng.randn(n),
            # O/U features
            "ou_feat_a": rng.randn(n),
            "ou_feat_b": rng.randn(n),
            "ou_feat_c": rng.randn(n),
            # Optional market lines
            "market_moneyline_home": [-150, -120, 110, -200],
            "market_moneyline_away": [130, 100, -130, 170],
            "market_spread": [-3.5, -2.5, 1.5, -6.5],
            "market_total": [47.5, 44.5, 49.5, 43.5],
        }
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPipelineLoadModels:
    """load_models populates all three artifact dicts from disk."""

    def test_load_models_sets_all_three_artifacts(self, artifacts_dir):
        """After load_models, all three artifacts are non-None dicts."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        assert pipeline.wp_artifact is not None
        assert pipeline.ats_artifact is not None
        assert pipeline.ou_artifact is not None

    def test_loaded_artifacts_have_correct_keys(self, artifacts_dir):
        """Each artifact dict contains model, metadata, feature_list, calibrator, params."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        for artifact in [
            pipeline.wp_artifact,
            pipeline.ats_artifact,
            pipeline.ou_artifact,
        ]:
            assert artifact is not None
            assert "model" in artifact
            assert "metadata" in artifact
            assert "feature_list" in artifact
            assert "calibrator" in artifact
            assert "params" in artifact

    def test_loaded_wp_model_supports_predict_proba(self, artifacts_dir):
        """The loaded WP model is a real classifier with predict_proba."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        assert pipeline.wp_artifact is not None
        wp_model = pipeline.wp_artifact["model"]
        assert hasattr(wp_model, "predict_proba")

    def test_loaded_ats_model_supports_predict(self, artifacts_dir):
        """The loaded ATS model is a real regressor with predict."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        assert pipeline.ats_artifact is not None
        ats_model = pipeline.ats_artifact["model"]
        assert hasattr(ats_model, "predict")

    def test_loaded_ou_model_supports_predict(self, artifacts_dir):
        """The loaded O/U model is a real regressor with predict."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        assert pipeline.ou_artifact is not None
        ou_model = pipeline.ou_artifact["model"]
        assert hasattr(ou_model, "predict")


class TestPredictGamesEndToEnd:
    """predict_games returns valid UnifiedGamePrediction objects."""

    def test_predict_games_returns_list_of_unified_predictions(
        self, artifacts_dir, games_df
    ):
        """predict_games returns a list of UnifiedGamePrediction."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_df)

        assert isinstance(results, list)
        assert len(results) == len(games_df)
        for pred in results:
            assert isinstance(pred, UnifiedGamePrediction)

    def test_predict_games_wp_probabilities_in_range(self, artifacts_dir, games_df):
        """WP probabilities from predict_games are in [0, 1]."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_df)

        for pred in results:
            assert 0.0 <= pred.wp_home_probability <= 1.0
            assert 0.0 <= pred.wp_away_probability <= 1.0
            # Must sum to ~1
            total = pred.wp_home_probability + pred.wp_away_probability
            assert abs(total - 1.0) < 0.01, f"WP probabilities don't sum to 1: {total}"

    def test_predict_games_ats_margin_is_finite(self, artifacts_dir, games_df):
        """ATS predicted margin is a finite float for every game."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_df)

        for pred in results:
            assert np.isfinite(pred.predicted_margin), (
                f"Non-finite ATS margin: {pred.predicted_margin}"
            )

    def test_predict_games_ats_cover_probability_in_range(
        self, artifacts_dir, games_df
    ):
        """ATS cover probability is in [0, 1] for every game."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_df)

        for pred in results:
            assert 0.0 <= pred.ats_cover_probability <= 1.0

    def test_predict_games_ou_total_is_reasonable(self, artifacts_dir, games_df):
        """O/U predicted total is a finite positive float."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_df)

        for pred in results:
            assert np.isfinite(pred.predicted_total), (
                f"Non-finite O/U total: {pred.predicted_total}"
            )

    def test_predict_games_ou_probs_sum_to_one(self, artifacts_dir, games_df):
        """Over + Under probabilities sum to approximately 1."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_df)

        for pred in results:
            total = pred.over_probability + pred.under_probability
            assert abs(total - 1.0) < 0.01, f"Over+Under don't sum to 1: {total}"

    def test_predict_games_correct_game_ids(self, artifacts_dir, games_df):
        """Each UnifiedGamePrediction has the correct game_id from the input."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_df)

        for i, pred in enumerate(results):
            expected_id = games_df.iloc[i]["game_id"]
            assert pred.game_id == expected_id

    def test_predict_games_fair_lines_populated(self, artifacts_dir, games_df):
        """predict_games populates fair_lines dict with all 6 bet types."""
        from models.prediction_pipeline import BetType

        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_df)

        expected_bet_types = {
            BetType.MONEYLINE_HOME,
            BetType.MONEYLINE_AWAY,
            BetType.SPREAD_HOME,
            BetType.SPREAD_AWAY,
            BetType.OVER,
            BetType.UNDER,
        }

        for pred in results:
            assert set(pred.fair_lines.keys()) == expected_bet_types

    def test_predict_games_without_market_lines(self, artifacts_dir):
        """predict_games works even when market line columns are absent."""
        rng = np.random.RandomState(55)
        n = 3
        games_no_market = pd.DataFrame(
            {
                "game_id": ["g1", "g2", "g3"],
                "home_team": ["KC", "BAL", "SF"],
                "away_team": ["BUF", "MIA", "PHI"],
                "wp_feat_a": rng.randn(n),
                "wp_feat_b": rng.randn(n),
                "wp_feat_c": rng.randn(n),
                "ats_feat_a": rng.randn(n),
                "ats_feat_b": rng.randn(n),
                "ats_feat_c": rng.randn(n),
                "ou_feat_a": rng.randn(n),
                "ou_feat_b": rng.randn(n),
                "ou_feat_c": rng.randn(n),
                # No market line columns
            }
        )

        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=artifacts_dir)

        results = pipeline.predict_games(games_no_market)

        assert len(results) == n
        for pred in results:
            assert isinstance(pred, UnifiedGamePrediction)
            assert 0.0 <= pred.wp_home_probability <= 1.0
