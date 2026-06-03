"""Unit tests for the rewired NFLPredictionPipeline.

Tests cover:
- Pipeline initialization with artifact dicts
- Model loading via load_model_artifact
- Model loaded checks
- OddsConverter round-trip conversions
- Prediction dataclass instantiation
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from xgboost import XGBRegressor

from models.artifacts import save_model_artifact
from models.prediction_pipeline import (
    ATSPrediction,
    NFLPredictionPipeline,
    OddsConverter,
    OUPrediction,
    WPPrediction,
)

# -- Fixtures --


@pytest.fixture
def tiny_wp_data():
    """Generate minimal training data for a WP classifier."""
    rng = np.random.RandomState(42)
    n = 30
    X = pd.DataFrame(
        {
            "feat_a": rng.randn(n),
            "feat_b": rng.randn(n),
            "feat_c": rng.randn(n),
        }
    )
    y = (X["feat_a"] + X["feat_b"] > 0).astype(int)
    return X, y


@pytest.fixture
def tiny_ats_data():
    """Generate minimal training data for an ATS regressor."""
    rng = np.random.RandomState(42)
    n = 30
    X = pd.DataFrame(
        {
            "ats_f1": rng.randn(n),
            "ats_f2": rng.randn(n),
        }
    )
    y = X["ats_f1"] * 3 + X["ats_f2"] * 2 + rng.randn(n) * 0.5
    return X, y


@pytest.fixture
def tiny_ou_data():
    """Generate minimal training data for an O/U regressor."""
    rng = np.random.RandomState(42)
    n = 30
    X = pd.DataFrame(
        {
            "ou_f1": rng.randn(n),
            "ou_f2": rng.randn(n),
        }
    )
    y = 45 + X["ou_f1"] * 5 + X["ou_f2"] * 3 + rng.randn(n)
    return X, y


@pytest.fixture
def saved_artifacts(tmp_path, tiny_wp_data, tiny_ats_data, tiny_ou_data):
    """Save minimal model artifacts to a tmp directory, return the dir."""
    # Win-probability artifact uses a logistic-regression classifier.
    wp_X, wp_y = tiny_wp_data
    wp_model = LogisticRegression(max_iter=200)
    wp_model.fit(wp_X, wp_y)
    save_model_artifact(
        model=wp_model,
        target="wp",
        metadata={"version": "test-wp-v1", "residual_std": 13.5},
        feature_list=list(wp_X.columns),
        artifacts_dir=tmp_path,
        update_latest=True,
    )

    # Against-the-spread artifact uses a gradient-boosted regressor.
    ats_X, ats_y = tiny_ats_data
    ats_model = XGBRegressor(n_estimators=10, max_depth=2, random_state=42)
    ats_model.fit(ats_X, ats_y)
    save_model_artifact(
        model=ats_model,
        target="ats",
        metadata={"version": "test-ats-v1", "residual_std": 10.0},
        feature_list=list(ats_X.columns),
        artifacts_dir=tmp_path,
        update_latest=True,
    )

    # Over/under artifact uses a gradient-boosted regressor.
    ou_X, ou_y = tiny_ou_data
    ou_model = XGBRegressor(n_estimators=10, max_depth=2, random_state=42)
    ou_model.fit(ou_X, ou_y)
    save_model_artifact(
        model=ou_model,
        target="ou",
        metadata={"version": "test-ou-v1", "residual_std": 12.0},
        feature_list=list(ou_X.columns),
        artifacts_dir=tmp_path,
        update_latest=True,
    )

    return tmp_path


# -- Test classes --


class TestPipelineInit:
    """Test NFLPredictionPipeline initialization."""

    def test_init_default_none_artifacts(self):
        """Pipeline initializes with None artifacts by default."""
        pipeline = NFLPredictionPipeline()
        assert pipeline.wp_artifact is None
        assert pipeline.ats_artifact is None
        assert pipeline.ou_artifact is None

    def test_init_with_artifact_dicts(self):
        """Pipeline accepts artifact dicts at construction."""
        wp = {"model": "fake_wp", "feature_list": ["a"]}
        ats = {"model": "fake_ats", "feature_list": ["b"]}
        ou = {"model": "fake_ou", "feature_list": ["c"]}
        pipeline = NFLPredictionPipeline(
            wp_artifact=wp, ats_artifact=ats, ou_artifact=ou
        )
        assert pipeline.wp_artifact is wp
        assert pipeline.ats_artifact is ats
        assert pipeline.ou_artifact is ou

    def test_init_stores_config(self):
        """Pipeline stores edge threshold, confidence threshold, kelly fraction."""
        pipeline = NFLPredictionPipeline(
            min_edge_threshold=0.05,
            min_confidence_threshold=0.7,
            max_kelly_fraction=0.10,
        )
        assert pipeline.min_edge_threshold == 0.05
        assert pipeline.min_confidence_threshold == 0.7
        assert pipeline.max_kelly_fraction == 0.10

    def test_init_creates_utility_classes(self):
        """Pipeline creates OddsConverter, EdgeCalculator, and BetRecommendationEngine."""
        pipeline = NFLPredictionPipeline()
        assert pipeline.odds_converter is not None
        assert pipeline.edge_calculator is not None
        assert pipeline.recommendation_engine is not None


class TestLoadModels:
    """Test model loading from artifact storage."""

    def test_load_models_populates_all_artifacts(self, saved_artifacts):
        """load_models loads all three artifact dicts from disk."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=saved_artifacts)

        assert pipeline.wp_artifact is not None
        assert pipeline.ats_artifact is not None
        assert pipeline.ou_artifact is not None

    def test_loaded_artifacts_have_expected_keys(self, saved_artifacts):
        """Each loaded artifact dict has model, metadata, feature_list keys."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=saved_artifacts)

        for artifact in [
            pipeline.wp_artifact,
            pipeline.ats_artifact,
            pipeline.ou_artifact,
        ]:
            assert "model" in artifact
            assert "metadata" in artifact
            assert "feature_list" in artifact

    def test_loaded_wp_model_has_predict_proba(self, saved_artifacts):
        """The loaded WP model is a real sklearn classifier with predict_proba."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=saved_artifacts)
        assert hasattr(pipeline.wp_artifact["model"], "predict_proba")

    def test_loaded_ats_model_has_predict(self, saved_artifacts):
        """The loaded ATS model is a real xgboost regressor with predict."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=saved_artifacts)
        assert hasattr(pipeline.ats_artifact["model"], "predict")

    def test_loaded_ou_model_has_predict(self, saved_artifacts):
        """The loaded O/U model is a real xgboost regressor with predict."""
        pipeline = NFLPredictionPipeline()
        pipeline.load_models(artifacts_dir=saved_artifacts)
        assert hasattr(pipeline.ou_artifact["model"], "predict")


class TestCheckModelsLoaded:
    """Test _check_models_loaded validation."""

    def test_raises_when_all_none(self):
        """Raises ValueError listing all missing models."""
        pipeline = NFLPredictionPipeline()
        with pytest.raises(ValueError, match="WP model"):
            pipeline._check_models_loaded()

    def test_raises_when_wp_missing(self):
        """Raises ValueError when only WP artifact is None."""
        pipeline = NFLPredictionPipeline(
            ats_artifact={"model": "x"},
            ou_artifact={"model": "x"},
        )
        with pytest.raises(ValueError, match="WP model"):
            pipeline._check_models_loaded()

    def test_raises_when_ats_missing(self):
        """Raises ValueError when only ATS artifact is None."""
        pipeline = NFLPredictionPipeline(
            wp_artifact={"model": "x"},
            ou_artifact={"model": "x"},
        )
        with pytest.raises(ValueError, match="ATS model"):
            pipeline._check_models_loaded()

    def test_raises_when_ou_missing(self):
        """Raises ValueError when only O/U artifact is None."""
        pipeline = NFLPredictionPipeline(
            wp_artifact={"model": "x"},
            ats_artifact={"model": "x"},
        )
        with pytest.raises(ValueError, match="O/U model"):
            pipeline._check_models_loaded()

    def test_passes_when_all_set(self):
        """No error when all three artifacts are set."""
        pipeline = NFLPredictionPipeline(
            wp_artifact={"model": "x"},
            ats_artifact={"model": "x"},
            ou_artifact={"model": "x"},
        )
        # Should not raise
        pipeline._check_models_loaded()


class TestOddsConverter:
    """Test OddsConverter conversion functions."""

    def test_american_to_probability_negative(self):
        """American odds -150 should yield approximately 0.6 probability."""
        prob = OddsConverter.american_to_probability(-150)
        assert abs(prob - 0.6) < 0.001

    def test_american_to_probability_positive(self):
        """American odds +200 should yield approximately 0.333 probability."""
        prob = OddsConverter.american_to_probability(200)
        assert abs(prob - 1 / 3) < 0.001

    def test_probability_to_american_favorite(self):
        """Probability 0.6 should yield approximately -150."""
        odds = OddsConverter.probability_to_american(0.6)
        assert abs(odds - (-150)) <= 1

    def test_probability_to_american_underdog(self):
        """Probability 0.4 should yield approximately +150."""
        odds = OddsConverter.probability_to_american(0.4)
        assert abs(odds - 150) <= 1

    def test_round_trip_american_probability(self):
        """Round-trip: american -> probability -> american preserves value.

        Note: +100 (even money) is excluded because probability_to_american(0.5)
        returns -100, which is mathematically equivalent.
        """
        for odds in [-200, -150, -110, 150, 250]:
            prob = OddsConverter.american_to_probability(odds)
            recovered = OddsConverter.probability_to_american(prob)
            assert abs(recovered - odds) <= 2, (
                f"Round trip failed for {odds}: got {recovered}"
            )

    def test_american_to_decimal_favorite(self):
        """American -200 should be decimal 1.5."""
        dec = OddsConverter.american_to_decimal(-200)
        assert abs(dec - 1.5) < 0.001

    def test_american_to_decimal_underdog(self):
        """American +200 should be decimal 3.0."""
        dec = OddsConverter.american_to_decimal(200)
        assert abs(dec - 3.0) < 0.001

    def test_decimal_to_american_roundtrip(self):
        """Round-trip: american -> decimal -> american preserves value."""
        for odds in [-200, -150, -110, 200, 300]:
            dec = OddsConverter.american_to_decimal(odds)
            recovered = OddsConverter.decimal_to_american(dec)
            assert abs(recovered - odds) <= 1, (
                f"Round trip failed for {odds}: got {recovered}"
            )


class TestPredictionDataclasses:
    """Test WPPrediction, ATSPrediction, OUPrediction instantiation."""

    def test_wp_prediction_fields(self):
        """WPPrediction stores all expected fields."""
        pred = WPPrediction(
            game_id="2024_01_KC_BAL",
            home_team="BAL",
            away_team="KC",
            raw_win_probability=0.55,
            calibrated_win_probability=0.57,
            prediction_confidence=0.14,
            feature_importances={"feat_a": 0.3},
        )
        assert pred.game_id == "2024_01_KC_BAL"
        assert pred.raw_win_probability == 0.55
        assert pred.calibrated_win_probability == 0.57
        assert pred.feature_importances == {"feat_a": 0.3}

    def test_wp_prediction_defaults(self):
        """WPPrediction optional fields default correctly."""
        pred = WPPrediction(
            game_id="g1",
            home_team="A",
            away_team="B",
            raw_win_probability=0.5,
        )
        assert pred.calibrated_win_probability is None
        assert pred.prediction_confidence is None
        assert pred.feature_importances == {}

    def test_ats_prediction_fields(self):
        """ATSPrediction stores all expected fields."""
        pred = ATSPrediction(
            predicted_margin=3.5,
            predicted_spread=-3.0,
            cover_probability=0.55,
            confidence=0.10,
            feature_importances={"ats_f1": 0.5},
        )
        assert pred.predicted_margin == 3.5
        assert pred.cover_probability == 0.55
        assert pred.feature_importances == {"ats_f1": 0.5}

    def test_ou_prediction_fields(self):
        """OUPrediction stores all expected fields."""
        pred = OUPrediction(
            predicted_total=44.5,
            over_probability=0.52,
            under_probability=0.48,
            confidence=0.04,
        )
        assert pred.predicted_total == 44.5
        assert pred.over_probability == 0.52
        assert pred.under_probability == 0.48
        assert pred.feature_importances == {}
