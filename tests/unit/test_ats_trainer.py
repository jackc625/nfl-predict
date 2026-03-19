"""Tests for ATSTrainer -- ATS (Against the Spread) model trainer.

Validates MODL-03: XGBoost regression for margin prediction with
cover probability derived from ResidualDistributionConverter.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from models.trainers.ats_trainer import ATSTrainer
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trainer() -> ATSTrainer:
    """Create a default ATSTrainer instance."""
    return ATSTrainer()


@pytest.fixture()
def synthetic_features_df() -> pd.DataFrame:
    """Create synthetic feature DataFrame mimicking NFL game data.

    100 games per season, seasons 2018-2024, with 15 numeric features,
    home_margin target, and snapshot_spread for cover probability testing.
    """
    np.random.seed(42)

    seasons = list(range(2018, 2025))
    games_per_season = 100
    rows = []

    for season in seasons:
        for week_idx in range(games_per_season):
            week = (week_idx % 18) + 1
            row = {
                "game_id": f"{season}_{week:02d}_TEAM{week_idx:03d}",
                "season": season,
                "week": week,
                "home_team": f"HM{week_idx:02d}",
                "away_team": f"AW{week_idx:02d}",
                "home_margin": np.random.normal(2.5, 14.0),
                "snapshot_spread": np.random.normal(-2.5, 6.0),
            }
            for f_idx in range(1, 16):
                row[f"feature_{f_idx}"] = np.random.normal(0, 1)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ats_trainer_creates_xgbregressor(trainer: ATSTrainer) -> None:
    """ATSTrainer._create_model({}) returns an XGBRegressor instance."""
    model = trainer._create_model({})
    assert isinstance(model, XGBRegressor)


def test_ats_target_column(trainer: ATSTrainer) -> None:
    """ATSTrainer._get_target_column() returns 'home_margin'."""
    assert trainer._get_target_column() == "home_margin"


def test_ats_predict_raw_returns_float(
    trainer: ATSTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """_predict_raw returns numeric margin predictions (not probabilities)."""
    np.random.seed(42)
    feature_cols = [f"feature_{i}" for i in range(1, 16)]
    X = synthetic_features_df[feature_cols].iloc[:200]
    y = synthetic_features_df["home_margin"].iloc[:200]

    model = trainer._create_model(trainer._get_default_params())
    model.fit(X, y)

    preds = trainer._predict_raw(model, X.iloc[:10])
    assert isinstance(preds, np.ndarray)
    assert preds.dtype in (np.float32, np.float64)
    # Margin predictions should be in a reasonable range, not 0-1 probabilities
    assert np.any(np.abs(preds) > 1.0), (
        "Predictions look like probabilities, not margins"
    )


def test_ats_residual_converter_fitted(
    trainer: ATSTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """After training, self.residual_converter.is_fitted is True."""
    trainer.train_and_evaluate(synthetic_features_df)

    assert trainer.residual_converter is not None
    assert trainer.residual_converter.is_fitted is True


def test_ats_cover_probability_range(
    trainer: ATSTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """Cover probabilities derived from converter are in [0, 1] range."""
    trainer.train_and_evaluate(synthetic_features_df)

    # Use converter to compute cover probabilities
    predicted_margins = np.array([3.0, -7.0, 0.0, 14.0, -14.0])
    spreads = np.array([-3.0, 7.0, -1.0, -10.0, 3.0])

    cover_probs = trainer.predict_cover_probability(predicted_margins, spreads)
    assert isinstance(cover_probs, np.ndarray)
    assert np.all(cover_probs >= 0.0)
    assert np.all(cover_probs <= 1.0)


def test_ats_no_classification_model(trainer: ATSTrainer) -> None:
    """ATSTrainer does not contain 'XGBClassifier' or 'classification'."""
    source = inspect.getsource(ATSTrainer)
    assert "XGBClassifier" not in source
    assert "classification" not in source.lower()


def test_ats_feature_count_soft_target(
    trainer: ATSTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """With 35 input features, feature selection produces <= 25 features."""
    np.random.seed(42)

    # Add more features to get 35 total
    df = synthetic_features_df.copy()
    for i in range(16, 36):
        df[f"feature_{i}"] = np.random.normal(0, 1, len(df))

    results = trainer.train_and_evaluate(df)
    n_features = len(results["feature_names"])
    assert n_features <= 25, f"Expected <= 25 features, got {n_features}"


def test_ats_train_on_synthetic_data(
    trainer: ATSTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """Train on synthetic data produces margin predictions with reasonable range."""
    results = trainer.train_and_evaluate(synthetic_features_df)

    # Check that season results exist for holdout seasons
    assert len(results["season_results"]) > 0, "No season results produced"

    # Check that the model is stored
    assert trainer.model is not None

    # Check predictions are in reasonable range
    feature_cols = results["feature_names"]
    test_data = synthetic_features_df[synthetic_features_df["season"] == 2024][
        feature_cols
    ].head(10)
    preds = trainer._predict_raw(trainer.model, test_data)
    assert np.all(np.abs(preds) < 50), f"Predictions out of range: {preds}"
