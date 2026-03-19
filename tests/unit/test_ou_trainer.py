"""Tests for OUTrainer -- O/U (Over/Under) model trainer.

Validates MODL-04: XGBoost regression for total points prediction with
over/under probabilities derived from TotalDistributionConverter.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest
from models.trainers.ou_trainer import OUTrainer
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def trainer() -> OUTrainer:
    """Create a default OUTrainer instance."""
    return OUTrainer()


@pytest.fixture()
def synthetic_features_df() -> pd.DataFrame:
    """Create synthetic feature DataFrame mimicking NFL game data.

    100 games per season, seasons 2018-2024, with 15 numeric features,
    total_points target, and snapshot_total for over/under probability testing.
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
                "total_points": np.random.normal(44.0, 10.0),
                "snapshot_total": np.random.normal(44.0, 4.0),
            }
            for f_idx in range(1, 16):
                row[f"feature_{f_idx}"] = np.random.normal(0, 1)
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_ou_trainer_creates_xgbregressor(trainer: OUTrainer) -> None:
    """OUTrainer._create_model({}) returns an XGBRegressor instance."""
    model = trainer._create_model({})
    assert isinstance(model, XGBRegressor)


def test_ou_target_column(trainer: OUTrainer) -> None:
    """OUTrainer._get_target_column() returns 'total_points'."""
    assert trainer._get_target_column() == "total_points"


def test_ou_predict_raw_returns_float(
    trainer: OUTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """_predict_raw returns numeric total predictions (not probabilities)."""
    np.random.seed(42)
    feature_cols = [f"feature_{i}" for i in range(1, 16)]
    X = synthetic_features_df[feature_cols].iloc[:200]
    y = synthetic_features_df["total_points"].iloc[:200]

    model = trainer._create_model(trainer._get_default_params())
    model.fit(X, y)

    preds = trainer._predict_raw(model, X.iloc[:10])
    assert isinstance(preds, np.ndarray)
    assert preds.dtype in (np.float32, np.float64)
    # Total predictions should be in a reasonable range, not 0-1 probabilities
    assert np.all(preds > 1.0), "Predictions look like probabilities, not totals"


def test_ou_total_converter_fitted(
    trainer: OUTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """After training, self.total_converter.is_fitted is True."""
    trainer.train_and_evaluate(synthetic_features_df)

    assert trainer.total_converter is not None
    assert trainer.total_converter.is_fitted is True


def test_ou_over_probability_range(
    trainer: OUTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """Over probabilities from converter are in [0, 1] range."""
    trainer.train_and_evaluate(synthetic_features_df)

    # Use converter to compute over probabilities
    predicted_totals = np.array([48.0, 38.0, 44.0, 55.0, 30.0])
    market_totals = np.array([44.5, 44.5, 44.5, 44.5, 44.5])

    over_probs, under_probs = trainer.predict_over_under_probability(
        predicted_totals, market_totals
    )
    assert isinstance(over_probs, np.ndarray)
    assert np.all(over_probs >= 0.0)
    assert np.all(over_probs <= 1.0)


def test_ou_under_probability_complement(
    trainer: OUTrainer, synthetic_features_df: pd.DataFrame
) -> None:
    """under_probability = 1 - over_probability for each prediction."""
    trainer.train_and_evaluate(synthetic_features_df)

    predicted_totals = np.array([48.0, 38.0, 44.0, 55.0, 30.0])
    market_totals = np.array([44.5, 44.5, 44.5, 44.5, 44.5])

    over_probs, under_probs = trainer.predict_over_under_probability(
        predicted_totals, market_totals
    )
    np.testing.assert_allclose(over_probs + under_probs, 1.0, atol=1e-6)


def test_ou_no_weather_submodel(trainer: OUTrainer) -> None:
    """OUTrainer does not contain 'WeatherImpactModel' or 'weather_submodel'."""
    source = inspect.getsource(OUTrainer)
    assert "WeatherImpactModel" not in source
    assert "weather_submodel" not in source


def test_ou_no_poisson_submodel(trainer: OUTrainer) -> None:
    """OUTrainer does not contain 'PoissonScoreModel' or 'poisson'."""
    source = inspect.getsource(OUTrainer)
    assert "PoissonScoreModel" not in source
    assert "poisson" not in source.lower()


def test_ou_feature_count_soft_target(
    trainer: OUTrainer, synthetic_features_df: pd.DataFrame
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
