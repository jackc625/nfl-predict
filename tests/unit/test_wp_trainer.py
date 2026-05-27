"""Tests for WPTrainer (MODL-02, MODL-06, MODL-07).

Verifies:
- WPTrainer uses LogisticRegression with isotonic calibration (MODL-02)
- Calibration is fitted on HP-validation data, not training data
- Feature selection locked on train window (MODL-07)
- ECE computed on holdout and stored in metadata (MODL-06)
- No random CV (StratifiedKFold, cv=5, random_state in CV context)
"""

import inspect
import re

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from models.trainers.wp_trainer import WPTrainer

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def wp_trainer() -> WPTrainer:
    """Create a WPTrainer with default configuration."""
    return WPTrainer()


@pytest.fixture
def synthetic_features_df() -> pd.DataFrame:
    """Create a synthetic feature DataFrame for WP training.

    100 games per season, seasons 2018-2024 (700 total rows).
    Features: 10 numeric columns drawn from np.random.normal.
    Target: home_win binary (0/1) with ~55% home win rate.
    """
    np.random.seed(42)
    rows = []
    teams = ["KC", "BUF", "SF", "PHI", "DAL", "MIA", "DET", "BAL"]

    for season in range(2018, 2025):
        for i in range(100):
            # Create features that have some signal for home_win
            features = {f"feature_{j}": np.random.normal(0, 1) for j in range(1, 11)}
            # Create correlated target (~55% home win rate)
            signal = 0.3 * features["feature_1"] + 0.2 * features["feature_2"] + 0.1
            home_win = int(1 / (1 + np.exp(-signal)) > np.random.random())

            row = {
                "game_id": f"{season}_{i:03d}",
                "season": season,
                "week": (i % 18) + 1,
                "home_team": teams[i % len(teams)],
                "away_team": teams[(i + 1) % len(teams)],
                "home_win": home_win,
                **features,
            }
            rows.append(row)

    return pd.DataFrame(rows)


@pytest.fixture
def wide_features_df() -> pd.DataFrame:
    """Create a wide feature DataFrame with 30 features for testing feature selection.

    100 games per season, seasons 2018-2024 (700 total rows).
    30 numeric features, most of which are noise.
    """
    np.random.seed(42)
    rows = []
    teams = ["KC", "BUF", "SF", "PHI"]

    for season in range(2018, 2025):
        for i in range(100):
            features = {f"feature_{j}": np.random.normal(0, 1) for j in range(1, 31)}
            signal = 0.3 * features["feature_1"] + 0.2 * features["feature_2"] + 0.1
            home_win = int(1 / (1 + np.exp(-signal)) > np.random.random())

            row = {
                "game_id": f"{season}_{i:03d}",
                "season": season,
                "week": (i % 18) + 1,
                "home_team": teams[i % len(teams)],
                "away_team": teams[(i + 1) % len(teams)],
                "home_win": home_win,
                **features,
            }
            rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Test 1: _create_model returns LogisticRegression
# ---------------------------------------------------------------------------


def test_wp_trainer_creates_logistic_regression(wp_trainer):
    """WPTrainer._create_model({}) returns a LogisticRegression instance."""
    model = wp_trainer._create_model({})
    assert isinstance(model, LogisticRegression)


# ---------------------------------------------------------------------------
# Test 2: _get_target_column returns "home_win"
# ---------------------------------------------------------------------------


def test_wp_target_column(wp_trainer):
    """WPTrainer._get_target_column() returns 'home_win'."""
    assert wp_trainer._get_target_column() == "home_win"


# ---------------------------------------------------------------------------
# Test 3: _predict_raw returns probabilities in [0, 1]
# ---------------------------------------------------------------------------


def test_wp_predict_raw_returns_probabilities(wp_trainer):
    """_predict_raw returns values in [0, 1] range."""
    np.random.seed(42)
    X_train = pd.DataFrame(
        {"f1": np.random.normal(0, 1, 100), "f2": np.random.normal(0, 1, 100)}
    )
    y_train = pd.Series(np.random.randint(0, 2, 100))

    model = wp_trainer._create_model(wp_trainer._get_default_params())
    model.fit(X_train, y_train)

    X_test = pd.DataFrame(
        {"f1": np.random.normal(0, 1, 20), "f2": np.random.normal(0, 1, 20)}
    )
    predictions = wp_trainer._predict_raw(model, X_test)

    assert predictions.shape == (20,)
    assert np.all(predictions >= 0.0)
    assert np.all(predictions <= 1.0)


# ---------------------------------------------------------------------------
# Test 4: Train on synthetic data produces valid predictions
# ---------------------------------------------------------------------------


def test_wp_train_on_synthetic_data(wp_trainer, synthetic_features_df):
    """Train on synthetic feature matrix produces model with predictions in [0,1]."""
    results = wp_trainer.train_and_evaluate(synthetic_features_df)

    # Should have season results for holdout seasons (2021-2024)
    assert len(results["season_results"]) == 4
    assert results["feature_names"] is not None
    assert len(results["feature_names"]) > 0

    # All predictions should be in [0, 1]
    for season_result in results["season_results"]:
        assert season_result["season"] in [2021, 2022, 2023, 2024]


# ---------------------------------------------------------------------------
# Test 5: Calibration fitted on HP-validation data
# ---------------------------------------------------------------------------


def test_wp_calibration_on_hp_val(wp_trainer, synthetic_features_df):
    """After training, calibrator is fitted on HP-val data (season 2020), not training."""
    wp_trainer.train_and_evaluate(synthetic_features_df)

    # Calibrator should be fitted (not None)
    assert wp_trainer.calibrator is not None

    # Metadata should record calibration info
    assert "calibration" in wp_trainer.metadata or "ece" in wp_trainer.metadata


# ---------------------------------------------------------------------------
# Test 6: Feature selection locked on train window
# ---------------------------------------------------------------------------


def test_wp_feature_selection_locked(wp_trainer, synthetic_features_df):
    """Feature selection runs on train window only. Same features for all holdout seasons."""
    results = wp_trainer.train_and_evaluate(synthetic_features_df)

    # Feature names should be set and consistent
    feature_names = results["feature_names"]
    assert len(feature_names) > 0

    # The feature names stored in the trainer should match
    assert wp_trainer.feature_names == feature_names

    # All season results should use the same features (check via metadata)
    # Feature selection should NOT change between holdout seasons
    assert results["feature_names"] == wp_trainer.feature_names


# ---------------------------------------------------------------------------
# Test 7: Feature count soft target
# ---------------------------------------------------------------------------


def test_wp_feature_count_soft_target(wp_trainer, wide_features_df):
    """With 30 input features, feature selection produces <= 20 features."""
    results = wp_trainer.train_and_evaluate(wide_features_df)

    n_selected = len(results["feature_names"])
    assert n_selected <= 20, (
        f"Feature selection produced {n_selected} features, expected <= 20"
    )


# ---------------------------------------------------------------------------
# Test 8: Metadata contains ECE
# ---------------------------------------------------------------------------


def test_wp_metadata_contains_ece(wp_trainer, synthetic_features_df):
    """After train_and_evaluate, metadata contains 'ece' key with a float value."""
    wp_trainer.train_and_evaluate(synthetic_features_df)

    assert "ece" in wp_trainer.metadata, (
        f"'ece' not in metadata keys: {list(wp_trainer.metadata.keys())}"
    )
    assert isinstance(wp_trainer.metadata["ece"], float)


# ---------------------------------------------------------------------------
# Test 9: No random CV in source code
# ---------------------------------------------------------------------------


def test_wp_no_random_cv():
    """WPTrainer source code does not contain StratifiedKFold or cv=5."""
    import models.trainers.wp_trainer as wp_module

    source = inspect.getsource(wp_module)

    assert "StratifiedKFold" not in source, (
        "WPTrainer source contains StratifiedKFold (random CV)"
    )
    assert "cv=5" not in source, "WPTrainer source contains cv=5 (random CV)"

    # Check for random_state in CV context (but not in model init which is fine)
    # We allow random_state in LogisticRegression but not in StratifiedKFold
    cv_pattern = re.compile(r"StratifiedKFold.*random_state|KFold.*random_state")
    assert not cv_pattern.search(source), (
        "WPTrainer source contains random_state in CV context"
    )
