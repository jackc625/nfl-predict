"""Tests for BacktestEngine (BACK-01 temporal isolation, BACK-02 CLV headline, BACK-04 per-season breakdown).

Validates that the engine creates correct temporal splits, fresh trainer instances,
filters 2025 data, and produces well-structured BacktestResults.
"""

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from backtest.engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResults,
    SeasonResult,
    TargetResult,
)
from models.temporal import TemporalSplitConfig


class TestBacktestConfig:
    """Tests for BacktestConfig defaults."""

    def test_default_holdout_seasons(self) -> None:
        config = BacktestConfig()
        assert config.holdout_seasons == [2021, 2022, 2023, 2024]

    def test_default_first_data_season(self) -> None:
        config = BacktestConfig()
        assert config.first_data_season == 2018

    def test_default_targets(self) -> None:
        config = BacktestConfig()
        assert config.targets == ["wp", "ats", "ou"]

    def test_default_max_backtest_season(self) -> None:
        config = BacktestConfig()
        assert config.max_backtest_season == 2024


class TestCreateSplitConfig:
    """Tests for BacktestEngine._create_split_config()."""

    def setup_method(self) -> None:
        self.engine = BacktestEngine()

    def test_create_split_config_2021(self) -> None:
        config = self.engine._create_split_config(2021)
        assert config.train_seasons == [2018, 2019]
        assert config.hp_val_seasons == [2020]
        assert config.holdout_seasons == [2021]

    def test_create_split_config_2022(self) -> None:
        config = self.engine._create_split_config(2022)
        assert config.train_seasons == [2018, 2019, 2020]
        assert config.hp_val_seasons == [2021]
        assert config.holdout_seasons == [2022]

    def test_create_split_config_2023(self) -> None:
        config = self.engine._create_split_config(2023)
        assert config.train_seasons == [2018, 2019, 2020, 2021]
        assert config.hp_val_seasons == [2022]
        assert config.holdout_seasons == [2023]

    def test_create_split_config_2024(self) -> None:
        config = self.engine._create_split_config(2024)
        assert config.train_seasons == [2018, 2019, 2020, 2021, 2022]
        assert config.hp_val_seasons == [2023]
        assert config.holdout_seasons == [2024]

    def test_temporal_isolation(self) -> None:
        """Verify strict temporal ordering for all holdout seasons."""
        for holdout in [2021, 2022, 2023, 2024]:
            config = self.engine._create_split_config(holdout)
            assert max(config.train_seasons) < min(config.hp_val_seasons), (
                f"Train/HP-val overlap for holdout={holdout}"
            )
            assert max(config.hp_val_seasons) < min(config.holdout_seasons), (
                f"HP-val/holdout overlap for holdout={holdout}"
            )


class TestCreateTrainer:
    """Tests for BacktestEngine._create_trainer()."""

    def setup_method(self) -> None:
        self.engine = BacktestEngine()

    def test_fresh_trainer_instances(self) -> None:
        """Verify each call returns a NEW trainer instance (no state leakage)."""
        config = self.engine._create_split_config(2021)
        trainer_a = self.engine._create_trainer("wp", config)
        trainer_b = self.engine._create_trainer("wp", config)
        assert trainer_a is not trainer_b

    def test_creates_wp_trainer(self) -> None:
        from models.trainers.wp_trainer import WPTrainer

        config = self.engine._create_split_config(2021)
        trainer = self.engine._create_trainer("wp", config)
        assert isinstance(trainer, WPTrainer)

    def test_creates_ats_trainer(self) -> None:
        from models.trainers.ats_trainer import ATSTrainer

        config = self.engine._create_split_config(2021)
        trainer = self.engine._create_trainer("ats", config)
        assert isinstance(trainer, ATSTrainer)

    def test_creates_ou_trainer(self) -> None:
        from models.trainers.ou_trainer import OUTrainer

        config = self.engine._create_split_config(2021)
        trainer = self.engine._create_trainer("ou", config)
        assert isinstance(trainer, OUTrainer)

    def test_invalid_target_raises(self) -> None:
        config = self.engine._create_split_config(2021)
        with pytest.raises(ValueError, match="Unknown target"):
            self.engine._create_trainer("invalid", config)


class TestFilters2025Data:
    """Tests for _load_features filtering."""

    def test_filters_2025_data(self) -> None:
        """Create a mock features_df with seasons [2018..2025], verify filtering."""
        engine = BacktestEngine()

        # Build fake features DataFrame with seasons 2018-2025
        rows = []
        for season in range(2018, 2026):
            for i in range(5):
                rows.append(
                    {
                        "game_id": f"{season}_W{i + 1:02d}_TEST",
                        "season": season,
                        "week": i + 1,
                        "home_team": "KC",
                        "away_team": "BUF",
                        "home_win": 1,
                        "feature_a": np.random.random(),
                    }
                )
        mock_df = pd.DataFrame(rows)

        with patch("backtest.engine.pd.read_parquet", return_value=mock_df):
            result = engine._load_features("wp")
            assert result["season"].max() <= 2024
            assert 2025 not in result["season"].values


class TestBacktestResultsStructure:
    """Tests for BacktestResults structure validation."""

    def test_backtest_results_has_headline_clv(self) -> None:
        """BacktestResults should contain headline_clv dict."""
        results = BacktestResults(
            config=BacktestConfig(),
            season_results=[],
            all_predictions={},
            all_clv={},
            headline_clv={"wp": 0.02, "ats": 0.01, "ou": -0.005},
            odds_coverage={"with_odds": 100, "without_odds": 5},
            covid_annotation={"seasons": [2020]},
            era_info={2021: 18, 2020: 17},
        )
        assert "wp" in results.headline_clv
        assert "ats" in results.headline_clv
        assert "ou" in results.headline_clv

    def test_backtest_results_has_covid_annotation(self) -> None:
        results = BacktestResults(
            config=BacktestConfig(),
            season_results=[],
            all_predictions={},
            all_clv={},
            headline_clv={},
            odds_coverage={},
            covid_annotation={"seasons": [2020], "home_win_pct": 0.496},
            era_info={},
        )
        assert results.covid_annotation["seasons"] == [2020]

    def test_target_result_structure(self) -> None:
        tr = TargetResult(
            target="wp",
            season=2021,
            predictions_df=pd.DataFrame({"game_id": ["g1"], "model_prob": [0.6]}),
            clv_df=None,
            metrics={"accuracy": 0.65},
            feature_names=["elo_diff"],
            best_params={"C": 1.0},
        )
        assert tr.target == "wp"
        assert tr.season == 2021

    def test_season_result_structure(self) -> None:
        sr = SeasonResult(
            season=2021,
            target_results={},
            split_config=TemporalSplitConfig(
                train_seasons=[2018, 2019],
                hp_val_seasons=[2020],
                holdout_seasons=[2021],
            ),
        )
        assert sr.season == 2021
