"""Tests for the v1.0 baseline capture script.

Validates that the BaselineCapture class produces all expected artifacts:
metrics JSON per target, predictions Parquet per target, feature metadata
JSON, and a comparison template markdown file.

All tests mock BacktestEngine to avoid running the actual backtest
(which takes minutes). The mock fixtures produce realistic structure
matching BacktestResults.
"""

from __future__ import annotations

import json
from dataclasses import field
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from backtest.engine import (
    BacktestConfig,
    BacktestResults,
    SeasonResult,
    TargetResult,
)
from models.temporal import TemporalSplitConfig


# -----------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------


def _make_predictions_df(target: str, n_games: int = 50) -> pd.DataFrame:
    """Create a realistic predictions DataFrame for a given target."""
    import numpy as np

    rng = np.random.default_rng(42)
    game_ids = [f"2023_0{i // 16 + 1}_{chr(65 + i % 26)}@{chr(66 + i % 26)}" for i in range(n_games)]
    seasons = [2023] * (n_games // 2) + [2024] * (n_games - n_games // 2)

    data = {
        "game_id": game_ids,
        "season": seasons,
        "week": list(range(1, n_games + 1)),
        "home_team": ["KC"] * n_games,
        "away_team": ["BUF"] * n_games,
    }

    if target == "wp":
        data["predicted_prob"] = rng.uniform(0.3, 0.8, n_games).tolist()
        data["actual_outcome"] = rng.integers(0, 2, n_games).tolist()
    else:
        data["predicted_prob"] = rng.normal(3.0, 7.0, n_games).tolist()
        data["actual_outcome"] = rng.normal(2.5, 8.0, n_games).tolist()

    # CLV-related columns that BacktestEngine produces
    data["probability_clv"] = rng.uniform(-0.05, 0.05, n_games).tolist()
    data["has_closing_odds"] = [True] * (n_games - 5) + [False] * 5

    return pd.DataFrame(data)


def _make_clv_df(target: str, n_games: int = 50) -> pd.DataFrame:
    """Create a realistic CLV DataFrame."""
    import numpy as np

    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "game_id": [f"game_{i}" for i in range(n_games)],
        "season": [2023] * (n_games // 2) + [2024] * (n_games - n_games // 2),
        "probability_clv": rng.uniform(-0.05, 0.05, n_games).tolist(),
        "has_closing_odds": [True] * (n_games - 5) + [False] * 5,
    })


@pytest.fixture
def mock_backtest_results() -> BacktestResults:
    """Create a BacktestResults with realistic structure for testing."""
    targets = ["wp", "ats", "ou"]

    all_predictions = {t: _make_predictions_df(t) for t in targets}
    all_clv = {t: _make_clv_df(t) for t in targets}

    headline_clv = {"wp": 0.02, "ats": -0.01, "ou": 0.005}
    odds_coverage = {
        "wp_with_odds": 900, "wp_without_odds": 100,
        "ats_with_odds": 850, "ats_without_odds": 150,
        "ou_with_odds": 880, "ou_without_odds": 120,
    }

    split_config = TemporalSplitConfig(
        train_seasons=[2018, 2019],
        hp_val_seasons=[2020],
        holdout_seasons=[2021],
    )

    season_results = []
    for season in [2021, 2022, 2023, 2024]:
        target_results = {}
        for t in targets:
            target_results[t] = TargetResult(
                target=t,
                season=season,
                predictions_df=_make_predictions_df(t, n_games=20),
                clv_df=_make_clv_df(t, n_games=20),
                metrics={"accuracy": 0.55, "brier_score": 0.23, "n_games": 20},
                feature_names=["feat_a", "feat_b"],
                best_params={"max_depth": 6},
            )
        season_results.append(SeasonResult(
            season=season,
            target_results=target_results,
            split_config=split_config,
        ))

    return BacktestResults(
        config=BacktestConfig(),
        season_results=season_results,
        all_predictions=all_predictions,
        all_clv=all_clv,
        headline_clv=headline_clv,
        odds_coverage=odds_coverage,
        covid_annotation={"2020": "shortened"},
        era_info={2021: 18, 2022: 18, 2023: 18, 2024: 18},
        is_blended=False,
    )


@pytest.fixture
def mock_gold_features() -> dict[str, pd.DataFrame]:
    """Create mock gold feature matrices for feature metadata capture."""
    import numpy as np

    rng = np.random.default_rng(99)
    targets = ["wp", "ats", "ou"]
    result = {}
    for t in targets:
        n = 200
        result[t] = pd.DataFrame({
            "game_id": [f"g_{i}" for i in range(n)],
            "season": [2023] * n,
            "home_elo": rng.normal(1550, 80, n),
            "away_elo": rng.normal(1500, 75, n),
            "elo_diff": rng.normal(50, 60, n),
        })
    return result


@pytest.fixture
def output_dir(tmp_path: Path) -> Path:
    """Provide a temporary output directory for baseline files."""
    return tmp_path / "baselines" / "v1.0"


# -----------------------------------------------------------------------
# Tests
# -----------------------------------------------------------------------


class TestBaselineCapture:
    """Tests for the BaselineCapture class."""

    def test_capture_creates_expected_files(
        self,
        mock_backtest_results: BacktestResults,
        mock_gold_features: dict[str, pd.DataFrame],
        output_dir: Path,
    ):
        """Capture with a mock BacktestEngine produces all expected files."""
        from scripts.capture_baseline import BaselineCapture

        with (
            patch("scripts.capture_baseline.BacktestEngine") as mock_engine_cls,
            patch("scripts.capture_baseline.pd.read_parquet") as mock_read_pq,
        ):
            mock_engine_cls.return_value.run.return_value = mock_backtest_results
            mock_read_pq.side_effect = lambda path: mock_gold_features.get(
                Path(path).stem.replace("features_", ""),
                pd.DataFrame(),
            )

            capture = BaselineCapture(output_dir=output_dir)
            result_path = capture.run()

        expected_files = [
            "metrics_wp.json",
            "metrics_ats.json",
            "metrics_ou.json",
            "predictions_wp.parquet",
            "predictions_ats.parquet",
            "predictions_ou.parquet",
            "feature_metadata.json",
            "comparison_template.md",
        ]
        for fname in expected_files:
            assert (output_dir / fname).exists(), f"Missing: {fname}"

    def test_metrics_json_structure(
        self,
        mock_backtest_results: BacktestResults,
        mock_gold_features: dict[str, pd.DataFrame],
        output_dir: Path,
    ):
        """Each metrics JSON contains headline_clv, per_season_results, odds_coverage."""
        from scripts.capture_baseline import BaselineCapture

        with (
            patch("scripts.capture_baseline.BacktestEngine") as mock_engine_cls,
            patch("scripts.capture_baseline.pd.read_parquet") as mock_read_pq,
        ):
            mock_engine_cls.return_value.run.return_value = mock_backtest_results
            mock_read_pq.side_effect = lambda path: mock_gold_features.get(
                Path(path).stem.replace("features_", ""),
                pd.DataFrame(),
            )

            capture = BaselineCapture(output_dir=output_dir)
            capture.run()

        for target in ["wp", "ats", "ou"]:
            metrics_path = output_dir / f"metrics_{target}.json"
            data = json.loads(metrics_path.read_text())
            assert "headline_clv" in data, f"metrics_{target}.json missing headline_clv"
            assert isinstance(data["headline_clv"], float), "headline_clv must be float"
            assert "per_season_results" in data, f"metrics_{target}.json missing per_season_results"
            assert isinstance(data["per_season_results"], list)
            assert "odds_coverage" in data, f"metrics_{target}.json missing odds_coverage"
            assert isinstance(data["odds_coverage"], dict)

    def test_predictions_parquet_has_required_columns(
        self,
        mock_backtest_results: BacktestResults,
        mock_gold_features: dict[str, pd.DataFrame],
        output_dir: Path,
    ):
        """Predictions parquet has game_id, season, week, predicted_prob, actual_outcome columns."""
        from scripts.capture_baseline import BaselineCapture

        with (
            patch("scripts.capture_baseline.BacktestEngine") as mock_engine_cls,
            patch("scripts.capture_baseline.pd.read_parquet") as mock_read_pq,
        ):
            mock_engine_cls.return_value.run.return_value = mock_backtest_results
            mock_read_pq.side_effect = lambda path: mock_gold_features.get(
                Path(path).stem.replace("features_", ""),
                pd.DataFrame(),
            )

            capture = BaselineCapture(output_dir=output_dir)
            capture.run()

        for target in ["wp", "ats", "ou"]:
            pq_path = output_dir / f"predictions_{target}.parquet"
            df = pd.read_parquet(pq_path)
            assert len(df) > 0, f"predictions_{target}.parquet is empty"
            required_cols = ["game_id", "season", "week", "predicted_prob", "actual_outcome"]
            for col in required_cols:
                assert col in df.columns, f"predictions_{target}.parquet missing column: {col}"

    def test_feature_metadata_structure(
        self,
        mock_backtest_results: BacktestResults,
        mock_gold_features: dict[str, pd.DataFrame],
        output_dir: Path,
    ):
        """feature_metadata.json has per-target entries with columns, row_count, column_stats."""
        from scripts.capture_baseline import BaselineCapture

        with (
            patch("scripts.capture_baseline.BacktestEngine") as mock_engine_cls,
            patch("scripts.capture_baseline.pd.read_parquet") as mock_read_pq,
        ):
            mock_engine_cls.return_value.run.return_value = mock_backtest_results
            mock_read_pq.side_effect = lambda path: mock_gold_features.get(
                Path(path).stem.replace("features_", ""),
                pd.DataFrame(),
            )

            capture = BaselineCapture(output_dir=output_dir)
            capture.run()

        meta_path = output_dir / "feature_metadata.json"
        data = json.loads(meta_path.read_text())

        for target in ["wp", "ats", "ou"]:
            assert target in data, f"feature_metadata.json missing target: {target}"
            entry = data[target]
            assert "columns" in entry, f"{target} missing 'columns'"
            assert isinstance(entry["columns"], list)
            assert "row_count" in entry, f"{target} missing 'row_count'"
            assert isinstance(entry["row_count"], int)
            assert "column_stats" in entry, f"{target} missing 'column_stats'"
            assert isinstance(entry["column_stats"], dict)
            # Check that column_stats has mean/std/nulls for numeric columns
            for col_name, stats in entry["column_stats"].items():
                assert "mean" in stats, f"{target}/{col_name} missing 'mean'"
                assert "std" in stats, f"{target}/{col_name} missing 'std'"
                assert "nulls" in stats, f"{target}/{col_name} missing 'nulls'"

    def test_comparison_template(
        self,
        mock_backtest_results: BacktestResults,
        mock_gold_features: dict[str, pd.DataFrame],
        output_dir: Path,
    ):
        """comparison_template.md contains v1.0 column, v2.0 placeholder, and headline CLV values."""
        from scripts.capture_baseline import BaselineCapture

        with (
            patch("scripts.capture_baseline.BacktestEngine") as mock_engine_cls,
            patch("scripts.capture_baseline.pd.read_parquet") as mock_read_pq,
        ):
            mock_engine_cls.return_value.run.return_value = mock_backtest_results
            mock_read_pq.side_effect = lambda path: mock_gold_features.get(
                Path(path).stem.replace("features_", ""),
                pd.DataFrame(),
            )

            capture = BaselineCapture(output_dir=output_dir)
            capture.run()

        template_path = output_dir / "comparison_template.md"
        content = template_path.read_text()

        assert "v1.0" in content, "Template missing 'v1.0'"
        assert "v2.0" in content, "Template missing 'v2.0'"
        assert "---" in content, "Template missing '---' placeholder"
        # Check that headline CLV values from mock are present
        assert "0.02" in content, "Template missing WP headline CLV value"

    def test_output_directory_creation(
        self,
        mock_backtest_results: BacktestResults,
        mock_gold_features: dict[str, pd.DataFrame],
        tmp_path: Path,
    ):
        """Script creates output directory if it does not exist."""
        from scripts.capture_baseline import BaselineCapture

        deeply_nested = tmp_path / "a" / "b" / "c" / "baselines" / "v1.0"
        assert not deeply_nested.exists()

        with (
            patch("scripts.capture_baseline.BacktestEngine") as mock_engine_cls,
            patch("scripts.capture_baseline.pd.read_parquet") as mock_read_pq,
        ):
            mock_engine_cls.return_value.run.return_value = mock_backtest_results
            mock_read_pq.side_effect = lambda path: mock_gold_features.get(
                Path(path).stem.replace("features_", ""),
                pd.DataFrame(),
            )

            capture = BaselineCapture(output_dir=deeply_nested)
            capture.run()

        assert deeply_nested.exists(), "Output directory was not created"
        assert (deeply_nested / "metrics_wp.json").exists(), "Files not written to created directory"
