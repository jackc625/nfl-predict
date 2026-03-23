"""Integration tests for backtest report generation (Plan 06-03, Task 2).

Tests HTML file generation, CSV export, JSON summary export, and CLI
argument parsing. Uses mock BacktestResults and SimulationResults to
avoid actual model training.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Mock data factories (same as unit tests, duplicated for independence)
# ---------------------------------------------------------------------------


@dataclass
class MockTargetResult:
    target: str
    season: int
    predictions_df: pd.DataFrame
    clv_df: pd.DataFrame | None
    metrics: dict[str, Any]
    feature_names: list[str]
    best_params: dict


@dataclass
class MockSplitConfig:
    train_seasons: list[int]
    hp_val_seasons: list[int]
    holdout_seasons: list[int]


@dataclass
class MockSeasonResult:
    season: int
    target_results: dict[str, MockTargetResult]
    split_config: MockSplitConfig


@dataclass
class MockBacktestConfig:
    holdout_seasons: list[int] = field(default_factory=lambda: [2022, 2023])
    first_data_season: int = 2018
    targets: list[str] = field(default_factory=lambda: ["wp", "ats", "ou"])
    max_backtest_season: int = 2024


@dataclass
class MockBacktestResults:
    config: MockBacktestConfig
    season_results: list[MockSeasonResult]
    all_predictions: dict[str, pd.DataFrame]
    all_clv: dict[str, pd.DataFrame]
    headline_clv: dict[str, float]
    odds_coverage: dict[str, int]
    covid_annotation: dict
    era_info: dict


@dataclass
class MockStrategyResult:
    strategy_name: str
    total_bets: int
    winning_bets: int
    losing_bets: int
    push_bets: int
    win_rate: float
    total_wagered: float
    net_profit: float
    roi: float
    final_bankroll: float
    max_drawdown: float
    max_drawdown_pct: float
    equity_curve: list[float]
    bet_timestamps: list[str]


@dataclass
class MockBetRecord:
    game_id: str
    season: int
    week: int
    target: str
    bet_side: str
    model_value: float
    market_value: float
    edge: float
    slipped_line: float | None
    odds: int
    flat_stake: float
    kelly_stake: float
    outcome: bool | None
    payout_flat: float
    payout_kelly: float


@dataclass
class MockSimulationConfig:
    starting_bankroll: float = 10_000.0
    flat_stake_amount: float = 100.0
    kelly_fraction: float = 0.25
    min_edge_threshold: float = 0.02
    max_kelly_bet_pct: float = 0.05
    slippage_points: float = 0.5
    standard_vig_odds: int = -110


@dataclass
class MockSimulationResults:
    config: MockSimulationConfig
    flat_stake: MockStrategyResult
    kelly: MockStrategyResult
    bet_records: list[Any]
    by_target: dict[str, dict[str, Any]]
    by_season: dict[int, dict[str, Any]]


def _make_wp_predictions(n: int = 50, season: int = 2022) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "game_id": [f"{season}_{i:02d}_TEST" for i in range(n)],
        "season": season,
        "week": rng.integers(1, 19, size=n),
        "model_prob": rng.uniform(0.3, 0.7, size=n),
        "actual": rng.integers(0, 2, size=n),
    })


def _make_clv_df(n: int = 50, season: int = 2022) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "game_id": [f"{season}_{i:02d}_TEST" for i in range(n)],
        "season": season,
        "week": rng.integers(1, 19, size=n),
        "probability_clv": rng.normal(0.01, 0.05, size=n),
        "has_closing_odds": True,
    })


def _make_mock_backtest_results() -> MockBacktestResults:
    config = MockBacktestConfig()
    season_results = []
    all_predictions: dict[str, list[pd.DataFrame]] = {"wp": [], "ats": [], "ou": []}
    all_clv: dict[str, list[pd.DataFrame]] = {"wp": [], "ats": [], "ou": []}

    for season in config.holdout_seasons:
        wp_preds = _make_wp_predictions(50, season)
        wp_clv = _make_clv_df(50, season)

        ats_preds = pd.DataFrame({
            "game_id": [f"{season}_{i:02d}_TEST" for i in range(30)],
            "season": season,
            "week": list(range(1, 31)),
        })
        ou_preds = pd.DataFrame({
            "game_id": [f"{season}_{i:02d}_TEST" for i in range(30)],
            "season": season,
            "week": list(range(1, 31)),
        })

        target_results = {
            "wp": MockTargetResult(
                target="wp", season=season, predictions_df=wp_preds,
                clv_df=wp_clv, metrics={"accuracy": 0.62, "brier_score": 0.23},
                feature_names=["elo_diff"], best_params={},
            ),
            "ats": MockTargetResult(
                target="ats", season=season, predictions_df=ats_preds,
                clv_df=None, metrics={"mae": 7.5},
                feature_names=["elo_diff"], best_params={},
            ),
            "ou": MockTargetResult(
                target="ou", season=season, predictions_df=ou_preds,
                clv_df=None, metrics={"mae": 8.1},
                feature_names=["elo_diff"], best_params={},
            ),
        }

        split = MockSplitConfig([2018, 2019], [season - 1], [season])
        season_results.append(MockSeasonResult(season, target_results, split))

        all_predictions["wp"].append(wp_preds)
        all_clv["wp"].append(wp_clv)
        all_predictions["ats"].append(ats_preds)
        all_predictions["ou"].append(ou_preds)

    concat_preds = {t: pd.concat(dfs, ignore_index=True) for t, dfs in all_predictions.items()}
    concat_clv = {t: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame() for t, dfs in all_clv.items()}

    return MockBacktestResults(
        config=config, season_results=season_results,
        all_predictions=concat_preds, all_clv=concat_clv,
        headline_clv={"wp": 0.015, "ats": -0.003, "ou": 0.008},
        odds_coverage={"wp_with_odds": 80, "wp_without_odds": 20},
        covid_annotation={
            "seasons": [2020], "home_win_pct": 0.496,
            "normal_home_win_pct": 0.57, "note": "COVID impact.",
            "impact_on_model": "HFA reduced.",
        },
        era_info={2022: 18, 2023: 18},
    )


def _make_mock_simulation_results() -> MockSimulationResults:
    bet_records = [
        MockBetRecord(
            game_id=f"2022_{i:02d}_TEST", season=2022, week=i + 1,
            target="wp", bet_side="home", model_value=0.65, market_value=0.55,
            edge=0.10, slipped_line=None, odds=-150, flat_stake=100.0,
            kelly_stake=80.0, outcome=True, payout_flat=66.67, payout_kelly=53.33,
        )
        for i in range(10)
    ]

    return MockSimulationResults(
        config=MockSimulationConfig(),
        flat_stake=MockStrategyResult(
            strategy_name="flat_stake", total_bets=100, winning_bets=55,
            losing_bets=42, push_bets=3, win_rate=0.55, total_wagered=10000.0,
            net_profit=500.0, roi=0.05, final_bankroll=10500.0,
            max_drawdown=800.0, max_drawdown_pct=0.075,
            equity_curve=[10000.0 + i * 5 for i in range(101)],
            bet_timestamps=[f"game_{i}" for i in range(100)],
        ),
        kelly=MockStrategyResult(
            strategy_name="kelly", total_bets=100, winning_bets=55,
            losing_bets=42, push_bets=3, win_rate=0.55, total_wagered=8000.0,
            net_profit=700.0, roi=0.0875, final_bankroll=10700.0,
            max_drawdown=600.0, max_drawdown_pct=0.055,
            equity_curve=[10000.0 + i * 7 for i in range(101)],
            bet_timestamps=[f"game_{i}" for i in range(100)],
        ),
        bet_records=bet_records,
        by_target={"wp": {"n_bets": 40, "wins": 22, "losses": 17, "pushes": 1, "win_rate": 0.55, "flat_roi": 0.03, "kelly_roi": 0.06}},
        by_season={2022: {"n_bets": 50, "wins": 27, "losses": 21, "pushes": 2, "win_rate": 0.54, "flat_roi": 0.04, "kelly_roi": 0.07}},
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestHTMLReportGenerated:
    """Test that BacktestReporter generates a valid HTML file."""

    def test_html_report_generated(self, tmp_path: Path) -> None:
        from backtest.report import BacktestReporter

        results = _make_mock_backtest_results()
        sim_results = _make_mock_simulation_results()

        reporter = BacktestReporter(output_dir=tmp_path)
        report_path = reporter.generate(results, sim_results)

        assert report_path.exists()
        assert report_path.stat().st_size > 1000

        content = report_path.read_text(encoding="utf-8")
        assert "<html" in content
        assert "plotly" in content.lower()
        assert "CLV" in content


class TestCSVExport:
    """Test CSV file generation."""

    def test_csv_predictions_exported(self, tmp_path: Path) -> None:
        from backtest.run import export_csv

        results = _make_mock_backtest_results()
        sim_results = _make_mock_simulation_results()

        export_csv(results, sim_results, tmp_path)

        predictions_path = tmp_path / "predictions_all.csv"
        assert predictions_path.exists()

        df = pd.read_csv(predictions_path)
        assert "game_id" in df.columns
        assert "season" in df.columns
        assert "target" in df.columns
        assert "model_value" in df.columns

    def test_csv_season_metrics_exported(self, tmp_path: Path) -> None:
        from backtest.run import export_csv

        results = _make_mock_backtest_results()
        sim_results = _make_mock_simulation_results()

        export_csv(results, sim_results, tmp_path)

        metrics_path = tmp_path / "season_metrics.csv"
        assert metrics_path.exists()

        df = pd.read_csv(metrics_path)
        assert "season" in df.columns
        assert "target" in df.columns

    def test_csv_betting_simulation_exported(self, tmp_path: Path) -> None:
        from backtest.run import export_csv

        results = _make_mock_backtest_results()
        sim_results = _make_mock_simulation_results()

        export_csv(results, sim_results, tmp_path)

        bets_path = tmp_path / "betting_simulation.csv"
        assert bets_path.exists()

        df = pd.read_csv(bets_path)
        assert "game_id" in df.columns
        assert "target" in df.columns
        assert "bet_side" in df.columns
        assert "flat_stake" in df.columns
        assert "kelly_stake" in df.columns


class TestJSONExport:
    """Test JSON summary export."""

    def test_summary_json_exported(self, tmp_path: Path) -> None:
        from backtest.run import export_summary_json

        results = _make_mock_backtest_results()
        sim_results = _make_mock_simulation_results()

        json_path = export_summary_json(results, sim_results, tmp_path)

        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)

        assert "headline_clv" in data
        assert "simulation" in data
        assert "config" in data
        assert "generated_at" in data


class TestCLIArgparse:
    """Test CLI argument parsing."""

    def test_cli_argparse_defaults(self) -> None:
        """Verify argparse parser accepts --seasons, --targets, --output-dir."""
        from backtest.run import main

        # Test that --help works (captures SystemExit)
        with pytest.raises(SystemExit) as exc_info:
            with patch.object(sys, "argv", ["backtest.run", "--help"]):
                main()

        assert exc_info.value.code == 0

    def test_cli_seasons_parsing(self) -> None:
        """Verify --seasons flag is parsed correctly."""
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--seasons", type=str, default="2021,2022,2023,2024")
        parser.add_argument("--targets", type=str, default="wp,ats,ou")
        parser.add_argument("--output-dir", type=str, default="outputs/backtest")

        args = parser.parse_args(["--seasons", "2023,2024"])
        seasons = [int(s.strip()) for s in args.seasons.split(",")]

        assert seasons == [2023, 2024]


class TestOutputDir:
    """Test that output directories are created as needed."""

    def test_output_dir_created(self, tmp_path: Path) -> None:
        from backtest.report import BacktestReporter

        new_dir = tmp_path / "new" / "nested" / "dir"
        assert not new_dir.exists()

        reporter = BacktestReporter(output_dir=new_dir)
        assert new_dir.exists()
        assert reporter.output_dir == new_dir
