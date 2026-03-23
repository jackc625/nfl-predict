"""Unit tests for backtest report generation (Plan 06-03, Task 1).

Tests Plotly chart generation functions, COVID annotation in context,
chart-to-HTML conversion, and BacktestReporter initialization.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Mock data factories
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
    """Create a mock WP predictions DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "game_id": [f"2022_{i:02d}_TEST" for i in range(n)],
        "season": season,
        "week": rng.integers(1, 19, size=n),
        "model_prob": rng.uniform(0.3, 0.7, size=n),
        "actual": rng.integers(0, 2, size=n),
    })


def _make_clv_df(n: int = 50, season: int = 2022) -> pd.DataFrame:
    """Create a mock CLV DataFrame."""
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "game_id": [f"2022_{i:02d}_TEST" for i in range(n)],
        "season": season,
        "week": rng.integers(1, 19, size=n),
        "probability_clv": rng.normal(0.01, 0.05, size=n),
        "has_closing_odds": True,
    })


def _make_mock_backtest_results() -> MockBacktestResults:
    """Create a complete mock BacktestResults."""
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
                target="wp",
                season=season,
                predictions_df=wp_preds,
                clv_df=wp_clv,
                metrics={
                    "accuracy": 0.62,
                    "brier_score": 0.23,
                    "ece": 0.04,
                    "log_loss": 0.65,
                },
                feature_names=["elo_diff", "rolling_wp"],
                best_params={"C": 1.0},
            ),
            "ats": MockTargetResult(
                target="ats",
                season=season,
                predictions_df=ats_preds,
                clv_df=None,
                metrics={"mae": 7.5, "rmse": 10.2},
                feature_names=["elo_diff"],
                best_params={},
            ),
            "ou": MockTargetResult(
                target="ou",
                season=season,
                predictions_df=ou_preds,
                clv_df=None,
                metrics={"mae": 8.1, "rmse": 11.0},
                feature_names=["elo_diff"],
                best_params={},
            ),
        }

        split = MockSplitConfig(
            train_seasons=[2018, 2019],
            hp_val_seasons=[season - 1],
            holdout_seasons=[season],
        )
        season_results.append(MockSeasonResult(
            season=season,
            target_results=target_results,
            split_config=split,
        ))

        all_predictions["wp"].append(wp_preds)
        all_clv["wp"].append(wp_clv)
        all_predictions["ats"].append(ats_preds)
        all_predictions["ou"].append(ou_preds)

    concat_preds = {
        t: pd.concat(dfs, ignore_index=True) for t, dfs in all_predictions.items()
    }
    concat_clv = {
        t: pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
        for t, dfs in all_clv.items()
    }

    return MockBacktestResults(
        config=config,
        season_results=season_results,
        all_predictions=concat_preds,
        all_clv=concat_clv,
        headline_clv={"wp": 0.015, "ats": -0.003, "ou": 0.008},
        odds_coverage={"wp_with_odds": 80, "wp_without_odds": 20},
        covid_annotation={
            "seasons": [2020],
            "home_win_pct": 0.496,
            "normal_home_win_pct": 0.57,
            "note": "COVID-19 empty stadiums reduced HFA.",
            "impact_on_model": "2020 used as HP-val with reduced HFA.",
        },
        era_info={2022: 18, 2023: 18},
    )


def _make_mock_simulation_results() -> MockSimulationResults:
    """Create a complete mock SimulationResults."""
    return MockSimulationResults(
        config=MockSimulationConfig(),
        flat_stake=MockStrategyResult(
            strategy_name="flat_stake",
            total_bets=100,
            winning_bets=55,
            losing_bets=42,
            push_bets=3,
            win_rate=0.55,
            total_wagered=10000.0,
            net_profit=500.0,
            roi=0.05,
            final_bankroll=10500.0,
            max_drawdown=800.0,
            max_drawdown_pct=0.075,
            equity_curve=[10000.0 + i * 5 for i in range(101)],
            bet_timestamps=[f"game_{i}" for i in range(100)],
        ),
        kelly=MockStrategyResult(
            strategy_name="kelly",
            total_bets=100,
            winning_bets=55,
            losing_bets=42,
            push_bets=3,
            win_rate=0.55,
            total_wagered=8000.0,
            net_profit=700.0,
            roi=0.0875,
            final_bankroll=10700.0,
            max_drawdown=600.0,
            max_drawdown_pct=0.055,
            equity_curve=[10000.0 + i * 7 for i in range(101)],
            bet_timestamps=[f"game_{i}" for i in range(100)],
        ),
        bet_records=[],
        by_target={
            "wp": {"n_bets": 40, "wins": 22, "losses": 17, "pushes": 1, "win_rate": 0.55, "flat_roi": 0.03, "kelly_roi": 0.06},
            "ats": {"n_bets": 30, "wins": 16, "losses": 13, "pushes": 1, "win_rate": 0.53, "flat_roi": 0.02, "kelly_roi": 0.04},
            "ou": {"n_bets": 30, "wins": 17, "losses": 12, "pushes": 1, "win_rate": 0.57, "flat_roi": 0.07, "kelly_roi": 0.10},
        },
        by_season={
            2022: {"n_bets": 50, "wins": 27, "losses": 21, "pushes": 2, "win_rate": 0.54, "flat_roi": 0.04, "kelly_roi": 0.07},
            2023: {"n_bets": 50, "wins": 28, "losses": 21, "pushes": 1, "win_rate": 0.56, "flat_roi": 0.06, "kelly_roi": 0.09},
        },
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCalibrationChart:
    """Tests for generate_calibration_chart."""

    def test_calibration_chart_returns_figure(self) -> None:
        from backtest.report import generate_calibration_chart

        results = _make_mock_backtest_results()
        fig = generate_calibration_chart(results)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0

    def test_calibration_chart_has_diagonal(self) -> None:
        from backtest.report import generate_calibration_chart

        results = _make_mock_backtest_results()
        fig = generate_calibration_chart(results)

        # The first trace should be the perfect calibration diagonal
        diagonal_trace = fig.data[0]
        assert diagonal_trace.name == "Perfect Calibration"

    def test_calibration_chart_has_per_season_traces(self) -> None:
        from backtest.report import generate_calibration_chart

        results = _make_mock_backtest_results()
        fig = generate_calibration_chart(results)

        # Should have diagonal + per-season + overall = at least 4 traces
        assert len(fig.data) >= 4


class TestCLVChart:
    """Tests for generate_clv_chart."""

    def test_clv_chart_returns_figure(self) -> None:
        from backtest.report import generate_clv_chart

        results = _make_mock_backtest_results()
        fig = generate_clv_chart(results)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestSeasonHeatmap:
    """Tests for generate_season_heatmap."""

    def test_season_heatmap_returns_figure(self) -> None:
        from backtest.report import generate_season_heatmap

        results = _make_mock_backtest_results()
        fig = generate_season_heatmap(results)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) > 0


class TestEquityChart:
    """Tests for generate_equity_chart."""

    def test_equity_chart_returns_figure(self) -> None:
        from backtest.report import generate_equity_chart

        sim_results = _make_mock_simulation_results()
        fig = generate_equity_chart(sim_results)

        assert isinstance(fig, go.Figure)
        assert len(fig.data) >= 2  # flat + kelly traces

    def test_equity_chart_has_two_strategies(self) -> None:
        from backtest.report import generate_equity_chart

        sim_results = _make_mock_simulation_results()
        fig = generate_equity_chart(sim_results)

        trace_names = [t.name for t in fig.data if hasattr(t, "name") and t.name]
        flat_traces = [n for n in trace_names if "Flat" in n]
        kelly_traces = [n for n in trace_names if "Kelly" in n]
        assert len(flat_traces) >= 1
        assert len(kelly_traces) >= 1


class TestCovidAnnotation:
    """Tests for COVID annotation in metrics context."""

    def test_covid_annotation_in_context(self) -> None:
        from backtest.report import BacktestReporter

        reporter = BacktestReporter(output_dir="outputs/test_backtest")
        results = _make_mock_backtest_results()
        sim_results = _make_mock_simulation_results()

        context = reporter._build_metrics_context(results, sim_results)

        assert "covid" in context
        assert context["covid"]["home_win_pct"] == 0.496
        assert context["covid"]["normal_home_win_pct"] == 0.57

    def test_context_has_all_required_keys(self) -> None:
        from backtest.report import BacktestReporter

        reporter = BacktestReporter(output_dir="outputs/test_backtest")
        results = _make_mock_backtest_results()
        sim_results = _make_mock_simulation_results()

        context = reporter._build_metrics_context(results, sim_results)

        required_keys = [
            "headline_clv", "per_season", "per_target", "simulation",
            "covid", "era", "odds_coverage", "config", "generated_at",
        ]
        for key in required_keys:
            assert key in context, f"Missing key: {key}"


class TestChartToHtml:
    """Tests for chart to HTML div conversion."""

    def test_chart_to_html_div(self) -> None:
        from backtest.report import generate_calibration_chart

        results = _make_mock_backtest_results()
        fig = generate_calibration_chart(results)

        html = fig.to_html(
            full_html=False, include_plotlyjs=False, div_id="test-chart"
        )

        assert "<div" in html
        assert "plotly" in html.lower() or "test-chart" in html


class TestReporterInit:
    """Tests for BacktestReporter initialization."""

    def test_reporter_output_dir_created(self, tmp_path: Path) -> None:
        from backtest.report import BacktestReporter

        output_dir = tmp_path / "new_output_dir"
        assert not output_dir.exists()

        reporter = BacktestReporter(output_dir=output_dir)

        assert output_dir.exists()
        assert reporter.output_dir == output_dir
