"""Backtesting module for NFL Prediction System.

New Phase 6 modules:
- engine: BacktestEngine orchestrating per-season walk-forward retraining
- era: Era normalization (16-vs-17 game) and COVID annotation utilities
- metrics: Brier decomposition, per-target metrics, season summary
- simulation: Betting simulation with slippage, flat-stake, and Kelly strategies

Legacy modules (pre-Phase 4, imported lazily):
- clv_tracking: BetRecord, CLVSummary dataclasses
- walkforward: WalkForwardBacktester (superseded by BacktestEngine)
- visualization: MetricsVisualizer
- reporting: BacktestReporter
"""

# Phase 6 engine and data types
from .engine import (
    BacktestConfig,
    BacktestEngine,
    BacktestResults,
    SeasonResult,
    TargetResult,
)

# Era normalization utilities
from .era import (
    ERA_TRANSITION_SEASON,
    get_covid_hfa_annotation,
    get_season_total_weeks,
    normalize_week_to_progress,
)

# Metrics (Phase 6 rewrite)
from .metrics import (
    brier_decomposition,
    compute_season_summary,
    compute_target_metrics,
)

# Betting simulation (Phase 6 Plan 02)
from .simulation import (
    BetRecord,
    BettingSimulator,
    SimulationConfig,
    SimulationResults,
    StrategyResult,
)

__all__ = [
    "ERA_TRANSITION_SEASON",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResults",
    "BetRecord",
    "BettingSimulator",
    "SeasonResult",
    "SimulationConfig",
    "SimulationResults",
    "StrategyResult",
    "TargetResult",
    "brier_decomposition",
    "compute_season_summary",
    "compute_target_metrics",
    "get_covid_hfa_annotation",
    "get_season_total_weeks",
    "normalize_week_to_progress",
]
