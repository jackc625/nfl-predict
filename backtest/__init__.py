"""Backtesting module for NFL Prediction System.

New Phase 6 modules:
- engine: BacktestEngine orchestrating per-season walk-forward retraining
- era: Era normalization (16-vs-17 game) and COVID annotation utilities
- metrics: Brier decomposition, per-target metrics, season summary

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

__all__ = [
    "ERA_TRANSITION_SEASON",
    "BacktestConfig",
    "BacktestEngine",
    "BacktestResults",
    "SeasonResult",
    "TargetResult",
    "get_covid_hfa_annotation",
    "get_season_total_weeks",
    "normalize_week_to_progress",
]
