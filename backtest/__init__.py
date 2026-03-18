"""Backtesting module for NFL Prediction System."""

from .clv_tracking import (
    BetRecord,
    CLVMetric,
    CLVSummary,
    CLVTracker,
)
from .metrics import (
    CalibrationMetrics,
    ClassificationMetrics,
    ComprehensiveMetrics,
    EdgeBucketMetrics,
    MetricsCalculator,
    MetricType,
    ModelType,
    RegressionMetrics,
    SignificanceTest,
    create_metrics_summary,
)
from .visualization import (
    MetricsVisualizer,
)
from .walkforward import (
    BacktestConfig,
    BacktestPhase,
    BacktestResult,
    BacktestSummary,
    DataLeakageValidator,
    SeasonSplit,
    ValidationLevel,
    WalkForwardBacktester,
    create_default_config,
    create_strict_config,
)

__all__ = [
    "BacktestConfig",
    "BacktestPhase",
    "BacktestResult",
    "BacktestSummary",
    "BetRecord",
    "CLVMetric",
    "CLVSummary",
    # CLV tracking
    "CLVTracker",
    "CalibrationMetrics",
    "ClassificationMetrics",
    "ComprehensiveMetrics",
    "DataLeakageValidator",
    "EdgeBucketMetrics",
    "MetricType",
    # Evaluation metrics
    "MetricsCalculator",
    # Visualization
    "MetricsVisualizer",
    "ModelType",
    "RegressionMetrics",
    "SeasonSplit",
    "SignificanceTest",
    "ValidationLevel",
    # Walk-forward backtesting
    "WalkForwardBacktester",
    "create_default_config",
    "create_metrics_summary",
    "create_strict_config",
]
