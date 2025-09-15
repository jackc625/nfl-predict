"""Backtesting module for NFL Prediction System."""

from .walkforward import (
    WalkForwardBacktester,
    BacktestConfig,
    BacktestPhase,
    ValidationLevel,
    SeasonSplit,
    BacktestResult,
    BacktestSummary,
    DataLeakageValidator,
    create_default_config,
    create_strict_config,
)
from .metrics import (
    MetricsCalculator,
    ModelType,
    MetricType,
    ClassificationMetrics,
    RegressionMetrics,
    CalibrationMetrics,
    EdgeBucketMetrics,
    SignificanceTest,
    ComprehensiveMetrics,
    create_metrics_summary,
)
from .visualization import (
    MetricsVisualizer,
)
from .clv_tracking import (
    CLVTracker,
    CLVSummary,
    BetRecord,
    CLVMetric,
)

__all__ = [
    # Walk-forward backtesting
    "WalkForwardBacktester",
    "BacktestConfig",
    "BacktestPhase",
    "ValidationLevel",
    "SeasonSplit",
    "BacktestResult",
    "BacktestSummary",
    "DataLeakageValidator",
    "create_default_config",
    "create_strict_config",
    # Evaluation metrics
    "MetricsCalculator",
    "ModelType",
    "MetricType",
    "ClassificationMetrics",
    "RegressionMetrics",
    "CalibrationMetrics",
    "EdgeBucketMetrics",
    "SignificanceTest",
    "ComprehensiveMetrics",
    "create_metrics_summary",
    # Visualization
    "MetricsVisualizer",
    # CLV tracking
    "CLVTracker",
    "CLVSummary",
    "BetRecord",
    "CLVMetric",
]