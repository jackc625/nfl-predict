"""
Models module for NFL prediction system.

This module contains all model training and prediction components:
- Model training utilities (walk-forward validation, cross-validation)
- Model management (serialization, loading, versioning)
- Training pipeline orchestration
- Model metadata and performance tracking

All utilities enforce strict temporal ordering to prevent look-ahead bias
and ensure reproducible, reliable model training and evaluation.
"""

from .calibrate import CalibrationResults, ProbabilityCalibrator
from .evaluation import (
    BettingSimulationResult,
    EvaluationMetrics,
    ModelEvaluationFramework,
)
from .prediction_pipeline import (
    BetRecommendation,
    BetRecommendationEngine,
    BetType,
    EdgeCalculator,
    FairLine,
    MarketEdge,
    NFLPredictionPipeline,
    OddsConverter,
    UnifiedGamePrediction,
)
from .train_ats import (
    ATSModel,
    ATSModelPrediction,
    ATSModelResults,
    ResidualDistributionConverter,
)
from .train_ou import (
    OUModel,
    OUModelPrediction,
    OUModelResults,
    PoissonScoreModel,
    TotalDistributionConverter,
    WeatherImpactModel,
)
from .train_wp import WinProbabilityModel, WPModelPrediction, WPModelResults
from .utils import (
    CrossValidator,
    ModelManager,
    ModelMetadata,
    TrainingPipeline,
    TrainTestSplit,
    WalkForwardValidator,
)

__all__ = [
    "ATSModel",
    "ATSModelPrediction",
    "ATSModelResults",
    "BetRecommendation",
    "BetRecommendationEngine",
    "BetType",
    "BettingSimulationResult",
    "CalibrationResults",
    "CrossValidator",
    "EdgeCalculator",
    "EvaluationMetrics",
    "FairLine",
    "MarketEdge",
    "ModelEvaluationFramework",
    "ModelManager",
    "ModelMetadata",
    "NFLPredictionPipeline",
    "OUModel",
    "OUModelPrediction",
    "OUModelResults",
    "OddsConverter",
    "PoissonScoreModel",
    "ProbabilityCalibrator",
    "ResidualDistributionConverter",
    "TotalDistributionConverter",
    "TrainTestSplit",
    "TrainingPipeline",
    "UnifiedGamePrediction",
    "WPModelPrediction",
    "WPModelResults",
    "WalkForwardValidator",
    "WeatherImpactModel",
    "WinProbabilityModel",
]
