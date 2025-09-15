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

from .utils import (
    WalkForwardValidator,
    CrossValidator,
    ModelManager,
    TrainingPipeline,
    TrainTestSplit,
    ModelMetadata
)
from .calibrate import (
    ProbabilityCalibrator,
    CalibrationResults
)
from .evaluation import (
    ModelEvaluationFramework,
    EvaluationMetrics,
    BettingSimulationResult
)
from .train_wp import (
    WinProbabilityModel,
    WPModelPrediction,
    WPModelResults
)
from .train_ats import (
    ATSModel,
    ATSModelPrediction,
    ATSModelResults,
    ResidualDistributionConverter
)
from .train_ou import (
    OUModel,
    OUModelPrediction,
    OUModelResults,
    TotalDistributionConverter,
    PoissonScoreModel,
    WeatherImpactModel
)
from .prediction_pipeline import (
    NFLPredictionPipeline,
    UnifiedGamePrediction,
    FairLine,
    MarketEdge,
    BetRecommendation,
    BetType,
    OddsConverter,
    EdgeCalculator,
    BetRecommendationEngine
)

__all__ = [
    'WalkForwardValidator',
    'CrossValidator',
    'ModelManager',
    'TrainingPipeline',
    'TrainTestSplit',
    'ModelMetadata',
    'ProbabilityCalibrator',
    'CalibrationResults',
    'ModelEvaluationFramework',
    'EvaluationMetrics',
    'BettingSimulationResult',
    'WinProbabilityModel',
    'WPModelPrediction',
    'WPModelResults',
    'ATSModel',
    'ATSModelPrediction',
    'ATSModelResults',
    'ResidualDistributionConverter',
    'OUModel',
    'OUModelPrediction',
    'OUModelResults',
    'TotalDistributionConverter',
    'PoissonScoreModel',
    'WeatherImpactModel',
    'NFLPredictionPipeline',
    'UnifiedGamePrediction',
    'FairLine',
    'MarketEdge',
    'BetRecommendation',
    'BetType',
    'OddsConverter',
    'EdgeCalculator',
    'BetRecommendationEngine'
]