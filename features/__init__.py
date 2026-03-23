"""
Features module for NFL prediction system.

This module contains all feature engineering components:
- Team form calculations (rolling EPA, success rates)
- Elo rating system with uncertainty tracking
- Contextual features (travel, weather, venue, rest)
- Weather impact modeling
- Market anchor features (betting odds, line movements)
- Feature validation system (data leakage detection, distribution validation)

All feature calculators follow a consistent interface and provide
comprehensive validation and testing capabilities.
"""

from .contextual import ContextualFeaturesCalculator
from .elo_features import EloFeatureBuilder
from .market_anchors import MarketAnchorFeaturesCalculator
from .opponent_adj import OpponentAdjuster
from .qb_tracking import QBTracker
from .team_form import TeamFormCalculator
from .validation import FeatureValidator
from .weather import WeatherFeaturesCalculator

__all__ = [
    "ContextualFeaturesCalculator",
    "EloFeatureBuilder",
    "FeatureValidator",
    "MarketAnchorFeaturesCalculator",
    "OpponentAdjuster",
    "QBTracker",
    "TeamFormCalculator",
    "WeatherFeaturesCalculator",
]
