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

from .team_form import TeamFormCalculator
from .elo_features import EloFeatureBuilder
from .contextual import ContextualFeaturesCalculator
from .weather import WeatherFeaturesCalculator
from .market_anchors import MarketAnchorFeaturesCalculator
from .validation import FeatureValidator

__all__ = [
    'TeamFormCalculator',
    'EloFeatureBuilder',
    'ContextualFeaturesCalculator',
    'WeatherFeaturesCalculator',
    'MarketAnchorFeaturesCalculator',
    'FeatureValidator'
]