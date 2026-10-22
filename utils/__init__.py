"""Utilities module for NFL Prediction System.

The re-exported names below resolve LAZILY (PEP 562 module ``__getattr__``). The request path
imports three small utils modules (``get_logger``, ``utils.date_utils``, ``utils.team_data``), and
the old eager ``__init__`` dragged pandas, numpy, pyarrow and ``conf`` into every one of them
(Phase 36 D-11, SPEC R1). Now ``from utils import X`` imports only X's own submodule, on first use.

Every existing import site keeps working unedited: ``from utils import X`` goes through
``__getattr__``, and ``import utils.game_lock as lock_rule`` or a monkeypatch string such as
``"utils.date_utils.get_current_nfl_week"`` work because importing a submodule sets the package
attribute itself. The ``if TYPE_CHECKING:`` block lets pyright see every type; nothing in it runs.
``_LAZY`` must name exactly ``__all__`` (a test pins it). CLEAN-02 (Phase 38) removed the dead
v1.0 ``BetSelector`` and ``BetRecommender`` from the package surface; their modules stay on disk
and are imported by module path only.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pyright reads these; nothing is imported at run time
    from .bankroll_manager import AlertType as AlertType
    from .bankroll_manager import BankrollAlert as BankrollAlert
    from .bankroll_manager import BankrollManager as BankrollManager
    from .bankroll_manager import BettingSession as BettingSession
    from .bankroll_manager import RiskLevel as RiskLevel
    from .bet_recommender import BetRecommendation as BetRecommendation
    from .bet_recommender import RecommendationAction as RecommendationAction
    from .bet_recommender import RecommendationPortfolio as RecommendationPortfolio
    from .bet_recommender import RecommendationTier as RecommendationTier
    from .bet_selector import BetCandidate as BetCandidate
    from .bet_selector import BetSelectionResult as BetSelectionResult
    from .bet_selector import FilterCriteria as FilterCriteria
    from .bet_selector import FilterReason as FilterReason
    from .bet_selector import create_aggressive_criteria as create_aggressive_criteria
    from .bet_selector import create_default_criteria as create_default_criteria
    from .betting_utils import BettingResult as BettingResult
    from .betting_utils import BetType as BetType
    from .betting_utils import (
        analyze_game_betting_opportunities as analyze_game_betting_opportunities,
    )
    from .betting_utils import calculate_moneyline_ev as calculate_moneyline_ev
    from .betting_utils import calculate_spread_ev as calculate_spread_ev
    from .betting_utils import calculate_total_ev as calculate_total_ev
    from .betting_utils import summarize_betting_session as summarize_betting_session
    from .date_utils import get_current_nfl_season as get_current_nfl_season
    from .date_utils import get_current_nfl_week as get_current_nfl_week
    from .date_utils import is_game_time as is_game_time
    from .date_utils import parse_nfl_date as parse_nfl_date
    from .exceptions import BacktestError as BacktestError
    from .exceptions import ConfigurationError as ConfigurationError
    from .exceptions import DataIngestionError as DataIngestionError
    from .exceptions import DataValidationError as DataValidationError
    from .exceptions import ExternalAPIError as ExternalAPIError
    from .exceptions import FeatureEngineeringError as FeatureEngineeringError
    from .exceptions import ModelPredictionError as ModelPredictionError
    from .exceptions import ModelTrainingError as ModelTrainingError
    from .exceptions import NFLPredictException as NFLPredictException
    from .exceptions import StorageError as StorageError
    from .exceptions import WeatherDataError as WeatherDataError
    from .ingestion_args import (
        add_standard_ingestion_args as add_standard_ingestion_args,
    )
    from .ingestion_args import get_current_season_weeks as get_current_season_weeks
    from .ingestion_args import get_ingestion_summary as get_ingestion_summary
    from .ingestion_args import parse_season_week_args as parse_season_week_args
    from .kelly_criterion import KellyCalculator as KellyCalculator
    from .kelly_criterion import KellyMode as KellyMode
    from .kelly_criterion import KellyResult as KellyResult
    from .logging_config import get_logger as get_logger
    from .logging_config import log_data_operation as log_data_operation
    from .logging_config import setup_logging as setup_logging
    from .probability_utils import devig_probabilities as devig_probabilities
    from .probability_utils import edge_calculation as edge_calculation
    from .probability_utils import implied_probability as implied_probability
    from .probability_utils import moneyline_to_probability as moneyline_to_probability
    from .probability_utils import probability_to_moneyline as probability_to_moneyline
    from .unit_sizing import ConfidenceMethod as ConfidenceMethod
    from .unit_sizing import ConfidenceMetrics as ConfidenceMetrics
    from .unit_sizing import UnitRecommendation as UnitRecommendation
    from .unit_sizing import UnitScale as UnitScale
    from .unit_sizing import UnitSizer as UnitSizer
    from .validation import validate_data_quality as validate_data_quality
    from .validation import validate_game_data as validate_game_data
    from .validation import validate_nfl_business_rules as validate_nfl_business_rules
    from .validation import validate_odds_data as validate_odds_data
    from .validation import validate_prediction_data as validate_prediction_data
    from .validation import (
        validate_temporal_consistency as validate_temporal_consistency,
    )

# Each re-exported name and the submodule (relative to this package) that defines it.
_LAZY: dict[str, str] = {
    "AlertType": ".bankroll_manager",
    "BankrollAlert": ".bankroll_manager",
    "BankrollManager": ".bankroll_manager",
    "BettingSession": ".bankroll_manager",
    "RiskLevel": ".bankroll_manager",
    "BetRecommendation": ".bet_recommender",
    "RecommendationAction": ".bet_recommender",
    "RecommendationPortfolio": ".bet_recommender",
    "RecommendationTier": ".bet_recommender",
    "BetCandidate": ".bet_selector",
    "BetSelectionResult": ".bet_selector",
    "FilterCriteria": ".bet_selector",
    "FilterReason": ".bet_selector",
    "create_aggressive_criteria": ".bet_selector",
    "create_default_criteria": ".bet_selector",
    "BettingResult": ".betting_utils",
    "BetType": ".betting_utils",
    "analyze_game_betting_opportunities": ".betting_utils",
    "calculate_moneyline_ev": ".betting_utils",
    "calculate_spread_ev": ".betting_utils",
    "calculate_total_ev": ".betting_utils",
    "summarize_betting_session": ".betting_utils",
    "get_current_nfl_season": ".date_utils",
    "get_current_nfl_week": ".date_utils",
    "is_game_time": ".date_utils",
    "parse_nfl_date": ".date_utils",
    "BacktestError": ".exceptions",
    "ConfigurationError": ".exceptions",
    "DataIngestionError": ".exceptions",
    "DataValidationError": ".exceptions",
    "ExternalAPIError": ".exceptions",
    "FeatureEngineeringError": ".exceptions",
    "ModelPredictionError": ".exceptions",
    "ModelTrainingError": ".exceptions",
    "NFLPredictException": ".exceptions",
    "StorageError": ".exceptions",
    "WeatherDataError": ".exceptions",
    "add_standard_ingestion_args": ".ingestion_args",
    "get_current_season_weeks": ".ingestion_args",
    "get_ingestion_summary": ".ingestion_args",
    "parse_season_week_args": ".ingestion_args",
    "KellyCalculator": ".kelly_criterion",
    "KellyMode": ".kelly_criterion",
    "KellyResult": ".kelly_criterion",
    "get_logger": ".logging_config",
    "log_data_operation": ".logging_config",
    "setup_logging": ".logging_config",
    "devig_probabilities": ".probability_utils",
    "edge_calculation": ".probability_utils",
    "implied_probability": ".probability_utils",
    "moneyline_to_probability": ".probability_utils",
    "probability_to_moneyline": ".probability_utils",
    "ConfidenceMethod": ".unit_sizing",
    "ConfidenceMetrics": ".unit_sizing",
    "UnitRecommendation": ".unit_sizing",
    "UnitScale": ".unit_sizing",
    "UnitSizer": ".unit_sizing",
    "validate_data_quality": ".validation",
    "validate_game_data": ".validation",
    "validate_nfl_business_rules": ".validation",
    "validate_odds_data": ".validation",
    "validate_prediction_data": ".validation",
    "validate_temporal_consistency": ".validation",
}

__all__ = [
    "AlertType",
    "BacktestError",
    "BankrollAlert",
    "BankrollManager",
    "BetCandidate",
    "BetRecommendation",
    "BetSelectionResult",
    # Betting utilities
    "BetType",
    "BettingResult",
    "BettingSession",
    # Unit sizing
    "ConfidenceMethod",
    "ConfidenceMetrics",
    "ConfigurationError",
    "DataIngestionError",
    "DataValidationError",
    "ExternalAPIError",
    "FeatureEngineeringError",
    "FilterCriteria",
    # Bet selection
    "FilterReason",
    "KellyCalculator",
    # Kelly criterion
    "KellyMode",
    "KellyResult",
    "ModelPredictionError",
    "ModelTrainingError",
    # Exceptions
    "NFLPredictException",
    "RecommendationAction",
    "RecommendationPortfolio",
    # Bet recommendation
    "RecommendationTier",
    # Bankroll management
    "RiskLevel",
    "StorageError",
    "UnitRecommendation",
    "UnitScale",
    "UnitSizer",
    "WeatherDataError",
    "add_standard_ingestion_args",
    "analyze_game_betting_opportunities",
    "calculate_moneyline_ev",
    "calculate_spread_ev",
    "calculate_total_ev",
    "create_aggressive_criteria",
    "create_default_criteria",
    "devig_probabilities",
    "edge_calculation",
    "get_current_nfl_season",
    # Date utilities
    "get_current_nfl_week",
    "get_current_season_weeks",
    "get_ingestion_summary",
    "get_logger",
    "implied_probability",
    "is_game_time",
    "log_data_operation",
    # Probability utilities
    "moneyline_to_probability",
    "parse_nfl_date",
    "parse_season_week_args",
    "probability_to_moneyline",
    # Logging
    "setup_logging",
    "summarize_betting_session",
    "validate_data_quality",
    # Validation
    "validate_game_data",
    "validate_nfl_business_rules",
    "validate_odds_data",
    "validate_prediction_data",
    "validate_temporal_consistency",
]


def __getattr__(name: str) -> Any:
    """Import the submodule that defines *name* on first access, then cache the attribute."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(importlib.import_module(module_name, __name__), name)
    globals()[name] = (
        value  # later access is a plain attribute, so monkeypatch keeps working
    )
    return value


def __dir__() -> list[str]:
    """List the loaded attributes together with every lazily re-exported name."""
    return sorted(set(globals()) | set(__all__))
