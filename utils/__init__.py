"""Utilities module for NFL Prediction System."""

from .logging_config import setup_logging, get_logger, log_data_operation
from .date_utils import (
    get_current_nfl_week,
    get_current_nfl_season,
    parse_nfl_date,
    is_game_time,
    get_snapshot_time,
)
from .probability_utils import (
    moneyline_to_probability,
    probability_to_moneyline,
    devig_probabilities,
    implied_probability,
    edge_calculation,
)
from .validation import (
    validate_game_data,
    validate_odds_data,
    validate_prediction_data,
    validate_data_quality,
    validate_nfl_business_rules,
    validate_temporal_consistency,
)
from .exceptions import (
    NFLPredictException,
    DataIngestionError,
    DataValidationError,
    ModelTrainingError,
    ModelPredictionError,
    ConfigurationError,
    ExternalAPIError,
    BacktestError,
    FeatureEngineeringError,
)
from .betting_utils import (
    BetType,
    BettingResult,
    calculate_moneyline_ev,
    calculate_spread_ev,
    calculate_total_ev,
    analyze_game_betting_opportunities,
    summarize_betting_session,
)
from .kelly_criterion import (
    KellyMode,
    KellyResult,
    KellyCalculator,
)
from .bankroll_manager import (
    RiskLevel,
    AlertType,
    BankrollAlert,
    BettingSession,
    BankrollManager,
)
from .unit_sizing import (
    ConfidenceMethod,
    UnitScale,
    ConfidenceMetrics,
    UnitRecommendation,
    UnitSizer,
)
from .bet_selector import (
    FilterReason,
    FilterCriteria,
    BetCandidate,
    BetSelectionResult,
    BetSelector,
    create_default_criteria,
    create_aggressive_criteria,
)
from .bet_recommender import (
    RecommendationTier,
    RecommendationAction,
    UnitRecommendation,
    BetRecommendation,
    RecommendationPortfolio,
    BetRecommender,
)

__all__ = [
    # Logging
    "setup_logging",
    "get_logger",
    "log_data_operation",
    # Date utilities
    "get_current_nfl_week",
    "get_current_nfl_season", 
    "parse_nfl_date",
    "is_game_time",
    "get_snapshot_time",
    # Probability utilities
    "moneyline_to_probability",
    "probability_to_moneyline",
    "devig_probabilities",
    "implied_probability",
    "edge_calculation",
    # Validation
    "validate_game_data",
    "validate_odds_data",
    "validate_prediction_data",
    "validate_data_quality",
    "validate_nfl_business_rules",
    "validate_temporal_consistency",
    # Exceptions
    "NFLPredictException",
    "DataIngestionError",
    "DataValidationError",
    "ModelTrainingError",
    "ModelPredictionError",
    "ConfigurationError",
    "ExternalAPIError",
    "BacktestError",
    "FeatureEngineeringError",
    # Betting utilities
    "BetType",
    "BettingResult",
    "calculate_moneyline_ev",
    "calculate_spread_ev",
    "calculate_total_ev",
    "analyze_game_betting_opportunities",
    "summarize_betting_session",
    # Kelly criterion
    "KellyMode",
    "KellyResult",
    "KellyCalculator",
    # Bankroll management
    "RiskLevel",
    "AlertType",
    "BankrollAlert",
    "BettingSession",
    "BankrollManager",
    # Unit sizing
    "ConfidenceMethod",
    "UnitScale",
    "ConfidenceMetrics",
    "UnitRecommendation",
    "UnitSizer",
    # Bet selection
    "FilterReason",
    "FilterCriteria",
    "BetCandidate",
    "BetSelectionResult",
    "BetSelector",
    "create_default_criteria",
    "create_aggressive_criteria",
    # Bet recommendation
    "RecommendationTier",
    "RecommendationAction",
    "UnitRecommendation",
    "BetRecommendation",
    "RecommendationPortfolio",
    "BetRecommender",
]