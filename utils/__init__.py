"""Utilities module for NFL Prediction System."""

from .bankroll_manager import (
    AlertType,
    BankrollAlert,
    BankrollManager,
    BettingSession,
    RiskLevel,
)
from .bet_recommender import (
    BetRecommendation,
    BetRecommender,
    RecommendationAction,
    RecommendationPortfolio,
    RecommendationTier,
)
from .bet_selector import (
    BetCandidate,
    BetSelectionResult,
    BetSelector,
    FilterCriteria,
    FilterReason,
    create_aggressive_criteria,
    create_default_criteria,
)
from .betting_utils import (
    BettingResult,
    BetType,
    analyze_game_betting_opportunities,
    calculate_moneyline_ev,
    calculate_spread_ev,
    calculate_total_ev,
    summarize_betting_session,
)
from .date_utils import (
    get_current_nfl_season,
    get_current_nfl_week,
    get_snapshot_time,
    is_game_time,
    parse_nfl_date,
)
from .exceptions import (
    BacktestError,
    ConfigurationError,
    DataIngestionError,
    DataValidationError,
    ExternalAPIError,
    FeatureEngineeringError,
    ModelPredictionError,
    ModelTrainingError,
    NFLPredictException,
    StorageError,
    WeatherDataError,
)
from .ingestion_args import (
    add_standard_ingestion_args,
    get_current_season_weeks,
    get_ingestion_summary,
    parse_season_week_args,
)
from .kelly_criterion import (
    KellyCalculator,
    KellyMode,
    KellyResult,
)
from .logging_config import get_logger, log_data_operation, setup_logging
from .probability_utils import (
    devig_probabilities,
    edge_calculation,
    implied_probability,
    moneyline_to_probability,
    probability_to_moneyline,
)
from .unit_sizing import (
    ConfidenceMethod,
    ConfidenceMetrics,
    UnitRecommendation,
    UnitScale,
    UnitSizer,
)
from .validation import (
    validate_data_quality,
    validate_game_data,
    validate_nfl_business_rules,
    validate_odds_data,
    validate_prediction_data,
    validate_temporal_consistency,
)

__all__ = [
    "AlertType",
    "BacktestError",
    "BankrollAlert",
    "BankrollManager",
    "BetCandidate",
    "BetRecommendation",
    "BetRecommender",
    "BetSelectionResult",
    "BetSelector",
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
    "get_snapshot_time",
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
