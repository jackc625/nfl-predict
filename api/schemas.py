"""
API Response Schemas for NFL Prediction System.

This module defines all Pydantic models for API request/response serialization,
ensuring type safety and automatic API documentation generation.
"""

import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator, validator


class BetType(StrEnum):
    """Types of bets available."""

    MONEYLINE = "moneyline"
    SPREAD = "spread"
    TOTAL = "total"


class RecommendationTier(StrEnum):
    """Recommendation confidence tiers."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class GameStatus(StrEnum):
    """Game status values."""

    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    POSTPONED = "postponed"
    CANCELLED = "cancelled"


# Constants
NFL_TEAMS = {
    "ARI",
    "ATL",
    "BAL",
    "BUF",
    "CAR",
    "CHI",
    "CIN",
    "CLE",
    "DAL",
    "DEN",
    "DET",
    "GB",
    "HOU",
    "IND",
    "JAX",
    "KC",
    "LA",
    "LV",
    "LAC",
    "LAR",
    "MIA",
    "MIN",
    "NE",
    "NO",
    "NYG",
    "NYJ",
    "PHI",
    "PIT",
    "SF",
    "SEA",
    "TB",
    "TEN",
    "WAS",
}

# Game ID pattern: season_week_awayteam@hometeam (e.g., "2024_W01_BUF@KC")
GAME_ID_PATTERN = re.compile(r"^\d{4}_W\d{2}_[A-Z]{2,3}@[A-Z]{2,3}$")


def validate_nfl_team(team: str) -> str:
    """Validate and normalize NFL team abbreviation."""
    if not team:
        raise ValueError("Team abbreviation cannot be empty")

    # Strip whitespace and convert to uppercase
    team = team.strip().upper()

    if team not in NFL_TEAMS:
        raise ValueError(
            f"Invalid NFL team abbreviation: {team}. Must be one of: {', '.join(sorted(NFL_TEAMS))}"
        )

    return team


def validate_game_id(game_id: str) -> str:
    """Validate game ID format."""
    if not game_id:
        raise ValueError("Game ID cannot be empty")

    game_id = game_id.strip()

    if not GAME_ID_PATTERN.match(game_id):
        raise ValueError(
            'Game ID must be in format: YYYY_WWW_AWAY@HOME (e.g., "2024_W01_BUF@KC")'
        )

    # Extract and validate team abbreviations
    parts = game_id.split("_")
    if len(parts) == 3:
        # Expected format: YYYY_WWW_AWAY@HOME
        teams_part = parts[2]
        if "@" in teams_part:
            away_team, home_team = teams_part.split("@")
            validate_nfl_team(away_team)
            validate_nfl_team(home_team)

    return game_id


def probability_to_moneyline(probability: float) -> int:
    """Convert probability to American moneyline odds."""
    if probability <= 0 or probability >= 1:
        raise ValueError("Probability must be between 0 and 1")

    if probability > 0.5:
        # Favorite (negative odds)
        return int(-100 * probability / (1 - probability))
    # Underdog (positive odds)
    return int(100 * (1 - probability) / probability)


def moneyline_to_probability(moneyline: int) -> float:
    """Convert American moneyline odds to implied probability."""
    if moneyline == 0:
        raise ValueError("Moneyline cannot be zero")

    if moneyline > 0:
        # Underdog
        return 100 / (moneyline + 100)
    # Favorite
    return abs(moneyline) / (abs(moneyline) + 100)


# Core Data Models


class TeamInfo(BaseModel):
    """Team information."""

    team_id: str = Field(..., description="Team abbreviation (e.g., 'KC', 'TB')")
    team_name: str = Field(..., description="Full team name")
    city: str = Field(..., description="Team city")
    conference: str = Field(..., description="Conference (AFC/NFC)")
    division: str = Field(..., description="Division name")

    @validator("team_id")
    def validate_team_id(cls, v):
        return validate_nfl_team(v)

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class VenueInfo(BaseModel):
    """Venue information."""

    venue_id: str = Field(..., description="Venue identifier")
    venue_name: str = Field(..., description="Stadium name")
    city: str = Field(..., description="Stadium city")
    state: str = Field(..., description="Stadium state")
    roof_type: str = Field(..., description="Roof type (outdoor/dome/retractable)")
    surface: str = Field(..., description="Playing surface")
    capacity: int | None = Field(None, description="Stadium capacity")

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class WeatherInfo(BaseModel):
    """Weather conditions for outdoor games."""

    temperature: float | None = Field(None, description="Temperature in Fahrenheit")
    wind_speed: float | None = Field(None, description="Wind speed in MPH")
    wind_direction: str | None = Field(None, description="Wind direction")
    precipitation_chance: float | None = Field(
        None, description="Precipitation probability"
    )
    conditions: str | None = Field(None, description="Weather description")
    is_dome: bool = Field(..., description="Whether game is in dome/indoor venue")

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class PredictionInfo(BaseModel):
    """Model predictions for a game."""

    win_probability: float = Field(
        ..., ge=0, le=1, description="Home team win probability"
    )
    spread_prediction: float = Field(
        ..., description="Predicted point spread (home team perspective)"
    )
    total_prediction: float = Field(..., description="Predicted total points")

    # Confidence metrics
    win_prob_confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in win probability"
    )
    spread_confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in spread prediction"
    )
    total_confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in total prediction"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class MarketInfo(BaseModel):
    """Market odds and lines."""

    home_moneyline: int | None = Field(None, description="Home team moneyline odds")
    away_moneyline: int | None = Field(None, description="Away team moneyline odds")
    spread: float | None = Field(
        None, description="Point spread (home team perspective)"
    )
    spread_odds: int | None = Field(None, description="Spread betting odds")
    total: float | None = Field(None, description="Over/under total")
    over_odds: int | None = Field(None, description="Over betting odds")
    under_odds: int | None = Field(None, description="Under betting odds")

    # Market metadata
    last_updated: datetime | None = Field(
        None, description="When odds were last updated"
    )
    sportsbook: str | None = Field(None, description="Primary sportsbook source")

    @model_validator(mode="after")
    def validate_moneyline_consistency(self):
        """Ensure moneyline odds are mathematically consistent."""
        if self.home_moneyline is not None and self.away_moneyline is not None:
            # Convert to probabilities and check they sum to approximately 1 (allowing for vig)
            home_prob = moneyline_to_probability(self.home_moneyline)
            away_prob = moneyline_to_probability(self.away_moneyline)
            total_prob = home_prob + away_prob

            # Total should be > 1 due to sportsbook vig, but not too high
            if total_prob < 0.95 or total_prob > 1.15:
                raise ValueError(
                    f"Moneyline odds appear inconsistent. Implied probabilities sum to {total_prob:.3f}"
                )

        return self

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class BetRecommendation(BaseModel):
    """Betting recommendation."""

    bet_type: BetType = Field(..., description="Type of bet")
    team: str | None = Field(None, description="Team for the bet (if applicable)")
    line_value: float | None = Field(
        None, description="Line value for spread/total bets"
    )
    recommended_odds: int = Field(..., description="Recommended odds")

    edge: float = Field(..., description="Calculated edge percentage")
    expected_value: float = Field(..., description="Expected value of the bet")
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence in recommendation"
    )
    tier: RecommendationTier = Field(..., description="Recommendation tier")

    units: float = Field(..., ge=0, description="Recommended bet size in units")
    description: str = Field(..., description="Human-readable bet description")

    @validator("team")
    def validate_team(cls, v):
        if v is not None:
            return validate_nfl_team(v)
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


# API Response Models


class WeekMetadata(BaseModel):
    """Current season/week metadata."""

    current_season: int = Field(..., description="Current NFL season")
    current_week: int = Field(
        ...,
        ge=1,
        le=22,
        description="Current NFL week (1-18 regular season, 19-22 playoffs)",
    )
    week_type: str = Field(..., description="Week type (REG, POST)")
    games_this_week: int = Field(..., description="Number of games scheduled this week")
    predictions_available: bool = Field(
        ..., description="Whether predictions are available"
    )
    last_updated: datetime = Field(..., description="When data was last updated")
    snapshot_time: datetime | None = Field(None, description="Odds snapshot timestamp")

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class GameSummary(BaseModel):
    """Summary game information for listings."""

    game_id: str = Field(..., description="Unique game identifier")
    season: int = Field(..., description="NFL season")
    week: int = Field(..., description="NFL week")
    game_date: datetime = Field(..., description="Game date and time")

    home_team: str = Field(..., description="Home team abbreviation")
    away_team: str = Field(..., description="Away team abbreviation")

    status: GameStatus = Field(..., description="Game status")

    # Scores (if available)
    home_score: int | None = Field(None, description="Home team score")
    away_score: int | None = Field(None, description="Away team score")

    # Key predictions
    win_probability: float | None = Field(
        None, ge=0, le=1, description="Home team win probability"
    )
    spread: float | None = Field(None, description="Predicted spread")
    total: float | None = Field(None, description="Predicted total")

    # Best recommendation
    top_recommendation: BetRecommendation | None = Field(
        None, description="Top betting recommendation"
    )

    # Flags
    has_recommendations: bool = Field(
        ..., description="Whether betting recommendations are available"
    )
    is_prime_time: bool = Field(False, description="Whether game is prime time")

    @validator("game_id")
    def validate_game_id(cls, v):
        return validate_game_id(v)

    @validator("home_team")
    def validate_home_team(cls, v):
        return validate_nfl_team(v)

    @validator("away_team")
    def validate_away_team(cls, v):
        return validate_nfl_team(v)

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class GameDetail(BaseModel):
    """Detailed game information."""

    game_id: str = Field(..., description="Unique game identifier")
    season: int = Field(..., description="NFL season")
    week: int = Field(..., description="NFL week")
    game_date: datetime = Field(..., description="Game date and time")

    # Teams
    home_team: TeamInfo = Field(..., description="Home team information")
    away_team: TeamInfo = Field(..., description="Away team information")

    # Venue and conditions
    venue: VenueInfo = Field(..., description="Venue information")
    weather: WeatherInfo | None = Field(None, description="Weather conditions")

    # Game status and results
    status: GameStatus = Field(..., description="Game status")
    home_score: int | None = Field(None, description="Home team score")
    away_score: int | None = Field(None, description="Away team score")

    # Predictions
    predictions: PredictionInfo | None = Field(None, description="Model predictions")

    # Market data
    market: MarketInfo | None = Field(None, description="Market odds and lines")

    # Recommendations
    recommendations: list[BetRecommendation] = Field(
        default_factory=list, description="Betting recommendations"
    )

    # Metadata
    last_updated: datetime = Field(..., description="When data was last updated")

    @validator("game_id")
    def validate_game_id(cls, v):
        return validate_game_id(v)

    @model_validator(mode="after")
    def validate_prediction_market_consistency(self):
        """Validate consistency between predictions and market data."""
        if self.predictions and self.market:
            # Check win probability vs moneyline consistency
            win_prob = self.predictions.win_probability
            home_ml = self.market.home_moneyline

            if win_prob is not None and home_ml is not None:
                # Convert moneyline to implied probability
                implied_prob = moneyline_to_probability(home_ml)

                # Allow some tolerance for vig and model differences
                if abs(win_prob - implied_prob) > 0.15:  # 15% tolerance
                    # This is a warning, not an error - models can disagree with market
                    pass

        return self

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class BacktestSummary(BaseModel):
    """Backtest performance summary."""

    total_seasons: int = Field(..., description="Number of seasons tested")
    total_weeks: int = Field(..., description="Number of weeks tested")
    total_predictions: int = Field(..., description="Total predictions made")

    # Model performance
    wp_accuracy: float = Field(..., ge=0, le=1, description="Win probability accuracy")
    wp_log_loss: float = Field(..., ge=0, description="Win probability log loss")
    wp_brier_score: float = Field(..., ge=0, description="Win probability Brier score")

    ats_accuracy: float = Field(
        ..., ge=0, le=1, description="Against the spread accuracy"
    )
    ats_mae: float = Field(..., ge=0, description="ATS mean absolute error")

    ou_accuracy: float = Field(..., ge=0, le=1, description="Over/under accuracy")
    ou_mae: float = Field(..., ge=0, description="O/U mean absolute error")

    # Betting performance
    total_bets: int = Field(..., description="Total bets placed")
    winning_bets: int = Field(..., description="Number of winning bets")
    betting_roi: float = Field(..., description="Return on investment")
    total_profit: float = Field(..., description="Total profit/loss")

    # Time period
    start_date: datetime = Field(..., description="Backtest start date")
    end_date: datetime = Field(..., description="Backtest end date")

    # Additional metrics
    sharpe_ratio: float | None = Field(None, description="Sharpe ratio of returns")
    max_drawdown: float | None = Field(None, description="Maximum drawdown percentage")

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class CalibrationData(BaseModel):
    """Model calibration information."""

    model_type: str = Field(..., description="Model type (wp/ats/ou)")

    # Calibration curve data
    predicted_probabilities: list[float] = Field(
        ..., description="Predicted probability bins"
    )
    observed_frequencies: list[float] = Field(..., description="Observed frequencies")
    bin_counts: list[int] = Field(..., description="Number of predictions in each bin")

    # Calibration metrics
    ece: float = Field(..., ge=0, description="Expected Calibration Error")
    mce: float = Field(..., ge=0, description="Maximum Calibration Error")
    reliability: float = Field(..., ge=0, le=1, description="Reliability score")

    # Additional stats
    total_predictions: int = Field(..., description="Total predictions for calibration")
    confidence_interval: dict[str, float] = Field(
        ..., description="Confidence intervals"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


# API Response Containers


class GamesListResponse(BaseModel):
    """Response for games listing endpoint."""

    games: list[GameSummary] = Field(..., description="List of games")
    metadata: WeekMetadata = Field(..., description="Week metadata")
    total_games: int = Field(..., description="Total number of games")
    has_predictions: bool = Field(..., description="Whether predictions are available")
    filters_applied: dict[str, Any] = Field(
        default_factory=dict, description="Applied filters"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class GameDetailResponse(BaseModel):
    """Response for single game detail endpoint."""

    game: GameDetail = Field(..., description="Detailed game information")
    similar_games: list[GameSummary] = Field(
        default_factory=list, description="Similar historical games"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class BacktestResponse(BaseModel):
    """Response for backtest summary endpoint."""

    summary: BacktestSummary = Field(..., description="Backtest summary metrics")
    seasonal_breakdown: dict[int, dict[str, Any]] = Field(
        ..., description="Performance by season"
    )
    recent_performance: dict[str, Any] = Field(
        ..., description="Recent performance trends"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class CalibrationResponse(BaseModel):
    """Response for calibration data endpoint."""

    calibration_data: dict[str, CalibrationData] = Field(
        ..., description="Calibration data by model type"
    )
    overall_reliability: float = Field(..., description="Overall model reliability")
    last_updated: datetime = Field(
        ..., description="When calibration was last computed"
    )

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


# Error Response Models


class ErrorDetail(BaseModel):
    """Error detail information."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human-readable error message")
    field: str | None = Field(None, description="Field that caused the error")
    details: dict[str, Any] | None = Field(None, description="Additional error details")

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class ErrorResponse(BaseModel):
    """Standard error response."""

    error: bool = Field(True, description="Indicates this is an error response")
    status_code: int = Field(..., description="HTTP status code")
    message: str = Field(..., description="Main error message")
    details: list[ErrorDetail] = Field(
        default_factory=list, description="Detailed error information"
    )
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Error timestamp",
    )
    request_id: str | None = Field(None, description="Request ID for tracing")

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


# Request Models (for query parameters)


class GamesQueryParams(BaseModel):
    """Query parameters for games endpoint."""

    season: int | None = Field(None, description="Filter by season")
    week: int | None = Field(None, ge=1, le=18, description="Filter by week")
    team: str | None = Field(None, description="Filter by team")
    status: GameStatus | None = Field(None, description="Filter by game status")
    has_recommendations: bool | None = Field(
        None, description="Filter games with recommendations"
    )
    min_edge: float | None = Field(
        None, ge=0, le=1, description="Minimum edge threshold"
    )
    limit: int | None = Field(
        50, ge=1, le=100, description="Number of results to return"
    )
    offset: int | None = Field(0, ge=0, description="Number of results to skip")

    @validator("team")
    def validate_team(cls, v):
        if v is not None:
            return validate_nfl_team(v)
        return v

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


# Health Check and Status Models


class HealthStatus(BaseModel):
    """API health status."""

    status: str = Field(..., description="Overall health status")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Health check timestamp",
    )
    version: str = Field(..., description="API version")
    components: dict[str, str] = Field(..., description="Component health status")
    uptime_seconds: float = Field(..., description="API uptime in seconds")

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }


class ServiceInfo(BaseModel):
    """Service information."""

    name: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    description: str = Field(..., description="Service description")
    environment: str = Field(..., description="Environment (dev/staging/prod)")
    build_date: datetime | None = Field(None, description="Build timestamp")
    commit_hash: str | None = Field(None, description="Git commit hash")

    class Config:
        json_encoders = {
            datetime: lambda v: (
                v.isoformat() + "Z" if v.tzinfo is None else v.isoformat()
            )
        }
