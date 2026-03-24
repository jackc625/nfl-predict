"""API schemas for NFL Prediction System.

Defines Pydantic v2 models for the DuckDB-backed API layer. These models
represent the data shapes served by the web UI and JSON export endpoints.

Note: This is a clean rewrite for Phase 8. The previous schemas assumed
direct model inference; these schemas work with precomputed cache data.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# API response models
# ---------------------------------------------------------------------------


class GamePrediction(BaseModel):
    """A single game prediction row from the DuckDB cache."""

    game_id: str
    season: int
    week: int
    game_date: datetime | None = None
    home_team: str
    away_team: str
    status: GameStatus = GameStatus.SCHEDULED
    home_score: int | None = None
    away_score: int | None = None
    wp_prob: float | None = None
    wp_confidence: str | None = None
    ats_prediction: float | None = None
    ats_confidence: str | None = None
    ou_prediction: float | None = None
    ou_confidence: str | None = None
    market_spread: float | None = None
    market_total: float | None = None
    wp_edge: float | None = None
    ats_edge: float | None = None
    ou_edge: float | None = None
    blended_wp: float | None = None
    blended_ats: float | None = None
    blended_ou: float | None = None


class CacheMeta(BaseModel):
    """Metadata about the DuckDB cache."""

    last_updated: datetime | None = None
    prediction_count: int = 0
    season_range: str = ""


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(default="ok", description="Health status")
    cache_ready: bool = Field(
        default=False, description="Whether the DuckDB cache file exists"
    )
    last_updated: datetime | None = Field(
        default=None, description="Last cache update timestamp"
    )
