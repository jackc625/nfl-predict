"""API schemas for NFL Prediction System.

Defines Pydantic v2 models for the DuckDB-backed API layer. These models
represent the data shapes served by the web UI and JSON export endpoints.

Note: This is a clean rewrite for Phase 8. The previous schemas assumed
direct model inference; these schemas work with precomputed cache data.
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

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


class PipelineStatusResponse(BaseModel):
    """Pipeline execution status from the latest execution log."""

    last_run_status: str | None = Field(
        default=None,
        description=(
            "Last pipeline run status: success, failed, degraded, finished_with_skips, "
            "or None"
        ),
    )
    last_run_time: datetime | None = Field(
        default=None, description="Last pipeline run start time"
    )
    last_run_duration_ms: float | None = Field(
        default=None, description="Last pipeline run duration in milliseconds"
    )
    last_run_season: int | None = Field(
        default=None, description="Season of last pipeline run"
    )
    last_run_week: int | None = Field(
        default=None, description="Week of last pipeline run"
    )
    forced: bool = Field(default=False, description="Whether the run was forced")
    warnings_count: int = Field(
        default=0, description="Number of warnings from last run"
    )
    pid: int | None = Field(default=None, description="Process ID of last pipeline run")


class DataFreshnessResponse(BaseModel):
    """Data freshness status from Silver layer file modification times."""

    games_age_hours: float | None = Field(
        default=None, description="Age of games data in hours"
    )
    odds_age_hours: float | None = Field(
        default=None, description="Age of odds snapshot data in hours"
    )
    weather_age_hours: float | None = Field(
        default=None, description="Age of weather data in hours"
    )
    all_fresh: bool = Field(
        default=False, description="Whether all data sources are fresh (<168 hours)"
    )


class ModelStatusResponse(BaseModel):
    """Model artifact status from artifacts/models/ directory."""

    wp_model_exists: bool = Field(
        default=False, description="Whether WP model artifact exists"
    )
    ats_model_exists: bool = Field(
        default=False, description="Whether ATS model artifact exists"
    )
    ou_model_exists: bool = Field(
        default=False, description="Whether O/U model artifact exists"
    )
    wp_model_age_hours: float | None = Field(
        default=None, description="Age of WP model artifact in hours"
    )
    ats_model_age_hours: float | None = Field(
        default=None, description="Age of ATS model artifact in hours"
    )
    ou_model_age_hours: float | None = Field(
        default=None, description="Age of O/U model artifact in hours"
    )
    all_models_exist: bool = Field(
        default=False, description="Whether all three model artifacts exist"
    )


#: What a laptop-only /health section reads on the hosted copy (SPEC R10, D-21): the host has no
#: pipeline log, silver files or model artifacts, so it states that instead of judging their absence.
HostedUnavailable = Literal["not available on the hosted copy"]
HOSTED_UNAVAILABLE: HostedUnavailable = "not available on the hosted copy"


class HealthResponse(BaseModel):
    """Health check response.

    Backward compatible: existing status, cache_ready, last_updated fields
    remain. New pipeline, data_freshness, model_status fields default to None.

    Phase 36 (SPEC R10) adds the publish fields below, on both copies. On the hosted copy the three
    laptop sections read ``HOSTED_UNAVAILABLE``. The publish stamp is whatever the LATEST push
    attempt stamped into this cache; on the laptop that is the latest attempt's stamp, not proof
    these local bytes were deployed (an attempt whose upload failed still stamped the local copy).
    """

    status: str = Field(
        default="ok", description="Health status: ok, degraded, unhealthy"
    )
    cache_ready: bool = Field(
        default=False, description="Whether the DuckDB cache file exists"
    )
    last_updated: datetime | None = Field(
        default=None, description="Last cache update timestamp"
    )
    pipeline: PipelineStatusResponse | HostedUnavailable | None = Field(
        default=None, description="Pipeline execution status from latest log"
    )
    data_freshness: DataFreshnessResponse | HostedUnavailable | None = Field(
        default=None, description="Data freshness status from Silver layer files"
    )
    model_status: ModelStatusResponse | HostedUnavailable | None = Field(
        default=None, description="Model artifact status"
    )
    publish_id: str | None = Field(
        default=None,
        description=(
            "Publish id the latest push attempt stamped into this cache; on the laptop the latest "
            "attempt's stamp, not proof these local bytes were deployed"
        ),
    )
    push_time: datetime | None = Field(
        default=None,
        description=(
            "UTC time the latest push attempt stamped into this cache (never the cache build "
            "time); on the laptop the latest attempt's stamp, not proof of a deployment"
        ),
    )
    data_season: int | None = Field(
        default=None,
        description="Season of the stamped current slate, None when unstamped",
    )
    data_week: int | None = Field(
        default=None,
        description="Week of the stamped current slate, None when unstamped",
    )
    cache_data_format: str | None = Field(
        default=None,
        description="The cache's stored data-format version text, None when it carries none",
    )
    code_data_format: int | None = Field(
        default=None, description="The data-format version this code expects"
    )
    data_format_matches: bool | None = Field(
        default=None,
        description="Whether the cache's data format is the one the code expects",
    )
    stale: bool | None = Field(
        default=None,
        description="Whether the last push is more than 36 hours old or absent (SPEC R8 rule)",
    )
    cache_sha256: str | None = Field(
        default=None,
        description="SHA-256 of the cache file this copy serves, read on this request",
    )
