"""Data schemas for NFL Prediction System using Pydantic v2."""

from datetime import UTC, datetime
from enum import IntEnum, StrEnum

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class VenueRoof(StrEnum):
    """Venue roof types."""

    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    RETRACTABLE = "retractable"


class GameResult(IntEnum):
    """Game result from home team perspective."""

    AWAY_WIN = -1
    TIE = 0
    HOME_WIN = 1


class GameSchema(BaseModel):
    """Schema for NFL game data (silver layer)."""

    model_config = ConfigDict(use_enum_values=True)

    game_id: str = Field(
        ..., description="Unique game identifier (e.g., 2024_W06_KC@BUF)"
    )
    season: int = Field(..., ge=2000, le=2030, description="NFL season year")
    week: int = Field(
        ..., ge=1, le=22, description="Week number (1-18 regular, 19-22 playoffs)"
    )
    kickoff_et: datetime = Field(..., description="Game kickoff time in ET")
    home_team: str = Field(
        ..., min_length=2, max_length=5, description="Home team abbreviation"
    )
    away_team: str = Field(
        ..., min_length=2, max_length=5, description="Away team abbreviation"
    )
    venue: str = Field(..., description="Stadium/venue name")
    venue_roof: VenueRoof = Field(..., description="Venue roof type")
    home_score: int | None = Field(None, ge=0, description="Home team final score")
    away_score: int | None = Field(None, ge=0, description="Away team final score")
    result: GameResult | None = Field(
        None, description="Game result (1=home win, 0=tie, -1=away win)"
    )

    # Additional game metadata
    game_type: str | None = Field(
        None, description="Game type (REG, WC, DIV, CONF, SB)"
    )
    season_type: str | None = Field(None, description="Season type (Regular, Playoffs)")
    neutral_site: bool | None = Field(
        False, description="Whether game is at neutral site"
    )
    stadium_id: str | None = Field(
        None,
        description=(
            "nflverse stadium code such as BUF00 or RIO00. DECLARED HERE BECAUSE "
            "validate_bronze_to_silver calls model_dump(): an undeclared column is "
            "silently dropped between the transform and the silver write, so it "
            "would gate-pass while never arriving. Matched exactly and "
            "case-sensitively downstream -- never normalized."
        ),
    )

    # Data lineage metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when record was created/ingested",
    )

    @field_validator("home_team", "away_team")
    @classmethod
    def validate_team_format(cls, v: str) -> str:
        """Validate team abbreviation format."""
        return v.upper().strip()

    @field_validator("home_score", "away_score", mode="before")
    @classmethod
    def validate_scores(cls, v):
        """Convert NaN to None for optional score fields."""
        if v is not None and pd.isna(v):
            return None
        return v

    @field_validator("result", mode="before")
    @classmethod
    def validate_result(cls, v):
        """Convert NaN to None for optional result field."""
        if v is not None and pd.isna(v):
            return None
        return v

    @field_validator("kickoff_et", "created_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v):
        """Ensure timestamps are timezone-aware."""
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (ValueError, TypeError):
            pass
        if isinstance(v, str):
            return pd.to_datetime(v)
        if isinstance(v, datetime) and v.tzinfo is None:
            from zoneinfo import ZoneInfo

            return v.replace(tzinfo=ZoneInfo("America/New_York"))
        return v

    @field_validator("game_id")
    @classmethod
    def validate_game_id_format(cls, v: str) -> str:
        """Validate game ID format."""
        if not v or len(v.split("_")) != 3:
            raise ValueError("Game ID must be in format: SEASON_WEEK_AWAY@HOME")
        return v

    @model_validator(mode="after")
    def validate_scores_for_completed_games(self):
        """Completed games MUST have both scores. Future games may have None for both."""
        has_home = self.home_score is not None
        has_away = self.away_score is not None
        if has_home != has_away:
            raise ValueError(
                f"Game {self.game_id}: partial scores "
                f"(home={self.home_score}, away={self.away_score}). "
                f"Completed games must have both scores, future games must have neither."
            )
        return self


class OddsSchema(BaseModel):
    """Schema for odds data (silver layer)."""

    model_config = ConfigDict(use_enum_values=True)

    game_id: str = Field(..., description="Foreign key to games table")
    snapshot_ts: datetime = Field(..., description="Timestamp when odds were captured")
    sportsbook: str = Field(..., description="Sportsbook identifier")

    # Moneyline odds
    ml_home: int | None = Field(None, description="Home team moneyline")
    ml_away: int | None = Field(None, description="Away team moneyline")

    # Spread betting
    spread: float | None = Field(
        None, description="Point spread (home team perspective)"
    )
    spread_ju_home: int | None = Field(-110, description="Home spread juice")
    spread_ju_away: int | None = Field(-110, description="Away spread juice")

    # Totals betting
    total: float | None = Field(None, description="Game total (over/under)")
    total_over_ju: int | None = Field(-110, description="Over juice")
    total_under_ju: int | None = Field(-110, description="Under juice")

    # Additional odds metadata
    is_live: bool | None = Field(False, description="Whether odds are live/in-game")
    last_update: datetime | None = Field(None, description="Last odds update time")

    # Data lineage metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when record was created/ingested",
    )

    @field_validator(
        "ml_home",
        "ml_away",
        "spread_ju_home",
        "spread_ju_away",
        "total_over_ju",
        "total_under_ju",
    )
    @classmethod
    def validate_moneyline_range(cls, v):
        """Validate moneyline/juice values are reasonable."""
        if v is not None and not (-10000 <= v <= 10000):
            raise ValueError("Moneyline/juice values must be between -10000 and +10000")
        return v

    @field_validator("snapshot_ts", "last_update", "created_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v):
        """Ensure timestamps are timezone-aware."""
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (ValueError, TypeError):
            pass
        if isinstance(v, str):
            return pd.to_datetime(v)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v

    @field_validator("spread")
    @classmethod
    def validate_spread_range(cls, v):
        """Validate spread is reasonable."""
        if v is not None and not (-50.0 <= v <= 50.0):
            raise ValueError("Spread must be between -50.0 and +50.0 points")
        return v

    @field_validator("total")
    @classmethod
    def validate_total_range(cls, v):
        """Validate total is reasonable."""
        if v is not None and not (10.0 <= v <= 100.0):
            raise ValueError("Total must be between 10.0 and 100.0 points")
        return v


class OddsTimelineSchema(BaseModel):
    """Schema for a single odds-trajectory snapshot row (silver layer).

    Represents ONE point on a game's line-movement trajectory at the
    ``(game_id, snapshot_ts)`` grain -- the additive sibling of
    ``odds_snapshot`` (D-11). ``odds_snapshot`` is ``game_id`` latest-wins
    (the freeze/closing anchor consumed by ``market_anchors`` + CLV grading);
    this schema instead keeps every distinct snapshot so the PATH of the line
    can be reconstructed.

    Totals are primary for the line-movement signal (D-07); the consensus
    spread is conditional/optional.

    CRITICAL (review 29-02 HIGH): unlike ``OddsSchema`` -- whose timestamp
    validator silently attaches UTC to a NAIVE datetime (schemas.py:194-195) --
    a timezone-naive ``snapshot_ts`` is REJECTED here, never coerced. The
    trajectory grain must never store an ambiguous wall-clock time as if it
    were UTC.
    """

    model_config = ConfigDict(use_enum_values=True)

    game_id: str = Field(..., description="Foreign key to games table")
    snapshot_ts: datetime = Field(
        ...,
        description="Trajectory snapshot timestamp (tz-aware UTC; naive rejected)",
    )

    # Totals are primary for the line-movement signal (D-07); spread optional.
    total: float | None = Field(None, description="Consensus game total (over/under)")
    spread: float | None = Field(
        None, description="Consensus point spread (home perspective; optional)"
    )

    # Provenance: which book/region the consensus snapshot was derived from,
    # so a stored row's source is auditable (mirrors OddsSchema.sportsbook).
    sportsbook: str | None = Field(
        None, description="Sportsbook / consensus source identifier"
    )
    region: str | None = Field(None, description="Odds region (e.g. 'us')")

    # Data lineage metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when record was created/ingested",
    )

    @field_validator("snapshot_ts", mode="before")
    @classmethod
    def reject_naive_snapshot_ts(cls, v):
        """Require a tz-aware ``snapshot_ts``; REJECT naive (review 29-02 HIGH).

        Unlike ``OddsSchema.validate_timestamps`` (which silently attaches UTC
        to a naive datetime, schemas.py:194-195), a naive ``snapshot_ts`` is a
        HARD error here -- it is never silently coerced to UTC.
        """
        if v is None:
            raise ValueError("snapshot_ts is required and must be timezone-aware")
        if isinstance(v, str):
            v = pd.to_datetime(v)
        tzinfo = getattr(v, "tzinfo", None)
        if tzinfo is None:
            raise ValueError(
                "snapshot_ts must be timezone-aware (tz-aware UTC); a naive "
                "datetime was rejected and is never silently coerced to UTC "
                "(review 29-02 HIGH)"
            )
        return v

    @field_validator("total")
    @classmethod
    def validate_total_range(cls, v):
        """Validate total is reasonable (mirrors OddsSchema)."""
        if v is not None and not (10.0 <= v <= 100.0):
            raise ValueError("Total must be between 10.0 and 100.0 points")
        return v

    @field_validator("spread")
    @classmethod
    def validate_spread_range(cls, v):
        """Validate spread is reasonable (mirrors OddsSchema)."""
        if v is not None and not (-50.0 <= v <= 50.0):
            raise ValueError("Spread must be between -50.0 and +50.0 points")
        return v


class WeatherSchema(BaseModel):
    """Schema for weather forecast data (silver layer)."""

    model_config = ConfigDict(use_enum_values=True)

    # Nullable numeric fields whose pandas None -> NaN round-trip must be
    # coerced back to None BEFORE Pydantic's ge/le range checks run. Any new
    # nullable numeric field MUST be added here too.
    _NULLABLE_NUMERIC_WEATHER_FIELDS = (
        "temp_f",
        "temp_c",
        "wind_mph",
        "wind_direction",
        "humidity_pct",
        "precip_prob",
        "precip_mm",
        "visibility_km",
        "condition_code",
        "dew_point_f",
        "apparent_temp_f",
        "snowfall_cm",
        "wind_gusts_mph",
        "cloud_cover_pct",
    )

    game_id: str = Field(..., description="Foreign key to games table")
    forecast_time: datetime = Field(..., description="When forecast was made")
    game_time: datetime = Field(..., description="Game kickoff time")

    # Weather conditions
    temp_f: float | None = Field(None, description="Temperature in Fahrenheit")
    temp_c: float | None = Field(None, description="Temperature in Celsius")
    wind_mph: float | None = Field(None, ge=0, description="Wind speed in MPH")
    wind_direction: float | None = Field(
        None,
        ge=0,
        lt=360,
        description="Wind direction in degrees (meteorological; 0=N, 90=E)",
    )
    humidity_pct: float | None = Field(
        None, ge=0, le=100, description="Humidity percentage"
    )
    precip_prob: float | None = Field(
        None, ge=0, le=1, description="Precipitation probability"
    )
    precip_mm: float | None = Field(
        None, ge=0, description="Precipitation amount in mm"
    )

    # Weather conditions
    condition: str | None = Field(None, description="Weather condition description")
    condition_code: int | None = Field(None, description="Weather condition code")
    visibility_km: float | None = Field(
        None, ge=0, description="Visibility in kilometers"
    )

    # Open-Meteo additional fields (previously emitted by ingest but dropped
    # by Pydantic extra='ignore' default). Declaring them here preserves the
    # values into the Silver parquet.
    dew_point_f: float | None = Field(None, description="Dew point in Fahrenheit")
    apparent_temp_f: float | None = Field(
        None, description="Apparent (feels-like) temperature in Fahrenheit"
    )
    snowfall_cm: float | None = Field(None, ge=0, description="Snowfall in centimeters")
    wind_gusts_mph: float | None = Field(None, ge=0, description="Wind gusts in MPH")
    cloud_cover_pct: float | None = Field(
        None, ge=0, le=100, description="Cloud cover percentage"
    )

    # Derived fields
    is_outdoor: bool = Field(..., description="Whether weather affects the game")

    # DOES THIS ROW CARRY THE WEATHER RECORD IT IS ENTITLED TO (Plan 33.1-02
    # Ruling E). REQUIRED with NO default, mirroring `is_outdoor` directly above:
    # a row that does not say whether it is covered is precisely the state this
    # column exists to make impossible, and a defaulted field would be silently
    # filled rather than refused.
    #
    # True  -- an outdoor game with a real observation, OR an indoor game where
    #          weather does not apply. Both are entitled to what they carry.
    # False -- the venue RESOLVED but no observation exists, and every weather
    #          column on this row is NULL.
    #
    # It composes with `is_outdoor` to recover all three D33.1-07 states:
    # (coverage=1, outdoor=0) is a dome, (coverage=1, outdoor=1) is an
    # observation, (coverage=0, outdoor=1) is an absence. Before this column a
    # dome, a fetch failure and an unmapped stadium were indistinguishable
    # downstream, which is SPEC R4's defect.
    #
    # DECLARED IN THE SAME TASK THAT EMITS IT, deliberately (RESEARCH P-6):
    # `validate_bronze_to_silver` rebuilds every row as
    # `schema_class(**row).model_dump()` (data/quality_gates.py:52-57) and
    # Pydantic v2 defaults to extra='ignore', so an emitted-but-undeclared column
    # disappears between bronze and silver with no error at all. The comment
    # eleven lines above records that this already happened once to the five
    # Open-Meteo fields.
    weather_coverage: bool = Field(
        ...,
        description=(
            "Whether this row carries the weather record it is entitled to: "
            "True for a real observation or an indoor game, False when the venue "
            "resolved but no observation exists (every weather column then NULL)"
        ),
    )

    is_cold: bool | None = Field(None, description="Temperature below 32F")
    is_windy: bool | None = Field(None, description="Wind speed above 12 MPH")
    is_precipitation: bool | None = Field(None, description="Precipitation expected")

    # Data lineage metadata
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when record was created/ingested",
    )

    # WHERE THIS ROW CAME FROM (D33-26). Declared with NO default, so a row that
    # does not say is rejected at validation rather than stored as null: an
    # archive row and a forecast row are otherwise the same columns of equally
    # plausible numbers, and the difference is what a weather-aware re-fit needs.
    # The closed vocabulary lives in scripts/ingest_weather.WEATHER_SOURCE_VOCABULARY
    # and is enforced there by validate_weather_source; this field carries it.
    weather_source: str = Field(
        ...,
        description=(
            "Provenance: 'archive' (ERA5 reanalysis, measured after the fact), "
            "'forecast' (predicted before kickoff), or 'historical_forecast' "
            "(what the forecast said at the time)"
        ),
    )

    @field_validator("weather_source")
    @classmethod
    def validate_weather_source_value(cls, v):
        """Reject a provenance value outside the closed vocabulary, by name.

        Imported inside the validator rather than at module scope: `data.schemas`
        is imported BY `scripts.ingest_weather`, so a top-level import here would
        be circular.
        """
        from scripts.ingest_weather import validate_weather_source

        return validate_weather_source(v)

    @field_validator("temp_f")
    @classmethod
    def validate_temperature_f(cls, v):
        """Validate Fahrenheit temperature is reasonable."""
        if v is not None and not (-50 <= v <= 120):
            raise ValueError("Temperature must be between -50F and 120F")
        return v

    @field_validator("wind_mph")
    @classmethod
    def validate_wind_speed(cls, v):
        """Validate wind speed is reasonable."""
        if v is not None and v > 100:
            raise ValueError("Wind speed cannot exceed 100 MPH")
        return v

    @field_validator(*_NULLABLE_NUMERIC_WEATHER_FIELDS, mode="before")
    @classmethod
    def nan_to_none(cls, v):
        """Pandas NaN/pd.NA -> None for nullable numeric fields.

        The DataFrame -> to_dict round-trip in validate_bronze_to_silver converts
        Python None into float NaN for numeric columns; this validator maps NaN
        back to None before Pydantic's ge/le range checks or the after-mode
        validators (validate_temperature_f, validate_wind_speed) reject NaN.
        Mirrors the pattern in GameSchema.validate_scores.
        """
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (ValueError, TypeError):
            # pd.isna raises on non-scalar inputs; defensive fallthrough matches GameSchema.
            pass
        return v

    @field_validator("forecast_time", "game_time", "created_at", mode="before")
    @classmethod
    def validate_timestamps(cls, v):
        """Ensure timestamps are timezone-aware."""
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except (ValueError, TypeError):
            pass
        if isinstance(v, str):
            return pd.to_datetime(v)
        if isinstance(v, datetime) and v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v


class TeamFormSchema(BaseModel):
    """Schema for team form/performance metrics (silver layer)."""

    model_config = ConfigDict(use_enum_values=True)

    team_id: str = Field(..., description="Team abbreviation")
    season: int = Field(..., ge=2000, le=2030, description="NFL season")
    week: int = Field(..., ge=1, le=22, description="Week number (as of)")
    games_played: int = Field(..., ge=0, description="Games played up to this week")

    # Offensive metrics (rolling 4-week)
    epa_off_l4: float | None = Field(
        None, description="Offensive EPA/play (last 4 games)"
    )
    epa_off_pass_l4: float | None = Field(
        None, description="Passing EPA/play (last 4 games)"
    )
    epa_off_rush_l4: float | None = Field(
        None, description="Rushing EPA/play (last 4 games)"
    )
    succ_off_l4: float | None = Field(
        None, ge=0, le=1, description="Offensive success rate"
    )

    # Defensive metrics (rolling 4-week)
    epa_def_l4: float | None = Field(None, description="Defensive EPA/play allowed")
    epa_def_pass_l4: float | None = Field(None, description="Passing EPA/play allowed")
    epa_def_rush_l4: float | None = Field(None, description="Rushing EPA/play allowed")
    succ_def_l4: float | None = Field(
        None, ge=0, le=1, description="Defensive success rate"
    )

    # Situational metrics
    neutral_pass_rate_l4: float | None = Field(
        None, ge=0, le=1, description="Neutral situation pass rate"
    )
    red_zone_eff_l4: float | None = Field(
        None, ge=0, le=1, description="Red zone efficiency"
    )
    third_down_conv_l4: float | None = Field(
        None, ge=0, le=1, description="3rd down conversion rate"
    )

    # Game context
    rest_days: int | None = Field(None, ge=0, description="Days since last game")
    is_home: bool | None = Field(None, description="Playing at home this week")

    # Season-to-date metrics
    wins: int = Field(..., ge=0, description="Wins this season")
    losses: int = Field(..., ge=0, description="Losses this season")
    ties: int = Field(0, ge=0, description="Ties this season")

    @field_validator("team_id")
    @classmethod
    def validate_team_id(cls, v: str) -> str:
        """Validate team ID format."""
        return v.upper().strip()


class VenueSchema(BaseModel):
    """Schema for venue/stadium data (static reference)."""

    model_config = ConfigDict(use_enum_values=True)

    venue_id: str = Field(..., description="Unique venue identifier")
    venue_name: str = Field(..., description="Official venue name")
    city: str = Field(..., description="City location")
    state: str = Field(..., description="State/province")
    country: str = Field(default="USA", description="Country")

    # Geographic coordinates
    latitude: float = Field(..., ge=-90, le=90, description="Latitude coordinate")
    longitude: float = Field(..., ge=-180, le=180, description="Longitude coordinate")
    elevation_ft: int | None = Field(None, description="Elevation in feet")

    # Venue characteristics
    roof_type: VenueRoof = Field(..., description="Roof type")
    surface: str | None = Field(None, description="Playing surface type")
    capacity: int | None = Field(None, ge=0, description="Seating capacity")

    # Climate zone for weather modeling
    climate_zone: str | None = Field(None, description="Climate classification")
    timezone: str = Field(..., description="Local timezone")

    # Teams that play here
    home_teams: list[str] = Field(..., description="Teams that call this venue home")


class FeatureMatrixSchema(BaseModel):
    """Schema for feature matrices (gold layer)."""

    model_config = ConfigDict(use_enum_values=True)

    game_id: str = Field(..., description="Game identifier")
    season: int = Field(..., description="Season")
    week: int = Field(..., description="Week")
    feature_timestamp: datetime = Field(..., description="When features were generated")

    # Target variables
    target_wp: int | None = Field(
        None, description="Win probability target (1=home win)"
    )
    target_ats: float | None = Field(
        None, description="ATS target (home margin - spread)"
    )
    target_ou: float | None = Field(None, description="O/U target (total points)")

    # Features will be added dynamically based on feature engineering
    features: dict[str, float] = Field(..., description="Feature values")

    @field_validator("features")
    @classmethod
    def validate_features_not_empty(cls, v):
        """Ensure features dict is not empty."""
        if not v:
            raise ValueError("Features dictionary cannot be empty")
        return v


class PredictionSchema(BaseModel):
    """Schema for model predictions (outputs)."""

    model_config = ConfigDict(use_enum_values=True)

    game_id: str = Field(..., description="Game identifier")
    season: int = Field(..., description="Season")
    week: int = Field(..., description="Week")
    model_timestamp: datetime = Field(..., description="When prediction was made")
    model_version: str = Field(..., description="Model version identifier")

    # Win probability predictions
    wp_home: float = Field(..., ge=0, le=1, description="Home team win probability")
    wp_away: float = Field(..., ge=0, le=1, description="Away team win probability")

    # Against the spread predictions
    ats_home_prob: float = Field(
        ..., ge=0, le=1, description="Home team cover probability"
    )
    ats_away_prob: float = Field(
        ..., ge=0, le=1, description="Away team cover probability"
    )
    fair_spread: float | None = Field(None, description="Model's fair spread")

    # Over/Under predictions
    over_prob: float = Field(..., ge=0, le=1, description="Over probability")
    under_prob: float = Field(..., ge=0, le=1, description="Under probability")
    fair_total: float | None = Field(None, description="Model's fair total")

    # Market edges (if odds available)
    edge_ml_home: float | None = Field(None, description="Moneyline edge (home)")
    edge_ml_away: float | None = Field(None, description="Moneyline edge (away)")
    edge_spread_home: float | None = Field(None, description="Spread edge (home)")
    edge_spread_away: float | None = Field(None, description="Spread edge (away)")
    edge_over: float | None = Field(None, description="Over edge")
    edge_under: float | None = Field(None, description="Under edge")

    # Model confidence/uncertainty
    confidence_wp: float | None = Field(
        None, ge=0, le=1, description="WP prediction confidence"
    )
    confidence_ats: float | None = Field(
        None, ge=0, le=1, description="ATS prediction confidence"
    )
    confidence_ou: float | None = Field(
        None, ge=0, le=1, description="O/U prediction confidence"
    )

    @model_validator(mode="after")
    def validate_probability_sums(self):
        """Validate that paired probabilities sum to approximately 1."""
        pairs = [
            ("wp_home", "wp_away", self.wp_home, self.wp_away),
            ("ats_home_prob", "ats_away_prob", self.ats_home_prob, self.ats_away_prob),
            ("over_prob", "under_prob", self.over_prob, self.under_prob),
        ]
        for name_a, name_b, val_a, val_b in pairs:
            total = val_a + val_b
            if abs(total - 1.0) > 0.01:
                raise ValueError(
                    f"{name_a} + {name_b} = {total:.4f}, must sum to approximately 1.0"
                )
        return self


# Utility functions for schema validation


def validate_dataframe_schema(df: pd.DataFrame, schema_class: type) -> list[str]:
    """
    Validate DataFrame against Pydantic schema.

    Args:
        df: DataFrame to validate
        schema_class: Pydantic model class

    Returns:
        List of validation errors
    """
    errors = []

    for idx, row in df.iterrows():
        try:
            schema_class(**row.to_dict())
        except (ValueError, TypeError) as e:
            errors.append(f"Row {idx}: {e!s}")

    return errors


def convert_df_to_schema(df: pd.DataFrame, schema_class: type) -> list[BaseModel]:
    """
    Convert DataFrame to list of schema objects.

    Args:
        df: DataFrame to convert
        schema_class: Pydantic model class

    Returns:
        List of validated schema objects
    """
    return [schema_class(**row.to_dict()) for _, row in df.iterrows()]


def schema_to_df(schema_objects: list[BaseModel]) -> pd.DataFrame:
    """
    Convert list of schema objects to DataFrame.

    Args:
        schema_objects: List of Pydantic model instances

    Returns:
        DataFrame
    """
    return pd.DataFrame([obj.model_dump() for obj in schema_objects])
