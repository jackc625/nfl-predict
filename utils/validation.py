"""Data validation utilities for NFL prediction system."""

from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel, ValidationError, validator

# The zone the NFL season calendar is published in. Season bounds are calendar DATES, so a
# tz-aware kickoff is compared against those dates in this zone rather than against naive
# timestamps, which pandas refuses to compare with an aware one.
_NFL_CALENDAR_TZ = "America/New_York"


class GameData(BaseModel):
    """Game data validation schema."""

    game_id: str
    season: int
    week: int
    kickoff_et: datetime
    home_team: str
    away_team: str
    venue: str
    venue_roof: str
    home_score: int | None = None
    away_score: int | None = None
    result: int | None = None

    @validator("season")
    def validate_season(cls, v):
        if not (2000 <= v <= 2030):
            raise ValueError("Season must be between 2000 and 2030")
        return v

    @validator("week")
    def validate_week(cls, v):
        if not (1 <= v <= 22):
            raise ValueError("Week must be between 1 and 22")
        return v

    @validator("venue_roof")
    def validate_venue_roof(cls, v):
        valid_roofs = ["indoor", "outdoor", "retractable"]
        if v.lower() not in valid_roofs:
            raise ValueError(f"Venue roof must be one of: {valid_roofs}")
        return v.lower()

    @validator("home_team", "away_team")
    def validate_team_names(cls, v):
        if len(v) < 2 or len(v) > 5:
            raise ValueError("Team names must be 2-5 characters")
        return v.upper()

    @validator("result")
    def validate_result(cls, v):
        if v is not None and v not in [-1, 0, 1]:
            raise ValueError("Result must be -1 (away win), 0 (tie), or 1 (home win)")
        return v


class OddsData(BaseModel):
    """Odds data validation schema."""

    game_id: str
    snapshot_ts: datetime
    sportsbook: str
    ml_home: int | None = None
    ml_away: int | None = None
    spread: float | None = None
    spread_ju_home: int | None = None
    spread_ju_away: int | None = None
    total: float | None = None
    total_over_ju: int | None = None
    total_under_ju: int | None = None
    is_live: bool | None = None
    last_update: datetime | None = None

    @validator(
        "ml_home",
        "ml_away",
        "spread_ju_home",
        "spread_ju_away",
        "total_over_ju",
        "total_under_ju",
    )
    def validate_moneylines(cls, v):
        if v is not None and not (-10000 <= v <= 10000):
            raise ValueError("Moneyline/juice values must be between -10000 and +10000")
        return v

    @validator("spread")
    def validate_spread(cls, v):
        if v is not None and not (-50.0 <= v <= 50.0):
            raise ValueError("Spread must be between -50.0 and +50.0")
        return v

    @validator("total")
    def validate_total(cls, v):
        if v is not None and not (10.0 <= v <= 100.0):
            raise ValueError("Total must be between 10.0 and 100.0")
        return v


class PredictionData(BaseModel):
    """Prediction data validation schema."""

    game_id: str
    season: int
    week: int
    model_timestamp: datetime
    wp_home: float
    wp_away: float
    ats_home_prob: float
    ats_away_prob: float
    over_prob: float
    under_prob: float
    fair_spread: float | None = None
    fair_total: float | None = None
    edge_ml_home: float | None = None
    edge_ml_away: float | None = None
    edge_spread_home: float | None = None
    edge_spread_away: float | None = None
    edge_over: float | None = None
    edge_under: float | None = None

    @validator(
        "wp_home",
        "wp_away",
        "ats_home_prob",
        "ats_away_prob",
        "over_prob",
        "under_prob",
    )
    def validate_probabilities(cls, v):
        if not (0.0 <= v <= 1.0):
            raise ValueError("Probabilities must be between 0.0 and 1.0")
        return v

    @validator("fair_spread")
    def validate_fair_spread(cls, v):
        if v is not None and not (-50.0 <= v <= 50.0):
            raise ValueError("Fair spread must be between -50.0 and +50.0")
        return v

    @validator("fair_total")
    def validate_fair_total(cls, v):
        if v is not None and not (10.0 <= v <= 100.0):
            raise ValueError("Fair total must be between 10.0 and 100.0")
        return v


def validate_game_data(data: dict | pd.DataFrame | list[dict]) -> list[str]:
    """
    Validate game data against schema.

    Args:
        data: Game data to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if isinstance(data, pd.DataFrame):
        data = data.to_dict("records")
    elif isinstance(data, dict):
        data = [data]

    for i, record in enumerate(data):
        try:
            GameData(**record)
        except ValidationError as e:
            for error in e.errors():
                field = error["loc"][0] if error["loc"] else "unknown"
                msg = error["msg"]
                errors.append(f"Row {i}: {field} - {msg}")

    return errors


def validate_odds_data(data: dict | pd.DataFrame | list[dict]) -> list[str]:
    """
    Validate odds data against schema.

    Args:
        data: Odds data to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    import pandas as pd

    errors = []

    if isinstance(data, pd.DataFrame):
        data = data.to_dict("records")
    elif isinstance(data, dict):
        data = [data]

    for i, record in enumerate(data):
        try:
            # Convert NaN values to None for proper validation
            cleaned_record = {}
            for key, value in record.items():
                if pd.isna(value):
                    cleaned_record[key] = None
                else:
                    cleaned_record[key] = value

            OddsData(**cleaned_record)
        except ValidationError as e:
            for error in e.errors():
                field = error["loc"][0] if error["loc"] else "unknown"
                msg = error["msg"]
                errors.append(f"Row {i}: {field} - {msg}")

    return errors


def validate_prediction_data(data: dict | pd.DataFrame | list[dict]) -> list[str]:
    """
    Validate prediction data against schema.

    Args:
        data: Prediction data to validate

    Returns:
        List of validation error messages (empty if valid)
    """
    errors = []

    if isinstance(data, pd.DataFrame):
        data = data.to_dict("records")
    elif isinstance(data, dict):
        data = [data]

    for i, record in enumerate(data):
        try:
            PredictionData(**record)
        except ValidationError as e:
            for error in e.errors():
                field = error["loc"][0] if error["loc"] else "unknown"
                msg = error["msg"]
                errors.append(f"Row {i}: {field} - {msg}")

    return errors


def validate_data_quality(
    df: pd.DataFrame,
    table_name: str,
    expected_columns: list[str] | None = None,
    min_rows: int = 1,
    max_missing_pct: float = 0.1,
) -> dict[str, Any]:
    """
    Perform general data quality validation.

    Args:
        df: DataFrame to validate
        table_name: Name of the table/dataset
        expected_columns: Expected column names
        min_rows: Minimum number of rows required
        max_missing_pct: Maximum percentage of missing values allowed

    Returns:
        Dictionary with validation results
    """
    validation_results = {
        "table_name": table_name,
        "is_valid": True,
        "row_count": len(df),
        "column_count": len(df.columns),
        "errors": [],
        "warnings": [],
    }

    # Check minimum rows
    if len(df) < min_rows:
        validation_results["errors"].append(
            f"Insufficient rows: {len(df)} < {min_rows}"
        )
        validation_results["is_valid"] = False

    # Check expected columns
    if expected_columns:
        missing_cols = set(expected_columns) - set(df.columns)
        extra_cols = set(df.columns) - set(expected_columns)

        if missing_cols:
            validation_results["errors"].append(
                f"Missing columns: {list(missing_cols)}"
            )
            validation_results["is_valid"] = False

        if extra_cols:
            validation_results["warnings"].append(
                f"Unexpected columns: {list(extra_cols)}"
            )

    # Check for excessive missing values
    if not df.empty:
        missing_pcts = df.isnull().mean()
        problematic_cols = missing_pcts[missing_pcts > max_missing_pct]

        if not problematic_cols.empty:
            validation_results["warnings"].append(
                f"High missing value percentage in columns: "
                f"{dict(problematic_cols.round(3))}"
            )

    # Check for duplicate rows (skip if columns contain unhashable types like numpy arrays)
    try:
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            validation_results["warnings"].append(
                f"Found {duplicate_count} duplicate rows"
            )
    except TypeError as e:
        if "unhashable type" in str(e):
            validation_results["warnings"].append(
                "Could not check for duplicate rows due to unhashable data types (e.g., numpy arrays)"
            )
        else:
            raise

    # Check data types for common issues
    for col in df.columns:
        if df[col].dtype == "object":
            # Check for mixed types in object columns
            non_null_values = df[col].dropna()
            if not non_null_values.empty:
                try:
                    types = {type(x).__name__ for x in non_null_values.head(100)}
                    if len(types) > 1:
                        validation_results["warnings"].append(
                            f"Mixed data types in column '{col}': {list(types)}"
                        )
                except TypeError:
                    # Handle unhashable types (like numpy arrays)
                    type_names = [type(x).__name__ for x in non_null_values.head(100)]
                    unique_types = list(
                        dict.fromkeys(type_names)
                    )  # Preserve order, remove duplicates
                    if len(unique_types) > 1:
                        validation_results["warnings"].append(
                            f"Mixed data types in column '{col}': {unique_types}"
                        )

    return validation_results


def validate_nfl_business_rules(df: pd.DataFrame, table_type: str) -> list[str]:
    """
    Validate NFL-specific business rules.

    Args:
        df: DataFrame to validate
        table_type: Type of table ('games', 'odds', 'predictions')

    Returns:
        List of business rule violations
    """
    violations = []

    if table_type == "games":
        # Check for valid NFL seasons
        if "season" in df.columns:
            invalid_seasons = df[~df["season"].between(2000, 2030)]
            if not invalid_seasons.empty:
                violations.append(
                    f"Invalid seasons found: {invalid_seasons['season'].unique()}"
                )

        # Check for valid weeks
        if "week" in df.columns:
            invalid_weeks = df[~df["week"].between(1, 22)]
            if not invalid_weeks.empty:
                violations.append(
                    f"Invalid weeks found: {invalid_weeks['week'].unique()}"
                )

        # Check that home and away teams are different
        if "home_team" in df.columns and "away_team" in df.columns:
            same_teams = df[df["home_team"] == df["away_team"]]
            if not same_teams.empty:
                violations.append(
                    f"Found {len(same_teams)} games where home and away teams are the same"
                )

    elif table_type == "odds":
        # Check spread reasonableness
        if "spread" in df.columns:
            extreme_spreads = df[df["spread"].abs() > 30]
            if not extreme_spreads.empty:
                violations.append(
                    f"Found {len(extreme_spreads)} games with extreme spreads (>30 points)"
                )

        # Check total reasonableness
        if "total" in df.columns:
            extreme_totals = df[(df["total"] < 30) | (df["total"] > 70)]
            if not extreme_totals.empty:
                violations.append(
                    f"Found {len(extreme_totals)} games with extreme totals (<30 or >70)"
                )

    elif table_type == "predictions":
        # Check probability constraints
        prob_cols = [
            col for col in df.columns if "prob" in col.lower() or col.startswith("wp_")
        ]
        for col in prob_cols:
            if col in df.columns:
                invalid_probs = df[(df[col] < 0) | (df[col] > 1)]
                if not invalid_probs.empty:
                    violations.append(
                        f"Found {len(invalid_probs)} invalid probabilities in column '{col}'"
                    )

        # Check that complementary probabilities sum to ~1
        if "wp_home" in df.columns and "wp_away" in df.columns:
            prob_sums = df["wp_home"] + df["wp_away"]
            bad_sums = df[prob_sums.abs() - 1 > 0.01]  # Allow 1% tolerance
            if not bad_sums.empty:
                violations.append(
                    f"Found {len(bad_sums)} games where WP probabilities don't sum to 1"
                )

    return violations


def validate_temporal_consistency(
    df: pd.DataFrame,
    date_col: str = "kickoff_et",
    season_col: str = "season",
    week_col: str = "week",
) -> list[str]:
    """
    Validate temporal consistency of NFL data.

    Args:
        df: DataFrame with temporal data
        date_col: Column with game dates
        season_col: Column with season years
        week_col: Column with week numbers

    Returns:
        List of temporal consistency violations
    """
    violations = []

    if not all(col in df.columns for col in [date_col, season_col, week_col]):
        return ["Required temporal columns missing"]

    # Check that dates align with seasons
    for _, row in df.iterrows():
        game_date = pd.to_datetime(row[date_col])
        season = row[season_col]

        # NFL season runs from September of season to February of (season+1)
        season_start = pd.Timestamp(f"{season}-09-01")
        season_end = pd.Timestamp(f"{season + 1}-03-01")
        # COMPARE LIKE WITH LIKE (Plan 33-18, owner ruling R1 of 2026-09-15). The bounds are
        # calendar dates, and the stored ``kickoff_et`` is tz-AWARE, so comparing the two
        # raised "Cannot compare tz-naive and tz-aware timestamps" and the games temporal
        # check never ran at all. An aware kickoff is judged against the SAME calendar dates
        # in Eastern time -- the zone the NFL calendar is published in. A naive kickoff keeps
        # its naive bounds, exactly as before. The data is not touched; only the comparison.
        if game_date.tzinfo is not None:
            season_start = season_start.tz_localize(_NFL_CALENDAR_TZ)
            season_end = season_end.tz_localize(_NFL_CALENDAR_TZ)

        if not (season_start <= game_date <= season_end):
            violations.append(
                f"Game date {game_date} inconsistent with season {season}"
            )

    # Check week ordering within seasons
    for season in df[season_col].unique():
        season_data = df[df[season_col] == season].sort_values(week_col)
        if not season_data.empty:
            dates = pd.to_datetime(season_data[date_col])
            weeks = season_data[week_col]

            # Check that weeks generally increase with dates
            for i in range(1, len(weeks)):
                if (
                    weeks.iloc[i] < weeks.iloc[i - 1]
                    and dates.iloc[i] > dates.iloc[i - 1]
                ):
                    violations.append(
                        f"Week ordering inconsistent in season {season}: "
                        f"week {weeks.iloc[i]} after week {weeks.iloc[i - 1]} "
                        f"but date is later"
                    )

    return violations
