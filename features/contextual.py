"""
Contextual Features Calculator

This module calculates contextual features for NFL games:
- Travel time-zone difference calculations
- Short week detection
- Venue roof type encoding
- Home/away team indicators
- Rest days calculations

These features capture situational factors that may impact game outcomes
beyond pure team performance metrics.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from conf.settings import get_settings
from data.storage import load_dataframe
from utils import get_logger

logger = get_logger(__name__)


class ContextualFeaturesCalculator:
    """
    Calculate contextual features for NFL games.

    Features calculated:
    - Travel distance and time zone differences
    - Short week indicators (Thursday games, etc.)
    - Venue roof type encoding (indoor/outdoor/retractable)
    - Home field advantage indicators
    - Rest days since last game
    - Travel fatigue metrics
    """

    def __init__(self):
        """Initialize contextual features calculator."""
        self.settings = get_settings()

        # Load venue data
        self.venues_data = self._load_venues_data()

        # Time zone mappings
        self.timezone_map = self._build_timezone_map()

        # Team to venue mapping
        self.team_venues = self._build_team_venue_mapping()

    def _load_venues_data(self) -> dict[str, Any]:
        """Load venue data from JSON file."""
        try:
            # Use current working directory as project root
            project_root = Path.cwd()
            venues_path = project_root / "data" / "venues.json"

            with open(venues_path) as f:
                venues_data = json.load(f)

            logger.info("Loaded venues data", venues_count=len(venues_data["venues"]))
            return venues_data

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error("Failed to load venues data", error=str(e))
            raise

    def _build_timezone_map(self) -> dict[str, str]:
        """Build mapping of venue IDs to timezones."""
        timezone_map = {}

        for venue in self.venues_data["venues"]:
            timezone_map[venue["venue_id"]] = venue["timezone"]

        return timezone_map

    def _build_team_venue_mapping(self) -> dict[str, str]:
        """Build mapping of teams to their home venue IDs."""
        team_venues = {}

        for venue in self.venues_data["venues"]:
            for team in venue["home_teams"]:
                team_venues[team] = venue["venue_id"]

        return team_venues

    def _get_venue_by_id(self, venue_id: str) -> dict[str, Any] | None:
        """Get venue information by venue ID."""
        for venue in self.venues_data["venues"]:
            if venue["venue_id"] == venue_id:
                return venue
        return None

    def _get_venue_id_by_name(self, venue_name: str) -> str:
        """Map venue name to venue ID."""
        if not venue_name:
            return ""

        # Try exact match first
        for venue in self.venues_data["venues"]:
            if venue.get("venue_name") == venue_name:
                return venue["venue_id"]

        # Try partial match (handle name variations)
        venue_name_lower = venue_name.lower()
        for venue in self.venues_data["venues"]:
            venue_name_in_data = venue.get("venue_name", "").lower()
            if (
                venue_name_lower in venue_name_in_data
                or venue_name_in_data in venue_name_lower
            ):
                return venue["venue_id"]

        # No match found - create a fallback ID
        logger.warning("Could not map venue name to ID", venue_name=venue_name)
        return venue_name.lower().replace(" ", "_").replace("&", "and")

    def calculate_travel_metrics(
        self,
        away_team: str,
        home_team: str,
        game_venue_id: str,
        kickoff_datetime: datetime,
    ) -> dict[str, float]:
        """
        Calculate travel-related metrics for the away team.

        Args:
            away_team: Away team abbreviation
            home_team: Home team abbreviation
            game_venue_id: Venue ID where game is played
            kickoff_datetime: Game kickoff time (timezone-aware)

        Returns:
            Dictionary with travel metrics
        """
        try:
            # Get away team's home venue
            away_home_venue_id = self.team_venues.get(away_team)
            if not away_home_venue_id:
                logger.warning("No home venue found for away team", away_team=away_team)
                return self._default_travel_metrics()

            # Get venue information
            away_venue = self._get_venue_by_id(away_home_venue_id)
            game_venue = self._get_venue_by_id(game_venue_id)

            if not away_venue or not game_venue:
                logger.warning(
                    "Venue data not found",
                    away_venue_id=away_home_venue_id,
                    game_venue_id=game_venue_id,
                )
                return self._default_travel_metrics()

            # Calculate distance using Haversine formula
            travel_distance = self._calculate_distance(
                away_venue["latitude"],
                away_venue["longitude"],
                game_venue["latitude"],
                game_venue["longitude"],
            )

            # Calculate timezone difference
            away_tz = ZoneInfo(away_venue["timezone"])
            game_tz = ZoneInfo(game_venue["timezone"])

            # Convert kickoff to both timezones to calculate difference
            kickoff_away_tz = kickoff_datetime.astimezone(away_tz)
            kickoff_game_tz = kickoff_datetime.astimezone(game_tz)

            # Time zone difference in hours (positive = westward travel)
            tz_diff_hours = (
                kickoff_away_tz.utcoffset() - kickoff_game_tz.utcoffset()
            ).total_seconds() / 3600

            # Travel fatigue score (combination of distance and timezone change)
            # Higher score = more fatigue
            distance_factor = min(
                travel_distance / 2500, 1.0
            )  # Normalize to max distance ~2500 miles
            timezone_factor = (
                abs(tz_diff_hours) / 3.0
            )  # Normalize to max 3 hour difference
            travel_fatigue_score = (distance_factor * 0.6) + (timezone_factor * 0.4)

            # Cross-country travel indicator (>= 2000 miles or >= 2 time zones)
            cross_country_travel = (travel_distance >= 2000) or (
                abs(tz_diff_hours) >= 2
            )

            return {
                "travel_distance_miles": travel_distance,
                "timezone_diff_hours": tz_diff_hours,
                "abs_timezone_diff_hours": abs(tz_diff_hours),
                "travel_fatigue_score": travel_fatigue_score,
                "cross_country_travel": 1.0 if cross_country_travel else 0.0,
                "eastward_travel": 1.0 if tz_diff_hours < 0 else 0.0,
                "westward_travel": 1.0 if tz_diff_hours > 0 else 0.0,
            }

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to calculate travel metrics",
                away_team=away_team,
                game_venue_id=game_venue_id,
                error=str(e),
            )
            return self._default_travel_metrics()

    def _default_travel_metrics(self) -> dict[str, float]:
        """Return default travel metrics when calculation fails."""
        return {
            "travel_distance_miles": 0.0,
            "timezone_diff_hours": 0.0,
            "abs_timezone_diff_hours": 0.0,
            "travel_fatigue_score": 0.0,
            "cross_country_travel": 0.0,
            "eastward_travel": 0.0,
            "westward_travel": 0.0,
        }

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula.

        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point

        Returns:
            Distance in miles
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in miles
        earth_radius_miles = 3959.0

        return earth_radius_miles * c

    def detect_short_week(
        self, kickoff_datetime: datetime, season: int, week: int
    ) -> dict[str, float]:
        """
        Detect short week scenarios (Thursday night games, etc.).

        Args:
            kickoff_datetime: Game kickoff time
            season: Season year
            week: Week number

        Returns:
            Dictionary with short week indicators
        """
        try:
            # Get day of week (0=Monday, 6=Sunday)
            game_day = kickoff_datetime.weekday()

            # Thursday games (weekday 3)
            thursday_game = 1.0 if game_day == 3 else 0.0

            # Monday games (weekday 0)
            monday_game = 1.0 if game_day == 0 else 0.0

            # Saturday games (weekday 5) - typically late season/playoffs
            saturday_game = 1.0 if game_day == 5 else 0.0

            # Short week indicator (Thursday or Monday)
            short_week = 1.0 if (game_day in {3, 0}) else 0.0

            # Rest advantage for teams coming off bye week
            # This will be calculated separately when we have game-by-game data

            return {
                "thursday_game": thursday_game,
                "monday_game": monday_game,
                "saturday_game": saturday_game,
                "short_week": short_week,
                "game_day_of_week": float(game_day),
            }

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to detect short week",
                kickoff_datetime=kickoff_datetime,
                error=str(e),
            )
            return {
                "thursday_game": 0.0,
                "monday_game": 0.0,
                "saturday_game": 0.0,
                "short_week": 0.0,
                "game_day_of_week": 6.0,  # Default to Sunday
            }

    def encode_venue_features(self, venue_id: str) -> dict[str, float]:
        """
        Encode venue-specific features.

        Args:
            venue_id: Venue identifier

        Returns:
            Dictionary with venue features
        """
        try:
            venue = self._get_venue_by_id(venue_id)
            if not venue:
                logger.warning("Venue not found", venue_id=venue_id)
                return self._default_venue_features()

            # Roof type encoding
            roof_type = (venue.get("roof_type") or "outdoor").lower()
            outdoor = 1.0 if roof_type == "outdoor" else 0.0
            indoor = 1.0 if roof_type == "indoor" else 0.0
            retractable = 1.0 if roof_type == "retractable" else 0.0

            # Elevation (affects kicking, oxygen levels)
            elevation_ft = venue["elevation_ft"]
            high_altitude = 1.0 if elevation_ft >= 3000 else 0.0  # Denver is ~5280 ft

            # Climate zone encoding
            climate = venue.get("climate_zone", "unknown").lower()
            cold_climate = 1.0 if climate in ["humid_continental"] else 0.0
            warm_climate = (
                1.0 if climate in ["humid_subtropical", "tropical", "desert"] else 0.0
            )

            # Stadium capacity (crowd noise factor)
            capacity = venue.get("capacity", 70000)
            large_stadium = 1.0 if capacity >= 75000 else 0.0

            return {
                "venue_outdoor": outdoor,
                "venue_indoor": indoor,
                "venue_retractable": retractable,
                "venue_elevation_ft": float(elevation_ft),
                "venue_high_altitude": high_altitude,
                "venue_cold_climate": cold_climate,
                "venue_warm_climate": warm_climate,
                "venue_capacity": float(capacity),
                "venue_large_stadium": large_stadium,
            }

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to encode venue features", venue_id=venue_id, error=str(e)
            )
            return self._default_venue_features()

    def _default_venue_features(self) -> dict[str, float]:
        """Return default venue features when encoding fails."""
        return {
            "venue_outdoor": 1.0,  # Default to outdoor
            "venue_indoor": 0.0,
            "venue_retractable": 0.0,
            "venue_elevation_ft": 500.0,  # Average elevation
            "venue_high_altitude": 0.0,
            "venue_cold_climate": 0.0,
            "venue_warm_climate": 0.0,
            "venue_capacity": 70000.0,  # Average capacity
            "venue_large_stadium": 0.0,
        }

    def calculate_rest_days(
        self, team: str, current_game_date: datetime, games_df: pd.DataFrame
    ) -> float:
        """
        Calculate rest days for a team since their last game.

        Args:
            team: Team abbreviation
            current_game_date: Date of current game
            games_df: DataFrame with all games

        Returns:
            Number of rest days
        """
        try:
            # Find team's previous games before current date
            team_games = games_df[
                ((games_df["home_team"] == team) | (games_df["away_team"] == team))
                & (games_df["kickoff_et"] < current_game_date)
            ].sort_values("kickoff_et")

            if len(team_games) == 0:
                # First game of season, use standard rest
                return 7.0

            # Get most recent game
            last_game = team_games.iloc[-1]
            last_game_date = last_game["kickoff_et"]

            # Calculate rest days
            rest_days = (current_game_date - last_game_date).days

            return float(rest_days)

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error("Failed to calculate rest days", team=team, error=str(e))
            return 7.0  # Default to standard week

    def build_contextual_features(
        self,
        games_df: pd.DataFrame,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """
        Build contextual features for all games (legacy interface).

        .. deprecated::
            Use :meth:`build_features` instead (conforms to FeatureBuilder Protocol).

        Args:
            games_df: DataFrame with game information
            target_season: Specific season to calculate features for
            target_week: Specific week to calculate features for

        Returns:
            DataFrame with contextual features added
        """
        warnings.warn(
            "build_contextual_features is deprecated; use build_features instead",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(
            "Building contextual features",
            games=len(games_df),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Filter to target if specified
            if target_season and target_week:
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

            contextual_features = []

            for _, game in games_df.iterrows():
                game_id = game["game_id"]
                home_team = game["home_team"]
                away_team = game["away_team"]
                # Map venue name to venue_id
                venue_name = game.get("venue", "")
                venue_id = self._get_venue_id_by_name(venue_name)
                kickoff_dt = game["kickoff_et"]
                season = game["season"]
                week = game["week"]

                # Basic game identifiers
                game_features = {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                }

                # Home/away indicators
                game_features.update(
                    {
                        "is_home_game": 1.0,  # For home team perspective
                        "is_away_game": 0.0,  # For home team perspective
                    }
                )

                # Travel metrics (for away team)
                travel_metrics = self.calculate_travel_metrics(
                    away_team, home_team, venue_id, kickoff_dt
                )

                # Add prefix to indicate these are for away team
                for key, value in travel_metrics.items():
                    game_features[f"away_{key}"] = value

                # Short week detection
                short_week_features = self.detect_short_week(kickoff_dt, season, week)
                game_features.update(short_week_features)

                # Venue features
                venue_features = self.encode_venue_features(venue_id)
                game_features.update(venue_features)

                # Rest days (calculate for both teams)
                home_rest_days = self.calculate_rest_days(
                    home_team, kickoff_dt, games_df
                )
                away_rest_days = self.calculate_rest_days(
                    away_team, kickoff_dt, games_df
                )

                game_features.update(
                    {
                        "home_rest_days": home_rest_days,
                        "away_rest_days": away_rest_days,
                        "rest_advantage": home_rest_days
                        - away_rest_days,  # Positive = home team more rested
                        "both_short_rest": 1.0
                        if (home_rest_days <= 4 and away_rest_days <= 4)
                        else 0.0,
                        "home_short_rest": 1.0 if home_rest_days <= 4 else 0.0,
                        "away_short_rest": 1.0 if away_rest_days <= 4 else 0.0,
                    }
                )

                contextual_features.append(game_features)

            # Convert to DataFrame
            features_df = pd.DataFrame(contextual_features)

            logger.info(
                "Built contextual features",
                features_count=len(features_df),
                feature_columns=len(features_df.columns),
            )

            return features_df

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error("Failed to build contextual features", error=str(e))
            raise

    def get_contextual_features_for_game(
        self, game_id: str, season: int, week: int
    ) -> dict[str, float]:
        """
        Get contextual features for a specific game.

        Args:
            game_id: Game identifier
            season: Season year
            week: Week number

        Returns:
            Dictionary with contextual features
        """
        try:
            # Load contextual features
            features_df = load_dataframe("contextual_features", layer="silver")

            # Filter to specific game
            game_features = features_df[
                (features_df["game_id"] == game_id)
                & (features_df["season"] == season)
                & (features_df["week"] == week)
            ]

            if len(game_features) == 0:
                logger.warning(
                    "No contextual features found for game",
                    game_id=game_id,
                    season=season,
                    week=week,
                )
                return {}

            # Convert to dictionary, excluding non-feature columns
            exclude_cols = ["game_id", "season", "week", "home_team", "away_team"]
            features_dict = {}

            game_row = game_features.iloc[0]
            for col in game_features.columns:
                if col not in exclude_cols:
                    features_dict[col] = game_row[col]

            return features_dict

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to get contextual features for game",
                game_id=game_id,
                season=season,
                week=week,
                error=str(e),
            )
            return {}

    def validate_contextual_features(self, features_df: pd.DataFrame) -> bool:
        """
        Validate contextual features for data quality.

        Args:
            features_df: Contextual features DataFrame

        Returns:
            True if validation passes
        """
        if len(features_df) == 0:
            logger.error("No contextual features found")
            return False

        # Check for required columns
        required_cols = ["game_id", "season", "week", "home_team", "away_team"]
        missing_cols = set(required_cols) - set(features_df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing_columns=list(missing_cols))
            return False

        # Check travel distance ranges (should be 0-3000 miles)
        if "away_travel_distance_miles" in features_df.columns:
            distances = features_df["away_travel_distance_miles"].dropna()
            if len(distances) > 0:
                if distances.min() < 0 or distances.max() > 5000:
                    logger.warning(
                        "Travel distances outside reasonable range",
                        min_distance=distances.min(),
                        max_distance=distances.max(),
                    )

        # Check timezone differences (should be -3 to +3 hours)
        if "away_timezone_diff_hours" in features_df.columns:
            tz_diffs = features_df["away_timezone_diff_hours"].dropna()
            if len(tz_diffs) > 0:
                if tz_diffs.min() < -4 or tz_diffs.max() > 4:
                    logger.warning(
                        "Timezone differences outside reasonable range",
                        min_tz_diff=tz_diffs.min(),
                        max_tz_diff=tz_diffs.max(),
                    )

        # Check rest days (should be 3-21 typically)
        rest_cols = ["home_rest_days", "away_rest_days"]
        for col in rest_cols:
            if col in features_df.columns:
                rest_days = features_df[col].dropna()
                if len(rest_days) > 0:
                    if rest_days.min() < 0 or rest_days.max() > 30:
                        logger.warning(
                            f"Rest days outside reasonable range for {col}",
                            min_rest=rest_days.min(),
                            max_rest=rest_days.max(),
                        )

        logger.info(
            "Contextual features validation completed",
            records=len(features_df),
            feature_columns=len(
                [
                    col
                    for col in features_df.columns
                    if col
                    not in ["game_id", "season", "week", "home_team", "away_team"]
                ]
            ),
        )

        return True

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build contextual features conforming to FeatureBuilder Protocol.

        Uses only schedule data available before ``as_of_datetime``.

        Delegates to the existing ``build_contextual_features`` logic
        but adds ``as_of_datetime`` filtering.

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff.
            target_season: Optional season filter.
            target_week: Optional week filter.

        Returns:
            DataFrame with contextual features.
        """
        logger.info(
            "Building contextual features (Protocol)",
            games=len(games_df),
            as_of=as_of_datetime.isoformat(),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Time-fence: only use games with kickoff before as_of_datetime
            # for rest-days / schedule context (the target games themselves
            # may have kickoff after the cutoff, but prior games used for
            # rest calculations must be before the cutoff).
            if target_season and target_week:
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

            contextual_features = []

            for _, game in games_df.iterrows():
                game_id = game["game_id"]
                home_team = game["home_team"]
                away_team = game["away_team"]
                venue_name = game.get("venue", "")
                venue_id = self._get_venue_id_by_name(venue_name)
                kickoff_dt = game["kickoff_et"]
                season = game["season"]
                week = game["week"]

                game_features: dict[str, object] = {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                }

                game_features.update(
                    {
                        "is_home_game": 1.0,
                        "is_away_game": 0.0,
                    }
                )

                travel_metrics = self.calculate_travel_metrics(
                    away_team, home_team, venue_id, kickoff_dt
                )
                for key, value in travel_metrics.items():
                    game_features[f"away_{key}"] = value

                short_week_features = self.detect_short_week(kickoff_dt, season, week)
                game_features.update(short_week_features)

                venue_features = self.encode_venue_features(venue_id)
                game_features.update(venue_features)

                # Rest days: use only games with kickoff before as_of_datetime
                prior_games = games_df[games_df["kickoff_et"] < as_of_datetime]
                home_rest = self.calculate_rest_days(home_team, kickoff_dt, prior_games)
                away_rest = self.calculate_rest_days(away_team, kickoff_dt, prior_games)

                game_features.update(
                    {
                        "home_rest_days": home_rest,
                        "away_rest_days": away_rest,
                        "rest_advantage": home_rest - away_rest,
                        "both_short_rest": (
                            1.0 if (home_rest <= 4 and away_rest <= 4) else 0.0
                        ),
                        "home_short_rest": 1.0 if home_rest <= 4 else 0.0,
                        "away_short_rest": 1.0 if away_rest <= 4 else 0.0,
                    }
                )

                contextual_features.append(game_features)

            features_df = pd.DataFrame(contextual_features)

            logger.info(
                "Built contextual features (Protocol)",
                features_count=len(features_df),
                feature_columns=len(features_df.columns),
            )

            return features_df

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to build contextual features (Protocol)",
                error=str(e),
            )
            raise

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get contextual features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        try:
            features_df = load_dataframe("contextual_features", layer="silver")

            game_features = features_df[features_df["game_id"] == game_id]

            if len(game_features) == 0:
                logger.warning(
                    "No contextual features found for game",
                    game_id=game_id,
                )
                return {}

            exclude_cols = {
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
            }
            game_row = game_features.iloc[0]
            return {
                col: float(game_row[col])
                for col in game_features.columns
                if col not in exclude_cols
            }

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to get contextual features for game",
                game_id=game_id,
                error=str(e),
            )
            return {}
