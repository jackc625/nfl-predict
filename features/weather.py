"""
Weather Features Calculator

This module calculates weather-based features for NFL games:
- Wind speed features (primary impact factor)
- Temperature features (cold weather effects)
- Precipitation features (rain, snow impact)
- Weather impact scoring for outdoor games only
- Historical weather trend analysis

Weather has the most impact on:
- Kicking accuracy (field goals, extra points)
- Passing efficiency (wind, precipitation)
- Total scoring (cold, wind, precipitation)
- Turnovers (wet conditions)
"""

import warnings
from datetime import datetime
from typing import Any

import pandas as pd

from conf.settings import get_settings
from data.storage import load_dataframe
from utils import get_logger

logger = get_logger(__name__)

# THE SILVER TABLE THIS BUILDER READS. Stated ONCE, because it was wrong twice.
#
# Plan 33.1-02 Ruling G, and it is load-bearing rather than tidy. Both weather
# writers -- `scripts/ingest_weather.py` (the live forecast) and
# `scripts/backfill_historical_weather.py` (the ERA5 archive) -- write the silver
# table `weather`. This module was reading `weather_forecast`. There is no
# `data/silver/weather_forecast.parquet` at all; `data.storage.load_dataframe`
# tries DuckDB FIRST under `source="auto"` and found a stale `weather_forecast`
# table there holding 14 rows, all `2025_W05_*`, 18 columns, no `weather_source`
# -- a DIFFERENT 14 rows from the `2024_W06_*` rows in `data/silver/weather.parquet`.
# That is why RESEARCH found gold's "real" weather rows are 2025 W05 while silver's
# are 2024 W06, and it is the mechanical reason the mild-temperature default was
# reached for 6,485 of 6,499 gold rows.
#
# A perfect 6,499-row silver promotion would change NOTHING in gold without this
# repair, which is why it sits in the tracer: the tracer's whole job is to prove a
# fetched observation reaches a feature frame.
#
# The reads below state `source="parquet"` rather than leaving `auto`, so a DuckDB
# table created later cannot silently win over the parquet the promotion writes.
#
# NOT FIXED HERE, deliberately: `scripts/data_qa.py` still names `weather_forecast`
# at :168, :309, :519, :868 and :1235. That is a QA reporting surface rather than a
# model input, it is outside this phase's named scope, and Plan 33.1-11's readout
# records it as a standing finding.
SILVER_WEATHER_TABLE: str = "weather"


class WeatherFeaturesCalculator:
    """
    Calculate weather features for NFL games.

    Features calculated:
    - Wind speed features (primary factor for kicking/passing)
    - Temperature features (cold weather effects)
    - Precipitation features (rain/snow impact)
    - Weather severity scoring
    - Historical weather trends for venues
    - Weather impact limited to outdoor/retractable venues
    """

    def __init__(self):
        """Initialize weather features calculator."""
        self.settings = get_settings()

        # Weather thresholds from configuration
        self.wind_threshold = (
            self.settings.config.models.features.weather.wind_threshold
        )  # 12 MPH
        self.temp_threshold = (
            self.settings.config.models.features.weather.temp_threshold
        )  # 32°F

        # Additional weather thresholds
        self.severe_wind_threshold = 20.0  # MPH
        self.very_cold_threshold = 20.0  # °F
        self.heavy_precip_threshold = 5.0  # mm

    def _calculate_apparent_temperature(
        self, temp_f: float, wind_mph: float, humidity_pct: float
    ) -> float:
        """
        Calculate apparent temperature (wind chill or heat index).

        Research shows apparent temperature is better than raw temp because it folds
        in wind/humidity effects that impact player performance.

        Args:
            temp_f: Temperature in Fahrenheit
            wind_mph: Wind speed in MPH
            humidity_pct: Relative humidity percentage

        Returns:
            Apparent temperature in Fahrenheit
        """
        if temp_f <= 50 and wind_mph >= 3:
            # Wind chill formula (applicable when temp <= 50°F and wind >= 3 MPH)
            wind_chill = (
                35.74
                + 0.6215 * temp_f
                - 35.75 * (wind_mph**0.16)
                + 0.4275 * temp_f * (wind_mph**0.16)
            )
            return wind_chill
        if temp_f >= 80 and humidity_pct >= 40:
            # Heat index formula (applicable when temp >= 80°F and humidity >= 40%)
            hi = (
                -42.379
                + 2.04901523 * temp_f
                + 10.14333127 * humidity_pct
                - 0.22475541 * temp_f * humidity_pct
                - 0.00683783 * temp_f * temp_f
                - 0.05481717 * humidity_pct * humidity_pct
                + 0.00122874 * temp_f * temp_f * humidity_pct
                + 0.00085282 * temp_f * humidity_pct * humidity_pct
                - 0.00000199 * temp_f * temp_f * humidity_pct * humidity_pct
            )
            return hi
        # No adjustment needed for moderate conditions
        return temp_f

    def calculate_wind_features(self, weather_data: dict[str, Any]) -> dict[str, float]:
        """
        Calculate wind-related features.

        Wind is the primary weather factor affecting:
        - Field goal accuracy (significant impact >12 MPH)
        - Passing effectiveness (dropoff >15 MPH)
        - Punt/kickoff distance and accuracy

        Args:
            weather_data: Weather data dictionary

        Returns:
            Dictionary with wind features
        """
        try:
            wind_mph = weather_data.get("wind_mph", 0.0) or 0.0

            # Basic wind features
            wind_features = {
                "wind_mph": float(wind_mph),
                "wind_calm": 1.0 if wind_mph <= 5.0 else 0.0,
                "wind_moderate": 1.0 if 5.0 < wind_mph <= self.wind_threshold else 0.0,
                "wind_high": 1.0
                if self.wind_threshold < wind_mph <= self.severe_wind_threshold
                else 0.0,
                "wind_severe": 1.0 if wind_mph > self.severe_wind_threshold else 0.0,
            }

            # Wind impact scoring (0-1 scale)
            # Based on research showing kicking accuracy drops significantly >12 MPH
            if wind_mph <= 5.0:
                wind_impact = 0.0  # No impact
            elif wind_mph <= self.wind_threshold:
                wind_impact = 0.2  # Minimal impact
            elif wind_mph <= self.severe_wind_threshold:
                wind_impact = 0.6  # Moderate impact
            else:
                wind_impact = 1.0  # Severe impact

            wind_features["wind_impact_score"] = wind_impact

            # Kicking difficulty multiplier
            # Field goal accuracy drops ~5% per 5 MPH over 10 MPH
            if wind_mph <= 10.0:
                kicking_difficulty = 1.0
            else:
                # Scale from 1.0 to 2.0 based on wind speed
                kicking_difficulty = 1.0 + min((wind_mph - 10.0) / 20.0, 1.0)

            wind_features["kicking_difficulty"] = kicking_difficulty

            # Passing difficulty (affects completion percentage and accuracy)
            if wind_mph <= 8.0:
                passing_difficulty = 1.0
            else:
                # Passing becomes noticeably harder >15 MPH
                passing_difficulty = 1.0 + min((wind_mph - 8.0) / 17.0, 1.0)

            wind_features["passing_difficulty"] = passing_difficulty

            return wind_features

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to calculate wind features", error=str(e))
            return self._default_wind_features()

    def _default_wind_features(self) -> dict[str, float]:
        """Return default wind features when calculation fails."""
        return {
            "wind_mph": 0.0,
            "wind_calm": 1.0,
            "wind_moderate": 0.0,
            "wind_high": 0.0,
            "wind_severe": 0.0,
            "wind_impact_score": 0.0,
            "kicking_difficulty": 1.0,
            "passing_difficulty": 1.0,
        }

    def calculate_temperature_features(
        self, weather_data: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate temperature-related features based on research findings.

        Research-backed temperature effects:
        - 55-85°F: No effect on scoring
        - 25-55°F: ~5% scoring reduction (passing efficiency drops)
        - <25°F or >85°F: ~8% scoring reduction (significant effects)

        Uses apparent temperature (wind chill/heat index) for more accurate
        impact assessment than raw temperature alone.

        Key findings:
        - Wind is often larger driver than temperature
        - Cold effects mainly impact passing efficiency and kicking
        - Extreme temperature thresholds are required for measurable effects

        Args:
            weather_data: Weather data dictionary with temp_f, wind_mph, humidity_pct

        Returns:
            Dictionary with temperature features including apparent_temp_f
        """
        try:
            temp_f = weather_data.get("temp_f")
            wind_mph = weather_data.get("wind_mph", 0)
            humidity_pct = weather_data.get("humidity_pct", 50)

            if temp_f is None:
                return self._default_temperature_features()

            temp_f = float(temp_f)
            wind_mph = float(wind_mph or 0)
            humidity_pct = float(humidity_pct or 50)

            # Basic temperature categories
            temp_features = {
                "temp_f": temp_f,
                "temp_hot": 1.0 if temp_f >= 85.0 else 0.0,
                "temp_warm": 1.0 if 70.0 <= temp_f < 85.0 else 0.0,
                "temp_mild": 1.0 if 50.0 <= temp_f < 70.0 else 0.0,
                "temp_cool": 1.0 if self.temp_threshold <= temp_f < 50.0 else 0.0,
                "temp_cold": 1.0
                if self.very_cold_threshold <= temp_f < self.temp_threshold
                else 0.0,
                "temp_very_cold": 1.0 if temp_f < self.very_cold_threshold else 0.0,
            }

            # Calculate apparent temperature (wind chill / heat index)
            apparent_temp = self._calculate_apparent_temperature(
                temp_f, wind_mph, humidity_pct
            )
            temp_features["apparent_temp_f"] = round(apparent_temp, 1)

            # Research-based temperature impact (using apparent temperature)
            # Cold weather impact (0-1 scale) - only significant below 25°F
            if apparent_temp >= 25.0:
                cold_impact = 0.0  # No cold impact above 25°F
            else:
                # Linear scaling from 25°F to 0°F
                cold_impact = min(1.0, (25.0 - apparent_temp) / 25.0)

            temp_features["cold_impact_score"] = cold_impact

            # Research-based scoring impact (piecewise function)
            # Based on data from Sharp Football Analysis and others
            if 55 <= apparent_temp <= 85:
                scoring_impact = 1.00  # No effect in normal range
            elif 25 <= apparent_temp < 55:
                scoring_impact = 0.95  # 5% reduction in cool weather
            elif apparent_temp < 25 or apparent_temp > 85:
                scoring_impact = 0.92  # 8% reduction in extreme weather
            else:
                scoring_impact = 1.00  # Fallback

            temp_features["scoring_multiplier"] = scoring_impact

            # Ball handling difficulty (fumbles increase in cold)
            if temp_f >= 40.0:
                handling_difficulty = 1.0
            else:
                # Fumbles increase ~10% per 10°F below 40°F
                handling_difficulty = 1.0 + (40.0 - temp_f) * 0.01

            temp_features["ball_handling_difficulty"] = min(handling_difficulty, 1.5)

            # Heat impact (extreme heat also affects performance)
            if temp_f <= 90.0:
                heat_impact = 0.0
            else:
                # Heat exhaustion becomes factor >90°F
                heat_impact = min((temp_f - 90.0) / 20.0, 1.0)

            temp_features["heat_impact_score"] = heat_impact

            return temp_features

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to calculate temperature features", error=str(e))
            return self._default_temperature_features()

    def _default_temperature_features(self) -> dict[str, float]:
        """Return default temperature features when calculation fails."""
        return {
            "temp_f": 65.0,  # Default mild temperature
            "apparent_temp_f": 65.0,  # Default apparent temperature
            "temp_hot": 0.0,
            "temp_warm": 1.0,
            "temp_mild": 0.0,
            "temp_cool": 0.0,
            "temp_cold": 0.0,
            "temp_very_cold": 0.0,
            "cold_impact_score": 0.0,
            "scoring_multiplier": 1.0,
            "ball_handling_difficulty": 1.0,
            "heat_impact_score": 0.0,
        }

    def calculate_precipitation_features(
        self, weather_data: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate precipitation-related features.

        Precipitation effects:
        - Increased turnovers (wet ball, slippery field)
        - Reduced passing accuracy
        - Advantage to running game
        - Reduced scoring overall

        Args:
            weather_data: Weather data dictionary

        Returns:
            Dictionary with precipitation features
        """
        try:
            precip_prob = weather_data.get("precip_prob", 0.0) or 0.0
            precip_mm = weather_data.get("precip_mm", 0.0) or 0.0
            condition = (weather_data.get("condition") or "").lower()
            temp_f = weather_data.get("temp_f", 40.0) or 40.0

            # Basic precipitation features
            precip_features = {
                "precip_prob": float(precip_prob),
                "precip_mm": float(precip_mm),
                "precip_none": 1.0 if precip_prob <= 0.2 and precip_mm <= 0.5 else 0.0,
                "precip_light": 1.0
                if 0.2 < precip_prob <= 0.5 or 0.5 < precip_mm <= 2.0
                else 0.0,
                "precip_moderate": 1.0
                if 0.5 < precip_prob <= 0.8
                or 2.0 < precip_mm <= self.heavy_precip_threshold
                else 0.0,
                "precip_heavy": 1.0
                if precip_prob > 0.8 or precip_mm > self.heavy_precip_threshold
                else 0.0,
            }

            # Precipitation type (snow vs rain has different effects)
            is_snow = temp_f <= 35.0 and (
                precip_prob > 0.3 or precip_mm > 0.5 or "snow" in condition
            )
            is_rain = temp_f > 35.0 and (
                precip_prob > 0.3 or precip_mm > 0.5 or "rain" in condition
            )

            precip_features.update(
                {
                    "is_snow": 1.0 if is_snow else 0.0,
                    "is_rain": 1.0 if is_rain else 0.0,
                    "is_dry": 1.0 if not (is_snow or is_rain) else 0.0,
                }
            )

            # Precipitation impact scoring
            if precip_prob <= 0.2 and precip_mm <= 0.5:
                precip_impact = 0.0  # No impact
            elif precip_prob <= 0.5 or precip_mm <= 2.0:
                precip_impact = 0.3  # Light impact
            elif precip_prob <= 0.8 or precip_mm <= self.heavy_precip_threshold:
                precip_impact = 0.6  # Moderate impact
            else:
                precip_impact = 1.0  # Heavy impact

            # Snow has different impact than rain
            if is_snow:
                precip_impact *= 1.2  # Snow generally worse than rain

            precip_features["precip_impact_score"] = min(precip_impact, 1.0)

            # Turnover multiplier (wet conditions increase fumbles/interceptions)
            if precip_prob <= 0.3 and precip_mm <= 1.0:
                turnover_multiplier = 1.0
            else:
                # Turnovers can increase 20-40% in wet conditions
                base_increase = (
                    0.2 + (precip_prob * 0.2) + (min(precip_mm, 10.0) / 10.0 * 0.2)
                )
                turnover_multiplier = 1.0 + base_increase

            precip_features["turnover_multiplier"] = min(turnover_multiplier, 1.5)

            # Passing efficiency reduction
            if precip_prob <= 0.3:
                passing_efficiency = 1.0
            else:
                # Completion percentage drops in wet conditions
                passing_efficiency = max(
                    0.85, 1.0 - (precip_prob * 0.15) - (min(precip_mm, 5.0) / 5.0 * 0.1)
                )

            precip_features["passing_efficiency"] = passing_efficiency

            return precip_features

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to calculate precipitation features", error=str(e))
            return self._default_precipitation_features()

    def _default_precipitation_features(self) -> dict[str, float]:
        """Return default precipitation features when calculation fails."""
        return {
            "precip_prob": 0.0,
            "precip_mm": 0.0,
            "precip_none": 1.0,
            "precip_light": 0.0,
            "precip_moderate": 0.0,
            "precip_heavy": 0.0,
            "is_snow": 0.0,
            "is_rain": 0.0,
            "is_dry": 1.0,
            "precip_impact_score": 0.0,
            "turnover_multiplier": 1.0,
            "passing_efficiency": 1.0,
        }

    def calculate_weather_severity(
        self,
        wind_features: dict[str, float],
        temp_features: dict[str, float],
        precip_features: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate overall weather severity scores.

        Args:
            wind_features: Wind feature dictionary
            temp_features: Temperature feature dictionary
            precip_features: Precipitation feature dictionary

        Returns:
            Dictionary with overall weather severity features
        """
        try:
            # Individual impact scores
            wind_impact = wind_features.get("wind_impact_score", 0.0)
            cold_impact = temp_features.get("cold_impact_score", 0.0)
            heat_impact = temp_features.get("heat_impact_score", 0.0)
            precip_impact = precip_features.get("precip_impact_score", 0.0)

            # Combined weather severity (research-based weights)
            # Wind is the larger driver, temperature has independent but smaller signal
            weather_severity = (
                wind_impact * 0.5  # Wind: 50% weight (larger driver)
                + (cold_impact + heat_impact)
                * 0.2  # Temperature: 20% weight (independent signal)
                + precip_impact * 0.3  # Precipitation: 30% weight
            )

            # Weather advantage factors
            # Home teams typically have advantage in severe weather (familiarity)
            home_weather_advantage = (
                weather_severity * 0.1
            )  # 10% of severity becomes home advantage

            # Game style impact (weather favors defense and running game)
            defensive_advantage = weather_severity * 0.15
            rushing_advantage = weather_severity * 0.2

            # Scoring impact (severe weather reduces total points)
            scoring_reduction = min(weather_severity * 0.8, 0.3)  # Max 30% reduction

            return {
                "weather_severity_score": weather_severity,
                "home_weather_advantage": home_weather_advantage,
                "defensive_advantage": defensive_advantage,
                "rushing_advantage": rushing_advantage,
                "scoring_reduction": scoring_reduction,
                "weather_game": 1.0 if weather_severity >= 0.6 else 0.0,
                "extreme_weather": 1.0 if weather_severity >= 0.8 else 0.0,
            }

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to calculate weather severity", error=str(e))
            return {
                "weather_severity_score": 0.0,
                "home_weather_advantage": 0.0,
                "defensive_advantage": 0.0,
                "rushing_advantage": 0.0,
                "scoring_reduction": 0.0,
                "weather_game": 0.0,
                "extreme_weather": 0.0,
            }

    def build_weather_features(
        self,
        games_df: pd.DataFrame,
        target_season: int | None = None,
        target_week: int | None = None,
        *,
        weather_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build weather features for all games.

        .. deprecated::
            Use :meth:`build_features` instead (conforms to FeatureBuilder Protocol).

        Args:
            games_df: DataFrame with game information
            target_season: Specific season to calculate features for
            target_week: Specific week to calculate features for
            weather_df: The silver weather frame to build over. When given it is
                used AS-IS and no store is read; when ``None`` the silver read
                happens as before. This is the injection seam a sandboxed run or a
                test uses -- `monkeypatch` is not a sandbox, and a test that
                patched a table NAME while the write still resolved the production
                root is how Phase 33 Wave 6 destroyed production gold. Threading
                the frame is the same discipline
                ``backtest.ou_divergence.run_ou_divergence_diagnosis`` already uses
                for ``preds`` and ``odds``.

        Returns:
            DataFrame with weather features added (full 38+ columns, uncompressed)
        """
        warnings.warn(
            "build_weather_features is deprecated; use build_features instead",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(
            "Building weather features",
            games=len(games_df),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Load weather data, unless the caller supplied the frame.
            #
            # ONE LINE, layer and source POSITIONAL -- `load_dataframe(table_name,
            # layer, source)`. Plan 33.1-02's guard scans for `load_dataframe(`,
            # `SILVER_WEATHER_TABLE` and `parquet` CO-LOCATED on a line, and a
            # wrapped call reads to it as absent. "parquet" is stated rather than
            # left at "auto" so a DuckDB table created later cannot silently win
            # over the parquet the promotion actually writes.
            if weather_df is None:
                weather_df = load_dataframe(SILVER_WEATHER_TABLE, "silver", "parquet")
            logger.info("Loaded weather data", weather_records=len(weather_df))

            # Filter to target if specified
            if target_season and target_week:
                # Filter games
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

                # Filter weather data to matching games
                game_ids = games_df["game_id"].tolist()
                weather_df = weather_df[weather_df["game_id"].isin(game_ids)]

            weather_features = []

            for _, game in games_df.iterrows():
                game_id = game["game_id"]
                season = game["season"]
                week = game["week"]

                # Find weather data for this game
                game_weather = weather_df[weather_df["game_id"] == game_id]

                if len(game_weather) == 0:
                    logger.warning("No weather data found for game", game_id=game_id)
                    # Use default weather (no impact)
                    weather_data = {
                        "is_outdoor": False,
                        "temp_f": 65.0,
                        "wind_mph": 0.0,
                        "precip_prob": 0.0,
                        "precip_mm": 0.0,
                        "condition": "Clear",
                    }
                else:
                    # Use most recent weather forecast for this game
                    latest_weather = game_weather.sort_values("forecast_time").iloc[-1]
                    weather_data = latest_weather.to_dict()

                # Basic game identifiers
                game_features = {"game_id": game_id, "season": season, "week": week}

                # Check if weather should impact this game (outdoor only)
                is_outdoor = weather_data.get("is_outdoor", False)
                game_features["weather_affects_game"] = 1.0 if is_outdoor else 0.0

                if not is_outdoor:
                    # Indoor game - weather has no impact
                    game_features.update(self._default_wind_features())
                    game_features.update(self._default_temperature_features())
                    game_features.update(self._default_precipitation_features())

                    # No weather severity for indoor games
                    game_features.update(
                        {
                            "weather_severity_score": 0.0,
                            "home_weather_advantage": 0.0,
                            "defensive_advantage": 0.0,
                            "rushing_advantage": 0.0,
                            "scoring_reduction": 0.0,
                            "weather_game": 0.0,
                            "extreme_weather": 0.0,
                        }
                    )
                else:
                    # Outdoor game - calculate weather features
                    wind_features = self.calculate_wind_features(weather_data)
                    temp_features = self.calculate_temperature_features(weather_data)
                    precip_features = self.calculate_precipitation_features(
                        weather_data
                    )
                    severity_features = self.calculate_weather_severity(
                        wind_features, temp_features, precip_features
                    )

                    # Combine all weather features
                    game_features.update(wind_features)
                    game_features.update(temp_features)
                    game_features.update(precip_features)
                    game_features.update(severity_features)

                # Add raw weather values for reference
                game_features.update(
                    {
                        "raw_temp_f": weather_data.get("temp_f"),
                        "raw_wind_mph": weather_data.get("wind_mph"),
                        "raw_precip_prob": weather_data.get("precip_prob"),
                        "raw_precip_mm": weather_data.get("precip_mm"),
                        "raw_humidity_pct": weather_data.get("humidity_pct"),
                        "weather_condition": weather_data.get("condition", "Unknown"),
                    }
                )

                weather_features.append(game_features)

            # Convert to DataFrame
            features_df = pd.DataFrame(weather_features)

            logger.info(
                "Built weather features",
                features_count=len(features_df),
                outdoor_games=features_df["weather_affects_game"].sum(),
                weather_games=features_df.get("weather_game", pd.Series([0])).sum(),
            )

            return features_df

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to build weather features", error=str(e))
            raise

    def get_weather_features_for_game(
        self, game_id: str, season: int, week: int
    ) -> dict[str, float]:
        """
        Get weather features for a specific game.

        Args:
            game_id: Game identifier
            season: Season year
            week: Week number

        Returns:
            Dictionary with weather features
        """
        try:
            # Load weather features
            features_df = load_dataframe("weather_features", layer="silver")

            # Filter to specific game
            game_features = features_df[
                (features_df["game_id"] == game_id)
                & (features_df["season"] == season)
                & (features_df["week"] == week)
            ]

            if len(game_features) == 0:
                logger.warning(
                    "No weather features found for game",
                    game_id=game_id,
                    season=season,
                    week=week,
                )
                return {}

            # Convert to dictionary, excluding non-feature columns
            exclude_cols = ["game_id", "season", "week"]
            features_dict = {}

            game_row = game_features.iloc[0]
            for col in game_features.columns:
                if col not in exclude_cols:
                    features_dict[col] = game_row[col]

            return features_dict

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Failed to get weather features for game",
                game_id=game_id,
                season=season,
                week=week,
                error=str(e),
            )
            return {}

    def validate_weather_features(self, features_df: pd.DataFrame) -> bool:
        """
        Validate weather features for data quality.

        Args:
            features_df: Weather features DataFrame

        Returns:
            True if validation passes
        """
        if len(features_df) == 0:
            logger.error("No weather features found")
            return False

        # Check for required columns
        required_cols = ["game_id", "season", "week", "weather_affects_game"]
        missing_cols = set(required_cols) - set(features_df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing_columns=list(missing_cols))
            return False

        # Check temperature ranges
        if "temp_f" in features_df.columns:
            temps = features_df["temp_f"].dropna()
            if len(temps) > 0 and (temps.min() < -20 or temps.max() > 130):
                logger.warning(
                    "Temperature values outside reasonable range",
                    min_temp=temps.min(),
                    max_temp=temps.max(),
                )

        # Check wind speeds
        if "wind_mph" in features_df.columns:
            winds = features_df["wind_mph"].dropna()
            if len(winds) > 0 and (winds.min() < 0 or winds.max() > 100):
                logger.warning(
                    "Wind speeds outside reasonable range",
                    min_wind=winds.min(),
                    max_wind=winds.max(),
                )

        # Check precipitation probabilities
        if "precip_prob" in features_df.columns:
            precip_probs = features_df["precip_prob"].dropna()
            if len(precip_probs) > 0:
                if precip_probs.min() < 0 or precip_probs.max() > 1:
                    logger.warning(
                        "Precipitation probabilities outside 0-1 range",
                        min_precip=precip_probs.min(),
                        max_precip=precip_probs.max(),
                    )

        # Check severity scores are 0-1
        severity_cols = [
            "weather_severity_score",
            "wind_impact_score",
            "cold_impact_score",
            "precip_impact_score",
        ]
        for col in severity_cols:
            if col in features_df.columns:
                values = features_df[col].dropna()
                if len(values) > 0:
                    if values.min() < 0 or values.max() > 1:
                        logger.warning(
                            f"Severity scores outside 0-1 range for {col}",
                            min_value=values.min(),
                            max_value=values.max(),
                        )

        # Check outdoor vs indoor games
        outdoor_games = features_df["weather_affects_game"].sum()
        total_games = len(features_df)

        logger.info(
            "Weather features validation completed",
            records=total_games,
            outdoor_games=int(outdoor_games),
            indoor_games=int(total_games - outdoor_games),
            weather_columns=len(
                [
                    col
                    for col in features_df.columns
                    if col not in ["game_id", "season", "week"]
                ]
            ),
        )

        return True

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods (compressed output)
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
        weather_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build compressed weather features (4 features) for games.

        Conforms to the FeatureBuilder Protocol. Uses only weather data
        available before ``as_of_datetime``.

        Output columns:
            game_id, weather_severity_score, wind_mph, is_precipitation, is_outdoor

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff. Only data before this
                timestamp may be used.
            target_season: Optional season filter.
            target_week: Optional week filter.
            weather_df: The silver weather frame to build over. When given it is
                used AS-IS and no store is read; when ``None`` the silver read
                happens as before. See :meth:`build_weather_features` for why this
                seam exists rather than a monkeypatch.

        Returns:
            DataFrame with exactly 5 columns (game_id + 4 features).
        """
        logger.info(
            "Building compressed weather features",
            games=len(games_df),
            as_of=as_of_datetime.isoformat(),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Load weather data, unless the caller supplied the frame.
            #
            # ONE LINE, layer and source POSITIONAL -- `load_dataframe(table_name,
            # layer, source)`. Plan 33.1-02's guard scans for `load_dataframe(`,
            # `SILVER_WEATHER_TABLE` and `parquet` CO-LOCATED on a line, and a
            # wrapped call reads to it as absent. "parquet" is stated rather than
            # left at "auto" so a DuckDB table created later cannot silently win
            # over the parquet the promotion actually writes.
            if weather_df is None:
                weather_df = load_dataframe(SILVER_WEATHER_TABLE, "silver", "parquet")
            logger.info("Loaded weather data", weather_records=len(weather_df))

            # Time-fence: only use forecasts available before as_of_datetime
            if "forecast_time" in weather_df.columns:
                weather_df = weather_df[weather_df["forecast_time"] <= as_of_datetime]

            # Filter to target if specified
            if target_season and target_week:
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

                game_ids = games_df["game_id"].tolist()
                weather_df = weather_df[weather_df["game_id"].isin(game_ids)]

            compressed_rows: list[dict[str, object]] = []

            for _, game in games_df.iterrows():
                game_id = game["game_id"]

                # Find weather data for this game
                game_weather = weather_df[weather_df["game_id"] == game_id]

                if len(game_weather) == 0:
                    logger.warning("No weather data found for game", game_id=game_id)
                    weather_data: dict[str, Any] = {
                        "is_outdoor": False,
                        "temp_f": 65.0,
                        "wind_mph": 0.0,
                        "precip_prob": 0.0,
                        "precip_mm": 0.0,
                        "condition": "Clear",
                        "humidity_pct": 50.0,
                    }
                else:
                    latest_weather = game_weather.sort_values("forecast_time").iloc[-1]
                    weather_data = latest_weather.to_dict()

                is_outdoor = bool(weather_data.get("is_outdoor", False))

                if not is_outdoor:
                    # Indoor/dome: all features zeroed
                    compressed_rows.append(
                        {
                            "game_id": game_id,
                            "weather_severity_score": 0.0,
                            "wind_mph": 0.0,
                            "is_precipitation": 0.0,
                            "is_outdoor": 0.0,
                        }
                    )
                else:
                    # Outdoor game: compute full features, then compress
                    wind_features = self.calculate_wind_features(weather_data)
                    temp_features = self.calculate_temperature_features(weather_data)
                    precip_features = self.calculate_precipitation_features(
                        weather_data
                    )
                    severity_features = self.calculate_weather_severity(
                        wind_features, temp_features, precip_features
                    )

                    # Determine is_precipitation: binary flag
                    precip_prob = float(weather_data.get("precip_prob", 0.0) or 0.0)
                    precip_mm = float(weather_data.get("precip_mm", 0.0) or 0.0)
                    is_snow = precip_features.get("is_snow", 0.0)
                    is_rain = precip_features.get("is_rain", 0.0)

                    is_precip = (
                        1.0
                        if (
                            precip_prob > 0.3
                            or precip_mm > 0.5
                            or is_snow == 1.0
                            or is_rain == 1.0
                        )
                        else 0.0
                    )

                    compressed_rows.append(
                        {
                            "game_id": game_id,
                            "weather_severity_score": severity_features[
                                "weather_severity_score"
                            ],
                            "wind_mph": wind_features["wind_mph"],
                            "is_precipitation": is_precip,
                            "is_outdoor": 1.0,
                        }
                    )

            features_df = pd.DataFrame(compressed_rows)

            logger.info(
                "Built compressed weather features",
                features_count=len(features_df),
                outdoor_games=int(features_df["is_outdoor"].sum()),
            )

            return features_df

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to build compressed weather features", error=str(e))
            raise

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get compressed weather features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        try:
            # Build a minimal games DataFrame for the single game
            games_df = pd.DataFrame([{"game_id": game_id, "season": 0, "week": 0}])
            result = self.build_features(games_df, as_of_datetime)

            if len(result) == 0:
                logger.warning(
                    "No weather features for game",
                    game_id=game_id,
                )
                return {}

            row = result.iloc[0]
            return {col: float(row[col]) for col in result.columns if col != "game_id"}

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Failed to get weather features for game",
                game_id=game_id,
                error=str(e),
            )
            return {}
