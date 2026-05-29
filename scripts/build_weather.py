#!/usr/bin/env python3
"""
Build weather features for NFL prediction system.

This script builds weather features including:
- Wind speed features (primary factor for kicking/passing)
- Temperature features (cold weather effects)
- Precipitation features (rain/snow impact)
- Weather severity scoring for outdoor games only

Usage:
    python scripts/build_weather.py --season 2024 --week 1
    python scripts/build_weather.py --season 2024  # All weeks in season
    python scripts/build_weather.py  # All available data
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage import load_dataframe, save_dataframe
from features.weather import WeatherFeaturesCalculator
from utils import get_logger

logger = get_logger(__name__)


def main():
    """Build weather features."""
    parser = argparse.ArgumentParser(description="Build weather features")
    parser.add_argument("--season", type=int, help="Target season (e.g., 2024)")
    parser.add_argument("--week", type=int, help="Target week (1-18)")
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save features to silver layer",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate features after building",
    )

    args = parser.parse_args()

    logger.info("Building weather features", season=args.season, week=args.week)

    try:
        # Load games data from silver layer
        games_df = load_dataframe("games", layer="silver")
        logger.info("Loaded games data", games_count=len(games_df))

        # Initialize weather features calculator
        calculator = WeatherFeaturesCalculator()

        # Build weather features
        features_df = calculator.build_weather_features(
            games_df=games_df, target_season=args.season, target_week=args.week
        )

        if len(features_df) == 0:
            logger.warning("No weather features generated")
            return

        logger.info(
            "Generated weather features",
            games=len(features_df),
            outdoor_games=int(features_df["weather_affects_game"].sum()),
            weather_games=int(features_df.get("weather_game", pd.Series([0])).sum()),
        )

        # Validate features if requested
        if args.validate:
            is_valid = calculator.validate_weather_features(features_df)
            if not is_valid:
                logger.error("Weather features validation failed")
                return
            logger.info("Weather features validation passed")

        # Display sample weather features
        print("\nWeather features summary:")
        print("=" * 60)

        outdoor_games = features_df[features_df["weather_affects_game"] == 1.0]
        indoor_games = features_df[features_df["weather_affects_game"] == 0.0]

        print(f"Total games: {len(features_df)}")
        print(f"Outdoor games: {len(outdoor_games)}")
        print(f"Indoor games: {len(indoor_games)}")

        if len(outdoor_games) > 0:
            weather_games = outdoor_games[outdoor_games.get("weather_game", 0) == 1.0]
            extreme_games = outdoor_games[
                outdoor_games.get("extreme_weather", 0) == 1.0
            ]

            print(f"Weather games (severity >= 0.6): {len(weather_games)}")
            print(f"Extreme weather games (severity >= 0.8): {len(extreme_games)}")

            # Show weather statistics for outdoor games
            print("\nOutdoor weather statistics:")
            if "temp_f" in outdoor_games.columns:
                temp_stats = outdoor_games["temp_f"].describe()
                print(
                    f"  Temperature: {temp_stats['mean']:.1f}°F avg (range: {temp_stats['min']:.0f} to {temp_stats['max']:.0f})"
                )

            if "wind_mph" in outdoor_games.columns:
                wind_stats = outdoor_games["wind_mph"].describe()
                print(
                    f"  Wind speed: {wind_stats['mean']:.1f} mph avg (range: {wind_stats['min']:.0f} to {wind_stats['max']:.0f})"
                )

            if "weather_severity_score" in outdoor_games.columns:
                severity_stats = outdoor_games["weather_severity_score"].describe()
                print(
                    f"  Weather severity: {severity_stats['mean']:.3f} avg (range: {severity_stats['min']:.3f} to {severity_stats['max']:.3f})"
                )

            # Show most severe weather games
            if len(weather_games) > 0:
                print("\nMost severe weather games:")
                severe_games = outdoor_games.nlargest(3, "weather_severity_score")
                for _, game in severe_games.iterrows():
                    severity = game["weather_severity_score"]
                    temp = game.get("temp_f", "N/A")
                    wind = game.get("wind_mph", "N/A")
                    precip = game.get("precip_prob", 0) * 100

                    print(f"  {game['game_id']}: severity {severity:.3f}")
                    print(f"    {temp}°F, {wind} mph wind, {precip:.0f}% precip chance")

                    conditions = []
                    if game.get("temp_very_cold", 0) == 1:
                        conditions.append("very cold")
                    elif game.get("temp_cold", 0) == 1:
                        conditions.append("cold")
                    if game.get("wind_severe", 0) == 1:
                        conditions.append("severe wind")
                    elif game.get("wind_high", 0) == 1:
                        conditions.append("high wind")
                    if game.get("precip_heavy", 0) == 1:
                        conditions.append("heavy precipitation")
                    elif game.get("precip_moderate", 0) == 1:
                        conditions.append("moderate precipitation")
                    if game.get("is_snow", 0) == 1:
                        conditions.append("snow")
                    elif game.get("is_rain", 0) == 1:
                        conditions.append("rain")

                    if conditions:
                        print(f"    Conditions: {', '.join(conditions)}")

        # Save to silver layer if requested.
        # Single self-contained file (no directory partitioning). The per-season
        # --save path appends into this one file with game_id-level latest-wins
        # dedup, so a rebuild is idempotent. The prior partition_cols=["season"]
        # routed to pq.write_to_dataset against the SHARED data/silver/ root,
        # producing the catastrophic ~1048x weather_features bloat (and the
        # cross-table mixing that crashed the rebuild with KeyError:
        # 'forecast_time'). This now matches pipeline/steps.py. weather_features
        # is read by build_features.py, so de-bloating it is the enabling
        # correctness fix for the gold weather merge. (FIX-01, D-13)
        if args.save:
            save_dataframe(
                features_df,
                table_name="weather_features",
                layer="silver",
            )

            logger.info(
                "Saved weather features to silver layer", table_name="weather_features"
            )

        logger.info("Weather features building completed successfully")

    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
        logger.error("Failed to build weather features", error=str(e))
        raise


if __name__ == "__main__":
    main()
