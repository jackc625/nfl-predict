#!/usr/bin/env python3
"""
Test script for weather features calculator.

This script validates the weather features implementation with various weather scenarios.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytz

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.weather import WeatherFeaturesCalculator
from utils import get_logger

logger = get_logger(__name__)


def create_sample_weather_data() -> pd.DataFrame:
    """Create sample weather data for testing different scenarios."""
    et_tz = pytz.timezone("America/New_York")
    base_time = et_tz.localize(datetime(2024, 1, 10, 18, 0))

    sample_weather = [
        {
            # Scenario 1: Perfect weather (indoor dome)
            "game_id": "TEST_2024_01_NO_ATL",
            "forecast_time": base_time,
            "game_time": base_time + timedelta(hours=2),
            "is_outdoor": False,
            "temp_f": 72.0,
            "wind_mph": 0.0,
            "precip_prob": 0.0,
            "precip_mm": 0.0,
            "condition": "Indoor",
            "humidity_pct": 45.0,
        },
        {
            # Scenario 2: Cold, windy game (Green Bay in January)
            "game_id": "TEST_2024_17_DAL_GB",
            "forecast_time": base_time,
            "game_time": base_time + timedelta(hours=2),
            "is_outdoor": True,
            "temp_f": 8.0,
            "wind_mph": 25.0,
            "precip_prob": 0.2,
            "precip_mm": 0.5,
            "condition": "Snow",
            "humidity_pct": 85.0,
        },
        {
            # Scenario 3: Hot, humid game (Miami in September)
            "game_id": "TEST_2024_02_NE_MIA",
            "forecast_time": base_time,
            "game_time": base_time + timedelta(hours=2),
            "is_outdoor": True,
            "temp_f": 94.0,
            "wind_mph": 8.0,
            "precip_prob": 0.7,
            "precip_mm": 8.0,
            "condition": "Rain",
            "humidity_pct": 92.0,
        },
        {
            # Scenario 4: Moderate conditions (typical fall game)
            "game_id": "TEST_2024_08_KC_DEN",
            "forecast_time": base_time,
            "game_time": base_time + timedelta(hours=2),
            "is_outdoor": True,
            "temp_f": 58.0,
            "wind_mph": 12.0,
            "precip_prob": 0.1,
            "precip_mm": 0.0,
            "condition": "Partly Cloudy",
            "humidity_pct": 55.0,
        },
        {
            # Scenario 5: Extreme weather (Chicago Bears weather)
            "game_id": "TEST_2024_15_MIN_CHI",
            "forecast_time": base_time,
            "game_time": base_time + timedelta(hours=2),
            "is_outdoor": True,
            "temp_f": 15.0,
            "wind_mph": 35.0,
            "precip_prob": 0.9,
            "precip_mm": 12.0,
            "condition": "Heavy Snow",
            "humidity_pct": 90.0,
        },
    ]

    return pd.DataFrame(sample_weather)


def create_sample_games_data() -> pd.DataFrame:
    """Create sample games data corresponding to weather scenarios."""
    sample_games = [
        {
            "game_id": "TEST_2024_01_NO_ATL",
            "season": 2024,
            "week": 1,
            "home_team": "ATL",
            "away_team": "NO",
            "kickoff_et": datetime(2024, 1, 10, 20, 0),
        },
        {
            "game_id": "TEST_2024_17_DAL_GB",
            "season": 2024,
            "week": 17,
            "home_team": "GB",
            "away_team": "DAL",
            "kickoff_et": datetime(2024, 1, 10, 20, 0),
        },
        {
            "game_id": "TEST_2024_02_NE_MIA",
            "season": 2024,
            "week": 2,
            "home_team": "MIA",
            "away_team": "NE",
            "kickoff_et": datetime(2024, 1, 10, 20, 0),
        },
        {
            "game_id": "TEST_2024_08_KC_DEN",
            "season": 2024,
            "week": 8,
            "home_team": "DEN",
            "away_team": "KC",
            "kickoff_et": datetime(2024, 1, 10, 20, 0),
        },
        {
            "game_id": "TEST_2024_15_MIN_CHI",
            "season": 2024,
            "week": 15,
            "home_team": "CHI",
            "away_team": "MIN",
            "kickoff_et": datetime(2024, 1, 10, 20, 0),
        },
    ]

    return pd.DataFrame(sample_games)


def test_wind_features():
    """Test wind features calculation."""
    logger.info("Testing wind features calculation...")

    calculator = WeatherFeaturesCalculator()

    # Test different wind scenarios
    wind_scenarios = [
        {"wind_mph": 3.0, "description": "Calm conditions"},
        {"wind_mph": 10.0, "description": "Light wind"},
        {"wind_mph": 15.0, "description": "Moderate wind"},
        {"wind_mph": 25.0, "description": "Strong wind"},
        {"wind_mph": 40.0, "description": "Extreme wind"},
    ]

    for scenario in wind_scenarios:
        wind_features = calculator.calculate_wind_features(scenario)
        print(f"\n{scenario['description']} ({scenario['wind_mph']} mph):")
        for key, value in wind_features.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    logger.info("Wind features test completed")


def test_temperature_features():
    """Test temperature features calculation."""
    logger.info("Testing temperature features calculation...")

    calculator = WeatherFeaturesCalculator()

    # Test different temperature scenarios
    temp_scenarios = [
        {"temp_f": -5.0, "description": "Extreme cold"},
        {"temp_f": 15.0, "description": "Very cold"},
        {"temp_f": 35.0, "description": "Cold"},
        {"temp_f": 65.0, "description": "Comfortable"},
        {"temp_f": 85.0, "description": "Warm"},
        {"temp_f": 105.0, "description": "Extreme heat"},
    ]

    for scenario in temp_scenarios:
        temp_features = calculator.calculate_temperature_features(scenario)
        print(f"\n{scenario['description']} ({scenario['temp_f']}°F):")
        for key, value in temp_features.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    logger.info("Temperature features test completed")


def test_precipitation_features():
    """Test precipitation features calculation."""
    logger.info("Testing precipitation features calculation...")

    calculator = WeatherFeaturesCalculator()

    # Test different precipitation scenarios
    precip_scenarios = [
        {
            "precip_prob": 0.0,
            "precip_mm": 0.0,
            "temp_f": 65.0,
            "condition": "clear",
            "description": "Clear skies",
        },
        {
            "precip_prob": 0.3,
            "precip_mm": 1.0,
            "temp_f": 45.0,
            "condition": "light rain",
            "description": "Light rain",
        },
        {
            "precip_prob": 0.7,
            "precip_mm": 5.0,
            "temp_f": 38.0,
            "condition": "rain",
            "description": "Moderate rain",
        },
        {
            "precip_prob": 0.9,
            "precip_mm": 15.0,
            "temp_f": 28.0,
            "condition": "heavy snow",
            "description": "Heavy snow",
        },
        {
            "precip_prob": 0.8,
            "precip_mm": 10.0,
            "temp_f": 85.0,
            "condition": "thunderstorm",
            "description": "Thunderstorm",
        },
    ]

    for scenario in precip_scenarios:
        precip_features = calculator.calculate_precipitation_features(scenario)
        print(f"\n{scenario['description']}:")
        print(
            f"  Probability: {scenario['precip_prob']:.1%}, Amount: {scenario['precip_mm']} mm, Temp: {scenario['temp_f']}°F"
        )
        for key, value in precip_features.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.3f}")
            else:
                print(f"  {key}: {value}")

    logger.info("Precipitation features test completed")


def test_weather_severity():
    """Test overall weather severity calculation."""
    logger.info("Testing weather severity calculation...")

    calculator = WeatherFeaturesCalculator()

    # Test combined weather scenarios
    scenarios = [
        {
            "name": "Perfect weather",
            "wind": {"wind_mph": 2.0},
            "temp": {"temp_f": 72.0},
            "precip": {
                "precip_prob": 0.0,
                "precip_mm": 0.0,
                "temp_f": 72.0,
                "condition": "clear",
            },
        },
        {
            "name": "Typical fall game",
            "wind": {"wind_mph": 10.0},
            "temp": {"temp_f": 55.0},
            "precip": {
                "precip_prob": 0.2,
                "precip_mm": 0.0,
                "temp_f": 55.0,
                "condition": "partly cloudy",
            },
        },
        {
            "name": "Lambeau Field in January",
            "wind": {"wind_mph": 20.0},
            "temp": {"temp_f": 8.0},
            "precip": {
                "precip_prob": 0.6,
                "precip_mm": 3.0,
                "temp_f": 8.0,
                "condition": "snow",
            },
        },
    ]

    for scenario in scenarios:
        wind_features = calculator.calculate_wind_features(scenario["wind"])
        temp_features = calculator.calculate_temperature_features(scenario["temp"])
        precip_features = calculator.calculate_precipitation_features(
            scenario["precip"]
        )
        severity_features = calculator.calculate_weather_severity(
            wind_features, temp_features, precip_features
        )

        print(f"\n{scenario['name']}:")
        print(f"  Wind: {scenario['wind']['wind_mph']} mph")
        print(f"  Temperature: {scenario['temp']['temp_f']}°F")
        print(
            f"  Precipitation: {scenario['precip']['precip_prob']:.1%} chance, {scenario['precip']['precip_mm']} mm"
        )

        print("  Severity Scores:")
        for key, value in severity_features.items():
            print(f"    {key}: {value:.3f}")

    logger.info("Weather severity test completed")


def test_weather_features_pipeline():
    """Test end-to-end weather features pipeline using mock data."""
    logger.info("Testing weather features pipeline...")

    # Create mock weather data in memory
    weather_df = create_sample_weather_data()
    games_df = create_sample_games_data()

    print("\nSample weather scenarios:")
    for _, weather in weather_df.iterrows():
        game_id = weather["game_id"]
        outdoor = "Outdoor" if weather["is_outdoor"] else "Indoor"
        temp = weather["temp_f"]
        wind = weather["wind_mph"]
        precip = weather["precip_prob"] * 100
        print(
            f"  {game_id}: {outdoor}, {temp}°F, {wind} mph wind, {precip:.0f}% precip chance"
        )

    # Initialize calculator and manually process each game
    calculator = WeatherFeaturesCalculator()

    weather_features = []

    for _, game in games_df.iterrows():
        game_id = game["game_id"]

        # Find weather data for this game
        game_weather = weather_df[weather_df["game_id"] == game_id]

        if len(game_weather) == 0:
            logger.warning(f"No weather data found for {game_id}")
            continue

        weather_data = game_weather.iloc[0].to_dict()

        # Basic game info
        game_features = {
            "game_id": game_id,
            "season": game["season"],
            "week": game["week"],
        }

        # Check if weather affects this game
        is_outdoor = weather_data.get("is_outdoor", False)
        game_features["weather_affects_game"] = 1.0 if is_outdoor else 0.0

        if not is_outdoor:
            # Indoor game - no weather impact
            game_features.update(calculator._default_wind_features())
            game_features.update(calculator._default_temperature_features())
            game_features.update(calculator._default_precipitation_features())
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
            wind_features = calculator.calculate_wind_features(weather_data)
            temp_features = calculator.calculate_temperature_features(weather_data)
            precip_features = calculator.calculate_precipitation_features(weather_data)
            severity_features = calculator.calculate_weather_severity(
                wind_features, temp_features, precip_features
            )

            game_features.update(wind_features)
            game_features.update(temp_features)
            game_features.update(precip_features)
            game_features.update(severity_features)

        weather_features.append(game_features)

    # Convert to DataFrame
    features_df = pd.DataFrame(weather_features)

    print(f"\nGenerated weather features for {len(features_df)} games")
    print(f"Outdoor games: {int(features_df['weather_affects_game'].sum())}")
    print(f"Weather games (severity >= 0.6): {int(features_df['weather_game'].sum())}")
    print(
        f"Extreme weather games (severity >= 0.8): {int(features_df['extreme_weather'].sum())}"
    )

    # Show detailed results for outdoor games
    print("\nDetailed weather impact by game:")
    print("=" * 80)

    outdoor_games = features_df[features_df["weather_affects_game"] == 1.0]
    for _, game in outdoor_games.iterrows():
        game_weather = weather_df[weather_df["game_id"] == game["game_id"]].iloc[0]

        print(f"\n{game['game_id']} (Week {game['week']}):")
        print(
            f"  Conditions: {game_weather['temp_f']}°F, {game_weather['wind_mph']} mph wind, "
            f"{game_weather['precip_prob']:.1%} precip chance"
        )
        print(f"  Weather severity: {game['weather_severity_score']:.3f}")
        print(f"  Kicking difficulty: {game['kicking_difficulty']:.2f}x")
        print(f"  Passing efficiency: {game['passing_efficiency']:.3f}")
        print(f"  Scoring reduction: {game['scoring_reduction']:.1%}")

        if game["weather_game"] > 0:
            print("  *** WEATHER GAME ***")
        if game["extreme_weather"] > 0:
            print("  *** EXTREME WEATHER ***")

    # Validate features
    is_valid = calculator.validate_weather_features(features_df)
    print(f"\nValidation result: {'PASS' if is_valid else 'FAIL'}")

    logger.info("Weather features pipeline test completed")


def main():
    """Run all weather features tests."""
    print("=" * 60)
    print("WEATHER FEATURES CALCULATOR TEST")
    print("=" * 60)

    try:
        test_wind_features()
        print("\n" + "-" * 40)

        test_temperature_features()
        print("\n" + "-" * 40)

        test_precipitation_features()
        print("\n" + "-" * 40)

        test_weather_severity()
        print("\n" + "-" * 40)

        test_weather_features_pipeline()

        print("\n" + "=" * 60)
        print("ALL WEATHER FEATURES TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        logger.error("Test failed", error=str(e))
        print(f"\nTEST FAILED: {e}")
        raise


if __name__ == "__main__":
    main()
