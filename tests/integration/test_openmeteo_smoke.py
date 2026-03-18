"""Smoke test: Open-Meteo returns weather data for a known NFL venue."""

import httpx


def test_openmeteo_returns_weather_for_metlife():
    """FOUN-02: Open-Meteo returns weather for MetLife Stadium (Giants/Jets)."""
    # MetLife Stadium coordinates: East Rutherford, NJ
    params = {
        "latitude": 40.8128,
        "longitude": -74.0742,
        "start_date": "2024-09-08",  # Week 1, 2024 season
        "end_date": "2024-09-08",
        "hourly": "temperature_2m,wind_speed_10m,precipitation",
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/New_York",
    }

    response = httpx.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=30.0,
    )

    assert response.status_code == 200, f"API returned {response.status_code}"
    data = response.json()
    assert "hourly" in data, f"Response missing 'hourly' key: {list(data.keys())}"
    assert "temperature_2m" in data["hourly"], "Missing temperature_2m in hourly data"
    assert "wind_speed_10m" in data["hourly"], "Missing wind_speed_10m in hourly data"
    assert len(data["hourly"]["temperature_2m"]) == 24, (
        f"Expected 24 hourly values, got {len(data['hourly']['temperature_2m'])}"
    )


def test_openmeteo_returns_all_required_fields():
    """Verify all weather fields needed by the system are available."""
    required_fields = [
        "temperature_2m",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "snowfall",
        "weather_code",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "wind_gusts_10m",
    ]
    params = {
        "latitude": 40.8128,
        "longitude": -74.0742,
        "start_date": "2024-01-07",
        "end_date": "2024-01-07",
        "hourly": ",".join(required_fields),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": "America/New_York",
    }

    response = httpx.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params=params,
        timeout=30.0,
    )

    assert response.status_code == 200
    data = response.json()
    for field in required_fields:
        assert field in data["hourly"], f"Missing required field: {field}"
