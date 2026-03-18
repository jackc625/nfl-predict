"""Smoke test: nflreadpy can fetch 2024 NFL schedule data."""


def test_nflreadpy_loads_2024_schedule():
    """FOUN-01: nflreadpy returns game data for 2024 season."""
    import nflreadpy as nfl

    schedules = nfl.load_schedules([2024]).to_pandas()

    assert len(schedules) > 0, "No schedule data returned for 2024"
    assert "game_id" in schedules.columns or "gameday" in schedules.columns, (
        f"Expected game identifier column, got: {list(schedules.columns)[:10]}"
    )
    # 2024 season should have ~285 games (preseason + regular + postseason)
    assert len(schedules) >= 200, f"Expected 200+ games, got {len(schedules)}"


def test_nflreadpy_returns_pandas():
    """Verify .to_pandas() conversion works correctly."""
    import nflreadpy as nfl

    result = nfl.load_schedules([2024]).to_pandas()
    import pandas as pd

    assert isinstance(result, pd.DataFrame), (
        f"Expected pandas DataFrame, got {type(result)}"
    )
