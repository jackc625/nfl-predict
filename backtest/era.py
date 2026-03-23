"""Era normalization and COVID annotation utilities.

Handles the NFL's transition from 16 games (17 regular-season weeks) to
17 games (18 regular-season weeks) starting in 2021, and provides
contextual annotations about the COVID-affected 2020 season.

Provides:
- normalize_week_to_progress: Normalize week numbers across eras
- get_season_total_weeks: Total regular-season weeks per season
- get_covid_hfa_annotation: COVID-2020 home field advantage context
"""

from __future__ import annotations

# -----------------------------------------------------------------------
# Era Constants
# -----------------------------------------------------------------------

ERA_TRANSITION_SEASON = 2021
"""First season with 17 games (18 regular-season weeks)."""

SEASON_TOTAL_WEEKS: dict[int, int] = {
    2018: 17,
    2019: 17,
    2020: 17,
    2021: 18,
    2022: 18,
    2023: 18,
    2024: 18,
}
"""Number of regular-season weeks per season.

Pre-2021: 16 games played over 17 weeks (each team has 1 bye).
2021+: 17 games played over 18 weeks (each team has 1 bye).
"""


# -----------------------------------------------------------------------
# Public Functions
# -----------------------------------------------------------------------


def get_season_total_weeks(season: int) -> int:
    """Return the total number of regular-season weeks for a given season.

    Args:
        season: NFL season year.

    Returns:
        17 for seasons before 2021, 18 for 2021 and later.
    """
    if season in SEASON_TOTAL_WEEKS:
        return SEASON_TOTAL_WEEKS[season]
    # Fall back to era-based logic for seasons outside the dict
    return 18 if season >= ERA_TRANSITION_SEASON else 17


def normalize_week_to_progress(week: int, season: int) -> float:
    """Normalize a week number to a season-progress fraction.

    Converts week numbers to a [0, 1] scale so that weeks from different
    eras are comparable. Week 12 in a 17-week season (70.6% through) is
    different from week 12 in an 18-week season (66.7% through).

    Args:
        week: Game week number (1-indexed).
        season: NFL season year.

    Returns:
        Season progress as a float in (0, 1].
    """
    total_weeks = get_season_total_weeks(season)
    return week / total_weeks


def get_covid_hfa_annotation() -> dict:
    """Return contextual annotation about the COVID-affected 2020 season.

    Provides factual context about how COVID-19 impacted home-field
    advantage in 2020, useful for interpreting backtest results that
    include 2020 data.

    Returns:
        Dict with keys: seasons, home_win_pct, normal_home_win_pct,
        note, impact_on_model.
    """
    return {
        "seasons": [2020],
        "home_win_pct": 0.496,
        "normal_home_win_pct": 0.57,
        "note": (
            "COVID-19 empty/limited stadiums reduced home field advantage. "
            "2020 home teams went 127-128-1 (49.6%) vs historical ~57%."
        ),
        "impact_on_model": (
            "2020 used as HP-validation window; HFA parameter set to "
            "48 Elo points (down from 65) based on post-2020 research."
        ),
    }
