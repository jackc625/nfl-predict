"""
Game ID Utilities for NFL Prediction System.

This module provides standardized game ID creation and validation to ensure
consistency across all ingestion scripts and data processing components.

Standard Format: {season}_W{week:02d}_{away_team}@{home_team}
Examples: 2024_W10_KC@BUF, 2024_W22_KC@PHI
"""

import contextlib
import re
from typing import Any

from utils import get_logger
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# Standard game ID pattern
GAME_ID_PATTERN = re.compile(r"^(\d{4})_W(\d{2})_([A-Z]{2,3})@([A-Z]{2,3})$")

# Full team name / city name mappings that resolve through normalize_team_abbreviation.
# These handle the case where odds APIs or other sources send full names like
# "Kansas City Chiefs" or "Los Angeles Rams" instead of abbreviations.
_FULL_NAME_MAP = {
    "KANSAS_CITY": "KC",
    "LAS_VEGAS": "LV",
    "LOS_ANGELES": "LA",
    "NEW_ENGLAND": "NE",
    "NEW_ORLEANS": "NO",
    "NEW_YORK_GIANTS": "NYG",
    "NEW_YORK_JETS": "NYJ",
    "SAN_FRANCISCO": "SF",
    "TAMPA_BAY": "TB",
    "GREEN_BAY": "GB",
    "KANSAS_CITY_CHIEFS": "KC",
    "BUFFALO_BILLS": "BUF",
    "MIAMI_DOLPHINS": "MIA",
    "NEW_ENGLAND_PATRIOTS": "NE",
    "BALTIMORE_RAVENS": "BAL",
    "CINCINNATI_BENGALS": "CIN",
    "CLEVELAND_BROWNS": "CLE",
    "PITTSBURGH_STEELERS": "PIT",
    "HOUSTON_TEXANS": "HOU",
    "INDIANAPOLIS_COLTS": "IND",
    "JACKSONVILLE_JAGUARS": "JAX",
    "TENNESSEE_TITANS": "TEN",
    "DENVER_BRONCOS": "DEN",
    "LAS_VEGAS_RAIDERS": "LV",
    "LOS_ANGELES_CHARGERS": "LAC",
    "LOS_ANGELES_RAMS": "LA",
    "CHICAGO_BEARS": "CHI",
    "DETROIT_LIONS": "DET",
    "GREEN_BAY_PACKERS": "GB",
    "MINNESOTA_VIKINGS": "MIN",
    "ATLANTA_FALCONS": "ATL",
    "CAROLINA_PANTHERS": "CAR",
    "NEW_ORLEANS_SAINTS": "NO",
    "TAMPA_BAY_BUCCANEERS": "TB",
    "DALLAS_COWBOYS": "DAL",
    "PHILADELPHIA_EAGLES": "PHI",
    "WASHINGTON_COMMANDERS": "WAS",
    "ARIZONA_CARDINALS": "ARI",
    "SAN_FRANCISCO_49ERS": "SF",
    "SEATTLE_SEAHAWKS": "SEA",
}


def normalize_team_name(team: str) -> str:
    """
    Normalize team name to canonical abbreviation.

    Handles both short abbreviations (delegated to utils.team_data) and
    full team names / city names (looked up in _FULL_NAME_MAP).

    Args:
        team: Team name or abbreviation in any format

    Returns:
        Canonical team abbreviation (e.g., 'KC', 'BUF', 'LA')

    Raises:
        ValueError: If team name cannot be normalized
    """
    from utils.exceptions import DataValidationError

    if not team:
        raise ValueError("Team name cannot be empty")

    team_clean = team.upper().strip().replace(" ", "_")

    # Try abbreviation lookup first (delegates to team_data.py)
    try:
        return normalize_team_abbreviation(team_clean)
    except DataValidationError:
        pass

    # Try without underscores (e.g. "KANSASCITY" -> not useful, but short forms)
    team_no_underscore = team_clean.replace("_", "")
    try:
        return normalize_team_abbreviation(team_no_underscore)
    except DataValidationError:
        pass

    # Try full team name / city name lookup
    if team_clean in _FULL_NAME_MAP:
        return normalize_team_abbreviation(_FULL_NAME_MAP[team_clean])

    raise DataValidationError(f"Cannot normalize team name: '{team}'")


def create_standard_game_id(
    season: int, week: int, away_team: str, home_team: str
) -> str:
    """
    Create standardized game ID.

    Args:
        season: NFL season year (e.g., 2024)
        week: NFL week number (1-22)
        away_team: Away team name/abbreviation
        home_team: Home team name/abbreviation

    Returns:
        Standardized game ID: {season}_W{week:02d}_{away_team}@{home_team}

    Example:
        create_standard_game_id(2024, 10, 'KC', 'BUF') -> '2024_W10_KC@BUF'
    """
    # Normalize team names
    away_norm = normalize_team_name(away_team)
    home_norm = normalize_team_name(home_team)

    # Validate inputs
    if not (2000 <= season <= 2100):
        raise ValueError(f"Invalid season: {season}")
    if not (1 <= week <= 22):
        raise ValueError(f"Invalid week: {week}")

    game_id = f"{season}_W{week:02d}_{away_norm}@{home_norm}"

    # Validate the result
    if not is_valid_game_id(game_id):
        raise ValueError(f"Generated invalid game ID: {game_id}")

    return game_id


def is_valid_game_id(game_id: str) -> bool:
    """
    Validate game ID format.

    Args:
        game_id: Game ID to validate

    Returns:
        True if valid, False otherwise
    """
    if not game_id or not isinstance(game_id, str):
        return False

    return bool(GAME_ID_PATTERN.match(game_id))


def parse_game_id(game_id: str) -> dict[str, Any]:
    """
    Parse game ID into components.

    Args:
        game_id: Standard format game ID

    Returns:
        Dictionary with season, week, away_team, home_team

    Raises:
        ValueError: If game ID format is invalid
    """
    match = GAME_ID_PATTERN.match(game_id)
    if not match:
        raise ValueError(f"Invalid game ID format: {game_id}")

    return {
        "season": int(match.group(1)),
        "week": int(match.group(2)),
        "away_team": match.group(3),
        "home_team": match.group(4),
    }


def convert_legacy_game_id(legacy_id: str) -> str:
    """
    Convert legacy game ID formats to standard format.

    Handles formats like:
    - 2024_10_KC_BUF -> 2024_W10_KC@BUF
    - 2024_W10_KC_BUF -> 2024_W10_KC@BUF

    Args:
        legacy_id: Legacy format game ID

    Returns:
        Standard format game ID

    Raises:
        ValueError: If legacy ID cannot be converted
    """
    if is_valid_game_id(legacy_id):
        return legacy_id  # Already in standard format

    # Try to parse legacy formats
    parts = legacy_id.split("_")

    if len(parts) == 4:
        season_str, week_str, away_team, home_team = parts

        try:
            season = int(season_str)

            # Handle week format (may have 'W' prefix or not)
            week = int(week_str[1:]) if week_str.startswith("W") else int(week_str)

            return create_standard_game_id(season, week, away_team, home_team)

        except ValueError as e:
            raise ValueError(f"Cannot convert legacy game ID '{legacy_id}': {e}")

    raise ValueError(f"Unrecognized game ID format: {legacy_id}")


def validate_team_mapping() -> bool:
    """
    Validate that all team mappings are consistent.

    Delegates to utils.team_data as the single source of truth.

    Returns:
        True if all mappings are valid
    """
    from utils.team_data import get_all_teams

    teams = get_all_teams()
    for canonical in teams:
        if len(canonical) not in [2, 3] or not canonical.isalpha():
            logger.error(f"Invalid canonical team abbreviation: {canonical}")
            return False

    logger.info(f"Team mapping validation passed: {len(teams)} valid teams")
    return True


def get_canonical_teams() -> set:
    """
    Get set of all canonical team abbreviations.

    Returns:
        Set of canonical team abbreviations
    """
    from utils.team_data import get_all_teams

    return set(get_all_teams())


# Validation function for use in ingestion scripts
def validate_game_data(
    season: int, week: int, home_team: str, away_team: str
) -> tuple[bool, str]:
    """
    Validate game data before creating game ID.

    Args:
        season: NFL season
        week: NFL week
        home_team: Home team
        away_team: Away team

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        # This will raise ValueError if invalid
        create_standard_game_id(season, week, away_team, home_team)
        return True, ""
    except ValueError as e:
        return False, str(e)


if __name__ == "__main__":
    # Run validation
    validate_team_mapping()

    # Test examples
    test_cases = [
        (2024, 10, "KC", "BUF"),
        (2024, 22, "Kansas City", "Buffalo Bills"),
        (2024, 1, "TB", "DAL"),
    ]

    for season, week, away, home in test_cases:
        with contextlib.suppress(Exception):
            game_id = create_standard_game_id(season, week, away, home)
