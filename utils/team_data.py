"""
NFL Team Data and Information Utilities.

This module provides comprehensive team information including full names,
cities, conferences, divisions, and other metadata for all 32 NFL teams.
"""

from typing import Any

# Comprehensive NFL team data with all current teams (2024 season)
NFL_TEAMS_DATA = {
    # AFC East
    "BUF": {
        "name": "Buffalo Bills",
        "city": "Buffalo",
        "state": "NY",
        "conference": "AFC",
        "division": "East",
        "full_name": "Buffalo Bills",
        "abbreviations": ["BUF"],
        "colors": ["#00338D", "#C60C30"],
        "founded": 1960,
    },
    "MIA": {
        "name": "Miami Dolphins",
        "city": "Miami",
        "state": "FL",
        "conference": "AFC",
        "division": "East",
        "full_name": "Miami Dolphins",
        "abbreviations": ["MIA"],
        "colors": ["#008E97", "#FC4C02"],
        "founded": 1966,
    },
    "NE": {
        "name": "New England Patriots",
        "city": "Foxborough",
        "state": "MA",
        "conference": "AFC",
        "division": "East",
        "full_name": "New England Patriots",
        "abbreviations": ["NE", "NWE"],
        "colors": ["#002244", "#C60C30"],
        "founded": 1960,
    },
    "NYJ": {
        "name": "New York Jets",
        "city": "East Rutherford",
        "state": "NJ",
        "conference": "AFC",
        "division": "East",
        "full_name": "New York Jets",
        "abbreviations": ["NYJ"],
        "colors": ["#125740", "#FFFFFF"],
        "founded": 1960,
    },
    # AFC North
    "BAL": {
        "name": "Baltimore Ravens",
        "city": "Baltimore",
        "state": "MD",
        "conference": "AFC",
        "division": "North",
        "full_name": "Baltimore Ravens",
        "abbreviations": ["BAL"],
        "colors": ["#241773", "#000000"],
        "founded": 1996,
    },
    "CIN": {
        "name": "Cincinnati Bengals",
        "city": "Cincinnati",
        "state": "OH",
        "conference": "AFC",
        "division": "North",
        "full_name": "Cincinnati Bengals",
        "abbreviations": ["CIN"],
        "colors": ["#FB4F14", "#000000"],
        "founded": 1968,
    },
    "CLE": {
        "name": "Cleveland Browns",
        "city": "Cleveland",
        "state": "OH",
        "conference": "AFC",
        "division": "North",
        "full_name": "Cleveland Browns",
        "abbreviations": ["CLE", "CLV"],
        "colors": ["#311D00", "#FF3C00"],
        "founded": 1946,
    },
    "PIT": {
        "name": "Pittsburgh Steelers",
        "city": "Pittsburgh",
        "state": "PA",
        "conference": "AFC",
        "division": "North",
        "full_name": "Pittsburgh Steelers",
        "abbreviations": ["PIT"],
        "colors": ["#FFB612", "#000000"],
        "founded": 1933,
    },
    # AFC South
    "HOU": {
        "name": "Houston Texans",
        "city": "Houston",
        "state": "TX",
        "conference": "AFC",
        "division": "South",
        "full_name": "Houston Texans",
        "abbreviations": ["HOU", "HST"],
        "colors": ["#03202F", "#A71930"],
        "founded": 2002,
    },
    "IND": {
        "name": "Indianapolis Colts",
        "city": "Indianapolis",
        "state": "IN",
        "conference": "AFC",
        "division": "South",
        "full_name": "Indianapolis Colts",
        "abbreviations": ["IND"],
        "colors": ["#002C5F", "#A2AAAD"],
        "founded": 1953,
    },
    "JAX": {
        "name": "Jacksonville Jaguars",
        "city": "Jacksonville",
        "state": "FL",
        "conference": "AFC",
        "division": "South",
        "full_name": "Jacksonville Jaguars",
        "abbreviations": ["JAX", "JAC"],
        "colors": ["#006778", "#9F792C"],
        "founded": 1995,
    },
    "TEN": {
        "name": "Tennessee Titans",
        "city": "Nashville",
        "state": "TN",
        "conference": "AFC",
        "division": "South",
        "full_name": "Tennessee Titans",
        "abbreviations": ["TEN"],
        "colors": ["#0C2340", "#4B92DB"],
        "founded": 1960,
    },
    # AFC West
    "DEN": {
        "name": "Denver Broncos",
        "city": "Denver",
        "state": "CO",
        "conference": "AFC",
        "division": "West",
        "full_name": "Denver Broncos",
        "abbreviations": ["DEN"],
        "colors": ["#FB4F14", "#002244"],
        "founded": 1960,
    },
    "KC": {
        "name": "Kansas City Chiefs",
        "city": "Kansas City",
        "state": "MO",
        "conference": "AFC",
        "division": "West",
        "full_name": "Kansas City Chiefs",
        "abbreviations": ["KC"],
        "colors": ["#E31837", "#FFB612"],
        "founded": 1960,
    },
    "LV": {
        "name": "Las Vegas Raiders",
        "city": "Las Vegas",
        "state": "NV",
        "conference": "AFC",
        "division": "West",
        "full_name": "Las Vegas Raiders",
        "abbreviations": ["LV", "LVR", "OAK"],  # Include Oakland for historical data
        "colors": ["#000000", "#A5ACAF"],
        "founded": 1960,
    },
    "LAC": {
        "name": "Los Angeles Chargers",
        "city": "Los Angeles",
        "state": "CA",
        "conference": "AFC",
        "division": "West",
        "full_name": "Los Angeles Chargers",
        "abbreviations": ["LAC", "SD"],  # Include San Diego for historical data
        "colors": ["#0080C6", "#FFC20E"],
        "founded": 1960,
    },
    # NFC East
    "DAL": {
        "name": "Dallas Cowboys",
        "city": "Arlington",
        "state": "TX",
        "conference": "NFC",
        "division": "East",
        "full_name": "Dallas Cowboys",
        "abbreviations": ["DAL"],
        "colors": ["#003594", "#041E42"],
        "founded": 1960,
    },
    "NYG": {
        "name": "New York Giants",
        "city": "East Rutherford",
        "state": "NJ",
        "conference": "NFC",
        "division": "East",
        "full_name": "New York Giants",
        "abbreviations": ["NYG"],
        "colors": ["#0B2265", "#A71930"],
        "founded": 1925,
    },
    "PHI": {
        "name": "Philadelphia Eagles",
        "city": "Philadelphia",
        "state": "PA",
        "conference": "NFC",
        "division": "East",
        "full_name": "Philadelphia Eagles",
        "abbreviations": ["PHI"],
        "colors": ["#004C54", "#A5ACAF"],
        "founded": 1933,
    },
    "WAS": {
        "name": "Washington Commanders",
        "city": "Landover",
        "state": "MD",
        "conference": "NFC",
        "division": "East",
        "full_name": "Washington Commanders",
        "abbreviations": ["WAS", "WSH"],
        "colors": ["#5A1414", "#FFB612"],
        "founded": 1932,
    },
    # NFC North
    "CHI": {
        "name": "Chicago Bears",
        "city": "Chicago",
        "state": "IL",
        "conference": "NFC",
        "division": "North",
        "full_name": "Chicago Bears",
        "abbreviations": ["CHI"],
        "colors": ["#0B162A", "#C83803"],
        "founded": 1920,
    },
    "DET": {
        "name": "Detroit Lions",
        "city": "Detroit",
        "state": "MI",
        "conference": "NFC",
        "division": "North",
        "full_name": "Detroit Lions",
        "abbreviations": ["DET"],
        "colors": ["#0076B6", "#B0B7BC"],
        "founded": 1930,
    },
    "GB": {
        "name": "Green Bay Packers",
        "city": "Green Bay",
        "state": "WI",
        "conference": "NFC",
        "division": "North",
        "full_name": "Green Bay Packers",
        "abbreviations": ["GB", "GNB"],
        "colors": ["#203731", "#FFB612"],
        "founded": 1919,
    },
    "MIN": {
        "name": "Minnesota Vikings",
        "city": "Minneapolis",
        "state": "MN",
        "conference": "NFC",
        "division": "North",
        "full_name": "Minnesota Vikings",
        "abbreviations": ["MIN"],
        "colors": ["#4F2683", "#FFC62F"],
        "founded": 1961,
    },
    # NFC South
    "ATL": {
        "name": "Atlanta Falcons",
        "city": "Atlanta",
        "state": "GA",
        "conference": "NFC",
        "division": "South",
        "full_name": "Atlanta Falcons",
        "abbreviations": ["ATL"],
        "colors": ["#A71930", "#000000"],
        "founded": 1966,
    },
    "CAR": {
        "name": "Carolina Panthers",
        "city": "Charlotte",
        "state": "NC",
        "conference": "NFC",
        "division": "South",
        "full_name": "Carolina Panthers",
        "abbreviations": ["CAR"],
        "colors": ["#0085CA", "#000000"],
        "founded": 1995,
    },
    "NO": {
        "name": "New Orleans Saints",
        "city": "New Orleans",
        "state": "LA",
        "conference": "NFC",
        "division": "South",
        "full_name": "New Orleans Saints",
        "abbreviations": ["NO", "NOR"],
        "colors": ["#D3BC8D", "#000000"],
        "founded": 1967,
    },
    "TB": {
        "name": "Tampa Bay Buccaneers",
        "city": "Tampa",
        "state": "FL",
        "conference": "NFC",
        "division": "South",
        "full_name": "Tampa Bay Buccaneers",
        "abbreviations": ["TB", "TAM"],
        "colors": ["#D50A0A", "#FF7900"],
        "founded": 1976,
    },
    # NFC West
    "ARI": {
        "name": "Arizona Cardinals",
        "city": "Glendale",
        "state": "AZ",
        "conference": "NFC",
        "division": "West",
        "full_name": "Arizona Cardinals",
        "abbreviations": ["ARI"],
        "colors": ["#97233F", "#000000"],
        "founded": 1898,
    },
    "LA": {
        "name": "Los Angeles Rams",
        "city": "Los Angeles",
        "state": "CA",
        "conference": "NFC",
        "division": "West",
        "full_name": "Los Angeles Rams",
        "abbreviations": ["LA", "LAR", "STL", "SL"],
        "colors": ["#003594", "#FFA300"],
        "founded": 1937,
    },
    "SF": {
        "name": "San Francisco 49ers",
        "city": "Santa Clara",
        "state": "CA",
        "conference": "NFC",
        "division": "West",
        "full_name": "San Francisco 49ers",
        "abbreviations": ["SF", "SFO"],
        "colors": ["#AA0000", "#B3995D"],
        "founded": 1946,
    },
    "SEA": {
        "name": "Seattle Seahawks",
        "city": "Seattle",
        "state": "WA",
        "conference": "NFC",
        "division": "West",
        "full_name": "Seattle Seahawks",
        "abbreviations": ["SEA"],
        "colors": ["#002244", "#69BE28"],
        "founded": 1976,
    },
}


# Create reverse mapping for abbreviation lookup
_ABBREVIATION_MAP = {}
for team_id, data in NFL_TEAMS_DATA.items():
    # Add primary abbreviation
    _ABBREVIATION_MAP[team_id] = team_id
    # Add all alternative abbreviations
    for abbr in data["abbreviations"]:
        _ABBREVIATION_MAP[abbr.upper()] = team_id


def get_team_info(team_abbr: str) -> dict[str, Any]:
    """
    Get comprehensive team information by abbreviation.

    Args:
        team_abbr: Team abbreviation (e.g., 'KC', 'SF', 'NE')

    Returns:
        Dictionary with team information or default values if not found
    """
    if not team_abbr:
        return _get_default_team_info(team_abbr)

    # Normalize and lookup
    normalized_abbr = team_abbr.strip().upper()
    canonical_id = _ABBREVIATION_MAP.get(normalized_abbr)

    if canonical_id and canonical_id in NFL_TEAMS_DATA:
        return NFL_TEAMS_DATA[canonical_id].copy()

    return _get_default_team_info(team_abbr)


def _get_default_team_info(team_abbr: str) -> dict[str, Any]:
    """Return default team info for unknown teams."""
    return {
        "name": team_abbr,
        "city": team_abbr,
        "state": "Unknown",
        "conference": "Unknown",
        "division": "Unknown",
        "full_name": team_abbr,
        "abbreviations": [team_abbr],
        "colors": ["#000000", "#FFFFFF"],
        "founded": None,
    }


def get_team_conference(team_abbr: str) -> str:
    """Get team conference (AFC/NFC)."""
    team_info = get_team_info(team_abbr)
    return team_info.get("conference", "Unknown")


def get_team_division(team_abbr: str) -> str:
    """Get team division (East/North/South/West)."""
    team_info = get_team_info(team_abbr)
    return team_info.get("division", "Unknown")


def get_team_full_name(team_abbr: str) -> str:
    """Get team full name."""
    team_info = get_team_info(team_abbr)
    return team_info.get("name", team_abbr)


def get_all_teams() -> list[str]:
    """Get list of all valid team abbreviations."""
    return list(NFL_TEAMS_DATA.keys())


def get_teams_by_conference(conference: str) -> list[str]:
    """Get all teams in a conference."""
    conference = conference.upper()
    return [
        team_id
        for team_id, data in NFL_TEAMS_DATA.items()
        if data["conference"].upper() == conference
    ]


def get_teams_by_division(conference: str, division: str) -> list[str]:
    """Get all teams in a division."""
    conference = conference.upper()
    division = division.title()
    return [
        team_id
        for team_id, data in NFL_TEAMS_DATA.items()
        if data["conference"].upper() == conference and data["division"] == division
    ]


def normalize_team_abbreviation(team_abbr: str) -> str:
    """
    Normalize team abbreviation to canonical form.

    Raises DataValidationError for unknown abbreviations with closest-match
    suggestions. This is intentionally strict -- silent pass-through of
    unknown abbreviations causes data corruption downstream.

    Args:
        team_abbr: Any valid team abbreviation

    Returns:
        Canonical team abbreviation (e.g., 'OAK' -> 'LV', 'LAR' -> 'LA')

    Raises:
        DataValidationError: If abbreviation is empty or not recognized
    """
    from utils.exceptions import DataValidationError

    if not team_abbr or not team_abbr.strip():
        raise DataValidationError("Team abbreviation cannot be empty")

    normalized = team_abbr.strip().upper()
    canonical = _ABBREVIATION_MAP.get(normalized)

    if canonical is None:
        from difflib import get_close_matches

        suggestions = get_close_matches(
            normalized, list(_ABBREVIATION_MAP.keys()), n=3, cutoff=0.4
        )
        suggestion_text = (
            f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        )
        raise DataValidationError(
            f"Unknown team abbreviation: '{team_abbr}'.{suggestion_text}"
        )

    return canonical


def validate_team_abbreviation(team_abbr: str) -> bool:
    """Check if team abbreviation is valid."""
    if not team_abbr:
        return False

    normalized = team_abbr.strip().upper()
    return normalized in _ABBREVIATION_MAP


def get_all_teams_info() -> dict[str, dict[str, Any]]:
    """Get information for all NFL teams."""
    return NFL_TEAMS_DATA.copy()


# Export commonly used team sets
AFC_TEAMS = get_teams_by_conference("AFC")
NFC_TEAMS = get_teams_by_conference("NFC")
ALL_TEAMS = get_all_teams()
