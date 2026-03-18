"""
NFL Game Utilities.

This module provides utilities for NFL game-specific logic including
week type determination, prime time game detection, and other game metadata.
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

# NFL week type constants
NFL_REGULAR_SEASON_WEEKS = 18
NFL_WILDCARD_WEEK = 19
NFL_DIVISIONAL_WEEK = 20
NFL_CONFERENCE_WEEK = 21
NFL_SUPER_BOWL_WEEK = 22


def determine_week_type(season: int, week: int) -> str:
    """
    Determine NFL week type based on season and week number.

    Args:
        season: NFL season year
        week: Week number (1-22)

    Returns:
        Week type: 'REG', 'WC', 'DIV', 'CONF', or 'SB'
    """
    if not week or week <= 0:
        return "REG"  # Default to regular season

    if week <= NFL_REGULAR_SEASON_WEEKS:
        return "REG"  # Regular season (weeks 1-18)
    if week == NFL_WILDCARD_WEEK:
        return "WC"  # Wild Card playoffs
    if week == NFL_DIVISIONAL_WEEK:
        return "DIV"  # Divisional playoffs
    if week == NFL_CONFERENCE_WEEK:
        return "CONF"  # Conference Championships
    if week == NFL_SUPER_BOWL_WEEK:
        return "SB"  # Super Bowl
    return "REG"  # Default fallback for any unusual week numbers


def is_prime_time_game(game_date: datetime, strict_mode: bool = False) -> bool:
    """
    Determine if a game is a prime time game based on kickoff time.

    Prime time games are typically:
    - Thursday Night Football (TNF): Thursday ~8:20 PM ET
    - Sunday Night Football (SNF): Sunday ~8:20 PM ET
    - Monday Night Football (MNF): Monday ~8:15 PM ET
    - Some Saturday playoff games

    Args:
        game_date: Game kickoff datetime
        strict_mode: If True, use exact prime time thresholds. If False, be more inclusive.

    Returns:
        True if the game is considered prime time
    """
    if not game_date:
        return False

    # Convert to ET for consistent time checking
    et_tz = ZoneInfo("America/New_York")

    if game_date.tzinfo is None:
        # Assume UTC if no timezone info
        utc_date = game_date.replace(tzinfo=UTC)
        game_et = utc_date.astimezone(et_tz)
    else:
        game_et = game_date.astimezone(et_tz)

    hour = game_et.hour
    minute = game_et.minute
    weekday = game_et.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday

    # Define prime time thresholds
    if strict_mode:
        # Strict: must be very close to actual prime time kickoffs
        prime_time_threshold = 20  # 8:00 PM
        late_threshold = 22  # 10:00 PM (too late for prime time)
    else:
        # Inclusive: any evening game could be considered prime time
        prime_time_threshold = 19  # 7:00 PM
        late_threshold = 23  # 11:00 PM

    # Check if time is in prime time window
    is_prime_time_hour = hour >= prime_time_threshold and hour < late_threshold

    # Or if it's exactly on the threshold, check minutes too
    if hour == (prime_time_threshold - 1) and minute >= 45:
        is_prime_time_hour = True

    if not is_prime_time_hour:
        return False

    # Thursday Night Football: Thursday evening games
    if weekday == 3:  # Thursday
        return True

    # Sunday Night Football: Sunday evening games
    if weekday == 6:  # Sunday
        return True

    # Monday Night Football: Monday evening games
    if weekday == 0:  # Monday
        return True

    # Saturday games (playoffs, late season, special cases)
    if weekday == 5:  # Saturday
        # Saturday games are often prime time during playoffs
        # or special late-season flex scheduling
        return True

    # International games or special cases
    # Some Tuesday/Wednesday games for makeup games could be prime time
    return weekday in [1, 2] and is_prime_time_hour


def get_game_day_type(game_date: datetime) -> str:
    """
    Get the type of day for the game.

    Returns:
        Day type: 'Thursday', 'Sunday', 'Monday', 'Saturday', 'Weekday', or 'Weekend'
    """
    if not game_date:
        return "Unknown"

    # Convert to ET
    et_tz = ZoneInfo("America/New_York")
    if game_date.tzinfo is None:
        utc_date = game_date.replace(tzinfo=UTC)
        game_et = utc_date.astimezone(et_tz)
    else:
        game_et = game_date.astimezone(et_tz)

    weekday = game_et.weekday()  # 0=Monday, 1=Tuesday, ..., 6=Sunday

    # Map to commonly understood NFL game days
    if weekday == 3:  # Thursday
        return "Thursday"
    if weekday == 6:  # Sunday
        return "Sunday"
    if weekday == 0:  # Monday
        return "Monday"
    if weekday == 5:  # Saturday
        return "Saturday"
    if weekday in [1, 2, 4]:  # Tuesday, Wednesday, Friday
        return "Weekday"
    return "Weekend"


def is_playoff_week(week: int) -> bool:
    """Check if the given week is a playoff week."""
    return week > NFL_REGULAR_SEASON_WEEKS


def is_regular_season_week(week: int) -> bool:
    """Check if the given week is a regular season week."""
    return 1 <= week <= NFL_REGULAR_SEASON_WEEKS


def get_week_type_description(week_type: str) -> str:
    """Get human-readable description of week type."""
    descriptions = {
        "REG": "Regular Season",
        "WC": "Wild Card Playoffs",
        "DIV": "Divisional Playoffs",
        "CONF": "Conference Championships",
        "SB": "Super Bowl",
    }
    return descriptions.get(week_type, "Regular Season")


def is_short_week_game(
    game_date: datetime, last_game_date: datetime | None = None
) -> bool:
    """
    Determine if this is a "short week" game (less than 7 days rest).

    Args:
        game_date: Current game datetime
        last_game_date: Previous game datetime (optional)

    Returns:
        True if it's a short week game
    """
    if not game_date:
        return False

    # If we don't have the last game date, use heuristics based on day of week
    if not last_game_date:
        weekday = game_date.weekday()
        # Thursday games are typically short week (from previous Sunday)
        # Monday games could be short week depending on previous game
        return weekday == 3  # Thursday

    # Calculate days between games
    days_between = (game_date - last_game_date).days
    return days_between < 7


def normalize_game_time_to_et(game_date: datetime) -> datetime:
    """
    Normalize game datetime to Eastern Time for consistent processing.

    Args:
        game_date: Game datetime in any timezone

    Returns:
        Game datetime converted to Eastern Time
    """
    if not game_date:
        return game_date

    et_tz = ZoneInfo("America/New_York")

    if game_date.tzinfo is None:
        # Assume UTC if no timezone info, then convert to ET
        utc_date = game_date.replace(tzinfo=UTC)
        return utc_date.astimezone(et_tz)
    return game_date.astimezone(et_tz)


def get_game_time_slot(game_date: datetime) -> str:
    """
    Categorize game into time slots for analysis.

    Returns:
        Time slot: 'Early', 'Afternoon', 'Prime Time', 'Late Night'
    """
    if not game_date:
        return "Unknown"

    game_et = normalize_game_time_to_et(game_date)
    hour = game_et.hour

    if hour < 13:  # Before 1 PM ET
        return "Early"
    if hour < 17:  # 1 PM - 4 PM ET
        return "Afternoon"
    if hour < 22:  # 5 PM - 9 PM ET
        return "Prime Time"
    # 10 PM ET and later
    return "Late Night"
