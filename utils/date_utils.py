"""Date and time utilities for NFL data processing."""

import re
from datetime import datetime, timedelta
from typing import Optional, Tuple
import pytz
from dateutil import parser


# NFL season typically starts first Thursday in September
NFL_SEASON_START_MONTH = 9
NFL_SEASON_START_DAY_RANGE = (3, 10)  # First Thursday is between Sept 3-10

# Regular season is 18 weeks, playoffs add ~4 more weeks  
NFL_REGULAR_SEASON_WEEKS = 18
NFL_TOTAL_WEEKS = 22

# Time zones
ET = pytz.timezone('America/New_York')
UTC = pytz.UTC


def get_current_nfl_season() -> int:
    """
    Get the current NFL season year.
    
    The NFL season spans two calendar years. We use the year the season ends.
    For example, the 2024 season runs from Sept 2024 to Feb 2025.
    
    Returns:
        NFL season year
    """
    now = datetime.now(ET)
    
    # If it's January-July, we're in the previous season's playoffs/offseason
    if now.month <= 7:
        return now.year
    
    # If it's August-December, we're in the current season
    return now.year + 1


def get_nfl_season_start(season: int) -> datetime:
    """
    Get the approximate start date of an NFL season.
    
    Args:
        season: NFL season year
        
    Returns:
        Season start datetime (first Thursday in September)
    """
    # Start with September 1st of the previous calendar year
    year = season - 1
    sept_first = datetime(year, NFL_SEASON_START_MONTH, 1, tzinfo=ET)
    
    # Find first Thursday (weekday 3)
    days_to_thursday = (3 - sept_first.weekday()) % 7
    first_thursday = sept_first + timedelta(days=days_to_thursday)
    
    # If first Thursday is before Sept 3, move to next week
    if first_thursday.day < NFL_SEASON_START_DAY_RANGE[0]:
        first_thursday += timedelta(days=7)
    
    return first_thursday


def get_current_nfl_week() -> Tuple[int, int]:
    """
    Get the current NFL season and week.
    
    Returns:
        Tuple of (season, week) where week is 1-18 for regular season
    """
    season = get_current_nfl_season()
    season_start = get_nfl_season_start(season)
    now = datetime.now(ET)
    
    # Calculate weeks since season start
    if now < season_start:
        # We're before the season starts, return previous season's last week
        return season - 1, NFL_REGULAR_SEASON_WEEKS
    
    days_since_start = (now - season_start).days
    week = min(days_since_start // 7 + 1, NFL_TOTAL_WEEKS)
    
    return season, week


def parse_nfl_date(date_str: str) -> datetime:
    """
    Parse various NFL date formats into a datetime object.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Parsed datetime in ET timezone
    """
    # Common NFL date patterns
    patterns = [
        r'(\d{4})-(\d{2})-(\d{2})',  # YYYY-MM-DD
        r'(\d{2})/(\d{2})/(\d{4})',  # MM/DD/YYYY
        r'(\d{1,2})/(\d{1,2})/(\d{4})',  # M/D/YYYY
    ]
    
    # Try regex patterns first
    for pattern in patterns:
        match = re.match(pattern, date_str.strip())
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if pattern.startswith(r'(\d{4})'):  # YYYY-MM-DD
                    year, month, day = map(int, groups)
                else:  # MM/DD/YYYY or M/D/YYYY
                    month, day, year = map(int, groups)
                
                dt = datetime(year, month, day, tzinfo=ET)
                return dt
    
    # Fall back to dateutil parser
    try:
        dt = parser.parse(date_str)
        # Convert to ET if timezone-naive
        if dt.tzinfo is None:
            dt = ET.localize(dt)
        else:
            dt = dt.astimezone(ET)
        return dt
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unable to parse date string: {date_str}") from e


def is_game_time(kickoff_time: datetime, check_time: Optional[datetime] = None) -> bool:
    """
    Check if it's currently game time (within 4 hours of kickoff).
    
    Args:
        kickoff_time: Game kickoff time
        check_time: Time to check against (defaults to now)
        
    Returns:
        True if it's game time
    """
    if check_time is None:
        check_time = datetime.now(ET)
    
    # Ensure both times are in ET
    if kickoff_time.tzinfo != ET:
        kickoff_time = kickoff_time.astimezone(ET)
    if check_time.tzinfo != ET:
        check_time = check_time.astimezone(ET)
    
    time_diff = abs((kickoff_time - check_time).total_seconds() / 3600)  # hours
    return time_diff <= 4


def get_snapshot_time(date: Optional[datetime] = None, time_str: str = "Friday 18:00") -> datetime:
    """
    Get the snapshot time for a given week.
    
    Args:
        date: Reference date (defaults to now)
        time_str: Snapshot time specification (e.g., "Friday 18:00")
        
    Returns:
        Snapshot datetime in ET timezone
    """
    if date is None:
        date = datetime.now(ET)
    
    # Ensure date is in ET
    if date.tzinfo != ET:
        date = date.astimezone(ET)
    
    # Parse time specification
    day_time_pattern = r'(\w+day)\s+(\d{1,2}):(\d{2})'
    match = re.match(day_time_pattern, time_str)
    
    if not match:
        raise ValueError(f"Invalid time specification: {time_str}")
    
    day_name, hour_str, minute_str = match.groups()
    hour, minute = int(hour_str), int(minute_str)
    
    # Map day names to weekday numbers (Monday=0)
    day_map = {
        'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
        'friday': 4, 'saturday': 5, 'sunday': 6
    }
    
    target_weekday = day_map.get(day_name.lower())
    if target_weekday is None:
        raise ValueError(f"Invalid day name: {day_name}")
    
    # Find the target day in the current week
    current_weekday = date.weekday()
    days_ahead = target_weekday - current_weekday
    
    # If the target day has passed this week, get next week's
    if days_ahead < 0:
        days_ahead += 7
    
    target_date = date + timedelta(days=days_ahead)
    snapshot_time = target_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
    
    return snapshot_time


def get_week_start_end(season: int, week: int) -> Tuple[datetime, datetime]:
    """
    Get start and end times for a given NFL week.
    
    Args:
        season: NFL season year
        week: Week number (1-18 for regular season)
        
    Returns:
        Tuple of (week_start, week_end) in ET timezone
    """
    season_start = get_nfl_season_start(season)
    
    # Week 1 starts on season start date
    week_start = season_start + timedelta(weeks=week - 1)
    week_end = week_start + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    return week_start, week_end


def format_nfl_date(dt: datetime, include_time: bool = True) -> str:
    """
    Format a datetime for NFL display.
    
    Args:
        dt: Datetime to format
        include_time: Whether to include time
        
    Returns:
        Formatted date string
    """
    if dt.tzinfo != ET:
        dt = dt.astimezone(ET)
    
    if include_time:
        return dt.strftime("%Y-%m-%d %I:%M %p ET")
    else:
        return dt.strftime("%Y-%m-%d")


def get_rest_days(last_game_date: datetime, current_game_date: datetime) -> int:
    """
    Calculate rest days between games.
    
    Args:
        last_game_date: Date of previous game
        current_game_date: Date of current game
        
    Returns:
        Number of rest days
    """
    # Ensure both dates are in ET and date-only
    if last_game_date.tzinfo != ET:
        last_game_date = last_game_date.astimezone(ET)
    if current_game_date.tzinfo != ET:
        current_game_date = current_game_date.astimezone(ET)
    
    last_date = last_game_date.date()
    current_date = current_game_date.date()
    
    return (current_date - last_date).days


def is_short_week(rest_days: int) -> bool:
    """
    Determine if a game is on a short week.
    
    Args:
        rest_days: Number of rest days
        
    Returns:
        True if short week (< 6 days rest)
    """
    return rest_days < 6


def get_timezone_difference(home_timezone: str, away_timezone: str) -> float:
    """
    Calculate timezone difference for travel impact.
    
    Args:
        home_timezone: Home team timezone name
        away_timezone: Away team timezone name
        
    Returns:
        Time difference in hours (positive = away team traveling east)
    """
    try:
        home_tz = pytz.timezone(home_timezone)
        away_tz = pytz.timezone(away_timezone)
        
        # Use a reference time to calculate offset
        ref_time = datetime.now()
        home_offset = home_tz.localize(ref_time).utcoffset().total_seconds() / 3600
        away_offset = away_tz.localize(ref_time).utcoffset().total_seconds() / 3600
        
        return home_offset - away_offset
    except Exception:
        return 0.0  # Default to no difference if calculation fails