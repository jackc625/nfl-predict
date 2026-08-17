"""Date and time utilities for NFL data processing."""

import re
from datetime import UTC as _UTC
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from dateutil import parser

# NFL season starts the Thursday after Labor Day (first Monday in September)
NFL_SEASON_START_MONTH = 9
NFL_SEASON_START_DAY_RANGE = (4, 10)  # Kickoff Thursday falls between Sept 4-10

# Regular season is 18 weeks, playoffs add ~4 more weeks
NFL_REGULAR_SEASON_WEEKS = 18
NFL_TOTAL_WEEKS = 22

# Time zones
ET = ZoneInfo("America/New_York")
UTC = _UTC


def get_current_nfl_season(now: datetime | None = None) -> int:
    """
    Get the current NFL season year.

    The NFL season spans two calendar years. We use the year the season starts.
    For example, the 2024 season runs from Sept 2024 to Feb 2025.

    Args:
        now: Reference instant, for testing against a frozen clock. Defaults to
            ``datetime.now(ET)``. Purely additive -- every existing call site
            passes nothing and behaves identically.

    Returns:
        NFL season year
    """
    if now is None:
        now = datetime.now(ET)

    # If it's January-July, we're in the previous season's playoffs/offseason
    if now.month <= 7:
        return now.year - 1

    # If it's August-December, we're in the current season that started this year
    return now.year


def get_nfl_season_start(season: int) -> datetime:
    """
    Get the kickoff date of an NFL season.

    The season opens on the Thursday following Labor Day (the first Monday in
    September), so kickoff always lands between Sept 4 and Sept 10.

    A plain "first Thursday in September" rule agrees with this in most years
    but diverges whenever Sept 1 falls on a Tuesday: the first Thursday
    (Sept 3) then precedes Labor Day (Sept 7), and every week number derived
    from it runs a week high for the entire season. That hits 2020 and 2026.

    Args:
        season: NFL season year

    Returns:
        Season kickoff datetime (Thursday after Labor Day, ET)
    """
    sept_first = datetime(season, NFL_SEASON_START_MONTH, 1, tzinfo=ET)

    # Labor Day is the first Monday in September (weekday 0).
    labor_day = sept_first + timedelta(days=(0 - sept_first.weekday()) % 7)

    # Kickoff Thursday is three days after Labor Day.
    kickoff = labor_day + timedelta(days=3)

    earliest, latest = NFL_SEASON_START_DAY_RANGE
    if not earliest <= kickoff.day <= latest:
        raise ValueError(
            f"Derived NFL kickoff {kickoff.date()} for season {season} falls "
            f"outside the expected Sept {earliest}-{latest} window"
        )

    return kickoff


def get_current_nfl_week(now: datetime | None = None) -> tuple[int, int]:
    """
    Get the current NFL season and week based on when games actually finish.

    NFL weeks transition on Tuesday after the last game of the previous week
    (typically Monday Night Football). This ensures consistency with betting
    markets and data availability.

    WR-05 correction. The previous implementation returned a week ONE TOO HIGH on
    Thursday, Friday, Saturday and Sunday -- every game day except Monday -- for
    every week of the season, contradicting the contract stated in the paragraph
    above. The cause was double counting: ``days_since_start // 7 + 1`` is measured
    from the season's opening THURSDAY and is therefore already Thursday-anchored,
    so the conditional ``+1`` for "Tuesday or later" added a second week from
    Thursday onward. Every consumer (the Friday orchestrator, the ingest scripts,
    the prediction filename) uses the value verbatim with no compensation, so the
    orchestrator was generating NEXT week's predictions.

    The fix anchors the buckets on the transition day itself -- the Tuesday two days
    before the opening Thursday -- instead of patching a Thursday-anchored bucket.
    The pre-season guard below is deliberately KEPT: without it, anchoring two days
    earlier would also flip the pre-season Tuesday and Wednesday from the previous
    season's week 18 to the new season's week 1, which is a behaviour change nobody
    asked for.

    Args:
        now: Reference instant, for testing against a frozen clock. Defaults to
            ``datetime.now(ET)``. Purely additive -- every existing call site passes
            nothing and behaves identically.

    Returns:
        Tuple of (season, week) where week is 1-18 for regular season
    """
    if now is None:
        now = datetime.now(ET)

    season = get_current_nfl_season(now)
    season_start = get_nfl_season_start(season)

    # Calculate weeks since season start
    if now < season_start:
        # We're before the season starts, return previous season's last week
        return season - 1, NFL_REGULAR_SEASON_WEEKS

    # NFL weeks transition on the Tuesday after the previous week's last game
    # (Monday Night Football), so bucket from that Tuesday.
    week_anchor = season_start - timedelta(days=2)
    return season, min((now - week_anchor).days // 7 + 1, NFL_TOTAL_WEEKS)


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
        r"(\d{4})-(\d{2})-(\d{2})",  # YYYY-MM-DD
        r"(\d{2})/(\d{2})/(\d{4})",  # MM/DD/YYYY
        r"(\d{1,2})/(\d{1,2})/(\d{4})",  # M/D/YYYY
    ]

    # Try regex patterns first
    for pattern in patterns:
        match = re.match(pattern, date_str.strip())
        if match:
            groups = match.groups()
            if len(groups) == 3:
                if pattern.startswith(r"(\d{4})"):  # YYYY-MM-DD
                    year, month, day = map(int, groups)
                else:  # MM/DD/YYYY or M/D/YYYY
                    month, day, year = map(int, groups)

                dt = datetime(year, month, day, tzinfo=ET)
                return dt

    # Fall back to dateutil parser
    #
    # WR-04: this branch previously called ``ET.localize(dt)`` -- the pytz API on a
    # ``zoneinfo.ZoneInfo`` object, which has no such method. So any date string
    # that missed all three regexes and parsed naive raised ``AttributeError``, and
    # ``AttributeError`` was not in the except tuple, which made the documented
    # ValueError below unreachable.
    try:
        dt = parser.parse(date_str)
        # Attach ET to a naive parse; convert an aware one.
        dt = dt.replace(tzinfo=ET) if dt.tzinfo is None else dt.astimezone(ET)
        return dt
    except (ValueError, TypeError, AttributeError) as e:
        raise ValueError(f"Unable to parse date string: {date_str}") from e


def is_game_time(kickoff_time: datetime, check_time: datetime | None = None) -> bool:
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


def get_snapshot_time(
    date: datetime | None = None, time_str: str = "Friday 18:00"
) -> datetime:
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
    day_time_pattern = r"(\w+day)\s+(\d{1,2}):(\d{2})"
    match = re.match(day_time_pattern, time_str)

    if not match:
        raise ValueError(f"Invalid time specification: {time_str}")

    day_name, hour_str, minute_str = match.groups()
    hour, minute = int(hour_str), int(minute_str)

    # Map day names to weekday numbers (Monday=0)
    day_map = {
        "monday": 0,
        "tuesday": 1,
        "wednesday": 2,
        "thursday": 3,
        "friday": 4,
        "saturday": 5,
        "sunday": 6,
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
    snapshot_time = target_date.replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )

    return snapshot_time


def get_week_start_end(season: int, week: int) -> tuple[datetime, datetime]:
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
        home_tz = ZoneInfo(home_timezone)
        away_tz = ZoneInfo(away_timezone)

        # Use a reference time to calculate offset
        ref_time = datetime.now()
        home_offset = (
            ref_time.replace(tzinfo=home_tz).utcoffset().total_seconds() / 3600
        )
        away_offset = (
            ref_time.replace(tzinfo=away_tz).utcoffset().total_seconds() / 3600
        )

        return home_offset - away_offset
    except (KeyError, ValueError, AttributeError):
        return 0.0  # Default to no difference if calculation fails


def ensure_utc_aware(dt: datetime) -> datetime:
    """Return *dt* as a timezone-aware datetime in UTC.

    Fix-forward helper for callers that were previously passing naive
    datetimes to storage and now need to be explicit about timezone.
    Phase 15-04 made ``data/storage.py`` reject naive datetimes at write
    time; this helper gives producers a cheap way to keep their existing
    code shape while becoming explicit about UTC.

    Behavior:
    - If *dt* has tzinfo, it is converted to UTC via ``astimezone(UTC)``.
    - If *dt* is naive (``tzinfo is None``), it is REINTERPRETED as UTC by
      attaching ``tzinfo=UTC`` without shifting the wall clock. This is
      the correct semantic when the caller knows the source was already
      UTC but was stored as naive.
    - If *dt* is None or not a ``datetime`` instance, raises ValueError.

    Callers that know their source is in a DIFFERENT timezone (e.g. ET)
    should NOT use this helper -- instead, use
    ``dt.replace(tzinfo=ET).astimezone(UTC)`` to get the correct UTC
    conversion.

    Args:
        dt: A datetime instance (naive or aware).

    Returns:
        A timezone-aware datetime in UTC.

    Raises:
        ValueError: If *dt* is None or not a datetime instance.
    """
    if dt is None:
        raise ValueError("ensure_utc_aware() received None; expected datetime")
    if not isinstance(dt, datetime):
        raise ValueError(
            f"ensure_utc_aware() received {type(dt).__name__}; expected datetime"
        )
    if dt.tzinfo is not None:
        return dt.astimezone(UTC)
    return dt.replace(tzinfo=UTC)
