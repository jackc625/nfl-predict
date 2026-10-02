"""Presentation helpers for the Broadcast UI: team colours, nicknames and kickoff windows.

Pure functions over static team data and a game's own kickoff timestamp. Nothing here reads the
cache or computes a metric (UIAP-01): which colour a team's block uses, what a team is called and
which TV window a kickoff falls in are formatting decisions, made identically on every request.

``predictions.game_date`` is a naive TIMESTAMP already in US Eastern time -- a 1 PM ET kickoff is
stored as ``13:00:00`` -- so a naive value is read as Eastern as-is, and an aware one is converted
to Eastern and then made naive, so every kickoff in a group can be compared with every other.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, NamedTuple

# The project's one America/New_York constant, not a third copy of it.
from utils.date_utils import ET
from utils.team_data import get_team_info, validate_team_abbreviation

#: The card colour team blocks sit on (``--color-panel`` in web/static/input.css).
PANEL_HEX = "#151B29"
#: Below this contrast against the panel a block stops reading as a shape: the near-black
#: primaries (CHI, HOU, NE, SEA, TEN, CLE's brown) vanish into the card. Dark-but-distinct
#: primaries such as NYG's navy (1.17) and BAL's purple (1.18) stay above it and keep their colour.
MIN_BLOCK_CONTRAST = 1.15
TEXT_LIGHT = "#FFFFFF"
TEXT_DARK = "#0B0F17"
#: An abbreviation utils.team_data does not know: a neutral panel-2 block, never a guess.
UNKNOWN_BLOCK = "#1D2436"
#: A known team whose primary and secondary are both unusable. No current team reaches this.
FALLBACK_BLOCK = "#5B6478"
#: Pure black disappears on the dark panel, and pure white is never a team's identity colour.
_NEVER_A_BLOCK = frozenset({"#000000", "#FFFFFF"})

TIME_TBD = "Time TBD"
_DAY_NAMES = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)
_DAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_ABBR = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)
#: TV-window bands by ET kickoff hour. 7 PM ET or later is a national night window on any day.
_NIGHT_FROM_HOUR = 19
_LATE_FROM_HOUR = 16
_EARLY_FROM_HOUR = 12
_DOT = "·"
_TBD_KEY: tuple[str, ...] = ("tbd",)


class TeamColors(NamedTuple):
    """A team block's background, and the text colour that reads on it."""

    bg: str
    fg: str


def _channel(value: int) -> float:
    c = value / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    """WCAG relative luminance of a ``#RRGGBB`` colour."""
    digits = hex_color.lstrip("#")
    r, g, b = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two ``#RRGGBB`` colours, from 1.0 to 21.0."""
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _usable_block(hex_color: str) -> bool:
    return (
        hex_color.upper() not in _NEVER_A_BLOCK
        and contrast_ratio(hex_color, PANEL_HEX) >= MIN_BLOCK_CONTRAST
    )


def _text_on(bg: str) -> str:
    """White or ink, whichever contrasts more with *bg*."""
    if contrast_ratio(bg, TEXT_LIGHT) >= contrast_ratio(bg, TEXT_DARK):
        return TEXT_LIGHT
    return TEXT_DARK


def team_block_colors(abbr: str | None) -> TeamColors:
    """The block colour for *abbr*: its primary, else its secondary, else a neutral slate.

    A colour is usable when it is neither pure black nor pure white and reads as a shape on the
    dark panel. An unknown abbreviation gets a neutral block rather than a guessed team colour.
    """
    if not abbr or not validate_team_abbreviation(abbr):
        return TeamColors(UNKNOWN_BLOCK, TEXT_LIGHT)
    primary, secondary = (color.upper() for color in get_team_info(abbr)["colors"][:2])
    for candidate in (primary, secondary):
        if _usable_block(candidate):
            return TeamColors(candidate, _text_on(candidate))
    return TeamColors(FALLBACK_BLOCK, _text_on(FALLBACK_BLOCK))


def team_nickname(abbr: str | None) -> str:
    """``"KC"`` -> ``"Chiefs"``. An unknown abbreviation is returned as-is; ``None`` is ``""``."""
    if not abbr:
        return ""
    if not validate_team_abbreviation(abbr):
        return abbr
    # Every current team name ends in its nickname ("San Francisco 49ers" -> "49ers").
    return str(get_team_info(abbr)["name"]).rsplit(" ", 1)[-1]


def _to_eastern(
    game_date: datetime | date | str | None,
) -> tuple[date, datetime | None] | None:
    """(ET calendar date, ET kickoff or None when no time is known), or None when unknown.

    The kickoff is always NAIVE Eastern wall-clock time. An aware input is converted and then
    stripped of its zone: one window may mix a naive stored stamp with an aware one, and Python
    refuses to sort a naive datetime against an aware one.
    """
    if game_date is None:
        return None
    value: datetime | date | str = game_date
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None
    if isinstance(value, datetime):
        # pandas' NaT is a datetime that is not equal to itself.
        if value != value:  # noqa: PLR0124
            return None
        if value.tzinfo is not None:
            value = value.astimezone(ET).replace(tzinfo=None)
        # A midnight stamp is a date with no kickoff time: no NFL game kicks off at 00:00 ET.
        if (value.hour, value.minute, value.second) == (0, 0, 0):
            return value.date(), None
        return value.date(), value
    if isinstance(value, date):
        return value, None
    return None


def _clock(moment: datetime) -> str:
    return f"{moment.hour % 12 or 12}:{moment.minute:02d}"


def _meridiem(moment: datetime) -> str:
    return "AM" if moment.hour < 12 else "PM"


def _band(moment: datetime) -> str:
    if moment.hour >= _NIGHT_FROM_HOUR:
        return "night"
    if moment.hour >= _LATE_FROM_HOUR:
        return "late"
    if moment.hour >= _EARLY_FROM_HOUR:
        return "early"
    return "morning"


def _window_label(day: date, band: str, moments: list[datetime]) -> str:
    weekday = _DAY_NAMES[day.weekday()]
    if band == "allday" or not moments:
        return weekday
    if band == "night":
        return f"{weekday} Night"
    ordered = sorted(moments)
    clocks: list[str] = []
    for moment in ordered:
        clock = _clock(moment)
        if clock not in clocks:
            clocks.append(clock)
    # Every time in one band shares its AM/PM (bands split at noon), so one suffix is honest.
    return f"{weekday} {_DOT} {' / '.join(clocks)} {_meridiem(ordered[0])} ET"


def kickoff_label(game_date: datetime | date | str | None) -> str:
    """``"Sun 1:00 PM ET"``; a date with no time ``"Tue Sep 10"``; nothing known ``"Time TBD"``."""
    parsed = _to_eastern(game_date)
    if parsed is None:
        return TIME_TBD
    day, moment = parsed
    if moment is None:
        return f"{_DAY_ABBR[day.weekday()]} {_MONTH_ABBR[day.month - 1]} {day.day}"
    return f"{_DAY_ABBR[day.weekday()]} {_clock(moment)} {_meridiem(moment)} ET"


def kickoff_window(game_date: datetime | date | str | None) -> str:
    """The TV-window label for ONE game: ``"Sunday Night"``, ``"Sunday - 4:25 PM ET"``, ...

    (The separator written ``-`` here is really U+00B7, a middle dot; see ``_DOT``.)
    ``group_games_by_window`` builds the label for a whole group, which lists every distinct
    time in the group ("Sunday - 4:05 / 4:25 PM ET"); pages print the group's label.
    """
    parsed = _to_eastern(game_date)
    if parsed is None:
        return TIME_TBD
    day, moment = parsed
    if moment is None:
        return _window_label(day, "allday", [])
    return _window_label(day, _band(moment), [moment])


def decorate_game(game: dict[str, Any]) -> dict[str, Any]:
    """A NEW dict: *game* plus the presentation fields the Broadcast templates read."""
    away = game.get("away_team")
    home = game.get("home_team")
    away_colors = team_block_colors(away)
    home_colors = team_block_colors(home)
    return {
        **game,
        "away_color": away_colors.bg,
        "away_fg": away_colors.fg,
        "home_color": home_colors.bg,
        "home_fg": home_colors.fg,
        "away_name": team_nickname(away),
        "home_name": team_nickname(home),
        "kickoff_label": kickoff_label(game.get("game_date")),
        "window_label": kickoff_window(game.get("game_date")),
    }


def group_games_by_window(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``[{"label": str, "games": [...]}]`` in first-appearance order; input order kept inside.

    A window is an ET calendar day plus a band (morning, early, late, night). A date with no time
    is its own all-day group, and a game with no date at all lands in ``"Time TBD"``.
    """
    members: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    days: dict[tuple[str, ...], date] = {}
    moments: dict[tuple[str, ...], list[datetime]] = {}
    for game in games:
        parsed = _to_eastern(game.get("game_date"))
        key: tuple[str, ...]
        if parsed is None:
            key = _TBD_KEY
        else:
            day, moment = parsed
            key = (day.isoformat(), "allday" if moment is None else _band(moment))
            days[key] = day
            if moment is not None:
                moments.setdefault(key, []).append(moment)
        members.setdefault(key, []).append(game)
    groups: list[dict[str, Any]] = []
    for key, members_in_window in members.items():
        if key == _TBD_KEY:
            label = TIME_TBD
        else:
            label = _window_label(days[key], key[1], moments.get(key, []))
        groups.append({"label": label, "games": members_in_window})
    return groups
