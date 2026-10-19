"""Presentation helpers for the Broadcast UI: team colours, nicknames, kickoff windows, last week.

Pure functions over static team data, a game's own kickoff timestamp and stored figures. Nothing
here reads the cache or computes a metric (UIAP-01): which colour a team's block uses, what a team
is called, which TV window a kickoff falls in and how a stored return reads are formatting
decisions, made identically on every request. The out-of-date bar's state is one stored push
time against the shared 36-hour rule (``api.cache_schema.push_is_stale``), not a computation.

``predictions.game_date`` is a naive TIMESTAMP already in US Eastern time -- a 1 PM ET kickoff is
stored as ``13:00:00`` -- so a naive value is read as Eastern as-is, and an aware one is converted
to Eastern and then made naive, so every kickoff in a group can be compared with every other.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, date, datetime
from typing import Any, NamedTuple

# The project's one America/New_York constant, not a third copy of it.
from utils.date_utils import ET
from utils.team_data import get_team_info, validate_team_abbreviation

from .cache_schema import PUSH_TIME_KEY, push_is_stale

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


#: Per-team block overrides, applied before the primary/secondary rule. Seattle's secondary is
#: Action Green (#69BE28), which reads as the site's "won" green; its College Navy primary
#: (#002244) sits at 1.08 against the panel, under MIN_BLOCK_CONTRAST, so the block takes Wolf
#: Grey with ink text instead.
_BLOCK_OVERRIDES: dict[str, TeamColors] = {"SEA": TeamColors("#A5ACAF", TEXT_DARK)}


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
    """The block colour for *abbr*: an override, else its primary, else its secondary, else slate.

    A colour is usable when it is neither pure black nor pure white and reads as a shape on the
    dark panel. An unknown abbreviation gets a neutral block rather than a guessed team colour.
    """
    if not abbr or not validate_team_abbreviation(abbr):
        return TeamColors(UNKNOWN_BLOCK, TEXT_LIGHT)
    override = _BLOCK_OVERRIDES.get(abbr.upper())
    if override is not None:
        return override
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


def _stored_instant_in_eastern(value: object) -> datetime | str:
    """A stored instant converted to Eastern time, or the text to show when there is none.

    The ONE conversion both Eastern timestamp labels share, so the two can never disagree about a
    zone or the 2026-11-01 DST change (research Pitfall 8). The cache writes its instants in UTC
    (``datetime.now(tz=UTC).isoformat()``) or with an explicit offset, so an aware value is
    converted to Eastern and a NAIVE one is read as UTC. Returns an aware Eastern ``datetime``;
    otherwise ``""`` for None, empty and NaT, and ``str(value)`` unchanged for anything
    unparseable -- the page shows what is stored rather than a guess.
    """
    if value is None:
        return ""
    moment: datetime
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return str(value)
    else:
        return str(value)
    # pandas' NaT is a datetime that is not equal to itself.
    if moment != moment:  # noqa: PLR0124
        return ""
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(ET)


def et_timestamp_label(value: object) -> str:
    """A stored timestamp as Eastern wall-clock text: ``"Sep 6, 2025, 6:00 PM ET"``.

    Accepts an ISO-8601 string, a ``datetime`` or a ``pandas.Timestamp``; see
    ``_stored_instant_in_eastern`` for the zone rule and what an unusable value renders as.
    """
    local = _stored_instant_in_eastern(value)
    if isinstance(local, str):
        return local
    month = _MONTH_ABBR[local.month - 1]
    return f"{month} {local.day}, {local.year}, {_clock(local)} {_meridiem(local)} ET"


def _weekday_label(local: datetime) -> str:
    return (
        f"{_DAY_ABBR[local.weekday()]}, {_MONTH_ABBR[local.month - 1]} {local.day}, "
        f"{_clock(local)} {_meridiem(local)} ET"
    )


def et_weekday_timestamp_label(value: object) -> str:
    """A stored timestamp as Eastern text with its weekday: ``"Sat, Oct 31, 5:12 PM ET"``.

    The out-of-date bar's label (SPEC R8, the owner's mock). No year: the bar only ever names an
    instant from the last day or two. Same conversion and fallbacks as ``et_timestamp_label``.
    """
    local = _stored_instant_in_eastern(value)
    if isinstance(local, str):
        return local
    return _weekday_label(local)


# ---------------------------------------------------------------------------
# The out-of-date bar (Phase 36, SPEC R8, D-06, D-07)
# ---------------------------------------------------------------------------
#
# ONE stored-value read (cache_meta's push_time, which every page already receives) and ONE
# comparison through the shared 36-hour rule -- nothing is computed beyond that (UIAP-01). The same
# code runs on the laptop and on the hosted copy, so the laptop shows the bar too when no push has
# landed (SPEC R8: same rule locally).


def _utc_now() -> datetime:
    return datetime.now(UTC)


#: The clock the bar and hosted ``/health`` both read, looked up at CALL time so one patch of
#: ``api.presentation.NOW`` pins both in a test.
NOW: Callable[[], datetime] = _utc_now

STALE_BAR_NO_PUSH = "Out of date: no push recorded."


class StaleBar(NamedTuple):
    """Whether a page shows the out-of-date bar, and the bar's exact text."""

    show: bool
    text: str


def stale_bar_state(cache_meta: Mapping[str, Any] | None) -> StaleBar:
    """The bar for a page whose cache_meta is *cache_meta* (None when the page has none).

    Not stale -> no bar. Stale with a readable push time -> "Out of date: last update <weekday
    label>. The laptop has not reported since." Stale with no push time, or one that cannot be
    read -> "Out of date: no push recorded." (absent is a stated state, never a blank).
    """
    raw = cache_meta.get(PUSH_TIME_KEY) if cache_meta else None
    push_time = raw if isinstance(raw, str) else None
    if not push_is_stale(push_time, NOW()):
        return StaleBar(show=False, text="")
    local = _stored_instant_in_eastern(push_time)
    if isinstance(local, str):
        return StaleBar(show=True, text=STALE_BAR_NO_PUSH)
    return StaleBar(
        show=True,
        text=(
            f"Out of date: last update {_weekday_label(local)}. "
            "The laptop has not reported since."
        ),
    )


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


# ---------------------------------------------------------------------------
# The not-advice line (SPEC must-NOT: the email carries /bets' not-betting-advice line)
# ---------------------------------------------------------------------------
#
# ONE source. components/_not_advice_banner.html renders these word for word on /bets, and the
# weekly email renders the same two constants, so the email can never carry a different line.

NOT_ADVICE_KEY = "Not wagering advice"
NOT_ADVICE_LINE = (
    "Research tool: no wager is placed, stakes are units, and no edge over the betting market "
    "has been demonstrated."
)


# ---------------------------------------------------------------------------
# Last week: actual vs expected (SPEC R5, D-08)
# ---------------------------------------------------------------------------
#
# ONE formatter for two surfaces. /bets renders the dict below and the weekly email calls the same
# function, so the two can never format a figure differently (SPEC R6). The figures were computed
# at cache build by backtest.bet_week_results; nothing here counts, sums or divides.

LAST_WEEK_CAVEAT = "One week is far too few bets to judge the model."
NOTHING_GRADED = "nothing graded"
#: Last week's stored rows could not be read (a cache older than the table, or a failed query).
LAST_WEEK_NOT_AVAILABLE = "not available"
NOT_MEASURED = "not measured"
#: The stored counting class, as the reader is told it. A class outside this map is shown raw.
COUNTING_CLASS_LABELS: dict[str, str] = {
    "forward_realized": "Counted toward the verdict",
    "pre_verdict": "Decided before the counting start week -- not counted",
}


def percent_of_stake(value: float | None) -> str:
    """A stored return, a FRACTION of stake, as ``"+32.7% of stake"``; None or NaN is ``"not measured"``.

    The x100 is a unit render (fraction -> percent), the same kind the EV column applies, not a
    derivation (UIAP-01). An unmeasured figure is words, never ``+0.0%`` (D-08).
    """
    if value is None or math.isnan(value):
        return NOT_MEASURED
    return f"{value * 100:+.1f}% of stake"


def _last_week_row_display(row: dict[str, Any]) -> dict[str, Any]:
    """One counting class's display strings. A class with nothing graded carries no figure."""
    counting_class = row["counting_class"]
    display: dict[str, Any] = {
        "class_label": COUNTING_CLASS_LABELS.get(counting_class, counting_class),
        "record": None,
        "actual": None,
        "expected": None,
        "ungraded": f"Not yet graded: {row['bets_ungraded']}",
    }
    if row["bets_graded"]:
        display["record"] = f"{row['wins']}-{row['losses']}-{row['pushes']} (W-L-P)"
        display["actual"] = "Actual return: " + percent_of_stake(
            row["actual_return_units"]
        )
        display["expected"] = "Expected return: " + percent_of_stake(
            row["expected_return_units"]
        )
    return display


def last_week_display(
    rows: Sequence[dict[str, Any]] | None, season: int | None, week: int | None
) -> dict[str, Any]:
    """The last-week block's strings for the week before *week*, from its stored rows (D-08).

    *rows* are ``DataService.get_bet_week_results(season, week - 1)`` in their stored class order;
    each becomes one entry of ``blocks``. Week 1 (or no week) has no last week -- the caller never
    reaches into the prior season -- so it, an empty *rows* and a week where no class graded
    anything are all ``nothing_graded``. *rows* None means they could not be read: the block
    then shows no figure and says ``not available`` in the same place -- absent is never
    "nothing graded" (35-REVIEW WR-03).
    """
    heading = "Last week"
    blocks: list[dict[str, Any]] = []
    unavailable = False
    if season is not None and week is not None and week > 1:
        heading = f"Last week -- {season} week {week - 1}"
        unavailable = rows is None
        blocks = [_last_week_row_display(row) for row in rows or ()]
    return {
        "heading": heading,
        "nothing_graded": not any(block["record"] for block in blocks),
        "text_nothing_graded": LAST_WEEK_NOT_AVAILABLE
        if unavailable
        else NOTHING_GRADED,
        "blocks": blocks,
        "caveat": LAST_WEEK_CAVEAT,
    }
