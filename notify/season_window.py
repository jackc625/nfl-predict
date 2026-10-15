"""The watcher's season window: is today a day the outside watcher should hear from the run? (R7).

THE WINDOW (SPEC R7, D-01)
--------------------------
A season S is in its window from :data:`WINDOW_LEAD_DAYS` days before its first lock day (the lock
of its earliest recorded kickoff, ``utils.game_lock.game_lock``) through the day its Super Bowl
result is recorded. Outside every season's window the daily run pauses the watcher's check and never
pings it; the first in-window ping resumes it (the check has ``manual_resume`` false).

EVALUATED AT RUN START
----------------------
The daily run reads the window once, at its start, BEFORE its schedule refresh. The 17:00 ET run
the day after the Super Bowl is the one that ingests the result, and at its start silver still
lacks the score, so that day is in; the next day pauses -- "through the day the result is
recorded".

WHY A GRACE DAY AS WELL AS THE SCORE (review 35-13)
---------------------------------------------------
The daily run is not the only writer of silver ``games``: ``scripts/ingest_games.py`` can be run by
hand. A hand run that records the Super Bowl score earlier on the day after the game would otherwise
pause the watcher before the scheduled run that day. So a season counts as over only when its Super
Bowl has both scores AND the run date is more than :data:`FINAL_GAME_GRACE_DAYS` after the game's
ET date. A longer window can only cost a false alarm, never silence (D-17's direction).

UNKNOWN (D-17)
--------------
A frame that is empty, lacks a needed column or holds a value that cannot be read gives
:data:`SEASON_WINDOW_UNKNOWN`, and the run checks in as if in season. Nothing here raises.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING

from utils.date_utils import kickoff_wall_clock_et
from utils.game_lock import game_lock

if TYPE_CHECKING:
    import pandas as pd

__all__ = [
    "FINAL_GAME_GRACE_DAYS",
    "SEASON_WINDOW_IN",
    "SEASON_WINDOW_OUT",
    "SEASON_WINDOW_UNKNOWN",
    "WINDOW_LEAD_DAYS",
    "current_season_window",
    "season_window_state",
]

SEASON_WINDOW_IN = "in"
SEASON_WINDOW_OUT = "out"
SEASON_WINDOW_UNKNOWN = "unknown"

# The window opens this many days before the season's first lock day (SPEC R7).
WINDOW_LEAD_DAYS = 7
# The day after the Super Bowl is the first day a scheduled 17:00 ET run can record its result, so
# it is always in: a score another process wrote earlier that day must not pause the watcher
# before that run (review 35-13: a same-day second writer).
FINAL_GAME_GRACE_DAYS = 1

_SUPER_BOWL = "SB"
_NEEDED_COLUMNS = frozenset(
    {"season", "kickoff_et", "game_type", "home_score", "away_score"}
)


def _season_is_over(rows: pd.DataFrame, run_date_et: date) -> bool:
    """Whether the season in *rows* has a recorded Super Bowl result and its grace day has passed."""
    import pandas as pd

    super_bowl = rows.loc[rows["game_type"] == _SUPER_BOWL]
    for _, game in super_bowl.iterrows():
        if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
            continue
        game_day = kickoff_wall_clock_et(game["kickoff_et"]).date()
        if run_date_et > game_day + timedelta(days=FINAL_GAME_GRACE_DAYS):
            return True
    return False


def _window_opens(rows: pd.DataFrame) -> date | None:
    """The first day of the season's window, or None when it has no recorded kickoff."""
    import pandas as pd

    kickoffs = [
        kickoff_wall_clock_et(value) for value in rows["kickoff_et"] if pd.notna(value)
    ]
    if not kickoffs:
        return None
    return game_lock(min(kickoffs)).date() - timedelta(days=WINDOW_LEAD_DAYS)


def season_window_state(games: pd.DataFrame | None, run_date_et: date) -> str:
    """``in``, ``out`` or ``unknown`` for *run_date_et*, from the silver *games* frame. Pure.

    In iff some season's window has opened on or before *run_date_et* and that season is not over.
    """
    if games is None or games.empty or not set(games.columns) >= _NEEDED_COLUMNS:
        return SEASON_WINDOW_UNKNOWN
    try:
        for _season, rows in games.groupby("season"):
            opens = _window_opens(rows)
            if opens is None or opens > run_date_et:
                continue
            if not _season_is_over(rows, run_date_et):
                return SEASON_WINDOW_IN
    except Exception:  # noqa: BLE001 - an unreadable value means unknown, which checks in (D-17)
        return SEASON_WINDOW_UNKNOWN
    return SEASON_WINDOW_OUT


def current_season_window(run_date_et: date) -> str:
    """The window for *run_date_et* from silver ``games`` as it is now. Never raises.

    Any failure to read the table gives ``unknown`` (D-17).
    """
    try:
        from data.storage import load_dataframe

        games = load_dataframe("games", layer="silver")
    except Exception:  # noqa: BLE001 - an unreadable schedule checks in (D-17)
        return SEASON_WINDOW_UNKNOWN
    return season_window_state(games, run_date_et)
