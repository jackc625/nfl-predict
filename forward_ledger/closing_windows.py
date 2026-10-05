"""When the laptop wakes for closing-line readings: windows from real kickoffs (Phase 34, D-07).

Pure functions over a schedule. The scheduled-task registration that turns these windows into
``TimeTrigger`` elements is Plan 34-11; the capture each window runs is Plan 34-14.

THE RULE (D-07)
---------------
Kickoffs are sorted. A window opens ``READING_LEAD`` before the earliest kickoff not yet served and
serves every kickoff no more than ``CLOSING_WINDOW`` (60:00) after its start; the next window opens
at the first kickoff it could not serve. Because the reading is taken at or after the window's
start, it is in-window (``forward_ledger.closing``) for every kickoff the window serves:

  * 4:05 PM and 4:25 PM ET share one window opening at 3:55 PM;
  * a 7:15 PM / 8:15 PM Monday doubleheader needs two (8:15 is 70 minutes after 7:05);
  * international 9:30 AM, Saturday, Thanksgiving, Christmas and Wednesday games get windows
    exactly like any other kickoff, because windows come from the schedule, not from a slot list;
  * a day with no game gets no window. There is never a repeating polling timer.

KICKOFFS ARE INSTANTS
---------------------
Silver ``games.kickoff_et`` is a tz-aware UTC instant despite its name. Every kickoff is read
through ``utils.date_utils.kickoff_wall_clock_et`` -- THE accessor, which converts an aware value
and never relabels it (and ET-localizes a naive one, matching the schedule writer) -- and is then
held as a UTC instant.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from forward_ledger.closing import CLOSING_WINDOW
from utils.date_utils import kickoff_wall_clock_et

__all__ = [
    "MAX_TRIGGERS",
    "READING_LEAD",
    "ReadingWindow",
    "kickoff_instant",
    "reading_windows",
    "windows_for_schedule",
]

# How long before the earliest kickoff it serves a window opens: margin for waking from sleep and
# reaching the network, while keeping a 20-minute pair (4:05 / 4:25) inside one 60:00 window.
READING_LEAD = timedelta(minutes=10)

# Task Scheduler's per-task trigger limit ("A task can have a maximum of 48 triggers").
MAX_TRIGGERS = 48


@dataclass(frozen=True)
class ReadingWindow:
    """One closing reading: when it starts and which kickoffs it serves (sorted, UTC)."""

    start_utc: datetime
    kickoffs_utc: tuple[datetime, ...]
    game_ids: tuple[str, ...]


def kickoff_instant(value: Any) -> datetime:
    """A ``games.kickoff_et`` value as a tz-aware UTC instant (through the one accessor)."""
    return kickoff_wall_clock_et(value).astimezone(UTC)


def reading_windows(kickoffs: Mapping[str, Any]) -> list[ReadingWindow]:
    """The reading windows serving *kickoffs* (``game_id -> kickoff``), earliest first.

    A game with no kickoff has no instant to serve and gets no window. No kickoff -> ``[]``.
    """
    ordered = sorted(
        (kickoff_instant(kickoff), str(game_id))
        for game_id, kickoff in kickoffs.items()
        if not pd.isna(kickoff)
    )
    windows: list[ReadingWindow] = []
    position = 0
    while position < len(ordered):
        start = ordered[position][0] - READING_LEAD
        served: list[tuple[datetime, str]] = []
        while (
            position < len(ordered) and ordered[position][0] - start <= CLOSING_WINDOW
        ):
            served.append(ordered[position])
            position += 1
        windows.append(
            ReadingWindow(
                start_utc=start,
                kickoffs_utc=tuple(kickoff for kickoff, _ in served),
                game_ids=tuple(game_id for _, game_id in served),
            )
        )
    return windows


def windows_for_schedule(
    games: pd.DataFrame, now: datetime, *, days: int = 8
) -> list[ReadingWindow]:
    """The windows to register now: unplayed games kicking off within the next *days* days.

    A game is kept when it has no recorded result (``home_score`` NULL) and its kickoff is in
    ``(now + READING_LEAD, now + days]`` -- so every returned window starts after *now* (a window
    that has already opened cannot be registered as a future trigger) and a game already kicked
    off is never served. At most :data:`MAX_TRIGGERS` windows, earliest first; a later window
    left off is registered by the next day's run, because the horizon rolls daily (D-07).

    Args:
        games: Silver ``games`` rows: ``game_id``, ``kickoff_et``, ``home_score``.
        now: The registration instant; must be timezone-aware.
        days: The horizon in days.

    Raises:
        scripts.ingest_historical_odds.NaiveTimestampError: *now* is naive.
    """
    # Lazy import: scripts.ingest_historical_odds cycles back through backtest.bet_selector.
    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    now_utc = require_aware_snapshot_ts(now)
    horizon = now_utc + timedelta(days=days)
    unplayed = games.loc[games["home_score"].isna() & games["kickoff_et"].notna()]
    schedulable: dict[str, datetime] = {}
    for game_id, kickoff in zip(
        unplayed["game_id"], unplayed["kickoff_et"], strict=True
    ):
        instant = kickoff_instant(kickoff)
        if now_utc + READING_LEAD < instant <= horizon:
            schedulable[str(game_id)] = instant
    return reading_windows(schedulable)[:MAX_TRIGGERS]
