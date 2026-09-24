"""THE current NFL slate, read from the RECORDED SCHEDULE -- never from calendar arithmetic.

Every "which (season, week) is it now?" question in the project is answered here, and
``utils.date_utils.get_current_nfl_week`` is a thin delegate to :func:`resolve_current_slate`, so
every caller that already imports that name is on this answer without changing a call site.

WHY THIS EXISTS (Plan 33.2-24 step 24c)
---------------------------------------
The previous resolver counted weeks from a COMPUTED Thursday-after-Labor-Day. MEASURED 2026-09-23
over silver ``games`` 2018-2026 at each game's own lock and kickoff, it disagreed with the schedule
on 18 games: every season opener at its lock (a lock before the computed Thursday resolved to the
PREVIOUS season's week 18), the whole Wednesday 2026 opener ``2026_W01_NE@SEA``, the 2020/2021
Tuesday reschedules, and the 17-week-era Super Bowls (the bye week before them is invisible to a
count). The failure mode was an OMITTED capture or prediction -- the resolved slate simply did not
contain the game -- behind the Friday orchestrator, the staleness gate, the trainers' ``--week``
default, ``generate_bet_list``, ``data_qa`` and several ingests. Step 24b fixed the same count for
odds KEYS (``scripts/ingest_odds_timeline.match_event_to_schedule``); this is the SLATE half.

THE RULE
--------
The current slate is the week of the EARLIEST SCHEDULED GAME WHOSE KICKOFF ET CALENDAR DAY IS
TODAY OR LATER, where "today" is the ET calendar date of the instant asked about. So a Monday-night
game keeps Monday in its week, Tuesday moves to the next week, a Tuesday reschedule is its own
week on its own day, and every game's lock instant (18:00 ET the day before kickoff, D33.2-01)
resolves to that game's own week -- asserted over every 2018-2026 game by
``tests/unit/test_current_slate_schedule_keyed.py``. Days, not instants, because a slate is a set
of games a daily run acts on: at 23:30 after Monday night the week is not over until the day is.

A SEASON OPENS ON ITS OPENER'S LOCK DAY. Between two seasons the "earliest game at or after today"
is next season's opener for months, so the rule alone would call the whole offseason week 1. The
season instead opens on the ET date of the opener's lock, derived through ``utils.game_lock`` --
the one lock rule -- so the first day a daily run may act on the opener is the first day this
resolver names its slate. Before that day the offseason branch below answers.

THE BRANCHES WHERE THE SCHEDULE NAMES NO SLATE, EACH EXPLICIT
--------------------------------------------------------------
* ``offseason`` -- the schedule says no slate is open: before an opener's lock day, or after a
  season whose Super Bowl is recorded with the next season not yet scheduled. The value is the
  RETIRED CALENDAR COUNT (:func:`retired_calendar_week`), kept for exactly this one case because
  it is the documented pre-season contract every offseason caller already reads (the previous
  season's week 18 before a season starts; week 22 through the spring). It is never consulted
  while a slate is open, and a calendar value naming a season the schedule has not opened is
  replaced by the documented pre-season value, so the count cannot open a season early.
* ``ScheduleNotRecordedError`` -- nothing is scheduled after a completed season AND the calendar
  season has already turned (August onward). The next schedule has been published since spring;
  resolving the offseason here would hide the season opener behind a missing ingest, which is the
  silent in-season failure this module exists to remove. Refused by name, with the command.
  The daily run is the one caller that answers it itself: :func:`refresh_target` names that
  next season, whose capture and ingest record its schedule (step 27b).
* ``next_unrecorded_round`` -- a season is in its playoffs and the next round is not recorded yet
  (nflverse adds each round's games only once the previous round is decided, so a store refreshed
  before the weekend it describes lacks them). The next round is the recorded week plus one: the
  schedule numbers every round consecutively (REG 17/18, then WC, DIV, CON, SB). Valid only while
  the last recorded game is a playoff round or the final regular week AND is no older than the
  measured ``NEXT_ROUND_VALID_DAYS`` for its round; anything else is refused
  (``ScheduleIncompleteError``).
* An empty or unreadable schedule, a naive instant, a null kickoff, a day that names two slates, a
  season whose first recorded week is not week 1: each refused by name.

IMPORT DISCIPLINE: pandas and the settings are imported lazily, and ``utils.date_utils`` imports
this module only inside ``get_current_nfl_week``, so importing ``utils.date_utils`` reads nothing
from disk and loads no resolver (asserted in a fresh interpreter).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Final

from utils.date_utils import (
    ET,
    NFL_REGULAR_SEASON_WEEKS,
    NFL_TOTAL_WEEKS,
    get_current_nfl_season,
    get_nfl_season_start,
    kickoff_wall_clock_et,
)

if TYPE_CHECKING:  # pragma: no cover - typing only; pandas stays a lazy import
    import pandas as pd

__all__ = [
    "BASIS_NEXT_UNRECORDED_ROUND",
    "BASIS_OFFSEASON",
    "BASIS_SCHEDULED_SLATE",
    "AmbiguousSlateError",
    "CurrentSlate",
    "ScheduleIncompleteError",
    "ScheduleNotRecordedError",
    "ScheduleUnavailableError",
    "SlateResolutionError",
    "cli_target_season_week",
    "first_recorded_kickoff",
    "load_recorded_schedule",
    "prepare_schedule",
    "refresh_target",
    "resolve_current_slate",
    "retired_calendar_week",
    "season_has_kicked_off",
    "season_is_complete",
]

#: The slate is a recorded week of the schedule.
BASIS_SCHEDULED_SLATE: Final[str] = "scheduled_slate"
#: The slate is the playoff round after the last recorded one, not yet in the store.
BASIS_NEXT_UNRECORDED_ROUND: Final[str] = "next_unrecorded_round"
#: No slate is open; the value is the documented offseason contract.
BASIS_OFFSEASON: Final[str] = "offseason"

#: How many days after the last recorded game of a round the NEXT round is certainly still the
#: current slate: the FEWEST days from that round's last game to the next round's last game,
#: MEASURED 2026-09-23 over silver ``games`` 2002-2025 (final regular week 6-8, wild card 6-7,
#: divisional 7-7, conference 7-14). Past it, the next round may already be over, so a store
#: whose last game is older is stale by more than one round and the slate is refused rather
#: than inferred. Keyed by the last recorded game's ``game_type``; ``REG`` means the final
#: regular week.
NEXT_ROUND_VALID_DAYS: Final[dict[str, int]] = {"REG": 6, "WC": 6, "DIV": 7, "CON": 7}

#: The first season with 18 regular-season weeks (the 17-game schedule). Every earlier season
#: this project holds (2002-2020) has 17. A schedule FACT, not calendar arithmetic: it says which
#: recorded week ends a regular season, so a store ending there is complete through the regular
#: season rather than truncated.
FIRST_EIGHTEEN_WEEK_SEASON: Final[int] = 2021

#: The recorded game types that name a playoff round before the Super Bowl.
_PLAYOFF_ROUNDS_BEFORE_SUPER_BOWL: Final[frozenset[str]] = frozenset(
    {"WC", "DIV", "CON"}
)
_SUPER_BOWL: Final[str] = "SB"

#: The silver ``games`` columns the resolver reads.
SCHEDULE_COLUMNS: Final[tuple[str, ...]] = (
    "game_id",
    "season",
    "week",
    "kickoff_et",
    "game_type",
)

_INGEST_COMMAND: Final[str] = "uv run python -m scripts.ingest_games --season {season}"


class SlateResolutionError(LookupError):
    """The recorded schedule cannot name the current slate.

    A ``LookupError`` rather than a ``ValueError``: the question is a lookup against a store, and
    a caller's broad ``except ValueError`` around bad INPUT must not swallow a missing schedule.
    """


class ScheduleUnavailableError(SlateResolutionError):
    """No schedule could be read at all: the file is missing, unreadable or empty."""


class ScheduleNotRecordedError(SlateResolutionError):
    """The calendar season has turned and its schedule is not in the store.

    Attributes:
        season: The season whose schedule is due and not recorded, when the raiser knows it.
    """

    def __init__(self, message: str, *, season: int | None = None) -> None:
        super().__init__(message)
        self.season = season


class ScheduleIncompleteError(SlateResolutionError):
    """The store stops short of today in a way that is not one unrecorded playoff round.

    Attributes:
        season: The recorded season whose re-capture would heal the store, when known.
        week: The week that re-capture is filed under (the first week the store lacks).
    """

    def __init__(
        self, message: str, *, season: int | None = None, week: int | None = None
    ) -> None:
        super().__init__(message)
        self.season = season
        self.week = week


class AmbiguousSlateError(SlateResolutionError):
    """One ET calendar day carries games of two different (season, week) slates."""


@dataclass(frozen=True)
class CurrentSlate:
    """The resolved slate, and how it was resolved.

    Attributes:
        season: NFL season year.
        week: NFL week number in the schedule's own numbering (playoff rounds continue it).
        in_season: ``True`` when a slate is open -- ``False`` only on the offseason branch.
        basis: One of the ``BASIS_*`` constants, naming the branch that answered.
    """

    season: int
    week: int
    in_season: bool
    basis: str

    def as_tuple(self) -> tuple[int, int]:
        """``(season, week)`` -- the shape ``get_current_nfl_week`` has always returned."""
        return self.season, self.week


# ---------------------------------------------------------------------------
# The schedule
# ---------------------------------------------------------------------------


def default_schedule_path() -> Path:
    """Silver ``games`` under the configured data root (``DATA_ROOT_PATH`` still redirects it)."""
    from conf.settings import get_settings

    return Path(get_settings().config.data.root_path) / "silver" / "games.parquet"


def prepare_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """The schedule as the resolver reads it: one row per game, ordered by ET kickoff day.

    Adds ``et_day`` (the ET calendar date, through ``kickoff_wall_clock_et`` -- THE one accessor,
    which converts an aware kickoff and never relabels it) and ``et_ordinal`` (its ordinal, for
    a binary search).

    Raises:
        ScheduleUnavailableError: when a required column is absent.
        utils.game_lock.MissingKickoffError: naming every game with no kickoff. A game with no
            kickoff has no day, so it cannot be placed in any slate.
        AmbiguousSlateError: naming each ET day that carries two slates.
    """
    from utils.game_lock import MissingKickoffError

    missing_columns = [c for c in SCHEDULE_COLUMNS if c not in schedule.columns]
    if missing_columns:
        msg = (
            f"the schedule is missing {missing_columns}; the current slate is read from each "
            "game's season, week, kickoff and game type"
        )
        raise ScheduleUnavailableError(msg)

    frame = schedule.loc[:, list(SCHEDULE_COLUMNS)].copy()
    null_kickoffs = frame["kickoff_et"].isna()
    if bool(null_kickoffs.any()):
        offenders = sorted(map(str, frame.loc[null_kickoffs, "game_id"]))
        msg = (
            f"{len(offenders)} scheduled game(s) have no kickoff and so no ET day and no "
            f"slate: {offenders}. Refusing rather than dropping them from the schedule."
        )
        raise MissingKickoffError(msg)

    frame["season"] = frame["season"].astype(int)
    frame["week"] = frame["week"].astype(int)
    frame["game_type"] = frame["game_type"].astype(str)
    frame["et_day"] = [kickoff_wall_clock_et(v).date() for v in frame["kickoff_et"]]
    frame["et_ordinal"] = [day.toordinal() for day in frame["et_day"]]

    slates_per_day = frame.groupby("et_ordinal")[["season", "week"]].nunique()
    clashing = slates_per_day[
        (slates_per_day["season"] > 1) | (slates_per_day["week"] > 1)
    ]
    if not clashing.empty:
        days = [date.fromordinal(int(o)).isoformat() for o in clashing.index]
        msg = (
            f"ET day(s) {days} carry games of more than one (season, week); a day names one "
            "slate, so the schedule is malformed and no slate can be read from it"
        )
        raise AmbiguousSlateError(msg)

    return frame.sort_values(["et_ordinal", "kickoff_et", "game_id"]).reset_index(
        drop=True
    )


#: Prepared schedules keyed by resolved path, stamped with (mtime_ns, size) so a rewritten file
#: is re-read. The storage writers replace the parquet file, which moves both.
_PREPARED_BY_PATH: dict[str, tuple[tuple[int, int], pd.DataFrame]] = {}


def load_recorded_schedule(path: Path | None = None) -> pd.DataFrame:
    """Silver ``games``, prepared, re-read only when the file changes. Treat it as read-only.

    Args:
        path: The ``games.parquet`` to read; defaults to :func:`default_schedule_path`.

    Raises:
        ScheduleUnavailableError: when the file is missing or cannot be read.
    """
    import pandas as pd

    schedule_path = Path(path) if path is not None else default_schedule_path()
    try:
        stat = schedule_path.stat()
    except OSError as exc:
        msg = (
            f"the recorded schedule {schedule_path} is missing or unreadable ({exc}); the "
            "current slate is read from it and is never guessed from the calendar"
        )
        raise ScheduleUnavailableError(msg) from exc

    key = str(schedule_path.resolve())
    stamp = (stat.st_mtime_ns, stat.st_size)
    cached = _PREPARED_BY_PATH.get(key)
    if cached is not None and cached[0] == stamp:
        return cached[1]

    try:
        raw = pd.read_parquet(schedule_path, columns=list(SCHEDULE_COLUMNS))
    except (OSError, ValueError, KeyError) as exc:
        msg = f"the recorded schedule {schedule_path} could not be read ({exc})"
        raise ScheduleUnavailableError(msg) from exc

    prepared = prepare_schedule(raw)
    _PREPARED_BY_PATH[key] = (stamp, prepared)
    return prepared


# ---------------------------------------------------------------------------
# The retired calendar count -- the offseason branch's value ONLY
# ---------------------------------------------------------------------------


def retired_calendar_week(now: datetime) -> tuple[int, int]:
    """The RETIRED calendar count, kept only as the offseason branch's documented value.

    NEVER a slate resolver: it counts weeks from the computed Thursday after Labor Day, which is
    the step-24b defect (18 games 2018-2026 resolved to another slate). :func:`resolve_current_slate`
    consults it only when the schedule itself says no slate is open, where its output is the
    documented pre-season contract: before a season starts, the previous season's week 18; through
    the spring, the previous season's week clamped to 22.

    The body is the post-WR-05 arithmetic unchanged: buckets anchored on the Tuesday two days
    before the computed opener (WR-05 removed a double count that ran one week high Thursday
    through Sunday), with the pre-season guard that holds the previous season's week 18 until the
    computed Thursday.

    Args:
        now: A timezone-aware instant.

    Returns:
        ``(season, week)`` by the calendar.
    """
    season = get_current_nfl_season(now)
    season_start = get_nfl_season_start(season)
    if now < season_start:
        return season - 1, NFL_REGULAR_SEASON_WEEKS
    week_anchor = season_start - timedelta(days=2)
    return season, min((now - week_anchor).days // 7 + 1, NFL_TOTAL_WEEKS)


# ---------------------------------------------------------------------------
# The resolver
# ---------------------------------------------------------------------------


def _require_aware(now: datetime | None) -> datetime:
    """*now*, or the current ET instant; a naive instant has no ET day and is refused."""
    if now is None:
        return datetime.now(ET)
    if now.tzinfo is None or now.utcoffset() is None:
        msg = (
            f"the current slate needs a timezone-aware instant; {now!r} is naive, and its ET "
            "calendar day -- which decides the slate -- cannot be known without its timezone"
        )
        raise ValueError(msg)
    return now


def _final_regular_week(season: int) -> int:
    """The last regular-season week of *season* (17 through 2020, 18 from 2021)."""
    return 18 if season >= FIRST_EIGHTEEN_WEEK_SEASON else 17


def _offseason(now: datetime, completed_season: int) -> CurrentSlate:
    """The offseason value: the retired calendar count, never naming an unopened season."""
    season, week = retired_calendar_week(now)
    if season > completed_season:
        # The calendar thinks a season began that the schedule has not opened (its opener's
        # lock day is still ahead). The schedule wins; the documented pre-season value stands.
        season, week = completed_season, NFL_REGULAR_SEASON_WEEKS
    return CurrentSlate(season, week, in_season=False, basis=BASIS_OFFSEASON)


def _opening_slate(
    frame: pd.DataFrame, first: int, today: date, now: datetime
) -> CurrentSlate:
    """The slate when the next recorded game opens its season (nothing of it is behind us).

    ``first`` is the row index of the season's first recorded game. The season opens on the ET
    date of that game's lock, taken from the one lock rule.
    """
    from utils.game_lock import game_lock

    opener = frame.iloc[first]
    season, week = int(opener["season"]), int(opener["week"])
    if week != 1:
        msg = (
            f"the recorded {season} schedule starts at week {week} ({opener['game_id']}), not "
            "week 1; its opening weeks are missing, so no slate can be read from it. Re-ingest "
            f"the season: `{_INGEST_COMMAND.format(season=season)}`"
        )
        raise ScheduleIncompleteError(msg, season=season, week=1)

    opener_lock_day = game_lock(
        opener["kickoff_et"], game_id=str(opener["game_id"])
    ).date()
    if today >= opener_lock_day:
        return CurrentSlate(season, 1, in_season=True, basis=BASIS_SCHEDULED_SLATE)
    return _offseason(now, completed_season=season - 1)


def _slate_after_the_last_recorded_game(
    frame: pd.DataFrame, today: date, now: datetime
) -> CurrentSlate:
    """Nothing is recorded at or after today: offseason, next playoff round, or refusal."""
    last = frame.iloc[-1]
    season, week = int(last["season"]), int(last["week"])
    game_type = str(last["game_type"])
    season_types = set(frame.loc[frame["season"] == season, "game_type"])

    if _SUPER_BOWL in season_types:
        calendar_season = get_current_nfl_season(now)
        if calendar_season > season:
            msg = (
                f"the {season} season is complete and nothing is scheduled after it, but the "
                f"calendar season is already {calendar_season}: its schedule is not recorded in "
                "silver games. Resolving the offseason here would hide the season opener behind "
                "a missing ingest. Record it first: "
                f"`{_INGEST_COMMAND.format(season=calendar_season)}`"
            )
            raise ScheduleNotRecordedError(msg, season=calendar_season)
        return _offseason(now, completed_season=season)

    days_since_last = (today - last["et_day"]).days
    ends_a_recorded_phase = game_type in _PLAYOFF_ROUNDS_BEFORE_SUPER_BOWL or (
        game_type == "REG" and week == _final_regular_week(season)
    )
    valid_days = NEXT_ROUND_VALID_DAYS.get(game_type)
    if (
        ends_a_recorded_phase
        and valid_days is not None
        and days_since_last <= valid_days
    ):
        return CurrentSlate(
            season, week + 1, in_season=True, basis=BASIS_NEXT_UNRECORDED_ROUND
        )

    msg = (
        f"the recorded {season} schedule ends at week {week} ({last['game_id']}, "
        f"{last['et_day']}), {days_since_last} day(s) before {today}, with no Super Bowl and "
        "nothing scheduled after it. That is not one unrecorded playoff round (the next round "
        "is certain only within "
        f"{NEXT_ROUND_VALID_DAYS} days of a playoff round or the final regular week), so the "
        f"current slate is unknown. Refresh the schedule: "
        f"`{_INGEST_COMMAND.format(season=season)}`"
    )
    raise ScheduleIncompleteError(msg, season=season, week=week + 1)


def resolve_current_slate(
    now: datetime | None = None, *, schedule: pd.DataFrame | None = None
) -> CurrentSlate:
    """The current slate, read from the recorded schedule (see the module docstring).

    Args:
        now: The instant asked about; timezone-aware. Defaults to ``datetime.now(ET)``.
        schedule: A silver ``games``-shaped frame to read instead of the recorded store --
            for tests and for callers that already hold the schedule.

    Returns:
        The :class:`CurrentSlate`.

    Raises:
        ValueError: *now* is naive.
        SlateResolutionError: one of the named refusals in the module docstring.
    """
    import numpy as np

    instant = _require_aware(now)
    frame = load_recorded_schedule() if schedule is None else prepare_schedule(schedule)
    if frame.empty:
        raise ScheduleUnavailableError(
            "the recorded schedule holds no games; the current slate is read from it and is "
            "never guessed from the calendar"
        )

    today = instant.astimezone(ET).date()
    ordinals = frame["et_ordinal"].to_numpy()
    upcoming = int(np.searchsorted(ordinals, today.toordinal(), side="left"))

    if upcoming == len(frame):
        return _slate_after_the_last_recorded_game(frame, today, instant)

    following = frame.iloc[upcoming]
    season, week = int(following["season"]), int(following["week"])
    previous_season = int(frame.iloc[upcoming - 1]["season"]) if upcoming > 0 else None
    if previous_season == season:
        return CurrentSlate(season, week, in_season=True, basis=BASIS_SCHEDULED_SLATE)
    return _opening_slate(frame, upcoming, today, instant)


def cli_target_season_week(
    season_arg: int | None, week_arg: str | None
) -> tuple[int, int | None]:
    """A trainer CLI's ``(season, week)`` target: explicit values win, else ONE resolution.

    Replaces ``args.season or get_current_nfl_season()`` plus
    ``int(args.week) if args.week else get_current_nfl_week()`` in the three trainers, which
    paired a CALENDAR season with a week that was the whole ``(season, week)`` TUPLE -- compared
    against the ``week`` column, so the default path could never filter to a week.

    Args:
        season_arg: ``--season`` as parsed, or ``None``.
        week_arg: ``--week`` as parsed: a number, ``"all"`` (every week) or ``None``.

    Returns:
        ``(season, week)``; ``week`` is ``None`` for ``"all"``.
    """
    if season_arg is not None and week_arg is not None:
        return season_arg, None if week_arg == "all" else int(week_arg)

    current = resolve_current_slate()
    season = current.season if season_arg is None else season_arg
    if week_arg is None:
        return season, current.week
    return season, None if week_arg == "all" else int(week_arg)


# ---------------------------------------------------------------------------
# The season a daily run refreshes (step 27b)
# ---------------------------------------------------------------------------


def refresh_target(
    now: datetime | None = None, *, schedule: pd.DataFrame | None = None
) -> tuple[int, int] | None:
    """The ``(season, week)`` a daily run captures and ingests to refresh the schedule, or None.

    * IN SEASON: the current slate.
    * THE CALENDAR SEASON HAS TURNED past a completed season whose successor is not recorded
      (:class:`ScheduleNotRecordedError`, August onward): that NEXT season's week 1 -- capturing
      it is how its schedule gets recorded, so the new season needs no manual switch.
    * OFFSEASON WITH THE NEXT SEASON RECORDED (August until the opener's lock day): the next
      season's week 1, so a moved opener is picked up from a current record (33.2 review B WR-03
      = C1 WR-03; this used to refresh the COMPLETED season, filed under "week 18").
    * OFFSEASON OTHERWISE (the spring): ``None`` -- nothing to refresh. The completed season
      cannot change, and capturing it daily appended ~200 git-tracked capture entries a year,
      then failed every day once that season was sealed.

    Every other refusal propagates.
    """
    try:
        slate = resolve_current_slate(now, schedule=schedule)
    except ScheduleNotRecordedError as due:
        if due.season is None:
            raise
        return due.season, 1
    if slate.in_season:
        return slate.as_tuple()
    following = slate.season + 1
    if first_recorded_kickoff(following, schedule=schedule) is not None:
        return following, 1
    return None


def _season_rows(season: int, schedule: pd.DataFrame | None) -> pd.DataFrame:
    frame = load_recorded_schedule() if schedule is None else prepare_schedule(schedule)
    return frame.loc[frame["season"] == season]


def season_is_complete(season: int, *, schedule: pd.DataFrame | None = None) -> bool:
    """Whether the recorded schedule holds *season*'s Super Bowl."""
    return _SUPER_BOWL in set(_season_rows(season, schedule)["game_type"])


def season_has_kicked_off(
    season: int, now: datetime, *, schedule: pd.DataFrame | None = None
) -> bool:
    """Whether any recorded game of *season* kicked off at or before *now*.

    ``False`` for a season the store does not hold yet: none of its games can be on record as
    played.
    """
    instant = _require_aware(now)
    kickoffs = [
        kickoff_wall_clock_et(value)
        for value in _season_rows(season, schedule)["kickoff_et"]
    ]
    return any(kickoff <= instant for kickoff in kickoffs)


def first_recorded_kickoff(
    season: int, *, schedule: pd.DataFrame | None = None
) -> datetime | None:
    """The earliest recorded kickoff of *season* (ET), or ``None`` when the store lacks it.

    The one fact that PROVES a season has not begun (step 27c): a recorded first kickoff later
    than an instant means no game of the season had been played at that instant. ``None`` proves
    nothing either way -- the season's schedule is simply not recorded yet.
    """
    kickoffs = [
        kickoff_wall_clock_et(value)
        for value in _season_rows(season, schedule)["kickoff_et"]
    ]
    return min(kickoffs, default=None)
