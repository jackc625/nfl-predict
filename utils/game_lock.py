"""THE per-game lock rule: one deterministic rule, with its evidence beside it (D33.2-01).

This module IS the rule, not a description of one. Each game's information locks at
18:00 America/New_York on the ET CALENDAR DAY BEFORE its kickoff. Information timed AT
the lock is admissible (``<=``); information timed one second after it is not.

It REPLACES the preceding-Friday freeze (``scripts.ingest_historical_odds.
get_synthetic_snapshot_ts``) everywhere a freeze decides what is known: the gold
information-time check, the Elo replay, the builders, weekly bet-list selection and live
capture stamping. D33.2-01 says the other derivations are "retired, not reused", so this
module is not a second rule standing beside the Friday one -- it is the one that is left
once the sweep (Plan 33.2-02) finishes. Two rules on disk would let two answers disagree
silently, which is the D30-02 failure this project has already paid for twice.

WHY THIS MODULE IS A LEAF, AND MUST STAY ONE
--------------------------------------------
The obvious home is beside ``get_synthetic_snapshot_ts`` in
``scripts/ingest_historical_odds.py``. It cannot go there: that module imports
``backtest.ev_chain_constants``, which cycles back through ``backtest.bet_selector``
(documented at ``backtest/bet_selector.py:255-258``). ``backtest/weekly_bet_list.py``
already has to import the Friday rule LAZILY inside two functions for exactly that
reason. The gold fence lives under ``features/``, which must be able to import the lock
EAGERLY -- so the lock lives in a module that imports only the standard library at module
scope. The two project helpers it reuses (``utils.date_utils.kickoff_wall_clock_et`` and
``scripts.ingest_historical_odds.require_aware_snapshot_ts``) and pandas are imported
LAZILY, inside the functions that need them -- the same discipline
``utils/date_utils.py:371-409`` already uses for pandas. A test asserts that a
fresh-interpreter IMPORT of this module drags in no ``backtest``, ``models``,
``features`` or ``scripts`` package. The assertion is scoped to the IMPORT, not to a
call: the first CALL legitimately reaches ``scripts`` through the lazy import.

WHY THE NAIVE REFUSAL IS REUSED, NOT HAND-ROLLED
------------------------------------------------
``require_aware_snapshot_ts`` is the project's ONE strict naive-instant parser (D33-27).
It checks naiveness on the PARSED value, so a naive ``datetime``, a naive
``pandas.Timestamp``, a naive ISO string and a bare ``numpy.datetime64`` all reach the
same refusal. A hand-rolled ``tzinfo is None`` check here would be a second parse path
that answers the same question differently for at least one of those shapes.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from zoneinfo import ZoneInfo

if TYPE_CHECKING:  # pragma: no cover - typing only; pandas stays a lazy import
    import pandas as pd

__all__ = [
    "ET",
    "LOCK_HOUR_ET",
    "LOCK_RULE_EVIDENCE",
    "MissingKickoffError",
    "game_lock",
    "is_admissible",
    "lock_frame",
]

#: The zone the rule is stated in. The lock's WALL CLOCK is ET; its UTC offset is whatever
#: is in force on the LOCK date, so the rule survives both daylight-saving transitions.
ET: ZoneInfo = ZoneInfo("America/New_York")

#: The lock hour, ET, on the calendar day before kickoff. MEASURED read-only 2026-09-15 on
#: silver ``games`` 2002-2026 with week-keyed prior-game windows and a 4 h game duration:
#: 0 games whose inputs include a result that ended after their lock, at EVERY hour tested
#: from 09:00 to 23:59 ET -- the hour is not load-bearing for outcome leakage. The retired
#: preceding-Friday rule produced 324 such games (310 Thursday, 11 Friday, 3 Wednesday).
#: Lock-to-kickoff gap at 18:00: 15.0 to 29.5 hours (D33.2-01).
LOCK_HOUR_ET: int = 18

#: The citations behind the rule, as strings, so a readout quotes them instead of
#: re-deriving them. The prose form is the module docstring and the comment on
#: ``LOCK_HOUR_ET``; this is the machine-readable index of both.
LOCK_RULE_EVIDENCE: tuple[str, ...] = (
    "D33.2-01 (owner ruling 2026-09-15): each game locks at 18:00 America/New_York on the "
    "ET calendar day before its kickoff; at-lock information is admissible (<=).",
    "MEASURED 2026-09-15 on silver games 2002-2026, week-keyed prior-game windows, 4 h game "
    "duration: 0 games whose inputs include a result that ended after their lock, at every "
    "hour tested from 09:00 to 23:59 ET.",
    "The retired preceding-Friday 18:00 ET rule (get_synthetic_snapshot_ts) produced 324 "
    "such games: 310 Thursday, 11 Friday, 3 Wednesday.",
    "Lock-to-kickoff gap under the 18:00 day-before rule: 15.0 to 29.5 hours.",
    "The ET DATE is read through utils.date_utils.kickoff_wall_clock_et, which CONVERTS an "
    "aware kickoff and never relabels it; reading games.kickoff_et the other way turns 156 "
    "night games into phantom Friday kickoffs.",
)


class MissingKickoffError(ValueError):
    """A game has no kickoff time, so it cannot have a lock.

    Raised by name rather than defaulted. A game with no kickoff has no ET calendar day
    and therefore no day before it; any instant substituted here -- the build clock, a
    week's Friday, the schedule's gameday at midnight -- would be a manufactured lock that
    makes every comparison against it meaningless.

    A ``ValueError`` so a caller that already handles a bad schedule value handles this
    one; the distinct type is what lets a test assert WHICH refusal it got.
    """


def _is_missing(value: Any) -> bool:
    """True when *value* is a scalar null (None, NaN, NaT). Array-likes are never null."""
    if value is None:
        return True
    import pandas as pd  # local import: this module is a leaf at import time

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _require_aware(value: Any) -> datetime:
    """*value* as a UTC-aware instant through the project's ONE strict parser.

    Lazy import, deliberately: ``scripts.ingest_historical_odds`` cycles through
    ``backtest.bet_selector`` and must not be reached at this module's import time.
    """
    from scripts.ingest_historical_odds import require_aware_snapshot_ts

    return require_aware_snapshot_ts(value)


def game_lock(kickoff_value: Any, *, game_id: str | None = None) -> datetime:
    """18:00 ET on the ET calendar day BEFORE *kickoff_value*.

    The ET date is derived through ``utils.date_utils.kickoff_wall_clock_et`` -- THE one
    accessor for ``games.kickoff_et``, which converts an aware value and never relabels
    it. One day is subtracted from the ET DATE (never from the instant, which would put a
    DST transition inside the subtraction), and the lock is built on that date at
    ``LOCK_HOUR_ET`` in ``ET``, so its UTC offset is the one in force on the LOCK date.

    Args:
        kickoff_value: A kickoff instant -- a ``datetime``, a ``pandas.Timestamp`` or
            anything the strict parser accepts. It MUST be timezone-aware.
        game_id: The game the kickoff belongs to, named in the refusal when supplied.

    Returns:
        The lock as a tz-aware ``datetime`` in ``ET``.

    Raises:
        MissingKickoffError: when *kickoff_value* is null, NaN or NaT.
        scripts.ingest_historical_odds.NaiveTimestampError: when *kickoff_value* carries no
            timezone. A naive kickoff is refused, never relabelled.
    """
    if _is_missing(kickoff_value):
        named = f" for game {game_id}" if game_id is not None else ""
        msg = (
            f"no kickoff time{named}: a game with no kickoff has no ET calendar day and "
            "therefore no lock. Refusing rather than substituting a manufactured instant."
        )
        raise MissingKickoffError(msg)

    from utils.date_utils import kickoff_wall_clock_et

    aware_kickoff = _require_aware(kickoff_value)
    kickoff_et_date = kickoff_wall_clock_et(aware_kickoff).date()
    lock_date = kickoff_et_date - timedelta(days=1)
    return datetime.combine(lock_date, time(LOCK_HOUR_ET, 0), tzinfo=ET)


def lock_frame(games_df: pd.DataFrame) -> pd.Series:
    """One lock per game, ``game_id``-indexed and tz-aware -- built ONCE per build.

    Every value is :func:`game_lock` applied to that game's ``kickoff_et``, so the frame is
    the rule applied per game rather than a second, vectorised derivation of it.

    Args:
        games_df: A frame carrying ``game_id`` and ``kickoff_et`` (silver ``games`` shape).

    Returns:
        A ``Series`` named ``lock``, indexed by ``game_id``, of tz-aware ET instants.

    Raises:
        MissingKickoffError: naming EVERY game whose kickoff is null, or when the
            ``kickoff_et`` column is absent altogether.
        ValueError: naming the repeated ids when a ``game_id`` appears twice -- a lock frame
            with two locks for one game has no single answer to give.
    """
    import pandas as pd  # local import: this module is a leaf at import time

    for column in ("game_id", "kickoff_et"):
        if column not in games_df.columns:
            msg = (
                f"cannot build a lock frame: the games frame has no {column!r} column "
                f"(columns: {sorted(games_df.columns)})"
            )
            raise MissingKickoffError(msg)

    duplicated = sorted(
        {str(gid) for gid in games_df.loc[games_df["game_id"].duplicated(), "game_id"]}
    )
    if duplicated:
        msg = (
            f"cannot build a lock frame: {len(duplicated)} game_id(s) appear more than "
            f"once, e.g. {duplicated[:10]}"
        )
        raise ValueError(msg)

    missing_mask = games_df["kickoff_et"].isna()
    if bool(missing_mask.any()):
        offenders = sorted(map(str, games_df.loc[missing_mask, "game_id"]))
        msg = (
            f"{len(offenders)} game(s) have no kickoff time and therefore no lock: "
            f"{offenders}. Refusing the whole lock frame rather than dropping them."
        )
        raise MissingKickoffError(msg)

    locks = [
        game_lock(kickoff, game_id=str(gid))
        for gid, kickoff in zip(
            games_df["game_id"], games_df["kickoff_et"], strict=True
        )
    ]
    index = pd.Index(games_df["game_id"].astype(str), name="game_id")
    return pd.Series(pd.DatetimeIndex(locks), index=index, name="lock")


def is_admissible(information_time: Any, lock: Any) -> bool:
    """True when *information_time* is AT or BEFORE *lock*: ``information_time <= lock``.

    AT-LOCK IS ADMISSIBLE. The operator below is ``<=``, and this comment says so because
    it is the code, not a hope about it -- the WR-14 note at ``features/validation.py``
    records why documenting an operator the code does not implement is worse than
    documenting nothing.

    Both operands go through the project's ONE strict parser first, so a naive value on
    EITHER side raises rather than being relabelled or aligned. A null value raises the
    parser's own ``ValueError``: a missing information time is never admissible by default.

    Args:
        information_time: When the information was known. Must be timezone-aware.
        lock: The game's lock, normally from :func:`game_lock`. Must be timezone-aware.

    Returns:
        ``True`` when the information was known at or before the lock.
    """
    return _require_aware(information_time) <= _require_aware(lock)
