"""The schedule facts a game may use at its lock: ONE accessor (Plan 33.2-10 Task 4).

SPEC R8, D33.2-04, D33.2-21. Schedule and venue facts are known at the lock for the spring
schedule and for rule-bound changes. They are NOT known at the lock for a hurricane, a
roof collapse or a wildfire: a game moved by an emergency announced AFTER its lock must be
built with the facts as they stood BEFORE the move.

``config/schedule_moves.toml`` is the evidence table (one row per move, each with a dated
source and a verdict). Silver ``games`` stores the POST-MOVE facts for every game, so the
only place that knows what a game looked like at its lock is that table. This module reads
it and answers one question per game: which venue, which kickoff and which week may a
feature builder use for this game?

WHY ONE ACCESSOR. The venue, roof, surface, travel, time zone, weekday and weather location
all derive from the same two facts (``stadium_id`` and the kickoff). Letting each builder
re-implement "the last move announced at or before the lock" would give five chances to
disagree -- the D30-02 failure this project has paid for twice. ``facts_at_lock`` is the
only place the selection is made; every caller asks it.

THE SELECTION RULE (the table header states the same rule). Moves are ordered by
``move_index``. The game's facts at its lock are the facts after the LAST move whose
verdict is ``pre_lock``; with none, the facts before the first move. Every later move is
REVERTED, newest first, back to its ``from_value``. A ``scheduled_not_moved`` row reverts
to itself (its ``from_value`` equals its ``to_value``), so it never changes anything under
either verdict.

WHAT A REVERT CAN AND CANNOT RESTORE.
  venue  ``stadium_id`` becomes ``from_value``; roof, surface, coordinates, time zone and
         the weather location follow from the venue record.
  date   the kickoff's ET calendar date becomes ``from_value``. The table records DATES,
         not times, so the pre-move ET wall-clock time is taken to be the post-move one.
  week   the week number and the ET date become ``from_value``'s, on the same terms.
The TRAVEL ORIGIN (the away team's home venue) is a fact about the team, not the game, and
no move changes it.

EVERY REVERT IS CHECKED AGAINST THE ROW IT IS APPLIED TO. Before a move is undone, the fact
it names must equal its ``to_value``. Silver disagreeing with the table would otherwise
neutralise a game to facts that were never true, silently; it raises instead.

THE TABLE IS PARSED ONCE PER PROCESS. The builders call the accessor inside per-game
loops, so ``load_schedule_moves`` validates the table once into an immutable mapping, keyed
by the table's path, and ``facts_at_lock`` never touches TOML.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime
from functools import cache
from pathlib import Path
from types import MappingProxyType
from typing import Any

from utils.date_utils import kickoff_wall_clock_et

SCHEDULE_MOVES_PATH: Path = (
    Path(__file__).resolve().parent.parent / "config" / "schedule_moves.toml"
)

WHAT_MOVED_VOCABULARY: frozenset[str] = frozenset(
    {"venue", "date", "week", "scheduled_not_moved"}
)
VERDICTS: frozenset[str] = frozenset({"pre_lock", "post_lock"})

# The summary block that PUBLISHES the post-lock list (R8). The loader recomputes it from
# the rows and refuses a block that disagrees, so the published list cannot drift.
POST_LOCK_SUMMARY_KEY: str = "post_lock_summary"

_STADIUM_ID = r"[A-Z]{3}\d{2}"
_ISO_DATE = r"\d{4}-\d{2}-\d{2}"
_VALUE_PATTERNS: dict[str, re.Pattern[str]] = {
    "venue": re.compile(rf"^{_STADIUM_ID}$"),
    "date": re.compile(rf"^{_ISO_DATE}$"),
    "week": re.compile(rf"^W\d{{2}} {_ISO_DATE}$"),
    "scheduled_not_moved": re.compile(rf"^{_STADIUM_ID} {_ISO_DATE}$"),
}


class ScheduleMoveTableError(ValueError):
    """The move table is malformed, or disagrees with the game row it is applied to."""


@dataclass(frozen=True)
class ScheduleMove:
    """One validated row of the evidence table."""

    game_id: str
    move_index: int
    what_moved: str
    from_value: str
    to_value: str
    announced_at_utc: datetime | None
    lock_utc: datetime
    verdict: str
    source_url: str

    @property
    def is_real_move(self) -> bool:
        """True for a venue, date or week move; False for a scheduled game."""
        return self.what_moved != "scheduled_not_moved"


@dataclass(frozen=True)
class FactsAtLock:
    """The schedule facts a feature builder may use for one game.

    Attributes:
        game_id: The game.
        stadium_id: The venue as it stood at the lock. Venue, roof, surface, time zone and
            the WEATHER LOCATION all resolve from it through ``data/venues.json``.
        kickoff_et: The kickoff as it stood at the lock, tz-aware in America/New_York
            (the weekday is read from it), or None when the row carries no kickoff.
        week: The week number as it stood at the lock.
        neutralised_moves: The moves that were reverted, newest first. Empty when the
            game's facts are the facts silver records.
    """

    game_id: str | None
    stadium_id: Any
    kickoff_et: datetime | None
    week: Any
    neutralised_moves: tuple[ScheduleMove, ...] = ()

    @property
    def weather_stadium_id(self) -> Any:
        """The venue whose location the game's weather must describe."""
        return self.stadium_id

    @property
    def neutralised(self) -> bool:
        """True when at least one REAL move was reverted for this game."""
        return any(move.is_real_move for move in self.neutralised_moves)


def _parse_utc(value: object, *, field: str, where: str) -> datetime:
    """A tz-aware instant, or a refusal naming the row. A naive value is never relabelled."""
    text = str(value).strip()
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as error:
        raise ScheduleMoveTableError(
            f"{where}: {field} {text!r} is not an ISO-8601 timestamp"
        ) from error
    if parsed.tzinfo is None:
        raise ScheduleMoveTableError(
            f"{where}: {field} {text!r} carries no time zone. Every timestamp in the "
            "move table must be tz-aware; a naive value is refused, never relabelled."
        )
    return parsed


def _validated_move(row: Mapping[str, Any]) -> ScheduleMove:
    """One table row as a frozen record, or a refusal naming the row and the rule."""
    where = f"move {row.get('game_id')!r}#{row.get('move_index')!r}"
    what = str(row.get("what_moved", ""))
    if what not in WHAT_MOVED_VOCABULARY:
        raise ScheduleMoveTableError(
            f"{where}: what_moved {what!r} is outside {sorted(WHAT_MOVED_VOCABULARY)}"
        )
    verdict = str(row.get("verdict", ""))
    if verdict not in VERDICTS:
        raise ScheduleMoveTableError(
            f"{where}: verdict {verdict!r} is not one of {sorted(VERDICTS)}"
        )
    from_value, to_value = str(row.get("from_value", "")), str(row.get("to_value", ""))
    for field, value in (("from_value", from_value), ("to_value", to_value)):
        if not _VALUE_PATTERNS[what].match(value):
            raise ScheduleMoveTableError(
                f"{where}: {field} {value!r} is not a {what} value "
                f"({_VALUE_PATTERNS[what].pattern})"
            )
    if what == "scheduled_not_moved" and from_value != to_value:
        raise ScheduleMoveTableError(
            f"{where}: a scheduled_not_moved row must have from_value == to_value"
        )
    lock = _parse_utc(row.get("lock_utc", ""), field="lock_utc", where=where)
    announced_text = str(row.get("announced_at_utc", "")).strip()
    announced = (
        _parse_utc(announced_text, field="announced_at_utc", where=where)
        if announced_text
        else None
    )
    source_url = str(row.get("source_url", "")).strip()
    if verdict == "pre_lock":
        if not source_url:
            raise ScheduleMoveTableError(
                f"{where}: a pre_lock move needs a source_url (D33.2-21); without one "
                "the move is post_lock by default"
            )
        if announced is None or announced > lock:
            raise ScheduleMoveTableError(
                f"{where}: a pre_lock move must be announced AT or before its lock "
                f"({lock.isoformat()}); announced_at_utc is {announced_text!r}"
            )
    return ScheduleMove(
        game_id=str(row["game_id"]),
        move_index=int(row["move_index"]),
        what_moved=what,
        from_value=from_value,
        to_value=to_value,
        announced_at_utc=announced,
        lock_utc=lock,
        verdict=verdict,
        source_url=source_url,
    )


def _parse_schedule_move_table(path: Path) -> dict[str, Any]:
    """Read and TOML-parse the table. The ONE parse; tests spy on it to count calls."""
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _post_lock_summary(
    moves: Mapping[str, tuple[ScheduleMove, ...]],
) -> dict[str, list[str]]:
    """The post-lock list the rows imply, split into real moves and scheduled games."""
    real: set[str] = set()
    scheduled: set[str] = set()
    for game_id, game_moves in moves.items():
        for move in game_moves:
            if move.verdict != "post_lock":
                continue
            (real if move.is_real_move else scheduled).add(game_id)
    return {"real_moves": sorted(real), "scheduled_not_moved": sorted(scheduled)}


@cache
def _load_cached(resolved_path: str) -> Mapping[str, tuple[ScheduleMove, ...]]:
    table = _parse_schedule_move_table(Path(resolved_path))
    grouped: dict[str, list[ScheduleMove]] = {}
    for row in table.get("moves", []):
        move = _validated_move(row)
        grouped.setdefault(move.game_id, []).append(move)
    ordered: dict[str, tuple[ScheduleMove, ...]] = {}
    for game_id, game_moves in grouped.items():
        game_moves.sort(key=lambda move: move.move_index)
        indexes = [move.move_index for move in game_moves]
        if indexes != list(range(1, len(game_moves) + 1)):
            raise ScheduleMoveTableError(
                f"{game_id}: move_index values {indexes} are not 1-based and contiguous"
            )
        locks = {move.lock_utc for move in game_moves}
        if len(locks) != 1:
            raise ScheduleMoveTableError(
                f"{game_id}: its moves disagree about the lock ({sorted(locks)})"
            )
        ordered[game_id] = tuple(game_moves)
    frozen = MappingProxyType(ordered)

    published = table.get(POST_LOCK_SUMMARY_KEY)
    if published is not None:
        implied = _post_lock_summary(frozen)
        stated = {key: sorted(published.get(key, [])) for key in implied}
        if stated != implied:
            raise ScheduleMoveTableError(
                f"the published [{POST_LOCK_SUMMARY_KEY}] block disagrees with the rows: "
                f"it states {stated}, the rows imply {implied}. Regenerate the block "
                "from the rows; never edit it by hand."
            )
    return frozen


def load_schedule_moves(
    path: Path | str = SCHEDULE_MOVES_PATH,
) -> Mapping[str, tuple[ScheduleMove, ...]]:
    """The validated move table: ``game_id`` -> its moves in ``move_index`` order.

    Parsed and validated ONCE per path per process; the returned mapping is read-only and
    its records are frozen. A malformed table raises here, at load, rather than partway
    through a build.

    Raises:
        ScheduleMoveTableError: a row breaks the table's contract, or the published
            post-lock block disagrees with the rows.
    """
    return _load_cached(str(Path(path).resolve()))


def post_lock_summary(
    path: Path | str = SCHEDULE_MOVES_PATH,
) -> dict[str, list[str]]:
    """The published post-lock list, recomputed from the rows (real moves apart from
    scheduled games, so a reader is never told a Super Bowl was neutralised)."""
    return _post_lock_summary(load_schedule_moves(path))


def _is_null(value: object) -> bool:
    """True for None, NaN and NaT; never raises on an array-like."""
    if value is None:
        return True
    import pandas as pd  # local import: the accessor's hot path never needs it

    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _et_date(kickoff: datetime) -> str:
    return kickoff.date().isoformat()


def _with_et_date(kickoff: datetime, iso_date: str) -> datetime:
    """*kickoff* moved to *iso_date* at the same ET wall-clock time."""
    target = date.fromisoformat(iso_date)
    naive = datetime.combine(target, kickoff.timetz().replace(tzinfo=None))
    return kickoff_wall_clock_et(naive)


def _revert(
    move: ScheduleMove, stadium_id: Any, kickoff: datetime | None, week: Any
) -> tuple[Any, datetime | None, Any]:
    """Undo *move*, after checking the facts it lands on are the facts in hand."""
    where = f"{move.game_id}#{move.move_index} ({move.what_moved})"
    if move.what_moved in {"date", "week"} and kickoff is None:
        raise ScheduleMoveTableError(
            f"{where}: the game row carries no kickoff, so a {move.what_moved} move "
            "cannot be reverted. Refusing rather than guessing the pre-move kickoff."
        )
    if move.what_moved == "venue":
        if stadium_id != move.to_value:
            raise ScheduleMoveTableError(
                f"{where}: the move lands on {move.to_value!r} but the game row names "
                f"{stadium_id!r}. Silver and the table disagree; refusing to neutralise "
                "to facts that were never true."
            )
        return move.from_value, kickoff, week
    if move.what_moved == "date":
        if _et_date(kickoff) != move.to_value:
            raise ScheduleMoveTableError(
                f"{where}: the move lands on {move.to_value} but the kickoff's ET date "
                f"is {_et_date(kickoff)}. Refusing to neutralise."
            )
        return stadium_id, _with_et_date(kickoff, move.from_value), week
    if move.what_moved == "week":
        to_week, to_date = move.to_value.split(" ")
        from_week, from_date = move.from_value.split(" ")
        if f"W{int(week):02d}" != to_week or _et_date(kickoff) != to_date:
            raise ScheduleMoveTableError(
                f"{where}: the move lands on {move.to_value!r} but the game row is week "
                f"{week} on {_et_date(kickoff)}. Refusing to neutralise."
            )
        return stadium_id, _with_et_date(kickoff, from_date), int(from_week[1:])
    return stadium_id, kickoff, week  # scheduled_not_moved: from == to, nothing to undo


def facts_at_lock(
    game_id: Any,
    games_row: Mapping[str, Any],
    *,
    table_path: Path | str = SCHEDULE_MOVES_PATH,
) -> FactsAtLock:
    """The venue, kickoff and week a feature builder may use for *game_id*.

    THE ONE PLACE the last-admissible-move selection is made. A game with no row, or
    whose every move is ``pre_lock``, gets the facts *games_row* records. A game whose
    moves after its last ``pre_lock`` move include a ``post_lock`` one gets those moves
    reverted, newest first.

    Args:
        game_id: The game. ``None`` (a synthetic row) is a game with no row.
        games_row: The silver games row: ``stadium_id``, ``kickoff_et`` and ``week`` are
            read (each may be absent). Any mapping-like row works, including a
            ``pandas.Series``.
        table_path: The move table; a test points it at a temporary table.

    Returns:
        The facts at the lock.

    Raises:
        ScheduleMoveTableError: the table is malformed, or a move to revert does not land
            on the facts the row records.
    """
    stadium_id = games_row.get("stadium_id")
    raw_kickoff = games_row.get("kickoff_et")
    kickoff = None if _is_null(raw_kickoff) else kickoff_wall_clock_et(raw_kickoff)
    week = games_row.get("week")
    moves = load_schedule_moves(table_path).get(str(game_id)) if game_id else None
    if not moves:
        return FactsAtLock(game_id, stadium_id, kickoff, week)

    last_pre_lock = max(
        (move.move_index for move in moves if move.verdict == "pre_lock"), default=0
    )
    to_revert = [move for move in moves if move.move_index > last_pre_lock]
    if not any(move.verdict == "post_lock" for move in to_revert):
        return FactsAtLock(game_id, stadium_id, kickoff, week)

    reverted: list[ScheduleMove] = []
    for move in reversed(to_revert):
        stadium_id, kickoff, week = _revert(move, stadium_id, kickoff, week)
        reverted.append(move)
    return FactsAtLock(game_id, stadium_id, kickoff, week, tuple(reverted))
