"""The Phase 37 records: append-only JSONL files under ``<ledger>/refit/`` (D-19, SPEC R8, R9).

WHERE THEY LIVE, AND WHY
------------------------
Every Phase 37 record lives inside the ledger directory, under :data:`REFIT_DIRNAME`, so the
Phase 34 private backup carries it and the ledger stays self-contained. The record kinds are:

* the shadow hole (:data:`SHADOW_HOLES_FILENAME`): a new live row written without a frozen shadow
  twin, with the reason named (an exception class, or ``no_shadow_row``); recorded, never
  back-filled;
* the cadence-date event (:data:`DATES_FILENAME`): every ``attempt``, ``install_intent`` and
  ``closed`` event of a pre-registered re-fit date (:func:`append_date_event` names the fields);
* the A/A week (:data:`AA_WEEKS_FILENAME`): one week's A/A result, ``clean`` or not;
* the bet rule (:data:`BET_RULES_DIRNAME`): each generation's re-measured numbers, one
  content-addressed, write-once JSON named by its sha256.

APPEND-ONLY, THROUGH THE WRITE SINK
-----------------------------------
Each record is one JSON line appended through ``data.write_sink.append_jsonl``, so a dry run
records the intended write and persists nothing, and the test-write guard sees every write. Every
cadence-date event and every bet-rule file is flushed and fsynced: the ``install_intent`` and the
bet rule before the install's ``latest.json`` swap can follow them, matching that manifest's own
fsync, and the ``attempt`` and ``closed`` events because the decision path reads the same file.

ONE RULE, ENFORCED BOTH WAYS
----------------------------
Each record file has one private validator. The writer refuses a record it names a problem in
(``ValueError``, nothing written); the reader refuses a stored line it names a problem in
(:class:`RefitRecordFormatError` naming the file, the 1-based line and the field). A well-formed
JSON line with a missing or out-of-vocabulary field is never read as a valid record.

A MALFORMED RECORD IS NEVER READ AS "NO RECORDS"
------------------------------------------------
:func:`read_records` returns ``[]`` only for an absent file. A bad line raises
:class:`RefitRecordFormatError`; when it is the file's final line and ends in no newline (the
shape a crash mid-append leaves) the message gives the torn-append repair, and for any other line
it says to compare with the private backup instead. An append to a file that ends in a torn line
is refused by name (WR-B01): it would put a whole record on the torn line, which that repair then
deletes.

ONE STORED BET-RULE PATH
------------------------
An ``install_intent`` stores ``bet_rule_path`` as the ledger-relative POSIX text built by
:func:`bet_rule_relpath`, and only :func:`bet_rule_record_path` joins a ledger dir to it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import subprocess
import uuid
from collections.abc import Callable, Iterable, Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from backtest.refit_preregistration_2026 import REFIT_DATE_LISTS
from data.write_sink import append_jsonl, current_sink
from utils.file_lock import exclusive_file_lock
from utils.logging_config import get_logger

__all__ = [
    "AA_WEEKS_FILENAME",
    "ATTEMPT_LOCK_FILENAME",
    "ATTEMPT_OUTCOMES",
    "BET_RULES_DIRNAME",
    "CLOSED_STATUSES",
    "DATES_FILENAME",
    "DATE_EVENTS",
    "REFIT_CODE_PATHS",
    "REFIT_DIRNAME",
    "SHADOW_HOLES_FILENAME",
    "RefitRecordFormatError",
    "append_aa_record",
    "append_date_event",
    "append_record",
    "attempt_lock",
    "bet_rule_record_path",
    "bet_rule_relpath",
    "code_commit",
    "read_aa_records",
    "read_bet_rule_record",
    "read_date_events",
    "read_records",
    "record_shadow_holes",
    "refit_dir",
    "write_bet_rule_record",
]

logger = get_logger(__name__)

REFIT_DIRNAME: str = "refit"
SHADOW_HOLES_FILENAME: str = "shadow_holes.jsonl"
DATES_FILENAME: str = "dates.jsonl"
AA_WEEKS_FILENAME: str = "aa_weeks.jsonl"
BET_RULES_DIRNAME: str = "bet_rules"
# The private backup ignores ``*.lock``.
ATTEMPT_LOCK_FILENAME: str = "attempt.lock"

# The closed vocabularies of a cadence-date event.
DATE_EVENTS: tuple[str, ...] = ("attempt", "install_intent", "closed")
ATTEMPT_OUTCOMES: tuple[str, ...] = ("installed", "refused", "crashed")
CLOSED_STATUSES: tuple[str, ...] = ("ran", "ran late", "not run")

# The code paths whose uncommitted change marks a re-fit's code commit ``+dirty``: every tracked
# path that produces or loads the training data. ``ratings`` is the Elo engine every Elo feature
# comes out of; ``pyproject.toml`` and ``uv.lock`` fix the library versions every fit runs on.
# ``api`` is imported only for ledger status constants, so it is not listed. A whole-tree check
# is not used: tracked ``.planning/`` files and the daily run's own ``config/upstream_*`` writes
# are always modified, and would refuse every re-fit.
REFIT_CODE_PATHS: tuple[str, ...] = (
    "refit",
    "forward_ledger",
    "backtest",
    "models",
    "features",
    "ratings",
    "pipeline",
    "scripts",
    "conf",
    "utils",
    "data",
    "config/*.py",
    "config/*.toml",
    "pyproject.toml",
    "uv.lock",
)

# The write-sink kinds.
_SHADOW_HOLE_KIND: str = "refit_shadow_hole"
_RECORD_KIND: str = "refit_record"
_BET_RULE_KIND: str = "refit_bet_rule"

_DIRTY_SUFFIX: str = "+dirty"
_GIT_TIMEOUT_SECONDS: int = 30
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CODE_COMMIT_PATTERN = re.compile(r"^[0-9a-f]{40}(\+dirty)?$")

# Keys every cadence-date event carries, then the keys each event type adds.
_DATE_EVENT_COMMON_KEYS: tuple[str, ...] = (
    "event",
    "list_week",
    "date_index",
    "date",
    "nfl_week",
)
_DATE_EVENT_KEYS: dict[str, tuple[str, ...]] = {
    "attempt": (
        "cutoff_utc",
        "attempted_at_utc",
        "run_date_et",
        "code_commit",
        "outcome",
        "check",
        "exception_class",
        "before_ids",
        "after_ids",
    ),
    "install_intent": ("ids", "bet_rule_path", "bet_rule_sha256", "code_commit"),
    "closed": ("status", "closed_at_utc", "before_ids", "after_ids"),
}
_AA_RECORD_KEYS: tuple[str, ...] = ("season", "week", "clean")

_TORN_LAST_LINE_REPAIR: str = "a torn last line is a crash mid-append: copy the file aside, then delete that one line"
_DAMAGED_LINE_ADVICE: str = (
    "a bad line that ends in a newline is damage, not a torn append: compare the file with "
    "the private backup before changing anything"
)
# The bytes a whole last line ends in: a text-mode write ends ``\r\n`` on Windows, and a tear
# between the two leaves a complete line that the universal-newline reader still splits.
_LINE_END_BYTES: tuple[bytes, ...] = (b"\n", b"\r")


class RefitRecordFormatError(Exception):
    """A Phase 37 record file holds a bad line or a bet-rule file that no longer hashes to its name.

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): an unreadable record must never
    be degraded into "no records".

    Attributes:
        location: Which record, built by this module from the path and the line number only
            (``ledger/refit/dates.jsonl line 7``), never from record content, so the daily run
            may print and push it (WR-C05); None when no file is named.
    """

    def __init__(self, message: str, *, location: str | None = None) -> None:
        super().__init__(message)
        self.location = location


def refit_dir(ledger_dir: Path | str) -> Path:
    """The directory holding every Phase 37 record of the ledger at *ledger_dir*."""
    return Path(ledger_dir) / REFIT_DIRNAME


def append_record(
    ledger_dir: Path | str,
    filename: str,
    record: Mapping[str, Any],
    *,
    kind: str,
    durable: bool = False,
) -> bool:
    """Append *record* to ``<ledger>/refit/<filename>``, stamped ``recorded_at_utc``.

    Args:
        durable: Flush and fsync the line before returning (install-critical records).

    Returns:
        True when the line was written (False under a recording sink).

    Raises:
        RefitRecordFormatError: the file ends in a torn line; nothing is appended.
    """
    path = refit_dir(ledger_dir) / filename
    _refuse_a_torn_tail(path)
    stamped = {**record, "recorded_at_utc": datetime.now(tz=UTC).isoformat()}
    return append_jsonl(path, stamped, kind=kind, durable=durable)


def _refuse_a_torn_tail(path: Path) -> None:
    """Refuse to append to *path* when its last byte ends no line (a crash mid-append).

    Appending would put a whole record on the torn line, and the torn-line repair, which deletes
    that line, would then delete the record too. The torn line needs a human first.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("rb") as handle:
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) in _LINE_END_BYTES:
            return
    msg = (
        f"refit record {path.as_posix()} ends in a torn line, so nothing was appended; "
        f"{_TORN_LAST_LINE_REPAIR}"
    )
    raise RefitRecordFormatError(msg, location=f"{path.as_posix()} last line")


def read_records(ledger_dir: Path | str, filename: str) -> list[dict[str, Any]]:
    """Every record in ``<ledger>/refit/<filename>``, oldest first; ``[]`` when it is absent.

    Raises:
        RefitRecordFormatError: a non-blank line is not one JSON object.
    """
    return _read_validated(refit_dir(ledger_dir) / filename, None)


def _read_validated(
    path: Path,
    problem_of: Callable[[Mapping[str, Any]], str | None] | None,
    *,
    skip_torn_tail: bool = False,
) -> list[dict[str, Any]]:
    """Every JSON-object line of *path*, each passed through *problem_of* when it is given.

    With *skip_torn_tail*, an unterminated final line that is not JSON (a crash mid-append, never
    a whole record) is logged by name and left out; every other bad line still raises.
    """
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    numbered = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    # Only an unterminated final line can be a torn append; any other bad line is damage.
    torn_number = len(lines) if lines and not text.endswith("\n") else 0
    records: list[dict[str, Any]] = []
    for number, line in numbered:
        torn = number == torn_number
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            detail = f"is not JSON: {error}"
            if torn and skip_torn_tail:
                logger.error(
                    "Re-fit record torn last line left out of the read",
                    path=path.as_posix(),
                    line=number,
                )
                break
            raise _line_error(path, number, torn, detail) from error
        if not isinstance(record, dict):
            raise _line_error(path, number, torn, "is not one JSON object")
        problem = problem_of(record) if problem_of is not None else None
        if problem is not None:
            raise _line_error(path, number, torn, problem)
        records.append(record)
    return records


def _line_error(
    path: Path, number: int, torn: bool, detail: str
) -> RefitRecordFormatError:
    """The error naming the file, the line, the problem and the safe repair."""
    advice = _TORN_LAST_LINE_REPAIR if torn else _DAMAGED_LINE_ADVICE
    location = f"{path.as_posix()} line {number}"
    return RefitRecordFormatError(
        f"refit record {location} {detail}; {advice}", location=location
    )


def record_shadow_holes(
    ledger_dir: Path | str,
    holes: Iterable[tuple[tuple[str, int, int, str], str]],
    *,
    decided_at: datetime,
) -> None:
    """Append one record per hole: the live row it names, the reason and the decision instant.

    Args:
        ledger_dir: The ledger directory.
        holes: ``((game_id, season, week, target), reason)`` per new live row with no shadow twin.
        decided_at: The decision instant of the run that wrote the live rows (tz-aware).
    """
    for (game_id, season, week, target), reason in holes:
        append_record(
            ledger_dir,
            SHADOW_HOLES_FILENAME,
            {
                "game_id": game_id,
                "season": season,
                "week": week,
                "target": target,
                "reason": reason,
                "decided_at_utc": decided_at.isoformat(),
            },
            kind=_SHADOW_HOLE_KIND,
        )


# ---------------------------------------------------------------------------
# The one bet-rule path convention
# ---------------------------------------------------------------------------


def bet_rule_relpath(sha256: str) -> str:
    """The ledger-relative POSIX path an ``install_intent`` stores for the bet rule *sha256*.

    Raises:
        ValueError: *sha256* is not 64 lowercase hex characters.
    """
    if not isinstance(sha256, str) or not _SHA256_PATTERN.match(sha256):
        msg = f"a bet-rule sha256 must be 64 lowercase hex characters, got {sha256!r}"
        raise ValueError(msg)
    return f"{REFIT_DIRNAME}/{BET_RULES_DIRNAME}/{sha256}.json"


def bet_rule_record_path(ledger_dir: Path | str, sha256: str) -> Path:
    """The bet-rule file of *sha256* in the ledger at *ledger_dir* (the one join of a stored path)."""
    return Path(ledger_dir) / bet_rule_relpath(sha256)


# ---------------------------------------------------------------------------
# Cadence-date events
# ---------------------------------------------------------------------------


def _vocabulary_problem(
    field: str, value: object, vocabulary: tuple[str, ...]
) -> str | None:
    if value in vocabulary:
        return None
    return (
        f"field {field!r} value {value!r} is not in the closed vocabulary {vocabulary}"
    )


def _missing_key_problem(record: Mapping[str, Any], keys: Iterable[str]) -> str | None:
    for key in keys:
        if key not in record:
            return f"field {key!r} is missing"
    return None


def _date_event_problem(event: Mapping[str, Any]) -> str | None:
    """The first problem with a cadence-date event, or None when it is valid."""
    problem = _missing_key_problem(event, ("event",)) or _vocabulary_problem(
        "event", event["event"], DATE_EVENTS
    )
    if problem is not None:
        return problem
    kind = event["event"]
    problem = _missing_key_problem(
        event, (*_DATE_EVENT_COMMON_KEYS, *_DATE_EVENT_KEYS[kind])
    ) or _date_identity_problem(event)
    if problem is not None:
        return problem
    if kind == "attempt":
        return _vocabulary_problem("outcome", event["outcome"], ATTEMPT_OUTCOMES)
    if kind == "closed":
        return _vocabulary_problem("status", event["status"], CLOSED_STATUSES)
    return _install_intent_problem(event)


def _date_identity_problem(event: Mapping[str, Any]) -> str | None:
    """Whether the event names one frozen cadence date exactly (WR-B02).

    ``list_week``, ``date_index`` and ``nfl_week`` must be plain ints and ``date`` the ISO text of
    that list's date, so every reader matches the event to its date whether it coerces or not.
    """
    for key in ("list_week", "date_index", "nfl_week"):
        value = event[key]
        if not isinstance(value, int) or isinstance(value, bool):
            return f"field {key!r} value {value!r} is not an integer"
    dates = REFIT_DATE_LISTS.get(event["list_week"])
    if dates is None:
        return (
            f"field 'list_week' value {event['list_week']!r} names no pre-registered list "
            f"{sorted(REFIT_DATE_LISTS)}"
        )
    if not 1 <= event["date_index"] <= len(dates):
        return (
            f"field 'date_index' value {event['date_index']!r} is outside 1-{len(dates)} for "
            f"list week {event['list_week']}"
        )
    day, nfl_week = dates[event["date_index"] - 1]
    if event["date"] != day or event["nfl_week"] != nfl_week:
        return (
            f"fields 'date' and 'nfl_week' ({event['date']!r}, {event['nfl_week']!r}) are not "
            f"list week {event['list_week']} date {event['date_index']} ({day!r}, {nfl_week})"
        )
    return None


def _install_intent_problem(event: Mapping[str, Any]) -> str | None:
    """The bet-rule path and code-commit forms an ``install_intent`` must hold."""
    sha = event["bet_rule_sha256"]
    try:
        expected = bet_rule_relpath(sha)
    except ValueError as error:
        return f"field 'bet_rule_sha256': {error}"
    stored = event["bet_rule_path"]
    if stored != expected:
        return (
            f"field 'bet_rule_path' value {stored!r} is not the one stored convention "
            f"bet_rule_relpath(bet_rule_sha256) = {expected!r}"
        )
    commit = event["code_commit"]
    if commit is not None and not (
        isinstance(commit, str) and _CODE_COMMIT_PATTERN.match(commit)
    ):
        return (
            f"field 'code_commit' value {commit!r} is not None, 40 lowercase hex, "
            "or 40 lowercase hex followed by '+dirty'"
        )
    return None


def append_date_event(ledger_dir: Path | str, event: Mapping[str, Any]) -> bool:
    """Append one cadence-date event to ``<ledger>/refit/dates.jsonl``.

    Field names (Plans 37-05, 37-11, 37-13, 37-14 read them):

    * every event: ``event`` (one of :data:`DATE_EVENTS`), ``list_week``, ``date_index``
      (1-based), ``date`` (ISO), ``nfl_week`` -- the three numbers plain ints, and the date and
      NFL week exactly that frozen list's entry (``REFIT_DATE_LISTS``);
    * ``attempt`` adds ``cutoff_utc``, ``attempted_at_utc``, ``run_date_et``, ``code_commit``,
      ``outcome`` (one of :data:`ATTEMPT_OUTCOMES`), ``check``, ``exception_class``,
      ``before_ids``, ``after_ids``, and may carry ``no_final_score_ids`` (the date's eligibility
      snapshot, read at its first attempt and held by every later one: the sorted game ids that
      kicked off before the cutoff with no final score; None only when the attempt stopped
      before eligibility could be read);
    * ``install_intent`` adds ``ids`` (wp, ats, ou, blend, converter), ``bet_rule_path`` (exactly
      ``bet_rule_relpath(bet_rule_sha256)``), ``bet_rule_sha256`` and ``code_commit`` (the value
      its attempt records: 40 lowercase hex, optionally ``+dirty``, or None);
    * ``closed`` adds ``status`` (one of :data:`CLOSED_STATUSES`), ``closed_at_utc``,
      ``before_ids``, ``after_ids``.

    Every event is appended durably (flush + fsync): an ``install_intent`` must precede its swap
    on disk, and once a re-fit generation is in force every decision run reads this file, so a
    torn ``attempt`` or ``closed`` line after a power cut would stop the re-fit step (WR-C05).

    Returns:
        The sink's answer: False means it declined and nothing was written (Plan 37-11 refuses the
        install on False).

    Raises:
        ValueError: the event names a problem (out-of-vocabulary value, missing key, another
            bet-rule path form, a malformed code commit); nothing is written.
    """
    problem = _date_event_problem(event)
    if problem is not None:
        raise ValueError(f"refusing a cadence-date event: {problem}")
    return append_record(
        ledger_dir,
        DATES_FILENAME,
        event,
        kind=_RECORD_KIND,
        durable=True,
    )


def read_date_events(
    ledger_dir: Path | str, *, skip_torn_tail: bool = False
) -> list[dict[str, Any]]:
    """Every cadence-date event, in file order; ``[]`` when the file is absent.

    Args:
        skip_torn_tail: Leave out (and log) an unterminated, unparseable last line. Only the live
            decision path passes True (WR-A02, D-15): a torn append there is never a served
            install, because an ``install_intent`` is fsynced before ``latest.json`` is swapped.
            The re-fit hook reads strictly, so the torn line still fails that day's re-fit step.

    Raises:
        RefitRecordFormatError: a line is not JSON, or names a problem the writer would refuse.
    """
    return _read_validated(
        refit_dir(ledger_dir) / DATES_FILENAME,
        _date_event_problem,
        skip_torn_tail=skip_torn_tail,
    )


# ---------------------------------------------------------------------------
# A/A weeks
# ---------------------------------------------------------------------------


def _aa_record_problem(record: Mapping[str, Any]) -> str | None:
    """The first problem with an A/A week record, or None when it is valid."""
    problem = _missing_key_problem(record, _AA_RECORD_KEYS)
    if problem is not None:
        return problem
    for key in ("season", "week"):
        value = record[key]
        if not isinstance(value, int) or isinstance(value, bool):
            return f"field {key!r} value {value!r} is not an integer"
    if not isinstance(record["clean"], bool):
        return f"field 'clean' value {record['clean']!r} is not a bool"
    return None


def append_aa_record(ledger_dir: Path | str, record: Mapping[str, Any]) -> bool:
    """Append one A/A week result (``season``, ``week``, ``clean`` and its detail).

    Raises:
        ValueError: a required field is missing or of the wrong type; nothing is written.
    """
    problem = _aa_record_problem(record)
    if problem is not None:
        raise ValueError(f"refusing an A/A week record: {problem}")
    return append_record(ledger_dir, AA_WEEKS_FILENAME, record, kind=_RECORD_KIND)


def read_aa_records(ledger_dir: Path | str) -> list[dict[str, Any]]:
    """Every A/A week record, in file order; ``[]`` when the file is absent.

    Raises:
        RefitRecordFormatError: a line is not JSON, or names a problem the writer would refuse.
    """
    return _read_validated(
        refit_dir(ledger_dir) / AA_WEEKS_FILENAME, _aa_record_problem
    )


# ---------------------------------------------------------------------------
# Bet-rule records
# ---------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bet_rule_record(
    ledger_dir: Path | str, payload: Mapping[str, Any]
) -> tuple[Path, str]:
    """Write *payload* once as ``<ledger>/refit/bet_rules/<sha256>.json``; return (path, sha256).

    The bytes are canonical JSON (sorted keys, ASCII, no spaces, no NaN), so one payload has one
    name. The write goes through the sink, into a temp file flushed and fsynced before its replace.

    Raises:
        RefitRecordFormatError: an existing file under that name no longer hashes to it.
    """
    data = json.dumps(
        payload,
        sort_keys=True,
        ensure_ascii=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("ascii")
    sha = hashlib.sha256(data).hexdigest()
    destination = bet_rule_record_path(ledger_dir, sha)
    if destination.exists():
        if _file_sha256(destination) != sha:
            msg = f"the bet-rule record {destination.as_posix()} no longer hashes to its name {sha}"
            raise RefitRecordFormatError(msg)
        return destination, sha
    if not current_sink().authorize(str(destination), _BET_RULE_KIND):
        return destination, sha

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return destination, sha


def read_bet_rule_record(path: Path | str) -> dict[str, Any]:
    """The bet rule at *path*, after checking its bytes hash to the sha256 in its name.

    Raises:
        RefitRecordFormatError: the bytes do not hash to the name, or are not one JSON object.
    """
    path = Path(path)
    sha = path.stem
    if _file_sha256(path) != sha:
        msg = (
            f"the bet-rule record {path.as_posix()} no longer hashes to its name {sha}"
        )
        raise RefitRecordFormatError(msg)
    record = json.loads(path.read_text(encoding="ascii"))
    if not isinstance(record, dict):
        msg = f"the bet-rule record {path.as_posix()} is not one JSON object"
        raise RefitRecordFormatError(msg)
    return record


# ---------------------------------------------------------------------------
# The attempt lock and the code commit
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def attempt_lock(ledger_dir: Path | str) -> Iterator[Path]:
    """Hold ``<ledger>/refit/attempt.lock`` for one re-fit attempt; a second holder is refused.

    Raises:
        FileLockHeldError: another invocation holds the lock (no wait).
    """
    directory = refit_dir(ledger_dir)
    directory.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(
        directory / ATTEMPT_LOCK_FILENAME, wait_seconds=0.0
    ) as path:
        yield path


def code_commit(repo_root: Path) -> str | None:
    """HEAD's sha, plus ``+dirty`` when a :data:`REFIT_CODE_PATHS` path differs; None if unknown.

    The shape of ``pipeline.site_push._code_commit``: an edit or an untracked file under a listed
    path means the code that ran is in no commit. Gitignored files are not changes.
    """
    git = ("git", "-C", str(repo_root))
    try:
        head = subprocess.run(
            [*git, "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
        status = subprocess.run(
            [*git, "status", "--porcelain", "--", *REFIT_CODE_PATHS],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    sha = head.stdout.strip()
    if head.returncode != 0 or status.returncode != 0 or not sha:
        return None
    return f"{sha}{_DIRTY_SUFFIX}" if status.stdout.strip() else sha
