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
records the intended write and persists nothing, and the test-write guard sees every write. An
``install_intent`` and a bet-rule file are install-critical: they are flushed and fsynced before
the install's ``latest.json`` swap can follow them, matching that manifest's own fsync.

ONE RULE, ENFORCED BOTH WAYS
----------------------------
Each record file has one private validator. The writer refuses a record it names a problem in
(``ValueError``, nothing written); the reader refuses a stored line it names a problem in
(:class:`RefitRecordFormatError` naming the file, the 1-based line and the field). A well-formed
JSON line with a missing or out-of-vocabulary field is never read as a valid record.

A MALFORMED RECORD IS NEVER READ AS "NO RECORDS"
------------------------------------------------
:func:`read_records` returns ``[]`` only for an absent file. A bad line raises
:class:`RefitRecordFormatError`; when it is the file's last non-blank line the message gives the
torn-append repair, and for any other line it says to compare with the private backup instead.

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

from data.write_sink import append_jsonl, current_sink
from utils.file_lock import exclusive_file_lock

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
    "a bad line that is not the last line is damage, not a torn append: compare the file with "
    "the private backup before changing anything"
)


class RefitRecordFormatError(Exception):
    """A Phase 37 record file holds a bad line or a bet-rule file that no longer hashes to its name.

    Inherits bare ``Exception`` (the ``data.graded_weeks`` rule): an unreadable record must never
    be degraded into "no records".
    """


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
    """
    stamped = {**record, "recorded_at_utc": datetime.now(tz=UTC).isoformat()}
    return append_jsonl(
        refit_dir(ledger_dir) / filename, stamped, kind=kind, durable=durable
    )


def read_records(ledger_dir: Path | str, filename: str) -> list[dict[str, Any]]:
    """Every record in ``<ledger>/refit/<filename>``, oldest first; ``[]`` when it is absent.

    Raises:
        RefitRecordFormatError: a non-blank line is not one JSON object.
    """
    return _read_validated(refit_dir(ledger_dir) / filename, None)


def _read_validated(
    path: Path, problem_of: Callable[[Mapping[str, Any]], str | None] | None
) -> list[dict[str, Any]]:
    """Every JSON-object line of *path*, each passed through *problem_of* when it is given."""
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8").splitlines()
    numbered = [(number, line) for number, line in enumerate(lines, 1) if line.strip()]
    last_number = numbered[-1][0] if numbered else 0
    records: list[dict[str, Any]] = []
    for number, line in numbered:
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            detail = f"is not JSON: {error}"
            raise RefitRecordFormatError(
                _line_error(path, number, last_number, detail)
            ) from error
        if not isinstance(record, dict):
            raise RefitRecordFormatError(
                _line_error(path, number, last_number, "is not one JSON object")
            )
        problem = problem_of(record) if problem_of is not None else None
        if problem is not None:
            raise RefitRecordFormatError(
                _line_error(path, number, last_number, problem)
            )
        records.append(record)
    return records


def _line_error(path: Path, number: int, last_number: int, detail: str) -> str:
    """The message naming the file, the line, the problem and the safe repair."""
    advice = _TORN_LAST_LINE_REPAIR if number == last_number else _DAMAGED_LINE_ADVICE
    return f"refit record {path.as_posix()} line {number} {detail}; {advice}"


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
    )
    if problem is not None:
        return problem
    if kind == "attempt":
        return _vocabulary_problem("outcome", event["outcome"], ATTEMPT_OUTCOMES)
    if kind == "closed":
        return _vocabulary_problem("status", event["status"], CLOSED_STATUSES)
    return _install_intent_problem(event)


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
      (1-based), ``date`` (ISO), ``nfl_week``;
    * ``attempt`` adds ``cutoff_utc``, ``attempted_at_utc``, ``run_date_et``, ``code_commit``,
      ``outcome`` (one of :data:`ATTEMPT_OUTCOMES`), ``check``, ``exception_class``,
      ``before_ids``, ``after_ids``, and may carry ``no_final_score_ids`` (the sorted game ids the
      attempt's training excluded as no_final_score, or None when it stopped before training);
    * ``install_intent`` adds ``ids`` (wp, ats, ou, blend, converter), ``bet_rule_path`` (exactly
      ``bet_rule_relpath(bet_rule_sha256)``), ``bet_rule_sha256`` and ``code_commit`` (the value
      its attempt records: 40 lowercase hex, optionally ``+dirty``, or None);
    * ``closed`` adds ``status`` (one of :data:`CLOSED_STATUSES`), ``closed_at_utc``,
      ``before_ids``, ``after_ids``.

    An ``install_intent`` is appended durably (flush + fsync).

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
        durable=event["event"] == "install_intent",
    )


def read_date_events(ledger_dir: Path | str) -> list[dict[str, Any]]:
    """Every cadence-date event, in file order; ``[]`` when the file is absent.

    Raises:
        RefitRecordFormatError: a line is not JSON, or names a problem the writer would refuse.
    """
    return _read_validated(refit_dir(ledger_dir) / DATES_FILENAME, _date_event_problem)


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
