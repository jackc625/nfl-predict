"""The committed, append-only record of every game a live run skipped (D33.2-05, Plan 33.2-03).

WHY THIS FILE EXISTS
--------------------
D33.2-05's live half says a game caught using post-lock information is LEFT OUT: no prediction,
no bet, the reason recorded, and that day's clean games go ahead. "The reason recorded" is the
part a run log cannot honour. ``logs/friday_pipeline.json`` is rewritten by every run, so a skip
written only there survives until the next night. This log is the durable half: ONE LINE PER
SKIPPED (game, source, reason), appended once and never edited.

It is the same store class as ``config/upstream_probe_log.jsonl`` (``data/sealed_probe_log.py``),
and deliberately so -- that module worked out every property a durable record needs, and this one
copies them rather than re-deriving them.

WHO READS IT
------------
Phase 35 owns notifications and will read this file through :func:`skip_records_for_run_date`.
This phase builds NO alert channel (SPEC boundary): it writes the record Phase 35 reads. The run
log's ``skipped_games`` field is the cross-reference that tells an operator reading one file to
look in the other.

APPEND PROTOCOL
---------------
Each line is appended once by the run that produced it and never edited afterwards. Nothing in
this module rewrites, reorders or deletes a line. A re-run of the same live day that would append
a line whose natural key already exists appends NOTHING -- see :data:`IDEMPOTENCY_KEY`.

THE SINGLE-WRITER ASSUMPTION, STATED RATHER THAN SILENT
--------------------------------------------------------
Idempotency is a scan-then-append, which is correct for ONE writer at a time and not for two
overlapping runs: both could miss the key and both append. The scheduler's overlap policy is set
by Plan 33.2-28, which owns the scheduled-task definition (D33.2-18 moves the trigger to a
pre-lock daily fire); pinning a lock contract here would decide that plan's question before it
exists. Deferred by owner scope decision 2026-09-20, recorded in ``33.2-03-PLAN.md``'s review
ledger. Until then the assumption is this sentence, not an accident.

AN UNREADABLE LOG BLOCKS
------------------------
``read_skip_records`` raises rather than returning ``[]`` when a line will not decode, and the
append refuses on the same condition, because the idempotency scan reads the file first. Treating
an unparseable line as absent would make a truncated write look exactly like a clean slate -- and
would let the next append duplicate the very record the truncation damaged.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

__all__ = [
    "IDEMPOTENCY_KEY",
    "NULLABLE_ENTRY_KEYS",
    "REQUIRED_ENTRY_KEYS",
    "SKIP_REASONS",
    "SKIP_RECORD_PATH",
    "SkipRecordCorrupt",
    "append_skip_record",
    "read_skip_records",
    "skip_records_for_run_date",
]

# The committed skip record.
#
# JSONL, NOT ONE JSON ARRAY, for the reason ``data/sealed_probe_log.py`` states: appending a line
# is an O(1) operation that physically CANNOT rewrite a prior entry -- the handle is opened in
# append mode and the earlier bytes are never addressed. Re-rendering an array would put every
# prior entry back through a writer on every run, so "prior entries are byte-unchanged" would
# depend on the writer being correct rather than on the file mode making the alternative
# impossible. A record whose integrity rests on a serialiser round-tripping identically for a
# whole season is not a record.
#
# UNDER config/, NOT logs/ OR outputs/. ``logs/`` is regenerable by definition, and ``outputs/``
# is gitignored derived data (LDGR-01's reasoning). A skipped game is a FACT about a live night,
# and it has to survive a fresh clone.
#
# NOT CREATED UNTIL THE FIRST REAL SKIP. The file comes into existence when a live run first
# drops a game, so its first committed lines are real verdicts rather than a seeded placeholder a
# later reader would have to learn to discount. Nothing here touches the path at import time.
SKIP_RECORD_PATH: Path = Path("config/skip_records.jsonl")

# The eight fields without which a line cannot answer the questions this record exists for:
# WHICH run, on WHICH day, dropped WHICH game, because of WHICH source, whose information was
# known WHEN relative to WHICH lock, for WHAT reason -- and when was that written down.
REQUIRED_ENTRY_KEYS: tuple[str, ...] = (
    "run_id",
    "run_date_et",
    "game_id",
    "source",
    "information_time",
    "lock",
    "reason",
    "recorded_at",
)

# The two keys that must be PRESENT but may be null, and the reason each null is honest:
#
# * ``information_time`` -- an ``undated`` refusal is precisely a value whose information time
#   cannot be established. Writing any timestamp there would manufacture the fact whose absence
#   is the finding.
# * ``lock`` -- a coverage refusal (``no_provenance``) names the game without carrying its lock
#   in the payload. Recording null says "the refusal did not state it"; deriving one here would
#   put a second lock derivation in a module that is not the lock rule.
#
# Every OTHER key identifies the record, and a null identity cannot be deduplicated or read.
NULLABLE_ENTRY_KEYS: frozenset[str] = frozenset({"information_time", "lock"})

# THE CLOSED VOCABULARY OF WHY A GAME WAS DROPPED.
#
# THIS IS NOT AN EXEMPTION MECHANISM (T-33.2-03-04). Every member means the game was DROPPED --
# no prediction, no bet. There is no member meaning "checked and waived", and an unknown reason is
# refused rather than recorded, so the vocabulary cannot grow into a list of accepted exceptions
# by accretion.
#
# * ``post_lock``     -- information known AFTER the game's lock (a late value, or a decision
#                         instant past the lock).
# * ``undated``       -- a value whose information time cannot be established.
# * ``no_provenance`` -- a game the source's provenance does not cover one-to-one.
SKIP_REASONS: tuple[str, ...] = ("post_lock", "undated", "no_provenance")

# THE NATURAL KEY, and the one key it must never be (RESEARCH 3.4). ``run_id`` changes on every
# run by construction, so keying on it would turn every re-run of the same live day into a new
# record -- the duplicate SPEC R3 forbids. A game dropped from the same day's slate, by the same
# source, for the same reason, is ONE fact however many times the run is repeated.
IDEMPOTENCY_KEY: tuple[str, ...] = ("run_date_et", "game_id", "source", "reason")


class SkipRecordCorrupt(Exception):
    """A skip record cannot be read, or an entry cannot honestly be written.

    Inherits ``Exception`` and NOT ``RuntimeError`` / ``ValueError`` / ``ImportError``, the same
    reasoning as ``data.sealed_probe_log.SealedProbeLogCorrupt``: several call sites in this
    repository catch that tuple and degrade quietly -- ``scripts/build_features._SOURCE_LOAD_ERRORS``
    turns a ``ValueError`` into an empty source, and the orchestrator's retry filter keys on
    ``OSError`` -- and "the skip record is unwritable" degraded into "carry on" is the precise
    failure this log exists to make impossible. A game the run dropped without a durable trace is
    a game nobody can later account for.
    """


def _resolved(path: Path | str | None) -> Path:
    """The target file, resolved at CALL time so a redirected module path is honoured."""
    return Path(path).resolve() if path is not None else SKIP_RECORD_PATH.resolve()


def _refuse_incomplete(entry: Mapping[str, Any], *, where: str) -> None:
    """Raise naming EVERY missing or illegally-null key, or return having found none."""
    missing = [key for key in REQUIRED_ENTRY_KEYS if key not in entry]
    if missing:
        msg = (
            f"{where}: a skip record missing {', '.join(missing)}. Every line must carry "
            f"{', '.join(REQUIRED_ENTRY_KEYS)}: a record that cannot say which run dropped which "
            "game, why, and against which lock cannot be audited a season later, so it is refused "
            "rather than written."
        )
        raise SkipRecordCorrupt(msg)

    null_identity = [
        key
        for key in REQUIRED_ENTRY_KEYS
        if key not in NULLABLE_ENTRY_KEYS and entry.get(key) in (None, "")
    ]
    if null_identity:
        msg = (
            f"{where}: a skip record whose identity field(s) {', '.join(null_identity)} are "
            "null or empty. Only information_time and lock may be null (an undated value has no "
            "time; a coverage refusal does not state its lock); every other field identifies the "
            "record, and a null identity can be neither deduplicated nor read."
        )
        raise SkipRecordCorrupt(msg)


def _refuse_unknown_reason(reason: object, *, where: str) -> None:
    if reason not in SKIP_REASONS:
        msg = (
            f"{where}: skip reason {reason!r} is outside the closed vocabulary "
            f"{SKIP_REASONS}. Every reason means the game was DROPPED; there is no reason that "
            "means 'checked and waived', and an unknown one is refused rather than recorded."
        )
        raise SkipRecordCorrupt(msg)


def _natural_key(entry: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(str(entry[key]) for key in IDEMPOTENCY_KEY)


def read_skip_records(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Return every appended skip record, in append order.

    An ABSENT file returns ``[]``: a live history with no skip yet legitimately has nothing to
    read. A file that EXISTS and has an unreadable line RAISES, naming the line number.

    Raises:
        SkipRecordCorrupt: when the file cannot be read, when a line is blank, will not decode,
            decodes to something other than a mapping, or lacks a required key.
    """
    resolved = _resolved(path)
    if not resolved.is_file():
        return []

    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        msg = (
            f"the skip record at {resolved} exists but could not be read ({exc}). An unreadable "
            "record BLOCKS: it is indistinguishable from a crashed append whose content never "
            "landed."
        )
        raise SkipRecordCorrupt(msg) from exc

    records: list[dict[str, Any]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        where = f"the skip record at {resolved}, line {number}"
        if not line.strip():
            msg = (
                f"{where} is BLANK. append_skip_record never writes one, so this is an anomaly "
                "rather than whitespace -- most likely a partial write. Resolve it by hand."
            )
            raise SkipRecordCorrupt(msg)
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = (
                f"{where} could not be decoded ({exc}). An unreadable record BLOCKS rather "
                "than reading as absent: a truncated write must not look like a clean slate."
            )
            raise SkipRecordCorrupt(msg) from exc
        if not isinstance(decoded, dict):
            msg = f"{where} is a {type(decoded).__name__}, not a mapping."
            raise SkipRecordCorrupt(msg)
        _refuse_incomplete(decoded, where=where)
        records.append(decoded)
    return records


def append_skip_record(
    entry: Mapping[str, Any], *, path: Path | str | None = None
) -> bool:
    """Append exactly ONE line recording *entry*, unless its natural key is already recorded.

    The resolve-and-``mkdir`` discipline is ``pipeline/execution_log.py``'s
    ``write_execution_log_atomic``; the append mode is ``data/sealed_probe_log.py``'s.
    ``sort_keys=True`` makes a line's bytes a function of its CONTENT alone, so a diff means a real
    difference rather than a dictionary-ordering accident. ``newline="\\n"`` is explicit because
    this file is committed and must not acquire CRLF on Windows -- a line-ending flip would rewrite
    every line in the diff and destroy the append-only reading. ``default=str`` keeps a stray
    timestamp object from failing the write outright.

    Args:
        entry: The record. Must carry every key in :data:`REQUIRED_ENTRY_KEYS`, with a reason from
            :data:`SKIP_REASONS`.
        path: An override, for tests. ``None`` uses :data:`SKIP_RECORD_PATH`, resolved now.

    Returns:
        ``True`` when a line was appended; ``False`` when the natural key
        ``(run_date_et, game_id, source, reason)`` was already recorded, in which case the file is
        byte-identical afterwards.

    Raises:
        SkipRecordCorrupt: when *entry* is not a mapping, lacks a required key, carries a null
            identity field or an unknown reason, or when the existing file is unreadable. Nothing
            is written in any of those cases.
    """
    if not isinstance(entry, Mapping):
        msg = f"a skip record must be a mapping, got {type(entry).__name__}. Nothing was appended."
        raise SkipRecordCorrupt(msg)

    where = "refusing to append"
    _refuse_incomplete(entry, where=where)
    _refuse_unknown_reason(entry["reason"], where=where)

    # The scan reads through the strict reader, so a corrupt file blocks the append as well as the
    # read: an unreadable line must not count as "this key is absent".
    key = _natural_key(entry)
    if any(_natural_key(existing) == key for existing in read_skip_records(path)):
        return False

    resolved = _resolved(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(dict(entry), sort_keys=True, default=str)
    with resolved.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")
    return True


def skip_records_for_run_date(
    run_date_et: str, *, path: Path | str | None = None
) -> list[dict[str, Any]]:
    """Every record for one Eastern run date, in append order -- the accessor Phase 35 reads.

    Raises:
        SkipRecordCorrupt: via :func:`read_skip_records`.
    """
    return [
        record
        for record in read_skip_records(path)
        if str(record["run_date_et"]) == run_date_et
    ]
