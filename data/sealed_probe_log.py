"""The committed, append-only sealed-probe log (D32-08, Plan 32-05).

WHY THIS FILE EXISTS
--------------------
``data/sealed_probe.py`` can tell you what it saw on the run you are watching. It cannot
tell you whether it ran last week. A detector that quietly stopped running produces no
alarm, no error and no output -- which is indistinguishable from a detector that ran every
week and found nothing. This log closes that window by the cheapest possible means: ONE
LINE PER RUN, including ``clean`` and ``unknown`` alike. A season of appended lines is
itself the proof the detector was alive, and a GAP in them is the evidence that it was not.

APPEND PROTOCOL
---------------
Each line is APPENDED ONCE by the run that produced it and is NOT edited afterwards. A
later run appends a NEW line rather than amending an old one, and a correction to an
earlier reading is itself a new line. Nothing in this module rewrites, reorders or deletes
a line, and ``tests/unit/test_sealed_probe_log.py`` asserts the first line's raw bytes are
unchanged after a second append -- because append mode is the only thing making that true,
and nothing else in this repository proved it before.

COMMITTED RECORD, GITIGNORED BYTES
-----------------------------------
The same asymmetry ``data/upstream_pin.py`` states for its manifest, one turn further out.
``config/`` is in git and ``data/`` is not, so this log is COMMITTED while the parquet
snapshots its verdicts are about are absent from a fresh checkout. That is exactly what
lets a reader on a clean clone say WHICH upstream revision a verdict was measured against,
and when, without having any of the bytes.

AN UNREADABLE LOG BLOCKS
------------------------
``read_probe_log`` raises rather than returning ``[]`` when a line will not decode, in the
contract ``backtest/profitability_2025.py::read_run_ledger`` already set for this project:
treating an unparseable record as absent would make a truncated write -- a crash between
the append and the flush -- look exactly like a clean slate, which is the very window this
log closes.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from data.write_sink import current_sink

__all__ = [
    "REQUIRED_ENTRY_KEYS",
    "SEALED_PROBE_LOG_PATH",
    "SealedProbeLogCorrupt",
    "append_probe_entry",
    "latest_entry",
    "probe_log_staleness",
    "read_probe_log",
]

# The committed liveness record.
#
# JSONL, NOT ONE JSON ARRAY, and the choice is mechanical rather than stylistic. Appending
# a line is an O(1) operation that physically CANNOT rewrite a prior entry -- the file
# handle is opened in append mode and the earlier bytes are never addressed. Re-rendering a
# JSON array would put every prior entry back through a writer every single week, so the
# "prior entries are byte-unchanged" property would depend on the writer being correct
# rather than on the file mode making the alternative impossible. A record whose integrity
# rests on a serialiser round-tripping identically for a whole season is not a record.
#
# It is NOT created by Plan 32-05. The file comes into existence when Plan 32-08 runs the
# first probe, so its first committed lines are real verdicts rather than a seeded
# placeholder that a later reader would have to learn to discount.
SEALED_PROBE_LOG_PATH: Path = Path("config/upstream_probe_log.jsonl")

# The five fields without which a line cannot answer the only question this log exists to
# answer: did the detector run, and did it see everything? ``checked`` and ``expected`` are
# in that set deliberately -- a verdict recorded without its coverage is a claim nobody can
# audit, and an unauditable line is not worth appending.
REQUIRED_ENTRY_KEYS: tuple[str, ...] = (
    "event_class",
    "severity",
    "probed_at_utc",
    "checked",
    "expected",
)


class SealedProbeLogCorrupt(Exception):
    """A probe log line cannot be read, or an entry cannot honestly be written.

    Inherits ``Exception`` and NOT ``RuntimeError`` / ``ValueError`` / ``ImportError``, the
    same reasoning as ``data.upstream_pin.UpstreamPinError`` and
    ``data.sealed_probe.SealedProbeUnavailable``: several call sites in this repository
    catch that tuple and degrade quietly, and "the liveness record is unreadable" degraded
    into "carry on" is the precise failure this log was built to make impossible.
    """


def _resolved(path: Path | str | None) -> Path:
    return Path(path).resolve() if path is not None else SEALED_PROBE_LOG_PATH.resolve()


def append_probe_entry(
    entry: dict[str, Any], *, path: Path | str | None = None
) -> None:
    """Append exactly ONE line recording *entry*. Never rewrites a prior line.

    The resolve-and-``mkdir(parents=True, exist_ok=True)`` discipline is
    ``pipeline/execution_log.py::write_execution_log_atomic``'s; the append mode is this
    module's own, and is the only genuinely new writer in Phase 32.

    ``sort_keys=True`` makes a line's bytes a function of its CONTENT alone, so two runs
    recording the same verdict produce the same line and a diff therefore means a real
    difference rather than a dictionary-ordering accident. ``newline="\\n"`` is explicit
    because this file is committed and must not acquire CRLF on Windows -- a line-ending
    flip would rewrite every line in the diff and destroy the append-only reading.
    ``default=str`` keeps a stray enum or datetime from failing the write outright; the
    vocabulary in ``data/revision_events.py`` is a ``StrEnum`` precisely so this fallback
    is never actually needed for the fields that matter.

    Args:
        entry: The verdict mapping. Must carry every key in :data:`REQUIRED_ENTRY_KEYS`.
        path: An override, for tests. ``None`` uses :data:`SEALED_PROBE_LOG_PATH`.

    Raises:
        SealedProbeLogCorrupt: If *entry* is not a mapping or lacks a required key.
    """
    if not isinstance(entry, dict):
        msg = (
            f"a probe log entry must be a mapping, got {type(entry).__name__}. Nothing "
            "was appended."
        )
        raise SealedProbeLogCorrupt(msg)

    missing = [key for key in REQUIRED_ENTRY_KEYS if key not in entry]
    if missing:
        msg = (
            f"refusing to append a probe log entry missing {', '.join(missing)}. Every "
            f"line must carry {', '.join(REQUIRED_ENTRY_KEYS)}: a line that cannot answer "
            "'did the detector see everything?' is not worth appending, because a reader "
            "a season later cannot tell it from a line that saw nothing."
        )
        raise SealedProbeLogCorrupt(msg)

    resolved = _resolved(path)
    # The write sink (Plan 33.2-27 Task 2b): this file is COMMITTED, so a dry run that appended
    # to it would forge the very record it exists to keep.
    if not current_sink().authorize(resolved.as_posix(), "append_probe_entry"):
        return
    resolved.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, sort_keys=True, default=str)
    with resolved.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(line + "\n")


def read_probe_log(path: Path | str | None = None) -> list[dict]:
    """Return every appended entry, in append order.

    An ABSENT file returns ``[]``: a first-ever run legitimately has nothing to read, and a
    module that raised on that would have to be wrapped in a ``try`` at every call site,
    which is how the raise stops being read at all.

    A file that EXISTS and has a line that will not decode RAISES, naming the line number.
    That asymmetry is the whole point -- see the module docstring.

    Raises:
        SealedProbeLogCorrupt: If the file cannot be read, if any line fails to decode, if
            a line decodes to something other than a mapping, or if a line is blank. A
            blank line is refused rather than skipped: nothing in this module ever writes
            one, so its presence is an anomaly to surface, not whitespace to tidy away.
    """
    resolved = _resolved(path)
    if not resolved.is_file():
        return []

    try:
        text = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        msg = (
            f"the sealed probe log at {resolved} exists but could not be read ({exc}). An "
            "unreadable log BLOCKS: it is indistinguishable from a crashed append whose "
            "content never landed, and this log's entire value is that a gap in it means "
            "something."
        )
        raise SealedProbeLogCorrupt(msg) from exc

    entries: list[dict] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            msg = (
                f"the sealed probe log at {resolved} has a BLANK line at line {number}. "
                "append_probe_entry never writes one, so this is an anomaly rather than "
                "whitespace -- most likely a partial write. Resolve it by hand."
            )
            raise SealedProbeLogCorrupt(msg)
        try:
            decoded = json.loads(line)
        except json.JSONDecodeError as exc:
            msg = (
                f"the sealed probe log at {resolved} could not be decoded at line "
                f"{number} ({exc}). An unreadable log BLOCKS rather than reading as "
                "absent: a truncated write must not look exactly like a clean slate."
            )
            raise SealedProbeLogCorrupt(msg) from exc
        if not isinstance(decoded, dict):
            msg = (
                f"the sealed probe log at {resolved} has a non-mapping entry at line "
                f"{number} (a {type(decoded).__name__})."
            )
            raise SealedProbeLogCorrupt(msg)
        entries.append(decoded)
    return entries


def latest_entry(path: Path | str | None = None) -> dict | None:
    """Return the most recently APPENDED entry, or ``None`` when the log is empty.

    "Most recently appended" is positional, not chronological, and deliberately so: the
    order in the file is the order the runs happened, and re-sorting by a recorded
    timestamp would let a wrong clock reorder history.
    """
    entries = read_probe_log(path)
    return entries[-1] if entries else None


def _parse_instant(value: Any, *, resolved: Path) -> datetime:
    if not isinstance(value, str) or not value.strip():
        msg = (
            f"the newest entry in the sealed probe log at {resolved} carries "
            f"probed_at_utc={value!r}, which is not a non-empty string."
        )
        raise SealedProbeLogCorrupt(msg)
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        msg = (
            f"the newest entry in the sealed probe log at {resolved} carries "
            f"probed_at_utc={value!r}, which does not parse as ISO-8601 ({exc})."
        )
        raise SealedProbeLogCorrupt(msg) from exc
    if parsed.tzinfo is None:
        msg = (
            f"the newest entry in the sealed probe log at {resolved} carries a NAIVE "
            f"probed_at_utc={value!r}. An age computed against a naive stamp is a "
            "different number on every machine that reads it."
        )
        raise SealedProbeLogCorrupt(msg)
    return parsed


def probe_log_staleness(
    *,
    path: Path | str | None = None,
    max_age_days: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Report how long it has been since the detector last wrote a line.

    This is the second, cheaper liveness check ``32-RESEARCH.md`` asks for, and it costs
    one file read. The expensive check -- the coverage assertion in
    ``data.sealed_probe.probe_sealed`` -- proves a RUN saw everything. This one proves a
    run happened at all.

    AN EMPTY OR ABSENT LOG IS STALE, with ``latest_probed_at_utc: None``. A detector that
    never ran and a detector that ran and found nothing are DIFFERENT FACTS, and only one
    of them leaves a line. Reporting the never-ran case as healthy would rebuild, at the
    log layer, the exact ambiguity the coverage assertion removes at the verdict layer.

    Args:
        path: An override, for tests.
        max_age_days: The cadence the log is expected to be written at. An age strictly
            greater than this is stale.
        now: An injected timezone-aware instant. ``None`` uses the current UTC instant.

    Returns:
        ``{"latest_probed_at_utc", "age_days", "stale", "entries"}``.

    Raises:
        SealedProbeLogCorrupt: Via :func:`read_probe_log`, or when the newest entry's
            ``probed_at_utc`` is missing, unparseable or naive.
        ValueError: When *now* is naive.
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        msg = (
            "probe_log_staleness's `now` must be timezone-aware; a naive instant makes "
            "the computed age depend on the machine that ran it."
        )
        raise ValueError(msg)

    resolved = _resolved(path)
    entries = read_probe_log(path)
    if not entries:
        return {
            "latest_probed_at_utc": None,
            "age_days": None,
            "stale": True,
            "entries": 0,
        }

    newest = entries[-1]
    stamp = newest.get("probed_at_utc")
    parsed = _parse_instant(stamp, resolved=resolved)
    age_days = (now - parsed).total_seconds() / 86400.0
    return {
        "latest_probed_at_utc": stamp,
        "age_days": age_days,
        "stale": age_days > max_age_days,
        "entries": len(entries),
    }
