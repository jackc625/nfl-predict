"""The weekly email's data-change section (D-03, D-21), read from stored output only (NOTIF-05).

WHAT THE SECTION LISTS
----------------------
* SEALED ZONE (2001-2025): the findings of the LATEST line of ``config/upstream_probe_log.jsonl``
  that are not acknowledged, when that line's severity ranks at WARNING or louder. An older line's
  finding that the latest line no longer carries is not listed (D-21).
* LIVE ZONE (the running season): the live manifest's verdicts at WARNING or louder RECORDED SINCE
  THE PREVIOUS DELIVERED DIGEST, so each change appears once, the week it is recorded; the first
  digest lists the season so far. A verdict's time is its CAPTURE's ``captured_at_utc`` -- the
  verdict block under ``data.upstream_live.CAPTURE_VERDICT_KEY`` carries none of its own -- the
  traversal ``forward_ledger.corrections.owed_correction_events`` uses.
* CORRECTED GRADES: every live-arm graded bet whose result was corrected since the previous
  delivered digest, from the web cache's ``bet_list_corrections``.

INFORMATIONAL findings are never listed. With nothing to list the section says so in one line, and
a source that could not be read is named as unreadable -- never shown as "nothing".

"FIXED BY LEDGER CORRECTION" IS CLAIMED ONLY WHEN PROVEN
---------------------------------------------------------
Phase 34's correction pass stamps every correction's ``detected_at_utc`` with the
``captured_at_utc`` of the owed capture that produced it (``forward_ledger.corrections``). A live
verdict is therefore marked fixed only when a stored correction has the scope's season, a week in
the scope's weeks AND that same instant (compared as instants, never as text). An unrelated
correction in the same week never marks a verdict fixed.

SEVERITY IS RANKED, NEVER COMPARED AS TEXT
------------------------------------------
``RevisionSeverity`` is a ``StrEnum``, and alphabetically ``"critical" < "informational" <
"warning"``. Every comparison here goes through ``data.live_revision.verdict_severity_rank`` and
``data.revision_events.severity_rank``.

THE CLOSING SECTION (SPEC R4 item 7, D-23)
------------------------------------------
Before the data-change section, the email lists, from ``logs/ledger_runs.jsonl``
(``forward_ledger.run_log.read_events``) since the previous delivered digest: every skipped or
failed closing reading, every game whose closing value was finalized as missing, every first-pick
refusal, and every closing-trigger registration whose read-back did not match. A run log that
cannot be read is named, with its line number -- never shown as "nothing missed".

Seams read at call time, so no test reads the real config files: :data:`PROBE_LOG_PATH` and
:data:`MANIFEST_DIR` (None = ``data.upstream_live.LIVE_MANIFEST_DIR``); the run log is
``forward_ledger.run_log.LEDGER_RUN_LOG``, which the test suite quarantines.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from api.cache import ARM_LIVE
from data.live_revision import verdict_severity_rank
from data.revision_events import (
    CORRECTION_OWED,
    CORRECTION_OWED_SCOPE,
    RevisionEventClass,
    RevisionSeverity,
    severity_rank,
)
from data.sealed_probe import PROBE_STRATEGY_CONTENT, PROBE_STRATEGY_METADATA
from data.sealed_probe_log import SEALED_PROBE_LOG_PATH
from data.upstream_live import CAPTURE_VERDICT_KEY
from forward_ledger import run_log
from notify.messages import game_labels
from notify.transport import OUTCOME_DELIVERED

__all__ = [
    "ACKNOWLEDGE_COMMAND",
    "CLOSING_EMPTY_LINE",
    "DATA_CHANGE_EMPTY_LINE",
    "MANIFEST_DIR",
    "PROBE_LOG_PATH",
    "UNAFFECTED_LINE",
    "UNREADABLE_CORRECTIONS",
    "UNREADABLE_MANIFEST",
    "UNREADABLE_PROBE_LOG",
    "closing_section",
    "closing_section_lines",
    "corrected_grades",
    "data_change_section",
    "live_changes",
    "previous_digest_cutoff",
    "sealed_changes",
]

#: Seam, read at call time: the sealed probe log whose LATEST line the section reads.
PROBE_LOG_PATH: Path = SEALED_PROBE_LOG_PATH

#: Seam, read at call time: the live manifests' directory (None = the default config directory).
MANIFEST_DIR: Path | None = None

#: The D-04 owner command (Plan 35-09) that marks one sealed change as seen.
ACKNOWLEDGE_COMMAND = (
    "uv run python -m scripts.acknowledge_upstream_change "
    '--dataset <dataset> --season <season> --reason "<why>"'
)

DATA_CHANGE_EMPTY_LINE = "No unacknowledged upstream data change and no corrected grade since the last digest."
UNAFFECTED_LINE = "Runs are unaffected: every run reads pinned bytes."
ACKNOWLEDGE_LINE = f"Mark a sealed change as seen with: {ACKNOWLEDGE_COMMAND}"

# A source that could not be read is named, never shown as "nothing". No character HTML escapes.
UNREADABLE_PROBE_LOG = (
    "The upstream probe log could not be read, so sealed-season changes are not listed."
)
UNREADABLE_MANIFEST = (
    "The live-season manifest could not be read, so live-season changes are not listed."
)
UNREADABLE_CORRECTIONS = (
    "The corrected grades in the web cache could not be read, so they are not listed."
)

CLOSING_EMPTY_LINE = "No skipped or missed closing reading and no first-pick refusal since the last digest."
TRIGGER_READBACK_LINE = "the closing-line schedule read back wrong"

# The ledger run log's line number inside read_events' refusal message.
_LINE_NUMBER = re.compile(r"\bline (\d+)\b")

# What moved, by the probe strategy that ruled on the pair (data.sealed_probe).
_WHAT_MOVED: dict[str, str] = {
    PROBE_STRATEGY_CONTENT: "content digest moved",
    PROBE_STRATEGY_METADATA: "updated time or size moved",
}

# /bets' bet-type labels (web/templates/pages/bets.html BET_TYPE_LABELS).
_BET_TYPE_LABELS: dict[str, str] = {"wp": "Winner", "ats": "Spread", "ou": "Totals"}

_LIVE_REVISION_GRADED = str(RevisionEventClass.LIVE_REVISION_GRADED)
_UNKNOWN = str(RevisionEventClass.UNKNOWN)


def _instant(value: object) -> datetime | None:
    """A stored timestamp as an aware instant (naive reads as UTC); None when it does not parse."""
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str) and value.strip():
        try:
            moment = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
        except ValueError:
            return None
    else:
        return None
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def _after(value: object, cutoff: datetime | None) -> bool:
    """True when *value* is after *cutoff*; always True without a cutoff or a readable instant.

    An instant that cannot be read is listed rather than hidden.
    """
    if cutoff is None:
        return True
    moment = _instant(value)
    return moment is None or moment > cutoff


def _loud(verdict: Mapping[str, Any]) -> bool:
    """True when *verdict* ranks at WARNING or louder (ranked, never compared as text)."""
    return verdict_severity_rank(dict(verdict)) >= severity_rank(
        RevisionSeverity.WARNING
    )


def _bet_label(game_id: object, target: object) -> str:
    """``AWAY @ HOME <bet type>``, as /bets names a bet."""
    return f"{game_labels((str(game_id),))} {_BET_TYPE_LABELS.get(str(target), str(target))}"


def _live_arm(arm: object) -> bool:
    """A row with no recorded arm reads as the live arm (the /bets rule, Plan 35-12)."""
    return arm is None or arm == ARM_LIVE


def previous_digest_cutoff(
    notifications: Sequence[Mapping[str, Any]], run_date_et: date
) -> datetime | None:
    """The ``sent_at`` of the latest delivered daily-run digest of an EARLIER run date, or None.

    A same-day ``(update)`` therefore uses the same cutoff as that day's first send. No such
    record: no cutoff, so the first digest lists the season so far.
    """
    sent: list[datetime] = []
    for record in notifications:
        if (
            record.get("event_class") == "digest"
            and record.get("source") == "daily_run"
            and record.get("outcome") == OUTCOME_DELIVERED
            and str(record.get("run_date_et")) < run_date_et.isoformat()
        ):
            moment = _instant(record.get("sent_at"))
            if moment is not None:
                sent.append(moment)
    return max(sent) if sent else None


def sealed_changes(probe_line: Mapping[str, Any] | None) -> list[str]:
    """The latest probe line's unacknowledged findings, when the line is WARNING or louder."""
    if probe_line is None or not _loud(probe_line):
        return []
    lines: list[str] = []
    for finding in probe_line.get("findings") or ():
        if finding.get("acknowledged") is True:
            continue
        moved = _WHAT_MOVED.get(
            str(finding.get("strategy")), "what moved was not recorded"
        )
        lines.append(
            f"{finding.get('dataset')} {finding.get('season')}: the upstream file changed "
            f"after it was pinned ({moved})"
        )
    return lines


def _discharged(
    scope: Mapping[str, Any],
    captured_at: datetime | None,
    corrections: Sequence[Mapping[str, Any]],
) -> bool:
    """True when a stored correction provably belongs to this verdict (season, week, instant)."""
    if captured_at is None:
        return False
    weeks = {int(week) for week in scope.get("weeks") or ()}
    for row in corrections:
        if (
            row.get("season") == scope.get("season")
            and row.get("week") in weeks
            and _instant(row.get("detected_at_utc")) == captured_at
        ):
            return True
    return False


def _live_line(
    dataset: str,
    season: object,
    verdict: Mapping[str, Any],
    captured_at: datetime | None,
    corrections: Sequence[Mapping[str, Any]],
) -> str:
    event_class = str(verdict.get("event_class"))
    if event_class == _LIVE_REVISION_GRADED:
        scope = verdict.get(CORRECTION_OWED_SCOPE) or {}
        weeks = ", ".join(str(week) for week in scope.get("weeks") or ())
        line = (
            f"{dataset} {season} week(s) {weeks}: upstream restated already-graded data"
        )
        if verdict.get(CORRECTION_OWED) is True and _discharged(
            scope, captured_at, corrections
        ):
            line += " -- fixed by ledger correction"
        return line
    if event_class == _UNKNOWN:
        return f"{dataset} {season}: a change that could not be compared (not ruled on)"
    return f"{dataset} {season}: {event_class}"


def live_changes(
    manifest: Mapping[str, Any] | None,
    cutoff: datetime | None,
    corrections: Sequence[Mapping[str, Any]],
) -> list[str]:
    """The live manifest's WARNING-or-louder verdicts recorded after *cutoff* (all when None).

    Walks ``datasets.<dataset>.captures[]`` explicitly: the time is the CAPTURE's
    ``captured_at_utc`` and the verdict is the capture's ``CAPTURE_VERDICT_KEY`` block. Listed in
    capture order (oldest first).
    """
    if manifest is None:
        return []
    found: list[tuple[datetime | None, str, int, str]] = []
    for dataset, block in (manifest.get("datasets") or {}).items():
        for capture in block.get("captures") or ():
            verdict = capture.get(CAPTURE_VERDICT_KEY)
            if not isinstance(verdict, Mapping) or not _loud(verdict):
                continue
            if not _after(capture.get("captured_at_utc"), cutoff):
                continue
            captured_at = _instant(capture.get("captured_at_utc"))
            name = str(verdict.get("dataset", dataset))
            season = verdict.get("season", manifest.get("season"))
            line = _live_line(name, season, verdict, captured_at, corrections)
            found.append((captured_at, name, int(capture.get("sequence") or 0), line))
    oldest_first = sorted(
        found,
        key=lambda item: (
            item[0] or datetime.min.replace(tzinfo=UTC),
            item[1],
            item[2],
        ),
    )
    return [line for *_order, line in oldest_first]


def corrected_grades(
    corrections: Sequence[Mapping[str, Any]], cutoff: datetime | None
) -> list[str]:
    """Each live-arm bet whose result was corrected after *cutoff* (all when None)."""
    lines: list[str] = []
    for row in corrections:
        if not _live_arm(row.get("arm")):
            continue
        if not _after(row.get("corrected_at_utc"), cutoff):
            continue
        lines.append(
            f"{_bet_label(row.get('game_id'), row.get('target'))}: graded "
            f"{row.get('original_grading_status')}, corrected to "
            f"{row.get('corrected_grading_status')}"
        )
    return lines


def data_change_section(
    *,
    probe_line: Mapping[str, Any] | None,
    manifest: Mapping[str, Any] | None,
    corrections: Sequence[Mapping[str, Any]],
    cutoff: datetime | None,
    unreadable: Sequence[str] = (),
) -> dict[str, Any]:
    """The section's lines and its fixed statements, selected from stored output.

    Args:
        probe_line: The probe log's latest line, or None when the log is empty.
        manifest: The live season's manifest, or None when nothing was captured yet.
        corrections: The cache's ``bet_list_corrections`` rows.
        cutoff: The previous delivered digest's ``sent_at`` (None: the first digest).
        unreadable: One :data:`UNREADABLE_PROBE_LOG`-style line per source that could not be
            read; listed with the items, so the section never reads "nothing" in their place.

    Returns:
        ``{lines, empty_line, unaffected_line, acknowledge_line}``.
    """
    lines = [
        *sealed_changes(probe_line),
        *live_changes(manifest, cutoff, corrections),
        *corrected_grades(corrections, cutoff),
        *unreadable,
    ]
    return {
        "lines": lines,
        "empty_line": DATA_CHANGE_EMPTY_LINE,
        "unaffected_line": UNAFFECTED_LINE,
        "acknowledge_line": ACKNOWLEDGE_LINE,
    }


def _closing_line(event: Mapping[str, Any]) -> str | None:
    """One plain line for a run-log record worth the owner's attention, else None."""
    kind = event.get("event")
    if kind == "closing_capture" and event.get("outcome") in ("skipped", "failed"):
        return (
            f"closing reading {event.get('outcome')} ({event.get('reason')}): "
            f"{game_labels(event.get('game_ids') or ())}"
        )
    if kind == "closing_triggers":
        registered_wrong = (
            event.get("outcome") == "registered"
            and event.get("readback_match") is False
        )
        return TRIGGER_READBACK_LINE if registered_wrong else None
    if kind not in ("closing_finalized", "first_pick_refusal"):
        return None
    game_id, _season, _week, target, arm = event.get("key") or (None,) * 5
    if not _live_arm(arm):
        return None
    if kind == "first_pick_refusal":
        fields = ", ".join(str(name) for name in event.get("fields") or ())
        return (
            f"{_bet_label(game_id, target)}: a later pick was refused, the first pick "
            f"stands (differing {fields})"
        )
    reason = event.get("closing_null_reason")
    if reason is None:
        return None
    return f"{_bet_label(game_id, target)}: no closing reading ({reason})"


def closing_section_lines(
    events: Sequence[Mapping[str, Any]], cutoff: datetime | None
) -> list[str]:
    """SPEC R4 item 7: the run log's skipped, failed or missed closing readings, first-pick
    refusals and mismatched closing-schedule read-backs (D-23) after *cutoff* (all when None),
    one plain line each, in log order."""
    lines: list[str] = []
    for event in events:
        if not _after(event.get("at_utc"), cutoff):
            continue
        line = _closing_line(event)
        if line is not None:
            lines.append(line)
    return lines


def closing_section(cutoff: datetime | None) -> dict[str, Any]:
    """The closing section from ``forward_ledger.run_log.read_events()`` (path read at call time).

    A run log that cannot be read is ONE line naming the log and, when known, the line number:
    a malformed log is never shown as "nothing missed".

    Returns:
        ``{lines, empty_line}``.
    """
    try:
        events = run_log.read_events()
    except Exception as exc:  # noqa: BLE001 - named in the email, never read as "nothing"
        found = _LINE_NUMBER.search(str(exc))
        where = f"at line {found.group(1)}" if found else f"({type(exc).__name__})"
        lines = [
            f"The ledger run log {run_log.LEDGER_RUN_LOG.as_posix()} could not be read "
            f"{where}, so missed closing readings and first-pick refusals are not listed."
        ]
    else:
        lines = closing_section_lines(events, cutoff)
    return {"lines": lines, "empty_line": CLOSING_EMPTY_LINE}
