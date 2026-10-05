"""Publish the ledger's state after a run: anchor the head, back up the directory, never fail the run.

ONE FUNCTION, CALLED AFTER EVERY LEDGER-CHANGING RUN
----------------------------------------------------
:func:`publish_ledger_state` runs after the writer has released the ledger. In order:

  1. read the store and verify the chain -- a broken chain is never published (``sync_refused``);
  2. compare the ledger head with the local anchor; when it MOVED (a row or a correction entry)
     write exactly one anchor commit (``anchor_commit``) -- an unmoved head writes none
     (SPEC LDGR-05 empty edge; research Q5);
  3. push the anchor whenever the local anchor is ahead of the last successful push
     (``anchor_push``) -- so a push that failed on an earlier run is retried, and the remote then
     carries the CURRENT head;
  4. back up: commit the ledger directory when anything in it changed (``backup_commit``) --
     grading, fill and closing writes included, which make no anchor commit -- and push it
     whenever HEAD is ahead of the last successful push (``backup_push``). An absent backup
     repository is recorded as ``backup_repo_missing`` and NOT created here (creation is the
     owner-approved setup, Plan 34-13).

NEVER RAISES (LDGR-05, D-01, threat T-34-20)
--------------------------------------------
Every step catches its own failure into the :class:`SyncOutcome` and the run log; a failed push,
a missing key, a git that cannot start -- none of them fails the daily run. Even a failing log
callable is caught (and reported through :mod:`logging`).

EVERY EVENT NAMES THE HEAD IT PUBLISHED
---------------------------------------
Each event carries ``head_hash`` and ``entry_count`` -- the exact ``ledger_head`` this sync read
-- so a push outcome can be matched to the append that produced its head (Plan 34-23 correlates
on these plus the daily run's ``run_id``, which the caller's log adds).

``log`` is a callable ``(event, **fields)``; the caller passes ``forward_ledger.run_log.
record_event``. This module does not import the run log, so it carries no ordering dependency.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from forward_ledger.anchor import push_anchor, read_local_anchor, write_anchor_commit
from forward_ledger.backup import backup_repo_exists, commit_backup, push_backup
from forward_ledger.canonical import GENESIS_HASH
from forward_ledger.remote_config import (
    ANCHOR_PUSHED_REF,
    ANCHOR_REF,
    ANCHOR_SPEC,
    BACKUP_PUSHED_REF,
    BACKUP_SPEC,
    RemoteSpec,
)
from forward_ledger.store import ledger_head, read_entries, verify_chain
from forward_ledger.transport import GitRunner, resolve_ref, run_git

__all__ = ["BACKUP_REPO_MISSING", "SyncOutcome", "publish_ledger_state"]

logger = logging.getLogger(__name__)

# The recorded reason when the ledger directory is not (yet) the backup repository.
BACKUP_REPO_MISSING = "backup_repo_missing"

# The head of a ledger with no entries; an absent anchor is treated as anchoring it.
_EMPTY_HEAD: tuple[str, int] = (GENESIS_HASH, 0)

LogCallable = Callable[..., Any]


@dataclass(frozen=True)
class SyncOutcome:
    """What one sync did. ``*_error`` holds every failure of that half, joined by ``"; "``."""

    anchor_committed: bool
    anchor_pushed: bool
    anchor_error: str | None
    backup_committed: bool
    backup_pushed: bool
    backup_error: str | None
    refused_reason: str | None


def _describe(error: BaseException) -> str:
    return f"{type(error).__name__}: {error}"


def _join(errors: list[str]) -> str | None:
    return "; ".join(errors) if errors else None


class _EventLog:
    """Emits every event with the head this sync read; a failing log never breaks the sync."""

    def __init__(
        self, log: LogCallable | None, head_hash: str | None, entry_count: int | None
    ):
        self._log = log
        self._head = {"head_hash": head_hash, "entry_count": entry_count}

    def __call__(self, event: str, **fields: Any) -> None:
        if self._log is None:
            return
        try:
            self._log(event, **fields, **self._head)
        except Exception:
            logger.exception("the ledger run log refused the %s event", event)


def _publish_anchor(
    repo_dir: Path,
    head: tuple[str, int],
    *,
    spec: RemoteSpec,
    runner: GitRunner,
    emit: _EventLog,
) -> tuple[bool, bool, str | None]:
    """Steps 2-3: one anchor commit when the head moved; push when ahead of the last push."""
    errors: list[str] = []
    committed = False
    try:
        anchored = read_local_anchor(repo_dir, runner=runner) or _EMPTY_HEAD
        if anchored != head:
            written = write_anchor_commit(repo_dir, head[0], head[1], runner=runner)
            committed = True
            emit("anchor_commit", ok=True, sha=written.sha, parent=written.parent)
    except Exception as error:  # noqa: BLE001 - publish_ledger_state never raises
        errors.append(_describe(error))
        emit("anchor_commit", ok=False, error=_describe(error))

    pushed = False
    try:
        local = resolve_ref(repo_dir, ANCHOR_REF, runner=runner)
        if local is not None and local != resolve_ref(
            repo_dir, ANCHOR_PUSHED_REF, runner=runner
        ):
            outcome = push_anchor(repo_dir, spec=spec, runner=runner)
            pushed = outcome.ok
            if not outcome.ok and outcome.error:
                errors.append(outcome.error)
            emit(
                "anchor_push",
                ok=outcome.ok,
                pushed_sha=outcome.pushed_sha,
                error=outcome.error,
            )
    except Exception as error:  # noqa: BLE001 - publish_ledger_state never raises
        errors.append(_describe(error))
        emit("anchor_push", ok=False, error=_describe(error))
    return committed, pushed, _join(errors)


def _publish_backup(
    ledger_dir: Path,
    message: str,
    *,
    spec: RemoteSpec,
    runner: GitRunner,
    emit: _EventLog,
) -> tuple[bool, bool, str | None]:
    """Step 4: commit the ledger directory when it changed; push when ahead of the last push."""
    if not backup_repo_exists(ledger_dir):
        emit("backup_push", ok=False, reason=BACKUP_REPO_MISSING)
        return False, False, BACKUP_REPO_MISSING

    errors: list[str] = []
    committed = False
    try:
        sha = commit_backup(ledger_dir, message=message, runner=runner)
        if sha is not None:
            committed = True
            emit("backup_commit", ok=True, sha=sha)
    except Exception as error:  # noqa: BLE001 - publish_ledger_state never raises
        errors.append(_describe(error))
        emit("backup_commit", ok=False, error=_describe(error))

    pushed = False
    try:
        local = resolve_ref(ledger_dir, "HEAD", runner=runner)
        if local is not None and local != resolve_ref(
            ledger_dir, BACKUP_PUSHED_REF, runner=runner
        ):
            outcome = push_backup(ledger_dir, spec=spec, runner=runner)
            pushed = outcome.ok
            if not outcome.ok and outcome.error:
                errors.append(outcome.error)
            emit(
                "backup_push",
                ok=outcome.ok,
                pushed_sha=outcome.pushed_sha,
                error=outcome.error,
            )
    except Exception as error:  # noqa: BLE001 - publish_ledger_state never raises
        errors.append(_describe(error))
        emit("backup_push", ok=False, error=_describe(error))
    return committed, pushed, _join(errors)


def publish_ledger_state(
    *,
    ledger_dir: Path,
    repo_dir: Path,
    anchor_spec: RemoteSpec = ANCHOR_SPEC,
    backup_spec: RemoteSpec = BACKUP_SPEC,
    runner: GitRunner = run_git,
    log: LogCallable | None = None,
    now: datetime | None = None,
) -> SyncOutcome:
    """Anchor the ledger head when it moved and back up whatever changed. NEVER raises.

    Args:
        ledger_dir: The ledger directory (also the backup repository).
        repo_dir: The public repository whose ``ledger-anchor`` branch carries the head.
        anchor_spec: Where and how the anchor is pushed.
        backup_spec: Where and how the backup is pushed.
        runner: The git runner.
        log: ``(event, **fields)``, e.g. ``forward_ledger.run_log.record_event``; None logs
            nothing.
        now: The instant named in the backup commit message; ``datetime.now(UTC)`` by default.

    Returns:
        What was committed and pushed, every failure, and why a sync was refused.
    """
    ledger = Path(ledger_dir)
    try:
        entries = read_entries(ledger)
        verdict = verify_chain(entries)
        head = ledger_head(entries)
    except Exception as error:  # noqa: BLE001 - publish_ledger_state never raises
        reason = f"the ledger could not be read: {_describe(error)}"
        _EventLog(log, None, None)("sync_refused", ok=False, reason=reason)
        return SyncOutcome(False, False, None, False, False, None, reason)

    emit = _EventLog(log, head[0], head[1])
    if not verdict.ok:
        reason = (
            f"the ledger chain is broken at seq {verdict.first_broken_seq} "
            f"(key {verdict.first_broken_key}): {verdict.reason}"
        )
        emit("sync_refused", ok=False, reason=reason)
        return SyncOutcome(False, False, None, False, False, None, reason)

    anchor_committed, anchor_pushed, anchor_error = _publish_anchor(
        Path(repo_dir), head, spec=anchor_spec, runner=runner, emit=emit
    )
    instant = (now or datetime.now(tz=UTC)).isoformat()
    message = f"ledger: {head[1]} entries, head {head[0][:12]}, {instant}"
    backup_committed, backup_pushed, backup_error = _publish_backup(
        ledger, message, spec=backup_spec, runner=runner, emit=emit
    )
    return SyncOutcome(
        anchor_committed=anchor_committed,
        anchor_pushed=anchor_pushed,
        anchor_error=anchor_error,
        backup_committed=backup_committed,
        backup_pushed=backup_pushed,
        backup_error=backup_error,
        refused_reason=None,
    )
