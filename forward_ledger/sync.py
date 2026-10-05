"""RED skeleton (Plan 34-06 Task 2): interface only; the GREEN commit implements it."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from forward_ledger.remote_config import ANCHOR_SPEC, BACKUP_SPEC, RemoteSpec
from forward_ledger.transport import GitRunner, run_git


@dataclass(frozen=True)
class SyncOutcome:
    anchor_committed: bool
    anchor_pushed: bool
    anchor_error: str | None
    backup_committed: bool
    backup_pushed: bool
    backup_error: str | None
    refused_reason: str | None


def publish_ledger_state(
    *,
    ledger_dir: Path,
    repo_dir: Path,
    anchor_spec: RemoteSpec = ANCHOR_SPEC,
    backup_spec: RemoteSpec = BACKUP_SPEC,
    runner: GitRunner = run_git,
    log: Callable[..., Any] | None = None,
    now: datetime | None = None,
) -> SyncOutcome:
    raise NotImplementedError
