"""RED skeleton (Plan 34-06 Task 2): interface only; the GREEN commit implements it."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from forward_ledger.remote_config import BACKUP_SPEC, RemoteSpec
from forward_ledger.transport import GitRunner, PushOutcome, run_git


def ensure_backup_repo(
    ledger_dir: Path,
    *,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> bool:
    raise NotImplementedError


def commit_backup(
    ledger_dir: Path,
    *,
    message: str,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> str | None:
    raise NotImplementedError


def push_backup(
    ledger_dir: Path,
    *,
    spec: RemoteSpec = BACKUP_SPEC,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> PushOutcome:
    raise NotImplementedError
