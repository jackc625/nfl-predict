"""RED skeleton (Plan 34-06 Task 1): interface only; the GREEN commit implements it."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from forward_ledger.remote_config import (
    ANCHOR_REMOTE_HTTPS_URL,
    ANCHOR_SPEC,
    RemoteSpec,
)
from forward_ledger.transport import GitRunner, PushOutcome, run_git


class AnchorWriteError(Exception):
    pass


class AnchorRefspecError(Exception):
    pass


class RemoteAnchorUnreachableError(Exception):
    pass


@dataclass(frozen=True)
class AnchorCommit:
    sha: str
    head_hash: str
    row_count: int
    parent: str | None


def write_anchor_commit(
    repo_dir: Path,
    head_hash: str,
    row_count: int,
    *,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> AnchorCommit:
    raise NotImplementedError


def read_local_anchor(
    repo_dir: Path, *, runner: GitRunner = run_git
) -> tuple[str, int] | None:
    raise NotImplementedError


def _assert_anchor_refspec(push_args: Sequence[str]) -> None:
    raise NotImplementedError


def push_anchor(
    repo_dir: Path,
    *,
    spec: RemoteSpec = ANCHOR_SPEC,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> PushOutcome:
    raise NotImplementedError


def read_remote_anchor(
    repo_dir: Path, *, url: str = ANCHOR_REMOTE_HTTPS_URL, runner: GitRunner = run_git
) -> tuple[str, int] | None:
    raise NotImplementedError
