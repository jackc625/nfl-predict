"""RED skeleton (Plan 34-06 Task 1): interface only; the GREEN commit implements it."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from forward_ledger.remote_config import RemoteSpec


class GitRunner(Protocol):
    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]: ...


class MissingDeployKeyError(Exception):
    pass


@dataclass(frozen=True)
class PushOutcome:
    ok: bool
    pushed_sha: str | None
    error: str | None


def run_git(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    raise NotImplementedError


def ssh_command(spec: RemoteSpec) -> str:
    raise NotImplementedError


def git_push_env(
    spec: RemoteSpec, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    raise NotImplementedError
