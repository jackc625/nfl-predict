"""RED skeleton for Plan 34-13 Task 1 (replaced by the GREEN commit)."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from forward_ledger.remote_config import (
    ANCHOR_KEY_PATH,
    BACKUP_KEY_PATH,
    KNOWN_HOSTS_PATH,
    PUBLIC_REPO_SLUG,
)
from forward_ledger.store import LEDGER_DIR

MASTER_RULESET_NAME = "master: owner-only updates (ledger deploy key refused)"
ANCHOR_RULESET_NAME = "ledger-anchor: no deletion, no force-push"


class SetupRefused(Exception):
    """A setup step refused to act."""


@dataclass(frozen=True)
class SetupPaths:
    anchor_key: Path = ANCHOR_KEY_PATH
    backup_key: Path = BACKUP_KEY_PATH
    known_hosts: Path = KNOWN_HOSTS_PATH
    ledger_dir: Path = LEDGER_DIR


DEFAULT_PATHS = SetupPaths()


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    input_bytes: bytes | None = None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[bytes]:
    raise NotImplementedError


RUNNER = _run


def master_ruleset_payload() -> dict:
    raise NotImplementedError


def anchor_ruleset_payload() -> dict:
    raise NotImplementedError


def create_ruleset(payload: dict, *, slug: str = PUBLIC_REPO_SLUG) -> int:
    raise NotImplementedError


def main(argv: list[str] | None = None, *, paths: SetupPaths = DEFAULT_PATHS) -> int:
    raise NotImplementedError
