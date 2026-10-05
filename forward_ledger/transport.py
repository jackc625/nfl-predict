"""How the ledger talks to git: one injectable runner and a push environment pinned to a deploy key.

EVERY GIT CALL GOES THROUGH A RUNNER
------------------------------------
The anchor, the backup and the sync never call ``subprocess`` themselves; they take a
:class:`GitRunner` (``run_git`` by default). Tests pass a spy or a failing runner around real git,
so they can prove "refused before git ran" and "a failed push never raises" without a network.

THE PUSH CAN NEVER FALL BACK TO THE OWNER'S KEY (D-03, 34-RESEARCH D)
---------------------------------------------------------------------
Probed on this machine: ssh with no ``-i`` silently offers the owner's default
``~/.ssh/id_ed25519`` and GitHub greets the ACCOUNT -- an admin push that would bypass the master
ruleset (D-04). So :func:`git_push_env` pins everything:

* the absolute Windows OpenSSH executable (never PATH resolution);
* ``-F none`` so no ``~/.ssh/config`` block can add or replace an identity;
* ``-i <deploy key>`` with ``IdentitiesOnly=yes`` and ``IdentityAgent=none`` (no agent keys);
* ``BatchMode=yes`` (never prompt) and ``StrictHostKeyChecking=yes`` against the pinned
  known_hosts file only (never ``accept-new``);
* ``GIT_SSH`` and ``SSH_AUTH_SOCK`` removed, ``GIT_TERMINAL_PROMPT=0``, and an explicit
  author/committer identity that does not depend on the S4U profile's ``.gitconfig``.

A missing key or known_hosts file raises :class:`MissingDeployKeyError` BEFORE git is invoked.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import os
import shlex
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from forward_ledger.remote_config import (
    CONNECT_TIMEOUT_SECONDS,
    GIT_IDENTITY_EMAIL,
    GIT_IDENTITY_NAME,
    PUSH_TIMEOUT_SECONDS,
    SSH_EXECUTABLE,
    RemoteSpec,
)

__all__ = [
    "GitCommandError",
    "GitRunner",
    "MissingDeployKeyError",
    "PushOutcome",
    "git_identity_env",
    "git_push_env",
    "push_ref",
    "resolve_ref",
    "run_checked",
    "run_git",
    "ssh_command",
]

# A push or fetch error is kept in the outcome and the run log; long stderr is cut to this.
_ERROR_TEXT_LIMIT = 500

# Inherited variables that could route a push through another ssh or another key.
_DROPPED_PUSH_VARIABLES: tuple[str, ...] = ("GIT_SSH", "SSH_AUTH_SOCK")


class GitRunner(Protocol):
    """Runs one ``git`` command and returns its completed process; never raises on exit status."""

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
    """A deploy key or the pinned known_hosts file is absent, so no push was attempted (D-03).

    Inherits bare ``Exception``: a broad ``OSError`` handler must not read it as a transient
    network failure.
    """


class GitCommandError(Exception):
    """A local git command exited non-zero; the message names the command and git's stderr."""


@dataclass(frozen=True)
class PushOutcome:
    """What one push did. A failed push is an outcome, never an exception (LDGR-05, D-01).

    ``pushed_sha`` is the local commit the push published; ``error`` carries git's stderr (cut to
    a bounded length) when ``ok`` is False, or a note when the push landed but recording it did
    not.
    """

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
    """Run ``git <args>`` in *cwd*, capturing bytes. A non-zero exit is returned, not raised.

    Raises:
        subprocess.TimeoutExpired: *timeout* elapsed (the child is killed).
        OSError: git could not be started.
    """
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        env=None if env is None else dict(env),
        input=input_bytes,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


def _error_text(result: subprocess.CompletedProcess[bytes]) -> str:
    text = (
        (result.stderr or result.stdout or b"")
        .decode("utf-8", errors="replace")
        .strip()
    )
    return text[:_ERROR_TEXT_LIMIT] or f"git exited {result.returncode}"


def run_checked(
    runner: GitRunner,
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    input_bytes: bytes | None = None,
) -> str:
    """Run a local git command that must succeed; return its stdout as stripped ASCII text.

    Raises:
        GitCommandError: the command exited non-zero.
    """
    result = runner(args, cwd=cwd, env=env, input_bytes=input_bytes)
    if result.returncode != 0:
        msg = f"git {' '.join(args[:2])} failed in {Path(cwd).as_posix()}: {_error_text(result)}"
        raise GitCommandError(msg)
    return result.stdout.decode("ascii", errors="replace").strip()


def resolve_ref(repo_dir: Path, ref: str, *, runner: GitRunner) -> str | None:
    """The commit *ref* points at, or None when it does not exist.

    Raises:
        GitCommandError: git failed for another reason (for example, *repo_dir* is not a repo).
    """
    result = runner(["rev-parse", "-q", "--verify", f"{ref}^{{commit}}"], cwd=repo_dir)
    if result.returncode == 0:
        return result.stdout.decode("ascii").strip()
    if result.returncode == 1:
        return None
    msg = f"git rev-parse {ref} failed in {Path(repo_dir).as_posix()}: {_error_text(result)}"
    raise GitCommandError(msg)


def git_identity_env(base_env: Mapping[str, str] | None = None) -> dict[str, str]:
    """A copy of *base_env* (the process environment by default) with the ledger's git identity.

    Commits written by the ledger never depend on the S4U profile's ``.gitconfig``, and git never
    prompts.
    """
    env = dict(os.environ if base_env is None else base_env)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GIT_AUTHOR_NAME"] = GIT_IDENTITY_NAME
    env["GIT_AUTHOR_EMAIL"] = GIT_IDENTITY_EMAIL
    env["GIT_COMMITTER_NAME"] = GIT_IDENTITY_NAME
    env["GIT_COMMITTER_EMAIL"] = GIT_IDENTITY_EMAIL
    return env


def _require_file(path: Path, what: str) -> None:
    if not Path(path).is_file():
        msg = (
            f"the {what} {Path(path).as_posix()} does not exist; refusing to push, because "
            "without it ssh could fall back to another identity"
        )
        raise MissingDeployKeyError(msg)


def ssh_command(spec: RemoteSpec) -> str:
    """The ``GIT_SSH_COMMAND`` that authenticates with *spec*'s deploy key and nothing else.

    Raises:
        MissingDeployKeyError: the key or the known_hosts file is absent.
    """
    _require_file(spec.key_path, "deploy key")
    _require_file(spec.known_hosts_path, "pinned known_hosts file")
    parts = [
        SSH_EXECUTABLE,
        "-F",
        "none",
        "-i",
        Path(spec.key_path).as_posix(),
        "-o",
        "IdentitiesOnly=yes",
        "-o",
        "IdentityAgent=none",
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={Path(spec.known_hosts_path).as_posix()}",
        "-o",
        f"ConnectTimeout={CONNECT_TIMEOUT_SECONDS}",
    ]
    # Git runs the command through a shell; quote any part a shell would split.
    return " ".join(shlex.quote(part) for part in parts)


def git_push_env(
    spec: RemoteSpec, base_env: Mapping[str, str] | None = None
) -> dict[str, str]:
    """The environment for one push to *spec*: deploy key pinned, no prompt, explicit identity.

    *base_env* is copied, never modified.

    Raises:
        MissingDeployKeyError: the key or the known_hosts file is absent (checked first, so a
            caller that raises here has run no git command).
    """
    command = ssh_command(spec)
    env = git_identity_env(base_env)
    for name in _DROPPED_PUSH_VARIABLES:
        env.pop(name, None)
    env["GIT_SSH_COMMAND"] = command
    return env


def push_ref(
    repo_dir: Path,
    push_args: Sequence[str],
    *,
    source_sha: str,
    pushed_ref: str,
    env: Mapping[str, str],
    runner: GitRunner,
) -> PushOutcome:
    """Run one already-guarded push; on success record *source_sha* under *pushed_ref*.

    Never raises for a git failure, a timeout or a git that cannot start: each becomes a failed
    :class:`PushOutcome`, so the next run retries (LDGR-05).
    """
    try:
        result = runner(push_args, cwd=repo_dir, env=env, timeout=PUSH_TIMEOUT_SECONDS)
    except (subprocess.TimeoutExpired, OSError) as error:
        return PushOutcome(
            ok=False, pushed_sha=None, error=f"push did not complete: {error}"
        )
    if result.returncode != 0:
        return PushOutcome(ok=False, pushed_sha=None, error=_error_text(result))

    try:
        recorded = runner(["update-ref", pushed_ref, source_sha], cwd=repo_dir, env=env)
    except (subprocess.TimeoutExpired, OSError) as error:
        note = f"pushed, but recording {pushed_ref} did not complete: {error}"
        return PushOutcome(ok=True, pushed_sha=source_sha, error=note)
    if recorded.returncode != 0:
        note = f"pushed, but recording {pushed_ref} failed: {_error_text(recorded)}"
        return PushOutcome(ok=True, pushed_sha=source_sha, error=note)
    return PushOutcome(ok=True, pushed_sha=source_sha, error=None)
