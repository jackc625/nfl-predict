"""The ledger anchor: the chain's head, committed by plumbing on its own branch and pushed (LDGR-05).

WHY AN ANCHOR
-------------
A hash chain detects an edited or reordered entry, but not a truncated tail: a ledger with its last
k entries removed still verifies. The head committed somewhere the laptop cannot rewrite -- a
branch on GitHub -- is what makes truncation detectable (the verify CLI compares the two).

WHAT THE ANCHOR IS (D-12)
-------------------------
One commit per head move on ``refs/heads/ledger-anchor``. Its tree holds exactly one file,
``ANCHOR``, whose content is exactly::

    head_hash=<64 lowercase hex>
    row_count=<non-negative int>

and nothing else -- no game, pick, line or stake ever crosses into the public repository (SPEC
must-not). The content is validated BEFORE any blob is written.

WRITTEN BY PLUMBING, SO THE CHECKOUT IS NEVER TOUCHED
-----------------------------------------------------
``hash-object -w --stdin`` (exact bytes; autocrlf does not apply to stdin), ``mktree``,
``commit-tree`` and ``update-ref <ref> <new> <old>`` -- a compare-and-swap, so a ref moved by a
concurrent writer makes this write fail (:class:`AnchorWriteError`) instead of overwriting it. HEAD,
the index and the working tree of the checked-out repository are never read or written, so a live
run can anchor while ``master`` is checked out (34-RESEARCH E3, probed).

PUSHED BY ONE REFSPEC, NEVER FORCED
-----------------------------------
:func:`push_anchor` pushes exactly ``refs/heads/ledger-anchor:refs/heads/ledger-anchor`` to an
explicit URL with the anchor deploy key (:mod:`forward_ledger.transport`). The argument vector is
checked by an allow-list before git runs: any other refspec (``master`` included), a ``+`` prefix
or any option but ``--porcelain`` raises :class:`AnchorRefspecError`. A failed push returns a
failed :class:`PushOutcome` and never raises; the next sync pushes the current head.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from forward_ledger.remote_config import (
    ANCHOR_FILE_NAME,
    ANCHOR_PUSHED_REF,
    ANCHOR_REF,
    ANCHOR_REMOTE_HTTPS_URL,
    ANCHOR_SPEC,
    PUSH_TIMEOUT_SECONDS,
    REMOTE_ANCHOR_FETCH_REF,
    RemoteSpec,
)
from forward_ledger.transport import (
    GitCommandError,
    GitRunner,
    PushOutcome,
    git_identity_env,
    git_push_env,
    push_ref,
    resolve_ref,
    run_checked,
    run_git,
)

__all__ = [
    "ANCHOR_REFSPEC",
    "AnchorCommit",
    "AnchorFormatError",
    "AnchorRefspecError",
    "AnchorWriteError",
    "PushOutcome",
    "RemoteAnchorUnreachableError",
    "anchor_content",
    "push_anchor",
    "read_local_anchor",
    "read_remote_anchor",
    "write_anchor_commit",
]

# The only refspec an anchor push may carry.
ANCHOR_REFSPEC: str = f"{ANCHOR_REF}:{ANCHOR_REF}"

# ``update-ref``'s old value meaning "this ref must not exist yet".
_ZERO_OID = "0" * 40

_HEAD_HASH_RE = re.compile(r"[0-9a-f]{64}")
_ANCHOR_BLOB_RE = re.compile(rb"head_hash=([0-9a-f]{64})\nrow_count=(0|[1-9][0-9]*)\n")

# The only push option allowed; everything else (every force spelling, --mirror, --all, --tags,
# --delete, --prune ...) is refused by the allow-list rather than chased by a deny-list.
_ALLOWED_PUSH_OPTIONS: frozenset[str] = frozenset({"--porcelain"})

# git ls-remote --exit-code: "Exit with status 2 when no matching refs are found".
_LS_REMOTE_NO_MATCH = 2


class AnchorWriteError(Exception):
    """The anchor commit was not written: refused content, a failed git step, or a lost CAS."""


class AnchorRefspecError(Exception):
    """An anchor push named anything but the one anchor refspec, or a forbidden option (D-12)."""


class AnchorFormatError(Exception):
    """An anchor commit exists but is not exactly the one two-line ``ANCHOR`` file."""


class RemoteAnchorUnreachableError(Exception):
    """The remote could not be read (network, auth or timeout) -- NOT a claim the branch is absent."""


@dataclass(frozen=True)
class AnchorCommit:
    """One written anchor commit; ``parent`` is None for the first."""

    sha: str
    head_hash: str
    row_count: int
    parent: str | None


def anchor_content(head_hash: str, row_count: int) -> bytes:
    """The exact bytes of the ``ANCHOR`` file for (*head_hash*, *row_count*).

    Raises:
        AnchorWriteError: *head_hash* is not 64 lowercase hex characters, or *row_count* is not a
            non-negative ``int`` (``bool`` refused).
    """
    if not isinstance(head_hash, str) or _HEAD_HASH_RE.fullmatch(head_hash) is None:
        msg = f"anchor refused: head hash {head_hash!r} is not 64 lowercase hex characters"
        raise AnchorWriteError(msg)
    if type(row_count) is not int or row_count < 0:
        msg = f"anchor refused: row count {row_count!r} is not a non-negative integer"
        raise AnchorWriteError(msg)
    return f"head_hash={head_hash}\nrow_count={row_count}\n".encode("ascii")


def write_anchor_commit(
    repo_dir: Path,
    head_hash: str,
    row_count: int,
    *,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> AnchorCommit:
    """Commit (*head_hash*, *row_count*) on ``ANCHOR_REF`` by plumbing; never touch the checkout.

    Args:
        repo_dir: The repository whose ``ledger-anchor`` branch is written.
        head_hash: The ledger's head chain hash.
        row_count: The ledger's entry count.
        runner: The git runner.
        env: The base environment for the identity (the process environment by default).

    Raises:
        AnchorWriteError: the content was refused (before any git call), a git step failed, or
            ``ANCHOR_REF`` moved after its parent was read (compare-and-swap lost; the ref is left
            as the other writer set it).
    """
    content = anchor_content(head_hash, row_count)
    commit_env = git_identity_env(env)
    repo = Path(repo_dir)
    try:
        blob = run_checked(
            runner, ["hash-object", "-w", "--stdin"], cwd=repo, input_bytes=content
        )
        tree_line = f"100644 blob {blob}\t{ANCHOR_FILE_NAME}\n".encode("ascii")
        tree = run_checked(runner, ["mktree"], cwd=repo, input_bytes=tree_line)
        parent = resolve_ref(repo, ANCHOR_REF, runner=runner)
        parent_args = ["-p", parent] if parent is not None else []
        sha = run_checked(
            runner,
            [
                "commit-tree",
                tree,
                *parent_args,
                "-m",
                f"ledger anchor: rows={row_count}",
            ],
            cwd=repo,
            env=commit_env,
        )
        run_checked(
            runner,
            ["update-ref", "-m", "ledger anchor", ANCHOR_REF, sha, parent or _ZERO_OID],
            cwd=repo,
            env=commit_env,
        )
    except GitCommandError as error:
        msg = f"the anchor commit was not written: {error}"
        raise AnchorWriteError(msg) from error
    return AnchorCommit(
        sha=sha, head_hash=head_hash, row_count=row_count, parent=parent
    )


def _read_anchor_at(repo: Path, ref: str, runner: GitRunner) -> tuple[str, int]:
    """Parse the anchor commit at *ref* strictly: one ``ANCHOR`` file of exactly two lines."""
    try:
        names = run_checked(runner, ["ls-tree", "--name-only", ref], cwd=repo)
    except GitCommandError as error:
        raise AnchorFormatError(str(error)) from error
    if names.splitlines() != [ANCHOR_FILE_NAME]:
        msg = f"the anchor commit at {ref} holds {names.splitlines()}, not exactly one ANCHOR"
        raise AnchorFormatError(msg)

    blob = runner(["cat-file", "blob", f"{ref}:{ANCHOR_FILE_NAME}"], cwd=repo)
    match = _ANCHOR_BLOB_RE.fullmatch(blob.stdout) if blob.returncode == 0 else None
    if match is None:
        msg = f"the ANCHOR file at {ref} is not exactly 'head_hash=<64 hex>' and 'row_count=<int>'"
        raise AnchorFormatError(msg)
    return match.group(1).decode("ascii"), int(match.group(2))


def read_local_anchor(
    repo_dir: Path, *, runner: GitRunner = run_git
) -> tuple[str, int] | None:
    """``(head hash, row count)`` of the local ``ledger-anchor`` tip, or None when none exists.

    Raises:
        AnchorFormatError: the tip is not exactly the one two-line ``ANCHOR`` file.
        GitCommandError: git failed for another reason.
    """
    repo = Path(repo_dir)
    if resolve_ref(repo, ANCHOR_REF, runner=runner) is None:
        return None
    return _read_anchor_at(repo, ANCHOR_REF, runner)


def _assert_anchor_refspec(push_args: Sequence[str]) -> None:
    """Refuse any push but ``push [--porcelain] <url> refs/heads/ledger-anchor:<same>``.

    Raises:
        AnchorRefspecError: a forbidden option (every force spelling included), a refspec other
            than :data:`ANCHOR_REFSPEC` (``master``, ``HEAD`` and ``+``-prefixed ones included),
            or not exactly one URL and one refspec.
    """
    if not push_args or push_args[0] != "push":
        msg = f"an anchor push must be a 'git push', got {list(push_args)}"
        raise AnchorRefspecError(msg)
    arguments = list(push_args[1:])
    forbidden = [
        arg
        for arg in arguments
        if arg.startswith("-") and arg not in _ALLOWED_PUSH_OPTIONS
    ]
    if forbidden:
        msg = f"an anchor push refuses the option(s) {forbidden}; it is never forced"
        raise AnchorRefspecError(msg)
    positional = [arg for arg in arguments if not arg.startswith("-")]
    if len(positional) != 2 or positional[1] != ANCHOR_REFSPEC:
        msg = (
            f"an anchor push names exactly one URL and the refspec {ANCHOR_REFSPEC!r}; "
            f"got {positional[1:]}"
        )
        raise AnchorRefspecError(msg)


def push_anchor(
    repo_dir: Path,
    *,
    spec: RemoteSpec = ANCHOR_SPEC,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> PushOutcome:
    """Push the local ``ledger-anchor`` to *spec* with its deploy key; record it when it lands.

    Returns a failed :class:`PushOutcome` (never raises) when git fails, times out, or there is no
    local anchor to push. On success ``ANCHOR_PUSHED_REF`` is moved to the pushed commit.

    Raises:
        MissingDeployKeyError: the deploy key or known_hosts file is absent (before any git call).
        AnchorRefspecError: the push vector failed the guard (before the push runs).
    """
    push_env = git_push_env(spec, base_env=env)
    push_args = ["push", "--porcelain", spec.url, ANCHOR_REFSPEC]
    _assert_anchor_refspec(push_args)

    repo = Path(repo_dir)
    try:
        local = resolve_ref(repo, ANCHOR_REF, runner=runner)
    except GitCommandError as error:
        return PushOutcome(ok=False, pushed_sha=None, error=str(error))
    if local is None:
        return PushOutcome(
            ok=False, pushed_sha=None, error="no local anchor commit to push"
        )
    return push_ref(
        repo,
        push_args,
        source_sha=local,
        pushed_ref=ANCHOR_PUSHED_REF,
        env=push_env,
        runner=runner,
    )


def _remote_read(
    runner: GitRunner, args: Sequence[str], repo: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess[bytes]:
    try:
        return runner(args, cwd=repo, env=env, timeout=PUSH_TIMEOUT_SECONDS)
    except (subprocess.TimeoutExpired, OSError) as error:
        msg = f"the remote anchor could not be read: {error}"
        raise RemoteAnchorUnreachableError(msg) from error


def read_remote_anchor(
    repo_dir: Path, *, url: str = ANCHOR_REMOTE_HTTPS_URL, runner: GitRunner = run_git
) -> tuple[str, int] | None:
    """``(head hash, row count)`` of the anchor on *url*, or None when the branch is absent there.

    Read-only on the remote: ``ls-remote --exit-code`` tells "absent" (exit 2) from "unreachable",
    then the branch is fetched into the local ``REMOTE_ANCHOR_FETCH_REF`` (``+``: a local mirror
    ref, never pushed) with no credential helper and no prompt.

    Raises:
        RemoteAnchorUnreachableError: network, authentication or timeout failure.
        AnchorFormatError: the remote anchor is not exactly the one two-line ``ANCHOR`` file.
    """
    repo = Path(repo_dir)
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    no_helper = ["-c", "credential.helper="]

    probe = _remote_read(
        runner, [*no_helper, "ls-remote", "--exit-code", url, ANCHOR_REF], repo, env
    )
    if probe.returncode == _LS_REMOTE_NO_MATCH:
        return None
    if probe.returncode != 0:
        detail = probe.stderr.decode("utf-8", errors="replace").strip()
        msg = f"the remote anchor at {url} could not be listed: {detail}"
        raise RemoteAnchorUnreachableError(msg)

    fetch = _remote_read(
        runner,
        [
            *no_helper,
            "fetch",
            "--no-tags",
            "--no-write-fetch-head",
            url,
            f"+{ANCHOR_REF}:{REMOTE_ANCHOR_FETCH_REF}",
        ],
        repo,
        env,
    )
    if fetch.returncode != 0:
        detail = fetch.stderr.decode("utf-8", errors="replace").strip()
        msg = f"the remote anchor at {url} could not be fetched: {detail}"
        raise RemoteAnchorUnreachableError(msg)
    return _read_anchor_at(repo, REMOTE_ANCHOR_FETCH_REF, runner)
