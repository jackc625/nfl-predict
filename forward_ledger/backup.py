"""The private backup: ``ledger/`` is its own git repository, pushed to the private repo (D-01, D-02).

WHY A NESTED REPOSITORY
-----------------------
D-02 requires that the ledger restore, verify and replay from the backup ALONE, with a dated
history so a damaged local file can never overwrite the last good copy. A git repository inside
the ledger directory gives exactly that: every run that changed anything is one commit, pushed
fast-forward only to the private repository's ``main``. The outer public repository ignores
``/ledger/``, so it never records the nested repository.

WHY ``.gitattributes`` COMES FIRST
----------------------------------
``core.autocrlf=true`` is set system-wide on this machine (34-RESEARCH E4). Without ``* -text``,
a restore by ``git clone`` would rewrite every LF in the JSON-lines store to CRLF -- the chain
would still verify (it hashes canonical content, not file bytes), but the restored files would no
longer be the bytes that were backed up. The first commit therefore holds exactly
``.gitattributes``, ``.gitignore`` (the writer lock and temp files never enter history) and a
``README.md``, and nothing else.

CREATION IS A SETUP STEP, NOT A RUN STEP
----------------------------------------
:func:`ensure_backup_repo` is called by the owner-approved setup (Plan 34-13); the daily sync
never creates the repository -- it records ``backup_repo_missing`` instead.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from forward_ledger.remote_config import (
    BACKUP_BRANCH_REF,
    BACKUP_PUSHED_REF,
    BACKUP_REPO_SLUG,
    BACKUP_SPEC,
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
    "BACKUP_REFSPEC",
    "BackupError",
    "backup_repo_exists",
    "commit_backup",
    "ensure_backup_repo",
    "push_backup",
]

# The only refspec a backup push carries: the current commit onto the private repo's main.
BACKUP_REFSPEC: str = f"HEAD:{BACKUP_BRANCH_REF}"

_BRANCH_NAME = BACKUP_BRANCH_REF.removeprefix("refs/heads/")

_GITATTRIBUTES = b"* -text\n"
_GITIGNORE = b"*.lock\n*.tmp\n"
_README = (
    "# nfl-predict forward ledger (private backup)\n"
    "\n"
    "This repository is the private backup of the 2026 forward bet ledger, pushed to\n"
    f"`{BACKUP_REPO_SLUG}` after every run that changed anything in `ledger/` (D-01).\n"
    "It holds the rows, the chain, correction entries, decision-input snapshots and the model\n"
    "artifacts the rows reference, so the ledger can be restored, verified and replayed from\n"
    "this repository alone (D-02).\n"
    "\n"
    "`.gitattributes` (`* -text`) keeps every file byte-identical on restore. To restore, follow\n"
    "the procedure in `docs/guides/RUNBOOK.md` of the nfl-predict repository.\n"
).encode("ascii")

# The first commit, by file name, in the order written.
_INITIAL_FILES: tuple[tuple[str, bytes], ...] = (
    (".gitattributes", _GITATTRIBUTES),
    (".gitignore", _GITIGNORE),
    ("README.md", _README),
)


class BackupError(Exception):
    """A local backup git step failed; the message names the step and git's stderr."""


def backup_repo_exists(ledger_dir: Path) -> bool:
    """True when *ledger_dir* is already a git repository (the backup has been set up)."""
    return (Path(ledger_dir) / ".git").exists()


def ensure_backup_repo(
    ledger_dir: Path,
    *,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Make *ledger_dir* the backup repository, once. Returns True when it was created now.

    Initializes ``main`` and commits exactly ``.gitattributes``, ``.gitignore`` and ``README.md``
    as the first commit -- any ledger file already present is left for the first backup commit.

    Raises:
        BackupError: a git step failed.
    """
    directory = Path(ledger_dir)
    if backup_repo_exists(directory):
        return False
    directory.mkdir(parents=True, exist_ok=True)
    commit_env = git_identity_env(env)
    try:
        run_checked(runner, ["init", "-b", _BRANCH_NAME], cwd=directory)
        for name, content in _INITIAL_FILES:
            (directory / name).write_bytes(content)
        names = [name for name, _ in _INITIAL_FILES]
        run_checked(runner, ["add", "--", *names], cwd=directory)
        run_checked(
            runner,
            ["commit", "-m", "ledger backup: initialize", "--", *names],
            cwd=directory,
            env=commit_env,
        )
    except GitCommandError as error:
        msg = f"the backup repository was not initialized: {error}"
        raise BackupError(msg) from error
    return True


def commit_backup(
    ledger_dir: Path,
    *,
    message: str,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Commit everything that changed in *ledger_dir*; None when nothing did.

    Ignored files (the writer lock, temp files) never enter history.

    Raises:
        BackupError: a git step failed.
    """
    directory = Path(ledger_dir)
    try:
        status = run_checked(
            runner, ["status", "--porcelain", "--untracked-files=all"], cwd=directory
        )
        if not status:
            return None
        run_checked(runner, ["add", "-A"], cwd=directory)
        run_checked(
            runner,
            ["commit", "-m", message],
            cwd=directory,
            env=git_identity_env(env),
        )
        return run_checked(runner, ["rev-parse", "HEAD"], cwd=directory)
    except GitCommandError as error:
        msg = f"the backup commit was not made: {error}"
        raise BackupError(msg) from error


def push_backup(
    ledger_dir: Path,
    *,
    spec: RemoteSpec = BACKUP_SPEC,
    runner: GitRunner = run_git,
    env: Mapping[str, str] | None = None,
) -> PushOutcome:
    """Push the backup's HEAD to *spec*'s ``main`` with the backup deploy key; never forced.

    Returns a failed :class:`PushOutcome` (never raises) when git fails or times out. On success
    ``BACKUP_PUSHED_REF`` is moved to the pushed commit.

    Raises:
        MissingDeployKeyError: the deploy key or known_hosts file is absent (before any git call).
    """
    push_env = git_push_env(spec, base_env=env)
    directory = Path(ledger_dir)
    try:
        head = resolve_ref(directory, "HEAD", runner=runner)
    except GitCommandError as error:
        return PushOutcome(ok=False, pushed_sha=None, error=str(error))
    if head is None:
        return PushOutcome(
            ok=False, pushed_sha=None, error="the backup has no commit to push"
        )
    return push_ref(
        directory,
        ["push", "--porcelain", spec.url, BACKUP_REFSPEC],
        source_sha=head,
        pushed_ref=BACKUP_PUSHED_REF,
        env=push_env,
        runner=runner,
    )
