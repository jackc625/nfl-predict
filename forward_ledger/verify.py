"""Verifying the forward ledger: chain, anchors and backup (Phase 34, Plan 34-10; LDGR-05, LDGR-06).

WHAT A CHAIN ALONE CANNOT SEE
-----------------------------
Recomputing every chain hash from ``GENESIS_HASH`` catches an edited, reordered or deleted entry,
but a ledger with its last k entries removed still verifies. The head committed on the
``ledger-anchor`` branch is what catches that, so this module compares the two:

* the LOCAL anchor is authoritative. Absent while the ledger holds entries, anchoring more rows
  than the ledger holds (a truncated tail), or naming a hash that is not the ledger's hash at its
  row count -- each is a FAILURE. Rows written after the last anchor are only a warning: the next
  sync anchors them.
* the REMOTE anchor (the public repository on GitHub, read anonymously) is the copy a local
  attacker cannot rewrite. Unreachable, skipped or behind is a WARNING -- a failed push is
  non-fatal by LDGR-05, so the remote may legitimately lag. A remote that DISAGREES (a row count
  beyond the ledger, or a hash that is not the ledger's at its count) is a FAILURE: that is not
  lag, it is a different history.
* the private BACKUP (the ``ledger/`` directory's own repository, D-01) is reported by how many
  commits it is ahead of its last successful push and how many files are uncommitted. Backup lag
  is always a warning.

LOCAL AND EXTERNAL EVIDENCE STAY SEPARATE
-----------------------------------------
:attr:`VerifyReport.local_ok` covers every check made against the files and refs on this machine;
:attr:`VerifyReport.remote` carries the external evidence on its own. A report is ``ok`` only when
the local checks pass and the remote does not disagree.

ONLY THE CLI CALLS THIS
-----------------------
Verification is a linear pass over the season, so it runs only in ``scripts/verify_ledger.py``,
never in a route (D-18, UIAP-01) and never in the daily run. It writes nothing to the ledger; the
remote read fetches the anchor into the local mirror ref ``refs/ledger/remote-anchor`` only. Every
git call goes through the injectable runner.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from forward_ledger.anchor import (
    AnchorFormatError,
    RemoteAnchorUnreachableError,
    read_local_anchor,
    read_remote_anchor,
)
from forward_ledger.backup import backup_repo_exists
from forward_ledger.canonical import GENESIS_HASH
from forward_ledger.remote_config import ANCHOR_REMOTE_HTTPS_URL, BACKUP_PUSHED_REF
from forward_ledger.store import (
    ChainVerdict,
    LedgerEntry,
    read_entries,
    verify_chain,
)
from forward_ledger.sync import BACKUP_REPO_MISSING
from forward_ledger.transport import (
    GitCommandError,
    GitRunner,
    resolve_ref,
    run_checked,
    run_git,
)

__all__ = [
    "REMOTE_BEHIND",
    "REMOTE_DISAGREES",
    "REMOTE_SKIPPED",
    "REMOTE_UNREACHABLE",
    "REMOTE_VERIFIED",
    "AnchorCheck",
    "BackupCheck",
    "RemoteCheck",
    "VerifyReport",
    "verify_ledger",
]

# The five states of the external evidence. Only a disagreement fails verification.
REMOTE_VERIFIED = "verified"
REMOTE_BEHIND = "behind"
REMOTE_UNREACHABLE = "unreachable"
REMOTE_SKIPPED = "skipped"
REMOTE_DISAGREES = "disagrees"

# The reason a truncated tail is reported with (LDGR-06 acceptance).
TRUNCATED_REASON = "ledger shorter than the anchored row count"


@dataclass(frozen=True)
class AnchorCheck:
    """The local anchor against the ledger. ``rows`` / ``head_hash`` are None when no anchor exists."""

    rows: int | None
    head_hash: str | None
    ok: bool
    reason: str | None


@dataclass(frozen=True)
class RemoteCheck:
    """The remote anchor against the ledger: one of the five ``REMOTE_*`` states.

    ``behind_by`` is the ledger's entry count minus the remote's row count when the remote was
    read and agrees; None otherwise.
    """

    state: str
    rows: int | None
    head_hash: str | None
    behind_by: int | None
    reason: str | None

    @property
    def failed(self) -> bool:
        return self.state == REMOTE_DISAGREES


@dataclass(frozen=True)
class BackupCheck:
    """The private backup's lag. Counts are None when the repository is absent or unreadable."""

    present: bool
    behind_by: int | None
    uncommitted: int | None
    error: str | None


@dataclass(frozen=True)
class VerifyReport:
    """Everything one verification found. ``failures`` and ``warnings`` are owner-readable lines."""

    entries: int
    head_hash: str
    chain: ChainVerdict
    local_anchor: AnchorCheck
    unanchored: int
    remote: RemoteCheck
    backup: BackupCheck
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def local_ok(self) -> bool:
        """Every check against this machine's files and refs passed."""
        return self.chain.ok and self.local_anchor.ok

    @property
    def ok(self) -> bool:
        """The local checks passed and the remote anchor does not disagree (lag is allowed)."""
        return self.local_ok and not self.remote.failed


def head_at(entries: Sequence[LedgerEntry], count: int) -> str:
    """The stored chain head after the first *count* entries (``GENESIS_HASH`` for none)."""
    return GENESIS_HASH if count == 0 else entries[count - 1].chain_hash


def _check_local_anchor(
    entries: Sequence[LedgerEntry], repo_dir: Path, runner: GitRunner
) -> AnchorCheck:
    count = len(entries)
    try:
        anchored = read_local_anchor(repo_dir, runner=runner)
    except (AnchorFormatError, GitCommandError) as error:
        return AnchorCheck(
            None, None, False, f"the local anchor could not be read: {error}"
        )

    if anchored is None:
        if count == 0:
            return AnchorCheck(None, None, True, None)
        reason = (
            f"no local ledger-anchor commit while the ledger holds {count} entries; the "
            "authoritative head is missing"
        )
        return AnchorCheck(None, None, False, reason)

    anchored_hash, anchored_rows = anchored
    if anchored_rows > count:
        reason = (
            f"{TRUNCATED_REASON} (the anchor commits {anchored_rows} entries, the ledger "
            f"holds {count})"
        )
        return AnchorCheck(anchored_rows, anchored_hash, False, reason)
    if head_at(entries, anchored_rows) != anchored_hash:
        reason = (
            f"the ledger's hash after {anchored_rows} entries is not the anchored head "
            f"{anchored_hash}"
        )
        return AnchorCheck(anchored_rows, anchored_hash, False, reason)
    return AnchorCheck(anchored_rows, anchored_hash, True, None)


def _check_remote(
    entries: Sequence[LedgerEntry],
    repo_dir: Path,
    *,
    url: str,
    runner: GitRunner,
    check_remote: bool,
) -> RemoteCheck:
    if not check_remote:
        return RemoteCheck(
            REMOTE_SKIPPED, None, None, None, "the remote anchor was not checked"
        )
    try:
        remote = read_remote_anchor(repo_dir, url=url, runner=runner)
    except RemoteAnchorUnreachableError as error:
        return RemoteCheck(REMOTE_UNREACHABLE, None, None, None, str(error))
    except AnchorFormatError as error:
        reason = f"the remote anchor at {url} is malformed: {error}"
        return RemoteCheck(REMOTE_DISAGREES, None, None, None, reason)

    count = len(entries)
    absent = remote is None
    remote_hash, remote_rows = (GENESIS_HASH, 0) if remote is None else remote
    if remote_rows > count:
        reason = f"the remote anchor commits {remote_rows} entries but the ledger holds {count}"
        return RemoteCheck(REMOTE_DISAGREES, remote_rows, remote_hash, None, reason)
    if head_at(entries, remote_rows) != remote_hash:
        reason = (
            f"the remote anchor's head {remote_hash} is not the ledger's hash after "
            f"{remote_rows} entries"
        )
        return RemoteCheck(REMOTE_DISAGREES, remote_rows, remote_hash, None, reason)

    behind = count - remote_rows
    reason = "the remote ledger-anchor branch does not exist yet" if absent else None
    state = REMOTE_VERIFIED if behind == 0 else REMOTE_BEHIND
    return RemoteCheck(state, remote_rows, remote_hash, behind, reason)


def _check_backup(ledger_dir: Path, runner: GitRunner) -> BackupCheck:
    if not backup_repo_exists(ledger_dir):
        return BackupCheck(False, None, None, BACKUP_REPO_MISSING)
    try:
        pushed = resolve_ref(ledger_dir, BACKUP_PUSHED_REF, runner=runner)
        revisions = f"{BACKUP_PUSHED_REF}..HEAD" if pushed is not None else "HEAD"
        behind = int(
            run_checked(runner, ["rev-list", "--count", revisions], cwd=ledger_dir)
        )
        status = run_checked(
            runner, ["status", "--porcelain", "--untracked-files=all"], cwd=ledger_dir
        )
    except GitCommandError as error:
        return BackupCheck(
            True, None, None, f"the backup repository could not be read: {error}"
        )
    return BackupCheck(True, behind, len(status.splitlines()), None)


def _anchor_messages(
    anchor: AnchorCheck, unanchored: int
) -> tuple[list[str], list[str]]:
    failures = [] if anchor.ok else [f"local anchor: {anchor.reason}"]
    warnings = (
        [
            f"{unanchored} ledger entries are newer than the local anchor; the next sync anchors them"
        ]
        if unanchored > 0
        else []
    )
    return failures, warnings


def _remote_messages(remote: RemoteCheck) -> tuple[list[str], list[str]]:
    if remote.failed:
        return [f"remote anchor disagrees: {remote.reason}"], []
    warnings: list[str] = []
    if remote.state in (REMOTE_UNREACHABLE, REMOTE_SKIPPED):
        warnings.append(f"remote anchor {remote.state}: {remote.reason}")
    elif remote.reason is not None:
        warnings.append(f"remote anchor: {remote.reason}")
    if remote.behind_by:
        warnings.append(
            f"the remote anchor is {remote.behind_by} entries behind the ledger"
        )
    return [], warnings


def _backup_messages(backup: BackupCheck) -> list[str]:
    if backup.error is not None:
        return [f"backup: {backup.error}"]
    warnings: list[str] = []
    if backup.behind_by:
        warnings.append(f"the backup holds {backup.behind_by} commits not yet pushed")
    if backup.uncommitted:
        warnings.append(f"the backup has {backup.uncommitted} uncommitted files")
    return warnings


def verify_ledger(
    ledger_dir: Path | str,
    repo_dir: Path | str,
    *,
    runner: GitRunner = run_git,
    remote_url: str = ANCHOR_REMOTE_HTTPS_URL,
    check_remote: bool = True,
) -> VerifyReport:
    """Verify the ledger in *ledger_dir* against the anchors in *repo_dir* and its backup.

    Args:
        ledger_dir: The ledger directory (also the private backup repository once set up).
        repo_dir: The public repository whose ``ledger-anchor`` branch carries the head.
        runner: The git runner.
        remote_url: Where the remote anchor is read from (the public repository over HTTPS).
        check_remote: False skips the remote read (reported as ``skipped``, a warning).

    Returns:
        The report; nothing is written to the ledger.

    Raises:
        forward_ledger.store.LedgerFormatError: the store exists but cannot be read.
    """
    ledger = Path(ledger_dir)
    repo = Path(repo_dir)
    entries = read_entries(ledger)
    chain = verify_chain(entries)

    local_anchor = _check_local_anchor(entries, repo, runner)
    unanchored = (
        len(entries) - local_anchor.rows
        if local_anchor.ok and local_anchor.rows is not None
        else 0
    )
    remote = _check_remote(
        entries, repo, url=remote_url, runner=runner, check_remote=check_remote
    )
    backup = _check_backup(ledger, runner)

    failures: list[str] = []
    warnings: list[str] = []
    if not chain.ok:
        failures.append(
            f"chain broken at seq {chain.first_broken_seq} (key {chain.first_broken_key}): "
            f"{chain.reason}"
        )
    anchor_failures, anchor_warnings = _anchor_messages(local_anchor, unanchored)
    remote_failures, remote_warnings = _remote_messages(remote)
    failures += anchor_failures + remote_failures
    warnings += anchor_warnings + remote_warnings + _backup_messages(backup)

    return VerifyReport(
        entries=len(entries),
        head_hash=chain.head_hash,
        chain=chain,
        local_anchor=local_anchor,
        unanchored=unanchored,
        remote=remote,
        backup=backup,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )
