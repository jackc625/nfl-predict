"""Verifying the forward ledger: chain, anchors, backup, verdict weeks and settled results (Plan 34-10).

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

THE VERDICT WEEKS (LDGR-10, LDGR-11)
------------------------------------
Once the verdict-scope declaration is committed, every live row of its season must carry the label
the declaration gives its week, and every row in weeks W..end must carry every stamp, a recipe that
resolves in the committed registry and the declared fill convention. Every correction entry must
name an existing row. No declaration yet is a warning, never a default scope.

SETTLED RESULTS ARE RE-GRADED (D-19)
------------------------------------
The chain covers only the immutable half (LDGR-05 as locked), so an edited ``grading_status`` or
``payout_flat`` would still verify. A result is fully determined by the chained pick plus the public
score, so :func:`settled_result_check` re-grades EVERY terminal row (win/loss/push, any arm,
migrated rows included) from the silver scores through the one re-grade path
(``forward_ledger.settle.regrade_row`` -- no second grader) and compares it with the outcome IN
FORCE (the latest correction entry, else the row's own grade; D-05/D-06). A difference, a terminal
row whose game has no recorded score, or a store that cannot be read while terminal rows exist is a
failure naming the row. Fill and closing columns are never checked (D-19). Nothing is repaired: a
mismatch with no correction entry means a score changed without its owed correction or a stored
grade was edited, and both are the owner's to look at.

LOCAL AND EXTERNAL EVIDENCE STAY SEPARATE
-----------------------------------------
:attr:`VerifyReport.local_ok` covers every check made against the files and refs on this machine;
:attr:`VerifyReport.remote` carries the external evidence on its own. A report is ``ok`` only when
the local checks pass and the remote does not disagree.

ONLY THE CLI CALLS THIS
-----------------------
Verification is a linear pass over the season, so it runs only in ``scripts/verify_ledger.py``,
never in a route (D-18, UIAP-01) and never in the daily run. It writes nothing to the ledger or the
silver store; the remote read fetches the anchor into the local mirror ref
``refs/ledger/remote-anchor`` only. Every git call goes through the injectable runner.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from api.cache import GRADING_STATUS_PENDING, GRADING_STATUSES
from backtest.weekly_bet_list import DEFAULT_CHAIN_FIT_PATH, FrozenChainFitError
from forward_ledger.anchor import (
    AnchorFormatError,
    RemoteAnchorUnreachableError,
    read_local_anchor,
    read_remote_anchor,
)
from forward_ledger.backup import backup_repo_exists
from forward_ledger.canonical import ENTRY_KIND_CORRECTION, ENTRY_KIND_ROW, GENESIS_HASH
from forward_ledger.corrections import InForce, in_force_outcomes
from forward_ledger.declarations import (
    VERDICT_SCOPE_MODULE,
    UnknownRecipeError,
    VerdictScope,
    VerdictScopeMalformedError,
    VerdictScopeUndeclaredError,
    load_verdict_scope,
    resolve_recipe,
    verdict_scope_label,
)
from forward_ledger.remote_config import ANCHOR_REMOTE_HTTPS_URL, BACKUP_PUSHED_REF
from forward_ledger.schema import ARM_LIVE, LEDGER_ROW_KEY
from forward_ledger.settle import (
    DEFAULT_SILVER_DIR,
    live_strategies,
    load_silver_games,
    realized_values_from_scores,
    regrade_row,
)
from forward_ledger.store import (
    LEDGER_STAMP_COLUMNS,
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
    "REGRADE_TOLERANCE",
    "REMOTE_BEHIND",
    "REMOTE_DISAGREES",
    "REMOTE_SKIPPED",
    "REMOTE_UNREACHABLE",
    "REMOTE_VERIFIED",
    "AnchorCheck",
    "BackupCheck",
    "RegradeCheck",
    "RegradeMismatch",
    "RemoteCheck",
    "VerdictCheck",
    "VerifyReport",
    "settled_result_check",
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

# The replay tolerance: a re-graded payout or realized-units value within this of the stored one
# is equal. Re-grading is the same arithmetic on the same doubles, so it is exact in practice.
REGRADE_TOLERANCE: float = 1e-9

# A row whose own grade is one of these is re-graded (D-19); a pending row is the settle pass's.
_TERMINAL_GRADING_STATUSES: frozenset[str] = frozenset(
    status for status in GRADING_STATUSES if status != GRADING_STATUS_PENDING
)

# The mismatch reasons a re-grade can name besides a value difference.
NO_RECORDED_SCORE = "no_recorded_score"
NO_STRATEGY = "no_strategy_for_target"
REGRADE_UNAVAILABLE = "regrade_unavailable"


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
class VerdictCheck:
    """The verdict-week rules against the committed declaration, and the corrections' keys.

    ``stamps_ok`` / ``labels_ok`` are None when no declaration exists (nothing to check them
    against). ``declaration_error`` names a declaration that exists but is malformed.
    """

    declared: bool
    start_week: int | None
    end_week: int | None
    stamps_ok: bool | None
    labels_ok: bool | None
    corrections_ok: bool
    declaration_error: str | None

    @property
    def ok(self) -> bool:
        return (
            self.declaration_error is None
            and self.stamps_ok is not False
            and self.labels_ok is not False
            and self.corrections_ok
        )


@dataclass(frozen=True)
class RegradeMismatch:
    """One settled row whose in-force outcome the re-grade does not reproduce (D-19).

    ``seq`` is the ROW entry's seq; ``key`` its ``LEDGER_ROW_KEY`` values. ``reason`` is
    ``regrade_differs`` / ``correction_differs`` for a value difference (the latter when a
    correction entry is in force), ``no_recorded_score``, ``no_strategy_for_target``, or the
    class name of the exception the re-grade raised.
    """

    seq: int
    key: tuple[Any, ...]
    field: str
    stored: Any
    regraded: Any
    reason: str


@dataclass(frozen=True)
class RegradeCheck:
    """The D-19 settled-result check. ``unavailable`` names a store or record that could not be read."""

    checked: int
    mismatches: tuple[RegradeMismatch, ...]
    unavailable: str | None

    @property
    def ok(self) -> bool:
        return self.unavailable is None and not self.mismatches


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
    verdict: VerdictCheck
    regrade: RegradeCheck
    failures: tuple[str, ...]
    warnings: tuple[str, ...]

    @property
    def local_ok(self) -> bool:
        """Every check against this machine's files and refs passed."""
        return (
            self.chain.ok
            and self.local_anchor.ok
            and self.verdict.ok
            and self.regrade.ok
        )

    @property
    def ok(self) -> bool:
        """The local checks passed and the remote anchor does not disagree (lag is allowed)."""
        return self.local_ok and not self.remote.failed


def head_at(entries: Sequence[LedgerEntry], count: int) -> str:
    """The stored chain head after the first *count* entries (``GENESIS_HASH`` for none)."""
    return GENESIS_HASH if count == 0 else entries[count - 1].chain_hash


def key_text(values: Mapping[str, Any]) -> str:
    """An entry's ``LEDGER_ROW_KEY`` as ``game_id|season|week|target|arm``."""
    return "|".join(str(values.get(name)) for name in LEDGER_ROW_KEY)


def _stamp_problems(row: Mapping[str, Any], scope: VerdictScope) -> list[str]:
    """Why a verdict-week row's stamps fail: NULL stamps, an unregistered recipe, a foreign fill."""
    problems: list[str] = []
    missing = [name for name in LEDGER_STAMP_COLUMNS if row.get(name) is None]
    if missing:
        problems.append(f"NULL {', '.join(missing)}")
    recipe_id = row.get("recipe_id")
    if recipe_id is not None:
        try:
            resolve_recipe(recipe_id)
        except UnknownRecipeError:
            problems.append(
                f"recipe_id {recipe_id!r} does not resolve in the committed recipe registry"
            )
    fill_id = row.get("fill_convention_id")
    if fill_id is not None and fill_id != scope.fill_convention_id:
        problems.append(
            f"fill_convention_id {fill_id!r} is not the declaration's "
            f"{scope.fill_convention_id!r}"
        )
    return problems


def _verdict_week_failures(
    entries: Sequence[LedgerEntry], scope: VerdictScope
) -> tuple[list[str], list[str]]:
    """``(label failures, stamp failures)`` for every live row of the declared season.

    Every row's ``verdict_scope`` must be the label the declaration gives its week (Pitfall 13);
    every row inside ``start_week..end_week`` must carry every stamp, a registered recipe and the
    declaration's fill convention (LDGR-10, LDGR-11). Rows before W -- migrated rows with NULL
    stamps among them -- carry ``pre_verdict`` and need no stamp.
    """
    label_failures: list[str] = []
    stamp_failures: list[str] = []
    for entry in entries:
        row = entry.immutable
        if (
            entry.kind != ENTRY_KIND_ROW
            or row.get("season") != scope.season
            or row.get("arm") != ARM_LIVE
        ):
            continue
        where = f"seq {entry.seq} {key_text(row)}"
        expected = verdict_scope_label(row["season"], row["week"], scope)
        if row.get("verdict_scope") != expected:
            label_failures.append(
                f"verdict label: {where} is labelled {row.get('verdict_scope')!r}; the "
                f"declaration gives week {row['week']} the label {expected!r}"
            )
        if scope.start_week <= row["week"] <= scope.end_week:
            problems = _stamp_problems(row, scope)
            if problems:
                stamp_failures.append(f"verdict stamps: {where}: {'; '.join(problems)}")
    return label_failures, stamp_failures


def _orphan_correction_failures(entries: Sequence[LedgerEntry]) -> list[str]:
    """A correction entry whose key names no ledger row (the writer refuses one; verify re-checks)."""
    row_keys = {
        key_text(entry.immutable) for entry in entries if entry.kind == ENTRY_KIND_ROW
    }
    return [
        f"correction: seq {entry.seq} {key_text(entry.immutable)} names no ledger row"
        for entry in entries
        if entry.kind == ENTRY_KIND_CORRECTION
        and key_text(entry.immutable) not in row_keys
    ]


def _check_verdict(
    entries: Sequence[LedgerEntry], module_name: str
) -> tuple[VerdictCheck, list[str], list[str]]:
    """The verdict section: ``(check, failures, warnings)``.

    No declaration is a warning, never a default scope: before Plan 34-22 commits it there is
    legitimately none, and rows written then are labelled ``pre_verdict`` and never count.
    """
    failures: list[str] = []
    warnings: list[str] = []
    scope: VerdictScope | None = None
    declaration_error: str | None = None
    try:
        scope = load_verdict_scope(module_name)
    except VerdictScopeUndeclaredError:
        warnings.append(
            f"no verdict-scope declaration ({module_name}); the verdict-week stamp and label "
            "rules are not checked until it is committed"
        )
    except VerdictScopeMalformedError as error:
        declaration_error = str(error)
        failures.append(f"verdict scope declaration: {error}")

    stamps_ok: bool | None = None
    labels_ok: bool | None = None
    if scope is not None:
        label_failures, stamp_failures = _verdict_week_failures(entries, scope)
        labels_ok, stamps_ok = not label_failures, not stamp_failures
        failures += label_failures + stamp_failures

    orphans = _orphan_correction_failures(entries)
    failures += orphans
    check = VerdictCheck(
        declared=scope is not None,
        start_week=None if scope is None else scope.start_week,
        end_week=None if scope is None else scope.end_week,
        stamps_ok=stamps_ok,
        labels_ok=labels_ok,
        corrections_ok=not orphans,
        declaration_error=declaration_error,
    )
    return check, failures, warnings


def _is_terminal(entry: LedgerEntry) -> bool:
    return (
        entry.kind == ENTRY_KIND_ROW
        and (entry.grading or {}).get("grading_status") in _TERMINAL_GRADING_STATUSES
    )


def _same_value(stored: Any, regraded: Any) -> bool:
    """Equal within ``REGRADE_TOLERANCE``; None equals only None."""
    if stored is None or regraded is None:
        return stored is None and regraded is None
    return abs(float(stored) - float(regraded)) <= REGRADE_TOLERANCE


def _value_mismatches(
    seq: int, key: tuple[Any, ...], in_force: InForce, regraded: Mapping[str, Any]
) -> list[RegradeMismatch]:
    """Every field on which the in-force outcome differs from the re-grade."""
    reason = "correction_differs" if in_force.corrected else "regrade_differs"
    mismatches: list[RegradeMismatch] = []
    if in_force.grading_status != regraded["grading_status"]:
        mismatches.append(
            RegradeMismatch(
                seq,
                key,
                "grading_status",
                in_force.grading_status,
                regraded["grading_status"],
                reason,
            )
        )
    for field in ("payout_flat", "realized_units"):
        stored = getattr(in_force, field)
        if not _same_value(stored, regraded[field]):
            mismatches.append(
                RegradeMismatch(seq, key, field, stored, regraded[field], reason)
            )
    return mismatches


def settled_result_check(
    entries: Sequence[LedgerEntry],
    games: pd.DataFrame,
    strategies: Mapping[str, Any],
) -> tuple[int, list[RegradeMismatch]]:
    """Re-grade every terminal row from *games* scores; compare with its in-force outcome (D-19).

    Pure: nothing is written, and no row is mutated (``regrade_row`` grades an in-memory copy).

    Args:
        entries: The ledger, in file order.
        games: The silver ``games`` table (``forward_ledger.settle.load_silver_games``).
        strategies: ``{target -> strategy}`` (``forward_ledger.settle.live_strategies``).

    Returns:
        ``(rows checked, mismatches)``. Every terminal row is checked -- any arm, migrated
        pre-verdict rows included; pending rows are not, and fill and closing columns never are.
    """
    in_force = in_force_outcomes(entries)
    realized = realized_values_from_scores(games)
    checked = 0
    mismatches: list[RegradeMismatch] = []
    for entry in entries:
        if not _is_terminal(entry):
            continue
        checked += 1
        row = {**entry.immutable, **(entry.grading or {})}
        key = tuple(row[name] for name in LEDGER_ROW_KEY)
        force = in_force[key]
        target = str(row["target"])
        value = realized.get(target, {}).get(str(row["game_id"]))
        strategy = strategies.get(target)
        if value is None or strategy is None:
            reason = NO_RECORDED_SCORE if value is None else NO_STRATEGY
            mismatches.append(
                RegradeMismatch(
                    entry.seq, key, "grading_status", force.grading_status, None, reason
                )
            )
            continue
        # The row's own settlement instant, carried through unchanged; it is never compared.
        own_graded_at: Any = row.get("graded_at")
        try:
            regraded = regrade_row(row, strategy, value, graded_at=own_graded_at)
        except Exception as error:  # noqa: BLE001 - any re-grade failure is named, never skipped
            mismatches.append(
                RegradeMismatch(
                    entry.seq,
                    key,
                    "grading_status",
                    force.grading_status,
                    None,
                    type(error).__name__,
                )
            )
            continue
        mismatches += _value_mismatches(entry.seq, key, force, regraded)
    return checked, mismatches


def _check_regrade(
    entries: Sequence[LedgerEntry],
    silver_dir: Path,
    chain_fit_path: Path,
    strategies: Mapping[str, Any] | None,
) -> RegradeCheck:
    """Load the scores and strategies only when a terminal row exists; never skip silently."""
    if not any(_is_terminal(entry) for entry in entries):
        return RegradeCheck(0, (), None)
    try:
        games = load_silver_games(silver_dir)
    except (OSError, ValueError) as error:
        return RegradeCheck(0, (), f"{REGRADE_UNAVAILABLE}: {error}")
    if strategies is None:
        try:
            strategies = live_strategies(chain_fit_path)
        except FrozenChainFitError as error:
            return RegradeCheck(0, (), f"{REGRADE_UNAVAILABLE}: {error}")
    try:
        checked, mismatches = settled_result_check(entries, games, strategies)
    except KeyError as error:
        reason = (
            f"{REGRADE_UNAVAILABLE}: the silver games store under {silver_dir.as_posix()} "
            f"lacks the column {error}"
        )
        return RegradeCheck(0, (), reason)
    return RegradeCheck(checked, tuple(mismatches), None)


def _regrade_failures(regrade: RegradeCheck) -> list[str]:
    if regrade.unavailable is not None:
        return [regrade.unavailable]
    return [
        f"settled result: seq {mismatch.seq} {'|'.join(str(part) for part in mismatch.key)} "
        f"{mismatch.field} is {mismatch.stored!r} in force but re-grades to "
        f"{mismatch.regraded!r} ({mismatch.reason})"
        for mismatch in regrade.mismatches
    ]


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
    module_name: str = VERDICT_SCOPE_MODULE,
    silver_dir: Path | str = DEFAULT_SILVER_DIR,
    chain_fit_path: Path | str = DEFAULT_CHAIN_FIT_PATH,
    strategies: Mapping[str, Any] | None = None,
) -> VerifyReport:
    """Verify the ledger in *ledger_dir* against the anchors in *repo_dir* and its backup.

    Args:
        ledger_dir: The ledger directory (also the private backup repository once set up).
        repo_dir: The public repository whose ``ledger-anchor`` branch carries the head.
        runner: The git runner.
        remote_url: Where the remote anchor is read from (the public repository over HTTPS).
        check_remote: False skips the remote read (reported as ``skipped``, a warning).
        module_name: The verdict-scope declaration module (tests inject a fixture module).
        silver_dir: The silver store whose ``games.parquet`` holds the scores (D-19).
        chain_fit_path: The chain-fit record the strategies are built from.
        strategies: ``{target -> strategy}``; built from *chain_fit_path* when None.

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
    verdict, verdict_failures, verdict_warnings = _check_verdict(entries, module_name)
    regrade = _check_regrade(
        entries, Path(silver_dir), Path(chain_fit_path), strategies
    )

    failures: list[str] = []
    warnings: list[str] = []
    if not chain.ok:
        failures.append(
            f"chain broken at seq {chain.first_broken_seq} "
            f"{'|'.join(str(part) for part in chain.first_broken_key or ())}: {chain.reason}"
        )
    anchor_failures, anchor_warnings = _anchor_messages(local_anchor, unanchored)
    remote_failures, remote_warnings = _remote_messages(remote)
    failures += (
        anchor_failures
        + verdict_failures
        + _regrade_failures(regrade)
        + remote_failures
    )
    warnings += (
        anchor_warnings + verdict_warnings + remote_warnings + _backup_messages(backup)
    )

    return VerifyReport(
        entries=len(entries),
        head_hash=chain.head_hash,
        chain=chain,
        local_anchor=local_anchor,
        unanchored=unanchored,
        remote=remote,
        backup=backup,
        verdict=verdict,
        regrade=regrade,
        failures=tuple(failures),
        warnings=tuple(warnings),
    )
