#!/usr/bin/env python3
"""The 2026 season-close readout (Phase 38; SPEC R7-R9, D-08, D-16, D-20, D-25, D-26).

Accounts for every scheduled game of the verdict scope (season 2026, weeks 5-22, read from the
committed declaration through ``forward_ledger.declarations.load_verdict_scope``) and reports the
counted live-arm rows. It runs any time as a PREVIEW, labelled "season to date -- not the
verdict"; the one binding run happens after the Super Bowl is graded (Plan 38-09).

Output is ``KEY= value`` lines on stdout, in this order:

* ``READOUT= preview: season to date -- not the verdict``;
* ``SCOPE=`` -- the declared season, weeks, counted arm, playoffs, neutral sites, outcome rule and
  fill convention;
* ``LEDGER= head=<hash> entries=<n>`` -- the chain head and entry count of the ledger read;
* one ``EVIDENCE=`` line per evidence file (the run log, then the skip records): its path, line
  count and SHA-256, or ``missing`` (D-08);
* one ``COVERAGE=`` line per scope week, in week order: scheduled, decided, bet, no_bet,
  failed_or_refused, missed, not_yet_due, not_played (the per-game rule is
  ``scripts.season_coverage.classify_coverage``), then ``COVERAGE_TOTAL=`` with the same fields
  summed and ``MISSED_GAMES=``;
* one ``PARTIAL= <game_id> targets=<a>,<b>`` line per decided game with rows for fewer than the
  verdict's targets, or ``PARTIAL= none``;
* one ``NOT_PLAYED= <game_id> week=<w> reason=inferred: ...`` line per game inferred never played
  (D-26, the reason it was inferred stated), or ``NOT_PLAYED= none``;
* one ``VOID_BETS= <game_id> target=<t>`` line per counted bet on a not-played game (D-20: void,
  in no figure), or ``VOID_BETS= none``;
* one ``FIGURES=`` line per verdict target, in ``VERDICT_TARGETS`` order: counted_rows, bets,
  graded, wins, losses, pushes, pending, units and flat ROI -- raw figures only;
* last, ``REFIT_SECTION= omitted in preview: ...`` (D-25).

THE PREVIEW SHOWS NO TEST RESULT (R9, D-16, D-25, Phase 37 D-13). It computes no p-value, no
Benjamini-Hochberg adjustment and no verdict word, and it does not run the re-fit comparison
(whose output carries p-values and verdict words): a mid-season number read as the verdict is the
misreading the season-close rule forbids.

ONE COMPUTATION OF THE FIGURES. :func:`target_bets` is the readout's one bet-selection rule and
:func:`target_figures` the one computation of the per-target raw figures built on it. The binding
verdict (Plan 38-09) takes its bets, wins, losses, pushes, units and flat ROI from
:func:`target_figures` and its per-bet test frame from :func:`target_bets`, and never recomputes
them, so the preview's and the binding run's figures cannot drift apart.

The ledger is read through its chain check (``forward_ledger.store.verify_chain``): nothing is
reported over a ledger whose chain does not verify. The preview writes no file anywhere (D-16).

THE BINDING RUN (``--binding``; Plan 38-09, SPEC R8-R9, D-09, D-24)
--------------------------------------------------------------------
Run ONCE, after the Super Bowl is graded. It reads the same stores, checks every condition below
IN THIS ORDER before anything is computed or written, and prints ``BINDING_REFUSED= <sentence>``
naming the first one unmet:

0. the ledger holds no entry, the repository holds no ``ledger-anchor`` commit, or the ledger does
   not extend that anchored head (``forward_ledger.store.assert_extends_anchor``, the check the
   daily run's settle pass applies): an absent, emptied or stale-backup ledger is never judged;
1. a counted bet (status live) of a played game is not graded (suppressed rows and void bets never
   block);
2. an in-scope game that is not "not played" lacks a final score, or a game inferred not played
   carries a counted bet already graded win, loss or push (a grade is direct evidence that a final
   score existed, so its missing silver score is a store contradiction, never a void game);
3. the schedule holds no played week-22 game;
4. the binding record ``<repo root>/docs/records/SEASON-2026-READOUT.md`` exists, or that path
   appears anywhere in git history (``git log --all``);
5. the R6 lock commit (the commit that added ``backtest/verdict_test_2026.py``) cannot be found,
   or cannot be proven: the repository is a shallow clone (its boundary commit reads as adding
   every file), or the oldest commit adding the rule is not the witnessed lock
   (:data:`WITNESSED_LOCK_SHA`), was not committed before ``PREREGISTRATION_DEADLINE_UTC``, or does
   not touch exactly the two R6 files (rewritten or squashed history);
6. a commit after the lock touches ``SEASON-VERDICT-PREREGISTRATION.md`` or
   ``backtest/verdict_test_2026.py`` without naming the lock sha as INVALIDATED, or either file
   has an uncommitted change;
7. a frozen value (``FROZEN_VALUE_PINS``: alpha, the bootstrap B and seed, the family size) no
   longer holds its pinned number;
8. the re-fit comparison (``scripts.compare_refit_arms``) cannot run (exit other than 0).

What each refusal means: (0) needs the ledger restored from the private backup (RUNBOOK); (1)-(3)
clear by waiting for the next 17:00 ET daily run to grade the
bets and record the scores; (4) means the binding run already happened and will never run again;
(5)-(7), and any failed git command, are repository-integrity refusals -- stop and investigate,
never work around them; (8) needs the store the comparison names fixed.

When every condition holds it computes each target's verdict (:func:`measure_targets`), embeds the
comparison's stdout byte for byte, and writes the one document atomically (sibling ``.tmp`` then
replace, ASCII, LF line endings) to ``<repo root>/docs/records/SEASON-2026-READOUT.md`` -- the path
is DERIVED from :data:`REPO_ROOT` and never chosen (there is no option naming another path or
another repository). ``REPO_ROOT`` is the checkout this file was imported from, so the history, the
uncommitted changes and the one-shot record the run checks belong to the same tree whose rule and
verdict code it executes. In binding mode a relative store path (every default is one) resolves
against that root, never against the working directory the command happens to run from. It then
prints ``BINDING_WRITTEN= <path>`` and one ``VERDICT_<TARGET>= <word>`` line per target. No run
ledger is kept (D-09): the output is deterministic (fixed bootstrap seed; nothing writes the ledger
after the Super Bowl).

No module under ``api/`` may import this CLI; nothing on the daily run imports it either.

Exit codes: 0 the readout was printed (preview) or the binding document was written (binding); 2 a
store cannot be read -- the ledger is unreadable or its chain does not verify
(``LEDGER_UNREADABLE=``), or the silver schedule, the run log or the skip records cannot be read
(``STORE_UNREADABLE=``, naming the exception class); 3 the binding run refused
(``BINDING_REFUSED=``). A malformed evidence line refuses; it is never read as absent.

Usage:
    uv run python -m scripts.season_readout
    uv run python -m scripts.season_readout --ledger-dir path/to/ledger --silver-dir data/silver
    uv run python -m scripts.season_readout --binding

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import backtest.verdict_test_2026 as rule
from api.cache import (
    BET_STATUS_LIVE,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
)
from backtest.profitability_2025 import (
    TargetMeasurement,
    assign_verdict_token,
    benjamini_hochberg_adjusted,
)
from backtest.roi_significance import flat_roi, roi_ci_and_p
from backtest.weekly_bet_list import _replace_atomically
from forward_ledger.anchor import AnchorFormatError, read_local_anchor
from forward_ledger.declarations import VerdictScope, load_verdict_scope
from forward_ledger.settle import (
    DEFAULT_SILVER_DIR,
    SILVER_GAMES_FILENAME,
    load_silver_games,
    realized_values_from_scores,
)
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerChainBrokenError,
    LedgerEntry,
    LedgerFormatError,
    assert_extends_anchor,
    ledger_head,
    read_entries,
    verify_chain,
)
from forward_ledger.transport import GitCommandError
from forward_ledger.verdict import verdict_rows
from pipeline.skip_log import SKIP_RECORD_PATH, SkipRecordCorrupt, parse_skip_records
from scripts import compare_refit_arms
from scripts.season_coverage import (
    STATE_NOT_PLAYED,
    CoverageReport,
    EvidenceFile,
    EvidenceUnreadableError,
    WeekCoverage,
    classify_coverage,
    parse_run_log,
    read_evidence,
    refused_game_ids,
)
from utils.current_slate import (
    AmbiguousSlateError,
    ScheduleUnavailableError,
    load_recorded_schedule,
)
from utils.game_lock import MissingKickoffError

PREVIEW_LABEL: str = "season to date -- not the verdict"

# The repository this file lives in, and the only one the binding run checks and writes into:
# the checked tree is always the executing tree (tests point it at a synthetic repository).
REPO_ROOT: Path = Path(__file__).resolve().parent.parent

# The binding command a reader re-runs; named in the document's first sentence.
BINDING_COMMAND: str = "uv run python -m scripts.season_readout --binding"

# The two files of the R6 record (one pre-registration in two files), repository-relative.
RULE_MODULE_PATH: str = "backtest/verdict_test_2026.py"
RULE_PATHS: tuple[str, ...] = ("SEASON-VERDICT-PREREGISTRATION.md", RULE_MODULE_PATH)

# The R6 lock commit as witnessed after it was pushed: the commit of both files in
# tests/phase38_state.py P38_PREREGISTRATION_WITNESS (Plan 38-03), asserted equal there by
# tests/unit/test_season_readout_binding.py. Copied, not imported: a CLI never imports a test
# module. A lock git finds that is not this commit is a lock that cannot be proven (refusal 5).
WITNESSED_LOCK_SHA: str = "922b010b70d973a21945c15ca88067dae4149ad5"

# Pins the binding run CHECKS (refusal 7), never values it computes with: the computation reads
# backtest.verdict_test_2026, whose alpha, bootstrap B and seed are imported by identity from
# upstream modules an edit could move without touching either R6 file.
# tests/unit/test_phase38_preregistration_ancestry.py (Plan 38-03) pins the same values.
FROZEN_VALUE_PINS: dict[str, float | int] = {
    "ALPHA": 0.05,
    "BOOTSTRAP_B": 2000,
    "BOOTSTRAP_SEED": 2704,
    "BH_FAMILY_SIZE": 3,
}

# The daily run's record of every run, written by scripts/daily_lock_pipeline.py
# (DAILY_RUN_RECORDS). Named here, not imported: that module loads the whole daily run.
DEFAULT_RUN_LOG: Path = Path("logs/daily_lock_runs.jsonl")

# The per-week fields, in output order (decided is a property of WeekCoverage).
_COVERAGE_FIELDS: tuple[str, ...] = (
    "scheduled",
    "decided",
    "bet",
    "no_bet",
    "failed_or_refused",
    "missed",
    "not_yet_due",
    "not_played",
)

# The columns a bet frame carries even when no row was counted (an empty ledger's verdict_rows
# frame may carry no columns at all): the figures and the binding run read these.
BET_COLUMNS: tuple[str, ...] = (
    "target",
    "season",
    "week",
    "game_id",
    "status",
    "grading_status",
    "realized_units",
    "payout_flat",
    "flat_stake",
)

# The columns the flat ROI is computed over (backtest.roi_significance.flat_roi).
_ROI_COLUMNS: list[str] = ["season", "week", "payout_flat", "flat_stake"]

_GRADED_STATUSES: tuple[str, ...] = (
    GRADING_STATUS_WIN,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PUSH,
)

REFIT_SECTION_LINE: str = (
    "REFIT_SECTION= omitted in preview: the re-fit comparison carries the season-close test "
    "results, which only the binding run shows (D-25)"
)

_STORE_ERRORS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    ScheduleUnavailableError,
    MissingKickoffError,
    AmbiguousSlateError,
    EvidenceUnreadableError,
    SkipRecordCorrupt,
)


@dataclass(frozen=True)
class TargetFigures:
    """One target's raw season-to-date figures (no test, no verdict word).

    Attributes:
        target: ``wp``, ``ats`` or ``ou``.
        counted_rows: The target's counted live-arm rows, rows of a void game left out.
        bets: Its counted bets (status live) on games that are not void.
        graded: Bets whose in-force grade is win, loss or push.
        wins: Bets graded win.
        losses: Bets graded loss.
        pushes: Bets graded push.
        pending: Bets not yet graded.
        units: The sum of the graded bets' in-force realized units.
        flat_roi: The graded bets' flat-stake ROI, or None when no bet is graded.
    """

    target: str
    counted_rows: int
    bets: int
    graded: int
    wins: int
    losses: int
    pushes: int
    pending: int
    units: float
    flat_roi: float | None


def _on_games(counted: pd.DataFrame, game_ids: frozenset[str]) -> pd.Series:
    """True for each counted row whose game is one of *game_ids*."""
    on_games = counted["game_id"].astype(str).isin(sorted(game_ids))
    return pd.Series(on_games, index=counted.index, dtype=bool)


def target_bets(
    counted: pd.DataFrame, target: str, void_game_ids: frozenset[str]
) -> pd.DataFrame:
    """THE readout's one bet-selection rule: *target*'s counted bets on games that are not void.

    A bet is a counted row with status ``live``; a bet on a void (not-played) game leaves every
    figure (D-20). Rows keep the counted frame's order. The binding verdict (Plan 38-09
    ``measure_targets``) takes its per-bet test frame from here and never re-selects.

    Args:
        counted: The counted rows (``forward_ledger.verdict.verdict_rows``), in-force grades.
        target: ``wp``, ``ats`` or ``ou``.
        void_game_ids: The games whose bets are void.

    Returns:
        The selected rows; an empty frame carrying :data:`BET_COLUMNS` when nothing is counted.
    """
    if counted.empty:
        return pd.DataFrame(columns=pd.Index(BET_COLUMNS))
    keep = (
        (counted["target"] == target)
        & (counted["status"] == BET_STATUS_LIVE)
        & ~_on_games(counted, void_game_ids)
    )
    return counted.loc[keep]


def _counted_rows(
    counted: pd.DataFrame, target: str, void_game_ids: frozenset[str]
) -> int:
    """The target's counted rows, rows of a void game left out."""
    if counted.empty:
        return 0
    keep = (counted["target"] == target) & ~_on_games(counted, void_game_ids)
    return int(keep.sum())


def target_figures(
    counted: pd.DataFrame, void_game_ids: frozenset[str]
) -> tuple[TargetFigures, ...]:
    """THE per-target raw figures, one per ``VERDICT_TARGETS`` entry, in that order.

    Every bet field is derived from :func:`target_bets`. The binding verdict (Plan 38-09
    ``measure_targets``) takes its bets, wins, losses, pushes, units and flat ROI from here and
    never recomputes them, so the preview's and the binding run's figures cannot drift apart.
    """
    figures: list[TargetFigures] = []
    for target in rule.VERDICT_TARGETS:
        bets = target_bets(counted, target, void_game_ids)
        status = bets["grading_status"]
        graded = bets.loc[status.isin(_GRADED_STATUSES)]
        units = float(graded["realized_units"].astype(float).sum(skipna=True))
        figures.append(
            TargetFigures(
                target=target,
                counted_rows=_counted_rows(counted, target, void_game_ids),
                bets=len(bets),
                graded=len(graded),
                wins=int((status == GRADING_STATUS_WIN).sum()),
                losses=int((status == GRADING_STATUS_LOSS).sum()),
                pushes=int((status == GRADING_STATUS_PUSH).sum()),
                pending=int((status == GRADING_STATUS_PENDING).sum()),
                units=units,
                flat_roi=None if graded.empty else flat_roi(graded[_ROI_COLUMNS]),
            )
        )
    return tuple(figures)


def not_played_game_ids(report: CoverageReport) -> frozenset[str]:
    """Every game inferred not played (D-26): the void games every figure leaves out.

    The set of GAMES, never the games some target happened to bet on, so whether a target's
    rows on a not-played game count never depends on what another target did there.
    """
    return frozenset(game.game_id for game in report.not_played)


def void_bets(counted: pd.DataFrame, report: CoverageReport) -> pd.DataFrame:
    """The counted bets on not-played games (D-20: void), in the counted frame's order.

    For listing the void bets by name only; the figures leave out every row of every game in
    :func:`not_played_game_ids`.
    """
    if counted.empty or not report.not_played:
        return pd.DataFrame(columns=pd.Index(BET_COLUMNS))
    not_played = not_played_game_ids(report)
    keep = (counted["status"] == BET_STATUS_LIVE) & _on_games(counted, not_played)
    return counted.loc[keep]


# ---------------------------------------------------------------------------
# The binding verdict (Plan 38-09; SPEC R8, D-01..D-04, D-12, D-13)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetResult:
    """One target's pre-registered verdict, built on the preview's figures.

    Every value is UNROUNDED (SPEC AC 25): rounding happens only in the rendered document.

    Attributes:
        figures: The target's raw figures, from :func:`target_figures` unchanged.
        raw_p: ``roi_ci_and_p``'s one-sided p-value, or None when it gives none.
        adjusted_p: The Benjamini-Hochberg adjusted p at m = ``BH_FAMILY_SIZE``, or None.
        p_absent_reason: ``roi_ci_and_p``'s reason the p-value is absent, when it is.
        n_blocks: The (season, week) blocks the target's bets span.
        token: The verdict word ``assign_verdict_token`` gave.
        meaning: The word's 2026 meaning (``VERDICT_TOKEN_MEANINGS_2026``), never the 2025 reason.
    """

    figures: TargetFigures
    raw_p: float | None
    adjusted_p: float | None
    p_absent_reason: str | None
    n_blocks: int
    token: str
    meaning: str


def measure_targets(
    counted: pd.DataFrame, void_game_ids: frozenset[str]
) -> tuple[TargetResult, ...]:
    """Each target's verdict under the R6 rule, in ``VERDICT_TARGETS`` order.

    Built ON the preview's computation, never beside it: the bets, wins, losses, pushes, units and
    flat ROI are :func:`target_figures`'s, and the per-bet test frame is :func:`target_bets`'s,
    reduced to the four columns the statistic reads (no closing-line value is in scope). The raw
    p-values present are corrected ONCE at m = ``BH_FAMILY_SIZE`` (a target without a p-value
    keeps its place in the family), and each word comes from ``assign_verdict_token`` on a
    ``TargetMeasurement`` filled by ``MEASUREMENT_MAPPING_2026``.

    Raises:
        ValueError: a target has a bet that is not graded. The binding run's ordered refusals make
            this impossible there; the check keeps the verdict from testing a partial bet frame.
    """
    figures = target_figures(counted, void_game_ids)
    stats: dict[str, dict[str, Any]] = {}
    for fig in figures:
        if fig.graded != fig.bets:
            msg = (
                f"[{fig.target}] {fig.bets - fig.graded} counted bet(s) are not graded; the "
                "verdict is never computed over a partial bet frame (D-24)."
            )
            raise ValueError(msg)
        per_bet = target_bets(counted, fig.target, void_game_ids).loc[:, _ROI_COLUMNS]
        stats[fig.target] = roi_ci_and_p(per_bet)

    present = [
        fig.target for fig in figures if stats[fig.target]["p_value"] is not None
    ]
    adjusted = dict(
        zip(
            present,
            benjamini_hochberg_adjusted(
                [float(stats[target]["p_value"]) for target in present],
                m=rule.BH_FAMILY_SIZE,
            ),
            strict=True,
        )
    )

    results: list[TargetResult] = []
    for fig in figures:
        stat = stats[fig.target]
        measurement = TargetMeasurement(
            target=fig.target,
            chain_resolved=fig.counted_rows > 0,
            control_passed=True,
            control_bet_count=0,
            bet_count=fig.bets,
            flat_roi=fig.flat_roi,
            adjusted_p=adjusted.get(fig.target),
            raw_p=stat["p_value"],
            p_absent_reason=stat["p_value_absent_reason"],
            n_blocks=int(stat["n_blocks"]),
            n_push=fig.pushes,
            n_ungraded=0,
            fallback_fired=False,
            fallback_trigger=None,
        )
        token, _reason_2025 = assign_verdict_token(
            measurement,
            alpha=rule.ALPHA,
            min_attainable_p=rule.ROI_MIN_ATTAINABLE_P,
        )
        results.append(
            TargetResult(
                figures=fig,
                raw_p=measurement.raw_p,
                adjusted_p=measurement.adjusted_p,
                p_absent_reason=measurement.p_absent_reason,
                n_blocks=measurement.n_blocks,
                token=token,
                meaning=rule.VERDICT_TOKEN_MEANINGS_2026[token],
            )
        )
    return tuple(results)


# ---------------------------------------------------------------------------
# The binding run's checks on the frozen rule (D-09, SPEC R9)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RuleLock:
    """The R6 lock commit as git records it.

    Attributes:
        sha: The oldest commit that added ``backtest/verdict_test_2026.py`` (the lock commit).
        committed_at: Its committer date, ISO 8601 with offset (``%cI``).
        last_touch_sha: The newest commit touching either R6 file; the lock itself unless an
            honest later commit named the lock INVALIDATED. The rule the run applies was frozen
            there.
        last_touch_committed_at: That commit's committer date (``%cI``).
    """

    sha: str
    committed_at: str
    last_touch_sha: str
    last_touch_committed_at: str


class GitCheckError(Exception):
    """A git command the binding run depends on failed. The run refuses (fail closed, D-09)."""


class RuleLockMissingError(Exception):
    """The repository holds no commit that added the R6 record module."""


def _git(repo_root: Path, *args: str) -> str:
    """The stdout of ``git -C <repo_root> <args>``; any failure raises :class:`GitCheckError`."""
    command = ["git", "-C", str(repo_root), *args]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        stderr = getattr(error, "stderr", None) or str(error)
        msg = f"`git {' '.join(args)}` failed in {Path(repo_root).as_posix()}: {stderr.strip()}"
        raise GitCheckError(msg) from error
    return completed.stdout


def lock_commit(repo_root: Path) -> RuleLock:
    """The R6 lock commit, its committer date and the last commit touching either R6 file.

    Raises:
        RuleLockMissingError: no commit added the record module, or the repository is a shallow
            clone, where the boundary commit reads as adding every file.
        GitCheckError: a git call failed (never a default answer).
    """
    if _git(repo_root, "rev-parse", "--is-shallow-repository").strip() == "true":
        msg = (
            f"{Path(repo_root).as_posix()} is a shallow clone, so the oldest commit adding "
            f"{RULE_MODULE_PATH} cannot be proven"
        )
        raise RuleLockMissingError(msg)
    added = _git(
        repo_root,
        "log",
        "--diff-filter=A",
        "--format=%H%x09%cI",
        "--",
        RULE_MODULE_PATH,
    ).strip()
    if not added:
        msg = f"no commit in {Path(repo_root).as_posix()} added {RULE_MODULE_PATH}"
        raise RuleLockMissingError(msg)
    # git lists newest first, so the last line is the oldest commit that added the module.
    sha, committed_at = added.splitlines()[-1].split("\t")
    last_touch = _git(
        repo_root, "log", "-1", "--format=%H%x09%cI", "--", *RULE_PATHS
    ).strip()
    last_touch_sha, last_touch_committed_at = last_touch.split("	")
    return RuleLock(
        sha=sha,
        committed_at=committed_at,
        last_touch_sha=last_touch_sha,
        last_touch_committed_at=last_touch_committed_at,
    )


def unwitnessed_lock_problem(repo_root: Path, lock: RuleLock) -> str | None:
    """Why *lock* is not the lock that was witnessed, or None when it is.

    ``git log --diff-filter=A`` names whatever commit history now says added the rule, so a
    rewritten or squashed history that re-adds it would hand back a different, later "lock".
    The lock must be the witnessed commit, committed before the pre-registration deadline, and
    touch exactly the two R6 files.

    Raises:
        GitCheckError: a git call failed.
    """
    if lock.sha != WITNESSED_LOCK_SHA:
        return (
            f"the oldest commit adding {RULE_MODULE_PATH} is {lock.sha}, not the witnessed lock "
            f"{WITNESSED_LOCK_SHA}"
        )
    committed = datetime.fromisoformat(lock.committed_at)
    if committed >= datetime.fromisoformat(rule.PREREGISTRATION_DEADLINE_UTC):
        return (
            f"the lock commit {lock.sha} was committed at {lock.committed_at}, not before the "
            f"pre-registration deadline {rule.PREREGISTRATION_DEADLINE_UTC}"
        )
    touched = set(_git(repo_root, "show", "--name-only", "--format=", lock.sha).split())
    if touched != set(RULE_PATHS):
        return (
            f"the lock commit {lock.sha} touches {sorted(touched)}, not exactly the two R6 "
            f"files {sorted(RULE_PATHS)}"
        )
    return None


def quiet_rule_edits(repo_root: Path, lock_sha: str) -> list[tuple[str, str]]:
    """Changes to the R6 files the binding run must not consume under the original lock.

    The query of ``tests/unit/test_phase33_preregistration_ancestry.py``'s
    ``uninvalidating_commits_touching_the_rule``, reimplemented here because that helper lives in
    a test module that imports pytest: a commit after *lock_sha* touching either R6 file is
    honest only when its message says "invalidat" (any case) and names the lock sha (full or its
    first 12 characters).

    Returns:
        ``(sha, subject)`` per offending commit, newest first, then ``("uncommitted", path)`` per
        R6 file with an uncommitted change. Empty is the good case.

    Raises:
        GitCheckError: a git call failed; an empty list always means git answered.
    """
    listing = _git(
        repo_root,
        "log",
        "--format=%H%x00%B%x1e",
        f"{lock_sha}..HEAD",
        "--",
        *RULE_PATHS,
    )
    offenders: list[tuple[str, str]] = []
    for record in listing.split("\x1e"):
        if not record.strip():
            continue
        sha, _, message = record.strip().partition("\x00")
        honest = "invalidat" in message.lower() and (
            lock_sha in message or lock_sha[:12] in message
        )
        if not honest:
            subject = message.strip().splitlines()[0] if message.strip() else ""
            offenders.append((sha, subject))
    status = _git(repo_root, "status", "--porcelain", "--", *RULE_PATHS)
    offenders.extend(
        ("uncommitted", line[3:].strip())
        for line in status.splitlines()
        if line.strip()
    )
    return offenders


def refit_section(ledger_dir: Path, silver_dir: Path) -> tuple[int, str]:
    """The re-fit comparison command's exit code and its stdout, captured unaltered (D-25)."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        code = compare_refit_arms.main(
            ["--ledger-dir", str(ledger_dir), "--silver-dir", str(silver_dir)]
        )
    return code, captured.getvalue()


def _first_five(names: Sequence[str]) -> str:
    """Up to five names, comma-separated, with how many more there are."""
    shown = ", ".join(names[:5])
    return shown if len(names) <= 5 else f"{shown} (and {len(names) - 5} more)"


def ledger_refusal(entries: Sequence[LedgerEntry], repo_root: Path) -> str | None:
    """Refusal (0): the ledger read is the published one, as a plain sentence, or None.

    A chain that verifies proves no entry was edited, not that none was cut off: an absent store
    reads as ``[]`` and a stale backup is a shorter prefix, and both verify. So the ledger must
    hold an entry and extend the repository's ``ledger-anchor`` head, checked by
    ``assert_extends_anchor``, the same check the daily run's settle pass applies.
    """
    if not entries:
        return (
            "the ledger holds no entry; the binding run never judges an absent or emptied "
            "ledger. Restore it from the private backup (RUNBOOK) and run it again."
        )
    try:
        anchored = read_local_anchor(repo_root)
    except (AnchorFormatError, GitCommandError) as error:
        return f"a git check failed, so the run refuses (fail closed): {error}"
    if anchored is None:
        return (
            f"{Path(repo_root).as_posix()} holds no ledger-anchor commit while the ledger holds "
            f"{len(entries)} entries, so the ledger cannot be checked against its published head."
        )
    try:
        assert_extends_anchor(entries, anchored)
    except LedgerChainBrokenError as error:
        return f"the ledger does not extend its published anchor: {error}"
    return None


def binding_refusal(
    *,
    counted: pd.DataFrame,
    schedule: pd.DataFrame,
    scores: pd.DataFrame,
    report: CoverageReport,
    scope: VerdictScope,
    repo_root: Path,
    comparison: Callable[[], tuple[int, str]],
) -> str | None:
    """The first unmet binding condition as a plain sentence, or None when every one holds.

    Evaluated in the documented order (module docstring), before anything is computed or
    written. *comparison* runs only when conditions (1)-(7) hold.
    """
    played = set(realized_values_from_scores(scores)["ats"])

    # (1) A counted bet of a played game is ungraded (suppressed rows and void bets never block).
    if not counted.empty:
        live = counted.loc[counted["status"] == BET_STATUS_LIVE]
        ungraded = live.loc[
            (live["grading_status"] == GRADING_STATUS_PENDING)
            & _on_games(live, frozenset(played))
        ]
        if not ungraded.empty:
            keys = [
                f"{game_id} {target}"
                for game_id, target in zip(
                    ungraded["game_id"], ungraded["target"], strict=True
                )
            ]
            return (
                f"{len(keys)} counted bet(s) of played games are not graded yet: "
                f"{_first_five(keys)}. Wait for the next 17:00 ET daily run to grade them."
            )

    # (2) An in-scope game that is not 'not played' lacks a final score, or a not-played
    # game carries a graded counted bet (its score existed; silver lost it).
    unscored = [
        game.game_id
        for game in report.games
        if game.state != STATE_NOT_PLAYED and game.game_id not in played
    ]
    if unscored:
        return (
            f"{len(unscored)} in-scope game(s) lack a final score: {_first_five(unscored)}. "
            "Wait for the next 17:00 ET daily run to record the scores."
        )
    if not counted.empty:
        live = counted.loc[counted["status"] == BET_STATUS_LIVE]
        graded_void = live.loc[
            live["grading_status"].isin(_GRADED_STATUSES)
            & _on_games(live, not_played_game_ids(report))
        ]
        contradicted = sorted({str(game_id) for game_id in graded_void["game_id"]})
        if contradicted:
            return (
                f"{len(contradicted)} game(s) inferred not played carry a graded counted bet: "
                f"{_first_five(contradicted)}. A grade means a final score existed, so silver "
                "lost it; restore the schedule's score instead of waiting (voiding it would drop "
                "a settled result from the verdict)."
            )

    # (3) The schedule holds no played game of the scope's last week.
    final_week = schedule.loc[
        (schedule["season"] == scope.season) & (schedule["week"] == scope.end_week)
    ]
    if not any(str(game_id) in played for game_id in final_week["game_id"]):
        return (
            f"the schedule holds no played week-{scope.end_week} game; the binding run waits "
            "for the Super Bowl's final score."
        )

    # (4) The one-shot rule: the record exists, or it ever existed in git history.
    record = repo_root / rule.BINDING_RECORD_PATH
    if record.exists():
        return f"the binding record {record.as_posix()} already exists; the binding run happened."
    try:
        history = _git(
            repo_root, "log", "--all", "--format=%H", "--", rule.BINDING_RECORD_PATH
        ).strip()
    except GitCheckError as error:
        return f"a git check failed, so the run refuses (fail closed): {error}"
    if history:
        first = history.splitlines()[-1]
        return (
            f"{rule.BINDING_RECORD_PATH} appears in git history (first in {first}); the binding "
            "run happened, and deleting or moving the record never reopens it."
        )

    # (5) The R6 lock commit must be in this repository, and be the witnessed one.
    try:
        lock = lock_commit(repo_root)
        unwitnessed = unwitnessed_lock_problem(repo_root, lock)
    except GitCheckError as error:
        return f"a git check failed, so the run refuses (fail closed): {error}"
    except RuleLockMissingError as error:
        return f"the R6 lock commit cannot be found: {error}."
    if unwitnessed is not None:
        return f"the R6 lock commit cannot be proven: {unwitnessed}."

    # (6) No quiet edit and no uncommitted change to either R6 file.
    try:
        edits = quiet_rule_edits(repo_root, lock.sha)
    except GitCheckError as error:
        return f"a git check failed, so the run refuses (fail closed): {error}"
    if edits:
        named = "; ".join(f"{sha} {subject}" for sha, subject in edits)
        return f"the rule moved after its lock {lock.sha} without naming it INVALIDATED: {named}."

    # (7) Every frozen value still holds its pinned number.
    for name, pin in FROZEN_VALUE_PINS.items():
        held = getattr(rule, name)
        if held != pin:
            return f"the frozen value {name} is pinned at {pin} but now holds {held}."

    # (8) The re-fit comparison must run.
    code, text = comparison()
    if code != 0:
        first_line = text.splitlines()[0] if text.strip() else "(no output)"
        return f"the re-fit comparison could not run (exit {code}): {first_line}"
    return None


# ---------------------------------------------------------------------------
# The binding document (SPEC R8, D-09)
# ---------------------------------------------------------------------------


def _probability(value: float | None) -> str:
    """A p-value at six significant digits, or ``none``."""
    return "none" if value is None else f"{value:.6g}"


def _counted_word(counted: bool) -> str:
    return "counted" if counted else "not counted"


def weeks_graded_at(counted: pd.DataFrame, committed_at: str) -> list[int]:
    """The weeks with a counted bet graded at or before *committed_at* (an ISO 8601 instant)."""
    if counted.empty:
        return []
    cutoff = pd.Timestamp(datetime.fromisoformat(committed_at)).tz_convert(UTC)
    bets = counted.loc[counted["status"] == BET_STATUS_LIVE]
    graded_at = pd.to_datetime(bets["graded_at"], utc=True)
    weeks = bets.loc[graded_at.notna() & (graded_at <= cutoff), "week"]
    return sorted({int(week) for week in weeks})


def _coverage_section(report: CoverageReport, void: pd.DataFrame) -> list[str]:
    """Section 1: the missed-game count, the per-week table and the named exceptions."""
    header = "| Week | Scheduled | Decided | Bet | No bet | Failed or refused | Missed | Not played |"
    lines = [
        "## 1. Coverage",
        "",
        f"Missed games: {report.missed_total}",
        "",
        header,
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for week in report.weeks:
        lines.append(
            f"| {week.week} | {week.scheduled} | {week.decided} | {week.bet} | {week.no_bet} "
            f"| {week.failed_or_refused} | {week.missed} | {week.not_played} |"
        )
    totals = [
        sum(getattr(week, field) for week in report.weeks)
        for field in (
            "scheduled",
            "decided",
            "bet",
            "no_bet",
            "failed_or_refused",
            "missed",
            "not_played",
        )
    ]
    lines.append("| Total | " + " | ".join(str(total) for total in totals) + " |")
    lines += ["", "Partial games (rows for fewer than the three targets):"]
    lines += [
        f"- {game.game_id} (targets: {', '.join(game.targets)})"
        for game in report.partial
    ] or ["- none"]
    lines += ["", "Games not played (each inferred, with its reason):"]
    lines += [
        f"- {game.game_id} (week {game.week}): {game.reason}"
        for game in report.not_played
    ] or ["- none"]
    lines += ["", "Void bets (D-20: in no bet count, no ROI and no test):"]
    lines += [
        f"- {game_id} {target}"
        for game_id, target in zip(void["game_id"], void["target"], strict=True)
    ] or ["- none"]
    return lines


def _scope_section(scope: VerdictScope) -> list[str]:
    """Section 2: the committed verdict scope, as read."""
    return [
        "## 2. Verdict scope",
        "",
        f"- Season: {scope.season}",
        f"- Weeks: {scope.start_week}-{scope.end_week}",
        f"- Counted arm: {scope.counted_arm}",
        f"- Playoff weeks: {_counted_word(scope.includes_playoff_weeks)} (D40-08)",
        f"- Neutral-site games: {_counted_word(scope.includes_neutral_site_games)}",
        f"- Outcome rule: {scope.outcome_rule}",
        f"- Fill convention: {scope.fill_convention_id}",
        f"- Source: {rule.VERDICT_SCOPE_MODULE.replace('.', '/')}.py",
    ]


def _verdict_section(results: Sequence[TargetResult]) -> list[str]:
    """Section 3: the per-target table, each word's 2026 meaning, and the rounding note."""
    lines = [
        "## 3. The verdict per target",
        "",
        f"| Target | Bets | Wins | Losses | Pushes | Units | Flat ROI | Raw p "
        f"| Adjusted p (BH, m = {rule.BH_FAMILY_SIZE}) | Verdict |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        fig = result.figures
        lines.append(
            f"| {fig.target} | {fig.bets} | {fig.wins} | {fig.losses} | {fig.pushes} "
            f"| {fig.units:.6f} | {_number(fig.flat_roi)} | {_probability(result.raw_p)} "
            f"| {_probability(result.adjusted_p)} | {result.token} |"
        )
    lines += ["", "What each verdict means in 2026:", ""]
    lines += [
        f"- {result.figures.target}, {result.token}: {result.meaning}"
        for result in results
    ]
    absent = [result for result in results if result.raw_p is None]
    if absent:
        lines += ["", "Targets without a p-value:", ""]
        lines += [
            f"- {result.figures.target}: {result.p_absent_reason}" for result in absent
        ]
    lines += [
        "",
        "Every decision used the unrounded values; this table is rounded for display only "
        "(6 decimals, p-values at 6 significant digits).",
    ]
    return lines


def _weeks(weeks: Sequence[int]) -> str:
    return ", ".join(str(week) for week in weeks) or "none"


def _lock_section(
    lock: RuleLock, graded_weeks: Sequence[int], frozen_weeks: Sequence[int]
) -> list[str]:
    """Section 4: the R6 lock commit, its time and the weeks already graded then.

    After an honest INVALIDATES commit the rule applied was frozen at that later commit, so its
    instant and the weeks graded by then are reported too: what was visible when the APPLIED
    rule was frozen is the fact this section exists to disclose.
    """
    weeks = _weeks(graded_weeks)
    lines = [
        "## 4. The rule's lock",
        "",
        f"- Lock commit: {lock.sha}",
        f"- Committed at: {lock.committed_at}",
        f"- Weeks with a counted bet graded at or before the lock: {weeks}",
    ]
    if lock.last_touch_sha != lock.sha:
        lines.append(
            f"- A later commit touches the rule and names this lock INVALIDATED: "
            f"{lock.last_touch_sha}"
        )
        lines += [
            f"- The applied rule was frozen at: {lock.last_touch_sha}, committed at "
            f"{lock.last_touch_committed_at}",
            "- Weeks with a counted bet graded at or before the applied rule was frozen: "
            f"{_weeks(frozen_weeks)}",
        ]
    return lines


def _refit_section_lines(refit_text: str) -> list[str]:
    """Section 5: the comparison command's stdout inside one fenced block, unaltered."""
    body = refit_text if refit_text.endswith("\n") else refit_text + "\n"
    return [
        "## 5. Re-fit comparison (ROLLING-REFIT-PREREGISTRATION.md section 11)",
        "",
        "```text\n" + body + "```",
    ]


def _read_section(
    ledger: tuple[str, int], evidence: Sequence[EvidenceFile], silver_dir: Path
) -> list[str]:
    """Section 6: the ledger chain head and entry count, the evidence files, the silver store."""
    head, entry_count = ledger
    lines = [
        "## 6. What this readout read",
        "",
        f"- Ledger chain head: {head}",
        f"- Ledger entries: {entry_count}",
    ]
    for item in evidence:
        if item.missing:
            lines.append(f"- Evidence {item.path}: missing")
        else:
            lines.append(
                f"- Evidence {item.path}: {item.lines} lines, sha256 {item.sha256}"
            )
    lines.append(f"- Silver directory: {silver_dir.as_posix()}")
    return lines


def render_document(
    *,
    report: CoverageReport,
    void: pd.DataFrame,
    scope: VerdictScope,
    results: Sequence[TargetResult],
    lock: RuleLock,
    graded_weeks: Sequence[int],
    frozen_weeks: Sequence[int],
    refit_text: str,
    ledger: tuple[str, int],
    evidence: Sequence[EvidenceFile],
    silver_dir: Path,
    written_at: datetime,
) -> str:
    """The one binding document: ASCII, LF line endings, the six sections in order."""
    sections = [
        _coverage_section(report, void),
        _scope_section(scope),
        _verdict_section(results),
        _lock_section(lock, graded_weeks, frozen_weeks),
        _refit_section_lines(refit_text),
        _read_section(ledger, evidence, silver_dir),
    ]
    lines = [
        "# The 2026 season-close readout (binding)",
        "",
        f"Written once by `{BINDING_COMMAND}` at "
        f"{written_at.astimezone(UTC).isoformat(timespec='seconds')} (UTC), under the rule "
        "frozen in SEASON-VERDICT-PREREGISTRATION.md.",
    ]
    for section in sections:
        lines += ["", *section]
    return "\n".join(lines) + "\n"


def _write_text(text: str, path: Path) -> None:
    """Write *text* as ASCII bytes (LF kept) at *path*."""
    path.write_bytes(text.encode("ascii"))


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "The 2026 season-close readout: per-game coverage of the verdict scope and the "
            f"season-to-date figures (preview: {PREVIEW_LABEL})"
        )
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=LEDGER_DIR,
        help=f"The ledger directory (default: {LEDGER_DIR.as_posix()})",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=DEFAULT_SILVER_DIR,
        help=f"The silver directory holding games.parquet (default: {DEFAULT_SILVER_DIR.as_posix()})",
    )
    parser.add_argument(
        "--run-log",
        type=Path,
        default=DEFAULT_RUN_LOG,
        help=f"The daily-run record (default: {DEFAULT_RUN_LOG.as_posix()})",
    )
    parser.add_argument(
        "--skip-records",
        type=Path,
        default=SKIP_RECORD_PATH,
        help=f"The skip record (default: {SKIP_RECORD_PATH.as_posix()})",
    )
    parser.add_argument(
        "--binding",
        action="store_true",
        help=(
            "The ONE binding run, after week 22 is graded: refuses until every condition holds, "
            f"then writes <repo root>/{rule.BINDING_RECORD_PATH}"
        ),
    )
    return parser


def read_refusal_evidence(
    run_log: Path, skip_records: Path
) -> tuple[tuple[EvidenceFile, EvidenceFile], dict[str, frozenset[str]]]:
    """Both evidence files, each read ONCE: their provenance and the games they name.

    Raises:
        EvidenceUnreadableError: the run log cannot be read or a line is malformed.
        pipeline.skip_log.SkipRecordCorrupt: a skip-record line is malformed.
    """
    run_file, run_text = read_evidence(run_log)
    skip_file, skip_text = read_evidence(skip_records)
    skips = (
        [] if skip_text is None else parse_skip_records(skip_text, source=skip_records)
    )
    refused = refused_game_ids(parse_run_log(run_text, run_log), skips)
    return (run_file, skip_file), refused


def _counted_label(counted: bool) -> str:
    return "counted" if counted else "excluded"


def scope_line(scope: VerdictScope) -> str:
    """The declared scope, as read."""
    return (
        f"SCOPE= season={scope.season} weeks={scope.start_week}-{scope.end_week} "
        f"arm={scope.counted_arm} playoffs={_counted_label(scope.includes_playoff_weeks)} "
        f"neutral_sites={_counted_label(scope.includes_neutral_site_games)} "
        f"outcome={scope.outcome_rule} fill={scope.fill_convention_id}"
    )


def evidence_line(evidence: EvidenceFile) -> str:
    """One evidence file's provenance, or that it is missing (D-08)."""
    if evidence.missing:
        return f"EVIDENCE= path={evidence.path} missing"
    return f"EVIDENCE= path={evidence.path} lines={evidence.lines} sha256={evidence.sha256}"


def _coverage_values(week: WeekCoverage) -> str:
    return " ".join(f"{field}={getattr(week, field)}" for field in _COVERAGE_FIELDS)


def coverage_lines(report: CoverageReport) -> list[str]:
    """The per-week lines, the total, and the named missed, partial and not-played games."""
    lines = [
        f"COVERAGE= week={week.week} {_coverage_values(week)}" for week in report.weeks
    ]
    totals = " ".join(
        f"{field}={sum(getattr(week, field) for week in report.weeks)}"
        for field in _COVERAGE_FIELDS
    )
    lines.append(f"COVERAGE_TOTAL= {totals}")
    lines.append(f"MISSED_GAMES= {report.missed_total}")
    lines.extend(
        f"PARTIAL= {game.game_id} targets={','.join(game.targets)}"
        for game in report.partial
    )
    if not report.partial:
        lines.append("PARTIAL= none")
    lines.extend(
        f"NOT_PLAYED= {game.game_id} week={game.week} reason={game.reason}"
        for game in report.not_played
    )
    if not report.not_played:
        lines.append("NOT_PLAYED= none")
    return lines


def _number(value: float | None) -> str:
    """A figure at six decimals, or ``none`` when it does not exist."""
    return "none" if value is None else f"{value:.6f}"


def figure_lines(counted: pd.DataFrame, report: CoverageReport) -> list[str]:
    """The void bets, then one ``FIGURES=`` line per target -- raw figures, no test."""
    void = void_bets(counted, report)
    lines = [
        f"VOID_BETS= {game_id} target={target}"
        for game_id, target in zip(void["game_id"], void["target"], strict=True)
    ]
    if not lines:
        lines.append("VOID_BETS= none")
    void_game_ids = not_played_game_ids(report)
    lines.extend(
        f"FIGURES= target={fig.target} counted_rows={fig.counted_rows} bets={fig.bets} "
        f"graded={fig.graded} wins={fig.wins} losses={fig.losses} pushes={fig.pushes} "
        f"pending={fig.pending} units={fig.units:.6f} flat_roi={_number(fig.flat_roi)}"
        for fig in target_figures(counted, void_game_ids)
    )
    return lines


def _binding_run(
    args: argparse.Namespace,
    *,
    entries: Sequence[LedgerEntry],
    evidence: Sequence[EvidenceFile],
    counted: pd.DataFrame,
    schedule: pd.DataFrame,
    scores: pd.DataFrame,
    report: CoverageReport,
    scope: VerdictScope,
) -> int:
    """The one binding run: refuse (exit 3) or write the document (exit 0)."""
    repo_root = REPO_ROOT
    unpublished = ledger_refusal(entries, repo_root)
    if unpublished is not None:
        print(f"BINDING_REFUSED= {unpublished}")
        return 3
    # Run at most once: the refusal's last check and the document read the same capture.
    comparison = functools.cache(
        lambda: refit_section(args.ledger_dir, args.silver_dir)
    )
    refusal = binding_refusal(
        counted=counted,
        schedule=schedule,
        scores=scores,
        report=report,
        scope=scope,
        repo_root=repo_root,
        comparison=comparison,
    )
    if refusal is not None:
        print(f"BINDING_REFUSED= {refusal}")
        return 3

    _code, refit_text = comparison()
    void = void_bets(counted, report)
    results = measure_targets(counted, not_played_game_ids(report))
    lock = lock_commit(repo_root)
    text = render_document(
        report=report,
        void=void,
        scope=scope,
        results=results,
        lock=lock,
        graded_weeks=weeks_graded_at(counted, lock.committed_at),
        frozen_weeks=weeks_graded_at(counted, lock.last_touch_committed_at),
        refit_text=refit_text,
        ledger=ledger_head(entries),
        evidence=evidence,
        silver_dir=Path(args.silver_dir),
        written_at=datetime.now(UTC),
    )
    record = repo_root / rule.BINDING_RECORD_PATH
    record.parent.mkdir(parents=True, exist_ok=True)
    _replace_atomically(functools.partial(_write_text, text), record)
    print(f"BINDING_WRITTEN= {record.as_posix()}")
    for result in results:
        print(f"VERDICT_{result.figures.target.upper()}= {result.token}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    if args.binding:
        for store in ("ledger_dir", "silver_dir", "run_log", "skip_records"):
            path = Path(getattr(args, store))
            setattr(args, store, path if path.is_absolute() else REPO_ROOT / path)
    try:
        entries = read_entries(args.ledger_dir)
    except LedgerFormatError as error:
        print(f"LEDGER_UNREADABLE= {error}")
        return 2
    verdict = verify_chain(entries)
    if not verdict.ok:
        print(
            f"LEDGER_UNREADABLE= the chain does not verify at seq {verdict.first_broken_seq} "
            f"(key {verdict.first_broken_key}): {verdict.reason}"
        )
        return 2
    scope = load_verdict_scope()
    counted = verdict_rows(entries, scope)
    try:
        schedule = load_recorded_schedule(args.silver_dir / SILVER_GAMES_FILENAME)
        scores = load_silver_games(args.silver_dir)
        evidence, refused = read_refusal_evidence(args.run_log, args.skip_records)
        # The binding run passes no instant: there, nothing is ever "not yet due".
        now = None if args.binding else datetime.now(UTC)
        report = classify_coverage(schedule, scores, counted, refused, scope, now=now)
    except _STORE_ERRORS as error:
        print(f"STORE_UNREADABLE= {type(error).__name__}: {error}")
        return 2

    if args.binding:
        return _binding_run(
            args,
            entries=entries,
            evidence=evidence,
            counted=counted,
            schedule=schedule,
            scores=scores,
            report=report,
            scope=scope,
        )

    head, entry_count = ledger_head(entries)
    print(f"READOUT= preview: {PREVIEW_LABEL}")
    print(scope_line(scope))
    print(f"LEDGER= head={head} entries={entry_count}")
    for evidence_file in evidence:
        print(evidence_line(evidence_file))
    for line in [*coverage_lines(report), *figure_lines(counted, report)]:
        print(line)
    print(REFIT_SECTION_LINE)
    return 0


if __name__ == "__main__":
    sys.exit(main())
