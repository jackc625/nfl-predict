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
reported over a ledger whose chain does not verify. The readout writes no file anywhere (D-16).

No module under ``api/`` may import this CLI; nothing on the daily run imports it either.

Exit codes: 0 the readout was printed; 2 a store cannot be read -- the ledger is unreadable or its
chain does not verify (``LEDGER_UNREADABLE=``), or the silver schedule, the run log or the skip
records cannot be read (``STORE_UNREADABLE=``, naming the exception class). A malformed evidence
line refuses; it is never read as absent.

Usage:
    uv run python -m scripts.season_readout
    uv run python -m scripts.season_readout --ledger-dir path/to/ledger --silver-dir data/silver

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
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
from forward_ledger.declarations import VerdictScope, load_verdict_scope
from forward_ledger.settle import (
    DEFAULT_SILVER_DIR,
    SILVER_GAMES_FILENAME,
    load_silver_games,
)
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerFormatError,
    ledger_head,
    read_entries,
    verify_chain,
)
from forward_ledger.verdict import verdict_rows
from pipeline.skip_log import SKIP_RECORD_PATH, SkipRecordCorrupt, parse_skip_records
from scripts.season_coverage import (
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


def void_bets(counted: pd.DataFrame, report: CoverageReport) -> pd.DataFrame:
    """The counted bets on not-played games (D-20: void), in the counted frame's order."""
    if counted.empty or not report.not_played:
        return pd.DataFrame(columns=pd.Index(BET_COLUMNS))
    not_played = frozenset(game.game_id for game in report.not_played)
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
    void_game_ids = frozenset(str(game_id) for game_id in void["game_id"])
    lines.extend(
        f"FIGURES= target={fig.target} counted_rows={fig.counted_rows} bets={fig.bets} "
        f"graded={fig.graded} wins={fig.wins} losses={fig.losses} pushes={fig.pushes} "
        f"pending={fig.pending} units={fig.units:.6f} flat_roi={_number(fig.flat_roi)}"
        for fig in target_figures(counted, void_game_ids)
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
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
        report = classify_coverage(
            schedule, scores, counted, refused, scope, now=datetime.now(UTC)
        )
    except _STORE_ERRORS as error:
        print(f"STORE_UNREADABLE= {type(error).__name__}: {error}")
        return 2

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
