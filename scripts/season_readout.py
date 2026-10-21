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
  (D-26, the reason it was inferred stated), or ``NOT_PLAYED= none``.

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
from datetime import UTC, datetime
from pathlib import Path

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

_STORE_ERRORS: tuple[type[Exception], ...] = (
    FileNotFoundError,
    ScheduleUnavailableError,
    MissingKickoffError,
    AmbiguousSlateError,
    EvidenceUnreadableError,
    SkipRecordCorrupt,
)


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
    for line in coverage_lines(report):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
