#!/usr/bin/env python3
"""The 2026 season-close readout (Phase 38; SPEC R7-R9).

Accounts for every scheduled game of the verdict scope (season 2026, weeks 5-22, read from the
committed declaration through ``forward_ledger.declarations.load_verdict_scope``) and reports the
counted live-arm rows. It runs any time as a PREVIEW, labelled "season to date -- not the
verdict"; the one binding run happens after the Super Bowl is graded (Plan 38-09).

Output is ``KEY= value`` lines on stdout:

* ``READOUT= preview: season to date -- not the verdict``;
* one ``COVERAGE= week=<w> scheduled=<n> decided=<n> bet=<n> no_bet=<n> missed=<n>`` line per
  scope week, in week order (the per-game rule is ``scripts.season_coverage.classify_coverage``).

The ledger is read through its chain check (``forward_ledger.store.verify_chain``): nothing is
reported over a ledger whose chain does not verify. The readout writes no file anywhere.

No module under ``api/`` may import this CLI; nothing on the daily run imports it either.

Exit codes: 0 the readout was printed; 2 a store cannot be read -- the ledger is unreadable or its
chain does not verify (``LEDGER_UNREADABLE=``), or the silver schedule cannot be read
(``STORE_UNREADABLE=``, naming the exception class).

Usage:
    uv run python -m scripts.season_readout
    uv run python -m scripts.season_readout --ledger-dir path/to/ledger --silver-dir data/silver

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from forward_ledger.declarations import load_verdict_scope
from forward_ledger.settle import (
    DEFAULT_SILVER_DIR,
    SILVER_GAMES_FILENAME,
    load_silver_games,
)
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerFormatError,
    read_entries,
    verify_chain,
)
from forward_ledger.verdict import verdict_rows
from scripts.season_coverage import CoverageReport, classify_coverage
from utils.current_slate import (
    AmbiguousSlateError,
    ScheduleUnavailableError,
    load_recorded_schedule,
)
from utils.game_lock import MissingKickoffError

PREVIEW_LABEL: str = "season to date -- not the verdict"


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
    return parser


def coverage_lines(report: CoverageReport) -> list[str]:
    """One ``COVERAGE=`` line per scope week, in week order."""
    return [
        f"COVERAGE= week={week.week} scheduled={week.scheduled} decided={week.decided} "
        f"bet={week.bet} no_bet={week.no_bet} missed={week.missed}"
        for week in report.weeks
    ]


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
    except (
        FileNotFoundError,
        ScheduleUnavailableError,
        MissingKickoffError,
        AmbiguousSlateError,
    ) as error:
        print(f"STORE_UNREADABLE= {type(error).__name__}: {error}")
        return 2

    report = classify_coverage(schedule, scores, counted, scope)
    print(f"READOUT= preview: {PREVIEW_LABEL}")
    for line in coverage_lines(report):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
