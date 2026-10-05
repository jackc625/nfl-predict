#!/usr/bin/env python3
"""Move the old writer's 2026 forward rows into the ledger, as stored (Phase 34, D-14). Owner-run.

A DRY RUN BY DEFAULT: it reads ``outputs/bet_list/bet_list.parquet`` and the ledger and prints what
``--apply`` would do, writing nothing. ``--apply`` writes every 2026 forward row into an EMPTY
ledger -- stored order, chained from the published genesis, ``arm='live'``, every stamp NULL,
``pre_verdict``, ``bootstrap_regime`` on weeks 2-4 -- verifies it, and only then rewrites the
outputs pair with the 2025 ``backtest_replay`` rows alone. ``--finish`` completes an interrupted
outputs rewrite after proving the ledger holds exactly the stored forward rows.
``forward_ledger.migration`` holds the rules.

RUN ONCE, BY HAND, IN THE GO-LIVE WINDOW (Plan 34-19), outside the daily task's 16:55-18:00 ET
window. The order after ``--apply``: sync and verify the ledger (``scripts/sync_ledger.py``,
``scripts/verify_ledger.py``), and only then flip ``forward_ledger/cutover.py`` -- so the first
anchor and backup are published and verified with the switch still off.

Output is one ``FIELD= value`` line each: ``MIGRATION_MODE=``, ``MIGRATE_ROWS=``, ``WEEKS=``,
``STATUS_COUNTS=``, ``LEDGER_ENTRIES_BEFORE=``, ``HEAD_HASH=``, ``LEDGER_ENTRIES=``,
``OUTPUTS_REPLAY_ROWS=``, ``LEDGER_WRITTEN=``, ``OUTPUTS_REWRITTEN=``, then ``NEXT=`` once the
outputs were rewritten. A refusal prints ``MIGRATION_REFUSED=`` naming the error.

Exit codes: 0 done (or dry run); 1 a named refusal -- nothing past the refusing step was written.

Usage:
    uv run python -m scripts.migrate_forward_rows
    uv run python -m scripts.migrate_forward_rows --apply
    uv run python -m scripts.migrate_forward_rows --finish
    uv run python -m scripts.migrate_forward_rows --output-dir path --ledger-dir path

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtest.weekly_bet_list import (
    DEFAULT_BET_LIST_DIR,
    DecidedAfterFreezeError,
    MissingDecidedAtError,
)
from forward_ledger.canonical import CanonicalValueError
from forward_ledger.migration import (
    AlreadyMigratedError,
    MigrationMismatchError,
    MigrationReport,
    MigrationStampError,
    migrate_forward_rows,
)
from forward_ledger.run_log import record_event
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerChainBrokenError,
    LedgerEntryRefusedError,
    LedgerFormatError,
    LedgerWriterBusyError,
)

# The run-log seam: tests replace it so nothing is appended to logs/ledger_runs.jsonl.
LOG = record_event

NEXT_STEP = (
    "run scripts/sync_ledger.py and scripts/verify_ledger.py, then flip "
    "forward_ledger/cutover.py before the next decision run (Plan 34-19)"
)

# Every refusal the migration names. Anything else is a defect and keeps its traceback.
_REFUSALS: tuple[type[BaseException], ...] = (
    AlreadyMigratedError,
    MigrationStampError,
    MigrationMismatchError,
    LedgerWriterBusyError,
    LedgerChainBrokenError,
    LedgerEntryRefusedError,
    LedgerFormatError,
    CanonicalValueError,
    DecidedAfterFreezeError,
    MissingDecidedAtError,
)


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Move the old writer's 2026 forward rows into the ledger as stored (dry run by "
            "default)"
        )
    )
    step = parser.add_mutually_exclusive_group()
    step.add_argument(
        "--apply",
        action="store_true",
        help="Write the ledger, verify it, then rewrite the outputs with the replay rows only",
    )
    step.add_argument(
        "--finish",
        action="store_true",
        help="Complete an interrupted outputs rewrite after proving the ledger holds the rows",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BET_LIST_DIR,
        help=(
            "The old writer's bet-list directory (default: "
            f"{DEFAULT_BET_LIST_DIR.as_posix()})"
        ),
    )
    parser.add_argument(
        "--ledger-dir",
        type=Path,
        default=LEDGER_DIR,
        help=f"The ledger directory (default: {LEDGER_DIR.as_posix()})",
    )
    return parser


def report_lines(report: MigrationReport) -> list[str]:
    """One ``FIELD= value`` line per fact of *report*."""
    counts = " ".join(f"{status}={n}" for status, n in report.status_counts.items())
    lines = [
        f"MIGRATION_MODE= {report.mode}",
        f"MIGRATE_ROWS= {report.migrate_rows}",
        f"WEEKS= {','.join(str(week) for week in report.weeks)}",
        f"STATUS_COUNTS= {counts}",
        f"LEDGER_ENTRIES_BEFORE= {report.ledger_entries_before}",
        f"HEAD_HASH= {report.head_hash}",
        f"LEDGER_ENTRIES= {report.entry_count}",
        f"OUTPUTS_REPLAY_ROWS= {report.outputs_replay_rows}",
        f"LEDGER_WRITTEN= {report.ledger_written}",
        f"OUTPUTS_REWRITTEN= {report.outputs_rewritten}",
    ]
    if report.outputs_rewritten:
        lines.append(f"NEXT= {NEXT_STEP}")
    return lines


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        report = migrate_forward_rows(
            args.output_dir,
            args.ledger_dir,
            apply=args.apply,
            finish=args.finish,
            log=LOG,
        )
    except _REFUSALS as refusal:
        print(f"MIGRATION_REFUSED= {type(refusal).__name__}: {refusal}")
        return 1
    for line in report_lines(report):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
