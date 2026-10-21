#!/usr/bin/env python3
"""Compare the live arm with the frozen arm, per target and re-fit generation (Phase 37, SPEC R12).

The pre-registered comparison of ROLLING-REFIT-PREREGISTRATION.md section 11, computed by
``refit.comparison`` from the ledger and the silver scores, READ ONLY. The deciding run happens once,
at season close (Phase 38); the command is runnable at any time before then.

Output is ``KEY= value`` lines: ``COMPARISON_SEASON=``, ``UNPAIRED_WEEKS=`` (weeks with live rows but
no shadow row at all, excluded), ``HOLES=`` (live rows with no shadow twin), ``INVARIANT_FAILURES=``
(pairs whose twins disagree on a comparability field, excluded), then one ``CELL=`` line per
(target, generation) with paired games, in increasing generation order. A gap is live loss minus
frozen loss, so a negative gap means the live arm was more accurate. No line carries CLV.

No module under ``api/`` may import this CLI; nothing on the daily run imports it either.

Exit codes: 0 the comparison was computed; 2 a store cannot be read.

Usage:
    uv run python -m scripts.compare_refit_arms
    uv run python -m scripts.compare_refit_arms --ledger-dir path/to/ledger --silver-dir data/silver

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtest.refit_preregistration_2026 import REFIT_SEASON
from forward_ledger.settle import DEFAULT_SILVER_DIR, load_silver_games
from forward_ledger.store import LEDGER_DIR, LedgerFormatError, read_entries
from refit.comparison import cell_gaps, paired_frame, paired_test
from refit.generations import installed_generations


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare the live arm with the frozen arm: paired per-game accuracy per target and "
            "re-fit generation (the pre-registered season-close test)"
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


def _number(value: float | None) -> str:
    """A figure at six decimals, or ``none`` when it does not exist."""
    return "none" if value is None else f"{value:.6f}"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = build_parser().parse_args(argv)
    try:
        entries = read_entries(args.ledger_dir)
    except LedgerFormatError as error:
        print(f"LEDGER_UNREADABLE= {error}")
        return 2
    paired = paired_frame(
        entries,
        load_silver_games(args.silver_dir),
        installed_generations(args.ledger_dir),
    )

    print(f"COMPARISON_SEASON= {REFIT_SEASON}")
    weeks = ",".join(str(week) for week in paired.unpaired_weeks)
    print(f"UNPAIRED_WEEKS= {weeks or 'none'}")
    print(f"HOLES= {len(paired.holes)}")
    print(f"INVARIANT_FAILURES= {paired.excluded['invariant_failure']}")
    for (target, generation), cell in cell_gaps(paired).items():
        test = paired_test(cell["gap"].tolist())
        print(
            f"CELL= {target} gen={generation} n={test['n']} "
            f"live_loss={_number(float(cell['live_loss'].mean()))} "
            f"frozen_loss={_number(float(cell['frozen_loss'].mean()))} "
            f"mean_gap={_number(test['mean_gap'])} {test['status']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
