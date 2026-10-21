#!/usr/bin/env python3
"""Compare the live arm with the frozen arm, per target and re-fit generation (Phase 37, SPEC R12).

The pre-registered comparison of ROLLING-REFIT-PREREGISTRATION.md section 11, computed by
``refit.comparison`` from the ledger and the silver scores, READ ONLY. The deciding run happens once,
at season close (Phase 38); the command is runnable at any time before then.

Output is ``KEY= value`` lines: ``COMPARISON_SEASON=``, ``UNPAIRED_WEEKS=`` (weeks with live rows but
no shadow row at all, excluded), ``HOLES=`` (live rows with no shadow twin), one ``EXCLUDED_<REASON>=``
count per ``refit.comparison.EXCLUDED_REASONS`` entry, in that order (``EXCLUDED_NO_GENERATION=``,
``EXCLUDED_INVARIANT_FAILURE=``, ``EXCLUDED_NO_MODEL_VALUE=``, ``EXCLUDED_NO_FINAL_SCORE=``: live rows
left out of every test, so no exclusion is silent), then, in this order:

* one ``CELL=`` line per reported (target, generation) cell, in increasing generation order: its
  paired games, both arms' mean loss, the mean gap, the raw and Benjamini-Hochberg adjusted
  p-values, and its result (``better`` / ``worse`` / ``no difference`` / ``too few to test``, or
  ``never tested (generation 0)``);
* one ``VERDICT=`` line per target: the family size and the pre-registered roll-up word;
* one ``POOLED_SECONDARY=`` line per target: the same test over the target's family cells, with
  its own p-value, outside the family -- it never changes the word;
* one ``ROI_SECONDARY=`` line per generation and arm: flat ROI, report-only, with no test.

A gap is live loss minus frozen loss, so a negative gap means the live arm was more accurate. No
line carries CLV.

No module under ``api/`` may import this CLI; nothing on the daily run imports it either.

The ledger is read through its chain check (``forward_ledger.store.verify_chain``), as the cache build
reads it: a verdict is never computed over a ledger whose chain does not verify.

Exit codes: 0 the comparison was computed; 2 a store cannot be read -- the ledger is unreadable or
its chain does not verify (``LEDGER_UNREADABLE=``), or a re-fit record, the silver store or a row's
generation cannot be read (``STORE_UNREADABLE=``, naming the exception class).

Usage:
    uv run python -m scripts.compare_refit_arms
    uv run python -m scripts.compare_refit_arms --ledger-dir path/to/ledger --silver-dir data/silver

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from backtest.refit_preregistration_2026 import ALPHA, REFIT_SEASON
from forward_ledger.settle import DEFAULT_SILVER_DIR, load_silver_games
from forward_ledger.store import (
    LEDGER_DIR,
    LedgerFormatError,
    read_entries,
    verify_chain,
)
from refit.comparison import (
    EXCLUDED_REASONS,
    STATUS_TESTED,
    ComparisonResult,
    PairedFrame,
    decide,
    paired_frame,
)
from refit.generations import UnknownGenerationError, installed_generations
from refit.records import RefitRecordFormatError


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


def _probability(value: float | None) -> str:
    """A p-value at six significant digits, or ``none`` when no test produced one."""
    return "none" if value is None else f"{value:.6g}"


def report_lines(paired: PairedFrame, result: ComparisonResult) -> list[str]:
    """Every output line, in the pre-registered order (cells, verdicts, pooled, ROI)."""
    weeks = ",".join(str(week) for week in paired.unpaired_weeks)
    lines = [
        f"COMPARISON_SEASON= {REFIT_SEASON}",
        f"UNPAIRED_WEEKS= {weeks or 'none'}",
        f"HOLES= {len(paired.holes)}",
    ]
    lines.extend(
        f"EXCLUDED_{reason.upper()}= {paired.excluded[reason]}"
        for reason in EXCLUDED_REASONS
    )
    for cell in result.cells:
        outcome = cell.result if cell.status == STATUS_TESTED else cell.status
        lines.append(
            f"CELL= {cell.target} gen={cell.generation} n={cell.n} "
            f"live_loss={_number(cell.live_loss)} frozen_loss={_number(cell.frozen_loss)} "
            f"mean_gap={_number(cell.mean_gap)} p={_probability(cell.p)} "
            f"adjusted_p={_probability(cell.adjusted_p)} {outcome}"
        )
    for target, word in result.verdicts.items():
        lines.append(
            f"VERDICT= {target} bh_family_m={result.family_size} alpha={ALPHA} {word}"
        )
    for pooled in result.pooled:
        lines.append(
            f"POOLED_SECONDARY= {pooled.target} n={pooled.n} "
            f"mean_gap={_number(pooled.mean_gap)} p={_probability(pooled.p)} {pooled.status} "
            "(secondary: outside the BH family, never changes the VERDICT)"
        )
    for roi in result.roi:
        lines.append(
            f"ROI_SECONDARY= gen={roi.generation} arm={roi.arm} bets={roi.bets} "
            f"flat_roi={_number(roi.flat_roi)} (report-only, no test)"
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
    try:
        generations = installed_generations(args.ledger_dir)
        paired = paired_frame(entries, load_silver_games(args.silver_dir), generations)
    except (
        RefitRecordFormatError,
        FileNotFoundError,
        UnknownGenerationError,
    ) as error:
        print(f"STORE_UNREADABLE= {type(error).__name__}: {error}")
        return 2
    for line in report_lines(paired, decide(paired, generations)):
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
