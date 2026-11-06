"""Run Study 1b and write its report into ``outputs/study_1b/``.

Scores every 2020-2024 game with an early-week line from its v2 inputs as known at that
capture, tests per target whether the model's disagreement with the early line predicts the
line's move to the nflverse close and to the day-before line (``backtest.study_1b``), and writes
``study_1b.json`` and ``report.md``. Reads the live lake; writes nothing under ``data/`` or
``artifacts/``. Do not run it between 17:00 and 17:20 ET (it can block the daily run's DuckDB
writer).

Usage:
    uv run python -m scripts.run_study_1b [--v2-gold-dir outputs/gold_v2]
        [--out-dir outputs/study_1b] [--artifacts-dir <checkout>/artifacts]

ASCII only, no emoji.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from backtest.study_1a import refuse_data_dir
from backtest.study_1b import (
    DEFAULT_OUT_DIR,
    run_study_1b,
    summary_lines,
    write_report,
)

DEFAULT_V2_GOLD_DIR = Path("outputs/gold_v2")
SILVER_DIR = Path("data/silver")


def build_parser() -> argparse.ArgumentParser:
    """The CLI: where the v2 gold is read from and where the report goes."""
    parser = argparse.ArgumentParser(
        description="Study 1b: does the v2 model predict the early-week line's move?"
    )
    parser.add_argument(
        "--v2-gold-dir",
        type=Path,
        default=DEFAULT_V2_GOLD_DIR,
        help=f"the v2 gold matrices (default {DEFAULT_V2_GOLD_DIR.as_posix()})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"report directory, never under data/ (default {DEFAULT_OUT_DIR.as_posix()})",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=None,
        help="where the WP market converter is read (default: beside the live data lake)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the study to the current instant, write the report and print its summary."""
    args = build_parser().parse_args(argv)
    # Before the long replay, not after it.
    refuse_data_dir(
        args.out_dir, what="the Study 1b report", default_dir=DEFAULT_OUT_DIR
    )
    cutoff = datetime.now(UTC)
    result = run_study_1b(
        args.v2_gold_dir, SILVER_DIR, cutoff, artifacts_dir=args.artifacts_dir
    )
    write_report(result, args.out_dir)
    for line in summary_lines(result):
        print(line)
    print(f"wrote {args.out_dir.as_posix()}/report.md and study_1b.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
