"""Run Study 1a and write its report into ``outputs/study_1a/``.

Replays the current recipe (live gold) and v2 (``outputs/gold_v2``) walk-forward to the run
instant, judges the install bar and each target's earned market weight
(``backtest.study_1a``), and writes ``study_1a.json`` and ``report.md``. Reads the live lake;
writes nothing under ``data/`` or ``artifacts/``. Do not run it between 17:00 and 17:20 ET
(it can block the daily run's DuckDB writer).

Usage:
    uv run python -m scripts.run_study_1a [--v2-gold-dir outputs/gold_v2]
        [--out-dir outputs/study_1a] [--artifacts-dir <checkout>/artifacts]

ASCII only, no emoji.
"""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from backtest.study_1a import (
    DEFAULT_OUT_DIR,
    TARGETS,
    VARIANTS,
    refuse_data_dir,
    run_study_1a,
    summary_lines,
    write_report,
)

DEFAULT_V2_GOLD_DIR = Path("outputs/gold_v2")
SILVER_DIR = Path("data/silver")


def build_parser() -> argparse.ArgumentParser:
    """The CLI: where the v2 gold is read from and where the report goes."""
    parser = argparse.ArgumentParser(
        description="Study 1a: replay current vs v2, judge the install and the weights."
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
    refuse_data_dir(args.out_dir)  # before the long replay, not after it
    cutoff = datetime.now(UTC)
    result = run_study_1a(
        args.v2_gold_dir, SILVER_DIR, cutoff, artifacts_dir=args.artifacts_dir
    )
    write_report(result, args.out_dir)
    for line in summary_lines(result):
        print(line)
    for target in TARGETS:
        pooled = ", ".join(
            f"{variant} {result.scores[target][variant].losses.pooled:.4f}"
            for variant in VARIANTS
        )
        print(f"pooled 2017-2024 {target}: {pooled}")
    print(f"wrote {args.out_dir.as_posix()}/report.md and study_1a.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
