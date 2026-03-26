"""Populate the DuckDB web cache from artifacts and backtest outputs.

CLI entry point for building the web_cache.duckdb file that the
FastAPI API serves data from. This script reads model artifacts,
backtest outputs, and gold data to build a read-optimized cache.

Usage:
    python scripts/populate_cache.py
    python scripts/populate_cache.py --db-path data/custom_cache.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

from api.cache import populate_cache


def main() -> None:
    """Run cache population with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Populate DuckDB web cache from artifacts and backtest outputs."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("data/web_cache.duckdb"),
        help="Output path for the DuckDB cache file (default: data/web_cache.duckdb)",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root artifacts directory (default: artifacts/)",
    )
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("outputs/backtest"),
        help="Backtest outputs directory (default: outputs/backtest/)",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/gold"),
        help="Gold data directory (default: data/gold/)",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("data/silver"),
        help="Silver data directory (default: data/silver/)",
    )
    args = parser.parse_args()

    populate_cache(
        db_path=args.db_path,
        artifacts_dir=args.artifacts_dir,
        outputs_dir=args.outputs_dir,
        gold_dir=args.gold_dir,
        silver_dir=args.silver_dir,
    )


if __name__ == "__main__":
    main()
