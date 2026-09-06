"""Populate the DuckDB web cache from artifacts and backtest outputs.

CLI entry point for building the web_cache.duckdb file that the
FastAPI API serves data from. This script reads model artifacts,
backtest outputs, and gold data to build a read-optimized cache.

This is also the RECOVERY command the /bets stale-cache refusal names, so it must build a
cache the page can serve: it reads the same bet-list cache sources the Friday orchestrator's
populate_web_cache step reads (the durable outputs/bet_list/ artifacts plus the schedule-derived
per-game freeze), through the one shared reader, so a hand-run rebuild and a scheduled one can
never produce different tables.

Usage:
    python scripts/populate_cache.py
    python scripts/populate_cache.py --db-path data/custom_cache.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

from api.cache import populate_cache
from backtest.weekly_bet_list import (
    DEFAULT_BET_LIST_DIR,
    read_bet_list_cache_sources,
)


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
    parser.add_argument(
        "--bet-list-dir",
        type=Path,
        default=DEFAULT_BET_LIST_DIR,
        help=(
            "Directory holding the durable bet-list artifacts the cache loads "
            f"(default: {DEFAULT_BET_LIST_DIR.as_posix()}/)"
        ),
    )
    args = parser.parse_args()

    # Read the bet-list cache sources HERE and pass frames: api/cache.py may import no backtest
    # module (UIAP-01), so it cannot know these filenames or derive a per-game freeze instant.
    sources = read_bet_list_cache_sources(args.bet_list_dir, args.silver_dir)

    populate_cache(
        db_path=args.db_path,
        artifacts_dir=args.artifacts_dir,
        outputs_dir=args.outputs_dir,
        gold_dir=args.gold_dir,
        silver_dir=args.silver_dir,
        bet_list_df=sources.bet_list,
        bet_tracker_df=sources.tracker,
        bet_schedule_df=sources.schedule,
    )


if __name__ == "__main__":
    main()
