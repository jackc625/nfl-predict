"""Populate the DuckDB web cache from artifacts and backtest outputs.

CLI entry point for building the web_cache.duckdb file that the
FastAPI API serves data from. This script reads model artifacts,
backtest outputs, and gold data to build a read-optimized cache.

This is also the RECOVERY command the /bets stale-cache refusal names, so it must build a
cache the page can serve: it reads the same bet-list cache sources the Friday orchestrator's
populate_web_cache step reads (the durable outputs/bet_list/ artifacts plus the schedule-derived
per-game freeze), through the one shared reader, so a hand-run rebuild and a scheduled one can
never produce different tables.

It holds the publish lock (``pipeline.cache_lock``, Phase 36 D-18) while it rebuilds, like the
scheduled step, and refuses with exit code 1 while a website push or another rebuild holds it.

The edge bands follow the in-force re-fit generation (Phase 37, D-36), resolved from THIS
command's ``--artifacts-dir`` -- the directory the cache is built from -- so a custom-directory
rebuild never bands with production's generation.

Usage:
    python scripts/populate_cache.py
    python scripts/populate_cache.py --db-path data/custom_cache.duckdb
"""

from __future__ import annotations

import argparse
from pathlib import Path

from api.cache import populate_cache
from backtest.weekly_bet_list import DEFAULT_BET_LIST_DIR
from forward_ledger.cache_sources import read_bet_cache_sources
from forward_ledger.store import LEDGER_DIR
from pipeline.cache_lock import REBUILD_LOCK_WAIT_SECONDS, cache_publish_lock
from refit.generations import in_force_edge_thresholds
from utils.file_lock import FileLockHeldError


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
        "--predictions-dir",
        type=Path,
        default=Path("outputs/predictions"),
        help=(
            "Directory of the current-week predictions_<season>_week<week>.csv files; every "
            "game in them is shown on the site (default: outputs/predictions/)"
        ),
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

    try:
        with cache_publish_lock(wait_seconds=REBUILD_LOCK_WAIT_SECONDS):
            _rebuild(args)
    except FileLockHeldError:
        print(
            "POPULATE_CACHE= refused: a website push or another cache rebuild holds the lock; "
            "try again shortly"
        )
        raise SystemExit(1)


def _rebuild(args: argparse.Namespace) -> None:
    """Read the bet-list cache sources and rebuild the cache (the caller holds the lock)."""
    # Read the bet-list cache sources HERE and pass frames: api/cache.py may import no backtest
    # module (UIAP-01), so it cannot know these filenames or derive a per-game freeze instant.
    # The ONE reader shared with the scheduled step; it follows the forward-rows cutover switch.
    sources = read_bet_cache_sources(args.bet_list_dir, silver_dir=args.silver_dir)
    # The in-force generation's bands (D-36), from the SAME artifacts directory the cache is built
    # from and the ledger directory the reader above defaults to; None for generation 0.
    edge_thresholds = in_force_edge_thresholds(args.artifacts_dir, LEDGER_DIR)

    populate_cache(
        db_path=args.db_path,
        artifacts_dir=args.artifacts_dir,
        outputs_dir=args.outputs_dir,
        gold_dir=args.gold_dir,
        silver_dir=args.silver_dir,
        bet_list_df=sources.bet_list,
        bet_tracker_df=sources.tracker,
        bet_schedule_df=sources.schedule,
        predictions_dir=args.predictions_dir,
        bet_corrections_df=sources.corrections,
        bet_graded_outcomes_df=sources.graded_outcomes,
        forward_verdict_context=sources.verdict_context,
        bet_week_results_df=sources.week_results,
        edge_thresholds_by_target=edge_thresholds,
        refit_panel_df=sources.refit_panel,
        refit_aa_status_df=sources.refit_aa_status,
    )


if __name__ == "__main__":
    main()
