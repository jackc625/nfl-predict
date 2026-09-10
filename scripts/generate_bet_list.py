"""Produce the durable weekly bet-list artifacts that the /bets page ultimately serves.

THIS IS THE PRODUCER. It writes ``outputs/bet_list/bet_list.parquet`` and its companion
``outputs/bet_list/bet_tracker.json`` by delegating to the ONE weekly selection facade,
``backtest.weekly_bet_list.generate_weekly_bet_list``.

WHY IT EXISTS (Plan 31-22, G-31-123b). Until this script those artifacts had no entry point
outside the 19-step Friday orchestrator: the only caller of the facade was
``pipeline/steps.py::step_generate_recommendations``, step 15 of that run. There was no CLI, no
module ``__main__``, no Makefile target, no scheduler action and no documented manual stage. So
the /bets cold-start recovery instruction was unfollowable -- from a fresh checkout an operator
could not reach a populated bet list by any documented route.

IT IS STAGE ONE OF TWO. Stage two is ``scripts/populate_cache.py``, which COPIES these artifacts
into the DuckDB web cache the site reads. Run in that order:

    uv run python scripts/generate_bet_list.py
    uv run python scripts/populate_cache.py

RUNNING THE COPY STEP ALONE FROM A COLD START PRODUCES AN EMPTY TABLE, NOT A LIST. The copy reads
these artifacts through ``read_bet_list_cache_sources``, which by documented design degrades an
absent source to a zero-row frame rather than raising -- so it succeeds, reports
"Bet list loaded count=0", stamps no per-week marker, and leaves /bets refusing the week. That is
the honest representation of a cache with no bet rows in it; it is not a substitute for producing
them.

NO REPLAY MODE IS EXPOSED. The facade accepts a ``run_mode``, but this command runs FORWARD only.
A replay flag on a recovery command is a way to stamp replay provenance onto a live week -- the
one mislabel the two orthogonal honesty labels exist to prevent -- and the forward mode is the
only one an operator recovering a live page wants.

Usage:
    python scripts/generate_bet_list.py
    python scripts/generate_bet_list.py --season 2025 --week 3
    python scripts/generate_bet_list.py --output-dir outputs/bet_list_scratch

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backtest.weekly_bet_list import (
    BET_LIST_ARTIFACT_NAME,
    BET_TRACKER_ARTIFACT_NAME,
    DEFAULT_BET_LIST_DIR,
    DEFAULT_CHAIN_FIT_PATH,
    generate_weekly_bet_list,
)
from utils import get_logger
from utils.date_utils import get_current_nfl_week

logger = get_logger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    """The CLI surface, modelled on ``scripts/populate_cache.py``.

    Every default is TAKEN from the module that owns it rather than restated, so this script adds
    no second set of defaults that could drift from the facade's.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate the durable weekly bet-list artifacts under outputs/bet_list/. Stage one "
            "of two: run scripts/populate_cache.py afterwards to load them into the web cache."
        )
    )
    parser.add_argument(
        "--season",
        type=int,
        default=None,
        help=(
            "NFL season to select. Defaults, TOGETHER with --week, to the current NFL week -- the "
            "same resolver the Friday orchestrator step uses."
        ),
    )
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="NFL week to select. Defaults together with --season (see above).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_BET_LIST_DIR,
        help=(
            "Directory the two durable artifacts are written to "
            f"(default: {DEFAULT_BET_LIST_DIR.as_posix()}/)"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Root artifacts directory holding the deployed models (default: artifacts/)",
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=Path("data/gold"),
        help="Gold feature matrices directory (default: data/gold/)",
    )
    parser.add_argument(
        "--silver-dir",
        type=Path,
        default=Path("data/silver"),
        help="Silver directory holding games.parquet and odds_snapshot.parquet (default: data/silver/)",
    )
    parser.add_argument(
        "--chain-fit-path",
        type=Path,
        default=DEFAULT_CHAIN_FIT_PATH,
        help=(
            "The pre-registered tune-only fit the per-target EV floors are read from "
            f"(default: {DEFAULT_CHAIN_FIT_PATH.as_posix()})"
        ),
    )
    return parser


def _resolve_week(season: int | None, week: int | None) -> tuple[int, int]:
    """Resolve the (season, week) to select, refusing a half-specified pair.

    Supplying exactly one of the two would mix an EXPLICIT value with a RESOLVED one -- selecting,
    say, week 3 of whatever season "now" happens to be. That is a week nobody asked for, so it is
    refused by name rather than silently completed.
    """
    if (season is None) != (week is None):
        msg = (
            "--season and --week must be supplied TOGETHER or not at all; supplying one would "
            "pair an explicit value with a resolved one and select a week nobody asked for."
        )
        raise SystemExit(msg)
    if season is None or week is None:
        # The SAME resolver step_generate_recommendations uses, so the scheduled and the manual
        # path cannot disagree about which week "now" is.
        return get_current_nfl_week()
    return season, week


def main() -> None:
    """Resolve the week, delegate to the facade, and report both written paths."""
    args = _build_parser().parse_args()
    season, week = _resolve_week(args.season, args.week)

    logger.info(
        "Generating the weekly bet list",
        season=season,
        week=week,
        week_source="explicit" if args.season is not None else "get_current_nfl_week",
        output_dir=Path(args.output_dir).as_posix(),
    )

    # A DELEGATION, and nothing is caught. A week that cannot be selected -- an absent frozen fit,
    # a season the pre-registered bias does not cover, a missing gold matrix -- must fail LOUDLY,
    # exactly as the orchestrator step's docstring already requires. Catching and summarizing here
    # would turn a refusal into a recovery command that appears to have worked.
    generate_weekly_bet_list(
        season,
        week,
        output_dir=args.output_dir,
        artifacts_dir=args.artifacts_dir,
        gold_dir=args.gold_dir,
        silver_dir=args.silver_dir,
        chain_fit_path=args.chain_fit_path,
    )

    output_dir = Path(args.output_dir)
    logger.info(
        "Weekly bet-list artifacts written",
        bet_list=(output_dir / BET_LIST_ARTIFACT_NAME).as_posix(),
        bet_tracker=(output_dir / BET_TRACKER_ARTIFACT_NAME).as_posix(),
        next_step="scripts/populate_cache.py",
    )


if __name__ == "__main__":
    main()
