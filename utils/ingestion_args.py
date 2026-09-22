"""Standardized argument parsing for data ingestion scripts."""

import argparse

from utils.date_utils import get_current_nfl_week


def season_token(token: str) -> list[int]:
    """One ``--seasons`` token: a single year (``2025``) or an inclusive range (``2025-2026``).

    Returns a list either way, so ``parse_season_week_args`` flattens both shapes. A range
    whose end precedes its start is refused rather than silently read as empty.
    """
    start_text, sep, end_text = token.partition("-")
    try:
        start = int(start_text)
        end = int(end_text) if sep else start
    except ValueError as error:
        msg = (
            f"{token!r} is not a season (2025) or an inclusive season range (2025-2026)"
        )
        raise argparse.ArgumentTypeError(msg) from error
    if end < start:
        msg = f"season range {token!r} ends before it starts"
        raise argparse.ArgumentTypeError(msg)
    return list(range(start, end + 1))


def get_current_season_weeks() -> tuple[int, list[int]]:
    """
    Get weeks 1 through current week of current season.

    Returns:
        Tuple of (current_season, list_of_weeks_1_to_current)
    """
    current_season, current_week = get_current_nfl_week()
    weeks = list(range(1, current_week + 1))
    return current_season, weeks


def add_standard_ingestion_args(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """
    Add standardized ingestion arguments to an ArgumentParser.

    Args:
        parser: ArgumentParser to add arguments to

    Returns:
        ArgumentParser with standard arguments added
    """
    # Season arguments
    season_group = parser.add_mutually_exclusive_group()
    season_group.add_argument(
        "--season", type=int, help="Single season to ingest (default: current)"
    )
    season_group.add_argument(
        "--seasons",
        nargs="+",
        type=season_token,
        help="Multiple seasons to ingest: years and/or inclusive ranges (2013 2025-2026)",
    )

    # Week arguments
    week_group = parser.add_mutually_exclusive_group()
    week_group.add_argument("--week", type=int, help="Single week to ingest")
    week_group.add_argument(
        "--weeks", nargs="+", type=int, help="Multiple weeks to ingest"
    )
    week_group.add_argument(
        "--current",
        action="store_true",
        help="Ingest weeks 1 through current week of current season",
    )
    week_group.add_argument(
        "--all", action="store_true", help="Ingest all weeks of specified season(s)"
    )

    return parser


def parse_season_week_args(args) -> tuple[list[int], list[int] | None]:
    """
    Parse standardized season/week arguments into concrete lists.

    Args:
        args: Parsed arguments from ArgumentParser

    Returns:
        Tuple of (seasons_list, weeks_list_or_none)

    Examples:
        --season 2024 --weeks 1 2 3  -> ([2024], [1, 2, 3])
        --seasons 2023 2024 --all    -> ([2023, 2024], None)
        --current                    -> ([2025], [1, 2, 3])  # if current is 2025 week 3
    """
    # Determine seasons
    if args.seasons:
        # Each token is a list (season_token) when parsed from the CLI; a caller building the
        # Namespace by hand may still pass plain ints.
        seasons = [
            season
            for token in args.seasons
            for season in (token if isinstance(token, list) else [token])
        ]
    elif args.season:
        seasons = [args.season]
    elif args.current:
        current_season, _ = get_current_season_weeks()
        seasons = [current_season]
    else:
        # Default to current season
        current_season, _ = get_current_nfl_week()
        seasons = [current_season]

    # Determine weeks
    if args.weeks:
        weeks = args.weeks
    elif args.week:
        weeks = [args.week]
    elif args.current:
        _, current_weeks = get_current_season_weeks()
        weeks = current_weeks
    elif args.all:
        weeks = None  # All weeks
    else:
        weeks = None  # Default behavior (often all weeks)

    return seasons, weeks


def get_ingestion_summary(seasons: list[int], weeks: list[int] | None) -> str:
    """
    Get a human-readable summary of what will be ingested.

    Args:
        seasons: List of seasons
        weeks: List of weeks or None for all weeks

    Returns:
        Human-readable summary string
    """
    if len(seasons) == 1:
        season_str = f"season {seasons[0]}"
    else:
        season_str = f"seasons {', '.join(map(str, seasons))}"

    if weeks is None:
        week_str = "all weeks"
    elif len(weeks) == 1:
        week_str = f"week {weeks[0]}"
    else:
        week_str = f"weeks {', '.join(map(str, weeks))}"

    return f"{season_str}, {week_str}"
