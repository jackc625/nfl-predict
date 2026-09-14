#!/usr/bin/env python3
"""
Build contextual features for NFL prediction system.

This script builds contextual features including:
- Travel time-zone difference calculations
- Short week detection
- Venue roof type encoding
- Home/away team indicators
- Rest days calculations

Usage:
    python scripts/build_contextual.py --season 2024 --week 1
    python scripts/build_contextual.py --season 2024  # All weeks in season
    python scripts/build_contextual.py --all-seasons  # All available data
    python scripts/build_contextual.py  # Same as --all-seasons
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage import load_dataframe, save_dataframe
from features.contextual import ContextualFeaturesCalculator
from utils import get_logger

logger = get_logger(__name__)


def main():
    """Build contextual features."""
    parser = argparse.ArgumentParser(description="Build contextual features")
    # --all-seasons is the EXPLICIT name for the full historical build, which is
    # also what omitting --season already did. It is a NAMING alias and not new
    # build logic: it leaves target_season None, byte-identically to omission.
    #
    # It exists for the reason scripts/build_features.py:2122-2129 records about
    # its own copy of this flag -- the full build should be ASKED FOR by name
    # rather than reached by leaving an argument out. PIPELINE.md's Stage 2 and
    # Plan 33.1-07's Ruling N3 both name
    # `uv run python scripts/build_contextual.py --all-seasons` as the ONE
    # command for data/silver/contextual_features.parquet, and before this flag
    # existed argparse rejected that documented command with exit 2. The
    # "correction" an operator then reaches for is --season <YEAR>, which builds
    # one season and is the wrong act entirely.
    scope_group = parser.add_mutually_exclusive_group()
    scope_group.add_argument("--season", type=int, help="Target season (e.g., 2024)")
    scope_group.add_argument(
        "--all-seasons",
        action="store_true",
        help=(
            "Build contextual features over every available season (the "
            "default when no --season is given)."
        ),
    )
    parser.add_argument("--week", type=int, help="Target week (1-18)")
    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save features to silver layer",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        default=True,
        help="Validate features after building",
    )

    args = parser.parse_args()

    logger.info("Building contextual features", season=args.season, week=args.week)

    try:
        # Load games data from silver layer
        games_df = load_dataframe("games", layer="silver")
        logger.info("Loaded games data", games_count=len(games_df))

        # Initialize contextual features calculator
        calculator = ContextualFeaturesCalculator()

        # Build contextual features
        features_df = calculator.build_contextual_features(
            games_df=games_df, target_season=args.season, target_week=args.week
        )

        if len(features_df) == 0:
            logger.warning("No contextual features generated")
            return

        logger.info(
            "Generated contextual features",
            games=len(features_df),
            feature_columns=len(features_df.columns),
        )

        # Validate features if requested
        if args.validate:
            is_valid = calculator.validate_contextual_features(features_df)
            if not is_valid:
                logger.error("Contextual features validation failed")
                return
            logger.info("Contextual features validation passed")

        # Display sample features
        print("\nSample contextual features:")
        print("=" * 60)

        # Show first few games
        sample_df = features_df.head(3)
        for _idx, row in sample_df.iterrows():
            print(
                f"\nGame: {row['away_team']} @ {row['home_team']} (Week {row['week']})"
            )
            print("-" * 40)

            # Travel features
            print(f"Travel distance: {row['away_travel_distance_miles']:.0f} miles")
            print(f"Travel fatigue score: {row['away_travel_fatigue_score']:.3f}")
            if row["away_cross_country_travel"] > 0:
                print("Cross-country travel: YES")

            # Rest features
            print(
                f"Rest days - Home: {row['home_rest_days']:.0f}, Away: {row['away_rest_days']:.0f}"
            )
            if row["rest_advantage"] != 0:
                advantage_team = "Home" if row["rest_advantage"] > 0 else "Away"
                print(
                    f"Rest advantage: {advantage_team} (+{abs(row['rest_advantage']):.0f} days)"
                )

            # Game timing
            if row["thursday_game"] > 0:
                print("Thursday Night Football")
            if row["short_week"] > 0:
                print("Short week game")

            # Venue features
            roof_type = (
                "Indoor"
                if row["venue_indoor"] > 0
                else ("Retractable" if row["venue_retractable"] > 0 else "Outdoor")
            )
            print(f"Venue: {roof_type}, {row['venue_elevation_ft']:.0f} ft elevation")
            if row["venue_high_altitude"] > 0:
                print("High altitude venue")

        # Save to silver layer if requested.
        # Write a single self-contained file (no directory partitioning). The
        # per-season --save path appends into this one file with game_id-level
        # latest-wins dedup, so re-running a season replaces its rows while other
        # seasons accumulate -- idempotent. The prior partition_cols=["season"]
        # routed to pq.write_to_dataset against the SHARED data/silver/ root,
        # which mixed this table's files with weather/market/elo and appended a
        # new hash-named file every run (the contextual ~4.9x bloat). This now
        # matches the orchestrator pattern in pipeline/steps.py. (FIX-01, D-13)
        if args.save:
            save_dataframe(
                features_df,
                table_name="contextual_features",
                layer="silver",
            )

            logger.info(
                "Saved contextual features to silver layer",
                table_name="contextual_features",
            )

        logger.info("Contextual features building completed successfully")

    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
        logger.error("Failed to build contextual features", error=str(e))
        raise


if __name__ == "__main__":
    main()
