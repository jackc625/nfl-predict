#!/usr/bin/env python3
"""
Build market anchor features for NFL prediction system.

This script builds market anchor features from betting odds:
- Opening line capture (earliest available odds ≥24 hours before kickoff)
- Snapshot line capture (Friday 6 PM ET cutoff - STRICT no-leakage)
- Devigged probability calculations
- Line movement tracking and sharp money indicators
- Market efficiency metrics

Usage:
    python scripts/build_market_anchors.py --season 2024 --week 1
    python scripts/build_market_anchors.py --season 2024  # All weeks in season
    python scripts/build_market_anchors.py  # All available data
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.market_anchors import MarketAnchorFeaturesCalculator
from data.storage import load_dataframe, save_dataframe
from utils import get_logger

logger = get_logger(__name__)


def main():
    """Build market anchor features."""
    parser = argparse.ArgumentParser(description='Build market anchor features')
    parser.add_argument('--season', type=int, help='Target season (e.g., 2024)')
    parser.add_argument('--week', type=int, help='Target week (1-18)')
    parser.add_argument('--save', action='store_true', default=True,
                       help='Save features to silver layer')
    parser.add_argument('--validate', action='store_true', default=True,
                       help='Validate features after building')

    args = parser.parse_args()

    logger.info("Building market anchor features",
               season=args.season, week=args.week)

    try:
        # Load games data from silver layer
        games_df = load_dataframe('games', layer='silver')
        logger.info("Loaded games data", games_count=len(games_df))

        # Initialize market anchor features calculator
        calculator = MarketAnchorFeaturesCalculator()

        # Build market anchor features
        features_df = calculator.build_market_anchor_features(
            games_df=games_df,
            target_season=args.season,
            target_week=args.week
        )

        if len(features_df) == 0:
            logger.warning("No market anchor features generated")
            return

        logger.info("Generated market anchor features",
                   games=len(features_df),
                   games_with_lines=int(features_df['has_snapshot_lines'].sum()),
                   games_with_movement=int(features_df['has_line_movement'].sum()))

        # Validate features if requested
        if args.validate:
            is_valid = calculator.validate_market_anchor_features(features_df)
            if not is_valid:
                logger.error("Market anchor features validation failed")
                return
            logger.info("Market anchor features validation passed")

        # Display market anchor features summary
        print("\nMarket anchor features summary:")
        print("=" * 60)

        games_with_lines = features_df[features_df['has_snapshot_lines'] == 1.0]
        games_without_lines = features_df[features_df['has_snapshot_lines'] == 0.0]
        games_with_movement = features_df[features_df['has_line_movement'] == 1.0]

        print(f"Total games: {len(features_df)}")
        print(f"Games with snapshot lines: {len(games_with_lines)}")
        print(f"Games without lines (defaults): {len(games_without_lines)}")
        print(f"Games with line movement: {len(games_with_movement)}")

        if len(games_with_lines) > 0:
            # Analyze market features
            significant_movement = games_with_lines[games_with_lines.get('significant_line_movement', 0) == 1.0]
            reverse_movement = games_with_lines[games_with_lines.get('reverse_line_movement', 0) == 1.0]

            print(f"Games with significant line movement: {len(significant_movement)}")
            print(f"Games with reverse line movement (sharp money): {len(reverse_movement)}")

            # Show vig statistics
            vig_cols = ['snapshot_ml_vig', 'snapshot_spread_vig', 'snapshot_total_vig']
            for vig_col in vig_cols:
                if vig_col in games_with_lines.columns:
                    vig_values = games_with_lines[vig_col].dropna()
                    if len(vig_values) > 0:
                        print(f"{vig_col.replace('snapshot_', '').replace('_', ' ').title()}: {vig_values.mean():.3f} avg")

            # Show probability ranges
            prob_cols = [col for col in games_with_lines.columns if 'prob' in col and 'fair' in col]
            if prob_cols:
                print(f"\nDevigged probability ranges:")
                for prob_col in prob_cols[:6]:  # Show first 6
                    prob_values = games_with_lines[prob_col].dropna()
                    if len(prob_values) > 0:
                        print(f"  {prob_col}: {prob_values.min():.3f} - {prob_values.max():.3f}")

        # Show sample market movements
        if len(games_with_movement) > 0:
            print(f"\nSample line movements:")
            movement_cols = ['ml_home_movement', 'spread_movement', 'total_movement']
            for _, game in games_with_movement.head(3).iterrows():
                print(f"\n{game['game_id']}:")
                for col in movement_cols:
                    if col in game and pd.notna(game[col]):
                        movement_type = col.replace('_movement', '').replace('_', ' ').upper()
                        print(f"  {movement_type}: {game[col]:+.1f}")

                # Market indicators
                if game.get('significant_line_movement', 0) == 1.0:
                    print("  *** SIGNIFICANT LINE MOVEMENT ***")
                if game.get('reverse_line_movement', 0) == 1.0:
                    print("  *** REVERSE LINE MOVEMENT (SHARP MONEY) ***")

        # Show consensus quality indicators
        if len(games_with_lines) > 0:
            sportsbook_counts = games_with_lines.get('snapshot_num_sportsbooks', pd.Series([0]))
            if sportsbook_counts.sum() > 0:
                print(f"\nConsensus quality:")
                print(f"  Average sportsbooks per game: {sportsbook_counts.mean():.1f}")
                print(f"  Min sportsbooks: {sportsbook_counts.min()}")
                print(f"  Max sportsbooks: {sportsbook_counts.max()}")

        # Save to silver layer if requested
        if args.save:
            # Determine partition columns
            partition_cols = []
            if args.season:
                partition_cols.append('season')

            save_dataframe(
                features_df,
                table_name='market_anchor_features',
                layer='silver',
                partition_cols=partition_cols if partition_cols else None
            )

            logger.info("Saved market anchor features to silver layer",
                       table_name='market_anchor_features')

        # CRITICAL: Validate no closing lines are present
        closing_check_cols = [col for col in features_df.columns if 'closing' in col.lower()]
        if closing_check_cols:
            logger.error("CRITICAL ERROR: Closing line data detected in features",
                        closing_columns=closing_check_cols)
            print(f"\n❌ CRITICAL ERROR: Closing line data found: {closing_check_cols}")
            print("This violates the strict no-leakage policy!")
            return

        print(f"\n✅ NO-LEAKAGE VALIDATION PASSED: No closing line data detected")
        logger.info("Market anchor features building completed successfully")

    except Exception as e:
        logger.error("Failed to build market anchor features", error=str(e))
        raise


if __name__ == "__main__":
    main()