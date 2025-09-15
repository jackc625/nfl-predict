#!/usr/bin/env python3
"""
Feature Validation Runner

This script validates NFL prediction features for data leakage,
distribution properties, and statistical consistency.

Usage:
    python scripts/validate_features.py --season 2024 --week 1
    python scripts/validate_features.py --season 2024  # All weeks in season
    python scripts/validate_features.py  # All available features
    python scripts/validate_features.py --target wp --report validation_report.txt
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.validation import FeatureValidator
from data.storage import load_dataframe
from utils import get_logger

logger = get_logger(__name__)


def main():
    """Run feature validation."""
    parser = argparse.ArgumentParser(description='Validate NFL prediction features')
    parser.add_argument('--season', type=int, help='Target season (e.g., 2024)')
    parser.add_argument('--week', type=int, help='Target week (1-18)')
    parser.add_argument('--target', choices=['wp', 'ats', 'ou', 'all'], default='all',
                       help='Target type to validate')
    parser.add_argument('--report', type=str, help='Path to save validation report')
    parser.add_argument('--strict', action='store_true',
                       help='Use strict validation (fail on warnings)')

    args = parser.parse_args()

    logger.info("Starting feature validation",
               season=args.season, week=args.week, target=args.target)

    try:
        # Initialize validator
        validator = FeatureValidator()

        # Load features from gold layer
        features_df = None

        if args.target == 'all':
            # Try to load combined features first
            try:
                features_df = load_dataframe('features_combined', layer='gold')
                logger.info("Loaded combined features", records=len(features_df))
            except FileNotFoundError:
                logger.warning("Combined features not found, trying individual targets")

        # If combined features not available, try target-specific features
        if features_df is None:
            target_tables = ['features_wp', 'features_ats', 'features_ou']
            all_features = []

            for table in target_tables:
                try:
                    target_df = load_dataframe(table, layer='gold')
                    target_df['target_type'] = table.replace('features_', '')
                    all_features.append(target_df)
                    logger.info(f"Loaded {table}", records=len(target_df))
                except FileNotFoundError:
                    logger.warning(f"Features table {table} not found")

            if all_features:
                features_df = pd.concat(all_features, ignore_index=True)
                logger.info("Combined target-specific features", total_records=len(features_df))

        # If still no features, try silver layer feature tables
        if features_df is None:
            logger.info("No gold layer features found, checking silver layer...")

            silver_tables = [
                'team_form_features', 'elo_features', 'contextual_features',
                'weather_features', 'market_anchor_features'
            ]

            feature_dfs = []
            base_games_df = None

            for table in silver_tables:
                try:
                    table_df = load_dataframe(table, layer='silver')
                    logger.info(f"Loaded {table}", records=len(table_df))

                    if base_games_df is None:
                        base_games_df = table_df[['game_id', 'season', 'week']].copy()

                    feature_dfs.append(table_df)
                except FileNotFoundError:
                    logger.warning(f"Silver layer table {table} not found")

            if feature_dfs and base_games_df is not None:
                # Merge all feature tables
                features_df = base_games_df.copy()
                for feature_df in feature_dfs:
                    features_df = features_df.merge(
                        feature_df,
                        on=['game_id', 'season', 'week'],
                        how='left'
                    )
                logger.info("Combined silver layer features", total_features=len(features_df.columns))

        if features_df is None:
            logger.error("No feature data found to validate")
            print("❌ No feature data found. Please run feature building scripts first.")
            return

        # Filter by season/week if specified
        original_count = len(features_df)
        if args.season:
            features_df = features_df[features_df['season'] == args.season]
        if args.week:
            features_df = features_df[features_df['week'] == args.week]

        if len(features_df) == 0:
            logger.error("No features found for specified season/week",
                        season=args.season, week=args.week)
            print(f"❌ No features found for season {args.season}, week {args.week}")
            return

        logger.info(f"Filtered features: {original_count} → {len(features_df)} games")

        # Run validation
        print("🔍 Running feature validation...")
        print("=" * 60)

        validation_results = validator.validate_features(
            features_df=features_df,
            target_type=args.target,
            season=args.season,
            week=args.week
        )

        # Display key results
        status = "[PASS]" if validation_results['validation_passed'] else "[FAIL]"
        print(f"\nValidation Status: {status}")
        print(f"Features validated: {validation_results['total_features']}")
        print(f"Games validated: {validation_results['total_games']}")
        print(f"Errors: {len(validation_results['errors'])}")
        print(f"Warnings: {len(validation_results['warnings'])}")

        # Show critical errors
        if validation_results['errors']:
            print(f"\n[ERRORS] CRITICAL ERRORS:")
            for error in validation_results['errors']:
                print(f"   {error}")

        # Show key warnings (first 10)
        if validation_results['warnings']:
            print(f"\n[WARN] WARNINGS ({len(validation_results['warnings'])} total):")
            for warning in validation_results['warnings'][:10]:
                print(f"   {warning}")
            if len(validation_results['warnings']) > 10:
                print(f"   ... and {len(validation_results['warnings']) - 10} more warnings")

        # Display validation check summaries
        print(f"\n[SUMMARY] VALIDATION CHECK SUMMARY:")
        print("-" * 40)

        check_summaries = {
            'data_leakage': 'Data Leakage Detection',
            'missing_data': 'Missing Data Analysis',
            'distributions': 'Feature Distributions',
            'correlations': 'Feature Correlations',
            'temporal_consistency': 'Temporal Consistency',
            'completeness': 'Feature Completeness',
            'statistical_properties': 'Statistical Properties'
        }

        for check_name, check_title in check_summaries.items():
            if check_name in validation_results['checks']:
                check_result = validation_results['checks'][check_name]
                if isinstance(check_result, dict) and 'passed' in check_result:
                    status_icon = "[PASS]" if check_result['passed'] else "[FAIL]"
                    print(f"{status_icon} {check_title}")

                    # Add specific details
                    if check_name == 'data_leakage':
                        leakage_cols = check_result.get('leakage_columns', [])
                        if leakage_cols:
                            print(f"    -> Leakage columns: {len(leakage_cols)}")
                        else:
                            print("    -> No data leakage detected")

                    elif check_name == 'missing_data':
                        high_missing = check_result.get('high_missing_features', [])
                        if high_missing:
                            print(f"    -> High missing features: {len(high_missing)}")

                    elif check_name == 'correlations':
                        high_corr = check_result.get('high_correlations', [])
                        if high_corr:
                            print(f"    -> High correlation pairs: {len(high_corr)}")

                    elif check_name == 'distributions':
                        constant = len(check_result.get('constant_features', []))
                        outliers = len(check_result.get('outlier_features', []))
                        if constant > 0:
                            print(f"    -> Constant features: {constant}")
                        if outliers > 0:
                            print(f"    -> Features with outliers: {outliers}")

                    elif check_name == 'completeness':
                        missing_groups = check_result.get('missing_feature_groups', [])
                        if missing_groups:
                            print(f"    -> Missing feature groups: {missing_groups}")

        # Generate and save detailed report if requested
        if args.report:
            report_text = validator.generate_validation_report(
                validation_results, args.report
            )
            print(f"\n[INFO] Detailed validation report saved to: {args.report}")
        else:
            # Generate report to console
            print(f"\n[REPORT] DETAILED VALIDATION REPORT:")
            print("=" * 60)
            report_text = validator.generate_validation_report(validation_results)
            print(report_text)

        # Exit with error code if validation failed or strict mode enabled
        if not validation_results['validation_passed']:
            logger.error("Feature validation failed")
            sys.exit(1)
        elif args.strict and validation_results['warnings']:
            logger.error("Strict mode: validation failed due to warnings")
            sys.exit(1)
        else:
            logger.info("Feature validation completed successfully")
            print(f"\n[SUCCESS] Feature validation completed successfully!")

    except Exception as e:
        logger.error("Feature validation failed", error=str(e))
        print(f"\n[ERROR] Feature validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()