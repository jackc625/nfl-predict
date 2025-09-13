"""Data Quality Assurance and monitoring for NFL prediction system."""

import argparse
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
import json

from conf.settings import get_settings
from data.storage import get_db_connection, load_dataframe, get_database_stats
from utils import (
    get_logger, 
    get_current_nfl_week,
    validate_game_data,
    validate_odds_data,
    validate_prediction_data,
    validate_data_quality,
    validate_nfl_business_rules,
    validate_temporal_consistency,
    log_data_operation
)


logger = get_logger(__name__)


class DataQualityMonitor:
    """Data Quality Assurance and monitoring system."""
    
    def __init__(self):
        """Initialize data quality monitor."""
        self.settings = get_settings()
        self.db = get_db_connection()
        
        # QA thresholds from configuration
        self.qa_config = self.settings.config.monitoring.data_quality
        
        # Tables to monitor
        self.monitored_tables = {
            'games': {
                'layer': 'silver',
                'validator': validate_game_data,
                'business_rules': 'games',
                'required_columns': ['game_id', 'season', 'week', 'home_team', 'away_team']
            },
            'odds_snapshot': {
                'layer': 'silver', 
                'validator': validate_odds_data,
                'business_rules': 'odds',
                'required_columns': ['game_id', 'snapshot_ts', 'sportsbook']
            },
            'weather_forecast': {
                'layer': 'silver',
                'validator': None,  # Use general validation
                'business_rules': None,
                'required_columns': ['game_id', 'forecast_time', 'is_outdoor']
            },
            'venues': {
                'layer': 'silver',
                'validator': None,
                'business_rules': None,
                'required_columns': ['venue_id', 'venue_name', 'home_teams']
            }
        }
    
    def check_data_freshness(self, table_name: str, max_age_hours: int = 24) -> Dict[str, Any]:
        """Check if data is fresh (recently updated)."""
        logger.info("Checking data freshness", table=table_name, max_age_hours=max_age_hours)
        
        result = {
            'table': table_name,
            'is_fresh': False,
            'last_update': None,
            'age_hours': None,
            'status': 'unknown'
        }
        
        try:
            # Try to load the table
            df = load_dataframe(table_name, layer='silver')
            
            if df.empty:
                result['status'] = 'empty'
                return result
            
            # Look for timestamp columns
            timestamp_cols = [col for col in df.columns 
                            if any(ts in col.lower() for ts in 
                                  ['timestamp', 'time', 'date', 'created', 'updated'])]
            
            if not timestamp_cols:
                result['status'] = 'no_timestamp'
                return result
            
            # Use the most recent timestamp
            latest_times = []
            for col in timestamp_cols:
                try:
                    latest_time = pd.to_datetime(df[col]).max()
                    if pd.notna(latest_time):
                        latest_times.append(latest_time)
                except:
                    continue
            
            if not latest_times:
                result['status'] = 'invalid_timestamps'
                return result
            
            last_update = max(latest_times)
            now = datetime.now(last_update.tz) if last_update.tz else datetime.now()
            age_hours = (now - last_update).total_seconds() / 3600
            
            result.update({
                'last_update': last_update,
                'age_hours': round(age_hours, 2),
                'is_fresh': age_hours <= max_age_hours,
                'status': 'fresh' if age_hours <= max_age_hours else 'stale'
            })
            
        except Exception as e:
            result['status'] = f'error: {str(e)}'
            logger.warning("Data freshness check failed", 
                          table=table_name, error=str(e))
        
        return result
    
    def check_data_completeness(self, table_name: str, season: int, week: int) -> Dict[str, Any]:
        """Check data completeness for a specific season/week."""
        logger.info("Checking data completeness", 
                   table=table_name, season=season, week=week)
        
        result = {
            'table': table_name,
            'season': season,
            'week': week,
            'expected_count': None,
            'actual_count': 0,
            'completeness_pct': 0.0,
            'missing_data': [],
            'status': 'unknown'
        }
        
        try:
            df = load_dataframe(table_name, layer='silver')
            
            if df.empty:
                result['status'] = 'empty'
                return result
            
            # Filter for season/week if applicable
            if 'season' in df.columns and 'week' in df.columns:
                filtered_df = df[(df['season'] == season) & (df['week'] == week)]
            else:
                filtered_df = df
            
            result['actual_count'] = len(filtered_df)
            
            # Determine expected counts based on table type
            if table_name == 'games':
                # Expect 16 games per week in regular season
                expected_count = 16 if week <= 18 else None
            elif table_name == 'odds_snapshot':
                # Expect odds for each game (games * sportsbooks)
                games_count = self._get_games_count(season, week)
                expected_count = games_count * 3 if games_count else None  # Assume 3 sportsbooks avg
            elif table_name == 'weather_forecast':
                # Expect weather for each outdoor game
                games_count = self._get_games_count(season, week)
                expected_count = games_count if games_count else None
            else:
                expected_count = None
            
            result['expected_count'] = expected_count
            
            if expected_count:
                completeness_pct = (result['actual_count'] / expected_count) * 100
                result['completeness_pct'] = round(completeness_pct, 1)
                
                if completeness_pct >= 90:
                    result['status'] = 'complete'
                elif completeness_pct >= 70:
                    result['status'] = 'mostly_complete'
                else:
                    result['status'] = 'incomplete'
            else:
                result['status'] = 'unknown_expected'
            
        except Exception as e:
            result['status'] = f'error: {str(e)}'
            logger.warning("Data completeness check failed",
                          table=table_name, error=str(e))
        
        return result
    
    def _get_games_count(self, season: int, week: int) -> Optional[int]:
        """Get count of games for season/week."""
        try:
            games_df = load_dataframe('games', layer='silver')
            count = len(games_df[(games_df['season'] == season) & (games_df['week'] == week)])
            return count if count > 0 else None
        except:
            return None
    
    def check_data_quality(self, table_name: str) -> Dict[str, Any]:
        """Comprehensive data quality check for a table."""
        logger.info("Running data quality check", table=table_name)
        
        table_config = self.monitored_tables.get(table_name, {})
        
        result = {
            'table': table_name,
            'timestamp': datetime.now(),
            'checks': {}
        }
        
        try:
            df = load_dataframe(table_name, layer='silver')
            
            if df.empty:
                result['checks']['empty_table'] = {'status': 'fail', 'message': 'Table is empty'}
                return result
            
            # 1. Basic data quality validation
            dq_result = validate_data_quality(
                df, 
                table_name,
                expected_columns=table_config.get('required_columns'),
                max_missing_pct=0.2  # 20% max missing values
            )
            
            result['checks']['basic_quality'] = {
                'status': 'pass' if dq_result['is_valid'] else 'fail',
                'row_count': dq_result['row_count'],
                'column_count': dq_result['column_count'],
                'errors': dq_result['errors'],
                'warnings': dq_result['warnings']
            }
            
            # 2. Schema validation if validator available
            validator_func = table_config.get('validator')
            if validator_func:
                validation_errors = validator_func(df)
                result['checks']['schema_validation'] = {
                    'status': 'pass' if not validation_errors else 'fail',
                    'error_count': len(validation_errors),
                    'sample_errors': validation_errors[:5]
                }
            
            # 3. Business rules validation
            business_rules_type = table_config.get('business_rules')
            if business_rules_type:
                business_violations = validate_nfl_business_rules(df, business_rules_type)
                result['checks']['business_rules'] = {
                    'status': 'pass' if not business_violations else 'fail',
                    'violation_count': len(business_violations),
                    'violations': business_violations[:5]
                }
            
            # 4. Temporal consistency (if applicable)
            if 'season' in df.columns and 'week' in df.columns:
                date_col = None
                for col in ['kickoff_et', 'snapshot_ts', 'forecast_time', 'game_time']:
                    if col in df.columns:
                        date_col = col
                        break
                
                if date_col:
                    temporal_violations = validate_temporal_consistency(
                        df, date_col, 'season', 'week'
                    )
                    result['checks']['temporal_consistency'] = {
                        'status': 'pass' if not temporal_violations else 'fail',
                        'violation_count': len(temporal_violations),
                        'violations': temporal_violations[:3]
                    }
            
            # 5. Data distribution checks
            result['checks']['distributions'] = self._check_data_distributions(df, table_name)
            
        except Exception as e:
            result['checks']['error'] = {
                'status': 'error',
                'message': str(e)
            }
            logger.error("Data quality check failed", table=table_name, error=str(e))
        
        return result
    
    def _check_data_distributions(self, df: pd.DataFrame, table_name: str) -> Dict[str, Any]:
        """Check data distributions for anomalies."""
        checks = {}
        
        try:
            # Numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                values = df[col].dropna()
                if len(values) > 0:
                    checks[f'{col}_stats'] = {
                        'mean': round(values.mean(), 2),
                        'std': round(values.std(), 2),
                        'min': round(values.min(), 2),
                        'max': round(values.max(), 2),
                        'outliers': len(values[np.abs(values - values.mean()) > 3 * values.std()])
                    }
            
            # Categorical columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                value_counts = df[col].value_counts()
                checks[f'{col}_categories'] = {
                    'unique_count': len(value_counts),
                    'most_common': value_counts.head(3).to_dict(),
                    'null_count': df[col].isnull().sum()
                }
        
        except Exception as e:
            checks['error'] = str(e)
        
        return checks
    
    def check_data_consistency(self) -> Dict[str, Any]:
        """Check consistency across related tables."""
        logger.info("Checking cross-table data consistency")
        
        result = {
            'timestamp': datetime.now(),
            'checks': {}
        }
        
        try:
            # Check game_id consistency between tables
            games_df = load_dataframe('games', layer='silver')
            game_ids = set(games_df['game_id']) if not games_df.empty else set()
            
            for table in ['odds_snapshot', 'weather_forecast']:
                try:
                    table_df = load_dataframe(table, layer='silver')
                    if not table_df.empty and 'game_id' in table_df.columns:
                        table_game_ids = set(table_df['game_id'])
                        
                        # Check for orphaned records
                        orphaned = table_game_ids - game_ids
                        missing = game_ids - table_game_ids
                        
                        result['checks'][f'{table}_game_id_consistency'] = {
                            'status': 'pass' if not orphaned else 'warning',
                            'orphaned_count': len(orphaned),
                            'missing_count': len(missing),
                            'sample_orphaned': list(orphaned)[:5],
                            'sample_missing': list(missing)[:5]
                        }
                except Exception as e:
                    result['checks'][f'{table}_consistency_error'] = str(e)
            
            # Check team name consistency
            if not games_df.empty:
                all_teams = set(games_df['home_team']).union(set(games_df['away_team']))
                
                try:
                    venues_df = load_dataframe('venues', layer='silver')
                    if not venues_df.empty:
                        venue_teams = set()
                        for teams_list in venues_df['home_teams']:
                            if isinstance(teams_list, list):
                                venue_teams.update(teams_list)
                            else:
                                venue_teams.add(teams_list)
                        
                        teams_without_venues = all_teams - venue_teams
                        
                        result['checks']['team_venue_consistency'] = {
                            'status': 'pass' if not teams_without_venues else 'warning',
                            'teams_without_venues': list(teams_without_venues)
                        }
                except Exception as e:
                    result['checks']['team_venue_consistency_error'] = str(e)
        
        except Exception as e:
            result['checks']['error'] = str(e)
            logger.error("Data consistency check failed", error=str(e))
        
        return result
    
    def generate_qa_report(
        self, 
        season: Optional[int] = None, 
        week: Optional[int] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive QA report."""
        if season is None or week is None:
            current_season, current_week = get_current_nfl_week()
            season = season or current_season
            week = week or current_week
        
        logger.info("Generating QA report", season=season, week=week)
        
        report = {
            'timestamp': datetime.now(),
            'season': season,
            'week': week,
            'summary': {},
            'table_reports': {},
            'consistency_check': {},
            'database_stats': {},
            'recommendations': []
        }
        
        try:
            # Overall summary
            total_checks = 0
            passed_checks = 0
            failed_checks = 0
            warnings = 0
            
            # Check each monitored table
            for table_name in self.monitored_tables:
                logger.info("Processing table for QA report", table=table_name)
                
                table_report = {
                    'freshness': self.check_data_freshness(table_name),
                    'completeness': self.check_data_completeness(table_name, season, week),
                    'quality': self.check_data_quality(table_name)
                }
                
                report['table_reports'][table_name] = table_report
                
                # Update summary counts
                for check_type, check_result in table_report.items():
                    if isinstance(check_result, dict):
                        status = check_result.get('status', 'unknown')
                        total_checks += 1
                        if status == 'pass' or status in ['fresh', 'complete']:
                            passed_checks += 1
                        elif status == 'fail' or status in ['stale', 'incomplete']:
                            failed_checks += 1
                        elif status in ['warning', 'mostly_complete']:
                            warnings += 1
            
            # Cross-table consistency
            report['consistency_check'] = self.check_data_consistency()
            
            # Database statistics
            try:
                report['database_stats'] = get_database_stats()
            except Exception as e:
                report['database_stats'] = {'error': str(e)}
            
            # Summary
            report['summary'] = {
                'total_checks': total_checks,
                'passed': passed_checks,
                'failed': failed_checks,
                'warnings': warnings,
                'pass_rate': round((passed_checks / total_checks) * 100, 1) if total_checks > 0 else 0,
                'overall_status': 'healthy' if failed_checks == 0 else 'issues_detected'
            }
            
            # Generate recommendations
            report['recommendations'] = self._generate_recommendations(report)
            
            # Log summary
            logger.info("QA report generated",
                       total_checks=total_checks,
                       passed=passed_checks,
                       failed=failed_checks,
                       warnings=warnings,
                       overall_status=report['summary']['overall_status'])
            
        except Exception as e:
            report['error'] = str(e)
            logger.error("QA report generation failed", error=str(e))
        
        return report
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on QA results."""
        recommendations = []
        
        try:
            # Check for stale data
            for table, table_report in report.get('table_reports', {}).items():
                freshness = table_report.get('freshness', {})
                if freshness.get('status') == 'stale':
                    age_hours = freshness.get('age_hours', 0)
                    recommendations.append(
                        f"Update {table} data - last updated {age_hours:.1f} hours ago"
                    )
            
            # Check for incomplete data
            for table, table_report in report.get('table_reports', {}).items():
                completeness = table_report.get('completeness', {})
                if completeness.get('status') in ['incomplete', 'mostly_complete']:
                    pct = completeness.get('completeness_pct', 0)
                    recommendations.append(
                        f"Investigate {table} completeness - only {pct}% complete"
                    )
            
            # Check for quality issues
            for table, table_report in report.get('table_reports', {}).items():
                quality = table_report.get('quality', {})
                checks = quality.get('checks', {})
                
                for check_name, check_result in checks.items():
                    if check_result.get('status') == 'fail':
                        recommendations.append(
                            f"Fix {check_name} issues in {table} table"
                        )
            
            # Check consistency issues
            consistency = report.get('consistency_check', {}).get('checks', {})
            for check_name, check_result in consistency.items():
                if isinstance(check_result, dict) and check_result.get('status') == 'warning':
                    recommendations.append(f"Review {check_name} data consistency")
            
        except Exception as e:
            recommendations.append(f"Error generating recommendations: {e}")
        
        return recommendations
    
    def save_qa_report(self, report: Dict[str, Any], output_dir: Optional[str] = None) -> str:
        """Save QA report to file."""
        if output_dir is None:
            output_dir = self.settings.get_data_path("outputs")
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_report_{timestamp}.json"
        filepath = output_dir / filename
        
        # Convert datetime objects to strings for JSON serialization
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=serialize_datetime)
        
        logger.info("QA report saved", filepath=str(filepath))
        return str(filepath)


def main():
    """CLI entry point for data QA monitoring."""
    parser = argparse.ArgumentParser(description="NFL data quality assurance")
    parser.add_argument('--season', type=int, help='Season to check (default: current)')
    parser.add_argument('--week', type=int, help='Week to check (default: current)')
    parser.add_argument('--current', action='store_true', help='Check current week')
    parser.add_argument('--table', type=str, 
                       choices=['games', 'odds_snapshot', 'weather_forecast', 'venues'],
                       help='Check specific table only')
    parser.add_argument('--check', type=str,
                       choices=['freshness', 'completeness', 'quality', 'consistency'],
                       help='Run specific check only')
    parser.add_argument('--output', type=str, help='Output directory for report')
    parser.add_argument('--save', action='store_true', help='Save report to file')
    
    args = parser.parse_args()
    
    try:
        # Setup logging
        from utils import setup_logging
        setup_logging()
        
        # Determine season and week
        if args.current or (not args.season and not args.week):
            current_season, current_week = get_current_nfl_week()
            season = args.season or current_season
            week = args.week or current_week
        else:
            season = args.season
            week = args.week
        
        # Initialize monitor
        monitor = DataQualityMonitor()
        
        if args.table and args.check:
            # Run specific check on specific table
            if args.check == 'freshness':
                result = monitor.check_data_freshness(args.table)
            elif args.check == 'completeness':
                result = monitor.check_data_completeness(args.table, season, week)
            elif args.check == 'quality':
                result = monitor.check_data_quality(args.table)
            elif args.check == 'consistency':
                result = monitor.check_data_consistency()
            
            print(f"Check: {args.check} on table: {args.table}")
            print(f"Status: {result.get('status', 'unknown')}")
            if 'message' in result:
                print(f"Message: {result['message']}")
            
        elif args.table:
            # Check specific table
            print(f"Checking table: {args.table}")
            
            freshness = monitor.check_data_freshness(args.table)
            print(f"  Freshness: {freshness['status']}")
            
            completeness = monitor.check_data_completeness(args.table, season, week)
            print(f"  Completeness: {completeness['status']} ({completeness.get('completeness_pct', 0)}%)")
            
            quality = monitor.check_data_quality(args.table)
            quality_status = 'pass' if all(
                check.get('status') == 'pass' 
                for check in quality.get('checks', {}).values()
                if isinstance(check, dict)
            ) else 'issues'
            print(f"  Quality: {quality_status}")
            
        else:
            # Generate full report
            report = monitor.generate_qa_report(season, week)
            
            print(f"QA Report for Season {season}, Week {week}")
            print(f"Overall Status: {report['summary']['overall_status']}")
            print(f"Total Checks: {report['summary']['total_checks']}")
            print(f"Pass Rate: {report['summary']['pass_rate']}%")
            print(f"Failed: {report['summary']['failed']}")
            print(f"Warnings: {report['summary']['warnings']}")
            
            if report['recommendations']:
                print("\nRecommendations:")
                for rec in report['recommendations']:
                    print(f"  • {rec}")
            
            # Save report if requested
            if args.save:
                filepath = monitor.save_qa_report(report, args.output)
                print(f"\nReport saved to: {filepath}")
        
    except Exception as e:
        logger.error("Data QA CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()