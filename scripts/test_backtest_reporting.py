"""
Test suite for backtest reporting system.

This script validates the BacktestReporter functionality including HTML generation,
seasonal breakdown, cohort analysis, sensitivity analysis, and CSV exports.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tempfile
from datetime import datetime, timedelta
from typing import List, Dict
import numpy as np
import pandas as pd

from backtest.reporting import (
    BacktestReporter, ReportData, CohortResult, CohortType,
    SensitivityResult, ReportSection
)
from backtest.walkforward import BacktestSummary, BacktestResult, BacktestConfig
from backtest.clv_tracking import CLVSummary
from utils.logging_config import get_logger

logger = get_logger(__name__)


def create_mock_backtest_summary() -> BacktestSummary:
    """Create mock backtest summary for testing."""

    config = BacktestConfig(
        start_season=2020,
        end_season=2023,
        model_types=["wp", "ats", "ou"]
    )

    # Create mock results by season and model
    results_by_season = {}
    results_by_model = {"wp": [], "ats": [], "ou": []}

    for season in range(2020, 2024):
        season_results = []
        for week in range(5, 18):
            for model_type in ["wp", "ats", "ou"]:
                result = BacktestResult(
                    season=season,
                    week=week,
                    model_type=model_type,
                    predictions_made=16,
                    accuracy=0.55 + np.random.normal(0, 0.05),
                    log_loss=0.68 + np.random.normal(0, 0.1),
                    brier_score=0.24 + np.random.normal(0, 0.02),
                    mae=0.12 + np.random.normal(0, 0.03),
                    bets_placed=8,
                    betting_roi=0.02 + np.random.normal(0, 0.15),
                    betting_profit=100 + np.random.normal(0, 200),
                    execution_time=45.0
                )
                season_results.append(result)
                results_by_model[model_type].append(result)

        results_by_season[season] = season_results

    overall_metrics = {
        "wp_mean_accuracy": 0.567,
        "wp_mean_log_loss": 0.678,
        "ats_mean_mae": 0.125,
        "ou_mean_mae": 0.118,
        "total_predictions": 6240,
        "total_weeks": 52
    }

    return BacktestSummary(
        config=config,
        total_seasons=4,
        total_weeks=52,
        total_predictions=6240,
        results_by_season=results_by_season,
        results_by_model=results_by_model,
        overall_metrics=overall_metrics,
        seasonal_trends={"wp_accuracy": [0.56, 0.57, 0.55, 0.58]},
        start_time=datetime(2024, 1, 1),
        end_time=datetime(2024, 1, 2),
        total_execution_time=3600.0
    )


def create_mock_raw_results() -> List[BacktestResult]:
    """Create mock raw results for testing."""

    results = []
    for season in range(2020, 2024):
        for week in range(5, 18):
            for model_type in ["wp", "ats", "ou"]:
                result = BacktestResult(
                    season=season,
                    week=week,
                    model_type=model_type,
                    predictions_made=16,
                    accuracy=0.55 + np.random.normal(0, 0.05),
                    log_loss=0.68 + np.random.normal(0, 0.1),
                    brier_score=0.24 + np.random.normal(0, 0.02),
                    mae=0.12 + np.random.normal(0, 0.03),
                    rmse=0.18 + np.random.normal(0, 0.04),
                    bets_placed=8,
                    betting_roi=0.02 + np.random.normal(0, 0.15),
                    betting_profit=100 + np.random.normal(0, 200),
                    units_wagered=800,
                    execution_time=45.0 + np.random.normal(0, 10),
                    feature_count=25,
                    data_quality_score=0.95 + np.random.normal(0, 0.05)
                )
                results.append(result)

    return results


def create_mock_clv_summary() -> CLVSummary:
    """Create mock CLV summary for testing."""

    return CLVSummary(
        total_bets=1200,
        bets_with_clv_data=1150,
        average_absolute_clv=0.015,
        average_percentage_clv=0.025,
        positive_clv_rate=0.58,
        total_ev_from_clv=2400.0,
        average_ev_clv=2.1,
        clv_bucket_performance={
            "Positive CLV": {
                "count": 670,
                "win_rate": 0.54,
                "roi": 0.08,
                "average_clv": 0.025
            },
            "Negative CLV": {
                "count": 480,
                "win_rate": 0.48,
                "roi": -0.03,
                "average_clv": -0.018
            }
        },
        start_date=datetime(2020, 9, 1),
        end_date=datetime(2023, 12, 31),
        clv_t_statistic=2.34,
        clv_p_value=0.019
    )


def create_mock_additional_data() -> Dict:
    """Create mock additional data for enhanced analysis."""

    # Mock weather data
    weather_data = {}
    for season in range(2020, 2024):
        for week in range(5, 18):
            weather_data[f"{season}_{week}"] = {
                "condition": np.random.choice(["clear", "rain", "snow", "cloudy"]),
                "temperature": np.random.randint(20, 80),
                "wind_speed": np.random.randint(0, 25)
            }

    # Mock venue data
    venue_data = {}
    for season in range(2020, 2024):
        for week in range(5, 18):
            venue_data[f"{season}_{week}"] = {
                "roof_type": np.random.choice(["outdoor", "dome", "retractable"]),
                "surface": np.random.choice(["grass", "turf"])
            }

    # Mock travel data
    travel_data = {}
    for season in range(2020, 2024):
        for week in range(5, 18):
            travel_data[f"{season}_{week}"] = {
                "distance_miles": np.random.randint(0, 3000),
                "timezone_diff": np.random.randint(-3, 4)
            }

    return {
        "weather_data": weather_data,
        "venue_data": venue_data,
        "travel_data": travel_data,
        "clv_summary": create_mock_clv_summary()
    }


def test_reporter_initialization():
    """Test reporter initialization and directory creation."""
    logger.info("Running test_reporter_initialization")

    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = BacktestReporter(output_dir=temp_dir)

        assert reporter.output_dir.exists(), "Output directory should be created"
        assert str(reporter.output_dir) == temp_dir, "Should use specified output directory"

    logger.info("Reporter initialization successful")
    return True


def test_results_dataframe_creation():
    """Test conversion of raw results to DataFrame."""
    logger.info("Running test_results_dataframe_creation")

    reporter = BacktestReporter()
    raw_results = create_mock_raw_results()

    results_df = reporter._create_results_dataframe(raw_results)

    # Verify DataFrame structure
    assert isinstance(results_df, pd.DataFrame), "Should return DataFrame"
    assert len(results_df) == len(raw_results), "Should have same number of rows as raw results"

    # Verify columns
    expected_columns = [
        'season', 'week', 'model_type', 'predictions_made', 'accuracy',
        'log_loss', 'brier_score', 'mae', 'rmse', 'bets_placed',
        'betting_roi', 'betting_profit', 'units_wagered', 'execution_time',
        'feature_count', 'data_quality_score'
    ]

    for col in expected_columns:
        assert col in results_df.columns, f"Should have {col} column"

    # Verify data types
    assert results_df['season'].dtype in [np.int64, np.int32], "Season should be integer"
    assert results_df['accuracy'].dtype in [np.float64, np.float32], "Accuracy should be float"

    logger.info(f"DataFrame creation successful: {len(results_df)} rows, {len(results_df.columns)} columns")
    return True


def test_seasonal_analysis():
    """Test seasonal performance analysis."""
    logger.info("Running test_seasonal_analysis")

    reporter = BacktestReporter()
    raw_results = create_mock_raw_results()
    results_df = reporter._create_results_dataframe(raw_results)

    seasonal_breakdown = reporter._analyze_seasonal_performance(results_df)

    # Verify structure
    assert isinstance(seasonal_breakdown, dict), "Should return dictionary"
    assert len(seasonal_breakdown) > 0, "Should have seasonal data"

    # Verify each season
    for season, season_data in seasonal_breakdown.items():
        assert isinstance(season, int), "Season should be integer"
        assert "total_predictions" in season_data, "Should have total predictions"
        assert "total_weeks" in season_data, "Should have total weeks"
        assert "model_performance" in season_data, "Should have model performance"

        # Verify model performance structure
        model_performance = season_data["model_performance"]
        for model_type in ["wp", "ats", "ou"]:
            if model_type in model_performance:
                model_data = model_performance[model_type]
                assert "accuracy" in model_data or "roi" in model_data, f"Should have performance metrics for {model_type}"

    logger.info(f"Seasonal analysis successful: {len(seasonal_breakdown)} seasons analyzed")
    return True


def test_cohort_analysis():
    """Test cohort analysis functionality."""
    logger.info("Running test_cohort_analysis")

    reporter = BacktestReporter()
    raw_results = create_mock_raw_results()
    results_df = reporter._create_results_dataframe(raw_results)
    additional_data = create_mock_additional_data()

    cohort_analysis = reporter._perform_cohort_analysis(results_df, additional_data)

    # Verify structure
    assert isinstance(cohort_analysis, dict), "Should return dictionary"

    # Should have at least season timing cohorts
    assert CohortType.SEASON_TIMING in cohort_analysis, "Should have season timing cohorts"

    # Verify season timing cohorts
    timing_cohorts = cohort_analysis[CohortType.SEASON_TIMING]
    assert len(timing_cohorts) > 0, "Should have timing cohorts"

    for cohort in timing_cohorts:
        assert isinstance(cohort, CohortResult), "Should be CohortResult objects"
        assert cohort.cohort_name in ["Early Season", "Mid Season", "Late Season", "Playoffs"], "Should have valid cohort names"
        assert cohort.sample_size > 0, "Should have positive sample size"
        assert 0 <= cohort.accuracy <= 1, "Accuracy should be between 0 and 1"

    # Should have weather cohorts if weather data provided
    if CohortType.WEATHER in cohort_analysis:
        weather_cohorts = cohort_analysis[CohortType.WEATHER]
        assert len(weather_cohorts) > 0, "Should have weather cohorts"

    logger.info(f"Cohort analysis successful: {len(cohort_analysis)} cohort types analyzed")
    return True


def test_sensitivity_analysis():
    """Test sensitivity analysis functionality."""
    logger.info("Running test_sensitivity_analysis")

    reporter = BacktestReporter()
    raw_results = create_mock_raw_results()
    results_df = reporter._create_results_dataframe(raw_results)

    sensitivity_analysis = reporter._perform_sensitivity_analysis(results_df)

    # Verify structure
    assert isinstance(sensitivity_analysis, dict), "Should return dictionary"

    # Should have EV threshold and Kelly fraction analysis
    expected_analyses = ["ev_threshold", "kelly_fraction"]
    for analysis_name in expected_analyses:
        if analysis_name in sensitivity_analysis:
            result = sensitivity_analysis[analysis_name]
            assert isinstance(result, SensitivityResult), f"Should be SensitivityResult for {analysis_name}"
            assert len(result.parameter_values) > 0, f"Should have parameter values for {analysis_name}"
            assert len(result.roi_values) == len(result.parameter_values), f"Should have matching ROI values for {analysis_name}"
            assert result.optimal_roi_value in result.parameter_values, f"Optimal value should be in parameter range for {analysis_name}"

    logger.info(f"Sensitivity analysis successful: {len(sensitivity_analysis)} parameters analyzed")
    return True


def test_html_report_generation():
    """Test HTML report generation."""
    logger.info("Running test_html_report_generation")

    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = BacktestReporter(output_dir=temp_dir)

        # Create test data
        backtest_summary = create_mock_backtest_summary()
        raw_results = create_mock_raw_results()
        additional_data = create_mock_additional_data()

        # Generate report
        report_path = reporter.generate_full_report(
            backtest_summary=backtest_summary,
            raw_results=raw_results,
            additional_data=additional_data
        )

        # Verify report file was created
        assert os.path.exists(report_path), "HTML report file should exist"
        assert report_path.endswith('.html'), "Report should be HTML file"

        # Verify report content
        with open(report_path, 'r', encoding='utf-8') as f:
            html_content = f.read()

        # Check for key sections
        assert "NFL Prediction System" in html_content, "Should have title"
        assert "Executive Summary" in html_content, "Should have executive summary"
        assert "Performance Overview" in html_content, "Should have performance overview"
        assert "Season-by-Season Performance" in html_content, "Should have seasonal breakdown"
        assert "Cohort Analysis" in html_content, "Should have cohort analysis"
        assert "Sensitivity Analysis" in html_content, "Should have sensitivity analysis"

        # Check for data
        assert "Total Predictions" in html_content, "Should show total predictions"
        assert "Average Accuracy" in html_content, "Should show average accuracy"

    logger.info("HTML report generation successful")
    return True


def test_csv_exports():
    """Test CSV export functionality."""
    logger.info("Running test_csv_exports")

    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = BacktestReporter(output_dir=temp_dir)

        # Create test data
        backtest_summary = create_mock_backtest_summary()
        raw_results = create_mock_raw_results()
        additional_data = create_mock_additional_data()

        # Prepare report data
        report_data = reporter._prepare_report_data(backtest_summary, raw_results, additional_data)

        # Generate CSV exports
        csv_files = reporter._generate_csv_exports(report_data, raw_results)

        # Verify CSV files were created
        assert len(csv_files) >= 3, "Should create at least 3 CSV files"

        expected_files = ["detailed_results.csv", "seasonal_summary.csv", "cohort_analysis.csv"]
        for expected_file in expected_files:
            file_path = os.path.join(temp_dir, expected_file)
            assert os.path.exists(file_path), f"Should create {expected_file}"

            # Verify CSV content
            import csv
            with open(file_path, 'r') as f:
                reader = csv.reader(f)
                header = next(reader)
                assert len(header) > 0, f"Should have header in {expected_file}"

                # Count data rows
                data_rows = list(reader)
                assert len(data_rows) > 0, f"Should have data rows in {expected_file}"

    logger.info(f"CSV export successful: {len(csv_files)} files created")
    return True


def test_report_data_preparation():
    """Test comprehensive report data preparation."""
    logger.info("Running test_report_data_preparation")

    reporter = BacktestReporter()

    # Create test data
    backtest_summary = create_mock_backtest_summary()
    raw_results = create_mock_raw_results()
    additional_data = create_mock_additional_data()

    # Prepare report data
    report_data = reporter._prepare_report_data(backtest_summary, raw_results, additional_data)

    # Verify report data structure
    assert isinstance(report_data, ReportData), "Should return ReportData"
    assert report_data.backtest_summary == backtest_summary, "Should include backtest summary"
    assert isinstance(report_data.seasonal_breakdown, dict), "Should have seasonal breakdown"
    assert isinstance(report_data.cohort_analysis, dict), "Should have cohort analysis"
    assert isinstance(report_data.sensitivity_analysis, dict), "Should have sensitivity analysis"
    assert report_data.clv_summary is not None, "Should include CLV summary"

    # Verify seasonal breakdown
    assert len(report_data.seasonal_breakdown) > 0, "Should have seasonal data"

    # Verify cohort analysis
    assert len(report_data.cohort_analysis) > 0, "Should have cohort data"

    logger.info("Report data preparation successful")
    return True


def test_html_template_sections():
    """Test individual HTML template sections."""
    logger.info("Running test_html_template_sections")

    reporter = BacktestReporter()

    # Create test data
    backtest_summary = create_mock_backtest_summary()
    raw_results = create_mock_raw_results()
    additional_data = create_mock_additional_data()
    report_data = reporter._prepare_report_data(backtest_summary, raw_results, additional_data)

    # Test executive summary section
    exec_summary = reporter._create_executive_summary_section(report_data)
    assert "Executive Summary" in exec_summary, "Should have section title"
    assert "Total Predictions" in exec_summary, "Should show total predictions"
    assert "metric-value" in exec_summary, "Should have metric values"

    # Test performance overview section
    perf_overview = reporter._create_performance_overview_section(report_data)
    assert "Performance Overview" in perf_overview, "Should have section title"
    assert "performance-table" in perf_overview, "Should have performance table"

    # Test seasonal breakdown section
    seasonal_section = reporter._create_seasonal_breakdown_section(report_data)
    assert "Season-by-Season Performance" in seasonal_section, "Should have section title"
    assert "2020" in seasonal_section or "2021" in seasonal_section, "Should show season data"

    logger.info("HTML template sections validated")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    logger.info("Running test_edge_cases")

    reporter = BacktestReporter()

    # Test with minimal data
    minimal_summary = BacktestSummary(
        config=BacktestConfig(),
        total_seasons=1,
        total_weeks=1,
        total_predictions=10,
        results_by_season={2023: []},
        results_by_model={"wp": []},
        overall_metrics={},
        seasonal_trends={},
        start_time=datetime.now()
    )

    minimal_results = [BacktestResult(
        season=2023,
        week=5,
        model_type="wp",
        predictions_made=10,
        accuracy=0.6
    )]

    try:
        # Should handle minimal data without crashing
        report_data = reporter._prepare_report_data(minimal_summary, minimal_results, None)
        assert report_data is not None, "Should handle minimal data"

        # Should handle empty cohort analysis
        results_df = reporter._create_results_dataframe(minimal_results)
        cohort_analysis = reporter._perform_cohort_analysis(results_df, None)
        assert isinstance(cohort_analysis, dict), "Should return dict for empty cohort analysis"

    except Exception as e:
        logger.error(f"Edge case handling failed: {e}")
        return False

    logger.info("Edge cases handled successfully")
    return True


def test_full_integration():
    """Test full integration of reporting system."""
    logger.info("Running test_full_integration")

    with tempfile.TemporaryDirectory() as temp_dir:
        reporter = BacktestReporter(output_dir=temp_dir)

        # Create comprehensive test data
        backtest_summary = create_mock_backtest_summary()
        raw_results = create_mock_raw_results()
        additional_data = create_mock_additional_data()

        # Generate full report
        report_path = reporter.generate_full_report(
            backtest_summary=backtest_summary,
            raw_results=raw_results,
            additional_data=additional_data
        )

        # Verify all outputs
        assert os.path.exists(report_path), "HTML report should exist"

        # Check for CSV files
        csv_files = [f for f in os.listdir(temp_dir) if f.endswith('.csv')]
        assert len(csv_files) >= 3, "Should create multiple CSV files"

        # Verify file sizes (should not be empty)
        assert os.path.getsize(report_path) > 1000, "HTML report should have substantial content"

        for csv_file in csv_files:
            csv_path = os.path.join(temp_dir, csv_file)
            assert os.path.getsize(csv_path) > 100, f"CSV file {csv_file} should have content"

    logger.info("Full integration test successful")
    return True


def run_all_tests():
    """Run all backtest reporting tests."""
    logger.info("Starting backtest reporting test suite")

    tests = [
        test_reporter_initialization,
        test_results_dataframe_creation,
        test_seasonal_analysis,
        test_cohort_analysis,
        test_sensitivity_analysis,
        test_html_report_generation,
        test_csv_exports,
        test_report_data_preparation,
        test_html_template_sections,
        test_edge_cases,
        test_full_integration
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                logger.error(f"Test {test_func.__name__} failed")
        except Exception as e:
            failed += 1
            logger.error(f"Test {test_func.__name__} failed with exception: {e}")
            import traceback
            logger.error(traceback.format_exc())

    logger.info(f"Backtest reporting tests completed: {passed} passed, {failed} failed")

    if failed == 0:
        logger.info("All backtest reporting tests passed!")
    else:
        logger.error(f"{failed} tests failed")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)