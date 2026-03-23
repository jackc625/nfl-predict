#!/usr/bin/env python3
"""
Feature Validation System

This module validates NFL prediction features for:
- Data leakage detection (future information)
- Feature distribution validation
- Missing data analysis
- Feature correlation analysis
- Statistical properties validation
- Time series consistency checks

The validation system ensures features are properly constructed and safe
for use in machine learning models without introducing look-ahead bias.

Includes:
- FeatureValidator: Report-only validation (existing, unchanged)
- LeakageViolation: Exception for temporal leakage violations
- LeakageGate: Hard-fail validation gate for the feature pipeline
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# LeakageViolation Exception
# ---------------------------------------------------------------------------


class LeakageViolation(Exception):
    """Raised when temporal data leakage is detected in features.

    Carries a details dict with structured information about the violation
    for diagnostic report generation.
    """

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.details = details or {}


# ---------------------------------------------------------------------------
# LeakageGate: Hard-fail validation gate
# ---------------------------------------------------------------------------


class LeakageGate:
    """Hard-fail validation gate for the feature pipeline.

    Two stages:
    1. Per-builder time-fence check (fast, runs after each builder)
    2. Full matrix validation (runs on combined features)

    Hard failures: leakage violations, missing required feature groups,
                   Elo ordering violations
    Warnings only: distribution issues, correlation issues, missing data
                   below threshold
    """

    # Columns that indicate post-game / future information
    LEAKAGE_KEYWORDS = [
        "closing",
        "final",
        "result",
        "outcome",
        "actual",
        "post_game",
        "final_score",
        "winner",
        "loser",
        "margin",
        "total_score",
    ]

    # Required feature groups: at least one feature from each must exist
    # Missing any of these is a hard failure
    REQUIRED_FEATURE_GROUPS = {
        "elo": "elo_",
        "team_form": "rolling_",
    }

    # Optional feature groups: warn if missing but do not fail
    OPTIONAL_FEATURE_GROUPS = {
        "weather": "weather_",
        "market": "snapshot_",
        "qb": "qb_",
        "contextual_new": "season_progress",
        "divisional": "is_divisional",
    }

    def __init__(self) -> None:
        self.logger = get_logger(f"{__name__}.LeakageGate")

    # -- Stage 1: Per-builder time-fence check --------------------------------

    def check_time_fence(
        self,
        features_df: pd.DataFrame,
        as_of_datetime: datetime,
        builder_name: str,
    ) -> None:
        """Check that no row in features_df has a timestamp after as_of_datetime.

        Inspects columns: game_date, kickoff_et, snapshot_ts.

        Args:
            features_df: Output DataFrame from a single feature builder.
            as_of_datetime: The time-fence cutoff.
            builder_name: Name of the builder (for diagnostics).

        Raises:
            LeakageViolation: If any row violates the time-fence.
        """
        timestamp_cols = ["game_date", "kickoff_et", "snapshot_ts"]

        for col in timestamp_cols:
            if col not in features_df.columns:
                continue

            col_values = pd.to_datetime(features_df[col], errors="coerce")
            as_of_ts = pd.Timestamp(as_of_datetime)

            # Align timezone awareness: if column is tz-aware, make cutoff tz-aware too
            if col_values.dt.tz is not None and as_of_ts.tz is None:
                as_of_ts = as_of_ts.tz_localize(col_values.dt.tz)
            elif col_values.dt.tz is None and as_of_ts.tz is not None:
                as_of_ts = as_of_ts.tz_localize(None)

            # For snapshot_ts: use <= (snapshot at the cutoff is allowed)
            if col == "snapshot_ts":
                future_mask = col_values > as_of_ts
            else:
                future_mask = col_values > as_of_ts

            future_count = future_mask.sum()
            if future_count > 0:
                latest = col_values[future_mask].max()
                raise LeakageViolation(
                    f"Time-fence violation in {builder_name}: "
                    f"{future_count} rows have {col} after {as_of_datetime}",
                    details={
                        "builder": builder_name,
                        "violation_type": "time_fence",
                        "column": col,
                        "affected_rows": int(future_count),
                        "latest_timestamp": str(latest),
                        "cutoff": str(as_of_datetime),
                    },
                )

        self.logger.debug(
            "Time-fence check passed",
            builder=builder_name,
            rows=len(features_df),
        )

    # -- Stage 2: Full combined-matrix validation -----------------------------

    def validate_combined_matrix(
        self,
        combined_df: pd.DataFrame,
        as_of_datetime: datetime,
    ) -> None:
        """Validate the combined feature matrix for leakage and completeness.

        Hard failures:
        - Leakage keywords found in column names
        - Required feature groups missing entirely

        Warnings only (logged but not raised):
        - Optional feature groups missing
        - Distribution anomalies

        Args:
            combined_df: The fully merged feature matrix.
            as_of_datetime: The time-fence cutoff.

        Raises:
            LeakageViolation: On leakage keyword or missing required group.
        """
        columns_lower = {col: col.lower() for col in combined_df.columns}

        # -- Check leakage keywords in column names --
        leaked_cols = []
        for col, col_lower in columns_lower.items():
            for keyword in self.LEAKAGE_KEYWORDS:
                if keyword in col_lower:
                    leaked_cols.append((col, keyword))

        if leaked_cols:
            raise LeakageViolation(
                f"Leakage keyword columns found: {[c for c, _ in leaked_cols]}",
                details={
                    "violation_type": "leakage_keyword",
                    "affected_features": [c for c, _ in leaked_cols],
                    "keywords_matched": [k for _, k in leaked_cols],
                },
            )

        # -- Check required feature groups --
        missing_required = []
        for group_name, prefix in self.REQUIRED_FEATURE_GROUPS.items():
            has_group = any(prefix in col_lower for col_lower in columns_lower.values())
            if not has_group:
                missing_required.append(group_name)

        if missing_required:
            raise LeakageViolation(
                f"Missing required feature groups: {missing_required}",
                details={
                    "violation_type": "missing_required_group",
                    "missing_groups": missing_required,
                },
            )

        # -- Warn on optional feature groups (no raise) --
        for group_name, prefix in self.OPTIONAL_FEATURE_GROUPS.items():
            has_group = any(prefix in col_lower for col_lower in columns_lower.values())
            if not has_group:
                self.logger.warning(
                    "Optional feature group missing",
                    group=group_name,
                    prefix=prefix,
                )

        self.logger.info(
            "Combined matrix validation passed",
            columns=len(combined_df.columns),
            rows=len(combined_df),
        )

    # -- Elo chronological ordering check -------------------------------------

    def check_elo_ordering(
        self,
        elo_history: list[dict[str, Any]] | pd.DataFrame,
    ) -> None:
        """Verify that Elo updates are strictly ordered by game_date within each season.

        Args:
            elo_history: Elo update records with season, game_date, team columns.

        Raises:
            LeakageViolation: If updates are out of chronological order.
        """
        if isinstance(elo_history, list):
            df = pd.DataFrame(elo_history)
        else:
            df = elo_history.copy()

        if "game_date" not in df.columns:
            self.logger.warning(
                "No game_date column in Elo history; skipping ordering check"
            )
            return

        df["game_date"] = pd.to_datetime(df["game_date"])

        out_of_order = []

        # Check within each (season, team) group
        group_cols = ["season", "team"] if "team" in df.columns else ["season"]
        for group_key, group_df in df.groupby(group_cols):
            sorted_group = group_df.sort_index()  # preserve insertion order
            dates = sorted_group["game_date"].values

            for i in range(1, len(dates)):
                if dates[i] < dates[i - 1]:
                    out_of_order.append(
                        {
                            "group": str(group_key),
                            "index": int(sorted_group.index[i]),
                            "date": str(dates[i]),
                            "prev_date": str(dates[i - 1]),
                        }
                    )

        if out_of_order:
            raise LeakageViolation(
                f"Elo updates are not chronologically ordered: "
                f"{len(out_of_order)} out-of-order entries found",
                details={
                    "violation_type": "elo_ordering",
                    "out_of_order_games": out_of_order,
                },
            )

        self.logger.debug("Elo ordering check passed", records=len(df))

    # -- Diagnostic report ----------------------------------------------------

    def write_diagnostic_report(
        self,
        violation: LeakageViolation,
        output_dir: str | None = None,
    ) -> str:
        """Write a JSON diagnostic report for a LeakageViolation.

        Args:
            violation: The violation to report.
            output_dir: Directory to write the report to. Defaults to
                outputs/diagnostics/.

        Returns:
            Absolute path to the written report file.
        """
        if output_dir is None:
            output_dir = str(project_root / "outputs" / "diagnostics")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"leakage_{timestamp}.json"
        report_path = output_path / report_filename

        report = {
            "violation_type": violation.details.get("violation_type", "unknown"),
            "message": str(violation),
            "timestamp": datetime.now().isoformat(),
            "details": violation.details,
        }

        report_path.write_text(
            json.dumps(report, indent=2, default=str), encoding="utf-8"
        )

        self.logger.error(
            "Diagnostic report written",
            path=str(report_path),
            violation_type=report["violation_type"],
        )

        return str(report_path)


class FeatureValidator:
    """
    Comprehensive feature validation system for NFL prediction features.

    Validates features for data leakage, distribution properties, and
    statistical consistency across seasons and weeks.
    """

    def __init__(self):
        """Initialize feature validator."""
        self.logger = get_logger(__name__)

        # Validation parameters
        self.max_missing_rate = 0.8  # Maximum allowed missing data rate
        self.min_variance_threshold = 1e-8  # Minimum variance for features
        self.max_correlation_threshold = 0.95  # Maximum correlation between features
        self.outlier_z_threshold = 5.0  # Z-score threshold for outlier detection

        # Data leakage detection patterns
        self.leakage_keywords = [
            "closing",
            "final",
            "result",
            "outcome",
            "actual",
            "post_game",
            "final_score",
            "winner",
            "loser",
            "margin",
            "total_score",
        ]

        # Features that should be constant within games
        self.game_constant_features = [
            "home_team",
            "away_team",
            "season",
            "week",
            "game_id",
            "kickoff_et",
            "venue_id",
            "venue_roof_type",
        ]

        # Features that should have temporal consistency
        self.temporal_features = [
            "elo_home",
            "elo_away",
            "form_epa_home",
            "form_epa_away",
            "rest_days_home",
            "rest_days_away",
        ]

    def validate_features(
        self,
        features_df: pd.DataFrame,
        target_type: str = "all",
        season: int | None = None,
        week: int | None = None,
    ) -> dict[str, Any]:
        """
        Run comprehensive feature validation.

        Args:
            features_df: Feature matrix to validate
            target_type: Target type ('wp', 'ats', 'ou', 'all')
            season: Specific season to validate
            week: Specific week to validate

        Returns:
            Validation results dictionary
        """
        logger.info(
            "Running feature validation",
            features=len(features_df.columns),
            games=len(features_df),
            target_type=target_type,
        )

        validation_results = {
            "timestamp": datetime.now().isoformat(),
            "target_type": target_type,
            "season": season,
            "week": week,
            "total_features": len(features_df.columns),
            "total_games": len(features_df),
            "validation_passed": True,
            "errors": [],
            "warnings": [],
            "checks": {},
        }

        try:
            # 1. Data leakage detection
            leakage_results = self.check_data_leakage(features_df)
            validation_results["checks"]["data_leakage"] = leakage_results
            if not leakage_results["passed"]:
                validation_results["validation_passed"] = False
                validation_results["errors"].extend(leakage_results["errors"])

            # 2. Missing data validation
            missing_results = self.validate_missing_data(features_df)
            validation_results["checks"]["missing_data"] = missing_results
            if not missing_results["passed"]:
                validation_results["warnings"].extend(missing_results["warnings"])

            # 3. Feature distribution validation
            distribution_results = self.validate_feature_distributions(features_df)
            validation_results["checks"]["distributions"] = distribution_results
            if not distribution_results["passed"]:
                validation_results["warnings"].extend(distribution_results["warnings"])

            # 4. Feature correlation analysis
            correlation_results = self.analyze_feature_correlations(features_df)
            validation_results["checks"]["correlations"] = correlation_results
            if not correlation_results["passed"]:
                validation_results["warnings"].extend(correlation_results["warnings"])

            # 5. Temporal consistency validation
            temporal_results = self.validate_temporal_consistency(features_df)
            validation_results["checks"]["temporal_consistency"] = temporal_results
            if not temporal_results["passed"]:
                validation_results["warnings"].extend(temporal_results["warnings"])

            # 6. Feature completeness validation
            completeness_results = self.validate_feature_completeness(features_df)
            validation_results["checks"]["completeness"] = completeness_results
            if not completeness_results["passed"]:
                validation_results["warnings"].extend(completeness_results["warnings"])

            # 7. Statistical properties validation
            stats_results = self.validate_statistical_properties(features_df)
            validation_results["checks"]["statistical_properties"] = stats_results
            if not stats_results["passed"]:
                validation_results["warnings"].extend(stats_results["warnings"])

            logger.info(
                "Feature validation completed",
                passed=validation_results["validation_passed"],
                errors=len(validation_results["errors"]),
                warnings=len(validation_results["warnings"]),
            )

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Feature validation failed", error=str(e))
            validation_results["validation_passed"] = False
            validation_results["errors"].append(f"Validation exception: {e!s}")

        return validation_results

    def check_data_leakage(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """
        Check for potential data leakage in features.

        Detects:
        - Column names containing future information keywords
        - Features with impossible temporal relationships
        - Post-game information contamination

        Args:
            features_df: Feature matrix to check

        Returns:
            Data leakage validation results
        """
        logger.info("Checking for data leakage")

        results = {
            "passed": True,
            "errors": [],
            "warnings": [],
            "leakage_columns": [],
            "suspicious_columns": [],
        }

        # Check column names for leakage keywords
        for col in features_df.columns:
            col_lower = col.lower()

            # Critical leakage keywords
            for keyword in self.leakage_keywords:
                if keyword in col_lower:
                    results["leakage_columns"].append(col)
                    results["errors"].append(
                        f"CRITICAL: Column '{col}' contains leakage keyword '{keyword}'"
                    )
                    results["passed"] = False

        # Check for suspicious temporal patterns
        if "kickoff_et" in features_df.columns:
            current_time = datetime.now()

            # Check if any features reference future games
            future_games = (
                features_df[pd.to_datetime(features_df["kickoff_et"]) > current_time]
                if "kickoff_et" in features_df.columns
                else pd.DataFrame()
            )

            if len(future_games) > 0:
                # For future games, certain features should be missing or default
                future_team_stats = [
                    "form_epa_home",
                    "form_epa_away",
                    "form_success_rate_home",
                    "form_success_rate_away",
                ]

                for stat_col in future_team_stats:
                    if stat_col in future_games.columns:
                        non_null_future = future_games[stat_col].notna().sum()
                        if non_null_future > 0:
                            results["warnings"].append(
                                f"Future games have {stat_col} data ({non_null_future} games). "
                                f"Ensure this represents only past performance."
                            )

        # Check for impossible feature values
        impossible_checks = [
            ("elo_home", 800, 2200, "Elo ratings outside reasonable range"),
            ("elo_away", 800, 2200, "Elo ratings outside reasonable range"),
            ("rest_days_home", -1, 21, "Rest days outside possible range"),
            ("rest_days_away", -1, 21, "Rest days outside possible range"),
            ("travel_distance", 0, 3000, "Travel distance outside reasonable range"),
        ]

        for col_name, min_val, max_val, error_msg in impossible_checks:
            if col_name in features_df.columns:
                invalid_values = (
                    (features_df[col_name] < min_val)
                    | (features_df[col_name] > max_val)
                ).sum()

                if invalid_values > 0:
                    results["warnings"].append(
                        f"{error_msg}: {invalid_values} games with invalid {col_name} values"
                    )

        # Check for closing line contamination
        closing_patterns = ["closing_ml", "closing_spread", "closing_total", "closing_"]
        for pattern in closing_patterns:
            closing_cols = [
                col for col in features_df.columns if pattern in col.lower()
            ]
            if closing_cols:
                results["leakage_columns"].extend(closing_cols)
                results["errors"].append(
                    f"CRITICAL: Closing line data detected: {closing_cols}"
                )
                results["passed"] = False

        logger.info(
            "Data leakage check completed",
            leakage_columns=len(results["leakage_columns"]),
            passed=results["passed"],
        )

        return results

    def validate_missing_data(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """
        Validate missing data patterns in features.

        Args:
            features_df: Feature matrix to validate

        Returns:
            Missing data validation results
        """
        logger.info("Validating missing data patterns")

        results = {
            "passed": True,
            "warnings": [],
            "missing_rates": {},
            "high_missing_features": [],
            "missing_patterns": {},
        }

        # Calculate missing rates for each feature
        total_games = len(features_df)

        for col in features_df.columns:
            if col in self.game_constant_features:
                continue  # Skip ID columns

            missing_count = features_df[col].isna().sum()
            missing_rate = missing_count / total_games
            results["missing_rates"][col] = missing_rate

            if missing_rate > self.max_missing_rate:
                results["high_missing_features"].append(col)
                results["warnings"].append(
                    f"High missing data rate for '{col}': {missing_rate:.2%}"
                )
                results["passed"] = False

        # Analyze missing data patterns by season/week
        if "season" in features_df.columns and "week" in features_df.columns:
            for season in features_df["season"].unique():
                season_data = features_df[features_df["season"] == season]
                season_missing = {}

                for col in features_df.columns:
                    if col not in self.game_constant_features:
                        season_missing_rate = season_data[col].isna().sum() / len(
                            season_data
                        )
                        if season_missing_rate > 0.5:  # More than 50% missing
                            season_missing[col] = season_missing_rate

                if season_missing:
                    results["missing_patterns"][f"season_{season}"] = season_missing

        # Check for systematic missing data (e.g., all weather data missing for indoor games)
        if "venue_roof_type" in features_df.columns:
            indoor_games = features_df[features_df["venue_roof_type"] == "indoor"]
            if len(indoor_games) > 0:
                weather_cols = [
                    col for col in features_df.columns if "weather" in col.lower()
                ]
                for weather_col in weather_cols:
                    if weather_col in indoor_games.columns:
                        indoor_weather_present = indoor_games[weather_col].notna().sum()
                        if indoor_weather_present > 0:
                            results["warnings"].append(
                                f"Indoor games have weather data for '{weather_col}' "
                                f"({indoor_weather_present} games). This may be incorrect."
                            )

        logger.info(
            "Missing data validation completed",
            high_missing_features=len(results["high_missing_features"]),
            passed=results["passed"],
        )

        return results

    def validate_feature_distributions(
        self, features_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Validate statistical distributions of features.

        Args:
            features_df: Feature matrix to validate

        Returns:
            Distribution validation results
        """
        logger.info("Validating feature distributions")

        results = {
            "passed": True,
            "warnings": [],
            "distribution_stats": {},
            "outlier_features": [],
            "constant_features": [],
            "skewed_features": [],
        }

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in self.game_constant_features
        ]

        for col in numeric_cols:
            col_data = features_df[col].dropna()

            if len(col_data) == 0:
                continue

            # Calculate basic statistics
            stats = {
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "max": float(col_data.max()),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
            }
            results["distribution_stats"][col] = stats

            # Check for constant features (zero variance)
            if stats["std"] < self.min_variance_threshold:
                results["constant_features"].append(col)
                results["warnings"].append(
                    f"Feature '{col}' has near-zero variance: {stats['std']:.2e}"
                )

            # Check for extreme skewness
            if abs(stats["skewness"]) > 3.0:
                results["skewed_features"].append(col)
                results["warnings"].append(
                    f"Feature '{col}' is highly skewed: {stats['skewness']:.2f}"
                )

            # Check for outliers using Z-score
            if stats["std"] > 0:
                z_scores = np.abs((col_data - stats["mean"]) / stats["std"])
                outlier_count = (z_scores > self.outlier_z_threshold).sum()
                outlier_rate = outlier_count / len(col_data)

                if outlier_rate > 0.05:  # More than 5% outliers
                    results["outlier_features"].append(col)
                    results["warnings"].append(
                        f"Feature '{col}' has {outlier_rate:.2%} extreme outliers "
                        f"(Z-score > {self.outlier_z_threshold})"
                    )

        # Check for reasonable ranges in specific feature types
        range_checks = [
            ("elo_", 1200, 1800, "Elo ratings"),
            ("form_epa_", -0.5, 0.5, "EPA per play"),
            ("form_success_rate_", 0.2, 0.8, "Success rates"),
            ("weather_wind_mph", 0, 50, "Wind speed"),
            ("weather_temp_f", -20, 120, "Temperature"),
        ]

        for prefix, min_expected, max_expected, feature_type in range_checks:
            matching_cols = [col for col in numeric_cols if prefix in col]
            for col in matching_cols:
                col_data = features_df[col].dropna()
                if len(col_data) > 0:
                    if col_data.min() < min_expected or col_data.max() > max_expected:
                        results["warnings"].append(
                            f"{feature_type} '{col}' has unusual range: "
                            f"[{col_data.min():.3f}, {col_data.max():.3f}], "
                            f"expected: [{min_expected}, {max_expected}]"
                        )

        if len(results["warnings"]) > 10:  # Too many warnings
            results["passed"] = False

        logger.info(
            "Feature distribution validation completed",
            constant_features=len(results["constant_features"]),
            outlier_features=len(results["outlier_features"]),
            passed=results["passed"],
        )

        return results

    def analyze_feature_correlations(self, features_df: pd.DataFrame) -> dict[str, Any]:
        """
        Analyze correlations between features.

        Args:
            features_df: Feature matrix to analyze

        Returns:
            Correlation analysis results
        """
        logger.info("Analyzing feature correlations")

        results = {
            "passed": True,
            "warnings": [],
            "high_correlations": [],
            "correlation_groups": {},
        }

        # Select numeric features for correlation analysis
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in self.game_constant_features
        ]

        if len(numeric_cols) < 2:
            logger.warning("Insufficient numeric features for correlation analysis")
            return results

        # Calculate correlation matrix
        correlation_matrix = features_df[numeric_cols].corr()

        # Find high correlations (excluding self-correlations)
        high_corr_pairs = []

        for i in range(len(correlation_matrix.columns)):
            for j in range(i + 1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]

                if (
                    not np.isnan(corr_value)
                    and abs(corr_value) > self.max_correlation_threshold
                ):
                    feature1 = correlation_matrix.columns[i]
                    feature2 = correlation_matrix.columns[j]
                    high_corr_pairs.append((feature1, feature2, corr_value))

        results["high_correlations"] = high_corr_pairs

        # Generate warnings for highly correlated features
        for feature1, feature2, corr_value in high_corr_pairs:
            results["warnings"].append(
                f"High correlation between '{feature1}' and '{feature2}': {corr_value:.3f}"
            )

        # Group features by correlation clusters
        # This helps identify redundant feature groups
        processed_features = set()
        cluster_id = 0

        for feature1, feature2, _corr_value in high_corr_pairs:
            if (
                feature1 not in processed_features
                and feature2 not in processed_features
            ):
                cluster_name = f"cluster_{cluster_id}"
                results["correlation_groups"][cluster_name] = [feature1, feature2]
                processed_features.add(feature1)
                processed_features.add(feature2)
                cluster_id += 1
            elif feature1 in processed_features:
                # Add feature2 to existing cluster
                for _cluster_name, cluster_features in results[
                    "correlation_groups"
                ].items():
                    if feature1 in cluster_features:
                        cluster_features.append(feature2)
                        processed_features.add(feature2)
                        break
            elif feature2 in processed_features:
                # Add feature1 to existing cluster
                for _cluster_name, cluster_features in results[
                    "correlation_groups"
                ].items():
                    if feature2 in cluster_features:
                        cluster_features.append(feature1)
                        processed_features.add(feature1)
                        break

        if len(high_corr_pairs) > 5:  # Many highly correlated features
            results["warnings"].append(
                f"Many highly correlated feature pairs detected ({len(high_corr_pairs)}). "
                "Consider feature selection or dimensionality reduction."
            )

        logger.info(
            "Feature correlation analysis completed",
            high_correlations=len(high_corr_pairs),
            correlation_groups=len(results["correlation_groups"]),
            passed=results["passed"],
        )

        return results

    def validate_temporal_consistency(
        self, features_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Validate temporal consistency of features across seasons and weeks.

        Args:
            features_df: Feature matrix to validate

        Returns:
            Temporal consistency validation results
        """
        logger.info("Validating temporal consistency")

        results = {
            "passed": True,
            "warnings": [],
            "temporal_breaks": [],
            "season_jumps": {},
            "week_progression": {},
        }

        if "season" not in features_df.columns or "week" not in features_df.columns:
            results["warnings"].append(
                "No season/week columns found for temporal validation"
            )
            return results

        # Check for logical week progression within seasons
        for season in sorted(features_df["season"].unique()):
            season_data = features_df[features_df["season"] == season].copy()
            weeks = sorted(season_data["week"].unique())

            # Check for missing weeks
            expected_weeks = list(range(min(weeks), max(weeks) + 1))
            missing_weeks = [w for w in expected_weeks if w not in weeks]

            if missing_weeks:
                results["warnings"].append(
                    f"Season {season} missing weeks: {missing_weeks}"
                )

            # Check temporal features for consistency
            for temp_feature in self.temporal_features:
                if temp_feature in season_data.columns:
                    # Features should generally progress smoothly
                    for team in ["home", "away"]:
                        if (
                            f"{temp_feature.split('_')[0]}_{team}"
                            in season_data.columns
                        ):
                            feature_col = f"{temp_feature.split('_')[0]}_{team}"

                            # Check for sudden jumps between weeks
                            for i, week in enumerate(weeks[1:], 1):
                                prev_week_data = season_data[
                                    season_data["week"] == weeks[i - 1]
                                ][feature_col].dropna()
                                curr_week_data = season_data[
                                    season_data["week"] == week
                                ][feature_col].dropna()

                                if len(prev_week_data) > 0 and len(curr_week_data) > 0:
                                    prev_mean = prev_week_data.mean()
                                    curr_mean = curr_week_data.mean()

                                    # Check for large jumps (more than 2 standard deviations)
                                    if (
                                        abs(curr_mean - prev_mean)
                                        > 2 * prev_week_data.std()
                                    ):
                                        jump_info = {
                                            "season": season,
                                            "from_week": weeks[i - 1],
                                            "to_week": week,
                                            "feature": feature_col,
                                            "jump_magnitude": abs(
                                                curr_mean - prev_mean
                                            ),
                                        }
                                        results["temporal_breaks"].append(jump_info)

        # Check for season-to-season carryover patterns
        seasons = sorted(features_df["season"].unique())
        for i, season in enumerate(seasons[1:], 1):
            prev_season = seasons[i - 1]

            # Check final week of previous season vs first week of current season
            prev_season_final = features_df[
                (features_df["season"] == prev_season)
                & (
                    features_df["week"]
                    == features_df[features_df["season"] == prev_season]["week"].max()
                )
            ]

            curr_season_first = features_df[
                (features_df["season"] == season)
                & (
                    features_df["week"]
                    == features_df[features_df["season"] == season]["week"].min()
                )
            ]

            # Check Elo ratings carryover (should have some continuity)
            elo_features = [col for col in features_df.columns if "elo_" in col]
            for elo_col in elo_features:
                if (
                    elo_col in prev_season_final.columns
                    and elo_col in curr_season_first.columns
                ):
                    prev_elos = prev_season_final[elo_col].dropna()
                    curr_elos = curr_season_first[elo_col].dropna()

                    if len(prev_elos) > 0 and len(curr_elos) > 0:
                        # Elo ratings should regress toward mean between seasons
                        prev_mean = prev_elos.mean()
                        curr_mean = curr_elos.mean()

                        # Expected regression toward 1500
                        expected_regression = (
                            prev_mean - 1500
                        ) * 0.75 + 1500  # 25% regression
                        actual_change = abs(curr_mean - expected_regression)

                        if (
                            actual_change > 100
                        ):  # Large deviation from expected regression
                            results["season_jumps"][f"{prev_season}_to_{season}"] = {
                                "feature": elo_col,
                                "prev_mean": prev_mean,
                                "curr_mean": curr_mean,
                                "expected_mean": expected_regression,
                                "deviation": actual_change,
                            }

        # Generate warnings for temporal inconsistencies
        if len(results["temporal_breaks"]) > 0:
            results["warnings"].append(
                f"Found {len(results['temporal_breaks'])} temporal feature breaks"
            )

        if len(results["season_jumps"]) > 0:
            results["warnings"].append(
                f"Found {len(results['season_jumps'])} unusual season carryover patterns"
            )

        logger.info(
            "Temporal consistency validation completed",
            temporal_breaks=len(results["temporal_breaks"]),
            season_jumps=len(results["season_jumps"]),
            passed=results["passed"],
        )

        return results

    def validate_feature_completeness(
        self, features_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Validate completeness of expected features.

        Args:
            features_df: Feature matrix to validate

        Returns:
            Completeness validation results
        """
        logger.info("Validating feature completeness")

        results = {
            "passed": True,
            "warnings": [],
            "missing_feature_groups": [],
            "expected_features": {},
            "actual_features": list(features_df.columns),
        }

        # Define expected feature groups
        expected_feature_groups = {
            "basic_game_info": ["game_id", "season", "week", "home_team", "away_team"],
            "elo_features": ["elo_home", "elo_away", "elo_diff", "elo_home_advantage"],
            "team_form_features": [
                "form_epa_home",
                "form_epa_away",
                "form_success_rate_home",
                "form_success_rate_away",
            ],
            "contextual_features": [
                "rest_days_home",
                "rest_days_away",
                "travel_distance",
                "timezone_diff",
            ],
            "venue_features": ["venue_roof_type", "is_home_advantage"],
        }

        # Optional feature groups (may not always be present)
        optional_feature_groups = {
            "weather_features": [
                "weather_wind_mph",
                "weather_temp_f",
                "weather_precip_prob",
            ],
            "market_features": [
                "opening_ml_home",
                "snapshot_ml_home",
                "fair_ml_prob_home",
            ],
            "advanced_elo": ["elo_uncertainty_home", "elo_uncertainty_away"],
        }

        # Check for presence of expected feature groups
        for group_name, expected_features in expected_feature_groups.items():
            results["expected_features"][group_name] = expected_features
            missing_features = [
                f for f in expected_features if f not in features_df.columns
            ]

            if missing_features:
                results["missing_feature_groups"].append(group_name)
                results["warnings"].append(f"Missing {group_name}: {missing_features}")
                results["passed"] = False

        # Check optional features (warnings only)
        for group_name, optional_features in optional_feature_groups.items():
            missing_optional = [
                f for f in optional_features if f not in features_df.columns
            ]
            if missing_optional:
                results["warnings"].append(
                    f"Optional {group_name} missing: {missing_optional}"
                )

        # Check for target-specific features
        target_indicators = ["wp_", "ats_", "ou_"]
        target_features_found = {}

        for target in target_indicators:
            target_cols = [col for col in features_df.columns if target in col]
            target_features_found[target.rstrip("_")] = target_cols

        results["target_features"] = target_features_found

        # Validate feature naming conventions
        unconventional_names = []
        for col in features_df.columns:
            # Check for spaces, special characters, or unusual patterns
            if " " in col or any(
                char in col for char in ["@", "#", "$", "%", "&", "*"]
            ):
                unconventional_names.append(col)

        if unconventional_names:
            results["warnings"].append(
                f"Unconventional feature names: {unconventional_names[:5]}"  # Show first 5
            )

        logger.info(
            "Feature completeness validation completed",
            missing_groups=len(results["missing_feature_groups"]),
            passed=results["passed"],
        )

        return results

    def validate_statistical_properties(
        self, features_df: pd.DataFrame
    ) -> dict[str, Any]:
        """
        Validate statistical properties of features.

        Args:
            features_df: Feature matrix to validate

        Returns:
            Statistical properties validation results
        """
        logger.info("Validating statistical properties")

        results = {
            "passed": True,
            "warnings": [],
            "feature_stats": {},
            "normalization_issues": [],
            "scaling_recommendations": [],
        }

        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        numeric_cols = [
            col for col in numeric_cols if col not in self.game_constant_features
        ]

        for col in numeric_cols:
            col_data = features_df[col].dropna()

            if len(col_data) == 0:
                continue

            # Calculate comprehensive statistics
            stats = {
                "count": len(col_data),
                "mean": float(col_data.mean()),
                "std": float(col_data.std()),
                "min": float(col_data.min()),
                "q25": float(col_data.quantile(0.25)),
                "median": float(col_data.median()),
                "q75": float(col_data.quantile(0.75)),
                "max": float(col_data.max()),
                "skewness": float(col_data.skew()),
                "kurtosis": float(col_data.kurtosis()),
                "unique_values": len(col_data.unique()),
                "zero_rate": (col_data == 0).sum() / len(col_data),
            }

            results["feature_stats"][col] = stats

            # Check for normalization issues
            if "normalized" in col or "_norm" in col or "_z" in col:
                # Should have mean close to 0 and std close to 1
                if abs(stats["mean"]) > 0.1:
                    results["normalization_issues"].append(
                        f"Normalized feature '{col}' has mean {stats['mean']:.3f} (expected ~0)"
                    )

                if abs(stats["std"] - 1.0) > 0.2:
                    results["normalization_issues"].append(
                        f"Normalized feature '{col}' has std {stats['std']:.3f} (expected ~1)"
                    )

            # Check for features that might need scaling
            if abs(stats["max"]) > 1000 or abs(stats["min"]) > 1000:
                results["scaling_recommendations"].append(
                    f"Feature '{col}' has large scale (range: {stats['min']:.0f} to {stats['max']:.0f}). "
                    "Consider scaling for model training."
                )

            # Check for binary features that aren't properly encoded
            if stats["unique_values"] == 2:
                unique_vals = sorted(col_data.unique())
                if not (set(unique_vals) == {0, 1} or set(unique_vals) == {0.0, 1.0}):
                    results["warnings"].append(
                        f"Binary feature '{col}' not in 0/1 encoding: {unique_vals}"
                    )

            # Check for high zero rates (sparse features)
            if stats["zero_rate"] > 0.8:
                results["warnings"].append(
                    f"Feature '{col}' is very sparse: {stats['zero_rate']:.1%} zeros"
                )

        # Overall recommendations
        if len(results["scaling_recommendations"]) > 5:
            results["warnings"].append(
                f"Many features ({len(results['scaling_recommendations'])}) may need scaling"
            )

        if len(results["normalization_issues"]) > 0:
            results["passed"] = False

        logger.info(
            "Statistical properties validation completed",
            normalization_issues=len(results["normalization_issues"]),
            scaling_recommendations=len(results["scaling_recommendations"]),
            passed=results["passed"],
        )

        return results

    def generate_validation_report(
        self, validation_results: dict[str, Any], output_path: str | None = None
    ) -> str:
        """
        Generate a comprehensive validation report.

        Args:
            validation_results: Results from validate_features()
            output_path: Optional path to save report

        Returns:
            Validation report as string
        """
        logger.info("Generating validation report")

        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("NFL PREDICTION FEATURES VALIDATION REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {validation_results['timestamp']}")
        report_lines.append(f"Target Type: {validation_results['target_type']}")
        report_lines.append(f"Season: {validation_results.get('season', 'All')}")
        report_lines.append(f"Week: {validation_results.get('week', 'All')}")
        report_lines.append(f"Total Features: {validation_results['total_features']}")
        report_lines.append(f"Total Games: {validation_results['total_games']}")
        report_lines.append("")

        # Overall validation status
        status = "[PASS]" if validation_results["validation_passed"] else "[FAIL]"
        report_lines.append(f"VALIDATION STATUS: {status}")
        report_lines.append(f"Errors: {len(validation_results['errors'])}")
        report_lines.append(f"Warnings: {len(validation_results['warnings'])}")
        report_lines.append("")

        # Critical errors
        if validation_results["errors"]:
            report_lines.append("CRITICAL ERRORS:")
            report_lines.append("-" * 40)
            for error in validation_results["errors"]:
                report_lines.append(f"[ERROR] {error}")
            report_lines.append("")

        # Warnings
        if validation_results["warnings"]:
            report_lines.append("WARNINGS:")
            report_lines.append("-" * 40)
            for warning in validation_results["warnings"][:20]:  # Show first 20
                report_lines.append(f"[WARN] {warning}")
            if len(validation_results["warnings"]) > 20:
                report_lines.append(
                    f"... and {len(validation_results['warnings']) - 20} more warnings"
                )
            report_lines.append("")

        # Detailed check results
        for check_name, check_results in validation_results["checks"].items():
            report_lines.append(f"{check_name.upper().replace('_', ' ')}:")
            report_lines.append("-" * 40)

            if isinstance(check_results, dict):
                if "passed" in check_results:
                    status = "[PASS]" if check_results["passed"] else "[FAIL]"
                    report_lines.append(f"Status: {status}")

                # Add specific details for each check type
                if check_name == "data_leakage" and "leakage_columns" in check_results:
                    if check_results["leakage_columns"]:
                        report_lines.append(
                            f"Leakage columns detected: {check_results['leakage_columns']}"
                        )
                    else:
                        report_lines.append("No data leakage detected [OK]")

                elif (
                    check_name == "missing_data"
                    and "high_missing_features" in check_results
                ):
                    if check_results["high_missing_features"]:
                        report_lines.append(
                            f"High missing features: {len(check_results['high_missing_features'])}"
                        )
                    else:
                        report_lines.append("Missing data rates acceptable [OK]")

                elif (
                    check_name == "correlations"
                    and "high_correlations" in check_results
                ):
                    if check_results["high_correlations"]:
                        report_lines.append(
                            f"High correlations found: {len(check_results['high_correlations'])}"
                        )
                        for feat1, feat2, corr in check_results["high_correlations"][
                            :5
                        ]:
                            report_lines.append(f"  {feat1} <-> {feat2}: {corr:.3f}")
                    else:
                        report_lines.append("No excessive feature correlations [OK]")

                elif (
                    check_name == "distributions"
                    and "constant_features" in check_results
                ):
                    if check_results["constant_features"]:
                        report_lines.append(
                            f"Constant features: {check_results['constant_features']}"
                        )
                    if check_results["outlier_features"]:
                        report_lines.append(
                            f"Features with outliers: {len(check_results['outlier_features'])}"
                        )

            report_lines.append("")

        # Summary and recommendations
        report_lines.append("RECOMMENDATIONS:")
        report_lines.append("-" * 40)

        if validation_results["validation_passed"]:
            report_lines.append(
                "[PASS] Features passed validation and are ready for model training"
            )
        else:
            report_lines.append(
                "[FAIL] Features FAILED validation - address errors before proceeding"
            )

        if len(validation_results["warnings"]) > 0:
            report_lines.append("[WARN] Review warnings for potential improvements")

        report_lines.append("")
        report_lines.append("=" * 80)

        report_text = "\n".join(report_lines)

        # Save report if path provided
        if output_path:
            Path(output_path).write_text(report_text, encoding="utf-8")
            logger.info(f"Validation report saved to {output_path}")

        return report_text
