"""Data Quality Assurance and monitoring for NFL prediction system."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from conf.season_partition import LATEST_COMPLETED_SEASON
from conf.settings import get_settings
from data.storage import get_database_stats, get_db_connection, load_dataframe
from utils import (
    get_current_nfl_week,
    get_logger,
    validate_data_quality,
    validate_game_data,
    validate_nfl_business_rules,
    validate_odds_data,
    validate_temporal_consistency,
)
from utils.exceptions import DataIngestionError, DataValidationError
from utils.game_id_utils import parse_game_id
from utils.team_data import get_all_teams, normalize_team_abbreviation

logger = get_logger(__name__)

# Gold-layer feature matrices and their expected column counts (AUDIT-02).
# Verified on-disk 2026-08-22 after the Phase 30 rung-3 narrowing: features_wp
# 194, features_ats 195, features_ou 194. The single-column ATS difference is its
# extra target/margin columns (target_ats, home_margin, point_differential).
#
# The record below is ADDED TO, never overwritten: each phase's paragraph states
# what IT moved, so the width's history reads in order.
#
# Phase 30 NARROWED each matrix by -15 columns, back to the Phase-28 widths
# (SPEC R3, D29-07-01, Plan 30-07 -- rung 3 of the D30-17 rebuild ladder). The
# removed set is exactly the fifteen Phase-29 line-movement columns listed in the
# Phase-29 paragraph below. This is the FIRST rebuild in this project's history
# that removes columns rather than adding them, which is why the gold write now
# passes replace_mode=True: the append path's concat unions columns and would
# have written the fifteen back as all-null. The family's removal is a deliberate
# decision about what belongs in gold; the paid odds_timeline archive,
# features/line_movement.py and the line_movement group registration all remain.
# Only line_movement physically leaves gold (D30-03) -- any group the Phase-30
# Stage-1 gate drops or leaves undetermined keeps its columns here and is excluded
# at TRAIN time instead, which is what makes 194/195/194 the correct width.
#
# Phase 29 had widened each matrix by +15 columns vs the 194/195/194 Phase-28
# baseline (SIG-04, Plan 29-06): the seven D-09 totals line-movement features
# (opening_total, total_drift, total_drift_dir, total_late_drift,
# total_abs_travel, total_reversals, total_range), the shared
# line_movement_coverage flag, and the seven Tier (a) spread siblings
# (opening_spread, spread_drift, spread_drift_dir, spread_late_drift,
# spread_abs_travel, spread_reversals, spread_range). These are GAME-level
# columns (a line trajectory belongs to the game), so unlike the Phase-28
# families they are not home_/away_-expanded.
#
# Phase 28 had widened each matrix by +38 columns vs the prior 156/157/156
# baseline (SIG-06, Plan 28-06): +20 snap columns (home_/away_ x
# {snap_continuity, snap_concentration, rolling_snap_share_{db,dl,lb,ol,qb,rb,
# te,wr}}), +12 injury columns (home_/away_ x {qb_out_flag,
# backup_quality_delta, availability_fraction, injury_coverage,
# availability_coverage, date_modified_coverage}), and +6 contextual
# situational-spot columns (home_/away_ x {look_ahead_spot, letdown_spot,
# off_bye}).
#
# These counts are DELIBERATELY reconciled to the real rebuilt widths
# (IN-03, Pitfall 6 -- count empirically, never silence). A width that moves for a
# reason nobody can name is the finding; widening a tolerance to make it pass is
# the failure this tripwire exists to prevent. The companion assertion in
# tests/unit/test_data_qa_gold_width.py pins the DELTA to a named column set, so a
# build that removed one intended column while incidentally adding an unrelated
# one cannot satisfy the integer alone.
# PHASE 33.1 (Plan 33.1-07 Task 3, rebuilt 2026-09-14) moves each matrix by
# exactly +1, from 194/195/194 to 195/196/195. The added column is
# ``weather_coverage``, and the delta is pinned to that NAME in
# tests/unit/test_data_qa_gold_width.py rather than to the integer alone -- a
# build that added an unrelated column while omitting the flag satisfies +1
# exactly as well as the right one does.
#
# UPDATED, NOT SILENCED. This tripwire's own failure message instructs the reader
# to reconcile it to the new empirically-counted width with a reason on the
# record, and that is what happened here: the widths below were COUNTED off the
# rebuilt matrices, and the rung that moved them is attributed under its own
# document prefix (``p331_``) with a compound cause declared in committed source
# BEFORE the rebuild ran.
# PHASE 33.2 RUNG 4 (Plan 33.2-12 Task 3, rebuilt 2026-09-21) moves each matrix by
# exactly -2, from 195/196/195 to 193/194/193. The removed columns are ``precip_mm``
# and ``raw_precip_mm``: history's weather is now the archived day-before NWS MOS
# bulletin, which carries no millimetre amount, so both are forecast-less inputs and
# leave gold through the ``weather_unsupplied`` registry group. The widths below were
# PREDICTED in tests/integration/test_p332_rung4_attribution.py before the rebuild and
# written here only after the rebuilt matrices MEASURED equal to that prediction; the
# delta is pinned to those two NAMES by the ``P332_12_GOLD_WIDTH_DELTA`` manifest slot
# (tests/phase33_state.py), which tests/unit/test_data_qa_gold_width.py discovers.
# PHASE 33.2 RUNG 7 (Plan 33.2-16 Task 3, rebuilt 2026-09-22) moves each matrix by
# exactly +4, from 193/194/193 to 197/198/197. The added columns are the four per-side
# opponent-adjustment coverage flags ``home_off_rolling_opp_adj_coverage``,
# ``home_def_rolling_opp_adj_coverage``, ``away_off_rolling_opp_adj_coverage`` and
# ``away_def_rolling_opp_adj_coverage`` (features.opponent_adj.OPP_ADJ_COVERAGE_COLUMN):
# the opponent adjustment now actually runs, and where it cannot reach, its value is NaN
# beside a flag at 0.0 instead of raw EPA wearing an adjusted name. The widths below were
# PREDICTED in tests/integration/test_p332_rung7_attribution.py before the rebuild and
# written here only after the rebuilt matrices MEASURED equal to that prediction; the
# delta is pinned to those four NAMES by the ``P332_16_GOLD_WIDTH_DELTA`` manifest slot.
# PHASE 33.2 RUNG 8 (Plan 33.2-18 Task 3, rebuilt 2026-09-22) moves each matrix by
# exactly +4, from 197/198/197 to 201/202/201. The added columns are two coverage flags
# for the source-limited team-form metric, ``home_off_rolling_cpoe_coverage`` and
# ``away_off_rolling_cpoe_coverage`` (the pinned play-by-play carries no completion
# probability before 2006), and two snap coverage flags, ``home_snap_coverage`` and
# ``away_snap_coverage`` (no snap counts before 2013): the coverage floors are retired, so a
# value before its family's coverage is NaN beside a flag at 0.0 rather than a 0.0 that reads
# as "exactly average". The widths below were PREDICTED in
# tests/integration/test_p332_rung8_attribution.py before the rebuild and written here only
# after the rebuilt matrices MEASURED equal to that prediction; the delta is pinned to those
# four NAMES by the ``P332_18_RUNG8_GOLD_WIDTH_DELTA`` manifest slot.
# PHASE 33.2 EXTRA STEP 8e (Plan 33.2-19, orchestrator-assigned; owner ruling 2026-09-22,
# rebuilt 2026-09-22) moves each matrix by exactly -8, from 201/202/201 to 193/194/193. The
# removed columns are the eight NEVER-POPULATED DEFENSIVE COPIES of the four offence-only
# team-form metrics: ``{home,away}_def_rolling_cpoe``,
# ``{home,away}_def_rolling_avg_drive_start_yardline``,
# ``{home,away}_def_rolling_neutral_pace`` and
# ``{home,away}_def_rolling_neutral_pass_rate``. ``features/team_form.py`` writes NaN for
# them on a defence row by design, so they had never held a measured value in any season and
# gold carried them as a flat 0.0 that a model reads as "exactly average". The widths below
# were PREDICTED in tests/integration/test_p332_step8e_attribution.py before the rebuild and
# written here only after the rebuilt matrices MEASURED equal to that prediction; the delta
# is pinned to those eight NAMES by the ``P332_19_STEP8E_GOLD_WIDTH_DELTA`` manifest slot.
# The OFFENSIVE copies are untouched: the metrics are offence-only, not absent.
# PHASE 33.2 RUNG 9 (Plan 33.2-19 Task 3, rebuilt 2026-09-22) is the LAST width-moving rung
# of the ladder, so the widths below are the FINAL ones: 193/194/193 to 188/188/187. Every
# matrix loses the five market-line columns ``snapshot_spread``, ``snapshot_total``,
# ``snapshot_ml_prob_home_fair``, ``spread_movement`` and ``total_movement`` -- no betting
# line is a model input for any target (D33.2-03), and the exclusion comes from the ONE
# registry group ``market``. THE DELTA IS NOT UNIFORM, which is the one place this rung
# departs from its plan's text: ``features_ats`` also loses ``target_ats`` and
# ``features_ou`` also loses ``target_ou``, the line-derived target columns that were
# arithmetic children of exactly the columns removed above (``home_covered_spread`` and
# ``game_went_over`` were their boolean children and never reached a matrix). The ats and
# ou matrices are now selected on ``home_margin`` and ``total_points``, their trainers' own
# targets, so no row moved. The widths below were PREDICTED in
# tests/integration/test_p332_rung9_attribution.py before the rebuild and written here only
# after the rebuilt matrices MEASURED equal to that prediction; the delta is pinned to those
# NAMES, per matrix, by the ``P332_19_RUNG9_GOLD_WIDTH_DELTA`` manifest slot. Plan 33.2-20
# ASSERTS this pin against its clean build and re-pins nothing.
GOLD_FEATURE_MATRICES = {
    "features_wp": 188,
    "features_ats": 188,
    "features_ou": 187,
}

# Tables whose DuckDB and parquet copies must agree on row-set MEMBERSHIP (D30-18).
#
# Data, not logic: a future table is added here without touching
# ``check_duckdb_parquet_consistency``. Seeded with silver ``games``, the table the
# N-01 divergence was measured on (6,499 parquet rows against 6,292 in DuckDB at
# Phase-30 start, all 207 of them season 2025).
_DUCKDB_PARQUET_CONSISTENCY_TABLES: dict[str, dict[str, str]] = {
    "games": {"layer": "silver", "key": "game_id"},
}

# A table that is ABSENT from one of the two stores is not a consistency FAILURE -- it
# is a check that could not run, and it is reported as not_applicable.
#
# WR-06: that degradation must be reserved for genuine ABSENCE. This tuple used to also
# catch ``duckdb.Error``, ``OSError``, ``ValueError``, ``KeyError`` and ``TypeError``,
# and ``generate_quality_report`` skips ``not_applicable`` when counting -- so a locked
# or corrupt DuckDB file, the precise failure this guard exists to detect, produced a QA
# report with the check silently ABSENT from the totals and no fail recorded.
# ``TypeError`` and ``KeyError`` are programming errors rather than "the table is not in
# this store" signals, so a bug INSIDE the guard was indistinguishable from a
# legitimately inapplicable check.
#
# The two names below are what "this table is not in that store" actually raises:
# ``load_dataframe`` raises ``DataIngestionError`` for a table it cannot resolve in
# either store, and a directly-connected or injected reader raises ``FileNotFoundError``.
# Anything else is counted as a FAILURE by ``_COPY_READ_FAILURES``: a guard that cannot
# run is not a guard that passed.
_TABLE_ABSENT_ERRORS = (
    DataIngestionError,
    FileNotFoundError,
)

# Everything the read or the key access can raise that is NOT an absence signal. Caught
# so one broken table cannot abort the whole quality report, but recorded as a FAIL.
_COPY_READ_FAILURES = (
    duckdb.Error,
    OSError,
    KeyError,
    ValueError,
    TypeError,
)

# A QA report that inlines two hundred missing ids is a report nobody reads. The
# COUNT is exact; the id list is a bounded sample for diagnosis.
_CONSISTENCY_SAMPLE_LIMIT = 20

# Tables produced by a pipeline step registered AFTER ``data_qa``, mapped to that step's name.
#
# WHY A TABLE-LEVEL CHECK ON THESE IS NOT APPLICABLE HERE (Plan 33-18, owner ruling R1 of
# 2026-09-15). ``step_data_qa`` is the fourth DATA step; ``ingest_odds`` is the first
# PREDICTIONS step. So at the instant this gate runs, the odds are ALWAYS last week's, and a
# freshness or completeness demand on them refuses every correct Friday. Plan 33-18's live
# acceptance run, attempt 1, counted exactly that as one of its five failures.
#
# NOT COUNTED IS NOT UNREPORTED. The freshness check still measures and records the age; it
# only stops counting it toward the gate. The step name is pinned against the registry by
# ``tests/unit/test_data_qa_live_friday_checks.py``, so if the registry ever moves the
# producer ahead of ``data_qa``, that test fails and names this exemption for removal.
_TABLES_PRODUCED_AFTER_DATA_QA: dict[str, str] = {
    "odds_snapshot": "ingest_odds",
}

# Last season for which gold is considered fully ingested. Seasons beyond this are
# treated as expected, documented trailing-coverage gaps (D-05), NOT failures.
#
# DERIVED from the committed season rule (review WR-14). It was the literal 2024,
# written when 2025 was ingested through about week 4 only. Phase 33.1 completed
# 2025 -- gold holds all of 2002-2025 and the rule's holdout is 2024-2025 -- so the
# literal had become a licence to ignore a REAL coverage gap in the most recent
# completed season, which is the opposite of what it is for.
GOLD_LAST_COMPLETE_SEASON = LATEST_COMPLETED_SEASON


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
            "games": {
                "layer": "silver",
                "validator": validate_game_data,
                "business_rules": "games",
                "required_columns": [
                    "game_id",
                    "season",
                    "week",
                    "home_team",
                    "away_team",
                ],
            },
            "odds_snapshot": {
                "layer": "silver",
                "validator": validate_odds_data,
                "business_rules": "odds",
                "required_columns": ["game_id", "snapshot_ts", "sportsbook"],
            },
            # silver ``weather`` -- the table ``scripts/ingest_weather.py`` actually writes.
            #
            # IT WAS ``weather_forecast`` UNTIL PLAN 33-18 (owner ruling R1, 2026-09-15): a
            # 14-row legacy DuckDB-only table last written 2025-10-07, which the live
            # ingest never touches. So the gate watched nothing the run does, and could not
            # see that attempt 1's weather ingest logged "No games found", wrote NOTHING and
            # reported success.
            #
            # ``freshness_column`` is declared because this table has no season/week and
            # carries ``game_time`` -- a KICKOFF -- among its time-like columns. Maxing over
            # all of them reads an upcoming kickoff as the moment the ingest wrote, so stale
            # weather would read fresh forever. ``created_at`` is the write time.
            "weather": {
                "layer": "silver",
                "validator": None,  # Use general validation
                "business_rules": None,
                "required_columns": ["game_id", "forecast_time", "is_outdoor"],
                "freshness_column": "created_at",
            },
            "venues": {
                "layer": "silver",
                "validator": None,
                "business_rules": None,
                "required_columns": ["venue_id", "venue_name", "home_teams"],
            },
        }

    def check_data_freshness(
        self, table_name: str, max_age_hours: int = 24
    ) -> dict[str, Any]:
        """Check if data is fresh (recently updated).

        A table produced by a step registered AFTER ``data_qa`` is still MEASURED, and its
        result is marked ``not_applicable`` with the producing step named, so the age is on
        the record without refusing a correct run. See ``_TABLES_PRODUCED_AFTER_DATA_QA``.
        """
        result = self._measure_data_freshness(table_name, max_age_hours)
        producer = _TABLES_PRODUCED_AFTER_DATA_QA.get(table_name)
        if producer is not None:
            result["measured_status"] = result["status"]
            result["status"] = "not_applicable"
            result["not_applicable_reason"] = (
                f"{table_name} is produced by {producer}, which is registered AFTER "
                "data_qa, so at this boundary it is always the previous run's data"
            )
        return result

    def _measure_data_freshness(
        self, table_name: str, max_age_hours: int = 24
    ) -> dict[str, Any]:
        """Measure a table's age from its timestamp columns."""
        logger.info(
            "Checking data freshness", table=table_name, max_age_hours=max_age_hours
        )

        result = {
            "table": table_name,
            "is_fresh": False,
            "last_update": None,
            "age_hours": None,
            "status": "unknown",
        }

        try:
            # Try to load the table
            df = load_dataframe(table_name, layer="silver")

            if df.empty:
                result["status"] = "empty"
                return result

            # Look for timestamp columns. A table that DECLARES its write-time column is
            # judged on that column alone: a kickoff time is time-like by name and is in
            # the future for every upcoming game, so it must not stand in for a write.
            declared = self.monitored_tables.get(table_name, {}).get("freshness_column")
            if declared is not None and declared in df.columns:
                timestamp_cols = [declared]
            else:
                timestamp_cols = [
                    col
                    for col in df.columns
                    if any(
                        ts in col.lower()
                        for ts in ["timestamp", "time", "date", "created", "updated"]
                    )
                ]

            if not timestamp_cols:
                result["status"] = "no_timestamp"
                return result

            # Use the most recent timestamp
            latest_times = []
            for col in timestamp_cols:
                try:
                    latest_time = pd.to_datetime(df[col], errors="coerce").max()
                    if pd.notna(latest_time):
                        latest_times.append(latest_time)
                except (ValueError, TypeError, KeyError):
                    continue

            if not latest_times:
                result["status"] = "invalid_timestamps"
                return result

            last_update = max(latest_times)

            # Handle timezone comparison properly
            if last_update.tz is not None:
                # If last_update is timezone-aware, make now timezone-aware too
                from datetime import UTC

                now = datetime.now(UTC).astimezone(last_update.tz)
            else:
                # If last_update is timezone-naive, use naive datetime for now
                now = datetime.now()

            age_hours = (now - last_update).total_seconds() / 3600

            result.update(
                {
                    "last_update": last_update,
                    "age_hours": round(age_hours, 2),
                    "is_fresh": age_hours <= max_age_hours,
                    "status": "fresh" if age_hours <= max_age_hours else "stale",
                }
            )

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            result["status"] = f"error: {e!s}"
            logger.warning(
                "Data freshness check failed", table=table_name, error=str(e)
            )

        return result

    def check_data_completeness(
        self,
        table_name: str,
        season: int,
        week: int,
        game_ids: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Check data completeness for a specific season/week.

        Args:
            table_name: The monitored table.
            season: The season checked.
            week: The week checked.
            game_ids: The games the RUNNING cycle fetched weather for. The daily lock-time
                run passes its slate (``pipeline.daily_steps.DailySlate.game_ids``, selected
                through ``utils.game_lock``): it forecasts tomorrow's games only, so the rest
                of the week cannot have weather yet. None (the Friday step and the CLI) keeps
                the whole week, which the weekly ingest forecasts in full.
        """
        logger.info(
            "Checking data completeness", table=table_name, season=season, week=week
        )

        result = {
            "table": table_name,
            "season": season,
            "week": week,
            "expected_count": None,
            "actual_count": 0,
            "completeness_pct": 0.0,
            "missing_data": [],
            "status": "unknown",
        }

        producer = _TABLES_PRODUCED_AFTER_DATA_QA.get(table_name)
        if producer is not None:
            result["status"] = "not_applicable"
            result["not_applicable_reason"] = (
                f"{table_name} is produced by {producer}, which is registered AFTER "
                "data_qa, so its rows for this week cannot exist yet at this boundary"
            )
            return result

        try:
            df = load_dataframe(table_name, layer="silver")

            if df.empty:
                result["status"] = "empty"
                return result

            # Filter for season/week if applicable. A table keyed by game_id with NO
            # season/week (silver weather) is filtered to the week's scheduled game ids.
            # Before Plan 33-18 it fell through to the whole table, so a week the ingest
            # wrote NOTHING for was "complete" on the strength of every other week's rows.
            if "season" in df.columns and "week" in df.columns:
                filtered_df = df[(df["season"] == season) & (df["week"] == week)]
            elif "game_id" in df.columns:
                wanted = (
                    game_ids
                    if game_ids is not None
                    else self._week_game_ids(season, week)
                )
                filtered_df = df[df["game_id"].isin(wanted)]
            else:
                filtered_df = df

            result["actual_count"] = len(filtered_df)

            # Determine expected counts based on table type
            if table_name == "games":
                # Expect 16 games per week in regular season
                expected_count = 16 if week <= 18 else None
            elif table_name == "odds_snapshot":
                # Expect odds for each game (games * sportsbooks)
                games_count = self._get_games_count(season, week)
                expected_count = (
                    games_count * 3 if games_count else None
                )  # Assume 3 sportsbooks avg
            elif table_name == "weather":
                # The weekly forecast ingest covers EVERY scheduled game in the week (it
                # refuses an incomplete payload), so the week's game count is the target.
                # The daily run forecasts its slate only (daily_steps.ingest_slate_weather),
                # so its slate is.
                if game_ids is not None:
                    expected_count = len(game_ids) or None
                else:
                    games_count = self._get_games_count(season, week)
                    expected_count = games_count if games_count else None
            else:
                expected_count = None

            result["expected_count"] = expected_count

            if expected_count:
                completeness_pct = (result["actual_count"] / expected_count) * 100
                result["completeness_pct"] = round(completeness_pct, 1)

                if completeness_pct >= 90:
                    result["status"] = "complete"
                elif completeness_pct >= 70:
                    result["status"] = "mostly_complete"
                else:
                    result["status"] = "incomplete"
            else:
                result["status"] = "unknown_expected"

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            result["status"] = f"error: {e!s}"
            logger.warning(
                "Data completeness check failed", table=table_name, error=str(e)
            )

        return result

    def _week_game_ids(self, season: int, week: int) -> set[str]:
        """The game ids scheduled for ``(season, week)``, read the way the pipeline reads."""
        games_df = load_dataframe("games", layer="silver")
        week_games = games_df[
            (games_df["season"] == season) & (games_df["week"] == week)
        ]
        return set(week_games["game_id"])

    def _get_games_count(self, season: int, week: int) -> int | None:
        """Get count of games for season/week."""
        try:
            games_df = load_dataframe("games", layer="silver")
            count = len(
                games_df[(games_df["season"] == season) & (games_df["week"] == week)]
            )
            return count if count > 0 else None
        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError):
            return None

    def check_data_quality(self, table_name: str) -> dict[str, Any]:
        """Comprehensive data quality check for a table."""
        logger.info("Running data quality check", table=table_name)

        table_config = self.monitored_tables.get(table_name, {})

        result = {"table": table_name, "timestamp": datetime.now(), "checks": {}}

        try:
            df = load_dataframe(table_name, layer="silver")

            if df.empty:
                result["checks"]["empty_table"] = {
                    "status": "fail",
                    "message": "Table is empty",
                }
                return result

            # 1. Basic data quality validation
            dq_result = validate_data_quality(
                df,
                table_name,
                expected_columns=table_config.get("required_columns"),
                max_missing_pct=0.2,  # 20% max missing values
            )

            result["checks"]["basic_quality"] = {
                "status": "pass" if dq_result["is_valid"] else "fail",
                "row_count": dq_result["row_count"],
                "column_count": dq_result["column_count"],
                "errors": dq_result["errors"],
                "warnings": dq_result["warnings"],
            }

            # 2. Schema validation if validator available
            validator_func = table_config.get("validator")
            if validator_func:
                validation_errors = validator_func(df)
                result["checks"]["schema_validation"] = {
                    "status": "pass" if not validation_errors else "fail",
                    "error_count": len(validation_errors),
                    "sample_errors": validation_errors[:5],
                }

            # 3. Business rules validation
            business_rules_type = table_config.get("business_rules")
            if business_rules_type:
                business_violations = validate_nfl_business_rules(
                    df, business_rules_type
                )
                result["checks"]["business_rules"] = {
                    "status": "pass" if not business_violations else "fail",
                    "violation_count": len(business_violations),
                    "violations": business_violations[:5],
                }

            # 4. Temporal consistency (if applicable)
            if "season" in df.columns and "week" in df.columns:
                date_col = None
                for col in ["kickoff_et", "snapshot_ts", "forecast_time", "game_time"]:
                    if col in df.columns:
                        date_col = col
                        break

                if date_col:
                    temporal_violations = validate_temporal_consistency(
                        df, date_col, "season", "week"
                    )
                    result["checks"]["temporal_consistency"] = {
                        "status": "pass" if not temporal_violations else "fail",
                        "violation_count": len(temporal_violations),
                        "violations": temporal_violations[:3],
                    }

            # 5. Data distribution checks
            result["checks"]["distributions"] = self._check_data_distributions(
                df, table_name
            )

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            result["checks"]["error"] = {"status": "error", "message": str(e)}
            logger.error("Data quality check failed", table=table_name, error=str(e))

        return result

    def _check_data_distributions(
        self, df: pd.DataFrame, table_name: str
    ) -> dict[str, Any]:
        """Check data distributions for anomalies."""
        checks = {}

        try:
            # Numeric columns
            try:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    values = df[col].dropna()
                    if len(values) > 0:
                        checks[f"{col}_stats"] = {
                            "mean": round(values.mean(), 2),
                            "std": round(values.std(), 2),
                            "min": round(values.min(), 2),
                            "max": round(values.max(), 2),
                            "outliers": len(
                                values[
                                    np.abs(values - values.mean()) > 3 * values.std()
                                ]
                            ),
                        }
            except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
                checks["numeric_error"] = f"Numeric columns processing failed: {e!s}"

            # Categorical columns
            try:
                categorical_cols = df.select_dtypes(
                    include=["object", "category"]
                ).columns
                for col in categorical_cols:
                    try:
                        # Handle columns with numpy arrays by converting to string representation
                        col_values = df[col].copy()

                        # Convert numpy arrays to string representation for value_counts
                        mask = col_values.apply(lambda x: isinstance(x, np.ndarray))
                        if mask.any():
                            col_values.loc[mask] = col_values.loc[mask].apply(
                                lambda x: (
                                    str(x.tolist()) if isinstance(x, np.ndarray) else x
                                )
                            )

                        value_counts = col_values.value_counts()
                        checks[f"{col}_categories"] = {
                            "unique_count": len(value_counts),
                            "most_common": value_counts.head(3).to_dict(),
                            "null_count": df[col].isnull().sum(),
                        }
                    except (
                        ValueError,
                        KeyError,
                        TypeError,
                        FileNotFoundError,
                        OSError,
                    ) as e:
                        # If there's still an issue with this column, skip it
                        checks[f"{col}_categories"] = {
                            "error": f"Could not process categorical column: {e!s}"
                        }
            except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
                checks["categorical_error"] = (
                    f"Categorical columns processing failed: {e!s}"
                )

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            checks["error"] = str(e)

        return checks

    def check_data_consistency(self) -> dict[str, Any]:
        """Check consistency across related tables."""
        logger.info("Checking cross-table data consistency")

        result = {"timestamp": datetime.now(), "checks": {}}

        try:
            # Check game_id consistency between tables
            games_df = load_dataframe("games", layer="silver")
            game_ids = set(games_df["game_id"]) if not games_df.empty else set()

            for table in ["odds_snapshot", "weather_forecast"]:
                try:
                    table_df = load_dataframe(table, layer="silver")
                    if not table_df.empty and "game_id" in table_df.columns:
                        table_game_ids = set(table_df["game_id"])

                        # Check for orphaned records
                        orphaned = table_game_ids - game_ids
                        missing = game_ids - table_game_ids

                        result["checks"][f"{table}_game_id_consistency"] = {
                            "status": "pass" if not orphaned else "warning",
                            "orphaned_count": len(orphaned),
                            "missing_count": len(missing),
                            "sample_orphaned": list(orphaned)[:5],
                            "sample_missing": list(missing)[:5],
                        }
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    FileNotFoundError,
                    OSError,
                ) as e:
                    result["checks"][f"{table}_consistency_error"] = str(e)

            # Check team name consistency
            if not games_df.empty:
                all_teams = set(games_df["home_team"]).union(set(games_df["away_team"]))

                try:
                    venues_df = load_dataframe("venues", layer="silver")
                    if not venues_df.empty:
                        venue_teams = set()
                        for teams_list in venues_df["home_teams"]:
                            if isinstance(teams_list, (list, np.ndarray)):
                                # Convert numpy array to list if needed
                                teams = (
                                    teams_list.tolist()
                                    if isinstance(teams_list, np.ndarray)
                                    else teams_list
                                )
                                venue_teams.update(teams)
                            else:
                                # Handle single string values
                                venue_teams.add(teams_list)

                        teams_without_venues = all_teams - venue_teams

                        result["checks"]["team_venue_consistency"] = {
                            "status": "pass" if not teams_without_venues else "warning",
                            "teams_without_venues": list(teams_without_venues),
                        }
                except (
                    ValueError,
                    KeyError,
                    TypeError,
                    FileNotFoundError,
                    OSError,
                ) as e:
                    result["checks"]["team_venue_consistency_error"] = str(e)

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            result["checks"]["error"] = str(e)
            logger.error("Data consistency check failed", error=str(e))

        return result

    def check_gold_integrity(self) -> dict[str, Any]:
        """Verify Gold-layer feature-matrix integrity (AUDIT-02).

        For each of the three Gold matrices (features_wp / features_ats /
        features_ou) this checks:

        - the matrix loads and is non-empty (row count),
        - its column count matches the expected schema width declared in
          ``GOLD_FEATURE_MATRICES`` (named there rather than repeated here, so
          the two cannot drift apart across phase widenings),
        - the season span and the latest season present,
        - no feature column is entirely null (an all-null column signals a
          broken builder),
        - the trailing 2025 partial-season coverage, reported as an EXPECTED
          documented gap (D-05) rather than a hard failure.

        Returns a dict mirroring the ``check_data_consistency`` result shape:
        ``{"timestamp", "checks": {...}}`` where each per-matrix entry carries
        a ``status`` of ``pass`` / ``fail`` and the trailing-gap entry carries
        an informational ``status`` of ``expected_gap``.
        """
        logger.info("Checking Gold-layer feature-matrix integrity")

        result = {"timestamp": datetime.now(), "checks": {}}

        for table_name, expected_columns in GOLD_FEATURE_MATRICES.items():
            try:
                df = load_dataframe(table_name, layer="gold")

                if df.empty:
                    result["checks"][table_name] = {
                        "status": "fail",
                        "message": "Gold matrix is empty",
                        "row_count": 0,
                    }
                    continue

                actual_columns = len(df.columns)
                all_null_columns = [col for col in df.columns if df[col].isna().all()]

                season_min = int(df["season"].min()) if "season" in df.columns else None
                season_max = int(df["season"].max()) if "season" in df.columns else None

                schema_ok = actual_columns == expected_columns
                no_all_null = not all_null_columns

                matrix_result = {
                    "status": "pass" if (schema_ok and no_all_null) else "fail",
                    "row_count": len(df),
                    "column_count": actual_columns,
                    "expected_column_count": expected_columns,
                    "schema_width_ok": schema_ok,
                    "season_min": season_min,
                    "season_max": season_max,
                    "all_null_columns": all_null_columns,
                }

                # Trailing partial-season coverage (D-05) -- informational, not a fail.
                if "season" in df.columns and "week" in df.columns:
                    trailing = df[df["season"] > GOLD_LAST_COMPLETE_SEASON]
                    if not trailing.empty:
                        matrix_result["trailing_season_gap"] = {
                            "status": "expected_gap",
                            "season": season_max,
                            "weeks_present": sorted(
                                int(w) for w in trailing["week"].unique()
                            ),
                            "row_count": len(trailing),
                            "note": (
                                f"Season {season_max} is ingested partially and "
                                "treated as a documented expected gap (D-05); "
                                "not backfilled this phase."
                            ),
                        }

                result["checks"][table_name] = matrix_result

            except (
                ValueError,
                KeyError,
                TypeError,
                FileNotFoundError,
                OSError,
            ) as e:
                result["checks"][f"{table_name}_error"] = {
                    "status": "fail",
                    "message": str(e),
                }
                logger.warning(
                    "Gold integrity check failed", table=table_name, error=str(e)
                )

        return result

    def check_duckdb_parquet_consistency(self, loader=None) -> dict[str, Any]:
        """Verify each store's copy of a table agrees with the other (D30-18, N-01).

        THE MECHANISM THIS GUARDS. ``data.storage.load_dataframe(source="auto")``
        prefers DuckDB whenever the table exists and only falls back to parquet.
        So a DuckDB copy that has fallen behind its parquet makes every
        ``upsert_silver`` write since the divergence INVISIBLE to the whole
        pipeline -- silently, with no error raised anywhere and no warning logged.
        That is not hypothetical: silver ``games`` was measured at 6,499 parquet
        rows against 6,292 DuckDB rows at Phase-30 start, a 207-row gap, all of it
        season 2025.

        WHAT THIS CHECK DELIBERATELY DOES NOT DO. It does not touch the shared read
        path. The ``upsert_silver`` / ``load_dataframe`` write-path asymmetry that
        CAUSES the divergence is explicitly out of scope (D30-18) -- correcting the
        seam every builder, trainer and backtest reads through, inside the phase
        that is trying to measure a gold rebuild, would put an uncontrolled change
        in the same artifact as the measurement. This guard instead sits where the
        pipeline already looks and fires loudly on a re-divergence.

        MEMBERSHIP, NOT COUNTS. Equal row counts with DIFFERENT membership is the
        subtler failure and is reported as one. Pairing the integer with a named
        delta is the 29-06 lesson applied to rows: a swapped row must not be able
        to hide behind a matching total.

        THE GOLD MIRROR TOO. Gold lives in both stores and they agree today, but
        they are read by different consumers -- ``check_gold_integrity`` reads gold
        through the DuckDB-preferring ``auto`` path while ``models/train.py``,
        ``backtest/diagnose.py``, ``backtest/signal_lift.py`` and
        ``scripts/fingerprint_gold.py`` all read the parquet directly. A phase whose
        whole premise is "a DuckDB copy fell behind its parquet" must not assume the
        gold mirror is exempt, so the three widths are compared too.

        Args:
            loader: A ``load_dataframe``-shaped callable, injected by the hermetic
                positive control so the guard can be proven capable of failing
                without a live lake. Defaults to ``data.storage.load_dataframe``.
                This method opens no DuckDB connection of its own -- every read goes
                through the loader, so no handle is left for the next opener to
                collide with (T-30-53, and on Windows that collision is fatal, not
                tolerated).

        Returns:
            The house per-check contract: ``{"timestamp", "checks"}`` where each
            entry carries a ``status`` of ``pass`` / ``fail`` / ``not_applicable``.
        """
        logger.info("Checking DuckDB-vs-parquet consistency")

        read = loader if loader is not None else load_dataframe
        result = {"timestamp": datetime.now(), "checks": {}}

        for table_name, spec in _DUCKDB_PARQUET_CONSISTENCY_TABLES.items():
            layer = spec["layer"]
            key = spec["key"]
            try:
                # The accepted source literals are "auto", "db" and "parquet".
                # "duckdb" raises ValueError -- see data/storage.py load_dataframe.
                db_df = read(table_name, layer=layer, source="db")
                parquet_df = read(table_name, layer=layer, source="parquet")
                # WR-06: the key access belongs INSIDE the guarded block. Outside it, a
                # table present in BOTH stores but missing its key column raised KeyError
                # out of the whole quality report instead of being reported as one
                # table's failure.
                db_keys = set(db_df[key])
                parquet_keys = set(parquet_df[key])
            except _TABLE_ABSENT_ERRORS as e:
                result["checks"][table_name] = {
                    "status": "not_applicable",
                    "message": f"could not read both copies of {layer}.{table_name}: {e}",
                }
                continue
            except _COPY_READ_FAILURES as e:
                result["checks"][table_name] = {
                    "status": "fail",
                    "layer": layer,
                    "key_column": key,
                    "message": (
                        f"could not compare the two copies of {layer}.{table_name}: "
                        f"{type(e).__name__}: {e}. A guard that cannot run is not a "
                        "guard that passed."
                    ),
                }
                continue

            only_in_parquet = sorted(parquet_keys - db_keys)
            only_in_duckdb = sorted(db_keys - parquet_keys)
            consistent = not only_in_parquet and not only_in_duckdb

            result["checks"][table_name] = {
                "status": "pass" if consistent else "fail",
                "layer": layer,
                "key_column": key,
                "db_rows": len(db_df),
                "parquet_rows": len(parquet_df),
                "only_in_parquet_count": len(only_in_parquet),
                "only_in_duckdb_count": len(only_in_duckdb),
                "only_in_parquet_sample": only_in_parquet[:_CONSISTENCY_SAMPLE_LIMIT],
                "only_in_duckdb_sample": only_in_duckdb[:_CONSISTENCY_SAMPLE_LIMIT],
                "message": (
                    "DuckDB and parquet copies agree on row-set membership"
                    if consistent
                    else (
                        f"{len(only_in_parquet)} row(s) present only in parquet and "
                        f"{len(only_in_duckdb)} only in DuckDB. load_dataframe's 'auto' "
                        "source prefers DuckDB, so the parquet-only rows are invisible "
                        "to every consumer of this table."
                    )
                ),
            }

        for matrix in GOLD_FEATURE_MATRICES:
            try:
                db_width = len(read(matrix, layer="gold", source="db").columns)
                parquet_width = len(
                    read(matrix, layer="gold", source="parquet").columns
                )
            except _TABLE_ABSENT_ERRORS as e:
                result["checks"][f"{matrix}_width"] = {
                    "status": "not_applicable",
                    "message": f"could not read both copies of gold.{matrix}: {e}",
                }
                continue
            except _COPY_READ_FAILURES as e:
                # WR-06: same rule as the membership arm above. A locked or corrupt
                # store is the failure this tripwire exists to catch, and
                # ``generate_quality_report`` does not count not_applicable.
                result["checks"][f"{matrix}_width"] = {
                    "status": "fail",
                    "message": (
                        f"could not compare the two copies of gold.{matrix}: "
                        f"{type(e).__name__}: {e}. A guard that cannot run is not a "
                        "guard that passed."
                    ),
                }
                continue

            result["checks"][f"{matrix}_width"] = {
                "status": "pass" if db_width == parquet_width else "fail",
                "db_width": db_width,
                "parquet_width": parquet_width,
                "message": (
                    "the gold DuckDB mirror and the gold parquet agree on width"
                    if db_width == parquet_width
                    else (
                        "the width tripwire reads gold through the DuckDB-preferring "
                        "auto path while the training path reads the parquet directly, "
                        "so the two are now checking different artifacts"
                    )
                ),
            }

        return result

    def check_team_abbreviations(self) -> dict[str, Any]:
        """Verify 32-team completeness, canonical mapping, and no abbreviation
        mismatches against on-disk data (AUDIT-02).

        Uses the canonical ``utils.team_data`` source as the single source of
        truth -- NO hand-rolled team set:

        - ``get_all_teams()`` defines the canonical 32-team abbreviation set,
        - every ``home_team`` / ``away_team`` value across the Silver tables that
          carry them is asserted to be a subset of that canonical set,
        - team abbreviations encoded in the Gold matrices' ``game_id`` values are
          likewise verified canonical,
        - ``normalize_team_abbreviation`` is invoked against every on-disk
          abbreviation and must not hard-fail on any of them (the
          abbreviation-mismatch check, mirroring
          ``test_data_completeness.py:347-360``).

        Returns a dict in the ``check_data_consistency`` result shape.
        """
        logger.info("Checking 32-team completeness and canonical abbreviations")

        result = {"timestamp": datetime.now(), "checks": {}}

        canonical = set(get_all_teams())

        # 1. Canonical-set sanity: exactly 32 teams.
        result["checks"]["canonical_team_count"] = {
            "status": "pass" if len(canonical) == 32 else "fail",
            "count": len(canonical),
            "expected": 32,
        }

        # 2. Silver tables that carry home_team / away_team columns.
        observed_abbreviations: set[str] = set()
        for table in ["games", "odds_snapshot", "weather_forecast"]:
            try:
                table_df = load_dataframe(table, layer="silver")
                if table_df.empty:
                    continue

                team_columns = [
                    col
                    for col in ("home_team", "away_team", "team")
                    if col in table_df.columns
                ]
                if not team_columns:
                    continue

                table_values: set[str] = set()
                for col in team_columns:
                    table_values.update(str(v) for v in table_df[col].dropna().unique())
                observed_abbreviations.update(table_values)

                non_canonical = table_values - canonical
                result["checks"][f"{table}_canonical_teams"] = {
                    "status": "pass" if not non_canonical else "fail",
                    "team_columns": team_columns,
                    "distinct_team_count": len(table_values),
                    "non_canonical": sorted(non_canonical),
                }
            except (
                ValueError,
                KeyError,
                TypeError,
                FileNotFoundError,
                OSError,
            ) as e:
                result["checks"][f"{table}_canonical_teams_error"] = {
                    "status": "fail",
                    "message": str(e),
                }

        # 3. Gold matrices encode teams in game_id ({season}_W{week}_{away}@{home}).
        for table in GOLD_FEATURE_MATRICES:
            try:
                gold_df = load_dataframe(table, layer="gold")
                if gold_df.empty or "game_id" not in gold_df.columns:
                    continue

                gold_teams: set[str] = set()
                unparseable: list[str] = []
                for game_id in gold_df["game_id"].dropna().unique():
                    try:
                        parsed = parse_game_id(str(game_id))
                        gold_teams.add(parsed["home_team"])
                        gold_teams.add(parsed["away_team"])
                    except ValueError:
                        unparseable.append(str(game_id))

                observed_abbreviations.update(gold_teams)
                non_canonical = gold_teams - canonical
                result["checks"][f"{table}_game_id_teams"] = {
                    "status": "pass"
                    if (not non_canonical and not unparseable)
                    else "fail",
                    "distinct_team_count": len(gold_teams),
                    "non_canonical": sorted(non_canonical),
                    "unparseable_sample": unparseable[:5],
                }
            except (
                ValueError,
                KeyError,
                TypeError,
                FileNotFoundError,
                OSError,
            ) as e:
                result["checks"][f"{table}_game_id_teams_error"] = {
                    "status": "fail",
                    "message": str(e),
                }

        # 4. Abbreviation-mismatch check: normalize_team_abbreviation must not
        #    hard-fail on ANY on-disk abbreviation (it raises on unknowns).
        normalize_failures: list[str] = []
        for abbr in observed_abbreviations:
            try:
                normalize_team_abbreviation(abbr)
            except DataValidationError:
                normalize_failures.append(abbr)

        result["checks"]["abbreviation_mismatch"] = {
            "status": "pass" if not normalize_failures else "fail",
            "abbreviations_checked": len(observed_abbreviations),
            "normalize_failures": sorted(normalize_failures),
        }

        return result

    def generate_qa_report(
        self,
        season: int | None = None,
        week: int | None = None,
        weather_game_ids: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Generate comprehensive QA report.

        *weather_game_ids* is the daily run's slate: the weather completeness check then
        expects a forecast for those games, not the whole week (see
        :meth:`check_data_completeness`). None keeps the whole week.
        """
        if season is None or week is None:
            current_season, current_week = get_current_nfl_week()
            season = season or current_season
            week = week or current_week

        logger.info("Generating QA report", season=season, week=week)

        report = {
            "timestamp": datetime.now(),
            "season": season,
            "week": week,
            "summary": {},
            "table_reports": {},
            "consistency_check": {},
            "gold_integrity": {},
            "team_abbreviations": {},
            "duckdb_parquet_consistency": {},
            "database_stats": {},
            "recommendations": [],
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
                    "freshness": self.check_data_freshness(table_name),
                    "completeness": self.check_data_completeness(
                        table_name,
                        season,
                        week,
                        weather_game_ids if table_name == "weather" else None,
                    ),
                    "quality": self.check_data_quality(table_name),
                }

                report["table_reports"][table_name] = table_report

                # Update summary counts
                for check_type, check_result in table_report.items():
                    if isinstance(check_result, dict):
                        status = check_result.get("status", "unknown")

                        # Skip checks that are not applicable (e.g., no timestamps, unknown expected counts)
                        # "not_applicable" joins the list for tables produced by a later
                        # registered step (_TABLES_PRODUCED_AFTER_DATA_QA); the section
                        # checks below already skip it.
                        not_applicable_statuses = [
                            "not_applicable",
                            "no_timestamp",
                            "invalid_timestamps",
                            "unknown_expected",
                            "empty",
                            "error",
                            "unknown",
                        ]

                        if any(
                            na_status in status for na_status in not_applicable_statuses
                        ):
                            continue  # Don't count these in total checks

                        # For quality checks, count individual sub-checks
                        if check_type == "quality" and "checks" in check_result:
                            for _sub_check_name, sub_check_result in check_result[
                                "checks"
                            ].items():
                                if (
                                    isinstance(sub_check_result, dict)
                                    and "status" in sub_check_result
                                ):
                                    sub_status = sub_check_result["status"]
                                    if sub_status not in not_applicable_statuses:
                                        total_checks += 1
                                        if sub_status == "pass":
                                            passed_checks += 1
                                        elif sub_status == "fail":
                                            failed_checks += 1
                                        elif sub_status == "warning":
                                            warnings += 1
                        else:
                            total_checks += 1
                            if status == "pass" or status in ["fresh", "complete"]:
                                passed_checks += 1
                            elif status == "fail" or status in ["stale", "incomplete"]:
                                failed_checks += 1
                            elif status in ["warning", "mostly_complete"]:
                                warnings += 1

            # Cross-table consistency
            report["consistency_check"] = self.check_data_consistency()

            # Gold-layer integrity + 32-team / canonical-abbreviation checks (AUDIT-02)
            report["gold_integrity"] = self.check_gold_integrity()
            report["team_abbreviations"] = self.check_team_abbreviations()

            # DuckDB-vs-parquet consistency (D30-18). Wired in beside gold_integrity
            # so the guard runs wherever the pipeline already runs data QA -- a check
            # that has to be remembered is a check that will not be run.
            report["duckdb_parquet_consistency"] = (
                self.check_duckdb_parquet_consistency()
            )

            # Count check results from the flat-checks sections. The trailing
            # "expected_gap" status (D-05) is intentionally NOT counted as a
            # pass/fail/warning -- it is an informational documented gap. Neither is
            # "not_applicable", which means a copy could not be read at all.
            for section in (
                "consistency_check",
                "gold_integrity",
                "team_abbreviations",
                "duckdb_parquet_consistency",
            ):
                section_checks = report[section].get("checks", {})
                for _check_name, check_result in section_checks.items():
                    if isinstance(check_result, dict) and "status" in check_result:
                        status = check_result["status"]
                        if status in ("expected_gap", "not_applicable"):
                            continue
                        total_checks += 1
                        if status == "pass":
                            passed_checks += 1
                        elif status == "fail":
                            failed_checks += 1
                        elif status == "warning":
                            warnings += 1

            # Database statistics
            try:
                report["database_stats"] = get_database_stats()
            except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
                report["database_stats"] = {"error": str(e)}

            # Summary
            report["summary"] = {
                "total_checks": total_checks,
                "passed": passed_checks,
                "failed": failed_checks,
                "warnings": warnings,
                "pass_rate": round((passed_checks / total_checks) * 100, 1)
                if total_checks > 0
                else 0,
                "overall_status": "healthy"
                if failed_checks == 0
                else "issues_detected",
            }

            # Generate recommendations
            report["recommendations"] = self._generate_recommendations(report)

            # Log summary
            logger.info(
                "QA report generated",
                total_checks=total_checks,
                passed=passed_checks,
                failed=failed_checks,
                warnings=warnings,
                overall_status=report["summary"]["overall_status"],
            )

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            report["error"] = str(e)
            logger.error("QA report generation failed", error=str(e))

        return report

    def _generate_recommendations(self, report: dict[str, Any]) -> list[str]:
        """Generate recommendations based on QA results."""
        recommendations = []

        try:
            # Check for stale data
            for table, table_report in report.get("table_reports", {}).items():
                freshness = table_report.get("freshness", {})
                if freshness.get("status") == "stale":
                    age_hours = freshness.get("age_hours", 0)
                    recommendations.append(
                        f"Update {table} data - last updated {age_hours:.1f} hours ago"
                    )

            # Check for incomplete data
            for table, table_report in report.get("table_reports", {}).items():
                completeness = table_report.get("completeness", {})
                if completeness.get("status") in ["incomplete", "mostly_complete"]:
                    pct = completeness.get("completeness_pct", 0)
                    recommendations.append(
                        f"Investigate {table} completeness - only {pct}% complete"
                    )

            # Check for quality issues
            for table, table_report in report.get("table_reports", {}).items():
                quality = table_report.get("quality", {})
                checks = quality.get("checks", {})

                for check_name, check_result in checks.items():
                    if check_result.get("status") == "fail":
                        recommendations.append(
                            f"Fix {check_name} issues in {table} table"
                        )

            # Check consistency issues
            consistency = report.get("consistency_check", {}).get("checks", {})
            for check_name, check_result in consistency.items():
                if (
                    isinstance(check_result, dict)
                    and check_result.get("status") == "warning"
                ):
                    recommendations.append(f"Review {check_name} data consistency")

            # Check Gold-integrity + abbreviation failures (AUDIT-02)
            for section in ("gold_integrity", "team_abbreviations"):
                section_checks = report.get(section, {}).get("checks", {})
                for check_name, check_result in section_checks.items():
                    if (
                        isinstance(check_result, dict)
                        and check_result.get("status") == "fail"
                    ):
                        recommendations.append(
                            f"Investigate {section} check '{check_name}'"
                        )

        except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
            recommendations.append(f"Error generating recommendations: {e}")

        return recommendations

    def save_qa_report(
        self, report: dict[str, Any], output_dir: str | None = None
    ) -> str:
        """Save QA report to file."""
        if output_dir is None:
            output_dir = self.settings.get_data_path("outputs")
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"qa_report_{timestamp}.json"
        filepath = output_dir / filename

        # Convert datetime objects and numpy types to JSON serializable formats
        def serialize_datetime(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=serialize_datetime)

        logger.info("QA report saved", filepath=str(filepath))
        return str(filepath)


def main():
    """CLI entry point for data QA monitoring."""
    parser = argparse.ArgumentParser(description="NFL data quality assurance")
    parser.add_argument("--season", type=int, help="Season to check (default: current)")
    parser.add_argument("--week", type=int, help="Week to check (default: current)")
    parser.add_argument("--current", action="store_true", help="Check current week")
    parser.add_argument(
        "--table",
        type=str,
        choices=["games", "odds_snapshot", "weather", "venues"],
        help="Check specific table only",
    )
    parser.add_argument(
        "--check",
        type=str,
        choices=[
            "freshness",
            "completeness",
            "quality",
            "consistency",
            "gold_integrity",
            "team_abbreviations",
            "duckdb_parquet_consistency",
        ],
        help="Run specific check only",
    )
    parser.add_argument("--output", type=str, help="Output directory for report")
    parser.add_argument("--save", action="store_true", help="Save report to file")

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

        table_independent = (
            "gold_integrity",
            "team_abbreviations",
            "duckdb_parquet_consistency",
        )
        if args.check in table_independent and not args.table:
            # Table-independent checks: AUDIT-02 gold matrices / canonical teams, and
            # the D30-18 DuckDB-vs-parquet membership guard.
            if args.check == "gold_integrity":
                result = monitor.check_gold_integrity()
            elif args.check == "duckdb_parquet_consistency":
                result = monitor.check_duckdb_parquet_consistency()
            else:
                result = monitor.check_team_abbreviations()

            print(f"Check: {args.check}")
            checks = result.get("checks", {})
            failed = [
                name
                for name, sub in checks.items()
                if isinstance(sub, dict) and sub.get("status") == "fail"
            ]
            print(f"Sub-checks: {len(checks)}  Failed: {len(failed)}")
            for name, sub in checks.items():
                if isinstance(sub, dict) and "status" in sub:
                    print(f"  [{sub['status']}] {name}")

        elif args.table and args.check:
            # Run specific check on specific table
            if args.check == "freshness":
                result = monitor.check_data_freshness(args.table)
            elif args.check == "completeness":
                result = monitor.check_data_completeness(args.table, season, week)
            elif args.check == "quality":
                result = monitor.check_data_quality(args.table)
            elif args.check == "consistency":
                result = monitor.check_data_consistency()

            print(f"Check: {args.check} on table: {args.table}")
            print(f"Status: {result.get('status', 'unknown')}")
            if "message" in result:
                print(f"Message: {result['message']}")

        elif args.table:
            # Check specific table
            print(f"Checking table: {args.table}")

            freshness = monitor.check_data_freshness(args.table)
            print(f"  Freshness: {freshness['status']}")

            completeness = monitor.check_data_completeness(args.table, season, week)
            print(
                f"  Completeness: {completeness['status']} ({completeness.get('completeness_pct', 0)}%)"
            )

            quality = monitor.check_data_quality(args.table)
            quality_status = (
                "pass"
                if all(
                    check.get("status") == "pass"
                    for check in quality.get("checks", {}).values()
                    if isinstance(check, dict)
                )
                else "issues"
            )
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

            if report["recommendations"]:
                print("\nRecommendations:")
                for rec in report["recommendations"]:
                    print(f"  • {rec}")

            # Save report if requested
            if args.save:
                filepath = monitor.save_qa_report(report, args.output)
                print(f"\nReport saved to: {filepath}")

    except (ValueError, KeyError, TypeError, FileNotFoundError, OSError) as e:
        logger.error("Data QA CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
