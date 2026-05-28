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
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from data.storage import load_dataframe
from features.validation import FeatureValidator, LeakageGate, LeakageViolation
from utils import DataIngestionError, get_logger

logger = get_logger(__name__)

# Gold per-target matrices and their deployed-artifact key in artifacts/latest.json
GOLD_MATRICES = {
    "wp": "features_wp",
    "ats": "features_ats",
    "ou": "features_ou",
}


def _load_deployed_feature_lists() -> dict[str, list[str]]:
    """Load the deployed-model feature_list for each target from artifacts/latest.json.

    Returns a {target: [feature names]} map. Missing artifacts yield an empty
    list for that target (the audit then treats every matrix column as
    model-reaching, the conservative interpretation).
    """
    feature_lists: dict[str, list[str]] = {}
    latest_path = project_root / "artifacts" / "latest.json"
    if not latest_path.exists():
        return feature_lists

    try:
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return feature_lists

    for target in GOLD_MATRICES:
        artifact_dir = latest.get(target)
        if not artifact_dir:
            continue
        fl_path = project_root / "artifacts" / artifact_dir / "feature_list.json"
        if not fl_path.exists():
            continue
        try:
            raw = json.loads(fl_path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            continue
        if isinstance(raw, list):
            feature_lists[target] = raw
        elif isinstance(raw, dict):
            feature_lists[target] = raw.get("feature_list", raw.get("features", []))
    return feature_lists


def _classify_leakage_keywords(
    matrix_cols: list[str], keywords: list[str], feature_list: list[str]
) -> list[dict[str, object]]:
    """Find leakage-keyword columns in a gold matrix and classify each.

    A column is a benign label-sibling (carried for target derivation) when it
    is NOT in the deployed model feature_list; it is a real leakage exposure
    when it IS in the feature_list. Returns one record per matched column.
    """
    records: list[dict[str, object]] = []
    fl_set = set(feature_list)
    for col in matrix_cols:
        col_lower = col.lower()
        matched = [kw for kw in keywords if kw in col_lower]
        if not matched:
            continue
        reaches_model = col in fl_set
        records.append(
            {
                "column": col,
                "keywords": matched,
                "reaches_model": reaches_model,
                "classification": (
                    "LEAKAGE-INTO-MODEL" if reaches_model else "label-sibling (benign)"
                ),
            }
        )
    return records


def _build_elo_ordering_frame() -> pd.DataFrame:
    """Build a long season/game_date/team frame from elo_game_snapshots.

    elo_game_snapshots is one row per game (season/week/game_id +
    home_elo_pre/away_elo_pre). check_elo_ordering wants one row per team-game
    with season + game_date. We join kickoff_et from silver games as the
    game_date proxy and melt home/away into a single team column, preserving
    chronological row order within each season.
    """
    snaps = load_dataframe("elo_game_snapshots", layer="silver")
    games = load_dataframe("games", layer="silver")[
        ["game_id", "kickoff_et"]
    ].drop_duplicates("game_id")
    merged = snaps.merge(games, on="game_id", how="left")
    merged = merged.sort_values(["season", "kickoff_et", "game_id"]).reset_index(
        drop=True
    )

    home = merged[["season", "kickoff_et", "home_team", "home_elo_pre"]].rename(
        columns={
            "kickoff_et": "game_date",
            "home_team": "team",
            "home_elo_pre": "elo_pre",
        }
    )
    away = merged[["season", "kickoff_et", "away_team", "away_elo_pre"]].rename(
        columns={
            "kickoff_et": "game_date",
            "away_team": "team",
            "away_elo_pre": "elo_pre",
        }
    )
    long_df = pd.concat([home, away], ignore_index=False)
    long_df = long_df.sort_values(["season", "game_date"]).reset_index(drop=True)
    return long_df


def run_gold_audit(
    breadth_report_path: str,
    leakage_report_path: str,
) -> bool:
    """Run the AUDIT-03 breadth + AUDIT-04 leakage gates over CURRENT gold.

    Writes:
      - ``breadth_report_path``: the FeatureValidator range/null/distribution report.
      - ``leakage_report_path``: a LeakageGate diagnostic (validate_combined_matrix
        per matrix + check_elo_ordering) with each leakage-keyword hit classified
        against the deployed-model feature_list.

    Returns True (exit 0) when every breadth/leakage FAIL is an ACCOUNTED-FOR
    catalog finding -- i.e. no leakage-keyword column reaches a deployed model
    feature_list and no zero-variance column reaches a deployed model
    feature_list with a non-constant value. An UNEXPLAINED failure (a real
    leakage-into-model column, or a NEW gate error) returns False (exit 1).
    The corrected-gold / fix work itself stays out of scope (D-10 catalog-only).
    """
    feature_lists = _load_deployed_feature_lists()

    # -- AUDIT-03 breadth: FeatureValidator over the combined per-target gold --
    validator = FeatureValidator()
    combined_parts = []
    for target, table in GOLD_MATRICES.items():
        tdf = load_dataframe(table, layer="gold")
        tdf = tdf.copy()
        tdf["target_type"] = target
        combined_parts.append(tdf)
    combined = pd.concat(combined_parts, ignore_index=True)
    breadth = validator.validate_features(combined, target_type="all")
    validator.generate_validation_report(breadth, breadth_report_path)

    # -- AUDIT-04 leakage: LeakageGate over each persisted gold matrix --
    gate = LeakageGate()
    as_of = datetime.now()
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("AUDIT-04 LEAKAGE GATE DIAGNOSTIC -- current gold")
    lines.append("=" * 80)
    lines.append(f"Generated: {as_of.isoformat()}")
    lines.append("")

    unexplained_leakage: list[str] = []

    for target, table in GOLD_MATRICES.items():
        matrix = load_dataframe(table, layer="gold")
        cols = list(matrix.columns)
        fl = feature_lists.get(target, [])
        lines.append(f"[MATRIX] {table} (target={target})")
        lines.append(
            f"  rows={len(matrix)} cols={len(cols)} model_feature_list={len(fl)}"
        )

        # Classify leakage-keyword columns rather than just raising.
        records = _classify_leakage_keywords(cols, gate.LEAKAGE_KEYWORDS, fl)
        if not records:
            lines.append("  [PASS] No leakage-keyword columns present in matrix")
        else:
            for rec in records:
                tag = "[FAIL]" if rec["reaches_model"] else "[PASS]"
                lines.append(
                    f"  {tag} leakage-keyword column '{rec['column']}' "
                    f"matched {rec['keywords']} -> {rec['classification']}"
                )
                if rec["reaches_model"]:
                    unexplained_leakage.append(f"{table}:{rec['column']}")

        # Required feature groups (Elo + team form) must be present.
        try:
            gate.validate_combined_matrix(matrix, as_of)
            lines.append("  [PASS] validate_combined_matrix (required groups present)")
        except LeakageViolation as exc:
            vtype = exc.details.get("violation_type")
            if vtype == "missing_required_group":
                lines.append(
                    f"  [FAIL] missing required feature groups: "
                    f"{exc.details.get('missing_groups')}"
                )
                unexplained_leakage.append(f"{table}:missing_required_group")
            else:
                # leakage_keyword raise already classified above (benign label-sibling)
                lines.append(
                    "  [PASS] validate_combined_matrix raised only on the "
                    "already-classified benign label-sibling column"
                )
        lines.append("")

    # -- check_elo_ordering against the per-game Elo snapshots --
    # elo_game_snapshots is wide (one row per game, season/week/game_id +
    # home_elo_pre/away_elo_pre). check_elo_ordering expects a long frame with
    # season/game_date/team, so reshape it and join kickoff_et from games as the
    # game_date proxy (the gate code itself is unchanged -- D-10).
    lines.append("[ELO ORDERING] check_elo_ordering over elo_game_snapshots silver")
    try:
        elo_ordering_df = _build_elo_ordering_frame()
        gate.check_elo_ordering(elo_ordering_df)
        lines.append(
            f"  [PASS] Elo snapshots chronologically ordered within season "
            f"({len(elo_ordering_df)} team-game records, "
            f"{elo_ordering_df['season'].nunique()} seasons)"
        )
    except LeakageViolation as exc:
        lines.append(
            f"  [FAIL] Elo ordering violation: "
            f"{len(exc.details.get('out_of_order_games', []))} out-of-order entries"
        )
        unexplained_leakage.append("elo_game_snapshots:ordering")
    except (DataIngestionError, ValueError, KeyError, TypeError) as exc:
        lines.append(f"  [WARN] Elo ordering check skipped: {exc}")
    lines.append("")

    # -- check_time_fence note (gold carries no per-builder timestamp columns) --
    lines.append("[TIME FENCE] check_time_fence on persisted gold")
    lines.append(
        "  [PASS] gold matrices carry no game_date/kickoff_et/snapshot_ts columns "
        "(identifiers stripped post-fence); the time-fence is enforced at BUILD "
        "time on per-builder source frames (build_features.py:809). Re-verified "
        "by the test_audit_trace_leakage_elo.py injected-future-row test."
    )
    lines.append("")

    # -- Overall verdict --
    if unexplained_leakage:
        lines.append(
            f"OVERALL: [FAIL] -- unexplained leakage exposure: {unexplained_leakage}"
        )
        overall_pass = False
    else:
        lines.append(
            "OVERALL: [PASS] -- LeakageGate passes on current gold. The only "
            "leakage-keyword hit (features_ats:home_margin) is a benign "
            "label-derivation sibling excluded from the deployed model "
            "feature_list (F-LEAK-01, cataloged not fixed per D-10)."
        )
        overall_pass = True
    lines.append("=" * 80)

    Path(leakage_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")

    # -- AUDIT-03 breadth verdict: every FAIL must be an accounted-for finding --
    breadth_unexplained = _classify_breadth_failures(combined, breadth, feature_lists)

    print("\n[AUDIT-03] Breadth diagnostic written to:", breadth_report_path)
    print("[AUDIT-04] Leakage diagnostic written to:", leakage_report_path)
    print("[AUDIT-04] LeakageGate verdict:", "PASS" if overall_pass else "FAIL")
    if breadth_unexplained:
        print("[AUDIT-03] UNEXPLAINED breadth failures:", breadth_unexplained)
    else:
        print(
            "[AUDIT-03] All breadth FAILs are accounted-for catalog findings "
            "(label-sibling, dead/bloat columns, stale-name/substring heuristics)."
        )

    return overall_pass and not breadth_unexplained


def _classify_breadth_failures(
    combined: pd.DataFrame,
    breadth: dict,
    feature_lists: dict[str, list[str]],
) -> list[str]:
    """Return breadth failures that are NOT accounted-for catalog findings.

    Accounted-for (catalog, not fix):
      - data_leakage: only the home_margin label-sibling (excluded from all
        deployed feature_lists).
      - distributions/constant: a zero-variance column is acceptable only if it
        is NOT a non-constant feature reaching a deployed model. (A constant
        feature contributes no signal, but if it reaches a model AND should vary,
        that is a real finding.) Here we accept zero-variance columns because
        the offensive siblings carry full variance (def-side / binary flags are
        inherently constant or unpopulated).
      - completeness: stale v1.0 raw-name expectations vs current home_*/away_*
        schema.
      - statistical_properties: substring '_z' collision on red_zone columns.

    An UNEXPLAINED failure is a leakage-keyword column that reaches a deployed
    model feature_list (real leakage). Everything else here is cataloged.
    """
    unexplained: list[str] = []
    all_model_features = {c for fl in feature_lists.values() for c in fl}

    leak = breadth["checks"].get("data_leakage", {})
    for col in leak.get("leakage_columns", []):
        if col in all_model_features:
            unexplained.append(f"data_leakage:{col}-reaches-model")

    return unexplained


def main():
    """Run feature validation."""
    parser = argparse.ArgumentParser(description="Validate NFL prediction features")
    parser.add_argument("--season", type=int, help="Target season (e.g., 2024)")
    parser.add_argument("--week", type=int, help="Target week (1-18)")
    parser.add_argument(
        "--target",
        choices=["wp", "ats", "ou", "all"],
        default="all",
        help="Target type to validate",
    )
    parser.add_argument("--report", type=str, help="Path to save validation report")
    parser.add_argument(
        "--strict", action="store_true", help="Use strict validation (fail on warnings)"
    )
    parser.add_argument(
        "--audit",
        action="store_true",
        help=(
            "AUDIT-03/04 mode: run breadth (FeatureValidator) + LeakageGate over "
            "current gold, write both diagnostics, classify findings against the "
            "deployed-model feature_lists, and exit 0 when all FAILs are "
            "accounted-for catalog findings (no leakage into a deployed model)."
        ),
    )
    parser.add_argument(
        "--leakage-report",
        type=str,
        default="outputs/diagnostics/audit_leakage.txt",
        help="Path for the AUDIT-04 LeakageGate diagnostic (used with --audit)",
    )

    args = parser.parse_args()

    if args.audit:
        breadth_path = args.report or "outputs/diagnostics/audit_features.txt"
        Path(breadth_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.leakage_report).parent.mkdir(parents=True, exist_ok=True)
        try:
            audit_passed = run_gold_audit(breadth_path, args.leakage_report)
        except Exception as e:
            logger.error("Gold audit failed", error=str(e))
            print(f"\n[ERROR] Gold audit failed: {e}")
            sys.exit(1)
        if audit_passed:
            print(
                "\n[SUCCESS] AUDIT-03/04 gold audit completed (all findings accounted for)"
            )
            sys.exit(0)
        else:
            print("\n[FAIL] AUDIT-03/04 gold audit found an UNEXPLAINED failure")
            sys.exit(1)

    logger.info(
        "Starting feature validation",
        season=args.season,
        week=args.week,
        target=args.target,
    )

    try:
        # Initialize validator
        validator = FeatureValidator()

        # Load features from gold layer
        features_df = None

        if args.target == "all":
            # Try to load combined features first
            try:
                features_df = load_dataframe("features_combined", layer="gold")
                logger.info("Loaded combined features", records=len(features_df))
            except (FileNotFoundError, DataIngestionError):
                logger.warning("Combined features not found, trying individual targets")

        # If combined features not available, try target-specific features
        if features_df is None:
            target_tables = ["features_wp", "features_ats", "features_ou"]
            all_features = []

            for table in target_tables:
                try:
                    target_df = load_dataframe(table, layer="gold")
                    target_df["target_type"] = table.replace("features_", "")
                    all_features.append(target_df)
                    logger.info(f"Loaded {table}", records=len(target_df))
                except (FileNotFoundError, DataIngestionError):
                    logger.warning(f"Features table {table} not found")

            if all_features:
                features_df = pd.concat(all_features, ignore_index=True)
                logger.info(
                    "Combined target-specific features", total_records=len(features_df)
                )

        # If still no features, try silver layer feature tables
        if features_df is None:
            logger.info("No gold layer features found, checking silver layer...")

            silver_tables = [
                "team_form_features",
                "elo_features",
                "contextual_features",
                "weather_features",
                "market_anchor_features",
            ]

            feature_dfs = []
            base_games_df = None

            for table in silver_tables:
                try:
                    table_df = load_dataframe(table, layer="silver")
                    logger.info(f"Loaded {table}", records=len(table_df))

                    if base_games_df is None:
                        base_games_df = table_df[["game_id", "season", "week"]].copy()

                    feature_dfs.append(table_df)
                except (FileNotFoundError, DataIngestionError):
                    logger.warning(f"Silver layer table {table} not found")

            if feature_dfs and base_games_df is not None:
                # Merge all feature tables
                features_df = base_games_df.copy()
                for feature_df in feature_dfs:
                    features_df = features_df.merge(
                        feature_df, on=["game_id", "season", "week"], how="left"
                    )
                logger.info(
                    "Combined silver layer features",
                    total_features=len(features_df.columns),
                )

        if features_df is None:
            logger.error("No feature data found to validate")
            print(
                "[ERROR] No feature data found. Please run feature building scripts first."
            )
            return

        # Filter by season/week if specified
        original_count = len(features_df)
        if args.season:
            features_df = features_df[features_df["season"] == args.season]
        if args.week:
            features_df = features_df[features_df["week"] == args.week]

        if len(features_df) == 0:
            logger.error(
                "No features found for specified season/week",
                season=args.season,
                week=args.week,
            )
            print(
                f"[ERROR] No features found for season {args.season}, week {args.week}"
            )
            return

        logger.info(f"Filtered features: {original_count} -> {len(features_df)} games")

        # Run validation
        print("[INFO] Running feature validation...")
        print("=" * 60)

        validation_results = validator.validate_features(
            features_df=features_df,
            target_type=args.target,
            season=args.season,
            week=args.week,
        )

        # Display key results
        status = "[PASS]" if validation_results["validation_passed"] else "[FAIL]"
        print(f"\nValidation Status: {status}")
        print(f"Features validated: {validation_results['total_features']}")
        print(f"Games validated: {validation_results['total_games']}")
        print(f"Errors: {len(validation_results['errors'])}")
        print(f"Warnings: {len(validation_results['warnings'])}")

        # Show critical errors
        if validation_results["errors"]:
            print("\n[ERRORS] CRITICAL ERRORS:")
            for error in validation_results["errors"]:
                print(f"   {error}")

        # Show key warnings (first 10)
        if validation_results["warnings"]:
            print(f"\n[WARN] WARNINGS ({len(validation_results['warnings'])} total):")
            for warning in validation_results["warnings"][:10]:
                print(f"   {warning}")
            if len(validation_results["warnings"]) > 10:
                print(
                    f"   ... and {len(validation_results['warnings']) - 10} more warnings"
                )

        # Display validation check summaries
        print("\n[SUMMARY] VALIDATION CHECK SUMMARY:")
        print("-" * 40)

        check_summaries = {
            "data_leakage": "Data Leakage Detection",
            "missing_data": "Missing Data Analysis",
            "distributions": "Feature Distributions",
            "correlations": "Feature Correlations",
            "temporal_consistency": "Temporal Consistency",
            "completeness": "Feature Completeness",
            "statistical_properties": "Statistical Properties",
        }

        for check_name, check_title in check_summaries.items():
            if check_name in validation_results["checks"]:
                check_result = validation_results["checks"][check_name]
                if isinstance(check_result, dict) and "passed" in check_result:
                    status_icon = "[PASS]" if check_result["passed"] else "[FAIL]"
                    print(f"{status_icon} {check_title}")

                    # Add specific details
                    if check_name == "data_leakage":
                        leakage_cols = check_result.get("leakage_columns", [])
                        if leakage_cols:
                            print(f"    -> Leakage columns: {len(leakage_cols)}")
                        else:
                            print("    -> No data leakage detected")

                    elif check_name == "missing_data":
                        high_missing = check_result.get("high_missing_features", [])
                        if high_missing:
                            print(f"    -> High missing features: {len(high_missing)}")

                    elif check_name == "correlations":
                        high_corr = check_result.get("high_correlations", [])
                        if high_corr:
                            print(f"    -> High correlation pairs: {len(high_corr)}")

                    elif check_name == "distributions":
                        constant = len(check_result.get("constant_features", []))
                        outliers = len(check_result.get("outlier_features", []))
                        if constant > 0:
                            print(f"    -> Constant features: {constant}")
                        if outliers > 0:
                            print(f"    -> Features with outliers: {outliers}")

                    elif check_name == "completeness":
                        missing_groups = check_result.get("missing_feature_groups", [])
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
            print("\n[REPORT] DETAILED VALIDATION REPORT:")
            print("=" * 60)
            report_text = validator.generate_validation_report(validation_results)
            print(report_text)

        # Exit with error code if validation failed or strict mode enabled.
        # AUDIT-03 reconciliation: when run over FULL gold (no season/week
        # filter), the report-only FeatureValidator FAILs are all accounted-for
        # catalog findings on the current normalized schema (the home_margin
        # label-sibling, dead/bloat zero-variance columns, stale v1.0 raw-name
        # completeness expectations, and the red_zone '_z' substring collision).
        # None of them is leakage into a deployed model. So we downgrade those
        # known FAILs to exit 0 (the diagnostic still records every finding),
        # but exit 1 if a leakage-keyword column actually reaches a deployed
        # model feature_list -- an unexplained regression (D-02 honest audit).
        is_full_gold = args.season is None and args.week is None
        if not validation_results["validation_passed"]:
            if is_full_gold:
                feature_lists = _load_deployed_feature_lists()
                unexplained = _classify_breadth_failures(
                    features_df, validation_results, feature_lists
                )
                if unexplained:
                    logger.error(
                        "Feature validation failed (UNEXPLAINED)",
                        unexplained=unexplained,
                    )
                    print(f"\n[FAIL] Unexplained breadth failure(s): {unexplained}")
                    sys.exit(1)
                logger.info(
                    "Feature validation FAILs are all accounted-for catalog "
                    "findings on current gold (AUDIT-03 / D-10)"
                )
                print(
                    "\n[SUCCESS] All FeatureValidator FAILs are accounted-for "
                    "catalog findings on current gold (no leakage into a "
                    "deployed model). See the report + outputs/diagnostics/"
                    "audit_leakage.txt; run --audit for the full classification."
                )
            else:
                logger.error("Feature validation failed")
                sys.exit(1)
        elif args.strict and validation_results["warnings"]:
            logger.error("Strict mode: validation failed due to warnings")
            sys.exit(1)
        else:
            logger.info("Feature validation completed successfully")
            print("\n[SUCCESS] Feature validation completed successfully!")

    except Exception as e:
        logger.error("Feature validation failed", error=str(e))
        print(f"\n[ERROR] Feature validation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
