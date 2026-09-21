"""DuckDB cache schema and population logic for the web API.

Defines the CACHE_SCHEMA for the web_cache.duckdb database and provides
the populate_cache() function that builds the cache from model artifacts,
backtest outputs, and precomputed chart HTML.

The cache is the sole data source for the API (UIAP-01). No model classes
are imported here -- only artifacts and outputs are read.
"""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd
from scipy.special import expit, logit

from utils import get_logger
from utils.edge_tier import edge_tier_series
from utils.probability_utils import moneyline_to_probability
from utils.team_data import get_team_conference, get_team_division

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    game_id VARCHAR PRIMARY KEY,
    season INTEGER,
    week INTEGER,
    game_date TIMESTAMP,
    home_team VARCHAR,
    away_team VARCHAR,
    status VARCHAR,
    home_score INTEGER,
    away_score INTEGER,
    wp_prob DOUBLE,
    wp_confidence VARCHAR,
    ats_prediction DOUBLE,
    ats_confidence VARCHAR,
    ou_prediction DOUBLE,
    ou_confidence VARCHAR,
    market_spread DOUBLE,
    market_total DOUBLE,
    market_ml_home INTEGER,
    market_ml_away INTEGER,
    wp_edge DOUBLE,
    ats_edge DOUBLE,
    ou_edge DOUBLE,
    blended_wp DOUBLE,
    blended_ats DOUBLE,
    blended_ou DOUBLE
);

CREATE TABLE IF NOT EXISTS feature_importances (
    game_id VARCHAR,
    target VARCHAR,
    feature_name VARCHAR,
    importance DOUBLE,
    PRIMARY KEY (game_id, target, feature_name)
);

CREATE TABLE IF NOT EXISTS backtest_metrics (
    season INTEGER,
    target VARCHAR,
    metric_name VARCHAR,
    metric_value DOUBLE,
    PRIMARY KEY (season, target, metric_name)
);

CREATE TABLE IF NOT EXISTS backtest_predictions (
    game_id VARCHAR,
    season INTEGER,
    week INTEGER,
    target VARCHAR,
    model_prob DOUBLE,
    actual DOUBLE,
    probability_clv DOUBLE,
    has_closing_odds BOOLEAN,
    PRIMARY KEY (game_id, target)
);

CREATE TABLE IF NOT EXISTS simulation_results (
    strategy VARCHAR,
    metric_name VARCHAR,
    metric_value DOUBLE,
    PRIMARY KEY (strategy, metric_name)
);

CREATE TABLE IF NOT EXISTS equity_curve (
    strategy VARCHAR,
    bet_index INTEGER,
    bankroll DOUBLE,
    PRIMARY KEY (strategy, bet_index)
);

CREATE TABLE IF NOT EXISTS betting_bets (
    game_id VARCHAR,
    season INTEGER,
    week INTEGER,
    target VARCHAR,
    bet_side VARCHAR,
    model_value DOUBLE,
    market_value DOUBLE,
    edge DOUBLE,
    slipped_line DOUBLE,
    odds DOUBLE,
    flat_stake DOUBLE,
    kelly_stake DOUBLE,
    outcome BOOLEAN,
    payout_flat DOUBLE,
    payout_kelly DOUBLE
    -- No PRIMARY KEY: betting_bets is a flat per-bet ledger (read only with
    -- SELECT *, never joined on a key). A (game_id, target) PK would let
    -- INSERT OR REPLACE silently drop same-target re-bets on one game (WR-02).
);

CREATE TABLE IF NOT EXISTS chart_cache (
    chart_id VARCHAR PRIMARY KEY,
    html_div TEXT,
    generated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS game_context (
    game_id VARCHAR PRIMARY KEY,
    home_elo DOUBLE,
    away_elo DOUBLE,
    home_last5 VARCHAR,
    away_last5 VARCHAR,
    h2h_record VARCHAR,
    venue_name VARCHAR,
    surface VARCHAR,
    roof_type VARCHAR,
    weather_severity DOUBLE,
    weather_severity_band VARCHAR,
    wind_mph DOUBLE,
    is_outdoor BOOLEAN,
    is_divisional BOOLEAN,
    is_primetime BOOLEAN
);

CREATE TABLE IF NOT EXISTS cache_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR,
    updated_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bet_list (
    -- IMMUTABLE recommendation facts (BET_LIST_IMMUTABLE_COLUMNS, 23) --------------
    game_id VARCHAR,
    season INTEGER,
    week INTEGER,
    target VARCHAR,
    bet_side VARCHAR,
    model_value DOUBLE,
    market_value DOUBLE,
    line DOUBLE,
    slipped_line DOUBLE,
    calibrated_p_side DOUBLE,
    per_bet_ev DOUBLE,
    stake_units DOUBLE,
    ev_tier VARCHAR,
    status VARCHAR,
    rejection_reason VARCHAR,
    eligibility_label VARCHAR,
    snapshot_ts VARCHAR,
    -- VALUE is each game's day-before-kickoff lock (D33.2-01); the NAME keeps the retired
    -- 'freeze' wording by ruling: renaming a published cache column is HOST-07's schema change.
    freeze_ts VARCHAR,
    selected_odds DOUBLE,
    flat_stake DOUBLE,
    provenance VARCHAR,
    validation_type VARCHAR,
    decided_at_utc VARCHAR,
    -- MUTABLE grading facts (BET_LIST_GRADING_COLUMNS, 6) --------------------------
    grading_status VARCHAR,
    outcome BOOLEAN,
    clv DOUBLE,
    payout_flat DOUBLE,
    realized_units DOUBLE,
    graded_at TIMESTAMP
    -- No PRIMARY KEY (mirrors betting_bets WR-02): a flat per-bet ledger so multiple
    -- bets on one game_id are permitted and a re-bet is never silently dropped. A
    -- (game_id, target) PK would let INSERT OR REPLACE drop a same-target re-bet.
    -- This is the target-agnostic Phase-31 bet list (D31-20). It REPLACES the
    -- 0-row, 0-reader ou_bet_list table retired in the same commit.
);

CREATE TABLE IF NOT EXISTS available_bet_weeks (
    season INTEGER,
    week INTEGER,
    game_count INTEGER
    -- SCHEDULE-derived navigation (REVIEW-NAV). get_available_weeks reads predictions
    -- and get_available_seasons reads backtest_metrics, so a scheduled week with no
    -- prediction row would be absent from /bets navigation rather than selectable.
);

CREATE TABLE IF NOT EXISTS bet_week_freeze (
    season INTEGER,
    week INTEGER,
    -- VALUE is each game's day-before-kickoff lock (D33.2-01); the NAME keeps the retired
    -- 'freeze' wording by ruling: renaming a published cache column is HOST-07's schema change.
    latest_game_freeze_ts TIMESTAMP WITH TIME ZONE
    -- SCHEDULE-derived freshness source (REVIEW-STALE). The failure the stale-cache
    -- block guards is a MISSING bet-list insertion, in which state there may be no
    -- bet rows to read a freeze from -- so the freeze must not come from bet rows.
    -- The column is TIMESTAMPTZ, NOT a naive TIMESTAMP (CR-01). The upstream freeze
    -- is tz-aware UTC, and a naive column made DuckDB cast it through the SESSION
    -- TimeZone -- storing 18:00 for a 22:00Z instant on an America/New_York host and
    -- moving the staleness threshold by the server's own offset. NO SEMICOLONS in
    -- this comment: readers of CACHE_SCHEMA split it on that character.
);

CREATE TABLE IF NOT EXISTS bet_tracker_blocks (
    provenance VARCHAR,
    validation_type VARCHAR,
    bets_graded INTEGER,
    wins INTEGER,
    losses INTEGER,
    pushes INTEGER,
    hit_rate DOUBLE,
    flat_return_units DOUBLE
    -- PRECOMPUTED tracker blocks. The aggregation lives in backtest/bet_tracker.py and
    -- is called from pipeline/steps.py. api/cache.py imports no backtest module and
    -- computes nothing here (REVIEW-IMPORT, UIAP-01).
);
"""


# ---------------------------------------------------------------------------
# Population helpers
# ---------------------------------------------------------------------------


def _load_json(path: Path) -> Any:
    """Load and parse a JSON file."""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_feature_importances(
    conn: duckdb.DuckDBPyConnection,
    artifacts_dir: Path,
    latest: dict[str, str],
) -> int:
    """Load feature importances from artifact metadata.json files.

    For each target listed in latest.json, reads the metadata.json from
    the artifact directory and inserts feature importances. Since these
    are model-level (not per-game), we use target name as the game_id
    placeholder to conform to the schema.

    Returns the number of rows inserted.
    """
    total_rows = 0
    for target in ("wp", "ats", "ou"):
        artifact_name = latest.get(target)
        if artifact_name is None:
            logger.warning("No artifact found for target", target=target)
            continue

        metadata_path = artifacts_dir / artifact_name / "metadata.json"
        if not metadata_path.exists():
            logger.warning(
                "Metadata file not found",
                path=str(metadata_path),
                target=target,
            )
            continue

        metadata = _load_json(metadata_path)
        importances = metadata.get("feature_importances") or metadata.get(
            "top_feature_importances", {}
        )

        if not importances:
            logger.info("No feature importances in metadata", target=target)
            continue

        rows = [
            ("_model_", target, feat, float(imp)) for feat, imp in importances.items()
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO feature_importances VALUES (?, ?, ?, ?)",
            rows,
        )
        total_rows += len(rows)
        logger.info(
            "Loaded feature importances",
            target=target,
            count=len(rows),
        )

    return total_rows


def _load_backtest_predictions(
    conn: duckdb.DuckDBPyConnection,
    outputs_dir: Path,
) -> int:
    """Load backtest predictions from predictions_all.csv.

    Returns the number of rows inserted.
    """
    csv_path = outputs_dir / "predictions_all.csv"
    if not csv_path.exists():
        logger.warning("Backtest predictions file not found", path=str(csv_path))
        return 0

    df = pd.read_csv(csv_path)
    logger.info(
        "Read backtest predictions CSV",
        rows=len(df),
        columns=list(df.columns),
    )

    # Map alternate column names to expected names
    col_renames = {
        "model_value": "model_prob",
        "actual_value": "actual",
    }
    df = df.rename(columns={k: v for k, v in col_renames.items() if k in df.columns})

    # Derive 'week' from game_id if missing (format: YYYY_W01_AWAY@HOME)
    if "week" not in df.columns and "game_id" in df.columns:
        df["week"] = df["game_id"].str.extract(r"_W(\d+)_")[0].astype("Int64")

    required_cols = {"game_id", "season", "target", "model_prob", "actual"}
    if not required_cols.issubset(set(df.columns)):
        logger.warning(
            "Backtest predictions CSV missing required columns",
            missing=required_cols - set(df.columns),
        )
        return 0

    # Ensure week column exists (default to 0 if still missing)
    if "week" not in df.columns:
        df["week"] = 0

    # Fill optional columns
    if "probability_clv" not in df.columns:
        df["probability_clv"] = None
    if "has_closing_odds" not in df.columns:
        df["has_closing_odds"] = None

    select_cols = [
        "game_id",
        "season",
        "week",
        "target",
        "model_prob",
        "actual",
        "probability_clv",
        "has_closing_odds",
    ]
    subset = df[select_cols].copy()

    conn.execute("INSERT OR REPLACE INTO backtest_predictions SELECT * FROM subset")
    return len(subset)


def _load_metrics_summary(
    conn: duckdb.DuckDBPyConnection,
    outputs_dir: Path,
) -> int:
    """Load overall metrics from metrics_summary.json.

    Uses season=0 to indicate overall/aggregate metrics.
    Returns the number of rows inserted.
    """
    json_path = outputs_dir / "metrics_summary.json"
    if not json_path.exists():
        logger.warning("Metrics summary file not found", path=str(json_path))
        return 0

    data = _load_json(json_path)
    rows: list[tuple[int, str, str, float]] = []

    # The JSON structure may be nested by target or flat
    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, dict):
                # Nested: key is target, value is metrics dict
                for metric_name, metric_value in value.items():
                    if isinstance(metric_value, (int, float)):
                        rows.append(
                            (
                                0,
                                str(key),
                                str(metric_name),
                                float(metric_value),
                            )
                        )
            elif isinstance(value, (int, float)):
                # Flat: key is metric name
                rows.append((0, "overall", str(key), float(value)))

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO backtest_metrics VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def _load_season_metrics(
    conn: duckdb.DuckDBPyConnection,
    outputs_dir: Path,
) -> int:
    """Load per-season metrics from season_metrics.csv.

    Returns the number of rows inserted.
    """
    csv_path = outputs_dir / "season_metrics.csv"
    if not csv_path.exists():
        logger.warning("Season metrics file not found", path=str(csv_path))
        return 0

    df = pd.read_csv(csv_path)
    logger.info("Read season metrics CSV", rows=len(df), columns=list(df.columns))

    rows: list[tuple[int, str, str, float]] = []

    # Expect columns like: season, target, metric_name, metric_value
    # or: season, target, + metric columns as wide format
    if {"season", "target", "metric_name", "metric_value"}.issubset(set(df.columns)):
        # Long format
        for _, row in df.iterrows():
            rows.append(
                (
                    int(row["season"]),
                    str(row["target"]),
                    str(row["metric_name"]),
                    float(row["metric_value"]),
                )
            )
    elif "season" in df.columns and "target" in df.columns:
        # Wide format: melt non-id columns into long format
        id_cols = ["season", "target"]
        metric_cols = [c for c in df.columns if c not in id_cols]
        for _, row in df.iterrows():
            for metric_col in metric_cols:
                val = row[metric_col]
                if pd.notna(val):
                    with contextlib.suppress(ValueError, TypeError):
                        rows.append(
                            (
                                int(row["season"]),
                                str(row["target"]),
                                str(metric_col),
                                float(val),
                            )
                        )

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO backtest_metrics VALUES (?, ?, ?, ?)",
            rows,
        )
    return len(rows)


def _load_simulation_results(
    conn: duckdb.DuckDBPyConnection,
    outputs_dir: Path,
) -> int:
    """Load betting simulation results from betting_simulation.csv.

    Returns the number of rows inserted.
    """
    csv_path = outputs_dir / "betting_simulation.csv"
    if not csv_path.exists():
        logger.warning("Betting simulation file not found", path=str(csv_path))
        return 0

    df = pd.read_csv(csv_path)
    logger.info("Read simulation CSV", rows=len(df), columns=list(df.columns))

    rows: list[tuple[str, str, float]] = []

    if {"strategy", "metric_name", "metric_value"}.issubset(set(df.columns)):
        # Pre-aggregated long format
        for _, row in df.iterrows():
            rows.append(
                (
                    str(row["strategy"]),
                    str(row["metric_name"]),
                    float(row["metric_value"]),
                )
            )
    elif {"flat_stake", "kelly_stake", "payout_flat", "payout_kelly"}.issubset(
        set(df.columns)
    ):
        # Per-bet format from betting_simulation.csv -- aggregate into metrics
        for strategy in ("flat", "kelly"):
            stake_col = f"{strategy}_stake"
            payout_col = f"payout_{strategy}"
            total_bets = len(df)
            total_wagered = df[stake_col].sum()
            total_pnl = df[payout_col].sum()
            wins = (df[payout_col] > 0).sum()
            roi = (total_pnl / total_wagered * 100) if total_wagered else 0.0

            rows.extend(
                [
                    (strategy, "total_bets", float(total_bets)),
                    (strategy, "total_wagered", float(total_wagered)),
                    (strategy, "total_pnl", float(total_pnl)),
                    (
                        strategy,
                        "win_rate",
                        float(wins / total_bets * 100) if total_bets else 0.0,
                    ),
                    (strategy, "roi", float(roi)),
                ]
            )

            # Build equity curve from cumulative P&L
            cumulative = df[payout_col].cumsum()
            bankroll_start = 10000.0
            equity_rows: list[tuple[str, int, float]] = []
            for idx, val in enumerate(cumulative):
                equity_rows.append((strategy, idx, bankroll_start + float(val)))
            if equity_rows:
                conn.executemany(
                    "INSERT OR REPLACE INTO equity_curve VALUES (?, ?, ?)",
                    equity_rows,
                )
    elif "strategy" in df.columns:
        # Wide format
        id_cols = ["strategy"]
        metric_cols = [c for c in df.columns if c not in id_cols]
        for _, row in df.iterrows():
            for metric_col in metric_cols:
                val = row[metric_col]
                if pd.notna(val):
                    with contextlib.suppress(ValueError, TypeError):
                        rows.append(
                            (
                                str(row["strategy"]),
                                str(metric_col),
                                float(val),
                            )
                        )

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO simulation_results VALUES (?, ?, ?)",
            rows,
        )
    return len(rows)


def _load_betting_bets(
    conn: duckdb.DuckDBPyConnection,
    outputs_dir: Path,
) -> int:
    """Load the full per-bet rows from betting_simulation.csv into betting_bets.

    Supplements (does NOT replace) ``_load_simulation_results``: that loader
    aggregates the same CSV into the lossy ``simulation_results`` /
    ``equity_curve`` tables that ``/backtest`` still uses (D-19). This loader
    keeps every per-bet row -- including the low/negative-edge bets the
    simulation placed -- so the Phase 17 betting dashboard can break results
    down by target / season / edge bucket / outcome and apply the
    ``kelly_stake > 0`` "recommended" scope filter (D-15, D-16 REVISED, D-20).

    Critical ``outcome`` parsing (Pitfall 1): the CSV ``outcome`` column is an
    object column holding real Python ``bool`` values (3,111 rows) mixed with
    ``float`` NaN (47 push rows). It is NOT strings. ``astype(bool)`` would turn
    NaN into ``True``, ``bool("False")`` is ``True``, and ``.map({"True": ...})``
    returns all-NaN because the keys are bools not strings -- every naive cast is
    wrong. We normalize NaN to ``None`` BEFORE insert so DuckDB stores a nullable
    BOOLEAN (``True`` win / ``False`` loss / ``NULL`` push); downstream win-rate
    math then excludes pushes via identity checks.

    The 15 columns are selected in the exact CSV header order so the
    ``INSERT ... SELECT *`` lines up positionally with the betting_bets schema.

    Returns the number of rows inserted.
    """
    csv_path = outputs_dir / "betting_simulation.csv"
    if not csv_path.exists():
        logger.warning("Betting simulation file not found", path=str(csv_path))
        return 0

    df = pd.read_csv(csv_path)
    logger.info("Read betting bets CSV", rows=len(df), columns=list(df.columns))

    # outcome is object dtype: real bool + float NaN. Normalize NaN -> None so it
    # stores as a nullable DuckDB BOOLEAN (pushes become SQL NULL). Do NOT cast
    # with astype(bool) / bool(x) / .map -- each silently corrupts pushes.
    df["outcome"] = df["outcome"].where(df["outcome"].notna(), None)

    # Column order MUST match the CSV header and the betting_bets schema so the
    # positional INSERT ... SELECT * aligns.
    cols = [
        "game_id",
        "season",
        "week",
        "target",
        "bet_side",
        "model_value",
        "market_value",
        "edge",
        "slipped_line",
        "odds",
        "flat_stake",
        "kelly_stake",
        "outcome",
        "payout_flat",
        "payout_kelly",
    ]
    subset = df[cols].copy()

    # Plain INSERT: betting_bets is dropped/recreated each rebuild and has no
    # PRIMARY KEY, so every per-bet row is preserved (no silent same-target
    # re-bet drop). See schema note above (WR-02).
    conn.execute("INSERT INTO betting_bets SELECT * FROM subset")
    return len(subset)


# ---------------------------------------------------------------------------
# Generic bet-list materialization (Phase 31, plan 31-01; PROD-02, D31-20/22)
# ---------------------------------------------------------------------------

# The LOCKED column order, spelled in TWO named DISJOINT halves whose concatenation IS
# BET_LIST_COLUMNS. WHY the split (REVIEW-FWD-GRADE): a forward row is written at freeze, when the
# result does not yet exist. If the WHOLE row were immutable the row could never be graded and the
# self-grading honesty loop would be silently disabled. So the RECOMMENDATION facts below are
# immutable -- once written they are the record of what was recommended and when -- and the GRADING
# facts are the ONLY ones permitted to transition, and only out of ``pending``.
BET_LIST_IMMUTABLE_COLUMNS: list[str] = [
    "game_id",
    "season",
    "week",
    "target",
    "bet_side",
    "model_value",
    "market_value",
    "line",
    "slipped_line",
    "calibrated_p_side",
    "per_bet_ev",
    "stake_units",
    "ev_tier",
    "status",
    "rejection_reason",
    "eligibility_label",
    "snapshot_ts",
    # VALUE is each game's day-before-kickoff lock (D33.2-01); the NAME keeps the retired
    # 'freeze' wording by ruling: renaming a published cache column is HOST-07's schema change.
    "freeze_ts",
    "selected_odds",
    "flat_stake",
    "provenance",
    "validation_type",
    # THE ROW'S OWN OBSERVATION TIME (Phase 33, D33-27; OWNER RULING 2026-09-12). An ISO-8601
    # string with an explicit UTC offset saying when THIS run decided THIS bet. It is an
    # IMMUTABLE recommendation fact, not a grading one: a claim about when a pick was made must
    # never transition, or the forward ledger's central guarantee -- that no pick was made after
    # its own game froze -- becomes unfalsifiable.
    #
    # VARCHAR, NOT ``graded_at``'s TIMESTAMP, and that is the ruling rather than a preference. It
    # is compared against ``snapshot_ts`` and ``freeze_ts``, which are offset-carrying strings, so
    # one representation is what lets a single strict parse helper serve every comparison
    # (``scripts.ingest_historical_odds.require_aware_snapshot_ts``). A second representation
    # would need a second code path.
    #
    # LAST IN THE IMMUTABLE HALF, at index 22 of the locked order, so Phase 34's own bump for
    # ``arm``, the artifact/recipe/fill-convention stamps, the real-fill columns and the
    # reproduction key (D33-06) appends against a base that was written down.
    #
    # The 234 stored ``backtest_replay`` rows take NULL and are NEVER backfilled: every one is
    # already past its freeze, and filling them from ``snapshot_ts`` would stamp an observation
    # time nobody observed.
    "decided_at_utc",
]

# WHY these six, and not ``outcome`` alone (REVIEW-SCHEMA):
#   * ``outcome`` alone cannot carry the record. The retired ``materialize_ou_bet_list`` stored BOTH
#     a push and an ungraded forward game as SQL NULL, which the tracker must tell apart -- so
#     ``grading_status`` carries an explicit four-state vocabulary (pending / win / loss / push) and
#     ``outcome`` is retained only as the boolean the existing consumers expect.
#   * ``outcome`` also cannot yield a flat return under ASYMMETRIC American prices, which is why
#     ``selected_odds`` / ``flat_stake`` (immutable) and ``payout_flat`` / ``realized_units``
#     (grading) are stored per row -- exactly as the shipped ``betting_bets`` ledger already does.
BET_LIST_GRADING_COLUMNS: list[str] = [
    "grading_status",
    "outcome",
    "clv",
    "payout_flat",
    "realized_units",
    "graded_at",
]

# The single source of the column order shared by the schema, the explicit-column INSERT and the
# contract tests. 23 immutable + 6 grading = 29 (Phase 33 added ``decided_at_utc``; the width was
# 22 + 6 = 28 before it, which is the width every stored parquet still carries and the reason
# ``backtest.weekly_bet_list.read_bet_list_with_schema_shim`` exists).
BET_LIST_COLUMNS: list[str] = [
    *BET_LIST_IMMUTABLE_COLUMNS,
    *BET_LIST_GRADING_COLUMNS,
]

# The CLOSED four-state grading vocabulary. ``pending`` is an ungraded forward row; ``push`` is a
# graded tie. Both carry a NULL ``outcome`` and are distinguishable ONLY by this column.
GRADING_STATUS_PENDING = "pending"
GRADING_STATUS_WIN = "win"
GRADING_STATUS_LOSS = "loss"
GRADING_STATUS_PUSH = "push"
GRADING_STATUSES: tuple[str, ...] = (
    GRADING_STATUS_PENDING,
    GRADING_STATUS_WIN,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PUSH,
)

# The two orthogonal D31-22 honesty axes.
PROVENANCE_BACKTEST_REPLAY = "backtest_replay"
PROVENANCE_FORWARD = "forward"
VALIDATION_TYPE_CONTAMINATED = "contaminated"
VALIDATION_TYPE_CLEAN_HOLDOUT = "clean_holdout"
VALIDATION_TYPE_FORWARD_REALIZED = "forward_realized"

RUN_MODE_REPLAY = "replay"
RUN_MODE_FORWARD = "forward"

# The contaminated replay window (2021-2024 were burned across Phases 26/27) and the single
# unburned clean-holdout season this milestone spends exactly once.
_REPLAY_CONTAMINATED_SEASONS: frozenset[int] = frozenset({2021, 2022, 2023, 2024})
_REPLAY_CLEAN_HOLDOUT_SEASON = 2025

# The standalone CREATE, so materialize_bet_list can run against any connection (the web cache, or
# an in-memory test DB) without first building the whole CACHE_SCHEMA. This is the SAME definition
# embedded in CACHE_SCHEMA above (the LOCKED column order + NO PRIMARY KEY).
#
# THERE ARE THREE SITES, NOT TWO, AND THIS COMMENT IS WHAT MAKES THE THIRD DISCOVERABLE: the two
# ``conn.execute(BET_LIST_SCHEMA)`` calls share this constant, and CACHE_SCHEMA carries its own
# literal copy. A column added here and not there produces a real cache whose table is narrower
# than the explicit-column INSERT names, and the INSERT then fails in production rather than in
# the suite. The 29th column is ``decided_at_utc VARCHAR`` (Phase 33), and
# ``tests/api/test_cache_betting.py`` now BUILDS each site and compares the resulting column
# lists so the agreement is measured rather than asserted in prose.
BET_LIST_SCHEMA = """
CREATE TABLE IF NOT EXISTS bet_list (
    game_id VARCHAR,
    season INTEGER,
    week INTEGER,
    target VARCHAR,
    bet_side VARCHAR,
    model_value DOUBLE,
    market_value DOUBLE,
    line DOUBLE,
    slipped_line DOUBLE,
    calibrated_p_side DOUBLE,
    per_bet_ev DOUBLE,
    stake_units DOUBLE,
    ev_tier VARCHAR,
    status VARCHAR,
    rejection_reason VARCHAR,
    eligibility_label VARCHAR,
    snapshot_ts VARCHAR,
    -- VALUE is each game's day-before-kickoff lock (D33.2-01); the NAME keeps the retired
    -- 'freeze' wording by ruling: renaming a published cache column is HOST-07's schema change.
    freeze_ts VARCHAR,
    selected_odds DOUBLE,
    flat_stake DOUBLE,
    provenance VARCHAR,
    validation_type VARCHAR,
    decided_at_utc VARCHAR,
    grading_status VARCHAR,
    outcome BOOLEAN,
    clv DOUBLE,
    payout_flat DOUBLE,
    realized_units DOUBLE,
    graded_at TIMESTAMP
)
"""

# The two schedule-derived navigation/freshness tables and the precomputed tracker table, in their
# own LOCKED column orders (same explicit-column INSERT discipline as bet_list).
AVAILABLE_BET_WEEKS_COLUMNS: list[str] = ["season", "week", "game_count"]
AVAILABLE_BET_WEEKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS available_bet_weeks (
    season INTEGER,
    week INTEGER,
    game_count INTEGER
)
"""

# latest_game_freeze_ts: VALUE is the week's latest per-game day-before-kickoff lock (D33.2-01);
# the NAME keeps the retired 'freeze' wording by ruling -- a rename is HOST-07's schema change.
BET_WEEK_FREEZE_COLUMNS: list[str] = ["season", "week", "latest_game_freeze_ts"]
BET_WEEK_FREEZE_SCHEMA = """
CREATE TABLE IF NOT EXISTS bet_week_freeze (
    season INTEGER,
    week INTEGER,
    -- VALUE is each game's day-before-kickoff lock (D33.2-01); the NAME keeps the retired
    -- 'freeze' wording by ruling: renaming a published cache column is HOST-07's schema change.
    latest_game_freeze_ts TIMESTAMP WITH TIME ZONE
)
"""

BET_TRACKER_BLOCK_COLUMNS: list[str] = [
    "provenance",
    "validation_type",
    "bets_graded",
    "wins",
    "losses",
    "pushes",
    "hit_rate",
    "flat_return_units",
]
BET_TRACKER_BLOCKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS bet_tracker_blocks (
    provenance VARCHAR,
    validation_type VARCHAR,
    bets_graded INTEGER,
    wins INTEGER,
    losses INTEGER,
    pushes INTEGER,
    hit_rate DOUBLE,
    flat_return_units DOUBLE
)
"""

# The standalone ``cache_meta`` CREATE, so the per-week marker can be stamped against any
# connection (a partially built cache, or an in-memory test DB) without first building the whole
# CACHE_SCHEMA. This is the SAME definition embedded in CACHE_SCHEMA above, and
# ``tests/unit/test_bet_list_marker.py`` compares the two column by column so the convenience
# cannot silently drift into a second, different table.
CACHE_META_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_meta (
    key VARCHAR PRIMARY KEY,
    value VARCHAR,
    updated_at TIMESTAMP
)
"""

# The PREFIX of the cache_meta keys stamping when a week's bet list was last materialized. It is
# a prefix and not a key: see ``bet_list_populated_at_key``.
BET_LIST_POPULATED_AT_KEY = "bet_list_populated_at"

# The separator between the prefix and the (season, week) pair. Named so a reader of a raw
# cache_meta dump can decompose a key, and so the two sides -- the writer here and
# ``DataService.get_bet_list_populated_at`` -- cannot spell it differently.
BET_LIST_POPULATED_AT_SEPARATOR = ":"


def bet_list_populated_at_key(season: int, week: int) -> str:
    """The cache_meta key stamping when THAT week's bet list was materialized (D31-29).

    PER-WEEK, not global, and that is the whole design. ``/bets`` hard-blocks a week whose bet list
    was populated BEFORE that week's latest per-game line freeze, and the value it compares is this
    marker. Reading freshness off the generic ``last_updated`` timestamp was REJECTED because it
    advances whenever ANY cache table is repopulated: a run that populated predictions and then
    failed on the bet list would look fresh, and the guard would be defeated by the exact failure
    it exists to catch. Keying by week closes the second half of the same hole -- a run that
    populated week 2 must not vouch for a week 1 it never touched.

    Args:
        season: NFL season. Accepts a numpy integer (what a DataFrame column yields) as well as a
            Python one; both key the same cell.
        week: NFL week, same.

    Returns:
        ``bet_list_populated_at:<season>:<week>``.
    """
    return (
        f"{BET_LIST_POPULATED_AT_KEY}"
        f"{BET_LIST_POPULATED_AT_SEPARATOR}{int(season)}"
        f"{BET_LIST_POPULATED_AT_SEPARATOR}{int(week)}"
    )


def bet_list_source_is_absent(bet_list_df: pd.DataFrame | None) -> bool:
    """True when a cache population has NO bet rows to load -- ``None`` OR an EMPTY frame.

    THE WIDENING IS THE POINT (Plan 31-22, G-31-123b, T-31-118). :func:`populate_cache` has always
    carried an honest warning saying that a run with no bet list leaves the cache with zero rows
    and no populated-at marker, so ``/bets`` will REFUSE the current week rather than render it as
    one in which nothing was recommended. That warning was guarded on ``bet_list_df is None``, and
    BOTH production callers -- ``pipeline/steps.py::step_populate_web_cache`` and
    ``scripts/populate_cache.py`` -- pass frames from
    ``backtest.weekly_bet_list.read_bet_list_cache_sources``, which by documented design degrades
    an ABSENT artifact to an empty frame and never returns ``None``.

    So the one diagnostic that would have explained a zero-row cache was UNREACHABLE in
    production, and the case production actually hits -- a cold start with no durable artifact --
    logged the benign informational ``Bet list loaded count=0`` instead. The operator was not told
    anything was wrong; they were told a count.

    This predicate is a QUESTION about the source, not a policy: it changes no branch and no
    on-disk result. ``materialize_bet_list`` creates the table it writes into and
    ``materialize_bet_list_with_marker`` stamps nothing for an empty frame, so the ``None`` case
    and the empty-frame case already converge on the same cache state.

    Args:
        bet_list_df: The frame a caller is about to load, or ``None`` for "not supplied".

    Returns:
        True when there is nothing to load.
    """
    return bet_list_df is None or bet_list_df.empty


def bet_tracker_source_is_absent(bet_tracker_df: pd.DataFrame | None) -> bool:
    """True when a cache population has NO tracker blocks to load -- ``None`` OR an EMPTY frame.

    THE SYMMETRIC HALF (WR-04). :func:`bet_list_source_is_absent` made a zero-row BET list
    audible, and the tracker half had no equivalent -- so the one state nobody could see was a
    populated bet list beside an absent tracker: ``/bets`` renders the ranked list with a
    realized-versus-expected tracker computed over a DIFFERENT row set, or over none at all, and
    says nothing. That is precisely the "a page that has quietly stopped grading itself and does
    not say so" failure ``generate_weekly_bet_list``'s docstring claims to have closed, and the
    claim it makes -- "THE WRITE IS THE PAIR, AND THE PAIR IS INDIVISIBLE" -- is true of CALLERS
    and not of execution: the two writes are sequential, so a failure between them leaves a new
    parquet beside a stale or missing tracker.

    Deliberately a QUESTION and not a policy, exactly like its sibling: it changes no branch and
    no on-disk result. The mismatch it exists to surface is checked at the ONE population call
    site, where an absent tracker beside a POPULATED bet list is warned about and an absent
    tracker beside an absent bet list is not -- the latter being an ordinary cold start, already
    reported by the bet-list warning, and reporting it twice would just teach an operator to
    ignore both.

    Args:
        bet_tracker_df: The tracker blocks a caller is about to load, or ``None``.

    Returns:
        True when there is nothing to load.
    """
    return bet_tracker_df is None or bet_tracker_df.empty


# The ``status`` value that means a bet was actually PLACED. ``DataService.get_bet_list`` selects
# rows equal to it and ``DataService.get_suppressed_bets`` selects the exact COMPLEMENT
# (``IS DISTINCT FROM``, so a NULL status lands in the suppressed list rather than vanishing from
# both). Deriving both readers from ONE constant is what makes "never silently dropped" (SPEC R6)
# a partition of the table rather than two whitelists that can drift apart -- and drifting
# whitelists are the failure mode this repo has already had once.
BET_STATUS_LIVE = "live"


def classify_row_provenance(season: int, run_mode: str) -> tuple[str, str]:
    """Return the two orthogonal D31-22 honesty labels as ``(provenance, validation_type)``.

    They are ORTHOGONAL on purpose: ``provenance`` says HOW the row was produced (a backtest
    replay of a past week versus a genuine forward recommendation), while ``validation_type`` says
    what the row is EVIDENCE of (a contaminated split, the single clean 2025 holdout, or a live
    forward record). Collapsing them into one label is what would let a burned-holdout replay bet
    be read as a clean proof.

    Args:
        season: The NFL season of the row.
        run_mode: ``"replay"`` (a historical week re-selected) or ``"forward"`` (a live week).

    Returns:
        ``(provenance, validation_type)``.

    Raises:
        ValueError: on any combination outside the pre-registered vocabulary -- never a silent
            default, because a defaulted label is exactly the mislabel this function exists to
            prevent.
    """
    if run_mode == RUN_MODE_FORWARD:
        return PROVENANCE_FORWARD, VALIDATION_TYPE_FORWARD_REALIZED

    if run_mode == RUN_MODE_REPLAY:
        if season in _REPLAY_CONTAMINATED_SEASONS:
            return PROVENANCE_BACKTEST_REPLAY, VALIDATION_TYPE_CONTAMINATED
        if season == _REPLAY_CLEAN_HOLDOUT_SEASON:
            return PROVENANCE_BACKTEST_REPLAY, VALIDATION_TYPE_CLEAN_HOLDOUT
        msg = (
            f"classify_row_provenance: season {season!r} has no pre-registered validation_type in "
            "run_mode 'replay'; the replay window is "
            f"{sorted(_REPLAY_CONTAMINATED_SEASONS)} (contaminated) plus "
            f"{_REPLAY_CLEAN_HOLDOUT_SEASON} (clean_holdout). No default is applied."
        )
        raise ValueError(msg)

    msg = (
        f"classify_row_provenance: run_mode {run_mode!r} is outside the vocabulary "
        f"('{RUN_MODE_REPLAY}', '{RUN_MODE_FORWARD}'); refusing to guess a provenance label."
    )
    raise ValueError(msg)


def stamp_bet_list_provenance(
    bet_list_df: pd.DataFrame,
    run_mode: str,
) -> pd.DataFrame:
    """Stamp the two orthogonal D31-22 honesty labels onto every row of *bet_list_df*.

    THE SINGLE STAMPING SITE (Phase 31, plan 31-13). ``classify_row_provenance`` is called from
    HERE and from nowhere else in the write path, which
    ``tests/unit/test_provenance_mapping.py`` asserts by AST scan over the production tree and
    reports the count it found. The reason is not tidiness: a second stamping site is exactly how
    the two labels drift apart -- one caller updated when the vocabulary moves, another left
    writing the retired hardcoded ``PROVISIONAL_CONTAMINATED`` constant, and the clean 2025
    holdout rows published as contaminated. That is the inverse of the prohibition this phase
    exists to honour, and equally false.

    It is a LABELLING pass, not a metric: it derives no EV, no stake, no return and no tier, so
    the zero-math contract of this module is untouched (UIAP-01). Callers build the bet-list frame
    upstream, stamp it here, and hand the stamped frame to :func:`materialize_bet_list`.

    The classification is performed ONCE PER DISTINCT SEASON rather than once per row -- the
    mapping is pure and keyed only on (season, run_mode), so a per-row call would multiply the
    same lookup by the row count while proving nothing extra.

    Args:
        bet_list_df: The bet-list records, carrying at least a ``season`` column. Any existing
            ``provenance`` / ``validation_type`` values are OVERWRITTEN: this function is the
            authority on those two columns, not a filler for absent ones.
        run_mode: ``"replay"`` (historical weeks re-selected) or ``"forward"`` (a live week).

    Returns:
        A NEW frame carrying ``provenance`` and ``validation_type``. The caller's frame is not
        mutated.

    Raises:
        KeyError: if *bet_list_df* carries no ``season`` column.
        ValueError: if ANY season in the frame is unclassifiable under *run_mode*. The refusal is
            whole-frame: a row that cannot be labelled must never inherit its neighbours' label.
    """
    if "season" not in bet_list_df.columns:
        msg = (
            "stamp_bet_list_provenance: bet_list_df carries no 'season' column, so the D31-22 "
            "validation_type cannot be derived. No default is applied."
        )
        raise KeyError(msg)

    stamped = bet_list_df.copy()

    if stamped.empty:
        # Both columns still appear, so a zero-row frame has the same SHAPE as a populated one and
        # a downstream writer's column check behaves identically on it.
        stamped["provenance"] = pd.Series(dtype="object")
        stamped["validation_type"] = pd.Series(dtype="object")
        return stamped

    seasons = [int(season) for season in stamped["season"]]
    labels = {
        season: classify_row_provenance(season, run_mode)
        for season in sorted(set(seasons))
    }
    stamped["provenance"] = [labels[season][0] for season in seasons]
    stamped["validation_type"] = [labels[season][1] for season in seasons]
    return stamped


def assert_grading_transition(current_status: str, new_status: str) -> None:
    """Validate a ``grading_status`` transition on an existing bet_list row (REVIEW-FWD-GRADE).

    Only the GRADING half of the row may transition, and only forward out of ``pending``:
    ``pending -> win`` / ``loss`` / ``push`` is permitted; every other transition raises. An
    already-graded row moving back to ``pending`` is REFUSED -- ungrading a settled bet would let
    a losing record be quietly reopened, which is the honesty failure this schema exists to
    prevent. Re-asserting the SAME terminal status is a no-op and is permitted, so a re-run of the
    grader is idempotent.

    Raises:
        ValueError: on an out-of-vocabulary status or a forbidden transition.
    """
    for label, value in (
        ("current_status", current_status),
        ("new_status", new_status),
    ):
        if value not in GRADING_STATUSES:
            msg = (
                f"assert_grading_transition: {label}={value!r} is outside the closed grading "
                f"vocabulary {GRADING_STATUSES}."
            )
            raise ValueError(msg)

    if current_status == GRADING_STATUS_PENDING:
        return
    if new_status == current_status:
        return  # idempotent re-grade of an already-settled row
    msg = (
        f"assert_grading_transition: refusing {current_status!r} -> {new_status!r}; a graded row "
        "may not be re-graded or returned to 'pending' (REVIEW-FWD-GRADE)."
    )
    raise ValueError(msg)


def _validate_grading_status_column(bet_list_df: pd.DataFrame) -> None:
    """Reject any ``grading_status`` value outside the closed four-state vocabulary."""
    values = bet_list_df["grading_status"]
    offending = sorted(
        {str(v) for v in values.dropna().unique() if str(v) not in GRADING_STATUSES}
    )
    if offending:
        msg = (
            f"materialize_bet_list: grading_status carries out-of-vocabulary value(s) {offending}; "
            f"the closed vocabulary is {GRADING_STATUSES}. A fifth state would make a push and an "
            "ungraded forward row indistinguishable again (REVIEW-SCHEMA)."
        )
        raise ValueError(msg)


def _require_columns(
    df: pd.DataFrame, required: list[str], fn_name: str, arg: str
) -> None:
    """Raise a NAMED KeyError listing exactly which *required* columns are absent from *df*."""
    missing = [c for c in required if c not in df.columns]
    if missing:
        msg = (
            f"{fn_name}: {arg} missing required column(s) {missing}; every field must be present "
            "so the explicit-column INSERT can never silently mis-write."
        )
        raise KeyError(msg)


def _explicit_column_insert(
    conn: duckdb.DuckDBPyConnection,
    table: str,
    columns: list[str],
    frame: pd.DataFrame,
) -> int:
    """Insert *frame* into *table* naming every column explicitly, in the LOCKED order.

    The column list is spelled out on BOTH sides of the statement and the SELECT picks columns BY
    NAME, so a shuffled input frame still lands every value in the correct column. This is never a
    positional ``INSERT ... SELECT *`` (SPEC R4 ordering, T-31-02).
    """
    subset = frame[columns].copy()
    col_list = ", ".join(columns)
    conn.register("_gsd_insert_subset", subset)
    try:
        conn.execute(
            f"INSERT INTO {table} ({col_list}) SELECT {col_list} FROM _gsd_insert_subset"
        )
    finally:
        conn.unregister("_gsd_insert_subset")
    return len(subset)


def materialize_bet_list(
    conn: duckdb.DuckDBPyConnection,
    bet_list_df: pd.DataFrame,
) -> int:
    """Materialize the precomputed bet-list blob into the generic ``bet_list`` table (D31-20).

    ZERO-MATH CONTRACT (UIAP-01): this function WRITES what the selector produced and performs
    ZERO metric math. Every number arrives precomputed; nothing here re-derives an EV, a stake, a
    tier or a return.

    Integrity controls, carried verbatim in FORM from the retired ``materialize_ou_bet_list``:
      - an EMPTY frame returns 0 without error (SPEC R4 empty);
      - a MISSING column raises a NAMED ``KeyError`` listing exactly which ``BET_LIST_COLUMNS``
        fields are absent -- never a silent mis-write;
      - ``outcome`` is pushed to SQL NULL through a ``notna`` mask and never coerced (the
        ``betting_bets`` pitfall: ``astype(bool)`` turns NaN into True);
      - the INSERT names every column explicitly in the LOCKED ``BET_LIST_COLUMNS`` order and
        selects them BY NAME, never positionally;
      - ``grading_status`` is validated against the closed four-state vocabulary, so a fifth value
        raises rather than being stored.

    CALL-SITE CONTRACT -- getting this wrong DESTROYS the write (REVIEW-CACHE). ``conn`` MUST be
    the TEMPORARY database ``populate_cache`` builds, opened BEFORE the atomic swap.
    ``populate_cache`` ends with ``db_path.unlink()`` followed by ``tmp_path.rename(db_path)``, so
    anything written to the LIVE cache before a population run is DELETED by it. The durable source
    of these rows is therefore the bet-list artifact Plan 31-17 writes, which the Plan 31-18
    population step reads INTO the temp build. Passing a live-cache connection is a caller error.

    Args:
        conn: An open DuckDB connection -- the ``populate_cache`` TEMP database, or an in-memory
            test DB. The ``bet_list`` table is created if absent.
        bet_list_df: The precomputed per-bet records, carrying at least every
            ``BET_LIST_COLUMNS`` field. Column ORDER is irrelevant (the INSERT is
            explicit-column).

    Returns:
        The number of rows inserted.

    Raises:
        KeyError: if *bet_list_df* is missing a required ``BET_LIST_COLUMNS`` column.
        ValueError: if ``grading_status`` carries a value outside ``GRADING_STATUSES``.
    """
    conn.execute(BET_LIST_SCHEMA)

    # An empty frame writes zero rows without error (SPEC R4 empty). A week in which no candidate
    # cleared the floor is a first-class outcome, not a failure.
    if bet_list_df.empty:
        return 0

    _require_columns(
        bet_list_df, BET_LIST_COLUMNS, "materialize_bet_list", "bet_list_df"
    )

    # pandas-stubs widens DataFrame __getitem__ with a list key to DataFrame | Series, so the
    # .copy() result and the .where/.notna chain below lose their overload match at type-check
    # time though both are a DataFrame / Series at runtime (the same stub gap the gold/last5
    # renames elsewhere in this module carry).
    subset: pd.DataFrame = bet_list_df[BET_LIST_COLUMNS].copy()  # pyright: ignore[reportAssignmentType]
    _validate_grading_status_column(subset)

    # Push-NULL convention (the betting_bets pitfall): a push OR an ungraded forward game stores
    # ``outcome`` as SQL NULL, never coerced. The two are told apart by ``grading_status``.
    subset["outcome"] = subset["outcome"].where(subset["outcome"].notna(), None)  # pyright: ignore[reportAttributeAccessIssue]

    return _explicit_column_insert(conn, "bet_list", BET_LIST_COLUMNS, subset)


def materialize_bet_list_with_marker(
    conn: duckdb.DuckDBPyConnection,
    bet_list_df: pd.DataFrame,
    *,
    populated_at: datetime,
) -> int:
    """Insert the bet list, then stamp one per-week marker STRICTLY AFTER a successful insert.

    THE ORDERING IS THE POINT (D31-29). :func:`materialize_bet_list` validates the frame and then
    performs ONE ``INSERT``; every way it can fail -- a missing ``BET_LIST_COLUMNS`` field, a
    ``grading_status`` outside the closed vocabulary, a database error -- raises BEFORE this
    function reaches the stamp. A failed population therefore leaves the previous marker value in
    place, so the ``/bets`` hard-block fires deterministically instead of reading a failed run as
    fresh.

    ONE MARKER PER (season, week) PRESENT IN THE FRAME. A week the frame does not mention is not
    stamped, which is what stops a run on one week vouching for another. An EMPTY frame stamps
    NOTHING and raises nothing: zero rows is a first-class result (SPEC R4 empty), but it is also
    exactly the state a failed insertion leaves behind, so it must not be vouched for either --
    the page resolves that pair to the refusal by reading the schedule-derived ``bet_week_freeze``
    rather than the rows.

    CALL-SITE CONTRACT: as with :func:`materialize_bet_list`, ``conn`` MUST be the TEMPORARY
    database ``populate_cache`` builds. The marker travels with the rows it describes, so writing
    it to the live cache would leave it to be deleted by the next population run's rename.

    Args:
        conn: The ``populate_cache`` temp connection, or an in-memory test DB. Both the
            ``bet_list`` and ``cache_meta`` tables are created if absent.
        bet_list_df: The precomputed rows, carrying at least every ``BET_LIST_COLUMNS`` field.
        populated_at: The instant to stamp. Passed in rather than read from the clock so the
            marker and the run's ``last_updated`` describe the SAME instant, and so the ordering
            behaviour is testable.

    Returns:
        The number of rows inserted.

    Raises:
        KeyError: if *bet_list_df* is missing a required column (nothing is stamped).
        ValueError: if ``grading_status`` is outside ``GRADING_STATUSES`` (nothing is stamped).
    """
    conn.execute(CACHE_META_SCHEMA)
    inserted = materialize_bet_list(conn, bet_list_df)

    if bet_list_df.empty:
        return inserted

    weeks = (
        bet_list_df[["season", "week"]]
        .drop_duplicates()
        .sort_values(["season", "week"])
        .itertuples(index=False)
    )
    stamped = [
        [
            bet_list_populated_at_key(season, week),
            populated_at.isoformat(),
            populated_at,
        ]
        for season, week in weeks
    ]
    conn.executemany("INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)", stamped)
    logger.info(
        "Bet-list populated-at markers stamped",
        n_rows=inserted,
        n_weeks=len(stamped),
        populated_at=populated_at.isoformat(),
    )
    return inserted


def materialize_available_bet_weeks(
    conn: duckdb.DuckDBPyConnection,
    schedule_df: pd.DataFrame,
) -> int:
    """Materialize the SCHEDULE-derived ``available_bet_weeks`` navigation table (REVIEW-NAV).

    Every row is derived from *schedule_df* alone. This writer reads neither ``predictions`` nor
    ``backtest_metrics``: ``get_available_weeks`` reads the former and ``get_available_seasons``
    the latter, so a scheduled week carrying no prediction row would be ABSENT from ``/bets``
    navigation rather than selectable. Deriving from the schedule is what makes every scheduled
    week reachable.

    Args:
        conn: An open DuckDB connection. The table is created if absent.
        schedule_df: A schedule frame carrying ``season``, ``week`` and ``game_id``.

    Returns:
        The number of (season, week) rows inserted.
    """
    conn.execute(AVAILABLE_BET_WEEKS_SCHEMA)
    if schedule_df.empty:
        return 0

    _require_columns(
        schedule_df,
        ["season", "week", "game_id"],
        "materialize_available_bet_weeks",
        "schedule_df",
    )
    # pandas-stubs loses the DataFrameGroupBy overload through the ["col"].count() chain, so
    # .rename is unresolved at type-check time though it is a DataFrame at runtime.
    grouped = (
        schedule_df.groupby(["season", "week"], as_index=False)["game_id"]
        .count()
        .rename(columns={"game_id": "game_count"})  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        .sort_values(["season", "week"])
    )
    return _explicit_column_insert(
        conn, "available_bet_weeks", AVAILABLE_BET_WEEKS_COLUMNS, grouped
    )


def _to_utc_series(values: pd.Series) -> pd.Series:
    """Return *values* as a tz-aware UTC datetime Series (CR-01).

    Naive input is localized to UTC -- the convention ``api.routes.pages._as_utc`` documents --
    rather than being left for DuckDB to reinterpret through the session ``TimeZone``.
    """
    parsed = pd.to_datetime(values, errors="coerce")
    if getattr(parsed.dtype, "tz", None) is not None:
        return parsed.dt.tz_convert("UTC")
    return parsed.dt.tz_localize("UTC")


def materialize_bet_week_freeze(
    conn: duckdb.DuckDBPyConnection,
    schedule_df: pd.DataFrame,
) -> int:
    """Materialize the SCHEDULE-derived ``bet_week_freeze`` freshness table (REVIEW-STALE).

    The per-week freeze is the LATEST per-game freeze instant in that week -- a Thursday game
    locks days earlier than that week's Sunday games (D31-18), so a single week-level freeze
    claim would be false for every Thursday game. Since Plan 33.2-02 each per-game value is that
    game's day-before-kickoff LOCK (D33.2-01); the ``game_freeze_ts`` / ``latest_game_freeze_ts``
    NAMES keep the retired wording by ruling, because renaming a published cache column is a
    schema change owned by HOST-07, not a tidy-up. The per-game ``game_freeze_ts`` is computed
    UPSTREAM (Plan 31-18) and passed in; this writer only persists the per-week maximum, so
    ``api/cache.py`` stays a pure persistence layer.

    Every row is derived from *schedule_df* alone. It reads no bet rows, because the failure the
    stale-cache hard-block guards is precisely a MISSING bet-list insertion -- in which state
    there may be no bet rows to read a freeze from.

    THE INSTANT IS PERSISTED TZ-AWARE, AND THE ZONE IS PINNED HERE (CR-01). ``game_freeze_ts``
    arrives from ``build_bet_week_schedule`` as tz-aware UTC. Handing a tz-aware column to a naive
    ``TIMESTAMP`` column made DuckDB perform a ``TIMESTAMPTZ -> TIMESTAMP`` cast through the
    SESSION ``TimeZone`` setting, so a 22:00Z freeze was stored as 18:00 on an America/New_York
    host and read back naive -- moving the staleness threshold by the server's own UTC offset and
    defeating the hard-block. The column is now ``TIMESTAMP WITH TIME ZONE`` and the frame is
    normalized to UTC before the insert, so no implicit conversion is possible. A NAIVE input is
    localized to UTC here, which is the one place the "naive means UTC" convention can be applied
    before DuckDB has already reinterpreted it.

    Args:
        conn: An open DuckDB connection. The table is created if absent.
        schedule_df: A schedule frame carrying ``season``, ``week`` and ``game_freeze_ts``.

    Returns:
        The number of (season, week) rows inserted.
    """
    conn.execute(BET_WEEK_FREEZE_SCHEMA)
    if schedule_df.empty:
        return 0

    _require_columns(
        schedule_df,
        ["season", "week", "game_freeze_ts"],
        "materialize_bet_week_freeze",
        "schedule_df",
    )
    normalized = schedule_df[["season", "week", "game_freeze_ts"]].copy()
    normalized["game_freeze_ts"] = _to_utc_series(normalized["game_freeze_ts"])
    # Same pandas-stubs groupby gap as materialize_available_bet_weeks above.
    grouped = (
        normalized.groupby(["season", "week"], as_index=False)["game_freeze_ts"]
        .max()
        .rename(columns={"game_freeze_ts": "latest_game_freeze_ts"})  # pyright: ignore[reportCallIssue, reportAttributeAccessIssue]
        .sort_values(["season", "week"])
    )
    return _explicit_column_insert(
        conn, "bet_week_freeze", BET_WEEK_FREEZE_COLUMNS, grouped
    )


def materialize_bet_tracker_blocks(
    conn: duckdb.DuckDBPyConnection,
    tracker_df: pd.DataFrame,
) -> int:
    """Persist the PRECOMPUTED realized-vs-expected tracker blocks (REVIEW-IMPORT, UIAP-01).

    A PURE persistence writer. It takes a frame that is ALREADY aggregated and performs no
    arithmetic of its own -- no hit rate, no return, no count. The aggregation lives in
    ``backtest/bet_tracker.py`` and is called from ``pipeline/steps.py``, which is already
    permitted to import ``backtest``; ``api/cache.py`` imports no ``backtest`` module, so the
    ``tests/api/test_import_guard_bets.py`` sibling guard needs no allow-list entry from this
    phase. An earlier draft had this module import ``backtest.bet_tracker`` directly, which would
    have made that guard and the tracker plan mutually unsatisfiable; the resolution is this
    pure-persistence seam, not an allow-list widening.

    Args:
        conn: An open DuckDB connection. The table is created if absent.
        tracker_df: The precomputed per-provenance blocks carrying every
            ``BET_TRACKER_BLOCK_COLUMNS`` field.

    Returns:
        The number of rows inserted.
    """
    conn.execute(BET_TRACKER_BLOCKS_SCHEMA)
    if tracker_df.empty:
        return 0

    _require_columns(
        tracker_df,
        BET_TRACKER_BLOCK_COLUMNS,
        "materialize_bet_tracker_blocks",
        "tracker_df",
    )
    return _explicit_column_insert(
        conn, "bet_tracker_blocks", BET_TRACKER_BLOCK_COLUMNS, tracker_df
    )


# The edge band used to be computed HERE, by a ``_compute_confidence`` that was a byte-equivalent
# twin of ``scripts.generate_current_week_predictions.compute_confidence``. Plan 31-17 collapsed the
# two into ONE shared source and RENAMED the concept to an EDGE BAND (D31-23): see
# ``utils/edge_tier.py`` for the thresholds, the surviving three-incompatible-units defect
# (DEF-31-17, de-duplicated and renamed this phase, NOT repaired), which write is authoritative for
# which artifact, and why the stored ``*_confidence`` column names are deliberately left alone.
#
# THE SEPARATION ``backtest/simulation.py`` ASKS FOR IS NOW IN PLACE. That module warns that on the
# selector path the edge field carries per-bet EXPECTED VALUE rather than a points or probability
# edge, and that such rows must not reach this band. They do not: selector rows live in the
# ``bet_list`` table and are banded by ``assign_ev_tier``. The risk is recorded as CLOSED, and
# ``tests/api/test_cache_betting.py`` asserts the two are different functions with different names
# and that no call site feeds a per-bet EV into the edge band.


def _blend_weight_array(
    blend_data: dict[str, Any],
    target: str,
    season: pd.Series,
    week: pd.Series,
) -> np.ndarray:
    """Per-row model weight for *target*, honouring the DEPLOYED dynamic schedule (WR-03).

    The cache used to read only ``blend_data["weights"]`` and apply ONE scalar per target to every
    game. The deployed artifact ``blend_dynamic_20260606_020635`` carries a ``dynamic`` section
    with ``mode_by_target = {"wp": "dynamic", "ats": "dynamic", "ou": "dynamic"}``, which
    ``MarketBlender.from_artifacts`` auto-detects and applies as a per-week sigmoid. The
    current-week CSV went through that path and the cache did not, so ``/``, ``/games/{id}`` and
    both export endpoints published a DIFFERENT ``blended_*`` for the same game than the CSV --
    and the one on the website was the one the deployed blend config says is wrong. Measured on
    the live artifact: WP 0.5917 static against 0.5241 at week 1 and 0.6686 at week 18.

    The schedule is pure arithmetic and is reproduced here rather than imported, so UIAP-01's "no
    model import in the request path" is untouched. It mirrors
    ``models.blending.DynamicBlendWeights.get_weight`` exactly, including the pre-2021 17-week era
    (D-05) and the playoff clamp (D-04):

        max_week = 17 if season <= 2020 else 18
        t        = min(week, max_week) / max_week
        weight   = low + (high - low) / (1 + exp(-steepness * (t - midpoint)))

    A row whose target is not in dynamic mode, or whose season/week cannot be read, falls back to
    the STATIC weight for that target -- the same value this function returned for every row
    before. Falling back per row rather than for the whole frame keeps one unusable week from
    silently restaticizing the entire cache.

    Args:
        blend_data: The parsed ``blend_weights.json``.
        target: One of ``wp`` / ``ats`` / ``ou``.
        season: The season of each row.
        week: The week of each row.

    Returns:
        One weight per row, aligned with *season* / *week*.
    """
    static = float(blend_data["weights"][target])
    weights = np.full(len(season), static, dtype=float)

    dynamic = blend_data.get("dynamic")
    if not isinstance(dynamic, dict):
        return weights
    if dynamic.get("mode_by_target", {}).get(target) != "dynamic":
        return weights
    params = dynamic.get(target)
    if not isinstance(params, dict):
        return weights

    season_num = pd.to_numeric(season, errors="coerce").to_numpy(dtype="float64")
    week_num = pd.to_numeric(week, errors="coerce").to_numpy(dtype="float64")
    usable = np.isfinite(season_num) & np.isfinite(week_num) & (week_num >= 1)
    if not usable.any():
        return weights

    low = float(dynamic.get("low", 0.30))
    high = float(dynamic.get("high", 0.80))
    midpoint = float(params["midpoint"])
    steepness = float(params["steepness"])

    max_week = np.where(season_num[usable] <= 2020, 17.0, 18.0)
    t = np.minimum(week_num[usable], max_week) / max_week
    weights[usable] = low + (high - low) / (1.0 + np.exp(-steepness * (t - midpoint)))
    return weights


def _load_predictions(
    conn: duckdb.DuckDBPyConnection,
    outputs_dir: Path,
    silver_dir: Path,
) -> int:
    """Load predictions from backtest outputs into the predictions table.

    Transforms predictions_all.csv (one row per game per target: wp/ats/ou)
    into the predictions table (one row per game with all three targets as
    columns). Joins with silver games for team names, scores, and dates.

    Args:
        conn: Active DuckDB connection.
        outputs_dir: Directory containing predictions_all.csv.
        silver_dir: Directory containing games.parquet.

    Returns:
        Number of rows inserted.
    """
    csv_path = outputs_dir / "predictions_all.csv"
    if not csv_path.exists():
        logger.warning(
            "Predictions CSV not found, skipping predictions table",
            path=str(csv_path),
        )
        return 0

    df = pd.read_csv(csv_path)
    logger.info("Read predictions CSV for pivot", rows=len(df))

    # Extract week from game_id (format: YYYY_W01_AWAY@HOME)
    df["week"] = df["game_id"].str.extract(r"_W(\d+)_")[0].astype("Int64")

    # -- Pivot: filter each target and rename model columns --
    wp = df[df["target"] == "wp"][
        [
            "game_id",
            "season",
            "week",
            "model_value",
            "ml_home",
            "ml_away",
            "spread",
            "total",
            "probability_clv",
        ]
    ].copy()
    wp = wp.rename(columns={"model_value": "wp_prob"})

    ats = df[df["target"] == "ats"][["game_id", "model_spread"]].copy()
    ats = ats.rename(columns={"model_spread": "ats_prediction"})

    ou = df[df["target"] == "ou"][["game_id", "model_total"]].copy()
    ou = ou.rename(columns={"model_total": "ou_prediction"})

    # Merge targets into one row per game
    merged = wp.merge(ats, on="game_id", how="left").merge(ou, on="game_id", how="left")

    # Join with silver games for teams, scores, dates
    games_path = silver_dir / "games.parquet"
    if not games_path.exists():
        logger.warning("Silver games.parquet not found", path=str(games_path))
        return 0

    games = pd.read_parquet(games_path)
    game_cols = [
        "game_id",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "kickoff_et",
    ]
    merged = merged.merge(
        games[game_cols],
        on="game_id",
        how="left",
    )

    # Map to predictions schema
    merged["game_date"] = pd.to_datetime(merged["kickoff_et"])
    merged["status"] = np.where(merged["home_score"].notna(), "completed", "scheduled")
    merged["home_score"] = merged["home_score"].astype("Int64")
    merged["away_score"] = merged["away_score"].astype("Int64")
    merged["market_spread"] = merged["spread"]
    merged["market_total"] = merged["total"]
    merged["market_ml_home"] = merged["ml_home"].astype("Int64")
    merged["market_ml_away"] = merged["ml_away"].astype("Int64")

    # Compute edges
    # WP edge: use probability_clv if available, else compute from moneyline
    if "probability_clv" in merged.columns:
        merged["wp_edge"] = merged["probability_clv"]
    # Fill NaN wp_edge from moneyline-derived fair probability
    wp_edge_mask = merged["wp_edge"].isna() & merged["ml_home"].notna()
    if wp_edge_mask.any():
        fair_prob = merged.loc[wp_edge_mask, "ml_home"].apply(
            lambda ml: moneyline_to_probability(int(ml)) if pd.notna(ml) else np.nan
        )
        merged.loc[wp_edge_mask, "wp_edge"] = (
            merged.loc[wp_edge_mask, "wp_prob"] - fair_prob
        )

    # ATS edge: model home margin MINUS market home margin, in POINTS.
    #
    # THE UNIT IS POINTS (R13 / D33-05, Plan 33-10). There is NO DENOMINATOR. This used to be
    # ``(ats_prediction - market_spread) / abs(market_spread)`` -- an unbounded ratio whose
    # divisor is a point count that approaches zero -- and ``utils.edge_tier`` applies the same
    # 0.05 / 0.02 threshold pair to it that it applies to a WP probability. On a half-point line
    # a three-point disagreement scored 6.0, so the ATS band read "high" on 92.64 per cent of the
    # 1,087 gate-holdout rows carrying a computable edge. The band is rendered and sorted on by
    # ``/`` and ``/betting``: a published figure. The pre-repair distribution is recorded in
    # ``tests/phase33_state.ATS_BAND_SHARES_BEFORE`` as the "before" half of the label-movement
    # table Plan 33-16 publishes, and Plan 33-16 derives its frozen ATS thresholds from THIS
    # edge's distribution, which is why the repair lands first.
    #
    # THE ZERO BRANCH IS GONE (D33-31, owner ruling). A pick-em used to be forced to an edge of
    # 0.0. That was right while the edge was a ratio -- dividing by a zero line is undefined --
    # but a point margin has no such problem: a model predicting the home team by three against a
    # pick'em line disagrees with the market by exactly three points, and reporting zero there
    # would have misstated every nonzero disagreement on a pick'em. A pick'em is a REAL line, not
    # a missing one.
    #
    # ONE BRANCH SURVIVES: a game with NO STORED SPREAD has NO edge (NULL), because there is no
    # disagreement to measure. The absent case and the zero-line case stay distinct.
    #
    # THE SIGN INVARIANT. ``market_spread`` is the nflverse ``spread_line``, POSITIVE when the
    # home team is favored; ``ats_prediction`` is a predicted HOME MARGIN per the DEF-31-01
    # ruling. The two are on the same scale, so a POSITIVE ``ats_edge`` means the model expects
    # the home team to BEAT the line. (The earlier DEF-31-03 / WR-02 defect negated the stored
    # spread and so computed model PLUS market: a model agreeing exactly with the market scored
    # the largest edge the formula could produce. That negation, and the ``.clip(lower=0.5)``
    # denominator floor that followed it, are both long gone.)
    #
    # The identical expression lives in
    # ``scripts/generate_current_week_predictions.compute_edges``, which api/cache never reads --
    # which is exactly why a defect once survived here, in the copy the dashboard renders. Both
    # copies were repaired together under D33-22 and
    # ``tests/unit/test_current_week_ats_edge.py`` pins them value-for-value against one shared
    # table so they cannot drift again. This value drives ``ats_edge``, ``ats_confidence``, the
    # ``/`` page's ``sort=edge`` ordering and both exports.
    market_spread = merged["market_spread"]
    merged["ats_edge"] = (merged["ats_prediction"] - market_spread).where(
        market_spread.notna()
    )

    # O/U edge: model total vs market total, normalized
    market_total_safe = merged["market_total"].clip(lower=30)
    merged["ou_edge"] = (
        merged["ou_prediction"] - merged["market_total"]
    ) / market_total_safe

    # The EDGE BAND from the ONE shared source (D31-23). The stored column names keep their
    # historical ``*_confidence`` spelling -- renaming them would move every export header, which
    # is a published figure -- but the value they carry is the edge band, not a confidence.
    #
    # THE VALUES NOW COME FROM A PER-TARGET RULER (CLEAN-01, D33-22). The three columns above are
    # in three INCOMPATIBLE UNITS -- wp a probability delta, ats POINTS since Plan 33-10, ou a
    # fraction of the market total -- and until this plan one 0.05 / 0.02 pair was applied to all
    # three alike, which put 98.80% of ATS games in "high". ``target`` is REQUIRED with no
    # default, so the unit each column is in is stated at the call site rather than assumed; the
    # pairs themselves are frozen in ``backtest.cold_start_constants``. The column NAMES are
    # unchanged, deliberately: the band each carries moved, the header it is published under
    # did not.
    merged["wp_confidence"] = edge_tier_series(merged["wp_edge"], "wp")
    merged["ats_confidence"] = edge_tier_series(merged["ats_edge"], "ats")
    merged["ou_confidence"] = edge_tier_series(merged["ou_edge"], "ou")

    # Compute blended predictions from blend artifact JSON (UIAP-01: no model imports)
    merged["blended_wp"] = None
    merged["blended_ats"] = None
    merged["blended_ou"] = None
    try:
        latest_path = Path("artifacts") / "latest.json"
        if not latest_path.exists():
            raise FileNotFoundError("latest.json not found in artifacts/")
        manifest = json.loads(latest_path.read_text())
        if "blend" not in manifest:
            raise KeyError("'blend' not found in latest.json manifest")
        blend_dir = Path("artifacts") / manifest["blend"]
        weights_path = blend_dir / "blend_weights.json"
        if not weights_path.exists():
            raise FileNotFoundError(f"blend_weights.json not found in {blend_dir}")
        blend_data = json.loads(weights_path.read_text())
        # ``blend_data["weights"]`` is read PER TARGET inside _blend_weight_array, which returns
        # the deployed DYNAMIC per-week weight when the artifact carries one and falls back to
        # that static scalar otherwise (WR-03). Reading the scalar here and applying it to every
        # game is what made the cache disagree with the current-week CSV.
        if "weights" not in blend_data:
            raise KeyError("'weights' not found in blend_weights.json")

        # WP blending in log-odds space
        valid_ml = merged["ml_home"].notna() & merged["ml_away"].notna()
        if valid_ml.any():
            home_raw = merged.loc[valid_ml, "ml_home"].apply(
                lambda ml: moneyline_to_probability(int(ml))
            )
            away_raw = merged.loc[valid_ml, "ml_away"].apply(
                lambda ml: moneyline_to_probability(int(ml))
            )
            fair_home = home_raw / (home_raw + away_raw)
            clip_min, clip_max = 0.001, 0.999
            model_clipped = np.clip(
                merged.loc[valid_ml, "wp_prob"].values.astype(float), clip_min, clip_max
            )
            market_clipped = np.clip(fair_home.values.astype(float), clip_min, clip_max)
            w = _blend_weight_array(
                blend_data,
                "wp",
                merged.loc[valid_ml, "season"],
                merged.loc[valid_ml, "week"],
            )
            blended_wp_vals = expit(
                w * logit(model_clipped) + (1 - w) * logit(market_clipped)
            )
            merged.loc[valid_ml, "blended_wp"] = blended_wp_vals

        # ATS blending (linear)
        valid_spread = merged["market_spread"].notna()
        if valid_spread.any():
            w = _blend_weight_array(
                blend_data,
                "ats",
                merged.loc[valid_spread, "season"],
                merged.loc[valid_spread, "week"],
            )
            blended_ats_vals = w * merged.loc[
                valid_spread, "ats_prediction"
            ].values.astype(float) + (1 - w) * merged.loc[
                valid_spread, "market_spread"
            ].values.astype(float)
            merged.loc[valid_spread, "blended_ats"] = blended_ats_vals

        # O/U blending (linear)
        valid_total = merged["market_total"].notna()
        if valid_total.any():
            w = _blend_weight_array(
                blend_data,
                "ou",
                merged.loc[valid_total, "season"],
                merged.loc[valid_total, "week"],
            )
            blended_ou_vals = w * merged.loc[
                valid_total, "ou_prediction"
            ].values.astype(float) + (1 - w) * merged.loc[
                valid_total, "market_total"
            ].values.astype(float)
            merged.loc[valid_total, "blended_ou"] = blended_ou_vals

        logger.info(
            "Computed blended predictions for cache",
            blended_wp=int(valid_ml.sum()),
            blended_ats=int(valid_spread.sum()),
            blended_ou=int(valid_total.sum()),
        )
    except (FileNotFoundError, KeyError) as e:
        logger.warning(
            "Blend artifacts not available, blended columns will be NULL",
            error=str(e),
        )

    # Select final columns matching schema order
    final_cols = [
        "game_id",
        "season",
        "week",
        "game_date",
        "home_team",
        "away_team",
        "status",
        "home_score",
        "away_score",
        "wp_prob",
        "wp_confidence",
        "ats_prediction",
        "ats_confidence",
        "ou_prediction",
        "ou_confidence",
        "market_spread",
        "market_total",
        "market_ml_home",
        "market_ml_away",
        "wp_edge",
        "ats_edge",
        "ou_edge",
        "blended_wp",
        "blended_ats",
        "blended_ou",
    ]
    final_df = merged[final_cols].copy()

    conn.execute("INSERT OR REPLACE INTO predictions SELECT * FROM final_df")
    row_count = len(final_df)
    logger.info("Predictions table populated", count=row_count)
    return row_count


def _build_last5_records(games: pd.DataFrame) -> pd.DataFrame:
    """Build last-5 win/loss records for every team-game combination.

    Uses a vectorized approach: creates two rows per game (one for each team),
    computes results, then uses groupby + rolling to find the last 5 results
    prior to each game.

    Args:
        games: Silver games DataFrame with game_id, season, week, home_team,
               away_team, home_score, away_score.

    Returns:
        DataFrame with columns: game_id, team, is_home, last5 (JSON string).
    """
    # Build team-game results: two rows per game (home and away perspective)
    cols = ["game_id", "season", "week", "home_team", "home_score", "away_score"]
    home_rows = games[cols].copy()
    home_rows = home_rows.rename(columns={"home_team": "team"})
    home_rows["result"] = np.where(
        home_rows["home_score"] > home_rows["away_score"],
        "W",
        np.where(home_rows["home_score"] < home_rows["away_score"], "L", "T"),
    )
    home_rows["is_home"] = True

    away_cols = ["game_id", "season", "week", "away_team", "home_score", "away_score"]
    away_rows = games[away_cols].copy()
    away_rows = away_rows.rename(columns={"away_team": "team"})
    away_rows["result"] = np.where(
        away_rows["away_score"] > away_rows["home_score"],
        "W",
        np.where(away_rows["away_score"] < away_rows["home_score"], "L", "T"),
    )
    away_rows["is_home"] = False

    team_games = pd.concat([home_rows, away_rows], ignore_index=True)
    team_games = team_games.sort_values(
        ["team", "season", "week"],
    ).reset_index(drop=True)

    # For each team-game, collect the last 5 results from prior weeks in the same season
    records: list[dict[str, Any]] = []
    for (team, _season), group in team_games.groupby(["team", "season"]):
        group = group.sort_values("week")
        results_so_far: list[str] = []
        for _, row in group.iterrows():
            # last5 is from games before this one
            last5 = results_so_far[-5:] if results_so_far else []
            records.append(
                {
                    "game_id": row["game_id"],
                    "team": team,
                    "is_home": row["is_home"],
                    "last5": json.dumps(list(reversed(last5))),  # most recent first
                }
            )
            results_so_far.append(row["result"])

    return pd.DataFrame(records)


def _build_h2h_records(games: pd.DataFrame) -> pd.DataFrame:
    """Build head-to-head records for each game's matchup over last 5 seasons.

    Args:
        games: Silver games DataFrame.

    Returns:
        DataFrame with columns: game_id, h2h_record (JSON string).
    """
    records: list[dict[str, str]] = []

    # Pre-sort for efficiency
    games_sorted = games.sort_values(["season", "week"]).reset_index(drop=True)

    for _, game in games_sorted.iterrows():
        home = game["home_team"]
        away = game["away_team"]
        season = game["season"]
        week = game["week"]

        # Find prior matchups between these two teams
        gs = games_sorted
        is_matchup = ((gs["home_team"] == home) & (gs["away_team"] == away)) | (
            (gs["home_team"] == away) & (gs["away_team"] == home)
        )
        # Last 5 seasons (strictly prior)
        prior_seasons = (
            is_matchup & (gs["season"] >= season - 5) & (gs["season"] < season)
        )
        # Earlier weeks of the same season
        same_season = is_matchup & (gs["season"] == season) & (gs["week"] < week)
        prior = gs[prior_seasons | same_season]

        home_wins = 0
        away_wins = 0
        for _, prior_game in prior.iterrows():
            if prior_game["home_score"] is None or pd.isna(prior_game["home_score"]):
                continue
            if prior_game["home_team"] == home:
                if prior_game["home_score"] > prior_game["away_score"]:
                    home_wins += 1
                elif prior_game["away_score"] > prior_game["home_score"]:
                    away_wins += 1
            # Teams are reversed in this matchup
            elif prior_game["home_score"] > prior_game["away_score"]:
                away_wins += 1
            elif prior_game["away_score"] > prior_game["home_score"]:
                home_wins += 1

        records.append(
            {
                "game_id": game["game_id"],
                "h2h_record": json.dumps(
                    {"home_wins": home_wins, "away_wins": away_wins}
                ),
            }
        )

    return pd.DataFrame(records)


def _is_divisional_matchup(home: str, away: str) -> bool:
    """Return True iff *home* and *away* share an NFL division.

    UIAP-01 forbids the API layer from importing model/feature/rating code, so
    this reimplements ``ratings.elo.is_divisional_game`` using the canonical
    team metadata in ``utils.team_data`` (an allowed namespace) rather than
    importing ``ratings``. A division is uniquely identified by the
    (conference, division) PAIR -- "East" exists in both AFC and NFC -- so both
    must match. Unknown teams resolve to ``"Unknown"`` for both fields; we
    treat any unknown as non-divisional, matching ``is_divisional_game``'s
    ``None``-division short-circuit (verified identical across all 32x32 team
    pairs plus unknown-team edge cases).
    """
    home_conf, home_div = get_team_conference(home), get_team_division(home)
    away_conf, away_div = get_team_conference(away), get_team_division(away)
    if "Unknown" in (home_conf, home_div, away_conf, away_div):
        return False
    return home_conf == away_conf and home_div == away_div


def _compute_is_divisional(
    home_teams: pd.Series,
    away_teams: pd.Series,
) -> pd.Series:
    """Recompute the divisional flag from the canonical team mapping.

    Pairs the home and away abbreviations elementwise and evaluates each
    matchup with :func:`_is_divisional_matchup` (a pure division-lookup over
    ``utils.team_data`` that returns False for unknown teams). The gold
    ``is_divisional`` column must NOT be used: it is expanding-window z-score
    normalized, so a naive ``.astype(bool)`` flags nearly every game as
    divisional.

    Args:
        home_teams: Series of canonical home-team abbreviations.
        away_teams: Series of canonical away-team abbreviations.

    Returns:
        Boolean Series (real ``bool`` dtype, matching the BOOLEAN schema
        column) aligned to ``home_teams.index``.
    """
    divisional = [
        _is_divisional_matchup(home, away)
        for home, away in zip(home_teams, away_teams, strict=True)
    ]
    return pd.Series(divisional, index=home_teams.index, dtype=bool)


def _attach_raw_elo(
    context: pd.DataFrame,
    snapshots: pd.DataFrame,
) -> pd.DataFrame:
    """Attach RAW pre-game Elo to context from the silver Elo snapshot.

    LEFT-joins ``home_elo_pre`` / ``away_elo_pre`` from
    ``elo_game_snapshots.parquet`` onto ``context`` by ``game_id`` (renamed to
    ``home_elo`` / ``away_elo``). The gold ``home_elo`` / ``away_elo`` columns
    must NOT be displayed: they are expanding-window z-scores (e.g. -0.78) plus
    a 1500.0 placeholder leakage for the earliest games. This helper is the
    raw-Elo counterpart of ``_compute_is_divisional`` -- never trust the
    normalized gold columns for display values.

    The snapshot covers seasons 2018-2025 only (1991 of 6263 gold games). Games
    without a snapshot get NaN, which DuckDB stores as SQL NULL in the DOUBLE
    columns and the UI renders as "N/A". This is deliberate: NaN propagation
    simultaneously removes the z-score display AND the 1500.0 leakage, so do
    NOT fillna here.

    Only ``["game_id", "home_elo_pre", "away_elo_pre"]`` are selected before the
    merge so snapshot-only columns (season/week/team) cannot collide with the
    silver-merge columns and create ``_x`` / ``_y`` suffixes. The snapshot
    ``game_id`` is unique, so the LEFT join cannot duplicate context rows.

    Args:
        context: Game-context frame containing a ``game_id`` column. Callers
            must drop ``home_elo`` / ``away_elo`` from gold first so this merge
            cleanly ADDS the columns (no suffix collision).
        snapshots: Silver Elo-snapshot frame with ``game_id``,
            ``home_elo_pre``, ``away_elo_pre``.

    Returns:
        ``context`` with raw float ``home_elo`` / ``away_elo`` columns added
        (NaN for games lacking a snapshot).
    """
    # pandas-stubs widens DataFrame __getitem__ with a list key to
    # Series | DataFrame, so .rename loses its overload match at type-check
    # time though it is a DataFrame at runtime (same stub gap as the gold/last5
    # renames elsewhere in this module).
    elo_raw = snapshots[["game_id", "home_elo_pre", "away_elo_pre"]].rename(  # pyright: ignore[reportCallIssue]
        columns={"home_elo_pre": "home_elo", "away_elo_pre": "away_elo"}
    )
    return context.merge(elo_raw, on="game_id", how="left")


def _weather_severity_band(severity: pd.Series) -> pd.Series:
    """Map raw composite weather severity to a qualitative display band.

    The cache must display ``raw_weather_severity`` (the un-normalized gold
    passthrough, a composite in ~[0, 0.8]), NOT the normalized
    ``weather_severity_score`` -- the latter is an expanding-window z-score
    (~[-1.7, 3.7]) and surfaces as a meaningless "-0.2" in the UI. This helper
    is the weather-severity counterpart of ``_attach_raw_elo`` /
    ``_compute_is_divisional``: it derives a stored display value rather than
    trusting a normalized gold column.

    The 0.60 ("Significant") and 0.80 ("Extreme") anchors are the code's own
    thresholds from ``features.weather`` (``weather_game`` >= 0.6,
    ``extreme_weather`` >= 0.8), not invented cutoffs. The sub-0.6 cutoffs
    (0.10, 0.25) were calibrated from the silver ``weather_features``
    distribution -- they sit in the valleys between the 0.09 / 0.19 / 0.39
    severity clusters, giving every band non-trivial mass.

    Bins are LEFT-CLOSED (``right=False``) so a value exactly at a cutoff
    falls into the UPPER band: 0.25 -> "Moderate", 0.60 -> "Significant",
    0.80 -> "Extreme".

    Args:
        severity: Series of raw composite severities in ~[0, 1]. The gold
            fallback for missing/indoor games is 0.0, so this is never NaN.

    Returns:
        Object-dtype Series of band labels aligned to ``severity.index`` (so
        it stores as a DuckDB VARCHAR and assigns straight back as a column).
    """
    bins = [-float("inf"), 0.10, 0.25, 0.60, 0.80, float("inf")]
    labels = ["Clear", "Mild", "Moderate", "Significant", "Extreme"]
    # pandas-stubs widens pd.cut to include the retbins=True tuple overload, so
    # the chained .astype and the Series return lose their overload match at
    # type-check time though pd.cut returns a Series here at runtime (right=False
    # + no retbins). Same class of pandas-stubs gap as the .rename calls above.
    return pd.cut(severity, bins=bins, labels=labels, right=False).astype("object")  # pyright: ignore[reportAttributeAccessIssue, reportReturnType]


def _load_game_context(
    conn: duckdb.DuckDBPyConnection,
    gold_dir: Path,
    silver_dir: Path,
) -> int:
    """Load game context from gold features and silver games.

    Builds the game_context table with Elo ratings, venue info, weather,
    last-5 records, and head-to-head history.

    Args:
        conn: Active DuckDB connection.
        gold_dir: Directory containing features_wp.parquet.
        silver_dir: Directory containing games.parquet.

    Returns:
        Number of rows inserted.
    """
    gold_path = gold_dir / "features_wp.parquet"
    if not gold_path.exists():
        logger.warning(
            "Gold features not found, skipping game_context",
            path=str(gold_path),
        )
        return 0

    games_path = silver_dir / "games.parquet"
    if not games_path.exists():
        logger.warning(
            "Silver games not found, skipping game_context",
            path=str(games_path),
        )
        return 0

    # Read gold features for weather and venue. Two columns are deliberately
    # NOT read raw from the normalized gold:
    #   * home_elo/away_elo -- expanding-window z-scores (plus a 1500.0
    #     leakage); raw pre-game Elo is sourced from the silver snapshot via
    #     _attach_raw_elo after the silver games merge below.
    #   * weather_severity -- read from the un-normalized raw_weather_severity
    #     passthrough (the normalized weather_severity_score is a z-score and
    #     would surface a meaningless "-0.2" in the UI). The qualitative
    #     weather_severity_band is derived from it via _weather_severity_band.
    gold_df = pd.read_parquet(gold_path)
    gold_cols = [
        "game_id",
        "raw_weather_severity",
        "raw_wind_mph",
        "venue_outdoor",
    ]
    context = gold_df[gold_cols].copy()
    context = context.rename(
        columns={
            "raw_weather_severity": "weather_severity",
            "venue_outdoor": "is_outdoor",
            "raw_wind_mph": "wind_mph",
        }
    )

    # Read silver games for venue, scores, and last-5 computation
    games = pd.read_parquet(games_path)
    games_merge_cols = [
        "game_id",
        "venue",
        "venue_roof",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "season",
        "week",
    ]
    context = context.merge(
        games[games_merge_cols],
        on="game_id",
        how="left",
    )
    context = context.rename(
        columns={
            "venue": "venue_name",
            "venue_roof": "roof_type",
        }
    )

    # Derive surface from roof_type
    surface_map = {
        "indoor": "FieldTurf",
        "outdoor": "Grass",
        "retractable": "Grass",
    }
    context["surface"] = context["roof_type"].map(surface_map).fillna("Unknown")

    # Primetime: not easily derivable from current data, set False
    context["is_primetime"] = False

    # Derive is_outdoor from roof_type and recompute is_divisional from the
    # canonical team mapping. Both mirror the same pattern: never trust the
    # normalized gold columns for these flags.
    context["is_outdoor"] = context["roof_type"].isin(["outdoor", "retractable"])
    # pandas-stubs widens DataFrame __getitem__ to Series | DataFrame; the
    # columns are Series at runtime (same stub gap as _compute_confidence above).
    context["is_divisional"] = _compute_is_divisional(
        context["home_team"],  # pyright: ignore[reportArgumentType]
        context["away_team"],  # pyright: ignore[reportArgumentType]
    )

    # Source raw pre-game Elo from the silver snapshot (read directly per the
    # UIAP-01 self-contained-cache convention, mirroring the games.parquet read
    # above -- not data.storage). The gold home_elo/away_elo are z-scores plus a
    # 1500.0 leakage; snapshot-less games get NaN -> SQL NULL -> "N/A".
    snapshots = pd.read_parquet(silver_dir / "elo_game_snapshots.parquet")
    context = _attach_raw_elo(context, snapshots)

    # Derive the qualitative weather-severity band from the raw composite
    # (anchored to the features.weather 0.6/0.8 thresholds). Co-located with
    # the other cache-derived display fields; the raw weather_severity is
    # retained for the tooltip. pandas-stubs widens DataFrame __getitem__ to
    # Series | DataFrame, so the column is a Series at runtime (same stub gap
    # as the _compute_is_divisional call above).
    context["weather_severity_band"] = _weather_severity_band(
        context["weather_severity"]  # pyright: ignore[reportArgumentType]
    )

    # Compute last-5 records (vectorized by team+season)
    logger.info("Computing last-5 records...")
    last5_df = _build_last5_records(games)

    # Join home last5
    home_last5 = last5_df[last5_df["is_home"]][["game_id", "last5"]].rename(
        columns={"last5": "home_last5"}
    )
    context = context.merge(home_last5, on="game_id", how="left")

    # Join away last5
    away_last5 = last5_df[~last5_df["is_home"]][["game_id", "last5"]].rename(
        columns={"last5": "away_last5"}
    )
    context = context.merge(away_last5, on="game_id", how="left")

    # Fill missing last5 with empty array
    context["home_last5"] = context["home_last5"].fillna("[]")
    context["away_last5"] = context["away_last5"].fillna("[]")

    # Compute H2H records
    logger.info("Computing H2H records...")
    h2h_df = _build_h2h_records(games)
    context = context.merge(h2h_df, on="game_id", how="left")
    context["h2h_record"] = context["h2h_record"].fillna(
        json.dumps({"home_wins": 0, "away_wins": 0})
    )

    # Select final columns matching game_context schema
    final_cols = [
        "game_id",
        "home_elo",
        "away_elo",
        "home_last5",
        "away_last5",
        "h2h_record",
        "venue_name",
        "surface",
        "roof_type",
        "weather_severity",
        "weather_severity_band",
        "wind_mph",
        "is_outdoor",
        "is_divisional",
        "is_primetime",
    ]
    context_df = context[final_cols].copy()

    conn.execute("INSERT OR REPLACE INTO game_context SELECT * FROM context_df")
    row_count = len(context_df)
    logger.info("Game context table populated", count=row_count)
    return row_count


def _prerender_charts(
    conn: duckdb.DuckDBPyConnection,
) -> int:
    """Pre-render Plotly charts and store HTML divs in chart_cache.

    Uses api.charts to generate dashboard-compatible Plotly HTML divs
    from the already-populated DuckDB tables. No model classes are imported
    (UIAP-01 compliance).

    Args:
        conn: Active DuckDB connection (used for both reading data
              and writing chart_cache).

    Returns the number of charts cached.
    """
    from api.charts import prerender_charts_from_conn

    now = datetime.now(tz=UTC)

    # Generate charts directly from the write connection (can't open
    # a second read-only connection to the same DuckDB file)
    charts = prerender_charts_from_conn(conn)

    charts_cached = 0
    for chart_id, html_div in charts.items():
        conn.execute(
            "INSERT OR REPLACE INTO chart_cache VALUES (?, ?, ?)",
            [chart_id, html_div, now],
        )
        charts_cached += 1

    return charts_cached


# ---------------------------------------------------------------------------
# Main population function
# ---------------------------------------------------------------------------


def populate_cache(
    db_path: Path,
    artifacts_dir: Path,
    outputs_dir: Path,
    gold_dir: Path,
    silver_dir: Path = Path("data/silver"),
    *,
    bet_list_df: pd.DataFrame | None = None,
    bet_tracker_df: pd.DataFrame | None = None,
    bet_schedule_df: pd.DataFrame | None = None,
) -> None:
    """Populate the DuckDB web cache from artifacts and backtest outputs.

    Builds a complete cache by:
    1. Creating all tables per CACHE_SCHEMA
    2. Loading feature importances from model artifacts
    3. Loading backtest predictions, metrics, and simulation results
    4. Loading predictions (pivoted) and game context tables
    5. Loading the bet list, its per-week marker, the schedule-derived navigation and
       freshness tables, and the precomputed tracker blocks
    6. Pre-rendering chart placeholders
    7. Setting cache metadata

    Uses atomic rename: writes to a .tmp.duckdb file, then renames.

    THE BET-LIST SOURCES ARRIVE AS FRAMES, NEVER AS PATHS (REVIEW-CACHE, REVIEW-IMPORT). Two
    separate constraints produce that seam:

    * ``api/`` may import no ``backtest`` module (UIAP-01, ``tests/api/test_import_guard_bets.py``),
      so this module cannot know the artifact filenames, read the durable parquet, or derive a
      per-game freeze instant. The caller -- ``pipeline/steps.py`` or ``scripts/populate_cache.py``,
      both of which may import ``backtest`` -- reads them through
      ``backtest.weekly_bet_list.read_bet_list_cache_sources`` and hands the frames over. This
      module stays a pure persistence layer.
    * The rows must land in the TEMPORARY database built here, BEFORE the atomic swap. This
      function ends with ``db_path.unlink()`` then ``tmp_path.rename(db_path)``, so anything
      written to the LIVE cache beforehand is destroyed by it. Loading from the durable artifact
      into the temp build is what makes forward recommendation history survive a rebuild at all,
      since the database is replaced wholesale every time.

    Every bet-list argument is OPTIONAL and defaults to ``None``, which means "not supplied":
    the tables are still CREATED (so the page can tell an empty week from a pre-Phase-31 cache)
    but no rows and no marker are written. That is the honest representation of a run that did not
    produce a bet list, and it is the state the ``/bets`` hard-block refuses on rather than
    rendering as an empty week. An EMPTY FRAME reaches the same on-disk state and is now warned
    about through the same predicate (:func:`bet_list_source_is_absent`), because that -- not
    ``None`` -- is the case a cold start actually produces: both production callers pass frames
    from ``read_bet_list_cache_sources``, which degrades an absent artifact to zero rows.

    Args:
        db_path: Final path for the cache database (e.g. data/web_cache.duckdb).
        artifacts_dir: Root artifacts directory containing latest.json.
        outputs_dir: Backtest outputs directory.
        gold_dir: Gold data directory containing features_wp.parquet.
        silver_dir: Silver data directory containing games.parquet.
        bet_list_df: The durable bet-list rows, carrying every ``BET_LIST_COLUMNS`` field.
        bet_tracker_df: The PRECOMPUTED tracker blocks (``BET_TRACKER_BLOCK_COLUMNS``). Aggregated
            upstream; no arithmetic happens here.
        bet_schedule_df: The schedule frame carrying ``game_id``, ``season``, ``week`` and the
            per-game ``game_freeze_ts`` (since Plan 33.2-02 each game's day-before-kickoff lock; the
            name is kept by ruling -- a rename is HOST-07's schema change). Both schedule-derived
            tables are built from it, and the
            freeze table deliberately reads NO bet row -- the failure the stale-cache block guards
            is a MISSING bet-list insertion, in which state there may be no rows to read a freeze
            from (REVIEW-STALE).
    """
    tmp_path = db_path.with_suffix(".tmp.duckdb")
    logger.info(
        "Populating web cache",
        db_path=str(db_path),
        tmp_path=str(tmp_path),
        artifacts_dir=str(artifacts_dir),
        outputs_dir=str(outputs_dir),
    )

    # Ensure parent directory exists
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Remove stale temp file if present
    if tmp_path.exists():
        tmp_path.unlink()

    conn = duckdb.connect(str(tmp_path))
    try:
        # Create schema
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)

        # Load feature importances from artifacts
        latest_path = artifacts_dir / "latest.json"
        if latest_path.exists():
            latest = _load_json(latest_path)
            fi_count = _load_feature_importances(conn, artifacts_dir, latest)
            logger.info("Feature importances loaded", count=fi_count)
        else:
            logger.warning(
                "latest.json not found, skipping feature importances",
                path=str(latest_path),
            )

        # Load backtest data
        bp_count = _load_backtest_predictions(conn, outputs_dir)
        logger.info("Backtest predictions loaded", count=bp_count)

        ms_count = _load_metrics_summary(conn, outputs_dir)
        logger.info("Metrics summary loaded", count=ms_count)

        sm_count = _load_season_metrics(conn, outputs_dir)
        logger.info("Season metrics loaded", count=sm_count)

        sr_count = _load_simulation_results(conn, outputs_dir)
        logger.info("Simulation results loaded", count=sr_count)

        # Load full per-bet rows (Phase 17 betting dashboard, D-19). Supplements
        # the lossy simulation_results/equity_curve tables above -- both are kept
        # for /backtest. Must run before _prerender_charts so the betting chart
        # generators can read betting_bets.
        bb_count = _load_betting_bets(conn, outputs_dir)
        logger.info("Betting bets loaded", count=bb_count)

        # Load predictions from backtest data into the predictions table
        pred_table_count = _load_predictions(conn, outputs_dir, silver_dir)
        logger.info("Predictions loaded into cache", count=pred_table_count)

        # Load game context from gold features + silver games
        gc_count = _load_game_context(conn, gold_dir, silver_dir)
        logger.info("Game context loaded", count=gc_count)

        # ONE instant for this whole population run, resolved BEFORE the writes that stamp it.
        # The per-week bet-list markers and ``last_updated`` below therefore describe the same
        # moment, rather than two clock reads a few seconds apart.
        now = datetime.now(tz=UTC)

        # The bet-list cache sources, loaded into the TEMPORARY database before the swap
        # (Plan 31-18, REVIEW-CACHE). See this function's docstring for why they arrive as frames.
        # AUDIBLE FOR THE CASE PRODUCTION ACTUALLY HITS (Plan 31-22, T-31-118). This is an ADDED
        # STATEMENT and not a restructured branch: the ``is None`` branch below keeps its control
        # flow exactly, because the two cases already converge on the same on-disk result -- an
        # existing ``bet_list`` table with zero rows and no per-week marker. The defect was never
        # that the wrong branch ran; it was that nobody was told. See
        # ``bet_list_source_is_absent`` for why the ``is None`` guard alone was unreachable.
        if bet_list_source_is_absent(bet_list_df):
            logger.warning(
                "NO BET ROWS to load into the cache -- the durable bet-list artifact is absent or "
                "empty. The cache will carry zero bet rows and no populated-at marker, so /bets "
                "will REFUSE the current week rather than render it as one in which nothing was "
                "recommended. Produce the rows first with "
                "`uv run python scripts/generate_bet_list.py`, then re-run this population."
            )
        elif bet_tracker_source_is_absent(bet_tracker_df):
            # THE HALF-PAIR (WR-04). Bet rows WITHOUT tracker blocks is the state a run that
            # died between the two sequential artifact writes leaves behind, and it is the one
            # the operator is most likely to walk into: the run exits non-zero, they re-run the
            # SECOND command (`populate_cache.py`) as the documented two-command recovery
            # invites, and this copies both halves happily. /bets then serves the current
            # ranked list beside a realized-versus-expected tracker computed over a different
            # row set -- or none -- with no diagnostic anywhere. ``elif`` because an absent
            # tracker beside an absent bet list is an ordinary cold start that the branch above
            # already reported; saying it twice teaches an operator to ignore both.
            logger.warning(
                "BET ROWS WITHOUT TRACKER BLOCKS -- the durable pair is HALF PRESENT. The "
                "bet-list artifact has rows but the tracker artifact is absent or empty, which "
                "is what a run interrupted between the two writes leaves on disk. /bets will "
                "serve the ranked list with an EMPTY realized-vs-expected tracker and will not "
                "say so. Regenerate the pair with "
                "`uv run python scripts/generate_bet_list.py`, then re-run this population."
            )

        if bet_list_df is None:
            conn.execute(BET_LIST_SCHEMA)
            logger.warning(
                "No bet list supplied to cache population -- the cache will carry zero bet rows "
                "and no populated-at marker, and /bets will refuse the current week rather than "
                "render it as an empty one"
            )
        else:
            bl_count = materialize_bet_list_with_marker(
                conn, bet_list_df, populated_at=now
            )
            logger.info("Bet list loaded", count=bl_count)

        if bet_schedule_df is None:
            conn.execute(AVAILABLE_BET_WEEKS_SCHEMA)
            conn.execute(BET_WEEK_FREEZE_SCHEMA)
            logger.warning(
                "No bet schedule supplied to cache population -- /bets navigation and the "
                "per-week freeze the stale-cache block compares against will both be empty"
            )
        else:
            weeks_count = materialize_available_bet_weeks(conn, bet_schedule_df)
            freeze_count = materialize_bet_week_freeze(conn, bet_schedule_df)
            logger.info(
                "Bet navigation and freeze loaded",
                weeks=weeks_count,
                freeze_rows=freeze_count,
            )

        if bet_tracker_df is None:
            conn.execute(BET_TRACKER_BLOCKS_SCHEMA)
        else:
            tracker_count = materialize_bet_tracker_blocks(conn, bet_tracker_df)
            logger.info("Bet tracker blocks loaded", count=tracker_count)

        # Pre-render charts from populated data
        chart_count = _prerender_charts(conn)
        logger.info("Charts pre-rendered", count=chart_count)

        # Set cache metadata
        pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]

        season_range_row = conn.execute(
            "SELECT MIN(season), MAX(season) FROM predictions"
        ).fetchone()
        if season_range_row and season_range_row[0] is not None:
            season_range = f"{season_range_row[0]}-{season_range_row[1]}"
        else:
            season_range = "none"

        meta_rows = [
            ("last_updated", now.isoformat(), now),
            ("prediction_count", str(pred_count), now),
            ("season_range", season_range, now),
        ]
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            meta_rows,
        )

        logger.info(
            "Cache metadata set",
            prediction_count=pred_count,
            season_range=season_range,
        )

    finally:
        conn.close()

    # Atomic rename
    if db_path.exists():
        db_path.unlink()
    tmp_path.rename(db_path)

    logger.info("Web cache populated successfully", db_path=str(db_path))
