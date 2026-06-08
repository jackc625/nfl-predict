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

CREATE TABLE IF NOT EXISTS ou_bet_list (
    game_id VARCHAR,
    season INTEGER,
    week INTEGER,
    bet_side VARCHAR,
    totals_regime VARCHAR,
    subpop_label VARCHAR,
    model_total DOUBLE,
    closing_total DOUBLE,
    calibrated_p_side DOUBLE,
    per_bet_ev DOUBLE,
    slipped_line DOUBLE,
    kelly_stake DOUBLE,
    outcome BOOLEAN,
    clv DOUBLE,
    validation_type VARCHAR
    -- No PRIMARY KEY (mirrors betting_bets WR-02): a flat per-bet O/U ledger so a
    -- re-bet on a game is never silently dropped. This is the SECOND consumer of the
    -- BetSelector decision blob (Phase 27, BET-01). Phase 31 reads this table for the
    -- bet list. validation_type carries PROVISIONAL_CONTAMINATED so a burned-holdout
    -- bet can never be read as a clean proof (#6, D27-01).
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
# O/U BetSelector bet-list materialization (Phase 27, plan 27-04; BET-01, D27-15)
# ---------------------------------------------------------------------------

# The LOCKED column order for the sibling ou_bet_list table (mirrors the CREATE TABLE block above).
# The explicit-column INSERT spells this order out so the write is NEVER positional (#9, T-27-24):
# a shuffled input DataFrame still lands every value in the correct column. This list is the single
# source of the column order shared by the schema, the INSERT, and the smoke test.
OU_BET_LIST_COLUMNS: list[str] = [
    "game_id",
    "season",
    "week",
    "bet_side",
    "totals_regime",
    "subpop_label",
    "model_total",
    "closing_total",
    "calibrated_p_side",
    "per_bet_ev",
    "slipped_line",
    "kelly_stake",
    "outcome",
    "clv",
]

# The structural honesty label every materialized O/U bet-list row carries (#6, D27-01): a
# burned-2023-2024-holdout bet can never be read as a clean proof. The binding clean verdict is
# Phase 30; until then every persisted row is PROVISIONAL_CONTAMINATED.
_OU_VALIDATION_TYPE_PROVISIONAL = "PROVISIONAL_CONTAMINATED"

# The standalone CREATE for the sibling table, so materialize_ou_bet_list can run against any
# connection (the web cache, or an in-memory test DB) without first building the whole CACHE_SCHEMA.
# This is the SAME definition embedded in CACHE_SCHEMA above (the LOCKED column order + no PK).
OU_BET_LIST_SCHEMA = """
CREATE TABLE IF NOT EXISTS ou_bet_list (
    game_id VARCHAR,
    season INTEGER,
    week INTEGER,
    bet_side VARCHAR,
    totals_regime VARCHAR,
    subpop_label VARCHAR,
    model_total DOUBLE,
    closing_total DOUBLE,
    calibrated_p_side DOUBLE,
    per_bet_ev DOUBLE,
    slipped_line DOUBLE,
    kelly_stake DOUBLE,
    outcome BOOLEAN,
    clv DOUBLE,
    validation_type VARCHAR
)
"""


def materialize_ou_bet_list(
    conn: duckdb.DuckDBPyConnection,
    bet_list_df: pd.DataFrame,
) -> int:
    """Materialize the BetSelector bet-list blob into the sibling ou_bet_list table (BET-01, D27-15).

    The THIN cache-population fn -- BET-01's SECOND consumer (the backtest is the first). It WRITES
    the blob the BetSelector produced and performs ZERO metric math (UIAP-01: every number arrives
    precomputed; no request-path computation, no re-derivation). Phase 31 reads this table for the
    ``/bets`` list -- NO ``/bets`` UI and NO request-path computation are added here.

    Materialization (the integrity controls):
      - the table is created via ``CREATE TABLE IF NOT EXISTS ou_bet_list (...)`` with a LOCKED column
        order (no PRIMARY KEY, mirroring betting_bets so a re-bet on a game is never silently
        dropped, WR-02);
      - an EXPLICIT-COLUMN INSERT spells out the column list in the LOCKED ``OU_BET_LIST_COLUMNS``
        order -- NOT a positional ``INSERT ... SELECT *`` (#9, T-27-24); a shuffled input frame still
        lands every value in the correct column;
      - the push-NULL convention (``df["outcome"].where(notna, None)``) stores a push (or an ungraded
        forward game) as SQL NULL -- never coerced to win/loss (the betting_bets pitfall);
      - every row gets ``validation_type = PROVISIONAL_CONTAMINATED`` (#6).

    Args:
        conn: An open DuckDB connection (the web cache, or an in-memory test DB). The
            ``ou_bet_list`` table is created if absent.
        bet_list_df: The BetSelector ``selected`` records as a DataFrame, carrying at least the
            ``OU_BET_LIST_COLUMNS`` (game_id, season, week, bet_side, totals_regime, subpop_label,
            model_total, closing_total, calibrated_p_side, per_bet_ev, slipped_line, kelly_stake,
            outcome, clv). Column order is irrelevant (the INSERT is explicit-column).

    Returns:
        The number of rows inserted.

    Raises:
        KeyError: if ``bet_list_df`` is missing a required ``OU_BET_LIST_COLUMNS`` column (a named
            error, never a silent mis-write).
    """
    # Create the sibling table (additive; existing betting_bets consumers untouched).
    conn.execute(OU_BET_LIST_SCHEMA)

    # An empty BetSelector frame writes zero rows without error (#8 edge case).
    if bet_list_df.empty:
        return 0

    missing = [c for c in OU_BET_LIST_COLUMNS if c not in bet_list_df.columns]
    if missing:
        msg = (
            f"materialize_ou_bet_list: bet_list_df missing required column(s) {missing}; "
            "the BetSelector blob must carry every OU_BET_LIST_COLUMNS field (no silent mis-write)."
        )
        raise KeyError(msg)

    # Select the LOCKED column order (so the explicit-column INSERT below is order-stable) and add
    # the structural validation_type label. The fn performs ZERO metric math -- it only re-shapes the
    # precomputed blob and stamps the honesty label (UIAP-01).
    subset = bet_list_df[OU_BET_LIST_COLUMNS].copy()

    # Push-NULL convention (the betting_bets pitfall): a push / ungraded game (outcome None or NaN)
    # is stored as SQL NULL, never coerced. df.where(notna, None) leaves real booleans intact.
    subset["outcome"] = subset["outcome"].where(subset["outcome"].notna(), None)

    subset["validation_type"] = _OU_VALIDATION_TYPE_PROVISIONAL

    insert_cols = [*OU_BET_LIST_COLUMNS, "validation_type"]
    col_list = ", ".join(insert_cols)
    # EXPLICIT-COLUMN INSERT (#9, T-27-24): the column list is spelled out in the LOCKED order, so a
    # shuffled input frame still lands every value in the correct column (NOT positional SELECT *).
    conn.execute(f"INSERT INTO ou_bet_list ({col_list}) SELECT {col_list} FROM subset")
    return len(subset)


def _compute_confidence(edge: pd.Series) -> pd.Series:
    """Map absolute edge values to confidence labels.

    Args:
        edge: Series of edge values (can contain NaN).

    Returns:
        Series of "high", "medium", or "low" strings.
    """
    abs_edge = edge.abs()
    return pd.Series(
        np.where(
            abs_edge > 0.05,
            "high",
            np.where(abs_edge > 0.02, "medium", "low"),
        ),
        index=edge.index,
    )


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

    # ATS edge: model spread vs negative market spread, normalized
    market_spread_safe = merged["market_spread"].abs().clip(lower=0.5)
    merged["ats_edge"] = (
        merged["ats_prediction"] - (-merged["market_spread"])
    ) / market_spread_safe

    # O/U edge: model total vs market total, normalized
    market_total_safe = merged["market_total"].clip(lower=30)
    merged["ou_edge"] = (
        merged["ou_prediction"] - merged["market_total"]
    ) / market_total_safe

    # Confidence levels from edge magnitudes
    merged["wp_confidence"] = _compute_confidence(merged["wp_edge"])
    merged["ats_confidence"] = _compute_confidence(merged["ats_edge"])
    merged["ou_confidence"] = _compute_confidence(merged["ou_edge"])

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
        weights = blend_data["weights"]  # {"wp": float, "ats": float, "ou": float}

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
            w = weights["wp"]
            blended_wp_vals = expit(
                w * logit(model_clipped) + (1 - w) * logit(market_clipped)
            )
            merged.loc[valid_ml, "blended_wp"] = blended_wp_vals

        # ATS blending (linear)
        valid_spread = merged["market_spread"].notna()
        if valid_spread.any():
            w = weights["ats"]
            blended_ats_vals = w * merged.loc[
                valid_spread, "ats_prediction"
            ].values.astype(float) + (1 - w) * merged.loc[
                valid_spread, "market_spread"
            ].values.astype(float)
            merged.loc[valid_spread, "blended_ats"] = blended_ats_vals

        # O/U blending (linear)
        valid_total = merged["market_total"].notna()
        if valid_total.any():
            w = weights["ou"]
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
) -> None:
    """Populate the DuckDB web cache from artifacts and backtest outputs.

    Builds a complete cache by:
    1. Creating all tables per CACHE_SCHEMA
    2. Loading feature importances from model artifacts
    3. Loading backtest predictions, metrics, and simulation results
    4. Loading predictions (pivoted) and game context tables
    5. Pre-rendering chart placeholders
    6. Setting cache metadata

    Uses atomic rename: writes to a .tmp.duckdb file, then renames.

    Args:
        db_path: Final path for the cache database (e.g. data/web_cache.duckdb).
        artifacts_dir: Root artifacts directory containing latest.json.
        outputs_dir: Backtest outputs directory.
        gold_dir: Gold data directory containing features_wp.parquet.
        silver_dir: Silver data directory containing games.parquet.
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

        # Pre-render charts from populated data
        chart_count = _prerender_charts(conn)
        logger.info("Charts pre-rendered", count=chart_count)

        # Set cache metadata
        now = datetime.now(tz=UTC)
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
