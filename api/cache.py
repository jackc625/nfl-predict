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
import pandas as pd

from utils import get_logger

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
        importances = metadata.get("feature_importances", {})

        if not importances:
            logger.info(
                "No feature importances in metadata", target=target
            )
            continue

        rows = [
            ("_model_", target, feat, float(imp))
            for feat, imp in importances.items()
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
        logger.warning(
            "Backtest predictions file not found", path=str(csv_path)
        )
        return 0

    df = pd.read_csv(csv_path)
    logger.info(
        "Read backtest predictions CSV",
        rows=len(df),
        columns=list(df.columns),
    )

    # Map columns -- the CSV may have varying column names
    required_cols = {"game_id", "season", "week", "target", "model_prob", "actual"}
    if not required_cols.issubset(set(df.columns)):
        logger.warning(
            "Backtest predictions CSV missing required columns",
            missing=required_cols - set(df.columns),
        )
        return 0

    # Fill optional columns
    if "probability_clv" not in df.columns:
        df["probability_clv"] = None
    if "has_closing_odds" not in df.columns:
        df["has_closing_odds"] = None

    select_cols = [
        "game_id", "season", "week", "target",
        "model_prob", "actual", "probability_clv", "has_closing_odds",
    ]
    subset = df[select_cols].copy()

    conn.execute(
        "INSERT OR REPLACE INTO backtest_predictions "
        "SELECT * FROM subset"
    )
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
        logger.warning(
            "Metrics summary file not found", path=str(json_path)
        )
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
                        rows.append((0, str(key), str(metric_name), float(metric_value)))
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
        logger.warning(
            "Season metrics file not found", path=str(csv_path)
        )
        return 0

    df = pd.read_csv(csv_path)
    logger.info(
        "Read season metrics CSV", rows=len(df), columns=list(df.columns)
    )

    rows: list[tuple[int, str, str, float]] = []

    # Expect columns like: season, target, metric_name, metric_value
    # or: season, target, + metric columns as wide format
    if {"season", "target", "metric_name", "metric_value"}.issubset(set(df.columns)):
        # Long format
        for _, row in df.iterrows():
            rows.append((
                int(row["season"]),
                str(row["target"]),
                str(row["metric_name"]),
                float(row["metric_value"]),
            ))
    elif "season" in df.columns and "target" in df.columns:
        # Wide format: melt non-id columns into long format
        id_cols = ["season", "target"]
        metric_cols = [c for c in df.columns if c not in id_cols]
        for _, row in df.iterrows():
            for metric_col in metric_cols:
                val = row[metric_col]
                if pd.notna(val):
                    with contextlib.suppress(ValueError, TypeError):
                        rows.append((
                            int(row["season"]),
                            str(row["target"]),
                            str(metric_col),
                            float(val),
                        ))

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
        logger.warning(
            "Betting simulation file not found", path=str(csv_path)
        )
        return 0

    df = pd.read_csv(csv_path)
    logger.info(
        "Read simulation CSV", rows=len(df), columns=list(df.columns)
    )

    rows: list[tuple[str, str, float]] = []

    if {"strategy", "metric_name", "metric_value"}.issubset(set(df.columns)):
        for _, row in df.iterrows():
            rows.append((
                str(row["strategy"]),
                str(row["metric_name"]),
                float(row["metric_value"]),
            ))
    elif "strategy" in df.columns:
        # Wide format
        id_cols = ["strategy"]
        metric_cols = [c for c in df.columns if c not in id_cols]
        for _, row in df.iterrows():
            for metric_col in metric_cols:
                val = row[metric_col]
                if pd.notna(val):
                    with contextlib.suppress(ValueError, TypeError):
                        rows.append((
                            str(row["strategy"]),
                            str(metric_col),
                            float(val),
                        ))

    if rows:
        conn.executemany(
            "INSERT OR REPLACE INTO simulation_results VALUES (?, ?, ?)",
            rows,
        )
    return len(rows)


def _prerender_charts(
    conn: duckdb.DuckDBPyConnection,
) -> int:
    """Pre-render Plotly charts and store HTML divs in chart_cache.

    Reads data back from the already-populated DuckDB tables and generates
    charts without importing any model classes. Falls back gracefully if
    data is insufficient.

    Returns the number of charts cached.
    """
    now = datetime.now(tz=UTC)
    charts_cached = 0

    # We cannot call the backtest report generators directly since they
    # require BacktestResults/SimulationResults typed objects. Instead,
    # we store placeholder entries that will be populated by the populate
    # script if the full backtest results are available, or by a separate
    # chart generation step.
    #
    # For now, mark the chart_cache entries as needing generation.
    chart_ids = ["calibration", "clv_cumulative", "season_heatmap", "equity_curve"]
    for chart_id in chart_ids:
        # Check if we already have this chart
        existing = conn.execute(
            "SELECT chart_id FROM chart_cache WHERE chart_id = ?",
            [chart_id],
        ).fetchone()
        if existing is None:
            conn.execute(
                "INSERT INTO chart_cache VALUES (?, ?, ?)",
                [chart_id, "", now],
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
) -> None:
    """Populate the DuckDB web cache from artifacts and backtest outputs.

    Builds a complete cache by:
    1. Creating all tables per CACHE_SCHEMA
    2. Loading feature importances from model artifacts
    3. Loading backtest predictions, metrics, and simulation results
    4. Pre-rendering chart placeholders
    5. Setting cache metadata

    Uses atomic rename: writes to a .tmp.duckdb file, then renames.

    Args:
        db_path: Final path for the cache database (e.g. data/web_cache.duckdb).
        artifacts_dir: Root artifacts directory containing latest.json.
        outputs_dir: Backtest outputs directory.
        gold_dir: Gold data directory (for future feature matrix loading).
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

        # Pre-render chart placeholders
        chart_count = _prerender_charts(conn)
        logger.info("Chart placeholders created", count=chart_count)

        # Set cache metadata
        now = datetime.now(tz=UTC)
        pred_count = conn.execute(
            "SELECT COUNT(*) FROM backtest_predictions"
        ).fetchone()[0]

        season_range_row = conn.execute(
            "SELECT MIN(season), MAX(season) FROM backtest_predictions"
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
