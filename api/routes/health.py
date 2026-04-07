"""Health check endpoint with pipeline status, data freshness, and model status.

Expands the /health endpoint to include:
- Pipeline execution status from logs/friday_pipeline.json
- Data freshness from Silver layer file modification times
- Model artifact existence and age
- Deterministic summary status computation (D-17)

Fallback rules for pipeline log (addresses HIGH review concern):
- No log file exists     -> pipeline = None, does not affect status
- Corrupt log (bad JSON) -> pipeline = None, status = "degraded"
- Stale log (>7 days)    -> pipeline populated, status = "degraded"
- Valid log              -> pipeline populated normally
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import duckdb
from fastapi import APIRouter, Request

from api.dependencies import DB_PATH
from api.schemas import (
    DataFreshnessResponse,
    HealthResponse,
    ModelStatusResponse,
    PipelineStatusResponse,
)

router = APIRouter(tags=["health"])

# Absolute path to pipeline execution log (per D-18)
PIPELINE_LOG_PATH = Path("logs/friday_pipeline.json").resolve()

# Silver layer data file paths
SILVER_GAMES_PATH = Path("data/silver/games.parquet").resolve()
SILVER_ODDS_PATH = Path("data/silver/odds_snapshot.parquet").resolve()
SILVER_WEATHER_DIR = Path("data/silver/weather").resolve()

# Model artifact paths
MODEL_DIR = Path("artifacts/models").resolve()
MODEL_FILES = {
    "wp": MODEL_DIR / "wp_model.pkl",
    "ats": MODEL_DIR / "ats_model.pkl",
    "ou": MODEL_DIR / "ou_model.pkl",
}

# Freshness thresholds
DATA_FRESHNESS_HOURS = 168  # 7 days
LOG_STALENESS_SECONDS = 7 * 24 * 3600  # 7 days in seconds


def _read_pipeline_log() -> tuple[PipelineStatusResponse | None, bool, bool]:
    """Read and parse the pipeline execution log.

    Returns:
        Tuple of (pipeline_status, log_corrupt, log_stale).
        - pipeline_status: Populated PipelineStatusResponse or None
        - log_corrupt: True if log exists but has invalid JSON
        - log_stale: True if log is older than 7 days
    """
    if not PIPELINE_LOG_PATH.exists():
        return None, False, False

    try:
        raw = PIPELINE_LOG_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None, True, False

    # Parse start_time to check staleness
    log_stale = False
    last_run_time: datetime | None = None
    if data.get("start_time"):
        try:
            last_run_time = datetime.fromisoformat(data["start_time"])
            age_seconds = time.time() - last_run_time.timestamp()
            if age_seconds > LOG_STALENESS_SECONDS:
                log_stale = True
        except (ValueError, TypeError):
            pass

    pipeline = PipelineStatusResponse(
        last_run_status=data.get("status"),
        last_run_time=last_run_time,
        last_run_duration_ms=data.get("total_duration_ms"),
        last_run_season=data.get("season"),
        last_run_week=data.get("week"),
        forced=data.get("forced", False),
        warnings_count=len(data.get("warnings", [])),
        pid=data.get("pid"),
    )

    return pipeline, False, log_stale


def _check_file_age_hours(path: Path) -> float | None:
    """Return the age of a file in hours, or None if it does not exist."""
    try:
        return (time.time() - path.stat().st_mtime) / 3600
    except (FileNotFoundError, OSError):
        return None


def _get_data_freshness() -> DataFreshnessResponse:
    """Check modification times of Silver layer data files."""
    games_age = _check_file_age_hours(SILVER_GAMES_PATH)
    odds_age = _check_file_age_hours(SILVER_ODDS_PATH)

    # For weather directory, use the most recently modified file inside it
    weather_age: float | None = None
    try:
        if SILVER_WEATHER_DIR.exists() and SILVER_WEATHER_DIR.is_dir():
            weather_files = list(SILVER_WEATHER_DIR.iterdir())
            if weather_files:
                newest_mtime = max(f.stat().st_mtime for f in weather_files)
                weather_age = (time.time() - newest_mtime) / 3600
    except OSError:
        weather_age = None

    # all_fresh requires games and odds to be within threshold
    # None means file not found, which is not fresh
    all_fresh = (
        games_age is not None
        and games_age < DATA_FRESHNESS_HOURS
        and odds_age is not None
        and odds_age < DATA_FRESHNESS_HOURS
    )

    return DataFreshnessResponse(
        games_age_hours=round(games_age, 2) if games_age is not None else None,
        odds_age_hours=round(odds_age, 2) if odds_age is not None else None,
        weather_age_hours=round(weather_age, 2) if weather_age is not None else None,
        all_fresh=all_fresh,
    )


def _get_model_status() -> ModelStatusResponse:
    """Check existence and age of model artifacts."""
    results: dict[str, tuple[bool, float | None]] = {}

    for name, path in MODEL_FILES.items():
        if path.exists():
            age_hours = (time.time() - path.stat().st_mtime) / 3600
            results[name] = (True, round(age_hours, 2))
        else:
            results[name] = (False, None)

    wp_exists, wp_age = results["wp"]
    ats_exists, ats_age = results["ats"]
    ou_exists, ou_age = results["ou"]

    return ModelStatusResponse(
        wp_model_exists=wp_exists,
        ats_model_exists=ats_exists,
        ou_model_exists=ou_exists,
        wp_model_age_hours=wp_age,
        ats_model_age_hours=ats_age,
        ou_model_age_hours=ou_age,
        all_models_exist=wp_exists and ats_exists and ou_exists,
    )


def _read_last_updated(request: Request) -> datetime | None:
    """Read the cache_meta.last_updated value via the shared DuckDB connection.

    Acquires ``app.state.db_lock`` while touching ``app.state.db_conn`` so the
    read does not race the reconnect path in
    :func:`api.dependencies._reconnect_under_lock`. Without the lock, a
    concurrent reconnect could close the handle between the ``getattr`` and
    the ``execute`` call, raising ``duckdb.Error`` and causing this endpoint
    to silently report ``last_updated=None`` even though the cache is healthy.

    Falls back to opening a short-lived read-only connection only when the
    shared lifespan handle is unavailable (e.g. tests that bypass lifespan or
    a transient startup failure). The fallback path does not need the lock
    because the connection is local to this call.
    """
    lock = getattr(request.app.state, "db_lock", None)
    shared_conn = getattr(request.app.state, "db_conn", None)

    row: tuple | None = None
    try:
        if shared_conn is not None and lock is not None:
            with lock:
                # Re-fetch under the lock in case a reconnect replaced it
                # while we were waiting.
                shared_conn = getattr(request.app.state, "db_conn", None)
                if shared_conn is None:
                    return None
                row = shared_conn.execute(
                    "SELECT value FROM cache_meta WHERE key = 'last_updated'"
                ).fetchone()
        else:
            with duckdb.connect(str(DB_PATH), read_only=True) as conn:
                row = conn.execute(
                    "SELECT value FROM cache_meta WHERE key = 'last_updated'"
                ).fetchone()
    except (duckdb.Error, ValueError, OSError):
        # Cache exists but may be empty/corrupt -- still report ready
        return None

    if row and row[0]:
        try:
            return datetime.fromisoformat(row[0])
        except (ValueError, TypeError):
            return None
    return None


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """Return API health status including cache, pipeline, data, and model status.

    Reads cache metadata via the shared ``app.state.db_conn`` when available so
    the request does not have to open its own DuckDB handle. The handler must
    still respond gracefully when the DB file is missing or the connection is
    None — it bypasses the ``get_db`` dependency for that reason and reports
    ``cache_ready: false`` instead of raising 503 from the dependency.

    Summary status computation order (deterministic, most severe wins, D-17):
    Priority 1: unhealthy conditions (checked first)
      - Pipeline last_run_status == "failed"
      - Not all models exist
    Priority 2: degraded conditions
      - Pipeline last_run_status == "degraded"
      - Corrupt pipeline log
      - Stale pipeline log (>7 days old)
      - Not cache_ready
      - Data not fresh (not all_fresh)
    Priority 3: ok (default if no issues)
    """
    # File-level cache check (unchanged for backward compatibility)
    cache_ready = DB_PATH.exists()
    last_updated: datetime | None = None

    if cache_ready:
        # Routed through ``_read_last_updated`` so the shared-connection read
        # happens under ``app.state.db_lock`` (matches every other reader and
        # avoids racing the reconnect path in api.dependencies).
        last_updated = _read_last_updated(request)

    # Pipeline status from execution log
    pipeline, log_corrupt, log_stale = _read_pipeline_log()

    # Data freshness from Silver layer files
    data_freshness = _get_data_freshness()

    # Model artifact status
    model_status = _get_model_status()

    # Deterministic summary status computation (D-17)
    # Priority 1: unhealthy conditions (checked first)
    has_unhealthy = (
        pipeline is not None and pipeline.last_run_status == "failed"
    ) or not model_status.all_models_exist
    # Priority 2: degraded conditions
    has_degraded = (
        (pipeline is not None and pipeline.last_run_status == "degraded")
        or log_corrupt
        or log_stale
        or not cache_ready
        or not data_freshness.all_fresh
    )

    if has_unhealthy:
        status = "unhealthy"
    elif has_degraded:
        status = "degraded"
    else:
        status = "ok"

    return HealthResponse(
        status=status,
        cache_ready=cache_ready,
        last_updated=last_updated,
        pipeline=pipeline,
        data_freshness=data_freshness,
        model_status=model_status,
    )
