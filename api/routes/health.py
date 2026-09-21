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

from api.dependencies import DB_PATH, cache_file_changed, get_db
from api.exceptions import NFLPredictionAPIException
from api.schemas import (
    DataFreshnessResponse,
    HealthResponse,
    ModelStatusResponse,
    PipelineStatusResponse,
)
from utils.date_utils import ensure_utc_aware

router = APIRouter(tags=["health"])

# Absolute path to pipeline execution log (per D-18)
PIPELINE_LOG_PATH = Path("logs/friday_pipeline.json").resolve()

# The run status a live run ends in when it DROPPED games for a post-lock input (D33.2-05,
# Plan 33.2-03). The wire value of ``pipeline.steps.RunStatus.FINISHED_WITH_SKIPS``, spelled here
# as a literal because this layer stays stdlib-only (UIAP-01); a test asserts the two agree.
PIPELINE_STATUS_FINISHED_WITH_SKIPS = "finished_with_skips"

# Silver layer data file paths
SILVER_GAMES_PATH = Path("data/silver/games.parquet").resolve()
SILVER_ODDS_PATH = Path("data/silver/odds_snapshot.parquet").resolve()
SILVER_WEATHER_DIR = Path("data/silver/weather").resolve()

# Model artifact resolution -- the ACTIVE deployed models are the versioned
# artifact directories named in artifacts/latest.json (the same pointer the
# prediction pipeline loads via load_model_artifact), NOT fixed-name stubs under
# artifacts/models/. Resolved with pure stdlib (json/Path) to keep the api/ layer
# UIAP-01 compliant (no models/features/ratings imports).
ARTIFACTS_DIR = Path("artifacts").resolve()
ARTIFACTS_LATEST_PATH = ARTIFACTS_DIR / "latest.json"
MODEL_TARGETS = ("wp", "ats", "ou")


def _resolve_active_model_files() -> dict[str, Path]:
    """Map each target to its active ``model.pkl`` via artifacts/latest.json.

    Falls back to a non-existent placeholder path when latest.json is missing,
    unparseable, or lacks a target -- so health reports the model as absent
    rather than crashing.
    """
    try:
        latest = json.loads(ARTIFACTS_LATEST_PATH.read_text())
        if not isinstance(latest, dict):
            latest = {}
    except (OSError, json.JSONDecodeError):
        latest = {}

    resolved: dict[str, Path] = {}
    for target in MODEL_TARGETS:
        version = latest.get(target)
        if isinstance(version, str) and version:
            resolved[target] = ARTIFACTS_DIR / version / "model.pkl"
        else:
            resolved[target] = ARTIFACTS_DIR / "__missing__" / f"{target}_model.pkl"
    return resolved


MODEL_FILES = _resolve_active_model_files()

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

    # Parse start_time to check staleness.
    #
    # The pipeline log is documented to write UTC timestamps but pre-Phase
    # 15-04 writers may still emit naive ISO strings. ``datetime.fromisoformat``
    # returns a naive datetime in that case, and naive ``.timestamp()`` is
    # interpreted as LOCAL time per Python's documented behavior -- on a
    # non-UTC host the staleness check would be wrong by the local UTC offset
    # (e.g. up to 5 hours off in ET). Route through ``ensure_utc_aware`` so
    # naive datetimes are reinterpreted as UTC without shifting the wall
    # clock, matching the documented producer contract.
    log_stale = False
    last_run_time: datetime | None = None
    if data.get("start_time"):
        try:
            parsed = datetime.fromisoformat(data["start_time"])
            last_run_time = ensure_utc_aware(parsed)
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

    Falls back to opening a short-lived read-only connection when the shared
    lifespan handle is unavailable (e.g. tests that bypass lifespan or a
    transient startup failure). The fallback path does not need the lock because
    the connection is local to this call.

    THE SWAP CHECK IS WHY THIS ENDPOINT CAN BE TRUSTED AFTER A REBUILD (WR-02).
    ``/bets`` and every other page reach the cache through
    ``api.dependencies.get_db``, which reconnects when the file underneath the
    shared handle has been REPLACED. This function deliberately does NOT route
    through that dependency -- it must never 503 -- and so, until it consulted the
    same detector, it was the ONE reader that stayed blind to the swap: a
    ``populate_cache`` run would leave ``/health`` reporting the ``last_updated``
    of the DELETED pre-swap file until some unrelated request happened to hit a
    ``get_db``-backed route and trigger the reconnect. That is not a cosmetic
    lag. RUNBOOK.md operation 7 and PIPELINE.md stage 8 both send the operator
    HERE to confirm a rebuild took effect, so a pre-swap timestamp reads as "the
    population did not work" and invites them to run it again.

    WHY A DETECTED SWAP DELEGATES TO ``get_db`` RATHER THAN OPENING ITS OWN
    HANDLE. Reading the post-swap file from a short-lived connection of our own
    was tried first and DOES NOT WORK: DuckDB caches the database instance BY
    PATH within a process, so while the stale shared handle is still open, a
    fresh ``duckdb.connect`` on the same path hands back that same stale instance
    and this endpoint would keep reporting the pre-swap timestamp with more
    machinery behind it. The close-before-connect inside
    ``_reconnect_under_lock`` is the only thing that actually clears the instance
    (proven live in ``.planning/debug/bets-stale-cache-recovery-noop.md``,
    E1(D)/E4 ALT-3), so the repair has to be the same repair the pages get.

    That delegation is GUARDED, not unconditional, so this endpoint's contract is
    unchanged in every state except the one it was lying in.
    :func:`api.dependencies.cache_file_changed` can only return True when an
    identity was RECORDED, which only the two openers do -- so a caller that
    bypasses the lifespan never reaches ``get_db`` from here, and neither does
    the writer's unlink-to-rename window (no on-disk identity, so the detector
    returns False and the already-open handle keeps answering). Every failure
    ``get_db`` can raise, ``ModelUnavailableError`` included, is caught and
    degraded to the pre-existing behaviour: ``/health`` reports rather than
    raises.
    """
    lock = getattr(request.app.state, "db_lock", None)
    shared_conn = getattr(request.app.state, "db_conn", None)

    if shared_conn is not None and cache_file_changed(request):
        try:
            shared_conn = get_db(request)
        except (NFLPredictionAPIException, duckdb.Error, OSError):
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
      - Pipeline last_run_status == "finished_with_skips"
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
    # A run that FINISHED WITH SKIPS is DEGRADED here, decided explicitly (Plan 33.2-03 audit):
    # it completed, so it is not unhealthy, but at least one game the owner expected has no
    # prediction and no bet. Reporting it "ok" would repeat, on this endpoint, the clean-success
    # misreport the orchestrator's own alert branch exists to prevent (T-33.2-03-05).
    has_degraded = (
        (pipeline is not None and pipeline.last_run_status == "degraded")
        or (
            pipeline is not None
            and pipeline.last_run_status == PIPELINE_STATUS_FINISHED_WITH_SKIPS
        )
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
