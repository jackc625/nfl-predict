"""Pipeline health checker module -- importable refactor of scripts/health_check.py.

Provides two check modes:
- **Preflight** (3 checks): Quick, read-only, bounded-latency checks run before
  pipeline execution. Completes within seconds.
- **Post-run** (6 checks): Comprehensive checks run after pipeline completes.

Addresses review concern: Health check latency bounds (MEDIUM, Codex).
Pre-flight checks are strictly read-only -- no long-running queries, no external
API calls.

Health checks are NOT bypassed by --force. Even with --force, health checks run
and report results (advisory when forced, blocking when not forced).
"""

import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from conf.settings import get_settings
from data.storage import get_db_connection
from utils.logging_config import get_logger

logger = get_logger(__name__)


class PipelineHealthChecker:
    """Health checker for the Friday pipeline with preflight and post-run modes.

    Refactored from scripts/health_check.py HealthChecker with the same check
    methods, adapted for importable pipeline use.
    """

    def __init__(self) -> None:
        """Initialize health checker."""
        self.settings = get_settings()

    # ------------------------------------------------------------------
    # Public pipeline methods
    # ------------------------------------------------------------------

    def run_preflight(self) -> dict[str, Any]:
        """Run quick, read-only, bounded-latency pre-flight checks.

        Runs only:
        1. check_database_connectivity (single SELECT 1 query)
        2. check_model_artifacts (file existence only)
        3. check_disk_space (shutil.disk_usage, instant)

        Does NOT run: check_api_endpoints (may be slow), check_data_freshness
        (covered by staleness gate), check_prediction_pipeline (not relevant pre-run).

        Returns:
            {"status": "healthy"|"unhealthy", "checks": [...], "duration_ms": float}
        """
        start = time.time()

        checks = [
            self.check_database_connectivity(),
            self.check_model_artifacts(),
            self.check_disk_space(),
        ]

        overall = (
            "healthy" if all(c["status"] == "healthy" for c in checks) else "unhealthy"
        )
        duration = round((time.time() - start) * 1000, 2)

        logger.info(
            "Preflight health check complete",
            status=overall,
            duration_ms=duration,
        )

        return {
            "status": overall,
            "checks": checks,
            "duration_ms": duration,
        }

    def run_postrun(self) -> dict[str, Any]:
        """Run comprehensive post-run health checks (all 6 checks).

        Returns:
            {"status": "healthy"|"unhealthy", "checks": [...], "duration_ms": float}
        """
        start = time.time()

        checks = [
            self.check_database_connectivity(),
            self.check_data_freshness(),
            self.check_model_artifacts(),
            self.check_api_endpoints(),
            self.check_prediction_pipeline(),
            self.check_disk_space(),
        ]

        overall = (
            "healthy" if all(c["status"] == "healthy" for c in checks) else "unhealthy"
        )
        duration = round((time.time() - start) * 1000, 2)

        logger.info(
            "Post-run health check complete",
            status=overall,
            duration_ms=duration,
            checks_count=len(checks),
        )

        return {
            "status": overall,
            "checks": checks,
            "duration_ms": duration,
        }

    # ------------------------------------------------------------------
    # Individual check methods (adapted from scripts/health_check.py)
    # ------------------------------------------------------------------

    def check_database_connectivity(self) -> dict[str, Any]:
        """Check database connection and basic queries."""
        result: dict[str, Any] = {
            "name": "database_connectivity",
            "status": "unknown",
            "details": {},
            "duration_ms": 0,
        }
        start = time.time()

        try:
            db = get_db_connection()
            test_result = db.execute("SELECT 1 as test").fetchone()

            if test_result and test_result[0] == 1:
                tables = db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
                result.update(
                    {
                        "status": "healthy",
                        "details": {
                            "connection": "ok",
                            "table_count": len(tables),
                            "tables": [t[0] for t in tables[:5]],
                        },
                    }
                )
            else:
                result.update(
                    {
                        "status": "unhealthy",
                        "details": {"error": "Test query failed"},
                    }
                )
        except Exception as e:
            result.update({"status": "unhealthy", "details": {"error": str(e)}})

        result["duration_ms"] = round((time.time() - start) * 1000, 2)
        return result

    def check_data_freshness(self) -> dict[str, Any]:
        """Check if critical data is fresh enough."""
        result: dict[str, Any] = {
            "name": "data_freshness",
            "status": "unknown",
            "details": {},
            "duration_ms": 0,
        }
        start = time.time()

        try:
            db = get_db_connection()
            critical_tables = ["games", "odds_snapshot", "weather"]
            freshness_results = {}

            for table in critical_tables:
                try:
                    query = f"""
                    SELECT COUNT(*) as count,
                           MAX(created_at) as latest
                    FROM {table}
                    """
                    table_result = db.execute(query).fetchone()

                    if table_result:
                        count, latest = table_result
                        if latest:
                            latest_dt = pd.to_datetime(latest)
                            age_hours = (
                                datetime.now() - latest_dt
                            ).total_seconds() / 3600
                            freshness_results[table] = {
                                "count": count,
                                "latest": latest,
                                "age_hours": round(age_hours, 2),
                                "is_fresh": age_hours <= 6,
                            }
                        else:
                            freshness_results[table] = {
                                "count": count,
                                "latest": None,
                                "age_hours": float("inf"),
                                "is_fresh": False,
                            }
                except Exception as e:
                    freshness_results[table] = {"error": str(e), "is_fresh": False}

            all_fresh = all(
                t.get("is_fresh", False) for t in freshness_results.values()
            )
            result.update(
                {
                    "status": "healthy" if all_fresh else "unhealthy",
                    "details": {"tables": freshness_results, "all_fresh": all_fresh},
                }
            )
        except Exception as e:
            result.update({"status": "unhealthy", "details": {"error": str(e)}})

        result["duration_ms"] = round((time.time() - start) * 1000, 2)
        return result

    def check_model_artifacts(self) -> dict[str, Any]:
        """Check if model artifacts exist and are loadable."""
        result: dict[str, Any] = {
            "name": "model_artifacts",
            "status": "unknown",
            "details": {},
            "duration_ms": 0,
        }
        start = time.time()

        try:
            artifacts_dir = Path("artifacts/models")
            required_models = ["wp_model.pkl", "ats_model.pkl", "ou_model.pkl"]
            model_status = {}

            for model_file in required_models:
                model_path = artifacts_dir / model_file
                if model_path.exists():
                    stat = model_path.stat()
                    model_status[model_file] = {
                        "exists": True,
                        "size_mb": round(stat.st_size / (1024 * 1024), 2),
                        "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "age_hours": round((time.time() - stat.st_mtime) / 3600, 2),
                    }
                else:
                    model_status[model_file] = {"exists": False}

            all_exist = all(m.get("exists", False) for m in model_status.values())
            result.update(
                {
                    "status": "healthy" if all_exist else "unhealthy",
                    "details": {"models": model_status, "all_models_exist": all_exist},
                }
            )
        except Exception as e:
            result.update({"status": "unhealthy", "details": {"error": str(e)}})

        result["duration_ms"] = round((time.time() - start) * 1000, 2)
        return result

    def check_api_endpoints(self) -> dict[str, Any]:
        """Check critical API endpoints."""
        result: dict[str, Any] = {
            "name": "api_endpoints",
            "status": "unknown",
            "details": {},
            "duration_ms": 0,
        }
        start = time.time()

        try:
            from fastapi.testclient import TestClient

            from api.main import app

            client = TestClient(app)
            endpoints_to_check = [
                ("GET", "/health"),
                ("GET", "/current-week"),
                ("GET", "/games"),
            ]
            endpoint_results = {}

            for method, path in endpoints_to_check:
                try:
                    endpoint_start = time.time()
                    if method == "GET":
                        response = client.get(path)
                    else:
                        continue

                    response_time = round((time.time() - endpoint_start) * 1000, 2)
                    endpoint_results[f"{method} {path}"] = {
                        "status_code": response.status_code,
                        "response_time_ms": response_time,
                        "is_healthy": 200 <= response.status_code < 300,
                    }
                except Exception as e:
                    endpoint_results[f"{method} {path}"] = {
                        "error": str(e),
                        "is_healthy": False,
                    }

            all_healthy = all(
                ep.get("is_healthy", False) for ep in endpoint_results.values()
            )
            result.update(
                {
                    "status": "healthy" if all_healthy else "unhealthy",
                    "details": {
                        "endpoints": endpoint_results,
                        "all_healthy": all_healthy,
                    },
                }
            )
        except Exception as e:
            result.update({"status": "unhealthy", "details": {"error": str(e)}})

        result["duration_ms"] = round((time.time() - start) * 1000, 2)
        return result

    def check_prediction_pipeline(self) -> dict[str, Any]:
        """Check if prediction pipeline can generate predictions."""
        result: dict[str, Any] = {
            "name": "prediction_pipeline",
            "status": "unknown",
            "details": {},
            "duration_ms": 0,
        }
        start = time.time()

        try:
            predictions_dir = Path("outputs/predictions")
            prediction_files = list(
                predictions_dir.glob("current_predictions*.parquet")
            )

            if prediction_files:
                latest_file = max(prediction_files, key=lambda x: x.stat().st_mtime)
                file_stat = latest_file.stat()
                age_hours = (time.time() - file_stat.st_mtime) / 3600
                predictions_df = pd.read_parquet(latest_file)

                result.update(
                    {
                        "status": "healthy"
                        if age_hours <= 24 and len(predictions_df) > 0
                        else "unhealthy",
                        "details": {
                            "latest_file": latest_file.name,
                            "file_age_hours": round(age_hours, 2),
                            "prediction_count": len(predictions_df),
                            "file_size_mb": round(file_stat.st_size / (1024 * 1024), 2),
                            "is_recent": age_hours <= 24,
                            "has_predictions": len(predictions_df) > 0,
                        },
                    }
                )
            else:
                result.update(
                    {
                        "status": "unhealthy",
                        "details": {
                            "error": "No prediction files found",
                            "prediction_count": 0,
                        },
                    }
                )
        except Exception as e:
            result.update({"status": "unhealthy", "details": {"error": str(e)}})

        result["duration_ms"] = round((time.time() - start) * 1000, 2)
        return result

    def check_disk_space(self) -> dict[str, Any]:
        """Check disk space usage."""
        result: dict[str, Any] = {
            "name": "disk_space",
            "status": "unknown",
            "details": {},
            "duration_ms": 0,
        }
        start = time.time()

        try:
            directories = {
                "root": ".",
                "data": "data",
                "outputs": "outputs",
                "logs": "logs",
            }
            usage_details = {}

            for name, path in directories.items():
                if Path(path).exists():
                    usage = shutil.disk_usage(path)
                    free_gb = usage.free / (1024**3)
                    total_gb = usage.total / (1024**3)
                    used_pct = ((usage.total - usage.free) / usage.total) * 100

                    usage_details[name] = {
                        "free_gb": round(free_gb, 2),
                        "total_gb": round(total_gb, 2),
                        "used_pct": round(used_pct, 2),
                        "is_healthy": free_gb > 1.0,
                    }

            all_healthy = all(
                d.get("is_healthy", False) for d in usage_details.values()
            )
            result.update(
                {
                    "status": "healthy" if all_healthy else "unhealthy",
                    "details": {
                        "directories": usage_details,
                        "all_healthy": all_healthy,
                    },
                }
            )
        except Exception as e:
            result.update({"status": "unhealthy", "details": {"error": str(e)}})

        result["duration_ms"] = round((time.time() - start) * 1000, 2)
        return result
