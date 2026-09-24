"""Tests for the expanded /health API endpoint.

Covers pipeline status reading with explicit fallback rules, data freshness,
model status, and deterministic summary status computation (D-17).
"""

from __future__ import annotations

import json
import time
from datetime import UTC
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _make_pipeline_log(
    *,
    status: str = "success",
    start_time: str | None = None,
    total_duration_ms: float = 5000.0,
    season: int = 2025,
    week: int = 3,
    forced: bool = False,
    warnings: list[str] | None = None,
    pid: int = 12345,
) -> str:
    """Build a JSON pipeline log string for test mocking."""
    if start_time is None:
        from datetime import datetime

        start_time = datetime.now(tz=UTC).isoformat()

    data = {
        "status": status,
        "start_time": start_time,
        "end_time": start_time,
        "season": season,
        "week": week,
        "total_duration_ms": total_duration_ms,
        "forced": forced,
        "mode": "full",
        "pid": pid,
        "steps": [],
        "warnings": warnings or [],
        "error": None,
    }
    return json.dumps(data)


def _mock_stat(st_mtime: float | None = None, st_size: int = 1024):
    """Create a mock stat_result with given mtime."""
    mock = MagicMock()
    mock.st_mtime = st_mtime if st_mtime is not None else time.time()
    mock.st_size = st_size
    return mock


class TestHealthOk:
    """Tests for /health returning 'ok' status."""

    def test_health_ok_all_good(self, tmp_path: Path) -> None:
        """All checks pass: cache ready, models exist, data fresh, pipeline success."""
        log_content = _make_pipeline_log(status="success")
        now = time.time()
        recent_mtime = now - 3600  # 1 hour ago

        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            # Cache ready
            mock_db_path.exists.return_value = True
            # Patch out duckdb connect to avoid real DB
            with patch("api.routes.health.duckdb", create=True) as mock_duckdb:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = (
                    "2026-04-01T12:00:00",
                )
                mock_duckdb.connect.return_value = mock_conn

                # Pipeline log -- valid, recent
                mock_log_path.exists.return_value = True
                mock_log_path.read_text.return_value = log_content

                # Data freshness -- recent files
                mock_games_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_odds_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)

                mock_weather_dir.exists.return_value = True
                mock_weather_dir.is_dir.return_value = True
                mock_weather_file = MagicMock()
                mock_weather_file.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.iterdir.return_value = [mock_weather_file]

                # Model files -- all exist
                wp_path = MagicMock()
                wp_path.exists.return_value = True
                wp_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                ats_path = MagicMock()
                ats_path.exists.return_value = True
                ats_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                ou_path = MagicMock()
                ou_path.exists.return_value = True
                ou_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_model_files.__iter__ = MagicMock(
                    return_value=iter(["wp", "ats", "ou"])
                )
                mock_model_files.__getitem__ = lambda self, key: {
                    "wp": wp_path,
                    "ats": ats_path,
                    "ou": ou_path,
                }[key]
                mock_model_files.items.return_value = [
                    ("wp", wp_path),
                    ("ats", ats_path),
                    ("ou", ou_path),
                ]

                response = client.get("/health")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "ok"
            assert data["cache_ready"] is True
            assert data["pipeline"] is not None
            assert data["pipeline"]["last_run_status"] == "success"
            assert data["data_freshness"] is not None
            assert data["data_freshness"]["all_fresh"] is True
            assert data["model_status"] is not None
            assert data["model_status"]["all_models_exist"] is True


class TestHealthDegraded:
    """Tests for /health returning 'degraded' status."""

    def test_health_degraded_stale_data(self) -> None:
        """Data files are old (>168 hours). Status = degraded."""
        log_content = _make_pipeline_log(status="success")
        now = time.time()
        old_mtime = now - (200 * 3600)  # 200 hours ago

        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            mock_db_path.exists.return_value = True
            with patch("api.routes.health.duckdb", create=True) as mock_duckdb:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = (
                    "2026-04-01T12:00:00",
                )
                mock_duckdb.connect.return_value = mock_conn

                mock_log_path.exists.return_value = True
                mock_log_path.read_text.return_value = log_content

                # Old data files
                mock_games_path.stat.return_value = _mock_stat(st_mtime=old_mtime)
                mock_odds_path.stat.return_value = _mock_stat(st_mtime=old_mtime)
                mock_weather_dir.exists.return_value = True
                mock_weather_dir.is_dir.return_value = True
                mock_weather_file = MagicMock()
                mock_weather_file.stat.return_value = _mock_stat(st_mtime=old_mtime)
                mock_weather_dir.iterdir.return_value = [mock_weather_file]

                # Models exist
                wp_path = MagicMock()
                wp_path.exists.return_value = True
                wp_path.stat.return_value = _mock_stat()
                ats_path = MagicMock()
                ats_path.exists.return_value = True
                ats_path.stat.return_value = _mock_stat()
                ou_path = MagicMock()
                ou_path.exists.return_value = True
                ou_path.stat.return_value = _mock_stat()
                mock_model_files.items.return_value = [
                    ("wp", wp_path),
                    ("ats", ats_path),
                    ("ou", ou_path),
                ]

                response = client.get("/health")

            data = response.json()
            assert data["status"] == "degraded"
            assert data["data_freshness"]["all_fresh"] is False

    def test_health_corrupt_pipeline_log(self) -> None:
        """Pipeline log exists but is invalid JSON. pipeline=None, status=degraded."""
        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            mock_db_path.exists.return_value = True
            with patch("api.routes.health.duckdb", create=True) as mock_duckdb:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = (
                    "2026-04-01T12:00:00",
                )
                mock_duckdb.connect.return_value = mock_conn

                # Corrupt log
                mock_log_path.exists.return_value = True
                mock_log_path.read_text.return_value = "NOT VALID JSON {{{{"

                # Data fresh
                recent_mtime = time.time() - 3600
                mock_games_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_odds_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.exists.return_value = True
                mock_weather_dir.is_dir.return_value = True
                mock_weather_file = MagicMock()
                mock_weather_file.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.iterdir.return_value = [mock_weather_file]

                # Models exist
                wp_path = MagicMock()
                wp_path.exists.return_value = True
                wp_path.stat.return_value = _mock_stat()
                ats_path = MagicMock()
                ats_path.exists.return_value = True
                ats_path.stat.return_value = _mock_stat()
                ou_path = MagicMock()
                ou_path.exists.return_value = True
                ou_path.stat.return_value = _mock_stat()
                mock_model_files.items.return_value = [
                    ("wp", wp_path),
                    ("ats", ats_path),
                    ("ou", ou_path),
                ]

                response = client.get("/health")

            data = response.json()
            assert data["status"] == "degraded"
            assert data["pipeline"] is None

    def test_health_stale_pipeline_log(self) -> None:
        """Pipeline log exists, valid JSON, but start_time >7 days old. Status=degraded."""
        from datetime import datetime, timedelta

        old_time = (datetime.now(tz=UTC) - timedelta(days=10)).isoformat()
        log_content = _make_pipeline_log(status="success", start_time=old_time)

        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            mock_db_path.exists.return_value = True
            with patch("api.routes.health.duckdb", create=True) as mock_duckdb:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = (
                    "2026-04-01T12:00:00",
                )
                mock_duckdb.connect.return_value = mock_conn

                mock_log_path.exists.return_value = True
                mock_log_path.read_text.return_value = log_content

                # Data fresh
                recent_mtime = time.time() - 3600
                mock_games_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_odds_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.exists.return_value = True
                mock_weather_dir.is_dir.return_value = True
                mock_weather_file = MagicMock()
                mock_weather_file.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.iterdir.return_value = [mock_weather_file]

                # Models exist
                wp_path = MagicMock()
                wp_path.exists.return_value = True
                wp_path.stat.return_value = _mock_stat()
                ats_path = MagicMock()
                ats_path.exists.return_value = True
                ats_path.stat.return_value = _mock_stat()
                ou_path = MagicMock()
                ou_path.exists.return_value = True
                ou_path.stat.return_value = _mock_stat()
                mock_model_files.items.return_value = [
                    ("wp", wp_path),
                    ("ats", ats_path),
                    ("ou", ou_path),
                ]

                response = client.get("/health")

            data = response.json()
            assert data["status"] == "degraded"
            assert data["pipeline"] is not None
            assert data["pipeline"]["last_run_status"] == "success"


class TestHealthUnhealthy:
    """Tests for /health returning 'unhealthy' status."""

    def test_health_unhealthy_missing_models(self) -> None:
        """Model files don't exist. Status = unhealthy."""
        log_content = _make_pipeline_log(status="success")

        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            mock_db_path.exists.return_value = True
            with patch("api.routes.health.duckdb", create=True) as mock_duckdb:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = (
                    "2026-04-01T12:00:00",
                )
                mock_duckdb.connect.return_value = mock_conn

                mock_log_path.exists.return_value = True
                mock_log_path.read_text.return_value = log_content

                recent_mtime = time.time() - 3600
                mock_games_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_odds_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.exists.return_value = True
                mock_weather_dir.is_dir.return_value = True
                mock_weather_file = MagicMock()
                mock_weather_file.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.iterdir.return_value = [mock_weather_file]

                # Models do NOT exist
                wp_path = MagicMock()
                wp_path.exists.return_value = False
                ats_path = MagicMock()
                ats_path.exists.return_value = False
                ou_path = MagicMock()
                ou_path.exists.return_value = False
                mock_model_files.items.return_value = [
                    ("wp", wp_path),
                    ("ats", ats_path),
                    ("ou", ou_path),
                ]

                response = client.get("/health")

            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["model_status"]["all_models_exist"] is False

    def test_health_status_priority_unhealthy_wins(self) -> None:
        """Pipeline failed AND data stale. Status = unhealthy (not degraded)."""
        log_content = _make_pipeline_log(status="failed")
        now = time.time()
        old_mtime = now - (200 * 3600)  # 200 hours ago

        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            mock_db_path.exists.return_value = True
            with patch("api.routes.health.duckdb", create=True) as mock_duckdb:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = (
                    "2026-04-01T12:00:00",
                )
                mock_duckdb.connect.return_value = mock_conn

                mock_log_path.exists.return_value = True
                mock_log_path.read_text.return_value = log_content

                # Stale data
                mock_games_path.stat.return_value = _mock_stat(st_mtime=old_mtime)
                mock_odds_path.stat.return_value = _mock_stat(st_mtime=old_mtime)
                mock_weather_dir.exists.return_value = True
                mock_weather_dir.is_dir.return_value = True
                mock_weather_file = MagicMock()
                mock_weather_file.stat.return_value = _mock_stat(st_mtime=old_mtime)
                mock_weather_dir.iterdir.return_value = [mock_weather_file]

                # Models exist
                wp_path = MagicMock()
                wp_path.exists.return_value = True
                wp_path.stat.return_value = _mock_stat()
                ats_path = MagicMock()
                ats_path.exists.return_value = True
                ats_path.stat.return_value = _mock_stat()
                ou_path = MagicMock()
                ou_path.exists.return_value = True
                ou_path.stat.return_value = _mock_stat()
                mock_model_files.items.return_value = [
                    ("wp", wp_path),
                    ("ats", ats_path),
                    ("ou", ou_path),
                ]

                response = client.get("/health")

            data = response.json()
            # unhealthy wins over degraded (failed pipeline > stale data)
            assert data["status"] == "unhealthy"


class TestHealthEdgeCases:
    """Tests for edge cases in /health endpoint."""

    def test_health_no_pipeline_log(self) -> None:
        """Pipeline log doesn't exist. pipeline=None. Status based on other checks."""
        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            mock_db_path.exists.return_value = True
            with patch("api.routes.health.duckdb", create=True) as mock_duckdb:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = (
                    "2026-04-01T12:00:00",
                )
                mock_duckdb.connect.return_value = mock_conn

                # No pipeline log
                mock_log_path.exists.return_value = False

                # Data fresh
                recent_mtime = time.time() - 3600
                mock_games_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_odds_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.exists.return_value = True
                mock_weather_dir.is_dir.return_value = True
                mock_weather_file = MagicMock()
                mock_weather_file.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.iterdir.return_value = [mock_weather_file]

                # Models exist
                wp_path = MagicMock()
                wp_path.exists.return_value = True
                wp_path.stat.return_value = _mock_stat()
                ats_path = MagicMock()
                ats_path.exists.return_value = True
                ats_path.stat.return_value = _mock_stat()
                ou_path = MagicMock()
                ou_path.exists.return_value = True
                ou_path.stat.return_value = _mock_stat()
                mock_model_files.items.return_value = [
                    ("wp", wp_path),
                    ("ats", ats_path),
                    ("ou", ou_path),
                ]

                response = client.get("/health")

            data = response.json()
            # No log = pipeline is None, does NOT make it unhealthy
            assert data["pipeline"] is None
            assert data["status"] == "ok"

    def test_health_response_schema_backward_compatible(self) -> None:
        """Verify response JSON has existing fields plus new optional fields."""
        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            mock_db_path.exists.return_value = False
            mock_log_path.exists.return_value = False

            # Data paths raise FileNotFoundError
            mock_games_path.stat.side_effect = FileNotFoundError
            mock_odds_path.stat.side_effect = FileNotFoundError
            mock_weather_dir.exists.return_value = False

            # No models
            wp_path = MagicMock()
            wp_path.exists.return_value = False
            ats_path = MagicMock()
            ats_path.exists.return_value = False
            ou_path = MagicMock()
            ou_path.exists.return_value = False
            mock_model_files.items.return_value = [
                ("wp", wp_path),
                ("ats", ats_path),
                ("ou", ou_path),
            ]

            response = client.get("/health")

        data = response.json()

        # Original fields still present
        assert "status" in data
        assert "cache_ready" in data
        assert "last_updated" in data

        # New optional fields present
        assert "pipeline" in data
        assert "data_freshness" in data
        assert "model_status" in data

    def test_health_pipeline_includes_pid(self) -> None:
        """Pipeline log has pid field. Verify response pipeline.pid matches."""
        log_content = _make_pipeline_log(status="success", pid=99999)

        with (
            patch("api.routes.health.DB_PATH") as mock_db_path,
            patch("api.routes.health.PIPELINE_LOG_PATH") as mock_log_path,
            patch("api.routes.health.SILVER_GAMES_PATH") as mock_games_path,
            patch("api.routes.health.SILVER_ODDS_PATH") as mock_odds_path,
            patch("api.routes.health.SILVER_WEATHER_DIR") as mock_weather_dir,
            patch("api.routes.health.MODEL_FILES") as mock_model_files,
        ):
            mock_db_path.exists.return_value = True
            with patch("api.routes.health.duckdb", create=True) as mock_duckdb:
                mock_conn = MagicMock()
                mock_conn.__enter__ = MagicMock(return_value=mock_conn)
                mock_conn.__exit__ = MagicMock(return_value=False)
                mock_conn.execute.return_value.fetchone.return_value = (
                    "2026-04-01T12:00:00",
                )
                mock_duckdb.connect.return_value = mock_conn

                mock_log_path.exists.return_value = True
                mock_log_path.read_text.return_value = log_content

                recent_mtime = time.time() - 3600
                mock_games_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_odds_path.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.exists.return_value = True
                mock_weather_dir.is_dir.return_value = True
                mock_weather_file = MagicMock()
                mock_weather_file.stat.return_value = _mock_stat(st_mtime=recent_mtime)
                mock_weather_dir.iterdir.return_value = [mock_weather_file]

                wp_path = MagicMock()
                wp_path.exists.return_value = True
                wp_path.stat.return_value = _mock_stat()
                ats_path = MagicMock()
                ats_path.exists.return_value = True
                ats_path.stat.return_value = _mock_stat()
                ou_path = MagicMock()
                ou_path.exists.return_value = True
                ou_path.stat.return_value = _mock_stat()
                mock_model_files.items.return_value = [
                    ("wp", wp_path),
                    ("ats", ats_path),
                    ("ou", ou_path),
                ]

                response = client.get("/health")

            data = response.json()
            assert data["pipeline"]["pid"] == 99999


class TestDailyRunStaleness:
    """33.2 review C2 WR-11: the pipeline log is judged against a DAILY run, not a weekly one.

    The run fires every day: a predicting day writes the pipeline log and a no-prediction day
    (no games tomorrow, the offseason) appends a daily-run record. The old 7-day window let a
    task that had stopped firing read "ok" for a week.
    """

    @staticmethod
    def _read(tmp_path: Path, log_age_hours: float, record_age_hours: float | None):
        from datetime import datetime, timedelta

        from api.routes import health

        now = datetime.now(tz=UTC)
        log_path = tmp_path / "friday_pipeline.json"
        log_path.write_text(
            _make_pipeline_log(
                start_time=(now - timedelta(hours=log_age_hours)).isoformat()
            ),
            encoding="utf-8",
        )
        records_path = tmp_path / "daily_lock_runs.jsonl"
        if record_age_hours is not None:
            recorded = (now - timedelta(hours=record_age_hours)).isoformat()
            records_path.write_text(
                json.dumps({"outcome": "no_games", "recorded_at": recorded}) + "\n",
                encoding="utf-8",
            )
        with (
            patch.object(health, "PIPELINE_LOG_PATH", log_path),
            patch.object(health, "DAILY_RUN_RECORDS_PATH", records_path),
        ):
            return health._read_pipeline_log()

    def test_three_days_with_no_run_recorded_is_stale(self, tmp_path: Path) -> None:
        pipeline, corrupt, stale = self._read(tmp_path, 72, None)
        assert pipeline is not None and not corrupt
        assert stale is True, "a daily task silent for three days read as fresh"

    def test_a_recent_no_prediction_day_keeps_the_run_fresh(
        self, tmp_path: Path
    ) -> None:
        """An offseason or no-games day records itself, so the last PREDICTING run may be old."""
        _pipeline, _corrupt, stale = self._read(tmp_path, 72, 2)
        assert stale is False

    def test_a_run_within_a_day_is_fresh(self, tmp_path: Path) -> None:
        _pipeline, _corrupt, stale = self._read(tmp_path, 20, None)
        assert stale is False
