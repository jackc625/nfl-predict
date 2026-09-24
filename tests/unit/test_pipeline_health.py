"""Tests for pipeline.health -- PipelineHealthChecker with preflight and postrun checks."""

from unittest.mock import MagicMock, patch

import pandas as pd

# ---------------------------------------------------------------------------
# PipelineHealthChecker tests
# ---------------------------------------------------------------------------


class TestRunPreflight:
    """Tests for PipelineHealthChecker.run_preflight()."""

    @patch("pipeline.health.get_db_connection")
    @patch("pipeline.health.get_settings")
    def test_run_preflight_healthy(self, mock_settings, mock_db):
        """All quick checks pass -- returns status='healthy'."""
        mock_settings.return_value = MagicMock()
        mock_db.return_value = MagicMock()
        # Mock DB query
        mock_db.return_value.execute.return_value.fetchone.return_value = (1,)
        mock_db.return_value.execute.return_value.fetchall.return_value = [
            ("games",),
            ("predictions",),
        ]

        from pipeline.health import PipelineHealthChecker

        checker = PipelineHealthChecker()
        result = checker.run_preflight()

        assert result["status"] == "healthy"
        assert "checks" in result
        assert "duration_ms" in result

    @patch("pipeline.health.get_db_connection")
    @patch("pipeline.health.get_settings")
    def test_run_preflight_unhealthy_db_down(self, mock_settings, mock_db):
        """DB check fails -- returns status='unhealthy'."""
        mock_settings.return_value = MagicMock()
        mock_db.side_effect = Exception("Connection refused")

        from pipeline.health import PipelineHealthChecker

        checker = PipelineHealthChecker()
        result = checker.run_preflight()

        assert result["status"] == "unhealthy"

    @patch("pipeline.health.get_db_connection")
    @patch("pipeline.health.get_settings")
    def test_run_preflight_only_runs_3_checks(self, mock_settings, mock_db):
        """Preflight runs exactly 3 checks: DB, models, disk."""
        mock_settings.return_value = MagicMock()
        mock_db.return_value = MagicMock()
        mock_db.return_value.execute.return_value.fetchone.return_value = (1,)
        mock_db.return_value.execute.return_value.fetchall.return_value = []

        from pipeline.health import PipelineHealthChecker

        checker = PipelineHealthChecker()
        result = checker.run_preflight()

        check_names = [c["name"] for c in result["checks"]]
        assert len(check_names) == 3
        assert "database_connectivity" in check_names
        assert "model_artifacts" in check_names
        assert "disk_space" in check_names
        # These should NOT be in preflight
        assert "data_freshness" not in check_names
        assert "api_endpoints" not in check_names
        assert "prediction_pipeline" not in check_names

    @patch("pipeline.health.get_db_connection")
    @patch("pipeline.health.get_settings")
    def test_preflight_returns_duration_ms(self, mock_settings, mock_db):
        """Verify duration_ms field present and is a positive float."""
        mock_settings.return_value = MagicMock()
        mock_db.return_value = MagicMock()
        mock_db.return_value.execute.return_value.fetchone.return_value = (1,)
        mock_db.return_value.execute.return_value.fetchall.return_value = []

        from pipeline.health import PipelineHealthChecker

        checker = PipelineHealthChecker()
        result = checker.run_preflight()

        assert "duration_ms" in result
        assert isinstance(result["duration_ms"], float)
        assert result["duration_ms"] >= 0


class TestRunPostrun:
    """Tests for PipelineHealthChecker.run_postrun()."""

    @patch("pipeline.health.get_db_connection")
    @patch("pipeline.health.get_settings")
    def test_run_postrun_comprehensive(self, mock_settings, mock_db):
        """Post-run runs all 6 checks."""
        mock_settings.return_value = MagicMock()
        mock_db.return_value = MagicMock()
        mock_db.return_value.execute.return_value.fetchone.return_value = (1,)
        mock_db.return_value.execute.return_value.fetchall.return_value = []

        from pipeline.health import PipelineHealthChecker

        checker = PipelineHealthChecker()
        result = checker.run_postrun()

        assert "checks" in result
        assert len(result["checks"]) == 6
        check_names = [c["name"] for c in result["checks"]]
        assert "database_connectivity" in check_names
        assert "data_freshness" in check_names
        assert "model_artifacts" in check_names
        assert "api_endpoints" in check_names
        assert "prediction_pipeline" in check_names
        assert "disk_space" in check_names


def _checker_rooted_at(root):
    """A health checker whose data root is *root* (the storage settings' ``root_path``)."""
    from pipeline.health import PipelineHealthChecker

    checker = PipelineHealthChecker()
    checker.settings = MagicMock()
    checker.settings.config.data.root_path = str(root)
    return checker


def _write_silver(root, table, created_at):
    silver = root / "silver"
    silver.mkdir(parents=True, exist_ok=True)
    stamps = pd.to_datetime(created_at, utc=True)
    pd.DataFrame(
        {"game_id": [f"G{i}" for i in range(len(stamps))], "created_at": stamps}
    ).to_parquet(silver / f"{table}.parquet")


class TestCheckDataFreshness:
    """The freshness check reads the silver PARQUET tables the daily run writes.

    Step 27b: it queried DuckDB tables ``odds_snapshot`` and ``weather``, which exist only as
    parquet, and subtracted an aware ``created_at`` from a naive clock, so it reported
    ``unhealthy`` after every run, good or bad.
    """

    TABLES = ("games", "odds_snapshot", "weather")

    def test_a_run_that_just_wrote_every_table_is_fresh(self, tmp_path):
        now = pd.Timestamp.now(tz="UTC")
        for table in self.TABLES:
            _write_silver(tmp_path, table, [now - pd.Timedelta(hours=2), now])

        result = _checker_rooted_at(tmp_path).check_data_freshness()

        assert result["status"] == "healthy", result
        assert set(result["details"]["tables"]) == set(self.TABLES)

    def test_a_table_the_run_did_not_refresh_is_stale(self, tmp_path):
        now = pd.Timestamp.now(tz="UTC")
        for table in self.TABLES:
            _write_silver(tmp_path, table, [now])
        _write_silver(tmp_path, "weather", [now - pd.Timedelta(days=2)])

        result = _checker_rooted_at(tmp_path).check_data_freshness()

        assert result["status"] == "unhealthy"
        assert result["details"]["tables"]["weather"]["is_fresh"] is False
        assert result["details"]["tables"]["games"]["is_fresh"] is True

    def test_a_missing_table_is_unhealthy_by_name(self, tmp_path):
        now = pd.Timestamp.now(tz="UTC")
        for table in ("games", "weather"):
            _write_silver(tmp_path, table, [now])

        result = _checker_rooted_at(tmp_path).check_data_freshness()

        assert result["status"] == "unhealthy"
        assert "error" in result["details"]["tables"]["odds_snapshot"]


class TestCheckApiEndpoints:
    """The endpoint check asks for pages the app actually serves.

    Step 27b: ``/current-week`` and ``/games`` were removed with the JSON API, so they
    answered 404 on every run.
    """

    def test_every_checked_path_is_a_route_of_the_app(self):
        from api.main import app
        from pipeline.health import API_ENDPOINTS_CHECKED

        served = {getattr(route, "path", None) for route in app.routes}
        for _method, path in API_ENDPOINTS_CHECKED:
            assert path in served, f"{path} is not served by api.main.app"


class TestCheckPredictionPipeline:
    """Tests for check_prediction_pipeline -- AUTO-02-F1 glob/format fix.

    The post-run prediction-pipeline health check must discover the REAL
    orchestrator output (``predictions_<season>_week<week>.csv``, written by
    ``step_generate_predictions`` into ``_predictions_output_dir()``) and read it
    as CSV. The pre-fix code globbed ``current_predictions*.parquet`` -- a pattern
    the real run never produces -- so the check always reported "No prediction
    files found" even on a successful Friday run.
    """

    def test_check_prediction_pipeline_matches_real_csv(self, tmp_path, monkeypatch):
        """A real predictions_<S>_week<W>.csv present -> reports healthy."""
        from pipeline import steps

        # Redirect the shared prediction-output dir to the tmp dir (the same
        # monkeypatch idiom the prediction-step integration tests use).
        monkeypatch.setattr(steps, "_predictions_output_dir", lambda: tmp_path)

        pred_csv = tmp_path / "predictions_2024_week1.csv"
        pd.DataFrame(
            {
                "game_id": ["G1", "G2"],
                "wp_prob": [0.62, 0.48],
            }
        ).to_csv(pred_csv, index=False)

        from pipeline.health import PipelineHealthChecker

        checker = PipelineHealthChecker()
        result = checker.check_prediction_pipeline()

        assert result["name"] == "prediction_pipeline"
        assert result["status"] == "healthy", result
        assert result["details"]["prediction_count"] == 2
        assert result["details"]["latest_file"] == "predictions_2024_week1.csv"

    def test_check_prediction_pipeline_missing_when_empty(self, tmp_path, monkeypatch):
        """Empty predictions dir -> reports unhealthy / no files found."""
        from pipeline import steps

        monkeypatch.setattr(steps, "_predictions_output_dir", lambda: tmp_path)

        from pipeline.health import PipelineHealthChecker

        checker = PipelineHealthChecker()
        result = checker.check_prediction_pipeline()

        assert result["status"] == "unhealthy"
        assert result["details"]["prediction_count"] == 0
