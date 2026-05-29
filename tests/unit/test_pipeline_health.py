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
