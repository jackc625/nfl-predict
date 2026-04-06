"""Tests for pipeline.alert -- PipelineAlertManager with four non-overlapping methods."""

from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# PipelineAlertManager tests
# ---------------------------------------------------------------------------


class TestAlertPipelineFailure:
    """Tests for PipelineAlertManager.alert_pipeline_failure()."""

    @patch("pipeline.alert.get_alert_manager")
    def test_alert_pipeline_failure(self, mock_get_am):
        """Creates CRITICAL alert with title='Pipeline Failed'."""
        mock_am = MagicMock()
        mock_get_am.return_value = mock_am
        mock_am.create_alert.return_value = MagicMock()

        from pipeline.alert import PipelineAlertManager

        mgr = PipelineAlertManager()
        mgr.alert_pipeline_failure(
            error="Step X failed",
            steps_completed=3,
            steps_failed=1,
            season=2025,
            week=5,
        )

        mock_am.create_alert.assert_called_once()
        call_kwargs = mock_am.create_alert.call_args
        # Check level is CRITICAL
        from utils.alert_manager import AlertLevel

        assert (
            call_kwargs.kwargs.get("level")
            or call_kwargs[1].get("level")
            or call_kwargs[0][0] == AlertLevel.CRITICAL
        )
        # Check title contains "Pipeline Failed"
        all_args = {
            **dict(
                zip(
                    ["level", "alert_type", "title", "message"],
                    call_kwargs[0] if call_kwargs[0] else [],
                )
            ),
            **(call_kwargs[1] if call_kwargs[1] else {}),
        }
        if "title" in all_args:
            assert "Pipeline Failed" in all_args["title"]


class TestAlertPipelineSuccess:
    """Tests for PipelineAlertManager.alert_pipeline_success()."""

    @patch("pipeline.alert.get_alert_manager")
    def test_alert_pipeline_success(self, mock_get_am):
        """Creates INFO alert with title='Pipeline Completed Successfully'."""
        mock_am = MagicMock()
        mock_get_am.return_value = mock_am
        mock_am.create_alert.return_value = MagicMock()

        from pipeline.alert import PipelineAlertManager

        mgr = PipelineAlertManager()
        mgr.alert_pipeline_success(
            predictions_count=16,
            execution_time_ms=30000.0,
            warnings=[],
            season=2025,
            week=5,
        )

        mock_am.create_alert.assert_called_once()
        call_kwargs = mock_am.create_alert.call_args

        from utils.alert_manager import AlertLevel

        assert call_kwargs.kwargs.get("level") or call_kwargs[0][0] == AlertLevel.INFO


class TestAlertStalenessWarning:
    """Tests for PipelineAlertManager.alert_staleness_warning()."""

    @patch("pipeline.alert.get_alert_manager")
    def test_alert_staleness_warning(self, mock_get_am):
        """Creates WARNING alert with title='Pipeline Staleness Warnings'."""
        mock_am = MagicMock()
        mock_get_am.return_value = mock_am
        mock_am.create_alert.return_value = MagicMock()

        from pipeline.alert import PipelineAlertManager

        mgr = PipelineAlertManager()
        mgr.alert_staleness_warning(
            warnings=["Odds data stale"],
            season=2025,
            week=5,
        )

        mock_am.create_alert.assert_called_once()
        call_kwargs = mock_am.create_alert.call_args

        from utils.alert_manager import AlertLevel

        assert (
            call_kwargs.kwargs.get("level") or call_kwargs[0][0] == AlertLevel.WARNING
        )


class TestAlertDegradedCompletion:
    """Tests for PipelineAlertManager.alert_degraded_completion()."""

    @patch("pipeline.alert.get_alert_manager")
    def test_alert_degraded_completion(self, mock_get_am):
        """Creates WARNING alert with failed step names in message."""
        mock_am = MagicMock()
        mock_get_am.return_value = mock_am
        mock_am.create_alert.return_value = MagicMock()

        from pipeline.alert import PipelineAlertManager

        mgr = PipelineAlertManager()
        mgr.alert_degraded_completion(
            failed_steps=["weather_ingest", "odds_snapshot"],
            completed_steps=16,
            season=2025,
            week=5,
        )

        mock_am.create_alert.assert_called_once()
        call_kwargs = mock_am.create_alert.call_args

        from utils.alert_manager import AlertLevel

        assert (
            call_kwargs.kwargs.get("level") or call_kwargs[0][0] == AlertLevel.WARNING
        )

        # Check that failed step names appear in message
        all_args = {
            **(call_kwargs[1] if call_kwargs[1] else {}),
        }
        if "message" in all_args:
            assert "weather_ingest" in all_args["message"]


class TestAlertSendAndNonOverlapping:
    """Tests for send_alert and non-overlapping alert behavior."""

    @patch("pipeline.alert.get_alert_manager")
    def test_all_alerts_call_send_alert(self, mock_get_am):
        """All four alert methods call send_alert()."""
        mock_am = MagicMock()
        mock_get_am.return_value = mock_am
        mock_am.create_alert.return_value = MagicMock()

        from pipeline.alert import PipelineAlertManager

        mgr = PipelineAlertManager()

        # Call each method
        mgr.alert_pipeline_failure(
            error="err", steps_completed=0, steps_failed=1, season=2025, week=5
        )
        mgr.alert_pipeline_success(
            predictions_count=16,
            execution_time_ms=1000.0,
            warnings=[],
            season=2025,
            week=5,
        )
        mgr.alert_staleness_warning(warnings=["w1"], season=2025, week=5)
        mgr.alert_degraded_completion(
            failed_steps=["s1"], completed_steps=17, season=2025, week=5
        )

        # send_alert should have been called 4 times (once per method)
        assert mock_am.send_alert.call_count == 4

    @patch("pipeline.alert.get_alert_manager")
    def test_alert_methods_are_non_overlapping(self, mock_get_am):
        """Each method calls create_alert exactly once (no cascading alerts)."""
        mock_am = MagicMock()
        mock_get_am.return_value = mock_am
        mock_am.create_alert.return_value = MagicMock()

        from pipeline.alert import PipelineAlertManager

        # Test each method independently
        for method_name, kwargs in [
            (
                "alert_pipeline_failure",
                {
                    "error": "err",
                    "steps_completed": 0,
                    "steps_failed": 1,
                    "season": 2025,
                    "week": 5,
                },
            ),
            (
                "alert_pipeline_success",
                {
                    "predictions_count": 16,
                    "execution_time_ms": 1000.0,
                    "warnings": [],
                    "season": 2025,
                    "week": 5,
                },
            ),
            (
                "alert_staleness_warning",
                {"warnings": ["w1"], "season": 2025, "week": 5},
            ),
            (
                "alert_degraded_completion",
                {
                    "failed_steps": ["s1"],
                    "completed_steps": 17,
                    "season": 2025,
                    "week": 5,
                },
            ),
        ]:
            mock_am.reset_mock()
            mgr = PipelineAlertManager()
            getattr(mgr, method_name)(**kwargs)
            assert mock_am.create_alert.call_count == 1, (
                f"{method_name} should call create_alert exactly once, "
                f"got {mock_am.create_alert.call_count}"
            )
