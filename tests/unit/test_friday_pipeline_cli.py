"""CLI-layer tests for scripts/friday_pipeline.py.

Focus: the D-06 offseason no-op short-circuit. A live (scheduled, unforced)
offseason run must exit 0 as a clean no-op WITHOUT constructing/running the
orchestrator -- so no CRITICAL "Pipeline Failed" alert is ever fired. With
--force the short-circuit is bypassed and the pipeline still runs.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

from utils.date_utils import ET


class TestOffseasonNoOp:
    """D-06: offseason short-circuit at the CLI layer."""

    def test_offseason_unforced_run_is_clean_noop_no_critical_alert(self):
        """An unforced offseason run exits 0 without constructing FridayPipeline.

        Because the orchestrator (and therefore its CRITICAL ``alert_pipeline_failure``
        path) is never reached, the offseason scheduled run cannot cry wolf.
        """
        from scripts.friday_pipeline import main

        # May 29 is squarely in the NFL offseason (season starts early September,
        # season_end ~= start + 22 weeks lands in early February).
        offseason_now = datetime(2026, 5, 29, 12, 0, tzinfo=ET)

        with (
            patch("sys.argv", ["friday_pipeline.py"]),
            patch(
                "scripts.friday_pipeline.get_current_nfl_week",
                return_value=(2025, 22),
            ),
            patch("scripts.friday_pipeline.datetime") as mock_dt,
            patch("pipeline.orchestrator.FridayPipeline") as mock_pipeline_cls,
        ):
            # datetime.now(ET) -> offseason; delegate direct construction to the
            # real datetime so any timedelta arithmetic still works.
            mock_dt.now.return_value = offseason_now
            mock_dt.side_effect = datetime

            result = main()

        assert result == 0
        # The orchestrator (and its alert paths) is never constructed.
        mock_pipeline_cls.assert_not_called()

    def test_offseason_forced_run_bypasses_shortcircuit(self):
        """With --force the offseason short-circuit is bypassed; pipeline runs."""
        from scripts.friday_pipeline import main

        offseason_now = datetime(2026, 5, 29, 12, 0, tzinfo=ET)

        # A passing fake run log so the forced path returns 0 cleanly.
        fake_log = MagicMock()
        fake_log.status = "success"
        fake_log.total_duration_ms = 1.0

        with (
            patch("sys.argv", ["friday_pipeline.py", "--force"]),
            patch(
                "scripts.friday_pipeline.get_current_nfl_week",
                return_value=(2025, 22),
            ),
            patch("scripts.friday_pipeline.datetime") as mock_dt,
            patch("pipeline.orchestrator.FridayPipeline") as mock_pipeline_cls,
        ):
            mock_dt.now.return_value = offseason_now
            mock_dt.side_effect = datetime
            mock_instance = mock_pipeline_cls.return_value
            mock_instance.run.return_value = fake_log

            result = main()

        assert result == 0
        # Forced run bypasses the offseason no-op: the pipeline IS constructed and run.
        mock_pipeline_cls.assert_called_once()
        mock_instance.run.assert_called_once()

    def test_offseason_dry_run_lists_steps_without_force(self):
        """Bare ``--dry-run`` in the offseason lists steps and exits 0 (WR-04).

        ``--dry-run`` is a read-only inspection tool that must work year-round. It
        bypasses the D-06 offseason no-op short-circuit (without needing ``--force``),
        so the operator can inspect the step plan out of season. The crying-wolf
        protection is unaffected because the scheduled task never passes ``--dry-run``.
        """
        from scripts.friday_pipeline import main

        offseason_now = datetime(2026, 5, 29, 12, 0, tzinfo=ET)

        with (
            patch("sys.argv", ["friday_pipeline.py", "--dry-run"]),
            patch(
                "scripts.friday_pipeline.get_current_nfl_week",
                return_value=(2025, 22),
            ),
            patch("scripts.friday_pipeline.datetime") as mock_dt,
            patch("pipeline.orchestrator.FridayPipeline") as mock_pipeline_cls,
        ):
            mock_dt.now.return_value = offseason_now
            mock_dt.side_effect = datetime
            mock_instance = mock_pipeline_cls.return_value
            mock_instance.dry_run.return_value = ["step_a", "step_b", "step_c"]

            result = main()

        assert result == 0
        # The no-op short-circuit was bypassed: the pipeline IS constructed and its
        # dry_run() step listing is produced (the real run path is never invoked).
        mock_pipeline_cls.assert_called_once()
        mock_instance.dry_run.assert_called_once()
        mock_instance.run.assert_not_called()

    def test_in_season_unforced_run_does_not_shortcircuit(self):
        """During the season the short-circuit does not fire; pipeline runs."""
        from scripts.friday_pipeline import main

        # Mid-October is in-season for the 2025 NFL season.
        in_season_now = datetime(2025, 10, 17, 18, 0, tzinfo=ET)

        fake_log = MagicMock()
        fake_log.status = "success"
        fake_log.total_duration_ms = 1.0

        with (
            patch("sys.argv", ["friday_pipeline.py"]),
            patch(
                "scripts.friday_pipeline.get_current_nfl_week",
                return_value=(2025, 7),
            ),
            patch("scripts.friday_pipeline.datetime") as mock_dt,
            patch("pipeline.orchestrator.FridayPipeline") as mock_pipeline_cls,
        ):
            mock_dt.now.return_value = in_season_now
            mock_dt.side_effect = datetime
            mock_instance = mock_pipeline_cls.return_value
            mock_instance.run.return_value = fake_log

            result = main()

        assert result == 0
        mock_pipeline_cls.assert_called_once()
        mock_instance.run.assert_called_once()
