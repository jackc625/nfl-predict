"""Tests for deployment/setup_scheduling.py scheduling setup.

Verifies the unified pipeline scheduling: single task creation,
old task cleanup, idempotent behavior, and command structure.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deployment.setup_scheduling import SchedulingSetup


@pytest.fixture()
def setup_with_mock_home(tmp_path: Path) -> SchedulingSetup:
    """Create a SchedulingSetup with a mocked project home."""
    # Create the expected directory structure
    scripts_dir = tmp_path / "scripts"
    scripts_dir.mkdir()
    (scripts_dir / "friday_pipeline.py").write_text("# unified pipeline")

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir()

    with patch.object(Path, "exists", return_value=True):
        setup = SchedulingSetup(str(tmp_path))

    # Reset the home to the actual tmp_path after init
    setup.nfl_predict_home = tmp_path
    return setup


class TestValidatePrerequisites:
    """Tests for validate_prerequisites()."""

    def test_setup_scheduling_validates_friday_pipeline(self, tmp_path: Path) -> None:
        """Verify validate_prerequisites() checks for friday_pipeline.py."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        (scripts_dir / "friday_pipeline.py").write_text("# pipeline")
        (tmp_path / "logs").mkdir()

        setup = SchedulingSetup.__new__(SchedulingSetup)
        setup.nfl_predict_home = tmp_path
        setup.platform = "windows"
        setup.python_path = "python"

        with patch.object(Path, "exists", return_value=True):
            result = setup.validate_prerequisites()

        assert result is True

    def test_setup_scheduling_validates_rejects_missing_script(
        self, tmp_path: Path
    ) -> None:
        """Verify validate_prerequisites() returns False when friday_pipeline.py missing."""
        scripts_dir = tmp_path / "scripts"
        scripts_dir.mkdir()
        # Do NOT create friday_pipeline.py
        (tmp_path / "logs").mkdir()

        setup = SchedulingSetup.__new__(SchedulingSetup)
        setup.nfl_predict_home = tmp_path
        setup.platform = "windows"
        setup.python_path = "python"

        result = setup.validate_prerequisites()

        assert result is False


class TestWindowsScheduler:
    """Tests for setup_windows_scheduler()."""

    def test_setup_scheduling_creates_single_task(
        self, setup_with_mock_home: SchedulingSetup
    ) -> None:
        """Verify only ONE schtasks /create call with NFL_Predict_Pipeline."""
        setup = setup_with_mock_home

        with patch("deployment.setup_scheduling.subprocess.run") as mock_run:
            # schtasks /? check passes
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            setup.setup_windows_scheduler(dry_run=False)

        # Filter for /create calls only
        create_calls = [
            c
            for c in mock_run.call_args_list
            if isinstance(c[0][0], list) and "/create" in c[0][0]
        ]

        assert len(create_calls) == 1
        cmd = create_calls[0][0][0]
        assert "NFL_Predict_Pipeline" in cmd
        assert "17:00" in cmd

    def test_setup_scheduling_cleans_old_tasks(
        self, setup_with_mock_home: SchedulingSetup
    ) -> None:
        """Verify schtasks /delete called for old task names before creating new task."""
        setup = setup_with_mock_home

        with patch("deployment.setup_scheduling.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            setup.setup_windows_scheduler(dry_run=False)

        # Filter for /delete calls
        delete_calls = [
            c
            for c in mock_run.call_args_list
            if isinstance(c[0][0], list) and "/delete" in c[0][0]
        ]

        assert len(delete_calls) == 2
        deleted_names = []
        for c in delete_calls:
            cmd = c[0][0]
            tn_idx = cmd.index("/tn")
            deleted_names.append(cmd[tn_idx + 1])

        assert "NFL_Predict_DataUpdate" in deleted_names
        assert "NFL_Predict_Predictions" in deleted_names

    def test_setup_scheduling_cleanup_idempotent(
        self, setup_with_mock_home: SchedulingSetup
    ) -> None:
        """Verify no error raised when old task deletion fails (task doesn't exist)."""
        setup = setup_with_mock_home

        def mock_run_side_effect(cmd, **kwargs):
            result = MagicMock()
            if isinstance(cmd, list):
                if "/delete" in cmd:
                    # Simulate task not found
                    result.returncode = 1
                    result.stderr = "ERROR: The system cannot find the file specified."
                    return result
                if "/?":
                    result.returncode = 0
                    return result
            result.returncode = 0
            result.stderr = ""
            result.stdout = ""
            return result

        with patch(
            "deployment.setup_scheduling.subprocess.run",
            side_effect=mock_run_side_effect,
        ):
            # Should not raise
            setup.setup_windows_scheduler(dry_run=False)

    def test_setup_scheduling_command_includes_working_dir(
        self, setup_with_mock_home: SchedulingSetup
    ) -> None:
        """Verify the generated schtasks command runs in the correct working directory."""
        setup = setup_with_mock_home

        with patch("deployment.setup_scheduling.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            setup.setup_windows_scheduler(dry_run=False)

        # Find the /create call and verify cwd
        create_calls = [
            c
            for c in mock_run.call_args_list
            if isinstance(c[0][0], list) and "/create" in c[0][0]
        ]

        assert len(create_calls) == 1
        # The cwd keyword should be set to project home
        assert create_calls[0][1].get("cwd") == str(setup.nfl_predict_home)

    def test_setup_scheduling_dry_run(
        self, setup_with_mock_home: SchedulingSetup
    ) -> None:
        """With dry_run=True, verify no subprocess.run calls for task creation."""
        setup = setup_with_mock_home

        with patch("deployment.setup_scheduling.subprocess.run") as mock_run:
            # Only schtasks /? check should be called
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            setup.setup_windows_scheduler(dry_run=True)

        # No /create or /delete calls when dry_run
        create_calls = [
            c
            for c in mock_run.call_args_list
            if isinstance(c[0][0], list) and "/create" in c[0][0]
        ]
        delete_calls = [
            c
            for c in mock_run.call_args_list
            if isinstance(c[0][0], list) and "/delete" in c[0][0]
        ]

        assert len(create_calls) == 0
        assert len(delete_calls) == 0

    def test_setup_scheduling_uses_uv_run(
        self, setup_with_mock_home: SchedulingSetup
    ) -> None:
        """Verify the schtasks /tr argument uses 'uv run' for the interpreter."""
        setup = setup_with_mock_home

        with patch("deployment.setup_scheduling.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stderr="", stdout="")

            setup.setup_windows_scheduler(dry_run=False)

        # Find the /create call
        create_calls = [
            c
            for c in mock_run.call_args_list
            if isinstance(c[0][0], list) and "/create" in c[0][0]
        ]

        assert len(create_calls) == 1
        cmd = create_calls[0][0][0]
        # Find the /tr argument
        tr_idx = cmd.index("/tr")
        tr_value = cmd[tr_idx + 1]
        assert "uv run" in tr_value
        assert "friday_pipeline.py" in tr_value
