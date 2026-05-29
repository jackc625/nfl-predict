"""Tests for deployment/setup_scheduling.py scheduling setup.

Verifies the unified pipeline scheduling: single task creation,
old task cleanup, idempotent behavior, and command structure.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from deployment.setup_scheduling import SchedulingSetup

# Repo root resolved from this test file (tests/integration/<this file>) so the
# XML-content test is independent of pytest's working directory.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCHEDULER_XML = _REPO_ROOT / "deployment" / "windows_scheduler.xml"


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
        # Installer registers the canonical XML via /xml (D-07), not an inline /st time
        assert "/xml" in cmd
        xml_idx = cmd.index("/xml")
        assert cmd[xml_idx + 1].endswith("windows_scheduler.xml")
        assert "/f" in cmd

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

    def test_setup_scheduling_installs_canonical_xml(
        self, setup_with_mock_home: SchedulingSetup
    ) -> None:
        """Verify the installer registers the canonical XML via /xml, not an inline /tr."""
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

        # Installs the committed XML (single source of truth, D-07)
        assert "/xml" in cmd
        xml_idx = cmd.index("/xml")
        xml_path = cmd[xml_idx + 1]
        assert xml_path.endswith("windows_scheduler.xml")
        assert "deployment" in xml_path
        assert "/f" in cmd

        # The inline-command shape is gone: no /tr, /sc, /st, or 17:00
        assert "/tr" not in cmd
        assert "/sc" not in cmd
        assert "/st" not in cmd
        assert "17:00" not in cmd

    def test_setup_scheduling_has_no_cron_branch(self) -> None:
        """Verify the dead Linux/mac cron branch was removed (D-07: one story)."""
        assert not hasattr(SchedulingSetup, "setup_cron_linux_mac")


class TestWindowsSchedulerXml:
    """Content assertions on the committed canonical windows_scheduler.xml."""

    def test_windows_scheduler_xml_is_reconciled(self) -> None:
        """Verify the canonical XML carries the reconciled principal, time, and command.

        The XML declares UTF-16 (line 1), so it MUST be read with encoding="utf-16";
        reading as utf-8 would corrupt the assertions.
        """
        xml = _SCHEDULER_XML.read_text(encoding="utf-16")

        # Principal: owner user, NOT the SYSTEM SID (D-10)
        assert "S-1-5-18" not in xml
        assert "<UserId>jackc</UserId>" in xml
        assert "<LogonType>S4U</LogonType>" in xml
        assert "S4U" in xml
        assert "HighestAvailable" in xml

        # Trigger: 18:00 local = 6 PM ET, no timezone offset (D-09)
        assert "2026-09-12T18:00:00" in xml
        assert "17:00" not in xml

        # Command: uv-run invocation, not the .venv python.exe (D-10)
        assert "<Command>uv</Command>" in xml
        assert "run python scripts/friday_pipeline.py" in xml
        assert "friday_pipeline.py" in xml
        assert ".venv\\Scripts\\python.exe" not in xml

        # Robustness settings preserved (D-10)
        assert "IgnoreNew" in xml
        assert "PT2H" in xml
        assert "WakeToRun" in xml
        assert "StartWhenAvailable" in xml
        assert "RunOnlyIfNetworkAvailable" in xml

    def test_windows_scheduler_xml_declares_utf16(self) -> None:
        """Verify the XML declaration still pins UTF-16 (encoding not corrupted to utf-8)."""
        xml = _SCHEDULER_XML.read_text(encoding="utf-16")
        assert 'encoding="UTF-16"' in xml


class TestWindowsSchedulerXmlWellFormed:
    """Well-formedness guards on the committed canonical windows_scheduler.xml.

    Regression guard for the malformed-comment defect (D-08): XML 1.0 forbids the
    literal "--" sequence anywhere inside a comment, which made schtasks reject the
    file with "incorrect comment syntax". The earlier content tests only string-matched
    body text and never parsed the document, so the illegal comment shipped undetected.
    """

    def test_windows_scheduler_xml_is_well_formed(self) -> None:
        """Verify the canonical XML parses without a ParseError (schtasks-importable).

        ElementTree.parse honors the in-document encoding declaration (UTF-16), so it
        reads the BOM-prefixed file directly; a malformed comment raises ParseError.
        """
        # Raises xml.etree.ElementTree.ParseError if the document is not well-formed.
        tree = ET.parse(_SCHEDULER_XML)

        ns = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
        root = tree.getroot()

        # The reconciled principal, trigger, and command survive the parse intact.
        user_id = root.find(f".//{ns}Principal/{ns}UserId")
        logon_type = root.find(f".//{ns}Principal/{ns}LogonType")
        command = root.find(f".//{ns}Actions/{ns}Exec/{ns}Command")
        start_boundary = root.find(f".//{ns}CalendarTrigger/{ns}StartBoundary")

        assert user_id is not None and user_id.text == "jackc"
        assert logon_type is not None and logon_type.text == "S4U"
        assert command is not None and command.text == "uv"
        assert start_boundary is not None
        assert start_boundary.text == "2026-09-12T18:00:00"

    def test_windows_scheduler_xml_comment_has_no_double_hyphen(self) -> None:
        """Verify the header comment contains no '--' (illegal inside an XML comment).

        A literal double-hyphen anywhere between '<!--' and '-->' is forbidden by the
        XML 1.0 spec and causes schtasks to reject the task XML as malformed.
        """
        xml = _SCHEDULER_XML.read_text(encoding="utf-16")

        match = re.search(r"<!--(.*?)-->", xml, re.DOTALL)
        assert match is not None, "expected a header comment in windows_scheduler.xml"

        comment_body = match.group(1)
        assert "--" not in comment_body, (
            "windows_scheduler.xml comment contains an illegal '--' sequence; "
            "XML 1.0 forbids double-hyphens inside comments and schtasks will reject it"
        )
