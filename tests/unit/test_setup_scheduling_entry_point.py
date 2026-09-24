"""The installer validates, rehearses and guards the entry point it actually schedules (Plan 33.2-28).

Measured 2026-09-20: ``validate_prerequisites`` looked for ``scripts/friday_pipeline.py`` and
``test_scripts`` ran it with ``--dry-run --force``, so an install could report success while never
exercising ``scripts/daily_lock_pipeline.py`` -- which declares no ``--force`` at all. And the
trigger's ``StartBoundary`` carries no zone, so it fires in machine-local time: an install on a
machine not on Eastern time would fire at the wrong hour. The installer now refuses that.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import sys
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import deployment.setup_scheduling as scheduling
from deployment.setup_scheduling import SchedulingSetup

_MODULE_PATH = Path(scheduling.__file__)


def _bare_setup(home: Path) -> SchedulingSetup:
    setup = SchedulingSetup.__new__(SchedulingSetup)
    setup.nfl_predict_home = home
    setup.platform = "windows"
    setup.python_path = sys.executable
    return setup


def _string_constants() -> set[str]:
    tree = ast.parse(_MODULE_PATH.read_text(encoding="utf-8"))
    return {
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def test_the_module_names_the_daily_entry_point_and_never_the_friday_one() -> None:
    constants = _string_constants()
    assert scheduling.DAILY_ENTRY_POINT == "scripts/daily_lock_pipeline.py"
    assert "scripts/daily_lock_pipeline.py" in constants
    assert not [c for c in constants if "friday_pipeline" in c]


def test_validate_prerequisites_requires_the_daily_script(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "friday_pipeline.py").write_text("# the retired entry")
    assert _bare_setup(tmp_path).validate_prerequisites() is False

    (tmp_path / "scripts" / "daily_lock_pipeline.py").write_text("# the daily entry")
    assert _bare_setup(tmp_path).validate_prerequisites() is True


def test_the_rehearsal_runs_the_daily_script_with_only_the_flags_it_declares(
    tmp_path: Path,
) -> None:
    setup = _bare_setup(tmp_path)
    with patch.object(scheduling.subprocess, "run") as run:
        run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        assert setup.test_scripts(rehearsal_date=date(2026, 9, 26)) is True

    (call,) = run.call_args_list
    cmd = call.args[0]
    assert cmd[1:] == [
        "scripts/daily_lock_pipeline.py",
        "--dry-run",
        "--date",
        "2026-09-26",
    ]
    assert "--force" not in cmd
    # A rehearsal never spends the paid Odds API quota: an empty key puts the client in mock mode.
    assert call.kwargs["env"]["ODDS_API_KEY"] == ""


def test_the_rehearsal_without_a_date_runs_today(tmp_path: Path) -> None:
    with patch.object(scheduling.subprocess, "run") as run:
        run.return_value = MagicMock(returncode=0, stderr="", stdout="")
        _bare_setup(tmp_path).test_scripts()
    assert run.call_args.args[0][1:] == ["scripts/daily_lock_pipeline.py", "--dry-run"]


# ---------------------------------------------------------------------------
# The time-zone guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "zone", ["Pacific Standard Time", "UTC", "Central Standard Time", ""]
)
def test_a_non_eastern_zone_is_refused(zone: str) -> None:
    with pytest.raises(scheduling.NonEasternTimeZoneError):
        scheduling.require_eastern_time_zone(zone)


def test_the_eastern_zone_is_accepted() -> None:
    scheduling.require_eastern_time_zone("Eastern Standard Time")


def test_an_install_on_a_non_eastern_machine_never_reaches_schtasks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scheduling, "read_windows_time_zone", lambda: "Pacific Standard Time"
    )
    installed: list[bool] = []
    monkeypatch.setattr(
        SchedulingSetup,
        "setup_windows_scheduler",
        lambda _self, dry_run=False: installed.append(dry_run) or True,
    )
    monkeypatch.setattr(
        "sys.argv", ["setup_scheduling.py", "--platform", "windows", "--install"]
    )

    assert scheduling.main() == 1
    assert installed == []


def test_an_install_on_an_eastern_machine_proceeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        scheduling, "read_windows_time_zone", lambda: "Eastern Standard Time"
    )
    installed: list[bool] = []
    monkeypatch.setattr(
        SchedulingSetup,
        "setup_windows_scheduler",
        lambda _self, dry_run=False: installed.append(dry_run) or True,
    )
    monkeypatch.setattr(
        "sys.argv", ["setup_scheduling.py", "--platform", "windows", "--install"]
    )

    assert scheduling.main() == 0
    assert installed == [False]
