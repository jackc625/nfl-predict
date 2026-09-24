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


# ---------------------------------------------------------------------------
# The installed-task read-back
# ---------------------------------------------------------------------------

_COMMITTED = (
    (Path(scheduling.__file__).parent / "windows_scheduler.xml")
    .read_bytes()
    .decode("utf-16")
)

# The shape Windows exports (measured 2026-09-24 with schtasks /query /xml on the WEEKLY task this
# plan replaces): default-valued settings omitted, a SID for the user, ANSI text.
_WEEKLY_EXPORT = r"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Settings>
    <StartWhenAvailable>true</StartWhenAvailable>
  </Settings>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2026-09-12T18:00:00</StartBoundary>
      <ScheduleByWeek>
        <WeeksInterval>1</WeeksInterval>
        <DaysOfWeek>
          <Friday />
        </DaysOfWeek>
      </ScheduleByWeek>
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>C:\Users\jackc\AppData\Roaming\Python\Python313\Scripts\uv.exe</Command>
      <Arguments>run python scripts/friday_pipeline.py --log-level INFO</Arguments>
    </Exec>
  </Actions>
</Task>"""


def _daily_export() -> str:
    """The committed definition as Windows would export it: StartWhenAvailable (false) omitted."""
    return _COMMITTED.replace("<StartWhenAvailable>false</StartWhenAvailable>", "")


def test_the_committed_definition_reads_back_as_itself() -> None:
    rows = scheduling.compare_task_fields(_COMMITTED, _daily_export())
    assert [name for name, *_ in rows] == list(scheduling.READBACK_FIELDS)
    assert all(match for *_, match in rows), rows
    fields = scheduling.task_fields(_daily_export())
    assert fields["StartWhenAvailable"] == "false"
    assert fields["Trigger"] == "CalendarTrigger ScheduleByDay(DaysInterval=1)"


def test_the_weekly_task_does_not_read_back_as_the_daily_one() -> None:
    mismatched = {
        name
        for name, _want, _got, match in scheduling.compare_task_fields(
            _COMMITTED, _WEEKLY_EXPORT
        )
        if not match
    }
    assert mismatched == {"Trigger", "StartTime", "Arguments", "StartWhenAvailable"}


def _fake_schtasks(export: str, listing: str):
    def _run(cmd: list[str], **_kwargs: object) -> MagicMock:
        if "/xml" in cmd:
            return MagicMock(returncode=0, stdout=export.encode("ascii"))
        return MagicMock(returncode=0, stdout=listing)

    return _run


@pytest.mark.parametrize(
    ("listing", "expected"),
    [
        ('"\\NFL_Predict_Pipeline","9/24/2026 5:00:00 PM","Ready"\n', True),
        (
            '"\\NFL_Predict_Pipeline","9/24/2026 5:00:00 PM","Ready"\n'
            '"\\NFL_Predict_Predictions","N/A","Ready"\n',
            False,
        ),
    ],
)
def test_the_read_back_needs_every_field_and_exactly_one_nfl_task(
    capsys: pytest.CaptureFixture[str], listing: str, expected: bool
) -> None:
    setup = _bare_setup(Path(scheduling.__file__).resolve().parents[1])
    with patch.object(
        scheduling.subprocess,
        "run",
        side_effect=_fake_schtasks(_daily_export(), listing),
    ):
        assert setup.verify_installed() is expected
    out = capsys.readouterr().out
    assert f"READBACK_MATCH= {expected}" in out
    assert "StartWhenAvailable= false | false | MATCH" in out
