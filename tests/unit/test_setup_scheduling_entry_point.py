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
import xml.etree.ElementTree as ET
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

import deployment.setup_scheduling as scheduling
from deployment.setup_scheduling import SchedulingSetup
from forward_ledger import closing_schedule
from forward_ledger.closing_windows import windows_for_schedule

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


@pytest.mark.parametrize("action", ["--install", "--dry-run"])
def test_a_failed_rehearsal_is_never_overwritten_by_an_install(
    monkeypatch: pytest.MonkeyPatch, action: str
) -> None:
    """33.2 review C2 WR-09: ``--test --install`` with a failing rehearsal installs NOTHING.

    The rehearsal's result used to be overwritten by the install's, so the task was installed
    anyway and the run reported success with exit 0.
    """
    monkeypatch.setattr(
        scheduling, "read_windows_time_zone", lambda: "Eastern Standard Time"
    )
    monkeypatch.setattr(
        SchedulingSetup, "test_scripts", lambda _self, rehearsal_date=None: False
    )
    installed: list[bool] = []
    monkeypatch.setattr(
        SchedulingSetup,
        "setup_windows_scheduler",
        lambda _self, dry_run=False: installed.append(dry_run) or True,
    )
    monkeypatch.setattr(
        "sys.argv", ["setup_scheduling.py", "--platform", "windows", "--test", action]
    )

    assert scheduling.main() == 1
    assert installed == []


def test_a_passing_rehearsal_still_installs(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: the gate is on a FAILED rehearsal, not on rehearsing at all."""
    monkeypatch.setattr(
        scheduling, "read_windows_time_zone", lambda: "Eastern Standard Time"
    )
    monkeypatch.setattr(
        SchedulingSetup, "test_scripts", lambda _self, rehearsal_date=None: True
    )
    installed: list[bool] = []
    monkeypatch.setattr(
        SchedulingSetup,
        "setup_windows_scheduler",
        lambda _self, dry_run=False: installed.append(dry_run) or True,
    )
    monkeypatch.setattr(
        "sys.argv",
        ["setup_scheduling.py", "--platform", "windows", "--test", "--install"],
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
    # The weekly export names no principal and no working directory, so those differ too.
    assert mismatched == {
        "Trigger",
        "StartTime",
        "Arguments",
        "StartWhenAvailable",
        "RunAs",
        "WorkingDirectory",
    }


# 33.2 review C2 WR-07: the run-as user, the enabled flags and the working directory are compared.


def _mismatches(installed: str) -> set[str]:
    return {
        name
        for name, _want, _got, match in scheduling.compare_task_fields(
            _COMMITTED, installed
        )
        if not match
    }


def _nth_enabled_false(text: str, n: int) -> str:
    """*text* with its *n*-th ``<Enabled>true</Enabled>`` (0 = the trigger's) set false."""
    parts = text.split("<Enabled>true</Enabled>")
    assert len(parts) == 3, "the committed XML has a trigger flag and a settings flag"
    return "<Enabled>true</Enabled>".join(parts[: n + 1]) + (
        "<Enabled>false</Enabled>" + "<Enabled>true</Enabled>".join(parts[n + 1 :])
    )


@pytest.mark.parametrize(
    ("change", "field"),
    [
        (
            lambda x: x.replace("<UserId>jackc</UserId>", "<UserId>SYSTEM</UserId>"),
            "RunAs",
        ),
        (lambda x: _nth_enabled_false(x, 0), "Enabled"),  # the trigger disabled
        (lambda x: _nth_enabled_false(x, 1), "Enabled"),  # the task disabled
        (
            lambda x: x.replace(
                r"<WorkingDirectory>C:\Users\jackc\Code\nfl-predict<",
                r"<WorkingDirectory>C:\Users\jackc<",
            ),
            "WorkingDirectory",
        ),
    ],
)
def test_a_wrong_user_a_disabled_task_or_a_wrong_directory_is_a_mismatch(
    change, field: str
) -> None:
    exported = _daily_export()
    changed = change(exported)
    assert changed != exported, "the fixture edit did not apply"
    assert _mismatches(changed) == {field}


def test_the_exported_sid_matches_the_committed_account_name() -> None:
    """Windows exports the principal as a SID; the committed XML names the account."""
    sid = scheduling._lookup_account_sid("jackc")
    if sid is None:
        pytest.skip("the committed account does not resolve on this machine")
    exported = _daily_export().replace(
        "<UserId>jackc</UserId>", f"<UserId>{sid}</UserId>"
    )
    assert "RunAs" not in _mismatches(exported)
    assert scheduling.task_fields(exported)["RunAs"] == sid.upper()


def test_the_default_valued_enabled_flags_read_as_true_when_omitted() -> None:
    exported = _daily_export().replace("<Enabled>true</Enabled>", "")
    assert scheduling.task_fields(exported)["Enabled"] == "task=true; triggers=true"
    assert "Enabled" not in _mismatches(exported)


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


# ---------------------------------------------------------------------------
# The closing-line wake task (Plan 34-11, D-07): a SECOND allowed NFL task, read back on its
# settings AND on its full (StartBoundary, EndBoundary, Enabled) trigger set. Every schtasks call
# below is a fake: no test registers, queries or deletes a real scheduled task.
# ---------------------------------------------------------------------------

_CLOSING_NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)
_PIPELINE_ROW = '"\\NFL_Predict_Pipeline","9/24/2026 5:00:00 PM","Ready"\n'
_CLOSING_ROW = '"\\NFL_Predict_Closing","10/11/2026 12:50:00 PM","Ready"\n'
_OTHER_ROW = '"\\NFL_Predict_Predictions","N/A","Ready"\n'


def _closing_games(kickoffs: list[datetime]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"game_{n}" for n in range(len(kickoffs))],
            "kickoff_et": pd.to_datetime(kickoffs, utc=True),
            "home_score": [float("nan")] * len(kickoffs),
        }
    )


_SUNDAY_GAMES = [
    datetime(2026, 10, 11, 17, 0, tzinfo=UTC),  # 1:00 PM ET
    datetime(2026, 10, 12, 0, 20, tzinfo=UTC),  # 8:20 PM ET
]


def _closing_export(edit=lambda text: text) -> bytes:
    windows = windows_for_schedule(
        _closing_games(_SUNDAY_GAMES), _CLOSING_NOW, days=closing_schedule.HORIZON_DAYS
    )
    built = closing_schedule.build_closing_task_xml(
        closing_schedule.CLOSING_TEMPLATE_PATH.read_bytes(), windows
    )
    return b"\xff\xfe" + edit(built.decode("utf-16")).encode("utf-16-le")


def _fake_two_tasks(listing: str, closing_export: bytes):
    def _run(cmd: list[str], **_kwargs: object) -> MagicMock:
        if "/xml" in cmd:
            if cmd[cmd.index("/tn") + 1] == closing_schedule.CLOSING_TASK_NAME:
                return MagicMock(returncode=0, stdout=closing_export)
            return MagicMock(returncode=0, stdout=_daily_export().encode("ascii"))
        return MagicMock(returncode=0, stdout=listing)

    return _run


_DRIFTED_END = (
    "<EndBoundary>2026-10-11T13:20:00</EndBoundary>",
    "<EndBoundary>2026-10-11T13:50:00</EndBoundary>",
)


@pytest.mark.parametrize(
    ("listing", "closing_export", "expected", "closing_lines"),
    [
        (_PIPELINE_ROW, _closing_export(), True, []),
        (
            _PIPELINE_ROW + _CLOSING_ROW,
            _closing_export(),
            True,
            ["CLOSING_TRIGGERS= 2 | 2 | MATCH", "CLOSING_READBACK_MATCH= True"],
        ),
        (
            _PIPELINE_ROW + _CLOSING_ROW,
            _closing_export(lambda text: text.replace(*_DRIFTED_END)),
            False,
            ["CLOSING_TRIGGERS= 2 | 2 | MISMATCH", "CLOSING_READBACK_MATCH= False"],
        ),
        (_PIPELINE_ROW + _CLOSING_ROW + _OTHER_ROW, _closing_export(), False, []),
    ],
    ids=["pipeline-only", "both-tasks", "closing-end-drifted", "a-third-nfl-task"],
)
def test_verify_installed_accepts_both_tasks(
    capsys: pytest.CaptureFixture[str],
    listing: str,
    closing_export: bytes,
    expected: bool,
    closing_lines: list[str],
) -> None:
    setup = _bare_setup(Path(scheduling.__file__).resolve().parents[1])
    with patch.object(
        scheduling.subprocess,
        "run",
        side_effect=_fake_two_tasks(listing, closing_export),
    ):
        assert (
            setup.verify_installed(
                games=_closing_games(_SUNDAY_GAMES), now=_CLOSING_NOW
            )
            is expected
        )
    out = capsys.readouterr().out
    assert f"READBACK_MATCH= {expected}" in out
    for line in closing_lines:
        assert line in out
    if _CLOSING_ROW not in listing:
        assert "CLOSING_READBACK_MATCH" not in out


class _RecordingRunner:
    """A fake schtasks for --install-closing: registers in memory, exports what it registered."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []
        self.registered: bytes | None = None

    def __call__(self, args) -> closing_schedule.SchtasksResult:
        args = list(args)
        self.calls.append(args)
        if args[0] == "/create":
            self.registered = Path(args[args.index("/xml") + 1]).read_bytes()
            return closing_schedule.SchtasksResult(0, b"SUCCESS", b"")
        assert self.registered is not None
        return closing_schedule.SchtasksResult(0, self.registered, b"")


def _closing_home(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "daily_lock_pipeline.py").write_text("# the daily entry")
    (tmp_path / "deployment").mkdir()
    (tmp_path / "deployment" / "windows_closing_scheduler.xml").write_bytes(
        closing_schedule.CLOSING_TEMPLATE_PATH.read_bytes()
    )
    return tmp_path


@pytest.mark.parametrize(
    ("zone", "expected_exit"),
    [("Eastern Standard Time", 0), ("Pacific Standard Time", 1)],
)
def test_install_closing_flag(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, zone: str, expected_exit: int
) -> None:
    home = _closing_home(tmp_path)
    now = datetime.now(UTC)
    games = _closing_games(
        [now + timedelta(days=1), now + timedelta(days=3), now + timedelta(days=9)]
    )
    loads: list[tuple[tuple, dict]] = []

    def _load(*args, **kwargs):
        loads.append((args, kwargs))
        return games

    runner = _RecordingRunner()
    monkeypatch.setattr(scheduling, "read_windows_time_zone", lambda: zone)
    monkeypatch.setattr(scheduling, "load_dataframe", _load)
    monkeypatch.setattr(closing_schedule, "run_schtasks", runner)
    monkeypatch.setattr(
        "sys.argv",
        [
            "setup_scheduling.py",
            "--platform",
            "windows",
            "--install-closing",
            "--project-home",
            str(home),
        ],
    )

    assert scheduling.main() == expected_exit
    if expected_exit == 1:
        assert runner.calls == []
        assert loads == []
        return

    assert loads == [(("games",), {"layer": "silver"})]
    (create,) = [call for call in runner.calls if call[0] == "/create"]
    generated = home / "logs" / "closing_task_generated.xml"
    assert create == [
        "/create",
        "/tn",
        closing_schedule.CLOSING_TASK_NAME,
        "/xml",
        str(generated),
        "/f",
    ]
    # Two kickoffs inside the 8-day horizon, two days apart: two windows. The day-9 game is
    # registered by a later run, when the horizon has rolled forward.
    root = ET.fromstring(generated.read_bytes().decode("utf-16"))
    triggers = root.find(f"{scheduling._TASK_NS}Triggers")
    assert triggers is not None
    assert [t.tag.split("}")[-1] for t in triggers] == ["TimeTrigger", "TimeTrigger"]
