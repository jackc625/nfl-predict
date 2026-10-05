"""The closing task's triggers come from the real schedule and are read back in full (Plan 34-11).

D-07: the laptop wakes for closing-line readings only at real kickoff windows, on AC power, never
late and never on a repeating timer. ``forward_ledger.closing_schedule`` turns the schedule's
reading windows into ``TimeTrigger`` elements (StartBoundary in naive Eastern wall-clock time,
EndBoundary 30 minutes later so a stale trigger cannot fire late) and re-registers the one closing
task, then reads it back. Every ``schtasks`` call goes through an injected runner: NO test here
touches the real Task Scheduler.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from forward_ledger.closing_schedule import (
    CLOSING_TASK_NAME,
    CLOSING_TEMPLATE_PATH,
    SchtasksResult,
    build_closing_task_xml,
    expected_trigger_starts,
    expected_triggers,
    register_closing_triggers,
)
from forward_ledger.closing_windows import ReadingWindow

_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"
_EASTERN = "Eastern Standard Time"

#: Friday 2026-10-09, 8:00 AM ET: the registration instant for the Sunday slate below.
_NOW = datetime(2026, 10, 9, 12, 0, tzinfo=UTC)


def _games() -> pd.DataFrame:
    """A Sunday slate: 1:00, 4:05 and 4:25 (one shared window), 8:20 PM ET -> three windows."""
    kickoffs = {
        "2026_06_AAA_BBB": datetime(2026, 10, 11, 17, 0, tzinfo=UTC),  # 1:00 PM ET
        "2026_06_CCC_DDD": datetime(2026, 10, 11, 20, 5, tzinfo=UTC),  # 4:05 PM ET
        "2026_06_EEE_FFF": datetime(2026, 10, 11, 20, 25, tzinfo=UTC),  # 4:25 PM ET
        "2026_06_GGG_HHH": datetime(2026, 10, 12, 0, 20, tzinfo=UTC),  # 8:20 PM ET
    }
    return pd.DataFrame(
        {
            "game_id": list(kickoffs),
            "kickoff_et": pd.to_datetime(list(kickoffs.values()), utc=True),
            "home_score": [float("nan")] * len(kickoffs),
        }
    )


_EXPECTED_STARTS = [
    "2026-10-11T12:50:00",
    "2026-10-11T15:55:00",
    "2026-10-11T20:10:00",
]


def _template() -> bytes:
    return CLOSING_TEMPLATE_PATH.read_bytes()


def _window(start_utc: datetime) -> ReadingWindow:
    return ReadingWindow(
        start_utc=start_utc, kickoffs_utc=(start_utc,), game_ids=("g",)
    )


def _time_triggers(xml_bytes: bytes) -> list[ET.Element]:
    root = ET.fromstring(xml_bytes.decode("utf-16"))
    triggers = root.find(f"{_NS}Triggers")
    assert triggers is not None
    return list(triggers)


def _text(element: ET.Element, name: str) -> str:
    child = element.find(f"{_NS}{name}")
    assert child is not None, f"<{name}> is missing"
    return (child.text or "").strip()


def _without_triggers(xml_bytes: bytes) -> str:
    root = ET.fromstring(xml_bytes.decode("utf-16"))
    triggers = root.find(f"{_NS}Triggers")
    assert triggers is not None
    for trigger in list(triggers):
        triggers.remove(trigger)
    return ET.canonicalize(ET.tostring(root, encoding="unicode"), strip_text=True)


# ---------------------------------------------------------------------------
# Building the task definition
# ---------------------------------------------------------------------------


def test_build_inserts_one_time_trigger_per_window() -> None:
    windows = [
        _window(datetime(2026, 10, 11, 16, 50, tzinfo=UTC)),
        _window(datetime(2026, 10, 11, 19, 55, tzinfo=UTC)),
        _window(datetime(2026, 10, 12, 0, 10, tzinfo=UTC)),
    ]
    built = build_closing_task_xml(_template(), windows)

    assert built[:2] == b"\xff\xfe", "UTF-16 LE with a byte-order mark"
    body = built[2:].decode("utf-16-le")
    assert "\n" not in body.replace("\r\n", ""), "CRLF line endings"

    triggers = _time_triggers(built)
    assert [t.tag for t in triggers] == [f"{_NS}TimeTrigger"] * 3
    assert [_text(t, "StartBoundary") for t in triggers] == _EXPECTED_STARTS
    assert [_text(t, "EndBoundary") for t in triggers] == [
        "2026-10-11T13:20:00",
        "2026-10-11T16:25:00",
        "2026-10-11T20:40:00",
    ]
    assert [_text(t, "Enabled") for t in triggers] == ["true"] * 3
    # Everything but the triggers is the template, unchanged.
    assert _without_triggers(built) == _without_triggers(_template())


def test_build_handles_dst_change() -> None:
    """Sunday 2026-11-01 (DST ends at 2 AM): a 1:00 PM EST kickoff's window is 12:50 EST."""
    before = _window(datetime(2026, 10, 25, 16, 50, tzinfo=UTC))  # 12:50 PM EDT (UTC-4)
    after = _window(datetime(2026, 11, 1, 17, 50, tzinfo=UTC))  # 12:50 PM EST (UTC-5)
    triggers = _time_triggers(build_closing_task_xml(_template(), [before, after]))

    assert [_text(t, "StartBoundary") for t in triggers] == [
        "2026-10-25T12:50:00",
        "2026-11-01T12:50:00",
    ]
    assert [_text(t, "EndBoundary") for t in triggers] == [
        "2026-10-25T13:20:00",
        "2026-11-01T13:20:00",
    ]


def test_expected_starts_and_triggers_come_from_the_schedule() -> None:
    assert expected_trigger_starts(_games(), _NOW) == _EXPECTED_STARTS
    assert expected_triggers(_games(), _NOW) == frozenset(
        {
            ("2026-10-11T12:50:00", "2026-10-11T13:20:00", "true"),
            ("2026-10-11T15:55:00", "2026-10-11T16:25:00", "true"),
            ("2026-10-11T20:10:00", "2026-10-11T20:40:00", "true"),
        }
    )


# ---------------------------------------------------------------------------
# The daily registration
# ---------------------------------------------------------------------------


class _FakeSchtasks:
    """Stands in for ``schtasks``: records every call, never touches Task Scheduler."""

    def __init__(
        self,
        *,
        installed: bool = True,
        status: str = "Ready",
        export_edit: Callable[[str], str] = lambda text: text,
    ) -> None:
        self.installed = installed
        self.status = status
        self.export_edit = export_edit
        self.calls: list[list[str]] = []
        self.registered: bytes | None = None

    def __call__(self, args: Sequence[str]) -> SchtasksResult:
        args = list(args)
        self.calls.append(args)
        if args[0] == "/create":
            self.registered = Path(args[args.index("/xml") + 1]).read_bytes()
            return SchtasksResult(0, b"SUCCESS", b"")
        if "/xml" in args:
            assert self.registered is not None, "export before any registration"
            text = self.export_edit(self.registered.decode("utf-16"))
            return SchtasksResult(0, b"\xff\xfe" + text.encode("utf-16-le"), b"")
        if not self.installed:
            return SchtasksResult(1, b"", b"ERROR: The system cannot find the file.")
        row = f'"\\{CLOSING_TASK_NAME}","10/11/2026 12:50:00 PM","{self.status}"\r\n'
        return SchtasksResult(0, row.encode("ascii"), b"")

    @property
    def created(self) -> bool:
        return any(call[0] == "/create" for call in self.calls)


def _register(fake: _FakeSchtasks, tmp_path: Path, zone: str = _EASTERN):
    return register_closing_triggers(
        _games(),
        _NOW,
        runner=fake,
        zone_reader=lambda: zone,
        generated_path=tmp_path / "closing_task_generated.xml",
    )


def test_register_skips_when_not_installed(tmp_path: Path) -> None:
    fake = _FakeSchtasks(installed=False)
    result = _register(fake, tmp_path)
    assert (result.outcome, result.reason) == ("skipped", "closing_task_not_installed")
    assert not fake.created


def test_register_skips_when_running(tmp_path: Path) -> None:
    fake = _FakeSchtasks(status="Running")
    result = _register(fake, tmp_path)
    assert (result.outcome, result.reason) == ("skipped", "closing_task_running")
    assert not fake.created


def test_register_refuses_non_eastern_machine(tmp_path: Path) -> None:
    fake = _FakeSchtasks()
    result = _register(fake, tmp_path, zone="Pacific Standard Time")
    assert (result.outcome, result.reason) == ("skipped", "machine_not_eastern")
    assert fake.calls == []


def _first(text: str, old: str, new: str) -> str:
    assert old in text, f"the fixture edit {old!r} did not apply"
    return text.replace(old, new, 1)


_END = "<EndBoundary>2026-10-11T13:20:00</EndBoundary>"


@pytest.mark.parametrize(
    ("export_edit", "expected_match"),
    [
        (lambda text: text, True),
        # The StartBoundary set is unchanged in each case below; only the full tuple catches it.
        (lambda text: _first(text, _END, _END.replace("13:20", "13:50")), False),
        (lambda text: _first(text, _END, ""), False),
        (
            lambda text: _first(
                text, "<Enabled>true</Enabled>", "<Enabled>false</Enabled>"
            ),
            False,
        ),
        # A settings field drifted.
        (
            lambda text: _first(
                text, "scripts.capture_closing_lines", "scripts.something_else"
            ),
            False,
        ),
    ],
    ids=[
        "match",
        "end-drifted",
        "end-missing",
        "trigger-disabled",
        "arguments-drifted",
    ],
)
def test_register_creates_and_reads_back(
    tmp_path: Path, export_edit: Callable[[str], str], expected_match: bool
) -> None:
    fake = _FakeSchtasks(export_edit=export_edit)
    result = _register(fake, tmp_path)

    (create,) = [call for call in fake.calls if call[0] == "/create"]
    generated = tmp_path / "closing_task_generated.xml"
    assert create == [
        "/create",
        "/tn",
        CLOSING_TASK_NAME,
        "/xml",
        str(generated),
        "/f",
    ]
    assert fake.registered == generated.read_bytes()
    assert ["/query", "/tn", CLOSING_TASK_NAME, "/xml"] in fake.calls

    assert result.outcome == "registered"
    assert result.expected_starts == tuple(_EXPECTED_STARTS)
    assert result.readback_match is expected_match
    assert result.readback_line == f"CLOSING_READBACK_MATCH= {expected_match}"
    assert result.reason == (None if expected_match else "readback_mismatch")


def test_a_failed_create_is_recorded_not_raised(tmp_path: Path) -> None:
    class _Refusing(_FakeSchtasks):
        def __call__(self, args: Sequence[str]) -> SchtasksResult:
            if args[0] == "/create":
                self.calls.append(list(args))
                return SchtasksResult(1, b"", b"ERROR: Access is denied.")
            return super().__call__(args)

    result = _register(_Refusing(), tmp_path)
    assert result.outcome == "failed"
    assert result.reason is not None and "Access is denied" in result.reason
    assert result.readback_match is False
