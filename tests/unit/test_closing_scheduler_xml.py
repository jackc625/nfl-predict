"""The closing-line wake task's committed template is PINNED and says what D-07 requires (Plan 34-11).

WHY A PIN
---------
``deployment/windows_closing_scheduler.xml`` is the definition Plan 34-11's registration function
re-registers every day with generated triggers, and the owner-run ``--install-closing`` installs.
The installed task's read-back compares against what this file says, so an unreviewed edit here
would silently become the installed definition. The hash is over the RAW BYTES (UTF-16 LE with a
BOM, CRLF): a line-ending or encoding normalisation is a real change to a file Windows Task
Scheduler imports verbatim, so it must fail here rather than be smoothed away. If the pin is being
updated deliberately, say so in the commit and update EXPECTED_SHA256 and EXPECTED_BYTES together.

WHY THE SETTINGS ARE ASSERTED AGAINST THE DAILY TASK
----------------------------------------------------
D-07: the closing task wakes "exactly like the daily run" -- S4U logon, wake on AC power only,
never run a missed trigger late, absolute ``uv.exe``. Asserting each value is not enough on its
own: the two definitions could each pass their own tests and still drift apart. So every
Principal field and every shared Settings field is compared with ``windows_scheduler.xml``'s.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import pathlib
import xml.etree.ElementTree as ET

CLOSING_PATH = pathlib.Path("deployment") / "windows_closing_scheduler.xml"
DAILY_PATH = pathlib.Path("deployment") / "windows_scheduler.xml"

# PINNED by Plan 34-11 Task 1 (2026-10-05): the template as first committed.
EXPECTED_SHA256 = "0000000000000000000000000000000000000000000000000000000000000000"
EXPECTED_BYTES = 0

#: The one Settings field the closing task deliberately does NOT share with the daily task: a
#: closing reading is one small odds request, so 20 minutes bounds a hung run well inside the
#: shortest gap between two reading windows.
CLOSING_EXECUTION_TIME_LIMIT = "PT20M"

_TASK_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"


def _parsed(path: pathlib.Path) -> ET.Element:
    return ET.fromstring(path.read_bytes().decode("utf-16"))


def _child_texts(parent: ET.Element) -> dict[str, str]:
    """Every direct child of *parent* that holds text, by local name (nested blocks flattened)."""
    texts: dict[str, str] = {}
    for element in parent.iter():
        if element is parent or len(element):
            continue
        texts[element.tag.split("}")[-1]] = (element.text or "").strip()
    return texts


def _block(root: ET.Element, name: str) -> ET.Element:
    block = root.find(f"{_TASK_NS}{name}")
    assert block is not None, f"<{name}> is missing"
    return block


def test_template_byte_pin() -> None:
    raw = CLOSING_PATH.read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    assert len(raw) == EXPECTED_BYTES, (
        f"{CLOSING_PATH.as_posix()} is {len(raw)} bytes, pinned at {EXPECTED_BYTES}"
    )
    assert actual == EXPECTED_SHA256, (
        f"{CLOSING_PATH.as_posix()} content changed.\n  expected sha256 {EXPECTED_SHA256}\n"
        f"  actual   sha256 {actual}"
    )


def test_template_is_utf16_bom_crlf() -> None:
    raw = CLOSING_PATH.read_bytes()
    assert raw[:2] == b"\xff\xfe", (
        "the template must start with a UTF-16 LE byte-order mark"
    )
    text = raw[2:].decode("utf-16-le")
    assert "\r\n" in text
    assert "\n" not in text.replace("\r\n", ""), "every line ending must be CRLF"
    assert text.isascii(), "the template is ASCII text in a UTF-16 encoding"


def test_template_settings() -> None:
    root = _parsed(CLOSING_PATH)
    principal = _child_texts(_block(root, "Principals"))
    settings = _child_texts(_block(root, "Settings"))
    action = _child_texts(_block(root, "Actions"))

    assert principal["LogonType"] == "S4U"
    assert principal["RunLevel"] == "HighestAvailable"
    assert principal["UserId"] == "jackc"
    assert settings["WakeToRun"] == "true"
    assert settings["StartWhenAvailable"] == "false"
    assert settings["DisallowStartIfOnBatteries"] == "false"
    assert settings["StopIfGoingOnBatteries"] == "false"
    assert settings["MultipleInstancesPolicy"] == "IgnoreNew"
    assert settings["ExecutionTimeLimit"] == CLOSING_EXECUTION_TIME_LIMIT

    command = pathlib.PureWindowsPath(action["Command"])
    assert command.is_absolute(), "the S4U logon does not load the owner's PATH"
    assert command.name == "uv.exe"
    assert action["Arguments"] == "run python -m scripts.capture_closing_lines"
    assert action["WorkingDirectory"] == r"C:\Users\jackc\Code\nfl-predict"

    triggers = _block(root, "Triggers")
    assert len(triggers) == 0, "the template carries NO trigger; triggers are generated"


def test_principal_and_settings_match_the_daily_task() -> None:
    closing, daily = _parsed(CLOSING_PATH), _parsed(DAILY_PATH)

    assert _child_texts(_block(closing, "Principals")) == _child_texts(
        _block(daily, "Principals")
    )

    closing_settings = _child_texts(_block(closing, "Settings"))
    daily_settings = _child_texts(_block(daily, "Settings"))
    assert set(closing_settings) == set(daily_settings)
    differing = {
        name
        for name in daily_settings
        if closing_settings[name] != daily_settings[name]
    }
    assert differing == {"ExecutionTimeLimit"}, sorted(differing)

    closing_action = _child_texts(_block(closing, "Actions"))
    daily_action = _child_texts(_block(daily, "Actions"))
    assert closing_action["Command"] == daily_action["Command"]
    assert closing_action["WorkingDirectory"] == daily_action["WorkingDirectory"]
