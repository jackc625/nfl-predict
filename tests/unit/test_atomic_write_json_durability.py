"""``_atomic_write_json`` flushes to disk BEFORE the atomic rename.

Review fix WR-13 (commit c83623e). ``artifacts/latest.json`` is the only production model-swap
surface. Renaming a temp file over it is atomic against a crashed process, but without an
``os.fsync`` first, power loss can land the rename while the contents are still only in the page
cache -- a renamed but EMPTY ``latest.json``. The ORDER is the whole fix, so it is what is pinned:
fsync of the temp file, then the replace.

Two behaviors, both writing only under ``tmp_path``:

* the happy path records ``fsync`` before ``replace``, round-trips the dict, and leaves no temp file;
* a failure mid-write (an unserializable payload, or an fsync that raises) leaves the existing
  destination byte-for-byte untouched and removes the temp file.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from models import artifacts

_PAYLOAD = {"wp": "wp_20260824_113325", "ats": "ats_20260605_220128", "n": [1, 2]}


def test_fsync_happens_before_the_rename_and_the_file_round_trips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    events: list[str] = []
    real_fsync = os.fsync
    real_replace = Path.replace

    def recording_fsync(fd: int) -> None:
        events.append("fsync")
        real_fsync(fd)

    def recording_replace(self: Path, target: str | Path) -> Path:
        events.append("replace")
        return real_replace(self, target)

    monkeypatch.setattr(artifacts.os, "fsync", recording_fsync)
    monkeypatch.setattr(Path, "replace", recording_replace)

    destination = tmp_path / "latest.json"
    artifacts._atomic_write_json(destination, _PAYLOAD)

    assert events == ["fsync", "replace"], (
        f"saw {events}: the temp file must be fsynced before the rename, or power loss can "
        "leave a renamed but empty latest.json"
    )
    assert json.loads(destination.read_text(encoding="utf-8")) == _PAYLOAD
    assert [p.name for p in tmp_path.iterdir()] == ["latest.json"]


def test_an_unserializable_payload_leaves_the_destination_untouched(
    tmp_path: Path,
) -> None:
    destination = tmp_path / "latest.json"
    destination.write_text('{"keep": "me"}', encoding="utf-8")

    with pytest.raises(TypeError):
        artifacts._atomic_write_json(destination, {"bad": object()})

    assert destination.read_text(encoding="utf-8") == '{"keep": "me"}'
    assert [p.name for p in tmp_path.iterdir()] == ["latest.json"]


def test_a_failing_fsync_leaves_the_destination_untouched_and_cleans_the_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "latest.json"
    destination.write_text('{"keep": "me"}', encoding="utf-8")

    def failing_fsync(fd: int) -> None:
        raise OSError("simulated disk failure")

    monkeypatch.setattr(artifacts.os, "fsync", failing_fsync)

    with pytest.raises(OSError, match="simulated disk failure"):
        artifacts._atomic_write_json(destination, _PAYLOAD)

    assert destination.read_text(encoding="utf-8") == '{"keep": "me"}', (
        "the old manifest must survive a write that never reached the rename"
    )
    assert [p.name for p in tmp_path.iterdir()] == ["latest.json"], (
        "the temp file must be removed when the write fails"
    )
