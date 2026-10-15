"""The owner command that marks one observed sealed-zone upstream change as seen (Phase 35 D-04).

``scripts/acknowledge_upstream_change.py`` wraps ``data.sealed_probe.acknowledge_divergence``:

* it copies the observed values from the LATEST probe line's finding for the pair, so the owner
  never types a digest, and records ``ruled_by``, ``ruled_at_utc`` and a REQUIRED reason;
* it writes only the sealed lock -- atomically, so an interrupted write leaves the previous lock --
  and never opens the pin manifest (an acknowledgement is not a re-freeze, D32-10);
* the probe then reads the change as acknowledged for the same observed values.

Every lock, probe log and pin manifest lives under ``tmp_path``; the real ``config/`` files are
never read or written. The probe line is produced by the real ``probe_sealed`` and appended by the
real ``append_probe_entry``, so its shape is the shape the daily capture writes.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import builtins
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pytest

from data.sealed_probe import probe_sealed
from data.sealed_probe_log import append_probe_entry
from data.upstream_pin import SEALED_LOCK_SCHEMA_VERSION, load_sealed_lock
from scripts import acknowledge_upstream_change

BASELINE_SHA = "a" * 64
OBSERVED_SHA = "b" * 64
PIN_MANIFEST_NAME = "upstream_pin.json"


def write_lock(path: Path) -> None:
    """A sealed lock tracking schedules 2025 under the content strategy (as the real one does)."""
    lock = {
        "schema_version": SEALED_LOCK_SCHEMA_VERSION,
        "datasets": {"schedules": {"2025": {"sha256": BASELINE_SHA}}},
    }
    path.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8")


def write_probe_log(path: Path, lock_path: Path) -> None:
    """One real probe line: schedules 2025 moved and is not acknowledged."""
    verdict = probe_sealed(
        load_sealed_lock(lock_path),
        assets_by_tag={},
        content_digests={("schedules", 2025): OBSERVED_SHA},
    )
    assert verdict["findings"][0]["acknowledged"] is False
    append_probe_entry(verdict, path=path)


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    """A tmp sealed lock, a pin manifest beside it, and a probe log in its own directory."""
    lock = tmp_path / "config" / "upstream_pin.sealed.lock"
    lock.parent.mkdir()
    write_lock(lock)
    pin = lock.parent / PIN_MANIFEST_NAME
    pin.write_text('{"pinned": "do not touch"}\n', encoding="utf-8")
    probe_log = tmp_path / "probe" / "upstream_probe_log.jsonl"
    write_probe_log(probe_log, lock)
    return {"lock": lock, "pin": pin, "probe_log": probe_log}


def run(
    paths: dict[str, Path], *extra: str, reason: str | None = "checked by hand"
) -> int:
    """Run the CLI against the tmp files for schedules 2025 (overridable by *extra*)."""
    argv = [
        "--lock-path",
        str(paths["lock"]),
        "--probe-log",
        str(paths["probe_log"]),
    ]
    if not any(flag == "--dataset" for flag in extra):
        argv += ["--dataset", "schedules", "--season", "2025"]
    if reason is not None:
        argv += ["--reason", reason]
    return acknowledge_upstream_change.main([*argv, *extra])


def acknowledgement(lock_path: Path) -> dict[str, Any]:
    lock = load_sealed_lock(lock_path)
    assert lock is not None
    return lock["datasets"]["schedules"]["2025"]["acknowledgement"]


def test_an_acknowledgement_is_written_with_the_observed_values(
    paths: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    code = run(paths, reason="schedules 2025 re-upload checked: bytes match the pin")

    out = capsys.readouterr().out
    assert code == 0
    assert "ACK_RECORDED= schedules 2025" in out
    recorded = acknowledgement(paths["lock"])
    assert recorded["observed_sha256"] == OBSERVED_SHA
    assert recorded["ruled_by"] == "owner"
    assert recorded["reason"] == "schedules 2025 re-upload checked: bytes match the pin"
    assert recorded["supersedes"] is None
    ruled_at = datetime.fromisoformat(recorded["ruled_at_utc"])
    offset = ruled_at.utcoffset()
    assert offset is not None
    assert offset.total_seconds() == 0


def test_the_probe_reads_it_as_acknowledged(paths: dict[str, Path]) -> None:
    assert run(paths) == 0

    verdict = probe_sealed(
        load_sealed_lock(paths["lock"]),
        assets_by_tag={},
        content_digests={("schedules", 2025): OBSERVED_SHA},
    )

    (finding,) = verdict["findings"]
    assert finding["acknowledged"] is True
    assert finding["ruled_by"] == "owner"


def test_a_reason_is_required(
    paths: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    before = paths["lock"].read_bytes()

    with pytest.raises(SystemExit) as missing:
        run(paths, reason=None)
    assert missing.value.code == 2

    code = run(paths, reason="   ")
    out = capsys.readouterr().out
    assert code == 2
    assert "ACK_REFUSED=" in out
    assert paths["lock"].read_bytes() == before


def test_an_unknown_pair_is_refused(
    paths: dict[str, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    before = paths["lock"].read_bytes()

    code = run(paths, "--dataset", "schedules", "--season", "2024")

    out = capsys.readouterr().out
    assert code == 1
    assert "ACK_REFUSED= schedules 2024 has no finding in the latest probe line" in out
    assert paths["lock"].read_bytes() == before


def test_the_pin_manifest_is_never_touched(
    paths: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    pin_before = paths["pin"].read_bytes()
    real_open = builtins.open
    real_path_open = Path.open

    def guarded_open(file: Any, *args: Any, **kwargs: Any) -> Any:
        if Path(str(file)).name == PIN_MANIFEST_NAME:
            pytest.fail("the acknowledge command opened the pin manifest")
        return real_open(file, *args, **kwargs)

    def guarded_path_open(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == PIN_MANIFEST_NAME:
            pytest.fail("the acknowledge command opened the pin manifest")
        return real_path_open(self, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)

    assert run(paths) == 0

    monkeypatch.undo()
    assert acknowledgement(paths["lock"])["observed_sha256"] == OBSERVED_SHA
    assert paths["pin"].read_bytes() == pin_before


def test_a_second_acknowledgement_supersedes_the_first(paths: dict[str, Path]) -> None:
    assert run(paths, reason="first ruling") == 0
    first = acknowledgement(paths["lock"])

    assert run(paths, "--ruled-by", "jack", reason="second ruling") == 0

    second = acknowledgement(paths["lock"])
    assert second["reason"] == "second ruling"
    assert second["ruled_by"] == "jack"
    assert second["supersedes"] == first


def test_an_interrupted_write_leaves_the_previous_lock_intact(
    paths: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    before = paths["lock"].read_bytes()

    def interrupted(source: Any, target: Any) -> None:
        raise OSError("simulated interruption")

    monkeypatch.setattr(acknowledge_upstream_change.os, "replace", interrupted)

    code = run(paths)

    out = capsys.readouterr().out
    assert code == 1
    assert "ACK_REFUSED= the sealed lock could not be written (OSError)" in out
    assert paths["lock"].read_bytes() == before
    leftovers = [
        name for name in os.listdir(paths["lock"].parent) if name.endswith(".tmp")
    ]
    assert leftovers == []
