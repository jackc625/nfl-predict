"""The owner-run real-fill CLI and the manual sync CLI (Plan 34-12 Task 2; LDGR-04, D-01, D-17).

``scripts/record_fill.py`` is the ONLY path a real bet's facts reach the ledger: each fill column
moves from NULL to a value once, a second write is refused by name with the file unchanged, a
naive fill instant is refused, and a successful fill triggers the backup sync (D-01: every
ledger-changing run pushes the backup). ``scripts/sync_ledger.py`` runs that one sync by hand and
prints every outcome; it fails only on a broken chain.

Ledgers live under ``tmp_path``; the sync is a fake for the fill CLI, and the sync CLI's injected
runner runs real git on ``tmp_path`` repositories while REFUSING every push, fetch and ls-remote --
nothing reaches GitHub, the project's refs or ``logs/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from forward_ledger.backup import ensure_backup_repo
from forward_ledger.store import append_rows, ledger_path, read_entries
from forward_ledger.sync import SyncOutcome
from forward_ledger.transport import run_git
from scripts import record_fill, sync_ledger
from tests.unit.test_forward_ledger_store import make_row
from tests.unit.test_ledger_anchor import make_repo

ROW = make_row()
KEY_ARGS = [
    "--game-id",
    ROW["game_id"],
    "--season",
    str(ROW["season"]),
    "--week",
    str(ROW["week"]),
    "--target",
    ROW["target"],
]
FILL_ARGS = [
    "--sportsbook",
    "draftkings",
    "--line",
    "-3.5",
    "--odds",
    "-110",
    "--stake-dollars",
    "50",
    "--filled-at",
    "2026-10-18T12:01:00-04:00",
]


class _Recorder:
    """Stands in for the run log and the sync: records every call, writes nothing."""

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append((args, kwargs))
        return SyncOutcome(False, False, None, True, True, None, None)


@pytest.fixture
def fill_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    ledger = tmp_path / "ledger"
    append_rows(ledger, [ROW])
    log, sync = _Recorder(), _Recorder()
    monkeypatch.setattr(record_fill, "LOG", log)
    monkeypatch.setattr(record_fill, "PUBLISH", sync)
    return {"ledger": ledger, "log": log, "sync": sync}


def _fill(ledger: Path, *extra: str) -> int:
    return record_fill.main(["--ledger-dir", str(ledger), *KEY_ARGS, *extra])


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------------------
# scripts/record_fill.py
# ---------------------------------------------------------------------------


def test_fill_recorded_once(fill_world: dict[str, Any], capsys) -> None:
    ledger = fill_world["ledger"]
    before = read_entries(ledger)[0]

    assert _fill(ledger, *FILL_ARGS) == 0

    out = capsys.readouterr().out
    assert "FILL_RECORDED=" in out
    after = read_entries(ledger)[0]
    assert after.fill == {
        "fill_sportsbook": "draftkings",
        "fill_line": -3.5,
        "fill_odds": -110.0,
        "fill_stake_dollars": 50.0,
        "fill_at_utc": "2026-10-18T16:01:00+00:00",
    }
    assert after.immutable == before.immutable
    assert after.chain_hash == before.chain_hash
    assert after.grading == before.grading
    assert [call[0][0] for call in fill_world["log"].calls] == ["fill_recorded"]


def test_second_fill_refused(fill_world: dict[str, Any], capsys) -> None:
    ledger = fill_world["ledger"]
    assert _fill(ledger, "--sportsbook", "draftkings") == 0
    stored = _sha(ledger_path(ledger))
    capsys.readouterr()

    assert _fill(ledger, "--sportsbook", "fanduel") == 1

    out = capsys.readouterr().out
    assert "FILL_REFUSED=" in out
    assert "fill_sportsbook" in out
    assert _sha(ledger_path(ledger)) == stored


def test_naive_filled_at_refused(fill_world: dict[str, Any], capsys) -> None:
    ledger = fill_world["ledger"]
    stored = _sha(ledger_path(ledger))

    assert _fill(ledger, "--odds", "-110", "--filled-at", "2026-10-18T12:01:00") == 2

    assert "FILL_REFUSED=" in capsys.readouterr().out
    assert _sha(ledger_path(ledger)) == stored
    assert fill_world["sync"].calls == []


def test_no_fill_value_refused(fill_world: dict[str, Any]) -> None:
    assert _fill(fill_world["ledger"]) == 2
    assert fill_world["sync"].calls == []


def test_unknown_key_refused(fill_world: dict[str, Any], capsys) -> None:
    ledger = fill_world["ledger"]
    stored = _sha(ledger_path(ledger))

    code = record_fill.main(
        [
            "--ledger-dir",
            str(ledger),
            *KEY_ARGS[:-1],
            "wp",
            "--sportsbook",
            "draftkings",
        ]
    )

    assert code == 1
    assert "FILL_REFUSED=" in capsys.readouterr().out
    assert _sha(ledger_path(ledger)) == stored


def test_fill_triggers_backup_sync(fill_world: dict[str, Any]) -> None:
    ledger = fill_world["ledger"]
    sync = fill_world["sync"]

    assert _fill(ledger, *FILL_ARGS) == 0
    assert len(sync.calls) == 1
    _, kwargs = sync.calls[0]
    assert Path(kwargs["ledger_dir"]) == ledger
    assert kwargs["log"] is fill_world["log"]

    assert _fill(ledger, "--line", "-4.0") == 1
    assert len(sync.calls) == 1


# ---------------------------------------------------------------------------
# scripts/sync_ledger.py
# ---------------------------------------------------------------------------


def _offline_runner(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Real git for local commands; every push, fetch and ls-remote fails as if offline."""
    if args and args[0] in ("push", "fetch", "ls-remote"):
        return subprocess.CompletedProcess(
            list(args), 128, b"", b"simulated: could not resolve host"
        )
    return run_git(args, cwd=cwd, env=env, timeout=timeout, input_bytes=input_bytes)


@pytest.fixture
def sync_world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = make_repo(tmp_path / "public")
    ledger = tmp_path / "ledger"
    ensure_backup_repo(ledger)
    append_rows(ledger, [ROW])
    log = _Recorder()
    monkeypatch.setattr(sync_ledger, "RUNNER", _offline_runner)
    monkeypatch.setattr(sync_ledger, "LOG", log)
    return {"repo": repo, "ledger": ledger, "log": log}


def _sync(world: dict[str, Any]) -> int:
    return sync_ledger.main(
        ["--ledger-dir", str(world["ledger"]), "--repo-dir", str(world["repo"])]
    )


def test_sync_cli_prints_outcomes(sync_world: dict[str, Any], capsys) -> None:
    assert _sync(sync_world) == 0

    out = capsys.readouterr().out
    assert "ANCHOR_COMMITTED= True" in out
    assert "ANCHOR_PUSHED= False" in out
    assert "BACKUP_COMMITTED= True" in out
    assert "BACKUP_PUSHED= False" in out
    assert "SYNC_RESULT=" in out
    events = [call[0][0] for call in sync_world["log"].calls]
    assert "anchor_commit" in events
    assert "backup_push" in events


def test_sync_cli_fails_on_a_broken_chain(sync_world: dict[str, Any], capsys) -> None:
    path = ledger_path(sync_world["ledger"])
    path.write_bytes(
        path.read_bytes().replace(b'"bet_side":"home"', b'"bet_side":"away"')
    )

    assert _sync(sync_world) == 1

    out = capsys.readouterr().out
    assert "SYNC_REFUSED=" in out
    assert "ANCHOR_COMMITTED= False" in out
