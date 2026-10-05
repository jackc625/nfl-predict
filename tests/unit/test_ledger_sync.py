"""The one sync step: anchor the head when it moved, back up whatever changed, never fail a run.

``forward_ledger.sync.publish_ledger_state`` runs after the ledger is written (Plan 34-06 Task 2).
It verifies the chain, makes ONE anchor commit when the head moved -- a new row or a correction
entry -- and none otherwise, pushes the anchor when it is ahead of the last push, and commits and
pushes the private backup whenever anything in ``ledger/`` changed (grading and closing writes
included). Every failure is recorded in the outcome and the run log; the function never raises
(LDGR-05, D-01). Every event carries the ``head_hash`` and ``entry_count`` the sync read.

Real git against ``tmp_path`` repositories and local bare "remotes"; nothing reaches GitHub, the
project's own refs, or the owner's ssh directory.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from forward_ledger.anchor import read_local_anchor, read_remote_anchor
from forward_ledger.backup import ensure_backup_repo
from forward_ledger.remote_config import ANCHOR_FILE_NAME, ANCHOR_REF, RemoteSpec
from forward_ledger.store import (
    append_rows,
    apply_updates,
    commit_changes,
    ledger_head,
    ledger_path,
    read_entries,
)
from forward_ledger.sync import SyncOutcome, publish_ledger_state
from forward_ledger.transport import GitRunner, run_git
from tests.unit.test_forward_ledger_store import graded, key_of, make_row, week_rows
from tests.unit.test_ledger_anchor import (
    fake_spec,
    git,
    git_out,
    make_bare,
    make_repo,
    ref_sha,
)

NOW = datetime(2026, 10, 15, 1, 0, tzinfo=UTC)
SYNC_EVENTS = (
    "anchor_commit",
    "anchor_push",
    "backup_commit",
    "backup_push",
    "sync_refused",
)


@dataclass
class World:
    """A public repo with its bare remote, and a backed-up ledger with its bare remote."""

    repo: Path
    public_remote: Path
    ledger: Path
    backup_remote: Path
    anchor_spec: RemoteSpec
    backup_spec: RemoteSpec
    events: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def log(self, event: str, **fields: Any) -> None:
        self.events.append((event, fields))

    def sync(self, runner: GitRunner = run_git) -> SyncOutcome:
        return publish_ledger_state(
            ledger_dir=self.ledger,
            repo_dir=self.repo,
            anchor_spec=self.anchor_spec,
            backup_spec=self.backup_spec,
            runner=runner,
            log=self.log,
            now=NOW,
        )

    def head(self) -> tuple[str, int]:
        return ledger_head(read_entries(self.ledger))

    def anchor_commits(self) -> int:
        if ref_sha(self.repo, ANCHOR_REF) is None:
            return 0
        return int(git_out(self.repo, "rev-list", "--count", ANCHOR_REF))


@pytest.fixture
def world(tmp_path: Path) -> World:
    repo = make_repo(tmp_path / "public")
    public_remote = make_bare(tmp_path / "public.git")
    git_out(
        repo, "push", public_remote.as_posix(), "refs/heads/master:refs/heads/master"
    )
    ledger = tmp_path / "ledger"
    ensure_backup_repo(ledger)
    backup_remote = make_bare(tmp_path / "backup.git")
    return World(
        repo=repo,
        public_remote=public_remote,
        ledger=ledger,
        backup_remote=backup_remote,
        anchor_spec=fake_spec(tmp_path, public_remote.as_posix()),
        backup_spec=fake_spec(tmp_path, backup_remote.as_posix()),
    )


def _failing_anchor_push(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """Real git, except that the anchor push fails as an unreachable network would."""
    if args and args[0] == "push" and args[-1] == f"{ANCHOR_REF}:{ANCHOR_REF}":
        return subprocess.CompletedProcess(
            list(args), 128, b"", b"simulated: could not resolve host"
        )
    return run_git(args, cwd=cwd, env=env, timeout=timeout, input_bytes=input_bytes)


def test_sync_no_head_move_no_anchor_commit(world: World) -> None:
    empty = world.sync()
    assert empty.anchor_committed is False
    assert world.anchor_commits() == 0

    append_rows(world.ledger, week_rows(6, ("KC_BUF",)))
    first = world.sync()
    assert first.anchor_committed is True
    anchored = ref_sha(world.repo, ANCHOR_REF)

    again = world.sync()

    assert again.anchor_committed is False
    assert again.anchor_pushed is False
    assert again.backup_committed is False
    assert ref_sha(world.repo, ANCHOR_REF) == anchored
    assert world.anchor_commits() == 1


def test_sync_head_move_one_anchor_commit_and_push(world: World) -> None:
    append_rows(world.ledger, week_rows(6, ("KC_BUF", "DAL_PHI")))
    head_hash, entry_count = world.head()

    outcome = world.sync()

    assert outcome.anchor_committed is True
    assert outcome.anchor_pushed is True
    assert outcome.anchor_error is None
    assert outcome.backup_committed is True
    assert outcome.backup_pushed is True
    assert outcome.backup_error is None
    assert outcome.refused_reason is None
    assert world.anchor_commits() == 1
    assert ref_sha(world.public_remote, ANCHOR_REF) == ref_sha(world.repo, ANCHOR_REF)
    assert read_local_anchor(world.repo) == (head_hash, entry_count)
    assert ref_sha(world.backup_remote, "refs/heads/main") == git_out(
        world.ledger, "rev-parse", "HEAD"
    )
    assert [event for event, _ in world.events] == [
        "anchor_commit",
        "anchor_push",
        "backup_commit",
        "backup_push",
    ]
    for _, fields in world.events:
        assert fields["head_hash"] == head_hash
        assert fields["entry_count"] == entry_count
        assert fields["ok"] is True


def test_sync_correction_only_moves_anchor(world: World) -> None:
    row = make_row()
    append_rows(world.ledger, [row])
    apply_updates(
        world.ledger,
        grading_updates={
            key_of(row): graded("win", 0.9090909090909091, 1.1363636363636365)
        },
    )
    world.sync()
    correction = {
        "game_id": row["game_id"],
        "season": 2026,
        "week": 6,
        "target": "ats",
        "arm": "live",
        "original_grading_status": "win",
        "original_payout_flat": 0.9090909090909091,
        "prior_grading_status": "win",
        "prior_payout_flat": 0.9090909090909091,
        "prior_realized_units": 1.1363636363636365,
        "prior_correction_seq": None,
        "corrected_grading_status": "loss",
        "corrected_outcome": False,
        "corrected_payout_flat": -1.0,
        "corrected_realized_units": -1.25,
        "realized_value": -4.0,
        "source_dataset": "schedules",
        "source_season": 2026,
        "source_week": 6,
        "source_sequence": 4,
        "detected_at_utc": "2026-10-21T21:00:00+00:00",
        "corrected_at_utc": "2026-10-21T21:00:05+00:00",
    }
    commit_changes(world.ledger, new_corrections=[correction])

    outcome = world.sync()

    assert outcome.anchor_committed is True
    assert world.anchor_commits() == 2
    assert read_local_anchor(world.repo) == world.head()
    assert world.head()[1] == 2


def test_sync_grading_only_backs_up_without_anchor(world: World) -> None:
    row = make_row()
    append_rows(world.ledger, [row])
    world.sync()
    backup_before = git_out(world.ledger, "rev-parse", "HEAD")
    apply_updates(
        world.ledger,
        grading_updates={key_of(row): graded("loss", -1.0, -1.25)},
    )

    outcome = world.sync()

    assert outcome.anchor_committed is False
    assert outcome.anchor_pushed is False
    assert outcome.backup_committed is True
    assert outcome.backup_pushed is True
    assert world.anchor_commits() == 1
    backup_after = git_out(world.ledger, "rev-parse", "HEAD")
    assert backup_after != backup_before
    assert ref_sha(world.backup_remote, "refs/heads/main") == backup_after


def test_sync_failed_push_recorded_not_raised(world: World) -> None:
    append_rows(world.ledger, week_rows(6, ("KC_BUF",)))

    failed = world.sync(runner=_failing_anchor_push)

    assert failed.anchor_committed is True
    assert failed.anchor_pushed is False
    assert failed.anchor_error is not None
    assert "simulated" in failed.anchor_error
    assert ref_sha(world.public_remote, ANCHOR_REF) is None
    pushes = [fields for event, fields in world.events if event == "anchor_push"]
    assert len(pushes) == 1
    assert pushes[0]["ok"] is False
    assert "simulated" in pushes[0]["error"]
    assert failed.backup_pushed is True

    append_rows(world.ledger, week_rows(7, ("KC_BUF",)))
    retried = world.sync()

    assert retried.anchor_pushed is True
    assert ref_sha(world.public_remote, ANCHOR_REF) == ref_sha(world.repo, ANCHOR_REF)
    assert (
        read_remote_anchor(world.repo, url=world.public_remote.as_posix())
        == world.head()
    )


def test_sync_refuses_broken_chain(world: World) -> None:
    append_rows(world.ledger, week_rows(6, ("KC_BUF", "DAL_PHI")))
    world.sync()
    anchored = ref_sha(world.repo, ANCHOR_REF)
    backup_remote_main = ref_sha(world.backup_remote, "refs/heads/main")
    path = ledger_path(world.ledger)
    path.write_bytes(path.read_bytes().replace(b"DAL_PHI", b"DAL_NYG", 1))
    world.events.clear()

    outcome = world.sync()

    assert outcome.refused_reason is not None
    assert outcome.anchor_committed is False
    assert outcome.anchor_pushed is False
    assert outcome.backup_committed is False
    assert outcome.backup_pushed is False
    assert ref_sha(world.repo, ANCHOR_REF) == anchored
    assert ref_sha(world.backup_remote, "refs/heads/main") == backup_remote_main
    assert [event for event, _ in world.events] == ["sync_refused"]
    refused = world.events[0][1]
    assert refused["entry_count"] == 2
    assert "head_hash" in refused


def test_anchor_blob_has_no_pick_data(world: World) -> None:
    rows = week_rows(6, ("KC_BUF", "DAL_PHI"))
    append_rows(world.ledger, rows)

    world.sync()

    blob = git(
        world.repo, "cat-file", "blob", f"{ANCHOR_REF}:{ANCHOR_FILE_NAME}"
    ).stdout
    head_hash, entry_count = world.head()
    assert blob == f"head_hash={head_hash}\nrow_count={entry_count}\n"
    for row in rows:
        for token in (
            row["game_id"],
            "KC",
            "BUF",
            "DAL",
            "PHI",
            row["bet_side"],
            row["target"],
            repr(row["line"]),
            repr(row["stake_units"]),
            repr(row["selected_odds"]),
        ):
            assert token not in blob


def test_master_untouched(world: World) -> None:
    remote_master = ref_sha(world.public_remote, "refs/heads/master")
    local_master = ref_sha(world.repo, "refs/heads/master")
    append_rows(world.ledger, week_rows(6, ("KC_BUF",)))

    world.sync(runner=_failing_anchor_push)
    world.sync()
    append_rows(world.ledger, week_rows(7, ("KC_BUF",)))
    world.sync()

    assert ref_sha(world.public_remote, "refs/heads/master") == remote_master
    assert ref_sha(world.repo, "refs/heads/master") == local_master
    assert git_out(world.repo, "symbolic-ref", "HEAD") == "refs/heads/master"
    assert git_out(world.repo, "status", "--porcelain") == ""


def _raising_runner(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    raise OSError("git is not installed")


def _refusing_runner(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    return subprocess.CompletedProcess(list(args), 128, b"", b"fatal: simulated")


@pytest.mark.parametrize("runner", [_raising_runner, _refusing_runner])
def test_sync_never_raises_when_every_git_call_fails(
    world: World, runner: GitRunner
) -> None:
    append_rows(world.ledger, week_rows(6, ("KC_BUF",)))
    head_hash, entry_count = world.head()

    outcome = world.sync(runner=runner)

    assert outcome.anchor_committed is False
    assert outcome.anchor_pushed is False
    assert outcome.anchor_error is not None
    assert outcome.backup_committed is False
    assert outcome.backup_pushed is False
    assert outcome.backup_error is not None
    assert outcome.refused_reason is None
    assert world.events
    for event, fields in world.events:
        assert event in SYNC_EVENTS
        assert fields["ok"] is False
        assert fields["head_hash"] == head_hash
        assert fields["entry_count"] == entry_count


def test_sync_missing_backup_repo_is_recorded_not_created(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "public")
    ledger = tmp_path / "ledger"
    append_rows(ledger, week_rows(6, ("KC_BUF",)))
    events: list[tuple[str, dict[str, Any]]] = []

    outcome = publish_ledger_state(
        ledger_dir=ledger,
        repo_dir=repo,
        anchor_spec=fake_spec(tmp_path, make_bare(tmp_path / "public.git").as_posix()),
        backup_spec=fake_spec(tmp_path, make_bare(tmp_path / "backup.git").as_posix()),
        log=lambda event, **fields: events.append((event, fields)),
        now=NOW,
    )

    assert outcome.backup_error == "backup_repo_missing"
    assert outcome.backup_committed is False
    assert not (ledger / ".git").exists()
    backup_events = [fields for event, fields in events if event == "backup_push"]
    assert backup_events == [
        {
            "ok": False,
            "reason": "backup_repo_missing",
            "head_hash": ledger_head(read_entries(ledger))[0],
            "entry_count": 1,
        }
    ]
