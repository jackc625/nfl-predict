"""The private backup: ``ledger/`` as its own git repository, pushed to the private repo (Plan 34-06).

D-01/D-02: a second copy of the whole ledger lives in a private repository and must restore,
verify and replay on its own. The backup is a nested git repository in the ledger directory whose
FIRST commit is ``.gitattributes`` (``* -text``) -- ``core.autocrlf=true`` is set system-wide on
this machine, and without it a restore would rewrite every LF to CRLF -- plus a ``.gitignore``
that keeps the writer lock and temp files out of history.

Real git only, against ``tmp_path`` repositories and local bare "remotes"; nothing reaches GitHub.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

from forward_ledger.backup import commit_backup, ensure_backup_repo, push_backup
from forward_ledger.remote_config import BACKUP_PUSHED_REF, BACKUP_REPO_SLUG
from forward_ledger.store import (
    LEDGER_FILENAME,
    LEDGER_WRITER_LOCK_NAME,
    append_rows,
    ledger_path,
    read_entries,
    verify_chain,
)
from tests.unit.test_forward_ledger_store import week_rows
from tests.unit.test_ledger_anchor import (
    SpyRunner,
    fake_spec,
    git,
    git_out,
    make_bare,
    ref_sha,
)


def _ledger_with_rows(tmp_path: Path) -> Path:
    ledger = tmp_path / "ledger"
    ensure_backup_repo(ledger)
    append_rows(ledger, week_rows(6, ("KC_BUF", "DAL_PHI")))
    return ledger


def test_backup_repo_init_commits_gitattributes_first(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    ledger.mkdir()
    (ledger / LEDGER_FILENAME).write_bytes(b"already here before the backup existed\n")

    assert ensure_backup_repo(ledger) is True

    assert git_out(ledger, "symbolic-ref", "HEAD") == "refs/heads/main"
    assert git_out(ledger, "rev-list", "--count", "HEAD") == "1"
    first = git_out(ledger, "ls-tree", "--name-only", "HEAD").splitlines()
    assert sorted(first) == [".gitattributes", ".gitignore", "README.md"]
    assert git(ledger, "cat-file", "blob", "HEAD:.gitattributes").stdout == "* -text\n"
    ignored = git(ledger, "cat-file", "blob", "HEAD:.gitignore").stdout.splitlines()
    assert "*.lock" in ignored
    assert "*.tmp" in ignored
    readme = git(ledger, "cat-file", "blob", "HEAD:README.md").stdout
    assert BACKUP_REPO_SLUG in readme
    assert "docs/guides/RUNBOOK.md" in readme

    assert ensure_backup_repo(ledger) is False
    assert git_out(ledger, "rev-list", "--count", "HEAD") == "1"


def test_backup_commit_only_when_dirty(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger"
    ensure_backup_repo(ledger)
    assert commit_backup(ledger, message="nothing changed") is None

    append_rows(ledger, week_rows(6, ("KC_BUF",)))
    (ledger / LEDGER_WRITER_LOCK_NAME).write_bytes(b"")
    (ledger / "forward_2026.jsonl.tmp").write_bytes(b"partial")

    sha = commit_backup(ledger, message="ledger: 1 entries")

    assert sha == git_out(ledger, "rev-parse", "HEAD")
    assert git_out(ledger, "rev-list", "--count", "HEAD") == "2"
    tracked = git_out(ledger, "ls-files").splitlines()
    assert LEDGER_FILENAME in tracked
    assert LEDGER_WRITER_LOCK_NAME not in tracked
    assert "forward_2026.jsonl.tmp" not in tracked
    assert commit_backup(ledger, message="still nothing") is None
    assert git_out(ledger, "rev-list", "--count", "HEAD") == "2"


def test_backup_push_never_forces(tmp_path: Path) -> None:
    ledger = _ledger_with_rows(tmp_path)
    commit_backup(ledger, message="ledger: 2 entries")
    remote = make_bare(tmp_path / "backup.git")
    spy = SpyRunner()

    outcome = push_backup(
        ledger, spec=fake_spec(tmp_path, remote.as_posix()), runner=spy
    )

    assert outcome.ok, outcome.error
    pushes = [call for call in spy.calls if call[0] == "push"]
    assert len(pushes) == 1
    assert [arg for arg in pushes[0] if arg.startswith("-")] == ["--porcelain"]
    assert not any(arg.startswith("+") for arg in pushes[0])
    assert pushes[0][-1] == "HEAD:refs/heads/main"
    local_head = git_out(ledger, "rev-parse", "HEAD")
    assert ref_sha(remote, "refs/heads/main") == local_head
    assert ref_sha(ledger, BACKUP_PUSHED_REF) == local_head


def test_restore_round_trip_is_byte_identical(tmp_path: Path) -> None:
    ledger = _ledger_with_rows(tmp_path)
    commit_backup(ledger, message="ledger: 2 entries")
    remote = make_bare(tmp_path / "backup.git")
    assert push_backup(ledger, spec=fake_spec(tmp_path, remote.as_posix())).ok
    original = ledger_path(ledger).read_bytes()
    assert b"\r\n" not in original

    clone = git(tmp_path, "clone", "--branch", "main", remote.as_posix(), "restored")
    assert clone.returncode == 0, clone.stderr

    restored = tmp_path / "restored"
    assert ledger_path(restored).read_bytes() == original
    assert verify_chain(read_entries(restored)).ok
