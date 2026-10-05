"""The ledger anchor: a plumbing commit on its own branch, pushed with its own deploy key (Plan 34-06).

LDGR-05 needs the chain's head off the machine, because a chain alone cannot detect truncation. The
anchor is one commit on ``refs/heads/ledger-anchor`` whose tree holds exactly one file, ``ANCHOR``,
with exactly two lines -- the head hash and the row count. These tests prove, with REAL git:

* the commit is written by plumbing, so HEAD, the index and the working tree are untouched (D-12);
* the content is exactly two lines, and anything else is refused before a blob is written;
* the ref moves by compare-and-swap, so a concurrent move fails instead of being overwritten;
* the push names one refspec, ``ledger-anchor`` to ``ledger-anchor``, and never forces;
* every push pins the deploy key, and a missing key is refused before git runs (D-03).

Every repository here is built under ``tmp_path``, and every "remote" is a local bare repository.
Nothing touches GitHub, the project's own refs, or the owner's ssh directory: the key and the
known_hosts file are empty placeholder files in ``tmp_path`` (a local path remote never runs ssh).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from forward_ledger import anchor
from forward_ledger.anchor import (
    AnchorRefspecError,
    AnchorWriteError,
    push_anchor,
    read_local_anchor,
    read_remote_anchor,
    write_anchor_commit,
)
from forward_ledger.remote_config import (
    ANCHOR_FILE_NAME,
    ANCHOR_PUSHED_REF,
    ANCHOR_REF,
    SSH_EXECUTABLE,
    RemoteSpec,
)
from forward_ledger.transport import MissingDeployKeyError, git_push_env, run_git

HEAD_ONE = "1" * 64
HEAD_TWO = "abcdef0123456789" * 4

_TEST_IDENTITY = (
    "-c",
    "user.name=Ledger Test",
    "-c",
    "user.email=ledger@example.invalid",
)


# ---------------------------------------------------------------------------
# Real-git helpers, all confined to tmp_path
# ---------------------------------------------------------------------------


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run one git command in *cwd*, never raising on a non-zero exit."""
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True, check=False
    )


def git_out(cwd: Path, *args: str) -> str:
    result = git(cwd, *args)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def ref_sha(cwd: Path, ref: str) -> str | None:
    result = git(cwd, "rev-parse", "-q", "--verify", ref)
    return result.stdout.strip() if result.returncode == 0 else None


def make_repo(path: Path) -> Path:
    """A non-bare repository on ``master`` with one commit."""
    path.mkdir(parents=True)
    git_out(path, "init", "-b", "master")
    (path / "README.md").write_bytes(b"public repo\n")
    git_out(path, "add", "README.md")
    git_out(path, *_TEST_IDENTITY, "commit", "-m", "initial")
    return path


def make_bare(path: Path) -> Path:
    git_out(path.parent, "init", "--bare", path.name)
    return path


def fake_spec(tmp_path: Path, url: str) -> RemoteSpec:
    """A spec whose key and known_hosts exist as placeholders; a local URL never runs ssh."""
    key = tmp_path / "fake_deploy_key"
    known_hosts = tmp_path / "fake_known_hosts"
    key.write_bytes(b"placeholder, not a key\n")
    known_hosts.write_bytes(b"placeholder\n")
    return RemoteSpec(url=url, key_path=key, known_hosts_path=known_hosts)


class SpyRunner:
    """Records every git call and forwards it to the real runner."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def __call__(
        self,
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        self.calls.append(list(args))
        return run_git(args, cwd=cwd, env=env, timeout=timeout, input_bytes=input_bytes)


# ---------------------------------------------------------------------------
# The plumbing commit
# ---------------------------------------------------------------------------


def test_anchor_commit_leaves_worktree_and_head(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "public")
    master_before = ref_sha(repo, "refs/heads/master")

    write_anchor_commit(repo, HEAD_ONE, 3)

    assert git_out(repo, "symbolic-ref", "HEAD") == "refs/heads/master"
    assert ref_sha(repo, "refs/heads/master") == master_before
    assert git_out(repo, "status", "--porcelain") == ""
    assert git(repo, "diff", "--cached", "--quiet").returncode == 0


def test_anchor_content_is_exactly_two_lines(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "public")

    result = write_anchor_commit(repo, HEAD_ONE, 3)

    blob = git(repo, "cat-file", "blob", f"{ANCHOR_REF}:{ANCHOR_FILE_NAME}")
    assert blob.stdout == f"head_hash={HEAD_ONE}\nrow_count=3\n"
    listing = git_out(repo, "ls-tree", "--name-only", ANCHOR_REF).splitlines()
    assert listing == [ANCHOR_FILE_NAME]
    assert result.sha == ref_sha(repo, ANCHOR_REF)
    assert result.parent is None
    assert read_local_anchor(repo) == (HEAD_ONE, 3)


@pytest.mark.parametrize(
    ("head_hash", "row_count"),
    [
        ("1" * 63, 3),
        ("1" * 65, 3),
        ("A" * 64, 3),
        ("g" * 64, 3),
        ("1" * 63 + "\n", 3),
        (HEAD_ONE, -1),
        (HEAD_ONE, True),
        (HEAD_ONE, 3.0),
        (HEAD_ONE, "3"),
    ],
)
def test_anchor_refuses_non_head_content(
    tmp_path: Path, head_hash: object, row_count: object
) -> None:
    repo = make_repo(tmp_path / "public")
    spy = SpyRunner()

    with pytest.raises(AnchorWriteError):
        write_anchor_commit(repo, head_hash, row_count, runner=spy)  # type: ignore[arg-type]

    assert spy.calls == []
    assert ref_sha(repo, ANCHOR_REF) is None


def test_second_anchor_chains_to_first(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "public")

    first = write_anchor_commit(repo, HEAD_ONE, 3)
    second = write_anchor_commit(repo, HEAD_TWO, 5)

    assert second.parent == first.sha
    assert git_out(repo, "rev-parse", f"{ANCHOR_REF}^") == first.sha
    assert read_local_anchor(repo) == (HEAD_TWO, 5)


def test_stale_parent_cas_fails(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "public")
    first = write_anchor_commit(repo, HEAD_ONE, 3)
    concurrent = git_out(repo, "rev-parse", "refs/heads/master")

    def racing_runner(
        args: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
        input_bytes: bytes | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        if args and args[0] == "update-ref":
            # Another writer moves the anchor ref between our read of the parent and our CAS.
            git_out(repo, "update-ref", ANCHOR_REF, concurrent, first.sha)
        return run_git(args, cwd=cwd, env=env, timeout=timeout, input_bytes=input_bytes)

    with pytest.raises(AnchorWriteError):
        write_anchor_commit(repo, HEAD_TWO, 5, runner=racing_runner)

    assert ref_sha(repo, ANCHOR_REF) == concurrent


# ---------------------------------------------------------------------------
# The push: one refspec, never forced, deploy key pinned
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "push_args",
    [
        ["push", "--porcelain", "URL", "refs/heads/ledger-anchor:refs/heads/master"],
        ["push", "--porcelain", "URL", "refs/heads/ledger-anchor:master"],
        ["push", "--porcelain", "URL", "refs/heads/ledger-anchor:HEAD"],
        ["push", "--porcelain", "URL", "HEAD:refs/heads/ledger-anchor"],
        [
            "push",
            "--porcelain",
            "URL",
            "+refs/heads/ledger-anchor:refs/heads/ledger-anchor",
        ],
        ["push", "--force", "URL", "refs/heads/ledger-anchor:refs/heads/ledger-anchor"],
        ["push", "-f", "URL", "refs/heads/ledger-anchor:refs/heads/ledger-anchor"],
        [
            "push",
            "--force-with-lease",
            "URL",
            "refs/heads/ledger-anchor:refs/heads/ledger-anchor",
        ],
        ["push", "--mirror", "URL"],
        [
            "push",
            "URL",
            "refs/heads/ledger-anchor:refs/heads/ledger-anchor",
            "refs/heads/master:refs/heads/master",
        ],
        ["push", "URL"],
    ],
)
def test_refspec_guard_refuses_master(push_args: list[str]) -> None:
    with pytest.raises(AnchorRefspecError):
        anchor._assert_anchor_refspec(push_args)


def test_refspec_guard_accepts_the_one_anchor_refspec() -> None:
    anchor._assert_anchor_refspec(
        [
            "push",
            "--porcelain",
            "URL",
            "refs/heads/ledger-anchor:refs/heads/ledger-anchor",
        ]
    )


def test_push_to_local_bare_remote(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "public")
    remote = make_bare(tmp_path / "remote.git")
    git_out(repo, "push", str(remote), "refs/heads/master:refs/heads/master")
    remote_master_before = ref_sha(remote, "refs/heads/master")
    written = write_anchor_commit(repo, HEAD_ONE, 3)
    spy = SpyRunner()

    outcome = push_anchor(repo, spec=fake_spec(tmp_path, remote.as_posix()), runner=spy)

    assert outcome.ok, outcome.error
    assert outcome.pushed_sha == written.sha
    assert ref_sha(remote, ANCHOR_REF) == written.sha
    assert ref_sha(remote, "refs/heads/master") == remote_master_before
    assert ref_sha(repo, ANCHOR_PUSHED_REF) == written.sha
    pushes = [call for call in spy.calls if call[0] == "push"]
    assert len(pushes) == 1
    assert not any(arg.startswith("-f") or "force" in arg for arg in pushes[0])
    assert pushes[0][-1] == "refs/heads/ledger-anchor:refs/heads/ledger-anchor"
    assert read_remote_anchor(repo, url=remote.as_posix()) == (HEAD_ONE, 3)


def test_transport_env_pins_the_deploy_key(tmp_path: Path) -> None:
    spec = fake_spec(tmp_path, "git@github.com:owner/repo.git")
    base = {"PATH": "x", "GIT_SSH": "plink.exe", "SSH_AUTH_SOCK": "agent.sock"}

    env = git_push_env(spec, base_env=base)

    command = env["GIT_SSH_COMMAND"]
    assert command.startswith(SSH_EXECUTABLE + " ")
    assert Path(SSH_EXECUTABLE).is_absolute()
    for fragment in (
        "-F none",
        f"-i {spec.key_path.as_posix()}",
        "-o IdentitiesOnly=yes",
        "-o IdentityAgent=none",
        "-o BatchMode=yes",
        "-o StrictHostKeyChecking=yes",
        f"-o UserKnownHostsFile={spec.known_hosts_path.as_posix()}",
    ):
        assert fragment in command
    assert env["GIT_TERMINAL_PROMPT"] == "0"
    assert "GIT_SSH" not in env
    assert "SSH_AUTH_SOCK" not in env
    assert env["PATH"] == "x"
    for name in (
        "GIT_AUTHOR_NAME",
        "GIT_AUTHOR_EMAIL",
        "GIT_COMMITTER_NAME",
        "GIT_COMMITTER_EMAIL",
    ):
        assert env[name]
    assert base == {"PATH": "x", "GIT_SSH": "plink.exe", "SSH_AUTH_SOCK": "agent.sock"}


@pytest.mark.parametrize("missing", ["key", "known_hosts"])
def test_missing_key_refused_before_git(tmp_path: Path, missing: str) -> None:
    repo = make_repo(tmp_path / "public")
    write_anchor_commit(repo, HEAD_ONE, 3)
    spec = fake_spec(tmp_path, (tmp_path / "remote.git").as_posix())
    (spec.key_path if missing == "key" else spec.known_hosts_path).unlink()
    spy = SpyRunner()

    with pytest.raises(MissingDeployKeyError):
        push_anchor(repo, spec=spec, runner=spy)

    assert spy.calls == []


def test_read_remote_anchor_absent_branch(tmp_path: Path) -> None:
    repo = make_repo(tmp_path / "public")
    remote = make_bare(tmp_path / "remote.git")
    git_out(repo, "push", str(remote), "refs/heads/master:refs/heads/master")

    assert read_remote_anchor(repo, url=remote.as_posix()) is None
