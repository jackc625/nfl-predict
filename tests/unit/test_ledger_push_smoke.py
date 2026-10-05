"""The S4U push smoke (Plan 34-18, D-03, D-04): offline, through a fake runner.

``scripts/ledger_push_smoke.py`` proves, before go-live, that the unattended S4U task can push
with each deploy key and is greeted as the KEY (the repository), not as the owner. Every external
command it makes -- ``whoami``, ``ssh``, ``git``, ``gh``, ``schtasks`` -- goes through the
module-level ``RUNNER`` seam. These tests replace it with :class:`FakeRunner`, which answers from
canned routes and FAILS on any command it was not told about, so nothing here reaches GitHub,
pushes anything or touches the real Task Scheduler. Key and known_hosts files are placeholder
``tmp_path`` files (never real keys).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import subprocess
import xml.etree.ElementTree as ET
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pytest

from forward_ledger.remote_config import (
    ANCHOR_REMOTE_URL,
    BACKUP_REMOTE_URL,
    SSH_EXECUTABLE,
    RemoteSpec,
)
from scripts import ledger_push_smoke as smoke

DAILY_TEMPLATE = Path("deployment/windows_scheduler.xml")
TASK_NS = "{http://schemas.microsoft.com/windows/2004/02/mit/task}"

BLOB_SHA = "b" * 40
TREE_SHA = "c" * 40
SMOKE_COMMIT = "d" * 40
LEDGER_HEAD = "e" * 40
MASTER_SHA = "f" * 40

ANCHOR_GREETING = (
    "Hi jackc625/nfl-predict! You've successfully authenticated, but GitHub does not "
    "provide shell access.\r\n"
)
BACKUP_GREETING = (
    "Hi jackc625/nfl-predict-ledger! You've successfully authenticated, but GitHub does "
    "not provide shell access.\r\n"
)
USER_GREETING = (
    "Hi jackc625! You've successfully authenticated, but GitHub does not provide shell "
    "access.\r\n"
)

Handler = Callable[[list[str], dict[str, Any]], subprocess.CompletedProcess[bytes]]


def done(
    argv: Sequence[str], out: str | bytes = b"", code: int = 0, err: str | bytes = b""
) -> subprocess.CompletedProcess[bytes]:
    """A completed process with byte output."""
    out_bytes = out.encode("ascii") if isinstance(out, str) else out
    err_bytes = err.encode("ascii") if isinstance(err, str) else err
    return subprocess.CompletedProcess(list(argv), code, out_bytes, err_bytes)


class FakeRunner:
    """Answers commands by argv prefix; an unknown command fails the test.

    Routes added later take precedence, so a test overrides one answer of a good world.
    """

    def __init__(self) -> None:
        self.routes: list[tuple[tuple[str, ...], Handler]] = []
        self.calls: list[tuple[list[str], dict[str, Any]]] = []

    def route(
        self,
        prefix: Sequence[str],
        answer: Handler | subprocess.CompletedProcess[bytes],
    ) -> None:
        if isinstance(answer, subprocess.CompletedProcess):
            fixed = answer
            self.routes.insert(0, (tuple(prefix), lambda argv, kw: fixed))
        else:
            self.routes.insert(0, (tuple(prefix), answer))

    def __call__(
        self, argv: Sequence[str], **kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        command = list(argv)
        self.calls.append((command, kwargs))
        for prefix, handler in self.routes:
            if tuple(command[: len(prefix)]) == prefix:
                return handler(command, kwargs)
        pytest.fail(f"unexpected command: {command}")

    def argvs(self) -> list[list[str]]:
        return [argv for argv, _kwargs in self.calls]

    def calls_starting(self, *prefix: str) -> list[tuple[list[str], dict[str, Any]]]:
        return [
            (argv, kw)
            for argv, kw in self.calls
            if tuple(argv[: len(prefix)]) == prefix
        ]


@pytest.fixture()
def paths(tmp_path: Path) -> smoke.SmokePaths:
    """Smoke paths under tmp_path, with placeholder key and known_hosts files."""
    ssh_dir = tmp_path / "ssh"
    ssh_dir.mkdir()
    anchor_key = ssh_dir / "anchor_key"
    backup_key = ssh_dir / "backup_key"
    known_hosts = ssh_dir / "known_hosts"
    for placeholder in (anchor_key, backup_key, known_hosts):
        placeholder.write_text("placeholder, not a key\n", encoding="ascii")
    repo_dir = tmp_path / "repo"
    ledger_dir = repo_dir / "ledger"
    ledger_dir.mkdir(parents=True)
    return smoke.SmokePaths(
        repo_dir=repo_dir,
        ledger_dir=ledger_dir,
        result_path=repo_dir / "logs" / "ledger_push_smoke.json",
        task_xml_path=repo_dir / "logs" / "ledger_smoke_task.xml",
        dummy_xml_path=repo_dir / "logs" / "ledger_smoke_dummy_task.xml",
        template_path=DAILY_TEMPLATE,
        anchor_spec=RemoteSpec(
            url=ANCHOR_REMOTE_URL, key_path=anchor_key, known_hosts_path=known_hosts
        ),
        backup_spec=RemoteSpec(
            url=BACKUP_REMOTE_URL, key_path=backup_key, known_hosts_path=known_hosts
        ),
    )


@pytest.fixture()
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeRunner:
    runner = FakeRunner()
    monkeypatch.setattr(smoke, "RUNNER", runner)
    return runner


# ---------------------------------------------------------------------------------------------
# 1. Greeting classification -- identity is read from the greeting, not the exit status
# ---------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exit_code", "output", "expected_slug", "identity_ok", "kind"),
    [
        # GitHub's ssh -T exits 1 on success (no shell): the repository greeting is the proof.
        (1, ANCHOR_GREETING, "jackc625/nfl-predict", True, "deploy_key"),
        (1, BACKUP_GREETING, "jackc625/nfl-predict-ledger", True, "deploy_key"),
        # The owner's USER key: authenticated, but as the account -- the identity is wrong.
        (1, USER_GREETING, "jackc625/nfl-predict", False, "user"),
        # A deploy key, but for the OTHER repository.
        (1, BACKUP_GREETING, "jackc625/nfl-predict", False, "deploy_key"),
        # ssh itself failed: no greeting at all.
        (
            255,
            "git@github.com: Permission denied (publickey).\r\n",
            "jackc625/nfl-predict",
            False,
            None,
        ),
        (0, "", "jackc625/nfl-predict", False, None),
    ],
)
def test_greeting_classification(
    exit_code: int, output: str, expected_slug: str, identity_ok: bool, kind: str | None
) -> None:
    verdict = smoke.classify_greeting(exit_code, output, expected_slug)
    assert verdict["identity_ok"] is identity_ok
    assert verdict["identity_kind"] == kind
    assert verdict["exit_code"] == exit_code
    if not identity_ok:
        assert verdict["error"], "a failed identity check must carry the ssh output"


def test_greeting_exit_255_is_a_failure_even_with_a_greeting() -> None:
    verdict = smoke.classify_greeting(255, ANCHOR_GREETING, "jackc625/nfl-predict")
    assert verdict["identity_ok"] is False


# ---------------------------------------------------------------------------------------------
# 2. The smoke push can only carry the throwaway smoke ref
# ---------------------------------------------------------------------------------------------


def test_smoke_refspec_guard_accepts_only_the_smoke_ref() -> None:
    smoke.check_smoke_push_args(
        ["push", "--porcelain", ANCHOR_REMOTE_URL, smoke.SMOKE_REFSPEC]
    )
    assert (
        smoke.SMOKE_REFSPEC
        == "refs/heads/ledger-anchor-smoke:refs/heads/ledger-anchor-smoke"
    )


@pytest.mark.parametrize(
    "push_args",
    [
        [
            "push",
            "--porcelain",
            ANCHOR_REMOTE_URL,
            "refs/heads/master:refs/heads/master",
        ],
        ["push", "--porcelain", ANCHOR_REMOTE_URL, "HEAD:refs/heads/master"],
        [
            "push",
            "--porcelain",
            ANCHOR_REMOTE_URL,
            "refs/heads/ledger-anchor:refs/heads/ledger-anchor",
        ],
        [
            "push",
            "--porcelain",
            ANCHOR_REMOTE_URL,
            "+refs/heads/ledger-anchor-smoke:refs/heads/ledger-anchor-smoke",
        ],
        [
            "push",
            "--porcelain",
            ANCHOR_REMOTE_URL,
            "refs/heads/ledger-anchor-smoke:refs/heads/master",
        ],
        ["push", "--force", ANCHOR_REMOTE_URL, smoke.SMOKE_REFSPEC],
        ["push", "--porcelain", BACKUP_REMOTE_URL, smoke.SMOKE_REFSPEC],
        [
            "push",
            "--porcelain",
            ANCHOR_REMOTE_URL,
            smoke.SMOKE_REFSPEC,
            "refs/heads/master:refs/heads/master",
        ],
    ],
)
def test_smoke_refspec_guard(push_args: list[str]) -> None:
    with pytest.raises(smoke.SmokeRefspecError):
        smoke.check_smoke_push_args(push_args)


# ---------------------------------------------------------------------------------------------
# 3. --run writes the result file and never raises
# ---------------------------------------------------------------------------------------------


def _good_run_world(fake: FakeRunner, paths: smoke.SmokePaths) -> None:
    def ssh(argv: list[str], _kw: dict[str, Any]) -> subprocess.CompletedProcess[bytes]:
        key = argv[argv.index("-i") + 1]
        if key == paths.anchor_spec.key_path.as_posix():
            return done(argv, code=1, err=ANCHOR_GREETING)
        return done(argv, code=1, err=BACKUP_GREETING)

    def rev_parse(
        argv: list[str], kw: dict[str, Any]
    ) -> subprocess.CompletedProcess[bytes]:
        if Path(kw["cwd"]) == paths.ledger_dir:
            return done(argv, LEDGER_HEAD + "\n")
        return done(argv, code=1)  # no local smoke ref yet

    def query(
        argv: list[str], _kw: dict[str, Any]
    ) -> subprocess.CompletedProcess[bytes]:
        return done(argv, paths.dummy_xml_path.read_bytes())

    fake.route(["whoami"], done(["whoami"], "jackslaptop\\jackc\r\n"))
    fake.route([SSH_EXECUTABLE], ssh)
    fake.route(["git", "hash-object"], done([], BLOB_SHA + "\n"))
    fake.route(["git", "mktree"], done([], TREE_SHA + "\n"))
    fake.route(["git", "rev-parse"], rev_parse)
    fake.route(["git", "commit-tree"], done([], SMOKE_COMMIT + "\n"))
    fake.route(["git", "update-ref"], done([]))
    fake.route(["git", "push"], done([]))
    fake.route(["schtasks", "/create"], done([]))
    fake.route(["schtasks", "/query"], query)
    fake.route(["schtasks", "/delete"], done([]))


def test_run_writes_result_json(fake: FakeRunner, paths: smoke.SmokePaths) -> None:
    _good_run_world(fake, paths)

    assert smoke.main(["--run"], paths=paths) == 0

    result = json.loads(paths.result_path.read_text(encoding="ascii"))
    assert result["whoami"] == "jackslaptop\\jackc"
    assert result["greetings"]["anchor"]["identity_ok"] is True
    assert result["greetings"]["backup"]["identity_ok"] is True
    assert result["anchor_push"]["ok"] is True
    assert result["anchor_push"]["smoke_commit"] == SMOKE_COMMIT
    assert result["backup_push"]["ok"] is True
    assert result["backup_push"]["pushed_sha"] == LEDGER_HEAD
    assert result["dummy_task"]["ok"] is True
    assert result["errors"] == {}

    # The smoke commit holds only a timestamp.
    [(_argv, hash_kw)] = fake.calls_starting("git", "hash-object")
    assert hash_kw["input_bytes"].startswith(b"smoke_at=")

    # The anchor push carries exactly the smoke refspec, with the anchor key pinned.
    [(anchor_push, anchor_kw)] = [
        (argv, kw)
        for argv, kw in fake.calls_starting("git", "push")
        if ANCHOR_REMOTE_URL in argv
    ]
    assert anchor_push == [
        "git",
        "push",
        "--porcelain",
        ANCHOR_REMOTE_URL,
        smoke.SMOKE_REFSPEC,
    ]
    assert paths.anchor_spec.key_path.as_posix() in anchor_kw["env"]["GIT_SSH_COMMAND"]

    # The backup push carries HEAD onto main of the private repo, with the backup key.
    [(backup_push, backup_kw)] = [
        (argv, kw)
        for argv, kw in fake.calls_starting("git", "push")
        if BACKUP_REMOTE_URL in argv
    ]
    assert backup_push[-1] == "HEAD:refs/heads/main"
    assert paths.backup_spec.key_path.as_posix() in backup_kw["env"]["GIT_SSH_COMMAND"]

    # Nothing pushed master or ledger-anchor.
    for argv in fake.argvs():
        if argv[:2] == ["git", "push"]:
            assert not any("master" in arg for arg in argv), argv
            assert "refs/heads/ledger-anchor:refs/heads/ledger-anchor" not in argv

    # The dummy task was created, read back and deleted -- and no other task was touched.
    scheduled = [argv for argv in fake.argvs() if argv[0] == "schtasks"]
    assert [argv[1] for argv in scheduled] == ["/create", "/query", "/delete"]
    assert all(
        argv[argv.index("/tn") + 1] == smoke.DUMMY_TASK_NAME for argv in scheduled
    )


def test_run_never_raises_and_records_each_failure(
    fake: FakeRunner, paths: smoke.SmokePaths
) -> None:
    def broken(
        argv: list[str], _kw: dict[str, Any]
    ) -> subprocess.CompletedProcess[bytes]:
        raise OSError(f"cannot start {argv[0]}")

    for prefix in (["whoami"], [SSH_EXECUTABLE], ["git"], ["schtasks"]):
        fake.route(prefix, broken)
    paths.backup_spec.key_path.unlink()  # a missing key is recorded, not raised

    assert smoke.main(["--run"], paths=paths) == 0

    result = json.loads(paths.result_path.read_text(encoding="ascii"))
    assert result["greetings"]["anchor"]["identity_ok"] is False
    assert result["greetings"]["backup"]["identity_ok"] is False
    assert result["anchor_push"]["ok"] is False
    assert result["backup_push"]["ok"] is False
    assert result["dummy_task"]["ok"] is False
    assert set(result["errors"]) >= {
        "whoami",
        "anchor_push",
        "backup_push",
        "dummy_task",
    }
    assert (
        "backup_key" in result["errors"]["backup_push"]
        or "deploy key" in (result["errors"]["backup_push"])
    )


# ---------------------------------------------------------------------------------------------
# 4. --register: a trigger-less task with the daily task's principal and settings
# ---------------------------------------------------------------------------------------------


def _element(root: ET.Element, name: str) -> ET.Element:
    found = root.find(f"{TASK_NS}{name}")
    assert found is not None, f"<{name}> missing"
    return found


def test_register_generates_triggerless_task_with_daily_principal(
    fake: FakeRunner, paths: smoke.SmokePaths
) -> None:
    fake.route(["schtasks", "/create"], done([]))
    fake.route(["schtasks", "/run"], done([]))
    fake.route(["git", "ls-remote"], done([], f"{MASTER_SHA}\trefs/heads/master\n"))
    paths.result_path.parent.mkdir(parents=True, exist_ok=True)
    paths.result_path.write_text(
        "{}", encoding="ascii"
    )  # a stale result from an old run

    assert smoke.main(["--register"], paths=paths) == 0

    assert not paths.result_path.exists(), "a stale result file must not be read back"
    generated = ET.fromstring(paths.task_xml_path.read_bytes().decode("utf-16").strip())
    template = ET.fromstring(DAILY_TEMPLATE.read_bytes().decode("utf-16").strip())

    triggers = generated.find(f"{TASK_NS}Triggers")
    assert triggers is None or len(triggers) == 0, "the smoke task must have no trigger"
    for name in ("Principals", "Settings"):
        assert ET.tostring(_element(generated, name)) == ET.tostring(
            _element(template, name)
        ), f"<{name}> differs from the daily task's"

    exec_generated = _element(_element(generated, "Actions"), "Exec")
    exec_template = _element(_element(template, "Actions"), "Exec")
    assert (
        _element(exec_generated, "Command").text
        == _element(exec_template, "Command").text
    )
    assert (
        _element(exec_generated, "Arguments").text
        == "run python -m scripts.ledger_push_smoke --run"
    )
    assert (
        _element(exec_generated, "WorkingDirectory").text
        == _element(exec_template, "WorkingDirectory").text
    )

    scheduled = [argv for argv in fake.argvs() if argv[0] == "schtasks"]
    assert scheduled == [
        [
            "schtasks",
            "/create",
            "/tn",
            smoke.SMOKE_TASK_NAME,
            "/xml",
            str(paths.task_xml_path),
            "/f",
        ],
        ["schtasks", "/run", "/tn", smoke.SMOKE_TASK_NAME],
    ]


# ---------------------------------------------------------------------------------------------
# 5. --cleanup deletes the remote branch, the local ref and both tasks
# ---------------------------------------------------------------------------------------------


def test_cleanup_deletes_branch_and_tasks(
    fake: FakeRunner, paths: smoke.SmokePaths, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.route(["gh", "api", "-X", "DELETE"], done([]))
    fake.route(["git", "ls-remote"], done([], ""))  # gone after the delete
    fake.route(["git", "rev-parse"], done([], SMOKE_COMMIT + "\n"))
    fake.route(["git", "update-ref", "-d"], done([]))
    fake.route(["schtasks", "/delete"], done([]))
    fake.route(["schtasks", "/query"], done([], code=1, err="ERROR: not found"))

    assert smoke.main(["--cleanup"], paths=paths) == 0

    argvs = fake.argvs()
    assert [
        "gh",
        "api",
        "-X",
        "DELETE",
        "repos/jackc625/nfl-predict/git/refs/heads/ledger-anchor-smoke",
    ] in argvs
    assert ["git", "update-ref", "-d", smoke.SMOKE_REF, SMOKE_COMMIT] in argvs
    assert ["schtasks", "/delete", "/tn", "NFL_Ledger_Push_Smoke", "/f"] in argvs
    assert ["schtasks", "/delete", "/tn", "NFL_Ledger_Smoke_Dummy", "/f"] in argvs
    for argv in argvs:
        assert "NFL_Predict_Pipeline" not in argv
        assert "NFL_Predict_Closing" not in argv
    assert "SMOKE_CLEANUP_OK= True" in capsys.readouterr().out


def test_cleanup_reports_a_branch_that_is_still_there(
    fake: FakeRunner, paths: smoke.SmokePaths, capsys: pytest.CaptureFixture[str]
) -> None:
    fake.route(["gh", "api", "-X", "DELETE"], done([], code=1, err="HTTP 403"))
    fake.route(["git", "ls-remote"], done([], f"{SMOKE_COMMIT}\t{smoke.SMOKE_REF}\n"))
    fake.route(["git", "rev-parse"], done([], code=1))
    fake.route(["schtasks", "/delete"], done([]))
    fake.route(["schtasks", "/query"], done([], code=1, err="ERROR: not found"))

    assert smoke.main(["--cleanup"], paths=paths) == 1
    assert "SMOKE_CLEANUP_OK= False" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# --readback: the owner context reads GitHub back and prints the verdict
# ---------------------------------------------------------------------------------------------


def _write_result(paths: smoke.SmokePaths, **overrides: Any) -> None:
    result: dict[str, Any] = {
        "whoami": "jackslaptop\\jackc",
        "greetings": {
            "anchor": {"identity_ok": True, "greeting": ANCHOR_GREETING.strip()},
            "backup": {"identity_ok": True, "greeting": BACKUP_GREETING.strip()},
        },
        "anchor_push": {"ok": True, "smoke_commit": SMOKE_COMMIT, "error": None},
        "backup_push": {"ok": True, "pushed_sha": LEDGER_HEAD, "error": None},
        "dummy_task": {"ok": True},
        "errors": {},
    }
    result.update(overrides)
    paths.result_path.parent.mkdir(parents=True, exist_ok=True)
    paths.result_path.write_text(json.dumps(result), encoding="ascii")


def _readback_world(fake: FakeRunner, remote_smoke: str) -> None:
    def ls_remote(
        argv: list[str], _kw: dict[str, Any]
    ) -> subprocess.CompletedProcess[bytes]:
        if argv[-1] == smoke.SMOKE_REF:
            return done(argv, f"{remote_smoke}\t{smoke.SMOKE_REF}\n")
        return done(argv, f"{MASTER_SHA}\trefs/heads/master\n")

    fake.route(["git", "ls-remote"], ls_remote)
    fake.route(["gh", "api"], done([], LEDGER_HEAD + "\n"))
    fake.route(["git", "-C"], done([], LEDGER_HEAD + "\n"))


def test_readback_pass(
    fake: FakeRunner, paths: smoke.SmokePaths, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_result(paths)
    _readback_world(fake, SMOKE_COMMIT)

    code = smoke.readback_smoke(paths, wait_seconds=0, sleep=lambda _s: None)

    out = capsys.readouterr().out
    assert code == 0
    for line in (
        "SMOKE_WHOAMI= jackslaptop\\jackc",
        "SMOKE_ANCHOR_IDENTITY_OK= True",
        "SMOKE_BACKUP_IDENTITY_OK= True",
        "SMOKE_ANCHOR_PUSH_OK= True",
        "SMOKE_BACKUP_PUSH_OK= True",
        "SMOKE_SCHTASKS_FROM_S4U_OK= True",
        "SMOKE_RESULT= PASS",
    ):
        assert line in out, out


def test_readback_fails_when_the_remote_disagrees(
    fake: FakeRunner, paths: smoke.SmokePaths, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_result(paths)
    _readback_world(fake, "0" * 40)

    assert smoke.readback_smoke(paths, wait_seconds=0, sleep=lambda _s: None) == 1
    out = capsys.readouterr().out
    assert "SMOKE_ANCHOR_PUSH_OK= False" in out
    assert "SMOKE_RESULT= FAIL" in out


def test_readback_schtasks_false_is_recorded_not_a_failure(
    fake: FakeRunner, paths: smoke.SmokePaths, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_result(paths, dummy_task={"ok": False})
    _readback_world(fake, SMOKE_COMMIT)

    assert smoke.readback_smoke(paths, wait_seconds=0, sleep=lambda _s: None) == 0
    out = capsys.readouterr().out
    assert "SMOKE_SCHTASKS_FROM_S4U_OK= False" in out
    assert "SMOKE_RESULT= PASS" in out


def test_readback_times_out_without_a_result_file(
    fake: FakeRunner, paths: smoke.SmokePaths, capsys: pytest.CaptureFixture[str]
) -> None:
    _readback_world(fake, SMOKE_COMMIT)

    assert smoke.readback_smoke(paths, wait_seconds=0, sleep=lambda _s: None) == 1
    assert "SMOKE_RESULT= FAIL" in capsys.readouterr().out


def test_help_lists_the_four_modes(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exit_info:
        smoke.main(["--help"])
    assert exit_info.value.code == 0
    out = capsys.readouterr().out
    for flag in ("--run", "--register", "--readback", "--cleanup"):
        assert flag in out
