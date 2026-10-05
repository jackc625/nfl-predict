"""The one owner-run setup command for the ledger's remotes and keys (Plan 34-13, D-01, D-03, D-04).

Every ``gh``, ``ssh-keygen``, ``icacls`` and ``git`` call the command makes goes through the
module-level ``RUNNER`` seam. These tests replace it with :class:`FakeRunner`, which answers from
canned responses and FAILS on any command it was not told about -- so no test here reaches GitHub,
generates a real key or changes a real ACL. Key and known_hosts paths are ``tmp_path`` files.

The fake ``ssh-keygen -lf`` computes REAL SHA256 fingerprints (base64 of the SHA256 of the key
blob), so the host-key test checks GitHub's actual published keys against their published
fingerprints rather than against a lookup table the test wrote itself.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import os
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest

from forward_ledger.remote_config import BACKUP_REPO_SLUG, PUBLIC_REPO_SLUG
from scripts import setup_ledger_remote as setup

# GitHub's published host keys (docs.github.com "GitHub's SSH key fingerprints"), as served by
# ``gh api meta`` under ``ssh_keys``.
GITHUB_ED25519 = (
    "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl"
)
GITHUB_ECDSA = (
    "ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBEmKSENjQEezOmxkZMy"
    "7opKgwFB9nkt5YRrYMjNuG5N87uRgg6CLrbo5wAdT/y6v0mKV0U2w0WZ2YB/++Tpockg="
)
GITHUB_RSA = (
    "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQCj7ndNxQowgcQnjshcLrqPEiiphnt+VTTvDP6mHBL9j1aNUkY4Ue1g"
    "vwnGLVlOhGeYrnZaMgRK6+PKCUXaDbC7qtbW8gIkhL7aGCsOr/C56SJMy/BCZfxd1nWzAOxSDPgVsmerOBYfNqltV9/h"
    "WCqBywINIR+5dIg6JTJ72pcEpEjcYgXkE2YEFXV1JHnsKgbLWNlhScqb2UmyRkQyytRLtL+38TGxkxCflmO+5Z8CSSNY7G"
    "idjMIZ7Q4zMjA2n1nGrlTDkzwDCsw+wqFPGQA179cnfGWOWRVruj16z6XyvxvjJwbz0wQZ75XK5tKSb7FNyeIEs4TT4jk+"
    "S4dhPeAUC5y+bDYirYgM4GC7uEnztnZyaVWQ7B381AK4Qdrwt51ZqExKbQpTUNn+EjqoTwvqNj4kqx5QUCI0ThS/YkOxJ"
    "CXmPUWZbhjpCg56i+2aB6CmK2JGhn57K5mj0MNdBXA4/WnwH6XoPWJzK5Nyu2zB3nAZp+S5hpQs+p1vN1/wsjk="
)
GITHUB_HOST_KEYS = (GITHUB_ED25519, GITHUB_ECDSA, GITHUB_RSA)

_KEY_LABELS = {
    "ssh-ed25519": "ED25519",
    "ecdsa-sha2-nistp256": "ECDSA",
    "ssh-rsa": "RSA",
}

PRIVATE_KEY_TEXT = "-----BEGIN OPENSSH PRIVATE KEY-----\nNOT-A-REAL-KEY\n-----END OPENSSH PRIVATE KEY-----\n"

MASTER_RULESET_ID = 7001
ANCHOR_RULESET_ID = 7002
ROOT_COMMIT = "a" * 40

Handler = Callable[[list[str], Path | None], subprocess.CompletedProcess[bytes]]


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
        self.calls: list[list[str]] = []
        self._routes: list[tuple[tuple[str, ...], Handler]] = []

    def on(self, prefix: Sequence[str], handler: Handler) -> None:
        self._routes.insert(0, (tuple(prefix), handler))

    def reply(
        self,
        prefix: Sequence[str],
        out: str | bytes = b"",
        code: int = 0,
        err: str = "",
    ) -> None:
        self.on(prefix, lambda argv, _cwd: done(argv, out, code, err))

    def reply_json(self, prefix: Sequence[str], payload: object) -> None:
        self.reply(prefix, json.dumps(payload))

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: object = None,
        input_bytes: bytes | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]:
        command = [str(part) for part in argv]
        self.calls.append(command)
        for prefix, handler in self._routes:
            if tuple(command[: len(prefix)]) == prefix:
                return handler(command, cwd)
        msg = f"the setup ran a command the test did not expect: {command}"
        raise AssertionError(msg)

    def ran(self, *prefix: str) -> list[list[str]]:
        return [call for call in self.calls if tuple(call[: len(prefix)]) == prefix]


def fingerprint_of(blob_b64: str) -> str:
    digest = hashlib.sha256(base64.b64decode(blob_b64)).digest()
    return "SHA256:" + base64.b64encode(digest).decode("ascii").rstrip("=")


def fake_keygen_list(
    argv: list[str], _cwd: Path | None
) -> subprocess.CompletedProcess[bytes]:
    """``ssh-keygen -lf <file>`` over a public key or known_hosts file, with real fingerprints."""
    lines = []
    for raw in Path(argv[-1]).read_text(encoding="ascii").splitlines():
        tokens = raw.split()
        if not tokens or tokens[0].startswith("#"):
            continue
        if tokens[0] in _KEY_LABELS:
            key_type, blob, name = (
                tokens[0],
                tokens[1],
                " ".join(tokens[2:]) or "no comment",
            )
        else:
            name, key_type, blob = tokens[0], tokens[1], tokens[2]
        lines.append(f"256 {fingerprint_of(blob)} {name} ({_KEY_LABELS[key_type]})")
    return done(argv, "\n".join(lines) + "\n")


def fake_keygen_create(
    argv: list[str], _cwd: Path | None
) -> subprocess.CompletedProcess[bytes]:
    """``ssh-keygen -t ed25519 ... -f <path>``: writes a pair and, like the real one, talks a lot.

    Its stdout deliberately carries private-key text, so a setup that echoed it would be caught.
    """
    path = Path(argv[argv.index("-f") + 1])
    comment = argv[argv.index("-C") + 1]
    path.write_text(PRIVATE_KEY_TEXT, encoding="ascii")
    blob = base64.b64encode(f"public-{path.name}".encode("ascii")).decode("ascii")
    Path(f"{path}.pub").write_text(f"ssh-ed25519 {blob} {comment}\n", encoding="ascii")
    return done(argv, f"Your identification has been saved.\n{PRIVATE_KEY_TEXT}")


def owner_only_acl(
    argv: list[str], _cwd: Path | None
) -> subprocess.CompletedProcess[bytes]:
    return done(
        argv,
        f"{argv[1]} JACKSLAPTOP\\jackc:(F)\n\n"
        "Successfully processed 1 files; Failed processing 0 files\n",
    )


def inherited_acl(
    argv: list[str], _cwd: Path | None
) -> subprocess.CompletedProcess[bytes]:
    pad = " " * len(argv[1])
    return done(
        argv,
        f"{argv[1]} NT AUTHORITY\\SYSTEM:(I)(F)\n"
        f"{pad} BUILTIN\\Administrators:(I)(F)\n"
        f"{pad} JACKSLAPTOP\\jackc:(I)(F)\n\n"
        "Successfully processed 1 files; Failed processing 0 files\n",
    )


def make_paths(tmp_path: Path) -> setup.SetupPaths:
    ssh_dir = tmp_path / "ssh"
    ssh_dir.mkdir()
    return setup.SetupPaths(
        anchor_key=ssh_dir / "nfl_ledger_anchor_ed25519",
        backup_key=ssh_dir / "nfl_ledger_backup_ed25519",
        known_hosts=ssh_dir / "nfl_ledger_github_known_hosts",
        ledger_dir=tmp_path / "ledger",
    )


def public_key_material(key_path: Path) -> str:
    """``<type> <base64>`` of a public key file, as GitHub's keys API returns it."""
    return " ".join(Path(f"{key_path}.pub").read_text(encoding="ascii").split()[:2])


def ruleset_detail(payload: dict, ruleset_id: int, can_bypass: str) -> dict:
    detail = copy.deepcopy(payload)
    detail.update(
        id=ruleset_id,
        source_type="Repository",
        source=PUBLIC_REPO_SLUG,
        current_user_can_bypass=can_bypass,
    )
    return detail


def branch_rules(payload: dict, ruleset_id: int) -> list[dict]:
    return [
        {
            "type": rule["type"],
            "ruleset_source_type": "Repository",
            "ruleset_source": PUBLIC_REPO_SLUG,
            "ruleset_id": ruleset_id,
        }
        for rule in payload["rules"]
    ]


def good_world(tmp_path: Path) -> tuple[FakeRunner, setup.SetupPaths]:
    """A machine and account where everything was set up exactly as specified (pre-go-live)."""
    paths = make_paths(tmp_path)
    runner = FakeRunner()
    for key in (paths.anchor_key, paths.backup_key):
        fake_keygen_create(
            ["ssh-keygen", "-C", f"comment for {key.name}", "-f", str(key)], None
        )
    paths.known_hosts.write_text(
        "".join(f"github.com {key}\n" for key in GITHUB_HOST_KEYS), encoding="ascii"
    )
    (paths.ledger_dir / ".git").mkdir(parents=True)

    runner.on(["icacls"], owner_only_acl)
    runner.on(["ssh-keygen", "-lf"], fake_keygen_list)
    runner.reply_json(
        ["gh", "api", f"repos/{BACKUP_REPO_SLUG}"],
        {"full_name": BACKUP_REPO_SLUG, "visibility": "private", "private": True},
    )
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/keys"],
        [
            {
                "id": 101,
                "key": public_key_material(paths.anchor_key),
                "title": "ledger anchor (S4U)",
                "read_only": False,
            }
        ],
    )
    runner.reply_json(
        ["gh", "api", f"repos/{BACKUP_REPO_SLUG}/keys"],
        [
            {
                "id": 202,
                "key": public_key_material(paths.backup_key),
                "title": "ledger backup (S4U)",
                "read_only": False,
            }
        ],
    )
    master = setup.master_ruleset_payload()
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets?per_page=100"],
        [{"id": MASTER_RULESET_ID, "name": master["name"], "enforcement": "active"}],
    )
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets/{MASTER_RULESET_ID}"],
        ruleset_detail(master, MASTER_RULESET_ID, "always"),
    )
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rules/branches/master"],
        branch_rules(master, MASTER_RULESET_ID),
    )
    runner.reply(["git", "-c", "credential.helper=", "ls-remote"], b"")
    runner.reply(["git", "check-ignore"], b"")
    runner.reply(["git", "rev-parse", "--abbrev-ref", "HEAD"], "main\n")
    runner.reply(["git", "rev-list", "--max-parents=0", "HEAD"], f"{ROOT_COMMIT}\n")
    runner.reply(
        ["git", "show", "--name-only", "--format=", ROOT_COMMIT],
        "\n.gitattributes\n.gitignore\nREADME.md\n",
    )
    runner.reply(["git", "show", f"{ROOT_COMMIT}:.gitattributes"], "* -text\n")
    return runner, paths


def with_anchor_branch_and_ruleset(runner: FakeRunner, *, ruleset: bool) -> None:
    """The world after the first anchor push; with or without the anchor ruleset."""
    runner.reply(
        ["git", "-c", "credential.helper=", "ls-remote"],
        f"{'b' * 40}\trefs/heads/ledger-anchor\n",
    )
    if not ruleset:
        return
    master = setup.master_ruleset_payload()
    anchor = setup.anchor_ruleset_payload()
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets?per_page=100"],
        [
            {"id": MASTER_RULESET_ID, "name": master["name"], "enforcement": "active"},
            {"id": ANCHOR_RULESET_ID, "name": anchor["name"], "enforcement": "active"},
        ],
    )
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets/{ANCHOR_RULESET_ID}"],
        ruleset_detail(anchor, ANCHOR_RULESET_ID, "never"),
    )
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rules/branches/ledger-anchor"],
        branch_rules(anchor, ANCHOR_RULESET_ID),
    )


def run_cli(
    monkeypatch: pytest.MonkeyPatch,
    runner: FakeRunner,
    paths: setup.SetupPaths,
    *argv: str,
) -> int:
    monkeypatch.setattr(setup, "RUNNER", runner)
    return setup.main(list(argv), paths=paths)


def output_lines(capsys: pytest.CaptureFixture[str]) -> list[str]:
    captured = capsys.readouterr()
    return (captured.out + captured.err).splitlines()


def line_for(lines: list[str], field: str) -> str:
    found = [line for line in lines if line.startswith(f"{field}= ")]
    assert len(found) == 1, (field, lines)
    return found[0]


# ---------------------------------------------------------------------------------------------
# create-keys
# ---------------------------------------------------------------------------------------------


def test_create_keys_refuses_existing_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_paths(tmp_path)
    paths.backup_key.write_text("an existing key the owner did not ask to replace\n")
    runner = FakeRunner()

    code = run_cli(monkeypatch, runner, paths, "create-keys")

    assert code == 1
    assert runner.calls == [], "nothing may run once an existing key is found"
    assert not paths.anchor_key.exists(), (
        "neither key is generated when one already exists"
    )
    lines = output_lines(capsys)
    assert any(
        line.startswith("SETUP_REFUSED= ") and paths.backup_key.name in line
        for line in lines
    ), lines


def test_create_keys_sets_owner_only_acl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_paths(tmp_path)
    runner = FakeRunner()
    runner.on(["ssh-keygen", "-t"], fake_keygen_create)
    runner.on(["ssh-keygen", "-lf"], fake_keygen_list)
    runner.on(["icacls"], owner_only_acl)

    code = run_cli(monkeypatch, runner, paths, "create-keys")

    assert code == 0, output_lines(capsys)
    for key, slug in (
        (paths.anchor_key, PUBLIC_REPO_SLUG),
        (paths.backup_key, BACKUP_REPO_SLUG),
    ):
        generate = [
            call for call in runner.ran("ssh-keygen", "-t") if call[-1] == str(key)
        ]
        assert len(generate) == 1, runner.calls
        call = generate[0]
        assert call[:5] == ["ssh-keygen", "-t", "ed25519", "-N", ""]
        assert call[5] == "-C" and slug in call[6], call
        assert call[7:] == ["-f", str(key)]
        acl = ["icacls", str(key), "/inheritance:r", "/grant:r", "jackc:(F)"]
        assert acl in runner.calls
        assert runner.calls.index(acl) > runner.calls.index(call), (
            "the ACL follows the key"
        )
    lines = output_lines(capsys)
    expected = fingerprint_of(public_key_material(paths.anchor_key).split()[1])
    assert line_for(lines, "ANCHOR_KEY_FINGERPRINT").endswith(expected)
    assert line_for(lines, "ANCHOR_KEY_ACL").endswith("| MATCH")
    assert line_for(lines, "BACKUP_KEY_ACL").endswith("| MATCH")


def test_key_acl_with_inherited_entries_is_a_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, paths = good_world(tmp_path)
    runner.on(["icacls"], inherited_acl)

    code = run_cli(monkeypatch, runner, paths, "verify")

    lines = output_lines(capsys)
    assert code == 1
    assert line_for(lines, "ANCHOR_KEY_ACL").endswith("| MISMATCH")
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= False"


# ---------------------------------------------------------------------------------------------
# pin-host-keys
# ---------------------------------------------------------------------------------------------


def test_pin_host_keys_writes_the_published_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_paths(tmp_path)
    runner = FakeRunner()
    runner.reply_json(["gh", "api", "meta"], {"ssh_keys": list(GITHUB_HOST_KEYS)})
    runner.on(["ssh-keygen", "-lf"], fake_keygen_list)

    code = run_cli(monkeypatch, runner, paths, "pin-host-keys")

    assert code == 0, output_lines(capsys)
    pinned = paths.known_hosts.read_text(encoding="ascii").splitlines()
    assert sorted(pinned) == sorted(f"github.com {key}" for key in GITHUB_HOST_KEYS)
    assert line_for(output_lines(capsys), "KNOWN_HOSTS_FINGERPRINTS").endswith(
        "| MATCH"
    )


def test_pin_host_keys_rejects_wrong_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_paths(tmp_path)
    impostor = "ssh-ed25519 " + base64.b64encode(b"not github's ed25519 key").decode(
        "ascii"
    )
    runner = FakeRunner()
    runner.reply_json(
        ["gh", "api", "meta"], {"ssh_keys": [impostor, GITHUB_ECDSA, GITHUB_RSA]}
    )
    runner.on(["ssh-keygen", "-lf"], fake_keygen_list)

    code = run_cli(monkeypatch, runner, paths, "pin-host-keys")

    assert code == 1
    assert not paths.known_hosts.exists(), "a refused pin writes no known_hosts file"
    lines = output_lines(capsys)
    assert any(line.startswith("SETUP_REFUSED= ") for line in lines), lines
    assert "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU" in "\n".join(lines)


# ---------------------------------------------------------------------------------------------
# rulesets
# ---------------------------------------------------------------------------------------------


def _capture_post(runner: FakeRunner, ruleset_id: int) -> list[dict]:
    posted: list[dict] = []

    def handler(
        argv: list[str], _cwd: Path | None
    ) -> subprocess.CompletedProcess[bytes]:
        input_path = Path(argv[argv.index("--input") + 1])
        posted.append(json.loads(input_path.read_text(encoding="utf-8")))
        return done(argv, json.dumps({"id": ruleset_id, "name": posted[-1]["name"]}))

    runner.on(
        ["gh", "api", "-X", "POST", f"repos/{PUBLIC_REPO_SLUG}/rulesets"], handler
    )
    return posted


def test_master_ruleset_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, paths = good_world(tmp_path)
    listing = [["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets?per_page=100"]]
    runner.reply_json(listing[0], [])
    posted = _capture_post(runner, MASTER_RULESET_ID)

    def listed_after_post(
        argv: list[str], _cwd: Path | None
    ) -> subprocess.CompletedProcess[bytes]:
        rows = [
            {
                "id": MASTER_RULESET_ID,
                "name": setup.MASTER_RULESET_NAME,
                "enforcement": "active",
            }
        ]
        return done(argv, json.dumps(rows if posted else []))

    runner.on(listing[0], listed_after_post)

    code = run_cli(monkeypatch, runner, paths, "create-master-ruleset")

    assert code == 0, output_lines(capsys)
    assert len(posted) == 1
    body = posted[0]
    assert body["name"] == setup.MASTER_RULESET_NAME
    assert body["target"] == "branch"
    assert body["enforcement"] == "active"
    assert body["conditions"] == {
        "ref_name": {"include": ["refs/heads/master"], "exclude": []}
    }
    assert body["rules"] == [
        {"type": "update", "parameters": {"update_allows_fetch_and_merge": False}},
        {"type": "deletion"},
        {"type": "non_fast_forward"},
    ]
    assert body["bypass_actors"] == [
        {"actor_id": 5, "actor_type": "RepositoryRole", "bypass_mode": "always"}
    ]
    lines = output_lines(capsys)
    assert (
        line_for(lines, "MASTER_RULESET_ID")
        == f"MASTER_RULESET_ID= {MASTER_RULESET_ID}"
    )
    assert line_for(lines, "MASTER_RULESET_BYPASS_ACTORS").endswith("| MATCH")


def test_create_ruleset_refuses_a_duplicate_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, paths = good_world(tmp_path)
    posted = _capture_post(runner, 9999)

    code = run_cli(monkeypatch, runner, paths, "create-master-ruleset")

    assert code == 1
    assert posted == []
    assert any(line.startswith("SETUP_REFUSED= ") for line in output_lines(capsys))


def test_anchor_ruleset_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, paths = good_world(tmp_path)
    with_anchor_branch_and_ruleset(runner, ruleset=True)
    master = setup.master_ruleset_payload()
    posted = _capture_post(runner, ANCHOR_RULESET_ID)

    def listed(
        argv: list[str], _cwd: Path | None
    ) -> subprocess.CompletedProcess[bytes]:
        rows = [
            {"id": MASTER_RULESET_ID, "name": master["name"], "enforcement": "active"}
        ]
        if posted:
            rows.append(
                {
                    "id": ANCHOR_RULESET_ID,
                    "name": setup.ANCHOR_RULESET_NAME,
                    "enforcement": "active",
                }
            )
        return done(argv, json.dumps(rows))

    runner.on(["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets?per_page=100"], listed)

    code = run_cli(monkeypatch, runner, paths, "create-anchor-ruleset")

    assert code == 0, output_lines(capsys)
    assert len(posted) == 1
    body = posted[0]
    assert body["name"] == setup.ANCHOR_RULESET_NAME
    assert body["enforcement"] == "active"
    assert body["conditions"] == {
        "ref_name": {"include": ["refs/heads/ledger-anchor"], "exclude": []}
    }
    assert body["rules"] == [{"type": "deletion"}, {"type": "non_fast_forward"}]
    assert body["bypass_actors"] == []


def test_anchor_ruleset_refused_before_the_branch_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, paths = good_world(tmp_path)
    posted = _capture_post(runner, ANCHOR_RULESET_ID)

    code = run_cli(monkeypatch, runner, paths, "create-anchor-ruleset")

    assert code == 1
    assert posted == []
    lines = output_lines(capsys)
    assert any(
        line.startswith("SETUP_REFUSED= ") and "ledger-anchor" in line for line in lines
    )


def test_no_ruleset_on_private_repo(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = FakeRunner()
    monkeypatch.setattr(setup, "RUNNER", runner)

    with pytest.raises(setup.SetupRefused, match="private"):
        setup.create_ruleset(setup.master_ruleset_payload(), slug=BACKUP_REPO_SLUG)

    assert runner.calls == [], "the refusal comes before any GitHub call"


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------


def test_verify_parses_readbacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, paths = good_world(tmp_path)

    code = run_cli(monkeypatch, runner, paths, "verify")

    lines = output_lines(capsys)
    assert code == 0, lines
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= True"
    for field in (
        "ANCHOR_KEY_ACL",
        "BACKUP_KEY_ACL",
        "KNOWN_HOSTS_FINGERPRINTS",
        "BACKUP_REPO_VISIBILITY",
        "ANCHOR_DEPLOY_KEY",
        "BACKUP_DEPLOY_KEY",
        "MASTER_RULESET",
        "MASTER_RULESET_ENFORCEMENT",
        "MASTER_RULESET_RULES",
        "MASTER_RULESET_BYPASS_ACTORS",
        "MASTER_RULESET_DEPLOYKEY_ACTORS",
        "MASTER_RULESET_CURRENT_USER_CAN_BYPASS",
        "MASTER_RULESET_ACTIVE_ON_BRANCH",
        "NESTED_REPO_IGNORED",
        "NESTED_REPO_FIRST_COMMIT_FILES",
    ):
        line = line_for(lines, field)
        assert line.count(" | ") == 2 and line.endswith("| MATCH"), line
    assert line_for(lines, "BACKUP_REPO_VISIBILITY") == (
        "BACKUP_REPO_VISIBILITY= private | private | MATCH"
    )
    assert not [line for line in lines if line.endswith("MISMATCH")]


def _with_deploy_key_bypass(runner: FakeRunner) -> str:
    detail = ruleset_detail(setup.master_ruleset_payload(), MASTER_RULESET_ID, "always")
    detail["bypass_actors"].append(
        {"actor_id": None, "actor_type": "DeployKey", "bypass_mode": "always"}
    )
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets/{MASTER_RULESET_ID}"], detail
    )
    return "MASTER_RULESET_DEPLOYKEY_ACTORS"


def _with_read_only_key(runner: FakeRunner) -> str:
    keys = json.loads(
        runner(["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/keys"]).stdout.decode("ascii")
    )
    keys[0]["read_only"] = True
    runner.reply_json(["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/keys"], keys)
    return "ANCHOR_DEPLOY_KEY"


def _with_public_backup(runner: FakeRunner) -> str:
    runner.reply_json(
        ["gh", "api", f"repos/{BACKUP_REPO_SLUG}"],
        {"full_name": BACKUP_REPO_SLUG, "visibility": "public", "private": False},
    )
    return "BACKUP_REPO_VISIBILITY"


@pytest.mark.parametrize(
    "break_world",
    [_with_deploy_key_bypass, _with_read_only_key, _with_public_backup],
    ids=["deploy-key-bypass-actor", "read-only-deploy-key", "public-backup-repo"],
)
def test_verify_parses_readbacks_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    break_world: Callable[[FakeRunner], str],
) -> None:
    runner, paths = good_world(tmp_path)
    field = break_world(runner)

    code = run_cli(monkeypatch, runner, paths, "verify")

    lines = output_lines(capsys)
    assert code == 1
    assert line_for(lines, field).endswith("| MISMATCH"), lines
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= False"


@pytest.mark.parametrize("count", [0, 2])
def test_verify_finds_rulesets_by_exact_name(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    count: int,
) -> None:
    runner, paths = good_world(tmp_path)
    near_miss = {
        "id": 1,
        "name": setup.MASTER_RULESET_NAME + " ",
        "enforcement": "active",
    }
    named = [
        {
            "id": MASTER_RULESET_ID + n,
            "name": setup.MASTER_RULESET_NAME,
            "enforcement": "active",
        }
        for n in range(count)
    ]
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets?per_page=100"],
        [near_miss, *named],
    )

    code = run_cli(monkeypatch, runner, paths, "verify")

    lines = output_lines(capsys)
    assert code == 1
    line = line_for(lines, "MASTER_RULESET")
    assert line.endswith("| MISMATCH") and f"{count} found" in line, line
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= False"


def test_verify_refuses_a_ruleset_that_is_not_active(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, paths = good_world(tmp_path)
    detail = ruleset_detail(setup.master_ruleset_payload(), MASTER_RULESET_ID, "always")
    detail["enforcement"] = "disabled"
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets?per_page=100"],
        [
            {
                "id": MASTER_RULESET_ID,
                "name": setup.MASTER_RULESET_NAME,
                "enforcement": "disabled",
            }
        ],
    )
    runner.reply_json(
        ["gh", "api", f"repos/{PUBLIC_REPO_SLUG}/rulesets/{MASTER_RULESET_ID}"], detail
    )

    code = run_cli(monkeypatch, runner, paths, "verify")

    lines = output_lines(capsys)
    assert code == 1
    assert line_for(lines, "MASTER_RULESET_ENFORCEMENT").endswith("| MISMATCH")
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= False"


def test_verify_requires_anchor_ruleset_once_the_branch_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Bootstrap: no remote ledger-anchor branch and no anchor ruleset -> match.
    runner, paths = good_world(tmp_path / "bootstrap")
    assert run_cli(monkeypatch, runner, paths, "verify") == 0
    lines = output_lines(capsys)
    assert line_for(lines, "ANCHOR_RULESET").endswith("| MATCH")
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= True"

    # After the first anchor push, no anchor ruleset -> mismatch naming ANCHOR_RULESET.
    runner, paths = good_world(tmp_path / "pushed")
    with_anchor_branch_and_ruleset(runner, ruleset=False)
    assert run_cli(monkeypatch, runner, paths, "verify") == 1
    lines = output_lines(capsys)
    assert line_for(lines, "ANCHOR_RULESET").endswith("| MISMATCH")
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= False"

    # After the first anchor push, the anchor ruleset correct -> match.
    runner, paths = good_world(tmp_path / "protected")
    with_anchor_branch_and_ruleset(runner, ruleset=True)
    assert run_cli(monkeypatch, runner, paths, "verify") == 0
    lines = output_lines(capsys)
    assert line_for(lines, "ANCHOR_RULESET_BYPASS_ACTORS").endswith("| MATCH")
    assert line_for(lines, "ANCHOR_RULESET_CURRENT_USER_CAN_BYPASS").endswith("| MATCH")
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= True"


def test_verify_unreachable_remote_requires_the_anchor_ruleset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    runner, paths = good_world(tmp_path)
    runner.reply(
        ["git", "-c", "credential.helper=", "ls-remote"],
        b"",
        code=128,
        err="network down",
    )

    assert run_cli(monkeypatch, runner, paths, "verify") == 1
    lines = output_lines(capsys)
    assert line_for(lines, "ANCHOR_BRANCH_ON_REMOTE").endswith("| MISMATCH")
    assert line_for(lines, "SETUP_READBACK_MATCH") == "SETUP_READBACK_MATCH= False"


# ---------------------------------------------------------------------------------------------
# init-backup-repo, the private-key rule and the CLI surface
# ---------------------------------------------------------------------------------------------


def test_init_backup_repo_runs_git_through_the_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_paths(tmp_path)
    runner = FakeRunner()
    runner.reply(["git", "check-ignore"], b"")

    def real_git(
        argv: list[str], cwd: Path | None
    ) -> subprocess.CompletedProcess[bytes]:
        env = {
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
        }
        return subprocess.run(
            argv, cwd=cwd, env={**os.environ, **env}, capture_output=True, check=False
        )

    for verb in ("init", "add", "commit", "rev-parse", "rev-list", "show"):
        runner.on(["git", verb], real_git)

    code = run_cli(monkeypatch, runner, paths, "init-backup-repo")

    lines = output_lines(capsys)
    assert code == 0, lines
    assert (paths.ledger_dir / ".git").is_dir()
    assert runner.ran("git", "init"), "the nested repository is created through RUNNER"
    assert line_for(lines, "NESTED_REPO_FIRST_COMMIT_FILES").endswith("| MATCH")
    assert line_for(lines, "NESTED_REPO_GITATTRIBUTES").endswith("| MATCH")


def test_init_backup_repo_refuses_when_the_public_repo_would_see_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_paths(tmp_path)
    runner = FakeRunner()
    runner.reply(["git", "check-ignore"], b"", code=1)

    assert run_cli(monkeypatch, runner, paths, "init-backup-repo") == 1
    assert not paths.ledger_dir.exists()
    assert any(line.startswith("SETUP_REFUSED= ") for line in output_lines(capsys))


def test_never_prints_private_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = make_paths(tmp_path)
    runner = FakeRunner()
    runner.on(["ssh-keygen", "-t"], fake_keygen_create)
    runner.on(["ssh-keygen", "-lf"], fake_keygen_list)
    runner.on(["icacls"], owner_only_acl)
    assert run_cli(monkeypatch, runner, paths, "create-keys") == 0

    world_runner, world_paths = good_world(tmp_path / "world")
    run_cli(monkeypatch, world_runner, world_paths, "verify")

    captured = "\n".join(output_lines(capsys))
    assert "PRIVATE KEY" not in captured
    assert "NOT-A-REAL-KEY" not in captured


def test_help_lists_the_eight_subcommands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exited:
        setup.main(["--help"])

    assert exited.value.code == 0
    text = capsys.readouterr().out
    for name in (
        "create-keys",
        "pin-host-keys",
        "create-backup-repo",
        "register-deploy-keys",
        "create-master-ruleset",
        "create-anchor-ruleset",
        "init-backup-repo",
        "verify",
    ):
        assert name in text, name
