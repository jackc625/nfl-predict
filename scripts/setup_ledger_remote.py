#!/usr/bin/env python3
"""Owner-run setup of the ledger's remotes, keys and GitHub rules, with a field-by-field read-back.

Phase 34, Plan 34-13 (D-01, D-03, D-04). Everything the unattended pushes need is created here,
in the owner's own shell, after an explicit owner approval -- never by the daily run:

* ``create-keys``: two passphrase-less ed25519 deploy keys under the owner's ``.ssh`` directory,
  OUTSIDE the repository, each with an owner-only ACL (``icacls /inheritance:r /grant:r``).
  Refuses to overwrite an existing key. Prints public fingerprints only.
* ``pin-host-keys``: GitHub's SSH host keys from ``gh api meta``, each checked with
  ``ssh-keygen -lf`` against the fingerprints GitHub publishes, written to the dedicated
  known_hosts file the push uses with ``StrictHostKeyChecking=yes`` (no trust on first use).
* ``create-backup-repo``: the PRIVATE backup repository, visibility read back.
* ``register-deploy-keys``: one write-scoped deploy key per repository (GitHub refuses one key on
  two repositories), read back with ``read_only`` false.
* ``create-master-ruleset``: the public repo's ``master`` ruleset -- only the repository admin
  role may bypass, so a deploy-key push to ``master`` is refused BY GITHUB (D-04). It is verified
  by reading it back, never by a test push.
* ``create-anchor-ruleset``: ``ledger-anchor`` may not be deleted or force-pushed by anyone.
  Refused until the remote branch exists (Plan 34-19 creates it right after the first push).
* ``init-backup-repo``: ``ledger/`` becomes the nested backup repository
  (``forward_ledger.backup.ensure_backup_repo``); refused unless the public repo ignores it.
* ``verify``: reads ALL of it back and prints ``FIELD= want | got | MATCH|MISMATCH`` lines and
  ``SETUP_READBACK_MATCH= True|False``.

NO RULESET ON THE PRIVATE REPO. Rulesets on private repositories need GitHub Pro (34-RESEARCH
E2); the backup is protected only by the writer never force-pushing.

RULESETS ARE FOUND BY NAME. No ruleset id is stored anywhere: ``verify`` lists the public repo's
rulesets and requires exactly one with each committed name (zero or two-plus is a mismatch). The
anchor ruleset is required exactly when the remote has ``refs/heads/ledger-anchor`` -- so the same
command passes at bootstrap and demands the anchor ruleset after go-live, with no stage flag.

DEPLOY KEYS ARE TIED TO THE gh LOGIN. ``gh`` documents that keys it adds are removed if its token
is de-authorised; after a ``gh auth logout`` re-run ``register-deploy-keys`` (RUNBOOK).

Every ``gh``, ``ssh-keygen``, ``icacls`` and ``git`` call goes through the module-level
``RUNNER`` seam, so the unit tests touch no account, no key and no ACL. No private key is ever
read, printed or logged: only ``.pub`` files are read, and :func:`_emit` redacts any line that
carries private-key text.

Exit codes: 0 match; 1 mismatch or refusal.

Usage:
    uv run python -m scripts.setup_ledger_remote <subcommand>

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from forward_ledger.backup import ensure_backup_repo
from forward_ledger.remote_config import (
    ANCHOR_KEY_PATH,
    ANCHOR_REF,
    ANCHOR_REMOTE_HTTPS_URL,
    BACKUP_KEY_PATH,
    BACKUP_REPO_SLUG,
    KNOWN_HOSTS_PATH,
    PUBLIC_REPO_SLUG,
)
from forward_ledger.store import LEDGER_DIR

__all__ = [
    "ADMIN_REPOSITORY_ROLE_ID",
    "ANCHOR_RULESET_NAME",
    "DEFAULT_PATHS",
    "GITHUB_HOST_KEY_FINGERPRINTS",
    "KEY_OWNER_ACCOUNT",
    "MASTER_RULESET_NAME",
    "RUNNER",
    "SetupPaths",
    "SetupRefused",
    "anchor_ruleset_payload",
    "create_ruleset",
    "main",
    "master_ruleset_payload",
]

# GitHub's published SSH host-key fingerprints, keyed by the type label ``ssh-keygen -l`` prints.
# Source: docs.github.com/en/authentication/keeping-your-account-and-data-secure/
# githubs-ssh-key-fingerprints (34-RESEARCH D).
GITHUB_HOST_KEY_FINGERPRINTS: dict[str, str] = {
    "ED25519": "SHA256:+DiY3wvvV6TuJJhbpZisF/zLDA0zPMSvHdkr4UvCOqU",
    "ECDSA": "SHA256:p2QAMXNIC1TJYWeIOttrVc98/R1BUFWu3/LiyKgUfQM",
    "RSA": "SHA256:uNiVztksCsDhcc0u9e8BujQXVUpKZIDTMczCvj3tD2s",
}
GITHUB_HOST = "github.com"

MASTER_RULESET_NAME = "master: owner-only updates (ledger deploy key refused)"
ANCHOR_RULESET_NAME = "ledger-anchor: no deletion, no force-push"

# GitHub's built-in repository role id for "admin" (34-RESEARCH E2; confirmed by the read-back).
ADMIN_REPOSITORY_ROLE_ID = 5

# The Windows account that alone may read the deploy keys (the S4U task's user, D-03).
KEY_OWNER_ACCOUNT = "jackc"

BACKUP_REPO_DESCRIPTION = (
    "Private backup of the nfl-predict 2026 forward bet ledger (D-01)"
)

# The nested repository's first commit (forward_ledger.backup) and its attributes line.
NESTED_FIRST_COMMIT_FILES: tuple[str, ...] = (
    ".gitattributes",
    ".gitignore",
    "README.md",
)
NESTED_GITATTRIBUTES = "* -text"
NESTED_BRANCH = "main"

_COMMAND_TIMEOUT_SECONDS = 120
_ERROR_TEXT_LIMIT = 400


class Runner(Protocol):
    """Runs one external command; returns the completed process and never raises on exit status."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        cwd: Path | None = None,
        env: Mapping[str, str] | None = None,
        input_bytes: bytes | None = None,
        timeout: float | None = None,
    ) -> subprocess.CompletedProcess[bytes]: ...


def _run(
    argv: Sequence[str],
    *,
    cwd: Path | None = None,
    env: Mapping[str, str] | None = None,
    input_bytes: bytes | None = None,
    timeout: float | None = _COMMAND_TIMEOUT_SECONDS,
) -> subprocess.CompletedProcess[bytes]:
    """Run *argv* with captured bytes and an empty stdin (nothing can prompt)."""
    return subprocess.run(
        list(argv),
        cwd=cwd,
        env=None if env is None else dict(env),
        input=b"" if input_bytes is None else input_bytes,
        capture_output=True,
        timeout=timeout,
        check=False,
    )


#: The seam every external command goes through; tests replace it.
RUNNER: Runner = _run


class SetupRefused(Exception):
    """A setup step refused to act; the message says why and what the owner can do."""


@dataclass(frozen=True)
class SetupPaths:
    """The machine-side files the setup creates (the committed ``remote_config`` paths)."""

    anchor_key: Path = ANCHOR_KEY_PATH
    backup_key: Path = BACKUP_KEY_PATH
    known_hosts: Path = KNOWN_HOSTS_PATH
    ledger_dir: Path = LEDGER_DIR


DEFAULT_PATHS = SetupPaths()


@dataclass(frozen=True)
class DeployKey:
    """One deploy key and the one repository it may write."""

    label: str
    key_path: Path
    slug: str
    title: str

    @property
    def public_path(self) -> Path:
        return Path(f"{self.key_path}.pub")

    @property
    def comment(self) -> str:
        return f"nfl-predict ledger {self.label.lower()} deploy key for {self.slug}"


def _deploy_keys(paths: SetupPaths) -> tuple[DeployKey, DeployKey]:
    return (
        DeployKey("ANCHOR", paths.anchor_key, PUBLIC_REPO_SLUG, "ledger anchor (S4U)"),
        DeployKey("BACKUP", paths.backup_key, BACKUP_REPO_SLUG, "ledger backup (S4U)"),
    )


# One read-back row: (field, want, got, match).
Check = tuple[str, str, str, bool]


# ---------------------------------------------------------------------------------------------
# Output and command helpers
# ---------------------------------------------------------------------------------------------


def _emit(line: str) -> None:
    """One line on stdout. A line carrying private-key text is redacted, never printed."""
    if "PRIVATE KEY" in line:
        line = "[redacted: a line carrying private-key text was not printed]"
    sys.stdout.write(line + "\n")


def _emit_checks(checks: Sequence[Check], verdict_field: str) -> int:
    for field, want, got, match in checks:
        _emit(f"{field}= {want} | {got} | {'MATCH' if match else 'MISMATCH'}")
    matched = bool(checks) and all(check[3] for check in checks)
    _emit(f"{verdict_field}= {matched}")
    return 0 if matched else 1


def _text(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace").strip()


def _error_text(result: subprocess.CompletedProcess[bytes]) -> str:
    text = _text(result.stderr) or f"exit {result.returncode}"
    return " ".join(text.split())[:_ERROR_TEXT_LIMIT]


def _checked(argv: Sequence[str], *, cwd: Path | None = None) -> str:
    """Run a command that must succeed; its stdout. Raises :class:`SetupRefused` otherwise."""
    result = RUNNER(argv, cwd=cwd)
    if result.returncode != 0:
        msg = f"{' '.join(argv[:3])} failed: {_error_text(result)}"
        raise SetupRefused(msg)
    return _text(result.stdout)


def _gh_json(api_path: str) -> tuple[Any, str | None]:
    """``gh api <path>`` parsed; ``(None, error)`` when gh fails or the body is not JSON."""
    result = RUNNER(["gh", "api", api_path])
    if result.returncode != 0:
        return None, _error_text(result)
    try:
        return json.loads(result.stdout.decode("utf-8")), None
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        return None, f"not JSON: {error}"


def _git_runner(
    args: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    input_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[bytes]:
    """A ``forward_ledger.transport.GitRunner`` that routes ``git`` through :data:`RUNNER`."""
    return RUNNER(
        ["git", *args], cwd=cwd, env=env, timeout=timeout, input_bytes=input_bytes
    )


# ---------------------------------------------------------------------------------------------
# Keys, ACLs and host keys
# ---------------------------------------------------------------------------------------------


def _parse_fingerprints(listing: str) -> list[tuple[str, str]]:
    """``(type label, fingerprint)`` per ``ssh-keygen -l`` line, e.g. ``256 SHA256:x host (RSA)``."""
    pairs = []
    for line in listing.splitlines():
        tokens = line.split()
        if len(tokens) >= 3 and tokens[1].startswith("SHA256:"):
            pairs.append((tokens[-1].strip("()"), tokens[1]))
    return pairs


def _fingerprints(path: Path) -> list[tuple[str, str]] | None:
    """Sorted fingerprints of every key in a public-key or known_hosts *path*; None on failure."""
    result = RUNNER(["ssh-keygen", "-lf", str(path)])
    if result.returncode != 0:
        return None
    return sorted(_parse_fingerprints(_text(result.stdout)))


def _published_fingerprints() -> list[tuple[str, str]]:
    return sorted(GITHUB_HOST_KEY_FINGERPRINTS.items())


def _describe(pairs: Sequence[tuple[str, str]] | None) -> str:
    if pairs is None:
        return "unreadable"
    return "; ".join(f"{label} {fingerprint}" for label, fingerprint in pairs) or "none"


def _acl_entries(output: str, key_path: Path) -> list[str]:
    """The access entries ``icacls <key_path>`` lists, one per line, path prefix removed."""
    entries: list[str] = []
    lines = output.splitlines()
    for index, line in enumerate(lines):
        if not line.strip():
            break
        text = line
        if index == 0:
            position = line.casefold().find(key_path.name.casefold())
            if position != -1:
                text = line[position + len(key_path.name) :]
        if text.strip():
            entries.append(text.strip())
    return entries


def _owner_only(entries: Sequence[str]) -> bool:
    if len(entries) != 1:
        return False
    principal, _, rights = entries[0].partition(":")
    account = principal.rsplit("\\", maxsplit=1)[-1].casefold()
    return account == KEY_OWNER_ACCOUNT.casefold() and rights == "(F)"


def _key_checks(key: DeployKey) -> list[Check]:
    present = key.key_path.is_file() and key.public_path.is_file()
    got = (
        "present"
        if present
        else (
            f"private {'present' if key.key_path.is_file() else 'absent'}, "
            f"public {'present' if key.public_path.is_file() else 'absent'}"
        )
    )
    checks: list[Check] = [(f"{key.label}_KEY_FILES", "present", got, present)]
    if not key.key_path.is_file():
        return checks
    result = RUNNER(["icacls", str(key.key_path)])
    entries = (
        _acl_entries(_text(result.stdout), key.key_path)
        if result.returncode == 0
        else []
    )
    checks.append(
        (
            f"{key.label}_KEY_ACL",
            f"{KEY_OWNER_ACCOUNT}:(F) only",
            ", ".join(entries) if entries else f"unreadable ({_error_text(result)})",
            result.returncode == 0 and _owner_only(entries),
        )
    )
    return checks


def _public_fingerprint(key: DeployKey) -> str:
    pairs = _fingerprints(key.public_path) if key.public_path.is_file() else None
    return pairs[0][1] if pairs else "unreadable"


def _public_key_material(key: DeployKey) -> str | None:
    """``<type> <base64>`` of the PUBLIC key file (the form GitHub's keys API returns)."""
    if not key.public_path.is_file():
        return None
    tokens = key.public_path.read_text(encoding="ascii").split()
    return " ".join(tokens[:2]) if len(tokens) >= 2 else None


def _known_hosts_checks(path: Path) -> list[Check]:
    if not path.is_file():
        return [("KNOWN_HOSTS_FILE", "present", "absent", False)]
    hosts = sorted(
        {
            line.split()[0]
            for line in path.read_text(encoding="ascii").splitlines()
            if line.strip() and not line.startswith("#")
        }
    )
    observed = _fingerprints(path)
    published = _published_fingerprints()
    return [
        (
            "KNOWN_HOSTS_HOSTS",
            GITHUB_HOST,
            ",".join(hosts) or "none",
            hosts == [GITHUB_HOST],
        ),
        (
            "KNOWN_HOSTS_FINGERPRINTS",
            _describe(published),
            _describe(observed),
            observed == published,
        ),
    ]


def cmd_create_keys(paths: SetupPaths) -> int:
    """Generate both deploy keys, restrict each to the owner, print public fingerprints."""
    keys = _deploy_keys(paths)
    existing = [
        path.as_posix()
        for key in keys
        for path in (key.key_path, key.public_path)
        if path.exists()
    ]
    if existing:
        msg = (
            f"refusing to overwrite existing key file(s): {', '.join(existing)}. A key that "
            "exists may already be registered; remove it by hand (and its GitHub deploy key) "
            "only if it is to be replaced."
        )
        raise SetupRefused(msg)
    for key in keys:
        _checked(
            [
                "ssh-keygen",
                "-t",
                "ed25519",
                "-N",
                "",
                "-C",
                key.comment,
                "-f",
                str(key.key_path),
            ]
        )
        _checked(
            [
                "icacls",
                str(key.key_path),
                "/inheritance:r",
                "/grant:r",
                f"{KEY_OWNER_ACCOUNT}:(F)",
            ]
        )
        _emit(f"{key.label}_KEY_FINGERPRINT= {_public_fingerprint(key)}")
    return _emit_checks(
        [check for key in keys for check in _key_checks(key)], "STEP_READBACK_MATCH"
    )


def cmd_pin_host_keys(paths: SetupPaths) -> int:
    """Pin GitHub's host keys after checking every one against the published fingerprints."""
    meta, error = _gh_json("meta")
    keys = meta.get("ssh_keys") if isinstance(meta, dict) else None
    if not isinstance(keys, list) or not keys:
        msg = f"gh api meta returned no ssh_keys ({error or 'empty'})"
        raise SetupRefused(msg)
    content = "".join(f"{GITHUB_HOST} {key}\n" for key in keys)
    with tempfile.TemporaryDirectory() as scratch:
        candidate = Path(scratch) / "candidate_known_hosts"
        candidate.write_bytes(content.encode("ascii"))
        observed = _fingerprints(candidate)
    published = _published_fingerprints()
    if observed != published:
        msg = (
            "GitHub's served host keys do not fingerprint to the published set; nothing was "
            f"written. Published: {_describe(published)}. Served: {_describe(observed)}."
        )
        raise SetupRefused(msg)
    staging = paths.known_hosts.with_name(paths.known_hosts.name + ".tmp")
    staging.write_bytes(content.encode("ascii"))
    staging.replace(paths.known_hosts)
    return _emit_checks(_known_hosts_checks(paths.known_hosts), "STEP_READBACK_MATCH")


# ---------------------------------------------------------------------------------------------
# Repository and deploy keys
# ---------------------------------------------------------------------------------------------


def _visibility_check() -> Check:
    repo, error = _gh_json(f"repos/{BACKUP_REPO_SLUG}")
    got = repo.get("visibility") if isinstance(repo, dict) else None
    return (
        "BACKUP_REPO_VISIBILITY",
        "private",
        str(got) if got else f"unreadable ({error})",
        got == "private",
    )


def _registered(key: DeployKey) -> tuple[list[dict], str | None]:
    """The repository's deploy keys whose key material is *key*'s public key."""
    material = _public_key_material(key)
    listing, error = _gh_json(f"repos/{key.slug}/keys")
    if not isinstance(listing, list):
        return [], error or "not a list"
    return [entry for entry in listing if entry.get("key") == material], None


def _deploy_key_check(key: DeployKey) -> Check:
    want = f"{key.slug} read_only=False"
    if _public_key_material(key) is None:
        return (f"{key.label}_DEPLOY_KEY", want, "no public key file", False)
    entries, error = _registered(key)
    if error is not None:
        return (f"{key.label}_DEPLOY_KEY", want, f"unreadable ({error})", False)
    if len(entries) != 1:
        return (
            f"{key.label}_DEPLOY_KEY",
            want,
            f"registered {len(entries)} times",
            False,
        )
    entry = entries[0]
    got = f"id={entry.get('id')} read_only={entry.get('read_only')}"
    return (f"{key.label}_DEPLOY_KEY", want, got, entry.get("read_only") is False)


def cmd_create_backup_repo(_paths: SetupPaths) -> int:
    """Create the private backup repository and read its visibility back."""
    _checked(
        [
            "gh",
            "repo",
            "create",
            BACKUP_REPO_SLUG,
            "--private",
            "--description",
            BACKUP_REPO_DESCRIPTION,
        ]
    )
    return _emit_checks([_visibility_check()], "STEP_READBACK_MATCH")


def cmd_register_deploy_keys(paths: SetupPaths) -> int:
    """Add each key, write-scoped, to its one repository (skipping one already there)."""
    keys = _deploy_keys(paths)
    for key in keys:
        if _public_key_material(key) is None:
            msg = f"{key.public_path.as_posix()} is missing; run create-keys first"
            raise SetupRefused(msg)
        entries, error = _registered(key)
        if error is not None:
            msg = f"the deploy keys of {key.slug} could not be read: {error}"
            raise SetupRefused(msg)
        if entries:
            _emit(
                f"{key.label}_DEPLOY_KEY_ADD= already registered on {key.slug}; not added"
            )
            continue
        _checked(
            [
                "gh",
                "repo",
                "deploy-key",
                "add",
                str(key.public_path),
                "-R",
                key.slug,
                "--allow-write",
                "--title",
                key.title,
            ]
        )
    return _emit_checks([_deploy_key_check(key) for key in keys], "STEP_READBACK_MATCH")


# ---------------------------------------------------------------------------------------------
# Rulesets (public repository only)
# ---------------------------------------------------------------------------------------------


def master_ruleset_payload() -> dict[str, Any]:
    """The public ``master`` ruleset: admin role bypasses, a deploy key is refused (D-04)."""
    return {
        "name": MASTER_RULESET_NAME,
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": ["refs/heads/master"], "exclude": []}},
        "rules": [
            {"type": "update", "parameters": {"update_allows_fetch_and_merge": False}},
            {"type": "deletion"},
            {"type": "non_fast_forward"},
        ],
        "bypass_actors": [
            {
                "actor_id": ADMIN_REPOSITORY_ROLE_ID,
                "actor_type": "RepositoryRole",
                "bypass_mode": "always",
            }
        ],
    }


def anchor_ruleset_payload() -> dict[str, Any]:
    """The ``ledger-anchor`` ruleset: nobody may delete or force-push it (D-12)."""
    return {
        "name": ANCHOR_RULESET_NAME,
        "target": "branch",
        "enforcement": "active",
        "conditions": {"ref_name": {"include": [ANCHOR_REF], "exclude": []}},
        "rules": [{"type": "deletion"}, {"type": "non_fast_forward"}],
        "bypass_actors": [],
    }


@dataclass(frozen=True)
class RulesetSpec:
    """A committed ruleset: its read-back field prefix, branch and expected bypass for the owner."""

    prefix: str
    branch: str
    payload: Callable[[], dict[str, Any]]
    owner_can_bypass: str


MASTER_RULESET = RulesetSpec(
    "MASTER_RULESET", "master", master_ruleset_payload, "always"
)
ANCHOR_RULESET = RulesetSpec(
    "ANCHOR_RULESET",
    ANCHOR_REF.removeprefix("refs/heads/"),
    anchor_ruleset_payload,
    "never",
)


def _list_rulesets() -> tuple[list[dict] | None, str | None]:
    listing, error = _gh_json(f"repos/{PUBLIC_REPO_SLUG}/rulesets?per_page=100")
    if not isinstance(listing, list):
        return None, error or "not a list"
    return listing, None


def create_ruleset(payload: dict[str, Any], *, slug: str = PUBLIC_REPO_SLUG) -> int:
    """POST *payload* as a ruleset on the public repository; the new ruleset's id.

    Raises:
        SetupRefused: *slug* is not the public repository, the rulesets cannot be listed, a
            ruleset of the same name already exists, or the POST fails.
    """
    if slug != PUBLIC_REPO_SLUG:
        msg = (
            f"rulesets are created only on {PUBLIC_REPO_SLUG}; {slug} is the private backup, "
            "where rulesets need GitHub Pro -- it is protected by never force-pushing"
        )
        raise SetupRefused(msg)
    listing, error = _list_rulesets()
    if listing is None:
        msg = f"the rulesets of {slug} could not be listed: {error}"
        raise SetupRefused(msg)
    same = [entry for entry in listing if entry.get("name") == payload["name"]]
    if same:
        ids = ", ".join(str(entry.get("id")) for entry in same)
        msg = f"a ruleset named {payload['name']!r} already exists (id {ids}); not creating another"
        raise SetupRefused(msg)
    with tempfile.TemporaryDirectory() as scratch:
        body = Path(scratch) / "ruleset.json"
        body.write_text(json.dumps(payload, indent=2), encoding="ascii")
        created = _checked(
            ["gh", "api", "-X", "POST", f"repos/{slug}/rulesets", "--input", str(body)]
        )
    return int(json.loads(created)["id"])


# GitHub's read-back omits a rule parameter that holds its default (observed: an ``update`` rule
# POSTed with ``update_allows_fetch_and_merge: false`` reads back with no ``parameters``), so an
# omitted parameter is read as this default, and an explicit non-default value still mismatches.
_RULE_PARAMETER_DEFAULTS: dict[str, dict[str, Any]] = {
    "update": {"update_allows_fetch_and_merge": False},
}


def _rules_signature(rules: Sequence[dict] | None) -> str:
    if rules is None:
        return "unreported"
    parts = []
    for rule in rules:
        parameters = {
            **_RULE_PARAMETER_DEFAULTS.get(str(rule.get("type")), {}),
            **(rule.get("parameters") or {}),
        }
        detail = ",".join(f"{k}={json.dumps(v)}" for k, v in sorted(parameters.items()))
        parts.append(
            f"{rule.get('type')}({detail})" if detail else str(rule.get("type"))
        )
    return "; ".join(sorted(parts)) or "none"


def _actors_signature(actors: Sequence[dict] | None) -> str:
    if actors is None:
        return "unreported"
    parts = sorted(
        f"{actor.get('actor_type')}:{actor.get('actor_id')}:{actor.get('bypass_mode')}"
        for actor in actors
    )
    return "; ".join(parts) or "none"


def _ref_signature(ruleset: dict) -> str:
    ref_name = (ruleset.get("conditions") or {}).get("ref_name") or {}
    include = sorted(ref_name.get("include") or [])
    exclude = sorted(ref_name.get("exclude") or [])
    return f"include={json.dumps(include)} exclude={json.dumps(exclude)}"


def _ruleset_checks(spec: RulesetSpec, listing: Sequence[dict]) -> list[Check]:
    """Find *spec*'s ruleset by exact name, then compare it field by field with the payload."""
    payload = spec.payload()
    named = [entry for entry in listing if entry.get("name") == payload["name"]]
    checks: list[Check] = [
        (
            spec.prefix,
            f"1 named {payload['name']!r}",
            f"{len(named)} found",
            len(named) == 1,
        )
    ]
    if len(named) != 1:
        return checks
    ruleset_id = named[0].get("id")
    detail, error = _gh_json(f"repos/{PUBLIC_REPO_SLUG}/rulesets/{ruleset_id}")
    if not isinstance(detail, dict):
        checks.append(
            (
                f"{spec.prefix}_DETAIL",
                f"id {ruleset_id}",
                f"unreadable ({error})",
                False,
            )
        )
        return checks
    compared = (
        ("ENFORCEMENT", payload["enforcement"], str(detail.get("enforcement"))),
        ("TARGET", payload["target"], str(detail.get("target"))),
        ("REF_NAME", _ref_signature(payload), _ref_signature(detail)),
        (
            "RULES",
            _rules_signature(payload["rules"]),
            _rules_signature(detail.get("rules")),
        ),
        (
            "BYPASS_ACTORS",
            _actors_signature(payload["bypass_actors"]),
            _actors_signature(detail.get("bypass_actors")),
        ),
        (
            "CURRENT_USER_CAN_BYPASS",
            spec.owner_can_bypass,
            str(detail.get("current_user_can_bypass")),
        ),
    )
    checks.extend(
        (f"{spec.prefix}_{name}", want, got, want == got)
        for name, want, got in compared
    )
    deploy_key_actors = [
        actor
        for actor in detail.get("bypass_actors") or []
        if actor.get("actor_type") == "DeployKey"
    ]
    checks.append(
        (
            f"{spec.prefix}_DEPLOYKEY_ACTORS",
            "0",
            str(len(deploy_key_actors)),
            not deploy_key_actors,
        )
    )
    active, error = _gh_json(f"repos/{PUBLIC_REPO_SLUG}/rules/branches/{spec.branch}")
    want_types = sorted({rule["type"] for rule in payload["rules"]})
    if isinstance(active, list):
        got_types = sorted(
            {
                rule.get("type")
                for rule in active
                if rule.get("ruleset_id") == ruleset_id
            }
        )
        got_text = ",".join(got_types) or "none"
    else:
        got_types, got_text = [], f"unreadable ({error})"
    checks.append(
        (
            f"{spec.prefix}_ACTIVE_ON_BRANCH",
            f"{spec.branch}: {','.join(want_types)}",
            f"{spec.branch}: {got_text}",
            got_types == want_types,
        )
    )
    return checks


def _anchor_branch_on_remote() -> tuple[bool | None, str]:
    """Whether the public remote has ``ANCHOR_REF``; ``(None, error)`` when it cannot be read."""
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    result = RUNNER(
        [
            "git",
            "-c",
            "credential.helper=",
            "ls-remote",
            ANCHOR_REMOTE_HTTPS_URL,
            ANCHOR_REF,
        ],
        env=env,
    )
    if result.returncode != 0:
        return None, _error_text(result)
    present = any(
        line.split()[-1] == ANCHOR_REF
        for line in _text(result.stdout).splitlines()
        if line.split()
    )
    return present, "present" if present else "absent"


def _anchor_stage_checks(listing: Sequence[dict]) -> list[Check]:
    """The anchor ruleset is required exactly when the remote ledger-anchor branch exists."""
    present, state = _anchor_branch_on_remote()
    checks: list[Check] = [
        (
            "ANCHOR_BRANCH_ON_REMOTE",
            "readable",
            state if present is not None else f"unreachable ({state})",
            present is not None,
        )
    ]
    named = [entry for entry in listing if entry.get("name") == ANCHOR_RULESET_NAME]
    if present is False and not named:
        checks.append(
            (
                ANCHOR_RULESET.prefix,
                "not required before the first anchor push",
                "0 found",
                True,
            )
        )
        return checks
    return checks + _ruleset_checks(ANCHOR_RULESET, listing)


def _ruleset_command(spec: RulesetSpec) -> int:
    ruleset_id = create_ruleset(spec.payload())
    _emit(f"{spec.prefix}_ID= {ruleset_id}")
    listing, error = _list_rulesets()
    if listing is None:
        return _emit_checks(
            [(f"{spec.prefix}_LISTING", "readable", f"unreadable ({error})", False)],
            "STEP_READBACK_MATCH",
        )
    return _emit_checks(_ruleset_checks(spec, listing), "STEP_READBACK_MATCH")


def cmd_create_master_ruleset(_paths: SetupPaths) -> int:
    """Create the master ruleset (D-04) and read it back."""
    return _ruleset_command(MASTER_RULESET)


def cmd_create_anchor_ruleset(_paths: SetupPaths) -> int:
    """Create the ledger-anchor ruleset once the branch exists on the remote, and read it back."""
    present, state = _anchor_branch_on_remote()
    if present is not True:
        msg = (
            f"the remote has no {ANCHOR_REF} ({state}); create this ruleset right after the "
            "first anchor push (Plan 34-19)"
        )
        raise SetupRefused(msg)
    return _ruleset_command(ANCHOR_RULESET)


# ---------------------------------------------------------------------------------------------
# The nested backup repository
# ---------------------------------------------------------------------------------------------


def _ignored_by_public_repo(ledger_dir: Path) -> bool:
    result = RUNNER(
        ["git", "check-ignore", "-q", "--", f"{Path(ledger_dir).as_posix()}/"]
    )
    return result.returncode == 0


def _nested_repo_checks(ledger_dir: Path) -> list[Check]:
    ignored = _ignored_by_public_repo(ledger_dir)
    checks: list[Check] = [
        (
            "NESTED_REPO_IGNORED",
            "ignored by the public repo",
            "ignored" if ignored else "NOT ignored",
            ignored,
        )
    ]
    directory = Path(ledger_dir)
    if not (directory / ".git").exists():
        checks.append(("NESTED_REPO", "initialised", "absent", False))
        return checks

    def git_text(*args: str) -> str | None:
        result = RUNNER(["git", *args], cwd=directory)
        return _text(result.stdout) if result.returncode == 0 else None

    branch = git_text("rev-parse", "--abbrev-ref", "HEAD")
    checks.append(
        ("NESTED_REPO_BRANCH", NESTED_BRANCH, str(branch), branch == NESTED_BRANCH)
    )
    roots = (git_text("rev-list", "--max-parents=0", "HEAD") or "").split()
    if len(roots) != 1:
        checks.append(
            ("NESTED_REPO_FIRST_COMMIT", "1 root commit", f"{len(roots)} found", False)
        )
        return checks
    files = sorted(
        (git_text("show", "--name-only", "--format=", roots[0]) or "").split()
    )
    want_files = sorted(NESTED_FIRST_COMMIT_FILES)
    checks.append(
        (
            "NESTED_REPO_FIRST_COMMIT_FILES",
            ",".join(want_files),
            ",".join(files) or "none",
            files == want_files,
        )
    )
    attributes = git_text("show", f"{roots[0]}:.gitattributes")
    checks.append(
        (
            "NESTED_REPO_GITATTRIBUTES",
            NESTED_GITATTRIBUTES,
            str(attributes),
            attributes == NESTED_GITATTRIBUTES,
        )
    )
    return checks


def cmd_init_backup_repo(paths: SetupPaths) -> int:
    """Initialise ``ledger/`` as the nested backup repository, once."""
    if not _ignored_by_public_repo(paths.ledger_dir):
        msg = (
            f"{paths.ledger_dir.as_posix()}/ is not ignored by the public repository; a nested "
            "repository there would be recorded as an embedded gitlink"
        )
        raise SetupRefused(msg)
    created = ensure_backup_repo(paths.ledger_dir, runner=_git_runner)
    _emit(f"NESTED_REPO_CREATED= {created}")
    return _emit_checks(_nested_repo_checks(paths.ledger_dir), "STEP_READBACK_MATCH")


# ---------------------------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------------------------


def cmd_verify(paths: SetupPaths) -> int:
    """Read every created thing back and print ``SETUP_READBACK_MATCH=``. Read-only."""
    keys = _deploy_keys(paths)
    checks: list[Check] = []
    for key in keys:
        _emit(f"{key.label}_KEY_FINGERPRINT= {_public_fingerprint(key)}")
        checks += _key_checks(key)
    checks += _known_hosts_checks(paths.known_hosts)
    checks.append(_visibility_check())
    checks += [_deploy_key_check(key) for key in keys]
    listing, error = _list_rulesets()
    if listing is None:
        checks.append(("RULESETS", "readable", f"unreadable ({error})", False))
    else:
        checks += _ruleset_checks(MASTER_RULESET, listing)
        checks += _anchor_stage_checks(listing)
    checks += _nested_repo_checks(paths.ledger_dir)
    return _emit_checks(checks, "SETUP_READBACK_MATCH")


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------

COMMANDS: dict[str, tuple[Callable[[SetupPaths], int], str]] = {
    "create-keys": (cmd_create_keys, "generate both deploy keys with owner-only ACLs"),
    "pin-host-keys": (
        cmd_pin_host_keys,
        "pin GitHub's fingerprint-checked SSH host keys",
    ),
    "create-backup-repo": (
        cmd_create_backup_repo,
        f"create the private {BACKUP_REPO_SLUG}",
    ),
    "register-deploy-keys": (
        cmd_register_deploy_keys,
        "add each key, write-scoped, to its repo",
    ),
    "create-master-ruleset": (
        cmd_create_master_ruleset,
        "refuse deploy-key pushes to the public master (admin bypass only)",
    ),
    "create-anchor-ruleset": (
        cmd_create_anchor_ruleset,
        "forbid deleting or force-pushing ledger-anchor (after its first push)",
    ),
    "init-backup-repo": (
        cmd_init_backup_repo,
        "make ledger/ the nested backup repository",
    ),
    "verify": (cmd_verify, "read everything back; prints SETUP_READBACK_MATCH="),
}


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser: one subcommand per setup step, plus ``verify``."""
    parser = argparse.ArgumentParser(
        description=(
            "Set up the forward ledger's GitHub remotes, deploy keys and rulesets, with a "
            "field-by-field read-back (owner-run; Plan 34-13)"
        )
    )
    subcommands = parser.add_subparsers(
        dest="command", required=True, metavar="SUBCOMMAND"
    )
    for name, (_handler, summary) in COMMANDS.items():
        subcommands.add_parser(name, help=summary, description=summary)
    return parser


def main(argv: list[str] | None = None, *, paths: SetupPaths = DEFAULT_PATHS) -> int:
    """CLI entry point. Exit 0 on a full match; 1 on a mismatch or a refusal."""
    args = build_parser().parse_args(argv)
    handler, _summary = COMMANDS[args.command]
    try:
        return handler(paths)
    except SetupRefused as refusal:
        _emit(f"SETUP_REFUSED= {refusal}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
