#!/usr/bin/env python3
"""The S4U push smoke: prove the unattended deploy-key pushes before go-live (Plan 34-18, D-03, D-04).

WHY THIS EXISTS
---------------
The daily run executes as a Task Scheduler S4U logon, which has no access to Git Credential
Manager or DPAPI. The transport (:mod:`forward_ledger.transport`) pins each push to a deploy-key
file, but research could only prove it from the owner's interactive session. This module proves
it from the REAL S4U context, with the exact transport the daily run will use
(:func:`forward_ledger.transport.git_push_env`), and checks that GitHub greets each key as the
REPOSITORY -- a push that fell back to the owner's personal key would be an admin push that
bypasses the master ruleset (D-04).

FOUR MODES
----------
* ``--register`` (owner): write a trigger-less task definition built from the DAILY task's
  principal and settings (``deployment/windows_scheduler.xml``) whose action is
  ``uv.exe run python -m scripts.ledger_push_smoke --run``; register it as
  ``NFL_Ledger_Push_Smoke`` and start it once with ``schtasks /run``.
* ``--run`` (executed BY that task, under S4U): record ``whoami``; run ``ssh -T git@github.com``
  with each deploy key through the pinned transport and record the greeting; commit a timestamp by
  plumbing on ``refs/heads/ledger-anchor-smoke`` and push exactly that ref with the anchor key;
  push the backup repository's existing first commit with the backup key
  (:func:`forward_ledger.backup.push_backup`); create, export and delete a disabled dummy task
  ``NFL_Ledger_Smoke_Dummy`` (can the S4U context register scheduled tasks at all? 34-RESEARCH
  A1). Writes ``logs/ledger_push_smoke.json`` and never raises.
* ``--readback`` (owner): wait for the result file, then read GitHub back -- ``git ls-remote``
  of the smoke branch against the smoke commit, ``gh api`` of the backup's ``main`` against the
  local ``ledger/`` HEAD -- and print ``SMOKE_*=`` lines ending in ``SMOKE_RESULT= PASS|FAIL``.
* ``--cleanup`` (owner): delete the remote smoke branch, the local smoke ref and both tasks, and
  confirm each is gone. Run it after EVERY read-back outcome.

WHAT IT CAN NEVER TOUCH
-----------------------
The anchor push vector is checked by :func:`check_smoke_push_args` before git runs: one URL (the
public repository) and exactly ``refs/heads/ledger-anchor-smoke:refs/heads/ledger-anchor-smoke``.
``master`` and ``refs/heads/ledger-anchor`` (undeletable once its ruleset exists) are refused, as
is any ``+`` refspec or option but ``--porcelain``. Every ``schtasks`` call names one of the two
smoke tasks only (:func:`_check_task_name`); ``NFL_Predict_Pipeline`` and ``NFL_Predict_Closing``
are never named.

Every external command goes through the module-level ``RUNNER`` seam, so the unit tests touch no
network, no key and no scheduler. No private key is ever read or printed.

Usage:
    uv run python -m scripts.ledger_push_smoke --register | --run | --readback | --cleanup

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from forward_ledger.backup import push_backup
from forward_ledger.remote_config import (
    ANCHOR_REMOTE_HTTPS_URL,
    ANCHOR_REMOTE_URL,
    ANCHOR_SPEC,
    BACKUP_REPO_SLUG,
    BACKUP_SPEC,
    PUBLIC_REPO_SLUG,
    PUSH_TIMEOUT_SECONDS,
    RemoteSpec,
)
from forward_ledger.store import LEDGER_DIR
from forward_ledger.transport import (
    git_identity_env,
    git_push_env,
    resolve_ref,
    run_checked,
)

__all__ = [
    "DUMMY_TASK_NAME",
    "RUNNER",
    "SMOKE_REF",
    "SMOKE_REFSPEC",
    "SMOKE_TASK_NAME",
    "SmokePaths",
    "SmokeRefspecError",
    "check_smoke_push_args",
    "classify_greeting",
    "cleanup_smoke",
    "main",
    "readback_smoke",
    "register_smoke",
    "run_smoke",
]

REPO_ROOT = Path(__file__).resolve().parents[1]

SMOKE_TASK_NAME = "NFL_Ledger_Push_Smoke"
DUMMY_TASK_NAME = "NFL_Ledger_Smoke_Dummy"
_SMOKE_TASK_NAMES = frozenset({SMOKE_TASK_NAME, DUMMY_TASK_NAME})

# The throwaway branch on the PUBLIC repository; deleted again by --cleanup.
SMOKE_REF = "refs/heads/ledger-anchor-smoke"
SMOKE_REFSPEC = f"{SMOKE_REF}:{SMOKE_REF}"
_SMOKE_FILE_NAME = "SMOKE"

SMOKE_TASK_ARGUMENTS = "run python -m scripts.ledger_push_smoke --run"

# The dummy task is disabled, its one trigger is disabled and years away, and it runs nothing.
_DUMMY_TRIGGER_START = "2030-01-01T03:00:00"
_DUMMY_TRIGGER_END = "2030-01-01T03:30:00"
_DUMMY_COMMAND = "C:\\Windows\\System32\\cmd.exe"
_DUMMY_ARGUMENTS = "/c exit 0"

READBACK_WAIT_SECONDS = 300
_POLL_SECONDS = 5
_COMMAND_TIMEOUT_SECONDS = 120
_ERROR_TEXT_LIMIT = 500

_TASK_NS = "http://schemas.microsoft.com/windows/2004/02/mit/task"
_NS = f"{{{_TASK_NS}}}"
_XML_DECLARATION = '<?xml version="1.0" encoding="UTF-16"?>\n'

# GitHub's ``ssh -T`` answer: ``Hi <user>!`` for a user key, ``Hi <owner>/<repo>!`` for a
# deploy key. It exits 1 even on success, because GitHub provides no shell.
_GREETING_RE = re.compile(r"Hi ([^\s!]+)! You've successfully authenticated")
_SSH_FAILURE_EXIT = 255


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


class SmokeRefspecError(Exception):
    """A smoke push named anything but the one smoke refspec on the public repository."""


@dataclass(frozen=True)
class SmokePaths:
    """The files and remotes one smoke uses; the defaults are the real ones."""

    repo_dir: Path = REPO_ROOT
    ledger_dir: Path = REPO_ROOT / LEDGER_DIR
    result_path: Path = REPO_ROOT / "logs" / "ledger_push_smoke.json"
    task_xml_path: Path = REPO_ROOT / "logs" / "ledger_smoke_task.xml"
    dummy_xml_path: Path = REPO_ROOT / "logs" / "ledger_smoke_dummy_task.xml"
    template_path: Path = REPO_ROOT / "deployment" / "windows_scheduler.xml"
    anchor_spec: RemoteSpec = field(default=ANCHOR_SPEC)
    backup_spec: RemoteSpec = field(default=BACKUP_SPEC)


DEFAULT_PATHS = SmokePaths()


def _emit(line: str) -> None:
    sys.stdout.write(line + "\n")


def _text(raw: bytes | None) -> str:
    return (raw or b"").decode("utf-8", errors="replace")


def _error_text(result: subprocess.CompletedProcess[bytes]) -> str:
    text = (_text(result.stderr) + _text(result.stdout)).strip()
    return text[:_ERROR_TEXT_LIMIT] or f"exited {result.returncode}"


def _decode_task_xml(raw: bytes) -> str:
    """A task definition: UTF-16 with a BOM (as schtasks writes it), else UTF-8."""
    if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return raw.decode("utf-16")
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------------------------
# Greeting and refspec rules
# ---------------------------------------------------------------------------------------------


def classify_greeting(
    exit_code: int | None, output: str, expected_slug: str
) -> dict[str, Any]:
    """Judge one ``ssh -T git@github.com`` by its GREETING, not by its exit status.

    GitHub exits 1 on a successful ``ssh -T`` (it provides no shell), so exit 1 with
    ``Hi <expected_slug>!`` proves the deploy key's identity. ``Hi <user>!`` is the owner's
    personal key (the identity that must never push); exit 255 or no greeting is a failure.
    """
    match = _GREETING_RE.search(output)
    name = match.group(1) if match else None
    kind = None if name is None else ("deploy_key" if "/" in name else "user")
    identity_ok = (
        name == expected_slug
        and exit_code is not None
        and exit_code != _SSH_FAILURE_EXIT
    )
    error = None
    if not identity_ok:
        error = output.strip()[:_ERROR_TEXT_LIMIT] or f"no greeting (exit {exit_code})"
    return {
        "exit_code": exit_code,
        "greeting": match.group(0) if match else None,
        "identity_name": name,
        "identity_kind": kind,
        "expected_identity": expected_slug,
        "identity_ok": identity_ok,
        "error": error,
    }


def check_smoke_push_args(push_args: Sequence[str]) -> None:
    """Refuse any push but ``push --porcelain <public SSH URL> <SMOKE_REFSPEC>``.

    Raises:
        SmokeRefspecError: any other option, URL, refspec or argument count.
    """
    expected = ["push", "--porcelain", ANCHOR_REMOTE_URL, SMOKE_REFSPEC]
    if list(push_args) != expected:
        msg = f"a smoke push must be exactly {expected}; refused {list(push_args)}"
        raise SmokeRefspecError(msg)


def _check_task_name(name: str) -> str:
    """Only the two smoke tasks may be created, run or deleted by this module."""
    if name not in _SMOKE_TASK_NAMES:
        msg = f"the smoke never touches the task {name!r}"
        raise ValueError(msg)
    return name


# ---------------------------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------------------------


def _child(parent: ET.Element, name: str) -> ET.Element:
    element = parent.find(f"{_NS}{name}")
    if element is None:
        msg = f"the task template has no <{name}> element"
        raise ValueError(msg)
    return element


def _serialize_task(root: ET.Element) -> bytes:
    """*root* as Task Scheduler writes a definition: UTF-16 LE with a BOM and CRLF."""
    ET.indent(root, space="  ")
    ET.register_namespace("", _TASK_NS)
    body = ET.tostring(root, encoding="unicode")
    text = (_XML_DECLARATION + body + "\n").replace("\r\n", "\n").replace("\n", "\r\n")
    return b"\xff\xfe" + text.encode("utf-16-le")


def _template_root(template_bytes: bytes) -> ET.Element:
    """The daily task's definition with its triggers removed."""
    root = ET.fromstring(_decode_task_xml(template_bytes).strip())
    triggers = root.find(f"{_NS}Triggers")
    if triggers is not None:
        root.remove(triggers)
    _child(root, "RegistrationInfo")
    return root


def _set_action(root: ET.Element, command: str | None, arguments: str) -> None:
    exec_element = _child(_child(root, "Actions"), "Exec")
    if command is not None:
        _child(exec_element, "Command").text = command
    _child(exec_element, "Arguments").text = arguments


def _set_description(root: ET.Element, text: str) -> None:
    _child(_child(root, "RegistrationInfo"), "Description").text = text


def build_smoke_task_xml(template_bytes: bytes) -> bytes:
    """The daily task's principal and settings, no trigger, and the smoke ``--run`` action."""
    root = _template_root(template_bytes)
    _set_description(
        root, "One-off S4U deploy-key push smoke (Plan 34-18); deleted after use"
    )
    _set_action(root, None, SMOKE_TASK_ARGUMENTS)
    return _serialize_task(root)


def build_dummy_task_xml(template_bytes: bytes) -> bytes:
    """A disabled task with one disabled, far-future trigger that runs nothing (A1 probe)."""
    root = _template_root(template_bytes)
    _set_description(
        root, "S4U scheduler probe (Plan 34-18); created and deleted at once"
    )
    triggers = ET.Element(f"{_NS}Triggers")
    trigger = ET.SubElement(triggers, f"{_NS}TimeTrigger")
    ET.SubElement(trigger, f"{_NS}StartBoundary").text = _DUMMY_TRIGGER_START
    ET.SubElement(trigger, f"{_NS}EndBoundary").text = _DUMMY_TRIGGER_END
    ET.SubElement(trigger, f"{_NS}Enabled").text = "false"
    # The schema orders Triggers right after RegistrationInfo.
    root.insert(list(root).index(_child(root, "RegistrationInfo")) + 1, triggers)
    settings = _child(root, "Settings")
    _child(settings, "Enabled").text = "false"
    _child(settings, "WakeToRun").text = "false"
    _set_action(root, _DUMMY_COMMAND, _DUMMY_ARGUMENTS)
    return _serialize_task(root)


# ---------------------------------------------------------------------------------------------
# --run (inside the S4U task)
# ---------------------------------------------------------------------------------------------


def _whoami() -> str:
    result = RUNNER(["whoami"])
    if result.returncode != 0:
        msg = f"whoami failed: {_error_text(result)}"
        raise RuntimeError(msg)
    return _text(result.stdout).strip()


def _greeting(spec: RemoteSpec, expected_slug: str) -> dict[str, Any]:
    """``ssh -T git@github.com`` with *spec*'s key, through the push transport's own command."""
    command = git_push_env(spec)["GIT_SSH_COMMAND"]
    argv = [*shlex.split(command), "-T", "git@github.com"]
    result = RUNNER(argv, timeout=PUSH_TIMEOUT_SECONDS)
    output = _text(result.stderr) + _text(result.stdout)
    return classify_greeting(result.returncode, output, expected_slug)


def _smoke_commit(repo_dir: Path, smoke_at: str) -> str:
    """Commit ``smoke_at=<ISO>`` by plumbing on :data:`SMOKE_REF`; never touch the checkout."""
    content = f"smoke_at={smoke_at}\n".encode("ascii")
    env = git_identity_env()
    blob = run_checked(
        _git_runner, ["hash-object", "-w", "--stdin"], cwd=repo_dir, input_bytes=content
    )
    tree_line = f"100644 blob {blob}\t{_SMOKE_FILE_NAME}\n".encode("ascii")
    tree = run_checked(_git_runner, ["mktree"], cwd=repo_dir, input_bytes=tree_line)
    parent = resolve_ref(repo_dir, SMOKE_REF, runner=_git_runner)
    parent_args = ["-p", parent] if parent is not None else []
    sha = run_checked(
        _git_runner,
        ["commit-tree", tree, *parent_args, "-m", f"ledger push smoke: {smoke_at}"],
        cwd=repo_dir,
        env=env,
    )
    run_checked(
        _git_runner,
        ["update-ref", "-m", "ledger push smoke", SMOKE_REF, sha],
        cwd=repo_dir,
        env=env,
    )
    return sha


def _anchor_push(paths: SmokePaths, smoke_at: str) -> dict[str, Any]:
    push_args = ["push", "--porcelain", paths.anchor_spec.url, SMOKE_REFSPEC]
    check_smoke_push_args(push_args)
    env = git_push_env(paths.anchor_spec)
    sha = _smoke_commit(paths.repo_dir, smoke_at)
    result = _git_runner(
        push_args, cwd=paths.repo_dir, env=env, timeout=PUSH_TIMEOUT_SECONDS
    )
    ok = result.returncode == 0
    return {
        "ok": ok,
        "smoke_commit": sha,
        "exit_code": result.returncode,
        "error": None if ok else _error_text(result),
    }


def _backup_push(paths: SmokePaths) -> dict[str, Any]:
    outcome = push_backup(paths.ledger_dir, spec=paths.backup_spec, runner=_git_runner)
    return {"ok": outcome.ok, "pushed_sha": outcome.pushed_sha, "error": outcome.error}


def _schtasks(*args: str, name: str) -> subprocess.CompletedProcess[bytes]:
    return RUNNER(["schtasks", args[0], "/tn", _check_task_name(name), *args[1:]])


def _dummy_task(paths: SmokePaths) -> dict[str, Any]:
    """Create, export and delete the disabled dummy task: may S4U register tasks at all (A1)?"""
    generated = build_dummy_task_xml(paths.template_path.read_bytes())
    paths.dummy_xml_path.parent.mkdir(parents=True, exist_ok=True)
    paths.dummy_xml_path.write_bytes(generated)
    steps: dict[str, Any] = {}

    created = _schtasks(
        "/create", "/xml", str(paths.dummy_xml_path), "/f", name=DUMMY_TASK_NAME
    )
    steps["create_ok"] = created.returncode == 0
    steps["create_error"] = None if steps["create_ok"] else _error_text(created)

    exported = _schtasks("/query", "/xml", name=DUMMY_TASK_NAME)
    readback = _decode_task_xml(exported.stdout) if exported.returncode == 0 else ""
    steps["readback_ok"] = exported.returncode == 0 and _DUMMY_TRIGGER_START in readback
    steps["readback_error"] = None if steps["readback_ok"] else _error_text(exported)

    deleted = _schtasks("/delete", "/f", name=DUMMY_TASK_NAME)
    steps["delete_ok"] = deleted.returncode == 0
    steps["delete_error"] = None if steps["delete_ok"] else _error_text(deleted)

    steps["ok"] = steps["create_ok"] and steps["readback_ok"] and steps["delete_ok"]
    return steps


def _write_result(path: Path, result: Mapping[str, Any]) -> None:
    """Write the result file whole (temp file, then replace), so a reader never sees half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="ascii"
    )
    temporary.replace(path)


def run_smoke(
    paths: SmokePaths = DEFAULT_PATHS, *, now: Callable[[], datetime] | None = None
) -> dict[str, Any]:
    """The smoke itself (``--run``). Every step is recorded; none raises."""
    clock = now or (lambda: datetime.now(UTC))
    smoke_at = clock().isoformat(timespec="seconds")
    errors: dict[str, str] = {}
    failed: dict[str, Any] = {"ok": False}

    def step(name: str, action: Callable[[], Any], fallback: Any) -> Any:
        try:
            return action()
        except Exception as error:  # noqa: BLE001 -- recorded in the result, never raised
            errors[name] = f"{type(error).__name__}: {error}"[:_ERROR_TEXT_LIMIT]
            return fallback

    result: dict[str, Any] = {"started_at": smoke_at}
    result["whoami"] = step("whoami", _whoami, None)
    result["greetings"] = {
        "anchor": step(
            "anchor_greeting",
            lambda: _greeting(paths.anchor_spec, PUBLIC_REPO_SLUG),
            {"identity_ok": False},
        ),
        "backup": step(
            "backup_greeting",
            lambda: _greeting(paths.backup_spec, BACKUP_REPO_SLUG),
            {"identity_ok": False},
        ),
    }
    result["anchor_push"] = step(
        "anchor_push", lambda: _anchor_push(paths, smoke_at), failed
    )
    result["backup_push"] = step("backup_push", lambda: _backup_push(paths), failed)
    result["dummy_task"] = step("dummy_task", lambda: _dummy_task(paths), failed)
    for name in ("anchor_push", "backup_push"):
        if name not in errors and not result[name]["ok"]:
            errors[name] = str(result[name].get("error"))
    if "dummy_task" not in errors and not result["dummy_task"]["ok"]:
        errors["dummy_task"] = "; ".join(
            str(result["dummy_task"][key])
            for key in ("create_error", "readback_error", "delete_error")
            if result["dummy_task"][key]
        )
    result["errors"] = errors
    result["finished_at"] = clock().isoformat(timespec="seconds")
    try:
        _write_result(paths.result_path, result)
    except Exception as error:  # noqa: BLE001 -- the task's own log is the last resort
        _emit(f"SMOKE_RESULT_NOT_WRITTEN= {error}")
    return result


# ---------------------------------------------------------------------------------------------
# --register, --readback, --cleanup (owner context)
# ---------------------------------------------------------------------------------------------


def _remote_sha(ref: str) -> str | None:
    """The commit *ref* points at on the public repository (anonymous read), or None."""
    result = RUNNER(["git", "ls-remote", ANCHOR_REMOTE_HTTPS_URL, ref])
    if result.returncode != 0:
        msg = f"git ls-remote {ref} failed: {_error_text(result)}"
        raise RuntimeError(msg)
    for line in _text(result.stdout).splitlines():
        sha, _, name = line.partition("\t")
        if name.strip() == ref:
            return sha.strip()
    return None


def register_smoke(paths: SmokePaths = DEFAULT_PATHS) -> int:
    """Register the trigger-less smoke task and start it once. Exit 0 when both succeeded."""
    _emit(f"SMOKE_MASTER_SHA= {_remote_sha('refs/heads/master')}")
    paths.result_path.unlink(missing_ok=True)  # never read back a stale result
    paths.task_xml_path.parent.mkdir(parents=True, exist_ok=True)
    paths.task_xml_path.write_bytes(
        build_smoke_task_xml(paths.template_path.read_bytes())
    )

    created = _schtasks(
        "/create", "/xml", str(paths.task_xml_path), "/f", name=SMOKE_TASK_NAME
    )
    _emit(f"SMOKE_TASK_CREATED= {created.returncode == 0}")
    if created.returncode != 0:
        _emit(f"SMOKE_TASK_CREATE_ERROR= {_error_text(created)}")
        return 1
    started = _schtasks("/run", name=SMOKE_TASK_NAME)
    _emit(f"SMOKE_TASK_STARTED= {started.returncode == 0}")
    if started.returncode != 0:
        _emit(f"SMOKE_TASK_RUN_ERROR= {_error_text(started)}")
        return 1
    return 0


def _wait_for_result(
    path: Path, wait_seconds: float, sleep: Callable[[float], None]
) -> dict[str, Any] | None:
    deadline = time.monotonic() + wait_seconds
    while True:
        if path.exists():
            return json.loads(path.read_text(encoding="ascii"))
        if time.monotonic() >= deadline:
            return None
        sleep(_POLL_SECONDS)


def readback_smoke(
    paths: SmokePaths = DEFAULT_PATHS,
    *,
    wait_seconds: float = READBACK_WAIT_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Read GitHub back from the owner context. Exit 0 on ``SMOKE_RESULT= PASS``."""
    result = _wait_for_result(paths.result_path, wait_seconds, sleep)
    if result is None:
        _emit(f"SMOKE_READBACK_ERROR= no result file within {wait_seconds:g} s")
        _emit("SMOKE_RESULT= FAIL")
        return 1

    anchor = result.get("anchor_push", {})
    backup = result.get("backup_push", {})
    greetings = result.get("greetings", {})
    remote_smoke = _remote_sha(SMOKE_REF)
    remote_backup = _text(
        RUNNER(
            ["gh", "api", f"repos/{BACKUP_REPO_SLUG}/commits/main", "--jq", ".sha"]
        ).stdout
    ).strip()
    local_backup = _text(
        RUNNER(["git", "-C", str(paths.ledger_dir), "rev-parse", "HEAD"]).stdout
    ).strip()

    flags = {
        "ANCHOR_IDENTITY_OK": greetings.get("anchor", {}).get("identity_ok") is True,
        "BACKUP_IDENTITY_OK": greetings.get("backup", {}).get("identity_ok") is True,
        "ANCHOR_PUSH_OK": anchor.get("ok") is True
        and remote_smoke is not None
        and remote_smoke == anchor.get("smoke_commit"),
        "BACKUP_PUSH_OK": backup.get("ok") is True
        and bool(remote_backup)
        and remote_backup == local_backup,
    }
    _emit(f"SMOKE_WHOAMI= {result.get('whoami')}")
    for label in ("anchor", "backup"):
        _emit(
            f"SMOKE_{label.upper()}_GREETING= {greetings.get(label, {}).get('greeting')}"
        )
    _emit(f"SMOKE_ANCHOR_COMMIT= {anchor.get('smoke_commit')} | remote {remote_smoke}")
    _emit(f"SMOKE_BACKUP_COMMIT= {local_backup} | remote {remote_backup or None}")
    _emit(f"SMOKE_MASTER_SHA= {_remote_sha('refs/heads/master')}")
    for name, value in flags.items():
        _emit(f"SMOKE_{name}= {value}")
    _emit(
        f"SMOKE_SCHTASKS_FROM_S4U_OK= {result.get('dummy_task', {}).get('ok') is True}"
    )
    for name, text in sorted(result.get("errors", {}).items()):
        _emit(f"SMOKE_STEP_ERROR= {name}: {text}")
    passed = all(flags.values())
    _emit(f"SMOKE_RESULT= {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


def cleanup_smoke(paths: SmokePaths = DEFAULT_PATHS) -> int:
    """Delete the remote smoke branch, the local smoke ref and both tasks; confirm each is gone."""
    gone: dict[str, bool] = {}

    deleted = RUNNER(
        ["gh", "api", "-X", "DELETE", f"repos/{PUBLIC_REPO_SLUG}/git/{SMOKE_REF}"]
    )
    if deleted.returncode != 0:
        _emit(f"SMOKE_CLEANUP_REMOTE_DELETE= {_error_text(deleted)}")
    gone["REMOTE_BRANCH"] = _remote_sha(SMOKE_REF) is None

    local = resolve_ref(paths.repo_dir, SMOKE_REF, runner=_git_runner)
    if local is not None:
        removed = _git_runner(
            ["update-ref", "-d", SMOKE_REF, local], cwd=paths.repo_dir
        )
        gone["LOCAL_REF"] = removed.returncode == 0
    else:
        gone["LOCAL_REF"] = True

    for label, name in (
        ("SMOKE_TASK", SMOKE_TASK_NAME),
        ("DUMMY_TASK", DUMMY_TASK_NAME),
    ):
        _schtasks(
            "/delete", "/f", name=name
        )  # absent already is fine; the query decides
        gone[label] = _schtasks("/query", name=name).returncode != 0

    for label, value in gone.items():
        _emit(f"SMOKE_CLEANUP_{label}_GONE= {value}")
    ok = all(gone.values())
    _emit(f"SMOKE_CLEANUP_OK= {ok}")
    return 0 if ok else 1


# ---------------------------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prove the unattended S4U deploy-key pushes before go-live (Plan 34-18): only "
            "refs/heads/ledger-anchor-smoke and the backup's first commit are ever pushed"
        )
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument(
        "--run", action="store_true", help="the smoke itself (run BY the S4U task)"
    )
    modes.add_argument(
        "--register",
        action="store_true",
        help=f"owner: register {SMOKE_TASK_NAME} (no trigger) and start it once",
    )
    modes.add_argument(
        "--readback",
        action="store_true",
        help="owner: wait for the result, read GitHub back, print SMOKE_RESULT=",
    )
    modes.add_argument(
        "--cleanup",
        action="store_true",
        help="owner: delete the smoke branch, the local ref and both tasks",
    )
    return parser


def main(argv: list[str] | None = None, *, paths: SmokePaths = DEFAULT_PATHS) -> int:
    """CLI entry point. ``--run`` always exits 0; the owner modes exit 1 on a failure."""
    args = build_parser().parse_args(argv)
    if args.run:
        run_smoke(paths)
        return 0
    if args.register:
        return register_smoke(paths)
    if args.readback:
        return readback_smoke(paths)
    return cleanup_smoke(paths)


if __name__ == "__main__":
    sys.exit(main())
