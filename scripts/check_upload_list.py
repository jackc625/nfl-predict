#!/usr/bin/env python3
"""Judge a Vercel upload list, and the committed ``.vercelignore``, against SPEC R2.

WHAT IT PROVES
--------------
Only files the hosted app needs may ever reach Vercel. Before any deploy (Plan 36-09) the CLI's own
upload list is written to a file and this command judges it:

- every path is one the ``.vercelignore`` allowlist admits: one of :data:`ALLOWED_ROOT_FILES`,
  ``data/web_cache.duckdb``, or a file under one of :data:`ALLOWED_DIRS` -- anything else (another
  folder, another root file, ``.git/``, ``.vercel/``, ``tools/``) is outside the allowlist;
- ``data/web_cache.duckdb`` is the only file under ``data/`` (its temp, WAL and backup siblings are
  forbidden like any other data file);
- nothing anywhere that the allowlist's deny rules exclude: a ``.env*`` file, a ``__pycache__/``
  file, a ``*.tmp.duckdb``, ``*.wal``, ``*.bak``, ``*.pem`` or ``*.key`` file;
- no ``pyproject.toml`` or ``uv.lock`` (either would make Vercel install the full local stack
  instead of the hosted ``requirements.txt``);
- ``index.py``, ``requirements.txt`` and ``data/web_cache.duckdb`` are present.

``allowlist_problems`` keeps the committed ``.vercelignore`` shaped as an allowlist; the committed
test ``tests/unit/test_vercel_project_files.py`` runs it.

The list file holds one path per line, or a JSON array of paths. Backslashes are read as forward
slashes, so a Windows list judges the same as a POSIX one.

Output: one ``UPLOAD_LIST_VIOLATION= <path>: <reason>`` line per violation, or
``UPLOAD_LIST= ok <n> files``. An unreadable list prints ``UPLOAD_LIST= refused: <error class>``.

Exit codes: 0 the list is clean; 1 any violation, or the list could not be read.

Usage:
    uv run python -m scripts.check_upload_list <list-file>

No module under ``api/`` may import this script: the request path never judges its own upload.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Sequence
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parent.parent

CACHE_FILE = "data/web_cache.duckdb"

# What the .vercelignore allowlist admits: these root files, the cache, and these folders' files.
ALLOWED_ROOT_FILES: frozenset[str] = frozenset(
    {"index.py", "requirements.txt", ".python-version", "vercel.json"}
)
ALLOWED_DIRS: frozenset[str] = frozenset({"api", "web", "utils", "conf"})
# File names its deny rules exclude anywhere, even inside an admitted folder.
DENIED_NAME_PATTERNS: tuple[str, ...] = (
    ".env*",
    "*.tmp.duckdb",
    "*.wal",
    "*.bak",
    "*.pem",
    "*.key",
)
FORBIDDEN_FILES: frozenset[str] = frozenset({"pyproject.toml", "uv.lock"})
REQUIRED_FILES: tuple[str, ...] = ("index.py", "requirements.txt", CACHE_FILE)

# The exact names .vercelignore may re-admit, and the deny rules that must follow the last one.
ALLOWED_REINCLUDES: frozenset[str] = (
    ALLOWED_ROOT_FILES | ALLOWED_DIRS | {"data", CACHE_FILE}
)
REQUIRED_DENY_RULES: tuple[str, ...] = (
    "**/.env*",
    "**/__pycache__/",
    "*.tmp.duckdb",
    "*.wal",
    "*.bak",
    "*.pem",
    "*.key",
)

_ROOT_RULE = "/*"
_DATA_DIR_RULE = "!data"
_DATA_CONTENTS_RULE = "data/*"
_CACHE_RULE = f"!{CACHE_FILE}"
_WILDCARDS = ("*", "?", "[")


def _normalize(path: str) -> str:
    """*path* as a repo-relative POSIX path (backslashes read as slashes, root prefix removed)."""
    posix = path.strip().replace("\\", "/")
    root = REPO_ROOT.as_posix().rstrip("/") + "/"
    if posix.casefold().startswith(root.casefold()):
        posix = posix[len(root) :]
    while posix.startswith("./"):
        posix = posix[2:]
    return posix.lstrip("/")


def _path_reasons(path: str) -> list[str]:
    """Every R2 rule *path* breaks, as reason texts."""
    parts = PurePosixPath(path).parts
    reasons: list[str] = []
    name = parts[-1] if parts else ""
    if parts and parts[0] == "data":
        if path != CACHE_FILE:
            reasons.append(f"a data/ file other than {CACHE_FILE}")
    elif not (
        path in ALLOWED_ROOT_FILES or (len(parts) > 1 and parts[0] in ALLOWED_DIRS)
    ):
        reasons.append("outside the upload allowlist")
    denied = [pattern for pattern in DENIED_NAME_PATTERNS if fnmatch(name, pattern)]
    if denied:
        reasons.append(f"a denied file ({denied[0]})")
    if "__pycache__" in parts:
        reasons.append("a __pycache__ file")
    if name in FORBIDDEN_FILES:
        reasons.append(
            f"{name} must not be uploaded (Vercel would install the full local stack)"
        )
    return reasons


def upload_list_violations(paths: Iterable[str]) -> list[str]:
    """One ``"<path>: <reason>"`` per rule broken by *paths*, plus one per missing required file."""
    normalized = [p for p in (_normalize(str(raw)) for raw in paths) if p]
    violations = [
        f"{path}: {reason}" for path in normalized for reason in _path_reasons(path)
    ]
    present = set(normalized)
    violations.extend(
        f"{required}: required file missing"
        for required in REQUIRED_FILES
        if required not in present
    )
    return violations


def allowlist_problems(lines: Iterable[str]) -> list[str]:
    """Every way the ``.vercelignore`` *lines* stop being the R2 allowlist (empty when it is one)."""
    rules = [
        line.strip()
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]
    problems: list[str] = []
    if not rules or rules[0] != _ROOT_RULE:
        problems.append(f"the first rule must be {_ROOT_RULE} (ignore everything)")
    for rule in rules:
        if not rule.startswith("!"):
            continue
        name = rule[1:]
        if any(mark in name for mark in _WILDCARDS):
            problems.append(f"{rule} re-admits through a wildcard")
        elif name not in ALLOWED_REINCLUDES:
            problems.append(f"{rule} re-admits {name}, which is not in the allowlist")

    positions = {rule: index for index, rule in enumerate(rules)}
    data_rules = (_DATA_DIR_RULE, _DATA_CONTENTS_RULE, _CACHE_RULE)
    missing = [rule for rule in data_rules if rule not in positions]
    if missing:
        problems.extend(f"the data rule {rule} is missing" for rule in missing)
    elif (
        not positions[_DATA_DIR_RULE]
        < positions[_DATA_CONTENTS_RULE]
        < positions[_CACHE_RULE]
    ):
        problems.append(
            f"{_DATA_CONTENTS_RULE} must sit after {_DATA_DIR_RULE} and before {_CACHE_RULE}"
        )

    last_readmit = max(
        (i for i, rule in enumerate(rules) if rule.startswith("!")), default=-1
    )
    after = set(rules[last_readmit + 1 :])
    problems.extend(
        f"the deny rule {deny} is missing after the last re-admission"
        for deny in REQUIRED_DENY_RULES
        if deny not in after
    )
    return problems


def _read_list(list_file: Path) -> list[str]:
    """The paths in *list_file*: a JSON array, or one path per line."""
    text = list_file.read_text(encoding="utf-8")
    if text.lstrip().startswith("["):
        return [str(entry) for entry in json.loads(text)]
    return text.splitlines()


def build_parser() -> argparse.ArgumentParser:
    """The CLI parser: one positional, the upload-list file."""
    parser = argparse.ArgumentParser(
        description="Judge a Vercel upload list against the SPEC R2 allowlist rules"
    )
    parser.add_argument(
        "list_file", type=Path, help="one path per line, or a JSON array"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print the contract lines for the list file in *argv*; 0 when clean, else 1."""
    args = build_parser().parse_args(argv)
    try:
        paths = _read_list(args.list_file)
    except (OSError, ValueError) as error:
        print(f"UPLOAD_LIST= refused: {type(error).__name__}")
        return 1
    violations = upload_list_violations(paths)
    for violation in violations:
        print(f"UPLOAD_LIST_VIOLATION= {violation}")
    if violations:
        return 1
    count = sum(1 for raw in paths if _normalize(raw))
    print(f"UPLOAD_LIST= ok {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
