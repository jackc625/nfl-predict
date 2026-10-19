"""The upload allowlist stays an allowlist, and the upload-list checker judges real lists (SPEC R2).

The Vercel CLI reads no ``.gitignore`` and does not ignore ``.env`` by default, so ``.vercelignore``
is the only thing standing between the working tree and a third party. These tests keep it shaped
as an allowlist (``/*`` first, exact re-admissions, deny rules last) and prove that
``scripts.check_upload_list`` names every forbidden path in a produced upload list. Plan 36-09 runs
that checker on the CLI's real list before any deploy.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.check_upload_list import allowlist_problems, main, upload_list_violations

REPO_ROOT = Path(__file__).resolve().parents[2]
VERCELIGNORE = REPO_ROOT / ".vercelignore"

CLEAN_UPLOAD = [
    "index.py",
    "requirements.txt",
    ".python-version",
    "vercel.json",
    "api/main.py",
    "web/templates/base.html",
    "utils/date_utils.py",
    "conf/settings.py",
    "data/web_cache.duckdb",
]


def _committed_lines() -> list[str]:
    return VERCELIGNORE.read_text(encoding="utf-8").splitlines()


def _with_added(rule: str) -> list[str]:
    """The committed lines with *rule* inserted among the re-admissions."""
    lines = _committed_lines()
    lines.insert(lines.index("!conf") + 1, rule)
    return lines


def _without(rule: str) -> list[str]:
    return [line for line in _committed_lines() if line != rule]


def _data_star_moved_last() -> list[str]:
    lines = _without("data/*")
    lines.insert(lines.index("!data/web_cache.duckdb") + 1, "data/*")
    return lines


def test_the_committed_allowlist_has_no_problems() -> None:
    assert allowlist_problems(_committed_lines()) == []


@pytest.mark.parametrize(
    ("loosen", "named"),
    [
        (lambda: _with_added("!tests"), "tests"),
        (lambda: _with_added("!pyproject.toml"), "pyproject.toml"),
        (lambda: _without("**/.env*"), "**/.env*"),
        (_data_star_moved_last, "data/*"),
        (lambda: _without("/*"), "/*"),
    ],
    ids=[
        "readmits-tests",
        "readmits-pyproject",
        "env-deny-removed",
        "data-star-misordered",
        "root-rule-removed",
    ],
)
def test_a_loosened_allowlist_is_caught(
    loosen: Callable[[], list[str]], named: str
) -> None:
    problems = allowlist_problems(loosen())
    assert any(named in problem for problem in problems), problems


@pytest.mark.parametrize(
    "forbidden",
    [
        "data/silver/games.parquet",
        "data/web_cache.tmp.duckdb",
        ".env",
        "conf/.env.local",
        "pyproject.toml",
        "uv.lock",
        "tests/conftest.py",
        ".planning/STATE.md",
        "pipeline/steps.py",
    ],
)
def test_an_upload_list_is_judged(forbidden: str) -> None:
    assert upload_list_violations(CLEAN_UPLOAD) == []
    violations = upload_list_violations([*CLEAN_UPLOAD, forbidden])
    assert violations and all(v.startswith(f"{forbidden}: ") for v in violations), (
        violations
    )


def test_a_list_without_the_cache_names_the_missing_file() -> None:
    without_cache = [path for path in CLEAN_UPLOAD if path != "data/web_cache.duckdb"]
    violations = upload_list_violations(without_cache)
    assert len(violations) == 1 and violations[0].startswith("data/web_cache.duckdb: ")


def test_the_cli_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text(
        "\n".join([*CLEAN_UPLOAD, "logs\\daily.log"]) + "\n", encoding="utf-8"
    )
    assert main([str(bad)]) == 1
    out = capsys.readouterr().out
    assert "UPLOAD_LIST_VIOLATION= logs/daily.log: " in out

    clean = tmp_path / "clean.json"
    clean.write_text(
        json.dumps([p.replace("/", "\\") for p in CLEAN_UPLOAD]), encoding="utf-8"
    )
    assert main([str(clean)]) == 0
    assert (
        capsys.readouterr().out.strip() == f"UPLOAD_LIST= ok {len(CLEAN_UPLOAD)} files"
    )
