"""The push's stylesheet check (Phase 36, D-17, SPEC R6(c)).

The check recompiles the stylesheet with the vendored Tailwind binary into a temporary file and
compares it, CRLF normalized, with the committed sheet. These tests run the REAL binary when it is
on this checkout (it is on the owner's laptop), and prove the committed sheet is never written.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import stylesheet_check
from pipeline.stylesheet_check import check_stylesheet

_COMMITTED = stylesheet_check.REPO_ROOT / stylesheet_check.COMMITTED_CSS
_needs_binary = pytest.mark.skipif(
    not stylesheet_check.TAILWIND_EXE.is_file(),
    reason="tools/tailwindcss.exe not present",
)


@_needs_binary
def test_today_s_tree_is_in_step_with_the_real_binary() -> None:
    before = _COMMITTED.read_bytes()

    assert check_stylesheet() is None
    assert _COMMITTED.read_bytes() == before


def test_a_missing_binary_is_unverifiable(tmp_path: Path) -> None:
    assert check_stylesheet(exe=tmp_path / "absent.exe") == "stylesheet_unverifiable"


@_needs_binary
def test_a_sheet_that_differs_is_out_of_step(tmp_path: Path) -> None:
    drifted = tmp_path / "tailwind-compiled.css"
    drifted.write_text(
        _COMMITTED.read_text(encoding="utf-8") + ".never-compiled{color:red}\n",
        encoding="utf-8",
    )

    assert check_stylesheet(committed=drifted) == "stylesheet_out_of_step"
