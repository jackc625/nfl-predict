"""Is the committed stylesheet in step with the templates? The push asks before it deploys (D-17).

WHY RECOMPILE AND COMPARE
-------------------------
The site serves ``web/static/css/tailwind-compiled.css``, generated from ``web/static/input.css``
by the vendored Tailwind binary. A template that starts using a utility class the committed sheet
lacks renders unstyled with no error anywhere. Scanning the templates for class names cannot catch
that reliably, because Jinja builds class strings at render time. So the check runs the same
compile into a temporary file and compares the result with the committed sheet: any difference
means the committed sheet is out of step (SPEC R6(c)).

WHY THE COMMITTED SHEET IS NEVER WRITTEN
----------------------------------------
This runs inside the unattended daily push. An unattended run must leave a clean tree, so the
compile goes to a temporary directory and the committed file is only read. Fixing a stale sheet is
the owner's recompile and commit, not the push's.

WHY CRLF IS NORMALIZED
----------------------
git's autocrlf checks the committed sheet out with CRLF line endings while the compiler writes LF.
That is the only difference measured between a fresh compile and today's committed sheet, so both
texts have CRLF turned into LF (and surrounding whitespace stripped) before they are compared.

The compiler's own output is never printed: the check answers a reason identifier or None.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

__all__ = [
    "COMMITTED_CSS",
    "COMPILE_TIMEOUT_SECONDS",
    "INPUT_CSS",
    "TAILWIND_EXE",
    "check_stylesheet",
]

REPO_ROOT = Path(__file__).resolve().parents[1]
# Absolute on purpose: a relative `tools/tailwindcss.exe` argv[0] fails Windows CreateProcess with
# FileNotFoundError even when the working directory is the repository root.
TAILWIND_EXE = REPO_ROOT / "tools" / "tailwindcss.exe"
# Both relative to the repository root, which is the compile's working directory -- the exact
# command docs/guides/PIPELINE.md gives the owner.
INPUT_CSS = Path("web/static/input.css")
COMMITTED_CSS = Path("web/static/css/tailwind-compiled.css")
COMPILE_TIMEOUT_SECONDS = 60

STYLESHEET_OUT_OF_STEP = "stylesheet_out_of_step"
STYLESHEET_UNVERIFIABLE = "stylesheet_unverifiable"


def _normalized(text: str) -> str:
    return text.replace("\r\n", "\n").strip()


def check_stylesheet(
    *,
    repo_root: Path = REPO_ROOT,
    exe: Path | None = None,
    committed: Path | None = None,
) -> str | None:
    """Compile the stylesheet into a temporary file and compare it with the committed sheet.

    Args:
        repo_root: The repository root, the compile's working directory.
        exe: The Tailwind binary (default :data:`TAILWIND_EXE`).
        committed: The committed sheet to compare with (default ``repo_root / COMMITTED_CSS``).

    Returns:
        None when the two match; ``stylesheet_out_of_step`` when they differ;
        ``stylesheet_unverifiable`` when the binary is missing, the compile fails or times out, or
        either sheet cannot be read.
    """
    exe_path = TAILWIND_EXE if exe is None else exe
    committed_path = repo_root / COMMITTED_CSS if committed is None else committed
    if not exe_path.is_file():
        return STYLESHEET_UNVERIFIABLE

    with tempfile.TemporaryDirectory() as scratch:
        fresh = Path(scratch) / "fresh.css"
        try:
            completed = subprocess.run(
                [str(exe_path), "-i", str(INPUT_CSS), "-o", str(fresh), "--minify"],
                cwd=repo_root,
                capture_output=True,
                timeout=COMPILE_TIMEOUT_SECONDS,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError):
            return STYLESHEET_UNVERIFIABLE
        if completed.returncode != 0:
            return STYLESHEET_UNVERIFIABLE
        try:
            fresh_text = fresh.read_text(encoding="utf-8")
            committed_text = committed_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return STYLESHEET_UNVERIFIABLE

    if _normalized(fresh_text) != _normalized(committed_text):
        return STYLESHEET_OUT_OF_STEP
    return None
