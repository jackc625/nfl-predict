"""Shared helpers for the D31-26 week-selector regression (plan 31-15).

The selector is ONE partial serving TWO pages, so its assertions live in two modules: the This
Week regression in ``test_pages.py`` (where a future editor of ``/`` will meet it) and the bets
side in ``test_bets_page.py``. The extraction rule and the recorded pre-parameterisation context
are defined HERE, once. Two copies of an extractor is the same duplicated-definition failure the
partial itself was parameterised to avoid.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
COMPONENTS_DIR = (
    Path(__file__).resolve().parents[2] / "web" / "templates" / "components"
)

#: The selector's outermost element. Its class string is pinned by the snapshots too, so a change
#: here is caught rather than silently retargeting the extraction.
SELECTOR_OPEN = '<div class="flex flex-wrap items-center gap-3">'

#: The three htmx failure events the bets selector wires. An HTTP error response, a timeout and a
#: dropped connection are THREE different events; wiring only the first leaves the other two
#: silent, and silence on this page means one week's rows under another week's heading.
FAILURE_EVENTS = ("hx-on::response-error", "hx-on::timeout", "hx-on::send-error")

#: The context ``/`` passed the selector BEFORE the parameterisation, recorded with THREE weeks so
#: both prev/next buttons are enabled. The shipped test fixture yields one week, under which both
#: buttons render ``disabled`` and the two attributes carrying the baked-in ``sort`` parameter are
#: never emitted at all -- so a snapshot taken only from the page would not compare them.
PRE_PARAM_CONTEXT: dict[str, Any] = {
    "available_weeks": [
        {"season": 2024, "week": 3},
        {"season": 2024, "week": 2},
        {"season": 2024, "week": 1},
    ],
    "available_seasons": [2024, 2023],
    "current_season": 2024,
    "current_week": 2,
    "current_sort": "confidence",
}


def read_snapshot(name: str) -> str:
    """Read a recorded snapshot, normalising line endings only.

    CRLF-versus-LF in the working tree is decided by git's ``core.autocrlf``, not by the template,
    so it is normalised on BOTH sides of every comparison. Every other byte is compared exactly.
    """
    return (SNAPSHOT_DIR / name).read_bytes().decode("utf-8").replace("\r\n", "\n")


def extract_selector(html: str) -> str:
    """Return the selector's rendered markup from *html*, balanced on its own ``div``."""
    start = html.index(SELECTOR_OPEN)
    depth = 0
    for match in re.finditer(r"<div\b|</div>", html[start:]):
        depth += 1 if match.group(0) != "</div>" else -1
        if depth == 0:
            return html[start : start + match.end()]
    raise AssertionError("the week selector's markup is unbalanced")


def class_values(markup: str) -> list[str]:
    """Every ``class`` attribute value in *markup*, in document order."""
    return re.findall(r'class="([^"]*)"', markup)
