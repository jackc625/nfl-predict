"""Guards the Broadcast design system's compiled output (redesign Task 1).

Tailwind v4 emits only what it is told to: a token missing from ``@theme static``, a component
class missing from ``@layer components`` or an ``@font-face`` pointing at a file that was never
committed renders as silently unstyled HTML. Each of those is checked against the COMPILED sheet
the app actually serves, not against ``input.css``.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILED_CSS = REPO_ROOT / "web" / "static" / "css" / "tailwind-compiled.css"
FONTS_DIR = REPO_ROOT / "web" / "static" / "fonts"

TOKENS = (
    "--color-ink",
    "--color-ink-2",
    "--color-panel",
    "--color-panel-2",
    "--color-line",
    "--color-fg",
    "--color-muted",
    "--color-dim",
    "--color-accent",
    "--color-target-wp",
    "--color-target-ats",
    "--color-target-ou",
    "--font-display",
    "--font-sans",
    "--font-mono",
)

COMPONENT_CLASSES = (
    "skew",
    "unskew",
    "display",
    "label",
    "num",
    "stat-num",
    "panel",
    "panel-title",
    "section-head",
    "tag",
    "tag-ghost",
    "skew-control",
    "skew-control-active",
    "nav-link",
    "wordmark",
    "skip-link",
    "team-block",
    "team-block-lg",
    "score-bug",
    "score-row",
    "score-name",
    "score-value",
    "is-fav",
    "is-dog",
    "edge-chip",
    "edge-chip-soft",
    "stat-tile",
    "stat-tile-value",
    "stat-tile-sub",
    "bet-slip",
    "honesty-note",
    "honesty-note-key",
    "honesty-note-line",
    "honesty-note-why",
    "honesty-note-body",
    "band",
    "band-high",
    "band-medium",
    "band-low",
    "evidence-chip",
    "evidence-chip-strong",
    "grow-in",
)

FONT_FILES = (
    "barlow-condensed-latin-600-normal.woff2",
    "barlow-condensed-latin-700-italic.woff2",
    "barlow-condensed-latin-700-normal.woff2",
    "barlow-condensed-latin-800-italic.woff2",
    "barlow-condensed-latin-800-normal.woff2",
    "inter-latin-wght-normal.woff2",
    "jetbrains-mono-latin-500-normal.woff2",
    "jetbrains-mono-latin-700-normal.woff2",
)

LICENCES = ("OFL-BarlowCondensed.txt", "OFL-Inter.txt", "OFL-JetBrainsMono.txt")


@pytest.fixture(scope="module")
def css() -> str:
    return COMPILED_CSS.read_text(encoding="utf-8")


@pytest.mark.parametrize("token", TOKENS)
def test_every_design_token_is_emitted(css: str, token: str) -> None:
    """``@theme static`` emits each token even when no utility uses it yet."""
    assert f"{token}:" in css, f"{token} is not defined in the compiled stylesheet"


@pytest.mark.parametrize("name", COMPONENT_CLASSES)
def test_every_component_class_is_compiled(css: str, name: str) -> None:
    """The exact class, not merely a longer class that starts with the same letters."""
    assert re.search(rf"\.{re.escape(name)}(?![\w-])", css), (
        f".{name} is missing from tailwind-compiled.css"
    )


def test_each_font_face_points_at_a_committed_file(css: str) -> None:
    urls = re.findall(r"url\([\"']?/static/fonts/([^\"')]+)[\"']?\)", css)
    assert sorted(set(urls)) == sorted(FONT_FILES)
    for name in urls:
        assert (FONTS_DIR / name).is_file(), (
            f"@font-face points at a missing file: {name}"
        )


def test_the_three_families_are_declared(css: str) -> None:
    assert css.count("@font-face") == len(FONT_FILES)
    for family in ("Barlow Condensed", "Inter", "JetBrains Mono"):
        assert family in css


@pytest.mark.parametrize("name", FONT_FILES)
def test_every_font_file_is_real_woff2(name: str) -> None:
    assert (FONTS_DIR / name).read_bytes()[:4] == b"wOF2"


@pytest.mark.parametrize("name", LICENCES)
def test_the_open_font_licences_ship_with_the_fonts(name: str) -> None:
    text = (FONTS_DIR / name).read_text(encoding="utf-8")
    assert "SIL Open Font License" in text
