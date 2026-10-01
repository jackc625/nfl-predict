"""The shared Broadcast macros render in a BARE Jinja environment (redesign Task 3).

The game card must render with no app globals or filters (tests/unit/
test_current_week_rows_reach_the_site.py renders it through a plain FileSystemLoader), and it
imports these macros -- so they are checked the same way, with autoescape on as the app has it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = Path(__file__).resolve().parents[2] / "web" / "templates"


@pytest.fixture(scope="module")
def bc():
    env = Environment(
        loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape()
    )
    return env.get_template("components/_broadcast.html").module


def test_section_head_renders_a_yellow_tag_heading(bc) -> None:
    html = str(bc.section_head("Live bets", meta="3 bets", anchor="live"))
    assert '<div class="section-head mb-3" id="live">' in html
    assert '<h2 class="tag"><span class="unskew">Live bets</span></h2>' in html
    assert '<span class="flex-1 h-px bg-line" aria-hidden="true"></span>' in html
    assert '<span class="text-xs text-muted whitespace-nowrap">3 bets</span>' in html
    # The rule follows the heading, so anything placed after the <h2> sits before the first </div>.
    assert (
        html.index("</h2>") < html.index("flex-1 h-px bg-line") < html.index("</div>")
    )


def test_section_head_ghost_and_no_meta(bc) -> None:
    html = str(bc.section_head("Backtest replay", ghost=True))
    assert (
        '<h2 class="tag-ghost"><span class="unskew">Backtest replay</span></h2>' in html
    )
    assert "text-xs text-muted" not in html
    assert " id=" not in html


def test_team_block_carries_its_colours_and_an_upright_label(bc) -> None:
    html = str(bc.team_block("KC", "#E31837", "#FFFFFF"))
    assert (
        '<span class="team-block" style="--team-bg: #E31837; --team-fg: #FFFFFF;">'
        '<span class="unskew">KC</span></span>'
    ) in html
    assert "team-block-lg" in str(bc.team_block("KC", large=True))


def test_score_row_marks_favourite_and_underdog(bc) -> None:
    fav = str(bc.score_row("KC", "Chiefs", "58.8%", "#E31837", "#FFFFFF", fav=True))
    dog = str(bc.score_row("MIA", "Dolphins", "41.2%", dog=True))
    assert '<div class="score-row is-fav">' in fav
    assert '<span class="score-name">Chiefs</span>' in fav
    assert '<span class="score-value">58.8%</span>' in fav
    assert '<div class="score-row is-dog">' in dog
    assert "--team-bg: #1D2436;" in dog


def test_edge_chip_solid_and_soft(bc) -> None:
    assert (
        str(bc.edge_chip("+2.1"))
        == '<span class="edge-chip"><span class="unskew">+2.1</span></span>'
    )
    assert (
        str(bc.edge_chip("3.1", soft=True))
        == '<span class="edge-chip edge-chip-soft"><span class="unskew">3.1</span></span>'
    )


def test_stat_tile_with_accent_sub_and_value_class(bc) -> None:
    html = str(
        bc.stat_tile(
            "Hit rate",
            "58.0%",
            value_class="text-green-400",
            sub="51 of 88",
            accent="var(--color-target-wp)",
        )
    )
    assert (
        '<div class="stat-tile" style="--tile-accent: var(--color-target-wp);">' in html
    )
    assert '<p class="label">Hit rate</p>' in html
    assert '<p class="stat-tile-value text-green-400">58.0%</p>' in html
    assert '<p class="stat-tile-sub">51 of 88</p>' in html


def test_stat_tile_plain(bc) -> None:
    html = str(bc.stat_tile("Bets graded", 90))
    assert '<div class="stat-tile">' in html
    assert '<p class="stat-tile-value">90</p>' in html
    assert "stat-tile-sub" not in html


def test_macro_arguments_are_escaped(bc) -> None:
    assert "&lt;b&gt;" in str(bc.section_head("<b>"))
