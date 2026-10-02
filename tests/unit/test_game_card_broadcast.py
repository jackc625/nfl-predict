"""The Broadcast game card reads cleanly in every state (Broadcast redesign, Task 7).

The card is rendered here exactly as tests/unit/test_current_week_rows_reach_the_site.py renders
it: a plain jinja2 Environment with no app globals or filters, handed only ``game=``. That is the
contract the card keeps so a bare prediction row always renders.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from api.cache import PREDICTIONS_TABLE_COLUMNS
from api.presentation import decorate_game

TEMPLATES = Path(__file__).resolve().parents[2] / "web" / "templates"

# A green or red Tailwind colour utility in a class attribute. On this site those two hues mean a
# REALISED result only; an edge, a band or a flag must never carry one.
_HUE = re.compile(r"\b(?:text|bg|border)-(green|red)-\d{2,3}")


def _render(game: dict[str, Any]) -> str:
    return (
        Environment(loader=FileSystemLoader(TEMPLATES))
        .get_template("components/_game_card.html")
        .render(game=game)
    )


def _hues(html: str) -> list[str]:
    return [
        match.group(1)
        for classes in re.findall(r'class="([^"]*)"', html)
        for match in _HUE.finditer(classes)
    ]


def _row(**overrides: Any) -> dict[str, Any]:
    """KC at MIA, Week 3 2026, model numbers only unless *overrides* add more."""
    row = dict.fromkeys(PREDICTIONS_TABLE_COLUMNS)
    row.update(
        game_id="2026_W03_KC@MIA",
        season=2026,
        week=3,
        game_date=datetime(2026, 9, 27, 13, 0),
        home_team="MIA",
        away_team="KC",
        status="scheduled",
        wp_prob=0.412,
        ats_prediction=-2.5,
        ou_prediction=41.3,
    )
    row.update(overrides)
    return row


_MARKET = {
    "market_wp": 0.449,
    "market_spread": -1.5,
    "market_total": 44.5,
    "wp_edge": -0.037,
    "ats_edge": -1.0,
    "ou_edge": -0.0719,
}


def test_a_week_with_no_market_lines_reads_cleanly() -> None:
    """Review Focus 1: today's real Week 3 -- no line anywhere -- shows "No line" and nothing else.

    No empty chip, no "None", no "nan", no pick words without a line to pick against.
    """
    card = _render(decorate_game(_row()))

    assert card.count("No line") == 3, "each of the three market slots must say No line"
    # 12px text in the muted token: dim is 3.0:1 on a panel, below AA for text this small.
    assert card.count('<span class="text-muted italic">No line</span>') == 3
    assert "data-edge" not in card, "an edge chip rendered for a game with no edge"
    assert "None" not in card
    assert "nan" not in card.lower()
    for pick_word in ("covers", "Over", "Under", "No pick"):
        assert pick_word not in card, (
            f"{pick_word!r} rendered with no line to pick against"
        )
    assert " vs " not in card
    assert "data-on-bet-list" not in card
    assert "KC 58.8%" in card, "the model's own win probability is missing"
    assert "KC by 2.5" in card and "41.3" in card


def test_model_and_market_numbers_pick_words_and_edges_render() -> None:
    card = _render(decorate_game(_row(**_MARKET)))

    assert "KC 58.8%" in card and "KC 55.1%" in card
    assert "KC by 2.5" in card and "KC by 1.5" in card
    assert "KC covers" in card, (
        "model margin below the home-margin line backs the away side"
    )
    assert "Under" in card and "44.5" in card
    for target, text in (("wp", r"-3\.7%"), ("ats", r"-1\.0 pts"), ("ou", r"-7\.2%")):
        assert re.search(rf'data-edge="{target}".*?{text}', card, re.DOTALL), (
            f"the {target} edge chip does not show {text}"
        )
    assert _hues(card) == [], "a pre-game number was drawn in green or red"


def test_a_bet_game_carries_the_flag_and_a_solid_chip_for_its_bet_type() -> None:
    game = {**decorate_game(_row(**_MARKET)), "bet_targets": ["ou"]}
    card = _render(game)
    plain_card = _render(decorate_game(_row(**_MARKET)))

    assert "data-on-bet-list" in card
    # The TOP RULE's colour, not any "border-accent": every card also carries hover:border-accent,
    # so a bare substring check would pass for a card with no bet. The plain card is the control.
    top_rule = "border-t-[3px] border-accent "
    assert top_rule in card.split(">", 1)[0], (
        "the bet card's top edge is not the accent"
    )
    assert top_rule not in plain_card.split(">", 1)[0], (
        "a card with no bet has the accent top edge"
    )
    # Three edges, one of them the bet's own: two outlined chips and one solid.
    assert card.count("edge-chip-soft") == 2


def test_a_final_game_shows_the_score_the_grade_and_the_prediction() -> None:
    game = {
        **decorate_game(
            _row(
                status="completed",
                away_score=20,
                home_score=27,
                wp_prob=0.62,
                **_MARKET,
            )
        ),
        "wp_correct": True,
    }
    card = _render(game)
    flat = " ".join(card.split())

    assert "Final" in card
    assert "Correct" in card and "Incorrect" not in card
    assert "Predicted: MIA 62% WP" in flat
    assert set(_hues(card)) == {"green"}, "only the realised result may be green"


def test_a_tie_is_graded_neither_correct_nor_incorrect() -> None:
    game = {
        **decorate_game(
            _row(status="completed", away_score=20, home_score=20, wp_prob=0.40)
        ),
        "wp_correct": None,
    }
    card = _render(game)

    assert "Tie" in card
    assert "Correct" not in card and "Incorrect" not in card
    assert _hues(card) == []


def test_the_card_renders_from_a_bare_row_with_no_decoration() -> None:
    """A row nobody decorated still renders: every new field has a default."""
    card = _render(_row())

    assert "game-card" in card
    assert "KC 58.8%" in card
    assert "View Details" in card
