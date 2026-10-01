"""Rendering guards for the restyled shared components (redesign Tasks 4-5).

Each partial is rendered through the app's own Jinja environment and checked for what the
redesign promises: honesty text kept word for word inside a native, collapsed <details>; badges
that carry no outcome hue; and controls that keep every piece of their htmx wiring.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from api.dependencies import templates

NOT_ADVICE_FULL_TEXT = (
    "This is a personal research tool. No wager is placed and no currency amount is shown -- "
    "stakes are expressed in units, where 1 unit = 1% of a notional bankroll. The deployed "
    "models have not demonstrated a positive edge against the closing market: the "
    "win-probability model's pooled closing-line value is negative, and the spread and totals "
    "models were retained after failing their most recent re-fit gate. A bet appearing on this "
    "list means it cleared a pre-registered expected-value floor, not that it is expected to win."
)
OLD_RULE_FULL_TEXT = (
    "Built under the old rule on inputs later found defective; not evidence. These figures come "
    "from before the September 2026 fix to what the models were fed, and they stay here for the "
    "record only. Only the 2026 season, recorded live with each game locked at 6 PM Eastern the "
    "day before kickoff, counts as evidence."
)
OUTCOME_HUES = ("green", "red", "amber", "yellow")


def render(name: str, **context: Any) -> str:
    return templates.env.get_template(f"components/{name}").render(**context)


def flat(html: str) -> str:
    return " ".join(html.split())


def class_attributes(html: str) -> list[str]:
    return re.findall(r'class="([^"]*)"', html)


def summary_of(html: str) -> str:
    return html[html.index("<summary>") : html.index("</summary>")]


class TestNotAdviceBanner:
    def test_it_is_a_collapsed_native_disclosure_inside_a_note(self) -> None:
        html = render("_not_advice_banner.html")
        assert '<aside role="note"' in html
        assert '<details class="honesty-note">' in html
        assert "<details open" not in html
        assert "<button" not in html and "<script" not in html

    def test_the_summary_names_it_and_offers_why(self) -> None:
        summary = summary_of(render("_not_advice_banner.html"))
        assert "Not wagering advice" in summary
        assert "Why?" in summary

    def test_the_full_text_is_kept_word_for_word(self) -> None:
        assert NOT_ADVICE_FULL_TEXT in flat(render("_not_advice_banner.html"))


class TestOldRuleLabel:
    def test_an_unwired_block_labels_once_as_a_collapsed_disclosure(self) -> None:
        html = render("_old_rule_label.html")
        assert html.count("data-old-rule-label") == 1
        assert '<details class="honesty-note">' in html
        assert "<details open" not in html

    def test_the_summary_states_the_date_and_not_evidence(self) -> None:
        summary = summary_of(render("_old_rule_label.html"))
        assert "Old-rule numbers" in summary
        assert "2026-09-15" in summary
        assert "not evidence" in summary

    def test_the_full_sentence_is_kept_word_for_word(self) -> None:
        assert OLD_RULE_FULL_TEXT in flat(render("_old_rule_label.html"))


class TestMonochromeBadges:
    @pytest.mark.parametrize("band", ["high", "medium", "low"])
    def test_ev_band(self, band: str) -> None:
        html = render("_ev_band_badge.html", band=band)
        assert f'<span class="band band-{band}" title="EV band {band}:' in html
        for classes in class_attributes(html):
            for hue in OUTCOME_HUES:
                assert hue not in classes

    def test_an_unknown_ev_band_renders_as_low(self) -> None:
        assert 'class="band band-low"' in render("_ev_band_badge.html", band="weird")

    @pytest.mark.parametrize("level", ["high", "medium", "low"])
    def test_confidence(self, level: str) -> None:
        html = render("_confidence_badge.html", level=level, value=level.title())
        assert (
            f'<span class="band band-{level}" data-confidence-band="{level}">'
            f"{level.title()}</span>"
        ) in html

    def test_an_unknown_confidence_renders_as_low(self) -> None:
        html = render("_confidence_badge.html", level="weird", value="Weird")
        assert 'data-confidence-band="low"' in html

    @pytest.mark.parametrize(
        ("provenance", "validation_type", "classes", "label"),
        [
            ("backtest_replay", "contaminated", "evidence-chip", "Contaminated split"),
            (
                "backtest_replay",
                "clean_holdout",
                "evidence-chip evidence-chip-strong",
                "Old rule -- 2025, not evidence",
            ),
            (
                "forward",
                "forward_realized",
                "evidence-chip",
                "Live forward record",
            ),
        ],
    )
    def test_provenance(
        self, provenance: str, validation_type: str, classes: str, label: str
    ) -> None:
        html = render(
            "_provenance_badge.html",
            provenance=provenance,
            validation_type=validation_type,
        )
        assert f'<span class="{classes}" data-provenance="{provenance}"' in html
        assert f'data-validation-type="{validation_type}"' in html
        assert f'>{label}<span class="sr-only">' in html
        assert html.strip().endswith("</span></span>")

    def test_an_impossible_provenance_pair_renders_its_raw_code(self) -> None:
        html = render(
            "_provenance_badge.html",
            provenance="forward",
            validation_type="contaminated",
        )
        assert ">contaminated<span" in html

    @pytest.mark.parametrize(
        ("status", "text"),
        [
            ("completed", "Completed"),
            ("scheduled", "Scheduled"),
            ("in_progress", "In Progress"),
            ("postponed", "Postponed"),
            ("cancelled", "Cancelled"),
            ("delayed", "Delayed"),
        ],
    )
    def test_status_badges_never_use_an_outcome_hue(
        self, status: str, text: str
    ) -> None:
        html = render("_status_badge.html", status=status)
        assert f">{text}</span>" in html
        for classes in class_attributes(html):
            for hue in ("green", "red", "amber", "blue"):
                assert hue not in classes
