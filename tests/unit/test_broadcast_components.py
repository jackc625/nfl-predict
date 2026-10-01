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


class TestStates:
    def test_empty_state_is_a_grey_box_with_a_heading_and_a_paragraph(self) -> None:
        html = render(
            "_empty_state.html",
            heading="Nothing graded yet",
            body="Grades appear after the games are played.",
            action_text=None,
            action_url=None,
        )
        assert "border-dashed" in html
        assert '<h3 class="display text-xl text-fg">Nothing graded yet</h3>' in html
        assert "<p " in html and "Grades appear after the games are played." in html
        assert "<a " not in html
        for classes in class_attributes(html):
            assert "red" not in classes

    def test_empty_state_action_is_the_accent_control(self) -> None:
        html = render(
            "_empty_state.html",
            heading="Game not found",
            body="This game does not exist in the prediction database.",
            action_text="Back to This Week",
            action_url="/",
        )
        assert (
            '<a href="/" class="skew-control skew-control-active">'
            '<span class="unskew">Back to This Week</span></a>'
        ) in html

    def test_error_state_uses_the_reserved_error_reds(self) -> None:
        html = render(
            "_error_state.html",
            message="Could not load season data",
            recovery_text="Refresh.",
        )
        assert (
            'class="rounded bg-red-950 border border-red-800 p-6 text-center"' in html
        )
        assert "text-red-200" in html and "Could not load season data" in html
        assert '<p class="mt-1 text-sm text-red-100">Refresh.</p>' in html
        assert "text-red-400" not in html, (
            "the realised-loss red is not an error colour"
        )


class TestControls:
    def test_export_buttons_keep_ids_urls_and_touch_height(self) -> None:
        html = render(
            "_export_buttons.html",
            csv_url="/api/export/csv?season=2026&week=3",
            json_url="/api/export/json?season=2026&week=3",
            season_csv_url="/api/export/csv?season=2026",
        )
        assert 'id="export-buttons"' in html
        for element_id in ("export-csv", "export-json", "export-season-csv"):
            assert f'id="{element_id}"' in html
        assert html.count('data-base-url="/api/export/') == 3
        assert html.count("min-h-[44px]") == 3
        assert html.count('class="skew-control min-h-[44px]"') == 3
        assert html.count('<span class="unskew inline-flex items-center gap-1.5">') == 3
        assert "nfl-" not in html

    def test_export_buttons_without_a_season_link(self) -> None:
        html = render("_export_buttons.html", csv_url="/c", json_url="/j")
        assert "export-season-csv" not in html

    def test_scope_toggle_marks_the_current_scope_only(self) -> None:
        html = render("_betting_scope_toggle.html", current_scope="all")
        assert 'hx-get="/fragments/betting?scope=recommended"' in html
        assert 'hx-get="/fragments/betting?scope=all"' in html
        assert html.count('hx-target="#betting-content"') == 2
        assert html.count('hx-indicator="#betting-loading"') == 2
        assert html.count('aria-pressed="true"') == 1
        assert html.count("skew-control-active") == 1
        pressed = html[html.index('aria-pressed="true"') :]
        assert "skew-control-active" in pressed[: pressed.index(">")]
        assert '<span class="unskew">All bets</span></button>' in html

    def test_sort_controls_keep_their_wiring(self) -> None:
        html = render("_sort_controls.html", current_sort="edge")
        assert 'id="sort-select"' in html and 'for="sort-select"' in html
        assert 'hx-get="/fragments/games"' in html
        assert 'hx-target="#game-grid"' in html
        assert "hx-include=\"[name='week'],[name='season']\"" in html
        assert '<option value="edge" selected>Edge</option>' in html
        assert 'class="skew-control"' in html

    def test_season_selector_keeps_its_wiring(self) -> None:
        html = render(
            "_season_selector.html", available_seasons=[2024, 2023], current_season=2024
        )
        assert 'id="season-select"' in html
        assert 'hx-get="/fragments/performance"' in html
        assert 'hx-target="#performance-content"' in html
        assert 'id="perf-loading"' in html
        assert ">All Seasons</option>" in html
        assert '<option value="2024" selected>2024 Season</option>' in html

    def test_season_tracking_selector_keeps_its_error_handler(self) -> None:
        html = render(
            "_season_tracking_selector.html",
            available_seasons=[2026, 2025],
            current_season=2026,
        )
        assert 'id="season-track-select"' in html
        assert 'hx-get="/fragments/season"' in html
        assert 'hx-target="#season-content"' in html
        assert "hx-on::response-error=" in html
        assert "season-error-template" in html
        assert "All Seasons" not in html

    @pytest.mark.parametrize("variant", ["cards", "chart", "table"])
    def test_loading_skeleton_is_dark(self, variant: str) -> None:
        html = render("_loading_skeleton.html", variant=variant)
        assert "animate-pulse" in html
        assert "bg-white" not in html and "bg-gray-" not in html


class TestWeekSummary:
    SUMMARY = {
        "total_games": 13,
        "wp_correct": 9,
        "wp_total": 13,
        "wp_pct": 69,
        "ats_correct": 0,
        "ats_total": 0,
        "ats_pct": 0,
        "ou_correct": 7,
        "ou_total": 12,
        "ou_pct": 58,
    }

    def test_three_tiles_keyed_to_the_bet_type_colours(self) -> None:
        html = render("_week_summary.html", week_summary=self.SUMMARY)
        assert 'aria-label="Weekly prediction accuracy summary"' in html
        assert html.count('class="stat-tile"') == 3
        for label in ("Win Prob", "Spread", "Total"):
            assert f'<p class="label">{label}</p>' in html
        assert "--tile-accent: var(--color-target-wp);" in html
        assert "--tile-accent: var(--color-target-ats);" in html
        assert "--tile-accent: var(--color-target-ou);" in html
        assert ">9/13</p>" in html and "(69%)" in html
        assert ">7/12</p>" in html and "(58%)" in html

    def test_a_target_without_odds_reads_n_a(self) -> None:
        html = render("_week_summary.html", week_summary=self.SUMMARY)
        assert '<p class="stat-tile-value text-dim">N/A</p>' in html
        assert "No odds data" in html

    def test_nothing_renders_for_a_week_with_no_completed_game(self) -> None:
        assert render("_week_summary.html", week_summary={}).strip() == ""
