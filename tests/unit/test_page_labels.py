"""Permanent rendering guard for the dated old-rule label on the website (Phase 33.2, R16).

D33.2-07: only the 2026 season, recorded live under the day-before-kickoff lock, is evidence. Every
website block that renders a pre-fix past-season number carries the shared label partial
``web/templates/components/_old_rule_label.html``; a block showing only 2026 numbers, or no numbers
at all, carries none.

This is a PERMANENT committed test, not a throwaway ``scripts/check_*.py``. It certifies the
TEMPLATES: each page is rendered directly with a context shaped like the one its route builds, and
the label is COUNTED in the output. ``tests/api/test_page_labels_routes.py`` certifies the PAGES
through the real routes, because a template test cannot prove that production ever builds the
context it was fed.

Three properties are held together here:

  - COVERAGE IS ENUMERATED, NOT REMEMBERED. ``EXPECTED_PREFIX_BLOCKS`` is keyed by every file in
    ``web/templates/pages/`` and its key set is asserted EQUAL to the directory listing at
    collection time, so a page added later fails here instead of going uncovered.
  - THE INCLUDE IS UNIVERSAL AND THE PARTIAL DECIDES. Every page includes the partial, and the
    partial reads the block's season scope itself. That is what makes a declared count MEASURED: a
    page whose count is wrong renders a different number of labels, whatever its markup says.
  - AN UNWIRED BLOCK LABELS. The partial emits when its scope is missing, so forgetting to wire a
    block produces a visible label rather than a silent omission.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from api.dependencies import templates
from api.services import DataService, clear_cache
from backtest.ev_chain_constants import READOUT_FORBIDDEN_WORDS
from tests.unit.test_old_rule_labels import LABEL_PHRASE

REPO_ROOT = Path(__file__).resolve().parents[2]
PAGES_DIR = REPO_ROOT / "web" / "templates" / "pages"
PARTIAL_PATH = REPO_ROOT / "web" / "templates" / "components" / "_old_rule_label.html"
COMPILED_CSS = REPO_ROOT / "web" / "static" / "css" / "tailwind-compiled.css"

# The attribute every rendered label carries. Counting it is how a test measures labels.
LABEL_MARKER = "data-old-rule-label"

# The newest season whose numbers are pre-fix (D33.2-07); 2026 is the first new-rule season.
PAST_SEASON = 2024
NEW_RULE_SEASON = 2026

# How many pre-fix blocks each page renders when it shows past-season numbers. Every page in
# web/templates/pages/ is a key; a page inspected and found to render no pre-fix number would be
# declared 0 with its reason in DELIBERATELY_UNLABELLED_REASONS.
EXPECTED_PREFIX_BLOCKS: dict[str, int] = {
    # The selected week's live list (a past week is a replay), and the replay tracker sections.
    "bets.html": 2,
    # The game header: its season line and the completed-game result overlay (score, badge, CLV).
    "game_detail.html": 1,
    # Calibration, feature importance and the accuracy trend: one backtest corpus.
    "how_it_works.html": 1,
    # The selected season's KPI strip, week strip and its cumulative and weekly charts.
    "season.html": 1,
    # The selected week's summary banner and game grid.
    "this_week.html": 1,
    # The all-history summary (tiles + season heatmap), the season-scoped metrics table, the
    # model-vs-market section (charts, aggregate table, cumulative CLV), and the betting
    # simulation: four blocks, four scopes.
    "track_record.html": 4,
}

# Pages inspected and deliberately left with no labelled block, each with its reason. Every page
# renders at least one pre-fix block when a past season is on display, so none is here today.
DELIBERATELY_UNLABELLED_REASONS: dict[str, str] = {}


def _page_listing() -> list[str]:
    """Every page template on disk, by filename."""
    return sorted(path.name for path in PAGES_DIR.glob("*.html"))


if sorted(EXPECTED_PREFIX_BLOCKS) != _page_listing():
    raise AssertionError(
        "EXPECTED_PREFIX_BLOCKS must be keyed by exactly the files in web/templates/pages/: "
        f"declared {sorted(EXPECTED_PREFIX_BLOCKS)}, on disk {_page_listing()}"
    )


@pytest.fixture(autouse=True)
def _isolate_data_service_cache() -> Iterator[None]:
    """DataService memoises cache_meta at module level, so each test starts from a cold cache."""
    clear_cache()
    try:
        yield
    finally:
        clear_cache()


class _StubRequest:
    """The one attribute the base template needs from a request: ``url_for`` for static files."""

    def url_for(self, name: str, **path_params: Any) -> str:
        return f"/{name}/{path_params.get('path', '')}"


def scope_for(season: int | None) -> dict[str, Any]:
    """The provenance scope a route supplies for a block showing *season* (None: no numbers)."""
    return DataService.old_rule_scope([] if season is None else [season])


def _game(season: int) -> dict[str, Any]:
    """One completed game, shaped like a DataService prediction row."""
    return {
        "game_id": f"{season}_W01_BUF@KC",
        "season": season,
        "week": 1,
        "game_date": f"{season}-09-10",
        "home_team": "KC",
        "away_team": "BUF",
        "status": "completed",
        "home_score": 27,
        "away_score": 20,
        "wp_prob": 0.62,
        "wp_confidence": "medium",
        "ats_prediction": 3.5,
        "ats_confidence": "high",
        "ou_prediction": 48.5,
        "ou_confidence": "medium",
        "market_spread": 3.0,
        "market_total": 47.5,
        "market_ml_home": -155,
        "market_ml_away": 135,
        "market_wp": 0.59,
        "wp_edge": 0.05,
        "ats_edge": 0.5,
        "ou_edge": 0.01,
        "blended_wp": 0.60,
        "blended_ats": 3.2,
        "blended_ou": 48.0,
        "wp_correct": True,
        "wp_clv": 1.5,
        "context": None,
        "feature_importances": {},
    }


def _block(provenance: str, validation_type: str) -> dict[str, Any]:
    """One precomputed tracker block with graded bets."""
    return {
        "provenance": provenance,
        "validation_type": validation_type,
        "bets_graded": 12,
        "wins": 7,
        "losses": 5,
        "pushes": 0,
        "hit_rate": 0.583,
        "flat_return_units": 0.41,
    }


def page_context(page: str, season: int) -> dict[str, Any]:
    """A context shaped like the one *page*'s route builds, every block showing *season*."""
    scope = scope_for(season)
    common: dict[str, Any] = {"request": _StubRequest(), "cache_meta": {}}
    if page == "track_record.html":
        metrics = [
            {"season": season, "target": "wp", "games": 256, "accuracy": 64.0}
            | {"mae": None, "rmse": None, "r2": None}
        ]
        return {
            **common,
            "charts": {"heatmap": "<div>heatmap</div>", "clv": "<div>clv</div>"},
            "available_seasons": [season],
            "current_season": None,
            "season_metrics": metrics,
            "summary": {"total_games": 256, "overall_clv": -1.2, "wp_accuracy": 64.0},
            "aggregate_table": [
                {
                    "target": "wp",
                    "metric": "Brier",
                    "model_fmt": "0.220",
                    "market_fmt": "0.210",
                    "gap_fmt": "+0.010",
                    "gap_favorable": False,
                }
            ],
            "kpis": {"total_bets": 5, "win_rate": 60.0, "roi_flat": 1.2},
            "roi_table": [],
            "current_scope": "recommended",
            "current_path": "/track-record",
            "summary_old_rule_scope": scope,
            "season_metrics_old_rule_scope": scope,
            "backtest_old_rule_scope": scope,
            "betting_old_rule_scope": scope,
        }
    if page == "how_it_works.html":
        return {
            **common,
            "charts": {"calibration": "<div>calibration</div>"},
            "current_path": "/how-it-works",
            "old_rule_scope": scope,
        }
    if page == "season.html":
        return {
            **common,
            "charts": {f"season_cumulative_{season}": "<div>cumulative</div>"},
            "kpis": {"wp_hit_rate": 61.0, "ats_hit_rate": 50.0, "ou_hit_rate": 49.0},
            "available_seasons": [season],
            "current_season": season,
            "current_path": "/season",
            "old_rule_scope": scope,
        }
    if page == "this_week.html":
        return {
            **common,
            "games": [_game(season)],
            "available_weeks": [{"season": season, "week": 1}],
            "available_seasons": [season],
            "current_week": 1,
            "current_season": season,
            "current_sort": "time",
            "current_path": "/",
            "week_summary": {
                "total_games": 1,
                "wp_correct": 1,
                "wp_total": 1,
                "wp_pct": 100,
            }
            | {"ats_correct": 0, "ats_total": 0, "ou_correct": 0, "ou_total": 0},
            "old_rule_scope": scope,
        }
    if page == "game_detail.html":
        return {
            **common,
            "game": _game(season),
            "current_path": "",
            "old_rule_scope": scope,
        }
    if page == "bets.html":
        replay = season <= DataService.LAST_OLD_RULE_SEASON
        return {
            **common,
            "bets": [],
            "suppressed_bets": [],
            "available_bet_weeks": [{"season": season, "week": 1, "game_count": 1}],
            "bet_seasons": [season],
            "current_season": season,
            "current_week": 1,
            "bet_week_freeze": None,
            "current_path": "/bets",
            "bet_list_available": True,
            "bet_list_populated_at": "2026-09-18T22:00:00+00:00",
            "bets_blocked": False,
            "tracker_blocks": (
                [_block("backtest_replay", "contaminated")]
                if replay
                else [_block("forward", "forward_realized")]
            ),
            "week_old_rule_scope": scope,
            "replay_old_rule_scope": scope if replay else scope_for(None),
        }
    raise AssertionError(f"no context builder for {page}")


def render_page(page: str, context: dict[str, Any]) -> str:
    """Render one page template with *context*, exactly as the route's TemplateResponse would."""
    return templates.env.get_template(f"pages/{page}").render(context)


def label_count(html: str) -> int:
    """How many labels a rendered page carries."""
    return html.count(LABEL_MARKER)


# ---------------------------------------------------------------------------
# The provenance accessor (api/services.py)
# ---------------------------------------------------------------------------


class TestTheProvenanceAccessor:
    """``DataService.old_rule_scope`` turns a block's season span into the label decision."""

    def test_an_unknown_span_labels(self) -> None:
        """No span at all is the unwired case, and it must label."""
        scope = DataService.old_rule_scope(None)
        assert scope == {
            "min_season": None,
            "max_season": None,
            "contains_old_rule_results": True,
        }

    def test_a_known_empty_span_does_not_label(self) -> None:
        """A block that renders no numbers has an empty span, and a number-free block is bare."""
        assert DataService.old_rule_scope([])["contains_old_rule_results"] is False

    def test_a_pre_fix_season_labels(self) -> None:
        scope = DataService.old_rule_scope([2025])
        assert scope["contains_old_rule_results"] is True
        assert (scope["min_season"], scope["max_season"]) == (2025, 2025)

    def test_a_2026_only_span_does_not_label(self) -> None:
        scope = DataService.old_rule_scope([2026, 2026])
        assert scope["contains_old_rule_results"] is False
        assert (scope["min_season"], scope["max_season"]) == (2026, 2026)

    def test_a_mixed_span_labels(self) -> None:
        """A block spanning the boundary carries pre-fix numbers, so it labels."""
        scope = DataService.old_rule_scope([2026, 2021])
        assert scope["contains_old_rule_results"] is True
        assert (scope["min_season"], scope["max_season"]) == (2021, 2026)

    def test_the_boundary_is_2025(self) -> None:
        """D33.2-07: 2026 is the first season recorded under the new rule."""
        assert DataService.LAST_OLD_RULE_SEASON == 2025


# ---------------------------------------------------------------------------
# The partial
# ---------------------------------------------------------------------------


class TestThePartial:
    """The shared partial carries the label sentence and its own condition."""

    def test_the_partial_is_ascii_and_carries_the_label_sentence(self) -> None:
        text = PARTIAL_PATH.read_text(encoding="utf-8")
        assert text.isascii()
        assert LABEL_PHRASE in text.lower()

    def test_the_partial_carries_no_over_claim_word(self) -> None:
        text = PARTIAL_PATH.read_text(encoding="utf-8").lower()
        present = [word for word in READOUT_FORBIDDEN_WORDS if word in text]
        assert not present, present

    def test_every_class_the_partial_uses_exists_in_the_compiled_stylesheet(
        self,
    ) -> None:
        """Tailwind v4 emits only classes it saw at build time; an absent class renders unstyled."""
        text = PARTIAL_PATH.read_text(encoding="utf-8")
        css = COMPILED_CSS.read_text(encoding="utf-8")
        classes = {
            name
            for attr in re.findall(r'class="([^"]+)"', text)
            for name in attr.split()
        }
        assert classes, "the partial declares no classes"
        missing = sorted(name for name in classes if f".{name}" not in css)
        assert not missing, f"classes absent from tailwind-compiled.css: {missing}"


# ---------------------------------------------------------------------------
# Coverage: every page, rendered with past-season numbers
# ---------------------------------------------------------------------------


class TestCoverage:
    """Each page renders exactly its declared number of labels when it shows pre-fix numbers."""

    def test_every_page_includes_the_partial(self) -> None:
        """The include is universal, so every page's count is decided by the partial."""
        without = [
            page
            for page in _page_listing()
            if "_old_rule_label.html"
            not in (PAGES_DIR / page).read_text(encoding="utf-8")
        ]
        assert without == []

    def test_every_zero_page_carries_a_reason(self) -> None:
        zero = [page for page, count in EXPECTED_PREFIX_BLOCKS.items() if count == 0]
        assert [
            page for page in zero if page not in DELIBERATELY_UNLABELLED_REASONS
        ] == []

    @pytest.mark.parametrize("page", sorted(EXPECTED_PREFIX_BLOCKS))
    def test_past_season_label_count_matches_the_declaration(self, page: str) -> None:
        html = render_page(page, page_context(page, PAST_SEASON))
        assert label_count(html) == EXPECTED_PREFIX_BLOCKS[page], (
            f"{page} rendered {label_count(html)} labels, declared "
            f"{EXPECTED_PREFIX_BLOCKS[page]}"
        )


# ---------------------------------------------------------------------------
# The population-time season span (api/cache.py), read back through get_cache_meta
# ---------------------------------------------------------------------------


def _cache_with(rows: dict[str, list[tuple]]) -> Any:
    """An in-memory cache built from CACHE_SCHEMA, carrying *rows* per table."""
    import duckdb

    from api.cache import CACHE_SCHEMA

    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement.strip())
    for table, values in rows.items():
        width = len(values[0])
        conn.executemany(
            f"INSERT INTO {table} VALUES ({', '.join('?' * width)})",
            values,
        )
    return conn


_BACKTEST_PREDICTION = ("{s}_W01_BUF@KC", None, 1, "wp", 0.6, 1.0, 0.01, True)
# game_id, season, week, target, bet_side, model_value, market_value, edge, slipped_line, odds,
# flat_stake, kelly_stake, outcome, payout_flat, payout_kelly
_BETTING_BET = ("g", None, 1, "wp", "home", 0.6, 0.5, 0.1, None, -110.0)
_BETTING_BET_TAIL = (100.0, 10.0, True, 90.9, 9.1)


def _prediction_row(season: int) -> tuple:
    row = list(_BACKTEST_PREDICTION)
    row[0] = row[0].format(s=season)
    row[1] = season
    return tuple(row)


def _betting_row(season: int) -> tuple:
    row = [*_BETTING_BET, *_BETTING_BET_TAIL]
    row[1] = season
    return tuple(row)


class TestTheSeasonSpanStamp:
    """Pre-rendered chart blobs carry no season column, so population stamps their span."""

    def test_the_backtest_and_betting_spans_are_stamped_from_the_data(self) -> None:
        from datetime import UTC, datetime

        from api.cache import (
            BACKTEST_SEASON_RANGE_KEY,
            BETTING_SEASON_RANGE_KEY,
            stamp_old_rule_season_ranges,
        )

        conn = _cache_with(
            {
                "backtest_predictions": [_prediction_row(2021), _prediction_row(2024)],
                "backtest_metrics": [(0, "overall", "total_games", 9.0)],
                "betting_bets": [_betting_row(2022), _betting_row(2023)],
            }
        )
        stamp_old_rule_season_ranges(conn, datetime.now(tz=UTC))
        service = DataService(conn)
        meta = service.get_cache_meta()
        assert meta[BACKTEST_SEASON_RANGE_KEY] == "2021-2024"
        assert meta[BETTING_SEASON_RANGE_KEY] == "2022-2023"
        backtest = service.cached_span_old_rule_scope(BACKTEST_SEASON_RANGE_KEY)
        assert backtest == {
            "min_season": 2021,
            "max_season": 2024,
            "contains_old_rule_results": True,
        }

    def test_a_2026_only_corpus_stamps_a_span_that_does_not_label(self) -> None:
        from datetime import UTC, datetime

        from api.cache import BACKTEST_SEASON_RANGE_KEY, stamp_old_rule_season_ranges

        conn = _cache_with({"backtest_predictions": [_prediction_row(2026)]})
        stamp_old_rule_season_ranges(conn, datetime.now(tz=UTC))
        scope = DataService(conn).cached_span_old_rule_scope(BACKTEST_SEASON_RANGE_KEY)
        assert scope["contains_old_rule_results"] is False

    def test_an_empty_corpus_stamps_none_which_does_not_label(self) -> None:
        from datetime import UTC, datetime

        from api.cache import BETTING_SEASON_RANGE_KEY, stamp_old_rule_season_ranges

        conn = _cache_with({"backtest_metrics": [(0, "overall", "total_games", 0.0)]})
        stamp_old_rule_season_ranges(conn, datetime.now(tz=UTC))
        service = DataService(conn)
        assert service.get_cache_meta()[BETTING_SEASON_RANGE_KEY] == "none"
        scope = service.cached_span_old_rule_scope(BETTING_SEASON_RANGE_KEY)
        assert scope["contains_old_rule_results"] is False

    def test_an_absent_key_is_an_unknown_span_which_labels(self) -> None:
        """A cache built before the stamp existed cannot say what its charts cover."""
        conn = _cache_with({"backtest_metrics": [(0, "overall", "total_games", 0.0)]})
        scope = DataService(conn).cached_span_old_rule_scope("backtest_season_range")
        assert scope["contains_old_rule_results"] is True

    @pytest.mark.parametrize("value", ["garbage", "2021", "2021-x", ""])
    def test_an_unreadable_span_labels(self, value: str) -> None:
        from api.cache import parse_season_range

        assert parse_season_range(value) is None


# ---------------------------------------------------------------------------
# Two-way rendering: 2026-only, unwired, and the non-vacuity control
# ---------------------------------------------------------------------------

# A string each page renders only when the page itself rendered, so a template error that yields an
# empty or truncated string cannot satisfy a zero-count assertion.
PAGE_MARKERS: dict[str, str] = {
    "bets.html": "Weekly Bet List",
    "game_detail.html": "BUF @ KC",
    "how_it_works.html": "What the models rely on",
    "season.html": "Season Tracking",
    "this_week.html": "This Week's Predictions",
    "track_record.html": "Season by season",
}


def _without_scopes(context: dict[str, Any]) -> dict[str, Any]:
    """The same context with every block scope removed -- the page as if nobody wired it."""
    return {
        key: value
        for key, value in context.items()
        if not key.endswith("old_rule_scope")
    }


class TestTwoWayRendering:
    """The label appears exactly where a pre-fix number does, and nowhere else."""

    def test_page_markers_cover_every_page(self) -> None:
        assert sorted(PAGE_MARKERS) == sorted(EXPECTED_PREFIX_BLOCKS)

    @pytest.mark.parametrize("page", sorted(EXPECTED_PREFIX_BLOCKS))
    def test_past_season_render_is_the_page_itself(self, page: str) -> None:
        html = render_page(page, page_context(page, PAST_SEASON))
        assert PAGE_MARKERS[page] in html

    @pytest.mark.parametrize("page", sorted(EXPECTED_PREFIX_BLOCKS))
    def test_a_2026_only_page_renders_no_label(self, page: str) -> None:
        html = render_page(page, page_context(page, NEW_RULE_SEASON))
        assert html.strip(), f"{page} rendered an empty string"
        assert PAGE_MARKERS[page] in html, f"{page} did not render its page"
        assert label_count(html) == 0, f"{page} labelled a 2026-only render"

    @pytest.mark.parametrize("page", sorted(EXPECTED_PREFIX_BLOCKS))
    def test_an_unwired_page_labels_every_block(self, page: str) -> None:
        """Strip every scope from a 2026 context: each block must then label, not go bare."""
        html = render_page(page, _without_scopes(page_context(page, NEW_RULE_SEASON)))
        assert PAGE_MARKERS[page] in html
        assert label_count(html) == EXPECTED_PREFIX_BLOCKS[page]

    def test_the_partial_with_an_empty_context_labels_once(self) -> None:
        """The unknown-scope fail-safe: no context at all is the un-wired case, and it labels."""
        html = templates.env.get_template("components/_old_rule_label.html").render({})
        assert label_count(html) == 1
        assert LABEL_PHRASE in html.lower()

    def test_the_partial_with_a_none_scope_labels_once(self) -> None:
        html = templates.env.get_template("components/_old_rule_label.html").render(
            {"scope": None}
        )
        assert label_count(html) == 1

    def test_the_partial_with_a_known_empty_scope_renders_nothing(self) -> None:
        html = templates.env.get_template("components/_old_rule_label.html").render(
            {"scope": scope_for(None)}
        )
        assert html.strip() == ""

    def test_a_page_with_an_empty_context_labels_its_block_once(self) -> None:
        """How It Works with no scope at all: its one block labels once, never zero times."""
        context = _without_scopes(page_context("how_it_works.html", NEW_RULE_SEASON))
        html = render_page("how_it_works.html", context)
        assert label_count(html) == 1

    def test_the_rendered_label_is_dated(self) -> None:
        html = render_page(
            "how_it_works.html", page_context("how_it_works.html", PAST_SEASON)
        )
        assert "2026-09-15" in html
