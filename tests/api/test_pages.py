"""Tests for HTML page routes.

Validates the predictions dashboard (landing page), performance page,
backtest page, and HTMX block rendering.

Plan 31-15 adds the D31-26 regression: the week selector was PARAMETERISED IN PLACE so that /
and /bets share one partial, and this module pins the claim that the This Week page renders
byte-identically after that edit. It lives here rather than beside the bets tests because it
is a This Week regression -- the next person to change / needs to meet it.
"""

from __future__ import annotations

import re
from collections import Counter

from fastapi.testclient import TestClient

from api.dependencies import get_db, templates
from tests.api.week_selector_snapshot import (
    FAILURE_EVENTS,
    PRE_PARAM_CONTEXT,
    extract_selector,
    read_snapshot,
)


def test_this_week_page(test_client: TestClient):
    """UIAP-02: Landing page shows games with predictions."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    # Page title and heading
    assert "This Week" in html
    # Prediction column headers on game cards
    assert "Win Prob" in html
    assert "Spread" in html
    assert "Total" in html


def test_this_week_page_has_game_cards(test_client: TestClient):
    """Game cards render with team names from sample data."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    # Sample data has KC, BUF, GB, PHI for 2024 week 1
    assert "game-card" in html
    assert "View Details" in html


def test_this_week_page_responsive_grid(test_client: TestClient):
    """D-07: Grid uses responsive column classes."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "grid-cols-1" in html
    assert "md:grid-cols-2" in html
    assert "lg:grid-cols-3" in html


def test_this_week_page_confidence_badges(test_client: TestClient):
    """D-08: Confidence badges with color-coded levels."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    # Sample data contains high, medium, and low confidence values
    assert "bg-green-100" in html  # high
    assert "bg-amber-100" in html  # medium
    assert "bg-red-100" in html  # low


def test_this_week_page_with_week_filter(test_client: TestClient):
    """Week and season query params filter predictions."""
    response = test_client.get("/?week=1&season=2024")
    assert response.status_code == 200
    html = response.text
    assert "This Week" in html


def test_this_week_page_with_sort(test_client: TestClient):
    """Sort query param changes game ordering."""
    response = test_client.get("/?sort=confidence")
    assert response.status_code == 200


def test_this_week_htmx_returns_fragment(test_client: TestClient):
    """UIAP-08: HX-Request header returns game_grid block only."""
    response = test_client.get("/", headers={"HX-Request": "true"})
    assert response.status_code == 200
    html = response.text
    # Fragment should NOT contain full page elements
    assert "<html" not in html
    assert "<head" not in html
    assert "<nav" not in html
    # But should contain game content or empty state
    assert "game-card" in html or "No predictions" in html


def test_this_week_empty_state(empty_test_client: TestClient):
    """Shows empty state when no predictions exist."""
    response = empty_test_client.get("/")
    assert response.status_code == 200
    assert "No predictions available" in response.text


def test_performance_page(test_client: TestClient):
    """UIAP-03: Historical performance view loads and renders."""
    response = test_client.get("/performance")
    assert response.status_code == 200
    assert "Historical Performance" in response.text
    assert "Season Metrics" in response.text or "No backtest data" in response.text


def test_performance_season_filter(test_client: TestClient):
    """Performance page accepts season query parameter."""
    response = test_client.get("/performance?season=2024")
    assert response.status_code == 200


def test_backtest_page(test_client: TestClient):
    """UIAP-05: Backtest dashboard with charts loads and renders."""
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert "Backtest Results" in response.text
    assert "Calibration" in response.text
    assert "CLV" in response.text


def test_performance_fragment(test_client: TestClient):
    """UIAP-08: Season selector HTMX swap returns partial HTML."""
    response = test_client.get(
        "/fragments/performance?season=2024",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    # Fragment should NOT contain full HTML document structure
    assert "<html" not in response.text


def test_performance_fragment_all_seasons(test_client: TestClient):
    """Fragment route returns content when no season is specified."""
    response = test_client.get(
        "/fragments/performance",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200


def test_backtest_page_has_chart_containers(test_client: TestClient):
    """Backtest page includes all 4 chart container sections."""
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert "Calibration Reliability" in response.text
    assert "Cumulative CLV" in response.text
    assert "Season Comparison" in response.text
    assert "Betting Equity Curves" in response.text


def test_backtest_page_responsive_grid(test_client: TestClient):
    """Backtest page uses responsive 2x2 grid layout."""
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert "grid-cols-1 lg:grid-cols-2" in response.text


def test_performance_page_htmx_returns_block(test_client: TestClient):
    """Performance page with HX-Request header returns only the block."""
    response = test_client.get(
        "/performance",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    # Should be a fragment, not a full page
    assert "<!DOCTYPE" not in response.text


def test_game_detail_page(test_client: TestClient):
    """UIAP-04: Game detail page shows feature importances and market comparison."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    assert "Feature Importance" in html
    assert "Prediction vs Market" in html
    # Team header
    assert "BUF" in html
    assert "KC" in html


def test_game_detail_team_context(test_client: TestClient):
    """Game detail shows team context (Elo, form, H2H)."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    assert "Team Context" in html
    assert "Elo Ratings" in html
    assert "1550" in html  # home Elo


def test_game_detail_venue_weather(test_client: TestClient):
    """Game detail shows venue and weather information."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    assert "Venue &amp; Weather" in html or "Venue & Weather" in html
    assert "Arrowhead" in html
    assert "Grass" in html


def test_game_detail_result_overlay(test_client: TestClient):
    """D-15: Completed game shows result and correct/incorrect indicator."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    # Score overlay (away_score - home_score)
    assert "20 - 27" in html
    # wp_prob=0.62 > 0.5 (predicted home win), home_score=27 > away_score=20 (home won)
    assert "Correct" in html


def test_game_detail_feature_chart(test_client: TestClient):
    """D-13: Feature importance bar chart with Plotly."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    assert 'Plotly.newPlot("feature-chart"' in html
    assert "elo_diff" in html  # from sample feature importances


def test_game_detail_multi_target_importances(test_client: TestClient):
    """UIAP-04: Feature importance tabs for WP, ATS, and O/U targets."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    # Tab buttons for all three targets
    assert 'id="tab-wp"' in html
    assert 'id="tab-ats"' in html
    assert 'id="tab-ou"' in html
    # Tab strip has tablist role
    assert 'role="tablist"' in html
    # All feature names from test data appear in the JSON blob
    assert "elo_diff" in html
    assert "rolling_off_epa" in html
    assert "rolling_total_epa" in html
    # showFeatureChart function exists
    assert "showFeatureChart" in html


def test_game_detail_not_found(test_client: TestClient):
    """Missing game shows 404-style empty state."""
    response = test_client.get("/games/nonexistent_game")
    assert response.status_code == 200
    assert "Game not found" in response.text
    assert "Back to This Week" in response.text


def test_game_detail_export_buttons(test_client: TestClient):
    """Game detail page includes export buttons."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    assert "/api/export/csv" in html
    assert "/api/export/json" in html


# ---------------------------------------------------------------------------
# Phase 16: /insights page (DASH-10) route tests
# ---------------------------------------------------------------------------
# Activated by Plan 16-03 (route, template, and nav link are now wired).


def test_insights_page_200(test_client: TestClient):
    """DASH-10: /insights returns 200 with all three section headings."""
    response = test_client.get("/insights")
    assert response.status_code == 200
    html = response.text
    assert "Model Insights" in html
    assert "Calibration" in html
    assert "Feature Importance" in html
    assert "Model vs Market" in html


def test_insights_page_cache_control(test_client: TestClient):
    """D-20 / PAGE_CACHE_CONTROL: Cache-Control header is set on TemplateResponse."""
    response = test_client.get("/insights")
    assert response.status_code == 200
    cc = response.headers.get("Cache-Control", "")
    assert "public" in cc and "max-age" in cc


def test_insights_page_has_nav_link(test_client: TestClient):
    """D-16: Insights link appears in rendered HTML (desktop + mobile menus)."""
    response = test_client.get("/insights")
    # Exactly two occurrences expected: desktop nav + mobile menu.
    assert response.text.count('href="/insights"') >= 2


def test_insights_page_renders_expected_chart_ids(test_client: TestClient):
    """REVIEWS Codex HIGH #7: route reads exactly the 9 insights chart IDs
    plus the existing ``calibration`` chart_id (D-22)."""
    from api.charts import INSIGHTS_CHART_IDS

    assert len(INSIGHTS_CHART_IDS) == 9
    response = test_client.get("/insights")
    html = response.text
    # conftest inserts marker divs for each chart_id; assert all 9 markers render.
    for chart_id in INSIGHTS_CHART_IDS:
        assert f'data-chart-id="{chart_id}"' in html, f"Missing chart_id: {chart_id}"
    # WP calibration reuses the existing `calibration` chart_id.
    assert 'data-chart-id="calibration"' in html


def test_insights_page_empty_db(empty_test_client: TestClient):
    """D-29: empty DB renders empty-state cards, not 500."""
    response = empty_test_client.get("/insights")
    assert response.status_code == 200
    assert "Chart unavailable" in response.text


# ---------------------------------------------------------------------------
# Phase 17: /betting page + /fragments/betting route scaffolds (skip-gated)
# ---------------------------------------------------------------------------
# Full assertion bodies now; gated by ``@pytest.mark.skip(reason="activated in
# 17-04")`` until Plan 17-04 lands the ``betting_page`` handler, the
# ``/fragments/betting`` route, ``web/templates/pages/betting.html``, and the
# ``base.html`` nav link. Plan 17-04 activates them by deleting the skip marker
# (Phase 16 skip-gated pattern). Default scope = "recommended" (D-17).


def test_betting_page_200(test_client: TestClient):
    """DASH-10: GET /betting returns 200 with the four section headings, a
    Cache-Control header, and a nav link to /betting (desktop + mobile)."""
    response = test_client.get("/betting")
    assert response.status_code == 200
    html = response.text
    # Section headings (D-04 order: KPI strip -> Equity -> ROI -> Edge).
    assert "Equity" in html
    assert "ROI" in html
    assert "Edge" in html
    # Cache-Control set on the returned TemplateResponse (Phase 15 D-07).
    cc = response.headers.get("Cache-Control", "")
    assert "public" in cc and "max-age" in cc
    # Nav link present in both desktop nav and mobile menu (D-02).
    assert html.count('href="/betting"') >= 2


def test_betting_page_renders_recommended_chart_ids(test_client: TestClient):
    """Default load (scope=recommended) consumes the betting_*_recommended
    chart_id markers the conftest fixture inserts."""
    from tests.api.conftest import _BETTING_CHART_BASES, BETTING_SCOPES

    assert "recommended" in BETTING_SCOPES
    response = test_client.get("/betting")
    html = response.text
    # Every recommended-scope chart-HTML marker should render. The two JSON-blob
    # families (kpis / roi_table) are decoded server-side, not emitted as markers.
    for base in _BETTING_CHART_BASES:
        if base in ("betting_kpis", "betting_roi_table"):
            continue
        chart_id = f"{base}_recommended"
        assert f'data-chart-id="{chart_id}"' in html, f"Missing {chart_id}"


def test_betting_fragment(test_client: TestClient):
    """D-17: GET /fragments/betting?scope=all with HX-Request returns only the
    betting_content block (no full document), and the scope switch changes which
    betting_*_<scope> chart ids appear."""
    response = test_client.get(
        "/fragments/betting?scope=all",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    html = response.text
    # Fragment only: no full HTML document wrapper.
    assert "<!DOCTYPE" not in html
    assert "<html" not in html
    # scope=all surfaces the betting_*_all chart markers.
    assert 'data-chart-id="betting_equity_all"' in html
    # ... and not the recommended-scope equity marker.
    assert 'data-chart-id="betting_equity_recommended"' not in html


def test_betting_fragment_scope_whitelist(test_client: TestClient):
    """Security V5: an out-of-whitelist scope falls back to the default
    ("recommended") rather than erroring or interpolating raw input."""
    response = test_client.get(
        "/fragments/betting?scope=bogus",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    # Falls back to recommended-scope content.
    assert 'data-chart-id="betting_equity_recommended"' in response.text


def test_betting_empty_db(empty_test_client: TestClient):
    """Empty DB renders empty-state cards (not a 500): no betting_* chart_ids
    are cached, so every chart slot falls back to 'Chart unavailable'."""
    response = empty_test_client.get("/betting")
    assert response.status_code == 200
    assert "Chart unavailable" in response.text


# ---------------------------------------------------------------------------
# Phase 18: /season page (DASH-07/08/09/10) route tests
# ---------------------------------------------------------------------------
# Mirror the test_betting_page_* structure. The dynamic-default + whitelist
# expectations are DERIVED from _FIXTURE_SEASONS so no literal year is asserted
# (D-01 / T-V5-01; 18-RESEARCH Pitfall 1).


def test_season_page_200(test_client: TestClient):
    """DASH-09/10: GET /season returns 200 with the two section headings, a
    Cache-Control header, and a nav link to /season (desktop + mobile)."""
    response = test_client.get("/season")
    assert response.status_code == 200
    html = response.text
    # Page heading + the two stacked section headings (D-07 / D-08).
    assert "Season Tracking" in html
    assert "Cumulative Accuracy" in html
    assert "Weekly Performance" in html
    # Cache-Control set on the returned TemplateResponse (Phase 15 D-07).
    cc = response.headers.get("Cache-Control", "")
    assert "public" in cc and "max-age" in cc
    assert cc == "public, max-age=60"
    # Nav link present in both desktop nav and mobile menu (D-09).
    assert html.count('href="/season"') >= 2


def test_season_page_default_is_latest_season(test_client: TestClient):
    """D-01 / T-V5-01: no season param selects the dynamically-resolved latest
    season, and an out-of-range season falls back to that same latest -- without
    500ing and without hardcoding a year in the assertion."""
    from tests.api.conftest import _FIXTURE_SEASONS

    latest = max(_FIXTURE_SEASONS)

    # No param -> latest season is selected in the rendered selector.
    response = test_client.get("/season")
    assert response.status_code == 200
    html = response.text
    # The selector marks the resolved season's <option> as selected.
    assert f'value="{latest}" selected' in html
    # The latest season's cumulative chart marker renders (proves the route
    # built the season_cumulative_<latest> cache-id from the dynamic default).
    assert f'data-chart-id="season_cumulative_{latest}"' in html

    # Out-of-range season (1999 is not in _FIXTURE_SEASONS) -> fall back to
    # latest, not a 500 and not a raw-interpolated id (T-V5-01 whitelist).
    fallback = test_client.get("/season?season=1999")
    assert fallback.status_code == 200
    fb_html = fallback.text
    assert f'value="{latest}" selected' in fb_html
    assert f'data-chart-id="season_cumulative_{latest}"' in fb_html
    assert 'data-chart-id="season_cumulative_1999"' not in fb_html


def test_season_page_in_range_season_passthrough(test_client: TestClient):
    """An in-range season param renders that season's content (not the latest)."""
    from tests.api.conftest import _FIXTURE_SEASONS

    # Pick an older in-range season distinct from the latest.
    older = min(_FIXTURE_SEASONS)
    response = test_client.get(f"/season?season={older}")
    assert response.status_code == 200
    html = response.text
    assert f'value="{older}" selected' in html
    assert f'data-chart-id="season_cumulative_{older}"' in html


def test_season_page_empty_db(empty_test_client: TestClient):
    """D-02: empty DB renders the whole-season empty state (not a 500). With no
    seasons present, _normalize_season returns None and the page shows the
    'No completed games yet' empty state."""
    response = empty_test_client.get("/season")
    assert response.status_code == 200
    html = response.text
    # Either the whole-season empty state or the per-chart fallback is acceptable
    # per the plan; with an empty DB the whole-season guard fires.
    assert "No completed games yet" in html or "Chart unavailable" in html


def test_season_page_htmx_returns_block(test_client: TestClient):
    """D-11: GET /season with HX-Request returns only the season_tracking_content
    block, not the full HTML document."""
    response = test_client.get("/season", headers={"HX-Request": "true"})
    assert response.status_code == 200
    html = response.text
    assert "<!DOCTYPE" not in html
    assert "<html" not in html
    # Still contains season content (a section heading).
    assert "Cumulative Accuracy" in html


def test_season_error_state_wired_and_distinct_from_empty(test_client: TestClient):
    """DASH-09 / D-17: the /season page wires `_error_state.html` for the failed-
    swap path, and the error state (red) is DISTINCT from the empty state (gray).

    The page reads cached HTML/JSON only and the error wiring is HTMX-native: a
    hidden ``<template id="season-error-template">`` carries the server-rendered
    ``_error_state.html`` markup (with the LOCKED ``/season`` error copy) and the
    selector's ``hx-on::response-error`` handler copies it into ``#season-content``
    on a 4xx/5xx swap (HTMX leaves the target untouched on non-200 by default --
    ``detail.shouldSwap`` is false). This route-level assertion proves the error
    path is reachable and renders the red error copy, deterministically and with
    no network, and that the two states are not conflated.
    """
    response = test_client.get("/season")
    assert response.status_code == 200
    html = response.text

    # The orphaned error component is now wired: the LOCKED red /season error copy
    # is server-rendered into the page (inside the hidden error <template>).
    assert "Could not load season data" in html
    assert "Refresh the page or rebuild the cache" in html

    # The error markup is reachable via the wired HTMX error handler + template,
    # not left blank: the response-error handler and the error <template> exist.
    assert "hx-on::response-error" in html
    assert 'id="season-error-template"' in html

    # Distinction (must NOT be conflated): the error card uses the red _error_state
    # styling, while the empty state uses the gray styling. Assert the error path
    # carries the red error card class so it is not the gray "no data" state.
    assert "bg-red-50" in html

    # The populated fixture renders real season content, so the gray whole-season
    # empty copy must NOT appear here -- error (red) and empty (gray) are distinct
    # states and are not conflated. On a populated page the KPI strip renders (not
    # the whole-season empty state), proving the visible content is real data with
    # the error state held in reserve in the hidden template.
    assert "No completed games yet" not in html
    assert "WP Hit Rate" in html


def test_season_error_copy_renders_via_error_component(empty_test_client: TestClient):
    """D-17: even when the page falls back to the EMPTY state (empty DB), the
    red ERROR component is still wired and reachable -- the two states coexist and
    are distinct. The empty DB shows the gray "No completed games yet" empty state
    in the content area, while the red "Could not load season data" error copy is
    held in the hidden error <template> for the failed-swap path.
    """
    response = empty_test_client.get("/season")
    assert response.status_code == 200
    html = response.text

    # Empty state (gray) is the active content for an empty DB (no completed games).
    assert "No completed games yet" in html

    # Error state (red) is STILL wired and reachable (distinct concern, not shown
    # as the active content): the LOCKED error copy + the red card markup exist.
    assert "Could not load season data" in html
    assert "bg-red-50" in html


# ---------------------------------------------------------------------------
# D31-26: the shared week selector, parameterised in place (plan 31-15)
# ---------------------------------------------------------------------------


def test_the_this_week_selector_renders_byte_identically(
    test_client: TestClient,
) -> None:
    """The This Week page's selector markup equals a snapshot recorded BEFORE the edit (D31-26)."""
    rendered = extract_selector(test_client.get("/").text).replace("\r\n", "\n")
    assert rendered == read_snapshot("week_selector_this_week.html")


def test_the_prev_next_branch_renders_byte_identically() -> None:
    """The enabled prev/next branch -- the two baked-in sort URLs -- is byte-identical too.

    The This Week fixture yields ONE week, under which both buttons render `disabled` and their
    hx-get URLs are never emitted. Rendering the partial directly against a three-week context is
    the only way those two of the six hard-wired attributes get compared at all.
    """
    rendered = templates.env.get_template("components/_week_selector.html").render(
        **PRE_PARAM_CONTEXT
    )
    assert (
        rendered.replace("\r\n", "\n").strip()
        == read_snapshot("week_selector_prev_next.html").strip()
    )


def test_the_this_week_page_still_targets_the_games_grid(
    test_client: TestClient,
) -> None:
    """Defaults reproduce every one of the six hard-wired attributes on the shipped page."""
    markup = extract_selector(test_client.get("/").text)
    assert 'hx-get="/fragments/games"' in markup
    assert 'hx-target="#game-grid"' in markup
    assert "hx-include=\"[name='sort']\"" in markup
    assert "hx-include=\"[name='sort'],[name='season']\"" in markup
    assert 'id="season-select"' in markup
    assert 'id="week-select"' in markup
    assert 'for="season-select"' in markup
    assert 'for="week-select"' in markup
    # The shipped page gains NO failure handler and NO indicator: both default to omitted.
    for event in FAILURE_EVENTS:
        assert event not in markup
    # And NO request timeout (plan 31-21). The timeout exists to make hx-on::timeout reachable on
    # /bets; this page has no such handler, so a timeout here would abort a hung request with
    # nowhere to send the event -- a behaviour change on a page that never asked for one.
    assert "hx-request" not in markup, (
        "the shipped This Week page gained a request timeout it never asked for"
    )
    # And NEITHER concurrency attribute (plan 31-24). Serialising the four controls on the swap
    # target and disabling the one that is asking is a /bets decision the owner ruled on, made
    # against a measured /bets defect; this page keeps htmx's shipped queueing behaviour, so a
    # rapid second week change here still resolves rather than cancelling the first request.
    for attribute in ("hx-sync", "hx-disabled-elt"):
        assert attribute not in markup, (
            f"the shipped This Week page gained {attribute}, which it never asked for"
        )


# ---------------------------------------------------------------------------
# The renamed edge band renders the SAME labels and the SAME sort (31-17, D31-23)
# ---------------------------------------------------------------------------
#
# EXTENDED, not rewritten. Plan 31-17 collapsed two duplicate tier helpers into
# ``utils/edge_tier.py`` and renamed the concept to an EDGE BAND. The claim these three tests pin
# is the one that makes that safe: the pages the SPEC says keep their job render exactly what they
# rendered before. The stored column names (``*_confidence``) are deliberately unchanged -- see
# ``utils/edge_tier.py`` for why renaming them was out of scope.


def test_the_landing_page_still_renders_all_three_edge_band_labels(
    test_client: TestClient,
) -> None:
    """The three-label vocabulary on / is UNCHANGED by the collapse."""
    from utils.edge_tier import EDGE_TIER_LABELS

    html = test_client.get("/").text
    assert set(EDGE_TIER_LABELS) == {"low", "medium", "high"}
    # The three colour classes the confidence badge maps the three labels onto, one per band.
    for badge_class in ("bg-green-100", "bg-amber-100", "bg-red-100"):
        assert badge_class in html, (
            f"the {badge_class} badge disappeared from /; an edge band label has moved"
        )


def test_the_landing_page_renders_the_band_it_was_served_and_never_rederives_one(
    test_client: TestClient,
) -> None:
    """The page RENDERS the stored band verbatim; it does not recompute one (UIAP-01, D31-23).

    This is the substantive half. The collapse moved WHERE the band is computed -- into
    ``utils/edge_tier.py``, called at CACHE-BUILD time -- so the regression that matters on the
    page is that the request path still just renders what it was handed. Asserted against the
    SERVED rows rather than against a re-derivation, because the fixture cache is hand-authored
    (its stored bands were never produced by either helper) and re-deriving would test the
    fixture's internal consistency rather than the page's behaviour.

    Every stored band must be a member of the closed vocabulary and must appear in the markup for
    its own game, so a page that silently re-banded a row -- the exact thing the two duplicate
    helpers made easy -- fails here.
    """
    from api.services import DataService
    from utils.edge_tier import EDGE_TIER_LABELS

    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text

    service = DataService(test_client.app.dependency_overrides[get_db]())
    rows = service.get_predictions(season=2024, week=1)
    assert rows, (
        "the fixture served no predictions; this regression would prove nothing"
    )

    served: list[str] = []
    for row in rows:
        for target in ("wp", "ats", "ou"):
            stored_band = row.get(f"{target}_confidence")
            if stored_band is None:
                continue
            assert stored_band in EDGE_TIER_LABELS, (
                f"{row.get('game_id')} {target}: stored band {stored_band!r} is outside the "
                f"closed vocabulary {EDGE_TIER_LABELS}"
            )
            served.append(stored_band)
    assert served, "no band was served; the check would pass vacuously"

    # The badge partial maps each band onto ONE colour class and renders the label title-cased.
    # Comparing the MULTISET of rendered badges against the multiset of served bands is what makes
    # this a re-banding check rather than a spelling check: a page that turned one served "low"
    # into a "high" would leave the vocabulary intact and the counts different.
    #
    # The class triple below is the CONFIDENCE badge's, not the STATUS badge's. Both live on game
    # cards and both use ``bg-green-100``; only the confidence badge carries the matching
    # ``border border-<colour>-200``. Keying on the bare background colour matches the status
    # badge's "Completed" pill and reports it as a mis-banded row -- which is how this assertion
    # first failed, and why the selector is the full triple.
    band_by_class = {"green": "high", "amber": "medium", "red": "low"}
    rendered: list[str] = []
    for colour, band in band_by_class.items():
        pattern = (
            rf"bg-{colour}-100 text-{colour}-\d+ border border-{colour}-200[^>]*>"
            r"([^<]+)</span>"
        )
        for match in re.finditer(pattern, html):
            assert match.group(1).strip().lower() == band, (
                f"a bg-{colour}-100 confidence badge renders {match.group(1)!r}, which is not "
                f"the {band!r} band that colour is reserved for"
            )
            rendered.append(band)

    assert Counter(rendered) == Counter(served), (
        "the bands rendered on / are not the bands the page was served -- the request path "
        f"re-banded a row.\n  served:   {sorted(Counter(served).items())}\n"
        f"  rendered: {sorted(Counter(rendered).items())}"
    )


def test_the_landing_page_sort_by_band_is_unchanged(test_client: TestClient) -> None:
    """The sort controls that order by the band still return 200 and the same row count.

    The band feeds a sort on /, so a changed label would change the ORDER as well as the text.
    Comparing the served row count and the status across the default and the sorted render is the
    behavioural half of "existing page behaviour is unchanged".
    """
    default_html = test_client.get("/").text
    sorted_response = test_client.get("/", params={"sort": "confidence"})
    assert sorted_response.status_code == 200
    assert default_html.count("game-card") == sorted_response.text.count("game-card")
