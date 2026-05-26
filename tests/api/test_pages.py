"""Tests for HTML page routes.

Validates the predictions dashboard (landing page), performance page,
backtest page, and HTMX block rendering.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


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
