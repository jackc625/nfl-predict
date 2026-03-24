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
    assert "bg-red-100" in html    # low


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
    response = test_client.get(
        "/", headers={"HX-Request": "true"}
    )
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
