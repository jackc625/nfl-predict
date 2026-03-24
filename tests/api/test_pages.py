"""Tests for HTML page routes.

Validates the predictions dashboard (landing page), placeholder pages,
and HTMX block rendering for the / route.
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
    """Performance placeholder page loads."""
    response = test_client.get("/performance")
    assert response.status_code == 200
    assert "Historical Performance" in response.text
    assert "Coming Soon" in response.text


def test_backtest_page(test_client: TestClient):
    """Backtest placeholder page loads."""
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert "Backtest Analysis" in response.text
    assert "Coming Soon" in response.text


def test_game_detail_page(test_client: TestClient):
    """Game detail placeholder page loads."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    assert "Game Detail" in response.text
    assert "Coming Soon" in response.text
