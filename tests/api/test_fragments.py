"""Tests for HTMX fragment endpoints.

Validates that fragment routes return partial HTML suitable for
HTMX innerHTML swaps, not full page responses.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_games_fragment_returns_partial(test_client: TestClient):
    """UIAP-08: HTMX fragment returns partial HTML, not full page."""
    response = test_client.get(
        "/fragments/games?week=1&season=2024",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    html = response.text
    # Fragment should NOT contain full page structure
    assert "<html" not in html
    assert "<head" not in html
    assert "<nav" not in html
    # But should contain game card content
    assert "game-card" in html or "No predictions" in html


def test_games_fragment_contains_predictions(test_client: TestClient):
    """Fragment contains prediction data from sample games."""
    response = test_client.get(
        "/fragments/games?week=1&season=2024",
    )
    assert response.status_code == 200
    html = response.text
    # Should have prediction column headers
    assert "Win Prob" in html
    assert "Spread" in html
    assert "Total" in html


def test_games_fragment_sort_confidence(test_client: TestClient):
    """Sort=confidence parameter works on fragment endpoint."""
    response = test_client.get("/fragments/games?sort=confidence")
    assert response.status_code == 200


def test_games_fragment_sort_edge(test_client: TestClient):
    """Sort=edge parameter works on fragment endpoint."""
    response = test_client.get("/fragments/games?sort=edge")
    assert response.status_code == 200


def test_games_fragment_empty_week(test_client: TestClient):
    """Fragment for a week with no games shows empty state."""
    response = test_client.get("/fragments/games?week=99&season=2024")
    assert response.status_code == 200
    assert "No predictions available" in response.text


def test_games_fragment_no_params(test_client: TestClient):
    """Fragment with no params returns all games."""
    response = test_client.get("/fragments/games")
    assert response.status_code == 200
    html = response.text
    # Should have game content (3 sample games across 2 seasons)
    assert "game-card" in html or "No predictions" in html
