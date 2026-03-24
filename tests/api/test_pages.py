"""Tests for the performance and backtest page routes.

Validates:
- UIAP-03: Historical performance view
- UIAP-05: Backtest dashboard with charts
- UIAP-08: Season selector HTMX swap
"""

from __future__ import annotations


def test_performance_page(test_client):
    """UIAP-03: Historical performance view loads and renders."""
    response = test_client.get("/performance")
    assert response.status_code == 200
    assert "Historical Performance" in response.text
    assert "Season Metrics" in response.text or "No backtest data" in response.text


def test_performance_season_filter(test_client):
    """Performance page accepts season query parameter."""
    response = test_client.get("/performance?season=2024")
    assert response.status_code == 200


def test_backtest_page(test_client):
    """UIAP-05: Backtest dashboard with charts loads and renders."""
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert "Backtest Results" in response.text
    assert "Calibration" in response.text
    assert "CLV" in response.text


def test_performance_fragment(test_client):
    """UIAP-08: Season selector HTMX swap returns partial HTML."""
    response = test_client.get(
        "/fragments/performance?season=2024",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    # Fragment should NOT contain full HTML document structure
    assert "<html" not in response.text


def test_performance_fragment_all_seasons(test_client):
    """Fragment route returns content when no season is specified."""
    response = test_client.get(
        "/fragments/performance",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200


def test_backtest_page_has_chart_containers(test_client):
    """Backtest page includes all 4 chart container sections."""
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert "Calibration Reliability" in response.text
    assert "Cumulative CLV" in response.text
    assert "Season Comparison" in response.text
    assert "Betting Equity Curves" in response.text


def test_backtest_page_responsive_grid(test_client):
    """Backtest page uses responsive 2x2 grid layout."""
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert "grid-cols-1 lg:grid-cols-2" in response.text


def test_performance_page_htmx_returns_block(test_client):
    """Performance page with HX-Request header returns only the block."""
    response = test_client.get(
        "/performance",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    # Should be a fragment, not a full page
    assert "<!DOCTYPE" not in response.text
