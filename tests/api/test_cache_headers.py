"""Tests for HTTP Cache-Control headers (OPS-06).

Every page/fragment/export route MUST set ``Cache-Control: public, max-age=60``
so browser back/forward navigation does not hammer the server. The health
endpoint MUST NOT set Cache-Control -- it must always reflect live state.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_this_week_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_performance_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/performance")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_backtest_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_game_detail_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_games_fragment_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/fragments/games?season=2024&week=1")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_performance_fragment_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/fragments/performance?season=2024")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_export_csv_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/api/export/csv?season=2024")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_export_json_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/api/export/json?season=2024")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_health_endpoint_has_no_cache_control(test_client: TestClient) -> None:
    """Health must NOT advertise cache-control; it reflects live state."""
    response = test_client.get("/health")
    header_keys_lower = {k.lower() for k in response.headers}
    assert "cache-control" not in header_keys_lower
