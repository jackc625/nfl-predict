"""Tests for HTTP Cache-Control headers (OPS-06).

Every page/fragment/export route MUST set ``Cache-Control: public, max-age=60``
so browser back/forward navigation does not hammer the server. The health
endpoint MUST NOT set Cache-Control -- it must always reflect live state.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


def test_this_week_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_track_record_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/track-record")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_how_it_works_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/how-it-works")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


@pytest.mark.parametrize("url", ["/performance", "/backtest", "/insights", "/betting"])
def test_a_retired_page_redirect_has_cache_control(
    test_client: TestClient, url: str
) -> None:
    """A 301 is cacheable by default; the header bounds it to a minute so it can be re-pointed."""
    response = test_client.get(url, follow_redirects=False)
    assert response.status_code == 301
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


def test_health_endpoint_has_no_cache_control(
    test_client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Health must NOT advertise cache-control; it reflects live state.

    Patches ``api.routes.health.DB_PATH`` to a tmp path so the test does
    not read from the host filesystem's production cache file. Without
    the patch, ``cache_ready`` would reflect whatever ``data/web_cache.duckdb``
    happens to exist on the dev machine, leaking host state into the
    test and exercising different code paths between CI and local runs.
    The Cache-Control assertion holds either way, but routing through a
    deterministic tmp path lets the test catch a regression where /health
    starts emitting Cache-Control under specific cache states.
    """
    monkeypatch.setattr("api.routes.health.DB_PATH", tmp_path / "no.duckdb")
    response = test_client.get("/health")
    header_keys_lower = {k.lower() for k in response.headers}
    assert "cache-control" not in header_keys_lower
