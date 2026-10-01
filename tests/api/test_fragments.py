"""Tests for HTMX fragment endpoints.

Validates that fragment routes return partial HTML suitable for
HTMX innerHTML swaps, not full page responses.
"""

from __future__ import annotations

import pytest
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


# ---------------------------------------------------------------------------
# Phase 18: /fragments/season route tests (DASH-10)
# ---------------------------------------------------------------------------
# Mirror test_betting_fragment*: block-only render + whitelist fallback.


def test_season_fragment_returns_block_only(test_client: TestClient):
    """D-11: GET /fragments/season?season=<fixture season> returns ONLY the
    season_tracking_content block (no full HTML document) for an innerHTML swap,
    and the body carries season content."""
    from tests.api.conftest import _FIXTURE_SEASONS

    season = max(_FIXTURE_SEASONS)
    response = test_client.get(
        f"/fragments/season?season={season}",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    html = response.text
    # Block only: no full HTML document wrapper.
    assert "<!DOCTYPE" not in html
    assert "<html" not in html
    assert "<nav" not in html
    # ... but DOES contain season-content markers (a section heading + a KPI
    # label) and the selected season's cumulative chart marker.
    assert "Cumulative Accuracy" in html
    assert "Winner hit rate" in html
    assert f'data-chart-id="season_cumulative_{season}"' in html


def test_season_fragment_scope_whitelist(test_client: TestClient):
    """T-V5-01: an out-of-range season falls back to the latest available season
    rather than erroring or interpolating raw input into a cache-id."""
    from tests.api.conftest import _FIXTURE_SEASONS

    latest = max(_FIXTURE_SEASONS)
    response = test_client.get(
        "/fragments/season?season=1999",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    html = response.text
    # Falls back to the latest season's content; never builds a _1999 id.
    assert f'data-chart-id="season_cumulative_{latest}"' in html
    assert 'data-chart-id="season_cumulative_1999"' not in html


def test_season_fragment_default_no_param(test_client: TestClient):
    """No season param on the fragment route resolves to the latest season
    (D-01), same dynamic default the full page uses."""
    from tests.api.conftest import _FIXTURE_SEASONS

    latest = max(_FIXTURE_SEASONS)
    response = test_client.get(
        "/fragments/season",
        headers={"HX-Request": "true"},
    )
    assert response.status_code == 200
    html = response.text
    assert "<!DOCTYPE" not in html
    assert f'data-chart-id="season_cumulative_{latest}"' in html


# 33.2 review C2 WR-03: a malformed query string degrades to the default on every fragment
# route, exactly as it does on the full pages -- never a 500.
_MALFORMED = ["abc", "--5", "\u00b2", "", " 12 ", "1.5"]


@pytest.mark.parametrize("raw", _MALFORMED)
@pytest.mark.parametrize(
    "route",
    [
        "/fragments/games?week={raw}",
        "/fragments/games?season={raw}",
        "/fragments/performance?season={raw}",
        "/fragments/season?season={raw}",
    ],
)
def test_a_malformed_fragment_query_degrades_rather_than_500ing(
    test_client: TestClient, route: str, raw: str
):
    response = test_client.get(route.format(raw=raw))
    assert response.status_code == 200, (
        f"{route.format(raw=raw)!r} returned {response.status_code}; an unparseable value "
        "degrades to the default, it does not raise"
    )
