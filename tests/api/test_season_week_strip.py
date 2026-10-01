"""The Season page's week-by-week strip, its KPI tiles and its context (redesign Task 14).

The strip is drawn from the per-week records population stores in the season KPI blob
(``api.season_metrics.compute_weekly_records``). Nothing is computed on the request path: the
win/loss bar is sized by CSS flex-grow set to the stored counts.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi.testclient import TestClient

from api.dependencies import templates


def _render_strip(weeks: list[dict[str, int]], current_week: int | None) -> str:
    return templates.env.get_template("components/_week_strip.html").render(
        weeks=weeks, current_week=current_week
    )


def _tile(html: str, week: int) -> str:
    match = re.search(rf'<li data-week="{week}"[^>]*>.*?</li>', html, re.S)
    assert match, f"no tile for week {week}"
    return match.group(0)


class TestTheWeekStrip:
    def test_renders_the_eighteen_regular_season_weeks(self) -> None:
        html = _render_strip([{"week": 1, "wins": 27, "losses": 21}], None)
        assert html.count("<li data-week=") == 18

    def test_a_graded_week_shows_its_record_and_a_win_loss_bar(self) -> None:
        tile = _tile(_render_strip([{"week": 2, "wins": 26, "losses": 22}], None), 2)
        assert "26-22" in tile
        assert "flex: 26 1 0%" in tile
        assert "flex: 22 1 0%" in tile
        assert "bg-green-500" in tile
        assert "bg-red-500" in tile
        assert 'aria-label="Week 2: 26 wins, 22 losses"' in tile
        assert "opacity-40" not in tile

    def test_a_week_with_no_graded_pick_is_dimmed_and_never_reads_0_0(self) -> None:
        tile = _tile(_render_strip([{"week": 1, "wins": 27, "losses": 21}], None), 5)
        assert "opacity-40" in tile
        assert "0-0" not in tile
        assert "bg-green-500" not in tile
        assert 'aria-label="Week 5: no graded picks"' in tile

    def test_the_current_week_is_outlined_and_marked_once(self) -> None:
        html = _render_strip([{"week": 3, "wins": 26, "losses": 22}], 3)
        tile = _tile(html, 3)
        assert "outline-accent" in tile
        assert 'aria-current="true"' in tile
        assert html.count("outline-accent") == 1
        assert html.count('aria-current="true"') == 1

    def test_no_current_week_outlines_nothing(self) -> None:
        html = _render_strip([{"week": 3, "wins": 26, "losses": 22}], None)
        assert "outline-accent" not in html
        assert "aria-current" not in html

    def test_postseason_weeks_extend_the_strip(self) -> None:
        html = _render_strip([{"week": 20, "wins": 2, "losses": 1}], None)
        assert html.count("<li data-week=") == 20
        assert "2-1" in _tile(html, 20)


class _Request:
    def url_for(self, name: str, **path_params: object) -> str:
        return f"/{name}/{path_params.get('path', '')}"


class _Service:
    """The five reads ``_build_season_context`` makes, with a chosen current slate."""

    def __init__(self, slate: tuple[int, int] | None, kpis: dict[str, Any]) -> None:
        self._slate = slate
        self._kpis = kpis

    def get_chart_html(self, chart_id: str) -> str:
        return f'<div data-chart-id="{chart_id}">chart</div>'

    def get_season_kpis(self, season: int) -> dict[str, Any]:
        return self._kpis

    def get_prediction_seasons(self) -> list[int]:
        return [2026, 2025]

    def get_cache_meta(self) -> dict[str, Any]:
        return {}

    def get_current_slate(self) -> tuple[int, int] | None:
        return self._slate


_GRADED_KPIS: dict[str, Any] = {
    "wp_hit_rate": 50.0,
    "wp_decided": 2,
    "wp_hits": 1,
    "ats_hit_rate": None,
    "ats_decided": 0,
    "ats_hits": 0,
    "ou_hit_rate": None,
    "ou_decided": 0,
    "ou_hits": 0,
    "record": "1-1",
}


class TestTheSeasonContext:
    def test_the_current_slate_week_is_passed_only_for_its_own_season(self) -> None:
        from api.routes.pages import _build_season_context

        kpis = _GRADED_KPIS | {"weeks": [{"week": 1, "wins": 1, "losses": 1}]}
        live = _build_season_context(_Service((2026, 4), kpis), 2026, _Request())  # type: ignore[arg-type]
        past = _build_season_context(_Service((2026, 4), kpis), 2025, _Request())  # type: ignore[arg-type]
        offseason = _build_season_context(_Service(None, kpis), 2026, _Request())  # type: ignore[arg-type]
        assert live["current_slate_week"] == 4
        assert past["current_slate_week"] is None
        assert offseason["current_slate_week"] is None

    def test_a_blob_without_weeks_renders_no_strip(self) -> None:
        """A cache built before the strip existed has no ``weeks`` key: no strip, and no error."""
        from api.routes.pages import _build_season_context

        service: Any = _Service(None, dict(_GRADED_KPIS))
        context = _build_season_context(service, 2026, _Request())  # type: ignore[arg-type]
        html = templates.env.get_template("pages/season.html").render(context)
        assert "Winner hit rate" in html
        assert "Week by week" not in html
        assert "<li data-week=" not in html


class TestTheSeasonPage:
    def test_renders_the_week_strip_from_the_kpi_blob(
        self, test_client: TestClient
    ) -> None:
        from tests.api.conftest import _FIXTURE_SEASONS

        html = test_client.get(f"/season?season={max(_FIXTURE_SEASONS)}").text
        assert "Week by week" in html
        strip = html.split("Week by week", 1)[1]
        assert "14-9" in _tile(strip, 1)
        assert "12-11" in _tile(strip, 2)

    def test_the_kpi_tiles_use_the_bet_type_names_and_counts(
        self, test_client: TestClient
    ) -> None:
        from tests.api.conftest import _FIXTURE_SEASONS

        html = test_client.get(f"/season?season={max(_FIXTURE_SEASONS)}").text
        for label in (
            "Winner hit rate",
            "Spread hit rate",
            "Totals hit rate",
            "Record (W-L)",
        ):
            assert label in html, label
        assert "67.7%" in html
        assert "11 of 16" in html  # wp_hits of wp_decided, both stored in the blob
        assert "straight-up winner picks" in html
        assert "WP Hit Rate" not in html

    def test_the_season_heading_is_inside_the_swapped_block(
        self, test_client: TestClient
    ) -> None:
        """A season swap must never leave last season's name above this season's numbers."""
        from tests.api.conftest import _FIXTURE_SEASONS

        older = min(_FIXTURE_SEASONS)
        response = test_client.get(
            f"/fragments/season?season={older}", headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
        assert f"{older} Season" in response.text
