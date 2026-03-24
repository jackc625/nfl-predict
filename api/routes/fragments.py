"""HTMX fragment route handlers for the NFL Prediction System.

Dedicated router for endpoints that return partial HTML fragments
for HTMX-powered dynamic updates without full page reloads.

Routes:
    GET /fragments/games       -- Game grid fragment (swapped into #game-grid)
    GET /fragments/performance -- Performance content fragment (season swap)
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from api.dependencies import get_data_service, templates
from api.routes.pages import _pivot_season_metrics

router = APIRouter(prefix="/fragments", tags=["fragments"])


@router.get("/games")
def games_fragment(
    request: Request,
    week: str | None = Query(None),
    season: str | None = Query(None),
    sort: str = Query("time"),
):
    """Return the game grid as an HTMX fragment.

    Used by the week selector and sort controls to swap the game grid
    without reloading the full page.
    """
    week_int = int(week) if week and week.strip() else None
    season_int = int(season) if season and season.strip() else None
    service = get_data_service()
    games = service.get_predictions(season=season_int, week=week_int, sort=sort)

    context = {
        "games": games,
        "current_week": week_int,
        "current_season": season_int,
        "current_sort": sort,
    }
    return templates.TemplateResponse(
        request, "pages/this_week.html", context, block_name="game_grid"
    )


@router.get("/performance")
def performance_fragment(
    request: Request,
    season: str | None = Query(None),
):
    """Return the performance_content block for HTMX season swap.

    Renders only the season metrics table portion of the performance
    page, used when the season selector dropdown changes.
    """
    season_int = int(season) if season and season.strip() else None
    service = get_data_service()
    raw_metrics = service.get_backtest_metrics(season=season_int)
    season_metrics = _pivot_season_metrics(raw_metrics)

    context = {
        "request": request,
        "season_metrics": season_metrics,
        "current_season": season_int,
    }
    return templates.TemplateResponse(
        request,
        "pages/performance.html",
        context,
        block_name="performance_content",
    )
