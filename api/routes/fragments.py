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
from api.routes.pages import (
    _annotate_wp_correct,
    _compute_week_summary,
    _pivot_season_metrics,
)

router = APIRouter(prefix="/fragments", tags=["fragments"])


@router.get("/games")
def games_fragment(
    request: Request,
    week: str | None = Query(None),
    season: str | None = Query(None),
    sort: str = Query("time"),
):
    """Return the game grid as an HTMX fragment.

    Used by the week selector, season dropdown, and sort controls to
    swap the game grid without reloading the full page.

    When season changes and week is empty, defaults to the latest week
    for that season.
    """
    week_int = int(week) if week and week.strip() else None
    season_int = int(season) if season and season.strip() else None
    service = get_data_service()

    # When season changes, default to latest week for that season
    if week_int is None and season_int is not None:
        available_weeks = service.get_available_weeks(season=season_int)
        if available_weeks:
            week_int = available_weeks[0]["week"]

    games = service.get_predictions(season=season_int, week=week_int, sort=sort)

    # Compute wp_correct for correct/incorrect indicators (D-03)
    _annotate_wp_correct(games)

    context = {
        "games": games,
        "current_week": week_int,
        "current_season": season_int,
        "current_sort": sort,
        "week_summary": _compute_week_summary(games),
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
