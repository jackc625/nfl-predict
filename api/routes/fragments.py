"""HTMX fragment route handlers for the NFL Prediction System.

Dedicated router for endpoints that return partial HTML fragments
for HTMX-powered dynamic updates without full page reloads.

Routes:
    GET /fragments/games -- Game grid fragment (swapped into #game-grid)
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from api.dependencies import get_data_service, templates

router = APIRouter(prefix="/fragments", tags=["fragments"])


@router.get("/games")
def games_fragment(
    request: Request,
    week: int | None = Query(None),
    season: int | None = Query(None),
    sort: str = Query("time"),
):
    """Return the game grid as an HTMX fragment.

    Used by the week selector and sort controls to swap the game grid
    without reloading the full page.
    """
    service = get_data_service()
    games = service.get_predictions(season=season, week=week, sort=sort)

    context = {
        "request": request,
        "games": games,
        "current_week": week,
        "current_season": season,
        "current_sort": sort,
    }
    return templates.TemplateResponse(
        "pages/this_week.html", context, block_name="game_grid"
    )
