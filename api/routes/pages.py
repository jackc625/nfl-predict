"""HTML page route handlers for the NFL Prediction System.

Serves full HTML pages using Jinja2Blocks templates. When an HX-Request
header is present, only the relevant block is returned (HTMX fragment).

Routes:
    GET /            -- This Week's predictions dashboard (landing page)
    GET /performance -- Historical performance view (placeholder)
    GET /backtest    -- Backtest analysis view (placeholder)
    GET /games/{id}  -- Game detail drill-down (placeholder)
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from api.dependencies import get_data_service, templates

router = APIRouter(tags=["pages"])


@router.get("/")
def this_week_page(
    request: Request,
    week: int | None = Query(None),
    season: int | None = Query(None),
    sort: str = Query("time"),
):
    """Serve the predictions dashboard (landing page).

    Fetches predictions for the selected (or latest) week and renders
    the full page. If the request comes from HTMX, returns only the
    game_grid block.
    """
    service = get_data_service()
    available_weeks = service.get_available_weeks(season=season)
    cache_meta = service.get_cache_meta()

    # Default to latest available week if none specified
    if week is None and available_weeks:
        week = available_weeks[0]["week"]
        season = available_weeks[0]["season"]

    games = service.get_predictions(season=season, week=week, sort=sort)

    context = {
        "request": request,
        "games": games,
        "available_weeks": available_weeks,
        "current_week": week,
        "current_season": season,
        "current_sort": sort,
        "current_path": "/",
        "cache_meta": cache_meta,
    }

    block_name = "game_grid" if request.headers.get("HX-Request") else None
    return templates.TemplateResponse(
        "pages/this_week.html", context, block_name=block_name
    )


@router.get("/performance")
def performance_page(request: Request):
    """Serve the historical performance page (placeholder)."""
    service = get_data_service()
    cache_meta = service.get_cache_meta()
    return templates.TemplateResponse(
        "pages/performance.html",
        {
            "request": request,
            "current_path": "/performance",
            "cache_meta": cache_meta,
        },
    )


@router.get("/backtest")
def backtest_page(request: Request):
    """Serve the backtest analysis page (placeholder)."""
    service = get_data_service()
    cache_meta = service.get_cache_meta()
    return templates.TemplateResponse(
        "pages/backtest.html",
        {
            "request": request,
            "current_path": "/backtest",
            "cache_meta": cache_meta,
        },
    )


@router.get("/games/{game_id}")
def game_detail_page(request: Request, game_id: str):
    """Serve the game detail drill-down page (placeholder)."""
    service = get_data_service()
    cache_meta = service.get_cache_meta()
    return templates.TemplateResponse(
        "pages/game_detail.html",
        {
            "request": request,
            "game_id": game_id,
            "current_path": f"/games/{game_id}",
            "cache_meta": cache_meta,
        },
    )
