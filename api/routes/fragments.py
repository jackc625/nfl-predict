"""Fragment route handlers for HTMX partial page updates.

Each fragment route returns a specific Jinja2Blocks block from a full
page template, enabling HTMX to swap content without full page reloads.
"""

from __future__ import annotations

from fastapi import APIRouter, Query, Request

from api.dependencies import get_data_service, templates

router = APIRouter(tags=["fragments"])


@router.get("/fragments/performance")
def performance_fragment(
    request: Request,
    season: int | None = Query(None),
):
    """Return the performance_content block for HTMX season swap.

    Renders only the season metrics table portion of the performance
    page, used when the season selector dropdown changes.
    """
    service = get_data_service()
    season_metrics = service.get_backtest_metrics(season=season)

    context = {
        "request": request,
        "season_metrics": season_metrics,
        "current_season": season,
    }
    return templates.TemplateResponse(
        "pages/performance.html",
        context,
        block_name="performance_content",
    )
