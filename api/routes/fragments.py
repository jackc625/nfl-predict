"""HTMX fragment route handlers for the NFL Prediction System.

Dedicated router for endpoints that return partial HTML fragments
for HTMX-powered dynamic updates without full page reloads.

Routes:
    GET /fragments/games       -- Game grid fragment (swapped into #game-grid)
    GET /fragments/performance -- Performance content fragment (season swap)
    GET /fragments/betting     -- Betting content fragment (All/Recommended scope swap)
    GET /fragments/season      -- Season tracking content fragment (season swap)
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request

from api.dependencies import get_data_service, templates
from api.routes.pages import (
    _DEFAULT_BETTING_SCOPE,
    PAGE_CACHE_CONTROL,
    _annotate_wp_correct,
    _build_betting_context,
    _build_season_context,
    _compute_week_summary,
    _normalize_betting_scope,
    _normalize_season,
    _parse_int_param,
    _pivot_season_metrics,
    _rows_seasons,
    _this_week_grid_context,
)
from api.services import DataService

router = APIRouter(prefix="/fragments", tags=["fragments"])


@router.get("/games")
def games_fragment(
    request: Request,
    week: str | None = Query(None),
    season: str | None = Query(None),
    sort: str = Query("time"),
    service: DataService = Depends(get_data_service),
):
    """Return the game grid as an HTMX fragment.

    Used by the week selector, season dropdown, and sort controls to
    swap the game grid without reloading the full page.

    When season changes and week is empty, defaults to the latest week
    for that season.

    Every integer goes through ``pages._parse_int_param`` (33.2 review C2 WR-03): a bare
    ``int(week)`` made ``/fragments/games?week=abc`` a 500, and an unparseable value now
    degrades to the default exactly as it does on the full pages.
    """
    week_int = _parse_int_param(week)
    season_int = _parse_int_param(season)

    # When season changes, default to latest week for that season
    if week_int is None and season_int is not None:
        available_weeks = service.get_available_weeks(season=season_int)
        if available_weeks:
            week_int = available_weeks[0]["week"]

    games = service.get_predictions(season=season_int, week=week_int, sort=sort)

    # Compute wp_correct on a fresh list so the DataService TTLCache source
    # is never mutated (plan 15-02 review item #4).
    games = _annotate_wp_correct(games)
    # The SAME games / groups / headliner this_week_page builds, so a week change through HTMX
    # renders exactly what a full navigation renders.
    grid = _this_week_grid_context(service, games, season_int, week_int, sort, request)

    context = {
        "games": grid["games"],
        "slate_groups": grid["slate_groups"],
        "headliner": grid["headliner"],
        "current_week": week_int,
        "current_season": season_int,
        "current_sort": sort,
        "week_summary": _compute_week_summary(games),
        # The same scope this_week_page builds for the game_grid block (R16 / D33.2-07).
        "old_rule_scope": DataService.old_rule_scope(
            _rows_seasons(games) + _rows_seasons(grid["headliner"]["bets"])
        ),
    }
    template_response = templates.TemplateResponse(
        request, "pages/this_week.html", context, block_name="game_grid"
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/performance")
def performance_fragment(
    request: Request,
    season: str | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Return the performance_content block for HTMX season swap.

    Renders only the season metrics table portion of the performance
    page, used when the season selector dropdown changes.
    """
    season_int = _parse_int_param(season)
    raw_metrics = service.get_backtest_metrics(season=season_int)
    season_metrics = _pivot_season_metrics(raw_metrics)

    context = {
        "request": request,
        "season_metrics": season_metrics,
        "current_season": season_int,
        # The same scope performance_page builds for the performance_content block (R16 / D33.2-07).
        "season_metrics_old_rule_scope": DataService.old_rule_scope(
            _rows_seasons(season_metrics)
        ),
    }
    template_response = templates.TemplateResponse(
        request,
        "pages/performance.html",
        context,
        block_name="performance_content",
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/betting")
def betting_fragment(
    request: Request,
    scope: str = Query(_DEFAULT_BETTING_SCOPE),
    service: DataService = Depends(get_data_service),
):
    """Return the ``betting_content`` block for the HTMX All/Recommended swap.

    Renders only the swappable portion of the betting page (KPI strip + the
    equity / ROI / edge sections) for the selected *scope*, used when the
    scope toggle flips between "Recommended" and "All bets" (D-17).

    ``scope`` is whitelisted to {"all", "recommended"} (default ``"recommended"``)
    via the same chokepoint the full page uses, and the context is built by the
    shared ``_build_betting_context`` helper so the cached-read logic is not
    duplicated. Reads cached HTML/JSON only -- zero metric logic on the request
    path (D-20). Cache-Control is set on the returned TemplateResponse.
    """
    scope = _normalize_betting_scope(scope)
    context = _build_betting_context(service, scope, request)
    template_response = templates.TemplateResponse(
        request,
        "pages/betting.html",
        context,
        block_name="betting_content",
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/season")
def season_fragment(
    request: Request,
    season: str | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Return the ``season_tracking_content`` block for the HTMX season swap.

    Renders only the swappable portion of the season page (KPI strip + the
    cumulative and weekly chart sections) for the selected *season*, used when
    the season selector dropdown changes (D-11).

    ``season`` is whitelisted to ``service.get_prediction_seasons()`` (an
    out-of-range value falls back to the latest, D-01) via the same chokepoint
    the full page uses, and the context is built by the shared
    ``_build_season_context`` helper so the cached-read logic is not duplicated.
    Reads cached HTML/JSON only -- zero metric logic on the request path (D-12).
    Cache-Control is set on the returned TemplateResponse.

    ``season`` is accepted as a raw string (matching the sibling ``/fragments/games``
    and ``/fragments/performance`` routes) and parsed defensively so an unparseable
    value degrades to the dynamic latest-season default rather than raising a 422
    (WR-02); the T-V5-01 whitelist in ``_normalize_season`` still gates the cache id.
    """
    available = service.get_prediction_seasons()
    # The SHARED parser (33.2 review C2 WR-03): the old ``lstrip("-").isdigit()`` guard let
    # ``"--5"`` through to ``int("--5")``, a 500, exactly as pages.py WR-07 documented.
    season_int = _parse_int_param(season)
    season_resolved = _normalize_season(season_int, available)
    context = _build_season_context(service, season_resolved, request)
    template_response = templates.TemplateResponse(
        request,
        "pages/season.html",
        context,
        block_name="season_tracking_content",
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response
