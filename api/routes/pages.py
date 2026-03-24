"""HTML page route handlers for the NFL Prediction System.

Serves full HTML pages using Jinja2Blocks templates. When an HX-Request
header is present, only the relevant block is returned (HTMX fragment).

Routes:
    GET /            -- This Week's predictions dashboard (landing page)
    GET /performance -- Historical performance view
    GET /backtest    -- Backtest analysis with Plotly charts
    GET /games/{id}  -- Game detail drill-down (feature importance, market comparison)
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Query, Request

from api.dependencies import get_data_service, templates

router = APIRouter(tags=["pages"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_summary(service: Any) -> dict[str, Any]:
    """Aggregate all-time summary metrics from backtest data.

    Computes: total_games, overall_clv (mean wp CLV), wp_accuracy, brier_score.

    Args:
        service: DataService instance.

    Returns:
        Dict with summary metric values.
    """
    all_metrics = service.get_backtest_metrics()

    summary: dict[str, Any] = {
        "total_games": 0,
        "overall_clv": 0.0,
        "wp_accuracy": 0.0,
        "brier_score": 0.0,
    }

    if not all_metrics:
        return summary

    # Gather WP-specific metrics across seasons
    wp_accuracy_values: list[float] = []
    wp_brier_values: list[float] = []
    wp_clv_values: list[float] = []

    for m in all_metrics:
        target = m.get("target", "")
        metric_name = m.get("metric_name", "")
        metric_value = m.get("metric_value", 0.0)
        season = m.get("season", 0)

        if season == 0:
            # Overall aggregate metrics
            if target == "overall" and metric_name == "total_games":
                summary["total_games"] = int(metric_value)
            continue

        if target == "wp":
            if metric_name == "accuracy":
                wp_accuracy_values.append(float(metric_value))
            elif metric_name in ("brier_score", "mae"):
                # Use MAE as proxy if brier_score not available
                wp_brier_values.append(float(metric_value))

    # Compute CLV from predictions
    predictions = service.get_backtest_predictions()
    wp_preds = [
        p for p in predictions
        if p.get("target") == "wp"
        and p.get("probability_clv") is not None
        and p.get("has_closing_odds") is not False
    ]

    if wp_preds:
        wp_clv_values = [float(p["probability_clv"]) for p in wp_preds]
        summary["overall_clv"] = sum(wp_clv_values) / len(wp_clv_values) * 100

    if not summary["total_games"]:
        # Count from predictions if not in aggregate metrics
        summary["total_games"] = len({
            p["game_id"] for p in predictions if p.get("game_id")
        })

    if wp_accuracy_values:
        summary["wp_accuracy"] = (
            sum(wp_accuracy_values) / len(wp_accuracy_values) * 100
        )
    if wp_brier_values:
        summary["brier_score"] = sum(wp_brier_values) / len(wp_brier_values)

    return summary


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


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
        request, "pages/this_week.html", context, block_name=block_name
    )


@router.get("/performance")
def performance_page(
    request: Request,
    season: int | None = Query(None),
):
    """Serve the historical performance page.

    Shows all-time summary metrics with a season selector that swaps
    season-specific metrics via HTMX.
    """
    service = get_data_service()
    available_seasons = service.get_available_seasons()
    season_metrics = service.get_backtest_metrics(season=season)
    summary = _compute_summary(service)
    cache_meta = service.get_cache_meta()

    context = {
        "request": request,
        "available_seasons": available_seasons,
        "current_season": season,
        "season_metrics": season_metrics,
        "summary": summary,
        "current_path": "/performance",
        "cache_meta": cache_meta,
    }

    # If HTMX request, return only the performance_content block
    block = "performance_content" if request.headers.get("HX-Request") else None
    return templates.TemplateResponse(
        request, "pages/performance.html", context, block_name=block
    )


@router.get("/backtest")
def backtest_page(request: Request):
    """Serve the backtest results page with 4 Plotly charts.

    Charts are pre-rendered in the DuckDB chart_cache for fast serving.
    Falls back to empty state components if charts are not available.
    """
    service = get_data_service()
    charts = {
        "calibration": service.get_chart_html("calibration"),
        "clv": service.get_chart_html("clv"),
        "heatmap": service.get_chart_html("heatmap"),
        "equity": service.get_chart_html("equity"),
    }
    cache_meta = service.get_cache_meta()

    context = {
        "request": request,
        "charts": charts,
        "current_path": "/backtest",
        "cache_meta": cache_meta,
    }
    return templates.TemplateResponse(request, "pages/backtest.html", context)


@router.get("/games/{game_id}")
def game_detail_page(request: Request, game_id: str):
    """Serve the game detail drill-down page.

    Shows full prediction breakdown including feature importance bars,
    prediction vs market comparison, team context (Elo, form, H2H),
    venue/weather, and result overlay for completed games (D-13 to D-15).
    """
    service = get_data_service()
    game = service.get_game_detail(game_id)
    cache_meta = service.get_cache_meta()
    context = {
        "request": request,
        "game": game,
        "current_path": "",
        "cache_meta": cache_meta,
    }
    return templates.TemplateResponse(
        request, "pages/game_detail.html", context
    )
