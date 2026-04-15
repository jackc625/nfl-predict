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

from fastapi import APIRouter, Depends, Query, Request

from api.charts import INSIGHTS_CHART_IDS
from api.dependencies import get_data_service, templates
from api.services import DataService

router = APIRouter(tags=["pages"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pivot_season_metrics(raw_metrics: list[dict]) -> list[dict]:
    """Pivot long-format (season, target, metric_name, metric_value) rows
    into one row per (season, target) with named metric columns.

    Returns list of dicts with keys: season, target, games, accuracy, mae, rmse, r2.
    """
    grouped: dict[tuple[int, str], dict[str, Any]] = {}

    for m in raw_metrics:
        season = m.get("season", 0)
        target = m.get("target", "")
        metric_name = m.get("metric_name", "")
        metric_value = m.get("metric_value")

        if season == 0:
            continue

        key = (season, target)
        if key not in grouped:
            grouped[key] = {
                "season": season,
                "target": target,
                "games": None,
                "accuracy": None,
                "mae": None,
                "rmse": None,
                "r2": None,
            }

        if metric_name in ("n_games", "n_predictions"):
            grouped[key]["games"] = int(metric_value) if metric_value else None
        elif metric_name == "accuracy":
            grouped[key]["accuracy"] = (
                float(metric_value) * 100 if metric_value else None
            )
        elif metric_name == "mae":
            grouped[key]["mae"] = float(metric_value) if metric_value else None
        elif metric_name == "rmse":
            grouped[key]["rmse"] = float(metric_value) if metric_value else None
        elif metric_name == "r2":
            grouped[key]["r2"] = float(metric_value) if metric_value else None

    rows = sorted(grouped.values(), key=lambda r: (r["season"], r["target"]))
    return rows


def _compute_week_summary(games: list[dict]) -> dict[str, Any]:
    """Compute per-target accuracy for completed games in a week.

    For each target (WP, ATS, O/U), counts correct predictions and
    returns counts plus percentages.

    Args:
        games: List of game prediction dicts.

    Returns:
        Dict with total_games, wp_correct/wp_total/wp_pct,
        ats_correct/ats_total/ats_pct, ou_correct/ou_total/ou_pct.
        Empty dict if no completed games.
    """
    completed = [g for g in games if g.get("status") == "completed"]
    if not completed:
        return {}

    # WP correct: predicted home win (wp_prob > 0.5) matches actual home win
    wp_total = len([g for g in completed if g.get("wp_prob") is not None])
    wp_correct = sum(
        1
        for g in completed
        if g.get("wp_prob") is not None
        and g.get("home_score") is not None
        and g.get("away_score") is not None
        and ((g["wp_prob"] > 0.5) == (g["home_score"] > g["away_score"]))
    )

    # ATS correct: model spread prediction vs actual margin
    ats_total = len(
        [
            g
            for g in completed
            if g.get("ats_prediction") is not None
            and g.get("market_spread") is not None
        ]
    )
    ats_correct = sum(
        1
        for g in completed
        if g.get("ats_prediction") is not None
        and g.get("market_spread") is not None
        and g.get("home_score") is not None
        and g.get("away_score") is not None
        and (
            (g["home_score"] - g["away_score"] > -g["market_spread"])
            == (g["ats_prediction"] > -g["market_spread"])
        )
    )

    # O/U correct: model total prediction vs actual total
    ou_total = len(
        [
            g
            for g in completed
            if g.get("ou_prediction") is not None and g.get("market_total") is not None
        ]
    )
    ou_correct = sum(
        1
        for g in completed
        if g.get("ou_prediction") is not None
        and g.get("market_total") is not None
        and g.get("home_score") is not None
        and g.get("away_score") is not None
        and (
            (g["home_score"] + g["away_score"] > g["market_total"])
            == (g["ou_prediction"] > g["market_total"])
        )
    )

    return {
        "total_games": len(completed),
        "wp_correct": wp_correct,
        "wp_total": wp_total,
        "wp_pct": round(wp_correct / wp_total * 100) if wp_total else 0,
        "ats_correct": ats_correct,
        "ats_total": ats_total,
        "ats_pct": round(ats_correct / ats_total * 100) if ats_total else 0,
        "ou_correct": ou_correct,
        "ou_total": ou_total,
        "ou_pct": round(ou_correct / ou_total * 100) if ou_total else 0,
    }


PAGE_CACHE_CONTROL = "public, max-age=60"


def _annotate_wp_correct(games: list[dict]) -> list[dict]:
    """Return a new list of games with wp_correct annotated.

    Builds fresh shallow copies of each dict so the source list (which may
    come from the DataService TTLCache in plan 15-02) is never mutated.
    A shallow copy is sufficient because wp_correct is a scalar -- no
    nested structures are touched.

    wp_correct is True if the WP prediction was correct, False if
    incorrect, or None if the game is not completed or data is missing.
    """
    annotated: list[dict] = []
    for game in games:
        new_game = dict(game)
        new_game["wp_correct"] = None
        if (
            new_game.get("status") == "completed"
            and new_game.get("wp_prob") is not None
            and new_game.get("home_score") is not None
            and new_game.get("away_score") is not None
        ):
            home_won = new_game["home_score"] > new_game["away_score"]
            predicted_home = new_game["wp_prob"] > 0.5
            new_game["wp_correct"] = home_won == predicted_home
        annotated.append(new_game)
    return annotated


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
        p
        for p in predictions
        if p.get("target") == "wp"
        and p.get("probability_clv") is not None
        and p.get("has_closing_odds") is not False
    ]

    if wp_preds:
        wp_clv_values = [float(p["probability_clv"]) for p in wp_preds]
        summary["overall_clv"] = sum(wp_clv_values) / len(wp_clv_values) * 100

    if not summary["total_games"]:
        # Count from predictions if not in aggregate metrics
        summary["total_games"] = len(
            {p["game_id"] for p in predictions if p.get("game_id")}
        )

    if wp_accuracy_values:
        summary["wp_accuracy"] = sum(wp_accuracy_values) / len(wp_accuracy_values) * 100
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
    service: DataService = Depends(get_data_service),
):
    """Serve the predictions dashboard (landing page).

    Fetches predictions for the selected (or latest) week and renders
    the full page. If the request comes from HTMX, returns only the
    game_grid block.

    Defaults to the most recent season/week with prediction data (D-01, D-02).
    Computes wp_correct for correct/incorrect indicators (D-03) and
    week_summary for per-target accuracy banner (D-04).
    """
    available_seasons = service.get_prediction_seasons()
    cache_meta = service.get_cache_meta()

    # Default to latest season with predictions (D-01)
    if season is None and available_seasons:
        season = available_seasons[0]

    available_weeks = service.get_available_weeks(season=season)

    # Default to latest week (not Week 1) per D-02
    if week is None and available_weeks:
        week = available_weeks[0]["week"]
        season = available_weeks[0]["season"]

    games = service.get_predictions(season=season, week=week, sort=sort)

    # Compute wp_correct on a fresh list so the DataService TTLCache source
    # is never mutated (plan 15-02 review item #4).
    games = _annotate_wp_correct(games)

    context = {
        "request": request,
        "games": games,
        "available_weeks": available_weeks,
        "available_seasons": available_seasons,
        "current_week": week,
        "current_season": season,
        "current_sort": sort,
        "current_path": "/",
        "cache_meta": cache_meta,
        "week_summary": _compute_week_summary(games),
    }

    block_name = "game_grid" if request.headers.get("HX-Request") else None
    template_response = templates.TemplateResponse(
        request, "pages/this_week.html", context, block_name=block_name
    )
    # Cache-Control is set directly on the returned TemplateResponse because
    # headers set on an injected ``response: Response`` parameter are NOT
    # propagated when the handler returns its own response object (a known
    # FastAPI/Starlette behaviour).
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/performance")
def performance_page(
    request: Request,
    season: int | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Serve the historical performance page.

    Shows all-time summary metrics with a season selector that swaps
    season-specific metrics via HTMX.
    """
    available_seasons = service.get_available_seasons()
    raw_metrics = service.get_backtest_metrics(season=season)
    season_metrics = _pivot_season_metrics(raw_metrics)
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
    template_response = templates.TemplateResponse(
        request, "pages/performance.html", context, block_name=block
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/backtest")
def backtest_page(
    request: Request,
    service: DataService = Depends(get_data_service),
):
    """Serve the backtest results page with 4 Plotly charts.

    Charts are pre-rendered in the DuckDB chart_cache for fast serving.
    Falls back to empty state components if charts are not available.
    """
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
    template_response = templates.TemplateResponse(
        request, "pages/backtest.html", context
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/insights")
def insights_page(
    request: Request,
    service: DataService = Depends(get_data_service),
):
    """Serve the Model Insights page.

    All charts are pre-rendered during cache population (Plan 16-02) and
    stored in the DuckDB chart_cache. The aggregate Model-vs-Market table
    is also precomputed during pre-render and stored under chart_id
    ``insights_aggregate_table``. This handler contains NO statistical
    logic -- Plan 16-02 is the single source of truth for metric formulas
    (REVIEWS Codex HIGH #1).

    Page is static with no filters (D-19). Cache-Control header matches
    the other static-artifact pages (Phase 15 D-07 / REVIEWS Codex MEDIUM
    #12).
    """
    # Fetch the 9 new insights chart HTML blobs.
    charts: dict[str, str | None] = {
        chart_id: service.get_chart_html(chart_id) for chart_id in INSIGHTS_CHART_IDS
    }
    # Reuse the existing WP reliability chart (not part of INSIGHTS_CHART_IDS per D-22).
    charts["calibration"] = service.get_chart_html("calibration")

    # Precomputed aggregate table (list of dicts). Falls back to [] if missing/malformed.
    aggregate_table = service.get_insights_aggregate_table()

    cache_meta = service.get_cache_meta()
    context = {
        "request": request,
        "charts": charts,
        "aggregate_table": aggregate_table,
        "current_path": "/insights",
        "cache_meta": cache_meta,
    }
    template_response = templates.TemplateResponse(
        request, "pages/insights.html", context
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/games/{game_id}")
def game_detail_page(
    request: Request,
    game_id: str,
    service: DataService = Depends(get_data_service),
):
    """Serve the game detail drill-down page.

    Shows full prediction breakdown including feature importance bars,
    prediction vs market comparison, team context (Elo, form, H2H),
    venue/weather, and result overlay for completed games (D-13 to D-15).
    """
    game = service.get_game_detail(game_id)
    cache_meta = service.get_cache_meta()
    context = {
        "request": request,
        "game": game,
        "current_path": "",
        "cache_meta": cache_meta,
    }
    template_response = templates.TemplateResponse(
        request, "pages/game_detail.html", context
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response
