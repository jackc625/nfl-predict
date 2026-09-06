"""CSV and JSON export endpoints for the NFL Prediction System.

Provides downloadable exports of prediction data in CSV and JSON formats.
Supports filtering by season, week, and game_id, plus a ``type=backtest``
option for exporting backtest predictions and a ``type=bets`` option for
exporting the Phase-31 weekly bet list (D31-32).

THE BETS EXPORT READS THE SAME ROWS THE PAGE READS. It calls the same two
``DataService`` getters ``/bets`` calls, in the same order, so an exported row
and a screenshot of the page cannot disagree -- a second query shaped "like" the
page's is the two-lists failure that makes an export a separate claim instead of
the same one. It returns BOTH halves of the candidate universe, live and
suppressed, and every row carries ``status``, ``rejection_reason``,
``provenance`` and ``validation_type``, which is what stops a suppressed
candidate being read as a bet that was placed.

It adds NO route: both branches live inside the two shipped handlers, following
their existing ``type``-parameter dispatch idiom.

Routes:
    GET /api/export/csv   -- Download predictions as CSV
    GET /api/export/json  -- Download predictions as JSON
"""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse

from api.dependencies import get_data_service
from api.routes.pages import PAGE_CACHE_CONTROL
from api.services import DataService

router = APIRouter(prefix="/api/export", tags=["export"])

# The ``type`` parameter value selecting the Phase-31 bet-list export (D31-32).
EXPORT_TYPE_BETS = "bets"


def _bets_rows(
    service: DataService, season: int | None, week: int | None
) -> list[dict]:
    """The week's FULL record: the ranked live rows, then the suppressed ones.

    Both halves come from the getters ``/bets`` itself calls -- ``get_bet_list`` (per-bet EV
    descending, ties broken season/week/game_id/target) and ``get_suppressed_bets`` (grouped by
    reason under the same four-key tie-break). Concatenating them in that order reproduces the
    page's reading order exactly, so an export can be compared against a screenshot position by
    position.

    Nothing is recomputed and nothing is re-ordered here. The two getters are the exact
    complement of each other over one ``status`` constant, so their union is every row the cache
    holds for the week -- a declined candidate cannot be dropped from the export any more than it
    can be dropped from the page.
    """
    return [
        *service.get_bet_list(season, week),
        *service.get_suppressed_bets(season, week),
    ]


def _bets_filename(season: int | None, week: int | None, suffix: str) -> str:
    """The download filename, naming the season and the week whenever both are known."""
    if season and week:
        return f"bets_{season}_week{week}.{suffix}"
    if season:
        return f"bets_{season}_full_season.{suffix}"
    return f"nfl_bets.{suffix}"


@router.get("/csv")
def export_csv(
    season: int | None = Query(None),
    week: int | None = Query(None),
    game_id: str | None = Query(None),
    type: str | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Export prediction data as a downloadable CSV file.

    Query parameters:
        season: Filter by NFL season year.
        week: Filter by NFL week number.
        game_id: Export a single game's detail.
        type: Set to ``backtest`` to export backtest predictions, or to ``bets``
            to export the week's full bet-list record (live rows followed by
            suppressed rows, in the page's own order).

    Returns:
        StreamingResponse with ``text/csv`` content type and
        ``Content-Disposition: attachment`` header.
    """
    if type == EXPORT_TYPE_BETS:
        rows = _bets_rows(service, season, week)
        filename = _bets_filename(season, week, "csv")
    elif type == "backtest":
        rows = service.get_backtest_predictions(season=season)
        filename = f"nfl_backtest{'_' + str(season) if season else ''}.csv"
    elif game_id:
        detail = service.get_game_detail(game_id)
        rows = [_flatten_game_detail(detail)] if detail else []
        filename = f"nfl_game_{game_id}.csv"
    else:
        rows = service.export_predictions(season=season, week=week)
        if season and week:
            filename = f"predictions_{season}_week{week}.csv"
        elif season:
            filename = f"predictions_{season}_full_season.csv"
        else:
            filename = "nfl_predictions.csv"

    if not rows:
        return JSONResponse({"error": "No data found"}, status_code=404)

    buffer = io.StringIO()
    fieldnames = list(rows[0].keys())
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    buffer.seek(0)

    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": PAGE_CACHE_CONTROL,
        },
    )


@router.get("/json")
def export_json(
    season: int | None = Query(None),
    week: int | None = Query(None),
    game_id: str | None = Query(None),
    type: str | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Export prediction data as a downloadable JSON file.

    Query parameters are identical to the CSV endpoint.

    Returns:
        StreamingResponse with ``application/json`` content type and
        ``Content-Disposition: attachment`` header.
    """
    if type == EXPORT_TYPE_BETS:
        rows = _bets_rows(service, season, week)
    elif type == "backtest":
        rows = service.get_backtest_predictions(season=season)
    elif game_id:
        detail = service.get_game_detail(game_id)
        rows = [_flatten_game_detail(detail)] if detail else []
    else:
        rows = service.export_predictions(season=season, week=week)

    if not rows:
        return JSONResponse({"error": "No data found"}, status_code=404)

    if type == EXPORT_TYPE_BETS:
        filename = _bets_filename(season, week, "json")
    elif type == "backtest":
        filename = f"nfl_backtest{'_' + str(season) if season else ''}.json"
    elif game_id:
        filename = f"nfl_game_{game_id}.json"
    elif season and week:
        filename = f"predictions_{season}_week{week}.json"
    elif season:
        filename = f"predictions_{season}_full_season.json"
    else:
        filename = "nfl_export.json"

    content = json.dumps(rows, default=str, indent=2)
    return StreamingResponse(
        iter([content]),
        media_type="application/json",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "Cache-Control": PAGE_CACHE_CONTROL,
        },
    )


def _flatten_game_detail(detail: dict | None) -> dict:
    """Flatten nested game detail dict for CSV/JSON export.

    The game detail dict may contain nested ``context`` and
    ``feature_importances`` dicts which need to be flattened or
    serialised for tabular export.
    """
    if detail is None:
        return {}
    skip = ("context", "feature_importances")
    flat = {k: v for k, v in detail.items() if k not in skip}
    ctx = detail.get("context")
    if ctx and isinstance(ctx, dict):
        for ck, cv in ctx.items():
            if isinstance(cv, (list, dict)):
                flat[f"context_{ck}"] = json.dumps(cv, default=str)
            else:
                flat[f"context_{ck}"] = cv
    return flat
