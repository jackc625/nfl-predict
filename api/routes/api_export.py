"""CSV and JSON export endpoints for the NFL Prediction System.

Provides downloadable exports of prediction data in CSV and JSON formats.
Supports filtering by season, week, and game_id, plus a ``type=backtest``
option for exporting backtest predictions.

Routes:
    GET /api/export/csv   -- Download predictions as CSV
    GET /api/export/json  -- Download predictions as JSON
"""

from __future__ import annotations

import csv
import io
import json

from fastapi import APIRouter, Query
from fastapi.responses import JSONResponse, StreamingResponse

from api.dependencies import get_data_service

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/csv")
def export_csv(
    season: int | None = Query(None),
    week: int | None = Query(None),
    game_id: str | None = Query(None),
    type: str | None = Query(None),
):
    """Export prediction data as a downloadable CSV file.

    Query parameters:
        season: Filter by NFL season year.
        week: Filter by NFL week number.
        game_id: Export a single game's detail.
        type: Set to ``backtest`` to export backtest predictions.

    Returns:
        StreamingResponse with ``text/csv`` content type and
        ``Content-Disposition: attachment`` header.
    """
    service = get_data_service()

    if type == "backtest":
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
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/json")
def export_json(
    season: int | None = Query(None),
    week: int | None = Query(None),
    game_id: str | None = Query(None),
    type: str | None = Query(None),
):
    """Export prediction data as a downloadable JSON file.

    Query parameters are identical to the CSV endpoint.

    Returns:
        StreamingResponse with ``application/json`` content type and
        ``Content-Disposition: attachment`` header.
    """
    service = get_data_service()

    if type == "backtest":
        rows = service.get_backtest_predictions(season=season)
    elif game_id:
        detail = service.get_game_detail(game_id)
        rows = [_flatten_game_detail(detail)] if detail else []
    else:
        rows = service.export_predictions(season=season, week=week)

    if not rows:
        return JSONResponse({"error": "No data found"}, status_code=404)

    if type == "backtest":
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
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
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
