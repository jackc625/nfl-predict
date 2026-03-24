"""Data access layer for the NFL Prediction web API.

Provides DataService, which queries the DuckDB web cache in read-only mode.
Each method opens a fresh connection, queries, and closes -- DuckDB read-only
connections are lightweight and this avoids holding locks.

No model classes are imported here (UIAP-01 compliance).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from utils import get_logger

logger = get_logger(__name__)


class DataService:
    """Read-only data service backed by the DuckDB web cache."""

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def _connect(self) -> duckdb.DuckDBPyConnection:
        """Open a read-only DuckDB connection to the cache."""
        return duckdb.connect(str(self._db_path), read_only=True)

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def get_predictions(
        self,
        season: int | None = None,
        week: int | None = None,
        sort: str = "time",
    ) -> list[dict[str, Any]]:
        """Fetch predictions with optional season/week filters.

        Args:
            season: Filter by NFL season year.
            week: Filter by NFL week number.
            sort: Sort order -- "time" (game_date), "confidence" (wp_edge desc),
                  or "edge" (ats_edge desc).

        Returns:
            List of prediction dicts.
        """
        query = "SELECT * FROM predictions"
        conditions: list[str] = []
        params: list[Any] = []

        if season is not None:
            conditions.append("season = ?")
            params.append(season)
        if week is not None:
            conditions.append("week = ?")
            params.append(week)

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        order_map = {
            "time": "game_date ASC",
            "confidence": "wp_edge DESC NULLS LAST",
            "edge": "ats_edge DESC NULLS LAST",
        }
        query += " ORDER BY " + order_map.get(sort, "game_date ASC")

        with self._connect() as conn:
            result = conn.execute(query, params)
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_game_detail(self, game_id: str) -> dict[str, Any] | None:
        """Fetch full detail for a single game including context and features.

        Returns None if the game is not found.
        """
        with self._connect() as conn:
            # Prediction data
            result = conn.execute(
                "SELECT * FROM predictions WHERE game_id = ?", [game_id]
            )
            columns = [desc[0] for desc in result.description]
            row = result.fetchone()
            if row is None:
                return None

            game = dict(zip(columns, row))

            # Game context
            ctx_result = conn.execute(
                "SELECT * FROM game_context WHERE game_id = ?", [game_id]
            )
            ctx_cols = [desc[0] for desc in ctx_result.description]
            ctx_row = ctx_result.fetchone()
            if ctx_row is not None:
                game["context"] = dict(zip(ctx_cols, ctx_row))

            # Feature importances for this game (or model-level)
            fi_result = conn.execute(
                "SELECT target, feature_name, importance "
                "FROM feature_importances "
                "WHERE game_id = ? OR game_id = '_model_' "
                "ORDER BY importance DESC",
                [game_id],
            )
            fi_rows = fi_result.fetchall()
            importances: dict[str, list[dict[str, Any]]] = {}
            for target, feature_name, importance in fi_rows:
                importances.setdefault(target, []).append({
                    "feature_name": feature_name,
                    "importance": importance,
                })
            game["feature_importances"] = importances

            return game

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def get_backtest_metrics(
        self, season: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch backtest metrics, optionally filtered by season."""
        query = "SELECT * FROM backtest_metrics"
        params: list[Any] = []

        if season is not None:
            query += " WHERE season = ?"
            params.append(season)

        query += " ORDER BY season, target, metric_name"

        with self._connect() as conn:
            result = conn.execute(query, params)
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_backtest_predictions(
        self, season: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch backtest predictions, optionally filtered by season."""
        query = "SELECT * FROM backtest_predictions"
        params: list[Any] = []

        if season is not None:
            query += " WHERE season = ?"
            params.append(season)

        query += " ORDER BY season, week, game_id"

        with self._connect() as conn:
            result = conn.execute(query, params)
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_chart_html(self, chart_id: str) -> str | None:
        """Fetch a pre-rendered chart HTML div from the cache."""
        with self._connect() as conn:
            result = conn.execute(
                "SELECT html_div FROM chart_cache WHERE chart_id = ?",
                [chart_id],
            )
            row = result.fetchone()
            return row[0] if row else None

    def get_simulation_results(self) -> list[dict[str, Any]]:
        """Fetch all betting simulation results."""
        with self._connect() as conn:
            result = conn.execute(
                "SELECT * FROM simulation_results ORDER BY strategy, metric_name"
            )
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_equity_curve(self) -> list[dict[str, Any]]:
        """Fetch equity curve data for all strategies."""
        with self._connect() as conn:
            result = conn.execute(
                "SELECT * FROM equity_curve ORDER BY strategy, bet_index"
            )
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def get_available_weeks(
        self, season: int | None = None
    ) -> list[dict[str, Any]]:
        """Return distinct season/week combinations from predictions."""
        query = "SELECT DISTINCT season, week FROM predictions"
        params: list[Any] = []

        if season is not None:
            query += " WHERE season = ?"
            params.append(season)

        query += " ORDER BY season DESC, week DESC"

        with self._connect() as conn:
            result = conn.execute(query, params)
            columns = [desc[0] for desc in result.description]
            return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_available_seasons(self) -> list[int]:
        """Return distinct seasons from predictions."""
        with self._connect() as conn:
            result = conn.execute(
                "SELECT DISTINCT season FROM predictions ORDER BY season DESC"
            )
            return [row[0] for row in result.fetchall()]

    def get_cache_meta(self) -> dict[str, Any]:
        """Fetch all cache metadata as a dict."""
        with self._connect() as conn:
            result = conn.execute("SELECT key, value FROM cache_meta")
            return {row[0]: row[1] for row in result.fetchall()}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_predictions(
        self,
        season: int | None = None,
        week: int | None = None,
    ) -> list[dict[str, Any]]:
        """Export full prediction data for download/API consumption."""
        return self.get_predictions(season=season, week=week, sort="time")
