"""Data access layer for the NFL Prediction web API.

Provides DataService, which queries the DuckDB web cache in read-only mode.
Each method opens a fresh connection, queries, and closes -- DuckDB read-only
connections are lightweight and this avoids holding locks.

No model classes are imported here (UIAP-01 compliance).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

from utils import get_logger

logger = get_logger(__name__)


def _parse_json_or_default(value: Any, default: Any) -> Any:
    """Parse a JSON string, returning *default* if parsing fails or value is None."""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


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

        Returns a dict with:
        - All prediction fields from the predictions table
        - ``context``: nested dict from game_context (with parsed lists)
        - ``feature_importances``: dict keyed by target, values are
          ``{feature_name: importance}`` dicts (ready for template tojson)
        - ``wp_correct``: bool indicating whether the WP prediction was right
        - ``wp_clv``: closing line value for the WP prediction (if available)

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

            # --- Derived fields for completed games (D-15) ---
            game["wp_correct"] = None
            game["wp_clv"] = None

            home_score = game.get("home_score")
            away_score = game.get("away_score")
            wp_prob = game.get("wp_prob")

            if (
                game.get("status") == "completed"
                and home_score is not None
                and away_score is not None
                and wp_prob is not None
            ):
                home_won = home_score > away_score
                predicted_home = wp_prob > 0.5
                game["wp_correct"] = home_won == predicted_home

            # CLV from backtest predictions (if available)
            clv_result = conn.execute(
                "SELECT probability_clv FROM backtest_predictions "
                "WHERE game_id = ? AND target = 'wp'",
                [game_id],
            )
            clv_row = clv_result.fetchone()
            if clv_row is not None and clv_row[0] is not None:
                game["wp_clv"] = float(clv_row[0]) * 100  # as percentage

            # --- Game context ---
            ctx_result = conn.execute(
                "SELECT * FROM game_context WHERE game_id = ?", [game_id]
            )
            ctx_cols = [desc[0] for desc in ctx_result.description]
            ctx_row = ctx_result.fetchone()
            if ctx_row is not None:
                context = dict(zip(ctx_cols, ctx_row))

                # Parse JSON strings into Python lists for template rendering
                context["home_last5_list"] = _parse_json_or_default(
                    context.get("home_last5"), []
                )
                context["away_last5_list"] = _parse_json_or_default(
                    context.get("away_last5"), []
                )

                # Parse H2H record JSON
                h2h = _parse_json_or_default(context.get("h2h_record"), {})
                context["h2h_home_wins"] = h2h.get("home_wins", 0)
                context["h2h_away_wins"] = h2h.get("away_wins", 0)

                game["context"] = context
            else:
                game["context"] = None

            # --- Feature importances ---
            # Return as {target: {feature_name: importance}} for tojson in template
            fi_result = conn.execute(
                "SELECT target, feature_name, importance "
                "FROM feature_importances "
                "WHERE game_id = ? OR game_id = '_model_' "
                "ORDER BY importance DESC",
                [game_id],
            )
            fi_rows = fi_result.fetchall()
            importances: dict[str, dict[str, float]] = {}
            for target, feature_name, importance in fi_rows:
                importances.setdefault(target, {})[feature_name] = float(
                    importance
                )
            game["feature_importances"] = importances if importances else None

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
        """Return distinct seasons from backtest data."""
        with self._connect() as conn:
            result = conn.execute(
                "SELECT DISTINCT season FROM backtest_metrics "
                "WHERE season > 0 ORDER BY season DESC"
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
