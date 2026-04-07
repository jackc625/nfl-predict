"""Data access layer for the NFL Prediction web API.

Provides :class:`DataService`, which queries the DuckDB web cache in read-only
mode using a connection injected via FastAPI dependency injection
(:func:`api.dependencies.get_db`). DataService NEVER opens or reconnects DuckDB
connections itself -- reconnect ownership lives in :func:`api.dependencies.get_db`.

If a query raises ``duckdb.Error`` due to a dead connection, the exception
propagates and the next request will receive a fresh connection from
``get_db()``.

No model classes are imported here (UIAP-01 compliance).

Caching (plan 15-02)
--------------------
Hot-path methods are backed by a module-level :class:`cachetools.TTLCache`
guarded by a :class:`threading.RLock`. Every cache read and write goes through
``copy.deepcopy`` so route-layer mutation of returned values cannot poison the
cached copy. The TTL is 5 minutes (predictions change at most weekly).

Concurrency envelope: see the deployment-assumption docstring in
``api.main.lifespan`` -- this app runs under single-worker uvicorn, so the
module-level cache plus the RLock is sufficient. If N>1 workers are ever
introduced, each worker will hold its own copy of the cache; cross-worker
invalidation would require out-of-process state.

``clear_cache()`` is a module-level function intended to be called from:
- ``api.main.lifespan`` startup, after ``app.state.db_conn`` is established
- Pipeline scripts, after the Friday snapshot refreshes the cache
- Test fixtures, between tests for isolation
"""

from __future__ import annotations

import copy
import json
import threading
from typing import Any

import duckdb
from cachetools import TTLCache

from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Module-level TTL cache (plan 15-02)
# ---------------------------------------------------------------------------

# maxsize=128 is ample for this workload (a handful of season/week combos).
# ttl=300 (5 minutes per D-06) because predictions update at most weekly.
# _cache_lock guards all reads and writes; see plan 15-01 for the single-worker
# uvicorn assumption that bounds the concurrency envelope.
_cache: TTLCache = TTLCache(maxsize=128, ttl=300)
_cache_lock = threading.RLock()


def clear_cache() -> None:
    """Clear the DataService TTL cache.

    Call this:
    - From the FastAPI lifespan startup hook AFTER app.state.db_conn is set
    - From pipeline scripts after Friday snapshot refresh
    - From test fixtures between tests for isolation
    """
    with _cache_lock:
        _cache.clear()


def _cache_get(key: tuple) -> Any | None:
    """Return a deep copy of the cached value at *key*, or None if missing.

    The deep copy prevents callers from mutating the cached copy via Python
    aliasing even if they ignore the ``_cache_set`` deep-copy on the way in.
    """
    with _cache_lock:
        value = _cache.get(key)
    if value is None:
        return None
    return copy.deepcopy(value)


def _cache_set(key: tuple, value: Any) -> None:
    """Store a deep copy of *value* under *key*.

    The deep copy decouples the cached payload from the caller's reference so
    the caller cannot mutate the cache by mutating its own return value.
    """
    with _cache_lock:
        _cache[key] = copy.deepcopy(value)


# ---------------------------------------------------------------------------
# Explicit column lists (plan 15-02 D-09)
# ---------------------------------------------------------------------------
# These match ``api.cache.CACHE_SCHEMA`` in column order. Keeping them here as
# module-level constants makes schema drift a compile-time (well, import-time)
# concern instead of a silent runtime one.

_PREDICTIONS_COLUMNS = (
    "game_id, season, week, game_date, home_team, away_team, "
    "status, home_score, away_score, "
    "wp_prob, wp_confidence, ats_prediction, ats_confidence, "
    "ou_prediction, ou_confidence, "
    "market_spread, market_total, market_ml_home, market_ml_away, "
    "wp_edge, ats_edge, ou_edge, "
    "blended_wp, blended_ats, blended_ou"
)

_GAME_CONTEXT_COLUMNS = (
    "game_id, home_elo, away_elo, home_last5, away_last5, h2h_record, "
    "venue_name, surface, roof_type, weather_severity, wind_mph, "
    "is_outdoor, is_divisional, is_primetime"
)

_BACKTEST_METRICS_COLUMNS = "season, target, metric_name, metric_value"

_BACKTEST_PREDICTIONS_COLUMNS = (
    "game_id, season, week, target, model_prob, actual, "
    "probability_clv, has_closing_odds"
)

_SIMULATION_RESULTS_COLUMNS = "strategy, metric_name, metric_value"

_EQUITY_CURVE_COLUMNS = "strategy, bet_index, bankroll"


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
    """Read-only data service backed by an injected DuckDB connection."""

    def __init__(self, conn: duckdb.DuckDBPyConnection) -> None:
        self._conn = conn

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
        key = ("predictions", season, week, sort)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_predictions_uncached(season, week, sort)
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_predictions_uncached(
        self,
        season: int | None,
        week: int | None,
        sort: str,
    ) -> list[dict[str, Any]]:
        query = f"SELECT {_PREDICTIONS_COLUMNS} FROM predictions"
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

        result = self._conn.execute(query, params)
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

        Returns None if the game is not found. All four queries below run
        through the same injected ``self._conn`` -- no per-query reconnect.

        NOT cached: unique per game_id, cache hit rate would be ~0.
        """
        # Prediction data
        result = self._conn.execute(
            f"SELECT {_PREDICTIONS_COLUMNS} FROM predictions WHERE game_id = ?",
            [game_id],
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
        clv_result = self._conn.execute(
            "SELECT probability_clv FROM backtest_predictions "
            "WHERE game_id = ? AND target = 'wp'",
            [game_id],
        )
        clv_row = clv_result.fetchone()
        if clv_row is not None and clv_row[0] is not None:
            game["wp_clv"] = float(clv_row[0]) * 100  # as percentage

        # --- Game context ---
        ctx_result = self._conn.execute(
            f"SELECT {_GAME_CONTEXT_COLUMNS} FROM game_context WHERE game_id = ?",
            [game_id],
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
        fi_result = self._conn.execute(
            "SELECT target, feature_name, importance "
            "FROM feature_importances "
            "WHERE game_id = ? OR game_id = '_model_' "
            "ORDER BY importance DESC",
            [game_id],
        )
        fi_rows = fi_result.fetchall()
        importances: dict[str, dict[str, float]] = {}
        for target, feature_name, importance in fi_rows:
            importances.setdefault(target, {})[feature_name] = float(importance)
        game["feature_importances"] = importances if importances else None

        return game

    # ------------------------------------------------------------------
    # Backtest
    # ------------------------------------------------------------------

    def get_backtest_metrics(self, season: int | None = None) -> list[dict[str, Any]]:
        """Fetch backtest metrics, optionally filtered by season."""
        key = ("backtest_metrics", season)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_backtest_metrics_uncached(season)
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_backtest_metrics_uncached(
        self, season: int | None
    ) -> list[dict[str, Any]]:
        query = f"SELECT {_BACKTEST_METRICS_COLUMNS} FROM backtest_metrics"
        params: list[Any] = []

        if season is not None:
            query += " WHERE season = ?"
            params.append(season)

        query += " ORDER BY season, target, metric_name"

        result = self._conn.execute(query, params)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_backtest_predictions(
        self, season: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch backtest predictions, optionally filtered by season."""
        key = ("backtest_predictions", season)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_backtest_predictions_uncached(season)
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_backtest_predictions_uncached(
        self, season: int | None
    ) -> list[dict[str, Any]]:
        query = f"SELECT {_BACKTEST_PREDICTIONS_COLUMNS} FROM backtest_predictions"
        params: list[Any] = []

        if season is not None:
            query += " WHERE season = ?"
            params.append(season)

        query += " ORDER BY season, week, game_id"

        result = self._conn.execute(query, params)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_chart_html(self, chart_id: str) -> str | None:
        """Fetch a pre-rendered chart HTML div from the cache.

        NOT cached at the DataService layer: pre-rendered HTML is already fast
        to fetch and rarely changes; route-level Cache-Control headers cover
        browser caching.
        """
        result = self._conn.execute(
            "SELECT html_div FROM chart_cache WHERE chart_id = ?",
            [chart_id],
        )
        row = result.fetchone()
        return row[0] if row else None

    def get_simulation_results(self) -> list[dict[str, Any]]:
        """Fetch all betting simulation results."""
        key = ("simulation_results",)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_simulation_results_uncached()
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_simulation_results_uncached(self) -> list[dict[str, Any]]:
        result = self._conn.execute(
            f"SELECT {_SIMULATION_RESULTS_COLUMNS} FROM simulation_results "
            "ORDER BY strategy, metric_name"
        )
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_equity_curve(self) -> list[dict[str, Any]]:
        """Fetch equity curve data for all strategies."""
        key = ("equity_curve",)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_equity_curve_uncached()
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_equity_curve_uncached(self) -> list[dict[str, Any]]:
        result = self._conn.execute(
            f"SELECT {_EQUITY_CURVE_COLUMNS} FROM equity_curve "
            "ORDER BY strategy, bet_index"
        )
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def get_available_weeks(self, season: int | None = None) -> list[dict[str, Any]]:
        """Return distinct season/week combinations from predictions."""
        key = ("available_weeks", season)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_available_weeks_uncached(season)
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_available_weeks_uncached(self, season: int | None) -> list[dict[str, Any]]:
        query = "SELECT DISTINCT season, week FROM predictions"
        params: list[Any] = []

        if season is not None:
            query += " WHERE season = ?"
            params.append(season)

        query += " ORDER BY season DESC, week DESC"

        result = self._conn.execute(query, params)
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_available_seasons(self) -> list[int]:
        """Return distinct seasons from backtest data."""
        key = ("available_seasons",)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_available_seasons_uncached()
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_available_seasons_uncached(self) -> list[int]:
        result = self._conn.execute(
            "SELECT DISTINCT season FROM backtest_metrics "
            "WHERE season > 0 ORDER BY season DESC"
        )
        return [row[0] for row in result.fetchall()]

    def get_prediction_seasons(self) -> list[int]:
        """Return distinct seasons from predictions table."""
        key = ("prediction_seasons",)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_prediction_seasons_uncached()
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_prediction_seasons_uncached(self) -> list[int]:
        result = self._conn.execute(
            "SELECT DISTINCT season FROM predictions ORDER BY season DESC"
        )
        return [row[0] for row in result.fetchall()]

    def get_cache_meta(self) -> dict[str, Any]:
        """Fetch all cache metadata as a dict."""
        key = ("cache_meta",)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_cache_meta_uncached()
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_cache_meta_uncached(self) -> dict[str, Any]:
        result = self._conn.execute("SELECT key, value FROM cache_meta")
        return {row[0]: row[1] for row in result.fetchall()}

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def export_predictions(
        self,
        season: int | None = None,
        week: int | None = None,
    ) -> list[dict[str, Any]]:
        """Export full prediction data for download/API consumption.

        NOT cached directly: delegates to ``get_predictions`` which is already
        cached. The double-cache would just double-deep-copy.
        """
        return self.get_predictions(season=season, week=week, sort="time")
