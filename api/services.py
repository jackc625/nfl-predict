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

from api.cache import (
    BET_LIST_COLUMNS,
    BET_STATUS_LIVE,
    BET_TRACKER_BLOCK_COLUMNS,
    bet_list_populated_at_key,
)
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
    "venue_name, surface, roof_type, weather_severity, weather_severity_band, "
    "wind_mph, is_outdoor, is_divisional, is_primetime"
)

_BACKTEST_METRICS_COLUMNS = "season, target, metric_name, metric_value"

_BACKTEST_PREDICTIONS_COLUMNS = (
    "game_id, season, week, target, model_prob, actual, "
    "probability_clv, has_closing_odds"
)

_SIMULATION_RESULTS_COLUMNS = "strategy, metric_name, metric_value"

_EQUITY_CURVE_COLUMNS = "strategy, bet_index, bankroll"

# The bet-list SELECT list is DERIVED from ``api.cache.BET_LIST_COLUMNS`` rather than re-typed, so
# the locked 28-column order has exactly one source and a schema change cannot drift the reader
# away from the writer. ``api.cache`` is a sibling API module (no ``backtest`` import is involved,
# so the UIAP-01 guards are unaffected).
_BET_LIST_COLUMNS_SQL = ", ".join(BET_LIST_COLUMNS)

# Same rule for the precomputed tracker blocks: the reader's column list is DERIVED from the
# writer's, so a figure added to the block cannot be silently dropped on the way out.
_BET_TRACKER_BLOCK_COLUMNS_SQL = ", ".join(BET_TRACKER_BLOCK_COLUMNS)


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
    # Connection accessor (Codex MEDIUM #11)
    # ------------------------------------------------------------------

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Return the injected read-only DuckDB connection.

        Public accessor used by pre-render hooks that need to run ad-hoc
        queries not covered by a cached ``get_*`` method. Do NOT use this on
        the request path — all request handlers use the typed getters so the
        TTL cache and schema-drift guards stay in effect.
        """
        return self._conn

    # ------------------------------------------------------------------
    # Insights aggregate table (Phase 16 — precomputed during pre-render)
    # ------------------------------------------------------------------

    def get_insights_aggregate_table(self) -> list[dict[str, Any]]:
        """Read the precomputed insights aggregate table from ``chart_cache``.

        The table is stored as a JSON blob under ``chart_id =
        "insights_aggregate_table"`` during pre-render (Codex HIGH #1). This
        accessor decodes the JSON into a list of dicts so the template can
        render it without re-running any statistics at request time.

        Returns an empty list if the entry is missing or malformed.
        """
        html = self.get_chart_html("insights_aggregate_table")
        if not html:
            return []
        try:
            decoded = json.loads(html)
        except (json.JSONDecodeError, TypeError):
            return []
        return decoded if isinstance(decoded, list) else []

    # ------------------------------------------------------------------
    # Betting KPI strip + ROI table (Phase 17 — precomputed during pre-render)
    # ------------------------------------------------------------------

    def get_betting_kpis(self, scope: str) -> dict[str, Any]:
        """Read the precomputed betting KPI strip for *scope* from ``chart_cache``.

        The KPI dict is stored as a JSON blob under ``chart_id =
        f"betting_kpis_{scope}"`` during pre-render (D-20), one blob per scope
        (``all`` / ``recommended``). This accessor decodes the JSON into a dict
        so the template renders the 7-card scoreboard without re-running any
        statistics at request time (zero metric logic on the request path).

        ``scope`` whitelisting happens in the Plan 04 handlers; this accessor
        simply reads the cached id. Returns an empty dict if the entry is
        missing or malformed.
        """
        html = self.get_chart_html(f"betting_kpis_{scope}")
        if not html:
            return {}
        try:
            decoded = json.loads(html)
        except (json.JSONDecodeError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

    def get_betting_roi_table(self, scope: str) -> list[dict[str, Any]]:
        """Read the precomputed betting ROI table for *scope* from ``chart_cache``.

        The per-slice ROI rows are stored as a JSON blob under ``chart_id =
        f"betting_roi_table_{scope}"`` during pre-render (D-20), one blob per
        scope. This accessor decodes the JSON into a list of dicts so the
        template renders the ROI summary table without re-running any statistics
        at request time.

        Returns an empty list if the entry is missing or malformed.
        """
        html = self.get_chart_html(f"betting_roi_table_{scope}")
        if not html:
            return []
        try:
            decoded = json.loads(html)
        except (json.JSONDecodeError, TypeError):
            return []
        return decoded if isinstance(decoded, list) else []

    # ------------------------------------------------------------------
    # Season KPI strip (Phase 18 — precomputed during pre-render)
    # ------------------------------------------------------------------

    def get_season_kpis(self, season: int) -> dict[str, Any]:
        """Read the precomputed per-season KPI strip from ``chart_cache``.

        The KPI dict is stored as a JSON blob under ``chart_id =
        f"season_kpis_{season}"`` during pre-render (D-12), one blob per season
        present in ``predictions``. This accessor decodes the JSON into a dict so
        the template renders the season-to-date scoreboard without re-running any
        statistics at request time (zero metric logic on the request path, D-12).

        ``season`` whitelisting happens in the Plan 03 handlers; this accessor
        simply reads the cached id. Returns an empty dict if the entry is missing
        or malformed.
        """
        html = self.get_chart_html(f"season_kpis_{season}")
        if not html:
            return {}
        try:
            decoded = json.loads(html)
        except (json.JSONDecodeError, TypeError):
            return {}
        return decoded if isinstance(decoded, dict) else {}

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

    # ------------------------------------------------------------------
    # Bet list (Phase 31 -- PROD-02, D31-20/26/28)
    # ------------------------------------------------------------------

    def get_bet_list(
        self, season: int | None, week: int | None
    ) -> list[dict[str, Any]]:
        """Return the LIVE bet-list rows for *season* / *week*, ranked per D31-28.

        Order is per-bet EV DESCENDING, ties broken by ``(season, week, game_id, target)`` -- the
        SPEC R5 tie-break, so two requests render byte-identical order. ZERO computation happens
        here or in the route: every number was precomputed by the selector and written by
        ``api.cache.materialize_bet_list`` (UIAP-01).
        """
        key = ("bet_list", season, week)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_bet_list_uncached(season, week)
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_bet_list_uncached(
        self, season: int | None, week: int | None
    ) -> list[dict[str, Any]]:
        query = f"SELECT {_BET_LIST_COLUMNS_SQL} FROM bet_list WHERE status = ?"
        params: list[Any] = [BET_STATUS_LIVE]
        if season is not None:
            query += " AND season = ?"
            params.append(season)
        if week is not None:
            query += " AND week = ?"
            params.append(week)
        query += " ORDER BY per_bet_ev DESC NULLS LAST, season, week, game_id, target"

        try:
            result = self._conn.execute(query, params)
        except duckdb.Error:
            # The cache predates Phase 31 (no bet_list table). The route renders the
            # cache-absent empty state rather than 500ing.
            logger.warning("bet_list table not available in cache")
            return []
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_suppressed_bets(
        self, season: int | None, week: int | None
    ) -> list[dict[str, Any]]:
        """Return the SUPPRESSED bet-list rows for *season* / *week* (SPEC R6, plan 31-15).

        The exact COMPLEMENT of :meth:`get_bet_list`: both are derived from the one
        ``api.cache.BET_STATUS_LIVE`` constant, and the complement is expressed as
        ``IS DISTINCT FROM`` rather than ``<>`` so a row carrying a NULL status lands HERE instead
        of vanishing from both lists. Every row in ``bet_list`` therefore appears in exactly one of
        the two, which is what makes "a declined candidate is never silently dropped" a property of
        the partition rather than a claim about two independently-written WHERE clauses.

        Ordered by ``rejection_reason`` and then by the SPEC R5 four-key tie-break, so the grouped
        render is byte-identical across requests. As with the live list, ZERO computation happens
        here: a suppressed candidate was never priced, so there is nothing to recompute -- the page
        renders the reason the selector recorded (UIAP-01).
        """
        key = ("suppressed_bets", season, week)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_suppressed_bets_uncached(season, week)
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_suppressed_bets_uncached(
        self, season: int | None, week: int | None
    ) -> list[dict[str, Any]]:
        query = (
            f"SELECT {_BET_LIST_COLUMNS_SQL} FROM bet_list "
            "WHERE status IS DISTINCT FROM ?"
        )
        params: list[Any] = [BET_STATUS_LIVE]
        if season is not None:
            query += " AND season = ?"
            params.append(season)
        if week is not None:
            query += " AND week = ?"
            params.append(week)
        query += " ORDER BY rejection_reason NULLS LAST, season, week, game_id, target"

        try:
            result = self._conn.execute(query, params)
        except duckdb.Error:
            # The cache predates Phase 31 (no bet_list table). The route renders the
            # cache-absent empty state rather than 500ing.
            logger.warning("bet_list table not available in cache")
            return []
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def bet_list_table_exists(self) -> bool:
        """Return whether the ``bet_list`` cache table exists at all (plan 31-15).

        The page needs to tell TWO different absences apart, and an empty row list cannot:
        a cache that predates Phase 31 has no bet-list table and needs
        ``scripts/populate_cache.py`` run, while a populated cache whose week admitted nothing is
        a RESULT and needs no action. Both return ``[]`` from :meth:`get_bet_list`, so the page
        would otherwise tell a reader to rebuild a cache that is already correct -- or, worse,
        report a missing table as "no bets cleared the floor", which is a claim about the models
        made from the absence of a table.

        A zero-row probe: it reads the catalog, not the rows.
        """
        key = ("bet_list_table_exists",)
        cached = _cache_get(key)
        if cached is not None:
            return bool(cached)
        try:
            self._conn.execute("SELECT 1 FROM bet_list LIMIT 0")
        except duckdb.Error:
            logger.warning("bet_list table not available in cache")
            _cache_set(key, False)
            return False
        _cache_set(key, True)
        return True

    def get_available_bet_weeks(
        self, season: int | None = None
    ) -> list[dict[str, Any]]:
        """Return SCHEDULE-derived (season, week, game_count) rows, latest first (REVIEW-NAV).

        Reads ``available_bet_weeks`` ONLY. It never falls back to ``get_available_weeks``:
        that getter reads ``predictions``, so a scheduled week carrying no prediction row would
        vanish from navigation -- exactly the dependency this table exists to remove.
        """
        key = ("available_bet_weeks", season)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_available_bet_weeks_uncached(season)
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_available_bet_weeks_uncached(
        self, season: int | None
    ) -> list[dict[str, Any]]:
        query = "SELECT season, week, game_count FROM available_bet_weeks"
        params: list[Any] = []
        if season is not None:
            query += " WHERE season = ?"
            params.append(season)
        query += " ORDER BY season DESC, week DESC"

        try:
            result = self._conn.execute(query, params)
        except duckdb.Error:
            logger.warning("available_bet_weeks table not available in cache")
            return []
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_bet_seasons(self) -> list[int]:
        """Return the distinct SCHEDULE-derived bet seasons, latest first (REVIEW-NAV).

        Reads ``available_bet_weeks`` ONLY -- never ``get_available_seasons``, which reads
        ``backtest_metrics`` and therefore cannot see a scheduled-but-unbacktested season.
        """
        key = ("bet_seasons",)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_bet_seasons_uncached()
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_bet_seasons_uncached(self) -> list[int]:
        try:
            result = self._conn.execute(
                "SELECT DISTINCT season FROM available_bet_weeks ORDER BY season DESC"
            )
        except duckdb.Error:
            logger.warning("available_bet_weeks table not available in cache")
            return []
        return [row[0] for row in result.fetchall()]

    def get_bet_week_freeze(self, season: int | None, week: int | None) -> Any | None:
        """Return the latest per-game freeze instant for *season* / *week*, or None.

        The SCHEDULE-derived freshness source the Plan 31-18 stale-cache hard-block compares the
        cache stamp against (REVIEW-STALE). Returns None when the table is absent or the week has
        no row -- never a fallback to a bet-row-derived freeze, because the failure being guarded
        is a MISSING bet-list insertion.
        """
        key = ("bet_week_freeze", season, week)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_bet_week_freeze_uncached(season, week)
        if result is not None:
            _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_bet_week_freeze_uncached(
        self, season: int | None, week: int | None
    ) -> Any | None:
        if season is None or week is None:
            return None
        try:
            result = self._conn.execute(
                "SELECT latest_game_freeze_ts FROM bet_week_freeze "
                "WHERE season = ? AND week = ?",
                [season, week],
            )
        except duckdb.Error:
            logger.warning("bet_week_freeze table not available in cache")
            return None
        row = result.fetchone()
        return row[0] if row else None

    def get_bet_list_populated_at(
        self, season: int | None, week: int | None
    ) -> str | None:
        """Return when THAT week's bet list was last materialized, or None (D31-29).

        One half of the ``/bets`` stale-cache comparison; :meth:`get_bet_week_freeze` is the
        other. Both are keyed LOOKUPS and neither depends on a bet row existing, which is what
        lets the hard-block fire in the zero-row case it was built for.

        It reads the PER-WEEK ``bet_list_populated_at:<season>:<week>`` key and deliberately does
        NOT fall back to the generic ``cache_meta['last_updated']`` timestamp, nor to the bare
        ``bet_list_populated_at`` prefix. Both fallbacks would reintroduce the defect the per-week
        marker exists to close: a run that populated predictions and failed on the bet list would
        read as fresh, and the guard would be defeated by the exact failure it guards against
        (D31-29). ``None`` here means "this week's population never recorded a success", which the
        page resolves to the refusal when a freeze exists for the week.

        Returns None when either identifier is absent (there is no week to ask about) or when the
        cache predates Phase 31 and has no ``cache_meta`` row for the key.
        """
        if season is None or week is None:
            return None
        key = ("bet_list_populated_at", season, week)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_bet_list_populated_at_uncached(season, week)
        if result is not None:
            _cache_set(key, result)
        return result

    def _get_bet_list_populated_at_uncached(self, season: int, week: int) -> str | None:
        try:
            result = self._conn.execute(
                "SELECT value FROM cache_meta WHERE key = ?",
                [bet_list_populated_at_key(season, week)],
            )
        except duckdb.Error:
            logger.warning("cache_meta table not available in cache")
            return None
        row = result.fetchone()
        return row[0] if row else None

    def get_bet_tracker_blocks(self) -> list[dict[str, Any]]:
        """Return the PRECOMPUTED realized-vs-expected tracker blocks (SPEC R8, D31-22).

        A read, not a computation. Every figure was aggregated by
        ``backtest.bet_tracker.aggregate_by_provenance`` at population time and persisted by
        ``api.cache.materialize_bet_tracker_blocks``; this getter selects the stored columns and
        does no arithmetic of its own -- no count, no rate, no return, and no SQL aggregate
        function. Aggregation is computation, and computation does not happen in the request path
        (UIAP-01).

        The rows are already partitioned one per (provenance, validation_type) class, and nothing
        here pools them: a caller that wanted a figure spanning two classes would have to compute
        it itself, which is exactly what the tracker refuses to provide (SPEC R8 no-mixing).

        A NULL ``hit_rate`` beside ``bets_graded = 0`` means the rate was NOT COMPUTED for that
        block -- the template renders its empty state from that pair. It does not mean a measured
        zero.
        """
        key = ("bet_tracker_blocks",)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_bet_tracker_blocks_uncached()
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_bet_tracker_blocks_uncached(self) -> list[dict[str, Any]]:
        try:
            result = self._conn.execute(
                f"SELECT {_BET_TRACKER_BLOCK_COLUMNS_SQL} FROM bet_tracker_blocks"
            )
        except duckdb.Error:
            # The cache predates the tracker (no bet_tracker_blocks table). The page renders the
            # tracker-absent empty state rather than 500ing.
            logger.warning("bet_tracker_blocks table not available in cache")
            return []
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

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
