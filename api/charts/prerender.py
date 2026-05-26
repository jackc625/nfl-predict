"""Pre-render chart HTML for cache population.

This module houses the entry points called during DuckDB cache rebuild. Each
generator call is wrapped in per-chart try/except so a single failing generator
cannot poison the entire cache population (Codex HIGH #6).

The insights pre-render path ALSO produces the `insights_aggregate_table`
entry — a JSON blob consumed by Plan 16-03's route via
`DataService.get_insights_aggregate_table()`. Computing the aggregate table
here (not on request) keeps the request path cheap.

The Phase 17 betting path renders BOTH scope variants (``all`` /
``recommended``) of every betting chart and pre-computes the per-scope KPI strip
+ ROI summary table as JSON blobs (``betting_kpis_<scope>`` /
``betting_roi_table_<scope>``), consumed by Plan 17-04's route via
``DataService.get_betting_kpis()`` / ``get_betting_roi_table()``.

UIAP-01 COMPLIANCE: imports only from api.charts.core, api.charts.insights,
api.charts.betting, api.insights_metrics, and stdlib.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

# Import MODULES rather than individual symbols so ``unittest.mock.patch`` of
# ``api.charts.insights.generate_insights_*`` is observable here — patching the
# attribute on the source module mutates what we look up at call time through
# ``insights.generate_*``. Binding the name locally via ``from ... import X``
# would freeze the reference at import time and defeat the patch.
from api.charts import betting as _betting_charts
from api.charts import core as _core_charts
from api.charts import insights as _insights_charts
from api.charts.betting import BETTING_CHART_IDS
from api.charts.core import _empty_chart_div
from api.charts.insights import INSIGHTS_CHART_IDS
from api.insights_metrics import build_aggregate_rows

if TYPE_CHECKING:
    from api.services import DataService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-chart failure isolation (Codex HIGH #6)
# ---------------------------------------------------------------------------


def _safe_render(chart_id: str, fn: Callable[[], str]) -> str:
    """Run a chart generator; on failure return an empty-state div and WARN-log.

    The entire Phase 16 pre-render path is wrapped in this helper so one
    failing generator cannot abort the cache population for the others.
    """
    try:
        return fn()
    except Exception:  # noqa: BLE001 — per-chart isolation is intentional
        logger.warning("Pre-render failed for %s", chart_id, exc_info=True)
        return _empty_chart_div("Chart unavailable")


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------


# Allow-list of table names that the prerender path is permitted to query.
# DuckDB does not support parameterized table identifiers, so we validate the
# identifier against this set before f-string interpolation. Every current
# call site passes a string literal, but the allow-list prevents a future
# caller from accidentally wiring in a request-derived value.
_ALLOWED_TABLES: frozenset[str] = frozenset(
    {
        "backtest_predictions",
        "backtest_metrics",
        "predictions",
        "feature_importances",
        "equity_curve",
        "betting_bets",
    }
)


def _query_table(conn: Any, table: str) -> list[dict]:
    """Return every row of *table* as a list of column-keyed dicts.

    Returns an empty list if the table is empty. Avoids the
    ``conn.description`` reliance pattern the legacy code used by executing
    a single SELECT and zipping column names in one pass.

    Raises:
        ValueError: if *table* is not in :data:`_ALLOWED_TABLES`.
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"Refusing to query disallowed table: {table!r}")
    result = conn.execute(f"SELECT * FROM {table}")
    cols = [d[0] for d in result.description]
    return [dict(zip(cols, row)) for row in result.fetchall()]


def _extract_data_bundle(source: Any) -> dict[str, list[dict]]:
    """Normalize *source* into a data bundle dict.

    *source* may be:
      - a :class:`~api.services.DataService` instance (production / route path)
      - a raw DuckDB connection (cache-rebuild CLI path)
      - a pre-built dict mapping table names to lists of dicts (test path —
        see Plan 01 ``test_prerender_charts_failure_isolation``).

    The returned bundle always carries keys:
    ``backtest_predictions``, ``backtest_metrics``, ``predictions``,
    ``feature_importances``, ``equity_curve``, ``betting_bets``.
    """
    # Dict bundle shortcut (used directly by tests).
    if isinstance(source, dict):
        return {
            "backtest_predictions": source.get("backtest_predictions", []),
            "backtest_metrics": source.get("backtest_metrics", []),
            "predictions": source.get("predictions", []),
            "feature_importances": source.get("feature_importances", []),
            "equity_curve": source.get("equity_curve", []),
            "betting_bets": source.get("betting_bets", []),
        }

    # DataService instance: use public getters + public connection accessor.
    if hasattr(source, "get_backtest_predictions"):
        backtest_preds = source.get_backtest_predictions()
        metrics = source.get_backtest_metrics()
        preds = source.get_predictions()
        equity = source.get_equity_curve()
        conn = source.get_connection()
        fi_rows = _query_table(conn, "feature_importances")
        # betting_bets has no DataService request-path accessor (D-20: the
        # handler reads cached HTML/JSON only). Read the per-bet rows directly
        # via the allow-list-gated connection for the pre-render path only,
        # mirroring how feature_importances is fetched here.
        bets = _query_table(conn, "betting_bets")
        return {
            "backtest_predictions": backtest_preds,
            "backtest_metrics": metrics,
            "predictions": preds,
            "feature_importances": fi_rows,
            "equity_curve": equity,
            "betting_bets": bets,
        }

    # Raw DuckDB connection: read every table directly. Each table identifier
    # is validated against _ALLOWED_TABLES before f-string interpolation so a
    # future caller passing an attacker-controlled name cannot reach the SQL
    # stream.
    conn = source
    bundle: dict[str, list[dict]] = {}
    for table in (
        "backtest_predictions",
        "backtest_metrics",
        "predictions",
        "feature_importances",
        "betting_bets",
    ):
        if table not in _ALLOWED_TABLES:
            raise ValueError(f"Refusing to query disallowed table: {table!r}")
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        bundle[table] = _query_table(conn, table) if count > 0 else []
    # equity_curve has an explicit ORDER BY in the legacy code; keep the shape.
    if "equity_curve" not in _ALLOWED_TABLES:
        raise ValueError("Refusing to query disallowed table: 'equity_curve'")
    eq_count = conn.execute("SELECT COUNT(*) FROM equity_curve").fetchone()[0]
    if eq_count > 0:
        result = conn.execute("SELECT * FROM equity_curve ORDER BY strategy, bet_index")
        cols = [d[0] for d in result.description]
        bundle["equity_curve"] = [dict(zip(cols, row)) for row in result.fetchall()]
    else:
        bundle["equity_curve"] = []
    return bundle


# ---------------------------------------------------------------------------
# Insights generator orchestration
# ---------------------------------------------------------------------------


def _render_all(bundle: dict[str, list[dict]]) -> dict[str, str]:
    """Render every dashboard + insights chart from a normalized *bundle*.

    Each generator call is wrapped in :func:`_safe_render` so a failure in one
    generator falls back to the "Chart unavailable" empty-state div without
    affecting any other chart.
    """
    charts: dict[str, str] = {}

    backtest_preds = bundle["backtest_predictions"]
    metrics = bundle["backtest_metrics"]
    market_data = bundle["predictions"]
    feature_imps = bundle["feature_importances"]
    equity_data = bundle["equity_curve"]
    bets = bundle["betting_bets"]

    # --- Dashboard charts (pre-existing) ---
    charts["calibration"] = _safe_render(
        "calibration",
        lambda: _core_charts.generate_dashboard_calibration_chart(backtest_preds),
    )
    charts["clv"] = _safe_render(
        "clv",
        lambda: _core_charts.generate_dashboard_clv_chart(backtest_preds),
    )
    charts["heatmap"] = _safe_render(
        "heatmap",
        lambda: _core_charts.generate_dashboard_heatmap(metrics),
    )
    charts["equity"] = _safe_render(
        "equity",
        lambda: _core_charts.generate_dashboard_equity_chart(equity_data),
    )

    # --- Insights charts (Phase 16) ---
    # Each lambda looks up the generator through the module so
    # ``unittest.mock.patch("api.charts.insights.<fn>", ...)`` behaves as
    # expected (see tests/test_charts.py::test_prerender_charts_failure_isolation).
    charts["insights_calibration_ats"] = _safe_render(
        "insights_calibration_ats",
        lambda: _insights_charts.generate_insights_ats_calibration(backtest_preds),
    )
    charts["insights_calibration_ou"] = _safe_render(
        "insights_calibration_ou",
        lambda: _insights_charts.generate_insights_ou_calibration(backtest_preds),
    )
    charts["insights_feature_importance_wp"] = _safe_render(
        "insights_feature_importance_wp",
        lambda: _insights_charts.generate_insights_feature_importance_wp(feature_imps),
    )
    charts["insights_feature_importance_ats"] = _safe_render(
        "insights_feature_importance_ats",
        lambda: _insights_charts.generate_insights_feature_importance_ats(feature_imps),
    )
    charts["insights_feature_importance_ou"] = _safe_render(
        "insights_feature_importance_ou",
        lambda: _insights_charts.generate_insights_feature_importance_ou(feature_imps),
    )
    charts["insights_accuracy_trend"] = _safe_render(
        "insights_accuracy_trend",
        lambda: _insights_charts.generate_insights_accuracy_trend(metrics),
    )
    charts["insights_model_vs_market_wp"] = _safe_render(
        "insights_model_vs_market_wp",
        lambda: _insights_charts.generate_insights_model_vs_market_wp(
            backtest_preds,
            market_data,
        ),
    )
    charts["insights_model_vs_market_ats"] = _safe_render(
        "insights_model_vs_market_ats",
        lambda: _insights_charts.generate_insights_model_vs_market_ats(
            backtest_preds,
            market_data,
        ),
    )
    charts["insights_model_vs_market_ou"] = _safe_render(
        "insights_model_vs_market_ou",
        lambda: _insights_charts.generate_insights_model_vs_market_ou(
            backtest_preds,
            market_data,
        ),
    )

    # --- Aggregate table (Codex HIGH #1): computed once via shared helper ---
    try:
        market_by_game = {p["game_id"]: p for p in market_data}
        rows = build_aggregate_rows(backtest_preds, market_by_game)
        charts["insights_aggregate_table"] = json.dumps(rows)
    except Exception:  # noqa: BLE001 — match per-chart isolation contract
        logger.warning("Aggregate table build failed", exc_info=True)
        charts["insights_aggregate_table"] = json.dumps([])

    # --- Betting charts (Phase 17): BOTH scope variants per chart (D-20) ---
    # The one genuinely new behavior of the phase: loop the two scope variants
    # and store each chart + the KPI/ROI-table JSON blobs under a
    # ``betting_<chart>_<scope>`` id. ``filter_scope`` is computed once per scope
    # and every generator is reached through ``_betting_charts`` (module lookup)
    # so ``unittest.mock.patch("api.charts.betting.generate_*")`` is observable.
    # Each ``lambda`` binds ``s=scoped`` to avoid the closure-over-loop-var bug.
    for scope in ("all", "recommended"):
        scoped = _betting_charts.filter_scope(bets, scope)
        charts[f"betting_equity_{scope}"] = _safe_render(
            f"betting_equity_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_equity_chart(s),
        )
        charts[f"betting_equity_mini_wp_{scope}"] = _safe_render(
            f"betting_equity_mini_wp_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_equity_mini_wp(s),
        )
        charts[f"betting_equity_mini_ats_{scope}"] = _safe_render(
            f"betting_equity_mini_ats_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_equity_mini_ats(s),
        )
        charts[f"betting_equity_mini_ou_{scope}"] = _safe_render(
            f"betting_equity_mini_ou_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_equity_mini_ou(s),
        )
        charts[f"betting_roi_type_{scope}"] = _safe_render(
            f"betting_roi_type_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_roi_type(s),
        )
        charts[f"betting_roi_season_{scope}"] = _safe_render(
            f"betting_roi_season_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_roi_season(s),
        )
        charts[f"betting_roi_bucket_{scope}"] = _safe_render(
            f"betting_roi_bucket_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_roi_bucket(s),
        )
        charts[f"betting_edge_hist_wp_{scope}"] = _safe_render(
            f"betting_edge_hist_wp_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_edge_hist_wp(s),
        )
        charts[f"betting_edge_hist_ats_{scope}"] = _safe_render(
            f"betting_edge_hist_ats_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_edge_hist_ats(s),
        )
        charts[f"betting_edge_hist_ou_{scope}"] = _safe_render(
            f"betting_edge_hist_ou_{scope}",
            lambda s=scoped: _betting_charts.generate_betting_edge_hist_ou(s),
        )

        # KPI strip + ROI summary table as per-scope JSON blobs (D-20). Each is
        # wrapped in its own try/except mirroring the aggregate-table isolation:
        # a build failure falls back to an empty dict / list, never raising.
        try:
            charts[f"betting_kpis_{scope}"] = json.dumps(
                _betting_charts.compute_kpis(scoped),
            )
        except Exception:  # noqa: BLE001 — match per-chart isolation contract
            logger.warning("Betting KPI build failed for %s", scope, exc_info=True)
            charts[f"betting_kpis_{scope}"] = json.dumps({})
        try:
            charts[f"betting_roi_table_{scope}"] = json.dumps(
                _betting_charts.compute_roi_table(scoped),
            )
        except Exception:  # noqa: BLE001 — match per-chart isolation contract
            logger.warning(
                "Betting ROI table build failed for %s", scope, exc_info=True
            )
            charts[f"betting_roi_table_{scope}"] = json.dumps([])

    # Self-check: every declared insights + betting chart_id produced a value.
    for chart_id in INSIGHTS_CHART_IDS:
        assert chart_id in charts, f"Missing chart_id in prerender output: {chart_id}"
    for chart_id in BETTING_CHART_IDS:
        assert chart_id in charts, f"Missing chart_id in prerender output: {chart_id}"

    return charts


# ---------------------------------------------------------------------------
# Public entry points (API preserved)
# ---------------------------------------------------------------------------


def prerender_charts_for_cache(
    service: DataService | dict[str, list[dict]],
) -> dict[str, str]:
    """Generate dashboard + insights charts from a :class:`DataService`.

    Args:
        service: DataService instance backed by a populated DuckDB connection,
            OR a dict bundle (test path — see Plan 16-01
            ``test_prerender_charts_failure_isolation``) keyed by table name.

    Returns:
        Dict mapping chart_id to HTML div string. The return also carries the
        ``insights_aggregate_table`` key (JSON-encoded list of aggregate rows
        consumed by Plan 16-03's template) and the per-scope
        ``betting_kpis_<scope>`` (JSON dict) / ``betting_roi_table_<scope>``
        (JSON list) blobs consumed by Plan 17-04's route.
    """
    bundle = _extract_data_bundle(service)
    return _render_all(bundle)


def prerender_charts_from_conn(conn: Any) -> dict[str, str]:
    """Generate dashboard + insights charts directly from a DuckDB connection.

    Used during cache population when the database is already open in write
    mode (can't open a second read-only connection to the same file).

    Args:
        conn: Active DuckDB connection with populated tables.

    Returns:
        Dict mapping chart_id to HTML div string.
    """
    bundle = _extract_data_bundle(conn)
    return _render_all(bundle)
