"""The chart-id tuples, in a module that imports nothing (Phase 36 D-11).

The page routes need only these ids to read pre-rendered charts out of the cache. They used to
import them from ``api.charts``, and that package imports plotly, numpy and scikit-learn, so every
page request loaded the plotting and modelling stack. The ids now live here, with the constants
``BETTING_CHART_IDS`` is built from, and ``api.charts.betting`` / ``api.charts.insights`` import
them back, so ``api.charts.BETTING_CHART_IDS is api.chart_ids.BETTING_CHART_IDS``.

Both tuples must match the independent copies in ``tests/api/conftest.py``, element for element.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

# ---------------------------------------------------------------------------
# Authoritative betting chart-id tuple (must match tests/api/conftest.py BETTING_CHART_IDS)
# ---------------------------------------------------------------------------
# Naming convention: ``betting_<chart>_<scope>`` for scope in ("all",
# "recommended"). 12 chart bases x 2 scopes = 24 ids (D-20). The two JSON-blob
# bases (kpis, roi_table) carry decodable JSON rather than chart HTML; their ids
# are part of this tuple so the prerender self-check covers them too.

_BETTING_CHART_BASES: tuple[str, ...] = (
    "betting_equity",
    "betting_equity_mini_wp",
    "betting_equity_mini_ats",
    "betting_equity_mini_ou",
    "betting_roi_type",
    "betting_roi_season",
    "betting_roi_bucket",
    "betting_edge_hist_wp",
    "betting_edge_hist_ats",
    "betting_edge_hist_ou",
    "betting_kpis",
    "betting_roi_table",
)
BETTING_SCOPES: tuple[str, ...] = ("all", "recommended")
BETTING_CHART_IDS: tuple[str, ...] = tuple(
    f"{base}_{scope}" for base in _BETTING_CHART_BASES for scope in BETTING_SCOPES
)
assert len(BETTING_CHART_IDS) == 24  # 12 chart bases x 2 scopes


# ---------------------------------------------------------------------------
# Authoritative insights chart-id tuple (Codex HIGH #7)
# ---------------------------------------------------------------------------

INSIGHTS_CHART_IDS: tuple[str, ...] = (
    "insights_calibration_ats",
    "insights_calibration_ou",
    "insights_feature_importance_wp",
    "insights_feature_importance_ats",
    "insights_feature_importance_ou",
    "insights_accuracy_trend",
    "insights_model_vs_market_wp",
    "insights_model_vs_market_ats",
    "insights_model_vs_market_ou",
)
assert len(INSIGHTS_CHART_IDS) == 9  # Codex HIGH #7
