"""Back-compat facade for the legacy flat ``api/charts.py`` module.

The package layout is:
- core.py       pre-existing chart generators + helpers
- insights.py   Phase 16 insights generators (``INSIGHTS_CHART_IDS``)
- betting.py    Phase 17 betting generators (``BETTING_CHART_IDS``)
- prerender.py  ``prerender_charts_for_cache`` / ``prerender_charts_from_conn``
                with per-chart failure isolation

Every symbol previously accessible via ``from api.charts import X`` remains
accessible through the re-exports below. Callers under ``api/routes/``,
``api/cache.py``, and tests do not need any import changes.
"""

from api.charts.betting import (  # noqa: F401
    BETTING_CHART_IDS,
    compute_kpis,
    compute_roi_table,
    filter_scope,
    generate_betting_edge_hist_ats,
    generate_betting_edge_hist_ou,
    generate_betting_edge_hist_wp,
    generate_betting_equity_chart,
    generate_betting_equity_mini_ats,
    generate_betting_equity_mini_ou,
    generate_betting_equity_mini_wp,
    generate_betting_roi_bucket,
    generate_betting_roi_season,
    generate_betting_roi_type,
)
from api.charts.core import (  # noqa: F401
    CHART_LAYOUT_DEFAULTS,
    DEFAULT_COLOR,
    PLOTLY_CONFIG,
    _apply_layout_defaults,
    _empty_chart_div,
    _get_season_color,
    _get_target_color,
    _to_html,
    generate_dashboard_calibration_chart,
    generate_dashboard_clv_chart,
    generate_dashboard_equity_chart,
    generate_dashboard_heatmap,
)
from api.charts.insights import (  # noqa: F401
    INSIGHTS_CHART_IDS,
    generate_insights_accuracy_trend,
    generate_insights_ats_calibration,
    generate_insights_feature_importance,
    generate_insights_feature_importance_ats,
    generate_insights_feature_importance_ou,
    generate_insights_feature_importance_wp,
    generate_insights_model_vs_market_ats,
    generate_insights_model_vs_market_ou,
    generate_insights_model_vs_market_wp,
    generate_insights_ou_calibration,
)
from api.charts.prerender import (  # noqa: F401
    prerender_charts_for_cache,
    prerender_charts_from_conn,
)
from api.charts.season import (  # noqa: F401
    compute_cumulative_series,
    compute_season_kpis,
    compute_weekly_series,
    generate_season_cumulative,
    generate_season_weekly,
)
