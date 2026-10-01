"""The dark Broadcast chart theme every dashboard chart shares (redesign Task 17).

Layouts are read back through ``fig.to_dict()`` so the assertions see exactly what Plotly
serialises into the prerendered HTML.
"""

from __future__ import annotations

from typing import Any, cast

import plotly.graph_objects as go
import plotly.io as pio
import pytest
from plotly.subplots import make_subplots

from api.charts import theme


def _figure() -> go.Figure:
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[2, 1, 3], name="Model"))
    fig.update_layout(
        title={"text": "Example", "x": 0.5}, legend={"x": 0.02, "y": 0.98}
    )
    return fig


def _layout(fig: go.Figure) -> dict[str, Any]:
    return fig.to_dict()["layout"]


def test_backgrounds_are_transparent_so_the_panel_shows_through() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    layout = _layout(fig)
    assert layout["paper_bgcolor"] == theme.TRANSPARENT
    assert layout["plot_bgcolor"] == theme.TRANSPARENT


def test_the_default_light_template_is_dropped() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    assert _layout(fig)["template"] == pio.templates["none"].to_plotly_json()


def test_axis_text_uses_the_display_face_and_hover_uses_the_body_face() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    layout = _layout(fig)
    assert layout["font"]["family"] == theme.FONT_DISPLAY
    assert layout["font"]["color"] == theme.MUTED
    assert layout["title"]["font"]["family"] == theme.FONT_DISPLAY
    assert layout["title"]["font"]["color"] == theme.FG
    assert layout["hoverlabel"]["font"]["family"] == theme.FONT_BODY


def test_gridlines_are_faint_and_zero_lines_are_off() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    layout = _layout(fig)
    for axis in ("xaxis", "yaxis"):
        assert layout[axis]["gridcolor"] == theme.GRID
        assert layout[axis]["showgrid"] is True
        assert layout[axis]["zeroline"] is False


def test_every_subplot_axis_and_subplot_title_is_themed() -> None:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "MAE"))
    fig.add_trace(go.Scatter(x=[1], y=[1]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1], y=[1]), row=1, col=2)
    theme.apply_dark_theme(fig, legend_position="bottom")
    layout = _layout(fig)
    assert layout["xaxis2"]["gridcolor"] == theme.GRID
    assert layout["yaxis2"]["zeroline"] is False
    assert layout["annotations"], "make_subplots should have produced subplot titles"
    assert all(a["font"]["family"] == theme.FONT_DISPLAY for a in layout["annotations"])


def test_hover_label_is_dark_with_the_accent_border() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    hover = _layout(fig)["hoverlabel"]
    assert hover["bgcolor"] == theme.INK
    assert hover["bordercolor"] == theme.ACCENT


def test_legend_sits_on_top_by_default_and_overrides_a_leftover_position() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    legend = _layout(fig)["legend"]
    assert legend["orientation"] == "h"
    assert legend["yanchor"] == "bottom"
    assert legend["y"] >= 1.0
    assert legend["x"] == 0
    assert legend["bgcolor"] == theme.TRANSPARENT


def test_legend_can_sit_below_the_plot() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig, legend_position="bottom")
    layout = _layout(fig)
    assert layout["legend"]["yanchor"] == "top"
    assert layout["legend"]["y"] < 0
    assert layout["margin"]["b"] >= theme.MARGINS["top"]["b"]


def test_an_unknown_legend_position_is_refused() -> None:
    with pytest.raises(ValueError, match="legend_position"):
        theme.apply_dark_theme(_figure(), legend_position=cast(Any, "left"))


def test_bet_type_colours_are_the_spec_keys() -> None:
    assert theme.TARGET_COLORS == {"wp": "#FFD400", "ats": "#4CC9F0", "ou": "#C77DFF"}


def test_season_colours_cover_2018_to_2026_distinctly_and_avoid_reserved_colours() -> (
    None
):
    assert set(range(2018, 2027)) <= set(theme.SEASON_COLORS)
    colours = {colour.upper() for colour in theme.SEASON_COLORS.values()}
    assert len(colours) == len(theme.SEASON_COLORS)
    # Green/red mean a realised result, and the bet-type colours and the accent mean
    # Winner / Spread / Totals and emphasis on every chart (spec 7.4), so no season takes one.
    reserved = {
        theme.WIN_COLOR,
        theme.LOSS_COLOR,
        theme.ACCENT,
        *theme.TARGET_COLORS.values(),
    }
    assert not colours & {colour.upper() for colour in reserved}


def test_strategy_colours_are_not_outcome_colours() -> None:
    assert not {theme.WIN_COLOR, theme.LOSS_COLOR} & set(theme.STRATEGY_COLORS.values())


def test_core_takes_its_palette_from_the_theme() -> None:
    from api.charts import core

    assert core.TARGET_COLORS is theme.TARGET_COLORS
    assert core.SEASON_COLORS is theme.SEASON_COLORS
    assert core.DEFAULT_COLOR == theme.MUTED
    assert core._get_target_color("ats") == "#4CC9F0"


def test_layout_defaults_delegate_to_the_theme() -> None:
    from api.charts.core import _apply_layout_defaults

    fig = _figure()
    _apply_layout_defaults(fig)
    layout = _layout(fig)
    assert layout["paper_bgcolor"] == theme.TRANSPARENT
    assert layout["hoverlabel"]["bordercolor"] == theme.ACCENT

    bottom = _figure()
    _apply_layout_defaults(bottom, legend_position="bottom")
    assert _layout(bottom)["legend"]["y"] < 0


def test_empty_chart_div_uses_dark_tokens_and_keeps_its_message() -> None:
    from api.charts.core import _empty_chart_div

    html = _empty_chart_div("Chart unavailable")
    assert "Chart unavailable" in html
    assert "text-muted" in html
    assert "text-gray-500" not in html
