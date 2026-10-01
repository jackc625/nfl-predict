"""Dark "Broadcast" chart theme shared by every prerendered dashboard chart.

One module owns the chart look, so the generators in ``api.charts`` cannot drift apart:
- the figure is transparent, so the page's panel shows through;
- axis, legend and subplot-title text uses the condensed display face the HTML uses;
- the hover label matches the site's dark tooltip, with the yellow accent border.

Colour meaning follows the site-wide rule (spec section 4). ``WIN_COLOR`` and
``LOSS_COLOR`` are for realised win/loss and profit/loss series only. They are never used
for a model, market, season or strategy series. Bet types always take ``TARGET_COLORS``.
A market series is ``MUTED`` and dashed, beside its solid bet-type colour.

This module imports nothing from ``backtest/`` (tests/api/test_import_guard_bets.py). The
web palette is therefore independent of the white-background backtest HTML reports, which
keep ``backtest.report``'s own palette.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import plotly.graph_objects as go

# Base colours: the same values as the Tailwind tokens in web/static/input.css.
INK: Final = "#0B0F17"
PANEL: Final = "#151B29"
FG: Final = "#F3F5F9"
MUTED: Final = "#8A93A8"
ACCENT: Final = "#FFD400"
TRANSPARENT: Final = "rgba(0,0,0,0)"
GRID: Final = "rgba(255,255,255,0.06)"
REFERENCE_LINE: Final = "rgba(255,255,255,0.45)"
# Season-boundary verticals: visible, but quieter than a reference line.
BOUNDARY_LINE: Final = "rgba(255,255,255,0.14)"
LEGEND_TEXT: Final = "#C4CAD8"

# Realised outcomes only: Tailwind green-500 / red-500, the same hues the HTML uses.
WIN_COLOR: Final = "#22C55E"
LOSS_COLOR: Final = "#EF4444"
NEUTRAL_COLOR: Final = "#5B6478"

FONT_DISPLAY: Final = "Barlow Condensed, Inter, sans-serif"
FONT_BODY: Final = "Inter, system-ui, sans-serif"
FONT_MONO: Final = "JetBrains Mono, monospace"

TARGET_COLORS: dict[str, str] = {"wp": "#FFD400", "ats": "#4CC9F0", "ou": "#C77DFF"}

# Distinct on the dark panel and free of every reserved colour: green/red (realised
# outcomes), the three bet-type colours and the accent (Winner / Spread / Totals and emphasis
# on every chart, spec 7.4). 2026, the live season, is the brightest line (near-white; not
# #FFFFFF, which the chart-layer literal checks treat as a leftover light-theme colour).
SEASON_COLORS: dict[int, str] = {
    2018: "#94A3B8",
    2019: "#A78BFA",
    2020: "#F472B6",
    2021: "#60A5FA",
    2022: "#818CF8",
    2023: "#FDBA74",
    2024: "#E879F9",
    2025: "#D6D3D1",
    2026: "#F8FAFC",
}

# Flat and Kelly are staking strategies, not outcomes, so they never take green/red.
STRATEGY_COLORS: dict[str, str] = {"flat_stake": "#E6E9F0", "kelly": "#FF8A3D"}

# Monochrome metric heatmap: brighter = a higher value. Each heatmap trace keeps its
# pre-existing direction (reversed exactly where the old red-yellow-green scale was), so
# brighter is not "better" on every column. The bright end is held to #646E86 so the FG cell
# numbers stay at 4.67:1 on the brightest cell (WCAG AA; #7C869C gave 3.35:1), while the two
# ends of the scale still differ by 3:1.
HEATMAP_SCALE: list[list[float | str]] = [[0.0, "#1D2436"], [1.0, "#646E86"]]

LegendPosition = Literal["top", "bottom"]

# "top": just above the plot area, under the title.
# "bottom": under the x-axis title, for charts whose top edge already carries subplot
# titles or a long legend. Its offset is a fraction of the plot height, so it must leave room
# for the tick labels and axis title on a short plot too: at -0.22 a narrow card's wrapped
# legend sat on the x-axis title.
_LEGENDS: dict[str, dict[str, Any]] = {
    "top": {
        "orientation": "h",
        "x": 0,
        "xanchor": "left",
        "y": 1.02,
        "yanchor": "bottom",
    },
    "bottom": {
        "orientation": "h",
        "x": 0,
        "xanchor": "left",
        "y": -0.3,
        "yanchor": "top",
    },
}
MARGINS: dict[str, dict[str, int]] = {
    "top": {"l": 52, "r": 16, "t": 76, "b": 48},
    "bottom": {"l": 52, "r": 16, "t": 64, "b": 96},
}


def apply_dark_theme(
    fig: go.Figure, *, legend_position: LegendPosition = "top"
) -> None:
    """Apply the dark Broadcast look to *fig*, in place.

    Call it LAST, after the generator's own layout. It drops Plotly's default light
    template, then sets backgrounds, fonts, gridlines, the hover label and the legend
    placement. That overrides any per-chart leftovers, so every chart reads as one system.
    """
    if legend_position not in _LEGENDS:
        msg = f"legend_position must be 'top' or 'bottom', got {legend_position!r}"
        raise ValueError(msg)
    fig.update_layout(
        template="none",
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font={"family": FONT_DISPLAY, "size": 13, "color": MUTED},
        title_font={"family": FONT_DISPLAY, "size": 17, "color": FG},
        legend={
            **_LEGENDS[legend_position],
            "bgcolor": TRANSPARENT,
            "font": {"family": FONT_DISPLAY, "size": 13, "color": LEGEND_TEXT},
        },
        hoverlabel={
            "bgcolor": INK,
            "bordercolor": ACCENT,
            "font": {"family": FONT_BODY, "size": 13, "color": FG},
        },
        margin=MARGINS[legend_position],
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, showline=False, automargin=True
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, showline=False, automargin=True
    )
    # Subplot titles and reference-line labels are annotations, so they get the same
    # styling as axis text.
    fig.update_annotations(font={"family": FONT_DISPLAY, "size": 13, "color": MUTED})
