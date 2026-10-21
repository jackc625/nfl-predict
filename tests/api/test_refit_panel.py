"""The /season comparison panel: re-fitting vs standing still, raw numbers only (Plan 37-19).

Phase 37 D-12..D-14, SPEC R13, must-NOT P7. The panel is Section 5 of the Season page, rendered
for the one season the re-fit belongs to (``api.cache_schema.REFIT_PANEL_SEASON``) from the two
cache tables Plan 37-18 writes (``refit_panel``, ``refit_aa_status``). The request path reads the
stored rows and formats them; it never computes or tests anything, so the page can never show a
p-value or anything that reads as the season-close verdict or as profit.

Every cache here is built under ``tmp_path`` and served through a ``get_db`` override; nothing
reads ``data/web_cache.duckdb``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from fastapi.testclient import TestClient

from api.cache import CACHE_SCHEMA, materialize_refit_panel
from api.cache_schema import (
    REFIT_AA_STATUS_COLUMNS,
    REFIT_PANEL_COLUMNS,
    REFIT_PANEL_PERIOD,
    REFIT_PANEL_SEASON,
    REFIT_PANEL_SEASON_TO_DATE,
)
from api.services import clear_cache

_TITLE = "Re-fitting vs standing still -- not the verdict"
_SEASON_CLOSE_LINE = "The pre-registered test runs once, at season close. These numbers move week to week."
_OTHER_SEASON = REFIT_PANEL_SEASON - 1

# The words the owner rule bans from the section, matched as WHOLE words over the visible text.
_BANNED = re.compile(
    r"\b(?:p-value|significant|confidence|profit|edge|bet)s?\b", re.IGNORECASE
)

_UNITS = {"wp": "log loss", "ats": "points", "ou": "points"}

# A KPI blob with one graded winner pick, so the page's KPI guard shows the scoreboard.
_GRADED_KPIS: dict[str, Any] = {
    "wp_hit_rate": 100.0,
    "wp_decided": 1,
    "wp_hits": 1,
    "ats_hit_rate": None,
    "ats_decided": 0,
    "ats_hits": 0,
    "ou_hit_rate": None,
    "ou_decided": 0,
    "ou_hits": 0,
    "record": "1-0",
}


def _panel_row(
    target: str,
    period_index: int | None,
    label: str,
    *,
    live: float | None,
    frozen: float | None,
    games: int,
    start: str | None = None,
) -> dict[str, Any]:
    """One stored ``refit_panel`` row, the shape ``refit.comparison.period_summary`` writes."""
    gap = None if live is None or frozen is None else live - frozen
    ahead = (
        None
        if gap is None
        else ("live" if gap < 0 else "frozen" if gap > 0 else "even")
    )
    return {
        "season": REFIT_PANEL_SEASON,
        "target": target,
        "period_index": period_index,
        "row_kind": REFIT_PANEL_PERIOD
        if period_index is not None
        else REFIT_PANEL_SEASON_TO_DATE,
        "period_label": label,
        "period_start": start,
        "live_error": live,
        "frozen_error": frozen,
        "gap": gap,
        "ahead": ahead,
        "paired_games": games,
        "unit": _UNITS[target],
    }


def _aa_row(week: int, clean: bool, field: str | None = None) -> dict[str, Any]:
    """One stored ``refit_aa_status`` row."""
    return {
        "season": REFIT_PANEL_SEASON,
        "week": week,
        "evaluated_at": f"2026-11-0{week - 7}T22:00:00+00:00",
        "clean": clean,
        "reason": "clean" if clean else "field_mismatch",
        "live_rows": 40,
        "twins": 40,
        "mismatch_field": field,
        "mismatch_game": None if clean else "2026_W09_AAA@BBB",
        "mismatch_target": None if clean else "wp",
    }


def _build_cache(
    db_path: Path,
    *,
    panel: list[dict[str, Any]] | None = None,
    aa: list[dict[str, Any]] | None = None,
    kpis: dict[str, Any] | None = None,
    with_tables: bool = True,
) -> None:
    """A cache with predictions for two seasons, the season KPI blob and the panel tables."""
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            if statement.strip():
                conn.execute(statement.strip())
        for season in (_OTHER_SEASON, REFIT_PANEL_SEASON):
            conn.execute(
                "INSERT INTO predictions (game_id, season, week) VALUES (?, ?, 1)",
                [f"{season}_W01_AAA@BBB", season],
            )
            conn.execute(
                "INSERT INTO chart_cache (chart_id, html_div) VALUES (?, ?)",
                [
                    f"season_kpis_{season}",
                    json.dumps(_GRADED_KPIS if kpis is None else kpis),
                ],
            )
        if with_tables:
            materialize_refit_panel(
                conn,
                pd.DataFrame(panel or [], columns=pd.Index(REFIT_PANEL_COLUMNS)),
                pd.DataFrame(aa or [], columns=pd.Index(REFIT_AA_STATUS_COLUMNS)),
            )
        else:
            conn.execute("DROP TABLE IF EXISTS refit_panel")
            conn.execute("DROP TABLE IF EXISTS refit_aa_status")
    finally:
        conn.close()


@contextmanager
def _client(db_path: Path) -> Iterator[TestClient]:
    from api.dependencies import get_db
    from api.main import app

    clear_cache()
    conn = duckdb.connect(str(db_path), read_only=True)
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = lambda: conn
    try:
        yield TestClient(app, raise_server_exceptions=False)
    finally:
        app.dependency_overrides.pop(get_db, None)
        conn.close()


def _page(tmp_path: Path, season: int = REFIT_PANEL_SEASON, **cache: Any) -> str:
    db_path = tmp_path / "panel_cache.duckdb"
    _build_cache(db_path, **cache)
    with _client(db_path) as client:
        response = client.get(f"/season?season={season}")
    assert response.status_code == 200, response.text[:500]
    return response.text


def _section(html: str) -> str:
    """The panel section's markup, or "" when the page has none."""
    match = re.search(r'<section id="refit-comparison".*?</section>', html, re.S)
    return match.group(0) if match else ""


def _visible(markup: str) -> str:
    """The section's visible text: tags removed, entities decoded, whitespace collapsed."""
    text = re.sub(r"<[^>]+>", " ", markup)
    text = text.replace("&#39;", "'").replace("&amp;", "&").replace("&quot;", '"')
    return re.sub(r"\s+", " ", text)


def _full_season_rows() -> list[dict[str, Any]]:
    """Before the first re-fit, Period 1 and season to date, for all three targets."""
    rows: list[dict[str, Any]] = []
    figures = {
        "wp": ((0.6812, 0.6812, 40), (0.6531, 0.6619, 31)),
        "ats": ((10.4321, 10.4321, 40), (9.8765, 10.1234, 31)),
        "ou": ((11.1111, 11.1111, 40), (10.5561, 10.2222, 31)),
    }
    for target, (before, period) in figures.items():
        rows.append(
            _panel_row(
                target,
                0,
                "Before the first re-fit",
                live=before[0],
                frozen=before[1],
                games=before[2],
            )
        )
        rows.append(
            _panel_row(
                target,
                1,
                "Period 1, from Tue Nov 3",
                live=period[0],
                frozen=period[1],
                games=period[2],
                start="2026-11-03",
            )
        )
        rows.append(
            _panel_row(
                target,
                None,
                "Season to date: every paired game since Tue Nov 3",
                live=period[0],
                frozen=period[1],
                games=period[2],
                start="2026-11-03",
            )
        )
    return rows


def test_panel_renders_from_cache(tmp_path: Path) -> None:
    """The title, the explainer, each target's stored figures and the season-close line."""
    section = _section(
        _page(tmp_path, panel=_full_season_rows(), aa=[_aa_row(8, True)])
    )
    text = _visible(section)
    assert _TITLE in text
    assert "Lower is better" in text
    assert _SEASON_CLOSE_LINE in text
    for label in ("Before the first re-fit", "Period 1, from Tue Nov 3"):
        assert label in text
    # Stored figures, formatted only: three decimals of log loss, two of points.
    for figure in (
        "0.653",
        "0.662",
        "-0.009",
        "9.88",
        "10.12",
        "-0.25",
        "10.56",
        "10.22",
        "+0.33",
    ):
        assert figure in text, figure
    assert "31" in text and "40" in text
    for ahead in ("Live", "Frozen", "Even"):
        assert ahead in text, ahead
    assert "No re-fit yet" not in text


def test_panel_has_no_test_or_profit_words(tmp_path: Path) -> None:
    """P7, D-13: none of the six banned words, as whole words; "verdict" only in "not the verdict"."""
    section = _section(
        _page(
            tmp_path,
            panel=_full_season_rows(),
            aa=[_aa_row(8, True), _aa_row(9, False, "per_bet_ev")],
        )
    )
    text = _visible(section)
    assert _BANNED.findall(text) == []
    assert "Lower is better" in text
    assert "did not match (per_bet_ev)" in text
    assert text.lower().count("verdict") == text.lower().count("not the verdict") == 1


def test_before_the_first_refit(tmp_path: Path) -> None:
    """Edge R13 empty: generation-0 rows only -> "No re-fit yet" and the A/A weeks."""
    rows = [
        _panel_row(t, 0, "Before the first re-fit", live=0.5, frozen=0.5, games=12)
        for t in ("wp", "ats", "ou")
    ]
    text = _visible(
        _section(
            _page(
                tmp_path,
                panel=rows,
                aa=[_aa_row(8, True), _aa_row(9, False, "per_bet_ev")],
            )
        )
    )
    assert "No re-fit yet" in text
    assert "Week 8: matched" in text
    assert "Week 9: did not match (per_bet_ev)" in text


def test_periods_in_order(tmp_path: Path) -> None:
    """Edge R13 ordering: periods in increasing order, the season-to-date row last."""
    rows = [
        _panel_row(
            "wp",
            None,
            "Season to date: every paired game since Tue Nov 3",
            live=0.6,
            frozen=0.7,
            games=60,
        ),
        _panel_row(
            "wp",
            2,
            "Period 2, from Tue Dec 1",
            live=0.6,
            frozen=0.7,
            games=29,
            start="2026-12-01",
        ),
        _panel_row("wp", 0, "Before the first re-fit", live=0.6, frozen=0.6, games=40),
        _panel_row(
            "wp",
            1,
            "Period 1, from Tue Nov 3",
            live=0.6,
            frozen=0.7,
            games=31,
            start="2026-11-03",
        ),
    ]
    text = _visible(_section(_page(tmp_path, panel=rows)))
    positions = [
        text.find(label)
        for label in (
            "Before the first re-fit",
            "Period 1, from Tue Nov 3",
            "Period 2, from Tue Dec 1",
            "Season to date",
        )
    ]
    assert -1 not in positions, positions
    assert positions == sorted(positions), positions


def test_htmx_fragment_carries_the_panel(tmp_path: Path) -> None:
    """The selector's HX swap returns the season block, and the panel is inside it."""
    db_path = tmp_path / "panel_cache.duckdb"
    _build_cache(db_path, panel=_full_season_rows())
    with _client(db_path) as client:
        response = client.get(
            f"/season?season={REFIT_PANEL_SEASON}", headers={"HX-Request": "true"}
        )
    assert response.status_code == 200
    assert "<html" not in response.text.lower()
    assert _TITLE in _visible(_section(response.text))


def test_absent_tables_read_not_available(tmp_path: Path) -> None:
    """A cache without the tables shows the section and "not available", never zero, never 500."""
    text = _visible(_section(_page(tmp_path, with_tables=False)))
    assert _TITLE in text
    assert "not available" in text
    assert "No re-fit yet" not in text


def test_other_seasons_show_no_panel(tmp_path: Path) -> None:
    html = _page(tmp_path, season=_OTHER_SEASON, panel=_full_season_rows())
    assert _section(html) == ""
    assert "not the verdict" not in html


def test_the_panel_shows_when_there_are_no_graded_kpis(tmp_path: Path) -> None:
    """Review round 1: the panel sits outside the KPI empty-state guard."""
    empty_kpis = {
        key: (0 if key.endswith(("_decided", "_hits")) else None)
        for key in _GRADED_KPIS
    }
    html = _page(tmp_path, kpis=empty_kpis, aa=[_aa_row(8, True)])
    assert "No completed games yet" in html
    text = _visible(_section(html))
    assert _TITLE in text
    assert "Week 8: matched" in text


def test_an_install_with_no_pairs_is_not_no_refit_yet(tmp_path: Path) -> None:
    """Review round 1: an installed period with no completed pair is a period, not "No re-fit yet"."""
    rows = [
        _panel_row(
            t,
            1,
            "Period 1, from Tue Nov 3",
            live=None,
            frozen=None,
            games=0,
            start="2026-11-03",
        )
        for t in ("wp", "ats", "ou")
    ]
    text = _visible(_section(_page(tmp_path, panel=rows)))
    assert "No re-fit yet" not in text
    assert "Period 1, from Tue Nov 3" in text
    assert "not measured yet (0 paired games)" in text
