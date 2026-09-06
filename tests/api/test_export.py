"""Tests for CSV and JSON export endpoints.

Validates UIAP-06: CSV and JSON export with Content-Disposition headers
and valid file content.

Plan 31-16 EXTENDS this module (it does not rewrite it) with the ``type=bets``
branch D31-32 adds to both shipped handlers: the week's full bet-list record,
read through the same two service getters ``/bets`` reads, in the same order,
carrying the status and the two D31-22 honesty labels on every row.

Selectors (``-k``): bets, cross_link, export_buttons, order, honesty.
"""

from __future__ import annotations

import json
import re
import subprocess
import threading
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.cache import (
    BET_LIST_COLUMNS,
    BET_LIST_POPULATED_AT_KEY,
    CACHE_SCHEMA,
    GRADING_STATUS_PENDING,
    materialize_available_bet_weeks,
    materialize_bet_list,
)
from api.services import clear_cache


def test_csv_export(test_client: TestClient):
    """UIAP-06: CSV export returns valid file with Content-Disposition."""
    response = test_client.get("/api/export/csv")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/csv")
    assert "Content-Disposition" in response.headers
    assert "attachment" in response.headers["Content-Disposition"]
    # Verify CSV structure
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2  # header + at least 1 row
    assert "game_id" in lines[0]


def test_json_export(test_client: TestClient):
    """UIAP-06: JSON export returns valid JSON array."""
    response = test_client.get("/api/export/json")
    assert response.status_code == 200
    data = json.loads(response.text)
    assert isinstance(data, list)
    assert len(data) >= 1
    assert "game_id" in data[0]


def test_csv_export_with_season_filter(test_client: TestClient):
    """CSV export with season filter returns matching data."""
    response = test_client.get("/api/export/csv?season=2024")
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2  # header + at least 1 row
    # All data rows should be from 2024
    assert "2024" in response.text


def test_csv_export_with_week_filter(test_client: TestClient):
    """CSV export with week filter returns matching data."""
    response = test_client.get("/api/export/csv?season=2024&week=1")
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) >= 2


def test_csv_export_no_data(test_client: TestClient):
    """CSV export returns 404 when no matching data exists."""
    response = test_client.get("/api/export/csv?season=1900")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_json_export_no_data(test_client: TestClient):
    """JSON export returns 404 when no matching data exists."""
    response = test_client.get("/api/export/json?season=1900")
    assert response.status_code == 404
    data = response.json()
    assert "error" in data


def test_csv_export_single_game(test_client: TestClient):
    """CSV export for a single game_id returns one data row."""
    response = test_client.get("/api/export/csv?game_id=2024_W01_BUF@KC")
    assert response.status_code == 200
    lines = response.text.strip().split("\n")
    assert len(lines) == 2  # header + 1 row
    assert "2024_W01_BUF@KC" in response.text


def test_json_export_single_game(test_client: TestClient):
    """JSON export for a single game_id returns one-element array."""
    response = test_client.get("/api/export/json?game_id=2024_W01_BUF@KC")
    assert response.status_code == 200
    data = json.loads(response.text)
    assert len(data) == 1
    assert data[0]["game_id"] == "2024_W01_BUF@KC"


def test_backtest_export(test_client: TestClient):
    """Backtest CSV export returns backtest predictions."""
    response = test_client.get("/api/export/csv?type=backtest")
    assert response.status_code == 200
    assert "game_id" in response.text
    assert "model_prob" in response.text


def test_backtest_json_export(test_client: TestClient):
    """Backtest JSON export returns backtest predictions."""
    response = test_client.get("/api/export/json?type=backtest")
    assert response.status_code == 200
    data = json.loads(response.text)
    assert isinstance(data, list)
    assert len(data) >= 1


def test_csv_content_disposition_filename(test_client: TestClient):
    """Content-Disposition header has appropriate filename."""
    response = test_client.get("/api/export/csv")
    disposition = response.headers["Content-Disposition"]
    assert "nfl_predictions" in disposition
    assert ".csv" in disposition


def test_json_content_disposition_filename(test_client: TestClient):
    """JSON Content-Disposition header has appropriate filename."""
    response = test_client.get("/api/export/json")
    disposition = response.headers["Content-Disposition"]
    assert "nfl_export.json" in disposition


# ---------------------------------------------------------------------------
# The bets export (plan 31-16 Task 3, D31-32, SPEC R8, UI-SPEC E8)
# ---------------------------------------------------------------------------
#
# The legacy ``recommendations_{season}_week{week}.json`` is retired in plan 31-17. THIS is what
# replaces it, so a week's record does not become a binary DuckDB file nobody can read. The claim
# under test is that an export and a screenshot of the page cannot disagree: the export reads the
# SAME two service getters ``/bets`` reads, in the same order, and returns BOTH halves of the
# candidate universe with their honesty labels attached.

_BETS_SEASON = 2023
_BETS_WEEK = 1
_BETS_SNAPSHOT_TS = "2023-09-07T18:00:00-04:00"
_BETS_FREEZE_TS = "2023-09-08T18:00:00-04:00"
_BETS_CROSS_LINK = "See the backtest evidence behind these bets"


def _bets_row(
    game_id: str,
    target: str,
    *,
    status: str,
    rejection_reason: str | None = None,
    per_bet_ev: float | None = None,
    validation_type: str = "contaminated",
) -> dict[str, Any]:
    """One ``bet_list`` row carrying every column the export is asserted to surface."""
    row = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": game_id,
            "season": _BETS_SEASON,
            "week": _BETS_WEEK,
            "target": target,
            "bet_side": "under",
            "line": 47.5,
            "per_bet_ev": per_bet_ev,
            "stake_units": 1.5 if per_bet_ev is not None else None,
            "ev_tier": "high" if per_bet_ev is not None else None,
            "status": status,
            "rejection_reason": rejection_reason,
            "snapshot_ts": _BETS_SNAPSHOT_TS,
            "freeze_ts": _BETS_FREEZE_TS,
            "selected_odds": -110.0,
            "flat_stake": 1.0,
            "provenance": "backtest_replay",
            "validation_type": validation_type,
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
        }
    )
    return row


# Three live rows with DISTINCT EVs (so the ranked order is unambiguous) and two suppressed ones.
_BETS_FIXTURE_ROWS: list[dict[str, Any]] = [
    _bets_row("2023_W01_DET@KC", "ou", status="live", per_bet_ev=0.0625),
    _bets_row("2023_W01_CAR@ATL", "ou", status="live", per_bet_ev=0.0410),
    _bets_row(
        "2023_W01_CIN@CLE",
        "ou",
        status="live",
        per_bet_ev=0.0325,
        validation_type="clean_holdout",
    ),
    _bets_row(
        "2023_W01_JAX@IND",
        "wp",
        status="suppressed",
        rejection_reason="ev_below_floor",
    ),
    _bets_row(
        "2023_W01_DET@KC",
        "ats",
        status="suppressed",
        rejection_reason="missing_snapshot",
    ),
]


@pytest.fixture()
def bets_export_client(tmp_path: Path) -> Iterator[TestClient]:
    """A TestClient serving BOTH ``/bets`` and ``/api/export`` from one bet-list cache.

    One client for both surfaces on purpose: the assertion being made is that the two agree, and
    two clients over two databases could only ever compare two fixtures against each other.
    """
    clear_cache()
    db_path = tmp_path / "bets_export_cache.duckdb"
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        materialize_bet_list(conn, pd.DataFrame(_BETS_FIXTURE_ROWS))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": row["game_id"],
                        "season": _BETS_SEASON,
                        "week": _BETS_WEEK,
                    }
                    for row in _BETS_FIXTURE_ROWS
                ]
            ),
        )
        stamped_at = datetime(2023, 9, 8, 22, 30, 0)
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            [
                [
                    BET_LIST_POPULATED_AT_KEY,
                    "2023-09-08T22:30:00+00:00",
                    stamped_at,
                ],
                ["last_updated", stamped_at.isoformat(), stamped_at],
            ],
        )
    finally:
        conn.close()

    from api.dependencies import get_db
    from api.main import app

    served = duckdb.connect(str(db_path), read_only=True)
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = lambda: served
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        served.close()


def _bets_query() -> str:
    return f"type=bets&season={_BETS_SEASON}&week={_BETS_WEEK}"


def _page_row_identifiers(body: str) -> list[tuple[str, str]]:
    """Every (game_id, bet-type label) the page renders, in document order.

    Both tables are read: the live list and the collapsed suppressed disclosure, whose rows are in
    the document even while closed. That is the set an export must equal.
    """
    return [
        (match.group(1), match.group(2))
        for match in re.finditer(
            r'<a href="/games/([^"]+)"[^>]*>[^<]*</a>\s*</td>\s*'
            r'<td class="px-4 py-3 text-left text-gray-700 whitespace-nowrap">([^<]+)</td>',
            body,
        )
    ]


def test_both_handlers_accept_the_bets_type_and_name_the_season_and_week(
    bets_export_client: TestClient,
) -> None:
    """CSV and JSON both stream an attachment whose filename names the season and the week."""
    for suffix, media_type in (("csv", "text/csv"), ("json", "application/json")):
        response = bets_export_client.get(f"/api/export/{suffix}?{_bets_query()}")
        assert response.status_code == 200, f"{suffix} export failed"
        assert response.headers["content-type"].startswith(media_type)
        disposition = response.headers["Content-Disposition"]
        assert "attachment" in disposition
        assert f"bets_{_BETS_SEASON}_week{_BETS_WEEK}.{suffix}" in disposition


def test_the_exported_row_set_equals_the_pages_live_and_suppressed_rows(
    bets_export_client: TestClient,
) -> None:
    """The export is the union of both halves of the page, compared identifier by identifier.

    Not "roughly the same rows": the (game_id, target) pairs of the export must equal the pairs
    the page rendered for that week, in both directions. A suppressed row dropped from either
    surface fails here.
    """
    exported = json.loads(
        bets_export_client.get(f"/api/export/json?{_bets_query()}").text
    )
    page = bets_export_client.get(f"/bets?season={_BETS_SEASON}&week={_BETS_WEEK}").text

    exported_ids = {(row["game_id"], row["target"]) for row in exported}
    labels = {"Winner": "wp", "Spread": "ats", "Totals": "ou"}
    page_ids = {
        (game_id, labels[label]) for game_id, label in _page_row_identifiers(page)
    }

    assert page_ids, "the page rendered no identifiable rows"
    assert exported_ids == page_ids, (
        f"export-only={exported_ids - page_ids}, page-only={page_ids - exported_ids}"
    )
    assert len(exported) == len(_BETS_FIXTURE_ROWS)


def test_every_exported_row_carries_its_status_and_honesty_labels(
    bets_export_client: TestClient,
) -> None:
    """status, provenance and validation_type on EVERY row; a reason on every non-live one.

    This is what makes R8's in-every-exported-figure clause mean something, and what stops a
    suppressed candidate being read as a bet that was placed (D31-32).
    """
    exported = json.loads(
        bets_export_client.get(f"/api/export/json?{_bets_query()}").text
    )
    assert exported

    for row in exported:
        for column in ("status", "provenance", "validation_type"):
            assert row.get(column), (
                f"exported row {row['game_id']}/{row['target']} has no {column}"
            )
        if row["status"] != "live":
            assert row.get("rejection_reason"), (
                f"suppressed row {row['game_id']}/{row['target']} carries no reason"
            )

    assert {row["status"] for row in exported} == {"live", "suppressed"}, (
        "the export did not return both halves of the candidate universe"
    )
    # The CSV carries the same four columns as named header fields.
    header = (
        bets_export_client.get(f"/api/export/csv?{_bets_query()}")
        .text.strip()
        .split("\n")[0]
    )
    for column in ("status", "rejection_reason", "provenance", "validation_type"):
        assert column in header, f"the CSV header omits {column}"


def test_exported_row_order_equals_the_pages_order_position_by_position(
    bets_export_client: TestClient,
) -> None:
    """Position by position, not as a set: the export reproduces the page's reading order.

    Live rows first, ranked per-bet EV descending under the four-key tie-break, then the
    suppressed rows grouped by reason -- the two getters' own orders, concatenated.
    """
    exported = json.loads(
        bets_export_client.get(f"/api/export/json?{_bets_query()}").text
    )
    page = bets_export_client.get(f"/bets?season={_BETS_SEASON}&week={_BETS_WEEK}").text

    labels = {"Winner": "wp", "Spread": "ats", "Totals": "ou"}
    page_order = [
        (game_id, labels[label]) for game_id, label in _page_row_identifiers(page)
    ]
    exported_order = [(row["game_id"], row["target"]) for row in exported]

    assert exported_order == page_order, (
        f"export order {exported_order} does not match page order {page_order}"
    )
    # And the live half really is EV-descending, so the comparison above is not two identical
    # arbitrary orders agreeing by accident.
    live_evs = [row["per_bet_ev"] for row in exported if row["status"] == "live"]
    assert live_evs == sorted(live_evs, reverse=True)


def test_the_bets_export_adds_no_route(bets_export_client: TestClient) -> None:
    """The export route table is exactly the two shipped paths -- the branch is a parameter."""
    from api.main import app

    export_paths = {
        route.path  # pyright: ignore[reportAttributeAccessIssue]
        for route in app.routes
        if getattr(route, "path", "").startswith("/api/export")
    }
    assert export_paths == {"/api/export/csv", "/api/export/json"}, (
        f"a new export route was added: {sorted(export_paths)}"
    )


def test_the_export_buttons_partial_is_reused_byte_for_byte() -> None:
    """D31-32 reuses ``_export_buttons.html`` VERBATIM; no class inside it is edited."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "web/templates/components/_export_buttons.html",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"the export buttons partial was modified:\n{result.stdout}"
    )


def test_the_page_offers_both_exports_and_the_cross_link(
    bets_export_client: TestClient,
) -> None:
    """The trailing utilities row carries the reused buttons and the authored cross-link CTA."""
    body = bets_export_client.get(f"/bets?season={_BETS_SEASON}&week={_BETS_WEEK}").text

    assert (
        f'href="/api/export/csv?type=bets&amp;season={_BETS_SEASON}&amp;week={_BETS_WEEK}"'
        in body
    )
    assert (
        f'href="/api/export/json?type=bets&amp;season={_BETS_SEASON}&amp;week={_BETS_WEEK}"'
        in body
    )
    # The reused partial's 44-pixel minimum touch height came along with it.
    assert 'id="export-buttons"' in body
    assert body.count("min-h-[44px]") >= 2
    assert _BETS_CROSS_LINK in body
    assert 'href="/betting"' in body


def test_the_export_links_follow_the_week_the_page_is_showing(
    bets_export_client: TestClient,
) -> None:
    """The buttons sit INSIDE the swap target, so a week change moves them with the rows.

    Outside it they would keep pointing at the week the page first loaded with, handing the reader
    a file for a week other than the one on screen.
    """
    fragment = bets_export_client.get(
        f"/bets?season={_BETS_SEASON}&week={_BETS_WEEK}",
        headers={"HX-Request": "true"},
    ).text
    assert (
        f'href="/api/export/csv?type=bets&amp;season={_BETS_SEASON}&amp;week={_BETS_WEEK}"'
        in fragment
    ), "the export links are outside the week swap target"
    assert _BETS_CROSS_LINK in fragment


def test_an_unknown_week_exports_nothing_rather_than_the_whole_cache(
    bets_export_client: TestClient,
) -> None:
    """A week with no rows returns the shipped 404 shape, not a silent full-cache dump."""
    for suffix in ("csv", "json"):
        response = bets_export_client.get(
            f"/api/export/{suffix}?type=bets&season={_BETS_SEASON}&week=17"
        )
        assert response.status_code == 404
        assert "error" in response.json()
