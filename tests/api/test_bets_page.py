"""End-to-end tracer proof for the Phase-31 bet list (plan 31-01, PROD-02, SPEC R5).

This module is the tracer's evidence. It wires ONE path all the way through every layer the
phase touches and asserts the values did not change on the way:

    BetSelector.select  ->  BET_LIST_COLUMNS rows  ->  materialize_bet_list  ->  bet_list table
                        ->  DataService.get_bet_list  ->  GET /bets  ->  rendered HTML

The claim it defends is SERVED-EQUALS-SELECTOR: the EV and the unit stake rendered on the page
are the numbers the Phase-27 selector produced, at the rendered precision, in the pre-registered
D31-28 order. A page that renders plausible-but-recomputed numbers passes a smoke test and fails
this one.

The selector is the REAL ``backtest.bet_selector.BetSelector`` over a historical O/U week -- not a
stub -- so the seam being proven is the production seam. The candidate rows are authored (fixed
model/closing/actual totals) so the assertion is deterministic and hermetic; nothing under
``data/``, ``artifacts/`` or ``outputs/`` is read.

Selectors (``-k``): served_equals_selector, served_order, tie_break, not_advice_banner,
no_currency, nav_link, ev_band_badge, empty_week.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.cache import (
    BET_LIST_COLUMNS,
    CACHE_SCHEMA,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_WIN,
    classify_row_provenance,
    materialize_available_bet_weeks,
    materialize_bet_list,
)
from backtest.bet_selector import BetSelector
from backtest.ev_chain_constants import assign_ev_tier
from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD

# ---------------------------------------------------------------------------
# Fixture constants (kept explicit so a silent drift is caught)
# ---------------------------------------------------------------------------

_SEASON = 2023
_WEEK = 1
_EMPTY_WEEK = 2  # scheduled, but no candidate cleared the floor
_BANKROLL = 10_000.0
_UNIT = _BANKROLL * 0.01  # 1 unit = 1% of a notional bankroll (D27-09)
_EV_FLOOR_T = 0.0
_FROZEN_SD = 13.0
_SEASON_BIAS = {_SEASON: -1.0}
_SNAPSHOT_TS = "2023-09-07T18:00:00-04:00"
_FREEZE_TS = "2023-09-08T18:00:00-04:00"
_MINUS_110 = -110.0

# One historical O/U week. Three of the four candidates clear the sub-pop UNION and the EV floor;
# the fourth (a low-total over) is rejected as ``not_subpop``.
_CANDIDATES: list[dict[str, Any]] = [
    {
        "game_id": "2023_W01_DET@KC",
        "season": _SEASON,
        "week": _WEEK,
        "model_total": 48.0,
        "closing_total": 53.0,
        "actual": 41.0,
        "sportsbook": "consensus",
        "is_live": False,
    },
    {
        "game_id": "2023_W01_CAR@ATL",
        "season": _SEASON,
        "week": _WEEK,
        "model_total": 39.0,
        "closing_total": 43.5,
        "actual": 34.0,
        "sportsbook": "consensus",
        "is_live": False,
    },
    {
        "game_id": "2023_W01_CIN@CLE",
        "season": _SEASON,
        "week": _WEEK,
        "model_total": 42.0,
        "closing_total": 47.5,
        "actual": 27.0,
        "sportsbook": "consensus",
        "is_live": False,
    },
    {
        "game_id": "2023_W01_JAX@IND",
        "season": _SEASON,
        "week": _WEEK,
        "model_total": 46.0,
        "closing_total": 45.5,
        "actual": 52.0,
        "sportsbook": "consensus",
        "is_live": False,
    },
]


def _selector() -> BetSelector:
    """The REAL Phase-27 O/U selector, constructed exactly as production constructs it."""
    return BetSelector(
        frozen_sd=_FROZEN_SD,
        season_bias_by_season=_SEASON_BIAS,
        ev_floor_t=_EV_FLOOR_T,
        bankroll=_BANKROLL,
        high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
    )


def _grading_status(outcome: bool | None) -> str:
    """Map the selector's push-aware outcome onto the closed four-state vocabulary."""
    if outcome is None:
        return GRADING_STATUS_PENDING
    return GRADING_STATUS_WIN if outcome else GRADING_STATUS_LOSS


def _to_bet_list_row(record: dict[str, Any]) -> dict[str, Any]:
    """Map ONE selector record onto the locked 28-column bet_list schema.

    The only transforms are a UNIT conversion (dollars -> units, 1 unit = 1% of bankroll) and the
    two honesty labels. No metric is re-derived: ``per_bet_ev``, ``calibrated_p_side``, the line
    and the slipped line are carried through unchanged.
    """
    provenance, validation_type = classify_row_provenance(record["season"], "replay")
    row: dict[str, Any] = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": record["game_id"],
            "season": record["season"],
            "week": record["week"],
            "target": "ou",
            "bet_side": record["bet_side"],
            "model_value": record["model_total"],
            "market_value": record["closing_total"],
            "line": record["closing_total"],
            "slipped_line": record["slipped_line"],
            "calibrated_p_side": record["calibrated_p_side"],
            "per_bet_ev": record["per_bet_ev"],
            "stake_units": record["kelly_stake"] / _UNIT,
            "ev_tier": assign_ev_tier(record["per_bet_ev"], _EV_FLOOR_T),
            "status": "live",
            "rejection_reason": None,
            "eligibility_label": record["subpop_label"],
            "snapshot_ts": _SNAPSHOT_TS,
            "freeze_ts": _FREEZE_TS,
            "selected_odds": _MINUS_110,
            "flat_stake": 1.0,
            "provenance": provenance,
            "validation_type": validation_type,
            "grading_status": _grading_status(record["outcome"]),
            "outcome": record["outcome"],
            "clv": record["clv"],
            "payout_flat": None,
            "realized_units": None,
            "graded_at": None,
        }
    )
    return row


@pytest.fixture()
def selected_records() -> list[dict[str, Any]]:
    """The REAL selector output for the historical week, EV-descending with the R5 tie-break."""
    result = _selector().select(_CANDIDATES)
    assert result.selected, (
        "fixture week selected no bets -- the tracer would prove nothing"
    )
    return sorted(
        result.selected,
        key=lambda r: (-r["per_bet_ev"], r["season"], r["week"], r["game_id"], "ou"),
    )


def _build_cache(
    db_path: Path, bet_rows: list[dict[str, Any]], *, week_rows: list[dict[str, Any]]
) -> None:
    """Build a cache DB carrying the bet list and the schedule-derived navigation table."""
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            stmt = statement.strip()
            if stmt:
                conn.execute(stmt)
        materialize_bet_list(conn, pd.DataFrame(bet_rows))
        materialize_available_bet_weeks(conn, pd.DataFrame(week_rows))
    finally:
        conn.close()


def _client(db_path: Path) -> Iterator[TestClient]:
    from api.dependencies import get_db
    from api.main import app

    conn = duckdb.connect(str(db_path), read_only=True)
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = lambda: conn
    client = TestClient(app, raise_server_exceptions=False)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()
        conn.close()


@pytest.fixture()
def bets_client(
    tmp_path: Path, selected_records: list[dict[str, Any]]
) -> Iterator[TestClient]:
    """A TestClient serving /bets from a cache built out of the REAL selector output."""
    db_path = tmp_path / "bets_cache.duckdb"
    _build_cache(
        db_path,
        [_to_bet_list_row(r) for r in selected_records],
        week_rows=[
            # Week 1 carries the selected bets; week 2 is scheduled but has zero admitted
            # bets, which is what makes the zero-bets empty state reachable through the
            # SCHEDULE-derived navigation whitelist rather than through a rejected param.
            *(
                {"game_id": c["game_id"], "season": _SEASON, "week": _WEEK}
                for c in _CANDIDATES
            ),
            {"game_id": "2023_W02_AAA@BBB", "season": _SEASON, "week": _EMPTY_WEEK},
            {"game_id": "2023_W02_CCC@DDD", "season": _SEASON, "week": _EMPTY_WEEK},
        ],
    )
    yield from _client(db_path)


def _rendered_ev_values(body: str) -> list[str]:
    """Every rendered signed two-decimal EV percentage, in document order."""
    return re.findall(r"([+-]\d+\.\d{2})%", body)


def _rendered_matchups(body: str) -> list[str]:
    """Every rendered matchup label, in document order."""
    return re.findall(
        r'<a href="/games/[^"]+" class="[^"]*">([A-Z]{2,3} @ [A-Z]{2,3})</a>', body
    )


# ---------------------------------------------------------------------------
# The tracer assertion: served == selector
# ---------------------------------------------------------------------------


def test_served_equals_selector_top_row(
    bets_client: TestClient, selected_records: list[dict[str, Any]]
) -> None:
    """The top-ranked served EV and stake EQUAL the selector's per_bet_ev and stake.

    This is the whole point of the tracer. The page renders what the selector decided; nothing in
    the request path recomputes an EV or a stake (UIAP-01, SPEC R5).
    """
    response = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}")
    assert response.status_code == 200
    body = response.text

    top = selected_records[0]
    expected_ev = "%+.2f" % (top["per_bet_ev"] * 100)
    expected_stake = "%.2f" % (top["kelly_stake"] / _UNIT)

    assert _rendered_ev_values(body)[0] == expected_ev
    assert f"{expected_ev}%" in body
    assert (
        f'class="px-4 py-3 text-right stat-num whitespace-nowrap">{expected_stake}<'
        in body
    )

    expected_matchup = top["game_id"].split("_")[-1].replace("@", " @ ")
    assert _rendered_matchups(body)[0] == expected_matchup


def test_served_order_is_ev_descending(
    bets_client: TestClient, selected_records: list[dict[str, Any]]
) -> None:
    """Served order is per-bet EV DESCENDING -- the pre-registered D31-28 rank."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    served = [float(v) for v in _rendered_ev_values(body)]
    assert served == sorted(served, reverse=True)
    assert len(served) == len(selected_records)

    expected = ["%+.2f" % (r["per_bet_ev"] * 100) for r in selected_records]
    assert _rendered_ev_values(body) == expected


def test_tie_break_is_four_key(tmp_path: Path) -> None:
    """An EV tie breaks on (season, week, game_id, target), so two requests agree byte-for-byte.

    Constructed rather than sampled: a genuine float tie out of the selector is not reproducible,
    but the tie-break is the contract SPEC R5 pins, so it is exercised directly.
    """
    base = dict.fromkeys(BET_LIST_COLUMNS)
    base.update(
        {
            "season": _SEASON,
            "week": _WEEK,
            "target": "ou",
            "bet_side": "under",
            "line": 44.0,
            "per_bet_ev": 0.0625,
            "stake_units": 1.5,
            "ev_tier": "high",
            "status": "live",
            "snapshot_ts": _SNAPSHOT_TS,
            "freeze_ts": _FREEZE_TS,
            "selected_odds": _MINUS_110,
            "flat_stake": 1.0,
            "provenance": "backtest_replay",
            "validation_type": "contaminated",
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
        }
    )
    rows = [
        {**base, "game_id": "2023_W01_ZZZ@AAA"},
        {**base, "game_id": "2023_W01_AAA@ZZZ"},
    ]
    db_path = tmp_path / "tie_cache.duckdb"
    _build_cache(
        db_path,
        rows,
        week_rows=[
            {"game_id": r["game_id"], "season": _SEASON, "week": _WEEK} for r in rows
        ],
    )

    first: list[str] = []
    second: list[str] = []
    with contextmanager(_client)(db_path) as client:
        first = _rendered_matchups(
            client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
        )
        second = _rendered_matchups(
            client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
        )

    assert first == second, "two requests rendered different order for a tied EV"
    assert first == ["AAA @ ZZZ", "ZZZ @ AAA"], "tie did not break on game_id ascending"


# ---------------------------------------------------------------------------
# Honesty surface: the banner, the units-not-currency rule, the badge, the nav
# ---------------------------------------------------------------------------


def test_not_advice_banner_present(bets_client: TestClient) -> None:
    """The non-dismissable not-advice banner renders on the page (SPEC prohibition)."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert "Not wagering advice" in body
    assert "1 unit = 1% of a notional bankroll" in body


def test_not_advice_banner_present_in_empty_week(bets_client: TestClient) -> None:
    """The banner survives the empty state -- it sits outside every conditional branch (UI E1)."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_EMPTY_WEEK}").text
    assert "Not wagering advice" in body


def test_no_currency_symbol_anywhere(bets_client: TestClient) -> None:
    """No currency amount appears anywhere in the response: stakes are UNITS only."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert "$" not in body
    assert "Stake (units)" in body


def test_ev_band_badge_is_monochrome(bets_client: TestClient) -> None:
    """The EV band badge renders with the monochrome ramp, never the green/amber/red one."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert "EV band" in body
    assert "bg-gray-200 text-gray-900 border border-gray-300" in body
    for forbidden in ("bg-green-100", "bg-amber-100", "bg-red-100"):
        assert forbidden not in body, (
            f"/bets rendered the confidence-badge colour {forbidden}"
        )


def test_nav_carries_the_seventh_item(bets_client: TestClient) -> None:
    """The seventh nav item reaches /bets from both the desktop and the mobile list."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert body.count('href="/bets"') == 2
    assert ">Bets</a>" in body


def test_raw_target_codes_are_never_rendered(bets_client: TestClient) -> None:
    """Bet type renders Winner / Spread / Totals, never the raw wp / ats / ou codes."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert ">Totals</td>" in body


def test_empty_week_renders_empty_state_not_500(bets_client: TestClient) -> None:
    """A week with no admitted bets renders the explicit empty state, never a blank page or a 500."""
    response = bets_client.get(f"/bets?season={_SEASON}&week={_EMPTY_WEEK}")
    assert response.status_code == 200
    assert "No bets cleared the floor this week" in response.text
