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

Plan 31-15 extends it past the happy path: the suppressed-candidates disclosure (the declined
half of the SAME candidate universe), the four distinct non-happy renders, and the parameterised
week selector shared with the This Week page.

Plan 31-16 extends it again: the realized-versus-expected tracker rendered as one section per
honesty class, the provenance badge that labels them, and the /betting cross-link.

Selectors (``-k``): served_equals_selector, served_order, tie_break, not_advice_banner,
no_currency, nav_link, ev_band_badge, empty_week, taxonomy, reason_code, suppressed, disclosure,
caption, moneyline, tracker, pushes, graded, return, provenance, badge, cross_link, timeout,
sync.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
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
    bet_list_populated_at_key,
    classify_row_provenance,
    materialize_available_bet_weeks,
    materialize_bet_list,
    materialize_bet_tracker_blocks,
    materialize_bet_week_freeze,
)
from api.dependencies import templates as app_templates
from api.services import DataService, clear_cache
from backtest.bet_selector import REJECTION_REASONS, BetSelector
from backtest.bet_tracker import (
    EmptyTrackerBlock,
    TrackerBlock,
    aggregate_all_blocks,
    to_tracker_frame,
)
from backtest.ev_chain_constants import assign_ev_tier
from tests.api.week_selector_snapshot import (
    COMPONENTS_DIR,
    FAILURE_EVENTS,
    PRE_PARAM_CONTEXT,
    SELECTOR_OPEN,
    class_values,
    extract_selector,
)

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
_SNAPSHOT_TS_LABEL = "Sep 7, 2023, 6:00 PM ET"
_FREEZE_TS = "2023-09-08T18:00:00-04:00"
_MINUS_110 = -110.0

# One historical O/U week. Three of the four candidates clear the EV floor. The fourth, a low-total
# over, is the case the deleted sub-pop UNION used to refuse as ``not_subpop``; since D33.2-24 it
# is a candidate like the rest, reaches the EV floor, and is rejected there (``ev_below_floor``).
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


def _class_attributes(markup: str) -> list[str]:
    """Every class attribute value inside *markup*.

    Hue assertions are scoped to CLASS VALUES rather than to the whole markup: the word "measured"
    contains "red" and "expected" contains no hue at all, so a substring search over authored prose
    is a coin flip rather than a check.
    """
    return re.findall(r'class="([^"]*)"', markup)


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
    assert f"data-slip-stake>{expected_stake}<" in body

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
    assert re.search(
        r'<span class="[^"]*\bband-high\b[^"]*"[^>]*title="EV band high', body
    ), "the high EV band did not render with the monochrome band-high style"
    for forbidden in ("bg-green-100", "bg-amber-100", "bg-red-100"):
        assert forbidden not in body, (
            f"/bets rendered the confidence-badge colour {forbidden}"
        )


def test_nav_carries_the_bets_item(bets_client: TestClient) -> None:
    """The Bets nav item reaches /bets from both the desktop row and the mobile panel."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert body.count('href="/bets"') == 2
    # The label sits in the skewed link's .unskew child (spec 10).
    assert '<span class="unskew">Bets</span></a>' in body


def test_raw_target_codes_are_never_rendered(bets_client: TestClient) -> None:
    """Bet type renders Winner / Spread / Totals, never the raw wp / ats / ou codes."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert "data-bet-type>Totals<" in body
    assert "data-bet-type>ou<" not in body


def test_empty_week_renders_empty_state_not_500(bets_client: TestClient) -> None:
    """A week with no admitted bets renders the explicit empty state, never a blank page or a 500."""
    response = bets_client.get(f"/bets?season={_SEASON}&week={_EMPTY_WEEK}")
    assert response.status_code == 200
    assert "No bets cleared the floor this week" in response.text


# ---------------------------------------------------------------------------
# The suppressed-candidates disclosure (plan 31-15 Task 1, SPEC R6, UI-SPEC E3)
# ---------------------------------------------------------------------------

_UNRECOGNISED_REASON = "reason_invented_by_a_future_plan"

# The thirteen labels the page maps REJECTION_REASONS onto, transcribed from the UI-SPEC's
# suppression-reason table. Kept here as a LITERAL rather than imported from the template so a
# silent edit to either side is a test failure rather than a tautology.
_EXPECTED_LABELS: dict[str, str] = {
    "ev_below_floor": "Expected value below the floor",
    "not_subpop": "Outside the eligible sub-population",
    "stale_line": "Line captured after the lock, or not timed",
    "missing_snapshot": "No market line for this bet type",
    "missing_prediction": "No model prediction for this game",
    "real_odds_failed": "Odds failed the real-market check",
    "zero_kelly_stake": "Sizing returned no stake",
    "ev_not_finite": "Expected value could not be computed",
    "no_bet_side": "Model agrees with the market",
    "no_honest_ev_floor": "No honest threshold for this bet type",
    # Plan 33.2-26: the second test of a 2026 win bet, and a bet type with no honest threshold.
    "edge_below_threshold": "Win edge over the spread line below the threshold",
    "no_honest_edge_threshold": "No honest edge threshold for this bet type",
    # A33.2-review IN-06: no converter bound, told apart from a market-data gap.
    "no_bound_converter": "No spread converter bound to the blend",
}

# The reason cell is located by its data hook rather than by a class string, so a restyle cannot
# turn this "must not appear" check vacuous by changing the classes it spelled.
_BLANK_REASON_CELL = "data-reason-cell></td>"


def _suppressed_row(
    game_id: str, target: str, reason: str | None, *, week: int = _WEEK
) -> dict[str, Any]:
    """One SUPPRESSED bet_list row: the core record only.

    A suppressed candidate is never priced (plan 31-09), so ``per_bet_ev``, ``stake_units`` and
    ``ev_tier`` stay NULL here exactly as the selector leaves them. If the page derived anything
    from those columns this row would render a blank or raise; it renders the reason instead.
    """
    row = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": game_id,
            "season": _SEASON,
            "week": week,
            "target": target,
            "status": "suppressed",
            "rejection_reason": reason,
            "snapshot_ts": _SNAPSHOT_TS,
            "freeze_ts": _FREEZE_TS,
            "provenance": "backtest_replay",
            "validation_type": "contaminated",
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
        }
    )
    return row


def _live_row(game_id: str, target: str, *, week: int = _WEEK) -> dict[str, Any]:
    """One LIVE bet_list row carrying the minimum the live table renders."""
    row = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": game_id,
            "season": _SEASON,
            "week": week,
            "target": target,
            "bet_side": "under",
            "line": 47.5,
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
    return row


def _client_for(tmp_path: Path, rows: list[dict[str, Any]], name: str) -> Any:
    """Build a one-week cache out of *rows* and return a client context manager serving it.

    ``clear_cache()`` is called here, not left to the autouse conftest fixture: that fixture runs
    once per TEST, and a test that serves TWO different cache DBs would otherwise have the first
    DB's rows answered out of the module-level TTLCache for the second one. That is not a
    hypothetical -- it silently emptied the suppressed section under a second client.
    """
    clear_cache()
    db_path = tmp_path / f"{name}.duckdb"
    _build_cache(
        db_path,
        rows,
        week_rows=[
            {"game_id": r["game_id"], "season": _SEASON, "week": r["week"]}
            for r in rows
        ],
    )
    return contextmanager(_client)(db_path)


def test_every_live_taxonomy_reason_has_a_label(tmp_path: Path) -> None:
    """Every member of the LIVE REJECTION_REASONS tuple renders a label, none renders blank.

    Iterates the EXPORTED taxonomy rather than a literal count: ``no_bet_side`` was added as a
    NINTH member in plan 31-10 after the design contract's header said eight, and a hardcoded
    count would have passed while the newest reason rendered as an empty cell.
    """
    rows = [
        _suppressed_row(f"2023_W01_A{i:02d}@B{i:02d}", "ou", reason)
        for i, reason in enumerate(REJECTION_REASONS)
    ]
    with _client_for(tmp_path, rows, "taxonomy") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert set(REJECTION_REASONS) == set(_EXPECTED_LABELS), (
        "the exported taxonomy and the page's label vocabulary have diverged: "
        f"taxonomy-only={set(REJECTION_REASONS) - set(_EXPECTED_LABELS)}, "
        f"labels-only={set(_EXPECTED_LABELS) - set(REJECTION_REASONS)}"
    )
    for reason in REJECTION_REASONS:
        label = _EXPECTED_LABELS[reason]
        assert label in body, f"reason {reason} rendered no label on the page"
        # The RAW code must not leak into a rendered reason cell when a label exists.
        assert f">{reason}</td>" not in body, (
            f"reason {reason} rendered its raw code even though it has a label"
        )
    assert f"Suppressed candidates ({len(REJECTION_REASONS)})" in body
    assert "data-reason-cell>" in body, (
        "no reason cell rendered, so the blank check is vacuous"
    )
    assert _BLANK_REASON_CELL not in body


def test_unrecognised_reason_code_renders_the_raw_code(tmp_path: Path) -> None:
    """A reason code with no label renders the RAW CODE, never an empty cell (T-31-76)."""
    rows = [_suppressed_row("2023_W01_XXX@YYY", "ou", _UNRECOGNISED_REASON)]
    with _client_for(tmp_path, rows, "unknown_reason") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert f">{_UNRECOGNISED_REASON}</td>" in body
    assert f"{_UNRECOGNISED_REASON} (1)" in body
    assert "data-reason-cell>" in body, (
        "no reason cell rendered, so the blank check is vacuous"
    )
    assert _BLANK_REASON_CELL not in body


def test_null_reason_code_renders_an_explicit_marker(tmp_path: Path) -> None:
    """A suppressed row carrying NO reason renders a visible marker, not a blank and not a 500.

    Grouping a NULL alongside strings would raise a TypeError and 500 the page; showing it as a
    blank would hide a data defect. It is shown as the defect it is.
    """
    rows = [_suppressed_row("2023_W01_XXX@YYY", "ou", None)]
    with _client_for(tmp_path, rows, "null_reason") as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    assert "(no reason recorded)" in response.text


def test_zero_suppressed_rows_renders_the_line_and_no_disclosure(
    bets_client: TestClient,
) -> None:
    """A week with zero suppressed rows gets the header plus one line -- never an empty details."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert "Suppressed candidates (0)" in body
    assert "No candidate was suppressed this week." in body
    # The condensed honesty notes are <details> too; only the suppressed disclosure is absent.
    assert '<details id="suppressed-candidates"' not in body


def test_the_disclosure_is_collapsed_by_default(tmp_path: Path) -> None:
    """The disclosure is a NATIVE details element carrying no open attribute (D31-28)."""
    rows = [_suppressed_row("2023_W01_XXX@YYY", "ou", "ev_below_floor")]
    with _client_for(tmp_path, rows, "collapsed") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert '<details id="suppressed-candidates">' in body
    assert "<details open" not in body
    assert "<summary" in body


def test_suppressed_rows_are_in_the_document_while_collapsed(tmp_path: Path) -> None:
    """The rows are in the RAW HTML even though the disclosure is collapsed (T-31-75).

    This is the whole reason native disclosure was chosen over a scripted toggle: the record
    survives into the document and into an HTML export for a reader who never expands it.
    """
    rows = [_suppressed_row("2023_W01_SEA@SFO", "ats", "stale_line")]
    with _client_for(tmp_path, rows, "in_document") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert "<details open" not in body, "the disclosure was not collapsed"
    assert "SEA @ SFO" in body
    assert "Line captured after the lock, or not timed" in body
    # The suppressed table's "Line as of" cell reads as Eastern time, stored text on hover.
    assert f'title="{_SNAPSHOT_TS}">{_SNAPSHOT_TS_LABEL}</time>' in body


def test_the_caption_is_present_whether_or_not_the_disclosure_is_expanded(
    tmp_path: Path, bets_client: TestClient
) -> None:
    """The caption sits inside <summary>, so it renders collapsed, expanded and in the zero state."""
    caption = (
        "Every scheduled game is evaluated for all three bet types. This section is the "
        "complete record of what was not bet, and why."
    )
    assert caption in bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    rows = [_suppressed_row("2023_W01_XXX@YYY", "ou", "not_subpop")]
    with _client_for(tmp_path, rows, "caption") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert caption in body
    start = body.index('<details id="suppressed-candidates"')
    summary = body[body.index("<summary", start) : body.index("</summary>", start)]
    assert caption in summary, "the caption is hidden while the disclosure is collapsed"


def test_total_but_no_moneyline_yields_one_live_and_one_suppressed_row(
    tmp_path: Path,
) -> None:
    """A game with a total but no moneyline: a LIVE Totals row AND a SUPPRESSED Winner row.

    Both in the same response, on the same game identifier. That is the R6 per-target rule, not a
    duplicated row (UI-SPEC E2 partial / E3 partial).
    """
    game_id = "2023_W01_DEN@LVR"
    rows = [
        _live_row(game_id, "ou"),
        _suppressed_row(game_id, "wp", "missing_snapshot"),
    ]
    with _client_for(tmp_path, rows, "partial_market") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert body.count(f'href="/games/{game_id}"') == 2, (
        "the same game must appear once in the live list and once in the disclosure"
    )
    assert "data-bet-type>Totals<" in body
    assert "data-bet-type>Winner<" in body
    assert "No market line for this bet type" in body
    assert "Suppressed candidates (1)" in body


def _ats_live_row(game_id: str, bet_side: str, line: float) -> dict[str, Any]:
    """One LIVE spread row; ``line`` is the stored HOME MARGIN (positive = home favoured)."""
    row = _live_row(game_id, "ats")
    row.update({"bet_side": bet_side, "line": line})
    return row


def test_a_spread_pick_shows_the_picked_teams_own_line(tmp_path: Path) -> None:
    """33.2 review C2 CR-03: a spread pick reads the line from the PICKED side.

    The stored ``line`` is the home margin, positive when the home team is favoured. The real
    2025 week 1 rows this pins: CIN@CLE home_cover at -5.5 (CIN favoured, CLE the 5.5-point dog)
    rendered "Home_cover -5.5", which reads as CLE laying 5.5 -- the opposite line. It must read
    CLE +5.5. A home favourite (MIA@IND at +1.5) is IND -1.5; the away sides keep the margin's
    sign (DAL@PHI at +8.5 is DAL +8.5, TB@ATL at -1.5 is TB -1.5).
    """
    rows = [
        _ats_live_row("2023_W01_CIN@CLE", "home_cover", -5.5),
        _ats_live_row("2023_W01_MIA@IND", "home_cover", 1.5),
        _ats_live_row("2023_W01_DAL@PHI", "away_cover", 8.5),
        _ats_live_row("2023_W01_TB@ATL", "away_cover", -1.5),
    ]
    with _client_for(tmp_path, rows, "ats_side_line") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    for pick in ("CLE +5.5", "IND -1.5", "DAL +8.5", "TB -1.5"):
        assert pick in body, f"the spread pick {pick!r} is not rendered"
    for wrong in ("CLE -5.5", "IND +1.5", "Home_cover", "Away_cover", "home_cover"):
        assert wrong not in body, f"the page renders {wrong!r}"
    # The slip's Line figure carries the same side-perspective number as the pick.
    assert "data-slip-line>+5.5<" in body
    assert "data-slip-line>-1.5<" in body


def test_each_live_bet_renders_one_ranked_slip(
    bets_client: TestClient, selected_records: list[dict[str, Any]]
) -> None:
    """Redesign: one bet slip per live bet, carrying both teams and its line capture time.

    The slip replaces the ten-column table. The per-row "Line as of" the table showed in its own
    column is now visible text on the slip, so the reader can still check each row's capture time
    against the per-game lock rule stated at the foot of the page.
    """
    from api.presentation import team_nickname

    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    count = len(selected_records)

    assert body.count('<li class="bet-slip') == count
    # Read as Eastern time, with the exact stored text kept on hover and machine-readable.
    assert f'title="{_SNAPSHOT_TS}">{_SNAPSHOT_TS_LABEL}</time>' in body
    assert f'data-slip-asof datetime="{_SNAPSHOT_TS}"' in body
    assert f"data-slip-asof>{_SNAPSHOT_TS}<" not in body
    assert "Line as of</span>" in body
    # The header meta, with its row count -- not the <ol>'s aria-label, which also says
    # "ranked by expected value" and would satisfy a bare substring check on its own.
    noun = "bet" if count == 1 else "bets"
    assert f"ranked by expected value &middot; {count} {noun}</span>" in body
    for record in selected_records:
        away, home = record["game_id"].split("_")[-1].split("@")
        assert team_nickname(away) in body
        assert team_nickname(home) in body


def test_the_suppressed_summary_counts_each_reason_while_collapsed(
    tmp_path: Path,
) -> None:
    """Redesign: the closed disclosure's summary names each reason with its row count.

    Each count is the length of the SAME group the disclosure body renders under that reason, so
    the summary and the body cannot disagree. It is a count of rows on the page, not a metric.
    """
    rows = [
        _suppressed_row("2023_W01_AAA@BBB", "ou", "ev_below_floor"),
        _suppressed_row("2023_W01_CCC@DDD", "ou", "ev_below_floor"),
        _suppressed_row("2023_W01_EEE@FFF", "ats", "stale_line"),
    ]
    with _client_for(tmp_path, rows, "summary_counts") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    start = body.index('<details id="suppressed-candidates"')
    summary = body[body.index("<summary", start) : body.index("</summary>", start)]
    assert "data-suppressed-reason-counts" in summary
    assert 'Expected value below the floor <b class="num">2</b>' in summary
    assert 'Line captured after the lock, or not timed <b class="num">1</b>' in summary
    # The body's per-reason headings carry the same counts.
    assert "Expected value below the floor (2)" in body
    assert "Line captured after the lock, or not timed (1)" in body


# ---------------------------------------------------------------------------
# The four distinct non-happy renders and the cache stamp
# (plan 31-15 Task 2, SPEC R5, UI-SPEC E2 empty / E7 / E12)
# ---------------------------------------------------------------------------

# The four headings, one per state, transcribed from the design contract. A state that rendered
# another state's heading would pass a "not blank" check and fail this one.
_CACHE_ABSENT_HEADING = "Bet list not built yet"
_ZERO_ADMITTED_HEADING = "No bets cleared the floor this week"
_NO_CURRENT_WEEK_HEADING = "No current week"
_HARD_BLOCK_MESSAGE = (
    "This week&#39;s list is missing -- its locked games have no list in the cache"
)

# The TWO-COMMAND recovery sequence, in the order an operator must run it (plan 31-23,
# G-31-123b). Both non-happy renders that name a recovery must name BOTH, generation first.
# Naming only the population command is the shipped defect: from a cold start that command is a
# pure COPY step over an artifact nothing has produced yet, so it builds an EMPTY bet_list table,
# stamps no per-week marker, and -- because it DOES build the schedule-derived freeze table --
# flips the page into the refusal, whose own recovery text named the same command again.
_GENERATE_COMMAND = "scripts/generate_bet_list.py"
_POPULATE_COMMAND = "scripts/populate_cache.py"
# The clause that states the RELATIONSHIP between the two, asserted separately so a future editor
# cannot shorten the copy back to two bare commands and leave a reader to infer why order matters.
_COPY_ONLY_CLAUSE = "only COPIES what the first produces"

_BANNER_EYEBROW = "Not wagering advice"

# The per-game freeze sentence, and the week-level claims it deliberately does NOT make. A
# week-level "lines were frozen Friday 6 PM ET" is FALSE for every Thursday game (D31-18).
# Authored LITERALLY in the template rather than interpolated, so its apostrophes are not
# HTML-escaped -- unlike the two _error_state.html slots, which pass through {{ }}.
_PER_GAME_FREEZE_SENTENCE = (
    "Each game's line locks at 6:00 PM Eastern the day before its own kickoff "
    "-- a Thursday game locks on the Wednesday, that week's Sunday games on the Saturday."
)
_FORBIDDEN_WEEK_LEVEL_CLAIMS = (
    "before this week",
    "this week's lines were frozen",
    "this week&#39;s lines were frozen",
    "lines for this week were frozen",
    "all lines frozen",
    "lines frozen for the week",
    "lines frozen friday 6 pm et",
)

_POPULATED_AT = "2023-09-08T22:30:00+00:00"
_POPULATED_AT_LABEL = "Sep 8, 2023, 6:30 PM ET"
# Strictly LATER than _POPULATED_AT, so the population run finished BEFORE the freeze moved.
# TZ-AWARE UTC, like every instant ``build_bet_week_schedule`` produces (CR-01). A naive fixture
# here would exercise a shape the real writer never emits, and the ``bet_week_freeze`` column is
# now TIMESTAMPTZ so the stored value carries its zone into the rendered refusal text.
_LATER_FREEZE = datetime(2023, 9, 8, 23, 0, 0, tzinfo=UTC)


def _build_bare_cache(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create every cache table and return the OPEN connection for further writes."""
    conn = duckdb.connect(str(db_path))
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)
    return conn


def _stamp_populated_at(
    conn: duckdb.DuckDBPyConnection,
    value: str,
    *,
    season: int = _SEASON,
    week: int = _WEEK,
) -> None:
    """Stamp the PER-WEEK bet-list populated-at key the way ``populate_cache`` stamps it.

    The key is ``bet_list_populated_at:<season>:<week>`` (plan 31-18, D31-29), not the bare
    prefix: a generic marker advances whenever ANY cache table is repopulated, so a run that
    populated predictions and failed on the bet list would read as fresh. Fixtures stamp the same
    key the production writer stamps, or they would be testing a cache shape production cannot
    produce.

    ``last_updated`` is written alongside because ``populate_cache`` always writes it in the same
    run. ``base.html``'s footer read of it is now GUARDED on the key (DEF-31-16, fixed in plan
    31-18), so a cache_meta carrying rows but no ``last_updated`` renders "Unknown" instead of
    500ing every page -- but production still writes both, so the fixture still writes both.
    """
    stamped_at = datetime(2023, 9, 8, 22, 30, 0)
    conn.executemany(
        "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
        [
            [bet_list_populated_at_key(season, week), value, stamped_at],
            ["last_updated", stamped_at.isoformat(), stamped_at],
        ],
    )


def test_state_one_cache_table_absent(tmp_path: Path) -> None:
    """A cache that predates Phase 31 renders the cache-absent state, and NOT the stamp.

    The distinction is load-bearing: reporting a missing TABLE as "no bets cleared the floor"
    would be a claim about the models made from the absence of a table.

    The recovery assertion below used to accept HALF the sequence -- it pinned only
    ``scripts/populate_cache.py``. That is the copy step, and from the cold state this fixture
    builds it cannot reach a served list (plan 31-23, G-31-123b), so the copy named an action the
    reader could follow to no effect. It now requires both commands, in order, plus the clause
    that says why the order matters. That a command is NAMED is still a weaker claim than that
    FOLLOWING it reaches a served list; the latter is asserted end-to-end in
    ``tests/api/test_cold_start_bet_list_recovery.py``.
    """
    clear_cache()
    db_path = tmp_path / "no_table.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_AAA@BBB", "season": _SEASON, "week": _WEEK}]
            ),
        )
        conn.execute("DROP TABLE bet_list")
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    body = response.text
    assert _CACHE_ABSENT_HEADING in body
    assert _GENERATE_COMMAND in body, (
        "the cache-absent state does not name the command that PRODUCES the rows, so the only "
        "action it offers is a copy step with nothing to copy"
    )
    assert _POPULATE_COMMAND in body, (
        "the cache-absent state does not name the command that loads the rows into the cache"
    )
    assert body.index(_GENERATE_COMMAND) < body.index(_POPULATE_COMMAND), (
        "the cache-absent state names the two recovery commands in the WRONG order; running the "
        "copy step first is exactly the sequence that leaves an empty table and a refusal"
    )
    assert _COPY_ONLY_CLAUSE in body, (
        "the cache-absent state does not state that the second command only copies what the "
        "first produces, so an operator who runs only the copy step cannot explain the empty "
        "table they get"
    )
    assert _ZERO_ADMITTED_HEADING not in body
    assert "Bet list last populated" not in body, (
        "the cache stamp rendered against a cache that has no bet list"
    )
    assert _BANNER_EYEBROW in body


def test_state_two_week_with_zero_admitted_bets(bets_client: TestClient) -> None:
    """A week that admitted nothing renders the first-class result state, with its own copy."""
    response = bets_client.get(f"/bets?season={_SEASON}&week={_EMPTY_WEEK}")

    assert response.status_code == 200
    body = response.text
    assert _ZERO_ADMITTED_HEADING in body
    assert "An empty list is a result, not a failure or an outage." in body
    assert "Every scheduled game was evaluated for all three bet types" in body
    assert _CACHE_ABSENT_HEADING not in body
    assert _NO_CURRENT_WEEK_HEADING not in body
    assert _BANNER_EYEBROW in body


def test_state_three_off_season_no_current_week(tmp_path: Path) -> None:
    """With no scheduled week at all the page renders the no-current-week state, not a blank."""
    clear_cache()
    db_path = tmp_path / "off_season.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        response = client.get("/bets")

    assert response.status_code == 200
    body = response.text
    assert _NO_CURRENT_WEEK_HEADING in body
    assert "Use the week selector to read the list for any completed week." in body
    assert _CACHE_ABSENT_HEADING not in body
    assert _ZERO_ADMITTED_HEADING not in body
    assert _BANNER_EYEBROW in body


def test_state_four_a_week_with_no_list_for_its_locked_games_is_refused(
    tmp_path: Path,
) -> None:
    """Games past their lock with NO list anywhere in the week: the refusal, not an empty week.

    D31-27, judged game by game since 33.2 review C2 CR-02: the week is refused only when its
    locked games have no bet-list row at all. The refusal is a server-rendered 200 -- it is NOT
    the failed-fragment path -- and names the games so the claim can be checked.
    """
    clear_cache()
    db_path = tmp_path / "stale.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}]
            ),
        )
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": "2023_W01_DET@KC",
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": _LATER_FREEZE,
                    }
                ]
            ),
        )
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    body = response.text
    assert _HARD_BLOCK_MESSAGE in body
    assert "bg-red-950" in body, "the refusal did not render through _error_state.html"
    assert "Past weeks below are unaffected and remain readable." in body
    assert "DET @ KC" in body, "the refusal does not name the game that has no list"
    # Both timestamps are repeated in the recovery text so the two claims can be compared.
    assert _POPULATED_AT in body
    assert str(_LATER_FREEZE) in body
    # The list itself is withheld: no bet row and no suppressed disclosure.
    assert "Stake (units)" not in body
    assert "Suppressed candidates" not in body
    assert _BANNER_EYEBROW in body
    # The stamp still renders under the refusal (UI-SPEC E12 error).
    assert "Bet list last populated" in body


def test_the_four_states_render_four_distinct_bodies(
    tmp_path: Path, bets_client: TestClient
) -> None:
    """No two of the four states render the same heading, and none returns a 500 or a blank body."""
    headings = {
        _CACHE_ABSENT_HEADING,
        _ZERO_ADMITTED_HEADING,
        _NO_CURRENT_WEEK_HEADING,
        _HARD_BLOCK_MESSAGE,
    }
    assert len(headings) == 4

    response = bets_client.get(f"/bets?season={_SEASON}&week={_EMPTY_WEEK}")
    assert response.status_code == 200
    assert len(response.text) > 0
    present = {h for h in headings if h in response.text}
    assert present == {_ZERO_ADMITTED_HEADING}, (
        f"the zero-admitted state also rendered another state's heading: {present}"
    )


def test_a_fresh_cache_is_not_blocked(tmp_path: Path) -> None:
    """A cache populated AFTER the week's freeze serves the list -- the block is not always-on.

    Without this control the hard-block test would pass against a branch that fired for every
    week, and the page would be refusing lists that are perfectly current.
    """
    clear_cache()
    db_path = tmp_path / "fresh.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(conn, pd.DataFrame([_live_row("2023_W01_DET@KC", "ou")]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}]
            ),
        )
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": "2023_W01_DET@KC",
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": datetime(2023, 9, 8, 22, 0, 0, tzinfo=UTC),
                    }
                ]
            ),
        )
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert _HARD_BLOCK_MESSAGE not in body
    assert "Stake (units)" in body
    assert "data-missing-games" not in body


# 2026 week 3 under the daily pre-lock run (33.2 review C2 CR-02): a Thursday game locking Wed
# 18:00 ET, a Sunday game locking Sat 18:00 ET and a Monday-night game locking Sun 18:00 ET. The
# daily run builds each game's list at about 17:20 ET on its lock day, so every population finishes
# BEFORE the week's latest lock -- which is what the old week-level comparison could never accept.
_DAILY_WEEK = 3
_THURSDAY_GAME = "2026_W03_BUF@MIA"
_SUNDAY_GAME = "2026_W03_NYJ@NE"
_MONDAY_GAME = "2026_W03_DAL@CHI"
_DAILY_LOCKS = {
    _THURSDAY_GAME: datetime(2026, 9, 23, 22, 0, tzinfo=UTC),  # Wed 18:00 ET
    _SUNDAY_GAME: datetime(2026, 9, 26, 22, 0, tzinfo=UTC),  # Sat 18:00 ET
    _MONDAY_GAME: datetime(2026, 9, 27, 22, 0, tzinfo=UTC),  # Sun 18:00 ET
}


def _daily_week_service(
    tmp_path: Path, name: str, built: list[str]
) -> tuple[DataService, duckdb.DuckDBPyConnection]:
    """A cache for the 2026 week-3 schedule in which only *built* games have bet-list rows."""
    clear_cache()
    db_path = tmp_path / f"{name}.duckdb"
    conn = _build_bare_cache(db_path)
    rows = [
        {**_suppressed_row(game_id, "ou", "ev_below_floor"), "season": 2026}
        for game_id in built
    ]
    for row in rows:
        row["week"] = _DAILY_WEEK
    if rows:
        materialize_bet_list(conn, pd.DataFrame(rows))
    materialize_bet_week_freeze(
        conn,
        pd.DataFrame(
            [
                {
                    "game_id": game_id,
                    "season": 2026,
                    "week": _DAILY_WEEK,
                    "game_freeze_ts": lock,
                }
                for game_id, lock in _DAILY_LOCKS.items()
            ]
        ),
    )
    return DataService(conn), conn


@pytest.mark.parametrize(
    ("name", "built", "now", "missing", "pending"),
    [
        # Saturday 17:30 ET: the Saturday run (17:20) built the Sunday game; Monday's lock is ahead.
        (
            "saturday",
            [_THURSDAY_GAME, _SUNDAY_GAME],
            datetime(2026, 9, 26, 21, 30, tzinfo=UTC),
            [],
            [_MONDAY_GAME],
        ),
        # Sunday 17:30 ET: the Sunday run built the Monday-night game; the week is complete.
        (
            "sunday",
            [_THURSDAY_GAME, _SUNDAY_GAME, _MONDAY_GAME],
            datetime(2026, 9, 27, 21, 30, tzinfo=UTC),
            [],
            [],
        ),
        # The Saturday run FAILED: the Sunday game passed its lock with no list, and is named.
        (
            "saturday_failed",
            [_THURSDAY_GAME],
            datetime(2026, 9, 27, 16, 0, tzinfo=UTC),
            [_SUNDAY_GAME],
            [_MONDAY_GAME],
        ),
    ],
)
def test_a_week_built_before_each_lock_is_shown_game_by_game(
    tmp_path: Path,
    name: str,
    built: list[str],
    now: datetime,
    missing: list[str],
    pending: list[str],
) -> None:
    """33.2 review C2 CR-02: the current week is never withheld for being built before its locks.

    The old verdict compared the week's populated-at marker (17:20 ET on the lock day) against the
    week's LATEST lock (Sunday 18:00 ET for a Monday-night week), so every population during the
    live week read as stale and the week was withheld until after its games were played. Each game
    is now judged against its own lock: built games are shown, a game past its lock with no list is
    named as missing, and a game whose lock is ahead is not evaluated yet.
    """
    from api.routes.pages import _bet_week_coverage

    service, conn = _daily_week_service(tmp_path, name, built)
    try:
        coverage = _bet_week_coverage(service, 2026, _DAILY_WEEK, now=now)
    finally:
        conn.close()

    assert coverage.blocked is False, (
        "a week with built games was withheld; the populations finish before the week's latest "
        "lock by design, so a week-level comparison against it can never pass"
    )
    assert coverage.built is True
    assert coverage.missing_games == missing
    assert coverage.pending_games == pending
    assert coverage.not_evaluated is False


def test_a_game_exactly_at_its_lock_counts_as_locked(tmp_path: Path) -> None:
    """At-lock counts as passed (D33.2-01): a game with no list AT its lock is missing, not pending."""
    from api.routes.pages import _bet_week_coverage

    service, conn = _daily_week_service(tmp_path, "at_lock", [_THURSDAY_GAME])
    try:
        coverage = _bet_week_coverage(
            service, 2026, _DAILY_WEEK, now=_DAILY_LOCKS[_SUNDAY_GAME]
        )
    finally:
        conn.close()

    assert coverage.missing_games == [_SUNDAY_GAME]
    assert coverage.pending_games == [_MONDAY_GAME]


def test_a_partly_built_week_shows_its_rows_and_names_the_rest(tmp_path: Path) -> None:
    """The rendered page for a week built in part: the list, the missing game and the pending one.

    DET@KC has a list, CAR@ATL passed its lock with none (2023), and JAX@IND's lock is far ahead.
    """
    clear_cache()
    db_path = tmp_path / "partial_week.duckdb"
    conn = _build_bare_cache(db_path)
    schedule = [
        ("2023_W01_DET@KC", _LATER_FREEZE),
        ("2023_W01_CAR@ATL", _LATER_FREEZE),
        ("2023_W01_JAX@IND", datetime(2099, 9, 8, 22, 0, tzinfo=UTC)),
    ]
    try:
        materialize_bet_list(conn, pd.DataFrame([_live_row("2023_W01_DET@KC", "ou")]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {"game_id": game_id, "season": _SEASON, "week": _WEEK}
                    for game_id, _lock in schedule
                ]
            ),
        )
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": game_id,
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": lock,
                    }
                    for game_id, lock in schedule
                ]
            ),
        )
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert _HARD_BLOCK_MESSAGE not in body
    assert "Stake (units)" in body, "the built game's list was withheld"
    assert "DET @ KC" in body
    missing = body[body.index("data-missing-games") :]
    assert "CAR @ ATL" in missing[: missing.index("</div>")]
    assert "Not evaluated yet: JAX @ IND." in body
    assert "Suppressed candidates (0)" in body


_PARTIAL_ZERO_ADMITTED_HEADING = (
    "No bets cleared the floor among the games with a list so far"
)


@pytest.mark.parametrize(
    ("other_game", "other_lock"),
    [
        (
            "2023_W01_JAX@IND",
            datetime(2099, 9, 8, 22, 0, tzinfo=UTC),
        ),  # still ahead: pending
        ("2023_W01_CAR@ATL", _LATER_FREEZE),  # past its lock with no list: missing
    ],
    ids=["pending", "missing"],
)
def test_the_empty_heading_is_scoped_to_the_games_checked_in_a_partial_week(
    tmp_path: Path, other_game: str, other_lock: datetime
) -> None:
    """A week built in part, with nothing cleared among the built games, must not say "this week".

    DET@KC has a list (one suppressed candidate, no live bet) and a second game has none, so the
    claim "no bets cleared the floor this week" would cover a game that was never checked.
    """
    clear_cache()
    db_path = tmp_path / "partial_empty.duckdb"
    conn = _build_bare_cache(db_path)
    schedule = [("2023_W01_DET@KC", _LATER_FREEZE), (other_game, other_lock)]
    try:
        materialize_bet_list(
            conn,
            pd.DataFrame([_suppressed_row("2023_W01_DET@KC", "ou", "ev_below_floor")]),
        )
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {"game_id": game_id, "season": _SEASON, "week": _WEEK}
                    for game_id, _lock in schedule
                ]
            ),
        )
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": game_id,
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": lock,
                    }
                    for game_id, lock in schedule
                ]
            ),
        )
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert _PARTIAL_ZERO_ADMITTED_HEADING in body
    assert _ZERO_ADMITTED_HEADING not in body
    assert "Every game with a list so far was checked" in body


def test_the_empty_heading_says_this_week_when_the_whole_week_was_checked(
    bets_client: TestClient,
) -> None:
    body = bets_client.get(f"/bets?season={_SEASON}&week={_EMPTY_WEEK}").text
    assert _ZERO_ADMITTED_HEADING in body
    assert _PARTIAL_ZERO_ADMITTED_HEADING not in body
    assert "Every scheduled game was evaluated" in body


def test_a_week_whose_locks_are_all_ahead_is_not_evaluated_yet(tmp_path: Path) -> None:
    """33.2 review C2 WR-01: a future week is 'not evaluated yet', never the stale-cache refusal."""
    clear_cache()
    db_path = tmp_path / "future_week.duckdb"
    conn = _build_bare_cache(db_path)
    future_lock = datetime(2099, 9, 8, 22, 0, tzinfo=UTC)
    try:
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}]
            ),
        )
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": "2023_W01_DET@KC",
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": future_lock,
                    }
                ]
            ),
        )
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert "Not evaluated yet" in body
    assert _HARD_BLOCK_MESSAGE not in body
    assert _FORWARD_WITHHELD_MESSAGE not in body
    assert _ZERO_ADMITTED_HEADING not in body
    assert "Suppressed candidates" not in body


def test_the_cache_stamp_makes_the_per_game_claim_and_no_week_level_one(
    tmp_path: Path,
) -> None:
    """The stamp states the PER-GAME freeze rule and makes no week-level claim (T-31-77)."""
    clear_cache()
    db_path = tmp_path / "stamp.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(conn, pd.DataFrame([_live_row("2023_W01_DET@KC", "ou")]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}]
            ),
        )
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert (
        f'Bet list last populated: <time datetime="{_POPULATED_AT}" '
        f'title="{_POPULATED_AT}">{_POPULATED_AT_LABEL}</time>.'
    ) in body
    assert _PER_GAME_FREEZE_SENTENCE in body
    lowered = body.lower()
    for claim in _FORBIDDEN_WEEK_LEVEL_CLAIMS:
        assert claim.lower() not in lowered, (
            f"the page makes the week-level freeze claim {claim!r}, which is false for every "
            "Thursday game (D31-18)"
        )


def test_the_stamp_names_a_missing_timestamp_rather_than_interpolating_a_blank(
    tmp_path: Path,
) -> None:
    """A bet_list table with no populated-at stamp renders 'not recorded', never an empty value.

    The design contract assumes the stamp key is always written beside the table; the writer that
    guarantees that lands in Plan 31-18. Until then the page names the absence instead of
    rendering 'Bet list last populated: .' against a null.
    """
    clear_cache()
    db_path = tmp_path / "unstamped.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(conn, pd.DataFrame([_live_row("2023_W01_DET@KC", "ou")]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}]
            ),
        )
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert "Bet list last populated: not recorded." in body
    assert "Bet list last populated: ." not in body


def test_no_state_renders_an_empty_main_region(
    tmp_path: Path, bets_client: TestClient
) -> None:
    """Every state carries the banner, the h1 and a body -- never a blank page (SPEC R5)."""
    bodies = [
        bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text,
        bets_client.get(f"/bets?season={_SEASON}&week={_EMPTY_WEEK}").text,
    ]
    for body in bodies:
        assert _BANNER_EYEBROW in body
        assert "Weekly Bet List" in body
        main = body[body.index("<main") : body.index("</main>")]
        assert len(main) > 500, "the main region rendered essentially empty"


# ---------------------------------------------------------------------------
# The parameterised shared week selector (plan 31-15 Task 3, D31-26, UI-SPEC E5)
# ---------------------------------------------------------------------------
#
# TEST-BOUNDARY HONESTY. An API response test can prove the SERVER returned what it should. It
# cannot prove what a BROWSER does on a timeout or a dropped socket, because in those cases no
# response exists for the test client to inspect. So the three failure handlers are asserted at
# the MARKUP level -- present, correctly named, all pointing at the same target and template --
# and their browser behaviour is a manual backstop, not something these tests cover. Claiming
# otherwise would be its own small dishonesty in a phase about not overstating things.


def test_exactly_one_week_selector_partial_exists() -> None:
    """No fork. A second selector file is the duplicated definition D31-26 forbids."""
    found = sorted(p.name for p in COMPONENTS_DIR.glob("*week_selector*.html"))
    assert found == ["_week_selector.html"], (
        f"a second week selector partial appeared: {found}"
    )


def test_the_bets_selector_targets_bets_and_omits_the_sort_parameter(
    bets_client: TestClient,
) -> None:
    """The bets selector calls /bets, swaps #bets-content, and carries no sort parameter at all."""
    markup = extract_selector(
        bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    )
    assert 'hx-get="/bets"' in markup
    assert 'hx-target="#bets-content"' in markup
    assert 'hx-indicator="#bets-loading"' in markup
    assert "sort" not in markup, (
        "the bets selector carries a sort parameter for a control the page does not have"
    )
    assert "/fragments/games" not in markup


def test_the_element_ids_differ_between_the_two_pages(
    test_client: TestClient, bets_client: TestClient
) -> None:
    """Ids are derived from the swap target, so two pages cannot emit duplicate DOM ids."""
    this_week = extract_selector(test_client.get("/").text)
    bets = extract_selector(
        bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    )

    this_week_ids = set(re.findall(r'id="([^"]+)"', this_week))
    bets_ids = set(re.findall(r'id="([^"]+)"', bets))

    assert this_week_ids, "the This Week selector emitted no ids"
    assert bets_ids, "the bets selector emitted no ids"
    assert not (this_week_ids & bets_ids), (
        f"the two pages share element ids: {this_week_ids & bets_ids}"
    )
    assert "week-select" in this_week_ids
    assert "bets-content-week-select" in bets_ids
    # The label association moved with the id in the same edit.
    for markup, ids in ((this_week, this_week_ids), (bets, bets_ids)):
        for element_id in ids:
            if element_id.endswith("-select"):
                assert f'for="{element_id}"' in markup


def test_all_three_failure_events_are_wired_in_kebab_case(
    bets_client: TestClient,
) -> None:
    """response-error, timeout and send-error are ALL present, all pointing at the same target.

    A timeout and a dropped connection are DIFFERENT htmx events from an HTTP error response;
    wiring only response-error -- the shipped pattern -- would leave both silent, and silence here
    means one week's rows under another week's heading (T-31-74b).

    MARKUP LEVEL ONLY. This proves the handlers are present and correctly named. It does NOT prove
    what a browser does when htmx:timeout fires; that is a manual backstop check.
    """
    markup = extract_selector(
        bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    )

    for event in FAILURE_EVENTS:
        assert event in markup, f"{event} is not wired"
        # Lowercase kebab-case: DOM attributes are lowercased, so a camel-case name would look
        # present and silently never bind.
        assert event == event.lower()
    assert "responseError" not in markup
    assert "sendError" not in markup

    handlers = re.findall(r'hx-on::[a-z-]+="([^"]*)"', markup)
    assert len(handlers) >= len(FAILURE_EVENTS)
    assert len(set(handlers)) == 1, "the three failure events run different handlers"
    handler = handlers[0]
    assert "bets-failure-template" in handler
    assert "#bets-content" in handler
    # The displayed week label reverts to the week actually being shown.
    assert f"w.value='{_WEEK}'" in handler
    assert f"s.value='{_SEASON}'" in handler


# ---------------------------------------------------------------------------
# The timeout that makes the handler above REACHABLE (plan 31-21, G-31-127)
# ---------------------------------------------------------------------------
#
# The three handlers asserted above were bound correctly and could never run. htmx gives the
# XMLHttpRequest it creates a timeout of ZERO unless one is supplied through one of its three
# channels, and per the XHR spec a timeout of zero means "no timeout" -- so the XHR timeout event
# never fires and htmx never emits htmx:timeout. In htmx 2.0.4 (the version base.html loads)
# htmx:timeout has exactly one emitter, a pure passthrough of that XHR event.
#
# These cases pin the SUPPLY of that timeout. They do not, and cannot, prove that a browser then
# runs the handler -- that requires a request with no response, which a response test has nothing
# to inspect. See the TEST-BOUNDARY HONESTY note above; the browser half is a named owner check.

#: The selector's per-element request-configuration attribute, single-quoted so its JSON value
#: keeps its own double quotes.
_HX_REQUEST_RE = re.compile(r"hx-request='([^']*)'")

#: A timeout must leave real headroom over the measured endpoint latency and must still be short
#: enough that a reader is not left waiting. Every htmx endpoint this app serves answers in under
#: 40 ms because nothing is computed in the request path (UIAP-01) -- the /bets fragment itself
#: measured 7.5-15.5 ms -- so 5000 ms is already ~125x the worst case. This is deliberately a
#: RANGE and not the exact value: the number is tunable, and pinning it here would turn a policy
#: decision into a change detector.
_MIN_TIMEOUT_MS = 5_000
_MAX_TIMEOUT_MS = 60_000

_PAGES_DIR = COMPONENTS_DIR.parent / "pages"


def _bets_ws_timeout() -> int:
    """The timeout ``/bets`` actually passes the selector, read from the page's own source.

    Read rather than hardcoded so the direct-render case below exercises the value the PAGE
    chose. A literal here would assert the range of a number this test itself supplied.
    """
    source = (_PAGES_DIR / "bets.html").read_text(encoding="utf-8")
    match = re.search(r"ws_timeout=(\d+)", source)
    assert match is not None, (
        "pages/bets.html no longer passes ws_timeout to the week selector; every failure handler "
        "on that page is bound but unreachable again (G-31-127)"
    )
    return int(match.group(1))


def _bets_selector_context(
    *, with_timeout: bool, with_sync: bool = False
) -> dict[str, Any]:
    """The parameters ``/bets`` passes the shared selector, for a direct render of the partial.

    Both opt-ins are separately switchable because each one's anti-vacuity control renders the
    SAME context minus that one parameter: a shared "everything on" context could not distinguish
    "the guard works" from "the render broke".
    """
    context: dict[str, Any] = dict(PRE_PARAM_CONTEXT)
    context.update(
        {
            "ws_endpoint": "/bets",
            "ws_target": "#bets-content",
            "ws_include_season": "",
            "ws_include_week": "[name='season']",
            "ws_extra_params": {},
            "ws_indicator": "#bets-loading",
            "ws_failure_template": "bets-failure-template",
        }
    )
    if with_timeout:
        context["ws_timeout"] = _bets_ws_timeout()
    if with_sync:
        context["ws_sync"] = True
    return context


def _render_selector(context: dict[str, Any]) -> str:
    return app_templates.env.get_template("components/_week_selector.html").render(
        **context
    )


def test_the_timeout_reaches_all_four_selector_controls() -> None:
    """All FOUR controls carry the same parseable timeout -- one definition, not four.

    Rendered DIRECTLY rather than read off the page, because the ``/bets`` fixture supplies ONE
    season and puts the current week at the END of the available list: the season select is not
    rendered at all and the next button renders ``disabled``, so two of the four controls never
    emit their conditional attributes on that fixture. A page-level assertion would silently
    check two controls while claiming four. The three-week, two-season context recorded in
    ``week_selector_snapshot.PRE_PARAM_CONTEXT`` exists for exactly this reason.

    All four matter: all four already carry the failure handlers, and a control with a handler
    but no timeout is precisely the defect being fixed.
    """
    markup = _render_selector(_bets_selector_context(with_timeout=True))
    values = _HX_REQUEST_RE.findall(markup)

    assert len(values) == 4, (
        f"expected a request timeout on all four controls, found {len(values)}: {values}"
    )
    assert len(set(values)) == 1, (
        f"the four controls carry DIFFERENT timeouts and can drift apart: {sorted(set(values))}"
    )

    parsed = json.loads(values[0])
    assert set(parsed) == {"timeout"}, (
        f"the request configuration carries keys beyond the timeout: {sorted(parsed)}"
    )
    assert isinstance(parsed["timeout"], int)
    assert _MIN_TIMEOUT_MS <= parsed["timeout"] <= _MAX_TIMEOUT_MS, (
        f"{parsed['timeout']}ms is outside the sane band "
        f"[{_MIN_TIMEOUT_MS}, {_MAX_TIMEOUT_MS}]"
    )


def test_the_live_bets_page_passes_the_selector_a_timeout(
    bets_client: TestClient,
) -> None:
    """The WIRING check: the partial having the capability proves nothing if the page withholds it.

    "At least one" rather than four here, deliberately -- on the ``/bets`` fixture two of the four
    controls are absent or disabled and emit no conditional attributes at all. The four-control
    claim is made by the direct-render case above, which controls the fixture shape.
    """
    markup = extract_selector(
        bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    )
    values = _HX_REQUEST_RE.findall(markup)

    assert values, (
        "the served /bets selector carries no request timeout, so its hx-on::timeout handler is "
        "bound but can never fire (G-31-127)"
    )
    for value in values:
        assert json.loads(value)["timeout"] > 0, (
            f"a non-positive timeout is what htmx already had: {value}"
        )


def test_omitting_the_timeout_parameter_emits_no_request_configuration() -> None:
    """The anti-vacuity control for the D31-26 default: no parameter, no attribute.

    This is the assertion that catches someone "simplifying" the ``{% if %}`` guard away. Without
    it the six pages that pass no failure template would silently acquire abort behaviour with
    nowhere to send the event, and the two recorded This Week snapshots would be the only thing
    standing between that change and the shipped page.
    """
    markup = _render_selector(_bets_selector_context(with_timeout=False))
    assert not _HX_REQUEST_RE.findall(markup)
    assert "hx-request" not in markup
    # The rest of the selector is unaffected: this is a guard on one attribute, not a kill switch.
    for event in FAILURE_EVENTS:
        assert event in markup


# ---------------------------------------------------------------------------
# The concurrency guard on the window the timeout BOUNDS (plan 31-24, owner-ruled)
# ---------------------------------------------------------------------------
#
# The timeout above bounds how long a wedged request can lie. It does not stop the page lying
# DURING that window, and the debug session measured exactly how (evidence E10, against a real
# socket-level hung server): htmx's default for a second request from the same element is to QUEUE
# it while leaving the element interactive, so selecting a second week issued NO request at all and
# yet the control still moved -- the selector read "4" over Week 2's rows. Clicking Next Week, a
# different element, wedged a SECOND request. Neither control was ever disabled and only a full
# page reload recovered.
#
# Two attributes close that, and they are separate concerns: hx-disabled-elt stops the control that
# is asking from being moved while its own request is in flight, and hx-sync makes a request from
# any of the four REPLACE the in-flight one instead of queueing behind it, so a stale response
# cannot land after a newer selection.
#
# WHY THE SYNC TARGET IS THE SWAP TARGET. The four controls sit at different nesting depths, so no
# single ancestor selector covers all four; the swap target is the one element they are all
# competing to replace, which is the thing that actually needs serialising; and it requires no
# attribute on the selector's ROOT div, which ``week_selector_snapshot.SELECTOR_OPEN`` matches by
# its exact opening tag and which would therefore break every existing selector assertion.
#
# MARKUP LEVEL ONLY, same boundary as the timeout cases. These prove the attributes are emitted.
# They cannot prove htmx acted on them -- and the id-selector form of hx-sync is the one part of
# this the debug session did not exercise live -- so the behavioural half is a named owner browser
# check, not something these tests cover.

_HX_SYNC_RE = re.compile(r'hx-sync="([^"]*)"')
_HX_DISABLED_ELT_RE = re.compile(r'hx-disabled-elt="([^"]*)"')
_HX_TARGET_RE = re.compile(r'hx-target="([^"]*)"')

#: htmx 2.0.4's grammar is ``<selector>:<strategy>``; ``replace`` is the strategy that aborts the
#: in-flight request and continues, rather than dropping the new one (``drop``, the default) or
#: dropping itself (``abort``) or queueing (``queue``).
_SYNC_STRATEGY = "replace"


def test_the_request_sync_and_disable_reach_all_four_selector_controls() -> None:
    """All FOUR controls serialise on the SAME element, and that element is the swap target.

    Rendered DIRECTLY rather than read off the page, for the same reason the timeout case above
    is: the ``/bets`` fixture supplies one season and puts the current week at the END of the
    available list, so the season select is not rendered and the next button renders ``disabled``
    -- two of the four controls emit no conditional attributes at all on that fixture, and a
    page-level assertion would silently check two while claiming four.

    The sync target is read OUT OF THE SAME RENDER rather than hardcoded, so a page that retargets
    the selector cannot leave the serialisation pointing at an element it no longer swaps.
    """
    markup = _render_selector(_bets_selector_context(with_timeout=True, with_sync=True))

    sync_values = _HX_SYNC_RE.findall(markup)
    disable_values = _HX_DISABLED_ELT_RE.findall(markup)

    assert len(sync_values) == 4, (
        f"expected request synchronisation on all four controls, found "
        f"{len(sync_values)}: {sync_values}"
    )
    assert len(disable_values) == 4, (
        f"expected a disable directive on all four controls, found "
        f"{len(disable_values)}: {disable_values}"
    )

    # THE COUPLING (WR-06). The two attributes are not merely both nice to have: hx-disabled-elt
    # ALONE is a defect. htmx's shouldInclude drops a disabled element from a sibling's
    # hx-include, and the week select on /bets includes [name='season'] -- so a disabled season
    # select would have its value dropped and /bets would be called with a week and no season,
    # falling back to a default season and serving a different season's week N under the
    # displayed one. Silent wrong data. What prevents it is the SYNC half: the abort it fires
    # runs removeRequestIndicators (which clears `disabled`) synchronously, BEFORE
    # getInputValues gathers the new request's values. Asserted as an equality, and not only as
    # two counts of four, so the claim being defended is stated where it can be read: whatever
    # the number of controls becomes, neither attribute may be emitted without the other.
    assert len(sync_values) == len(disable_values), (
        f"{len(sync_values)} control(s) carry hx-sync but {len(disable_values)} carry "
        "hx-disabled-elt. These MUST be emitted together: a control that disables itself "
        "without the synchronisation that re-enables it on abort can have its value dropped "
        "from a sibling's hx-include, and /bets would then serve a different season's week "
        "under the displayed season. See the ws_sync header block in _week_selector.html."
    )

    assert len(set(sync_values)) == 1, (
        "the four controls serialise on DIFFERENT elements, so they do not serialise with each "
        f"other at all: {sorted(set(sync_values))}"
    )
    assert set(disable_values) == {"this"}, (
        f"a control disables something other than itself: {sorted(set(disable_values))}"
    )

    targets = set(_HX_TARGET_RE.findall(markup))
    assert len(targets) == 1, (
        f"the render swaps more than one target, so 'the' swap target is ambiguous: {targets}"
    )
    swap_target = targets.pop()

    selector, _, strategy = sync_values[0].rpartition(":")
    assert selector == swap_target, (
        f"the controls serialise on {selector!r} but swap {swap_target!r}; a retargeted selector "
        "would leave the serialisation pointing at a stale element"
    )
    assert strategy == _SYNC_STRATEGY, (
        f"expected the {_SYNC_STRATEGY!r} strategy, which aborts the in-flight request and "
        f"continues; found {strategy!r}"
    )


def test_the_live_bets_page_passes_the_selector_the_sync_opt_in(
    bets_client: TestClient,
) -> None:
    """The WIRING check: the partial having the capability proves nothing if the page withholds it.

    "At least one" rather than four here, deliberately -- on the ``/bets`` fixture two of the four
    controls are absent or disabled and emit no conditional attributes at all. The four-control
    claim is made by the direct-render case above, which controls the fixture shape.
    """
    markup = extract_selector(
        bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    )

    sync_values = _HX_SYNC_RE.findall(markup)
    disable_values = _HX_DISABLED_ELT_RE.findall(markup)

    assert sync_values, (
        "the served /bets selector serialises nothing, so a second interaction during a wedged "
        "request still queues behind it and the control still moves (E10)"
    )
    assert disable_values, (
        "the served /bets selector disables nothing, so the control that issued the in-flight "
        "request can still be moved to a week the content does not describe (E10)"
    )
    for value in sync_values:
        assert value.endswith(f":{_SYNC_STRATEGY}"), (
            f"the served selector does not use the {_SYNC_STRATEGY!r} strategy: {value}"
        )
    assert set(disable_values) == {"this"}


def test_omitting_the_sync_parameter_emits_neither_concurrency_attribute() -> None:
    """The anti-vacuity control for the default: no parameter, neither attribute.

    This is the assertion that catches someone "simplifying" the ``{% if %}`` guard away. Without
    it the six pages that pass no failure template would silently acquire request cancellation and
    control disabling that nobody on those pages asked for, and the two recorded This Week
    snapshots would be the only thing standing between that change and the shipped page.
    """
    markup = _render_selector(
        _bets_selector_context(with_timeout=True, with_sync=False)
    )
    assert not _HX_SYNC_RE.findall(markup)
    assert not _HX_DISABLED_ELT_RE.findall(markup)
    assert "hx-sync" not in markup
    assert "hx-disabled-elt" not in markup
    # The rest of the selector is unaffected: this is a guard on two attributes, not a kill switch.
    assert _HX_REQUEST_RE.findall(markup)
    for event in FAILURE_EVENTS:
        assert event in markup


def test_the_failure_template_carries_the_message_and_a_retry(
    bets_client: TestClient,
) -> None:
    """The template the handlers copy holds the error-state geometry plus a retry affordance."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    start = body.index('<template id="bets-failure-template">')
    template = body[start : body.index("</template>", start)]

    assert "bg-red-950" in template, (
        "the failure state does not reuse the error-state geometry"
    )
    assert "could not be loaded" in template
    assert "Retry this week" in template
    # The retry navigates to the week ACTUALLY being shown, not the one that was requested.
    assert f"/bets?season={_SEASON}&amp;week={_WEEK}" in template


def test_the_two_pages_share_every_layout_and_type_class(
    test_client: TestClient, bets_client: TestClient
) -> None:
    """Parameterisation changed NO layout class and NO type class: both renders agree.

    The two snapshots pin the shipped page exactly; this pins the claim that the new call site did
    not acquire a different look through a parameter.
    """
    this_week = set(class_values(extract_selector(test_client.get("/").text)))
    bets = set(
        class_values(
            extract_selector(
                bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
            )
        )
    )
    assert bets <= this_week, (
        f"the bets selector introduced new class strings: {bets - this_week}"
    )


def test_bets_navigation_reads_the_schedule_getters_and_not_the_predictions_ones() -> (
    None
):
    """STRUCTURAL, not textual: an ast walk over the two functions collects the service calls.

    A text search would be tripped by a COMMENT naming the rejected getters -- and both functions
    carry exactly such a comment, explaining why those getters are wrong here. The walk reads the
    code (REVIEW-NAV, T-31-74c).
    """
    source = (
        Path(__file__).resolve().parents[2] / "api" / "routes" / "pages.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)

    called: set[str] = set()
    seen: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef):
            continue
        if node.name not in {"_build_bets_context", "_normalize_week"}:
            continue
        seen.add(node.name)
        for inner in ast.walk(node):
            if (
                isinstance(inner, ast.Call)
                and isinstance(inner.func, ast.Attribute)
                and isinstance(inner.func.value, ast.Name)
                and inner.func.value.id == "service"
            ):
                called.add(inner.func.attr)

    assert seen == {"_build_bets_context", "_normalize_week"}, (
        f"functions not found: {seen}"
    )
    assert {"get_available_bet_weeks", "get_bet_seasons"} <= called, (
        f"the bets navigation getters are not both called: {sorted(called)}"
    )
    assert "get_available_weeks" not in called, (
        "the bets navigation fell back to the predictions-derived week getter"
    )
    assert "get_available_seasons" not in called, (
        "the bets navigation fell back to the backtest_metrics-derived season getter"
    )


def test_a_week_with_no_prediction_row_is_still_offered_and_explains_itself(
    tmp_path: Path,
) -> None:
    """A scheduled week absent from `predictions` IS selectable and resolves to an empty state.

    Built with exactly that gap: the cache carries available_bet_weeks rows and an EMPTY
    predictions table, so the predictions-derived getter returns nothing for the same weeks. A
    reader who wants to check that week must find it in the control, not discover it missing.
    """
    clear_cache()
    db_path = tmp_path / "prediction_gap.duckdb"
    gap_week = 5
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(conn, pd.DataFrame([_live_row("2023_W01_DET@KC", "ou")]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK},
                    {
                        "game_id": "2023_W05_AAA@BBB",
                        "season": _SEASON,
                        "week": gap_week,
                    },
                ]
            ),
        )
        _stamp_populated_at(conn, _POPULATED_AT)
        assert conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0] == 0
    finally:
        conn.close()

    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        assert DataService(probe).get_available_weeks(season=_SEASON) == [], (
            "the fixture does not actually carry a prediction-row gap"
        )
    finally:
        probe.close()
    clear_cache()

    with contextmanager(_client)(db_path) as client:
        markup = extract_selector(
            client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
        )
        assert f'value="{gap_week}"' in markup, (
            "a scheduled week with no prediction row vanished from the selector"
        )

        gap_body = client.get(f"/bets?season={_SEASON}&week={gap_week}").text

    assert _ZERO_ADMITTED_HEADING in gap_body, (
        "the gap week did not resolve to an explanatory empty state"
    )


def test_the_selector_still_renders_in_the_off_season_state(tmp_path: Path) -> None:
    """UI-SPEC E5 empty: the no-current-week body points at the selector, so it must be there."""
    clear_cache()
    db_path = tmp_path / "off_season_selector.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        body = client.get("/bets").text

    assert _NO_CURRENT_WEEK_HEADING in body
    assert SELECTOR_OPEN in body, (
        "the selector vanished in the state whose copy points at it"
    )
    markup = extract_selector(body)
    assert 'id="bets-content-week-select"' in markup


# ---------------------------------------------------------------------------
# The realized-versus-expected tracker (plan 31-16 Task 1, SPEC R8, D31-22, UI-SPEC E4)
# ---------------------------------------------------------------------------
#
# The tracker partitions on the (provenance, validation_type) PAIR, not on provenance alone
# (plan 31-13). A cache carrying BOTH the burned 2021-2024 replay rows and the single clean 2025
# holdout therefore produces THREE blocks, not two -- and the page renders three sections. Pooling
# the two replay classes would publish the phase's one unspent split inside a contaminated figure,
# which is exactly what D31-22's second column exists to prevent.
#
# Every figure below is transcribed from the design contract as a LITERAL rather than imported from
# the template, so a silent edit to either side is a failure rather than a tautology.

_REPLAY_HEADING = "Backtest replay -- reconstructed after the fact"
_FORWARD_HEADING = "Forward record -- recommended before kickoff"
_REPLAY_CAPTION = (
    "These bets were never recommended in advance. They were reconstructed from historical "
    "data to show how the selection rule would have behaved. Do not read them as a track record."
)
_FORWARD_CAPTION = (
    "Each of these was decided before its game locked at 6:00 PM Eastern the day before "
    "kickoff, before the result existed. This is the only block that is a track record."
)
_PUSH_FOOTNOTE = (
    "A push returns the stake. Pushes are excluded from the hit-rate denominator and are never "
    "counted as a win or a loss."
)
_ZERO_RESULT_LINE = "A negative return here is the measurement, not a display problem."
_NOTHING_GRADED_HEADING = "Nothing graded yet"
_TRACKER_FIGURE_LABELS = (
    "Bets graded",
    "Wins",
    "Losses",
    "Pushes",
    "Hit rate",
    "Return (flat, units)",
)
_FORWARD_WITHHELD_MESSAGE = "The forward record is withheld -- this week&#39;s locked games have no list in the cache"

# The three honesty classes, in the declared display order (weakest evidence to strongest).
_CONTAMINATED = ("backtest_replay", "contaminated")
_CLEAN_HOLDOUT = ("backtest_replay", "clean_holdout")
_FORWARD_CLASS = ("forward", "forward_realized")


def _block(
    pair: tuple[str, str],
    *,
    bets_graded: int,
    wins: int,
    losses: int,
    pushes: int,
    hit_rate: float,
    flat_return_units: float | None,
) -> TrackerBlock:
    """One POPULATED tracker block, built through the production dataclass."""
    return TrackerBlock(
        provenance=pair[0],
        validation_type=pair[1],
        bets_graded=bets_graded,
        wins=wins,
        losses=losses,
        pushes=pushes,
        hit_rate=hit_rate,
        flat_return_units=flat_return_units,
    )


def _client_with_tracker(
    tmp_path: Path,
    blocks: list[Any],
    name: str,
    *,
    rows: list[dict[str, Any]] | None = None,
    freeze: datetime | None = None,
) -> Any:
    """Build a cache carrying PRECOMPUTED tracker blocks and return a client serving it.

    The blocks are written through ``backtest.bet_tracker.to_tracker_frame`` and
    ``api.cache.materialize_bet_tracker_blocks`` -- the production producer and the production
    writer -- so a NULL rate reaches DuckDB as SQL NULL rather than as a float NaN, and the
    not-measured / measured-zero distinction the page renders is the one the pipeline stores.
    """
    clear_cache()
    bet_rows = rows if rows is not None else [_live_row("2023_W01_DET@KC", "ou")]
    db_path = tmp_path / f"{name}.duckdb"
    conn = _build_bare_cache(db_path)
    schedule = [
        {"game_id": r["game_id"], "season": _SEASON, "week": r["week"]}
        for r in bet_rows
    ] or [{"game_id": "2023_W01_AAA@BBB", "season": _SEASON, "week": _WEEK}]
    try:
        if bet_rows:
            materialize_bet_list(conn, pd.DataFrame(bet_rows))
        materialize_available_bet_weeks(conn, pd.DataFrame(schedule))
        materialize_bet_tracker_blocks(conn, to_tracker_frame(blocks))
        if freeze is not None:
            # Every scheduled game locks at *freeze*: with no bet rows, that is the hard block.
            materialize_bet_week_freeze(
                conn,
                pd.DataFrame([{**game, "game_freeze_ts": freeze} for game in schedule]),
            )
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()
    return contextmanager(_client)(db_path)


def _tracker_sections(body: str) -> dict[str, str]:
    """Split the rendered page into its tracker sections, keyed by the data attribute.

    Structural, not textual: each section is located by its ``data-tracker-block`` key and cut at
    the next section boundary, so a figure can be attributed to exactly one class.
    """
    starts = [
        (m.group(1), m.start())
        for m in re.finditer(r'<section [^>]*data-tracker-block="([^"]+)"', body)
    ]
    sections: dict[str, str] = {}
    for index, (key, start) in enumerate(starts):
        end = starts[index + 1][1] if index + 1 < len(starts) else len(body)
        sections[key] = body[start:end]
    return sections


# ---------------------------------------------------------------------------
# The result strip's source: the graded outcomes behind each tracker block (redesign Task 10)
# ---------------------------------------------------------------------------
#
# The strip draws one mark per graded bet. It must be drawn from EXACTLY the rows
# backtest.bet_tracker counts into a block -- status live, grading_status win/loss/push -- or it
# would show a different record from the tiles beside it. These tests pin that row set against the
# real aggregator rather than against a transcription of its filter.


def _graded_row(
    game_id: str,
    grading_status: str,
    *,
    pair: tuple[str, str] = _CONTAMINATED,
    week: int = _WEEK,
    target: str = "ou",
) -> dict[str, Any]:
    """One LIVE bet_list row carrying a stored grade, in the shape the grader writes it."""
    row = _live_row(game_id, target, week=week)
    row.update(
        {
            "provenance": pair[0],
            "validation_type": pair[1],
            "grading_status": grading_status,
            "outcome": {"win": True, "loss": False}.get(grading_status),
            "payout_flat": {"win": 0.909, "loss": -1.0, "push": 0.0}.get(
                grading_status
            ),
        }
    )
    return row


def _graded_fixture_rows() -> list[dict[str, Any]]:
    """Graded, ungraded and never-bet rows across two classes and two weeks."""
    return [
        _graded_row("2023_W01_DET@KC", "win"),
        _graded_row("2023_W01_CAR@ATL", "loss"),
        _graded_row("2023_W01_CIN@CLE", "push"),
        # Ungraded: a live bet whose result is not known yet. Not part of any record.
        _graded_row("2023_W01_JAX@IND", "pending"),
        # Never bet: a suppressed candidate. Not part of any record.
        _suppressed_row("2023_W01_SEA@SFO", "ats", "ev_below_floor"),
        _graded_row("2023_W02_AAA@BBB", "win", week=_EMPTY_WEEK),
        _graded_row("2023_W01_DEN@LVR", "win", pair=_FORWARD_CLASS),
    ]


def _graded_cache(tmp_path: Path, name: str) -> Path:
    """A cache whose bet_list carries the graded fixture rows and their schedule."""
    clear_cache()
    rows = _graded_fixture_rows()
    db_path = tmp_path / f"{name}.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(conn, pd.DataFrame(rows))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {"game_id": r["game_id"], "season": _SEASON, "week": r["week"]}
                    for r in rows
                ]
            ),
        )
    finally:
        conn.close()
    return db_path


def test_the_graded_outcomes_are_the_live_graded_rows_in_a_fixed_order(
    tmp_path: Path,
) -> None:
    """Live AND graded only, ordered by class then the four-key tie-break, as stored."""
    db_path = _graded_cache(tmp_path, "graded_order")
    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = DataService(probe).get_graded_bet_outcomes()
    finally:
        probe.close()

    assert [
        (r["provenance"], r["validation_type"], r["game_id"], r["grading_status"])
        for r in rows
    ] == [
        ("backtest_replay", "contaminated", "2023_W01_CAR@ATL", "loss"),
        ("backtest_replay", "contaminated", "2023_W01_CIN@CLE", "push"),
        ("backtest_replay", "contaminated", "2023_W01_DET@KC", "win"),
        ("backtest_replay", "contaminated", "2023_W02_AAA@BBB", "win"),
        ("forward", "forward_realized", "2023_W01_DEN@LVR", "win"),
    ]
    assert set(rows[0]) == {
        "provenance",
        "validation_type",
        "season",
        "week",
        "game_id",
        "target",
        "grading_status",
    }


def test_the_graded_outcomes_count_exactly_what_the_tracker_aggregates(
    tmp_path: Path,
) -> None:
    """Per class, the outcome marks equal the REAL aggregator's wins, losses and pushes."""
    db_path = _graded_cache(tmp_path, "graded_vs_tracker")
    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        outcomes = DataService(probe).get_graded_bet_outcomes()
    finally:
        probe.close()

    blocks = aggregate_all_blocks(pd.DataFrame(_graded_fixture_rows()))
    assert blocks, "the fixture produced no tracker block, so this proves nothing"
    for block in blocks:
        assert isinstance(block, TrackerBlock)
        statuses = [
            r["grading_status"]
            for r in outcomes
            if (r["provenance"], r["validation_type"])
            == (block.provenance, block.validation_type)
        ]
        assert statuses.count("win") == block.wins
        assert statuses.count("loss") == block.losses
        assert statuses.count("push") == block.pushes
        assert len(statuses) == block.bets_graded


def test_the_graded_outcomes_tolerate_a_cache_without_a_bet_list(
    tmp_path: Path,
) -> None:
    """A cache that predates the bet list returns no outcomes rather than raising."""
    clear_cache()
    db_path = tmp_path / "graded_no_table.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        conn.execute("DROP TABLE bet_list")
    finally:
        conn.close()
    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        assert DataService(probe).get_graded_bet_outcomes() == []
    finally:
        probe.close()


def test_the_bets_context_partitions_the_outcomes_by_honesty_class(
    tmp_path: Path,
) -> None:
    """The context keys each class by the same string its tracker section renders."""
    from api.routes.pages import _build_bets_context

    db_path = _graded_cache(tmp_path, "graded_context")
    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        context = _build_bets_context(DataService(probe), _SEASON, _WEEK, None)  # pyright: ignore[reportArgumentType]
    finally:
        probe.close()

    assert context["graded_outcomes"] == {
        "backtest_replay:contaminated": ["loss", "push", "win", "win"],
        "forward:forward_realized": ["win"],
    }


def test_the_tracker_renders_one_section_per_honesty_class_and_never_pools_them(
    tmp_path: Path,
) -> None:
    """Three classes give THREE sections with separate totals; no figure spans two of them.

    THIS IS THE DIVERGENCE FROM THE DESIGN CONTRACT'S TWO-BLOCK SKETCH, and it is deliberate. The
    aggregator partitions on the PAIR, so the replay provenance holds two classes: the burned
    2021-2024 contaminated rows and the single clean 2025 holdout. Rendering them as one section
    would publish the one unspent split inside a contaminated figure. The contract names two
    HEADINGS; it does not cap the section count.
    """
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=40,
            wins=18,
            losses=20,
            pushes=2,
            hit_rate=0.4736842105263158,
            flat_return_units=-0.0528,
        ),
        _block(
            _CLEAN_HOLDOUT,
            bets_graded=11,
            wins=7,
            losses=4,
            pushes=0,
            hit_rate=0.6363636363636364,
            flat_return_units=0.0325,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=3,
            wins=2,
            losses=1,
            pushes=0,
            hit_rate=0.6666666666666666,
            flat_return_units=0.0144,
        ),
    ]
    with _client_with_tracker(tmp_path, blocks, "three_classes") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    sections = _tracker_sections(body)
    assert set(sections) == {
        "backtest_replay:contaminated",
        "backtest_replay:clean_holdout",
        "forward:forward_realized",
    }, f"the tracker did not render one section per class: {sorted(sections)}"

    # Two headings, three sections. Both replay sections carry the replay heading and caption.
    assert body.count(_REPLAY_HEADING) == 2
    assert body.count(_FORWARD_HEADING) == 1
    assert body.count(_REPLAY_CAPTION) == 2
    assert body.count(_FORWARD_CAPTION) == 1

    # Separate totals, and NO pooled figure anywhere. 40 + 11 + 3 = 54 graded bets pooled; the
    # pooled hit rate over the two replay classes would be 25/49. Neither may appear.
    assert ">40<" in sections["backtest_replay:contaminated"]
    assert ">11<" in sections["backtest_replay:clean_holdout"]
    assert ">3<" in sections["forward:forward_realized"]
    assert ">54<" not in body, "a pooled bets-graded total reached the page"
    assert ">51<" not in body, "a pooled replay bets-graded total reached the page"

    # Every rendered figure tile lives inside exactly one section, located by its data hook.
    inside = sum(section.count("data-figure-label") for section in sections.values())
    tracker_start = min(body.index(s) for s in sections.values())
    tracker_region = body[tracker_start:]
    assert inside == tracker_region.count("data-figure-label"), (
        "a tracker figure rendered outside one of the sections"
    )
    assert inside == 18, "three sections of six figures each did not all render"


def test_each_block_renders_exactly_the_six_named_figures(tmp_path: Path) -> None:
    """Six figures per block, no more. An extra figure on the page is an extra claim."""
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=10,
            wins=5,
            losses=4,
            pushes=1,
            hit_rate=0.5555555555555556,
            flat_return_units=0.012,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "six_figures") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    section = _tracker_sections(body)["backtest_replay:contaminated"]
    labels = re.findall(r'<p class="label" data-figure-label>([^<]+)</p>', section)
    assert labels == list(_TRACKER_FIGURE_LABELS), (
        f"the block did not render exactly the six named figures in order: {labels}"
    )


def test_pushes_sits_outside_the_group_that_holds_the_hit_rate(tmp_path: Path) -> None:
    """STRUCTURAL, not visual: the pushes figure is in its own group wrapper.

    A push settled without either side winning, so it is not in the hit-rate denominator. The
    three ``data-figure-group`` wrappers are what make that assertable rather than a claim about
    pixel order.
    """
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=10,
            wins=5,
            losses=4,
            pushes=1,
            hit_rate=0.5555555555555556,
            flat_return_units=0.012,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "push_group") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    section = _tracker_sections(body)["backtest_replay:contaminated"]
    groups = dict(
        re.findall(
            r'<div data-figure-group="([a-z-]+)"(.*?)(?=<div data-figure-group=|</div>\s*</div>\s*</div>)',
            section,
            flags=re.S,
        )
    )
    assert {"counts", "pushes", "rates"} <= set(groups), (
        f"the figure groups are not all present: {sorted(groups)}"
    )
    assert "Pushes" in groups["pushes"]
    assert "Hit rate" in groups["rates"]
    assert "Pushes" not in groups["rates"], (
        "the pushes figure sits inside the group that holds the hit rate"
    )
    assert "Hit rate" not in groups["pushes"]
    assert _PUSH_FOOTNOTE in section


def test_a_block_with_zero_graded_rows_renders_the_empty_state_and_no_hit_rate(
    tmp_path: Path,
) -> None:
    """Zero graded gives the nothing-graded-yet state -- never a row of zeros, never 0 / 0."""
    blocks = [
        EmptyTrackerBlock(provenance=_CONTAMINATED[0], validation_type=_CONTAMINATED[1])
    ]
    with _client_with_tracker(tmp_path, blocks, "zero_graded") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    section = _tracker_sections(body)["backtest_replay:contaminated"]
    assert _NOTHING_GRADED_HEADING in section
    assert "No recommendation in this class has a final result yet." in section
    assert "Hit rate" not in section, (
        "an ungraded block rendered a hit rate the aggregator never computed"
    )
    assert "Bets graded" not in section, "an ungraded block rendered a row of zeros"
    assert "0 / 0" not in body
    # The replay heading and caption still render: the class is named, only its figures are absent.
    assert _REPLAY_HEADING in section
    assert _REPLAY_CAPTION in section


def test_a_one_row_block_uses_the_same_noun_phrase_labels(tmp_path: Path) -> None:
    """One graded row is a normal block: the labels are noun phrases that do not inflect."""
    blocks = [
        _block(
            _FORWARD_CLASS,
            bets_graded=1,
            wins=1,
            losses=0,
            pushes=0,
            hit_rate=1.0,
            flat_return_units=0.909,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "one_row") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    section = _tracker_sections(body)["forward:forward_realized"]
    for label in _TRACKER_FIGURE_LABELS:
        assert f">{label}</p>" in section
    for singular in ("1 bet graded", "Bet graded", "Win</p>", "Loss</p>", "Push</p>"):
        assert singular not in section, (
            f"a singular copy variant {singular!r} was minted for a one-row block"
        )
    assert _NOTHING_GRADED_HEADING not in section


def test_a_negative_return_states_the_measurement_and_a_positive_one_does_not(
    tmp_path: Path,
) -> None:
    """The honesty line renders on a negative or zero return, and ONLY on one."""
    losing = [
        _block(
            _CONTAMINATED,
            bets_graded=10,
            wins=4,
            losses=6,
            pushes=0,
            hit_rate=0.4,
            flat_return_units=-0.0528,
        )
    ]
    with _client_with_tracker(tmp_path, losing, "negative_return") as client:
        negative_body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    negative = _tracker_sections(negative_body)["backtest_replay:contaminated"]
    assert "-0.053" in negative
    # The RETURN tile itself carries the loss red. The Losses tile is red in every section, so a
    # bare "text-red-400" substring check would pass whatever colour the return took.
    assert 'text-red-400" data-figure-value>-0.053<' in negative
    assert _ZERO_RESULT_LINE in negative

    winning = [
        _block(
            _CONTAMINATED,
            bets_graded=10,
            wins=6,
            losses=4,
            pushes=0,
            hit_rate=0.6,
            flat_return_units=0.0325,
        )
    ]
    with _client_with_tracker(tmp_path, winning, "positive_return") as client:
        positive_body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    positive = _tracker_sections(positive_body)["backtest_replay:contaminated"]
    assert "+0.033" in positive
    assert _ZERO_RESULT_LINE not in positive, (
        "the honesty line rendered against a positive return"
    )

    flat = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=0,
            losses=0,
            pushes=4,
            hit_rate=0.0,
            flat_return_units=0.0,
        )
    ]
    with _client_with_tracker(tmp_path, flat, "zero_return") as client:
        zero_body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert (
        _ZERO_RESULT_LINE
        in _tracker_sections(zero_body)["backtest_replay:contaminated"]
    )


def test_an_unmeasured_return_never_renders_as_a_zero(tmp_path: Path) -> None:
    """A NULL return renders "not measured" and carries no honesty line.

    The aggregator returns None (not 0.0) when the graded rows carry no stake, mirroring
    ``BetSelector._clv_report`` defaulting ``clv`` to None. A blank and a zero must not render
    identically: "+0.000" would read as a break-even result nobody measured.
    """
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=6,
            wins=3,
            losses=3,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=None,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "unmeasured_return") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    section = _tracker_sections(body)["backtest_replay:contaminated"]
    assert "not measured" in section
    assert "+0.000" not in section, "an unmeasured return rendered as a measured zero"
    # Scoped to the rendered FIGURE VALUES: the word "provenance" contains the letters "nan", so
    # a whole-section substring search would be a false positive rather than a check.
    values = re.findall(
        r'<p class="stat-tile-value[^"]*" data-figure-value>([^<]*)</p>', section
    )
    assert values, "the block rendered no figures"
    assert not any("nan" in value.lower() for value in values), (
        f"a non-finite value reached a rendered figure: {values}"
    )
    assert _ZERO_RESULT_LINE not in section, (
        "the honesty line claims a negative measurement where nothing was measured"
    )


# The reds components/_error_state.html reserves for a FAILURE: its box, border, icon, heading and
# body. Deliberately not the realised-loss red-400 / red-500 the tracker uses.
_ERROR_STATE_REDS = frozenset(
    {"bg-red-950", "border-red-800", "text-red-300", "text-red-200", "text-red-100"}
)


def test_green_and_red_appear_only_inside_the_tracker_sections(tmp_path: Path) -> None:
    """Realized-outcome colour is TRACKER ONLY; the EV band badges carry none of it.

    Scoped to the badge markup rather than to the whole page: a green EV band above a green
    won-bet would read as a prediction of winning (UI-SPEC Deviation 1).
    """
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=10,
            wins=6,
            losses=4,
            pushes=0,
            hit_rate=0.6,
            flat_return_units=-0.01,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "colour_scope") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    sections = _tracker_sections(body)
    replay = sections["backtest_replay:contaminated"]
    # Each realized-outcome hue is pinned to the tile that carries it. The Wins tile is always
    # green and the Losses tile always red, so a bare substring check anywhere in the section
    # could not tell whether the RETURN (-0.01 here) took its colour from its sign.
    assert 'text-green-400" data-figure-value>6<' in replay
    assert 'text-red-400" data-figure-value>-0.010<' in replay

    # Every EV band badge on the page, located by its authored title prefix.
    badges = re.findall(
        r'<span class="[^"]*"[^>]*title="EV band[^"]*"[^>]*>.*?</span>', body
    )
    assert badges, "no EV band badge rendered, so the scoping assertion proves nothing"
    for badge in badges:
        for classes in _class_attributes(badge):
            for forbidden in ("green", "red", "amber"):
                assert forbidden not in classes, (
                    f"an EV band badge carries the {forbidden} hue: {classes}"
                )

    # Outside the tracker region the realized-outcome colours do not appear at all: no green
    # shade whatever, and no red shade but the error state's own family, which the failure
    # template above the tracker carries and which can never read as "a bet lost". Any shade, not
    # a list of exact ones, so a new green or red class elsewhere on the page fails here.
    tracker_start = min(body.index(s) for s in sections.values())
    last_section = max(body.index(s) for s in sections.values())
    tracker_end = body.index("</section>", last_section) + len("</section>")
    outside = body[:tracker_start] + body[tracker_end:]
    greens = re.findall(r"\b(?:text|bg|border)-green-\d+", outside)
    assert not greens, f"green rendered outside the tracker: {sorted(set(greens))}"
    reds = set(re.findall(r"\b(?:text|bg|border)-red-\d+", outside))
    assert reds - _ERROR_STATE_REDS == set(), (
        f"a red outside the tracker is not the error state's: {sorted(reds - _ERROR_STATE_REDS)}"
    )


def test_the_forward_block_is_withheld_under_the_hard_block_while_replay_stays_readable(
    tmp_path: Path,
) -> None:
    """The refusal is SCOPED: the forward totals are withheld, the replay figures are not.

    A replay figure does not depend on the current week's locks, so refusing it would be a
    refusal nothing justified (UI-SPEC E4 error). The week has a locked game and NO bet row, which
    is the hard block since 33.2 review C2 CR-02.
    """
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=40,
            wins=18,
            losses=20,
            pushes=2,
            hit_rate=0.4736842105263158,
            flat_return_units=-0.0528,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=3,
            wins=2,
            losses=1,
            pushes=0,
            hit_rate=0.6666666666666666,
            flat_return_units=0.0144,
        ),
    ]
    with _client_with_tracker(
        tmp_path, blocks, "scoped_refusal", rows=[], freeze=_LATER_FREEZE
    ) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    body = response.text
    assert _HARD_BLOCK_MESSAGE in body, "the fixture did not trigger the hard block"

    sections = _tracker_sections(body)
    assert "backtest_replay:contaminated" in sections
    replay = sections["backtest_replay:contaminated"]
    assert ">40<" in replay, "the replay block was withheld alongside the forward one"
    assert "Hit rate" in replay

    forward = sections["forward"]
    assert _FORWARD_WITHHELD_MESSAGE in forward
    assert "bg-red-950" in forward, (
        "the withheld forward block did not use the error state"
    )
    assert "Hit rate" not in forward
    assert ">3<" not in forward, "a forward figure was served under the hard block"
    assert _FORWARD_HEADING in forward, (
        "the forward section vanished rather than declaring itself withheld"
    )


def test_the_page_renders_the_stored_aggregate_and_computes_nothing(
    tmp_path: Path,
) -> None:
    """The route reads the precomputed blocks; the template applies formatting only.

    Two claims, both asserted. STRUCTURALLY: an ast walk over ``_build_bets_context`` finds the
    tracker getter and no other aggregate source. NUMERICALLY: the rendered hit rate is the stored
    fraction under a per-cent label, and the rendered return is the stored value -- a page that
    re-derived a rate from the wins and losses would produce a different number for a block whose
    stored rate deliberately disagrees with its counts.
    """
    source = (
        Path(__file__).resolve().parents[2] / "api" / "routes" / "pages.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(source)
    called: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_build_bets_context":
            for inner in ast.walk(node):
                if (
                    isinstance(inner, ast.Call)
                    and isinstance(inner.func, ast.Attribute)
                    and isinstance(inner.func.value, ast.Name)
                    and inner.func.value.id == "service"
                ):
                    called.add(inner.func.attr)
    assert "get_bet_tracker_blocks" in called, (
        f"the bets context does not read the precomputed tracker: {sorted(called)}"
    )

    # A stored rate that a recomputation could not produce: 3 wins and 1 loss would give 75.0%.
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=3,
            losses=1,
            pushes=0,
            hit_rate=0.125,
            flat_return_units=-0.75,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "stored_aggregate") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    section = _tracker_sections(body)["backtest_replay:contaminated"]
    assert ">12.5%<" in section, "the page did not render the STORED hit rate"
    assert "75.0%" not in section, "the page recomputed the hit rate from the counts"
    assert ">-0.750<" in section


def test_the_tracker_shares_the_single_week_swap_indicator(tmp_path: Path) -> None:
    """Both blocks sit inside the swap target and point at its skeleton (UI-SPEC E4 loading)."""
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=2,
            wins=1,
            losses=1,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=-0.05,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=2,
            wins=1,
            losses=1,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=0.05,
        ),
    ]
    with _client_with_tracker(tmp_path, blocks, "swap_target") as client:
        full = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
        fragment = client.get(
            f"/bets?season={_SEASON}&week={_WEEK}", headers={"HX-Request": "true"}
        ).text

    for key, section in _tracker_sections(full).items():
        assert 'hx-indicator="#bets-loading"' in section, (
            f"tracker section {key} does not share the week swap indicator"
        )
    # The fragment IS the swap target, so both tracker sections must survive into it.
    assert set(_tracker_sections(fragment)) == {
        "backtest_replay:contaminated",
        "forward:forward_realized",
    }, "the tracker sits outside the week swap target"


def _render_strip(outcomes: list[str], block: dict[str, int]) -> str:
    """Render the result-strip macro on its own, through the app's template environment."""
    module = app_templates.env.get_template("components/_result_strip.html").module
    return str(module.result_strip(outcomes, block))  # pyright: ignore[reportAttributeAccessIssue]


def test_the_result_strip_renders_only_when_it_agrees_with_its_block() -> None:
    """Review Focus 4: a strip that disagrees with the stored block renders NOTHING.

    The tiles read the precomputed block; the marks read the graded rows. A population run
    interrupted between the two writes leaves them out of step, and two disagreeing records on one
    page would each contradict the other. The stored tiles stay the authority.
    """
    block = {"bets_graded": 3, "wins": 2, "losses": 1, "pushes": 0}

    assert _render_strip([], block).strip() == "", "an empty class drew a strip"
    assert _render_strip(["win", "win", "win"], block).strip() == "", (
        "a strip whose wins disagree with the stored block was drawn"
    )
    # Every named count agrees but the length does not: a value outside the grading vocabulary
    # slipped in. Still a disagreement, still no strip.
    assert _render_strip(["win", "win", "loss", "pending"], block).strip() == ""

    agreeing = _render_strip(["win", "loss", "win"], block)
    assert agreeing.count("data-result-mark") == 3
    assert "wins 2, losses 1, pushes 0" in agreeing
    assert agreeing.count("bg-green-500") == 2
    assert agreeing.count("bg-red-500") == 1


def test_the_result_strip_draws_one_mark_per_graded_bet_in_its_own_section(
    tmp_path: Path,
) -> None:
    """Each class's strip draws that class's graded bets and no other class's."""
    rows = [
        _graded_row("2023_W01_DET@KC", "win"),
        _graded_row("2023_W01_CAR@ATL", "loss"),
        _graded_row("2023_W01_CIN@CLE", "win"),
        _graded_row("2023_W01_JAX@IND", "push"),
        _graded_row("2023_W01_DEN@LVR", "win", pair=_FORWARD_CLASS),
    ]
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=2,
            losses=1,
            pushes=1,
            hit_rate=2 / 3,
            flat_return_units=0.1,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=1,
            wins=1,
            losses=0,
            pushes=0,
            hit_rate=1.0,
            flat_return_units=0.909,
        ),
    ]
    with _client_with_tracker(tmp_path, blocks, "strip_agrees", rows=rows) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    sections = _tracker_sections(body)
    replay = sections["backtest_replay:contaminated"]
    forward = sections["forward:forward_realized"]
    assert replay.count("data-result-mark") == 4
    assert "wins 2, losses 1, pushes 1" in replay
    assert ">2-1-1<" in replay, "the record line does not restate the stored counts"
    assert forward.count("data-result-mark") == 1
    assert "wins 1, losses 0, pushes 0" in forward
    # Each mark carries its outcome's colour: green won, red lost, grey push. Counted by colour
    # class, so a push drawn red (or a loss drawn grey) fails.
    replay_marks = _mark_colours(replay)
    assert replay_marks.count("bg-green-500") == 2
    assert replay_marks.count("bg-red-500") == 1
    assert replay_marks.count("bg-[#5B6478]") == 1
    assert _mark_colours(forward) == ["bg-green-500"]


def _mark_colours(section: str) -> list[str]:
    """The background colour class of every mark in *section*'s result strip, in order."""
    strip = section[section.index("data-result-strip") :]
    strip = strip[: strip.index("</div>")]
    return [
        next(c for c in classes.split() if c.startswith("bg-"))
        for classes in re.findall(r'<span data-result-mark class="([^"]*)"', strip)
    ]


def test_a_result_strip_that_disagrees_with_its_stored_block_is_omitted(
    tmp_path: Path,
) -> None:
    """The page-level half of Review Focus 4: no strip, and the stored tiles still render."""
    rows = [
        _graded_row("2023_W01_DET@KC", "win"),
        _graded_row("2023_W01_CAR@ATL", "win"),
        _graded_row("2023_W01_CIN@CLE", "win"),
        _graded_row("2023_W01_JAX@IND", "loss"),
    ]
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=2,
            losses=1,
            pushes=1,
            hit_rate=2 / 3,
            flat_return_units=0.1,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "strip_disagrees", rows=rows) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    replay = _tracker_sections(body)["backtest_replay:contaminated"]
    assert "data-result-strip" not in replay
    assert "data-result-mark" not in replay
    # The STORED block stays the authority: its tiles and its record line still render.
    assert "data-figure-value>4<" in replay
    assert "Hit rate" in replay
    assert ">2-1-1<" in replay
    # Without the strip's label, the record line alone must say what "2-1-1" means to a screen
    # reader, from the same stored counts.
    record = replay[replay.index("data-tracker-record") :]
    record = record[: record.index("</span></span>")]
    assert '<span aria-hidden="true">2-1-1</span>' in record
    assert '<span class="sr-only">2 wins, 1 loss, 1 push' in record


# ---------------------------------------------------------------------------
# The provenance badge (plan 31-16 Task 2, SPEC R8, D31-22, UI-SPEC E10)
# ---------------------------------------------------------------------------
#
# ONE partial keyed on the two orthogonal D31-22 columns owns the honesty-label vocabulary. The
# labels and their class sets are transcribed here as LITERALS from the design contract rather than
# imported from the template, so a silent edit to either side is a failure and not a tautology.

_VALIDATION_LABELS: dict[str, str] = {
    "contaminated": "Contaminated split",
    "clean_holdout": "Old rule -- 2025, not evidence",
    "forward_realized": "Live forward record",
}
_VALIDATION_CLASSES: dict[str, str] = {
    "contaminated": "evidence-chip",
    "clean_holdout": "evidence-chip",  # review WR-06: "not evidence" takes the plain chip
    "forward_realized": "evidence-chip",
}
_UNKNOWN_VALIDATION_TYPE = "validation_type_invented_by_a_future_plan"


def _badges(body: str) -> list[str]:
    """Every rendered provenance badge, located by its two data attributes."""
    return re.findall(
        r"<span [^>]*data-provenance=\"[^\"]*\"[^>]*>.*?</span></span>", body
    )


def test_every_validation_type_renders_its_declared_label_and_classes(
    tmp_path: Path,
) -> None:
    """All three validation types render the contract's label and its class set.

    One assertion per type, driven off the transcribed table, so adding a fourth type to the
    template without adding it here leaves the new type unproven rather than silently covered.
    """
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=2,
            losses=2,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=-0.02,
        ),
        _block(
            _CLEAN_HOLDOUT,
            bets_graded=4,
            wins=3,
            losses=1,
            pushes=0,
            hit_rate=0.75,
            flat_return_units=0.03,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=4,
            wins=2,
            losses=2,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=0.01,
        ),
    ]
    with _client_with_tracker(tmp_path, blocks, "badge_all_types") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    sections = _tracker_sections(body)
    for pair, key in (
        (_CONTAMINATED, "backtest_replay:contaminated"),
        (_CLEAN_HOLDOUT, "backtest_replay:clean_holdout"),
        (_FORWARD_CLASS, "forward:forward_realized"),
    ):
        validation_type = pair[1]
        section = sections[key]
        assert _VALIDATION_LABELS[validation_type] in section, (
            f"{validation_type} rendered no label on its block header"
        )
        # The WHOLE class attribute, anchored on the data-provenance that follows it: a bare
        # "evidence-chip" would also match inside "evidence-chip evidence-chip-strong".
        assert (
            f'class="{_VALIDATION_CLASSES[validation_type]}" data-provenance="{pair[0]}"'
            in section
        ), f"{validation_type} rendered without its declared class set"
        assert f'data-validation-type="{validation_type}"' in section
        assert f'data-provenance="{pair[0]}"' in section


def test_an_unknown_validation_type_renders_the_raw_code(tmp_path: Path) -> None:
    """A type outside the fixed three renders its RAW CODE -- not a blank, not a fallback label.

    Matches the suppression-label rule: a newly added member of either vocabulary must be visible
    rather than silent.
    """
    row = _live_row("2023_W01_DET@KC", "ou")
    row["validation_type"] = _UNKNOWN_VALIDATION_TYPE
    with _client_with_tracker(tmp_path, [], "badge_unknown_type", rows=[row]) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    badges = _badges(body)
    assert badges, "no provenance badge rendered on the live row"
    assert any(f">{_UNKNOWN_VALIDATION_TYPE}<" in badge for badge in badges), (
        f"the unknown validation type did not render its raw code: {badges}"
    )
    for label in _VALIDATION_LABELS.values():
        assert label not in body, (
            f"an unknown validation type fell back to the {label!r} label"
        )


def test_an_impossible_pair_renders_the_raw_code_rather_than_mislabelling_it(
    tmp_path: Path,
) -> None:
    """The lookup is keyed on the PAIR, so a forward row carrying a replay type is not mislabelled.

    Keying on validation_type alone would confidently print "Contaminated split" beside a forward
    provenance -- a label asserting the row was reconstructed after the fact when the other column
    says it was recommended before kickoff. The pair keying makes that contradiction visible.
    """
    row = _live_row("2023_W01_DET@KC", "ou")
    row["provenance"] = "forward"
    row["validation_type"] = "contaminated"
    with _client_with_tracker(
        tmp_path, [], "badge_impossible_pair", rows=[row]
    ) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    badges = _badges(body)
    assert badges, "no provenance badge rendered on the live row"
    assert any(">contaminated<" in badge for badge in badges), (
        f"the impossible pair did not render its raw code: {badges}"
    )
    assert "Contaminated split" not in body, (
        "a forward row was labelled with the replay vocabulary"
    )


def test_the_badge_markup_carries_no_green_amber_or_red_class(tmp_path: Path) -> None:
    """Monochrome, for the reason the EV band badge is monochrome (UI-SPEC Deviation 1)."""
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=2,
            losses=2,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=-0.02,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=4,
            wins=2,
            losses=2,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=0.01,
        ),
    ]
    with _client_with_tracker(tmp_path, blocks, "badge_monochrome") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    badges = _badges(body)
    assert badges, "no provenance badge rendered"
    for badge in badges:
        for classes in _class_attributes(badge):
            for forbidden in ("green", "amber", "red"):
                assert forbidden not in classes, (
                    f"a provenance badge carries the {forbidden} hue: {classes}"
                )


def test_the_badge_appears_on_both_tracker_block_headers(tmp_path: Path) -> None:
    """A response carrying both classes carries a badge on each block header."""
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=2,
            losses=2,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=-0.02,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=4,
            wins=2,
            losses=2,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=0.01,
        ),
    ]
    with _client_with_tracker(tmp_path, blocks, "badge_both_headers") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    sections = _tracker_sections(body)
    for key, validation_type in (
        ("backtest_replay:contaminated", "contaminated"),
        ("forward:forward_realized", "forward_realized"),
    ):
        header = sections[key][: sections[key].index("</div>")]
        assert f'data-validation-type="{validation_type}"' in header, (
            f"{key} carries no provenance badge on its header"
        )


def test_the_badge_renders_on_every_displayed_row(tmp_path: Path) -> None:
    """R8's every-displayed-row clause: a live row AND a suppressed row both carry the labels."""
    rows = [
        _live_row("2023_W01_DET@KC", "ou"),
        _suppressed_row("2023_W01_CAR@ATL", "wp", "ev_below_floor"),
    ]
    with _client_with_tracker(tmp_path, [], "badge_every_row", rows=rows) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    # The live list is a column of bet slips and the suppressed list a table: each carries its
    # own provenance slot, so both halves still label every displayed row.
    assert body.count("Provenance</th>") == 1, (
        "the suppressed table does not carry its provenance column"
    )
    assert body.count("data-slip-provenance") == 1, (
        "the live bet slip does not carry its provenance slot"
    )
    # Two rows, two badges -- and both rows are in the document even though the disclosure is
    # collapsed, so the suppressed row's labels survive into an HTML export too.
    assert len(_badges(body)) == 2, (
        f"expected one badge per displayed row: {_badges(body)}"
    )
    assert body.count("Contaminated split") == 2


def test_exactly_one_partial_owns_the_validation_type_vocabulary() -> None:
    """A search finds no SECOND source of the three labels anywhere under web/templates.

    The one-registry-never-two-lists rule. A second spelling of these strings is the drift this
    repository has already paid for once, and it is the failure a grep -- not a render -- catches.
    """
    templates = Path(__file__).resolve().parents[2] / "web" / "templates"
    for label in _VALIDATION_LABELS.values():
        sources = sorted(
            path.relative_to(templates).as_posix()
            for path in templates.rglob("*.html")
            if label in path.read_text(encoding="utf-8")
        )
        assert sources == ["components/_provenance_badge.html"], (
            f"the label {label!r} is spelled in more than one template: {sources}"
        )
    found = sorted(p.name for p in COMPONENTS_DIR.glob("*provenance*.html"))
    assert found == ["_provenance_badge.html"], (
        f"a second provenance partial appeared: {found}"
    )


# ---------------------------------------------------------------------------
# The SCOPED hard-block, judged game by game against each game's own lock (33.2 review C2 CR-02)
# ---------------------------------------------------------------------------
#
# WHAT CHANGED. Plan 31-18 compared the per-week populated-at marker against the week's LATEST
# per-game lock. Under the daily pre-lock run every population finishes BEFORE that lock by
# design, so the current week was withheld for its whole live window. The verdict is now per game:
# a game with a bet-list row is built, a game past its lock with none is missing (named), a game
# whose lock is ahead is pending. The week is refused only when games have locked and NONE has a
# row -- the failed insertion.
#
# THE LOCKS COME FROM THE SCHEDULE, NOT FROM THE BET ROWS (REVIEW-STALE). The failure being
# guarded is a MISSING bet-list insertion, and in that state the week may have NO rows at all. A
# guard reading its own threshold out of the data it is checking could not fire in the one case it
# was built for, and would render a blank list as an honest empty week. That is exactly what
# ``test_the_zero_row_case_renders_the_refusal_and_not_the_zero_admitted_state`` pins.

_REFUSAL_BRANCH_FUNCTION = "_bet_week_coverage"
_REFUSAL_REQUIRED_GETTERS = frozenset({"get_bet_game_locks", "get_bet_game_ids"})
# ``get_cache_meta`` is the generic-timestamp source D31-29 rejected; ``get_bet_list`` is the
# ranked live list, which is empty for a week whose games were all suppressed and so cannot say
# whether a game was evaluated. Neither may decide the verdict.
_REFUSAL_FORBIDDEN_GETTERS = frozenset({"get_cache_meta", "get_bet_list"})


def _service_calls_in(function_name: str) -> set[str]:
    """Every attribute called on the ``service`` object inside *function_name* in pages.py.

    STRUCTURAL, not a text search. A comment or docstring naming the rejected sources -- and this
    module's own code names all four -- cannot trip it, while an actual call to one of them does.
    """
    import api.routes.pages as pages_module

    tree = ast.parse(Path(pages_module.__file__).read_text(encoding="utf-8"))
    target = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == function_name
        ),
        None,
    )
    assert target is not None, (
        f"api/routes/pages.py defines no function named {function_name!r}; the refusal branch "
        "moved and this guard has gone stale rather than passing"
    )
    return {
        node.func.attr
        for node in ast.walk(target)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "service"
    }


def test_the_refusal_branch_reads_the_schedule_locks_and_the_week_coverage() -> None:
    """The verdict is computed from the per-game schedule locks and the rows' game ids only."""
    called = _service_calls_in(_REFUSAL_BRANCH_FUNCTION)

    missing = _REFUSAL_REQUIRED_GETTERS - called
    assert not missing, (
        f"{_REFUSAL_BRANCH_FUNCTION} does not call {sorted(missing)}; the verdict is not being "
        "computed from each game's schedule-derived lock and the week's bet-list coverage"
    )
    forbidden = _REFUSAL_FORBIDDEN_GETTERS & called
    assert not forbidden, (
        f"{_REFUSAL_BRANCH_FUNCTION} calls {sorted(forbidden)}. get_cache_meta is the generic "
        "timestamp D31-29 rejected (it advances when ANY table is repopulated); get_bet_list is "
        "the ranked live list, empty for a week whose games were all suppressed. Neither may "
        "decide the verdict."
    )


def test_the_structural_guard_is_not_vacuous() -> None:
    """The scan finds real calls, so an empty result cannot pass as 'no forbidden getters'."""
    assert _service_calls_in(_REFUSAL_BRANCH_FUNCTION), (
        "the AST scan found no service calls at all in the refusal branch; it would report any "
        "forbidden getter as absent"
    )
    # And it CAN see a forbidden name: _build_bets_context legitimately calls get_cache_meta for
    # the footer, so a scan that never reports it is a scan that is not looking.
    assert "get_cache_meta" in _service_calls_in("_build_bets_context")


def _blocked_cache(
    tmp_path: Path,
    name: str,
    *,
    bet_rows: list[dict[str, Any]],
    freeze_weeks: list[dict[str, Any]],
    week_rows: list[dict[str, Any]],
    stamp: tuple[int, int] | None = None,
    stamp_last_updated: bool = True,
) -> Path:
    """Build a cache with explicit control over the rows, the freeze table and the marker."""
    clear_cache()
    db_path = tmp_path / f"{name}.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        if bet_rows:
            materialize_bet_list(conn, pd.DataFrame(bet_rows))
        materialize_available_bet_weeks(conn, pd.DataFrame(week_rows))
        if freeze_weeks:
            materialize_bet_week_freeze(conn, pd.DataFrame(freeze_weeks))
        if stamp is not None:
            _stamp_populated_at(conn, _POPULATED_AT, season=stamp[0], week=stamp[1])
        elif stamp_last_updated:
            stamped_at = datetime(2023, 9, 8, 22, 30, 0)
            conn.executemany(
                "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
                [["last_updated", stamped_at.isoformat(), stamped_at]],
            )
    finally:
        conn.close()
    return db_path


def test_the_zero_row_case_renders_the_refusal_and_not_the_zero_admitted_state(
    tmp_path: Path,
) -> None:
    """THE failure the guard exists for: the week is scheduled, has NO bet rows, and no marker.

    A version reading its threshold from the bet rows could not produce this render -- there are no
    rows to read a freeze from -- and would show a blank list as though the week had honestly
    admitted nothing. That is a claim about the models made from the absence of an insertion.
    """
    db_path = _blocked_cache(
        tmp_path,
        "zero_rows",
        bet_rows=[],
        freeze_weeks=[
            {
                "game_id": "2023_W01_DET@KC",
                "season": _SEASON,
                "week": _WEEK,
                "game_freeze_ts": _LATER_FREEZE,
            }
        ],
        week_rows=[{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}],
        stamp=None,
    )

    with contextmanager(_client)(db_path) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    body = response.text
    assert _HARD_BLOCK_MESSAGE in body, (
        "a scheduled week with a known freeze, ZERO bet rows and no populated-at marker rendered "
        "something other than the refusal; this is precisely the failed-insertion state D31-29 "
        "designed the marker to catch"
    )
    assert _ZERO_ADMITTED_HEADING not in body, (
        "the failed insertion was reported as 'no bets cleared the floor', which is a claim about "
        "the models made from the absence of an insertion"
    )
    assert _BANNER_EYEBROW in body


def test_a_week_with_no_freeze_row_at_all_renders_the_no_current_week_state(
    tmp_path: Path,
) -> None:
    """No schedule -> no freeze threshold and no selectable week -> the empty state, not a refusal.

    An absent expected freeze means the week is not in the schedule at all, and a refusal there
    would assert that a cache is out of date relative to a line freeze nothing recorded. In
    production the two schedule-derived tables cannot disagree -- see the test below -- so this is
    the only shape the absence takes.
    """
    db_path = _blocked_cache(
        tmp_path,
        "no_freeze",
        bet_rows=[],
        freeze_weeks=[],
        week_rows=[],
        stamp=None,
    )

    with contextmanager(_client)(db_path) as client:
        body = client.get("/bets").text

    assert _NO_CURRENT_WEEK_HEADING in body
    assert _HARD_BLOCK_MESSAGE not in body, (
        "the page refused a week it has no freeze threshold for; the refusal would be asserting "
        "staleness against a line freeze nothing recorded"
    )


def test_the_two_schedule_derived_tables_carry_the_same_weeks(tmp_path: Path) -> None:
    """``available_bet_weeks`` and ``bet_week_freeze`` are built from ONE frame, so they agree.

    This is the premise the routing above rests on: a week reachable through the selector always
    has a freeze threshold, so "absent from the freeze table" can only mean "absent from the
    schedule". Asserted through the real ``populate_cache``, not by reading the source.
    """
    from api.cache import populate_cache

    empty = tmp_path / "empty"
    empty.mkdir()
    db_path = tmp_path / "agree.duckdb"
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2023_W01_DET@KC",
                "season": _SEASON,
                "week": _WEEK,
                "game_freeze_ts": _LATER_FREEZE,
            },
            {
                "game_id": "2023_W02_AAA@BBB",
                "season": _SEASON,
                "week": _EMPTY_WEEK,
                "game_freeze_ts": _LATER_FREEZE,
            },
        ]
    )
    populate_cache(
        db_path=db_path,
        artifacts_dir=empty,
        outputs_dir=empty,
        gold_dir=empty,
        silver_dir=empty,
        bet_schedule_df=schedule,
    )

    conn = duckdb.connect(str(db_path), read_only=True)
    try:
        weeks = set(
            conn.execute("SELECT season, week FROM available_bet_weeks").fetchall()
        )
        freezes = set(
            conn.execute("SELECT season, week FROM bet_week_freeze").fetchall()
        )
        game_locks = set(
            conn.execute("SELECT game_id, season, week FROM bet_game_lock").fetchall()
        )
    finally:
        conn.close()

    assert weeks == freezes == {(_SEASON, _WEEK), (_SEASON, _EMPTY_WEEK)}, (
        f"the two schedule-derived tables disagree: navigation {sorted(weeks)} vs freeze "
        f"{sorted(freezes)}. A selectable week with no freeze threshold would resolve to the "
        "no-current-week state while plainly being a current week."
    )
    # The per-game locks the /bets verdict reads come from the SAME frame (33.2 review C2 CR-02).
    assert game_locks == {
        ("2023_W01_DET@KC", _SEASON, _WEEK),
        ("2023_W02_AAA@BBB", _SEASON, _EMPTY_WEEK),
    }


def test_a_past_week_renders_normally_in_the_same_response_shape_as_a_blocked_one(
    tmp_path: Path,
) -> None:
    """The refusal is SCOPED: one week is refused while another is served, from ONE cache.

    Week 1's game has a list; week 2's locked game has none. Requesting week 2 refuses;
    requesting week 1 in the same cache serves the list. Blocking the whole page was rejected --
    it punishes the reader for an unrelated failure and trains people to ignore the guard.
    """
    early_freeze = datetime(2023, 9, 8, 22, 0, 0, tzinfo=UTC)  # BEFORE _POPULATED_AT
    db_path = _blocked_cache(
        tmp_path,
        "scoped",
        bet_rows=[_live_row("2023_W01_DET@KC", "ou")],
        freeze_weeks=[
            {
                "game_id": "2023_W01_DET@KC",
                "season": _SEASON,
                "week": _WEEK,
                "game_freeze_ts": early_freeze,
            },
            {
                "game_id": "2023_W02_AAA@BBB",
                "season": _SEASON,
                "week": _EMPTY_WEEK,
                "game_freeze_ts": _LATER_FREEZE,
            },
        ],
        week_rows=[
            {"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK},
            {"game_id": "2023_W02_AAA@BBB", "season": _SEASON, "week": _EMPTY_WEEK},
        ],
        stamp=(_SEASON, _WEEK),
    )

    with contextmanager(_client)(db_path) as client:
        blocked = client.get(f"/bets?season={_SEASON}&week={_EMPTY_WEEK}").text
        served = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert _HARD_BLOCK_MESSAGE in blocked, (
        "week 2's locked game has no list, yet the week was not refused"
    )
    assert _HARD_BLOCK_MESSAGE not in served, (
        "week 1 was refused although its game has a list; the refusal is not scoped to the week "
        "that failed"
    )
    assert "Stake (units)" in served
    # The replay tracker survives the refusal: a replay figure does not depend on the current
    # week's line freeze.
    assert _REPLAY_HEADING in blocked
    assert _FORWARD_WITHHELD_MESSAGE in blocked
    assert _BANNER_EYEBROW in blocked


def test_the_recovery_text_names_the_full_sequence_and_both_timestamps(
    tmp_path: Path,
) -> None:
    """The reader can CHECK the claim rather than take it (UI-SPEC E7 error).

    WHY THIS TEST WAS ITSELF THE COVERAGE GAP (plan 31-23, G-31-123b). Its old name and its old
    failure message -- "the refusal does not name the command that fixes it" -- asserted that a
    command STRING appeared in the response body. That is a strictly weaker claim than the one the
    reader needs, which is that FOLLOWING the named command reaches a served list. The command it
    pinned was ``scripts/populate_cache.py``, a pure copy step over an artifact that from a cold
    start does not exist; so this assertion was green while the refusal pointed the reader back at
    the very command that had put them in the refusal. It now requires the FULL sequence, in
    order, with the copy-only clause. The end-to-end claim it cannot make -- that following the
    copy actually serves a list -- lives in ``tests/api/test_cold_start_bet_list_recovery.py``.
    """
    db_path = _blocked_cache(
        tmp_path,
        "recovery",
        # No bet row for the week's locked game: the failed insertion (33.2 review C2 CR-02). The
        # marker is still stamped, so the refusal has a populated-at value to show.
        bet_rows=[],
        freeze_weeks=[
            {
                "game_id": "2023_W01_DET@KC",
                "season": _SEASON,
                "week": _WEEK,
                "game_freeze_ts": _LATER_FREEZE,
            }
        ],
        week_rows=[{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}],
        stamp=(_SEASON, _WEEK),
    )

    with contextmanager(_client)(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert _GENERATE_COMMAND in body, (
        "the refusal does not name the FULL sequence that fixes it -- the command that PRODUCES "
        "the rows is missing, so the only action offered is the copy step the reader has already "
        "run, which is the loop G-31-123b reported"
    )
    assert _POPULATE_COMMAND in body, (
        "the refusal does not name the command that loads the produced rows into the cache"
    )
    assert body.index(_GENERATE_COMMAND) < body.index(_POPULATE_COMMAND), (
        "the refusal names the two recovery commands in the WRONG order; the copy step run first "
        "is what leaves the cache with no rows and no marker, i.e. still refused"
    )
    assert _COPY_ONLY_CLAUSE in body, (
        "the refusal does not state that the second command only copies what the first produces"
    )
    assert "no restart is needed" in body, (
        "the refusal does not say the running server picks the rebuilt cache up on the next "
        "request; without it a reader has no way to know the recovery took effect (plan 31-20)"
    )
    assert _POPULATED_AT in body, (
        "the refusal does not interpolate the populated-at timestamp"
    )
    assert str(_LATER_FREEZE) in body, (
        "the refusal does not interpolate the latest game freeze"
    )
    # The list and the suppressed disclosure are both withheld, and nothing else is.
    assert "Stake (units)" not in body
    assert "Suppressed candidates" not in body


# ---------------------------------------------------------------------------
# DEF-31-16: base.html's footer guards the KEY, not the dict (plan 31-18 owns this)
# ---------------------------------------------------------------------------
#
# ``web/templates/base.html`` renders the footer timestamp as
# ``cache_meta.last_updated if cache_meta else 'Unknown'``. The guard tests the DICT. An EMPTY
# cache_meta takes the else branch and renders "Unknown", which is why this never fired. A
# NON-EMPTY cache_meta that happens not to carry ``last_updated`` takes the FIRST branch,
# ``cache_meta.last_updated`` resolves to Jinja ``Undefined``, and ``api/dependencies.py``'s
# ``format_datetime`` calls ``.strftime`` on it -- ``UndefinedError``, a 500 on EVERY page on the
# site, not just the one being built.
#
# It was unreachable while ONE writer populated cache_meta and always wrote all three keys
# together. Plan 31-18 adds a SECOND, INDEPENDENT writer (the per-week bet-list marker), which is
# exactly what makes the shape reachable: a partial rebuild, a resumed run or a hand-built cache
# can now carry marker rows without ``last_updated``. Fixed here, and proven with a render that
# 500s before the fix.

_FOOTER_UNKNOWN = "Unknown"


def _cache_meta_without_last_updated(tmp_path: Path, name: str) -> Path:
    """A cache whose ``cache_meta`` has ROWS but no ``last_updated`` -- the DEF-31-16 shape."""
    clear_cache()
    db_path = tmp_path / f"{name}.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(conn, pd.DataFrame([_live_row("2023_W01_DET@KC", "ou")]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}]
            ),
        )
        stamped_at = datetime(2023, 9, 8, 22, 30, 0)
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            [[bet_list_populated_at_key(_SEASON, _WEEK), _POPULATED_AT, stamped_at]],
        )
        rows = conn.execute("SELECT key FROM cache_meta").fetchall()
    finally:
        conn.close()

    keys = {row[0] for row in rows}
    assert keys and "last_updated" not in keys, (
        f"the fixture does not carry the DEF-31-16 shape (rows, no last_updated): {sorted(keys)}"
    )
    return db_path


def test_a_cache_meta_with_rows_but_no_last_updated_does_not_500_the_bets_page(
    tmp_path: Path,
) -> None:
    """DEF-31-16: the footer guard is on the KEY, so a missing key renders 'Unknown'."""
    db_path = _cache_meta_without_last_updated(tmp_path, "def_31_16_bets")

    with contextmanager(_client)(db_path) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200, (
        "a cache_meta carrying rows but no last_updated 500'd the page. base.html's footer guard "
        "tests the DICT rather than the KEY, so cache_meta.last_updated resolves to Jinja "
        "Undefined and format_datetime calls .strftime on it (DEF-31-16)"
    )
    assert _FOOTER_UNKNOWN in response.text, (
        "the footer rendered without naming the missing timestamp; a blank would hide the absence"
    )
    assert "Data updated:" in response.text


def test_the_same_cache_meta_does_not_500_any_other_page_either(
    tmp_path: Path,
) -> None:
    """The blast radius is the WHOLE SITE, not /bets: the footer is in the shared base template.

    Asserted across every top-nav route, because the defect's severity is exactly that it is not
    scoped to the page whose data is missing.
    """
    db_path = _cache_meta_without_last_updated(tmp_path, "def_31_16_site")

    with contextmanager(_client)(db_path) as client:
        for route in (
            "/",
            "/season",
            "/track-record",
            "/how-it-works",
        ):
            response = client.get(route)
            assert response.status_code == 200, (
                f"{route} returned {response.status_code} for a cache_meta with rows but no "
                "last_updated; the footer read is unguarded on the key (DEF-31-16)"
            )
            assert _FOOTER_UNKNOWN in response.text


def test_the_footer_still_renders_the_timestamp_when_it_is_present(
    tmp_path: Path,
) -> None:
    """The control: the fix must not turn every footer into 'Unknown'.

    Without this, a guard that dropped the value entirely would satisfy the two tests above while
    silently removing the cache stamp from every page on the site.
    """
    clear_cache()
    db_path = tmp_path / "footer_present.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(conn, pd.DataFrame([_live_row("2023_W01_DET@KC", "ou")]))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}]
            ),
        )
        _stamp_populated_at(conn, _POPULATED_AT)
    finally:
        conn.close()

    with contextmanager(_client)(db_path) as client:
        response = client.get(f"/bets?season={_SEASON}&week={_WEEK}")

    assert response.status_code == 200
    body = response.text
    assert 'data-utc="2023-09-08T22:30:00"' in body, (
        "the footer no longer renders the cache timestamp it does have; the DEF-31-16 fix "
        "swallowed the value instead of guarding the key"
    )
    assert "Sep 08, 2023" in body


# ---------------------------------------------------------------------------
# WR-07: a malformed query parameter degrades, it does not 500
# ---------------------------------------------------------------------------


_MALFORMED_PARAMS = [
    "--5",  # lstrip("-") stripped BOTH hyphens, then int("--5") raised
    "\u00b2",  # "2".isdigit() is True, int("2") raises
    "\u00bd",  # a vulgar fraction: isdigit() False but isnumeric() True
    "abc",  # the case the docstring already promised
    "",  # an empty selector value
    " 12 ",  # whitespace the guard used to require be absent
    "1_2",  # int() accepts underscores in literals but not with a leading digit group here
    "9" * 400,  # absurdly long, still an int
]


@pytest.mark.parametrize("raw", _MALFORMED_PARAMS)
def test_a_malformed_week_degrades_to_the_default_rather_than_500ing(
    bets_client: TestClient, raw: str
) -> None:
    """``_parse_int_param``'s own docstring promised this and the guard contradicted it.

    ``candidate = raw.strip().lstrip("-")`` strips ALL leading hyphens, so ``"--5"`` passed as
    ``"5".isdigit()`` and ``int("--5")`` then raised -- an unhandled ValueError, a 500 with a stack
    trace on a public page, from a two-character query string.
    """
    response = bets_client.get(f"/bets?week={raw}")

    assert response.status_code == 200, (
        f"/bets?week={raw!r} returned {response.status_code}; an unparseable value degrades to "
        "the dynamic default, it does not raise"
    )
    assert _BANNER_EYEBROW in response.text


@pytest.mark.parametrize("raw", _MALFORMED_PARAMS)
def test_a_malformed_season_degrades_on_both_pages(
    bets_client: TestClient, raw: str
) -> None:
    """``season_tracking_page`` inlined the identical broken guard, so it 500'd identically."""
    assert bets_client.get(f"/bets?season={raw}").status_code == 200
    assert bets_client.get(f"/season?season={raw}").status_code == 200


def test_a_well_formed_negative_week_is_still_parsed_and_then_whitelisted_away(
    bets_client: TestClient,
) -> None:
    """The control: the fix must not turn every value into None and make the tests vacuous."""
    from api.routes.pages import _parse_int_param

    assert _parse_int_param("-5") == -5
    assert _parse_int_param(str(_WEEK)) == _WEEK
    assert _parse_int_param("--5") is None
    assert _parse_int_param(None) is None
    # A parsed-but-unavailable week still resolves to a real scheduled week (T-31-01).
    assert bets_client.get("/bets?week=-5").status_code == 200


# ---------------------------------------------------------------------------
# 33.2 review C2 WR-01: /bets opens on the current slate, not the season's last week
# ---------------------------------------------------------------------------


def _three_week_cache(
    tmp_path: Path, name: str, *, slate: str | None, built_week: int
) -> Path:
    """Weeks 1-3 of ``_SEASON`` scheduled, one game with a list in *built_week* only."""
    from api.cache import CURRENT_SLATE_KEY

    clear_cache()
    db_path = tmp_path / f"{name}.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(
            conn,
            pd.DataFrame(
                [_live_row(f"2023_W0{built_week}_DET@KC", "ou", week=built_week)]
            ),
        )
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": f"2023_W0{week}_DET@KC",
                        "season": _SEASON,
                        "week": week,
                    }
                    for week in (1, 2, 3)
                ]
            ),
        )
        if slate is not None:
            conn.execute(
                "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
                [CURRENT_SLATE_KEY, slate, datetime(2023, 9, 10, 12, 0)],
            )
    finally:
        conn.close()
    return db_path


def test_bets_opens_on_the_stamped_current_slate(tmp_path: Path) -> None:
    """With a current slate stamped, a bare /bets opens on that week -- not on week 3."""
    db_path = _three_week_cache(tmp_path, "slate", slate=f"{_SEASON}:2", built_week=1)
    with contextmanager(_client)(db_path) as client:
        body = client.get("/bets").text

    assert f"Live bets -- {_SEASON} Week 2" in body


def test_bets_without_a_slate_opens_on_the_latest_built_week(tmp_path: Path) -> None:
    """No slate (offseason, or an older cache): the latest week with a list, not the last week."""
    db_path = _three_week_cache(tmp_path, "no_slate", slate=None, built_week=2)
    with contextmanager(_client)(db_path) as client:
        body = client.get("/bets").text

    assert f"Live bets -- {_SEASON} Week 2" in body
    assert f"Live bets -- {_SEASON} Week 3" not in body


def test_an_explicit_week_still_wins_over_the_slate(tmp_path: Path) -> None:
    """The slate is a DEFAULT only: a valid requested week passes through unchanged."""
    db_path = _three_week_cache(
        tmp_path, "explicit", slate=f"{_SEASON}:2", built_week=1
    )
    with contextmanager(_client)(db_path) as client:
        body = client.get(f"/bets?season={_SEASON}&week=3").text

    assert f"Live bets -- {_SEASON} Week 3" in body


def _silver_two_weeks(silver_dir: Path) -> Path:
    silver_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "game_id": "2023_W01_DET@KC",
                "season": 2023,
                "week": 1,
                "kickoff_et": pd.Timestamp("2023-09-08 00:20", tz="UTC"),
                "game_type": "REG",
            },
            {
                "game_id": "2023_W02_MIN@PHI",
                "season": 2023,
                "week": 2,
                "kickoff_et": pd.Timestamp("2023-09-15 00:15", tz="UTC"),
                "game_type": "REG",
            },
        ]
    ).to_parquet(silver_dir / "games.parquet", index=False)
    return silver_dir


def test_population_stamps_the_in_season_slate_and_nothing_in_the_offseason(
    tmp_path: Path,
) -> None:
    """The stamp comes from utils.current_slate over the population's own silver schedule."""
    from api.cache import CURRENT_SLATE_KEY, parse_current_slate, stamp_current_slate

    silver = _silver_two_weeks(tmp_path / "silver")
    conn = duckdb.connect(":memory:")
    try:
        conn.execute(
            "CREATE TABLE cache_meta (key VARCHAR PRIMARY KEY, value VARCHAR, updated_at TIMESTAMP)"
        )
        in_season = stamp_current_slate(
            conn, silver, datetime(2023, 9, 10, 16, 0, tzinfo=UTC)
        )
        stored = conn.execute(
            "SELECT value FROM cache_meta WHERE key = ?", [CURRENT_SLATE_KEY]
        ).fetchone()
        assert in_season == "2023:2"
        assert stored is not None and parse_current_slate(stored[0]) == (2023, 2)

        conn.execute("DELETE FROM cache_meta")
        offseason = stamp_current_slate(
            conn, silver, datetime(2023, 7, 1, 16, 0, tzinfo=UTC)
        )
        assert offseason is None
        assert conn.execute("SELECT COUNT(*) FROM cache_meta").fetchone()[0] == 0
    finally:
        conn.close()

    assert parse_current_slate("garbage") is None
    assert parse_current_slate(None) is None


def test_no_rendered_page_states_the_retired_friday_rule(
    tmp_path: Path, bets_client: TestClient
) -> None:
    """33.2 review C2 WR-10: /bets and /season no longer state the retired Friday rule.

    Since D33.2-01 each game locks at 6:00 PM Eastern the day before its own kickoff and the run
    is daily. The pages still told readers that lines froze "on the Friday before" each game and
    that the forward record was "written to the cache on the Friday before its game".
    """
    rows = [_suppressed_row("2023_W01_SEA@SFO", "ats", "stale_line")]
    with _client_for(tmp_path, rows, "friday_suppressed") as client:
        suppressed = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    blocked = _blocked_cache(
        tmp_path,
        "friday_blocked",
        bet_rows=[],
        freeze_weeks=[
            {
                "game_id": "2023_W01_DET@KC",
                "season": _SEASON,
                "week": _WEEK,
                "game_freeze_ts": _LATER_FREEZE,
            }
        ],
        week_rows=[{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}],
        stamp=(_SEASON, _WEEK),
    )
    with contextmanager(_client)(blocked) as client:
        refused = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    bodies = {
        "/bets": bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text,
        "/bets suppressed": suppressed,
        "/bets refused": refused,
        "/season": bets_client.get("/season").text,
    }
    for page, body in bodies.items():
        assert "friday" not in body.lower(), (
            f"{page} still states the retired Friday rule"
        )
    assert _PER_GAME_FREEZE_SENTENCE in bodies["/bets"]


def test_every_tracker_section_closes_every_div_it_opens(tmp_path: Path) -> None:
    """33.2 review C2 IN-02: the tracker section's heading row closed only one of its two divs.

    The caption landed inside the flex row and the figures inside the header block until the
    closing ``</section>`` auto-closed it. Checked on every rendered section, populated and empty.
    """
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=2,
            losses=2,
            pushes=0,
            hit_rate=0.5,
            flat_return_units=-0.1,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "balanced_divs") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    sections = _tracker_sections(body)
    assert sections, "no tracker section rendered"
    for key, markup in sections.items():
        section = markup[: markup.index("</section>")]
        assert section.count("<div") == section.count("</div>"), (
            f"the {key} tracker section leaves a <div> unclosed"
        )


def test_no_ranked_count_header_without_a_current_week(tmp_path: Path) -> None:
    """A populated bet list with no current week shows only the no-current-week state.

    With season and week both None the getter carries no week filter, so a count in the header
    would tally rows across the whole cache above the "No current week" heading.
    """
    clear_cache()
    db_path = tmp_path / "no_week_populated.duckdb"
    _build_cache(db_path, [_live_row("2023_W01_DET@KC", "ou")], week_rows=[])
    with contextmanager(_client)(db_path) as client:
        body = client.get("/bets").text

    assert _NO_CURRENT_WEEK_HEADING in body
    assert "ranked by expected value &middot;" not in body
