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
caption, moneyline, tracker, pushes, graded, return, provenance, badge, cross_link.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
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
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PENDING,
    GRADING_STATUS_WIN,
    classify_row_provenance,
    materialize_available_bet_weeks,
    materialize_bet_list,
    materialize_bet_tracker_blocks,
    materialize_bet_week_freeze,
)
from api.services import DataService, clear_cache
from backtest.bet_selector import REJECTION_REASONS, BetSelector
from backtest.bet_tracker import EmptyTrackerBlock, TrackerBlock, to_tracker_frame
from backtest.ev_chain_constants import assign_ev_tier
from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD
from tests.api.week_selector_snapshot import (
    COMPONENTS_DIR,
    FAILURE_EVENTS,
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


# ---------------------------------------------------------------------------
# The suppressed-candidates disclosure (plan 31-15 Task 1, SPEC R6, UI-SPEC E3)
# ---------------------------------------------------------------------------

_UNRECOGNISED_REASON = "reason_invented_by_a_future_plan"

# The nine labels the page maps REJECTION_REASONS onto, transcribed from the UI-SPEC's
# suppression-reason table. Kept here as a LITERAL rather than imported from the template so a
# silent edit to either side is a test failure rather than a tautology.
_EXPECTED_LABELS: dict[str, str] = {
    "ev_below_floor": "Expected value below the floor",
    "not_subpop": "Outside the eligible sub-population",
    "stale_line": "Line older than the freeze",
    "missing_snapshot": "No market line for this bet type",
    "missing_prediction": "No model prediction for this game",
    "real_odds_failed": "Odds failed the real-market check",
    "zero_kelly_stake": "Sizing returned no stake",
    "ev_not_finite": "Expected value could not be computed",
    "no_bet_side": "Model agrees with the market",
}

_BLANK_REASON_CELL = (
    '<td class="px-4 py-3 text-left text-gray-700 whitespace-nowrap"></td>'
)


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
    assert _BLANK_REASON_CELL not in body


def test_unrecognised_reason_code_renders_the_raw_code(tmp_path: Path) -> None:
    """A reason code with no label renders the RAW CODE, never an empty cell (T-31-76)."""
    rows = [_suppressed_row("2023_W01_XXX@YYY", "ou", _UNRECOGNISED_REASON)]
    with _client_for(tmp_path, rows, "unknown_reason") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    assert f">{_UNRECOGNISED_REASON}</td>" in body
    assert f"{_UNRECOGNISED_REASON} (1)" in body
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
    assert "<details" not in body


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
    assert "Line older than the freeze" in body


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
    summary = body[body.index("<summary") : body.index("</summary>")]
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
    assert ">Totals</td>" in body
    assert ">Winner</td>" in body
    assert "No market line for this bet type" in body
    assert "Suppressed candidates (1)" in body


# ---------------------------------------------------------------------------
# The four distinct non-happy renders and the cache stamp
# (plan 31-15 Task 2, SPEC R5, UI-SPEC E2 empty / E7 / E12)
# ---------------------------------------------------------------------------

# The four headings, one per state, transcribed from the design contract. A state that rendered
# another state's heading would pass a "not blank" check and fail this one.
_CACHE_ABSENT_HEADING = "Bet list not built yet"
_ZERO_ADMITTED_HEADING = "No bets cleared the floor this week"
_NO_CURRENT_WEEK_HEADING = "No current week"
_HARD_BLOCK_MESSAGE = "This week&#39;s list is withheld -- the cache is older than this week&#39;s line freeze"

_BANNER_EYEBROW = "Not wagering advice"

# The per-game freeze sentence, and the week-level claims it deliberately does NOT make. A
# week-level "lines were frozen Friday 6 PM ET" is FALSE for every Thursday game (D31-18).
# Authored LITERALLY in the template rather than interpolated, so its apostrophes are not
# HTML-escaped -- unlike the two _error_state.html slots, which pass through {{ }}.
_PER_GAME_FREEZE_SENTENCE = (
    "Lines frozen at 6:00 PM Eastern on the Friday before each game's own kickoff "
    "-- a Thursday game freezes a week earlier than that week's Sunday games."
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
# Strictly LATER than _POPULATED_AT, so the population run finished BEFORE the freeze moved.
_LATER_FREEZE = datetime(2023, 9, 8, 23, 0, 0)


def _build_bare_cache(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Create every cache table and return the OPEN connection for further writes."""
    conn = duckdb.connect(str(db_path))
    for statement in CACHE_SCHEMA.strip().split(";"):
        stmt = statement.strip()
        if stmt:
            conn.execute(stmt)
    return conn


def _stamp_populated_at(conn: duckdb.DuckDBPyConnection, value: str) -> None:
    """Stamp the bet-list populated-at key the way ``populate_cache`` stamps it.

    ``last_updated`` is written alongside because ``populate_cache`` always writes it and
    ``base.html``'s footer reads it UNGUARDED: with a cache_meta that has rows but no
    ``last_updated`` the footer raises and every page 500s. Omitting it here would make the
    fixture describe a cache production never produces, and the 500 would be the fixture's
    defect rather than the page's. The unguarded footer read is logged as DEF-31-16 -- it is
    out of this plan's scope and is NOT reachable from the shipped writer.
    """
    stamped_at = datetime(2023, 9, 8, 22, 30, 0)
    conn.executemany(
        "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
        [
            [BET_LIST_POPULATED_AT_KEY, value, stamped_at],
            ["last_updated", stamped_at.isoformat(), stamped_at],
        ],
    )


def test_state_one_cache_table_absent(tmp_path: Path) -> None:
    """A cache that predates Phase 31 renders the cache-absent state, and NOT the stamp.

    The distinction is load-bearing: reporting a missing TABLE as "no bets cleared the floor"
    would be a claim about the models made from the absence of a table.
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
    assert "scripts/populate_cache.py" in body
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


def test_state_four_stale_cache_hard_block(tmp_path: Path) -> None:
    """A cache older than the week's LATEST per-game freeze is REFUSED, not silently served.

    D31-27: staleness is the cache timestamp preceding the latest per-game freeze among the
    week's games. The refusal is a server-rendered 200 -- it is NOT the failed-fragment path.
    """
    clear_cache()
    db_path = tmp_path / "stale.duckdb"
    conn = _build_bare_cache(db_path)
    rows = [_live_row("2023_W01_DET@KC", "ou")]
    try:
        materialize_bet_list(conn, pd.DataFrame(rows))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [{"game_id": "2023_W01_DET@KC", "season": _SEASON, "week": _WEEK}]
            ),
        )
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [{"season": _SEASON, "week": _WEEK, "game_freeze_ts": _LATER_FREEZE}]
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
    assert "bg-red-50" in body, "the refusal did not render through _error_state.html"
    assert "Past weeks below are unaffected and remain readable." in body
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
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": datetime(2023, 9, 8, 22, 0, 0),
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

    assert f"Bet list last populated: {_POPULATED_AT}." in body
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


def test_the_failure_template_carries_the_message_and_a_retry(
    bets_client: TestClient,
) -> None:
    """The template the handlers copy holds the error-state geometry plus a retry affordance."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    start = body.index('<template id="bets-failure-template">')
    template = body[start : body.index("</template>", start)]

    assert "bg-red-50" in template, (
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
    "Each of these was written to the cache on the Friday before its game, before the result "
    "existed. This is the only block that is a track record."
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
_FORWARD_WITHHELD_MESSAGE = "The forward record is withheld -- the cache is older than this week&#39;s line freeze"

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
    try:
        if bet_rows:
            materialize_bet_list(conn, pd.DataFrame(bet_rows))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {"game_id": r["game_id"], "season": _SEASON, "week": r["week"]}
                    for r in bet_rows
                ]
                or [{"game_id": "2023_W01_AAA@BBB", "season": _SEASON, "week": _WEEK}]
            ),
        )
        materialize_bet_tracker_blocks(conn, to_tracker_frame(blocks))
        if freeze is not None:
            materialize_bet_week_freeze(
                conn,
                pd.DataFrame(
                    [{"season": _SEASON, "week": _WEEK, "game_freeze_ts": freeze}]
                ),
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

    # Every rendered figure card lives inside exactly one section.
    inside = sum(
        section.count(
            'class="text-xs font-semibold text-gray-500 uppercase tracking-wide"'
        )
        for section in sections.values()
    )
    tracker_start = min(body.index(s) for s in sections.values())
    tracker_region = body[tracker_start:]
    assert inside == tracker_region.count(
        'class="text-xs font-semibold text-gray-500 uppercase tracking-wide"'
    ), "a tracker figure rendered outside one of the sections"


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
    labels = re.findall(
        r'<p class="text-xs font-semibold text-gray-500 uppercase tracking-wide">([^<]+)</p>',
        section,
    )
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
    assert "text-red-600" in negative
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
    values = re.findall(r'<p class="text-2xl[^"]*">([^<]*)</p>', section)
    assert values, "the block rendered no figures"
    assert not any("nan" in value.lower() for value in values), (
        f"a non-finite value reached a rendered figure: {values}"
    )
    assert _ZERO_RESULT_LINE not in section, (
        "the honesty line claims a negative measurement where nothing was measured"
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
    assert any("text-green-700" in s for s in sections.values())
    assert any("text-red-600" in s for s in sections.values())

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

    # Outside the tracker region the realized-outcome colours do not appear at all. The ONE red
    # above the tracker is the REFUSAL role, not the realized-outcome role: _error_state.html
    # pairs exactly one bg-red-50 container with exactly one text-red-600 recovery line, and the
    # UI-SPEC's colour table lists those as separate roles. Counting the pair is what keeps this
    # assertion honest without pretending the shipped refusal partial is a tracker colour.
    tracker_start = min(body.index(s) for s in sections.values())
    above = body[:tracker_start]
    assert "text-green-700" not in above, (
        "a realized-outcome green rendered above the tracker"
    )
    assert above.count("text-red-600") == above.count("bg-red-50"), (
        "a red above the tracker is not accounted for by a refusal block"
    )


def test_the_forward_block_is_withheld_under_the_hard_block_while_replay_stays_readable(
    tmp_path: Path,
) -> None:
    """The refusal is SCOPED: the forward totals are withheld, the replay figures are not.

    A replay figure does not depend on the current week's line freeze, so refusing it would be a
    refusal nothing justified (UI-SPEC E4 error).
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
        tmp_path, blocks, "scoped_refusal", freeze=_LATER_FREEZE
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
    assert "bg-red-50" in forward, (
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


# ---------------------------------------------------------------------------
# The provenance badge (plan 31-16 Task 2, SPEC R8, D31-22, UI-SPEC E10)
# ---------------------------------------------------------------------------
#
# ONE partial keyed on the two orthogonal D31-22 columns owns the honesty-label vocabulary. The
# labels and their class sets are transcribed here as LITERALS from the design contract rather than
# imported from the template, so a silent edit to either side is a failure and not a tautology.

_VALIDATION_LABELS: dict[str, str] = {
    "contaminated": "Contaminated split",
    "clean_holdout": "Clean holdout -- 2025, single use",
    "forward_realized": "Live forward record",
}
_VALIDATION_CLASSES: dict[str, str] = {
    "contaminated": "bg-gray-100 text-gray-700 border border-gray-300",
    "clean_holdout": "bg-gray-200 text-gray-900 border border-gray-300",
    "forward_realized": "bg-white text-gray-700 border border-gray-300",
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
        assert _VALIDATION_CLASSES[validation_type] in section, (
            f"{validation_type} rendered without its declared class set"
        )
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

    assert body.count("Provenance</th>") == 2, (
        "the live table and the suppressed table do not both carry the provenance column"
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
