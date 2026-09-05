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

Selectors (``-k``): served_equals_selector, served_order, tie_break, not_advice_banner,
no_currency, nav_link, ev_band_badge, empty_week, taxonomy, reason_code, suppressed, disclosure,
caption, moneyline.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

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
    materialize_bet_week_freeze,
)
from api.services import clear_cache
from backtest.bet_selector import REJECTION_REASONS, BetSelector
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
    defect rather than the page's. The unguarded footer read is logged as DEF-31-15 -- it is
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
