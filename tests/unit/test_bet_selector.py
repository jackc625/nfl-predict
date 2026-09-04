"""Unit tests for the Phase-27 O/U BetSelector (BET-01/02, OUM-04/06) and the LOCKED-1
pre-hold high-total boundary derivation.

Covers:
- Task 1 (LOCKED-1): the high-total boundary RE-DERIVED on PRE-HOLD (2018-2022) closing totals,
  the hold-season leakage assertion, and the legacy-46.5 comparison
  (``backtest.ou_divergence.derive_high_total_boundary`` / ``HIGH_TOTAL_BOUNDARY_PREHOLD``).
- Task 2/3 (BET-01/02, OUM-04/06, D27-04/05/06/14): the single-source ``BetSelector.select()``
  decision engine -- sub-pop UNION filter, EV-floor admission, high-total-OVER pocket drop, the
  BET-02 calibrated-P Kelly sizing fix (model_prob <= 1 on an 8-point gap), the mock/synthetic-odds
  hard-fail, push handling, report-only CLV with the wording distinction, the unfiltered
  cross-check, and selected+rejected-with-reasons output.
- Phase 31, plan 31-06 (D31-01/02): the D31-01 split into a target-agnostic core plus per-target
  strategies behind the UNCHANGED facade. The O/U path is held to a RECORDED pre-refactor output
  (commit 6190f92) rather than to "the other tests still pass", the EV floor is pinned at exact
  equality (SPEC R1 adjacency), and ``OUStrategy``'s delegation to the LOCKED simulator convention
  is proven with a stand-in simulator.

A TRAP FOR THE NEXT GUARD AUTHOR -- key on the MODULE PATH, never on the class name.

  There are TWO classes named ``BetSelector`` and both are loaded in EVERY process. The live one
  is ``backtest.bet_selector.BetSelector``; the other is a DEAD v1.0 class at
  ``utils/bet_selector.py``, which ``utils/__init__.py`` re-exports into the ``utils`` namespace
  and lists in ``utils.__all__``. Because ``backtest/bet_selector.py`` does ``from utils import
  get_logger``, importing the live class EXECUTES ``utils/__init__`` and pulls in the dead one as
  a side effect -- they are never separable at runtime. A structural guard that asks "is there a
  second selection path?" by matching the class NAME will therefore always find the dead cluster
  and always false-positive; it must match on ``node.module`` (``backtest.bet_selector`` vs
  ``utils`` / ``utils.bet_selector``) instead. The converse trap is equally real: the simulator
  receives the selector by INJECTION (``BettingSimulator(..., ou_bet_selector=...)``), so there is
  no import edge to find and an import-graph test alone cannot prove the production path routes
  through the real selector. Plan 31-17 owns the one-path guard that rests on both halves of this.

Run the boundary group only:  pytest tests/unit/test_bet_selector.py -q -k boundary
Run the full module:          pytest tests/unit/test_bet_selector.py -x -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

from backtest.ou_divergence import (
    HIGH_TOTAL_BOUNDARY_PREHOLD,
    HOLD_SEASONS,
    PRE_HOLD_SEASONS,
    TOTALS_REGIME_BOUNDARIES,
    HoldSeasonLeakageError,
    derive_high_total_boundary,
)

# ---------------------------------------------------------------------------
# Number anchors (kept explicit so a silent drift is caught).
# ---------------------------------------------------------------------------

_LEGACY_HIGH_BOUNDARY = 46.5  # the hold-informed TOTALS_REGIME_BOUNDARIES["high_min"]
_OU_BREAKEVEN = 110.0 / 210.0  # 0.52380952... (flat -110 cover breakeven)


def _ou_row(
    game_id: str,
    *,
    model_total: float,
    closing_total: float,
    actual_total: float,
    season: int = 2021,
    week: int = 1,
    sportsbook: str = "consensus",
    is_live: bool = False,
) -> dict:
    """Build a single synthetic O/U candidate row for the BetSelector tests."""
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "model_total": model_total,
        "closing_total": closing_total,
        "actual": actual_total,
        "sportsbook": sportsbook,
        "is_live": is_live,
    }


# ---------------------------------------------------------------------------
# Task 1 (LOCKED-1): pre-hold high-total boundary derivation + leakage assertion
# ---------------------------------------------------------------------------


class TestHighTotalBoundaryDerivation:
    """The high-total boundary is re-derived on PRE-HOLD data only (LOCKED-1, T-27-22)."""

    def test_high_total_boundary_excludes_hold(self) -> None:
        """The derivation reads only pre-hold rows; a hold-season row raises (leakage assertion).

        Feeding a frame that contains a 2023/2024 (HOLD) row must raise HoldSeasonLeakageError --
        the eligibility boundary may never be informed by the burned holdout. A clean pre-hold
        frame derives the boundary from its rows only.
        """
        # (a) A frame carrying a hold-season row is rejected (the leakage assertion fires).
        leaky = pd.DataFrame(
            {
                "game_id": [
                    "2019_W01_DAL@NYG",
                    "2023_W05_KC@BUF",  # HOLD season -> must trip the assertion
                ],
                "total": [44.0, 49.0],
            }
        )
        with pytest.raises(HoldSeasonLeakageError) as exc:
            derive_high_total_boundary(odds_df=leaky)
        assert "2023" in str(exc.value)

        # (b) A clean pre-hold-only frame derives from its rows only (no hold contamination).
        clean = pd.DataFrame(
            {
                "game_id": [
                    "2018_W01_DAL@NYG",
                    "2019_W02_KC@BUF",
                    "2020_W03_SF@SEA",
                    "2021_W04_GB@CHI",
                    "2022_W05_NE@MIA",
                    "2022_W06_LA@ARI",
                ],
                "total": [40.0, 42.0, 45.0, 48.0, 50.0, 52.0],
            }
        )
        derived = derive_high_total_boundary(odds_df=clean)
        # Upper-tertile (q=2/3) of the six pre-hold totals -- a value drawn ONLY from these rows.
        expected = float(
            pd.Series([40.0, 42.0, 45.0, 48.0, 50.0, 52.0]).quantile(2.0 / 3.0)
        )
        assert derived == pytest.approx(expected)

    def test_high_total_boundary_rejects_hold_window_request(self) -> None:
        """Requesting a derivation window that intersects HOLD_SEASONS is itself a leakage error."""
        with pytest.raises(HoldSeasonLeakageError):
            derive_high_total_boundary(pre_hold_seasons=(2022, 2023))

    def test_high_total_boundary_value_reported_vs_legacy(self) -> None:
        """HIGH_TOTAL_BOUNDARY_PREHOLD is a sane totals value, surfaced alongside the legacy 46.5.

        The new pre-hold boundary need NOT equal the legacy 46.5 (it is re-derived on a different,
        leakage-clean window); the test records both values so the readout can compare them, and
        asserts the new value sits in a sane NFL-totals range.
        """
        new_value = HIGH_TOTAL_BOUNDARY_PREHOLD
        legacy_value = TOTALS_REGIME_BOUNDARIES["high_min"]

        assert legacy_value == _LEGACY_HIGH_BOUNDARY
        # The constant must have resolved on the populated data lake (not the NaN bare-checkout
        # fallback) for the unit suite, and sit in a sane totals band.
        assert not math.isnan(new_value), (
            "HIGH_TOTAL_BOUNDARY_PREHOLD did not resolve -- the silver odds lake is required for "
            "the pre-hold derivation (LOCKED-1)."
        )
        assert 40.0 <= new_value <= 50.0
        # Both values are surfaced together (the readout comparison); the pre-hold derivation is a
        # leakage-clean replacement for the hold-informed legacy boundary, not necessarily equal.
        reported = {"pre_hold": new_value, "legacy_hold_informed": legacy_value}
        assert set(reported) == {"pre_hold", "legacy_hold_informed"}

    def test_pre_hold_window_excludes_hold_seasons(self) -> None:
        """The pre-hold derivation window and the hold window are disjoint (LOCKED-1 invariant)."""
        assert set(PRE_HOLD_SEASONS).isdisjoint(set(HOLD_SEASONS))
        assert set(HOLD_SEASONS) == {2023, 2024}


# ---------------------------------------------------------------------------
# Task 2/3: BetSelector single-source decision engine (BET-01/02, OUM-04/06)
#
# These tests target the not-yet-built ``backtest.bet_selector`` module (RED until Task 3).
# Synthetic, unit-fast frames only (never the real artifact). The selector API under test is a
# BetSelector constructed with frozen_sd, season_bias_by_season (prior-season walk-forward bias,
# negative for an over-biased model), ev_floor_t (the EV-floor scalar t Plan 04 tunes), bankroll,
# and high_total_boundary (the pre-hold value). Its select(candidates, raw_odds_df=None) returns a
# result that exposes: selected, rejected (each with a rejection_reason), filtered (the eligible
# acceptance basis), unfiltered (the whole-population cross-check), and clv_report. Each per-bet
# record carries at least the game id, season, week, bet side, totals regime, model and closing
# totals, the calibrated P(side), the per-bet EV, the Kelly stake, the sub-pop label, the
# push-aware outcome, and the report-only CLV. The _make_selector helper below builds the selector
# with the fixture SD and per-season bias.
# ---------------------------------------------------------------------------

# A frozen SD + per-season bias used across the BetSelector fixtures. Bias is NEGATIVE (the
# over-biased model, D26-18) so the correction pulls totals DOWN.
_FIXTURE_SD = 13.0
_FIXTURE_BIAS = {2021: -1.0, 2022: -2.0}


def _make_selector(ev_floor_t: float = 0.0, **kwargs):
    """Construct a BetSelector with the fixture SD/bias (import deferred -- RED until Task 3)."""
    from backtest.bet_selector import BetSelector

    return BetSelector(
        frozen_sd=_FIXTURE_SD,
        season_bias_by_season=_FIXTURE_BIAS,
        ev_floor_t=ev_floor_t,
        bankroll=10_000.0,
        high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
        **kwargs,
    )


class TestBetSelectorSubpopFilter:
    """The HARD eligibility gate is the UNION {under picks} OR {high-total} (D27-04/05)."""

    def test_subpop_filter_admits_only_under_or_high_total(self) -> None:
        """Eligible set = exactly {bet_side==under} OR {totals_regime==high}; low-total over out.

        A low-total OVER row (model predicts more points than a low line) is NOT in the union and
        must be rejected with reason ``not_subpop``. An under pick and a high-total over are both
        eligible (the high-total over may still be dropped later by the EV floor -- that is a
        different reason).
        """
        candidates = [
            # under pick (model below line) -- eligible via the UNDER arm of the union.
            _ou_row(
                "2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0
            ),
            # high-total over (line above the pre-hold boundary) -- eligible via the HIGH arm.
            _ou_row(
                "2021_W02_C@D", model_total=58.0, closing_total=50.0, actual_total=55.0
            ),
            # low-total over (model above a LOW line) -- NOT in the union -> not_subpop.
            _ou_row(
                "2021_W03_E@F", model_total=42.0, closing_total=38.0, actual_total=41.0
            ),
        ]
        result = _make_selector().select(candidates)

        eligible_ids = {r["game_id"] for r in result.filtered}
        assert "2021_W01_A@B" in eligible_ids  # under pick
        assert "2021_W02_C@D" in eligible_ids  # high-total over
        assert "2021_W03_E@F" not in eligible_ids  # low-total over excluded

        not_subpop = {
            r["game_id"]
            for r in result.rejected
            if r["rejection_reason"] == "not_subpop"
        }
        assert "2021_W03_E@F" in not_subpop


class TestBetSelectorEvFloor:
    """Within the eligible set, a bet is admitted iff per-bet EV >= the EV-floor t (D27-14)."""

    def test_ev_floor_decision_within_subpop(self) -> None:
        """An eligible row with EV < t is rejected (ev_below_floor) though it is in the sub-pop.

        Set a HIGH ev_floor_t so a marginal-but-eligible under pick (small positive EV) falls below
        the floor and is rejected with reason ``ev_below_floor`` -- proving the floor acts WITHIN
        the eligible set, not as a sub-pop filter.
        """
        # A barely-under pick: model just below the line -> small positive EV that a high floor cuts.
        candidates = [
            _ou_row(
                "2021_W01_A@B", model_total=44.0, closing_total=45.0, actual_total=43.0
            ),
        ]
        # A very high floor (0.9) exceeds any realistic per-bet EV -> the eligible bet is rejected.
        result = _make_selector(ev_floor_t=0.9).select(candidates)

        assert all(r["game_id"] != "2021_W01_A@B" for r in result.selected)
        reasons = {r["game_id"]: r["rejection_reason"] for r in result.rejected}
        assert reasons.get("2021_W01_A@B") == "ev_below_floor"


class TestBetSelectorHighTotalOverPocket:
    """The high-total OVER over-bias pocket is dropped by the calibrated EV chain (D27-05)."""

    def test_high_total_over_pocket_dropped(self) -> None:
        """High-total OVER picks with a low bias-corrected P(over) are REJECTED; UNDERs survive.

        RESEARCH Finding 4 / D27-05: the high-total OVER pocket grades below breakeven. After
        bias-correction and the half-point slippage against the OVER side, those picks have
        per-bet EV < 0 and are rejected by the EV floor (t=0). A high-total UNDER pick survives.
        Asserts the surviving high-total OVER count is 0 while high-total UNDER survive.
        """
        candidates = [
            # high-total OVER trap picks: model barely over a high line; correction + slippage
            # pushes the calibrated P(over) below breakeven (negative EV).
            _ou_row(
                "2022_W01_A@B", model_total=52.0, closing_total=50.0, actual_total=49.0
            ),
            _ou_row(
                "2022_W02_C@D", model_total=53.0, closing_total=51.0, actual_total=50.0
            ),
            # high-total UNDER pick: model well below a high line -> strong P(under), survives.
            _ou_row(
                "2022_W03_E@F", model_total=44.0, closing_total=51.0, actual_total=43.0
            ),
        ]
        result = _make_selector(ev_floor_t=0.0).select(candidates)

        selected_over_high = [
            r
            for r in result.selected
            if r["bet_side"] == "over" and r["totals_regime"] == "high"
        ]
        selected_under_high = [
            r
            for r in result.selected
            if r["bet_side"] == "under" and r["totals_regime"] == "high"
        ]
        assert len(selected_over_high) == 0  # the over-bias pocket is dropped
        assert len(selected_under_high) >= 1  # high-total UNDER survive


class TestBetSelectorBet02SizingFix:
    """BET-02: Kelly consumes the calibrated P(side), never implied + points-distance."""

    def test_bet02_kelly_model_prob_le_one_on_8pt_gap(self) -> None:
        """On an 8-point model_total-line gap, the Kelly model_prob is a probability <= 1.0.

        Tightening #7: a high-total OVER with model_total - line == 8.0 must hand Kelly a
        calibrated probability in (0, 1], NOT the legacy points-distance value
        ``implied + 8.0 == 8.524``. Asserts the recorded calibrated_p_side is <= 1.0 and is NOT
        equal to implied + 8.0.
        """
        from utils.probability_utils import moneyline_to_probability

        candidates = [
            _ou_row(
                "2022_W01_A@B", model_total=58.0, closing_total=50.0, actual_total=55.0
            ),
        ]
        result = _make_selector(ev_floor_t=0.0).select(candidates)

        # The candidate is eligible (high-total) and present in the filtered set with a calibrated
        # probability handed to Kelly.
        rec = next(r for r in result.filtered if r["game_id"] == "2022_W01_A@B")
        p_side = rec["calibrated_p_side"]
        assert 0.0 < p_side <= 1.0

        implied = moneyline_to_probability(-110)
        legacy_buggy = implied + 8.0  # the points-distance-as-probability bug (~8.524)
        assert p_side != pytest.approx(legacy_buggy)


class TestBetSelectorRealOddsGuard:
    """OUM-06: mock/synthetic odds hard-fail BEFORE any selection."""

    def test_mock_odds_hard_fail(self) -> None:
        """assert_real_odds raises ValueError naming offenders on a bad sportsbook or is_live=True.

        A clean consensus/draftkings, is_live=False frame passes. A frame with a sportsbook outside
        {consensus, draftkings} OR an is_live=True row raises a ValueError naming the offending
        game_ids (OUM-06). The selector calls this BEFORE selecting.
        """
        from backtest.bet_selector import assert_real_odds

        clean = pd.DataFrame(
            {
                "game_id": ["2021_W01_A@B", "2021_W02_C@D"],
                "sportsbook": ["consensus", "draftkings"],
                "is_live": [False, False],
            }
        )
        # Clean frame: no raise.
        assert_real_odds(clean)

        # Bad sportsbook.
        bad_book = pd.DataFrame(
            {
                "game_id": ["2021_W03_E@F"],
                "sportsbook": ["bovada"],
                "is_live": [False],
            }
        )
        with pytest.raises(ValueError, match="2021_W03_E@F"):
            assert_real_odds(bad_book)

        # is_live=True row.
        live = pd.DataFrame(
            {
                "game_id": ["2021_W04_G@H"],
                "sportsbook": ["consensus"],
                "is_live": [True],
            }
        )
        with pytest.raises(ValueError, match="2021_W04_G@H"):
            assert_real_odds(live)

    def test_select_hard_fails_on_mock_raw_odds(self) -> None:
        """select() raises if a mock raw_odds_df is supplied (provenance check BEFORE selection)."""
        candidates = [
            _ou_row(
                "2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0
            ),
        ]
        mock_raw = pd.DataFrame(
            {
                "game_id": ["2021_W01_A@B"],
                "sportsbook": ["mock_book"],
                "is_live": [False],
            }
        )
        with pytest.raises(ValueError, match="2021_W01_A@B"):
            _make_selector().select(candidates, raw_odds_df=mock_raw)


class TestBetSelectorPushHandling:
    """Tightening #8: pushes are carried via the LOCKED _resolve_ou_outcome, never coerced."""

    def test_push_carried_on_record(self) -> None:
        """A row that grades to a PUSH carries outcome=None (a push marker), not win/loss.

        Construct an under pick whose actual total equals the slipped line exactly so
        _resolve_ou_outcome returns None (push). The BetSelector record must carry outcome None
        (the documented push marker), NOT coerced to True/False.
        """
        # under pick: closing 45.0, slipped (under) -> 44.5; actual == 44.5 is a push.
        candidates = [
            _ou_row(
                "2021_W01_A@B",
                model_total=40.0,
                closing_total=45.0,
                actual_total=44.5,  # equals the under-slipped line 44.5 -> push
            ),
        ]
        result = _make_selector(ev_floor_t=0.0).select(candidates)

        rec = next(
            (r for r in result.selected if r["game_id"] == "2021_W01_A@B"),
            None,
        )
        assert rec is not None, (
            "the under pick should be selected (eligible + positive EV)"
        )
        assert rec["bet_side"] == "under"
        assert rec["outcome"] is None  # push marker, never coerced to win/loss


class TestBetSelectorClvReportOnly:
    """D27-06/12 (tightening #11): CLV is computed/reported but NEVER a selection gate."""

    def test_clv_reported_not_gated_and_wording_distinct(self) -> None:
        """CLV (the model-edge line_clv) is attached per bet; selection is unaffected by its value.

        Tightening #11: the BetSelector computes ``compute_line_clv(model_total, closing_total,
        direction="total")`` -- the MODEL EDGE vs the line (the non-zero line_clv), DISTINCT from
        the separate freeze-vs-close forward metric (structurally ~0). A bet with a "bad"
        (negative) line_clv that still clears the EV floor is STILL admitted: CLV never gates
        (D27-06). The clv value is recorded per bet and a report-only summary is exposed.
        """
        # An under pick with model_total > closing_total would give a positive line_clv; here the
        # under pick has model_total < closing_total -> NEGATIVE line_clv, yet it clears the floor
        # and must STILL be selected (CLV does not gate).
        candidates = [
            _ou_row(
                "2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0
            ),
        ]
        result = _make_selector(ev_floor_t=0.0).select(candidates)

        rec = next(r for r in result.selected if r["game_id"] == "2021_W01_A@B")
        # line_clv = model_total - closing_total = 38 - 45 = -7 (negative), yet admitted.
        assert rec["clv"] == pytest.approx(38.0 - 45.0)
        assert rec["game_id"] in {r["game_id"] for r in result.selected}

        # A report-only CLV summary is exposed (mean / significance), NOT used to gate.
        assert result.clv_report is not None
        assert "mean" in result.clv_report


class TestBetSelectorCrossCheck:
    """D27-04: the unfiltered whole-population result is a REPORTED cross-check only."""

    def test_unfiltered_cross_check_reported(self) -> None:
        """select() reports BOTH the filtered (acceptance basis) and unfiltered population.

        Both ``filtered`` and ``unfiltered`` are present; the filtered set excludes the ineligible
        (low-total OVER) rows that the unfiltered set includes.
        """
        candidates = [
            _ou_row(
                "2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0
            ),
            _ou_row(
                "2021_W03_E@F", model_total=42.0, closing_total=38.0, actual_total=41.0
            ),
        ]
        result = _make_selector(ev_floor_t=0.0).select(candidates)

        filtered_ids = {r["game_id"] for r in result.filtered}
        unfiltered_ids = {r["game_id"] for r in result.unfiltered}
        # The low-total OVER is in the unfiltered cross-check but NOT in the filtered basis.
        assert "2021_W03_E@F" in unfiltered_ids
        assert "2021_W03_E@F" not in filtered_ids
        assert "2021_W01_A@B" in filtered_ids


class TestBetSelectorSelectedAndRejected:
    """select() returns selected AND rejected eligible candidates with rejection reasons."""

    def test_select_returns_selected_and_rejected_with_reasons(self) -> None:
        """Rejection reasons cover {not_subpop, ev_below_floor, real_odds_failed} as applicable.

        Construct one offender per reason and assert each reason appears:
        - not_subpop: a low-total OVER (outside the union).
        - ev_below_floor: an eligible pick with EV under a high floor.
        - real_odds_failed: an offender row in the raw odds frame (named per-game).
        """
        # not_subpop offender + an eligible-but-floored offender. Use a high floor so the eligible
        # under pick is rejected by the floor (ev_below_floor) while the low-total over is
        # rejected as not_subpop.
        candidates = [
            _ou_row(
                "2021_W01_A@B", model_total=44.0, closing_total=45.0, actual_total=43.0
            ),
            _ou_row(
                "2021_W03_E@F", model_total=42.0, closing_total=38.0, actual_total=41.0
            ),
        ]
        result = _make_selector(ev_floor_t=0.9).select(candidates)
        reasons = {r["rejection_reason"] for r in result.rejected}
        assert "not_subpop" in reasons
        assert "ev_below_floor" in reasons

        # real_odds_failed: a per-game offender surfaced as a named rejection (or a raised
        # ValueError naming the offender). The selector exposes the reason taxonomy explicitly.
        from backtest.bet_selector import REJECTION_REASONS

        assert {"not_subpop", "ev_below_floor", "real_odds_failed"} <= set(
            REJECTION_REASONS
        )


# ---------------------------------------------------------------------------
# Phase 27 code-review fixes (WR-03 / WR-05 / WR-07)
# ---------------------------------------------------------------------------


class TestPhase27ReviewFixes:
    """Regression coverage for the WR-03/05/07 review fixes."""

    def test_nan_high_total_boundary_rejected(self) -> None:
        """WR-03: a non-finite high_total_boundary hard-fails construction.

        Without the guard, ``closing_total > NaN`` is always False, silently collapsing the
        under-OR-high UNION to under-only. The selector must refuse to construct.
        """
        from backtest.bet_selector import BetSelector

        with pytest.raises(ValueError, match="finite"):
            BetSelector(
                frozen_sd=_FIXTURE_SD,
                season_bias_by_season=_FIXTURE_BIAS,
                high_total_boundary=float("nan"),
            )

    def test_assert_real_odds_missing_game_id_raises_valueerror(self) -> None:
        """WR-05: an offending row on a frame missing 'game_id' raises a named ValueError.

        The provenance hard-fail must not degrade to a bare ``KeyError`` when the frame lacks
        a game_id column; it names offenders by row index instead and still raises ValueError.
        """
        from backtest.bet_selector import assert_real_odds

        df = pd.DataFrame(
            {"sportsbook": ["bovada"], "is_live": [False]}
        )  # bad book, no game_id
        with pytest.raises(ValueError, match="provenance"):
            assert_real_odds(df)

    def test_admitted_bet_zeroed_by_kelly_is_rejected_not_booked(self) -> None:
        """WR-07: a bet admitted by a negative EV floor but zeroed by the Kelly -110 gate is
        rejected with 'zero_kelly_stake', never booked as a zero-stake 'selected' bet.
        """
        selector = _make_selector(
            ev_floor_t=-1.0
        )  # a negative floor admits a sub-breakeven side
        # calibrated_p_side 0.50 is below the -110 breakeven (0.5238): admitted (per_bet_ev -0.02
        # exceeds the -1.0 floor) but the inner Kelly calculator zeroes the stake.
        zeroed = {
            "game_id": "Z",
            "season": 2023,
            "week": 5,
            "bet_side": "under",
            "calibrated_p_side": 0.50,
            "per_bet_ev": -0.02,
        }
        selected: list[dict] = []
        rejected: list[dict] = []
        selector._admit_and_size_week([zeroed], selected, rejected)
        assert selected == []
        assert len(rejected) == 1
        assert rejected[0]["rejection_reason"] == "zero_kelly_stake"

        # Positive control: a genuinely stakeable admitted bet is still selected with a > 0 stake.
        ok = {
            "game_id": "Y",
            "season": 2023,
            "week": 5,
            "bet_side": "under",
            "calibrated_p_side": 0.60,
            "per_bet_ev": 0.05,
            "slipped_line": 44.5,
            "_actual_total": 40.0,
        }
        sel2: list[dict] = []
        rej2: list[dict] = []
        _make_selector(ev_floor_t=0.0)._admit_and_size_week([ok], sel2, rej2)
        assert len(sel2) == 1
        assert sel2[0]["kelly_stake"] > 0.0


# ---------------------------------------------------------------------------
# Phase 31, plan 31-06 (D31-01): the target-agnostic core + per-target strategies
#
# The refactor's whole claim is that the O/U path MOVED rather than changed. These tests hold that
# claim to a recorded pre-refactor output rather than to "the other tests still pass" -- a rewrite
# that broke a number no existing test happened to read would otherwise sail through.
# ---------------------------------------------------------------------------

# The fixed O/U week the recorded-output comparison runs on. Chosen to exercise every branch of the
# decision path in one frame: an under pick, a high-total OVER trap (eligible, EV below the floor),
# a low-total OVER (outside the union), an under+high-total pick, a PUSH, and a second (season,
# week) group so the per-week sizing loop runs more than once.
_GOLDEN_WEEK = [
    _ou_row("2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0),
    _ou_row("2021_W01_C@D", model_total=52.0, closing_total=50.0, actual_total=49.0),
    _ou_row("2021_W01_E@F", model_total=42.0, closing_total=38.0, actual_total=41.0),
    _ou_row("2021_W01_G@H", model_total=44.0, closing_total=51.0, actual_total=43.0),
    _ou_row("2021_W01_I@J", model_total=40.0, closing_total=45.0, actual_total=44.5),
    _ou_row(
        "2022_W03_K@L",
        model_total=36.0,
        closing_total=47.0,
        actual_total=38.0,
        season=2022,
        week=3,
    ),
    _ou_row(
        "2022_W03_M@N",
        model_total=41.0,
        closing_total=49.0,
        actual_total=52.0,
        season=2022,
        week=3,
    ),
]

# RECORDED from the PRE-REFACTOR selector at commit 6190f92 (the parent of the D31-01 refactor),
# one entry per candidate, keyed by game_id. The values are exact reprs -- no rounding, no
# tolerance -- so a one-ulp drift in the calibrated chain fails here.
_GOLDEN_RECORDS: dict[str, dict] = {
    "2021_W01_A@B": {
        "_actual_total": 40.0,
        "bet_side": "under",
        "calibrated_p_side": 0.7180042896910152,
        "closing_total": 45.0,
        "clv": -7.0,
        "eligible": True,
        "game_id": "2021_W01_A@B",
        "kelly_stake": 288.6751345948129,
        "model_total": 38.0,
        "outcome": True,
        "per_bet_ev": 0.3707354621373925,
        "season": 2021,
        "slipped_line": 44.5,
        "subpop_label": "under",
        "totals_regime": "not_high",
        "week": 1,
    },
    "2021_W01_C@D": {
        "_actual_total": 49.0,
        "bet_side": "over",
        "calibrated_p_side": 0.5153401516797045,
        "closing_total": 50.0,
        "clv": 2.0,
        "eligible": True,
        "game_id": "2021_W01_C@D",
        "kelly_stake": 0.0,
        "model_total": 52.0,
        "outcome": None,
        "per_bet_ev": -0.016168801338745986,
        "season": 2021,
        "slipped_line": 50.5,
        "subpop_label": "high_total",
        "totals_regime": "high",
        "week": 1,
    },
    "2021_W01_E@F": {
        "_actual_total": 41.0,
        "bet_side": "over",
        "calibrated_p_side": None,
        "closing_total": 38.0,
        "clv": 4.0,
        "eligible": False,
        "game_id": "2021_W01_E@F",
        "kelly_stake": 0.0,
        "model_total": 42.0,
        "outcome": None,
        "per_bet_ev": None,
        "season": 2021,
        "slipped_line": None,
        "subpop_label": "none",
        "totals_regime": "not_high",
        "week": 1,
    },
    "2021_W01_G@H": {
        "_actual_total": 43.0,
        "bet_side": "under",
        "calibrated_p_side": 0.7180042896910152,
        "closing_total": 51.0,
        "clv": -7.0,
        "eligible": True,
        "game_id": "2021_W01_G@H",
        "kelly_stake": 288.6751345948129,
        "model_total": 44.0,
        "outcome": True,
        "per_bet_ev": 0.3707354621373925,
        "season": 2021,
        "slipped_line": 50.5,
        "subpop_label": "under+high_total",
        "totals_regime": "high",
        "week": 1,
    },
    "2021_W01_I@J": {
        "_actual_total": 44.5,
        "bet_side": "under",
        "calibrated_p_side": 0.6638804306366265,
        "closing_total": 45.0,
        "clv": -5.0,
        "eligible": True,
        "game_id": "2021_W01_I@J",
        "kelly_stake": 288.6751345948129,
        "model_total": 40.0,
        "outcome": None,
        "per_bet_ev": 0.26740809485174144,
        "season": 2021,
        "slipped_line": 44.5,
        "subpop_label": "under",
        "totals_regime": "not_high",
        "week": 1,
    },
    "2022_W03_K@L": {
        "_actual_total": 38.0,
        "bet_side": "under",
        "calibrated_p_side": 0.8318592517696612,
        "closing_total": 47.0,
        "clv": -11.0,
        "eligible": True,
        "game_id": "2022_W03_K@L",
        "kelly_stake": 353.5533905932737,
        "model_total": 36.0,
        "outcome": True,
        "per_bet_ev": 0.588094935196626,
        "season": 2022,
        "slipped_line": 46.5,
        "subpop_label": "under",
        "totals_regime": "not_high",
        "week": 3,
    },
    "2022_W03_M@N": {
        "_actual_total": 52.0,
        "bet_side": "under",
        "calibrated_p_side": 0.7675399396877096,
        "closing_total": 49.0,
        "clv": -8.0,
        "eligible": True,
        "game_id": "2022_W03_M@N",
        "kelly_stake": 353.5533905932737,
        "model_total": 41.0,
        "outcome": False,
        "per_bet_ev": 0.46530352122199103,
        "season": 2022,
        "slipped_line": 48.5,
        "subpop_label": "under+high_total",
        "totals_regime": "high",
        "week": 3,
    },
}

_GOLDEN_SELECTED_IDS = [
    "2021_W01_A@B",
    "2021_W01_G@H",
    "2021_W01_I@J",
    "2022_W03_K@L",
    "2022_W03_M@N",
]
_GOLDEN_FILTERED_IDS = [
    "2021_W01_A@B",
    "2021_W01_C@D",
    "2021_W01_G@H",
    "2021_W01_I@J",
    "2022_W03_K@L",
    "2022_W03_M@N",
]
_GOLDEN_UNFILTERED_IDS = [
    "2021_W01_A@B",
    "2021_W01_C@D",
    "2021_W01_E@F",
    "2021_W01_G@H",
    "2021_W01_I@J",
    "2022_W03_K@L",
    "2022_W03_M@N",
]
_GOLDEN_REJECTED = [
    ("2021_W01_E@F", "not_subpop"),
    ("2021_W01_C@D", "ev_below_floor"),
]
_GOLDEN_CLV_REPORT = {
    "n": 5,
    "mean": -7.6,
    "t": None,
    "p": None,
    "ci95": None,
    "metric": "model_edge_line_clv (model_total - closing_total); REPORT-ONLY (D27-06)",
}

# The keys the D31-01 refactor ADDS, named exhaustively so a field cannot appear unannounced.
# ``target`` is stamped on every record (a pooled week must be splittable back out); the four
# sizing-provenance fields are added only to records that were actually staked.
_NEW_KEYS_ON_EVERY_RECORD = {"target"}
_NEW_KEYS_ON_STAKED_RECORDS = _NEW_KEYS_ON_EVERY_RECORD | {
    "same_side_group_size",
    "same_game_group_size",
    "binding_group",
    "weekly_scale_factor",
}


class TestD3101RefactorReproducesPreRefactorOutput:
    """T-31-23: the O/U path MOVED behind the seam; it did not change."""

    def test_recorded_output_matches_pre_refactor_exactly(self) -> None:
        """Every pre-refactor key/value is reproduced EXACTLY, and only named keys are added.

        The comparison is two-sided on purpose. Forwards: every recorded key still holds the
        recorded value, compared with ``==`` and no tolerance. Backwards: the set of keys the
        refactor added is exactly the set named above, so the seam cannot smuggle a field onto a
        published bet record without this test being edited.
        """
        result = _make_selector(ev_floor_t=0.0).select(_GOLDEN_WEEK)

        assert [r["game_id"] for r in result.selected] == _GOLDEN_SELECTED_IDS
        assert [r["game_id"] for r in result.filtered] == _GOLDEN_FILTERED_IDS
        assert [r["game_id"] for r in result.unfiltered] == _GOLDEN_UNFILTERED_IDS
        assert [
            (r["game_id"], r["rejection_reason"]) for r in result.rejected
        ] == _GOLDEN_REJECTED
        assert result.clv_report == _GOLDEN_CLV_REPORT

        for list_name in ("selected", "rejected", "filtered", "unfiltered"):
            for record in getattr(result, list_name):
                golden = _GOLDEN_RECORDS[record["game_id"]]
                for key, expected in golden.items():
                    assert record[key] == expected, (
                        f"{list_name} record {record['game_id']} field {key} drifted: "
                        f"{record[key]!r} != {expected!r}"
                    )
                added = set(record) - set(golden) - {"rejection_reason"}
                expected_added = (
                    _NEW_KEYS_ON_STAKED_RECORDS
                    if "weekly_scale_factor" in record
                    else _NEW_KEYS_ON_EVERY_RECORD
                )
                assert added == expected_added, (
                    f"{list_name} record {record['game_id']} added unexpected keys: {added}"
                )

    def test_every_record_is_stamped_with_its_target(self) -> None:
        """A pooled week must be splittable back out, so every record names its target."""
        result = _make_selector(ev_floor_t=0.0).select(_GOLDEN_WEEK)
        for list_name in ("selected", "rejected", "filtered", "unfiltered"):
            for record in getattr(result, list_name):
                assert record["target"] == "ou"


class TestEvFloorAdjacency:
    """T-31-24: the EV-floor comparison rejects STRICTLY BELOW, so EV == t is ADMITTED."""

    def test_ev_exactly_at_the_floor_is_admitted(self) -> None:
        """A candidate whose per-bet EV EQUALS the floor is selected, not rejected (SPEC R1).

        The floor is set to the candidate's own per-bet EV, read off a first pass, so the
        comparison is exercised at exact equality rather than near it. Inverting the comparison
        during the refactor -- ``<`` to ``<=`` -- would flip this candidate out of ``selected``
        and is the single most plausible silent regression in a move of this size.
        """
        candidates = [
            _ou_row(
                "2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0
            ),
        ]
        probe = _make_selector(ev_floor_t=0.0).select(candidates)
        exact_ev = probe.selected[0]["per_bet_ev"]

        at_floor = _make_selector(ev_floor_t=exact_ev).select(candidates)
        assert [r["game_id"] for r in at_floor.selected] == ["2021_W01_A@B"]
        assert at_floor.selected[0]["per_bet_ev"] == exact_ev
        assert all(r["game_id"] != "2021_W01_A@B" for r in at_floor.rejected)

        # Non-vacuity: a floor one ulp ABOVE that EV does reject it, so the assertion above sits
        # exactly on the boundary rather than comfortably inside the admitted region.
        just_above = _make_selector(ev_floor_t=math.nextafter(exact_ev, math.inf))
        result = just_above.select(candidates)
        assert result.selected == []
        assert result.rejected[0]["rejection_reason"] == "ev_below_floor"


class TestOUStrategyMovedVerbatim:
    """D31-01: OUStrategy delegates to the LOCKED convention and re-implements nothing."""

    def test_side_resolution_delegates_to_the_locked_simulator(self) -> None:
        """``OUStrategy`` asks ``BettingSimulator._determine_bet_side_ou`` for the side.

        Proven behaviourally with a stand-in simulator: if the strategy re-implemented the side
        rule, its answer would ignore the stand-in and follow model_total vs closing_total. It
        returns the stand-in's sentinel instead.
        """
        from backtest.selector_strategies import OUStrategy

        class _StandInSimulator:
            """Records the arguments and answers with a sentinel side."""

            def __init__(self) -> None:
                self.calls: list[tuple[float, float]] = []

            def _determine_bet_side_ou(
                self, model_total: float, closing_total: float
            ) -> str:
                self.calls.append((model_total, closing_total))
                return "sentinel_side"

        sim = _StandInSimulator()
        strategy = OUStrategy(
            frozen_sd=_FIXTURE_SD,
            season_bias_by_season=_FIXTURE_BIAS,
            high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
            simulator=sim,
        )
        row = {"model_total": 38.0, "closing_total": 45.0, "season": 2021}
        assert strategy.resolve_bet_side(row) == "sentinel_side"
        assert sim.calls == [(38.0, 45.0)]

    def test_strategy_module_does_not_re_implement_the_side_threshold(self) -> None:
        """The side threshold lives in ``SimulationConfig``, never copied into the strategy."""
        source = Path("backtest/selector_strategies.py").read_text(encoding="utf-8")
        assert "min_edge_threshold" not in source

    def test_missing_prior_season_bias_still_raises_by_name(self) -> None:
        """D27-07: a season with no prior-season bias raises, naming the season and estimator.

        The no-silent-fallback rule is the reason the raise exists: falling back to the raw,
        over-biased total would quietly price every bet in that season off an uncorrected model.
        """
        candidates = [
            _ou_row(
                "2024_W01_A@B",
                model_total=38.0,
                closing_total=45.0,
                actual_total=40.0,
                season=2024,
            ),
        ]
        with pytest.raises(ValueError) as exc:
            _make_selector(ev_floor_t=0.0).select(candidates)
        message = str(exc.value)
        assert "2024" in message
        assert "estimate_prior_season_bias" in message

    def test_no_strategy_method_raises_notimplementederror(self) -> None:
        """An unimplemented-but-registerable strategy is worse than an absent one (D31-01).

        ATS and WP are deliberately ABSENT from this module until Plan 31-10 can add them
        complete. A stub whose methods raise would be registerable, turning a loud
        "unregistered target" error into a failure part-way through a selection run.
        """
        tree = ast.parse(
            Path("backtest/selector_strategies.py").read_text(encoding="utf-8")
        )
        offenders = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Raise)
            and (
                (
                    isinstance(node.exc, ast.Name)
                    and node.exc.id == "NotImplementedError"
                )
                or (
                    isinstance(node.exc, ast.Call)
                    and isinstance(node.exc.func, ast.Name)
                    and node.exc.func.id == "NotImplementedError"
                )
            )
        ]
        assert offenders == []

        class_names = {
            node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        }
        assert "ATSStrategy" not in class_names
        assert "WPStrategy" not in class_names


# ---------------------------------------------------------------------------
# Phase 31, plan 31-06, Task 2 (D31-02): the POOLED weekly exposure cap
#
# D27-10 pre-registered the 10% cap as a per-week TOTAL exposure cap, and the bankroll does not
# grow because targets were added. A per-target 10% cap would be a 30% total weekly ceiling -- a
# post-hoc tripling of a pre-registered ruin guard -- and per-target sub-caps would be a second
# threshold nobody pre-registered, chosen with knowledge of which targets bet. So the cap is
# applied ONCE over the union of a week's bets.
#
# Exercising a MIXED week needs a second registered target, and Plan 31-06 deliberately ships only
# the O/U strategy (ATS and WP arrive in Plan 31-10 complete rather than stubbed). The probe
# strategy below supplies one for tests ONLY -- it is not a production selection path, in the same
# spirit as ``_same_side_only_deweight`` in tests/unit/test_kelly_exposure_caps.py.
# ---------------------------------------------------------------------------

_BANKROLL = 10_000.0
_WEEKLY_CAP = (
    _BANKROLL * 0.10
)  # WEEKLY_CAP_PCT; named locally so the number is visible here.


class _ProbeStrategy:
    """A TEST-ONLY second target, satisfying ``TargetStrategy`` by structural subtyping.

    Deliberately minimal and deliberately NOT an ATS or WP stand-in: it makes no claim about how
    either target will price a bet. Its only job is to put a second target's bets into the same
    week so the POOLED cap, the cross-target same-game de-weighting and the per-target dispatch
    are exercised before Plan 31-10 lands the real strategies.

    It reports no CLV, which is itself part of the contract under test: a target that does not
    report a closing-line value must yield "not reported", never a silent 0.0 that would drag a
    published CLV mean toward zero.
    """

    target = "probe"
    required_market_fields: tuple[str, ...] = ("probe_p_side", "probe_line")

    def resolve_bet_side(self, row: dict) -> str | None:
        return row.get("probe_side")

    def eligibility(self, row: dict, bet_side: str | None) -> str | None:
        return None if bet_side is not None else "not_subpop"

    def eligibility_label(self, row: dict, bet_side: str | None) -> str:
        return "all"

    def side_probability(self, row: dict, bet_side: str) -> tuple[float, float]:
        return float(row["probe_p_side"]), float(row["probe_line"])

    def decision_extras(self, row: dict, bet_side: str | None) -> dict:
        return {"probe_regime": "n/a"}

    def grade(self, record: dict) -> bool | None:
        return None


def _as_target(row: dict, target: str = "ou") -> dict:
    """Tag a candidate row with its target code.

    A row's ``target`` is optional only while ONE strategy is registered -- with more than one
    there is no defensible default, and the selector refuses to guess rather than booking a bet
    under the wrong target's rules. The mixed-week fixtures below therefore tag every row.
    """
    return {**row, "target": target}


def _probe_row(
    game_id: str,
    *,
    side: str,
    p_side: float = 0.72,
    season: int = 2021,
    week: int = 1,
) -> dict:
    """Build one probe-target candidate row for the pooled-week fixtures."""
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "target": "probe",
        "probe_side": side,
        "probe_p_side": p_side,
        "probe_line": 0.0,
        "sportsbook": "consensus",
        "is_live": False,
    }


def _pooled_selector(ev_floor_t: float = 0.0):
    """A BetSelector registering BOTH the real O/U strategy and the test-only probe target."""
    from backtest.bet_selector import BetSelector
    from backtest.selector_strategies import OUStrategy

    return BetSelector(
        frozen_sd=_FIXTURE_SD,
        season_bias_by_season=_FIXTURE_BIAS,
        ev_floor_t=ev_floor_t,
        bankroll=_BANKROLL,
        high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
        strategies=[
            OUStrategy(
                frozen_sd=_FIXTURE_SD,
                season_bias_by_season=_FIXTURE_BIAS,
                high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
            ),
            _ProbeStrategy(),
        ],
    )


# Three O/U unders and three probe bets, ALL in season 2021 week 1, on six distinct games.
# Each side's own group of three de-weights by 1/sqrt(3), so each target ALONE stakes
# 3 * 500 / sqrt(3) = 866.03 -- comfortably under the 1000 cap. Their union is 1732.05, which is
# over it. A per-target cap would therefore not bind at all and the week would stake 1732; the
# pooled cap binds and the week stakes exactly 1000.
_MIXED_WEEK = [
    _as_target(
        _ou_row("2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0)
    ),
    _as_target(
        _ou_row("2021_W01_C@D", model_total=37.0, closing_total=45.0, actual_total=40.0)
    ),
    _as_target(
        _ou_row("2021_W01_E@F", model_total=36.0, closing_total=45.0, actual_total=40.0)
    ),
    _probe_row("2021_W01_G@H", side="home"),
    _probe_row("2021_W01_I@J", side="home"),
    _probe_row("2021_W01_K@L", side="home"),
]

# The same six bets minus the probe target: the O/U-only control that must stake 866.03 and be
# untouched by the weekly cap.
_OU_ONLY_WEEK = _MIXED_WEEK[:3]


class TestPooledWeeklyExposureCap:
    """T-31-25 / D31-02: ONE 10% cap over the union of a week's bets, not one per target."""

    def test_union_exposure_is_capped_once_not_once_per_target(self) -> None:
        """A two-target week stakes the cap in TOTAL, not the cap per target.

        The fixture is built so each target alone sits UNDER the cap and only their union exceeds
        it. That makes the assertion discriminating: a per-target cap would leave the week at
        1732.05 (both targets unscaled), while the pooled cap brings it to exactly 1000.
        """
        result = _pooled_selector().select(_MIXED_WEEK)

        assert len(result.selected) == 6
        total = sum(r["kelly_stake"] for r in result.selected)
        assert total == pytest.approx(_WEEKLY_CAP)

        # The per-target totals each sit under the cap on their own -- which is exactly why an
        # unpooled implementation would not scale anything here.
        per_target: dict[str, float] = {}
        for record in result.selected:
            per_target[record["target"]] = (
                per_target.get(record["target"], 0.0) + record["kelly_stake"]
            )
        assert set(per_target) == {"ou", "probe"}
        for target_total in per_target.values():
            assert target_total < _WEEKLY_CAP

        # Non-vacuity: the same week WITHOUT pooling (each target selected on its own) really
        # would have staked 2x as much, so the cap did real work here.
        ou_alone = sum(
            r["kelly_stake"]
            for r in _make_selector(ev_floor_t=0.0).select(_OU_ONLY_WEEK).selected
        )
        assert ou_alone == pytest.approx(3 * 500.0 / math.sqrt(3))
        assert 2 * ou_alone > _WEEKLY_CAP

    def test_every_bet_carries_the_same_weekly_pro_rata_factor(self) -> None:
        """Pro-rata scaling means ONE factor for the whole week, asserted with exact equality.

        Compared with ``==`` and not ``approx``: a per-bet factor that merely rounds to the same
        value would mean the week was scaled bet-by-bet, which is a different rule (it would not
        preserve the relative ordering of stakes) wearing the same name.
        """
        result = _pooled_selector().select(_MIXED_WEEK)

        factors = {r["weekly_scale_factor"] for r in result.selected}
        assert len(factors) == 1
        factor = factors.pop()
        assert factor < 1.0  # the cap really bound on this week

        # Relative ordering by stake is preserved: scaling every bet by one factor cannot reorder
        # them, so the ranking by de-weighted stake and by final stake agree.
        by_final = [
            r["game_id"]
            for r in sorted(result.selected, key=lambda r: -r["kelly_stake"])
        ]
        by_pre_cap = [
            r["game_id"]
            for r in sorted(
                result.selected,
                key=lambda r: -(r["kelly_stake"] / r["weekly_scale_factor"]),
            )
        ]
        assert by_final == by_pre_cap

    def test_sizing_pipeline_is_called_exactly_once_per_week(self) -> None:
        """T-31-25: one ``apply_sizing_pipeline`` call per week, over the pooled union.

        Counted with a spy rather than inferred from the numbers: two calls that each happened to
        stay under the cap would produce a correct-looking total on some weeks while being the
        per-target rule this decision rejected.
        """
        from backtest import bet_selector as selector_module

        calls: list[int] = []
        real = selector_module.apply_sizing_pipeline

        def _spy(bets, bankroll):
            calls.append(len(bets))
            return real(bets, bankroll)

        selector = _pooled_selector()
        # Two distinct (season, week) groups -> exactly two calls, one per week, each over that
        # week's union across BOTH targets.
        second_week = [
            _as_target(
                _ou_row(
                    "2021_W02_M@N",
                    model_total=38.0,
                    closing_total=45.0,
                    actual_total=40.0,
                    week=2,
                )
            ),
            _probe_row("2021_W02_O@P", side="away", week=2),
        ]

        original = selector_module.apply_sizing_pipeline
        selector_module.apply_sizing_pipeline = _spy
        try:
            selector.select([*_MIXED_WEEK, *second_week])
        finally:
            selector_module.apply_sizing_pipeline = original

        assert calls == [6, 2]

    def test_single_target_week_stakes_are_bit_identical_to_the_pre_refactor_path(
        self,
    ) -> None:
        """A week containing only O/U bets is untouched by pooling (the Phase-27 reproduction).

        Asserted with ``==`` against the RECORDED pre-refactor stakes -- no tolerance. Pooling a
        union of one is the identity, and if it is not, every Phase-27 O/U publication moved.
        """
        result = _make_selector(ev_floor_t=0.0).select(_GOLDEN_WEEK)
        stakes = {r["game_id"]: r["kelly_stake"] for r in result.selected}
        for game_id, stake in stakes.items():
            assert stake == _GOLDEN_RECORDS[game_id]["kelly_stake"]

        # And the same week run through a selector that ALSO has the probe target registered
        # stakes identically, because no probe bet is present to pool with.
        pooled = _pooled_selector().select([_as_target(row) for row in _GOLDEN_WEEK])
        assert {r["game_id"]: r["kelly_stake"] for r in pooled.selected} == stakes

    def test_pooling_cannot_change_which_bets_are_selected(self) -> None:
        """Sizing runs AFTER admission, so adding a second target cannot unselect an O/U bet.

        The EV-floor loop reads ``per_bet_ev`` and no stake at all, so the admitted set is a
        function of the EV chain alone. Compared as a set of (game_id, target) PAIRS: two runs can
        agree on how many bets they took while disagreeing about which.
        """
        alone = {
            (r["game_id"], r["target"])
            for r in _make_selector(ev_floor_t=0.0).select(_OU_ONLY_WEEK).selected
        }
        pooled = {
            (r["game_id"], r["target"])
            for r in _pooled_selector().select(_MIXED_WEEK).selected
            if r["target"] == "ou"
        }
        assert alone == pooled
        assert len(alone) == 3

    def test_cap_order_is_unchanged_and_consumed_not_reimplemented(self) -> None:
        """The LOCKED CAP_ORDER still reads kelly -> per-bet -> de-weight -> weekly.

        The selector consumes ``apply_sizing_pipeline`` rather than re-ordering the steps itself,
        so the order is enforced in one place. Pinned here as well because the pooled path is the
        one that now feeds it.
        """
        from utils.kelly_criterion import CAP_ORDER

        assert CAP_ORDER == (
            "kelly_stake",
            "per_bet_5pct_cap",
            "same_game_and_side_deweight",
            "weekly_10pct_cap",
        )

        source = Path("backtest/bet_selector.py").read_text(encoding="utf-8")
        assert "apply_sizing_pipeline(" in source
        # The de-weight helper is reached only THROUGH the pipeline; the selector never calls it
        # directly, which is what keeps the order un-reorderable from here.
        assert "apply_same_game_and_side_deweight" not in source


class TestPooledSizingProvenanceOnTheRecord:
    """D31-02/03: the caps are VISIBLE on the record, not inferable from the ordering."""

    def test_each_record_carries_the_grouping_and_weekly_factor(self) -> None:
        """Every selected record names both group sizes, which bound, and the weekly factor.

        /bets shows a stake beside its EV; without these four fields a reader can see that a stake
        is smaller than Kelly asked for but not why, and "why" is the difference between a cap
        working and a bug.
        """
        result = _pooled_selector().select(_MIXED_WEEK)
        for record in result.selected:
            assert record["same_side_group_size"] == 3
            assert record["same_game_group_size"] == 1
            assert record["binding_group"] == "same_side"
            assert 0.0 < record["weekly_scale_factor"] < 1.0

    def test_a_cross_target_same_game_pair_is_grouped_by_game(self) -> None:
        """D31-03 becomes load-bearing under pooling: two targets on ONE game group by game.

        Before pooling, a week carried at most one O/U bet per game and the same-game group size
        was always 1. With a second target betting the same game, the game grouping is what
        catches a pair that is close to one leveraged wager -- the side strings never mix, so
        same-side grouping alone would see two groups of one and de-weight nothing.
        """
        shared_game_week = [
            _as_target(
                _ou_row(
                    "2021_W05_A@B",
                    model_total=38.0,
                    closing_total=45.0,
                    actual_total=40.0,
                    week=5,
                )
            ),
            _probe_row("2021_W05_A@B", side="home", week=5),
        ]
        result = _pooled_selector().select(shared_game_week)

        assert len(result.selected) == 2
        for record in result.selected:
            assert record["same_side_group_size"] == 1
            assert record["same_game_group_size"] == 2
            assert record["binding_group"] == "same_game"
            # 1/sqrt(2) of the 5% per-bet cap; the weekly cap does not bind on two bets.
            assert record["kelly_stake"] == pytest.approx(500.0 / math.sqrt(2))
            assert record["weekly_scale_factor"] == 1.0

    def test_a_target_reporting_no_clv_yields_not_reported_never_zero(self) -> None:
        """A target with no CLV definition reports None, and the CLV summary excludes it.

        Defaulting an unreported CLV to 0.0 would drag a published mean toward zero and read as
        "no edge measured" rather than "not measured", which is the honesty-of-record failure this
        project's threat model is actually about.
        """
        result = _pooled_selector().select(_MIXED_WEEK)

        probe_records = [r for r in result.selected if r["target"] == "probe"]
        assert probe_records
        for record in probe_records:
            assert record["clv"] is None

        # The report is computed over the three O/U bets only; their model-edge CLVs are -7, -8
        # and -9, so a probe CLV silently entering as 0.0 would move both n and the mean.
        assert result.clv_report is not None
        assert result.clv_report["n"] == 3
        assert result.clv_report["mean"] == pytest.approx(-8.0)


# ---------------------------------------------------------------------------
# Phase 31, plan 31-06, Task 3: the facade contract, pinned so the refactor cannot drift
#
# T-31-26 / T-31-27. These are STRUCTURAL assertions, not behavioural ones: they hold the shape
# the D31-01 split promised, so a later edit that quietly adds a public name, registers a
# half-built strategy, or lets a missing target produce an empty bet list fails here rather than
# in a published readout.
# ---------------------------------------------------------------------------

# The entire public surface of ``backtest.bet_selector``. R4's source scan pins this module as the
# ONE import target for a bet decision, so the surface is enumerated rather than sampled.
_FACADE_PUBLIC_NAMES = {
    "REJECTION_REASONS",
    "BetSelector",
    "SelectionResult",
    "assert_real_odds",
}


class _IncompleteStrategy:
    """A NEGATIVE CONTROL: a strategy missing ``side_probability`` and its data members.

    Deliberately never registered. Its only job is to prove the conformance check below can
    actually fail -- a runtime Protocol assertion that nothing has ever failed is indistinguishable
    from one that always passes.
    """

    def resolve_bet_side(self, row: dict) -> str | None:
        return None

    def eligibility(self, row: dict, bet_side: str | None) -> str | None:
        return None

    def eligibility_label(self, row: dict, bet_side: str | None) -> str:
        return "all"

    def decision_extras(self, row: dict, bet_side: str | None) -> dict:
        return {}

    def grade(self, record: dict) -> bool | None:
        return None


class TestFacadePublicSurface:
    """The facade's exported surface is exactly four names and does not grow by accident."""

    def test_all_names_exactly_the_four_public_symbols(self) -> None:
        """``__all__`` is the four names, no more and no fewer.

        Enumerated rather than subset-checked in both directions: a subset check would let a fifth
        export appear silently, and R4's guard rests on this module having ONE decision surface.
        """
        from backtest import bet_selector as module

        assert set(module.__all__) == _FACADE_PUBLIC_NAMES
        assert len(module.__all__) == len(_FACADE_PUBLIC_NAMES)

    def test_every_exported_name_resolves_in_this_module(self) -> None:
        """Each exported name resolves, and the three objects are DEFINED here, not re-exported.

        Checked by ``__module__`` and not by identity against an import: an alias re-pointed at
        another module would still satisfy an identity check made through the same alias.
        """
        from backtest import bet_selector as module

        for name in module.__all__:
            assert hasattr(module, name), (
                f"__all__ names {name} but the module does not define it"
            )

        assert module.BetSelector.__module__ == "backtest.bet_selector"
        assert module.SelectionResult.__module__ == "backtest.bet_selector"
        assert module.assert_real_odds.__module__ == "backtest.bet_selector"
        # REJECTION_REASONS is a tuple of strings and carries no ``__module__``; pin its contents.
        assert module.REJECTION_REASONS == (
            "not_subpop",
            "ev_below_floor",
            "real_odds_failed",
            "zero_kelly_stake",
        )


class TestStrategyRegistryContract:
    """T-31-27: the registry is keyed by target, complete, and fails loudly when it is not."""

    def test_every_registered_strategy_satisfies_the_protocol(self) -> None:
        """A registered strategy is checked structurally at TEST time, not at selection time.

        ``TargetStrategy`` is ``runtime_checkable``, so ``isinstance`` verifies that every member
        the core dispatches through is actually present. Without this, a strategy missing a member
        would raise part-way through a selection run, after some bets had already been priced.
        """
        from backtest.selector_strategies import TargetStrategy

        for selector in (_make_selector(), _pooled_selector()):
            assert selector.strategies
            for code, strategy in selector.strategies.items():
                assert isinstance(strategy, TargetStrategy), (
                    f"registered strategy for target {code!r} does not satisfy TargetStrategy"
                )
                assert strategy.target == code

    def test_an_incomplete_strategy_fails_the_conformance_check(self) -> None:
        """The negative control: a class missing members is NOT a ``TargetStrategy``.

        Both failure modes are covered -- a missing method (``side_probability``) and missing data
        members (``target`` / ``required_market_fields``) -- because a data protocol that only
        checked methods would wave through a strategy with no registry key.
        """
        from backtest.selector_strategies import OUStrategy, TargetStrategy

        assert not isinstance(_IncompleteStrategy(), TargetStrategy)
        # Positive control on the same assertion, so a broken check cannot pass by always
        # returning False.
        assert isinstance(
            OUStrategy(frozen_sd=_FIXTURE_SD, season_bias_by_season=_FIXTURE_BIAS),
            TargetStrategy,
        )

    def test_registry_is_keyed_by_target_and_rejects_a_duplicate_key(self) -> None:
        """Two strategies claiming one target is a construction error, not a last-one-wins.

        Silently keeping the last registration would mean the target's rules depend on argument
        order, which is exactly the "one source, or one convention" distinction D31-01 exists to
        keep on the right side of.
        """
        from backtest.bet_selector import BetSelector
        from backtest.selector_strategies import OUStrategy

        assert set(_pooled_selector().strategies) == {"ou", "probe"}

        def _ou() -> OUStrategy:
            return OUStrategy(
                frozen_sd=_FIXTURE_SD,
                season_bias_by_season=_FIXTURE_BIAS,
                high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
            )

        with pytest.raises(ValueError, match="duplicate strategy"):
            BetSelector(
                frozen_sd=_FIXTURE_SD,
                season_bias_by_season=_FIXTURE_BIAS,
                strategies=[_ou(), _ou()],
            )

    def test_an_empty_registry_is_refused(self) -> None:
        """A selector with no strategies could only ever return an empty bet list."""
        from backtest.bet_selector import BetSelector

        with pytest.raises(ValueError, match="at least one target strategy"):
            BetSelector(
                frozen_sd=_FIXTURE_SD,
                season_bias_by_season=_FIXTURE_BIAS,
                strategies=[],
            )

    def test_unregistered_target_raises_a_named_error_not_a_keyerror(self) -> None:
        """T-31-27: a candidate for an unregistered target fails LOUDLY, naming both sides.

        An empty result for a target nobody registered is indistinguishable, after the fact, from
        a target that genuinely had no +EV bets that week -- which is the one thing a
        profitability readout must never be ambiguous about.
        """
        from backtest.selector_strategies import UnregisteredTargetError

        row = _as_target(
            _ou_row(
                "2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0
            ),
            target="ats",
        )
        with pytest.raises(UnregisteredTargetError) as exc:
            _make_selector().select([row])
        message = str(exc.value)
        assert "ats" in message  # the target that was asked for
        assert "ou" in message  # the registered set

        # It is a LookupError subclass, so an existing ``except LookupError`` still catches it,
        # but it is NOT a bare KeyError from a dict lookup.
        assert isinstance(exc.value, LookupError)
        assert not isinstance(exc.value, KeyError)

    def test_untagged_candidate_is_ambiguous_once_two_targets_are_registered(
        self,
    ) -> None:
        """With more than one strategy there is no defensible default target, so it refuses.

        Guessing would book a bet under another target's eligibility rule and price it with
        another target's chain -- a wrong number that looks entirely ordinary on the page.
        """
        from backtest.selector_strategies import UnregisteredTargetError

        untagged = _ou_row(
            "2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0
        )
        with pytest.raises(UnregisteredTargetError, match="ambiguous"):
            _pooled_selector().select([untagged])

        # The same untagged row is UNAMBIGUOUS with a single strategy registered, which is why
        # every pre-D31-01 caller keeps working untouched.
        assert _make_selector().select([untagged]).selected


class TestNameCollisionTrapIsDocumented:
    """T-31-26: the dead v1.0 ``BetSelector`` must never satisfy a name-keyed guard."""

    def test_both_bet_selector_classes_are_live_in_one_process(self) -> None:
        """The collision is a FACT of every run, not a latent risk on disk.

        ``backtest/bet_selector.py`` does ``from utils import get_logger``, which executes
        ``utils/__init__`` and imports the DEAD v1.0 class as a side effect of importing the live
        one. Any structural guard over this area must key on the MODULE PATH, never the class
        name; Plan 31-17 owns the one-path guard that depends on this.
        """
        import utils
        from backtest.bet_selector import BetSelector as LiveSelector

        dead_selector = utils.BetSelector
        assert dead_selector is not LiveSelector
        assert dead_selector.__name__ == LiveSelector.__name__ == "BetSelector"
        assert LiveSelector.__module__ == "backtest.bet_selector"
        assert dead_selector.__module__ == "utils.bet_selector"

    def test_this_module_records_the_module_path_rule(self) -> None:
        """The rule is written where the next guard author will read it -- this file's docstring."""
        docstring = Path(__file__).read_text(encoding="utf-8").split('"""')[1]
        assert "module path" in docstring.lower()
        assert "utils.bet_selector" in docstring
        assert "31-17" in docstring
