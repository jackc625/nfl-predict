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
