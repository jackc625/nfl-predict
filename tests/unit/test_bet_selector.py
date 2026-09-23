"""Unit tests for the Phase-27 O/U BetSelector (BET-01/02, OUM-04/06).

Covers:
- (DELETED BY RULING, D33.2-24) Task 1 (LOCKED-1) used to pin the pre-hold high-total boundary
  derivation, its hold-season leakage assertion and the legacy-46.5 comparison. The boundary, its
  derivation and the O/U eligibility UNION it served were deleted together, so those four tests
  went with their subject; ``tests/unit/test_ou_eligibility_gate_removed.py`` now guards that no
  gate returns.
- Task 2/3 (BET-01/02, OUM-04/06, D27-06/14): the single-source ``BetSelector.select()``
  decision engine -- universal O/U candidacy (every sided candidate reaches the EV floor, which
  replaced the sub-pop UNION filter under D33.2-24), EV-floor admission, high-total-OVER pocket drop, the
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

Run the full module:          pytest tests/unit/test_bet_selector.py -x -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Number anchors (kept explicit so a silent drift is caught).
# ---------------------------------------------------------------------------

_OU_BREAKEVEN = 110.0 / 210.0  # 0.52380952... (flat -110 cover breakeven)

# The closing totals the retired high-total boundary (~48.0) used to split. The universal-candidacy
# test below walks candidates across it on BOTH sides, so a boundary reappearing anywhere in the
# O/U decision path would change the answer for some of them.
_TOTALS_STRADDLING_THE_RETIRED_BOUNDARY = (38.0, 44.5, 48.0, 51.5)


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
# Task 1 (LOCKED-1) -- DELETED BY RULING (D33.2-24).
#
# Four tests used to live here: the pre-hold high-total boundary excluded hold-season rows, a
# derivation window naming a hold season raised, the boundary sat in a sane band beside the legacy
# 46.5, and the pre-hold window was disjoint from the hold window. All four pinned the derivation
# of a constant that D33.2-24 deleted together with the O/U eligibility UNION it served; the
# derivation window, the leakage error and the constant have no reader left. They are not
# re-expressed, because their subject no longer exists -- re-deriving the boundary on re-measured
# past seasons is the exact step D33.2-24 rejects. What replaces their protective intent is the
# standing guard ``tests/unit/test_ou_eligibility_gate_removed.py``, which fails if any gate
# returns to the O/U path.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Task 2/3: BetSelector single-source decision engine (BET-01/02, OUM-04/06)
#
# These tests target the not-yet-built ``backtest.bet_selector`` module (RED until Task 3).
# Synthetic, unit-fast frames only (never the real artifact). The selector API under test is a
# BetSelector constructed with frozen_sd, season_bias_by_season (prior-season walk-forward bias,
# negative for an over-biased model), ev_floor_t (the EV-floor scalar t Plan 04 tunes) and
# bankroll. Its select(candidates, raw_odds_df=None) returns a
# result that exposes: selected, rejected (each with a rejection_reason), filtered (the eligible
# acceptance basis), unfiltered (the whole-population cross-check), and clv_report. Each per-bet
# record carries at least the game id, season, week, bet side, model and closing
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
        **kwargs,
    )


class TestBetSelectorUniversalCandidacy:
    """Every sided O/U candidate reaches the EV floor; no sub-population gate (D33.2-24).

    Re-expresses the Phase-27 sub-pop UNION test (D27-04/05). Its "an under pick is eligible" and
    "a high-total over is eligible" intents fold into ONE universal-candidacy assertion, asserted
    for both sides across totals that straddle the retired ~48.0 boundary. Its third intent -- "a
    low-total over is INELIGIBLE" -- is DELETED BY RULING: D33.2-24 removed the rule that made it
    so, and the low-total over below is now asserted to be a candidate like the rest.
    """

    def test_every_sided_candidate_reaches_the_ev_floor(self) -> None:
        """Both sides, at totals below, at and above the retired boundary, are all eligible.

        Eligible means "in ``filtered``" -- the set the EV floor then judges. No candidate may be
        rejected as ``not_subpop``, and the only reason any of them can fail is the floor.
        """
        candidates = []
        for index, closing_total in enumerate(_TOTALS_STRADDLING_THE_RETIRED_BOUNDARY):
            candidates.append(
                _ou_row(
                    f"2021_W{index + 1:02d}_UND@ER",
                    model_total=closing_total - 6.0,
                    closing_total=closing_total,
                    actual_total=closing_total - 3.0,
                    week=index + 1,
                )
            )
            candidates.append(
                _ou_row(
                    f"2021_W{index + 1:02d}_OVE@R",
                    model_total=closing_total + 6.0,
                    closing_total=closing_total,
                    actual_total=closing_total + 3.0,
                    week=index + 1,
                )
            )
        result = _make_selector().select(candidates)

        eligible_ids = {r["game_id"] for r in result.filtered}
        assert eligible_ids == {row["game_id"] for row in candidates}
        assert {r["bet_side"] for r in result.filtered} == {"over", "under"}
        assert all(r["subpop_label"] == "no_subpopulation" for r in result.unfiltered)
        assert {r["rejection_reason"] for r in result.rejected} <= {"ev_below_floor"}

    def test_the_low_total_over_is_a_candidate_decided_by_the_ev_floor(self) -> None:
        """The case the deleted UNION refused is now judged by the EV floor alone.

        ``2021_W03_E@F`` (model 42 over a 38 line) was the canonical ``not_subpop`` row. It is now
        eligible, and whether it is BET depends only on the floor: selected at t=0, rejected as
        ``ev_below_floor`` under a punishing floor -- never as outside a sub-population.
        """
        row = _ou_row(
            "2021_W03_E@F", model_total=42.0, closing_total=38.0, actual_total=41.0
        )
        permissive = _make_selector(ev_floor_t=0.0).select([row])
        assert [r["game_id"] for r in permissive.filtered] == ["2021_W03_E@F"]
        assert [r["game_id"] for r in permissive.selected] == ["2021_W03_E@F"]

        punishing = _make_selector(ev_floor_t=0.9).select([row])
        assert punishing.selected == []
        assert [(r["game_id"], r["rejection_reason"]) for r in punishing.rejected] == [
            ("2021_W03_E@F", "ev_below_floor")
        ]


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
    """The high-total OVER over-bias pocket is dropped by the calibrated EV chain (D27-05).

    KEPT AND RE-EXPRESSED (D33.2-24). The intent never depended on the eligibility UNION: it is the
    EV chain -- bias correction plus the half-point slippage -- that prices these overs below
    breakeven. The assertion used to select them by the record's ``totals_regime`` field, which
    died with the union; it now names the rows by game id, so the claim is unchanged.
    """

    def test_high_total_over_pocket_dropped(self) -> None:
        """High-total OVER picks with a low bias-corrected P(over) are REJECTED; UNDERs survive.

        RESEARCH Finding 4 / D27-05: the high-total OVER pocket grades below breakeven. After
        bias-correction and the half-point slippage against the OVER side, those picks have
        per-bet EV < 0 and are rejected by the EV floor (t=0). A high-total UNDER pick survives.
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

        selected_ids = {r["game_id"] for r in result.selected}
        rejected = {r["game_id"]: r["rejection_reason"] for r in result.rejected}
        # The over-bias pocket is dropped -- by the EV floor, not by any eligibility rule.
        for trap in ("2022_W01_A@B", "2022_W02_C@D"):
            assert trap not in selected_ids
            assert rejected[trap] == "ev_below_floor"
        assert "2022_W03_E@F" in selected_ids  # high-total UNDER survives


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

        # The candidate is eligible (every sided candidate is, D33.2-24) and present in the
        # filtered set with a calibrated probability handed to Kelly.
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
        rows that the unfiltered set includes. Since D33.2-24 the only ineligible O/U shape is a
        SIDELESS candidate (the model agrees with the market inside the LOCKED no-bet band); the
        low-total over this test used to exclude is now in both sets.
        """
        candidates = [
            _ou_row(
                "2021_W01_A@B", model_total=38.0, closing_total=45.0, actual_total=40.0
            ),
            _ou_row(
                "2021_W03_E@F", model_total=42.0, closing_total=38.0, actual_total=41.0
            ),
            # model == line: no side, so nothing to price.
            _ou_row(
                "2021_W04_G@H", model_total=45.0, closing_total=45.0, actual_total=41.0
            ),
        ]
        result = _make_selector(ev_floor_t=0.0).select(candidates)

        filtered_ids = {r["game_id"] for r in result.filtered}
        unfiltered_ids = {r["game_id"] for r in result.unfiltered}
        # The sideless row is in the unfiltered cross-check but NOT in the filtered basis.
        assert "2021_W04_G@H" in unfiltered_ids
        assert "2021_W04_G@H" not in filtered_ids
        assert {"2021_W01_A@B", "2021_W03_E@F"} <= filtered_ids


class TestBetSelectorSelectedAndRejected:
    """select() returns selected AND rejected eligible candidates with rejection reasons."""

    def test_select_returns_selected_and_rejected_with_reasons(self) -> None:
        """Rejection reasons cover {no_bet_side, ev_below_floor, real_odds_failed} as applicable.

        Construct one offender per reason and assert each reason appears:
        - no_bet_side: a sideless candidate (model == line). This replaces the ``not_subpop``
          offender (a low-total over), which D33.2-24 made a candidate -- the O/U selector can no
          longer produce ``not_subpop`` at all, and that is asserted too.
        - ev_below_floor: an eligible pick with EV under a high floor.
        - real_odds_failed: an offender row in the raw odds frame (named per-game).
        """
        # A sideless offender + two eligible-but-floored offenders (the second is the low-total
        # over the deleted UNION used to reject). A high floor rejects every eligible pick.
        candidates = [
            _ou_row(
                "2021_W01_A@B", model_total=44.0, closing_total=45.0, actual_total=43.0
            ),
            _ou_row(
                "2021_W03_E@F", model_total=42.0, closing_total=38.0, actual_total=41.0
            ),
            _ou_row(
                "2021_W04_G@H", model_total=45.0, closing_total=45.0, actual_total=41.0
            ),
        ]
        result = _make_selector(ev_floor_t=0.9).select(candidates)
        reasons = {r["game_id"]: r["rejection_reason"] for r in result.rejected}
        assert reasons == {
            "2021_W01_A@B": "ev_below_floor",
            "2021_W03_E@F": "ev_below_floor",
            "2021_W04_G@H": "no_bet_side",
        }
        assert "not_subpop" not in reasons.values()

        # real_odds_failed: a per-game offender surfaced as a named rejection (or a raised
        # ValueError naming the offender). The selector exposes the reason taxonomy explicitly.
        # ``not_subpop`` stays IN the taxonomy: published pre-33.2 records carry it and the page
        # must still label them, even though no production strategy emits it any more.
        from backtest.bet_selector import REJECTION_REASONS

        assert {
            "not_subpop",
            "no_bet_side",
            "ev_below_floor",
            "real_odds_failed",
        } <= set(REJECTION_REASONS)


# ---------------------------------------------------------------------------
# Phase 27 code-review fixes (WR-03 / WR-05 / WR-07)
# ---------------------------------------------------------------------------


class TestPhase27ReviewFixes:
    """Regression coverage for the WR-05/07 review fixes.

    The WR-03 test (a non-finite high-total boundary hard-fails construction) is DELETED BY RULING
    (D33.2-24): the selector no longer takes a boundary, so there is nothing to be non-finite.
    What WR-03 protected -- the UNION silently collapsing to under-only -- cannot happen to a rule
    that does not exist. Passing the retired keyword is now a ``TypeError``, asserted below.
    """

    def test_the_retired_boundary_keyword_is_refused(self) -> None:
        """``BetSelector`` and ``OUStrategy`` refuse ``high_total_boundary`` by name (D33.2-24).

        A parameter with no reader is a second answer waiting to be revived, so it is gone from
        both signatures rather than accepted and ignored.
        """
        from backtest.bet_selector import BetSelector
        from backtest.selector_strategies import OUStrategy

        with pytest.raises(TypeError, match="high_total_boundary"):
            BetSelector(
                frozen_sd=_FIXTURE_SD,
                season_bias_by_season=_FIXTURE_BIAS,
                high_total_boundary=48.0,  # type: ignore[call-arg]
            )
        with pytest.raises(TypeError, match="high_total_boundary"):
            OUStrategy(
                frozen_sd=_FIXTURE_SD,
                season_bias_by_season=_FIXTURE_BIAS,
                high_total_boundary=48.0,  # type: ignore[call-arg]
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
            # An eligible record always carries the price it was priced at (D31-04); these two
            # bypass ``_build_decision_record`` and so state it explicitly.
            "selected_odds": -110,
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
            "selected_odds": -110,
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
# a low-total OVER (outside the retired union -- a candidate since D33.2-24), an under pick on a
# high total, a PUSH, and a second (season, week) group so the per-week sizing loop runs more than
# once.
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
#
# DELIBERATELY RE-PINNED under D33.2-24 (Plan 33.2-06), in the same commit as the rule change, in
# the shape ``tests/unit/test_scheduler_xml_unchanged.py`` declares a re-pin. What moved, and only
# this:
#   * ``2021_W01_E@F`` (the low-total over) is no longer refused as ``not_subpop``. It is priced
#     and SELECTED (calibrated P(over) 0.5762..., EV +0.1001...), so its record is re-recorded.
#   * Week 1 now carries FOUR staked bets instead of three, so the pooled 10% weekly cap scales
#     each of ``A@B`` / ``G@H`` / ``I@J`` to 265.748... (was 288.675...). The week-3 stakes are
#     untouched.
#   * ``subpop_label`` is ``no_subpopulation`` on every record, and ``totals_regime`` is gone from
#     every record (the field died with the union).
# EVERY OTHER recorded value -- each calibrated P(side), per-bet EV, slipped line, CLV and outcome
# of the six rows the gate never touched -- is still exactly the 6190f92 recording, which is the
# evidence that the calibrated chain itself did not move.
_GOLDEN_RECORDS: dict[str, dict] = {
    "2021_W01_A@B": {
        "_actual_total": 40.0,
        "bet_side": "under",
        "calibrated_p_side": 0.7180042896910152,
        "closing_total": 45.0,
        "clv": -7.0,
        "eligible": True,
        "game_id": "2021_W01_A@B",
        "kelly_stake": 265.74826192127836,
        "model_total": 38.0,
        "outcome": True,
        "per_bet_ev": 0.3707354621373925,
        "season": 2021,
        "slipped_line": 44.5,
        "subpop_label": "no_subpopulation",
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
        "subpop_label": "no_subpopulation",
        "week": 1,
    },
    "2021_W01_E@F": {
        "_actual_total": 41.0,
        "bet_side": "over",
        "calibrated_p_side": 0.5762494033659527,
        "closing_total": 38.0,
        "clv": 4.0,
        "eligible": True,
        "game_id": "2021_W01_E@F",
        "kelly_stake": 202.75521423616496,
        "model_total": 42.0,
        "outcome": True,
        "per_bet_ev": 0.1001124973350005,
        "season": 2021,
        "slipped_line": 38.5,
        "subpop_label": "no_subpopulation",
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
        "kelly_stake": 265.74826192127836,
        "model_total": 44.0,
        "outcome": True,
        "per_bet_ev": 0.3707354621373925,
        "season": 2021,
        "slipped_line": 50.5,
        "subpop_label": "no_subpopulation",
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
        "kelly_stake": 265.74826192127836,
        "model_total": 40.0,
        "outcome": None,
        "per_bet_ev": 0.26740809485174144,
        "season": 2021,
        "slipped_line": 44.5,
        "subpop_label": "no_subpopulation",
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
        "subpop_label": "no_subpopulation",
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
        "subpop_label": "no_subpopulation",
        "week": 3,
    },
}

_GOLDEN_SELECTED_IDS = [
    "2021_W01_A@B",
    "2021_W01_E@F",
    "2021_W01_G@H",
    "2021_W01_I@J",
    "2022_W03_K@L",
    "2022_W03_M@N",
]
_GOLDEN_FILTERED_IDS = [
    "2021_W01_A@B",
    "2021_W01_C@D",
    "2021_W01_E@F",
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
    ("2021_W01_C@D", "ev_below_floor"),
]
_GOLDEN_CLV_REPORT = {
    "n": 6,
    "mean": -5.666666666666667,
    "t": None,
    "p": None,
    "ci95": None,
    "metric": "model_edge_line_clv (model_total - closing_total); REPORT-ONLY (D27-06)",
}

# The keys the D31-01 refactor ADDS, named exhaustively so a field cannot appear unannounced.
# ``target`` is stamped on every record (a pooled week must be splittable back out); the four
# sizing-provenance fields are added only to records that were actually staked.
#
# Plan 31-09 (D31-17) adds two more to EVERY record: the candidate's own ``snapshot_ts`` and the
# ``freeze_ts`` it is judged against, both tz-aware and Eastern-expressed. They travel on the
# record so the page renders them without recomputing either -- and a recomputed freeze on the
# display side is exactly the second derivation D31-18 exists to prevent. On this fixture week,
# whose rows carry no kickoff date, both are None: a historical backtest frame makes no forward
# freshness claim.
#
# Plan 31-10 (D31-04) adds one more to EVERY record: ``selected_odds``, the American price the bet
# was judged at. It travels on the record so the per-bet EV, the Kelly stake and the published
# price are the SAME number.
#
# DEF-31-13 (ruled 2026-09-05) adds ``devig_method`` to the two LINE targets' records -- the label
# distinguishing a price the market really quoted from the flat -110 fallback an ABSENT price gets.
# The values on THIS fixture week are unchanged by that ruling and are asserted unchanged above:
# these rows carry no juice columns, so every one of them still prices at -110, and the new key
# says ``flat_-110`` rather than leaving that fact to be inferred from the number.
_NEW_KEYS_ON_EVERY_RECORD = {
    "target",
    "snapshot_ts",
    "freeze_ts",
    "selected_odds",
    "devig_method",
}
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

        Until Plan 31-10 the ATS and WP classes were deliberately ABSENT from this module rather
        than present-and-stubbed, and this test asserted their absence. 31-10 added them COMPLETE,
        so the assertion inverts: they must now be present, and the "no method raises" half is what
        keeps the original guarantee -- a stub whose methods raise would be registerable, turning a
        loud "unregistered target" error into a failure part-way through a selection run.
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
        assert "ATSStrategy" in class_names
        assert "WPStrategy" in class_names


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
        strategies=[
            OUStrategy(
                frozen_sd=_FIXTURE_SD,
                season_bias_by_season=_FIXTURE_BIAS,
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
        # Plan 31-09 GREW it from four members to eight (D31-17/19) and plan 31-10 to nine
        # (D31-05): earlier members keep their positions and a new reason is appended. The taxonomy
        # is still ONE exported list -- growing it here is the designed way to add a reason, and
        # editing this tuple is what makes an undeclared TENTH reason fail.
        assert module.REJECTION_REASONS == (
            "not_subpop",
            "ev_below_floor",
            "real_odds_failed",
            "zero_kelly_stake",
            "stale_line",
            "missing_snapshot",
            "missing_prediction",
            "ev_not_finite",
            "no_bet_side",
            "no_honest_ev_floor",
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


# ---------------------------------------------------------------------------
# Phase 31, plan 31-10, Task 1 (D31-05): the spread and winner strategies
#
# Three claims, and each is pinned by a test that can only pass under ONE sign convention or ONE
# price source -- a test that passes under both conventions would pin nothing:
#
#   1. The spread strategy converts at the TWO LINE-CONVENTION HELPERS and nowhere else. DEF-31-01
#      was RULED on 2026-09-04: the stored ``spread`` is nflverse ``spread_line``, POSITIVE when
#      the home team is favored, and it IS the cover threshold on the home-MARGIN scale -- the same
#      scale ``model_spread`` (a predicted home margin) lives on. ``_determine_bet_side_ats`` and
#      ``apply_slippage_spread`` are the two helpers written in the OPPOSITE "line" convention, so
#      the strategy NEGATES INTO them and NEGATES BACK OUT. ``_resolve_ats_outcome`` already
#      compares an actual margin against its threshold, so the slipped market spread reaches it
#      UN-negated. The legacy simulator path keeps its pre-existing convention (DEF-31-02).
#      Every assertion below is paired with a control proving the OLD reading gives a DIFFERENT
#      answer on the same row -- a test that passes under both conventions would pin nothing.
#   2. Neither new strategy has an eligibility gate (D31-05), and neither can emit ``not_subpop``.
#      Since D33.2-24 the same holds for the O/U strategy, and the test below asserts all three.
#   3. The winner strategy is priced and sized at its OWN moneyline, never at the flat -110 the
#      other two targets are quoted at.
# ---------------------------------------------------------------------------

# The ATS frozen SD is on the HOME-MARGIN scale and the O/U one is on the TOTAL scale. They are
# different quantities that happen to be numerically similar, so they are named separately here
# and are separate arguments to ``default_strategies`` -- one shared ``frozen_sd`` would be a
# category error waiting to be made.
_ATS_FIXTURE_SD = 13.0
_ATS_FIXTURE_BIAS = {2021: 0.5, 2022: 0.5}
_WP_FIXTURE_BIAS = {2021: 0.0, 2022: 0.0}


def _ats_row(
    game_id: str,
    *,
    model_spread: float,
    closing_spread: float,
    actual_margin: float,
    season: int = 2021,
    week: int = 1,
) -> dict:
    """One ATS candidate: a predicted home MARGIN against the market's home MARGIN (DEF-31-01).

    BOTH are POSITIVE when the home team is favored, and both are on the HOME-MARGIN scale.
    ``model_spread`` is the model's predicted home margin; ``closing_spread`` is the stored
    nflverse ``spread_line``, which is the margin the home team must EXCEED to cover. The two are
    directly comparable, which is the whole content of the 2026-09-04 ruling.
    """
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "target": "ats",
        "model_spread": model_spread,
        "closing_spread": closing_spread,
        "actual": actual_margin,
        "sportsbook": "consensus",
        "is_live": False,
    }


def _wp_row(
    game_id: str,
    *,
    model_prob: float,
    ml_home: float,
    ml_away: float,
    actual_home_win: int,
    season: int = 2021,
    week: int = 1,
) -> dict:
    """One WP candidate: the deployed isotonic P(home) against BOTH real moneylines."""
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "target": "wp",
        "model_prob": model_prob,
        "ml_home": ml_home,
        "ml_away": ml_away,
        "actual": actual_home_win,
        "sportsbook": "consensus",
        "is_live": False,
    }


def _three_target_strategies(**overrides):
    """The production three-strategy registry over the fixture parameters."""
    from backtest.selector_strategies import default_strategies

    kwargs = {
        "ou_frozen_sd": _FIXTURE_SD,
        "ou_season_bias_by_season": _FIXTURE_BIAS,
        "ats_frozen_sd": _ATS_FIXTURE_SD,
        "ats_season_bias_by_season": _ATS_FIXTURE_BIAS,
        "wp_season_bias_by_season": _WP_FIXTURE_BIAS,
    }
    kwargs.update(overrides)
    return default_strategies(**kwargs)


def _three_target_selector(ev_floor_t: float = 0.0, **overrides):
    """A BetSelector registering all three production strategies."""
    return _make_selector(
        ev_floor_t=ev_floor_t, strategies=_three_target_strategies(**overrides)
    )


class TestThreeTargetRegistry:
    """All three targets select through ONE facade in ONE call (D31-01/02)."""

    def test_all_three_strategies_conform_to_the_protocol(self) -> None:
        """Structural conformance, checked the same way the O/U strategy already is."""
        from backtest.selector_strategies import (
            ATSStrategy,
            TargetStrategy,
            WPStrategy,
        )

        strategies = _three_target_strategies()
        assert [s.target for s in strategies] == ["wp", "ats", "ou"]
        for strategy in strategies:
            assert isinstance(strategy, TargetStrategy)
        assert isinstance(strategies[0], WPStrategy)
        assert isinstance(strategies[1], ATSStrategy)

    def test_a_mixed_week_produces_records_for_all_three_targets(self) -> None:
        """One ``select`` call over a hand-built mixed week yields all three target codes.

        Asserted on the UNFILTERED cross-check, which carries every candidate regardless of the
        decision, so the claim is about the registry rather than about which bets happened to win
        admission on this fixture.
        """
        week = [
            _as_target(
                _ou_row(
                    "2021_W01_A@B",
                    model_total=38.0,
                    closing_total=45.0,
                    actual_total=40.0,
                )
            ),
            _ats_row(
                "2021_W01_C@D",
                model_spread=7.0,
                closing_spread=-3.0,
                actual_margin=10.0,
            ),
            _wp_row(
                "2021_W01_E@F",
                model_prob=0.75,
                ml_home=-150.0,
                ml_away=130.0,
                actual_home_win=1,
            ),
        ]
        result = _three_target_selector().select(week)
        assert {r["target"] for r in result.unfiltered} == {"wp", "ats", "ou"}
        assert {r["target"] for r in result.selected} == {"wp", "ats", "ou"}


class TestSpreadStrategySignConventions:
    """The spread strategy is on the MEASURED convention (DEF-31-01, ruled 2026-09-04).

    The stored ``spread`` is nflverse ``spread_line``: POSITIVE when the home team is favored, and
    it IS the cover threshold the actual home margin must EXCEED. ``model_spread`` is a predicted
    home margin on the SAME scale. Every test below carries a control showing the OLD reading --
    "the market spread is a line, negative when home is favored" -- returns a DIFFERENT answer on
    the very same row, so none of them can pass under both conventions.
    """

    def _strategy(self):
        from backtest.selector_strategies import ATSStrategy

        return ATSStrategy(
            frozen_sd=_ATS_FIXTURE_SD, season_bias_by_season=_ATS_FIXTURE_BIAS
        )

    def test_side_compares_two_margins_and_not_a_margin_against_a_line(self) -> None:
        """A row where the two readings produce OPPOSITE sides, and its mirror.

        ``model_spread=+1`` (model: home wins by 1) against ``closing_spread=-3`` (market: home
        LOSES by 3, because a negative stored spread is a home UNDERDOG). The model is four points
        more bullish on the home team, so the bet is ``home_cover``. The OLD reading fed the
        negated margin against the raw stored spread and returned ``away_cover`` on this row.
        """
        strategy = self._strategy()

        home_row = _ats_row(
            "2021_W01_A@B", model_spread=1.0, closing_spread=-3.0, actual_margin=10.0
        )
        assert strategy.resolve_bet_side(home_row) == "home_cover"
        # The control: the OLD call shape really does return the opposite side on this row.
        assert strategy._sim._determine_bet_side_ats(-1.0, -3.0) == "away_cover"

        # The mirror, so the pin is not an artifact of one sign. Market: home favored by 3; model:
        # home loses by 1. The model is four points LESS bullish on home, so it is an away bet.
        away_row = _ats_row(
            "2021_W01_C@D", model_spread=-1.0, closing_spread=3.0, actual_margin=10.0
        )
        assert strategy.resolve_bet_side(away_row) == "away_cover"
        assert strategy._sim._determine_bet_side_ats(1.0, 3.0) == "home_cover"

    def test_slippage_moves_the_stored_spread_against_the_bettor(self) -> None:
        """On a -3.0 stored spread the home-cover threshold RISES and the away-cover one FALLS.

        Home covers when ``actual_margin > spread``, so making the bet harder for a home-cover
        bettor means raising the threshold: -3.0 becomes -2.5. For an away-cover bettor -- who
        needs ``actual_margin < spread`` -- it means lowering it: -3.0 becomes -3.5. The LOCKED
        ``apply_slippage_spread`` is written in the opposite (line) convention and returns exactly
        the SWAPPED pair, which is the control below; the strategy negates into it and back out
        rather than re-implementing it.
        """
        from backtest.simulation import apply_slippage_spread

        strategy = self._strategy()
        row = _ats_row(
            "2021_W01_A@B", model_spread=7.0, closing_spread=-3.0, actual_margin=10.0
        )

        _, home_slipped = strategy.side_probability(row, "home_cover")
        _, away_slipped = strategy.side_probability(row, "away_cover")
        assert home_slipped == -2.5
        assert away_slipped == -3.5

        # Delegation, pinned numerically: each value is the LOCKED helper evaluated in ITS
        # convention and converted back, not a locally written +/- 0.5.
        assert home_slipped == -apply_slippage_spread(3.0, "home_cover", 0.5)
        assert away_slipped == -apply_slippage_spread(3.0, "away_cover", 0.5)

        # The control: calling the helper on the RAW stored spread -- what the old reading did --
        # returns the two values SWAPPED, i.e. it moves both bets in the bettor's favour.
        assert apply_slippage_spread(-3.0, "home_cover", 0.5) == -3.5
        assert apply_slippage_spread(-3.0, "away_cover", 0.5) == -2.5

    def test_probability_is_measured_against_the_slipped_stored_spread(self) -> None:
        """P(home cover) is hand-computed from scratch against the slipped spread, un-negated.

        The slipped stored spread for a home-cover bet on -3.0 is -2.5, and that IS the cover
        threshold. The old reading measured against +3.5 and returned about 0.62 where the correct
        value is about 0.78, so the two are nowhere near each other.
        """
        from scipy.stats import norm

        strategy = self._strategy()
        row = _ats_row(
            "2021_W01_A@B", model_spread=7.0, closing_spread=-3.0, actual_margin=10.0
        )
        p_side, slipped_line = strategy.side_probability(row, "home_cover")

        assert slipped_line == -2.5
        corrected_margin = 7.0 + _ATS_FIXTURE_BIAS[2021]
        expected = 1.0 - float(norm.cdf((-2.5 - corrected_margin) / _ATS_FIXTURE_SD))
        assert p_side == pytest.approx(expected, abs=1e-12)

        # The control: the OLD threshold (+3.5, the negated old slipped line) is far away.
        old_convention = 1.0 - float(
            norm.cdf((3.5 - corrected_margin) / _ATS_FIXTURE_SD)
        )
        assert abs(expected - old_convention) > 0.10

    def test_grading_uses_the_slipped_stored_spread_un_negated(self) -> None:
        """A margin that covers under the measured convention and loses under the old one.

        Home wins by 1 against a slipped stored spread of -2.5. Home covers when the margin
        EXCEEDS the spread, and 1.0 > -2.5, so the home-cover bet WINS. Negating the threshold
        first -- the old reading -- would compare ``1.0 > 2.5`` and call the same bet a LOSS.
        """
        strategy = self._strategy()
        record = {
            "bet_side": "home_cover",
            "slipped_line": -2.5,
            "_actual_total": 1.0,
        }
        assert strategy.grade(record) is True

        # The control: the old, negated call really does return the opposite outcome.
        assert strategy._sim._resolve_ats_outcome("home_cover", 1.0, 2.5) is False

    def test_a_push_lands_exactly_on_the_slipped_stored_spread(self) -> None:
        """The push is at ``actual_margin == slipped_line``, carried as None and never coerced."""
        strategy = self._strategy()
        record = {"bet_side": "home_cover", "slipped_line": -2.5, "_actual_total": -2.5}
        assert strategy.grade(record) is None

        # The control: the old reading put the push at the NEGATED value, and this strategy grades
        # that margin as an ordinary win rather than as a push.
        assert (
            strategy.grade(
                {"bet_side": "home_cover", "slipped_line": -2.5, "_actual_total": 2.5}
            )
            is True
        )


class TestWinnerStrategyPricesAtItsOwnMoneyline:
    """The winner target is quoted per game, so a flat -110 payout would be a made-up price."""

    def test_per_bet_ev_uses_the_side_moneyline_not_flat_110(self) -> None:
        """A home favourite at -320 is NOT a bet; at a flat -110 payout it would look like one."""
        week = [
            _wp_row(
                "2021_W01_A@B",
                model_prob=0.75,
                ml_home=-320.0,
                ml_away=260.0,
                actual_home_win=1,
            )
        ]
        result = _three_target_selector().select(week)
        record = result.unfiltered[0]

        assert record["bet_side"] == "home"
        assert record["selected_odds"] == -320.0
        # p * payout - (1 - p) with payout = 100/320.
        assert record["per_bet_ev"] == pytest.approx(
            0.75 * (100.0 / 320.0) - 0.25, abs=1e-12
        )
        # At the flat -110 payout the same bet would price at +0.4318 and be selected.
        assert 0.75 * (100.0 / 110.0) - 0.25 > 0.0
        assert [r["rejection_reason"] for r in result.rejected] == ["ev_below_floor"]
        assert result.selected == []

    def test_kelly_sizes_against_the_side_moneyline(self) -> None:
        """The Kelly stake is computed at the bet's OWN price, not at the reference juice."""
        from utils.kelly_criterion import KellyCalculator, KellyMode

        # 0.62, NOT a larger probability: a bigger edge saturates the 5% per-bet cap at BOTH
        # prices, and two capped stakes are equal no matter which odds produced them -- the test
        # would then pass while proving nothing. At 0.62 both stakes are below the cap.
        week = [
            _wp_row(
                "2021_W01_A@B",
                model_prob=0.62,
                ml_home=-150.0,
                ml_away=130.0,
                actual_home_win=1,
            )
        ]
        result = _three_target_selector().select(week)
        assert len(result.selected) == 1
        record = result.selected[0]
        assert record["selected_odds"] == -150.0

        def _stake(odds: int) -> float:
            calc = KellyCalculator(
                starting_bankroll=10_000.0,
                max_bet_pct=0.05,
                base_unit_size=100.0,
                default_kelly_fraction=0.25,
                confidence_threshold=0.0,
            )
            return calc.calculate_optimal_bet_size(
                model_prob=0.62, market_odds=odds, mode=KellyMode.FRACTIONAL
            ).recommended_bet

        assert record["kelly_stake"] == pytest.approx(_stake(-150), abs=1e-9)
        assert _stake(-150) != pytest.approx(_stake(-110), abs=1e-9)
        # Neither stake is at the 5% per-bet ceiling, so the difference is the PRICE and not the
        # cap, whose ceiling here is five percent of a ten-thousand bankroll.
        assert _stake(-150) < 500.0
        assert _stake(-110) < 500.0


# ---------------------------------------------------------------------------
# DEF-31-13 (owner ruling, 2026-09-05): the two LINE targets price on the stored two-sided juice.
#
# The fixtures below are chosen so the 5% per-bet Kelly cap is NOT binding. A bigger edge
# saturates that cap at BOTH prices, and two capped stakes are equal no matter which odds produced
# them -- the sizing half of these tests would then pass while proving nothing (the same trap the
# WP moneyline test above documents).
# ---------------------------------------------------------------------------

# One ATS candidate whose home-cover stake lands well inside the cap (p_side ~ 0.575).
_JUICE_ATS_ARGS = {
    "model_spread": -1.0,
    "closing_spread": -3.0,
    "actual_margin": 10.0,
}
# One O/U candidate: an UNDER pick on a high total, stake ~ 145 of a 500 ceiling (p_side ~ 0.570).
_JUICE_OU_ARGS = {
    "model_total": 48.0,
    "closing_total": 49.0,
    "actual_total": 40.0,
}


def _with_juice(row: dict, **prices: float | None) -> dict:
    """Attach stored juice columns to a candidate row.

    The base row helpers deliberately do NOT take juice arguments: the columns are OPTIONAL on a
    candidate frame (they are absent for most stored seasons), and every other test in this module
    exercises the absent case by construction.
    """
    return {**row, **prices}


class TestTheLineTargetsPriceOnTheStoredTwoSidedJuice:
    """DEF-31-13, ruled 2026-09-05: the frozen pre-registration's devig step governs.

    ``PROFITABILITY-PREREGISTRATION.md`` section 3.2 step 4 devigs the real two-sided spread
    prices and section 3.3 step 4 devigs the real two-sided total prices. D31-04 called those two
    targets "flat-quoted" and gave the optional ``bet_odds`` member to ``WPStrategy`` alone, so
    both fell through to -110. Two ratified documents disagreed and the owner ruled that the frozen
    pre-registration governs the 2025 verdict, so D31-04's characterisation is SUPERSEDED for the
    selection path.

    The three tests below are the three things that had to be true for that to be a PRICE change
    and not an arithmetic one: it binds where a real asymmetric price exists, it is inert where the
    stored price really is -110, and it degrades to the documented flat fallback -- visibly -- where
    no price is stored at all.
    """

    def test_asymmetric_stored_juice_moves_the_price_ev_stake_and_payout(self) -> None:
        """A stored price other than -110 moves all four published numbers, in the right direction.

        The comparison is against the SAME candidate with no juice columns, so the only difference
        between the two runs is the price. All four numbers a bet publishes are checked -- the
        price, the per-bet EV, the Kelly stake and the flat payout -- because the price feeds each
        of them by a different route and a partial wiring would leave one of the four still struck
        at -110.
        """
        from backtest.ou_ev_chain import american_to_payout, per_bet_ev
        from backtest.profitability_2025 import _per_bet_frame

        cases = (
            (
                _ats_row("2021_W01_C@D", **_JUICE_ATS_ARGS),
                {"spread_ju_home": -105.0, "spread_ju_away": -115.0},
                -105,
            ),
            (
                _as_target(_ou_row("2021_W01_A@B", **_JUICE_OU_ARGS)),
                {"total_over_ju": -120.0, "total_under_ju": 100.0},
                100,
            ),
        )

        for base_row, prices, expected_odds in cases:
            flat = _three_target_selector().select([base_row]).selected
            juiced = (
                _three_target_selector()
                .select([_with_juice(base_row, **prices)])
                .selected
            )
            assert len(flat) == len(juiced) == 1, base_row["game_id"]
            flat_bet, juiced_bet = flat[0], juiced[0]

            # (1) the price itself, and its provenance label
            assert flat_bet["selected_odds"] == -110
            assert flat_bet["devig_method"] == "flat_-110"
            assert juiced_bet["selected_odds"] == expected_odds
            assert juiced_bet["devig_method"] == "real_two_sided"

            # The side and the calibrated probability are UNCHANGED: this is a price change, so a
            # moved p_side would mean the arithmetic moved with it.
            assert juiced_bet["bet_side"] == flat_bet["bet_side"]
            assert juiced_bet["calibrated_p_side"] == flat_bet["calibrated_p_side"]

            # (2) the per-bet EV, computed at the real payout
            p_side = juiced_bet["calibrated_p_side"]
            assert juiced_bet["per_bet_ev"] == pytest.approx(
                per_bet_ev(p_side, american_to_payout(expected_odds)), abs=1e-12
            )
            assert juiced_bet["per_bet_ev"] != pytest.approx(
                flat_bet["per_bet_ev"], abs=1e-9
            )

            # (3) the Kelly stake, sized at the real price and NOT at the per-bet ceiling
            assert juiced_bet["kelly_stake"] != pytest.approx(
                flat_bet["kelly_stake"], abs=1e-9
            )
            assert max(juiced_bet["kelly_stake"], flat_bet["kelly_stake"]) < 500.0

            # (4) the flat payout the profitability runner publishes
            flat_payout = _per_bet_frame([flat_bet])["payout_flat"].iloc[0]
            juiced_payout = _per_bet_frame([juiced_bet])["payout_flat"].iloc[0]
            assert flat_payout != pytest.approx(juiced_payout, abs=1e-9)
            assert juiced_payout == pytest.approx(
                american_to_payout(expected_odds), abs=1e-12
            )

    def test_a_stored_price_that_really_is_minus_110_reproduces_the_flat_numbers(
        self,
    ) -> None:
        """The proof that the PRICE SOURCE changed and the arithmetic did not.

        A row carrying a genuine symmetric -110 / -110 two-sided price produces a record equal
        FIELD FOR FIELD, with ``==`` and no tolerance, to the same row with no juice columns at
        all -- except for ``devig_method``, which is precisely the field that tells the two apart.
        If the ruling had changed any step of the chain rather than only where the price is read
        from, this is the test that would fail.
        """
        cases = (
            (
                _ats_row("2021_W01_C@D", **_JUICE_ATS_ARGS),
                {"spread_ju_home": -110.0, "spread_ju_away": -110.0},
            ),
            (
                _as_target(_ou_row("2021_W01_A@B", **_JUICE_OU_ARGS)),
                {"total_over_ju": -110.0, "total_under_ju": -110.0},
            ),
        )

        for base_row, prices in cases:
            flat = _three_target_selector().select([base_row]).selected[0]
            real = (
                _three_target_selector()
                .select([_with_juice(base_row, **prices)])
                .selected[0]
            )

            assert flat["devig_method"] == "flat_-110"
            assert real["devig_method"] == "real_two_sided"
            assert real["selected_odds"] == flat["selected_odds"] == -110

            drift = {
                key
                for key in set(flat) | set(real)
                if key != "devig_method" and flat.get(key) != real.get(key)
            }
            assert drift == set(), (
                f"{base_row['game_id']}: a real -110 must reproduce the flat numbers exactly; "
                f"these fields moved: {sorted(drift)}"
            )

    def test_absent_or_null_juice_falls_back_to_the_reference_juice_and_says_so(
        self,
    ) -> None:
        """The documented flat -110 fallback (D27-13), and it is legible rather than inferred.

        Three shapes of "no stored price" are driven: the columns absent entirely (every
        pre-promotion stored season), both columns null (a DataFrame's empty cell), and ONE side
        priced (which cannot be devigged, so it is not a two-sided price either). None raises, none
        prices at zero, and every one of them is labelled ``flat_-110`` -- which is what makes an
        absent price distinguishable from the genuine -110 the test above pins.
        """
        nan = float("nan")
        shapes = (
            ("absent", {}, {}),
            (
                "null",
                {"spread_ju_home": nan, "spread_ju_away": nan},
                {"total_over_ju": nan, "total_under_ju": nan},
            ),
            (
                "one-sided",
                {"spread_ju_home": -105.0, "spread_ju_away": None},
                {"total_over_ju": None, "total_under_ju": 100.0},
            ),
        )

        for label, ats_prices, ou_prices in shapes:
            week = [
                _with_juice(_ats_row("2021_W01_C@D", **_JUICE_ATS_ARGS), **ats_prices),
                _with_juice(
                    _as_target(_ou_row("2021_W01_A@B", **_JUICE_OU_ARGS)), **ou_prices
                ),
            ]
            result = _three_target_selector().select(week)

            assert len(result.unfiltered) == 2, label
            for record in result.unfiltered:
                assert record["selected_odds"] == -110, (label, record["target"])
                assert record["devig_method"] == "flat_-110", (label, record["target"])
                assert record["kelly_stake"] > 0.0, (label, record["target"])

    def test_a_malformed_stored_price_is_refused_rather_than_read_as_absent(
        self,
    ) -> None:
        """A present-but-non-numeric price is a DATA DEFECT and must not wear the absent label.

        Filing it under ``flat_-110`` would hide a broken ingest behind the same label an honestly
        absent price carries, which is the distinction ``_juice_price`` exists to keep.
        """
        from backtest.selector_strategies import ATSStrategy

        strategy = ATSStrategy(
            frozen_sd=_ATS_FIXTURE_SD, season_bias_by_season=_ATS_FIXTURE_BIAS
        )
        with pytest.raises(ValueError, match="spread_ju_home"):
            strategy.bet_odds(
                {"spread_ju_home": "not-a-price", "spread_ju_away": -110.0},
                "home_cover",
            )

    def test_an_unknown_side_is_refused_rather_than_reading_the_wrong_half(
        self,
    ) -> None:
        """The two-sided price has two halves, so guessing a side would price the wrong one."""
        from backtest.selector_strategies import ATSStrategy, OUStrategy

        ats = ATSStrategy(
            frozen_sd=_ATS_FIXTURE_SD, season_bias_by_season=_ATS_FIXTURE_BIAS
        )
        ou = OUStrategy(frozen_sd=_FIXTURE_SD, season_bias_by_season=_FIXTURE_BIAS)
        with pytest.raises(ValueError, match="unknown ATS bet side"):
            ats.bet_odds({"spread_ju_home": -105.0, "spread_ju_away": -115.0}, "over")
        with pytest.raises(ValueError, match="unknown O/U bet side"):
            ou.bet_odds(
                {"total_over_ju": -105.0, "total_under_ju": -115.0}, "home_cover"
            )

    def test_the_reader_and_the_writer_spell_the_juice_columns_the_same_way(
        self,
    ) -> None:
        """The four column names live in three modules; a typo in any one prices nothing.

        The ingest script WRITES them, the ATS chain and the O/U strategy READ them, and the
        profitability loader CARRIES them. A misspelling anywhere would fall back to -110 forever
        with no error at all, which is exactly the silent failure the ruling was opened over.
        """
        from backtest.ats_ev_chain import ATS_JUICE_FIELDS
        from backtest.profitability_2025 import _JUICE_COLUMNS_BY_TARGET
        from backtest.selector_strategies import OU_JUICE_FIELDS
        from scripts.ingest_historical_odds import JUICE_COLUMNS

        assert (*ATS_JUICE_FIELDS, *OU_JUICE_FIELDS) == JUICE_COLUMNS
        assert _JUICE_COLUMNS_BY_TARGET["ats"] == ATS_JUICE_FIELDS
        assert _JUICE_COLUMNS_BY_TARGET["ou"] == OU_JUICE_FIELDS
        # WP needs none: both moneylines are already REQUIRED market fields for that target, so it
        # can never reach the fallback at all.
        assert _JUICE_COLUMNS_BY_TARGET["wp"] == ()

    def test_the_loader_carries_the_juice_only_when_the_odds_table_has_it(self) -> None:
        """Coverage is a fact about the stored table, not a softened requirement.

        The stored silver odds table carries the four columns for some seasons and not others, so
        the loader intersects with what is present. A target whose columns are entirely absent
        prices at the fallback and says so on every record -- it does not raise, and it does not
        silently drop the target.
        """
        from backtest.profitability_2025 import _juice_columns_for
        from scripts.ingest_historical_odds import JUICE_COLUMNS

        full = ["game_id", "spread", "total", *JUICE_COLUMNS]
        assert _juice_columns_for("ats", full) == ["spread_ju_home", "spread_ju_away"]
        assert _juice_columns_for("ou", full) == ["total_over_ju", "total_under_ju"]
        assert _juice_columns_for("wp", full) == []
        assert _juice_columns_for("ats", ["game_id", "spread"]) == []
        assert _juice_columns_for("ou", ["game_id", "total", "total_over_ju"]) == [
            "total_over_ju"
        ]


class TestNoEligibilityGateOnTheTwoNewTargets:
    """D31-05 gave ATS and WP no sub-population; D33.2-24 took O/U's away. No target says ``not_subpop``.

    The class name is kept (it is how this contract has been cited since Plan 31-10); its scope
    widened to the O/U strategy when D33.2-24 deleted the O/U eligibility UNION.
    """

    def test_neither_new_strategy_can_emit_not_subpop(self) -> None:
        """Behavioural AND structural: the reason never appears, and the string is not in scope.

        The behavioural half drives a grid of sided and sideless rows; the structural half reads
        the three class bodies, because a strategy that only happened not to reach the branch on
        this fixture would pass the behavioural half alone.
        """
        import inspect

        from backtest.selector_strategies import ATSStrategy, OUStrategy, WPStrategy

        ats = ATSStrategy(
            frozen_sd=_ATS_FIXTURE_SD, season_bias_by_season=_ATS_FIXTURE_BIAS
        )
        wp = WPStrategy(season_bias_by_season=_WP_FIXTURE_BIAS)
        ou = OUStrategy(frozen_sd=_FIXTURE_SD, season_bias_by_season=_FIXTURE_BIAS)
        ou_rows = [
            _ou_row("g7", model_total=42.0, closing_total=38.0, actual_total=41.0),
            _ou_row("g8", model_total=38.0, closing_total=45.0, actual_total=40.0),
            _ou_row("g9", model_total=45.0, closing_total=45.0, actual_total=40.0),
        ]

        ats_rows = [
            _ats_row("g1", model_spread=7.0, closing_spread=-3.0, actual_margin=1.0),
            _ats_row("g2", model_spread=-7.0, closing_spread=-3.0, actual_margin=1.0),
            _ats_row("g3", model_spread=3.0, closing_spread=-3.0, actual_margin=1.0),
        ]
        wp_rows = [
            _wp_row(
                "g4", model_prob=0.75, ml_home=-150.0, ml_away=130.0, actual_home_win=1
            ),
            _wp_row(
                "g5", model_prob=0.25, ml_home=150.0, ml_away=-170.0, actual_home_win=0
            ),
            _wp_row(
                "g6", model_prob=0.50, ml_home=-110.0, ml_away=-110.0, actual_home_win=1
            ),
        ]
        reasons = (
            {ats.eligibility(row, ats.resolve_bet_side(row)) for row in ats_rows}
            | {wp.eligibility(row, wp.resolve_bet_side(row)) for row in wp_rows}
            | {ou.eligibility(row, ou.resolve_bet_side(row)) for row in ou_rows}
        )
        assert "not_subpop" not in reasons
        assert reasons == {None, "no_bet_side"}

        for cls in (ATSStrategy, WPStrategy, OUStrategy):
            assert "not_subpop" not in inspect.getsource(cls)

    def test_a_sideless_candidate_is_suppressed_as_no_bet_side(self) -> None:
        """A model that agrees with the market inside the LOCKED band has no bet to price.

        It is reported with its own reason rather than as an eligibility failure (there is no
        eligibility rule on these targets) or as an EV failure (nothing was priced, so no EV was
        measured). ``model_prob == 0.5`` is inside ``_determine_bet_side_wp``'s no-bet band.
        """
        from backtest.bet_selector import REJECTION_REASONS

        assert "no_bet_side" in REJECTION_REASONS
        week = [
            _wp_row(
                "2021_W01_A@B",
                model_prob=0.50,
                ml_home=-110.0,
                ml_away=-110.0,
                actual_home_win=1,
            )
        ]
        result = _three_target_selector().select(week)
        assert [(r["game_id"], r["rejection_reason"]) for r in result.rejected] == [
            ("2021_W01_A@B", "no_bet_side")
        ]
        assert result.selected == []
        # Nothing was priced, so nothing is claimed about the expected value.
        assert result.rejected[0]["per_bet_ev"] is None
        assert result.rejected[0]["calibrated_p_side"] is None


class TestSideResolutionDelegatesToTheLockedSimulator:
    """Neither new strategy re-implements the side convention (D-18)."""

    def test_resolve_bet_side_calls_the_locked_method_and_compares_nothing(
        self,
    ) -> None:
        """An AST scan: one call to the target's LOCKED method, no comparison, no side literal.

        A behavioural test cannot tell a delegation apart from a re-implementation that agrees on
        the fixture, so the shape is pinned in the source instead.
        """
        import inspect
        import textwrap

        from backtest.selector_strategies import ATSStrategy, OUStrategy, WPStrategy

        expected_locked = {
            ATSStrategy: "_determine_bet_side_ats",
            WPStrategy: "_determine_bet_side_wp",
            OUStrategy: "_determine_bet_side_ou",
        }
        for cls, locked_name in expected_locked.items():
            tree = ast.parse(textwrap.dedent(inspect.getsource(cls.resolve_bet_side)))
            called = {
                node.func.attr
                for node in ast.walk(tree)
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            }
            assert locked_name in called or "_bet_side" in called, (
                f"{cls.__name__}.resolve_bet_side does not delegate to {locked_name}"
            )
            assert not [n for n in ast.walk(tree) if isinstance(n, ast.Compare)], (
                f"{cls.__name__}.resolve_bet_side contains its own comparison"
            )
            returned_literals = {
                node.value.value
                for node in ast.walk(tree)
                if isinstance(node, ast.Return)
                and isinstance(node.value, ast.Constant)
                and isinstance(node.value.value, str)
            }
            assert not returned_literals, (
                f"{cls.__name__}.resolve_bet_side returns a hard-coded side string"
            )


# ---------------------------------------------------------------------------
# WR-06: an UNPLAYED game is UNGRADED, in every target and in every frame shape
# ---------------------------------------------------------------------------


class TestAnUnplayedGameIsNeverGraded:
    """The push / ungraded / absent contracts were written against ``None`` and arrive as NaN.

    Every candidate frame on both live paths -- ``weekly_bet_list.select_weekly_bets`` and
    ``profitability_2025._select`` -- is a DataFrame, and a DataFrame spells a missing cell as
    NaN, never ``None``. ``float("nan") is None`` is False, so the ``is None`` guards in the three
    ``grade()`` implementations let a NaN straight through:

    * ``OUStrategy``  -> ``abs(nan - line) < 1e-9`` False (no push) and ``nan > line`` False, so an
      UNDER on a game nobody has played is graded a WIN and an OVER a LOSS;
    * ``ATSStrategy`` -> ``nan > line`` False, so the away side covers by default;
    * ``WPStrategy``  -> ``int(nan)`` raises "cannot convert float NaN to integer", which
      propagates out of ``BetSelector.select`` and fails the whole selection.

    Reproduced on this checkout before the fix: an O/U candidate carrying ``actual = NaN`` came
    back ``outcome=True``.

    The module already knew the difference -- ``_is_absent`` existed precisely to catch NaN -- it
    just was not used at the grading seam.
    """

    @staticmethod
    def _selected(rows: list[dict]) -> dict[str, dict]:
        selector = _three_target_selector(ev_floor_t=0.0)
        result = selector.select(pd.DataFrame(rows))
        return {r["game_id"]: r for r in result.selected}

    def test_an_ou_bet_on_an_unplayed_game_is_ungraded_not_a_win(self) -> None:
        rows = [
            _ou_row(
                "2021_W01_AAA@BBB",
                model_total=38.0,
                closing_total=45.0,
                actual_total=float("nan"),
            ),
            _ou_row(
                "2021_W02_CCC@DDD",
                model_total=39.0,
                closing_total=46.0,
                actual_total=40.0,
            ),
        ]
        for row in rows:
            row["target"] = "ou"

        selected = self._selected(rows)

        assert selected["2021_W01_AAA@BBB"]["outcome"] is None, (
            "a game with no realized total was GRADED. Under the old `is None` guard the NaN "
            "reached _resolve_ou_outcome, which found no push and no over, and paid the under."
        )
        assert selected["2021_W02_CCC@DDD"]["outcome"] is not None, (
            "the played game came back ungraded too, so this module proves nothing"
        )

    def test_an_ats_bet_on_an_unplayed_game_is_ungraded_not_an_away_cover(self) -> None:
        rows = [
            _ats_row(
                "2021_W01_EEE@FFF",
                model_spread=7.0,
                closing_spread=1.0,
                actual_margin=float("nan"),
            ),
            _ats_row(
                "2021_W02_GGG@HHH",
                model_spread=7.0,
                closing_spread=1.0,
                actual_margin=10.0,
            ),
        ]

        selected = self._selected(rows)

        assert selected["2021_W01_EEE@FFF"]["outcome"] is None
        assert selected["2021_W02_GGG@HHH"]["outcome"] is not None

    def test_a_wp_bet_on_an_unplayed_game_is_ungraded_and_does_not_crash_the_selection(
        self,
    ) -> None:
        """The WP symptom is not a wrong grade but a ``ValueError`` out of the whole ``select``."""
        rows = [
            _wp_row(
                "2021_W01_III@JJJ",
                model_prob=0.75,
                ml_home=-150,
                ml_away=130,
                actual_home_win=float("nan"),  # type: ignore[arg-type]
            ),
            _wp_row(
                "2021_W02_KKK@LLL",
                model_prob=0.75,
                ml_home=-150,
                ml_away=130,
                actual_home_win=1,
            ),
        ]

        selected = self._selected(rows)

        assert selected["2021_W01_III@JJJ"]["outcome"] is None
        assert selected["2021_W02_KKK@LLL"]["outcome"] is not None

    def test_the_stash_normalizes_absence_to_one_spelling(self) -> None:
        """``_actual_total`` is None, not NaN -- so the push/ungraded split downstream is right.

        ``profitability_2025._measure_and_judge`` classifies a push as
        ``outcome is None and _actual_total is not None``. A NaN there is ``not None``, so an
        UNPLAYED game would be counted as a PUSH on the verdict artifact.
        """
        rows = [
            _ou_row(
                "2021_W01_MMM@NNN",
                model_total=38.0,
                closing_total=45.0,
                actual_total=float("nan"),
            )
        ]
        rows[0]["target"] = "ou"

        record = self._selected(rows)["2021_W01_MMM@NNN"]

        assert record["_actual_total"] is None, (
            f"_actual_total came back as {record['_actual_total']!r}; the downstream "
            "push-versus-ungraded split reads `is not None` and would count an unplayed game "
            "as a push"
        )
