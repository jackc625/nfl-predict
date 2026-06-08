"""Tests for the LOCKED-2 end-to-end routing of simulation.py's O/U branch through BetSelector.

LOCKED-2 / BET-01 (proof == production): the O/U decision in ``backtest/simulation.py`` --
eligibility AND side AND threshold AND EV admission AND sizing -- must come from
``BetSelector.select()``, NOT a sizing-only patch and NOT the inline points-distance logic. The
LOCKED grading helpers (apply_slippage_total, _resolve_ou_outcome) and the BettingSimulator sign
convention (D-18) are reused UNCHANGED; only the DECISION ownership moves to the BetSelector. The
WP/ATS branches are OUT of scope and must be behavior-unchanged.

Tests:
- test_invariant_every_ou_bet_from_betselector: every O/U bet placed originates from
  BetSelector.select()'s selected set (the phase invariant).
- test_regression_changing_ev_floor_changes_sim_ou_bets: two different EV-floor t values yield
  different O/U bet sets (proves the sim is actually routed through the BetSelector's EV admission).
- test_source_scan_no_ou_decision_threshold_inline: the BET-02 inline O/U `implied + edge` Kelly
  branch and the points-distance O/U admission are no longer in the O/U path of simulation.py.
- test_ou_kelly_model_prob_le_one: the O/U path no longer produces a Kelly model_prob > 1 on a
  large points gap (BET-02 fix).
- test_wp_ats_sizing_unchanged: WP/ATS sizing behavior is intact (the inline implied+edge path
  remains for the non-O/U branches).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect

import pandas as pd

from backtest.bet_selector import BetSelector
from backtest.simulation import BettingSimulator, SimulationConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FROZEN_SD = 13.0
_SEASON_BIAS = {2021: -1.0, 2022: -2.0}


def _ou_predictions_frame() -> pd.DataFrame:
    """A small O/U predictions frame spanning under picks, high-total overs, and a low-total over.

    Carries the merged-odds columns (ml_home, ml_away, spread, total) + has_closing_odds so the
    simulator takes the already-merged path, plus model_total and the actual label.
    """
    rows = [
        # under pick (eligible): model below line.
        {
            "game_id": "2021_W01_AAA@BBB",
            "season": 2021,
            "week": 1,
            "model_total": 38.0,
            "actual": 40.0,
            "total": 45.0,
        },
        # high-total OVER trap (eligible via high, but EV-floor should drop it).
        {
            "game_id": "2022_W02_CCC@DDD",
            "season": 2022,
            "week": 2,
            "model_total": 52.0,
            "actual": 49.0,
            "total": 50.0,
        },
        # high-total UNDER (eligible, strong EV -> selected).
        {
            "game_id": "2022_W03_EEE@FFF",
            "season": 2022,
            "week": 3,
            "model_total": 44.0,
            "actual": 43.0,
            "total": 51.0,
        },
        # low-total OVER (NOT in the union -> never bet).
        {
            "game_id": "2021_W04_GGG@HHH",
            "season": 2021,
            "week": 4,
            "model_total": 42.0,
            "actual": 41.0,
            "total": 38.0,
        },
    ]
    frame = pd.DataFrame(rows)
    # Closing-odds columns the simulator's already-merged path expects.
    frame["ml_home"] = -110
    frame["ml_away"] = -110
    frame["spread"] = 0.0
    frame["has_closing_odds"] = True
    return frame


class _ResultsLike:
    """Minimal BacktestResults stand-in exposing only .all_predictions (the simulator contract)."""

    def __init__(self, predictions: dict[str, pd.DataFrame]) -> None:
        self.all_predictions = predictions


def _make_selector(ev_floor_t: float = 0.0) -> BetSelector:
    from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD

    return BetSelector(
        frozen_sd=_FROZEN_SD,
        season_bias_by_season=_SEASON_BIAS,
        ev_floor_t=ev_floor_t,
        bankroll=10_000.0,
        high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
    )


def _run_ou_sim(ev_floor_t: float = 0.0):
    """Run the simulator on the O/U frame with a BetSelector injected (LOCKED-2 routing)."""
    selector = _make_selector(ev_floor_t)
    sim = BettingSimulator(SimulationConfig(), ou_bet_selector=selector)
    results = sim.simulate(
        _ResultsLike({"ou": _ou_predictions_frame()}),
        closing_odds_df=pd.DataFrame(),
    )
    return results


# ---------------------------------------------------------------------------
# LOCKED-2 invariant: every O/U sim bet originates from BetSelector.select()
# ---------------------------------------------------------------------------


class TestRoutingInvariant:
    def test_invariant_every_ou_bet_from_betselector(self) -> None:
        """Every O/U bet placed appears in BetSelector.select()'s selected set (LOCKED-2).

        Spy on the injected selector's select() to record the selected game_ids, then assert the
        simulator's O/U bet records are a SUBSET of that selected set (the simulator never places an
        O/U bet the BetSelector did not return).
        """
        selector = _make_selector(ev_floor_t=0.0)

        selected_ids: set[str] = set()
        original_select = selector.select

        def _spy(candidates, raw_odds_df=None):
            result = original_select(candidates, raw_odds_df=raw_odds_df)
            selected_ids.update(r["game_id"] for r in result.selected)
            return result

        selector.select = _spy  # type: ignore[method-assign]

        sim = BettingSimulator(SimulationConfig(), ou_bet_selector=selector)
        results = sim.simulate(
            _ResultsLike({"ou": _ou_predictions_frame()}),
            closing_odds_df=pd.DataFrame(),
        )

        ou_bet_ids = {r.game_id for r in results.bet_records if r.target == "ou"}
        assert ou_bet_ids, "expected at least one O/U bet to be placed"
        assert ou_bet_ids <= selected_ids, (
            "an O/U sim bet was placed that BetSelector.select() did not return (LOCKED-2 broken)"
        )
        # The low-total OVER (not in the union) is never bet.
        assert "2021_W04_GGG@HHH" not in ou_bet_ids


# ---------------------------------------------------------------------------
# LOCKED-2 regression: changing the EV floor changes the sim's O/U bets
# ---------------------------------------------------------------------------


class TestRoutingRegression:
    def test_regression_changing_ev_floor_changes_sim_ou_bets(self) -> None:
        """Two different EV-floor t values yield DIFFERENT O/U bet sets (the sim IS routed).

        With a permissive floor (t=0.0) more O/U bets clear; with a punishing floor (t=0.9) none
        clear. If the sim were bypassing the BetSelector's EV admission, the bet set would not move.
        """
        low_floor = _run_ou_sim(ev_floor_t=0.0)
        high_floor = _run_ou_sim(ev_floor_t=0.9)

        low_ids = {r.game_id for r in low_floor.bet_records if r.target == "ou"}
        high_ids = {r.game_id for r in high_floor.bet_records if r.target == "ou"}

        assert low_ids != high_ids, (
            "changing the EV floor did not change the O/U bet set -- the sim is bypassing the "
            "BetSelector EV admission (LOCKED-2 broken)"
        )
        assert len(low_ids) > len(
            high_ids
        )  # the punishing floor admits fewer (here zero)


# ---------------------------------------------------------------------------
# LOCKED-2 source-scan / behavioral: no inline O/U decision threshold remains
# ---------------------------------------------------------------------------


class TestRoutingSourceScan:
    def test_source_scan_no_ou_decision_threshold_inline(self) -> None:
        """The inline O/U `implied + edge` Kelly branch / points-distance admission is gone.

        Source-scan the O/U-handling region of BettingSimulator.simulate: the BET-02 bug pattern
        (`kelly_model_prob = implied + edge` applied to the O/U side) must no longer appear in the
        O/U path. The WP/ATS `implied + edge` branch may remain (it is out of scope), so the scan is
        scoped to the O/U routing: the simulate source must reference BetSelector for O/U.
        """
        src = inspect.getsource(BettingSimulator.simulate)
        # The O/U path must route through the BetSelector (proof == production).
        assert "bet_selector" in src.lower() or "betselector" in src.lower(), (
            "simulate() does not reference a BetSelector for the O/U path (LOCKED-2 broken)"
        )

        # Behavioral guard (robust to source phrasing): an O/U bet is only ever placed when the
        # BetSelector returned it. A selector that returns NOTHING must yield ZERO O/U bets even
        # though the inline points-distance logic would have placed several.
        from backtest.bet_selector import SelectionResult

        empty_selector = _make_selector(ev_floor_t=0.0)
        empty_selector.select = lambda candidates, raw_odds_df=None: SelectionResult()  # type: ignore[method-assign]

        sim = BettingSimulator(SimulationConfig(), ou_bet_selector=empty_selector)
        results = sim.simulate(
            _ResultsLike({"ou": _ou_predictions_frame()}),
            closing_odds_df=pd.DataFrame(),
        )
        ou_bets = [r for r in results.bet_records if r.target == "ou"]
        assert ou_bets == [], (
            "the simulator placed an O/U bet the BetSelector did not return -- inline O/U decision "
            "logic still active (LOCKED-2 broken)"
        )


# ---------------------------------------------------------------------------
# BET-02: O/U Kelly model_prob is a probability <= 1 (never points-distance)
# ---------------------------------------------------------------------------


class TestOuKellyModelProb:
    def test_ou_kelly_model_prob_le_one(self) -> None:
        """The O/U path never sizes Kelly off a model_prob > 1 (BET-02 fix).

        On a large model_total-line gap, the legacy bug produced kelly_model_prob = implied + gap
        (e.g. ~8.5). Routed through the BetSelector, the calibrated P(side) handed to Kelly is in
        (0, 1]. Asserts every selected O/U bet's recorded model_value (the calibrated P(side)) is a
        probability <= 1.
        """
        results = _run_ou_sim(ev_floor_t=0.0)
        ou_bets = [r for r in results.bet_records if r.target == "ou"]
        assert ou_bets, "expected at least one O/U bet"
        for rec in ou_bets:
            assert 0.0 < rec.model_value <= 1.0, (
                f"O/U bet {rec.game_id} sized off a non-probability model_value={rec.model_value}"
            )


# ---------------------------------------------------------------------------
# WP/ATS unchanged (the non-O/U `else` branch is intact)
# ---------------------------------------------------------------------------


class TestWpAtsUnchanged:
    def test_wp_ats_sizing_unchanged(self) -> None:
        """WP and ATS still bet via their inline path (no BetSelector needed; out of scope).

        Build a WP frame and an ATS frame and run the simulator with NO BetSelector injected. Both
        must still place bets (the WP/ATS branches are behavior-unchanged), proving the O/U rewire
        did not break the non-O/U `else` path.
        """
        wp_frame = pd.DataFrame(
            [
                {
                    "game_id": "2021_W01_AAA@BBB",
                    "season": 2021,
                    "week": 1,
                    "model_prob": 0.70,
                    "actual": 1,
                    "ml_home": -150,
                    "ml_away": 130,
                    "spread": -3.0,
                    "total": 45.0,
                    "has_closing_odds": True,
                }
            ]
        )
        ats_frame = pd.DataFrame(
            [
                {
                    "game_id": "2021_W02_CCC@DDD",
                    "season": 2021,
                    "week": 2,
                    "model_prob": -6.0,
                    "model_spread": -6.0,
                    "actual": -7.0,
                    "ml_home": -110,
                    "ml_away": -110,
                    "spread": -3.0,
                    "total": 45.0,
                    "has_closing_odds": True,
                }
            ]
        )

        # No BetSelector injected -> WP/ATS unaffected; O/U absent so the rewire is not exercised.
        sim = BettingSimulator(SimulationConfig())
        results = sim.simulate(
            _ResultsLike({"wp": wp_frame, "ats": ats_frame}),
            closing_odds_df=pd.DataFrame(),
        )

        targets_bet = {r.target for r in results.bet_records}
        assert "wp" in targets_bet
        assert "ats" in targets_bet
