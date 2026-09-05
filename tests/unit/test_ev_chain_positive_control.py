"""Per-target positive controls and empty-input cases (Phase 31, plan 31-07; SPEC R1).

WHY A POSITIVE CONTROL IS WHAT MAKES A ZERO RESULT EVIDENCE
------------------------------------------------------------
SPEC R1 makes a chain that emits ZERO bets an explicitly defined PASS -- but only when its
positive control demonstrates the chain WOULD emit on synthetic positive-edge input.
Without that control, "zero" is ambiguous between "no edge" and "broken chain", and the
frozen verdict vocabulary's split between ``UNDISCHARGEABLE_NO_BETS`` and
``UNDISCHARGEABLE_NO_CHAIN`` is exactly what the control buys: the first token asserts the
chain RAN and declined, and nothing but a control can support that assertion.

The empty case is asserted beside it, in the same module and for the same reason: a chain
over zero candidates must return an empty record list and a NULL CLV report WITHOUT raising,
and the positive control must still run. A chain that raised on an empty frame would report
NO_CHAIN for a week with no games.

WHAT EACH ARM EXERCISES, AND WHY THEY ARE NOT THE SAME SHAPE
-------------------------------------------------------------
O/U's ``OUStrategy`` exists today, so the O/U arm runs the REAL end-to-end
``BetSelector.select`` path -- the single bet-decision source (LOCKED-2). ATS and WP have no
registered strategy until Plan 31-10, and ``backtest/selector_strategies.py`` states its own
rule that a stubbed strategy which cannot decide is worse than an absent one. Their arms
therefore exercise the chain PRICING functions and apply the EV floor in the test. That is
not a weaker claim about the chain -- pricing is the whole of what these plans built -- but
it is a different one, and it is stated rather than blurred.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from backtest.ats_ev_chain import (
    THRESHOLD_WINDOW_P31,
    ChainFit,
    price_ats_candidates,
)
from backtest.bet_selector import BetSelector
from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD
from backtest.ou_ev_chain import EV_FLOOR_GRID, MIN_BIN_OBS, N_BINS
from backtest.selector_strategies import OUStrategy
from backtest.wp_ev_chain import (
    WP_PROB_CLIP,
    apply_wp_fallback_correction,
    calibrated_p_home_side,
    price_wp_candidates,
    run_wp_calibration_gate,
    wp_two_sided_prices,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
WP_CHAIN_PATH = REPO_ROOT / "backtest" / "wp_ev_chain.py"

# The lowest pre-registered EV floor. A positive control asserts the chain CAN clear a floor,
# so it uses the most permissive one in the frozen grid; the armed run selects its own t by
# the tune-side sweep.
LOWEST_EV_FLOOR: float = EV_FLOOR_GRID[0]

# The synthetic O/U control's fitted parameters. A tight SD and a mild negative bias are what
# make a 15-point model-vs-market gap resolve to a near-certain UNDER; production values are
# neither used nor implied here.
_OU_FROZEN_SD: float = 5.0
_OU_SEASON_BIAS: dict[int, float] = {2025: -1.0}


def _wp_fit(season_bias_by_season: dict[int, float] | None = None) -> ChainFit:
    """A WP ``ChainFit``. ``frozen_sd`` is None BY DESIGN (D31-07).

    WP fits no residual standard deviation: a calibrated classifier has no residual to take
    one of, and inventing a logit-space SD was explicitly REJECTED. The None is the
    pre-registration showing through the type, not an omission.
    """
    return ChainFit(
        target="wp",
        frozen_sd=None,
        season_bias_by_season=season_bias_by_season or {},
        tune_fit_seasons=(2021, 2022, 2023, 2024),
        threshold_window=THRESHOLD_WINDOW_P31,
        bias_pool_by_season={},
    )


def _ats_fit(season_bias_by_season: dict[int, float] | None = None) -> ChainFit:
    """An ATS ``ChainFit`` with a frozen residual SD, which ATS DOES fit (unlike WP)."""
    return ChainFit(
        target="ats",
        frozen_sd=11.0,
        season_bias_by_season=season_bias_by_season or {2025: 0.5},
        tune_fit_seasons=(2021, 2022, 2023, 2024),
        threshold_window=THRESHOLD_WINDOW_P31,
        bias_pool_by_season={2024: (2021, 2022, 2023)},
    )


def _ou_selector(ev_floor_t: float = 0.0) -> BetSelector:
    """A selector registered with the O/U strategy ALONE -- the real Phase-27 path.

    Built through the public facade with an explicitly injected ``OUStrategy`` so the arm
    exercises the single bet-decision source (LOCKED-2) rather than a parallel pricing
    function. ``frozen_sd`` and the season bias are the synthetic control's, not production
    values; the control asks whether the chain CAN emit, never what it should emit.
    """
    return BetSelector(
        frozen_sd=_OU_FROZEN_SD,
        season_bias_by_season=_OU_SEASON_BIAS,
        ev_floor_t=ev_floor_t,
        bankroll=10_000.0,
        high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
        strategies=[
            OUStrategy(
                frozen_sd=_OU_FROZEN_SD,
                season_bias_by_season=_OU_SEASON_BIAS,
                high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
            )
        ],
    )


def _well_calibrated_tune_split() -> tuple[np.ndarray, np.ndarray]:
    """A tune split whose realized rate equals its predicted rate in every bin.

    Five blocks of 40 at 0.1 / 0.3 / 0.5 / 0.7 / 0.9, each carrying exactly its own rate in
    wins. ``n = 200`` clears the gate's ``N_BINS * MIN_BIN_OBS`` precondition with room, and
    equal-count binning over five distinct predicted values lands one block per bin, so every
    per-bin deviation is exactly zero.
    """
    probabilities: list[float] = []
    realized: list[float] = []
    for level in (0.1, 0.3, 0.5, 0.7, 0.9):
        block = 40
        wins = round(level * block)
        probabilities.extend([level] * block)
        realized.extend([1.0] * wins + [0.0] * (block - wins))
    return np.asarray(probabilities), np.asarray(realized)


def _miscalibrated_tune_split() -> tuple[np.ndarray, np.ndarray]:
    """A tune split that predicts 0.9 everywhere and realizes 0.1 everywhere.

    Every bin's ``|mean_pred - realized|`` is 0.8, far outside the frozen
    ``RELIABILITY_TOLERANCE`` of 0.10, so the gate FAILS and the registered fallback fires.
    """
    n = 200
    probabilities = np.full(n, 0.9)
    realized = np.asarray([1.0 if index % 10 == 0 else 0.0 for index in range(n)])
    return probabilities, realized


class TestWpPositiveControl:
    """WP emits on synthetic positive-edge input, so a WP zero means no edge."""

    def test_wp_positive_control_emits_at_least_one_bet(self) -> None:
        """SPEC R1 / D31-08: a large synthetic edge clears the lowest frozen EV floor."""
        rows = [
            {
                "game_id": "2025_01_AAA",
                "season": 2025,
                "week": 1,
                # A 95% home win probability against a pick-em price is an enormous,
                # deliberately unrealistic edge. The control asks whether the chain CAN
                # emit, not whether this edge is plausible.
                "model_prob": 0.95,
                "ml_home": -110,
                "ml_away": -110,
            },
            {
                "game_id": "2025_01_BBB",
                "season": 2025,
                "week": 1,
                "model_prob": 0.04,
                "ml_home": -110,
                "ml_away": -110,
            },
        ]
        pricing = price_wp_candidates(rows, _wp_fit())

        clearing = [
            record
            for record in pricing.records
            if record["per_bet_ev"] is not None
            and record["per_bet_ev"] >= LOWEST_EV_FLOOR
        ]
        assert len(clearing) >= 1, (
            "the WP chain priced no candidate above the lowest pre-registered EV floor on "
            "synthetic positive-edge input. A zero-bet 2025 result would then be "
            "uninterpretable: UNDISCHARGEABLE_NO_BETS asserts the chain ran and declined, "
            "and only this control supports that assertion."
        )
        # Both sides are exercised: the 0.95 row is a home bet, the 0.04 row an away bet.
        assert {record["bet_side"] for record in pricing.records} == {"home", "away"}

    def test_wp_empty_frame_returns_empty_records_and_a_null_clv_report(self) -> None:
        """SPEC R1 empty-case: zero candidates is an empty result, never a raise."""
        pricing = price_wp_candidates([], _wp_fit())

        assert pricing.records == []
        assert pricing.clv_report is None

    def test_wp_empty_frame_does_not_disable_the_positive_control(self) -> None:
        """The control still runs after an empty pass -- the two are independent."""
        assert price_wp_candidates([], _wp_fit()).records == []
        rows = [
            {
                "game_id": "2025_01_CCC",
                "season": 2025,
                "week": 1,
                "model_prob": 0.95,
                "ml_home": -110,
                "ml_away": -110,
            }
        ]
        assert price_wp_candidates(rows, _wp_fit()).records[0]["per_bet_ev"] > 0.0


class TestWpSideProbability:
    """The two sides of one game are exactly complementary."""

    def test_wp_home_and_away_probabilities_sum_to_exactly_one(self) -> None:
        """A home bet and an away bet on the same game sum to exactly 1.0."""
        model_prob = 0.6234567890123456
        home = calibrated_p_home_side(model_prob, "home")
        away = calibrated_p_home_side(model_prob, "away")

        assert home == model_prob
        assert home + away == 1.0

    def test_wp_side_probability_rejects_an_unknown_side(self) -> None:
        """A side outside the LOCKED WP vocabulary is a caller error, never a default."""
        with pytest.raises(ValueError, match="home"):
            calibrated_p_home_side(0.6, "home_cover")


class TestWpDevig:
    """WP reuses the EXISTING real two-sided moneyline devig; it writes no devig math."""

    def test_wp_devig_passes_both_moneylines_and_selects_real_two_sided(self) -> None:
        """D31-07: both prices go into the existing ``devig``; no proportional math here.

        Hand-worked against ``american_to_implied``: -150 implies 150/250 = 0.6 and +130
        implies 100/230 = 0.434782608695652174. Their sum is the overround, and the fair
        home probability is 0.6 divided by it.
        """
        prices = wp_two_sided_prices(-150, 130)

        implied_home = 150.0 / 250.0
        implied_away = 100.0 / 230.0
        overround = implied_home + implied_away

        assert prices["method"] == "real_two_sided"
        assert prices["fair_home"] == pytest.approx(implied_home / overround, rel=1e-15)
        assert prices["fair_away"] == pytest.approx(implied_away / overround, rel=1e-15)
        assert prices["fair_home"] + prices["fair_away"] == pytest.approx(
            1.0, abs=1e-15
        )
        # The payouts come from the SAME explicit prices, so the EV cannot silently
        # diverge from a changed devig.
        assert prices["payout_home"] == pytest.approx(100.0 / 150.0, rel=1e-15)
        assert prices["payout_away"] == pytest.approx(130.0 / 100.0, rel=1e-15)

    def test_wp_devig_raises_when_a_moneyline_is_absent(self) -> None:
        """A missing moneyline never falls back to a fabricated flat -110 price.

        The flat default in ``devig`` exists for the O/U total, whose stored rows carry no
        two-sided juice. A moneyline market has a real two-sided price by construction, so
        a missing one is missing DATA and pricing it at -110 would invent a market.
        """
        with pytest.raises(ValueError, match="moneyline"):
            wp_two_sided_prices(-150, None)
        with pytest.raises(ValueError, match="moneyline"):
            wp_two_sided_prices(None, 130)


class TestWpCalibrationGate:
    """The gate runs on the TUNE split and its fallback can never fire silently."""

    def test_wp_gate_passes_and_leaves_the_fallback_unfired(self) -> None:
        """D31-07 DEFAULT path: a calibrated model is used UNCHANGED, no bias correction."""
        probabilities, realized = _well_calibrated_tune_split()
        gate = run_wp_calibration_gate(probabilities, realized)

        assert gate.passed is True
        assert gate.fallback_fired is False
        assert gate.fallback_trigger is None
        assert gate.report["max_bin_deviation"] == pytest.approx(0.0, abs=1e-12)
        assert gate.report["n_bins"] == N_BINS
        assert gate.report["min_bin_obs"] >= MIN_BIN_OBS

    def test_wp_gate_fails_and_registers_a_non_empty_fallback_trigger(self) -> None:
        """D31-07 FALLBACK path: a gate failure is RECORDED, never silent."""
        probabilities, realized = _miscalibrated_tune_split()
        gate = run_wp_calibration_gate(probabilities, realized)

        assert gate.passed is False
        assert gate.fallback_fired is True
        assert gate.fallback_trigger
        assert "max_bin_deviation" in gate.fallback_trigger
        # The trigger names the measured deviation and the tolerance it breached, so a
        # reader of the trial registry can check the trigger rather than trust it.
        assert str(round(gate.report["max_bin_deviation"], 4)) in gate.fallback_trigger

    def test_wp_gate_refuses_to_report_on_an_under_filled_tune_split(self) -> None:
        """WR-02: fewer observations than the bins can hold is a REFUSAL, not a pass.

        The refusal must also not be silently converted into a FAILURE, because a failure
        fires the registered fallback -- a chain that fell back because its tune split was
        too small would be reporting a correction it never had the evidence to justify.
        """
        n = N_BINS * MIN_BIN_OBS - 1
        probabilities = np.full(n, 0.5)
        realized = np.asarray([float(index % 2) for index in range(n)])

        with pytest.raises(ValueError, match="calibration_gate"):
            run_wp_calibration_gate(probabilities, realized)


class TestWpFallbackCorrection:
    """The registered fallback is a PROBABILITY-scale shift, clipped, with no logit."""

    def test_wp_fallback_correction_shifts_and_clips(self) -> None:
        """A prior-season probability-scale bias is ADDED, then clipped to the band."""
        assert apply_wp_fallback_correction(0.60, 0.05) == pytest.approx(
            0.65, abs=1e-15
        )
        assert apply_wp_fallback_correction(0.60, -0.05) == pytest.approx(
            0.55, abs=1e-15
        )
        # A correction that would leave the unit interval resolves to a disclosed bound.
        assert apply_wp_fallback_correction(0.99, 0.50) == WP_PROB_CLIP[1]
        assert apply_wp_fallback_correction(0.01, -0.50) == WP_PROB_CLIP[0]

    def test_wp_fallback_fires_only_when_the_gate_failed(self) -> None:
        """A passing gate leaves the priced probability byte-identical to the deployed one."""
        rows = [
            {
                "game_id": "2025_01_DDD",
                "season": 2025,
                "week": 1,
                "model_prob": 0.72,
                "ml_home": -150,
                "ml_away": 130,
            }
        ]
        passing = run_wp_calibration_gate(*_well_calibrated_tune_split())
        failing = run_wp_calibration_gate(*_miscalibrated_tune_split())
        fit = _wp_fit({2025: 0.05})

        default_record = price_wp_candidates(rows, fit, gate=passing).records[0]
        fallback_record = price_wp_candidates(rows, fit, gate=failing).records[0]

        assert default_record["p_home"] == 0.72
        assert default_record["fallback_fired"] is False
        assert fallback_record["p_home"] == pytest.approx(0.77, abs=1e-15)
        assert fallback_record["fallback_fired"] is True
        assert fallback_record["fallback_trigger"]


class TestWpInventsNoConverter:
    """R1 narrowing 3: the same CONTRACT, not the same STEPS. No SD, no logit."""

    @staticmethod
    def _module_tree() -> ast.Module:
        return ast.parse(WP_CHAIN_PATH.read_text(encoding="utf-8"), filename="wp")

    def test_wp_module_fits_no_residual_standard_deviation(self) -> None:
        """The SD estimator is neither imported nor referenced anywhere in the module.

        Scanned by AST rather than by substring, because the module docstring NAMES the
        rejected logit-space SD alternative on purpose -- a substring guard would redden on
        the very disclosure that makes the rejection checkable, and a guard that reddens on
        an innocent word gets weakened until it asserts nothing.
        """
        tree = self._module_tree()
        referenced: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id in {
                "fit_frozen_residual_sd",
                "frozen_sd",
                "residual_sd",
            }:
                referenced.append(f"{node.id} at line {node.lineno}")
            elif isinstance(node, ast.ImportFrom):
                referenced.extend(
                    f"import {alias.name} at line {node.lineno}"
                    for alias in node.names
                    if alias.name == "fit_frozen_residual_sd"
                )
        assert not referenced, (
            "backtest/wp_ev_chain.py references a residual standard deviation "
            f"({referenced}). WP has no residual to take one of; inventing a fitted "
            "parameter a calibrated classifier does not need, on the target with the "
            "worst measured CLV, is exactly what D31-07 rejected."
        )

    def test_wp_module_applies_no_logit_transform(self) -> None:
        """No log / exp / logit call: the correction is on the PROBABILITY scale."""
        tree = self._module_tree()
        offenders: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func: Any = node.func
                name = (
                    func.attr
                    if isinstance(func, ast.Attribute)
                    else (func.id if isinstance(func, ast.Name) else "")
                )
                if name in {"log", "log1p", "exp", "expit", "logit"}:
                    offenders.append(f"{name}() at line {node.lineno}")
        assert not offenders, (
            f"backtest/wp_ev_chain.py performs a log-space transform ({offenders}); the "
            "registered fallback is a PROBABILITY-scale shift (D31-07)."
        )

    def test_wp_module_writes_no_proportional_devig_arithmetic(self) -> None:
        """The devig is CONSUMED from the O/U chain; no overround division appears here."""
        tree = self._module_tree()
        imported: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "backtest.ou_ev_chain"
            ):
                imported.update(alias.name for alias in node.names)
        assert "devig" in imported
        assert "per_bet_ev" in imported
        assert "calibration_gate" in imported

        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert not (defined & {"devig", "per_bet_ev", "calibration_gate"})


class TestAtsPositiveControl:
    """ATS emits on synthetic positive-edge input, so an ATS zero means no edge."""

    def test_ats_positive_control_emits_at_least_one_bet(self) -> None:
        """SPEC R1 / D31-08: a large synthetic spread edge clears the lowest frozen floor.

        The stored spread of -3.0 has the home team as a 3-point UNDERDOG (DEF-31-01: the stored
        spread is nflverse ``spread_line``, positive when home is favored) while the model
        predicts a 20-point home MARGIN. Both are on the home-margin scale, so that is a
        23-point home-cover edge; the mirrored row is the same edge on the away side.
        """
        rows = [
            {
                "game_id": "2025_01_AAA",
                "season": 2025,
                "week": 1,
                "model_spread": 20.0,
                "closing_spread": -3.0,
            },
            {
                "game_id": "2025_01_BBB",
                "season": 2025,
                "week": 1,
                "model_spread": -20.0,
                "closing_spread": -3.0,
            },
        ]
        pricing = price_ats_candidates(rows, _ats_fit())

        clearing = [
            record
            for record in pricing.records
            if record["per_bet_ev"] is not None
            and record["per_bet_ev"] >= LOWEST_EV_FLOOR
        ]
        assert len(clearing) >= 1, (
            "the ATS chain priced no candidate above the lowest pre-registered EV floor on "
            "synthetic positive-edge input, so a zero-bet 2025 ATS result could not be "
            "distinguished from a broken chain."
        )
        assert {record["bet_side"] for record in pricing.records} == {
            "home_cover",
            "away_cover",
        }
        # The fence ran and is published on the result, not merely asserted and discarded.
        assert pricing.fence_report["fence_held"] is True

    def test_ats_empty_frame_returns_empty_records_and_a_null_clv_report(self) -> None:
        """SPEC R1 empty-case: zero candidates is an empty result, never a raise."""
        pricing = price_ats_candidates([], _ats_fit())

        assert pricing.records == []
        assert pricing.clv_report is None
        # The fence still ran: an empty week is not an excuse to skip the leakage check.
        assert pricing.fence_report["fence_held"] is True

    def test_ats_empty_frame_does_not_disable_the_positive_control(self) -> None:
        """The control still runs after an empty pass -- the two are independent."""
        assert price_ats_candidates([], _ats_fit()).records == []
        rows = [
            {
                "game_id": "2025_01_CCC",
                "season": 2025,
                "week": 1,
                "model_spread": 20.0,
                "closing_spread": -3.0,
            }
        ]
        assert price_ats_candidates(rows, _ats_fit()).records[0]["per_bet_ev"] > 0.0


class TestOuPositiveControl:
    """O/U emits through the REAL BetSelector path, so an O/U zero means no edge."""

    def test_ou_positive_control_emits_at_least_one_bet(self) -> None:
        """SPEC R1 / D31-08, through the single bet-decision source (LOCKED-2).

        A model total 15 points under a 45-point market line is an UNDER pick, which
        satisfies the frozen Phase-26/27 eligibility UNION on its under arm alone -- so the
        control does not depend on the high-total boundary and therefore cannot be quietly
        weakened by a boundary that failed to resolve.
        """
        rows = [
            {
                "game_id": "2025_01_AAA",
                "season": 2025,
                "week": 1,
                "model_total": 30.0,
                "closing_total": 45.0,
                "actual": 20.0,
            }
        ]
        result = _ou_selector().select(rows)

        assert len(result.selected) >= 1, (
            "the O/U chain selected no bet on synthetic positive-edge input, so a zero-bet "
            "2025 O/U result could not be distinguished from a broken chain."
        )
        selected = result.selected[0]
        assert selected["bet_side"] == "under"
        assert selected["kelly_stake"] > 0.0
        assert selected["per_bet_ev"] >= LOWEST_EV_FLOOR

    def test_ou_empty_frame_returns_empty_selection_and_a_null_clv_report(self) -> None:
        """SPEC R1 empty-case, on the real selector: no bets, no CLV report, no raise."""
        result = _ou_selector().select([])

        assert result.selected == []
        assert result.rejected == []
        assert result.filtered == []
        assert result.unfiltered == []
        assert result.clv_report is None

    def test_ou_empty_frame_does_not_disable_the_positive_control(self) -> None:
        """The control still runs after an empty pass -- the two are independent."""
        assert _ou_selector().select([]).selected == []
        rows = [
            {
                "game_id": "2025_01_CCC",
                "season": 2025,
                "week": 1,
                "model_total": 30.0,
                "closing_total": 45.0,
                "actual": 20.0,
            }
        ]
        assert len(_ou_selector().select(rows).selected) == 1


class TestEveryTargetHasAPositiveControl:
    """All three targets are covered -- the claim SPEC R1 actually makes."""

    def test_all_three_targets_are_covered_by_a_positive_control(self) -> None:
        """A control on two of three targets would leave the third's zero uninterpretable.

        Asserted as a property of the module rather than left to a reader counting classes,
        because the failure this guards is a target being QUIETLY dropped from the set.
        """
        module_source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(module_source, filename="positive_control")
        control_tests = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name.endswith("_positive_control_emits_at_least_one_bet")
        }
        assert control_tests == {
            "test_wp_positive_control_emits_at_least_one_bet",
            "test_ats_positive_control_emits_at_least_one_bet",
            "test_ou_positive_control_emits_at_least_one_bet",
        }, control_tests
