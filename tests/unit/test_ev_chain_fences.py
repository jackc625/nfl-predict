"""The Phase-31 fit-window fence, PROVEN TO FIRE per target (plan 31-07; T-31-28, SPEC R1).

A FENCE THAT IS NEVER PROVEN TO FIRE IS NOT A FENCE
----------------------------------------------------
The prohibition this module discharges is "MUST NOT fit any converter, residual SD, bias, or
EV floor on 2025 data". A fence that has only ever been observed passing is indistinguishable
from a fence that cannot fail, and the failure mode it guards against is specifically one that
LOOKS like a passing suite: ``backtest/ou_monetization``'s Phase-27 fence helper reads the
Phase-27 window, in which 2023-2024 are HOLD seasons. Under the Phase-31 window those two are
TUNE seasons, so a Phase-31 runner reusing that helper would fence against the wrong seasons,
raise nothing, and publish a fence report naming a hold that is not this phase's hold.

So every violation below is DELIBERATE, and each is asserted to raise ``LeakageError`` with a
message naming BOTH the offending season and the fit stage. Three violation shapes, applied to
all three targets:

  1. a 2025 row in a tune-only fit input,
  2. a hold season inside a tune-only estimator's window label,
  3. a target season inside its OWN prior-season bias window.

Shape 3 is the one the Phase-27 helper's check would have missed: it tested only for hold
seasons in the pool, which is sufficient when the whole tune window precedes the whole hold,
and insufficient here where the most direct leak available is a season debiasing itself.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backtest.ats_ev_chain import (
    FENCE_STAGE_BIAS,
    FENCE_STAGE_THRESHOLD,
    FENCE_STAGE_TUNE_FIT,
    THRESHOLD_WINDOW_P31,
    ChainFit,
    LeakageError,
    assert_fit_window_p31,
    price_ats_candidates,
)
from backtest.ev_chain_constants import (
    HOLD_SEASONS_P31,
    PRIOR_RESIDUAL_SEASONS_P31,
    TUNE_SEASONS_P31,
)
from backtest.wp_ev_chain import price_wp_candidates

REPO_ROOT = Path(__file__).resolve().parents[2]

# The Phase-27 fence helper. Phase-31 code and Phase-31 TESTS must both stay clear of it: a
# test that called it would be checking the Phase-27 window and reporting the result as if it
# were this phase's.
PHASE_27_FENCE_HELPER = "_assert_fit_window"

# The three Phase-31 test modules this plan adds, scanned below for that helper.
P31_TEST_MODULES: tuple[str, ...] = (
    "tests/unit/test_ev_chain_fences.py",
    "tests/unit/test_ev_chain_positive_control.py",
    "tests/unit/test_ev_chain_determinism.py",
)

TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

HOLD_SEASON: int = HOLD_SEASONS_P31[0]

# Candidate rows a clean fit prices without complaint. They sit in a TUNE season on purpose:
# the leak under test is in the FIT, so the row must never be the reason a call raises.
_PRICEABLE_ATS_ROWS: tuple[dict[str, object], ...] = (
    {
        "game_id": "2024_01_AAA",
        "season": 2024,
        "week": 1,
        "model_spread": 6.0,
        "closing_spread": -3.0,
    },
)

_PRICEABLE_WP_ROWS: tuple[dict[str, object], ...] = (
    {
        "game_id": "2024_01_AAA",
        "season": 2024,
        "week": 1,
        "model_prob": 0.62,
        "ml_home": -150,
        "ml_away": 130,
    },
)


def _clean_fit(target: str) -> ChainFit:
    """A fit that HOLDS: tune-only inputs, the Phase-31 window label, strictly-prior pools."""
    return ChainFit(
        target=target,
        frozen_sd=None if target == "wp" else 11.0,
        season_bias_by_season=dict.fromkeys(TUNE_SEASONS_P31, 0.5),
        tune_fit_seasons=TUNE_SEASONS_P31,
        threshold_window=THRESHOLD_WINDOW_P31,
        bias_pool_by_season={
            2021: PRIOR_RESIDUAL_SEASONS_P31,
            2022: (*PRIOR_RESIDUAL_SEASONS_P31, 2021),
            2023: (*PRIOR_RESIDUAL_SEASONS_P31, 2021, 2022),
            2024: (*PRIOR_RESIDUAL_SEASONS_P31, 2021, 2022, 2023),
        },
    )


def _fit_with(target: str, **overrides: object) -> ChainFit:
    """A clean fit with one field replaced -- so each test violates exactly one thing."""
    base = _clean_fit(target)
    fields = {
        "target": base.target,
        "frozen_sd": base.frozen_sd,
        "season_bias_by_season": base.season_bias_by_season,
        "tune_fit_seasons": base.tune_fit_seasons,
        "threshold_window": base.threshold_window,
        "bias_pool_by_season": base.bias_pool_by_season,
        "high_total_boundary": base.high_total_boundary,
    }
    fields.update(overrides)
    return ChainFit(**fields)  # type: ignore[arg-type]


class TestTheFenceHoldsOnACleanFit:
    """Anti-vacuity: the fence must PASS on a legitimate fit, or its raises prove nothing."""

    @pytest.mark.parametrize("target", TARGETS)
    def test_a_clean_fit_passes_and_reports_the_phase31_hold(self, target: str) -> None:
        """The report names the PHASE-31 hold, which is the whole point of D31-14.

        A fence that raised on everything would pass every violation test below while making
        the chain unusable, and a fence that inherited the Phase-27 window would pass this
        test while reporting ``hold_seasons`` of 2023-2024 -- seasons this phase TUNES on.
        """
        report = assert_fit_window_p31(_clean_fit(target))

        assert report["fence_held"] is True
        assert report["target"] == target
        assert report["hold_seasons"] == list(HOLD_SEASONS_P31) == [2025]
        assert (
            report["tune_seasons"]
            == list(TUNE_SEASONS_P31)
            == [
                2021,
                2022,
                2023,
                2024,
            ]
        )
        # The two seasons Phase 27 held out are TUNE seasons here. A fence report naming
        # them as hold would be the silent-inheritance failure, not a passing fence.
        assert 2023 not in report["hold_seasons"]
        assert 2024 not in report["hold_seasons"]


class TestViolationShapeOneHoldRowInAFitInput:
    """Shape 1: a 2025 row reaches a tune-only fit input."""

    @pytest.mark.parametrize("target", TARGETS)
    def test_a_2025_row_in_the_tune_only_fit_raises(self, target: str) -> None:
        """T-31-28: the frozen SD (ATS/OU) and the gate tune split (WP) never see 2025."""
        leaky = _fit_with(target, tune_fit_seasons=(*TUNE_SEASONS_P31, HOLD_SEASON))

        with pytest.raises(LeakageError) as excinfo:
            assert_fit_window_p31(leaky)

        message = str(excinfo.value)
        assert str(HOLD_SEASON) in message
        assert FENCE_STAGE_TUNE_FIT in message
        assert target in message


class TestViolationShapeTwoHoldSeasonInATuneOnlyEstimator:
    """Shape 2: a hold season enters the tune-only EV-floor sweep's window."""

    @pytest.mark.parametrize("target", TARGETS)
    def test_a_hold_season_in_the_threshold_window_raises(self, target: str) -> None:
        """The EV floor t is a fitted parameter; tuning it on 2025 spends the hold twice."""
        leaky = _fit_with(target, threshold_window=f"tune_2021_{HOLD_SEASON}")

        with pytest.raises(LeakageError) as excinfo:
            assert_fit_window_p31(leaky)

        message = str(excinfo.value)
        assert str(HOLD_SEASON) in message
        assert FENCE_STAGE_THRESHOLD in message
        assert target in message


class TestViolationShapeThreeSelfBias:
    """Shape 3: a target season appears in its OWN prior-season bias window."""

    @pytest.mark.parametrize("target", TARGETS)
    def test_a_season_in_its_own_bias_pool_raises(self, target: str) -> None:
        """A walk-forward estimate reads only EARLIER seasons; 2023 debiasing itself is a leak.

        This shape is why the Phase-31 fence adds a strict-priority check the Phase-27 helper
        did not need: a hold-only check passes on this fit, because 2023 is a TUNE season.
        """
        pools = dict(_clean_fit(target).bias_pool_by_season)
        pools[2023] = (*PRIOR_RESIDUAL_SEASONS_P31, 2021, 2022, 2023)
        leaky = _fit_with(target, bias_pool_by_season=pools)

        with pytest.raises(LeakageError) as excinfo:
            assert_fit_window_p31(leaky)

        message = str(excinfo.value)
        assert "2023" in message
        assert FENCE_STAGE_BIAS in message
        assert target in message

    @pytest.mark.parametrize("target", TARGETS)
    def test_a_hold_season_in_a_bias_pool_raises(self, target: str) -> None:
        """The 2025 outcomes can never inform the correction applied to any season."""
        pools = dict(_clean_fit(target).bias_pool_by_season)
        pools[2024] = (*PRIOR_RESIDUAL_SEASONS_P31, 2021, 2022, 2023, HOLD_SEASON)
        leaky = _fit_with(target, bias_pool_by_season=pools)

        with pytest.raises(LeakageError) as excinfo:
            assert_fit_window_p31(leaky)

        message = str(excinfo.value)
        assert str(HOLD_SEASON) in message
        assert FENCE_STAGE_BIAS in message


class TestTheChainsCallTheFence:
    """The fence is not merely available: the chains RUN it before pricing anything.

    Deleting the ``assert_fit_window_p31`` call from either pricing function makes these two
    tests fail. That was verified during execution by removing the ATS call, observing the
    failure, and reverting -- a fence nothing calls is a fence that never fires in production.

    The candidate rows below are DELIBERATELY ones the chain can price successfully. Only the
    FIT leaks. If the row itself were unpriceable the test would still go red without the
    fence, but for the wrong reason, and it would then be asserting the chain's input
    validation rather than its fence.
    """

    def test_the_ats_chain_refuses_to_price_under_a_leaking_fit(self) -> None:
        """The fence runs FIRST, so a leaking fit yields no bet list to retract later."""
        leaky = _fit_with(
            "ats",
            frozen_sd=11.0,
            tune_fit_seasons=(*TUNE_SEASONS_P31, HOLD_SEASON),
        )

        with pytest.raises(LeakageError, match=FENCE_STAGE_TUNE_FIT):
            price_ats_candidates(_PRICEABLE_ATS_ROWS, leaky)

        # And the same row DOES price under a clean fit, so the raise above is attributable
        # to the fence and to nothing else.
        clean = price_ats_candidates(
            _PRICEABLE_ATS_ROWS, _fit_with("ats", frozen_sd=11.0)
        )
        assert clean.records[0]["per_bet_ev"] is not None

    def test_the_wp_chain_refuses_to_price_under_a_leaking_fit(self) -> None:
        """WP fits no residual SD, but its gate split and fallback pool are fenced the same."""
        leaky = _fit_with("wp", threshold_window=f"tune_2021_{HOLD_SEASON}")

        with pytest.raises(LeakageError, match=FENCE_STAGE_THRESHOLD):
            price_wp_candidates(_PRICEABLE_WP_ROWS, leaky)

        clean = price_wp_candidates(_PRICEABLE_WP_ROWS, _clean_fit("wp"))
        assert clean.records[0]["per_bet_ev"] is not None

    def test_a_clean_fit_prices_normally_through_both_chains(self) -> None:
        """The fence does not block legitimate work -- the fail-open half of the control.

        It also publishes its report ONTO the pricing result, so the seasons each fit
        consumed travel with the bets rather than being asserted once and discarded.
        """
        ats_pricing = price_ats_candidates(
            _PRICEABLE_ATS_ROWS, _fit_with("ats", frozen_sd=11.0)
        )
        assert ats_pricing.fence_report["fence_held"] is True
        assert ats_pricing.fence_report["hold_seasons"] == [2025]
        assert len(ats_pricing.records) == 1

        wp_pricing = price_wp_candidates(_PRICEABLE_WP_ROWS, _clean_fit("wp"))
        assert wp_pricing.fence_report["fence_held"] is True
        assert wp_pricing.fence_report["hold_seasons"] == [2025]
        assert len(wp_pricing.records) == 1


class TestNoTestModuleReachesThePhase27Fence:
    """No Phase-31 test module calls the Phase-27 fence helper (D31-14).

    Scanned by AST, not by substring: this very module NAMES the helper in prose and in a
    string constant on purpose, so a substring guard would redden on its own documentation.
    Only a Name or Attribute node counts as a reference.
    """

    def test_no_phase31_test_module_references_the_phase27_fence_helper(self) -> None:
        """A test fencing against the Phase-27 window would report the wrong hold seasons."""
        offenders: list[str] = []
        scanned: list[str] = []
        for relative_path in P31_TEST_MODULES:
            path = REPO_ROOT / relative_path
            assert path.is_file(), (
                f"{relative_path} is missing; the scan would silently visit a shorter list "
                "and pass while asserting less than it claims."
            )
            scanned.append(relative_path)
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Name) and node.id == PHASE_27_FENCE_HELPER
                ) or (
                    isinstance(node, ast.Attribute)
                    and node.attr == PHASE_27_FENCE_HELPER
                ):
                    offenders.append(f"{relative_path}:{node.lineno}")

        assert len(scanned) == len(P31_TEST_MODULES)
        assert not offenders, (
            f"Phase-31 test module(s) reference {PHASE_27_FENCE_HELPER} at {offenders}; "
            "that helper reads the Phase-27 window (D31-14)."
        )
