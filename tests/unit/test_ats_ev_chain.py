"""Unit suite for the ATS EV chain (Phase 31, plan 31-07; SPEC R1, PROD-02).

Covers ``backtest/ats_ev_chain.py`` -- the spread-residual to cover-probability converter
that gives ATS the same pre-registration apparatus the O/U chain already carries.

THE SIGN GUARD BELOW IS HAND-COMPUTED FOR ATS AND IS NOT ADAPTED FROM THE O/U GUARD.
That is deliberate and it is the point of this module. The ATS tune-window POOLED residual
is POSITIVE (``+0.58937727047262922``), the OPPOSITE sign to O/U's, so a guard copied from
``tests/unit/test_ou_ev_chain.py`` would assert the wrong direction and would pass while
proving the reverse of what it claims. Every arithmetic step below is worked from scratch,
and the reference probability is computed through ``math.erf`` rather than through the
``scipy.stats.norm`` the module under test uses, so the assertion is an INDEPENDENT
computation and not a restatement of the implementation.

Mirrors the ``tests/unit/test_ou_ev_chain.py`` class-grouped style: exact-value asserts,
requirement-ID docstrings, and source-level import guards.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import itertools
import math
from pathlib import Path

import pytest

import backtest.ev_chain_constants as p31
from backtest.ats_ev_chain import (
    ATS_RESIDUAL_CONTRACT,
    P_COVER_CLIP,
    ats_side_probability,
    calibrated_p_cover,
    season_bias_for,
)
from backtest.ou_ev_chain import P_OVER_CLIP

REPO_ROOT = Path(__file__).resolve().parents[2]
ATS_CHAIN_PATH = REPO_ROOT / "backtest" / "ats_ev_chain.py"

# The five numeric primitives the ATS chain must CONSUME from the O/U chain rather than
# re-derive. A second implementation of any of them would be a second answer to a settled
# question (D31-14 / T-31-31).
CONSUMED_PRIMITIVES: tuple[str, ...] = (
    "devig",
    "per_bet_ev",
    "estimate_prior_season_bias",
    "fit_frozen_residual_sd",
    "calibration_gate",
)


def _standard_normal_cdf(x: float) -> float:
    """Phi(x) via ``math.erf`` -- an INDEPENDENT route to the reference probability.

    The module under test uses ``scipy.stats.norm.cdf``. Computing the expected value with
    the same call would assert only that the function is deterministic. ``math.erf`` is a
    different implementation of a different special function, related by the identity
    ``Phi(x) = 0.5 * (1 + erf(x / sqrt(2)))``, so agreement to 12 significant digits is
    evidence about the arithmetic rather than about the library.
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


class TestAtsResidualContract:
    """The ATS contract states the POSITIVE pooled direction and names its one exception."""

    def test_contract_is_the_frozen_pre_registration_by_identity(self) -> None:
        """SPEC R1: the chain does not restate the frozen contract, it BINDS to it.

        ``backtest/ev_chain_constants.py`` is frozen and ratified. A second copy of its
        prose inside the chain module would be a second source of truth that could drift,
        and the frozen half could never be edited to catch up. Identity is asserted, not
        equality, so a copy-paste that happens to match today still fails.
        """
        assert ATS_RESIDUAL_CONTRACT is p31.ATS_RESIDUAL_CONTRACT

    def test_contract_states_the_positive_pooled_direction(self) -> None:
        """The contract claims a POSITIVE pooled bias that RAISES the cover probability."""
        contract = ATS_RESIDUAL_CONTRACT
        assert "actual home margin - predicted home spread" in contract
        assert "POSITIVE" in contract
        assert "RAISES the cover probability" in contract
        assert str(p31.ATS_RESIDUAL_POOLED_P31) in contract

    def test_contract_cross_references_the_opposite_ou_sign(self) -> None:
        """A sign guard copied from O/U would assert the wrong thing; the contract says so."""
        assert "OPPOSITE sign to the O/U contract" in ATS_RESIDUAL_CONTRACT
        assert "LOWERS P(over)" in ATS_RESIDUAL_CONTRACT

    def test_contract_names_2022_and_disclaims_a_constant_sign(self) -> None:
        """REVIEW-ATS: the ONE measured negative tune season is NAMED, not hidden.

        An unstated exception is the quiet form of overstating a claim. The contract must
        also say why the negative season does not undermine the chain: the correction the
        chain applies is the PER-SEASON WALK-FORWARD estimate, which assumes no constant
        sign.
        """
        contract = ATS_RESIDUAL_CONTRACT
        assert "2022" in contract
        assert "NEGATIVE" in contract
        assert str(p31.ATS_RESIDUAL_BY_SEASON_P31[2022]) in contract
        assert "PER-SEASON WALK-FORWARD" in contract
        assert "never assumes a constant sign" in contract
        assert p31.ATS_RESIDUAL_BY_SEASON_P31[2022] < 0.0
        assert p31.ATS_RESIDUAL_POOLED_P31 > 0.0


class TestHandComputedSignGuard:
    """The direction and the scale are pinned by arithmetic worked here for ATS."""

    def test_hand_computed_probability_matches_an_independent_normal_cdf(self) -> None:
        """SPEC R1: the returned probability reproduces a hand-worked value exactly.

        Worked from scratch, in the margin convention the residual contract fixes:

            predicted home margin      =  3.0
            prior-season bias          = +1.0   (POSITIVE -- the ATS direction)
            corrected predicted margin =  3.0 + 1.0 = 4.0
            cover threshold (line)     =  2.0
            frozen residual SD         = 10.0
            z = (line - corrected) / sd = (2.0 - 4.0) / 10.0 = -0.2
            P(home cover) = 1 - Phi(-0.2) = Phi(0.2)

        The expected value is computed through ``math.erf``, never through the scipy call
        the implementation uses.
        """
        p_cover = calibrated_p_cover(
            model_spread=3.0, line=2.0, frozen_sd=10.0, season_bias=1.0
        )
        expected = _standard_normal_cdf(0.2)

        assert isinstance(p_cover, float)
        assert p_cover == pytest.approx(expected, rel=1e-12)
        # 12 significant digits, asserted as digits rather than as a relative tolerance.
        assert f"{p_cover:.12g}" == f"{expected:.12g}"

    def test_a_positive_bias_raises_the_cover_probability(self) -> None:
        """T-31-30: adding a POSITIVE bias RAISES P(home cover) -- the O/U guard inverted.

        Same case as above with the bias removed. Hand-worked:

            corrected = 3.0 + 0.0 = 3.0
            z = (2.0 - 3.0) / 10.0 = -0.1
            P(home cover) = Phi(0.1)

        Phi(0.2) > Phi(0.1), so the biased case must be STRICTLY GREATER. In the O/U chain
        the analogous assertion runs the other way, because the O/U bias is negative.
        """
        with_bias = calibrated_p_cover(
            model_spread=3.0, line=2.0, frozen_sd=10.0, season_bias=1.0
        )
        without_bias = calibrated_p_cover(
            model_spread=3.0, line=2.0, frozen_sd=10.0, season_bias=0.0
        )

        assert without_bias == pytest.approx(_standard_normal_cdf(0.1), rel=1e-12)
        assert with_bias > without_bias

    def test_raising_the_corrected_margin_strictly_raises_the_cover_probability(
        self,
    ) -> None:
        """The direction is pinned by BEHAVIOUR, not by reading the formula.

        Holds the slipped line fixed and walks the corrected predicted home margin upward;
        every step must strictly increase P(home cover). A test that only read the formula
        would still pass if the formula were rewritten with the sign flipped.
        """
        probabilities = [
            calibrated_p_cover(
                model_spread=margin, line=0.0, frozen_sd=10.0, season_bias=0.0
            )
            for margin in (-6.0, -3.0, 0.0, 3.0, 6.0)
        ]
        for lower, higher in itertools.pairwise(probabilities):
            assert higher > lower
        # And the symmetric midpoint is exactly one half: a zero corrected margin against a
        # zero line is a coin flip under any symmetric residual distribution.
        assert probabilities[2] == pytest.approx(0.5, abs=1e-15)

    def test_the_two_sides_sum_to_exactly_one(self) -> None:
        """P(away cover) is EXACTLY one minus P(home cover) inside the clip band.

        Plan 31-10's ``ATSStrategy`` maps its calibrated side probability through
        ``ats_side_probability``; the identity is asserted here so the two sides can never
        be priced as two independent numbers that happen to nearly agree.
        """
        p_home_cover = calibrated_p_cover(
            model_spread=3.0, line=2.0, frozen_sd=10.0, season_bias=1.0
        )
        assert P_COVER_CLIP[0] < p_home_cover < P_COVER_CLIP[1]

        p_home_side = ats_side_probability("home_cover", p_home_cover)
        p_away_side = ats_side_probability("away_cover", p_home_cover)

        assert p_home_side == p_home_cover
        assert p_home_side + p_away_side == 1.0

    def test_an_unknown_side_raises_rather_than_guessing(self) -> None:
        """A side string outside the LOCKED pair is a caller error, never a default."""
        with pytest.raises(ValueError, match="home_cover"):
            ats_side_probability("over", 0.6)


class TestClipBand:
    """Extreme z-scores resolve to the disclosed bounds, never to 0.0, 1.0 or NaN."""

    def test_clip_band_is_the_ou_band_by_identity(self) -> None:
        """The pre-registration says ATS is clipped to the SAME disclosed band as O/U.

        Bound by identity rather than by a matching literal: two clip bands that agree
        today and drift tomorrow would be an unregistered per-target difference.
        """
        assert P_COVER_CLIP is P_OVER_CLIP
        assert P_COVER_CLIP == (0.001, 0.999)

    def test_extreme_z_resolves_to_the_bounds(self) -> None:
        """z of plus and minus 50 returns the bounds exactly -- not 0.0, 1.0 or NaN.

        With ``frozen_sd = 1.0`` and a zero corrected margin, ``line = 50.0`` gives
        ``z = +50`` and ``line = -50.0`` gives ``z = -50``.
        """
        lower = calibrated_p_cover(
            model_spread=0.0, line=50.0, frozen_sd=1.0, season_bias=0.0
        )
        upper = calibrated_p_cover(
            model_spread=0.0, line=-50.0, frozen_sd=1.0, season_bias=0.0
        )

        assert lower == P_COVER_CLIP[0]
        assert upper == P_COVER_CLIP[1]
        assert lower != 0.0
        assert upper != 1.0
        assert not math.isnan(lower)
        assert not math.isnan(upper)

    def test_the_clip_is_disclosed_in_the_converter_docstring(self) -> None:
        """The clip bounds the Kelly tail; an undisclosed clip is a silent cap."""
        doc = calibrated_p_cover.__doc__ or ""
        assert "P_COVER_CLIP" in doc
        assert "Kelly" in doc

    def test_the_z_definition_is_pinned_in_the_converter_docstring(self) -> None:
        """The direction is recorded BY FORMULA in the docstring, not by prose alone."""
        doc = calibrated_p_cover.__doc__ or ""
        assert "z         = (line - corrected) / frozen_sd" in doc
        assert "p_cover   = clip(1 - norm.cdf(z), *P_COVER_CLIP)" in doc
        assert "P(away cover) = 1 - P(home cover)" in doc


class TestNoSilentFallback:
    """A missing per-season bias RAISES; it never degrades to the raw biased spread."""

    def test_a_missing_season_bias_raises_naming_the_season_and_estimator(self) -> None:
        """SPEC R1 / D27-07 shape: no silent fallback to the uncorrected prediction."""
        with pytest.raises(ValueError) as excinfo:
            season_bias_for(2025, {2021: 0.5, 2022: -0.1}, target="ats")

        message = str(excinfo.value)
        assert "2025" in message
        assert "estimate_prior_season_bias" in message
        assert "ats" in message

    def test_a_present_season_bias_is_returned_as_a_float(self) -> None:
        """The resolver is a lookup, not a computation: it re-derives nothing."""
        assert season_bias_for(2022, {2021: 0.25, 2022: -0.5}, target="ats") == -0.5

    def test_calibrated_p_cover_rejects_a_none_season_bias(self) -> None:
        """A None bias is the same failure as a missing one and raises identically."""
        with pytest.raises(ValueError, match="season_bias"):
            calibrated_p_cover(
                model_spread=3.0,
                line=2.0,
                frozen_sd=10.0,
                season_bias=None,  # type: ignore[arg-type]
            )


class TestPrimitivesAreConsumedNotRedefined:
    """T-31-31: every numeric primitive is IMPORTED from the O/U chain."""

    @staticmethod
    def _module_tree() -> ast.Module:
        return ast.parse(ATS_CHAIN_PATH.read_text(encoding="utf-8"), filename="ats")

    def test_no_primitive_is_defined_locally(self) -> None:
        """No local ``def`` shadows a devig, an EV formula, a bias or SD fit, or the gate."""
        tree = self._module_tree()
        defined = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        clashes = sorted(defined & set(CONSUMED_PRIMITIVES))
        assert not clashes, (
            f"backtest/ats_ev_chain.py defines {clashes} locally; every one of "
            f"{list(CONSUMED_PRIMITIVES)} must be IMPORTED from backtest.ou_ev_chain -- a "
            "second implementation is a second answer to a settled question (T-31-31)."
        )

    def test_every_primitive_is_imported_from_the_ou_chain(self) -> None:
        """Each of the five names is imported from ``backtest.ou_ev_chain`` by name."""
        tree = self._module_tree()
        imported: set[str] = set()
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == "backtest.ou_ev_chain"
            ):
                imported.update(alias.name for alias in node.names)

        missing = sorted(set(CONSUMED_PRIMITIVES) - imported)
        assert not missing, (
            f"backtest/ats_ev_chain.py does not import {missing} from backtest.ou_ev_chain."
        )

    def test_the_module_does_not_reimplement_the_normal_cdf(self) -> None:
        """The CDF comes from scipy, exactly as the O/U converter's does."""
        source = ATS_CHAIN_PATH.read_text(encoding="utf-8")
        assert "from scipy.stats import norm" in source
        assert "math.erf" not in source
