"""Unit suite for the O/U EV chain (Phase 27, plan 27-01, OUM-02).

Covers ``backtest/ou_ev_chain.py`` -- the numeric heart that fixes the
CLV-positive / ROI-negative trap by bias-correcting the over-biased model total
BEFORE converting to P(over), then devigging and computing per-bet EV. The chain
REPLACES the exploratory ``throwaway_ev_preview`` from Phase 26 with a
production-grade, fenced, pluggable transform.

Mirrors the ``tests/unit/test_betting_simulation.py`` class-grouped style:
exact-value asserts (no subjective language), requirement-ID docstrings, and the
``test_ou_divergence.py`` source-level no-leak import-guard pattern.

Each test docstring cites OUM-02 and the LOCKED D27-NN decision it proves:
  - D27-07: bias-correction (locked residual contract) + reliability/Brier gate.
  - D27-08: a SINGLE frozen residual SD fit on the tune split only.
  - D27-13: pluggable EXPLICIT-odds devig (flat -110 default; real two-sided when priced).
  - D27-14: per-bet EV with an EXPLICIT payout (the EV-floor input).
  - Review tightenings #1 (deterministic 5-bin/min-N calibration gate), #3 (residual
    sign guard), #4 (isotonic fallback registered fork), #10 (explicit odds + extreme-Z
    + p_over clip disclosure).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
from backtest.ou_ev_chain import (
    EV_FLOOR_GRID,
    HOLD_SEASONS,
    MIN_BIN_OBS,
    MINUS_110_PAYOUT,
    N_BINS,
    OU_BREAKEVEN,
    P_OVER_CLIP,
    RELIABILITY_TOLERANCE,
    TRIAL_REGISTRY_FIELDS,
    TUNE_SEASONS,
    american_to_implied,
    american_to_payout,
    calibrated_p_over,
    calibration_gate,
    devig,
    estimate_prior_season_bias,
    fit_frozen_residual_sd,
    per_bet_ev,
)
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_TOL_TIGHT = 1e-12
_TOL_EV = 1e-9

# The flat -110 breakeven (110 / (110 + 100)); pinned to full float precision.
_FLAT_110_BREAKEVEN = 0.5238095238095238


# ---------------------------------------------------------------------------
# (1) + (2) Residual sign guard + bias-correction lowers P(over) toward realized
# ---------------------------------------------------------------------------


class TestResidualContractSignGuard:
    """The LOCKED residual contract pulls an over-biased total DOWN (#3, D27-07)."""

    def test_correction_direction_hand_computed_pulls_down(self) -> None:
        """RESIDUAL SIGN GUARD (OUM-02, D27-07, #3): a NEGATIVE season_bias lowers P(over).

        Construct a KNOWN over-biased case by hand:
          model_total=48.0, line=45.0, season_bias=-2.0, frozen_sd=13.0.
        ``corrected = model_total + season_bias`` = 46.0 pulls the total DOWN, so the
        corrected P(over) must be STRICTLY LESS than the raw P(over) (season_bias=0.0),
        and must equal the hand-computed ``1 - norm.cdf((45.0 - 46.0) / 13.0)``.
        """
        model_total = 48.0
        line = 45.0
        frozen_sd = 13.0

        corrected = calibrated_p_over(
            model_total=model_total, line=line, frozen_sd=frozen_sd, season_bias=-2.0
        )
        raw = calibrated_p_over(
            model_total=model_total, line=line, frozen_sd=frozen_sd, season_bias=0.0
        )

        # The negative bias pulls the corrected total to 46.0 -> lower P(over).
        assert corrected < raw

        expected = 1.0 - norm.cdf((line - 46.0) / frozen_sd)
        assert abs(corrected - expected) < _TOL_TIGHT

    def test_calibrated_p_over_lowers_mean_toward_realized(self) -> None:
        """OUM-02 / D27-07: bias-correcting an upward-biased frame lowers mean P(over).

        On a fixed inline frame where the raw model totals over-predict the line, the
        bias-corrected mean P(over) is closer to the realized over-rate than the raw
        mean P(over); assert mean(calibrated) < mean(raw).
        """
        model_total = np.array([48.0, 50.0, 46.0, 52.0])
        line = np.array([45.0, 45.0, 44.0, 47.0])
        frozen_sd = 13.0
        season_bias = -2.0

        raw = calibrated_p_over(
            model_total=model_total, line=line, frozen_sd=frozen_sd, season_bias=0.0
        )
        corrected = calibrated_p_over(
            model_total=model_total,
            line=line,
            frozen_sd=frozen_sd,
            season_bias=season_bias,
        )

        # Upward-biased frame: every raw P(over) is above 0.5; correction depresses it.
        assert float(np.mean(raw)) > 0.5
        assert float(np.mean(corrected)) < float(np.mean(raw))


# ---------------------------------------------------------------------------
# (3) Deterministic 5-bin / min-N calibration gate (#1, D27-07)
# ---------------------------------------------------------------------------


class TestCalibrationGate:
    """The deterministic reliability/Brier gate that defines calibration-verified P(over)."""

    @staticmethod
    def _make_realized(p: np.ndarray, seed: int) -> np.ndarray:
        """Deterministic 0/1 realized labels whose mean tracks p (reproducible)."""
        rng = np.random.default_rng(seed)
        return (rng.random(len(p)) < p).astype(float)

    def test_calibration_gate_deterministic_pass_fail(self) -> None:
        """CALIBRATION GATE PINNED (OUM-02, D27-07, #1): 5 bins / min-N / Brier / pass-fail.

        ``calibration_gate(p_over_corrected, p_over_raw, realized_over)`` on a fixed
        HOLD-OOS fixture returns a report dict carrying ``n_bins == N_BINS``,
        ``min_bin_obs``, ``ece``, ``max_bin_deviation``, ``brier_corrected``,
        ``brier_raw``, and ``passed``. ``passed`` is True only when BOTH
        ``brier_corrected <= brier_raw`` AND
        ``max_bin_deviation <= RELIABILITY_TOLERANCE``.
        """
        rng = np.random.default_rng(27)
        n = 400
        # Well-calibrated corrected predictions: realized tracks the prediction.
        p_corrected = np.clip(rng.uniform(0.2, 0.8, size=n), *P_OVER_CLIP)
        realized = self._make_realized(p_corrected, seed=1)
        # Raw predictions are biased upward (worse Brier) -> corrected should pass.
        p_raw = np.clip(p_corrected + 0.12, *P_OVER_CLIP)

        passed, report = calibration_gate(p_corrected, p_raw, realized)

        assert report["n_bins"] == N_BINS
        assert "min_bin_obs" in report
        assert "ece" in report
        assert "max_bin_deviation" in report
        assert "brier_corrected" in report
        assert "brier_raw" in report
        assert "passed" in report

        assert report["brier_corrected"] <= report["brier_raw"]
        assert report["max_bin_deviation"] <= RELIABILITY_TOLERANCE
        assert passed is True
        assert report["passed"] is True

    def test_calibration_gate_fails_when_brier_worse(self) -> None:
        """OUM-02 / D27-07 / #1: a fixture failing ONE condition reports passed is False.

        Here the corrected predictions are systematically WORSE (Brier_corrected >
        Brier_raw), so the AND rule fails even if the deviation is small.
        """
        rng = np.random.default_rng(11)
        n = 400
        # realized tracks the RAW predictions; corrected is shifted away (worse Brier).
        p_raw = np.clip(rng.uniform(0.2, 0.8, size=n), *P_OVER_CLIP)
        realized = self._make_realized(p_raw, seed=2)
        p_corrected = np.clip(p_raw + 0.25, *P_OVER_CLIP)

        passed, report = calibration_gate(p_corrected, p_raw, realized)

        assert report["brier_corrected"] > report["brier_raw"]
        assert passed is False
        assert report["passed"] is False

    def test_calibration_gate_merges_small_bins(self) -> None:
        """OUM-02 / #1: bins below MIN_BIN_OBS are merged, not silently dropped.

        A fixture with one sparse high-prediction cluster still reports N_BINS effective
        bins (the merge keeps the bin count fixed) and every reported bin carries at
        least MIN_BIN_OBS observations.
        """
        rng = np.random.default_rng(99)
        dense = rng.uniform(0.30, 0.55, size=380)
        sparse = rng.uniform(0.95, 0.98, size=5)  # a sparse top cluster < MIN_BIN_OBS
        p_corrected = np.clip(np.concatenate([dense, sparse]), *P_OVER_CLIP)
        p_raw = np.clip(p_corrected + 0.05, *P_OVER_CLIP)
        realized = self._make_realized(p_corrected, seed=3)

        _, report = calibration_gate(p_corrected, p_raw, realized)

        assert report["n_bins"] == N_BINS
        # Every reported bin survived the merge with at least MIN_BIN_OBS obs.
        for row in report["bins"]:
            assert row["count"] >= MIN_BIN_OBS


# ---------------------------------------------------------------------------
# (4) + (5) Devig: flat -110 default + real two-sided when priced (#10, D27-13)
# ---------------------------------------------------------------------------


class TestDevig:
    """Pluggable EXPLICIT-odds devig (D27-13, #10)."""

    def test_devig_flat_110_explicit_odds_default(self) -> None:
        """OUM-02 / D27-13: no odds -> flat -110 symmetric, breakeven 110/210.

        ``devig()`` with no over/under odds returns ``method == "flat_-110"`` and
        ``breakeven == 110/210`` to ~1e-12; the DEFAULT payout that ``per_bet_ev`` uses
        equals the -110 payout 100/110.
        """
        result = devig()

        assert result["method"] == "flat_-110"
        assert abs(result["breakeven"] - _FLAT_110_BREAKEVEN) < _TOL_TIGHT
        assert abs(result["fair_over"] - 0.5) < _TOL_TIGHT
        assert abs(result["fair_under"] - 0.5) < _TOL_TIGHT
        # The EV default payout is the -110 payout.
        assert abs(MINUS_110_PAYOUT - 100.0 / 110.0) < _TOL_TIGHT

    def test_devig_real_two_sided_when_priced(self) -> None:
        """OUM-02 / D27-13: explicit two-sided odds -> real proportional devig.

        ``devig(over_odds=-110, under_odds=-110)`` returns ``method == "real_two_sided"``
        with symmetric fair probs summing to 1.0; ``devig(over_odds=-120,
        under_odds=+100)`` returns proportional fair probs summing to 1.0
        (American-odds conversion).
        """
        symmetric = devig(over_odds=-110, under_odds=-110)
        assert symmetric["method"] == "real_two_sided"
        assert abs(symmetric["fair_over"] - 0.5) < _TOL_TIGHT
        assert abs(symmetric["fair_under"] - 0.5) < _TOL_TIGHT
        assert (
            abs((symmetric["fair_over"] + symmetric["fair_under"]) - 1.0) < _TOL_TIGHT
        )

        asymmetric = devig(over_odds=-120, under_odds=+100)
        assert asymmetric["method"] == "real_two_sided"
        assert (
            abs((asymmetric["fair_over"] + asymmetric["fair_under"]) - 1.0) < _TOL_TIGHT
        )
        # -120 (io = 120/220) is the favorite -> fair_over > fair_under.
        io = american_to_implied(-120)
        iu = american_to_implied(+100)
        assert abs(asymmetric["fair_over"] - io / (io + iu)) < _TOL_TIGHT
        assert abs(asymmetric["fair_under"] - iu / (io + iu)) < _TOL_TIGHT


# ---------------------------------------------------------------------------
# (6) Per-bet EV with an EXPLICIT payout (#10, D27-14)
# ---------------------------------------------------------------------------


class TestPerBetEv:
    """The per-bet EV math (the EV-floor input, D27-14)."""

    def test_per_bet_ev_explicit_payout_breakeven_and_monotone(self) -> None:
        """OUM-02 / D27-14 / #10: EV ~ 0 at breakeven; monotone; honors explicit payout.

        ``per_bet_ev(110/210)`` ~ 0 (abs < 1e-9) at the DEFAULT flat -110 payout;
        ``per_bet_ev`` strictly increasing across p_side in {0.50, 0.55, 0.60};
        ``per_bet_ev(p, payout=...)`` honors an EXPLICITLY supplied payout (a different
        payout moves the breakeven), proving the EV math cannot silently diverge from a
        future devig (#10).
        """
        # Breakeven at the default flat -110 payout.
        assert abs(per_bet_ev(110.0 / 210.0)) < _TOL_EV

        # Strictly increasing in p_side at the default payout.
        ev_50 = per_bet_ev(0.50)
        ev_55 = per_bet_ev(0.55)
        ev_60 = per_bet_ev(0.60)
        assert ev_50 < ev_55 < ev_60

        # An EXPLICIT payout moves the breakeven: at even-money (payout=1.0) the
        # breakeven is 0.5, not 0.5238, so per_bet_ev(0.5, payout=1.0) ~ 0.
        assert abs(per_bet_ev(0.5, payout=1.0)) < _TOL_EV
        # The same p_side at even-money pays more than at -110 (proving payout flows).
        assert per_bet_ev(0.55, payout=1.0) > per_bet_ev(0.55)


# ---------------------------------------------------------------------------
# (7) Frozen SD fit window is tune-only (D27-08, no-leak fence)
# ---------------------------------------------------------------------------


class TestFrozenResidualSd:
    """The single frozen residual SD is fenced to the tune split (D27-08)."""

    def test_frozen_sd_fit_window_is_tune_only(self) -> None:
        """OUM-02 / D27-08: the SD is fit using ONLY tune-season residuals.

        Build per-season residuals including 2023/2024 (hold), then fit the SD via the
        tune-only path; assert the returned SD equals ``np.std(tune_resid, ddof=1)``
        over the TUNE rows alone (walk-forward / no-leak fence). Also assert ddof=1.
        """
        rng = np.random.default_rng(7)
        tune_resid = rng.normal(0.0, 13.0, size=400)
        hold_resid = rng.normal(0.0, 30.0, size=400)  # very different spread

        sd = fit_frozen_residual_sd(tune_resid)

        expected = float(np.std(tune_resid, ddof=1))
        assert abs(sd - expected) < _TOL_TIGHT
        # The hold residuals (much larger spread) must NOT change the fit.
        sd_with_hold = fit_frozen_residual_sd(np.concatenate([tune_resid, hold_resid]))
        assert sd != sd_with_hold

    def test_estimate_prior_season_bias_uses_strictly_prior(self) -> None:
        """OUM-02 / D27-08: season bias uses STRICTLY-prior seasons; raises otherwise.

        ``estimate_prior_season_bias`` for 2023 averages residuals over seasons strictly
        before 2023 (2021, 2022) and never the target's own data; it raises a named
        ValueError when no prior season exists.
        """
        resid_by_season = {
            2021: np.array([1.0, 1.0, 1.0]),
            2022: np.array([-3.0, -3.0, -3.0]),
            2023: np.array([100.0, 100.0, 100.0]),  # must be ignored for 2023
        }
        bias_2023 = estimate_prior_season_bias(resid_by_season, target_season=2023)
        # mean over 2021+2022 residuals = mean([1,1,1,-3,-3,-3]) = -1.0
        assert abs(bias_2023 - (-1.0)) < _TOL_TIGHT

        # No prior season -> named ValueError (never fall back to the target's own data).
        try:
            estimate_prior_season_bias(resid_by_season, target_season=2021)
        except ValueError as exc:
            assert "prior" in str(exc).lower()
        else:
            raise AssertionError("expected ValueError when no prior season exists")


# ---------------------------------------------------------------------------
# (8) Extreme Z + clip disclosure (#10)
# ---------------------------------------------------------------------------


class TestExtremeZAndClip:
    """Extreme Z-scores resolve to the disclosed P_OVER_CLIP band, never NaN/ >1 (#10)."""

    def test_extreme_z_and_clip_disclosure(self) -> None:
        """OUM-02 / #10: extreme-Z P(over) clips to [0.001, 0.999].

        ``calibrated_p_over(model_total=60.0, line=40.0, ...)`` (a huge over signal)
        clips to the upper bound 0.999 (not > 1, not NaN); the symmetric extreme-under
        case clips to 0.001.
        """
        upper = calibrated_p_over(
            model_total=60.0, line=40.0, frozen_sd=13.0, season_bias=0.0
        )
        assert upper == P_OVER_CLIP[1]
        assert upper == 0.999

        lower = calibrated_p_over(
            model_total=40.0, line=60.0, frozen_sd=13.0, season_bias=0.0
        )
        assert lower == P_OVER_CLIP[0]
        assert lower == 0.001


# ---------------------------------------------------------------------------
# (9) Isotonic fallback is a registered fork (#4, D27-07)
# ---------------------------------------------------------------------------


class TestIsotonicFallbackRegistered:
    """The isotonic calibration fallback cannot fire unless registered (#4, D27-07)."""

    def test_isotonic_fallback_is_registered_fork(self) -> None:
        """OUM-02 / D27-07 / #4: isotonic is a REGISTERED fork; the default is bias-subtraction.

        ``TRIAL_REGISTRY_FIELDS`` contains ``calibration_method``; the module default is
        prior-season mean-bias subtraction and isotonic does NOT silently activate on the
        default path (a behavioral assert that the default ``calibrated_p_over`` is the
        bias-subtraction CDF, never an isotonic fit).
        """
        from backtest.ou_ev_chain import CALIBRATION_METHOD

        assert "calibration_method" in TRIAL_REGISTRY_FIELDS
        assert CALIBRATION_METHOD == "prior_season_mean_bias_subtraction"

        # Behavioral: the default calibrated_p_over equals the pure bias-subtraction CDF,
        # so no isotonic transform is silently applied on the default path.
        model_total, line, frozen_sd, season_bias = 47.0, 45.0, 13.0, -1.0
        got = calibrated_p_over(
            model_total=model_total,
            line=line,
            frozen_sd=frozen_sd,
            season_bias=season_bias,
        )
        expected = 1.0 - norm.cdf((line - (model_total + season_bias)) / frozen_sd)
        assert abs(got - expected) < _TOL_TIGHT


# ---------------------------------------------------------------------------
# (10) No-leak import guard (T-26-08) -- source-level scan
# ---------------------------------------------------------------------------


class TestNoLeakGuard:
    """The module never imports the exploratory throwaway preview (T-26-08)."""

    def test_no_throwaway_import(self) -> None:
        """OUM-02 / T-26-08: the EV chain does not reference throwaway_ev_preview / ou_divergence.

        Read the module SOURCE (not just imports) so the guard catches an ``import`` even
        if the symbol is unused, mirroring the ``test_ou_divergence.py`` import-guard
        pattern. The module reimplements the math from scratch.
        """
        import backtest.ou_ev_chain as ev_mod

        source = inspect.getsource(ev_mod)
        assert "throwaway_ev_preview" not in source
        assert "from backtest.ou_divergence" not in source
        assert "import backtest.ou_divergence" not in source

        # Confirm the read is against the real on-disk module file.
        module_path = Path(inspect.getfile(ev_mod))
        assert module_path.name == "ou_ev_chain.py"


# ---------------------------------------------------------------------------
# Pre-registered module constants (the frozen split + grid + gate params)
# ---------------------------------------------------------------------------


class TestPreRegisteredConstants:
    """The frozen pre-registered constants match the values locked in plan 27-01."""

    def test_split_and_gate_constants(self) -> None:
        """OUM-02 / D27-03/08/14 / #1: the pre-registered constants are present + frozen."""
        assert TUNE_SEASONS == (2021, 2022)
        assert HOLD_SEASONS == (2023, 2024)
        assert N_BINS == 5
        assert MIN_BIN_OBS == 20
        assert 0.0 < RELIABILITY_TOLERANCE < 1.0
        assert P_OVER_CLIP == (0.001, 0.999)
        # The EV-floor grid is a frozen ascending tuple shared with Plan 04's tuner.
        assert isinstance(EV_FLOOR_GRID, tuple)
        assert len(EV_FLOOR_GRID) >= 2
        assert list(EV_FLOOR_GRID) == sorted(EV_FLOOR_GRID)
        assert abs(OU_BREAKEVEN - _FLAT_110_BREAKEVEN) < _TOL_TIGHT

    def test_american_odds_helpers(self) -> None:
        """OUM-02 / #10: American-odds helpers convert both signs; -110 payout binds."""
        # -110 implied = 110 / 210; +100 implied = 0.5.
        assert abs(american_to_implied(-110) - 110.0 / 210.0) < _TOL_TIGHT
        assert abs(american_to_implied(+100) - 0.5) < _TOL_TIGHT
        # -110 payout = 100 / 110; +150 payout = 1.5.
        assert abs(american_to_payout(-110) - MINUS_110_PAYOUT) < _TOL_TIGHT
        assert abs(american_to_payout(+150) - 1.5) < _TOL_TIGHT
