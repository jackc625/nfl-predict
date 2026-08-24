"""Wave-0 pure-logic suite for the per-target deploy gate (Phase 24, plan 24-02).

These are PURE-LOGIC tests: no canonical gold, no model training, no
``@pytest.mark.integration`` marker. They prove the gate LOGIC with synthetic CLV arrays
and tiny in-memory frames, mirroring the two idioms from
``tests/integration/test_diag_diagnosis.py`` -- the ``inspect``-free identity/parity
assertions and the synthetic-array deterministic-decision idiom (its lines 256-269 build
``rng.normal`` / ``np.full`` arrays to force a verdict).

Coverage (each test cites its 24-VALIDATION.md automated command):
  - test_gate_uses_diagnose_clv_columns / test_clv_column_parity: D24-13 import parity
    (``gate.CLV_COLUMN_FOR is diagnose.CLV_COLUMN_FOR``) -- ACTV-01 / Crit 1.
  - test_per_season_must_pass: a significantly-negative single season fails the target (D24-04).
  - test_wp_calibration_in_gate: WP ECE non-regression participates in the WP gate (D24-05).
  - test_clv_floor_near_zero_noise: a tiny-gaussian near-zero CLV passes the floor on a real,
    non-zero-variance t-test (review concern #9 -- not the np.zeros NaN path).
  - test_load_gate_config_season_keys_int: baseline season keys are int; a wrong holdout set
    raises (review concern #4).
  - test_build_candidate_bundle_shape: the single bundle builder yields
    clv_values / per_season (int keys) / accuracy (WP) or mae (ATS/OU).
  - test_forced_fail_floor_and_evaluate: a significantly-negative pooled CLV fails the floor
    and makes evaluate_target passed False.
  - test_save_does_not_autoswap: save_model_artifact defaults leave no latest.json (depends on
    Plan 24-01's update_latest=False default) -- ACTV-03 / Crit 3.
  - test_gate_config_committed: config/gate.toml exists AND is in ``git ls-files`` -- ACTV-02.
  - test_build_candidate_bundle_mae_uses_explicit_line_column: CR-01 regression guard --
    ATS/OU MAE reads model_spread/model_total (not model_prob); fails if reverted (CR-01).
  - test_build_candidate_bundle_missing_line_column_raises: CR-01 guard -- ValueError raised
    when model_spread (ATS) or model_total (OU) is absent from the scored frame (CR-01).
  - test_update_manifest_atomic_write_preserves_on_failure: WR-02 atomicity guard --
    a simulated mid-rename crash leaves latest.json byte-unchanged and no temp file (WR-02).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import copy
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import backtest.diagnose as diag
import models.deploy_gate as gate

# Repo root resolved from this file: tests/unit/test_deploy_gate.py -> repo root.
REPO_ROOT = Path(__file__).resolve().parents[2]

# A minimal valid gate config for the pure-logic evaluate_target tests. Mirrors the
# config/gate.toml schema (gate.alpha + flags + gate.secondary tolerances). Baselines are
# supplied per-test as synthetic bundles, so the baseline tables here stay empty (Wave-1
# tolerated shape).
_TEST_CFG = {
    "gate": {
        "alpha": 0.05,
        # D25-01: the unit tests exercise the non-regression floor (paired candidate-minus-
        # baseline delta) by default. The legacy absolute-vs-zero floor is exercised via the
        # floor_mode="absolute" variant in test_floor_mode_absolute_legacy.
        "floor_mode": "non_regression",
        "per_season_must_pass": True,
        "calibration_in_gate": True,
        "seasons": {"holdout": [2021, 2022, 2023, 2024]},
        "secondary": {
            "evaluation": "pooled",
            "wp_accuracy_max_drop": 0.01,
            "regression_mae_max_increase": 0.0,
            "wp_ece_max_increase": 0.0,
            "wp_brier_max_increase": 0.0,
        },
    },
    "baseline": {"wp": {}, "ats": {}, "ou": {}},
}


def _cfg_with(floor_mode: str | None = None, **secondary_overrides: float) -> dict:
    """Deep-copy _TEST_CFG, optionally overriding floor_mode + secondary tolerances.

    Used by the floor-mode and calibration-band tests so a test can flip floor_mode to
    "absolute" or set a non-zero in-memory calibration band WITHOUT mutating the shared
    _TEST_CFG (and without depending on the committed gate.toml, which stays 0.0 until
    Plan 25-02).
    """
    cfg = copy.deepcopy(_TEST_CFG)
    if floor_mode is not None:
        cfg["gate"]["floor_mode"] = floor_mode
    cfg["gate"]["secondary"].update(secondary_overrides)
    return cfg


def _sig(arr: np.ndarray) -> dict:
    """Build a real clv_significance bundle (matches per_season_clv's stored shape)."""
    return gate.clv_significance(arr)


# ---------------------------------------------------------------------------
# D24-13 column parity (ACTV-01 / Crit 1)
# ---------------------------------------------------------------------------


def test_gate_uses_diagnose_clv_columns() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_gate_uses_diagnose_clv_columns.

    The gate's CLV column map is the SAME object as diagnose.py's (identity), so the gate
    and the honest diagnosis cannot silently judge on different CLV columns (D24-13).
    """
    assert gate.CLV_COLUMN_FOR is diag.CLV_COLUMN_FOR


def test_clv_column_parity() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_clv_column_parity.

    Identity AND value: the shared map is the canonical {wp: probability_clv, ats/ou:
    line_clv} mapping (Pitfall 2 -- ATS/OU use line_clv, not the WP probability_clv).
    """
    assert gate.CLV_COLUMN_FOR is diag.CLV_COLUMN_FOR
    assert gate.CLV_COLUMN_FOR == {
        "wp": "probability_clv",
        "ats": "line_clv",
        "ou": "line_clv",
    }


# ---------------------------------------------------------------------------
# Per-season must-pass (D24-04)
# ---------------------------------------------------------------------------


def test_per_season_must_pass() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_per_season_must_pass.

    A candidate whose POOLED CLV passes the floor but whose 2023 slice is significantly
    negative must FAIL the target (per-season-must-pass bites); a candidate whose every
    season is ~zero must PASS the per-season floor.

    This exercises the LEGACY absolute-vs-zero floor (floor_mode="absolute") -- the D25-01
    non-regression floor is covered by the test_non_regression_* tests using the paired-delta
    keys. The two coexist; floor_mode selects which the gate consumes.
    """
    cfg_abs = _cfg_with(floor_mode="absolute")
    rng = np.random.default_rng(7)
    # Pooled passes: a slightly-positive (clearly not significantly-negative) pooled CLV.
    # A passing model has non-negative CLV; a slightly-positive array passes the floor
    # deterministically (the floor only fails on a SIGNIFICANTLY-NEGATIVE mean).
    pooled_pass = rng.normal(0.05, 0.2, 1120)

    # 2023 slice is significantly negative; the other seasons are slightly positive (pass).
    bad_per_season = {
        2021: _sig(rng.normal(0.05, 0.2, 280)),
        2022: _sig(rng.normal(0.05, 0.2, 280)),
        2023: _sig(rng.normal(-0.5, 0.1, 280)),  # significantly negative
        2024: _sig(rng.normal(0.05, 0.2, 280)),
    }
    candidate_bad = {
        "clv_values": pooled_pass,
        "mean": float(np.mean(pooled_pass)),
        "t": 0.0,
        "p": 1.0,
        "per_season": bad_per_season,
        "mae": 10.0,
    }
    baseline = {"mae": 10.0}
    result_bad = gate.evaluate_target("ats", candidate_bad, baseline, cfg_abs)
    assert result_bad["passed"] is False
    assert any("2023" in r for r in result_bad["reasons"]), result_bad["reasons"]

    # All seasons slightly positive -> per-season floor passes; pooled passes; MAE equal -> pass.
    good_per_season = {
        s: _sig(rng.normal(0.05, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
    }
    candidate_good = {
        "clv_values": pooled_pass,
        "mean": float(np.mean(pooled_pass)),
        "t": 0.0,
        "p": 1.0,
        "per_season": good_per_season,
        "mae": 10.0,
    }
    result_good = gate.evaluate_target("ats", candidate_good, baseline, cfg_abs)
    assert result_good["passed"] is True, result_good["reasons"]


def test_empty_per_season_fails_closed() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_empty_per_season_fails_closed.

    WR-04: when per_season_must_pass is on but the candidate supplies NO per-season slices
    (empty or absent map), evaluate_target must FAIL closed -- it must NOT report a hollow
    "all holdout seasons passed". A pooled-passing candidate with an empty per_season is the
    minimal trigger. (Legacy absolute-mode fixture; the non_regression equivalent is
    test_non_regression_empty_per_season_fails_closed.)
    """
    cfg_abs = _cfg_with(floor_mode="absolute")
    rng = np.random.default_rng(13)
    pooled_pass = rng.normal(0.05, 0.2, 1120)
    candidate_empty = {
        "clv_values": pooled_pass,
        "mean": float(np.mean(pooled_pass)),
        "t": 0.0,
        "p": 1.0,
        "per_season": {},  # no per-season evidence -> must fail closed
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate_empty, {"mae": 10.0}, cfg_abs)
    assert result["passed"] is False, result["reasons"]
    assert any("no per-season" in r.lower() for r in result["reasons"]), result[
        "reasons"
    ]
    assert not any("all holdout seasons" in r.lower() for r in result["reasons"]), (
        f"empty per_season must not claim an all-seasons PASS: {result['reasons']}"
    )

    # An absent per_season key (not just empty) is treated the same way.
    candidate_absent = {
        "clv_values": pooled_pass,
        "mean": float(np.mean(pooled_pass)),
        "t": 0.0,
        "p": 1.0,
        "mae": 10.0,
    }
    result_absent = gate.evaluate_target(
        "ats", candidate_absent, {"mae": 10.0}, cfg_abs
    )
    assert result_absent["passed"] is False, result_absent["reasons"]


# ---------------------------------------------------------------------------
# WP calibration-in-gate (D24-05)
# ---------------------------------------------------------------------------


def test_wp_calibration_in_gate() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_wp_calibration_in_gate.

    A WP candidate that passes the CLV floor but whose ECE exceeds baseline by more than the
    tolerance must FAIL; an equal-ECE/Brier candidate passes. (Legacy absolute-mode floor; the
    D25-02 noise-band behavior is covered by the test_calibration_band_* tests.)
    """
    cfg_abs = _cfg_with(floor_mode="absolute")
    rng = np.random.default_rng(11)
    pooled_pass = rng.normal(0.05, 0.2, 1120)
    good_per_season = {
        s: _sig(rng.normal(0.05, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
    }
    baseline = {"accuracy": 0.667, "ece": 0.02, "brier_score": 0.21}

    def _wp_candidate(ece: float, brier: float, accuracy: float = 0.667) -> dict:
        return {
            "clv_values": pooled_pass,
            "mean": float(np.mean(pooled_pass)),
            "t": 0.0,
            "p": 1.0,
            "per_season": good_per_season,
            "accuracy": accuracy,
            "ece": ece,
            "brier_score": brier,
        }

    # ECE regressed by 0.05 (> wp_ece_max_increase of 0.0) -> FAIL despite a passing floor.
    bad = gate.evaluate_target(
        "wp", _wp_candidate(ece=0.07, brier=0.21), baseline, cfg_abs
    )
    assert bad["passed"] is False
    assert any("ece" in r.lower() for r in bad["reasons"]), bad["reasons"]

    # Equal ECE/Brier/accuracy -> PASS.
    good = gate.evaluate_target(
        "wp", _wp_candidate(ece=0.02, brier=0.21), baseline, cfg_abs
    )
    assert good["passed"] is True, good["reasons"]


# ---------------------------------------------------------------------------
# Calibration noise band honored (D25-02) -- Plan 25-01
# ---------------------------------------------------------------------------


def _wp_calibration_candidate(
    ece: float, brier: float, *, accuracy: float = 0.667
) -> dict:
    """A WP candidate bundle that passes the CLV floor, parameterized on ECE/Brier.

    Uses an exact-zero-mean delta and an exact-zero-mean per-season delta so the non_regression
    CLV floor PASSES deterministically -- isolating the calibration gate as the only verdict
    driver. The absolute candidate CLV may be anything; here it is ~0.
    """
    rng = np.random.default_rng(31)
    cand_clv = rng.normal(0.0, 0.2, 1120)
    base_clv = rng.normal(0.0, 0.2, 1120)
    delta = _demeaned(cand_clv - base_clv)
    per_season_delta = {
        s: _demeaned(rng.normal(0.0, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
    }
    return {
        "clv_values": cand_clv,
        "baseline_clv_values": base_clv,
        "clv_delta_values": delta,
        "per_season_clv_delta_values": per_season_delta,
        "per_season": {s: _sig(per_season_delta[s]) for s in per_season_delta},
        "accuracy": accuracy,
        "ece": ece,
        "brier_score": brier,
    }


def test_calibration_band_sub_noise_passes_over_band_fails() -> None:
    """25-01 (D25-02): the WP calibration gate honors a config-driven non-zero noise band.

    An ECE increase WITHIN the band PASSES (where the legacy 0.0 tolerance would have failed);
    an ECE increase clearly OVER the band FAILS. The band is parameterized to a small non-zero
    value IN THIS in-memory config -- it does NOT depend on the committed gate.toml (which stays
    0.0 until Plan 25-02 lands the bootstrap-justified value). The non_regression CLV floor is
    isolated to pass so the calibration verdict is the only driver.
    """
    band = 0.005  # a small non-zero noise band, set only in-memory for this test
    cfg = _cfg_with(wp_ece_max_increase=band, wp_brier_max_increase=band)
    baseline = {"accuracy": 0.667, "ece": 0.02, "brier_score": 0.21}

    # ECE increase of +0.002 (< 0.005 band) -> PASS (would FAIL under the old 0.0 tolerance).
    sub_noise = gate.evaluate_target(
        "wp", _wp_calibration_candidate(ece=0.022, brier=0.21), baseline, cfg
    )
    assert sub_noise["passed"] is True, sub_noise["reasons"]

    # ECE increase of +0.05 (>> 0.005 band) -> FAIL (a genuine calibration regression, Pitfall 4).
    over_band = gate.evaluate_target(
        "wp", _wp_calibration_candidate(ece=0.07, brier=0.21), baseline, cfg
    )
    assert over_band["passed"] is False, over_band["reasons"]
    assert any("ece" in r.lower() for r in over_band["reasons"]), over_band["reasons"]


def test_calibration_band_read_from_config_not_hardcoded() -> None:
    """25-01 (D25-02): the same ECE increase flips verdict when the config band changes.

    Proves the tolerance is READ from gate.secondary.wp_ece_max_increase, not hardcoded -- the
    identical candidate FAILS under a 0.0 band and PASSES under a 0.01 band. A revert to a
    hardcoded 0.0-only tolerance (or a hardcoded wide band) would make one of these flip.
    """
    baseline = {"accuracy": 0.667, "ece": 0.02, "brier_score": 0.21}
    candidate = _wp_calibration_candidate(ece=0.025, brier=0.21)  # +0.005 ECE increase

    # Under a 0.0 band -> a +0.005 increase FAILS.
    zero_band = _cfg_with(wp_ece_max_increase=0.0, wp_brier_max_increase=0.0)
    fail = gate.evaluate_target("wp", candidate, baseline, zero_band)
    assert fail["passed"] is False, fail["reasons"]

    # Under a 0.01 band -> the SAME +0.005 increase PASSES (the band is config-driven).
    wide_band = _cfg_with(wp_ece_max_increase=0.01, wp_brier_max_increase=0.01)
    ok = gate.evaluate_target("wp", candidate, baseline, wide_band)
    assert ok["passed"] is True, ok["reasons"]


def test_calibration_brier_band_honored() -> None:
    """25-01 (D25-02): the Brier noise band is honored independently of ECE.

    A Brier increase within the band passes; over the band fails -- proving wp_brier_max_increase
    is read and compared (not only ECE).
    """
    band = 0.005
    baseline = {"accuracy": 0.667, "ece": 0.02, "brier_score": 0.21}

    sub = _cfg_with(wp_ece_max_increase=band, wp_brier_max_increase=band)
    within = gate.evaluate_target(
        "wp", _wp_calibration_candidate(ece=0.02, brier=0.213), baseline, sub
    )
    assert within["passed"] is True, within["reasons"]

    over = gate.evaluate_target(
        "wp", _wp_calibration_candidate(ece=0.02, brier=0.25), baseline, sub
    )
    assert over["passed"] is False, over["reasons"]
    assert any("brier" in r.lower() for r in over["reasons"]), over["reasons"]


def test_test_cfg_carries_floor_mode() -> None:
    """25-01: the in-memory _TEST_CFG includes floor_mode so the unit tests exercise the revised
    evaluate_target path without config/gate.toml (the new-mode path is exercised by default).
    """
    assert "floor_mode" in _TEST_CFG["gate"]
    assert _TEST_CFG["gate"]["floor_mode"] == "non_regression"


# ---------------------------------------------------------------------------
# Near-zero non-degenerate floor (review concern #9)
# ---------------------------------------------------------------------------


def test_clv_floor_near_zero_noise() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_clv_floor_near_zero_noise.

    A tiny-gaussian-noise CLV centered at 0 (mean approx 0, large p) PASSES the floor on a
    real, non-zero-variance t-test -- explicitly NOT relying on the np.zeros NaN path.
    """
    arr = np.random.default_rng(0).normal(0.0, 1e-6, 200)
    assert gate.clv_floor_passes(arr) is True


# ---------------------------------------------------------------------------
# Config season-key int normalization + holdout assertion (review concern #4)
# ---------------------------------------------------------------------------


def test_load_gate_config_season_keys_int() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_load_gate_config_season_keys_int.

    load_gate_config returns baseline season keys as int (never the str keys tomllib
    produces) for any populated baseline; validate_gate_config rejects a baseline whose
    populated season key set is not the 2021-2024 holdout.
    """
    cfg = gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    # The committed config/gate.toml baselines are now POPULATED (IN-04), so this exercises real
    # season keys: it proves load_gate_config int-normalizes the str keys tomllib produces.
    for target in ("wp", "ats", "ou"):
        season = cfg["baseline"][target].get("season", {})
        assert all(isinstance(k, int) for k in season), (
            f"{target}: baseline season keys must be int, got {list(season)}"
        )

    # A populated baseline whose season keys are the WRONG set (missing 2024) must raise.
    wrong = {
        "gate": _TEST_CFG["gate"],
        "baseline": {
            "wp": {"season": {2021: {}, 2022: {}, 2023: {}}},
            "ats": {},
            "ou": {},
        },
    }
    with pytest.raises(ValueError, match="holdout"):
        gate.validate_gate_config(wrong)


def test_validate_gate_config_tolerates_empty_baseline() -> None:
    """24-VALIDATION: validate_gate_config accepts the committed config and a tolerated shape.

    The committed gate.toml (now populated baselines) validates; a present-but-empty baseline
    table is also tolerated (the Wave-1 ordering contract); an empty dict raises (IN-04).
    """
    cfg = gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    gate.validate_gate_config(cfg)  # the committed (populated) config must validate

    # A present-but-empty baseline table is still tolerated (Wave-1 ordering contract).
    tolerated = {"gate": _TEST_CFG["gate"], "baseline": {"wp": {}, "ats": {}, "ou": {}}}
    gate.validate_gate_config(tolerated)  # must not raise

    with pytest.raises(ValueError):
        gate.validate_gate_config({})


# ---------------------------------------------------------------------------
# floor_mode + calibration-band config validation (D25-01 / D25-02) -- Plan 25-01
# ---------------------------------------------------------------------------


def test_committed_config_has_floor_mode_non_regression() -> None:
    """25-01: the committed config/gate.toml carries floor_mode=non_regression and validates."""
    cfg = gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    gate.validate_gate_config(cfg)  # must not raise
    assert cfg["gate"]["floor_mode"] == "non_regression"


def test_validate_gate_config_requires_floor_mode() -> None:
    """25-01: a config missing gate.floor_mode raises ValueError (V5 input validation).

    T-25-01-validate: a partial config must not silently change the deploy decision -- an
    absent floor_mode is rejected BEFORE any decision.
    """
    no_floor_mode = {
        "gate": {
            "alpha": 0.05,
            # floor_mode deliberately omitted
            "per_season_must_pass": True,
            "calibration_in_gate": True,
            "seasons": {"holdout": [2021, 2022, 2023, 2024]},
            "secondary": {
                "evaluation": "pooled",
                "wp_accuracy_max_drop": 0.01,
                "regression_mae_max_increase": 0.0,
                "wp_ece_max_increase": 0.0,
                "wp_brier_max_increase": 0.0,
            },
        },
        "baseline": {"wp": {}, "ats": {}, "ou": {}},
    }
    with pytest.raises(ValueError, match="floor_mode"):
        gate.validate_gate_config(no_floor_mode)


def test_validate_gate_config_rejects_unknown_floor_mode() -> None:
    """25-01: a config whose floor_mode is neither non_regression nor absolute raises."""
    bad = _cfg_with(floor_mode="loosened")
    with pytest.raises(ValueError, match="floor_mode"):
        gate.validate_gate_config(bad)


def test_validate_gate_config_requires_calibration_band_keys() -> None:
    """25-01: a config missing the calibration noise-band keys raises ValueError.

    The wp_ece_max_increase / wp_brier_max_increase keys are required gate.secondary keys (the
    calibration band the D25-02 bootstrap fills in Plan 25-02); a config omitting them is
    rejected before any deploy decision.
    """
    missing_band = _cfg_with()
    del missing_band["gate"]["secondary"]["wp_ece_max_increase"]
    with pytest.raises(ValueError, match=r"wp_ece_max_increase|secondary"):
        gate.validate_gate_config(missing_band)


def test_committed_calibration_band_is_bootstrap_justified() -> None:
    """25-02 (D25-02): the committed gate.toml calibration band is the bootstrap-justified value.

    Plan 25-01 held wp_ece_max_increase / wp_brier_max_increase at the 0.0 placeholder; Plan 25-02
    replaces them with the bootstrap-justified noise band (paired bootstrap on the ~1087-game
    holdout: ECE delta +0.0068 with 95% CI [-0.012, +0.035]; Brier delta +0.0019 with 95% CI
    [-0.003, +0.007] -- both indistinguishable from zero). The band sits at ~1 bootstrap-std: it is
    a documented tolerance (NOT a blind loosening, Pitfall 4) -- non-zero but tight enough to reject
    a genuine 2+ std regression. The value is traceable to DIAGNOSIS-NOTES.md via a config comment.
    """
    cfg = gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    secondary = cfg["gate"]["secondary"]
    # The Plan 25-02 bootstrap-justified band (no longer the 0.0 placeholder).
    assert secondary["wp_ece_max_increase"] == 0.0125
    assert secondary["wp_brier_max_increase"] == 0.0030
    # Both bands are non-zero (the bootstrap diagnosis landed) but tight (well below a 2+ std band).
    assert 0.0 < secondary["wp_ece_max_increase"] < 0.05
    assert 0.0 < secondary["wp_brier_max_increase"] < 0.01
    # Traceability: the band must cite DIAGNOSIS-NOTES.md so the bootstrap justification is findable.
    raw = (REPO_ROOT / "config" / "gate.toml").read_text(encoding="utf-8")
    assert "DIAGNOSIS-NOTES" in raw, (
        "the calibration band must be traceable to DIAGNOSIS-NOTES.md via a comment"
    )


def test_committed_frozen_baseline_values_unchanged() -> None:
    """30-12: the frozen baseline VALUES in config/gate.toml match the Plan 30-12 re-freeze.

    History of this anchor set, kept rather than overwritten so the re-freezes are legible:

      * Plan 24-03 froze the baseline against the v1.0 artifacts.
      * Plan 25-05 (D25-11) re-froze it against the then-deployed incumbent -- WP/ATS moved to
        the activated re-fit values, OU stayed on retained v1.0 (honest refusal D25-14).
      * Plan 30-09 (SPEC R7) re-froze it AGAIN, pre-promotion. Nothing had swapped: the SAME
        three incumbents (wp_20260605_215552 / ats_20260605_220128 / ou_20260326_163930) were
        re-scored on the gold Plans 30-06..30-08 REBUILT. The rebuild alone invalidates a
        baseline, which is why that re-freeze was required even on a zero-swap outcome.
      * Plan 30-12 (SPEC R7) re-freezes it a SECOND time, post-promotion, against the END-STATE
        deployed artifacts. Plan 30-11's armed run promoted WP to ``wp_20260824_113325`` and
        REFUSED ATS and O/U on line-CLV regression, so they retain ``ats_20260605_220128`` and
        ``ou_20260326_163930``. Only the WP tables move here; the ATS and O/U tables are
        byte-identical to the Plan 30-09 block, which is the no-op demonstration SPEC R7 asks
        for on a target that did not swap. The D25-11 precedent puts this second re-freeze
        AFTER activation deliberately: re-freezing inside the plan that rendered the verdict
        would retroactively move the judge that produced it.

    The values are generator output block-pasted from ``scripts/freeze_gate_baseline.py``
    (D24-07), never hand-transcribed; this assertion is the guard against an accidental
    hand-edit of a [baseline.*] table AFTER a re-freeze. Each re-point is made by reading the
    regenerated block (``outputs/gate/baseline_refreeze_2.toml`` for this one), not by nudging
    a number until the test goes green.
    """
    cfg = gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    baseline = cfg["baseline"]
    # The Plan 30-12 re-freeze anchors: WP re-pointed at the promoted re-fit; ATS and O/U
    # unchanged from the Plan 30-09 block because the gate REFUSED both candidates.
    assert baseline["wp"]["pooled"]["mean"] == pytest.approx(-0.03800034)
    assert baseline["wp"]["pooled"]["accuracy"] == pytest.approx(0.67076383)
    assert baseline["ats"]["pooled"]["mean"] == pytest.approx(-0.00149507)
    assert baseline["ats"]["pooled"]["mae"] == pytest.approx(8.58072427)
    assert baseline["ou"]["pooled"]["mean"] == pytest.approx(1.09908061)
    assert baseline["ou"]["pooled"]["mae"] == pytest.approx(10.30191577)
    # A per-season anchor from each target to catch a season-table edit.
    assert baseline["wp"]["season"][2024]["mean"] == pytest.approx(-0.04886401)
    assert baseline["ats"]["season"][2023]["mean"] == pytest.approx(-0.63325109)
    assert baseline["ou"]["season"][2021]["mean"] == pytest.approx(-1.22226360)
    # The per-season SAMPLE SIZES did NOT move across the rebuild (the 2021-2024 holdout
    # population is unchanged; only the feature matrix behind it was rebuilt). _drift_tripwire
    # compares n EXACTLY, so pinning them here states what the rebuild did and did not touch.
    for target in ("wp", "ats", "ou"):
        for season, expected_n in ((2021, 272), (2022, 271), (2023, 272), (2024, 272)):
            assert baseline[target]["season"][season]["n"] == expected_n, (
                f"{target} {season} baseline sample size moved; the holdout population changed, "
                "which is a hard drift signal rather than a re-freeze"
            )


# ---------------------------------------------------------------------------
# The single bundle builder shape
# ---------------------------------------------------------------------------


def _tiny_scored_and_odds(target: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tiny scored frame + matching odds frame that merges cleanly (>= MIN_CLV_SAMPLE).

    Two seasons (2021, 2022), 12 games total so clv_significance can run (MIN_CLV_SAMPLE=10).
    WP: model_prob is a win probability; ATS: model_spread; OU: model_total.
    """
    n = 12
    game_ids = [f"G{i:02d}" for i in range(n)]
    seasons = [2021] * 6 + [2022] * 6
    rng = np.random.default_rng(3)

    scored = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": seasons,
            "week": list(range(1, n + 1)),
        }
    )
    odds = pd.DataFrame(
        {
            "game_id": game_ids,
            "ml_home": [-150] * n,
            "ml_away": [130] * n,
            "spread": rng.normal(-2.5, 1.0, n),
            "total": rng.normal(45.0, 2.0, n),
        }
    )

    if target == "wp":
        scored["model_prob"] = np.clip(rng.normal(0.55, 0.1, n), 0.05, 0.95)
        scored["actual"] = (rng.random(n) < 0.55).astype(int)
    elif target == "ats":
        margins = rng.normal(-2.0, 3.0, n)
        scored["model_prob"] = margins
        scored["model_spread"] = margins
        scored["actual"] = rng.normal(-2.0, 7.0, n)
    else:  # ou
        totals = rng.normal(45.0, 3.0, n)
        scored["model_prob"] = totals
        scored["model_total"] = totals
        scored["actual"] = rng.normal(45.0, 8.0, n)

    return scored, odds


def test_build_candidate_bundle_shape() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_build_candidate_bundle_shape.

    The single bundle builder yields clv_values, per_season (int season keys), pooled mean,
    and accuracy (WP) / mae (ATS/OU) -- proving one bundle shape for promote/retrain/tests.
    """
    # WP bundle.
    scored_wp, odds_wp = _tiny_scored_and_odds("wp")
    wp_bundle = gate.build_candidate_bundle("wp", scored_wp, odds_wp, _TEST_CFG)
    assert "clv_values" in wp_bundle
    assert "mean" in wp_bundle
    assert "per_season" in wp_bundle
    assert all(isinstance(k, int) for k in wp_bundle["per_season"])
    assert "accuracy" in wp_bundle
    assert "mae" not in wp_bundle

    # ATS / OU bundles carry mae, not accuracy.
    for target in ("ats", "ou"):
        scored, odds = _tiny_scored_and_odds(target)
        bundle = gate.build_candidate_bundle(target, scored, odds, _TEST_CFG)
        assert "clv_values" in bundle
        assert "per_season" in bundle
        assert all(isinstance(k, int) for k in bundle["per_season"])
        assert "mae" in bundle
        assert "accuracy" not in bundle


# ---------------------------------------------------------------------------
# Forced-FAIL floor + evaluate_target
# ---------------------------------------------------------------------------


def test_forced_fail_floor_and_evaluate() -> None:
    """24-VALIDATION: forced-FAIL gate logic (synthetic significantly-negative pooled CLV).

    A significantly-negative pooled CLV array makes clv_floor_passes False and
    evaluate_target passed False (the hard-block logic, with zero training).
    (Legacy absolute-mode floor; the non_regression forced-fail is
    test_non_regression_pooled_candidate_significantly_worse_fails.)
    """
    cfg_abs = _cfg_with(floor_mode="absolute")
    sig_neg = np.full(200, -0.5)
    assert gate.clv_floor_passes(sig_neg) is False

    rng = np.random.default_rng(5)
    candidate = {
        "clv_values": sig_neg,
        "mean": -0.5,
        "t": -999.0,
        "p": 0.0,
        "per_season": {
            s: _sig(rng.normal(0.05, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
        },
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, cfg_abs)
    assert result["passed"] is False
    assert any("CLV" in r for r in result["reasons"]), result["reasons"]


# ---------------------------------------------------------------------------
# Non-regression paired-delta floor (D25-01 / D25-15) -- Plan 25-01
# ---------------------------------------------------------------------------


def _demeaned(arr: np.ndarray) -> np.ndarray:
    """Return arr shifted to an exact sample mean of 0.0.

    A "candidate ~= baseline" paired delta should be DETERMINISTICALLY non-significant. Drawing
    rng.normal(0.0, sigma, n) leaves a small random sample mean that, at n>=270 with a small SE,
    can cross the alpha=0.05 negative-tail bar by chance. Subtracting the sample mean pins the
    delta at exactly mean 0 so the non-regression floor reliably PASSES (t ~ 0, p ~ 1).
    """
    return arr - float(np.mean(arr))


def test_clv_non_regression_passes_helper() -> None:
    """25-01: the paired-delta helper fails ONLY when the delta is significantly NEGATIVE.

    clv_non_regression_passes runs the SHARED clv_significance on the candidate-minus-
    baseline per-game delta and returns "fail only if significantly worse" (mean < 0 AND
    p < alpha). A near-zero or positive delta passes regardless of the candidate's absolute
    CLV sign; a significantly-negative delta fails; an untestable (too-small) delta is a
    strict FAIL matching clv_floor_passes.
    """
    rng = np.random.default_rng(101)

    # Delta mean exactly 0 (candidate ~= baseline) -> PASS even if absolute CLV is negative.
    near_zero_delta = _demeaned(rng.normal(0.0, 0.2, 600))
    assert gate.clv_non_regression_passes(near_zero_delta) is True

    # Delta clearly positive (candidate BETTER than baseline) -> PASS.
    positive_delta = rng.normal(0.10, 0.2, 600)
    assert gate.clv_non_regression_passes(positive_delta) is True

    # Delta significantly negative (candidate WORSE than baseline) -> FAIL.
    worse_delta = np.full(200, -0.5)
    assert gate.clv_non_regression_passes(worse_delta) is False

    # Untestable (n < MIN_CLV_SAMPLE) -> strict FAIL (matches clv_floor_passes convention).
    tiny_delta = np.array([0.1, 0.2, -0.1])
    assert gate.clv_non_regression_passes(tiny_delta) is False


def test_non_regression_pooled_candidate_not_worse_passes() -> None:
    """25-01: under floor_mode=non_regression a candidate that is NOT worse than the baseline
    PASSES the pooled floor even when its absolute CLV is negative-vs-zero.

    The candidate's raw CLV is significantly negative (would FAIL the legacy absolute floor),
    but the paired delta vs the baseline is ~zero (candidate ~= baseline), so the
    non-regression floor PASSES -- the WP -0.0567-baseline / -0.0443-candidate situation.
    """
    rng = np.random.default_rng(202)
    # Raw candidate CLV: significantly negative-vs-zero.
    cand_clv = rng.normal(-0.05, 0.2, 1120)
    base_clv = rng.normal(-0.05, 0.2, 1120)
    delta = _demeaned(cand_clv - base_clv)  # exactly mean 0 -> not significantly worse
    per_season_delta = {
        s: _demeaned(rng.normal(0.0, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
    }
    candidate = {
        "clv_values": cand_clv,
        "baseline_clv_values": base_clv,
        "clv_delta_values": delta,
        "mean": float(np.mean(cand_clv)),
        "t": -3.0,
        "p": 0.001,
        "per_season_clv_delta_values": per_season_delta,
        "per_season": {s: _sig(per_season_delta[s]) for s in per_season_delta},
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, _TEST_CFG)
    assert result["passed"] is True, result["reasons"]


def test_non_regression_pooled_candidate_significantly_worse_fails() -> None:
    """25-01: a candidate whose pooled paired delta is significantly NEGATIVE FAILS the floor."""
    rng = np.random.default_rng(303)
    cand_clv = rng.normal(-0.5, 0.2, 1120)
    base_clv = rng.normal(0.0, 0.2, 1120)
    delta = cand_clv - base_clv  # strongly negative -> significantly worse
    good_per_season = {
        s: _demeaned(rng.normal(0.0, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
    }
    candidate = {
        "clv_values": cand_clv,
        "baseline_clv_values": base_clv,
        "clv_delta_values": delta,
        "mean": float(np.mean(cand_clv)),
        "t": -50.0,
        "p": 0.0,
        "per_season_clv_delta_values": good_per_season,
        "per_season": {s: _sig(good_per_season[s]) for s in good_per_season},
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, _TEST_CFG)
    assert result["passed"] is False, result["reasons"]
    assert any("CLV" in r for r in result["reasons"]), result["reasons"]


def test_non_regression_per_season_significantly_worse_fails() -> None:
    """25-01: a candidate whose POOLED delta passes but whose 2023 season delta is
    significantly negative FAILS under per_season_must_pass.
    """
    rng = np.random.default_rng(404)
    cand_clv = rng.normal(0.0, 0.2, 1120)
    base_clv = rng.normal(0.0, 0.2, 1120)
    delta = _demeaned(cand_clv - base_clv)  # pooled exactly 0 -> pooled passes
    per_season_delta = {
        2021: _demeaned(rng.normal(0.0, 0.2, 280)),
        2022: _demeaned(rng.normal(0.0, 0.2, 280)),
        2023: rng.normal(-0.5, 0.1, 280),  # significantly worse this season
        2024: _demeaned(rng.normal(0.0, 0.2, 280)),
    }
    candidate = {
        "clv_values": cand_clv,
        "baseline_clv_values": base_clv,
        "clv_delta_values": delta,
        "mean": float(np.mean(cand_clv)),
        "t": 0.0,
        "p": 1.0,
        "per_season_clv_delta_values": per_season_delta,
        "per_season": {s: _sig(per_season_delta[s]) for s in per_season_delta},
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, _TEST_CFG)
    assert result["passed"] is False, result["reasons"]
    assert any("2023" in r for r in result["reasons"]), result["reasons"]

    # Every season delta ~zero -> per-season floor passes.
    good_per_season = {
        s: _demeaned(rng.normal(0.0, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
    }
    candidate_good = {
        **candidate,
        "per_season_clv_delta_values": good_per_season,
        "per_season": {s: _sig(good_per_season[s]) for s in good_per_season},
    }
    result_good = gate.evaluate_target("ats", candidate_good, {"mae": 10.0}, _TEST_CFG)
    assert result_good["passed"] is True, result_good["reasons"]


def test_non_regression_absolute_verdict_preserved() -> None:
    """25-01: REGARDLESS of mode, evaluate_target attaches the absolute-vs-zero candidate CLV
    significance (mean/t/p on the RAW clv_values, not the delta) for the bettable-bar readout.
    """
    rng = np.random.default_rng(505)
    cand_clv = rng.normal(-0.05, 0.2, 1120)  # significantly negative-vs-zero
    base_clv = rng.normal(-0.05, 0.2, 1120)
    delta = _demeaned(cand_clv - base_clv)
    per_season_delta = {
        s: _demeaned(rng.normal(0.0, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
    }
    candidate = {
        "clv_values": cand_clv,
        "baseline_clv_values": base_clv,
        "clv_delta_values": delta,
        "per_season_clv_delta_values": per_season_delta,
        "per_season": {s: _sig(per_season_delta[s]) for s in per_season_delta},
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, _TEST_CFG)
    absolute = result["absolute_verdict"]
    # The absolute verdict is computed on the RAW candidate CLV (not the delta).
    expected = gate.clv_significance(cand_clv)
    assert absolute["mean"] == pytest.approx(expected["mean"])
    assert absolute["t"] == pytest.approx(expected["t"])
    assert absolute["p"] == pytest.approx(expected["p"])
    # The raw candidate CLV is significantly negative-vs-zero -> the bettable bar is NOT met,
    # even though the deploy (non-regression) decision passed.
    assert absolute["mean"] < 0
    assert absolute["p"] < 0.05
    # Per-season absolute verdict is also available for the readout.
    assert set(result["absolute_per_season"]) == {2021, 2022, 2023, 2024}


def test_floor_mode_absolute_legacy() -> None:
    """25-01: floor_mode=absolute restores the legacy absolute-vs-zero floor (clv_floor_passes).

    Under floor_mode=absolute the pooled + per-season floor run on the RAW candidate CLV via
    the legacy clv_floor_passes -- a significantly-negative-vs-zero candidate FAILS even if it
    is not worse than the baseline.
    """
    cfg_abs = _cfg_with(floor_mode="absolute")
    rng = np.random.default_rng(606)
    cand_clv = np.full(1120, -0.5)  # significantly negative-vs-zero
    candidate = {
        "clv_values": cand_clv,
        "mean": -0.5,
        "t": -999.0,
        "p": 0.0,
        "per_season": {
            s: _sig(rng.normal(0.05, 0.2, 280)) for s in (2021, 2022, 2023, 2024)
        },
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, cfg_abs)
    assert result["passed"] is False, result["reasons"]


def test_bundle_delta_keys_pinned_in_builder() -> None:
    """25-01: build_candidate_bundle PINS the delta key contract (Codex HIGH).

    This plan DEFINES the keys (clv_values + baseline_clv_values + clv_delta_values, and the
    per-season equivalents) -- Plan 25-02 populates baseline/delta via the real merge-on-game_id
    pairing. The builder ships clv_values populated and the baseline/delta keys present-but-None
    so a shape/key mismatch when 25-02 wires the pairing fails loudly (the keys are pinned now).
    """
    scored, odds = _tiny_scored_and_odds("ats")
    bundle = gate.build_candidate_bundle("ats", scored, odds, _TEST_CFG)

    # The pinned pooled delta keys are PRESENT (clv_values populated; baseline/delta = None here).
    for key in ("clv_values", "baseline_clv_values", "clv_delta_values"):
        assert key in bundle, f"missing pinned bundle key: {key}"
    assert bundle["clv_values"] is not None
    assert bundle["baseline_clv_values"] is None  # Plan 25-02 populates
    assert bundle["clv_delta_values"] is None  # Plan 25-02 populates

    # The per-season delta keys are present, keyed by int season; per_season_clv_values is the
    # populated raw-candidate per-season array (the absolute per-season input).
    # The per-season delta keys cover the full 2021-2024 holdout (mirrors per_season, which
    # per_season_clv populates for every holdout season -- empty seasons get n==0 slices).
    for key in (
        "per_season_clv_values",
        "per_season_baseline_clv_values",
        "per_season_clv_delta_values",
    ):
        assert key in bundle, f"missing pinned per-season bundle key: {key}"
        assert set(bundle[key]) == {2021, 2022, 2023, 2024}, key
        assert all(isinstance(k, int) for k in bundle[key]), key
    # per_season_clv_values carries raw candidate arrays (the absolute per-season input);
    # baseline/delta per-season are None placeholders Plan 25-02 populates.
    assert all(v is not None for v in bundle["per_season_clv_values"].values())
    assert all(v is None for v in bundle["per_season_baseline_clv_values"].values())
    assert all(v is None for v in bundle["per_season_clv_delta_values"].values())


def test_bundle_delta_keys_internal_consistency() -> None:
    """25-01: the internal-consistency invariant Plan 25-02 must keep --
    clv_delta_values == clv_values - baseline_clv_values element-wise.

    This plan does not populate the paired arrays (25-02 does), so the invariant is asserted on
    a SYNTHETIC bundle here -- proving the contract a consumer (and Plan 25-02) must honor when
    the real pairing is wired.
    """
    rng = np.random.default_rng(909)
    clv_values = rng.normal(-0.05, 0.2, 1120)
    baseline_clv_values = rng.normal(-0.05, 0.2, 1120)
    clv_delta_values = clv_values - baseline_clv_values
    np.testing.assert_allclose(clv_delta_values, clv_values - baseline_clv_values)


def test_non_regression_untestable_delta_fails_closed() -> None:
    """25-01: a pooled delta too small to t-test (t is None) is a strict FAIL (fail-closed)."""
    candidate = {
        "clv_values": np.array([0.1, 0.2, -0.1]),
        "baseline_clv_values": np.array([0.0, 0.0, 0.0]),
        "clv_delta_values": np.array([0.1, 0.2, -0.1]),  # n=3 < MIN_CLV_SAMPLE
        "per_season_clv_delta_values": {
            s: np.full(280, 0.0) for s in (2021, 2022, 2023, 2024)
        },
        "per_season": {s: _sig(np.full(280, 0.0)) for s in (2021, 2022, 2023, 2024)},
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, _TEST_CFG)
    assert result["passed"] is False, result["reasons"]


def test_non_regression_empty_per_season_fails_closed() -> None:
    """25-01: the WR-04 fail-closed behavior is preserved under non_regression -- an absent
    per-season delta map must NOT be reported as an all-seasons PASS.
    """
    rng = np.random.default_rng(707)
    cand_clv = rng.normal(0.0, 0.2, 1120)
    base_clv = rng.normal(0.0, 0.2, 1120)
    candidate = {
        "clv_values": cand_clv,
        "baseline_clv_values": base_clv,
        "clv_delta_values": cand_clv - base_clv,
        "per_season_clv_delta_values": {},  # no evidence -> fail closed
        "per_season": {},
        "mae": 10.0,
    }
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, _TEST_CFG)
    assert result["passed"] is False, result["reasons"]
    assert any("no per-season" in r.lower() for r in result["reasons"]), result[
        "reasons"
    ]


# ---------------------------------------------------------------------------
# No-autoswap (ACTV-03 / Crit 3 -- depends on Plan 24-01 update_latest=False default)
# ---------------------------------------------------------------------------


def test_save_does_not_autoswap(tmp_path: Path) -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_save_does_not_autoswap.

    save_model_artifact with default args (update_latest=False, Plan 24-01) writes the
    versioned artifact dir but NEVER touches latest.json -- a bare train never auto-swaps
    production.
    """
    from models.artifacts import save_model_artifact

    save_model_artifact(
        model={"trivial": "picklable"},
        target="wp",
        metadata={"note": "unit test"},
        feature_list=["f1", "f2"],
        artifacts_dir=tmp_path,
    )
    assert not (tmp_path / "latest.json").exists()


# ---------------------------------------------------------------------------
# Committed config (ACTV-02 -- thresholds + baseline frozen and git-tracked)
# ---------------------------------------------------------------------------


def test_gate_config_committed() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_gate_config_committed.

    config/gate.toml exists at the repo root AND is tracked by git (in ``git ls-files``).
    The frozen judge must be in version control so a threshold change is a reviewable diff
    (mirrors tests/unit/test_pipeline_md.py + the tests/test_runner.py subprocess idiom).
    """
    config_path = REPO_ROOT / "config" / "gate.toml"
    assert config_path.is_file(), f"missing: {config_path}"

    tracked = subprocess.run(
        ["git", "ls-files", "config/gate.toml"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "config/gate.toml" in tracked.stdout, (
        "config/gate.toml must be git-tracked (the frozen gate config is committed, not "
        f"gitignored); git ls-files output: {tracked.stdout!r}"
    )


# ---------------------------------------------------------------------------
# CR-01 regression guards: ATS/OU MAE reads the explicit line column (not model_prob)
# ---------------------------------------------------------------------------


def _tiny_scored_and_odds_distinct_line(
    target: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a tiny scored frame where model_prob and the explicit line column DIFFER.

    This is the discriminating fixture for the CR-01 guard.  Every game gets a complete
    odds row (ml_home/ml_away/spread/total present) so all 12 rows survive the
    has_closing_odds filter.  model_prob is set to model_spread/model_total + 50.0 (a
    large offset) so the two candidate MAEs are unmistakably distinct: a revert to reading
    model_prob would produce a MAE ~50 larger than the correct model_spread/model_total MAE.
    """
    n = 12
    game_ids = [f"H{i:02d}" for i in range(n)]
    seasons = [2021] * 6 + [2022] * 6
    rng = np.random.default_rng(42)

    odds = pd.DataFrame(
        {
            "game_id": game_ids,
            "ml_home": [-120] * n,
            "ml_away": [100] * n,
            "spread": rng.normal(-3.0, 1.0, n),
            "total": rng.normal(46.0, 2.0, n),
        }
    )

    scored = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": seasons,
            "week": list(range(1, n + 1)),
        }
    )

    if target == "ats":
        line_values = rng.normal(-2.5, 3.0, n)
        scored["model_spread"] = line_values
        scored["model_prob"] = line_values + 50.0  # large offset -- clearly different
        scored["actual"] = rng.normal(-2.5, 7.0, n)
    else:  # ou
        line_values = rng.normal(44.0, 3.0, n)
        scored["model_total"] = line_values
        scored["model_prob"] = line_values + 50.0  # large offset -- clearly different
        scored["actual"] = rng.normal(44.0, 8.0, n)

    return scored, odds


def test_build_candidate_bundle_mae_uses_explicit_line_column() -> None:
    """24-VALIDATION: CR-01 regression guard -- ATS/OU MAE reads model_spread/model_total.

    Proves build_candidate_bundle computes the ATS regression MAE from model_spread (and
    OU MAE from model_total), NEVER from the overloaded model_prob alias (CR-01, commit
    934db5d).  The fixture sets model_prob = line_col + 50.0 so the two candidate MAEs
    are unmistakably distinct: if the code reverted to reading model_prob the computed MAE
    would be ~50 larger than the expected value, causing both assertions to fail.
    """
    for target, line_col_name in (("ats", "model_spread"), ("ou", "model_total")):
        scored, odds = _tiny_scored_and_odds_distinct_line(target)

        bundle = gate.build_candidate_bundle(target, scored, odds, _TEST_CFG)

        # Expected MAE: mean(|actual - line_col|) over the full n=12 rows (all have
        # complete odds, so has_closing_odds keeps all rows).
        expected_mae_from_line = float(
            np.mean(
                np.abs(scored["actual"].to_numpy() - scored[line_col_name].to_numpy())
            )
        )
        expected_mae_from_prob = float(
            np.mean(
                np.abs(scored["actual"].to_numpy() - scored["model_prob"].to_numpy())
            )
        )

        # The offset is 50.0, so the two expected MAEs must themselves differ significantly.
        assert abs(expected_mae_from_line - expected_mae_from_prob) > 10.0, (
            f"{target}: fixture offset too small -- the two MAEs are not distinguishable"
        )

        assert bundle["mae"] == pytest.approx(expected_mae_from_line, rel=1e-6), (
            f"{target}: bundle MAE {bundle['mae']:.6f} != expected line-col MAE "
            f"{expected_mae_from_line:.6f}; a revert to model_prob would give "
            f"{expected_mae_from_prob:.6f}"
        )
        assert bundle["mae"] != pytest.approx(expected_mae_from_prob, abs=1.0), (
            f"{target}: bundle MAE should NOT equal the model_prob MAE "
            f"({expected_mae_from_prob:.6f}) -- CR-01 fix is not in effect"
        )


def test_build_candidate_bundle_missing_line_column_raises() -> None:
    """24-VALIDATION: CR-01 guard -- ValueError when the explicit line column is absent.

    build_candidate_bundle must raise ValueError naming the missing column when:
      - ATS scored frame has model_prob but is missing model_spread
      - OU  scored frame has model_prob but is missing model_total
    game_id/season/actual are present and odds are complete so the only trigger is the
    absent line column (CR-01, deploy_gate.py:272-278).
    """
    n = 12
    game_ids = [f"M{i:02d}" for i in range(n)]
    seasons = [2021] * 6 + [2022] * 6
    rng = np.random.default_rng(7)

    odds = pd.DataFrame(
        {
            "game_id": game_ids,
            "ml_home": [-110] * n,
            "ml_away": [-110] * n,
            "spread": rng.normal(-2.5, 1.0, n),
            "total": rng.normal(45.0, 2.0, n),
        }
    )

    # ATS: has model_prob, missing model_spread -> ValueError naming "model_spread".
    ats_scored = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": seasons,
            "week": list(range(1, n + 1)),
            "model_prob": rng.normal(-2.5, 3.0, n),
            "actual": rng.normal(-2.5, 7.0, n),
        }
    )
    with pytest.raises(ValueError, match="model_spread"):
        gate.build_candidate_bundle("ats", ats_scored, odds, _TEST_CFG)

    # OU: has model_prob, missing model_total -> ValueError naming "model_total".
    ou_scored = pd.DataFrame(
        {
            "game_id": game_ids,
            "season": seasons,
            "week": list(range(1, n + 1)),
            "model_prob": rng.normal(44.0, 3.0, n),
            "actual": rng.normal(44.0, 8.0, n),
        }
    )
    with pytest.raises(ValueError, match="model_total"):
        gate.build_candidate_bundle("ou", ou_scored, odds, _TEST_CFG)


# ---------------------------------------------------------------------------
# WR-02 atomicity guard: update_manifest leaves latest.json intact on mid-rename failure
# ---------------------------------------------------------------------------


def test_update_manifest_atomic_write_preserves_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """24-VALIDATION: WR-02 atomicity guard -- a simulated mid-rename crash is safe.

    Proves _atomic_write_json's except-branch cleanup (deploy_gate.py commit 5749a19):
      - The production latest.json is BYTE-UNCHANGED after a Path.replace failure
        (the temp file was written but the atomic rename never landed).
      - NO stray temp file remains (the except branch calls unlink before re-raising).
    If the implementation dropped the except cleanup, the temp file would survive.
    If it wrote directly (non-atomically), the manifest could be truncated or absent.
    """
    import json

    from models.artifacts import update_manifest

    # Write a known production manifest.
    original_data = {"wp": "wp_old", "blend": {"ats": 0.5}}
    manifest_path = tmp_path / "latest.json"
    manifest_path.write_text(json.dumps(original_data, indent=2))
    original_bytes = manifest_path.read_bytes()

    # Monkeypatch Path.replace to simulate a crash mid-rename.  The temp file has been
    # flushed and closed by this point; Path.replace is what would atomically land the
    # new content over latest.json.  By raising here we confirm the except cleanup runs.
    def _boom(*args: object, **kwargs: object) -> None:
        raise OSError("simulated crash mid-rename")

    monkeypatch.setattr(Path, "replace", _boom)

    # update_manifest must propagate the OSError (not swallow it).
    with pytest.raises(OSError, match="simulated crash mid-rename"):
        update_manifest("wp", "wp_new", tmp_path)

    # The production manifest must be byte-unchanged.
    assert manifest_path.read_bytes() == original_bytes, (
        "latest.json was modified despite the rename failing -- atomic write is not safe"
    )

    # No stray temp file may remain (the except branch must have unlinked it).
    remaining = sorted(p.name for p in tmp_path.iterdir())
    assert remaining == ["latest.json"], (
        f"stray temp file(s) left after failed rename: {remaining}"
    )


# ---------------------------------------------------------------------------
# Plan 30-09 / SPEC R5: the phase-start threshold snapshot (T-30-39)
# ---------------------------------------------------------------------------


# The PHASE-START snapshot of every NON-BASELINE setting in config/gate.toml, read at the
# start of Phase 30 (Plan 30-09) and asserted unchanged for the rest of the phase.
#
# Why it exists: 30-SPEC's prohibition "MUST NOT loosen any config/gate.toml threshold, band,
# or frozen value to make a target pass" (R5) is assigned to the TEST tier, not the judgment
# tier. This snapshot IS that test. Phase 30 re-fits three targets against a frozen judge; the
# cheapest way to rescue a failing target is to widen a band by a hair, and that would be a
# silent redefinition of what "passing" means. Here it is a RED.
#
# The [baseline.*] block is DELIBERATELY EXCLUDED. It is regenerated TWICE in this phase by
# design -- Plan 30-09 re-freezes it against the still-deployed incumbents on the rebuilt gold
# (the rebuild alone invalidates it), and Plan 30-12 re-freezes it again against the end-state
# incumbents. Snapshotting it here would make this test fail for the wrong reason: a sanctioned
# generator block-paste would look identical to the prohibited hand-edit. The baseline has its
# own guards -- test_committed_frozen_baseline_values_unchanged (above) pins the current values,
# and tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore_all_fields
# re-verifies all 68 frozen fields against a fresh generator re-score.
#
# If a future phase deliberately changes one of these settings, RE-READ the snapshot from the
# new committed config in the same commit as the config edit, with the rationale -- do not
# delete the test.
PHASE_START_GATE_SETTINGS: dict[str, object] = {
    "alpha": 0.05,
    "per_season_must_pass": True,
    "floor_mode": "non_regression",
    "calibration_in_gate": True,
    "seasons.holdout": [2021, 2022, 2023, 2024],
    "secondary.evaluation": "pooled",
    "secondary.wp_accuracy_max_drop": 0.01,
    "secondary.regression_mae_max_increase": 0.0,
    "secondary.wp_ece_max_increase": 0.0125,
    "secondary.wp_brier_max_increase": 0.0030,
}

# The two declaration sites of the gate-time freshness tolerance. There are exactly TWO today
# and SPEC R5 says no new tolerance is introduced in this phase; test_freshness_tolerance_parity
# below IMPORTS both values and asserts they agree, rather than trusting the comment at
# scripts/promote_models.py:78-85 that names the places which must agree.
_FRESHNESS_TOL_SITES: tuple[str, ...] = (
    "scripts/promote_models.py",
    "tests/integration/test_promote_models.py",
)

# The phase-start READING of that tolerance, recorded so a WIDENED tolerance is caught as well
# as a DIVERGENT one. This is a historical reading for change-detection, NOT a third usable
# declaration: no comparison anywhere -- in production, in the gate, or in these tests -- reads
# a tolerance from here. The two live declarations above remain two.
_PHASE_START_FRESHNESS_TOL = 5e-3

# Matches a module-level declaration of the tolerance, so the "no third copy" scan below finds
# real declarations rather than the many places that merely mention the name.
_FRESHNESS_TOL_DECL = re.compile(r"^_FRESHNESS_TOL\s*=", re.MULTILINE)


def assert_phase_start_thresholds_unchanged(cfg: dict) -> None:
    """Assert every non-baseline gate setting still equals its phase-start value.

    Diagnostic by design: a failure names the setting, prints the snapshotted value beside the
    current one, and states the remediation -- because the tempting response to a failing target
    is to move the bar, and the message has to close that door at the moment it is opened.

    Args:
        cfg: A loaded gate config (``models.deploy_gate.load_gate_config`` shape).

    Raises:
        AssertionError: If any snapshotted setting has moved, or is missing from the config.
    """
    node_root = cfg["gate"]
    for key, expected in PHASE_START_GATE_SETTINGS.items():
        node: object = node_root
        for part in key.split("."):
            assert isinstance(node, dict) and part in node, (
                f"gate setting '{key}' is MISSING from config/gate.toml. 30-SPEC prohibition "
                "R5: MUST NOT loosen any config/gate.toml threshold, band, or frozen value to "
                "make a target pass -- removing a setting is a loosening. A fix-cycle must be "
                "a candidate-side change; the only permitted lever is the one pre-registered "
                "in backtest/group_gate_constants.py (D30-11)."
            )
            node = node[part]
        assert node == expected, (
            f"gate setting '{key}' MOVED: phase-start {expected!r} -> current {node!r}. "
            "30-SPEC prohibition R5: MUST NOT loosen any config/gate.toml threshold, band, or "
            "frozen value to make a target pass. If a target is failing, the fix-cycle must be "
            "a candidate-side change -- the only permitted lever is the one pre-registered in "
            "backtest/group_gate_constants.py (D30-11), never the judge. If this change IS "
            "deliberate, re-read PHASE_START_GATE_SETTINGS from the new committed config in "
            "the same commit as the config edit, with the rationale."
        )


def test_committed_thresholds_match_phase_start_snapshot() -> None:
    """30-09 (T-30-39): every non-baseline gate setting is unchanged since the phase start.

    The whole of Phase 30 measures candidates against config/gate.toml. This asserts the judge
    did not move underneath them: alpha, the per-season-must-pass flag, the floor mode, the
    calibration-in-gate flag, the holdout season list, the secondary evaluation mode and each
    secondary band. The [baseline.*] block is excluded on purpose -- see the comment on
    PHASE_START_GATE_SETTINGS.
    """
    cfg = gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    gate.validate_gate_config(cfg)
    assert_phase_start_thresholds_unchanged(cfg)


def test_freshness_tolerance_parity() -> None:
    """30-09 (T-30-40): the freshness tolerance is declared twice, agrees, and has not widened.

    ``_FRESHNESS_TOL`` is the recomputation band ``_drift_tripwire`` uses to decide whether a
    re-score of the deployed incumbents still reproduces the frozen [baseline.*] block. It is
    declared independently in TWO places -- ``scripts/promote_models.py`` (the gate-time abort)
    and ``tests/integration/test_promote_models.py`` (the committed freshness test). Two
    independent declarations of the same tolerance can drift silently, and a drifted test-side
    copy is the dangerous direction: a stale baseline would pass its own freshness check while
    the gate aborted on it -- or, if both were loosened, a baseline measured on gold that no
    longer exists would sail through both.

    Three assertions, because there are three ways this goes wrong:
      1. the two declarations DISAGREE -- IMPORTED through their modules and compared, never
         re-declared here;
      2. the shared value has WIDENED since the phase start (SPEC R5: no new tolerance is
         introduced in this phase);
      3. a THIRD declaration appears somewhere in the tree -- the source scan is what makes
         "no new tolerance anywhere" checkable rather than a promise.

    The constant's own comment names the places that must agree; this asserts the agreement
    instead of trusting the comment.
    """
    from scripts.promote_models import _FRESHNESS_TOL as promote_tol
    from tests.integration.test_promote_models import _FRESHNESS_TOL as test_tol

    assert promote_tol == test_tol, (
        f"the freshness tolerance DIVERGED: scripts/promote_models.py declares {promote_tol} "
        f"but tests/integration/test_promote_models.py declares {test_tol}. The gate-time drift "
        "abort and the committed freshness test would then disagree about whether the frozen "
        "baseline still describes the deployed artifacts. Bring the two back into step; do not "
        "pick whichever is more permissive."
    )
    assert promote_tol == _PHASE_START_FRESHNESS_TOL, (
        f"the freshness tolerance MOVED: phase-start {_PHASE_START_FRESHNESS_TOL} -> current "
        f"{promote_tol}. 30-SPEC R5: no new tolerance is introduced in this phase, and a widened "
        "recomputation band would let a baseline measured on different gold pass the drift "
        "tripwire. A fix-cycle must be a candidate-side change."
    )

    declarations = tuple(
        sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.rglob("*.py")
            if ".venv" not in path.parts
            and _FRESHNESS_TOL_DECL.search(path.read_text(encoding="utf-8"))
        )
    )
    assert declarations == _FRESHNESS_TOL_SITES, (
        f"the freshness tolerance is declared in {list(declarations)}, expected exactly "
        f"{list(_FRESHNESS_TOL_SITES)}. SPEC R5 forbids introducing a new tolerance in this "
        "phase; a third copy is one more thing that can drift silently. Import one of the two "
        "existing declarations instead of adding another."
    )


def test_threshold_snapshot_catches_a_widened_band() -> None:
    """30-09 (T-30-39): the snapshot check FAILS on a loosened band, not just on nothing.

    The fail-closed control for the snapshot test below. A check that has only ever been
    observed passing is indistinguishable from a check that cannot fail, and this one exists
    solely to make a loosening impossible to slip through -- so it has to be shown catching
    one. Widens ``wp_ece_max_increase`` in an in-memory copy of the committed config (the
    exact shape of the move SPEC R5 prohibits: nudging a band until a failing target passes)
    and asserts the check rejects it and says why.
    """
    cfg = gate.load_gate_config(REPO_ROOT / "config" / "gate.toml")
    loosened = copy.deepcopy(cfg)
    loosened["gate"]["secondary"]["wp_ece_max_increase"] = 0.05

    with pytest.raises(AssertionError) as excinfo:
        assert_phase_start_thresholds_unchanged(loosened)

    message = str(excinfo.value)
    assert "wp_ece_max_increase" in message, (
        f"the failure must name the setting that moved; got: {message}"
    )
    assert "candidate-side" in message, (
        f"the failure must state the remediation (a fix-cycle is candidate-side); got: {message}"
    )
