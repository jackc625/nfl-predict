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
    """
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
    result_bad = gate.evaluate_target("ats", candidate_bad, baseline, _TEST_CFG)
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
    result_good = gate.evaluate_target("ats", candidate_good, baseline, _TEST_CFG)
    assert result_good["passed"] is True, result_good["reasons"]


def test_empty_per_season_fails_closed() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_empty_per_season_fails_closed.

    WR-04: when per_season_must_pass is on but the candidate supplies NO per-season slices
    (empty or absent map), evaluate_target must FAIL closed -- it must NOT report a hollow
    "all holdout seasons passed". A pooled-passing candidate with an empty per_season is the
    minimal trigger.
    """
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
    result = gate.evaluate_target("ats", candidate_empty, {"mae": 10.0}, _TEST_CFG)
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
        "ats", candidate_absent, {"mae": 10.0}, _TEST_CFG
    )
    assert result_absent["passed"] is False, result_absent["reasons"]


# ---------------------------------------------------------------------------
# WP calibration-in-gate (D24-05)
# ---------------------------------------------------------------------------


def test_wp_calibration_in_gate() -> None:
    """24-VALIDATION: pytest .../test_deploy_gate.py::test_wp_calibration_in_gate.

    A WP candidate that passes the CLV floor but whose ECE exceeds baseline by more than the
    tolerance must FAIL; an equal-ECE/Brier candidate passes.
    """
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
        "wp", _wp_candidate(ece=0.07, brier=0.21), baseline, _TEST_CFG
    )
    assert bad["passed"] is False
    assert any("ece" in r.lower() for r in bad["reasons"]), bad["reasons"]

    # Equal ECE/Brier/accuracy -> PASS.
    good = gate.evaluate_target(
        "wp", _wp_candidate(ece=0.02, brier=0.21), baseline, _TEST_CFG
    )
    assert good["passed"] is True, good["reasons"]


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
    """
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
    result = gate.evaluate_target("ats", candidate, {"mae": 10.0}, _TEST_CFG)
    assert result["passed"] is False
    assert any("CLV" in r for r in result["reasons"]), result["reasons"]


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
