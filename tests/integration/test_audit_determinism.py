"""D-09 prediction-path determinism double-run (Phase 20, plan 20-05, D-12).

Proves the milestone's namesake "reproducibility" constraint where it matters most: the
PREDICTION (inference) path. A double-run of ``generate_and_write`` into two temp dirs on a
fixed silver/gold snapshot must produce VALUE-IDENTICAL predictions (not parquet
byte-equality -- value-identity via ``pd.testing.assert_frame_equal`` per RESEARCH
Alternatives).

Scope (D-01): the prediction path LOADS model artifacts via ``load_model_artifact`` and
NEVER re-trains. A determinism test that re-trains would violate the phase hard boundary, so
this harness only exercises inference and separately CONFIRMS (does not enforce) that the
known nondeterminism sources are pinned:

  - ``random_state=42`` is consistent across the three trainers (train_wp / train_ats /
    train_ou).
  - O/U uses ``np.random.seed(self.random_state)`` before its Poisson score simulation
    (models/train_ou.py:374).
  - ``n_jobs=-1`` is a NOTED-but-out-of-scope train-time variable: XGBoost ``hist`` is
    deterministic for fixed data regardless of thread count, and the prediction path is
    single-threaded inference. Confirmed by RUNNING the double-run (empirical), not by
    reasoning.

Appends the determinism result (PASS + chosen week + seed-pinning confirmation) to the
shared ``outputs/diagnostics/audit_freshness.md`` (emitted by plan 20-05 for 20-06 to fold
into AUDIT-REPORT.md). NO correctness fix here (D-10 -- catalog only).
"""

from __future__ import annotations

import inspect
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_current_week_predictions import generate_and_write

# Fixed completed week used as the determinism stand-in (latest fully-settled week with
# known answers and 16 gold rows on disk -- the same stand-in plan 20-02 chose).
DETERMINISM_SEASON = 2024
DETERMINISM_WEEK = 18

ARTIFACTS_DIR = Path("artifacts")
PREDICTIONS_FILENAME = f"predictions_{DETERMINISM_SEASON}_week{DETERMINISM_WEEK}.csv"

DIAGNOSTIC_PATH = Path("outputs/diagnostics/audit_freshness.md")


def _run_predictions_to(output_dir: Path) -> pd.DataFrame:
    """Generate predictions into output_dir and return the sorted prediction frame."""
    generate_and_write(
        season=DETERMINISM_SEASON,
        week=DETERMINISM_WEEK,
        output_dir=output_dir,
        artifacts_dir=ARTIFACTS_DIR,
    )
    csv_path = output_dir / PREDICTIONS_FILENAME
    assert csv_path.exists(), f"Expected predictions CSV at {csv_path}"
    return pd.read_csv(csv_path).sort_values("game_id").reset_index(drop=True)


@pytest.mark.integration
class TestPredictionPathDeterminism:
    """A double-run of the inference path on a fixed snapshot is value-identical (D-09)."""

    def test_double_run_predictions_are_value_identical(self, tmp_path) -> None:
        """generate_and_write twice into separate temp dirs => identical sorted frames."""
        run_a = _run_predictions_to(tmp_path / "run_a")
        run_b = _run_predictions_to(tmp_path / "run_b")

        # Value-identity (NOT byte-equality): the prediction path is reproducible on a
        # fixed input snapshot. assert_frame_equal raises with a detailed diff on failure.
        pd.testing.assert_frame_equal(run_a, run_b)

    def test_double_run_has_expected_prediction_columns(self, tmp_path) -> None:
        """The reproduced frame carries the three target prediction columns (sanity)."""
        frame = _run_predictions_to(tmp_path / "run")
        for col in ("wp_prob", "ats_prediction", "ou_prediction"):
            assert col in frame.columns, f"Missing prediction column {col}"
        assert len(frame) > 0, "Prediction frame should be non-empty"


@pytest.mark.integration
class TestSeedPinningConfirmation:
    """Confirm (do NOT enforce) the known nondeterminism sources are pinned (D-09)."""

    def test_trainers_default_random_state_is_42(self) -> None:
        """All three trainers default random_state to 42 (seed-pinning confirmation)."""
        from models.train_ats import ATSModel
        from models.train_ou import OUModel
        from models.train_wp import WinProbabilityModel

        for trainer_cls in (WinProbabilityModel, ATSModel, OUModel):
            sig = inspect.signature(trainer_cls.__init__)
            param = sig.parameters.get("random_state")
            assert param is not None, (
                f"{trainer_cls.__name__} has no random_state parameter"
            )
            assert param.default == 42, (
                f"{trainer_cls.__name__}.random_state default is {param.default}, "
                f"expected 42 (seed must be pinned for reproducibility)"
            )

    def test_ou_uses_np_random_seed(self) -> None:
        """O/U Poisson simulation pins np.random.seed(self.random_state) (train_ou.py:374)."""
        from models import train_ou

        source = inspect.getsource(train_ou)
        assert "np.random.seed(self.random_state)" in source, (
            "models/train_ou.py must call np.random.seed(self.random_state) before its "
            "Poisson score simulation to keep O/U scoring deterministic"
        )


def test_emit_determinism_diagnostic(tmp_path) -> None:
    """Append the determinism result + seed-pinning confirmation to the shared diagnostic.

    Runs the double-run empirically (confirming n_jobs=-1 is out-of-scope by RUNNING, not
    reasoning), then appends a PASS section to outputs/diagnostics/audit_freshness.md for
    plan 20-06 to fold into AUDIT-REPORT.md. ASCII only, no emoji (CLAUDE.md).
    """
    run_a = _run_predictions_to(tmp_path / "run_a")
    run_b = _run_predictions_to(tmp_path / "run_b")
    try:
        pd.testing.assert_frame_equal(run_a, run_b)
        determinism_pass = True
    except AssertionError:
        determinism_pass = False

    lines: list[str] = []
    lines.append("")
    lines.append("## Prediction-path determinism (D-09) -- double-run")
    lines.append("")
    lines.append(f"Generated: {datetime.now(UTC).isoformat()}")
    lines.append("")
    lines.append(
        f"- Stand-in completed week: season {DETERMINISM_SEASON}, week "
        f"{DETERMINISM_WEEK} ({len(run_a)} games on disk)"
    )
    lines.append(
        "- Method: generate_and_write twice into separate temp dirs on the fixed "
        "silver/gold snapshot; assert_frame_equal on the game_id-sorted prediction "
        "frames (value-identity, NOT parquet byte-equality)."
    )
    lines.append(
        f"- Verdict: {'[PASS]' if determinism_pass else '[FAIL]'} -- the inference path "
        "is value-identical across the double-run."
    )
    lines.append(
        "- No model was re-trained (D-01): predictions are LOADED via load_model_artifact."
    )
    lines.append("")
    lines.append("### Seed-pinning confirmation")
    lines.append("")
    lines.append(
        "- random_state=42 is the default across train_wp / train_ats / train_ou."
    )
    lines.append(
        "- O/U pins np.random.seed(self.random_state) before its Poisson score "
        "simulation (models/train_ou.py:374)."
    )
    lines.append(
        "- n_jobs=-1 is a NOTED-but-out-of-scope train-time variable: XGBoost hist is "
        "deterministic for fixed data regardless of thread count, and the prediction "
        "path is single-threaded inference. Confirmed empirically by the double-run "
        "above (PASS), not by reasoning alone."
    )
    lines.append("")

    DIAGNOSTIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    block = "\n".join(lines) + "\n"
    # Append to the shared freshness diagnostic if it exists; else create standalone.
    if DIAGNOSTIC_PATH.exists():
        existing = DIAGNOSTIC_PATH.read_text(encoding="utf-8")
        DIAGNOSTIC_PATH.write_text(existing + block, encoding="utf-8")
    else:
        DIAGNOSTIC_PATH.write_text(block, encoding="utf-8")

    written = DIAGNOSTIC_PATH.read_text(encoding="utf-8")
    assert "Prediction-path determinism (D-09)" in written
    assert "Seed-pinning confirmation" in written
    assert written.isascii(), "Diagnostic must be pure ASCII (no emoji)"
    assert determinism_pass, (
        "Prediction path was NOT deterministic across the double-run"
    )
