"""A p-value EXACTLY at alpha PASSES, and this module asserts the code's own convention.

Phase 33, Plan 33-08 Task 3 (COLD-04, R5 adjacency edge).

WHY THIS IS A PRESERVATION CONTROL AND NOT A CHANGE
-----------------------------------------------------
``models/deploy_gate.clv_non_regression_passes`` reads
``not (sig["mean"] < 0 and sig["p"] < alpha)`` -- a STRICT ``<``. So a p-value exactly
equal to alpha, with a negative mean, does NOT fail. That is already the behaviour; this
module pins it rather than choosing it. Asserting against a freshly-chosen convention
would be writing a second answer to a question the code has already answered, and the
next reader would have no way to tell which one the gate actually applies.

It is therefore GREEN from its first run, by design. There is no RED phase for a control
over an existing fact.

WHY THE JUST-BELOW COMPANION SHIPS WITH IT
-------------------------------------------
``test_a_p_value_exactly_at_alpha_passes`` alone would also pass on a gate that never
failed anything. ``test_a_p_value_just_below_alpha_fails`` is what makes the boundary
non-vacuous: the same negative mean, a p-value one representable step lower, and the
verdict flips. Two tests with distinct messages, because "the boundary is inclusive" and
"the boundary is a boundary at all" are two claims.

CONSTRUCTING A P EXACTLY AT ALPHA
----------------------------------
No sample of real numbers lands on p == 0.05 exactly, so the significance function is
stubbed with a synthetic result at the boundary and the DECISION is what gets tested.
Alpha itself is READ from ``load_gate_config`` -- the committed ``config/gate.toml`` --
never typed as 0.05 here, so a config change moves this test with it.

NO TEST HERE WRITES ANYTHING.

Run this module:  uv run pytest tests/unit/test_deploy_gate_alpha_boundary.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import Any

import numpy as np

from models import deploy_gate as gate

REPO_ROOT = Path(__file__).resolve().parents[2]
GATE_TOML = REPO_ROOT / "config" / "gate.toml"


def _committed_alpha() -> float:
    """The alpha the gate actually runs on, read from the committed config."""
    return float(gate.load_gate_config(GATE_TOML)["gate"]["alpha"])


def _significance_stub(mean: float, p: float) -> Any:
    """A ``clv_significance``-shaped result pinned to an exact (mean, p).

    Real samples never land on p == alpha exactly, so the boundary can only be driven by
    supplying the statistic. The DECISION rule is what is under test, not the t-test.
    """

    def _stub(_values: Any) -> dict[str, Any]:
        return {
            "n": 500,
            "mean": mean,
            "t": -1.96,
            "p": p,
            "ci95": (mean - 1, mean + 1),
        }

    return _stub


class TestTheAlphaBoundaryIsInclusive:
    """Exactly at alpha PASSES; a hair below FAILS. Both arms, distinct messages."""

    def test_a_p_value_exactly_at_alpha_passes(self, monkeypatch) -> None:
        """p == alpha with a negative mean is NOT significantly worse -- strict `<`."""
        alpha = _committed_alpha()
        monkeypatch.setattr(gate, "clv_significance", _significance_stub(-0.5, alpha))
        assert gate.clv_non_regression_passes(np.zeros(500), alpha=alpha) is True, (
            "a p-value EXACTLY at alpha must PASS: deploy_gate reads "
            "'sig[p] < alpha' with a strict less-than, and this asserts that operator "
            "rather than introducing a second convention"
        )

    def test_a_p_value_just_below_alpha_fails(self, monkeypatch) -> None:
        """The companion arm. Without it the test above would pass on a toothless gate."""
        alpha = _committed_alpha()
        just_below = math.nextafter(alpha, 0.0)
        assert just_below < alpha
        monkeypatch.setattr(
            gate, "clv_significance", _significance_stub(-0.5, just_below)
        )
        assert gate.clv_non_regression_passes(np.zeros(500), alpha=alpha) is False, (
            "a p-value one representable step BELOW alpha with a negative mean must "
            "FAIL, or the boundary test above is vacuous"
        )

    def test_the_same_boundary_holds_for_the_absolute_floor(self, monkeypatch) -> None:
        """The legacy D24-01 floor reads the same operator; one convention, not two."""
        alpha = _committed_alpha()
        monkeypatch.setattr(gate, "clv_significance", _significance_stub(-0.5, alpha))
        assert gate.clv_floor_passes(np.zeros(500), alpha=alpha) is True
        monkeypatch.setattr(
            gate,
            "clv_significance",
            _significance_stub(-0.5, math.nextafter(alpha, 0.0)),
        )
        assert gate.clv_floor_passes(np.zeros(500), alpha=alpha) is False

    def test_a_positive_mean_passes_at_any_p(self, monkeypatch) -> None:
        """Only the NEGATIVE tail can fail -- the D24-02 one-sided reading."""
        alpha = _committed_alpha()
        monkeypatch.setattr(gate, "clv_significance", _significance_stub(0.5, 1e-12))
        assert gate.clv_non_regression_passes(np.zeros(500), alpha=alpha) is True


class TestTheOperatorItselfHasNotMoved:
    """If the source operator changes, the adjacency edge above loses its premise."""

    def test_the_source_still_reads_a_strict_less_than(self) -> None:
        src = inspect.getsource(gate.clv_non_regression_passes)
        assert 'sig["p"] < alpha' in src or "sig['p'] < alpha" in src, src

    def test_the_untestable_branch_is_still_a_strict_fail(self) -> None:
        """``t is None`` -> False. Plan 33-08's empty-pairs edge stands on this branch."""
        src = inspect.getsource(gate.clv_non_regression_passes)
        assert "is None" in src, src

    def test_an_untestable_sample_fails_rather_than_defaulting_to_pass(
        self, monkeypatch
    ) -> None:
        """Driven, not read: below MIN_CLV_SAMPLE the gate refuses."""

        def _too_small(_values: Any) -> dict[str, Any]:
            return {"n": 3, "mean": -0.5, "t": None, "p": None, "ci95": None}

        monkeypatch.setattr(gate, "clv_significance", _too_small)
        assert gate.clv_non_regression_passes(np.zeros(3)) is False

    def test_alpha_is_read_from_the_committed_config_and_is_the_diagnosis_alpha(
        self,
    ) -> None:
        """D24-13 parity: the config mirrors the imported constant rather than forking it."""
        assert _committed_alpha() == gate.SIGNIFICANCE_ALPHA
