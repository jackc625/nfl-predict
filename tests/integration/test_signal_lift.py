"""Determinism scaffold for the add-one-in signal-lift screen (SIG-05 / SC5).

The lift screen (`backtest/signal_lift.py`) and its single re-runnable entry point
`run_signal_lift_screen(...)` do not exist yet -- they land in Plan 28-07. This file
is the Wave-0 RED scaffold for:

  1. The deterministic add-one-in screen on a fixture gold -- per-group, per-target
     paired incremental CLV delta with the D-05 keep/drop rule
     (keep on >=1 target AND not vetoed on ANY target).
  2. The `ou_divergence.py`-style determinism guard -- two runs over the same fixture
     gold produce identical structured output (the anti-rot guard the D-20
     SIGNAL-LIFT-READOUT.md doc-drift test runs against).

Every test is skipped with a plan-named reason until Plan 28-07 fills them.
"""

import pytest


class TestSignalLiftScreen:
    """SKIPPED scaffold: the add-one-in paired-CLV lift screen (Plan 28-07)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- backtest/signal_lift.py lands in Plan 28-07 "
        "(per-group/per-target paired CLV delta; D-05 keep on >=1 target AND not "
        "vetoed on ANY target)"
    )
    def test_add_one_in_screen_applies_keep_drop_rule(self):
        """Each feature group is screened add-one-in over a fixture gold; the
        paired incremental CLV delta per target drives the D-05 keep/drop rule.
        Filled by Plan 28-07."""
        raise NotImplementedError("Plan 28-07")


class TestSignalLiftDeterminism:
    """SKIPPED scaffold: the ou_divergence.py-style determinism guard (Plan 28-07)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- backtest/signal_lift.py lands in Plan 28-07 "
        "(two runs over the same fixture gold produce identical structured output)"
    )
    def test_run_signal_lift_screen_is_deterministic(self):
        """run_signal_lift_screen(...) over the same fixture gold twice yields
        byte-identical structured output -- the anti-rot guard backing the D-20
        SIGNAL-LIFT-READOUT.md. Filled by Plan 28-07."""
        raise NotImplementedError("Plan 28-07")
