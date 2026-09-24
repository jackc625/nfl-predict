"""What the fixed-weight blend fit reads: the corrected source models and the bound converter.

WHY THIS FILE EXISTS (A33.2-review WR-06)
-----------------------------------------
``backtest.tune`` -- a production CLI that writes a blend artifact -- used to import these
three recorded ids from ``tests.phase33_state``, the 18,000-line test manifest. That made the
CLI unusable wherever ``tests/`` is not importable, and let a test-side edit silently change
which artifacts a production fit reads. ``models/train.py`` already records that direction as
wrong and avoids it. The ids now live HERE, a committed constants module beside
``config/tuning_preregistration.py``, and the test manifest keeps its own record of the same
values: ``tests/unit/test_fixed_blend_weight.py`` asserts the two agree, so neither can drift
without a failing test.

WHAT THE VALUES ARE. Plan 33.2-24 step 24b re-fitted the converter on the repaired owned
corpus (1,344 graded games), and step 25b re-fitted the three models with the snap coverage
flag left out. The blend is tuned on exactly those three models' walk-forward predictions and
binds exactly that converter. The gold generation is the one all three were trained on; the
tuner refuses to run if the source artifacts, or the operator-measured live gold, disagree.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

__all__ = [
    "BLEND_CONVERTER_ARTIFACT_ID",
    "BLEND_SOURCE_ARTIFACT_IDS",
    "BLEND_SOURCE_GOLD_GENERATION",
]

#: The converter the blend binds (``tests.phase33_state.P332_24B_CONVERTER_ARTIFACT_ID``).
BLEND_CONVERTER_ARTIFACT_ID: str = "market_probability_20260923_195443"

#: ``(target, artifact_id)`` for the three corrected source models
#: (``tests.phase33_state.P332_25B_REFIT_ARTIFACT_IDS``).
BLEND_SOURCE_ARTIFACT_IDS: tuple[tuple[str, str], ...] = (
    ("wp", "wp_20260923_172144"),
    ("ats", "ats_20260923_172148"),
    ("ou", "ou_20260923_172152"),
)

#: The gold generation all three source models were trained on
#: (``tests.phase33_state.P332_25B_REFIT_GOLD_GENERATION``).
BLEND_SOURCE_GOLD_GENERATION: str = (
    "484397642530db5b28c49d9234ecfb90e3860783f1b1b41766ab6151a1522597"
)
