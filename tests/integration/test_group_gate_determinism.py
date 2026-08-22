"""SPEC R4's exact-reproduction guard for the BINDING Stage-1 grid (Phase 30, PROD-01).

SPEC R4 requires that "the measurement runs once and re-running it on the same gold reproduces
the grid exactly". This module runs ``run_group_gate`` twice against the same gold and odds and
asserts the two STRUCTURED results agree -- verdicts, raw per-cell delta means, raw two-sided
p-values, BH ranks and rejection flags.

WHY THE REPRODUCTION IS POSSIBLE, stated so a future reader knows what would break it:

* ``run_signal_lift_screen`` computes each target's baseline leg ONCE and reuses it across all
  three groups (backtest/signal_lift.py:745-777), so the baseline cannot differ cell to cell.
* Every leg -- baseline and candidate alike -- runs ``train_and_evaluate(..., tune=False)``
  (backtest/signal_lift.py:479). No Optuna study is opened, so no hyperparameter draw enters
  the delta and the D30-DEFER-01 zero-trial resume trap is not on this path at all.
* Every trainer carries ``random_state=42``, and O/U additionally pins
  ``np.random.seed(self.random_state)`` before its Poisson score simulation.
* Nothing in the path is seeded from the wall clock, and the temporal split comes from
  ``TemporalSplitConfig.default()`` (backtest/signal_lift.py:750), not from "today".

Break any one of those and this module goes red, which is the point.

COST -- READ THIS BEFORE FOLDING IT INTO ANYTHING FAST. Each ``run_group_gate`` call trains one
baseline leg per target plus one candidate leg per (group, target) cell: 3 + 9 = 12 walk-forward
model fits, so the double run is 24 fits. The naive accounting "2 legs x 9 cells x 2 runs = 36
fits" is the upper bound this module does NOT pay, precisely because the baseline leg is computed
once per target and reused -- the same property the determinism above rests on.

MEASURED, not estimated: 24 fits ran in ~39 s wall clock on the Plan 30-05 checkout (pre-rebuild
gold, 2002-2025 x 209/210/209 columns, ``TemporalSplitConfig.default()`` -- train 2018-2019,
holdout 2021-2024). The figure is recorded rather than guessed because the guess this module was
specified with ("minutes on CPU") was wrong by an order of magnitude, and an inflated cost note is
how a cheap guard ends up excluded from the suite for no reason. Expect it to grow with gold: the
Wave-4 rebuild widens the frame and adds rows, and a fix-cycle that widens the selection train
window (D25-05, the one pre-registered lever) lengthens every fit.

It is still NOT a unit test and must never be wired into a pre-commit hook: it reads real gold and
odds off disk, it trains models, and it is meant to be run at the Plan 30-10 binding measurement.
Per the phase-wide test-classification rule in Plan 30-04 that declaration lives here in the
docstring rather than being left implicit in the marker.

NON-BINDING. Anything this module computes is a determinism check on whatever gold happens to be
on disk. It is NOT the Stage-1 measurement, and no number produced here may be cited as a Stage-1
result -- Plan 30-10 runs the binding grid, on rebuilt gold, and records it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from backtest.group_gate import render_verdict_toml, run_group_gate

_GOLD_PATHS = tuple(
    Path("data") / "gold" / f"features_{target}.parquet"
    for target in ("wp", "ats", "ou")
)
_ODDS_PATH = Path("data") / "silver" / "odds_snapshot.parquet"

# The per-cell fields whose reproduction SPEC R4 is actually about. Compared on the STRUCTURED
# result rather than on printed text, so a change to the report or the TOML rendering can never
# mask a change to a number.
_CELL_FIELDS = (
    "measured",
    "exclusion_reason",
    "n_paired",
    "n_group_columns",
    "n_group_columns_selected",
    "delta_mean",
    "delta_p",
    "bh_rejected",
    "bh_rank",
    "q_display",
    "mde",
)


def _require_inputs() -> None:
    """Skip with an explicit message when gold or the odds snapshot is absent.

    Deliberately narrow: it guards on the ABSENCE OF INPUTS only. Widening it to swallow an
    exception from the run itself would turn a genuine non-determinism finding into a green
    skip, which is the one outcome this module exists to prevent.
    """
    missing = [path for path in (*_GOLD_PATHS, _ODDS_PATH) if not path.exists()]
    if missing:
        pytest.skip(
            "Canonical gold / odds not present at "
            f"{[str(path) for path in missing]}; build them first (see PIPELINE.md)"
        )


@pytest.fixture(scope="module")
def double_run() -> tuple[dict[str, Any], dict[str, Any]]:
    """Run the gate TWICE on the same on-disk gold and odds.

    Module-scoped on purpose: the pair costs 24 model fits, and every assertion below reads the
    same pair. Each run loads gold and odds through the default read-only path, so the fixture
    exercises the real re-run -- "run the command again tomorrow and get the same grid" -- rather
    than a single load handed to two callers.
    """
    _require_inputs()
    return run_group_gate(), run_group_gate()


def _exactly_equal(first: Any, second: Any) -> bool:
    """Exact equality, with NaN treated as equal to NaN.

    Exact, NOT approximate. This is a determinism guard, not a tolerance guard: a tolerance here
    would hide precisely the non-determinism the module exists to detect. The NaN handling is not
    a loosening -- ``float('nan') != float('nan')`` by IEEE-754, so without it a reproduced NaN
    would read as a difference.
    """
    if isinstance(first, float) and isinstance(second, float):
        if math.isnan(first) and math.isnan(second):
            return True
    return bool(first == second)


@pytest.mark.integration
class TestStage1GridReproducesExactly:
    """Two ``run_group_gate`` calls on the same gold agree cell for cell (SPEC R4)."""

    def test_the_two_runs_cover_the_same_groups_and_targets(
        self, double_run: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        first, second = double_run
        assert set(first["verdicts"]) == set(second["verdicts"]), (
            "The two runs screened different groups; the grid itself is not reproducible."
        )
        for group in first["verdicts"]:
            assert set(first["verdicts"][group]["cells"]) == set(
                second["verdicts"][group]["cells"]
            ), f"Group '{group}' covered different targets across the two runs."

    def test_the_verdicts_are_identical(
        self, double_run: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        first, second = double_run
        for group, entry in first["verdicts"].items():
            other = second["verdicts"][group]
            assert entry["verdict"] == other["verdict"], (
                f"Group '{group}' was ruled {entry['verdict']} on the first run and "
                f"{other['verdict']} on the second. A Stage-1 verdict that changes between two "
                "runs on the same gold cannot be pre-registered evidence of anything."
            )
            for key in (
                "measured_targets",
                "positive_targets",
                "rejected_positive_targets",
                "rejected_negative_targets",
            ):
                assert entry[key] == other[key], (
                    f"Group '{group}' reported different {key} across the two runs: "
                    f"{entry[key]} then {other[key]}."
                )

    def test_every_cell_field_reproduces_exactly(
        self, double_run: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Raw deltas, raw two-sided p-values, ranks, rejection flags, q-values and MDEs."""
        first, second = double_run
        for group, entry in first["verdicts"].items():
            for target, cell in entry["cells"].items():
                other = second["verdicts"][group]["cells"][target]
                for field in _CELL_FIELDS:
                    assert _exactly_equal(cell[field], other[field]), (
                        f"Cell {group}/{target} field '{field}' moved between two runs on the "
                        f"same gold: {cell[field]} then {other[field]}. Compared for EXACT "
                        "equality on purpose -- a tolerance would hide the non-determinism this "
                        "guard exists to detect."
                    )

    def test_the_correction_family_and_denominator_reproduce(
        self, double_run: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        first, second = double_run
        assert first["bh_denominator"] == second["bh_denominator"], (
            "The BH denominator differed between the two runs, so a different multiple-"
            "comparison correction was applied to the same measurement."
        )
        assert first["bh_family"] == second["bh_family"]
        assert first["bh_rank_order"] == second["bh_rank_order"], (
            "The BH rank order differed between the two runs. The readout publishes ranks, so "
            "the tie-break must be deterministic, not merely the rejection set."
        )
        assert first["excluded_cells"] == second["excluded_cells"]

    def test_the_rendered_verdict_block_is_byte_identical(
        self, double_run: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        """Supplementary: the same grid renders to the same bytes on REAL measured values.

        The binding comparison is the structured one above. This adds the end-to-end fact Plan
        30-10 depends on -- that the block-pasted config/group_gate_verdict.toml is reproducible
        from a re-run -- exercised against real p-values and MDEs rather than synthetic ones.
        """
        first, second = double_run
        assert render_verdict_toml(first) == render_verdict_toml(second), (
            "The ratified-verdict block differed between two runs on the same gold. Plan 30-10 "
            "asserts the committed config file is byte-identical to the generator's output, so "
            "this difference would read there as tampering with a measurement artifact."
        )
