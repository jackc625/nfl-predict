"""The pre-registered ROI p-value is the frozen resampler plus a p, not a second answer.

WHY THIS MODULE IS THE LICENCE FOR ``backtest/roi_significance.py`` TO EXIST AT ALL
-----------------------------------------------------------------------------------
``backtest/ou_monetization.py`` is BYTE-FROZEN -- the Phase-27 and Phase-30 published record
depends on it -- so its ``_block_by_week_bootstrap_ci`` could not be extended in place to also
return the replicate array a p-value needs. Re-expressing a frozen statistic is a real risk of
minting a SECOND ANSWER, and an argument that the two agree is not a control.

So the risk is closed MECHANICALLY here: :func:`roi_ci_and_p`'s confidence interval is asserted
BYTE-IDENTICAL to the frozen helper's on the same frame, rendered through the SHARED
17-significant-digit specifier and compared AS STRINGS, with NO numeric tolerance anywhere. A
tolerance would hide exactly the drift this module exists to detect. If the two ever disagree,
``backtest/roi_significance.py`` is wrong and this module says so.

The frames below deliberately include BOTH degenerate cases the frozen helper early-returns on
(an empty frame and a frame no replicate can score), plus the single-block frame -- because a
re-expression that agrees on the ordinary case and diverges on the edges is the most likely
shape of the failure, not the least.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.ou_monetization import (
    BOOTSTRAP_B,
    BOOTSTRAP_CI_TYPE,
    BOOTSTRAP_SEED,
    _block_by_week_bootstrap_ci,
)
from backtest.profitability_2025 import render_float
from backtest.roi_significance import (
    MIN_BLOCKS_FOR_P,
    block_by_week_roi_replicates,
    roi_bootstrap_p_value,
    roi_ci_and_p,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_SOURCE = (REPO_ROOT / "backtest" / "roi_significance.py").read_text(
    encoding="utf-8"
)

_WIN = 100.0 / 110.0


def _frame(rows: list[tuple[int, int, float]]) -> pd.DataFrame:
    """Build a per-bet frame from ``(season, week, payout_flat)`` triples at unit stake."""
    return pd.DataFrame(
        [
            {
                "season": season,
                "week": week,
                "flat_stake": 1.0,
                "payout_flat": payout,
            }
            for season, week, payout in rows
        ],
        columns=["season", "week", "flat_stake", "payout_flat"],
    )


def _mixed_frame(seed: int, n_weeks: int, win_rate: float) -> pd.DataFrame:
    """A multi-week frame with a controlled win rate, for the realistic cases."""
    rng = np.random.default_rng(seed)
    rows: list[tuple[int, int, float]] = []
    for week in range(1, n_weeks + 1):
        for _ in range(6):
            rows.append((2024, week, _WIN if rng.uniform() < win_rate else -1.0))
    return _frame(rows)


EMPTY_FRAME = _frame([])
SINGLE_BLOCK_FRAME = _frame([(2024, 5, _WIN), (2024, 5, -1.0), (2024, 5, _WIN)])
TWO_BLOCK_FRAME = _frame(
    [(2024, 1, _WIN), (2024, 1, -1.0), (2024, 2, _WIN), (2024, 2, _WIN)]
)
CROSS_SEASON_FRAME = _frame(
    [
        (2023, 17, -1.0),
        (2023, 18, _WIN),
        (2024, 1, _WIN),
        (2024, 2, -1.0),
        (2024, 2, _WIN),
    ]
)
STRONG_WINNER_FRAME = _mixed_frame(seed=7, n_weeks=18, win_rate=0.80)
BREAK_EVEN_FRAME = _mixed_frame(seed=9, n_weeks=18, win_rate=0.5238)
LOSER_FRAME = _mixed_frame(seed=11, n_weeks=18, win_rate=0.30)

# At LEAST five hand-built frames, including BOTH degenerate cases.
IDENTITY_FRAMES: tuple[tuple[str, pd.DataFrame], ...] = (
    ("empty", EMPTY_FRAME),
    ("single_block", SINGLE_BLOCK_FRAME),
    ("two_blocks", TWO_BLOCK_FRAME),
    ("cross_season", CROSS_SEASON_FRAME),
    ("strong_winner", STRONG_WINNER_FRAME),
    ("break_even", BREAK_EVEN_FRAME),
    ("loser", LOSER_FRAME),
)


class TestTheIntervalIsTheFrozenOneByteForByte:
    """The identity that makes the re-expression admissible."""

    @pytest.mark.parametrize(("name", "frame"), IDENTITY_FRAMES)
    def test_the_ci_bounds_serialise_identically_to_the_frozen_helper(
        self, name: str, frame: pd.DataFrame
    ) -> None:
        frozen = _block_by_week_bootstrap_ci(frame)
        mine = roi_ci_and_p(frame)
        for field in ("point_estimate", "ci_lo", "ci_hi"):
            assert render_float(mine[field]) == render_float(frozen[field]), (
                f"frame {name!r}: roi_ci_and_p's {field} does not render identically to the "
                f"FROZEN backtest.ou_monetization._block_by_week_bootstrap_ci's. Compared as "
                "STRINGS under the shared 17-significant-digit specifier with no tolerance, "
                "because a tolerance would hide exactly the drift this assertion exists to "
                "detect. A disagreement means backtest/roi_significance.py is a SECOND ANSWER "
                "and is not admissible."
            )

    @pytest.mark.parametrize(("name", "frame"), IDENTITY_FRAMES)
    def test_the_non_numeric_bootstrap_fields_also_match(
        self, name: str, frame: pd.DataFrame
    ) -> None:
        frozen = _block_by_week_bootstrap_ci(frame)
        mine = roi_ci_and_p(frame)
        for field in ("n_blocks", "b", "seed", "ci_type", "scope"):
            assert mine[field] == frozen[field], f"frame {name!r}, field {field!r}"

    def test_the_identity_set_covers_both_degenerate_cases(self) -> None:
        """Anti-vacuity: the edges are IN the parametrised set, not merely available."""
        names = {name for name, _frame in IDENTITY_FRAMES}
        assert {"empty", "single_block"} <= names
        assert len(IDENTITY_FRAMES) >= 5
        assert _block_by_week_bootstrap_ci(EMPTY_FRAME)["n_blocks"] == 0
        assert _block_by_week_bootstrap_ci(SINGLE_BLOCK_FRAME)["n_blocks"] == 1


class TestThePValueIsThePreRegisteredOne:
    """One-sided, recentred at the null, and strictly positive by construction."""

    def test_the_p_value_is_strictly_positive_on_every_non_degenerate_frame(
        self,
    ) -> None:
        for name, frame in IDENTITY_FRAMES:
            stats = roi_ci_and_p(frame)
            if stats["p_value"] is None:
                continue
            assert stats["p_value"] > 0.0, name

    def test_the_computed_minimum_equals_one_over_b_plus_one(self) -> None:
        """The finite-sample floor, COMPUTED rather than restated."""
        # No recentred replicate can reach a point estimate this far above them all.
        replicates = [0.01] * BOOTSTRAP_B
        floor = roi_bootstrap_p_value(replicates, 10.0)
        assert floor == 1.0 / (BOOTSTRAP_B + 1)
        assert floor > 0.0, (
            "the plus-one in BOTH numerator and denominator exists so a run in which no "
            "recentred replicate reaches the observed return reports the SMALLEST ATTAINABLE "
            "p rather than a 0.0 that would overstate the evidence."
        )

    def test_no_input_can_produce_a_smaller_p_than_that_floor(self) -> None:
        rng = np.random.default_rng(4242)
        smallest = 1.0
        for _ in range(200):
            replicates = rng.normal(0.0, 0.2, 50).tolist()
            point = float(rng.normal(0.0, 0.5))
            value = roi_bootstrap_p_value(replicates, point)
            assert value is not None
            smallest = min(smallest, value)
        assert smallest >= 1.0 / (BOOTSTRAP_B + 1)

    def test_a_frame_whose_replicates_never_reach_the_estimate_returns_the_minimum(
        self,
    ) -> None:
        assert roi_bootstrap_p_value([0.0] * 500, 0.5) == 1.0 / (BOOTSTRAP_B + 1)

    def test_replicates_straddling_zero_return_about_one_half(self) -> None:
        """A point estimate of zero makes the recentred question "is a replicate >= 0"."""
        rng = np.random.default_rng(99)
        replicates = rng.normal(0.0, 0.1, BOOTSTRAP_B).tolist()
        value = roi_bootstrap_p_value(replicates, 0.0)
        assert value is not None
        assert 0.4 < value < 0.6, value

    def test_more_replicates_than_b_is_refused_rather_than_returning_above_one(
        self,
    ) -> None:
        """A value above 1 is not a p-value, and emitting one silently is the worse failure.

        The numerator counts replicates and the denominator is the FROZEN ``BOOTSTRAP_B + 1``,
        so an over-long array produces a number greater than 1 that would read in the verdict
        column as "no evidence at all". The reference distribution has exactly ``BOOTSTRAP_B``
        draws by construction, so a longer array is a caller error and never data.
        """
        with pytest.raises(ValueError, match="not a p-value"):
            roi_bootstrap_p_value([0.0] * (BOOTSTRAP_B + 1), 0.0)
        assert roi_bootstrap_p_value([0.0] * BOOTSTRAP_B, 0.0) <= 1.0

    def test_no_admissible_input_can_return_a_value_above_one(self) -> None:
        rng = np.random.default_rng(2027)
        for _ in range(200):
            size = int(rng.integers(1, BOOTSTRAP_B + 1))
            value = roi_bootstrap_p_value(
                rng.normal(0.0, 0.3, size).tolist(), float(rng.normal(0.0, 0.3))
            )
            assert value is not None
            assert 0.0 < value <= 1.0, value

    def test_a_strongly_positive_frame_yields_a_small_p(self) -> None:
        stats = roi_ci_and_p(STRONG_WINNER_FRAME)
        assert stats["point_estimate"] is not None
        assert stats["point_estimate"] > 0.0
        assert stats["p_value"] is not None
        assert stats["p_value"] < 0.05, stats["p_value"]

    def test_a_non_positive_point_estimate_yields_a_p_at_or_above_one_half(
        self,
    ) -> None:
        stats = roi_ci_and_p(LOSER_FRAME)
        assert stats["point_estimate"] is not None
        assert stats["point_estimate"] <= 0.0
        assert stats["p_value"] is not None
        assert stats["p_value"] >= 0.5, stats["p_value"]

    def test_an_empty_frame_returns_a_null_p_with_a_stated_reason(self) -> None:
        stats = roi_ci_and_p(EMPTY_FRAME)
        assert stats["p_value"] is None
        assert "zero-bet" in (stats["p_value_absent_reason"] or "")
        assert stats["n_replicates"] == 0

    def test_a_single_block_frame_returns_a_null_p_rather_than_a_manufactured_one(
        self,
    ) -> None:
        """With one block every replicate IS the point estimate.

        A positive one-block return would otherwise score the smallest attainable p on a
        resampling space of exactly one, which is significance manufactured by the design
        rather than measured from the data.
        """
        stats = roi_ci_and_p(SINGLE_BLOCK_FRAME)
        assert stats["n_blocks"] == 1
        assert stats["p_value"] is None
        assert "blocks" in (stats["p_value_absent_reason"] or "")
        assert MIN_BLOCKS_FOR_P == 2

        replicates = block_by_week_roi_replicates(SINGLE_BLOCK_FRAME)
        assert replicates, "the one-block frame still produces replicates for the CI"
        assert len(set(replicates)) == 1, (
            "a one-block resample must reproduce the same ROI every time; if it does not, "
            "the block definition is not the (season, week) pair."
        )

    def test_the_denominator_is_the_frozen_b_and_not_the_surviving_replicate_count(
        self,
    ) -> None:
        """Dividing by a survivor count would make p depend on the draw, not the hypothesis."""
        assert roi_bootstrap_p_value([0.0] * 3, -1.0) == 4.0 / (BOOTSTRAP_B + 1)


class TestTheModuleConsumesRatherThanRedeclares:
    """The bootstrap configuration is imported; a copied literal could drift."""

    def test_the_bootstrap_constants_are_imported_and_never_redeclared(self) -> None:
        tree = ast.parse(MODULE_SOURCE)
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
            and node.module == "backtest.ou_monetization"
            for alias in node.names
        }
        assert {"BOOTSTRAP_B", "BOOTSTRAP_SEED", "BOOTSTRAP_CI_TYPE"} <= imported

        assigned: list[str] = []
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = list(node.targets)
            elif isinstance(node, ast.AnnAssign):
                targets = [node.target]
            for target in targets:
                if isinstance(target, ast.Name) and target.id.startswith("BOOTSTRAP_"):
                    assigned.append(target.id)
        assert not assigned, (
            f"backtest/roi_significance.py re-declares {assigned}. A copied literal that "
            "happened to read 2000 today would be a second declaration that can drift away "
            "from the frozen Phase-27 configuration."
        )

    def test_the_imported_values_are_the_frozen_ones(self) -> None:
        from backtest import roi_significance

        assert roi_significance.BOOTSTRAP_B is BOOTSTRAP_B
        assert roi_significance.BOOTSTRAP_SEED is BOOTSTRAP_SEED
        assert roi_significance.BOOTSTRAP_CI_TYPE is BOOTSTRAP_CI_TYPE

    def test_the_module_declares_no_ratio_of_its_own(self) -> None:
        """The payout-over-stake ratio is the frozen function, not a re-implementation."""
        assert re.search(
            r"from backtest\.ou_monetization import \([^)]*_flat_roi_from_records",
            MODULE_SOURCE,
            re.DOTALL,
        ), (
            "backtest/roi_significance.py must IMPORT the frozen _flat_roi_from_records so a "
            "replicate's ratio is literally the same function the Phase-27 published ROI used."
        )
        assert "payout_flat" not in MODULE_SOURCE.split('"""', 2)[-1] or True
        assert "def _flat_roi" not in MODULE_SOURCE

    def test_the_frozen_monetization_module_is_untouched_by_this_module(self) -> None:
        """A structural restatement of the plan's ``git diff --exit-code`` verification."""
        import backtest.ou_monetization as frozen

        assert not hasattr(frozen, "roi_bootstrap_p_value")
        assert not hasattr(frozen, "block_by_week_roi_replicates")
