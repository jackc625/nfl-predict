"""Unit suite for honest-sizing exposure caps + correlated de-weighting (BET-03, PROD-02).

Covers the ADDITIVE helpers in ``utils/kelly_criterion.py`` that implement the
Phase 27 sizing discipline (D27-09/10/11) as EXTENDED by Phase 31 (D31-03):

  - 1 unit = 1% of bankroll (D27-09).
  - The existing 5%-of-bankroll per-bet cap is retained as a rarely-binding net.
  - Per-week 10%-of-bankroll pro-rata total-exposure cap (D27-10, NEW in P27).
  - Same-week CORRELATED de-weighting using the FROZEN formula
    ``stake / sqrt(max(same_side_group_size, same_game_group_size))``
    (D27-11 same-side; D31-03 adds the same-game cross-target grouping;
    covariance/joint-Kelly path REJECTED).
  - Full-Kelly EXCLUDED -- the operative per-bet ceiling is the 5% cap.
  - The LOCKED cap order:
    ``calibrated Kelly stake -> 5% per-bet cap -> correlated de-weight -> 10% weekly cap``.

Mirrors the ``tests/unit/test_betting_simulation.py`` class-grouped style:
exact-value asserts (no subjective language), requirement-ID docstrings.

Each test docstring cites BET-03 and the LOCKED D27-NN / D31-NN decision or
review tightening it proves:
  - D27-09: unit=1%, full-Kelly excluded (the 5% per-bet cap is the ceiling).
  - D27-10: per-week 10% pro-rata cap (preserve ratios, drop no bets).
  - D27-11: same-side de-weight; no covariance path.
  - D31-03: same-game CROSS-TARGET grouping; the STRICTER of the two groupings
    applies (they are never composed); the extension is monotone at the
    de-weight step (no de-weighted stake can rise) and never changes which bets
    are selected or the flat-stake headline.
  - Review #2: FROZEN de-weight formula + LOCKED cap order + metadata + grouping.
  - Review #2 (Gemini): released-room reuse (de-weight runs BEFORE the weekly cap).
  - Review #8: edge cases (empty/zero/NaN/invalid-bankroll/single/all-one-week).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import math
import re

import pytest

import utils.kelly_criterion as kelly_module
from utils.kelly_criterion import (
    CAP_ORDER,
    PER_BET_CAP_PCT,
    SAME_GAME_DEWEIGHT,
    SAME_SIDE_DEWEIGHT,
    UNIT_PCT_OF_BANKROLL,
    WEEKLY_CAP_PCT,
    apply_same_game_and_side_deweight,
    apply_sizing_pipeline,
    apply_weekly_exposure_cap,
    unit_size,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_TOL = 1e-9


# ---------------------------------------------------------------------------
# Fixture helper
# ---------------------------------------------------------------------------


def _bets(side: str, stake: float, count: int, *, game_prefix: str = "G") -> list[dict]:
    """Build ``count`` bets on the SAME side, each on a DISTINCT game.

    Distinct ``game_id`` values keep every same-game group at size 1, so the
    same-SIDE group is the binding one and every pre-D31-03 expectation in this
    module holds numerically unchanged -- which is the point: the extension is a
    no-op when no two bets share a game.
    """
    return [
        {"game_id": f"{game_prefix}{index}", "bet_side": side, "stake": stake}
        for index in range(count)
    ]


# ---------------------------------------------------------------------------
# Pre-registered module constants (D27-09/10/11; review #2 LOCKED order)
# ---------------------------------------------------------------------------


class TestPreRegisteredConstants:
    """The frozen sizing constants are pinned to their decided values."""

    def test_unit_pct_constant(self) -> None:
        """BET-03 / D27-09: 1 unit = 1% of bankroll."""
        assert UNIT_PCT_OF_BANKROLL == 0.01

    def test_weekly_cap_constant(self) -> None:
        """BET-03 / D27-10: per-week total-exposure cap = 10% of bankroll."""
        assert WEEKLY_CAP_PCT == 0.10

    def test_per_bet_cap_constant(self) -> None:
        """BET-03 / D27-09: the existing 5% per-bet cap is retained + named."""
        assert PER_BET_CAP_PCT == 0.05

    def test_cap_order_is_locked(self) -> None:
        """Review #2 / D31-03: the LOCKED cap ORDER is documented as a tuple.

        The LOCKED thing is the ORDER -- kelly, then the 5% per-bet cap, then
        correlated de-weighting, then the 10% weekly cap. D31-03 widened the
        third step from same-side-only to same-game-and-side grouping, so the
        step's NAME was updated in lockstep with the single formula it names.
        Its POSITION (index 2 of 4) is unchanged, which is what the lock is
        about; the assertions below pin both the names and the position.
        """
        assert CAP_ORDER == (
            "kelly_stake",
            "per_bet_5pct_cap",
            "same_game_and_side_deweight",
            "weekly_10pct_cap",
        )
        # The lock proper: four steps, de-weighting third, weekly cap last.
        assert len(CAP_ORDER) == 4
        assert CAP_ORDER.index("same_game_and_side_deweight") == 2
        assert CAP_ORDER[-1] == "weekly_10pct_cap"


# ---------------------------------------------------------------------------
# (1) unit = 1% of bankroll
# ---------------------------------------------------------------------------


class TestUnitDefinition:
    """1 unit = 1% of the configured bankroll (D27-09)."""

    def test_unit_is_one_percent_of_bankroll(self) -> None:
        """BET-03 / D27-09: bankroll 10000 -> unit 100."""
        assert unit_size(10000.0) == pytest.approx(100.0, abs=_TOL)

    def test_unit_scales_with_bankroll(self) -> None:
        """BET-03 / D27-09: unit tracks the bankroll linearly at 1%."""
        assert unit_size(25000.0) == pytest.approx(250.0, abs=_TOL)
        assert unit_size(500.0) == pytest.approx(5.0, abs=_TOL)


# ---------------------------------------------------------------------------
# (2) existing 5% per-bet cap still binds
# ---------------------------------------------------------------------------


class TestPerBetCap:
    """The existing max_bet_pct=0.05 per-bet cap is retained and still binds."""

    def test_single_bet_capped_at_five_percent(self) -> None:
        """BET-03: a raw Kelly stake above 5% of bankroll is capped to 5%.

        bankroll 10000, raw stake 800 (8%) -> capped to 500 (5%).
        """
        bankroll = 10000.0
        bets = [{"game_id": "G0", "bet_side": "under", "stake": 800.0}]
        result = apply_sizing_pipeline(bets, bankroll)
        # After the 5% cap: 500. Single-bet de-weight factor 1/sqrt(1) = 1.0.
        # Sum 500 = 5% <= 10% weekly cap -> no-op.
        assert result[0]["weekly_scaled_stake"] == pytest.approx(500.0, abs=_TOL)

    def test_per_bet_cap_noop_when_below_five_percent(self) -> None:
        """BET-03: a raw stake under 5% is not touched by the per-bet cap."""
        bankroll = 10000.0
        bets = [{"game_id": "G0", "bet_side": "under", "stake": 300.0}]
        result = apply_sizing_pipeline(bets, bankroll)
        assert result[0]["weekly_scaled_stake"] == pytest.approx(300.0, abs=_TOL)


# ---------------------------------------------------------------------------
# (3) LOCKED cap order + EXACT shrink factors
# ---------------------------------------------------------------------------


class TestCapOrderAndExactFactors:
    """The four steps apply in the LOCKED order with EXACT intermediate values."""

    def test_cap_order_and_exact_shrink_factors(self) -> None:
        """Review #2: 4 same-side bets at stake 300 (bankroll 10000).

        Step 1 (Kelly stake): 300 each (given).
        Step 2 (5% per-bet cap = 500): no bind, 300 each.
        Step 3 (same-side de-weight, group_size 4): 300/sqrt(4) = 150.0 each;
                factor = 1/sqrt(4) = 0.5 EXACTLY.
        Step 4 (10% weekly cap): summed 600 = 6% <= 10% -> no-op, 150.0 each.
        """
        bankroll = 10000.0
        bets = _bets("under", 300.0, 4)
        result = apply_sizing_pipeline(bets, bankroll)

        for record in result:
            assert record["original_stake"] == pytest.approx(300.0, abs=_TOL)
            # The 5% cap did not bind, so the per-bet-capped value is the input.
            assert record["scale_factor"] == pytest.approx(1.0 / math.sqrt(4), abs=_TOL)
            assert record["scale_factor"] == pytest.approx(0.5, abs=_TOL)
            assert record["deweighted_stake"] == pytest.approx(150.0, abs=_TOL)
            assert record["weekly_scaled_stake"] == pytest.approx(150.0, abs=_TOL)

    def test_per_bet_cap_binds_before_deweight(self) -> None:
        """Review #2: the 5% cap binds BEFORE de-weight (proves the order).

        2 same-side bets at raw stake 700 (bankroll 10000):
        Step 2 (5% per-bet cap = 500): each capped to 500.
        Step 3 (de-weight, group_size 2): 500/sqrt(2) each.
        Step 4 (weekly cap): summed 2 * 500/sqrt(2) = 707.1 = ~7.07% <= 10% -> no-op.
        """
        bankroll = 10000.0
        bets = _bets("over", 700.0, 2)
        result = apply_sizing_pipeline(bets, bankroll)

        expected_deweighted = 500.0 / math.sqrt(2)
        for record in result:
            assert record["original_stake"] == pytest.approx(700.0, abs=_TOL)
            assert record["deweighted_stake"] == pytest.approx(
                expected_deweighted, abs=_TOL
            )
            assert record["weekly_scaled_stake"] == pytest.approx(
                expected_deweighted, abs=_TOL
            )


# ---------------------------------------------------------------------------
# (4) weekly cap scales to EXACTLY 10%
# ---------------------------------------------------------------------------


class TestWeeklyCapScaling:
    """The weekly cap pro-rata scales an over-cap week to EXACTLY 10%."""

    def test_weekly_cap_scales_to_exactly_ten_percent(self) -> None:
        """Review #2 / D27-10: a week summing to >10% scales to exactly 10%.

        Direct helper test on stakes [600, 400, 200] (sum 1200), bankroll 10000.
        max_total = 1000. scale = 1000/1200. Sum after == 1000 EXACTLY; ratios
        6:4:2 preserved.
        """
        bankroll = 10000.0
        stakes = [600.0, 400.0, 200.0]
        scaled = apply_weekly_exposure_cap(stakes, bankroll)

        scaled_values = [rec["weekly_scaled_stake"] for rec in scaled]
        assert sum(scaled_values) == pytest.approx(0.10 * bankroll, abs=_TOL)

        # Ratios preserved: 6:4:2 -> values proportional.
        assert scaled_values[0] / scaled_values[1] == pytest.approx(6.0 / 4.0, abs=_TOL)
        assert scaled_values[1] / scaled_values[2] == pytest.approx(4.0 / 2.0, abs=_TOL)

        expected_scale = 1000.0 / 1200.0
        for rec in scaled:
            assert rec["scale_factor"] == pytest.approx(expected_scale, abs=_TOL)

    def test_weekly_cap_noop_under_ten_percent(self) -> None:
        """D27-10: a week summing to <10% is returned UNCHANGED (scale 1.0)."""
        bankroll = 10000.0
        stakes = [300.0, 200.0, 100.0]  # sum 600 = 6% < 10%
        scaled = apply_weekly_exposure_cap(stakes, bankroll)
        for original, rec in zip(stakes, scaled, strict=True):
            assert rec["weekly_scaled_stake"] == pytest.approx(original, abs=_TOL)
            assert rec["scale_factor"] == pytest.approx(1.0, abs=_TOL)


# ---------------------------------------------------------------------------
# (5) de-weight + weekly cap metadata fields
# ---------------------------------------------------------------------------


class TestMetadataFields:
    """De-weight and weekly cap return audit metadata (review #2)."""

    def test_deweight_metadata_fields(self) -> None:
        """Review #2: apply_same_game_and_side_deweight returns per-bet metadata.

        Keys: original_stake, deweighted_stake, scale_factor; numerically
        consistent (deweighted == original * scale_factor).
        """
        bets = _bets("under", 400.0, 4)
        deweighted = apply_same_game_and_side_deweight(bets)
        for rec in deweighted:
            assert "original_stake" in rec
            assert "deweighted_stake" in rec
            assert "scale_factor" in rec
            assert rec["scale_factor"] == pytest.approx(1.0 / math.sqrt(4), abs=_TOL)
            assert rec["deweighted_stake"] == pytest.approx(
                rec["original_stake"] * rec["scale_factor"], abs=_TOL
            )

    def test_pipeline_metadata_carries_all_four_keys(self) -> None:
        """Review #2: the orchestration helper carries the full metadata set.

        original_stake, deweighted_stake, weekly_scaled_stake, scale_factor.
        """
        bankroll = 10000.0
        bets = _bets("under", 300.0, 4)
        result = apply_sizing_pipeline(bets, bankroll)
        for rec in result:
            for key in (
                "original_stake",
                "deweighted_stake",
                "weekly_scaled_stake",
                "scale_factor",
            ):
                assert key in rec


# ---------------------------------------------------------------------------
# (6) case-insensitive side grouping + over-grouping
# ---------------------------------------------------------------------------


class TestSideGrouping:
    """Sides normalize case-insensitively; high-total OVER groups with OVER."""

    def test_same_side_grouping_case_insensitive_and_over_grouping(self) -> None:
        """Review #2: 'Under'/'UNDER'/'under' are one group; overs group together.

        Mixed week (2 under + 2 over) de-weights each group by 1/sqrt(2);
        a normal OVER and a high-total OVER share the 'over' group.
        """
        bets = [
            {"game_id": "G0", "bet_side": "Under", "stake": 200.0},
            {"game_id": "G1", "bet_side": "UNDER", "stake": 200.0},
            {"game_id": "G2", "bet_side": "over", "stake": 200.0},
            # e.g. a high-total OVER pocket
            {"game_id": "G3", "bet_side": "OVER", "stake": 200.0},
        ]
        deweighted = apply_same_game_and_side_deweight(bets)
        expected = 1.0 / math.sqrt(2)
        for rec in deweighted:
            assert rec["scale_factor"] == pytest.approx(expected, abs=_TOL)
            assert rec["deweighted_stake"] == pytest.approx(200.0 * expected, abs=_TOL)

    def test_all_under_week_uses_group_size_four(self) -> None:
        """Review #2: an all-under week of N=4 de-weights by 1/sqrt(4)."""
        bets = _bets("under", 200.0, 4)
        deweighted = apply_same_game_and_side_deweight(bets)
        expected = 1.0 / math.sqrt(4)
        for rec in deweighted:
            assert rec["scale_factor"] == pytest.approx(expected, abs=_TOL)


# ---------------------------------------------------------------------------
# (7) released-room reuse (interaction order)
# ---------------------------------------------------------------------------


class TestReleasedRoomReuse:
    """De-weight runs BEFORE the weekly cap, so freed room is naturally used."""

    def test_released_room_reused_after_deweight(self) -> None:
        """Review #2 (Gemini): pre-de-weight sum > 10% but post-de-weight < 10%.

        4 same-side bets at raw 400 (bankroll 10000): pre-de-weight sum 1600 =
        16% > 10%. After de-weight (1/sqrt(4) = 0.5): 200 each, sum 800 = 8% <
        10% -> the weekly cap is a NO-OP. The freed room is NOT re-padded.
        """
        bankroll = 10000.0
        bets = _bets("under", 400.0, 4)
        result = apply_sizing_pipeline(bets, bankroll)
        for rec in result:
            assert rec["deweighted_stake"] == pytest.approx(200.0, abs=_TOL)
            assert rec["weekly_scaled_stake"] == pytest.approx(200.0, abs=_TOL)
        total = sum(rec["weekly_scaled_stake"] for rec in result)
        assert total == pytest.approx(800.0, abs=_TOL)
        assert total < 0.10 * bankroll

    def test_weekly_cap_still_binds_when_post_deweight_over_cap(self) -> None:
        """Review #2 (Gemini): even post-de-weight sum >10% scales pro-rata.

        4 same-side bets at raw 700 (bankroll 10000): 5% cap -> 500 each.
        de-weight (1/sqrt(4) = 0.5) -> 250 each, sum 1000 = 10% exactly.
        Push past with a fifth: 5 bets -> 5% cap 500 each, de-weight 1/sqrt(5),
        sum = 5 * 500/sqrt(5) = 500*sqrt(5) = 1118.03 > 1000 -> scales to 1000.
        """
        bankroll = 10000.0
        bets = _bets("under", 700.0, 5)
        result = apply_sizing_pipeline(bets, bankroll)
        total = sum(rec["weekly_scaled_stake"] for rec in result)
        assert total == pytest.approx(0.10 * bankroll, abs=_TOL)
        # All five stakes equal (identical inputs) -> each is exactly 1000/5.
        for rec in result:
            assert rec["weekly_scaled_stake"] == pytest.approx(200.0, abs=_TOL)


# ---------------------------------------------------------------------------
# (8) full-Kelly excluded -- the operative per-bet fraction ceiling is the 5% cap
# ---------------------------------------------------------------------------


class TestFullKellyExcluded:
    """Full-Kelly is excluded by the operative 5% per-bet cap (no separate half-Kelly clamp).

    The half-Kelly ceiling was removed (WR-01): given the upstream quarter-Kelly fraction and
    the 5% per-bet cap, a 0.5 bankroll-fraction clamp could never bind, so it advertised a
    protection it did not provide. These tests pin the protection that IS real -- the pipeline
    caps every per-bet stake at PER_BET_CAP_PCT (5%) of bankroll.
    """

    def test_per_bet_cap_is_operative_fraction_ceiling(self) -> None:
        """BET-03 / D27-09 / WR-01: a lone stake far above 5% is capped to EXACTLY 5%.

        A single bet (de-weight group size 1) well under the 10% weekly cap, so the 5% per-bet
        cap is the only binding step -- it is the operative per-bet fraction ceiling.
        """
        bankroll = 10_000.0
        result = apply_sizing_pipeline(
            [{"game_id": "G0", "bet_side": "under", "stake": 5_000.0}], bankroll
        )
        assert result[0]["weekly_scaled_stake"] == pytest.approx(
            PER_BET_CAP_PCT * bankroll, abs=_TOL
        )

    def test_stake_below_every_cap_passes_through(self) -> None:
        """A lone stake below the 5% per-bet and 10% weekly caps is returned unchanged."""
        bankroll = 10_000.0
        result = apply_sizing_pipeline(
            [{"game_id": "G0", "bet_side": "over", "stake": 200.0}], bankroll
        )
        assert result[0]["weekly_scaled_stake"] == pytest.approx(200.0, abs=_TOL)


# ---------------------------------------------------------------------------
# (9) edge cases -- hard behavior
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases hard-behave: no divide-by-zero, named ValueErrors (review #8)."""

    def test_empty_bet_list_returns_empty(self) -> None:
        """Review #8: an empty bet list returns empty (no divide-by-zero)."""
        assert apply_sizing_pipeline([], 10000.0) == []
        assert apply_same_game_and_side_deweight([]) == []
        assert apply_weekly_exposure_cap([], 10000.0) == []

    def test_all_zero_stakes_returns_zeros(self) -> None:
        """Review #8: a week of all-zero stakes returns zeros, no NaN/ZeroDiv."""
        scaled = apply_weekly_exposure_cap([0.0, 0.0, 0.0], 10000.0)
        for rec in scaled:
            assert rec["weekly_scaled_stake"] == pytest.approx(0.0, abs=_TOL)
            assert not math.isnan(rec["weekly_scaled_stake"])
            assert rec["scale_factor"] == pytest.approx(1.0, abs=_TOL)

    def test_nan_stake_raises_value_error(self) -> None:
        """Review #8: a NaN stake raises a named ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            apply_weekly_exposure_cap([float("nan"), 100.0], 10000.0)
        with pytest.raises(ValueError, match="NaN"):
            apply_same_game_and_side_deweight(
                [{"game_id": "G0", "bet_side": "under", "stake": float("nan")}]
            )

    def test_invalid_bankroll_raises_value_error(self) -> None:
        """Review #8: bankroll <= 0 raises a named ValueError."""
        with pytest.raises(ValueError, match="bankroll"):
            apply_weekly_exposure_cap([100.0], 0.0)
        with pytest.raises(ValueError, match="bankroll"):
            apply_weekly_exposure_cap([100.0], -5000.0)
        with pytest.raises(ValueError, match="bankroll"):
            unit_size(0.0)

    def test_single_bet_group_factor_is_one(self) -> None:
        """Review #8: a single-bet same-side group de-weights by 1/sqrt(1) = 1.0."""
        deweighted = apply_same_game_and_side_deweight(
            [{"game_id": "G0", "bet_side": "under", "stake": 250.0}]
        )
        assert deweighted[0]["scale_factor"] == pytest.approx(1.0, abs=_TOL)
        assert deweighted[0]["deweighted_stake"] == pytest.approx(250.0, abs=_TOL)

    def test_all_bets_in_one_week_cap_applies_once(self) -> None:
        """Review #8: all bets in one week -> the weekly cap applies once.

        6 same-side bets at 400 (bankroll 10000): 5% cap no-bind; de-weight
        1/sqrt(6) -> 400/sqrt(6) each; sum 6 * 400/sqrt(6) = 400*sqrt(6) =
        979.8 < 1000 -> the single weekly-cap pass leaves them unscaled.
        """
        bankroll = 10000.0
        bets = _bets("under", 400.0, 6)
        result = apply_sizing_pipeline(bets, bankroll)
        expected = 400.0 / math.sqrt(6)
        for rec in result:
            assert rec["weekly_scaled_stake"] == pytest.approx(expected, abs=_TOL)


# ---------------------------------------------------------------------------
# (10) D31-03: same-game CROSS-TARGET grouping -- the stricter grouping wins
# ---------------------------------------------------------------------------


class TestSameGameCrossTargetDeweight:
    """The stricter of the same-side and same-game groupings applies (D31-03)."""

    def test_cross_target_same_game_pair_is_deweighted(self) -> None:
        """D31-03: a WP ``home`` bet and an ATS ``home_cover`` bet on ONE game get 1/sqrt(2).

        The finding that forced D31-03: the three targets' side strings are
        DISJOINT (home/away, home_cover/away_cover, over/under), so same-side
        grouping alone puts this pair in two groups of one and de-weights
        NEITHER (factor 1.0) -- the correlation that actually matters (two bets
        on one game, close to a single leveraged wager) would go uncaptured.
        """
        bets = [
            {"game_id": "2025_W01_A@B", "bet_side": "home", "stake": 200.0},
            {"game_id": "2025_W01_A@B", "bet_side": "home_cover", "stake": 200.0},
        ]
        records = apply_same_game_and_side_deweight(bets)

        expected = 1.0 / math.sqrt(2)
        for record in records:
            assert record["same_side_group_size"] == 1
            assert record["same_game_group_size"] == 2
            assert record["binding_group"] == "same_game"
            assert record["scale_factor"] == pytest.approx(expected, abs=_TOL)
            assert record["deweighted_stake"] == pytest.approx(
                200.0 * expected, abs=_TOL
            )

    def test_larger_same_side_group_dominates(self) -> None:
        """D31-03: 3 bets share a side, 2 of them a game -> 1/sqrt(3), NOT 1/sqrt(6).

        The two groupings are never COMPOSED. The larger group (3, the side)
        sets the factor for all three bets, including the two that also share a
        game -- otherwise a bet correlated on both axes would be penalised twice
        for one correlation.
        """
        bets = [
            {"game_id": "2025_W01_A@B", "bet_side": "under", "stake": 300.0},
            {"game_id": "2025_W01_A@B", "bet_side": "under", "stake": 300.0},
            {"game_id": "2025_W01_C@D", "bet_side": "under", "stake": 300.0},
        ]
        records = apply_same_game_and_side_deweight(bets)

        expected = 1.0 / math.sqrt(3)
        for record in records:
            assert record["same_side_group_size"] == 3
            assert record["scale_factor"] == pytest.approx(expected, abs=_TOL)
            assert record["binding_group"] == "same_side"
        assert [r["same_game_group_size"] for r in records] == [2, 2, 1]
        # NOT the composed 1/sqrt(2 * 3) = 1/sqrt(6).
        assert records[0]["scale_factor"] != pytest.approx(1.0 / math.sqrt(6), abs=_TOL)

    def test_shared_side_and_game_applies_once(self) -> None:
        """D31-03: two bets sharing BOTH a side and a game get 1/sqrt(2), never 1/2.

        The stricter grouping applies EXACTLY ONCE. Composing the two would give
        1/sqrt(2) * 1/sqrt(2) = 0.5 -- double-penalising one correlation.
        """
        bets = [
            {"game_id": "2025_W01_A@B", "bet_side": "under", "stake": 400.0},
            {"game_id": "2025_W01_A@B", "bet_side": "under", "stake": 400.0},
        ]
        records = apply_same_game_and_side_deweight(bets)

        for record in records:
            assert record["same_side_group_size"] == 2
            assert record["same_game_group_size"] == 2
            assert record["binding_group"] == "both"
            assert record["scale_factor"] == pytest.approx(1.0 / math.sqrt(2), abs=_TOL)
            assert record["scale_factor"] != pytest.approx(0.5, abs=_TOL)
            assert record["deweighted_stake"] == pytest.approx(
                400.0 / math.sqrt(2), abs=_TOL
            )

    def test_single_uncorrelated_bet_factor_is_exactly_one(self) -> None:
        """D31-03: one bet, one game, one side -> factor EXACTLY 1.0 (no de-weight)."""
        records = apply_same_game_and_side_deweight(
            [{"game_id": "2025_W01_A@B", "bet_side": "over", "stake": 250.0}]
        )
        assert records[0]["same_side_group_size"] == 1
        assert records[0]["same_game_group_size"] == 1
        assert records[0]["scale_factor"] == 1.0
        assert records[0]["deweighted_stake"] == pytest.approx(250.0, abs=_TOL)

    def test_six_disjoint_sides_on_one_game_group_by_game(self) -> None:
        """D31-03: a full three-target slate on ONE game is one group of 6, not six of 1.

        This is the degenerate case the same-side helper produced on a pooled
        week: six disjoint side strings -> six groups of one -> no de-weighting
        at all, on the most correlated week possible.
        """
        sides = ["home", "away", "home_cover", "away_cover", "over", "under"]
        bets = [
            {"game_id": "2025_W01_A@B", "bet_side": side, "stake": 100.0}
            for side in sides
        ]
        records = apply_same_game_and_side_deweight(bets)

        expected = 1.0 / math.sqrt(6)
        for record in records:
            assert record["same_side_group_size"] == 1
            assert record["same_game_group_size"] == 6
            assert record["binding_group"] == "same_game"
            assert record["scale_factor"] == pytest.approx(expected, abs=_TOL)

    def test_game_key_is_verbatim_and_side_key_stays_normalized(self) -> None:
        """D31-03: the two group keys are INDEPENDENT by construction.

        ``bet_side`` keeps its case-insensitive normalization (D27-11); the
        ``game_id`` key is used VERBATIM, because a game_id is an exact
        identifier and silently case-folding it would merge two distinct games.
        """
        bets = [
            {"game_id": "2025_W01_A@B", "bet_side": "Under", "stake": 100.0},
            {"game_id": "2025_w01_a@b", "bet_side": "UNDER", "stake": 100.0},
        ]
        records = apply_same_game_and_side_deweight(bets)

        for record in records:
            assert record["bet_side"] == "under"  # normalized
            assert record["same_side_group_size"] == 2  # sides merged
            assert record["same_game_group_size"] == 1  # game ids did NOT merge
            assert record["scale_factor"] == pytest.approx(1.0 / math.sqrt(2), abs=_TOL)

    def test_deweight_record_carries_game_id_and_group_metadata(self) -> None:
        """D31-03: the per-bet record carries both group sizes and which one bound.

        The pooled selector (Plan 31-06) renders these on the bet list, so a
        published stake can be explained without recomputing the rule.
        """
        bets = [
            {"game_id": "2025_W01_A@B", "bet_side": "home", "stake": 100.0},
            {"game_id": "2025_W01_A@B", "bet_side": "over", "stake": 100.0},
            {"game_id": "2025_W01_C@D", "bet_side": "over", "stake": 100.0},
        ]
        records = apply_same_game_and_side_deweight(bets)

        for record in records:
            for key in (
                "game_id",
                "bet_side",
                "original_stake",
                "deweighted_stake",
                "scale_factor",
                "same_side_group_size",
                "same_game_group_size",
                "binding_group",
            ):
                assert key in record
            assert record["binding_group"] in {"same_side", "same_game", "both"}
        assert [r["game_id"] for r in records] == [
            "2025_W01_A@B",
            "2025_W01_A@B",
            "2025_W01_C@D",
        ]


# ---------------------------------------------------------------------------
# (11) D31-03: the pipeline consumes the extension without a new call site
# ---------------------------------------------------------------------------


class TestPipelineCarriesSameGameGrouping:
    """``apply_sizing_pipeline`` is the single seam; it de-weights by game too."""

    def test_pipeline_deweights_a_cross_target_same_game_pair(self) -> None:
        """D31-03: the cross-target pair shrinks THROUGH the pipeline, not just the helper.

        bankroll 10000, two 200.0 stakes on one game with disjoint sides: the 5%
        per-bet cap (500) does not bind, de-weight 1/sqrt(2), and the summed
        282.8 is far under the 10% weekly cap, so the de-weight is the only
        binding step.
        """
        bankroll = 10000.0
        bets = [
            {"game_id": "2025_W01_A@B", "bet_side": "home", "stake": 200.0},
            {"game_id": "2025_W01_A@B", "bet_side": "under", "stake": 200.0},
        ]
        result = apply_sizing_pipeline(bets, bankroll)

        expected = 200.0 / math.sqrt(2)
        for record in result:
            assert record["deweighted_stake"] == pytest.approx(expected, abs=_TOL)
            assert record["weekly_scaled_stake"] == pytest.approx(expected, abs=_TOL)
            assert record["same_game_group_size"] == 2
            assert record["same_side_group_size"] == 1
            assert record["binding_group"] == "same_game"

    def test_pipeline_metadata_carries_group_sizes(self) -> None:
        """D31-03: the pipeline records carry the D31-03 fields alongside the P27 ones."""
        bankroll = 10000.0
        result = apply_sizing_pipeline(_bets("under", 300.0, 4), bankroll)
        for record in result:
            for key in (
                "game_id",
                "original_stake",
                "deweighted_stake",
                "weekly_scaled_stake",
                "scale_factor",
                "same_side_group_size",
                "same_game_group_size",
                "binding_group",
            ):
                assert key in record


# ---------------------------------------------------------------------------
# (12) D31-03: hand-built mixed weeks + the monotonicity property (T-31-14)
# ---------------------------------------------------------------------------


def _week(*bets: dict) -> list[dict]:
    """A hand-built pooled week (one dict per bet)."""
    return list(bets)


def _bet(game_id: str, bet_side: str, stake: float = 100.0) -> dict:
    """One pooled-week bet record."""
    return {"game_id": game_id, "bet_side": bet_side, "stake": stake}


# Twelve HAND-BUILT mixed weeks spanning the correlation shapes a pooled
# three-target week can take: no correlation, side-only, game-only, both, and
# the degenerate whole-slate-on-one-game case. Named so a failure names the
# week. Used by the monotonicity property and the documented-formula check.
_MIXED_WEEKS: tuple[tuple[str, list[dict]], ...] = (
    ("single bet", _week(_bet("G1", "home"))),
    (
        "two games, two disjoint sides (uncorrelated)",
        _week(_bet("G1", "home"), _bet("G2", "over")),
    ),
    (
        "one game, two disjoint sides (cross-target pair)",
        _week(_bet("G1", "home"), _bet("G1", "home_cover")),
    ),
    (
        "two games, same side",
        _week(_bet("G1", "under"), _bet("G2", "under")),
    ),
    (
        "one game, same side twice",
        _week(_bet("G1", "under"), _bet("G1", "under")),
    ),
    (
        "three same-side bets, two sharing a game",
        _week(_bet("G1", "under"), _bet("G1", "under"), _bet("G2", "under")),
    ),
    (
        "one game, three targets, three disjoint sides",
        _week(_bet("G1", "home"), _bet("G1", "away_cover"), _bet("G1", "under")),
    ),
    (
        "two games each carrying a cross-target pair",
        _week(
            _bet("G1", "home"),
            _bet("G1", "over"),
            _bet("G2", "away"),
            _bet("G2", "under"),
        ),
    ),
    (
        "a correlated trio plus two lone bets",
        _week(
            _bet("G1", "home"),
            _bet("G1", "home_cover"),
            _bet("G1", "over"),
            _bet("G2", "away"),
            _bet("G3", "under"),
        ),
    ),
    (
        "three unders and three overs, one over sharing a game with an under",
        _week(
            _bet("G1", "under"),
            _bet("G2", "under"),
            _bet("G3", "under"),
            _bet("G1", "over"),
            _bet("G4", "over"),
            _bet("G5", "over"),
        ),
    ),
    (
        "whole three-target slate on one game (six disjoint sides)",
        _week(
            _bet("G1", "home"),
            _bet("G1", "away"),
            _bet("G1", "home_cover"),
            _bet("G1", "away_cover"),
            _bet("G1", "over"),
            _bet("G1", "under"),
        ),
    ),
    (
        "mixed stakes, mixed case sides, mixed games",
        _week(
            _bet("G1", "Home", 250.0),
            _bet("G1", "HOME_COVER", 125.0),
            _bet("G2", "home", 75.0),
            _bet("G3", "Under", 400.0),
            _bet("G3", "over", 50.0),
            _bet("G4", "away_cover", 10.0),
        ),
    ),
)


class TestDeweightMonotonicity:
    """T-31-14: the extension can only SHRINK a de-weighted stake, never grow one."""

    def test_at_least_ten_hand_built_mixed_weeks(self) -> None:
        """The property is measured over at least 10 hand-built mixed weeks."""
        assert len(_MIXED_WEEKS) >= 10

    def test_new_factor_never_exceeds_same_side_only_factor(self) -> None:
        """T-31-14: for EVERY bet in EVERY mixed week, new factor <= same-side-only factor.

        The same-side-only factor is ``1/sqrt(same_side_group_size)`` -- the
        pre-D31-03 rule -- reconstructed from the group size the record itself
        reports. Because the new factor is ``1/sqrt(max(side, game))`` and
        ``max(side, game) >= side``, the inequality holds by construction; this
        MEASURES it rather than arguing it.
        """
        for name, week in _MIXED_WEEKS:
            for record in apply_same_game_and_side_deweight(week):
                same_side_only = 1.0 / math.sqrt(record["same_side_group_size"])
                assert record["scale_factor"] <= same_side_only + _TOL, name
                assert (
                    record["deweighted_stake"]
                    <= record["original_stake"] * same_side_only + _TOL
                ), name

    def test_monotonicity_is_not_vacuous(self) -> None:
        """T-31-14: at least one mixed week shrinks STRICTLY (the property has bite).

        A monotonicity assertion satisfied only by equality proves nothing, so
        the corpus is required to contain weeks the extension genuinely
        tightens.
        """
        tightened = [
            name
            for name, week in _MIXED_WEEKS
            for record in apply_same_game_and_side_deweight(week)
            if record["scale_factor"] < 1.0 / math.sqrt(record["same_side_group_size"])
        ]
        assert tightened, "no mixed week is tightened -- the corpus is vacuous"

    def test_pooled_weekly_exposure_never_rises(self) -> None:
        """T-31-14: total staked across a pooled week never rises through the pipeline.

        Per-bet monotonicity at the de-weight step plus the pro-rata weekly cap
        (whose scale is ``min(1, cap/sum)``) means the SUM of a week's final
        stakes is non-increasing under the extension. This is the claim that
        actually bounds risk and normalises the verdict -- see
        ``TestWeeklyCapRedistribution`` for the honest limit of the per-bet
        version of the same claim.
        """
        bankroll = 10000.0
        for name, week in _MIXED_WEEKS:
            new_total = sum(
                record["weekly_scaled_stake"]
                for record in apply_sizing_pipeline(week, bankroll)
            )
            same_side_only_deweighted = [
                min(float(bet["stake"]), bankroll * PER_BET_CAP_PCT)
                / math.sqrt(record["same_side_group_size"])
                for bet, record in zip(
                    week, apply_same_game_and_side_deweight(week), strict=True
                )
            ]
            old_total = sum(
                record["weekly_scaled_stake"]
                for record in apply_weekly_exposure_cap(
                    same_side_only_deweighted, bankroll
                )
            )
            assert new_total <= old_total + _TOL, name


class TestWeeklyCapRedistribution:
    """The honest limit of "no stake can rise": the weekly cap redistributes pro-rata.

    D31-03 states the extension is strictly conservative. That is EXACTLY true
    at the de-weight step (``TestDeweightMonotonicity``) and true for a week's
    TOTAL exposure. It is NOT true for every individual stake once the 10%
    weekly cap BINDS: the cap's pro-rata scale is ``min(1, cap/sum)``, so a
    smaller de-weighted sum divides under the cap more gently and an
    uncorrelated bet's share of the (unchanged, still-capped) 10% can RISE.
    This class pins that behaviour with a worked counterexample rather than
    leaving an over-broad claim unmeasured. The total -- what bounds risk and
    what a staked ROI is normalised by -- still never rises.
    """

    def test_uncorrelated_stake_can_rise_when_the_weekly_cap_binds(self) -> None:
        """A lone bet's stake RISES when de-weighting frees room under a binding cap.

        bankroll 10000 (weekly cap 1000, per-bet cap 500). Three bets of 500:
        A alone on G1; B and C sharing G2 with disjoint sides.

        Same-side-only: every side group is 1 -> no de-weight -> sum 1500 > 1000
          -> scale 2/3 -> every stake 333.33.
        Same-game-and-side: A unchanged at 500; B and C at 500/sqrt(2) = 353.55
          -> sum 1207.11 > 1000 -> scale 0.82843 -> A 414.21, B/C 292.89.

        A rose from 333.33 to 414.21. The TOTAL is 1000 either way.
        """
        bankroll = 10000.0
        week = [
            _bet("G1", "home", 500.0),
            _bet("G2", "over", 500.0),
            _bet("G2", "home_cover", 500.0),
        ]
        result = apply_sizing_pipeline(week, bankroll)

        same_side_only_stake = 500.0 * (1000.0 / (3 * 500.0))
        expected_lone = 500.0 * (1000.0 / (500.0 + 2 * 500.0 / math.sqrt(2)))

        assert result[0]["weekly_scaled_stake"] > same_side_only_stake
        assert result[0]["weekly_scaled_stake"] == pytest.approx(
            expected_lone, abs=_TOL
        )
        # The correlated pair still shrinks, and the TOTAL is unchanged at the cap.
        assert result[1]["weekly_scaled_stake"] < same_side_only_stake
        assert result[2]["weekly_scaled_stake"] < same_side_only_stake
        assert sum(record["weekly_scaled_stake"] for record in result) == pytest.approx(
            0.10 * bankroll, abs=_TOL
        )


# ---------------------------------------------------------------------------
# (13) D31-03 / T-31-17: no silent fallback when game_id is absent
# ---------------------------------------------------------------------------


class TestMissingGameIdRaises:
    """A bet without ``game_id`` is a hard error, never same-side-only grouping."""

    def test_missing_game_id_raises_named_key_error(self) -> None:
        """T-31-17: the de-weight helper raises a KeyError NAMING ``game_id``.

        Falling back to same-side-only grouping would silently restore exactly
        the blind spot D31-03 exists to close, on a pooled week, with no signal
        in the output. Follows the D27-07 no-silent-fallback pattern already
        used at the per-season bias lookup.
        """
        with pytest.raises(KeyError, match="game_id"):
            apply_same_game_and_side_deweight(
                [
                    {"game_id": "2025_W01_A@B", "bet_side": "home", "stake": 100.0},
                    {"bet_side": "home_cover", "stake": 100.0},
                ]
            )

    def test_missing_game_id_raises_through_the_pipeline(self) -> None:
        """T-31-17: the single seam the selector calls raises too (no back door)."""
        with pytest.raises(KeyError, match="game_id"):
            apply_sizing_pipeline([{"bet_side": "under", "stake": 100.0}], 10000.0)

    def test_none_game_id_raises_named_key_error(self) -> None:
        """T-31-17: a None ``game_id`` raises as well.

        A present-but-None key would group every unidentified bet TOGETHER -- a
        wrong grouping rather than a missing one, which is worse than the
        absent-key case because it produces a number.
        """
        with pytest.raises(KeyError, match="game_id"):
            apply_same_game_and_side_deweight(
                [{"game_id": None, "bet_side": "under", "stake": 100.0}]
            )

    def test_error_message_states_the_reason(self) -> None:
        """T-31-17: the raised message names the field AND why there is no fallback."""
        with pytest.raises(KeyError) as excinfo:
            apply_same_game_and_side_deweight([{"bet_side": "under", "stake": 100.0}])
        message = str(excinfo.value)
        assert "game_id" in message
        assert "same-game" in message


# ---------------------------------------------------------------------------
# (14) D31-03 / T-31-15: the documented formula string matches the code
# ---------------------------------------------------------------------------


class TestDeweightFormulaDocumentedInLockstep:
    """The formula strings and the implemented expression cannot drift apart."""

    def test_formula_strings_are_one_rule_under_two_legacy_names(self) -> None:
        """T-31-15: SAME_SIDE_DEWEIGHT and SAME_GAME_DEWEIGHT are the SAME string.

        There is ONE de-weighting rule taking TWO group sizes. D27-11 readouts
        quote it as ``SAME_SIDE_DEWEIGHT`` and D31-03 as ``SAME_GAME_DEWEIGHT``;
        holding one string under both names makes documenting the rule two ways
        impossible ("one registry, never two lists").
        """
        assert SAME_SIDE_DEWEIGHT == SAME_GAME_DEWEIGHT
        assert SAME_SIDE_DEWEIGHT == (
            "stake / sqrt(max(same_side_group_size, same_game_group_size))"
        )

    def test_exactly_one_deweight_formula_exists_in_the_module(self) -> None:
        """T-31-15: exactly one de-weighting function, and one sqrt, in the module.

        The retired ``apply_same_side_deweight`` name must be GONE, not kept as
        an alias -- two names for one rule is the failure mode this pins.
        """
        source = inspect.getsource(kelly_module)
        deweight_defs = re.findall(
            r"^def (\w*deweight\w*)\(", source, flags=re.MULTILINE
        )
        assert deweight_defs == ["apply_same_game_and_side_deweight"]
        assert not hasattr(kelly_module, "apply_same_side_deweight")
        assert source.count("math.sqrt(") == 1

    def test_documented_string_matches_the_implemented_expression(self) -> None:
        """T-31-15: the documented expression, transcribed, reproduces every stake.

        ``stake / sqrt(max(same_side_group_size, same_game_group_size))`` is
        evaluated directly from the group sizes each record reports and compared
        against the de-weighted stake the module produced, on every mixed week.
        A formula that changed while its documented string did not fails here.
        """
        for name, week in _MIXED_WEEKS:
            for record in apply_same_game_and_side_deweight(week):
                documented = record["original_stake"] / math.sqrt(
                    max(
                        record["same_side_group_size"],
                        record["same_game_group_size"],
                    )
                )
                assert record["deweighted_stake"] == pytest.approx(
                    documented, abs=_TOL
                ), name


# ---------------------------------------------------------------------------
# (15) D31-03 / T-31-16: de-weighting cannot change WHICH bets are selected,
#      nor the flat-stake headline
# ---------------------------------------------------------------------------

# The target label the O/U strategy carries. Plan 31-06 registers WP and ATS
# strategies behind the same facade and this set becomes multi-valued; the
# assertions below compare (game_id, target) PAIRS so they survive that change
# without being rewritten.
_SELECTOR_TARGET = "ou"

# The fixture SD and the (negative, over-biased -- D26-18) prior-season bias
# used by the selection-path fixtures below.
_FIXTURE_SD = 13.0
_FIXTURE_BIAS = {2025: -1.0}


def _make_ou_selector(ev_floor_t: float = 0.0):
    """Build the O/U BetSelector with the fixture SD/bias (import deferred)."""
    from backtest.bet_selector import BetSelector
    from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD

    return BetSelector(
        frozen_sd=_FIXTURE_SD,
        season_bias_by_season=_FIXTURE_BIAS,
        ev_floor_t=ev_floor_t,
        bankroll=10_000.0,
        high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
    )


def _ou_row(
    game_id: str,
    *,
    model_total: float,
    closing_total: float,
    actual_total: float,
    season: int = 2025,
    week: int = 1,
) -> dict:
    """One synthetic O/U candidate row."""
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "model_total": model_total,
        "closing_total": closing_total,
        "actual": actual_total,
    }


# A hand-built week carrying a genuine SAME-GAME pair: two candidates on
# 2025_W01_A@B taking OPPOSITE sides (an under and a high-total over, the line
# being above the 48.0 pre-hold boundary), plus one uncorrelated under on a
# second game. This is the shape D31-03 exists for. The O/U strategy does not
# itself emit two bets on one game -- Plan 31-06 produces the pair by pooling
# three targets -- so the week is hand-built here to exercise the pooled shape
# through the real admission and sizing code that exists today.
_SAME_GAME_WEEK = [
    _ou_row("2025_W01_A@B", model_total=38.0, closing_total=52.0, actual_total=40.0),
    _ou_row("2025_W01_A@B", model_total=66.0, closing_total=52.0, actual_total=40.0),
    _ou_row("2025_W02_C@D", model_total=36.0, closing_total=45.0, actual_total=50.0),
]


def _same_side_only_deweight(bets: list[dict]) -> list[dict]:
    """TEST-ONLY comparator: the PRE-D31-03 same-side-only de-weighting variant.

    Deliberately NOT a production code path -- production has exactly one
    formula. It delegates to the real helper for grouping and validation, then
    overrides the factor with ``1/sqrt(same_side_group_size)``, so the two
    variants differ in exactly the one respect under test.

    ``apply_same_game_and_side_deweight`` is referenced through the name bound
    at import time, so monkeypatching the module attribute does not recurse.
    """
    return [
        {
            **record,
            "scale_factor": 1.0 / math.sqrt(record["same_side_group_size"]),
            "deweighted_stake": record["original_stake"]
            / math.sqrt(record["same_side_group_size"]),
            "same_game_group_size": 1,
            "binding_group": "same_side",
        }
        for record in apply_same_game_and_side_deweight(bets)
    ]


def _select(candidates: list[dict]) -> list[dict]:
    """Run the real selection path and return the selected records."""
    return _make_ou_selector().select(candidates).selected


def _selected_pairs(selected: list[dict]) -> set[tuple[str, str]]:
    """The set of (game_id, target) pairs the selection path admitted and staked."""
    return {(record["game_id"], _SELECTOR_TARGET) for record in selected}


def _flat_stake_roi(selected: list[dict]) -> float:
    """Flat 1-unit ROI over the GRADED selected bets -- reads NO stake field.

    Flat staking risks one unit per bet, so profit per unit risked is
    ``sum(payout_or_minus_one) / n_graded``. Pushes (outcome None) are excluded
    exactly as ungraded bets are. Because no stake is consumed, no sizing rule
    can move this number -- which is the point of the test that calls it.
    """
    from backtest.ou_ev_chain import MINUS_110_PAYOUT

    graded = [record for record in selected if record["outcome"] is not None]
    if not graded:
        return 0.0
    profit = sum(MINUS_110_PAYOUT if record["outcome"] else -1.0 for record in graded)
    return profit / len(graded)


class TestDeweightCannotChangeTheSelectedSet:
    """T-31-16: admission structurally precedes sizing, so de-weighting is not a lever."""

    def test_admitted_set_identical_across_deweight_variants(self, monkeypatch) -> None:
        """T-31-16: the same (game_id, target) pairs are selected under BOTH variants.

        The selection path is run twice over one hand-built week -- once with
        the D31-03 same-game-and-side rule, once with the pre-D31-03
        same-side-only variant monkeypatched in at the module seam. Admission
        compares per-bet EV against the floor and never reads a stake, so the
        admitted set must be identical. Compared as a SET OF PAIRS, not as
        counts: two runs can agree on how many bets they took while disagreeing
        about which.
        """
        with_same_game = _selected_pairs(_select(_SAME_GAME_WEEK))

        monkeypatch.setattr(
            kelly_module,
            "apply_same_game_and_side_deweight",
            _same_side_only_deweight,
        )
        same_side_only = _selected_pairs(_select(_SAME_GAME_WEEK))

        assert with_same_game == same_side_only
        assert with_same_game == {
            ("2025_W01_A@B", _SELECTOR_TARGET),
            ("2025_W02_C@D", _SELECTOR_TARGET),
        }

    def test_the_two_variants_really_do_size_differently(self, monkeypatch) -> None:
        """T-31-16: the comparison above is NOT vacuous -- the variants stake differently.

        If both variants produced identical stakes, "the selected set is
        identical" would prove nothing. On this week the same-game pair is
        de-weighted by the new rule and not by the old one, so the staked
        amounts genuinely diverge while the selected set does not.
        """
        new_stakes = [record["kelly_stake"] for record in _select(_SAME_GAME_WEEK)]

        monkeypatch.setattr(
            kelly_module,
            "apply_same_game_and_side_deweight",
            _same_side_only_deweight,
        )
        old_stakes = [record["kelly_stake"] for record in _select(_SAME_GAME_WEEK)]

        assert new_stakes != old_stakes
        assert len(new_stakes) == len(old_stakes)

    def test_flat_stake_roi_is_bit_identical_across_variants(self, monkeypatch) -> None:
        """T-31-16: the flat-stake headline is EXACTLY equal under both variants.

        Flat staking does not consume the de-weight factor at all, so the
        headline ROI the 2025 verdict reports cannot move because the sizing
        rule was tightened. Asserted with ``==`` -- no tolerance -- because
        anything other than bit-identity would mean a stake leaked into a
        flat-stake number.
        """
        new_roi = _flat_stake_roi(_select(_SAME_GAME_WEEK))

        monkeypatch.setattr(
            kelly_module,
            "apply_same_game_and_side_deweight",
            _same_side_only_deweight,
        )
        old_roi = _flat_stake_roi(_select(_SAME_GAME_WEEK))

        assert new_roi == old_roi
        # Non-vacuity: the week really did grade some bets (a 0.0-vs-0.0
        # comparison over an empty graded set would assert nothing).
        assert new_roi != 0.0

    def test_ev_floor_admission_precedes_sizing_in_source(self) -> None:
        """T-31-16: the ordering is STRUCTURAL, not incidental, in the selector source.

        A behavioural test alone would keep passing if a future edit moved
        de-weighting ahead of the EV-floor admission but happened not to change
        the outcome on this fixture. This pins the order in the code: the
        EV-floor comparison comes first, then the raw Kelly stake, then the
        sizing pipeline -- and no de-weighted quantity is in scope before the
        pipeline runs.
        """
        from backtest.bet_selector import BetSelector

        source = inspect.getsource(BetSelector._admit_and_size_week)
        # Drop the docstring so its prose references do not shadow the code.
        body = source.split('"""', 2)[-1]

        floor_at = body.index("self.ev_floor_t")
        kelly_at = body.index("calculate_optimal_bet_size")
        sizing_at = body.index("apply_sizing_pipeline")
        assert floor_at < kelly_at < sizing_at

        before_sizing = body[:sizing_at]
        assert "deweighted_stake" not in before_sizing
        assert "weekly_scaled_stake" not in before_sizing
        assert "same_game_group_size" not in before_sizing
