"""Unit suite for honest-sizing exposure caps + same-side de-weighting (BET-03).

Covers the ADDITIVE helpers in ``utils/kelly_criterion.py`` that implement the
Phase 27 sizing discipline (D27-09/10/11):

  - 1 unit = 1% of bankroll (D27-09).
  - The existing 5%-of-bankroll per-bet cap is retained as a rarely-binding net.
  - Per-week 10%-of-bankroll pro-rata total-exposure cap (D27-10, NEW).
  - Same-week SAME-SIDE de-weighting using the FROZEN formula ``stake /
    sqrt(group_size)`` (D27-11, NEW; covariance/joint-Kelly path REJECTED).
  - Full-Kelly EXCLUDED -- a half-Kelly ceiling 0.5 hard-clamp at the end.
  - The LOCKED cap order:
    ``calibrated Kelly stake -> 5% per-bet cap -> same-side de-weight -> 10% weekly cap``.

Mirrors the ``tests/unit/test_betting_simulation.py`` class-grouped style:
exact-value asserts (no subjective language), requirement-ID docstrings.

Each test docstring cites BET-03 and the LOCKED D27-NN decision / review
tightening it proves:
  - D27-09: unit=1%, full-Kelly excluded (half-Kelly ceiling).
  - D27-10: per-week 10% pro-rata cap (preserve ratios, drop no bets).
  - D27-11: same-side de-weight via stake/sqrt(group_size); no covariance path.
  - Review #2: FROZEN de-weight formula + LOCKED cap order + metadata + grouping.
  - Review #2 (Gemini): released-room reuse (de-weight runs BEFORE the weekly cap).
  - Review #8: edge cases (empty/zero/NaN/invalid-bankroll/single/all-one-week).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math

import pytest

from utils.kelly_criterion import (
    CAP_ORDER,
    PER_BET_CAP_PCT,
    UNIT_PCT_OF_BANKROLL,
    WEEKLY_CAP_PCT,
    apply_same_side_deweight,
    apply_sizing_pipeline,
    apply_weekly_exposure_cap,
    unit_size,
)

# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

_TOL = 1e-9


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
        """Review #2: the LOCKED cap order is documented as a tuple."""
        assert CAP_ORDER == (
            "kelly_stake",
            "per_bet_5pct_cap",
            "same_side_deweight",
            "weekly_10pct_cap",
        )


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
        bets = [{"bet_side": "under", "stake": 800.0}]
        result = apply_sizing_pipeline(bets, bankroll)
        # After the 5% cap: 500. Single-bet de-weight factor 1/sqrt(1) = 1.0.
        # Sum 500 = 5% <= 10% weekly cap -> no-op.
        assert result[0]["weekly_scaled_stake"] == pytest.approx(500.0, abs=_TOL)

    def test_per_bet_cap_noop_when_below_five_percent(self) -> None:
        """BET-03: a raw stake under 5% is not touched by the per-bet cap."""
        bankroll = 10000.0
        bets = [{"bet_side": "under", "stake": 300.0}]
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
        bets = [{"bet_side": "under", "stake": 300.0} for _ in range(4)]
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
        bets = [{"bet_side": "over", "stake": 700.0} for _ in range(2)]
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
        """Review #2: apply_same_side_deweight returns per-bet metadata.

        Keys: original_stake, deweighted_stake, scale_factor; numerically
        consistent (deweighted == original * scale_factor).
        """
        bets = [{"bet_side": "under", "stake": 400.0} for _ in range(4)]
        deweighted = apply_same_side_deweight(bets)
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
        bets = [{"bet_side": "under", "stake": 300.0} for _ in range(4)]
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
            {"bet_side": "Under", "stake": 200.0},
            {"bet_side": "UNDER", "stake": 200.0},
            {"bet_side": "over", "stake": 200.0},
            {"bet_side": "OVER", "stake": 200.0},  # e.g. a high-total OVER pocket
        ]
        deweighted = apply_same_side_deweight(bets)
        expected = 1.0 / math.sqrt(2)
        for rec in deweighted:
            assert rec["scale_factor"] == pytest.approx(expected, abs=_TOL)
            assert rec["deweighted_stake"] == pytest.approx(200.0 * expected, abs=_TOL)

    def test_all_under_week_uses_group_size_four(self) -> None:
        """Review #2: an all-under week of N=4 de-weights by 1/sqrt(4)."""
        bets = [{"bet_side": "under", "stake": 200.0} for _ in range(4)]
        deweighted = apply_same_side_deweight(bets)
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
        bets = [{"bet_side": "under", "stake": 400.0} for _ in range(4)]
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
        bets = [{"bet_side": "under", "stake": 700.0} for _ in range(5)]
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
            [{"bet_side": "under", "stake": 5_000.0}], bankroll
        )
        assert result[0]["weekly_scaled_stake"] == pytest.approx(
            PER_BET_CAP_PCT * bankroll, abs=_TOL
        )

    def test_stake_below_every_cap_passes_through(self) -> None:
        """A lone stake below the 5% per-bet and 10% weekly caps is returned unchanged."""
        bankroll = 10_000.0
        result = apply_sizing_pipeline([{"bet_side": "over", "stake": 200.0}], bankroll)
        assert result[0]["weekly_scaled_stake"] == pytest.approx(200.0, abs=_TOL)


# ---------------------------------------------------------------------------
# (9) edge cases -- hard behavior
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases hard-behave: no divide-by-zero, named ValueErrors (review #8)."""

    def test_empty_bet_list_returns_empty(self) -> None:
        """Review #8: an empty bet list returns empty (no divide-by-zero)."""
        assert apply_sizing_pipeline([], 10000.0) == []
        assert apply_same_side_deweight([]) == []
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
            apply_same_side_deweight([{"bet_side": "under", "stake": float("nan")}])

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
        deweighted = apply_same_side_deweight([{"bet_side": "under", "stake": 250.0}])
        assert deweighted[0]["scale_factor"] == pytest.approx(1.0, abs=_TOL)
        assert deweighted[0]["deweighted_stake"] == pytest.approx(250.0, abs=_TOL)

    def test_all_bets_in_one_week_cap_applies_once(self) -> None:
        """Review #8: all bets in one week -> the weekly cap applies once.

        6 same-side bets at 400 (bankroll 10000): 5% cap no-bind; de-weight
        1/sqrt(6) -> 400/sqrt(6) each; sum 6 * 400/sqrt(6) = 400*sqrt(6) =
        979.8 < 1000 -> the single weekly-cap pass leaves them unscaled.
        """
        bankroll = 10000.0
        bets = [{"bet_side": "under", "stake": 400.0} for _ in range(6)]
        result = apply_sizing_pipeline(bets, bankroll)
        expected = 400.0 / math.sqrt(6)
        for rec in result:
            assert rec["weekly_scaled_stake"] == pytest.approx(expected, abs=_TOL)
