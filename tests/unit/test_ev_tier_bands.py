"""The EV tier bands and the non-finite refusal (Phase 31, plan 31-09; SPEC R7, D31-24).

The tier is ONE absolute, unit-consistent scale across all three targets. Per-bet EV is already
expected profit per unit staked, so absolute bands are the only option that preserves
commensurability -- which is the whole reason R7 exists. Both rejected alternatives are pinned
here by their consequences rather than by a comment:

  * QUANTILE bands from the tune split would make a tier a FITTED quantity that must be fenced
    from 2025, and would move a bet's label when OTHER bets are added.
    ``test_adding_a_high_ev_bet_does_not_move_the_low_ones`` is the refutation.
  * PER-TARGET absolute bands would make "high" mean something different for WP than for O/U.
    ``test_assign_ev_tier_takes_no_target_parameter`` is the refutation: the function has no
    target argument to differ on.

The band edges are read from ``EV_TIER_BANDS`` rather than transcribed, so a boundary that moves
fails here instead of quietly relabelling every bet on the page.

Run this module:  .venv/Scripts/python.exe -m pytest tests/unit/test_ev_tier_bands.py -q

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
import math
from itertools import pairwise
from typing import Any

import pytest

from backtest.bet_selector import BetSelector
from backtest.ev_chain_constants import (
    EV_TIER_BANDS,
    EV_TIER_HIGH,
    EV_TIER_LABELS,
    EV_TIER_LOW,
    EV_TIER_MEDIUM,
    assign_ev_tier,
)
from backtest.ou_ev_chain import EV_FLOOR_GRID, MINUS_110_PAYOUT, per_bet_ev

# The one test-local strategy in this suite, reused rather than re-declared: a second stub could
# drift from the first, and this one is already checked against the Protocol where it is defined.
from tests.unit.test_suppression_freshness import _MiniStrategy

_SEASON = 2023
_WEEK = 1
_FROZEN_SD = 13.0
_SEASON_BIAS = {_SEASON: -1.0}
_BANKROLL = 10_000.0

# The three target codes the page will carry. Named here so the cross-target assertions read as
# claims about wp/ats/ou rather than about anonymous stubs.
_TARGET_CODES = ("wp", "ats", "ou")

_MARKET_FIELDS = ("model_edge", "closing_line")
_PREDICTION_FIELDS = ("model_edge",)


def _resolved_bands(ev_floor_t: float) -> list[tuple[str, float, float]]:
    """``EV_TIER_BANDS`` with the LOW band's sentinel lower bound resolved to the floor."""
    return [
        (label, ev_floor_t if lo is None else lo, hi) for label, lo, hi in EV_TIER_BANDS
    ]


def _p_for_ev(target_ev: float) -> float:
    """The calibrated P(side) that produces ``target_ev`` at the flat -110 payout.

    Inverted from ``per_bet_ev`` rather than hand-tabulated, so a change to the payout constant
    moves these fixtures with it instead of silently retargeting every band.
    """
    return (target_ev + 1.0) / (1.0 + MINUS_110_PAYOUT)


def _strategy(target: str, p_side: float) -> _MiniStrategy:
    return _MiniStrategy(target, _MARKET_FIELDS, _PREDICTION_FIELDS, p_side=p_side)


def _row(game_id: str, target: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "game_id": game_id,
        "season": _SEASON,
        "week": _WEEK,
        "target": target,
        "model_edge": 1.0,
        "closing_line": 45.0,
        "sportsbook": "consensus",
        "is_live": False,
    }
    row.update(overrides)
    return row


def _selector(strategies: list[Any], ev_floor_t: float = 0.0) -> BetSelector:
    return BetSelector(
        frozen_sd=_FROZEN_SD,
        season_bias_by_season=_SEASON_BIAS,
        ev_floor_t=ev_floor_t,
        bankroll=_BANKROLL,
        strategies=strategies,
    )


# ---------------------------------------------------------------------------
# The band contract itself
# ---------------------------------------------------------------------------


class TestBandContract:
    """The bands are three ordered, contiguous, half-open ranges on ONE absolute scale."""

    def test_labels_are_the_three_pre_registered_ones(self) -> None:
        assert EV_TIER_LABELS == (EV_TIER_LOW, EV_TIER_MEDIUM, EV_TIER_HIGH)
        assert EV_TIER_LABELS == ("low", "medium", "high")

    def test_bands_are_contiguous_and_ascending(self) -> None:
        """No gap and no overlap: every EV at or above the floor has exactly one band."""
        bands = _resolved_bands(0.0)
        for (_label, _lo, hi), (_next_label, next_lo, _next_hi) in pairwise(bands):
            assert hi == next_lo, "the bands are not contiguous"
        assert bands[0][1] == 0.0
        assert bands[-1][2] == math.inf

    def test_the_band_edges_are_frozen_ev_floor_grid_values(self) -> None:
        """D31-24 aligns the edges to the already-frozen grid rather than inventing thresholds."""
        interior_edges = [lo for _label, lo, _hi in _resolved_bands(0.0)][1:]
        assert interior_edges == [0.03, 0.05]
        for edge in interior_edges:
            assert edge in EV_FLOOR_GRID, (
                f"band edge {edge} is not an EV_FLOOR_GRID value, so the tier scale and the "
                "admission grid have drifted apart"
            )


# ---------------------------------------------------------------------------
# The adjacency edge: a boundary value falls in the HIGHER band
# ---------------------------------------------------------------------------


class TestBoundaryAdjacency:
    """SPEC R7 adjacency: bands are ``[lo, hi)``, so a boundary value is in the HIGHER band."""

    @pytest.mark.parametrize("ev_floor_t", EV_FLOOR_GRID)
    def test_every_lower_bound_lands_in_its_own_band(self, ev_floor_t: float) -> None:
        """Asserted directly ON each bound, at every pre-registered floor."""
        asserted = 0
        for label, lo, hi in _resolved_bands(ev_floor_t):
            if not (ev_floor_t <= lo < hi):
                # A floor at or above a band collapses that band out of existence, and a bound
                # below the floor belongs to no admitted row. At t=0.05 the low and medium bands
                # are empty; there is no value to assert on.
                continue
            assert assign_ev_tier(lo, ev_floor_t) == label
            asserted += 1
        assert asserted, (
            f"no band survives the floor {ev_floor_t}; the loop asserted nothing"
        )

    def test_every_upper_bound_lands_in_the_next_band_up(self) -> None:
        """0.03 is medium (not low) and 0.05 is high (not medium) -- stated, not sampled near."""
        bands = _resolved_bands(0.0)
        for (_label, _lo, hi), (next_label, _next_lo, _next_hi) in pairwise(bands):
            assert assign_ev_tier(hi, 0.0) == next_label

    def test_one_ulp_below_a_boundary_stays_in_the_lower_band(self) -> None:
        """The non-vacuity control: the assertion above sits ON the edge, not comfortably past it.

        Without this, an implementation using ``<=`` on the upper bound would also pass the
        boundary test by putting the value in BOTH bands' reach and returning the first match.
        """
        bands = _resolved_bands(0.0)
        for (label, _lo, hi), _next in pairwise(bands):
            assert assign_ev_tier(math.nextafter(hi, -math.inf), 0.0) == label

    @pytest.mark.parametrize("ev_floor_t", EV_FLOOR_GRID)
    def test_the_low_band_lower_bound_is_the_floor(self, ev_floor_t: float) -> None:
        """An EV exactly AT the floor is admitted (SPEC R1) and tiered, never left unlabelled.

        The expected label is derived from the BANDS rather than from the function under test, so
        this is a comparison against the contract and not against itself. At a floor of 0.03 the
        low band is empty and the floor value is medium; at 0.05 it is high.
        """
        expected = next(
            label
            for label, lo, hi in _resolved_bands(ev_floor_t)
            if lo <= ev_floor_t < hi
        )
        assert assign_ev_tier(ev_floor_t, ev_floor_t) == expected

    def test_a_value_below_the_floor_has_no_band(self) -> None:
        """An unadmitted row is rejected as ev_below_floor; manufacturing a tier would misreport."""
        with pytest.raises(ValueError, match="below the EV floor"):
            assign_ev_tier(math.nextafter(0.02, -math.inf), 0.02)


# ---------------------------------------------------------------------------
# One absolute scale for all three targets
# ---------------------------------------------------------------------------


class TestOneScaleAcrossTargets:
    """D31-24: a tier label means the same thing everywhere on the page."""

    def test_assign_ev_tier_takes_no_target_parameter(self) -> None:
        """The structural refutation of per-target bands: there is no target to differ on."""
        params = list(inspect.signature(assign_ev_tier).parameters)
        assert params == ["per_bet_ev", "ev_floor_t"]

    @pytest.mark.parametrize("value", [0.0, 0.0299, 0.03, 0.0499, 0.05, 0.5])
    def test_the_same_value_is_the_same_band_for_every_target_code(
        self, value: float
    ) -> None:
        """Stated across wp/ats/ou so the claim is about the page, not only about a function."""
        labels = {code: assign_ev_tier(value, 0.0) for code in _TARGET_CODES}
        assert len(set(labels.values())) == 1

    def test_three_targets_with_equal_ev_receive_equal_tiers_through_the_selector(
        self,
    ) -> None:
        """The same claim end to end: equal per-bet EV out of the selector, equal tier."""
        p_side = _p_for_ev(0.04)
        selector = _selector([_strategy(code, p_side) for code in _TARGET_CODES])
        rows = [
            _row(f"2023_W01_A{i}@B{i}", code) for i, code in enumerate(_TARGET_CODES)
        ]

        selected = selector.select(rows).selected
        assert len(selected) == len(_TARGET_CODES)

        evs = {record["per_bet_ev"] for record in selected}
        assert len(evs) == 1, (
            "the fixture did not produce equal EVs; it would prove nothing"
        )

        tiers = {assign_ev_tier(record["per_bet_ev"], 0.0) for record in selected}
        assert tiers == {EV_TIER_MEDIUM}


# ---------------------------------------------------------------------------
# No rescaling: the band is absolute, not relative to the week
# ---------------------------------------------------------------------------

_LOW_BAND_EVS = (0.005, 0.015, 0.025)
_HIGH_BAND_EV = 0.08


def _week_tiers(evs: tuple[float, ...]) -> dict[str, tuple[float, str]]:
    """Run one week whose bets carry ``evs`` and return {game_id: (per_bet_ev, tier)}."""
    strategies = [_strategy(f"t{i}", _p_for_ev(ev)) for i, ev in enumerate(evs)]
    selector = _selector(strategies)
    rows = [_row(f"2023_W01_A{i}@B{i}", f"t{i}") for i in range(len(evs))]

    selected = selector.select(rows).selected
    assert len(selected) == len(evs)
    return {
        record["game_id"]: (
            record["per_bet_ev"],
            assign_ev_tier(record["per_bet_ev"], 0.0),
        )
        for record in selected
    }


class TestNoRescaling:
    """A target whose bets all land in one band is reported as such -- honest reporting."""

    def test_a_week_entirely_in_one_band_reports_that_band_for_every_bet(self) -> None:
        """If every bet has near-zero EV, "all low" is the correct description, not a failure."""
        tiers = _week_tiers(_LOW_BAND_EVS)
        assert {tier for _ev, tier in tiers.values()} == {EV_TIER_LOW}
        for expected_ev, (actual_ev, _tier) in zip(
            _LOW_BAND_EVS, tiers.values(), strict=True
        ):
            assert actual_ev == pytest.approx(expected_ev)

    def test_adding_a_high_ev_bet_does_not_move_the_low_ones(self) -> None:
        """The refutation of quantile bands: a label depends on the VALUE, never on the company.

        Under tune-split quantile bands the top third of this four-bet week would relabel as the
        high tier the moment a genuinely high-EV bet joined it.
        """
        alone = _week_tiers(_LOW_BAND_EVS)
        with_high = _week_tiers((*_LOW_BAND_EVS, _HIGH_BAND_EV))

        for game_id, (ev, tier) in alone.items():
            assert with_high[game_id] == (ev, tier), (
                f"{game_id} changed band when another bet joined the week"
            )
        assert with_high["2023_W01_A3@B3"][1] == EV_TIER_HIGH


# ---------------------------------------------------------------------------
# The non-finite refusal
# ---------------------------------------------------------------------------


class TestNonFiniteIsRefusedNotLabelled:
    """T-31-43: suppression precedes tiering, and the tier helper refuses non-finite input."""

    @pytest.mark.parametrize(
        "value", [math.inf, -math.inf, math.nan, float("nan"), float("inf")]
    )
    def test_assign_ev_tier_raises_on_a_non_finite_value(self, value: float) -> None:
        """Raising rather than returning a label is what makes the badge's no-null-state real."""
        with pytest.raises(ValueError, match="not finite"):
            assign_ev_tier(value, 0.0)

    @pytest.mark.parametrize("floor", [math.inf, -math.inf, math.nan])
    def test_assign_ev_tier_raises_on_a_non_finite_floor(self, floor: float) -> None:
        """A floor that did not resolve cannot silently become the LOW band's lower bound."""
        with pytest.raises(ValueError, match="not finite"):
            assign_ev_tier(0.04, floor)

    def test_a_selector_run_suppresses_a_non_finite_ev_and_sets_no_tier(self) -> None:
        """The two halves together: the value never reaches the badge, and could not be tiered.

        The suppression is what makes the claim structural rather than a template guard -- and
        the raise below shows what would have happened had the row got through.
        """
        selector = _selector([_strategy("ou", float("nan"))])
        result = selector.select([_row("2023_W01_A0@B0", "ou")])

        assert result.selected == []
        assert len(result.rejected) == 1
        record = result.rejected[0]
        assert record["rejection_reason"] == "ev_not_finite"
        assert "ev_tier" not in record
        assert record.get("ev_tier") is None

        assert not math.isfinite(record["per_bet_ev"])
        with pytest.raises(ValueError, match="not finite"):
            assign_ev_tier(record["per_bet_ev"], 0.0)

    def test_the_non_finite_ev_would_otherwise_have_cleared_the_floor(self) -> None:
        """Why the branch must precede the floor: ``NaN < floor`` is False, so it does not reject.

        Stated on the ACTUAL computed EV rather than on a NaN literal, because that is the value
        the floor comparison would have seen.
        """
        nan_ev = per_bet_ev(float("nan"), MINUS_110_PAYOUT)
        assert math.isnan(nan_ev)
        assert not (nan_ev < 0.0), "a NaN EV does not compare below the floor"
        assert not (nan_ev >= 0.0), "and it does not compare at or above it either"
