"""Byte-reproducible chain output, per target (Phase 31, plan 31-07; SPEC R1, T-31-32).

WHY STRINGS AND NOT A FLOAT TOLERANCE
--------------------------------------
The claim SPEC R1 makes is byte-identity, not near-agreement. A comparison with a tolerance
would pass on a run that differed in the last bit -- and a last-bit difference is exactly the
signature of the non-determinism worth catching (a set iterated instead of a list, a dict
ordered by insertion in one pass and by rehash in another, a partially-ordered sort). So both
runs are rendered through ONE 17-significant-digit float format and the resulting STRINGS are
compared. Seventeen digits is the shortest format that round-trips an IEEE-754 double, and it
is the same ``{:.17g}`` this phase already records in
``tests/phase31_state.ATS_RESIDUAL_BY_SEASON_PROVENANCE``.

THE ORDER IS PART OF THE OUTPUT
--------------------------------
Order is asserted as ``game_id`` then ``target``, both ascending -- the canonical publication
order, and one implementation of it (``backtest.ats_ev_chain.chain_order_key``) shared by the
chains and by this module. Two runs that carried the same bets in different orders would
serialise differently, so ordering and value determinism are one assertion rather than two.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from typing import Any

from backtest.ats_ev_chain import (
    THRESHOLD_WINDOW_P31,
    ChainFit,
    chain_order_key,
    price_ats_candidates,
)
from backtest.bet_selector import BetSelector
from backtest.ou_divergence import HIGH_TOTAL_BOUNDARY_PREHOLD
from backtest.selector_strategies import OUStrategy
from backtest.wp_ev_chain import price_wp_candidates

# The single float format both runs are rendered through. Seventeen significant digits
# round-trip an IEEE-754 double exactly, so equal strings mean equal doubles.
FLOAT_FORMAT: str = "{:.17g}"

_OU_FROZEN_SD: float = 5.0
_OU_SEASON_BIAS: dict[int, float] = {2025: -1.0}


def _render(value: Any) -> str:
    """Render one field. Floats through :data:`FLOAT_FORMAT`; everything else by repr.

    ``bool`` is checked BEFORE the numeric branch, because ``isinstance(True, int)`` is True
    in Python and a bool rendered as ``1`` would make a True/False difference invisible in
    the very comparison this module exists to make.
    """
    if isinstance(value, bool) or value is None:
        return repr(value)
    if isinstance(value, float):
        return FLOAT_FORMAT.format(value)
    if isinstance(value, int):
        return repr(value)
    return repr(value)


def _serialize(records: Sequence[Mapping[str, Any]]) -> str:
    """Render a whole record sequence to ONE string, order included.

    Keys are emitted in sorted order so a dict whose insertion order changed but whose
    CONTENT did not is not reported as a difference -- the claim under test is about values
    and record order, not about the internal key order of a Python dict.
    """
    lines: list[str] = []
    for index, record in enumerate(records):
        fields = "|".join(f"{key}={_render(record[key])}" for key in sorted(record))
        lines.append(f"{index}#{fields}")
    return "\n".join(lines)


def _assert_canonically_ordered(records: Sequence[Mapping[str, Any]]) -> None:
    """The records are ascending by ``(game_id, target)`` under the ONE ordering helper."""
    keys = [chain_order_key(record) for record in records]
    for earlier, later in itertools.pairwise(keys):
        assert earlier <= later, (
            f"chain output is not in the canonical publication order: {earlier} precedes "
            f"{later}. The order is part of the output that the byte-identity claim covers."
        )


def _ats_fit() -> ChainFit:
    """A clean ATS fit. Constructed fresh per run so no state is shared between them."""
    return ChainFit(
        target="ats",
        frozen_sd=11.0,
        season_bias_by_season={2025: 0.5},
        tune_fit_seasons=(2021, 2022, 2023, 2024),
        threshold_window=THRESHOLD_WINDOW_P31,
        bias_pool_by_season={2024: (2021, 2022, 2023)},
    )


def _wp_fit() -> ChainFit:
    """A clean WP fit. ``frozen_sd`` is None BY DESIGN (D31-07)."""
    return ChainFit(
        target="wp",
        frozen_sd=None,
        season_bias_by_season={2025: 0.02},
        tune_fit_seasons=(2021, 2022, 2023, 2024),
        threshold_window=THRESHOLD_WINDOW_P31,
        bias_pool_by_season={2024: (2021, 2022, 2023)},
    )


def _ou_selector() -> BetSelector:
    """A fresh selector per run, so equality cannot come from a shared object."""
    return BetSelector(
        frozen_sd=_OU_FROZEN_SD,
        season_bias_by_season=_OU_SEASON_BIAS,
        ev_floor_t=0.0,
        bankroll=10_000.0,
        high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
        strategies=[
            OUStrategy(
                frozen_sd=_OU_FROZEN_SD,
                season_bias_by_season=_OU_SEASON_BIAS,
                high_total_boundary=HIGH_TOTAL_BOUNDARY_PREHOLD,
            )
        ],
    )


# Candidate frames deliberately supplied OUT of canonical order, so a chain that merely
# preserved input order would fail the ordering assertion rather than pass it by accident.
ATS_ROWS: tuple[dict[str, Any], ...] = (
    {
        "game_id": "2025_01_ZZZ",
        "season": 2025,
        "week": 1,
        "model_spread": 20.0,
        "closing_spread": -3.0,
    },
    {
        "game_id": "2025_01_AAA",
        "season": 2025,
        "week": 1,
        "model_spread": -20.0,
        "closing_spread": -3.0,
    },
    {
        "game_id": "2025_01_MMM",
        "season": 2025,
        "week": 1,
        "model_spread": 3.0,
        "closing_spread": -3.0,
    },
)

WP_ROWS: tuple[dict[str, Any], ...] = (
    {
        "game_id": "2025_01_ZZZ",
        "season": 2025,
        "week": 1,
        "model_prob": 0.81,
        "ml_home": -150,
        "ml_away": 130,
    },
    {
        "game_id": "2025_01_AAA",
        "season": 2025,
        "week": 1,
        "model_prob": 0.19,
        "ml_home": 145,
        "ml_away": -165,
    },
    {
        "game_id": "2025_01_MMM",
        "season": 2025,
        "week": 1,
        "model_prob": 0.50,
        "ml_home": -110,
        "ml_away": -110,
    },
)

OU_ROWS: tuple[dict[str, Any], ...] = (
    {
        "game_id": "2025_01_ZZZ",
        "season": 2025,
        "week": 1,
        "model_total": 30.0,
        "closing_total": 45.0,
        "actual": 20.0,
    },
    {
        "game_id": "2025_01_AAA",
        "season": 2025,
        "week": 1,
        "model_total": 31.5,
        "closing_total": 44.0,
        "actual": 51.0,
    },
    {
        "game_id": "2025_02_MMM",
        "season": 2025,
        "week": 2,
        "model_total": 33.0,
        "closing_total": 46.5,
        "actual": 38.0,
    },
)


class TestAtsDeterminism:
    """Two ATS runs over one frame are byte-identical and canonically ordered."""

    def test_ats_two_runs_serialize_to_the_same_string(self) -> None:
        """T-31-32: string equality under one 17-digit format, no float tolerance."""
        first = price_ats_candidates(ATS_ROWS, _ats_fit()).records
        second = price_ats_candidates(ATS_ROWS, _ats_fit()).records

        assert _serialize(first) == _serialize(second)
        # Non-vacuity: an empty pair of runs would serialise to two equal empty strings.
        assert len(first) == len(ATS_ROWS)
        assert _serialize(first)

    def test_ats_output_is_ordered_by_game_id_then_target(self) -> None:
        """The chain REORDERS its out-of-order input into the publication order."""
        records = price_ats_candidates(ATS_ROWS, _ats_fit()).records

        _assert_canonically_ordered(records)
        assert [record["game_id"] for record in records] == [
            "2025_01_AAA",
            "2025_01_MMM",
            "2025_01_ZZZ",
        ]


class TestWpDeterminism:
    """Two WP runs over one frame are byte-identical and canonically ordered."""

    def test_wp_two_runs_serialize_to_the_same_string(self) -> None:
        """T-31-32: string equality under one 17-digit format, no float tolerance."""
        first = price_wp_candidates(WP_ROWS, _wp_fit()).records
        second = price_wp_candidates(WP_ROWS, _wp_fit()).records

        assert _serialize(first) == _serialize(second)
        assert len(first) == len(WP_ROWS)
        assert _serialize(first)

    def test_wp_output_is_ordered_by_game_id_then_target(self) -> None:
        """The chain REORDERS its out-of-order input into the publication order."""
        records = price_wp_candidates(WP_ROWS, _wp_fit()).records

        _assert_canonically_ordered(records)
        assert [record["game_id"] for record in records] == [
            "2025_01_AAA",
            "2025_01_MMM",
            "2025_01_ZZZ",
        ]


class TestOuDeterminism:
    """Two O/U runs through the REAL selector are byte-identical once ordered.

    ``BetSelector`` groups by POOLED week before sizing (D31-02), so its own emission order
    is week-grouped insertion order rather than the publication order. That order is itself
    deterministic -- both runs below produce it identically -- and the canonical publication
    order is applied by the consumer through the same one helper the other two arms use.
    """

    def test_ou_two_runs_serialize_to_the_same_string(self) -> None:
        """T-31-32: string equality under one 17-digit format, no float tolerance."""
        first = sorted(_ou_selector().select(OU_ROWS).selected, key=chain_order_key)
        second = sorted(_ou_selector().select(OU_ROWS).selected, key=chain_order_key)

        assert _serialize(first) == _serialize(second)
        assert len(first) >= 1
        assert _serialize(first)

    def test_ou_selector_emission_order_is_itself_stable(self) -> None:
        """The raw pre-sort order agrees between runs, so the sort is not hiding a wobble."""
        first = _ou_selector().select(OU_ROWS).selected
        second = _ou_selector().select(OU_ROWS).selected

        assert [record["game_id"] for record in first] == [
            record["game_id"] for record in second
        ]

    def test_ou_output_is_ordered_by_game_id_then_target_after_the_shared_helper(
        self,
    ) -> None:
        """The publication order is the same rule for all three targets, applied once."""
        records = sorted(_ou_selector().select(OU_ROWS).selected, key=chain_order_key)

        _assert_canonically_ordered(records)
        assert {record["target"] for record in records} == {"ou"}


class TestTheSerializerCannotPassVacuously:
    """The comparison must be able to FAIL, or its passes mean nothing."""

    def test_a_last_bit_float_difference_is_reported(self) -> None:
        """A tolerance-based comparison would miss this; the 17-digit render does not."""
        left = [{"game_id": "g", "target": "ats", "per_bet_ev": 0.1}]
        right = [{"game_id": "g", "target": "ats", "per_bet_ev": 0.1 + 5e-17}]

        assert left[0]["per_bet_ev"] != right[0]["per_bet_ev"]
        assert _serialize(left) != _serialize(right)

    def test_a_reordering_is_reported(self) -> None:
        """Order is part of the output, so a permutation is a difference."""
        records = [
            {"game_id": "a", "target": "ats", "per_bet_ev": 0.1},
            {"game_id": "b", "target": "ats", "per_bet_ev": 0.2},
        ]
        assert _serialize(records) != _serialize(list(reversed(records)))

    def test_a_bool_is_not_rendered_as_an_integer(self) -> None:
        """``isinstance(True, int)`` is True in Python; a bool must still render as a bool."""
        assert _serialize([{"fallback_fired": True}]) != _serialize(
            [{"fallback_fired": 1}]
        )
