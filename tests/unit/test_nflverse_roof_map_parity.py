"""The two copies of the nflverse roof map must stay equal.

Phase 33, Plan 33-06 Task 2(e) (COLD-09, NF-05).

A RECORDED DEFERRAL, NOT AN OVERSIGHT
-------------------------------------
``scripts/ingest_weather.NFLVERSE_ROOF_MAP`` and
``scripts/ingest_games._NFLVERSE_ROOF_MAP`` are byte-identical dictionaries, which
violates the repo's single-source-constant convention. They are DELIBERATELY not
collapsed: CLAUDE.md forbids uninstructed refactors, and adding a second import edge
across two ingest scripts is not what a week-3 deadline is for.

What replaces the refactor is this module. A future divergence becomes a TEST FAILURE
rather than a silent one -- which matters here specifically, because ``dome -> indoor``
is the premise the D33-16 disagreement recorder is built on. If that mapping moved, the
recorder in ``tests/unit/test_venues_json_international.py`` would still pass while
meaning something else.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from scripts.ingest_games import _NFLVERSE_ROOF_MAP
from scripts.ingest_weather import NFLVERSE_ROOF_MAP


def test_the_two_copies_are_equal() -> None:
    assert NFLVERSE_ROOF_MAP == _NFLVERSE_ROOF_MAP, (
        "the two nflverse roof maps have diverged. They are duplicated by a RECORDED "
        "deferral (Plan 33-06 Task 2e), and this test is the whole mitigation for "
        "that duplication: either re-sync them or collapse them deliberately."
    )


def test_dome_maps_to_indoor() -> None:
    """The premise the D33-16 disagreement recorder rests on."""
    assert NFLVERSE_ROOF_MAP["dome"] == "indoor", (
        "dome no longer maps to indoor. The three feed disagreements (MEL00, PAR00, "
        "MUN01) were classified as disagreements BECAUSE dome becomes indoor becomes "
        "a weather skip. If the mapping moved, re-derive the disagreement set."
    )


def test_the_full_mapping_is_the_four_nflverse_values() -> None:
    assert NFLVERSE_ROOF_MAP == {
        "dome": "indoor",
        "closed": "indoor",
        "outdoors": "outdoor",
        "open": "retractable",
    }
