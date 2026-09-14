"""Gold-width tripwire parity test (IN-03, Plan 28-06 / 29-06 / 30-07).

``scripts.data_qa.GOLD_FEATURE_MATRICES`` hardcodes the expected column count of each
gold feature matrix as a deliberate audit tripwire: any stray/dropped column trips it
rather than passing silently. This test asserts the tripwire is itself HONEST -- the
declared expected width equals the real on-disk column count of each rebuilt matrix.

If a future builder legitimately adds or removes a column, this test fails until the
operator UPDATES ``GOLD_FEATURE_MATRICES`` to the new empirically-counted width
(Pitfall 6 -- never silence the tripwire by widening tolerances).

The second test is the reason the first is not enough. A width tripwire on its own
says only that the total moved by N; it cannot say WHICH N columns. Phase 29 pinned
its +15 widening to a named family here; Phase 30 rung 3 (SPEC R3, D29-07-01) removes
exactly that family again, so the same fifteen names now pin a NARROWING and the
tripwire returns to the Phase-28 widths. The direction inverted; the discipline did
not.

The named list is cross-checked against ``backtest.signal_lift.group_columns`` -- the
ONE group registry the screen, the drop and the rung-3 attribution all read (D30-02)
-- so a hand-written list here cannot silently drift away from the predicate that
actually did the removing.

PHASE 33.1 (Plan 33.1-07) adds ONE column on top of the Phase-28 width: the weather
coverage flag. The discipline is unchanged and the direction inverted again -- a
WIDENING this time -- so the Phase-30 residual assertion moved from ``== 0`` to the
length of a NAMED tuple, with the reason recorded in that test's docstring rather
than the assertion simply being relaxed. ``PHASE_28_WIDTHS`` and
``PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS`` are left exactly as they are: both are
historical records of what earlier phases did, and a historical record that gets
edited every time the present changes is not a record.
"""

from pathlib import Path

import pandas as pd
import pytest

from backtest.signal_lift import group_columns
from features.weather import WEATHER_COVERAGE_COLUMN
from scripts.data_qa import GOLD_FEATURE_MATRICES

GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"

# The Phase-29 line-movement family (Plan 29-06 / SIG-04): the seven D-09 totals
# features, the shared coverage flag, and the seven Tier (a) spread siblings -- 15
# game-level columns per matrix. Phase 29 added them; Plan 30-07 removes them.
PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS = (
    "opening_total",
    "total_drift",
    "total_drift_dir",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
    "line_movement_coverage",
    "opening_spread",
    "spread_drift",
    "spread_drift_dir",
    "spread_late_drift",
    "spread_abs_travel",
    "spread_reversals",
    "spread_range",
)

# The Phase-28 widths Phase 29 widened FROM and Phase 30 rung 3 returns TO.
PHASE_28_WIDTHS = {"features_wp": 194, "features_ats": 195, "features_ou": 194}

# The ONE column Phase 33.1 adds on top of the Phase-28 width (Plan 33.1-07).
#
# NAMED, NOT COUNTED. The Phase-30 residual assertion below used to read
# `== 0`, which was exactly right while the tripwire had returned to the
# Phase-28 width and nothing else had moved it. Phase 33.1's rung 1 added the
# weather coverage flag, so the residual is now +1 BY DECISION rather than by
# accident -- and the difference between those two is the whole reason this
# tuple exists instead of the integer 1.
#
# The flag is imported from the ONE module that owns the name rather than
# spelled here, so a rename cannot leave this file pinning a column that no
# longer exists while the integer still matches.
PHASE_331_ADDED_WEATHER_COLUMNS = (WEATHER_COVERAGE_COLUMN,)

# The market columns that must SURVIVE the drop. ``snapshot_total`` /
# ``snapshot_spread`` are the freeze anchors and ``total_movement`` /
# ``spread_movement`` are the pre-existing MarketAnchor columns; all four are
# baseline features, and all four are near-misses for a careless substring prune.
MARKET_SURVIVORS = (
    "snapshot_total",
    "snapshot_spread",
    "total_movement",
    "spread_movement",
)


@pytest.mark.parametrize(
    ("table_name", "expected_width"), list(GOLD_FEATURE_MATRICES.items())
)
def test_gold_matrix_width_matches_tripwire(
    table_name: str, expected_width: int
) -> None:
    """Each rebuilt gold matrix's real column count equals its tripwire value."""
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    actual_width = pd.read_parquet(path).shape[1]
    assert actual_width == expected_width, (
        f"{table_name}: real column count {actual_width} != "
        f"GOLD_FEATURE_MATRICES expected {expected_width}. If the change is "
        f"intentional, update GOLD_FEATURE_MATRICES in scripts/data_qa.py to the "
        f"new empirically-counted width (IN-03, never silence the tripwire)."
    )


def test_the_named_removed_set_agrees_with_the_group_registry() -> None:
    """The hand-written list above IS the registry's line_movement family (D30-02).

    Without this, the list here and the predicate that performed the drop could
    drift apart and the delta assertion below would be pinning a fiction. Checked
    against a header-only frame so it asserts about the NAMES, not about gold.
    """
    frame = pd.DataFrame(
        columns=pd.Index([*PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS, *MARKET_SURVIVORS])
    )

    assert set(group_columns(frame, "line_movement")) == set(
        PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS
    ), (
        "the named removed set and backtest.signal_lift's line_movement predicate "
        "disagree -- one of them has drifted, and the delta assertion below would "
        "be pinning a set nothing actually removed"
    )
    assert len(PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS) == 15


@pytest.mark.parametrize("table_name", list(GOLD_FEATURE_MATRICES))
def test_phase_30_narrowing_is_exactly_the_line_movement_family(
    table_name: str,
) -> None:
    """The Phase-30 width delta is accounted for, column by column.

    The width tripwire on its own only says the total moved by 15; it cannot say
    the 15 are the intended columns. This pins the delta to the NAMED
    line-movement family, so a build that removed an unrelated column while
    incidentally leaving a line-movement one behind would fail here even though
    the integer still matched.

    Both halves are asserted: the fifteen are ABSENT from the rebuilt matrix, and
    the tripwire sits at the Phase-28 width plus the columns Phase 33.1 added.

    THE RESIDUAL WAS ZERO AND IS NOW +1, UPDATED WITH A REASON RATHER THAN
    SILENCED (Plan 33.1-07 Task 4). Phase 30's narrowing returned each matrix to
    exactly its Phase-28 width, so a zero residual was the right assertion for as
    long as nothing else moved the width. Phase 33.1's rung 1 then added the
    weather coverage flag -- the column that lets a reader tell "no weather
    record" from "the weather was mild" -- so the residual is now +1 BY DECISION.

    The expected residual is ``len(PHASE_331_ADDED_WEATHER_COLUMNS)`` rather than
    the integer 1, and the NAME is asserted separately below. That is the
    difference between "the width moved by one" and "the width moved by one, and
    the one is the column we meant": a build that added an unrelated column while
    omitting the flag satisfies the integer exactly as well as the right one does.
    """
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    columns = list(pd.read_parquet(path).columns)
    surviving = [c for c in PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS if c in columns]
    assert not surviving, (
        f"{table_name} still carries line-movement columns {surviving} -- the SPEC "
        f"R3 drop is PARTIAL. A partial drop is worse than none: the three matrices "
        f"then disagree about the candidate feature set."
    )

    residual_delta = GOLD_FEATURE_MATRICES[table_name] - PHASE_28_WIDTHS[table_name]
    expected_residual = len(PHASE_331_ADDED_WEATHER_COLUMNS)
    assert residual_delta == expected_residual, (
        f"{table_name}: the tripwire reads "
        f"{GOLD_FEATURE_MATRICES[table_name]}, which is {residual_delta} columns "
        f"from the Phase-28 width {PHASE_28_WIDTHS[table_name]}. Removing the "
        f"{len(PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS)}-column line-movement family "
        f"returns the matrix to exactly its Phase-28 width, and Phase 33.1 adds "
        f"{expected_residual} on top of that: "
        f"{list(PHASE_331_ADDED_WEATHER_COLUMNS)}. A residual other than "
        f"{expected_residual} means something ELSE changed the gold width. "
        f"Identify it before updating the tripwire."
    )


@pytest.mark.parametrize("table_name", list(GOLD_FEATURE_MATRICES))
def test_the_phase_331_widening_is_pinned_to_the_named_coverage_flag(
    table_name: str,
) -> None:
    """The +1 is pinned to a NAME, in both directions (Plan 33.1-07 Task 4).

    Direction one: the named column is PRESENT in the matrix. Direction two: the
    width equals the Phase-28 width plus exactly the length of the named tuple.

    Asserting only the second would pass for a build that added an unrelated
    column while omitting the flag; asserting only the first would pass for a
    build that added the flag AND something else. Together they say the width
    moved by these columns and no others.

    Removing ``weather_coverage`` from a copy of a matrix makes THIS test fail by
    NAME, where the width tripwire alone would only have reported an integer.
    """
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    columns = set(pd.read_parquet(path).columns)
    missing = [c for c in PHASE_331_ADDED_WEATHER_COLUMNS if c not in columns]
    assert not missing, (
        f"{table_name} does not carry {missing}. Without the coverage flag a NULL "
        f"observation is indistinguishable from a measured one, which is the whole "
        f"point of the Phase-33.1 rung (SPEC R5) -- and the width integer alone "
        f"cannot say WHICH column arrived."
    )

    assert GOLD_FEATURE_MATRICES[table_name] == PHASE_28_WIDTHS[table_name] + len(
        PHASE_331_ADDED_WEATHER_COLUMNS
    ), (
        f"{table_name}: the tripwire reads {GOLD_FEATURE_MATRICES[table_name]}, "
        f"which is not the Phase-28 width {PHASE_28_WIDTHS[table_name]} plus the "
        f"{len(PHASE_331_ADDED_WEATHER_COLUMNS)} named Phase-33.1 column(s) "
        f"{list(PHASE_331_ADDED_WEATHER_COLUMNS)}."
    )


@pytest.mark.parametrize("table_name", list(GOLD_FEATURE_MATRICES))
def test_the_market_survivors_are_not_taken_with_the_family(table_name: str) -> None:
    """The drop must be a family removal, not a suffix sweep.

    ``snapshot_total`` ends in ``total`` and ``spread_movement`` contains
    ``spread``; a substring-based prune would take all four of these with the
    fifteen and quietly delete the freeze anchors the whole market leg rests on.
    """
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    columns = set(pd.read_parquet(path).columns)
    missing = [c for c in MARKET_SURVIVORS if c not in columns]
    assert not missing, (
        f"{table_name} lost market columns {missing} to the line-movement drop. "
        f"These are pre-existing baseline features, not Phase-29 columns."
    )
