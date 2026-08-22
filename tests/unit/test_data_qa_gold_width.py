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
"""

from pathlib import Path

import pandas as pd
import pytest

from backtest.signal_lift import group_columns
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
    the tripwire has returned to the Phase-28 width -- a zero delta, because the
    phase gives back exactly what Phase 29 took.
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
    assert residual_delta == 0, (
        f"{table_name}: the tripwire reads "
        f"{GOLD_FEATURE_MATRICES[table_name]}, which is {residual_delta} columns "
        f"from the Phase-28 width {PHASE_28_WIDTHS[table_name]}. Removing the "
        f"{len(PHASE_30_REMOVED_LINE_MOVEMENT_COLUMNS)}-column line-movement family "
        f"returns the matrix to exactly its Phase-28 width, so a non-zero residual "
        f"means something ELSE changed the gold width. Identify it before updating "
        f"the tripwire."
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
