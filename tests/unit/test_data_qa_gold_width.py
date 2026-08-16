"""Gold-width tripwire parity test (IN-03, Plan 28-06 / SIG-06, Plan 29-06 / SIG-04).

``scripts.data_qa.GOLD_FEATURE_MATRICES`` hardcodes the expected column count of each
gold feature matrix as a deliberate audit tripwire: any stray/dropped column trips it
rather than passing silently. This test asserts the tripwire is itself HONEST -- the
declared expected width equals the real on-disk column count of each rebuilt matrix.

If a future builder legitimately adds or removes a column, this test fails until the
operator UPDATES ``GOLD_FEATURE_MATRICES`` to the new empirically-counted width
(Pitfall 6 -- never silence the tripwire by widening tolerances).
"""

from pathlib import Path

import pandas as pd
import pytest

from scripts.data_qa import GOLD_FEATURE_MATRICES

GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"

# The Phase-29 widening (Plan 29-06 / SIG-04): the seven D-09 totals features,
# the shared coverage flag, and the seven Tier (a) spread siblings -- 15
# game-level columns per matrix. Pinned as a NAMED delta so a future width bump
# has to state which columns it added rather than silently moving an integer.
PHASE_29_LINE_MOVEMENT_COLUMNS = (
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

# The Phase-28 widths this phase widened FROM (scripts/data_qa.py header).
PHASE_28_WIDTHS = {"features_wp": 194, "features_ats": 195, "features_ou": 194}


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


@pytest.mark.parametrize("table_name", list(GOLD_FEATURE_MATRICES))
def test_phase_29_widening_is_exactly_the_line_movement_family(
    table_name: str,
) -> None:
    """The Phase-29 width delta is accounted for, column by column.

    The width tripwire on its own only says the total moved by 15; it cannot say
    the 15 are the intended columns. This pins the delta to the NAMED
    line-movement family, so a build that dropped a line-movement column while
    incidentally adding an unrelated one would fail here even though the integer
    still matched.
    """
    path = GOLD_DIR / f"{table_name}.parquet"
    if not path.exists():
        pytest.skip(f"{path} not built yet -- run scripts.build_features first")

    columns = list(pd.read_parquet(path).columns)
    missing = [c for c in PHASE_29_LINE_MOVEMENT_COLUMNS if c not in columns]
    assert not missing, f"{table_name} is missing line-movement columns {missing}"

    expected_delta = len(PHASE_29_LINE_MOVEMENT_COLUMNS)
    actual_delta = GOLD_FEATURE_MATRICES[table_name] - PHASE_28_WIDTHS[table_name]
    assert actual_delta == expected_delta, (
        f"{table_name}: the tripwire moved by {actual_delta} columns but the "
        f"Phase-29 line-movement family is {expected_delta} columns. Something "
        f"else changed the gold width -- identify it before updating the tripwire."
    )
