"""Integration: snap + injury columns survive the merge into all three gold matrices.

Plan 28-06 / review #1 / SIG-06. ``combine_features`` in ``scripts/build_features.py``
has NO generic loop over ``feature_sources`` keys -- it merges each source through an
EXPLICIT per-source block. Registering ``SnapCountBuilder`` / ``InjuryBuilder`` into
``feature_sources`` routes them through the LeakageGate, but their columns are SILENTLY
DROPPED from gold unless ``combine_features`` also gets explicit snap + injury merge
blocks. This test is the proof that the home_/away_ snap + injury columns actually reach
``features_wp``, ``features_ats``, AND ``features_ou`` (not merely pass the gate).

It reads the rebuilt gold parquet on disk (the deliverable of the Plan 28-06 full build).
If the gold has not been rebuilt yet the test skips rather than failing spuriously.
"""

import re
from pathlib import Path

import pandas as pd
import pytest

GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"
MATRICES = ["features_wp", "features_ats", "features_ou"]

# A snap column is named home_snap_* / home_rolling_snap_share_* (review #1); the
# one injury column that ends in "injury" is the home_/away_injury_coverage flag.
SNAP_HOME = re.compile(r"^home_.*snap")
SNAP_AWAY = re.compile(r"^away_.*snap")
INJURY_HOME = re.compile(r"^home_.*injury")
INJURY_AWAY = re.compile(r"^away_.*injury")

# Snaps are covered from 2013, injuries from 2009; 2023 is in coverage for both, so
# the home snap columns must carry REAL (non-null) values for 2023 games -- proof the
# merge delivered data, not just an all-NaN placeholder column.
IN_COVERAGE_SEASON = 2023


@pytest.fixture(scope="module")
def gold_frames() -> dict[str, pd.DataFrame]:
    """Load the three rebuilt gold matrices (skip the suite if not yet built)."""
    frames: dict[str, pd.DataFrame] = {}
    for name in MATRICES:
        path = GOLD_DIR / f"{name}.parquet"
        if not path.exists():
            pytest.skip(f"{path} not built yet -- run scripts.build_features first")
        frames[name] = pd.read_parquet(path)
    return frames


@pytest.mark.parametrize("matrix", MATRICES)
def test_snap_columns_present(
    gold_frames: dict[str, pd.DataFrame], matrix: str
) -> None:
    """Each gold matrix carries >=1 home_ and >=1 away_ snap column (review #1)."""
    cols = list(gold_frames[matrix].columns)
    assert any(SNAP_HOME.match(c) for c in cols), (
        f"{matrix} is missing a home_*snap column -- the snap merge block did not "
        f"reach this matrix. Columns sampled: {[c for c in cols if 'snap' in c][:10]}"
    )
    assert any(SNAP_AWAY.match(c) for c in cols), (
        f"{matrix} is missing an away_*snap column."
    )


@pytest.mark.parametrize("matrix", MATRICES)
def test_injury_columns_present(
    gold_frames: dict[str, pd.DataFrame], matrix: str
) -> None:
    """Each gold matrix carries >=1 home_ and >=1 away_ injury column (review #1)."""
    cols = list(gold_frames[matrix].columns)
    assert any(INJURY_HOME.match(c) for c in cols), (
        f"{matrix} is missing a home_*injury column -- the injury merge block did "
        f"not reach this matrix. Columns sampled: "
        f"{[c for c in cols if 'injury' in c][:10]}"
    )
    assert any(INJURY_AWAY.match(c) for c in cols), (
        f"{matrix} is missing an away_*injury column."
    )


@pytest.mark.parametrize("matrix", MATRICES)
def test_snap_columns_nonnull_in_coverage_season(
    gold_frames: dict[str, pd.DataFrame], matrix: str
) -> None:
    """Home snap columns carry real (non-null) values for an in-coverage season.

    A column can be PRESENT yet all-NaN (the merge wired the name but no data flowed).
    Spot-checking 2023 -- inside both the snap (2013+) and injury (2009+) coverage
    floors -- proves the merge delivered actual values.
    """
    df = gold_frames[matrix]
    if "season" not in df.columns:
        pytest.skip(f"{matrix} carries no season column for the coverage spot-check")
    season_rows = df[df["season"] == IN_COVERAGE_SEASON]
    if len(season_rows) == 0:
        pytest.skip(f"{matrix} has no {IN_COVERAGE_SEASON} rows")
    snap_home_cols = [c for c in df.columns if SNAP_HOME.match(c)]
    assert snap_home_cols, f"{matrix} has no home_*snap columns at all"
    assert any(season_rows[c].notna().any() for c in snap_home_cols), (
        f"{matrix}: every home_*snap column is entirely NaN for "
        f"{IN_COVERAGE_SEASON} -- the merge wired the column names but no snap data "
        f"reached gold."
    )
