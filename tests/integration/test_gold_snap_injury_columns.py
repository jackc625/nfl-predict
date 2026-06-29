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

# A snap column is named home_snap_* / home_rolling_snap_share_* (review #1).
SNAP_HOME = re.compile(r"^home_.*snap")
SNAP_AWAY = re.compile(r"^away_.*snap")

# WR-03: assert the real injury SIGNAL columns survive the merge -- not merely
# the *_injury_coverage flag (the only injury column whose name contains
# "injury", which a ``^home_.*injury`` regex would match). These are the
# InjuryBuilder per-side feature columns (features/injury.py:96-103). Matching
# only the coverage flag would still pass even if qb_out_flag /
# backup_quality_delta / availability_fraction were dropped from gold entirely.
INJURY_SIGNAL_HOME = (
    "home_qb_out_flag",
    "home_backup_quality_delta",
    "home_availability_fraction",
)
INJURY_SIGNAL_AWAY = (
    "away_qb_out_flag",
    "away_backup_quality_delta",
    "away_availability_fraction",
)
# A non-coverage injury SIGNAL column for the non-null spot-check: availability
# is populated (a 1.0 neutral default or an actual value) inside the 2009+
# injury coverage floor (D-10), so a non-null value proves the merge delivered
# real data rather than an all-NaN placeholder column.
INJURY_SIGNAL_NONNULL_COL = "home_availability_fraction"

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
    """Each gold matrix carries the real injury SIGNAL columns -- qb_out_flag /
    backup_quality_delta / availability_fraction for both sides -- not merely the
    *_injury_coverage flag (WR-03)."""
    cols = set(gold_frames[matrix].columns)
    required = (*INJURY_SIGNAL_HOME, *INJURY_SIGNAL_AWAY)
    missing = [c for c in required if c not in cols]
    assert not missing, (
        f"{matrix} is missing injury SIGNAL columns {missing} -- the injury merge "
        f"block did not deliver them (matching only *_injury_coverage would hide "
        f"this). Injury-ish columns present: "
        f"{sorted(c for c in cols if any(k in c for k in ('injur', 'qb_out', 'backup', 'availability')))[:12]}"
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


@pytest.mark.parametrize("matrix", MATRICES)
def test_injury_signal_column_nonnull_in_coverage_season(
    gold_frames: dict[str, pd.DataFrame], matrix: str
) -> None:
    """A real injury SIGNAL column carries non-null values for an in-coverage
    season (WR-03).

    Spot-checking ``home_availability_fraction`` -- a SIGNAL column, not the
    coverage flag -- inside the 2009+ injury coverage floor proves the injury
    merge delivered actual values, not merely the column name or an all-NaN
    placeholder.
    """
    df = gold_frames[matrix]
    if "season" not in df.columns:
        pytest.skip(f"{matrix} carries no season column for the coverage spot-check")
    assert INJURY_SIGNAL_NONNULL_COL in df.columns, (
        f"{matrix} is missing {INJURY_SIGNAL_NONNULL_COL} -- the injury merge block "
        f"did not deliver the availability signal."
    )
    season_rows = df[df["season"] == IN_COVERAGE_SEASON]
    if len(season_rows) == 0:
        pytest.skip(f"{matrix} has no {IN_COVERAGE_SEASON} rows")
    assert season_rows[INJURY_SIGNAL_NONNULL_COL].notna().any(), (
        f"{matrix}: {INJURY_SIGNAL_NONNULL_COL} is entirely NaN for "
        f"{IN_COVERAGE_SEASON} -- the merge wired the column name but no injury "
        f"signal data reached gold."
    )
