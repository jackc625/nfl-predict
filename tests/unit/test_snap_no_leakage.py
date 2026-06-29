"""Leakage proof for the snap-count signal (SIG-02 / SC2).

Three-part canonical standard (D-18a), following the
`tests/unit/test_elo_no_leakage.py` convention:

  1. LEAKAGE_KEYWORDS hard-fail (LIVE, this plan) -- a stray raw `offense_snaps`
     column in the combined matrix raises LeakageViolation, while the derived
     `rolling_snap_continuity` / `snap_concentration` names do NOT trip the
     substring guard (the D-13 substring-collision contract).
  2. Time-fence assertion (SKIPPED scaffold -- SnapCountBuilder lands in Plan 28-03).
  3. Withhold-future byte-unchanged (SKIPPED scaffold -- SnapCountBuilder, Plan 28-03).

The keyword tests are LIVE GREEN because they depend only on the now-present
LeakageGate keyword guard (Plan 28-01, Task 1). The builder-dependent parts are
RED scaffolds skipped with a plan-named reason until Plan 28-03 fills them.
"""

from datetime import datetime

import pandas as pd
import pytest

from features.validation import LeakageGate, LeakageViolation


def _make_combined_matrix(extra_columns: dict[str, list] | None = None) -> pd.DataFrame:
    """Build a minimal combined feature matrix that satisfies the required
    feature groups (elo_, rolling_) so the only thing under test is the
    leakage-keyword scan.

    Args:
        extra_columns: Optional extra columns to splice in (the snap columns
            under test).

    Returns:
        A 3-row combined feature matrix DataFrame.
    """
    df = pd.DataFrame(
        {
            "game_id": ["G1", "G2", "G3"],
            "season": [2024, 2024, 2024],
            "week": [1, 2, 3],
            "elo_home": [1520.0, 1530.0, 1525.0],
            "elo_away": [1480.0, 1470.0, 1475.0],
            "rolling_epa_home": [0.05, 0.08, 0.06],
            "rolling_epa_away": [-0.02, 0.01, -0.01],
        }
    )
    if extra_columns:
        for col, values in extra_columns.items():
            df[col] = values
    return df


class TestSnapKeywordHardFail:
    """LIVE: the raw snap spellings hard-fail the LeakageGate keyword guard."""

    def test_raw_offense_snaps_column_raises(self):
        """A stray raw `offense_snaps` column in the combined matrix raises
        LeakageViolation with violation_type=leakage_keyword (D-13, SIG-02)."""
        gate = LeakageGate()
        as_of = datetime(2024, 9, 6, 18, 0)
        df = _make_combined_matrix({"offense_snaps": [55, 60, 58]})

        with pytest.raises(LeakageViolation) as exc_info:
            gate.validate_combined_matrix(df, as_of)

        assert exc_info.value.details["violation_type"] == "leakage_keyword"
        assert "offense_snaps" in exc_info.value.details["affected_features"]

    @pytest.mark.parametrize(
        "raw_col",
        ["defense_snaps", "st_snaps", "offense_pct", "defense_pct", "st_pct"],
    )
    def test_every_raw_snap_spelling_raises(self, raw_col):
        """Each of the six raw snap spellings trips the guard, not just
        offense_snaps (proves the full D-13 keyword set is live)."""
        gate = LeakageGate()
        as_of = datetime(2024, 9, 6, 18, 0)
        df = _make_combined_matrix({raw_col: [0.5, 0.6, 0.55]})

        with pytest.raises(LeakageViolation) as exc_info:
            gate.validate_combined_matrix(df, as_of)

        assert exc_info.value.details["violation_type"] == "leakage_keyword"
        assert raw_col in exc_info.value.details["affected_features"]


class TestSnapDerivedNamesNoFalsePositive:
    """LIVE: derived snap-feature names must NOT trip the substring guard."""

    def test_derived_snap_names_do_not_raise(self):
        """A matrix whose only snap-derived columns are `rolling_snap_continuity`
        and `snap_concentration` does NOT raise -- the bare `snap`/`snaps`
        keyword is deliberately absent from LEAKAGE_KEYWORDS (D-13)."""
        gate = LeakageGate()
        as_of = datetime(2024, 9, 6, 18, 0)
        df = _make_combined_matrix(
            {
                "rolling_snap_continuity": [0.9, 0.85, 0.92],
                "snap_concentration": [0.3, 0.35, 0.31],
                "rolling_snap_off_share": [0.7, 0.72, 0.69],
            }
        )

        # Must NOT raise: derived names use _share / _concentration, never the
        # raw spellings or a bare snap/snaps keyword.
        gate.validate_combined_matrix(df, as_of)


class TestSnapTimeFence:
    """SKIPPED scaffold: SnapCountBuilder time-fence proof (Plan 28-03)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- SnapCountBuilder lands in Plan 28-03 "
        "(week-based fence: season < target_season | week < target_week)"
    )
    def test_snap_builder_respects_week_fence(self):
        """SnapCountBuilder must only use prior-week snap rows (week < target_week
        within target_season, or any prior season) -- no current/future-week rows
        feed the backward-rolling features. Filled by Plan 28-03."""
        raise NotImplementedError("Plan 28-03")


class TestSnapWithholdFuture:
    """SKIPPED scaffold: withhold-future byte-unchanged proof (Plan 28-03)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- SnapCountBuilder lands in Plan 28-03 "
        "(reveal a current-week post-game snap row, assert rolling_snap_* "
        "byte-unchanged)"
    )
    def test_revealing_current_week_snaps_does_not_change_features(self):
        """Building the rolling snap features once with a current-week post-game
        snap row present and once without must produce byte-identical
        rolling_snap_* values. Filled by Plan 28-03."""
        raise NotImplementedError("Plan 28-03")
