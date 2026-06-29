"""Leakage proof scaffold for the injury signal (SIG-01 / SC1).

Three-part canonical standard (D-18a), following the
`tests/unit/test_elo_no_leakage.py` convention. The InjuryBuilder does not exist
yet (it lands in Plan 28-05), so every test here is a RED scaffold skipped with a
plan-named reason. The Friday-freeze fence is the load-bearing novel code:
injury rows are filtered to `date_modified <= as_of_datetime` (the Friday 6 PM ET
freeze), so revealing a Saturday/Sunday game-day report must not change any
pre-game feature value.

Parts (filled by Plan 28-05):
  1. Time-fence assertion -- builder respects `as_of_datetime` (the date_modified fence).
  2. Withhold-future byte-unchanged -- reveal a Saturday/Sunday `date_modified` row,
     assert qb_out_flag (and availability) unchanged.
  3. Status-vocab -- report_status is drawn from {None, Questionable, Out, Doubtful}.
"""

import pandas as pd
import pytest


def _make_injury_fixture() -> pd.DataFrame:
    """Minimal injuries frame (shape of `nfl.load_injuries(season).to_pandas()`)
    with one Friday pre-freeze row and one Saturday game-day row for the same
    player, to exercise the `date_modified <= freeze` fence.

    Returns:
        A 2-row injuries DataFrame.
    """
    return pd.DataFrame(
        {
            "season": [2023, 2023],
            "week": [1, 1],
            "team": ["KC", "KC"],
            "gsis_id": ["00-0033873", "00-0033873"],
            "position": ["QB", "QB"],
            "full_name": ["Patrick Mahomes", "Patrick Mahomes"],
            "report_status": ["Questionable", "Out"],
            # Friday pre-freeze report vs Saturday game-day downgrade
            "date_modified": pd.to_datetime(
                ["2023-09-08T21:00:00Z", "2023-09-09T18:00:00Z"], utc=True
            ),
        }
    )


class TestInjuryTimeFence:
    """SKIPPED scaffold: InjuryBuilder date_modified fence (Plan 28-05)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- InjuryBuilder lands in Plan 28-05 "
        "(date_modified <= Friday-6PM-ET freeze fence)"
    )
    def test_injury_builder_respects_date_modified_fence(self):
        """InjuryBuilder must drop any injury row whose date_modified is after
        the as_of_datetime freeze. Filled by Plan 28-05."""
        raise NotImplementedError("Plan 28-05")


class TestInjuryWithholdFuture:
    """SKIPPED scaffold: withhold-future byte-unchanged proof (Plan 28-05)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- InjuryBuilder lands in Plan 28-05 "
        "(reveal a Saturday/Sunday game-day date_modified row, assert "
        "qb_out_flag byte-unchanged)"
    )
    def test_revealing_gameday_report_does_not_change_features(self):
        """Building the injury features once with the Saturday game-day row
        present and once without (only the Friday row) must produce a
        byte-identical qb_out_flag and availability. Filled by Plan 28-05."""
        raise NotImplementedError("Plan 28-05")


class TestInjuryStatusVocab:
    """SKIPPED scaffold: report_status vocabulary check (Plan 28-05)."""

    @pytest.mark.skip(
        reason="Wave-0 scaffold -- InjuryBuilder lands in Plan 28-05 "
        "(report_status in {None, Questionable, Out, Doubtful}; the Out/Doubtful "
        "test reads report_status, not a non-existent game_status column)"
    )
    def test_report_status_vocabulary(self):
        """The {Out, Doubtful} out-flag derivation reads `report_status` (there is
        no `game_status` column). Filled by Plan 28-05."""
        raise NotImplementedError("Plan 28-05")
