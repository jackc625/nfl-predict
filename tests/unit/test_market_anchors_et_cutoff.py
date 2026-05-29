"""G-03: market-anchor snapshot cutoff is localized to ET, not UTC (Phase 20, WR-02).

The project's load-bearing reproducibility invariant: odds/data freeze at "Friday 6 PM ET".

Bug (pre-fix): identify_snapshot_lines localized the cutoff to UTC, producing
Friday 18:00 UTC = Friday 14:00 ET (4 hours early), which silently dropped the
legitimate 18:00 ET (= 22:00 UTC) snapshots -- emptying the snapshot set on the
orchestrator path.

Fix (commit dcc7883): cutoff_time.replace(tzinfo=ET) -- the cutoff is now an ET-aware
datetime that compares correctly against the tz-aware (UTC) snapshot_ts column.

This test constructs snapshot timestamps that straddle the Friday 6 PM boundary in a
way where ET-vs-UTC selection DIFFERS:

    snapshot at Friday 18:00 UTC = Friday 14:00 ET  (before 6 PM ET, after 6 PM UTC)
    snapshot at Friday 22:00 UTC = Friday 18:00 ET  (at exactly 6 PM ET, the real cutoff)

Under the old UTC cutoff (<=18:00 UTC), the 22:00 UTC snapshot is EXCLUDED.
Under the correct ET cutoff (<=18:00 ET = <=22:00 UTC), the 22:00 UTC snapshot is INCLUDED.

The test asserts the ET-correct behavior: the 22:00 UTC snapshot IS included.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd

from features.market_anchors import MarketAnchorFeaturesCalculator
from utils.date_utils import ET

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_odds_row(
    game_id: str,
    sportsbook: str,
    snapshot_utc: datetime,
    spread: float = -3.5,
    total: float = 44.5,
    home_ml: int = -150,
    away_ml: int = 130,
) -> dict:
    """Build a minimal odds-snapshot row with all columns identify_snapshot_lines needs."""
    return {
        "game_id": game_id,
        "sportsbook": sportsbook,
        "snapshot_ts": snapshot_utc.isoformat(),
        "commence_time": (snapshot_utc + timedelta(days=2)).isoformat(),
        "home_spread": spread,
        "away_spread": -spread,
        "spread_home_price": -110,
        "spread_away_price": -110,
        "total_points": total,
        "over_price": -110,
        "under_price": -110,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSnapshotCutoffIsET:
    """identify_snapshot_lines selects the Friday 6 PM ET cutoff, not UTC (G-03)."""

    def test_snapshot_at_6pm_et_is_included(self):
        """A snapshot taken at exactly Friday 18:00 ET (22:00 UTC in EDT / 23:00 UTC in EST)
        must be INCLUDED in the snapshot set -- it is at the cutoff, not after it.

        This is the critical boundary the UTC bug broke: 22:00 UTC > 18:00 UTC, so the
        UTC cutoff incorrectly excluded it.  The ET cutoff (18:00 ET = 22:00 UTC in EDT
        or 23:00 UTC in EST) includes it.
        """
        # Use a November (EST = UTC-5) Friday so the math is simple:
        # Friday Nov 15 2024 18:00 ET = Friday Nov 15 2024 23:00 UTC
        friday = datetime(2024, 11, 15, 18, 0, 0, tzinfo=ET)
        at_cutoff_utc = friday.astimezone(UTC)

        # One snapshot at exactly 18:00 ET (the cutoff boundary)
        cutoff_snap = _make_odds_row(
            game_id="2024_W11_KC@BUF",
            sportsbook="DraftKings",
            snapshot_utc=at_cutoff_utc,
        )
        # One snapshot AFTER the cutoff (should be excluded)
        after_cutoff_utc = at_cutoff_utc + timedelta(hours=2)
        after_snap = _make_odds_row(
            game_id="2024_W11_KC@BUF",
            sportsbook="DraftKings",
            snapshot_utc=after_cutoff_utc,
        )

        odds_df = pd.DataFrame([cutoff_snap, after_snap])

        # Provide the target_date as the same Friday (naive -- function localizes it)
        target_date = datetime(2024, 11, 15, 18, 0, 0)

        calc = MarketAnchorFeaturesCalculator()
        result = calc.identify_snapshot_lines(odds_df, target_date=target_date)

        assert len(result) == 1, (
            f"Expected exactly 1 snapshot line (the at-cutoff 18:00 ET snapshot); "
            f"got {len(result)}. If 0: the ET cutoff is still treating 18:00 ET as "
            f"too late (UTC bug regression). If 2: the post-cutoff snapshot leaked in."
        )
        # The returned snapshot must be the 18:00 ET one, not the later one
        returned_ts = pd.to_datetime(result.iloc[0]["snapshot_ts"], utc=True)
        assert returned_ts <= at_cutoff_utc + timedelta(seconds=1), (
            f"Returned snapshot timestamp {returned_ts} is later than the 18:00 ET "
            f"cutoff {at_cutoff_utc} -- post-cutoff data leaked in"
        )

    def test_et_vs_utc_boundary_differs_by_offset(self):
        """Construct a snapshot where the ET and UTC cutoffs disagree, and assert
        the ET cutoff wins.

        Discriminating case -- a snapshot at 19:00 UTC on a Friday in EDT (UTC-4):
            19:00 UTC = 15:00 EDT (before 18:00 ET -> INCLUDED under the ET cutoff)
        - Old UTC cutoff (<=18:00 UTC): 19:00 > 18:00 -> EXCLUDED (the WR-02 bug).
        - Correct ET cutoff (18:00 ET = 22:00 UTC): 19:00 < 22:00 -> INCLUDED (right).

        Uses a September (EDT = UTC-4) Friday to produce this discriminating case.
        """
        # September 20 2024 is a real Friday. EDT = UTC-4, so 18:00 ET = 22:00 UTC.
        # Snapshot at 19:00 UTC (= 15:00 EDT) -- before the ET cutoff, after 18:00 UTC
        discriminating_utc = datetime(2024, 9, 20, 19, 0, 0, tzinfo=UTC)
        # 19:00 UTC < 22:00 UTC -> ET cutoff INCLUDES it
        # 19:00 UTC > 18:00 UTC -> UTC cutoff EXCLUDES it (the old bug)

        discriminating_snap = _make_odds_row(
            game_id="2024_W03_DAL@NYG",
            sportsbook="FanDuel",
            snapshot_utc=discriminating_utc,
        )
        odds_df = pd.DataFrame([discriminating_snap])

        target_date = datetime(2024, 9, 20, 18, 0, 0)  # naive Friday

        calc = MarketAnchorFeaturesCalculator()
        result = calc.identify_snapshot_lines(odds_df, target_date=target_date)

        assert len(result) == 1, (
            f"ET cutoff regression: a snapshot at 19:00 UTC (= 15:00 EDT) on a Friday "
            f"should be INCLUDED under the 18:00 ET cutoff (= 22:00 UTC). "
            f"Got {len(result)} rows. If 0: the cutoff is still using UTC (18:00 UTC "
            f"instead of 18:00 ET=22:00 UTC), excluding valid pre-cutoff snapshots "
            f"-- this is the WR-02 bug."
        )

    def test_post_et_cutoff_snapshot_is_excluded(self):
        """A snapshot at 23:00 UTC on a Friday in EDT (= 19:00 EDT) is AFTER 18:00 ET
        and must be EXCLUDED regardless of UTC value.

        This is the complementary side of the ET correctness invariant:
        the cutoff is 18:00 ET, and anything after that must not appear.
        """
        # September 20 2024 is a real Friday. EDT = UTC-4, so 18:00 ET = 22:00 UTC.
        # Snapshot at 23:00 UTC (= 19:00 EDT) -- after 18:00 ET, must be excluded
        post_cutoff_utc = datetime(2024, 9, 20, 23, 0, 0, tzinfo=UTC)

        post_snap = _make_odds_row(
            game_id="2024_W03_DAL@NYG",
            sportsbook="FanDuel",
            snapshot_utc=post_cutoff_utc,
        )
        # Also add a pre-cutoff snapshot to confirm the function works at all
        pre_snap = _make_odds_row(
            game_id="2024_W03_DAL@NYG",
            sportsbook="BetMGM",
            snapshot_utc=datetime(2024, 9, 20, 20, 0, 0, tzinfo=UTC),  # 16:00 EDT
        )
        odds_df = pd.DataFrame([pre_snap, post_snap])

        target_date = datetime(2024, 9, 20, 18, 0, 0)

        calc = MarketAnchorFeaturesCalculator()
        result = calc.identify_snapshot_lines(odds_df, target_date=target_date)

        assert len(result) == 1, (
            f"Post-ET-cutoff snapshot included: expected 1 row (pre-cutoff only), "
            f"got {len(result)}. The 23:00 UTC (19:00 EDT) snapshot is AFTER 18:00 ET "
            f"and must be excluded. If 2: post-cutoff data is leaking in."
        )
        # The returned row must be the pre-cutoff one (BetMGM, 20:00 UTC = 16:00 EDT)
        returned_sportsbook = result.iloc[0]["sportsbook"]
        assert returned_sportsbook == "BetMGM", (
            f"Expected the pre-cutoff BetMGM snapshot to be selected; "
            f"got sportsbook '{returned_sportsbook}' -- wrong snapshot selected"
        )
