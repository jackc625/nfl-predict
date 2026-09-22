"""G-03: market-anchor snapshot cutoff is an ET instant, not a UTC-localized 18:00 (WR-02).

The project's load-bearing reproducibility invariant is now each game's LOCK: 18:00 ET on the
ET calendar day before its kickoff (D33.2-01). Plan 33.2-02 renamed the method under test to
``select_snapshot_lines_at_lock`` and deleted its one-global-Friday derivation by ruling; the
ET-not-UTC property these tests were written for survives unchanged, retargeted at the lock.

Bug (pre-fix, Phase 20): the cutoff was localized to UTC, producing 18:00 UTC = 14:00 ET (4
hours early), which silently dropped the legitimate 18:00 ET (= 22:00 UTC) snapshots --
emptying the snapshot set on the orchestrator path.

Plan 33.2-14 moved admission onto the RECORDED CAPTURE TIME (``created_at``) by owner ruling
2026-09-22; each fixture row is captured at the instant it names, so the ET-not-UTC boundary
below is unchanged and now applies to the capture.

These tests construct snapshot timestamps that straddle a game's 6 PM ET lock in a way where
ET-vs-UTC selection DIFFERS:

    snapshot at 18:00 UTC = 14:00 ET  (before 6 PM ET, after 6 PM UTC)
    snapshot at 22:00 UTC = 18:00 ET  (exactly at the lock, in EDT)

Under a UTC-localized cutoff (<=18:00 UTC), the 22:00 UTC snapshot is EXCLUDED.
Under the correct ET lock (<=18:00 ET = <=22:00 UTC), the 22:00 UTC snapshot is INCLUDED.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from features.market_anchors import MarketAnchorFeaturesCalculator
from utils.date_utils import ET

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _games(*rows: tuple[str, datetime]) -> pd.DataFrame:
    """The games frame the per-game lock is derived from (``game_id``, aware ``kickoff_et``)."""
    return pd.DataFrame(
        {
            "game_id": [game_id for game_id, _ in rows],
            "kickoff_et": [pd.Timestamp(kickoff) for _, kickoff in rows],
        }
    )


def _make_odds_row(
    game_id: str,
    sportsbook: str,
    snapshot_utc: datetime,
    spread: float = -3.5,
    total: float = 44.5,
    home_ml: int = -150,
    away_ml: int = 130,
) -> dict:
    """Build a minimal odds-snapshot row with the columns the lock selection needs.

    The row is CAPTURED at *snapshot_utc*: ``created_at`` carries that instant, and it is
    what the lock selection reads (Plan 33.2-14, owner ruling 2026-09-22 -- a line counts
    only with a recorded capture time; the ``snapshot_ts`` label is never an information
    time). ``snapshot_ts`` carries the same instant so the returned label can be checked.
    """
    return {
        "game_id": game_id,
        "sportsbook": sportsbook,
        "created_at": pd.Timestamp(snapshot_utc),
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
    """select_snapshot_lines_at_lock cuts each game at its own 6 PM ET lock, not UTC (G-03)."""

    def test_snapshot_at_6pm_et_is_included(self):
        """A snapshot taken exactly AT the lock (18:00 ET = 23:00 UTC in EST) is INCLUDED.

        At-lock is admissible (D33.2-01). This is also the boundary the UTC bug broke:
        23:00 UTC > 18:00 UTC, so a UTC-localized cutoff excluded it.
        """
        # A November (EST = UTC-5) Sunday game, Nov 17 2024 1 PM ET. Its lock is Saturday
        # Nov 16 2024 18:00 ET = 23:00 UTC.
        kickoff = datetime(2024, 11, 17, 13, 0, 0, tzinfo=ET)
        at_cutoff_utc = datetime(2024, 11, 16, 18, 0, 0, tzinfo=ET).astimezone(UTC)
        assert at_cutoff_utc.hour == 23

        # One snapshot at exactly 18:00 ET (the lock)
        cutoff_snap = _make_odds_row(
            game_id="2024_W11_KC@BUF",
            sportsbook="DraftKings",
            snapshot_utc=at_cutoff_utc,
        )
        # One snapshot AFTER the lock (should be excluded)
        after_cutoff_utc = at_cutoff_utc + timedelta(hours=2)
        after_snap = _make_odds_row(
            game_id="2024_W11_KC@BUF",
            sportsbook="DraftKings",
            snapshot_utc=after_cutoff_utc,
        )

        odds_df = pd.DataFrame([cutoff_snap, after_snap])

        calc = MarketAnchorFeaturesCalculator()
        result = calc.select_snapshot_lines_at_lock(
            odds_df, _games(("2024_W11_KC@BUF", kickoff))
        )

        assert len(result) == 1, (
            f"Expected exactly 1 snapshot line (the at-lock 18:00 ET snapshot); "
            f"got {len(result)}. If 0: the cutoff is treating 18:00 ET as "
            f"too late (UTC bug regression). If 2: the post-lock snapshot leaked in."
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
        # The game kicks off Saturday Sep 21 2024, so its lock is Friday Sep 20 18:00 EDT.
        # Snapshot at 19:00 UTC (= 15:00 EDT) -- before the ET lock, after 18:00 UTC
        discriminating_utc = datetime(2024, 9, 20, 19, 0, 0, tzinfo=UTC)
        # 19:00 UTC < 22:00 UTC -> ET cutoff INCLUDES it
        # 19:00 UTC > 18:00 UTC -> UTC cutoff EXCLUDES it (the old bug)

        discriminating_snap = _make_odds_row(
            game_id="2024_W03_DAL@NYG",
            sportsbook="FanDuel",
            snapshot_utc=discriminating_utc,
        )
        odds_df = pd.DataFrame([discriminating_snap])

        calc = MarketAnchorFeaturesCalculator()
        result = calc.select_snapshot_lines_at_lock(
            odds_df,
            _games(("2024_W03_DAL@NYG", datetime(2024, 9, 21, 16, 30, tzinfo=ET))),
        )

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

        calc = MarketAnchorFeaturesCalculator()
        result = calc.select_snapshot_lines_at_lock(
            odds_df,
            _games(("2024_W03_DAL@NYG", datetime(2024, 9, 21, 16, 30, tzinfo=ET))),
        )

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


class TestTheCutoffIsPerGame:
    """Each game is cut at its OWN lock; one frame-wide cutoff could not satisfy both games."""

    def test_a_thursday_game_and_a_sunday_game_are_cut_at_different_locks(self):
        """The retired selection derived ONE Friday for the whole frame. Deleted by ruling.

        A Saturday-afternoon snapshot is after the Thursday game's Wednesday lock and before
        the Sunday game's Saturday lock, so it must be dropped for one and kept for the other.
        """
        thursday_game = "2024_W03_NE@NYJ"
        sunday_game = "2024_W03_DAL@BAL"
        saturday_snapshot = datetime(2024, 9, 21, 20, 0, 0, tzinfo=UTC)  # 16:00 EDT
        wednesday_snapshot = datetime(2024, 9, 18, 16, 0, 0, tzinfo=UTC)  # noon EDT

        odds_df = pd.DataFrame(
            [
                _make_odds_row(
                    thursday_game, "DraftKings", wednesday_snapshot, total=38.5
                ),
                _make_odds_row(
                    thursday_game, "DraftKings", saturday_snapshot, total=99.0
                ),
                _make_odds_row(
                    sunday_game, "DraftKings", wednesday_snapshot, total=47.0
                ),
                _make_odds_row(
                    sunday_game, "DraftKings", saturday_snapshot, total=48.5
                ),
            ]
        )
        games = _games(
            (thursday_game, datetime(2024, 9, 19, 20, 15, tzinfo=ET)),
            (sunday_game, datetime(2024, 9, 22, 16, 25, tzinfo=ET)),
        )

        result = (
            MarketAnchorFeaturesCalculator()
            .select_snapshot_lines_at_lock(odds_df, games)
            .set_index("game_id")
        )

        assert pd.to_datetime(result.loc[thursday_game, "snapshot_ts"], utc=True) == (
            pd.Timestamp(wednesday_snapshot)
        ), "the Thursday game took a snapshot captured after its own Wednesday lock"
        assert pd.to_datetime(result.loc[sunday_game, "snapshot_ts"], utc=True) == (
            pd.Timestamp(saturday_snapshot)
        ), "the Sunday game lost a snapshot captured before its own Saturday lock"
        assert (
            result.loc[thursday_game, "cutoff_time"]
            != result.loc[sunday_game, "cutoff_time"]
        )

    def test_a_wanted_game_with_no_kickoff_is_refused_by_name(self):
        from utils.game_lock import MissingKickoffError

        odds_df = pd.DataFrame(
            [
                _make_odds_row(
                    "2024_W03_DAL@BAL",
                    "DraftKings",
                    datetime(2024, 9, 18, 16, 0, 0, tzinfo=UTC),
                )
            ]
        )
        games = pd.DataFrame({"game_id": ["2024_W03_DAL@BAL"], "kickoff_et": [pd.NaT]})

        with pytest.raises(MissingKickoffError, match="2024_W03_DAL@BAL"):
            MarketAnchorFeaturesCalculator().select_snapshot_lines_at_lock(
                odds_df, games
            )
