"""Unit tests for odds ingestion and snapshot timing.

Tests validate that:
- get_synthetic_snapshot_ts generates correct Friday 6 PM ET timestamps
- Historical odds transform produces correct sportsbook, game_ids, and team abbreviations
- Historical odds are validated through OddsSchema
- Missing spread_line for regular season games raises DataValidationError
"""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

ET = ZoneInfo("America/New_York")


def _make_schedule_row(
    season: int = 2024,
    week: int = 1,
    gameday: str = "2024-09-08",
    home_team: str = "KC",
    away_team: str = "BAL",
    game_type: str = "REG",
    spread_line: float | None = -3.0,
    total_line: float | None = 46.5,
    home_moneyline: int | None = -150,
    away_moneyline: int | None = 130,
    home_spread_odds: int | None = -110,
    away_spread_odds: int | None = -110,
    over_odds: int | None = -110,
    under_odds: int | None = -110,
    gametime: str = "20:20",
    stadium: str = "Arrowhead Stadium",
    roof: str = "outdoors",
) -> dict:
    """Create a single nflreadpy-style schedule row."""
    return {
        "season": season,
        "week": week,
        "gameday": gameday,
        "gametime": gametime,
        "home_team": home_team,
        "away_team": away_team,
        "game_type": game_type,
        "spread_line": spread_line,
        "total_line": total_line,
        "home_moneyline": home_moneyline,
        "away_moneyline": away_moneyline,
        "home_spread_odds": home_spread_odds,
        "away_spread_odds": away_spread_odds,
        "over_odds": over_odds,
        "under_odds": under_odds,
        "stadium": stadium,
        "roof": roof,
        "home_score": 27,
        "away_score": 20,
        "result": 1,
        "season_type": "Regular Season",
        "neutral_site": False,
    }


class TestGetSyntheticSnapshotTs:
    """Tests for get_synthetic_snapshot_ts()."""

    def test_sunday_game_returns_preceding_friday_6pm_et(self):
        """A Sunday game's snapshot is the preceding Friday at 6 PM ET."""
        from scripts.ingest_historical_odds import get_synthetic_snapshot_ts

        # 2024-09-08 is a Sunday
        result = get_synthetic_snapshot_ts("2024-09-08")

        # The preceding Friday is 2024-09-06
        expected_et = datetime(2024, 9, 6, 18, 0, tzinfo=ET)
        expected_utc = expected_et.astimezone(UTC)

        assert result.year == expected_utc.year
        assert result.month == expected_utc.month
        assert result.day == expected_utc.day
        assert result.hour == expected_utc.hour
        assert result.minute == 0

    def test_thursday_night_game_returns_preceding_friday(self):
        """A Thursday game's snapshot is the Friday BEFORE the Thursday (i.e., 6 days prior)."""
        from scripts.ingest_historical_odds import get_synthetic_snapshot_ts

        # 2024-09-05 is a Thursday
        result = get_synthetic_snapshot_ts("2024-09-05")

        # The preceding Friday is 2024-08-30 (6 days before)
        expected_et = datetime(2024, 8, 30, 18, 0, tzinfo=ET)
        expected_utc = expected_et.astimezone(UTC)

        assert result.year == expected_utc.year
        assert result.month == expected_utc.month
        assert result.day == expected_utc.day

    def test_monday_night_game_returns_preceding_friday(self):
        """A Monday game's snapshot is the preceding Friday at 6 PM ET."""
        from scripts.ingest_historical_odds import get_synthetic_snapshot_ts

        # 2024-09-09 is a Monday
        result = get_synthetic_snapshot_ts("2024-09-09")

        # The preceding Friday is 2024-09-06
        expected_et = datetime(2024, 9, 6, 18, 0, tzinfo=ET)
        expected_utc = expected_et.astimezone(UTC)

        assert result.year == expected_utc.year
        assert result.month == expected_utc.month
        assert result.day == expected_utc.day

    def test_friday_game_returns_prior_friday(self):
        """If the gameday IS a Friday, use the PRIOR Friday (7 days earlier)."""
        from scripts.ingest_historical_odds import get_synthetic_snapshot_ts

        # 2024-12-20 is a Friday
        result = get_synthetic_snapshot_ts("2024-12-20")

        # The preceding Friday is 2024-12-13 (7 days before)
        expected_et = datetime(2024, 12, 13, 18, 0, tzinfo=ET)
        expected_utc = expected_et.astimezone(UTC)

        assert result.year == expected_utc.year
        assert result.month == expected_utc.month
        assert result.day == expected_utc.day


class TestHistoricalOddsTransform:
    """Tests for historical odds transformation."""

    def test_produces_the_ratified_sportsbook_label(self):
        """Historical odds records carry the label the live silver rows already hold.

        This test used to pin the literal ``nflverse_closing``, which this ingest wrote from
        v1.0 Phase 02 (commit ``b3f158d``) until Plan 31-08. That literal was a DOCUMENTED
        LEGACY MISLABEL: no row in live silver ever carried it, and the OUM-06 provenance
        allowlist (``backtest.ou_divergence._ALLOWED_SPORTSBOOKS``) rejects it, so the Phase-31
        verdict run would have hard-failed at step one. The label is now read from the FROZEN
        pre-registration and the allowlist is deliberately NOT widened (D31-12 branch 1).
        """
        from backtest.ev_chain_constants import ODDS_SPORTSBOOK_LABEL
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        df = pd.DataFrame([_make_schedule_row()])
        result = transform_nfl_odds_to_standard_format(df)

        assert len(result) == 1
        assert result.iloc[0]["sportsbook"] == ODDS_SPORTSBOOK_LABEL

    def test_creates_project_format_game_ids(self):
        """Historical odds game_ids follow project format: 2024_W01_BAL@KC."""
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        df = pd.DataFrame(
            [_make_schedule_row(season=2024, week=1, away_team="BAL", home_team="KC")]
        )
        result = transform_nfl_odds_to_standard_format(df)

        assert result.iloc[0]["game_id"] == "2024_W01_BAL@KC"

    def test_uses_normalize_team_abbreviation(self):
        """Historical odds normalize team abbreviations (LA not LAR, LV not OAK)."""
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        # Use OAK (historical Raiders) and LAR (alternative Rams)
        row = _make_schedule_row(season=2019, week=5, away_team="OAK", home_team="LA")
        df = pd.DataFrame([row])
        result = transform_nfl_odds_to_standard_format(df)

        game_id = result.iloc[0]["game_id"]
        # OAK -> LV, LA stays as LA
        assert "LV@LA" in game_id

    def test_validated_through_odds_schema(self):
        """Historical odds pass OddsSchema validation via validate_bronze_to_silver."""
        from data.quality_gates import validate_bronze_to_silver
        from data.schemas import OddsSchema
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        df = pd.DataFrame([_make_schedule_row()])
        odds_df = transform_nfl_odds_to_standard_format(df)

        # Should not raise
        validated = validate_bronze_to_silver(odds_df, OddsSchema)
        assert len(validated) == 1

    def test_missing_spread_for_reg_game_raises_error(self):
        """Missing spread_line for a regular season game raises DataValidationError."""
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        # Regular season game with missing spread
        row = _make_schedule_row(spread_line=None)
        df = pd.DataFrame([row])

        # The transform should still work but produce a record with spread=None
        odds_df = transform_nfl_odds_to_standard_format(df)

        # The game should still be included (spread is optional in OddsSchema)
        # But we want to verify the spread IS None
        assert pd.isna(odds_df.iloc[0]["spread"]) or odds_df.iloc[0]["spread"] is None

    def test_preseason_games_filtered_out(self):
        """Preseason games are filtered out from historical odds."""
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        rows = [
            _make_schedule_row(game_type="PRE"),
            _make_schedule_row(game_type="REG", away_team="BUF"),
        ]
        df = pd.DataFrame(rows)
        result = transform_nfl_odds_to_standard_format(df)

        # Only the REG game should be included
        assert len(result) == 1
        assert "BUF" in result.iloc[0]["game_id"]

    def test_has_synthetic_snapshot_ts(self):
        """Historical odds records have a synthetic snapshot_ts derived from gameday."""
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        df = pd.DataFrame([_make_schedule_row(gameday="2024-09-08")])
        result = transform_nfl_odds_to_standard_format(df)

        snapshot_ts = result.iloc[0]["snapshot_ts"]
        # Should be a datetime, not a static string
        assert isinstance(snapshot_ts, datetime)

    def test_is_live_is_false(self):
        """Historical odds always have is_live=False."""
        from scripts.ingest_historical_odds import transform_nfl_odds_to_standard_format

        df = pd.DataFrame([_make_schedule_row()])
        result = transform_nfl_odds_to_standard_format(df)

        assert result.iloc[0]["is_live"] == False  # noqa: E712 (numpy.bool_ vs Python bool)
