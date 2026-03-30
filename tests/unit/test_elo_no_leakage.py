"""Tests for Elo snapshot integrity and leakage prevention.

Verifies:
- Frozen-rating property: pre-game Elo for game B is identical regardless
  of whether later games exist in the dataset
- Per-game snapshots contain pre-game (not post-game) ratings
- Snapshot DataFrame schema matches expected columns
- EloFeatureBuilder uses pre-computed snapshots (no recomputation)
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from ratings.elo import EloRatingSystem, is_divisional_game


def _make_test_games(n: int, season: int = 2023) -> pd.DataFrame:
    """Create n synthetic games with sequential dates.

    Uses canonical team abbreviations (KC, BUF, PHI, SF) and alternates
    matchups to avoid self-play.

    Args:
        n: Number of games to create.
        season: Season year for the games.

    Returns:
        DataFrame with game_id, season, week, home_team, away_team,
        home_score, away_score, kickoff_et columns.
    """
    matchups = [
        ("KC", "BUF", 28, 21),
        ("PHI", "SF", 24, 17),
        ("BUF", "PHI", 31, 28),
        ("SF", "KC", 20, 27),
        ("KC", "PHI", 35, 14),
        ("BUF", "SF", 21, 24),
    ]

    games = []
    for i in range(n):
        home, away, h_score, a_score = matchups[i % len(matchups)]
        games.append({
            "game_id": f"TEST_{season}_{i:02d}_{away}@{home}",
            "season": season,
            "week": (i // 2) + 1,
            "home_team": home,
            "away_team": away,
            "home_score": h_score,
            "away_score": a_score,
            "kickoff_et": datetime(season, 9, 7 + i, 13, 0),
        })
    return pd.DataFrame(games)


def _build_snapshots_from_games(games_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Process games through EloRatingSystem and return per-game pre-game snapshots.

    This is the core logic that build_elo.py should implement: for each game,
    capture pre-game ratings BEFORE updating with the game result.

    Args:
        games_df: DataFrame with game records.
        season: Season year.

    Returns:
        DataFrame with per-game pre-game snapshots.
    """
    elo = EloRatingSystem()
    elo.apply_season_carryover(season)

    games_sorted = games_df.sort_values("kickoff_et")
    snapshots = []

    for _, game in games_sorted.iterrows():
        if pd.isna(game["home_score"]) or pd.isna(game["away_score"]):
            continue

        home = game["home_team"]
        away = game["away_team"]

        # Capture PRE-GAME ratings
        home_rating = elo.get_or_create_rating(home, season)
        away_rating = elo.get_or_create_rating(away, season)

        divisional = is_divisional_game(home, away)
        prediction = elo.predict_game(home, away, season, is_divisional=divisional)

        snapshots.append({
            "game_id": game["game_id"],
            "season": season,
            "week": game["week"],
            "home_team": home,
            "away_team": away,
            "home_elo_pre": home_rating.rating,
            "away_elo_pre": away_rating.rating,
            "home_elo_uncertainty": home_rating.uncertainty,
            "away_elo_uncertainty": away_rating.uncertainty,
            "elo_prob_home": prediction["home_win_prob"],
            "hfa_used": prediction["hfa_used"],
        })

        # THEN update ratings with game result
        elo.update_ratings(
            home_team=home,
            away_team=away,
            home_score=int(game["home_score"]),
            away_score=int(game["away_score"]),
            season=season,
            game_date=game["kickoff_et"],
            game_id=game["game_id"],
            is_divisional=divisional,
        )

    return pd.DataFrame(snapshots)


class TestFrozenRatingProperty:
    """Verify that pre-game Elo for a game is invariant to later games."""

    def test_frozen_rating_property(self):
        """Pre-game Elo for game B is identical whether game C exists or not.

        Process games [A, B, C] and record pre-game Elo for B.
        Process games [A, B] only and record pre-game Elo for B.
        Both must be identical. Also verify game B's own result does not
        affect its pre-game Elo by comparing with processing only [A].
        """
        all_games = _make_test_games(3, season=2023)
        game_a = all_games.iloc[:1]
        games_ab = all_games.iloc[:2]
        games_abc = all_games.iloc[:3]

        game_b_id = all_games.iloc[1]["game_id"]

        # Process [A, B, C] -- get snapshots for all three
        snapshots_abc = _build_snapshots_from_games(games_abc, season=2023)
        b_elo_from_abc = snapshots_abc[snapshots_abc["game_id"] == game_b_id].iloc[0]

        # Process [A, B] only -- get snapshots for two
        snapshots_ab = _build_snapshots_from_games(games_ab, season=2023)
        b_elo_from_ab = snapshots_ab[snapshots_ab["game_id"] == game_b_id].iloc[0]

        # Pre-game Elo for B must be identical regardless of C's existence
        assert b_elo_from_abc["home_elo_pre"] == b_elo_from_ab["home_elo_pre"], (
            f"Home Elo for game B differs: "
            f"ABC={b_elo_from_abc['home_elo_pre']}, AB={b_elo_from_ab['home_elo_pre']}"
        )
        assert b_elo_from_abc["away_elo_pre"] == b_elo_from_ab["away_elo_pre"], (
            f"Away Elo for game B differs: "
            f"ABC={b_elo_from_abc['away_elo_pre']}, AB={b_elo_from_ab['away_elo_pre']}"
        )

        # Also verify game B's pre-game Elo equals post-game-A Elo
        # (i.e., B's own result does not affect its pre-game rating)
        # Process only [A], then check what B's team rating would be
        elo_after_a = EloRatingSystem()
        elo_after_a.apply_season_carryover(2023)
        game_a_row = game_a.iloc[0]
        home_a = game_a_row["home_team"]
        away_a = game_a_row["away_team"]
        elo_after_a.get_or_create_rating(home_a, 2023)
        elo_after_a.get_or_create_rating(away_a, 2023)
        elo_after_a.update_ratings(
            home_team=home_a,
            away_team=away_a,
            home_score=int(game_a_row["home_score"]),
            away_score=int(game_a_row["away_score"]),
            season=2023,
            game_date=game_a_row["kickoff_et"],
            game_id=game_a_row["game_id"],
        )

        # Get rating for game B's teams after only game A
        game_b_row = all_games.iloc[1]
        b_home = game_b_row["home_team"]
        b_away = game_b_row["away_team"]
        b_home_elo_after_a = elo_after_a.get_or_create_rating(b_home, 2023).rating
        b_away_elo_after_a = elo_after_a.get_or_create_rating(b_away, 2023).rating

        assert b_elo_from_ab["home_elo_pre"] == b_home_elo_after_a, (
            f"Game B's pre-game home Elo should equal rating after only game A: "
            f"snapshot={b_elo_from_ab['home_elo_pre']}, after_A={b_home_elo_after_a}"
        )
        assert b_elo_from_ab["away_elo_pre"] == b_away_elo_after_a, (
            f"Game B's pre-game away Elo should equal rating after only game A: "
            f"snapshot={b_elo_from_ab['away_elo_pre']}, after_A={b_away_elo_after_a}"
        )


class TestSnapshotPreGameNotPostGame:
    """Verify snapshots contain pre-game, not post-game ratings."""

    def test_snapshot_has_pre_game_not_post_game(self):
        """After building snapshots for 3 games, the snapshot for game 2
        must reflect ratings AFTER game 1 but BEFORE game 2 result.

        Verify by checking that snapshot[game2].home_elo_pre differs from
        snapshot[game3].home_elo_pre (because game 2 result changed ratings).
        """
        # Use 3 games where the same team plays in games 2 and 3
        # so we can see the rating change
        games = pd.DataFrame([
            {
                "game_id": "TEST_G1_BUF@KC",
                "season": 2023,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "home_score": 28,
                "away_score": 21,
                "kickoff_et": datetime(2023, 9, 7, 13, 0),
            },
            {
                "game_id": "TEST_G2_PHI@KC",
                "season": 2023,
                "week": 2,
                "home_team": "KC",
                "away_team": "PHI",
                "home_score": 35,
                "away_score": 14,
                "kickoff_et": datetime(2023, 9, 14, 13, 0),
            },
            {
                "game_id": "TEST_G3_SF@KC",
                "season": 2023,
                "week": 3,
                "home_team": "KC",
                "away_team": "SF",
                "home_score": 24,
                "away_score": 17,
                "kickoff_et": datetime(2023, 9, 21, 13, 0),
            },
        ])

        snapshots = _build_snapshots_from_games(games, season=2023)

        snap_g2 = snapshots[snapshots["game_id"] == "TEST_G2_PHI@KC"].iloc[0]
        snap_g3 = snapshots[snapshots["game_id"] == "TEST_G3_SF@KC"].iloc[0]

        # KC's pre-game Elo should differ between game 2 and game 3
        # because game 2 result (KC won big) changed KC's rating
        assert snap_g2["home_elo_pre"] != snap_g3["home_elo_pre"], (
            f"KC's pre-game Elo should change between game 2 and 3: "
            f"G2={snap_g2['home_elo_pre']}, G3={snap_g3['home_elo_pre']}"
        )

        # KC won games 1 and 2, so their rating should increase
        assert snap_g3["home_elo_pre"] > snap_g2["home_elo_pre"], (
            f"KC won game 2, so their pre-game Elo for game 3 should be higher: "
            f"G2={snap_g2['home_elo_pre']}, G3={snap_g3['home_elo_pre']}"
        )


class TestSnapshotSchema:
    """Verify snapshot DataFrame has the expected schema."""

    def test_snapshot_schema(self):
        """elo_game_snapshots DataFrame has the required columns."""
        games = _make_test_games(3, season=2023)
        snapshots = _build_snapshots_from_games(games, season=2023)

        expected_columns = {
            "game_id",
            "season",
            "week",
            "home_team",
            "away_team",
            "home_elo_pre",
            "away_elo_pre",
            "home_elo_uncertainty",
            "away_elo_uncertainty",
            "elo_prob_home",
            "hfa_used",
        }

        actual_columns = set(snapshots.columns)
        missing = expected_columns - actual_columns
        assert not missing, f"Missing columns in snapshot schema: {missing}"

        # Verify types are numeric for rating columns
        for col in ["home_elo_pre", "away_elo_pre", "elo_prob_home", "hfa_used"]:
            assert pd.api.types.is_numeric_dtype(snapshots[col]), (
                f"Column {col} should be numeric, got {snapshots[col].dtype}"
            )

        # Verify no NaN in key columns
        for col in ["game_id", "home_elo_pre", "away_elo_pre"]:
            assert snapshots[col].notna().all(), (
                f"Column {col} should have no NaN values"
            )


class TestFeatureBuilderUsesSnapshots:
    """Verify EloFeatureBuilder uses pre-computed snapshots instead of recomputing."""

    def test_feature_builder_uses_snapshots_not_recomputation(self):
        """EloFeatureBuilder.build_features returns features by looking up
        pre-computed snapshots (mock load_dataframe to verify it reads
        elo_game_snapshots).
        """
        # Create mock snapshot data that would be in the silver layer
        mock_snapshots = pd.DataFrame([
            {
                "game_id": "2023_01_BUF_KC",
                "season": 2023,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "home_elo_pre": 1550.0,
                "away_elo_pre": 1580.0,
                "home_elo_uncertainty": 120.0,
                "away_elo_uncertainty": 115.0,
                "elo_prob_home": 0.48,
                "hfa_used": 48.0,
            },
            {
                "game_id": "2023_01_PHI_SF",
                "season": 2023,
                "week": 1,
                "home_team": "SF",
                "away_team": "PHI",
                "home_elo_pre": 1520.0,
                "away_elo_pre": 1560.0,
                "home_elo_uncertainty": 130.0,
                "away_elo_uncertainty": 110.0,
                "elo_prob_home": 0.45,
                "hfa_used": 48.0,
            },
        ])

        # Create games_df that matches the snapshots
        games_df = pd.DataFrame([
            {
                "game_id": "2023_01_BUF_KC",
                "season": 2023,
                "week": 1,
                "home_team": "KC",
                "away_team": "BUF",
                "home_score": 28,
                "away_score": 21,
                "kickoff_et": datetime(2023, 9, 7, 13, 0),
            },
            {
                "game_id": "2023_01_PHI_SF",
                "season": 2023,
                "week": 1,
                "home_team": "SF",
                "away_team": "PHI",
                "home_score": 24,
                "away_score": 17,
                "kickoff_et": datetime(2023, 9, 7, 16, 25),
            },
        ])

        with patch("features.elo_features.load_dataframe") as mock_load:
            mock_load.return_value = mock_snapshots

            from features.elo_features import EloFeatureBuilder

            builder = EloFeatureBuilder()
            result = builder.build_features(
                games_df,
                as_of_datetime=datetime(2023, 9, 8, 0, 0),
            )

            # Verify load_dataframe was called with elo_game_snapshots
            mock_load.assert_called_once_with("elo_game_snapshots", layer="silver")

            # Verify the result has the expected Elo feature columns
            from features.elo_features import ELO_FEATURE_COLUMNS

            for col in ELO_FEATURE_COLUMNS:
                assert col in result.columns, f"Missing Elo feature column: {col}"

            # Verify features come from snapshots (not recomputed from 1500)
            kc_game = result[result["game_id"] == "2023_01_BUF_KC"].iloc[0]
            assert kc_game["home_elo"] == 1550.0, (
                f"home_elo should come from snapshot (1550.0), got {kc_game['home_elo']}"
            )
            assert kc_game["away_elo"] == 1580.0, (
                f"away_elo should come from snapshot (1580.0), got {kc_game['away_elo']}"
            )

        # Verify the builder does NOT have an EloRatingSystem internally
        # (it should use snapshots, not recompute)
        builder2 = EloFeatureBuilder()
        assert not hasattr(builder2, "elo_system"), (
            "EloFeatureBuilder should not have self.elo_system "
            "(should use snapshot lookup, not recomputation)"
        )
