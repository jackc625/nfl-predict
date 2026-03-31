"""Integration tests for Elo rating convergence across full historical data.

These tests process the full 2002-2024 dataset and verify that:
- Ratings converge to meaningful differentiation by 2018 (after 16 years burn-in)
- Known good/bad teams are ranked correctly (2007 Patriots top 3, 2017 Browns bottom 3)

All tests are marked @pytest.mark.slow as they process 6000+ games.
"""

import pytest

from ratings.elo import EloRatingSystem


def _load_all_games():
    """Load all games from silver layer.

    Returns:
        DataFrame with all available games, sorted by season.

    Raises:
        pytest.skip: If silver layer is not available.
    """
    try:
        from data.storage import load_dataframe

        games_df = load_dataframe("games", layer="silver")
        if len(games_df) == 0:
            pytest.skip("No games data in silver layer")
        return games_df
    except (FileNotFoundError, OSError, ValueError) as e:
        pytest.skip(f"Silver layer not available: {e}")


def _process_all_seasons(games_df):
    """Process all seasons chronologically through EloRatingSystem.

    Args:
        games_df: DataFrame with all games.

    Returns:
        Tuple of (EloRatingSystem, list of per-season rating DataFrames).
    """
    elo = EloRatingSystem()
    seasons = sorted(games_df["season"].unique())

    season_ratings = {}
    for season in seasons:
        season_games = games_df[games_df["season"] == season]
        elo.process_season_chronologically(season_games, season)
        # Snapshot end-of-season ratings
        season_ratings[season] = elo.get_current_ratings()

    return elo, season_ratings


@pytest.mark.slow
class TestEloConvergence:
    """Test that Elo ratings converge by 2018 after burn-in from 2002."""

    def test_elo_convergence_by_2018(self):
        """Process 2002-2024 games. By 2018, ratings should be meaningfully
        differentiated (std > 50) but not divergent (std < 200), with mean
        near 1500 +/- 50.
        """
        games_df = _load_all_games()

        # Verify we have data from 2002
        min_season = games_df["season"].min()
        if min_season > 2002:
            pytest.skip(
                f"Need data from 2002 for burn-in, earliest available: {min_season}"
            )

        elo, season_ratings = _process_all_seasons(games_df)

        # Check 2018 ratings
        assert 2018 in season_ratings, "No 2018 season ratings found"
        ratings_2018 = season_ratings[2018]

        # Rating standard deviation should show meaningful differentiation
        rating_std = ratings_2018["rating"].std()
        assert rating_std > 50, (
            f"Rating std should be > 50 by 2018 (differentiated), got {rating_std:.1f}"
        )
        assert rating_std < 200, (
            f"Rating std should be < 200 by 2018 (not divergent), got {rating_std:.1f}"
        )

        # Mean should be near 1500
        rating_mean = ratings_2018["rating"].mean()
        assert abs(rating_mean - 1500) < 50, (
            f"Mean rating should be near 1500, got {rating_mean:.1f}"
        )

    def test_known_team_fixtures(self):
        """After full 2002-2024 processing:
        - 2007 Patriots (16-0 regular season) rank top 3 by end-of-season Elo
        - 2017 Browns (0-16) rank bottom 3
        """
        games_df = _load_all_games()

        min_season = games_df["season"].min()
        if min_season > 2002:
            pytest.skip(
                f"Need data from 2002 for burn-in, earliest available: {min_season}"
            )

        elo, season_ratings = _process_all_seasons(games_df)

        # Check 2007 Patriots -- should be top 3
        assert 2007 in season_ratings, "No 2007 season ratings found"
        ratings_2007 = season_ratings[2007].reset_index(drop=True)
        # Ratings are sorted descending by default from get_current_ratings
        top_3_teams_2007 = ratings_2007.head(3)["team"].tolist()
        assert "NE" in top_3_teams_2007, (
            f"2007 Patriots (NE) should be top 3, but top 3 are: {top_3_teams_2007}. "
            f"NE rating: {ratings_2007[ratings_2007['team'] == 'NE']['rating'].values}"
        )

        # Check 2017 Browns -- should be bottom 3
        assert 2017 in season_ratings, "No 2017 season ratings found"
        ratings_2017 = season_ratings[2017].reset_index(drop=True)
        bottom_3_teams_2017 = ratings_2017.tail(3)["team"].tolist()
        assert "CLE" in bottom_3_teams_2017, (
            f"2017 Browns (CLE) should be bottom 3, but bottom 3 are: {bottom_3_teams_2017}. "
            f"CLE rating: {ratings_2017[ratings_2017['team'] == 'CLE']['rating'].values}"
        )
