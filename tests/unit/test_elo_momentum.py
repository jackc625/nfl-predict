"""Tests for Elo momentum and rank/percentile features.

Verifies:
- Momentum (4-game lookback): Elo change per game for home/away teams
- NaN for week 1 (no prior games), partial lookback for early weeks
- Rank (1-32) and percentile (0-1): within-season relative positioning
- Updated ELO_FEATURE_COLUMNS contains all 14 columns
- build_features returns all new columns
"""

from datetime import datetime
from unittest.mock import patch

import pandas as pd


def _make_snapshot_df(games_data: list[dict]) -> pd.DataFrame:
    """Create a DataFrame matching elo_game_snapshots schema.

    Args:
        games_data: List of dicts with game_id, season, week, home_team,
            away_team, home_elo_pre, away_elo_pre and optional fields.

    Returns:
        DataFrame matching elo_game_snapshots silver layer schema.
    """
    defaults = {
        "home_elo_uncertainty": 100.0,
        "away_elo_uncertainty": 100.0,
        "elo_prob_home": 0.50,
        "hfa_used": 48.0,
    }
    rows = []
    for g in games_data:
        row = {**defaults, **g}
        rows.append(row)
    return pd.DataFrame(rows)


def _make_games_df(games_data: list[dict]) -> pd.DataFrame:
    """Create a games DataFrame for build_features input.

    Args:
        games_data: List of dicts with game_id, season, week, home_team,
            away_team columns. kickoff_et is auto-generated if not provided.

    Returns:
        DataFrame suitable for EloFeatureBuilder.build_features().
    """
    rows = []
    for g in games_data:
        row = {**g}
        if "kickoff_et" not in row:
            row["kickoff_et"] = datetime(row["season"], 9, 7 + row["week"], 13, 0)
        rows.append(row)
    return pd.DataFrame(rows)


# ---- Fixtures ----

def _momentum_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixture with 5+ games for KC and BUF across multiple weeks.

    KC plays weeks 1-5, BUF plays weeks 1-5, with known Elo progressions:
    - KC: 1500 -> 1520 -> 1540 -> 1560 -> 1580 (steady increase)
    - BUF: 1500 -> 1480 -> 1460 -> 1440 -> 1420 (steady decrease)
    """
    snapshot_data = [
        # Week 1
        {"game_id": "2023_01_BUF_KC", "season": 2023, "week": 1,
         "home_team": "KC", "away_team": "BUF",
         "home_elo_pre": 1500.0, "away_elo_pre": 1500.0},
        {"game_id": "2023_01_PHI_SF", "season": 2023, "week": 1,
         "home_team": "SF", "away_team": "PHI",
         "home_elo_pre": 1500.0, "away_elo_pre": 1500.0},
        # Week 2
        {"game_id": "2023_02_PHI_KC", "season": 2023, "week": 2,
         "home_team": "KC", "away_team": "PHI",
         "home_elo_pre": 1520.0, "away_elo_pre": 1480.0},
        {"game_id": "2023_02_SF_BUF", "season": 2023, "week": 2,
         "home_team": "BUF", "away_team": "SF",
         "home_elo_pre": 1480.0, "away_elo_pre": 1520.0},
        # Week 3
        {"game_id": "2023_03_SF_KC", "season": 2023, "week": 3,
         "home_team": "KC", "away_team": "SF",
         "home_elo_pre": 1540.0, "away_elo_pre": 1480.0},
        {"game_id": "2023_03_PHI_BUF", "season": 2023, "week": 3,
         "home_team": "BUF", "away_team": "PHI",
         "home_elo_pre": 1460.0, "away_elo_pre": 1520.0},
        # Week 4
        {"game_id": "2023_04_BUF_KC", "season": 2023, "week": 4,
         "home_team": "KC", "away_team": "BUF",
         "home_elo_pre": 1560.0, "away_elo_pre": 1440.0},
        {"game_id": "2023_04_PHI_SF", "season": 2023, "week": 4,
         "home_team": "SF", "away_team": "PHI",
         "home_elo_pre": 1440.0, "away_elo_pre": 1560.0},
        # Week 5
        {"game_id": "2023_05_PHI_KC", "season": 2023, "week": 5,
         "home_team": "KC", "away_team": "PHI",
         "home_elo_pre": 1580.0, "away_elo_pre": 1540.0},
        {"game_id": "2023_05_SF_BUF", "season": 2023, "week": 5,
         "home_team": "BUF", "away_team": "SF",
         "home_elo_pre": 1420.0, "away_elo_pre": 1460.0},
    ]

    snapshots = _make_snapshot_df(snapshot_data)

    # Games matching the snapshots (for build_features input)
    games_data = [
        {"game_id": g["game_id"], "season": g["season"], "week": g["week"],
         "home_team": g["home_team"], "away_team": g["away_team"]}
        for g in snapshot_data
    ]
    games = _make_games_df(games_data)

    return snapshots, games


def _rank_fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create fixture with all 32 teams having at least 1 game.

    Teams are arranged with known Elo values for ranking verification.
    Top team has Elo 1600, bottom team has Elo 1280, linear spacing.
    """
    teams = [
        "KC", "BUF", "SF", "PHI", "DAL", "MIA", "BAL", "CIN",
        "DET", "JAX", "CLE", "HOU", "NYJ", "LAC", "PIT", "DEN",
        "MIN", "GB", "SEA", "LA", "TB", "ATL", "NO", "CHI",
        "IND", "TEN", "LV", "NYG", "WAS", "CAR", "ARI", "NE",
    ]

    # Each team gets a distinct Elo: top = 1600, bottom = 1280
    elos = {team: 1600.0 - i * 10.0 for i, team in enumerate(teams)}

    snapshot_data = []
    games_data = []

    # Create 16 games (32 teams, each plays once)
    for i in range(16):
        home = teams[i]
        away = teams[31 - i]
        game_id = f"2023_01_{away}_{home}"
        snapshot_data.append({
            "game_id": game_id, "season": 2023, "week": 1,
            "home_team": home, "away_team": away,
            "home_elo_pre": elos[home], "away_elo_pre": elos[away],
        })
        games_data.append({
            "game_id": game_id, "season": 2023, "week": 1,
            "home_team": home, "away_team": away,
        })

    # Add week 5 games to check rank evolution
    # After several weeks, some teams have changed Elo
    adjusted_elos = elos.copy()
    adjusted_elos["NE"] = 1550.0  # NE improved dramatically
    adjusted_elos["KC"] = 1580.0  # KC slightly worse

    for i in range(16):
        home = teams[i]
        away = teams[31 - i]
        game_id = f"2023_05_{away}_{home}"
        snapshot_data.append({
            "game_id": game_id, "season": 2023, "week": 5,
            "home_team": home, "away_team": away,
            "home_elo_pre": adjusted_elos[home],
            "away_elo_pre": adjusted_elos[away],
        })
        games_data.append({
            "game_id": game_id, "season": 2023, "week": 5,
            "home_team": home, "away_team": away,
        })

    snapshots = _make_snapshot_df(snapshot_data)
    games = _make_games_df(games_data)
    return snapshots, games


# ---- Test Classes ----

class TestMomentumBasic:
    """Test basic momentum computation."""

    def test_momentum_basic(self):
        """KC has Elo [1500, 1520, 1540, 1560] over 4 games.
        Momentum before game 5 = (1560 - 1500) / 4 = 15.0.
        """
        snapshots, games = _momentum_fixture()

        # Patch at the module level for the refactored EloFeatureBuilder
        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        with patch.object(elo_mod, "load_dataframe", return_value=snapshots):
            builder = elo_mod.EloFeatureBuilder()
            # Get features for week 5 KC home game
            week5_games = games[games["week"] == 5]
            result = builder.build_features(
                week5_games,
                as_of_datetime=datetime(2023, 10, 15, 13, 0),
            )

        kc_game = result[result["game_id"] == "2023_05_PHI_KC"].iloc[0]
        # KC played weeks 1-4 with Elo: 1500, 1520, 1540, 1560
        # Momentum = (1560 - 1500) / 4 = 15.0
        assert abs(kc_game["home_elo_momentum"] - 15.0) < 0.01, (
            f"KC momentum should be 15.0, got {kc_game['home_elo_momentum']}"
        )


class TestMomentumNanWeek1:
    """Test NaN momentum for week 1 (no prior games)."""

    def test_momentum_nan_week1(self):
        """A team's first game of the season returns NaN for momentum.
        Exactly 32 NaN per season for week 1 (one per team, but we have
        4 teams * 2 home+away = 4 NaN values in our fixture).
        """
        snapshots, games = _momentum_fixture()

        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        with patch.object(elo_mod, "load_dataframe", return_value=snapshots):
            builder = elo_mod.EloFeatureBuilder()
            week1_games = games[games["week"] == 1]
            result = builder.build_features(
                week1_games,
                as_of_datetime=datetime(2023, 9, 10, 13, 0),
            )

        # All week 1 teams should have NaN momentum
        assert result["home_elo_momentum"].isna().all(), (
            "Week 1 home teams should have NaN momentum"
        )
        assert result["away_elo_momentum"].isna().all(), (
            "Week 1 away teams should have NaN momentum"
        )


class TestMomentumFewerThanLookback:
    """Test momentum with fewer than lookback games."""

    def test_momentum_fewer_than_lookback(self):
        """A team with only 2 prior games uses those 2 games.
        Momentum = (newest_elo - oldest_elo) / 2.
        """
        snapshots, games = _momentum_fixture()

        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        with patch.object(elo_mod, "load_dataframe", return_value=snapshots):
            builder = elo_mod.EloFeatureBuilder()
            # Week 3: KC has 2 prior games (weeks 1, 2)
            week3_games = games[games["week"] == 3]
            result = builder.build_features(
                week3_games,
                as_of_datetime=datetime(2023, 9, 25, 13, 0),
            )

        kc_game = result[result["game_id"] == "2023_03_SF_KC"].iloc[0]
        # KC: week 1 = 1500, week 2 = 1520. Momentum = (1520 - 1500) / 2 = 10.0
        assert abs(kc_game["home_elo_momentum"] - 10.0) < 0.01, (
            f"KC momentum with 2 games should be 10.0, got {kc_game['home_elo_momentum']}"
        )


class TestMomentumSign:
    """Test momentum sign reflects team trend direction."""

    def test_momentum_sign_reflects_trend(self):
        """Winning streak team (KC, Elo increasing) has positive momentum.
        Losing streak team (BUF, Elo decreasing) has negative momentum.
        """
        snapshots, games = _momentum_fixture()

        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        with patch.object(elo_mod, "load_dataframe", return_value=snapshots):
            builder = elo_mod.EloFeatureBuilder()
            week5_games = games[games["week"] == 5]
            result = builder.build_features(
                week5_games,
                as_of_datetime=datetime(2023, 10, 15, 13, 0),
            )

        kc_game = result[result["game_id"] == "2023_05_PHI_KC"].iloc[0]
        buf_game = result[result["game_id"] == "2023_05_SF_BUF"].iloc[0]

        assert kc_game["home_elo_momentum"] > 0, (
            f"KC (winning streak) should have positive momentum, got {kc_game['home_elo_momentum']}"
        )
        assert buf_game["home_elo_momentum"] < 0, (
            f"BUF (losing streak) should have negative momentum, got {buf_game['home_elo_momentum']}"
        )


class TestRankFeaturesRange:
    """Test rank and percentile feature value ranges."""

    def test_rank_features_range(self):
        """home_elo_rank and away_elo_rank are in [1, 32].
        home_elo_percentile and away_elo_percentile are in [0.0, 1.0].
        """
        snapshots, games = _rank_fixture()

        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        with patch.object(elo_mod, "load_dataframe", return_value=snapshots):
            builder = elo_mod.EloFeatureBuilder()
            result = builder.build_features(
                games,
                as_of_datetime=datetime(2023, 10, 15, 13, 0),
            )

        # Filter out rows where rank might be NaN (shouldn't happen with our fixture)
        valid = result.dropna(subset=["home_elo_rank", "away_elo_rank"])
        assert len(valid) > 0, "Should have valid rank data"

        assert valid["home_elo_rank"].min() >= 1, (
            f"Min home rank should be >= 1, got {valid['home_elo_rank'].min()}"
        )
        assert valid["home_elo_rank"].max() <= 32, (
            f"Max home rank should be <= 32, got {valid['home_elo_rank'].max()}"
        )
        assert valid["away_elo_rank"].min() >= 1
        assert valid["away_elo_rank"].max() <= 32

        assert valid["home_elo_percentile"].min() >= 0.0
        assert valid["home_elo_percentile"].max() <= 1.0
        assert valid["away_elo_percentile"].min() >= 0.0
        assert valid["away_elo_percentile"].max() <= 1.0


class TestRankTopTeam:
    """Test top-rated team gets rank 1."""

    def test_rank_top_team_is_rank_1(self):
        """The team with the highest pre-game Elo has rank 1 and percentile 1.0."""
        snapshots, games = _rank_fixture()

        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        with patch.object(elo_mod, "load_dataframe", return_value=snapshots):
            builder = elo_mod.EloFeatureBuilder()
            week1_games = games[games["week"] == 1]
            result = builder.build_features(
                week1_games,
                as_of_datetime=datetime(2023, 9, 15, 13, 0),
            )

        # KC has the highest Elo (1600) in week 1
        kc_game = result[result["home_team"] == "KC"].iloc[0]
        assert kc_game["home_elo_rank"] == 1, (
            f"KC (top Elo) should have rank 1, got {kc_game['home_elo_rank']}"
        )
        assert abs(kc_game["home_elo_percentile"] - 1.0) < 0.01, (
            f"KC should have percentile ~1.0, got {kc_game['home_elo_percentile']}"
        )


class TestRankVaryAcrossSeason:
    """Test rank changes over the season."""

    def test_rank_features_vary_across_season(self):
        """A team's rank changes from early season to late season."""
        snapshots, games = _rank_fixture()

        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        with patch.object(elo_mod, "load_dataframe", return_value=snapshots):
            builder = elo_mod.EloFeatureBuilder()
            result = builder.build_features(
                games,
                as_of_datetime=datetime(2023, 10, 15, 13, 0),
            )

        # NE starts at rank 32 (Elo 1280) in week 1, jumps to ~1550 in week 5
        # Find NE's games
        ne_w1 = result[(result["week"] == 1) & (result["away_team"] == "NE")]
        ne_w5 = result[(result["week"] == 5) & (result["away_team"] == "NE")]

        if len(ne_w1) > 0 and len(ne_w5) > 0:
            rank_w1 = ne_w1.iloc[0]["away_elo_rank"]
            rank_w5 = ne_w5.iloc[0]["away_elo_rank"]
            assert rank_w1 != rank_w5, (
                f"NE's rank should change: week 1={rank_w1}, week 5={rank_w5}"
            )
            # NE started last (32) and improved significantly
            assert rank_w5 < rank_w1, (
                f"NE improved, rank should decrease: week 1={rank_w1}, week 5={rank_w5}"
            )


class TestEloFeatureColumnsUpdated:
    """Test ELO_FEATURE_COLUMNS has all 14 columns."""

    def test_elo_feature_columns_updated(self):
        """ELO_FEATURE_COLUMNS contains all 14 columns: 8 original + 6 new."""
        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        cols = elo_mod.ELO_FEATURE_COLUMNS

        # Original 8
        assert "home_elo" in cols
        assert "away_elo" in cols
        assert "elo_diff" in cols
        assert "elo_prob_home" in cols
        assert "elo_prob_away" in cols
        assert "hfa_used" in cols
        assert "home_elo_uncertainty" in cols
        assert "away_elo_uncertainty" in cols

        # New 6
        assert "home_elo_momentum" in cols, "Missing home_elo_momentum"
        assert "away_elo_momentum" in cols, "Missing away_elo_momentum"
        assert "home_elo_rank" in cols, "Missing home_elo_rank"
        assert "away_elo_rank" in cols, "Missing away_elo_rank"
        assert "home_elo_percentile" in cols, "Missing home_elo_percentile"
        assert "away_elo_percentile" in cols, "Missing away_elo_percentile"

        assert len(cols) == 14, f"Expected 14 columns, got {len(cols)}: {cols}"


class TestBuildFeaturesReturnsAllNewColumns:
    """Test build_features output has all new columns."""

    def test_build_features_returns_all_new_columns(self):
        """build_features output DataFrame has all new columns present."""
        snapshots, games = _momentum_fixture()

        import importlib
        import sys
        from pathlib import Path

        import features.elo_features as elo_mod
        if not hasattr(elo_mod.EloFeatureBuilder, "_load_snapshots"):
            worktree_root = str(Path(__file__).resolve().parent.parent.parent)
            for key in list(sys.modules.keys()):
                if key.startswith("features.elo") or key == "features":
                    del sys.modules[key]
            if worktree_root in sys.path:
                sys.path.remove(worktree_root)
            sys.path.insert(0, worktree_root)
            elo_mod = importlib.import_module("features.elo_features")

        with patch.object(elo_mod, "load_dataframe", return_value=snapshots):
            builder = elo_mod.EloFeatureBuilder()
            result = builder.build_features(
                games,
                as_of_datetime=datetime(2023, 10, 15, 13, 0),
            )

        expected_new_cols = [
            "home_elo_momentum", "away_elo_momentum",
            "home_elo_rank", "away_elo_rank",
            "home_elo_percentile", "away_elo_percentile",
        ]
        for col in expected_new_cols:
            assert col in result.columns, f"Missing column in result: {col}"
