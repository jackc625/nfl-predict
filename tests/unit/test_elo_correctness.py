"""Tests for Elo rating system correctness.

Verifies:
- HFA initialization at 48 (not 65)
- Prior-season-only HFA learning (no lookahead)
- 75/25 season carryover formula
- Divisional game detection
- Divisional HFA reduction at 0.54 factor
- Chronological update ordering
- HFA clamp range 20-80
"""

from datetime import datetime

import pandas as pd

from ratings.elo import (
    DIVISIONAL_HFA_FACTOR,
    EloRatingSystem,
    is_divisional_game,
)


def _make_games_df(games: list[dict]) -> pd.DataFrame:
    """Helper to create a games DataFrame from a list of game dicts."""
    return pd.DataFrame(games)


def _make_season_games(
    season: int,
    num_games: int = 10,
    home_win_rate: float = 0.6,
) -> pd.DataFrame:
    """Create synthetic season games with a controlled home win rate.

    Args:
        season: Season year
        num_games: Number of games to create
        home_win_rate: Fraction of games where home team wins
    """
    games = []
    home_wins = int(num_games * home_win_rate)

    for i in range(num_games):
        home_score = 28 if i < home_wins else 17
        away_score = 17 if i < home_wins else 28

        games.append(
            {
                "game_id": f"SYNTH_{season}_G{i:02d}",
                "season": season,
                "week": (i // 2) + 1,
                "home_team": "MIA" if i % 2 == 0 else "DEN",
                "away_team": "BUF" if i % 2 == 0 else "KC",
                "home_score": home_score,
                "away_score": away_score,
                "kickoff_et": datetime(season, 9, 7 + i, 13, 0),
            }
        )

    return pd.DataFrame(games)


class TestEloInitialization:
    """Tests for Elo system parameter initialization."""

    def test_hfa_init_is_48(self):
        """Test 4: EloRatingSystem initializes with hfa_init=48.0."""
        elo = EloRatingSystem()
        assert elo.hfa_init == 48.0, f"Expected hfa_init=48.0, got {elo.hfa_init}"

    def test_default_season_carryover(self):
        """Season carryover should default to 0.75."""
        elo = EloRatingSystem()
        assert elo.season_carryover == 0.75


class TestHFALearning:
    """Tests for home field advantage learning."""

    def test_hfa_uses_prior_season_only(self):
        """Test 5: learn_home_field_advantage uses prior-season games only.

        Calling with season=2024 should use season=2023 data and return
        a non-default HFA value.
        """
        # Create games for BOTH seasons 2023 and 2024
        games_2023 = _make_season_games(2023, num_games=20, home_win_rate=0.65)
        games_2024 = _make_season_games(2024, num_games=20, home_win_rate=0.40)

        all_games = pd.concat([games_2023, games_2024], ignore_index=True)

        elo = EloRatingSystem()
        learned_hfa = elo.learn_home_field_advantage(all_games, season=2024)

        # HFA should be based on 2023 data (65% home win rate), not 2024 (40%)
        # 65% home win rate -> (0.65 - 0.5) * 800 = 120 -> clamped to 80
        # Blended: 0.7 * 80 + 0.3 * 48 = 56 + 14.4 = 70.4 -> clamped to 70.4
        # The exact value depends on blending but should be > 48 (since 2023 was 65%)
        assert learned_hfa > 48.0, (
            f"Expected HFA > 48 (from 2023 data with 65% home win rate), got {learned_hfa}"
        )

        # If it were using 2024 data (40% home win rate), HFA would be < 48
        # So verify it's not using 2024 data
        assert learned_hfa > 50.0, (
            "HFA should reflect 2023's high home win rate, not 2024's low rate"
        )

    def test_hfa_first_season_returns_init(self):
        """Test 6: For the first season (no prior data), return hfa_init (48)."""
        # Only provide 2024 data, ask for 2024 HFA (no 2023 data)
        games_2024 = _make_season_games(2024, num_games=20, home_win_rate=0.65)

        elo = EloRatingSystem()
        learned_hfa = elo.learn_home_field_advantage(games_2024, season=2024)

        assert learned_hfa == 48.0, (
            f"Expected hfa_init=48.0 for first season, got {learned_hfa}"
        )

    def test_hfa_clamp_range_is_20_to_80(self):
        """Test 13: HFA clamp range is 20-80 (not 20-120)."""
        # Create a prior season with extreme home win rate (90%)
        games_2023 = _make_season_games(2023, num_games=20, home_win_rate=0.95)
        games_2024 = _make_season_games(2024, num_games=0)

        all_games = pd.concat([games_2023, games_2024], ignore_index=True)

        elo = EloRatingSystem()
        learned_hfa = elo.learn_home_field_advantage(all_games, season=2024)

        assert learned_hfa <= 80.0, (
            f"HFA should be clamped at max 80.0, got {learned_hfa}"
        )
        assert learned_hfa >= 20.0, (
            f"HFA should be clamped at min 20.0, got {learned_hfa}"
        )


class TestSeasonCarryover:
    """Tests for season carryover formula."""

    def test_carryover_75_25_formula(self):
        """Test 7: apply_season_carryover applies 75/25 formula.

        New rating = old_rating * 0.75 + 1500 * 0.25
        """
        elo = EloRatingSystem()

        # Set up a team with a known rating in season 2023
        team_rating = elo.get_or_create_rating("BUF", 2023)
        team_rating.rating = 1600.0
        team_rating.season = 2023

        # Apply carryover to 2024
        elo.apply_season_carryover(2024)

        expected = 1600.0 * 0.75 + 1500.0 * 0.25  # = 1575.0
        actual = elo.ratings["BUF"].rating

        assert abs(actual - expected) < 0.01, (
            f"Expected carryover result {expected}, got {actual}"
        )


class TestDivisionalGameDetection:
    """Tests for divisional game detection."""

    def test_same_division_returns_true(self):
        """Test 8: is_divisional_game returns True for same-division teams.

        BUF and MIA are both AFC East.
        """
        assert is_divisional_game("BUF", "MIA") is True

    def test_different_division_returns_false(self):
        """Test 9: is_divisional_game returns False for different-division teams.

        BUF (AFC East) and KC (AFC West) are not in the same division.
        """
        assert is_divisional_game("BUF", "KC") is False

    def test_nfc_divisional_game(self):
        """NFC divisional games should also be detected."""
        # DAL and PHI are both NFC East
        assert is_divisional_game("DAL", "PHI") is True

    def test_cross_conference_returns_false(self):
        """Cross-conference matchups are never divisional."""
        # BUF (AFC East) and DAL (NFC East) -- same "East" but different conference
        assert is_divisional_game("BUF", "DAL") is False

    def test_rams_chargers_not_divisional(self):
        """LA (Rams, NFC West) and LAC (Chargers, AFC West) are NOT divisional."""
        assert is_divisional_game("LA", "LAC") is False

    def test_nfc_west_divisional(self):
        """LA (Rams) and SF are both NFC West -- divisional."""
        assert is_divisional_game("LA", "SF") is True

    def test_afc_west_divisional(self):
        """LAC (Chargers) and KC are both AFC West -- divisional."""
        assert is_divisional_game("LAC", "KC") is True


class TestDivisionalHFAReduction:
    """Tests for divisional HFA reduction in predictions."""

    def test_predict_divisional_reduces_hfa(self):
        """Test 10: predict_game applies divisional HFA reduction (HFA * 0.54)."""
        elo = EloRatingSystem()

        # Set known HFA for the season
        elo.hfa_by_season[2024] = 48.0

        # Divisional game: BUF at MIA (both AFC East)
        result_div = elo.predict_game("MIA", "BUF", 2024, is_divisional=True)

        # The HFA used should be 48.0 * 0.54 = 25.92
        expected_hfa = 48.0 * DIVISIONAL_HFA_FACTOR
        assert abs(result_div["hfa_used"] - expected_hfa) < 0.01, (
            f"Expected divisional HFA={expected_hfa}, got {result_div['hfa_used']}"
        )

    def test_predict_non_divisional_uses_full_hfa(self):
        """Test 11: predict_game applies full HFA for non-divisional matchups."""
        elo = EloRatingSystem()
        elo.hfa_by_season[2024] = 48.0

        # Non-divisional game: BUF at DEN (AFC East vs AFC West)
        result_non_div = elo.predict_game("DEN", "BUF", 2024, is_divisional=False)

        assert abs(result_non_div["hfa_used"] - 48.0) < 0.01, (
            f"Expected full HFA=48.0, got {result_non_div['hfa_used']}"
        )

    def test_divisional_hfa_factor_value(self):
        """DIVISIONAL_HFA_FACTOR should be 0.54."""
        assert DIVISIONAL_HFA_FACTOR == 0.54


class TestChronologicalOrdering:
    """Tests for chronological game processing."""

    def test_sequential_elo_updates(self):
        """Test 12: Elo updates are strictly sequential by game date.

        For the same team, rating after earlier game < later game (or vice
        versa), but the key is that ordering is respected.
        """
        elo = EloRatingSystem()
        elo.hfa_by_season[2024] = 48.0

        # Game A: Team KC wins on Sept 7
        elo.update_ratings(
            home_team="KC",
            away_team="DEN",
            home_score=28,
            away_score=17,
            season=2024,
            game_date=datetime(2024, 9, 7, 13, 0),
            game_id="GAME_A",
        )
        rating_after_a = elo.ratings["KC"].rating

        # Game B: Team KC wins again on Sept 14
        elo.update_ratings(
            home_team="KC",
            away_team="LV",
            home_score=31,
            away_score=14,
            season=2024,
            game_date=datetime(2024, 9, 14, 13, 0),
            game_id="GAME_B",
        )
        rating_after_b = elo.ratings["KC"].rating

        # After two wins, KC's rating should have increased from the initial 1500
        assert rating_after_a > 1500.0, "KC should gain Elo after winning game A"
        assert rating_after_b > rating_after_a, (
            "KC should gain more Elo after winning game B (cumulative)"
        )

    def test_process_season_sorts_by_date(self):
        """process_season_chronologically processes games in date order."""
        elo = EloRatingSystem()

        # Create games out of chronological order
        games = pd.DataFrame(
            [
                {
                    "game_id": "LATE_GAME",
                    "season": 2024,
                    "week": 2,
                    "home_team": "KC",
                    "away_team": "DEN",
                    "home_score": 28,
                    "away_score": 17,
                    "kickoff_et": datetime(2024, 9, 14, 13, 0),
                },
                {
                    "game_id": "EARLY_GAME",
                    "season": 2024,
                    "week": 1,
                    "home_team": "BUF",
                    "away_team": "MIA",
                    "home_score": 24,
                    "away_score": 21,
                    "kickoff_et": datetime(2024, 9, 7, 13, 0),
                },
            ]
        )

        result = elo.process_season_chronologically(games, 2024)

        # Verify games are sorted by kickoff_et in result
        assert result.iloc[0]["game_id"] == "EARLY_GAME"
        assert result.iloc[1]["game_id"] == "LATE_GAME"


class TestUpdateRatingsDivisional:
    """Tests for divisional flag in update_ratings."""

    def test_update_ratings_divisional_reduces_hfa(self):
        """update_ratings with is_divisional=True should use reduced HFA."""
        elo = EloRatingSystem()
        elo.hfa_by_season[2024] = 48.0

        # Set up known ratings
        elo.get_or_create_rating("MIA", 2024)
        elo.get_or_create_rating("BUF", 2024)

        # Non-divisional update
        elo_non_div = EloRatingSystem()
        elo_non_div.hfa_by_season[2024] = 48.0
        elo_non_div.get_or_create_rating("MIA", 2024)
        elo_non_div.get_or_create_rating("BUF", 2024)

        # Divisional game
        home_change_div, _ = elo.update_ratings(
            "MIA",
            "BUF",
            24,
            21,
            2024,
            datetime(2024, 9, 7, 13, 0),
            is_divisional=True,
        )

        # Non-divisional game (same score)
        home_change_non_div, _ = elo_non_div.update_ratings(
            "MIA",
            "BUF",
            24,
            21,
            2024,
            datetime(2024, 9, 7, 13, 0),
            is_divisional=False,
        )

        # Divisional HFA is lower, so the home team was less "expected" to win,
        # meaning a home win gives MORE Elo change with lower HFA
        # (because expected was lower). Verify the changes differ.
        assert home_change_div != home_change_non_div, (
            "Divisional and non-divisional should produce different rating changes"
        )


class TestNFLDivisions:
    """Tests for the NFL divisions data structure."""

    def test_exactly_8_divisions(self):
        """NFL_DIVISIONS should contain exactly 8 divisions."""
        from ratings.elo import NFL_DIVISIONS

        assert len(NFL_DIVISIONS) == 8

    def test_4_teams_per_division(self):
        """Each division should have exactly 4 teams."""
        from ratings.elo import NFL_DIVISIONS

        for div_name, teams in NFL_DIVISIONS.items():
            assert len(teams) == 4, (
                f"Division {div_name} has {len(teams)} teams, expected 4"
            )

    def test_32_total_teams(self):
        """All 32 NFL teams should be represented."""
        from ratings.elo import NFL_DIVISIONS

        all_teams = set()
        for teams in NFL_DIVISIONS.values():
            all_teams.update(teams)
        assert len(all_teams) == 32

    def test_canonical_abbreviations(self):
        """Divisions use canonical abbreviations (LA=Rams, LAC=Chargers)."""
        from ratings.elo import NFL_DIVISIONS

        all_teams = set()
        for teams in NFL_DIVISIONS.values():
            all_teams.update(teams)

        # LA is Rams (NFC West) -- canonical
        assert "LA" in all_teams, "LA (Rams) should be in divisions"
        assert "LAR" not in all_teams, (
            "LAR should not be used (LA is canonical for Rams)"
        )

        # LAC is Chargers (AFC West) -- canonical
        assert "LAC" in all_teams, "LAC (Chargers) should be in divisions"


# ---------------------------------------------------------------------------
# ONE HOME-FIELD-ADVANTAGE CALL SHAPE (Plan 33-03 Task 3, COLD-04, T-33-15)
#
# `learn_home_field_advantage` filters its input to `season - 1`
# (ratings/elo.py:395-400). Handed a SINGLE-SEASON frame that filter is empty, the
# function silently returns `hfa_init` (48) and records it as the season's learned
# value. That is not an error anywhere; it is a systematic ~14-to-22-point
# home-field-advantage error, applied to every game of the live season, arrived at by
# passing the wrong frame.
#
# The canonical builder passes `all_games` and gets a learned value. The live weekly
# path passed one season and got 48. Both write into the ratings three deployed models
# consume, so the two paths must learn the SAME way. The assertion below is parity to
# full float equality, with an anti-vacuity companion: a parity test that would also
# pass while both paths returned `hfa_init` is not a test.
# ---------------------------------------------------------------------------


def _with_home_win_rate(frame: pd.DataFrame, home_win_rate: float) -> pd.DataFrame:
    """Flip scores so that *home_win_rate* of the GRADED games are home wins.

    The shared fixture grades every game as a home win, which learns a clamped HFA of
    80. A realistic rate produces a value inside the band the repository's own history
    reports, so the parity assertion is made against a plausible number rather than
    against a clamp boundary.
    """
    graded = frame["home_score"].notna()
    indices = list(frame.index[graded])
    away_wins = indices[int(len(indices) * home_win_rate) :]
    flipped = frame.copy()
    for index in away_wins:
        home, away = flipped.loc[index, "home_score"], flipped.loc[index, "away_score"]
        flipped.loc[index, "home_score"] = away
        flipped.loc[index, "away_score"] = home
    return flipped


class TestTheLiveAndCanonicalPathsLearnHFATheSameWay:
    """Parity to full float equality, plus the assertion that makes it non-vacuous."""

    @staticmethod
    def _two_season_games() -> pd.DataFrame:
        from tests.fixtures.elo_sandbox import make_season_games

        # 9 of the 16 graded games are home wins -> a learned HFA of 49.4, INSIDE
        # the 20-80 clamp rather than at a boundary. A clamped value would agree
        # with itself for the wrong reason: the clamp, not the learning.
        prior = _with_home_win_rate(make_season_games(2025, weeks=4), 0.5625)
        live = make_season_games(2026, weeks=2)
        return pd.concat([prior, live], ignore_index=True)

    def test_live_hfa_equals_canonical_hfa_for_the_same_season(
        self, tmp_path, monkeypatch
    ) -> None:
        from scripts.build_elo import EloBuilder
        from tests.fixtures.elo_sandbox import (
            redirect_storage_to_sandbox,
            sandbox_builder,
        )

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        games = self._two_season_games()

        canonical = sandbox_builder(sandbox, games)
        canonical.build_elo_with_snapshots(start_season=2025)
        canonical_hfa = canonical.elo_system.hfa_by_season[2026]

        live = EloBuilder(data_root=sandbox)
        live.update_current_season(season=2026)
        live_hfa = live.elo_system.hfa_by_season[2026]

        assert live_hfa == canonical_hfa, (
            "the live weekly path and the canonical builder learned DIFFERENT "
            f"home-field advantages for 2026 ({live_hfa} vs {canonical_hfa}). Both "
            "write the ratings the deployed WP, ATS and O/U models read; two answers "
            "means one of them is a systematic error applied to every live game."
        )

    def test_the_live_hfa_is_not_the_hfa_init_default(
        self, tmp_path, monkeypatch
    ) -> None:
        """Anti-vacuity. Equal-and-both-wrong is the failure mode being removed.

        Handed a single-season frame, `learn_home_field_advantage`'s `season - 1`
        filter selects nothing and the function returns `hfa_init` unchanged. A parity
        test alone would pass if BOTH paths did that.
        """
        from scripts.build_elo import EloBuilder
        from tests.fixtures.elo_sandbox import (
            redirect_storage_to_sandbox,
            sandbox_builder,
        )

        sandbox = redirect_storage_to_sandbox(monkeypatch, tmp_path)
        sandbox_builder(sandbox, self._two_season_games())

        live = EloBuilder(data_root=sandbox)
        live.update_current_season(season=2026)
        live_hfa = live.elo_system.hfa_by_season[2026]

        assert live_hfa != EloRatingSystem().hfa_init, (
            f"the live path learned exactly hfa_init ({live_hfa}), which is what "
            "learn_home_field_advantage returns when its prior-season filter finds "
            "nothing -- i.e. when the live path is still passing a single-season frame."
        )
        assert 20.0 < live_hfa < 80.0, (
            f"the learned HFA sits ON a clamp boundary ({live_hfa}), where it would"
            " agree with the canonical value because of the clamp rather than"
            " because of the learning."
        )

    def test_no_hfa_value_is_hardcoded_for_the_live_season(self) -> None:
        """Pinning a 2026 constant would return the defect every following season."""
        import ast
        from pathlib import Path

        repo_root = Path(__file__).resolve().parents[2]
        for relative in ("ratings/elo.py", "scripts/build_elo.py"):
            tree = ast.parse((repo_root / relative).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Assign):
                    continue
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                for name in names:
                    assert "2026" not in name, (
                        f"{relative} declares {name}: a season-pinned HFA constant "
                        "fixes one season and returns the defect for every following "
                        "one."
                    )
