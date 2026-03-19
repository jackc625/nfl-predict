"""Unit tests for Elo rating system and probability conversions."""

from datetime import UTC, datetime

import numpy as np

from models.calibrate import ProbabilityCalibrator as IsotonicCalibrator
from models.utils import kelly_criterion, odds_to_probability, probability_to_odds
from ratings.elo import EloRatingSystem
from tests.conftest import assert_probability_range


class TestEloRatingSystem:
    """Test the Elo rating system implementation."""

    def _make_date(self, week: int = 1, season: int = 2024) -> datetime:
        """Helper to create a game date from week number."""
        # Approximate: NFL week 1 starts early September
        day = 7 + (week - 1) * 7
        return datetime(season, 9, min(day, 28), 13, 0, tzinfo=UTC)

    def test_initial_ratings(self):
        """Test initial Elo ratings are set correctly."""
        elo_system = EloRatingSystem()

        # Test default initialization via get_or_create_rating
        rating = elo_system.get_or_create_rating("BUF", 2024)
        assert rating.rating == 1500.0, (
            f"Initial rating should be 1500, got {rating.rating}"
        )

        # Test uncertainty initialization
        assert rating.uncertainty > 0, "Initial uncertainty should be positive"

    def test_home_field_advantage(self):
        """Test home field advantage configuration."""
        elo_system = EloRatingSystem()

        # Test default HFA (hfa_init)
        hfa = elo_system.hfa_init
        assert 0 <= hfa <= 100, f"HFA should be reasonable, got {hfa}"
        assert hfa == 48.0, f"Default HFA should be 48.0, got {hfa}"

    def test_elo_update_win(self):
        """Test Elo update for a win."""
        elo_system = EloRatingSystem()

        # Set up teams
        team_a_initial = 1500.0
        team_b_initial = 1500.0
        elo_system.get_or_create_rating("TEAM_A", 2024)
        elo_system.get_or_create_rating("TEAM_B", 2024)

        # Team A wins at home by 7 points
        elo_system.update_ratings(
            home_team="TEAM_A",
            away_team="TEAM_B",
            home_score=28,
            away_score=21,
            season=2024,
            game_date=self._make_date(1),
        )

        # Winner should gain rating, loser should lose rating
        team_a_new = elo_system.ratings["TEAM_A"].rating
        team_b_new = elo_system.ratings["TEAM_B"].rating

        assert team_a_new > team_a_initial, "Winner should gain Elo points"
        assert team_b_new < team_b_initial, "Loser should lose Elo points"

        # Total ratings should be approximately conserved
        total_before = team_a_initial + team_b_initial
        total_after = team_a_new + team_b_new
        assert abs(total_after - total_before) < 10, (
            "Total ratings should be approximately conserved"
        )

    def test_elo_update_blowout_vs_close(self):
        """Test that margin of victory affects Elo changes."""
        elo_system = EloRatingSystem()

        # Initialize teams
        for team in ["TEAM_A", "TEAM_B", "TEAM_C", "TEAM_D"]:
            rating = elo_system.get_or_create_rating(team, 2024)
            rating.rating = 1500.0
            rating.uncertainty = 100.0

        # Close game: Team A beats Team B by 3
        elo_system.update_ratings(
            "TEAM_A",
            "TEAM_B",
            24,
            21,
            season=2024,
            game_date=self._make_date(1),
        )
        close_game_change = elo_system.ratings["TEAM_A"].rating - 1500

        # Blowout: Team C beats Team D by 21
        elo_system.update_ratings(
            "TEAM_C",
            "TEAM_D",
            35,
            14,
            season=2024,
            game_date=self._make_date(1),
        )
        blowout_change = elo_system.ratings["TEAM_C"].rating - 1500

        assert blowout_change > close_game_change, (
            "Blowout should result in larger Elo change"
        )

    def test_elo_uncertainty_decay(self):
        """Test that uncertainty decreases with more games."""
        elo_system = EloRatingSystem()

        # Create teams
        elo_system.get_or_create_rating("TEAM_A", 2024)
        elo_system.get_or_create_rating("TEAM_B", 2024)
        initial_uncertainty = elo_system.ratings["TEAM_A"].uncertainty

        # Play several games
        for week in range(1, 6):
            elo_system.update_ratings(
                "TEAM_A",
                "TEAM_B",
                24,
                21,
                season=2024,
                game_date=self._make_date(week),
            )

        final_uncertainty = elo_system.ratings["TEAM_A"].uncertainty
        assert final_uncertainty < initial_uncertainty, (
            "Uncertainty should decrease with more games"
        )

    def test_season_carryover(self):
        """Test Elo carryover between seasons."""
        elo_system = EloRatingSystem()

        # Set team rating significantly above 1500
        rating = elo_system.get_or_create_rating("TEAM_A", 2024)
        rating.rating = 1700.0

        # Apply season carryover
        elo_system.apply_season_carryover(season=2025)

        new_rating = elo_system.ratings["TEAM_A"].rating

        # Should be between initial rating and 1500 (regression to mean)
        assert 1500 < new_rating < 1700, (
            f"Carryover rating should regress toward mean, got {new_rating}"
        )

    def test_win_probability_calculation(self):
        """Test win probability calculation from Elo difference."""
        elo_system = EloRatingSystem()

        # Test equal teams using predict_game
        for team in ["HOME", "AWAY", "HOME2", "AWAY2"]:
            elo_system.get_or_create_rating(team, 2024)

        # Equal teams with HFA should favor home team
        result_equal = elo_system.predict_game("HOME", "AWAY", season=2024)
        prob_equal = result_equal["home_win_prob"]
        assert 0.5 < prob_equal < 0.8, (
            f"Equal teams with HFA should favor home team, got {prob_equal}"
        )

        # Stronger home team
        elo_system.ratings["HOME2"].rating = 1600.0
        result_favorite = elo_system.predict_game("HOME2", "AWAY2", season=2024)
        prob_favorite = result_favorite["home_win_prob"]
        assert prob_favorite > prob_equal, (
            "Stronger home team should have higher win probability"
        )

        # Stronger away team
        elo_system.ratings["HOME"].rating = 1500.0
        elo_system.ratings["AWAY"].rating = 1600.0
        result_underdog = elo_system.predict_game("HOME", "AWAY", season=2024)
        prob_underdog = result_underdog["home_win_prob"]
        assert prob_underdog < prob_equal, (
            "Weaker home team should have lower win probability"
        )

        # All probabilities should be valid
        assert_probability_range([prob_equal, prob_favorite, prob_underdog])

    def test_chronological_updates(self):
        """Test that Elo updates maintain chronological order."""
        elo_system = EloRatingSystem()

        # Create game results in chronological order
        games = [
            {
                "home_team": "TEAM_A",
                "away_team": "TEAM_B",
                "home_score": 24,
                "away_score": 21,
                "game_date": self._make_date(1),
            },
            {
                "home_team": "TEAM_B",
                "away_team": "TEAM_A",
                "home_score": 28,
                "away_score": 14,
                "game_date": self._make_date(2),
            },
            {
                "home_team": "TEAM_A",
                "away_team": "TEAM_C",
                "home_score": 21,
                "away_score": 17,
                "game_date": self._make_date(3),
            },
        ]

        # Track ratings after each game
        ratings_history = []
        for game in games:
            elo_system.update_ratings(season=2024, **game)
            ratings_history.append(elo_system.ratings["TEAM_A"].rating)

        # Verify that we can reconstruct the same ratings by processing games in order
        elo_system2 = EloRatingSystem()
        for game in games:
            elo_system2.update_ratings(season=2024, **game)

        final_rating_1 = elo_system.ratings["TEAM_A"].rating
        final_rating_2 = elo_system2.ratings["TEAM_A"].rating

        assert abs(final_rating_1 - final_rating_2) < 0.1, (
            "Chronological updates should be deterministic"
        )


class TestProbabilityConversions:
    """Test probability and odds conversion utilities."""

    def test_probability_to_odds_conversion(self):
        """Test conversion from probability to American odds."""
        test_cases = [
            (0.5, 100),  # Even odds
            (0.52, -108),  # Slight favorite (approximately)
            (0.6, -150),  # Moderate favorite (approximately)
            (0.75, -300),  # Heavy favorite (approximately)
            (0.4, 150),  # Underdog (approximately)
            (0.25, 300),  # Heavy underdog (approximately)
        ]

        for prob, expected_odds in test_cases:
            odds = probability_to_odds(prob)

            # Allow some tolerance for rounding
            if expected_odds > 0:
                assert abs(odds - expected_odds) <= 25, (
                    f"Prob {prob} should convert to ~{expected_odds}, got {odds}"
                )
            else:
                assert abs(odds - expected_odds) <= 25, (
                    f"Prob {prob} should convert to ~{expected_odds}, got {odds}"
                )

    def test_odds_to_probability_conversion(self):
        """Test conversion from American odds to probability."""
        test_cases = [
            (100, 0.5),  # Even odds
            (-110, 0.524),  # Standard betting line
            (-200, 0.667),  # Heavy favorite
            (150, 0.4),  # Underdog
            (300, 0.25),  # Heavy underdog
        ]

        for odds, expected_prob in test_cases:
            prob = odds_to_probability(odds)
            assert abs(prob - expected_prob) < 0.01, (
                f"Odds {odds} should convert to ~{expected_prob}, got {prob}"
            )
            assert_probability_range([prob])

    def test_round_trip_conversion(self):
        """Test that odds->probability->odds conversions are consistent."""
        test_odds = [-300, -200, -150, -110, 100, 150, 200, 300]

        for original_odds in test_odds:
            prob = odds_to_probability(original_odds)
            converted_odds = probability_to_odds(prob)

            # Should be very close to original (within rounding)
            assert abs(converted_odds - original_odds) <= 5, (
                f"Round trip failed: {original_odds} -> {prob} -> {converted_odds}"
            )

    def test_extreme_probabilities(self):
        """Test conversion behavior at extreme probabilities."""
        # Very high probability
        high_prob = 0.99
        high_odds = probability_to_odds(high_prob)
        assert high_odds < -900, (
            f"Very high probability should give very negative odds, got {high_odds}"
        )

        # Very low probability
        low_prob = 0.01
        low_odds = probability_to_odds(low_prob)
        assert low_odds > 900, (
            f"Very low probability should give very positive odds, got {low_odds}"
        )

    def test_kelly_criterion(self):
        """Test Kelly criterion bet sizing calculation."""
        # Test profitable bet
        test_cases = [
            {"edge": 0.05, "prob": 0.55, "odds": -110, "expected_kelly": "positive"},
            {"edge": 0.0, "prob": 0.5, "odds": 100, "expected_kelly": "zero"},
            {
                "edge": -0.05,
                "prob": 0.45,
                "odds": 110,
                "expected_kelly": "zero_or_negative",
            },
        ]

        for case in test_cases:
            kelly_fraction = kelly_criterion(
                win_probability=case["prob"],
                decimal_odds=abs(case["odds"]) / 100 + 1
                if case["odds"] > 0
                else 100 / abs(case["odds"]) + 1,
            )

            if case["expected_kelly"] == "positive":
                assert kelly_fraction > 0, (
                    f"Profitable bet should have positive Kelly, got {kelly_fraction}"
                )
            elif case["expected_kelly"] == "zero":
                assert abs(kelly_fraction) < 0.01, (
                    f"Break-even bet should have ~zero Kelly, got {kelly_fraction}"
                )
            elif case["expected_kelly"] == "zero_or_negative":
                assert kelly_fraction <= 0.01, (
                    f"Unprofitable bet should have zero/negative Kelly, got {kelly_fraction}"
                )


class TestIsotonicCalibrator:
    """Test the isotonic calibration system."""

    def test_calibrator_initialization(self):
        """Test calibrator initializes correctly."""
        calibrator = IsotonicCalibrator()
        assert calibrator is not None
        assert calibrator.primary_method == "isotonic"

    def test_calibrator_fitting(self):
        """Test calibrator fitting with sample data."""
        calibrator = IsotonicCalibrator()

        # Create sample predictions and outcomes
        np.random.seed(42)
        n_samples = 100

        # Poorly calibrated predictions (too extreme)
        raw_probs = np.random.beta(2, 2, n_samples)  # U-shaped distribution
        raw_probs = np.clip(raw_probs * 1.2 - 0.1, 0.01, 0.99)  # Make more extreme

        # Outcomes based on true probabilities (less extreme)
        true_probs = 0.3 + 0.4 * raw_probs  # Shrink toward 0.5
        outcomes = np.random.binomial(1, true_probs, n_samples)

        # Calibrate using calibrate_probabilities
        results = calibrator.calibrate_probabilities(raw_probs, outcomes)

        assert results.calibrator is not None, (
            "Calibrator should be fitted after calibration"
        )

        # Test calibrated output
        calibrated_probs = results.calibrated_probabilities

        assert len(calibrated_probs) == len(raw_probs), (
            "Calibrated predictions should match input length"
        )
        assert_probability_range(calibrated_probs)

        # Calibrated probabilities should generally be less extreme than raw probabilities
        extreme_raw = np.sum((raw_probs < 0.2) | (raw_probs > 0.8))
        extreme_calibrated = np.sum((calibrated_probs < 0.2) | (calibrated_probs > 0.8))

        assert extreme_calibrated <= extreme_raw, (
            "Calibration should reduce extreme predictions"
        )

    def test_reliability_diagram_data(self):
        """Test generation of reliability diagram data."""
        calibrator = IsotonicCalibrator(n_bins=10)

        # Create well-calibrated sample data
        np.random.seed(42)
        probs = np.random.uniform(0.1, 0.9, 200)
        outcomes = np.random.binomial(1, probs, 200)

        results = calibrator.calibrate_probabilities(probs, outcomes)

        # Reliability curve data is in the results
        mean_predicted, fraction_positive, bin_edges = results.reliability_curve

        assert len(mean_predicted) > 0, "Should have reliability curve data"
        assert len(fraction_positive) > 0, "Should have observed frequencies"

        # All observed frequencies should be valid probabilities
        assert_probability_range(fraction_positive)

    def test_calibration_metrics(self):
        """Test calculation of calibration metrics."""
        calibrator = IsotonicCalibrator()

        # Create sample data
        np.random.seed(42)
        n_samples = 500

        # Perfectly calibrated data
        perfect_probs = np.random.uniform(0.1, 0.9, n_samples)
        perfect_outcomes = np.random.binomial(1, perfect_probs, n_samples)

        # Poorly calibrated data (overconfident)
        poor_probs = np.where(
            perfect_probs > 0.5,
            np.minimum(perfect_probs + 0.2, 0.99),
            np.maximum(perfect_probs - 0.2, 0.01),
        )

        # Calculate ECE using private method (same behavior, different API)
        perfect_ece = calibrator._calculate_ece(perfect_probs, perfect_outcomes)
        poor_ece = calibrator._calculate_ece(poor_probs, perfect_outcomes)

        # Perfect calibration should have lower ECE
        assert poor_ece > perfect_ece, (
            f"Poor calibration (ECE={poor_ece:.3f}) should be worse than perfect (ECE={perfect_ece:.3f})"
        )

        # ECE should be bounded
        assert 0 <= perfect_ece <= 1, f"ECE should be 0-1, got {perfect_ece}"
        assert 0 <= poor_ece <= 1, f"ECE should be 0-1, got {poor_ece}"
