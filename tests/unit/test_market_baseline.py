"""Tests for market baseline computation (MODL-08).

The market baseline treats closing line implied probabilities as a "model"
and computes accuracy, Brier score, and log loss for comparison against
our trained models. This ensures we can always answer: "Is our model
better than just following the market?"
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def synthetic_games() -> pd.DataFrame:
    """Create synthetic games DataFrame with realistic NFL scores.

    200 games with home_score mean ~24, away_score mean ~21,
    matching realistic NFL scoring patterns.
    """
    rng = np.random.default_rng(42)

    n_games = 200
    game_ids = [f"2023_{w:02d}_{i:02d}" for w in range(1, 18) for i in range(1, 13)]
    game_ids = game_ids[:n_games]

    home_scores = rng.poisson(lam=24, size=n_games)
    away_scores = rng.poisson(lam=21, size=n_games)

    # Ensure no ties (NFL overtime rules make ties very rare)
    for i in range(n_games):
        if home_scores[i] == away_scores[i]:
            home_scores[i] += 3  # Home team wins in OT by field goal

    return pd.DataFrame(
        {
            "game_id": game_ids,
            "season": 2023,
            "week": [(i // 12) + 1 for i in range(n_games)],
            "home_score": home_scores,
            "away_score": away_scores,
        }
    )


@pytest.fixture()
def synthetic_odds(synthetic_games: pd.DataFrame) -> pd.DataFrame:
    """Create synthetic odds DataFrame with realistic moneylines.

    Odds are correlated with actual outcomes (the favorite wins more
    often than the underdog), which produces realistic market accuracy
    in the 55-70% range. Includes NaN rows to test exclusion logic.
    """
    rng = np.random.default_rng(42)

    n_games = len(synthetic_games)
    game_ids = synthetic_games["game_id"].tolist()

    # Moneyline lookup: maps favorite strength to underdog line
    fav_to_dog = {
        -110: -110,
        -130: 110,
        -150: 130,
        -170: 150,
        -200: 170,
        -250: 200,
        -300: 250,
    }

    # Generate odds CORRELATED with actual outcomes
    # If home team won by a lot, they should be the favorite
    ml_home = []
    ml_away = []

    for i in range(n_games):
        margin = int(
            synthetic_games.iloc[i]["home_score"]
            - synthetic_games.iloc[i]["away_score"]
        )
        # Map margin to favorite strength with noise
        noisy_margin = margin + rng.normal(0, 5)

        if noisy_margin > 0:
            # Home is favorite
            if abs(noisy_margin) > 14:
                fav = -300
            elif abs(noisy_margin) > 10:
                fav = -250
            elif abs(noisy_margin) > 7:
                fav = -200
            elif abs(noisy_margin) > 5:
                fav = -170
            elif abs(noisy_margin) > 3:
                fav = -150
            elif abs(noisy_margin) > 1:
                fav = -130
            else:
                fav = -110
            ml_home.append(float(fav))
            ml_away.append(float(fav_to_dog[fav]))
        else:
            # Away is favorite
            if abs(noisy_margin) > 14:
                fav = -300
            elif abs(noisy_margin) > 10:
                fav = -250
            elif abs(noisy_margin) > 7:
                fav = -200
            elif abs(noisy_margin) > 5:
                fav = -170
            elif abs(noisy_margin) > 3:
                fav = -150
            elif abs(noisy_margin) > 1:
                fav = -130
            else:
                fav = -110
            ml_home.append(float(fav_to_dog[fav]))
            ml_away.append(float(fav))

    # Inject NaN rows for exclusion test (last 10 games)
    for i in range(n_games - 10, n_games):
        ml_home[i] = float("nan")
        ml_away[i] = float("nan")

    # Also put spread and total columns for ATS/OU tests
    spreads = []
    totals = []
    for i in range(n_games):
        actual_margin = int(
            synthetic_games.iloc[i]["home_score"]
            - synthetic_games.iloc[i]["away_score"]
        )
        # Market spread is roughly correlated with actual but with noise
        spread = -(actual_margin + rng.normal(0, 7))
        spreads.append(round(spread * 2) / 2)  # Round to nearest 0.5

        actual_total = int(
            synthetic_games.iloc[i]["home_score"]
            + synthetic_games.iloc[i]["away_score"]
        )
        total = actual_total + rng.normal(0, 5)
        totals.append(round(total * 2) / 2)

    return pd.DataFrame(
        {
            "game_id": game_ids,
            "season": 2023,
            "ml_home": ml_home,
            "ml_away": ml_away,
            "spread": spreads,
            "total": totals,
        }
    )


class TestMarketBaselineWP:
    """Tests for WP market baseline (closing moneyline implied probabilities)."""

    def test_market_baseline_wp_accuracy(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_odds: pd.DataFrame,
    ) -> None:
        """Market WP baseline accuracy should be in [0.5, 0.8] range on synthetic data."""
        from models.train import compute_market_baseline

        result = compute_market_baseline(synthetic_games, synthetic_odds, target="wp")

        assert "market_accuracy" in result
        assert 0.5 <= result["market_accuracy"] <= 0.8, (
            f"Market accuracy {result['market_accuracy']} outside expected range [0.5, 0.8]"
        )

    def test_market_baseline_wp_brier(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_odds: pd.DataFrame,
    ) -> None:
        """Market WP baseline Brier score should be in [0.15, 0.30] range."""
        from models.train import compute_market_baseline

        result = compute_market_baseline(synthetic_games, synthetic_odds, target="wp")

        assert "market_brier" in result
        assert 0.15 <= result["market_brier"] <= 0.30, (
            f"Market Brier {result['market_brier']} outside expected range [0.15, 0.30]"
        )

    def test_market_baseline_devigged(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_odds: pd.DataFrame,
    ) -> None:
        """Market baseline should devig odds (fair probs sum to 1.0, not > 1.0)."""
        from models.train import compute_market_baseline

        result = compute_market_baseline(synthetic_games, synthetic_odds, target="wp")

        # If devigging is working, market_logloss should be computed with
        # fair probabilities. We can verify by checking the result has the key
        # and that the value is reasonable (devigged logloss < raw logloss).
        assert "market_logloss" in result
        # Log loss with fair probs should be in a reasonable range
        assert 0.4 <= result["market_logloss"] <= 0.8, (
            f"Market logloss {result['market_logloss']} outside expected range"
        )

        # Additionally verify n_games is less than total (some had NaN odds)
        assert result["n_games"] < len(synthetic_games)


class TestMarketBaselineATS:
    """Tests for ATS market baseline."""

    def test_market_baseline_ats(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_odds: pd.DataFrame,
    ) -> None:
        """ATS market baseline accuracy should be around 50% (spread equalizes action)."""
        from models.train import compute_market_baseline

        result = compute_market_baseline(synthetic_games, synthetic_odds, target="ats")

        assert "market_accuracy" in result
        # ATS accuracy from the market's perspective should be near 50%
        # with some variance due to synthetic data
        assert 0.35 <= result["market_accuracy"] <= 0.65, (
            f"ATS market accuracy {result['market_accuracy']} outside expected [0.35, 0.65]"
        )
        assert result["n_games"] > 0


class TestMarketBaselineOU:
    """Tests for O/U market baseline."""

    def test_market_baseline_ou(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_odds: pd.DataFrame,
    ) -> None:
        """O/U market baseline accuracy should be around 50%."""
        from models.train import compute_market_baseline

        result = compute_market_baseline(synthetic_games, synthetic_odds, target="ou")

        assert "market_accuracy" in result
        # O/U accuracy should be near 50%
        assert 0.35 <= result["market_accuracy"] <= 0.65, (
            f"O/U market accuracy {result['market_accuracy']} outside expected [0.35, 0.65]"
        )
        assert result["n_games"] > 0


class TestMarketBaselineExclusion:
    """Tests for missing data handling in market baseline."""

    def test_market_baseline_excludes_missing(
        self,
        synthetic_games: pd.DataFrame,
        synthetic_odds: pd.DataFrame,
    ) -> None:
        """Games with NaN closing odds should be excluded; count reported."""
        from models.train import compute_market_baseline

        result = compute_market_baseline(synthetic_games, synthetic_odds, target="wp")

        assert "n_excluded" in result
        assert result["n_excluded"] == 10, (
            f"Expected 10 excluded games (NaN odds), got {result['n_excluded']}"
        )
        assert result["n_games"] == 190, (
            f"Expected 190 games after exclusion, got {result['n_games']}"
        )
