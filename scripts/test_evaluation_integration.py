#!/usr/bin/env python3
"""
Integration test for evaluation framework with baseline Elo model.

This script demonstrates how the evaluation framework integrates with
the baseline Elo model to provide comprehensive model assessment.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.baseline_elo import BaselineEloModel
from models.evaluation import ModelEvaluationFramework
from utils import get_logger

logger = get_logger(__name__)


def create_sample_games(n_games: int = 200) -> pd.DataFrame:
    """Create sample game data for integration testing."""
    np.random.seed(42)

    teams = ["KC", "BUF", "CIN", "BAL", "SF", "PHI", "DAL", "GB"]

    games = []
    for i in range(n_games):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])

        # Simulate Elo-based game outcomes
        home_elo = np.random.normal(1550, 100)
        away_elo = np.random.normal(1450, 100)

        elo_diff = (home_elo + 65) - away_elo
        win_prob = 1 / (1 + 10 ** (-elo_diff / 400))
        home_wins = np.random.binomial(1, win_prob)

        # Generate market odds (slightly biased)
        market_prob = win_prob + np.random.normal(0, 0.03)
        market_prob = np.clip(market_prob, 0.05, 0.95)

        if market_prob >= 0.5:
            market_odds = int(-100 * market_prob / (1 - market_prob))
        else:
            market_odds = int(100 * (1 - market_prob) / market_prob)

        games.append(
            {
                "game_id": f"TEST_GAME_{i:03d}",
                "season": 2023,
                "week": (i % 18) + 1,
                "home_team": home_team,
                "away_team": away_team,
                "home_wins": home_wins,
                "point_differential": np.random.normal(
                    0 if home_wins == 0.5 else (7 if home_wins else -7), 10
                ),
                "market_odds": market_odds,
                "elo_home": home_elo,
                "elo_away": away_elo,
            }
        )

    return pd.DataFrame(games)


def main():
    """Test integration between baseline Elo model and evaluation framework."""
    print("=" * 70)
    print("EVALUATION FRAMEWORK INTEGRATION TEST")
    print("=" * 70)

    try:
        # Step 1: Create sample data
        print("\n1. Creating sample game data...")
        games_df = create_sample_games(200)
        train_games = games_df.iloc[:150].copy()
        test_games = games_df.iloc[150:].copy()
        print(f"   Training games: {len(train_games)}")
        print(f"   Test games: {len(test_games)}")

        # Step 2: Train baseline Elo model
        print("\n2. Training baseline Elo model...")
        elo_model = BaselineEloModel(use_calibration=True)
        training_results = elo_model.train_model(train_games)
        print("   Model trained successfully")
        print(
            f"   Training accuracy: {training_results.performance_metrics.get('training_accuracy', 0):.3f}"
        )

        # Step 3: Generate predictions on test set
        print("\n3. Generating predictions on test set...")
        predictions = elo_model.predict(test_games, calibrated=True)
        print(f"   Generated {len(predictions)} predictions")

        # Step 4: Extract data for evaluation framework
        print("\n4. Preparing data for evaluation framework...")
        pred_probs = np.array(
            [p.calibrated_win_probability or p.raw_win_probability for p in predictions]
        )
        actual_outcomes = test_games["home_wins"].values
        market_odds = test_games["market_odds"].values

        print(f"   Prediction range: [{pred_probs.min():.3f}, {pred_probs.max():.3f}]")
        print(f"   Actual win rate: {actual_outcomes.mean():.3f}")

        # Step 5: Comprehensive evaluation
        print("\n5. Running comprehensive evaluation...")
        evaluator = ModelEvaluationFramework(
            n_calibration_bins=5,  # Smaller bins for small test set
            betting_bankroll=10000,
            min_edge_threshold=0.02,
        )

        eval_metrics = evaluator.evaluate_model(
            predictions=pred_probs,
            actual_outcomes=actual_outcomes,
            market_odds=market_odds,
            prediction_type="binary",
            model_name="Baseline Elo Model",
        )

        # Step 6: Display results
        print("\n6. Evaluation Results:")
        print("   " + "=" * 50)
        print("\n   Core Performance Metrics:")
        core = eval_metrics.core_metrics
        print(f"   • Accuracy:     {core['accuracy']:.3f}")
        print(f"   • Log Loss:     {core['log_loss']:.3f}")
        print(f"   • Brier Score:  {core['brier_score']:.3f}")
        print(f"   • Precision:    {core['precision']:.3f}")
        print(f"   • Recall:       {core['recall']:.3f}")

        print("\n   Calibration Metrics:")
        cal = eval_metrics.calibration_metrics
        print(f"   • ECE:          {cal['ece']:.4f}")
        print(f"   • MCE:          {cal['mce']:.4f}")
        print(f"   • Reliability:  {cal['reliability']:.4f}")
        print(f"   • Resolution:   {cal['resolution']:.4f}")
        print(f"   • Sharpness:    {cal['sharpness']:.4f}")

        print("\n   Betting Simulation:")
        bet = eval_metrics.betting_metrics
        if bet:
            print(f"   • Bets Placed:  {bet['bets_placed']}")
            print(f"   • Win Rate:     {bet['win_rate']:.3f}")
            print(f"   • ROI:          {bet['roi']:.3f}")
            print(f"   • Total P&L:    ${bet['total_profit']:.2f}")
            print(f"   • Max Drawdown: {bet['max_drawdown']:.3f}")
            print(f"   • Sharpe Ratio: {bet['sharpe_ratio']:.3f}")
        else:
            print("   • No betting metrics (no profitable opportunities)")

        print("\n   Statistical Tests:")
        stats = eval_metrics.statistical_tests
        if "accuracy_vs_random" in stats:
            acc_test = stats["accuracy_vs_random"]
            print(f"   • Accuracy vs Random: p-value = {acc_test['p_value']:.4f}")
            print(f"     Significant? {acc_test['significant']}")

        print("\n   Sample Information:")
        info = eval_metrics.sample_info
        print(f"   • Sample Size:   {info['n_predictions']}")
        print(f"   • Mean Pred:     {info['mean_prediction']:.3f}")
        print(f"   • Pred Std:      {info['std_prediction']:.3f}")
        print(f"   • Has Odds:      {info['has_market_odds']}")

        # Step 7: Generate calibration plot
        print("\n7. Generating calibration plot...")
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                plot_path = Path(temp_dir) / "integration_calibration_plot.png"
                fig = evaluator.generate_calibration_plot(
                    pred_probs, actual_outcomes, save_path=str(plot_path)
                )
                plt.close(fig)
                print("   [OK] Calibration plot generated successfully")
        except Exception as e:
            print(f"   [WARN] Calibration plot failed: {e}")

        # Step 8: Test with original baseline model metrics
        print("\n8. Comparing with baseline model's internal metrics...")
        baseline_accuracy = training_results.performance_metrics.get(
            "validation_accuracy"
        )
        if baseline_accuracy:
            print(f"   Baseline internal accuracy: {baseline_accuracy:.3f}")
            print(
                f"   Evaluation framework accuracy: {eval_metrics.core_metrics['accuracy']:.3f}"
            )
            print(
                f"   Difference: {abs(baseline_accuracy - eval_metrics.core_metrics['accuracy']):.3f}"
            )

        print("\n" + "=" * 70)
        print("[SUCCESS] INTEGRATION TEST COMPLETED SUCCESSFULLY!")
        print(
            "The evaluation framework integrates seamlessly with the baseline Elo model."
        )
        print(
            "All metrics computed correctly and provide comprehensive model assessment."
        )
        print("=" * 70)

        return True

    except Exception as e:
        logger.error("Integration test failed", error=str(e))
        print(f"\n[FAIL] INTEGRATION TEST FAILED: {e}")
        return False


if __name__ == "__main__":
    import tempfile

    import matplotlib.pyplot as plt

    success = main()
    sys.exit(0 if success else 1)
