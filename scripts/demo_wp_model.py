#!/usr/bin/env python3
"""
Win Probability Model Demo

This script demonstrates the complete WP model workflow including:
- Training on synthetic data
- Making predictions
- Model evaluation
- Persistence and loading
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.evaluation import ModelEvaluationFramework
from models.train_wp import WinProbabilityModel
from utils import get_logger

logger = get_logger(__name__)


def create_demo_data(n_games: int = 300) -> pd.DataFrame:
    """Create realistic demo NFL data."""
    np.random.seed(42)

    teams = [
        "KC",
        "BUF",
        "CIN",
        "BAL",
        "SF",
        "PHI",
        "DAL",
        "GB",
        "LAR",
        "TB",
        "NO",
        "ATL",
    ]
    seasons = [2022, 2023, 2024]

    games = []
    game_id = 1

    for season in seasons:
        for week in range(1, 19):
            games_this_week = n_games // (len(seasons) * 18)

            for _ in range(games_this_week):
                home_team = np.random.choice(teams)
                away_team = np.random.choice([t for t in teams if t != home_team])

                # Realistic Elo ratings
                elo_home = np.random.normal(1500, 100)
                elo_away = np.random.normal(1500, 100)
                elo_diff = elo_home - elo_away

                # Home field advantage
                hfa = np.random.normal(65, 10)
                total_advantage = elo_diff + hfa

                # Win probability with some noise
                win_prob = 1 / (1 + 10 ** (-total_advantage / 400))
                home_wins = np.random.binomial(1, win_prob)

                game = {
                    "game_id": f"DEMO_{season}_{week:02d}_{game_id:03d}",
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                    "home_wins": home_wins,
                    # Features
                    "elo_home": elo_home,
                    "elo_away": elo_away,
                    "elo_diff": elo_diff,
                    "rest_days_home": np.random.choice([3, 7, 10, 14]),
                    "rest_days_away": np.random.choice([3, 7, 10, 14]),
                    "is_playoff": int(week > 18),
                    "is_primetime": np.random.binomial(1, 0.25),
                    "temperature": np.random.normal(65, 20),
                    "wind_speed": np.random.exponential(5),
                    "home_epa_offense_4w": np.random.normal(0, 0.25),
                    "away_epa_offense_4w": np.random.normal(0, 0.25),
                    "home_epa_defense_4w": np.random.normal(0, 0.25),
                    "away_epa_defense_4w": np.random.normal(0, 0.25),
                    "opening_spread": np.random.normal(-elo_diff / 30, 2.5),
                    "closing_total": np.random.normal(47, 5),
                }

                games.append(game)
                game_id += 1

    return pd.DataFrame(games)


def main():
    """Demonstrate WP model capabilities."""
    print("=" * 70)
    print("WIN PROBABILITY MODEL DEMONSTRATION")
    print("=" * 70)

    # Step 1: Create demo data
    print("\n1. Creating demo NFL data...")
    games_df = create_demo_data(300)
    print(
        f"   Created {len(games_df)} games across {games_df['season'].nunique()} seasons"
    )
    print(f"   Teams: {', '.join(sorted(games_df['home_team'].unique()))}")
    print(f"   Season range: {games_df['season'].min()}-{games_df['season'].max()}")
    print(f"   Home win rate: {games_df['home_wins'].mean():.3f}")

    # Step 2: Split data
    print("\n2. Splitting data for training and testing...")
    train_data = games_df[games_df["season"].isin([2022, 2023])].copy()
    test_data = games_df[games_df["season"] == 2024].copy()

    print(f"   Training games: {len(train_data)}")
    print(f"   Test games: {len(test_data)}")

    # Step 3: Initialize and configure model
    print("\n3. Initializing Win Probability model...")
    wp_model = WinProbabilityModel(
        feature_selection_method="recursive",
        max_features=12,
        regularization_type="elasticnet",
        use_calibration=True,
        hyperparameter_tuning="grid_search",
        random_state=42,
    )

    print("   Model configuration:")
    print(f"   - Feature selection: {wp_model.feature_selection_method}")
    print(f"   - Regularization: {wp_model.regularization_type}")
    print(f"   - Max features: {wp_model.max_features}")
    print(f"   - Calibration: {wp_model.use_calibration}")
    print(f"   - Hyperparameter tuning: {wp_model.hyperparameter_tuning}")

    # Step 4: Train model
    print("\n4. Training Win Probability model...")
    training_results = wp_model.train_model(train_data)

    print("   Training Results:")
    print(f"   - Features selected: {len(training_results.feature_names)}")
    print(f"   - Top features: {list(training_results.feature_importances.keys())[:5]}")
    print(
        f"   - Training accuracy: {training_results.performance_metrics.get('training_accuracy', 0):.3f}"
    )
    print(
        f"   - Training log loss: {training_results.performance_metrics.get('training_log_loss', 0):.3f}"
    )
    print(
        f"   - Training Brier score: {training_results.performance_metrics.get('training_brier_score', 0):.3f}"
    )
    print(f"   - Final hyperparameters: {training_results.hyperparameters}")

    # Step 5: Make predictions
    print("\n5. Making predictions on test data...")
    predictions = wp_model.predict(test_data, calibrated=True)

    print(f"   Generated {len(predictions)} predictions")

    # Show sample predictions
    print("\n   Sample Predictions:")
    for i, pred in enumerate(predictions[:5]):
        actual = test_data.iloc[i]["home_wins"]
        print(f"   {i + 1}. {pred.home_team} vs {pred.away_team}")
        print(f"      Raw prob: {pred.raw_win_probability:.3f}")
        print(f"      Cal prob: {pred.calibrated_win_probability:.3f}")
        print(f"      Confidence: {pred.prediction_confidence:.3f}")
        print(f"      Actual: {'W' if actual else 'L'}")
        print()

    # Step 6: Evaluate predictions
    print("6. Evaluating predictions...")
    evaluator = ModelEvaluationFramework()

    pred_probs = np.array([p.calibrated_win_probability for p in predictions])
    actual_outcomes = test_data["home_wins"].values

    eval_results = evaluator.evaluate_model(
        predictions=pred_probs,
        actual_outcomes=actual_outcomes,
        prediction_type="binary",
        model_name="WP Demo Model",
    )

    print("   Test Set Performance:")
    print(f"   - Accuracy: {eval_results.core_metrics['accuracy']:.3f}")
    print(f"   - Log Loss: {eval_results.core_metrics['log_loss']:.3f}")
    print(f"   - Brier Score: {eval_results.core_metrics['brier_score']:.3f}")
    print(f"   - ECE (calibration): {eval_results.calibration_metrics['ece']:.4f}")
    print(f"   - Reliability: {eval_results.calibration_metrics['reliability']:.4f}")
    print(f"   - Resolution: {eval_results.calibration_metrics['resolution']:.4f}")

    # Statistical significance
    if "accuracy_vs_random" in eval_results.statistical_tests:
        acc_test = eval_results.statistical_tests["accuracy_vs_random"]
        print(
            f"   - Significant vs random: {acc_test['significant']} (p={acc_test['p_value']:.4f})"
        )

    # Step 7: Model persistence demo
    print("\n7. Demonstrating model persistence...")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "demo_wp_model.joblib"

        # Save model
        wp_model.save_model(str(model_path))
        print("   Model saved to temporary location")

        # Load model
        loaded_model = WinProbabilityModel()
        loaded_model.load_model(str(model_path))
        print("   Model loaded successfully")

        # Test loaded model
        test_sample = test_data.head(3)
        loaded_preds = loaded_model.predict(test_sample)
        original_preds = wp_model.predict(test_sample)

        # Verify predictions match
        match = all(
            abs(orig.raw_win_probability - loaded.raw_win_probability) < 1e-10
            for orig, loaded in zip(original_preds, loaded_preds, strict=False)
        )
        print(f"   Loaded model predictions match: {match}")

    # Step 8: Model summary
    print("\n8. Model Summary:")
    summary = wp_model.get_model_summary()
    print(f"   Model Type: {summary['model_type']}")
    print(f"   Trained: {summary['is_trained']}")
    print(f"   Features Selected: {summary['model_details']['features_selected']}")
    print(f"   Regularization: {summary['configuration']['regularization_type']}")
    print(
        f"   Feature Selection: {summary['configuration']['feature_selection_method']}"
    )
    print(f"   Calibrated: {summary['calibration_used']}")

    print("\n   Top Feature Importances:")
    for feat, imp in list(summary["model_details"]["feature_importances"].items())[:8]:
        print(f"   - {feat}: {imp:.4f}")

    # Step 9: Walk-forward validation demo (quick version)
    print("\n9. Quick walk-forward validation demo...")
    try:
        validation_results = wp_model.run_walk_forward_validation(
            games_df, start_season=2023, end_season=2024
        )

        print("   Walk-forward Results:")
        if "overall_metrics" in validation_results:
            print(
                f"   - Seasons validated: {validation_results['overall_metrics'].get('seasons_validated', 0)}"
            )
            print(
                f"   - Overall accuracy: {validation_results['overall_metrics'].get('overall_accuracy', 0):.3f}"
            )
            print(
                f"   - Overall log loss: {validation_results['overall_metrics'].get('overall_log_loss', 0):.3f}"
            )
    except Exception as e:
        print(f"   Walk-forward validation skipped: {e}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nThe Win Probability model has been successfully demonstrated with:")
    print("+ Feature engineering and selection")
    print("+ Hyperparameter tuning with cross-validation")
    print("+ Model training with regularization")
    print("+ Probability calibration")
    print("+ Comprehensive evaluation")
    print("+ Model persistence and loading")
    print("+ Walk-forward validation capability")
    print("\nThe model is ready for production use!")


if __name__ == "__main__":
    main()
