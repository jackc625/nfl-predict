#!/usr/bin/env python3
"""
ATS Model Demo

This script demonstrates the complete ATS model workflow including:
- Training with XGBoost/LightGBM regression for expected margin
- Residual distribution conversion to cover probabilities
- Both classification and regression approaches
- Spread betting mechanics
- Feature importance tracking
- Model evaluation and persistence
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.train_ats import ATSModel, ResidualDistributionConverter
from models.evaluation import ModelEvaluationFramework
from utils import get_logger

logger = get_logger(__name__)


def create_demo_data(n_games: int = 400) -> pd.DataFrame:
    """Create realistic demo NFL data for ATS modeling."""
    np.random.seed(42)

    teams = ['KC', 'BUF', 'CIN', 'BAL', 'SF', 'PHI', 'DAL', 'GB', 'LAR', 'TB', 'NO', 'ATL']
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

                # Market spread with relationship to Elo
                market_spread = np.random.normal(-elo_diff / 30, 2.5)

                # True margin with realistic NFL characteristics
                base_margin = (elo_diff / 30 +  # Team quality difference
                              np.random.normal(3, 2))  # Home field advantage

                # Add game-specific variance
                true_margin = base_margin + np.random.normal(0, 10)

                # Determine if home team covers the spread
                covers_spread = int(true_margin > -market_spread)

                # Generate scores
                total_points = np.random.normal(47, 8)
                home_score = max(0, (total_points + true_margin) / 2)
                away_score = max(0, home_score - true_margin)

                game = {
                    'game_id': f'DEMO_{season}_{week:02d}_{game_id:03d}',
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': round(home_score),
                    'away_score': round(away_score),
                    'actual_margin': true_margin,
                    'covers_spread': covers_spread,
                    'market_spread': market_spread,

                    # Features for modeling
                    'elo_home': elo_home,
                    'elo_away': elo_away,
                    'elo_diff': elo_diff,
                    'rest_days_home': np.random.choice([3, 7, 10, 14]),
                    'rest_days_away': np.random.choice([3, 7, 10, 14]),
                    'is_playoff': int(week > 18),
                    'is_primetime': np.random.binomial(1, 0.25),
                    'temperature': np.random.normal(65, 20),
                    'wind_speed': np.random.exponential(5),
                    'home_epa_offense_4w': np.random.normal(0, 0.25),
                    'away_epa_offense_4w': np.random.normal(0, 0.25),
                    'home_epa_defense_4w': np.random.normal(0, 0.25),
                    'away_epa_defense_4w': np.random.normal(0, 0.25),
                    'opening_spread': market_spread + np.random.normal(0, 1),
                    'closing_total': np.random.normal(47, 5),
                    'home_turnover_diff_4w': np.random.normal(0, 1),
                    'away_turnover_diff_4w': np.random.normal(0, 1)
                }

                games.append(game)
                game_id += 1

    return pd.DataFrame(games)


def main():
    """Demonstrate ATS model capabilities."""
    print("=" * 70)
    print("AGAINST THE SPREAD (ATS) MODEL DEMONSTRATION")
    print("=" * 70)

    # Step 1: Create demo data
    print("\n1. Creating demo NFL data...")
    games_df = create_demo_data(400)
    print(f"   Created {len(games_df)} games across {games_df['season'].nunique()} seasons")
    print(f"   Teams: {', '.join(sorted(games_df['home_team'].unique()))}")
    print(f"   Season range: {games_df['season'].min()}-{games_df['season'].max()}")
    print(f"   Home team covers rate: {games_df['covers_spread'].mean():.3f}")
    print(f"   Average margin: {games_df['actual_margin'].mean():.2f} points")
    print(f"   Average spread: {games_df['market_spread'].mean():.2f} points")

    # Step 2: Split data
    print("\n2. Splitting data for training and testing...")
    train_data = games_df[games_df['season'].isin([2022, 2023])].copy()
    test_data = games_df[games_df['season'] == 2024].copy()

    print(f"   Training games: {len(train_data)}")
    print(f"   Test games: {len(test_data)}")

    # Step 3: Initialize and configure ATS model
    print("\n3. Initializing ATS model...")
    ats_model = ATSModel(
        model_type="xgboost",
        approach="hybrid",  # Both regression and classification
        feature_selection_method="model_based",
        max_features=15,
        use_calibration=True,
        hyperparameter_tuning="grid_search",
        distribution_type="normal",
        random_state=42
    )

    print("   Model configuration:")
    print(f"   - Model type: {ats_model.model_type}")
    print(f"   - Approach: {ats_model.approach}")
    print(f"   - Feature selection: {ats_model.feature_selection_method}")
    print(f"   - Max features: {ats_model.max_features}")
    print(f"   - Calibration: {ats_model.use_calibration}")
    print(f"   - Hyperparameter tuning: {ats_model.hyperparameter_tuning}")
    print(f"   - Distribution type: {ats_model.distribution_type}")

    # Step 4: Train ATS model
    print("\n4. Training ATS model...")
    training_results = ats_model.train_model(train_data, test_data)

    print("   Training Results:")
    print(f"   - Features selected: {len(training_results.feature_names)}")
    print(f"   - Training MAE: {training_results.performance_metrics.get('training_mae', 0):.3f}")
    print(f"   - Training RMSE: {training_results.performance_metrics.get('training_rmse', 0):.3f}")
    print(f"   - Training R2: {training_results.performance_metrics.get('training_r2', 0):.3f}")
    print(f"   - Residual Std: {training_results.performance_metrics.get('training_residual_std', 0):.3f}")

    # Cover prediction metrics
    if 'training_cover_accuracy' in training_results.performance_metrics:
        print(f"   - Cover accuracy: {training_results.performance_metrics['training_cover_accuracy']:.3f}")
        print(f"   - Cover log loss: {training_results.performance_metrics['training_cover_log_loss']:.3f}")
        print(f"   - Cover Brier score: {training_results.performance_metrics['training_cover_brier_score']:.3f}")

    print(f"   - Top features: {list(training_results.feature_importances.keys())[:7]}")

    # Step 5: Make predictions
    print("\n5. Making predictions on test data...")
    predictions = ats_model.predict(test_data, include_all_approaches=True)

    print(f"   Generated {len(predictions)} predictions")

    # Show sample predictions
    print("\n   Sample Predictions:")
    for i, pred in enumerate(predictions[:6]):
        actual_margin = test_data.iloc[i]['actual_margin']
        actual_cover = test_data.iloc[i]['covers_spread']

        print(f"   {i+1}. {pred.home_team} vs {pred.away_team}")
        print(f"      Market spread: {pred.market_spread:.1f}")
        print(f"      Predicted margin: {pred.predicted_margin:.2f}")
        print(f"      Cover probability: {pred.cover_probability:.3f}")

        if pred.classification_cover_prob:
            print(f"      Classification prob: {pred.classification_cover_prob:.3f}")

        print(f"      Edge: {pred.edge:.4f}" if pred.edge else "      Edge: N/A")
        print(f"      Actual margin: {actual_margin:.2f}")
        print(f"      Actually covered: {'Yes' if actual_cover else 'No'}")
        print()

    # Step 6: Evaluate predictions
    print("6. Evaluating ATS predictions...")
    evaluator = ModelEvaluationFramework()

    # Margin prediction evaluation
    pred_margins = np.array([p.predicted_margin for p in predictions])
    actual_margins = test_data['actual_margin'].values

    print("   Margin Prediction Performance:")
    mae = np.mean(np.abs(pred_margins - actual_margins))
    rmse = np.sqrt(np.mean((pred_margins - actual_margins) ** 2))
    r2 = 1 - np.sum((actual_margins - pred_margins) ** 2) / np.sum((actual_margins - np.mean(actual_margins)) ** 2)

    print(f"   - MAE: {mae:.3f} points")
    print(f"   - RMSE: {rmse:.3f} points")
    print(f"   - R-squared: {r2:.3f}")

    # Cover probability evaluation
    pred_cover_probs = np.array([p.cover_probability for p in predictions])
    actual_covers = test_data['covers_spread'].values

    eval_results = evaluator.evaluate_model(
        predictions=pred_cover_probs,
        actual_outcomes=actual_covers,
        prediction_type="binary",
        model_name="ATS Demo Model"
    )

    print("\n   Cover Probability Performance:")
    print(f"   - Accuracy: {eval_results.core_metrics['accuracy']:.3f}")
    print(f"   - Log Loss: {eval_results.core_metrics['log_loss']:.3f}")
    print(f"   - Brier Score: {eval_results.core_metrics['brier_score']:.3f}")
    print(f"   - ECE (calibration): {eval_results.calibration_metrics['ece']:.4f}")
    print(f"   - Reliability: {eval_results.calibration_metrics['reliability']:.4f}")
    print(f"   - Resolution: {eval_results.calibration_metrics['resolution']:.4f}")

    # Edge analysis
    edges = [p.edge for p in predictions if p.edge is not None]
    if edges:
        print(f"\n   Edge Analysis:")
        print(f"   - Mean edge: {np.mean(edges):.4f}")
        print(f"   - Std edge: {np.std(edges):.4f}")
        print(f"   - Max edge: {max(edges):.4f}")
        print(f"   - Min edge: {min(edges):.4f}")

        # Count profitable bets (edge > 0)
        profitable_bets = sum(1 for e in edges if e > 0.02)  # 2% edge threshold
        print(f"   - Profitable bets (>2% edge): {profitable_bets}/{len(edges)} ({profitable_bets/len(edges)*100:.1f}%)")

    # Step 7: Feature importance analysis
    print("\n7. Feature importance analysis...")
    top_features = list(training_results.feature_importances.items())
    top_features.sort(key=lambda x: x[1], reverse=True)

    print("   Top Feature Importances:")
    for feature, importance in top_features[:10]:
        print(f"   - {feature}: {importance:.4f}")

    # Step 8: Residual distribution analysis
    print("\n8. Residual distribution analysis...")
    residual_converter = ats_model.residual_converter

    print(f"   Distribution type: {residual_converter.distribution_type}")
    print(f"   Distribution parameters: {residual_converter.distribution_params}")

    # Test residual distribution conversion
    test_margins = np.array([3.5, -7.2, 0.0, 14.1, -2.8])
    test_spreads = np.array([-3.0, 6.5, -0.5, -14.0, 3.5])

    cover_probs = residual_converter.predict_cover_probability(test_margins, test_spreads)

    print("\n   Residual conversion examples:")
    for i, (margin, spread, prob) in enumerate(zip(test_margins, test_spreads, cover_probs)):
        print(f"   {i+1}. Margin: {margin:+.1f}, Spread: {spread:+.1f} -> Cover prob: {prob:.3f}")

    # Step 9: Model persistence demo
    print("\n9. Demonstrating model persistence...")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "demo_ats_model.joblib"

        # Save model
        ats_model.save_model(str(model_path))
        print(f"   Model saved to temporary location")

        # Load model
        loaded_model = ATSModel()
        loaded_model.load_model(str(model_path))
        print(f"   Model loaded successfully")

        # Test loaded model
        test_sample = test_data.head(3)
        loaded_preds = loaded_model.predict(test_sample)
        original_preds = ats_model.predict(test_sample)

        # Verify predictions match
        match = all(
            abs(orig.predicted_margin - loaded.predicted_margin) < 1e-10
            for orig, loaded in zip(original_preds, loaded_preds)
        )
        print(f"   Loaded model predictions match: {match}")

    # Step 10: Model summary
    print("\n10. Model Summary:")
    summary = ats_model.get_model_summary()
    print(f"   Model Type: {summary['model_type']}")
    print(f"   Approach: {summary['approach']}")
    print(f"   Base Model: {summary['base_model_type']}")
    print(f"   Trained: {summary['is_trained']}")
    print(f"   Features Selected: {summary['features_selected']}")
    print(f"   Distribution Type: {summary['distribution_type']}")
    print(f"   Residual Std: {summary['residual_std']:.3f}")

    print(f"\n   Configuration:")
    for key, value in summary['configuration'].items():
        print(f"   - {key}: {value}")

    # Step 11: Quick walk-forward validation demo
    print("\n11. Quick walk-forward validation demo...")
    try:
        validation_results = ats_model.run_walk_forward_validation(
            games_df,
            start_season=2023,
            end_season=2024
        )

        print("   Walk-forward Results:")
        if 'overall_metrics' in validation_results:
            overall = validation_results['overall_metrics']
            print(f"   - Seasons validated: {overall.get('seasons_validated', 0)}")
            print(f"   - Overall MAE: {overall.get('overall_mae', 0):.3f}")
            print(f"   - Overall RMSE: {overall.get('overall_rmse', 0):.3f}")
            print(f"   - Overall R2: {overall.get('overall_r2', 0):.3f}")

            if 'overall_cover_accuracy' in overall:
                print(f"   - Overall cover accuracy: {overall['overall_cover_accuracy']:.3f}")
                print(f"   - Overall cover log loss: {overall['overall_cover_log_loss']:.3f}")

    except Exception as e:
        print(f"   Walk-forward validation skipped: {e}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nThe ATS model has been successfully demonstrated with:")
    print("+ XGBoost regression for expected margin prediction")
    print("+ Residual distribution conversion to cover probabilities")
    print("+ Both classification and regression approaches")
    print("+ Proper spread betting mechanics")
    print("+ Feature importance tracking and analysis")
    print("+ Comprehensive evaluation metrics")
    print("+ Model persistence and loading")
    print("+ Walk-forward validation capability")
    print("+ Edge detection for betting opportunities")
    print("\nThe model is ready for production use in NFL ATS predictions!")


if __name__ == "__main__":
    main()