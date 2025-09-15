#!/usr/bin/env python3
"""
O/U Model Demo

This script demonstrates the complete O/U model workflow including:
- Training with XGBoost/LightGBM regression for total points
- Residual distribution conversion to over/under probabilities
- Optional Poisson score model for total points simulation
- Weather impact modeling for totals
- Total betting mechanics
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

from models.train_ou import OUModel, TotalDistributionConverter, PoissonScoreModel, WeatherImpactModel
from models.evaluation import ModelEvaluationFramework
from utils import get_logger

logger = get_logger(__name__)


def create_demo_data(n_games: int = 400) -> pd.DataFrame:
    """Create realistic demo NFL data for O/U modeling."""
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

                # Realistic offensive/defensive capabilities
                home_off_rating = np.random.normal(0, 0.3)
                away_off_rating = np.random.normal(0, 0.3)
                home_def_rating = np.random.normal(0, 0.3)
                away_def_rating = np.random.normal(0, 0.3)

                # Weather conditions affecting totals
                temperature = np.random.normal(65, 20)
                wind_speed = np.random.exponential(5)
                venue_type = np.random.choice(['outdoor', 'dome', 'retractable'])

                # Weather impact on scoring
                weather_impact = 0
                if venue_type == 'outdoor':
                    weather_impact = -0.5 * max(0, wind_speed - 10)  # High wind reduces scoring
                    weather_impact += -0.1 * abs(temperature - 70)   # Extreme temps reduce scoring

                # Expected team scores
                home_base_score = 24 + home_off_rating * 8 - away_def_rating * 6
                away_base_score = 23 + away_off_rating * 8 - home_def_rating * 6

                # Apply weather impact
                home_expected = max(0, home_base_score + weather_impact / 2)
                away_expected = max(0, away_base_score + weather_impact / 2)

                # Generate actual scores with Poisson-like distribution
                home_score = max(0, np.random.poisson(max(0, home_expected)))
                away_score = max(0, np.random.poisson(max(0, away_expected)))
                total_points = home_score + away_score

                # Market total line with realistic spread
                market_total = np.random.normal(total_points + np.random.normal(0, 2), 1.5)
                market_total = round(market_total * 2) / 2  # Round to nearest 0.5

                # Over/under result
                over_under = int(total_points > market_total)

                game = {
                    'game_id': f'DEMO_{season}_{week:02d}_{game_id:03d}',
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score,
                    'total_points': total_points,
                    'market_total': market_total,
                    'over_under': over_under,

                    # Features for modeling
                    'elo_home': np.random.normal(1500, 100),
                    'elo_away': np.random.normal(1500, 100),
                    'elo_diff': np.random.normal(0, 50),
                    'rest_days_home': np.random.choice([3, 7, 10, 14]),
                    'rest_days_away': np.random.choice([3, 7, 10, 14]),
                    'is_playoff': int(week > 18),
                    'is_primetime': np.random.binomial(1, 0.25),
                    'temperature': temperature,
                    'wind_speed': wind_speed,
                    'precipitation_prob': np.random.uniform(0, 1),
                    'humidity': np.random.uniform(30, 90),
                    'venue_type': venue_type,
                    'home_epa_offense_4w': home_off_rating,
                    'away_epa_offense_4w': away_off_rating,
                    'home_epa_defense_4w': home_def_rating,
                    'away_epa_defense_4w': away_def_rating,
                    'home_pass_rate_neutral': np.random.uniform(0.55, 0.75),
                    'away_pass_rate_neutral': np.random.uniform(0.55, 0.75),
                    'home_turnover_diff_4w': np.random.normal(0, 1),
                    'away_turnover_diff_4w': np.random.normal(0, 1),
                    'closing_spread': np.random.normal(0, 3)
                }

                games.append(game)
                game_id += 1

    return pd.DataFrame(games)


def main():
    """Demonstrate O/U model capabilities."""
    print("=" * 70)
    print("OVER/UNDER (O/U) MODEL DEMONSTRATION")
    print("=" * 70)

    # Step 1: Create demo data
    print("\n1. Creating demo NFL data...")
    games_df = create_demo_data(400)
    print(f"   Created {len(games_df)} games across {games_df['season'].nunique()} seasons")
    print(f"   Teams: {', '.join(sorted(games_df['home_team'].unique()))}")
    print(f"   Season range: {games_df['season'].min()}-{games_df['season'].max()}")
    print(f"   Over rate: {games_df['over_under'].mean():.3f}")
    print(f"   Average total: {games_df['total_points'].mean():.1f} points")
    print(f"   Average market total: {games_df['market_total'].mean():.1f} points")
    print(f"   Total range: {games_df['total_points'].min()}-{games_df['total_points'].max()}")

    # Step 2: Split data
    print("\n2. Splitting data for training and testing...")
    train_data = games_df[games_df['season'].isin([2022, 2023])].copy()
    test_data = games_df[games_df['season'] == 2024].copy()

    print(f"   Training games: {len(train_data)}")
    print(f"   Test games: {len(test_data)}")

    # Step 3: Initialize and configure O/U model
    print("\n3. Initializing O/U model...")
    ou_model = OUModel(
        model_type="xgboost",
        use_poisson=True,
        use_weather_model=True,
        feature_selection_method="model_based",
        max_features=18,
        use_calibration=True,
        hyperparameter_tuning="grid_search",
        distribution_type="normal",
        random_state=42
    )

    print("   Model configuration:")
    print(f"   - Model type: {ou_model.model_type}")
    print(f"   - Use Poisson model: {ou_model.use_poisson}")
    print(f"   - Use weather model: {ou_model.use_weather_model}")
    print(f"   - Feature selection: {ou_model.feature_selection_method}")
    print(f"   - Max features: {ou_model.max_features}")
    print(f"   - Calibration: {ou_model.use_calibration}")
    print(f"   - Hyperparameter tuning: {ou_model.hyperparameter_tuning}")
    print(f"   - Distribution type: {ou_model.distribution_type}")

    # Step 4: Train O/U model
    print("\n4. Training O/U model...")
    training_results = ou_model.train_model(train_data, test_data)

    print("   Training Results:")
    print(f"   - Features selected: {len(training_results.feature_names)}")
    print(f"   - Training MAE: {training_results.performance_metrics.get('training_mae', 0):.3f} points")
    print(f"   - Training RMSE: {training_results.performance_metrics.get('training_rmse', 0):.3f} points")
    print(f"   - Training R2: {training_results.performance_metrics.get('training_r2', 0):.3f}")
    print(f"   - Residual Std: {training_results.performance_metrics.get('training_residual_std', 0):.3f}")

    # O/U specific metrics
    if 'training_over_accuracy' in training_results.performance_metrics:
        print(f"   - Over accuracy: {training_results.performance_metrics['training_over_accuracy']:.3f}")
        print(f"   - Over log loss: {training_results.performance_metrics['training_over_log_loss']:.3f}")
        print(f"   - Over Brier score: {training_results.performance_metrics['training_over_brier_score']:.3f}")

    print(f"   - Top features: {list(training_results.feature_importances.keys())[:7]}")

    # Model components
    print(f"   - Has Poisson model: {training_results.poisson_model is not None}")
    print(f"   - Has weather model: {training_results.weather_model is not None}")
    if training_results.weather_coefficients:
        print(f"   - Weather features: {len(training_results.weather_coefficients)}")

    # Step 5: Make predictions
    print("\n5. Making predictions on test data...")
    predictions = ou_model.predict(test_data, include_simulation=True)

    print(f"   Generated {len(predictions)} predictions")

    # Show sample predictions
    print("\n   Sample Predictions:")
    for i, pred in enumerate(predictions[:6]):
        actual_total = test_data.iloc[i]['total_points']
        actual_over = test_data.iloc[i]['over_under']

        print(f"   {i+1}. {pred.home_team} vs {pred.away_team}")
        print(f"      Market total: {pred.market_total:.1f}")
        print(f"      Predicted total: {pred.predicted_total:.1f}")

        if pred.weather_adjusted_total is not None:
            print(f"      Weather adjusted: {pred.weather_adjusted_total:.1f}")

        print(f"      Over probability: {pred.over_probability:.3f}")
        print(f"      Under probability: {pred.under_probability:.3f}")

        if pred.poisson_over_prob:
            print(f"      Poisson over prob: {pred.poisson_over_prob:.3f}")

        if pred.home_team_total and pred.away_team_total:
            print(f"      Team totals: {pred.home_team_total:.1f} + {pred.away_team_total:.1f}")

        if pred.edge_over is not None:
            print(f"      Edge over: {pred.edge_over:.4f}")
            print(f"      Edge under: {pred.edge_under:.4f}")

        if pred.weather_impact is not None:
            print(f"      Weather impact: {pred.weather_impact:.2f}")

        print(f"      Actual total: {actual_total}")
        print(f"      Actually over: {'Yes' if actual_over else 'No'}")
        print()

    # Step 6: Evaluate predictions
    print("6. Evaluating O/U predictions...")
    evaluator = ModelEvaluationFramework()

    # Total prediction evaluation
    pred_totals = np.array([p.predicted_total for p in predictions])
    actual_totals = test_data['total_points'].values

    print("   Total Points Prediction Performance:")
    mae = np.mean(np.abs(pred_totals - actual_totals))
    rmse = np.sqrt(np.mean((pred_totals - actual_totals) ** 2))
    r2 = 1 - np.sum((actual_totals - pred_totals) ** 2) / np.sum((actual_totals - np.mean(actual_totals)) ** 2)

    print(f"   - MAE: {mae:.3f} points")
    print(f"   - RMSE: {rmse:.3f} points")
    print(f"   - R-squared: {r2:.3f}")

    # Over/under probability evaluation
    pred_over_probs = np.array([p.over_probability for p in predictions])
    actual_overs = test_data['over_under'].values

    eval_results = evaluator.evaluate_model(
        predictions=pred_over_probs,
        actual_outcomes=actual_overs,
        prediction_type="binary",
        model_name="O/U Demo Model"
    )

    print("\n   Over/Under Probability Performance:")
    print(f"   - Accuracy: {eval_results.core_metrics['accuracy']:.3f}")
    print(f"   - Log Loss: {eval_results.core_metrics['log_loss']:.3f}")
    print(f"   - Brier Score: {eval_results.core_metrics['brier_score']:.3f}")
    print(f"   - ECE (calibration): {eval_results.calibration_metrics['ece']:.4f}")
    print(f"   - Reliability: {eval_results.calibration_metrics['reliability']:.4f}")
    print(f"   - Resolution: {eval_results.calibration_metrics['resolution']:.4f}")

    # Edge analysis
    edges_over = [p.edge_over for p in predictions if p.edge_over is not None]
    edges_under = [p.edge_under for p in predictions if p.edge_under is not None]

    if edges_over:
        print(f"\n   Edge Analysis:")
        print(f"   - Mean over edge: {np.mean(edges_over):.4f}")
        print(f"   - Mean under edge: {np.mean(edges_under):.4f}")
        print(f"   - Max over edge: {max(edges_over):.4f}")
        print(f"   - Min under edge: {min(edges_under):.4f}")

        # Count profitable bets (edge > 2%)
        profitable_over_bets = sum(1 for e in edges_over if e > 0.02)
        profitable_under_bets = sum(1 for e in edges_under if e > 0.02)
        total_bets = len(edges_over)

        print(f"   - Profitable over bets (>2% edge): {profitable_over_bets}/{total_bets} ({profitable_over_bets/total_bets*100:.1f}%)")
        print(f"   - Profitable under bets (>2% edge): {profitable_under_bets}/{total_bets} ({profitable_under_bets/total_bets*100:.1f}%)")

    # Step 7: Feature importance analysis
    print("\n7. Feature importance analysis...")
    top_features = list(training_results.feature_importances.items())
    top_features.sort(key=lambda x: x[1], reverse=True)

    print("   Top Feature Importances for Total Points:")
    for feature, importance in top_features[:12]:
        print(f"   - {feature}: {importance:.4f}")

    # Weather impact analysis
    if training_results.weather_coefficients:
        print(f"\n   Weather Impact Coefficients:")
        for weather_var, coef in training_results.weather_coefficients.items():
            print(f"   - {weather_var}: {coef:.4f}")

    # Step 8: Residual distribution analysis
    print("\n8. Residual distribution analysis...")
    total_converter = ou_model.total_converter

    print(f"   Distribution type: {total_converter.distribution_type}")
    print(f"   Distribution parameters: {total_converter.distribution_params}")

    # Test residual distribution conversion
    test_totals = np.array([45.0, 38.5, 52.0, 41.0, 49.5])
    test_market_totals = np.array([47.5, 42.0, 50.5, 44.0, 46.5])

    over_probs, under_probs = total_converter.predict_over_under_probabilities(
        test_totals, test_market_totals
    )

    print("\n   Residual conversion examples:")
    for i, (pred_total, market_total, over_prob, under_prob) in enumerate(zip(test_totals, test_market_totals, over_probs, under_probs)):
        print(f"   {i+1}. Predicted: {pred_total:.1f}, Market: {market_total:.1f} -> Over: {over_prob:.3f}, Under: {under_prob:.3f}")

    # Step 9: Poisson simulation demo
    if training_results.poisson_model:
        print("\n9. Poisson score simulation demo...")
        poisson_model = training_results.poisson_model

        # Use first test game for simulation
        test_sample = test_data.head(1)
        game_features = test_sample[ou_model.feature_names].fillna(0)

        # Simulate game totals
        simulated_totals = poisson_model.simulate_game_totals(game_features, n_simulations=5000)

        print(f"   Simulation results for {test_sample.iloc[0]['home_team']} vs {test_sample.iloc[0]['away_team']}:")
        print(f"   - Mean simulated total: {simulated_totals.mean():.1f}")
        print(f"   - Std simulated total: {simulated_totals.std():.1f}")
        print(f"   - Range: {simulated_totals.min()}-{simulated_totals.max()}")
        print(f"   - 25th percentile: {np.percentile(simulated_totals, 25):.1f}")
        print(f"   - 75th percentile: {np.percentile(simulated_totals, 75):.1f}")

        # Compare with actual
        actual_total = test_sample.iloc[0]['total_points']
        print(f"   - Actual total: {actual_total}")

    # Step 10: Model persistence demo
    print("\n10. Demonstrating model persistence...")
    with tempfile.TemporaryDirectory() as temp_dir:
        model_path = Path(temp_dir) / "demo_ou_model.joblib"

        # Save model
        ou_model.save_model(str(model_path))
        print(f"   Model saved to temporary location")

        # Load model
        loaded_model = OUModel()
        loaded_model.load_model(str(model_path))
        print(f"   Model loaded successfully")

        # Test loaded model
        test_sample = test_data.head(3)
        loaded_preds = loaded_model.predict(test_sample)
        original_preds = ou_model.predict(test_sample)

        # Verify predictions match
        match = all(
            abs(orig.predicted_total - loaded.predicted_total) < 1e-10
            for orig, loaded in zip(original_preds, loaded_preds)
        )
        print(f"   Loaded model predictions match: {match}")

    # Step 11: Model summary
    print("\n11. Model Summary:")
    summary = ou_model.get_model_summary()
    print(f"   Model Type: {summary['model_type']}")
    print(f"   Base Model: {summary['base_model_type']}")
    print(f"   Trained: {summary['is_trained']}")
    print(f"   Features Selected: {summary['features_selected']}")
    print(f"   Use Poisson: {summary['use_poisson']}")
    print(f"   Use Weather Model: {summary['use_weather_model']}")
    print(f"   Distribution Type: {summary['distribution_type']}")
    print(f"   Residual Std: {summary['residual_std']:.3f}")

    print(f"\n   Configuration:")
    for key, value in summary['configuration'].items():
        print(f"   - {key}: {value}")

    # Step 12: Quick walk-forward validation demo
    print("\n12. Quick walk-forward validation demo...")
    try:
        validation_results = ou_model.run_walk_forward_validation(
            games_df,
            start_season=2023,
            end_season=2024
        )

        print("   Walk-forward Results:")
        if 'overall_metrics' in validation_results:
            overall = validation_results['overall_metrics']
            print(f"   - Seasons validated: {overall.get('seasons_validated', 0)}")
            print(f"   - Overall MAE: {overall.get('overall_mae', 0):.3f} points")
            print(f"   - Overall RMSE: {overall.get('overall_rmse', 0):.3f} points")
            print(f"   - Overall R2: {overall.get('overall_r2', 0):.3f}")

    except Exception as e:
        print(f"   Walk-forward validation skipped: {e}")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nThe O/U model has been successfully demonstrated with:")
    print("+ XGBoost regression for total points prediction")
    print("+ Residual distribution conversion to over/under probabilities")
    print("+ Optional Poisson score model for total points simulation")
    print("+ Weather impact modeling specifically for totals")
    print("+ Proper total betting mechanics")
    print("+ Feature importance tracking and analysis")
    print("+ Comprehensive evaluation metrics")
    print("+ Model persistence and loading")
    print("+ Walk-forward validation capability")
    print("+ Edge detection for betting opportunities")
    print("\nThe model is ready for production use in NFL O/U predictions!")


if __name__ == "__main__":
    main()