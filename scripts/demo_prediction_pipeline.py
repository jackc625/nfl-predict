#!/usr/bin/env python3
"""
Prediction Pipeline Demo

This script demonstrates the complete unified prediction pipeline workflow:
- Training WP, ATS, and O/U models
- Combining models in unified prediction pipeline
- Generating fair lines and probabilities
- Calculating edges vs market lines
- Creating structured prediction output
- Bet recommendation engine
- Export functionality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.prediction_pipeline import NFLPredictionPipeline, BetType
from models.train_wp import WinProbabilityModel
from models.train_ats import ATSModel
from models.train_ou import OUModel
from utils import get_logger

logger = get_logger(__name__)


def create_demo_data(n_games: int = 200) -> pd.DataFrame:
    """Create realistic demo NFL data for pipeline demonstration."""
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

                # Team strength factors
                home_strength = np.random.normal(0, 0.3)
                away_strength = np.random.normal(0, 0.3)

                # Generate realistic game outcomes
                home_advantage = 3 + np.random.normal(0, 1)
                true_margin = (home_strength - away_strength) * 10 + home_advantage + np.random.normal(0, 8)

                # Generate scores
                total_base = 47 + (home_strength + away_strength) * 10
                total_points = max(20, np.random.normal(total_base, 6))
                home_score = max(0, (total_points + true_margin) / 2)
                away_score = max(0, total_points - home_score)

                # Market lines with realistic relationships
                market_spread = -true_margin + np.random.normal(0, 2)
                market_total = total_points + np.random.normal(0, 3)
                market_total = round(market_total * 2) / 2  # Round to nearest 0.5

                # Market moneylines based on spread
                if market_spread < -7:
                    ml_home = np.random.randint(-400, -250)
                    ml_away = np.random.randint(200, 350)
                elif market_spread < -3:
                    ml_home = np.random.randint(-200, -120)
                    ml_away = np.random.randint(100, 180)
                elif market_spread < 3:
                    ml_home = np.random.randint(-130, 130)
                    ml_away = np.random.randint(-130, 130)
                elif market_spread < 7:
                    ml_home = np.random.randint(100, 180)
                    ml_away = np.random.randint(-200, -120)
                else:
                    ml_home = np.random.randint(200, 350)
                    ml_away = np.random.randint(-400, -250)

                # Game outcomes
                home_wins = int(true_margin > 0)
                covers_spread = int(true_margin > -market_spread)
                over_under = int(total_points > market_total)

                game = {
                    'game_id': f'DEMO_{season}_{week:02d}_{game_id:03d}',
                    'season': season,
                    'week': week,
                    'home_team': home_team,
                    'away_team': away_team,

                    # Outcomes
                    'home_wins': home_wins,
                    'home_score': round(home_score),
                    'away_score': round(away_score),
                    'actual_margin': true_margin,
                    'covers_spread': covers_spread,
                    'total_points': round(home_score + away_score),
                    'over_under': over_under,

                    # Market lines
                    'market_spread': market_spread,
                    'market_total': market_total,
                    'market_moneyline_home': int(ml_home),
                    'market_moneyline_away': int(ml_away),

                    # Features
                    'elo_home': np.random.normal(1500, 100),
                    'elo_away': np.random.normal(1500, 100),
                    'elo_diff': np.random.normal(0, 50),
                    'rest_days_home': np.random.choice([3, 7, 10, 14]),
                    'rest_days_away': np.random.choice([3, 7, 10, 14]),
                    'is_playoff': int(week > 18),
                    'is_primetime': np.random.binomial(1, 0.25),
                    'temperature': np.random.normal(65, 20),
                    'wind_speed': np.random.exponential(5),
                    'precipitation_prob': np.random.uniform(0, 1),
                    'humidity': np.random.uniform(30, 90),
                    'venue_type': np.random.choice(['outdoor', 'dome', 'retractable']),
                    'home_epa_offense_4w': home_strength,
                    'away_epa_offense_4w': away_strength,
                    'home_epa_defense_4w': -away_strength * 0.8,
                    'away_epa_defense_4w': -home_strength * 0.8,
                    'home_pass_rate_neutral': np.random.uniform(0.55, 0.75),
                    'away_pass_rate_neutral': np.random.uniform(0.55, 0.75),
                    'home_turnover_diff_4w': np.random.normal(0, 1),
                    'away_turnover_diff_4w': np.random.normal(0, 1)
                }

                games.append(game)
                game_id += 1

    return pd.DataFrame(games)


def main():
    """Demonstrate unified prediction pipeline capabilities."""
    print("=" * 70)
    print("UNIFIED PREDICTION PIPELINE DEMONSTRATION")
    print("=" * 70)

    # Step 1: Create demo data
    print("\n1. Creating demo NFL data with market lines...")
    games_df = create_demo_data(200)
    print(f"   Created {len(games_df)} games across {games_df['season'].nunique()} seasons")
    print(f"   Teams: {', '.join(sorted(games_df['home_team'].unique()))}")
    print(f"   Season range: {games_df['season'].min()}-{games_df['season'].max()}")
    print(f"   Market spread range: {games_df['market_spread'].min():.1f} to {games_df['market_spread'].max():.1f}")
    print(f"   Market total range: {games_df['market_total'].min():.1f} to {games_df['market_total'].max():.1f}")

    # Step 2: Split data and train models
    print("\n2. Training individual models...")
    train_data = games_df[games_df['season'].isin([2022, 2023])].copy()
    test_data = games_df[games_df['season'] == 2024].copy()

    print(f"   Training games: {len(train_data)}")
    print(f"   Test games: {len(test_data)}")

    # Train WP Model
    print("\n   Training Win Probability model...")
    wp_model = WinProbabilityModel(
        feature_selection_method="model_based",
        max_features=12,
        hyperparameter_tuning="random_search",
        use_calibration=True,
        random_state=42
    )
    wp_results = wp_model.train_model(train_data)
    print(f"   WP Model - Training accuracy: {wp_results.performance_metrics.get('training_accuracy', 0):.3f}")

    # Train ATS Model
    print("\n   Training ATS model...")
    ats_model = ATSModel(
        model_type="xgboost",
        approach="hybrid",
        feature_selection_method="model_based",
        max_features=15,
        hyperparameter_tuning="random_search",
        use_calibration=True,
        random_state=42
    )
    ats_results = ats_model.train_model(train_data)
    print(f"   ATS Model - Training MAE: {ats_results.performance_metrics.get('training_mae', 0):.3f}")

    # Train O/U Model
    print("\n   Training O/U model...")
    ou_model = OUModel(
        model_type="xgboost",
        use_poisson=True,
        use_weather_model=True,
        feature_selection_method="model_based",
        max_features=18,
        hyperparameter_tuning="random_search",
        use_calibration=True,
        random_state=42
    )
    ou_results = ou_model.train_model(train_data)
    print(f"   O/U Model - Training MAE: {ou_results.performance_metrics.get('training_mae', 0):.3f} points")

    # Step 3: Initialize prediction pipeline
    print("\n3. Initializing unified prediction pipeline...")
    pipeline = NFLPredictionPipeline(
        wp_model=wp_model,
        ats_model=ats_model,
        ou_model=ou_model,
        min_edge_threshold=0.02,  # 2% minimum edge
        min_confidence_threshold=0.65,
        max_kelly_fraction=0.25
    )

    pipeline_summary = pipeline.get_pipeline_summary()
    print("   Pipeline Configuration:")
    print(f"   - WP Model loaded: {pipeline_summary['models_loaded']['wp_model']}")
    print(f"   - ATS Model loaded: {pipeline_summary['models_loaded']['ats_model']}")
    print(f"   - O/U Model loaded: {pipeline_summary['models_loaded']['ou_model']}")
    print(f"   - Min edge threshold: {pipeline_summary['configuration']['min_edge_threshold']}")
    print(f"   - Max Kelly fraction: {pipeline_summary['configuration']['max_kelly_fraction']}")
    print(f"   - Supported bet types: {len(pipeline_summary['supported_bet_types'])}")

    # Step 4: Generate unified predictions
    print("\n4. Generating unified predictions...")
    predictions = pipeline.predict_games(test_data)
    print(f"   Generated {len(predictions)} unified predictions")

    # Step 5: Display sample predictions
    print("\n5. Sample unified predictions:")
    for i, pred in enumerate(predictions[:5]):
        print(f"\n   Game {i+1}: {pred.home_team} vs {pred.away_team}")
        print(f"   ----------------------------------------")

        # Win Probability
        print(f"   Win Probability:")
        print(f"     {pred.home_team} win: {pred.wp_home_probability:.3f} ({pred.wp_home_probability*100:.1f}%)")
        print(f"     {pred.away_team} win: {pred.wp_away_probability:.3f} ({pred.wp_away_probability*100:.1f}%)")

        # Spread Analysis
        print(f"   Spread Analysis:")
        print(f"     Predicted margin: {pred.predicted_margin:+.1f}")
        print(f"     Market spread: {pred.market_spread:+.1f}")
        print(f"     Cover probability: {pred.ats_cover_probability:.3f}")

        # Total Analysis
        print(f"   Total Analysis:")
        print(f"     Predicted total: {pred.predicted_total:.1f}")
        print(f"     Market total: {pred.market_total:.1f}")
        print(f"     Over probability: {pred.over_probability:.3f}")
        print(f"     Under probability: {pred.under_probability:.3f}")

        # Market Lines
        if pred.market_moneyline_home:
            print(f"   Market Lines:")
            print(f"     {pred.home_team} ML: {pred.market_moneyline_home:+d}")
            print(f"     {pred.away_team} ML: {pred.market_moneyline_away:+d}")

        # Fair Lines (show a few examples)
        print(f"   Fair Lines:")
        if BetType.MONEYLINE_HOME in pred.fair_lines:
            fair_ml = pred.fair_lines[BetType.MONEYLINE_HOME]
            print(f"     {pred.home_team} fair ML: {fair_ml.fair_odds_american:+d} ({fair_ml.fair_probability:.3f})")

        if BetType.OVER in pred.fair_lines:
            fair_over = pred.fair_lines[BetType.OVER]
            print(f"     Over fair odds: {fair_over.fair_odds_american:+d} ({fair_over.fair_probability:.3f})")

        # Edges
        if pred.edges:
            print(f"   Top Edges:")
            sorted_edges = sorted(pred.edges.items(), key=lambda x: x[1].edge, reverse=True)
            for bet_type, edge in sorted_edges[:3]:
                if edge.edge > 0.01:  # Show positive edges
                    print(f"     {bet_type.value}: {edge.edge:+.3f} ({edge.edge*100:+.1f}%), "
                          f"EV: {edge.expected_value:+.3f}, Kelly: {edge.kelly_fraction:.3f}")

        # Recommendations
        if pred.recommendations:
            print(f"   Recommendations:")
            for rec in pred.recommendations[:2]:  # Show top 2
                print(f"     {rec.bet_type.value}: {rec.recommendation} - {rec.reasoning}")

        # Diagnostics
        print(f"   Model Diagnostics:")
        print(f"     Model agreement: {pred.model_agreement:.3f}")
        print(f"     Prediction confidence: {pred.prediction_confidence:.3f}")

    # Step 6: Analyze overall results
    print("\n6. Overall prediction analysis...")

    # Edge distribution
    all_edges = []
    profitable_bets = []
    strong_bets = []

    for pred in predictions:
        for bet_type, edge in pred.edges.items():
            all_edges.append(edge.edge)
            if edge.edge > 0.02:  # 2% edge threshold
                profitable_bets.append((pred.game_id, bet_type.value, edge.edge))
            if edge.edge > 0.05:  # 5% edge for strong bets
                strong_bets.append((pred.game_id, bet_type.value, edge.edge))

    print(f"   Edge Analysis:")
    print(f"   - Total betting opportunities analyzed: {len(all_edges)}")
    print(f"   - Mean edge: {np.mean(all_edges):+.4f}")
    print(f"   - Profitable bets (>2% edge): {len(profitable_bets)} ({len(profitable_bets)/len(all_edges)*100:.1f}%)")
    print(f"   - Strong bets (>5% edge): {len(strong_bets)} ({len(strong_bets)/len(all_edges)*100:.1f}%)")

    if profitable_bets:
        print(f"   - Best edge: {max(profitable_bets, key=lambda x: x[2])[2]:.3f}")
        print(f"   - Average profitable edge: {np.mean([bet[2] for bet in profitable_bets]):.3f}")

    # Recommendation distribution
    all_recommendations = []
    for pred in predictions:
        all_recommendations.extend([rec.recommendation for rec in pred.recommendations])

    if all_recommendations:
        from collections import Counter
        rec_counts = Counter(all_recommendations)
        print(f"\n   Recommendation Distribution:")
        for rec_type, count in rec_counts.most_common():
            print(f"   - {rec_type}: {count} ({count/len(all_recommendations)*100:.1f}%)")

    # Model agreement analysis
    agreements = [pred.model_agreement for pred in predictions if pred.model_agreement is not None]
    if agreements:
        print(f"\n   Model Agreement Analysis:")
        print(f"   - Mean agreement: {np.mean(agreements):.3f}")
        print(f"   - Min agreement: {min(agreements):.3f}")
        print(f"   - Max agreement: {max(agreements):.3f}")
        print(f"   - High agreement games (>0.8): {sum(1 for a in agreements if a > 0.8)}")

    # Step 7: Export demonstrations
    print("\n7. Export functionality demonstration...")

    # DataFrame export
    df_export = pipeline.export_predictions(predictions[:10], format="dataframe")
    print(f"   DataFrame export: {df_export.shape} shape")
    print(f"   Columns: {list(df_export.columns)}")

    # JSON export (first 3 games for readability)
    json_export = pipeline.export_predictions(predictions[:3], format="json")
    print(f"   JSON export: {len(json_export)} characters")

    # Show sample JSON structure
    json_data = json.loads(json_export)
    sample_game = json_data[0]
    print(f"   Sample JSON keys: {list(sample_game.keys())}")
    print(f"   Predictions keys: {list(sample_game['predictions'].keys())}")
    print(f"   Fair lines count: {len(sample_game['fair_lines'])}")
    print(f"   Edges count: {len(sample_game['edges'])}")

    # Step 8: Summary statistics
    print("\n8. Pipeline performance summary...")

    # Win probability accuracy (on available data)
    wp_accuracy_data = []
    ats_accuracy_data = []
    ou_accuracy_data = []

    for i, pred in enumerate(predictions):
        game_data = test_data.iloc[i]

        # WP accuracy
        predicted_home_win = pred.wp_home_probability > 0.5
        actual_home_win = game_data['home_wins'] == 1
        wp_accuracy_data.append(predicted_home_win == actual_home_win)

        # ATS accuracy
        predicted_cover = pred.ats_cover_probability > 0.5
        actual_cover = game_data['covers_spread'] == 1
        ats_accuracy_data.append(predicted_cover == actual_cover)

        # O/U accuracy
        predicted_over = pred.over_probability > 0.5
        actual_over = game_data['over_under'] == 1
        ou_accuracy_data.append(predicted_over == actual_over)

    print(f"   Model Accuracy on Test Data:")
    print(f"   - Win Probability: {np.mean(wp_accuracy_data):.3f} ({np.sum(wp_accuracy_data)}/{len(wp_accuracy_data)})")
    print(f"   - ATS Cover: {np.mean(ats_accuracy_data):.3f} ({np.sum(ats_accuracy_data)}/{len(ats_accuracy_data)})")
    print(f"   - Over/Under: {np.mean(ou_accuracy_data):.3f} ({np.sum(ou_accuracy_data)}/{len(ou_accuracy_data)})")

    # Margin and total accuracy
    margin_errors = []
    total_errors = []

    for i, pred in enumerate(predictions):
        game_data = test_data.iloc[i]
        margin_errors.append(abs(pred.predicted_margin - game_data['actual_margin']))
        total_errors.append(abs(pred.predicted_total - game_data['total_points']))

    print(f"\n   Prediction Accuracy:")
    print(f"   - Mean absolute margin error: {np.mean(margin_errors):.2f} points")
    print(f"   - Mean absolute total error: {np.mean(total_errors):.2f} points")

    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 70)
    print("\nThe unified prediction pipeline has been successfully demonstrated with:")
    print("+ Integration of WP, ATS, and O/U models")
    print("+ Fair line generation for all bet types")
    print("+ Edge calculation vs market lines")
    print("+ Structured prediction output with all betting information")
    print("+ Bet recommendation engine with Kelly sizing")
    print("+ Model agreement analysis")
    print("+ Comprehensive export functionality (DataFrame, JSON)")
    print("+ Performance analytics and accuracy metrics")
    print("\nThe pipeline is ready for production use in comprehensive NFL betting analysis!")


if __name__ == "__main__":
    main()