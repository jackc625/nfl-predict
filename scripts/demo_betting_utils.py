"""Demo script showcasing betting utilities and Expected Value calculations."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from utils.betting_utils import (
    BetType,
    calculate_moneyline_ev,
    calculate_spread_ev,
    calculate_total_ev,
    analyze_game_betting_opportunities,
    summarize_betting_session
)
from utils.probability_utils import moneyline_to_probability


def demo_moneyline_calculations():
    """Demonstrate moneyline EV calculations."""
    print("=== MONEYLINE EV CALCULATIONS ===")

    scenarios = [
        {"model_prob": 0.60, "odds": -110, "description": "Model 60% vs Standard -110"},
        {"model_prob": 0.55, "odds": -110, "description": "Model 55% vs Standard -110"},
        {"model_prob": 0.65, "odds": +150, "description": "Model 65% vs Underdog +150"},
        {"model_prob": 0.75, "odds": -200, "description": "Model 75% vs Heavy Favorite -200"}
    ]

    for scenario in scenarios:
        result = calculate_moneyline_ev(
            model_prob=scenario["model_prob"],
            market_odds=scenario["odds"],
            devig=True,
            opposite_odds=-110
        )

        print(f"\n{scenario['description']}:")
        print(f"  Market Prob (devigged): {result.market_prob:.3f}")
        print(f"  Edge: {result.edge:.3f} ({result.edge*100:.1f}%)")
        print(f"  Expected Value: ${result.expected_value:.2f}")

        if result.edge > 0.02:  # 2% threshold
            print(f"  [+] VIABLE BET (Edge > 2%)")
        else:
            print(f"  [-] No bet (Edge too small)")


def demo_spread_calculations():
    """Demonstrate spread EV calculations."""
    print("\n\n=== SPREAD EV CALCULATIONS ===")

    scenarios = [
        {
            "model_margin": 7.0,
            "market_spread": -3.5,
            "side": "home",
            "description": "Home team: Model +7, Market -3.5"
        },
        {
            "model_margin": -2.0,
            "market_spread": -6.5,
            "side": "away",
            "description": "Away team: Model -2 (home), Market -6.5 (home)"
        },
        {
            "model_margin": 3.0,
            "market_spread": -3.5,
            "side": "home",
            "description": "Close call: Model +3, Market -3.5"
        }
    ]

    for scenario in scenarios:
        result = calculate_spread_ev(
            model_margin=scenario["model_margin"],
            model_margin_std=14.0,  # Typical NFL margin std
            market_spread=scenario["market_spread"],
            side=scenario["side"]
        )

        print(f"\n{scenario['description']}:")
        print(f"  Cover Probability: {result.model_prob:.3f}")
        print(f"  Market Prob (devigged): {result.market_prob:.3f}")
        print(f"  Edge: {result.edge:.3f} ({result.edge*100:.1f}%)")
        print(f"  Expected Value: ${result.expected_value:.2f}")

        if result.edge > 0.02:
            print(f"  [+] VIABLE BET")
        else:
            print(f"  [-] No bet")


def demo_total_calculations():
    """Demonstrate total (over/under) EV calculations."""
    print("\n\n=== TOTAL EV CALCULATIONS ===")

    scenarios = [
        {
            "model_total": 52.0,
            "market_total": 45.0,
            "side": "over",
            "description": "Over bet: Model 52, Market 45"
        },
        {
            "model_total": 38.0,
            "market_total": 47.0,
            "side": "under",
            "description": "Under bet: Model 38, Market 47"
        },
        {
            "model_total": 45.5,
            "market_total": 45.0,
            "side": "over",
            "description": "Close over: Model 45.5, Market 45"
        }
    ]

    for scenario in scenarios:
        result = calculate_total_ev(
            model_total=scenario["model_total"],
            model_total_std=10.5,  # Typical NFL total std
            market_total=scenario["market_total"],
            side=scenario["side"]
        )

        print(f"\n{scenario['description']}:")
        print(f"  {scenario['side'].title()} Probability: {result.model_prob:.3f}")
        print(f"  Market Prob (devigged): {result.market_prob:.3f}")
        print(f"  Edge: {result.edge:.3f} ({result.edge*100:.1f}%)")
        print(f"  Expected Value: ${result.expected_value:.2f}")

        if result.edge > 0.02:
            print(f"  [+] VIABLE BET")
        else:
            print(f"  [-] No bet")


def demo_game_analysis():
    """Demonstrate comprehensive game analysis."""
    print("\n\n=== COMPREHENSIVE GAME ANALYSIS ===")

    # Sample game data (Chiefs @ Bills)
    game_data = {
        "game_id": "2024_W06_KC@BUF",
        "home_team": "BUF",
        "away_team": "KC",
        "ml_home": -130,
        "ml_away": +110,
        "spread": -2.5,
        "spread_juice_home": -110,
        "spread_juice_away": -110,
        "total": 47.5,
        "total_over_juice": -110,
        "total_under_juice": -110
    }

    # Sample model predictions
    model_predictions = {
        "wp_home": 0.58,  # 58% win probability for home
        "predicted_margin": 4.2,  # Home favored by 4.2
        "margin_std": 14.0,
        "predicted_total": 51.3,  # Predict 51.3 total points
        "total_std": 10.5
    }

    print(f"Game: {game_data['away_team']} @ {game_data['home_team']}")
    print(f"Market: {game_data['away_team']} {game_data['ml_away']:+d} / {game_data['home_team']} {game_data['ml_home']:+d}")
    print(f"Spread: {game_data['home_team']} {game_data['spread']:+.1f}")
    print(f"Total: {game_data['total']}")
    print(f"\nModel Predictions:")
    print(f"  Win Probability (Home): {model_predictions['wp_home']:.1%}")
    print(f"  Expected Margin (Home): {model_predictions['predicted_margin']:+.1f}")
    print(f"  Expected Total: {model_predictions['predicted_total']:.1f}")

    # Analyze opportunities
    opportunities = analyze_game_betting_opportunities(
        game_data=game_data,
        model_predictions=model_predictions,
        bankroll=10000.0,
        min_edge_threshold=0.015  # 1.5% minimum edge
    )

    print(f"\n=== BETTING OPPORTUNITIES (Edge >= 1.5%) ===")
    if not opportunities:
        print("No viable betting opportunities found.")
    else:
        for i, opp in enumerate(opportunities, 1):
            bet_type_name = {
                BetType.MONEYLINE: "Moneyline",
                BetType.SPREAD: "Spread",
                BetType.TOTAL: "Total"
            }[opp.bet_type]

            print(f"\n{i}. {bet_type_name} Bet:")
            print(f"   Model Prob: {opp.model_prob:.1%}")
            print(f"   Market Prob: {opp.market_prob:.1%}")
            print(f"   Edge: {opp.edge:.1%}")
            print(f"   Expected Value: ${opp.expected_value:.2f}")
            print(f"   Kelly Size: ${opp.kelly_size:.2f}")
            print(f"   Recommended Units: {opp.recommended_units:.1f}")

    # Summarize the session
    summary = summarize_betting_session(opportunities)
    print(f"\n=== SESSION SUMMARY ===")
    print(f"Total Viable Bets: {summary['total_bets']}")
    print(f"Total Expected Value: ${summary['total_ev']:.2f}")
    print(f"Total Kelly Allocation: ${summary['total_kelly_size']:.2f}")
    print(f"Average Edge: {summary['avg_edge']:.1%}")

    if summary['total_bets'] > 0:
        print(f"\nBreakdown by Bet Type:")
        for bet_type, stats in summary['bet_type_breakdown'].items():
            if stats['count'] > 0:
                print(f"  {bet_type.title()}: {stats['count']} bets, "
                      f"${stats['total_ev']:.2f} EV, "
                      f"{stats['avg_edge']:.1%} avg edge")


def demo_probability_conversions():
    """Demonstrate probability and odds conversions."""
    print("\n\n=== PROBABILITY & ODDS CONVERSIONS ===")

    odds_examples = [-200, -150, -110, +100, +150, +200, +300]

    print("American Odds -> Implied Probability:")
    for odds in odds_examples:
        prob = moneyline_to_probability(odds)
        print(f"  {odds:+4d} -> {prob:.1%}")

    print("\nKey Breakeven Points:")
    print(f"  -110 (standard juice): {moneyline_to_probability(-110):.1%}")
    print(f"  -105 (reduced juice):  {moneyline_to_probability(-105):.1%}")
    print(f"  Pick'em (+100):       {moneyline_to_probability(100):.1%}")


def main():
    """Run all betting utilities demos."""
    print("NFL BETTING UTILITIES DEMONSTRATION")
    print("=" * 50)

    demo_probability_conversions()
    demo_moneyline_calculations()
    demo_spread_calculations()
    demo_total_calculations()
    demo_game_analysis()

    print("\n" + "=" * 50)
    print("Demo completed! Check out scripts/test_betting_utils.py for comprehensive tests.")


if __name__ == "__main__":
    main()