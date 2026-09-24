"""Unit coverage for the market-anchor consensus arithmetic (33.2 review B WR-12).

The live gold builder (``MarketAnchorFeaturesCalculator.build_features``) compresses each
game's admitted lines through ``_compress_game_lines``; its fair home-win probability is the
median of each book's DEVIGGED probability, never of the raw American odds.

RETIRED (33.2 review batch 3): ``TestCreateConsensusLinesPrefixAware`` and the
``create_consensus_lines`` half of the WR-12 class. ``create_consensus_lines`` served only the
deprecated ``build_market_anchor_features`` path, which wrote a silver table no production code
read, and was deleted with the step that ran it.
"""

from __future__ import annotations

import pandas as pd
import pytest

from features.market_anchors import MarketAnchorFeaturesCalculator


class TestConsensusAcrossThePlusMinus100Discontinuity:
    """33.2 review B WR-12: the consensus is a median of PROBABILITIES, never of raw odds.

    American odds jump from -100 to +100 with nothing between, so a pick'em quoted -105 at one
    book and +100 at another had a raw median of -2 -- not a price -- and
    ``moneyline_to_probability(-2)`` returned 0.0196.
    """

    def test_the_compressed_fair_probability_is_the_median_of_each_books_devig(self):
        from utils.probability_utils import (
            devig_probabilities,
            moneyline_to_probability,
        )

        calc = MarketAnchorFeaturesCalculator()
        odds = pd.DataFrame(
            {
                "sportsbook": ["DraftKings", "FanDuel"],
                "information_time": pd.to_datetime(
                    ["2024-10-05T20:00:00Z", "2024-10-05T21:00:00Z"]
                ),
                "ml_home": [-105, 100],
                "ml_away": [-115, -120],
                "spread": [0.0, 0.0],
                "total": [44.0, 44.0],
            }
        )
        compressed = calc._compress_game_lines("2024_W05_KC@BUF", odds)

        per_book = [
            devig_probabilities(
                moneyline_to_probability(h),
                moneyline_to_probability(a),
                method=calc.devig_method,
            )[0]
            for h, a in ((-105, -115), (100, -120))
        ]
        expected = sorted(per_book)[0] + (sorted(per_book)[1] - sorted(per_book)[0]) / 2
        assert compressed["snapshot_ml_prob_home_fair"] == pytest.approx(expected)
        assert 0.45 < compressed["snapshot_ml_prob_home_fair"] < 0.55

    @pytest.mark.parametrize("not_a_price", [-2, 0, 50, -99])
    def test_a_value_between_minus_and_plus_100_is_refused(self, not_a_price: int):
        from utils.probability_utils import moneyline_to_probability

        with pytest.raises(ValueError, match="not an American moneyline"):
            moneyline_to_probability(not_a_price)
        # The controls: both edges of the discontinuity are real prices.
        assert moneyline_to_probability(-100) == pytest.approx(0.5)
        assert moneyline_to_probability(100) == pytest.approx(0.5)
