"""Unit coverage for the LIVE + CRITICAL market-anchor consensus path (WR-02).

`MarketAnchorFeaturesCalculator.create_consensus_lines` is reached on every Friday
run by the critical PREDICTIONS step 10 (`step_build_market_anchors` ->
`build_market_anchor_features` -> `create_consensus_lines`), once for the opening-line
frame and once for the snapshot-line frame. Those upstream frames prefix every odds
column with `opening_` / `snapshot_` (e.g. `opening_ml_home`, `snapshot_spread`); the
pre-D-11-E code read the bare `ml_home` / `spread` names that no longer exist, raising
`KeyError 'ml_home'` and aborting the Friday run with a CRITICAL alert.

These tests guard the prefix-aware fix so a future rename of the `opening_` / `snapshot_`
column prefixes fails here instead of re-breaking the live path silently.
"""

from __future__ import annotations

import pandas as pd

from features.market_anchors import MarketAnchorFeaturesCalculator


def _make_lines_frame(prefix: str) -> pd.DataFrame:
    """Build a consensus-input frame as identify_{opening,snapshot}_lines emits it.

    Every odds column carries the given prefix (``opening_`` or ``snapshot_``).
    Two sportsbooks per game so the median and the >1-book range columns are exercised.
    """
    return pd.DataFrame(
        [
            {
                "game_id": "2024_W05_KC@BUF",
                "sportsbook": "DraftKings",
                f"{prefix}ml_home": -150,
                f"{prefix}ml_away": 130,
                f"{prefix}spread": -3.0,
                f"{prefix}total": 44.0,
            },
            {
                "game_id": "2024_W05_KC@BUF",
                "sportsbook": "FanDuel",
                f"{prefix}ml_home": -160,
                f"{prefix}ml_away": 140,
                f"{prefix}spread": -3.5,
                f"{prefix}total": 45.0,
            },
        ]
    )


class TestCreateConsensusLinesPrefixAware:
    """create_consensus_lines reads the prefixed columns that actually exist (WR-02)."""

    def test_opening_prefixed_frame_produces_consensus_no_keyerror(self):
        """An ``opening_``-prefixed frame yields consensus columns with no KeyError.

        This is the exact shape `build_market_anchor_features` passes for the opening
        lines (`features/market_anchors.py:649`).
        """
        calc = MarketAnchorFeaturesCalculator()
        lines_df = _make_lines_frame("opening_")

        # Must NOT raise KeyError 'ml_home' (the pre-fix failure on the live path).
        result = calc.create_consensus_lines(lines_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["game_id"] == "2024_W05_KC@BUF"
        # Medians of the two books.
        assert row["consensus_ml_home"] == -155.0  # median(-150, -160)
        assert row["consensus_ml_away"] == 135.0  # median(130, 140)
        assert row["consensus_spread"] == -3.25  # median(-3.0, -3.5)
        assert row["consensus_total"] == 44.5  # median(44.0, 45.0)
        assert row["num_sportsbooks"] == 2
        # >1 book -> disagreement ranges present.
        assert row["spread_range"] == 0.5
        assert row["total_range"] == 1.0

    def test_snapshot_prefixed_frame_produces_consensus_no_keyerror(self):
        """A ``snapshot_``-prefixed frame yields consensus columns with no KeyError.

        This is the second live caller (`features/market_anchors.py:654`); the literal
        "read opening_ only" fix would still KeyError here, which is why the fix is
        prefix-aware.
        """
        calc = MarketAnchorFeaturesCalculator()
        lines_df = _make_lines_frame("snapshot_")

        result = calc.create_consensus_lines(lines_df)

        assert len(result) == 1
        row = result.iloc[0]
        assert row["game_id"] == "2024_W05_KC@BUF"
        assert row["consensus_ml_home"] == -155.0
        assert row["consensus_ml_away"] == 135.0
        assert row["consensus_spread"] == -3.25
        assert row["consensus_total"] == 44.5
        assert row["num_sportsbooks"] == 2

    def test_consensus_columns_present_for_both_prefixes(self):
        """Both prefix shapes produce the same consensus output schema (no KeyError)."""
        calc = MarketAnchorFeaturesCalculator()
        expected_cols = {
            "game_id",
            "consensus_ml_home",
            "consensus_ml_away",
            "consensus_spread",
            "consensus_total",
            "num_sportsbooks",
        }

        for prefix in ("opening_", "snapshot_"):
            result = calc.create_consensus_lines(_make_lines_frame(prefix))
            assert expected_cols.issubset(result.columns), (
                f"missing consensus columns for prefix {prefix!r}: "
                f"{expected_cols - set(result.columns)}"
            )
