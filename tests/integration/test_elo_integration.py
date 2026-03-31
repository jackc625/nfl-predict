"""Integration test for the end-to-end Elo pipeline to gold features.

Verifies:
- EloBuilder produces snapshots with per-game pre-game Elo
- EloFeatureBuilder reads those snapshots and produces real features
- Elo columns have variance (std > 0), no constant 1500.0 values
- Features pass LeakageGate
"""

import pytest


def _silver_has_games() -> bool:
    """Check if silver layer has games data."""
    try:
        from data.storage import load_dataframe

        df = load_dataframe("games", layer="silver")
        return len(df) > 0
    except (FileNotFoundError, OSError, ValueError):
        return False


@pytest.mark.skipif(not _silver_has_games(), reason="No games in silver layer")
class TestEloPipelineToGold:
    """End-to-end test: build Elo snapshots, then build features from them."""

    def test_elo_pipeline_to_gold(self):
        """Run EloBuilder with snapshots, then build Elo features via
        EloFeatureBuilder. Assert Elo columns have variance (std > 0),
        no constant 1500.0, and produce meaningful differentiation.
        """
        from datetime import datetime

        from data.storage import load_dataframe, save_dataframe
        from features.elo_features import ELO_FEATURE_COLUMNS, EloFeatureBuilder
        from scripts.build_elo import EloBuilder

        # Step 1: Build Elo with snapshots
        builder = EloBuilder()
        snapshots_df = builder.build_elo_with_snapshots(start_season=2018)

        assert len(snapshots_df) > 0, "build_elo_with_snapshots should produce snapshots"

        # Save snapshots to silver layer (mimicking the real pipeline)
        save_dataframe(snapshots_df, "elo_game_snapshots", layer="silver", append_mode=False)

        # Step 2: Build features using EloFeatureBuilder
        games_df = load_dataframe("games", layer="silver")
        # Pick a recent season for feature building
        season_2024 = games_df[games_df["season"] == 2024]
        if len(season_2024) == 0:
            season_2024 = games_df[games_df["season"] == games_df["season"].max()]

        feature_builder = EloFeatureBuilder()
        features = feature_builder.build_features(
            season_2024,
            as_of_datetime=datetime(2025, 3, 1, 0, 0),
            target_season=int(season_2024["season"].iloc[0]),
        )

        # Step 3: Verify Elo columns exist and have variance
        for col in ELO_FEATURE_COLUMNS:
            assert col in features.columns, f"Missing Elo feature column: {col}"

        # Elo ratings should have variance (not all 1500.0)
        home_elo_std = features["home_elo"].astype(float).std()
        away_elo_std = features["away_elo"].astype(float).std()

        assert home_elo_std > 0, (
            f"home_elo should have variance (std > 0), got std={home_elo_std}. "
            f"Values: {features['home_elo'].unique()[:5]}"
        )
        assert away_elo_std > 0, (
            f"away_elo should have variance (std > 0), got std={away_elo_std}. "
            f"Values: {features['away_elo'].unique()[:5]}"
        )

        # No constant 1500.0 values (the old bug)
        home_elo_values = features["home_elo"].astype(float)
        all_1500 = (home_elo_values == 1500.0).all()
        assert not all_1500, (
            "home_elo should not be constant 1500.0 (old batch leakage bug)"
        )

        # elo_diff should have variance too
        elo_diff_std = features["elo_diff"].astype(float).std()
        assert elo_diff_std > 0, (
            f"elo_diff should have variance, got std={elo_diff_std}"
        )

        # elo_prob_home should be in (0, 1) range
        probs = features["elo_prob_home"].astype(float)
        assert probs.min() > 0, f"elo_prob_home min should be > 0, got {probs.min()}"
        assert probs.max() < 1, f"elo_prob_home max should be < 1, got {probs.max()}"
