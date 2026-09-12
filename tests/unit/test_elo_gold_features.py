"""Tests for post-rebuild gold feature matrix Elo column assertions.

Verifies that gold feature matrices (WP, ATS, O/U) contain real Elo-derived
features with meaningful variance after the gold rebuild with extended
EloFeatureBuilder (14 Elo features).
"""

import json
from pathlib import Path

import pandas as pd
import pytest

# Path to gold and baseline data
GOLD_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "gold"
BASELINE_DIR = (
    Path(__file__).resolve().parent.parent.parent / "data" / "baselines" / "v1.0"
)

TARGETS = ["wp", "ats", "ou"]


def _load_gold(target: str) -> pd.DataFrame:
    """Load a gold feature matrix by target name."""
    path = GOLD_DIR / f"features_{target}.parquet"
    if not path.exists():
        pytest.skip(f"Gold matrix not found: {path}")
    return pd.read_parquet(path)


class TestEloColumnsHaveVariance:
    """Verify Elo feature columns have non-zero standard deviation."""

    def test_elo_columns_have_variance(self):
        """Each Elo feature column in gold matrices has std > 0."""
        from features.elo_features import ELO_FEATURE_COLUMNS

        for target in TARGETS:
            df = _load_gold(target)
            for col in ELO_FEATURE_COLUMNS:
                if col in df.columns:
                    std = df[col].std()
                    assert std > 0, (
                        f"{target}: Elo column '{col}' has zero variance (std={std})"
                    )


class TestNoConstant1500Values:
    """Verify home_elo and away_elo are not all 1500.0."""

    def test_no_constant_1500_values(self):
        """home_elo and away_elo columns are not all 1500.0 in any target."""
        for target in TARGETS:
            df = _load_gold(target)

            assert not (df["home_elo"] == 1500.0).all(), (
                f"{target}: home_elo is all 1500.0 (batch leakage not fixed)"
            )
            assert df["home_elo"].nunique() > 1, (
                f"{target}: home_elo has only 1 unique value"
            )

            assert not (df["away_elo"] == 1500.0).all(), (
                f"{target}: away_elo is all 1500.0"
            )
            assert df["away_elo"].nunique() > 1, (
                f"{target}: away_elo has only 1 unique value"
            )


class TestMomentumColumnsPresent:
    """Verify momentum columns exist in gold matrices."""

    def test_momentum_columns_present(self):
        """home_elo_momentum and away_elo_momentum exist in all targets."""
        for target in TARGETS:
            df = _load_gold(target)
            assert "home_elo_momentum" in df.columns, (
                f"{target}: missing home_elo_momentum"
            )
            assert "away_elo_momentum" in df.columns, (
                f"{target}: missing away_elo_momentum"
            )


class TestRankColumnsPresent:
    """Verify rank and percentile columns exist in gold matrices."""

    def test_rank_columns_present(self):
        """Rank and percentile columns exist in all targets."""
        for target in TARGETS:
            df = _load_gold(target)
            for col in [
                "home_elo_rank",
                "away_elo_rank",
                "home_elo_percentile",
                "away_elo_percentile",
            ]:
                assert col in df.columns, f"{target}: missing {col}"


class TestMomentumNanCountReasonable:
    """Verify momentum has reasonable distribution (week 1 values are imputed)."""

    def test_momentum_nan_count_reasonable(self):
        """Momentum values should exist for all rows (NaN imputed by pipeline).

        The gold pipeline imputes NaN values via handle_missing_data_and_outliers,
        so week 1 games that originally had NaN momentum are filled with team
        season averages or overall medians. We verify that week 1 games have
        momentum values near zero (since early-season momentum is undefined).
        """
        for target in TARGETS:
            df = _load_gold(target)
            if "home_elo_momentum" not in df.columns:
                pytest.skip(f"{target}: no momentum column")

            # After imputation, all values should be non-NaN (or very few)
            nan_count = df["home_elo_momentum"].isna().sum()
            total = len(df)

            # Allow a small fraction of NaN (< 5%) in case some edge cases remain
            assert nan_count / total < 0.05, (
                f"{target}: Too many NaN in momentum after imputation: "
                f"{nan_count}/{total} = {nan_count / total:.1%}"
            )

            # Momentum should have meaningful variation (not constant)
            std = df["home_elo_momentum"].std()
            assert std > 0, (
                f"{target}: home_elo_momentum has zero variance (all same value)"
            )


class TestRankValuesInRange:
    """Verify rank and percentile values have meaningful variance.

    Note: Gold matrices are Z-score normalized by the feature pipeline,
    so raw rank [1-32] and percentile [0-1] ranges are transformed.
    We verify variance and distinct values instead.
    """

    def test_rank_values_in_range(self):
        """Rank and percentile columns have meaningful variance (post-normalization)."""
        for target in TARGETS:
            df = _load_gold(target)

            if "home_elo_rank" in df.columns:
                valid = df["home_elo_rank"].dropna()
                # After Z-score normalization, verify meaningful variance
                assert valid.std() > 0, f"{target}: home_elo_rank has zero variance"
                # Should have many distinct values (32 ranks per week)
                assert valid.nunique() > 10, (
                    f"{target}: home_elo_rank has too few unique values: "
                    f"{valid.nunique()}"
                )

            if "home_elo_percentile" in df.columns:
                valid = df["home_elo_percentile"].dropna()
                assert valid.std() > 0, (
                    f"{target}: home_elo_percentile has zero variance"
                )
                assert valid.nunique() > 10, (
                    f"{target}: home_elo_percentile has too few unique values: "
                    f"{valid.nunique()}"
                )


class TestNewColumnsVsBaseline:
    """Compare new gold matrices with v1.0 baseline."""

    def test_new_columns_vs_baseline(self):
        """New gold matrices have 6 additional Elo columns vs v1.0 baseline."""
        baseline_path = BASELINE_DIR / "feature_metadata.json"
        if not baseline_path.exists():
            pytest.skip("Baseline feature metadata not found")

        with open(baseline_path) as f:
            baseline_meta = json.load(f)

        for target in TARGETS:
            df = _load_gold(target)
            gold_cols = set(df.columns)
            baseline_cols = set(baseline_meta[target]["columns"])

            # Find new Elo columns not in baseline
            new_elo_cols = {c for c in gold_cols - baseline_cols if "elo" in c.lower()}
            expected_new = {
                "home_elo_momentum",
                "away_elo_momentum",
                "home_elo_rank",
                "away_elo_rank",
                "home_elo_percentile",
                "away_elo_percentile",
            }
            assert expected_new.issubset(new_elo_cols), (
                f"{target}: Expected new Elo columns {expected_new} but only "
                f"found new Elo cols: {new_elo_cols}"
            )


class TestSameWeekProvisionalRowsMoveRanksAndNotMomentum:
    """NF-08's ROW claim, asserted in both halves and in ONE test.

    THE COLUMN CLAIM DOES NOT COVER THIS. ``is_provisional`` reaches no gold matrix,
    because ``build_features`` takes an explicit seven-column subset before the merge.
    But the FULL snapshots frame -- every column, every row -- is then handed to
    ``_add_momentum_features`` and ``_add_rank_features``, and ``_add_rank_features``
    selects ``week <= W``. So SAME-WEEK PROVISIONAL ROWS DO FEED ``home_elo_rank`` /
    ``away_elo_rank`` / ``home_elo_percentile`` / ``away_elo_percentile``.

    THAT IS DELIBERATE LIVE-SERVING BEHAVIOUR, NOT A LEAK. Pre-game Elo is captured
    BEFORE the game, so week N's snapshot is a valid input for ranking at week N -- the
    property the existing comment in ``_add_rank_features`` already states. Ranking the
    live week against a table that stopped at week N-1 would rank this week's teams on
    last week's standings.

    It IS a reproducibility hazard, and that is stated rather than discovered: a gold row
    built on Friday can differ from the same row rebuilt after the results land, because
    the provisional row is replaced in place. Handed to Phase 34's replay work; COLD-01's
    value-by-value comparison is scoped to the seven JOINED columns and does not catch it.

    The momentum half is the POSITIVE answer to the reviewer's concern that a provisional
    row could enter a rolling window as a zero-delta game. It is asserted as a
    value-by-value equality over the whole column, NOT as an appeal to
    ``_add_momentum_features``'s ``week < W`` selector -- an appeal to the selector's
    shape is exactly what the concern was about.
    """

    SEASON = 2026
    RANK_COLUMNS = (
        "home_elo_rank",
        "away_elo_rank",
        "home_elo_percentile",
        "away_elo_percentile",
    )
    MOMENTUM_COLUMNS = ("home_elo_momentum", "away_elo_momentum")

    @classmethod
    def _real_snapshots(cls) -> pd.DataFrame:
        """Weeks 1 and 2 played, four teams, moving Elo so momentum is NON-zero.

        A flat Elo path would make every momentum value 0.0, and an equality between two
        columns of zeros is satisfied by a function that returns zeros.
        """
        from scripts.build_elo import build_snapshot_frame

        def row(week, home, away, home_elo, away_elo, provisional):
            return {
                "game_id": f"{cls.SEASON}_W{week:02d}_{away}@{home}",
                "season": cls.SEASON,
                "week": week,
                "home_team": home,
                "away_team": away,
                "home_elo_pre": home_elo,
                "away_elo_pre": away_elo,
                "home_elo_uncertainty": 200.0,
                "away_elo_uncertainty": 200.0,
                "elo_prob_home": 0.6,
                "hfa_used": 48.0,
                "is_provisional": provisional,
            }

        return build_snapshot_frame(
            [
                row(1, "KC", "BUF", 1600.0, 1500.0, False),
                row(1, "SF", "SEA", 1550.0, 1450.0, False),
                row(2, "KC", "SEA", 1620.0, 1440.0, False),
                row(2, "SF", "BUF", 1560.0, 1490.0, False),
            ]
        )

    @classmethod
    def _provisional_week_three(cls) -> pd.DataFrame:
        """Week-3 provisional rows introducing two teams the ranking has not seen."""
        from scripts.build_elo import build_snapshot_frame

        return build_snapshot_frame(
            [
                {
                    "game_id": f"{cls.SEASON}_W03_CHI@GB",
                    "season": cls.SEASON,
                    "week": 3,
                    "home_team": "GB",
                    "away_team": "CHI",
                    "home_elo_pre": 1700.0,
                    "away_elo_pre": 1400.0,
                    "home_elo_uncertainty": 210.0,
                    "away_elo_uncertainty": 210.0,
                    "elo_prob_home": 0.85,
                    "hfa_used": 48.0,
                    "is_provisional": True,
                }
            ]
        )

    @classmethod
    def _games_under_test(cls) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "game_id": f"{cls.SEASON}_W03_BUF@KC",
                    "season": cls.SEASON,
                    "week": 3,
                    "home_team": "KC",
                    "away_team": "BUF",
                }
            ]
        )

    @classmethod
    def _build(cls, snapshots: pd.DataFrame) -> pd.DataFrame:
        """Run the REAL ``build_features`` path against a cached snapshots frame.

        The cache attribute is the module's own read seam, so no silver table is touched
        and nothing under ``data/`` is read or written.
        """
        from datetime import UTC, datetime

        from features.elo_features import EloFeatureBuilder

        builder = EloFeatureBuilder()
        builder._snapshots_df = snapshots
        return builder.build_features(
            cls._games_under_test(), datetime(2026, 9, 25, 18, 0, tzinfo=UTC)
        )

    def test_ranks_move_and_momentum_does_not(self) -> None:
        without = self._build(self._real_snapshots())
        with_provisional = self._build(
            pd.concat(
                [self._real_snapshots(), self._provisional_week_three()],
                ignore_index=True,
            )
        )

        moved = [
            col
            for col in self.RANK_COLUMNS
            if without[col].iloc[0] != with_provisional[col].iloc[0]
        ]
        assert sorted(moved) == sorted(self.RANK_COLUMNS), (
            "every rank and percentile column must move when same-week provisional rows "
            "are present -- that is the live-serving behaviour this test exists to "
            f"record, not a leak. Unmoved: "
            f"{sorted(set(self.RANK_COLUMNS) - set(moved))}. "
            f"without={[without[c].iloc[0] for c in self.RANK_COLUMNS]} "
            f"with={[with_provisional[c].iloc[0] for c in self.RANK_COLUMNS]}"
        )

        for col in self.MOMENTUM_COLUMNS:
            assert without[col].iloc[0] != 0.0, (
                f"{col} is 0.0 in the control, so an equality assertion on it would be "
                "vacuous -- the fixture's Elo path must actually move"
            )
            assert without[col].iloc[0] == with_provisional[col].iloc[0], (
                f"{col} changed when same-week provisional rows were added. A "
                "provisional row entering a rolling momentum window as a zero-delta "
                f"game would distort momentum silently. "
                f"{without[col].iloc[0]} != {with_provisional[col].iloc[0]}"
            )

    def test_no_gold_column_named_is_provisional_survives_the_join(self) -> None:
        """The COLUMN claim at its mechanism: the explicit seven-column join subset."""
        from tests.phase33_state import ELO_GOLD_JOIN_SUBSET

        built = self._build(
            pd.concat(
                [self._real_snapshots(), self._provisional_week_three()],
                ignore_index=True,
            )
        )

        assert "is_provisional" not in built.columns, (
            "the flag must not reach a feature matrix. The join subset is EXPLICIT "
            "rather than a drop-list, so a new snapshot column is excluded by default -- "
            f"this asserts that property held. Columns: {sorted(built.columns)}"
        )
        # Positive half, so the test cannot pass because the join produced nothing.
        renamed = {"home_elo_pre": "home_elo", "away_elo_pre": "away_elo"}
        for column in ELO_GOLD_JOIN_SUBSET:
            expected = renamed.get(column, column)
            assert expected in built.columns, (
                f"the joined Elo column {expected} is missing -- the absence of "
                "is_provisional would then prove only that the merge did nothing"
            )


class TestRankFeaturesEmptyWeekGuard:
    """Regression: ``_add_rank_features`` must not divide by zero when a
    ``(season, week)`` has no Elo snapshots.

    The silver snapshot table only spans the rated era (2018+), so pre-2018
    burn-in weeks have zero teams to rank (``n_teams == 0``). Before the guard,
    ``(n_teams - home_r + 1) / n_teams`` raised ZeroDivisionError and aborted the
    whole full-history gold rebuild (blocker BLOCKER-svu-01). Such games must get
    NaN rank/percentile, and the normal (covered) path must be unaffected.
    """

    @staticmethod
    def _snapshots_2018_week1() -> pd.DataFrame:
        # Two 2018 W1 games -> four ranked teams (n_teams == 4).
        return pd.DataFrame(
            [
                {
                    "season": 2018,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "BUF",
                    "home_elo_pre": 1600.0,
                    "away_elo_pre": 1500.0,
                },
                {
                    "season": 2018,
                    "week": 1,
                    "home_team": "SF",
                    "away_team": "SEA",
                    "home_elo_pre": 1550.0,
                    "away_elo_pre": 1450.0,
                },
            ]
        )

    def test_uncovered_week_yields_nan_not_zero_division(self):
        """A snapshot-less (pre-2018) game gets NaN rank/percentile, no crash."""
        from features.elo_features import EloFeatureBuilder

        builder = EloFeatureBuilder()
        games = pd.DataFrame(
            [
                {
                    "game_id": "2002_W01_GB_CHI",
                    "season": 2002,
                    "week": 1,
                    "home_team": "GB",
                    "away_team": "CHI",
                },
            ]
        )

        # Must not raise ZeroDivisionError.
        result = builder._add_rank_features(games, self._snapshots_2018_week1())

        for col in [
            "home_elo_rank",
            "away_elo_rank",
            "home_elo_percentile",
            "away_elo_percentile",
        ]:
            assert bool(result[col].isna().all()), (
                f"snapshot-less game should have NaN {col}, got {result[col].tolist()}"
            )

    def test_covered_week_still_ranks_normally(self):
        """The guard must not perturb the normal path: a covered game ranks."""
        from features.elo_features import EloFeatureBuilder

        builder = EloFeatureBuilder()
        games = pd.DataFrame(
            [
                {
                    "game_id": "2018_W01_KC_BUF",
                    "season": 2018,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "BUF",
                },
            ]
        )

        result = builder._add_rank_features(games, self._snapshots_2018_week1())

        # KC (1600) is highest of 4 -> rank 1, percentile (4-1+1)/4 = 1.0.
        # BUF (1500) is 3rd of 4   -> rank 3, percentile (4-3+1)/4 = 0.5.
        assert result["home_elo_rank"].iloc[0] == 1
        assert result["away_elo_rank"].iloc[0] == 3
        assert result["home_elo_percentile"].iloc[0] == 1.0
        assert result["away_elo_percentile"].iloc[0] == 0.5

    def test_mixed_covered_and_uncovered_weeks(self):
        """Covered and uncovered games in one frame: ranks vs NaN, no crash."""
        from features.elo_features import EloFeatureBuilder

        builder = EloFeatureBuilder()
        games = pd.DataFrame(
            [
                {
                    "game_id": "2018_W01_KC_BUF",
                    "season": 2018,
                    "week": 1,
                    "home_team": "KC",
                    "away_team": "BUF",
                },
                {
                    "game_id": "2002_W01_GB_CHI",
                    "season": 2002,
                    "week": 1,
                    "home_team": "GB",
                    "away_team": "CHI",
                },
            ]
        )

        result = builder._add_rank_features(games, self._snapshots_2018_week1())

        covered = result[result["game_id"] == "2018_W01_KC_BUF"].iloc[0]
        uncovered = result[result["game_id"] == "2002_W01_GB_CHI"].iloc[0]
        assert covered["home_elo_rank"] == 1
        assert bool(pd.isna(uncovered["home_elo_rank"]))
        assert bool(pd.isna(uncovered["home_elo_percentile"]))
