"""Integration: line-movement columns survive the merge into all three gold matrices.

Plan 29-06 / SIG-04. This is the Phase-28 (28-06) anti-silent-drop proof applied to
the line-movement family. ``combine_features`` in ``scripts/build_features.py`` has
NO generic loop over ``feature_sources`` -- it merges each source through an EXPLICIT
per-source block. Registering ``LineMovementBuilder`` into ``feature_sources`` routes
it through the LeakageGate, but its columns are SILENTLY DROPPED from gold unless
``combine_features`` also gets an explicit line-movement merge block. Phase 28 hit
exactly this, which is why a builder unit test alone is not sufficient evidence.

Three layers of proof, in increasing distance from the code:

1. ``TestCombineFeaturesSeam`` -- in-process, with a negative control: the columns
   appear in ``combine_features`` output ONLY when the ``line_movement`` source is
   present. This localizes the proof to the merge block itself.
2. ``TestBuilderSignalBeforeNormalization`` -- raw builder output for a covered
   season carries ``line_movement_coverage == 1.0`` and genuinely non-zero drift.
3. ``TestGoldMatrices`` -- the rebuilt on-disk parquet in ALL THREE matrices.

**Why layer 3 does not assert ``line_movement_coverage == 1.0``.** Gold is
expanding-window Z-SCORED (``scripts/build_features.py`` -> ``expanding_normalize``),
so no gold feature retains its raw value -- a literal ``== 1.0`` assertion on the
gold parquet would be asserting something false about every feature in the project,
not a weaker version of the real check. The raw-value check lives in layer 2, where
raw values actually exist. Layer 3 instead proves REAL SIGNAL landed the only way
that survives normalization: a covered season carries genuine VARIANCE in the
drift/path family, while an uncovered season (before the 2020-06-06 archive floor)
is exactly degenerate. Presence + non-null alone would pass even if every value were
a neutral default, which is the failure this pair of assertions rules out.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data.storage import load_dataframe
from features.line_movement import LineMovementBuilder
from scripts.build_features import FeatureMatrixBuilder

GOLD_DIR = Path(__file__).resolve().parents[2] / "data" / "gold"
MATRICES = ["features_wp", "features_ats", "features_ou"]

# The D-09 totals family + the shared coverage flag (features/line_movement.py).
LINE_MOVEMENT_COLUMNS = (
    "opening_total",
    "total_drift",
    "total_drift_dir",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
    "line_movement_coverage",
)

# Tier (a) spread siblings -- emitted only when odds_timeline carries spreads
# (D-07 keeps totals primary), so they are asserted all-or-none rather than
# unconditionally required.
SPREAD_SIBLING_COLUMNS = (
    "opening_spread",
    "spread_drift",
    "spread_drift_dir",
    "spread_late_drift",
    "spread_abs_travel",
    "spread_reversals",
    "spread_range",
)

# The drift/path metrics that carry the actual movement signal (as opposed to
# the opening LEVEL, which has a non-zero neutral default even when uncovered).
DRIFT_PATH_COLUMNS = (
    "total_drift",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
)

# 2023 sits fully inside the 2020-06-06 historical-odds floor: 271 of 272
# regular-season games have a real multi-snapshot trajectory (Plan 29-05).
COVERED_SEASON = 2023

# 2019 is entirely BEFORE the floor -- zero trajectory rows exist by
# construction, so every drift/path metric is a neutral default.
UNCOVERED_SEASON = 2019


@pytest.fixture(scope="module")
def gold_frames() -> dict[str, pd.DataFrame]:
    """Load the three rebuilt gold matrices (skip the suite if not yet built)."""
    frames: dict[str, pd.DataFrame] = {}
    for name in MATRICES:
        path = GOLD_DIR / f"{name}.parquet"
        if not path.exists():
            pytest.skip(f"{path} not built yet -- run scripts.build_features first")
        frames[name] = pd.read_parquet(path)
    return frames


@pytest.fixture(scope="module")
def covered_season_games() -> pd.DataFrame:
    """Silver games for the covered spot-check season."""
    games = load_dataframe("games", layer="silver")
    return games[games["season"] == COVERED_SEASON]


class TestCombineFeaturesSeam:
    """The 28-06 merge block is what lands the columns -- with a negative control."""

    def _sources(self, games: pd.DataFrame) -> dict[str, pd.DataFrame]:
        line_movement = LineMovementBuilder().build_features(
            games, datetime.now(), target_season=COVERED_SEASON
        )
        return {"games": games, "line_movement": line_movement}

    def test_columns_reach_combined_matrix(
        self, covered_season_games: pd.DataFrame
    ) -> None:
        """combine_features emits the line-movement columns when the source is present."""
        combined = FeatureMatrixBuilder().combine_features(
            self._sources(covered_season_games)
        )
        missing = [c for c in LINE_MOVEMENT_COLUMNS if c not in combined.columns]
        assert not missing, (
            f"combine_features dropped {missing} -- the explicit line-movement merge "
            f"block is missing or wrong (registration alone is NOT enough, 28-06)."
        )

    def test_columns_absent_without_the_source(
        self, covered_season_games: pd.DataFrame
    ) -> None:
        """Negative control: no OTHER source supplies these columns.

        Without this, the presence test above could pass on a coincidence (some
        unrelated builder emitting a same-named column) rather than on the merge
        block actually working.
        """
        combined = FeatureMatrixBuilder().combine_features(
            {"games": covered_season_games}
        )
        leaked = [c for c in LINE_MOVEMENT_COLUMNS if c in combined.columns]
        assert not leaked, (
            f"{leaked} appeared without the line_movement source -- the presence "
            f"test above is not proving what it claims to prove."
        )


class TestBuilderSignalBeforeNormalization:
    """Raw builder values for a covered season (gold is z-scored; see docstring)."""

    @pytest.fixture(scope="class")
    def builder_output(self, covered_season_games: pd.DataFrame) -> pd.DataFrame:
        return LineMovementBuilder().build_features(
            covered_season_games, datetime.now(), target_season=COVERED_SEASON
        )

    def test_covered_regular_season_games_score_full_coverage(
        self, builder_output: pd.DataFrame, covered_season_games: pd.DataFrame
    ) -> None:
        """Regular-season games in a covered season score coverage == 1.0.

        Not 100% by design: a game whose per-game Friday-6PM-ET freeze precedes
        every captured snapshot (the Week-1 Thursday opener, whose freeze is the
        PRIOR Friday) is honestly uncovered rather than back-filled. 271 of 272
        for 2023 (Plan 29-05), so a 95% floor is a real assertion with headroom
        for that structural handful.
        """
        weeks = covered_season_games[["game_id", "week"]]
        merged = builder_output.merge(weeks, on="game_id", how="left")
        regular = merged[merged["week"] <= 18]

        assert len(regular) > 0
        covered_fraction = (regular["line_movement_coverage"] == 1.0).mean()
        assert covered_fraction >= 0.95, (
            f"only {covered_fraction:.1%} of {COVERED_SEASON} regular-season games "
            f"scored line_movement_coverage == 1.0"
        )

    def test_real_movement_is_measured_not_defaulted(
        self, builder_output: pd.DataFrame
    ) -> None:
        """At least one drift/path metric carries a genuinely non-zero value.

        A column can be present, non-null and entirely neutral-default; this is the
        assertion that rules that out on RAW values.
        """
        non_zero = {
            col: int((builder_output[col] != 0.0).sum()) for col in DRIFT_PATH_COLUMNS
        }
        assert any(v > 0 for v in non_zero.values()), (
            f"every drift/path metric is entirely zero for {COVERED_SEASON} -- the "
            f"trajectory produced no movement signal at all: {non_zero}"
        )

    def test_no_emitted_column_names_a_closing_line(
        self, builder_output: pd.DataFrame
    ) -> None:
        """The closing line is reserved for CLV grading and is never a feature (D-15)."""
        closing = [c for c in builder_output.columns if "closing" in c.lower()]
        assert not closing, closing


class TestGoldMatrices:
    """The rebuilt on-disk gold -- the artifact the 29-07 lift screen consumes."""

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_line_movement_columns_present(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """Each gold matrix carries the full totals family + the coverage flag."""
        cols = set(gold_frames[matrix].columns)
        missing = [c for c in LINE_MOVEMENT_COLUMNS if c not in cols]
        assert not missing, (
            f"{matrix} is missing line-movement columns {missing} -- the merge block "
            f"did not deliver them to this matrix. Line-ish columns present: "
            f"{sorted(c for c in cols if 'total' in c or 'line_movement' in c)[:12]}"
        )

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_spread_siblings_are_all_or_none(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """The Tier (a) spread family lands whole or not at all (D-07)."""
        cols = set(gold_frames[matrix].columns)
        present = [c for c in SPREAD_SIBLING_COLUMNS if c in cols]
        assert len(present) in (0, len(SPREAD_SIBLING_COLUMNS)), (
            f"{matrix} carries a PARTIAL spread family {present} -- a partial merge "
            f"is the silent-drop failure mode in a subtler form."
        )

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_no_closing_column_reaches_gold(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """No ``closing_*`` column exists in gold (SC2, carried from D-15).

        The closing line is the CLV grading anchor; a model feature derived from it
        would be post-freeze information.
        """
        closing = [c for c in gold_frames[matrix].columns if "closing" in c.lower()]
        assert not closing, f"{matrix} carries closing-line columns: {closing}"

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_line_movement_columns_are_non_null(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """Every line-movement value is non-null, including pre-2020 rows.

        Uncovered seasons get NEUTRAL DEFAULTS, not NaN -- which is what keeps the
        ``scripts/data_qa.py`` all-null check satisfied on a matrix whose coverage
        starts mid-history.
        """
        df = gold_frames[matrix]
        null_counts = {
            c: int(df[c].isna().sum()) for c in LINE_MOVEMENT_COLUMNS if c in df.columns
        }
        assert all(v == 0 for v in null_counts.values()), (
            f"{matrix} has null line-movement values: "
            f"{ {k: v for k, v in null_counts.items() if v} }"
        )

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_covered_season_carries_real_variance(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """A covered season's drift/path family is non-degenerate in gold.

        Variance is the normalization-proof signature of real movement: an
        all-default column is constant within a season no matter how it is scaled.
        """
        df = gold_frames[matrix]
        season_rows = df[df["season"] == COVERED_SEASON]
        if len(season_rows) == 0:
            pytest.skip(f"{matrix} has no {COVERED_SEASON} rows")

        varying = [
            c
            for c in DRIFT_PATH_COLUMNS
            if c in season_rows.columns and season_rows[c].nunique() > 1
        ]
        assert varying, (
            f"{matrix}: every drift/path column is CONSTANT across {COVERED_SEASON} "
            f"-- the columns reached gold but carry no line-movement signal."
        )

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_uncovered_season_is_degenerate(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """A pre-floor season's drift/path family is exactly constant.

        This is the counterpart that makes the variance test above meaningful: if
        an uncovered season ALSO varied, the "signal" would be an artifact of the
        pipeline rather than measured market movement. The archive genuinely starts
        2020-06-06, so 2019 must be flat.
        """
        df = gold_frames[matrix]
        season_rows = df[df["season"] == UNCOVERED_SEASON]
        if len(season_rows) == 0:
            pytest.skip(f"{matrix} has no {UNCOVERED_SEASON} rows")

        varying = {
            c: int(season_rows[c].nunique())
            for c in DRIFT_PATH_COLUMNS
            if c in season_rows.columns and season_rows[c].nunique() > 1
        }
        assert not varying, (
            f"{matrix}: {UNCOVERED_SEASON} predates the 2020-06-06 archive floor yet "
            f"carries varying drift/path values {varying} -- movement was fabricated "
            f"for a season with no trajectory data."
        )

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_coverage_flag_distinguishes_the_two_regimes(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """``line_movement_coverage`` separates covered from uncovered seasons.

        The flag exists so a MEASURED zero drift and an UNMEASURABLE trajectory are
        not encoded identically. If the flag were constant across both regimes it
        would carry no information and the uncovered rows would masquerade as real
        zero-movement observations.
        """
        df = gold_frames[matrix]
        covered = df[df["season"] == COVERED_SEASON]["line_movement_coverage"]
        uncovered = df[df["season"] == UNCOVERED_SEASON]["line_movement_coverage"]
        if len(covered) == 0 or len(uncovered) == 0:
            pytest.skip(f"{matrix} lacks rows for both coverage regimes")

        assert uncovered.nunique() == 1, (
            f"{matrix}: {UNCOVERED_SEASON} coverage flag is not constant -- every "
            f"pre-floor game is uncovered by construction."
        )
        assert covered.nunique() > 1, (
            f"{matrix}: {COVERED_SEASON} coverage flag is constant -- the covered/"
            f"uncovered distinction did not reach gold."
        )
