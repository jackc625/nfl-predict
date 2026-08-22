"""Integration: the line-movement family has LEFT gold, via both merge seams.

Plan 30-07 / SPEC R3 / D29-07-01, inverting the Plan 29-06 module of the same name.
Phase 29 landed fifteen line-movement columns in all three gold matrices and this
file proved they arrived; Phase 30 rung 3 removes them as a deliberate decision
about what belongs in gold, and this file now proves they are gone.

WHY THE FILE IS INVERTED RATHER THAN DELETED. Three of its assertions go
legitimately RED after the drop and two become VACUOUS PASSES -- iterations over a
column list that is now empty, where ``all(...)`` over nothing is trivially true.
A green test whose subject vanished is worse than a red one, because nobody
revisits it. Both are rewritten below into assertions about the absence.

Three layers of proof, in increasing distance from the code:

1. ``TestCombineFeaturesSeam`` -- in-process, the 28-06 lesson run in REVERSE.
   ``combine_features`` in ``scripts/build_features.py`` has NO generic loop over
   ``feature_sources``; it merges each source through an EXPLICIT per-source block.
   That is what made the family need TWO seams to reach gold (registration AND a
   merge block), and it is why removing one and leaving the other would look fixed
   and resurrect the columns on the next rebuild. Handing ``combine_features`` a
   POPULATED line-movement source and getting none of its columns back is the
   strongest available proof that the merge seam is genuinely gone rather than
   merely unfed -- and the source frame is asserted populated FIRST, or "none
   arrived" would be trivially true.
2. ``TestBuilderSignalBeforeNormalization`` -- UNCHANGED. What left gold is the
   columns, not the builder and not the paid ``odds_timeline`` archive behind it.
   ``features/line_movement.py`` still works and the archive still carries real
   trajectory signal for a covered season; that remains true and remains asserted,
   because a phase that quietly broke the builder while claiming to have dropped
   its output would be indistinguishable from this one otherwise.
3. ``TestLineMovementDropped`` -- the rebuilt on-disk parquet in ALL THREE matrices.

**On ``total_points``.** It is the O/U LABEL column and lives only in
``features_ou`` (``backtest/diagnose.py``'s label map). It is asserted PRESENT in
``features_ou`` AND ABSENT from ``features_wp`` and ``features_ats`` -- both
directions, deliberately. A bare loop over all three would fail on WP and ATS for a
reason with nothing to do with the drop; a presence check on ``features_ou`` alone
would not catch an over-eager suffix or regex prune that took the label along with
the family it resembles.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from backtest.signal_lift import group_columns
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

# Tier (a) spread siblings -- the other half of the fifteen.
SPREAD_SIBLING_COLUMNS = (
    "opening_spread",
    "spread_drift",
    "spread_drift_dir",
    "spread_late_drift",
    "spread_abs_travel",
    "spread_reversals",
    "spread_range",
)

# The whole removed family, as the drop and the rung-3 attribution see it.
FULL_FAMILY = LINE_MOVEMENT_COLUMNS + SPREAD_SIBLING_COLUMNS

# The drift/path metrics that carry the actual movement signal (as opposed to
# the opening LEVEL, which has a non-zero neutral default even when uncovered).
DRIFT_PATH_COLUMNS = (
    "total_drift",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
)

# Market columns that must SURVIVE. The freeze anchors (snapshot_*) and the
# pre-existing, historically identically-0.0 MarketAnchor movement columns are all
# baseline features and all near-misses for a careless substring prune.
MARKET_SURVIVORS = (
    "snapshot_total",
    "snapshot_spread",
    "total_movement",
    "spread_movement",
)

# The O/U label column, and the ONLY matrix it has ever been in.
OU_LABEL_COLUMN = "total_points"
OU_LABEL_MATRIX = "features_ou"

# 2023 sits fully inside the 2020-06-06 historical-odds floor: 271 of 272
# regular-season games have a real multi-snapshot trajectory (Plan 29-05).
COVERED_SEASON = 2023


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
    """The 28-06 merge block is GONE -- proved by feeding it, not by inspecting it."""

    def _sources(self, games: pd.DataFrame) -> dict[str, pd.DataFrame]:
        line_movement = LineMovementBuilder().build_features(
            games, datetime.now(), target_season=COVERED_SEASON
        )
        return {"games": games, "line_movement": line_movement}

    def test_a_populated_source_lands_none_of_its_columns(
        self, covered_season_games: pd.DataFrame
    ) -> None:
        """Hand ``combine_features`` the real builder output; nothing must arrive.

        The source frame is asserted POPULATED first. Without that pin this test
        would pass just as happily against a builder that silently produced an
        empty frame, which is the vacuous-pass shape this rewrite exists to
        eliminate.
        """
        sources = self._sources(covered_season_games)
        offered = group_columns(sources["line_movement"], "line_movement")
        assert len(sources["line_movement"]) > 0, (
            "fixture sanity: the builder produced no rows, so 'nothing arrived' "
            "would be trivially true"
        )
        assert len(offered) == len(FULL_FAMILY), (
            "fixture sanity: the source frame must CARRY the whole family before "
            f"this test can say the merge dropped it. Offered: {offered}"
        )

        combined = FeatureMatrixBuilder().combine_features(sources)

        arrived = group_columns(combined, "line_movement")
        assert arrived == [], (
            f"combine_features merged {arrived} from a registered line_movement "
            f"source -- the explicit merge block has returned, and the family will "
            f"be back in gold on the next rebuild (SPEC R3). Registration alone "
            f"never landed these columns; the merge block did."
        )

    def test_the_source_is_not_registered_in_the_first_place(self) -> None:
        """The other seam. Both are removed, so neither can land the family alone."""
        import inspect

        source = inspect.getsource(FeatureMatrixBuilder.load_all_feature_sources)
        assert 'feature_sources["line_movement"]' not in source, (
            "load_all_feature_sources registers a line_movement source again. On "
            "its own that does not reach gold -- it routes the frame through the "
            "LeakageGate -- but it is half of the seam pair."
        )


class TestBuilderSignalBeforeNormalization:
    """UNCHANGED. The builder and the paid archive survive the drop intact."""

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
        assertion that rules that out on RAW values. It also keeps the drop honest:
        what left gold is a family that MEASURED something, not a family that was
        already inert.
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


class TestLineMovementDropped:
    """The rebuilt on-disk gold -- the artifact the Phase-30 gate consumes."""

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_no_line_movement_column_remains(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """Not one of the fifteen survives, in any matrix.

        Derived from the group registry as well as from the named tuple, so a
        column the predicate matches but the tuple forgot is still caught.
        """
        df = gold_frames[matrix]
        named_survivors = [c for c in FULL_FAMILY if c in df.columns]
        assert not named_survivors, (
            f"{matrix} still carries {named_survivors}. A PARTIAL drop is worse "
            f"than none: the three matrices then disagree about the candidate "
            f"feature set and the rung-3 attribution cannot balance."
        )
        assert group_columns(df, "line_movement") == [], (
            f"{matrix} carries a column the line_movement predicate matches but "
            f"the named tuple above does not list"
        )

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_the_market_survivors_are_present(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """The freeze anchors and the MarketAnchor movement columns must remain.

        ``snapshot_total`` ends in ``total``; ``spread_movement`` contains
        ``spread``. A substring prune rather than an exact-suffix one would take
        all four with the family and silently delete the market leg's anchors.
        """
        cols = set(gold_frames[matrix].columns)
        missing = [c for c in MARKET_SURVIVORS if c not in cols]
        assert not missing, (
            f"{matrix} lost market columns {missing} to the drop. These are "
            f"pre-existing baseline features, not Phase-29 line-movement columns."
        )

    def test_the_ou_label_survives_in_its_own_matrix(
        self, gold_frames: dict[str, pd.DataFrame]
    ) -> None:
        """``total_points`` is the O/U LABEL and must survive in ``features_ou``."""
        assert OU_LABEL_COLUMN in gold_frames[OU_LABEL_MATRIX].columns, (
            f"{OU_LABEL_MATRIX} lost {OU_LABEL_COLUMN} -- that is the O/U label "
            f"column, and an over-eager prune that matched on 'total' would take "
            f"it along with the family"
        )

    @pytest.mark.parametrize("matrix", ["features_wp", "features_ats"])
    def test_the_ou_label_is_absent_from_the_other_two(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """The other direction, and the half that makes the pair meaningful.

        ``total_points`` has only ever been in ``features_ou`` -- it is that
        matrix's label. Asserting only its PRESENCE there would pass while it
        quietly vanished from nowhere; asserting presence across all three would
        fail on WP and ATS for a reason with nothing to do with the drop. Both
        directions, together, say what is actually meant: this column belongs to
        exactly one matrix and the drop did not move it.
        """
        assert OU_LABEL_COLUMN not in gold_frames[matrix].columns, (
            f"{matrix} carries {OU_LABEL_COLUMN}, which is the O/U label column "
            f"and has never belonged to this matrix"
        )

    @pytest.mark.parametrize("matrix", MATRICES)
    def test_no_closing_column_reaches_gold(
        self, gold_frames: dict[str, pd.DataFrame], matrix: str
    ) -> None:
        """No ``closing_*`` column exists in gold (SC2, carried from D-15).

        The closing line is the CLV grading anchor; a model feature derived from it
        would be post-freeze information. Unchanged by the drop, and still
        meaningful: it is a statement about every column in gold, not about the
        fifteen that left.
        """
        closing = [c for c in gold_frames[matrix].columns if "closing" in c.lower()]
        assert not closing, f"{matrix} carries closing-line columns: {closing}"
