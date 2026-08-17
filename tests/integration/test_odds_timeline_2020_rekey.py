"""Integration: the paid 2020 odds_timeline archive is re-keyed and reaches the builder.

Permanent regression proof for quick task 260816-u0e. The 1,780 paid 2020 rows were
written under the pre-``45bff24`` season-start rule, so every 2020 ``game_id`` ran one
week high and failed to join ``games`` silver -- 2020 coverage was 0 of 269 games and
the whole 2020 slice of a 7,210-credit archive contributed nothing to gold.

``tests/integration/test_gold_line_movement_columns.py`` anchors on 2023 (covered) and
2019 (before the archive floor), so it would stay green through a total regression of
2020. That gap is why this file exists.

Two layers:

1. ``TestArchiveIntegrity`` -- the archive itself, read THROUGH ``load_dataframe``
   rather than through a direct parquet read. ``load_dataframe(..., source="auto")``
   resolves DuckDB FIRST and only falls back to parquet (``data/storage.py:916-921``),
   so a stale DuckDB copy would silently shadow the parquet and feed the gold rebuild
   wrong-keyed rows. Asserting through the same seam the feature builders use is what
   makes this a proof about what the pipeline sees.
2. ``TestBuilderSeesTheReKeyedSeason`` -- ``LineMovementBuilder`` on the live archive
   for 2020: coverage recovered, all fifteen columns non-constant, no closing column.

Read-only: nothing here writes under ``data/``.
"""

from datetime import datetime
from pathlib import Path

import pandas as pd
import pytest

from data.storage import load_dataframe
from features.line_movement import LineMovementBuilder

REPO_ROOT = Path(__file__).resolve().parents[2]
_TIMELINE_PATH = REPO_ROOT / "data" / "silver" / "odds_timeline.parquet"

TARGET_SEASON = 2020

# The paid archive as bought (Plan 29-05) and re-keyed in place (quick task 260816-u0e).
EXPECTED_TOTAL_ROWS = 9957
EXPECTED_SEASON_ROWS = {
    2020: 1780,
    2021: 1219,
    2022: 1452,
    2023: 2759,
    2024: 2747,
}

# 262 stored ids collapse onto 256 true ids: six early-September board listings whose
# provisional dates later moved merge into the id carrying that game's in-week
# snapshots, across disjoint snapshot timestamps (hence zero pair loss).
EXPECTED_TRUE_2020_IDS = 256

# Builder-level 2020 coverage after the re-key, measured 2026-08-16: 244 of 269 games.
# The floor is set just below the measured value -- a real assertion with headroom for
# the structural handful whose per-game Friday-6PM-ET freeze precedes every captured
# snapshot, not a tolerance widened to make a bar green.
MIN_COVERED_2020_GAMES = 240

# The full D-09 family: seven totals features, the shared coverage flag, and the seven
# Tier (a) spread siblings.
LINE_MOVEMENT_COLUMNS = (
    "opening_total",
    "total_drift",
    "total_drift_dir",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
    "line_movement_coverage",
    "opening_spread",
    "spread_drift",
    "spread_drift_dir",
    "spread_late_drift",
    "spread_abs_travel",
    "spread_reversals",
    "spread_range",
)


@pytest.fixture(scope="module")
def timeline() -> pd.DataFrame:
    """The live odds_timeline, loaded through the seam the feature builders use."""
    if not _TIMELINE_PATH.exists():
        pytest.skip(f"{_TIMELINE_PATH} not present -- the paid archive is not on disk")
    return load_dataframe("odds_timeline", layer="silver")


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    """Silver games."""
    return load_dataframe("games", layer="silver")


def _season_of(game_ids: pd.Series) -> pd.Series:
    return game_ids.str.slice(0, 4).astype(int)


class TestArchiveIntegrity:
    """The archive is neither reduced nor double-counted by the re-key."""

    def test_total_rows_preserved(self, timeline: pd.DataFrame) -> None:
        assert len(timeline) == EXPECTED_TOTAL_ROWS

    def test_distinct_pairs_equal_row_count(self, timeline: pd.DataFrame) -> None:
        """9,957 rows AND 9,957 distinct (game_id, snapshot_ts) pairs.

        A re-key written through ``upsert_silver_composite`` would have appended the
        re-keyed rows beside the old orphans (that function has no delete path), giving
        11,737 rows with 2020 present twice under two key sets. Equality of rows and
        distinct pairs is what rules that out.
        """
        pairs = len(timeline.drop_duplicates(subset=["game_id", "snapshot_ts"]))
        assert pairs == EXPECTED_TOTAL_ROWS

    def test_per_season_row_counts(self, timeline: pd.DataFrame) -> None:
        counts = _season_of(timeline["game_id"]).value_counts().sort_index()
        assert {int(k): int(v) for k, v in counts.items()} == EXPECTED_SEASON_ROWS

    def test_2020_collapsed_to_the_true_id_count(self, timeline: pd.DataFrame) -> None:
        slice_2020 = timeline[_season_of(timeline["game_id"]) == TARGET_SEASON]
        assert slice_2020["game_id"].nunique() == EXPECTED_TRUE_2020_IDS

    def test_every_2020_id_joins_games(
        self, timeline: pd.DataFrame, games: pd.DataFrame
    ) -> None:
        """The regression this file exists for: 2020 ids must join ``games`` silver."""
        slice_2020 = timeline[_season_of(timeline["game_id"]) == TARGET_SEASON]
        true_ids = set(games[games["season"] == TARGET_SEASON]["game_id"])
        orphans = sorted(set(slice_2020["game_id"]) - true_ids)
        assert not orphans, (
            f"{len(orphans)} 2020 odds_timeline ids fail to join games silver: "
            f"{orphans[:10]}"
        )

    def test_snapshot_ts_round_trips_as_utc(self, timeline: pd.DataFrame) -> None:
        """The trajectory grain depends on a real tz-aware timestamp, not a string."""
        assert str(timeline["snapshot_ts"].dtype) == "datetime64[ns, UTC]"


class TestBuilderSeesTheReKeyedSeason:
    """LineMovementBuilder on the live archive for 2020 (was 0 of 269 covered)."""

    @pytest.fixture(scope="class")
    def builder_output(self) -> pd.DataFrame:
        if not _TIMELINE_PATH.exists():
            pytest.skip(f"{_TIMELINE_PATH} not present")
        all_games = load_dataframe("games", layer="silver")
        season_games = all_games[all_games["season"] == TARGET_SEASON]
        return LineMovementBuilder(
            timeline_df=load_dataframe("odds_timeline", layer="silver")
        ).build_features(season_games, datetime.now(), target_season=TARGET_SEASON)

    def test_coverage_recovered(self, builder_output: pd.DataFrame) -> None:
        covered = int((builder_output["line_movement_coverage"] == 1.0).sum())
        assert covered >= MIN_COVERED_2020_GAMES, (
            f"only {covered} of {len(builder_output)} {TARGET_SEASON} games carry a "
            f"trajectory; the archive re-key has regressed (it was 0 before the fix)"
        )

    def test_all_fifteen_columns_are_non_constant(
        self, builder_output: pd.DataFrame
    ) -> None:
        """Presence is not enough -- an all-default column carries no signal."""
        constant = {
            col: int(builder_output[col].nunique())
            for col in LINE_MOVEMENT_COLUMNS
            if builder_output[col].nunique() <= 1
        }
        assert not constant, (
            f"{TARGET_SEASON} line-movement columns are constant: {constant}"
        )

    def test_no_emitted_column_names_a_closing_line(
        self, builder_output: pd.DataFrame
    ) -> None:
        """The closing line is reserved for CLV grading and is never a feature (D-15)."""
        closing = [c for c in builder_output.columns if "closing" in c.lower()]
        assert not closing, closing
