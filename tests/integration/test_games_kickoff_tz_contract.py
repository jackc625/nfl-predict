"""Integration: ``games.kickoff_et`` carries ONE contract, in BOTH silver copies.

WR-06 / N-01 / N-02. This is the permanent guard for the timezone normalization the
quick task 260817-dyp applied, and it exists because the pre-fix state was invisible
to every check the project already had.

The three facts pinned here, each of which was FALSE or unverified before:

1. **The two copies agree on kickoff INSTANTS for every shared row.**
   ``load_dataframe(..., source="auto")`` resolves DuckDB FIRST
   (``data/storage.py:916-921``) and ``db.table_exists("games")`` is True, while
   ``upsert_silver`` writes the PARQUET only. So the pipeline reads one copy and
   every later write lands in the other. A fix applied to only one copy is either
   invisible to the whole pipeline or a divergent fallback waiting to activate.

2. **No row is still stale.** The stale cohort is a single 2025-09-28 ingest run
   whose 1,926 rows held ET wall clocks mislabelled as UTC. Membership of that
   cohort is NOT the test -- the normalization shifts ``kickoff_et`` and never
   touches ``created_at``, so the cohort keeps its rows forever. Staleness is tested
   by the DST correlation, which is the discriminator that identified the defect.

3. **The DST correlation holds per season.** For a fixed ET wall-clock window a true
   UTC instant shifts its stored hour by one across the November EDT-to-EST boundary.
   Per season, so one surviving cohort cannot hide inside an aggregate.

READ-ONLY: nothing here writes under ``data/``. Skips cleanly when the data is absent.
"""

from pathlib import Path

import pandas as pd
import pytest

from data.storage import get_db_connection
from scripts.normalize_games_kickoff_tz import (
    ANCHOR_GAME_ID,
    EXPECTED_DUCKDB_ROWS,
    EXPECTED_PARQUET_ROWS,
    EXPECTED_SHIFTED_ROWS,
    cohort_is_stale,
    dst_correlation_holds,
    shifted_mask,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
_GAMES_PARQUET = REPO_ROOT / "data" / "silver" / "games.parquet"

ET = "America/New_York"

# The 2023 Black Friday game: the named anchor for the whole remediation.
ANCHOR_ET = "2023-11-24 15:00"


@pytest.fixture(scope="module")
def games_parquet() -> pd.DataFrame:
    if not _GAMES_PARQUET.exists():
        pytest.skip(f"{_GAMES_PARQUET} not present")
    return pd.read_parquet(_GAMES_PARQUET, engine="pyarrow")


@pytest.fixture(scope="module")
def games_duckdb() -> pd.DataFrame:
    db = get_db_connection()
    if not db.table_exists("games"):
        pytest.skip("no DuckDB games table on this machine")
    return db.fetch_df("SELECT * FROM games")


def _et_wall_clock(series: pd.Series) -> pd.Series:
    values = pd.to_datetime(series)
    if values.dt.tz is None:
        return values
    return values.dt.tz_convert(ET).dt.tz_localize(None)


class TestTheTwoCopiesAgree:
    """N-01: the parquet and the DuckDB table must not drift apart on VALUES."""

    def test_kickoff_instants_are_identical_on_every_shared_row(
        self, games_parquet: pd.DataFrame, games_duckdb: pd.DataFrame
    ) -> None:
        pq = pd.to_datetime(games_parquet.set_index("game_id")["kickoff_et"], utc=True)
        db = pd.to_datetime(games_duckdb.set_index("game_id")["kickoff_et"], utc=True)
        shared = pq.index.intersection(db.index)

        assert len(shared) > 6000, "the two copies barely overlap -- check the join key"

        disagreements = pq.loc[shared] != db.loc[shared]
        offenders = sorted(pq.loc[shared][disagreements].index)[:10]
        assert not disagreements.any(), (
            f"{int(disagreements.sum())} shared rows disagree on the kickoff instant. "
            f"The pipeline reads DuckDB first, so a parquet-only fix is invisible and "
            f"a DuckDB-only fix leaves a divergent fallback. Offenders: {offenders}"
        )

    def test_row_coverage_is_unchanged_by_the_normalization(
        self, games_parquet: pd.DataFrame, games_duckdb: pd.DataFrame
    ) -> None:
        """The normalization shifted values in place; it must never add or drop rows.

        The 207-row gap between the copies is the SEPARATE, still-open N-01 coverage
        divergence (2025 games written by upsert_silver into the parquet only). It is
        deliberately NOT closed here: re-syncing would change the gold row set and
        confound the corrected line-movement screen.
        """
        assert len(games_parquet) == EXPECTED_PARQUET_ROWS
        assert len(games_duckdb) == EXPECTED_DUCKDB_ROWS


class TestNoRowIsStale:
    """The cohort's kickoffs are true instants."""

    def test_the_parquet_cohort_is_not_stale(self, games_parquet: pd.DataFrame) -> None:
        assert cohort_is_stale(games_parquet) is False

    def test_the_duckdb_cohort_is_not_stale(self, games_duckdb: pd.DataFrame) -> None:
        assert cohort_is_stale(games_duckdb) is False

    def test_cohort_membership_is_deliberately_preserved(
        self, games_parquet: pd.DataFrame
    ) -> None:
        """Guards the idempotency trap.

        The tool shifts ``kickoff_et`` and never ``created_at``, so the cohort KEEPS
        its 1,926 rows after a successful run. Anything that keys "already done" on
        cohort emptiness will never fire and will double-shift on a re-run.
        """
        assert int(shifted_mask(games_parquet).sum()) == EXPECTED_SHIFTED_ROWS

    def test_no_kickoff_reads_earlier_than_the_earliest_real_window(
        self, games_parquet: pd.DataFrame
    ) -> None:
        """A stale row's 1 PM ET kickoff reads 08:00/09:00 ET -- not an NFL window.

        London games kick at 09:30 ET and are the genuine earliest, so a floor of
        09:00 is a real assertion with a little headroom, not a widened tolerance.
        """
        et_hour = _et_wall_clock(games_parquet["kickoff_et"]).dt.hour
        assert et_hour.min() >= 9, (
            f"a kickoff reads {et_hour.min()}:00 ET, earlier than any real NFL "
            f"window -- a stale cohort survived"
        )


class TestDstCorrelationHoldsPerSeason:
    """The discriminator that identified the defect, kept as the standing guard."""

    def test_every_season_shifts_its_stored_hour_across_the_dst_boundary(
        self, games_parquet: pd.DataFrame
    ) -> None:
        correlation = dst_correlation_holds(games_parquet)
        failing = sorted(season for season, ok in correlation.items() if not ok)

        assert correlation, "the correlation probe found no comparable season at all"
        assert not failing, (
            f"seasons {failing} do not shift their stored hour across the November "
            f"EDT-to-EST boundary for a fixed ET window, which is the signature of an "
            f"ET wall clock mislabelled as UTC"
        )

    def test_the_duckdb_copy_holds_the_correlation_too(
        self, games_duckdb: pd.DataFrame
    ) -> None:
        failing = sorted(
            season
            for season, ok in dst_correlation_holds(games_duckdb).items()
            if not ok
        )
        assert not failing


class TestTheNamedAnchor:
    """2023_W12_MIA@NYJ -- the Black Friday game the whole remediation turns on."""

    def test_the_anchor_reads_friday_afternoon_in_both_copies(
        self, games_parquet: pd.DataFrame, games_duckdb: pd.DataFrame
    ) -> None:
        for name, frame in (("parquet", games_parquet), ("duckdb", games_duckdb)):
            row = frame[frame["game_id"] == ANCHOR_GAME_ID]
            assert len(row) == 1, f"{ANCHOR_GAME_ID} missing from the {name} copy"

            wall_clock = _et_wall_clock(row["kickoff_et"]).iloc[0]
            assert wall_clock.strftime("%Y-%m-%d %H:%M") == ANCHOR_ET
            assert wall_clock.strftime("%a") == "Fri"
