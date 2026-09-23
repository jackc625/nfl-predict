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

THE COHORT HANDLE IS GONE, AND THAT IS WHY FACT 2 MOVED (Plan 33.2-20)
-----------------------------------------------------------------------
``shifted_mask`` finds the stale cohort by ``created_at == 2025-09-28 14:47:59``,
the instant of the one bad ingest run. MEASURED 2026-09-22: it now matches ZERO
rows. Silver ``games`` has been re-ingested through the corrected path since --
Plan 33.2-09's venue corrections, Plan 33.2-12's kickoff-hour repair and the live
2026 capture -- and ``upsert_silver`` re-stamps ``created_at`` on every row it
writes. Every ``created_at`` in the store is now one of 24 per-season stamps from
2026-09-14 06:43 plus one 2026-09-19 15:37 stamp for the 272 live 2026 rows.

That is the cohort MEMBERSHIP evaporating, not the staleness returning, and it was
the membership node that caught it -- which is exactly what a non-vacuity control
is for. The consequence is stated rather than absorbed: ``cohort_is_stale`` returns
False on an empty cohort, so the two cohort-staleness nodes below now pass
VACUOUSLY and say so. The load-bearing guard is fact 3, the DST correlation, which
reads the WHOLE frame per season and therefore covers a superset of the cohort --
25 of 25 seasons, none failing. The idempotency trap the membership node guards is
a property of the TOOL, so it is re-asserted against a synthetic frame that carries
the marker; a synthetic frame cannot be re-ingested away.

READ-ONLY: nothing here writes under ``data/``. Skips cleanly when the data is absent.
"""

from pathlib import Path

import pandas as pd
import pytest

from data.storage import get_db_connection
from scripts.normalize_games_kickoff_tz import (
    ANCHOR_GAME_ID,
    EXPECTED_SHIFTED_ROWS,
    cohort_is_stale,
    dst_correlation_holds,
    normalize_kickoffs,
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

        WR-10: this asserted absolute counts (6499 / 6292). ``games`` silver GROWS
        every time ``scripts/ingest_games.py`` runs -- weekly in season -- so the
        test turned red on the next routine ingest, for a reason having nothing to
        do with what its own docstring claims to guard. A brittle red test in a
        permanent suite gets muted, and this one sits beside the genuinely
        valuable instants-agree and DST-correlation assertions that would be muted
        with it.

        The INVARIANT is asserted instead: running the normalization changes no
        row count, whatever the current size. Contrast
        ``test_cohort_membership_is_deliberately_preserved``, which pins 1,926 --
        a fixed HISTORICAL cohort that cannot grow -- and is correctly stable.

        The gap between the copies is the SEPARATE, still-open N-01 coverage
        divergence (2025 games written by upsert_silver into the parquet only). It
        is deliberately NOT closed here -- re-syncing would change the gold row set
        and confound the corrected line-movement screen -- so it is asserted as a
        RELATION rather than as the constant 207.
        """
        assert len(normalize_kickoffs(games_parquet)) == len(games_parquet)
        assert len(normalize_kickoffs(games_duckdb)) == len(games_duckdb)

        assert len(games_parquet) >= len(games_duckdb), (
            "the DuckDB copy has overtaken the parquet; the N-01 coverage "
            "divergence has changed direction and needs re-diagnosing"
        )


class TestNoRowIsStale:
    """The cohort's kickoffs are true instants."""

    def test_the_parquet_cohort_is_not_stale(self, games_parquet: pd.DataFrame) -> None:
        """PASSES VACUOUSLY TODAY -- the cohort is empty (see the module docstring).

        Kept rather than deleted: the day a re-ingest re-creates a cohort carrying
        that ``created_at``, this is the node that judges it. The non-vacuous
        staleness guard meanwhile is ``TestDstCorrelationHoldsPerSeason``, which
        reads the whole frame.
        """
        assert cohort_is_stale(games_parquet) is False

    def test_the_duckdb_cohort_is_not_stale(self, games_duckdb: pd.DataFrame) -> None:
        """Same vacuity, same reason, same standing value."""
        assert cohort_is_stale(games_duckdb) is False

    def test_the_cohort_handle_was_re_ingested_away_and_the_guard_moved(
        self, games_parquet: pd.DataFrame
    ) -> None:
        """RE-ANCHORED (Plan 33.2-20): the handle is gone, so its absence is stated.

        Was ``test_cohort_membership_is_deliberately_preserved``, asserting the cohort
        still holds its 1,926 rows. It does not: ``shifted_mask`` keys on
        ``created_at == 2025-09-28 14:47:59`` and every row has been re-ingested since
        (Plan 33.2-09's venue corrections, Plan 33.2-12's kickoff-hour repair, the live
        2026 capture), each write re-stamping ``created_at``. Re-anchoring the count to
        0 alone would record a number and lose the point, so THREE things are asserted:

        1. the handle really is gone -- no row carries the marker;
        2. the store carries real re-ingest stamps rather than nulls, so "gone" means
           re-ingested and not erased;
        3. the guard the old node stood for is intact somewhere it cannot evaporate --
           the DST correlation covers every season in the frame, which is a SUPERSET
           of the cohort it replaced.
        """
        created = pd.to_datetime(games_parquet["created_at"], utc=True)
        assert int(shifted_mask(games_parquet).sum()) == 0, (
            "the stale-ingest cohort is back. It was re-ingested away by Phase 33.2; "
            f"a row carrying the {EXPECTED_SHIFTED_ROWS}-row cohort's stamp means "
            "something restored the pre-correction ingest."
        )
        assert int(created.isna().sum()) == 0, (
            "created_at is null somewhere, so the cohort marker is unmatched because "
            "the stamp was ERASED rather than replaced by a later ingest"
        )
        assert created.nunique() > 1, (
            "every row shares one created_at, which is the single-bad-run shape the "
            "cohort marker was built to find"
        )

        correlation = dst_correlation_holds(games_parquet)
        # The probe keys seasons as strings; compared in one representation so the
        # assertion is about coverage rather than about a type.
        covered = {str(season) for season in correlation}
        seasons = {str(season) for season in games_parquet["season"].unique()}
        assert covered == seasons, (
            f"the DST correlation covers {sorted(covered)} but the frame "
            f"holds {sorted(seasons)}. It is the non-vacuous replacement for the "
            "cohort check, so a season it cannot judge is a season with no guard."
        )
        assert not [season for season, ok in correlation.items() if not ok]

    def test_the_idempotency_trap_is_still_a_property_of_the_tool(self) -> None:
        """The claim the membership node really made, asserted where it cannot evaporate.

        The trap: the normalization shifts ``kickoff_et`` and never ``created_at``, so
        a run does NOT empty the cohort, and anything keying "already done" on cohort
        emptiness would double-shift on a re-run. That is a property of the TOOL, not
        of production data, so it is driven on a synthetic frame carrying the marker --
        which no re-ingest can take away.
        """
        from scripts.normalize_games_kickoff_tz import SHIFTED_COHORT_CREATED_AT

        frame = pd.DataFrame(
            {
                "game_id": ["2023_W12_AAA@BBB", "2023_W13_CCC@DDD"],
                "season": [2023, 2023],
                "kickoff_et": pd.to_datetime(
                    ["2023-11-24 15:00:00+00:00", "2023-12-03 13:00:00+00:00"],
                    utc=True,
                ),
                "created_at": [SHIFTED_COHORT_CREATED_AT, SHIFTED_COHORT_CREATED_AT],
            }
        )
        assert int(shifted_mask(frame).sum()) == 2, "fixture sanity"

        shifted = normalize_kickoffs(frame)

        assert int(shifted_mask(shifted).sum()) == 2, (
            "a run emptied the cohort, so `already normalized` could be keyed on "
            "cohort emptiness -- and the second --apply would shift every row again"
        )
        assert not shifted["kickoff_et"].equals(frame["kickoff_et"]), (
            "the tool moved no kickoff, so the assertion above is vacuous"
        )

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
