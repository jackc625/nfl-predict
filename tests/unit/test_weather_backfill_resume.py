"""The resume rule, the refusals around it, and the null-observation classifier.

WHY A FILENAME TEST CANNOT WORK HERE, WHICH IS THE WHOLE POINT OF THIS MODULE
-----------------------------------------------------------------------------
`data/bronze/weather_raw_bronze_2025_W00_20260407T025733.parquet` is a WHOLE-SEASON
`week=0` sentinel written by the current code, and it holds **78 rows covering weeks 1
to 5 only** -- `_load_games_data(2025, None)` read a silver `games` table that held 78
rows of 2025 at the time. A resume rule of "this season is done iff a `_W00_` snapshot
exists for it" therefore silently leaves 207 games of 2025 unfetched behind a green log.
That is defect N-06, and `test_the_2025_partial_snapshot_is_not_a_completed_season` is
the regression for it.

The opposite rule does not work either. "Done iff the union of this season's snapshots
covers every game id" fixes 2025 but marks 2018-2024 done from the SEVEN LEGACY
`weather_raw_bronze_{season}_season.parquet` files -- and re-fetching those seven
seasons under the corrected routing is the entire purpose of SPEC R2.

So the new corpus is written under the distinct table name `weather_backfill`, and the
coverage test reads that name ONLY. The glob is disjoint from all ten legacy filenames
BY CONSTRUCTION, which is how SPEC prohibition 6 ("the 1,942 pre-existing bronze rows
are neither overwritten nor deleted") is satisfied structurally rather than by care.

NO TEST IN THIS MODULE TOUCHES THE NETWORK. The archive fetch is replaced at the
`_fetch_openmeteo_weather` seam Plan 33.1-02 established, and every write goes into the
`sandbox_data_root` fixture -- a real directory somewhere else, not a patched name.
`monkeypatch` alone is NOT a sandbox: during Phase 33 Wave 6 a test that patched a table
name while the writer still resolved the production root destroyed all three production
gold matrices.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import fnmatch
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import scripts.backfill_historical_weather as backfill
from data.storage import save_bronze_snapshot
from tests import phase33_state
from tests.data_boundary import digest_file
from utils.exceptions import DataIngestionError, WeatherDataError

REPO_ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_BRONZE = REPO_ROOT / "data" / "bronze"

# A season whose games are read from the REAL silver table rather than typed. A fixture
# that describes a game the feed does not have is a fixture that cannot refuse the way
# production refuses -- Plan 33.1-02 replaced four such fixtures for exactly that reason.
SMALL_SEASON = 2003


# ---------------------------------------------------------------------------
# Helpers. Every games frame is DERIVED from the committed silver table, and every
# write lands under `sandbox_data_root`.
# ---------------------------------------------------------------------------


def _silver_games() -> pd.DataFrame:
    """The committed silver games table, read once per call. Read-only."""
    return pd.read_parquet(REPO_ROOT / "data" / "silver" / "games.parquet")


def _season_games(season: int, limit: int | None = None) -> pd.DataFrame:
    """The first *limit* games of *season*, in the silver frame's own column shape."""
    frame = _silver_games()
    frame = frame[frame["season"] == season].copy()
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame.reset_index(drop=True)


def _outdoor_season_games(season: int, limit: int | None = None) -> pd.DataFrame:
    """Only the games the pinned feed says weather applies to, so every row is fetched.

    The null-rate classifier's denominator is the games that were actually ASKED about.
    Mixing domes into a fixture would dilute the fraction and make a test that reads as
    "30 percent absent" assert something else.
    """
    facts = backfill.load_pinned_game_facts((season,))
    applicable = {
        str(row.game_id)
        for row in facts.itertuples()
        if str(row.roof).lower().strip() in ("outdoors", "open")
    }
    frame = _silver_games()
    frame = frame[
        (frame["season"] == season) & (frame["game_id"].isin(applicable))
    ].copy()
    if limit is not None:
        frame = frame.head(limit).copy()
    return frame.reset_index(drop=True)


MEASUREMENT = {
    "temp_f": 55.0,
    "temp_c": 12.8,
    "wind_mph": 8.0,
    "wind_direction": 180.0,
    "humidity_pct": 60.0,
    "precip_mm": 0.0,
    "precip_prob": None,
    "condition": None,
    "condition_code": 0,
    "visibility_km": None,
    "dew_point_f": 40.0,
    "apparent_temp_f": 54.0,
    "snowfall_cm": 0.0,
    "wind_gusts_mph": 12.0,
    "cloud_cover_pct": 30.0,
    "weather_code": 0,
}


def _install_stub_fetch(monkeypatch, absent_indices=()) -> dict:
    """Replace the archive fetch seam. Returns a mutable call record.

    *absent_indices* are positions IN FETCH ORDER that return the
    ``ABSENT_OBSERVATION`` sentinel, so a test can shape an absence pattern -- scattered
    or contiguous -- and drive the classifier through the real season path.
    """
    record: dict = {"calls": 0}
    absent = set(absent_indices)

    async def _fetch(self, *args, **kwargs):
        index = record["calls"]
        record["calls"] += 1
        if index in absent:
            return backfill.ABSENT_OBSERVATION
        return dict(MEASUREMENT)

    monkeypatch.setattr(
        backfill.HistoricalWeatherBackfiller,
        "_fetch_openmeteo_weather",
        _fetch,
        raising=True,
    )
    return record


def _write_coverage_snapshot(
    root: Path, season: int, game_ids, stamp: str = "20260101T000000"
) -> Path:
    """Write a `weather_backfill` snapshot carrying exactly *game_ids*.

    Only `game_id` is written because only `game_id` is what the coverage rule reads.
    A fixture that carried a full weather row would be asserting something the rule
    does not look at.
    """
    path = (
        root
        / "bronze"
        / f"{backfill.BACKFILL_BRONZE_TABLE}_raw_bronze_{season}_W00_{stamp}.parquet"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"game_id": sorted(str(v) for v in game_ids)}).to_parquet(
        path, index=False
    )
    return path


def _copy_legacy_corpus(root: Path) -> None:
    """Put the ten REAL legacy bronze files into the sandbox, bytes intact."""
    (root / "bronze").mkdir(parents=True, exist_ok=True)
    for name in backfill.LEGACY_WEATHER_BRONZE_FILENAMES:
        shutil.copy2(PRODUCTION_BRONZE / name, root / "bronze" / name)


def _backfill_files(root: Path) -> list[Path]:
    return sorted(
        (root / "bronze").glob(f"{backfill.BACKFILL_BRONZE_TABLE}_raw_bronze_*.parquet")
    )


class _FrozenClock:
    """A stand-in for `data.storage.datetime` whose `now` never advances.

    `save_bronze_snapshot`'s filename carries a SECOND-resolution stamp, so the
    same-second collision it guards against is a real race that a wall clock reproduces
    only by luck. Pinning the clock makes the collision deterministic rather than flaky.
    """

    def __init__(self, instant: datetime) -> None:
        self._instant = instant

    def now(self, tz=None):
        return self._instant if tz is None else self._instant.astimezone(tz)


# ---------------------------------------------------------------------------
# Ruling K: the resume rule
# ---------------------------------------------------------------------------


class TestTheResumeRule:
    """Coverage under a NEW table name, never a filename test."""

    def test_the_2025_partial_snapshot_is_not_a_completed_season(
        self, sandbox_data_root
    ):
        """THE N-06 REGRESSION, and the single highest-value test in this phase.

        With ONLY the legacy corpus on disk -- which includes the 78-row
        `weather_raw_bronze_2025_W00_20260407T025733.parquet` whole-season sentinel --
        2025 must still be reported as MISSING. A rule that read `_W00_` existence
        would report it done and leave 207 games unfetched behind a green log.
        """
        _copy_legacy_corpus(sandbox_data_root)

        missing = backfill.seasons_still_missing(base_path=sandbox_data_root)

        assert 2025 in missing, (
            "2025 is reported COVERED while only the 78-row legacy _W00_ partial is on "
            "disk. That file is a whole-season sentinel holding weeks 1 to 5 only; "
            "treating it as a completed season leaves 207 games unfetched."
        )
        assert not backfill.season_is_covered(2025, base_path=sandbox_data_root)

    def test_the_legacy_seasons_are_still_fetched(self, sandbox_data_root):
        """SPEC R2: the seven legacy seasons are RE-FETCHED, not skipped.

        The legacy files cover 2018-2024 completely, so a coverage rule that globbed
        the `weather` table name would mark all seven done and skip the re-fetch under
        the corrected routing -- which is the entire purpose of this phase.
        """
        _copy_legacy_corpus(sandbox_data_root)

        missing = backfill.seasons_still_missing(base_path=sandbox_data_root)

        assert tuple(missing) == tuple(
            range(backfill.CORPUS_FIRST_SEASON, backfill.CORPUS_LAST_SEASON + 1)
        )
        for season in range(2018, 2025):
            assert season in missing

    def test_a_complete_new_snapshot_marks_the_season_covered(self, sandbox_data_root):
        """The NEGATIVE CONTROL: without it every assertion above would pass on a rule
        that simply always returns False."""
        _copy_legacy_corpus(sandbox_data_root)
        ids = backfill.pinned_season_game_ids(SMALL_SEASON)
        _write_coverage_snapshot(sandbox_data_root, SMALL_SEASON, ids)

        missing = backfill.seasons_still_missing(base_path=sandbox_data_root)

        assert backfill.season_is_covered(SMALL_SEASON, base_path=sandbox_data_root)
        assert SMALL_SEASON not in missing
        assert len(missing) == (
            backfill.CORPUS_LAST_SEASON - backfill.CORPUS_FIRST_SEASON
        )

    def test_a_partial_new_snapshot_is_not_a_completed_season(self, sandbox_data_root):
        """An interrupted partial WRITE is not a completed season either."""
        ids = sorted(backfill.pinned_season_game_ids(SMALL_SEASON))
        _write_coverage_snapshot(sandbox_data_root, SMALL_SEASON, ids[:-1])

        assert not backfill.season_is_covered(SMALL_SEASON, base_path=sandbox_data_root)
        assert SMALL_SEASON in backfill.seasons_still_missing(
            base_path=sandbox_data_root
        )

    def test_the_new_table_glob_is_disjoint_from_every_legacy_filename(self):
        """SPEC prohibition 6, satisfied STRUCTURALLY: checked, not assumed."""
        pattern = f"{backfill.BACKFILL_BRONZE_TABLE}_raw_bronze_*"
        collisions = [
            name
            for name in backfill.LEGACY_WEATHER_BRONZE_FILENAMES
            if fnmatch.fnmatch(name, pattern)
        ]
        assert collisions == []
        assert len(backfill.LEGACY_WEATHER_BRONZE_FILENAMES) == 10


class TestTheLegacyBronzeIsUntouched:
    """SPEC prohibition 6, pinned by CONTENT DIGEST rather than by row count.

    A row count would miss a rewrite that preserved the count, and the legacy corpus is
    the ONLY evidence the routing fix can be regression-tested against.
    """

    def test_legacy_bronze_is_untouched(self):
        inventory = phase33_state.LEGACY_WEATHER_BRONZE_INVENTORY
        assert set(inventory["files"]) == set(
            backfill.LEGACY_WEATHER_BRONZE_FILENAMES
        ), (
            "the recorded inventory and the module's declared legacy filenames "
            "disagree about which files ARE the legacy corpus"
        )
        for name, recorded in inventory["files"].items():
            path = PRODUCTION_BRONZE / name
            assert path.is_file(), (
                f"{name} is missing from data/bronze. It is the only evidence the "
                "routing fix can be regression-tested against; its absence is a "
                "broken checkout, not a reason to skip."
            )
            assert digest_file(path) == recorded["digest"], (
                f"{name} has CHANGED since Plan 33.1-05 measured it.\n"
                f"  recorded: {recorded['digest']}\n"
                f"  measured: {digest_file(path)}\n"
                "The new corpus writes under a DISJOINT table name precisely so this "
                "cannot happen."
            )

    def test_the_recorded_row_counts_reproduce(self):
        inventory = phase33_state.LEGACY_WEATHER_BRONZE_INVENTORY
        for name, recorded in inventory["files"].items():
            frame = pd.read_parquet(PRODUCTION_BRONZE / name)
            assert len(frame) == recorded["rows"]
            assert len(frame.columns) == recorded["columns"]
        assert inventory["legacy_total_rows"] == 1942
        assert inventory["grand_total_rows"] == 2048
        assert (
            inventory["files"]["weather_raw_bronze_2025_W00_20260407T025733.parquet"][
                "rows"
            ]
            == 78
        )

    def test_a_run_leaves_the_legacy_files_byte_identical(
        self, sandbox_data_root, monkeypatch
    ):
        """The same digests hold AFTER a season is written beside them."""
        _copy_legacy_corpus(sandbox_data_root)
        before = {
            name: digest_file(sandbox_data_root / "bronze" / name)
            for name in backfill.LEGACY_WEATHER_BRONZE_FILENAMES
        }
        _install_stub_fetch(monkeypatch)

        backfill.backfill_season(
            SMALL_SEASON,
            base_path=sandbox_data_root,
            games_df=_season_games(SMALL_SEASON, limit=4),
            throttle=backfill.RequestThrottle(0.0),
        )

        for name, digest in before.items():
            assert digest_file(sandbox_data_root / "bronze" / name) == digest


# ---------------------------------------------------------------------------
# The interrupt, the double run and the same-second write
# ---------------------------------------------------------------------------


class TestAnInterruptedRunIsResumable:
    def test_interrupt_leaves_silver_untouched(self, sandbox_data_root, monkeypatch):
        """Three seasons, the third raising: two snapshots, silver byte-identical."""
        silver = sandbox_data_root / "silver" / "weather.parquet"
        pd.DataFrame({"game_id": ["2002_W01_SF@NYG"], "temp_f": [55.0]}).to_parquet(
            silver, index=False
        )
        before = digest_file(silver)
        _install_stub_fetch(monkeypatch)

        seasons = (2002, 2003, 2004)

        def _games(season: int) -> pd.DataFrame:
            if season == 2004:
                raise WeatherDataError("simulated interruption during season 2004")
            return _season_games(season, limit=3)

        with pytest.raises(WeatherDataError):
            backfill.backfill_corpus(
                seasons=seasons,
                base_path=sandbox_data_root,
                games_provider=_games,
                throttle=backfill.RequestThrottle(0.0),
                verify_archive_floor=False,
            )

        assert len(_backfill_files(sandbox_data_root)) == 2
        assert digest_file(silver) == before

    def test_a_resumed_run_fetches_only_the_seasons_still_missing(
        self, sandbox_data_root
    ):
        ids = backfill.pinned_season_game_ids(2002)
        _write_coverage_snapshot(sandbox_data_root, 2002, ids)

        missing = backfill.seasons_still_missing(base_path=sandbox_data_root)

        assert 2002 not in missing
        assert 2003 in missing


class TestTheWriteIsAppendOnlyAndIdempotent:
    def test_a_same_second_second_write_raises_rather_than_overwriting(
        self, sandbox_data_root, monkeypatch
    ):
        """`exclusive=True` decides the race at the CREATE, across processes."""
        monkeypatch.setattr(
            "data.storage.datetime",
            _FrozenClock(datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC)),
        )
        frame = pd.DataFrame({"game_id": ["2003_W01_SF@NYG"]})

        first = save_bronze_snapshot(
            frame,
            backfill.BACKFILL_BRONZE_TABLE,
            season=SMALL_SEASON,
            week=0,
            base_path=sandbox_data_root,
            exclusive=True,
        )
        first_bytes = first.read_bytes()

        with pytest.raises(FileExistsError):
            save_bronze_snapshot(
                pd.DataFrame({"game_id": ["OVERWRITTEN"]}),
                backfill.BACKFILL_BRONZE_TABLE,
                season=SMALL_SEASON,
                week=0,
                base_path=sandbox_data_root,
                exclusive=True,
            )

        assert first.read_bytes() == first_bytes

    def test_a_double_run_leaves_the_silver_row_count_unchanged(
        self, sandbox_data_root, monkeypatch
    ):
        """SPEC edge idempotency/R4: append a NEW snapshot, upsert latest-wins."""
        _install_stub_fetch(monkeypatch)
        games = _season_games(SMALL_SEASON, limit=5)

        for stamp in (
            datetime(2026, 9, 13, 12, 0, 0, tzinfo=UTC),
            datetime(2026, 9, 13, 12, 0, 1, tzinfo=UTC),
        ):
            monkeypatch.setattr("data.storage.datetime", _FrozenClock(stamp))
            backfill.backfill_season(
                SMALL_SEASON,
                base_path=sandbox_data_root,
                games_df=games,
                throttle=backfill.RequestThrottle(0.0),
            )
            backfill.promote_corpus_to_silver(base_path=sandbox_data_root)

        rows = len(pd.read_parquet(sandbox_data_root / "silver" / "weather.parquet"))

        assert len(_backfill_files(sandbox_data_root)) == 2
        assert rows == len(games)


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


class TestTheRefusalsThatFireBeforeAnyWrite:
    def test_an_empty_season_raises_and_writes_no_snapshot(self, sandbox_data_root):
        """SPEC edge empty/R2: no empty bronze snapshot is ever written."""
        empty = _season_games(SMALL_SEASON, limit=0)

        with pytest.raises(DataIngestionError) as excinfo:
            backfill.backfill_season(
                SMALL_SEASON,
                base_path=sandbox_data_root,
                games_df=empty,
                throttle=backfill.RequestThrottle(0.0),
            )

        assert str(SMALL_SEASON) in str(excinfo.value)
        assert _backfill_files(sandbox_data_root) == []

    @pytest.mark.parametrize("season", [2001, 2099])
    def test_a_season_outside_the_corpus_range_is_refused_by_name(
        self, sandbox_data_root, season
    ):
        """SPEC edge boundary/R3, refused BEFORE any request is made."""
        with pytest.raises(DataIngestionError) as excinfo:
            backfill.backfill_season(
                season,
                base_path=sandbox_data_root,
                throttle=backfill.RequestThrottle(0.0),
            )

        message = str(excinfo.value)
        assert str(season) in message
        assert str(backfill.CORPUS_FIRST_SEASON) in message
        assert str(backfill.CORPUS_LAST_SEASON) in message
        assert _backfill_files(sandbox_data_root) == []


# ---------------------------------------------------------------------------
# Ruling L4: the three-way null-observation classification
# ---------------------------------------------------------------------------


class TestTheNullObservationClassifier:
    """A fraction ALONE cannot tell an exhausted limit from an honest ERA5 gap.

    RESEARCH A2's unprobed assumption is that limit exhaustion returns a 400 carrying a
    `reason`. If the provider instead answers 200-with-null-arrays, the absences arrive
    CONSECUTIVELY, while genuine ERA5 gaps scatter across a season by geography and
    date. A 30 percent scattered season and a 30 percent consecutive season are
    DIFFERENT FACTS and the gate must say which one it saw.
    """

    def test_no_absences_is_clean(self):
        arm, evidence = backfill.classify_null_observations([False] * 200)
        assert arm == backfill.NULL_ARM_CLEAN
        assert evidence["absent"] == 0
        assert evidence["fraction"] == 0.0
        assert evidence["longest_contiguous_run"] == 0

    def test_an_isolated_fraction_is_clean(self):
        flags = [False] * 200
        flags[7] = True
        arm, evidence = backfill.classify_null_observations(flags)
        assert arm == backfill.NULL_ARM_CLEAN
        assert evidence["fraction"] == pytest.approx(0.005)

    def test_a_contiguous_run_of_eight_is_outage_like_even_below_the_fraction_floor(
        self,
    ):
        """The CONTIGUOUS-RUN test is the discriminating one, independent of fraction.

        Eight consecutive absences out of 200 is a 4 percent fraction -- well under the
        25 percent outage floor -- and is still OUTAGE-LIKE, because a provider that
        has stopped answering fails CONSECUTIVE requests.
        """
        flags = [False] * 200
        for i in range(40, 40 + backfill.OUTAGE_CONTIGUOUS_RUN):
            flags[i] = True

        arm, evidence = backfill.classify_null_observations(flags)

        assert arm == backfill.NULL_ARM_OUTAGE_LIKE
        assert evidence["fraction"] == pytest.approx(0.04)
        assert evidence["fraction"] < backfill.OUTAGE_NULL_FRACTION_FLOOR
        assert evidence["longest_contiguous_run"] == backfill.OUTAGE_CONTIGUOUS_RUN

    def test_a_scattered_thirty_percent_is_outage_like_on_the_fraction_alone(self):
        flags = [i % 3 == 0 for i in range(201)]
        arm, evidence = backfill.classify_null_observations(flags)
        assert arm == backfill.NULL_ARM_OUTAGE_LIKE
        assert evidence["longest_contiguous_run"] == 1

    def test_a_scattered_five_percent_is_ambiguous(self):
        flags = [False] * 200
        for i in range(0, 200, 20):
            flags[i] = True
        arm, evidence = backfill.classify_null_observations(flags)
        assert arm == backfill.NULL_ARM_AMBIGUOUS
        assert evidence["fraction"] == pytest.approx(0.05)

    def test_the_three_arms_are_the_recorded_ones(self):
        assert sorted(phase33_state.NULL_OBSERVATION_CLASSIFICATION["arms"]) == sorted(
            backfill.NULL_OBSERVATION_ARMS
        )
        assert not hasattr(backfill, "MAX_NULL_OBSERVATION_FRACTION_PER_SEASON"), (
            "the single hard bar was REPLACED by the three-way classification: R4 makes "
            "an isolated missing observation recordable data, so a flat 1 percent bar "
            "would convert legitimate ERA5 absence into a phase-stopping condition"
        )


class TestTheNullGateInsideASeasonRun:
    def test_an_outage_like_null_pattern_raises_and_writes_no_snapshot(
        self, sandbox_data_root, monkeypatch
    ):
        games = _outdoor_season_games(SMALL_SEASON, limit=20)
        _install_stub_fetch(monkeypatch, absent_indices=range(0, 6))

        with pytest.raises(DataIngestionError) as excinfo:
            backfill.backfill_season(
                SMALL_SEASON,
                base_path=sandbox_data_root,
                games_df=games,
                throttle=backfill.RequestThrottle(0.0),
            )

        message = str(excinfo.value)
        assert backfill.NULL_ARM_OUTAGE_LIKE in message
        assert str(SMALL_SEASON) in message
        assert "0.30" in message
        assert "6" in message
        assert _backfill_files(sandbox_data_root) == []

    def test_an_isolated_null_fraction_is_written_with_no_override(
        self, sandbox_data_root, monkeypatch
    ):
        games = _outdoor_season_games(SMALL_SEASON)
        assert len(games) >= 100
        _install_stub_fetch(monkeypatch, absent_indices=(3,))

        report = backfill.backfill_season(
            SMALL_SEASON,
            base_path=sandbox_data_root,
            games_df=games,
            throttle=backfill.RequestThrottle(0.0),
        )

        assert report["null_arm"] == backfill.NULL_ARM_CLEAN
        assert report["override_accepted"] is False
        assert len(_backfill_files(sandbox_data_root)) == 1

    def test_an_ambiguous_season_refuses_and_names_the_override_flag(
        self, sandbox_data_root, monkeypatch
    ):
        games = _outdoor_season_games(SMALL_SEASON, limit=20)
        _install_stub_fetch(monkeypatch, absent_indices=(4,))

        with pytest.raises(DataIngestionError) as excinfo:
            backfill.backfill_season(
                SMALL_SEASON,
                base_path=sandbox_data_root,
                games_df=games,
                throttle=backfill.RequestThrottle(0.0),
            )

        message = str(excinfo.value)
        assert backfill.NULL_ARM_AMBIGUOUS in message
        assert "--accept-null-fraction" in message
        assert f"{SMALL_SEASON}=0.050000" in message
        assert _backfill_files(sandbox_data_root) == []

    def test_an_override_carrying_the_observed_fraction_writes_the_season(
        self, sandbox_data_root, monkeypatch
    ):
        games = _outdoor_season_games(SMALL_SEASON, limit=20)
        _install_stub_fetch(monkeypatch, absent_indices=(4,))

        report = backfill.backfill_season(
            SMALL_SEASON,
            base_path=sandbox_data_root,
            games_df=games,
            throttle=backfill.RequestThrottle(0.0),
            accepted_null_fractions={SMALL_SEASON: 0.05},
        )

        assert report["null_arm"] == backfill.NULL_ARM_AMBIGUOUS
        assert report["override_accepted"] is True
        assert len(_backfill_files(sandbox_data_root)) == 1

    def test_an_override_carrying_the_wrong_fraction_still_refuses(
        self, sandbox_data_root, monkeypatch
    ):
        """A season cannot be pre-authorised BLIND: the flag carries the OBSERVED
        fraction, so the operator has to have read the number first."""
        games = _outdoor_season_games(SMALL_SEASON, limit=20)
        _install_stub_fetch(monkeypatch, absent_indices=(4,))

        with pytest.raises(DataIngestionError) as excinfo:
            backfill.backfill_season(
                SMALL_SEASON,
                base_path=sandbox_data_root,
                games_df=games,
                throttle=backfill.RequestThrottle(0.0),
                accepted_null_fractions={SMALL_SEASON: 0.10},
            )

        assert "--accept-null-fraction" in str(excinfo.value)
        assert _backfill_files(sandbox_data_root) == []

    def test_an_accepted_override_is_recorded_with_its_evidence(
        self, sandbox_data_root, monkeypatch
    ):
        """An override nobody can find afterwards is indistinguishable from no gate."""
        games = _outdoor_season_games(SMALL_SEASON, limit=20)
        _install_stub_fetch(monkeypatch, absent_indices=(4,))

        backfill.backfill_season(
            SMALL_SEASON,
            base_path=sandbox_data_root,
            games_df=games,
            throttle=backfill.RequestThrottle(0.0),
            accepted_null_fractions={SMALL_SEASON: 0.05},
        )

        recorded = backfill.recorded_null_overrides(base_path=sandbox_data_root)

        assert len(recorded) == 1
        entry = recorded[0]
        assert entry["season"] == SMALL_SEASON
        assert entry["fraction"] == pytest.approx(0.05)
        assert entry["longest_contiguous_run"] == 1
        assert entry["recorded_at"]
