"""The MOS bulletin decoder, the request plan and the bronze backfill (Plan 33.2-11 Task 2).

Every unit trap in the IEM payload has its own test (RESEARCH pitfall P6), the AVN -> GFS
changeover is pinned on both sides of 2003-12-16, the silent-empty 200 is a named refusal, and the
backfill is exercised end to end against an in-memory transport in a temporary data root: the first
run lands bronze and a second run over complete bronze makes zero requests and moves nothing.

No test here touches the network or the production data lake.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pandas as pd
import pytest

from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from scripts import backfill_mos_forecasts as backfill
from scripts.mos_decode import (
    KNOTS_TO_MPH,
    EmptyMosResponseError,
    MosDecodeError,
    MosModelMismatchError,
    build_weather_record,
    decode_record,
    model_for_run_date,
    nearest_record,
    parse_archive_instant,
    qpf_precip_level,
    relative_humidity_pct,
    run_instant,
)

# A verbatim 2002 AVN row as the archive returns it (probed 2026-09-21): note there is NO snw key.
AVN_2002_ROW: dict = {
    "runtime": "2002-09-07T12:00:00.000",
    "ftime": "2002-09-07T18:00:00.000",
    "model": "AVN",
    "n_x": None,
    "tmp": 83,
    "dpt": 67,
    "cld": "CL",
    "wdr": 210,
    "wsp": 9,
    "p06": None,
    "p12": None,
    "q06": None,
    "q12": None,
    "t06_1": 2.0,
    "t06_2": None,
    "t12_1": None,
    "t12_2": None,
    "cig": 7,
    "vis": 7.0,
    "obv": "N ",
    "poz": 0,
    "pos": 0,
    "typ": "R ",
    "station": "KGRB",
    "t06": None,
    "t12": None,
}


def _row(**overrides: object) -> dict:
    row = dict(AVN_2002_ROW)
    row.update(overrides)
    return row


# ---------------------------------------------------------------------------
# Unit traps
# ---------------------------------------------------------------------------


class TestUnitTraps:
    def test_ten_knots_decode_to_eleven_and_a_half_mph_exactly_once(self) -> None:
        record = decode_record(_row(wsp=10))
        assert record.wind_mph == pytest.approx(11.5078)
        assert record.wind_mph == pytest.approx(10 * KNOTS_TO_MPH)

    def test_wind_direction_is_already_degrees_and_is_not_multiplied_again(
        self,
    ) -> None:
        assert decode_record(_row(wdr=110)).wind_direction == 110.0

    def test_a_wind_direction_multiplied_twice_raises(self) -> None:
        with pytest.raises(MosDecodeError, match="multiplied by ten"):
            decode_record(_row(wdr=1100))

    def test_north_is_stored_as_zero_so_the_schema_bound_holds(self) -> None:
        assert decode_record(_row(wdr=360)).wind_direction == 0.0

    def test_qpf_category_two_maps_to_the_committed_moderate_level(self) -> None:
        record = decode_record(_row(q06=2))
        assert record.qpf_category == 2
        assert record.precip_level == 2
        assert qpf_precip_level(1) == 1
        assert qpf_precip_level(3) == qpf_precip_level(5) == 3

    def test_qpf_is_never_turned_into_a_millimetre_amount(self) -> None:
        runtime = "2002-09-07T12:00:00.000"
        rows = [_row(runtime=runtime, ftime="2002-09-08T18:00:00.000", q06=2, p06=40)]
        built = build_weather_record(
            "2002_W01_X@Y",
            datetime(2002, 9, 8, 17, 0, tzinfo=UTC),
            [decode_record(r) for r in rows],
        )
        assert built is not None
        assert built["precip_mm"] is None
        assert built["mos_qpf_category"] == 2
        assert built["mos_precip_level"] == 2
        assert built["precip_prob"] == pytest.approx(0.40)

    def test_the_missing_code_is_an_unknown_level_never_level_zero(self) -> None:
        assert qpf_precip_level(9) is None
        assert decode_record(_row(q06=9)).precip_level is None

    def test_an_impossible_qpf_category_raises(self) -> None:
        with pytest.raises(MosDecodeError):
            decode_record(_row(q06=7))

    def test_a_missing_snw_key_parses_and_is_told_apart_from_a_null(self) -> None:
        absent = decode_record(AVN_2002_ROW)
        null = decode_record(_row(snw=None))
        assert "snw" not in AVN_2002_ROW
        assert absent.snw_key_present is False
        assert null.snw_key_present is True
        assert absent.snw is None and null.snw is None

    def test_humidity_is_the_magnus_identity_and_saturates_at_one_hundred(self) -> None:
        assert relative_humidity_pct(68.0, 50.0) == pytest.approx(52.6, abs=0.5)
        assert relative_humidity_pct(40.0, 40.0) == 100.0
        assert relative_humidity_pct(40.0, 41.0) == 100.0
        assert relative_humidity_pct(None, 41.0) is None


class TestTimestampsAreLocalizedAtTheParseBoundary:
    def test_the_archive_naive_string_becomes_aware_utc(self) -> None:
        parsed = parse_archive_instant("2023-12-16T12:00:00.000", field="runtime")
        assert parsed == datetime(2023, 12, 16, 12, tzinfo=UTC)
        assert parsed.tzinfo is not None

    def test_a_string_carrying_an_offset_is_refused(self) -> None:
        with pytest.raises(MosDecodeError):
            parse_archive_instant("2023-12-16T12:00:00.000+00:00", field="runtime")

    def test_a_naive_kickoff_never_reaches_a_comparison(self) -> None:
        records = [decode_record(AVN_2002_ROW)]
        with pytest.raises(MosDecodeError, match="naive"):
            nearest_record(records, datetime(2002, 9, 7, 18))
        with pytest.raises(MosDecodeError, match="naive"):
            build_weather_record("g", datetime(2002, 9, 7, 18), records)

    def test_the_schema_refuses_a_naive_cycle_instant(self) -> None:
        row = _weather_row(forecast_issue_time=datetime(2002, 9, 7, 12))
        with pytest.raises(ValueError, match="no time zone"):
            WeatherSchema(**row)


class TestTheModelSwitchesOnTheRunDate:
    def test_the_eve_of_the_changeover_is_avn_and_the_day_itself_is_gfs(self) -> None:
        assert model_for_run_date(date(2003, 12, 15)) == "AVN"
        assert model_for_run_date(date(2003, 12, 16)) == "GFS"
        assert model_for_run_date(date(2002, 9, 7)) == "AVN"
        assert model_for_run_date(date(2025, 1, 4)) == "GFS"

    def test_a_2003_station_season_straddling_the_changeover_is_two_clipped_requests(
        self,
    ) -> None:
        lock_dates = [date(2003, 9, 6), date(2003, 12, 14), date(2003, 12, 20)]
        segments = backfill.plan_segments(_covered("KGRB", 2003, lock_dates))
        assert [s.model for s in segments] == ["AVN", "GFS"]
        avn, gfs = segments
        assert avn.lock_dates == (date(2003, 9, 6), date(2003, 12, 14))
        assert gfs.lock_dates == (date(2003, 12, 20),)
        assert avn.ets < "2003-12-16" <= gfs.sts
        assert set(avn.lock_dates).isdisjoint(gfs.lock_dates)

    def test_a_2004_station_season_is_exactly_one_request(self) -> None:
        segments = backfill.plan_segments(
            _covered("KGRB", 2004, [date(2004, 9, 11), date(2005, 1, 1)])
        )
        assert len(segments) == 1
        assert segments[0].model == "GFS"


# ---------------------------------------------------------------------------
# Content refusals
# ---------------------------------------------------------------------------


def _segment() -> backfill.MosSegment:
    return backfill.MosSegment("KGRB", 2002, "AVN", (date(2002, 9, 7),))


class TestContentRefusals:
    def test_an_empty_response_raises_by_name(self) -> None:
        with pytest.raises(EmptyMosResponseError, match="zero rows"):
            backfill.select_lock_runs([], _segment())

    def test_a_row_labelled_with_another_model_raises(self) -> None:
        with pytest.raises(MosModelMismatchError):
            backfill.select_lock_runs([_row(model="GFS")], _segment())

    def test_a_lock_run_missing_from_a_non_empty_response_is_named_not_filled(
        self,
    ) -> None:
        kept, missing = backfill.select_lock_runs(
            [_row(runtime="2002-09-07T00:00:00.000")], _segment()
        )
        assert kept == []
        assert missing == ["2002-09-07T12:00:00.000"]

    def test_only_the_lock_date_12_utc_run_is_kept(self) -> None:
        rows = [_row(), _row(runtime="2002-09-07T18:00:00.000")]
        kept, missing = backfill.select_lock_runs(rows, _segment())
        assert [r["runtime"] for r in kept] == ["2002-09-07T12:00:00.000"]
        assert missing == []


# ---------------------------------------------------------------------------
# Corpus: non-US venues, post-lock moves
# ---------------------------------------------------------------------------


def _games_frame(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestTheCorpus:
    def test_a_non_us_venue_produces_no_request_and_is_recorded_uncoverable(
        self,
    ) -> None:
        games = _games_frame(
            [
                {
                    "game_id": "2019_W05_CHI@OAK",
                    "season": 2019,
                    "week": 5,
                    "stadium_id": "LON02",
                    "kickoff_et": pd.Timestamp("2019-10-06 13:30", tz=UTC),
                }
            ]
        )
        covered, uncoverable = backfill.load_corpus(games)
        assert covered == []
        assert [u.stadium_id for u in uncoverable] == ["LON02"]
        assert backfill.plan_segments(covered) == []

    def test_a_post_lock_moved_game_uses_the_pre_move_venues_station(self) -> None:
        games = _games_frame(
            [
                {
                    "game_id": "2003_W08_MIA@LAC",
                    "season": 2003,
                    "week": 8,
                    "stadium_id": "PHO99",
                    "kickoff_et": pd.Timestamp("2003-10-27 14:00", tz=UTC),
                }
            ]
        )
        covered, _ = backfill.load_corpus(games)
        assert [(g.stadium_id, g.station) for g in covered] == [("SDG00", "KSAN")]
        assert covered[0].lock_date == date(2003, 10, 26)


# ---------------------------------------------------------------------------
# The schema carries the new columns, and the round-trip assertion fires
# ---------------------------------------------------------------------------


def _weather_row(**overrides: object) -> dict:
    row = {
        "game_id": "2002_W01_NYJ@BUF",
        "forecast_time": datetime(2002, 9, 7, 12, tzinfo=UTC),
        "game_time": datetime(2002, 9, 8, 17, tzinfo=UTC),
        "temp_f": 70.0,
        "wind_mph": 8.0,
        "is_outdoor": True,
        "weather_coverage": True,
        "weather_source": "historical_forecast",
        "forecast_issue_time": datetime(2002, 9, 7, 12, tzinfo=UTC),
        "mos_model": "AVN",
        "mos_station": "KBUF",
        "mos_qpf_category": 2,
        "mos_precip_level": 2,
    }
    row.update(overrides)
    return row


class TestTheSchemaCarriesEveryEmittedColumn:
    def test_every_mos_column_survives_validate_bronze_to_silver(self) -> None:
        source = pd.DataFrame([_weather_row()])
        promoted = validate_bronze_to_silver(source, WeatherSchema)
        backfill.assert_mos_columns_survived(promoted, source)
        for column in backfill.MOS_COLUMNS:
            assert promoted[column].notna().all(), column

    def test_the_round_trip_assertion_fires_when_a_column_is_dropped(self) -> None:
        source = pd.DataFrame([_weather_row()])
        promoted = validate_bronze_to_silver(source, WeatherSchema).drop(
            columns=["mos_model"]
        )
        with pytest.raises(Exception, match="NO mos_model column"):
            backfill.assert_mos_columns_survived(promoted, source)

    def test_the_cycle_instant_description_says_it_is_not_a_publication_time(
        self,
    ) -> None:
        description = (
            WeatherSchema.model_fields["forecast_issue_time"].description or ""
        )
        assert "model-CYCLE instant" in description
        assert "NOT a recorded" in description
        assert "config/mos_tolerance.py" in description


# ---------------------------------------------------------------------------
# End to end, in a temporary data root, against an in-memory archive
# ---------------------------------------------------------------------------


def _covered(
    station: str, season: int, lock_dates: list[date]
) -> list[backfill.CoveredGame]:
    return [
        backfill.CoveredGame(
            game_id=f"{season}_W{i + 1:02d}_A@B",
            season=season,
            stadium_id="GNB00",
            station=station,
            kickoff_utc=run_instant(d) + timedelta(hours=29),
            lock_date=d,
        )
        for i, d in enumerate(lock_dates)
    ]


def _run_rows(station: str, model: str, run: datetime) -> list[dict]:
    return [
        _row(
            station=station,
            model=model,
            runtime=run.strftime("%Y-%m-%dT%H:%M:%S.000"),
            ftime=(run + timedelta(hours=hours)).strftime("%Y-%m-%dT%H:%M:%S.000"),
            tmp=50,
            dpt=40,
            wsp=8,
            wdr=200,
        )
        for hours in range(6, 75, 3)
    ]


def _fake_archive(
    requests: list[httpx.Request],
    *,
    empty: bool = False,
    range_drops: frozenset[str] = frozenset(),
    archive_lacks: frozenset[str] = frozenset(),
) -> httpx.MockTransport:
    """An archive serving six-hourly runs, three-hourly hours out to +72 h.

    *range_drops*: runtimes the date-range endpoint omits (the single-run endpoint still has them).
    *archive_lacks*: runtimes absent from BOTH endpoints -- a genuine archive gap.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        params = request.url.params
        station, model = params["station"], params["model"]
        if request.url.path.endswith("/api/1/mos.json"):
            run = datetime.strptime(params["runtime"], "%Y-%m-%d %H:%MZ").replace(
                tzinfo=UTC
            )
            key = run.strftime("%Y-%m-%dT%H:%M:%S.000")
            data = [] if key in archive_lacks else _run_rows(station, model, run)
            return httpx.Response(200, text=json.dumps({"data": data}))
        if empty:
            return httpx.Response(200, text="[]")
        start = datetime.strptime(params["sts"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        end = datetime.strptime(params["ets"], "%Y-%m-%dT%H:%MZ").replace(tzinfo=UTC)
        rows = []
        run = start
        while run <= end:
            key = run.strftime("%Y-%m-%dT%H:%M:%S.000")
            if key not in range_drops and key not in archive_lacks:
                rows += _run_rows(station, model, run)
            run += timedelta(hours=6)
        return httpx.Response(200, text=json.dumps(rows))

    return httpx.MockTransport(handler)


def _games_for_backfill() -> pd.DataFrame:
    # Lambeau (KGRB) in 2003 on both sides of the changeover, and one London game.
    return _games_frame(
        [
            {
                "game_id": "2003_W01_A@GB",
                "season": 2003,
                "week": 1,
                "stadium_id": "GNB00",
                "kickoff_et": pd.Timestamp("2003-09-07 17:00", tz=UTC),
            },
            {
                "game_id": "2003_W15_B@GB",
                "season": 2003,
                "week": 15,
                "stadium_id": "GNB00",
                "kickoff_et": pd.Timestamp("2003-12-14 18:00", tz=UTC),
            },
            {
                "game_id": "2003_W16_C@GB",
                "season": 2003,
                "week": 16,
                "stadium_id": "GNB00",
                "kickoff_et": pd.Timestamp("2003-12-21 18:00", tz=UTC),
            },
            {
                "game_id": "2007_W08_NYG@MIA",
                "season": 2007,
                "week": 8,
                "stadium_id": "LON00",
                "kickoff_et": pd.Timestamp("2007-10-28 17:00", tz=UTC),
            },
        ]
    )


def _files(root: Path) -> dict[str, bytes]:
    return {
        str(p.relative_to(root)): p.read_bytes()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


class TestTheBackfillEndToEnd:
    def test_first_run_lands_bronze_and_second_run_is_a_no_op(
        self, tmp_path: Path
    ) -> None:
        requests: list[httpx.Request] = []
        client = httpx.Client(transport=_fake_archive(requests))

        first = backfill.run_backfill(
            [2003, 2007],
            apply=True,
            base_path=tmp_path,
            client=client,
            sleep=lambda _: None,
            games=_games_for_backfill(),
        )
        assert first.segments_planned == 2
        assert first.split_season_requests == 1
        assert first.requests_made == 2
        assert first.segments_fetched == 2
        assert first.segments_missing == 0
        assert first.empty_responses == 0
        assert first.uncoverable_games == 1
        assert first.resolved_share == 1.0
        assert {r.url.params["model"] for r in requests} == {"AVN", "GFS"}
        assert not (tmp_path / "bronze" / "mos" / ".mos_backfill.lock").exists()

        before = _files(tmp_path)
        second = backfill.run_backfill(
            [2003, 2007],
            apply=True,
            base_path=tmp_path,
            client=client,
            sleep=lambda _: None,
            games=_games_for_backfill(),
        )
        assert second.requests_made == 0
        assert second.segments_fetched == 0
        assert second.segments_already_complete == 2
        assert second.segments_missing == 0
        assert second.resolved_share == 1.0
        assert len(requests) == 2
        assert _files(tmp_path) == before

    def test_an_empty_response_is_a_named_refusal_and_writes_nothing(
        self, tmp_path: Path
    ) -> None:
        requests: list[httpx.Request] = []
        client = httpx.Client(transport=_fake_archive(requests, empty=True))
        report = backfill.run_backfill(
            [2003],
            apply=True,
            base_path=tmp_path,
            client=client,
            sleep=lambda _: None,
            games=_games_for_backfill(),
        )
        assert report.empty_responses == 1
        assert report.refusal is not None and "zero rows" in report.refusal
        assert report.segments_fetched == 0
        assert report.segments_missing == 2
        assert list((tmp_path / "bronze" / "mos").glob("*.parquet")) == []

    def test_a_confirmed_archive_gap_is_recorded_unresolved_and_never_filled(
        self, tmp_path: Path
    ) -> None:
        # The 2003-12-13 12 UTC run (lock date of 2003_W15_B@GB) is absent everywhere.
        gap = "2003-12-13T12:00:00.000"
        requests: list[httpx.Request] = []
        client = httpx.Client(
            transport=_fake_archive(requests, archive_lacks=frozenset({gap}))
        )
        first = backfill.run_backfill(
            [2003],
            apply=True,
            base_path=tmp_path,
            client=client,
            sleep=lambda _: None,
            games=_games_for_backfill(),
        )
        assert first.refusal is None
        assert first.empty_responses == 0
        assert first.absent_runs == 1
        assert first.requests_made == 3  # two segments plus one single-run re-ask
        assert first.segments_missing == 0
        assert first.resolved_games == 2 and first.covered_games == 3

        frames = [
            pd.read_parquet(p) for p in (tmp_path / "bronze" / "mos").glob("*.parquet")
        ]
        bronze = pd.concat(frames, ignore_index=True)
        marker = bronze[bronze["absent_in_archive"]]
        assert marker["runtime_raw"].tolist() == [gap]
        assert marker["raw_json"].isna().all()
        assert "api/1/mos.json" in marker["request_url"].iloc[0]

        before = _files(tmp_path)
        second = backfill.run_backfill(
            [2003],
            apply=True,
            base_path=tmp_path,
            client=client,
            sleep=lambda _: None,
            games=_games_for_backfill(),
        )
        assert second.requests_made == 0
        assert second.resolved_games == 2
        assert _files(tmp_path) == before

    def test_a_run_the_range_endpoint_dropped_is_recovered_from_the_single_run_lookup(
        self, tmp_path: Path
    ) -> None:
        dropped = "2003-12-13T12:00:00.000"
        requests: list[httpx.Request] = []
        client = httpx.Client(
            transport=_fake_archive(requests, range_drops=frozenset({dropped}))
        )
        report = backfill.run_backfill(
            [2003],
            apply=True,
            base_path=tmp_path,
            client=client,
            sleep=lambda _: None,
            games=_games_for_backfill(),
        )
        assert report.absent_runs == 0
        assert report.resolved_share == 1.0
        assert report.segments_missing == 0

    def test_without_apply_nothing_is_requested_or_written(
        self, tmp_path: Path
    ) -> None:
        requests: list[httpx.Request] = []
        client = httpx.Client(transport=_fake_archive(requests))
        report = backfill.run_backfill(
            [2003],
            apply=False,
            base_path=tmp_path,
            client=client,
            games=_games_for_backfill(),
        )
        assert report.requests_made == 0
        assert report.segments_missing == 2
        assert requests == []
        assert _files(tmp_path) == {}

    def test_the_output_contract_prints_every_required_key(self) -> None:
        keys = [line.split("=")[0] for line in backfill.BackfillReport().lines()]
        for required in (
            "SEGMENTS_PLANNED",
            "SPLIT_SEASON_REQUESTS",
            "SEGMENTS_ALREADY_COMPLETE",
            "SEGMENTS_FETCHED",
            "REQUESTS_MADE",
            "SEGMENTS_MISSING",
            "EMPTY_RESPONSES",
            "RESOLVED_SHARE",
        ):
            assert required in keys
