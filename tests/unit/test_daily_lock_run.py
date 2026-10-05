"""The daily lock-time run's own rules (Plan 33.2-27).

* A run that starts at or after today's 18:00 ET lock refuses BEFORE any request.
* A day with no games tomorrow is a recorded no-op.
* The slate is exactly tomorrow's games, selected by their shared lock.
* Every slate game is predicted, stamped with its three instants, and merged into the week's file
  without disturbing other games' rows; a game computed after its lock is refused, never
  back-dated; a game that silently produced no prediction is a named refusal.
* Elo gains a flagged provisional row for every unplayed game of the slate's week (review
  WR-08: the week's rank/percentile then ranks the same ratings training did).

Every test writes only under ``tmp_path``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

import scripts.daily_lock_pipeline as daily
from pipeline import daily_steps, live_skip
from pipeline.daily_steps import DailySlate, slate_lock

RUN_DATE = date(2026, 9, 26)


@pytest.fixture(autouse=True)
def trigger_calls(monkeypatch, tmp_path) -> list[datetime]:
    """No run here touches the real Task Scheduler or the production ledger run log (Plan 34-15).

    Every non-dry daily run now regenerates the closing wake triggers through ``schtasks``; the
    stand-in records the instant it was asked for instead. The ledger run log is redirected into
    ``tmp_path``.
    """
    from forward_ledger import closing_schedule, run_log

    calls: list[datetime] = []

    def _register(_games, now, **_kwargs):
        calls.append(now)
        return closing_schedule.TriggerRegistration(
            "skipped", "test_stand_in", (), False
        )

    monkeypatch.setattr(closing_schedule, "register_closing_triggers", _register)
    monkeypatch.setattr(run_log, "LEDGER_RUN_LOG", tmp_path / "ledger_runs.jsonl")
    return calls


def _schedule() -> pd.DataFrame:
    """A Saturday (today), two Sunday games (tomorrow) and a Monday game."""
    rows = [
        ("2026_W03_ATL@GB", "2026-09-26 17:00"),
        ("2026_W03_LAC@BUF", "2026-09-27 17:00"),
        ("2026_W03_LA@DEN", "2026-09-28 00:20"),  # 20:20 ET Sunday
        ("2026_W03_PHI@CHI", "2026-09-29 00:15"),  # 20:15 ET Monday
    ]
    return pd.DataFrame(
        {
            "game_id": [g for g, _ in rows],
            "season": 2026,
            "week": 3,
            "kickoff_et": [pd.Timestamp(k, tz="UTC") for _, k in rows],
            "home_team": ["GB", "BUF", "DEN", "CHI"],
            "away_team": ["ATL", "LAC", "LA", "PHI"],
        }
    )


# ---------------------------------------------------------------------------
# Before collection
# ---------------------------------------------------------------------------


def test_a_run_after_the_lock_refuses_before_any_request(monkeypatch, tmp_path, capsys):
    records = tmp_path / "daily.jsonl"
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", records)
    monkeypatch.setattr(daily, "_recorded_tomorrow_ids", lambda _d: ["a", "b"])

    def _no_request(*_a, **_k):
        raise AssertionError("a request was made after the lock")

    monkeypatch.setattr(daily, "_refresh_schedule", _no_request)

    start = slate_lock(RUN_DATE).astimezone(
        UTC
    )  # AT the lock: nothing may be collected
    assert daily.run_daily(RUN_DATE, start=start, dry_run=False) == 0

    out = capsys.readouterr().out
    assert "LOCK_PASSED_BEFORE_COLLECTION= 2" in out
    record = json.loads(records.read_text(encoding="utf-8"))
    assert record["outcome"] == "lock_passed"
    assert record["game_ids"] == ["a", "b"]


def test_a_day_with_no_games_tomorrow_is_a_recorded_no_op(
    monkeypatch, tmp_path, capsys
):
    records = tmp_path / "daily.jsonl"
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", records)
    monkeypatch.setattr(daily, "_refresh_schedule", lambda *_a, **_k: 2026)
    schedule = _schedule()
    schedule = schedule.loc[schedule["game_id"] == "2026_W03_PHI@CHI"]
    monkeypatch.setattr(
        "data.storage.load_dataframe", lambda *_a, **_k: schedule.copy()
    )

    def _no_steps(_slate):
        raise AssertionError("steps were built for a day with no games")

    monkeypatch.setattr(daily, "build_daily_step_registry", _no_steps)

    start = slate_lock(RUN_DATE).astimezone(UTC) - timedelta(hours=8)
    assert daily.run_daily(RUN_DATE, start=start, dry_run=False) == 0
    assert "NEXT_DAY_GAMES= 0" in capsys.readouterr().out
    assert json.loads(records.read_text(encoding="utf-8"))["outcome"] == "no_games"


def test_the_slate_is_exactly_tomorrows_games():
    start = slate_lock(RUN_DATE) - timedelta(hours=8)
    slate = daily.select_slate(_schedule(), RUN_DATE, decided_at=start)
    assert slate.game_ids == {"2026_W03_LAC@BUF", "2026_W03_LA@DEN"}
    assert slate.lock == slate_lock(RUN_DATE)
    assert slate.lock.hour == 18


def test_selecting_after_the_lock_is_refused():
    late = slate_lock(RUN_DATE) + timedelta(seconds=1)
    with pytest.raises(live_skip.LockPassedError):
        daily.select_slate(_schedule(), RUN_DATE, decided_at=late)


# ---------------------------------------------------------------------------
# --date is judged at its own lock; a past date writes nothing (C1 WR-09)
# ---------------------------------------------------------------------------


def _record_week_three(monkeypatch) -> None:
    """Make the fixture week (plus the next week's opener) the recorded schedule."""
    from utils import current_slate

    rows = _schedule()
    rows = pd.concat(
        [
            rows,
            pd.DataFrame(
                {
                    "game_id": ["2026_W04_SEA@ARI"],
                    "season": [2026],
                    "week": [4],
                    "kickoff_et": [pd.Timestamp("2026-10-02 00:15", tz="UTC")],
                    "home_team": ["ARI"],
                    "away_team": ["SEA"],
                }
            ),
        ],
        ignore_index=True,
    ).assign(game_type="REG")
    prepared = current_slate.prepare_schedule(rows)
    monkeypatch.setattr(current_slate, "load_recorded_schedule", lambda *_a: prepared)


def test_a_past_date_is_refused_before_anything_is_written(monkeypatch, tmp_path):
    records = tmp_path / "daily.jsonl"
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", records)

    def _no_request(*_a, **_k):
        raise AssertionError("a past --date reached the schedule refresh")

    monkeypatch.setattr(daily, "_refresh_schedule", _no_request)

    next_day = slate_lock(RUN_DATE).astimezone(UTC) + timedelta(hours=10)
    with pytest.raises(daily.RunDateRefusedError, match="before today"):
        daily.run_daily(RUN_DATE, start=next_day, dry_run=False)
    assert not records.exists(), "a past --date wrote a production record"


def test_the_cli_reports_a_refused_date_with_exit_code_two(monkeypatch, capsys):
    monkeypatch.setattr(
        daily,
        "run_daily",
        lambda *_a, **_k: (_ for _ in ()).throw(daily.RunDateRefusedError("past")),
    )
    assert daily.main(["--date", "2026-09-01"]) == 2
    assert "RUN_REFUSED= past" in capsys.readouterr().out


def test_a_future_date_in_another_week_is_refused_for_a_real_run_only(monkeypatch):
    _record_week_three(monkeypatch)
    monday = slate_lock(date(2026, 9, 28)).astimezone(UTC) - timedelta(hours=8)
    week_four_eve = date(2026, 9, 30)  # its slate is the week-4 Thursday game

    with pytest.raises(daily.RunDateRefusedError, match="week 4"):
        daily._refuse_an_unrunnable_date(week_four_eve, monday, dry_run=False)
    daily._refuse_an_unrunnable_date(week_four_eve, monday, dry_run=True)
    daily._refuse_an_unrunnable_date(date(2026, 9, 28), monday, dry_run=False)


def test_the_slate_week_is_judged_at_the_slate_lock_not_the_clock(monkeypatch):
    """Was: resolved at the wall clock, so a --date in any other week always refused."""
    _record_week_three(monkeypatch)
    week_four = pd.DataFrame(
        {
            "game_id": ["2026_W04_SEA@ARI"],
            "season": [2026],
            "week": [4],
            "kickoff_et": [pd.Timestamp("2026-10-02 00:15", tz="UTC")],
        }
    )
    slate = DailySlate(
        run_date_et=date(2026, 9, 30),
        lock=slate_lock(date(2026, 9, 30)),
        schedule=week_four,
    )
    daily._require_slate_is_current_week(slate)  # no refusal, whatever the clock says


# ---------------------------------------------------------------------------
# A run that cannot finish records the slate as unpredicted (C1 CR-05 = B WR-05)
# ---------------------------------------------------------------------------


def _records(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_a_refused_run_records_the_slate_as_getting_no_predictions(
    monkeypatch, tmp_path, capsys
):
    import pipeline.steps as steps_mod
    from pipeline import orchestrator

    records = tmp_path / "daily.jsonl"
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", records)
    monkeypatch.setattr(daily, "_refresh_schedule", lambda *_a, **_k: 2026)
    monkeypatch.setattr(
        "data.storage.load_dataframe", lambda *_a, **_k: _schedule().copy()
    )
    monkeypatch.setattr(daily, "_require_slate_is_current_week", lambda *_a: None)
    monkeypatch.setattr(daily, "build_daily_step_registry", lambda _slate: [])
    monkeypatch.setattr(steps_mod, "_predictions_output_dir", lambda: tmp_path)

    class _Refused:
        def __init__(self, **_k):
            pass

        def run(self):
            raise RuntimeError("Pre-flight staleness checks failed: a live prior run")

    monkeypatch.setattr(orchestrator, "FridayPipeline", _Refused)

    start = slate_lock(RUN_DATE).astimezone(UTC) - timedelta(hours=1)
    with pytest.raises(RuntimeError, match="staleness"):
        daily.run_daily(RUN_DATE, start=start, dry_run=False)

    (record,) = _records(records)
    assert record["outcome"] == "run_failed"
    assert record["game_ids"] == ["2026_W03_LA@DEN", "2026_W03_LAC@BUF"]
    assert "staleness" in record["error"]
    assert "NOT_PREDICTED 2026_W03_LAC@BUF" in capsys.readouterr().out


def test_a_refresh_that_fails_records_tomorrows_recorded_games(monkeypatch, tmp_path):
    records = tmp_path / "daily.jsonl"
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", records)
    monkeypatch.setattr(daily, "_recorded_tomorrow_ids", lambda _d: ["g1", "g2"])

    def _broken(*_a, **_k):
        raise KeyboardInterrupt

    monkeypatch.setattr(daily, "_refresh_schedule", _broken)

    start = slate_lock(RUN_DATE).astimezone(UTC) - timedelta(hours=1)
    with pytest.raises(KeyboardInterrupt):
        daily.run_daily(RUN_DATE, start=start, dry_run=False)
    (record,) = _records(records)
    assert (record["outcome"], record["game_ids"]) == ("run_failed", ["g1", "g2"])


def test_a_missing_model_feature_raises_a_recordable_error_not_system_exit(
    monkeypatch,
):
    import scripts.generate_current_week_predictions as gen

    monkeypatch.setattr(
        gen,
        "load_model_artifact",
        lambda target, **_k: {
            "model": object(),
            "feature_list": ["feature_gone"],
            "calibrator": None,
        },
    )
    monkeypatch.setattr(
        gen,
        "load_gold_features",
        lambda *_a: pd.DataFrame({"game_id": ["2026_W03_LAC@BUF"]}),
    )
    with pytest.raises(KeyError, match="feature_gone"):
        gen.run_predictions(Path("artifacts"), 2026, 3)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------


def _slate(days_from_now: int) -> DailySlate:
    """The two Sunday fixture games, moved so they kick off *days_from_now* days from now."""
    schedule = _schedule()
    tomorrow = schedule.loc[
        schedule["game_id"].isin(["2026_W03_LAC@BUF", "2026_W03_LA@DEN"])
    ].reset_index(drop=True)
    shift = (
        pd.Timestamp(datetime.now(UTC)).normalize()
        + pd.Timedelta(days=days_from_now)
        - tomorrow["kickoff_et"].min().normalize()
    )
    tomorrow["kickoff_et"] = tomorrow["kickoff_et"] + shift
    first_kickoff_et = tomorrow["kickoff_et"].min().tz_convert("America/New_York")
    run_date = first_kickoff_et.date() - timedelta(days=1)
    return DailySlate(
        run_date_et=run_date,
        lock=slate_lock(run_date),
        schedule=tomorrow,
        captured_at_utc=datetime.now(UTC),
    )


@pytest.fixture
def predict_env(monkeypatch, tmp_path):
    """Route the prediction step's reads and writes into *tmp_path*."""
    import pipeline.steps as steps_mod
    import scripts.generate_current_week_predictions as gen

    live_skip.reset_excluded_games()
    monkeypatch.setattr(steps_mod, "_predictions_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        gen, "build_game_context", lambda ids, s, w: pd.DataFrame({"game_id": ids})
    )
    scored: dict[str, list[str]] = {"ids": []}

    def _build(season, week, **kwargs):
        ids = sorted(scored["ids"] or kwargs["only_game_ids"])
        return pd.DataFrame(
            {
                "game_id": ids,
                "season": season,
                "week": week,
                "wp_prob": 0.6,
                "ats_prediction": 2.5,
                "ou_prediction": 44.0,
            }
        )

    monkeypatch.setattr(gen, "build_predictions", _build)
    return tmp_path, scored


def test_every_slate_game_is_predicted_stamped_and_merged(predict_env):
    out_dir, _ = predict_env
    week_file = out_dir / "predictions_2026_week3.csv"
    pd.DataFrame(
        {"game_id": ["2026_W03_ATL@GB"], "season": [2026], "week": [3]}
    ).to_csv(week_file, index=False)
    slate = _slate(30)  # its lock is ahead, so the computation instant precedes it

    daily_steps.predict_slate(slate)

    written = pd.read_csv(week_file)
    assert set(written["game_id"]) == {
        "2026_W03_ATL@GB",
        "2026_W03_LAC@BUF",
        "2026_W03_LA@DEN",
    }
    new = written.loc[written["game_id"].isin(slate.game_ids)]
    assert list(new["wp_prob"]) == [0.6, 0.6]
    for column in daily_steps.STAMP_COLUMNS:
        assert new[column].notna().all(), column
    computed = pd.to_datetime(new["computed_at_utc"], utc=True)
    assert (computed <= pd.Timestamp(slate.lock)).all()


def test_a_game_computed_after_its_lock_is_refused_and_not_written(predict_env):
    out_dir, _ = predict_env
    slate = _slate(-30)  # its lock is long past

    with pytest.raises(live_skip.LockPassedError):
        daily_steps.predict_slate(slate)
    assert not (out_dir / "predictions_2026_week3.csv").exists()


def test_a_slate_game_with_no_prediction_is_a_named_refusal(predict_env):
    _out_dir, scored = predict_env
    scored["ids"] = ["2026_W03_LAC@BUF"]
    slate = _slate(30)

    with pytest.raises(RuntimeError, match="2026_W03_LA@DEN"):
        daily_steps.predict_slate(slate)


# ---------------------------------------------------------------------------
# Elo
# ---------------------------------------------------------------------------


def test_elo_gains_provisional_rows_for_the_slates_whole_week(monkeypatch):
    """Every unplayed game of the slate's week gets its week-start row, not only the slate.

    CHANGED BY REVIEW WR-08 (owner ruling B). This test used to pin "exactly the slate".
    The rank/percentile ranks every team's latest pre-game rating at ``week <= W``, and
    training saw a row for every week-W game; a slate-only store ranked the week's later
    teams on a rating one result stale. ``PHI@CHI`` is not on the slate and now has a row.
    """
    from pipeline.steps import persist_current_season_elo
    from scripts import build_elo

    saved: dict[str, pd.DataFrame] = {}
    week_games = ["2026_W03_LAC@BUF", "2026_W03_LA@DEN", "2026_W03_PHI@CHI"]

    class _FakeElo:
        pending_snapshot_rows = 0

        def update_current_season(self):
            real = pd.DataFrame(
                {"game_id": ["2026_W02_DET@BUF"], "is_provisional": [False]}
            )
            return build_elo.LiveSeasonUpdate(season=2026, snapshots=real)

        def load_games_data(self, seasons=None):
            return pd.DataFrame({"game_id": week_games, "week": 3})

        def snapshot_upcoming_week(self, season, week, *, games=None):
            return pd.DataFrame({"game_id": week_games, "is_provisional": True})

        def save_live_append(self, season, *, snapshots):
            saved["frame"] = snapshots

    monkeypatch.setattr(build_elo, "EloBuilder", _FakeElo)
    persist_current_season_elo(frozenset({"2026_W03_LAC@BUF", "2026_W03_LA@DEN"}))

    frame = saved["frame"]
    provisional = set(frame.loc[frame["is_provisional"], "game_id"])
    assert provisional == set(week_games)
    assert "2026_W02_DET@BUF" in set(frame["game_id"])


# ---------------------------------------------------------------------------
# A slate game whose forecast failed (step 27b)
# ---------------------------------------------------------------------------


def test_a_slate_game_with_no_forecast_is_built_with_weather_unknown(monkeypatch):
    """It is carried as an explicit no-observation row and named with its reason."""
    import features.weather as weather_mod
    from data import storage

    schedule = _schedule()
    slate = DailySlate(
        run_date_et=RUN_DATE,
        lock=slate_lock(RUN_DATE),
        schedule=schedule.loc[
            schedule["game_id"].isin(["2026_W03_LAC@BUF", "2026_W03_LA@DEN"])
        ].reset_index(drop=True),
        weather_failures={"2026_W03_LA@DEN": "WeatherDataError: Open-Meteo timeout"},
    )
    weather = pd.DataFrame({"game_id": ["2026_W03_LAC@BUF"]})
    monkeypatch.setattr(
        storage,
        "load_dataframe",
        lambda name, *_a, **_k: schedule if name == "games" else weather,
    )
    monkeypatch.setattr(storage, "save_dataframe", lambda *_a, **_k: None)
    seen: dict[str, frozenset[str]] = {}

    class _Calculator:
        def build_weather_features(self, *, games_df, weather_df, unobserved_game_ids):
            seen["unobserved"] = unobserved_game_ids
            return pd.DataFrame()

    monkeypatch.setattr(weather_mod, "WeatherFeaturesCalculator", _Calculator)

    daily_steps.build_slate_weather_features(slate)

    assert "2026_W03_LA@DEN" in seen["unobserved"]
    assert "2026_W03_LAC@BUF" not in seen["unobserved"]
    assert slate.weather_unknown == {
        "2026_W03_LA@DEN": "WeatherDataError: Open-Meteo timeout"
    }


def test_the_slate_weather_capture_collects_failures_on_the_slate(monkeypatch):
    from scripts import ingest_weather

    schedule = _schedule()
    slate = DailySlate(
        run_date_et=RUN_DATE,
        lock=slate_lock(RUN_DATE),
        schedule=schedule.iloc[1:3].reset_index(drop=True),
    )
    handed: dict[str, object] = {}

    class _Ingester:
        def _load_games_data(self, season, week):
            return schedule

        def _load_venue_data(self):
            return pd.DataFrame()

        def ingest_week_forecast(self, games, venues, *, as_of_utc, failed_games):
            handed["games"] = sorted(games["game_id"])
            failed_games["2026_W03_LA@DEN"] = "WeatherDataError: timeout"
            return pd.DataFrame()

    monkeypatch.setattr(ingest_weather, "WeatherDataIngester", _Ingester)

    daily_steps.ingest_slate_weather(slate)

    assert handed["games"] == ["2026_W03_LA@DEN", "2026_W03_LAC@BUF"]
    assert slate.weather_failures == {"2026_W03_LA@DEN": "WeatherDataError: timeout"}


def test_an_unexpected_capture_error_records_every_slate_game_and_reraises(monkeypatch):
    """A non-WeatherDataError writes nothing, so every slate game carries its reason."""
    from scripts import ingest_weather

    slate = DailySlate(
        run_date_et=RUN_DATE,
        lock=slate_lock(RUN_DATE),
        schedule=_schedule().iloc[1:3].reset_index(drop=True),
    )

    class _Ingester:
        def _load_games_data(self, season, week):
            raise OSError("silver games unreadable")

    monkeypatch.setattr(ingest_weather, "WeatherDataIngester", _Ingester)

    with pytest.raises(OSError, match="unreadable"):
        daily_steps.ingest_slate_weather(slate)

    reason = "the forecast capture failed (OSError: silver games unreadable)"
    assert slate.weather_failures == {
        "2026_W03_LA@DEN": reason,
        "2026_W03_LAC@BUF": reason,
    }


def test_data_qa_expects_the_slate_minus_only_its_recorded_failures(
    monkeypatch, tmp_path
):
    """A recorded failure is not expected; a game missing with no reason still is (2026-09-30).

    Runs the REAL ``step_data_qa`` over a sandbox weather table last written 72 hours ago
    (review CR-03: the stubbed step never reached the freshness check, which halted the very
    night every slate forecast failed with a recorded reason). Only the whole-lake sections
    and the other monitored tables are cut away; the weather checks are the real ones.
    """
    import data.storage as storage_mod
    from data.storage import DuckDBConnection, ParquetManager
    from scripts import data_qa

    root = tmp_path / "lake"
    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(root)))
    monkeypatch.setattr(
        storage_mod, "_db_connection", DuckDBConnection(str(root / "qa.duckdb"))
    )
    written = datetime.now(UTC) - timedelta(hours=72)
    ParquetManager(str(root)).save(
        pd.DataFrame(
            {
                "game_id": ["2026_W03_ATL@GB"],  # an EARLIER slate's row, 72 hours old
                "forecast_time": [written],
                "is_outdoor": [True],
                "created_at": [written],
            }
        ),
        "silver/weather.parquet",
    )
    real_init = data_qa.DataQualityMonitor.__init__

    def _weather_only(self):
        real_init(self)
        self.monitored_tables = {"weather": self.monitored_tables["weather"]}

    monkeypatch.setattr(data_qa.DataQualityMonitor, "__init__", _weather_only)
    for section in (
        "check_data_consistency",
        "check_gold_integrity",
        "check_team_abbreviations",
        "check_duckdb_parquet_consistency",
    ):
        monkeypatch.setattr(
            data_qa.DataQualityMonitor, section, lambda *_a, **_k: {"checks": {}}
        )
    monkeypatch.setattr(data_qa, "get_database_stats", lambda: {})
    monkeypatch.setattr(data_qa, "get_current_nfl_week", lambda: (2026, 3))
    live_skip.reset_excluded_games()  # a dropped game is not expected (review WR-10)

    def _data_qa(failures: dict[str, str]) -> None:
        slate = DailySlate(
            run_date_et=RUN_DATE,
            lock=slate_lock(RUN_DATE),
            schedule=_schedule().iloc[1:3].reset_index(drop=True),
            weather_failures=failures,
        )
        registry = {s.name: s for s in daily_steps.build_daily_step_registry(slate)}
        registry["data_qa"].callable()

    try:
        # Every slate forecast failed WITH a reason: nothing was written tonight, and the
        # night goes on (each game is built with its weather unknown).
        timeout = "WeatherDataError: timeout"
        _data_qa({"2026_W03_LA@DEN": timeout, "2026_W03_LAC@BUF": timeout})

        # One failed with a reason; the other has no row and NO reason: still refused.
        with pytest.raises(RuntimeError, match="Data QA failed"):
            _data_qa({"2026_W03_LA@DEN": timeout})
    finally:
        storage_mod._db_connection.close()


# ---------------------------------------------------------------------------
# 33.2 review C1 WR-02: the no-write dry run makes no paid Odds API request
# ---------------------------------------------------------------------------


def test_the_dry_run_captures_odds_from_a_fixture_and_says_so(monkeypatch, capsys):
    slate = DailySlate(
        run_date_et=RUN_DATE,
        lock=slate_lock(RUN_DATE),
        schedule=_schedule().iloc[1:3].reset_index(drop=True),
    )
    for name in ("ingest_slate_weather", "close_collection"):
        monkeypatch.setattr(daily_steps, name, lambda _slate: None)
    for name in ("step_ingest_snaps", "step_ingest_injuries"):
        monkeypatch.setattr(daily_steps, name, lambda: None)
    calls: list[dict] = []
    monkeypatch.setattr(
        daily_steps,
        "ingest_slate_odds",
        lambda _slate, **kwargs: calls.append(kwargs),
    )

    daily._run_collection_only(slate)

    assert calls == [{"fixture": True}], "the dry run's odds step was not the fixture"
    assert "DRY_RUN_ODDS= fixture" in capsys.readouterr().out


def test_the_fixture_odds_capture_never_opens_the_network(monkeypatch):
    import httpx

    import scripts.ingest_odds as ingest_odds_module
    from utils import DataIngestionError

    slate = DailySlate(
        run_date_et=RUN_DATE,
        lock=slate_lock(RUN_DATE),
        schedule=_schedule().iloc[1:3].reset_index(drop=True),
    )
    monkeypatch.setattr(
        ingest_odds_module, "load_schedule_slice", lambda *_a: _schedule()
    )

    def _no_network(*_a, **_k):
        raise AssertionError("a paid Odds API request was made")

    monkeypatch.setattr(httpx.Client, "get", _no_network)
    boards: list[tuple] = []

    def _board(self, season=None, week=None):
        boards.append((season, week))
        return []

    monkeypatch.setattr(ingest_odds_module.OddsAPIClient, "_generate_mock_odds", _board)

    # An empty fixture board fails loudly (WR-08); the point here is WHICH board was asked.
    with pytest.raises(DataIngestionError):
        daily_steps.ingest_slate_odds(slate, fixture=True)
    assert boards == [(2026, 3)]


# ---------------------------------------------------------------------------
# An all-skipped slate finishes with skips, not a stale-artifact failure (B WR-02)
# ---------------------------------------------------------------------------


def test_an_all_skipped_slate_finishes_with_skips(monkeypatch, tmp_path):
    """Every slate game locked before collection closed: all are dropped and recorded.

    The week has no other slate with rows yet (the Thursday-game case), so the week-level
    checks found no gold row and an empty file and ended the run FAILED as a stale artifact.
    """
    import pipeline.steps as steps_mod
    from pipeline import orchestrator, skip_log
    from pipeline.alert import PipelineAlertManager
    from pipeline.health import PipelineHealthChecker
    from pipeline.staleness import StalenessGate, StalenessResult
    from pipeline.steps import RunStatus

    monkeypatch.setattr(steps_mod, "_predictions_output_dir", lambda: tmp_path)
    monkeypatch.setattr(orchestrator, "LOG_PATH", tmp_path / "pipeline.json")
    monkeypatch.setattr(skip_log, "SKIP_RECORD_PATH", tmp_path / "skips.jsonl")
    monkeypatch.setattr(
        StalenessGate, "run_all_checks", lambda _s: StalenessResult(passed=True)
    )
    monkeypatch.setattr(
        PipelineHealthChecker, "run_preflight", lambda _s: {"status": "healthy"}
    )
    monkeypatch.setattr(
        PipelineHealthChecker, "run_postrun", lambda _s: {"status": "healthy"}
    )
    for name in (
        "alert_pipeline_success",
        "alert_finished_with_skips",
        "alert_degraded_completion",
        "alert_pipeline_failure",
        "alert_staleness_warning",
    ):
        monkeypatch.setattr(PipelineAlertManager, name, lambda *_a, **_k: None)

    def _no_scoring(*_a, **_k):
        raise AssertionError("an all-skipped slate asked the models to score")

    import scripts.generate_current_week_predictions as gen

    monkeypatch.setattr(gen, "build_predictions", _no_scoring)

    slate = _slate(-30)  # its lock is long past: close_collection refuses every game
    wanted = (
        "close_collection",
        "verify_gold_currency",
        "generate_predictions",
        "verify_prediction_currency",
        "export_artifacts",
        "validate_predictions",
    )
    registry = [
        step
        for step in daily_steps.build_daily_step_registry(slate)
        if step.name in wanted
    ]
    live_skip.reset_excluded_games()
    try:
        log = orchestrator.FridayPipeline(steps=registry).run()
    finally:
        live_skip.reset_excluded_games()

    assert log.status == RunStatus.FINISHED_WITH_SKIPS.value
    assert set(log.skipped_games) == set(slate.game_ids)
    written = pd.read_csv(tmp_path / "predictions_2026_week3.csv")
    assert written.empty


# ---------------------------------------------------------------------------
# An odds failure never stops predictions (batch 1a follow-up, owner requirement)
# ---------------------------------------------------------------------------


def test_an_odds_failure_still_predicts_every_slate_game_and_names_them(
    monkeypatch, predict_env, capsys
):
    """Zero matched odds (a DataIngestionError) used to FAIL the run: the step was critical.

    The models use no market input, so every slate game is predicted with its market side
    blank, each game without a line is named, and the run completes.
    """
    from pipeline import orchestrator, skip_log
    from pipeline.alert import PipelineAlertManager
    from pipeline.health import PipelineHealthChecker
    from pipeline.staleness import StalenessGate, StalenessResult
    from utils import DataIngestionError

    out_dir, _ = predict_env
    monkeypatch.setattr(orchestrator, "LOG_PATH", out_dir / "pipeline.json")
    monkeypatch.setattr(skip_log, "SKIP_RECORD_PATH", out_dir / "skips.jsonl")
    monkeypatch.setattr(
        StalenessGate, "run_all_checks", lambda _s: StalenessResult(passed=True)
    )
    monkeypatch.setattr(
        PipelineHealthChecker, "run_preflight", lambda _s: {"status": "healthy"}
    )
    monkeypatch.setattr(
        PipelineHealthChecker, "run_postrun", lambda _s: {"status": "healthy"}
    )
    for name in (
        "alert_pipeline_success",
        "alert_finished_with_skips",
        "alert_degraded_completion",
        "alert_pipeline_failure",
        "alert_staleness_warning",
    ):
        monkeypatch.setattr(PipelineAlertManager, name, lambda *_a, **_k: None)

    def _no_odds(_slate, **_k):
        raise DataIngestionError("no odds row was captured for any scheduled game")

    monkeypatch.setattr(daily_steps, "ingest_slate_odds", _no_odds)
    import scripts.generate_current_week_predictions as gen

    monkeypatch.setattr(
        gen,
        "load_market_data",
        lambda ids: pd.DataFrame(columns=["game_id", "spread", "total"]),
    )

    slate = _slate(30)
    registry = [
        step
        for step in daily_steps.build_daily_step_registry(slate)
        if step.name in ("ingest_odds", "close_collection", "generate_predictions")
    ]
    assert next(s for s in registry if s.name == "ingest_odds").critical is False
    log = orchestrator.FridayPipeline(steps=registry).run()

    assert log.status == "degraded"
    written = pd.read_csv(out_dir / "predictions_2026_week3.csv")
    assert set(written["game_id"]) == set(slate.game_ids)
    assert set(slate.odds_missing) == set(slate.game_ids)
    assert all("DataIngestionError" in r for r in slate.odds_missing.values())


# ---------------------------------------------------------------------------
# The dry run's WRITES= line is a measurement, not a constant (C1 IN-01)
# ---------------------------------------------------------------------------


def test_the_dry_run_counts_a_write_that_bypassed_the_sink(monkeypatch, tmp_path):
    from conf.settings import get_settings

    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    data_root = tmp_path / "data"
    (data_root / "silver").mkdir(parents=True)
    monkeypatch.setattr(get_settings().config.data, "root_path", str(data_root))

    sink = daily._MeasuredRecordingSink()
    assert sink.measured_writes() == 0

    (data_root / "__pycache__").mkdir()
    (data_root / "__pycache__" / "storage.pyc").write_bytes(b"import cache")
    assert sink.measured_writes() == 0, "a byte-code cache is not a data write"

    (data_root / "silver" / "games.parquet").write_bytes(b"a side-door write")
    (tmp_path / "config" / "skip_records.jsonl").write_text("{}\n", encoding="utf-8")
    assert sink.measured_writes() == 2


def test_a_repeat_skip_on_the_same_day_is_still_reported(monkeypatch, tmp_path, capsys):
    """33.2 review B IN-02: the re-run appends no record (idempotent), yet skipped the game."""
    from pipeline import skip_log

    path = tmp_path / "skips.jsonl"
    monkeypatch.setattr(skip_log, "SKIP_RECORD_PATH", path)
    skip_log.append_skip_record(
        {
            "run_id": "2026-09-26T17:00:01-04:00",  # the FIRST run of the day
            "run_date_et": "2026-09-26",
            "game_id": "2026_W03_LAC@BUF",
            "source": "decision_instant",
            "information_time": None,
            "lock": None,
            "reason": "post_lock",
            "recorded_at": "2026-09-26T21:00:02+00:00",
        }
    )

    daily._report_skips(["2026_W03_LAC@BUF"], "2026-09-26T17:40:00-04:00")

    assert "SKIPPED 2026_W03_LAC@BUF: post_lock (source decision_instant)" in (
        capsys.readouterr().out
    )


def test_a_slate_outside_the_elo_season_is_refused_by_name(monkeypatch):
    """33.2 review B IN-05: was pandas' opaque "No objects to concatenate"."""
    from pipeline.steps import persist_current_season_elo
    from scripts import build_elo

    class _OtherSeasonElo:
        pending_snapshot_rows = 0

        def update_current_season(self):
            return build_elo.LiveSeasonUpdate(
                season=2025, snapshots=pd.DataFrame({"game_id": []})
            )

        def load_games_data(self, seasons=None):
            return pd.DataFrame({"game_id": ["2025_W22_SEA@NE"], "week": [22]})

    monkeypatch.setattr(build_elo, "EloBuilder", _OtherSeasonElo)
    with pytest.raises(ValueError, match="2026_W03_LAC@BUF"):
        persist_current_season_elo(frozenset({"2026_W03_LAC@BUF"}))


# ---------------------------------------------------------------------------
# The ledger behind the cutover switch (Plan 34-15): recommend, settle, triggers, sync, run id
# ---------------------------------------------------------------------------


def _cut_over(monkeypatch, on: bool = True) -> None:
    """Set the committed switch for this test only (the constant stays OFF in the tree)."""
    from forward_ledger import cutover

    monkeypatch.setattr(cutover, "FORWARD_ROWS_GO_TO_LEDGER", on)


def _recommend_env(monkeypatch, tmp_path, slate: DailySlate) -> str:
    """Route ``recommend_slate``'s reads into fixtures; return the week game outside the slate."""
    import pipeline.steps as steps_mod

    outside = "2026_W03_PHI@CHI"
    week = pd.concat(
        [slate.schedule, pd.DataFrame({"game_id": [outside]})], ignore_index=True
    )
    monkeypatch.setattr(steps_mod, "_week_schedule", lambda _s, _w: week)
    monkeypatch.setattr(steps_mod, "_bet_list_output_dir", lambda: tmp_path)
    live_skip.reset_excluded_games()
    return outside


def test_recommend_switch_off_unchanged(monkeypatch, tmp_path):
    from backtest import weekly_bet_list
    from forward_ledger import runner

    slate = _slate(30)
    outside = _recommend_env(monkeypatch, tmp_path, slate)
    calls: list[dict] = []
    monkeypatch.setattr(
        weekly_bet_list,
        "generate_weekly_bet_list",
        lambda **kwargs: calls.append(kwargs),
    )

    def _no_ledger(*_a, **_k):
        raise AssertionError("the ledger was written with the switch off")

    monkeypatch.setattr(runner, "record_forward_slate", _no_ledger)

    daily_steps.recommend_slate(slate)

    (call,) = calls
    assert set(call) == {
        "season",
        "week",
        "output_dir",
        "now",
        "excluded_game_ids",
        "publish_by",
    }
    assert (call["season"], call["week"]) == (2026, 3)
    assert call["output_dir"] == tmp_path
    assert call["excluded_game_ids"] == frozenset({outside})
    assert call["publish_by"] == slate.lock


def test_recommend_switch_on_uses_runner(monkeypatch, tmp_path):
    from backtest import weekly_bet_list
    from backtest.weekly_bet_list import PublishDeadlinePassedError
    from forward_ledger import runner

    _cut_over(monkeypatch)
    slate = _slate(30)
    outside = _recommend_env(monkeypatch, tmp_path, slate)

    def _no_bet_list(**_k):
        raise AssertionError("the old writer ran after the cutover")

    monkeypatch.setattr(weekly_bet_list, "generate_weekly_bet_list", _no_bet_list)
    calls: list[tuple] = []
    monkeypatch.setattr(
        runner,
        "record_forward_slate",
        lambda slate_arg, **kwargs: calls.append((slate_arg, kwargs)),
    )

    daily_steps.recommend_slate(slate)

    ((slate_arg, kwargs),) = calls
    assert slate_arg is slate
    assert set(kwargs) == {"decided_at", "excluded_game_ids", "publish_by"}
    assert kwargs["decided_at"].tzinfo is not None
    assert kwargs["excluded_game_ids"] == frozenset({outside})
    assert kwargs["publish_by"] == slate.lock

    def _late(*_a, **_k):
        raise PublishDeadlinePassedError("ready after the deadline")

    monkeypatch.setattr(runner, "record_forward_slate", _late)
    with pytest.raises(live_skip.GamesLockPassedError) as refused:
        daily_steps.recommend_slate(slate)
    assert sorted(refused.value.details["game_ids"]) == sorted(slate.game_ids)
    assert "2026_W03_LAC@BUF" in str(refused.value)


def _no_games_env(monkeypatch, tmp_path) -> list[str]:
    """A run date whose tomorrow has no game; returns the ordered record of what happened."""
    from forward_ledger import runner

    order: list[str] = []
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", tmp_path / "daily.jsonl")
    monkeypatch.setattr(daily, "_refresh_schedule", lambda *_a, **_k: 2026)
    schedule = _schedule()
    schedule = schedule.loc[schedule["game_id"] == "2026_W03_PHI@CHI"]
    monkeypatch.setattr(
        "data.storage.load_dataframe", lambda *_a, **_k: schedule.copy()
    )
    real_record = daily._record_no_prediction

    def _recording(run_date_et, outcome, game_ids, **kwargs):
        order.append(outcome)
        real_record(run_date_et, outcome, game_ids, **kwargs)

    monkeypatch.setattr(daily, "_record_no_prediction", _recording)

    def _settle(**_kwargs):
        order.append("settle")
        return runner.SettleOutcome(False, 0, 0, 0, 0)

    monkeypatch.setattr(runner, "settle_ledger", _settle)
    monkeypatch.setattr(
        runner, "sync_after_run", lambda **_k: order.append("sync") or None
    )
    monkeypatch.setattr(daily, "_write_census", dict)
    return order


def _start() -> datetime:
    return slate_lock(RUN_DATE).astimezone(UTC) - timedelta(hours=8)


def test_settle_runs_on_no_game_day_when_cut_over(monkeypatch, tmp_path, capsys):
    _cut_over(monkeypatch)
    order = _no_games_env(monkeypatch, tmp_path)

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0

    assert order[:2] == ["settle", "no_games"]
    assert "LEDGER_SETTLE= unchanged" in capsys.readouterr().out

    _cut_over(monkeypatch, on=False)
    order.clear()
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0
    assert order == ["no_games"], "the settle pass ran with the switch off"


def test_settle_failure_does_not_stop_the_run(monkeypatch, tmp_path, capsys):
    from forward_ledger import runner

    _cut_over(monkeypatch)
    order = _no_games_env(monkeypatch, tmp_path)

    def _broken(**_kwargs):
        raise RuntimeError("silver games unreadable")

    monkeypatch.setattr(runner, "settle_ledger", _broken)

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0

    out = capsys.readouterr().out
    assert "LEDGER_SETTLE= failed" in out
    assert "silver games unreadable" in out
    assert "no_games" in order, "the run stopped before slate selection"


def test_cache_rebuilt_after_changing_settle_on_no_game_day(
    monkeypatch, tmp_path, capsys
):
    import pipeline.steps as steps_mod
    from forward_ledger import runner

    _cut_over(monkeypatch)
    order = _no_games_env(monkeypatch, tmp_path)
    monkeypatch.setattr(
        runner, "settle_ledger", lambda **_k: runner.SettleOutcome(True, 1, 0, 1, 0)
    )
    monkeypatch.setattr(
        steps_mod, "step_populate_web_cache", lambda: order.append("cache")
    )

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0
    assert order.count("cache") == 1
    assert order.index("cache") < order.index("no_games")

    def _cache_down():
        raise OSError("the cache file is locked")

    monkeypatch.setattr(steps_mod, "step_populate_web_cache", _cache_down)
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0
    assert "the cache file is locked" in capsys.readouterr().out


def test_triggers_regenerated_before_no_games_return(
    monkeypatch, tmp_path, capsys, trigger_calls
):
    from forward_ledger import closing_schedule

    order = _no_games_env(monkeypatch, tmp_path)
    start = _start()

    def _register(_games, now, **_kwargs):
        order.append("triggers")
        trigger_calls.append(now)
        return closing_schedule.TriggerRegistration("skipped", "probe", (), False)

    monkeypatch.setattr(closing_schedule, "register_closing_triggers", _register)

    assert daily.run_daily(RUN_DATE, start=start, dry_run=False) == 0
    assert trigger_calls == [start]
    assert order.index("triggers") < order.index("no_games")
    assert "CLOSING_TRIGGERS= skipped probe" in capsys.readouterr().out

    trigger_calls.clear()
    assert daily.run_daily(RUN_DATE, start=start, dry_run=True) == 0
    assert trigger_calls == [], "the dry run regenerated the triggers"

    def _broken(*_a, **_k):
        raise RuntimeError("schtasks is not on PATH")

    monkeypatch.setattr(closing_schedule, "register_closing_triggers", _broken)
    assert daily.run_daily(RUN_DATE, start=start, dry_run=False) == 0
    assert "CLOSING_TRIGGERS= failed" in capsys.readouterr().out


def test_sync_at_end_when_cut_over(monkeypatch, tmp_path):
    import pipeline.steps as steps_mod
    from pipeline import orchestrator

    _cut_over(monkeypatch)
    order = _no_games_env(monkeypatch, tmp_path)

    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0
    assert order.count("sync") == 1
    assert order[-1] == "sync"

    # A run that raises still syncs, and the raise is unchanged.
    order.clear()
    monkeypatch.setattr(
        "data.storage.load_dataframe", lambda *_a, **_k: _schedule().copy()
    )
    monkeypatch.setattr(daily, "_require_slate_is_current_week", lambda *_a: None)
    monkeypatch.setattr(daily, "build_daily_step_registry", lambda _slate: [])
    monkeypatch.setattr(steps_mod, "_predictions_output_dir", lambda: tmp_path)

    class _Refused:
        def __init__(self, **_k):
            pass

        def run(self):
            raise RuntimeError("Pre-flight staleness checks failed")

    monkeypatch.setattr(orchestrator, "FridayPipeline", _Refused)
    with pytest.raises(RuntimeError, match="staleness"):
        daily.run_daily(RUN_DATE, start=_start(), dry_run=False)
    assert order.count("sync") == 1

    order.clear()
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=True) == 0
    assert "sync" not in order, "a dry run synced the ledger"

    _cut_over(monkeypatch, on=False)
    order.clear()
    with pytest.raises(RuntimeError, match="staleness"):
        daily.run_daily(RUN_DATE, start=_start(), dry_run=False)
    assert "sync" not in order, "the ledger synced with the switch off"


def test_one_run_id_per_daily_run(monkeypatch, tmp_path, capsys):
    from types import SimpleNamespace

    from forward_ledger import run_log, runner
    from forward_ledger.sync import SyncOutcome
    from pipeline import orchestrator

    _cut_over(monkeypatch)
    slate = _slate(30)
    _recommend_env(monkeypatch, tmp_path, slate)
    run_date = slate.run_date_et
    start = slate.lock.astimezone(UTC) - timedelta(hours=8)
    monkeypatch.setattr(daily, "DAILY_RUN_RECORDS", tmp_path / "daily.jsonl")
    monkeypatch.setattr(daily, "_refresh_schedule", lambda *_a, **_k: 2026)
    monkeypatch.setattr(daily, "_refuse_an_unrunnable_date", lambda *_a, **_k: None)
    monkeypatch.setattr(daily, "_require_slate_is_current_week", lambda *_a: None)
    monkeypatch.setattr(daily, "_report_skips", lambda *_a: None)
    monkeypatch.setattr(
        "data.storage.load_dataframe", lambda *_a, **_k: slate.schedule.copy()
    )

    def _record(_slate, **_kwargs):
        run_log.record_event("append", appended_rows=2)

    def _settle(**_kwargs):
        run_log.record_event("settle", graded=0)
        return runner.SettleOutcome(False, 0, 0, 0, 0)

    def _publish(*, ledger_dir, repo_dir, log):
        log("anchor_push", ok=True, head_hash="h", entry_count=2)
        return SyncOutcome(False, True, None, False, False, None, None)

    monkeypatch.setattr(runner, "record_forward_slate", _record)
    monkeypatch.setattr(runner, "settle_ledger", _settle)
    monkeypatch.setattr(runner, "publish_ledger_state", _publish)

    class _RecommendOnly:
        def __init__(self, *, steps, deadline):
            self.steps = steps

        def run(self):
            step = next(s for s in self.steps if s.name == "generate_recommendations")
            step.callable()
            return SimpleNamespace(
                status="success", skipped_games=[], start_time=start.isoformat()
            )

    monkeypatch.setattr(orchestrator, "FridayPipeline", _RecommendOnly)

    def _one_run() -> tuple[str, list[dict]]:
        log_path = run_log.LEDGER_RUN_LOG
        before = len(run_log.read_events(log_path))
        assert daily.run_daily(run_date, start=start, dry_run=False) == 0
        printed = [
            line.split("= ", 1)[1]
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("LEDGER_RUN_ID= ")
        ]
        assert len(printed) == 1, printed
        return printed[0], run_log.read_events(log_path)[before:]

    first_id, first_events = _one_run()
    names = {event["event"] for event in first_events}
    assert {"append", "settle", "anchor_push"} <= names
    assert {event["run_id"] for event in first_events} == {first_id}

    second_id, second_events = _one_run()
    assert second_id != first_id
    assert {event["run_id"] for event in second_events} == {second_id}

    _cut_over(monkeypatch, on=False)
    _no_games_env(monkeypatch, tmp_path)
    assert daily.run_daily(RUN_DATE, start=_start(), dry_run=False) == 0
    assert "LEDGER_RUN_ID=" not in capsys.readouterr().out


def test_generate_refuses_forward_mode_after_cutover(monkeypatch):
    from backtest import weekly_bet_list

    class _Reached(Exception):
        pass

    def _selection_started(*_a, **_k):
        raise _Reached

    monkeypatch.setattr(weekly_bet_list, "load_frozen_chain_fit", _selection_started)

    _cut_over(monkeypatch)
    with pytest.raises(
        getattr(weekly_bet_list, "ForwardRowsMovedToLedgerError", _Reached),
        match="daily",
    ):
        weekly_bet_list.generate_weekly_bet_list(2026, 6, run_mode="forward")
    with pytest.raises(_Reached):
        weekly_bet_list.generate_weekly_bet_list(2025, 6, run_mode="replay")

    _cut_over(monkeypatch, on=False)
    with pytest.raises(_Reached):
        weekly_bet_list.generate_weekly_bet_list(2026, 6, run_mode="forward")
