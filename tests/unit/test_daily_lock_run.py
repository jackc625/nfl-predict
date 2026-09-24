"""The daily lock-time run's own rules (Plan 33.2-27).

* A run that starts at or after today's 18:00 ET lock refuses BEFORE any request.
* A day with no games tomorrow is a recorded no-op.
* The slate is exactly tomorrow's games, selected by their shared lock.
* Every slate game is predicted, stamped with its three instants, and merged into the week's file
  without disturbing other games' rows; a game computed after its lock is refused, never
  back-dated; a game that silently produced no prediction is a named refusal.
* Elo gains a flagged provisional row for exactly the slate's games.

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


def test_elo_gains_provisional_rows_for_exactly_the_slate(monkeypatch):
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
    assert provisional == {"2026_W03_LAC@BUF", "2026_W03_LA@DEN"}
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
