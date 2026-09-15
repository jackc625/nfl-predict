"""The live weather-features step builds the season TO DATE and refuses what matters.

THE DEFECT THIS GUARDS
----------------------
``step_build_weather_features`` handed ``build_weather_features`` EVERY scheduled game in silver.
The builder refuses any game with no silver weather row -- correctly: the old answer was a 65 F
dome default that reached 6,485 of 6,499 gold rows. But during a live season that can never
pass. Plan 33-18's live acceptance run, attempt 2 on 2026-09-15, halted there on
``2026_W01_NE@SEA``: week 1 was never ingested (0 of 16 covered) and weeks 3-18 lie beyond any
forecast. And the step was registered ``critical=False``, so the real orchestrator would have
DEGRADED past it and rebuilt gold from a weather-features table ending in 2025 -- no weather at
all for the week being predicted.

THE OWNER'S RULING (W1, 2026-09-15) AND WHAT EACH CLASS PINS
------------------------------------------------------------
(a) The step builds completed seasons plus the current season THROUGH THE CURRENT WEEK, and
    never demands a record for a future week.
(b) A PLAYED current-season game with no weather record gets an explicit NO-OBSERVATION row:
    ``weather_coverage`` 0.0 and every weather value null. ``weather_affects_game`` is null too,
    because 0.0 is the value that means "indoor" -- asserting a dome for a game nobody observed
    is the exact defect the coverage flag exists to remove.
(c) The REFUSAL SURVIVES where it matters: a game in the week being predicted with no record, or
    a game in a completed season with no record, still stops the step.
(d) The step is registered ``critical=True``, so its refusal stops the run instead of feeding gold
    stale weather. The orchestrator half is pinned in
    ``tests/integration/test_weather_features_refusal_stops_pipeline.py``.

REAL SCHEMA, NOTHING INVENTED. The silver ``games`` and ``weather`` frames below carry exactly the
production columns, measured on 2026-09-15 -- the lesson of commit 28e73d6, where a fixture that
gave ``weather`` season/week columns it does not have made two broken checks look sound.

Sandboxed: the parquet manager and DuckDB connection point at ``tmp_path``, and the current week
is injected through ``pipeline.steps._resolve_current_week``. ASCII only, no emoji.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import data.storage as storage_mod
from data.storage import DuckDBConnection, ParquetManager
from features.weather import WeatherObservationError
from pipeline import steps

CURRENT = (2026, 2)
WRITTEN = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)

# game_id -> (season, week). A completed season, a played current week, the week being
# predicted, and a future week.
SCHEDULE: dict[str, tuple[int, int]] = {
    "2025_W01_DAL@PHI": (2025, 1),
    "2025_W02_KC@BUF": (2025, 2),
    "2026_W01_NE@SEA": (2026, 1),
    "2026_W01_DET@GB": (2026, 1),
    "2026_W02_CAR@ATL": (2026, 2),
    "2026_W02_JAX@DEN": (2026, 2),
    "2026_W03_MIA@NYJ": (2026, 3),
}
PAST_SEASON = {"2025_W01_DAL@PHI", "2025_W02_KC@BUF"}
PLAYED_CURRENT = {"2026_W01_NE@SEA", "2026_W01_DET@GB"}
PREDICTED_WEEK = {"2026_W02_CAR@ATL", "2026_W02_JAX@DEN"}
FUTURE = {"2026_W03_MIA@NYJ"}

# Numeric columns that carry a MEASUREMENT or a value derived from one. On a no-observation row
# every one of them must be null.
_NO_OBSERVATION_NULL_COLUMNS: tuple[str, ...] = (
    "temp_f",
    "wind_mph",
    "precip_prob",
    "weather_severity_score",
    "raw_temp_f",
    "raw_wind_mph",
)


def _games_row(game_id: str) -> dict:
    season, week = SCHEDULE[game_id]
    away, home = game_id.split("_", 2)[2].split("@")
    played = (season, week) < CURRENT
    return {
        "game_id": game_id,
        "season": season,
        "week": week,
        "kickoff_et": pd.Timestamp(
            f"{season}-09-{10 + 7 * (week - 1):02d} 17:00", tz="UTC"
        ),
        "home_team": home,
        "away_team": away,
        "venue": f"{home} Stadium",
        "venue_roof": "outdoor",
        "home_score": 24.0 if played else float("nan"),
        "away_score": 17.0 if played else float("nan"),
        "result": 7.0 if played else float("nan"),
        "game_type": "REG",
        "season_type": "Regular",
        "neutral_site": False,
        "created_at": WRITTEN,
        "stadium_id": f"{home}00",
    }


def _weather_row(game_id: str) -> dict:
    season, week = SCHEDULE[game_id]
    kickoff = pd.Timestamp(f"{season}-09-{10 + 7 * (week - 1):02d} 17:00", tz="UTC")
    return {
        "game_id": game_id,
        "forecast_time": WRITTEN,
        "game_time": kickoff,
        "temp_f": 74.2,
        "temp_c": 23.4,
        "wind_mph": 6.3,
        "wind_direction": 193.0,
        "humidity_pct": 73.0,
        "precip_prob": float("nan"),
        "precip_mm": 0.3,
        "condition": None,
        "condition_code": 51.0,
        "visibility_km": None,
        "dew_point_f": 65.1,
        "apparent_temp_f": 77.1,
        "snowfall_cm": 0.0,
        "wind_gusts_mph": 20.4,
        "cloud_cover_pct": 100.0,
        "is_outdoor": True,
        "is_cold": False,
        "is_windy": False,
        "is_precipitation": True,
        "created_at": WRITTEN,
        "weather_source": "archive" if season < CURRENT[0] else "forecast",
        "weather_coverage": True,
    }


@pytest.fixture
def lake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    root = tmp_path / "lake"
    monkeypatch.setattr(storage_mod, "_parquet_manager", ParquetManager(str(root)))
    monkeypatch.setattr(
        storage_mod, "_db_connection", DuckDBConnection(str(root / "sandbox.duckdb"))
    )
    monkeypatch.setattr(steps, "_resolve_current_week", lambda: CURRENT)

    def seed(weather_ids: set[str]) -> None:
        manager = ParquetManager(str(root))
        manager.save(
            pd.DataFrame([_games_row(g) for g in SCHEDULE]), "silver/games.parquet"
        )
        manager.save(
            pd.DataFrame([_weather_row(g) for g in sorted(weather_ids)]),
            "silver/weather.parquet",
        )

    yield root, seed
    storage_mod._db_connection.close()


def _built(root: Path) -> pd.DataFrame:
    return pd.read_parquet(root / "silver" / "weather_features.parquet").set_index(
        "game_id"
    )


# The live state attempt 2 met: history and the predicted week covered, week 1 never ingested,
# the future beyond any forecast.
LIVE_STATE = PAST_SEASON | PREDICTED_WEEK


def _build_expecting_success(lake) -> pd.DataFrame:
    """Seed the live state, run the step, and return what it wrote.

    A refusal here is reported as a TEST FAILURE naming the refusal, never as a fixture error:
    refusing the live state is exactly the behaviour under test, so it must read as that.
    """
    root, seed = lake
    seed(LIVE_STATE)
    try:
        steps.step_build_weather_features()
    except WeatherObservationError as refusal:
        pytest.fail(f"the step refused the live season-to-date state: {refusal}")
    return _built(root)


class TestTheSeasonIsBuiltToDate:
    """(a)"""

    def test_the_step_does_not_refuse_the_live_state(self, lake):
        assert len(_build_expecting_success(lake)) > 0

    def test_it_builds_past_seasons_and_the_current_season_through_this_week(
        self, lake
    ):
        built = _build_expecting_success(lake)
        assert set(built.index) == PAST_SEASON | PLAYED_CURRENT | PREDICTED_WEEK

    def test_it_never_builds_a_future_week(self, lake):
        assert not set(_build_expecting_success(lake).index) & FUTURE


class TestAPlayedGameWithNoRecordIsExplicitlyUnobserved:
    """(b)"""

    @pytest.fixture
    def built(self, lake):
        """Returns a BUILDER, not a frame, so the step runs inside each test body."""
        return lambda: _build_expecting_success(lake)

    @pytest.mark.parametrize("game_id", sorted(PLAYED_CURRENT))
    def test_the_coverage_flag_reads_uncovered(self, built, game_id):
        built = built()
        assert built.loc[game_id, "weather_coverage"] == 0.0

    @pytest.mark.parametrize("game_id", sorted(PLAYED_CURRENT))
    def test_no_measurement_is_filled(self, built, game_id):
        built = built()
        filled = {
            column: built.loc[game_id, column]
            for column in _NO_OBSERVATION_NULL_COLUMNS
            if not math.isnan(built.loc[game_id, column])
        }
        assert not filled, (
            f"{game_id} has no weather record but carries values {filled}; a no-observation "
            "row must carry nulls, never a default"
        )

    @pytest.mark.parametrize("game_id", sorted(PLAYED_CURRENT))
    def test_applicability_is_not_asserted_as_indoor(self, built, game_id):
        built = built()
        assert math.isnan(built.loc[game_id, "weather_affects_game"]), (
            f"{game_id}: weather_affects_game is {built.loc[game_id, 'weather_affects_game']!r}. "
            "0.0 means INDOOR; a game nobody observed must not be recorded as a dome."
        )

    @pytest.mark.parametrize("game_id", sorted(PREDICTED_WEEK | PAST_SEASON))
    def test_a_game_with_a_record_stays_covered(self, built, game_id):
        built = built()
        assert built.loc[game_id, "weather_coverage"] == 1.0
        assert built.loc[game_id, "temp_f"] == pytest.approx(74.2)


class TestTheRefusalSurvivesWhereItMatters:
    """(c)"""

    def test_a_predicted_week_game_with_no_record_is_refused(self, lake):
        _, seed = lake
        seed(LIVE_STATE - {"2026_W02_CAR@ATL"})
        with pytest.raises(WeatherObservationError, match="2026_W02_CAR@ATL"):
            steps.step_build_weather_features()

    def test_a_completed_season_game_with_no_record_is_refused(self, lake):
        _, seed = lake
        seed(LIVE_STATE - {"2025_W02_KC@BUF"})
        with pytest.raises(WeatherObservationError, match="2025_W02_KC@BUF"):
            steps.step_build_weather_features()

    def test_a_refused_build_writes_nothing(self, lake):
        root, seed = lake
        seed(LIVE_STATE - {"2026_W02_CAR@ATL"})
        with pytest.raises(WeatherObservationError):
            steps.step_build_weather_features()
        assert not (root / "silver" / "weather_features.parquet").exists()


class TestTheStepIsCritical:
    """(d)"""

    def test_build_weather_features_is_registered_critical(self):
        step = next(
            s for s in steps.build_step_registry() if s.name == "build_weather_features"
        )
        assert step.critical is True, (
            "build_weather_features is registered non-critical, so its refusal DEGRADES the "
            "run and gold is rebuilt with no weather for the week being predicted"
        )
