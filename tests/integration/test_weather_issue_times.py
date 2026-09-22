"""Every past forecast was known by its lock; every forecast-less row carries no time (R6).

Plan 33.2-12 Task 1 (SPEC R6, D33.2-02), p332_ rung 4. Three things are asserted:

1. PRODUCTION SILVER, BOTH DIRECTIONS. Every 2002-2025 row that CARRIES a forecast
   (``weather_coverage`` and ``is_outdoor`` both true) has a ``forecast_issue_time`` at or
   before its game's lock; every forecast-less row -- a dome or an absence -- has a NULL one.
   Both directions, because a blanket "every row is stamped" would force a fabricated issue
   time onto the games abroad this same plan sends down the NULL-plus-flag path. No row is of
   the observation kind. Non-vacuity: the forecast-carrying count is thousands, not zero.
2. THE ONE FENCE (``features.weather.select_weather_row``): a row is admitted only when its
   forecast time is at or before min(the game's lock, the build instant) -- a bulletin issued
   after the lock is not admitted, and neither is one issued after a PRE-lock build instant.
3. THE PROVENANCE SUPPLIER (``WeatherFeaturesCalculator.information_times``): a forecast
   game is ``per_row`` with its selected bulletin's issue time; a dome and an absence are
   ``no_information`` with a NULL time and values satisfying the signature; a mixed frame
   passes the real ``InformationTimeGate.check`` in ONE call.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

import utils.game_lock as lock_rule
from features.protocol import InformationTimeProvider
from features.provenance import InformationTimeGate, build_lock_frame
from features.weather import (
    OBSERVATION_WEATHER_SOURCES,
    WEATHER_NO_INFORMATION_SIGNATURE,
    WeatherFeaturesCalculator,
    select_weather_row,
)
from scripts.build_features import SUPPLIER_ATTRIBUTES
from scripts.mos_decode import relative_humidity_pct

REPO_ROOT = Path(__file__).resolve().parents[2]
SILVER = REPO_ROOT / "data" / "silver"
HISTORY = (2002, 2025)

#: The seven 2025 games played abroad (Plan 33.2-09 corrected their venue; their silver
#: weather used to describe the US stadium the feed named).
GAMES_ABROAD_2025: tuple[str, ...] = (
    "2025_W01_KC@LAC",
    "2025_W04_MIN@PIT",
    "2025_W05_MIN@CLE",
    "2025_W06_DEN@NYJ",
    "2025_W07_LA@JAX",
    "2025_W10_ATL@IND",
    "2025_W11_WAS@MIA",
)


def _history(frame: pd.DataFrame) -> pd.DataFrame:
    seasons = frame["game_id"].str[:4].astype(int)
    return frame[seasons.between(*HISTORY)]


@pytest.fixture(scope="module")
def silver_weather() -> pd.DataFrame:
    path = SILVER / "weather.parquet"
    if not path.exists():
        pytest.skip("silver weather is not built on this checkout")
    return _history(pd.read_parquet(path)).set_index("game_id")


@pytest.fixture(scope="module")
def silver_games() -> pd.DataFrame:
    return pd.read_parquet(SILVER / "games.parquet").set_index("game_id")


def _forecast_mask(frame: pd.DataFrame) -> pd.Series:
    return frame["weather_coverage"].astype(bool) & frame["is_outdoor"].astype(bool)


class TestProductionSilverCarriesHonestIssueTimes:
    def test_the_forecast_carrying_set_is_thousands_not_zero(self, silver_weather):
        assert _forecast_mask(silver_weather).sum() > 4000

    def test_every_forecast_row_was_issued_at_or_before_its_lock(
        self, silver_weather, silver_games
    ):
        carrying = silver_weather[_forecast_mask(silver_weather)]
        assert carrying["forecast_issue_time"].notna().all(), (
            "a forecast row is unstamped"
        )
        late = [
            game_id
            for game_id, issued in carrying["forecast_issue_time"].items()
            if not lock_rule.is_admissible(
                issued,
                lock_rule.game_lock(
                    silver_games.loc[game_id, "kickoff_et"], game_id=game_id
                ),
            )
        ]
        assert late == [], f"issued after the lock: {late[:5]}"

    def test_every_forecast_less_row_carries_no_issue_time(self, silver_weather):
        forecast_less = silver_weather[~_forecast_mask(silver_weather)]
        assert len(forecast_less) > 0
        stamped = forecast_less.index[forecast_less["forecast_issue_time"].notna()]
        assert list(stamped) == [], (
            "an issue time on a dome or an absence is provenance for a forecast that "
            f"does not exist: {list(stamped)[:5]}"
        )

    def test_no_row_is_an_observation(self, silver_weather):
        observed = silver_weather["weather_source"].isin(OBSERVATION_WEATHER_SOURCES)
        assert int(observed.sum()) == 0

    def test_no_millimetre_amount_is_invented(self, silver_weather):
        assert int(silver_weather["precip_mm"].notna().sum()) == 0

    def test_every_covered_game_has_a_real_probability(self, silver_weather):
        carrying = silver_weather[_forecast_mask(silver_weather)]
        assert carrying["precip_prob"].between(0.0, 1.0).all()

    def test_humidity_is_the_magnus_identity_of_temperature_and_dew_point(
        self, silver_weather
    ):
        carrying = silver_weather[_forecast_mask(silver_weather)]
        derived = [
            relative_humidity_pct(t, d)
            for t, d in zip(carrying["temp_f"], carrying["dew_point_f"], strict=True)
        ]
        assert list(carrying["humidity_pct"]) == derived

    def test_the_games_abroad_carry_no_us_station_reading(self, silver_weather):
        rows = silver_weather.loc[list(GAMES_ABROAD_2025)]
        assert rows["is_outdoor"].all(), "Sao Paulo and Berlin were marked indoor"
        assert not rows["weather_coverage"].any()
        assert rows["mos_station"].isna().all()
        assert rows["temp_f"].isna().all()

    def test_the_post_lock_game_uses_san_diegos_station(self, silver_weather):
        assert silver_weather.loc["2003_W08_MIA@LAC", "mos_station"] == "KSAN"

    def test_the_game_with_no_archived_bulletin_is_an_honest_absence(
        self, silver_weather
    ):
        """NULL plus the coverage flag, never a dome and never a stand-in run (step 4b).

        Rung 4 recorded it as a dome because the feed says NRG Stadium's roof was CLOSED.
        That state was decided near kickoff, so it is not known at the lock (owner ruling
        2026-09-21, p332_ extra step 4b): weather applies to a retractable stadium, and its
        KHOU 12 UTC run is a confirmed archive gap (Plan 33.2-11).
        """
        row = silver_weather.loc["2019_W18_BUF@HOU"]
        assert bool(row["is_outdoor"]) and not bool(row["weather_coverage"])
        assert pd.isna(row["temp_f"]) and pd.isna(row["forecast_issue_time"])


# ---------------------------------------------------------------------------
# The one fence and the provenance supplier, on planted frames.
# ---------------------------------------------------------------------------

KICKOFF = pd.Timestamp("2016-09-11 17:00", tz="UTC")  # Sunday 13:00 ET
LOCK = pd.Timestamp("2016-09-10 22:00", tz="UTC")  # Saturday 18:00 ET
BULLETIN = pd.Timestamp("2016-09-10 12:00", tz="UTC")
AFTER_LOCK = pd.Timestamp("2016-09-10 23:00", tz="UTC")
LATE_BUILD = datetime(2016, 9, 20, tzinfo=UTC)


def _forecast_row(game_id: str, issued: pd.Timestamp, temp: float = 50.0) -> dict:
    return {
        "game_id": game_id,
        "forecast_time": issued,
        "forecast_issue_time": issued,
        "weather_source": "historical_forecast",
        "is_outdoor": True,
        "weather_coverage": True,
        "temp_f": temp,
        "wind_mph": 8.0,
        "humidity_pct": 60.0,
        "precip_prob": 0.1,
        "precip_mm": None,
        "mos_precip_level": 0,
    }


def _forecast_less_row(game_id: str, *, is_outdoor: bool) -> dict:
    return {
        "game_id": game_id,
        "forecast_time": BULLETIN,
        "forecast_issue_time": None,
        "weather_source": "historical_forecast",
        "is_outdoor": is_outdoor,
        "weather_coverage": not is_outdoor,
    }


class TestTheOneFence:
    def test_a_bulletin_issued_before_the_lock_is_admitted(self):
        row, known = select_weather_row(
            pd.DataFrame([_forecast_row("G", BULLETIN)]), KICKOFF, LATE_BUILD
        )
        assert row is not None and known == BULLETIN

    def test_a_bulletin_issued_after_the_lock_is_not(self):
        row, known = select_weather_row(
            pd.DataFrame([_forecast_row("G", AFTER_LOCK)]), KICKOFF, LATE_BUILD
        )
        assert row is None and known is None

    def test_a_pre_lock_build_does_not_see_a_bulletin_that_does_not_exist_yet(self):
        """The `min` in min(lock, build_instant): issued before the lock, after the build."""
        build_before_it = datetime(2016, 9, 10, 6, 0, tzinfo=UTC)
        row, _ = select_weather_row(
            pd.DataFrame([_forecast_row("G", BULLETIN)]), KICKOFF, build_before_it
        )
        assert row is None

    def test_the_latest_admitted_bulletin_wins(self):
        earlier = BULLETIN - pd.Timedelta(hours=24)
        frame = pd.DataFrame(
            [_forecast_row("G", earlier, 40.0), _forecast_row("G", BULLETIN, 55.0)]
        )
        row, known = select_weather_row(frame, KICKOFF, LATE_BUILD)
        assert known == BULLETIN and row["temp_f"] == 55.0

    def test_the_fence_bound_is_this_games_lock(self):
        assert pd.Timestamp(lock_rule.game_lock(KICKOFF)).tz_convert("UTC") == LOCK

    def test_a_naive_build_instant_is_refused(self):
        with pytest.raises(ValueError, match="no timezone"):
            select_weather_row(
                pd.DataFrame([_forecast_row("G", BULLETIN)]),
                KICKOFF,
                datetime(2016, 9, 20),
            )


def _mixed_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    games = pd.DataFrame(
        [
            {"game_id": g, "season": 2016, "week": 1, "kickoff_et": KICKOFF}
            for g in ("2016_W01_FC@ST", "2016_W01_DM@DM", "2016_W01_AB@AB")
        ]
    )
    weather = pd.DataFrame(
        [
            _forecast_row("2016_W01_FC@ST", BULLETIN),
            _forecast_less_row("2016_W01_DM@DM", is_outdoor=False),
            _forecast_less_row("2016_W01_AB@AB", is_outdoor=True),
        ]
    )
    return games, weather


class TestTheProvenanceSupplier:
    def test_the_builder_is_the_registered_weather_supplier(self):
        assert SUPPLIER_ATTRIBUTES["weather"] == "weather_calc"
        assert isinstance(WeatherFeaturesCalculator(), InformationTimeProvider)

    def test_the_signature_is_non_empty(self):
        assert WeatherFeaturesCalculator().no_information_signature()
        assert WEATHER_NO_INFORMATION_SIGNATURE

    def test_each_state_reports_its_basis(self):
        games, weather = _mixed_frames()
        provenance = (
            WeatherFeaturesCalculator()
            .information_times(games, weather_df=weather, build_instant=LATE_BUILD)
            .set_index("game_id")
        )
        assert provenance.loc["2016_W01_FC@ST", "basis"] == "per_row"
        assert provenance.loc["2016_W01_FC@ST", "information_time"] == BULLETIN
        for game_id in ("2016_W01_DM@DM", "2016_W01_AB@AB"):
            assert provenance.loc[game_id, "basis"] == "no_information"
            assert pd.isna(provenance.loc[game_id, "information_time"])

    def test_a_mixed_frame_passes_the_gate_in_one_call(self):
        games, weather = _mixed_frames()
        calculator = WeatherFeaturesCalculator()
        with pytest.warns(DeprecationWarning):
            features = calculator.build_weather_features(
                games, weather_df=weather, build_instant=LATE_BUILD
            )
        dome = features.set_index("game_id").loc["2016_W01_DM@DM"]
        absent = features.set_index("game_id").loc["2016_W01_AB@AB"]
        assert dome["weather_affects_game"] == 0.0 and dome["weather_coverage"] == 1.0
        assert (
            absent["weather_affects_game"] == 1.0 and absent["weather_coverage"] == 0.0
        )
        for column in WEATHER_NO_INFORMATION_SIGNATURE:
            assert pd.isna(dome[column]) and pd.isna(absent[column])

        state = InformationTimeGate().check(
            "weather",
            features,
            calculator.information_times(
                games, weather_df=weather, build_instant=LATE_BUILD
            ),
            build_lock_frame(games),
            no_information_signature=calculator.no_information_signature(),
        )
        assert state.value == "checked"
