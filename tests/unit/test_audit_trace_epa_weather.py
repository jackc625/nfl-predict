"""AUDIT-03 deep hand-trace tests -- area 2 (QB-adj + opponent-adjusted EPA +
CPOE) and area 3 (weather / Open-Meteo).

Phase 20 plan 20-04 (D-08). EPA/CPOE are math-heavy and easy to get subtly
wrong; the QB-adjustment had an all-zeros regression in v1.0 (fixed P5). Weather
has an ingestion-bug history and a stale-docs provider drift (Meteostat in the
old maps vs Open-Meteo in reality). Per D-10 these CATALOG the as-found behavior
-- they do not fix anything.

Method (D-07 depth): synthetic fixtures with known answers (mirroring
test_opponent_adjustment.py) PLUS targeted real-row assertions. EPA is traced
for ORDER-OF-MAGNITUDE and lag-correctness, NOT bit-exact, because the nflverse
EPA model version moves between data pulls (A3). The weather real-row check is
network-optional: it re-fetches the Open-Meteo archive when online and falls
back to the on-disk weather_features parquet when offline (RESEARCH Environment
Availability).
"""

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

import features.weather as weather_mod
from data.storage import load_dataframe
from features.opponent_adj import OpponentAdjuster
from features.weather import WeatherFeaturesCalculator

# Open-Meteo Historical Weather API -- the canonical provider (NOT Meteostat;
# no Meteostat code exists in the repo). scripts/ingest_weather.py:37.
OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

# Arrowhead Stadium (KC, outdoor) -- the real-row weather trace venue.
ARROWHEAD_LAT = 39.0489
ARROWHEAD_LON = -94.4839


# ===========================================================================
# AREA 2 -- opponent-adjusted EPA + QB-adjustment + CPOE
# ===========================================================================


_ROUND_ROBIN_START = datetime(2023, 9, 10, 17, 0, tzinfo=UTC)  # Sunday 13:00 ET


def _round_robin_stats():
    """Synthetic 6-team round-robin with known per-team EPA (mirrors the
    test_opponent_adjustment analog) so the adjustment can be recomputed by
    hand.

    Plan 33.2-16: real abbreviations, play-by-play game ids and a timed schedule -- the
    adjuster resolves opponents through ``convert_legacy_game_id`` and admits every input at
    the lock of the game it informs, so the old ``game_2023_W01_B@A`` ids are refused.
    """
    a, b, c, d, e, f = "KC", "BUF", "MIA", "NYJ", "NE", "DEN"
    matchups = [
        [(a, b), (c, d), (e, f)],
        [(b, c), (d, e), (f, a)],
        [(a, d), (b, e), (c, f)],
        [(d, f), (e, a), (b, c)],
        [(a, c), (b, f), (d, e)],
        [(c, e), (d, a), (f, b)],
        [(a, f), (b, d), (c, e)],
    ]
    team_off_epa = {a: 0.10, b: -0.05, c: 0.05, d: 0.03, e: 0.08, f: 0.00}
    team_def_epa = {a: -0.05, b: 0.15, c: 0.10, d: 0.12, e: -0.03, f: 0.05}

    sched_rows, stats_rows = [], []
    season = 2023
    for week_idx, week in enumerate(matchups, start=1):
        kickoff = pd.Timestamp(_ROUND_ROBIN_START) + pd.Timedelta(weeks=week_idx - 1)
        for home, away in week:
            sched_rows.append(
                {
                    "game_id": f"{season}_W{week_idx:02d}_{away}@{home}",
                    "season": season,
                    "week": week_idx,
                    "home_team": home,
                    "away_team": away,
                    "kickoff_et": kickoff,
                }
            )
            pbp_id = f"{season}_{week_idx:02d}_{away}_{home}"
            for team in (home, away):
                for side, epa in (
                    ("offense", team_off_epa[team]),
                    ("defense", team_def_epa[team]),
                ):
                    stats_rows.append(
                        {
                            "game_id": pbp_id,
                            "season": season,
                            "week": week_idx,
                            "team": team,
                            "side": side,
                            "epa_per_play": epa,
                            "pass_epa_per_play": epa,
                            "rush_epa_per_play": epa,
                        }
                    )
    return pd.DataFrame(sched_rows), pd.DataFrame(stats_rows), team_off_epa


class TestArea2OpponentAdjEpa:
    """Recompute the opponent adjustment by hand and check lag-correctness."""

    def test_hand_recompute_order_of_magnitude(self):
        """Recompute adjustment = raw + (league_avg_def - opp_lagged_def) by hand
        and compare to rolling_opp_adj_epa_per_play for order-of-magnitude.

        With constant per-team EPA, the adjusted EPA stays within a small band
        of the raw EPA (the adjustment is the league-avg-vs-opponent gap, which
        is bounded by the spread of the synthetic def EPA values).
        """
        sched, stats, off_epa = _round_robin_stats()
        adjuster = OpponentAdjuster(window=10, min_opponent_games=4, schedule_df=sched)
        result = adjuster.build_features(
            sched,
            datetime(2023, 12, 31, tzinfo=UTC),
            target_season=2023,
            target_week=7,
            team_game_stats=stats,
        )
        off = result[result["side"] == "offense"]
        assert len(off) == 6
        # League def EPA spread is ~[-0.05, 0.15] -> adjustment magnitude < 0.25
        for _, row in off.iterrows():
            raw = off_epa[row["team"]]
            adj = row["rolling_opp_adj_epa_per_play"]
            assert not np.isnan(adj)
            assert abs(adj - raw) < 0.25, (
                f"Team {row['team']}: adjustment {abs(adj - raw):.3f} exceeds the "
                f"plausible league-spread band (raw={raw}, adj={adj})"
            )

    def test_one_week_lag_no_adjustment_below_threshold(self):
        """1-week-lag correctness: with only 1 prior week, opponents have fewer
        than min_opponent_games, so nothing is adjusted.

        Was: "adjusted == raw" -- raw EPA under an adjusted name, the fall-through Plan
        33.2-16 removed. Now the row is the flagged unknown: NaN, coverage 0.0.
        """
        from features.opponent_adj import OPP_ADJ_COVERAGE_COLUMN

        sched, stats, _ = _round_robin_stats()
        adjuster = OpponentAdjuster(window=10, min_opponent_games=4, schedule_df=sched)
        result = adjuster.build_features(
            sched,
            datetime(2023, 9, 15, tzinfo=UTC),  # the Protocol argument; not the fence
            target_season=2023,
            target_week=2,
            team_game_stats=stats,
        )
        off = result[result["side"] == "offense"]
        assert len(off) == 6
        assert off["rolling_opp_adj_epa_per_play"].isna().all(), (
            "below threshold nothing is adjusted -- confirms the 1-week lag + min-games guard"
        )
        assert (off[OPP_ADJ_COVERAGE_COLUMN] == 0.0).all()

    def test_uniform_opponents_zero_adjustment(self):
        """When every opponent has identical def EPA (= league avg), the
        adjustment is exactly zero (league_avg - opp_def == 0)."""
        sched, stats, _ = _round_robin_stats()
        # Overwrite def EPA to a single uniform value
        stats = stats.copy()
        stats.loc[stats["side"] == "defense", "epa_per_play"] = 0.05
        stats.loc[stats["side"] == "offense", "epa_per_play"] = 0.05
        adjuster = OpponentAdjuster(window=10, min_opponent_games=4, schedule_df=sched)
        result = adjuster.build_features(
            sched,
            datetime(2023, 12, 31, tzinfo=UTC),
            target_season=2023,
            target_week=7,
            team_game_stats=stats,
        )
        off = result[result["side"] == "offense"]
        assert len(off) == 6
        for _, row in off.iterrows():
            assert abs(row["rolling_opp_adj_epa_per_play"] - 0.05) < 0.001


class TestArea2QbAdjAndCpoe:
    """QB-adjustment all-zeros regression guard + CPOE plausibility on gold."""

    def test_qb_adjustment_has_nonzero_variance(self):
        """The v1.0 P5 all-zeros bug guard: home_qb_adjustment /
        away_qb_adjustment must have NON-ZERO variance on current gold (the
        adjustment is real, not a stuck-at-zero column)."""
        df = load_dataframe("features_wp", layer="gold")
        for col in ("home_qb_adjustment", "away_qb_adjustment"):
            assert col in df.columns, f"missing {col} in gold"
            s = df[col]
            assert s.std(skipna=True) > 0.01, (
                f"{col} has near-zero variance ({s.std():.3g}) -- the QB "
                f"all-zeros regression has returned"
            )
            assert (s != 0).sum() > 0.5 * len(s), (
                f"{col} is mostly zero ({(s != 0).sum()}/{len(s)} non-zero)"
            )

    def test_cpoe_in_plausible_range(self):
        """Offensive rolling CPOE is normalized in gold; confirm it has real
        variance and a plausible z-scored range (not a stuck/degenerate
        column)."""
        df = load_dataframe("features_wp", layer="gold")
        for col in ("home_off_rolling_cpoe", "away_off_rolling_cpoe"):
            assert col in df.columns
            s = df[col]
            assert s.std(skipna=True) > 0.05, f"{col} has too little variance"
            # Normalized feature: vast majority of the REAL values within +-5 sigma.
            # Blank cells are deliberate (no honest statistic exists there, owner rulings
            # 2026-09-22), so they are not counted as outliers; the column must still carry
            # real values for the range check to mean anything.
            real = s.dropna()
            assert len(real) > 0, f"{col} has no real values to range-check"
            within = (real.abs() <= 5.0).mean()
            assert within > 0.99, f"{col} has implausible outliers ({within:.3f})"

    def test_opp_adj_epa_present_with_variance_on_gold(self):
        """The six opponent-adjusted EPA columns reach gold with real variance
        (not the all-zero def-side bloat columns)."""
        df = load_dataframe("features_wp", layer="gold")
        for col in (
            "home_off_rolling_opp_adj_epa_per_play",
            "away_off_rolling_opp_adj_epa_per_play",
        ):
            assert col in df.columns
            assert df[col].std(skipna=True) > 0.05, f"{col} has too little variance"


# ===========================================================================
# AREA 3 -- weather / Open-Meteo
# ===========================================================================


# THE ONE WEATHER FENCE (Plan 33.2-12): a live forecast row is admitted only when its fetch
# instant is at or before min(the game's lock, the build instant), and every instant is aware.
# The Thursday-night opener's lock is 18:00 ET on the Wednesday, so the synthetic forecasts are
# fetched on the Wednesday morning.
WEATHER_KICKOFF_UTC = pd.Timestamp("2024-09-06 00:20", tz="UTC")
WEATHER_FETCH_UTC = pd.Timestamp("2024-09-04 12:00", tz="UTC")
WEATHER_AS_OF = datetime(2024, 9, 6, 0, 0, tzinfo=UTC)


def _weather_games():
    return pd.DataFrame(
        [
            {
                "game_id": "2024_W01_OUT@KC",
                "season": 2024,
                "week": 1,
                "kickoff_et": WEATHER_KICKOFF_UTC,
                "home_team": "KC",
                "away_team": "BAL",
            },
            {
                "game_id": "2024_W01_IN@DET",
                "season": 2024,
                "week": 1,
                "kickoff_et": WEATHER_KICKOFF_UTC,
                "home_team": "DET",
                "away_team": "LA",
            },
        ]
    )


def _weather_forecast_fixture():
    """One outdoor game with real weather, one indoor game (is_outdoor=False)."""
    return pd.DataFrame(
        [
            {
                "game_id": "2024_W01_OUT@KC",
                "forecast_time": WEATHER_FETCH_UTC,
                "weather_source": "forecast",
                "is_outdoor": True,
                "temp_f": 75.0,
                "wind_mph": 12.0,
                "precip_prob": 0.1,
                "precip_mm": 0.0,
                "humidity_pct": 55.0,
                "condition": "Clear",
            },
            {
                "game_id": "2024_W01_IN@DET",
                "forecast_time": WEATHER_FETCH_UTC,
                "weather_source": "forecast",
                "is_outdoor": False,
                "temp_f": 70.0,
                "wind_mph": 8.0,
                "precip_prob": 0.0,
                "precip_mm": 0.0,
                "humidity_pct": 50.0,
                "condition": "Clear",
            },
        ]
    )


class TestArea3WeatherBuilder:
    """Synthetic known-answer trace of WeatherFeaturesCalculator."""

    def test_outdoor_real_values_indoor_zeroed(self, monkeypatch):
        """Outdoor game gets real (non-zero) weather features; indoor game is
        fully zeroed (weather_severity_score=0, wind_mph=0, is_outdoor=0)."""
        forecast = _weather_forecast_fixture()
        monkeypatch.setattr(
            weather_mod, "load_dataframe", lambda *a, **k: forecast.copy()
        )
        calc = WeatherFeaturesCalculator()
        result = calc.build_features(
            _weather_games(),
            as_of_datetime=WEATHER_AS_OF,
        )

        out = result[result["game_id"] == "2024_W01_OUT@KC"].iloc[0]
        ind = result[result["game_id"] == "2024_W01_IN@DET"].iloc[0]

        # Outdoor: real wind, flagged outdoor
        assert out["is_outdoor"] == 1.0
        assert out["wind_mph"] > 0.0, "outdoor game should carry real wind"

        # Indoor: every weather feature zeroed
        assert ind["is_outdoor"] == 0.0
        assert ind["wind_mph"] == 0.0
        assert ind["weather_severity_score"] == 0.0
        assert ind["is_precipitation"] == 0.0

    def test_weather_severity_is_bounded_0_1(self, monkeypatch):
        """calculate_weather_severity returns a bounded 0-1 score for outdoor
        games (the severity-band invariant from the 260524-svu rework)."""
        forecast = _weather_forecast_fixture()
        monkeypatch.setattr(
            weather_mod, "load_dataframe", lambda *a, **k: forecast.copy()
        )
        calc = WeatherFeaturesCalculator()
        result = calc.build_features(_weather_games(), as_of_datetime=WEATHER_AS_OF)
        sev = result["weather_severity_score"]
        assert (sev >= 0.0).all() and (sev <= 1.0).all(), (
            f"weather_severity_score out of [0,1]: min={sev.min()}, max={sev.max()}"
        )


class TestArea3OpenMeteoRealRow:
    """Real outdoor game weather vs the Open-Meteo archive (network-optional)."""

    def test_real_outdoor_game_has_real_on_disk_weather(self):
        """The real 2024_W01_BAL@KC (Arrowhead, outdoor) has real Open-Meteo
        values on disk in weather_features silver (non-zero temp/wind)."""
        wf = load_dataframe("weather_features", layer="silver")
        row = wf[wf["game_id"] == "2024_W01_BAL@KC"]
        assert len(row) == 1, "expected one weather row for 2024_W01_BAL@KC"
        r = row.iloc[0]
        assert r["raw_temp_f"] > 0.0, "outdoor game should have a real temperature"
        assert r["raw_wind_mph"] >= 0.0
        # Early-September KC: warm; sanity-band, not bit-exact (A3)
        assert 40.0 < r["raw_temp_f"] < 110.0, (
            f"on-disk temp {r['raw_temp_f']}F outside the plausible Sept-KC band"
        )

    def test_open_meteo_archive_consistency_network_optional(self):
        """When online, re-fetch the Open-Meteo archive for Arrowhead on the
        2024 wk1 date and confirm it returns plausible temperatures consistent
        (order-of-magnitude) with the on-disk value. Offline -> skip with a
        documented gap (RESEARCH Environment Availability)."""
        try:
            import httpx
        except ImportError:  # pragma: no cover
            pytest.skip("httpx not available -- offline fallback (documented gap)")

        params = {
            "latitude": ARROWHEAD_LAT,
            "longitude": ARROWHEAD_LON,
            "start_date": "2024-09-05",
            "end_date": "2024-09-05",
            "hourly": "temperature_2m,wind_speed_10m,precipitation",
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "timezone": "America/Chicago",
        }
        try:
            resp = httpx.get(OPEN_METEO_URL, params=params, timeout=15.0)
        except (httpx.HTTPError, OSError):  # pragma: no cover
            pytest.skip(
                "Open-Meteo archive unreachable -- offline fallback "
                "(on-disk weather_features consistency already asserted; "
                "documented gap per RESEARCH Environment Availability)"
            )

        assert resp.status_code == 200, f"Open-Meteo returned {resp.status_code}"
        data = resp.json()
        temps = data["hourly"]["temperature_2m"]
        assert len(temps) == 24, f"expected 24 hourly temps, got {len(temps)}"
        day_high = max(t for t in temps if t is not None)
        # Order-of-magnitude consistency with the on-disk Sept-KC reading (A3)
        assert 40.0 < day_high < 110.0, (
            f"Open-Meteo Arrowhead 2024-09-05 day-high {day_high}F implausible"
        )


def test_provider_is_open_meteo_not_meteostat():
    """Resolve the stale-docs provider drift: the provider is Open-Meteo, not Meteostat.

    UPDATED BY PLAN 33-09. This test used to assert that
    `scripts/ingest_weather.py` contains the ARCHIVE host, which was true when the
    live path pointed there. D33-26 QUARANTINED that endpoint -- it cannot answer
    for a game that has not been played -- so the live module now holds the
    FORECAST endpoint and the archive endpoint lives in
    `scripts/backfill_historical_weather.py`.

    The question this test was written to settle is unchanged and is still
    answered: the provider is Open-Meteo on BOTH paths, and no Meteostat code
    exists anywhere. Only the file each endpoint lives in has moved.
    """
    import inspect

    import scripts.backfill_historical_weather as backfill
    import scripts.ingest_weather as ingest

    live_src = inspect.getsource(ingest)
    backfill_src = inspect.getsource(backfill)

    assert "api.open-meteo.com" in live_src, (
        "the LIVE ingest should hit the Open-Meteo forecast endpoint"
    )
    assert "archive-api.open-meteo.com" in backfill_src, (
        "the HISTORICAL backfill should hit the Open-Meteo archive endpoint"
    )
    assert "archive-api.open-meteo.com" not in live_src, (
        "the live ingest must NOT reach the archive endpoint (D33-26): a "
        "reanalysis product cannot answer for a game that has not been played"
    )
    for src in (live_src, backfill_src):
        assert "meteostat" not in src.lower(), (
            "no Meteostat code should exist (the stale maps are wrong)"
        )
