"""A retractable roof's open/closed state is not known at the lock, so it reaches NO model input.

p332_ extra step 4b (Plan 33.2-14, orchestrator-assigned; owner ruling 2026-09-21,
"Treat as unknown at lock").

THE LEAK THIS GUARDS AGAINST. Rung 4 (Plan 33.2-12) applied Phase 33.1's per-game roof rule:
a game at a retractable-roof stadium whose roof the pinned feed recorded as ``closed`` became a
dome with no weather. Whether a retractable roof closes is decided close to kickoff, often
BECAUSE of the weather, so the realized state is post-lock information -- "closed" encodes "the
weather turned bad". 620 games of 2002-2025 carried it. D33.2-04's "roof known at the lock"
covers the FIXED roof type (dome / open-air / retractable) only.

WHAT IS ASSERTED, as planted pairs: the realized roof of a retractable-venue game is set to
``closed`` in one run and ``open`` in the other, every other input held fixed, and the result
must be IDENTICAL --

* on the history path (``scripts.weather_from_mos.regenerated_weather_rows`` over the bronze
  bulletins on disk, then ``WeatherFeaturesCalculator.build_weather_features`` over those
  rows, which is what gold merges);
* on the live daily path (``WeatherDataIngester.fetch_forecast_for_games``), where the game
  row itself carries the planted roof;
* on the one model-visible roof flag, ``venue_retractable`` (``features.contextual``), which
  is read from the venue's FIXED type at the lock venue and must say 1.0 for every game at a
  retractable stadium whatever the game row says.

Both states must produce the day-before forecast, as an outdoor game does; a retractable game
whose bulletin is absent (``2019_W18_BUF@HOU``, a confirmed archive gap) is an honest ABSENCE
-- NULL values and the coverage flag down -- never a dome and never a stand-in run. A
fixed-roof dome stays a dome.

The history case FAILS on the rung-4 code, which read the planted roof through
``load_pinned_game_facts`` and turned the ``closed`` run's games into domes; that is what makes
it evidence rather than a description.

Read-only: the history case reads bronze and silver ``games`` on disk and writes nothing (it
requests ``data_boundary_guard``), and every network socket is refused and counted.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pandas as pd
import pytest

# The live weather ingest reads the pinned model's run time from meta.json once a
# forecast is fetched (Plan 33.2-27 Task 2); this module stubs the forecast, so the
# stamp is kept offline too.
pytestmark = pytest.mark.usefixtures("openmeteo_meta_offline")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "data"
SILVER_GAMES = DATA_ROOT / "silver" / "games.parquet"
MOS_BRONZE = DATA_ROOT / "bronze" / "mos"

#: A retractable-venue game whose roof the feed recorded CLOSED and whose bulletin exists.
CLOSED_WITH_BULLETIN = "2003_W03_KC@HOU"
#: The one retractable-venue game whose 12 UTC run is a confirmed archive gap (Plan 33.2-11).
CLOSED_ARCHIVE_GAP = "2019_W18_BUF@HOU"
#: A FIXED-roof dome (U.S. Bank Stadium): weather never applies, whatever the feed says.
FIXED_DOME = "2019_W01_ATL@MIN"

HISTORY_GAMES = (CLOSED_WITH_BULLETIN, CLOSED_ARCHIVE_GAP, FIXED_DOME)
RETRACTABLE_GAMES = (CLOSED_WITH_BULLETIN, CLOSED_ARCHIVE_GAP)

needs_history = pytest.mark.skipif(
    not (SILVER_GAMES.is_file() and MOS_BRONZE.is_dir()),
    reason=(
        "silver games or the MOS bronze is absent -- data/ is gitignored. Produce the "
        "bronze with scripts/backfill_mos_forecasts.py (Plan 33.2-11)"
    ),
)


def _history_games() -> pd.DataFrame:
    games = pd.read_parquet(SILVER_GAMES, engine="pyarrow")
    subset = games[games["game_id"].isin(HISTORY_GAMES)]
    assert set(subset["game_id"]) == set(HISTORY_GAMES), "a named game left silver"
    return subset.reset_index(drop=True)


def _planted_feed(games: pd.DataFrame, roof: str):
    """A stand-in for the sealed pinned feed in which every named game's roof is *roof*."""

    def load(seasons: Any) -> pd.DataFrame:
        del seasons
        return pd.DataFrame(
            {
                "game_id": games["game_id"].to_numpy(),
                "stadium_id": games["stadium_id"].to_numpy(),
                "roof": [roof] * len(games),
            }
        )

    return load


def _regenerate(monkeypatch: pytest.MonkeyPatch, roof: str) -> pd.DataFrame:
    """Silver rows for the named games with the realized roof planted as *roof*.

    The planted feed replaces BOTH places a caller could read the per-game roof from: the
    loader's home module and the name the regeneration module imported it under (absent
    since step 4b, hence ``raising=False``).
    """
    import scripts.backfill_historical_weather as backfill
    import scripts.weather_from_mos as regen

    games = _history_games()
    loader = _planted_feed(games, roof)
    monkeypatch.setattr(backfill, "load_pinned_game_facts", loader)
    monkeypatch.setattr(regen, "load_pinned_game_facts", loader, raising=False)
    calls = [0]
    with regen.deny_network(calls):
        rows, report = regen.regenerated_weather_rows(games, DATA_ROOT)
    assert calls[0] == 0
    assert report.observation_rows == 0
    return rows.sort_values("game_id").reset_index(drop=True)


@needs_history
class TestTheHistoryPathCannotSeeTheRealizedRoof:
    """Silver weather, and the features built from it, are identical for open and closed."""

    def test_open_and_closed_produce_identical_silver_rows(
        self, monkeypatch: pytest.MonkeyPatch, data_boundary_guard
    ) -> None:
        closed = _regenerate(monkeypatch, "closed")
        opened = _regenerate(monkeypatch, "open")
        pd.testing.assert_frame_equal(closed, opened)

    def test_open_and_closed_produce_identical_gold_bound_features(
        self, monkeypatch: pytest.MonkeyPatch, data_boundary_guard
    ) -> None:
        """The frame gold merges (``weather_features``) is identical as well."""
        from features.weather import WeatherFeaturesCalculator

        games = _history_games()
        build_instant = datetime(2026, 9, 21, tzinfo=UTC)
        built = []
        for roof in ("closed", "open"):
            rows = _regenerate(monkeypatch, roof)
            features = WeatherFeaturesCalculator().build_weather_features(
                games, weather_df=rows, build_instant=build_instant
            )
            built.append(features.sort_values("game_id").reset_index(drop=True))
        pd.testing.assert_frame_equal(built[0], built[1])

    def test_a_closed_retractable_game_carries_its_day_before_forecast(
        self, monkeypatch: pytest.MonkeyPatch, data_boundary_guard
    ) -> None:
        rows = _regenerate(monkeypatch, "closed").set_index("game_id")
        row = rows.loc[CLOSED_WITH_BULLETIN]
        assert bool(row["is_outdoor"]) is True
        assert bool(row["weather_coverage"]) is True
        assert row["temp_f"] is not None and not pd.isna(row["temp_f"])
        assert row["forecast_issue_time"] is not None

    def test_the_archive_gap_is_an_honest_absence_not_a_dome(
        self, monkeypatch: pytest.MonkeyPatch, data_boundary_guard
    ) -> None:
        """NULL values and the coverage flag down; no issue time; no stand-in run."""
        rows = _regenerate(monkeypatch, "closed").set_index("game_id")
        row = rows.loc[CLOSED_ARCHIVE_GAP]
        assert bool(row["is_outdoor"]) is True
        assert bool(row["weather_coverage"]) is False
        for column in ("temp_f", "wind_mph", "humidity_pct", "forecast_issue_time"):
            assert row[column] is None or pd.isna(row[column]), column

    def test_a_fixed_roof_dome_stays_a_dome(
        self, monkeypatch: pytest.MonkeyPatch, data_boundary_guard
    ) -> None:
        """NO-FALSE-POSITIVE CONTROL: the fix must not give a fixed dome weather."""
        rows = _regenerate(monkeypatch, "open").set_index("game_id")
        row = rows.loc[FIXED_DOME]
        assert bool(row["is_outdoor"]) is False
        assert bool(row["weather_coverage"]) is True
        assert row["temp_f"] is None or pd.isna(row["temp_f"])

    def test_the_regeneration_module_reads_no_per_game_roof(self) -> None:
        """Structural partner of the planted pair: the per-game roof has no reader here."""
        import scripts.weather_from_mos as regen

        source = Path(regen.__file__).read_text(encoding="utf-8")
        assert not hasattr(regen, "closed_roof_game_ids")
        assert not hasattr(regen, "load_pinned_game_facts")
        assert "NFLVERSE_ROOF_MAP" not in source


# ---------------------------------------------------------------------------
# The live daily path.
# ---------------------------------------------------------------------------

AS_OF = datetime(2026, 9, 11, 20, 0, tzinfo=UTC)

RETRACTABLE_VENUE: dict[str, Any] = {
    "stadium_id": "HOU00",
    "venue_id": "nrg_stadium",
    "venue_name": "NRG Stadium",
    "latitude": 29.6847,
    "longitude": -95.4107,
    "roof_type": "retractable",
    "timezone": "America/Chicago",
    "home_teams": ["HOU"],
}

FORECAST_RECORD: dict[str, Any] = {
    "temp_f": 88.0,
    "temp_c": 31.1,
    "wind_mph": 9.0,
    "wind_direction": 180.0,
    "humidity_pct": 64.0,
    "precip_mm": 0.0,
    "precip_prob": None,
    "condition": None,
    "condition_code": 1,
    "visibility_km": None,
    "dew_point_f": 74.0,
    "apparent_temp_f": 97.0,
    "snowfall_cm": 0.0,
    "wind_gusts_mph": 15.0,
    "cloud_cover_pct": 20.0,
    "weather_code": 1,
}


def _live_game(roof: str) -> pd.DataFrame:
    """One live game at a retractable stadium whose row carries a planted realized roof."""
    return pd.DataFrame(
        [
            {
                "game_id": "2026_W02_KC@HOU",
                "season": 2026,
                "week": 2,
                "home_team": "HOU",
                "away_team": "KC",
                "kickoff_et": AS_OF + timedelta(days=3),
                "stadium_id": "HOU00",
                "neutral_site": False,
                "roof": roof,
            }
        ]
    )


@pytest.fixture
def ingester():
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.ingest_weather import WeatherDataIngester

        return WeatherDataIngester()


class TestTheLiveDailyPathCannotSeeTheRealizedRoof:
    def test_open_and_closed_produce_identical_live_records(self, ingester) -> None:
        venues_df = pd.DataFrame([RETRACTABLE_VENUE])
        forecast_time = datetime(2026, 9, 11, 19, 0, tzinfo=UTC)
        records = []
        for roof in ("closed", "open"):
            with patch.object(
                ingester,
                "_fetch_openmeteo_forecast",
                AsyncMock(return_value=dict(FORECAST_RECORD)),
            ):
                records.append(
                    ingester.fetch_forecast_for_games(
                        _live_game(roof),
                        venues_df,
                        forecast_time=forecast_time,
                        as_of_utc=AS_OF,
                    )
                )
        pd.testing.assert_frame_equal(records[0], records[1])
        row = records[0].iloc[0]
        assert bool(row["is_outdoor"]) is True
        assert row["temp_f"] == FORECAST_RECORD["temp_f"]


# ---------------------------------------------------------------------------
# The model-visible flag: the venue MAY close.
# ---------------------------------------------------------------------------


class TestTheRetractableFlagIsTheFixedType:
    """``venue_retractable`` says "this roof may close", from the venue, never the game."""

    def test_every_retractable_stadium_reads_one_and_every_other_reads_zero(
        self,
    ) -> None:
        from features.contextual import ContextualFeaturesCalculator

        calculator = ContextualFeaturesCalculator()
        seen = set()
        for venue in calculator.venues_data["venues"]:
            encoded = calculator.encode_venue_features(venue["venue_id"])
            expected = 1.0 if venue["roof_type"] == "retractable" else 0.0
            assert encoded["venue_retractable"] == expected, venue["venue_id"]
            seen.add(venue["roof_type"])
        assert "retractable" in seen

    def test_the_flag_ignores_the_realized_roof_on_the_game_row(self) -> None:
        """The venue at the lock decides the flag; a planted per-game roof cannot move it."""
        from features.contextual import ContextualFeaturesCalculator

        calculator = ContextualFeaturesCalculator()
        flags = []
        for roof in ("closed", "open"):
            game = _live_game(roof).iloc[0]
            venue_id = calculator._resolve_venue_id_for_game(game)
            flags.append(calculator.encode_venue_features(venue_id))
        assert flags[0] == flags[1]
        assert flags[0]["venue_retractable"] == 1.0
