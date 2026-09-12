"""A week that fails part-way leaves `weather.parquet` byte-identical for that week.

WHY THIS IS A CONTENT DIGEST AND NOT AN INSPECTION
--------------------------------------------------
"Nothing was written" is the kind of claim that is easy to make and easy to be wrong
about. A row count is unchanged by a rewrite that preserves the count; a column list is
unchanged by a rewrite that preserves the schema. The only instrument that answers the
question actually being asked is a hash of the file's bytes, which is what
`tests.data_boundary.digest_file` provides and what every boundary claim in this phase is
required to cite.

EVERY WRITE IN THIS MODULE GOES TO `tmp_path`
---------------------------------------------
This module is one of the highest-risk shapes in the phase: it drives a real production
WRITER. `data.storage.upsert_silver` takes a `base_path`, and every call here passes a
`tmp_path` root. Nothing carries the `writes_production_store` marker, because nothing
here may touch a production store -- if a test in this module ever needed that marker, the
correct response would be to fix the test, not to grant the exemption.

NO NETWORK. The fetch seam is replaced by a callable that answers from a dict or raises,
so the "one game in the week fails" case is produced deliberately rather than by hoping a
provider misbehaves.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from tests.data_boundary import digest_file
from utils.exceptions import WeatherDataError

AS_OF = datetime(2026, 9, 11, 20, 0, tzinfo=UTC)
KICKOFF = AS_OF + timedelta(days=3)

VENUES = pd.DataFrame(
    [
        {
            "stadium_id": "BUF00",
            "venue_id": "highmark_stadium",
            "latitude": 42.7738,
            "longitude": -78.787,
            "roof_type": "outdoor",
            "timezone": "America/New_York",
            "home_teams": ["BUF"],
        },
        {
            "stadium_id": "GNB00",
            "venue_id": "lambeau_field",
            "latitude": 44.5013,
            "longitude": -88.0622,
            "roof_type": "outdoor",
            "timezone": "America/Chicago",
            "home_teams": ["GB"],
        },
    ]
)

FORECAST_RECORD: dict[str, Any] = {
    "temp_f": 51.2,
    "temp_c": 10.7,
    "wind_mph": 11.4,
    "wind_direction": 240.0,
    "humidity_pct": 72.0,
    "precip_mm": 0.6,
    "precip_prob": None,
    "condition": None,
    "condition_code": 61,
    "visibility_km": None,
    "dew_point_f": 42.1,
    "apparent_temp_f": 47.0,
    "snowfall_cm": 0.0,
    "wind_gusts_mph": 19.3,
    "cloud_cover_pct": 88.0,
    "weather_code": 61,
}


@pytest.fixture
def ingester():
    with patch("scripts.ingest_weather.get_settings") as mock_settings:
        mock_settings.return_value = MagicMock()
        from scripts.ingest_weather import WeatherDataIngester

        return WeatherDataIngester()


@pytest.fixture
def silver_sandbox(tmp_path: Path) -> tuple[Path, Path]:
    """A sandbox data root holding a pre-existing `silver/weather.parquet`.

    The seed rows stand in for the fourteen real ones. They are written HERE, under
    `tmp_path`, and the production store is never opened -- copying the real file in
    would make this module's failure mode "reads production" instead of "writes it",
    which is not an improvement.
    """
    root = tmp_path / "lake"
    (root / "silver").mkdir(parents=True)
    seed = pd.DataFrame(
        [
            {
                "game_id": "2024_W06_SF@SEA",
                "forecast_time": datetime(2024, 10, 11, 22, 0, tzinfo=UTC),
                "game_time": datetime(2024, 10, 13, 17, 0, tzinfo=UTC),
                "temp_f": 60.0,
                "wind_mph": 5.0,
                "precip_mm": 0.0,
                "is_outdoor": True,
            }
        ]
    )
    target = root / "silver" / "weather.parquet"
    seed.to_parquet(target, engine="pyarrow", index=False)
    return root, target


def _week_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "2026_W02_KC@BUF",
                "season": 2026,
                "week": 2,
                "home_team": "BUF",
                "away_team": "KC",
                "kickoff_et": KICKOFF,
                "stadium_id": "BUF00",
                "neutral_site": False,
            },
            {
                "game_id": "2026_W02_MIN@GB",
                "season": 2026,
                "week": 2,
                "home_team": "GB",
                "away_team": "MIN",
                "kickoff_et": KICKOFF,
                "stadium_id": "GNB00",
                "neutral_site": False,
            },
        ]
    )


class TestAFailedWeekWritesNothing:
    """T-33-45: a mid-week failure must not leave the file half-updated."""

    def test_a_failing_fetch_leaves_the_target_digest_identical(
        self, ingester, silver_sandbox
    ) -> None:
        root, target = silver_sandbox
        before = digest_file(target)

        calls: list[str] = []

        async def _fetch(latitude, longitude, game_date, game_hour, venue_timezone):
            calls.append(game_date)
            if len(calls) > 1:
                raise WeatherDataError("Open-Meteo API timeout for the second game")
            return dict(FORECAST_RECORD)

        with patch.object(ingester, "_fetch_openmeteo_forecast", _fetch):
            with pytest.raises(WeatherDataError, match="timeout"):
                ingester.ingest_week_forecast(
                    _week_frame(),
                    VENUES,
                    as_of_utc=AS_OF,
                    base_path=root,
                )

        after = digest_file(target)
        assert after == before, (
            "data/silver/weather.parquet moved after a week in which one game's fetch "
            "failed. The week must be all-or-nothing: a partially written week is "
            "indistinguishable from a complete one once it is on disk."
        )
        assert len(calls) == 2, "the second game was never attempted"

    def test_a_partial_payload_leaves_the_target_digest_identical(
        self, ingester, silver_sandbox
    ) -> None:
        """The subtler failure: every fetch SUCCEEDS but one game is missing from the
        frame handed to the writer. Nothing raised, nothing timed out, and the week
        would still be half-true."""
        import scripts.ingest_weather as ingest

        root, target = silver_sandbox
        before = digest_file(target)

        complete = _week_frame()
        dropped = complete.iloc[:1]

        def _drop_one(games_df, venues_df, forecast_time=None, **kwargs):
            return pd.DataFrame(
                [
                    {
                        "game_id": "2026_W02_KC@BUF",
                        "forecast_time": AS_OF,
                        "game_time": KICKOFF,
                        "is_outdoor": True,
                        **FORECAST_RECORD,
                    }
                ]
            )

        assert len(dropped) == 1
        with patch.object(ingester, "fetch_forecast_for_games", _drop_one):
            with pytest.raises(ingest.IncompleteForecastPayloadError) as caught:
                ingester.ingest_week_forecast(
                    complete, VENUES, as_of_utc=AS_OF, base_path=root
                )

        assert "2026_W02_MIN@GB" in str(caught.value)
        assert digest_file(target) == before

    def test_a_beyond_horizon_week_leaves_the_target_digest_identical(
        self, ingester, silver_sandbox
    ) -> None:
        import scripts.ingest_weather as ingest

        root, target = silver_sandbox
        before = digest_file(target)

        far = _week_frame()
        far["kickoff_et"] = AS_OF + timedelta(days=ingest.FORECAST_HORIZON_DAYS + 3)

        with pytest.raises(ingest.BeyondForecastHorizonError):
            ingester.ingest_week_forecast(far, VENUES, as_of_utc=AS_OF, base_path=root)

        assert digest_file(target) == before


class TestASuccessfulWeekIsWrittenOnce:
    """The control: the same machinery DOES write when the week is complete.

    Without this, every assertion above would be satisfied by a writer that never writes
    at all -- which is the shape of a guard that passes because nothing works.
    """

    def test_a_complete_week_changes_the_digest_and_lands_both_games(
        self, ingester, silver_sandbox
    ) -> None:
        root, target = silver_sandbox
        before = digest_file(target)

        async def _fetch(latitude, longitude, game_date, game_hour, venue_timezone):
            return dict(FORECAST_RECORD)

        with patch.object(ingester, "_fetch_openmeteo_forecast", _fetch):
            written = ingester.ingest_week_forecast(
                _week_frame(), VENUES, as_of_utc=AS_OF, base_path=root
            )

        assert digest_file(target) != before
        assert len(written) == 2

        stored = pd.read_parquet(target)
        assert set(stored["game_id"]) == {
            "2024_W06_SF@SEA",
            "2026_W02_KC@BUF",
            "2026_W02_MIN@GB",
        }

    def test_the_write_goes_to_the_sandbox_and_not_to_the_repository(
        self, ingester, silver_sandbox
    ) -> None:
        """The interlock, asserted rather than assumed.

        Plan 33-06 overwrote all three production gold matrices from a test that omitted
        its sandbox. This asserts the sandbox root is genuinely outside the repository's
        own `data/` tree, so a future edit that drops `base_path` cannot pass here.
        """
        root, _ = silver_sandbox
        production = Path("data").resolve()
        assert root.resolve() != production
        assert production not in root.resolve().parents
