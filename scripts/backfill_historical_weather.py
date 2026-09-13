"""Historical weather backfill -- the Open-Meteo ARCHIVE endpoint's ONLY caller.

THE ENDPOINT IS QUARANTINED HERE, NOT DELETED (D33-26)
------------------------------------------------------
`https://archive-api.open-meteo.com/v1/archive` is an ERA5 REANALYSIS product. It
describes weather that has already occurred, which means it is the right answer for
history and a category error for a game that has not been played. Pointing the live
2026 path at it was R8's defect: the request succeeds, the hours come back null, and
the only "repairs" available at that point -- read the archive anyway, or substitute a
seasonal average -- are the fabricated-data class this project has already disclosed
once.

So the live path can no longer reach it. `scripts/ingest_weather.py` holds the FORECAST
endpoint and nothing else, and `tests/unit/test_weather_archive_quarantined.py` scans it
structurally to keep that true: a source scan is the only instrument that works here,
because an archive fallback RETURNS A VALUE and therefore leaves a green test suite
behind it.

WHY IT IS KEPT AT ALL, WHICH IS THE OTHER HALF OF THE DECISION
---------------------------------------------------------------
Deleting the endpoint would strand two decades of history. Open-Meteo's Historical
Forecast API -- what the forecast SAID at the time -- reaches back only to about 2021.
For 2002-2020 the archive endpoint is the only source there is, and the historical
weather backfill it powers is the NAMED flip condition for the 2026 gold-default switch
in `features/weather.py`. A quarantine that removed the capability would quietly cancel
Phase 37's input.

This module is therefore a separately-named CLI command whose name says what it does. It
is deliberately EXCLUDED from the quarantine scan's module list, and that exclusion is
asserted rather than assumed -- see
`test_the_scan_does_not_flag_the_legitimate_backfill_module`.

WHAT LIVES HERE
---------------
* `ARCHIVE_ENDPOINT_URL` and `fetch_game_weather`, moved verbatim from the ingest module.
* `HistoricalWeatherBackfiller`, a subclass of the live `WeatherDataIngester`. It
  inherits the SHARED machinery -- venue resolution under the D33-15 stadium_id routing,
  the roof map, the record factories -- and adds only the archive fetch and the archive
  ingest. Subclassing rather than copying is what stops the two paths from drifting apart
  about which stadium a game is at.
* `stamp_weather_source_on_existing_rows`, the one-shot provenance backfill for the rows
  that predate the `weather_source` column.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

import argparse
import asyncio
import sys
import time
from datetime import UTC, datetime
from typing import Any

import httpx
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from data.storage import save_bronze_snapshot, upsert_silver
from data.upstream_pin import load_schedules
from scripts.ingest_weather import (
    HOURLY_VARIABLES,
    NFLVERSE_ROOF_MAP,
    WEATHER_SOURCE_VOCABULARY,
    WeatherDataIngester,
    normalize_wind_direction,
    resolve_venue_local_hour,
    validate_weather_source,
)
from utils import (
    DataIngestionError,
    get_current_nfl_week,
    get_logger,
    log_data_operation,
)
from utils.exceptions import WeatherDataError
from utils.game_id_utils import create_standard_game_id, is_valid_game_id

logger = get_logger(__name__)

# The Open-Meteo ARCHIVE (ERA5 reanalysis) endpoint. THE ONLY DECLARATION OF IT IN
# THIS REPOSITORY. See the module docstring for why it is quarantined here rather
# than deleted: the Historical Forecast API reaches back only to about 2021, so
# this is the sole source for 2002-2020.
ARCHIVE_ENDPOINT_URL = "https://archive-api.open-meteo.com/v1/archive"

# The two HISTORICAL members of the weather_source vocabulary, named here rather
# than in the live ingest module. That split is not cosmetic: the quarantine's
# mechanical form is that `scripts/ingest_weather.py` exposes no name containing
# "ARCHIVE", and a WEATHER_SOURCE_ARCHIVE constant over there would pass the
# source scan while making that runtime check false. The values are pinned
# against the shared vocabulary immediately below, so the split cannot drift.
WEATHER_SOURCE_ARCHIVE = "archive"
WEATHER_SOURCE_HISTORICAL_FORECAST = "historical_forecast"

assert WEATHER_SOURCE_ARCHIVE in WEATHER_SOURCE_VOCABULARY
assert WEATHER_SOURCE_HISTORICAL_FORECAST in WEATHER_SOURCE_VOCABULARY

# The bronze table name THIS backfill writes under. Declared here so Plan 33.1-05's
# per-season resume rule and this module read ONE name rather than two string
# literals that can drift. It is deliberately NOT "weather": the ten legacy weather
# bronze files live under that name, and a resume rule that globbed them would skip
# seasons SPEC R2 requires re-fetching under the corrected routing.
BACKFILL_BRONZE_TABLE: str = "weather_backfill"

# THE ABSENT-OBSERVATION PREDICATE (Plan 33.1-02 Ruling D3).
#
# D33.1-07 names three states -- dome, observation, absence -- but says nothing
# about how the middle one is DETECTED, and before this constant the code could not
# detect it at all: an all-null ERA5 day produced a row that LOOKED like an
# observation, with `temp_c` derived as None and every other measurement passed
# straight through.
#
# FIVE NAMES, and the count is the decision:
#
# * Not ONE. A single-variable test would call a day absent whenever ERA5 happens
#   to have temperature but no cloud cover, which is an ordinary partial day.
# * Not ALL TWELVE. `snowfall` and `wind_gusts_10m` are legitimately null on
#   ordinary days, so including them would make the predicate unreachable.
# * These five are the series the weather FEATURES are derived from, so "none of
#   the five" is exactly "nothing downstream can be computed".
ABSENT_OBSERVATION_MEASUREMENTS: tuple[str, ...] = (
    "temperature_2m",
    "wind_speed_10m",
    "precipitation",
    "relative_humidity_2m",
    "cloud_cover",
)

# The sentinel `fetch_game_weather` returns INSTEAD of a measurement dict when the
# predicate above fires. A sentinel rather than `None` so a caller that forgets to
# branch gets a value it cannot mistake for a payload, and rather than an exception
# because an absent observation is a RECORDED STATE, not a failure -- the row is
# still written, with every weather column NULL and `weather_coverage` False.
ABSENT_OBSERVATION: str = "ABSENT_OBSERVATION"


def observation_is_absent(hourly: dict[str, Any], idx: int) -> bool:
    """True only when EVERY one of the five key measurements is null at *idx*.

    RULING D3'S PAYLOAD PREDICATE, stated once. An observation is ABSENT when all
    five ``ABSENT_OBSERVATION_MEASUREMENTS`` series are ``None`` at the selected
    index -- the provider answered, the day exists, and there is nothing in it.

    A PARTIAL NULL IS NEITHER STATE, and that is the whole reason the predicate is
    five names wide instead of one. Some of the five present and some absent is an
    OBSERVATION WITH MISSING FIELDS: the present values are carried and the absent
    ones stay NULL, and Plan 33.1-04's Ruling J is what makes every column derived
    from a NULL inherit that NULL rather than defaulting. Do NOT later widen this
    into "any null" -- that would reclassify an ordinary day with no cloud-cover
    reading as an absence and throw away a real temperature.

    A series that is MISSING FROM THE PAYLOAD ENTIRELY, or too short to reach
    *idx*, counts as null for this test. The alternative is a ``KeyError`` or an
    ``IndexError`` raised from inside a predicate, which would turn "the provider
    sent a thin payload" into a crash rather than into the recorded absence this
    function exists to detect.

    Args:
        hourly: The response's ``hourly`` object.
        idx: The venue-local hour index the game is read at.

    Returns:
        True when the observation is absent, False for a full or partial one.
    """
    for name in ABSENT_OBSERVATION_MEASUREMENTS:
        series = hourly.get(name)
        if not isinstance(series, list) or idx >= len(series):
            continue
        if series[idx] is not None:
            return False
    return True


def _archive_http_error_detail(error: httpx.HTTPStatusError) -> str:
    """The status code AND the API's own ``reason`` string, when it gave one.

    The archive explains itself in a ``{"error": true, "reason": "..."}`` body on
    every 400 -- "Parameter 'start_date' is out of allowed range from 1940-01-01 to
    2026-09-12" is the literal text of one. A refusal that reports only the status
    code throws away the only diagnostic there is, and the operator's next move is
    to re-run into the same wall. Falls back to the bare status when the body is
    not JSON or carries no ``reason``.
    """
    status = error.response.status_code
    try:
        body = error.response.json()
    except (ValueError, TypeError):
        return str(status)
    reason = body.get("reason") if isinstance(body, dict) else None
    return f"{status} ({reason})" if reason else str(status)


# WHAT D33.1-08 CORRECTED IN `fetch_game_weather`, recorded here in COMMENTS
# rather than in the docstring below. That placement is deliberate: the Plan
# 33.1-02 source scan asserts this module contains ZERO non-comment lines naming
# the old fixed zone or the old clamp, and a docstring is not a comment. Writing
# the history as prose inside the function would make the scan read a correct
# module as a defective one -- and a guard that cannot distinguish a description
# of a fix from the fix's absence is a guard that gets deleted.
#
# 1. THE ZONE. The request hardcoded the Eastern zone -- "America" + "/New_York"
#    -- which is the wrong zone for 50 of the 55 venues in the corpus. A
#    Twickenham game read against an Eastern-zone array is the wrong hour, and for
#    a 09:30 London kickoff it is also the wrong DAY. The zone now arrives as the
#    `venue_timezone` argument, taken from the committed venue record.
#
# 2. THE HOUR. The index was CLAMPED to the last available entry with a
#    `min(...)` over the array length. A clamped hour is a real, plausible,
#    internally-consistent reading FOR THE WRONG MOMENT, and nothing in the number
#    says so. It is now the forecast sibling's two named refusals: an empty array
#    raises, and an array that does not reach the requested hour raises and says
#    the hour is not clamped.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    reraise=True,
)
async def fetch_game_weather(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    game_date: str,
    game_hour: int,
    venue_timezone: str,
) -> dict[str, Any] | str:
    """Fetch an ERA5 observation for one game, on the VENUE'S OWN clock.

    THE SIBLING of :func:`scripts.ingest_weather.fetch_game_forecast`, and
    deliberately the same shape: same retry policy, same three error branches, same
    two refusals, same direct indexing. The differences are the endpoint and the
    direction in time. The two D33.1-08 corrections this function carries -- the
    venue's own zone, and the refusal that replaced the clamp -- are recorded in
    the comment block immediately above the decorator.

    ``temp_c``'s derivation is left exactly as it was: it is the single definition
    of that column and the same 0.1 F tolerance the R2 cross-check compares at.

    Args:
        client: httpx async client.
        latitude: Venue latitude.
        longitude: Venue longitude.
        game_date: The venue-LOCAL date, ``YYYY-MM-DD``.
        game_hour: The venue-LOCAL hour, 0-23.
        venue_timezone: The venue's own IANA zone, never a fixed one.

    Returns:
        A measurement dict matching the system schema, or the
        :data:`ABSENT_OBSERVATION` sentinel when
        :func:`observation_is_absent` fires. The caller BRANCHES on the sentinel
        and routes it to ``_create_absent_observation_record``; that branch is what
        makes the absent path reachable from a real response rather than only from
        a test that calls the factory directly.

    Raises:
        WeatherDataError: the call failed, or the response is malformed, or it does
            not reach the requested hour.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": game_date,
        "end_date": game_date,
        "hourly": HOURLY_VARIABLES,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": venue_timezone,
    }

    try:
        response = await client.get(ARCHIVE_ENDPOINT_URL, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise WeatherDataError(
            f"Open-Meteo API returned {_archive_http_error_detail(e)} "
            f"for ({latitude}, {longitude}) on {game_date}"
        )
    except httpx.TimeoutException as e:
        raise WeatherDataError(
            f"Open-Meteo API timeout for ({latitude}, {longitude}) on {game_date}: {e}"
        )
    except httpx.RequestError as e:
        raise WeatherDataError(
            f"Open-Meteo API request failed for ({latitude}, {longitude}) "
            f"on {game_date}: {e}"
        )

    data = response.json()

    if "hourly" not in data:
        raise WeatherDataError(
            f"Open-Meteo response missing 'hourly' key for "
            f"({latitude}, {longitude}) on {game_date}"
        )

    hourly = data["hourly"]
    temperatures = hourly.get("temperature_2m", [])
    if not temperatures:
        raise WeatherDataError(
            f"Open-Meteo archive returned NO hourly data for "
            f"({latitude}, {longitude}) on {game_date}. Nothing is imputed for it."
        )
    if game_hour >= len(temperatures):
        raise WeatherDataError(
            f"Open-Meteo archive returned {len(temperatures)} hours for "
            f"({latitude}, {longitude}) on {game_date}, which does not reach hour "
            f"{game_hour}. The hour is NOT clamped to the last available one: a "
            "clamped hour is a real reading for the wrong moment."
        )

    idx = game_hour

    if observation_is_absent(hourly, idx):
        logger.warning(
            "Archive observation ABSENT -- every key measurement null at the hour",
            latitude=latitude,
            longitude=longitude,
            game_date=game_date,
            game_hour=game_hour,
            measurements=list(ABSENT_OBSERVATION_MEASUREMENTS),
        )
        return ABSENT_OBSERVATION

    temp_f = temperatures[idx]

    return {
        "temp_f": temp_f,
        "temp_c": round((temp_f - 32) * 5 / 9, 1) if temp_f is not None else None,
        "wind_mph": hourly["wind_speed_10m"][idx],
        "wind_direction": normalize_wind_direction(hourly["wind_direction_10m"][idx]),
        "humidity_pct": hourly["relative_humidity_2m"][idx],
        "precip_mm": hourly["precipitation"][idx],
        "precip_prob": None,  # Not available in historical reanalysis data
        "condition": None,  # Derive from weather_code if needed downstream
        "condition_code": hourly["weather_code"][idx],
        "visibility_km": None,  # Not available in ERA5
        # New fields available from Open-Meteo
        "dew_point_f": hourly["dew_point_2m"][idx],
        "apparent_temp_f": hourly["apparent_temperature"][idx],
        "snowfall_cm": hourly["snowfall"][idx],
        "wind_gusts_mph": hourly["wind_gusts_10m"][idx],
        "cloud_cover_pct": hourly["cloud_cover"][idx],
        "weather_code": hourly["weather_code"][idx],
    }


def load_pinned_game_facts(seasons: list[int] | tuple[int, ...]) -> pd.DataFrame:
    """``game_id``, ``stadium_id`` and ``roof`` for *seasons*, from the sealed pin.

    THE PINNED FEED IS THE SOURCE FOR BOTH FACTS, and Ruling D records why rather
    than adding a silver column: ``data/silver/games.parquet`` has never carried
    ``roof``, it already carries a VENUE-level ``venue_roof`` with a different
    distribution, and a per-game ``roof`` beside it would put two similarly-named
    roof columns in one table -- the exact confusion SPEC R4 exists to remove.

    THE JOIN KEY IS DERIVED, NOT COPIED, and this is load-bearing. The pinned feed
    is nflverse-shaped and its own ``game_id`` reads ``2016_01_DET_IND``; silver's
    reads ``2016_W01_DET@IND``. Across the whole 2002-2025 corpus the two forms
    share ZERO values, so joining on the feed's raw column would match nothing and
    the caller's "did not join" refusal would fire for every game. The canonical id
    is therefore rebuilt here through :func:`create_standard_game_id` -- the SAME
    helper ``scripts/ingest_games._create_game_id`` used to write those silver ids,
    so the two agree by construction and not by coincidence. Verified 2026-09-12:
    6,499 of 6,499 derived ids match silver exactly.

    Args:
        seasons: The seasons to read.

    Returns:
        One row per game with exactly the three columns named above.

    Raises:
        DataIngestionError: any ``stadium_id`` or ``roof`` cell is null, or an id
            cannot be rebuilt. Both are routing inputs; a null one would send a
            game to a default rather than to a stadium.
    """
    frame = load_schedules(list(seasons))

    for column in ("stadium_id", "roof"):
        missing = frame[frame[column].isna()]
        if not missing.empty:
            raise DataIngestionError(
                f"the pinned schedules carry {len(missing)} row(s) with a null "
                f"{column!r} for seasons {sorted(seasons)}: "
                f"{sorted(str(v) for v in missing['game_id'].head(10))}. Both "
                "stadium_id and roof are ROUTING inputs -- one decides which "
                "stadium a game is fetched for, the other decides whether it is "
                "fetched at all -- so a null is refused rather than defaulted. "
                "Re-pin the upstream capture for those seasons."
            )

    try:
        canonical = [
            create_standard_game_id(
                int(row.season), int(row.week), row.away_team, row.home_team
            )
            for row in frame.itertuples()
        ]
    except (ValueError, TypeError) as exc:
        raise DataIngestionError(
            f"a pinned schedule row for seasons {sorted(seasons)} could not be "
            f"rebuilt into a canonical game_id: {exc}. The pinned feed's own "
            "game_id is nflverse-shaped and shares no values with silver, so the "
            "canonical form is derived through utils.game_id_utils."
            "create_standard_game_id -- the same helper scripts/ingest_games.py "
            "used to write silver. Re-pin or correct the upstream capture."
        ) from exc

    facts = pd.DataFrame(
        {
            "game_id": canonical,
            "stadium_id": frame["stadium_id"].to_numpy(),
            "roof": frame["roof"].to_numpy(),
        }
    )

    duplicated = facts["game_id"][facts["game_id"].duplicated()].tolist()
    if duplicated:
        raise DataIngestionError(
            f"the pinned schedules produced duplicate canonical game_id values "
            f"for seasons {sorted(seasons)}: {sorted(set(duplicated))[:10]}. Two "
            "rows claiming one game would make the join ambiguous and silently "
            "duplicate weather rows."
        )

    return facts


def _reconcile_stadium_id(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse the merge's ``stadium_id`` columns to exactly ONE, or refuse.

    RULING D2. Phase 33 Wave 12 puts ``stadium_id`` into silver ``games``
    (``33-12-PLAN.md:23``) and ``scripts/ingest_games.py:404-408`` already emits it
    on every transformed row, so the games frame this backfill loads may or may not
    already carry the column depending on whether that wave has run. A plain
    ``pd.merge`` against a facts frame that ALSO carries it yields ``stadium_id_x``
    and ``stadium_id_y`` and NO ``stadium_id`` -- and then ``row.stadium_id`` raises
    in the tracer, which is wave 2 of an eleven-wave strictly serial chain.

    So the merge states ``suffixes=("_silver", "_pinned")`` and this function runs
    before ANY row is read. Three input shapes, all three handled, because the plan
    must not depend on guessing which state Wave 12 leaves the tree in:

    * BOTH present (post-Wave-12): the two are compared row by row and a
      disagreement RAISES, naming every disagreeing ``game_id`` with both values.
      That is a real signal rather than a formality -- silver's ``stadium_id`` is a
      transform of the same pinned feed, so a disagreement means the pin moved or
      the transform is wrong, and either is a finding worth stopping on rather than
      routing a game on a coin flip.
    * Only the PINNED column, arriving unsuffixed because pandas suffixes only
      OVERLAPPING names (pre-Wave-12): the pinned value is taken and the fact that
      it was taken is LOGGED, naming Wave 12 as what will make the comparison live.
    * NEITHER: the merge did not supply the column at all, which is a caller bug,
      and it raises rather than failing later at attribute access.

    Args:
        frame: The merged games frame.

    Returns:
        The frame carrying exactly one ``stadium_id`` column and no suffixed
        leftovers.

    Raises:
        DataIngestionError: the two sides disagree, or neither is present.
    """
    frame = frame.copy()
    silver_col = "stadium_id_silver"
    pinned_col = "stadium_id_pinned"

    if silver_col in frame.columns and pinned_col in frame.columns:
        silver = frame[silver_col].astype("string")
        pinned = frame[pinned_col].astype("string")
        disagreeing = frame[silver != pinned]
        if not disagreeing.empty:
            detail = ", ".join(
                f"{row.game_id}: silver={getattr(row, silver_col)!r} "
                f"pinned={getattr(row, pinned_col)!r}"
                for row in disagreeing.head(20).itertuples()
            )
            raise DataIngestionError(
                f"silver and the pinned feed DISAGREE about stadium_id for "
                f"{len(disagreeing)} game(s): {detail}. Neither side is allowed to "
                "win silently. Silver's stadium_id is a transform of this same "
                "pinned feed, so a disagreement means the pin moved or the "
                "transform is wrong. Re-pin the upstream capture, or re-run "
                "scripts/ingest_games.py to rebuild silver from the current pin."
            )
        frame["stadium_id"] = frame[pinned_col]
        frame = frame.drop(columns=[silver_col, pinned_col])
        return frame

    if pinned_col in frame.columns:
        # Defensive: a silver frame that somehow produced only the suffixed pinned
        # name. Treated as the pre-Wave-12 case, and said out loud.
        frame["stadium_id"] = frame[pinned_col]
        frame = frame.drop(columns=[pinned_col])

    if "stadium_id" not in frame.columns:
        raise DataIngestionError(
            "the merged games frame carries no stadium_id column at all, in any "
            "form. load_pinned_game_facts supplies one, so this means the merge "
            "did not run or dropped it. Nothing is routed on a guess."
        )

    if silver_col in frame.columns:
        frame = frame.drop(columns=[silver_col])

    logger.info(
        "stadium_id taken from the PINNED feed with no silver value to reconcile "
        "against -- this is the pre-Wave-12 tree shape; Phase 33 Wave 12 adds "
        "stadium_id to silver games and makes the reconciliation live",
        games=len(frame),
    )
    return frame


def assert_weather_coverage_survived(frame: pd.DataFrame) -> None:
    """Refuse a promoted frame that lost ``weather_coverage`` at the schema gate.

    A POSITIVE STATEMENT ABOUT THE BOUNDARY, made in PRODUCTION rather than only
    under pytest. RESEARCH P-6: ``validate_bronze_to_silver`` rebuilds every row as
    ``schema_class(**row).model_dump()`` (``data/quality_gates.py:52-57``) and
    Pydantic v2 defaults to ``extra="ignore"``, so a field that stops being
    declared -- or is renamed on one side only -- disappears with NO error at all.
    The schema's own comment records that this already happened once, to the five
    Open-Meteo fields.

    A passing round-trip test proves the column survives TODAY. This proves it
    survived THIS RUN, which is the claim the silver store actually rests on.

    Raises:
        DataIngestionError: ``weather_coverage`` is absent from *frame*.
    """
    if "weather_coverage" in frame.columns:
        return
    raise DataIngestionError(
        "the promoted weather frame has NO weather_coverage column. It was "
        "emitted by the record factories and is gone after "
        "validate_bronze_to_silver, which means it is not declared on "
        'data.schemas.WeatherSchema: Pydantic v2\'s extra="ignore" DROPS any '
        "column the schema does not name, silently and without an error. Declare "
        "weather_coverage on WeatherSchema. Without it a dome and a missing "
        "observation are indistinguishable in silver, which is SPEC R4's defect."
    )


def promote_weather_bronze_to_silver(
    frame: pd.DataFrame,
    schema_class: type = WeatherSchema,
) -> pd.DataFrame:
    """Validate a bronze weather frame and ASSERT the coverage flag survived.

    The one promotion path. ``schema_class`` is a parameter so a test can drive a
    temporary subclass that drops the field and prove this refusal FIRES -- a check
    that has only ever been observed finding nothing is indistinguishable from one
    that is not wired up.
    """
    validated = validate_bronze_to_silver(frame, schema_class)
    assert_weather_coverage_survived(validated)
    return validated


class HistoricalWeatherBackfiller(WeatherDataIngester):
    """The archive-endpoint weather path, subclassing the live forecast ingester.

    SUBCLASS RATHER THAN COPY. Venue resolution under the D33-15 stadium_id routing,
    the nflverse roof map, and the record factories are SHARED behaviour, and two
    copies of them would eventually disagree about which stadium a neutral-site game
    is at -- the exact defect COLD-09 exists to fix. What this class adds is only the
    two things that are genuinely different: the archive fetch, and the archive
    ingest.
    """

    async def _fetch_openmeteo_weather(
        self,
        latitude: float,
        longitude: float,
        game_date: str,
        game_hour: int,
        venue_timezone: str,
    ) -> dict[str, Any] | str:
        """Open a client and fetch one game's ARCHIVE observation.

        The seam tests replace, mirroring
        :meth:`WeatherDataIngester._fetch_openmeteo_forecast`.

        Returns:
            A measurement dict, or the :data:`ABSENT_OBSERVATION` sentinel.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await fetch_game_weather(
                client, latitude, longitude, game_date, game_hour, venue_timezone
            )

    def _per_game_roof_is_outdoor(self, roof: object) -> bool:
        """Does weather apply to THIS GAME, from the feed's own ``roof`` value?

        R4's rule, and the mapping it rests on was already correct -- only the
        SOURCE of the value was wrong. ``_resolve_venue_for_game_row`` returned the
        VENUE's ``roof_type``, so a retractable stadium answered the same way for
        every game ever played in it and a game's own ``roof`` never reached the
        decision::

            feed roof    NFLVERSE_ROOF_MAP      _is_outdoor_game    weather applies
            "outdoors" -> "outdoor"          -> True                yes
            "open"     -> "retractable"      -> True                yes
            "closed"   -> "indoor"           -> False               no
            "dome"     -> "indoor"           -> False               no

        AN UNKNOWN VALUE RAISES rather than defaulting.
        ``_map_nflverse_roof_type`` falls back to ``"outdoor"`` on an unrecognised
        string, which is a sensible default for the live ingest and the wrong one
        here: it would send a game with a garbled roof to the network and record
        the result as an outdoor observation, silently. The corpus is exact.

        Raises:
            WeatherDataError: *roof* is outside ``NFLVERSE_ROOF_MAP``.
        """
        key = str(roof).lower().strip()
        if key not in NFLVERSE_ROOF_MAP:
            raise WeatherDataError(
                f"the pinned feed gives roof {roof!r}, which is not one of "
                f"{sorted(NFLVERSE_ROOF_MAP)}. It is NOT defaulted to 'outdoors': "
                "a defaulted roof would fetch weather for a game played under a "
                "closed one, or skip the fetch for a game played in the open, and "
                "either lands in the corpus as a fact. Re-pin the upstream "
                "capture."
            )
        return self._is_outdoor_game(NFLVERSE_ROOF_MAP[key])

    def _prepare_games_frame(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Validate every game_id, join the pinned facts, reconcile ``stadium_id``.

        EVERYTHING THAT CAN REFUSE HAPPENS HERE, before the first request of the
        season is issued. A refusal that fires after 140 fetches has already spent
        budget that R3's resume cannot give back.
        """
        malformed = [
            str(value)
            for value in games_df["game_id"]
            if not is_valid_game_id(str(value))
        ]
        if malformed:
            seasons = sorted({int(v) for v in games_df["season"].unique()})
            raise WeatherDataError(
                f"invalid game_id(s) in the backfill frame for season(s) "
                f"{seasons}: {malformed[:20]}. They are NOT skipped (Ruling D4). "
                "Over a 4,847-game paced run a skip is a silent hole: the season "
                "coverage test would later report the season incomplete for a "
                "reason no log line explains, and the operator's next move would "
                "be to re-run the season and spend the budget again. The corpus "
                "is exact. Re-pin the upstream capture, or correct the offending "
                "row in data/silver/games.parquet, and re-run."
            )

        seasons = sorted({int(value) for value in games_df["season"].unique()})
        facts = load_pinned_game_facts(seasons)

        # EXPLICIT SUFFIXES so neither side can win silently (Ruling D2). `roof` is
        # JOINED because silver has never carried it; `stadium_id` is RECONCILED
        # below because silver may already carry it.
        if "roof" in games_df.columns:
            raise DataIngestionError(
                "the silver games frame already carries a `roof` column. This "
                "backfill joins `roof` from the pinned feed on the recorded fact "
                "that silver has never had one (Ruling D, N-07), and the "
                "reconciliation rule below covers stadium_id only. A silver "
                "`roof` needs a deliberate decision about which side wins, not a "
                "merge suffix. Stop and make it."
            )

        merged = games_df.merge(
            facts,
            on="game_id",
            how="left",
            suffixes=("_silver", "_pinned"),
        )

        unjoined = merged[merged["roof"].isna()]
        if not unjoined.empty:
            raise DataIngestionError(
                f"{len(unjoined)} game(s) in season(s) {seasons} did not join to "
                f"the pinned schedules: "
                f"{sorted(str(v) for v in unjoined['game_id'].head(20))}. Every "
                "game must carry its own roof and stadium_id; a game that does "
                "not join would be routed and branched on a default. The join key "
                "is the CANONICAL game_id, rebuilt from the pinned feed through "
                "utils.game_id_utils.create_standard_game_id -- the pinned feed's "
                "own game_id is nflverse-shaped and matches no silver row."
            )

        return _reconcile_stadium_id(merged)

    def fetch_weather_for_games(
        self,
        games_df: pd.DataFrame,
        venues_df: pd.DataFrame,
        forecast_time: datetime | None = None,
        *,
        request_interval_seconds: float = 0.0,
    ) -> pd.DataFrame:
        """Fetch an ERA5 observation for every game that needs one.

        THE CORRECTED PATH (Plan 33.1-02). Four things changed and each one was a
        wrong answer arriving silently:

        1. ROUTING is ``stadium_id``, unconditionally, for every season and every
           game -- no neutral-site test and no home-team fallback. The 141 Oakland
           Coliseum games are NOT neutral-site, so the old conjunction resolved
           them to Allegiant, 650 km away.
        2. THE ROOF BRANCH reads the GAME's own ``roof``, not the venue's, so a
           ``closed`` game and an ``open`` game at one stadium differ.
        3. THE HOUR comes from :func:`resolve_venue_local_hour` and the request is
           made in the venue's own IANA zone. It used to feed the UTC hour into an
           array requested in a fixed Eastern zone.
        4. AN ABSENT OBSERVATION is a recorded state with every column NULL and
           ``weather_coverage`` False, routed by the payload predicate inside
           ``fetch_game_weather`` -- not a row that looks like a measurement.

        An indoor game is written with NO network call. An outdoor game whose fetch
        fails raises; nothing is skipped and nothing is imputed.

        Args:
            games_df: The games to fetch. Needs ``game_id``, ``season`` and
                ``kickoff_et``; ``stadium_id`` is optional (see Ruling D2).
            venues_df: The venue table, carrying ``stadium_id`` and ``timezone``.
            forecast_time: When this run was taken. Defaults to now, in UTC.
            request_interval_seconds: Minimum wall-clock gap between requests. The
                free tier's binding cap is 5,000 calls per HOUR with no
                rate-limit header to read, so a run must pace itself against its
                own clock. INTERIM: Plan 33.1-05 moves pacing and a weighted
                ``CallBudget`` inside ``fetch_game_weather`` -- immediately before
                ``client.get``, so a retried game is counted three times -- at
                which point this parameter is redundant and should be removed.

        Returns:
            One record per game, in the order the games arrived.
        """
        if forecast_time is None:
            # tz-aware UTC -- Phase 15-04 requires storage writers to provide
            # timezone-aware datetimes. Storage will reject naive datetimes
            # at write time via _normalize_parquet_datetime_columns.
            forecast_time = datetime.now(UTC)

        # Every refusal fires here, before the first request of the season.
        prepared = self._prepare_games_frame(games_df)

        logger.info(
            "Fetching ARCHIVE weather for games",
            games=len(prepared),
            interval_seconds=request_interval_seconds,
        )

        weather_records = []
        last_request_at: float | None = None

        # `iterrows` rather than `itertuples`: `resolve_venue_local_hour` reads the
        # row with `game["kickoff_et"]`, which a namedtuple cannot answer. A Series
        # answers both that and the `row.stadium_id` attribute access below, which
        # is the access Ruling D2's reconciliation exists to keep working.
        for _, game in prepared.iterrows():
            game_id = str(game["game_id"])

            # UNCONDITIONAL stadium_id routing. An id absent from the venue table
            # raises by name inside this call; it is never resolved to the home
            # team's stadium, which for 1,082 historical games is a stadium built
            # after the game was played.
            venue = self._get_venue_record_by_stadium_id(game["stadium_id"], venues_df)

            # The venue-LOCAL day and hour, in the venue's own zone. No forecast
            # horizon rides along -- see resolve_venue_local_hour's docstring.
            selected = resolve_venue_local_hour(game, venue)

            # ONE if/elif/else, and NO `continue` anywhere in this loop. The three
            # branches are exhaustive and every one of them APPENDS, so the frame
            # carries exactly one row per game by construction.
            #
            # The absence of `continue` is also mechanical rather than stylistic:
            # Plan 33.1-02's source scan reads a bare `continue` in this loop as
            # Ruling D4's silent skip, and it cannot tell "go to the next game"
            # from "drop this game". Leaving it nothing to misread is cheaper than
            # teaching the scan to parse intent, and an exhaustive branch is the
            # clearer shape anyway.
            if not self._per_game_roof_is_outdoor(game["roof"]):
                weather_records.append(
                    self._create_indoor_weather_record(
                        game_id,
                        selected.kickoff_utc,
                        forecast_time,
                        weather_source=WEATHER_SOURCE_ARCHIVE,
                    )
                )
            else:
                if request_interval_seconds > 0 and last_request_at is not None:
                    elapsed = time.monotonic() - last_request_at
                    if elapsed < request_interval_seconds:
                        time.sleep(request_interval_seconds - elapsed)
                last_request_at = time.monotonic()

                # If this fails, WeatherDataError propagates (hard-fail).
                weather_data = asyncio.run(
                    self._fetch_openmeteo_weather(
                        venue["latitude"],
                        venue["longitude"],
                        selected.local_date,
                        selected.hour,
                        selected.timezone,
                    )
                )

                if isinstance(weather_data, str) and weather_data == ABSENT_OBSERVATION:
                    weather_records.append(
                        self._create_absent_observation_record(
                            game_id,
                            selected.kickoff_utc,
                            forecast_time,
                            weather_source=WEATHER_SOURCE_ARCHIVE,
                        )
                    )
                else:
                    weather_records.append(
                        self._create_weather_record(
                            game_id,
                            selected.kickoff_utc,
                            forecast_time,
                            weather_data,
                            # The GAME's roof, mapped -- not the venue's roof_type.
                            NFLVERSE_ROOF_MAP[str(game["roof"]).lower().strip()],
                            weather_source=WEATHER_SOURCE_ARCHIVE,
                        )
                    )

        weather_df = pd.DataFrame(weather_records)

        logger.info(
            "Weather data fetched",
            games=len(prepared),
            weather_records=len(weather_df),
        )

        return weather_df

    def backfill_weather(
        self,
        season: int | None = None,
        week: int | None = None,
        forecast_time: datetime | None = None,
        base_path=None,
    ) -> pd.DataFrame:
        """
        Full weather data ingestion pipeline.

        Uses Bronze/Silver separation:
        - Bronze: timestamped append-only snapshots via save_bronze_snapshot
        - Silver: validated, upserted via validate_bronze_to_silver + upsert_silver

        Args:
            season: Season to ingest (default: current)
            week: Week to ingest (default: current), None for entire season
            forecast_time: Time when forecast was made

        Returns:
            Ingested and validated weather data
        """
        # Handle default season
        if season is None:
            current_season, current_week = get_current_nfl_week()
            season = current_season
            # If no week specified, default to current week for current season
            if week is None:
                week = current_week

        if forecast_time is None:
            # tz-aware UTC -- Phase 15-04 requires storage writers to provide
            # timezone-aware datetimes. Storage will reject naive datetimes
            # at write time via _normalize_parquet_datetime_columns.
            forecast_time = datetime.now(UTC)

        scope = f"Week {week}" if week is not None else "Entire Season"
        logger.info(
            "Starting weather data ingestion",
            season=season,
            week=week,
            scope=scope,
            forecast_time=forecast_time.isoformat(),
        )

        try:
            # Load required data
            venues_df = self._load_venue_data()
            games_df = self._load_games_data(season, week)

            if games_df.empty:
                logger.warning("No games found", season=season, week=week)
                return pd.DataFrame()

            # Fetch weather data (hard-fails on missing outdoor weather)
            weather_df = self.fetch_weather_for_games(
                games_df, venues_df, forecast_time
            )

            if weather_df.empty:
                logger.warning("No weather data fetched")
                return weather_df

            # Save Bronze snapshot (append-only, timestamped)
            if week is not None:
                save_bronze_snapshot(
                    weather_df,
                    "weather",
                    season=season,
                    week=week,
                    base_path=base_path,
                )
            else:
                # For full-season ingestion, use week=0 as a sentinel
                save_bronze_snapshot(
                    weather_df,
                    "weather",
                    season=season,
                    week=0,
                    base_path=base_path,
                )

            # Validate through quality gate (hard-fail on bad rows) AND assert the
            # coverage flag survived the schema's extra="ignore" (RESEARCH P-6).
            validated_df = promote_weather_bronze_to_silver(weather_df)

            # Ensure no duplicate weather records per game
            initial_count = len(validated_df)
            validated_df = validated_df.drop_duplicates(
                subset=["game_id"], keep="first"
            )
            final_count = len(validated_df)

            if initial_count != final_count:
                logger.info(
                    "Deduplicated weather records",
                    initial_records=initial_count,
                    final_records=final_count,
                    duplicates_removed=initial_count - final_count,
                )

            # Add metadata timestamp
            validated_df["created_at"] = datetime.now(UTC)

            # Upsert to Silver (latest wins by game_id)
            upsert_silver(validated_df, "weather", base_path=base_path)

            log_data_operation(
                operation="ingest",
                table="weather",
                rows=len(validated_df),
                season=season,
                week=week,
            )

            logger.info(
                "Weather data ingestion completed successfully",
                total_records=len(validated_df),
                unique_games=len(validated_df),
                outdoor_games=int(validated_df["is_outdoor"].sum()),
                season=season,
                week=week,
            )

            return validated_df

        except DataIngestionError:
            raise
        except (httpx.HTTPStatusError, httpx.TimeoutException) as e:
            logger.error("Weather data ingestion failed", error=str(e))
            raise DataIngestionError(f"Weather ingestion failed: {e}")


def stamp_weather_source_on_existing_rows(
    table: str = "weather",
    base_path=None,
    *,
    source: str = WEATHER_SOURCE_ARCHIVE,
) -> pd.DataFrame:
    """Stamp `weather_source` onto silver weather rows that predate the column.

    A ONE-SHOT PROVENANCE BACKFILL, not an ingest. It reads the silver table, fills
    only the rows whose `weather_source` is absent or null, and writes the frame
    back through the same storage-layer path every other silver write uses. It
    fetches nothing and it changes no weather VALUE -- only the column that says
    where each value came from.

    Every pre-existing row is stamped `archive` because every pre-existing row was
    produced by the archive endpoint: those fourteen rows are 2024 Week 6, written
    by the only weather ingest this project has ever run, and that ingest hit
    `ARCHIVE_ENDPOINT_URL`. This is a statement of fact about how they were made,
    not a default.

    Args:
        table: Silver table name.
        base_path: Data root. Threaded so a test can redirect the read and write.
        source: The provenance value to stamp. Validated against the closed
            vocabulary before anything is written.

    Returns:
        The frame as written.

    Raises:
        ValueError: *source* is outside the closed vocabulary.
    """
    from pathlib import Path

    from conf.settings import get_settings

    validate_weather_source(source)

    # Read the parquet DIRECTLY rather than through `load_dataframe`, which has no
    # `base_path` and would resolve the production root regardless of what a caller
    # passed. A sandbox argument that is silently ignored is worse than none: it
    # reads as a guarantee and behaves as a comment.
    root = (
        Path(base_path)
        if base_path is not None
        else Path(get_settings().config.data.root_path)
    )
    silver_path = root / "silver" / f"{table}.parquet"
    if not silver_path.is_file():
        raise DataIngestionError(
            f"no silver weather table at {silver_path.as_posix()} to stamp."
        )
    frame = pd.read_parquet(silver_path, engine="pyarrow")
    if "weather_source" not in frame.columns:
        frame["weather_source"] = None

    unstamped = frame["weather_source"].isna()
    filled = int(unstamped.sum())
    frame.loc[unstamped, "weather_source"] = source

    upsert_silver(frame, table, base_path=base_path)
    logger.info(
        "Stamped weather_source on pre-existing rows",
        table=table,
        rows=len(frame),
        filled=filled,
        source=source,
    )
    return frame


def main():
    """CLI entry point for the HISTORICAL (archive-endpoint) weather backfill.

    Separately named from `scripts/ingest_weather.py` on purpose: the two commands
    reach different endpoints with different temporal meanings, and one command
    with a mode flag would put them one typo apart.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Backfill HISTORICAL NFL weather from the Open-Meteo ARCHIVE endpoint. "
            "For an unplayed game use scripts/ingest_weather.py, which reaches the "
            "FORECAST endpoint -- the archive cannot answer for a game that has not "
            "been played."
        )
    )
    parser.add_argument("--season", type=int, help="Season to backfill")
    parser.add_argument(
        "--week", type=int, default=None, help="Week to backfill (default: all)"
    )
    parser.add_argument(
        "--stamp-weather-source",
        action="store_true",
        help=(
            "Do not fetch anything. Stamp weather_source='archive' onto silver "
            "weather rows that predate the column, and exit."
        ),
    )

    args = parser.parse_args()

    from utils import setup_logging

    setup_logging()

    if args.stamp_weather_source:
        frame = stamp_weather_source_on_existing_rows()
        print(f"Stamped weather_source on {len(frame)} silver weather rows")
        return

    if args.season is None:
        print("--season is required unless --stamp-weather-source is given")
        sys.exit(1)

    try:
        backfiller = HistoricalWeatherBackfiller()
        weather_df = backfiller.backfill_weather(season=args.season, week=args.week)
        if weather_df.empty:
            print("No weather data backfilled")
            return
        print(f"Backfilled {len(weather_df)} weather records from the ARCHIVE endpoint")
        print(f"Season: {args.season}, Week: {args.week or 'all'}")
    except (DataIngestionError, WeatherDataError, OSError, ValueError) as exc:
        logger.error("Historical weather backfill failed", error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
