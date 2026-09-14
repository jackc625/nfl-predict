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
import contextlib
import fnmatch
import hashlib
import json
import os
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from data.storage import save_bronze_snapshot, upsert_silver
from data.upstream_pin import SEALED_THROUGH_SEASON, load_schedules
from scripts.ingest_weather import (
    HOURLY_VARIABLES,
    NFLVERSE_ROOF_MAP,
    WEATHER_SOURCE_VOCABULARY,
    WeatherDataIngester,
    normalize_wind_direction,
    resolve_venue_local_hour,
    validate_weather_source,
)
from scripts.weather_crosscheck_constants import (
    ABSENT_FROM_LEGACY_COLUMNS,
    COMPARABLE_SEASONS,
    COMPARED_COLUMNS,
    CROSSCHECK_TOLERANCE_F,
    JOIN_KEY,
    NON_COMPARABLE_INTERSECTING_COLUMNS,
    PREREGISTRATION_PATH,
)
from utils import (
    DataIngestionError,
    get_logger,
    log_data_operation,
)
from utils.exceptions import WeatherDataError
from utils.game_id_utils import (
    create_standard_game_id,
    is_valid_game_id,
    parse_game_id,
)

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

# THE CORPUS THIS BACKFILL OWNS: the SEALED upstream zone, 2002 through 2025.
#
# The upper bound is `data.upstream_pin.SEALED_THROUGH_SEASON`, IMPORTED rather
# than transcribed, and it is the right bound for a structural reason rather than
# a convenient one. Phase 32 split the pin into a SEALED zone (immutable history)
# and a LIVE zone (2026, append-only, forward). This module reads an ERA5
# REANALYSIS product -- it describes weather that has already occurred -- so the
# corpus it can honestly answer for is exactly the sealed zone. The live season
# belongs to `scripts/ingest_weather.py`'s FORECAST endpoint, and pointing the
# archive at an unplayed game is the category error D33-26 quarantined this
# endpoint over.
#
# Deriving the bound from the pin rather than from a wall clock also means the
# range cannot drift out from under the coverage rule on 1 January.
CORPUS_FIRST_SEASON: int = 2002
CORPUS_LAST_SEASON: int = SEALED_THROUGH_SEASON

# THE TEN PRE-EXISTING BRONZE WEATHER FILES, BY NAME (SPEC prohibition 6).
#
# They hold 1,942 legacy rows plus 106 more, and they are the ONLY evidence the
# routing fix can be regression-tested against. They are listed here so the
# disjointness assertion immediately below has something concrete to check, and so
# the cross-check comparator reads ONE list rather than re-globbing a pattern that
# could widen.
#
# Their measured row counts, column counts and content digests live in
# `tests.phase33_state.LEGACY_WEATHER_BRONZE_INVENTORY`.
LEGACY_WEATHER_BRONZE_FILENAMES: tuple[str, ...] = (
    "weather_raw_bronze_2018_season.parquet",
    "weather_raw_bronze_2019_season.parquet",
    "weather_raw_bronze_2020_season.parquet",
    "weather_raw_bronze_2021_season.parquet",
    "weather_raw_bronze_2022_season.parquet",
    "weather_raw_bronze_2023_season.parquet",
    "weather_raw_bronze_2024_season.parquet",
    "weather_raw_bronze_2025_W05.parquet",
    "weather_raw_bronze_2024_W06_20260416T165923.parquet",
    "weather_raw_bronze_2025_W00_20260407T025733.parquet",
)

# The glob the resume rule reads. Disjoint from every legacy filename BY
# CONSTRUCTION -- checked at import, not assumed, because "by construction" is only
# worth saying when something verifies it.
BACKFILL_BRONZE_GLOB: str = f"{BACKFILL_BRONZE_TABLE}_raw_bronze_*"

_COLLIDING_LEGACY_FILENAMES = tuple(
    name
    for name in LEGACY_WEATHER_BRONZE_FILENAMES
    if fnmatch.fnmatch(name, f"{BACKFILL_BRONZE_GLOB}.parquet")
)
if _COLLIDING_LEGACY_FILENAMES:  # pragma: no cover -- an import-time impossibility
    raise RuntimeError(
        f"the backfill bronze glob {BACKFILL_BRONZE_GLOB!r} matches legacy weather "
        f"bronze filenames {_COLLIDING_LEGACY_FILENAMES}. SPEC prohibition 6 is "
        "satisfied by the two name spaces being DISJOINT; a collision means a new "
        "run could overwrite the only evidence the routing fix can be "
        "regression-tested against."
    )

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


# ---------------------------------------------------------------------------
# The run's self-imposed interval between archive requests.
# ---------------------------------------------------------------------------

# THE DECLARED MINIMUM WALL-CLOCK INTERVAL BETWEEN ARCHIVE REQUESTS.
#
# The arithmetic, so the number is a conclusion rather than a preference. The
# corpus needs 4,847 fetches; at the extrapolated 1.2 weighted calls per request
# that is about 5,816 weighted calls, which fits the free tier's 10,000/day cap
# with roughly 42% headroom and does NOT fit its 5,000/HOUR cap. Staying under the
# hourly cap therefore requires at least 5,816 / 5,000 = 1.164 hours, i.e. about 70
# minutes, i.e. a mean inter-request interval of at least 0.87 s. 1.0 s gives about
# 81 minutes with margin.
#
# MEASURED, for contrast: the steady-state request latency is 0.134 s median (Plan
# 33.1-02's tracer, 200 real requests). Unpaced, the corpus would finish in about
# eleven minutes and present roughly 31,700 weighted calls inside that hour.
ARCHIVE_REQUEST_INTERVAL_SECONDS: float = 1.0

# THE FREE TIER'S PUBLISHED LIMITS, scraped VERBATIM from the vendor's pricing
# table. 600/minute, 5,000/HOUR, 10,000/day, 300,000/month. The two the run can
# plausibly reach in one sitting are declared here; the minutely cap is far above
# a 1.0 s interval and the monthly cap is 50x one corpus run.
#
# THE HOURLY CAP IS THE BINDING ONE, and that is the whole reason this module
# holds itself to a clock. See ARCHIVE_REQUEST_INTERVAL_SECONDS above for the
# arithmetic.
ARCHIVE_HOURLY_CALL_BUDGET: int = 5000
ARCHIVE_DAILY_CALL_BUDGET: int = 10000

# A REQUEST IS NOT ALWAYS ONE CALL. The vendor states -- CITED -- that "requests
# for data covering more than 10 weather variables ... are considered multiple API
# calls", with fractional counts, and gives ONE worked example: 15 variables over
# 14 days = 1.5 calls. `HOURLY_VARIABLES` sends TWELVE variables for ONE day, so
# 12/10 = 1.2 is EXTRAPOLATED from that single example rather than read off a
# published formula. Logged as assumption A1.
#
# THE CONSEQUENCE, recorded here so nobody has to re-derive it: 4,847 fetches at
# 1.2 is roughly 5,816 weighted calls. That fits the DAILY cap with about 42%
# headroom and does NOT fit the HOURLY cap, so the minimum run time is about 70
# minutes and the declared 1.0 s interval gives about 81 with margin.
ARCHIVE_CALL_WEIGHT_PER_REQUEST: float = 1.2


class CallBudgetExceededError(DataIngestionError):
    """The next request would breach a declared cap. Nothing was issued."""


class CallBudget:
    """Count this run's WEIGHTED calls and refuse BEFORE a breach.

    THE COUNTER IS THE ONLY INSTRUMENT THERE IS. The archive response carries no
    rate-limit header at all -- a live header scan of a successful response
    returned nothing matching ``rate``, ``request`` or ``limit`` -- so the
    Phase-29 pattern of reading ``x-requests-last`` back from the provider has no
    analogue here. A run that miscounts has no way to find out.

    THE DEBIT IS TAKEN PER ATTEMPT, NOT PER GAME, and :func:`fetch_game_weather`
    is where it is taken. That function is wrapped in
    ``@retry(stop=stop_after_attempt(3))``, so everything a CALLER does happens
    once per logical game while the request happens up to three times. A
    caller-side debit would undercount by up to 3x against a 5,000/hour binding
    cap: a 10% retry rate would issue roughly 5,331 real requests while the
    counter reported 4,847.

    :meth:`acquire` is the refusal point. It raises BEFORE the request when the
    next debit would breach either cap, so a breach cannot be discovered after
    the fact.
    """

    def __init__(
        self,
        *,
        weight: float = ARCHIVE_CALL_WEIGHT_PER_REQUEST,
        hourly_budget: float = ARCHIVE_HOURLY_CALL_BUDGET,
        daily_budget: float = ARCHIVE_DAILY_CALL_BUDGET,
        resume_hint: str = "",
    ) -> None:
        self.weight = float(weight)
        self.hourly_budget = hourly_budget
        self.daily_budget = daily_budget
        self.resume_hint = resume_hint
        self.weighted_total = 0.0
        self.debits = 0
        self._window: list[tuple[float, float]] = []

    @property
    def hourly_weighted_total(self) -> float:
        """The weighted calls issued inside the trailing hour."""
        cutoff = time.monotonic() - 3600.0
        self._window = [entry for entry in self._window if entry[0] > cutoff]
        return sum(weight for _, weight in self._window)

    def _refuse(self, scope: str, projected: float, cap: float) -> None:
        hint = f" {self.resume_hint}." if self.resume_hint else ""
        raise CallBudgetExceededError(
            f"the next archive request would take this run's {scope} weighted call "
            f"total to {projected:.1f}, past the declared cap of {cap}. NO request "
            "was issued. The weight is an EXTRAPOLATION (12 variables / 10 = 1.2) "
            "and the archive sends no rate-limit header, so this counter is the "
            "only instrument there is -- it refuses early rather than discovering "
            "a breach from a provider error. The run is RESUMABLE: wait for the "
            "limit window to clear and re-invoke, and only the seasons still "
            f"missing will be fetched.{hint}"
        )

    def acquire(self, weight: float | None = None) -> float:
        """Debit one ATTEMPT, or refuse by name before it is issued."""
        debit = self.weight if weight is None else float(weight)

        projected_hour = self.hourly_weighted_total + debit
        if projected_hour > self.hourly_budget:
            self._refuse("hourly", projected_hour, self.hourly_budget)

        projected_run = self.weighted_total + debit
        if projected_run > self.daily_budget:
            self._refuse("daily", projected_run, self.daily_budget)

        self._window.append((time.monotonic(), debit))
        self.weighted_total = projected_run
        self.debits += 1
        return self.weighted_total


class RequestThrottle:
    """Hold the run to a minimum wall-clock interval between archive requests.

    The free tier's BINDING limit is 5,000 calls per HOUR and the archive response
    carries no rate-limit header at all -- a live header scan of a successful
    response returned nothing matching ``rate``, ``request`` or ``limit`` -- so the
    run must hold itself to its own clock. There is no counter to read back.

    ``interval_seconds=0`` disables the wait, which is what a test wants: a stubbed
    season must not take one second per game.
    """

    def __init__(self, interval_seconds: float) -> None:
        self.interval_seconds = float(interval_seconds)
        # THE CLOCK STARTS AT CONSTRUCTION, so the FIRST attempt also waits. N
        # attempts then take at least N intervals rather than N-1, which matters
        # exactly when a run is re-invoked immediately after a budget refusal: the
        # provider's window has not moved, and a free first request is the one most
        # likely to land inside it.
        self._last_at: float = time.monotonic()
        self.waits = 0

    def sleep_until_due(self) -> None:
        """Block until the declared interval has elapsed since the last request."""
        self.waits += 1
        if self.interval_seconds > 0:
            elapsed = time.monotonic() - self._last_at
            if elapsed < self.interval_seconds:
                time.sleep(self.interval_seconds - elapsed)
        self._last_at = time.monotonic()


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
#
# WHY THE RETRY CANNOT SUBSTITUTE FOR THE INTERVAL (Ruling L2). Three attempts
# with at most about 30 s of backoff cannot clear an HOURLY limit, so a limit hit
# mid-season burns the retries and hard-fails the season. That is survivable only
# because the run is resumable per season, and it is the second reason to hold to
# an interval rather than to lean on the retry. Note also that the backoff now
# COMPOSES with the interval rather than replacing it: a retried game takes longer
# than three intervals, so the projected wall clock is a FLOOR, not an estimate.
#
# A BUDGET REFUSAL IS NOT RETRIED. `retry_if_not_exception_type` excludes
# `CallBudgetExceededError` because a cap breach is not transient: two more
# attempts at 2 s and 4 s cannot clear an hourly window, and each would re-enter
# this function only to refuse again.
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_not_exception_type(CallBudgetExceededError),
    reraise=True,
)
async def fetch_game_weather(
    client: httpx.AsyncClient,
    latitude: float,
    longitude: float,
    game_date: str,
    game_hour: int,
    venue_timezone: str,
    *,
    budget: "CallBudget | None" = None,
    throttle: "RequestThrottle | None" = None,
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
        budget: The run's :class:`CallBudget`. Debited HERE, on the line
            immediately before the request, and NEVER by a caller. THE PLACEMENT
            IS LOAD-BEARING -- see the comment block above the decorator, and do
            not "tidy" the debit up into the calling loop: this function is
            retried up to three times per logical game, so a caller-side debit
            undercounts by up to 3x against a 5,000/hour binding cap, and this
            counter is the only instrument there is.
        throttle: The run's :class:`RequestThrottle`, for the same reason and with
            the same placement. A wait taken by the caller would separate the
            GAMES and leave the retried attempts spaced only by tenacity's own
            backoff, whose maximum is 30 s and whose purpose is a transient
            failure rather than a rate limit.

    Returns:
        A measurement dict matching the system schema, or the
        :data:`ABSENT_OBSERVATION` sentinel when
        :func:`observation_is_absent` fires. The caller BRANCHES on the sentinel
        and routes it to ``_create_absent_observation_record``; that branch is what
        makes the absent path reachable from a real response rather than only from
        a test that calls the factory directly.

    Raises:
        CallBudgetExceededError: the attempt would breach a declared cap. Raised
            BEFORE anything is issued, and deliberately not retried.
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

    # RULING L2. These two lines are INSIDE the retried function, immediately
    # before the request, and that is the entire point of them being here: this
    # body runs once per ATTEMPT, while everything a caller does runs once per
    # logical GAME. Three retried attempts must cost three debits and three waits.
    if budget is not None:
        budget.acquire()
    if throttle is not None:
        throttle.sleep_until_due()

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


def _resolve_data_root(base_path: Any = None) -> Path:
    """The data lake root to read and write under.

    ``base_path`` is threaded through every function in this module for the reason
    ``stamp_weather_source_on_existing_rows`` already records: a sandbox argument
    that is silently ignored reads as a guarantee and behaves as a comment.
    """
    if base_path is not None:
        return Path(base_path)
    from conf.settings import get_settings

    return Path(get_settings().config.data.root_path)


def pinned_season_game_ids(season: int) -> frozenset[str]:
    """Every canonical ``game_id`` the pinned feed holds for *season*.

    Derived through :func:`load_pinned_game_facts` so there is exactly ONE
    derivation of the canonical id in this module. The pinned feed's own
    ``game_id`` is nflverse-shaped and shares zero values with silver's form, so a
    second, simpler derivation here would be a second chance to get that wrong.
    """
    return frozenset(
        str(value) for value in load_pinned_game_facts((season,))["game_id"]
    )


def season_is_covered(season: int, base_path: Any = None) -> bool:
    """Is every pinned game of *season* present in this corpus's bronze snapshots?

    RULING K, AND THE REASON IT IS NOT A FILENAME TEST. Two obvious rules both
    fail, and the next reader's instinct will be to simplify this back into one of
    them:

    * "This season is done iff a ``_W00_`` snapshot exists for it" SKIPS 2025.
      ``data/bronze/weather_raw_bronze_2025_W00_20260407T025733.parquet`` is a
      whole-season ``week=0`` sentinel written by the current code, and it holds 78
      rows covering weeks 1 to 5 ONLY -- ``_load_games_data(2025, None)`` read a
      silver ``games`` table that held 78 rows of 2025 at the time. That rule leaves
      **207 games unfetched behind a green log**. It is defect N-06.
    * "Done iff the union of this season's snapshots covers every game id" fixes
      2025 but marks 2018-2024 done from the SEVEN LEGACY
      ``weather_raw_bronze_{season}_season.parquet`` files, skipping the re-fetch
      under corrected routing that is the entire purpose of SPEC R2.

    So coverage is measured under :data:`BACKFILL_BRONZE_TABLE` ONLY -- a name
    disjoint from all ten legacy filenames by construction -- and the test is
    COVERAGE rather than existence, so an interrupted partial write cannot be
    mistaken for a completed season.

    Args:
        season: The season to test. Must lie inside the corpus range.
        base_path: Data lake root. Threaded so a test can redirect the read.

    Returns:
        True only when the union of this corpus's snapshots for *season* is a
        superset of the pinned game ids for *season*.
    """
    assert_season_in_corpus(season)
    bronze = _resolve_data_root(base_path) / "bronze"

    covered: set[str] = set()
    for path in sorted(
        bronze.glob(f"{BACKFILL_BRONZE_TABLE}_raw_bronze_{season}_*.parquet")
    ):
        frame = pd.read_parquet(path, columns=["game_id"], engine="pyarrow")
        covered.update(str(value) for value in frame["game_id"])

    return pinned_season_game_ids(season).issubset(covered)


def seasons_still_missing(base_path: Any = None) -> tuple[int, ...]:
    """The corpus seasons this backfill has NOT yet covered, ascending.

    The resume rule, and the reason an interrupted run costs at most one season
    rather than a whole day of budget.
    """
    return tuple(
        season
        for season in range(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON + 1)
        if not season_is_covered(season, base_path=base_path)
    )


def assert_season_in_corpus(season: int) -> int:
    """Refuse a season outside ``[CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON]`` by name.

    SPEC edge boundary/R3, fired BEFORE any request is issued. The upper bound is
    the sealed zone rather than the calendar: the archive is a reanalysis product
    and the live season belongs to the forecast path.

    Raises:
        DataIngestionError: *season* is outside the corpus.
    """
    if not CORPUS_FIRST_SEASON <= int(season) <= CORPUS_LAST_SEASON:
        raise DataIngestionError(
            f"season {season} is outside the historical weather corpus "
            f"[{CORPUS_FIRST_SEASON}, {CORPUS_LAST_SEASON}]. The upper bound is the "
            "SEALED upstream zone (data.upstream_pin.SEALED_THROUGH_SEASON), not "
            "the calendar: this module reads an ERA5 REANALYSIS product, which "
            "cannot answer for a game that has not been played. For a live-season "
            "game use scripts/ingest_weather.py, which reaches the FORECAST "
            "endpoint."
        )
    return int(season)


# ---------------------------------------------------------------------------
# Ruling L4: the three-way null-observation classification.
# ---------------------------------------------------------------------------

# The arm names, declared once so the state manifest and the refusal messages
# cannot drift apart from the code that decides them.
NULL_ARM_CLEAN: str = "CLEAN"
NULL_ARM_AMBIGUOUS: str = "AMBIGUOUS"
NULL_ARM_OUTAGE_LIKE: str = "OUTAGE_LIKE"
NULL_OBSERVATION_ARMS: tuple[str, ...] = (
    NULL_ARM_CLEAN,
    NULL_ARM_AMBIGUOUS,
    NULL_ARM_OUTAGE_LIKE,
)

# At or below this fraction an absence is ISOLATED and the season is written with
# no override. R4 explicitly makes a missing observation RECORDABLE DATA, and the
# whole D33.1-07 absent-observation state exists for exactly that case, so a flat
# bar that refused every season with two honest ERA5 gaps would be the opposite of
# what this phase is for.
ISOLATED_NULL_FRACTION_CEILING: float = 0.01

# Strictly above this fraction the pattern is OUTAGE-LIKE on the fraction alone.
OUTAGE_NULL_FRACTION_FLOOR: float = 0.25

# ... and a contiguous run of at least this many absences IN FETCH ORDER is
# OUTAGE-LIKE regardless of the fraction. THIS IS THE DISCRIMINATING TEST, and it
# is why the gate is not a fraction alone: RESEARCH A2's unprobed assumption is
# that limit exhaustion returns a 400 carrying a `reason`, and if the provider
# instead answers 200-with-null-arrays then the absences arrive CONSECUTIVELY,
# while genuine ERA5 gaps scatter across a season by geography and date. A 30%
# scattered season and a 30% consecutive season are different facts.
OUTAGE_CONTIGUOUS_RUN: int = 8

# The number of decimals the observed fraction is reported and compared at. An
# operator copies the printed number back in on --accept-null-fraction, so the two
# have to be the same number of digits or a correct override would still refuse.
NULL_FRACTION_DECIMALS: int = 6

# Where an ACCEPTED override is durably recorded. An override nobody can find
# afterwards is indistinguishable from no gate at all, so it lands on disk beside
# the corpus it authorised rather than only in a log line that scrolls away.
NULL_OVERRIDE_LEDGER_PATH: Path = Path("bronze") / ".weather_backfill_overrides.jsonl"


def classify_null_observations(
    absent_flags: Iterable[bool],
) -> tuple[str, dict[str, Any]]:
    """Classify a season's absence pattern as CLEAN, AMBIGUOUS or OUTAGE_LIKE.

    *absent_flags* is one boolean per game THAT WAS ACTUALLY FETCHED, in fetch
    order. A dome was never asked, so including it would dilute the fraction and
    make the gate report a number it did not measure.

    The order of the arms is load-bearing: the OUTAGE tests run FIRST, so a
    contiguous run of eight absences inside a 4% season is outage-like rather than
    ambiguous.

    Returns:
        ``(arm, evidence)`` where evidence carries the fraction, the absent count,
        the total and the LONGEST CONTIGUOUS RUN of absences in fetch order.
    """
    flags = [bool(value) for value in absent_flags]
    total = len(flags)
    absent = sum(flags)

    longest = run = 0
    for flag in flags:
        run = run + 1 if flag else 0
        longest = max(longest, run)

    fraction = (absent / total) if total else 0.0
    evidence: dict[str, Any] = {
        "total": total,
        "absent": absent,
        "fraction": fraction,
        "longest_contiguous_run": longest,
        "isolated_ceiling": ISOLATED_NULL_FRACTION_CEILING,
        "outage_floor": OUTAGE_NULL_FRACTION_FLOOR,
        "outage_contiguous_run": OUTAGE_CONTIGUOUS_RUN,
    }

    if fraction > OUTAGE_NULL_FRACTION_FLOOR or longest >= OUTAGE_CONTIGUOUS_RUN:
        return NULL_ARM_OUTAGE_LIKE, evidence
    if fraction <= ISOLATED_NULL_FRACTION_CEILING:
        return NULL_ARM_CLEAN, evidence
    return NULL_ARM_AMBIGUOUS, evidence


def _format_null_fraction(fraction: float) -> str:
    """The one rendering of an observed fraction, so the flag can be copied back."""
    return f"{fraction:.{NULL_FRACTION_DECIMALS}f}"


def parse_accept_null_fraction(values: Sequence[str] | None) -> dict[int, float]:
    """Parse ``--accept-null-fraction SEASON=FRACTION`` arguments into a mapping."""
    accepted: dict[int, float] = {}
    for raw in values or ():
        season_text, _, fraction_text = str(raw).partition("=")
        if not fraction_text:
            raise DataIngestionError(
                f"--accept-null-fraction expects SEASON=FRACTION; got {raw!r}. The "
                "fraction is REQUIRED and must be the OBSERVED one, so a season "
                "cannot be pre-authorised blind."
            )
        try:
            accepted[int(season_text)] = float(fraction_text)
        except ValueError as exc:
            raise DataIngestionError(
                f"--accept-null-fraction could not read {raw!r} as SEASON=FRACTION: "
                f"{exc}"
            ) from exc
    return accepted


def recorded_null_overrides(base_path: Any = None) -> list[dict[str, Any]]:
    """Every override this corpus has accepted, oldest first."""
    path = _resolve_data_root(base_path) / NULL_OVERRIDE_LEDGER_PATH
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _record_null_override(
    season: int, evidence: dict[str, Any], base_path: Any = None
) -> dict[str, Any]:
    """Append an accepted override to the durable ledger, and return the entry."""
    entry = {
        "season": int(season),
        "fraction": float(evidence["fraction"]),
        "absent": int(evidence["absent"]),
        "total": int(evidence["total"]),
        "longest_contiguous_run": int(evidence["longest_contiguous_run"]),
        "recorded_at": datetime.now(UTC).isoformat(),
    }
    path = _resolve_data_root(base_path) / NULL_OVERRIDE_LEDGER_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, sort_keys=True) + "\n")
    logger.warning(
        "ACCEPTED a null-observation override",
        season=season,
        fraction=entry["fraction"],
        longest_contiguous_run=entry["longest_contiguous_run"],
        ledger=str(path),
    )
    return entry


def _apply_null_observation_gate(
    season: int,
    absent_flags: Sequence[bool],
    accepted_null_fractions: dict[int, float] | None,
    base_path: Any = None,
) -> tuple[str, dict[str, Any], bool]:
    """Run Ruling L4's gate for one season, BEFORE anything is written.

    Returns ``(arm, evidence, override_accepted)``; raises on OUTAGE_LIKE and on an
    unauthorised AMBIGUOUS season.
    """
    arm, evidence = classify_null_observations(absent_flags)
    observed = _format_null_fraction(evidence["fraction"])

    if arm == NULL_ARM_OUTAGE_LIKE:
        raise DataIngestionError(
            f"season {season} classifies {NULL_ARM_OUTAGE_LIKE}: "
            f"{evidence['absent']} of {evidence['total']} observations are absent "
            f"(fraction {observed}), with a longest CONTIGUOUS run of "
            f"{evidence['longest_contiguous_run']} in fetch order (outage floor "
            f"{OUTAGE_NULL_FRACTION_FLOOR}, contiguous threshold "
            f"{OUTAGE_CONTIGUOUS_RUN}). NOTHING is written for this season. A "
            "provider that has stopped answering fails CONSECUTIVE requests, while "
            "genuine ERA5 gaps scatter across a season -- so this looks like a "
            "provider failure rather than honest absence. Wait for the limit "
            "window to clear and re-invoke; the run is RESUMABLE and will fetch "
            "only the seasons still missing."
        )

    if arm == NULL_ARM_AMBIGUOUS:
        supplied = (accepted_null_fractions or {}).get(int(season))
        authorised = (
            supplied is not None and _format_null_fraction(supplied) == observed
        )
        if not authorised:
            detail = (
                f" The flag supplied {_format_null_fraction(supplied)}, which is not "
                "the observed fraction, so it does not authorise this season."
                if supplied is not None
                else ""
            )
            raise DataIngestionError(
                f"season {season} classifies {NULL_ARM_AMBIGUOUS}: "
                f"{evidence['absent']} of {evidence['total']} observations are "
                f"absent (fraction {observed}), above the isolated ceiling "
                f"{ISOLATED_NULL_FRACTION_CEILING} and below the outage floor "
                f"{OUTAGE_NULL_FRACTION_FLOOR}, with a longest contiguous run of "
                f"{evidence['longest_contiguous_run']}. It REFUSES BY DEFAULT. To "
                "authorise it, re-invoke with "
                f"--accept-null-fraction {season}={observed} -- the flag must carry "
                "the OBSERVED fraction, so a season cannot be pre-authorised "
                f"blind.{detail}"
            )
        _record_null_override(season, evidence, base_path=base_path)
        return arm, evidence, True

    return arm, evidence, False


# ---------------------------------------------------------------------------
# Ruling L3: the run-level corpus lock.
# ---------------------------------------------------------------------------

# The lock's path RELATIVE to a data lake root. Relative rather than absolute so
# importing this module never has to resolve settings, and so a sandboxed run and a
# production run read the same rule rather than two.
CORPUS_LOCK_PATH: Path = Path("bronze") / ".weather_backfill.lock"


class CorpusLockedError(DataIngestionError):
    """Another run holds the corpus lock, or a previous one left it behind."""


def corpus_lock_path(base_path: Any = None) -> Path:
    """The absolute lock path under *base_path*."""
    return _resolve_data_root(base_path) / CORPUS_LOCK_PATH


class CorpusLock:
    """An OS-level exclusive lock over the WHOLE historical weather corpus.

    WHY ``exclusive=True`` ON A BRONZE SNAPSHOT IS NOT ENOUGH, which is the thing
    the next reader will assume. That parameter stops two writers creating the SAME
    snapshot FILENAME and says nothing about two runs racing the corpus. The
    corpus-level hazard is in :func:`data.storage.upsert_silver`
    (``data/storage.py:1110-1123``): it reads the existing parquet, filters it by
    key, concats, and only THEN writes atomically. The WRITE is atomic; the
    READ-MODIFY-WRITE is not. Two promoters that each read the pre-state produce two
    full frames, and the second write silently discards the first's rows -- with no
    error, and with a digest bracket that still reports exactly one CHANGED path,
    exactly as declared. Resume determination has the same shape: two runs can each
    call :func:`seasons_still_missing`, see the same gap, and spend double budget
    fetching one season twice.

    THE PRIMITIVE is ``open(path, "xb")``, so the CREATE is the exclusive step --
    the only place a race can actually be decided, and it holds ACROSS PROCESSES
    where a check-then-write cannot. The file lives under ``bronze/`` rather than in
    a system temp directory for the reason :func:`data.storage._atomic_write_parquet`
    already records: a lock that is not on the same filesystem as the thing it
    guards is guarding a different filesystem.

    A STALE LOCK IS A REFUSAL, NEVER A TAKEOVER. The refusal names the recorded pid,
    the age and the exact ``--force-unlock`` command. No age heuristic is used: a
    paced 81-minute run legitimately holds the lock for 81 minutes, so any threshold
    short enough to be useful is short enough to break a healthy run, and breaking a
    healthy run mid-corpus is the one failure this phase cannot afford.

    THE LOCK IS RETAINED WHEN THE CONTEXT EXITS WITH AN EXCEPTION. That is threat
    T-33.1-31c: a lock that silently disappears lets the very next run proceed over
    a half-written corpus. An exception mid-corpus means the corpus is in an unknown
    state, so the operator clears it deliberately after looking. A clean exit
    removes it.
    """

    def __init__(self, base_path: Any = None, *, force: bool = False) -> None:
        self._base_path = base_path
        self._force = force
        self._path = corpus_lock_path(base_path)
        self._held = False

    @property
    def path(self) -> Path:
        return self._path

    @property
    def held(self) -> bool:
        return self._held

    def _payload(self) -> dict[str, Any]:
        return {
            "pid": os.getpid(),
            "acquired_at_utc": datetime.now(UTC).isoformat(),
            "corpus": {
                "table": BACKFILL_BRONZE_TABLE,
                "first_season": CORPUS_FIRST_SEASON,
                "last_season": CORPUS_LAST_SEASON,
            },
        }

    def _refusal(self) -> CorpusLockedError:
        try:
            recorded = json.loads(self._path.read_text(encoding="utf-8"))
            pid = recorded.get("pid", "unknown")
            acquired = recorded.get("acquired_at_utc", "unknown")
            try:
                age = datetime.now(UTC) - datetime.fromisoformat(acquired)
                age_text = f"{age.total_seconds() / 60:.1f} minutes"
            except (TypeError, ValueError):
                age_text = "unknown"
        except (OSError, ValueError):
            pid, acquired, age_text = "unreadable", "unreadable", "unknown"

        return CorpusLockedError(
            f"the historical weather corpus is LOCKED by pid {pid}, acquired at "
            f"{acquired} ({age_text} ago), at {self._path.as_posix()}. This run "
            "REFUSES rather than taking over: a paced full-corpus run legitimately "
            "holds the lock for about 81 minutes, so no age heuristic can tell a "
            "healthy run from a dead one, and breaking a healthy run mid-corpus is "
            "worse than waiting. If you have CONFIRMED that pid is gone, clear it "
            "with: uv run python -m scripts.backfill_historical_weather "
            "--force-unlock"
        )

    def acquire(self) -> "CorpusLock":
        if self._held:
            raise CorpusLockedError(
                "this CorpusLock instance already holds the lock; acquiring twice "
                "would release it once and leave the corpus unguarded."
            )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._force:
            self.release()
        try:
            with self._path.open("xb") as handle:
                handle.write(json.dumps(self._payload(), indent=2).encode("utf-8"))
        except FileExistsError:
            raise self._refusal() from None
        self._held = True
        return self

    def release(self) -> bool:
        """Remove the lock file. Returns True when a file was actually removed."""
        existed = self._path.is_file()
        self._path.unlink(missing_ok=True)
        self._held = False
        return existed

    def __enter__(self) -> "CorpusLock":
        if not self._held:
            self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc_type is None:
            self.release()
        else:
            self._held = False
            logger.error(
                "CorpusLock RETAINED after an exception -- the corpus is in an "
                "unknown state and the next run will REFUSE until an operator "
                "clears it with --force-unlock",
                lock=str(self._path),
                error=str(exc),
            )
        return False


def acquire_corpus_lock(base_path: Any = None, force: bool = False) -> CorpusLock:
    """Take the corpus lock, or REFUSE naming the holder.

    Returns an already-held :class:`CorpusLock`, which is also a context manager, so
    ``with acquire_corpus_lock(root) as lock:`` releases it on a clean exit and
    retains it on an exception.
    """
    return CorpusLock(base_path, force=force).acquire()


def release_corpus_lock(base_path: Any = None) -> bool:
    """Clear a stale lock. The ``--force-unlock`` implementation."""
    return CorpusLock(base_path).release()


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
        *,
        budget: CallBudget | None = None,
        throttle: RequestThrottle | None = None,
    ) -> dict[str, Any] | str:
        """Open a client and fetch one game's ARCHIVE observation.

        The seam tests replace, mirroring
        :meth:`WeatherDataIngester._fetch_openmeteo_forecast`. The budget and the
        throttle are THREADED THROUGH rather than consumed here: they belong to the
        retried function, one level down.

        Returns:
            A measurement dict, or the :data:`ABSENT_OBSERVATION` sentinel.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            return await fetch_game_weather(
                client,
                latitude,
                longitude,
                game_date,
                game_hour,
                venue_timezone,
                budget=budget,
                throttle=throttle,
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
        budget: CallBudget | None = None,
        throttle: RequestThrottle | None = None,
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
            budget: The run's :class:`CallBudget`. THREADED THROUGH, never
                consumed here -- see Ruling L2 and the comment beside the loop.
            throttle: The run's :class:`RequestThrottle`, likewise threaded.

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
            interval_seconds=throttle.interval_seconds if throttle else 0.0,
        )

        weather_records = []

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
                # RULING L2. This loop DEBITS NOTHING AND WAITS FOR NOTHING. Both
                # happen one level down, inside the retried `fetch_game_weather`,
                # immediately before the request -- because this loop body runs once
                # per logical GAME while that request runs up to three times, and a
                # debit taken here would undercount a retried game by up to 3x
                # against a 5,000/hour binding cap. Do not move them up here.
                #
                # If this fails, WeatherDataError propagates (hard-fail).
                weather_data = asyncio.run(
                    self._fetch_openmeteo_weather(
                        venue["latitude"],
                        venue["longitude"],
                        selected.local_date,
                        selected.hour,
                        selected.timezone,
                        budget=budget,
                        throttle=throttle,
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


async def _probe_archive_day(
    client: Any,
    latitude: float,
    longitude: float,
    day: str,
    venue_timezone: str,
) -> dict[str, Any]:
    """Request ONE whole archive day and report its hour and null counts."""
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": day,
        "end_date": day,
        "hourly": HOURLY_VARIABLES,
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "timezone": venue_timezone,
    }
    try:
        response = await client.get(ARCHIVE_ENDPOINT_URL, params=params)
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise WeatherDataError(
            f"the archive REFUSED the corpus floor probe for {day} at "
            f"({latitude}, {longitude}): {_archive_http_error_detail(exc)}. The "
            "corpus cannot begin before the archive does, so the run stops here "
            "rather than discovering it 24 seasons in."
        ) from exc

    payload = response.json()
    hourly = payload.get("hourly") if isinstance(payload, dict) else None
    temperatures = (hourly or {}).get("temperature_2m") or []
    return {
        "date": day,
        "latitude": latitude,
        "longitude": longitude,
        "timezone": venue_timezone,
        "hours": len(temperatures),
        "nulls": sum(1 for value in temperatures if value is None),
    }


def assert_archive_covers_corpus_floor(
    *,
    client: Any = None,
    backfiller: HistoricalWeatherBackfiller | None = None,
) -> dict[str, Any]:
    """Prove ERA5 reaches the EARLIEST game in the corpus, with ONE request.

    SPEC edge boundary/R3, proven DIRECTLY rather than by citing the vendor's
    documented 1940-01-01 floor. The probe is made at the earliest pinned gameday,
    at that game's OWN venue and in that venue's OWN zone, because a coverage claim
    about a date says nothing until it is asked at a place.

    One request, issued before the first season of a corpus run. A refusal quotes
    the archive's own ``reason`` string, so the operator reads what the provider
    actually said rather than a status code.

    Args:
        client: An httpx-shaped async client. Injected so the refusal path can be
            proven against a stub 400 without exhausting anything.
        backfiller: The ingester, for venue resolution.

    Returns:
        The observed ``date``, coordinates, zone, ``hours`` and ``nulls``.

    Raises:
        WeatherDataError: the archive refused, or the day came back empty.
    """
    backfiller = backfiller or HistoricalWeatherBackfiller()
    schedules = load_schedules([CORPUS_FIRST_SEASON]).sort_values("gameday")
    earliest = schedules.iloc[0]
    day = str(earliest["gameday"])

    venues_df = backfiller._load_venue_data()
    venue = backfiller._get_venue_record_by_stadium_id(
        earliest["stadium_id"], venues_df, game_id=earliest["game_id"]
    )

    async def _run() -> dict[str, Any]:
        if client is not None:
            return await _probe_archive_day(
                client,
                venue["latitude"],
                venue["longitude"],
                day,
                str(venue["timezone"]),
            )
        async with httpx.AsyncClient(timeout=30.0) as owned:
            return await _probe_archive_day(
                owned,
                venue["latitude"],
                venue["longitude"],
                day,
                str(venue["timezone"]),
            )

    observed = asyncio.run(_run())
    observed["stadium_id"] = str(earliest["stadium_id"])
    observed["game_id"] = str(earliest["game_id"])

    if observed["hours"] == 0:
        raise WeatherDataError(
            f"the archive returned NO hourly data for the corpus floor {day} at "
            f"{observed['stadium_id']} ({observed['latitude']}, "
            f"{observed['longitude']}). The corpus starts here, so a run would "
            "produce 24 seasons of absent observations. Nothing is imputed for it."
        )

    logger.info(
        "Archive coverage of the corpus floor CONFIRMED",
        date=day,
        stadium_id=observed["stadium_id"],
        hours=observed["hours"],
        nulls=observed["nulls"],
    )
    return observed


def backfill_season(
    season: int,
    *,
    base_path: Any = None,
    games_df: pd.DataFrame | None = None,
    backfiller: HistoricalWeatherBackfiller | None = None,
    forecast_time: datetime | None = None,
    throttle: RequestThrottle | None = None,
    budget: CallBudget | None = None,
    accepted_null_fractions: dict[int, float] | None = None,
) -> dict[str, Any]:
    """Fetch ONE season and write ONE bronze snapshot. Promotes NOTHING.

    THE UNIT OF RESUMABILITY. A season is roughly 176 to 219 fetches -- under four
    minutes at the declared interval -- so an interrupted corpus run loses at most
    one season's budget rather than a whole day's.

    The snapshot is written with ``exclusive=True``, which is not optional here: it
    makes the CREATE the exclusive step, which is the only place a same-second
    collision can actually be decided, and it holds ACROSS PROCESSES where a
    check-then-write cannot. That is SPEC R2's concurrency BACKSTOP, obtained from
    an existing parameter rather than from new machinery -- and it is a FILENAME
    backstop only. The corpus-level guarantee is :class:`CorpusLock`.

    THE FRAME IS VALIDATED AT BRONZE-WRITE TIME as well as at promotion.
    :func:`data.quality_gates.validate_bronze_to_silver` raises if ANY row fails,
    killing the whole batch, so validating once over 6,499 rows at the end would let
    one bad 2003 row destroy an 81-minute run at its last step. Validating per
    season makes the failure local, named and cheap.

    Args:
        season: The season to fetch. Refused by name if outside the corpus.
        base_path: Data lake root. Threaded so a test writes somewhere else.
        games_df: The games to fetch. The INJECTION SEAM: when None, the season is
            read from the silver ``games`` table exactly as production does.
        backfiller: The ingester. Constructed when None.
        forecast_time: When this run was taken. Defaults to now, in UTC.
        throttle: The run's request throttle.
        budget: The run's weighted call budget. Shared ACROSS seasons by
            :func:`backfill_corpus`, because the caps are per hour and per day
            rather than per season.
        accepted_null_fractions: Operator-supplied ``{season: observed_fraction}``
            authorisations for AMBIGUOUS seasons (Ruling L4).

    Returns:
        A per-season report: the season, the game and fetch counts, the snapshot
        path, the null-observation arm and its evidence, and whether an override
        was accepted.

    Raises:
        DataIngestionError: the season is outside the corpus, its games frame is
            empty, or the null-observation gate refuses it.
    """
    assert_season_in_corpus(season)

    backfiller = backfiller or HistoricalWeatherBackfiller()
    if throttle is None:
        throttle = RequestThrottle(ARCHIVE_REQUEST_INTERVAL_SECONDS)
    if budget is None:
        budget = CallBudget()
    if forecast_time is None:
        forecast_time = datetime.now(UTC)

    if games_df is None:
        games_df = backfiller._load_games_data(season, None)

    if games_df.empty:
        raise DataIngestionError(
            f"season {season} has NO games to fetch. NOTHING is written for it: an "
            "empty bronze snapshot would make the season look attempted while "
            "covering no game, and the coverage-based resume rule would keep "
            "reporting it missing with no log line explaining why (SPEC edge "
            "empty/R2). Check data/silver/games.parquet for that season."
        )

    venues_df = backfiller._load_venue_data()
    weather_df = backfiller.fetch_weather_for_games(
        games_df, venues_df, forecast_time, budget=budget, throttle=throttle
    )

    if weather_df.empty:
        raise DataIngestionError(
            f"season {season} produced NO weather records from {len(games_df)} "
            "games. Nothing is written for it."
        )

    # The absent flags, in FETCH ORDER, over the games that were ACTUALLY ASKED. A
    # dome was never asked, so including it would dilute the fraction and make the
    # gate report a number it did not measure.
    fetched = weather_df[weather_df["is_outdoor"].astype(bool)]
    absent_flags = [not bool(value) for value in fetched["weather_coverage"]]

    arm, evidence, override_accepted = _apply_null_observation_gate(
        season, absent_flags, accepted_null_fractions, base_path=base_path
    )

    # Per-season validation, BEFORE the write. The frame written to bronze is the
    # RAW one -- bronze is the snapshot of record -- but it is proven promotable
    # first, so a bad row is caught where it is cheap to fix.
    promote_weather_bronze_to_silver(weather_df)

    path = save_bronze_snapshot(
        weather_df,
        BACKFILL_BRONZE_TABLE,
        season=season,
        week=0,
        base_path=base_path,
        exclusive=True,
    )

    logger.info(
        "Season written to bronze",
        season=season,
        games=len(weather_df),
        fetched=len(fetched),
        null_arm=arm,
        null_fraction=evidence["fraction"],
        weighted_calls_so_far=round(budget.weighted_total, 1),
        hourly_cap=budget.hourly_budget,
        daily_cap=budget.daily_budget,
        path=str(path),
    )

    return {
        "season": season,
        "weighted_calls_after": round(budget.weighted_total, 4),
        "games": len(weather_df),
        "fetched": len(fetched),
        "written_without_a_call": len(weather_df) - len(fetched),
        "path": str(path),
        "null_arm": arm,
        "null_evidence": evidence,
        "override_accepted": override_accepted,
    }


def _file_sha256(path: Path) -> str:
    """The sha256 of *path*'s bytes, read in chunks.

    THE SAME VALUE ``tests.data_boundary.digest_file`` returns for a readable file,
    computed here rather than imported because production code does not import from
    the test tree. It deliberately does NOT carry that function's locked-file stat
    fallback: a bronze snapshot this process has just finished reading is readable
    by construction, and a provenance record that could silently degrade to size
    and mtime would not be a provenance record.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def promote_corpus_to_silver(base_path: Any = None) -> dict[str, Any]:
    """Validate every `weather_backfill` bronze snapshot and upsert silver `weather`.

    LATEST SNAPSHOT WINS, per ``game_id``. Snapshot filenames carry a UTC timestamp,
    so reading them in sorted order and keeping the LAST row per game makes a re-run
    supersede its predecessor rather than duplicate it -- which, with
    :func:`data.storage.upsert_silver`'s latest-wins merge, is what makes a double
    run leave the silver row count unchanged (SPEC edge idempotency/R4).

    This function FETCHES NOTHING. It is the ``--promote-silver`` half of the run,
    and Plan 33.1-06 brackets it with a digest of the whole data tree.

    IT RECORDS WHICH BRONZE IT READ, AND ASSERTS IT READ NO LEGACY TABLE (Ruling X).
    "Silver holds 6,499 rows" is satisfied by the right bronze and by the wrong
    bronze. The seven legacy ``weather_raw_bronze_{2018..2024}_season.parquet`` files
    hold 1,942 rows at SEVENTEEN columns against this corpus's TWENTY-FIVE, so a
    promotion that accidentally globbed them would mix two schemas and two
    provenances -- and a ROW-COUNT MATCH CANNOT DETECT THAT. R2's acceptance is about
    provenance, not arithmetic. So the returned record carries, and the caller
    asserts:

    * ``bronze_tables_read`` -- the table name(s) globbed. ``weather_backfill``, and
      nothing else.
    * ``bronze_files_read`` -- every filename with its sha256 and its column count,
      so the promotion's INPUT is pinned as precisely as its output.
    * ``legacy_files_read`` -- asserted EMPTY against the ten legacy filenames, a
      POSITIVE statement about what was not read, in the same idiom as the DuckDB
      exclusion in the changed-file declaration.
    * ``column_counts_read`` -- two INDEPENDENT instruments on one property, because
      a 17-column legacy file could not contribute even if the filename check were
      wrong.
    """
    bronze = _resolve_data_root(base_path) / "bronze"
    paths = sorted(bronze.glob(f"{BACKFILL_BRONZE_GLOB}.parquet"))
    if not paths:
        raise DataIngestionError(
            f"there are no {BACKFILL_BRONZE_TABLE} bronze snapshots under "
            f"{bronze.as_posix()} to promote. Bronze is the snapshot of record, so "
            "silver is never written from anything else. Run the fetch first."
        )

    # Ruling X. Recorded from the paths that were ACTUALLY globbed, before any of
    # them is read, so the record describes the promotion's real input rather than
    # the input it was supposed to have.
    read_record: dict[str, dict[str, Any]] = {}
    frames = []
    for path in paths:
        frame = pd.read_parquet(path, engine="pyarrow")
        frames.append(frame)
        read_record[path.name] = {
            "digest": _file_sha256(path),
            "rows": len(frame),
            "columns": len(frame.columns),
        }

    legacy_read = sorted(
        name for name in read_record if name in LEGACY_WEATHER_BRONZE_FILENAMES
    )
    if legacy_read:
        raise DataIngestionError(
            "the promotion globbed LEGACY weather bronze files and REFUSES: "
            f"{legacy_read}. SPEC prohibition 6 keeps the 1,942 pre-existing rows as "
            "the only evidence the routing fix can be regression-tested against, and "
            f"the {BACKFILL_BRONZE_TABLE} glob is supposed to be disjoint from them. "
            "A silver frame mixing a 17-column and a 25-column provenance would pass "
            "a row-count check while answering a different question."
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["game_id"], keep="last")

    validated = promote_weather_bronze_to_silver(combined)
    validated["created_at"] = datetime.now(UTC)

    silver_path = upsert_silver(validated, "weather", base_path=base_path)

    log_data_operation(
        operation="promote",
        table="weather",
        rows=len(validated),
        snapshots=len(paths),
    )

    return {
        "snapshots": len(paths),
        "rows": len(validated),
        "silver_path": str(silver_path),
        "bronze_tables_read": (BACKFILL_BRONZE_TABLE,),
        "bronze_glob": BACKFILL_BRONZE_GLOB,
        "bronze_files_read": read_record,
        "legacy_files_read": tuple(legacy_read),
        "legacy_inventory_checked_against": LEGACY_WEATHER_BRONZE_FILENAMES,
        "column_counts_read": tuple(
            sorted({meta["columns"] for meta in read_record.values()})
        ),
    }


def backfill_corpus(
    seasons: Sequence[int] | None = None,
    *,
    base_path: Any = None,
    fetch: bool = True,
    promote: bool = False,
    dry_run: bool = False,
    crosscheck: bool = False,
    force_unlock: bool = False,
    games_provider: Callable[[int], pd.DataFrame] | None = None,
    backfiller: HistoricalWeatherBackfiller | None = None,
    throttle: RequestThrottle | None = None,
    budget: CallBudget | None = None,
    accepted_null_fractions: dict[int, float] | None = None,
    verify_archive_floor: bool = True,
    lock: "CorpusLock | None" = None,
) -> dict[str, Any]:
    """Run the corpus: take the lock, decide what is missing, fetch, then promote.

    THE LOCK SPANS THE WHOLE RUN (Ruling L3). It is taken BEFORE resume
    determination and held through every season and through the promotion, because
    a lock released between the pre-state digest and the promotion protects nothing.
    Plan 33.1-06's digest bracket runs inside it.

    A DRY RUN TAKES NO LOCK. It writes nothing and issues no request, so it is a
    pure read; the lock exists to serialise WRITERS, and making a read take it
    would put a transient file into the production bronze directory for no gain.

    Args:
        seasons: Explicit seasons, or None to use :func:`seasons_still_missing`.
        base_path: Data lake root.
        fetch: Fetch the targeted seasons. False for a promotion-only run.
        promote: Promote the bronze corpus into silver after fetching.
        dry_run: Report what WOULD be fetched, at zero network calls.
        crosscheck: After fetching and promoting, REPORT the old-versus-new diff.
            Never fails on a disagreement.
        force_unlock: Clear a stale lock first. The operator must have confirmed
            the recorded pid is gone.
        games_provider: Injection seam -- ``season -> games frame``.
        backfiller: The ingester. Constructed when None.
        throttle: The run's request throttle.
        budget: The run's weighted call budget, shared across every season.
        accepted_null_fractions: ``{season: observed_fraction}`` authorisations.
        verify_archive_floor: Probe ERA5 coverage of the corpus floor before the
            first season. One request.
        lock: An ALREADY-HELD :class:`CorpusLock`. When given, this function runs
            INSIDE the caller's acquisition instead of taking a second one -- which
            it could not do anyway, since the primitive is ``open(path, "xb")`` and
            a second acquire would refuse. This is what lets
            :func:`bracketed_corpus_operation` take the lock BEFORE the pre-state
            digest and hold it through post-state verification: a lock released
            between the digest and the write protects nothing, because a second
            promoter that read the same pre-state produces a frame silently
            discarding this one's rows while the bracket still reports exactly one
            CHANGED path, as declared (threat T-33.1-40c).

    Returns:
        A run report carrying the seasons considered, the per-season reports, the
        weighted call total and the promotion result.
    """
    if dry_run:
        return _dry_run_report(seasons, base_path=base_path)

    backfiller = backfiller or HistoricalWeatherBackfiller()
    if throttle is None:
        throttle = RequestThrottle(ARCHIVE_REQUEST_INTERVAL_SECONDS)
    if budget is None:
        budget = CallBudget()

    report: dict[str, Any] = {
        "seasons_requested": tuple(seasons) if seasons is not None else None,
        "seasons": [],
        "promotion": None,
    }

    # An already-held lock is entered as a no-op context, so the caller's single
    # acquisition -- taken before the pre-state digest -- spans this whole run.
    # Without a held lock this function takes its own, exactly as it always has.
    lock_context: Any = (
        contextlib.nullcontext(lock)
        if lock is not None and lock.held
        else acquire_corpus_lock(base_path=base_path, force=force_unlock)
    )
    report["lock_supplied_by_caller"] = lock is not None and lock.held

    with lock_context:
        missing = seasons_still_missing(base_path=base_path)
        report["seasons_still_missing"] = missing
        targets = tuple(seasons) if seasons is not None else missing
        report["seasons_targeted"] = targets

        if fetch and targets:
            if verify_archive_floor:
                report["archive_floor"] = assert_archive_covers_corpus_floor(
                    backfiller=backfiller
                )
            # The refusal has to name the operator's next action, and their next
            # action is to wait and re-invoke rather than to start over.
            budget.resume_hint = (
                f"Seasons still missing at the start of this run: {list(missing)}."
            )
            for season in targets:
                report["seasons"].append(
                    backfill_season(
                        season,
                        base_path=base_path,
                        games_df=(
                            games_provider(season)
                            if games_provider is not None
                            else None
                        ),
                        backfiller=backfiller,
                        throttle=throttle,
                        budget=budget,
                        accepted_null_fractions=accepted_null_fractions,
                    )
                )

        if promote:
            report["promotion"] = promote_corpus_to_silver(base_path=base_path)

        if crosscheck:
            # Inside the lock, like everything else in the run: the comparison is a
            # statement about a corpus, and a corpus another process is writing is
            # not a corpus this one can describe.
            report["crosscheck"] = crosscheck_corpus_against_legacy_bronze(
                base_path=base_path
            )

    report["weighted_calls"] = round(budget.weighted_total, 4)
    report["attempts"] = budget.debits
    return report


# The two halves of Plan 33.1-06's bracket, and the digest documents each writes.
# Named here rather than passed on the command line so the operator cannot point a
# verify at the wrong half's pre-state and get a clean answer about the wrong thing.
BRACKET_DOCUMENTS: dict[str, dict[str, str]] = {
    "run": {
        "data": "outputs/phase331_corpus_before.json",
        "artifacts": "outputs/phase331_artifacts_before.json",
    },
    "promote": {
        "data": "outputs/phase331_promote_before.json",
        "artifacts": "outputs/phase331_promote_artifacts_before.json",
    },
}


def bracketed_corpus_operation(
    operation: str,
    *,
    base_path: Any = None,
    accepted_null_fractions: dict[int, float] | None = None,
    force_unlock: bool = False,
    verify_archive_floor: bool = True,
) -> dict[str, Any]:
    """Take the lock, digest the production trees, run one half, then re-digest.

    THE LOCK IS STEP 0 AND THAT IS THE WHOLE POINT (Ruling L3, threat T-33.1-40c).
    ``backfill_corpus`` has always taken its own lock, but it takes it AFTER the
    caller's pre-state digest -- and a lock released between the digest and the
    write protects nothing. ``data.storage.upsert_silver`` is read-filter-concat-
    write: the WRITE is atomic, the SEQUENCE is not. A second promoter that read the
    same pre-state produces a frame that silently discards this one's rows, with no
    error, while the digest bracket still reports exactly ONE CHANGED path -- exactly
    as declared. The bracket cannot see that race; only the lock can prevent it. So
    the lock is acquired here, before the first digest, and held through resume
    determination, every season's fetch, the promotion and the post-state
    verification.

    THE DIGEST INSTRUMENT IS THE SHIPPED ONE. ``tests.data_boundary`` is imported
    lazily, and deliberately rather than re-implemented: ``data/`` is gitignored, so
    a git check is structurally incapable of failing here (COLD-05), and minting a
    second content-digest instrument for this one run would mean the bracket and the
    suite disagree about what "unchanged" means. This is an OPERATOR entry point on a
    CLI path; no production consumer imports it.

    BOTH SIDES USE ``content_digest_tree`` rather than ``digest_tree``. The degrading
    stat fallback exists so the CLI can keep reporting on a locked store, but a
    content hash on one side and a stat signature on the other is an UNDECIDED
    comparison (NF-02), and this bracket's whole output is a verdict. An unreadable
    file raises ``LockedStoreDigestError`` here instead of quietly becoming a MIXED
    key -- which is the intended behaviour, not a hazard.

    IT REPORTS; IT DOES NOT JUDGE. The measured diff is returned and printed. Whether
    it matches ``tests.phase33_state.WEATHER_PROMOTION_EXPECTED_CHANGED_FILES`` is
    decided against the DECLARATION, which lives in the test tree because that is
    where it was committed before the run. A file outside the declared set is a
    FINDING to report, never a reason to widen the set.

    Args:
        operation: ``"run"`` -- fetch every missing season, promote nothing; or
            ``"promote"`` -- promote the bronze corpus into silver, fetch nothing.
        base_path: Data lake root for the OPERATION. The digested trees are always
            the production roots.
        accepted_null_fractions: ``{season: observed_fraction}`` authorisations.
        force_unlock: Clear a stale lock first. Only after confirming the pid is gone.
        verify_archive_floor: Probe ERA5 coverage of the corpus floor. One request.

    Returns:
        The operation's own report plus ``bracket``: the documents written, the
        file counts digested, and the measured ``added``/``removed``/``changed``/
        ``mixed`` diff for each of the two production roots.
    """
    if operation not in BRACKET_DOCUMENTS:
        raise ValueError(
            f"unknown bracket half {operation!r}; expected one of "
            f"{sorted(BRACKET_DOCUMENTS)}"
        )

    from tests import data_boundary

    documents = BRACKET_DOCUMENTS[operation]
    roots = {
        "data": data_boundary.PRODUCTION_DATA_ROOT,
        "artifacts": data_boundary.PRODUCTION_ARTIFACTS_ROOT,
    }

    bracket: dict[str, Any] = {"operation": operation, "documents": documents}

    # STEP 0 -- the lock, BEFORE the pre-state digest.
    lock = acquire_corpus_lock(base_path=base_path, force=force_unlock)
    bracket["lock_path"] = str(lock.path)
    bracket["lock_payload"] = json.loads(lock.path.read_text(encoding="utf-8"))
    logger.info(
        "Corpus lock acquired BEFORE the pre-state digest",
        lock=str(lock.path),
        pid=bracket["lock_payload"].get("pid"),
        acquired_at_utc=bracket["lock_payload"].get("acquired_at_utc"),
        operation=operation,
    )

    with lock:
        # STEPS 1 and 2 -- the pre-state, written where a later `verify` can read it.
        before: dict[str, dict[str, str]] = {}
        for name, root in roots.items():
            before[name] = data_boundary.content_digest_tree(root)
            document = Path(documents[name])
            document.parent.mkdir(parents=True, exist_ok=True)
            document.write_text(json.dumps(before[name], indent=2), encoding="utf-8")
            bracket[f"{name}_files_digested"] = len(before[name])

        # STEP 3 -- the one thing this half does, inside the SAME acquisition.
        report = backfill_corpus(
            base_path=base_path,
            fetch=(operation == "run"),
            promote=(operation == "promote"),
            accepted_null_fractions=accepted_null_fractions,
            verify_archive_floor=verify_archive_floor and operation == "run",
            lock=lock,
        )

        # STEPS 4 and 5 -- the post-state, still inside the lock.
        for name, root in roots.items():
            after = data_boundary.content_digest_tree(root)
            bracket[f"{name}_diff"] = data_boundary.diff_digests(before[name], after)
            if name == "data":
                bracket["silver_weather_digest_before"] = before[name].get(
                    "silver/weather.parquet"
                )
                bracket["silver_weather_digest_after"] = after.get(
                    "silver/weather.parquet"
                )

    report["bracket"] = bracket

    # The run record, written where the state manifest can be transcribed from it
    # rather than from a scrollback. An 81-minute run whose only account of itself
    # is a terminal buffer is a run nobody can check afterwards.
    record = Path(f"outputs/phase331_bracket_{operation}_report.json")
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    bracket["report_document"] = str(record)
    return report


def _print_bracket_report(bracket: dict[str, Any]) -> None:
    """Print what the bracket measured, naming every path on both trees."""
    print(f"Bracket half:  {bracket['operation']}")
    print(
        f"  lock         {bracket['lock_path']} "
        f"(pid {bracket['lock_payload'].get('pid')}, "
        f"acquired {bracket['lock_payload'].get('acquired_at_utc')})"
    )
    for name in ("data", "artifacts"):
        diff = bracket[f"{name}_diff"]
        print(
            f"  {name}: {bracket[f'{name}_files_digested']} file(s) digested -> "
            f"{len(diff['added'])} added, {len(diff['changed'])} changed, "
            f"{len(diff['removed'])} removed, {len(diff.get('mixed', []))} undecided"
        )
        for label in ("added", "changed", "removed", "mixed"):
            for key in diff.get(label, []):
                print(f"    {label.upper():8s} {key}")
    print(
        "  A path outside the declaration committed before this run is a FINDING "
        "to report, never a reason to widen the declaration."
    )


def _dry_run_report(
    seasons: Sequence[int] | None, *, base_path: Any = None
) -> dict[str, Any]:
    """What a run WOULD do, at ZERO network calls.

    The split between games that need a fetch and games written with no call comes
    from the pinned feed's own ``roof`` value -- the same input the real run branches
    on -- so the projection is a statement about THIS corpus rather than a round
    number.

    THE PROJECTION ASSUMES ZERO RETRIES, and says so in a key. A retried game issues
    up to three requests and the provider counts three, so the OBSERVED weighted
    total a real run records will exceed this; the gap IS the retry rate.
    """
    missing = seasons_still_missing(base_path=base_path)
    targets = tuple(seasons) if seasons is not None else missing

    rows = []
    total_fetch = 0
    total_no_call = 0
    for season in targets:
        facts = load_pinned_game_facts((season,))
        roofs = facts["roof"].astype(str).str.lower().str.strip()
        fetch_count = int(roofs.isin(("outdoors", "open")).sum())
        no_call = int(len(facts) - fetch_count)
        total_fetch += fetch_count
        total_no_call += no_call
        rows.append(
            {
                "season": season,
                "games": len(facts),
                "would_fetch": fetch_count,
                "written_without_a_call": no_call,
            }
        )

    return {
        "dry_run": True,
        "requests_issued": 0,
        "seasons_still_missing": missing,
        "seasons_targeted": targets,
        "per_season": rows,
        "games_to_fetch": total_fetch,
        "games_without_a_call": total_no_call,
        "projected_weighted_calls": total_fetch * ARCHIVE_CALL_WEIGHT_PER_REQUEST,
        "retries_assumed": 0,
        "interval_seconds": ARCHIVE_REQUEST_INTERVAL_SECONDS,
        "projected_seconds": total_fetch * ARCHIVE_REQUEST_INTERVAL_SECONDS,
        "hourly_budget": ARCHIVE_HOURLY_CALL_BUDGET,
        "daily_budget": ARCHIVE_DAILY_CALL_BUDGET,
    }


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


# ---------------------------------------------------------------------------
# SPEC R2: the old-versus-new cross-check.
#
# THE EXPECTATION IT IS MEASURED AGAINST LIVES IN
# `scripts/weather_crosscheck_constants.py`, committed BEFORE this comparator ever
# ran, and witnessed from outside in a later commit. Nothing in this function
# predicts anything; it only measures, and the terms of the measurement -- the
# tolerance, the comparable columns, the join key -- are READ from that module
# rather than restated here, so the two cannot drift apart.
# ---------------------------------------------------------------------------


def _blank_counts() -> dict[str, int]:
    return {
        "compared": 0,
        "agreed": 0,
        "disagreed": 0,
        "became_null": 0,
        "became_present": 0,
        "both_null": 0,
    }


def _values_agree(new_value: Any, old_value: Any) -> bool:
    """Equal within the ONE declared tolerance, for numbers; exact otherwise.

    The epsilon guards float representation, not the contract: `70.0 + 0.1` is
    `70.09999999999999` in binary floating point, and a comparison that called that
    a disagreement would be reporting the machine rather than the weather.
    """
    if isinstance(new_value, bool) or isinstance(old_value, bool):
        return bool(new_value) == bool(old_value)
    if isinstance(new_value, (int, float)) and isinstance(old_value, (int, float)):
        return abs(float(new_value) - float(old_value)) <= (
            CROSSCHECK_TOLERANCE_F + 1e-9
        )
    return new_value == old_value


def compare_weather_frames(
    new_frame: pd.DataFrame,
    legacy_frame: pd.DataFrame,
) -> dict[str, Any]:
    """Compare the new corpus against the legacy bronze, per season and per home team.

    A DISAGREEMENT IS A FINDING, NEVER A FAILURE. This function does not raise on one,
    and that is the decision rather than an omission: Phase 33.1 PREDICTS a large
    disagreement -- the venue-local-hour fix alone moves nearly every comparable row --
    so a comparator that refused on disagreement would refuse the corpus it exists to
    validate.

    THE COMPARISON IS OVER INTERSECTING COLUMNS ONLY. The seven legacy season files
    carry seventeen columns; the current schema carries twenty-five. The eight columns
    the legacy side does not have are EXCLUDED rather than reported as universal
    disagreement -- a column that did not exist cannot have changed.

    A NUMBER BECOMING A NULL IS ITS OWN CATEGORY, counted separately from a value
    disagreement, because it is a different fact: the pre-registration predicts it is
    the ONLY shape the per-game roof rule produces, and folding it into "disagreed"
    would hide exactly the thing the check is for.

    Args:
        new_frame: The corpus this phase produced.
        legacy_frame: The pre-existing bronze rows.

    Returns:
        A report carrying the terms of the comparison, the join accounting, the
        per-column counts, and the per-season and per-home-team breakdowns.
    """
    compared_columns = tuple(
        column
        for column in COMPARED_COLUMNS
        if column in new_frame.columns and column in legacy_frame.columns
    )

    new_by_id = {str(row[JOIN_KEY]): row for _, row in new_frame.iterrows()}
    legacy_by_id = {str(row[JOIN_KEY]): row for _, row in legacy_frame.iterrows()}

    joined_ids = sorted(set(new_by_id) & set(legacy_by_id))
    new_only = sorted(set(new_by_id) - set(legacy_by_id))
    legacy_only = sorted(set(legacy_by_id) - set(new_by_id))

    per_column = {column: _blank_counts() for column in compared_columns}
    per_season: dict[int, dict[str, int]] = {}
    per_home_team: dict[str, dict[str, int]] = {}

    def _bucket(store: dict, key: Any) -> dict[str, int]:
        if key not in store:
            store[key] = {"rows": 0, **_blank_counts()}
        return store[key]

    for game_id in joined_ids:
        new_row = new_by_id[game_id]
        old_row = legacy_by_id[game_id]

        # The season and the home team come from the id itself. A parsed id cannot
        # disagree with a `season` column that a frame may or may not carry.
        try:
            parsed = parse_game_id(game_id)
            season_key: Any = parsed["season"]
            team_key: Any = parsed["home_team"]
        except ValueError:
            season_key = "UNPARSEABLE"
            team_key = "UNPARSEABLE"

        season_bucket = _bucket(per_season, season_key)
        team_bucket = _bucket(per_home_team, team_key)
        season_bucket["rows"] += 1
        team_bucket["rows"] += 1

        for column in compared_columns:
            new_value = new_row[column]
            old_value = old_row[column]
            new_null = pd.isna(new_value)
            old_null = pd.isna(old_value)

            if new_null and old_null:
                outcome = "both_null"
            elif new_null:
                outcome = "became_null"
            elif old_null:
                outcome = "became_present"
            elif _values_agree(new_value, old_value):
                outcome = "agreed"
            else:
                outcome = "disagreed"

            for bucket in (per_column[column], season_bucket, team_bucket):
                bucket["compared"] += 1
                bucket[outcome] += 1

    return {
        "tolerance": CROSSCHECK_TOLERANCE_F,
        "join_key": JOIN_KEY,
        "compared_columns": compared_columns,
        "excluded_columns": ABSENT_FROM_LEGACY_COLUMNS,
        "non_comparable_intersecting_columns": NON_COMPARABLE_INTERSECTING_COLUMNS,
        "rows_compared": len(joined_ids),
        "rows_new_only": len(new_only),
        "rows_legacy_only": len(legacy_only),
        "new_only_game_ids": tuple(new_only),
        "legacy_only_game_ids": tuple(legacy_only),
        "per_column": per_column,
        "per_season": per_season,
        "per_home_team": per_home_team,
        "total_disagreements": sum(
            counts["disagreed"] for counts in per_column.values()
        ),
        "number_to_null": sum(counts["became_null"] for counts in per_column.values()),
        "null_to_number": sum(
            counts["became_present"] for counts in per_column.values()
        ),
        "preregistration_path": PREREGISTRATION_PATH,
    }


def crosscheck_corpus_against_legacy_bronze(base_path: Any = None) -> dict[str, Any]:
    """Load both corpora from bronze and compare them over COMPARABLE_SEASONS.

    The LEGACY side is read from the ten filenames named in
    :data:`LEGACY_WEATHER_BRONZE_FILENAMES`, filtered to the seven whole-season files
    that cover 2018-2024 -- an explicit list rather than a glob, so widening the
    pattern cannot quietly change what "the legacy corpus" means.
    """
    bronze = _resolve_data_root(base_path) / "bronze"

    new_paths = sorted(bronze.glob(f"{BACKFILL_BRONZE_GLOB}.parquet"))
    if not new_paths:
        raise DataIngestionError(
            f"there are no {BACKFILL_BRONZE_TABLE} bronze snapshots under "
            f"{bronze.as_posix()} to cross-check. Run the fetch first."
        )

    legacy_names = [
        name
        for name in LEGACY_WEATHER_BRONZE_FILENAMES
        if any(f"_{season}_season" in name for season in COMPARABLE_SEASONS)
    ]
    legacy_paths = [bronze / name for name in legacy_names if (bronze / name).is_file()]
    if not legacy_paths:
        raise DataIngestionError(
            f"none of the legacy whole-season weather bronze files {legacy_names} is "
            f"present under {bronze.as_posix()}. They are the ONLY evidence the "
            "routing fix can be regression-tested against."
        )

    new_frame = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in new_paths],
        ignore_index=True,
    ).drop_duplicates(subset=[JOIN_KEY], keep="last")
    legacy_frame = pd.concat(
        [pd.read_parquet(path, engine="pyarrow") for path in legacy_paths],
        ignore_index=True,
    ).drop_duplicates(subset=[JOIN_KEY], keep="last")

    report = compare_weather_frames(new_frame, legacy_frame)
    report["comparable_seasons"] = COMPARABLE_SEASONS
    report["legacy_files"] = tuple(path.name for path in legacy_paths)
    report["new_files"] = tuple(path.name for path in new_paths)
    return report


def _print_crosscheck_report(report: dict[str, Any]) -> None:
    """Print the diff per season and per home team. It reports; it never verdicts."""
    print(
        f"Cross-check against the legacy bronze, tolerance "
        f"{report['tolerance']} F over {len(report['compared_columns'])} columns"
    )
    print(
        f"  rows compared {report['rows_compared']}, "
        f"new-only {report['rows_new_only']}, legacy-only {report['rows_legacy_only']}"
    )
    print(f"  total disagreements {report['total_disagreements']}")
    print(f"  number -> NULL      {report['number_to_null']}")

    print("  per season:")
    for season in sorted(report["per_season"]):
        counts = report["per_season"][season]
        print(
            f"    {season}: {counts['rows']} rows, {counts['agreed']} agreed, "
            f"{counts['disagreed']} disagreed, {counts['became_null']} became null"
        )

    print("  per home team:")
    for team in sorted(report["per_home_team"]):
        counts = report["per_home_team"][team]
        print(
            f"    {team}: {counts['rows']} rows, {counts['agreed']} agreed, "
            f"{counts['disagreed']} disagreed, {counts['became_null']} became null"
        )

    print(
        "  A DISAGREEMENT IS A FINDING, NOT A FAILURE. The expected shape was "
        f"registered in advance at {report['preregistration_path']}."
    )


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
    parser.add_argument(
        "--season",
        type=int,
        help=(
            "ONE season to backfill. The whole season is the unit -- `--week` is "
            "gone, because the corpus is whole seasons and the resume rule is "
            "keyed on a season's game-id coverage."
        ),
    )
    parser.add_argument(
        "--all-seasons",
        action="store_true",
        help=(
            "Backfill every season seasons_still_missing() reports, ascending. "
            "Safe to re-invoke after an interruption: a covered season is skipped."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Report the seasons still missing, what each would fetch and the "
            "PROJECTED weighted call total. Issues ZERO requests and takes no lock."
        ),
    )
    parser.add_argument(
        "--promote-silver",
        action="store_true",
        help=(
            "Validate the weather_backfill bronze corpus and upsert silver "
            "`weather`. Fetches nothing."
        ),
    )
    parser.add_argument(
        "--crosscheck",
        action="store_true",
        help=(
            "Compare the weather_backfill corpus against the legacy 2018-2024 bronze "
            "and REPORT the diff per season and per home team. Fetches nothing, and "
            "never fails on a disagreement -- the expected shape was registered in "
            "advance at scripts/weather_crosscheck_constants.py."
        ),
    )
    parser.add_argument(
        "--force-unlock",
        action="store_true",
        help=(
            "Clear a stale corpus lock before running. Use ONLY after confirming "
            "the pid the refusal named is gone: a healthy paced run legitimately "
            "holds the lock for about 81 minutes."
        ),
    )
    parser.add_argument(
        "--accept-null-fraction",
        action="append",
        metavar="SEASON=FRACTION",
        default=[],
        help=(
            "Authorise an AMBIGUOUS season's absent-observation fraction. The "
            "fraction must be the OBSERVED one the refusal printed, so a season "
            "cannot be pre-authorised blind. Repeatable."
        ),
    )
    parser.add_argument(
        "--bracket",
        choices=sorted(BRACKET_DOCUMENTS),
        help=(
            "Run ONE half of Plan 33.1-06's declared bracket: take the corpus lock "
            "FIRST, digest data/ and artifacts/, do the half, then re-digest -- all "
            "inside a single lock acquisition. `run` fetches every missing season "
            "and promotes nothing; `promote` promotes the bronze corpus into silver "
            "and fetches nothing. A lock released between the digest and the write "
            "protects nothing, which is why the ordinary --all-seasons path (which "
            "takes its lock AFTER the caller's digest) is not the bracket."
        ),
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

    if args.bracket:
        try:
            report = bracketed_corpus_operation(
                args.bracket,
                accepted_null_fractions=parse_accept_null_fraction(
                    args.accept_null_fraction
                ),
                force_unlock=bool(args.force_unlock),
            )
        except (DataIngestionError, WeatherDataError, OSError, ValueError) as exc:
            logger.error("Bracketed corpus operation failed", error=str(exc))
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        _print_run_report(report)
        _print_bracket_report(report["bracket"])
        return

    if args.force_unlock and not (
        args.all_seasons or args.season or args.promote_silver
    ):
        cleared = release_corpus_lock()
        print("Cleared the corpus lock" if cleared else "No corpus lock to clear")
        return

    if args.crosscheck and not (args.all_seasons or args.season or args.promote_silver):
        try:
            _print_crosscheck_report(crosscheck_corpus_against_legacy_bronze())
        except (DataIngestionError, OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        return

    if not (args.all_seasons or args.season or args.promote_silver):
        print(
            "nothing to do: give --all-seasons, --season, --promote-silver, "
            "--crosscheck, --stamp-weather-source or --force-unlock"
        )
        sys.exit(1)

    try:
        accepted = parse_accept_null_fraction(args.accept_null_fraction)
        report = backfill_corpus(
            seasons=(args.season,) if args.season else None,
            fetch=bool(args.all_seasons or args.season),
            promote=bool(args.promote_silver),
            dry_run=bool(args.dry_run),
            crosscheck=bool(args.crosscheck),
            force_unlock=bool(args.force_unlock),
            accepted_null_fractions=accepted,
        )
    except (DataIngestionError, WeatherDataError, OSError, ValueError) as exc:
        logger.error("Historical weather backfill failed", error=str(exc))
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    _print_run_report(report)


def _print_run_report(report: dict[str, Any]) -> None:
    """Print what a run did, in the terms the operator's next action needs."""
    print(f"Seasons still missing: {list(report.get('seasons_still_missing', ()))}")
    print(f"Seasons targeted:      {list(report.get('seasons_targeted', ()))}")

    if report.get("dry_run"):
        print(f"DRY RUN -- requests issued: {report['requests_issued']}")
        for row in report["per_season"]:
            print(
                f"  {row['season']}: {row['games']} games, "
                f"{row['would_fetch']} would be fetched, "
                f"{row['written_without_a_call']} written with no call"
            )
        print(f"Games to fetch:             {report['games_to_fetch']}")
        print(f"Games written with no call: {report['games_without_a_call']}")
        print(
            f"Projected weighted calls:   "
            f"{report['projected_weighted_calls']:.1f} "
            f"(retries assumed: {report['retries_assumed']}; hourly cap "
            f"{report['hourly_budget']}, daily cap {report['daily_budget']})"
        )
        print(
            f"Projected wall clock:       "
            f"{report['projected_seconds'] / 60:.1f} minutes at "
            f"{report['interval_seconds']} s per request (a FLOOR: a retried game "
            "costs more)"
        )
        return

    for season_report in report.get("seasons", ()):
        print(
            f"  {season_report['season']}: {season_report['games']} rows, "
            f"{season_report['fetched']} fetched, "
            f"{season_report['written_without_a_call']} written with no call, "
            f"null arm {season_report['null_arm']}, "
            f"weighted calls so far {season_report['weighted_calls_after']}"
        )
    if "weighted_calls" in report:
        print(
            f"Weighted calls OBSERVED: {report['weighted_calls']} over "
            f"{report['attempts']} attempt(s) -- this INCLUDES retries, unlike the "
            "zero-retry projection --dry-run prints"
        )
    promotion = report.get("promotion")
    if promotion:
        print(
            f"Promoted {promotion['rows']} rows from {promotion['snapshots']} "
            f"snapshot(s) into {promotion['silver_path']}"
        )
    crosscheck = report.get("crosscheck")
    if crosscheck:
        _print_crosscheck_report(crosscheck)


if __name__ == "__main__":
    main()
