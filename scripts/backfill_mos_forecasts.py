"""Backfill the archived day-before NWS forecast bulletins into bronze (Plan 33.2-11, D33.2-12).

WHAT THIS DOES
--------------
For every 2002-2025 game at a US outdoor or retractable-roof venue, the forecast that stood at the
game's lock is the 12 UTC MOS bulletin of the ET calendar day before kickoff (D33.2-13), from the
airport ``config/mos_stations.py`` assigns to the venue. This script fetches those bulletins from
the free, key-less IEM archive and lands the RAW rows in ``data/bronze/mos/``. Plan 33.2-12
regenerates silver weather from that bronze with zero network calls (CLAUDE.md reproducibility).
It writes NOTHING outside ``bronze/mos/``: silver, gold and artifacts do not move.

THE REQUEST UNIT: (station, season, MODEL SEGMENT)
--------------------------------------------------
One request per station per season would be ~888 requests, but one model-specific request cannot
span the AVN -> GFS changeover (run date 2003-12-16), which falls inside the 2003 season. So the
unit is the segment: the maximal run of a station-season's lock dates served by ONE model, with its
date range clipped at the changeover. Every season but 2003 is one segment; a 2003 station-season
whose lock dates straddle the changeover is two requests, AVN then GFS. The count is computed and
printed (``SEGMENTS_PLANNED=`` / ``SPLIT_SEASON_REQUESTS=``), never asserted as a literal.

THE TWO REFUSALS, AND ONE RECORDED ABSENCE
------------------------------------------
* An HTTP 200 with ZERO rows raises :class:`EmptyMosResponseError` by name (pitfall P4). It is
  never written as "no weather".
* A row whose ``model``, ``station`` or ``runtime`` is not the one requested raises
  :class:`MosModelMismatchError`: the archive labels pre-changeover rows AVN even under a GFS
  request, and storing them under GFS would mislabel a lineage.
* A NON-EMPTY, correctly labelled response that lacks the 12 UTC run of a lock date is a gap in
  the archive, not the silent-empty trap -- the same response carries that station's other runs.
  Found on the first real run: the GFS 12 UTC cycle of 2020-01-03 is missing for KHOU, KIAH and
  KDFW while their 06 and 18 UTC cycles are present. The run is re-asked ONCE through the
  single-run endpoint (``MOS_RUN_ENDPOINT_URL``). If it answers, those rows are used. If it too
  holds nothing for the station, the run is recorded in bronze as ABSENT FROM THE ARCHIVE (a row
  with ``absent_in_archive=True`` and no bulletin); the game it serves is UNRESOLVED, takes SPEC
  R6's NULL-plus-flag path, and counts against the pre-registered 99-in-100 coverage bound. No
  other run (the earlier 06 UTC cycle, NAM) is ever substituted: D33.2-13 admits exactly one run
  per game.

VALIDATE BEFORE THE BRONZE WRITE; RESUME ON COVERAGE
----------------------------------------------------
Every kept row is decoded, and every game the segment serves is turned into a WeatherSchema row
and pushed through ``validate_bronze_to_silver`` plus the per-column round-trip assertion, BEFORE
the bronze file is written -- the ``scripts/backfill_historical_weather.py`` pattern. A segment is
complete when every one of its lock dates has its 12 UTC run in bronze (COVERAGE, not file
existence, per ``season_is_covered`` at ``scripts/backfill_historical_weather.py:737-772``), so an
interrupted run resumes at the first incomplete segment and a re-run over complete bronze makes
zero requests and writes nothing.

THE OUTPUT CONTRACT, printed on EVERY run including a no-op one: ``SEGMENTS_PLANNED``,
``SPLIT_SEASON_REQUESTS``, ``SEGMENTS_ALREADY_COMPLETE``, ``SEGMENTS_FETCHED``, ``REQUESTS_MADE``,
``SEGMENTS_MISSING``, ``EMPTY_RESPONSES`` and ``RESOLVED_SHARE``. ``SEGMENTS_MISSING`` and
``RESOLVED_SHARE`` are computed over WHAT BRONZE HOLDS after the run, so they mean the same thing
on a run that fetched nothing because nothing was left.

No money is spent: IEM is free, needs no key and permits commercial use.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import contextlib
import fnmatch
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import httpx
import pandas as pd
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

import utils.game_lock as lock_rule
from config.mos_stations import (
    COVERED_ROOF_TYPES,
    UNCOVERABLE_NON_US_STADIUM_IDS,
    station_for_venue,
)
from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from data.upstream_pin import SEALED_THROUGH_SEASON
from features.schedule_moves import facts_at_lock
from scripts.backfill_historical_weather import LEGACY_WEATHER_BRONZE_FILENAMES
from scripts.mos_decode import (
    RUN_HOUR_UTC,
    EmptyMosResponseError,
    MosModelMismatchError,
    MosRecord,
    build_weather_record,
    decode_record,
    model_for_run_date,
    run_instant,
)
from utils import DataIngestionError

__all__ = [
    "CORPUS_FIRST_SEASON",
    "CORPUS_LAST_SEASON",
    "MOS_BRONZE_GLOB",
    "MOS_BRONZE_SUBDIR",
    "MOS_BRONZE_TABLE",
    "MOS_COLUMNS",
    "MOS_ENDPOINT_URL",
    "MOS_RUN_ENDPOINT_URL",
    "BackfillReport",
    "CoveredGame",
    "MosSegment",
    "UncoverableGame",
    "absent_runs_in_bronze",
    "assert_mos_columns_survived",
    "confirm_run",
    "load_bronze_run_records",
    "load_corpus",
    "plan_segments",
    "run_backfill",
    "select_lock_runs",
]

#: Endpoint A: date-range download, ONE station per request (a comma list returns an empty 200).
MOS_ENDPOINT_URL: str = "https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py"

#: Endpoint B: single-run lookup. Used only to re-ask for a run endpoint A's response lacked.
MOS_RUN_ENDPOINT_URL: str = "https://mesonet.agron.iastate.edu/api/1/mos.json"

#: Endpoint B's answer for a run it holds nothing for: HTTP 404 with this detail (2026-09-21).
RUN_ENDPOINT_NO_RESULTS_DETAIL: str = "Database query found no results"

#: The bronze landing zone, RELATIVE to ``<data root>/bronze``. A subdirectory no other writer
#: uses, so this backfill's files cannot collide with any existing bronze table.
MOS_BRONZE_SUBDIR: str = "mos"
MOS_BRONZE_TABLE: str = "mos_bulletins"
MOS_BRONZE_GLOB: str = f"{MOS_BRONZE_TABLE}_raw_bronze_*.parquet"

# THE FILENAME-COLLISION TRAP, named (the scripts/backfill_historical_weather.py:124-184 shape):
# a resume rule whose glob matched another table's files would read foreign rows as coverage.
# The subdirectory already separates the name spaces; this makes the disjointness a checked fact.
_COLLIDING_LEGACY_FILENAMES = tuple(
    name
    for name in LEGACY_WEATHER_BRONZE_FILENAMES
    if fnmatch.fnmatch(name, MOS_BRONZE_GLOB)
)
if _COLLIDING_LEGACY_FILENAMES:  # pragma: no cover -- an import-time impossibility
    raise RuntimeError(
        f"the MOS bronze glob {MOS_BRONZE_GLOB!r} matches legacy weather bronze files "
        f"{_COLLIDING_LEGACY_FILENAMES}"
    )

#: The corpus: the SEALED upstream zone. The live season belongs to the live forecast path.
CORPUS_FIRST_SEASON: int = 2002
CORPUS_LAST_SEASON: int = SEALED_THROUGH_SEASON

#: The columns this ingest adds to WeatherSchema (declared in the same task that emits them).
MOS_COLUMNS: tuple[str, ...] = (
    "forecast_issue_time",
    "mos_model",
    "mos_station",
    "mos_qpf_category",
    "mos_precip_level",
)

#: Politeness delay between requests. No published rate limit exists (RESEARCH A8); ~60 probe
#: requests at 0.3-1.0 s were never throttled.
REQUEST_INTERVAL_SECONDS: float = 0.5

#: A 429 with no Retry-After header waits this long before the retry.
DEFAULT_RETRY_AFTER_SECONDS: float = 30.0

REQUEST_TIMEOUT_SECONDS: float = 120.0

USER_AGENT: str = "nfl-predict-mos-backfill/1.0 (single-user research tool)"

_LOCK_FILENAME = ".mos_backfill.lock"


class MosBackfillError(DataIngestionError):
    """The backfill refused, by name, before writing anything for the offending segment."""


class MosRateLimitedError(RuntimeError):
    """HTTP 429: retried after the server's Retry-After, with exponential backoff."""


@dataclass(frozen=True)
class CoveredGame:
    """A game whose venue at the lock is a US outdoor or retractable-roof stadium."""

    game_id: str
    season: int
    stadium_id: str
    station: str
    kickoff_utc: datetime
    lock_date: date


@dataclass(frozen=True)
class UncoverableGame:
    """A game with no US forecast station: played abroad. Honest no-forecast, never a stand-in."""

    game_id: str
    season: int
    stadium_id: str


@dataclass(frozen=True)
class MosSegment:
    """One request: a station-season's lock dates served by ONE model."""

    station: str
    season: int
    model: str
    lock_dates: tuple[date, ...]

    @property
    def sts(self) -> str:
        return f"{self.lock_dates[0].isoformat()}T{RUN_HOUR_UTC:02d}:00Z"

    @property
    def ets(self) -> str:
        return f"{self.lock_dates[-1].isoformat()}T{RUN_HOUR_UTC:02d}:01Z"

    @property
    def file_stem(self) -> str:
        return (
            f"{MOS_BRONZE_TABLE}_raw_bronze_{self.station}_{self.season}_{self.model}"
        )

    @property
    def run_instants(self) -> tuple[datetime, ...]:
        return tuple(run_instant(d) for d in self.lock_dates)


@dataclass
class BackfillReport:
    """The output contract. Printed on every run, including a no-op and a refusal."""

    segments_planned: int = 0
    split_season_requests: int = 0
    segments_already_complete: int = 0
    segments_fetched: int = 0
    requests_made: int = 0
    segments_missing: int = 0
    empty_responses: int = 0
    absent_runs: int = 0
    absent_runs_in_bronze: tuple[str, ...] = ()
    covered_games: int = 0
    resolved_games: int = 0
    uncoverable_games: int = 0
    stations: int = 0
    refusal: str | None = None
    written: list[str] = field(default_factory=list)

    @property
    def resolved_share(self) -> float:
        return self.resolved_games / self.covered_games if self.covered_games else 0.0

    def lines(self) -> list[str]:
        out = [
            f"SEGMENTS_PLANNED= {self.segments_planned}",
            f"SPLIT_SEASON_REQUESTS= {self.split_season_requests}",
            f"SEGMENTS_ALREADY_COMPLETE= {self.segments_already_complete}",
            f"SEGMENTS_FETCHED= {self.segments_fetched}",
            f"REQUESTS_MADE= {self.requests_made}",
            f"SEGMENTS_MISSING= {self.segments_missing}",
            f"EMPTY_RESPONSES= {self.empty_responses}",
            f"RESOLVED_SHARE= {self.resolved_share:.4f}",
            f"ABSENT_RUNS_RECORDED_THIS_RUN= {self.absent_runs}",
            f"ABSENT_RUNS_IN_BRONZE= {list(self.absent_runs_in_bronze)}",
            f"COVERED_GAMES= {self.covered_games}",
            f"RESOLVED_GAMES= {self.resolved_games}",
            f"UNCOVERABLE_GAMES= {self.uncoverable_games}",
            f"STATIONS= {self.stations}",
        ]
        if self.refusal is not None:
            out.append(f"REFUSAL= {self.refusal}")
        return out


# ---------------------------------------------------------------------------
# The corpus and the request plan
# ---------------------------------------------------------------------------


def _resolve_data_root(base_path: Path | str | None) -> Path:
    if base_path is not None:
        return Path(base_path)
    from conf.settings import get_settings

    return Path(get_settings().config.data.root_path)


def _load_venues() -> dict[str, dict[str, Any]]:
    path = Path(__file__).resolve().parent.parent / "data" / "venues.json"
    records = json.loads(path.read_text(encoding="utf-8"))["venues"]
    return {str(r["stadium_id"]): r for r in records}


def load_corpus(
    games: pd.DataFrame,
    venues: Mapping[str, Mapping[str, Any]] | None = None,
    *,
    schedule_moves_path: Path | str | None = None,
) -> tuple[list[CoveredGame], list[UncoverableGame]]:
    """Split *games* into covered games and uncoverable (non-US) games.

    The venue is the one in force AT THE LOCK (``features.schedule_moves.facts_at_lock``), so a
    game moved by an emergency announced after its lock uses its PRE-MOVE venue's station. The
    lock is the one rule (``utils.game_lock.game_lock``) applied to the recorded kickoff. Indoor
    games are neither: weather does not apply under a closed roof and nothing is fetched.

    Raises:
        MosBackfillError: a game's venue has no record in ``data/venues.json``.
    """
    venues = _load_venues() if venues is None else venues
    kwargs = {} if schedule_moves_path is None else {"table_path": schedule_moves_path}
    covered: list[CoveredGame] = []
    uncoverable: list[UncoverableGame] = []
    for row in games.to_dict("records"):
        game_id = str(row["game_id"])
        facts = facts_at_lock(game_id, row, **kwargs)
        stadium_id = str(facts.weather_stadium_id)
        venue = venues.get(stadium_id)
        if venue is None:
            raise MosBackfillError(
                f"{game_id}: venue {stadium_id!r} has no venues.json record"
            )
        if str(venue["roof_type"]) not in COVERED_ROOF_TYPES:
            continue
        if venue["country"] != "USA" or stadium_id in UNCOVERABLE_NON_US_STADIUM_IDS:
            uncoverable.append(UncoverableGame(game_id, int(row["season"]), stadium_id))
            continue
        kickoff = pd.Timestamp(facts.kickoff_et).tz_convert(UTC).to_pydatetime()
        covered.append(
            CoveredGame(
                game_id=game_id,
                season=int(row["season"]),
                stadium_id=stadium_id,
                station=station_for_venue(stadium_id),
                kickoff_utc=kickoff,
                lock_date=lock_rule.game_lock(
                    row["kickoff_et"], game_id=game_id
                ).date(),
            )
        )
    return covered, uncoverable


def plan_segments(covered: Iterable[CoveredGame]) -> list[MosSegment]:
    """One segment per (station, season, model), split at the AVN -> GFS changeover.

    A segment's lock dates are the station-season's lock dates served by one model; its request
    range runs from the first to the last of them, so it can never cross 2003-12-16.
    """
    dates_by_key: dict[tuple[str, int], set[date]] = defaultdict(set)
    for game in covered:
        dates_by_key[(game.station, game.season)].add(game.lock_date)
    segments: list[MosSegment] = []
    for (station, season), lock_dates in sorted(dates_by_key.items()):
        by_model: dict[str, list[date]] = defaultdict(list)
        for lock_date in sorted(lock_dates):
            by_model[model_for_run_date(lock_date)].append(lock_date)
        for model in sorted(by_model, key=lambda m: by_model[m][0]):
            segments.append(MosSegment(station, season, model, tuple(by_model[model])))
    return segments


# ---------------------------------------------------------------------------
# Bronze: coverage, reading, writing
# ---------------------------------------------------------------------------


def _bronze_dir(base_path: Path | str | None) -> Path:
    return _resolve_data_root(base_path) / "bronze" / MOS_BRONZE_SUBDIR


def _segment_files(segment: MosSegment, base_path: Path | str | None) -> list[Path]:
    return sorted(_bronze_dir(base_path).glob(f"{segment.file_stem}_*.parquet"))


def _runtimes_in_bronze(segment: MosSegment, base_path: Path | str | None) -> set[str]:
    present: set[str] = set()
    for path in _segment_files(segment, base_path):
        frame = pd.read_parquet(path, columns=["runtime_raw"], engine="pyarrow")
        present.update(str(v) for v in frame["runtime_raw"])
    return present


def _runtime_raw(instant: datetime) -> str:
    return instant.strftime("%Y-%m-%dT%H:%M:%S.000")


def segment_is_complete(segment: MosSegment, base_path: Path | str | None) -> bool:
    """COVERAGE: is every lock date's 12 UTC run of this segment present in bronze?"""
    present = _runtimes_in_bronze(segment, base_path)
    return all(_runtime_raw(r) in present for r in segment.run_instants)


def load_bronze_run_records(
    base_path: Path | str | None = None,
) -> dict[tuple[str, datetime], list[MosRecord]]:
    """Every bronze bulletin row, decoded, keyed by (station, runtime). Zero network calls."""
    runs: dict[tuple[str, datetime], dict[datetime, MosRecord]] = defaultdict(dict)
    for path in sorted(_bronze_dir(base_path).glob(MOS_BRONZE_GLOB)):
        frame = pd.read_parquet(path, columns=["raw_json"], engine="pyarrow")
        for text in frame["raw_json"].dropna():  # an absent-run marker has no bulletin
            record = decode_record(json.loads(text))
            runs[(record.station, record.runtime)][record.ftime] = record
    return {key: sorted(v.values(), key=lambda r: r.ftime) for key, v in runs.items()}


def absent_runs_in_bronze(base_path: Path | str | None = None) -> tuple[str, ...]:
    """Every run recorded ABSENT FROM THE ARCHIVE, as ``STATION MODEL RUNTIME``, sorted."""
    found: set[str] = set()
    for path in sorted(_bronze_dir(base_path).glob(MOS_BRONZE_GLOB)):
        frame = pd.read_parquet(path, engine="pyarrow")
        if "absent_in_archive" not in frame.columns:
            continue
        for row in frame[frame["absent_in_archive"].astype(bool)].itertuples():
            found.add(f"{row.station} {row.model} {row.runtime_raw}")
    return tuple(sorted(found))


def _write_bronze_atomically(frame: pd.DataFrame, target: Path) -> None:
    """Write *target* whole or not at all, and never over an existing file (append-only)."""
    if target.exists():
        raise FileExistsError(f"bronze is append-only; {target} already exists")
    tmp = target.with_name(target.name + ".partial")
    frame.to_parquet(tmp, engine="pyarrow", index=False)
    try:
        os.link(
            tmp, target
        )  # fails, rather than overwrites, if target appeared meanwhile
    finally:
        tmp.unlink(missing_ok=True)


@contextlib.contextmanager
def _corpus_lock(base_path: Path | str | None) -> Iterator[None]:
    """An exclusive lock over bronze/mos while this run writes.

    Removed on ANY exit, including an exception. Unlike the ERA5 backfill's lock, this one does
    not need to outlive a crash: each segment is written whole or not at all, and resume is by
    coverage, so an interrupted run leaves bronze consistent and the next run fetches only what
    is missing.
    """
    directory = _bronze_dir(base_path)
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory / _LOCK_FILENAME
    try:
        with lock.open("xb") as handle:
            handle.write(f"pid={os.getpid()}\n".encode("ascii"))
    except FileExistsError as error:
        raise MosBackfillError(
            f"another MOS backfill holds {lock}. If no backfill is running, delete it."
        ) from error
    try:
        yield
    finally:
        lock.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Fetch, check, validate
# ---------------------------------------------------------------------------


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code == 429:
        retry_after = response.headers.get("Retry-After")
        try:
            wait = float(retry_after) if retry_after else DEFAULT_RETRY_AFTER_SECONDS
        except ValueError:
            wait = DEFAULT_RETRY_AFTER_SECONDS
        time.sleep(min(wait, 300.0))
        raise MosRateLimitedError(f"HTTP 429 from {response.request.url}")
    response.raise_for_status()


def _is_retryable(error: BaseException) -> bool:
    """Transport failures, 429 and 5xx are retried; a 4xx and every CONTENT refusal are not."""
    if isinstance(error, (httpx.TransportError, MosRateLimitedError)):
        return True
    return (
        isinstance(error, httpx.HTTPStatusError) and error.response.status_code >= 500
    )


@retry(
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    retry=retry_if_exception(_is_retryable),
    reraise=True,
)
def _get(
    client: httpx.Client,
    params: Mapping[str, str] | Sequence[tuple[str, str]],
    url: str = MOS_ENDPOINT_URL,
) -> httpx.Response:
    """One GET, retried on transport errors, 429 and 5xx -- never on content.

    The single-run endpoint answers a run it holds nothing for with HTTP 404 and the detail
    :data:`RUN_ENDPOINT_NO_RESULTS_DETAIL` (probed 2026-09-21). That exact answer is returned,
    not raised: it IS the confirmation that the run is absent. Any other 404 still raises.
    """
    query = list(params) if isinstance(params, Sequence) else list(params.items())
    response = client.get(url, params=query)
    if (
        url == MOS_RUN_ENDPOINT_URL
        and response.status_code == 404
        and RUN_ENDPOINT_NO_RESULTS_DETAIL in response.text
    ):
        return response
    _raise_for_status(response)
    return response


def _segment_params(segment: MosSegment) -> dict[str, str]:
    return {
        "station": segment.station,
        "model": segment.model,
        "sts": segment.sts,
        "ets": segment.ets,
        "format": "json",
    }


def select_lock_runs(
    rows: Sequence[Mapping[str, Any]], segment: MosSegment
) -> tuple[list[dict[str, Any]], list[str]]:
    """The segment's 12 UTC lock-date rows, and the lock runs this response lacks.

    Returns:
        ``(kept, missing)``: the kept rows, and the archive ``runtime`` strings of the lock-date
        12 UTC runs absent from this non-empty, correctly labelled response.

    Raises:
        EmptyMosResponseError: *rows* is empty (a 200 with zero rows).
        MosModelMismatchError: a row names another station or model.
    """
    if not rows:
        raise EmptyMosResponseError(
            f"EMPTY MOS response for {segment.station} {segment.model} "
            f"{segment.sts}..{segment.ets}: HTTP 200 with zero rows. Refusing; this is never "
            "written as 'no weather'."
        )
    wrong = sorted(
        {(str(r.get("station")), str(r.get("model"))) for r in rows}
        - {(segment.station, segment.model)}
    )
    if wrong:
        raise MosModelMismatchError(
            f"{segment.station} {segment.model} {segment.season}: the response carries rows "
            f"for {wrong}. A request never carries a model its whole range is not served by."
        )
    wanted = {_runtime_raw(r) for r in segment.run_instants}
    kept = [dict(r) for r in rows if str(r.get("runtime")) in wanted]
    missing = sorted(wanted - {str(r["runtime"]) for r in kept})
    return kept, missing


def confirm_run(
    client: httpx.Client, segment: MosSegment, runtime_raw: str
) -> tuple[list[dict[str, Any]], str]:
    """Re-ask ONE missing run through the single-run endpoint.

    Returns:
        ``(rows, url)``: the run's rows for the segment's station -- empty when the archive holds
        none for it, a CONFIRMED absence -- and the URL asked.

    Raises:
        MosModelMismatchError: a returned row names another station, model or runtime.
    """
    runtime_param = runtime_raw.replace("T", " ")[:16] + "Z"
    params = [
        ("station", segment.station),
        ("model", segment.model),
        ("runtime", runtime_param),
    ]
    response = _get(client, params, MOS_RUN_ENDPOINT_URL)
    if response.status_code == 404:  # the endpoint's own "no results" answer
        return [], str(response.request.url)
    rows = [dict(r) for r in response.json().get("data", [])]
    wrong = sorted(
        {
            (str(r.get("station")), str(r.get("model")), str(r.get("runtime")))
            for r in rows
        }
        - {(segment.station, segment.model, runtime_raw)}
    )
    if wrong:
        raise MosModelMismatchError(
            f"{segment.station} {segment.model} {runtime_raw}: the single-run lookup "
            f"returned rows for {wrong}"
        )
    return rows, str(response.request.url)


def assert_mos_columns_survived(frame: pd.DataFrame, source: pd.DataFrame) -> None:
    """Refuse a promoted frame that lost any MOS column at the schema gate.

    Copied per column from ``assert_weather_coverage_survived``: ``validate_bronze_to_silver``
    rebuilds each row through the schema and Pydantic v2 drops an undeclared column silently.
    Checks presence AND that no value the source carried came back null.

    Raises:
        DataIngestionError: a column is absent, or lost a value.
    """
    for column in MOS_COLUMNS:
        if column not in frame.columns:
            raise DataIngestionError(
                f"the promoted weather frame has NO {column} column. It was emitted by "
                "scripts.mos_decode.build_weather_record and is gone after "
                "validate_bronze_to_silver, which means it is not declared on "
                "data.schemas.WeatherSchema (Pydantic v2 extra='ignore' drops it silently)."
            )
        lost = int((source[column].notna() & frame[column].isna()).sum())
        if lost:
            raise DataIngestionError(
                f"{column}: {lost} value(s) present before the schema gate are null after it"
            )


def _validate_segment(
    kept: Sequence[Mapping[str, Any]],
    games: Sequence[CoveredGame],
) -> None:
    """Decode every kept row and prove every served game promotes, BEFORE the bronze write."""
    runs: dict[datetime, list[MosRecord]] = defaultdict(list)
    for raw in kept:
        record = decode_record(raw)
        runs[record.runtime].append(record)
    rows = []
    for game in games:
        built = build_weather_record(
            game.game_id, game.kickoff_utc, runs.get(run_instant(game.lock_date), [])
        )
        if built is not None:
            rows.append(built)
    if not rows:
        return
    source = pd.DataFrame(rows)
    promoted = validate_bronze_to_silver(source, WeatherSchema)
    assert_mos_columns_survived(promoted, source)


def _bronze_frame(
    kept: Sequence[Mapping[str, Any]],
    segment: MosSegment,
    response: httpx.Response,
    fetched_at: datetime,
    *,
    confirmed: Mapping[str, str] | None = None,
    absent: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """The bronze rows: every kept bulletin row verbatim, plus one marker per absent run.

    *confirmed* maps a runtime recovered through the single-run endpoint to the URL that
    answered; *absent* maps a runtime the archive holds nothing for to the URL that confirmed it.
    """
    confirmed = confirmed or {}
    absent = absent or {}
    records: list[dict[str, Any]] = [
        {
            "runtime_raw": str(r["runtime"]),
            "ftime_raw": str(r["ftime"]),
            "raw_json": json.dumps(r, sort_keys=True),
            "absent_in_archive": False,
            "request_url": confirmed.get(str(r["runtime"]), str(response.request.url)),
        }
        for r in kept
    ]
    records += [
        {
            "runtime_raw": runtime_raw,
            "ftime_raw": None,
            "raw_json": None,
            "absent_in_archive": True,
            "request_url": url,
        }
        for runtime_raw, url in sorted(absent.items())
    ]
    frame = pd.DataFrame(records)
    frame.insert(0, "station", segment.station)
    frame.insert(1, "season", segment.season)
    frame.insert(2, "model", segment.model)
    frame["response_sha256"] = hashlib.sha256(response.content).hexdigest()
    frame["fetched_at_utc"] = pd.Timestamp(fetched_at)
    return frame


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _count_resolved(
    covered: Sequence[CoveredGame],
    runs: Mapping[tuple[str, datetime], Sequence[MosRecord]],
) -> int:
    return sum(
        build_weather_record(
            game.game_id,
            game.kickoff_utc,
            runs.get((game.station, run_instant(game.lock_date)), []),
        )
        is not None
        for game in covered
    )


def _load_games(seasons: Sequence[int], base_path: Path | str | None) -> pd.DataFrame:
    path = _resolve_data_root(base_path) / "silver" / "games.parquet"
    games = pd.read_parquet(path, engine="pyarrow")
    return games[games["season"].isin(list(seasons))].reset_index(drop=True)


def run_backfill(
    seasons: Sequence[int],
    *,
    apply: bool,
    base_path: Path | str | None = None,
    client: httpx.Client | None = None,
    sleep: Callable[[float], None] = time.sleep,
    games: pd.DataFrame | None = None,
    schedule_moves_path: Path | str | None = None,
) -> BackfillReport:
    """Plan, fetch what is missing (only with *apply*), and report over what bronze holds."""
    for season in seasons:
        if not CORPUS_FIRST_SEASON <= season <= CORPUS_LAST_SEASON:
            raise MosBackfillError(
                f"season {season} is outside [{CORPUS_FIRST_SEASON}, {CORPUS_LAST_SEASON}]"
            )
    games = _load_games(seasons, base_path) if games is None else games
    covered, uncoverable = load_corpus(games, schedule_moves_path=schedule_moves_path)
    segments = plan_segments(covered)
    report = BackfillReport(
        segments_planned=len(segments),
        split_season_requests=len(segments)
        - len({(s.station, s.season) for s in segments}),
        covered_games=len(covered),
        uncoverable_games=len(uncoverable),
        stations=len({s.station for s in segments}),
    )
    games_by_segment: dict[tuple[str, int, str], list[CoveredGame]] = defaultdict(list)
    for game in covered:
        games_by_segment[
            (game.station, game.season, model_for_run_date(game.lock_date))
        ].append(game)

    incomplete = [s for s in segments if not segment_is_complete(s, base_path)]
    report.segments_already_complete = len(segments) - len(incomplete)

    if apply and incomplete:
        owned_client = client is None
        http = client or httpx.Client(
            timeout=REQUEST_TIMEOUT_SECONDS, headers={"User-Agent": USER_AGENT}
        )
        try:
            with _corpus_lock(base_path):
                for segment in incomplete:
                    if report.requests_made:
                        sleep(REQUEST_INTERVAL_SECONDS)
                    report.requests_made += 1
                    response = _get(http, _segment_params(segment))
                    fetched_at = datetime.now(UTC)
                    confirmed: dict[str, str] = {}
                    absent: dict[str, str] = {}
                    try:
                        kept, missing = select_lock_runs(response.json(), segment)
                        for runtime_raw in missing:
                            sleep(REQUEST_INTERVAL_SECONDS)
                            report.requests_made += 1
                            rows, url = confirm_run(http, segment, runtime_raw)
                            if rows:
                                kept += rows
                                confirmed[runtime_raw] = url
                            else:
                                absent[runtime_raw] = url
                    except EmptyMosResponseError as error:
                        report.empty_responses += 1
                        report.refusal = str(error)
                        break
                    except MosModelMismatchError as error:
                        report.refusal = str(error)
                        break
                    report.absent_runs += len(absent)
                    _validate_segment(
                        kept,
                        games_by_segment[
                            (segment.station, segment.season, segment.model)
                        ],
                    )
                    stamp = fetched_at.strftime("%Y%m%dT%H%M%S%f")
                    target = (
                        _bronze_dir(base_path) / f"{segment.file_stem}_{stamp}.parquet"
                    )
                    frame = _bronze_frame(
                        kept,
                        segment,
                        response,
                        fetched_at,
                        confirmed=confirmed,
                        absent=absent,
                    )
                    _write_bronze_atomically(frame, target)
                    report.written.append(str(target))
                    report.segments_fetched += 1
        finally:
            if owned_client:
                http.close()

    report.segments_missing = sum(
        not segment_is_complete(s, base_path) for s in segments
    )
    report.resolved_games = _count_resolved(covered, load_bronze_run_records(base_path))
    report.absent_runs_in_bronze = absent_runs_in_bronze(base_path)
    return report


def _parse_seasons(text: str) -> list[int]:
    first, _, last = text.partition("-")
    start, end = int(first), int(last or first)
    if start > end:
        raise argparse.ArgumentTypeError(f"--seasons {text!r}: start after end")
    return list(range(start, end + 1))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--seasons", type=_parse_seasons, required=True, help="e.g. 2002-2025"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="fetch and write; without it, plan and report only",
    )
    parser.add_argument(
        "--data-root", default=None, help="data lake root (default: settings)"
    )
    args = parser.parse_args(argv)
    report = run_backfill(args.seasons, apply=args.apply, base_path=args.data_root)
    for line in report.lines():
        print(line)
    return 1 if report.refusal is not None else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
