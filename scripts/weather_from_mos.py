"""Regenerate 2002-2025 silver ``weather`` and ``weather_features`` from the bronze MOS bulletins.

Plan 33.2-12 Task 1 (SPEC R6, D33.2-02 / D33.2-12 / D33.2-13), p332_ rung 4's silver half.

WHAT IT REPLACES. Every 2002-2025 silver weather row was an ERA5 reanalysis OBSERVATION of the
game itself (``weather_source = "archive"``). Training on what happened while serving a forecast
is the train/serve mismatch this phase exists to remove. Each row is replaced -- not blended --
by the forecast as it stood at the game's lock: the 12 UTC NWS MOS bulletin of the ET calendar
day before kickoff, decoded from ``data/bronze/mos/`` (Plan 33.2-11) by the ONE mapping,
``scripts.mos_decode.build_weather_record``.

THREE KINDS OF ROW, ALL ``weather_source = "historical_forecast"``:

* FORECAST -- a US outdoor or retractable-roof game whose bulletin resolved: the decoded values,
  ``weather_coverage`` True, ``forecast_issue_time`` the 12 UTC cycle. ``precip_mm`` stays NULL
  (a bulletin carries no amount); the QPF category is kept as its LEVEL. ``humidity_pct`` comes
  from temperature and dew point by the August-Roche-Magnus identity (the decoder's).
* ABSENCE -- a game where weather applies but no forecast exists: a venue abroad (MOS covers
  only the US, Puerto Rico and the US Virgin Islands; no stand-in station is ever used) or a US
  game whose bulletin did not resolve (a run confirmed absent from the archive, or one that does
  not reach the kickoff). ``weather_coverage`` False, every measurement NULL, NO issue time.
* DOME -- an indoor venue, or a retractable roof that was CLOSED for this game:
  ``is_outdoor`` False, ``weather_coverage`` True (a dome is not missing weather, it has none),
  every measurement NULL, NO issue time.

WHETHER WEATHER APPLIES is Phase 33.1's per-game rule (SPEC R4,
``scripts.backfill_historical_weather.WeatherBackfiller._per_game_roof_is_outdoor``), read at
the venue in force at the lock: an indoor venue never, an outdoor venue always, and a
retractable venue by the game's OWN roof in the sealed pinned feed ("closed" means no weather).
Where the feed's row describes a different stadium than the lock venue (the corrected 2025
games abroad), its roof says nothing about this venue, so a retractable lock venue counts as
weather-applies. Plan 33.2-11's bronze corpus fetched bulletins for every retractable-venue
game; a closed-roof game's bulletin is simply not used.

An issue time on an absence or a dome would be provenance for a forecast that does not exist.

The venue is the one in force AT THE LOCK (``features.schedule_moves.facts_at_lock``, through
``scripts.backfill_mos_forecasts.load_corpus``), so ``2003_W08_MIA@LAC`` takes SDG00's station,
not Tempe's, and the seven 2025 games abroad are absences.

ZERO NETWORK CALLS. Everything is read from bronze and silver on disk; the caller runs this under
:func:`deny_network`, which counts and refuses any socket connection, and prints the count.

DETERMINISTIC. ``created_at`` is the latest bronze capture instant, not the wall clock, and
both tables are written in a fixed row order, so a re-run over the same bronze writes the same
bytes.

The 2026 rows (the LIVE Open-Meteo path) are left exactly as they are.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import contextlib
import json
import re
import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

import utils.game_lock as lock_rule
from data.quality_gates import validate_bronze_to_silver
from data.schemas import WeatherSchema
from data.storage import upsert_silver
from features.schedule_moves import facts_at_lock
from features.weather import OBSERVATION_WEATHER_SOURCES, WeatherFeaturesCalculator
from scripts.backfill_historical_weather import load_pinned_game_facts
from scripts.backfill_mos_forecasts import (
    CORPUS_FIRST_SEASON,
    CORPUS_LAST_SEASON,
    MOS_BRONZE_GLOB,
    MOS_BRONZE_SUBDIR,
    assert_mos_columns_survived,
    load_bronze_run_records,
    load_corpus,
)
from scripts.ingest_weather import NFLVERSE_ROOF_MAP
from scripts.mos_decode import WEATHER_SOURCE_MOS, build_weather_record, run_instant

__all__ = [
    "NetworkCallRefused",
    "RegenerationReport",
    "bronze_capture_instant",
    "deny_network",
    "regenerate_history",
    "regenerated_weather_rows",
]

WEATHER_TABLE = "weather"
WEATHER_FEATURES_TABLE = "weather_features"

#: Why a regenerated row carries no forecast, for the published NULL list. Not a silver column.
REASON_NON_US_VENUE = "non_us_venue"
REASON_NO_RESOLVED_BULLETIN = "no_resolved_bulletin"

_CAPTURE_STAMP = re.compile(r"_(\d{8}T\d{12})\.parquet$")


class NetworkCallRefused(RuntimeError):
    """A socket connection was attempted during an offline regeneration."""


@dataclass
class RegenerationReport:
    """What a regeneration wrote, and the counts its output contract prints."""

    games: int = 0
    forecast_rows: int = 0
    absence_rows: int = 0
    dome_rows: int = 0
    observation_rows: int = 0
    absence_reasons: dict[str, str] = field(default_factory=dict)
    network_calls: int = 0

    def lines(self) -> list[str]:
        by_reason: dict[str, int] = {}
        for reason in self.absence_reasons.values():
            by_reason[reason] = by_reason.get(reason, 0) + 1
        return [
            f"GAMES= {self.games}",
            f"FORECAST_ROWS= {self.forecast_rows}",
            f"ABSENCE_ROWS= {self.absence_rows}",
            f"ABSENCE_BY_REASON= {dict(sorted(by_reason.items()))}",
            f"DOME_ROWS= {self.dome_rows}",
            f"OBSERVATION_ROWS= {self.observation_rows}",
            f"NETWORK_CALLS= {self.network_calls}",
        ]


@contextlib.contextmanager
def deny_network(counter: list[int]) -> Iterator[None]:
    """Refuse, and count, every socket connection made inside the block.

    ``counter`` is a one-element list the caller reads afterwards. A refused connection raises
    :class:`NetworkCallRefused`, so an offline run that tried to reach the network fails loudly
    AND says so in the count rather than quietly succeeding from a cache.
    """
    original_connect = socket.socket.connect
    original_create = socket.create_connection

    def refuse(*_args: Any, **_kwargs: Any) -> Any:
        counter[0] += 1
        raise NetworkCallRefused("network access attempted during an offline rebuild")

    socket.socket.connect = refuse  # type: ignore[method-assign]
    socket.create_connection = refuse  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.create_connection = original_create


def bronze_capture_instant(data_root: Path) -> datetime:
    """The latest capture instant stamped into a ``bronze/mos/`` file name, in UTC.

    Used as every regenerated row's ``created_at`` so the regeneration is a function of the
    bronze it reads, not of when it ran.

    Raises:
        FileNotFoundError: bronze holds no MOS file.
    """
    stamps = [
        datetime.strptime(match.group(1), "%Y%m%dT%H%M%S%f").replace(tzinfo=UTC)
        for path in (data_root / "bronze" / MOS_BRONZE_SUBDIR).glob(MOS_BRONZE_GLOB)
        if (match := _CAPTURE_STAMP.search(path.name))
    ]
    if not stamps:
        raise FileNotFoundError(
            f"no MOS bronze under {data_root / 'bronze' / MOS_BRONZE_SUBDIR}"
        )
    return max(stamps)


def _forecast_less_row(
    game_id: str,
    kickoff_utc: datetime,
    lock_date: Any,
    *,
    is_outdoor: bool,
    created_at: datetime,
) -> dict[str, Any]:
    """An absence (``is_outdoor`` True) or a dome (False): no measurement, no issue time.

    ``forecast_time`` is the 12 UTC day-before cycle the game's forecast would have come from
    -- a schema-required field -- while ``forecast_issue_time``, the provenance field, stays
    NULL because no forecast exists.
    """
    return {
        "game_id": game_id,
        "forecast_time": run_instant(lock_date),
        "game_time": kickoff_utc,
        "is_outdoor": is_outdoor,
        "weather_coverage": not is_outdoor,
        "weather_source": WEATHER_SOURCE_MOS,
        "created_at": created_at,
    }


def _derived_flags(row: dict[str, Any]) -> dict[str, Any]:
    """The three silver convenience flags, from the forecast values (NULL when absent)."""
    temp, wind = row.get("temp_f"), row.get("wind_mph")
    prob, level = row.get("precip_prob"), row.get("mos_precip_level")
    known = prob is not None or level is not None
    return {
        "is_cold": None if temp is None else temp < 32.0,
        "is_windy": None if wind is None else wind > 12.0,
        "is_precipitation": None
        if not known
        else bool(
            (prob is not None and prob > 0.3) or (level is not None and level >= 1)
        ),
    }


def _venue_roof_types() -> dict[str, str]:
    path = Path(__file__).resolve().parent.parent / "data" / "venues.json"
    records = json.loads(path.read_text(encoding="utf-8"))["venues"]
    return {str(r["stadium_id"]): str(r["roof_type"]) for r in records}


def closed_roof_game_ids(
    games: pd.DataFrame, lock_venues: dict[str, str]
) -> frozenset[str]:
    """Games at a retractable lock venue whose OWN roof in the pinned feed was closed.

    Args:
        games: The games (``game_id``, ``season``).
        lock_venues: ``game_id -> stadium_id`` of the venue in force at the lock.

    Raises:
        KeyError: a pinned roof value outside ``NFLVERSE_ROOF_MAP`` (never defaulted).
    """
    roof_types = _venue_roof_types()
    pinned = load_pinned_game_facts(sorted({int(s) for s in games["season"]}))
    pinned = pinned.set_index("game_id")
    closed: set[str] = set()
    for game_id, stadium_id in lock_venues.items():
        if roof_types[stadium_id] != "retractable" or game_id not in pinned.index:
            continue
        feed = pinned.loc[game_id]
        if str(feed["stadium_id"]) != stadium_id:
            continue  # the feed's roof describes another stadium
        if NFLVERSE_ROOF_MAP[str(feed["roof"]).lower().strip()] == "indoor":
            closed.add(game_id)
    return frozenset(closed)


def regenerated_weather_rows(
    games: pd.DataFrame, data_root: Path
) -> tuple[pd.DataFrame, RegenerationReport]:
    """One silver weather row per 2002-2025 game in *games*, from bronze. Pure: writes nothing.

    Raises:
        scripts.backfill_mos_forecasts.MosBackfillError: a game's venue has no record.
    """
    games = games[games["season"].between(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON)]
    covered, uncoverable = load_corpus(games)
    covered_by_id = {game.game_id: game for game in covered}
    uncoverable_ids = {game.game_id for game in uncoverable}
    closed_roof = closed_roof_game_ids(
        games,
        {g.game_id: g.stadium_id for g in covered}
        | {g.game_id: g.stadium_id for g in uncoverable},
    )
    runs = load_bronze_run_records(data_root)
    created_at = bronze_capture_instant(data_root)

    report = RegenerationReport(games=len(games))
    rows: list[dict[str, Any]] = []
    for game in games.to_dict("records"):
        game_id = str(game["game_id"])
        covered_game = covered_by_id.get(game_id)
        if covered_game is not None and game_id not in closed_roof:
            records = runs.get(
                (covered_game.station, run_instant(covered_game.lock_date))
            )
            built = (
                build_weather_record(game_id, covered_game.kickoff_utc, records)
                if records
                else None
            )
            if built is not None:
                rows.append(
                    {**built, **_derived_flags(built), "created_at": created_at}
                )
                report.forecast_rows += 1
                continue
            report.absence_reasons[game_id] = REASON_NO_RESOLVED_BULLETIN
            rows.append(
                _forecast_less_row(
                    game_id,
                    covered_game.kickoff_utc,
                    covered_game.lock_date,
                    is_outdoor=True,
                    created_at=created_at,
                )
            )
            report.absence_rows += 1
            continue
        facts = facts_at_lock(game_id, game)
        kickoff_utc = pd.Timestamp(facts.kickoff_et).tz_convert(UTC).to_pydatetime()
        lock_date = lock_rule.game_lock(game["kickoff_et"], game_id=game_id).date()
        abroad = game_id in uncoverable_ids and game_id not in closed_roof
        if abroad:
            report.absence_reasons[game_id] = REASON_NON_US_VENUE
            report.absence_rows += 1
        else:
            report.dome_rows += 1
        rows.append(
            _forecast_less_row(
                game_id,
                kickoff_utc,
                lock_date,
                is_outdoor=abroad,
                created_at=created_at,
            )
        )

    # A key a row does not carry is ABSENT: None, never pandas' NaN, which the schema would
    # read as a value of the wrong type in its string and boolean fields.
    frame = pd.DataFrame(rows)
    frame = frame.astype(object).where(frame.notna(), None)
    promoted = validate_bronze_to_silver(frame, WeatherSchema)
    assert_mos_columns_survived(promoted, frame)
    report.observation_rows = int(
        promoted["weather_source"].isin(OBSERVATION_WEATHER_SOURCES).sum()
    )
    return promoted, report


def _order_weather(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["game_time", "game_id"], kind="mergesort")


def _order_weather_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.sort_values(["season", "week", "game_id"], kind="mergesort")


def regenerate_history(
    data_root: Path, *, build_instant: datetime | None = None
) -> RegenerationReport:
    """Rewrite the 2002-2025 rows of silver ``weather`` and ``weather_features`` from bronze.

    Writes exactly ``silver/weather.parquet``, ``silver/weather_features.parquet`` and, through
    ``upsert_silver``, their DuckDB copies under *data_root*. Rows for other seasons (the live
    2026 rows) are kept as they are.

    Args:
        data_root: The data root to read bronze and silver from and write silver to.
        build_instant: The weather fence's build-instant bound; ``None`` means now. It moves no
            historical value (every bulletin predates its lock, which predates any build).
    """
    games = pd.read_parquet(data_root / "silver" / "games.parquet", engine="pyarrow")
    history = games[games["season"].between(CORPUS_FIRST_SEASON, CORPUS_LAST_SEASON)]
    weather_rows, report = regenerated_weather_rows(history, data_root)

    upsert_silver(
        weather_rows, WEATHER_TABLE, base_path=data_root, order_rows=_order_weather
    )
    stored = pd.read_parquet(
        data_root / "silver" / f"{WEATHER_TABLE}.parquet", engine="pyarrow"
    )

    calculator = WeatherFeaturesCalculator()
    features = calculator.build_weather_features(
        history,
        weather_df=stored,
        build_instant=build_instant if build_instant is not None else datetime.now(UTC),
    )
    upsert_silver(
        features,
        WEATHER_FEATURES_TABLE,
        base_path=data_root,
        order_rows=_order_weather_features,
    )
    return report
