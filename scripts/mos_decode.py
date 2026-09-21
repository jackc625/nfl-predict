"""Decode one archived NWS MOS bulletin row, correctly (Plan 33.2-11, D33.2-12/13).

The IEM archive (``https://mesonet.agron.iastate.edu/cgi-bin/request/mos.py``) returns one JSON
object per forecast hour. Four traps in that payload (RESEARCH 6.1, pitfall P6) are each handled
exactly once, here, so the backfill and the comparison cannot decode it two different ways:

1. ``runtime`` and ``ftime`` are NAIVE strings (UTC by convention, no offset). They are localized
   to UTC explicitly by :func:`parse_archive_instant`; a naive value never leaves this module
   (pitfall P5).
2. ``wdr`` is ALREADY in degrees (IEM multiplied the bulletin's tens of degrees by ten). It is
   NOT multiplied again: a value above 360 raises rather than being wrapped.
3. ``wsp`` is in KNOTS. It becomes mph exactly once, by :data:`KNOTS_TO_MPH`.
4. The 2002 AVN payload has NO ``snw`` key at all (absent, not null). A missing key parses; the
   record keeps ``snw_key_present`` so an absent key and a null value stay distinguishable.

And one unit trap that is not about arithmetic: ``q06`` is an ordinal QPF CATEGORY, not an amount,
despite IEM's variable picker labelling it ``[0.01 in]``. It becomes a precipitation LEVEL through
the pre-registered anchoring in ``config/mos_tolerance.py`` and never a millimetre value.

WHICH RUN, AND WHY ONLY ONE (D33.2-13)
--------------------------------------
Every past game uses the 12 UTC cycle of the ET calendar day before kickoff -- the day of its
18:00 ET lock. The argument that this cycle is public before the lock, and that the 18 UTC cycle
is never used, is committed in ``config/mos_tolerance.py`` (owner ruling ``commit-argument``,
2026-09-21). Do NOT reintroduce "the latest admissible run": it would switch from the 12 UTC to
the 18 UTC bulletin at the daylight-saving change (pitfall P9).

The guidance lineage switches on the RUN DATE, not the season: AVN before 2003-12-16, GFS (MAV)
from 2003-12-16, so the changeover falls inside the 2003 season. Probed 2026-09-21: the archive's
last AVN run is 2003-12-16 06 UTC and its first GFS run is 2003-12-16 12 UTC, and a GFS request
over earlier dates returns rows LABELLED AVN -- which is why every row's ``model`` is checked.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any

from config.mos_tolerance import QPF_CATEGORY_TO_PRECIP_LEVEL, QPF_MISSING_CATEGORY

__all__ = [
    "GFS_ERA_START",
    "KNOTS_TO_MPH",
    "MAX_NEAREST_FTIME_OFFSET",
    "MODEL_AVN",
    "MODEL_GFS",
    "RUN_HOUR_UTC",
    "WEATHER_SOURCE_MOS",
    "EmptyMosResponseError",
    "MosDecodeError",
    "MosModelMismatchError",
    "MosRecord",
    "build_weather_record",
    "decode_record",
    "model_for_run_date",
    "nearest_record",
    "parse_archive_instant",
    "precip_period_record",
    "qpf_precip_level",
    "relative_humidity_pct",
    "run_instant",
]

#: International knot to statute mph. Applied ONCE, in :func:`decode_record`.
KNOTS_TO_MPH: float = 1.15078

MODEL_AVN: str = "AVN"
MODEL_GFS: str = "GFS"

#: The first RUN DATE served by the GFS MAV guidance (https://mesonet.agron.iastate.edu/mos/).
GFS_ERA_START: date = date(2003, 12, 16)

#: The one cycle every past game uses (D33.2-13).
RUN_HOUR_UTC: int = 12

#: The MAV bulletin is three-hourly over the lead times used here, so the forecast hour nearest
#: any kickoff is at most 90 minutes away. Anything further means the run does not reach the
#: kickoff and the game is NOT resolved -- never filled from a more distant hour.
MAX_NEAREST_FTIME_OFFSET: timedelta = timedelta(minutes=90)

#: A 6-hour precipitation period covers the kickoff when it ends at most this long after it.
PRECIP_PERIOD: timedelta = timedelta(hours=6)

#: The weather_source vocabulary member a decoded bulletin carries: what the forecast said at the
#: time, for a game that has since been played.
WEATHER_SOURCE_MOS: str = "historical_forecast"

_ARCHIVE_FORMAT = "%Y-%m-%dT%H:%M:%S.%f"


class MosDecodeError(ValueError):
    """A bulletin row breaks a decoding rule (a unit trap, a bad instant, an unknown code)."""


class EmptyMosResponseError(RuntimeError):
    """The archive answered HTTP 200 with ZERO rows (the silent-empty trap, pitfall P4).

    Raised by name, never written as "no weather". IEM returns an empty 200 both for a
    comma-separated station list and for a run that does not exist.
    """


class MosModelMismatchError(RuntimeError):
    """A row's ``model`` or ``station`` is not the one requested.

    The archive answers a GFS request over pre-2003-12-16 dates with rows labelled AVN; storing
    them under the requested model would mislabel one lineage as the other.
    """


def model_for_run_date(run_date: date) -> str:
    """AVN before 2003-12-16, GFS from 2003-12-16 -- by RUN DATE, never by season."""
    return MODEL_GFS if run_date >= GFS_ERA_START else MODEL_AVN


def run_instant(lock_date: date) -> datetime:
    """The 12 UTC cycle instant of *lock_date* (the ET calendar day of the game's lock)."""
    return datetime(
        lock_date.year, lock_date.month, lock_date.day, RUN_HOUR_UTC, tzinfo=UTC
    )


def parse_archive_instant(text: object, *, field: str) -> datetime:
    """An archive ``runtime``/``ftime`` string as a UTC-AWARE instant.

    The archive writes ``2023-12-16T12:00:00.000`` with no offset; it is UTC by the archive's
    convention, so it is localized to UTC HERE, explicitly, once. A string that already carries
    an offset, or any other shape, is refused: this parser accepts exactly the archive's form.

    Raises:
        MosDecodeError: *text* is not the archive's naive ISO form.
    """
    if not isinstance(text, str):
        raise MosDecodeError(f"{field} {text!r} is not a string")
    try:
        # The archive's value is UTC by convention: localized here, explicitly, never relabelled
        # later. strptime without %z refuses a string that carries an offset.
        return datetime.strptime(text, _ARCHIVE_FORMAT).replace(tzinfo=UTC)
    except ValueError as error:
        raise MosDecodeError(
            f"{field} {text!r} is not the archive's naive ISO form {_ARCHIVE_FORMAT!r}"
        ) from error


def _optional_number(raw: Mapping[str, Any], key: str) -> float | None:
    value = raw.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise MosDecodeError(f"{key} {value!r} is not a number")
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _wind_direction_degrees(raw: Mapping[str, Any]) -> float | None:
    """``wdr`` in degrees, NOT multiplied again. 360 (north) is stored as 0."""
    value = _optional_number(raw, "wdr")
    if value is None:
        return None
    if not 0.0 <= value <= 360.0:
        raise MosDecodeError(
            f"wdr {value!r} is outside 0-360 degrees. The archive already stores DEGREES; a "
            "value above 360 means it was multiplied by ten a second time."
        )
    return 0.0 if value == 360.0 else value


def relative_humidity_pct(
    temp_f: float | None, dew_point_f: float | None
) -> float | None:
    """Relative humidity from temperature and dew point (August-Roche-Magnus identity).

    An identity, not an imputation (RESEARCH assumption A4): the bulletins carry temperature
    and dew point, and RH follows from the two. Coefficients 17.625 / 243.04 C (Alduchov and
    Eskridge 1996). A dew point above the temperature -- a whole-degree rounding artifact in
    the bulletin -- saturates at 100 rather than exceeding it.
    """
    if temp_f is None or dew_point_f is None:
        return None
    t_c = (temp_f - 32.0) * 5.0 / 9.0
    td_c = (dew_point_f - 32.0) * 5.0 / 9.0
    a, b = 17.625, 243.04
    rh = 100.0 * math.exp(a * td_c / (b + td_c) - a * t_c / (b + t_c))
    return round(min(rh, 100.0), 1)


def qpf_precip_level(category: int | None) -> int | None:
    """The pre-registered precipitation LEVEL of a QPF category (owner ruling qpf-midpoint).

    ``None`` (no QPF for this period) and ``9`` (the bulletin's missing code) both return
    ``None``: an unknown level, never level 0.

    Raises:
        MosDecodeError: *category* is not 0-5 or 9.
    """
    if category is None or category == QPF_MISSING_CATEGORY:
        return None
    if category not in QPF_CATEGORY_TO_PRECIP_LEVEL:
        raise MosDecodeError(f"q06 {category!r} is not a TPB 482 QPF category")
    return QPF_CATEGORY_TO_PRECIP_LEVEL[category]


def _qpf_category(raw: Mapping[str, Any]) -> int | None:
    value = _optional_number(raw, "q06")
    if value is None:
        return None
    if value != int(value):
        raise MosDecodeError(
            f"q06 {value!r} is not an integer category. It is an ORDINAL category, never an "
            "amount in hundredths of an inch."
        )
    category = int(value)
    qpf_precip_level(category)  # validates the code
    return category


@dataclass(frozen=True)
class MosRecord:
    """One decoded forecast hour. Every instant is UTC-aware; every unit is converted."""

    station: str
    model: str
    runtime: datetime
    ftime: datetime
    temp_f: float | None
    dew_point_f: float | None
    wind_mph: float | None
    wind_direction: float | None
    humidity_pct: float | None
    cloud_category: str | None
    p06_pct: float | None
    qpf_category: int | None
    precip_level: int | None
    snw_key_present: bool
    snw: float | None


def decode_record(raw: Mapping[str, Any]) -> MosRecord:
    """Decode one archive row. Every unit trap is handled here and nowhere else.

    Raises:
        MosDecodeError: the row breaks a decoding rule.
    """
    station = raw.get("station")
    model = raw.get("model")
    if not isinstance(station, str) or not station.strip():
        raise MosDecodeError(f"station {station!r} is not a station id")
    if model not in (MODEL_AVN, MODEL_GFS):
        raise MosDecodeError(f"model {model!r} is not {MODEL_AVN!r} or {MODEL_GFS!r}")
    temp_f = _optional_number(raw, "tmp")
    dew_point_f = _optional_number(raw, "dpt")
    wsp_knots = _optional_number(raw, "wsp")
    if wsp_knots is not None and wsp_knots < 0:
        raise MosDecodeError(f"wsp {wsp_knots!r} is negative")
    cloud = raw.get("cld")
    snw_key_present = "snw" in raw
    qpf_category = _qpf_category(raw)
    return MosRecord(
        station=station.strip(),
        model=str(model),
        runtime=parse_archive_instant(raw.get("runtime"), field="runtime"),
        ftime=parse_archive_instant(raw.get("ftime"), field="ftime"),
        temp_f=temp_f,
        dew_point_f=dew_point_f,
        wind_mph=None if wsp_knots is None else wsp_knots * KNOTS_TO_MPH,
        wind_direction=_wind_direction_degrees(raw),
        humidity_pct=relative_humidity_pct(temp_f, dew_point_f),
        cloud_category=cloud.strip()
        if isinstance(cloud, str) and cloud.strip()
        else None,
        p06_pct=_optional_number(raw, "p06"),
        qpf_category=qpf_category,
        precip_level=qpf_precip_level(qpf_category),
        snw_key_present=snw_key_present,
        snw=_optional_number(raw, "snw") if snw_key_present else None,
    )


def nearest_record(records: Iterable[MosRecord], target: datetime) -> MosRecord | None:
    """The forecast hour nearest *target* (ties go to the EARLIER hour), within 90 minutes.

    Returns ``None`` when no hour lies within :data:`MAX_NEAREST_FTIME_OFFSET`: the run does not
    reach the kickoff, and a more distant hour is never substituted.

    Raises:
        MosDecodeError: *target* is naive.
    """
    if target.tzinfo is None:
        raise MosDecodeError(f"target {target!r} is naive; a lock comparison needs UTC")
    best = min(records, key=lambda r: (abs(r.ftime - target), r.ftime), default=None)
    if best is None or abs(best.ftime - target) > MAX_NEAREST_FTIME_OFFSET:
        return None
    return best


def precip_period_record(
    records: Sequence[MosRecord], kickoff: datetime
) -> MosRecord | None:
    """The record whose 6-hour precipitation period CONTAINS *kickoff*.

    ``p06``/``q06`` describe the six hours ENDING at ``ftime`` and appear only at six-hourly
    marks, so the covering period is the earliest such mark at or after kickoff, within six
    hours of it. ``None`` when no period covers the kickoff.
    """
    candidates = [
        r
        for r in records
        if (r.p06_pct is not None or r.qpf_category is not None)
        and kickoff <= r.ftime < kickoff + PRECIP_PERIOD
    ]
    return min(candidates, key=lambda r: r.ftime, default=None)


def build_weather_record(
    game_id: str,
    kickoff_utc: datetime,
    run_records: Sequence[MosRecord],
) -> dict[str, Any] | None:
    """One WeatherSchema-shaped row for a game, from the decoded records of its 12 UTC run.

    Returns ``None`` when the run does not reach the kickoff (no hour within 90 minutes) or the
    nearest hour carries no temperature or wind: the game is then NOT resolved. Plan 33.2-12
    owns the silver write; this is the ONE mapping from a bulletin to a weather row, used by the
    backfill's validate-before-write step and by the decode comparison alike.

    Raises:
        MosDecodeError: *kickoff_utc* is naive, or the records mix runs or stations.
    """
    if kickoff_utc.tzinfo is None:
        raise MosDecodeError(f"{game_id}: kickoff {kickoff_utc!r} is naive")
    if len({(r.runtime, r.station, r.model) for r in run_records}) > 1:
        raise MosDecodeError(f"{game_id}: records from more than one run or station")
    nearest = nearest_record(run_records, kickoff_utc)
    if nearest is None or nearest.temp_f is None or nearest.wind_mph is None:
        return None
    period = precip_period_record(run_records, kickoff_utc)
    p06 = None if period is None else period.p06_pct
    category = None if period is None else period.qpf_category
    return {
        "game_id": game_id,
        "forecast_time": nearest.runtime,
        "game_time": kickoff_utc,
        "temp_f": nearest.temp_f,
        "temp_c": round((nearest.temp_f - 32.0) * 5.0 / 9.0, 1),
        "wind_mph": round(nearest.wind_mph, 2),
        "wind_direction": nearest.wind_direction,
        "humidity_pct": nearest.humidity_pct,
        "dew_point_f": nearest.dew_point_f,
        "precip_prob": None if p06 is None else p06 / 100.0,
        "precip_mm": None,
        "cloud_cover_pct": None,
        "is_outdoor": True,
        "weather_coverage": True,
        "weather_source": WEATHER_SOURCE_MOS,
        "forecast_issue_time": nearest.runtime,
        "mos_model": nearest.model,
        "mos_station": nearest.station,
        "mos_qpf_category": category,
        "mos_precip_level": qpf_precip_level(category),
    }
