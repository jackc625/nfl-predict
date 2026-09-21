"""Which NWS forecast station answers for which stadium (Plan 33.2-11, D33.2-12).

The archived day-before forecasts are airport bulletins (MOS: AVN, then GFS MAV). Each stadium
needs ONE airport whose bulletin stands in for the stadium's weather. This module is that map.

HOW THE MAP WAS MADE: DERIVED, THEN FROZEN
-----------------------------------------
The station for a venue is the NEAREST PRIMARY AIRPORT to the venue's latitude/longitude in
``data/venues.json`` (great-circle distance). "Primary" means an airport with scheduled commercial
passenger service; general-aviation relievers (Teterboro, KTEB) and military/research fields
(Moffett, KNUQ) are deliberately NOT candidates, because their bulletins describe a small field
rather than the metro's weather of record.

The derivation is :func:`derive_venue_stations`. Its output was then FROZEN below as
:data:`VENUE_STATIONS`, so the map is committed source rather than a computation that moves when a
coordinate is edited. ``tests/unit/test_mos_station_map.py`` re-derives it from ``data/venues.json``
and asserts equality, so the frozen map and the rule cannot silently part company: a venue edit that
changes the nearest airport fails the test and forces a deliberate re-freeze.

The candidate airports' coordinates are recorded in :data:`CANDIDATE_AIRPORTS`, taken from the IEM
ASOS station metadata (``https://mesonet.agron.iastate.edu/geojson/network/{STATE}_ASOS.geojson``,
fetched 2026-09-21). Every station the map uses was PROBED present in both eras the backfill reads
(AVN 2002-09-15 12Z and GFS 2024-11-17 12Z, 2026-09-21): :data:`PROBED_PRESENT_STATIONS`. KCGX
(Chicago Meigs, closed 2003) and KGEU (Glendale AZ, absent in 2002) are not candidates.

RESEARCH ASSUMPTION A2 (LOW-MEDIUM), RECORDED RATHER THAN RESOLVED
-----------------------------------------------------------------
Station AVAILABILITY is verified. "Nearest sensible airport" is a JUDGMENT. The arguable choices:

* ``BOS00`` (Foxborough): the rule gives KBOS (36.6 km); RESEARCH 6.2 proposed KPVD (43.3 km).
  Logan is coastal and Foxborough is inland; Providence is nearer the coast too. The rule stands.
* ``MIA00`` (Miami Gardens): the rule gives KFLL (15.5 km); RESEARCH 6.2 proposed KMIA (20.5 km).
* ``DAL99`` (Texas Stadium, Irving): the rule gives KDAL (5.6 km); RESEARCH 6.2 proposed KDFW
  (13.5 km).
* ``NYC00``/``NYC01`` (East Rutherford): KEWR (16.4-16.6 km) over KLGA (16.8-16.9 km); KTEB is
  nearer (5.3 km) but is not a candidate.
* ``SFO01`` (Santa Clara): KSJC (6.3 km); KNUQ is 7.0 km but is not a candidate.
* ``LAX97`` (Carson): KLAX (14.2 km); Long Beach is nearer but is not a primary airport.

Plan 33.2-11's comparison against the on-disk observations surfaces a bad choice as an outlier
cluster; the fix for a bad choice is this map, never the tolerance.

WHAT CANNOT BE COVERED
----------------------
The MAV bulletin "is valid for stations in the United States, Puerto Rico, and the U.S. Virgin
Islands" (https://vlab.noaa.gov/web/mdl/mav-card). A game played abroad therefore has NO forecast
station at all. :data:`UNCOVERABLE_NON_US_STADIUM_IDS` is every non-US venue in ``data/venues.json``;
a game there produces no request and takes SPEC R6's honest no-forecast path. No stand-in station is
ever used.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

__all__ = [
    "CANDIDATE_AIRPORTS",
    "COVERED_ROOF_TYPES",
    "EXCLUDED_NON_PRIMARY_AIRPORTS",
    "PROBED_PRESENT_STATIONS",
    "UNCOVERABLE_NON_US_STADIUM_IDS",
    "VENUE_STATIONS",
    "StationChoice",
    "derive_venue_stations",
    "great_circle_km",
    "station_for_venue",
]

#: Roof types a forecast is fetched for. A dome ("indoor") gets none: weather does not apply.
COVERED_ROOF_TYPES: frozenset[str] = frozenset({"outdoor", "retractable"})

#: Primary (scheduled commercial service) airports, ICAO -> (latitude, longitude). From the IEM
#: ASOS station metadata, fetched 2026-09-21, rounded to four decimals.
CANDIDATE_AIRPORTS: dict[str, tuple[float, float]] = {
    "KATL": (33.6301, -84.4418),
    "KBNA": (36.1189, -86.6892),
    "KBOS": (42.3606, -71.0097),
    "KBTR": (30.5372, -91.1469),
    "KBUF": (42.9408, -78.7358),
    "KBWI": (39.1733, -76.6841),
    "KCLE": (41.4056, -81.8519),
    "KCLT": (35.2226, -80.9543),
    "KCMI": (40.0324, -88.2755),
    "KCVG": (39.0431, -84.6717),
    "KDAL": (32.8471, -96.8518),
    "KDCA": (38.8472, -77.0346),
    "KDEN": (39.8328, -104.6575),
    "KDFW": (32.8968, -97.0380),
    "KEWR": (40.6827, -74.1693),
    "KFLL": (26.0787, -80.1622),
    "KGRB": (44.4794, -88.1367),
    "KHOU": (29.6375, -95.2824),
    "KIAD": (38.9348, -77.4473),
    "KIAH": (29.9844, -95.3607),
    "KIND": (39.7084, -86.3048),
    "KJAX": (30.4941, -81.6879),
    "KJFK": (40.6386, -73.7622),
    "KLAX": (33.9382, -118.3865),
    "KLGA": (40.7794, -73.8803),
    "KMCI": (39.2976, -94.7139),
    "KMDW": (41.7860, -87.7524),
    "KMIA": (25.7880, -80.3169),
    "KMSP": (44.8854, -93.2313),
    "KOAK": (37.7178, -122.2330),
    "KORD": (41.9602, -87.9316),
    "KPHL": (39.8734, -75.2266),
    "KPHX": (33.4343, -112.0116),
    "KPIT": (40.4915, -80.2329),
    "KPVD": (41.7219, -71.4325),
    "KSAN": (32.7339, -117.1845),
    "KSEA": (47.4447, -122.3144),
    "KSFO": (37.6190, -122.3749),
    "KSJC": (37.3594, -121.9244),
    "KTPA": (27.9619, -82.5403),
}

#: Airports nearer to some venue than its chosen station, excluded as NOT primary. Recorded so the
#: exclusion is a stated decision rather than an accident of the candidate list.
EXCLUDED_NON_PRIMARY_AIRPORTS: dict[str, str] = {
    "KTEB": "Teterboro: general-aviation reliever, no scheduled commercial service",
    "KNUQ": "Moffett Federal Airfield: military/NASA research field",
    "KCGX": "Chicago Meigs Field: closed 2003, absent from the archive after it",
    "KGEU": "Glendale AZ: absent from the 2002 AVN archive",
}

#: Stations probed present in BOTH eras the backfill reads -- AVN 2002-09-15 12Z and GFS
#: 2024-11-17 12Z -- through https://mesonet.agron.iastate.edu/api/1/mos.json on 2026-09-21.
PROBED_PRESENT_STATIONS: frozenset[str] = frozenset(
    {
        "KATL", "KBNA", "KBOS", "KBTR", "KBUF", "KBWI", "KCLE", "KCLT", "KCMI", "KCVG",
        "KDAL", "KDCA", "KDEN", "KDFW", "KEWR", "KFLL", "KGRB", "KHOU", "KIND", "KJAX",
        "KLAX", "KMCI", "KMDW", "KMIA", "KMSP", "KOAK", "KPHL", "KPHX", "KPIT", "KPVD",
        "KSAN", "KSEA", "KSFO", "KSJC", "KTPA",
    }
)  # fmt: skip


@dataclass(frozen=True)
class StationChoice:
    """The station chosen for one venue, and how far away it is."""

    station: str
    distance_km: float


#: THE FROZEN MAP: every US outdoor or retractable-roof venue in ``data/venues.json`` that any
#: 2002-2025 game used, to its nearest primary airport. Output of :func:`derive_venue_stations`,
#: frozen 2026-09-21; distances in km, rounded to one decimal.
VENUE_STATIONS: dict[str, StationChoice] = {
    "ATL97": StationChoice("KATL", 14.4),
    "BAL00": StationChoice("KBWI", 12.8),
    "BOS00": StationChoice("KBOS", 36.6),
    "BRG00": StationChoice("KBTR", 14.4),
    "BUF00": StationChoice("KBUF", 19.0),
    "CAR00": StationChoice("KCLT", 9.2),
    "CHI98": StationChoice("KMDW", 14.1),
    "CHI99": StationChoice("KCMI", 8.2),
    "CIN00": StationChoice("KCVG", 14.6),
    "CLE00": StationChoice("KCLE", 16.9),
    "DAL00": StationChoice("KDFW", 17.4),
    "DAL99": StationChoice("KDAL", 5.6),
    "DEN00": StationChoice("KDEN", 32.5),
    "GNB00": StationChoice("KGRB", 6.4),
    "HOU00": StationChoice("KHOU", 13.5),
    "IND00": StationChoice("KIND", 13.3),
    "JAX00": StationChoice("KJAX", 19.5),
    "KAN00": StationChoice("KMCI", 34.0),
    "LAX97": StationChoice("KLAX", 14.2),
    "LAX99": StationChoice("KLAX", 12.4),
    "MIA00": StationChoice("KFLL", 15.5),
    "MIN98": StationChoice("KMSP", 10.1),
    "NAS00": StationChoice("KBNA", 9.1),
    "NYC00": StationChoice("KEWR", 16.4),
    "NYC01": StationChoice("KEWR", 16.6),
    "OAK00": StationChoice("KOAK", 4.7),
    "PHI00": StationChoice("KPHL", 5.9),
    "PHI99": StationChoice("KPHL", 6.0),
    "PHO00": StationChoice("KPHX", 25.5),
    "PHO99": StationChoice("KPHX", 7.4),
    "PIT00": StationChoice("KPIT", 19.0),
    "SDG00": StationChoice("KSAN", 8.2),
    "SEA00": StationChoice("KSEA", 16.8),
    "SFO00": StationChoice("KSFO", 10.6),
    "SFO01": StationChoice("KSJC", 6.3),
    "TAM00": StationChoice("KTPA", 4.0),
    "WAS00": StationChoice("KDCA", 16.2),
}

#: Every non-US venue in ``data/venues.json``: no MAV bulletin exists for any of them.
UNCOVERABLE_NON_US_STADIUM_IDS: frozenset[str] = frozenset(
    {
        "BER00", "BUF01", "DUB00", "FRA00", "GER00", "LON00", "LON01", "LON02",
        "MAD01", "MEL00", "MEX00", "MUN01", "PAR00", "RIO00", "SAO00",
    }
)  # fmt: skip

_EARTH_RADIUS_KM = 6371.0088


def great_circle_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in kilometres between two latitude/longitude points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    h = (
        math.sin(d_phi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_KM * math.asin(math.sqrt(h))


def derive_venue_stations(
    venues: Iterable[Mapping[str, object]],
    stadium_ids: Iterable[str],
    candidates: Mapping[str, tuple[float, float]] = CANDIDATE_AIRPORTS,
) -> dict[str, StationChoice]:
    """The nearest candidate airport for each of *stadium_ids*, from venue coordinates.

    Args:
        venues: ``data/venues.json`` records (``stadium_id``, ``latitude``, ``longitude``).
        stadium_ids: The venues to map.
        candidates: ICAO -> (latitude, longitude). Defaults to :data:`CANDIDATE_AIRPORTS`.

    Returns:
        stadium_id -> :class:`StationChoice`, distance rounded to one decimal.

    Raises:
        KeyError: a requested stadium_id has no venue record.
    """
    by_id = {str(record["stadium_id"]): record for record in venues}
    derived: dict[str, StationChoice] = {}
    for stadium_id in stadium_ids:
        record = by_id[stadium_id]
        lat, lon = float(record["latitude"]), float(record["longitude"])  # type: ignore[arg-type]
        distance, station = min(
            (great_circle_km(lat, lon, a_lat, a_lon), icao)
            for icao, (a_lat, a_lon) in candidates.items()
        )
        derived[stadium_id] = StationChoice(station, round(distance, 1))
    return derived


def station_for_venue(stadium_id: str) -> str:
    """The frozen station for a covered venue.

    Raises:
        KeyError: naming the venue when it is not in the frozen map. A non-US venue is never
            given a stand-in; a US venue missing here means the map must be re-derived.
    """
    try:
        return VENUE_STATIONS[stadium_id].station
    except KeyError:
        reason = (
            "it is outside the USA, so no MAV bulletin exists for it"
            if stadium_id in UNCOVERABLE_NON_US_STADIUM_IDS
            else "it is not in the frozen map; re-derive with derive_venue_stations"
        )
        raise KeyError(
            f"no forecast station for venue {stadium_id!r}: {reason}"
        ) from None
