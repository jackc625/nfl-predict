#!/usr/bin/env python3
"""Derive and table the 22 historical venue records, for owner ratification.

Phase 33.1, Plan 33.1-01 Task 1 (R1, Rulings A/B/C/C2).

WHAT THIS COMMAND IS FOR
------------------------
`data/venues.json` holds 38 records. The pinned 2002-2025 schedules carry 55 distinct
`stadium_id` values over 6,499 games, so 22 stadiums covering 1,082 games have no record
at all -- Giants Stadium, Qualcomm, the Georgia Dome, the Metrodome, Candlestick, Texas
Stadium, the RCA Dome, Twickenham, Allianz Arena and the rest. Until they exist, every
historical game resolves by its PRESENT-DAY home team, which would hand 141 outdoor
Oakland Coliseum games Las Vegas desert weather under an indoor roof.

This command does not write those records. It TABLES them, so a human can ratify them,
and Plan 33.1-01 Task 3 generates `data/venues.json` from the ratified constant. The
governing rule is Plan 33-06's (`33-06-SUMMARY.md:40`): a reference value nothing in the
repository can settle is RATIFIED by the owner, committed ONCE as a constant, and the data
file is GENERATED from that constant -- so the file cannot drift from the record that
authorised it.

WHY THIS IS ITS OWN MODULE AND NOT AN EDIT TO EITHER WEATHER MODULE
--------------------------------------------------------------------
`api.open-meteo.com` is the FORECAST host. `tests/unit/test_weather_archive_quarantined.py`
scans `scripts/ingest_weather.py` and `features/weather.py` structurally, and adding a new
Open-Meteo endpoint constant to either of them is the kind of edit that invites a later
reader to reach for the wrong host from the live path. The new host lands in a module
neither of those two imports.

PROVENANCE IS PER FIELD, NOT PER VENUE (Ruling C2)
---------------------------------------------------
`surface` is read at `features/contextual.py:345-356` and feeds the grass/turf mismatch
feature; `capacity` is read at `:653` and feeds `venue_capacity` / `venue_large_stadium`.
A Wikidata `P625 coordinate location` statement substantiates a latitude and a longitude
and says NOTHING about what the playing surface was in 2003 or how many seats the building
held. So the provenance record is keyed `(stadium_id, field, source)` across all EIGHT
non-identity fields, and it distinguishes two KINDS:

* EXTERNALLY SOURCED -- `latitude`, `longitude`, `surface`, `capacity`. Each needs its own
  revision-pinned external citation (an `oldid=` or a Wikidata `Q` entity id). Four fields
  x 22 records = 88 cells the owner ratifies.
* DERIVED -- `roof_type` (from the pinned feed), `timezone` and `elevation_ft` (from the
  Open-Meteo API), `climate_zone` (from the stated same-metro-sibling rule). Each carries
  the citation the deriving function emits, which is reproducible rather than ratified.

`assert_every_external_field_is_sourced` runs BEFORE anything is printed or written and
REFUSES with every uncovered `<stadium_id>.<field>` pair NAMED -- never a count, because
the operator's next action is to go and source exactly those cells.

THIS COMMAND WRITES NOTHING. It reads `data/venues.json` and the pinned schedules, calls
two Open-Meteo endpoints, and prints. No file under `data/`, `outputs/` or `artifacts/` is
created or modified.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
import sys
import urllib.request
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

# The FORECAST host. Deliberately not the archive host, and deliberately not in either
# weather module -- see the module docstring.
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"

REQUEST_TIMEOUT_SECONDS = 60
USER_AGENT = "nfl-predict/33.1-01 historical venue derivation"

# The seasons the pinned feed covers, and the span R1 is stated over.
FIRST_SEASON = 2002
LAST_SEASON = 2025

# `venue_high_altitude` fires at elevation_ft >= 3000 (features/contextual.py:643). The
# boundary is printed beside every derived elevation so the ratifier can see at a glance
# that none of the 22 is near it (Ruling A).
HIGH_ALTITUDE_BOUNDARY_FT = 3000

# `venue_large_stadium` fires at capacity >= 75000 (features/contextual.py:653). Unlike a
# coordinate, a capacity has no tolerance: it crosses a BAND edge.
LARGE_STADIUM_BOUNDARY = 75000

# The grass side of the grass/synthetic split `surface_mismatch` reads. It USED to be a
# local copy of the two spellings `_is_grass_surface` then tested (Bermuda Grass and
# Kentucky Bluegrass), and that rule misread every natural or hybrid pitch spelled any
# other way as synthetic. Plan 33.2-10 step 3b made the classification complete in
# features.contextual.SURFACE_CLASS_BY_SPELLING; this printout now reads the grass set
# from there, so it cannot describe a rule the feature no longer applies. Printed beside
# every surface cell so the ratifier can see which side each record lands on.
sys.path.insert(0, str(REPO_ROOT))
from features.contextual import GRASS_SURFACES as GRASS_SURFACE_TOKENS

METRES_PER_FOOT = 0.3048

# The 22 ids RESEARCH section 3 re-derived, in descending game order. The tuple is
# DECLARED here and RE-DERIVED at run time from the pinned schedules; a mismatch between
# the two is a hard failure naming both sides, never a silent use of whichever is shorter.
HISTORICAL_STADIUM_IDS: tuple[str, ...] = (
    "OAK00",
    "NYC00",
    "SDG00",
    "ATL00",
    "STL00",
    "SFO00",
    "MIN00",
    "DAL99",
    "IND99",
    "PHO99",
    "LAX99",
    "LAX97",
    "MIN98",
    "PHI99",
    "CHI99",
    "BUF01",
    "BRG00",
    "SAN00",
    "LON01",
    "GER00",
    "FRA00",
    "SAO00",
)

EXTERNALLY_SOURCED_FIELDS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "surface",
    "capacity",
)

DERIVED_FIELDS: tuple[str, ...] = (
    "roof_type",
    "timezone",
    "elevation_ft",
    "climate_zone",
)

# The eight non-identity fields the provenance record covers. 22 x 8 = 176 rows.
NON_IDENTITY_FIELDS: tuple[str, ...] = EXTERNALLY_SOURCED_FIELDS + DERIVED_FIELDS

# The field order the Task-3 constant is stated in, and the order the table prints in.
TABLE_FIELDS: tuple[str, ...] = (
    "stadium_id",
    "venue_id",
    "venue_name",
    "city",
    "state",
    "country",
    "latitude",
    "longitude",
    "elevation_ft",
    "roof_type",
    "surface",
    "capacity",
    "climate_zone",
    "timezone",
)

# Values that would mean a cell was DEFAULTED rather than entered, copied in discipline
# from tests/unit/test_venues_json_international.py:71.
DEFAULT_SENTINELS: frozenset[object] = frozenset({None, "", 0, "unknown"})

# `climate_zone` is assigned by a STATED RULE rather than researched: take the zone of the
# existing data/venues.json record in the same metropolitan area. Five of the 22 have no
# same-metro sibling and take an explicit operator entry from the same closed vocabulary
# the existing 38 already use.
CLIMATE_ZONE_SIBLINGS: dict[str, str] = {
    "OAK00": "SFO01",
    "SFO00": "SFO01",
    "SDG00": "LAX01",
    "LAX99": "LAX01",
    "LAX97": "LAX01",
    "NYC00": "NYC01",
    "ATL00": "ATL97",
    "MIN00": "MIN01",
    "MIN98": "MIN01",
    "DAL99": "DAL00",
    "IND99": "IND00",
    "PHO99": "PHO00",
    "PHI99": "PHI00",
    "CHI99": "CHI98",
    "LON01": "LON00",
    "GER00": "MUN01",
    "SAO00": "RIO00",
}

# (value, the reason the operator entered it). Each names the existing record whose zone
# it is being kept consistent with, so the entry is checkable rather than asserted.
CLIMATE_ZONE_OPERATOR_ENTRIES: dict[str, tuple[str, str]] = {
    "STL00": (
        "humid_continental",
        "operator entry, no same-metro sibling; consistent with the repository's other "
        "Midwest records KAN00, IND00 and CIN00, all humid_continental",
    ),
    "BUF01": (
        "humid_continental",
        "operator entry, no same-metro sibling; Toronto sits across the lake from BUF00 "
        "Orchard Park, which is humid_continental",
    ),
    "BRG00": (
        "humid_subtropical",
        "operator entry, no same-metro sibling; Baton Rouge follows the nearest existing "
        "Gulf record NOR00 New Orleans, which is humid_subtropical",
    ),
    "SAN00": (
        "humid_subtropical",
        "operator entry, no same-metro sibling; San Antonio follows the nearest existing "
        "Texas Gulf record HOU00 Houston, which is humid_subtropical",
    ),
    "FRA00": (
        "oceanic",
        "operator entry, no same-metro sibling; Frankfurt follows the other existing "
        "German record MUN01 Munich, which is oceanic",
    ),
}

# city / state / country. These are IDENTITY cells, not the eight the provenance record
# covers, and they are read off the same cited documents the coordinates come from. `state`
# is the EMPTY STRING for the five non-US venues rather than a guessed province -- the same
# boundary 33-06-SUMMARY.md:70 records for the international eight.
LOCATION_FACTS: dict[str, tuple[str, str, str]] = {
    "OAK00": ("Oakland", "CA", "USA"),
    "NYC00": ("East Rutherford", "NJ", "USA"),
    "SDG00": ("San Diego", "CA", "USA"),
    "ATL00": ("Atlanta", "GA", "USA"),
    "STL00": ("St. Louis", "MO", "USA"),
    "SFO00": ("San Francisco", "CA", "USA"),
    "MIN00": ("Minneapolis", "MN", "USA"),
    "DAL99": ("Irving", "TX", "USA"),
    "IND99": ("Indianapolis", "IN", "USA"),
    "PHO99": ("Tempe", "AZ", "USA"),
    "LAX99": ("Los Angeles", "CA", "USA"),
    "LAX97": ("Carson", "CA", "USA"),
    "MIN98": ("Minneapolis", "MN", "USA"),
    "PHI99": ("Philadelphia", "PA", "USA"),
    "CHI99": ("Champaign", "IL", "USA"),
    "BUF01": ("Toronto", "", "Canada"),
    "BRG00": ("Baton Rouge", "LA", "USA"),
    "SAN00": ("San Antonio", "TX", "USA"),
    "LON01": ("London", "", "United Kingdom"),
    "GER00": ("Munich", "", "Germany"),
    "FRA00": ("Frankfurt", "", "Germany"),
    "SAO00": ("Sao Paulo", "", "Brazil"),
}

if len(HISTORICAL_STADIUM_IDS) != len(set(HISTORICAL_STADIUM_IDS)):
    raise AssertionError(
        "HISTORICAL_STADIUM_IDS carries a duplicate id: "
        f"{sorted({i for i in HISTORICAL_STADIUM_IDS if HISTORICAL_STADIUM_IDS.count(i) > 1})!r}"
    )
if len(HISTORICAL_STADIUM_IDS) != 22:
    raise AssertionError(
        f"HISTORICAL_STADIUM_IDS holds {len(HISTORICAL_STADIUM_IDS)} ids, expected the 22 "
        "the pinned 2002-2025 schedules carry that data/venues.json does not."
    )


class VenueDerivationError(RuntimeError):
    """A refusal this command makes rather than producing a record nobody sourced."""


def _http_get_json(url: str) -> dict[str, Any]:
    """GET *url* and parse the response as JSON.

    Args:
        url: A fully-formed absolute URL.

    Returns:
        The decoded JSON body.

    Raises:
        VenueDerivationError: The request failed or the body was not JSON.
    """
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            return json.loads(response.read().decode("utf-8"))
    except (OSError, ValueError) as exc:
        raise VenueDerivationError(f"request to {url} failed: {exc}") from exc


def load_existing_venue_records() -> list[dict[str, Any]]:
    """Every record currently in ``data/venues.json``."""
    return json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]


def existing_stadium_ids() -> set[str]:
    """The ``stadium_id`` values ``data/venues.json`` already carries."""
    return {
        str(record["stadium_id"])
        for record in load_existing_venue_records()
        if record.get("stadium_id")
    }


def load_pinned_schedules() -> Any:
    """The pinned 2002-2025 schedules, as a DataFrame.

    Imported lazily so ``import scripts.derive_historical_venue_facts`` stays cheap for
    the shape checks in this plan's verification block.
    """
    from data.upstream_pin import load_schedules

    return load_schedules(list(range(FIRST_SEASON, LAST_SEASON + 1)))


def rederive_missing_stadium_ids(schedules: Any) -> set[str]:
    """The feed's ``stadium_id`` set minus the ids ``data/venues.json`` already holds."""
    feed_ids = {str(value) for value in schedules["stadium_id"].dropna().unique()}
    return feed_ids - existing_stadium_ids()


def assert_declared_ids_match_the_feed(schedules: Any) -> None:
    """The declared tuple and the re-derived set must agree, in BOTH directions.

    Raises:
        VenueDerivationError: They disagree. Both sides are named; the shorter one is
            never silently preferred.
    """
    derived = rederive_missing_stadium_ids(schedules)
    declared = set(HISTORICAL_STADIUM_IDS)
    if derived == declared:
        return
    raise VenueDerivationError(
        "the declared HISTORICAL_STADIUM_IDS and the ids re-derived from the pinned "
        f"{FIRST_SEASON}-{LAST_SEASON} schedules disagree.\n"
        f"  declared but not in the feed-minus-file set: {sorted(declared - derived)!r}\n"
        f"  in the feed-minus-file set but not declared: {sorted(derived - declared)!r}\n"
        "Re-derive the inventory before tabling anything: a record for a stadium the feed "
        "never names resolves nothing, and a stadium the feed names with no record is the "
        "defect this plan exists to close."
    )


def _carries_revision_pin(source: str) -> bool:
    """True when *source* carries an ``oldid=`` or a bare Wikidata ``Q`` entity id.

    This is what makes a citation checkable a year from now rather than a pointer to
    whatever the page happens to say then. It is a check that a source is PINNED. It is
    not, and cannot be, a check that the source is CORRECT -- only the owner ratification
    asserts that.
    """
    if "oldid=" in source:
        return True
    return any(
        token.startswith("Q") and token[1:].isdigit()
        for token in re.split(r"[\s,()\[\]]+", source)
        if token
    )


def _names_an_existing_stadium_id(source: str) -> str | None:
    """The existing ``stadium_id`` a source string names, or None.

    The successor venue's own record plus an offset is the tempting wrong answer:
    Giants Stadium and MetLife are about 400 m apart in the same parking lot, and a
    geocoder asked for "Giants Stadium" returns MetLife.
    """
    for code in sorted(existing_stadium_ids()):
        if re.search(rf"\b{re.escape(code)}\b", source):
            return code
    return None


def _is_absent(value: object) -> bool:
    """True when a cell was defaulted rather than entered."""
    if isinstance(value, str):
        return value.strip().lower() in {"", "unknown"}
    return value in DEFAULT_SENTINELS


def assert_every_external_field_is_sourced(facts: dict[str, Any]) -> None:
    """Refuse unless every external cell on every one of the 22 carries its own citation.

    Five conditions are reported in ONE pass, each as a list of named
    ``<stadium_id>.<field>`` pairs rather than as a count, because the operator's next
    action is to go and source exactly those cells:

    1. the cell is absent, empty, zero or the string ``unknown``;
    2. the cell carries a value but no ``source``;
    3. the ``source`` carries neither an ``oldid=`` nor a Wikidata ``Q`` entity id;
    4. the ``source`` names a ``stadium_id`` that is already in ``data/venues.json`` --
       the successor venue is not a source for its predecessor's geography;
    5. the ``surface`` or ``capacity`` source is BYTE-IDENTICAL to that venue's
       ``latitude`` or ``longitude`` source. Ruling C2 requires a citation to be pinned
       AND entered PER FIELD; a citation names the statement it was read from, so one
       document legitimately backing three fields still yields three different strings
       ("infobox surface = ..." is not "P625 coordinate location = ..."). An identical
       string is therefore not one document doing three jobs -- it is the per-VENUE
       citation this record exists to replace, pasted into three slots. ``latitude`` and
       ``longitude`` MAY share a string: a coordinate statement genuinely states both.

    Args:
        facts: ``{"<stadium_id>": {"<field>": {"value": ..., "source": "..."}}}``.

    Raises:
        VenueDerivationError: Any of the five conditions holds anywhere.
    """
    unvalued: list[str] = []
    unsourced: list[str] = []
    unpinned: list[str] = []
    successor_sourced: list[str] = []
    coordinate_reuse: list[str] = []

    for stadium_id, field in itertools.product(
        HISTORICAL_STADIUM_IDS, EXTERNALLY_SOURCED_FIELDS
    ):
        pair = f"{stadium_id}.{field}"
        cell = (facts.get(stadium_id) or {}).get(field)
        if not isinstance(cell, dict):
            unvalued.append(pair)
            unsourced.append(pair)
            continue
        if _is_absent(cell.get("value")):
            unvalued.append(pair)
        source = cell.get("source")
        if not isinstance(source, str) or not source.strip():
            unsourced.append(pair)
            continue
        if not _carries_revision_pin(source):
            unpinned.append(pair)
        named = _names_an_existing_stadium_id(source)
        if named is not None:
            successor_sourced.append(f"{pair} -> names {named} in {source!r}")
        if field in {"surface", "capacity"}:
            venue_cells = facts.get(stadium_id) or {}
            for coordinate_field in ("latitude", "longitude"):
                coordinate_cell = venue_cells.get(coordinate_field)
                if (
                    isinstance(coordinate_cell, dict)
                    and coordinate_cell.get("source") == source
                ):
                    coordinate_reuse.append(
                        f"{pair} (reuses {stadium_id}.{coordinate_field})"
                    )
                    break

    if not (unvalued or unsourced or unpinned or successor_sourced or coordinate_reuse):
        return

    lines = [
        "the facts file does not cover every externally-sourced cell. A record cannot be "
        "generated with a field nobody sourced: `surface` feeds the grass/turf mismatch "
        "feature at features/contextual.py:345-356 and `capacity` feeds venue_capacity / "
        "venue_large_stadium at :653, so an unsubstantiated cell silently moves a gold "
        "column on 1,082 games.",
    ]
    if unvalued:
        lines.append(f"  NO VALUE ({len(unvalued)}): {', '.join(sorted(unvalued))}")
    if unsourced:
        lines.append(f"  NO SOURCE ({len(unsourced)}): {', '.join(sorted(unsourced))}")
    if unpinned:
        lines.append(
            f"  SOURCE NOT REVISION-PINNED ({len(unpinned)}): "
            f"{', '.join(sorted(unpinned))} -- each needs an `oldid=` or a Wikidata Q id"
        )
    if successor_sourced:
        lines.append(
            f"  SOURCE IS THE SUCCESSOR VENUE ({len(successor_sourced)}): "
            + "; ".join(sorted(successor_sourced))
            + " -- the successor venue is not a source for its predecessor's geography"
        )
    if coordinate_reuse:
        lines.append(
            f"  SOURCE IS THE COORDINATE CITATION, REUSED ({len(coordinate_reuse)}): "
            + ", ".join(sorted(coordinate_reuse))
            + " -- a coordinate statement substantiates a latitude and a longitude and "
            "says nothing about what the playing surface was or how many seats the "
            "building held. Cite the document that STATES the field."
        )
    raise VenueDerivationError("\n".join(lines))


def find_single_document_venues(facts: dict[str, Any]) -> list[tuple[str, str]]:
    """Venues where ONE document backs all four external fields (Task 2 check 2b).

    This is a REPORT, not a refusal, and the distinction is the plan's: a single article
    legitimately states all four for some buildings, so refusing would be wrong. What the
    ratifier is owed is the list, so check 2b is a spot-check of a named few rather than a
    manual scan of 176 lines.

    Returns:
        ``(stadium_id, document_url)`` pairs, sorted.
    """
    found: list[tuple[str, str]] = []
    for stadium_id in HISTORICAL_STADIUM_IDS:
        cells = facts.get(stadium_id) or {}
        documents = set()
        for field in EXTERNALLY_SOURCED_FIELDS:
            cell = cells.get(field)
            source = cell.get("source", "") if isinstance(cell, dict) else ""
            match = re.search(r"https?://\S+", source)
            documents.add(match.group(0) if match else source)
        if len(documents) == 1:
            found.append((stadium_id, next(iter(documents))))
    return sorted(found)


def derive_roof_types(schedules: Any) -> dict[str, dict[str, Any]]:
    """``roof_type`` for each of the 22, from the pinned feed's own ``roof`` value.

    `roof_type` is a VENUE property, so a multi-valued feed roof for one id would mean
    this derivation is answering the wrong question -- it is a hard failure naming the id
    and the values, never a majority vote.

    Deriving rather than looking this up is deliberate. `scripts/ingest_games.
    _get_venue_roof_type` resolves `stadium_id` FIRST (`:228-232`), so once these records
    exist that step starts WINNING for 1,082 games; a record that simply agrees with its
    feed roof changes nothing in silver, and one that disagrees moves `venue_roof` on the
    next ingest run.

    Returns:
        ``{stadium_id: {"value", "source", "games", "feed_roof"}}``.
    """
    from scripts.ingest_weather import NFLVERSE_ROOF_MAP

    derived: dict[str, dict[str, Any]] = {}
    multi_valued: list[str] = []
    unmapped: list[str] = []

    for stadium_id in HISTORICAL_STADIUM_IDS:
        rows = schedules[schedules["stadium_id"] == stadium_id]
        feed_values = sorted({str(v) for v in rows["roof"].dropna().unique()})
        if len(feed_values) != 1:
            multi_valued.append(f"{stadium_id}: {feed_values!r}")
            continue
        feed_roof = feed_values[0]
        roof_type = NFLVERSE_ROOF_MAP.get(feed_roof.lower().strip())
        if roof_type is None:
            unmapped.append(f"{stadium_id}: {feed_roof!r}")
            continue
        games = len(rows)
        derived[stadium_id] = {
            "value": roof_type,
            "feed_roof": feed_roof,
            "games": games,
            "source": (
                f"pinned schedules, load_schedules({FIRST_SEASON}..{LAST_SEASON}), "
                f"stadium_id={stadium_id}, roof={feed_roof} "
                f"(single-valued, n={games}) -> NFLVERSE_ROOF_MAP -> {roof_type}"
            ),
        }

    if multi_valued:
        raise VenueDerivationError(
            "the pinned feed carries more than one `roof` value for: "
            + "; ".join(multi_valued)
            + ". roof_type is a VENUE property; a multi-valued feed roof means this "
            "derivation is answering the wrong question and the record must be per game."
        )
    if unmapped:
        raise VenueDerivationError(
            "NFLVERSE_ROOF_MAP has no entry for: "
            + "; ".join(unmapped)
            + ". Map it explicitly rather than defaulting -- an unmapped roof that fell "
            "through to `outdoor` would zero the weather skip on a dome."
        )
    return derived


def derive_timezones(
    coordinates: dict[str, tuple[float, float]],
) -> dict[str, dict[str, Any]]:
    """The IANA zone Open-Meteo resolves for each coordinate, one request per venue.

    A response whose ``timezone`` echoes back ``auto`` or ``GMT`` unchanged is a hard
    failure rather than an accepted answer: `scripts/ingest_weather.
    select_forecast_hour_for_kickoff:269` hard-fails by name on a venue with no IANA zone,
    which is what makes this field load-bearing rather than decorative.

    Returns:
        ``{stadium_id: {"value", "source"}}``.
    """
    rejected: list[str] = []
    derived: dict[str, dict[str, Any]] = {}

    for stadium_id in HISTORICAL_STADIUM_IDS:
        latitude, longitude = coordinates[stadium_id]
        url = (
            f"{OPEN_METEO_FORECAST_URL}?latitude={latitude}&longitude={longitude}"
            "&timezone=auto&forecast_days=1&hourly=temperature_2m"
        )
        payload = _http_get_json(url)
        zone = str(payload.get("timezone", ""))
        if zone.lower() in {"", "auto", "gmt"}:
            rejected.append(f"{stadium_id}: {zone!r}")
            continue
        try:
            ZoneInfo(zone)
        except Exception:  # noqa: BLE001 - any failure to resolve is the same refusal
            rejected.append(f"{stadium_id}: {zone!r} does not resolve under ZoneInfo")
            continue
        derived[stadium_id] = {
            "value": zone,
            "source": (
                f'open-meteo timezone=auto @ ({latitude}, {longitude}) -> "{zone}"'
            ),
        }

    if rejected:
        raise VenueDerivationError(
            "Open-Meteo did not resolve an IANA zone for: "
            + "; ".join(rejected)
            + ". An unresolved zone is a refusal, not a default -- the hour resolver "
            "hard-fails by name on a venue with no zone."
        )
    return derived


def derive_elevations_ft(
    coordinates: dict[str, tuple[float, float]],
) -> dict[str, dict[str, Any]]:
    """Elevation in feet for all 22, from ONE batched ``/v1/elevation`` request.

    Ruling A, recorded honestly: Open-Meteo returns a ~90 m DEM TERRAIN value at the
    nearest grid cell, not the playing surface, so the 22 differ IN KIND from the 38
    hand-sourced records. That is acceptable because nothing in the repository reads
    `elevation_ft` more finely than the 3,000 ft `venue_high_altitude` band and none of the
    22 is within 1,500 ft of it -- the printed table shows every margin so the ratifier can
    see that rather than take it on trust.

    Returns:
        ``{stadium_id: {"value", "source", "metres"}}``.
    """
    latitudes = ",".join(str(coordinates[sid][0]) for sid in HISTORICAL_STADIUM_IDS)
    longitudes = ",".join(str(coordinates[sid][1]) for sid in HISTORICAL_STADIUM_IDS)
    payload = _http_get_json(
        f"{OPEN_METEO_ELEVATION_URL}?latitude={latitudes}&longitude={longitudes}"
    )
    metres = payload.get("elevation")
    if not isinstance(metres, list) or len(metres) != len(HISTORICAL_STADIUM_IDS):
        raise VenueDerivationError(
            f"/v1/elevation returned {metres!r} for {len(HISTORICAL_STADIUM_IDS)} "
            "coordinates. A short or absent list cannot be matched back to its venues "
            "positionally without guessing which one is missing."
        )

    derived: dict[str, dict[str, Any]] = {}
    for stadium_id, metre_value in zip(HISTORICAL_STADIUM_IDS, metres, strict=True):
        latitude, longitude = coordinates[stadium_id]
        feet = round(float(metre_value) / METRES_PER_FOOT)
        derived[stadium_id] = {
            "value": feet,
            "metres": float(metre_value),
            "source": (
                f"open-meteo /v1/elevation @ ({latitude}, {longitude}) -> "
                f"{metre_value} m -> {feet} ft"
            ),
        }
    return derived


def derive_climate_zones() -> dict[str, dict[str, Any]]:
    """``climate_zone`` from the same-metro-sibling rule, or an explicit operator entry.

    `climate_zone` is read at `features/contextual.py:646` and feeds `venue_cold_climate` /
    `venue_warm_climate`, so leaving it at a sentinel would silently move a live contextual
    feature on 1,082 games for a reason nobody chose (Ruling B).

    Returns:
        ``{stadium_id: {"value", "source"}}``.
    """
    existing = {
        str(record["stadium_id"]): record
        for record in load_existing_venue_records()
        if record.get("stadium_id")
    }
    derived: dict[str, dict[str, Any]] = {}
    missing_siblings: list[str] = []

    for stadium_id in HISTORICAL_STADIUM_IDS:
        sibling = CLIMATE_ZONE_SIBLINGS.get(stadium_id)
        if sibling is not None:
            record = existing.get(sibling)
            if record is None:
                missing_siblings.append(f"{stadium_id} -> {sibling}")
                continue
            zone = str(record["climate_zone"])
            derived[stadium_id] = {
                "value": zone,
                "source": (
                    f"same-metro sibling rule: data/venues.json stadium_id={sibling} "
                    f"({record['venue_name']}) climate_zone={zone}"
                ),
            }
            continue
        entry = CLIMATE_ZONE_OPERATOR_ENTRIES.get(stadium_id)
        if entry is None:
            missing_siblings.append(f"{stadium_id} -> no sibling and no operator entry")
            continue
        zone, reason = entry
        derived[stadium_id] = {"value": zone, "source": reason}

    if missing_siblings:
        raise VenueDerivationError(
            "climate_zone could not be assigned for: "
            + "; ".join(missing_siblings)
            + ". Every one of the 22 takes either a named same-metro sibling or an "
            "explicit operator entry; neither is a default."
        )
    return derived


def derive_venue_names(schedules: Any) -> dict[str, dict[str, str]]:
    """``venue_name`` and ``venue_id`` from the MOST RECENT feed ``stadium`` string.

    Ruling C: twenty of the 22 ids carry two to five feed `stadium` names each (`OAK00`
    has five), so no single `venue_name` can satisfy `features/contextual.
    _get_venue_id_by_name:400` for all of a venue's games. The name resolver stays LOSSY BY
    DESIGN and stops being load-bearing in Plan 33.1-03 when routing moves to `stadium_id`.
    """
    derived: dict[str, dict[str, str]] = {}
    for stadium_id in HISTORICAL_STADIUM_IDS:
        rows = schedules[schedules["stadium_id"] == stadium_id]
        ordered = rows.sort_values(["season", "week"])
        names = [str(v) for v in ordered["stadium"].dropna()]
        if not names:
            raise VenueDerivationError(
                f"{stadium_id} carries no `stadium` name anywhere in the pinned feed, so "
                "there is nothing to name the record after."
            )
        venue_name = names[-1]
        slug = re.sub(r"[^a-z0-9]+", "_", venue_name.lower()).strip("_")
        derived[stadium_id] = {
            "venue_name": venue_name,
            "venue_id": slug,
            "all_feed_names": ", ".join(sorted(set(names))),
        }
    return derived


def load_facts_file(path: Path) -> dict[str, Any]:
    """The operator-filled ``--facts`` file.

    Shape: ``{"<stadium_id>": {"<field>": {"value": ..., "source": "..."}}}`` -- value and
    source TOGETHER, per field, so a value cannot be entered without the document it came
    from.
    """
    if not path.exists():
        raise VenueDerivationError(
            f"no facts file at {path}. The coordinate, surface and capacity cells are NOT "
            "derivable and this command does not invent them; fill the file in first, one "
            '{"value": ..., "source": ...} object per field.'
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise VenueDerivationError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise VenueDerivationError(
            f"{path} must hold an object keyed by stadium_id, not {type(payload).__name__}."
        )
    return payload


def build_rows(
    facts: dict[str, Any],
    names: dict[str, dict[str, str]],
    roofs: dict[str, dict[str, Any]],
    zones: dict[str, dict[str, Any]],
    elevations: dict[str, dict[str, Any]],
    climates: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """One dict per venue, in ``TABLE_FIELDS`` order."""
    rows: list[dict[str, Any]] = []
    for stadium_id in HISTORICAL_STADIUM_IDS:
        city, state, country = LOCATION_FACTS[stadium_id]
        cells = facts[stadium_id]
        rows.append(
            {
                "stadium_id": stadium_id,
                "venue_id": names[stadium_id]["venue_id"],
                "venue_name": names[stadium_id]["venue_name"],
                "city": city,
                "state": state,
                "country": country,
                "latitude": cells["latitude"]["value"],
                "longitude": cells["longitude"]["value"],
                "elevation_ft": elevations[stadium_id]["value"],
                "roof_type": roofs[stadium_id]["value"],
                "surface": cells["surface"]["value"],
                "capacity": cells["capacity"]["value"],
                "climate_zone": climates[stadium_id]["value"],
                "timezone": zones[stadium_id]["value"],
            }
        )
    return rows


def build_field_sources(
    facts: dict[str, Any],
    roofs: dict[str, dict[str, Any]],
    zones: dict[str, dict[str, Any]],
    elevations: dict[str, dict[str, Any]],
    climates: dict[str, dict[str, Any]],
) -> list[tuple[str, str, str, str]]:
    """``(stadium_id, field, kind, source)`` for all 22 x 8 = 176 pairs."""
    derived_by_field = {
        "roof_type": roofs,
        "timezone": zones,
        "elevation_ft": elevations,
        "climate_zone": climates,
    }
    rows: list[tuple[str, str, str, str]] = []
    for stadium_id in HISTORICAL_STADIUM_IDS:
        for field in EXTERNALLY_SOURCED_FIELDS:
            rows.append(
                (stadium_id, field, "external", facts[stadium_id][field]["source"])
            )
        for field in DERIVED_FIELDS:
            rows.append(
                (
                    stadium_id,
                    field,
                    "derived",
                    derived_by_field[field][stadium_id]["source"],
                )
            )
    return rows


def render_ratification_table(
    rows: list[dict[str, Any]],
    roofs: dict[str, dict[str, Any]],
    elevations: dict[str, dict[str, Any]],
    names: dict[str, dict[str, str]],
) -> str:
    """The 22-row table, plus the two band columns the ratifier needs beside it."""
    widths = {
        field: max(len(field), *(len(str(r[field])) for r in rows))
        for field in TABLE_FIELDS
    }
    header = "  ".join(field.ljust(widths[field]) for field in TABLE_FIELDS)
    lines = [header, "  ".join("-" * widths[field] for field in TABLE_FIELDS)]
    for row in rows:
        lines.append(
            "  ".join(str(row[field]).ljust(widths[field]) for field in TABLE_FIELDS)
        )

    lines.append("")
    lines.append(
        f"BAND MARGINS -- venue_high_altitude fires at elevation_ft >= "
        f"{HIGH_ALTITUDE_BOUNDARY_FT}, venue_large_stadium at capacity >= "
        f"{LARGE_STADIUM_BOUNDARY}, and _is_grass_surface is membership in "
        f"{sorted(GRASS_SURFACE_TOKENS)!r}."
    )
    lines.append(
        "stadium_id  elevation_ft  ft_below_3000  high_altitude  capacity  large_stadium  "
        "surface_class"
    )
    for row in rows:
        elevation = int(row["elevation_ft"])
        capacity = int(row["capacity"])
        lines.append(
            f"{row['stadium_id']:<11} {elevation:>12}  {HIGH_ALTITUDE_BOUNDARY_FT - elevation:>13}  "
            f"{1.0 if elevation >= HIGH_ALTITUDE_BOUNDARY_FT else 0.0:>13}  "
            f"{capacity:>8}  {1.0 if capacity >= LARGE_STADIUM_BOUNDARY else 0.0:>13}  "
            f"{'grass' if row['surface'] in GRASS_SURFACE_TOKENS else 'synthetic'}"
        )

    lines.append("")
    lines.append(
        "FEED ROOF, AND THE NAMES THE RESOLVER WILL MISS (Ruling C -- the name path stays "
        "lossy by design and stops being load-bearing in Plan 33.1-03)."
    )
    lines.append("stadium_id  feed_roof  roof_type    games  every feed `stadium` name")
    total_games = 0
    for row in rows:
        stadium_id = row["stadium_id"]
        roof = roofs[stadium_id]
        total_games += int(roof["games"])
        lines.append(
            f"{stadium_id:<11} {roof['feed_roof']:<10} {roof['value']:<12} "
            f"{roof['games']:>5}  {names[stadium_id]['all_feed_names']}"
        )
    lines.append(f"{'TOTAL':<11} {'':<10} {'':<12} {total_games:>5}")

    lines.append("")
    lines.append("DERIVED ELEVATION, IN METRES AS THE PROVIDER RETURNED IT (Ruling A).")
    for row in rows:
        stadium_id = row["stadium_id"]
        lines.append(
            f"{stadium_id:<11} {elevations[stadium_id]['metres']:>8} m -> "
            f"{elevations[stadium_id]['value']:>5} ft"
        )
    return "\n".join(lines)


def render_field_sources_block(rows: list[tuple[str, str, str, str]]) -> str:
    """ONE LINE PER ``(stadium_id, field)`` pair -- 176 of them, never a per-venue block.

    A per-VENUE sources block is not acceptable output (Ruling C2): the ratifier has to be
    able to see which document backs `SDG00.capacity` specifically, and a venue-level line
    cannot say that.
    """
    return "\n".join(
        f"{stadium_id}.{field} [{kind}] -> {source}"
        for stadium_id, field, kind, source in rows
    )


def render_header(facts_path: Path) -> str:
    """What the ratifier is being asked to ratify, and at what precision."""
    return "\n".join(
        [
            "=" * 88,
            "THE 22 HISTORICAL VENUE RECORDS, TABLED FOR OWNER RATIFICATION",
            "Phase 33.1, Plan 33.1-01 Task 1 (R1). This command WRITES NOTHING.",
            f"facts file: {facts_path}",
            "=" * 88,
            "",
            "WHAT PRECISION IS BEING RATIFIED.",
            "  ERA5 snaps to a roughly 9-11 km grid cell, so coordinate precision finer",
            "  than about 0.01 degrees CANNOT change a fetched weather value. What is being",
            "  ratified is the right CITY-SCALE location. The distinguishing cases are",
            "  Oakland against Las Vegas at 600 km, St. Louis against Los Angeles at",
            "  2,500 km, and San Diego against Los Angeles at 180 km.",
            "",
            "  `surface` and `capacity` have NO such tolerance. capacity >= 75000 is a hard",
            "  band edge (features/contextual.py:653) and the surface comparison is a string",
            "  membership test (:345-356, :369), so a wrong cell there FLIPS a feature",
            "  rather than moving it.",
            "",
            "PROVENANCE IS PER FIELD (Ruling C2). The sources block below carries one line",
            "  per (stadium_id, field) pair over all eight non-identity fields -- 176 lines.",
            "  A coordinate citation does not substantiate a surface or a capacity.",
            "",
            "`home_teams` IS NOT ON THIS TABLE. It is `[]` for all 22 by Ruling B, because",
            "  `_get_venue_record` (scripts/ingest_weather.py:621) returns the FIRST record",
            "  whose `home_teams` contains the team -- a historical record claiming `LV`",
            "  would shadow VEG00 for the live 2026 season. Task 3 asserts it directly.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Table the 22 records and their per-field provenance. Writes nothing."""
    parser = argparse.ArgumentParser(
        description=(
            "Derive and table the 22 historical venue records for owner ratification "
            "(Phase 33.1, Plan 33.1-01 Task 1). Writes no file."
        )
    )
    parser.add_argument(
        "--facts",
        required=True,
        type=Path,
        help=(
            "JSON file holding the externally-sourced cells, shaped "
            '{"<stadium_id>": {"<field>": {"value": ..., "source": "..."}}}'
        ),
    )
    args = parser.parse_args(argv)

    try:
        facts = load_facts_file(args.facts)
        # The coverage gate runs BEFORE anything is printed or derived: a record must never
        # be tabled with a field nobody sourced.
        assert_every_external_field_is_sourced(facts)

        schedules = load_pinned_schedules()
        assert_declared_ids_match_the_feed(schedules)

        names = derive_venue_names(schedules)
        roofs = derive_roof_types(schedules)
        coordinates = {
            stadium_id: (
                float(facts[stadium_id]["latitude"]["value"]),
                float(facts[stadium_id]["longitude"]["value"]),
            )
            for stadium_id in HISTORICAL_STADIUM_IDS
        }
        zones = derive_timezones(coordinates)
        elevations = derive_elevations_ft(coordinates)
        climates = derive_climate_zones()
    except VenueDerivationError as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1

    rows = build_rows(facts, names, roofs, zones, elevations, climates)
    sources = build_field_sources(facts, roofs, zones, elevations, climates)

    print(render_header(args.facts))
    print(render_ratification_table(rows, roofs, elevations, names))
    print("")
    print("=" * 88)
    print(
        f"PER-FIELD PROVENANCE -- {len(sources)} lines, one per (stadium_id, field) pair "
        f"over {len(HISTORICAL_STADIUM_IDS)} records x {len(NON_IDENTITY_FIELDS)} fields."
    )
    print("=" * 88)
    print(render_field_sources_block(sources))

    single_document = find_single_document_venues(facts)
    print("")
    print("=" * 88)
    print(
        "TASK 2 CHECK 2b -- venues where ONE document backs all four external fields. A "
        "single article legitimately states all four for some buildings, so this is a "
        "SPOT-CHECK LIST rather than a refusal; what it rules out is a per-field record "
        "that has quietly degenerated back into the per-venue one."
    )
    print("=" * 88)
    if not single_document:
        print(
            "None. Every one of the 22 draws its coordinates and its surface/capacity from "
            "separately-cited statements."
        )
    else:
        for stadium_id, document in single_document:
            print(f"{stadium_id} -> {document}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
