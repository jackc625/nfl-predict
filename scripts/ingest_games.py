"""Game data ingestion using nflreadpy."""

import argparse
import functools
import json
import sys
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pandas as pd

from conf.settings import get_settings
from data import upstream_pin
from data.quality_gates import validate_bronze_to_silver
from data.schemas import GameSchema
from data.storage import save_bronze_snapshot, upsert_silver
from data.upstream_pin import UpstreamPinError
from features.schedule_moves import load_schedule_moves
from utils import (
    DataIngestionError,
    get_current_nfl_week,
    get_logger,
    log_data_operation,
)
from utils.exceptions import DataValidationError
from utils.game_id_utils import is_valid_game_id
from utils.team_data import normalize_team_abbreviation

logger = get_logger(__name__)

# nflreadpy roof type values -> project VenueRoof enum values
_NFLVERSE_ROOF_MAP = {
    "dome": "indoor",
    "closed": "indoor",
    "outdoors": "outdoor",
    "open": "retractable",
}


class IdentityColumnError(ValueError):
    """A row's identity fact cannot be derived from the columns the feed supplies.

    Raised rather than defaulted, and that distinction is the whole point of this
    plan. ``season_type`` and ``neutral_site`` were previously read as
    ``row.get(key, default)`` against a schedule frame that carries NEITHER key, so
    the default fired on all 6,499 silver rows and every WC/DIV/CON/SB game in
    project history read Regular while all eight of 2026's international games read
    False. A default that never loses is not a default; it is a constant nobody
    chose. Defaulting again on an absent column would rebuild that defect one layer
    up.
    """


# The feed's five game_type values, and the two-value partition D33-17 coarsens them
# to. TOTAL over the five: an unknown sixth RAISES by name rather than falling into
# either bucket.
#
# THIS IS A COARSENING, NOT A DUPLICATE OF game_type. Two values against five is
# deliberate -- the season-close readout partitions regular-versus-post.
_GAME_TYPE_TO_SEASON_TYPE = {
    "REG": "Regular",
    "WC": "Postseason",
    "DIV": "Postseason",
    "CON": "Postseason",
    "SB": "Postseason",
}

# The feed's location value marking a neutral-site game. Compared EXACTLY: the feed
# writes Neutral and Home, and treating a lowercased spelling as a match would be a
# normalization nobody asked for on a column that decides which stadium a game is
# played at.
_NEUTRAL_LOCATION = "Neutral"


def _raise_identity(column: str, message: str) -> str:
    """Raise an IdentityColumnError naming the column it could not derive from."""
    raise IdentityColumnError(f"[{column}] {message}")


def _derive_season_type(row: Any) -> str:
    """Derive season_type from the feed's game_type (D33-17).

    Args:
        row: A mapping-like schedule row carrying game_type.

    Returns:
        Regular for REG; Postseason for WC, DIV, CON and SB.

    Raises:
        IdentityColumnError: game_type is absent, null, or an unknown sixth value.
            The two columns can never disagree because this is a FUNCTION of
            game_type -- there is no second source for the answer to drift from.
    """
    game_type = row.get("game_type")
    if game_type is None or (isinstance(game_type, float) and pd.isna(game_type)):
        return _raise_identity(
            "game_type",
            "season_type cannot be derived: the row carries no game_type. It is NOT "
            "defaulted -- that default is exactly how every postseason game in this "
            "project came to be labelled a regular-season one.",
        )

    game_type = str(game_type)
    season_type = _GAME_TYPE_TO_SEASON_TYPE.get(game_type)
    if season_type is None:
        return _raise_identity(
            "game_type",
            f"season_type cannot be derived from game_type {game_type!r}: it is not "
            f"one of the five known values {sorted(_GAME_TYPE_TO_SEASON_TYPE)}. Add "
            "it to _GAME_TYPE_TO_SEASON_TYPE in scripts/ingest_games.py with a "
            "deliberate Regular/Postseason assignment; do not let it default.",
        )
    return season_type


def _derive_neutral_site(row: Any) -> bool:
    """Derive neutral_site from the feed's location column.

    Args:
        row: A mapping-like schedule row carrying location.

    Returns:
        True iff location is exactly the string Neutral.

    Raises:
        IdentityColumnError: location is absent or null.
    """
    location = row.get("location")
    if location is None or (isinstance(location, float) and pd.isna(location)):
        return _raise_identity(
            "location",
            "neutral_site cannot be derived: the row carries no location. It is NOT "
            "defaulted to False -- an ABSENT COLUMN is precisely how all 6,499 "
            "silver rows came to read False, including the 91 historical "
            "neutral-site games and all eight of 2026's international ones.",
        )
    return str(location) == _NEUTRAL_LOCATION


# ---------------------------------------------------------------------------
# GAME-SPECIFIC VENUE OVERRIDES (Plan 33.2-09, SPEC R8 venue half, T-33.2-09-03).
#
# The feed records all seven 2025 international games at the US home team's own
# stadium: the Sao Paulo opener at SoFi (indoor, so it would get no weather at all),
# the Dublin game at Pittsburgh, three London games at Cleveland, New York and
# Jacksonville, Berlin at Indianapolis and Madrid at Miami. `stadium_id` is copied from
# the feed verbatim, so correcting only the store would be undone by the next ingest.
#
# The correction lives in ONE committed, cited record, and ONE function resolves it.
# This ingest calls that function for every row, and the one-shot store repair
# (scripts/repair_international_venues.py) calls the same function -- so a re-ingest of
# 2025 writes exactly the rows the repair wrote.
# ---------------------------------------------------------------------------

VENUE_OVERRIDE_RECORD_PATH = (
    Path(__file__).resolve().parent.parent
    / "config"
    / "international_venue_corrections.toml"
)

_VENUES_JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "venues.json"

# The fields every [[correction]] entry must carry, all non-empty strings.
_VENUE_OVERRIDE_FIELDS: tuple[str, ...] = (
    "game_id",
    "old_stadium_id",
    "new_stadium_id",
    "venue_name",
    "city",
    "country",
    "source_url",
    "source_date",
)


class VenueOverrideRecordError(ValueError):
    """The venue correction record is malformed, and nothing is loaded from it.

    Raised for an entry with no source, a duplicated game, a target stadium that
    ``data/venues.json`` does not carry, or a name/city/country that disagrees with the
    venue record it points at. A correction with no source is not a correction.
    """


class VenueOverrideDriftError(ValueError):
    """A recorded game arrived from the feed with a stadium_id nobody has looked at.

    The record names the wrong value the feed carries (``old_stadium_id``) and the
    correction (``new_stadium_id``). Any THIRD value means the feed changed underneath
    the record, and silently overriding a value nobody has examined is the same class
    of defect the override exists to remove. It is re-raised past the broad skip handler
    in ``transform_schedule_data``, on the same reasoning as ``IdentityColumnError``.
    """


@dataclass(frozen=True)
class VenueOverride:
    """One cited correction: the game, the feed's wrong venue, and the real one."""

    game_id: str
    old_stadium_id: str
    new_stadium_id: str
    venue_name: str
    city: str
    country: str
    source_url: str
    source_date: str


def _venue_records_by_stadium_id() -> dict[str, dict[str, Any]]:
    with open(_VENUES_JSON_PATH, encoding="utf-8") as f:
        venues = json.load(f)["venues"]
    return {str(venue["stadium_id"]): venue for venue in venues}


def _parse_venue_override(
    entry: dict[str, Any], venues: dict[str, dict[str, Any]], path: Path
) -> VenueOverride:
    """One [[correction]] entry, checked field by field against data/venues.json."""
    label = entry.get("game_id") or "<entry with no game_id>"
    for field in _VENUE_OVERRIDE_FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise VenueOverrideRecordError(
                f"{path}: correction {label!r} has no {field} (got {value!r}). Every "
                "correction names its game, both venues and a source; one without is "
                "not written."
            )
    override = VenueOverride(
        **{field: entry[field] for field in _VENUE_OVERRIDE_FIELDS}
    )
    target = venues.get(override.new_stadium_id)
    if target is None:
        raise VenueOverrideRecordError(
            f"{path}: correction {label!r} points at new_stadium_id "
            f"{override.new_stadium_id!r}, which data/venues.json does not carry. Add "
            "the venue record first; a correction cannot point at nothing."
        )
    if override.old_stadium_id == override.new_stadium_id:
        raise VenueOverrideRecordError(
            f"{path}: correction {label!r} moves {override.old_stadium_id!r} to itself."
        )
    for field, venue_field in (
        ("venue_name", "venue_name"),
        ("city", "city"),
        ("country", "country"),
    ):
        recorded = getattr(override, field)
        if recorded != target.get(venue_field):
            raise VenueOverrideRecordError(
                f"{path}: correction {label!r} says {field} {recorded!r} but "
                f"data/venues.json says {target.get(venue_field)!r} for "
                f"{override.new_stadium_id}. Two answers for one venue fact; make them "
                "agree."
            )
    return override


@functools.cache
def _load_venue_overrides_from(path_text: str) -> Mapping[str, VenueOverride]:
    path = Path(path_text)
    record = tomllib.loads(path.read_text(encoding="utf-8"))
    venues = _venue_records_by_stadium_id()
    overrides: dict[str, VenueOverride] = {}
    for entry in record.get("correction", []):
        override = _parse_venue_override(entry, venues, path)
        if override.game_id in overrides:
            raise VenueOverrideRecordError(
                f"{path}: game {override.game_id!r} has more than one correction. The "
                "resolver would have to pick one, silently; refusing instead."
            )
        overrides[override.game_id] = override
    return MappingProxyType(overrides)


def load_venue_overrides(
    path: Path | str = VENUE_OVERRIDE_RECORD_PATH,
) -> Mapping[str, VenueOverride]:
    """The committed venue corrections, parsed ONCE per process per path.

    Args:
        path: The correction record. Defaults to
            ``config/international_venue_corrections.toml``.

    Returns:
        An immutable mapping ``game_id -> VenueOverride``.

    Raises:
        VenueOverrideRecordError: an entry has an empty ``source_url`` (or any other
            field), a ``game_id`` appears twice, the ``new_stadium_id`` is absent from
            ``data/venues.json``, or its name/city/country disagree with that record.
        FileNotFoundError: the record is absent. Not defaulted to "no overrides": a
            missing record would silently restore all seven defects on the next ingest.
    """
    return _load_venue_overrides_from(str(Path(path).resolve()))


RoofResolver = Callable[..., str]


def resolve_venue_override(
    game_id: str,
    upstream_stadium_id: Any,
    upstream_stadium: Any,
    upstream_roof: Any,
    *,
    roof_resolver: RoofResolver,
    overrides: Mapping[str, VenueOverride] | None = None,
) -> tuple[Any, Any, str]:
    """The ONE decision behind a game's ``(stadium_id, venue, venue_roof)``.

    Called by ``transform_schedule_data`` for every row and by the one-shot store repair
    for the recorded games, so the store and the next ingest cannot disagree.

    Args:
        game_id: The project game id (``2025_W04_MIN@PIT``).
        upstream_stadium_id: The feed's ``stadium_id`` (may be None or NaN).
        upstream_stadium: The feed's ``stadium`` name, or None when the column is absent.
        upstream_roof: The feed's ``roof`` value.
        roof_resolver: ``GameDataIngester._get_venue_roof_type`` -- the existing roof
            resolver, never a second roof table.
        overrides: The loaded corrections; ``load_venue_overrides()`` when None.

    Returns:
        For an UNRECORDED game, the feed-derived values exactly as the ingest always
        wrote them. For a RECORDED game, the corrected id, the corrected venue record's
        ``venue_name``, and the roof the corrected id resolves to.

    Raises:
        VenueOverrideDriftError: a recorded game's feed id is neither the recorded wrong
            value nor the recorded correction.
    """
    if overrides is None:
        overrides = load_venue_overrides()
    override = overrides.get(game_id)
    if override is None:
        venue = upstream_stadium if upstream_stadium is not None else "Unknown Stadium"
        roof = roof_resolver(
            upstream_stadium if upstream_stadium is not None else "",
            upstream_roof,
            stadium_id=upstream_stadium_id,
        )
        return upstream_stadium_id, venue, roof

    if upstream_stadium_id not in (override.old_stadium_id, override.new_stadium_id):
        raise VenueOverrideDriftError(
            f"game {game_id!r}: the feed carries stadium_id {upstream_stadium_id!r}, "
            f"but config/international_venue_corrections.toml records the feed's value "
            f"as {override.old_stadium_id!r} and the correction as "
            f"{override.new_stadium_id!r}. The feed moved underneath the record; look "
            "at the new value and update the record rather than override it blind."
        )
    roof = roof_resolver(
        override.venue_name, upstream_roof, stadium_id=override.new_stadium_id
    )
    return override.new_stadium_id, override.venue_name, roof


# ---------------------------------------------------------------------------
# GAME-SPECIFIC KICKOFF-HOUR CORRECTIONS (Plan 33.2-12, p332_ extra step 3c).
#
# The feed carries `gametime` "09:00" for every 2002-2005 Monday and Thursday NIGHT game
# (68 games). The ET DATE is right; the clock is a 12-hour AM/PM error: each game's archived
# Pro-Football-Reference box score, and its official NFL Gamebook where one is archived,
# records a start between 9:00 and 9:30 PM Eastern. NFL.com's own game pages carry the
# same wrong "09:00", so the defect is upstream of nflverse and a re-ingest would restore
# it. The date is unaffected, so no lock moves; the HOUR feeds rest-day counts and the
# day-before forecast selection.
#
# Same two-consumer shape as the venue overrides above: ONE cited record
# (config/kickoff_hour_corrections.toml), ONE resolver. This ingest calls it for every row
# and the one-shot store repair (scripts/repair_kickoff_hours.py) calls the same function,
# so a re-ingest of 2002-2005 writes exactly the kickoffs the repair wrote.
# ---------------------------------------------------------------------------

KICKOFF_HOUR_CORRECTION_RECORD_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "kickoff_hour_corrections.toml"
)

# The fields every [[correction]] entry must carry, all non-empty strings.
_KICKOFF_CORRECTION_FIELDS: tuple[str, ...] = (
    "game_id",
    "gameday",
    "feed_gametime",
    "corrected_gametime",
    "source_url",
    "source_start_time",
)


class KickoffCorrectionRecordError(ValueError):
    """The kickoff correction record is malformed, and nothing is loaded from it.

    Raised for an entry with a missing field (a correction with no source is not a
    correction), a clock that is not ``HH:MM``, a duplicated game, a correction that does
    not move the clock, or a cited start time that does not fall in the corrected hour.
    """


class KickoffCorrectionDriftError(ValueError):
    """A recorded game arrived from the feed with a date or clock nobody has looked at.

    The record names the wrong clock the feed carries (``feed_gametime``) and the
    correction (``corrected_gametime``) for one ``gameday``. Any other value means the feed
    changed underneath the record; re-raised past the broad skip handler in
    ``transform_schedule_data`` on the same reasoning as ``VenueOverrideDriftError``.
    """


@dataclass(frozen=True)
class KickoffHourCorrection:
    """One cited correction: the game, its date, the feed's wrong clock and the real one."""

    game_id: str
    gameday: str
    feed_gametime: str
    corrected_gametime: str
    source_url: str
    source_start_time: str


def _clock_minutes(clock: str, *, label: str, path: Path) -> int:
    """Minutes past midnight of an ``HH:MM`` (24-hour) clock, or refuse by name."""
    parts = clock.split(":")
    if len(parts) != 2 or not all(part.isdigit() and len(part) == 2 for part in parts):
        raise KickoffCorrectionRecordError(
            f"{path}: {label} is {clock!r}, not a 24-hour HH:MM clock."
        )
    hours, minutes = int(parts[0]), int(parts[1])
    if hours > 23 or minutes > 59:
        raise KickoffCorrectionRecordError(f"{path}: {label} {clock!r} is not a clock.")
    return hours * 60 + minutes


def _cited_start_minutes(text: str, *, label: str, path: Path) -> int:
    """Minutes past midnight ET of a cited start such as ``9:08 PM ET``, or refuse."""
    words = text.split()
    if len(words) != 3 or words[1] not in ("AM", "PM") or words[2] != "ET":
        raise KickoffCorrectionRecordError(
            f"{path}: {label} is {text!r}; write the cited start as 'H:MM AM|PM ET'."
        )
    hour_text, _, minute_text = words[0].partition(":")
    if not (hour_text.isdigit() and minute_text.isdigit() and len(minute_text) == 2):
        raise KickoffCorrectionRecordError(f"{path}: {label} {text!r} is not a clock.")
    hours = int(hour_text) % 12 + (12 if words[1] == "PM" else 0)
    return hours * 60 + int(minute_text)


def _parse_kickoff_correction(
    entry: dict[str, Any], path: Path
) -> KickoffHourCorrection:
    """One [[correction]] entry, checked field by field."""
    label = entry.get("game_id") or "<entry with no game_id>"
    for field in _KICKOFF_CORRECTION_FIELDS:
        value = entry.get(field)
        if not isinstance(value, str) or not value.strip():
            raise KickoffCorrectionRecordError(
                f"{path}: correction {label!r} has no {field} (got {value!r}). Every "
                "correction names its game, both clocks and a source; one without is "
                "not written."
            )
    correction = KickoffHourCorrection(
        **{field: entry[field] for field in _KICKOFF_CORRECTION_FIELDS}
    )
    feed = _clock_minutes(
        correction.feed_gametime, label=f"{label} feed_gametime", path=path
    )
    fixed = _clock_minutes(
        correction.corrected_gametime, label=f"{label} corrected_gametime", path=path
    )
    if feed == fixed:
        raise KickoffCorrectionRecordError(
            f"{path}: correction {label!r} moves {correction.feed_gametime!r} to itself."
        )
    cited = _cited_start_minutes(
        correction.source_start_time, label=f"{label} source_start_time", path=path
    )
    # The corrected clock is the SCHEDULED start (the convention of every other silver
    # kickoff); the cited source records the ACTUAL start, which follows it by minutes.
    # A cited start outside [corrected, corrected + 30 min) contradicts the correction.
    if not fixed <= cited < fixed + 30:
        raise KickoffCorrectionRecordError(
            f"{path}: correction {label!r} sets {correction.corrected_gametime} ET but "
            f"its source records a start of {correction.source_start_time}. The source "
            "must support the corrected hour; refusing a correction it contradicts."
        )
    return correction


@functools.cache
def _load_kickoff_corrections_from(
    path_text: str,
) -> Mapping[str, KickoffHourCorrection]:
    path = Path(path_text)
    record = tomllib.loads(path.read_text(encoding="utf-8"))
    corrections: dict[str, KickoffHourCorrection] = {}
    for entry in record.get("correction", []):
        correction = _parse_kickoff_correction(entry, path)
        if correction.game_id in corrections:
            raise KickoffCorrectionRecordError(
                f"{path}: game {correction.game_id!r} has more than one correction. The "
                "resolver would have to pick one, silently; refusing instead."
            )
        corrections[correction.game_id] = correction
    return MappingProxyType(corrections)


def load_kickoff_hour_corrections(
    path: Path | str = KICKOFF_HOUR_CORRECTION_RECORD_PATH,
) -> Mapping[str, KickoffHourCorrection]:
    """The committed kickoff-hour corrections, parsed ONCE per process per path.

    Args:
        path: The correction record. Defaults to ``config/kickoff_hour_corrections.toml``.

    Returns:
        An immutable mapping ``game_id -> KickoffHourCorrection``.

    Raises:
        KickoffCorrectionRecordError: an entry is incomplete, a clock is malformed, a game
            appears twice, or a cited start contradicts the corrected hour.
        FileNotFoundError: the record is absent. Not defaulted to "no corrections": a
            missing record would silently restore all 68 wrong hours on the next ingest.
    """
    return _load_kickoff_corrections_from(str(Path(path).resolve()))


def resolve_kickoff_gametime(
    game_id: str,
    gameday: Any,
    upstream_gametime: Any,
    *,
    corrections: Mapping[str, KickoffHourCorrection] | None = None,
) -> Any:
    """The ONE decision behind the clock a game's ``kickoff_et`` is built from.

    Called by ``transform_schedule_data`` for every row and by the one-shot store repair
    for the recorded games, so the store and the next ingest cannot disagree.

    Args:
        game_id: The project game id (``2003_W08_MIA@LAC``).
        gameday: The feed's ``gameday`` (``YYYY-MM-DD``).
        upstream_gametime: The feed's ``gametime`` (``HH:MM`` ET), possibly absent.

    Returns:
        For an UNRECORDED game, ``upstream_gametime`` unchanged. For a RECORDED game, the
        corrected clock.

    Raises:
        KickoffCorrectionDriftError: a recorded game's feed date is not the recorded date,
            or its feed clock is neither the recorded wrong clock nor the correction.
    """
    if corrections is None:
        corrections = load_kickoff_hour_corrections()
    correction = corrections.get(game_id)
    if correction is None:
        return upstream_gametime
    if str(gameday) != correction.gameday or upstream_gametime not in (
        correction.feed_gametime,
        correction.corrected_gametime,
    ):
        raise KickoffCorrectionDriftError(
            f"game {game_id!r}: the feed carries gameday {gameday!r} gametime "
            f"{upstream_gametime!r}, but config/kickoff_hour_corrections.toml records "
            f"{correction.gameday} {correction.feed_gametime!r} corrected to "
            f"{correction.corrected_gametime!r}. The feed moved underneath the record; "
            "look at the new value and update the record rather than override it blind."
        )
    return correction.corrected_gametime


def _load_venue_lookup() -> dict[str, str]:
    """Load venue roof types from data/venues.json.

    Returns:
        Dict mapping venue name (lowercased) -> roof_type string
    """
    venues_path = Path(__file__).resolve().parent.parent / "data" / "venues.json"
    lookup: dict[str, str] = {}
    if venues_path.exists():
        with open(venues_path) as f:
            venues_data = json.load(f)
        for venue in venues_data.get("venues", []):
            name = venue.get("venue_name", "").lower()
            if name:
                lookup[name] = venue.get("roof_type", "outdoor")
    return lookup


def _load_stadium_id_roof_lookup() -> dict[str, str]:
    """Load venue roof types from data/venues.json, keyed on nflverse stadium_id.

    A SECOND lookup rather than a widened first one, because the two answer
    different questions and fail differently: the NAME key is a lossy, lowercased
    string match that silently misses on a spelling variant, while the stadium_id
    key is exact. Consulting the exact one FIRST is what stops this third resolver
    from disagreeing with the other two (NF-05, T-33-29).

    Returns:
        Dict mapping stadium_id -> roof_type string. Matching is EXACT and
        CASE-SENSITIVE: no normalization, no fuzzy match (R11).
    """
    venues_path = Path(__file__).resolve().parent.parent / "data" / "venues.json"
    lookup: dict[str, str] = {}
    if venues_path.exists():
        with open(venues_path) as f:
            venues_data = json.load(f)
        for venue in venues_data.get("venues", []):
            code = venue.get("stadium_id")
            if isinstance(code, str) and code:
                lookup[code] = venue.get("roof_type", "outdoor")
    return lookup


class GameDataIngester:
    """NFL game data ingestion from nflreadpy."""

    def __init__(self):
        """Initialize game data ingester."""
        self.settings = get_settings()
        self._venue_lookup = _load_venue_lookup()
        self._stadium_id_roof_lookup = _load_stadium_id_roof_lookup()
        self._venue_overrides = load_venue_overrides()
        self._kickoff_corrections = load_kickoff_hour_corrections()

    def _get_venue_roof_type(
        self,
        venue: str,
        nflverse_roof: str | None = None,
        stadium_id: str | None = None,
    ) -> str:
        """Determine venue roof type, preferring the venues.json stadium_id key.

        Resolution order, and the order matters (NF-05, T-33-29, D33-16):

        1. ``stadium_id`` against ``data/venues.json`` -- EXACT and case-sensitive.
           This is the same key the other two resolvers use, so consulting it first
           is what makes all three agree.
        2. the lowercased venue NAME against ``data/venues.json``, unchanged.
        3. the nflverse ``roof`` column mapping.
        4. ``'outdoor'``.

        ``data/venues.json`` is AUTHORITATIVE on roof for the eight 2026
        international venues. The feed carries ``roof == 'dome'`` for MEL00, PAR00
        and MUN01 and all three are OPEN-AIR; ``dome`` maps to ``indoor``, and
        ``indoor`` makes the weather ingester skip the API call entirely. Inheriting
        the feed at step 3 for those three would zero the weather on genuinely
        outdoor games with no error raised.

        An UNRECOGNISED ``stadium_id`` falls through to the later steps rather than
        raising: ingestion is not the router, and turning a new code into a hard
        stop here would fail a whole season's ingest over a roof value.

        Args:
            venue: Stadium name string
            nflverse_roof: Optional roof value from nflreadpy (dome/outdoors/closed/open)
            stadium_id: Optional nflverse stadium code from the feed

        Returns:
            One of: 'indoor', 'outdoor', 'retractable'
        """
        # Try the exact stadium_id key first -- the key the other two resolvers use
        if isinstance(stadium_id, str) and stadium_id:
            roof = self._stadium_id_roof_lookup.get(stadium_id)
            if roof:
                return roof

        # Then the (lossy, lowercased) venue NAME key
        if venue:
            roof = self._venue_lookup.get(venue.lower())
            if roof:
                return roof

        # Fall back to nflverse roof column mapping
        if nflverse_roof:
            mapped = _NFLVERSE_ROOF_MAP.get(nflverse_roof.lower().strip())
            if mapped:
                return mapped

        return "outdoor"

    def _create_game_id(self, row: pd.Series) -> str:
        """Create standardized game ID."""
        season = row["season"]
        week = row["week"]
        home_team = normalize_team_abbreviation(row["home_team"])
        away_team = normalize_team_abbreviation(row["away_team"])

        game_id = f"{season}_W{week:02d}_{away_team}@{home_team}"

        # Validate the generated game ID
        if not is_valid_game_id(game_id):
            raise ValueError(f"Generated invalid game ID: {game_id}")

        return game_id

    def _determine_result(self, row: pd.Series) -> int | None:
        """Determine game result from home team perspective."""
        home_score = row.get("home_score")
        away_score = row.get("away_score")

        if pd.isna(home_score) or pd.isna(away_score):
            return None

        if home_score > away_score:
            return 1  # Home win
        if home_score < away_score:
            return -1  # Away win
        return 0  # Tie

    def fetch_schedule_data(
        self, seasons: list[int], weeks: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Fetch schedule data from nflreadpy.

        Args:
            seasons: List of seasons to fetch
            weeks: Optional list of specific weeks

        Returns:
            DataFrame with schedule data
        """
        try:
            logger.info("Fetching schedule data", seasons=seasons, weeks=weeks)

            # Read the PINNED schedule snapshot (data/upstream_pin.py) rather than
            # fetching live, so an ingest is reproducible from a recorded input.
            schedule_df = upstream_pin.load_schedules(seasons)

            if weeks:
                schedule_df = schedule_df[schedule_df["week"].isin(weeks)]

            logger.info(
                "Fetched schedule data", rows=len(schedule_df), seasons=len(seasons)
            )

            return schedule_df

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(
                "Failed to fetch schedule data",
                seasons=seasons,
                weeks=weeks,
                error=str(e),
            )
            raise DataIngestionError(f"Schedule data fetch failed: {e}")

    def fetch_pbp_data(
        self, seasons: list[int], weeks: list[int] | None = None
    ) -> pd.DataFrame:
        """
        Fetch play-by-play data for game results.

        Args:
            seasons: List of seasons to fetch
            weeks: Optional list of specific weeks

        Returns:
            DataFrame with game results
        """
        try:
            logger.info(
                "Fetching play-by-play data for results", seasons=seasons, weeks=weeks
            )

            # Read the PINNED play-by-play snapshot (data/upstream_pin.py).
            pbp_df = upstream_pin.load_pbp(seasons)

            if weeks:
                pbp_df = pbp_df[pbp_df["week"].isin(weeks)]

            # Extract game-level results
            game_results = (
                pbp_df.groupby(["game_id", "season", "week", "home_team", "away_team"])
                .agg({"home_score": "max", "away_score": "max"})
                .reset_index()
            )

            logger.info("Extracted game results from PBP data", rows=len(game_results))

            return game_results

        except (ConnectionError, TimeoutError, ValueError) as e:
            logger.error(
                "Failed to fetch PBP data", seasons=seasons, weeks=weeks, error=str(e)
            )
            raise DataIngestionError(f"PBP data fetch failed: {e}")

    def transform_schedule_data(self, schedule_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw schedule data to our schema format.

        Args:
            schedule_df: Raw schedule DataFrame from nflreadpy

        Returns:
            Transformed DataFrame matching GameSchema
        """
        logger.info("Transforming schedule data", input_rows=len(schedule_df))

        # Create transformed DataFrame
        transformed_data = []

        for _, row in schedule_df.iterrows():
            try:
                # Normalize team names using canonical mapping
                home_team = normalize_team_abbreviation(row["home_team"])
                away_team = normalize_team_abbreviation(row["away_team"])

                game_id = self._create_game_id(row)

                # ONE decision for all three venue columns, AFTER the game id exists
                # and BEFORE anything is derived from the venue: a recorded 2025
                # international game gets the venue it was played at, every other
                # game gets the feed's values exactly as before.
                stadium_id, venue, venue_roof = resolve_venue_override(
                    game_id,
                    row.get("stadium_id"),
                    row.get("stadium"),
                    row.get("roof"),
                    roof_resolver=self._get_venue_roof_type,
                    overrides=self._venue_overrides,
                )

                # ONE decision for the kickoff clock: a recorded 2002-2005 night game
                # whose feed clock is the 12-hour AM/PM error gets its cited evening
                # hour; every other game gets the feed's clock exactly as before.
                gametime = resolve_kickoff_gametime(
                    game_id,
                    row["gameday"],
                    row.get("gametime", "13:00"),
                    corrections=self._kickoff_corrections,
                )

                # Create game record
                game_record = {
                    "game_id": game_id,
                    "season": int(row["season"]),
                    "week": int(row["week"]),
                    "kickoff_et": pd.to_datetime(row["gameday"] + " " + gametime),
                    "home_team": home_team,
                    "away_team": away_team,
                    "venue": venue,
                    "venue_roof": venue_roof,
                    "home_score": row.get("home_score")
                    if pd.notna(row.get("home_score"))
                    else None,
                    "away_score": row.get("away_score")
                    if pd.notna(row.get("away_score"))
                    else None,
                    "result": self._determine_result(row),
                    "game_type": row.get("game_type", "REG"),
                    # DERIVED from columns the feed ACTUALLY HAS (COLD-09/D33-17).
                    # Both of these used to be row.get(<absent key>, <default>), so
                    # the default fired on every row ever ingested.
                    "season_type": _derive_season_type(row),
                    "neutral_site": _derive_neutral_site(row),
                    # Carried through UNCHANGED: no normalization and no casefolding,
                    # because R11's stadium_id matching is exact and case-sensitive.
                    # The only exception is a game with a cited, committed correction
                    # (resolve_venue_override above).
                    "stadium_id": stadium_id,
                }

                transformed_data.append(game_record)

            except IdentityColumnError:
                # AN IDENTITY REFUSAL IS NEVER DEGRADED AWAY, on the same reasoning
                # the UpstreamPinError arm below records. The broad handler under
                # this one SKIPS the row and logs a warning -- which would turn "this
                # row's season_type cannot be derived" into "this game silently does
                # not exist in silver". Losing a game is strictly worse than failing
                # an ingest, and the whole point of raising instead of defaulting is
                # that somebody sees it.
                raise
            except VenueOverrideDriftError:
                # SAME REASONING AS THE IdentityColumnError ARM ABOVE. The broad handler
                # below SKIPS the row with a warning, which would turn "the feed moved
                # underneath a recorded venue correction" into "this 2025 game silently
                # does not exist in silver". A drift is surfaced, never swallowed.
                raise
            except KickoffCorrectionDriftError:
                # Same reasoning again: a feed clock that moved underneath a recorded
                # kickoff correction is surfaced, never turned into a skipped game.
                raise
            except Exception as e:
                logger.warning(
                    "Failed to transform game record",
                    game_data=row.to_dict(),
                    error=str(e),
                )
                continue

        transformed_df = pd.DataFrame(transformed_data)
        logger.info(
            "Transformed schedule data",
            input_rows=len(schedule_df),
            output_rows=len(transformed_df),
        )

        return transformed_df

    def ingest_games(
        self,
        seasons: list[int] | None = None,
        weeks: list[int] | None = None,
        include_results: bool = True,
    ) -> pd.DataFrame:
        """
        Full game data ingestion pipeline.

        Args:
            seasons: Seasons to ingest (default: current season)
            weeks: Specific weeks to ingest (default: all)
            include_results: Whether to fetch game results from PBP data

        Returns:
            Ingested and validated game data
        """
        if seasons is None:
            current_season, _ = get_current_nfl_week()
            seasons = [current_season]

        logger.info(
            "Starting game data ingestion",
            seasons=seasons,
            weeks=weeks,
            include_results=include_results,
        )

        try:
            # Fetch the WHOLE season's schedule: it is what says which stored ids still exist
            # (``_games_dropped_from_schedule``), even when only some weeks are ingested.
            full_schedule = self.fetch_schedule_data(seasons)
            schedule_df = (
                full_schedule[full_schedule["week"].isin(weeks)]
                if weeks
                else full_schedule
            )

            # Transform to our schema
            games_df = self.transform_schedule_data(schedule_df)

            # Fetch results if requested and available
            if include_results:
                try:
                    results_df = self.fetch_pbp_data(seasons, weeks)

                    # Merge results into games data
                    games_df = self._merge_game_results(games_df, results_df)

                except UpstreamPinError:
                    # A PIN REFUSAL IS NEVER DEGRADED AWAY (WR-10). ``data/upstream_pin``'s
                    # module docstring makes a specific, checkable claim: UpstreamPinError
                    # deliberately inherits from Exception and NOT from RuntimeError/ValueError/
                    # ImportError, because every wired call site sits inside an ``except`` naming
                    # those types, and "a pin error caught by one of those handlers would be
                    # converted into an empty play-by-play frame and a silently degraded gold
                    # matrix". Two of the three call sites honour that; this one caught bare
                    # ``Exception``, so it caught the refusal anyway. ``fetch_pbp_data`` catches
                    # only (ConnectionError, TimeoutError, ValueError), so the pin error arrives
                    # here untouched.
                    #
                    # Concretely: after a season roll the manifest covers pbp through 2025 and
                    # nothing beyond. ``upstream_pin.load_pbp`` raises its long explicit refusal,
                    # this handler logged ONE warning line, and silver ``games`` was written with
                    # no home_score/away_score merged -- the ingest step reporting success while
                    # every downstream label built from those scores was wrong or absent.
                    raise
                except Exception as e:
                    logger.warning("Failed to fetch game results", error=str(e))

            # Validate data using hard-fail quality gate
            validated_df = validate_bronze_to_silver(games_df, GameSchema)

            # Add metadata timestamp
            validated_df["created_at"] = datetime.now(UTC)

            # Save to bronze layer (raw, timestamped, append-only)
            save_bronze_snapshot(
                schedule_df,
                "games",
                season=seasons[0],
                week=weeks[0] if weeks else 0,
            )

            # Save to silver layer (latest-wins upsert by game_id). The ids of games the
            # schedule no longer lists -- the old id of a game moved to another week -- are
            # deleted in the same atomic write (33.2 review C1 CR-01).
            dropped = self._games_dropped_from_schedule(full_schedule, seasons)
            upsert_silver(validated_df, "games", remove_keys=dropped)

            log_data_operation(
                operation="ingest",
                table="games",
                rows=len(validated_df),
                seasons=seasons,
                weeks=weeks,
            )

            logger.info(
                "Game data ingestion completed successfully",
                total_games=len(validated_df),
                seasons=seasons,
            )

            return validated_df

        except UpstreamPinError:
            # Re-raised UNWRAPPED (WR-10). This outer handler does not degrade -- it raises -- so
            # it was not the swallow hazard. But wrapping the refusal in DataIngestionError
            # RELABELS it: the operator loses the pin's own long explanation of which seasons the
            # manifest covers, and any caller that handles UpstreamPinError specifically stops
            # seeing it. A pin refusal keeps its own type all the way out.
            logger.error(
                "Game data ingestion refused by the upstream pin", exc_info=True
            )
            raise
        except Exception as e:
            logger.error("Game data ingestion failed", error=str(e))
            raise DataIngestionError(f"Game ingestion failed: {e}")

    def _stored_games(self) -> pd.DataFrame:
        """The silver ``games`` rows already stored, read from the root the upsert writes."""
        path = Path(self.settings.config.data.root_path) / "silver" / "games.parquet"
        if not path.exists():
            return pd.DataFrame(
                columns=["game_id", "season", "home_score", "away_score"]
            )
        return pd.read_parquet(
            path, columns=["game_id", "season", "home_score", "away_score"]
        )

    def _games_dropped_from_schedule(
        self, full_schedule: pd.DataFrame, seasons: list[int]
    ) -> list[str]:
        """Stored ids of *seasons* that the season's schedule no longer lists.

        THE GHOST ROW (33.2 review C1 CR-01). A game's id is built from its WEEK, so when
        nflverse moves a game to another week (2008 Ike, 2017 Irma, the 2020 COVID moves) the
        refreshed schedule carries a NEW id and the latest-wins upsert, which replaces only the
        ids the new frame carries, left the OLD one in silver forever: an unplayed game on its
        old date that the daily run would select, predict and bet. An ingest is therefore
        authoritative for the seasons it ingests, and the ids it no longer lists are removed.

        Two refusals, each by name, because a removal must never destroy evidence:

        * a dropped game that HAS a result is not a ghost -- the schedule changed underneath a
          played game (a renamed franchise, a changed id rule) -- and deleting it would delete
          history;
        * a dropped game with recorded move provenance (``config/schedule_moves.toml``) would
          orphan that evidence; the table is keyed on the POST-move id, so this never fires for
          a correctly recorded move.

        A schedule row whose id cannot be built is ignored here, exactly as
        ``transform_schedule_data`` skips it: it names no stored game.
        """
        stored = self._stored_games()
        if stored.empty:
            return []

        schedule_ids: set[str] = set()
        for _, row in full_schedule.iterrows():
            try:
                schedule_ids.add(self._create_game_id(row))
            except (KeyError, TypeError, ValueError, DataValidationError):
                continue

        stored_ids = stored["game_id"].astype(str)
        dropped = stored.loc[
            stored["season"].isin(seasons) & ~stored_ids.isin(sorted(schedule_ids))
        ]
        if dropped.empty:
            return []
        dropped_ids = sorted(dropped["game_id"].astype(str))

        scored = dropped.loc[
            dropped["home_score"].notna() | dropped["away_score"].notna()
        ]
        if not scored.empty:
            msg = (
                f"the {seasons} schedule no longer lists {len(scored)} stored game(s) that "
                f"HAVE results: {sorted(scored['game_id'].astype(str))}. A played game is not "
                "a ghost of a week move; refusing rather than deleting its history."
            )
            raise DataIngestionError(msg)

        recorded = sorted(set(dropped_ids) & set(load_schedule_moves()))
        if recorded:
            msg = (
                f"the {seasons} schedule no longer lists {recorded}, which carry recorded "
                "move provenance in config/schedule_moves.toml. Refusing to remove them and "
                "orphan that evidence; re-key the recorded move to the game's current id."
            )
            raise DataIngestionError(msg)

        logger.warning(
            "The schedule no longer lists these games (moved to another week); removing "
            "their old ids from silver games",
            seasons=seasons,
            game_ids=dropped_ids,
        )
        return dropped_ids

    def _merge_game_results(
        self, games_df: pd.DataFrame, results_df: pd.DataFrame
    ) -> pd.DataFrame:
        """Merge game results into games DataFrame."""
        logger.info(
            "Merging game results", games=len(games_df), results=len(results_df)
        )

        # Create merge keys
        games_df["merge_key"] = (
            games_df["season"].astype(str)
            + "_"
            + games_df["week"].astype(str)
            + "_"
            + games_df["home_team"]
            + "_"
            + games_df["away_team"]
        )

        results_df["merge_key"] = (
            results_df["season"].astype(str)
            + "_"
            + results_df["week"].astype(str)
            + "_"
            + results_df["home_team"]
            + "_"
            + results_df["away_team"]
        )

        # Merge results
        merged_df = games_df.merge(
            results_df[["merge_key", "home_score", "away_score"]],
            on="merge_key",
            how="left",
            suffixes=("", "_pbp"),
        )

        # Update scores where available
        merged_df["home_score"] = merged_df["home_score_pbp"].combine_first(
            merged_df["home_score"]
        )
        merged_df["away_score"] = merged_df["away_score_pbp"].combine_first(
            merged_df["away_score"]
        )

        # Recalculate results
        for idx, row in merged_df.iterrows():
            merged_df.loc[idx, "result"] = self._determine_result(row)

        # Clean up
        merged_df = merged_df.drop(
            ["merge_key", "home_score_pbp", "away_score_pbp"], axis=1
        )

        logger.info("Game results merged", final_games=len(merged_df))
        return merged_df


def main():
    """CLI entry point for game data ingestion."""
    parser = argparse.ArgumentParser(description="Ingest NFL game data")

    # Add standardized ingestion arguments
    from utils.ingestion_args import (
        add_standard_ingestion_args,
        get_ingestion_summary,
        parse_season_week_args,
    )

    parser = add_standard_ingestion_args(parser)

    # Add games-specific arguments
    parser.add_argument(
        "--no-results", action="store_true", help="Skip fetching game results"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate existing data"
    )

    args = parser.parse_args()

    try:
        # Setup logging
        from utils import setup_logging

        setup_logging()

        ingester = GameDataIngester()

        if args.validate_only:
            # Validate existing data
            from data.storage import load_dataframe

            games_df = load_dataframe("games", layer="silver")
            validated_df = validate_bronze_to_silver(games_df, GameSchema)
            print(f"Validated {len(validated_df)} games")
            return

        # Parse standardized season/week arguments
        seasons, weeks = parse_season_week_args(args)

        # Log what we're about to ingest
        summary = get_ingestion_summary(seasons, weeks)
        logger.info(f"Starting game data ingestion for {summary}")

        # Run ingestion
        games_df = ingester.ingest_games(
            seasons=seasons, weeks=weeks, include_results=not args.no_results
        )

        print(f"Successfully ingested {len(games_df)} games")
        print(f"Seasons: {sorted(games_df['season'].unique())}")
        print(f"Weeks: {sorted(games_df['week'].unique())}")

    except Exception as e:
        logger.error("Game ingestion CLI failed", error=str(e))
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
