"""
Contextual Features Calculator

This module calculates contextual features for NFL games:
- Travel time-zone difference calculations
- Short week detection
- Venue roof type encoding
- Home/away team indicators
- Rest days calculations
- Season-week position features (season progress, late season)
- Surface type mismatch detection
- Divisional game indicator

These features capture situational factors that may impact game outcomes
beyond pure team performance metrics.
"""

import json
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from conf.settings import get_settings
from data.storage import load_dataframe
from features.schedule_moves import FactsAtLock, facts_at_lock
from ratings.elo import is_divisional_game
from utils import DataIngestionError, get_logger
from utils.date_utils import kickoff_wall_clock_et

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PLAYING-SURFACE CLASSES FOR THE MISMATCH FEATURE (FEAT-18), made COMPLETE by Plan 33.2-10
# extra step 3b. `surface_mismatch` is 1.0 when the away team's home surface class differs
# from the game venue's.
#
# THE DEFECT THIS REPLACES. The grass set held two spellings, "Bermuda Grass" and "Kentucky
# Bluegrass", and every OTHER string was classified synthetic by default -- so Wembley,
# Tottenham, Frankfurt, Mexico City, Croke Park and the Olympiastadion ("Grass"), Twickenham
# and Arena Corinthians ("Desso GrassMaster") and the Allianz Arena ("Hybrid Grass") were all
# artificial turf to the feature.
#
# THE RULE NOW. Every distinct `surface` string in data/venues.json is listed here with its
# class, and a string that is not listed RAISES by name (UnknownSurfaceError) instead of
# defaulting. HYBRID GRASS IS GRASS: a Desso GrassMaster or other hybrid pitch is natural
# grass reinforced with a small share of synthetic fibres, and it plays as grass.
# "RealGrass" is NOT grass despite its name: it is an infilled artificial-turf product, and
# Texas Stadium, its only record, never had a natural pitch. "Sport Turf" is classified by
# what the spelling says (a synthetic surface), not by any guess about the venue.
# ---------------------------------------------------------------------------
SURFACE_CLASS_BY_SPELLING: dict[str, str] = {
    "AstroPlay": "synthetic",
    "AstroTurf": "synthetic",
    "Bermuda Grass": "grass",
    "Desso GrassMaster": "grass",
    "FieldTurf": "synthetic",
    "Grass": "grass",
    "Hybrid Grass": "grass",
    "Kentucky Bluegrass": "grass",
    "Matrix Turf": "synthetic",
    "NexTurf": "synthetic",
    "RealGrass": "synthetic",
    "Sport Turf": "synthetic",
}

# The grass spellings, DERIVED from the one mapping above (kept under its historical name
# for the readers that import it).
GRASS_SURFACES: frozenset[str] = frozenset(
    spelling for spelling, cls in SURFACE_CLASS_BY_SPELLING.items() if cls == "grass"
)


class UnknownSurfaceError(LookupError):
    """A playing-surface spelling ``SURFACE_CLASS_BY_SPELLING`` does not classify.

    Deliberately NOT a ``ValueError``: the gold build downgrades a ``ValueError`` from an
    optional source to an EMPTY frame (``scripts.build_features._SOURCE_LOAD_ERRORS``), and
    an unclassified surface must stop the build, not quietly remove the contextual family.
    """


def surface_is_grass(surface: object) -> bool:
    """True for a grass (natural or hybrid) surface, False for a synthetic one.

    Raises:
        UnknownSurfaceError: *surface* is not a spelling ``SURFACE_CLASS_BY_SPELLING``
            lists. It is never defaulted to either class.
    """
    cls = SURFACE_CLASS_BY_SPELLING.get(surface) if isinstance(surface, str) else None
    if cls is None:
        raise UnknownSurfaceError(
            f"surface {surface!r} is not classified in "
            "features.contextual.SURFACE_CLASS_BY_SPELLING. Add it there as 'grass' "
            "(natural or hybrid) or 'synthetic'; it is never defaulted to either class."
        )
    return cls == "grass"


def validate_surface_vocabulary(venues: list[dict[str, Any]] | None = None) -> None:
    """Every venue record's ``surface`` is classified, or refuse naming each offender.

    Raises:
        UnknownSurfaceError: one or more records carry an unclassified spelling.
    """
    offenders = [
        f"{venue.get('stadium_id') or venue.get('venue_id')}: {venue.get('surface')!r}"
        for venue in _load_venue_records(venues)
        if venue.get("surface") not in SURFACE_CLASS_BY_SPELLING
    ]
    if offenders:
        raise UnknownSurfaceError(
            "data/venues.json carries surface spelling(s) that "
            "features.contextual.SURFACE_CLASS_BY_SPELLING does not classify: "
            f"{'; '.join(offenders)}. Classify each spelling there; none is defaulted."
        )


# Look-ahead / letdown spot threshold (D-16), grounded in the RAW silver Elo
# scale (data/silver/elo_game_snapshots.parquet, home_elo_pre/away_elo_pre,
# std ~125). A 100-Elo step is ~0.8 std -- a CHOSEN, in-scale value, NOT a
# tuned one. The screen (Plan 28-07) adjudicates whether it carries signal.
ELO_SPOT_STEP: float = 100.0

# Bye-week rest threshold: a bye gives ~13-14 days between games, so
# off_bye = 1.0 when rest_days >= 13 (the genuinely-new add per D-15).
OFF_BYE_REST_DAYS: float = 13.0

# Exceptions tolerated when reloading the full-season schedule for the
# (weak, optional) spot flags -- a failure must degrade to neutral 0.0
# flags, never break the contextual build.
_SCHEDULE_LOAD_ERRORS = (
    ValueError,
    KeyError,
    TypeError,
    AttributeError,
    OSError,
    DataIngestionError,
)


# ---------------------------------------------------------------------------
# stadium_id-keyed venue routing: ONE RULE, EVERY SEASON, EVERY GAME
# (COLD-09 / R1 / D33-15 / D33.1-06 / T-33.1-15, T-33.1-16, T-33.1-17).
#
# THE DEFECT. All three venue resolvers used to key off the home team, so a game
# resolved to whatever stadium that franchise plays in TODAY. The 2026 Maracana
# game (BAL @ DAL, week 3) resolved to AT&T Stadium in Arlington: Dallas
# coordinates, Dallas timezone, Dallas weather, and a travel distance of ZERO miles
# for a trip to Brazil, with no error raised. History carried the same defect at
# scale: 141 outdoor Oakland Coliseum games took Las Vegas desert weather under an
# indoor roof, 112 St. Louis dome games and 125 San Diego games took Los Angeles
# weather. MEASURED against the pinned 2002-2025 feed, 1,153 of 6,499 games
# resolved to a stadium other than the one they were played at -- and not one of
# them raised.
#
# WHAT CHANGED HERE, AND ON WHOSE AUTHORITY. Plan 33-06 gated the repair on a
# CONJUNCTION -- a module constant `STADIUM_ID_ROUTING_FIRST_SEASON = 2026` AND the
# game being neutral-site -- so history kept its home-team answer and the 91
# historical neutral-site games stayed a recorded DISCLOSURE
# (tests/phase33_state.HISTORICAL_NEUTRAL_MISRESOLUTION) that D33-15 deliberately
# deferred. The owner took that repair at Phase 33.1 (decision D33.1-06,
# 2026-09-12). The constant is RETIRED and BOTH halves of the conjunction are gone:
# the season test AND the neutral-site test. Retiring only the season half would
# have fixed nothing for the Oakland games, because they are NOT neutral-site --
# only 83 of the 1,153 misroutes are (33.1-RESEARCH.md pitfall P-1).
#
# WHAT IT COSTS, STATED HERE RATHER THAN DISCOVERED LATER. This WIDENS SPEC R5's
# acceptance from "the measured column-level diff touches only weather-family
# columns and the new coverage flag" to "weather-family columns, the
# venue/travel/timezone/elevation family, and the new coverage flag". The rung Plan
# 33.1-07 rebuilds therefore carries a COMPOUND cause and is never called "the
# weather rung" without qualification.
#
# THE MATCH IS EXACT AND CASE-SENSITIVE, unchanged. No casefolding, no stripping,
# no fuzzy match -- and now no fallback either. An id absent from data/venues.json
# RAISES and names the game, because the home team's stadium is the wrong answer by
# construction and a plausible wrong answer arriving quietly is the whole defect.
# ---------------------------------------------------------------------------

_VENUES_JSON_PATH = Path(__file__).resolve().parent.parent / "data" / "venues.json"

# The nflverse `location` value that marks a neutral-site game, and the silver column
# that will carry the same fact once Plan 33-12 backfills it. Both spellings are read
# because the predicate below runs against BOTH shapes: the raw feed frame during
# ingestion and the silver games frame during the feature build.
#
# NOTE, since it changed under D33.1-06: `game_is_neutral_site` is NO LONGER part of
# the routing decision. Every game routes by its own `stadium_id`, neutral or not.
# The predicate survives because "was this game at a neutral site" is still a fact
# worth asking (the Plan 33.1-03 diff reports it, and Plan 33-12 backfills the silver
# column), but nothing in this module branches on it any more.
_NEUTRAL_LOCATION_VALUE = "Neutral"


class UnknownStadiumError(ValueError):
    """A game names a ``stadium_id`` ``data/venues.json`` does not carry.

    Raised rather than falling back to the home team's stadium: the home team's
    venue is the WRONG answer by construction for any game not played there, and it
    is precisely the wrong answer that shipped silently for the whole life of this
    project -- 1,153 games of it. The message follows the repo's
    refusal-carries-its-recovery-command convention: it names the id, the game, and
    the file to edit.

    Under D33-15 this was scoped to season >= 2026 neutral-site games. D33.1-06
    widened it to every game of every season.
    """


def _load_venue_records(
    venues: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """The venue records, either as handed in or read from ``data/venues.json``.

    Callers inside the feature build pass their already-loaded copy; the
    module-level helpers below read the file so they are usable from a one-line
    verification command without constructing a calculator.
    """
    if venues is not None:
        return venues
    with open(_VENUES_JSON_PATH, encoding="utf-8") as handle:
        return json.load(handle)["venues"]


def venue_record_for_stadium_id(
    stadium_id: object,
    venues: list[dict[str, Any]] | None = None,
    *,
    game_id: object = None,
) -> dict[str, Any]:
    """The ONE exact ``stadium_id`` -> venue-record resolver (Ruling I2).

    This is the single place the lookup DECISION lives. ``resolve_venue_for_game``
    below routes through it, and ``scripts.ingest_weather`` DELEGATES to it rather
    than carrying its own copy -- so neither the match nor the refusal string exists
    twice, and the contextual builder and both weather paths cannot drift apart
    about which stadium a game was played at or about what to tell an operator when
    the record is missing.

    WHY THE EXTRACTION IS REAL AND NOT DESCRIPTIVE. Before Plan 33.1-03 there were
    two independently maintained implementations -- this module's router and
    ``WeatherDataIngester._get_venue_record_by_stadium_id`` -- that happened to
    agree. Agreeing today is not being one rule: the next edit to either is where
    they diverge. ``tests/unit/test_venue_resolver_is_shared.py`` proves the merge
    by MUTATION (monkeypatching this function moves BOTH consumers' answers) rather
    than by an equality check over the real data, which would pass identically
    against two copies.

    THE MATCH IS EXACT AND CASE-SENSITIVE. No casefolding, no stripping, no fuzzy
    match, and no fallback to the home team.

    Args:
        stadium_id: The nflverse stadium code from the feed (e.g. ``OAK00``).
        venues: Optional pre-loaded venue records; ``data/venues.json`` otherwise.
        game_id: Optional, for the refusal message only. A refusal that cannot name
            the game leaves the reader to guess which of 6,499 it was about.

    Returns:
        The venue record.

    Raises:
        UnknownStadiumError: ``stadium_id`` is absent, empty, non-string, or not in
            the venue records.
    """
    if isinstance(stadium_id, str) and stadium_id:
        for venue in _load_venue_records(venues):
            if venue.get("stadium_id") == stadium_id:
                return venue

    subject = (
        f"game {game_id!r} names stadium_id {stadium_id!r}"
        if game_id is not None
        else f"No venue found for stadium_id {stadium_id!r}"
    )
    raise UnknownStadiumError(
        f"{subject}, which is not in data/venues.json. It is NOT resolved to the "
        "home team's stadium: that is the wrong answer by construction and produces "
        "a game with the wrong coordinates, the wrong timezone, the wrong weather "
        "and zero travel miles, silently. Add the venue record to data/venues.json "
        "with its stadium_id, latitude, longitude, elevation_ft, IANA timezone and "
        "roof_type entered explicitly, then re-run. Matching is exact and "
        "case-sensitive."
    )


def _get_venue_by_stadium_id(
    stadium_id: object,
    venues: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """A SOFT miss over :func:`venue_record_for_stadium_id`: the record, or None.

    A thin wrapper, deliberately NOT a second implementation. Some callers -- the
    corpus-completeness tests, the roof lookup in ``scripts.ingest_games`` -- are
    asking "is this code known?" rather than "where was this game played?", and for
    that question a miss is an answer rather than a refusal.
    """
    try:
        return venue_record_for_stadium_id(stadium_id, venues)
    except UnknownStadiumError:
        return None


def _get_venue_by_home_team(
    home_team: object,
    venues: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """The venue whose ``home_teams`` contains the team -- "where does this TEAM play".

    NO LONGER REACHABLE FROM ANY GAME-VENUE RESOLUTION PATH (D33.1-06). It answers a
    different question from "where was this game played", and that difference is
    exactly what produced 1,153 misroutes while it was the routing rule. It stays
    because the question it does answer is still asked -- the away team's home
    surface in ``_compute_surface_mismatch``, and the team-to-venue mapping the
    travel origin is read from.
    """
    if not isinstance(home_team, str) or not home_team:
        return None
    for venue in _load_venue_records(venues):
        if home_team in venue.get("home_teams", []):
            return venue
    return None


def game_is_neutral_site(game: Any) -> bool:
    """Is this game at a neutral site, in either spelling?

    The feed says ``location == 'Neutral'``; silver will say ``neutral_site == True``
    once Plan 33-12 backfills it. ``location`` is preferred where present because it
    is the source the other value is derived FROM.
    """
    location = game.get("location")
    if isinstance(location, str):
        return location.strip() == _NEUTRAL_LOCATION_VALUE

    neutral = game.get("neutral_site")
    if neutral is None or (isinstance(neutral, float) and pd.isna(neutral)):
        return False
    return bool(neutral)


def resolve_venue_for_game(
    game: Any,
    venues: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve the venue a game was actually played at (D33.1-06).

    ONE routing rule, applied to EVERY season and EVERY game: the game's own
    ``stadium_id``. There is no season test and no neutral-site test, and there is
    no home-team fallback -- an unresolvable id raises rather than returning a
    plausible wrong answer.

    ``season``, ``home_team``, ``location`` and ``neutral_site`` are no longer read
    by this function at all. That is the point: the game says where it was played,
    and nothing else gets a vote.

    Args:
        game: A mapping-like game row. Read: ``stadium_id``, and ``game_id`` for the
            refusal message.
        venues: Optional pre-loaded venue records.

    Returns:
        The venue record.

    Raises:
        UnknownStadiumError: the game's ``stadium_id`` is absent from
            ``data/venues.json``.
    """
    return venue_record_for_stadium_id(
        game.get("stadium_id"), venues, game_id=game.get("game_id")
    )


# ---------------------------------------------------------------------------
# Feature-bearing venue fields, VALIDATED rather than defaulted (Ruling I3).
#
# `encode_venue_features` wraps its whole body in an `except` that returns
# `_default_venue_features()`. Corpus-completeness tests protect a MISSING venue;
# nothing protected a PRESENT venue with a MALFORMED feature-bearing field, and each
# of the five such fields crosses a band or an equality:
#
#     roof_type     -> venue_outdoor / venue_indoor / venue_retractable
#     elevation_ft  -> venue_elevation_ft and the >= 3000 venue_high_altitude band
#     climate_zone  -> venue_cold_climate / venue_warm_climate
#     capacity      -> venue_capacity and the >= 75000 venue_large_stadium band
#     surface       -> the grass/turf mismatch feature
#
# A malformed cell would therefore move a gold column on up to 1,082 games for a
# reason nobody chose, silently, inside Plan 33.1-07's rung -- which is why that
# plan carries `validate_venue_feature_fields` as a PRECONDITION on the rebuild.
#
# THE VOCABULARIES ARE LITERALS AND ARE DELIBERATELY NOT DERIVED FROM
# data/venues.json. Deriving the allowed set from the file being validated lets a
# malformed record legalise itself by being present, which is a validator that
# cannot fail. The cross-check that these literals still cover the RATIFIED records
# lives on the test side (tests/unit/test_venue_feature_fields_are_valid.py), where
# `tests.phase33_state`'s owner-ratified tables are importable -- a production module
# must not import from `tests/`.
# ---------------------------------------------------------------------------

VENUE_FEATURE_BEARING_FIELDS: tuple[str, ...] = (
    "roof_type",
    "elevation_ft",
    "climate_zone",
    "capacity",
    "surface",
)

# The three values `encode_venue_features` actually tests for. Anything else encodes
# to all-zeros across the three roof flags, which is not a state any venue is in.
VENUE_ROOF_TYPE_VOCABULARY: frozenset[str] = frozenset(
    {"outdoor", "indoor", "retractable"}
)

# The closed climate vocabulary. Only `humid_continental` sets venue_cold_climate and
# only `humid_subtropical`/`tropical`/`desert` set venue_warm_climate; the rest are
# INERT BY DESIGN and are listed here so that being inert is a ratified state rather
# than the silent consequence of a typo (see test_the_new_climate_token_encodes_to_
# neither_flag, which pins exactly that for `subtropical_highland`).
VENUE_CLIMATE_ZONE_VOCABULARY: frozenset[str] = frozenset(
    {
        "desert",
        "humid_continental",
        "humid_subtropical",
        "mediterranean",
        "oceanic",
        "semi_arid",
        "subtropical_highland",
        "tropical",
    }
)


def validate_venue_feature_fields(
    venues: list[dict[str, Any]] | None = None,
) -> None:
    """Every feature-bearing field on every venue record is well formed, or refuse.

    Checks all of :data:`VENUE_FEATURE_BEARING_FIELDS` on every record in ONE pass
    and refuses naming EVERY offending ``<stadium_id>.<field>`` pair with its value,
    so an operator fixing the file sees the whole list rather than discovering the
    second fault after fixing the first.

    Args:
        venues: Optional pre-loaded venue records; ``data/venues.json`` otherwise.

    Raises:
        ValueError: one or more records carry a malformed feature-bearing field.
    """
    offences: list[str] = []

    for index, venue in enumerate(_load_venue_records(venues)):
        # A record with no stadium_id at all still has to be nameable in the
        # refusal, or the message points at nothing.
        name = venue.get("stadium_id") or venue.get("venue_id") or f"record[{index}]"

        roof_type = venue.get("roof_type")
        if roof_type not in VENUE_ROOF_TYPE_VOCABULARY:
            offences.append(
                f"{name}.roof_type is {roof_type!r}; allowed: "
                f"{sorted(VENUE_ROOF_TYPE_VOCABULARY)}"
            )

        climate_zone = venue.get("climate_zone")
        if climate_zone not in VENUE_CLIMATE_ZONE_VOCABULARY:
            offences.append(
                f"{name}.climate_zone is {climate_zone!r}; allowed: "
                f"{sorted(VENUE_CLIMATE_ZONE_VOCABULARY)}"
            )

        for field in ("elevation_ft", "capacity"):
            value = venue.get(field)
            # `bool` is an `int` in Python, and True would sail through every
            # numeric check below while meaning nothing as an elevation.
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                offences.append(
                    f"{name}.{field} is {value!r}; it must be a finite number, "
                    "entered explicitly"
                )
            elif not np.isfinite(float(value)):
                offences.append(
                    f"{name}.{field} is {value!r}; NaN and infinity are not "
                    "elevations or capacities"
                )

        surface = venue.get("surface")
        if not isinstance(surface, str) or not surface.strip():
            offences.append(
                f"{name}.surface is {surface!r}; it must be a non-empty string"
            )

    if offences:
        listed = "\n  ".join(offences)
        raise ValueError(
            f"{len(offences)} malformed feature-bearing venue field(s) in "
            "data/venues.json:\n  "
            f"{listed}\n"
            "Each of these fields crosses a band or an equality in "
            "features.contextual.encode_venue_features, so a malformed value would "
            "be caught there and SILENTLY replaced by _default_venue_features() -- "
            "moving a gold column for a reason nobody chose. Fix the named cells in "
            "data/venues.json; do not widen this validator."
        )


class ContextualFeaturesCalculator:
    """
    Calculate contextual features for NFL games.

    Features calculated:
    - Travel distance and time zone differences
    - Short week indicators (Thursday games, etc.)
    - Venue roof type encoding (indoor/outdoor/retractable)
    - Home field advantage indicators
    - Rest days since last game
    - Travel fatigue metrics
    """

    def __init__(self):
        """Initialize contextual features calculator."""
        self.settings = get_settings()

        # Load venue data
        self.venues_data = self._load_venues_data()
        # Every surface spelling must be classified BEFORE any game is built: an
        # unclassified one refuses here, outside the build's optional-source handler.
        validate_surface_vocabulary(self.venues_data["venues"])

        # Time zone mappings
        self.timezone_map = self._build_timezone_map()

        # Team to venue mapping
        self.team_venues = self._build_team_venue_mapping()

    def _load_venues_data(self) -> dict[str, Any]:
        """Load venue data from JSON file."""
        try:
            # Use current working directory as project root
            project_root = Path.cwd()
            venues_path = project_root / "data" / "venues.json"

            with open(venues_path) as f:
                venues_data = json.load(f)

            logger.info("Loaded venues data", venues_count=len(venues_data["venues"]))
            return venues_data

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error("Failed to load venues data", error=str(e))
            raise

    def _build_timezone_map(self) -> dict[str, str]:
        """Build mapping of venue IDs to timezones."""
        timezone_map = {}

        for venue in self.venues_data["venues"]:
            timezone_map[venue["venue_id"]] = venue["timezone"]

        return timezone_map

    def _build_team_venue_mapping(self) -> dict[str, str]:
        """Build mapping of teams to their home venue IDs."""
        team_venues = {}

        for venue in self.venues_data["venues"]:
            for team in venue["home_teams"]:
                team_venues[team] = venue["venue_id"]

        return team_venues

    def _get_venue_by_id(self, venue_id: str) -> dict[str, Any] | None:
        """Get venue information by venue ID."""
        for venue in self.venues_data["venues"]:
            if venue["venue_id"] == venue_id:
                return venue
        return None

    def _get_venue_by_stadium_id(self, stadium_id: object) -> dict[str, Any] | None:
        """Get venue information by nflverse ``stadium_id`` -- exact, case-sensitive.

        ``venue_id`` is a slug (``highmark_stadium``) and ``stadium_id`` is an
        nflverse code (``BUF00``); they are different keys and one is not a
        substitute for the other.
        """
        return _get_venue_by_stadium_id(stadium_id, self.venues_data["venues"])

    def resolve_venue_for_game(self, game: Any) -> dict[str, Any]:
        """Resolve a game's true venue under the D33.1-06 routing rule."""
        return resolve_venue_for_game(game, self.venues_data["venues"])

    def facts_for_game(self, game: Any) -> FactsAtLock:
        """The venue, kickoff and week this game's features may use (Plan 33.2-10).

        Read through ``features.schedule_moves.facts_at_lock``, the ONE place the
        last-admissible-move selection is made. A game moved by an emergency announced
        after its lock gets the facts as they stood before the move; every other game
        gets the facts silver records. The venue, roof, surface, travel, time zone,
        weekday and season position below all derive from these facts, so they cannot
        disagree with each other or with the weather writer about where and when a game
        was scheduled at its lock.
        """
        return facts_at_lock(game.get("game_id"), game)

    def _resolve_venue_id_for_game(self, game: Any) -> str:
        """The venue_id the feature build should use for this game.

        THE VENUE AT THE LOCK (Plan 33.2-10). The ``stadium_id`` is taken from
        :meth:`facts_for_game`, so a post-lock emergency move resolves to the venue the
        game was scheduled at before the move. For every other game it is the game's own
        ``stadium_id``, exactly as below.

        EVERY SEASON AND EVERY GAME ROUTES BY ``stadium_id`` (D33.1-06). This
        method's previous docstring said that seasons before 2026 never enter the
        new path, that they keep their venue-NAME resolution byte for byte, and that
        the 91 mis-resolved historical games stay mis-resolved on purpose. None of
        that holds any more: the owner took the repair D33-15 deferred, so history
        is re-resolved here and the resulting gold movement is a declared,
        separately-attributed part of Plan 33.1-07's rung rather than a surprise.

        The venue-NAME path is no longer how a game's own venue is found. It could
        never have been complete -- twenty ``stadium_id`` values carry two to five
        feed ``stadium`` names each -- which is Ruling I.

        Raises:
            UnknownStadiumError: the game's ``stadium_id`` is not in
                ``data/venues.json``, or the frame carries no ``stadium_id`` at all.
        """
        facts = self.facts_for_game(game)
        at_lock = {"game_id": game.get("game_id"), "stadium_id": facts.stadium_id}
        return self.resolve_venue_for_game(at_lock)["venue_id"]

    def _get_venue_surface(self, venue_id: str) -> str | None:
        """Get the surface type for a venue by its ID.

        Args:
            venue_id: Venue identifier.

        Returns:
            Surface string (e.g. 'FieldTurf', 'Bermuda Grass') or None.
        """
        for venue in self.venues_data["venues"]:
            if venue["venue_id"] == venue_id:
                return venue.get("surface")
        return None

    def _is_grass_surface(self, surface: str | None) -> bool:
        """Classify a surface as grass (True) or synthetic (False).

        Args:
            surface: Surface string from venues data.

        Returns:
            True if the surface is a grass type, False otherwise.
        """
        # Delegates to surface_is_grass, the one classification: an unlisted spelling
        # RAISES rather than defaulting to synthetic (Plan 33.2-10 step 3b). None (no
        # surface recorded) is never reached from _compute_surface_mismatch, which
        # returns 0.0 first; it is still answered False here for other callers.
        if surface is None:
            return False
        return surface_is_grass(surface)

    def _compute_surface_mismatch(self, away_team: str, game_venue_id: str) -> float:
        """Compute surface mismatch for the away team.

        Mismatch = 1.0 when the away team's home surface category (grass vs
        synthetic) differs from the game venue's surface category.

        Args:
            away_team: Away team abbreviation.
            game_venue_id: Venue ID where the game is played.

        Returns:
            1.0 if surface categories differ, 0.0 otherwise.
        """
        # Look up away team's home venue surface
        away_home_venue_id = self.team_venues.get(away_team)
        if not away_home_venue_id:
            return 0.0

        away_surface = self._get_venue_surface(away_home_venue_id)
        game_surface = self._get_venue_surface(game_venue_id)

        if away_surface is None or game_surface is None:
            return 0.0

        away_is_grass = self._is_grass_surface(away_surface)
        game_is_grass = self._is_grass_surface(game_surface)

        return 1.0 if away_is_grass != game_is_grass else 0.0

    def _get_venue_id_by_name(self, venue_name: str) -> str:
        """Map venue name to venue ID -- LOSSY, and no longer how a game is routed.

        NOT REACHABLE FROM ANY GAME-VENUE RESOLUTION PATH since D33.1-06 (Ruling I).
        It is kept rather than deleted because the name question is still asked
        elsewhere, and it is NOT repaired because it cannot be: twenty ``stadium_id``
        values carry two to five feed ``stadium`` names each (``OAK00`` has five), so
        no single ``venue_name`` can satisfy an exact-then-partial match for all of a
        venue's games. Its miss behaviour is the reason it had to stop being
        load-bearing -- it returns a synthesised slug rather than raising, and that
        slug flows into the default-metrics branch without a word.
        """
        if not venue_name:
            return ""

        # Try exact match first
        for venue in self.venues_data["venues"]:
            if venue.get("venue_name") == venue_name:
                return venue["venue_id"]

        # Try partial match (handle name variations)
        venue_name_lower = venue_name.lower()
        for venue in self.venues_data["venues"]:
            venue_name_in_data = venue.get("venue_name", "").lower()
            if (
                venue_name_lower in venue_name_in_data
                or venue_name_in_data in venue_name_lower
            ):
                return venue["venue_id"]

        # No match found - create a fallback ID
        logger.warning("Could not map venue name to ID", venue_name=venue_name)
        return venue_name.lower().replace(" ", "_").replace("&", "and")

    def calculate_travel_metrics(
        self,
        away_team: str,
        home_team: str,
        game_venue_id: str,
        kickoff_datetime: datetime,
    ) -> dict[str, float]:
        """
        Calculate travel-related metrics for the away team.

        Args:
            away_team: Away team abbreviation
            home_team: Home team abbreviation
            game_venue_id: Venue ID where game is played
            kickoff_datetime: Game kickoff time (timezone-aware)

        Returns:
            Dictionary with travel metrics
        """
        try:
            # Get away team's home venue
            away_home_venue_id = self.team_venues.get(away_team)
            if not away_home_venue_id:
                logger.warning("No home venue found for away team", away_team=away_team)
                return self._default_travel_metrics()

            # Get venue information
            away_venue = self._get_venue_by_id(away_home_venue_id)
            game_venue = self._get_venue_by_id(game_venue_id)

            if not away_venue or not game_venue:
                logger.warning(
                    "Venue data not found",
                    away_venue_id=away_home_venue_id,
                    game_venue_id=game_venue_id,
                )
                return self._default_travel_metrics()

            # Calculate distance using Haversine formula
            travel_distance = self._calculate_distance(
                away_venue["latitude"],
                away_venue["longitude"],
                game_venue["latitude"],
                game_venue["longitude"],
            )

            # Calculate timezone difference
            away_tz = ZoneInfo(away_venue["timezone"])
            game_tz = ZoneInfo(game_venue["timezone"])

            # Convert kickoff to both timezones to calculate difference
            kickoff_away_tz = kickoff_datetime.astimezone(away_tz)
            kickoff_game_tz = kickoff_datetime.astimezone(game_tz)

            # Time zone difference in hours (positive = westward travel)
            tz_diff_hours = (
                kickoff_away_tz.utcoffset() - kickoff_game_tz.utcoffset()
            ).total_seconds() / 3600

            # Travel fatigue score (combination of distance and timezone change)
            # Higher score = more fatigue
            distance_factor = min(
                travel_distance / 2500, 1.0
            )  # Normalize to max distance ~2500 miles
            timezone_factor = (
                abs(tz_diff_hours) / 3.0
            )  # Normalize to max 3 hour difference
            travel_fatigue_score = (distance_factor * 0.6) + (timezone_factor * 0.4)

            # Cross-country travel indicator (>= 2000 miles or >= 2 time zones)
            cross_country_travel = (travel_distance >= 2000) or (
                abs(tz_diff_hours) >= 2
            )

            return {
                "travel_distance_miles": travel_distance,
                "timezone_diff_hours": tz_diff_hours,
                "abs_timezone_diff_hours": abs(tz_diff_hours),
                "travel_fatigue_score": travel_fatigue_score,
                "cross_country_travel": 1.0 if cross_country_travel else 0.0,
                "eastward_travel": 1.0 if tz_diff_hours < 0 else 0.0,
                "westward_travel": 1.0 if tz_diff_hours > 0 else 0.0,
            }

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to calculate travel metrics",
                away_team=away_team,
                game_venue_id=game_venue_id,
                error=str(e),
            )
            return self._default_travel_metrics()

    def _default_travel_metrics(self) -> dict[str, float]:
        """Return default travel metrics when calculation fails."""
        return {
            "travel_distance_miles": 0.0,
            "timezone_diff_hours": 0.0,
            "abs_timezone_diff_hours": 0.0,
            "travel_fatigue_score": 0.0,
            "cross_country_travel": 0.0,
            "eastward_travel": 0.0,
            "westward_travel": 0.0,
        }

    def _calculate_distance(
        self, lat1: float, lon1: float, lat2: float, lon2: float
    ) -> float:
        """
        Calculate distance between two points using Haversine formula.

        Args:
            lat1, lon1: Latitude and longitude of first point
            lat2, lon2: Latitude and longitude of second point

        Returns:
            Distance in miles
        """
        # Convert to radians
        lat1_rad = np.radians(lat1)
        lon1_rad = np.radians(lon1)
        lat2_rad = np.radians(lat2)
        lon2_rad = np.radians(lon2)

        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (
            np.sin(dlat / 2) ** 2
            + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
        )
        c = 2 * np.arcsin(np.sqrt(a))

        # Earth radius in miles
        earth_radius_miles = 3959.0

        return earth_radius_miles * c

    def detect_short_week(
        self, kickoff_datetime: datetime, season: int, week: int
    ) -> dict[str, float]:
        """
        Detect short week scenarios (Thursday night games, etc.).

        Args:
            kickoff_datetime: Game kickoff time
            season: Season year
            week: Week number

        Returns:
            Dictionary with short week indicators
        """
        try:
            # Get day of week (0=Monday, 6=Sunday)
            game_day = kickoff_datetime.weekday()

            # Thursday games (weekday 3)
            thursday_game = 1.0 if game_day == 3 else 0.0

            # Monday games (weekday 0)
            monday_game = 1.0 if game_day == 0 else 0.0

            # Saturday games (weekday 5) - typically late season/playoffs
            saturday_game = 1.0 if game_day == 5 else 0.0

            # Short week indicator (Thursday or Monday)
            short_week = 1.0 if (game_day in {3, 0}) else 0.0

            # Rest advantage for teams coming off bye week
            # This will be calculated separately when we have game-by-game data

            return {
                "thursday_game": thursday_game,
                "monday_game": monday_game,
                "saturday_game": saturday_game,
                "short_week": short_week,
                "game_day_of_week": float(game_day),
            }

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to detect short week",
                kickoff_datetime=kickoff_datetime,
                error=str(e),
            )
            return {
                "thursday_game": 0.0,
                "monday_game": 0.0,
                "saturday_game": 0.0,
                "short_week": 0.0,
                "game_day_of_week": 6.0,  # Default to Sunday
            }

    def encode_venue_features(self, venue_id: str) -> dict[str, float]:
        """
        Encode venue-specific features.

        Args:
            venue_id: Venue identifier

        Returns:
            Dictionary with venue features
        """
        try:
            venue = self._get_venue_by_id(venue_id)
            if not venue:
                logger.warning("Venue not found", venue_id=venue_id)
                return self._default_venue_features()

            # Roof type encoding
            roof_type = (venue.get("roof_type") or "outdoor").lower()
            outdoor = 1.0 if roof_type == "outdoor" else 0.0
            indoor = 1.0 if roof_type == "indoor" else 0.0
            retractable = 1.0 if roof_type == "retractable" else 0.0

            # Elevation (affects kicking, oxygen levels)
            elevation_ft = venue["elevation_ft"]
            high_altitude = 1.0 if elevation_ft >= 3000 else 0.0  # Denver is ~5280 ft

            # Climate zone encoding
            climate = venue.get("climate_zone", "unknown").lower()
            cold_climate = 1.0 if climate in ["humid_continental"] else 0.0
            warm_climate = (
                1.0 if climate in ["humid_subtropical", "tropical", "desert"] else 0.0
            )

            # Stadium capacity (crowd noise factor)
            capacity = venue.get("capacity", 70000)
            large_stadium = 1.0 if capacity >= 75000 else 0.0

            return {
                "venue_outdoor": outdoor,
                "venue_indoor": indoor,
                "venue_retractable": retractable,
                "venue_elevation_ft": float(elevation_ft),
                "venue_high_altitude": high_altitude,
                "venue_cold_climate": cold_climate,
                "venue_warm_climate": warm_climate,
                "venue_capacity": float(capacity),
                "venue_large_stadium": large_stadium,
            }

        # RETAINED AS DEFENCE-IN-DEPTH, AND UNREACHABLE FOR A COMMITTED RECORD
        # (Ruling I3). Substituting neutral defaults for a malformed field is how a
        # gold column moves for a reason nobody chose, on up to 1,082 games, with
        # nothing but a log line. It is not deleted -- a genuinely absent venue_id
        # still has to land somewhere -- but every feature-bearing field on all 60
        # committed records is checked by `validate_venue_feature_fields` FIRST, and
        # Plan 33.1-07 carries that validation as a stated PRECONDITION on the gold
        # rebuild. The branch is what happens if the validator is skipped; the
        # validator is what stops the branch from being how the system behaves.
        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to encode venue features", venue_id=venue_id, error=str(e)
            )
            return self._default_venue_features()

    def _default_venue_features(self) -> dict[str, float]:
        """Return default venue features when encoding fails."""
        return {
            "venue_outdoor": 1.0,  # Default to outdoor
            "venue_indoor": 0.0,
            "venue_retractable": 0.0,
            "venue_elevation_ft": 500.0,  # Average elevation
            "venue_high_altitude": 0.0,
            "venue_cold_climate": 0.0,
            "venue_warm_climate": 0.0,
            "venue_capacity": 70000.0,  # Average capacity
            "venue_large_stadium": 0.0,
        }

    def calculate_rest_days(
        self,
        team: str,
        current_game_date: datetime,
        games_df: pd.DataFrame,
        kickoffs: pd.Series | None = None,
    ) -> float:
        """
        Calculate rest days for a team since their last game.

        Both sides of the comparison are resolved through ``kickoff_wall_clock_et``
        (WR-06) so a tz-aware caller can never meet a tz-naive ``kickoff_et`` column.
        That mismatch raises ``TypeError``, which the guard below would swallow and
        silently return the 7.0 default for -- turning a bye week into an ordinary
        week with no error and no log line. The live path carries a tz-aware column,
        so this is defensive; the cost of getting it wrong is a wrong ``off_bye``.

        Args:
            team: Team abbreviation
            current_game_date: Date of current game
            games_df: DataFrame with all games
            kickoffs: Optional PRE-COMPUTED ET wall clocks for ``games_df``,
                index-aligned to it. WR-08: mapping ``kickoff_wall_clock_et``
                over the whole games frame costs one ``pd.Timestamp``
                construction per row, and this method is called TWICE per game
                (home and away) inside the per-game loop -- roughly 13,000 x
                6,499 constructions on a full rebuild. Callers in a loop should
                compute the series ONCE and pass it in. Omitting it is still
                correct, just slow.

        Returns:
            Number of rest days
        """
        try:
            if kickoffs is None:
                kickoffs = games_df["kickoff_et"].map(kickoff_wall_clock_et)
            current = kickoff_wall_clock_et(current_game_date)

            # Find team's previous games before current date
            team_games = games_df[
                ((games_df["home_team"] == team) | (games_df["away_team"] == team))
                & (kickoffs < current)
            ]

            if len(team_games) == 0:
                # First game of season, use standard rest
                return 7.0

            # Get most recent game
            last_game_date = kickoffs.loc[team_games.index].max()

            # Calculate rest days
            rest_days = (current - last_game_date).days

            return float(rest_days)

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error("Failed to calculate rest days", team=team, error=str(e))
            return 7.0  # Default to standard week

    # ------------------------------------------------------------------
    # Situational spot features (D-15/D-16, SIG-03)
    #
    # off_bye, look_ahead_spot (trap) and letdown_spot are the genuinely-new
    # adds for this signal phase. They are WEAK signals, largely priced-in by
    # the market, and are NOT a standing bet angle (SC3) -- a drop in the Plan
    # 28-07 add-one-in lift screen is an expected, acceptable outcome (D-17).
    # The existing rest/travel/short-week/bye/divisional features are REUSED,
    # not re-derived (D-15, they are already in gold).
    # ------------------------------------------------------------------

    def _load_full_season_schedule(self, seasons: list[int]) -> pd.DataFrame:
        """Load the FULL-season schedule with raw silver Elo + results.

        Review #4 / T-28-08b: the look-ahead/letdown derivation needs the full
        season schedule (next-opponent identity/Elo and the "beat last week"
        prior result) -- but ``build_features`` hands this builder a ``games_df``
        already filtered to the target week (the :783 reassignment), which then
        collapses prior/next context to ~empty in a ``--current-week``
        incremental build. So the schedule is reloaded here INDEPENDENTLY of the
        handed-in frame. The full schedule + the pre-freeze Elo ratings are known
        at the Friday freeze, so next-opponent identity and Elo are NOT leakage;
        only a future game RESULT would be (and this builder never reads one).

        The raw silver Elo (``home_elo_pre``/``away_elo_pre``, std ~125) is the
        contract -- NEVER the z-scored gold Elo (T-28-09).

        Args:
            seasons: Seasons to load the schedule for.

        Returns:
            Per-game schedule with game_id, season, week, home_team, away_team,
            home_elo_pre, away_elo_pre, kickoff_et, home_score, away_score.
        """
        elo = load_dataframe("elo_game_snapshots", layer="silver")
        games = load_dataframe("games", layer="silver")

        elo = elo[elo["season"].isin(seasons)]
        score_cols = ["game_id", "kickoff_et", "home_score", "away_score"]
        games = games[games["season"].isin(seasons)][score_cols]

        # Bring kickoff + results onto the Elo schedule (keyed by game_id). The
        # Elo snapshot frame is the source of the raw pre-game ratings.
        return elo.merge(games, on="game_id", how="left")

    @staticmethod
    def _team_elo_in_game(row: pd.Series, team: str) -> float | None:
        """Return ``team``'s raw pre-game Elo in a schedule ``row``, or None."""
        if row["home_team"] == team:
            value = row.get("home_elo_pre")
        elif row["away_team"] == team:
            value = row.get("away_elo_pre")
        else:
            return None
        return float(value) if pd.notna(value) else None

    @staticmethod
    def _team_result_in_game(
        row: pd.Series, team: str
    ) -> tuple[float | None, float | None]:
        """Return ``(team_score, opp_score)`` for a schedule ``row``.

        Either side may be None when the score is missing (e.g. a future game
        whose result is not yet known).
        """
        if row["home_team"] == team:
            team_score, opp_score = row.get("home_score"), row.get("away_score")
        elif row["away_team"] == team:
            team_score, opp_score = row.get("away_score"), row.get("home_score")
        else:
            return None, None
        team_score = float(team_score) if pd.notna(team_score) else None
        opp_score = float(opp_score) if pd.notna(opp_score) else None
        return team_score, opp_score

    def _derive_spot_flags(
        self,
        target_games: pd.DataFrame,
        full_schedule: pd.DataFrame,
        as_of_datetime: datetime,
    ) -> dict[str, dict[str, float]]:
        """Derive look-ahead (trap) and letdown spot flags per target game.

        These spots are WEAK and largely priced-in (SC3); they are added for
        completeness and the Plan 28-07 screen, not as a bet angle.

        Leakage contract (D-16, T-28-08): the next-opponent identity and the
        pre-freeze Elo ratings are KNOWN at the freeze and are NOT leakage. Only
        a future (>= current week) game RESULT is future information, and this
        derivation NEVER reads one -- so revealing a future look-ahead result
        leaves the flags byte-unchanged (proven by
        tests/unit/test_situational_no_leakage.py). The letdown's "beat last
        week" component only counts a prior game whose kickoff is before
        ``as_of_datetime`` (the freeze), so a not-yet-played prior game
        contributes nothing.

        Thresholds use the raw silver Elo scale (``ELO_SPOT_STEP`` = 100 Elo,
        ~0.8 std) and reuse ``ratings.elo.is_divisional_game`` (T-28-09).

        Args:
            target_games: Games to emit spot flags for.
            full_schedule: FULL-season schedule with raw Elo + results, loaded
                independently of any target-week filter (review #4).
            as_of_datetime: Freeze cutoff; a prior RESULT only counts toward a
                letdown if its kickoff precedes this cutoff.

        Returns:
            Mapping game_id -> the four home/away look_ahead/letdown spot flags.
        """
        flags: dict[str, dict[str, float]] = {}
        if full_schedule is None or len(full_schedule) == 0:
            return flags

        sched = full_schedule.copy()
        # tz-aligned freeze cutoff for prior-result gating (mirrors the rest
        # fence idiom at build_features :829-833).
        kickoff_series = pd.to_datetime(sched["kickoff_et"])
        cutoff_ts = pd.Timestamp(as_of_datetime)
        if kickoff_series.dt.tz is not None and cutoff_ts.tz is None:
            cutoff_ts = cutoff_ts.tz_localize(kickoff_series.dt.tz)
        sched = sched.assign(_kickoff_ts=kickoff_series)

        for _, game in target_games.iterrows():
            game_id = game["game_id"]
            season = game["season"]
            week = game["week"]
            home_team = game["home_team"]
            away_team = game["away_team"]

            game_flags = {
                "home_look_ahead_spot": 0.0,
                "away_look_ahead_spot": 0.0,
                "home_letdown_spot": 0.0,
                "away_letdown_spot": 0.0,
            }
            flags[game_id] = game_flags

            cur = sched[sched["game_id"] == game_id]
            if cur.empty:
                # No raw Elo for this game -> leave neutral defaults.
                continue
            cur_row = cur.iloc[0]

            for side, team, opp in (
                ("home", home_team, away_team),
                ("away", away_team, home_team),
            ):
                team_elo = self._team_elo_in_game(cur_row, team)
                opp_elo = self._team_elo_in_game(cur_row, opp)
                if team_elo is None or opp_elo is None:
                    continue

                # Current opponent is "weak": their Elo is >= one step below us.
                current_weak = (team_elo - opp_elo) >= ELO_SPOT_STEP
                if not current_weak:
                    continue

                team_mask = (
                    (sched["home_team"] == team) | (sched["away_team"] == team)
                ) & (sched["season"] == season)
                team_sched = sched[team_mask].sort_values("week")
                next_games = team_sched[team_sched["week"] > week]
                prev_games = team_sched[team_sched["week"] < week]

                game_flags[f"{side}_look_ahead_spot"] = self._look_ahead_flag(
                    team, opp_elo, next_games, sched, season, week
                )
                game_flags[f"{side}_letdown_spot"] = self._letdown_flag(
                    team, opp_elo, prev_games, cutoff_ts
                )

        return flags

    def _freeze_known_elo(
        self, sched: pd.DataFrame, season: int, team: str, week: int
    ) -> float | None:
        """Return ``team``'s most-recent pre-game Elo from a week <= ``week`` game.

        This is the opponent strength KNOWN AT THE FREEZE (WR-01). The pre-game
        Elo entering week ``W`` is computed from results through week ``W-1``,
        all settled at the week-``W`` Friday freeze; the pre-game Elo entering
        week ``W+1`` is NOT -- it embeds the as-yet-unplayed week-``W`` result.
        So a next-opponent's strength must be read from a ``week <= W`` row,
        never their ``W+1`` pre-game Elo. Mirrors the letdown path, which only
        reads prior (``week < W``) rows.
        """
        team_rows = sched[
            ((sched["home_team"] == team) | (sched["away_team"] == team))
            & (sched["season"] == season)
            & (sched["week"] <= week)
        ].sort_values("week")
        for _, row in team_rows.iloc[::-1].iterrows():
            elo = self._team_elo_in_game(row, team)
            if elo is not None:
                return elo
        return None

    def _look_ahead_flag(
        self,
        team: str,
        opp_elo: float,
        next_games: pd.DataFrame,
        sched: pd.DataFrame,
        season: int,
        week: int,
    ) -> float:
        """Look-ahead (trap): current opp weak AND next opp notably stronger.

        "Notably stronger" = the next opponent's FREEZE-KNOWN Elo exceeds this
        week's opponent Elo by >= ELO_SPOT_STEP, OR the next opponent is
        divisional. The next opponent's strength is read via ``_freeze_known_elo``
        (their most recent ``week <= W`` pre-game Elo), NOT their ``W+1`` pre-game
        Elo, which would embed the as-yet-unplayed week-W result (WR-01). Uses
        the published schedule (next-opponent identity) + pre-freeze Elo only
        (no result) -- not leakage.
        """
        if len(next_games) == 0:
            return 0.0
        nrow = next_games.iloc[0]
        next_opp = nrow["away_team"] if nrow["home_team"] == team else nrow["home_team"]
        next_opp_elo = self._freeze_known_elo(sched, season, next_opp, week)
        notably_stronger = (
            next_opp_elo is not None and (next_opp_elo - opp_elo) >= ELO_SPOT_STEP
        )
        next_divisional = is_divisional_game(team, next_opp)
        return 1.0 if (notably_stronger or next_divisional) else 0.0

    def _letdown_flag(
        self,
        team: str,
        opp_elo: float,
        prev_games: pd.DataFrame,
        cutoff_ts: pd.Timestamp,
    ) -> float:
        """Letdown: current opp weak AND last week was an emotional game.

        "Emotional game" = last week the team BEAT an opponent whose pre-game
        Elo was >= ELO_SPOT_STEP above this week's opponent, OR last week's
        opponent was divisional. The "beat" component reads the prior RESULT, so
        the prior game must have been PLAYED before the freeze
        (``_kickoff_ts < cutoff_ts``) -- a not-yet-played prior game contributes
        nothing (the time-fence, T-28-08).
        """
        if len(prev_games) == 0:
            return 0.0
        prow = prev_games.iloc[-1]
        if not (prow["_kickoff_ts"] < cutoff_ts):
            # Last week's game has not been played as-of the freeze.
            return 0.0
        prev_opp = prow["away_team"] if prow["home_team"] == team else prow["home_team"]
        prev_opp_elo = self._team_elo_in_game(prow, prev_opp)
        team_score, opp_score = self._team_result_in_game(prow, team)
        won = (
            team_score is not None and opp_score is not None and team_score > opp_score
        )
        beat_strong = (
            won
            and prev_opp_elo is not None
            and (prev_opp_elo - opp_elo) >= ELO_SPOT_STEP
        )
        prev_divisional = is_divisional_game(team, prev_opp)
        return 1.0 if (beat_strong or prev_divisional) else 0.0

    def build_contextual_features(
        self,
        games_df: pd.DataFrame,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """
        Build contextual features for all games (legacy interface).

        .. deprecated::
            Use :meth:`build_features` instead (conforms to FeatureBuilder Protocol).

        Args:
            games_df: DataFrame with game information
            target_season: Specific season to calculate features for
            target_week: Specific week to calculate features for

        Returns:
            DataFrame with contextual features added
        """
        warnings.warn(
            "build_contextual_features is deprecated; use build_features instead",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(
            "Building contextual features",
            games=len(games_df),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Filter to target if specified
            if target_season and target_week:
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

            contextual_features = []

            # WR-08: hoist the ET wall-clock map OUT of the per-game loop. It was
            # recomputed inside calculate_rest_days on every call -- twice per
            # game -- so a full rebuild paid ~13,000 x 6,499 pd.Timestamp
            # constructions for a series that never changes.
            all_kickoffs = games_df["kickoff_et"].map(kickoff_wall_clock_et)

            for _, game in games_df.iterrows():
                game_id = game["game_id"]
                home_team = game["home_team"]
                away_team = game["away_team"]
                # Resolve the venue the game is actually PLAYED at (D33.1-06).
                # EVERY game, EVERY season, by its own stadium_id. An id absent
                # from data/venues.json raises here rather than resolving to the
                # home team's stadium.
                venue_id = self._resolve_venue_id_for_game(game)
                # WR-06 / N-02: resolve the kickoff to its ET WALL CLOCK exactly
                # once, here, so detect_short_week, the travel metrics and the
                # rest-days call all read the same instant through the one
                # documented accessor. Reading the raw cell makes the weekday
                # family (thursday_game / monday_game / saturday_game /
                # short_week / game_day_of_week) depend on which games copy
                # load_dataframe happened to resolve: it is correct on the
                # ET-typed DuckDB table and WRONG for 718 of 6,499 rows on the
                # UTC-typed parquet.
                kickoff_dt = self.facts_for_game(game).kickoff_et
                season = game["season"]
                week = game["week"]

                # Basic game identifiers
                game_features = {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                }

                # Home/away indicators
                game_features.update(
                    {
                        "is_home_game": 1.0,  # For home team perspective
                        "is_away_game": 0.0,  # For home team perspective
                    }
                )

                # Travel metrics (for away team)
                travel_metrics = self.calculate_travel_metrics(
                    away_team, home_team, venue_id, kickoff_dt
                )

                # Add prefix to indicate these are for away team
                for key, value in travel_metrics.items():
                    game_features[f"away_{key}"] = value

                # Short week detection
                short_week_features = self.detect_short_week(kickoff_dt, season, week)
                game_features.update(short_week_features)

                # Venue features
                venue_features = self.encode_venue_features(venue_id)
                game_features.update(venue_features)

                # Rest days (calculate for both teams)
                home_rest_days = self.calculate_rest_days(
                    home_team, kickoff_dt, games_df, kickoffs=all_kickoffs
                )
                away_rest_days = self.calculate_rest_days(
                    away_team, kickoff_dt, games_df, kickoffs=all_kickoffs
                )

                game_features.update(
                    {
                        "home_rest_days": home_rest_days,
                        "away_rest_days": away_rest_days,
                        "rest_advantage": home_rest_days
                        - away_rest_days,  # Positive = home team more rested
                        "both_short_rest": 1.0
                        if (home_rest_days <= 4 and away_rest_days <= 4)
                        else 0.0,
                        "home_short_rest": 1.0 if home_rest_days <= 4 else 0.0,
                        "away_short_rest": 1.0 if away_rest_days <= 4 else 0.0,
                    }
                )

                contextual_features.append(game_features)

            # Convert to DataFrame
            features_df = pd.DataFrame(contextual_features)

            logger.info(
                "Built contextual features",
                features_count=len(features_df),
                feature_columns=len(features_df.columns),
            )

            return features_df

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error("Failed to build contextual features", error=str(e))
            raise

    def get_contextual_features_for_game(
        self, game_id: str, season: int, week: int
    ) -> dict[str, float]:
        """
        Get contextual features for a specific game.

        Args:
            game_id: Game identifier
            season: Season year
            week: Week number

        Returns:
            Dictionary with contextual features
        """
        try:
            # Load contextual features
            features_df = load_dataframe("contextual_features", layer="silver")

            # Filter to specific game
            game_features = features_df[
                (features_df["game_id"] == game_id)
                & (features_df["season"] == season)
                & (features_df["week"] == week)
            ]

            if len(game_features) == 0:
                logger.warning(
                    "No contextual features found for game",
                    game_id=game_id,
                    season=season,
                    week=week,
                )
                return {}

            # Convert to dictionary, excluding non-feature columns
            exclude_cols = ["game_id", "season", "week", "home_team", "away_team"]
            features_dict = {}

            game_row = game_features.iloc[0]
            for col in game_features.columns:
                if col not in exclude_cols:
                    features_dict[col] = game_row[col]

            return features_dict

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to get contextual features for game",
                game_id=game_id,
                season=season,
                week=week,
                error=str(e),
            )
            return {}

    def validate_contextual_features(self, features_df: pd.DataFrame) -> bool:
        """
        Validate contextual features for data quality.

        Args:
            features_df: Contextual features DataFrame

        Returns:
            True if validation passes
        """
        if len(features_df) == 0:
            logger.error("No contextual features found")
            return False

        # Check for required columns
        required_cols = ["game_id", "season", "week", "home_team", "away_team"]
        missing_cols = set(required_cols) - set(features_df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing_columns=list(missing_cols))
            return False

        # Check travel distance ranges (should be 0-3000 miles)
        if "away_travel_distance_miles" in features_df.columns:
            distances = features_df["away_travel_distance_miles"].dropna()
            if len(distances) > 0:
                if distances.min() < 0 or distances.max() > 5000:
                    logger.warning(
                        "Travel distances outside reasonable range",
                        min_distance=distances.min(),
                        max_distance=distances.max(),
                    )

        # Check timezone differences (should be -3 to +3 hours)
        if "away_timezone_diff_hours" in features_df.columns:
            tz_diffs = features_df["away_timezone_diff_hours"].dropna()
            if len(tz_diffs) > 0:
                if tz_diffs.min() < -4 or tz_diffs.max() > 4:
                    logger.warning(
                        "Timezone differences outside reasonable range",
                        min_tz_diff=tz_diffs.min(),
                        max_tz_diff=tz_diffs.max(),
                    )

        # Check rest days (should be 3-21 typically)
        rest_cols = ["home_rest_days", "away_rest_days"]
        for col in rest_cols:
            if col in features_df.columns:
                rest_days = features_df[col].dropna()
                if len(rest_days) > 0:
                    if rest_days.min() < 0 or rest_days.max() > 30:
                        logger.warning(
                            f"Rest days outside reasonable range for {col}",
                            min_rest=rest_days.min(),
                            max_rest=rest_days.max(),
                        )

        logger.info(
            "Contextual features validation completed",
            records=len(features_df),
            feature_columns=len(
                [
                    col
                    for col in features_df.columns
                    if col
                    not in ["game_id", "season", "week", "home_team", "away_team"]
                ]
            ),
        )

        return True

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
    ) -> pd.DataFrame:
        """Build contextual features conforming to FeatureBuilder Protocol.

        Uses only schedule data available before ``as_of_datetime``.

        Delegates to the existing ``build_contextual_features`` logic
        but adds ``as_of_datetime`` filtering.

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff.
            target_season: Optional season filter.
            target_week: Optional week filter.

        Returns:
            DataFrame with contextual features.
        """
        logger.info(
            "Building contextual features (Protocol)",
            games=len(games_df),
            as_of=as_of_datetime.isoformat(),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Time-fence: only use games with kickoff before as_of_datetime
            # for rest-days / schedule context (the target games themselves
            # may have kickoff after the cutoff, but prior games used for
            # rest calculations must be before the cutoff).
            if target_season and target_week:
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

            # Situational spot flags (D-16, SIG-03): derive look-ahead/letdown
            # from the FULL season schedule reloaded INDEPENDENTLY of the
            # (possibly target-week-filtered) games_df above (review #4 /
            # T-28-08b). These spots are weak / optional, so any failure to load
            # the schedule degrades to neutral 0.0 flags and never breaks the
            # contextual build.
            spot_flags: dict[str, dict[str, float]] = {}
            full_schedule: pd.DataFrame | None = None
            if len(games_df) > 0:
                try:
                    target_seasons = sorted(
                        {int(s) for s in games_df["season"].unique()}
                    )
                    full_schedule = self._load_full_season_schedule(target_seasons)
                    spot_flags = self._derive_spot_flags(
                        games_df, full_schedule, as_of_datetime
                    )
                except _SCHEDULE_LOAD_ERRORS as e:
                    logger.warning(
                        "Situational spot-flag derivation skipped; "
                        "emitting neutral flags",
                        error=str(e),
                    )
                    spot_flags = {}
                    full_schedule = None

            # Rest-days source (WR-02): in the target-week (--current-week) build,
            # games_df is filtered to the target week (above), so a team's prior
            # game is invisible and rest collapses to the 7.0 first-game default
            # -- taking off_bye / short_rest / both_short_rest with it. Derive
            # rest from the FULL season schedule in that path (the same
            # independently-loaded schedule the spot flags use), so a team's prior
            # game is visible regardless of build mode. The full-batch build
            # (target=None) keeps games_df as the source so its already-built gold
            # numbers are byte-unchanged.
            rest_source_df = games_df
            if (
                target_season
                and target_week
                and full_schedule is not None
                and len(full_schedule) > 0
            ):
                rest_source_df = full_schedule

            # WR-08: hoist the ET wall-clock map for the rest source OUT of the
            # per-game loop. calculate_rest_days used to rebuild it on every call
            # -- twice per game -- costing one pd.Timestamp construction per row
            # of the source frame each time. Sliced per game below by index, which
            # is cheap; the map itself is paid once.
            rest_kickoffs = (
                rest_source_df["kickoff_et"].map(kickoff_wall_clock_et)
                if len(rest_source_df) > 0
                else None
            )

            contextual_features = []

            for _, game in games_df.iterrows():
                game_id = game["game_id"]
                home_team = game["home_team"]
                away_team = game["away_team"]
                # D33.1-06: EVERY game of EVERY season resolves by its own
                # stadium_id. The historical half of gold DOES move as a result,
                # deliberately, and that movement is a declared cause family in
                # Plan 33.1-07's rung rather than an untracked side effect.
                venue_id = self._resolve_venue_id_for_game(game)
                # WR-06 / N-02: resolve the kickoff to its ET WALL CLOCK exactly
                # once, here, so detect_short_week, the travel metrics and the
                # rest-days call all read the same instant through the one
                # documented accessor. Reading the raw cell makes the weekday
                # family (thursday_game / monday_game / saturday_game /
                # short_week / game_day_of_week) depend on which games copy
                # load_dataframe happened to resolve: it is correct on the
                # ET-typed DuckDB table and WRONG for 718 of 6,499 rows on the
                # UTC-typed parquet.
                #
                # THE FACTS AT THE LOCK (Plan 33.2-10): the kickoff and the week come from
                # facts_for_game, the same accessor the venue above resolved through, so a
                # post-lock emergency move is built with the facts before the move. The
                # accessor returns the ET wall clock exactly as kickoff_wall_clock_et does.
                # `week` stays the silver value: it is a merge key, not a feature.
                facts = self.facts_for_game(game)
                kickoff_dt = facts.kickoff_et
                season = game["season"]
                week = game["week"]
                week_at_lock = facts.week

                game_features: dict[str, object] = {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                    "home_team": home_team,
                    "away_team": away_team,
                }

                game_features.update(
                    {
                        "is_home_game": 1.0,
                        "is_away_game": 0.0,
                    }
                )

                travel_metrics = self.calculate_travel_metrics(
                    away_team, home_team, venue_id, kickoff_dt
                )
                for key, value in travel_metrics.items():
                    game_features[f"away_{key}"] = value

                short_week_features = self.detect_short_week(kickoff_dt, season, week)
                game_features.update(short_week_features)

                venue_features = self.encode_venue_features(venue_id)
                game_features.update(venue_features)

                # Rest days: use only games with kickoff before as_of_datetime.
                # Source from rest_source_df (the full schedule in the
                # target-week build, games_df in the full-batch build) so a
                # team's prior game is always visible (WR-02).
                kickoff_series = pd.to_datetime(rest_source_df["kickoff_et"])
                cutoff_ts = pd.Timestamp(as_of_datetime)
                if kickoff_series.dt.tz is not None and cutoff_ts.tz is None:
                    cutoff_ts = cutoff_ts.tz_localize(kickoff_series.dt.tz)
                prior_games = rest_source_df[kickoff_series < cutoff_ts]
                prior_kickoffs = (
                    None
                    if rest_kickoffs is None
                    else rest_kickoffs.loc[prior_games.index]
                )
                home_rest = self.calculate_rest_days(
                    home_team, kickoff_dt, prior_games, kickoffs=prior_kickoffs
                )
                away_rest = self.calculate_rest_days(
                    away_team, kickoff_dt, prior_games, kickoffs=prior_kickoffs
                )

                game_features.update(
                    {
                        "home_rest_days": home_rest,
                        "away_rest_days": away_rest,
                        "rest_advantage": home_rest - away_rest,
                        "both_short_rest": (
                            1.0 if (home_rest <= 4 and away_rest <= 4) else 0.0
                        ),
                        "home_short_rest": 1.0 if home_rest <= 4 else 0.0,
                        "away_short_rest": 1.0 if away_rest <= 4 else 0.0,
                    }
                )

                # SIG-03: off_bye is the genuinely-new add, derived from the
                # existing rest-days output (a bye gives ~13-14 days). The
                # existing rest/travel/short-week/bye/divisional features are
                # reused, not re-derived (D-15).
                game_features.update(
                    {
                        "home_off_bye": (
                            1.0 if home_rest >= OFF_BYE_REST_DAYS else 0.0
                        ),
                        "away_off_bye": (
                            1.0 if away_rest >= OFF_BYE_REST_DAYS else 0.0
                        ),
                    }
                )

                # SIG-03: look-ahead (trap) / letdown spots. These are weak /
                # largely priced-in and not a standing bet angle (SC3); a drop
                # in the Plan 28-07 lift screen is an expected outcome. Sourced
                # from the full-season schedule + raw silver Elo (D-16).
                game_spots = spot_flags.get(
                    game_id,
                    {
                        "home_look_ahead_spot": 0.0,
                        "away_look_ahead_spot": 0.0,
                        "home_letdown_spot": 0.0,
                        "away_letdown_spot": 0.0,
                    },
                )
                game_features.update(game_spots)

                # FEAT-17: Season-week position features
                season_progress = float(week_at_lock) / 18.0
                late_season = 1.0 if week_at_lock >= 14 else 0.0

                # FEAT-18: Surface type mismatch (away team perspective)
                surface_mismatch = self._compute_surface_mismatch(away_team, venue_id)

                # FEAT-19: Divisional game indicator
                is_div = 1.0 if is_divisional_game(home_team, away_team) else 0.0

                game_features.update(
                    {
                        "season_progress": season_progress,
                        "late_season": late_season,
                        "surface_mismatch": surface_mismatch,
                        "is_divisional": is_div,
                    }
                )

                contextual_features.append(game_features)

            features_df = pd.DataFrame(contextual_features)

            logger.info(
                "Built contextual features (Protocol)",
                features_count=len(features_df),
                feature_columns=len(features_df.columns),
            )

            return features_df

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to build contextual features (Protocol)",
                error=str(e),
            )
            raise

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get contextual features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        try:
            features_df = load_dataframe("contextual_features", layer="silver")

            game_features = features_df[features_df["game_id"] == game_id]

            if len(game_features) == 0:
                logger.warning(
                    "No contextual features found for game",
                    game_id=game_id,
                )
                return {}

            exclude_cols = {
                "game_id",
                "season",
                "week",
                "home_team",
                "away_team",
            }
            game_row = game_features.iloc[0]
            return {
                col: float(game_row[col])
                for col in game_features.columns
                if col not in exclude_cols
            }

        except (ValueError, KeyError, TypeError, AttributeError) as e:
            logger.error(
                "Failed to get contextual features for game",
                game_id=game_id,
                error=str(e),
            )
            return {}
