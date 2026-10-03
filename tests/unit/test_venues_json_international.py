"""The eight international venue records, and the feed-roof disagreement RECORDER.

Phase 33, Plan 33-06 Task 2 (COLD-09, R11, D33-04/D33-16, T-33-26/28/31).

WHAT THIS MODULE IS FOR
-----------------------
R11's acceptance is stated "value-by-value", and that phrasing is doing real work: a
wrong coordinate produces a test that PASSES against a wrong constant, which is worse
than a failing one. Nothing in this repository can settle what Wembley's elevation is.
So the values were researched, tabled, and RATIFIED by the owner at Plan 33-06's
blocking-human checkpoint on 2026-09-12, and committed ONCE to
``tests.phase33_state.INTERNATIONAL_VENUE_FACTS``. This module asserts that
``data/venues.json`` says the same thing that record does -- it does not re-decide the
values, it checks the file against the ratified table.

THE FEED IS WRONG ON THREE ROOFS AND THAT IS RECORDED, NOT INHERITED (D33-16)
-----------------------------------------------------------------------------
MEL00, PAR00 and MUN01 all carry ``roof == 'dome'`` in the 2026 capture and all three
are open-air. ``_NFLVERSE_ROOF_MAP`` sends ``dome`` to ``indoor`` and
``_is_outdoor_game`` sends ``indoor`` to a SKIP, so inheriting the feed would zero the
weather on three genuinely outdoor games with no error raised. ``data/venues.json`` is
AUTHORITATIVE on roof for all eight, and the disagreement set is recomputed here from
the capture and asserted against the committed tuple -- so a future feed correction
surfaces as a DIFF rather than as a silent flip.

THIS MODULE READS. IT NEVER WRITES. ``data/venues.json`` is a git-tracked SOURCE file
whose suffix is in ``tests.data_boundary.TRACKED_SUFFIXES``, so a test that wrote it
would be a COLD-05 boundary crossing under the guard Plan 33-01 armed. The eight
records were added as a SOURCE EDIT, outside pytest.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from tests import phase33_state
from tests.fixtures import season_2026

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

EXPECTED_EXISTING_RECORDS = 30

# ---------------------------------------------------------------------------
# THE TWO 2025 INTERNATIONAL VENUES (Plan 33.2-09 Task 1, SPEC R8 venue half).
#
# Seven 2025 games were played outside the United States, and the feed records every one
# of them at the US home team's own stadium (33.2-RESEARCH.md section 8.3). Five of the
# seven real venues already had records; Croke Park (Dublin) and the Olympiastadion
# (Berlin) did not, so no correction could point at them. These are the two records
# added, stated field for field so the file is checked against a table rather than
# against itself.
#
# IDs FOLLOW THE nflverse CONVENTION the other international records use (city code plus
# a two-digit serial: LON00, MUN01, SAO00). The feed has never carried a code for either
# venue -- it records both games at PIT00 and IND00 -- so there is no feed value to derive
# from, and the two ids are ASSIGNED here, once.
#
# `country` IS WRITTEN IN THE FILE'S OWN VOCABULARY. The field holds full country names,
# and Germany already appears three times under that spelling; an ISO code would put a
# second spelling of one country into the file (the D30-02 two-answers failure).
# ---------------------------------------------------------------------------

ADDED_2025_VENUE_RECORDS: dict[str, dict[str, object]] = {
    "DUB00": {
        "venue_id": "croke_park",
        "stadium_id": "DUB00",
        "venue_name": "Croke Park",
        "city": "Dublin",
        "state": "",
        "country": "Ireland",
        "latitude": 53.3608,
        "longitude": -6.2511,
        "elevation_ft": 10,
        "roof_type": "outdoor",
        "surface": "Grass",
        "capacity": 82300,
        "climate_zone": "oceanic",
        "timezone": "Europe/Dublin",
        "home_teams": [],
    },
    "BER00": {
        "venue_id": "olympiastadion_berlin",
        "stadium_id": "BER00",
        "venue_name": "Olympiastadion",
        "city": "Berlin",
        "state": "",
        "country": "Germany",
        "latitude": 52.5147,
        "longitude": 13.2394,
        "elevation_ft": 308,
        "roof_type": "outdoor",
        "surface": "Grass",
        "capacity": 74475,
        "climate_zone": "oceanic",
        "timezone": "Europe/Berlin",
        "home_teams": [],
    },
}

ADDED_2025_STADIUM_IDS: tuple[str, ...] = tuple(ADDED_2025_VENUE_RECORDS)

# Where each non-identity value came from, fetched 2026-09-21. Elevation uses the SAME
# instrument the historical records used (tests.phase33_state.HISTORICAL_VENUE_FIELD_
# SOURCES cites open-meteo /v1/elevation for SAO00), checked against two existing records
# first: Allianz Arena (491 m -> 1611 ft, the file says 1611) and Wembley (47 m -> 154 ft,
# the file says 154).
ADDED_2025_VENUE_FIELD_SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "DUB00",
        "latitude/longitude",
        "Wikidata Q478225 P625 (53.360833, -6.251111), rounded to 4 dp",
    ),
    (
        "DUB00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (53.3608, -6.2511) -> 3.0 m -> 10 ft",
    ),
    ("DUB00", "capacity", "Wikidata Q478225 P1083 = 82300"),
    ("DUB00", "surface", "en.wikipedia Croke_Park infobox: natural soil/grass pitch"),
    ("DUB00", "climate_zone", "Dublin is Koppen Cfb (temperate oceanic)"),
    ("DUB00", "timezone", "IANA Europe/Dublin"),
    (
        "BER00",
        "latitude/longitude",
        "Wikidata Q151374 P625 (52.514722, 13.239444), rounded to 4 dp",
    ),
    (
        "BER00",
        "elevation_ft",
        "open-meteo /v1/elevation @ (52.5147, 13.2394) -> 94.0 m -> 308 ft",
    ),
    ("BER00", "capacity", "Wikidata Q151374 P1083 = 74475"),
    ("BER00", "surface", "en.wikipedia Olympiastadion_(Berlin) infobox: Grass"),
    (
        "BER00",
        "climate_zone",
        "Berlin is Koppen Cfb (temperate oceanic), bordering Dfb",
    ),
    ("BER00", "timezone", "IANA Europe/Berlin"),
)

# 38 -> 60 (Plan 33.1-01, the 22 historical records) -> 62 (this plan).
# The Plan-33.1-01 count plus the two added above. DERIVED, so the three modules that
# assert the record count (this one, test_venues_json_historical and
# test_venue_feature_fields_are_valid) import ONE answer rather than each carrying 62.
EXPECTED_TOTAL_RECORDS = phase33_state.VENUE_RECORD_COUNT_AFTER + len(
    ADDED_2025_VENUE_RECORDS
)

# `country` over the 60 records BEFORE this plan, measured 2026-09-16 and again
# 2026-09-21: USA 47, United Kingdom 3, Germany 3, Brazil 2, and one each of Australia,
# France, Spain, Mexico and Canada.
PRE_EDIT_COUNTRY_VOCABULARY: frozenset[str] = frozenset(
    {
        "USA",
        "United Kingdom",
        "Germany",
        "Brazil",
        "Australia",
        "France",
        "Spain",
        "Mexico",
        "Canada",
    }
)

# The ONE value this plan adds to that vocabulary. The Olympiastadion joins the three
# German venues under their spelling, so Ireland is the only new string.
PERMITTED_COUNTRY_VOCABULARY: frozenset[str] = PRE_EDIT_COUNTRY_VOCABULARY | {"Ireland"}

# The values that mean "in the United States". Every record carries `country`, so this
# field IS the is-US marker; no second marker exists and none may be added.
US_COUNTRY_VALUES: frozenset[str] = frozenset({"US", "USA", "UNITED STATES"})

# The non-US membership, asserted as a SET so a venue silently joining or leaving it
# fails. Plan 33.2-12's weather backfill decides "outside MOS coverage" from this field.
EXPECTED_NON_US_STADIUM_IDS: frozenset[str] = frozenset(
    {
        "BUF01",
        "FRA00",
        "GER00",
        "LON00",
        "LON01",
        "LON02",
        "MAD01",
        "MEL00",
        "MEX00",
        "MUN01",
        "PAR00",
        "RIO00",
        "SAO00",
        "DUB00",
        "BER00",
    }
)

# sha256 of the 60 PRE-EDIT records, canonicalised (sorted by stadium_id, keys sorted,
# compact separators, ASCII), taken 2026-09-21 on commit 65cf31c before either record was
# added. "No existing record changed" is checked against this digest.
#
# SUPERSEDED BY PLAN 33.2-20's THREE SURFACE CORRECTIONS, and kept here as the value it
# WAS rather than deleted: the digest is the evidence that Plan 33.2-09 revised no
# existing record, and that claim is still true of Plan 33.2-09. What changed the bytes
# is a LATER, separately-recorded correction of three cells the owner had ratified
# (MEL00, RIO00 and PAR00's surface, `phase33_state.P332_20_VENUE_SURFACE_CORRECTIONS`),
# so the live assertion below compares against the post-correction digest and a second
# assertion proves the delta is exactly those three records.
PRE_EDIT_RECORDS_SHA256 = (
    "d836816d0349ae6adc84efa2507e2eb2ce082356982863f54169cbf1878b3a76"
)

# The live expectation: the 60 pre-2025-addition records as they stand AFTER Plan
# 33.2-20's three surface corrections (measured 2026-09-22) AND the Phase-33 review's
# MUN01 correction (WR-09, measured 2026-10-03).
CURRENT_PRE_EDIT_RECORDS_SHA256 = (
    phase33_state.REVIEW33_VENUE_RECORDS_SHA256_AFTER_MUN01_FIX
)

# ``stadium_id -> (field, was, now)``: every ratified cell a later record corrected, read
# from the records in `tests.phase33_state` rather than restated here -- Plan 33.2-20's
# three, then the Phase-33 review's MUN01 surface.
SURFACE_CORRECTIONS: dict[str, tuple[str, str, str]] = {
    **phase33_state.P332_20_VENUE_SURFACE_CORRECTIONS,
    **phase33_state.REVIEW33_VENUE_SURFACE_CORRECTIONS,
}

# One cited source per corrected cell, across both correction records.
SURFACE_CORRECTION_SOURCES: tuple[tuple[str, str, str], ...] = (
    *phase33_state.P332_20_VENUE_SURFACE_SOURCES,
    *phase33_state.REVIEW33_VENUE_SURFACE_SOURCES,
)

PRE_EDIT_RECORD_COUNT = 60

# The field order INTERNATIONAL_VENUE_FACTS is stated in, named here so the unpacking
# below is checkable rather than positional folklore.
FACT_FIELDS: tuple[str, ...] = (
    "stadium_id",
    "venue_id",
    "venue_name",
    "city",
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

# Values that would mean a cell was DEFAULTED rather than entered. `outdoor` is not in
# this set -- it is a legitimate ratified roof for six of the eight -- which is why the
# roof check below compares against the RATIFIED value rather than against a sentinel.
DEFAULT_SENTINELS: frozenset[object] = frozenset({None, "", 0, "unknown"})


def _load_venue_records() -> list[dict[str, object]]:
    return json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]


def _facts_by_stadium_id() -> dict[str, dict[str, object]]:
    """The ratified table, WITH the corrections a later plan recorded beside it.

    ``INTERNATIONAL_VENUE_FACTS`` is an append-once slot and is never edited, so a cell
    the owner ratified on 2026-09-12 that a later plan corrected is superseded HERE,
    from the correction record, rather than rewritten at the source. Both readings stay
    on the record: the ratified value, and what replaced it with its citation.
    """
    facts = {
        row[0]: dict(zip(FACT_FIELDS, row, strict=True))
        for row in phase33_state.INTERNATIONAL_VENUE_FACTS
    }
    for stadium_id, (field, was, now) in SURFACE_CORRECTIONS.items():
        ratified = facts[stadium_id]
        assert ratified[field] == was, (
            f"{stadium_id}.{field} reads {ratified[field]!r} in the ratified table, but "
            f"its correction record says it was {was!r}. The correction "
            "names a cell that no longer says what it corrected; re-measure rather than "
            "adjusting either side."
        )
        ratified[field] = now
    return facts


def _records_by_stadium_id() -> dict[str, dict[str, object]]:
    return {
        str(record["stadium_id"]): record
        for record in _load_venue_records()
        if record.get("stadium_id")
    }


def _require_capture() -> None:
    try:
        season_2026.load_captured_schedule()
    except season_2026.CapturedScheduleUnavailableError as exc:
        pytest.skip(str(exc))


class TestEveryRecordCarriesAStadiumId:
    """`venue_id` is a slug and `stadium_id` is an nflverse code; they are not the same."""

    def test_the_file_holds_every_ratified_record(self) -> None:
        """Renamed from `test_the_file_holds_thirty_eight_records` by Plan 33.1-01.

        The count moved to 60 and a test whose NAME still said thirty-eight would be a
        quiet lie in the one place a reader looks first.
        """
        records = _load_venue_records()
        assert len(records) == EXPECTED_TOTAL_RECORDS, (
            f"data/venues.json holds {len(records)} records, expected "
            f"{EXPECTED_EXISTING_RECORDS} existing US venues, the eight international "
            "ones ratified for 2026, and the 22 historical ones ratified by Plan "
            "33.1-01."
        )

    def test_every_record_carries_a_non_empty_stadium_id(self) -> None:
        missing = [
            record["venue_id"]
            for record in _load_venue_records()
            if not record.get("stadium_id")
        ]
        assert not missing, (
            f"{len(missing)} venue record(s) carry no stadium_id: {missing!r}. "
            "stadium_id-keyed routing cannot resolve a venue that does not declare "
            "its code, and a hand-typed code is the class of unsourced value the "
            "Task-1 checkpoint exists to prevent -- derive it from the captured "
            "schedule instead."
        )

    def test_stadium_ids_are_unique_across_the_file(self) -> None:
        ids = [
            record["stadium_id"]
            for record in _load_venue_records()
            if record.get("stadium_id")
        ]
        duplicates = sorted({code for code in ids if ids.count(code) > 1})
        assert not duplicates, (
            f"stadium_id(s) {duplicates!r} appear on more than one venue record. The "
            "lookup returns the FIRST match, so a duplicate silently decides which "
            "stadium a neutral-site game resolves to."
        )

    def test_stadium_id_is_not_a_copy_of_venue_id(self) -> None:
        """A slug is not a code; a record that reused one would look wired and not be."""
        reused = [
            record["venue_id"]
            for record in _load_venue_records()
            if record.get("stadium_id") == record.get("venue_id")
        ]
        assert not reused, (
            f"{reused!r} carry stadium_id == venue_id. venue_ids are slugs such as "
            "'highmark_stadium' and nflverse stadium_ids are codes such as 'BUF00'."
        )


class TestTheThirtyExistingCodesWereDerivedNotTyped:
    """Every CURRENTLY-IN-SERVICE venue's code must reproduce from the 2026 schedule."""

    def test_each_existing_record_reproduces_its_code_from_the_feed(self) -> None:
        """Join `home_teams` to the feed's NON-neutral home games and read the code.

        A venue whose derived code is not unique, or which resolves to none, is a
        FINDING to report by name -- never a value to guess. A team that relocated
        between the venue record's era and 2026 is exactly the case this surfaces.

        THREE SETS ARE EXCLUDED, AND FOR THE SAME REASON. (The third, the two 2025
        international venues Plan 33.2-09 added, host only neutral-site games too.) The eight international venues
        host only neutral-site games, so a non-neutral join cannot reach them. The 22
        HISTORICAL venues Plan 33.1-01 added (2026-09-12) carry `home_teams == []` by
        ratified design -- a demolished stadium must never win a home-team lookup and
        shadow its successor -- and their franchises play somewhere else now, so
        joining them to the 2026 feed asks a question with no answer. 33.1-01-PLAN.md
        records the same constraint from the other direction: the historical module is
        explicitly forbidden from copying this class, because "that join is meaningless
        for a demolished stadium". Their codes are derived instead in
        tests/unit/test_venues_json_historical.py::TestTheRatifiedTableIsTheSource,
        against the pinned 2002-2025 schedules, which is the feed that actually carries
        them.
        """
        _require_capture()
        feed = season_2026.load_captured_schedule()
        home_games = feed[feed["location"] != "Neutral"]
        not_in_service = (
            {row[0] for row in phase33_state.INTERNATIONAL_VENUE_FACTS}
            | set(phase33_state.HISTORICAL_STADIUM_IDS)
            | set(ADDED_2025_STADIUM_IDS)
        )

        unresolved: list[str] = []
        mismatched: list[str] = []
        for record in _load_venue_records():
            if record.get("stadium_id") in not_in_service:
                continue
            rows = home_games[home_games["home_team"].isin(record["home_teams"])]
            derived = sorted(set(rows["stadium_id"].dropna()))
            if len(derived) != 1:
                unresolved.append(f"{record['venue_id']}: derived {derived!r}")
                continue
            if derived[0] != record.get("stadium_id"):
                mismatched.append(
                    f"{record['venue_id']}: file says {record.get('stadium_id')!r}, "
                    f"the feed says {derived[0]!r}"
                )

        assert not unresolved, (
            "the stadium_id join did not resolve uniquely for: "
            + "; ".join(unresolved)
            + ". Report it by name -- do not type a code in."
        )
        assert not mismatched, (
            "the file's stadium_id disagrees with the captured feed for: "
            + "; ".join(mismatched)
        )


class TestTheEightRatifiedRecords:
    """Value-by-value, against the owner-ratified table -- not against the feed."""

    def test_all_eight_international_records_are_present(self) -> None:
        present = set(_records_by_stadium_id())
        expected = set(phase33_state.INTERNATIONAL_STADIUM_IDS)
        assert expected <= present, (
            f"international venue(s) {sorted(expected - present)!r} are absent from "
            "data/venues.json. Every one of the eight 2026 neutral-site games needs "
            "its own record; there is no historical record to reuse (MUN01 is not "
            "GER00 and RIO00 is not SAO00)."
        )

    @pytest.mark.parametrize(
        "stadium_id", list(phase33_state.INTERNATIONAL_STADIUM_IDS)
    )
    def test_each_ratified_cell_matches_the_file(self, stadium_id: str) -> None:
        record = _records_by_stadium_id().get(stadium_id)
        assert record is not None, f"{stadium_id} is not in data/venues.json"
        facts = _facts_by_stadium_id()[stadium_id]

        for field in FACT_FIELDS:
            assert record.get(field) == facts[field], (
                f"{stadium_id}.{field} is {record.get(field)!r} in data/venues.json "
                f"but the owner ratified {facts[field]!r} on 2026-09-12. The ratified "
                "table is the source; the file follows it."
            )

    @pytest.mark.parametrize(
        "stadium_id", list(phase33_state.INTERNATIONAL_STADIUM_IDS)
    )
    def test_no_ratified_geographic_cell_is_a_default(self, stadium_id: str) -> None:
        """The five R11 cells are ENTERED, never defaulted and never inherited."""
        record = _records_by_stadium_id().get(stadium_id)
        assert record is not None, f"{stadium_id} is not in data/venues.json"

        for field in ("latitude", "longitude", "elevation_ft", "timezone", "roof_type"):
            value = record.get(field)
            assert value not in DEFAULT_SENTINELS, (
                f"{stadium_id}.{field} reads {value!r}, which is a default sentinel. "
                "R11 requires these five entered EXPLICITLY for all eight."
            )

    @pytest.mark.parametrize(
        "stadium_id", list(phase33_state.INTERNATIONAL_STADIUM_IDS)
    )
    def test_the_eight_declare_no_home_team(self, stadium_id: str) -> None:
        """No NFL team calls these home, and an empty list is what makes the

        home-team resolvers decline them rather than claim them.
        """
        record = _records_by_stadium_id().get(stadium_id)
        assert record is not None, f"{stadium_id} is not in data/venues.json"
        assert record.get("home_teams") == [], (
            f"{stadium_id} declares home_teams {record.get('home_teams')!r}. A "
            "non-empty list would make _build_team_venue_mapping route that team's "
            "ordinary home games to an international stadium."
        )

    def test_the_eight_venue_names_match_the_feed_string_byte_for_byte(self) -> None:
        """`_load_venue_lookup` keys on the LOWERCASED NAME -- a mismatch is silent.

        Every feed string is PLAIN ASCII with no diacritic. A "corrected" spelling
        here would fall through to the nflverse roof map, re-introducing the three
        false domes on the ingest_games resolver while the other two are correct.
        Partially repaired, silently, is the worst available state.
        """
        _require_capture()
        neutral = season_2026.neutral_site_games()
        feed_names = dict(zip(neutral["stadium_id"], neutral["stadium"], strict=True))
        records = _records_by_stadium_id()

        for stadium_id, feed_name in feed_names.items():
            record = records.get(stadium_id)
            assert record is not None, f"{stadium_id} is not in data/venues.json"
            assert record["venue_name"] == feed_name, (
                f"{stadium_id}: data/venues.json says {record['venue_name']!r}, the "
                f"feed says {feed_name!r}. These must be byte-equal."
            )
            assert feed_name.isascii(), (
                f"{feed_name!r} is not plain ASCII, so the recorded expectation that "
                "the feed carries no diacritics has moved."
            )


class TestTheFeedRoofDisagreementRecorder:
    """D33-16: record the disagreements by name so a feed correction is a diff."""

    def test_the_measured_feed_roof_values_still_hold(self) -> None:
        _require_capture()
        neutral = season_2026.neutral_site_games()
        measured = {
            str(row.stadium_id): (None if row.roof is None else str(row.roof))
            for row in neutral.itertuples()
        }
        recorded = dict(phase33_state.FEED_ROOF_VALUES_2026)
        assert measured == recorded, (
            f"the capture's roof values moved: measured {measured!r}, recorded "
            f"{recorded!r}. Re-measure and append a new slot; do not edit the old one."
        )

    def test_the_live_disagreement_set_equals_the_committed_one(self) -> None:
        """Recomputed from the capture, compared to the committed tuple.

        A disagreement is a feed value that MAPS TO A DIFFERENT project roof than the
        ratified one. An ABSENT feed value is not a disagreement -- it contradicts
        nothing -- and folding the two together would leave this recorder unable to
        tell a feed correction from a feed backfill.
        """
        _require_capture()
        from scripts.ingest_games import _NFLVERSE_ROOF_MAP

        facts = _facts_by_stadium_id()
        live: list[str] = []
        for code, feed_roof in phase33_state.FEED_ROOF_VALUES_2026:
            if feed_roof is None:
                continue
            mapped = _NFLVERSE_ROOF_MAP.get(feed_roof.lower().strip())
            if mapped is not None and mapped != facts[code]["roof_type"]:
                live.append(code)

        assert tuple(live) == phase33_state.FEED_ROOF_DISAGREEMENTS, (
            f"the live disagreement set is {tuple(live)!r} but the committed record "
            f"is {phase33_state.FEED_ROOF_DISAGREEMENTS!r}. If the feed was corrected "
            "that is good news -- append a new measurement and say so; it is never a "
            "reason to loosen this assertion."
        )

    def test_the_three_disagreeing_venues_are_written_open_air(self) -> None:
        """The whole point: an inherited `dome` would zero the weather on all three."""
        records = _records_by_stadium_id()
        for code in phase33_state.FEED_ROOF_DISAGREEMENTS:
            record = records.get(code)
            assert record is not None, f"{code} is not in data/venues.json"
            assert record["roof_type"] != "indoor", (
                f"{code} is written as {record['roof_type']!r}. The feed says 'dome', "
                "which maps to 'indoor', which _is_outdoor_game turns into a weather "
                "SKIP -- on a genuinely open-air game, with no error raised."
            )


class TestTheRatifiedValuesStayInsideTheFeatureEncoding:
    """Two encoder thresholds the eight sit against, asserted rather than assumed."""

    def test_only_mexico_city_trips_the_high_altitude_flag(self) -> None:
        """`venue_high_altitude` fires at elevation_ft >= 3000 (contextual.py)."""
        facts = _facts_by_stadium_id()
        tripping = sorted(
            code for code, row in facts.items() if int(row["elevation_ft"]) >= 3000
        )
        assert tripping == ["MEX00"], (
            f"venues {tripping!r} trip venue_high_altitude. Only MEX00 (7365 ft) "
            "should; MAD01 at 2349 ft has 651 ft of margin, and a change here means "
            "a ratified elevation moved."
        )

    def test_the_new_climate_zone_token_is_inert(self) -> None:
        """`subtropical_highland` feeds neither cold_climate nor warm_climate.

        `climate_zone` reaches exactly two encoded flags: cold_climate
        (humid_continental) and warm_climate (humid_subtropical | tropical | desert).
        Mexico City's new token yields false/false -- inert, exactly as `oceanic` and
        `mediterranean` already are. Recorded so a later reader does not assume a new
        token silently created a feature.
        """
        cold = {"humid_continental"}
        warm = {"humid_subtropical", "tropical", "desert"}
        assert "subtropical_highland" not in cold | warm
        facts = _facts_by_stadium_id()
        assert facts["MEX00"]["climate_zone"] == "subtropical_highland"


def _canonical_digest(records: list[dict[str, object]]) -> str:
    canon = json.dumps(
        sorted(records, key=lambda record: str(record["stadium_id"])),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(canon.encode("ascii")).hexdigest()


def _country_values_outside_vocabulary(records: list[dict[str, object]]) -> set[str]:
    return {str(record["country"]) for record in records} - PERMITTED_COUNTRY_VOCABULARY


def _pre_existing_records() -> list[dict[str, object]]:
    return [
        record
        for record in _load_venue_records()
        if record["stadium_id"] not in ADDED_2025_STADIUM_IDS
    ]


class TestTheTwo2025InternationalVenues:
    """Croke Park and the Olympiastadion, added by Plan 33.2-09 Task 1."""

    def test_the_file_is_a_dict_with_one_venues_key(self) -> None:
        """Every reader must unwrap `venues`; iterating the top level sees ONE key."""
        document = json.loads(VENUES_PATH.read_text(encoding="utf-8"))
        assert isinstance(document, dict)
        assert list(document) == ["venues"]

    @pytest.mark.parametrize("stadium_id", ADDED_2025_STADIUM_IDS)
    def test_each_added_record_matches_its_table_row(self, stadium_id: str) -> None:
        record = _records_by_stadium_id().get(stadium_id)
        assert record is not None, f"{stadium_id} is not in data/venues.json"
        assert record == ADDED_2025_VENUE_RECORDS[stadium_id], (
            f"{stadium_id} in data/venues.json does not equal the table row "
            "ADDED_2025_VENUE_RECORDS states. The table is the source; the file follows."
        )

    @pytest.mark.parametrize("stadium_id", ADDED_2025_STADIUM_IDS)
    def test_each_added_record_carries_exactly_the_shared_field_set(
        self, stadium_id: str
    ) -> None:
        """No extra field and no missing one -- a missing field is what a default hides."""
        shared = {frozenset(record) for record in _pre_existing_records()}
        assert len(shared) == 1, (
            f"the pre-existing records disagree on fields: {shared}"
        )
        record = _records_by_stadium_id()[stadium_id]
        assert frozenset(record) == next(iter(shared))
        assert len(record) == 15
        assert {"stadium_id", "venue_id", "roof_type"} <= set(record)
        assert "roof" not in record

    @pytest.mark.parametrize("stadium_id", ADDED_2025_STADIUM_IDS)
    def test_each_added_record_is_outdoor_and_outside_the_united_states(
        self, stadium_id: str
    ) -> None:
        record = _records_by_stadium_id()[stadium_id]
        assert record["roof_type"] == "outdoor"
        assert str(record["country"]).upper() not in US_COUNTRY_VALUES
        assert record["home_teams"] == [], (
            "a non-empty home_teams would route that team's ordinary home games here"
        )

    def test_every_added_record_has_recorded_sources(self) -> None:
        covered = {stadium_id for stadium_id, _, _ in ADDED_2025_VENUE_FIELD_SOURCES}
        assert covered == set(ADDED_2025_STADIUM_IDS)
        assert all(source for _, _, source in ADDED_2025_VENUE_FIELD_SOURCES)

    def test_the_file_holds_the_expected_number_of_records(self) -> None:
        assert len(_load_venue_records()) == EXPECTED_TOTAL_RECORDS == 62

    def test_no_pre_existing_record_changed_except_the_recorded_corrections(
        self,
    ) -> None:
        """Was: ``test_no_pre_existing_record_changed``, against the PRE-EDIT digest.

        Plan 33.2-09 still adds two records and revises none. What moved the digest is
        Plan 33.2-20's three SURFACE corrections, each recorded with its source in
        ``phase33_state.P332_20_VENUE_SURFACE_SOURCES``. The assertion is not relaxed:
        it now pins the post-correction digest, and the companion test below proves the
        delta is exactly those three cells and nothing else.
        """
        pre_existing = _pre_existing_records()
        assert len(pre_existing) == PRE_EDIT_RECORD_COUNT
        assert _canonical_digest(pre_existing) == CURRENT_PRE_EDIT_RECORDS_SHA256, (
            "a record that existed before Plan 33.2-09 changed, beyond the recorded "
            "surface corrections (Plan 33.2-20's three and the review's MUN01). Static "
            "venue data is time-invariant "
            "(D33.2-04); a value that moves needs its own cited record."
        )

    def test_the_digest_delta_is_exactly_the_three_corrected_cells(self) -> None:
        """Non-vacuity: rewinding the corrections must reproduce the PRE-EDIT digest.

        This is what makes the re-pin above a RECORD rather than a rubber stamp. If any
        other byte of any pre-existing record had moved, undoing the three named cells
        would not land back on the 2026-09-21 digest.
        """
        rewound = []
        for record in _pre_existing_records():
            code = str(record["stadium_id"])
            if code in SURFACE_CORRECTIONS:
                field, was, now = SURFACE_CORRECTIONS[code]
                assert record[field] == now
                record = {**record, field: was}
            rewound.append(record)
        assert _canonical_digest(rewound) == PRE_EDIT_RECORDS_SHA256, (
            "undoing the recorded surface corrections does NOT reproduce the "
            "2026-09-21 digest, so something else in the 60 pre-existing records moved "
            "as well. Find it and record it; do not re-pin."
        )

    def test_every_corrected_cell_carries_a_cited_source(self) -> None:
        cited = {
            (code, field)
            for code, field, source in SURFACE_CORRECTION_SOURCES
            if source
        }
        expected = {
            (code, field) for code, (field, _, _) in SURFACE_CORRECTIONS.items()
        }
        assert cited == expected, (
            f"corrected cells without a cited source: {sorted(expected - cited)!r}; "
            f"sources for cells that were not corrected: {sorted(cited - expected)!r}. "
            "A venue value without a source is exactly the unsourced cell the Plan "
            "33-06 checkpoint exists to prevent."
        )
        assert phase33_state.P332_20_VENUE_SURFACE_RESEARCHED_BY
        assert phase33_state.REVIEW33_VENUE_SURFACE_RESEARCHED_BY

    def test_every_corrected_surface_is_a_classified_grass_spelling(self) -> None:
        """The correction may not introduce a spelling the classifier cannot read."""
        from features.contextual import SURFACE_CLASS_BY_SPELLING

        for code, (field, _was, now) in SURFACE_CORRECTIONS.items():
            assert field == "surface"
            assert SURFACE_CLASS_BY_SPELLING.get(now) == "grass", (
                f"{code}'s corrected surface {now!r} is not classified as grass by "
                "features.contextual.SURFACE_CLASS_BY_SPELLING. An unlisted spelling "
                "raises UnknownSurfaceError at build time."
            )

    def test_the_added_ids_are_new_codes(self) -> None:
        pre_existing = {record["stadium_id"] for record in _pre_existing_records()}
        assert not set(ADDED_2025_STADIUM_IDS) & pre_existing


class TestTheCountryFieldIsOneVocabulary:
    """The value guard the key-set scan cannot give: one spelling per country."""

    def test_every_country_value_is_in_the_permitted_vocabulary(self) -> None:
        outside = _country_values_outside_vocabulary(_load_venue_records())
        assert not outside, (
            f"country value(s) {sorted(outside)!r} are outside the permitted "
            f"vocabulary {sorted(PERMITTED_COUNTRY_VOCABULARY)!r}."
        )

    def test_ireland_is_the_only_value_added(self) -> None:
        assert {"Ireland"} == PERMITTED_COUNTRY_VOCABULARY - PRE_EDIT_COUNTRY_VOCABULARY
        seen = {str(record["country"]) for record in _load_venue_records()}
        assert seen == PERMITTED_COUNTRY_VOCABULARY

    def test_germany_is_spelled_once_and_now_carries_four_records(self) -> None:
        germany = [r for r in _load_venue_records() if r["country"] == "Germany"]
        assert sorted(str(r["stadium_id"]) for r in germany) == [
            "BER00",
            "FRA00",
            "GER00",
            "MUN01",
        ]

    def test_a_planted_iso_code_is_refused(self) -> None:
        """Non-vacuity: the guard must fail on the exact mistake it exists for."""
        planted = [
            *_load_venue_records(),
            {**ADDED_2025_VENUE_RECORDS["BER00"], "country": "DEU"},
        ]
        assert _country_values_outside_vocabulary(planted) == {"DEU"}

    def test_the_non_us_set_has_exactly_the_expected_membership(self) -> None:
        non_us = {
            str(record["stadium_id"])
            for record in _load_venue_records()
            if str(record["country"]).upper() not in US_COUNTRY_VALUES
        }
        assert non_us == EXPECTED_NON_US_STADIUM_IDS, (
            f"joined: {sorted(non_us - EXPECTED_NON_US_STADIUM_IDS)!r}; "
            f"left: {sorted(EXPECTED_NON_US_STADIUM_IDS - non_us)!r}"
        )


class TestEverySilverGameResolvesToAVenue:
    """Silver `games` 2002-2026 against the file, in both directions.

    Games -> venues: every stored `stadium_id` resolves, so a game pointing at a missing
    venue fails HERE rather than inside the weather backfill. Venues -> games: every
    record is either used by a stored game or named as a correction target in
    config/international_venue_corrections.toml -- a record nothing points at and nothing
    is recorded to point at is dead reference data.
    """

    @staticmethod
    def _silver_games():
        import pandas as pd

        path = REPO_ROOT / "data" / "silver" / "games.parquet"
        if not path.exists():
            pytest.skip("silver games is not built on this checkout")
        games = pd.read_parquet(path, columns=["game_id", "season", "stadium_id"])
        return games[games["season"].between(2002, 2026)]

    def test_every_stored_stadium_id_resolves(self) -> None:
        games = self._silver_games()
        assert len(games) > 0
        assert games["stadium_id"].notna().all()
        unresolved = sorted(set(games["stadium_id"]) - set(_records_by_stadium_id()))
        assert not unresolved, f"stadium_id(s) {unresolved!r} have no venue record"

    def test_every_record_is_used_or_is_a_recorded_correction_target(self) -> None:
        import tomllib

        games = self._silver_games()
        record_path = REPO_ROOT / "config" / "international_venue_corrections.toml"
        record = tomllib.loads(record_path.read_text(encoding="utf-8"))
        targets = {entry["new_stadium_id"] for entry in record["correction"]}
        unused = sorted(
            set(_records_by_stadium_id()) - set(games["stadium_id"]) - targets
        )
        assert not unused, f"venue record(s) {unused!r} are referenced by nothing"
