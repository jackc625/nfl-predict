"""The 22 historical venue records, value-by-value against the owner-ratified table.

Phase 33.1, Plan 33.1-01 Task 3 (R1, T-33.1-01/02/03/04/76).

WHAT THIS MODULE IS FOR
-----------------------
The pinned 2002-2025 schedules carry 55 distinct ``stadium_id`` values over 6,499 games.
``data/venues.json`` held 38, so 22 stadiums covering 1,082 games had no record at all and
every one of those games resolved by its PRESENT-DAY home team -- 141 outdoor Oakland
Coliseum games would have been given Las Vegas desert weather under an indoor roof.

Nothing in this repository can settle where Giants Stadium stood or what the Georgia Dome's
playing surface was in 2003. So the values were RESEARCHED, TABLED, and RATIFIED by the
owner at Plan 33.1-01's blocking checkpoint on 2026-09-12, committed ONCE to
``tests.phase33_state.HISTORICAL_VENUE_FACTS``, and ``data/venues.json`` was GENERATED from
that constant. This module asserts that the file says the same thing the record does -- it
does not re-decide the values.

THIS MODULE READS. IT NEVER WRITES. ``data/venues.json`` is a git-tracked SOURCE file whose
suffix is in ``tests.data_boundary.TRACKED_SUFFIXES``, so a test that wrote it would be a
COLD-05 boundary crossing under the guard Plan 33-01 armed. The 22 records were added as a
SOURCE EDIT, outside pytest.

WHAT THE PROVENANCE TESTS CAN AND CANNOT DO
--------------------------------------------
``test_every_added_record_carries_a_revision_pinned_source_for_every_external_field`` and
``test_no_external_field_is_uncovered_by_the_provenance_record`` assert that a source is
PRESENT, REVISION-PINNED and FIELD-SPECIFIC. They cannot assert it is CORRECT. Only the
Task-2 owner ratification does that, and no test will ever replace it.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import pytest

from conf.season_partition import CORPUS_FIRST_SEASON
from tests import phase33_state
from tests.unit.test_venues_json_international import EXPECTED_TOTAL_RECORDS

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

# The corpus floor, read from the one rule module (Plan 33.2-18: no floor literal outside
# conf/season_partition.py). Was: FIRST_SEASON = 2002.
FIRST_SEASON = CORPUS_FIRST_SEASON
LAST_SEASON = 2025

# The field order HISTORICAL_VENUE_FACTS is stated in, named here so the unpacking below is
# checkable rather than positional folklore. FOURTEEN fields, one more than
# INTERNATIONAL_VENUE_FACTS's thirteen: five of the 22 are non-US and `state` must be
# recordable as the empty string rather than guessed.
FACT_FIELDS: tuple[str, ...] = (
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

# Values that would mean a cell was DEFAULTED rather than entered. `outdoor` is not in this
# set -- it is a legitimate ratified roof for sixteen of the 22 -- and neither is `""`, for
# the `state` of the five non-US records, which is why the sentinel check below is scoped to
# the geographic cells rather than applied to every field.
DEFAULT_SENTINELS: frozenset[object] = frozenset({None, "", 0, "unknown"})

GEOGRAPHIC_FIELDS: tuple[str, ...] = (
    "latitude",
    "longitude",
    "elevation_ft",
    "timezone",
    "roof_type",
    "surface",
    "capacity",
    "climate_zone",
)

# ---------------------------------------------------------------------------
# THE PREDECESSOR/SUCCESSOR SEPARATION, AS THE OWNER RE-STATED IT ON 2026-09-12.
#
# 33.1-01-PLAN.md asked for a single assertion: that all six relocation pairs are
# "more than 0.1 degrees" apart. THAT THRESHOLD IS PHYSICALLY UNSATISFIABLE for three
# of the six, and not because any coordinate is wrong -- because the buildings really
# are that close. Measured against the ratified table:
#
#     OAK00/VEG00   dlat 1.6608  dlon  7.0173    650.38 km
#     SDG00/LAX01   dlat 1.1704  dlon  1.2198    172.53 km
#     STL00/LAX01   dlat 4.6793  dlon 28.1506   2565.39 km
#     LAX99/LAX01   dlat 0.0607  dlon  0.0514      8.25 km   <- under 0.1 on BOTH axes
#     LAX97/LAX01   dlat 0.0891  dlon  0.0781     12.25 km   <- under 0.1 on BOTH axes
#     NYC00/NYC01   dlat 0.0013  dlon  0.0024      0.25 km   <- under 0.1 on BOTH axes
#
# The LA Memorial Coliseum is 8 km from SoFi, StubHub Center is 12 km from it, and
# 33.1-RESEARCH.md itself records that Giants Stadium and MetLife are about 400 m apart
# in the same parking lot. No correct coordinate can clear 0.1 degrees for those three.
#
# THE OWNER RULED on 2026-09-12, after being shown the measurements above: assert that
# each predecessor differs from its successor at all, AND that the three genuinely
# distant pairs clear 0.1 degrees on at least one axis. The three close pairs are
# asserted NON-IDENTICAL only. They share an ERA5 grid cell (roughly 9-11 km) by
# real-world geography, not by error, and that is recorded here rather than discovered
# later by somebody wondering why the threshold moved.
#
# This is a DELIBERATE weakening of the plan text under an explicit ruling, not a test
# quietly relaxed to make it pass.
# ---------------------------------------------------------------------------

DISTANT_RELOCATION_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("OAK00", "VEG00", 650.38),
    ("SDG00", "LAX01", 172.53),
    ("STL00", "LAX01", 2565.39),
)

CLOSE_RELOCATION_PAIRS: tuple[tuple[str, str, float], ...] = (
    ("LAX99", "LAX01", 8.25),
    ("LAX97", "LAX01", 12.25),
    ("NYC00", "NYC01", 0.25),
)

MINIMUM_DISTANT_SEPARATION_DEGREES = 0.1


def _load_venue_records() -> list[dict[str, object]]:
    return json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]


def _facts_by_stadium_id() -> dict[str, dict[str, object]]:
    return {
        row[0]: dict(zip(FACT_FIELDS, row, strict=True))
        for row in phase33_state.HISTORICAL_VENUE_FACTS
    }


def _records_by_stadium_id() -> dict[str, dict[str, object]]:
    return {
        str(record["stadium_id"]): record
        for record in _load_venue_records()
        if record.get("stadium_id")
    }


def _sources_by_pair() -> dict[tuple[str, str], str]:
    return {
        (stadium_id, field): source
        for stadium_id, field, source in phase33_state.HISTORICAL_VENUE_FIELD_SOURCES
    }


def _external_pairs() -> set[tuple[str, str]]:
    """The 88-element cross product every external cell must be covered by."""
    return set(
        itertools.product(
            phase33_state.HISTORICAL_STADIUM_IDS,
            phase33_state.EXTERNALLY_SOURCED_VENUE_FIELDS,
        )
    )


def _carries_revision_pin(source: str) -> bool:
    """An ``oldid=`` or a bare Wikidata ``Q`` entity id.

    What makes a citation checkable a year from now rather than a pointer to whatever the
    page happens to say then.
    """
    if "oldid=" in source:
        return True
    return any(
        token.startswith("Q") and token[1:].isdigit()
        for token in source.replace(",", " ")
        .replace("(", " ")
        .replace(")", " ")
        .split()
    )


class TestTheFileHoldsOneRecordPerStadiumId:
    """R1's uniqueness requirement, over EVERY record rather than only the 22 added.

    Uniqueness is asserted on ``stadium_id`` and NEVER on coordinates: GER00 and MUN01
    legitimately carry identical geography, because they are the same building.
    """

    def test_the_file_holds_the_expected_number_of_records(self) -> None:
        """Renamed from `test_the_file_holds_sixty_records` by Plan 33.2-09.

        The count is imported from test_venues_json_international, the ONE place it is
        derived (the 60 recorded here plus Plan 33.2-09's two 2025 venues).
        """
        records = _load_venue_records()
        assert len(records) == EXPECTED_TOTAL_RECORDS, (
            f"data/venues.json holds {len(records)} records, expected "
            f"{EXPECTED_TOTAL_RECORDS} -- the 38 that were present before Plan 33.1-01, "
            "the 22 historical venues ratified on 2026-09-12, and the two 2025 "
            "international venues Plan 33.2-09 added."
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
            "stadium a game resolves to."
        )

    def test_every_record_carries_a_non_empty_stadium_id(self) -> None:
        missing = [
            record["venue_id"]
            for record in _load_venue_records()
            if not record.get("stadium_id")
        ]
        assert not missing, (
            f"{len(missing)} venue record(s) carry no stadium_id: {missing!r}. "
            "stadium_id-keyed routing cannot resolve a venue that does not declare its "
            "code."
        )

    def test_venue_ids_are_unique_across_the_file(self) -> None:
        """`venue_id` is the key `_get_venue_by_id` and `_get_venue_surface` resolve on.

        A collision between a historical slug and an existing one would send a contextual
        feature lookup to the wrong building without raising.
        """
        ids = [str(record["venue_id"]) for record in _load_venue_records()]
        duplicates = sorted({slug for slug in ids if ids.count(slug) > 1})
        assert not duplicates, (
            f"venue_id(s) {duplicates!r} appear on more than one record. "
            "features/contextual._get_venue_by_id returns the FIRST match."
        )


class TestEveryAddedRecordIsCompleteAndSourced:
    """Value-by-value, against the owner-ratified table -- not against the feed."""

    def test_all_twenty_two_historical_records_are_present(self) -> None:
        present = set(_records_by_stadium_id())
        expected = set(phase33_state.HISTORICAL_STADIUM_IDS)
        assert expected <= present, (
            f"historical venue(s) {sorted(expected - present)!r} are absent from "
            "data/venues.json. Each covers games the pinned feed carries, and an absent "
            "record means those games resolve by today's home team instead."
        )

    @pytest.mark.parametrize("stadium_id", list(phase33_state.HISTORICAL_STADIUM_IDS))
    def test_each_ratified_cell_matches_the_file(self, stadium_id: str) -> None:
        record = _records_by_stadium_id().get(stadium_id)
        assert record is not None, f"{stadium_id} is not in data/venues.json"
        facts = _facts_by_stadium_id()[stadium_id]

        for field in FACT_FIELDS:
            assert record.get(field) == facts[field], (
                f"{stadium_id}.{field} is {record.get(field)!r} in data/venues.json but "
                f"the owner ratified {facts[field]!r} on 2026-09-12. The ratified table "
                "is the source; the file follows it, never the reverse."
            )

    @pytest.mark.parametrize("stadium_id", list(phase33_state.HISTORICAL_STADIUM_IDS))
    def test_no_geographic_cell_is_a_default(self, stadium_id: str) -> None:
        """Eight cells are ENTERED for all 22, never defaulted and never inherited.

        `state` is deliberately excluded: it is the empty string for the five non-US
        records, which is a recorded boundary rather than a default.
        """
        record = _records_by_stadium_id().get(stadium_id)
        assert record is not None, f"{stadium_id} is not in data/venues.json"

        for field in GEOGRAPHIC_FIELDS:
            value = record.get(field)
            assert value not in DEFAULT_SENTINELS, (
                f"{stadium_id}.{field} reads {value!r}, which is a default sentinel. "
                "Every one of these feeds a live contextual or weather input; leaving one "
                "defaulted moves a gold column on 1,082 games for a reason nobody chose."
            )

    @pytest.mark.parametrize("stadium_id", list(phase33_state.HISTORICAL_STADIUM_IDS))
    def test_each_added_record_declares_no_home_team(self, stadium_id: str) -> None:
        """`home_teams == []` is what stops a historical record shadowing its successor.

        `scripts/ingest_weather._get_venue_record` returns the FIRST record whose
        `home_teams` contains the team, so an OAK00 record claiming `LV` would win the
        lookup for the LIVE 2026 season.
        """
        record = _records_by_stadium_id().get(stadium_id)
        assert record is not None, f"{stadium_id} is not in data/venues.json"
        assert record.get("home_teams") == [], (
            f"{stadium_id} declares home_teams {record.get('home_teams')!r}. A non-empty "
            "list would make the home-team resolvers route a live team's ordinary home "
            "games to a demolished stadium."
        )

    def test_no_live_team_resolves_to_a_historical_record(self) -> None:
        """The positive half: LV still resolves to VEG00, not to OAK00.

        Asserting the 22 declare no home team proves they cannot be claimed. This proves
        the teams whose stadiums they used to be are still answered by the CURRENT venue.
        """
        import pandas as pd

        from scripts.ingest_weather import WeatherDataIngester

        ingester = WeatherDataIngester()
        venues_df = pd.DataFrame(_load_venue_records())
        historical = set(phase33_state.HISTORICAL_STADIUM_IDS)

        for team, expected in (
            ("LV", "VEG00"),
            ("LA", "LAX01"),
            ("LAC", "LAX01"),
            ("NYG", "NYC01"),
            ("SF", "SFO01"),
            ("MIN", "MIN01"),
            ("ARI", "PHO00"),
            ("IND", "IND00"),
            ("DAL", "DAL00"),
            ("ATL", "ATL97"),
            ("PHI", "PHI00"),
            ("BUF", "BUF00"),
        ):
            resolved = ingester._get_venue_record(team, venues_df)
            assert resolved["stadium_id"] == expected, (
                f"{team} resolves to {resolved['stadium_id']!r}, expected {expected!r}."
            )
            assert resolved["stadium_id"] not in historical, (
                f"{team} resolves to the HISTORICAL record "
                f"{resolved['stadium_id']!r}. A historical record has shadowed the venue "
                "the team actually plays in today."
            )

    @pytest.mark.parametrize(
        ("stadium_id", "field"),
        sorted(
            itertools.product(
                phase33_state.HISTORICAL_STADIUM_IDS,
                phase33_state.EXTERNALLY_SOURCED_VENUE_FIELDS,
            )
        ),
    )
    def test_every_added_record_carries_a_revision_pinned_source_for_every_external_field(
        self, stadium_id: str, field: str
    ) -> None:
        """All 88 external cells, each against ITS OWN citation (Ruling C2).

        A Wikidata P625 coordinate statement substantiates a latitude and a longitude and
        says nothing about what the playing surface was or how many seats the building
        held -- and `surface` and `capacity` BOTH feed live contextual features
        (features/contextual.py:345-356 and :653).

        This asserts the source is PRESENT, REVISION-PINNED and FIELD-SPECIFIC. It cannot
        assert the source is CORRECT; the Task-2 owner ratification does that.
        """
        source = _sources_by_pair().get((stadium_id, field))
        assert source, (
            f"{stadium_id}.{field} has no entry in HISTORICAL_VENUE_FIELD_SOURCES. An "
            "externally-researched value with no citation of its own is the unsourced "
            "coordinate 33.1-SPEC.md's second prohibition forbids."
        )
        assert _carries_revision_pin(source), (
            f"{stadium_id}.{field} cites {source!r}, which carries neither an `oldid=` nor "
            "a Wikidata Q entity id. An unpinned citation points at whatever the page says "
            "today, which is not a record of what was ratified."
        )

    def test_no_external_field_is_uncovered_by_the_provenance_record(self) -> None:
        """The coverage assertion, which is what closes the per-venue loophole.

        The parametrize above only ever visits the entries that EXIST, so 87 covered cells
        and one absent one would pass it. This asserts the record is a SUPERSET of the
        88-element cross product and names every uncovered pair.
        """
        covered = set(_sources_by_pair())
        needed = _external_pairs()
        uncovered = sorted(needed - covered)
        assert not uncovered, (
            f"{len(uncovered)} externally-sourced cell(s) are uncovered by "
            f"HISTORICAL_VENUE_FIELD_SOURCES: "
            + ", ".join(f"{sid}.{field}" for sid, field in uncovered)
            + ". A field nobody sourced cannot be written into a production reference file."
        )

    def test_the_provenance_record_covers_all_eight_fields_on_all_twenty_two(
        self,
    ) -> None:
        """176 rows: 22 records x 8 non-identity fields, external AND derived."""
        rows = phase33_state.HISTORICAL_VENUE_FIELD_SOURCES
        expected = len(phase33_state.HISTORICAL_STADIUM_IDS) * len(GEOGRAPHIC_FIELDS)
        assert len(rows) == expected, (
            f"HISTORICAL_VENUE_FIELD_SOURCES holds {len(rows)} rows, expected {expected} "
            f"({len(phase33_state.HISTORICAL_STADIUM_IDS)} records x "
            f"{len(GEOGRAPHIC_FIELDS)} non-identity fields)."
        )
        pairs = set(_sources_by_pair())
        missing = sorted(
            set(
                itertools.product(
                    phase33_state.HISTORICAL_STADIUM_IDS, GEOGRAPHIC_FIELDS
                )
            )
            - pairs
        )
        assert not missing, "the provenance record does not cover: " + ", ".join(
            f"{sid}.{field}" for sid, field in missing
        )

    def test_no_surface_or_capacity_reuses_its_venue_coordinate_citation(self) -> None:
        """Ruling C2's "entered PER FIELD", asserted rather than trusted.

        A citation names the STATEMENT it was read from, so one document legitimately
        backing three fields still yields three different strings. A byte-identical string
        across a coordinate and a surface is therefore not one document doing two jobs --
        it is the per-VENUE citation this record exists to replace, pasted twice.
        """
        sources = _sources_by_pair()
        reused: list[str] = []
        for stadium_id in phase33_state.HISTORICAL_STADIUM_IDS:
            for field in ("surface", "capacity"):
                for coordinate_field in ("latitude", "longitude"):
                    if sources.get((stadium_id, field)) == sources.get(
                        (stadium_id, coordinate_field)
                    ):
                        reused.append(f"{stadium_id}.{field}")
                        break
        assert not reused, (
            "these cells cite their venue's COORDINATE statement: "
            + ", ".join(sorted(reused))
            + ". A coordinate statement says nothing about a playing surface or a seat "
            "count, and both feed live contextual features."
        )

    def test_no_source_names_an_existing_venue_record(self) -> None:
        """T-33.1-02: the successor venue is not a source for its predecessor's geography.

        Giants Stadium and MetLife are about 400 m apart in the same parking lot, so a
        geocoder asked for "Giants Stadium" returns MetLife -- which is exactly what the
        owner rejected in the Wikidata P625 value on 2026-09-12.
        """
        historical = set(phase33_state.HISTORICAL_STADIUM_IDS)
        successors = set(_records_by_stadium_id()) - historical
        offenders: list[str] = []
        for stadium_id, field, source in phase33_state.HISTORICAL_VENUE_FIELD_SOURCES:
            if field not in phase33_state.EXTERNALLY_SOURCED_VENUE_FIELDS:
                # Derived citations legitimately name a sibling record: the climate_zone
                # rule IS "take the zone of the same-metro record", and hiding which one
                # would make the derivation uncheckable.
                continue
            for code in successors:
                if code in source:
                    offenders.append(f"{stadium_id}.{field} names {code}")
        assert not offenders, (
            "these external citations name a venue that is still in service: "
            + "; ".join(sorted(offenders))
            + ". The successor's own record plus an offset is not a source."
        )


class TestPreRelocationGamesDoNotResolveToTheSuccessor:
    """T-33.1-02, in the form the owner re-stated on 2026-09-12.

    See the DISTANT_RELOCATION_PAIRS / CLOSE_RELOCATION_PAIRS comment above for the
    measurements and the reason the plan's single >0.1-degree threshold could not stand.
    """

    @pytest.mark.parametrize(
        ("predecessor", "successor", "measured_km"), DISTANT_RELOCATION_PAIRS
    )
    def test_a_distant_predecessor_is_far_from_its_successor(
        self, predecessor: str, successor: str, measured_km: float
    ) -> None:
        records = _records_by_stadium_id()
        before, after = records[predecessor], records[successor]
        delta_lat = abs(float(before["latitude"]) - float(after["latitude"]))
        delta_lon = abs(float(before["longitude"]) - float(after["longitude"]))
        assert (
            delta_lat > MINIMUM_DISTANT_SEPARATION_DEGREES
            or delta_lon > MINIMUM_DISTANT_SEPARATION_DEGREES
        ), (
            f"{predecessor} and {successor} differ by only dlat={delta_lat:.4f} "
            f"dlon={delta_lon:.4f}, but these two are about {measured_km:.0f} km apart in "
            "the world. A predecessor that has collapsed onto its successor's grid cell "
            "is the misrouting this phase exists to remove, not a rounding."
        )

    @pytest.mark.parametrize(
        ("predecessor", "successor", "measured_km"), CLOSE_RELOCATION_PAIRS
    )
    def test_a_close_predecessor_is_still_not_its_successor(
        self, predecessor: str, successor: str, measured_km: float
    ) -> None:
        """These three are genuinely within an ERA5 cell of their successors.

        8.25 km, 12.25 km and 0.25 km respectively -- so they will fetch the same weather,
        correctly. What must still hold is that they are DISTINCT records with DISTINCT
        coordinates, because everything else about them differs: roof, surface, capacity
        and climate all feed contextual features that are not grid-cell-limited.
        """
        records = _records_by_stadium_id()
        before, after = records[predecessor], records[successor]
        assert (before["latitude"], before["longitude"]) != (
            after["latitude"],
            after["longitude"],
        ), (
            f"{predecessor} carries byte-identical coordinates to {successor}, but the two "
            f"are about {measured_km:.2f} km apart. Identical geography here means the "
            "predecessor was given the successor's location rather than its own."
        )


class TestTheRatifiedTableIsTheSource:
    """The 22 were DERIVED from the pinned feed, not typed from memory."""

    def test_the_twenty_two_are_exactly_the_feed_ids_the_file_lacked(self) -> None:
        """Re-derived at test time, in BOTH directions.

        Deliberately NOT modelled on the international module's
        `TestTheThirtyExistingCodesWereDerivedNotTyped`: that class derives each code from
        the 2026 feed's non-neutral home games, a join that is meaningless for a demolished
        stadium whose franchise plays somewhere else now.
        """
        from data.upstream_pin import load_schedules

        schedules = load_schedules(list(range(FIRST_SEASON, LAST_SEASON + 1)))
        feed_ids = {str(value) for value in schedules["stadium_id"].dropna().unique()}
        all_file_ids = set(_records_by_stadium_id())
        historical = set(phase33_state.HISTORICAL_STADIUM_IDS)
        pre_existing = all_file_ids - historical

        derived = feed_ids - pre_existing
        assert derived == historical, (
            "the ratified id set and the ids re-derived from the pinned "
            f"{FIRST_SEASON}-{LAST_SEASON} schedules disagree.\n"
            f"  ratified but not derived: {sorted(historical - derived)!r}\n"
            f"  derived but not ratified: {sorted(derived - historical)!r}"
        )

    def test_every_feed_stadium_id_now_resolves_to_exactly_one_record(self) -> None:
        """R1's acceptance: all 55, resolved by id, with no miss and no ambiguity."""
        from data.upstream_pin import load_schedules
        from features import contextual

        schedules = load_schedules(list(range(FIRST_SEASON, LAST_SEASON + 1)))
        feed_ids = sorted({str(v) for v in schedules["stadium_id"].dropna().unique()})
        records = _load_venue_records()

        unresolved = [
            code
            for code in feed_ids
            if contextual._get_venue_by_stadium_id(code, records) is None
        ]
        assert not unresolved, (
            f"{len(unresolved)} stadium_id(s) in the pinned feed resolve to no venue "
            f"record: {unresolved!r}. Every one of them is games that would fall back to "
            "home-team routing."
        )
        assert len(feed_ids) == 55, (
            f"the pinned feed now carries {len(feed_ids)} distinct stadium_id values, not "
            "the 55 this plan was measured against. Re-measure before trusting the "
            "coverage claim above."
        )

    def test_the_ratified_roof_types_agree_with_the_feed_they_were_derived_from(
        self,
    ) -> None:
        """`roof_type` was DERIVED from the feed's own single-valued `roof`.

        `scripts/ingest_games._get_venue_roof_type` consults `stadium_id` FIRST, so once
        these records exist that step starts winning for 1,082 games. Deriving rather than
        looking up is what makes that change a no-op in silver; this asserts the no-op.
        """
        from data.upstream_pin import load_schedules
        from scripts.ingest_weather import NFLVERSE_ROOF_MAP

        schedules = load_schedules(list(range(FIRST_SEASON, LAST_SEASON + 1)))
        facts = _facts_by_stadium_id()
        disagreements: list[str] = []
        for stadium_id in phase33_state.HISTORICAL_STADIUM_IDS:
            rows = schedules[schedules["stadium_id"] == stadium_id]
            feed_values = sorted({str(v) for v in rows["roof"].dropna().unique()})
            assert len(feed_values) == 1, (
                f"{stadium_id} carries feed roof values {feed_values!r}. roof_type is a "
                "VENUE property; a multi-valued feed roof means the derivation is "
                "answering the wrong question."
            )
            mapped = NFLVERSE_ROOF_MAP[feed_values[0]]
            if mapped != facts[stadium_id]["roof_type"]:
                disagreements.append(
                    f"{stadium_id}: feed {feed_values[0]!r} -> {mapped!r}, ratified "
                    f"{facts[stadium_id]['roof_type']!r}"
                )
        assert not disagreements, (
            "ratified roof_type disagrees with the feed it was derived from for: "
            + "; ".join(disagreements)
            + ". A disagreement would change `venue_roof` in silver on the next "
            "ingest_games run -- which may be right, but it must be a DECISION, not a "
            "drift."
        )

    def test_every_ratified_timezone_is_a_real_iana_zone(self) -> None:
        """`select_forecast_hour_for_kickoff` hard-fails by name on a venue with no zone."""
        from zoneinfo import ZoneInfo

        for stadium_id, facts in _facts_by_stadium_id().items():
            zone = str(facts["timezone"])
            assert zone.lower() not in {"", "auto", "gmt"}, (
                f"{stadium_id} carries timezone {zone!r}, which is the provider echoing "
                "the request back rather than resolving it."
            )
            ZoneInfo(zone)

    def test_no_added_record_trips_the_high_altitude_band(self) -> None:
        """Ruling A's verdict, asserted -- and its MARGIN recorded honestly.

        `venue_high_altitude` fires at elevation_ft >= 3000 (features/contextual.py:643).
        All 22 sit below it, so the flag is 0.0 for every one of the 1,082 games.

        Ruling A justified deriving these elevations from a ~90 m DEM by asserting that
        "no venue among the 22 sits within 1,500 ft of that boundary -- the highest is
        PHO99 Sun Devil Stadium at roughly 1,150 ft". THAT PREMISE IS FALSE: SAO00 Arena
        Corinthians is 2,562 ft, 438 ft of margin, and GER00 is 1,611 ft. The verdict
        survives; the margin is 3.4x smaller than claimed. Recorded here so the next
        reader inherits the measurement rather than the ruling's arithmetic.
        """
        facts = _facts_by_stadium_id()
        tripping = sorted(
            code for code, row in facts.items() if int(row["elevation_ft"]) >= 3000
        )
        assert not tripping, (
            f"venues {tripping!r} trip venue_high_altitude. None of the 22 should; if an "
            "elevation moved, Ruling A's whole justification for deriving them needs "
            "re-taking."
        )
        highest = max(facts.items(), key=lambda item: int(item[1]["elevation_ft"]))
        assert highest[0] == "SAO00" and int(highest[1]["elevation_ft"]) == 2562, (
            f"the highest of the 22 is now {highest[0]} at "
            f"{highest[1]['elevation_ft']} ft, not SAO00 at 2562 ft. The recorded 438 ft "
            "of margin to the 3,000 ft band has moved and Ruling A must be re-stated."
        )
