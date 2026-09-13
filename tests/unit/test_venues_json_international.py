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

import json
from pathlib import Path

import pytest

from tests import phase33_state
from tests.fixtures import season_2026

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

# 38 -> 60: Plan 33.1-01 added the 22 historical venue records (R1), so that the 1,082
# games at stadiums this file could not describe stop resolving by today's home team.
# UPDATED here rather than forked -- a new module asserting 60 while this one asserts 38
# is two answers to one question.
EXPECTED_TOTAL_RECORDS = 60
EXPECTED_EXISTING_RECORDS = 30

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
    return {
        row[0]: dict(zip(FACT_FIELDS, row, strict=True))
        for row in phase33_state.INTERNATIONAL_VENUE_FACTS
    }


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

        TWO SETS ARE EXCLUDED, AND FOR THE SAME REASON. The eight international venues
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
        not_in_service = {
            row[0] for row in phase33_state.INTERNATIONAL_VENUE_FACTS
        } | set(phase33_state.HISTORICAL_STADIUM_IDS)

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
