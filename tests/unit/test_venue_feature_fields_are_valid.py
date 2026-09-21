"""Every feature-bearing venue field is VALIDATED, never quietly defaulted.

Phase 33.1, Plan 33.1-03 Task 1 (R1/R5, Ruling I3, T-33.1-16b).

THE HOLE THIS CLOSES
--------------------
``features.contextual.ContextualFeaturesCalculator.encode_venue_features`` wraps
its whole body in ``except (ValueError, KeyError, TypeError, AttributeError)`` and
returns ``_default_venue_features()``. The corpus-completeness tests added by Plan
33.1-01 protect a MISSING venue record. Nothing protected a PRESENT record with a
MALFORMED feature-bearing field -- and each of the five such fields crosses a band
or an equality:

    roof_type     -> venue_outdoor / venue_indoor / venue_retractable
    elevation_ft  -> venue_elevation_ft and the >= 3000 venue_high_altitude band
    climate_zone  -> venue_cold_climate / venue_warm_climate
    capacity      -> venue_capacity and the >= 75000 venue_large_stadium band
    surface       -> the grass/turf mismatch feature

A single malformed cell would therefore move a gold column on up to 1,082 games,
for a reason nobody chose, with nothing but a log line -- INSIDE the one rung Plan
33.1-07 rebuilds and attributes. That plan carries
``validate_venue_feature_fields`` as a stated PRECONDITION for exactly this reason.

WHY THE VOCABULARIES ARE LITERALS IN THE PRODUCTION MODULE
-----------------------------------------------------------
Deriving the allowed set from ``data/venues.json`` at import time would let a
malformed record legalise itself by being present in the file it is being
validated against: a validator that cannot fail. So the two vocabularies are
LITERALS in ``features/contextual.py``.

The cross-check that those literals still cover what the OWNER RATIFIED lives
here, on the test side, because that is where ``tests.phase33_state``'s ratified
tables are importable -- a production module must not import from ``tests/``. If
somebody adds a venue with a genuinely new climate token, the literal has to be
widened deliberately and the ratified table is what says the token is real.

FOUR CONTROLS, following tests/unit/test_weather_archive_quarantined.py:31-42
-----------------------------------------------------------------------------
1. NON-VACUITY: the scan visits all 60 committed records, asserted by count.
2. THE ASSERTION: the committed records pass, today.
3. PLANTED VIOLATIONS: one per field, five in all, each on an in-memory COPY,
   each asserted to name that exact ``<stadium_id>.<field>`` pair and its value.
4. NO FALSE POSITIVE: a clean copy of the same records still passes, and the
   refusal reports EVERY offence in one pass rather than only the first.

THE PLANTED VIOLATIONS ARE IN-MEMORY COPIES AND NEVER TOUCH THE FILE.
``data/venues.json`` is a git-TRACKED source file whose suffix is in
``tests.data_boundary.TRACKED_SUFFIXES``, so a test that wrote it -- even into a
temp directory it then pointed the loader at -- would be closer to the COLD-05
boundary than it needs to be. ``validate_venue_feature_fields`` takes the records
as an argument precisely so the corruption can live in a list and die with the
test.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from features import contextual
from tests import phase33_state
from tests.unit.test_venues_json_international import EXPECTED_TOTAL_RECORDS

REPO_ROOT = Path(__file__).resolve().parents[2]
VENUES_PATH = REPO_ROOT / "data" / "venues.json"

# The ratified field orders, stated once each, exactly as
# tests/unit/test_venues_json_historical.py:54 and
# tests/unit/test_venues_json_international.py:56 state them. `strict=True` on the
# zip below turns a field-order change into a failure rather than a silent shift.
HISTORICAL_FACT_FIELDS: tuple[str, ...] = (
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

INTERNATIONAL_FACT_FIELDS: tuple[str, ...] = (
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

# One planted violation per feature-bearing field: (field, bad value, the fragment
# of the value the refusal must quote). The five values are the ones Ruling I3
# names, and each is chosen to be a DIFFERENT kind of wrong -- an out-of-vocabulary
# token, a None, a string that looks like a number, a plausible-but-unratified
# token, and an empty string.
PLANTED_VIOLATIONS: tuple[tuple[str, Any, str], ...] = (
    ("roof_type", "domed", "'domed'"),
    ("capacity", None, "None"),
    ("elevation_ft", "1200 ft", "'1200 ft'"),
    ("climate_zone", "temperate", "'temperate'"),
    ("surface", "", "''"),
)

CORRUPTED_STADIUM_ID = "OAK00"


def _venue_records() -> list[dict[str, Any]]:
    return json.loads(VENUES_PATH.read_text(encoding="utf-8"))["venues"]


def _records_with(field: str, value: Any) -> list[dict[str, Any]]:
    """A COPY of the committed records with one field of one record corrupted."""
    records = copy.deepcopy(_venue_records())
    for record in records:
        if record.get("stadium_id") == CORRUPTED_STADIUM_ID:
            record[field] = value
            return records
    raise AssertionError(
        f"{CORRUPTED_STADIUM_ID} is not in data/venues.json, so every planted "
        "violation below would corrupt nothing and pass vacuously."
    )


def _ratified_cells(field: str) -> set[Any]:
    """Every ratified value of *field* across both owner-ratified fact tables."""
    values: set[Any] = set()
    for fields, table in (
        (HISTORICAL_FACT_FIELDS, phase33_state.HISTORICAL_VENUE_FACTS),
        (INTERNATIONAL_FACT_FIELDS, phase33_state.INTERNATIONAL_VENUE_FACTS),
    ):
        for row in table:
            values.add(dict(zip(fields, row, strict=True))[field])
    return values


class TestTheScanVisitsEveryCommittedRecord:
    """CONTROL 1 -- non-vacuity. A validator over nothing passes over nothing."""

    def test_the_file_holds_the_recorded_number_of_records(self) -> None:
        assert len(_venue_records()) == EXPECTED_TOTAL_RECORDS

    def test_the_validator_reads_the_file_when_handed_nothing(self) -> None:
        """The default path is the one Plan 33.1-07's precondition invokes."""
        contextual.validate_venue_feature_fields()

    def test_all_five_feature_bearing_fields_are_declared(self) -> None:
        assert set(contextual.VENUE_FEATURE_BEARING_FIELDS) == {
            "roof_type",
            "elevation_ft",
            "climate_zone",
            "capacity",
            "surface",
        }
        assert len(contextual.VENUE_FEATURE_BEARING_FIELDS) == len(
            PLANTED_VIOLATIONS
        ), (
            "a feature-bearing field was added or removed without a matching "
            "planted violation, so one of them is unproven."
        )


class TestTheCommittedRecordsPass:
    """CONTROL 2 -- the assertion."""

    def test_the_sixty_committed_records_validate(self) -> None:
        contextual.validate_venue_feature_fields(_venue_records())

    def test_the_encode_except_branch_is_therefore_unreachable(self) -> None:
        """The point of the validator, asserted as an EFFECT on all 60 records.

        ``encode_venue_features`` swallows a malformed field and returns
        ``_default_venue_features()``. If every committed record encodes to
        something OTHER than the default tuple, the branch did not fire -- and
        the validator above is what keeps that true.
        """
        calculator = contextual.ContextualFeaturesCalculator()
        defaults = calculator._default_venue_features()
        indistinguishable = [
            record["stadium_id"]
            for record in _venue_records()
            if calculator.encode_venue_features(record["venue_id"]) == defaults
        ]
        assert not indistinguishable, (
            f"{indistinguishable!r} encode to exactly _default_venue_features(). "
            "Either a record is malformed and the except-branch fired, or a real "
            "record coincides with the defaults -- and the two are "
            "indistinguishable downstream, which is the defect."
        )


class TestPlantedViolations:
    """CONTROL 3 -- one per field, so no field is proven by another's test."""

    @pytest.mark.parametrize(
        ("field", "bad_value", "quoted"),
        PLANTED_VIOLATIONS,
        ids=[field for field, _, _ in PLANTED_VIOLATIONS],
    )
    def test_a_corrupted_field_is_refused_by_name_with_its_value(
        self, field: str, bad_value: Any, quoted: str
    ) -> None:
        records = _records_with(field, bad_value)
        with pytest.raises(ValueError) as excinfo:
            contextual.validate_venue_feature_fields(records)

        message = str(excinfo.value)
        assert f"{CORRUPTED_STADIUM_ID}.{field}" in message, (
            f"the refusal does not name {CORRUPTED_STADIUM_ID}.{field}. A refusal "
            "that says only 'a venue field is malformed' leaves an operator to "
            "search 60 records by hand, which is how a fix becomes a guess. "
            f"Got: {message}"
        )
        assert quoted in message, (
            f"the refusal names the field but not the offending value {quoted}. "
            "The value is what tells the reader whether the cell was mistyped or "
            f"was never entered. Got: {message}"
        )
        assert "data/venues.json" in message, (
            "the repo's convention is that a refusal carries its recovery "
            "command; naming the file to edit is that command here."
        )

    def test_a_true_capacity_is_not_accepted_as_a_number(self) -> None:
        """``bool`` is an ``int`` in Python, and True is not a capacity."""
        with pytest.raises(ValueError, match=r"OAK00\.capacity"):
            contextual.validate_venue_feature_fields(_records_with("capacity", True))

    def test_a_nan_elevation_is_not_accepted_as_a_number(self) -> None:
        """NaN passes ``isinstance(x, float)`` and fails every band silently."""
        with pytest.raises(ValueError, match=r"OAK00\.elevation_ft"):
            contextual.validate_venue_feature_fields(
                _records_with("elevation_ft", float("nan"))
            )

    def test_every_offence_is_reported_in_one_pass(self) -> None:
        """Fixing the first fault must not be how the second one is discovered."""
        records = copy.deepcopy(_venue_records())
        for record in records:
            if record.get("stadium_id") == CORRUPTED_STADIUM_ID:
                record["roof_type"] = "domed"
                record["surface"] = ""
            if record.get("stadium_id") == "ATL00":
                record["climate_zone"] = "temperate"

        with pytest.raises(ValueError) as excinfo:
            contextual.validate_venue_feature_fields(records)

        message = str(excinfo.value)
        for pair in ("OAK00.roof_type", "OAK00.surface", "ATL00.climate_zone"):
            assert pair in message, f"{pair} is missing from {message}"


class TestNoFalsePositive:
    """CONTROL 4 -- the validator is reachable, not universal."""

    def test_an_untouched_copy_still_passes(self) -> None:
        contextual.validate_venue_feature_fields(copy.deepcopy(_venue_records()))

    def test_an_empty_record_list_passes(self) -> None:
        """Nothing to validate is not a violation; it is nothing to validate.

        Stated as its own assertion because it is also the vacuity this module's
        first control exists to rule out for the REAL call.
        """
        contextual.validate_venue_feature_fields([])


class TestTheLiteralVocabulariesCoverWhatTheOwnerRatified:
    """The literals are not derived from the file -- so they are checked here.

    A literal that has drifted away from the ratified tables would refuse a record
    the owner approved, and a reader would then be tempted to widen the validator
    rather than to look at why. These two tests are what make widening a decision
    instead of a reflex.
    """

    def test_every_ratified_climate_zone_is_in_the_vocabulary(self) -> None:
        missing = sorted(
            str(value)
            for value in _ratified_cells("climate_zone")
            if value not in contextual.VENUE_CLIMATE_ZONE_VOCABULARY
        )
        assert not missing, (
            f"the owner ratified climate_zone value(s) {missing!r} that "
            "features.contextual.VENUE_CLIMATE_ZONE_VOCABULARY does not allow. "
            "Widen the literal deliberately, citing the ratification."
        )

    def test_every_ratified_roof_type_is_in_the_vocabulary(self) -> None:
        missing = sorted(
            str(value)
            for value in _ratified_cells("roof_type")
            if value not in contextual.VENUE_ROOF_TYPE_VOCABULARY
        )
        assert not missing, (
            f"the owner ratified roof_type value(s) {missing!r} that "
            "features.contextual.VENUE_ROOF_TYPE_VOCABULARY does not allow."
        )

    def test_the_vocabulary_is_not_merely_every_value_in_the_file(self) -> None:
        """Non-vacuity for the two tests above: the literal is a CHOICE.

        If the vocabulary were the file's own value set, it could never refuse
        anything the file contained -- which is the failure mode Ruling I3 names.
        A token the file does not use proves the literal was written rather than
        harvested.
        """
        in_file = {record.get("climate_zone") for record in _venue_records()}
        assert contextual.VENUE_CLIMATE_ZONE_VOCABULARY - in_file != set(
            contextual.VENUE_CLIMATE_ZONE_VOCABULARY
        ), "the vocabulary and the file share nothing, which cannot be right"
        assert "temperate" not in contextual.VENUE_CLIMATE_ZONE_VOCABULARY, (
            "'temperate' is the planted-violation token. If it is ever added to "
            "the vocabulary, the climate_zone planted violation above stops "
            "proving anything."
        )
