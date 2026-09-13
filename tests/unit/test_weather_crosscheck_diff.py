"""The old-versus-new weather comparator, driven on FIXTURES with no network.

WHAT THIS MODULE ASSERTS, AND WHAT IT DELIBERATELY DOES NOT
-----------------------------------------------------------
It asserts the comparator's MECHANICS: that it joins on `game_id`, that it compares
within the ONE declared tolerance, that it excludes the eight columns the legacy side
does not have, that it reports per season AND per home team, and -- above all -- that
a disagreement NEVER raises.

It does NOT assert that the real diff has any particular shape. That prediction lives
in `scripts/weather_crosscheck_constants.py`, committed BEFORE the data it judges
exists, and Plan 33.1-06 is what measures against it. A test here that checked the
shape would be checking a prediction against fixtures written by the same hand on the
same day, which proves nothing at all.

WHY A DISAGREEMENT IS NOT A FAILURE. This phase PREDICTS a large one: the hour fix
alone moves nearly every comparable row, because the old code indexed an Eastern-local
hourly array at the UTC hour. A comparator that raised on disagreement would refuse
the corpus it exists to validate.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import scripts.weather_crosscheck_constants as preregistration
from scripts.backfill_historical_weather import compare_weather_frames

# Two REAL 2018 game ids, so the season and home-team breakdowns are derived from
# strings the corpus actually contains rather than from invented ones.
GAME_A = "2018_W01_ATL@PHI"
GAME_B = "2018_W02_LA@ARI"
GAME_C = "2019_W01_GB@CHI"


def _legacy_row(game_id: str, **overrides) -> dict:
    """A row in the SEVENTEEN-column legacy schema. No post-2018 columns at all."""
    row = {
        "game_id": game_id,
        "forecast_time": pd.Timestamp("2026-04-16T16:59:23Z"),
        "game_time": pd.Timestamp("2018-09-07T00:20:00Z"),
        "is_outdoor": True,
        "is_cold": False,
        "is_windy": False,
        "is_precipitation": False,
        "temp_f": 70.0,
        "temp_c": 21.1,
        "wind_mph": 6.0,
        "wind_direction": 180.0,
        "humidity_pct": 55.0,
        "precip_prob": None,
        "precip_mm": 0.0,
        "condition": None,
        "condition_code": 0,
        "visibility_km": None,
    }
    row.update(overrides)
    return row


def _new_row(game_id: str, **overrides) -> dict:
    """A row in the current TWENTY-FIVE-column schema."""
    row = _legacy_row(game_id)
    row.update(
        {
            "forecast_time": pd.Timestamp("2026-09-13T10:00:00Z"),
            "dew_point_f": 52.0,
            "apparent_temp_f": 69.0,
            "snowfall_cm": 0.0,
            "wind_gusts_mph": 11.0,
            "cloud_cover_pct": 25.0,
            "weather_coverage": True,
            "weather_source": "archive",
            "created_at": pd.Timestamp("2026-09-13T10:00:00Z"),
        }
    )
    row.update(overrides)
    return row


def _frames(new_rows, legacy_rows):
    return pd.DataFrame(new_rows), pd.DataFrame(legacy_rows)


class TestTheComparatorNeverRaises:
    """Five fixture cases, one per shape the real diff will contain."""

    def test_a_matching_pair_inside_tolerance_agrees(self):
        new, legacy = _frames(
            [_new_row(GAME_A, temp_f=70.09)], [_legacy_row(GAME_A, temp_f=70.0)]
        )

        report = compare_weather_frames(new, legacy)

        assert report["rows_compared"] == 1
        assert report["per_column"]["temp_f"]["disagreed"] == 0
        assert report["per_column"]["temp_f"]["agreed"] == 1

    def test_a_pair_outside_tolerance_disagrees_without_raising(self):
        new, legacy = _frames(
            [_new_row(GAME_A, temp_f=58.4)], [_legacy_row(GAME_A, temp_f=70.0)]
        )

        report = compare_weather_frames(new, legacy)

        assert report["per_column"]["temp_f"]["disagreed"] == 1
        assert report["total_disagreements"] == 1

    def test_a_column_absent_from_the_legacy_side_produces_no_disagreement(self):
        """EXCLUDED, not counted. The legacy side has nothing to disagree with."""
        new, legacy = _frames(
            [_new_row(GAME_A, dew_point_f=999.0, weather_source="historical_forecast")],
            [_legacy_row(GAME_A)],
        )

        report = compare_weather_frames(new, legacy)

        for column in preregistration.ABSENT_FROM_LEGACY_COLUMNS:
            assert column not in report["per_column"], (
                f"{column} was compared, but the legacy side does not carry it. "
                "Comparing across schema widths reports eight columns as universally "
                "changed when they simply did not exist before."
            )
        assert report["total_disagreements"] == 0

    def test_a_row_present_on_one_side_only_is_reported_not_dropped(self):
        new, legacy = _frames(
            [_new_row(GAME_A), _new_row(GAME_B)],
            [_legacy_row(GAME_A), _legacy_row(GAME_C)],
        )

        report = compare_weather_frames(new, legacy)

        assert report["rows_compared"] == 1
        assert report["rows_new_only"] == 1
        assert report["rows_legacy_only"] == 1
        assert GAME_B in report["new_only_game_ids"]
        assert GAME_C in report["legacy_only_game_ids"]

    def test_a_number_becoming_null_is_counted_as_its_own_category(self):
        """The per-game roof rule's signature: a `closed` game stops being fetched.

        It is counted SEPARATELY from a value disagreement because it is a different
        fact -- the pre-registration predicts it is the ONLY category where a number
        becomes a NULL.
        """
        new, legacy = _frames(
            [
                _new_row(
                    GAME_A,
                    temp_f=None,
                    temp_c=None,
                    wind_mph=None,
                    is_outdoor=False,
                    weather_coverage=True,
                )
            ],
            [_legacy_row(GAME_A)],
        )

        report = compare_weather_frames(new, legacy)

        assert report["per_column"]["temp_f"]["became_null"] == 1
        assert report["number_to_null"] >= 1


class TestTheBreakdowns:
    def test_it_reports_per_season_and_per_home_team(self):
        new, legacy = _frames(
            [
                _new_row(GAME_A, temp_f=58.0),
                _new_row(GAME_B, temp_f=70.0),
                _new_row(GAME_C, temp_f=30.0),
            ],
            [_legacy_row(GAME_A), _legacy_row(GAME_B), _legacy_row(GAME_C)],
        )

        report = compare_weather_frames(new, legacy)

        assert set(report["per_season"]) == {2018, 2019}
        assert set(report["per_home_team"]) == {"PHI", "ARI", "CHI"}
        assert report["per_season"][2018]["disagreed"] == 1
        assert report["per_season"][2019]["disagreed"] == 1
        assert report["per_home_team"]["ARI"]["disagreed"] == 0

    def test_an_empty_join_reports_zero_rather_than_raising(self):
        new, legacy = _frames([_new_row(GAME_A)], [_legacy_row(GAME_C)])

        report = compare_weather_frames(new, legacy)

        assert report["rows_compared"] == 0
        assert report["total_disagreements"] == 0
        assert report["per_season"] == {}


class TestTheComparatorReadsTheRegisteredTerms:
    def test_the_tolerance_comes_from_the_pre_registration(self):
        report = compare_weather_frames(
            *_frames([_new_row(GAME_A)], [_legacy_row(GAME_A)])
        )
        assert report["tolerance"] == preregistration.CROSSCHECK_TOLERANCE_F

    def test_exactly_the_registered_columns_are_compared(self):
        report = compare_weather_frames(
            *_frames([_new_row(GAME_A)], [_legacy_row(GAME_A)])
        )
        assert set(report["compared_columns"]) == set(preregistration.COMPARED_COLUMNS)
        assert "game_id" not in report["compared_columns"], (
            "the join key is equal by construction; reporting it would inflate the "
            "agreement rate with a tautology"
        )
        assert "forecast_time" not in report["compared_columns"], (
            "forecast_time is the RUN clock -- it differs on every row of every "
            "re-run and says nothing about the weather"
        )

    def test_a_tolerance_boundary_value_is_treated_as_agreement(self):
        """Exactly at the tolerance, not past it: `round(..., 1)` is the contract."""
        delta = preregistration.CROSSCHECK_TOLERANCE_F
        new, legacy = _frames(
            [_new_row(GAME_A, temp_f=70.0 + delta)], [_legacy_row(GAME_A, temp_f=70.0)]
        )

        report = compare_weather_frames(new, legacy)

        assert report["per_column"]["temp_f"]["disagreed"] == 0


class TestThePreRegistrationItself:
    """The registered expectation is READABLE and internally consistent.

    Nothing here checks the DATA -- there is none yet. These assert that the
    prediction says what SPEC R2 requires it to say, which is checkable today.
    """

    def test_it_names_exactly_three_causes_each_with_a_measured_population(self):
        shape = preregistration.EXPECTED_DIFF_SHAPE
        assert len(shape) == 3
        for name, cause in shape.items():
            assert cause["population"].strip(), name
            assert isinstance(cause["measured_size"], int), name
            assert cause["measured_size"] > 0, name
            assert cause["predicted_effect"].strip(), name

    def test_the_routing_cause_agrees_with_the_measured_routing_diff(self):
        from tests import phase33_state

        assert (
            preregistration.EXPECTED_DIFF_SHAPE["routing_fix"]["measured_size"]
            == (phase33_state.ROUTING_COORDINATE_DIFF["weather_applicable_games"])
        )

    def test_it_states_both_of_its_non_claims(self):
        """Asserted against the DECLARED constant, not against the source's wrapping.

        The prose block in the module header says the same two things, but line
        wrapping and string concatenation break the phrases across source lines, so a
        substring search over the file would be asserting a formatting choice. What
        matters is what the module DECLARES, which is what a consumer reads.
        """
        assert len(preregistration.NON_CLAIMS) == 2
        joined = " ".join(preregistration.NON_CLAIMS)

        assert "concentrated in relocated-franchise home games" in joined, (
            "the pre-registration must say OUT LOUD that it does NOT predict the "
            "SPEC's own concentration claim -- the hour fix moves nearly every "
            "comparable row, so a check written against that claim would read as a "
            "failure while working correctly"
        )
        assert "17-column" in joined, (
            "the pre-registration must name the legacy schema width, or it is "
            "comparing across schema widths without saying so"
        )

        # And the human-readable half is present too, so a reader of the file finds
        # the same two statements without importing anything.
        text = Path(preregistration.__file__).read_text(encoding="utf-8")
        assert text.count("MUST NOT") >= 2

    def test_it_registers_four_reported_quantities_including_the_probe(self):
        quantities = preregistration.REPORTED_QUANTITIES
        assert len(quantities) == 4
        assert any("archive-versus-forecast" in q for q in quantities)

    def test_the_provenance_probe_is_a_measurement_and_not_an_expectation(self):
        spec = preregistration.ARCHIVE_VERSUS_FORECAST_PROBE_SPEC

        assert "comparable_population" in spec
        assert "interpretation_rule" in spec
        assert spec["interpretation_rule"].strip()
        assert spec["n"] is None, (
            "a pre-registration that guessed its own sample size would be predicting "
            "the answer's precision"
        )
        assert len(spec["reported_differences"]) == 2
        assert "28" in spec["comparable_population"]
        assert "1,942" in spec["excluded_population"], (
            "the legacy bronze rows must be named as EXCLUDED, or the probe silently "
            "answers a different question while looking like this one"
        )
        assert "n=0" in spec["null_case"]
        for forbidden in ("bias correction", "Wave-15"):
            assert forbidden in spec["interpretation_rule"], forbidden

    def test_the_module_imports_nothing_from_the_project(self):
        """Constants only: no imports, no I/O, no logic."""
        source = Path(preregistration.__file__).read_text(encoding="utf-8")
        code = [
            line
            for line in source.splitlines()
            if line.startswith(("import ", "from "))
        ]
        assert code == [], (
            f"the pre-registration imports {code}. It is a constants module: an "
            "import is a way for its declared values to depend on something that can "
            "change after it was frozen."
        )
