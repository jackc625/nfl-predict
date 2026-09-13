"""The PRE-REGISTERED expected SHAPE of the old-versus-new weather diff (SPEC R2).

ONE-WAY BY CONSTRUCTION
=======================
This module is the pre-registration. It is committed BEFORE the diff it predicts is
computed, and the ordering is the only thing that makes it evidence rather than
commentary. A check written after seeing the data can always be rationalised into
agreement; a check written before it cannot.

EDITING THIS FILE AFTER THE MEASUREMENT COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE
EVIDENCE. There is no honest repair path. If the expectation is wrong, that is a
FINDING about the expectation, and the finding is what gets reported. Correcting the
prediction so that it matches the answer turns a pre-registration into a description.

THIS FILE DOES NOT RECORD ITS OWN HASH, and that is deliberate rather than an
oversight. A document that must CONTAIN and exactly REPRODUCE its own whole-file hash
is self-referential: writing the hash changes the bytes the hash was computed over, so
no fixed point exists without a canonical exclusion rule nobody has defined. A test
written against such a marker would have to be relaxed -- to "hash everything except
the marker line", or "everything above it" -- and each relaxation is a new rule
invented under execution pressure to make a failing assertion pass. That is how a
guard becomes decoration.

The witness therefore lives OUTSIDE this file, in a LATER commit:

    tests.phase33_state.WEATHER_CROSSCHECK_PREREGISTRATION_COMMIT
    tests.phase33_state.WEATHER_CROSSCHECK_PREREGISTRATION_FILE_SHA256

and `tests/unit/test_preregistration_ancestry.py` asserts the ancestry. This is the
pattern Phase 30 proved and Phase 31 reused; it is not re-invented here.

WHAT THIS FILE IS NOT
=====================
It is NOT Phase 31's pre-registration and it does not extend it.
`backtest.ev_chain_constants.PREREGISTRATION_PATHS` names exactly two paths and
`test_both_preregistration_files_exist_and_are_tracked` asserts that count. Adding to
that tuple would redefine a PUBLISHED pre-registration after the fact, which is the one
thing `ev_chain_constants.py`'s own header says destroys the evidence. This phase's
ancestry check is a NEW class in the existing test module, reusing its helpers.

NO IMPORTS, NO I/O, NO LOGIC. Constants only, so that reading this file tells a reader
everything the prediction says and nothing can change it at runtime.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

# The module's own repo-root-relative path, named here so the witness and the rule
# cannot disagree about which file IS the pre-registration.
PREREGISTRATION_PATH: str = "scripts/weather_crosscheck_constants.py"


# ---------------------------------------------------------------------------
# The comparison's terms.
# ---------------------------------------------------------------------------

# Two values are EQUAL when they differ by no more than this. Stated ONCE.
#
# It matches the existing derivation contract rather than being chosen: `temp_c` is
# derived as `round((temp_f - 32) * 5 / 9, 1)`, so one decimal place is the precision
# the pipeline actually carries. A tighter tolerance would report rounding as
# disagreement (SPEC edge adjacency/R2).
CROSSCHECK_TOLERANCE_F: float = 0.1

# The ONLY seasons the legacy bronze corpus covers, so the only seasons a
# before/after comparison can be made over at all. 2002-2017 and 2025 have no old
# side to compare against; that is a fact about the evidence, not a gap in the check.
COMPARABLE_SEASONS: tuple[int, ...] = (2018, 2019, 2020, 2021, 2022, 2023, 2024)

# The join key. A game id, never a row position (SPEC edge ordering/R2: dismissed
# because the diff joins rather than zips).
JOIN_KEY: str = "game_id"

# THE COMPARISON IS OVER INTERSECTING COLUMNS ONLY, AND SAYING SO IS PART OF THE
# PREDICTION. The seven legacy `weather_raw_bronze_{2018..2024}_season.parquet` files
# carry SEVENTEEN columns; the current WeatherSchema carries twenty-five. Comparing a
# 17-column frame against a 25-column frame without saying so would report eight
# columns as universally "changed" when they simply did not exist before.
INTERSECTING_COLUMNS: tuple[str, ...] = (
    "game_id",
    "forecast_time",
    "game_time",
    "is_outdoor",
    "is_cold",
    "is_windy",
    "is_precipitation",
    "temp_f",
    "temp_c",
    "wind_mph",
    "wind_direction",
    "humidity_pct",
    "precip_prob",
    "precip_mm",
    "condition",
    "condition_code",
    "visibility_km",
)

# The EIGHT columns the legacy side does not have. Seven were added to the schema
# after those files were written (`dew_point_f`, `apparent_temp_f`, `snowfall_cm`,
# `wind_gusts_mph`, `cloud_cover_pct`, `created_at`, `weather_source`); the eighth,
# `weather_coverage`, is added by THIS phase. None of them can disagree, because one
# side of the comparison has nothing to disagree with.
ABSENT_FROM_LEGACY_COLUMNS: tuple[str, ...] = (
    "dew_point_f",
    "apparent_temp_f",
    "snowfall_cm",
    "wind_gusts_mph",
    "cloud_cover_pct",
    "created_at",
    "weather_source",
    "weather_coverage",
)

# Present on both sides but NOT compared, each for a stated reason. `game_id` is the
# join key -- it is equal by construction and reporting it would inflate the agreement
# rate with a tautology. `forecast_time` is the RUN clock: it records when a snapshot
# was taken, so it differs on every row of every re-run and says nothing about the
# weather.
NON_COMPARABLE_INTERSECTING_COLUMNS: tuple[str, ...] = ("game_id", "forecast_time")

COMPARED_COLUMNS: tuple[str, ...] = tuple(
    column
    for column in INTERSECTING_COLUMNS
    if column not in NON_COMPARABLE_INTERSECTING_COLUMNS
)


# ---------------------------------------------------------------------------
# THE PREDICTION.
#
# Three independent causes will move values, and all three populations were measured
# from the pinned feed BEFORE any new weather was fetched.
# ---------------------------------------------------------------------------

EXPECTED_DIFF_SHAPE: dict[str, object] = {
    "hour_fix": {
        "cause": "D33.1-08 -- the venue-local hour, in the venue's own IANA zone",
        "population": (
            "every comparable game in the 2018-2024 legacy bronze whose old row "
            "carries a measurement (is_outdoor True under the OLD venue-level roof "
            "rule)"
        ),
        "measured_size": 1580,
        "population_total_rows": 1942,
        "predicted_effect": (
            "ESSENTIALLY EVERY COMPARABLE ROW DIFFERS. The old code indexed an "
            "Eastern-local hourly array at the UTC hour, so it read FOUR hours late "
            "in daylight time and FIVE in standard time -- afternoon-to-evening for "
            "a 1 pm kickoff. Expect temperature to FALL and wind direction to SHIFT "
            "on most rows, and expect the largest single-row deltas on late-season "
            "STANDARD-TIME games, where the offset is five hours and the diurnal "
            "temperature swing across that window is widest."
        ),
        "direction": "temperature mostly lower; wind direction shifted; magnitude "
        "largest on late-season standard-time games",
    },
    "routing_fix": {
        "cause": "D33.1-06 / R1 -- unconditional stadium_id routing",
        "population": (
            "the 737 weather-applicable games of the 1,153 whose resolved venue "
            "moved across the whole 2002-2025 corpus. Only those inside 2018-2024 "
            "are COMPARABLE against existing bronze, and in that window the "
            "relocations that matter are OAK00 (14 games, 2018-2019), LAX99 (16 "
            "games, 2018-2019) and LAX97 (14 games, 2018-2019) -- 44 games in all."
        ),
        "measured_size": 737,
        "comparable_window_size": 44,
        "comparable_window_stadium_ids": ("OAK00", "LAX99", "LAX97"),
        "predicted_effect": (
            "CATEGORICALLY DIFFERENT VALUES, not shifted ones. Oakland read as Las "
            "Vegas is a different climate 650 km away, not a warmer version of the "
            "same one. Expect these rows to be the largest absolute temperature and "
            "wind differences in the diff, and expect them NOT to share a sign."
        ),
        "direction": "unsigned; a different place, not a different hour",
    },
    "per_game_roof_rule": {
        "cause": "D33.1-07 / R4 -- the GAME's own feed roof decides, not the venue's",
        "population": (
            "the 621 games across the whole corpus at the five retractable-roof "
            "stadiums whose feed roof reads `closed`. Under the OLD venue-level rule "
            "all 926 games at those five venues were fetched, because "
            "_is_outdoor_game('retractable') is True; under the per-game rule only "
            "the 128 `open` ones are. In the 2018-2024 comparable window there are "
            "260 such `closed` games and 41 `open` ones."
        ),
        "measured_size": 621,
        "comparable_window_size": 260,
        "predicted_effect": (
            "THESE ROWS GO FROM A NUMBER TO A NULL TEMPERATURE, and this is the ONLY "
            "category where that happens. It is also a clean, checkable floor: "
            "across all 5,417 known-venue games these 621 are the ONLY place the "
            "feed's per-game roof disagrees with venues.json's roof_type about "
            "is_outdoor. There are zero other disagreements."
        ),
        "direction": "number -> NULL; is_outdoor True -> False",
    },
}


# ---------------------------------------------------------------------------
# THE TWO THINGS THIS PRE-REGISTRATION MUST NOT CLAIM.
#
# Written as prose because a reader has to be able to check, from this file alone,
# that the prediction did not quietly help itself.
#
# 1. IT MUST NOT PREDICT THAT DISAGREEMENTS ARE "CONCENTRATED IN
#    RELOCATED-FRANCHISE HOME GAMES". That is the SPEC's own wording and it is
#    WRONG once the hour fix lands: the hour fix moves nearly EVERY comparable row,
#    so relocation games are the LARGEST differences, not the ONLY ones. A check
#    written against the concentration claim would read as a FAILURE while working
#    exactly as intended.
#
# 2. IT MUST NOT COMPARE ACROSS SCHEMA WIDTHS WITHOUT SAYING SO. The seven legacy
#    season files are a SEVENTEEN-COLUMN schema; the current one is twenty-five.
#    The comparison is over INTERSECTING_COLUMNS only, and the eight columns absent
#    from the legacy side are named above rather than counted as disagreements.
# ---------------------------------------------------------------------------

NON_CLAIMS: tuple[str, ...] = (
    "disagreements are NOT predicted to be concentrated in relocated-franchise home "
    "games; the hour fix moves nearly every comparable row, so relocation games are "
    "the largest differences rather than the only ones",
    "the comparison is over intersecting columns only, because the legacy side is a "
    "17-column schema and the current side is 25; the eight absent columns are named "
    "rather than reported as universal disagreement",
)


# ---------------------------------------------------------------------------
# WHAT THE MEASUREMENT REPORTS. Four quantities, registered before any of them exists.
# ---------------------------------------------------------------------------

REPORTED_QUANTITIES: tuple[str, ...] = (
    "per-season agreement and disagreement counts over COMPARED_COLUMNS, within "
    "CROSSCHECK_TOLERANCE_F",
    "per-home-team agreement and disagreement counts over the same columns",
    "the number-to-NULL transitions, which the per-game roof rule predicts are the "
    "only ones",
    "the archive-versus-forecast provenance probe, with its explicit n",
)


# ---------------------------------------------------------------------------
# THE FOURTH QUANTITY: the archive-versus-forecast provenance probe.
#
# THE OPEN QUESTION, carried forward from 33.1-REVIEWS.md. Does the historical ERA5
# ARCHIVE carry a systematically different bias from the real-time FORECAST feed the
# 2026 live path uses? If it does, a Wave-15 re-fit on corrected history could learn a
# pattern that does not hold live.
#
# THE HONEST RESPONSE IS TO MEASURE IT AND REPORT IT WITH ITS SAMPLE SIZE, AND TO
# ASSERT NOTHING IN ADVANCE. What is pre-registered below is the MEASUREMENT and the
# INTERPRETATION RULE -- not an expectation. There is deliberately no predicted sign
# and no predicted magnitude, because none is known and inventing one would be the
# opposite of what a pre-registration is for.
# ---------------------------------------------------------------------------

ARCHIVE_VERSUS_FORECAST_PROBE_SPEC: dict[str, object] = {
    "question": (
        "does the historical ERA5 archive carry a systematically different bias from "
        "the real-time forecast feed the 2026 live path uses?"
    ),
    "comparable_population": (
        "the ONLY rows in this repository whose provenance is KNOWN to be the "
        "forecast feed: the 14 `2024_W06_*` rows currently in "
        "data/silver/weather.parquet, and the 14 `2025_W05_*` rows in the stale "
        "DuckDB `weather_forecast` table Ruling G identified. AT MOST 28 GAMES."
    ),
    "excluded_population": (
        "the 1,942 legacy 2018-2024 bronze rows are EXCLUDED. They carry no "
        "`weather_source` and RESEARCH established that their provenance cannot be "
        "reconstructed, so including them would answer a DIFFERENT question while "
        "looking like this one."
    ),
    "reported_differences": (
        "archive_temp_f - forecast_temp_f",
        "archive_wind_mph - forecast_wind_mph",
    ),
    "reported_statistics": ("mean signed difference", "standard deviation", "n"),
    "n": None,
    "n_is_none_because": (
        "it is MEASURED by Plan 33.1-06, not predicted here. A pre-registration that "
        "guessed its own sample size would be predicting the answer's precision."
    ),
    "interpretation_rule": (
        "WITH n AT MOST 28 THIS IS A DIRECTIONAL PROBE AND NOT AN ESTIMATE. No bias "
        "correction, no feature change and no Wave-15 instruction may rest on it. Its "
        "ONLY legitimate use is to say whether the question deserves its own "
        "measurement later, on a population built for it. A phase that measured 28 "
        "games and then acted on the sign would be doing exactly what this "
        "repository's pre-registration discipline exists to prevent. This rule is "
        "registered BEFORE any number exists so it cannot be chosen afterwards."
    ),
    "null_case": (
        "if the two sets do not overlap on any game_id -- which is possible, since "
        "the forecast rows may fall outside COMPARABLE_SEASONS -- the probe reports "
        "n=0 and says so. THAT IS A COMPLETE RESULT, not a failure: it means this "
        "repository holds no evidence on the question, which is itself the finding."
    ),
    "measured_by": "Plan 33.1-06 Task 3, recorded in WEATHER_CROSSCHECK_MEASURED",
    "reported_by": "Plan 33.1-11's readout, WITH its n and WITH this rule quoted",
}


# ---------------------------------------------------------------------------
# What the check does with a disagreement.
# ---------------------------------------------------------------------------

# A DISAGREEMENT NEVER FAILS THE RUN. This phase PREDICTS a large one -- the hour fix
# alone moves nearly every comparable row -- so a comparator that raised on
# disagreement would refuse the corpus it exists to validate. The diff is REPORTED,
# per season and per home team, and the readout states whether its shape matched.
DISAGREEMENT_IS_A_FINDING_NOT_A_FAILURE: bool = True
