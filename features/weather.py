"""
Weather Features Calculator

This module calculates weather-based features for NFL games:
- Wind speed features (primary impact factor)
- Temperature features (cold weather effects)
- Precipitation features (rain, snow impact)
- Weather impact scoring for outdoor games only
- Historical weather trend analysis

Weather has the most impact on:
- Kicking accuracy (field goals, extra points)
- Passing efficiency (wind, precipitation)
- Total scoring (cold, wind, precipitation)
- Turnovers (wet conditions)
"""

import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from conf.settings import get_settings
from data.storage import load_dataframe
from utils import get_logger

logger = get_logger(__name__)

# THE SILVER TABLE THIS BUILDER READS. Stated ONCE, because it was wrong twice.
#
# Plan 33.1-02 Ruling G, and it is load-bearing rather than tidy. Both weather
# writers -- `scripts/ingest_weather.py` (the live forecast) and
# `scripts/backfill_historical_weather.py` (the ERA5 archive) -- write the silver
# table `weather`. This module was reading `weather_forecast`. There is no
# `data/silver/weather_forecast.parquet` at all; `data.storage.load_dataframe`
# tries DuckDB FIRST under `source="auto"` and found a stale `weather_forecast`
# table there holding 14 rows, all `2025_W05_*`, 18 columns, no `weather_source`
# -- a DIFFERENT 14 rows from the `2024_W06_*` rows in `data/silver/weather.parquet`.
# That is why RESEARCH found gold's "real" weather rows are 2025 W05 while silver's
# are 2024 W06, and it is the mechanical reason the mild-temperature default was
# reached for 6,485 of 6,499 gold rows.
#
# A perfect 6,499-row silver promotion would change NOTHING in gold without this
# repair, which is why it sits in the tracer: the tracer's whole job is to prove a
# fetched observation reaches a feature frame.
#
# The reads below state `source="parquet"` rather than leaving `auto`, so a DuckDB
# table created later cannot silently win over the parquet the promotion writes.
#
# NOT FIXED HERE, deliberately: `scripts/data_qa.py` still names `weather_forecast`
# at :168, :309, :519, :868 and :1235. That is a QA reporting surface rather than a
# model input, it is outside this phase's named scope, and Plan 33.1-11's readout
# records it as a standing finding.
SILVER_WEATHER_TABLE: str = "weather"

# THE ONE SPELLING OF "WE DO NOT KNOW" IN THIS MODULE (Plan 33.1-04, D33.1-07).
#
# There is deliberately NO numeric stand-in beside it. The SPEC's first
# prohibition is that the mild-temperature default must not be replaced by any
# other number under any name: "a seasonal average or a venue mean is the same
# defect wearing a better label".
NAN: float = float("nan")


class WeatherObservationError(RuntimeError):
    """A weather calculation failed, or a game has no weather record at all.

    DERIVES FROM ``RuntimeError`` ON PURPOSE, and the base class is
    load-bearing rather than incidental. ``scripts/build_features.py`` guards
    every optional silver source with ``_SOURCE_LOAD_ERRORS``, which contains
    ``ValueError``, ``KeyError`` and ``TypeError`` -- so a refusal typed as any
    of those would be caught, logged as a warning and converted into an empty
    frame, which is the very swallow-and-continue shape this refusal exists to
    replace. ``features.elo_features.ProvisionalSnapshotAsTrainingInputError``
    is typed the same way for the same reason, and for the same recorded
    reason: a green build with a silently missing family is indistinguishable
    from one that worked.

    D33.1-07's third state: a caught exception is a BUG, not a state. Recording
    it as missing data is precisely how the mild-temperature default survived
    unnoticed for years.
    """


# ---------------------------------------------------------------------------
# THE COLUMNS EACH BUILDER CONTRIBUTES, STATED ONCE PER BUILDER.
#
# Ruling K1 (Plan 33.1-04): this is a MAPPING keyed by builder identity, not one
# broad constant, because the two builders in this module do not emit the same
# weather columns. `build_weather_features` (the "full" builder, which writes
# silver `weather_features` and therefore feeds gold) emits 47 columns;
# `build_features` (the "compressed" FeatureBuilder-Protocol builder) emits 4.
# A single constant would become an assertion about whichever builder a test
# happened to exercise while the other one silently median-filled.
#
# The families below are the source; the two builder entries and the union are
# DERIVED from them, and both builders assert their own emitted frame against
# their own entry at the end of the build. That is the same
# single-source-with-an-importing-drift-test shape `api/cache.BET_LIST_COLUMNS`
# uses, applied per builder.
# ---------------------------------------------------------------------------

TEMPERATURE_FEATURE_COLUMNS: tuple[str, ...] = (
    "temp_f",
    "apparent_temp_f",
    "temp_hot",
    "temp_warm",
    "temp_mild",
    "temp_cool",
    "temp_cold",
    "temp_very_cold",
    "cold_impact_score",
    "scoring_multiplier",
    "ball_handling_difficulty",
    "heat_impact_score",
)

WIND_FEATURE_COLUMNS: tuple[str, ...] = (
    "wind_mph",
    "wind_calm",
    "wind_moderate",
    "wind_high",
    "wind_severe",
    "wind_impact_score",
    "kicking_difficulty",
    "passing_difficulty",
)

PRECIPITATION_FEATURE_COLUMNS: tuple[str, ...] = (
    "precip_prob",
    "precip_mm",
    "precip_none",
    "precip_light",
    "precip_moderate",
    "precip_heavy",
    "is_snow",
    "is_rain",
    "is_dry",
    "precip_impact_score",
    "turnover_multiplier",
    "passing_efficiency",
)

# THE FOUR PRECIPITATION INTENSITY BANDS, IN ASCENDING ORDER, NAMED ONCE
# (Plan 33-14 Task 1, D33-34(b), .planning/WINDOWS.md row 39).
#
# They already appear inside PRECIPITATION_FEATURE_COLUMNS above, but that tuple
# is the family in EMISSION order and carries no statement about intensity. The
# partition rule below is an ORDER claim -- "take the larger of the two band
# indices" is meaningless without one -- so the order is stated here, once, and
# `_precipitation_bands`' forecast branch reads it from here rather than
# spelling the four strings a third time. A second hand-written list is the
# D30-02 failure mode.
PRECIPITATION_BAND_COLUMNS: tuple[str, ...] = (
    "precip_none",
    "precip_light",
    "precip_moderate",
    "precip_heavy",
)

# THE 0-1 IMPACT SCORE OF EACH BAND, in the same order.
#
# These four values are NOT new. They are the levels `_precipitation_impact`'s
# mm-only branch has always returned -- 0.0, 0.3, 0.6, 1.0 -- lifted out of that
# branch's if-ladder so the forecast branch can return the level of the band it
# actually fired instead of re-deriving one from the raw readings.
#
# WHY IT EXISTS (.planning/WINDOWS.md row 43). Plan 33-14 Task 1 made
# `_precipitation_bands` a partition under PRECIPITATION_PARTITION_RULE -- one
# hot band, chosen by the LARGER of the two band indices. `_precipitation_impact`
# was left carrying the pre-33-14 disjunction, which returned early whenever
# EITHER reading was low: a MINIMUM over the two readings, against the band
# rule's MAXIMUM. Two reducers pulling opposite ways in one call, so the score
# and the one-hot could not agree. Measured over the sixteen-point
# probability-by-rainfall grid, TEN points scored an impact contradicting the
# band the same call assigned; the worst was 9.0 mm of rain -- `precip_heavy`
# -- scoring 0.3, the level a 1.0 mm drizzle gets, because the forecast
# probability was 0.1. Heavy rainfall understated by a low probability was the
# dominant direction.
#
# A LOOKUP RATHER THAN A SECOND IF-LADDER, deliberately. Re-spelling the cut
# points a third time is the D30-02 failure mode; indexing the band the call
# already computed makes disagreement between score and one-hot unrepresentable
# rather than merely tested for.
PRECIPITATION_IMPACT_BY_BAND: tuple[float, ...] = (0.0, 0.3, 0.6, 1.0)

# THE RULE THE FORECAST BRANCH FOLLOWS, IN ONE QUOTABLE SENTENCE.
#
# It is quotable on purpose: a one-hot family that is not a partition is a
# defect that reads as a modelling choice unless the intended rule is written
# down somewhere a reader can compare the code against.
PRECIPITATION_PARTITION_RULE: str = (
    "Give the measured rainfall and the forecast probability each an ordinal "
    "band index over the cut points already in this module, then one-hot "
    "EXACTLY the band named by the LARGER of the two indices; with no "
    "probability the rainfall index stands alone. The probability WIDENS a "
    "band and never adds a level, so wherever the pre-33-14 dual-reading "
    "branch already fired exactly one band the rule fires the same one, and "
    "the inputs where it fired TWO -- a forecast of probability 0.4 with "
    "3.0 mm set both precip_light and precip_moderate -- are the defect "
    "WINDOWS.md row 39 registers."
)

SEVERITY_FEATURE_COLUMNS: tuple[str, ...] = (
    "weather_severity_score",
    "home_weather_advantage",
    "defensive_advantage",
    "rushing_advantage",
    "scoring_reduction",
    "weather_game",
    "extreme_weather",
)

RAW_WEATHER_COLUMNS: tuple[str, ...] = (
    "raw_temp_f",
    "raw_wind_mph",
    "raw_precip_prob",
    "raw_precip_mm",
    "raw_humidity_pct",
    "weather_condition",
)

# The two flags that answer "does weather apply" and "was there an observation".
# They are the ONLY two columns an absent-observation row carries a number in.
WEATHER_FLAG_COLUMNS: tuple[str, ...] = (
    "weather_affects_game",
    "weather_coverage",
)

# THE COVERAGE FLAG, NAMED ONCE (Plan 33.1-07 Task 4, 2026-09-14).
#
# It already appears inside WEATHER_FLAG_COLUMNS above, but a tuple membership
# is not a name a downstream module can import, and three of them now need to
# refer to THIS column specifically rather than to the flag family: the gold
# build's normalization exemption, the constancy measurement, and the width
# tripwire's delta pinning. Naming it here rather than spelling the string in
# each is the same single-source discipline the family tuples themselves follow.
#
# WHY IT NEEDS ITS OWN TREATMENT AT ALL: it is the one column whose LEVELS ARE
# ITS MEANING. `weather_affects_game` is a real predicate about the venue and
# survives a monotone transform; this one answers "is there an observation
# behind this row", and a z-score maps that answer onto window-dependent
# numbers. On the corrected corpus -- constant 1.0, because every game now HAS
# an observation -- the z-score of a constant is exactly 0.0, which is the value
# `_absent_observation_features` writes to mean NO OBSERVATION. The column ended
# up asserting the precise opposite of the truth on all 6,499 gold rows.
WEATHER_COVERAGE_COLUMN: str = "weather_coverage"

# Not a measurement and not a float. Callers that assert "every weather column
# is NaN" must skip it, because `numpy.isfinite` raises on a string rather than
# returning False -- so naming it here is what keeps that assertion honest
# instead of making it silently skip a column it could not evaluate.
WEATHER_NON_NUMERIC_FEATURE_COLUMNS: tuple[str, ...] = ("weather_condition",)

_FULL_BUILDER_COLUMNS: tuple[str, ...] = (
    *TEMPERATURE_FEATURE_COLUMNS,
    *WIND_FEATURE_COLUMNS,
    *PRECIPITATION_FEATURE_COLUMNS,
    *SEVERITY_FEATURE_COLUMNS,
    *RAW_WEATHER_COLUMNS,
    *WEATHER_FLAG_COLUMNS,
)

_COMPRESSED_BUILDER_COLUMNS: tuple[str, ...] = (
    "weather_severity_score",
    "wind_mph",
    "is_precipitation",
    "is_outdoor",
)

WEATHER_FEATURE_COLUMNS_BY_BUILDER: dict[str, tuple[str, ...]] = {
    "full": _FULL_BUILDER_COLUMNS,
    "compressed": _COMPRESSED_BUILDER_COLUMNS,
}

# The union over both builders. `dict.fromkeys` preserves first-seen order and
# de-duplicates the two columns the compressed builder shares with the full one.
WEATHER_FEATURE_COLUMNS: tuple[str, ...] = tuple(
    dict.fromkeys((*_FULL_BUILDER_COLUMNS, *_COMPRESSED_BUILDER_COLUMNS))
)

# The merge keys. Carried by the frame, contributed by neither builder as a
# feature, and therefore excluded from every declaration above.
WEATHER_MERGE_KEY_COLUMNS: tuple[str, ...] = ("game_id", "season", "week")


def _validate_column_declarations() -> None:
    """Fail at IMPORT time if the per-builder split has drifted.

    An explicit raise rather than an ``assert``: ``python -O`` strips asserts,
    and a declaration that silently stops being checked under an optimisation
    flag is not a declaration.
    """
    for key, columns in WEATHER_FEATURE_COLUMNS_BY_BUILDER.items():
        if not columns:
            msg = f"WEATHER_FEATURE_COLUMNS_BY_BUILDER[{key!r}] is empty"
            raise ValueError(msg)
        if len(set(columns)) != len(columns):
            msg = f"WEATHER_FEATURE_COLUMNS_BY_BUILDER[{key!r}] has duplicates"
            raise ValueError(msg)
    if WEATHER_COVERAGE_COLUMN not in WEATHER_FLAG_COLUMNS:
        msg = (
            f"WEATHER_COVERAGE_COLUMN {WEATHER_COVERAGE_COLUMN!r} is not a "
            "member of WEATHER_FLAG_COLUMNS. The single-name constant exists so "
            "downstream modules can refer to THIS column rather than spelling "
            "the string; a name that is not in the family is a second spelling."
        )
        raise ValueError(msg)
    if len(set(PRECIPITATION_BAND_COLUMNS)) != len(PRECIPITATION_BAND_COLUMNS):
        msg = "PRECIPITATION_BAND_COLUMNS has duplicates"
        raise ValueError(msg)
    stray_bands = [
        column
        for column in PRECIPITATION_BAND_COLUMNS
        if column not in PRECIPITATION_FEATURE_COLUMNS
    ]
    if stray_bands:
        msg = (
            f"PRECIPITATION_BAND_COLUMNS names {stray_bands}, which are not "
            "members of PRECIPITATION_FEATURE_COLUMNS. The ordered band tuple "
            "exists so the partition rule has an order to refer to; a band "
            "that is not in the emitted family is a second spelling, which is "
            "the D30-02 failure mode this declaration block exists to catch."
        )
        raise ValueError(msg)
    union = set(_FULL_BUILDER_COLUMNS) | set(_COMPRESSED_BUILDER_COLUMNS)
    if union != set(WEATHER_FEATURE_COLUMNS):
        msg = (
            "WEATHER_FEATURE_COLUMNS is not the union of the two builder "
            f"entries; symmetric difference {sorted(union ^ set(WEATHER_FEATURE_COLUMNS))}"
        )
        raise ValueError(msg)


_validate_column_declarations()


# ---------------------------------------------------------------------------
# THE DATED 2026 GOLD-DEFAULT SWITCH (Plan 33-09 Task 5, D33-25).
#
# APPENDED by Plan 33-09 Task 5. Nothing above this line was edited.
#
# WHAT IT DOES. For a season named in the switch, both builders in this module
# write the ABSENT-OBSERVATION weather family into the feature frame -- every
# weather column NULL, with the coverage flag at 0.0 -- no matter what the
# silver weather table holds for that game. For every other season nothing
# changes at all: silver is consumed exactly as it was before this block
# existed, and a frame-identity test on 2025 asserts that.
#
# WHY. This is a TRAIN/SERVE AGREEMENT decision and NOT an accuracy one. The
# deployed O/U artifact was fitted on gold whose weather columns do not vary at
# all inside its training window (the measurement below). A winsorization bound
# fitted on a constant collapses, so every live 2026 weather value would land
# outside it -- out-of-distribution columns entering a deployed model days
# before the season this milestone exists to measure. Holding gold at its
# default buys the time to re-fit rather than the right never to.
#
# WHAT IT IS NOT. It is NOT a restoration of the deleted mild-temperature
# default. Plan 33.1-04 removed `temp_f: 65.0` and its one-hot family, and
# D33.1-07 leaves exactly ONE spelling of "we do not know" in this module. The
# SPEC's first prohibition is that no other number may take the old default's
# place under any name -- "a seasonal average or a venue mean is the same defect
# wearing a better label". So the held state is NULL, and the row SAYS SO: the
# coverage flag reads 0.0 on a held row, which keeps it distinguishable in the
# data from an observed one and from a dome.
#
# IT DOES NOT REACH SILVER. `scripts/ingest_weather.py` carries no reference to
# this switch and is not meant to. The live forecast values accumulate from week
# one exactly as they would without it -- which is the point, because the re-fit
# that flips this switch needs them as its input.
#
# THE RECORD, and the record is what makes this honest rather than a dodge: the
# default is EXPLICIT, DATED and DISCLOSED in committed source, which is the
# exact opposite of the Elo case where the imputation was silent and unintended.
#
#   MEASURED on 2026-09-12 by Plan 33-09 Task 3, read-only, and recorded in
#   `tests/phase33_state.GOLD_WEATHER_CONSTANCY_MEASUREMENT`: 45 of 46 gold
#   weather columns are EXACTLY CONSTANT in the ATS train window 2015-2019
#   (1,335 rows), in the WP/OU train window 2018-2019 (534 rows) AND in the
#   2021-2024 gate holdout (1,139 rows); `venue_cold_climate` is the only column
#   that varies in any of them. `raw_temp_f` equals the imputed 65.0 default on
#   6,485 of 6,499 gold rows (99.7846%), and inside each of those three windows
#   the imputation is TOTAL. The same measurement NARROWED the blast radius, and
#   that is recorded rather than quietly dropped: WP and ATS consume ZERO
#   weather features, so this is 17 columns entering ONE deployed model (the
#   v1.0 pre-Elo O/U artifact), not 33 entering three.
#
#   RULED by the OWNER on 2026-09-14, at Plan 33-09's Task-4 blocking
#   checkpoint, against that re-derived constancy measurement and BEFORE this
#   switch was written: APPROVED, WITH ONE CHANGE. The plan asked for Phase 37
#   as the flip condition. The owner set it to Phase 33 Wave 15's re-fit
#   instead, in their own words -- "The only justification for holding 2026
#   weather back was that the O/U model had never seen weather vary. Phase 33.1
#   fixed the historical record, so that reason expires at the next re-fit
#   rather than a future phase." The hold is a BRIDGE, not a season-long policy,
#   and the flip condition below is written to expire it at the next re-fit.
#
# ---------------------------------------------------------------------------
# THE CONDITION WAS MET, AND THE HOLD IS REMOVED (Plan 33-15 Task 4, 2026-09-14).
#
# EVERYTHING ABOVE THIS LINE IS THE RECORD OF THE RULING THAT CREATED THE
# BRIDGE AND IS RETAINED UNEDITED. It is why the hold existed, and deleting it
# would leave a removal nobody could account for. What follows is the record of
# the hold ENDING.
#
# WHAT MET IT. Phase 33 Wave 15 re-fit all three targets on gold rebuilt from
# the corrected historical weather record, and the owner promoted all three on
# 2026-09-14:
#
#     wp   wp_20260824_113325  -> wp_20260914_221745
#     ats  ats_20260605_220128 -> ats_20260914_221751
#     ou   ou_20260326_163930  -> ou_20260914_221756
#
# Every one of those artifacts carries the
# `trained_on_real_weather_generation` marker naming gold generation
# `2a6ad6de...`, and each SELECTS real weather columns where the incumbent it
# replaced selected none or selected from a fabricated constant: WP 0 -> 3,
# ATS 0 -> 6, O/U 17 -> 6.
#
# RULED by the OWNER on 2026-09-14, in their own terms: the switch existed only
# because the models had never seen weather vary, having been trained on a
# record where every game was 65 F; the three models now promoted learned from
# the corrected historical weather, so feeding them the real 2026 forecast is
# what they expect. The owner's `promote-all-three` disposition also removed the
# last argument for keeping it -- under a split promotion there would still have
# been a production model fitted on fabricated weather, and there is not one now.
#
# WHY THE SET IS EMPTIED RATHER THAN THE SYMBOL DELETED. The flip condition says
# the switch "must be REMOVED, not re-dated", and an EMPTY held-season set is
# that removal: no season is held, `_game_is_held_at_gold_default` returns False
# for every game, and gold consumes silver exactly as it did before the switch
# was written. Deleting the two names instead would take the dated record above
# with them and would break the derivation
# `tests/unit/test_weather_bridge_expiry.defaulted_weather_columns` performs
# from this module's own held-state producer -- trading a recorded removal for
# an unrecorded one.
#
# THE CONSEQUENCE, PLAINLY: from the next gold build, season 2026 carries REAL
# FORECAST WEATHER into the gold weather family. That is the live path Plan
# 33-18 runs.
# ---------------------------------------------------------------------------

WEATHER_GOLD_DEFAULT_SEASONS: frozenset[int] = frozenset()

WEATHER_GOLD_DEFAULT_FLIP_CONDITION: str = (
    "MET AND REMOVED on 2026-09-14 by Phase 33 Wave 15. The condition was: "
    "remove 2026 from the held-season set when Phase 33 Wave 15's re-fit has "
    "trained the deployed artifacts on gold rebuilt from the corrected "
    "historical weather record (Plan 33.1-07's weather rung), and not before. "
    "It was met by the owner-ruled promotion of wp_20260914_221745, "
    "ats_20260914_221751 and ou_20260914_221756, each carrying the "
    "trained_on_real_weather_generation marker for gold generation 2a6ad6de, "
    "and each selecting real weather columns where its incumbent selected none "
    "or selected from a fabricated constant. The deployed models HAVE now seen "
    "weather vary, so the sole reason for the hold has expired and the "
    "held-season set is EMPTY rather than re-dated. From the next gold build, "
    "season 2026 carries real forecast weather."
)


def _resolve_season(game: Any, game_id: Any) -> int | None:
    """The season of *game*, from its column or, failing that, its ``game_id``.

    The fallback is not belt-and-braces. ``get_features_for_game`` builds its
    one-row frame with ``season: 0`` and a real ``game_id``, so a resolver that
    read the column alone would return nothing there -- and a season that cannot
    be resolved is a season that is NOT held. That fails OPEN on the
    single-game serving path, which is precisely the path that answers a live
    2026 prediction.

    Args:
        game: The games-frame row, or any mapping carrying ``season``.
        game_id: The game identifier, whose first underscore-delimited field is
            the four-digit season.

    Returns:
        The season, or ``None`` when neither source can supply one.
    """
    season = game.get("season") if hasattr(game, "get") else None
    # WRITTEN AS TYPE TESTS RATHER THAN A `try/except`, for the reason
    # `_is_missing` records: every `except` branch in this module raises, and a
    # probe that swallows a TypeError is the shape D33.1-07 removed.
    # `float.is_integer` is what rejects NaN and infinity here without one.
    if isinstance(season, int | float | np.integer | np.floating) and not isinstance(
        season, bool
    ):
        as_float = float(season)
        if as_float.is_integer() and as_float > 0:
            return int(as_float)
    if isinstance(season, str) and season.isdigit() and int(season) > 0:
        return int(season)
    prefix = str(game_id).split("_", 1)[0]
    if len(prefix) == 4 and prefix.isdigit():
        return int(prefix)
    return None


def _game_is_held_at_gold_default(game: Any, game_id: Any) -> bool:
    """Whether the gold weather family for *game* is held at its default.

    Reads the module constant at CALL time rather than binding it at import, so
    a test can empty the switch and drive the same builder over the same silver
    frame as a controlled A/B.
    """
    season = _resolve_season(game, game_id)
    return season is not None and season in WEATHER_GOLD_DEFAULT_SEASONS


def _is_missing(value: Any) -> bool:
    """True when a measurement is genuinely ABSENT (``None``, NaN, ``pd.NA``).

    Deliberately NOT ``not value``. The old code wrote ``weather_data.get(
    "wind_mph", 0.0) or 0.0``, which maps a real, measured calm of ``0.0`` and
    an absent reading onto the same number -- and then reports the absent one
    as ``wind_calm: 1.0``. Zero is a measurement; absence is not.
    """
    if value is None:
        return True
    if isinstance(value, str):
        # A condition string is never a missing measurement, and `pd.isna`
        # would be asked a question about it that it cannot answer usefully.
        return False
    missing = pd.isna(value)
    if isinstance(missing, bool | np.bool_):
        return bool(missing)
    # An array-like was handed in. That is not a scalar measurement, so it is
    # not an absent one either. Written as a type test rather than a
    # `try/except` on purpose: every `except` branch in this module raises, and
    # a probe that swallows a TypeError is the shape this plan is removing.
    return False


def _observation_failure(
    family: str, weather_data: dict[str, Any], error: Exception
) -> WeatherObservationError:
    """Build the ONE message shape all three family handlers raise.

    Three handlers, three raises, one message shape -- so a reader who sees one
    of them has seen all three, and a partial removal is visible as a message
    that does not match.
    """
    game_id = weather_data.get("game_id", "<unknown game_id>")
    return WeatherObservationError(
        f"{family} feature calculation FAILED for game {game_id!r}: {error!r}. "
        "A caught exception is a BUG, not a state (D33.1-07). Recording it as "
        "missing data is precisely how the mild-temperature default survived "
        "unnoticed for years and reached 6,485 of 6,499 gold rows. Fix the "
        "payload or the calculation -- do not substitute a default family."
    )


def _or_null(value: Any) -> Any:
    """Return *value*, or NaN when it is absent -- never a substitute reading."""
    return NAN if _is_missing(value) else value


def _row_is_covered(weather_data: dict[str, Any]) -> bool:
    """Whether this silver row carries an OBSERVATION.

    A row with no ``weather_coverage`` column at all is treated as covered.
    That is not a silent assumption about missing data: before Plan 33.1-02
    added the flag, every row in the table WAS an observation (or a dome
    record), because the state this flag names could not be written. A legacy
    frame therefore reads exactly as it always did, and a frame written by the
    current backfill reads the flag.
    """
    if "weather_coverage" not in weather_data:
        return True
    coverage = weather_data.get("weather_coverage")
    if _is_missing(coverage):
        return True
    return bool(coverage)


def _no_weather_row(game_id: str) -> WeatherObservationError:
    """The refusal that replaced both "no weather row" default blocks.

    After the historical backfill every game HAS a row, so the reachability of
    this path is itself the anomaly.
    """
    return WeatherObservationError(
        f"game {game_id!r} has NO row in the silver `{SILVER_WEATHER_TABLE}` "
        "table. This used to produce a default row carrying `temp_f: 65.0`, "
        '`is_outdoor: False` and `condition: "Clear"` -- which made a missing '
        "record indistinguishable from a dome BY CONSTRUCTION, and is how the "
        "default reached 6,485 of 6,499 gold rows. Run the historical weather "
        "backfill (`scripts/backfill_historical_weather.py`) so the game has a "
        "real record, or an explicit absent-observation record; do not restore "
        "a default."
    )


def _assert_builder_columns(features_df: pd.DataFrame, builder: str) -> None:
    """Fail if *builder* emitted anything other than its declared entry.

    Ruling K1's drift guard, run PER BUILDER at the end of each build. An empty
    frame carries no columns at all and is skipped: a build over zero games has
    nothing to disagree with, and failing there would turn an empty input into
    a declaration error.
    """
    if features_df.empty:
        return
    emitted = set(features_df.columns) - set(WEATHER_MERGE_KEY_COLUMNS)
    declared = set(WEATHER_FEATURE_COLUMNS_BY_BUILDER[builder])
    if emitted != declared:
        msg = (
            f"the {builder!r} weather builder emitted columns that do not match "
            f"WEATHER_FEATURE_COLUMNS_BY_BUILDER[{builder!r}]. "
            f"Emitted-not-declared: {sorted(emitted - declared)}. "
            f"Declared-not-emitted: {sorted(declared - emitted)}. "
            "The declaration and the builder are single-sourced on purpose; "
            "update the family tuples rather than hand-editing one side."
        )
        raise ValueError(msg)


class WeatherFeaturesCalculator:
    """
    Calculate weather features for NFL games.

    Features calculated:
    - Wind speed features (primary factor for kicking/passing)
    - Temperature features (cold weather effects)
    - Precipitation features (rain/snow impact)
    - Weather severity scoring
    - Historical weather trends for venues
    - Weather impact limited to outdoor/retractable venues
    """

    def __init__(self):
        """Initialize weather features calculator."""
        self.settings = get_settings()

        # Weather thresholds from configuration
        self.wind_threshold = (
            self.settings.config.models.features.weather.wind_threshold
        )  # 12 MPH
        self.temp_threshold = (
            self.settings.config.models.features.weather.temp_threshold
        )  # 32°F

        # Additional weather thresholds
        self.severe_wind_threshold = 20.0  # MPH
        self.very_cold_threshold = 20.0  # °F
        self.heavy_precip_threshold = 5.0  # mm

    def _calculate_apparent_temperature(
        self, temp_f: float, wind_mph: float, humidity_pct: float
    ) -> float:
        """
        Calculate apparent temperature (wind chill or heat index).

        Research shows apparent temperature is better than raw temp because it folds
        in wind/humidity effects that impact player performance.

        Args:
            temp_f: Temperature in Fahrenheit
            wind_mph: Wind speed in MPH
            humidity_pct: Relative humidity percentage

        Returns:
            Apparent temperature in Fahrenheit
        """
        if temp_f <= 50 and wind_mph >= 3:
            # Wind chill formula (applicable when temp <= 50°F and wind >= 3 MPH)
            wind_chill = (
                35.74
                + 0.6215 * temp_f
                - 35.75 * (wind_mph**0.16)
                + 0.4275 * temp_f * (wind_mph**0.16)
            )
            return wind_chill
        if temp_f >= 80 and humidity_pct >= 40:
            # Heat index formula (applicable when temp >= 80°F and humidity >= 40%)
            hi = (
                -42.379
                + 2.04901523 * temp_f
                + 10.14333127 * humidity_pct
                - 0.22475541 * temp_f * humidity_pct
                - 0.00683783 * temp_f * temp_f
                - 0.05481717 * humidity_pct * humidity_pct
                + 0.00122874 * temp_f * temp_f * humidity_pct
                + 0.00085282 * temp_f * humidity_pct * humidity_pct
                - 0.00000199 * temp_f * temp_f * humidity_pct * humidity_pct
            )
            return hi
        # No adjustment needed for moderate conditions
        return temp_f

    def calculate_wind_features(self, weather_data: dict[str, Any]) -> dict[str, float]:
        """
        Calculate wind-related features.

        Wind is the primary weather factor affecting:
        - Field goal accuracy (significant impact >12 MPH)
        - Passing effectiveness (dropoff >15 MPH)
        - Punt/kickoff distance and accuracy

        Args:
            weather_data: Weather data dictionary

        Returns:
            Dictionary with wind features
        """
        try:
            raw_wind = weather_data.get("wind_mph")
            if _is_missing(raw_wind):
                # A NULL wind reading is not a calm day. D33.1-07: anything
                # derived from a NULL observation is itself NULL.
                return dict.fromkeys(WIND_FEATURE_COLUMNS, NAN)

            wind_mph = float(raw_wind)

            # Basic wind features
            wind_features = {
                "wind_mph": float(wind_mph),
                "wind_calm": 1.0 if wind_mph <= 5.0 else 0.0,
                "wind_moderate": 1.0 if 5.0 < wind_mph <= self.wind_threshold else 0.0,
                "wind_high": 1.0
                if self.wind_threshold < wind_mph <= self.severe_wind_threshold
                else 0.0,
                "wind_severe": 1.0 if wind_mph > self.severe_wind_threshold else 0.0,
            }

            # Wind impact scoring (0-1 scale)
            # Based on research showing kicking accuracy drops significantly >12 MPH
            if wind_mph <= 5.0:
                wind_impact = 0.0  # No impact
            elif wind_mph <= self.wind_threshold:
                wind_impact = 0.2  # Minimal impact
            elif wind_mph <= self.severe_wind_threshold:
                wind_impact = 0.6  # Moderate impact
            else:
                wind_impact = 1.0  # Severe impact

            wind_features["wind_impact_score"] = wind_impact

            # Kicking difficulty multiplier
            # Field goal accuracy drops ~5% per 5 MPH over 10 MPH
            if wind_mph <= 10.0:
                kicking_difficulty = 1.0
            else:
                # Scale from 1.0 to 2.0 based on wind speed
                kicking_difficulty = 1.0 + min((wind_mph - 10.0) / 20.0, 1.0)

            wind_features["kicking_difficulty"] = kicking_difficulty

            # Passing difficulty (affects completion percentage and accuracy)
            if wind_mph <= 8.0:
                passing_difficulty = 1.0
            else:
                # Passing becomes noticeably harder >15 MPH
                passing_difficulty = 1.0 + min((wind_mph - 8.0) / 17.0, 1.0)

            wind_features["passing_difficulty"] = passing_difficulty

            return wind_features

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Wind feature calculation failed",
                game_id=weather_data.get("game_id"),
                error=str(e),
            )
            raise _observation_failure("wind", weather_data, e) from e

    def _indoor_wind_features(self) -> dict[str, float]:
        """The INDOOR wind state. Indoors the wind genuinely IS zero.

        These values are a TRUE statement about a covered game, in Ruling J's
        own terms: the wind column answers "what was the wind", and indoors the
        answer is calm. That is why D33.1-07 keeps indoor wind at a genuine
        ``0.0`` rather than calling it NULL.

        THIS IS NOT A FALLBACK FOR A FAILED CALCULATION. It was called
        ``_default_wind_features`` and was reached from an ``except`` branch,
        where it invented a perfectly calm day out of a malformed payload -- in
        the family the deployed O/U model reads most heavily. The rename is the
        guard: while it is called ``_default_*`` the next reader wires the next
        ``except`` branch to it by analogy. It has exactly ONE call site, the
        indoor branch, and a test asserts that count.
        """
        return {
            "wind_mph": 0.0,
            "wind_calm": 1.0,
            "wind_moderate": 0.0,
            "wind_high": 0.0,
            "wind_severe": 0.0,
            "wind_impact_score": 0.0,
            "kicking_difficulty": 1.0,
            "passing_difficulty": 1.0,
        }

    def calculate_temperature_features(
        self, weather_data: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate temperature-related features based on research findings.

        Research-backed temperature effects:
        - 55-85°F: No effect on scoring
        - 25-55°F: ~5% scoring reduction (passing efficiency drops)
        - <25°F or >85°F: ~8% scoring reduction (significant effects)

        Uses apparent temperature (wind chill/heat index) for more accurate
        impact assessment than raw temperature alone.

        Key findings:
        - Wind is often larger driver than temperature
        - Cold effects mainly impact passing efficiency and kicking
        - Extreme temperature thresholds are required for measurable effects

        Args:
            weather_data: Weather data dictionary with temp_f, wind_mph, humidity_pct

        Returns:
            Dictionary with temperature features including apparent_temp_f
        """
        try:
            raw_temp = weather_data.get("temp_f")
            raw_wind = weather_data.get("wind_mph")
            raw_humidity = weather_data.get("humidity_pct")

            if _is_missing(raw_temp):
                # NO TEMPERATURE MEANS NO TEMPERATURE FAMILY. The deleted
                # `_default_temperature_features` returned 65.0 here, plus
                # temp_warm 1.0, cold_impact_score 0.0, scoring_multiplier 1.0
                # and ball_handling_difficulty 1.0 -- the same fabrication in
                # one-hot clothing, which is why deleting the literal alone
                # would not have been the fix (RESEARCH 8.2).
                return dict.fromkeys(TEMPERATURE_FEATURE_COLUMNS, NAN)

            temp_f = float(raw_temp)

            # Basic temperature categories
            temp_features = {
                "temp_f": temp_f,
                "temp_hot": 1.0 if temp_f >= 85.0 else 0.0,
                "temp_warm": 1.0 if 70.0 <= temp_f < 85.0 else 0.0,
                "temp_mild": 1.0 if 50.0 <= temp_f < 70.0 else 0.0,
                "temp_cool": 1.0 if self.temp_threshold <= temp_f < 50.0 else 0.0,
                "temp_cold": 1.0
                if self.very_cold_threshold <= temp_f < self.temp_threshold
                else 0.0,
                "temp_very_cold": 1.0 if temp_f < self.very_cold_threshold else 0.0,
            }

            # Calculate apparent temperature (wind chill / heat index).
            #
            # The old code substituted `wind_mph or 0` and `humidity_pct or 50`
            # here. Those are stand-ins for a measurement, and `or 50` in
            # particular invents a humidity. What replaces them is NARROWER
            # than refusing on any null: the apparent temperature is NULL only
            # when the input it actually USES is absent. Below 50F the wind
            # chill needs wind; at or above 80F the heat index needs humidity;
            # in between the apparent temperature IS the temperature and no
            # second measurement is consulted at all.
            apparent_temp = self._apparent_temperature_or_null(
                temp_f, raw_wind, raw_humidity
            )
            temp_features["apparent_temp_f"] = (
                NAN if _is_missing(apparent_temp) else round(apparent_temp, 1)
            )

            # Research-based temperature impact (using apparent temperature).
            # Both scores below are pure functions of the apparent temperature,
            # so both are NULL when it is (D33.1-07's inheritance rule).
            if _is_missing(apparent_temp):
                cold_impact = NAN
                scoring_impact = NAN
            else:
                # Cold weather impact (0-1 scale) - only significant below 25°F
                if apparent_temp >= 25.0:
                    cold_impact = 0.0  # No cold impact above 25°F
                else:
                    # Linear scaling from 25°F to 0°F
                    cold_impact = min(1.0, (25.0 - apparent_temp) / 25.0)

                # Research-based scoring impact (piecewise function)
                # Based on data from Sharp Football Analysis and others
                if 55 <= apparent_temp <= 85:
                    scoring_impact = 1.00  # No effect in normal range
                elif 25 <= apparent_temp < 55:
                    scoring_impact = 0.95  # 5% reduction in cool weather
                else:
                    scoring_impact = 0.92  # 8% reduction in extreme weather

            temp_features["cold_impact_score"] = cold_impact
            temp_features["scoring_multiplier"] = scoring_impact

            # Ball handling difficulty (fumbles increase in cold)
            if temp_f >= 40.0:
                handling_difficulty = 1.0
            else:
                # Fumbles increase ~10% per 10°F below 40°F
                handling_difficulty = 1.0 + (40.0 - temp_f) * 0.01

            temp_features["ball_handling_difficulty"] = min(handling_difficulty, 1.5)

            # Heat impact (extreme heat also affects performance)
            if temp_f <= 90.0:
                heat_impact = 0.0
            else:
                # Heat exhaustion becomes factor >90°F
                heat_impact = min((temp_f - 90.0) / 20.0, 1.0)

            temp_features["heat_impact_score"] = heat_impact

            return temp_features

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Temperature feature calculation failed",
                game_id=weather_data.get("game_id"),
                error=str(e),
            )
            raise _observation_failure("temperature", weather_data, e) from e

    def _apparent_temperature_or_null(
        self, temp_f: float, raw_wind: Any, raw_humidity: Any
    ) -> float:
        """Apparent temperature, or NaN when the input it USES is absent.

        Args:
            temp_f: The measured temperature, already known to be present.
            raw_wind: The raw wind reading, possibly absent.
            raw_humidity: The raw humidity reading, possibly absent.

        Returns:
            The wind chill, the heat index, the temperature itself, or NaN.
        """
        # THE UNUSED ARGUMENT IS NAN, NOT A PLAUSIBLE NUMBER (code review WR-08).
        #
        # These two calls used to pass `50.0 if _is_missing(raw_humidity) else ...`
        # and `0.0 if _is_missing(raw_wind) else ...` -- stand-ins for a measurement,
        # in the module whose entire point is that it contains exactly ONE spelling of
        # "we do not know", and `or 50` in particular is named four screens up as the
        # defect being removed. They were inert, because
        # `_calculate_apparent_temperature`'s wind-chill branch never reads humidity
        # and its heat-index branch never reads wind -- but that safety was a coupling
        # to ANOTHER function's internal branch thresholds, not a property of this
        # one. If the wind chill ever gained a humidity term, a fabricated 50% would
        # have become a model input silently.
        #
        # NAN is now passed instead, and the formula's own guards decide: `NAN >= 3`
        # and `NAN >= 40` are both False, so an absent reading can only ever fall
        # THROUGH a branch, never satisfy one. Verified exhaustively over the
        # temperature/wind/humidity grid: every output is bit-identical to the
        # stand-in version, so no gold value moves.
        if temp_f <= 50:
            # The wind-chill branch consults wind, and `wind >= 3` cannot be
            # decided without it.
            if _is_missing(raw_wind):
                return NAN
            return self._calculate_apparent_temperature(
                temp_f,
                float(raw_wind),
                NAN if _is_missing(raw_humidity) else float(raw_humidity),
            )
        if temp_f >= 80:
            # The heat-index branch consults humidity, and `humidity >= 40`
            # cannot be decided without it.
            if _is_missing(raw_humidity):
                return NAN
            return self._calculate_apparent_temperature(
                temp_f,
                NAN if _is_missing(raw_wind) else float(raw_wind),
                float(raw_humidity),
            )
        # Between 50F and 80F neither formula applies and the apparent
        # temperature IS the temperature, so no second measurement is read.
        return temp_f

    def _indoor_temperature_features(self) -> dict[str, float]:
        """The INDOOR temperature state: NULL for every temperature column.

        Replaces the deleted ``_default_temperature_features``. There is no
        outdoor temperature indoors, and a band with no temperature to band has
        no answer -- so every one of the twelve columns is NaN rather than the
        old 65.0 / ``temp_warm: 1.0`` family. The four impact and multiplier
        columns are included for the same reason: each is a pure function of
        temperature, and they are exactly the stand-ins that survived deleting
        the literal (RESEARCH 8.2).
        """
        return dict.fromkeys(TEMPERATURE_FEATURE_COLUMNS, NAN)

    def _absent_observation_features(
        self, *, is_outdoor: bool
    ) -> dict[str, float | None]:
        """Every full-builder weather column NULL, bar the two flags.

        The state D33.1-07 calls "venue resolves, observation absent". Before
        the coverage flag existed this state was unrepresentable: a missing
        record was filled with ``temp_f: 65.0`` AND ``is_outdoor: False``, so it
        was indistinguishable from a dome BY CONSTRUCTION.

        Args:
            is_outdoor: Whether weather APPLIES to this game. What is absent is
                the observation, not the applicability -- which is exactly the
                distinction ``weather_coverage`` records.

        Returns:
            Every full-builder column at NaN (``weather_condition`` at None,
            because it is a string column), with ``weather_affects_game``
            derived from *is_outdoor* and ``weather_coverage`` at 0.0.
        """
        features: dict[str, float | None] = dict.fromkeys(_FULL_BUILDER_COLUMNS, NAN)
        for column in WEATHER_NON_NUMERIC_FEATURE_COLUMNS:
            features[column] = None
        features["weather_affects_game"] = 1.0 if is_outdoor else 0.0
        features["weather_coverage"] = 0.0
        return features

    def calculate_precipitation_features(
        self, weather_data: dict[str, Any]
    ) -> dict[str, float]:
        """
        Calculate precipitation-related features.

        Precipitation effects:
        - Increased turnovers (wet ball, slippery field)
        - Reduced passing accuracy
        - Advantage to running game
        - Reduced scoring overall

        Args:
            weather_data: Weather data dictionary

        Returns:
            Dictionary with precipitation features
        """
        try:
            raw_prob = weather_data.get("precip_prob")
            raw_mm = weather_data.get("precip_mm")
            condition = (weather_data.get("condition") or "").lower()
            raw_temp = weather_data.get("temp_f")

            if _is_missing(raw_mm):
                # THE GATE NARROWED, and what it now refuses on is the
                # MEASUREMENT (Plan 33.1-07 Task 4, 2026-09-14).
                #
                # It used to refuse when EITHER reading was absent. But
                # `precip_prob` is a FORECAST probability, and the corpus this
                # phase built is ERA5 REANALYSIS -- a reanalysis states what
                # happened, not what was forecast to happen. `COVERAGE.md`
                # records `precipitation_probability` as a deliberate opt-out
                # because the archive does not carry it, and
                # `scripts/ingest_weather.fetch_game_weather` writes
                # `precip_prob: None` for exactly that reason. So the old gate
                # fired on EVERY outdoor game in the corrected corpus and threw
                # away `precip_mm` -- the rain that ACTUALLY FELL -- because a
                # number nobody can ever supply was absent.
                #
                # MEASURED on production gold before the fix: `raw_precip_mm`
                # non-null on all 6,499 rows, `precip_mm` non-null on 1,652 --
                # exactly the indoor games, which take the dome branch and never
                # reach here. All 4,847 outdoor games carried NULL across twelve
                # precipitation columns, and the seven composites inherited it.
                #
                # THE GATE DID NOT DISAPPEAR. Without the MEASUREMENT there is
                # nothing to band, score or multiply, and the whole family is
                # still NULL. That is the difference between narrowing this gate
                # and restoring the pre-33.1 `or 0.0`, which reported an absent
                # reading as a dry day.
                return dict.fromkeys(PRECIPITATION_FEATURE_COLUMNS, NAN)

            precip_mm = float(raw_mm)

            # NO PROBABILITY IS INVENTED. Where the archive gave none, the
            # column stays NULL and every derived quantity is computed from the
            # millimetres alone. Inferring a probability from an observed
            # rainfall would be the same fabrication this phase exists to
            # delete, wearing a statistician's hat (SPEC prohibition 1).
            probability_known = not _is_missing(raw_prob)
            precip_prob: float | None = float(raw_prob) if probability_known else None

            # Basic precipitation features
            precip_features = {
                "precip_prob": float(precip_prob) if precip_prob is not None else NAN,
                "precip_mm": float(precip_mm),
                **self._precipitation_bands(precip_mm, precip_prob),
            }

            # Precipitation TYPE needs the temperature, and the temperature is
            # the one reading that can be absent while the rainfall measurement
            # is present. `or 40.0` used to supply it -- a 40F stand-in that
            # decided rain-versus-snow for a game nobody measured. Without it,
            # snow-versus-rain is unknown, so those three one-hots and the impact
            # score that reads them are NULL; the bands and the two multipliers
            # below do not consult temperature and stay real.
            temperature_known = not _is_missing(raw_temp)
            temp_f = float(raw_temp) if temperature_known else NAN

            # "Something fell" from the readings that EXIST. With a probability
            # this is the original disjunction unchanged; without one the
            # probability disjunct is simply not applied -- it is not replaced by
            # a zero-probability assumption, and the millimetres carry the test.
            measurable_precipitation = (
                precip_prob is not None and precip_prob > 0.3
            ) or precip_mm > 0.5

            is_snow = temperature_known and (
                temp_f <= 35.0 and (measurable_precipitation or "snow" in condition)
            )
            is_rain = temperature_known and (
                temp_f > 35.0 and (measurable_precipitation or "rain" in condition)
            )

            if temperature_known:
                precip_features.update(
                    {
                        "is_snow": 1.0 if is_snow else 0.0,
                        "is_rain": 1.0 if is_rain else 0.0,
                        "is_dry": 1.0 if not (is_snow or is_rain) else 0.0,
                    }
                )
            else:
                precip_features.update({"is_snow": NAN, "is_rain": NAN, "is_dry": NAN})

            # Precipitation impact scoring
            precip_impact = self._precipitation_impact(precip_mm, precip_prob)

            # Snow has different impact than rain
            if not temperature_known:
                # The snow multiplier below cannot be applied or ruled out, so
                # the impact score is unknown rather than un-multiplied.
                precip_features["precip_impact_score"] = NAN
            else:
                if is_snow:
                    precip_impact *= 1.2  # Snow generally worse than rain
                precip_features["precip_impact_score"] = min(precip_impact, 1.0)

            # Turnover multiplier (wet conditions increase fumbles/interceptions)
            precip_features["turnover_multiplier"] = self._turnover_multiplier(
                precip_mm, precip_prob
            )

            # Passing efficiency reduction
            precip_features["passing_efficiency"] = self._passing_efficiency(
                precip_mm, precip_prob
            )

            return precip_features

        except (ValueError, KeyError, TypeError) as e:
            logger.error(
                "Precipitation feature calculation failed",
                game_id=weather_data.get("game_id"),
                error=str(e),
            )
            raise _observation_failure("precipitation", weather_data, e) from e

    # ------------------------------------------------------------------
    # THE TWO PRECIPITATION PATHS, STATED ONCE EACH (Plan 33.1-07 Task 4).
    #
    # WHY THERE ARE TWO AT ALL. A live Open-Meteo FORECAST carries a
    # probability; the ERA5 REANALYSIS archive does not and cannot, because a
    # reanalysis records what happened rather than what was expected to. Both
    # feed this module, so both have to be answerable, and they are answerable
    # to DIFFERENT precision.
    #
    # THE RULE EACH HELPER FOLLOWS, and it is one rule: when the probability is
    # absent its term is NOT APPLIED. It is not replaced by zero, by a
    # climatological average, or by anything inferred from the rainfall -- the
    # mm-only branch is a NARROWER function with its own documented range, not
    # the same function fed a fabricated input. The ranges differ and are stated
    # per helper so nobody later reads the two branches as interchangeable.
    #
    # THE DUAL-READING BRANCHES ARE BYTE-UNCHANGED from the pre-33.1-07 code.
    # `tests/unit/test_precipitation_from_measurement.py` pins their exact
    # outputs, because narrowing a gate is the kind of change that quietly
    # rewrites the branch it was not aiming at.
    #
    # `precip_prob=None` is the ONLY spelling of "the probability is absent" in
    # these four helpers -- a float parameter carrying NaN would make every
    # comparison below silently False, which is the shape that produced
    # `weather_game: 0.0` for games nobody measured.
    # ------------------------------------------------------------------

    def _precipitation_bands(
        self, precip_mm: float, precip_prob: float | None
    ) -> dict[str, float]:
        """The four intensity one-hots.

        The mm cut points are NOT new: ``0.5``, ``2.0`` and
        ``self.heavy_precip_threshold`` are the same thresholds the dual-reading
        branch already used. Dropping the probability disjunct also makes the
        mm-only bands an ordered PARTITION, where the ``or`` form can report a
        game as both light and moderate.

        Args:
            precip_mm: The measured rainfall in millimetres.
            precip_prob: The forecast probability, or ``None`` when absent.

        Returns:
            The four band columns.
        """
        if precip_prob is None:
            return {
                "precip_none": 1.0 if precip_mm <= 0.5 else 0.0,
                "precip_light": 1.0 if 0.5 < precip_mm <= 2.0 else 0.0,
                "precip_moderate": 1.0
                if 2.0 < precip_mm <= self.heavy_precip_threshold
                else 0.0,
                "precip_heavy": 1.0 if precip_mm > self.heavy_precip_threshold else 0.0,
            }
        # THE FORECAST BRANCH IS A PARTITION (Plan 33-14 Task 1, D33-34(b),
        # .planning/WINDOWS.md row 39). See PRECIPITATION_PARTITION_RULE.
        #
        # WHAT IT USED TO DO. Each band was a DISJUNCTION over the two readings
        # -- `precip_light` on `0.2 < prob <= 0.5 OR 0.5 < mm <= 2.0`,
        # `precip_moderate` on `0.5 < prob <= 0.8 OR 2.0 < mm <= 5.0`. A
        # forecast of probability 0.4 with 3.0 mm of rain set BOTH to 1.0: one
        # game reported as two intensities at once, in a family whose entire
        # contract is that exactly one member is hot. Only a live Open-Meteo
        # FORECAST carries a probability, so the overlapping branch is the one
        # serving 2026 while the repaired archive branch above is the historical
        # one -- the defect sat on the path that matters most.
        #
        # WHY THE MAXIMUM AND NOT SOMETHING ELSE. `_precipitation_impact` below
        # already states the rule both branches were written to follow: "the
        # probability only ever widened the LIGHT and MODERATE bands, never
        # added a level". Taking the LARGER of the two band indices is that
        # sentence made exact. It reproduces the old output on every input where
        # the old branch already fired exactly one band -- which is every input
        # where the two indices agree, or where either is zero -- and differs
        # ONLY where the old branch fired two, which is the defect itself. The
        # existing regression control at
        # `tests/unit/test_precipitation_from_measurement.py::TestTheGateNarrowsRatherThanDisappears::test_a_payload_carrying_both_readings_is_byte_preserved`
        # (probability 0.6, 3.2 mm) therefore passes unchanged: both readings
        # sit in band 2, so it resolves to the same single band it always did.
        #
        # THE CUT POINTS ARE NOT NEW and are not chosen here. The two index
        # helpers restate the SAME thresholds the branches already used -- and
        # in particular the same ones the mm-only literal above carries, which
        # this change deliberately leaves byte-unchanged because it is the
        # branch Phase 33.1 repaired and pinned.
        # `test_the_mm_only_branch_is_unchanged_across_the_same_grid` is what
        # keeps the two spellings of the rainfall cut points honest.
        band = max(
            self._rainfall_band_index(precip_mm),
            self._probability_band_index(precip_prob),
        )
        return {
            column: 1.0 if position == band else 0.0
            for position, column in enumerate(PRECIPITATION_BAND_COLUMNS)
        }

    def _rainfall_band_index(self, precip_mm: float) -> int:
        """The ordinal intensity band of a MEASURED rainfall, 0 through 3.

        The caller reaches this only after
        ``calculate_precipitation_features``' measurement gate has established
        that the rainfall is present, so ``precip_mm`` is always a real float
        here and the index is always defined.

        Args:
            precip_mm: The measured rainfall in millimetres.

        Returns:
            0 for none, 1 for light, 2 for moderate, 3 for heavy -- positions
            into :data:`PRECIPITATION_BAND_COLUMNS`.
        """
        if precip_mm <= 0.5:
            return 0
        if precip_mm <= 2.0:
            return 1
        if precip_mm <= self.heavy_precip_threshold:
            return 2
        return 3

    def _probability_band_index(self, precip_prob: float) -> int:
        """The ordinal intensity band of a FORECAST probability, 0 through 3.

        Args:
            precip_prob: The forecast probability, already known to be present.
                ``None`` never reaches here -- the mm-only branch above owns
                the absent case, and ``precip_prob=None`` is this module's ONE
                spelling of "the probability is absent".

        Returns:
            0 through 3, positions into :data:`PRECIPITATION_BAND_COLUMNS`.
        """
        if precip_prob <= 0.2:
            return 0
        if precip_prob <= 0.5:
            return 1
        if precip_prob <= 0.8:
            return 2
        return 3

    def _precipitation_impact(
        self, precip_mm: float, precip_prob: float | None
    ) -> float:
        """The 0-1 impact score, before the snow multiplier.

        Both branches span the full 0.0 / 0.3 / 0.6 / 1.0 range, because the
        rainfall alone already separates all four levels; the probability only
        ever widened the LIGHT and MODERATE bands, never added a level.

        THE SCORE IS THE LEVEL OF THE BAND THAT FIRED, on both branches. It is
        the same quantity ``_precipitation_bands`` one-hots, read off
        :data:`PRECIPITATION_IMPACT_BY_BAND` instead of re-derived, so the two
        cannot disagree.
        """
        if precip_prob is None:
            if precip_mm <= 0.5:
                return 0.0
            if precip_mm <= 2.0:
                return 0.3
            if precip_mm <= self.heavy_precip_threshold:
                return 0.6
            return 1.0
        # THE FORECAST BRANCH FOLLOWS THE BAND RULE (Plan 33-14 closing fix,
        # .planning/WINDOWS.md row 43, owner-ruled 2026-09-14). Same maximum of
        # the two band indices `_precipitation_bands` takes, indexed into the
        # level mapping rather than re-tested against the raw readings.
        #
        # WHAT IT USED TO DO. `prob <= 0.5 OR mm <= 2.0` returned 0.3, and
        # `prob <= 0.8 OR mm <= heavy` returned 0.6 -- each returning early as
        # soon as EITHER reading was low. That is a MINIMUM over the two
        # readings, and the band rule directly above is a MAXIMUM, so ten of the
        # sixteen grid points scored an intensity the same call's one-hot
        # contradicted. The worst: 9.0 mm of rain, one-hot `precip_heavy`,
        # scored 0.3 on a 0.1 probability -- a downpour given a drizzle's score
        # because the forecast was not confident. The dominant direction was
        # heavy RAINFALL understated by a low PROBABILITY, which is the
        # direction that matters: the rainfall is the reading that is actually
        # measured.
        #
        # THE MM-ONLY BRANCH ABOVE IS DELIBERATELY BYTE-UNCHANGED. It is the
        # branch Phase 33.1 repaired and pinned, it is already the maximum over
        # a single index, and `test_the_mm_only_impact_branch_is_unchanged` is
        # what stops this fix reaching across into it -- the failure mode a
        # targeted repair has.
        return PRECIPITATION_IMPACT_BY_BAND[
            max(
                self._rainfall_band_index(precip_mm),
                self._probability_band_index(precip_prob),
            )
        ]

    def _turnover_multiplier(
        self, precip_mm: float, precip_prob: float | None
    ) -> float:
        """Wet-ball turnover multiplier.

        RANGE, stated because the two branches differ and the difference is a
        consequence rather than an oversight: with a probability the multiplier
        reaches 1.5; without one the ``precip_prob * 0.2`` term is not applied,
        so the mm-only branch spans 1.0 and then 1.2 to 1.4. The missing term is
        not assumed to be zero -- it is a quantity this observation does not
        carry, and the multiplier says so by being narrower rather than by
        inventing the input that would widen it.
        """
        if precip_prob is None:
            if precip_mm <= 1.0:
                return 1.0
            return min(1.0 + 0.2 + (min(precip_mm, 10.0) / 10.0 * 0.2), 1.5)
        if precip_prob <= 0.3 and precip_mm <= 1.0:
            return 1.0
        base_increase = 0.2 + (precip_prob * 0.2) + (min(precip_mm, 10.0) / 10.0 * 0.2)
        return min(1.0 + base_increase, 1.5)

    def _passing_efficiency(self, precip_mm: float, precip_prob: float | None) -> float:
        """Completion-percentage multiplier in the wet.

        RANGE: with a probability, 0.85 to 1.0; without one the
        ``precip_prob * 0.15`` term is not applied and the branch spans 0.9 to
        1.0. The dry gate also moves from ``precip_prob <= 0.3`` to
        ``precip_mm <= 0.5`` -- the module's OWN millimetre threshold for "no
        meaningful precipitation", already used by ``precip_none`` and by the
        rain/snow test, rather than a new number chosen here.
        """
        if precip_prob is None:
            if precip_mm <= 0.5:
                return 1.0
            return max(0.85, 1.0 - (min(precip_mm, 5.0) / 5.0 * 0.1))
        if precip_prob <= 0.3:
            return 1.0
        return max(0.85, 1.0 - (precip_prob * 0.15) - (min(precip_mm, 5.0) / 5.0 * 0.1))

    def _indoor_precipitation_features(self) -> dict[str, float]:
        """The INDOOR precipitation state. Indoors it genuinely IS dry.

        These values are a TRUE statement about a covered game, in Ruling J's
        own terms, which is why D33.1-07 keeps indoor precipitation at a genuine
        dry level rather than calling it NULL.

        THIS IS NOT A FALLBACK FOR A FAILED CALCULATION. It was called
        ``_default_precipitation_features`` and was reached from an ``except``
        branch, where it invented a dry day out of a malformed payload. The
        rename is the guard, and a test asserts the single indoor call site.
        """
        return {
            "precip_prob": 0.0,
            "precip_mm": 0.0,
            "precip_none": 1.0,
            "precip_light": 0.0,
            "precip_moderate": 0.0,
            "precip_heavy": 0.0,
            "is_snow": 0.0,
            "is_rain": 0.0,
            "is_dry": 1.0,
            "precip_impact_score": 0.0,
            "turnover_multiplier": 1.0,
            "passing_efficiency": 1.0,
        }

    def calculate_weather_severity(
        self,
        wind_features: dict[str, float],
        temp_features: dict[str, float],
        precip_features: dict[str, float],
    ) -> dict[str, float]:
        """
        Calculate overall weather severity scores.

        Args:
            wind_features: Wind feature dictionary
            temp_features: Temperature feature dictionary
            precip_features: Precipitation feature dictionary

        Returns:
            Dictionary with overall weather severity features
        """
        try:
            # Individual impact scores.
            #
            # The `.get` fallback is NaN rather than 0.0 on purpose: a missing
            # key is an unknown impact, and defaulting it to "no impact" is the
            # same fabrication one level up from the families below.
            wind_impact = wind_features.get("wind_impact_score", NAN)
            cold_impact = temp_features.get("cold_impact_score", NAN)
            heat_impact = temp_features.get("heat_impact_score", NAN)
            precip_impact = precip_features.get("precip_impact_score", NAN)

            if any(
                _is_missing(value)
                for value in (wind_impact, cold_impact, heat_impact, precip_impact)
            ):
                # D33.1-07's inheritance rule, applied to the composites. Note
                # what NaN arithmetic alone would have produced here:
                # `1.0 if nan >= 0.6 else 0.0` is 0.0, so `weather_game` would
                # have read as a confident "this was not a weather game" for a
                # game nobody measured. The comparisons must be skipped, not
                # merely fed a NaN.
                return dict.fromkeys(SEVERITY_FEATURE_COLUMNS, NAN)

            # Combined weather severity (research-based weights)
            # Wind is the larger driver, temperature has independent but smaller signal
            weather_severity = (
                wind_impact * 0.5  # Wind: 50% weight (larger driver)
                + (cold_impact + heat_impact)
                * 0.2  # Temperature: 20% weight (independent signal)
                + precip_impact * 0.3  # Precipitation: 30% weight
            )

            # Weather advantage factors
            # Home teams typically have advantage in severe weather (familiarity)
            home_weather_advantage = (
                weather_severity * 0.1
            )  # 10% of severity becomes home advantage

            # Game style impact (weather favors defense and running game)
            defensive_advantage = weather_severity * 0.15
            rushing_advantage = weather_severity * 0.2

            # Scoring impact (severe weather reduces total points)
            scoring_reduction = min(weather_severity * 0.8, 0.3)  # Max 30% reduction

            return {
                "weather_severity_score": weather_severity,
                "home_weather_advantage": home_weather_advantage,
                "defensive_advantage": defensive_advantage,
                "rushing_advantage": rushing_advantage,
                "scoring_reduction": scoring_reduction,
                "weather_game": 1.0 if weather_severity >= 0.6 else 0.0,
                "extreme_weather": 1.0 if weather_severity >= 0.8 else 0.0,
            }

        except (ValueError, KeyError, TypeError) as e:
            # The FOURTH fabricating handler, and not one the plan named. It
            # returned the whole composite family at 0.0 -- "weather reduced
            # scoring by nothing" asserted from a crash. D33.1-07's third state
            # is about the CALCULATION, and this is one.
            logger.error(
                "Weather severity calculation failed",
                error=str(e),
            )
            raise _observation_failure("weather severity", {}, e) from e

    def build_weather_features(
        self,
        games_df: pd.DataFrame,
        target_season: int | None = None,
        target_week: int | None = None,
        *,
        weather_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Build weather features for all games.

        .. deprecated::
            Use :meth:`build_features` instead (conforms to FeatureBuilder Protocol).

        Args:
            games_df: DataFrame with game information
            target_season: Specific season to calculate features for
            target_week: Specific week to calculate features for
            weather_df: The silver weather frame to build over. When given it is
                used AS-IS and no store is read; when ``None`` the silver read
                happens as before. This is the injection seam a sandboxed run or a
                test uses -- `monkeypatch` is not a sandbox, and a test that
                patched a table NAME while the write still resolved the production
                root is how Phase 33 Wave 6 destroyed production gold. Threading
                the frame is the same discipline
                ``backtest.ou_divergence.run_ou_divergence_diagnosis`` already uses
                for ``preds`` and ``odds``.

        Returns:
            DataFrame with weather features added (full 38+ columns, uncompressed)
        """
        warnings.warn(
            "build_weather_features is deprecated; use build_features instead",
            DeprecationWarning,
            stacklevel=2,
        )
        logger.info(
            "Building weather features",
            games=len(games_df),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Load weather data, unless the caller supplied the frame.
            #
            # ONE LINE, layer and source POSITIONAL -- `load_dataframe(table_name,
            # layer, source)`. Plan 33.1-02's guard scans for `load_dataframe(`,
            # `SILVER_WEATHER_TABLE` and `parquet` CO-LOCATED on a line, and a
            # wrapped call reads to it as absent. "parquet" is stated rather than
            # left at "auto" so a DuckDB table created later cannot silently win
            # over the parquet the promotion actually writes.
            if weather_df is None:
                weather_df = load_dataframe(SILVER_WEATHER_TABLE, "silver", "parquet")
            logger.info("Loaded weather data", weather_records=len(weather_df))

            # Filter to target if specified
            if target_season and target_week:
                # Filter games
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

                # Filter weather data to matching games
                game_ids = games_df["game_id"].tolist()
                weather_df = weather_df[weather_df["game_id"].isin(game_ids)]

            weather_features = []

            for _, game in games_df.iterrows():
                game_id = game["game_id"]
                season = game["season"]
                week = game["week"]

                # Find weather data for this game
                game_weather = weather_df[weather_df["game_id"] == game_id]

                if len(game_weather) == 0:
                    raise _no_weather_row(game_id)

                # Use most recent weather forecast for this game
                latest_weather = game_weather.sort_values("forecast_time").iloc[-1]
                weather_data = latest_weather.to_dict()

                # Basic game identifiers
                game_features: dict[str, Any] = {
                    "game_id": game_id,
                    "season": season,
                    "week": week,
                }

                covered = _row_is_covered(weather_data)
                is_outdoor = bool(weather_data.get("is_outdoor", False))

                if _game_is_held_at_gold_default(game, game_id):
                    # THE 2026 GOLD-DEFAULT SWITCH (D33-25), applied to the full
                    # builder. The silver row was still REQUIRED above -- a game
                    # with no record at all is an anomaly in every season and is
                    # still refused by name. What the switch overrides is the
                    # row's CONTENT: the observation exists, it is real, and
                    # gold deliberately does not consume it yet.
                    game_features.update(
                        self._absent_observation_features(is_outdoor=is_outdoor)
                    )
                    weather_features.append(game_features)
                    continue

                if not covered:
                    # RULING J, the middle row: the venue resolved, the ERA5
                    # observation did not arrive. Nothing is calculated, and the
                    # only two numbers on the row are the two flags.
                    game_features.update(
                        self._absent_observation_features(is_outdoor=is_outdoor)
                    )
                    weather_features.append(game_features)
                    continue

                game_features["weather_affects_game"] = 1.0 if is_outdoor else 0.0
                game_features["weather_coverage"] = 1.0

                if not is_outdoor:
                    # Indoor game. Wind and precipitation are genuinely calm and
                    # dry; the TEMPERATURE family is NULL, because there is no
                    # outdoor temperature to report and a band with nothing to
                    # band has no answer (Ruling J).
                    game_features.update(self._indoor_wind_features())
                    game_features.update(self._indoor_temperature_features())
                    game_features.update(self._indoor_precipitation_features())

                    # The seven composites keep their genuine no-impact level:
                    # "weather reduced scoring by nothing" is a TRUE statement
                    # about an indoor game rather than a stand-in.
                    game_features.update(dict.fromkeys(SEVERITY_FEATURE_COLUMNS, 0.0))
                else:
                    # Outdoor game - calculate weather features
                    wind_features = self.calculate_wind_features(weather_data)
                    temp_features = self.calculate_temperature_features(weather_data)
                    precip_features = self.calculate_precipitation_features(
                        weather_data
                    )
                    severity_features = self.calculate_weather_severity(
                        wind_features, temp_features, precip_features
                    )

                    # Combine all weather features
                    game_features.update(wind_features)
                    game_features.update(temp_features)
                    game_features.update(precip_features)
                    game_features.update(severity_features)

                # Add raw weather values for reference
                game_features.update(
                    {
                        "raw_temp_f": _or_null(weather_data.get("temp_f")),
                        "raw_wind_mph": _or_null(weather_data.get("wind_mph")),
                        "raw_precip_prob": _or_null(weather_data.get("precip_prob")),
                        "raw_precip_mm": _or_null(weather_data.get("precip_mm")),
                        "raw_humidity_pct": _or_null(weather_data.get("humidity_pct")),
                        "weather_condition": weather_data.get("condition", "Unknown"),
                    }
                )

                weather_features.append(game_features)

            # Convert to DataFrame
            features_df = pd.DataFrame(weather_features)
            _assert_builder_columns(features_df, "full")

            logger.info(
                "Built weather features",
                features_count=len(features_df),
                outdoor_games=features_df["weather_affects_game"].sum(),
                weather_games=features_df.get("weather_game", pd.Series([0])).sum(),
            )

            return features_df

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to build weather features", error=str(e))
            raise

    def get_weather_features_for_game(
        self, game_id: str, season: int, week: int
    ) -> dict[str, float]:
        """
        Get weather features for a specific game.

        Args:
            game_id: Game identifier
            season: Season year
            week: Week number

        Returns:
            Dictionary with weather features
        """
        try:
            # Load weather features. source="parquet", NOT the default "auto"
            # (code review WR-02): "auto" tries DuckDB first and db.table_exists is
            # layer-blind, which is the mechanism named in this module's
            # SILVER_WEATHER_TABLE comment above. That comment hardened the two raw
            # `weather` reads and left this one, inside the very module carrying the
            # warning.
            features_df = load_dataframe("weather_features", "silver", "parquet")

            # Filter to specific game
            game_features = features_df[
                (features_df["game_id"] == game_id)
                & (features_df["season"] == season)
                & (features_df["week"] == week)
            ]

            if len(game_features) == 0:
                logger.warning(
                    "No weather features found for game",
                    game_id=game_id,
                    season=season,
                    week=week,
                )
                return {}

            # Convert to dictionary, excluding non-feature columns
            exclude_cols = ["game_id", "season", "week"]
            features_dict = {}

            game_row = game_features.iloc[0]
            for col in game_features.columns:
                if col not in exclude_cols:
                    features_dict[col] = game_row[col]

            return features_dict

        except (ValueError, KeyError, TypeError) as e:
            # Returned `{}` before Plan 33.1-04. An empty feature dict is not a
            # smaller answer, it is a DIFFERENT one -- and a caller that merges
            # it gets a game with no weather at all, reported as success. In
            # this module an `except` branch that returns a value is the
            # fabrication pattern (D33.1-07).
            logger.error(
                "Failed to get weather features for game",
                game_id=game_id,
                season=season,
                week=week,
                error=str(e),
            )
            raise _observation_failure(
                "weather feature lookup", {"game_id": game_id}, e
            ) from e

    def validate_weather_features(self, features_df: pd.DataFrame) -> bool:
        """
        Validate weather features for data quality.

        Args:
            features_df: Weather features DataFrame

        Returns:
            True if validation passes
        """
        if len(features_df) == 0:
            logger.error("No weather features found")
            return False

        # Check for required columns
        required_cols = ["game_id", "season", "week", "weather_affects_game"]
        missing_cols = set(required_cols) - set(features_df.columns)
        if missing_cols:
            logger.error("Missing required columns", missing_columns=list(missing_cols))
            return False

        # Check temperature ranges
        if "temp_f" in features_df.columns:
            temps = features_df["temp_f"].dropna()
            if len(temps) > 0 and (temps.min() < -20 or temps.max() > 130):
                logger.warning(
                    "Temperature values outside reasonable range",
                    min_temp=temps.min(),
                    max_temp=temps.max(),
                )

        # Check wind speeds
        if "wind_mph" in features_df.columns:
            winds = features_df["wind_mph"].dropna()
            if len(winds) > 0 and (winds.min() < 0 or winds.max() > 100):
                logger.warning(
                    "Wind speeds outside reasonable range",
                    min_wind=winds.min(),
                    max_wind=winds.max(),
                )

        # Check precipitation probabilities
        if "precip_prob" in features_df.columns:
            precip_probs = features_df["precip_prob"].dropna()
            if len(precip_probs) > 0:
                if precip_probs.min() < 0 or precip_probs.max() > 1:
                    logger.warning(
                        "Precipitation probabilities outside 0-1 range",
                        min_precip=precip_probs.min(),
                        max_precip=precip_probs.max(),
                    )

        # Check severity scores are 0-1
        severity_cols = [
            "weather_severity_score",
            "wind_impact_score",
            "cold_impact_score",
            "precip_impact_score",
        ]
        for col in severity_cols:
            if col in features_df.columns:
                values = features_df[col].dropna()
                if len(values) > 0:
                    if values.min() < 0 or values.max() > 1:
                        logger.warning(
                            f"Severity scores outside 0-1 range for {col}",
                            min_value=values.min(),
                            max_value=values.max(),
                        )

        # Check outdoor vs indoor games
        outdoor_games = features_df["weather_affects_game"].sum()
        total_games = len(features_df)

        logger.info(
            "Weather features validation completed",
            records=total_games,
            outdoor_games=int(outdoor_games),
            indoor_games=int(total_games - outdoor_games),
            weather_columns=len(
                [
                    col
                    for col in features_df.columns
                    if col not in ["game_id", "season", "week"]
                ]
            ),
        )

        return True

    # ------------------------------------------------------------------
    # FeatureBuilder Protocol methods (compressed output)
    # ------------------------------------------------------------------

    def build_features(
        self,
        games_df: pd.DataFrame,
        as_of_datetime: datetime,
        *,
        target_season: int | None = None,
        target_week: int | None = None,
        weather_df: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Build compressed weather features (4 features) for games.

        Conforms to the FeatureBuilder Protocol. Uses only weather data
        available before ``as_of_datetime``.

        Output columns:
            game_id, weather_severity_score, wind_mph, is_precipitation, is_outdoor

        Args:
            games_df: DataFrame of games to build features for.
            as_of_datetime: Time-fence cutoff. Only data before this
                timestamp may be used.
            target_season: Optional season filter.
            target_week: Optional week filter.
            weather_df: The silver weather frame to build over. When given it is
                used AS-IS and no store is read; when ``None`` the silver read
                happens as before. See :meth:`build_weather_features` for why this
                seam exists rather than a monkeypatch.

        Returns:
            DataFrame with exactly 5 columns (game_id + 4 features).
        """
        logger.info(
            "Building compressed weather features",
            games=len(games_df),
            as_of=as_of_datetime.isoformat(),
            target_season=target_season,
            target_week=target_week,
        )

        try:
            # Load weather data, unless the caller supplied the frame.
            #
            # ONE LINE, layer and source POSITIONAL -- `load_dataframe(table_name,
            # layer, source)`. Plan 33.1-02's guard scans for `load_dataframe(`,
            # `SILVER_WEATHER_TABLE` and `parquet` CO-LOCATED on a line, and a
            # wrapped call reads to it as absent. "parquet" is stated rather than
            # left at "auto" so a DuckDB table created later cannot silently win
            # over the parquet the promotion actually writes.
            if weather_df is None:
                weather_df = load_dataframe(SILVER_WEATHER_TABLE, "silver", "parquet")
            logger.info("Loaded weather data", weather_records=len(weather_df))

            # Time-fence: only use forecasts available before as_of_datetime
            if "forecast_time" in weather_df.columns:
                weather_df = weather_df[weather_df["forecast_time"] <= as_of_datetime]

            # Filter to target if specified
            if target_season and target_week:
                games_df = games_df[
                    (games_df["season"] == target_season)
                    & (games_df["week"] == target_week)
                ].copy()

                game_ids = games_df["game_id"].tolist()
                weather_df = weather_df[weather_df["game_id"].isin(game_ids)]

            compressed_rows: list[dict[str, object]] = []

            for _, game in games_df.iterrows():
                game_id = game["game_id"]

                # Find weather data for this game
                game_weather = weather_df[weather_df["game_id"] == game_id]

                if len(game_weather) == 0:
                    raise _no_weather_row(game_id)

                latest_weather = game_weather.sort_values("forecast_time").iloc[-1]
                weather_data: dict[str, Any] = latest_weather.to_dict()

                is_outdoor = bool(weather_data.get("is_outdoor", False))

                if _game_is_held_at_gold_default(game, game_id):
                    # THE 2026 GOLD-DEFAULT SWITCH (D33-25), in the compressed
                    # shape. Both builders feed gold, so a switch on one of them
                    # would be half a switch: the full builder writes silver
                    # `weather_features`, and this one is the FeatureBuilder
                    # Protocol path. `is_outdoor` still records that weather
                    # APPLIES -- what is held back is the observation.
                    compressed_rows.append(
                        {
                            "game_id": game_id,
                            "weather_severity_score": NAN,
                            "wind_mph": NAN,
                            "is_precipitation": NAN,
                            "is_outdoor": 1.0 if is_outdoor else 0.0,
                        }
                    )
                    continue

                if not _row_is_covered(weather_data):
                    # Ruling J's middle row, in the compressed shape: the three
                    # feature columns are NULL, and `is_outdoor` still records
                    # that weather APPLIES. What is absent is the observation.
                    compressed_rows.append(
                        {
                            "game_id": game_id,
                            "weather_severity_score": NAN,
                            "wind_mph": NAN,
                            "is_precipitation": NAN,
                            "is_outdoor": 1.0 if is_outdoor else 0.0,
                        }
                    )
                elif not is_outdoor:
                    # Indoor/dome: all features zeroed
                    compressed_rows.append(
                        {
                            "game_id": game_id,
                            "weather_severity_score": 0.0,
                            "wind_mph": 0.0,
                            "is_precipitation": 0.0,
                            "is_outdoor": 0.0,
                        }
                    )
                else:
                    # Outdoor game: compute full features, then compress
                    wind_features = self.calculate_wind_features(weather_data)
                    temp_features = self.calculate_temperature_features(weather_data)
                    precip_features = self.calculate_precipitation_features(
                        weather_data
                    )
                    severity_features = self.calculate_weather_severity(
                        wind_features, temp_features, precip_features
                    )

                    # Determine is_precipitation: binary flag.
                    #
                    # THE LAST `or 0.0` IN THIS MODULE, removed by Plan 33.1-07
                    # Task 4. It read an ABSENT probability and an ABSENT
                    # rainfall as 0.0 and reported the game as dry -- the exact
                    # shape `_is_missing` exists to prevent, surviving here
                    # because this is the compressed FeatureBuilder-Protocol
                    # path and the full builder is what feeds gold. It is the
                    # SERVING path, which answers a live prediction, so the
                    # defect was reachable even though no gold column carries
                    # `is_precipitation` today.
                    #
                    # The readings are now consulted only where they EXIST, and
                    # a row with neither reading reports NULL rather than dry.
                    raw_prob = weather_data.get("precip_prob")
                    raw_mm = weather_data.get("precip_mm")
                    is_snow = precip_features.get("is_snow", 0.0)
                    is_rain = precip_features.get("is_rain", 0.0)

                    fell = (not _is_missing(raw_prob) and float(raw_prob) > 0.3) or (
                        not _is_missing(raw_mm) and float(raw_mm) > 0.5
                    )

                    if _is_missing(raw_prob) and _is_missing(raw_mm):
                        is_precip = NAN
                    else:
                        is_precip = (
                            1.0 if (fell or is_snow == 1.0 or is_rain == 1.0) else 0.0
                        )

                    compressed_rows.append(
                        {
                            "game_id": game_id,
                            "weather_severity_score": severity_features[
                                "weather_severity_score"
                            ],
                            "wind_mph": wind_features["wind_mph"],
                            "is_precipitation": is_precip,
                            "is_outdoor": 1.0,
                        }
                    )

            features_df = pd.DataFrame(compressed_rows)
            _assert_builder_columns(features_df, "compressed")

            logger.info(
                "Built compressed weather features",
                features_count=len(features_df),
                outdoor_games=int(features_df["is_outdoor"].sum()),
            )

            return features_df

        except (ValueError, KeyError, TypeError) as e:
            logger.error("Failed to build compressed weather features", error=str(e))
            raise

    def get_features_for_game(
        self,
        game_id: str,
        as_of_datetime: datetime,
    ) -> dict[str, float]:
        """Get compressed weather features for a single game.

        Conforms to the FeatureBuilder Protocol.

        Args:
            game_id: Unique game identifier.
            as_of_datetime: Time-fence cutoff.

        Returns:
            Dictionary mapping feature names to values.
        """
        try:
            # Build a minimal games DataFrame for the single game
            games_df = pd.DataFrame([{"game_id": game_id, "season": 0, "week": 0}])
            result = self.build_features(games_df, as_of_datetime)

            if len(result) == 0:
                logger.warning(
                    "No weather features for game",
                    game_id=game_id,
                )
                return {}

            row = result.iloc[0]
            return {col: float(row[col]) for col in result.columns if col != "game_id"}

        except (ValueError, KeyError, TypeError) as e:
            # See `get_weather_features_for_game`: the same swallow, the same
            # replacement. An `except` branch in this module raises.
            logger.error(
                "Failed to get weather features for game",
                game_id=game_id,
                error=str(e),
            )
            raise _observation_failure(
                "compressed weather feature lookup", {"game_id": game_id}, e
            ) from e
