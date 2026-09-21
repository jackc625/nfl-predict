"""The PRE-REGISTERED rules for decoding and checking the archived day-before forecasts (Plan 33.2-11).

READ, NEVER EDITED. This module is a pre-registration, in the shape of
``backtest/group_gate_constants.py``. Its LAST-MODIFYING COMMIT must be a STRICT git ancestor of
the commit that publishes ``MOS-DECODE-COMPARISON.md``; ``tests/unit/test_mos_tolerance_ancestry.py``
asserts exactly that, and ``scripts/compare_mos_to_era5.py`` refuses to compute anything unless it
already holds. That ancestry is the only mechanical evidence that these bounds were fixed before any
comparison number existed. Editing this file after the comparison does not fix a bound -- it
destroys the evidence. A bound that turns out wrong is reported as wrong; it is not amended here.

It carries three owner rulings, all given on 2026-09-21 at Plan 33.2-11's Task 1 checkpoint, before
any forecast had been compared with any observation:

1. ``bounds-as-proposed`` -- owner answer, verbatim: "Proposed limits". The six DETECTION BOUNDS
   below.
2. ``qpf-midpoint`` -- owner answer, verbatim: "Middle of each range". How a bulletin's ordinal
   precipitation category becomes one of the four existing precipitation levels.
3. ``commit-argument`` -- owner answer, verbatim: "Label figure unconfirmed". The argument that the
   12 UTC run is public before the lock is committed; the delay figure is marked ``[ASSUMED]``.

WHAT THE BOUNDS ARE. They are DETECTION BOUNDS ON OUR DECODING -- not a claim that the forecast was
accurate. A day-before forecast legitimately misses the weather that happened. A breach means our
decoding is wrong: a unit error, a wrong station, a wrong valid hour or a mis-joined game. The
bounds carry headroom over the 72-game read-only sample measured before this file existed
(temperature forecast-minus-observed mean +0.34 F, mean absolute difference 2.78 F, p95 8.2 F,
max 13.9 F; wind mean +2.33 mph, mean absolute difference 3.22 mph, p95 7.1 mph, max 10.0 mph), so
ordinary forecast error cannot trip them but a decode error will. SPEC R6: a breach STOPS the
backfill. The response to a breach is to fix the decode or the station map, never to widen a bound.

THE TRANSMISSION ARGUMENT (ruling 3). The IEM archive stores each bulletin's model CYCLE time, not
the moment the bulletin went out; there is no availability field of any kind in the payload. So
"published at or before the lock" cannot be checked row by row, and is discharged instead by this
standing argument:

- The lock is 18:00 America/New_York on the ET calendar day before kickoff: 22:00 UTC under daylight
  time (EDT) and 23:00 UTC under standard time (EST).
- The run used is the 12 UTC cycle of that same ET day (08:00 EDT / 07:00 EST). Its bulletin is
  public long before the lock: the argument needs only that the transmission delay is under about
  ten hours, and nobody disputes that.
- The next cycle, 18 UTC, is NOT used. Its bulletin cannot be relied on to be public before an EDT
  lock, and a "latest admissible run" rule would switch from the 12 UTC to the 18 UTC bulletin at the
  daylight-saving change -- giving the coldest games a fresher forecast than September games purely
  from a clock change (RESEARCH pitfall P9, D33.2-13). Exactly one run is therefore admissible per
  game by construction.
- [ASSUMED] The often-quoted figure that a MOS bulletin is on the wire "about 4h15m after cycle
  time" (about 16:15 UTC for the 12 UTC run) is UNVERIFIED. No official NWS document stating a
  transmission clock time was found (RESEARCH assumption A1, Open Question 1). It is recorded here
  as unconfirmed and is NOT relied on by anything; the argument above rests only on the
  under-ten-hours bound.

THE PRECIPITATION ANCHORING (ruling 2). The bulletins carry no millimetre amount, only an ordinal
category (NWS TPB 482). Each category is read at the MIDDLE of its published range, and that
midpoint is placed against the EXISTING precipitation-level boundaries in ``features/weather.py``
(0.5 / 2.0 / 5.0 mm) -- no new boundary is added. The category is never turned into a millimetre
value in any stored column; only its level is used.

HEADS-UP RECORDED WITH THE RULINGS (no decision was asked). The live forecast service and the
bulletins disagree on wind by a measured, systematic amount: the bulletins read about 1.6 mph
higher (18 stadiums, mean +1.56 mph, max gap 6.5 mph; temperature mean +0.37 F, effectively exact).
The calm/moderate wind line is 5.0 mph, so on a calm day a game can switch wind band between
training (bulletins) and live serving (the forecast service), and the spread model uses that wind
band today. This is a KNOWN, MEASURED, ACCEPTED train/serve difference, handed to the re-fit plans
(33.2-23) in the phase's deferred-items register.

ASCII only, no emoji (CLAUDE.md hard constraint). Pure literals: this module imports nothing, so no
change anywhere else in the repository can move a value here.
"""

from __future__ import annotations

__all__ = [
    "BOUND_NAMES",
    "GROSS_TEMP_MISS_F",
    "OWNER_RULINGS",
    "OWNER_RULINGS_DATE",
    "PRECIP_LEVEL_UPPER_EDGES_MM",
    "QPF_ANCHOR",
    "QPF_CATEGORY_INCHES",
    "QPF_CATEGORY_TO_PRECIP_LEVEL",
    "QPF_MISSING_CATEGORY",
    "RESOLVED_BULLETIN_SHARE_MIN",
    "TEMP_GROSS_MISS_SHARE_MAX",
    "TEMP_MEAN_ABS_DIFF_MAX_F",
    "TEMP_MEAN_SIGNED_DIFF_MAX_ABS_F",
    "TRANSMISSION_DELAY_FIGURE_STATUS",
    "UNCOVERABLE_GAMES_COUNT_AGAINST_COVERAGE",
    "VERDICT_REQUIRES_PASS_WITHOUT_KICKOFF_HOUR_SUBSET",
    "WIND_MEAN_ABS_DIFF_MAX_MPH",
    "WIND_MEAN_SIGNED_DIFF_MAX_MPH",
    "WIND_MEAN_SIGNED_DIFF_MIN_MPH",
]

# ---------------------------------------------------------------------------
# The rulings, recorded verbatim.
# ---------------------------------------------------------------------------

OWNER_RULINGS_DATE: str = "2026-09-21"

OWNER_RULINGS: dict[str, str] = {
    "bounds-as-proposed": "Proposed limits",
    "qpf-midpoint": "Middle of each range",
    "commit-argument": "Label figure unconfirmed",
}

# ---------------------------------------------------------------------------
# Ruling 1: the six detection bounds. Every difference is FORECAST minus OBSERVATION: the decoded
# bulletin value at the forecast hour nearest kickoff, minus the on-disk ERA5 observation for the
# same game. "Typical miss" is the MEAN ABSOLUTE DIFFERENCE, the plan's own definition.
# ---------------------------------------------------------------------------

# Mean signed temperature difference must lie within +/- this many degrees F. Catches a
# systematic offset: a unit error, a wrong station, a wrong valid hour.
TEMP_MEAN_SIGNED_DIFF_MAX_ABS_F: float = 3.0

# Mean absolute temperature difference must be at most this many degrees F. Catches a wrong
# station or a wrong valid time.
TEMP_MEAN_ABS_DIFF_MAX_F: float = 5.0

# No more than TEMP_GROSS_MISS_SHARE_MAX of compared games may miss by MORE than GROSS_TEMP_MISS_F
# (strictly greater). At most 1 game in 100 off by more than 25 F. Catches a mis-joined game.
GROSS_TEMP_MISS_F: float = 25.0
TEMP_GROSS_MISS_SHARE_MAX: float = 0.01

# Mean signed wind difference must lie between these two values, inclusive, in mph. The bulletin
# predicts an airport's sustained 10 m wind and ERA5 reports a gridded 10 m wind, so a positive
# instrument offset is expected. A missed knots-to-mph conversion reads about 13% low and falls
# out of this band.
WIND_MEAN_SIGNED_DIFF_MIN_MPH: float = 1.0
WIND_MEAN_SIGNED_DIFF_MAX_MPH: float = 4.0

# Mean absolute wind difference must be at most this many mph.
WIND_MEAN_ABS_DIFF_MAX_MPH: float = 6.0

# At least this share of COVERED games must have a resolved bulletin.
#
# COVERED means: a game played 2002-2025 at a US outdoor or retractable-roof venue, located through
# the venue in force at the lock (``features.schedule_moves.facts_at_lock``). A game played outside
# the USA has no US forecast station at all -- the bulletins cover only the United States, Puerto
# Rico and the US Virgin Islands -- so it is UNCOVERABLE, not a missed bulletin. Uncoverable games
# are OUTSIDE this share's denominator and are published as their own count with the honest "no
# forecast" outcome; no stand-in station is ever used for them. Indoor games are likewise outside
# the denominator: no forecast is fetched for a closed roof.
RESOLVED_BULLETIN_SHARE_MIN: float = 0.99
UNCOVERABLE_GAMES_COUNT_AGAINST_COVERAGE: bool = False

# The 68 Monday and Thursday night games of 2002-2005 stored with a 09:00 ET kickoff (an AM/PM
# error in the feed; Plan 33.2-12 corrects it) were observed by ERA5 at that same wrong hour, so
# the forecast-versus-observation pairing is internally consistent and they ARE compared. They must
# not be able to hide a breach, so the verdict is PASS only if every bound passes BOTH over all
# compared games AND over all compared games excluding those 68. The 68 are also reported alone.
VERDICT_REQUIRES_PASS_WITHOUT_KICKOFF_HOUR_SUBSET: bool = True

BOUND_NAMES: tuple[str, ...] = (
    "temp_mean_signed_diff",
    "temp_mean_abs_diff",
    "temp_gross_miss_share",
    "wind_mean_signed_diff",
    "wind_mean_abs_diff",
    "resolved_bulletin_share",
)

# ---------------------------------------------------------------------------
# Ruling 2: the precipitation-category anchoring.
# ---------------------------------------------------------------------------

QPF_ANCHOR: str = "midpoint"

# NWS TPB 482 category edges, inches. Category 5 is open-ended ("> 1.00 in").
QPF_CATEGORY_INCHES: dict[int, tuple[float, float | None]] = {
    0: (0.0, 0.0),
    1: (0.01, 0.09),
    2: (0.10, 0.24),
    3: (0.25, 0.49),
    4: (0.50, 0.99),
    5: (1.00, None),
}

# 9 is the bulletin's own "missing" code: an unknown level carrying a coverage flag, never level 0.
QPF_MISSING_CATEGORY: int = 9

# The EXISTING level boundaries (upper edge of none / light / moderate), restated from
# features/weather.py's rainfall band index, not chosen here. Above the last edge is heavy.
PRECIP_LEVEL_UPPER_EDGES_MM: tuple[float, float, float] = (0.5, 2.0, 5.0)

# Category -> level index (0 none, 1 light, 2 moderate, 3 heavy), positions into
# features.weather.PRECIPITATION_BAND_COLUMNS. Midpoints: 1 -> 1.27 mm (light), 2 -> 4.32 mm
# (moderate), 3 -> 9.40 mm, 4 -> 18.9 mm and 5 (above 25.4 mm) all heavy.
QPF_CATEGORY_TO_PRECIP_LEVEL: dict[int, int] = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 3,
    5: 3,
}

# ---------------------------------------------------------------------------
# Ruling 3: the transmission figure's status (the argument itself is in the docstring above).
# ---------------------------------------------------------------------------

TRANSMISSION_DELAY_FIGURE_STATUS: str = (
    "[ASSUMED] UNVERIFIED: the ~4h15m cycle-to-transmission figure is not confirmed by any official "
    "document and is relied on by nothing; admissibility rests only on the delay being under ~10 h"
)
