"""The pre-registered 2026 rolling re-fit and its frozen-arm comparison (Phase 37; RFIT-01, D-23).

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/verdict_scope_2026.py`` and
``backtest/ev_chain_constants.py``. This module IS the pre-registration of the 2026 rolling re-fit,
not a description of one: the calendar and its A/A slide, the training recipe, the re-measure of the
bet rule, the integrity checks, the gate relationship and the deciding comparison. It is committed
ALONE, together with ``ROLLING-REFIT-PREREGISTRATION.md`` at the repository root, and pushed to
GitHub's ``master`` before week 8's first decision run, Wed 2026-10-28 17:00 America/New_York
(``PREREGISTRATION_DEADLINE_UTC``, D-03).

IT IS ONE PRE-REGISTRATION IN TWO FILES. The root markdown document is the human-readable half and
states in prose every rule that has no constant here; this module is the half the code reads. Both
were landed in ONE commit that touches nothing else.

Stated plainly because it is easy to forget later in the season: EDITING THIS FILE AFTER THE
COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. A value that is wrong here is wrong for the
whole 2026 re-fit, and the only honest repair is a NEW, visibly later commit whose message says it
INVALIDATES this one by naming its commit sha. Never an edit in place, never a quiet supersession.

THIS MODULE DOES NOT RECORD ITS OWN CONTENT HASH OR COMMIT. A file that must contain its own
whole-file hash has no fixed point. The witness lives OUTSIDE it: ``tests/phase37_state.py`` and
``refit/witness.py`` record the commit, the newline-normalized sha256 of both files and the commit
time in LATER commits, and the Phase-37 ancestry tests recompute and compare.

Constants only: NO project imports, NO I/O and NO logic. Every value that restates a fact the code
also holds (a cap, a trainer setting, an EV floor, alpha, the excluded groups) is COPIED here as a
literal, never imported, so this record cannot change meaning when the code around it moves;
``tests/unit/test_refit_preregistration_content.py`` proves the copies agreed when it was frozen.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

__all__ = [
    "AA_EVALUATION_TUESDAY_BY_WEEK",
    "AA_EXCLUDED_FIELDS",
    "AA_FIRST_WEEK",
    "AA_LAST_WEEK",
    "ALPHA",
    "BIAS_SEED_SEASONS",
    "CADENCE_SPACING_NFL_WEEKS",
    "CADENCE_TUESDAY_BY_WEEK",
    "COMPARISON_TEST",
    "CUTOFF_LOCAL_TIME",
    "CUTOFF_TIMEZONE",
    "EDGE_QUANTILE_METHOD",
    "EDGE_THRESHOLD_DECIMALS",
    "EV_FLOORS",
    "EXCLUDED_FEATURE_GROUPS",
    "FILL_CONVENTION_ID",
    "FIRST_KICKOFF_EVIDENCE_UTC",
    "GATE_RELATIONSHIP",
    "GROUP_VERDICT_NORMALIZED_SHA256",
    "GROUP_VERDICT_PATH",
    "HISTORICAL_FOLD_SEASONS",
    "HISTORICAL_TUNE_SEASONS",
    "INPUT_CAPS",
    "INTEGRITY_CHECKS",
    "LAST_NFL_WEEK",
    "LEAN_POOL_RULE",
    "LINE_RESIDUAL_CONVERTER",
    "LOG_LOSS_CLIP",
    "MARKET_CONVERTER_ID",
    "MIN_PAIRED_GAMES",
    "MIN_USABLE_REMEASURE_ROWS",
    "MODEL_SETTINGS",
    "MULTIPLE_TEST_CORRECTION",
    "NO_CLEAN_WEEK_RULE",
    "OU_TOTAL_FLOOR",
    "PLAYOFF_FIRST_GAME_DATES",
    "PREREGISTRATION_DEADLINE_UTC",
    "REFIT_DATE_LISTS",
    "REFIT_RECIPE_ID",
    "REFIT_SEASON",
    "ROLLUP_WORDS",
    "SHADOW_RECIPE_ID",
    "SPENT_HOLD_SEASON",
    "THREAD_LIMIT",
    "TRAINING_START_SEASON",
    "UNTOUCHED_GATE_FILES",
    "WP_CALIBRATOR",
    "WP_EDGE_ANCHOR",
    "WP_HISTORICAL_TUNE_SEASONS",
    "WP_TIE_LABEL",
]


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------

# The season the rolling re-fit runs in.
REFIT_SEASON: int = 2026

# D-20: ONE recipe id for every re-fit generation. Each generation's own numbers are named through
# the generation records under ledger/refit/, never by a new recipe id.
REFIT_RECIPE_ID: str = "recipe-2026-refit-v1"

# The pricing and sizing convention every row is stamped with (backtest/fill_conventions.py).
FILL_CONVENTION_ID: str = "fill-v1"

# D-38 / RFIT-04: the frozen arm's recipe, the row-19 entry of backtest/recipe_registry.py.
SHADOW_RECIPE_ID: str = "recipe-2026-row19-v1"

# D-03: week 8's first decision run, Wed 2026-10-28 17:00 America/New_York (EDT), in UTC.
PREREGISTRATION_DEADLINE_UTC: str = "2026-10-28T21:00:00+00:00"


# ---------------------------------------------------------------------------
# The gate relationship (RFIT-06, R5)
# ---------------------------------------------------------------------------

# The rule, verbatim. A re-fit that passes INTEGRITY_CHECKS goes live with no performance test.
GATE_RELATIONSHIP: str = "gate-free by design; the shadow arm is the control and the season close is the verdict"

# The old deploy-gate path. Byte-unchanged by Phase 37; its two tripwires stay red.
UNTOUCHED_GATE_FILES: tuple[str, ...] = (
    "config/gate.toml",
    "models/deploy_gate.py",
    "scripts/promote_models.py",
)


# ---------------------------------------------------------------------------
# The calendar (D-01, D-02, D-05, D-06, D-33)
# ---------------------------------------------------------------------------

# D-05: each re-fit date's scheduled cutoff is 17:00 on that Tuesday in America/New_York. Every
# instant goes through the zone, never a fixed offset: DST ends Sun 2026-11-01, so every listed
# cutoff is 22:00 UTC.
CUTOFF_LOCAL_TIME: str = "17:00"
CUTOFF_TIMEZONE: str = "America/New_York"

# Week 22 is the Super Bowl. No re-fit date lies beyond it.
LAST_NFL_WEEK: int = 22

# D-02: consecutive re-fit dates are this many NFL weeks apart.
CADENCE_SPACING_NFL_WEEKS: int = 4

# The cadence Tuesday of each NFL week: the last Tuesday strictly before that week's first game
# (FIRST_KICKOFF_EVIDENCE_UTC for weeks 9-18, PLAYOFF_FIRST_GAME_DATES for weeks 19-22). Week 22's
# Tuesday follows the bye week after the conference championships.
CADENCE_TUESDAY_BY_WEEK: dict[int, str] = {
    9: "2026-11-03",
    10: "2026-11-10",
    11: "2026-11-17",
    12: "2026-11-24",
    13: "2026-12-01",
    14: "2026-12-08",
    15: "2026-12-15",
    16: "2026-12-22",
    17: "2026-12-29",
    18: "2027-01-05",
    19: "2027-01-12",
    20: "2027-01-19",
    21: "2027-01-26",
    22: "2027-02-09",
}

# The evidence the dates were derived from: each week's first kickoff in the recorded 2026 schedule
# (data/silver/games.parquet, read 2026-10-20), as a UTC instant. Week 18's value is the recorded
# placeholder (Sun 2027-01-10 13:00 ET). A game moved later never moves a pre-registered date; it
# only changes which games precede a cutoff.
FIRST_KICKOFF_EVIDENCE_UTC: dict[int, str] = {
    8: "2026-10-30T00:15:00+00:00",
    9: "2026-11-06T01:15:00+00:00",
    10: "2026-11-13T01:15:00+00:00",
    11: "2026-11-20T01:15:00+00:00",
    12: "2026-11-26T01:00:00+00:00",
    13: "2026-12-04T01:15:00+00:00",
    14: "2026-12-11T01:15:00+00:00",
    15: "2026-12-18T01:15:00+00:00",
    16: "2026-12-25T01:15:00+00:00",
    17: "2027-01-01T01:15:00+00:00",
    18: "2027-01-10T18:00:00+00:00",
}

# The playoff rounds are absent from the recorded schedule until each is set, so their first-game
# dates (America/New_York calendar dates) come from the published 2026-season calendar: Wild Card,
# Divisional, Conference Championships, Super Bowl LXI.
PLAYOFF_FIRST_GAME_DATES: dict[int, str] = {
    19: "2027-01-16",
    20: "2027-01-23",
    21: "2027-01-31",
    22: "2027-02-14",
}

# D-02: only weeks 8 through 16 are ever evaluated as A/A weeks (week 7 has no complete shadow).
AA_FIRST_WEEK: int = 8
AA_LAST_WEEK: int = 16

# Each A/A week is evaluated on the cadence Tuesday of the NFL week after it.
AA_EVALUATION_TUESDAY_BY_WEEK: dict[int, str] = {
    8: "2026-11-03",
    9: "2026-11-10",
    10: "2026-11-17",
    11: "2026-11-24",
    12: "2026-12-01",
    13: "2026-12-08",
    14: "2026-12-15",
    15: "2026-12-22",
    16: "2026-12-29",
}

# D-02 / D-33: the re-fit dates in force, keyed by the FIRST CLEAN A/A WEEK. Each entry is
# (cadence Tuesday, the NFL week whose first game it precedes). Written out as literals so no date
# is ever derived at run time. D-06: a list keeps its last date even when that period holds only a
# few playoff games.
REFIT_DATE_LISTS: dict[int, tuple[tuple[str, int], ...]] = {
    8: (("2026-11-03", 9), ("2026-12-01", 13), ("2026-12-29", 17), ("2027-01-26", 21)),
    9: (("2026-11-10", 10), ("2026-12-08", 14), ("2027-01-05", 18), ("2027-02-09", 22)),
    10: (("2026-11-17", 11), ("2026-12-15", 15), ("2027-01-12", 19)),
    11: (("2026-11-24", 12), ("2026-12-22", 16), ("2027-01-19", 20)),
    12: (("2026-12-01", 13), ("2026-12-29", 17), ("2027-01-26", 21)),
    13: (("2026-12-08", 14), ("2027-01-05", 18), ("2027-02-09", 22)),
    14: (("2026-12-15", 15), ("2027-01-12", 19)),
    15: (("2026-12-22", 16), ("2027-01-19", 20)),
    16: (("2026-12-29", 17), ("2027-01-26", 21)),
}

# D-33: the end rule.
NO_CLEAN_WEEK_RULE: str = (
    "no week 8-16 clean: no re-fit runs in 2026, and the season close says so"
)


# ---------------------------------------------------------------------------
# The A/A proof (D-02, R7)
# ---------------------------------------------------------------------------

# The only fields allowed to differ between a live row and its shadow twin: the arm itself, the
# ledger envelope's position and chain hash, and the five fill columns (real fills are recorded on
# the live arm only). Every other immutable column, the envelope kind and canon_v, the grading half
# and the closing half must be identical.
AA_EXCLUDED_FIELDS: tuple[str, ...] = (
    "arm",
    "seq",
    "chain_hash",
    "fill_sportsbook",
    "fill_line",
    "fill_odds",
    "fill_stake_dollars",
    "fill_at_utc",
)


# ---------------------------------------------------------------------------
# Training (R3, D-26, D-37, RFIT-07)
# ---------------------------------------------------------------------------

# Every finished game from this season through the cutoff is a training row (kickoff strictly
# before the cutoff, final score required). Equal to the corpus start of conf/season_partition.py.
TRAINING_START_SEASON: int = 2002

# The input caps, per target.
INPUT_CAPS: dict[str, int] = {
    "wp": 20,
    "ats": 25,
    "ou": 25,
}

# The standing excluded feature groups (the ratified group verdict), never re-opened in-season.
EXCLUDED_FEATURE_GROUPS: tuple[str, ...] = ("injury", "situational", "snap")

# The ratified verdict file and its newline-normalized sha256 at the time of freezing.
GROUP_VERDICT_PATH: str = "config/group_gate_verdict.toml"
GROUP_VERDICT_NORMALIZED_SHA256: str = (
    "06e147083ad2d11026a45d2587b75851a95ff2306465b997e13ebee8dc949112"
)

# The fixed model settings, copied from each trainer's own defaults. No search of any kind.
MODEL_SETTINGS: dict[str, dict[str, object]] = {
    "wp": {
        "max_iter": 1000,
        "solver": "lbfgs",
        "C": 1.0,
        "random_state": 42,
    },
    "ats": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    },
    "ou": {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": 42,
        "verbosity": 0,
        "n_jobs": -1,
    },
}

# R15: every fit runs with the thread count in force pinned to one; any other count is refused.
THREAD_LIMIT: int = 1

# The WP probability calibrator, as the WP trainer builds it.
WP_CALIBRATOR: str = "platt, isotonic fallback"

# The ATS and O/U residual converter: a normal distribution over the held-out residuals.
LINE_RESIDUAL_CONVERTER: str = "normal"


# ---------------------------------------------------------------------------
# The re-measure of the bet rule and blend (R4, D-22, D-27..D-30, D-36)
# ---------------------------------------------------------------------------

# Residual-only seed seasons: they feed the lean, never a line-dependent measurement.
BIAS_SEED_SEASONS: tuple[int, ...] = (2017, 2018, 2019)

# The owned pre-lock corpus seasons (ATS and O/U), extended by finished 2026 games.
HISTORICAL_TUNE_SEASONS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)

# WP's corpus seasons: a 2020 WP row has no prior-fold converter and is excluded and counted.
WP_HISTORICAL_TUNE_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024)

# The historical walk-forward folds, one season each.
HISTORICAL_FOLD_SEASONS: tuple[int, ...] = (
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
)

# The spent single-use hold: no 2025 game enters any re-measured number.
SPENT_HOLD_SEASON: int = 2025

# D-30: the converter whose slope_beta prices a finished 2026 game's WP market side (fitted on
# 2020-2024, strictly before 2026; the converter the live path and the blends are bound to).
MARKET_CONVERTER_ID: str = "market_probability_20260923_195443"

# The WP edge-band anchor pair (high, medium), unchanged in every generation.
WP_EDGE_ANCHOR: tuple[float, float] = (0.05, 0.02)

# The O/U edge's denominator floor (the edge is a ratio of the market total).
OU_TOTAL_FLOOR: float = 30.0

# ATS and O/U band pairs are numpy linear-method quantiles reproducing WP's band shares, rounded.
EDGE_QUANTILE_METHOD: str = "linear"
EDGE_THRESHOLD_DECIMALS: int = 4

# A target with fewer usable re-measure rows than this is an integrity refusal, never a no-bets
# generation.
MIN_USABLE_REMEASURE_ROWS: int = 100

# The EV floors, FIXED for every generation (owner ruling). Never re-picked.
EV_FLOORS: dict[str, float] = {
    "wp": 0.0,
    "ats": 0.0,
    "ou": 0.01,
}

# D-27 as clarified by owner ruling A (2026-10-20): the population of the lean and the error spread.
LEAN_POOL_RULE: str = (
    "lean pool: every game of the seed seasons 2017-2019, plus only the 2020-2024 and "
    "finished 2026 games that have an owned pre-lock line (the corpus); the error spread "
    "uses the same 2020-2024 and 2026 corpus games; a 2020-2024 or 2026 game without an "
    "owned pre-lock line enters neither (owner ruling 2026-10-20, D-27)"
)


# ---------------------------------------------------------------------------
# Integrity checks (R5, D-34, D-25, P4)
# ---------------------------------------------------------------------------

# Every check a re-fit must pass before it goes live, in the order the code runs them. The first
# failure refuses the attempt by name and leaves artifacts/latest.json byte-unchanged.
INTEGRITY_CHECKS: tuple[str, ...] = (
    "recipe_fingerprint",
    "aa_proof",
    "training_scope",
    "input_pool_and_caps",
    "fixed_settings",
    "out_of_sample_calibration",
    "remeasure_complete",
    "scores_frame",
    "dry_stamp_and_validate",
    "clean_code_commit",
    "bundle_validation",
)


# ---------------------------------------------------------------------------
# The deciding comparison (D-07..D-11, D-18, D-31, D-32)
# ---------------------------------------------------------------------------

# D-07: the test on per-game loss gaps (live loss minus frozen loss).
COMPARISON_TEST: str = "paired t-test on per-game loss gaps, two-sided (Diebold-Mariano, independent-games form)"

# D-08: two-sided significance level (equal to backtest.diagnose.SIGNIFICANCE_ALPHA).
ALPHA: float = 0.05

# D-08 / D-32: the family is every (target, generation >= 1) cell with MIN_PAIRED_GAMES or more.
MULTIPLE_TEST_CORRECTION: str = "benjamini-hochberg"

# D-09: below this many paired games a cell prints its count and "too few to test".
MIN_PAIRED_GAMES: int = 30

# D-11: WP probabilities are clipped to [LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP] before log loss.
LOG_LOSS_CLIP: float = 1e-15

# D-31: in WP log loss a tied game is labelled 0 (the home team did not win).
WP_TIE_LABEL: int = 0

# D-10: the per-target roll-up words.
ROLLUP_WORDS: tuple[str, ...] = (
    "HELPED",
    "HURT",
    "MIXED",
    "NO DETECTABLE DIFFERENCE",
)
