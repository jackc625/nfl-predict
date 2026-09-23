"""SUPERSEDING CORRECTION of the frozen 2026 cold-start rule (Plan 33.2-26, SPEC R14).

GENERATOR OUTPUT. Emitted by ``python -m scripts.derive_cold_start_constants --corrected``.
Do NOT hand-edit any value below; re-run the derivation.

WHAT IT SUPERSEDES. Commit ``11761c7`` (11761c7ece83ab9cab8ce73ffd6d7b58158ee703) froze the 2026 edge-band
thresholds and the 2026 chain-fit bias in ``backtest/cold_start_constants.py`` +
``COLD-START-PREREGISTRATION.md``. Those values were derived from models fitted on inputs later
found defective and from CLOSING lines, and those models are gone. This module SUPERSEDES them.
The originals are byte-unchanged and remain the record of what was frozen and when; nothing here
edits them. ``COLD-START-CORRECTION.md`` is the human-readable half of this record.

WHAT MOVED, symbol by symbol:
  * ``EDGE_TIER_THRESHOLDS_BY_TARGET`` / ``_UNROUNDED`` -- re-derived on the corrected models'
    walk-forward predictions over the owned PRE-LOCK lines (2020-2024), never a closing
    line. WP's market side is the spread-derived probability converted OUT OF FOLD
    (``models.market_probability.oof_market_probability``); its first season has no prior fold
    and leaves the WP part as ``no_prior_fold_converter``. A target below
    ``MIN_HONEST_THRESHOLD_ROWS`` gets ``None`` -- NO threshold, therefore NO bets and NO edge
    band -- never a default, a zero or the old value (``EDGE_TIER_THRESHOLD_REFUSALS`` says why);
  * ``WP_ANCHOR_BAND_SHARES`` -- re-measured on that population;
  * ``CHAIN_FIT_BIAS_2026`` / ``_SEASONS`` / ``_BY_SEASON`` -- the corrected chain fit's own
    walk-forward bias series (Plan 33.2-29, ``outputs/p332/corrected_chain_fit.json``) extended
    to ``CHAIN_FIT_BIAS_TARGET_SEASON``: the derivation reproduces every season that record prices
    before extending it. The pool ends at 2024 because no 2025 row is read, which is why the target
    season is NAMED here rather than derived as the pool's last season plus one;
  * ``THRESHOLD_DERIVATION_POPULATION``, ``EDGE_TIER_THRESHOLD_UNITS`` (WP's market side),
    ``CHAIN_FIT_BIAS_SOURCE_BY_TARGET``, ``DERIVATION_*`` and the band tables -- restated for the
    corrected population. "UNDER_CURRENT" now means under the superseded 11761c7 pairs.

WHAT DID NOT MOVE: WP's 0.05 / 0.02 pair (the anchor), the WP-anchored band-share quantile rule
(``THRESHOLD_QUANTILE_CONVENTION``, ``numpy.quantile`` method "linear"), the STRICT ``>`` bands,
the digest refusals, the walk-forward bias estimator
(``backtest.ou_ev_chain.estimate_prior_season_bias``) and ``FIX_CYCLE_ALLOWANCE``.

THE OTHER TWO MOVED PARTS OF THE 2026 BET RULE -- the EV floor and the frozen residual SD -- are
superseded separately, naming ``ee20773``: ``backtest.corrected_ev_chain_constants`` and
``EV-CHAIN-CORRECTION.md`` (Plan 33.2-29).

NOT CLEAN EVIDENCE (D33.2-07): every number here is re-measured on past seasons. It sets a
threshold; it does not show one is profitable. Only the 2026 season, recorded live, counts.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

# THE TWO FILES THAT ARE THIS CORRECTION, and the two it supersedes. Repo-root-relative POSIX.
PREREGISTRATION_PATHS: tuple[str, ...] = (
    "COLD-START-CORRECTION.md",
    "backtest/corrected_cold_start_constants.py",
)
SUPERSEDED_PREREGISTRATION_PATHS: tuple[str, ...] = (
    "COLD-START-PREREGISTRATION.md",
    "backtest/cold_start_constants.py",
)
SUPERSEDED_PREREGISTRATION_COMMIT: str = "11761c7ece83ab9cab8ce73ffd6d7b58158ee703"

# THE CORRECTED EDGE THRESHOLDS, (HIGH, MEDIUM) per target on its OWN unit, four decimal places.
# ``None`` = NO honest threshold: no bets and no edge band for that target.
EDGE_TIER_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float] | None] = {
    "ats": (1.7590, 0.7287),
    "ou": (0.0438, 0.0173),
    "wp": (0.0500, 0.0200),
}

EDGE_TIER_THRESHOLDS_UNROUNDED: dict[str, tuple[float, float] | None] = {
    "ats": (1.7590361478016774, 0.7287012093239492),
    "ou": (0.04384485976450175, 0.01731414904525186),
    "wp": (0.05, 0.02),
}

# Why a target has no threshold, when one has none. Empty when every target has one.
EDGE_TIER_THRESHOLD_REFUSALS: dict[str, str] = {}

# The pairs this correction supersedes, READ from the frozen 11761c7 module and restated so the
# old value sits beside the new one.
SUPERSEDED_EDGE_TIER_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float]] = {
    "ats": (1.9493, 0.8359),
    "ou": (0.0546, 0.0220),
    "wp": (0.0500, 0.0200),
}

EDGE_TIER_THRESHOLD_UNITS: dict[str, str] = {
    "ats": "points (signed model-minus-market home margin)",
    "ou": "ratio of the market total, floored at 30",
    "wp": "probability (model minus the spread-derived out-of-fold market probability)",
}

THRESHOLD_QUANTILE_CONVENTION: str = (
    'numpy.quantile(magnitudes, q, method="linear"); q taken from WP band shares under its '
    "unchanged 0.05 / 0.02 pair; bands assigned with STRICT > so a value exactly at a "
    "threshold falls in the LOWER band"
)

# Below this many rows with a computable edge a target gets NO threshold (the calibration gate's
# n >= 100 is the nearest existing floor).
MIN_HONEST_THRESHOLD_ROWS: int = 100

# WP's measured band shares under its unchanged pair -- the ANCHOR the other two reproduce.
WP_ANCHOR_BAND_SHARES: dict[str, float] | None = {
    "low": 0.1263,
    "medium": 0.1976,
    "high": 0.6761,
}

THRESHOLD_DERIVATION_SEASONS: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024)
THRESHOLD_WP_DERIVATION_SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024)
THRESHOLD_WINDOW_REASON: str = "2020-2024 is the whole honest corpus, not a preference: 2018-2019 are unbuyable at any price and 2025 has no free pre-lock source. The single-use 2025 hold is spent and no row of it enters this derivation."

# The WP rows with a pre-lock line whose season has no prior-fold converter slope. Counted,
# never filled with the serving slope or a neighbouring season's.
WP_EXCLUDED_NO_PRIOR_FOLD_CONVERTER: int = 255

# Scheduled games in the window with no owned line at or before their lock.
EXCLUDED_NO_PRELOCK_LINE: int = 60

THRESHOLD_DERIVATION_POPULATION: str = (
    "the owned pre-lock corpus (silver odds_timeline, the last line at or before each "
    "game's own lock) over 2020-2024, scored with the corrected models' walk-forward "
    "predictions (each season predicted by a fit on strictly earlier seasons); WP over "
    "2021-2024 only, its market side converted out of fold; shares taken over the rows "
    "carrying a COMPUTABLE edge"
)

# THE 2026 CHAIN-FIT BIAS, per target: the pooled mean residual (actual - predicted) over every
# strictly-prior season the corrected chain fit's walk-forward rule pools.
CHAIN_FIT_BIAS_2026: dict[str, float] = {
    "ats": -0.8288653630898939,
    "ou": 0.6443360853661155,
    "wp": 0.0004000931458235923,
}

CHAIN_FIT_BIAS_TARGET_SEASON: int = 2026

CHAIN_FIT_BIAS_SEASONS: tuple[int, ...] = (
    2017,
    2018,
    2019,
    2020,
    2021,
    2022,
    2023,
    2024,
)

# The per-season mean residual each pooled bias was estimated from. INT season keys; a JSON
# representation of the same mapping carries STRING keys.
CHAIN_FIT_BIAS_BY_SEASON: dict[str, dict[int, float]] = {
    "ats": {
        2017: 0.07538418019232287,
        2018: -0.30350575005907693,
        2019: -3.0113723609589513,
        2020: -3.0362185444317613,
        2021: -0.4195864728912163,
        2022: -0.2707259616508695,
        2023: 0.5933755579103223,
        2024: -0.41710414545838354,
    },
    "ou": {
        2017: -1.4282359005360121,
        2018: 4.206596710262227,
        2019: 0.7720666663923513,
        2020: 4.720768962186925,
        2021: 1.0451546837301815,
        2022: -2.1073476925107384,
        2023: -1.9253310736487894,
        2024: 0.1561770816501096,
    },
    "wp": {
        2017: 0.029658335076788927,
        2018: 0.002310226233564815,
        2019: -0.04571817191243725,
        2020: -0.08947241382178084,
        2021: 0.014619848060954056,
        2022: 0.049001418640106945,
        2023: 0.04081912074811254,
        2024: -0.0036416267692437653,
    },
}

SUPERSEDED_CHAIN_FIT_BIAS_2026: dict[str, float] = {
    "ats": 0.257407648096468,
    "ou": -0.35080281804116925,
    "wp": -0.03377244391544111,
}

CHAIN_FIT_BIAS_SOURCE_BY_TARGET: dict[str, dict[str, object]] = {
    "ats": {
        "artifact": "ats_20260923_172148",
        "kind": "owner_ruled_swap",
        "residuals": "walk-forward refits of this artifact's own recorded recipe",
        "verdict": "NOT_GATED (SPEC R13: owner readiness ruling, no pass/fail gate)",
    },
    "ou": {
        "artifact": "ou_20260923_172152",
        "kind": "owner_ruled_swap",
        "residuals": "walk-forward refits of this artifact's own recorded recipe",
        "verdict": "NOT_GATED (SPEC R13: owner readiness ruling, no pass/fail gate)",
    },
    "wp": {
        "artifact": "wp_20260923_172144",
        "kind": "owner_ruled_swap",
        "residuals": "walk-forward refits of this artifact's own recorded recipe",
        "verdict": "NOT_GATED (SPEC R13: owner readiness ruling, no pass/fail gate)",
    },
}

# The corrected chain fit this bias series continues (Plan 33.2-29), read by explicit path.
CHAIN_FIT_SOURCE_RECORD_PATH: str = "outputs/p332/corrected_chain_fit.json"
CHAIN_FIT_SOURCE_RECORD_ID: str = "corrected_chain_fit_20260923_231233"

# The corrected model artifacts every number above was derived from, the live blend, and the
# converter that blend binds.
DERIVATION_ARTIFACTS: dict[str, str] = {
    "ats": "ats_20260923_172148",
    "ou": "ou_20260923_172152",
    "wp": "wp_20260923_172144",
}
LIVE_BLEND_ARTIFACT_ID: str = "blend_20260923_212418"
MARKET_PROBABILITY_ARTIFACT_ID: str = "market_probability_20260923_195443"
DERIVATION_GOLD_GENERATION: str = (
    "484397642530db5b28c49d9234ecfb90e3860783f1b1b41766ab6151a1522597"
)

# The OpenMP / BLAS thread count every walk-forward fit was pinned to.
DERIVATION_THREAD_LIMIT: int = 1

# EVERY input this derivation read, with its digest (gitignored stores over RAW bytes; an
# ``artifact:<id>`` key is the sha256 over the directory's sorted file manifest).
DERIVATION_INPUT_DIGESTS: dict[str, str] = {
    "artifact:ats_20260923_172148": "a821e50b2440281e8abd4403139ffea92dd4dd1083862e7feb8c391f66b7c05a",
    "artifact:blend_20260923_212418": "14caf6ac3960dd21d6bd391c6eaa9fee0ed3f5280c7f4c7496ac76a248aedef8",
    "artifact:market_probability_20260923_195443": "aded80aeca70cebd537c008ba033dd909c316397bbfebe06b9889b7af1827d3b",
    "artifact:ou_20260923_172152": "7acdaeaa061aeb589c83177db03d7c53e10265aa29b71db9514096394e193037",
    "artifact:wp_20260923_172144": "65d49530697b28901ed064b58aa4594d0b64db7d97e2349746d5a1ee756523b6",
    "artifacts/latest.json": "3f1cbe3d9c3f8190a2ca3a4fed21b11510dcfea2159c036fb23874f32c9e8830",
    "data/gold/features_ats.parquet": "6b477f6878e86e641af5bc1eac4bd1dfa02a5521769f5c5d66f03635422bd246",
    "data/gold/features_ou.parquet": "96c655c53fa697176fc9fc6f41b8436960cdcbbf0b574f4bf93da922f67822b6",
    "data/gold/features_wp.parquet": "1807d2d8b3f845e9105761964ce3d319a8103856d47ac61fd3d1644346ecc2a3",
    "data/silver/games.parquet": "ef7cdf88adba09278cbb3b1f122327ed6f629d3cfc3309ba3cf0dacccddd46e8",
    "data/silver/odds_timeline.parquet": "cab6efa1e955e2fbd3dbd9ee35e2a34d5e258caad0e08604001ed852267a53ca",
    "outputs/p332/corrected_chain_fit.json": "dffa646531a51e6fee155df43d8c3f6e294791a5c8a6914f621898ecf8ac5b46",
}

DERIVATION_FORMATTER: str = "ruff 0.15.6"

DERIVATION_ELIGIBLE_COUNTS: dict[str, dict[str, int]] = {
    "ats": {
        "bias_rows": 2149,
        "bias_seasons": 8,
        "threshold_population_rows": 1348,
        "threshold_rows": 1348,
    },
    "ou": {
        "bias_rows": 2149,
        "bias_seasons": 8,
        "threshold_population_rows": 1348,
        "threshold_rows": 1348,
    },
    "wp": {
        "bias_rows": 2149,
        "bias_seasons": 8,
        "threshold_population_rows": 1348,
        "threshold_rows": 1093,
    },
}

# THE LABEL MOVEMENT on the corrected edges: under the superseded 11761c7 pairs, and under the
# corrected pairs. A target with no threshold has no row.
BAND_SHARES_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, float]] = {
    "ats": {"low": 0.1461, "medium": 0.2203, "high": 0.6335},
    "ou": {"low": 0.1558, "medium": 0.2507, "high": 0.5935},
    "wp": {"low": 0.1263, "medium": 0.1976, "high": 0.6761},
}

BAND_COUNTS_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, int]] = {
    "ats": {"low": 197, "medium": 297, "high": 854},
    "ou": {"low": 210, "medium": 338, "high": 800},
    "wp": {"low": 138, "medium": 216, "high": 739},
}

ATS_BAND_SHARES_AFTER: dict[str, dict[str, float]] = {
    "ats": {"low": 0.1269, "medium": 0.1973, "high": 0.6758},
    "ou": {"low": 0.1261, "medium": 0.1973, "high": 0.6766},
    "wp": {"low": 0.1263, "medium": 0.1976, "high": 0.6761},
}

ATS_BAND_COUNTS_AFTER: dict[str, dict[str, int]] = {
    "ats": {"low": 171, "medium": 266, "high": 911},
    "ou": {"low": 170, "medium": 266, "high": 912},
    "wp": {"low": 138, "medium": 216, "high": 739},
}

GAMES_CHANGING_BAND: dict[str, int] = {
    "ats": 83,
    "ou": 152,
    "wp": 0,
}

# Plan 33-08's declared allowance, carried unchanged.
FIX_CYCLE_ALLOWANCE: int = 0

# Where the other two moved parts of the 2026 bet rule live (Plan 33.2-29, superseding ee20773).
EV_CHAIN_CORRECTION_MODULE: str = "backtest.corrected_ev_chain_constants"
