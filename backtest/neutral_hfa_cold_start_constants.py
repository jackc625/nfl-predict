"""SUPERSEDING CORRECTION of the 2026 cold-start rule recorded at 9bb7568 (row 19).

GENERATOR OUTPUT. Emitted by ``python -m scripts.derive_cold_start_constants --corrected --correction row19``.
Do NOT hand-edit any value below; re-run the derivation.

WHAT IT SUPERSEDES. Commit ``9bb7568`` (9bb7568f78c3b2714fb03da0818104ab7c10f620) recorded the 2026 edge-band
thresholds and the 2026 chain-fit bias in ``backtest/corrected_cold_start_constants.py`` +
``COLD-START-CORRECTION.md``, measured on the three models production served from 2026-09-23.
Owner ruling 2026-10-03 ~22:12 ET (WINDOWS row 19, quick task 261003-vke): zero Elo
home-field advantage at every neutral site, neutral games left out of HFA learning. Elo, gold
and all three models were re-derived under that rule and swapped into production on 2026-10-04.
Owner ruling 2026-10-03 ~22:50 ET ("Re-measure, same recipe"): the unchanged recipe is re-run on
the new production models, and this module SUPERSEDES the 9bb7568 values. The 9bb7568 files are
byte-unchanged and stay importable as the record of what was measured and when; nothing here
edits them. ``NEUTRAL-HFA-BET-RULE-CORRECTION.md`` is the human-readable half of this record.

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
    walk-forward bias series (quick task 261003-vke, ``outputs/row19/neutral_hfa_chain_fit.json``) extended
    to ``CHAIN_FIT_BIAS_TARGET_SEASON``: the derivation reproduces every season that record prices
    before extending it. The pool ends at 2024 because no 2025 row is read, which is why the target
    season is NAMED here rather than derived as the pool's last season plus one;
  * ``THRESHOLD_DERIVATION_POPULATION``, ``EDGE_TIER_THRESHOLD_UNITS`` (WP's market side),
    ``CHAIN_FIT_BIAS_SOURCE_BY_TARGET``, ``DERIVATION_*`` and the band tables -- restated for the
    corrected population. "UNDER_CURRENT" now means under the superseded 9bb7568 pairs.

WHAT DID NOT MOVE: WP's 0.05 / 0.02 pair (the anchor), the WP-anchored band-share quantile rule
(``THRESHOLD_QUANTILE_CONVENTION``, ``numpy.quantile`` method "linear"), the STRICT ``>`` bands,
the digest refusals, the walk-forward bias estimator
(``backtest.ou_ev_chain.estimate_prior_season_bias``) and ``FIX_CYCLE_ALLOWANCE``.

THE OTHER TWO MOVED PARTS OF THE 2026 BET RULE -- the EV floor and the frozen residual SD -- are
re-measured by the same recipe on the same new models, superseding the 8c9675e record:
``backtest.neutral_hfa_ev_chain_constants`` (record ``outputs/row19/neutral_hfa_chain_fit.json``).

NOT CLEAN EVIDENCE (D33.2-07): every number here is re-measured on past seasons. It sets a
threshold; it does not show one is profitable. Only the 2026 season, recorded live, counts.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

# THE TWO FILES THAT ARE THIS CORRECTION, and the two it supersedes. Repo-root-relative POSIX.
PREREGISTRATION_PATHS: tuple[str, ...] = (
    "NEUTRAL-HFA-BET-RULE-CORRECTION.md",
    "backtest/neutral_hfa_cold_start_constants.py",
)
SUPERSEDED_PREREGISTRATION_PATHS: tuple[str, ...] = (
    "COLD-START-CORRECTION.md",
    "backtest/corrected_cold_start_constants.py",
)
SUPERSEDED_PREREGISTRATION_COMMIT: str = "9bb7568f78c3b2714fb03da0818104ab7c10f620"

# THE CORRECTED EDGE THRESHOLDS, (HIGH, MEDIUM) per target on its OWN unit, four decimal places.
# ``None`` = NO honest threshold: no bets and no edge band for that target.
EDGE_TIER_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float] | None] = {
    "ats": (1.6229, 0.6211),
    "ou": (0.0452, 0.0162),
    "wp": (0.0500, 0.0200),
}

EDGE_TIER_THRESHOLDS_UNROUNDED: dict[str, tuple[float, float] | None] = {
    "ats": (1.6228850478986372, 0.6211200976524457),
    "ou": (0.04515015626037483, 0.016176531289705882),
    "wp": (0.05, 0.02),
}

# Why a target has no threshold, when one has none. Empty when every target has one.
EDGE_TIER_THRESHOLD_REFUSALS: dict[str, str] = {}

# The pairs this correction supersedes, READ from the frozen 9bb7568 module and restated so the
# old value sits beside the new one.
SUPERSEDED_EDGE_TIER_THRESHOLDS_BY_TARGET: dict[str, tuple[float, float]] = {
    "ats": (1.7590, 0.7287),
    "ou": (0.0438, 0.0173),
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
    "low": 0.129,
    "medium": 0.1876,
    "high": 0.6834,
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
    "ats": -0.4754073948548693,
    "ou": 0.0974858578219975,
    "wp": -0.0001220760959361301,
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
        2017: -0.08844159563047609,
        2018: -0.09029806557387002,
        2019: -2.7688641342358866,
        2020: -1.6671241137765203,
        2021: 0.28774532249799983,
        2022: -0.41512355568554127,
        2023: 1.1600155813674278,
        2024: -0.32667665974163324,
    },
    "ou": {
        2017: -2.1129519716184237,
        2018: 1.1795482278316656,
        2019: 0.8075946279232868,
        2020: 4.87505708582261,
        2021: 0.6777592687045827,
        2022: -2.4235070851456197,
        2023: -2.1400879972121296,
        2024: 0.19594395589485442,
    },
    "wp": {
        2017: 0.03157727868170018,
        2018: 0.0071141113940819305,
        2019: -0.04714414557474496,
        2020: -0.0889215931085167,
        2021: 0.015370241513727727,
        2022: 0.04436912596516579,
        2023: 0.03563000367767827,
        2024: -0.004412048193864272,
    },
}

SUPERSEDED_CHAIN_FIT_BIAS_2026: dict[str, float] = {
    "ats": -0.8288653630898939,
    "ou": 0.6443360853661155,
    "wp": 0.0004000931458235923,
}

CHAIN_FIT_BIAS_SOURCE_BY_TARGET: dict[str, dict[str, object]] = {
    "ats": {
        "artifact": "ats_20261004_050228",
        "kind": "owner_ruled_swap",
        "residuals": "walk-forward refits of this artifact's own recorded recipe",
        "verdict": "NOT_GATED (owner standing rule: models fitted on the pre-fix Elo are dead; the row-19 re-fits are promoted regardless of any adverse measurement)",
    },
    "ou": {
        "artifact": "ou_20261004_050232",
        "kind": "owner_ruled_swap",
        "residuals": "walk-forward refits of this artifact's own recorded recipe",
        "verdict": "NOT_GATED (owner standing rule: models fitted on the pre-fix Elo are dead; the row-19 re-fits are promoted regardless of any adverse measurement)",
    },
    "wp": {
        "artifact": "wp_20261004_050223",
        "kind": "owner_ruled_swap",
        "residuals": "walk-forward refits of this artifact's own recorded recipe",
        "verdict": "NOT_GATED (owner standing rule: models fitted on the pre-fix Elo are dead; the row-19 re-fits are promoted regardless of any adverse measurement)",
    },
}

# The corrected chain fit this bias series continues (quick task 261003-vke), read by explicit path.
CHAIN_FIT_SOURCE_RECORD_PATH: str = "outputs/row19/neutral_hfa_chain_fit.json"
CHAIN_FIT_SOURCE_RECORD_ID: str = "neutral_hfa_chain_fit_20261004_052326"

# The corrected model artifacts every number above was derived from, the live blend, and the
# converter that blend binds.
DERIVATION_ARTIFACTS: dict[str, str] = {
    "ats": "ats_20261004_050228",
    "ou": "ou_20261004_050232",
    "wp": "wp_20261004_050223",
}
LIVE_BLEND_ARTIFACT_ID: str = "blend_20261004_050521"
MARKET_PROBABILITY_ARTIFACT_ID: str = "market_probability_20260923_195443"
DERIVATION_GOLD_GENERATION: str = (
    "9ba3a56885ab3b26524d2255e73b46bf674c9b18043cd7ca2bcc74a71cab9228"
)

# The OpenMP / BLAS thread count every walk-forward fit was pinned to.
DERIVATION_THREAD_LIMIT: int = 1

# EVERY input this derivation read, with its digest (gitignored stores over RAW bytes; an
# ``artifact:<id>`` key is the sha256 over the directory's sorted file manifest).
DERIVATION_INPUT_DIGESTS: dict[str, str] = {
    "artifact:ats_20261004_050228": "e26d2236ce75aa4d1664808b7125b3cf8f65a466a5fbb5b9d76c9f17aa91e837",
    "artifact:blend_20261004_050521": "207f682e5318403f1d8580c960647c5704ad3f4f77bf208e16c54220537eecf6",
    "artifact:market_probability_20260923_195443": "aded80aeca70cebd537c008ba033dd909c316397bbfebe06b9889b7af1827d3b",
    "artifact:ou_20261004_050232": "ed47d4075e1320bb0aae8bd6d723878c53afefaf863595df84093e23244c9aeb",
    "artifact:wp_20261004_050223": "6860b10f15cfe7d81a7ce979bc6591d7f3d60472d4a16c888712b82ddbe94d07",
    "artifacts/latest.json": "0a3a3b34b7be1004052f858aa1d425d93e37bd00e7398457d0ba2301e10a5b6a",
    "data/gold/features_ats.parquet": "017fa30b053a19f2256266bf2117a639e1d59d3de82113467d45915c9c879767",
    "data/gold/features_ou.parquet": "ffcc07afd6fc25e7557849e6dd85d83f475d9346d0eb0e3816d5f08bf1dd1eb8",
    "data/gold/features_wp.parquet": "c7bbecd28b56ac36148507d4f34be0d2a2185c7c0fcf960318610f08e5387ba7",
    "data/silver/games.parquet": "682f027fa6770ed3bbd82b95ca5f2f2e3a6073058cd7c78619090cbd278e797d",
    "data/silver/odds_timeline.parquet": "cab6efa1e955e2fbd3dbd9ee35e2a34d5e258caad0e08604001ed852267a53ca",
    "outputs/row19/neutral_hfa_chain_fit.json": "fab1118b8093bf55fd1cd4d7d8ba80eff3f95b7bd9537c4cf4f3a6b2a6ce2906",
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

# THE LABEL MOVEMENT on the corrected edges: under the superseded 9bb7568 pairs, and under the
# corrected pairs. A target with no threshold has no row.
BAND_SHARES_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, float]] = {
    "ats": {"low": 0.1491, "medium": 0.1951, "high": 0.6558},
    "ou": {"low": 0.1343, "medium": 0.1736, "high": 0.6921},
    "wp": {"low": 0.129, "medium": 0.1876, "high": 0.6834},
}

BAND_COUNTS_UNDER_CURRENT_THRESHOLDS: dict[str, dict[str, int]] = {
    "ats": {"low": 201, "medium": 263, "high": 884},
    "ou": {"low": 181, "medium": 234, "high": 933},
    "wp": {"low": 141, "medium": 205, "high": 747},
}

ATS_BAND_SHARES_AFTER: dict[str, dict[str, float]] = {
    "ats": {"low": 0.1291, "medium": 0.1877, "high": 0.6832},
    "ou": {"low": 0.1291, "medium": 0.1877, "high": 0.6832},
    "wp": {"low": 0.129, "medium": 0.1876, "high": 0.6834},
}

ATS_BAND_COUNTS_AFTER: dict[str, dict[str, int]] = {
    "ats": {"low": 174, "medium": 253, "high": 921},
    "ou": {"low": 174, "medium": 253, "high": 921},
    "wp": {"low": 141, "medium": 205, "high": 747},
}

GAMES_CHANGING_BAND: dict[str, int] = {
    "ats": 64,
    "ou": 19,
    "wp": 0,
}

# Plan 33-08's declared allowance, carried unchanged.
FIX_CYCLE_ALLOWANCE: int = 0

# Where the other two moved parts of the 2026 bet rule live (quick task 261003-vke, superseding 8c9675e).
EV_CHAIN_CORRECTION_MODULE: str = "backtest.neutral_hfa_ev_chain_constants"
