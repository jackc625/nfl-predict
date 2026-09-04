"""The FROZEN Phase-31 pre-registration for the three-target EV chain (SPEC R1/R3/R7, PROD-02).

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/group_gate_constants.py`` (D30-05/D30-11).
This module IS the Phase-31 pre-registration, not a description of one. Its LAST-MODIFYING COMMIT
is the git-ancestry anchor: the readout guard requires that commit to be a strict git ANCESTOR of
the commit recording the 2025 measurement output. That ancestry assertion is the only mechanical
proof this phase has that the rule was fixed before the numbers existed.

Stated plainly because it is easy to forget fifteen plans later: EDITING THIS FILE AFTER THE
MEASUREMENT COMMIT DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. There is no honest repair path.
A value that is wrong here is wrong for the remainder of the phase, and the only legitimate
response is to say so in the readout, not to amend the file.

THIS MODULE IS FROZEN AS OF THIS COMMIT (Plan 31-05, ratified at CHECKPOINT 1). Plan 31-01 seeded
it with the two facts D31-24 and D31-36 already fixed, and said Plan 31-05 would land the rest and
freeze it. That has now happened: the tune/hold season split, the game-type scope, the odds label
and juice source, the ATS residual contract, the WP chain policy, the eligibility rules, the
robustness cuts, the BH family spec, the ROI hypothesis test, the run-ledger states, the rehearsal
proxy split and the readout guards are all declared below. Adding a constant here is no longer
expected; changing one is not permitted.

IT IS ONE PRE-REGISTRATION IN TWO FILES. ``PROFITABILITY-PREREGISTRATION.md`` at the repo root is
the human-readable half the owner ratifies, and it states in prose everything that has no constant
(the per-target chain steps, the ingest write contract, the fingerprint ladder and its build-clock
exemption, the forward-row grading contract, the three narrowings). Their COMBINED last-modifying
commit is the anchor; ``PREREGISTRATION_PATHS`` below names both so every guard resolves them from
one place.

NEITHER FILE RECORDS ITS OWN CONTENT HASH. A document that must CONTAIN and exactly REPRODUCE its
own whole-file hash is self-referential: writing the hash changes the bytes the hash was computed
over, so no fixed point exists without a canonical exclusion rule nobody has defined. The witness
lives OUTSIDE the witnessed files, exactly as Phase 30 proved: ``tests/phase31_state.py`` records
``PRE_REGISTRATION_COMMIT`` and ``PRE_REGISTRATION_FILE_SHA256`` in a LATER commit under its APPEND
PROTOCOL, and ``tests/unit/test_preregistration_ancestry.py`` recomputes and compares.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import math
from collections.abc import Mapping

# Import-the-primitive parity seam (D24-13, applied again here). alpha is IMPORTED, never
# re-declared: a second alpha would be a second answer. This mirrors backtest/group_gate_constants
# from the Phase-30 side and config/gate.toml:29-30 from the config side.
from backtest.diagnose import SIGNIFICANCE_ALPHA

# The Phase-27 bootstrap configuration, CONSUMED unchanged (D31 discretion item 4). Imported
# rather than restated so a test can assert identity: a copied literal that happened to read 2000
# today would be a second declaration that can drift. Importing these does NOT import the Phase-27
# WINDOW -- TUNE_SEASONS and HOLD_SEASONS live in backtest/ou_ev_chain.py and are deliberately not
# imported anywhere in Phase-31 code (D31-14), which tests/unit/test_p31_constants_isolation.py
# asserts by AST scan.
from backtest.ou_monetization import (
    BOOTSTRAP_B,
    BOOTSTRAP_CI_TYPE,
    BOOTSTRAP_SEED,
)

__all__ = [
    "ALPHA",
    "ATS_BIAS_CARRIED",
    "ATS_RESIDUAL_BY_SEASON_P31",
    "ATS_RESIDUAL_CONTRACT",
    "ATS_RESIDUAL_POOLED_P31",
    "BH_FAMILY_SPEC",
    "BOOTSTRAP_B",
    "BOOTSTRAP_CI_TYPE",
    "BOOTSTRAP_DISCLOSURE_P31",
    "BOOTSTRAP_SEED",
    "CLV_P_VALUE_IS_REPORT_ONLY",
    "EV_TIER_BANDS",
    "EV_TIER_HIGH",
    "EV_TIER_LABELS",
    "EV_TIER_LOW",
    "EV_TIER_MEDIUM",
    "HOLD_GAME_TYPES",
    "HOLD_SEASONS_P31",
    "JUICE_SOURCE",
    "ODDS_SPORTSBOOK_LABEL",
    "OU_ELIGIBILITY_FROZEN",
    "PREREGISTRATION_PATHS",
    "PRIOR_RESIDUAL_SEASONS_P31",
    "READOUT_FORBIDDEN_FIGURES",
    "READOUT_FORBIDDEN_WORDS",
    "READOUT_PERMITTED_FIGURES",
    "READOUT_REQUIRED_FIELDS",
    "REHEARSAL_PROXY_SPLIT",
    "REPLAY_SNAPSHOT_TS_POLICY",
    "ROBUSTNESS_CUTS_P31",
    "ROBUSTNESS_CUT_BH_COUNT",
    "ROI_MIN_ATTAINABLE_P",
    "ROI_P_VALUE_METHOD",
    "ROI_SIGNIFICANCE_SPEC",
    "RUN_LEDGER_PATH",
    "RUN_LEDGER_STATES",
    "RUN_LEDGER_TRANSITION_RULE",
    "TRIAL_ENTRY_KIND_CONTROL",
    "TRIAL_ENTRY_KIND_INFERENCE",
    "TRIAL_REGISTRY_FIELDS_P31",
    "TUNE_GAME_TYPES",
    "TUNE_SEASONS_P31",
    "UNDISCHARGEABLE_NO_BETS",
    "VERDICT_TOKENS",
    "VERDICT_TOKEN_MEANINGS",
    "WP_ATS_ELIGIBILITY",
    "WP_CHAIN_POLICY",
    "assign_ev_tier",
]


# ---------------------------------------------------------------------------
# FORKING-PATHS GUARD: every value below is frozen HERE, BEFORE any 2025 number exists, and is
# NOT adjusted after seeing results (the D26-08 / D24-07 pre-registration discipline, mirroring
# backtest/ou_divergence.py, backtest/ou_ev_chain.py, backtest/ou_monetization.py and
# backtest/group_gate_constants.py).
#
# backtest/diagnose.py owns the significance primitive and alpha. backtest/ou_ev_chain.py owns the
# Phase-27 chain primitives (calibrated_p_over, devig, per_bet_ev, estimate_prior_season_bias,
# fit_frozen_residual_sd) and the Phase-27 WINDOW, which this phase does not read.
# backtest/ou_divergence.py owns the high-total eligibility boundary. backtest/ou_monetization.py
# owns the bootstrap configuration. This module adds ONLY the Phase-31 window, scope, policy and
# multiplicity declarations. It re-derives no metric and declares no second alpha.
# ---------------------------------------------------------------------------

# The significance level, bound BY IDENTITY to the one that already governs every CLV call and the
# model deploy gate in this repo. A float literal here would be a second alpha even if it happened
# to read 0.05 today.
ALPHA: float = SIGNIFICANCE_ALPHA


# ---------------------------------------------------------------------------
# The fit windows (D31-13, D31-38) and the game-type scope
# ---------------------------------------------------------------------------

# D31-13, an OWNER DECISION overriding 31-RESEARCH.md's recommended per-target split. The tune
# window is UNIFORM across all three targets and the single unburned 2025 season is the SOLE hold.
# Burnedness compromises EVALUATION, not FITTING, so leaving 2023-2024 unused as a
# nuisance-parameter source would waste two seasons of estimation data out of ceremony.
#
# THE NARROWING THIS FORCES, stated rather than discovered: "2023-2024 stays burned" now means NOT
# EVALUATED, not NOT TOUCHED. Those two seasons feed the frozen SD, the prior-season bias and the
# EV-floor sweep; no figure is EVALUATED on them and no verdict is drawn from them.
TUNE_SEASONS_P31: tuple[int, ...] = (2021, 2022, 2023, 2024)
HOLD_SEASONS_P31: tuple[int, ...] = (2025,)

# The strictly-prior residual pool that seeds the FIRST tune season's walk-forward bias estimate.
# 2021 has no prior tune season to estimate its bias from, so without a seed window the chain
# raises. 2018-2020 is the deployed artifacts' own train/val window: leakage-clean for this one
# purpose because these seasons seed ONLY the bias estimate. They are NEVER candidates, NEVER in
# the frozen-SD fit, NEVER in the threshold tuning and NEVER in the trial selection. Same value and
# same role as backtest/ou_monetization.PRIOR_RESIDUAL_SEASONS; restated rather than imported
# because the Phase-31 window is a different rule that happens to share this seed.
PRIOR_RESIDUAL_SEASONS_P31: tuple[int, ...] = (2018, 2019, 2020)

# D31-38, an OWNER DECISION overriding 31-RESEARCH.md's recommended tune-regular-season-only split.
# PLAYOFFS EVERYWHERE: playoff games are ingested and carried in BOTH the tune window and the hold.
# The 2025 hold is therefore 285 games, not 272.
#
# Two consequences carried rather than discovered. (1) The tune population is no longer
# byte-comparable to what Phase 27 and Phase 30 fit on -- but D31-13 ALREADY declared the O/U chain
# discontinuous with those publications by widening the window, so this is a SECOND reason for an
# already-declared discontinuity, not a fresh break in a clean lineage. The readout must state BOTH
# reasons. (2) `regular_season_only` becomes a GENUINE robustness cut on both sides of the split
# rather than a vacuous one, so it enters the BH family per D31-33 rather than leaving it.
#
# The literal set matches scripts/ingest_historical_odds.py's own admitted set
# ({"REG"} | _PLAYOFF_TYPES at :26,:68), so the ingest and the chain agree by construction.
TUNE_GAME_TYPES: tuple[str, ...] = ("REG", "WC", "DIV", "CON", "SB")
HOLD_GAME_TYPES: tuple[str, ...] = ("REG", "WC", "DIV", "CON", "SB")


# ---------------------------------------------------------------------------
# The odds population: label, juice source, and replay snapshot policy
# ---------------------------------------------------------------------------

# D31-12 BRANCH 1, settled by 31-RESEARCH.md Priority 1: the ~1855 stored rows labelled
# `consensus` are nflreadpy-derived, so 2025 carries the SAME label and the OUM-06 allowlist
# (backtest/ou_divergence._ALLOWED_SPORTSBOOKS) is NOT widened.
#
# The legacy mislabel is DOCUMENTED, not repaired: scripts/ingest_historical_odds.py:140 writes the
# literal "nflverse_closing" and has since v1.0 Phase 02 (commit b3f158d), yet no row in live
# silver carries it and no code in the repo writes the literal "consensus". Running that ingest
# unchanged would make the verdict run hard-fail at step one on assert_real_odds. The 2025 write
# uses THIS constant.
ODDS_SPORTSBOOK_LABEL: str = "consensus"

# D31-39, settled by MEASUREMENT in Plan 31-02 rather than by argument. `BRANCH_RULE` was stated as
# a module constant in scripts/audit_odds_preingest.py BEFORE the measurement ran; the measurement
# only evaluated it, so this branch is a consequence and not a post-hoc pick.
#
# Measured: all 23 partitioned files under data/silver/snapshot_ts=... are pyarrow per-write GUIDs
# naming ONE logical table (silver `odds_snapshot`) -- 7,829 accumulated rows holding 2,120
# distinct (game_id, sportsbook) pairs. Every file carries all four OddsSchema juice columns with
# real asymmetric prices (1,992 of 2,120 distinct pairs carry a non--110 value), and the rows agree
# with the flat table row-for-row on every shared value column with zero mismatches. All three
# BRANCH_RULE clauses hold, so the recommendation is PROMOTE and no network re-ingest is needed for
# the historical juice.
#
# 31-RESEARCH.md assumption A3 -- that the opaque filenames were ParquetManager table-name hashes
# -- is FALSE. ParquetManager.save delegates to pq.write_to_dataset with the SHARED data/silver
# root and no basename_template, so pyarrow names each file <guid>-<i>.parquet with a fresh guid
# per write call and never deletes prior files. That is why the store accumulates.
JUICE_SOURCE: str = (
    "PROMOTE the juice columns already present in the partitioned silver odds store. "
    "MEASURED (Plan 31-02, outputs/p31/odds_store_audit.json): 23 files / 8 partitions / "
    "7829 rows / 2120 distinct (game_id, sportsbook) pairs, all naming ONE logical table "
    "(silver odds_snapshot); all four juice columns present in every file with real asymmetric "
    "prices; row-for-row agreement with the flat table on every shared value column, zero "
    "mismatches. The pre-stated three-clause BRANCH_RULE evaluates to PROMOTE. "
    "THIS STORE IS NOT THE 2025 ODDS SOURCE -- the nflreadpy closing lines are (SPEC R2)."
)

# D31-37. Replay-row staleness is fixed at the SOURCE rather than exempted at the check.
REPLAY_SNAPSHOT_TS_POLICY: str = (
    "Every replay row's snapshot_ts is RE-DERIVED at ingest from that game's own preceding "
    "Friday 6 PM Eastern freeze (scripts/ingest_historical_odds.get_synthetic_snapshot_ts), so "
    "every replay row sits exactly AT its freeze and is therefore FRESH under the R6 rule that "
    "at-freeze is fresh. The stored column today is an object (string) column holding ONE fixed "
    "calendar date per season ({season}-09-19T18:00:00-04:00), and 2021-09-19 was a SUNDAY, so a "
    "naive per-game rule applied to the stored data would suppress everything past roughly week 2 "
    "of each season -- the rule was right and its input was invalid. Re-derivation also converts "
    "the column from str to a tz-aware datetime, which the freshness comparison needs regardless. "
    "The single non-consensus row holds a DIFFERENT string format "
    "(2025-09-29 18:44:09.707942+00:00, space-separated, UTC), so any comparison must PARSE and "
    "must never string-compare. REJECTED: exempting provenance == 'backtest_replay' from the "
    "freshness check, because that creates a SECOND freshness rule and one registry never two "
    "lists is the 29-06 lesson this project already paid for."
)


# ---------------------------------------------------------------------------
# The ATS residual contract (RE-DERIVED in Plan 31-02, NOT the research cache figure)
# ---------------------------------------------------------------------------

# The four re-derived per-season MEAN residuals, to 17 significant digits, from scoring the
# DEPLOYED artifact ats_20260605_220128 through backtest.diagnose.score_deployed_artifacts("ats")
# over canonical gold. The full (n, mean, sd, t, p) tuples live in tests/phase31_state.py; only the
# means are restated here, because the means are what the contract prose makes a claim about.
#
# THE NEGATIVE SEASON IS NAMED HERE ON PURPOSE. 2022 measures NEGATIVE. An unstated exception is
# the quiet form of overstating a claim, and a later reader who discovered it themselves would be
# right to conclude the direction had been overstated.
ATS_RESIDUAL_BY_SEASON_P31: Mapping[int, float] = {
    2021: 0.68147702779163399,
    2022: -0.03539119799896865,
    2023: 0.58219338768864415,
    2024: 1.1270376943443952,
}

# The POOLED mean over all 1139 tune-window games (sd 11.104272279463643, t 1.7912868714554835,
# p 0.073512927737913666).
ATS_RESIDUAL_POOLED_P31: float = 0.58937727047262922

ATS_RESIDUAL_CONTRACT: str = (
    "residual = actual home margin - predicted home spread; "
    "the measured tune-window POOLED bias is POSITIVE (+0.58937727047262922 over n=1139), i.e. "
    "the model UNDER-predicts the home margin on the window as a whole; "
    "corrected = predicted home spread + prior-season bias pushes the predicted margin UP and "
    "therefore RAISES the cover probability -- the OPPOSITE sign to the O/U contract, where "
    "adding the bias pulls the total DOWN and LOWERS P(over). "
    "THE DIRECTION CLAIM IS POOLED ONLY. One tune season, 2022, has a slightly NEGATIVE mean "
    "(-0.03539119799896865). That does NOT weaken the chain, because the correction the chain "
    "actually applies is the PER-SEASON WALK-FORWARD estimate from estimate_prior_season_bias, "
    "which reads only strictly-prior seasons and therefore never assumes a constant sign; the "
    "pooled figure supports ONLY the magnitude argument stated below. "
    "MAGNITUDE: Phi(pooled_mean / pooled_sd) - 0.5 = 0.021164571224560169 against the flat -110 "
    "breakeven edge of 0.023809523809523836, so the bias is 0.88891199143152611 of the entire "
    "house edge -- which is why omitting the correction is not a neutral simplification. "
    "PROVENANCE: RE-DERIVED from the deployed artifact ats_20260605_220128 in Plan 31-02. The "
    "31-RESEARCH.md cache-derived figures (pooled +0.714049, 2022 +0.118713, from "
    "data/web_cache.duckdb) are STALE relative to the Phase-25 re-fit and MUST NOT be used."
)

# The ATS chain CARRIES a prior-season bias correction, on the magnitude argument above. This is
# the answer to the Claude's-discretion question D31 left open ("whether ATS carries a prior-season
# bias correction analogous to O/U's should follow the same measured logic and be pre-registered
# either way"). It is pre-registered as YES, before any 2025 number exists.
ATS_BIAS_CARRIED: bool = True


# ---------------------------------------------------------------------------
# Per-target chain policy and eligibility (D31-05, D31-06, D31-07)
# ---------------------------------------------------------------------------

WP_CHAIN_POLICY: str = (
    "D31-07. DEFAULT: the deployed WP artifact's own isotonic-calibrated P(home) is used as-is; "
    "a reliability and Brier gate runs on the TUNE split only; the real two-sided moneyline "
    "devig() is reused verbatim from the existing Phase-27 chain; per-bet EV follows. "
    "REGISTERED FALLBACK: a prior-season PROBABILITY-SCALE bias correction fires ONLY when that "
    "gate FAILS, and its trigger is written into the trial registry's fallback_trigger field so "
    "it can never activate silently. "
    "This INVERTS D27-07's structure (there, bias-subtraction is the default and isotonic is the "
    "fallback) because the WP model is already calibrated and the O/U model is not. "
    "WP has NO residual to take a standard deviation of, so 'the same apparatus' for WP means the "
    "same CONTRACT, not the same STEPS. Forcing O/U's shape onto WP by inventing a logit-space "
    "residual SD was REJECTED: it manufactures a fitted parameter a calibrated classifier does not "
    "need, on the target with the worst measured CLV. Re-fitting calibration on the tune split was "
    "REJECTED: the bet list would then be priced off a model nobody runs."
)

OU_ELIGIBILITY_FROZEN: str = (
    "D31-06. O/U's eligibility gate is CONSUMED UNCHANGED from "
    "backtest.ou_divergence.HIGH_TOTAL_BOUNDARY_PREHOLD (the leakage-clean upper-tertile boundary "
    "re-derived on pre-hold 2018-2022 only, ~48.0) together with the Phase-26 UNION rule "
    "(UNDER picks OR high-total games). Because the constant and the rule are read rather than "
    "re-derived, O/U's eligibility pre-registration is provable BY GIT ANCESTRY ALONE. "
    "Re-deriving the boundary on 2018-2024 was REJECTED: it would re-derive a LOCKED-1 constant at "
    "exactly the moment it could change the answer, and would contradict D30-13 three months on. "
    "The pre-2025 reproduction is published BESIDE the 2025 result as CONTEXT ONLY, worded so it "
    "can never read as a second look at burned data."
)

WP_ATS_ELIGIBILITY: str = (
    "D31-05. WP and ATS get NO eligibility gate: the calibrated chain runs on every candidate and "
    "the EV floor t alone decides. O/U's under-OR-high-total UNION was EARNED by Phase 26 -- an "
    "entire phase of pre-registered sub-population sweeping with BH correction -- while WP "
    "(pooled CLV -0.0380, t -15.52) and ATS (-0.0015, p 0.990) have had no such diagnosis. "
    "A zero-bet chain is an explicitly defined PASS under SPEC R1, not a failure. "
    "REJECTED: a pre-registered sub-population search per target, which would discover a rule on "
    "contaminated and partly-burned 2021-2024 data and then spend the single clean split "
    "validating it -- aiming the most valuable asset in the project at the weakest question."
)


# ---------------------------------------------------------------------------
# EV tier bands (D31-24, SPEC R7) -- UNCHANGED from Plan 31-01
# ---------------------------------------------------------------------------

EV_TIER_LOW = "low"
EV_TIER_MEDIUM = "medium"
EV_TIER_HIGH = "high"

# The lower bound of the LOW band is the pre-registered EV floor ``t``, which is a per-target
# tuned scalar rather than a constant, so it is carried as ``None`` here and resolved from the
# ``ev_floor_t`` argument at call time. Recording it as a literal would freeze a number this
# module does not own.
_EV_FLOOR_SENTINEL: float | None = None

# Ordered half-open [lo, hi) bands. LOWER bound INCLUSIVE, UPPER bound EXCLUSIVE, so a value
# landing exactly on a boundary falls in the HIGHER band: 0.03 is medium (not low) and 0.05 is
# high (not medium). This is the SPEC R7 adjacency edge, fixed here rather than in the renderer.
EV_TIER_BANDS: tuple[tuple[str, float | None, float], ...] = (
    (EV_TIER_LOW, _EV_FLOOR_SENTINEL, 0.03),
    (EV_TIER_MEDIUM, 0.03, 0.05),
    (EV_TIER_HIGH, 0.05, math.inf),
)

EV_TIER_LABELS: tuple[str, ...] = tuple(label for label, _lo, _hi in EV_TIER_BANDS)


def assign_ev_tier(per_bet_ev: float, ev_floor_t: float) -> str:
    """Return the pre-registered EV band label for *per_bet_ev* under floor *ev_floor_t*.

    The bands are the half-open [lo, hi) triples in :data:`EV_TIER_BANDS`, with the LOW band's
    lower bound resolved to ``ev_floor_t``. A boundary value falls in the HIGHER band.

    A non-finite EV does NOT receive a label. Under the SPEC R7 suppression taxonomy such a row
    is rejected upstream with reason ``ev_not_finite`` and never reaches the live list, so
    returning a band here would manufacture a tier for a bet that is not a bet. The same applies
    to an EV below the floor: it was not admitted, so it has no band.

    Args:
        per_bet_ev: The calibrated per-bet expected value, in units of stake.
        ev_floor_t: The pre-registered per-target EV floor ``t`` (the LOW band's lower bound).

    Returns:
        One of ``"low"`` / ``"medium"`` / ``"high"``.

    Raises:
        ValueError: if *per_bet_ev* or *ev_floor_t* is non-finite, or if *per_bet_ev* is below
            *ev_floor_t* (an unadmitted row has no band).
    """
    if not math.isfinite(per_bet_ev):
        msg = (
            f"assign_ev_tier: per_bet_ev={per_bet_ev!r} is not finite; a non-finite EV is "
            "suppressed with reason 'ev_not_finite' (SPEC R7) and is never tiered."
        )
        raise ValueError(msg)
    if not math.isfinite(ev_floor_t):
        msg = (
            f"assign_ev_tier: ev_floor_t={ev_floor_t!r} is not finite; the EV floor is a "
            "pre-registered finite scalar and must resolve before any band is assigned."
        )
        raise ValueError(msg)
    if per_bet_ev < ev_floor_t:
        msg = (
            f"assign_ev_tier: per_bet_ev={per_bet_ev!r} is below the EV floor {ev_floor_t!r}; "
            "such a candidate is rejected with reason 'ev_below_floor' and has no band."
        )
        raise ValueError(msg)

    for label, lo, hi in EV_TIER_BANDS:
        lower = ev_floor_t if lo is _EV_FLOOR_SENTINEL else lo
        assert lower is not None  # narrowed by the sentinel resolution above
        if lower <= per_bet_ev < hi:
            return label

    # Unreachable by construction: the bands are contiguous from ev_floor_t to +inf and the
    # below-floor case is rejected above. Raise rather than return a label if it ever happens.
    msg = (
        f"assign_ev_tier: per_bet_ev={per_bet_ev!r} fell through every band under floor "
        f"{ev_floor_t!r}; EV_TIER_BANDS is no longer contiguous."
    )
    raise ValueError(msg)


# ---------------------------------------------------------------------------
# Robustness cuts and the multiplicity family (D31-33, D31-34, D31-38)
# ---------------------------------------------------------------------------

# BOTH Phase-27 cut names are kept, verbatim and in order, for git-ancestry continuity with
# backtest/ou_monetization.ROBUSTNESS_CUTS. Restated rather than imported because their MEANING
# changed under D31-38: with playoffs in the population these are no longer the vacuous pair
# Phase 27 disclosed, and the constant that changed meaning should be the one this phase declares.
ROBUSTNESS_CUTS_P31: tuple[str, ...] = ("regular_season_only", "playoffs_excluded")

# The two cuts compute the BYTE-IDENTICAL frame -- backtest/ou_monetization.py:690-710 assigns
# `_flat_roi_from_records(reg_season)` to both keys from the same `week <= 18` filter. One
# statistic reported twice is not two hypotheses, so the pair contributes ONE entry to the BH
# family, not two. Correcting for a duplicate would make a real result harder to detect for
# ceremonial rather than statistical reasons.
ROBUSTNESS_CUT_BH_COUNT: int = 1

# The multiplicity family, ENUMERATED IN ADVANCE (D31-33/34) so the denominator is fixed before
# any 2025 number exists rather than argued afterwards.
BH_FAMILY_SPEC: Mapping[str, object] = {
    "correction_method": "benjamini-hochberg",
    "scope": "hold-side inferences only, pooled across all three targets in ONE registry",
    "included": (
        "the PRIMARY 2025 profitability verdict for each of wp, ats and ou (3 entries)",
        "the robustness cut reported pass/fail for each target, counted ONCE per target "
        "because both cut names compute the byte-identical frame (3 entries)",
        "a per-target CALIBRATION FALLBACK verdict, but ONLY IF that target's fallback "
        "actually fired (0 to 3 entries, recorded in the registry's fallback_trigger field)",
    ),
    "excluded": (
        "the TUNE-SIDE EV-floor grid sweep: it is RECORDED in the registry for transparency "
        "but never enters the family, because the sweep runs entirely on the tune split and "
        "applies ONE selected t to the hold -- those grid cells never touched 2025",
        "every entry whose entry_kind is 'control': the shifted-edge counterfactual pass "
        "(D31-08) has no p-value to correct, so counting it would statistically penalise the "
        "verdict for running a code-liveness check",
    ),
    "denominator_no_fallback": 6,
    "denominator_with_fallbacks": "7, 8 or 9 -- one additional entry per target whose "
    "calibration fallback fired",
    "registry_rows_may_exceed_denominator": (
        "YES, visibly and by design. Every read of 2025 is on the record; the denominator "
        "counts only actual inferences."
    ),
}

# The trial-registry schema: the Phase-27 field tuple EXTENDED with the two fields D31-33/34
# require. Restated rather than imported because it is a superset, and a superset built by
# concatenation would silently inherit a future Phase-27 edit into a frozen Phase-31 rule.
TRIAL_ENTRY_KIND_INFERENCE: str = "inference"
TRIAL_ENTRY_KIND_CONTROL: str = "control"

TRIAL_REGISTRY_FIELDS_P31: tuple[str, ...] = (
    "threshold",
    "subpopulation_rule",
    "calibration_method",
    "sd_source",
    "devig_method",
    "sizing_policy",
    "sample_window",
    "robustness_cut",
    "raw_p",
    "adjusted_p",
    "roi",
    "ci",
    "bet_count",
    # Phase-31 additions.
    "entry_kind",  # TRIAL_ENTRY_KIND_INFERENCE or TRIAL_ENTRY_KIND_CONTROL
    "fallback_trigger",  # what made a registered fallback fire, or None
)

# The disclosure that must travel with any comparison between this phase's interval and Phase 27's.
BOOTSTRAP_DISCLOSURE_P31: str = (
    "BOOTSTRAP_B, BOOTSTRAP_SEED and BOOTSTRAP_CI_TYPE are CONSUMED UNCHANGED from "
    "backtest.ou_monetization (imported, not restated). But a ONE-SEASON hold gives at most 22 "
    "season-week blocks against Phase 27's roughly 36 (its published run reported n_blocks 17 "
    "over two seasons of selected bets), so the block-by-week interval published here is WIDER "
    "and the two intervals are NOT commensurable. This is a property of the design, disclosed "
    "before the run, not a limitation discovered after it."
)


# ---------------------------------------------------------------------------
# The ROI HYPOTHESIS TEST, defined and frozen HERE (REVIEW-ROI)
#
# No such test exists in this repository today, and a verdict cannot rest on a statistic invented
# after the numbers. backtest/diagnose.py's clv_significance (~249) tests CLV, not ROI -- its own
# docstring says it mirrors the ttest_1samp(clv, 0) idiom. backtest/ou_monetization.py's
# _block_by_week_bootstrap_ci (~624-686) returns a PERCENTILE INTERVAL with NO p-value. So the ROI
# p-value is registered in full below, before any 2025 read.
# ---------------------------------------------------------------------------

ROI_P_VALUE_METHOD: str = "one-sided achieved significance level from the NULL-RECENTRED block-by-week bootstrap"

# p = (1 + #{recentred replicates >= observed ROI}) / (BOOTSTRAP_B + 1).
ROI_MIN_ATTAINABLE_P: float = 1.0 / (BOOTSTRAP_B + 1)

ROI_SIGNIFICANCE_SPEC: Mapping[str, object] = {
    "null": (
        "H0: the POPULATION flat-stake ROI over the hold is LESS THAN OR EQUAL TO ZERO."
    ),
    "statistic": (
        "the OBSERVED flat-stake ROI over the hold bets, computed by the same "
        "sum(payout_flat) / sum(flat_stake) ratio backtest.ou_monetization "
        "_flat_roi_from_records already uses. No new estimator is introduced."
    ),
    "reference_distribution": (
        "the SAME block-by-week resamples the confidence interval already uses: the "
        "(season, week) pair is the BLOCK, blocks are resampled WITH REPLACEMENT from the HOLD "
        "bets only, at the frozen BOOTSTRAP_B and BOOTSTRAP_SEED -- then RECENTRED AT THE NULL "
        "by subtracting the observed ROI from each replicate. Resampling whole weeks preserves "
        "within-week correlation; resampling individual bets would understate the spread."
    ),
    "one_sided_rule": (
        "p = (1 + the count of RECENTRED replicates greater than or equal to the observed ROI) "
        "/ (BOOTSTRAP_B + 1)."
    ),
    "finite_sample_rule": (
        "the plus-one in BOTH numerator and denominator makes p STRICTLY POSITIVE and never "
        "zero, so a run in which no recentred replicate reaches the observed ROI reports the "
        "smallest attainable p rather than a p of 0.0 that would overstate the evidence."
    ),
    "min_attainable_p": ROI_MIN_ATTAINABLE_P,
    "min_attainable_p_consequence": (
        "IF the minimum attainable p EXCEEDS alpha, the test CANNOT reach significance at any "
        "observed ROI, and a positive return MUST report the INCONCLUSIVE_CLEAN token with that "
        "stated reason -- NEVER PROFITABLE_CLEAN. At the frozen BOOTSTRAP_B = 2000 the minimum "
        "is 1/2001 = 0.0004997501249375312, which is below alpha = 0.05, so the test CAN reach "
        "significance on this configuration. The rule is registered anyway, because a rule that "
        "only exists once it binds is a rule chosen after the fact."
    ),
    "resolution_disclosure": (
        "a ONE-SEASON hold gives at most 22 season-week blocks, so the reference distribution is "
        "COARSE: the attainable p-values are a discrete ladder in steps of 1/(BOOTSTRAP_B + 1) "
        "over a resampling space of only 22 distinct blocks. That is a limitation of the design, "
        "stated before the run rather than discovered after it."
    ),
    "alpha": ALPHA,
    "bootstrap_b": BOOTSTRAP_B,
    "bootstrap_seed": BOOTSTRAP_SEED,
}

# The CLV p-value is REPORTED beside the verdict and is FORBIDDEN from entering the multiplicity
# family or from driving any verdict token. Reusing it would silently convert "significant CLV"
# into "profitable" -- which is precisely the D25-14 and D26-09 trap this project has already been
# caught by once, and precisely what this phase exists to prevent. Plan 31-12 adds the structural
# assertion that the verdict-assignment path reads no CLV p-value at all.
CLV_P_VALUE_IS_REPORT_ONLY: bool = True


# ---------------------------------------------------------------------------
# The DISJOINT rehearsal proxy split (REVIEW-REHEARSAL)
# ---------------------------------------------------------------------------

# The split the Plan 31-14 plumbing rehearsal and the Plan 31-12 three-target integration run use.
# It is NOT the pre-registered rule and is NEVER consumable by the armed run; its result is
# DISCARDED and never reported (D31-16 checkpoint 3).
#
# WHY IT MUST BE DISJOINT ON BOTH SIDES. The frozen TUNE_SEASONS_P31 is 2021-2024, so overriding
# ONLY the hold to 2024 would make a TUNE season the HOLD season. The Phase-31 fence -- which
# mirrors the four checks in backtest/ou_monetization._assert_fit_window -- rejects a hold season
# consumed by the SD fit or the threshold tuning. The rehearsal would therefore either FAIL, or
# would only pass against a fence weakened enough to permit leakage, in a phase whose entire value
# is temporal honesty. So the rehearsal narrows the TUNE side too.
REHEARSAL_PROXY_SPLIT: Mapping[str, object] = {
    "tune_seasons": (2021, 2022, 2023),
    "hold_seasons": (2024,),
    "is_the_preregistered_rule": False,
    "consumable_by_the_armed_run": False,
    "result_disposition": "DISCARDED -- never reported, never published, never compared",
    "reason_disjoint": (
        "overriding only the hold to 2024 would put a TUNE season in the HOLD and either trip "
        "the fit-window fence or require weakening it"
    ),
}


# ---------------------------------------------------------------------------
# The durable exclusive one-shot run ledger (REVIEW-ONESHOT)
# ---------------------------------------------------------------------------

# Refusing only when the verdict artifact already exists is NOT crash-safe: the runner READS AND
# MEASURES 2025 BEFORE it writes the artifact, so a crash inside that window leaves no artifact and
# the next invocation is free to spend the single-use split again. The ledger closes that window by
# being created with an EXCLUSIVE file creation BEFORE the first 2025 read.
RUN_LEDGER_PATH: str = "config/profitability_2025_run_ledger.toml"

RUN_LEDGER_STATES: tuple[str, ...] = ("armed", "started", "completed", "failed")

RUN_LEDGER_TRANSITION_RULE: str = (
    "The ledger is created by an EXCLUSIVE file creation (open 'x') BEFORE the first 2025 read; "
    "if it already exists the runner REFUSES, and there is NO force flag. "
    "armed -> started is written before the first 2025 read. started -> completed is written "
    "after the verdict artifact lands. started -> failed is written on an error path. "
    "A ledger in state 'started' or 'failed' is NOT automatically rerunnable: a new attempt "
    "requires an OWNER RULING recorded in the ledger before the state may return to 'armed'. "
    "The runner additionally hard-refuses to overwrite an existing verdict artifact. "
    "Silently re-arming a crashed attempt would spend the single clean split twice while leaving "
    "a record that says it ran once."
)


# ---------------------------------------------------------------------------
# Verdict vocabulary (D31-36, SPEC R3/R11) -- tokens UNCHANGED from Plan 31-01
# ---------------------------------------------------------------------------

# The CLOSED set of verdict tokens the 2025 profitability run may emit. A target that selected
# zero bets reports UNDISCHARGEABLE_NO_BETS -- never "ROI 0" -- and a target whose EV chain did
# not resolve reports UNDISCHARGEABLE_NO_CHAIN. Both are first-class outcomes, not failures.
VERDICT_TOKENS: tuple[str, ...] = (
    "PROFITABLE_CLEAN",
    "UNPROFITABLE_CLEAN",
    "INCONCLUSIVE_CLEAN",
    "UNDISCHARGEABLE_NO_BETS",
    "UNDISCHARGEABLE_NO_CHAIN",
)

# The token a zero-bet target reports. Named separately because the readout template and the
# tracker both need to refer to it as a RESULT, and a string literal repeated across three modules
# is the second-list failure this project keeps paying for.
UNDISCHARGEABLE_NO_BETS: str = "UNDISCHARGEABLE_NO_BETS"

# Each token bound to the condition that produces it. The binding is here rather than in the
# runner so a reader can check afterwards that the mapping from numbers to words was fixed before
# the numbers existed.
VERDICT_TOKEN_MEANINGS: Mapping[str, str] = {
    "PROFITABLE_CLEAN": (
        "at least one bet was selected on the clean 2025 hold, the observed flat-stake ROI is "
        "strictly positive, and the pre-registered one-sided ROI p-value is BELOW alpha after "
        "the BH correction over the enumerated family."
    ),
    "UNPROFITABLE_CLEAN": (
        "at least one bet was selected and the observed flat-stake ROI is NEGATIVE OR ZERO. "
        "This is a measured result, not a failure to measure."
    ),
    "INCONCLUSIVE_CLEAN": (
        "at least one bet was selected and the observed flat-stake ROI is strictly positive, but "
        "the pre-registered one-sided ROI p-value does NOT clear alpha after the BH correction -- "
        "including the case where the minimum attainable p exceeds alpha, which is reported with "
        "that stated reason. A positive return that cannot be distinguished from zero is "
        "INCONCLUSIVE and is never called PROFITABLE."
    ),
    "UNDISCHARGEABLE_NO_BETS": (
        "the chain resolved and ran end to end on the 2025 frame, its positive control passed, "
        "and it selected ZERO bets. This is a RESULT: it fills the same readout template slots "
        "as any other verdict, and it is an explicitly defined PASS under SPEC R1."
    ),
    "UNDISCHARGEABLE_NO_CHAIN": (
        "the target's EV chain did not resolve on the 2025 frame at all (a required input was "
        "absent, so no candidate could be priced). Distinct from NO_BETS: there, the chain ran "
        "and declined; here, the chain could not run."
    ),
}


# ---------------------------------------------------------------------------
# The R11 readout guards (D31-35, D31-36, REVIEW-READOUT)
# ---------------------------------------------------------------------------

# The fields EVERY per-target statement must carry, so omission is structurally impossible rather
# than something to remember and the drift guard can assert field presence directly.
READOUT_REQUIRED_FIELDS: tuple[str, ...] = (
    "deployed_artifact",
    "promoted_or_retained",
    "promoted_or_retained_when",
    "absolute_clv_verdict",
    "verdict_token_2025",
    "bets_selected",
    "evidence_pointer",
)

# The hype vocabulary the readout may not use. SCOPED, not blanket: this is a deploy-adjacent
# honesty phase and the readout MUST be able to say that a model is deployed and that its
# closing-line value is negative. Matched by WORD BOUNDARY, never by naive substring -- a substring
# check for "proven" false-positives on the provenance column name and one for "validated"
# false-positives on a re-validation phrase, and a guard that reddens on an innocent word gets
# ignored (the D30-DEFER-05 lesson).
READOUT_FORBIDDEN_WORDS: tuple[str, ...] = (
    "proven",
    "validated",
    "guaranteed",
    "riskless",
    "infallible",
    "surefire",
)

# Forbidden VALUES, distinct from forbidden words. A number can be a lie without any adjective
# attached to it.
READOUT_FORBIDDEN_FIGURES: Mapping[str, str] = {
    "45.81043733209909": (
        "BacktestResults.headline_clv for O/U. It is the mean of the probability_clv column, "
        "which for O/U is not a line-CLV at all, so publishing it as a CLV would be publishing a "
        "roughly 40x overstatement of a quantity that does not exist (SPEC R11 prohibition, "
        "GATED-REFIT-READOUT.md section 7c). The readout states O/U's true line_clv mean instead."
    ),
}

# The EXPLICIT ALLOWLIST of prior-document numbers the closing readout may restate, each with the
# authoritative source it must be printed beside (REVIEW-READOUT).
#
# This REPLACES the blanket "restate no prior-document number" rule, which was UNSATISFIABLE: the
# same requirement separately demands the absolute CLV values, the previous ATS Kelly figure and
# the winner target's zero-staked ratio. An unsatisfiable guard gets weakened during execution
# until it means nothing, which is how a guard becomes decoration. Any number in the readout that
# is neither a 2025 figure from the verdict artifact nor an entry here is a VIOLATION, and a
# POINTER to the document that holds it is the required alternative.
READOUT_PERMITTED_FIGURES: Mapping[str, str] = {
    "per_target_absolute_pooled_clv_with_t_and_p": "config/gate.toml",
    "ats_kelly_return_previous_and_current": (
        ".planning/phases/31-ship-the-ev-bet-list-profitability-readout/31-10-SUMMARY.md"
    ),
    "winner_target_zero_staked_bet_count_and_ratio": (
        ".planning/phases/31-ship-the-ev-bet-list-profitability-readout/31-10-SUMMARY.md"
    ),
    "standing_quarantined_reproduction_count": "STATE-OF-SYSTEM.md",
    "every_2025_figure": "config/profitability_2025_verdict.toml",
}


# ---------------------------------------------------------------------------
# The two files that ARE the pre-registration
# ---------------------------------------------------------------------------

# Named here so every guard resolves them from ONE place: the ancestry test's
# `git log -1 --format=%H -- <paths>`, the content-hash comparison against
# tests/phase31_state.PRE_REGISTRATION_FILE_SHA256, and the Plan 31-14 assertion that this
# commit strictly precedes the 2025 measurement commit. Repo-root-relative POSIX paths.
PREREGISTRATION_PATHS: tuple[str, ...] = (
    "PROFITABILITY-PREREGISTRATION.md",
    "backtest/ev_chain_constants.py",
)
