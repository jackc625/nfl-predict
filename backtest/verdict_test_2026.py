"""The pre-registered test that judges the 2026 forward profit verdict (Phase 38; CLEAN-04, D-11).

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/verdict_scope_2026.py`` and
``backtest/refit_preregistration_2026.py``. This module IS the pre-registration of HOW the 2026
forward profit verdict is judged, not a description of one: the test, the multiplicity family, the
pass line, the verdict words and their 2026 meanings, which outcomes enter, and when the one
binding run may happen. ``backtest/verdict_scope_2026.py`` already fixed WHICH bets count; nothing
fixed how they are judged until this record. It is committed ALONE, together with
``SEASON-VERDICT-PREREGISTRATION.md`` at the repository root, and pushed to GitHub's ``master``
before week 8's first decision run, Wed 2026-10-28 17:00 America/New_York
(``PREREGISTRATION_DEADLINE_UTC``).

IT IS ONE PRE-REGISTRATION IN TWO FILES. The root markdown document is the human-readable half and
states every rule below in plain words; this module is the half the readout reads. Both were landed
in ONE commit that touches nothing else.

Stated plainly because it is easy to forget by February: EDITING THIS FILE AFTER THE COMMIT DOES
NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. A value that is wrong here is wrong for the whole 2026
verdict, and the only honest repair is a NEW, visibly later commit whose message says it
INVALIDATES this one by naming its commit sha. Never an edit in place, never a quiet supersession.

THIS MODULE DOES NOT RECORD ITS OWN CONTENT HASH OR COMMIT. A file that must contain its own
whole-file hash has no fixed point. The witness lives OUTSIDE it: ``tests/phase38_state.py``
records the commit, the newline-normalized sha256 of both files and the commit time in a LATER
commit (Plan 38-03), and the Phase-38 ancestry tests recompute and compare.

Constants only: NO I/O, NO functions and NO logic. UNLIKE the Phase 37 record, which copies every
value as a literal, this record IMPORTS alpha, the bootstrap settings and the verdict words BY
IDENTITY from ``backtest.ev_chain_constants`` (D-12): a second alpha would be a second answer, and
a retyped token would be a second vocabulary. ``tests/unit/test_phase38_preregistration_ancestry.py``
(Plan 38-03) pins the imported values, so an edit upstream cannot move them silently.

The five verdict words are named in prose only here and in the markdown half: PROFITABLE_CLEAN,
UNPROFITABLE_CLEAN, INCONCLUSIVE_CLEAN, UNDISCHARGEABLE_NO_BETS and UNDISCHARGEABLE_NO_CHAIN. Every
string constant below refers to them through the imported names or in plain words ("the profitable
verdict"), never by spelling a token, and each token gets its own 2026 meaning: CLEAN means
measured forward under a rule frozen before the rest of the season's results existed. "Not played"
is an inference read from the schedule on each run (``NOT_PLAYED_RULE``).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from backtest.ev_chain_constants import (
    ALPHA,
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    ROI_MIN_ATTAINABLE_P,
    ROI_P_VALUE_METHOD,
    UNDISCHARGEABLE_NO_BETS,
    VERDICT_TOKENS,
)

__all__ = [
    "ALPHA",
    "ALPHA_COMPARISON",
    "BH_FAMILY_RULE",
    "BH_FAMILY_SIZE",
    "BH_FUNCTION",
    "BINDING_RECORD_PATH",
    "BINDING_REFUSAL_RULE",
    "BLOCK_DESIGN_NOTE",
    "BOOTSTRAP_B",
    "BOOTSTRAP_SEED",
    "CLV_ENTERS_VERDICT",
    "DECISIONS_USE_UNROUNDED_VALUES",
    "MEASUREMENT_MAPPING_2026",
    "NOT_PLAYED_RULE",
    "ONE_SHOT_RULE",
    "OUTCOME_SOURCE",
    "PER_GENERATION_PROFIT_RULE",
    "PREREGISTRATION_DEADLINE_UTC",
    "PREVIEW_RULE",
    "PUSH_RULE",
    "P_VALUE_FUNCTION",
    "P_VALUE_RULE",
    "RFIT_08_READING",
    "ROI_MIN_ATTAINABLE_P",
    "ROI_P_VALUE_METHOD",
    "ROI_RULE",
    "STATISTIC",
    "TEST_UNIT",
    "TIE_DISCLOSURE",
    "TIE_RULE",
    "UNDISCHARGEABLE_NO_BETS",
    "UNGRADED_SCOPE_RULE",
    "VERDICT_RULE_FUNCTION",
    "VERDICT_SCOPE_MODULE",
    "VERDICT_TARGETS",
    "VERDICT_TOKENS",
    "VERDICT_TOKEN_MEANINGS_2026",
    "VOID_RULE",
    "WEEKS_GRADED_AT_RECORD_COMMIT",
]


# ---------------------------------------------------------------------------
# Scope and the unit of the test (D-01)
# ---------------------------------------------------------------------------

# The committed declaration of WHICH bets count (season 2026, weeks 5-22 inclusive, the live arm,
# playoffs and neutral sites in, in_force_correction, fill-v1). Named, never restated; it equals
# forward_ledger.declarations.VERDICT_SCOPE_MODULE (asserted in Plan 38-03, not imported here).
VERDICT_SCOPE_MODULE: str = "backtest.verdict_scope_2026"

# The three targets, each judged by its own single test, in the repository's canonical order.
VERDICT_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# D-01: one test per target over the whole live season.
TEST_UNIT: str = (
    "one flat-ROI test per target over every counted live-arm bet of the verdict scope, read by "
    "forward_ledger.verdict.verdict_rows under backtest/verdict_scope_2026.py, whichever re-fit "
    "generation decided it."
)

# D-01: a design statement, not a measured fact (research pitfall 14).
BLOCK_DESIGN_NOTE: str = (
    "re-fits install on Tuesday cutoffs, so by design no resampled (season, week) block mixes two "
    "generations. This is a statement of the cadence design, not a measured fact."
)

# D-01: per-generation profit carries no test.
PER_GENERATION_PROFIT_RULE: str = (
    "per-generation profit is shown only as plain numbers with no test: the ROI_SECONDARY lines "
    "of the re-fit comparison command."
)


# ---------------------------------------------------------------------------
# The statistic and its p-value (D-12)
# ---------------------------------------------------------------------------

# The 2025 statistic, reused unchanged.
STATISTIC: str = (
    "flat-stake ROI, sum(payout_flat) / sum(flat_stake) over the target's counted bets."
)

# The one function that computes the statistic and its p-value.
P_VALUE_FUNCTION: str = "backtest.roi_significance.roi_ci_and_p"

# D-12: the 2025 one-sided null-recentred block-by-week bootstrap, at the imported B and seed.
P_VALUE_RULE: str = (
    "the one-sided null-recentred block-by-week bootstrap: whole (season, week) blocks of the "
    "counted bets are resampled with replacement at BOOTSTRAP_B replicates and BOOTSTRAP_SEED, "
    "each replicate is recentred by subtracting the observed ROI, and p = (1 + the count of "
    "recentred replicates greater than or equal to the observed ROI) / (BOOTSTRAP_B + 1). Bets "
    "spanning fewer than two blocks give no p-value, with roi_ci_and_p's stated reason."
)


# ---------------------------------------------------------------------------
# The multiplicity family (D-02)
# ---------------------------------------------------------------------------

# The one function that applies the correction, with m passed explicitly.
BH_FUNCTION: str = "backtest.profitability_2025.benjamini_hochberg_adjusted"

# D-02: exactly the three primary tests.
BH_FAMILY_SIZE: int = 3

# D-02: what enters the family, and why it is not 2025's six.
BH_FAMILY_RULE: str = (
    "the Benjamini-Hochberg family is exactly the primary flat-ROI test of each of "
    "VERDICT_TARGETS; m = BH_FAMILY_SIZE is passed explicitly and nothing else enters. It "
    "differs from 2025's six-entry family (BH_FAMILY_SPEC) for two reasons: the 2025 "
    "regular-season re-check existed for 2025's own playoff ruling (D31-38), while 2026's "
    "playoffs already count under D40-08; and the 2025 calibration-fallback entry has no 2026 "
    "analogue, because the WP fallback is built into each generation's bet rule (Phase 37 "
    "D-29). The re-fit accuracy comparison keeps its own separate family, frozen in e25884d."
)


# ---------------------------------------------------------------------------
# The pass line (D-04)
# ---------------------------------------------------------------------------

# D-04: the return side of the line.
ROI_RULE: str = (
    "the profitable verdict needs at least one bet and a flat ROI strictly above 0; a flat ROI "
    "of 0 or less is the unprofitable verdict."
)

# D-04 / SPEC edge R6 boundary: the exact comparison against alpha.
ALPHA_COMPARISON: str = (
    "strictly less than: the Benjamini-Hochberg-adjusted p must be strictly below ALPHA, and a "
    "value equal to ALPHA is not significant. With BOOTSTRAP_B = 2000 the p-value moves in steps "
    "of 1/2001, so exactly 0.05 cannot occur; the rule is stated anyway. A positive ROI whose "
    "adjusted p is ALPHA or more, or is absent, or whose minimum attainable p exceeds ALPHA, is "
    "the inconclusive verdict."
)

# D-04 / SPEC AC 25: decisions read unrounded values; rounding is display only.
DECISIONS_USE_UNROUNDED_VALUES: bool = True

# The one function that maps a measurement to a verdict word, arms in its pre-registered order.
VERDICT_RULE_FUNCTION: str = "backtest.profitability_2025.assign_verdict_token"


# ---------------------------------------------------------------------------
# The verdict words and their 2026 meanings (D-03, D-13)
# ---------------------------------------------------------------------------

# D-03: the meanings follow VERDICT_TOKENS order exactly -- profitable, unprofitable,
# inconclusive, no bets, no chain. The tokens themselves are imported, never spelled here.
VERDICT_TOKEN_MEANINGS_2026: dict[str, str] = dict(
    zip(
        VERDICT_TOKENS,
        (
            (
                "at least one counted bet, the observed flat ROI over the whole live 2026 "
                "season is strictly above 0, and the one-sided ROI p-value after the "
                "Benjamini-Hochberg correction over the three-test family is strictly below "
                "alpha. CLEAN here means measured forward under a rule frozen before the rest "
                "of the season's results existed; the weeks already graded when it was frozen "
                "(WEEKS_GRADED_AT_RECORD_COMMIT) count under the same rule and are disclosed, "
                "never excluded."
            ),
            (
                "at least one counted bet and an observed flat ROI of 0 or less. This is a "
                "measured result, not a failure to measure."
            ),
            (
                "at least one counted bet and a strictly positive flat ROI, but the adjusted "
                "p-value is alpha or more, could not be computed, or the minimum attainable p "
                "exceeds alpha. A positive return that cannot be told apart from zero is never "
                "called profitable."
            ),
            (
                "the target has counted live-arm rows (its candidates were priced, status "
                "suppressed) but none of them is a bet (D-13)."
            ),
            (
                "the target has no counted live-arm row at all: it was never priced inside the "
                "verdict scope (D-13)."
            ),
        ),
        strict=True,
    )
)

# D-13: every TargetMeasurement field and its 2026 source. The readout prints the token and its
# 2026 meaning above, never assign_verdict_token's 2025-worded reason string.
MEASUREMENT_MAPPING_2026: dict[str, str] = {
    "target": "the target code, one of VERDICT_TARGETS.",
    "chain_resolved": "True when the target has at least one counted live-arm row.",
    "control_passed": (
        "always True in 2026: no counterfactual control exists, so the zero-bet split is "
        "decided by chain_resolved alone."
    ),
    "control_bet_count": "0.",
    "bet_count": (
        "the target's counted rows with status live, void bets excluded (D-20), pushes included."
    ),
    "flat_roi": "roi_ci_and_p point_estimate over those bets.",
    "adjusted_p": (
        "benjamini_hochberg_adjusted over the raw p-values present, with m = BH_FAMILY_SIZE."
    ),
    "raw_p": "roi_ci_and_p p_value.",
    "p_absent_reason": "roi_ci_and_p p_value_absent_reason.",
    "n_blocks": "roi_ci_and_p n_blocks.",
    "n_push": "the bets whose in-force grade is push.",
    "n_ungraded": "0: the binding run refuses while one exists (D-24).",
    "fallback_fired": "always False (Phase 37 D-29).",
    "fallback_trigger": "None.",
}


# ---------------------------------------------------------------------------
# Which outcomes enter (Phase 34 D-06, D-20, D-21, D-24, D-26)
# ---------------------------------------------------------------------------

# Phase 34 D-06: the in-force outcome, read by the one reader of counted rows.
OUTCOME_SOURCE: str = (
    "each counted bet's in-force outcome, read by forward_ledger.verdict.verdict_rows under "
    "backtest/verdict_scope_2026.py (OUTCOME_RULE in_force_correction)."
)

# The 2025 push rule, unchanged.
PUSH_RULE: str = (
    "a push enters the test at payout 0 and stays in the bet count and in the staked "
    "total."
)

# D-21: a tie is a push for a WP moneyline bet.
TIE_RULE: str = (
    "a tied game grades a WP moneyline bet as a push: stake back, payout 0, and the bet stays in "
    "the bet count. ATS and O/U pushes are unchanged. The forward ledger's bet grading applies "
    "this from this record's commit. The gold label the re-fit accuracy comparison reads still "
    "scores a tie as home did not win."
)

# D-21: the one tie before this commit. Its score is a 2026 result and is deliberately not written.
TIE_DISCLOSURE: str = (
    "the only 2026 game that ended in a tie before this record's commit, 2026_W05_NYG@WAS, has no "
    "ledger row, so no counted result changes and no row is re-graded."
)

# D-20: a bet on a game never played leaves the test.
VOID_RULE: str = (
    "a counted bet on a game that is never played is VOID: it leaves the test entirely (it is not "
    "a push, and it is in neither the bet count nor the ROI denominator), it is listed by name in "
    "the readout, and its game leaves the coverage denominator as not played. The binding run "
    "does not wait on it."
)

# D-26: how "not played" is decided -- an inference, re-read on every run.
NOT_PLAYED_RULE: str = (
    "a game is not played when it is in scope and has no final score while a game of a later "
    "week already has one, or when its id is named by ledger, run-log or skip-record evidence for "
    "an in-scope week but is absent from the schedule. This is an INFERENCE read from the "
    "schedule each time the readout runs: a postponed game reads as not played until its final "
    "score arrives, and then returns to the denominator with its bets. The readout prints each "
    "such game with the reason it was inferred, and the binding run reads it once, after week 22 "
    "is graded."
)

# D-24: "ungraded" means counted bets of played games.
UNGRADED_SCOPE_RULE: str = (
    "the binding run refuses while any counted bet (status live) of a played game is ungraded. "
    "Suppressed rows are never graded and never block; a void bet does not block."
)


# ---------------------------------------------------------------------------
# Readings and exclusions (SPEC amendment 3, R8)
# ---------------------------------------------------------------------------

# SPEC amendment 3: where RFIT-08 binds.
RFIT_08_READING: str = (
    "RFIT-08 (per-generation first, pooled only as a secondary labelled figure) binds the re-fit "
    "accuracy comparison frozen in ROLLING-REFIT-PREREGISTRATION.md section 11. The money "
    "verdict is one test per target over the whole live season."
)

# Closing-line value enters neither the test, the family, the verdict word nor the headline.
CLV_ENTERS_VERDICT: bool = False


# ---------------------------------------------------------------------------
# The binding run (D-09, D-16, D-25)
# ---------------------------------------------------------------------------

# D-09: the only path the binding run writes.
BINDING_RECORD_PATH: str = "docs/records/SEASON-2026-READOUT.md"

# D-09: one binding run, ever.
ONE_SHOT_RULE: str = (
    "the binding run happens once, after week 22 is graded. It refuses when the binding record "
    "exists in the working tree or appears anywhere in git history, so deleting or moving it "
    "never reopens the run."
)

# D-09 / Plan 38-09: the binding run's full refusal contract.
BINDING_REFUSAL_RULE: str = (
    "the binding run also refuses while the ledger chain does not verify, while a counted bet of "
    "a played game is ungraded, while an in-scope game that is not 'not played' lacks a final "
    "score, and while the schedule holds no played week-22 game. It refuses when the rule it "
    "would apply is not the rule this record froze: when this record's commit cannot be found, "
    "when a later commit edits either record file without naming this commit as INVALIDATED or "
    "either file has an uncommitted change, when a frozen value (alpha, the bootstrap B and "
    "seed, the family size) no longer holds its frozen number, or when a git check fails. And it "
    "refuses when the re-fit comparison cannot run."
)

# D-16 / D-25: what may run before the season is graded.
PREVIEW_RULE: str = (
    "before then the readout runs only as a preview labelled 'season to date -- not the "
    "verdict': raw figures and coverage, no p-value, no verdict word, no re-fit comparison "
    "section, and no file written."
)


# ---------------------------------------------------------------------------
# Timing and the graded-weeks disclosure -- D-11 and D-12
# ---------------------------------------------------------------------------

# D-11: week 8's first decision run, Wed 2026-10-28 17:00 America/New_York (EDT), in UTC.
PREREGISTRATION_DEADLINE_UTC: str = "2026-10-28T21:00:00+00:00"

# D-12: the weeks holding at least one counted live-arm bet already graded when this record was
# committed, DERIVED from the ledger's graded_at immediately before the commit. A disclosure of
# what was visible when the rule was frozen, never a change of scope: these weeks count.
WEEKS_GRADED_AT_RECORD_COMMIT: tuple[int, ...] = (5, 6)
