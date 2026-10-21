# SEASON-VERDICT-PREREGISTRATION.md -- how the 2026 profit verdict is judged

**Status:** FROZEN. This document and `backtest/verdict_test_2026.py` are ONE pre-registration in
two files, landed in ONE commit containing nothing else. This is the human-readable half; the
constants module is the half the readout reads. Every value written in both places is the same
value.

**Why this record exists.** Until now only WHICH bets count in the 2026 verdict was frozen
(`backtest/verdict_scope_2026.py`, Phase 34). Nothing fixed HOW they are judged: no test, no
alpha, no multiplicity family and no verdict words for 2026. Every one of those choices, made after
results were visible, would be a choice made with the answer in view. This record fixes them now,
before the rest of the season's results exist, so a reader can check that the February answer was
reached by a rule written in October (CLEAN-04, D-11, D-12).

**Editing after the commit does not fix a bug -- it destroys the evidence.** A value that is wrong
here is wrong for the whole 2026 verdict. The only legitimate response is a NEW, VISIBLY LATER
CORRECTIVE COMMIT whose message explicitly INVALIDATES this pre-registration by naming its commit
sha. Never an edit in place, and never a quiet supersession. A later commit that touches either
file without naming this one fails a test.

**The anchor is recorded outside the files it witnesses.** Neither this document nor
`backtest/verdict_test_2026.py` records its own content hash or its own commit. A file that must
CONTAIN and exactly REPRODUCE its own whole-file hash is self-referential: writing the hash changes
the bytes the hash was computed over, so no fixed point exists. The witness lands in a LATER commit,
in a test-state module.

```
preregistration_anchor_recorded_in: tests/phase38_state.py
preregistration_anchor_slots: P38_PREREGISTRATION_WITNESS, P38_PREREGISTRATION_COMMITTED_AT
preregistration_paths: SEASON-VERDICT-PREREGISTRATION.md, backtest/verdict_test_2026.py
preregistration_deadline: 2026-10-28T17:00:00-04:00
```

The deadline is week 8's first decision run, Wed 2026-10-28 17:00 America/New_York (21:00 UTC).
This commit's committer date must precede it, and the commit must be on GitHub's `master` before
it.

---

## 1. What this record fixes, and what it does not

**It fixes how the 2026 money verdict is judged:** the test, the family, the pass line, the verdict
words and their 2026 meanings, which outcomes enter, and when the one binding run may happen.

**It does not restate which bets count.** That is `backtest/verdict_scope_2026.py`'s, by reference:
season 2026, weeks 5-22 inclusive, playoff weeks and neutral-site games counted (D40-08), the live
arm only, the outcome rule `in_force_correction`, the pricing and sizing rule `fill-v1`. This record
adds no second scope rule.

**It does not restate the re-fit accuracy comparison.** That comparison (paired per-game loss gaps,
per generation, its own minimum count, its own correction family and roll-up words) is frozen in
`ROLLING-REFIT-PREREGISTRATION.md` section 11 and is not repeated here.

## 2. The test (D-01, D-12)

**One test per target over the whole live season (D-01).** Each of WP, ATS and O/U gets exactly
one flat-ROI test over every counted live-arm bet of the verdict scope, read by
`forward_ledger.verdict.verdict_rows` under `backtest/verdict_scope_2026.py`, whichever re-fit
generation decided it.

**The statistic (D-12).** Flat-stake ROI: sum(payout_flat) / sum(flat_stake) over the target's
counted bets. This is the 2025 statistic, unchanged.

**The p-value (D-12).** `backtest.roi_significance.roi_ci_and_p`, the 2025 one-sided
null-recentred block-by-week bootstrap. Whole (season, week) blocks of the counted bets are
resampled with replacement, B = 2000 replicates at seed 2704 (both imported by identity from
`backtest.ev_chain_constants`, never retyped, so the p-value reproduces exactly). Each replicate is
recentred by subtracting the observed ROI, and

    p = (1 + the count of recentred replicates greater than or equal to the observed ROI) / (B + 1)

Bets spanning fewer than two blocks give no p-value, with `roi_ci_and_p`'s stated reason. The
minimum attainable p is 1/2001, which is below alpha 0.05, so the test can reach significance.

**The block design (D-01).** Re-fits install on Tuesday cutoffs, so by design no resampled
(season, week) block mixes two generations. This is a statement of the cadence design, not a
measured fact.

## 3. The family (D-02)

The Benjamini-Hochberg family is exactly three tests: the primary flat-ROI test of WP, ATS and O/U.
**m = 3**, passed explicitly to `backtest.profitability_2025.benjamini_hochberg_adjusted`. Nothing
else enters it.

Why not 2025's six (`BH_FAMILY_SPEC`):

- 2025's regular-season re-check existed for 2025's own playoff ruling (D31-38). In 2026 the playoff
  question is already settled: playoffs count (D40-08).
- 2025's calibration-fallback entry has no 2026 analogue, because the WP fallback is built into
  each generation's bet rule (Phase 37 D-29).

The re-fit accuracy comparison keeps its own separate family, already frozen in `e25884d`.

## 4. The pass line (D-04)

The line is strict on both sides:

- The profitable verdict needs at least one bet AND a flat ROI strictly above 0 AND a
  Benjamini-Hochberg-adjusted p strictly below alpha.
- The comparison against alpha is **strictly less than**: an adjusted p equal to alpha is not
  significant. With B = 2000 the p-value moves in steps of 1/2001, so exactly 0.05 cannot occur;
  the rule is stated anyway.
- Alpha is 0.05, imported from `backtest.diagnose` through `backtest.ev_chain_constants`, never a
  second literal.
- A flat ROI of 0 or less is the unprofitable verdict. A positive ROI whose adjusted p is alpha or
  more, or is absent, or whose minimum attainable p exceeds alpha, is the inconclusive verdict.
- Decisions use unrounded values; rounding is display only (SPEC AC 25).

The word is assigned by `backtest.profitability_2025.assign_verdict_token`, which already
implements exactly this rule, its arms in their pre-registered order.

## 5. The verdict words (D-03, D-13)

The words are 2025's five, imported by identity from `backtest.ev_chain_constants`
(`VERDICT_TOKENS`, `UNDISCHARGEABLE_NO_BETS`), never retyped. Each gets its own 2026 meaning:

- `PROFITABLE_CLEAN` -- at least one counted bet, the observed flat ROI over the whole live 2026
  season is strictly above 0, and the one-sided ROI p-value after the Benjamini-Hochberg correction
  over the three-test family is strictly below alpha. CLEAN here means measured forward under a rule
  frozen before the rest of the season's results existed. The weeks already graded at this commit
  count under the same rule; they are named in section 10, disclosed and never excluded.
- `UNPROFITABLE_CLEAN` -- at least one counted bet and an observed flat ROI of 0 or less. A measured
  result, not a failure to measure.
- `INCONCLUSIVE_CLEAN` -- at least one counted bet and a strictly positive flat ROI, but the
  adjusted p-value is alpha or more, could not be computed, or the minimum attainable p exceeds
  alpha. A positive return that cannot be told apart from zero is never called profitable.
- `UNDISCHARGEABLE_NO_BETS` -- the target has counted live-arm rows (its candidates were priced,
  status suppressed) but none of them is a bet (D-13). This is the zero-bet word.
- `UNDISCHARGEABLE_NO_CHAIN` -- the target has no counted live-arm row at all: it was never priced
  inside the verdict scope (D-13).

**How each target's measurement is filled for 2026 (D-13).** `assign_verdict_token` reads a
`TargetMeasurement`; in 2026 each field comes from here:

| Field | 2026 source |
|---|---|
| `target` | the target code: wp, ats or ou |
| `chain_resolved` | True when the target has at least one counted live-arm row |
| `control_passed` | always True: no counterfactual control exists in 2026, so the zero-bet split is decided by `chain_resolved` alone |
| `control_bet_count` | 0 |
| `bet_count` | the target's counted rows with status live, void bets excluded (D-20), pushes included |
| `flat_roi` | `roi_ci_and_p` point estimate over those bets |
| `adjusted_p` | `benjamini_hochberg_adjusted` over the raw p-values present, with m = 3 |
| `raw_p` | `roi_ci_and_p` p-value |
| `p_absent_reason` | `roi_ci_and_p`'s stated reason when the p-value is absent |
| `n_blocks` | `roi_ci_and_p` block count |
| `n_push` | the bets whose in-force grade is push |
| `n_ungraded` | 0: the binding run refuses while one exists (D-24) |
| `fallback_fired` | always False (Phase 37 D-29) |
| `fallback_trigger` | None |

The readout prints the token and its 2026 meaning above, never the 2025-worded reason string
`assign_verdict_token` also returns.

## 6. Which outcomes enter (Phase 34 D-06, D-20, D-21, D-24, D-26)

**The outcome.** Each counted bet's in-force outcome, read by `forward_ledger.verdict.verdict_rows`
under `backtest/verdict_scope_2026.py`: if an official score is corrected, the corrected grade
counts (Phase 34 D-06).

**Pushes.** A push enters the test at payout 0 and stays in the bet count and in the staked total
(the 2025 rule).

**Ties (D-21).** A tied game grades a WP moneyline bet as a push: stake back, payout 0, and the bet
stays in the bet count. ATS and O/U pushes are unchanged. The forward ledger's bet grading applies
this from this record's commit. The gold label the re-fit accuracy comparison reads still scores a
tie as home did not win.

**The tie disclosure (D-21).** The only 2026 game that ended in a tie before this record's commit,
`2026_W05_NYG@WAS`, has no ledger row, so no counted result changes and no row is re-graded.

**Void bets (D-20).** A counted bet on a game that is never played is VOID: it leaves the test
entirely (it is not a push, and it is in neither the bet count nor the ROI denominator), it is
listed by name in the readout, and its game leaves the coverage denominator as not played. The
binding run does not wait on it.

**Not played (D-26).** A game is not played when it is in scope and has no final score while a game
of a later week already has one, or when its id is named by ledger, run-log or skip-record evidence
for an in-scope week but is absent from the schedule. This is an inference read from the schedule
each time the readout runs: a postponed game reads as not played until its final score arrives, and
then returns to the denominator with its bets. The readout prints each such game with the reason it
was inferred, and the binding run reads it once, after week 22 is graded.

**Ungraded (D-24).** The binding run refuses while any counted bet (status live) of a played game is
ungraded. Suppressed rows are never graded and never block; a void bet does not block.

## 7. RFIT-08 and per-generation profit (SPEC amendment 3, D-01)

RFIT-08 ("per-generation first, pooled only as a secondary labelled figure") binds the re-fit
accuracy comparison frozen in `ROLLING-REFIT-PREREGISTRATION.md` section 11. It does not bind the
money verdict, which is one test per target over the whole live season.

Per-generation profit is shown only as plain numbers with no test: the `ROI_SECONDARY` lines of the
re-fit comparison command.

## 8. Closing-line value never enters

Closing-line value enters neither the test, the family, the verdict word nor the headline
(`CLV_ENTERS_VERDICT = False`).

## 9. The binding run (D-09, D-16, D-25)

**Where it writes.** `docs/records/SEASON-2026-READOUT.md` is the only path the binding run writes.

**Once.** The binding run happens once, after week 22 is graded. It refuses when the binding record
exists in the working tree or appears anywhere in git history, so deleting or moving it never
reopens the run.

**Its refusals.** It also refuses while the ledger chain does not verify, while a counted bet of a
played game is ungraded, while an in-scope game that is not "not played" lacks a final score, and
while the schedule holds no played week-22 game. It refuses when the rule it would apply is not the
rule this record froze: when this record's commit cannot be found, when a later commit edits either
record file without naming this commit as INVALIDATED or either file has an uncommitted change,
when a frozen value (alpha, the bootstrap B and seed, the family size) no longer holds its frozen
number, or when a git check fails. And it refuses when the re-fit comparison cannot run.

**Before then: preview only.** Until the binding run, the readout runs only as a preview labelled
"season to date -- not the verdict": raw figures and coverage, no p-value, no verdict word, no re-fit
comparison section, and no file written.

## 10. Disclosures

**Weeks already graded at this commit (D-12).** When this record was committed, the counted bets of
weeks 5 and 6 had already been graded (`WEEKS_GRADED_AT_RECORD_COMMIT`). The list was derived from
the ledger's grading times at or before the commit, not typed from memory. Those weeks count under
this same rule; naming them discloses what was visible when the rule was frozen and changes nothing
about the scope.

**No 2026 result is in this record.** This record holds no number derived from a 2026 result. Every
value in it is either reused from the 2025 pre-registration or fixed a priori. The week numbers
above and the tied game's id in section 6 are disclosures of what was visible, not results.
