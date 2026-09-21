# METHODOLOGY.md -- Modeling & Feature-Engineering Deep-Dive (Phase 23)

**Milestone:** v2.1 Trust & Reproducibility
**Phase:** 23 -- Documentation, Runbook & State-of-System
**Authored:** 2026-05-31 (plan 23-03)

> This is the portfolio-facing deep-dive into the system's modeling and feature-
> engineering methodology. It is the consolidated, de-staled successor to the two
> v1.0-era reference docs (the former `model_methodology.md` + `feature_engineering.md`,
> both deleted from the repo in this plan). It deliberately CROSS-LINKS the current code that
> establishes ground truth rather than re-asserting ~1,274 lines of v1.0 prose that
> has drifted past Elo activation (Phase 11), Optuna tuning (Phase 12), and dynamic
> blending (Phase 13). Where a number or a per-target verdict matters, it points at
> the forensic sources (`MODEL-DIAGNOSIS.md`, `AUDIT-REPORT.md`) instead of restating
> them, so this file cannot drift from the harness-reproducible truth.
>
> Read the "Drift landmines / what changed since v1.0" section first if you last read
> the v1.0 docs -- four things changed materially.
>
> MAINTAINED FORWARD (2026-08-24, Phase 30). This file is authored as a v2.1 document and
> is kept current rather than frozen. Section 10 adds the two-stage decision unit and the
> pre-registration discipline that v3.0 Phase 30 introduced; Sections 3, 4, 5, 6 and
> landmines 3 and 4 are reconciled to the Phase-30 end state. Superseded readings are
> relabelled with their reason rather than deleted.
>
> Status tags are ASCII (no emoji, per CLAUDE.md). Arrows are `->`, dashes are `--`.

---

## 1. Modeling philosophy

Three targets, one shared temporal-safe pipeline:

- **Win Probability (WP)** -- probability the home team wins.
- **Against the Spread (ATS)** -- home-team margin -> cover probability.
- **Over/Under (O/U)** -- game total -> over probability.

The differentiating principles (still true, and the genuinely portfolio-worthy depth):

- **Temporal ordering is the invariant, not an afterthought.** Every feature respects
  the information available before kickoff; every evaluation is walk-forward. There is
  no random cross-validation anywhere on the model path.
- **Calibrated probabilities, not labels.** Each target outputs a calibrated
  probability so "70% means ~70%" -- calibration quality (ECE, Brier reliability) is a
  first-class metric, not an afterthought.
- **Closing-line value (CLV) is the honest scoreboard.** For a near-efficient market,
  raw hit-rate is noisy; CLV against the closing line is the primary edge metric. See
  `MODEL-DIAGNOSIS.md` for the per-target verdict.
- **Reproducibility.** Fixed seeds, deterministic walk-forward splits, versioned
  artifacts. Any prediction is reproducible from the same input snapshot.

## 2. The three-layer temporal-safety design (still-true core)

This is the most differentiating part of the methodology and the reason the leakage
audit (AUDIT-03/04) passed. Temporal safety is enforced at three independent layers:

1. **Build-time time-fencing.** Features are computed strictly from data dated before
   kickoff; the gold matrices strip raw timestamps after fencing. Rolling team-form
   windows only look backward. Ground truth: `features/team_form.py`,
   `features/opponent_adj.py`, and the `as_of_datetime` enforcement on the build path.
2. **A structural LeakageGate.** A gate scans the assembled matrix for label-sibling
   and future-information columns before any target is fit. Ground truth:
   `features/validation.py` (`LeakageGate`, `FeatureValidator`). The ATS matrix
   carries a benign `home_margin` label-sibling that is excluded from the ATS
   feature list (catalogued as F-LEAK-01 in `AUDIT-REPORT.md`).
3. **Expanding-window normalization.** Z-scores are computed on an expanding window so
   a game is never normalized using statistics from its own future. Ground truth:
   `features/normalization.py` (`expanding_normalize`). The deprecated within-season
   normalizer is defined-but-unused on the live path.

Walk-forward splits keep the same discipline at the model layer: train < hp-val <
holdout, no overlap, no random CV. Ground truth: `models/temporal.py`
(`TemporalSplitConfig`, `WalkForwardSplitter`).

## 3. Feature engineering (cross-linked to current builders)

All builders conform to a structural `FeatureBuilder` Protocol (not an ABC), in
`features/protocol.py`. The current builder set and what each owns:

| Builder | Module | What it produces |
|---------|--------|------------------|
| Team form | `features/team_form.py` | Backward-rolling offensive/defensive EPA + success-rate trends |
| Opponent-adjusted EPA | `features/opponent_adj.py` | Strength-of-schedule-adjusted EPA (lag-correct; falls back to raw below the min-opponent-games threshold) |
| Elo features | `features/elo_features.py` | Per-game Elo snapshot differentials (see Elo below) |
| Contextual | `features/contextual.py` | Rest, travel, venue, divisional/conference flags (`is_divisional_game`) |
| Weather | `features/weather.py` | Outdoor weather, compressed to a small severity-oriented set (was 38 raw cols -> ~4) |
| Market anchors | `features/market_anchors.py` | Devigged market lines, compressed (was ~40 -> ~5), Friday 6 PM ET snapshot |
| QB tracking | `features/qb_tracking.py` | Composite QB adjustment (CPOE/EPA based) |

Validation and normalization are shared infrastructure: `features/validation.py`
(LeakageGate + FeatureValidator) and `features/normalization.py` (expanding-window).

Note on counts: the v1.0 feature doc quoted "89 features (31/29/29)". Those counts predate
the Phase-20 canonical-gold rebuild, which read 156/157/156 over 6263 rows / 2002-2025.
Both readings are superseded. Gold stands at **194/195/194 (WP/ATS/O-U) over 6,499 rows /
2002-2025** after the Phase-30 four-rung rebuild, counted empirically off the parquet --
NOT off the build's own summary print, which is low by a constant and was never used for
any published number. Treat the code and the gold schema as ground truth, not any quoted
count. See `AUDIT-REPORT.md` for the Phase-20 schema and `GATED-REFIT-READOUT.md` for the
Phase-30 rebuild and its per-rung attribution.

Two later families are in gold and are worth naming, because their presence is easy to
misread as endorsement. The Phase-28 `snap`, `injury` and `situational` groups were
SCREENED in Phase 28 and BINDINGLY ruled on in Phase 30 (Section 10): `snap` and
`situational` were KEPT, `injury` was DROPPED and is excluded at TRAIN time while its
columns stay physically in gold. The Phase-29 `line_movement` family LEFT gold entirely
at rebuild rung 3 -- not because it was measured unhelpful, but because the historical
archive floor post-dates the canonical feature-selection window, so 0 of its 15 columns
could ever be selected into a candidate. A family that cannot enter a model is not a
candidate feature set.

## 4. Elo ratings (ACTIVE -- a 538-style sequential system)

Elo is a real, active rating system -- not an inactive default. Ground truth:
`ratings/elo.py` and `scripts/build_elo.py`.

- **538-style margin-of-victory K-factor** -- the update scales with the upset-adjusted
  margin (`_calculate_k_factor`).
- **Per-season home-field advantage** -- HFA is fit per season (`hfa_by_season`), not a
  single constant; divisional matchups get a reduced HFA (`DIVISIONAL_HFA_FACTOR = 0.54`,
  from nfelo research).
- **Snapshot-then-update (no batch leakage)** -- each game's pre-kickoff rating is
  snapshotted BEFORE its outcome updates the rating, so a game never sees its own
  result. One snapshot per game; a count is deliberately not pinned here, because the
  game population moved in Phase 30 (gold went 6,263 -> 6,499 rows when the stale silver
  `games` mirror was re-synced) and a restated count is a drift liability. Gold's Elo
  columns are populated for every row of the widened frame.
- **2002 burn-in** -- ratings warm up from 2002 (about 19 seasons before the 2021 backtest
  start), so by the evaluation window ratings are stable rather than near a cold-start.

## 5. The three model trainers

All trainers delegate hyperparameter search to the shared Optuna infrastructure and
walk-forward splitter; per-target specifics:

- **WP** -- Logistic Regression with `StandardScaler`, calibrated with isotonic
  regression (Platt scaling as the small-sample fallback). Ground truth:
  `models/trainers/wp_trainer.py` + `models/calibrate.py`. Calibration quality is
  reported as ECE + a Brier reliability/resolution decomposition (`MODEL-DIAGNOSIS.md`
  records WP ECE 0.05499).
- **ATS** -- `XGBRegressor` predicts the home margin, then a residual-distribution
  converter turns the margin into a cover probability. Ground truth:
  `models/trainers/ats_trainer.py` (the `ResidualDistributionConverter` still lives in
  the legacy `models/train_ats.py` module -- see the deferred-refactor note below).
- **O/U** -- `XGBRegressor` predicts the game total, then a total-distribution
  converter turns it into an over probability. Ground truth:
  `models/trainers/ou_trainer.py` (the `TotalDistributionConverter` still lives in
  `models/train_ou.py`).

Hyperparameter tuning is `models/tuning.py` (`OptunaTuner`): a TPE sampler with a
Hyperband pruner over SQLite-persisted (resumable) studies. This is NEW in v2.0
(Phase 12); the prior random-search approach was replaced.

Per-target gating decides what actually ships: a candidate must clear the per-target
non-regression CLV floor vs the frozen incumbent (plus the secondary accuracy/MAE and,
for WP, calibration) before it is deployed. In v2.0 the retrained models did not pass
gating on any target. The v3.0 Phase-25 gated re-fit on the canonical Elo gold then
activated WP and ATS (ATS via one documented fix-cycle) and RETAINED v1.0 O/U (its re-fit
failed the floor and was honestly refused). The v3.0 Phase-30 re-fit on rebuilt, widened
gold ran the same gate a second time: WP passed and was promoted, ATS and O/U both failed
and RETAINED their incumbents. See landmine 3 below, `ACTIVATION-READOUT.md` for the
Phase-25 record, `GATED-REFIT-READOUT.md` for the Phase-30 record, Section 10 for the
two-stage decision unit Phase 30 introduced, and `MODEL-DIAGNOSIS.md` DIAG-05 for the
frozen v2.1 diagnosis.

**What the gate does NOT assert.** Under `floor_mode = non_regression` it asks whether a
candidate is not WORSE than the incumbent, never whether it is positive in absolute terms.
WP ships on a pooled probability CLV of -0.0380: an improvement on its incumbent, and still
negative. Removing closing-line-value leakage and having a market edge are different claims,
and this methodology keeps those two bars apart deliberately.

## 6. Market blending

The model output is blended with the market line. Ground truth: `models/blending.py`
(`BlendWeights`, `DynamicBlendWeights`, `MarketBlender`) + `models/blending_data.py`
(`TUNING_SEASONS`).

- **WP blends in log-odds space** (probability-correct interpolation); **ATS/O-U blend
  linearly** on the line.
- **Blend weights are tuned on pre-2018 data only** (`TUNING_SEASONS`), strictly
  isolated from both the 2018-2020 training window and the 2021-2024 backtest holdout --
  no information leakage from the evaluation period into the weights.
- **Dynamic (week-of-season) blending is gated per target (D-19), and the deployed
  artifact currently runs dynamic for ALL THREE targets.** `mode_by_target` is persisted
  in the blend artifact (`blend_dynamic_20260606_020635`). An earlier version of this
  section said only O/U was adopted with WP and ATS on static weights; that described a
  superseded artifact and is corrected here. Phase 30 re-ran the comparison against the
  newly serving models, in a throwaway COPY of the artifacts tree so production could not
  be mutated, and measured that the gating rule would now select static for WP and ATS and
  dynamic for O/U. The WP and ATS margins are 1.3e-4 and 2.8e-5, so the honest reading is
  "indistinguishable", not "harmful". **That verdict landed in the copy and was NOT applied
  to production** -- it is recorded as an open register, and acting on it would need the
  same paired-significance treatment the model gate uses rather than a bare relative-delta
  rule. See `GATED-REFIT-READOUT.md` and `STATE-OF-SYSTEM.md`.

## 7. Evaluation (CLV-first, calibration-aware)

- **CLV is primary** -- `models/clv.py` computes probability CLV (WP) and line CLV
  (ATS/O-U). For a near-efficient market this is the honest edge signal.
- **Calibration** -- ECE + reliability diagrams + a Brier reliability/resolution
  decomposition for the probability targets.
- **Hit-rate uses the LOCKED simulator convention** -- displayed ATS/O-U hit-rates come
  from the `BettingSimulator` sign convention (the honest ~51% for ATS, not a
  sign-flipped inflated number). Do not recompute hit-rate with ad-hoc heuristics.
- The full per-target accuracy / calibration / CLV verdict, with confidence intervals
  and the efficient-market-ceiling-vs-methodology-flaw call, is in `MODEL-DIAGNOSIS.md`
  -- this file does not restate those numbers.

## 8. Drift landmines / what changed since v1.0

If you previously read the now-deleted v1.0 docs, four things are materially different.
These are the corrections that motivated this de-staled consolidation:

1. **Elo is ACTIVE, not a placeholder.** The v1.0 docs implied default/inactive Elo.
   Production runs the real sequential Elo described in Section 4 (6263 per-game
   snapshots, 2002 burn-in). Any "Elo is an inactive default" claim is false.
2. **Tuning is Optuna, not random search.** Hyperparameter search is
   `OptunaTuner` (TPE + Hyperband + SQLite), new in Phase 12. The v1.0 random-search
   description is obsolete.
3. **Production serves a MIXED set, from THREE different vintages -- not a uniform v1.0
   or v2.0 set.** Per-target gating (D-17) first rejected the v2.0 retrained models on all
   three targets. The v3.0 Phase-25 re-fit on canonical Elo gold then went THROUGH the
   hardened non-regression gate: WP and ATS were activated (ATS via one documented
   fix-cycle), and O/U RETAINED the v1.0 pre-Elo model (its re-fit failed the floor and was
   honestly refused, D25-14). The v3.0 Phase-30 re-fit on rebuilt, widened gold ran the same
   gate again and moved exactly one key. The deployed set today:

   | Target | Serving | Vintage |
   |---|---|---|
   | WP | `wp_20260824_113325` | the Phase-30 re-fit -- PROMOTED |
   | ATS | `ats_20260605_220128` | the Phase-25 re-fit -- RETAINED, its Phase-30 candidate REFUSED |
   | O/U | `ou_20260326_163930` | v1.0 pre-Elo -- RETAINED, refused by the gate TWICE |

   Do NOT assume all three are v1.0, do NOT assume the v2.0 retrained models are live, and
   do NOT assume a phase that ran a re-fit therefore deployed one. The per-target records
   are `ACTIVATION-READOUT.md` (Phase 25) and `GATED-REFIT-READOUT.md` (Phase 30).
4. **Dynamic blend is gated per target, and all three targets currently run dynamic.**
   The deployed blend artifact carries `mode_by_target` dynamic for WP, ATS and O/U (D-19).
   An earlier version of this landmine said only O/U was adopted with WP and ATS on static
   weights; that is superseded and corrected in Section 6, together with the Phase-30
   re-measurement that would now prefer static for WP and ATS by margins too small to call
   a difference -- recorded, and NOT applied. The v1.0 docs describe no dynamic blend at all.

The deployed-artifact population and the walk-forward backtest population remain DISTINCT
populations (a single deployed artifact scored across the whole holdout vs fresh per-fold
models) -- a distinction that holds regardless of which artifacts are deployed.
`MODEL-DIAGNOSIS.md` quantifies the v2.1 production-vs-backtest mismatch (DIAG-05) and
carried the gated-re-fit recommendation; the v3.0 Phase-25 activation that executed it (WP
and ATS now on canonical Elo gold, O/U retained on v1.0) is recorded in
`ACTIVATION-READOUT.md`, which reconciles DIAG-05's "~zero" prediction with the measured
candidate CLV via that two-population distinction. Read the numbers there; do not conflate
the two populations.

## 9. Deferred refactor (honest note)

The ATS/O-U residual/total distribution converters still live in the legacy
`models/train_ats.py` / `models/train_ou.py` modules alongside the newer
`models/trainers/` package (README "Current Limitations" item 4). This coexistence is a
catalogued, deferred refactor -- not a bug. It is recorded in `STATE-OF-SYSTEM.md`.

## 10. The two-stage decision unit (v3.0 Phase 30)

Phase 30 added the most methodologically interesting machinery in the project, and it is
worth reading as a method rather than as a phase log. The problem it solves: when you widen
gold with several new feature groups and re-fit three targets on it, "did this help?" is not
one question, and answering it as one question is how a group's cost on one target rides
into production on another target's benefit.

**The decision is split into two stages that answer different questions.**

- **Stage 1 is a per-GROUP selection rule.** For each (group, target) cell it measures the
  add-one-in CLV delta -- a baseline leg trained WITHOUT any of the registered signal groups
  against a candidate leg with exactly one group added -- on the 2021-2024 holdout, paired
  per game, both legs untuned so the comparison is not a hyperparameter search in disguise.
  The verdict is per-group, not per-cell, so a group whose evidence is directionally split is
  carried whole.
- **Stage 2 is the existing per-TARGET deploy gate.** It takes Stage 1's kept set as the
  candidate feature set and asks the separate question of whether THIS target's candidate is
  not worse than THIS target's incumbent.

**The split is what makes a split verdict safe.** On the Phase-30 run `snap` was KEPT on the
strength of its WP cell while being significantly NEGATIVE on ATS and O/U. Stage 1 carried
the group whole; Stage 2 then refused the ATS and O/U candidates, in the same neighbourhood
Stage 1 had predicted. Stage 1 measured the cost and Stage 2 refused it. Two refusals were
the predicted outcome of a ratified rule, not a surprise to debug.

**Pre-registration, checked from git rather than asserted in prose.** The Stage-1 rule --
alpha, the minimum detectable effect, the correction method, the family, the rank order, the
verdict vocabulary, the measurement-exclusion reasons and the permitted fix-cycle levers --
lives in a single frozen module and was committed in a commit containing no measurement. The
claim "the rule predates the result" is then an ANCESTRY relation between two specific
commits, both published, both resolved from git by a committed test, and asserted to be
non-equal: a rule and the results it produced landing in one commit is not a pre-registration,
only a claim of one. The MDE is pre-registered as a FORMULA at 80% power rather than as a
number, so the bar survives the rebuild moving the underlying standard deviations and nothing
can be shopped after the fact.

**The exclusion rule had to be frozen because it DETERMINES the denominator.** A cell is
excluded from the family for four pre-registered reasons, the sharpest being "no column of
this group was selected by this target's own feature selection" -- the candidate model never
saw the group, so the delta is selection churn rather than lift. That rule threw out the most
significant cell in the entire grid (`injury`/wp at p = 3.6e-15), and had that cell been
admitted the `injury` verdict would have flipped from DROP to KEEP on the strength of a number
that says nothing about injuries. Every exclusion also reduces `m`, so a denominator chosen
after seeing which cells were awkward would be the rule-shopping the correction exists to
prevent.

**The correction is applied across the FULL grid, and what that costs is stated rather than
hidden.** The family is all 3 groups x 3 targets, using Benjamini-Hochberg as a vendored
inclusive step-up. Choosing the full grid rather than a per-target family is the stricter
option and was chosen for that reason. Honesty runs in both directions here: the REALIZED
denominator was m = 6 rather than 9, because three cells were excluded, which made the
surviving family EASIER to reject in, not harder. The anticipating plan text stated the m = 9
case; that was wrong in the permissive direction, which is the direction that matters, and it
is corrected on the record rather than quietly restated.

**The verdict vocabulary is three-valued, and UNDETERMINED is never collapsed into DROP.**
`KEEP` / `DROP` / `UNDETERMINED` / `NOT MEASURED`. UNDETERMINED resolves to DROP for the
DEPLOY decision -- an undetermined group is excluded from the candidate set for the same
practical reason a dropped one is -- but it is REPORTED as UNDETERMINED, because "we measured
this and it hurt" and "we could not tell" are different findings and a record that merges them
loses the one a later phase needs. NOT MEASURED is not a verdict about the group at all; it is
a refusal to rule. On the Phase-30 run nothing landed on either arm, and that is stated rather
than passed over: the machinery exists, is tested, and simply had no occasion to fire.
The arm order is KEEP-before-DROP, which resolves a split group toward KEEP; it reverses
Phase 28's rule and was ratified AS a reversal rather than adopted silently.

**Normalization bounds are fitted on strictly-prior seasons, with an accepted residual.**
Imputation medians and q01/q99 winsorization bounds are refitted per season on strictly
prior seasons, which removes the cross-season leak that whole-frame fitting introduces.
It does NOT establish the within-season property -- that a value is fitted only on
information available before its own kickoff inside its own season. That residual is
accepted and recorded rather than closed, and `STATE-OF-SYSTEM.md` names it.

**One thing the deltas above cannot be read as.** Feature selection on the 534-row 2018-2019
training window was measured admitting synthetic noise columns over real features: with 50
unit-variance Gaussian columns appended, 6 of ATS's 25 and 9 of O/U's 25 selected features
were pure noise. So the per-group deltas are DIRECTIONAL EVIDENCE, not precise estimates.
That is a statement about how the numbers are read, not a task waiting to be done: a column
with variance but no target relationship is indistinguishable from a weak real signal at fit
time, and excluding it would require peeking at the target during selection -- a different
rule, and a leakage hazard of its own. `GATED-REFIT-READOUT.md` section 5d and
`SELECTION-CENSUS.md` carry the measurement.

**Rebuild attribution, and why it is not the same as health.** Gold was rebuilt FOUR times,
one named cause per rung, each judged mechanically against a signature declared before the
rung ran, because rebuilding once for four reasons makes every moved column unattributable.
The load-bearing lesson is the failure it did NOT catch: one rung attributed perfectly
cleanly while having silently flattened 18 columns through a bound-fitting cascade and a
degenerate clip. The judge could not see it, because that rung's signature deliberately
attributes every changed column. Only re-measuring per-column health against a committed
pre-rebuild fixture found it. **Measurement beats attribution**, and every later rung was
re-measured the same way.

## 11. The one-shot clean-split profitability measurement (v3.0 Phase 31)

Phase 30's machinery answers "is this candidate worse than what is serving". It never answers
"does any of this make money", and the two questions need different instruments. Phase 31 built
the second one and spent it once.

**Three per-target EV chains, one bet decision source.** Each target converts a model output into
a per-bet expected value against a real market price, and all three route through the SAME
`BetSelector.select` facade -- one bet decision source, so a bet that appears in the backtest, in
the weekly list and in the verdict was decided by one code path. The win-probability chain uses
the deployed isotonic calibration unchanged and prices at the game's own two-sided moneyline. The
spread chain applies a prior-season mean bias correction to the predicted home margin, converts
through a frozen tune-window residual SD, and prices by devigging the stored two-sided spread
juice. The totals chain does the same on the total scale, restricted to the pre-registered
under-or-high-total sub-population. Sizing is a LOCKED order: quarter Kelly, then a per-bet cap,
then a same-game-and-side de-weight, then a pooled weekly exposure cap.

**The design is ONE-SHOT, and that is a method rather than a precaution.** Every prior
profitability figure in this project was measured on a holdout that had already been looked at,
and each was labelled provisional for exactly that reason. The 2025 season was the only season
never used for anything. Spending it therefore had to be irreversible and unrepeatable, or it
would silently become another burned split:

- The rule -- windows, game-type scope, eligibility, calibration policy, devig source, sizing
  order, robustness cuts, the multiplicity family and the ROI hypothesis test -- was frozen in
  two files BEFORE any 2025 number existed. The claim that the rule predates the result is an
  ANCESTRY relation between two specific commits, resolved from git by a committed test, and
  asserted to be non-equal, exactly as in section 10.
- A durable run ledger is created by an EXCLUSIVE file creation BEFORE the first hold-season
  read. If it already exists the runner REFUSES, and there is no force flag. A crash between
  reading the split and writing the verdict leaves a `failed` attempt on disk rather than a
  silently re-spendable split; returning to `armed` requires an owner ruling recorded in the
  ledger, which is a deliberate human edit to a tracked file and leaves a diff.
- The runner additionally hard-refuses to overwrite an existing verdict artifact. Silently
  re-arming a crashed attempt would spend the split twice while leaving a record saying it ran
  once.

The cost of this design is that the result cannot be checked by re-running it, so the readout's
drift guard checks CONSISTENCY against the committed artifact instead of reproduction. That trade
is stated rather than discovered.

**The verdict vocabulary is a closed five-value set, and "no bets" is a RESULT.**
`PROFITABLE_CLEAN` / `UNPROFITABLE_CLEAN` / `INCONCLUSIVE_CLEAN` / `UNDISCHARGEABLE_NO_BETS` /
`UNDISCHARGEABLE_NO_CHAIN`, each bound in the frozen module to the condition that produces it, so
a reader can check afterwards that the mapping from numbers to words was fixed before the numbers
existed. Two distinctions carry the weight. A positive return whose p-value does not clear alpha
is `INCONCLUSIVE_CLEAN` and is NEVER called profitable -- including when the minimum attainable
bootstrap p exceeds alpha, which is reported with that stated reason rather than as a near miss.
And a chain that ran and selected zero bets reports `UNDISCHARGEABLE_NO_BETS`, filling the same
readout template slots as any other verdict, which is what stops a null result from being quietly
reported as "ROI 0" or omitted as a gap. `UNDISCHARGEABLE_NO_CHAIN` is kept distinct from it:
there the chain could not run at all, which is a different fact about a different failure.

**The hypothesis test is on ROI, and CLV is REPORT-ONLY.** The verdict rests on a one-sided
achieved significance level from a NULL-RECENTRED block-by-week bootstrap -- weeks are the block
unit because bets inside a week share line movement and a game-level resample would understate
the variance. Closing-line value is measured over the same selected bets and carried BESIDE the
verdict: it never entered the multiplicity family and is structurally incapable of driving a
token. **The two are different tests of different quantities and this project keeps them apart on
purpose.** A model can beat the closing number decisively and still return nothing
distinguishable from zero, which is precisely what the 2025 run measured for the win-probability
target; treating the CLV p-value as evidence of profitability would produce the exact
misreading Phase 26 diagnosed for totals.

**The correction family counts hold-side inferences only, and every other read is still on the
record.** The Benjamini-Hochberg denominator is the primary and robustness entries across all
three targets in ONE pooled registry. A tune-side threshold-sweep cell never touched the hold and
a counterfactual control has no p-value to correct, so neither enters the denominator -- but both
are RECORDED in the registry, which is what lets a reader confirm the denominator arithmetically
rather than take it on trust. A denominator that cannot be checked is a denominator that can be
quietly reduced.

## Cross-references

- **`PROFITABILITY-READOUT.md`** -- the Phase 31 milestone close: the per-target 2025
  clean-split verdict produced by the chain described in section 11, what is deployed and
  what was retained, the absolute closing-line value per target, and the accepted
  disclosures. The source for every 2025 figure referenced above.
- **`MODEL-DIAGNOSIS.md`** -- the Phase 22 honest accuracy diagnosis: per-target
  accuracy/calibration/CLV, the ceiling-vs-flaw verdict, and the production-vs-backtest
  mismatch (DIAG-05). The source for every quantified claim referenced above.
- **`AUDIT-REPORT.md`** -- the Phase 20 data & feature correctness audit: canonical-gold
  schema, the leakage/temporal verdict, and the F-* catalogue.
- **`ACTIVATION-READOUT.md`** -- the v3.0 Phase-25 activation record: the first gated
  re-fit, per-target CLV before/after, and the origin of the retained ATS and O/U
  incumbents.
- **`GATED-REFIT-READOUT.md`** -- the v3.0 Phase-30 gated re-fit record and the source for
  every Phase-30 number referenced above: the four attributed rebuild rungs, the frozen
  Stage-1 rule and its corrected 9-cell grid, the three group verdicts, the per-target
  deploy outcome (one promotion, two refusals), and the open registers and quarantines.
- **`SELECTION-CENSUS.md`** -- the feature-selection census behind Section 10's noise-column
  finding, including the ratified selection rule quoted verbatim.
- **`STATE-OF-SYSTEM.md`** -- the consolidated registry of what is trustworthy, what was
  fixed, and the single open list (including the accepted within-season residual named in
  Section 10).
- **`README.md`** -- the portfolio front door and the single architecture diagram.
- **`PIPELINE.md`** -- the canonical run sequence that produces the gold, artifacts,
  and predictions this methodology describes.

<!-- old-rule-addendum-2026-09-15 -->

## Old-rule addendum (2026-09-15)

This section was added on 2026-09-15. Nothing above it has been changed: every number, table and
heading is exactly as it was first published.

Phase 33.2 found that the model inputs behind the results in this document were defective, in five
ways. Weather observed after each game stood in for the forecast that was actually available the day
before kickoff. Closing betting lines, which are only known at kickoff, were fed into the models as
inputs. Feature builders took their cutoff from one global clock instead of each game's own lock
time. The opponent adjustment never actually ran. Early-season placeholder values read as exactly
league average, with nothing to say they were placeholders.

This document stays in the record, unedited, because deleting it would be worse: it would hide what
was claimed and when. Read it as history, not as a measure of how well the system works.

**Built under the old rule on inputs later found defective; not evidence.** Only the 2026 season,
recorded live under the day-before 6 PM ET lock, counts (D33.2-07). See Phase 33.2.

What this covers in this document: every result it quotes -- the calibration and closing-line-value figures and the deploy gate outcomes -- is old-rule, and where it describes the preceding-Friday freeze, the day-before 6 PM ET lock now replaces it.
