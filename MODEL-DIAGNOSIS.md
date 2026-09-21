# MODEL-DIAGNOSIS.md -- Honest Accuracy Diagnosis (Phase 22)

**Milestone:** v2.1 Trust & Reproducibility
**Phase:** 22 -- Honest Accuracy Diagnosis
**Authored:** 2026-05-31 (plan 22-03)
**Backtest window:** 2021-2024 walk-forward (weeks 1-22, postseason included)
**Realistic bars:** WP straight-up ~67%; ATS/O-U breakeven ~52.4%

> This is the forensic "state of the model" deliverable for Phase 22 (D-09). It reports the
> system's per-target accuracy (DIAG-01), calibration (DIAG-02), and closing-line value
> (DIAG-03) against realistic bars, renders a per-target efficient-market-ceiling-vs-
> methodology-flaw verdict (DIAG-04 / D-06), and quantifies the production-vs-backtest
> mismatch as a clean 2x2 with a re-fit recommendation for a FUTURE milestone (DIAG-05 /
> D-07 / D-08).
>
> Every load-bearing number is reproducible from the committed re-runnable harness
> `backtest/diagnose.py` (`run_diagnosis`) and the regenerated `outputs/backtest/
> metrics_summary.json` (plan 22-02). The numbers are anchored to the Phase-20
> post-rebuild canonical-gold values recorded in `AUDIT-REPORT.md` (WP pooled accuracy
> 0.66725, headline WP blended CLV -0.00207, ATS/OU blend delta 0.0, flat-stake ROI
> -0.00018 / win-rate 0.5677). Phase 23's state-of-system summary (DOC-03) cites this file.
>
> Status tags are ASCII `[CEILING]` / `[FLAW]` / `[MIXED]` (no emoji, per CLAUDE.md).
>
> HARD BOUNDARY (carried from Phase 20 D-01, the milestone's namesake): this is a
> measurement + interpretation + write-up deliverable ONLY. NO deployed-artifact re-fit,
> NO canonical-gold rebuild, NO new features / re-tuning / algorithm swaps were performed.
> DIAG-05 produces a RECOMMENDATION only; the re-fit work itself is explicitly deferred to
> a future model-improvement milestone. Loading and scoring the deployed artifacts over the
> backtest period is inference, NOT a re-fit (D-07).

---

## Reading guide -- two distinct model populations

This report carefully separates TWO different sets of models, because conflating them is the
single biggest source of misreading the numbers:

- **Backtest models** -- the walk-forward `BacktestEngine` fits a FRESH per-fold model for
  each holdout season (2021..2024) on the Phase-20 rebuilt canonical gold. This is
  inference-for-evaluation, the established backtest behavior; it is NOT a deployed-artifact
  re-fit. These are the "what the methodology can do on clean current data" numbers.
- **Deployed (production) artifacts** -- the v1.0 pre-Elo `wp_20260327_114739`,
  `ats_20260326_163724`, `ou_20260326_163930` models that production actually serves for
  live current-week predictions (resolved via `artifacts/latest.json`). DIAG-05 LOADS and
  SCORES these over the same 2021-2024 gold (inference only) to produce a true
  apples-to-apples comparison.

The DIAG-01..04 verdict basis is the **backtest models** on canonical gold (the honest
measure of the methodology). DIAG-05 is the **production-vs-backtest** comparison and is the
basis for the re-fit recommendation.

Per-target prediction streams: each metric is reported for BOTH the **raw** model output and
the **market-blended** output (D-02). The blend is static for WP/ATS and dynamic
week-of-season for O/U (D-19); consequently the ATS/OU blended-vs-raw headline CLV delta is
0.0 by design, while WP's blend is active.

**Which blend path the blended cuts reproduce (important):** the blended cuts here apply the
deployed dynamic blend artifact DIRECTLY via `MarketBlender.from_artifacts`, matching the
LIVE current-week path (`scripts.generate_current_week_predictions.apply_blending`), which
passes week/season and therefore exercises the O/U dynamic week-of-season schedule. Note that
`backtest/run.py --blend` passes only the static `BlendConfig` to the engine (it does NOT pass
`dynamic_weights`), so the backtest report's blended O/U cut is STATIC. The blended-backtest
O/U number reported below (`+1.02185`) is therefore reproducible only via `run_diagnosis`
(`backtest/diagnose.py`, mirroring live), NOT via `backtest/run.py --blend`. The harness
deliberately mirrors the LIVE production blend, not the static backtest-report blend.

---

## DIAG-01 Accuracy

### Population, bars, and the honest convention

- **WP** is graded straight-up (did the higher-win-probability side win?) against the
  realistic **~67% bar** (the practical ceiling for pre-game NFL win prediction).
- **ATS and O/U** are reported in BOTH populations (D-01), each against the **52.4%**
  breakeven bar (the rate needed to overcome standard -110 juice):
  - **all-games straight-pick rate** -- every game graded against the line (large N, tests
    raw spread/total-beating skill);
  - **edge-filtered bet rate** -- only the bets the `BettingSimulator` would actually place
    above its canonical `min_edge_threshold = 0.02` (smaller N, tests whether edge selection
    adds value).
- The hit-rate uses ONLY the LOCKED `BettingSimulator` sign convention (the same honest
  convention the season/betting dashboards use). It is explicitly NOT the sign-flipped ~78%
  figure, and explicitly NOT the simplified `metrics.py` `cover_accuracy` / `over_accuracy`
  heuristics (those are not graded against the actual line and must never be used as the
  hit-rate -- AUDIT-REPORT / 22-RESEARCH Anti-Patterns).

### Headline accuracy -- backtest models on canonical gold (pooled 2021-2024)

| Target | Metric | Population | Backtest (raw) | Bar | Read |
|--------|--------|------------|----------------|-----|------|
| WP  | straight-up accuracy | all games (n=1139) | **0.66725** | ~0.67 | essentially AT the bar |
| WP  | accuracy | edge-filtered (n=1034) | 0.67408 | 0.524 | well above breakeven |
| ATS | hit-rate | all-games straight-pick (n=1087) | 0.53266 | 0.524 | marginally above breakeven |
| ATS | hit-rate | edge-filtered (n=1077) | 0.53110 | 0.524 | marginally above breakeven |
| O/U | hit-rate | all-games straight-pick (n=1087) | 0.48482 | 0.524 | below breakeven |
| O/U | hit-rate | edge-filtered (n=1083) | 0.48661 | 0.524 | below breakeven |

WP pooled accuracy is the **n-games-weighted** pooled figure (0.66725), NOT the unweighted
mean of season accuracies (which happens to be 0.66726 here -- they coincide because season
N is nearly equal, but the weighted figure is the verdict basis per D-03). ATS/OU pooled MAE
(the regression-quality companion to the hit-rate) is ATS 10.122 points, O/U 10.499 points
(matches the AUDIT-REPORT post-rebuild values).

### All-games vs edge-filtered gap (D-01 finding)

The all-vs-filtered gap is small and -- importantly -- the edge filter does NOT
systematically improve the hit-rate on the backtest models:

| Target | straight-pick | edge-filtered | gap (straight - filtered) | n filtered out |
|--------|---------------|---------------|---------------------------|----------------|
| WP  | 0.66421 | 0.67408 | -0.00987 | 53 |
| ATS | 0.53266 | 0.53110 | +0.00155 | 10 |
| O/U | 0.48482 | 0.48661 | -0.00179 | 4 |

**Finding:** Edge selection moves the hit-rate by under 1 point in either direction on every
target. For WP the edge filter is mildly helpful (+0.99 pts); for ATS it is mildly harmful;
for O/U it is neutral. The edge filter is NOT adding meaningful selection value over the
all-games population on the backtest models -- consistent with a near-efficient market where
the model's per-game edge estimate is itself noisy. (Note: the WP "hit-rate" here is the
simulator's per-bet win-rate on the WP market, distinct from the straight-up accuracy in the
headline table above, which is why WP appears in this table at all.)

### Per-season breakout (D-03 robustness check -- WP straight-up accuracy)

| Season | n games | WP accuracy | WP MAE | ATS MAE | ATS R2 | O/U MAE | O/U R2 |
|--------|---------|-------------|--------|---------|--------|---------|--------|
| 2021 | 285 | 0.63509 | 0.42356 | 11.104 | 0.1698 | 10.749 | 0.0472 |
| 2022 | 284 | 0.67254 | 0.44123 |  9.209 | 0.0755 | 10.893 | 0.0320 |
| 2023 | 285 | 0.66316 | 0.43548 | 10.333 | 0.1186 | 10.682 | 0.0488 |
| 2024 | 285 | 0.69825 | 0.42475 |  9.838 | 0.2196 |  9.673 | 0.0913 |
| **Pooled** | **1139** | **0.66725** | **0.43125** | **10.122** | **0.1459** | **10.499** | **0.0549** |

**Finding:** WP accuracy is stable across seasons (0.635-0.698) -- the pooled 0.66725 verdict
is not driven by a single year (2021 is the low, 2024 the high). ATS R2 is modest (0.07-0.22)
and O/U R2 is low (0.03-0.09) across every season -- the regression targets explain only a
small fraction of margin/total variance, which is expected for an efficient line and frames
the DIAG-04 verdict for those targets onto CLV + calibration, NOT raw point-prediction error.

### Postseason inclusion (D-04)

Postseason **IS included** in the scored population. The backtest engine (`backtest/
engine.py`, used by `backtest/run.py`) applies no week filter; it scores ALL holdout-season
gold rows, weeks 1-22. Each 2021-2024 season carries ~284-285 games versus ~272
regular-season games -- i.e. roughly **13 postseason games per season (weeks 19-22), ~52
playoff games total** across the window. Caveat: playoff N is small and playoff lines behave
differently (sharper market, fewer public-money distortions); the pooled hit-rates and CLV
therefore include a small, structurally-different playoff slice and should not be read as
regular-season-only. (The older `backtest/walkforward.py` week-range config that filtered to
weeks 1-18 is a separate, unused module -- it does NOT govern this population.)

---

## DIAG-02 Calibration

Calibration is a **WP-specific** concept here: WP is a probability model, so it has a true
reliability curve; ATS and O/U are XGBoost regression targets (predicting margin / total),
so "calibration" for them is framed as residual quality (Brier-resolution-style), NOT a
probability reliability curve (Open Question 1).

### WP calibration -- backtest models (the reliability evidence)

| Metric | Backtest (raw WP) | Interpretation |
|--------|-------------------|----------------|
| ECE (expected calibration error) | **0.05499** | mean gap between predicted and observed win-rate across probability bins |
| Brier score | 0.22113 | overall probabilistic accuracy (lower better) |
| Brier reliability | 0.00521 | calibration component (lower better -- near 0 = well-calibrated) |
| Brier resolution | 0.03179 | discrimination component (higher better -- the model separates outcomes) |

The regenerated interactive HTML report (`outputs/backtest/backtest_report.html`, plan 22-02,
owner-verified) renders the WP reliability diagram showing per-season and Overall curves
against the perfect-calibration diagonal at **ECE = 0.0550**, plus the cumulative CLV chart.
That HTML report is the DIAG-02 / DIAG-03 visual evidence (D-09).

**Read:** a Brier reliability component of ~0.005 and ECE ~0.055 indicate WP is
**well-calibrated** -- when the model says 65%, the realized rate is close to 65%. The small
positive resolution (0.032) confirms the model is also discriminating, not just well-hedged.

### ATS / O/U residual quality (regression "calibration")

ATS/OU have no probability reliability curve; their quality is the residual / regression
fit already in `metrics_summary.json`: pooled ATS MAE 10.122 pts (R2 0.1459), O/U MAE 10.499
pts (R2 0.0549). The low R2 reflects an efficient line that already prices most of the signal
-- the residual spread is wide and roughly symmetric, so there is no "miscalibration" to
report in the WP sense; the honest edge metric for these targets is CLV (DIAG-03), not
point-error. Probability reliability is therefore reported for WP only, and ATS/OU are judged
on CLV + residual quality (consistent with the Core Value framing "ATS/OU judged on
calibration + CLV, not raw hit-rate").

---

## DIAG-03 CLV (Closing Line Value)

CLV measures whether the model's implied price beats the market's closing price -- the honest
edge metric. Per Pitfall 2, the right CLV column DIFFERS by target:

- **WP** -> `probability_clv` (devigged win-probability CLV; small dimensionless values).
- **ATS / O/U** -> `line_clv` (spread / total points; larger by design, NOT comparable to a
  probability). The `probability_clv` column that the engine also writes for ATS/OU is a
  WP-only-meaningful artifact (the margin/total is pushed through the WP devig path) and is
  NOT used for the ATS/OU verdict.

Each target is reported with its mean, t-statistic, p-value, and 95% CI over the n=1087 games
that carry closing odds (52 of 1139 games lack a closing snapshot and are excluded). The
D-05 gate: a negative CLV is only called a "systematically negative methodology concern" when
mean < 0 AND p < 0.05; otherwise it reads "indistinguishable from zero, consistent with an
efficient-market ceiling."

### CLV -- backtest models on canonical gold (the verdict basis)

| Target | CLV column | Stream | mean | t-stat | p-value | 95% CI | D-05 verdict |
|--------|-----------|--------|------|--------|---------|--------|--------------|
| WP  | probability_clv | raw     | -0.00347 | -1.321 | 0.1869 | (-0.00862, +0.00169) | indistinguishable from zero |
| WP  | probability_clv | blended | -0.00207 | -1.309 | 0.1907 | (-0.00516, +0.00103) | indistinguishable from zero |
| ATS | line_clv | raw     | -0.00070 | -0.008 | 0.9940 | (-0.18397, +0.18256) | indistinguishable from zero |
| ATS | line_clv | blended | -0.00038 | -0.008 | 0.9940 | (-0.10036, +0.09960) | indistinguishable from zero |
| O/U | line_clv | raw     | +1.62084 | +21.166 | 1.6e-83 | (+1.47058, +1.77110) | systematically POSITIVE -- a real edge |
| O/U | line_clv | blended | +1.02185 | +21.394 | 5.2e-85 | (+0.92813, +1.11557) | systematically POSITIVE -- a real edge |

**Headline note:** the `metrics_summary.json` headline_clv values (WP -0.00207, ATS +1.0762,
O/U +45.536) are plain MEANS over the merged stream and -- for ATS/OU -- carry the WP-path
`probability_clv` magnitude, which is why the ATS/OU headline numbers look large and are NOT
directly the `line_clv` means above. The significance verdict uses the per-target `line_clv`
(ATS/OU) / `probability_clv` (WP) means computed fresh via `scipy.stats.ttest_1samp` (Pitfall
1: the headline mean has no p-value; significance is computed separately in the harness).

### Raw-vs-blend CLV delta finding (D-02)

- **WP:** the blend IMPROVES CLV from raw -0.00347 to blended -0.00207 (delta +0.00140) --
  the market blend is doing useful work, pulling the model's price toward the closing line.
  Both stay statistically indistinguishable from zero.
- **ATS:** blend delta on the headline is 0.0 (WP/ATS static blend per D-19); the line_clv
  mean is statistically zero either way.
- **O/U:** the dynamic blend (D-19, the only adopted dynamic blend) shrinks the raw +1.62084
  line_clv edge to a blended +1.02185 -- i.e. blending toward the market REDUCES the O/U
  line_clv edge while remaining strongly significant. This is the documented raw-vs-blend gap:
  for O/U the raw model carries more line_clv than the blended production stream, but both are
  a real, significant edge.

---

## DIAG-04 Verdict (ceiling vs methodology flaw, per target)

> The numeric thresholds in this section are **OWNER-APPROVED as drafted, 2026-05-31 (v2.1
> Phase 22)** (D-06). They were surfaced for owner review and accepted without adjustment, so
> they are now ADOPTED rather than provisional. The verdict uses a TARGET-APPROPRIATE rubric
> (not one uniform metric): WP on calibration + ~67% hit-rate; ATS/OU on significance-gated
> CLV + calibration, with hit-rate vs 52.4% as supporting context.

### Numeric thresholds -- OWNER-APPROVED 2026-05-31 (adopted as drafted)

| Threshold | Drafted value | Rationale |
|-----------|---------------|-----------|
| WP ECE "well-calibrated" cutoff | ECE <= 0.06 | An ECE of ~5-6 win-probability points is a standard "good calibration" bar for a 50-bin reliability curve on ~1100 games; the WP model sits at 0.05499, inside it. |
| WP "near the bar" tolerance | within 1.5 pts of 0.67 (i.e. accuracy >= 0.655) | The ~67% bar is a soft practical ceiling, not a hard line; a 1.5-pt band absorbs season-to-season noise (the per-season range is 0.635-0.698). WP pooled 0.66725 is inside it. |
| ATS/OU "near the bar" tolerance | hit-rate within 1.0 pt of 0.524 counts as "at breakeven"; >= 1.0 pt below = "below the bar" | A 1-pt band reflects that ~10 games of the 1087 swing the rate by ~1 pt; tighter would over-read noise. |
| CLV significance line | "systematically negative methodology concern" ONLY when mean CLV < 0 AND p < 0.05; "real edge" when mean > 0 AND p < 0.05; otherwise "indistinguishable from zero" | The D-05 gate verbatim -- prevents calling sample-noise a flaw. |
| Brier-resolution "discriminating" floor | resolution > 0.02 | Resolution above ~0.02 on this Brier scale indicates the model meaningfully separates outcomes rather than predicting the base rate; WP is at 0.032. |

### Per-target verdict (backtest models on canonical gold)

**WP -- [CEILING].** WP is well-calibrated (ECE 0.05499 <= 0.06 cutoff; Brier reliability
0.00521 near zero; resolution 0.032 above the discriminating floor) and its pooled straight-up
accuracy 0.66725 sits essentially AT the ~67% bar (inside the 1.5-pt tolerance, stable across
seasons). CLV is indistinguishable from zero (p=0.19). This is the profile of a model that has
reached the practical efficient-market ceiling for pre-game win prediction -- it is accurate
and honest about its uncertainty, and it does not systematically beat the closing line (which
no public-feature WP model should be expected to do). Verdict: **efficient-market ceiling, not
a methodology flaw.**

**ATS -- [CEILING].** ATS hit-rate sits marginally above breakeven (0.533 straight-pick /
0.531 edge-filtered vs 0.524, inside the "at breakeven" band) and its `line_clv` is
statistically indistinguishable from zero (mean -0.0007, p=0.994). Under the target-appropriate
rubric (CLV + calibration, hit-rate as context), a non-significant zero CLV with an
at-breakeven hit-rate is the **efficient-market ceiling** signature -- the model is not
systematically losing to the close, it simply is not finding a persistent spread edge. The
modest R2 (0.146) is consistent with an efficient spread market. Verdict: **ceiling, not a
flaw.**

**O/U -- [MIXED] (ceiling on hit-rate, real edge on CLV).** O/U is the one target with a
**statistically significant positive CLV** (raw line_clv +1.62084, p=1.6e-83; blended
+1.02185, p=5.2e-85) -- a genuine, robust edge against the closing total. Yet its raw hit-rate
is BELOW breakeven (0.485 vs 0.524). This apparent contradiction is the honest, important
finding: the O/U model's predicted totals move in the right direction relative to where the
line closes (positive CLV = the model anticipates closing-line movement), but that directional
edge does not (yet) convert into a winning straight bet rate after the bet is graded against
the line with slippage. This is NOT a calibration/methodology flaw in the WP sense -- a
significant positive CLV is the strongest single piece of edge evidence in the whole diagnosis
-- but it is also NOT a bankable hit-rate. Verdict: **CLV shows a real, significant edge
(future-promising); hit-rate is at/below the ceiling.** The CLV-vs-hit-rate divergence is
itself the headline O/U finding and the strongest argument that the edge is real but not yet
monetizable at the current edge threshold.

**Cross-target read:** none of the three targets shows a methodology FLAW under the rubric
(no significant negative CLV on the backtest models; WP well-calibrated; ATS/OU at/around
breakeven with O/U carrying a real CLV edge). The backtest models on canonical gold are at or
near the efficient-market ceiling, which is the honest and expected outcome for a public-data
pre-game model -- and is exactly why the milestone deliberately does NOT chase accuracy gains.

---

## DIAG-05 Production-vs-Backtest Mismatch

The deployed production artifacts are v1.0 **pre-Elo** models fit on the OLD (2026-03) gold.
The backtest models above are fit per-fold on the Phase-20 **rebuilt canonical gold**.
Production currently serves the old artifacts for live current-week predictions, while the
backtest measures the new-gold methodology. DIAG-05 quantifies that gap by LOADING the deployed
artifacts and SCORING them over the same 2021-2024 gold (inference only -- NO re-fit, verified:
0 missing features, 1139 games scored per target), then comparing to the backtest models. Both
streams are scored raw AND blended, yielding the clean 2x2 (D-08).

### The 2x2 -- WP

| Cut | accuracy | ECE | CLV (probability_clv) mean | CLV p-value | CLV verdict |
|-----|----------|-----|----------------------------|-------------|-------------|
| **raw-prod** (deployed) | 0.66023 | 0.05788 | -0.05668 | 9.7e-130 | systematically NEGATIVE |
| **blended-prod** (deployed) | -- | -- | -0.03399 | 4.2e-132 | systematically NEGATIVE |
| **raw-backtest** (new gold) | 0.66725 | 0.05499 | -0.00347 | 0.1869 | indistinguishable from zero |
| **blended-backtest** (new gold) | -- | -- | -0.00207 | 0.1907 | indistinguishable from zero |

### The 2x2 -- ATS (line_clv)

| Cut | line_clv mean | CLV p-value | straight-pick hit | CLV verdict |
|-----|---------------|-------------|-------------------|-------------|
| **raw-prod** (deployed) | -0.40523 | 1.0e-06 | 0.42502 | systematically NEGATIVE |
| **blended-prod** (deployed) | -0.22108 | 1.0e-06 | -- | systematically NEGATIVE |
| **raw-backtest** (new gold) | -0.00070 | 0.9940 | 0.53266 | indistinguishable from zero |
| **blended-backtest** (new gold) | -0.00038 | 0.9940 | -- | indistinguishable from zero |

### The 2x2 -- O/U (line_clv)

| Cut | line_clv mean | CLV p-value | straight-pick hit | CLV verdict |
|-----|---------------|-------------|-------------------|-------------|
| **raw-prod** (deployed) | +1.10954 | 1.2e-55 | 0.50230 | systematically POSITIVE |
| **blended-prod** (deployed) | +0.69832 | 1.4e-57 | -- | systematically POSITIVE |
| **raw-backtest** (new gold) | +1.62084 | 1.6e-83 | 0.48482 | systematically POSITIVE |
| **blended-backtest** (new gold) | +1.02185 | 5.2e-85 | -- | systematically POSITIVE |

### Per-target deltas (backtest minus production -- the cost of NOT re-fitting)

| Target | Metric | Deployed (prod) | Backtest (new gold) | Delta (backtest - prod) | Direction |
|--------|--------|-----------------|---------------------|-------------------------|-----------|
| WP  | accuracy | 0.66023 | 0.66725 | +0.00702 | new gold better |
| WP  | ECE | 0.05788 | 0.05499 | -0.00289 | new gold better (lower) |
| WP  | raw probability_clv | -0.05668 | -0.00347 | +0.05321 | new gold much better |
| ATS | raw line_clv | -0.40523 | -0.00070 | +0.40453 | new gold much better |
| ATS | straight-pick hit | 0.42502 | 0.53266 | +0.10764 | new gold much better |
| O/U | raw line_clv | +1.10954 | +1.62084 | +0.51130 | new gold better (larger edge) |
| O/U | straight-pick hit | 0.50230 | 0.48482 | -0.01748 | prod slightly better |

**Finding:** On every accuracy / CLV axis except the O/U raw hit-rate, the new-gold backtest
models beat the deployed v1.0 artifacts -- and on two axes the gap is decisive:

1. **WP CLV:** the deployed model is **significantly negative** (-0.05668, p~1e-130), i.e. it
   systematically prices WORSE than the closing line, whereas the new-gold backtest model is
   indistinguishable from zero (-0.00347, p=0.19). The deployed WP model is leaving CLV on the
   table that the new-gold methodology does not.
2. **ATS:** the deployed model is **significantly negative** on line_clv (-0.40523, p=1e-6)
   AND its straight-pick hit-rate is a dismal 0.425 (worse than a coin flip against the line),
   while the new-gold backtest model is at-breakeven (0.533, line_clv ~0). This is the single
   largest production-vs-backtest gap.
3. **O/U:** both carry a significant positive edge; the new-gold model's edge is LARGER
   (+1.62 vs +1.11 line_clv), though the deployed model's raw hit-rate is marginally higher.

The mismatch is real and material: production is serving models that are measurably worse
(and in the WP/ATS cases, significantly negative on CLV) than what the verified-correct
current data supports.

### Re-fit recommendation -- OWNER-APPROVED 2026-05-31 (recommendation only, NO re-fit performed)

> Per the milestone hard boundary (D-01) and the DIAG-05 scope (D-07), this is a
> RECOMMENDATION for a FUTURE milestone only. No deployed artifact was re-fit, no gold was
> rebuilt, and `scripts/train_models.py` was NOT invoked in this phase. The owner reviewed and
> APPROVED this recommendation as drafted (2026-05-31, v2.1 Phase 22) -- no refinements
> requested; it is adopted as the phase's standing recommendation.

**Recommendation: schedule a deployed-artifact re-fit on the canonical Phase-20 gold as a
focused future model-improvement milestone -- WITH per-target gating, NOT a full Optuna
re-tune.** Grounding:

- The 2x2 shows the deployed v1.0 pre-Elo artifacts are significantly worse on CLV for WP and
  ATS (both significantly negative in production vs indistinguishable-from-zero on new gold)
  and worse on WP accuracy and ATS hit-rate. The cost of NOT re-fitting is concrete: live
  Friday predictions currently run on models that price below the closing line on WP/ATS.
- This is a **re-fit on already-verified-correct gold** (Phase 20 proved the gold is correct
  and leakage-free), NOT new feature work or re-tuning -- so it is lower-risk than the v2.0
  attempt.
- IMPORTANT GATING CAVEAT (D-17 history): the v2.0 retrain attempt (Phases 11-12) FAILED
  per-target gating on all three targets, which is why the v1.0 pre-Elo artifacts were
  retained. Any future re-fit MUST go through the same per-target gating (both CLV and
  accuracy must pass before deployment); a re-fit is recommended but must NOT be deployed
  blindly. A full Optuna re-tune is specifically NOT advised (it is the v2.0 path that failed
  gating); the recommended scope is a straight re-fit of the existing architectures on the
  new gold, gated per target.
- O/U already carries a significant positive CLV on BOTH streams; the re-fit value there is
  marginal (larger edge but not a hit-rate fix), so WP and ATS are the priority targets.

In short: the data strongly justifies a gated re-fit of the deployed WP and ATS artifacts on
canonical gold as a future milestone; this phase records the evidence and the recommendation
but performs no re-fit.

---

## Cross-references

- **Re-runnable DIAG harness (D-09, D-12/D-13):** `backtest/diagnose.py` (`run_diagnosis`
  produces the 2x2, both-population hit-rates, and CLV significance reported here);
  guarded by `tests/integration/test_diag_diagnosis.py` (11 tests, green -- numbers
  reproducible).
- **Regenerated metrics + charts (DIAG-02/03 visual evidence):**
  `outputs/backtest/metrics_summary.json` and `outputs/backtest/backtest_report.html`
  (regenerated on canonical gold by plan 22-02; the WP reliability diagram at ECE 0.0550 and
  the cumulative CLV chart are owner-verified). Both are gitignored generated outputs,
  reproducible from committed code.
- **Phase-20 audit + post-rebuild anchors:** `AUDIT-REPORT.md` (repo root) -- the single
  adopted canonical-gold rebuild, the 2021-2024 BEFORE->AFTER metric-move table, and the
  explicit note that "the production-vs-backtest mismatch is DIAG-05's subject (Phase 22)."
- **Automation context:** `AUTOMATION.md` (repo root) -- the live blend behavior (D-19) and
  what the Friday orchestrator serves in production.
- **Pipeline:** `PIPELINE.md` (repo root) -- the canonical 7-stage run sequence.
- **Production-vs-backtest mismatch background:** `.planning/STATE.md` Blockers/Concerns
  (deployed = old-gold pre-Elo artifacts; the cache's 2021-2024 predictions are backtest
  models, not production artifacts) and PROJECT.md D-17 (per-target gating; v1.0 retained).

---

*Phase 22 -- Honest Accuracy Diagnosis. Authored 2026-05-31 (plan 22-03). Numbers reproducible
from `backtest/diagnose.py` + the regenerated `outputs/backtest/metrics_summary.json`, anchored
to the Phase-20 post-rebuild canonical-gold values in `AUDIT-REPORT.md`. HARD BOUNDARY held: no
deployed-artifact re-fit, no gold rebuild, no model-improvement -- DIAG-05 is a recommendation
only. ASCII only, no emoji (CLAUDE.md).*

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

What this covers in this document: its per-target accuracy, calibration and closing-line-value findings were all measured on the defective inputs, so the diagnosis describes a system that no longer exists.
