# O/U CLV-to-ROI Divergence Diagnosis (Phase 26, OUM-01)

**Status:** FULL closeout (owner chose PROCEED / run at the D26-19 interim checkpoint).
**Verdict:** PENDING -- to be recorded by the owner at the D26-13 final go/no-go checkpoint (see Section 8).
**Deployed artifact under diagnosis:** `ou_20260326_163930` (the RETAINED v1.0 O/U model, D25-14).
**Window:** 2021-2024 canonical Elo gold, single-pass over the deployed artifact (D26-01).
**Reproducibility:** Every load-bearing number below is reproducible from
`backtest/ou_divergence.py` via `run_ou_divergence_diagnosis()`. The doc-drift guard
`tests/unit/test_ou_divergence_diagnosis_md.py` runs the orchestrator and compares numbers
against this doc, so the doc cannot silently drift (D26-15).

This is a DIAGNOSIS, not a build. It answers the milestone's central question with evidence
before any monetization code is written: why does the deployed O/U model's strong, significant
line-CLV edge (pooled +1.11, p ~ 1e-55) sit on a below-breakeven straight-pick hit-rate? A
NO-GO here is a successful, defensible phase outcome (honest-refusal-as-deliverable, D25-14) --
the doc states the evidence plainly, no hype.

---

## 1. Integrity preamble (D26-04)

The diagnosis OPENS with an odds-integrity preamble. Two DISTINCT provenance flags are reported
SEPARATELY (the Codex HIGH terminology split -- the overloaded word "synthetic" must not conflate
the odds-contamination concept with the timestamp-fabrication concept):

    mock_free_odds        : YES  (every odds row sportsbook in {consensus, draftkings}; is_live all False)
    synthetic_snapshot_ts : YES  (8 distinct snapshot_ts; max 1 line per game -> freeze == closing in stored data)

- `mock_free_odds = YES` means there is NO mock/synthetic ODDS contamination. This is a HARD
  assertion in the harness -- it raises an actionable error naming offending game_ids if any odds
  row carries a sportsbook outside {consensus, draftkings} or `is_live = True`.
- `synthetic_snapshot_ts = YES` is a DATA-REALITY DISCLOSURE, not a contamination risk. The stored
  silver odds carry ONE synthetic-Friday line per game (8 distinct timestamps across the whole
  table, exactly 1 line per game), so in the stored data the freeze line EQUALS the closing line.
  A true freeze-vs-closing CLV delta is therefore UNMEASURABLE from stored snapshots. This is why
  the bias decomposition (Section 2) stands on the over-share / directional reading -- the
  "anticipation" component a bettor could capture cannot be measured from a single stored snapshot.
  (The absence of a pre-close line to anticipate from itself strengthens the bias reading.)

Deployed-artifact identity (resolved via `artifact_dir.name` / `artifacts/latest.json`, NEVER a
metadata version key -- the OU metadata.json has none):

    deployed_artifact : ou_20260326_163930   (the RETAINED v1.0, D25-14)

Coverage / exclusions (D26-04 iii/iv):

    n_total      : 1139
    n_with_line  : 1087   (the verdict population)
    n_excluded   : 52     (closing-snapshot missing)
    coverage     : 0.9543 (>= 0.80 floor -> FULL evidence, NOT downgraded to partial)

52-excluded characterization (selection-bias check):

    by_season    : 2021=13, 2022=13, 2023=13, 2024=13   (perfectly even across seasons -- no per-season skew)
    by_week_grp  : early=0, mid=0, late=0, playoffs=52   (ALL 52 excluded games are PLAYOFF games)
    excluded mean actual total : 49.31   (playoff games -- higher-scoring on average)
    included mean actual total : 44.77

The exclusion is a clean structural cut: every missing-closing-line game is a playoff game, split
evenly across the four seasons. The excluded playoff games DO have a higher actual total (49.31 vs
44.77), so the with-line population skews very slightly toward lower-scoring regular-season games.
This is disclosed, not hidden; it dovetails with the pre-registered playoffs-out cut (Section 3).

D26-20 data-correctness bug found by the preamble: **none.** The provenance assertion passed, the
synthetic-timestamp reality is a disclosed data fact (not a bug), coverage is above the floor, and
the 52 exclusions are cleanly characterized. No production touch this phase.

---

## 2. Bias-vs-anticipation decomposition (D26-05) -- the LEAD finding

The production O/U `line_clv = model_total - closing_total` is UNCONDITIONED on bet direction and
never references a freeze line. A model that systematically predicts HIGH totals scores positive
CLV in any season where scoring rose. That directional coupling is the fingerprint of a systematic
upward total bias rather than market-beating close-anticipation.

    pooled line_clv (deployed v1.0, raw)    : +1.1095   (t=16.66, p ~ 1.2e-55, 95% CI [+0.979, +1.240], n=1087)
    pooled model-picks-over share           : 0.7268    (the model picks OVER ~73% of the time)
    blended-stream pooled line_clv (D26-02) : +0.6983   (the live dynamic blend, O/U weight 0.604)
    raw +1.11  vs  blended +0.70            : the blend roughly halves the raw edge

### ASCII bias direction-map (model-over-share by season -- Gemini suggestion)

The bar length is the model's OVER-pick share that season; the line_clv sign follows it. 2021 is
the canary: the model picks UNDER ~82% of the time and line_clv flips NEGATIVE.

    season   n     over_share   line_clv   over-share bar (each # ~ 5%)
    2021     272   0.184        -1.213     ###                       <- SIGN-FLIPPED (picks UNDER ~82%)
    2022     271   0.875        +1.432     #################
    2023     272   0.949        +2.567     ###################
    2024     272   0.901        +1.653     ##################

Reading: in 2022-2024 the model picks OVER 87-95% of the time and line_clv is strongly positive; in
2021 it picked OVER only 18.4% and line_clv flipped to -1.21. The "+1.11 edge" tracks a systematic
upward total bias that aligned with rising scoring 2022-2024 -- NOT symmetric close-anticipation.
(Corroborating, D25-14: the Phase-25 re-fit candidate had BETTER MAE 8.6 vs 10.3 yet LOWER line_clv
+0.78 -- a more-accurate total predicts closer to market and earns less of this metric, which is
exactly what a bias metric does.) An over-bias aligned with a 3-season scoring rise is NOT a durable
edge -- flag the 2021 sign-flip prominently.

### Prior-season bias-adjusted re-score (D26-18, walk-forward estimation only)

Each 2022-2024 season's mean total bias is estimated on PRIOR seasons ONLY (walk-forward); the
deployed prediction copy has that bias subtracted for the season; line_clv is recomputed by CALLING
the production CLV function (no hand-roll). 2021 is excluded (no prior seasons).

    season   n     bias_subtracted   de-biased line_clv
    2022     271   -0.722            +2.154
    2023     272   +0.522            +2.045
    2024     272   +1.008            +0.645

    pooled de-biased line_clv (2022-2024) : +1.614
    pre-registered residual threshold     : 0.25 (abs below -> "nothing underneath")
    mechanical interpretation             : "residual anticipation"

The prior-season bias correction does NOT collapse the +1.11 to ~zero: the pooled de-biased
line_clv is +1.614, ABOVE the 0.25 "nothing underneath" threshold, so the mechanical pre-registered
interpretation reads "residual anticipation." A SIMPLE per-season scalar mean-bias correction leaves
a material residual. Whether that residual is genuine anticipation or an artifact of the crude
per-season-mean correction is the open question the extended sweep (Section 3) addresses. Note the
within-window trend: the de-biased line_clv shrinks (2022 +2.15 -> 2023 +2.05 -> 2024 +0.65) as the
estimated bias the prior seasons supply grows -- the most recent, best-corrected season is closest
to neutral but still positive.

---

## 3. Extended bucket sweep -- all cuts with coverage counts (D26-06)

The FULL extended cut set (owner SWEEP_DISPOSITION = run) is graded for BOTH the raw and the blended
stream via the LOCKED `BettingSimulator(min_edge_threshold=0.0)` (the D26-03 all-games base); the
slippage-survival cut uses the sanctioned `SimulationConfig(slippage_points=0.0)` vs the default
0.5 knob (no hand-rolled half-point). Coverage counts (n + n_excluded) accompany every bucket. The
RAW stream is shown below; the blended stream is graded identically (the survivable findings appear
in both -- see Section 4).

Graded hit-rate is the bettable signal (breakeven 0.5238 at -110); line_clv is supporting context.

    cut                bucket            n     graded hit
    over_under         over              790   0.4747      <- the dominant +1.11 over-bias slice, BELOW breakeven
    over_under         under             297   0.5758      <- the model's UNDER picks, ABOVE breakeven
    totals_regime      low (<42.0)       292   0.4760
    totals_regime      mid (42.0-46.5)   469   0.4840
    totals_regime      high (>46.5)      326   0.5521      <- ABOVE breakeven
    key_total_distance 0.0 (on-integer)  543   0.5304
    key_total_distance 0.5 (off-half)    544   0.4743
    key_total_distance 1.0               0     n/a (no totals land >=0.75 off an integer)
    key_total_distance >=1.5             0     n/a
    season             2021              272   0.5515
    season             2022              271   0.4539
    season             2023              272   0.4632
    season             2024              272   0.5404
    week_grouping      early (W1-6)      373   0.4799
    week_grouping      mid (W7-13)       404   0.5124
    week_grouping      late (W14-18)     310   0.5161
    week_grouping      playoffs (W19-22) 0     n/a (the 52 playoff games lack a closing line -- Section 1)
    playoffs_out       regular_season    1087  0.5023      (== "all"; the with-line population is all <= W18)
    slippage_survival  with 0.5 (default) 1087 0.5023
    slippage_survival  no slippage (0.0)  1087 0.5198      (the half-point erodes ~1.75 pts of graded hit)
    weather_outdoor    outdoor           UNAVAILABLE -- gold 'venue_outdoor' is not a clean 0/1 indicator
    weather_outdoor    severe_weather    UNAVAILABLE -- gold 'weather_severity_score' has <=1 distinct value

The weather/outdoor cut is honestly marked UNAVAILABLE (with a coverage note), NOT silently skipped
and NOT hand-substituted: the gold weather features (`venue_outdoor`, `weather_severity_score`) are
present but DEGENERATE (never populated in this gold). The key-total cut confirms the
research's pre-registered finding that NFL totals have WEAK key numbers -- the on-integer bucket
(0.5304) is only modestly above the off-half bucket (0.4743), with no bucket landing >=0.75 off an
integer at all. The points-edge does not concentrate at exploitable key totals.

### Edge-magnitude monotonicity (D26-03 decisive check)

Does graded hit-rate improve as the model-vs-line gap grows?

    min_edge (pts)   graded hit   n_bets
    0.0              0.5023       1087
    0.5              0.5052       964
    1.0              0.5036       824
    1.5              0.5046       658
    2.0              0.5268       467
    3.0              0.5579       190

    monotone_improving : FALSE

NOT monotone overall (flat ~0.50 from 0.0 through 1.5), but it improves at the largest gaps
(0.5268 at 2.0, 0.5579 at 3.0 on n=190). The points-edge does not concentrate cleanly with gap size
except at the extreme tail (and the tail's small n is itself a caution).

---

## 4. Trial registry + corrected significance (D26-10)

The trial registry IS the multiple-comparisons denominator: ONE entry per (stream x cut x bucket),
logged whether or not it is testable. BH-FDR (`scipy.stats.false_discovery_control`, method "bh") is
computed across the registry's non-None raw p-values; the adjusted p is attached back aligned by
index.

    n_trials (testable, BH-FDR denominator) : 36
    total registry entries                  : 46   (incl. 4 unavailable weather + 6 insufficient-sample)

### Survivable sub-populations (D26-11 structural bar)

A sub-population is NAMED "survivable" ONLY if it clears ALL of: BH-adjusted p < 0.05, N >= 175, AND
the GRADED EDGE is in the PROFITABLE direction (graded hit-rate ABOVE the 0.5238 breakeven) in >= 3
of the 4 seasons individually. The per-season direction metric is GRADED-EDGE direction, NOT
line_clv sign (the Codex MED pre-registration). A consistently-below-breakeven slice is a consistent
NON-edge and is classified "suggestive_not_survivable" no matter how significant its line_clv.

    stream   cut             bucket   N     graded hit   BH-adj p     direction
    raw      over_under      under    297   0.5758       5.2e-47      [1, 1, 1, 1]
    raw      totals_regime   high     326   0.5521       0.0031       [1, 1, 1, 1]
    blended  over_under      under    297   0.5758       6.7e-48      [1, 1, 1, 1]
    blended  totals_regime   high     326   0.5521       0.0070       [1, 1, 1, 1]

Four sub-populations clear all three structural-bar conditions: the model's UNDER picks and
high-total (>46.5) games, in BOTH the raw and the blended stream, ABOVE breakeven in ALL FOUR
seasons. This is the INVERSE of the +1.11 over-bias story.

### The dominant over-bias slice is NOT survivable

    stream   cut          bucket   N     graded hit   BH-adj p     direction        classification
    raw      over_under   over     790   0.4747       3.3e-212     [1, -1, -1, 1]   suggestive_not_survivable

The over-bias slice that produces the headline +1.11 line_clv is itself CONSISTENTLY BELOW breakeven
(graded hit 0.4747) and points different directions across seasons ([1, -1, -1, 1]). Its line_clv is
hugely "significant" (BH-adj p ~ 3e-212) yet it is a consistently-LOSING bet -- the exact
CLV-positive / ROI-negative trap this phase exists to diagnose (Pitfall 4). It is correctly
classified `suggestive_not_survivable`.

---

## 5. Throwaway EV preview (D26-07/16) -- EXPLORATORY

This preview is clearly labeled EXPLORATORY and is NEVER imported by production code (Phase 27 builds
the real EV chain from scratch with an empirically-locked residual SD, OUM-02). It fits a residual SD
in-harness, applies a side-specific normal-approximation cover probability with the half-point
applied AGAINST the bet side (over uses closing+0.5, under uses closing-0.5), and devigs flat -110.

EV preview on the survivable `over_under = under` sub-population (n=297, devig method flat_-110,
breakeven 0.5238 cover prob):

    sd point   mean p_side   per-bet EV   >= breakeven?
    12.5       0.5347        +0.0208      YES
    13.0       0.5334        +0.0184      YES
    13.5       0.5322        +0.0161      YES
    14.0       0.5311        +0.0139      YES
    14.5       0.5301        +0.0119      YES
    fit 12.74  0.5341        +0.0196      YES   (the in-harness residual-SD fit)

The throwaway EV is POSITIVE across the ENTIRE pre-registered 12.5-14.5 SD sensitivity band (every
grid point above breakeven), with the in-harness fit SD (12.74) centered in-band. This would meet
the D26-16 EV clearance criterion. BUT it is exploratory (flat -110, normal approximation, no real
over/under juice in silver), and it must be weighed against the burned-holdout caveat below and the
bias reading above before naming a survivable edge.

---

## 6. Burned-holdout caveat (D26-09)

The owner explicitly chose selection POWER over split PURITY: this diagnosis explored all of
2021-2024 freely. Consequently, **Phase 27's 2023-2024 holdout (OUM-03) is PARTIALLY BURNED by this
free exploration.** The four survivable sub-populations named in Section 4 were found on data that
overlaps Phase 27's intended holdout. Any Phase-27 monetization that proceeds on these named
sub-populations must treat the 2023-2024 numbers as in-sample-contaminated and re-establish a clean
out-of-sample estimate (a fresh holdout window, or the Phase-30 widened-gold re-fit's holdout). This
caveat is part of the go bar's evidence, not a footnote.

---

## 7. The pre-registered go bar (D26-13/16) -- VERBATIM

The go bar below is copied VERBATIM from Plan 26-03 (written BEFORE the analysis ran -- the
forking-paths / goalpost-moving guard). Evidence is weighed against THIS bar, not a post-hoc one.

> ## Pre-Registered Go Bar (D26-13/16 -- written BEFORE the analysis, recorded verbatim by Plan 26-04)
>
> The sweep produces the evidence that Plan 26-04 weighs against this bar. NAMING a sub-population as carrying a survivable edge (a GO or SCOPED GO) requires ALL THREE:
> 1. Corrected significance: the sub-population's BH-FDR-adjusted p < 0.05 across the full trial registry denominator (raw-only p does not qualify).
> 2. Structural bar (D26-11): N >= 175 graded bets across 2021-2024 AND the GRADED EDGE points the same direction in >= 3 of the 4 seasons individually. The PER-SEASON DIRECTION METRIC is GRADED EDGE vs breakeven (the season's graded ROI sign, equivalently hit-rate above/below the 0.5238 breakeven at -110) -- NOT line_clv sign. (Pre-registered here per the Codex MED fix: ROI/EV direction leads because the phase goal is ROI divergence; a CLV-only direction can name a false edge.)
> 3. EV clearance (D26-16): the throwaway-EV estimate is positive under base assumptions (-110 pricing, half-point slippage applied against the bet side) AND remains >= breakeven (0.5238 cover prob at -110) across the entire SD sensitivity band 12.5-14.5. +EV only at the optimistic end of the band -> SCOPED GO at best.
>
> If no sub-population clears all three -> the edge is "real but unpriceable at half-point" -> NO-GO recommendation.

### Evidence weighed against each bar criterion

| # | Criterion | Evidence | Meets? |
|---|-----------|----------|--------|
| 1 | Corrected significance (BH-adj p < 0.05) | `over_under=under` BH-adj p = 5.2e-47 (raw) / 6.7e-48 (blended); `totals_regime=high` BH-adj p = 0.0031 (raw) / 0.0070 (blended) -- all < 0.05 across the 36-trial denominator | MET (for the under / high-total sub-populations, both streams) |
| 2 | Structural bar (N >= 175 AND profitable direction 3/4 seasons) | `over_under=under` N=297, profitable direction [1,1,1,1] (4/4); `totals_regime=high` N=326, [1,1,1,1] (4/4) -- both clear N>=175 and 4/4 PROFITABLE seasons | MET (for the under / high-total sub-populations) |
| 3 | EV clearance (+EV at base AND >= breakeven across the full 12.5-14.5 band) | `over_under=under` throwaway EV +0.0208 (sd 12.5) to +0.0119 (sd 14.5), mean p_side 0.530-0.535 > 0.5238 at EVERY grid point; in-harness fit +0.0196 | MET on the EXPLORATORY preview, across the full band (so SCOPED GO is not capped to "optimistic-end only") |

### Real-but-unpriceable vs real-for-a-sub-population

- The model's DOMINANT behavior -- the systematic OVER bias that produces the headline +1.11
  line_clv -- is **real but unpriceable**: the `over_under=over` slice (n=790) grades 0.4747, BELOW
  breakeven, with an inconsistent per-season direction [1,-1,-1,1]. Its enormous line_clv
  significance is the CLV-positive / ROI-negative trap. As a bet, the over bias LOSES money.
- A specific **sub-population carries a survivable graded edge**: the model's UNDER picks
  (`over_under=under`, n=297, graded 0.5758, 4/4 seasons) and high-total games
  (`totals_regime=high`, n=326, graded 0.5521, 4/4 seasons), in both the raw and blended streams.
  These clear all three pre-registered criteria on the explored data.

The named survivable sub-population (the model's UNDER picks / high-total games) is the candidate a
GO/SCOPED-GO would rest on. The decisive open questions for the owner: (a) the survivable numbers
sit on a PARTIALLY BURNED holdout (Section 6); (b) the EV clearance rests on an EXPLORATORY flat-110
normal-approximation preview, not a real devig; and (c) the "under picks win" finding is the inverse
of the model's dominant over-bias, so it is a narrow, possibly fragile slice rather than a broad
stream edge. Honest mechanical harness recommendation: **SCOPED GO** (a restricted sub-population
clears all three pre-registered criteria), with NO-GO fully defensible if the owner judges the
burned-holdout + exploratory-EV caveats to outweigh the in-sample survival. The owner rules at
Section 8.

---

## 8. Verdict (D26-13/14) -- TO BE RECORDED BY THE OWNER

> PENDING. This section is filled at the D26-13 final owner-attended checkpoint. The owner reads
> Sections 1-7, weighs the evidence against the pre-registered go bar (Section 7), and rules. The
> agent records the ruling verbatim here -- it does NOT decide the verdict.

**Verdict:** _PENDING -- owner decision at the D26-13 checkpoint._

**Basis:** _PENDING -- full sweep (the owner chose PROCEED / run at the D26-19 interim checkpoint,
so this closeout rests on the FULL extended sweep, not an interim early exit)._

**Rationale:** _PENDING -- owner's written rationale._

**Verdict semantics (D26-14), for reference when recording:**
- **GO** -- Phase 27 proceeds as roadmapped for the named population/stream.
- **SCOPED GO** -- Phase 27 proceeds RESTRICTED to the named sub-population; that restriction is
  recorded as a LOCKED input to Phase 27 planning. (The harness-named candidate is the model's
  UNDER picks / high-total games, both streams.)
- **NO-GO** -- no monetization code is built; this doc records the verdict + a RECOMMENDED redirect
  (the actual roadmap change is a SEPARATE owner decision through the normal planning flow -- this
  phase does NOT perform scope surgery on the milestone). On a NO-GO, the Researcher Observation
  paragraph below is completed to seed Phase 28/29.

**Researcher Observation (completed ONLY on a NO-GO, Gemini suggestion -- analysis-layer observation
only, no new feature work this phase):** _PENDING -- on a NO-GO, note here whether any simple raw
signal (e.g. the bet-direction asymmetry, the totals-regime split, or the residual structure)
correlates with the residuals in a way that could seed a Phase 28/29 signal. The most salient
candidate from this diagnosis: the model's UNDER picks / high-total games grade above breakeven 4/4
seasons while its dominant OVER picks lose -- a directional asymmetry worth a leakage-safe re-look._

---

*Phase: 26 -- O/U CLV-to-ROI Divergence Diagnosis (Go/No-Go)*
*Authored: 2026-06-07. Every number reproducible from `backtest/ou_divergence.py`
`run_ou_divergence_diagnosis()`; guarded by `tests/unit/test_ou_divergence_diagnosis_md.py`.*
