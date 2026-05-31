# METHODOLOGY.md -- Modeling & Feature-Engineering Deep-Dive (Phase 23)

**Milestone:** v2.1 Trust & Reproducibility
**Phase:** 23 -- Documentation, Runbook & State-of-System
**Authored:** 2026-05-31 (plan 23-03)

> This is the portfolio-facing deep-dive into the system's modeling and feature-
> engineering methodology. It is the consolidated, de-staled successor to the two
> v1.0-era reference docs (`docs/model_methodology.md` + `docs/feature_engineering.md`,
> both deleted in this plan). It deliberately CROSS-LINKS the current code that
> establishes ground truth rather than re-asserting ~1,274 lines of v1.0 prose that
> has drifted past Elo activation (Phase 11), Optuna tuning (Phase 12), and dynamic
> blending (Phase 13). Where a number or a per-target verdict matters, it points at
> the forensic sources (`MODEL-DIAGNOSIS.md`, `AUDIT-REPORT.md`) instead of restating
> them, so this file cannot drift from the harness-reproducible truth.
>
> Read the "Drift landmines / what changed since v1.0" section first if you last read
> the v1.0 docs -- four things changed materially.
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

Note on counts: the v1.0 feature doc quoted "89 features (31/29/29)". Those counts
predate the Phase-20 canonical-gold rebuild -- current gold widths are 156/157/156
(WP/ATS/O-U) over 6263 rows / 2002-2025. Treat the code and the gold schema as ground
truth, not any quoted count. See `AUDIT-REPORT.md` for the rebuilt schema.

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
  result. This produces 6263 per-game snapshots.
- **2002 burn-in** -- ratings warm up from 2002 (16 seasons before the 2018 backtest
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

Per-target gating decides what actually ships: a retrained model must beat the
incumbent on both CLV and accuracy before it is deployed. In v2.0 the retrained models
did not pass gating on any target, so production still serves the v1.0 pre-Elo
artifacts (see landmine 3 below and `MODEL-DIAGNOSIS.md` DIAG-05).

## 6. Market blending

The model output is blended with the market line. Ground truth: `models/blending.py`
(`BlendWeights`, `DynamicBlendWeights`, `MarketBlender`) + `models/blending_data.py`
(`TUNING_SEASONS`).

- **WP blends in log-odds space** (probability-correct interpolation); **ATS/O-U blend
  linearly** on the line.
- **Blend weights are tuned on pre-2018 data only** (`TUNING_SEASONS`), strictly
  isolated from the 2018+ backtest window -- no information leakage from the evaluation
  period into the weights.
- **Dynamic (week-of-season) blending is gated per target (D-19).** Only the O/U
  dynamic blend is ADOPTED (it beat static CLV); WP and ATS stay on static weights.
  `mode_by_target` is persisted in the blend artifact.

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
3. **Production serves the v1.0 pre-Elo artifacts, not the v2.0 retrained models.**
   Per-target gating (D-17) rejected the v2.0 retrained models on all three targets,
   so the deployed artifacts are still the v1.0 ones. Do NOT assume the v2.0 models are
   live.
4. **Dynamic blend is gated per target (only O/U adopted).** WP and ATS use static
   blend weights; only O/U uses the dynamic week-of-season blend (D-19). The v1.0 docs
   describe no dynamic blend at all.

The deployed-v1.0 population and the new-canonical-gold backtest population are
DISTINCT. `MODEL-DIAGNOSIS.md` quantifies that production-vs-backtest mismatch (DIAG-05)
and carries the gated-re-fit recommendation. Read its numbers there; do not conflate
the two populations.

## 9. Deferred refactor (honest note)

The ATS/O-U residual/total distribution converters still live in the legacy
`models/train_ats.py` / `models/train_ou.py` modules alongside the newer
`models/trainers/` package (README "Current Limitations" item 7). This coexistence is a
catalogued, deferred refactor -- not a bug. It is recorded in `STATE-OF-SYSTEM.md`.

## Cross-references

- **`MODEL-DIAGNOSIS.md`** -- the Phase 22 honest accuracy diagnosis: per-target
  accuracy/calibration/CLV, the ceiling-vs-flaw verdict, and the production-vs-backtest
  mismatch (DIAG-05). The source for every quantified claim referenced above.
- **`AUDIT-REPORT.md`** -- the Phase 20 data & feature correctness audit: canonical-gold
  schema, the leakage/temporal verdict, and the F-* catalogue.
- **`README.md`** -- the portfolio front door and the single architecture diagram.
- **`PIPELINE.md`** -- the canonical run sequence that produces the gold, artifacts,
  and predictions this methodology describes.
