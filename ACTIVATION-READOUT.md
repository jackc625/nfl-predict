# ACTIVATION-READOUT.md -- Phase 25 Gated Re-fit Activation (ACTV-06)

**Milestone:** v3.0 Accuracy & Profitability
**Phase:** 25 -- Activate the Gains (Gated WP/ATS Re-fit)
**Authored:** 2026-06-06
**Status:** committed (repo root, sibling of `MODEL-DIAGNOSIS.md`)

> This is the honest, point-in-time record of the FIRST real production swap through the
> hardened per-target deploy gate. It is the activation-time counterpart to the frozen v2.1
> `MODEL-DIAGNOSIS.md` (which stays unchanged). It records what actually deployed, what was
> refused, the per-target CLV before and after with BOTH the deploy verdict and the bettable
> verdict, the floor-policy revision, the diagnosis reconciliation, the calibration ruling,
> the blend-mode change, and the pre-swap manifest state for one-command rollback.
>
> NO HYPE, NO BETTING FRAMING. Removing a closing-line-value (CLV) leak is NOT the same thing
> as having a positive market edge; this readout states that plainly and keeps the two bars
> (the deploy bar and the bettable bar) separate throughout.
>
> ASCII only (no emoji, per CLAUDE.md). Arrows are `->`, dashes are `--`, quotes are straight.

---

## 1. Headline: what deployed, what was refused

The owner-attended armed `--promote` run (D25-18) plus one documented fix-cycle per failing
target (D25-05) produced this outcome. The gate is the safety rail; no threshold,
`floor_mode`, calibration band, or frozen baseline value was relaxed to ship anything.

| Target | Outcome | Deployed version | Why |
|--------|---------|------------------|-----|
| **WP** | **ACTIVATED** (armed run) | `wp_20260605_215552` | Re-fit on canonical Elo gold; non-regression PASS (paired delta +0.0124 vs v1.0, p=1.7e-6); calibration within the D25-02 noise band. |
| **ATS** | **ACTIVATED** (fix-cycle) | `ats_20260605_220128` | Re-fit; the original candidate failed the season-2023 per-season floor; one documented fix-cycle (widened feature-selection window 2018-2019 -> 2015-2019) removed that regression; non-regression PASS all seasons. |
| **O/U** | **RETAINED on v1.0** (honest refusal) | `ou_20260326_163930` | Re-fit FAILED the gate (pooled + seasons 2022/2023 significantly worse than v1.0); one fix-cycle attempted, still FAILED; v1.0's stronger line-CLV edge is protected (D25-14). |
| blend | re-validated | `blend_dynamic_20260606_020635` | Modes re-validated to all-dynamic post-swap; weights UNCHANGED (no re-tune). |

**2 of 3 gated targets activated (WP + ATS). O/U retained on v1.0.** The rail behaved exactly
as designed: it activated the leak-reducing WP and the fix-cycle-improved ATS, and refused the
O/U re-fit that would have regressed a strong positive incumbent edge.

---

## 2. The no-edge statement (Success Criterion 2, Pitfall 3)

Removing a CLV leak is **not a positive WP/ATS market edge** and must not be sized as one.

Two of the three activated/measured candidates carry a NEGATIVE absolute CLV (WP -0.0443;
ATS -0.0685). The re-fit made them significantly LESS negative than the deployed v1.0 -- that
is a genuine, paired, statistically-significant improvement on the SAME population -- but
"less negative than v1.0" is not "beats the closing line." The deploy decision is
NON-REGRESSION vs the frozen v1.0 baseline; it is NOT the bettable bar. The bettable bar is
absolute-vs-zero, and on that bar WP and ATS remain below zero. Do NOT interpret this
activation as a WP/ATS betting edge; sizing WP/ATS as if the re-fit created a market edge is
explicitly out of scope (REQUIREMENTS.md Out-of-Scope). The honest reading is: the deployed
models are better-calibrated, leak-reduced successors to v1.0 -- not profitable market-beaters.

---

## 3. Per-target CLV before/after -- BOTH verdicts (Pitfall 3)

The **deploy bar** is non-regression vs the frozen v1.0 baseline (paired per-game
candidate-minus-baseline delta, pooled AND per-season). The **bettable bar** is
absolute-vs-zero on the candidate's own CLV. Both are recorded for every target; only the
deploy bar gates.

### WP -- ACTIVATED

- **Deployed CLV (absolute, candidate):** pooled `probability_clv` mean = **-0.04430**
  (t=-14.82, p=2.2e-45, n=1087).
- **Prior v1.0 CLV (absolute):** pooled mean = **-0.05668** (t=-27.92, p~1e-130, n=1087).
- **Paired delta (candidate - v1.0, merge-on-game_id):** **+0.01238** (t=+4.81, p=1.7e-6,
  n=1087) -- the re-fit significantly REDUCES the leak vs v1.0.
- **Per-season CLV (candidate, absolute):** 2021 -0.0500 (n=272) | 2022 -0.0265 (n=271) |
  2023 -0.0463 (n=272) | 2024 -0.0543 (n=272).
- **Deploy verdict:** PASS -- pooled non-regression PASS, per-season non-regression PASS (all
  four holdout seasons), accuracy delta +0.0086 (within 0.01), ECE delta vs frozen baseline
  within 0.0125, Brier delta within 0.0030.
- **Bettable verdict:** absolute pooled CLV -0.0443 is NEGATIVE -> NOT a WP market edge.

### ATS -- ACTIVATED (via one fix-cycle)

- **Deployed CLV (absolute, fix-cycle candidate):** pooled `line_clv` mean = **-0.06845**
  (t=-0.55, p=0.5839, n ~1067 valid).
- **Prior v1.0 CLV (absolute):** pooled mean = **-0.40523** (t=-4.91, p=1.0e-6).
- **Per-season CLV (fix-cycle candidate, absolute):** 2021 +0.3518 | 2022 -0.4194 |
  2023 -0.6799 | 2024 +0.4723.
- **Fix-cycle delta:** the ORIGINAL straight re-fit failed because season-2023 CLV (-1.4176)
  was significantly worse than v1.0's -0.7735. Widening the feature-selection train window
  2018-2019 -> 2015-2019 (a candidate-side modeling choice; NOT a gate change, NOT an
  Optuna/seed sweep) selected a more stable feature set; the fix-cycle candidate's 2023 CLV
  -0.6799 is no longer significantly worse than v1.0's -0.7735.
- **Secondary:** MAE 8.4571 vs v1.0 9.4807 (delta -1.02, within floor).
- **Deploy verdict:** PASS -- pooled + all-season non-regression PASS after the fix-cycle.
- **Bettable verdict:** absolute pooled CLV -0.0685 (p=0.5839) is NEGATIVE and not significant
  -> NOT an ATS market edge.

### O/U -- RETAINED on v1.0 (honest refusal, D25-14)

- **Deployed (retained) CLV (absolute, v1.0):** pooled `line_clv` mean = **+1.10954**
  (t=16.66, p~1e-55) -- v1.0 O/U carries a strong, significant positive line-CLV edge.
- **Candidate CLV after one fix-cycle (absolute):** pooled mean = **+0.7761** (p=0.0000) --
  positive and significant, but SIGNIFICANTLY WORSE than v1.0's +1.11.
- **Per-season (fix-cycle candidate):** 2021 -1.1323 | 2022 +0.5808 (FAIL vs v1.0 +1.4318) |
  2023 +1.5106 (FAIL vs v1.0 +2.5674) | 2024 +2.1445. Seasons 2022 + 2023 fail the
  non-regression floor.
- **Why the more-accurate re-fit STILL fails:** the candidate's MAE is much better (8.60 vs
  v1.0 10.31), and that is exactly the structural problem -- a more accurate O/U regressor
  predicts totals CLOSER to the market, so its `line_clv` (`model_total - closing_total`)
  diverges LESS from the close than v1.0's, regressing the +1.11 edge the non-regression floor
  protects (D25-01: non-regression is STRICTER where v1.0 is positive). Closing the gap would
  require deliberately degrading accuracy to chase CLV -- which would be gate-gaming.
- **Deploy verdict:** FAIL after one fix-cycle -> v1.0 O/U RETAINED. The rail protecting the
  bigger live O/U edge IS the deliverable.
- **Bettable verdict:** the candidate IS positive-and-significant on absolute CLV, but it is a
  REGRESSION vs the deployed v1.0's larger edge, so it must not swap.

---

## 4. The deployed / retained 2x2

|        | Deployed = NEW re-fit (canonical Elo gold) | Retained = v1.0 pre-Elo |
|--------|--------------------------------------------|--------------------------|
| **Passed the gate** | WP (`wp_20260605_215552`), ATS (`ats_20260605_220128`, via fix-cycle) | -- |
| **Failed the gate** | -- | O/U (`ou_20260326_163930`) |

WP and ATS swapped to re-fits trained on the canonical Phase-20 Elo gold. O/U stayed on the
v1.0 pre-Elo artifact. The blend pointer was re-validated (Section 7).

---

## 5. The WR-01 / D25-01 floor-policy revision (recorded)

The Phase-25 gate floor was revised from **absolute-vs-zero** to **non-regression vs the
frozen v1.0 baseline** (a deliberate, reviewed `config/gate.toml` edit with the rationale in
git history, NOT a silent loosening):

- **What changed:** a candidate now FAILS a slice (per-season or pooled) ONLY when it is
  SIGNIFICANTLY WORSE than v1.0 for that slice (a paired per-game CLV delta, read on the
  negative tail: mean < 0 AND p < alpha).
- **Why:** for a replace-or-retain decision, an absolute floor can keep a significantly-WORSE
  incumbent in production over a technicality (v1.0 WP serves -0.0567 while the leak-free
  -0.0443 candidate would be blocked by an absolute-vs-zero floor). Non-regression is the
  honest replace-or-retain test.
- **The consequence, stated plainly:** non-regression is STRICTER where v1.0 is already
  positive (O/U pooled +1.11 cannot be regressed -- which is exactly why the O/U re-fit was
  refused) and ACHIEVABLE where v1.0 is itself sub-floor (WP/ATS, where v1.0 is negative, so a
  leak-reducing candidate can clear the bar without being positive in absolute terms).
- **The absolute-vs-zero verdict is still computed and recorded** on every candidate (the
  bettable-bar input for the O/U monetization phases) -- it is a readout, never the deploy
  decision (Section 3 keeps both).

---

## 6. The D25-04 diagnosis reconciliation + the WP calibration ruling

### Reconciling DIAG-05's "~zero" with the measured candidate WP CLV -0.0443 (two populations)

DIAG-05 (`MODEL-DIAGNOSIS.md`) observed that on the new Elo gold a leak-free model lands near
~zero, yet the candidate measured -0.0443, not ~zero. The reconciliation is that these are TWO
DIFFERENT MODEL POPULATIONS:

- **Deployed-artifact single-pass scoring** loads ONE fixed artifact and predicts over all of
  2021-2024 in a single pass. This is what the deploy GATE and `freeze_gate_baseline.py` score,
  and it is where the candidate's HONEST single-artifact CLV is **-0.0443** (it reproduces the
  Phase-24 demo number exactly).
- **Walk-forward backtest engine** fits a FRESH model per fold (expanding window), so each
  holdout season is scored by a model trained only on prior seasons. This is what the web cache
  `predictions` table serves, and it is the population DIAG-05's qualitative "leak-free ~zero"
  statement was about.

The candidate is NOT ~zero on the deployed-artifact population; it is significantly BETTER than
v1.0 on the SAME population (paired delta +0.01238, p=1.7e-6). The "~zero" expectation conflated
the two populations. The gate's deploy decision is made on the paired single-artifact delta
(PASS); the candidate's absolute single-artifact CLV (-0.0443) stays negative (the bettable
bar). Both numbers are in this readout (Section 3).

### WP calibration ruling (D25-02 -- within the bootstrap noise band)

The candidate WP accuracy is UP (+0.0101 vs v1.0) and its ECE/Brier are slightly up (+0.0068 /
+0.0019). A paired bootstrap (B=2000, resampling the same 1087 games for both models) found
BOTH calibration deltas statistically indistinguishable from zero:

| Metric | Observed delta (cand - v1.0) | 95% CI of the delta | Zero in CI? |
|--------|------------------------------|---------------------|-------------|
| ECE    | +0.00680                     | [-0.01209, +0.03471]| YES         |
| Brier  | +0.00190                     | [-0.00313, +0.00691]| YES         |

**Ruling:** WITHIN NOISE -> a documented tolerance, NOT a regression and NOT a blind loosening.
The bands written into `config/gate.toml` sit at ~1 bootstrap-std of each delta
(`wp_ece_max_increase = 0.0125`, `wp_brier_max_increase = 0.0030`): wide enough to admit the
observed sub-noise jitter, still narrow enough to REJECT a genuine 2+ std calibration
regression. WP deployed under this band (no calibration fix-cycle was needed because the delta
is sub-noise).

---

## 7. Blend-mode change (D25-08) -- including the blend-pointer rewrite

After the WP + ATS swaps, the blend modes were re-validated via the D-19 mode-gating path
(`backtest.tune --compare`) against the NEWLY-deployed artifacts. Weights were NOT re-tuned
(weight re-tuning is out of scope, ADVM backlog) -- only the per-target dynamic-vs-static MODE
was re-validated.

- **Re-validation result (per-target, against the deployed artifacts):** WP static=-0.0021 /
  dynamic=-0.0021 (matched, PASS); ATS static=-0.0004 / dynamic=-0.0004 (matched, PASS); O/U
  static=0.9791 / dynamic=1.0219 (dynamic +4.4%, PASS). All three gate PASS -> each adopts
  dynamic (Phase-13 rule).
- **mode_by_target CHANGED:** `{wp: static, ats: static, ou: dynamic}` ->
  `{wp: dynamic, ats: dynamic, ou: dynamic}` (WP + ATS flipped static -> dynamic because the
  re-validation found dynamic matches/improves against the newly-deployed candidates; O/U stays
  dynamic). This mode change is the legitimate consequence of activating new WP/ATS artifacts.
- **Blend pointer REWRITTEN:** because `any_passed` was True, `run_comparison` called
  `models.blending.save_blend_artifacts`, which re-saved the blend dir AND rewrote
  `latest.json['blend']` via `_atomic_write_json` -- the ONE sanctioned `latest.json` write
  outside the promotion swap (it has its own writer for the blend key). Pointer moved
  `blend_dynamic_20260526_194510` -> **`blend_dynamic_20260606_020635`**.
- **weights UNCHANGED:** wp=0.5917, ats=0.5456, ou=0.6040 (identical to before -- only MODE was
  re-validated). The per-target version swaps survived the blend rewrite intact.

---

## 8. Pre-swap manifest state (D25-17 rollback record)

Captured BEFORE any swap via stdlib `hashlib` only (never a hand-rolled hash). The prior
artifact dirs are all retained (never deleted), so rollback is a pure manifest restore.

- **PRE-SWAP `artifacts/latest.json` sha256:**
  `14f9093edab0e7cffef968621b5d73f29534082f23add4253c6b61e8fc8b2beb`
- **PRE-SWAP full content (verbatim):**

```json
{
  "wp": "wp_20260327_114739",
  "ats": "ats_20260326_163724",
  "ou": "ou_20260326_163930",
  "blend": "blend_dynamic_20260526_194510"
}
```

- **PRE-SWAP per-target version map (the rollback pointers):**
  - `wp`  -> `wp_20260327_114739`  (v1.0 pre-Elo)
  - `ats` -> `ats_20260326_163724` (v1.0 pre-Elo)
  - `ou`  -> `ou_20260326_163930`  (v1.0 pre-Elo)
  - `blend` -> `blend_dynamic_20260526_194510` (the D-19 dynamic blend pointer)

- **POST-SWAP + POST-COMPARE `artifacts/latest.json` sha256 (FINAL):**
  `9139e748b5c37167a048b08bba188c74beedbf71309d3ab37cbdd7ca359ebd11`
- **FINAL full content (verbatim):**

```json
{
  "wp": "wp_20260605_215552",
  "ats": "ats_20260605_220128",
  "ou": "ou_20260326_163930",
  "blend": "blend_dynamic_20260606_020635"
}
```

- **ROLLBACK (one operation, D25-17):** restore the pre-swap mapping above via
  `models.artifacts.update_manifest` per key (`wp=wp_20260327_114739`,
  `ats=ats_20260326_163724`, `ou=ou_20260326_163930`, `blend=blend_dynamic_20260526_194510`);
  verify the restored manifest against the pre-swap sha256 `14f9093e...2beb`. See RUNBOOK.md
  "Rollback (reverse a Promote)" for the exact command. The rollback is covered by the
  committed `tests/integration/test_rollback.py` (parsed-equality + recorded-seed sha256 +
  atomicity guard).

---

## 9. The baseline re-freeze (D25-11)

With WP and ATS activated, the gate baseline in `config/gate.toml` was re-frozen against the
deployed incumbent via `scripts/freeze_gate_baseline.py` (owner-reviewed, deterministic
block-paste; D24-07 anti-typo). WP and ATS baseline values now reflect the activated re-fits
(WP pooled -0.04430 / acc 0.66637; ATS pooled -0.06845 / mae 8.45712); the O/U baseline is
byte-identical (retained target). The prior v1.0 values are preserved in git history. This
gives Phases 28 (SIG-05 "lift over the post-activation baseline") and 30 a correct judge, and
clears the previously-expected `test_frozen_baseline_matches_rescore` drift signal and the WP
drift-tripwire abort.

---

## 10. What this activation is, and is not

- **It IS:** an honest, gated, reproducible production swap -- WP and ATS replaced with
  leak-reduced, better-calibrated re-fits on canonical Elo gold; O/U honestly retained on v1.0
  because the re-fit would regress its stronger edge; the blend modes re-validated; the gate
  baseline re-frozen; rollback documented and tested.
- **It is NOT:** a betting edge. Two of the three measured candidates carry a negative absolute
  CLV; "less negative than v1.0" is the deploy bar, not the bettable bar. There is no positive
  WP/ATS market edge here to size.

## Cross-references

- **`MODEL-DIAGNOSIS.md`** -- the frozen v2.1 point-in-time accuracy diagnosis (DIAG-05) this
  activation executed; it stays unchanged.
- **`config/gate.toml`** -- the frozen per-target deploy gate + the re-frozen post-activation
  baseline (D25-11).
- **`RUNBOOK.md`** -- the Promote and Rollback operations + the clean-checkout bootstrap.
- **`PIPELINE.md`** -- the 8-stage canonical sequence with the Promote stage.
- **`STATE-OF-SYSTEM.md`** -- the registry recording the DIAG-05 gated re-fit as EXECUTED.

---

*Phase 25 -- Activate the Gains (Gated WP/ATS Re-fit). ACTV-06 activation readout. No hype,
no betting framing; the deploy bar and the bettable bar are kept separate. ASCII only (no
emoji, per CLAUDE.md).*

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

What this covers in this document: the Phase-25 per-target closing-line-value comparison, the WP and ATS promotions and the O/U retention were all decided on the defective inputs, against a gate baseline measured on those same inputs.
