# COLD-START PRE-REGISTRATION -- the 2026 chain-fit bias and the six edge thresholds

**Status:** FROZEN. This document and `backtest/cold_start_constants.py` are ONE pre-registration in two files,
landed in ONE commit containing nothing else. This is the human-readable half; the constants
module is the half the code reads.

**What this document is for.** Phase 33 ships a live cold start. Two quantities have to exist
BEFORE the numbers they govern arrive, or they are not rules at all -- they are descriptions
written after the fact. The first is the 2026 chain-fit bias: without a frozen value, the
cold-start refusal prescribes re-running a spent, unrepeatable one-shot measurement. The second
is the six per-target edge thresholds: one threshold pair is currently applied to three
incompatible units, which put ATS in the "high" band on the overwhelming majority of games.
Both are frozen once here and NEVER recomputed in-season.

**Editing after the anchor commit does not fix a bug -- it destroys the evidence.** A value
that is wrong here is wrong for the remainder of the phase. The only legitimate response is a
NEW, VISIBLY-LATER CORRECTIVE COMMIT that explicitly INVALIDATES this pre-registration by
naming its commit sha. Never an edit in place, and never a quiet supersession.

**The anchor is recorded outside the files it witnesses.** Neither this document nor
`backtest/cold_start_constants.py` records its own content hash. A file that must CONTAIN and exactly REPRODUCE
its own whole-file hash is self-referential: writing the hash changes the bytes the hash was
computed over, so no fixed point exists without a canonical exclusion rule nobody has defined.

```
preregistration_anchor_recorded_in: tests/phase33_state.py
preregistration_anchor_slots: PRE_REGISTRATION_COMMIT, PRE_REGISTRATION_FILE_SHA256, PRE_REGISTRATION_AUTHOR_DATE, PRE_REGISTRATION_EXTERNAL_ANCHOR, PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND
preregistration_paths: COLD-START-PREREGISTRATION.md, backtest/cold_start_constants.py
preregistration_deadline: 2026-09-17T20:15:00-04:00
```

`tests/unit/test_phase33_preregistration_ancestry.py` resolves the commit from git against
those two paths, asserts it equals the recorded slot, recomputes each file's sha256 and
compares, and asserts the commit's AUTHOR date precedes the deadline above.

---

## 1. What anchors this, and what each anchor can actually carry

Three anchors, listed in descending order of what they prove.

1. **Git ancestry** against the committed Phase-33 readout commit. This proves ordering INSIDE
   the repository. It cannot reach wall-clock time, because `data/`, `outputs/` and
   `artifacts/` are ALL gitignored -- so ancestry alone anchors the rule to a readout we write
   ourselves.
2. **An external time anchor** -- a signed tag pushed to the remote, a remote push receipt, or
   a CI attestation. Only something produced by a system OTHER than this working tree can
   establish that the rule existed before kickoff, which is the property a pre-registration
   exists to have. Which kind was obtained is recorded in
   `tests.phase33_state.PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND`.
3. **The commit's author date**, against `2026-09-17T20:15:00-04:00`. This is **CORROBORATION
   ONLY**. A git author date is LOCALLY SETTABLE -- `GIT_AUTHOR_DATE` and `git commit --date`
   both set it -- so it CANNOT prove pre-kickoff existence on its own.

**If no external anchor was obtainable in this environment, this pre-registration is anchored
by git ancestry and a corroborating author date ONLY, and this document says so rather than
implying a strength the evidence does not have.** The authoritative record of which case holds
is `PRE_REGISTRATION_EXTERNAL_ANCHOR_KIND`: the literal `NONE_AVAILABLE` means no external
anchor could be produced, and the reason is stated in the plan's summary. That slot is written
in a LATER commit than this one, because a file cannot record a fact about the commit it is
part of.

**What no anchor can prove.** None of the three can prove that nobody tuned a threshold after
seeing the numbers. No test can prove intent. That clause is judgment-tier and is discharged by
the owner's ratification, recorded with its date.

---

## 2. The six edge thresholds

| target | medium | high | unit |
|---|---|---|---|
| wp | `0.0200` | `0.0500` | probability (model minus devigged fair closing probability) |
| ats | `0.8359` | `1.9493` | points (signed model-minus-market home margin) |
| ou | `0.0220` | `0.0546` | ratio of the market total, floored at 30 |

**WP's pair does not move.** It stays exactly `0.0500` / `0.0200`, so WP labels do not change
at all. That is deliberate: the 23-point `_EDGE_TIER_SNAPSHOT` recorded BEFORE the Phase-31
helper collapse survives unchanged and keeps its evidentiary value, instead of being rewritten
with a new expectation.

**ATS's and O/U's pairs are WP-ANCHORED QUANTILES.** Each is set to the value that reproduces
WP's own band shares on that target's own `|edge|` distribution. WP's measured shares on the
end-state artifacts are low `0.2171` / medium
`0.2649` / high
`0.5179`. The convention is
`numpy.quantile(..., method="linear")`, named explicitly so a future library default cannot
silently move a frozen threshold, and the bands are assigned with STRICT `>` so a value exactly
at a threshold falls in the LOWER band -- the behaviour `utils.edge_tier` already has.

**The ATS population includes REPAIRED PICK-EM ROWS.** Under the D33-31 owner ruling a pick-em
is a REAL line, so a game with `market_spread == 0` now carries its genuine nonzero
disagreement instead of a forced `0.0`. Those rows shift the ATS distribution's mass, which is
why this derivation had to run AFTER Plan 33-10's repair. A threshold derived from the
pre-repair distribution would have been frozen against a defect.

**The derivation is mildly circular, and it is stated here rather than discovered in review.**
The thresholds are derived on gold seasons 2021-2024,
and that is ALSO the D31-04 pinned population the label movement below is measured against.
Removing the circularity would mean deriving on a population the movement is not measured
against, which trades a stated caveat for an unstated mismatch.

---

## 3. The label movement, in games

Held at ONE model and ONE edge definition, so the thresholds are the only thing that varies.
Counts are `low / medium / high` over the rows with a computable edge.

| target | under the CURRENT 0.05 / 0.02 pair | under the FROZEN pairs | games changing band |
|---|---|---|---|
| wp | 236 / 288 / 563 | 236 / 288 / 563 | 0 |
| ats | 2 / 11 / 1074 | 236 / 288 / 563 | 522 |
| ou | 212 / 258 / 617 | 233 / 293 / 561 | 77 |

This is a **PUBLISHED-LABEL CHANGE**: the band is rendered and sorted on by `/` and `/betting`.
It is disclosed rather than slipped in, and the owner's acceptance of it is recorded with its
date.

Two "before" figures exist and they answer different questions. The one in the table is the
end-state model's edges under today's thresholds -- it isolates the THRESHOLD change. The
figure recorded in `tests.phase33_state.ATS_BAND_SHARES_BEFORE` (ATS high at 92.64%) was
measured on the PRE-rebuild artifacts under the PRE-repair ratio-scale ATS edge, so the
difference between it and anything here mixes three changes: the unit repair, the re-fit and
the thresholds. Both are recorded; neither overwrites the other.

---

## 4. The 2026 chain-fit bias

The pooled mean residual (`actual - predicted`) over the STRICTLY-PRIOR seasons
2021-2025, on each target's own scale, computed by the EXISTING
`backtest.ou_ev_chain.estimate_prior_season_bias`. **No second estimator was written.**

| target | 2026 bias | residual source artifact | disposition | gate verdict |
|---|---|---|---|---|
| wp | `-0.03377244391544111` | `wp_20260914_221745` | promoted_refit | PASS |
| ats | `0.257407648096468` | `ats_20260914_221751` | promoted_refit | FAIL |
| ou | `-0.35080281804116925` | `ou_20260914_221756` | promoted_refit | FAIL |

**A `promoted_refit` carrying a non-PASS verdict was shipped under a recorded owner override.**
The verdict was never softened to match the ruling; it stands as measured in
`config/phase33_gate_verdict.toml`, and the override sits BESIDE it. This table is where that
is visible rather than smoothed away.

**The pool is partly in-sample, and the direction of that bias is stated.** The deployed
artifacts were fitted through the final-fit entry point over every completed season 2002-2025,
which INCLUDES this residual pool. So these residuals are in-sample and the bias they produce
is ATTENUATED -- the true out-of-sample bias is likely LARGER in magnitude. That is the
conservative direction to know about.

**An EMPTY strictly-prior residual pool REFUSES BY NAME.** No bias is invented and there is NO
fallback to the target season's own data. The refusal is
`scripts.derive_cold_start_constants.EmptyResidualPoolError`.

**JSON round-trips season keys as strings while the lookup is by int.** Every season-keyed
mapping uses INT keys in Python; a JSON-facing representation of the same mapping carries
STRING keys, because JSON has no integer keys at all. A consumer that serializes and reloads
one must look up `"2026"` and not `2026`.

---

## 5. The fix-cycle allowance is ZERO, and it was declared before any verdict

`FIX_CYCLE_ALLOWANCE = 0`, declared in `scripts/run_phase33_gate.py` BEFORE
any Phase-33 verdict existed and restated in `backtest/cold_start_constants.py` as part of this record. It is not
a second declaration; it is the same one, carried into the frozen rule so a reader of the rule
meets it.

---

## 6. How to reproduce every number above

```
uv run python -m scripts.derive_cold_start_constants --check
```

The derivation is a COMMITTED, DETERMINISTIC program
(`scripts/derive_cold_start_constants.py`). It takes the end-state artifact ids and their
content digests as explicit arguments, REFUSES on any digest mismatch -- naming the input and
both digests -- and emits these two files. Running it twice against the same inputs produces
byte-identical output. Every input it read, with its digest, is recorded in
`DERIVATION_INPUT_DIGESTS`; the row count behind every derived number is in
`DERIVATION_ELIGIBLE_COUNTS`; the quantile convention is in
`THRESHOLD_QUANTILE_CONVENTION`.

---

## 7. What this pre-registration does NOT claim

It does not claim the thresholds are BETTER. No measurement here shows that a game in the new
"high" band is a better bet than a game in the old one. What it claims is narrower and
checkable: that the bands are now defined per target on that target's own unit, that the
values were fixed before the season they govern, and that the movement they cause was measured
and disclosed before anyone saw a 2026 result.

It does not claim the 2026 bias is correct. It claims the bias is the walk-forward estimate the
existing estimator produces from strictly-prior completed seasons, that its pool is partly
in-sample and therefore attenuated, and that it was frozen rather than chosen later.

---

*Phase: 33-live-cold-start-forward-temporal-integrity*
*Frozen by Plan 33-16, before 2026-09-17T20:15:00-04:00*
