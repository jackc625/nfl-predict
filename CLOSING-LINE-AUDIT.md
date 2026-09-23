# Closing-Line Fit Audit

**Written:** 2026-09-22 (Phase 33.2, Plan 33.2-21, under SPEC R7 and D33.2-03)
**Status:** committed record. Guarded by `tests/unit/test_closing_line_audit_md.py`.

## What this document is

In plain English: a closing line is the price a sportsbook was showing just before kickoff.
It is a perfectly good yardstick for **grading a bet we placed** and for **measuring closing-line
value**, and it stays one. What it is not is information we had. Each game's information locks at
18:00 America/New_York on the day before kickoff (D33.2-01), and a closing line did not exist then.

So the rule this phase enforces is narrow and absolute: **no closing line may fit anything a
prediction depends on.** Not a model's feature set, not a blend weight, not a bet threshold, not the
scale that turns a residual into an expected value. Closing lines remain valid for grading bets and
measuring CLV, and for nothing else.

This document lists every place in this repository where a closing line feeds, or fed, a **fit**, and
gives each one exactly one disposition. It exists because the alternative -- remembering -- is how a
number derived from tomorrow's price ends up inside today's decision and nobody notices for two
milestones.

One more boundary, because the two look alike from a distance. Plan 33.2-21 introduced
`models/market_probability.py`, which converts a **pre-lock spread** into the market's implied win
probability. That is a **blend input**: the market's own opinion, standing beside the model's, applied
**after** the model has predicted. It is not a model input. No betting line of any timing is a model
input for any target (D33.2-03, enforced in gold by Plan 33.2-19's rung 9).

## What history the fit actually has, stated plainly

The converter named above is fitted on the owned `odds_timeline` -- the Phase-29 purchase, and the only
store in this repository whose lines carry a **real capture time**. That store covers **2020-2024 only**,
and after the per-game lock join and the tie drop it supplies **1,342 graded games**.

It does not cover 2018-2019 (The Odds API's historical data begins 2020-06, so those seasons are
unbuyable at any price) and it does not cover 2025 (the forward-capture path has never run). The
stored 2018-2025 closing lines in silver `odds_snapshot` carry **no genuine capture time at all**, so
under the D33.2-22 ruling of Plan 33.2-14 they are inadmissible as pre-lock evidence and cannot stand
in for the missing seasons. Five seasons of honest line history is what this fit has. Anything in this
repository that implies broader coverage for a pre-lock fit is wrong, and this paragraph is the
correction.

## The vocabulary, and what each word commits to

Five values, closed. Every row takes exactly one.

- **`MOVED_TO_PRELOCK`** -- the fit still happens, but on honest pre-lock data or on the corrected
  models. The `Discharged by` cell names the plan that moves it.
- **`GRADING_ONLY`** -- the closing line is **STILL USED at that site**, to grade bets or to measure
  CLV, and feeds no fit. This word is narrow and must not be stretched: an artifact that grades nothing
  is never `GRADING_ONLY`.
- **`DELETED`** -- the site feeds no fit **and** is not retained as a comparator. Where the artifact
  itself leaves the tree, `Discharged by` names the plan that removes it. Where it remains on disk but
  is dead as a fit input with no reader, the `Reasoning` cell says so rather than implying a file edit
  that did not happen.
- **`OPEN_ACCEPTED`** -- the site still has a **NAMED LIVE READER**, this phase does not sever it, and
  the audit records that honestly instead of certifying a closure the tree does not have (D33.2-26). A
  row may take this disposition only if its `Reasoning` names the reader by file and symbol.
- **`NOT_A_CLOSING_LINE_FIT`** -- listed because a reader will wonder, and measured to be clean: no
  closing line enters that fit at all.

If a site is dead **and** unread, it is `DELETED`. If it is dead but still **read**, it is
`OPEN_ACCEPTED` with its reader named. That is what makes this document and the standing "old baselines
are dead" ruling say the same thing rather than two different things.

## The table

One row per site. One site per row. One disposition per row.

| Site | Used for | Disposition | Discharged by | Reasoning |
|---|---|---|---|---|
| The three models' feature lists | Line columns selected as model inputs | `MOVED_TO_PRELOCK` | 33.2-19 | The lines leave the inputs entirely through the one registry and nothing stands in for them. |
| `backtest/signal_lift._walkforward_clv_series` (feature-group keep/drop screen objective) | Selecting which feature groups the corrected models may use, by paired closing-line CLV lift | `MOVED_TO_PRELOCK` | 33.2-22 | The re-measured verdict is taken on each model's own out-of-sample outcome loss with no market line of any timing, and the closing-line objective survives only as the default path that reproduces the published Phase-28, Phase-29 and Phase-30 records. |
| `models/blending_data.TUNING_SEASONS` (blend sigmoid fit, 2010-2017 nflverse closing lines) | Fitting blend weights on closing lines | `MOVED_TO_PRELOCK` | 33.2-24 | Re-tuned on the owned pre-lock lines under D33.2-10, which is five seasons of real capture times rather than eight of closing prices. |
| Blend edge-threshold calibration | Calibrating the bet edge threshold against a closing-line market probability | `MOVED_TO_PRELOCK` | 33.2-26 | Re-derived against the spread-derived market probability under D33.2-09. |
| `models/blending_data.extract_noise_profile:175` (blend noise profile) | Synthesising predictions for the week-varying blend fit | `DELETED` | 33.2-24 | It exists only to synthesise predictions for the week-varying fit that 33.2-24 removes, so nothing survives to move. |
| `backtest/tune.run_comparison:805` (blend mode gating) | Comparing a fixed against a dynamic blend on a closing-line CLV metric | `DELETED` | 33.2-24 | With the dynamic shape gone under D33.2-10 there is no second mode to gate, and the metric is dead under D33.2-03. |
| `backtest/tune._gate_per_target:617` (per-target mode gate) | Deciding per target which blend mode wins on a closing-line CLV comparison | `DELETED` | 33.2-24 | Same reason as its caller. It is written as its own row because one row names one site. |
| EV floor sweep (`backtest/profitability_2025.py:1956-2003`, sweeping `backtest/ou_ev_chain.EV_FLOOR_GRID:113`, frozen at `ee20773`, with the chosen t served to the live path by `backtest/weekly_bet_list.load_frozen_chain_fit:359`) | Sweeping the per-target EV floor against closing-line outcomes on the pre-correction models | `MOVED_TO_PRELOCK` | 33.2-29 | Re-swept on the corrected models and the owned pre-lock lines, published as a superseding correction naming `ee20773`, before any 2026 bet row is written (D33.2-25). |
| Frozen residual SD (`backtest/ou_ev_chain.fit_frozen_residual_sd:234`, called at `backtest/ou_monetization.py:326` and `backtest/profitability_2025.py:1201`) | Fitting the frozen residual scale that turns a model residual into a bet's EV | `MOVED_TO_PRELOCK` | 33.2-29 | Re-fitted on the corrected models' residuals in the same superseding correction. The scale belongs to the model it was fitted on, and that model is replaced. |
| O/U high-total boundary 48.0 (upper tertile of 2018-2022 closing totals) | Gating which O/U candidates were eligible to bet | `DELETED` | 33.2-06 | D33.2-24 removed the gate and the constant together, so this is not a move and not an exception. Nothing in the tree defines the boundary today. |
| Frozen 2026 edge thresholds (`11761c7`) | Freezing the 2026 bet thresholds off the old models and closing lines | `MOVED_TO_PRELOCK` | 33.2-26 | Superseded visibly by a later commit rather than edited in place. |
| `config/gate.toml [baseline.*]` | Holding the deploy gate's per-target paired comparison baselines | `OPEN_ACCEPTED` | no plan in this phase -- an accepted open item under D33.2-26 | THREE LIVE READERS, MEASURED 2026-09-16 and re-measured 2026-09-22, all named here. `scripts/promote_models.py:915` DEFINES `_baseline_bundle`, which reads cfg[baseline][target] and flattens the 15 sections into `deploy_gate.evaluate_target` as the paired comparator, and calls it at `scripts/promote_models.py:1589`. `scripts/retrain_models.py:48` imports that same function and calls it at `scripts/retrain_models.py:380`, feeding evaluate_target on the RE-FIT path this phase exercises at plans 33.2-23 and 33.2-25. And `tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore` re-scores against the same sections. They grade nothing, so `GRADING_ONLY` would be false. They still serve as a comparator, so `DELETED` would be false too. Under the standing old-baselines-are-dead ruling this is a RETAINED dead comparator, and severing those read paths is a separate declared change this phase does not make. |
| Train-time ATS/O-U residual converters | Converting model residuals to cover and over probabilities | `NOT_A_CLOSING_LINE_FIT` | n/a | Pure actual-minus-model. No closing line enters the fit. |
| Chain-fit bias | Estimating the per-season chain bias | `NOT_A_CLOSING_LINE_FIT` | n/a | Pure actual-minus-model. No closing line enters the fit. |

## Notes on three rows a reader is likely to stop at

**`config/gate.toml [baseline.*]` is `OPEN_ACCEPTED`, and that is the honest answer.** An earlier draft
of this row asserted that this phase's enforcement severed every reader of those sections. Measured
live, it does not: two production modules and one integration test read them today, and neither
enforcement point that draft cited (Plan 33.2-23 on new artifact metadata, Plan 33.2-25 on published
results) reaches any of the three. Committing a repo-root honesty document that certifies an
enforcement the tree does not have would be precisely the failure this phase exists to stop, in the
phase named for it. So the row names all three readers and records the item as open (D33.2-26).

That row's `Reasoning` argues about two dispositions it is rejecting while its own verdict is a third.
This is deliberate, and it is why every check written against this table reads the **`Disposition`
cell**, resolved by name from the header row, rather than scanning the whole row line. Reasoning prose
is not a verdict.

**The EV floor and the frozen residual SD are discharged by Plan 33.2-29, not by D33.2-24.** An earlier
draft cited D33.2-24, which is the O/U eligibility-gate deletion and re-derives neither of them.
D33.2-25 is the ruling that orders both re-derived on the corrected models and the owned pre-lock lines.

**Nothing bundles.** Two rows in the starting list each named two distinct sites. They are split here,
one row per site, because a bundled row contributes one entry to the listed-sites set while covering two
sites in the tree -- and the anti-rot check below compares exactly those two things.

## Anti-rot

`tests/unit/test_closing_line_audit_md.py` structurally parses this repository for the symbols through
which a closing line reaches a fit, and fails when one of them exists in the tree but appears nowhere in
the table above. The guard carries a planted-violation control, so it is known to be capable of failing
rather than merely observed to pass.

The same module checks the header is exactly the five mandated columns in order, that no row is ragged,
that every `Disposition` cell holds exactly one token from the closed five-value vocabulary, that no
`Site` cell names two sites, that every `OPEN_ACCEPTED` row names its live reader by file and symbol,
and that the gate-baseline row names every one of its three live readers rather than some of them.

---

*Phase 33.2, Plan 33.2-21. SPEC R7; decisions D33.2-03, D33.2-09, D33.2-24, D33.2-25, D33.2-26.*
