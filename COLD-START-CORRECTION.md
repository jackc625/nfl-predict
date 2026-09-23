# COLD-START CORRECTION -- superseding the 2026 rule frozen at `11761c7`

**Status:** a SUPERSEDING CORRECTION. This document and `backtest/corrected_cold_start_constants.py` are one
record in two files. The record they supersede -- `COLD-START-PREREGISTRATION.md` and `backtest/cold_start_constants.py`,
frozen at commit `11761c7` (`11761c7ece83ab9cab8ce73ffd6d7b58158ee703`) -- is
**byte-unchanged** and stays the record of what was frozen and when. Nothing here edits it.

**Why it is superseded.** `11761c7` froze the 2026 edge thresholds and the 2026 chain-fit
bias from models fitted on inputs later found defective (Phase 33.2) and from CLOSING lines, which
did not exist at a game's lock. Those models are gone. A value derived from them sits inside the
live 2026 bet rule, so it is replaced visibly rather than edited quietly.

**Not clean evidence (D33.2-07).** Every number below is re-measured on past seasons. It sets a
threshold; it does not show that a bet in any band is profitable. Only the 2026 season, recorded
live under the new lock rule, counts as evidence.

---

## 1. What was swapped, and what was held

**Swapped:** the models -- the three corrected artifacts production serves
(`wp_20260923_172144`, `ats_20260923_172148`, `ou_20260923_172152`), through their own recorded recipes'
walk-forward predictions, each season predicted by a fit on strictly earlier seasons -- and the
market: the owned `odds_timeline` line at or before each game's lock, never a closing line.
WP's market side is the spread-derived probability from converter `market_probability_20260923_195443` (the one
the live blend `blend_20260923_212418` binds), converted OUT OF FOLD for every historical game.

**Held:** WP's `0.0500` / `0.0200` anchor pair, the WP-anchored band-share quantile rule
(`numpy.quantile`, method "linear"), the STRICT `>` bands, the digest refusals and the
walk-forward bias estimator.

**The window is 2020-2024.** 2020-2024 is the whole honest corpus, not a preference: 2018-2019 are unbuyable at any price and 2025 has no free pre-lock source. The single-use 2025 hold is spent and no row of it enters this derivation.
WP uses 2021-2024 only: the window's first season has no prior-fold converter slope, so
**255 games leave the WP derivation as `no_prior_fold_converter`**
(counted, never filled with the serving slope). They stay in the ATS and O/U derivations.
60 scheduled games in the window had no owned line at or before their
lock and are not in any part of it.

---

## 2. The edge thresholds, old beside new

`medium / high`, each on its target's own unit, over the rows with a computable edge.

| target | superseded (`11761c7`) | corrected | rows | unit |
|---|---|---|---|---|
| wp | `0.0200` / `0.0500` | `0.0200` / `0.0500` | 1093 | probability (model minus the spread-derived out-of-fold market probability) |
| ats | `0.8359` / `1.9493` | `0.7287` / `1.7590` | 1348 | points (signed model-minus-market home margin) |
| ou | `0.0220` / `0.0546` | `0.0173` / `0.0438` | 1348 | ratio of the market total, floored at 30 |

WP's measured band shares under its unchanged pair, the anchor ATS and O/U reproduce: low `0.1263` / medium `0.1976` / high `0.6761`.

**A target with too little honest data gets NO threshold.** Below
`MIN_HONEST_THRESHOLD_ROWS = 100` rows the derivation raises
`InsufficientHonestDataForThresholdError`, records `None`, and that target places no bets and
carries no edge band. The superseded value is never borrowed. Refusals in this derivation:

None. Every target cleared the honest-data floor, so every target has a threshold.

---

## 3. The label movement, in games

Counts are `low / medium / high` over the corrected edges, under the superseded pairs and under
the corrected ones.

| target | under the `11761c7` pairs | under the corrected pairs | games changing band |
|---|---|---|---|
| wp | 138 / 216 / 739 | 138 / 216 / 739 | 0 |
| ats | 197 / 297 / 854 | 171 / 266 / 911 | 83 |
| ou | 210 / 338 / 800 | 170 / 266 / 912 | 152 |

---

## 4. The 2026 chain-fit bias, old beside new

| target | superseded (`11761c7`) | corrected |
|---|---|---|
| wp | `-0.03377244391544111` | `0.0004000931458235923` |
| ats | `0.257407648096468` | `-0.8288653630898939` |
| ou | `-0.35080281804116925` | `0.6443360853661155` |

The corrected value continues the walk-forward bias series of the corrected chain fit
(`outputs/p332/corrected_chain_fit.json`, record `corrected_chain_fit_20260923_231233`, Plan 33.2-29): the same rows,
the same residual helper and the same estimator. The derivation first reproduces every season
that record prices, exactly, and only then extends the series to 2026, pooling every
strictly-prior season it covers (2017-2024). No 2025 row is read, so the pool ends at
2024 and the target season is named (`CHAIN_FIT_BIAS_TARGET_SEASON`) rather than inferred from
it. The superseded value was pooled IN-SAMPLE over the retired models' own training seasons; this
one is out of sample.

---

## 5. The other two moved parts of the 2026 bet rule: the EV floor and the frozen residual SD

The 2026 bet rule is these thresholds and this bias PLUS the per-target **EV floor** (the number
that decides whether a bet is placed at all) and the **frozen residual SD** (the scale that turns
a model's miss into a bet's expected value). D33.2-25 rules that all of them move together. The
EV floor and the residual SD are superseded separately -- naming `ee20773`, the Phase-31
pre-registration they came from -- by Plan 33.2-29: `backtest/corrected_ev_chain_constants.py`
and `EV-CHAIN-CORRECTION.md`. Their values, as the corrected chain fit records them:

| target | EV floor | frozen residual SD |
|---|---|---|
| wp | `0.0` | `None` |
| ats | `0.05` | `13.446837941873182` |
| ou | `0.0` | `13.67472671184805` |

WP fits no residual SD by design (D31-07). A `None` EV floor would mean no honest floor and no
bets for that target; none is `None` here.

---

## 6. How the live rule changes

The live rule changes in ONE commit (Plan 33.2-26 Task 3), which moves every live reader
together: the chain-fit record path, this bias, and the edge bands read by the web cache and the
current-week predictions. Before that commit the live bet list read the Phase-31 chain-fit record
and the `11761c7` bias; no run can judge corrected floors against an uncorrected bias,
or the reverse. From that commit on, a 2026 WIN bet must pass two tests:
its edge over the spread-derived market probability, on the side bet, is above WP's corrected
MEDIUM threshold (`0.0200`: the edge band is at least "medium"), and the moneyline captured at
that game's lock still leaves positive value after the book's cut (D33.2-11). A target whose
threshold is `None` places no bets at all.

---

## 7. How to reproduce every number above

```
OMP_NUM_THREADS=1 uv run python -m scripts.derive_cold_start_constants --corrected --trust-inputs
```

The derivation reads the four served artifact ids from `tests/phase33_state.py`, refuses unless
`artifacts/latest.json` serves them and the live blend binds the recorded converter, pins every
fit to 1 thread, and records every input's digest in `DERIVATION_INPUT_DIGESTS`.
Running it twice against the same inputs produces byte-identical files.

---

*Phase: 33.2-information-time-integrity-day-before-kickoff-lock-and-hones*
*Plan 33.2-26, superseding `11761c7`*
