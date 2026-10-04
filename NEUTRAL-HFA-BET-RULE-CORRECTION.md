# COLD-START CORRECTION -- superseding the 2026 rule frozen at `9bb7568`

**Status:** a SUPERSEDING CORRECTION. This document and `backtest/neutral_hfa_cold_start_constants.py` are one
record in two files. The record they supersede -- `COLD-START-CORRECTION.md` and `backtest/corrected_cold_start_constants.py`,
frozen at commit `9bb7568` (`9bb7568f78c3b2714fb03da0818104ab7c10f620`) -- is
**byte-unchanged** and stays the record of what was frozen and when. Nothing here edits it.

**Why it is superseded.** `9bb7568` measured the 2026 edge thresholds and the 2026 chain-fit
bias on the three models production served from 2026-09-23. Owner ruling 2026-10-03 ~22:12 ET
(WINDOWS row 19, "No home boost at any neutral site"): Elo home-field advantage is zero at every
neutral site and neutral games are left out of HFA learning. Elo, gold and all three models were
re-derived under that rule and swapped into production on 2026-10-04, so values measured on the
old models describe models that no longer serve. Owner ruling 2026-10-03 ~22:50 ET ("Re-measure,
same recipe"): the recipe that produced `9bb7568` is re-run, unchanged, on the new production
models, and the result replaces it visibly rather than by an edit in place.

**Not clean evidence (D33.2-07).** Every number below is re-measured on past seasons. It sets a
threshold; it does not show that a bet in any band is profitable. Only the 2026 season, recorded
live under the new lock rule, counts as evidence.

---

## 1. What was swapped, and what was held

**Swapped:** the models -- the three corrected artifacts production serves
(`wp_20261004_050223`, `ats_20261004_050228`, `ou_20261004_050232`), through their own recorded recipes'
walk-forward predictions, each season predicted by a fit on strictly earlier seasons -- and the
market: the owned `odds_timeline` line at or before each game's lock, never a closing line.
WP's market side is the spread-derived probability from converter `market_probability_20260923_195443` (the one
the live blend `blend_20261004_050521` binds), converted OUT OF FOLD for every historical game.

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

| target | superseded (`9bb7568`) | corrected | rows | unit |
|---|---|---|---|---|
| wp | `0.0200` / `0.0500` | `0.0200` / `0.0500` | 1093 | probability (model minus the spread-derived out-of-fold market probability) |
| ats | `0.7287` / `1.7590` | `0.6211` / `1.6229` | 1348 | points (signed model-minus-market home margin) |
| ou | `0.0173` / `0.0438` | `0.0162` / `0.0452` | 1348 | ratio of the market total, floored at 30 |

WP's measured band shares under its unchanged pair, the anchor ATS and O/U reproduce: low `0.1290` / medium `0.1876` / high `0.6834`.

**A target with too little honest data gets NO threshold.** Below
`MIN_HONEST_THRESHOLD_ROWS = 100` rows the derivation raises
`InsufficientHonestDataForThresholdError`, records `None`, and that target places no bets and
carries no edge band. The superseded value is never borrowed. Refusals in this derivation:

None. Every target cleared the honest-data floor, so every target has a threshold.

---

## 3. The label movement, in games

Counts are `low / medium / high` over the corrected edges, under the superseded pairs and under
the corrected ones.

| target | under the `9bb7568` pairs | under the corrected pairs | games changing band |
|---|---|---|---|
| wp | 141 / 205 / 747 | 141 / 205 / 747 | 0 |
| ats | 201 / 263 / 884 | 174 / 253 / 921 | 64 |
| ou | 181 / 234 / 933 | 174 / 253 / 921 | 19 |

---

## 4. The 2026 chain-fit bias, old beside new

| target | superseded (`9bb7568`) | corrected |
|---|---|---|
| wp | `0.0004000931458235923` | `-0.0001220760959361301` |
| ats | `-0.8288653630898939` | `-0.4754073948548693` |
| ou | `0.6443360853661155` | `0.0974858578219975` |

The corrected value continues the walk-forward bias series of the corrected chain fit
(`outputs/row19/neutral_hfa_chain_fit.json`, record `neutral_hfa_chain_fit_20261004_052326`, quick task 261003-vke): the same rows,
the same residual helper and the same estimator. The derivation first reproduces every season
that record prices, exactly, and only then extends the series to 2026, pooling every
strictly-prior season it covers (2017-2024). No 2025 row is read, so the pool ends at
2024 and the target season is named (`CHAIN_FIT_BIAS_TARGET_SEASON`) rather than inferred from
it. The superseded value was pooled the same way, out of sample, over the walk-forward predictions of
the 2026-09-23 models' recipes; only the models moved.

---

## 5. The other two moved parts of the 2026 bet rule: the EV floor and the frozen residual SD

The 2026 bet rule is these thresholds and this bias PLUS the per-target **EV floor** (the number
that decides whether a bet is placed at all) and the **frozen residual SD** (the scale that turns
a model's miss into a bet's expected value). All of them move together. The EV floor and the
residual SD are re-measured by the same recipe on the same new models, superseding the `8c9675e`
record: `backtest/neutral_hfa_ev_chain_constants.py` (record
`outputs/row19/neutral_hfa_chain_fit.json`). Their values, as that chain fit records them:

| target | EV floor | frozen residual SD |
|---|---|---|
| wp | `0.0` | `None` |
| ats | `0.0` | `13.42742143578699` |
| ou | `0.01` | `13.737115789484102` |

WP fits no residual SD by design (D31-07). A `None` EV floor would mean no honest floor and no
bets for that target; none is `None` here.

---

## 6. How the live rule changes

The live rule changes in ONE commit (the row-19 repoint), which moves every live reader
together: the chain-fit record path, this bias, and the edge bands read by the web cache and the
current-week predictions. Before that commit the live bet list read the `8c9675e` chain-fit record
and the `9bb7568` bias; no run can judge re-measured floors against a superseded bias, or the
reverse. The bet itself is unchanged from `9bb7568`: a 2026 WIN bet must pass two tests:
its edge over the spread-derived market probability, on the side bet, is above WP's corrected
MEDIUM threshold (`0.0200`: the edge band is at least "medium"), and the moneyline captured at
that game's lock still leaves positive value after the book's cut (D33.2-11). A target whose
threshold is `None` places no bets at all.

---

## 7. How to reproduce every number above

```
OMP_NUM_THREADS=1 uv run python -m scripts.derive_cold_start_constants --corrected --correction row19 --trust-inputs
```

The derivation reads the four served artifact ids from `tests/phase33_state.py`, refuses unless
`artifacts/latest.json` serves them and the live blend binds the recorded converter, pins every
fit to 1 thread, and records every input's digest in `DERIVATION_INPUT_DIGESTS`.
Running it twice against the same inputs produces byte-identical files.

---

*Quick task 261003-vke: zero Elo home-field advantage at neutral sites (WINDOWS row 19)*
*Superseding `9bb7568` by the same recipe*
