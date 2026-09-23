# EV-CHAIN-CORRECTION.md -- the EV floor and the frozen residual SD, re-derived

**Phase 33.2, Plan 33.2-29. Owner decision D33.2-25. Measured 2026-09-23.**

**Built on re-measured past seasons; not clean evidence.** Only the 2026 season, recorded live
under the day-before 6 PM ET lock, counts (D33.2-07). These numbers select a threshold. They say
nothing about whether betting is profitable.

---

## 1. What changed, in plain English

Two numbers decide whether the system places a bet:

- **The EV floor** is the smallest expected value a bet must show before it is placed at all. A
  floor of 0.00 means "any bet the model rates as positive"; 0.05 means "only bets it rates at
  least 5 cents per dollar".
- **The frozen residual SD** is the typical size of the model's miss, in points. It turns "the model
  says 47, the line says 44" into a probability, and so into that bet's expected value. The spread
  and total bets carry one; the win bet does not, by design (D31-07).

Both were chosen in Phase 31 against **closing-line** results on models this phase replaced. Those
models were fitted on inputs later found defective, so the numbers go with them (D33.2-25). They
are re-derived here on the three corrected models and on the lines we actually owned before each
game's lock.

## 2. What was superseded

Pre-registration commit `ee20773` (`ee20773b58c3a59de2450d56c64992e240282820`) froze the Phase-31
chain in `backtest/ev_chain_constants.py` and `PROFITABILITY-PREREGISTRATION.md`. **Both files are
byte-unchanged.** They remain the record of what was frozen and when. This correction sits beside
them in `backtest/corrected_ev_chain_constants.py`, which names `ee20773` and re-declares none of
the frozen constants. The Phase-31 run record `outputs/p31/profitability_2025_verdict.json` is also
left in place, unedited.

## 3. Old and new values

| Target | Old floor | New floor | Old SD | New SD |
|---|---|---|---|---|
| wp | 0.05 | 0.00 | none | none |
| ats | 0.05 | 0.05 | 11.13 | 13.45 |
| ou | 0.00 | 0.00 | 12.98 | 13.67 |

Every target got a floor; no target came back with "no threshold". The win bet has no SD by
design (D31-07), so "none" in both SD columns is not a gap.

**Read the floors with this beside them: on these past seasons every floor, for every bet type,
lost money.** The rule picks the floor with the best return on the grid, and the best was still
negative -- about -2% for win bets (357 bets), -9% for spread bets (820) and -6% for total bets
(1,038). A floor chosen this way is the least-bad threshold, not evidence of an edge.

## 4. The window

2020-2024 is the whole honest corpus, not a preference: 2018-2019 are unbuyable at any price and
2025 has no free pre-lock source. The win-bet sweep uses 2021-2024 only: 2020 has no earlier season
to fit the spread-to-probability converter on, so its 255 games are set aside as
`no_prior_fold_converter` rather than priced with a slope fitted on their own results. Spread and
total bets, and both SDs, keep 2020-2024. 60 games had no owned line before their lock and are
counted as `no_prelock_line`.

**The single-use 2025 hold was NOT re-spent.** No 2025 row enters any fit, a planted one is refused
by name, and the run ledger `config/profitability_2025_run_ledger.toml` is untouched (its bytes are
compared before and after every run).

## 5. How win bets are priced

The owned lines carry no moneyline, so each past win bet is priced from the market's out-of-fold
win probability. It is priced **with the bookmaker's cut**: a winning bet pays 100/110 of what the
no-vig price would pay, the standard -110 cut (a 4.76% two-way hold at even odds) that the spread
and total bets are graded at. Silver stores no live pre-lock moneyline to measure a cut from. As a
cross-check, the live pre-lock captures we do own (14 games, 9 books) show a median hold of 4.2%,
so -110 is realistic and slightly conservative. An earlier run of this correction priced win bets
at the no-vig line, which flattered their return (+5.6% over 412 bets); with the cut it is -2.0%.
The chosen win floor did not move.

## 6. The owner ruling on a target with no floor

One deliberate change from the Phase-31 recipe: when no floor on the grid admits a single bet,
Phase 31 silently used the loosest floor, 0.00. This correction refuses instead and records "no
threshold", which means that bet type places no bets. On 2026-09-23 the owner was asked what should
happen to such a target; the recommendation was `no-bets-for-that-target` -- that bet type places no
2026 bets, and its predictions are still produced and published -- and the owner answered "Proceed".
**Ruling: `no-bets-for-that-target`.** No target triggers it today.

## 7. Staged, not live

Nothing here changes what the system bets on yet. The live bet list still reads the Phase-31
record. Plan 33.2-26 Task 3 switches it to `outputs/p332/corrected_chain_fit.json` in the same
commit that switches the cold-start bias, so the live rule changes once and wholly.

## 8. Reproduce

`uv run python -m scripts.derive_corrected_ev_chain` -- thread-pinned at 1, over the four artifacts
now in production (`wp_20260923_172144`, `ats_20260923_172148`, `ou_20260923_172152`,
`blend_20260923_212418`) and converter `market_probability_20260923_195443`.

**Built on re-measured past seasons; not clean evidence.**
