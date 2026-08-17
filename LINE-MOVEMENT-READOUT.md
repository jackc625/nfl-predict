# Line-Movement Budget Gate Readout (Phase 29, SIG-04)

**Status:** GO/NO-GO record -- this is a budget-gate spike, not a build and not a gate ruling.
Nothing here is shipped; the line-movement signal is screened / carried to Phase 30 for the
binding gate, never described as a finished outcome.
**Scope:** price the historical-odds cost as a tiered menu (D-03), state the D-02 plausibility
(non-redundancy) argument, and record the owner's live branch decision (full backfill |
forward-collect-only | slip) BEFORE any build runs (SC1).
**Cost method (D-04):** every credit/dollar figure below is ESTIMATED from The Odds API
published per-credit pricing x the planned call count. The owner is on a FREE key today and the
historical endpoint is PAID-only; the free key 401s on `historical/`, so the endpoint is NOT
called here. The estimate is arithmetic on the VERIFIED credit formula
(`cost = 10 x markets x regions` per snapshot timestamp).
**Honesty framing (D-16):** the readout says "screened / carried to Phase 30". Phase 29 SCREENS
(on the backfill path only); Phase 30 runs the binding deploy gate. Forward-collect-only and slip
are FIRST-CLASS SUCCESS outcomes (D-14, SC3), not "incomplete" phases.

---

## 1. Tiered cost (estimated from The Odds API published pricing x planned calls; free key, historical endpoint not called -- D-04; pricing/formula/earliest-date checked 2026-06-29)

The historical credit formula is `cost = 10 x markets x regions` per snapshot timestamp
requested [VERIFIED against the-odds-api.com/liveapi/guides/v4, checked 2026-06-29]. One
historical call at a timestamp `T` returns the WHOLE NFL board at `T` -- you pay per
(timestamp x markets x regions), NOT per game (assumption A2). The earliest available historical
date is **2020-06-06** [VERIFIED, checked 2026-06-29]; the paid-plan requirement (free tier
excludes historical) is re-confirmed [VERIFIED, checked 2026-06-29]. The cheapest plan that fits
every tier below is the **20K plan at $30/month** (20,000 credits) [VERIFIED pricing, checked
2026-06-29]; the free plan is 500 credits/month and the next paid rung is 100K at $59/month.

**Call-count assumptions (flagged -- arithmetic on the VERIFIED formula):**
- A1: ~22 NFL weeks/season (18 regular + ~4 playoff weekends); full-trajectory cadence = 4
  snapshots/week (open ~Tue, intraweek Wed/Thu, late Thu/Fri, freeze Fri 6 PM ET); two-anchor
  cadence = 2 snapshots/week (open + freeze).
- A2: one historical call per timestamp returns the whole board (pay per timestamp, not per game).
  If the API instead paginated per-event the cost would rise sharply; the docs describe a
  board-wide snapshot, so the risk is low. If the owner proceeds on the backfill path, confirm A2
  with exactly ONE paid historical call (~10-20 credits) before the bulk pull.
- regions=us -> 1 region. Windows: **2020-2024** (~110 wk, adds the 2021 walk-forward training
  fold) or **2021-2024** (~88 wk, holdout-only). 2020 is fully covered since the season starts
  after 2020-06-06.

| Tier | Markets | Snaps/wk | Window | Snapshots | Credit cost (`10 x mkts x reg x snaps`) | $ (cheapest plan that fits) |
|------|---------|----------|--------|-----------|------------------------------------------|------------------------------|
| (a) full trajectory + spread+totals | 2 | 4 | 2020-2024 (110 wk) | 440 | 440 x 20 = **8,800** | $30 (20K), one month |
| (a) full trajectory + spread+totals | 2 | 4 | 2021-2024 (88 wk)  | 352 | 352 x 20 = **7,040** | $30 (20K), one month |
| (b) full trajectory + totals only   | 1 | 4 | 2020-2024 (110 wk) | 440 | 440 x 10 = **4,400** | $30 (20K), one month |
| (b) full trajectory + totals only   | 1 | 4 | 2021-2024 (88 wk)  | 352 | 352 x 10 = **3,520** | $30 (20K), one month |
| (c) two-anchor + totals only        | 1 | 2 | 2020-2024 (110 wk) | 220 | 220 x 10 = **2,200** | $30 (20K), one month |
| (c) two-anchor + totals only        | 1 | 2 | 2021-2024 (88 wk)  | 176 | 176 x 10 = **1,760** | $30 (20K), one month |

**Cost conclusion:** every tier -- including the richest (a)/2020-2024 at 8,800 credits -- fits
inside a SINGLE $30 month of the 20K plan with large headroom (even doubling the cadence to 8
snaps/week keeps Tier (a) at ~17,600 < 20,000). Adding the spreads market doubles credits but
stays in the same $30 plan, so the D-07 "spread conditional" gate is "yes, fold it in" on cost
grounds; the only reason to drop spreads would be payload/complexity, not credits. The forward
-collect path needs NO paid plan at all -- forward capture uses the REGULAR endpoint
(`cost = markets x regions`, ~1-2 credits/call, ~35 credits/month at 4 snaps/week x 2 markets),
comfortably within the free 500/month.

---

## 2. Plausibility (D-02): is line-movement non-redundant vs the freeze anchor + the Phase-28 injury signal?

**Cost is NOT the binding constraint -- plausibility is.** Section 1 shows every tier is a
trivial single $30 month, so the go/no-go pivots entirely on whether line-movement carries
information the model does not already have.

**vs the freeze market anchor (already a model feature).** The Friday-6 PM-ET freeze line is
ALREADY consumed as a feature in `features/market_anchors.py` (`MarketAnchorFeaturesCalculator`,
the `snapshot_total` / `snapshot_spread` anchors). That calculator even emits
`total_movement` / `spread_movement` columns today -- but with a single synthetic snapshot per
historical game, open == snapshot, so those movement columns are identically **0.0 for every
historical game** (a structurally dead column). Net drift (`freeze - open`) is therefore largely
drift against a line the model ALREADY sees: the freeze level is not new. The genuinely-new
information line-movement adds is (i) the **opening total LEVEL** (where the line started, before
it drifted to the known freeze) and (ii) the **path shape** (late/steam movement, reversals,
distance traveled). Net drift on its own is close to feeding the open alongside the already
-present freeze anchor; the level and the path are what could earn their place.

**vs the Phase-28 injury signal.** Phase 28 already ingests injury / snap-count / situational
signals into the widened gold. The market moves the total on the SAME injury news the Phase-28
features encode -- so part of any measured line-movement signal is the market's reaction to
information the model can already read directly from the injury feed. This is the priced-in
risk: buying a signal that is a noisier proxy for features already present.

**Plausibility verdict.** Line-movement is PLAUSIBLY non-redundant ONLY through the
opening-level + path-shape families (D-09 ii-iv), not through net drift against the freeze line.
This is a genuine but BOUNDED edge hypothesis -- exactly the kind the add-one-in CLV lift screen
(backfill path, Plan 29-07) exists to arbitrate. Per D-10 the readout records this priced-in
risk plainly and does NOT pre-decide the lift; the screen, not optimism, rules. The honest
framing carried throughout: if the owner backfills, the signal is screened and carried to
Phase 30, never asserted as an edge on the basis of this spike.

---

## 3. Decision: full backfill | forward-collect-only | slip (+ rationale)

<!-- Machine-readable branch marker. Downstream branch gating (Plans 29-02..29-08) reads this
literal token, not the prose. Task 2 (the blocking owner checkpoint) overwrites PENDING with
exactly one of: full-backfill | forward-collect-only | slip. -->
selected_branch: full-backfill

**Status:** DECIDED at the Task-2 blocking owner checkpoint -- the owner selected
**full-backfill** (build + screen) with the tiered cost menu (Section 1) and the D-02
plausibility argument (Section 2) in hand. SC1 is satisfied: the budget-gate decision is
recorded HERE, before any 29-02+ build runs.

**Owner rationale (full-backfill):**
- **Cost is not the binding constraint (D-01).** Every tier in Section 1 -- including the
  richest (a)/2020-2024 at 8,800 credits -- is a trivial single $30 month of the 20K plan
  with large headroom. With NO pre-registered dollar ceiling (D-01), the spend does not gate
  the decision.
- **The D-02 plausibility edge is genuine but bounded.** Line-movement is plausibly
  non-redundant only through the opening-total LEVEL and the path shape; net drift against the
  already-known freeze line is largely what the model already sees (the freeze anchor is a
  feature in `features/market_anchors.py`), and part of any measured movement is a noisier
  proxy for the Phase-28 injury signal the market reacts to. The edge is real enough to be
  worth measuring, not strong enough to assume.
- **The add-one-in CLV lift screen is the arbiter.** Because the edge is bounded, the owner
  declines to pre-decide it. The backfill path's in-process walk-forward CLV lift screen
  (Plan 29-07) measures the per-target line-movement lift against the activated baseline; that
  screen, not optimism, rules. The line-movement signal is SCREENED in Phase 29 and CARRIED TO
  PHASE 30 for the binding deploy gate -- never asserted as an edge on the basis of this spike.
- **Paid-key provisioning (D-04).** The owner will provision a paid ODDS_API_KEY for a
  one-month historical pull (one-month-paid-then-downgrade; not pre-committed). Plan 29-05
  pauses at its OWN blocking checkpoint before any paid historical call is made, so the bulk
  spend is gated a second time at execution.

**Downstream routing:** the full-backfill branch runs Plans 29-02, 29-03, 29-04, 29-05, 29-06,
and 29-07. (Both are now filled: Section 4 carries the Plan 29-07 lift results, and Section 5
records the honest scope-down of the verdict's strength.)

**The three branches:**
- **full-backfill** -- buy the 2020-2024 historical archive (one paid month, then downgrade --
  D-04, not pre-committed), build `LineMovementBuilder`, run the add-one-in CLV lift screen, and
  carry a measured per-target lift verdict to Phase 30. Runs Plans 29-02, 29-03, 29-04, 29-05,
  29-06, 29-07. Caveat: the 2018-2019 train window has ZERO coverage (floor 2020-06-06) ->
  neutral defaults + a coverage flag; the D-02 redundancy risk may make the lift flat.
- **forward-collect-only** -- ship the snapshot-capture pipeline now and wire the recurring
  intraweek capture into the Friday orchestrator (D-06) to accumulate 2025+ trajectories; fits
  the FREE tier. Infra lands, the lift screen slips honestly to a later milestone with a written
  caveat. Runs Plans 29-02, 29-03, 29-08.
- **slip** -- honest deferral with this written record; zero spend; a first-class success state
  (D-14). Runs only this plan (29-01).

---

## 4. Lift results (backfill path only): per-target paired CLV delta, D-05 keep/drop

**METHOD.** The lift is an add-one-in paired incremental-CLV screen run IN PROCESS by
`backtest/signal_lift.py` (the Phase-28 convention). Each leg is a per-season walk-forward re-fit
via `BaseTrainer.train_and_evaluate(tune=False)` -- train `<= Y-1`, measure `Y` -- so every CLV
reading is OUT OF SAMPLE. It is never a whole-frame re-score of the currently-serving artifacts,
which would score the holdout in sample. The two legs differ by exactly the line-movement family,
and the per-game delta is merged on `game_id`, so the comparison is PAIRED. Significance comes
from the canonical `clv_significance` / `CLV_COLUMN_FOR` / `SIGNIFICANCE_ALPHA` primitives in
`backtest/diagnose.py`; nothing is re-derived here.

**The baseline INCLUDES the kept Phase-28 groups.** The screen runs with
`baseline_exclude_groups=('line_movement',)`, so injury / snap / situational stay in the baseline
leg and the delta is line-movement incremental to the POST-Phase-28 feature set. That is what
makes the Section-2 D-02 question ("is line-movement just a noisier proxy for the injury news the
market is reacting to?") answerable at all. The baseline also still contains the Friday-6PM-ET
freeze anchor (`snapshot_total` / `snapshot_spread`) and the pre-existing, historically
identically-0.0 `total_movement` / `spread_movement` columns.

**Archive coverage actually joined into gold** (regular-season games, Plan 29-06):

| Season | Covered | Note |
|---|---|---|
| 2020 | **256 / 262 ids; 244 / 269 games** | **D29-06-01 RESOLVED 2026-08-16 by quick task 260816-u0e**, with NO further paid call. The whole 2020 archive had been orphaned by a `game_id` off-by-one: `get_nfl_season_start` derived the opener as the first Thursday in September instead of the Thursday after Labor Day, so every 2020 id ran one week high and 2020 coverage was `0 / 262`. The FUNCTION was corrected in `45bff24`; the 1,780 STORED rows were re-keyed in place afterwards by replaying the old rule over `games` silver and inverting it. The 262 stored ids collapse onto 256 true ids (six early-September board listings whose provisional dates later moved merge into the id holding that game's in-week snapshots, across disjoint timestamps), all of which join `games`. Zero paid rows were lost: 1,780 rows and 1,780 distinct `(game_id, snapshot_ts)` pairs before and after. |
| 2021 | 261 / 272 | |
| 2022 | 257 / 271 | |
| 2023 | 271 / 272 | |
| 2024 | 269 / 272 | |

### 4a. The canonical screen (pre-registered): `python -m backtest.signal_lift --phase 29`

Canonical temporal config -- train 2018-2019, hp-val 2020, measure 2021-2024; 1,019 paired games.

| Target | Paired CLV delta | t | p | Group columns the model used | Covered span |
|---|---|---|---|---|---|
| WP | +0.000000 | n/a | n/a | **0 / 15** | odds-timeline 2020-06-06+, measured 2021-2024 |
| ATS | +0.147814 | +1.651 | 0.09904 | **0 / 15** | odds-timeline 2020-06-06+, measured 2021-2024 |
| OU | -0.221188 | -2.283 | 0.02266 | **0 / 15** | odds-timeline 2020-06-06+, measured 2021-2024 |

**D-13 rule as written on this grid: DROP** (significantly-negative on OU).

**That DROP is not evidence about line movement, and this readout will not pretend otherwise.**
Look at the last column: *not one* of the 15 line-movement columns was selected by *any* target.
No candidate model in this grid ever saw the signal. The reason is structural, not statistical:

- every trainer runs its own `SelectFromModel` pass on `config.train_seasons` **only**, and LOCKS
  that feature set for the entire holdout walk-forward;
- the canonical training window is **2018-2019**, and The Odds API historical floor is
  **2020-06-06**, so no 2018-2019 game can ever have a trajectory;
- measured on the rebuilt gold, all 15 columns are therefore **exactly constant (0.0) across all
  534 rows of the selection window**. A constant column has zero model importance by construction
  and can never be selected.
  - *(Corrected 2026-08-17, WR-12.)* This bullet previously added "and constant across hp-val 2020
    too, because of the D29-06-01 orphaning above". That is **no longer true**: the D29-06-01
    re-key restored real trajectories for 244 of 269 games in 2020, as the coverage table in
    Section 4 states, so 2020 is NOT constant any more. The stated cause was therefore false while
    the table two hundred lines above said the opposite -- and in a record whose whole value
    proposition is that nothing is deleted and every superseded reading is retained WITH the reason
    it was superseded, a live self-contradiction is the one failure mode that matters. The `0 / 15`
    result is **unchanged**, for the correct reason: selection runs on `config.train_seasons`
    (2018-2019) **alone**, so the hp-val fold's contents never entered the question.

So the three deltas above are **selection churn**: adding 15 constant columns to the candidate
pool perturbs the fitted importances and the selection threshold, and each target's leg locks a
slightly *different* set of the OTHER 25 features. The OU "significant" -0.221188 is a
measurement of that churn, not of market movement. The WP +0.000000 (with an undefined
t-statistic) is the clean control: WP's top-20 was unchanged, so its two legs were byte-identical.
The harness now reports this itself -- the run prints `grp_cols_used` per cell and a `NOT
MEASURED` banner -- so this failure mode cannot be published as a lift again.

**RE-RUN 2026-08-16 as the upstream-drift control, and it did not move.** After the 2020 re-key
and a full gold rebuild, this exact command was re-run. All three cells reproduce to the digit --
WP +0.000000, ATS +0.147814, OU -0.221188 -- and all three still report **0 / 15**, so the `NOT
MEASURED` banner still fires and this section's "not evidence" framing still holds.

That non-movement is doing real work, and it is why the control was run. 4a's selection window is
2018-2019, which the 2020 re-key cannot reach: the family is constant there before and after, so
no line-movement column enters any locked feature set and 4a's walk-forward training frames are
untouched by the re-key even though they span 2020. Any movement in these numbers would therefore
have been upstream data drift entering through the rebuild (which reaches nflreadpy live with no
cache), and it would have contaminated the headline grid below in a way nothing else would catch.
There was none. The gold fingerprint agrees: comparing every column, per season, before and after
the rebuild, the ONLY columns that moved are the 15 line-movement columns and `feature_timestamp`
(the build stamp, which is not a feature and is excluded from the matrices). **No modelled feature
outside the line-movement family changed at all**, so nothing in 4d needs to be discounted for
upstream drift.

**RE-RUN 2026-08-17 after the leak fix, and this time IT MOVED. 4a IS NO LONGER A CLEAN
UPSTREAM-DRIFT CONTROL, and the paragraphs above must be read with that in mind.** The figures
above are retained verbatim as the reading taken before the leak fix; they are not the current
ones. Re-run after quick task 260817-dyp:

| Target | Paired CLV delta | t | p | Group columns the model used |
|---|---|---|---|---|
| WP | +0.000000 | n/a | n/a | **0 / 15** |
| ATS | -0.136848 | -1.417 | 0.15682 | **0 / 15** |
| OU | -0.294672 | -3.045 | 0.00239 | **0 / 15** |

The ATS cell moved from +0.147814 to -0.136848 and the OU cell from -0.221188 to -0.294672. The
`0 / 15` fact is unchanged, so the "not evidence about line movement" framing above still holds
and the D-13 ruling on this grid is still DROP (OU veto). What does NOT hold any more is the
claim that 4a is insulated from this project's own edits.

**Column-level attribution for the movement, from the gold fingerprint.** The rebuild changed
exactly 16 columns per matrix and nothing else. Twelve are line-movement drift/path columns
(seasons 2020+), which 4a's 2018-2019 selection window still cannot see -- the family is constant
0.0 there, which is why `0 / 15` is unchanged. One is `feature_timestamp`, a build stamp that is
not a feature. **The remaining three are `home_rest_days`, `away_rest_days` and `rest_advantage`,
and they moved in seasons 2018-2025 -- including 2018 and 2019, which ARE 4a's selection window.**
Those are BASELINE features present in both legs, so both legs' models changed: a different
locked feature set produces a different pair of models and hence a different paired delta. That is
the whole of the movement, and it is this project's own timezone correction (WR-06), not upstream
drift.

The honest consequence: this phase no longer has a clean control separating upstream nflreadpy
drift from its own edits. What it has instead is a complete per-column, per-season fingerprint
showing that NO column outside the line-movement family and the rest-days family changed at all,
which answers the same question directly rather than by proxy.

### 4b. Coverage-window diagnostic, RE-MEASURED on the re-keyed archive: `python -m backtest.signal_lift --phase 29 --coverage-window`

To find out whether the signal is *measurable at all*, the same screen was re-run under a
diagnostic window that puts covered seasons in the selection fold: train 2021-2022, hp-val 2023,
measure **2024 only**; 255 paired games. Here the selector CAN see the family, and it chose it.

The grid below was **re-measured on 2026-08-16 against the re-keyed archive and the rebuilt gold.**
It had to be: this window's walk-forward trains on every season below 2024, which includes 2020,
whose line-movement values changed from neutral defaults to real trajectories. A preserved number
that no longer reproduces is worse than no number, so the figures are the current ones and the
originals are kept below as history rather than presented as reproducible.

| Target | Paired CLV delta | t | p | Group columns the model used |
|---|---|---|---|---|
| WP | -0.003559 | -1.506 | 0.13324 | 4 / 15 (`opening_spread`, `spread_abs_travel`, `spread_range`, `total_drift`) |
| ATS | **+1.012106** | +4.198 | **0.00004** | 4 / 15 (`opening_spread`, `spread_drift`, `spread_drift_dir`, `total_range`) |
| OU | +0.655771 | +3.352 | 0.00092 | 3 / 15 (`opening_total`, `spread_late_drift`, `total_drift`) |

**Pre-re-key readings, retained as history:** the originally published grid read WP -0.001537
(p=0.39111), ATS **+0.816061** (t=+3.562, p=**0.00044**), OU +0.281204 (p=0.10784), each with
3 / 15 columns used. Those numbers were correct for the archive as it stood; they no longer
reproduce, and nothing about them was deleted.

**D-13 rule as written on this grid: KEEP** (positive on ATS and OU, not significantly-negative
on any target).

**What this grid is NOT.** It is a diagnostic, not the pre-registered screen, and its weaknesses
are load-bearing, not decorative:

- **One measured season.** 2024 alone, 255 paired games. A single season's CLV can be regime-driven.
- **It trains on holdout seasons.** 2021-2022 selection and 2023 validation are holdout years, and
  D26-09 already records 2023-24 as partially burned. This configuration spends holdout to buy a
  measurement.
- **It was chosen after seeing the canonical result.** Nine cells now exist across three windows
  with no multiple-comparison correction. The ATS p would clear a naive Bonferroni over nine cells
  (0.0056), but a window picked post hoc is exactly the garden-of-forking-paths risk that Phase
  30's binding gate exists to settle.
- **The pre-registered grid in 4d does not corroborate it.** Measured over three seasons on a
  window registered in advance, the ATS effect is +0.151237 (p=0.25739) -- roughly a seventh of
  this cell's +1.012106, and nowhere near significance. When a single-season post-hoc window and a
  three-season pre-registered window disagree by that margin, the pre-registered one is the
  reading that counts, and this cell is best understood as what a window chosen after seeing
  results tends to produce.

**THIRD READING, 2026-08-17, after the leak fix (quick task 260817-dyp).** The grid above is now
itself a superseded reading, retained in full rather than replaced. Re-run on post-leak-fix gold:

| Target | Paired CLV delta | t | p | Group columns the model used |
|---|---|---|---|---|
| WP | -0.003521 | -1.493 | 0.13669 | 4 / 15 |
| ATS | **+0.750368** | +3.446 | **0.00067** | 4 / 15 |
| OU | +0.445762 | +2.481 | 0.01375 | 2 / 15 |

**D-13 rule as written on this third reading: KEEP** (positive on ATS and OU, none
significantly-negative). This window's walk-forward trains on every season below 2024, so it sees
the corrected 2020-2023 line-movement values AND the corrected 2018-2024 rest-days values; both
legs moved. The ATS cell fell from +1.012106 to +0.750368 -- still the largest number anywhere in
this document, and still the one the pre-registered window does not corroborate. Its three
weaknesses above are unchanged and still load-bearing, and the corrected pre-registered grid in
4d-bis now disagrees with it far more sharply than the pre-leak-fix one did: this cell is
+0.750368 where the pre-registered three-season reading is **-0.208582**.

So the sequence of readings for this diagnostic is +0.816061 (pre-re-key), +1.012106 (post-re-key,
pre-leak-fix), +0.750368 (post-leak-fix). Each was correct for the inputs it was measured on. None
is deleted.

### 4c. Pre-registration, ORIGINAL (SUPERSEDED by 4c-bis -- retained verbatim, nothing edited)

**This registration was never executed.** The window it names cannot be run at all, for a reason
discovered on the FIRST attempt to run its command and BEFORE any number from any covered window
existed. It is preserved here unedited, because deleting a registration that did not survive
contact with the code would be exactly the kind of tidy-up that makes a pre-registration
worthless. The superseding registration is 4c-bis; what changed and why is stated there.

This subsection was written and committed to git ALONE, before any number from the covered
selection window existed anywhere in this repository -- and, in fact, before the window itself
existed in the codebase at all: the `--covered-selection-window` flag and the config it needs were
added in a LATER commit. Its whole value is that `git log` can check the ordering rather than
this document asserting it. The 4b diagnostic's central weakness was a window chosen after seeing
results; repeating that would make a new grid worth no more than the old one.

**1. The command, run exactly once.** Verbatim, copy-pasteable:

```
uv run python -m backtest.signal_lift --phase 29 --covered-selection-window
```

**2. The window.** Train seasons 2018, 2019, 2020; NO hp-val season; holdout / measure seasons
2021, 2022, 2023, 2024 -- the full four-season holdout. `groups=('line_movement',)`,
`baseline_exclude_groups=('line_movement',)`, `tune=False`, anchor
`BaseTrainer.train_and_evaluate(tune=False)`. This is a NON-DEFAULT configuration. The project's
canonical training window (train 2018-2019, hp-val 2020, holdout 2021-2024) is DELIBERATELY not
mutated (D-Q2): it is a project-wide gate-configuration decision that feeds Phase 30's binding
gate and every trainer, and changing it as a side effect of a measurement would be an invisible
global change made for a local reason. The consequence is stated in advance: Phase 30 must adopt a
covered selection window DELIBERATELY, or the family remains inert under the canonical window and
should be dropped (D29-07-01).

**3. The keep/drop rule.** Exactly as `decide_group_keep` implements it today
(`backtest/signal_lift.py:462-506`): KEEP if and only if the point-estimate delta is above zero on
at least one target AND no target is significantly-negative (mean below zero with
p < `SIGNIFICANCE_ALPHA`). Whatever that function returns on the headline grid is what gets
published. The rule is not re-stated in fresh prose that could be shaded after the fact -- it is
the committed function.

**4. The measurability precondition, and the pre-committed answer if it fails.** The grid is
evidence about line movement only for targets whose `n_group_columns_selected` is above zero. If
the family is STILL selected 0/15 even under a window that can see it, that is a real and complete
finding -- the family is not competitive on train-window importance even when visible -- and it
will be published as the phase's answer in the same voice as any other outcome. The Phase-30
recommendation in that case is pre-committed here as DROP, unless the family is admitted by some
route that does not depend on train-window importance.

**5. Multiplicity.** The grid is reported RAW with the existing multiplicity note. These three
cells bring the phase to nine cells across three windows with no correction applied here. The
binding BH-FDR / p<0.05 correction stays in the Phase-30 deploy gate. A single nominally
significant cell is a screening signal, not a gate ruling.

**6. The confound tell, registered before the run.** Inside the train 2018-2020 selection window
the family is exactly 0.0 for all 534 rows of 2018-2019 and real for roughly 244 of the 269 rows
of 2020. `line_movement_coverage` -- and anything collinear with it -- can therefore be selected
as a SEASON PROXY rather than as market information. Tell: if `line_movement_coverage` appears in
a target's `group_columns_selected`, that cell is declared CONFOUNDED and is not read as market
information.

**7. Limitations accepted in advance.** 2020 is a COVID season, and it sits inside the selection
window. The 2020 rows are recovered by an in-place re-key of already-purchased data, not
re-purchased. The headline rests on a non-default configuration (item 2).

**8. Null-result clause.** A flat, null, or negative grid is a COMPLETE and SUCCESSFUL outcome and
will be published as the phase's answer with the same prominence and detail as a positive one. The
7,210 credits already spent create no obligation to find a signal, and no result will be softened,
hedged, buried, or re-run until it is favourable.

**9. No further spend.** No Odds API call of any kind is made for this measurement.

**10. What will NOT change after seeing results.** The window definition, the decision rule, the
target set, and the measurement span. If any of them must change, the change will be reported as a
NEW post-hoc diagnostic with its own label, never folded into the headline grid.

### 4c-bis. Pre-registration, SUPERSEDING (written and committed BEFORE the covered-window screen was run)

This registration replaces 4c. Like 4c it was committed ALONE, before any number from any covered
selection window existed anywhere in this repository. The original 4c above is retained verbatim.

**Why 4c had to be replaced.** Running its command for the first time aborted before a single cell
was computed:

```
ValueError: Found array with 0 sample(s) (shape=(0, 20)) while a minimum of 1 is required
by StandardScaler.        models/trainers/wp_trainer.py:311
```

4c's window carried NO hp-val season, on the reasoning that an hp-val fold is unused when the
screen runs with `tune=False`. **That reasoning was wrong.** It is true of `BaseTrainer`, where
`hp_val` feeds only `combined_train` / `combined_targets`
(`models/trainers/base.py:355-360`) inside the `if tune` branch (`base.py:361-368`). It is false
of all three CONCRETE trainers, each of which overrides `train_and_evaluate` and fits a post-hoc
conversion component on the hp-val fold OUTSIDE that branch:

| Trainer | hp-val consumer | file:line | Behaviour with an empty fold |
|---|---|---|---|
| WP | Platt/isotonic probability calibrator | `wp_trainer.py:310-326` | hard crash (StandardScaler on 0 samples) |
| ATS | `ResidualDistributionConverter` on residuals | `ats_trainer.py:244-250` | runs, but `residual_std = np.std([]) = NaN` |
| OU | same pattern | `ou_trainer.py:244-252` | same |

So the original window would have produced one crash and two silently degenerate models. The
error was in the plan's premise, not in the measurement; it is recorded here rather than quietly
corrected. `TemporalSplitConfig.validate` now rejects an empty hp-val fold by name.

**1. The command, run exactly once.** Verbatim, copy-pasteable:

```
uv run python -m backtest.signal_lift --phase 29 --covered-selection-window
```

**2. The window.** Train seasons 2018, 2019, 2020; hp-val season 2021; holdout / measure seasons
2022, 2023, 2024. Roughly 764 paired games. `groups=('line_movement',)`,
`baseline_exclude_groups=('line_movement',)`, `tune=False`, anchor
`BaseTrainer.train_and_evaluate(tune=False)`.

The cost of this change is stated plainly and accepted in advance: roughly 25% less sample than
4c's window (about 764 paired games instead of about 1,019), and 2021 -- a covered season -- is
spent as a calibration fold rather than measured. What it buys is the only version of this window
in which all three targets are measured by models of the SAME CLASS as the canonical 4a grid's:
no uncalibrated WP, no NaN-scale residual converter, no dropped target, and no patch to the
trainers that would silently change what "the model" means and make the headline
non-comparable. Temporal ordering is strict and unchanged --
max(train)=2020 < hp-val=2021 < min(measure)=2022 -- so no season is both trained on and measured.

This remains a NON-DEFAULT configuration. The project's canonical training window is still
DELIBERATELY not mutated (D-Q2), so Phase 30 must adopt a covered selection window DELIBERATELY,
or the family remains inert under the canonical window and should be dropped (D29-07-01).

**3. The keep/drop rule.** Unchanged from 4c, and unchanged from the committed code: exactly as
`decide_group_keep` implements it (`backtest/signal_lift.py:462-506`), KEEP if and only if the
point-estimate delta is above zero on at least one target AND no target is significantly-negative
(mean below zero with p < `SIGNIFICANCE_ALPHA`). Whatever that function returns on the headline
grid is what gets published. The rule does not become easier to satisfy because the window moved.

**4. The measurability precondition, and the pre-committed answer if it fails.** Unchanged from
4c. The grid is evidence about line movement only for targets whose `n_group_columns_selected` is
above zero. If the family is STILL selected 0/15 even under a window that can see it, that is a
real and complete finding -- the family is not competitive on train-window importance even when
visible -- and it will be published as the phase's answer in the same voice as any other outcome.
The Phase-30 recommendation in that case is pre-committed as DROP, unless the family is admitted
by some route that does not depend on train-window importance.

**5. Multiplicity.** Unchanged from 4c. The grid is reported RAW with the existing multiplicity
note. These three cells bring the phase to nine cells across three windows with no correction
applied here. The binding BH-FDR / p<0.05 correction stays in the Phase-30 deploy gate. A single
nominally significant cell is a screening signal, not a gate ruling.

**6. The confound tell, registered before the run.** Unchanged from 4c. Inside the train 2018-2020
selection window the family is exactly 0.0 for all 534 rows of 2018-2019 and real for roughly 244
of the 269 rows of 2020, so `line_movement_coverage` -- and anything collinear with it -- can be
selected as a SEASON PROXY rather than as market information. Tell: if `line_movement_coverage`
appears in a target's `group_columns_selected`, that cell is declared CONFOUNDED and is not read
as market information.

**7. Limitations accepted in advance.** 2020 is a COVID season and it sits inside the selection
window. The 2020 rows are recovered by an in-place re-key of already-purchased data, not
re-purchased. The headline rests on a non-default configuration. The measured span is three
seasons, not four, and 2023-2024 is recorded as partially burned holdout (D26-09).

**8. Null-result clause.** Unchanged from 4c. A flat, null, or negative grid is a COMPLETE and
SUCCESSFUL outcome and will be published as the phase's answer with the same prominence and detail
as a positive one. The 7,210 credits already spent create no obligation to find a signal, and no
result will be softened, hedged, buried, or re-run until it is favourable.

**9. No further spend.** No Odds API call of any kind is made for this measurement.

**10. What will NOT change after seeing results.** The window definition, the decision rule, the
target set, and the measurement span. If any of them must change, the change will be reported as a
NEW post-hoc diagnostic with its own label, never folded into the headline grid.

### 4d. HEADLINE, PRE-LEAK-FIX (SUPERSEDED by 4d-bis -- retained verbatim, every number intact)

**Why this reading was superseded.** It was measured on gold that contained POST-KICKOFF, IN-PLAY
line values for three games. `features/line_movement.py` derived each game's freeze as the most
recent Friday 18:00 ET at or before the kickoff DATE and never capped it at kickoff, so a
Friday-afternoon kickoff was fenced AFTER it had started -- finding CR-01 of the Phase-29 deep code
review. `2023_W12_MIA@NYJ` (Black Friday, kickoff Fri 2023-11-24 15:00 ET) admitted a Friday 17:55
ET snapshot taken while the game was in play, by which point its spread had moved 9.5 to 20.5.
`2020_W16_MIN@NO` and `2024_W13_LV@KC` were affected the same way.

The grid below is therefore a measurement taken on contaminated inputs. It is retained here in
full, with every figure exactly as published on 2026-08-16, because deleting or silently
overwriting a superseded reading is how a record stops being one. The corrected reading is 4d-bis.

Reproduce with, verbatim:

```
uv run python -m backtest.signal_lift --phase 29 --covered-selection-window
```

Run ONCE, on 2026-08-16, against the re-keyed archive and the rebuilt gold, under the rule
committed in 4c-bis before the run. 764 paired games per target.

<!-- Machine-readable ordering markers. Both commits touch ONLY this file, contain no measured
number from any covered selection window, and are git ancestors of the commit that published the
grid below. The guard test in tests/unit/test_line_movement_readout_md.py resolves these SHAs and
checks that with `git merge-base --is-ancestor`, so the ordering is checkable rather than
asserted. -->
pre_registration_commit: 05878f671cc304ea280aa6341dcec2b00ad1d86a
superseding_pre_registration_commit: 5660ee387d20ba5c96441a3c4a1033b9a94bc4a1

| Target | Paired CLV delta | t | p | 95% CI | Group columns the model used |
|---|---|---|---|---|---|
| WP | -0.001592 | -0.975 | 0.32981 | [-0.004798, +0.001613] | 2 / 15 (`opening_spread`, `spread_drift_dir`) |
| ATS | **+0.151237** | +1.133 | 0.25739 | [-0.110701, +0.413175] | 5 / 15 (`opening_spread`, `spread_drift`, `total_abs_travel`, `total_drift_dir`, `total_late_drift`) |
| OU | +0.001429 | +0.011 | 0.99107 | [-0.249192, +0.252050] | 4 / 15 (`opening_total`, `total_drift`, `total_drift_dir`, `total_range`) |

**D-13 rule as written on this grid: KEEP** -- positive point-estimate on ATS and OU, and no
target significantly-negative. That is what `decide_group_keep` returns and it is published
unchanged.

**The result is FLAT, and that is the finding.** Read the grid rather than the ruling. The
family was finally VISIBLE to the selector -- 2, 5 and 4 of its 15 columns were chosen, the `NOT
MEASURED` banner does not fire, and every one of these cells is genuine evidence about line
movement rather than selection churn. Given the chance to help, it did essentially nothing. OU's
+0.001429 (p=0.99107) is indistinguishable from zero. WP is slightly negative and not significant.
ATS's +0.151237 is the largest cell and it carries p=0.25739 with a confidence interval
comfortably spanning zero -- and it is, to three decimal places, the same magnitude as the
+0.147814 that 4a produced from models that never saw the family at all.

The KEEP is therefore a weak, permissive pass by a screening rule, not a result. The rule asks
only for one positive point estimate and no significant harm; a family that does nothing at all
passes it. Read together with 4a and 4b, the honest summary is: **where line movement could not be
measured it looked like nothing, where it was measured on one post-hoc season it looked large, and
where it was measured on three pre-registered seasons it is flat.**

**The confound tell did not fire.** `line_movement_coverage` appears in NO target's selected set,
so no cell here is discounted as a season proxy (4c-bis item 6). The selected columns are led by
level (`opening_spread`, `opening_total`) and path (`total_abs_travel`, `total_range`,
`total_late_drift`), broadly the families Section 2 predicted would be the non-redundant ones --
which makes the flatness harder to dismiss, not easier: the selector picked the columns the
hypothesis said to pick, and they still did not pay.

**What this grid does and does not license.** It is a SCREEN. It licenses carrying the family into
Phase 30's binding gate under a covered selection window, and it licenses saying that the large
4b ATS effect is not corroborated. It does NOT license treating line movement as an edge, and no
part of it is a gate ruling. Three measured seasons (2022-2024) on a non-default configuration,
reported raw with no multiplicity correction, on a holdout that D26-09 already records as
partially burned, is a screen -- not a verdict.

### 4d-bis. HEADLINE, CORRECTED: the same pre-registered window, re-run on post-leak-fix gold

Reproduce with, verbatim:

```
uv run python -m backtest.signal_lift --phase 29 --covered-selection-window
```

Run ONCE, on 2026-08-17, after quick task 260817-dyp rebuilt gold on corrected inputs. 764 paired
games per target -- the same sample as 4d.

**THE RULE WAS NOT RE-OPENED.** Sections 4c and 4c-bis are BYTE-UNCHANGED by this correction, and
that is checkable rather than asserted: `git diff 5660ee3 HEAD -- LINE-MOVEMENT-READOUT.md` shows
no change inside either block, and the committed `TestPreRegistrationOrdering` still resolves both
`pre_registration_commit` SHAs and confirms they are ancestors of HEAD carrying no headline number
in their own diffs. This is a re-measurement of the SAME hypothesis under the SAME rule on
corrected inputs, not a new experiment. Re-registering after a flat result was already known is
exactly what the pre-registration discipline exists to prevent.

One bookkeeping note recorded HERE rather than by editing the frozen text: quick task 260817-dyp's
WR-02 fix added a `paired_sufficient` field and a NOT-MEASURED refusal arm to
`backtest/signal_lift.py`, which shifts the line numbers that 4c-bis item 3 cites as
`backtest/signal_lift.py:462-506`. The cited FUNCTION (`decide_group_keep`) and its behaviour at
or above `MIN_CLV_SAMPLE` are unchanged; only the line numbers moved.

| Target | Paired CLV delta | t | p | 95% CI | Group columns the model used |
|---|---|---|---|---|---|
| WP | -0.001594 | -0.981 | 0.32680 | [-0.004782, +0.001595] | 2 / 15 (`opening_spread`, `spread_drift_dir`) |
| ATS | **-0.208582** | -1.646 | 0.10021 | [-0.457372, +0.040209] | 4 / 15 (`opening_spread`, `spread_drift`, `total_drift_dir`, `total_late_drift`) |
| OU | -0.003519 | -0.025 | 0.97981 | [-0.276446, +0.269408] | 3 / 15 (`opening_total`, `total_abs_travel`, `total_drift`) |

**Every cell cleared the sufficiency threshold.** `n_paired` is 764 on all three targets against
`MIN_CLV_SAMPLE = 10`, and each cell reports `paired_sufficient: true` and `measurable: true`. The
NOT-MEASURED refusal arm added by WR-02 was therefore structurally unreachable in this run: the
ruling below comes from the same KEEP/DROP arms that produced 4d, not from the new one.

**D-13 rule as written on this grid: DROP** -- no positive point-estimate on any target. That is
what `decide_group_keep` returns and it is published unchanged. Note the ruling FLIPPED: the
pre-leak-fix grid returned KEEP on the strength of ATS +0.151237 and OU +0.001429, and both of
those point estimates are now negative.

**The corrected result is NEGATIVE, and that is the finding.** It is reported here with exactly
the prominence and detail a positive result would have received. All three targets are below zero.
No cell is significant: ATS at p=0.10021 is the closest and its confidence interval
[-0.457372, +0.040209] still spans zero, so this is not evidence that line movement HARMS CLV
either. The honest reading is that the family is flat-to-slightly-negative once the in-play values
are removed, and the direction of the correction is the informative part: **removing three games'
post-kickoff line values moved the ATS cell by -0.359819, from +0.151237 to -0.208582.** The
apparent ATS lift in 4d was substantially an artifact of the leak.

**Why three corrected games can move a 764-game paired delta by that much.** Two mechanisms, and
neither is a bug. Gold is expanding-window z-scored, so a corrected raw value shifts every
SUBSEQUENT row's normalized value within its season -- the fingerprint shows twelve line-movement
columns moving across seasons 2020-2025, not three rows. And the corrected inputs change which
features the trainer's `SelectFromModel` pass locks: ATS went from 5 of 15 group columns to 4, and
its selected set changed composition (`total_abs_travel` out, `total_drift_dir` and
`total_late_drift` in), so the candidate leg is a different model, not the same model on slightly
different numbers.

**The confound tell did not fire.** `line_movement_coverage` appears in NO target's selected set,
so no cell here is discounted as a season proxy -- the tell is applied exactly as 4c-bis item 6
registered it in advance.

**What changed in the inputs, and what did not.** Two of this project's own corrections reached
gold, and nothing else did:

- **CR-01, the leak fix.** Three games' raw line-movement values changed, reproducing the
  independently computed corrected table exactly: `2020_W16_MIN@NO` total 51.5 to 50.5 (drift
  -1.0), spread -6.0 to -7.0; `2023_W12_MIA@NYJ` total 46.5 to 41.0 (drift -5.5), spread -2.5 to
  9.5 (drift +12.0, down from a leaked +23.0); `2024_W13_LV@KC` total 46.0 to 42.5 (drift -3.5),
  spread -7.5 to -13.0.
- **WR-06, the timezone normalization.** 1,926 stale `games.kickoff_et` rows were shifted so the
  stored instant is true, in BOTH silver copies, with the row sets unchanged. This moved
  `home_rest_days`, `away_rest_days` and `rest_advantage` in seasons 2018-2025.

**Fingerprint attribution: nothing is unexplained.** Comparing every column, per season, before
and after the rebuild, exactly 16 columns moved in each of the three matrices: twelve
line-movement drift/path columns (CR-01), three rest-days columns (WR-06), and
`feature_timestamp`, which is a build stamp and not a feature. **No other modelled feature moved
at all, so there is no named upstream-drift column to discount this grid for.** The weekday family
(`thursday_game`, `monday_game`, `saturday_game`, `short_week`, `game_day_of_week`) and the
bye/short-rest flags did not move, which is the check that the timezone shift preserved every
kickoff DATE. `opening_total`, `opening_spread` and `line_movement_coverage` also did not move:
the kickoff cap removes only the in-play tail of a trajectory, never its earliest snapshot.

**One measured qualification, recorded because it cuts against the headline.** In the ATS matrix
the 2023 `spread_drift` column did NOT change, and `2023_W12_MIA@NYJ` still carries the same gold
value of +1.894965 that the code review flagged. That is not residual leakage. Gold winsorizes at
the 1st/99th percentile before normalizing, and that bound for `spread_drift` is +5.5, so the
leaked +23.0 and the corrected +12.0 both clamp to +5.5 and produce an identical cell. The
consequence is worth stating plainly in both directions: for that one column-season the model
never saw the leak, because winsorization had already absorbed it -- and the corrected +12.0 is a
genuinely large, genuinely legitimate pre-kickoff move, so a large value there is now CORRECT
rather than contaminated. The leak's effect on gold was real but smaller than the raw values
suggest wherever winsorization was already clamping.

**Leak-purge verification, on gold rather than in memory.** For every Friday-kickoff game present
in gold, the admitted-snapshot window was recomputed and checked against that game's own kickoff:
**zero games admit a snapshot at or after kickoff** (8 Friday-kickoff games in gold, 4 of them with
archive rows). Gold is 6,263 rows at widths 209 / 210 / 209 with no `_x` / `_y` merge-collision
columns. The paid archive was untouched throughout: 9,957 rows and 9,957 distinct
`(game_id, snapshot_ts)` pairs, per-season 2020=1,780 / 2021=1,219 / 2022=1,452 / 2023=2,759 /
2024=2,747, asserted through `load_dataframe` before and after every destructive step. No Odds API
call was made at any point.

**What this grid does and does not license.** Unchanged in kind from 4d: it is a SCREEN. It
licenses carrying nothing forward on the strength of a positive number, because there is no
positive number. It licenses saying that the pre-leak-fix KEEP rested on a contaminated ATS cell.
It does NOT license calling line movement harmful -- no cell is significant. Three measured
seasons on a non-default configuration, reported raw with no multiplicity correction, on a holdout
D26-09 already records as partially burned, remains a screen and not a verdict.

### 4e. Ruling, and the D-02 question answered

> **SUPERSEDED IN ITS CONCLUSION, 2026-08-17 -- see 4e-bis below.** Everything from here to the
> end of 4e was written against the PRE-LEAK-FIX headline grid (4d) and is retained verbatim as
> the record of that ruling. Its structural analysis of the calendar blocker still stands; its
> ruling of a flat KEEP does not, because the corrected grid is negative on all three targets and
> the rule returns DROP.

**The binding constraint was never redundancy -- it was the calendar.** Section 2 predicted the
go/no-go would turn on whether line movement carries information the model does not already have.
The actual blocker is that the archive floor (2020-06-06) lands entirely *after* the canonical
feature-selection window (2018-2019). Under the configuration the project trains today, the
line-movement family is **inert**: it sits in gold, adds 15 columns to all three matrices, and can
never be selected into a model no matter how informative it is.

That calendar blocker has now been REMOVED rather than merely described: the 2020 archive was
re-keyed in place (no paid call), gold was rebuilt, and a covered selection window was registered
in advance and run. So the question Section 2 actually asked can finally be answered with a
measurement instead of a structural excuse. **The answer is: measured over three pre-registered
seasons, the line-movement family adds essentially nothing.**

**On the D-02 priced-in risk: the family is CHOSEN, and it does not pay.** The priced-in worry
from Section 2 was that the market moves on the same injury news the Phase-28 features already
encode, making line movement a noisier proxy for signal the model can read directly. The headline
grid is the cleanest test of that yet run: the baseline contained the freeze anchor AND the
Phase-28 injury / snap / situational signal, the selector still chose 2-5 line-movement columns
per target over the alternatives, and the resulting CLV delta was flat (OU +0.001429 at p=0.99107;
ATS +0.151237 at p=0.25739; WP slightly negative). Being selected is not the same as being worth
something, and this is what that distinction looks like when it is finally measurable.

The Section-2 prediction was that only the opening **LEVEL** and the **PATH** would be
non-redundant, not net drift. The selected columns do line up with that -- led by `opening_spread`
/ `opening_total` and by `total_abs_travel` / `total_range` / `total_late_drift` -- so the
plausibility argument picked the right columns. They simply did not produce CLV. The most likely
reading, consistent with Section 2's own priced-in caveat, is that the freeze anchor plus the
Phase-28 injury features already carry most of what the market's movement encodes.

**Ruling (SIG-04): SCREENED, with a flat result and a named condition.** The rule as written says
CARRY the line-movement family to Phase 30, and that is what happens -- `decide_group_keep`
returns KEEP on the headline grid and is published unchanged. But the rule is permissive by
design, and a family that does nothing passes it. Nothing here establishes an edge.
Two things are now true and both belong in Phase 30's decision:

1. **The 2020 re-key is DONE** (D29-06-01, resolved 2026-08-16 by quick task 260816-u0e, no paid
   calls; 1,780 rows recovered, 244 of 269 games covered).
2. **The verdict rests on a NON-DEFAULT configuration (D-Q2).** The canonical training window was
   deliberately not mutated. If Phase 30 keeps train 2018-2019, the family remains **inert** --
   it cannot enter a candidate model at all, and the flat headline above is the *best* case rather
   than the expected one. So Phase 30 must either adopt a covered selection window DELIBERATELY as
   its binding config, or DROP the family. Carrying it indefinitely under a window that cannot see
   it is the one option ruled out.

Given the headline is flat, the honest default recommendation into Phase 30 is DROP unless the
binding gate is run under a covered selection window and produces something the screen did not.

**Cost against learning, stated plainly.** The archive cost **7,210 credits** of a 20,000-credit
$30 month (12,790 unused), and it is final -- no further paid call was needed for any of the work
above, including the re-key. What that bought: a real 2020-2024 trajectory archive (9,957 rows);
the measured fact that market movement is genuinely non-degenerate where the old `total_movement`
column was identically 0.0; a pre-registered three-season reading that the family is FLAT once it
is visible to the selector, which is a genuine answer where before there was only a structural
excuse; the demonstration that the eye-catching single-season 4b effect does not survive a
pre-registered window; and -- the finding with the most leverage -- the discovery that the
project's feature-selection window silently excludes any signal whose data floor is later than
2019. That last one applies to every future signal purchase, not just this one. A flat result was
accepted in advance as a complete outcome, and it is what came out; what would NOT have been
acceptable is publishing the 4a grid as a lift, or publishing 4b's +1.012106 as the answer.

### 4e-bis. Ruling, CORRECTED (2026-08-17, on the post-leak-fix headline grid)

**Ruling (SIG-04): SCREENED, with a NEGATIVE result.** The rule as written returns DROP on the
corrected headline grid -- no positive point-estimate on any target -- and that is published
unchanged. The line-movement family is SCREENED and its screening outcome is negative. It is not
shipped, not serving, and not established as an edge; nor is it established as harmful, since no
cell reaches significance.

**What changed from 4e, and why.** 4e ruled a flat KEEP on ATS +0.151237 and OU +0.001429. Those
cells were measured on gold holding post-kickoff, in-play line values for three games. Corrected,
they are ATS -0.208582 and OU -0.003519, and the ruling flips to DROP. The single largest
contributor is ATS, which moved -0.359819. So the earlier KEEP was carried by a cell that the leak
was inflating.

**The D-02 question, answered again on corrected inputs, with the same answer only stronger.**
Section 2 asked whether line movement carries information the model does not already have, or is a
noisier proxy for the injury news the Phase-28 features already encode. The baseline contained the
freeze anchor AND the Phase-28 injury / snap / situational groups; the selector still chose 2, 4
and 3 line-movement columns per target over the alternatives; and the resulting CLV deltas are all
negative. Being selected is still not the same as being worth something. The Section-2 prediction
that only the opening LEVEL and the PATH would be non-redundant is still borne out in WHICH columns
get picked -- `opening_spread`, `opening_total`, `total_abs_travel`, `total_late_drift` -- and they
still do not pay.

**Recommendation into Phase 30: DROP.** The pre-leak-fix text recommended DROP unless the binding
gate is run under a covered selection window and produces something the screen did not. That
recommendation stands and is now firmer, because the screen's own best window produces a negative
number rather than a flat one. The D-Q2 point is unchanged: under the canonical 2018-2019 training
window the family is inert and cannot enter a candidate model at all, so carrying it indefinitely
under a window that cannot see it remains the one option ruled out.

**Cost against learning, restated.** The 7,210 credits are unchanged and final; no paid call was
made by this correction. What the correction added to what the archive bought: a temporal leak
found and closed before it reached Phase 30's binding gate; the measured fact that the earlier
ATS lift was substantially an artifact of that leak; one documented timezone contract under the
whole feature layer; and a corrected pre-registered reading that answers the SIG-04 question in
the negative. A negative result was accepted in advance as a complete outcome (D-14), and it is
what came out.

---

## 5. Honest scope-down / slip note (if applicable) -- carried to Phase 30, not a deploy outcome

The branch was **full-backfill** and it ran end to end: Plans 29-02 through 29-07 all landed. The
archive was bought, the builder was written with a per-game ET freeze fence and a three-part
leakage proof, gold was rebuilt to 209/210/209 columns with the family surviving `combine_features`
into all three matrices, and the lift screen was registered and run. Nothing slipped for lack of
data or lack of budget.

**What DID scope down is the strength of the verdict, and that is recorded here rather than
smoothed over.** The pre-registered screen (4a) could not measure the signal, because the archive
floor lands after the canonical feature-selection window. The measurement that could be made (4b)
rests on a single season and a post-hoc window. So Phase 29 ends with a *conditional carry* and a
named prerequisite, not with a measured four-season lift number. Two follow-ups are recorded in
the phase's `deferred-items.md` rather than being quietly done here: the 2020 re-key (no paid
calls) and the Phase-30 training-window decision.

**UPDATE 2026-08-16 (quick task 260816-u0e): both follow-ups are now closed, and the phase has a
measured number.** The paragraph above describes the state at the end of Plan 29-07 and is left
standing as the record of it. Since then: the 2020 re-key was done in place with no paid call
(D29-06-01 RESOLVED), gold was rebuilt, a covered selection window was registered in advance and
run once, and Section 4d carries a pre-registered THREE-season grid. So Phase 29 no longer ends on
a conditional carry with no number -- it ends on a FLAT measured result. The one thing that did not
change is the D-Q2 consequence: the number rests on a non-default configuration, so the Phase-30
training-window decision (D29-07-01) is still open and still the thing that decides whether this
family is worth anything at all.

**Framing, unchanged from Section 3.** The line-movement signal is SCREENED and carried to Phase
30 for the binding gate. It is not shipped, not serving, and not established as an edge. Phase 29
screens; Phase 30 rules. A null result would have been a first-class success outcome (D-14) and was
accepted as such in advance -- the screen, not optimism, rules (D-10). It is worth saying plainly
that this is now the outcome that actually occurred, and it is being reported with the same
prominence a positive one would have received.

**UPDATE 2026-08-17 (quick task 260817-dyp): the measured result is now NEGATIVE, not flat, and
the reason is a temporal leak that was in gold when the paragraphs above were written.** The
Phase-29 deep code review found that each game's freeze fence was derived from its kickoff DATE
and never capped at kickoff, so a Friday-afternoon kickoff was fenced after the game had started.
Three games in the archive carried in-play lines into gold. The fence is now capped at one second
before kickoff, demonstrated by a seven-weekday leakage test that was RED against the pre-fix code
before the fix landed; the `games.kickoff_et` timezone contract it compares against was settled
first, in both silver copies; gold was rebuilt; and the pre-registered covered-window screen was
re-run ONCE under the rule already committed in `5660ee3`, which was not re-opened.

The corrected headline is WP -0.001594 / ATS -0.208582 / OU -0.003519 over the same 764 paired
games, and the rule returns DROP. The recommendation into Phase 30 is DROP. Everything above this
paragraph is left standing as the record of what was believed before the leak was found, and every
superseded grid is retained in place with the reason it was superseded. Nothing was deleted, and
the negative outcome is reported in the same voice the flat one was.
