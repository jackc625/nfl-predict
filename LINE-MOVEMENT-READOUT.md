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
selected_branch: PENDING

**Status:** PENDING owner checkpoint (Task 2). The tiered cost menu (Section 1) and the D-02
plausibility argument (Section 2) are recorded; the owner now makes the live go/no-go call with
the real numbers in hand (D-01: NO pre-registered dollar ceiling) and selects exactly one
branch. All three outcomes are honest, complete, successful phases (D-14, SC3). On selection,
this marker is overwritten with the chosen literal token and the rationale is recorded here in
screen-not-deploy language; SC1 is satisfied the moment the decision is recorded, before any
29-02+ build runs.

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

Pending backfill path (Plan 29-07), else N/A. On the full-backfill branch the add-one-in
in-process walk-forward CLV lift screen (`backtest/signal_lift.py`, the Phase-28 convention)
measures the line-movement group against the activated baseline per target (WP / ATS / OU);
a kept group is screened / carried to Phase 30 for the binding gate, never shipped here. On
forward-collect-only or slip this section stays N/A (no historical archive to backtest this
milestone).

---

## 5. Honest scope-down / slip note (if applicable) -- carried to Phase 30, not a deploy outcome

Pending the owner decision (Task 2). If the branch is forward-collect-only, this section records
that the snapshot-capture infra lands now and the lift screen / historical backtest slips
honestly to a later milestone (D-05 / D-14) -- a complete, successful phase, not "incomplete".
If the branch is slip, this section records the honest deferral with zero spend. Either way the
language stays screen-not-deploy: the line-movement signal is screened / carried to Phase 30 for
the binding gate, never asserted as an edge on the basis of this spike.
