# Broadcast UI Redesign -- Design Spec

Date: 2026-10-01
Branch: `redesign/broadcast-ui` (worktree `C:\Users\jackc\Code\nfl-predict-redesign`)
Status: awaiting owner review
Approved mockups: `docs/superpowers/specs/2026-10-01-broadcast-ui-mockups/` (open the HTML files in a browser)

---

## 1. Goal

Make the web dashboard look like a finished product rather than a default template, make it quicker to find what matters each week, and make the game cards readable at a glance -- without weakening any honesty rule the site enforces today.

Success means:

1. Every page uses one consistent dark "Broadcast" look (TV-score-bug style).
2. The top nav has 5 items instead of 7, and every old URL still works by redirecting.
3. On This Week, the week's bets are the first thing on the page, and each game card shows win, spread and total -- model beside market -- without awkward wrapping.
4. Every honesty label, disclaimer and evidence class that exists today still exists, with the same meaning, in the redesigned pages. The full original wording stays in the page HTML (behind a "Why?" toggle where condensed).
5. The production pipeline, the models, the bet-selection rule and every stored number are untouched. This is a presentation change only (plus three small read-only/presentation additions listed in section 9).

## 2. Decisions the owner made (2026-10-01 brainstorm)

| # | Decision |
|---|----------|
| O1 | Motivation: all four -- dated look, hard to navigate, unreadable cards, portfolio quality. |
| O2 | Weekly-use pages are This Week (+ game detail), Bets, Season. The four analysis pages are merged into fewer pages. |
| O3 | Merge split: **Track Record** (how it has done) and **How It Works** (why it predicts what it does). Nav: This Week, Bets, Season, Track Record, How It Works. Old URLs redirect. |
| O4 | Honesty labels and disclaimers: **condensed one-liner + "Why?" expander** that reveals the full original text. |
| O5 | Visual direction: **A3 "Broadcast"** (dark, score-bug team blocks, condensed italic display type, first-down-line yellow accent). |
| O6 | No team logos -- team-colour blocks with abbreviations. |
| O7 | Dark theme only (no light mode, no toggle). |
| O8 | Charts: restyle Plotly with one dark theme, plus a few hand-built visuals. |
| O9 | This Week layout: **L3** (headliner bet cards on top, full slate below) **with L1's TV-window tags** grouping the slate. |
| O10 | Bets page, game detail, Season: approved as mocked. |
| O11 | Build on a separate branch in a worktree; merge only after Plan 33-18 closes (section 12). |

## 3. Scope

In scope:

- `web/templates/**` (all pages and components), `web/static/input.css`, `web/static/css/custom.css`, recompiled `web/static/css/tailwind-compiled.css`, new self-hosted fonts under `web/static/fonts/`.
- `api/routes/pages.py`, `api/routes/fragments.py` (new pages, redirects, context for the headliners and TV windows).
- `api/charts/**` (dark chart theme only -- same charts, same ids, same data).
- `api/services.py` (one new read-only getter, section 9), `api/season_metrics.py` + `api/charts/season.py` (per-week record added to the existing season KPI blob, section 9).
- A small presentation helper for team colours and kickoff windows (section 9).
- The web-layer tests that assert styling or page structure (section 11).

Out of scope (not touched): ingestion, features, gold, models, artifacts, the blend, the gate, the bet selector and its EV floor, `backtest/**` logic, the daily pipeline steps, the cache schema of any existing table, exports' content, and every honesty rule's meaning.

Non-goals: light mode, team logos, a new chart library, new analysis content beyond a short methodology section, vendoring HTMX/Plotly (they stay on their current CDNs).

## 4. Constraints that must survive (behavioural contracts)

These are enforced today by tests and owner rulings. The redesign keeps every one; only the styling around them changes.

**Honesty wording (kept word-for-word, in the HTML):**
- Not-wagering-advice banner text, including "1 unit = 1% of a notional bankroll".
- The old-rule label sentence "Built under the old rule on inputs later found defective; not evidence." and its date "2026-09-15".
- The 13 suppression-reason labels and help lines, the suppressed-section caption, "(no reason recorded)".
- The three provenance labels ("Contaminated split", "Old rule -- 2025, not evidence", "Live forward record"), still defined only in `components/_provenance_badge.html`.
- Tracker headings, captions, the push footnote, "A negative return here is the measurement, not a display problem.", "Nothing graded yet", "not measured".
- The per-game lock sentence; no week-level lock claim; no "friday"; no `$` on /bets.
- Every /bets state heading and recovery text, with `generate_bet_list.py` named before `populate_cache.py`.

**Colour meaning (kept, and made stricter):**
- Green and red mean a realised result only: a bet won/lost, a pick correct/incorrect, a recorded W/L, a realised return. They are never used for a pre-game edge, a confidence level, an EV band or an evidence label.
- EV band, provenance and old-rule labels stay monochrome (grey scale; weight encodes size).
- The yellow accent is brand/emphasis only (active nav, the pick, edge chips, section tags). It carries no win/loss meaning.
- Error red stays distinct from the grey empty state.
- Outcome colours keep Tailwind's `green-*` / `red-*` class names (dark-friendly shades), so the existing hue guard tests stay meaningful instead of passing vacuously on renamed tokens.

**Structure:**
- Ids: `#game-grid`, `#bets-content`, `#bets-loading`, `bets-failure-template`, `season-error-template`, `suppressed-candidates`, `export-buttons`, the `*-select` ids, `tab-wp|ats|ou`, `feature-chart`, `#performance-content`, `#betting-content`, `#season-content`.
- Data attributes: `data-chart-id`, `data-tracker-block`, `data-figure-group`, `data-provenance`, `data-validation-type`, `data-missing-games`, `data-old-rule-label`, `data-utc`.
- Suppressed candidates stay a native, collapsed-by-default `<details>` with the caption inside `<summary>`; rows stay in the HTML while collapsed.
- One `<section>` per tracker block; pushes structurally outside the hit-rate group; declared block order (weakest evidence to strongest).
- Live bets ordered by EV with the existing tie-break; no sort control; page order equals export order.
- HTMX wiring unchanged: endpoints, targets, `hx-include`, the /bets timeout + `hx-sync` + failure handlers, fragment responses without `<html>`/`<nav>`.
- Cache-Control on every page exactly as today; malformed params fall back to defaults (200, never 500).
- Zero computation of metrics in the request path (UIAP-01). Grouping and formatting are presentation; aggregation is not.
- Number formatting stays in `components/_prediction_values.html` (win_prob, margin, total, market_wp_missing). No new sign logic anywhere -- the spread sign convention has caused real bugs before, so the redesign reuses the existing macros and the existing `side_line` macro on /bets.
- `components/_game_card.html` must still render with only `game=` in a plain Jinja environment (no app globals/filters); new fields read with `|default(...)` fallbacks.
- `components/_old_rule_label.html` uses only plain (non-variant) classes: a test requires each of its classes to appear literally in `tailwind-compiled.css`, which escaped variant classes like `md:x` do not.

## 5. Design system

**Look:** dark navy-black background with a faint vertical gradient; panels one step lighter; condensed italic display type; slanted (skewed -12 degrees) team-colour blocks, tags and tabs, with the text inside counter-skewed so it reads upright; a single yellow accent.

**Tokens** (defined once in `web/static/input.css` `@theme`, used everywhere):

| Token | Value | Use |
|-------|-------|-----|
| `ink` | `#0B0F17` | page background, text on yellow |
| `ink-2` | `#0E1320` | gradient end |
| `panel` | `#151B29` | cards, panels |
| `panel-2` | `#1D2436` | team-name strips, controls |
| `line` | `rgba(255,255,255,.08)` | dividers |
| `fg` | `#F3F5F9` | primary text |
| `muted` | `#8A93A8` | secondary text, labels |
| `dim` | `#5D667C` | absent values ("No line") |
| `accent` | `#FFD400` | brand/emphasis (section 4) |
| outcome | Tailwind `green-500`/`red-500` family | realised results only |

Contrast: all body text meets WCAG AA on `ink`/`panel`; `muted` on `panel` is used only for labels at 11px+ uppercase or 12px+.

**Type** (self-hosted woff2 under `web/static/fonts/`, SIL OFL, with `@font-face` in `input.css`; no runtime dependency on Google Fonts):
- Display: Barlow Condensed 600/700/800 + italic 700/800 -- headings, team abbreviations, big numbers, tags, tabs.
- Body: Inter 400-700.
- Numbers in tables and lines: JetBrains Mono 500/700 (tabular figures).

**Components** (Tailwind utilities plus a small set of component classes declared in `input.css` `@layer components`, so they land in the compiled CSS):
- `team-block` (skewed colour block + abbreviation), `score-bug` (two team rows: block, name strip, value cell), `tag` (yellow skewed section tag; `tag-ghost` outline variant), `skew-tab` (tabs/selectors), `edge-chip` (yellow skewed chip; `edge-chip-soft` outlined), `panel`, `stat-tile` (scoreboard tile with coloured top rule), `bet-slip`, `honesty-note` (the condensed one-liner `<details>` used by the not-advice banner and the old-rule label), monochrome `band-*` and `evidence-chip` styles.

**Team colours:** from `utils/team_data.py` (the existing single source; no second table). A helper picks the block colour per team: the primary colour, unless it is too dark to read on `panel` (e.g. CHI, LV, HOU, NE, SEA, CLE brown), in which case the secondary; and picks white or ink text by contrast. Unknown team -> a neutral `panel-2` block.

**Motion:** card lift + accent edge on hover; win-probability bar grows in on load; all of it disabled under `prefers-reduced-motion`. Focus rings are 2px yellow.

## 6. Information architecture

Nav (desktop row; mobile hamburger panel): **This Week** `/` · **Bets** `/bets` · **Season** `/season` · **Track Record** `/track-record` · **How It Works** `/how-it-works`. Active item is the yellow skewed tab.

Redirects (301, query string preserved):
- `/performance` -> `/track-record#seasons`
- `/backtest` -> `/track-record`
- `/betting` -> `/track-record#betting-sim`
- `/insights` -> `/how-it-works`

Fragment endpoints keep their URLs: `/fragments/performance` and `/fragments/betting` now render their blocks (`performance_content`, `betting_content`) from `pages/track_record.html`; `/fragments/games` and `/fragments/season` unchanged.

Footer: one quiet line -- data-updated time (localised as today), predictions loaded, and a "Research tool, not betting advice" note.

## 7. Pages

### 7.1 This Week (`/`) -- mockup 02, layout L3 + L1 tags

Top to bottom:
1. Header: "WEEK N" display heading; a sub-line with the game count, the bet count and the lock rule ("lines lock 6 PM ET the day before kickoff"); season/week selector restyled as skewed controls (same HTMX wiring).
2. Condensed not-advice note (one line + "Why?").
   Then, only when the week has graded games, the existing week-results summary (Winner x/y, Spread x/y, Total x/y, "N/A -- No odds data" where applicable) restyled as a scoreboard strip; same numbers, same route computation as today.
3. **This week's bets** (headliners): one card per live bet -- matchup, the pick in large yellow type, stake, edge, EV (no colour). A "full list on Bets" link. The headliner area is built for the same season and week the page is showing, and reflects exactly the state /bets computes for that season and week (it reuses the /bets context builder, not a second copy of the logic): bets listed; "No bets cleared the floor" (one line); "Not evaluated yet" (one line); list missing / blocked -> one line pointing to /bets. Changing the week on This Week re-renders the headliners with the slate (both live inside `#game-grid`).
4. **Full slate**, grouped under yellow TV-window tags (section 9), each group a 4-column grid (3/2/1 on narrower screens) of game cards. The existing sort control stays (Game Time / Confidence / Edge, same HTMX wiring): Game Time (the default) shows the TV-window groups; Confidence or Edge shows one ungrouped grid in that order, since time groups would contradict a non-time sort.
5. Export buttons (same URLs).

**Game card** (scheduled): kickoff time ("Sun 1:00 PM ET"); a BET flag + yellow top edge when the game has a live bet; the score bug (away row, home row: team block, team name, model win %, favourite bright/underdog dimmed); three rows -- Win, Spread, Total -- each "model value · market value" using the existing macros, with an edge chip where an edge exists, the pick words ("KC covers", "Under") under Spread/Total as today, and "No line" (dim) where the market has none. Whole card is the link; a small "View details" affordance stays.

**Game card** (completed): header shows FINAL and the result badge (Correct green / Incorrect red / No pick or Tie grey -- same grading as today); team rows show the final score; the Win row keeps "Predicted: X NN% WP".

Confidence pills are removed from the card (they repeat the edge band the edge chip already shows); see 7.2.

### 7.2 Game detail (`/games/{id}`) -- mockup 04

1. Back link "< Week N".
2. Old-rule note when the game's season is old-rule (condensed).
3. Broadcast matchup header: big team blocks and names, Elo and recent-form summary under each, big win probabilities (favourite in yellow), "@", kickoff and stadium, full-width win-probability bar in team colours. Status badge; for completed games the score and the Correct/Incorrect/No pick/Tie badge and CLV (realised -> green/red allowed).
4. **Model vs Market** panel: Win / Spread / Total rows; columns Model, Market, Published (renamed from "Blended" -- it is the number the site publishes), Edge (neutral yellow chip, with the edge band shown beside it as a monochrome label -- this is where confidence now lives). ATS edge stays in points.
5. **What drives the prediction**: the per-game feature-importance chart with skewed WIN / SPREAD / TOTAL tabs (same ids, same client-side Plotly call, dark styling, yellow bars).
6. **Tale of the tape**: Elo comparison bar, recent form (W/L chips green/red -- realised), head-to-head line.
7. **Venue & weather** list. Export buttons.

### 7.3 Bets (`/bets`) -- mockup 03

Same sections, same order, same states as today, restyled:
1. Header "BETS · WEEK N" + description + selector (same wiring, timeout, sync, failure template).
2. Not-advice note: condensed one-liner + "Why?" (`<details>`; full text in the body; outside the swap target, first in `<main>` as today; it cannot be dismissed, only expanded).
3. **Live bets** as ranked bet slips: rank, matchup score bug, bet type, the pick (yellow), EV %, stake (units), EV band (monochrome), evidence chip (monochrome), and "Line as of" (shown on the slip in small text; also in the `title`). Header line: "ranked by expected value · N bets · X.XXu total". Notices (missing games, not evaluated yet), the sizing footnote, and the four empty/blocked states keep their text.
4. **Suppressed candidates (N)**: native `<details>`, collapsed. While closed, the summary also shows a per-reason count row (counts come from the same rows the section renders). Inside: per-reason groups as today, styled like slips.
5. **Backtest replay** sections and **Forward record**: ghost/yellow tags, evidence chip, condensed old-rule note, then a W-L-P record line, a **result strip** (one tick per graded bet: green win, red loss, grey push), and scoreboard tiles in the existing three figure groups. The strip renders only when its win/loss/push counts equal the block's stored counts; otherwise it is omitted and the tiles stand alone (the stored figures stay authoritative; two disagreeing figures never appear together).
6. Footer row: export buttons, "List populated" stamp + per-game lock sentence, link to the backtest evidence (now `/track-record#betting-sim`).

### 7.4 Season (`/season`) -- mockup 04

1. Header "YYYY SEASON" + sub-line + season selector (same wiring, same error template).
2. Old-rule note (condensed) when applicable.
3. KPI tiles colour-keyed by bet type (Winner yellow, Spread cyan `#4CC9F0`, Totals violet `#C77DFF`, Record white) -- the same keys every chart uses. "Pushes are excluded from the denominator."
4. **Week by week** strip: 18 tiles, each with the week's combined W-L and a green/red proportion bar; current week outlined; future weeks dimmed. Data from the season KPI blob (section 9).
5. Cumulative accuracy and weekly performance charts (restyled), break-even note.
6. Empty state ("No completed games yet") and error state unchanged in wording.

### 7.5 Track Record (`/track-record`) -- new, from Performance + Backtest + Betting + Insights' model-vs-market

Sections (anchor ids in brackets), each with its old-rule note where it applies today:
1. **All-time summary** tiles: total games, overall CLV, WP accuracy, WP Brier.
2. **Season by season** [`seasons`]: season selector + season metrics table (`#performance-content`, `/fragments/performance`), season comparison heatmap, export buttons.
3. **Model vs market** [`vs-market`]: the three model-vs-market charts + the aggregate table.
4. **Closing-line value** [`clv`]: cumulative CLV chart.
5. **Betting simulation** [`betting-sim`]: scope toggle (`#betting-content`, `/fragments/betting`), the 7 KPI tiles, equity curve + 3 per-type minis, ROI by type/season/edge bucket + ROI summary table, edge distribution.

Removed as duplicates: Backtest's "Betting Equity Curves" (same story as the betting equity curve) and Backtest's calibration chart (lives on How It Works). Their prerendered chart ids stay in the cache unchanged (no cache change); they are simply not displayed.

### 7.6 How It Works (`/how-it-works`) -- new, from Insights

1. **How the model works** -- a short plain-English methodology section (about 5 short paragraphs: data sources; the day-before 6 PM ET lock; the three models; blending with the market; the deploy gate and what "old rule / not evidence" means; no proven edge). Drafted from `METHODOLOGY.md` / `PIPELINE.md` during the build and **reviewed by the owner before merge**.
2. **Calibration**: WP reliability, ATS predicted vs actual margin, O/U predicted vs actual total.
3. **What the models rely on**: top features for each model.
4. **Accuracy over time**: per-season accuracy and error trend.

## 8. Charts

- One dark theme applied in `api/charts/core.py`'s shared layout defaults: transparent paper/plot background (the panel shows through), Barlow Condensed axis/legend text, Inter for hover, faint gridlines, no zero-lines, dashed grey reference lines (break-even, perfect calibration), compact margins, legends on top, dark hover labels with a yellow border, no mode bar.
- Series colours: bet types Winner `#FFD400`, Spread `#4CC9F0`, Totals `#C77DFF`; model vs market = solid bet-type colour vs dashed grey; season colours from a dark-friendly sequential set. Green/red only for win/loss and profit/loss series.
- The dark palette lives in `api/charts` (a `theme` module), replacing the import of `TARGET_COLORS`/`SEASON_COLORS` from `backtest.report` (which stays as-is for the white-background backtest HTML reports). This reduces `api/charts/core.py`'s backtest imports from 2 to 1; the import-guard test is updated to the new count deliberately.
- Chart ids, data, titles' meaning and empty-state fallbacks are unchanged. Overlapping-title bugs (ATS/O-U calibration subplots) are fixed by spacing, not by changing content.
- Prerendered charts pick up the theme on the next cache rebuild (section 12).

Hand-built visuals (HTML/CSS, no Plotly): score-bug win-probability bars, the This Week headliner cards, the /bets result strip, the Season week strip, the game-detail tale of the tape.

## 9. Data and backend changes (all small, all presentation-side)

1. **Team colours + kickoff windows helper** (`api/presentation.py`, pure functions, unit-tested): `team_block_colors(abbr) -> (bg, fg)` from `utils/team_data.py`; `kickoff_window(game_date) -> label` grouping by ET day and time band: "Thursday Night", "Sunday · 9:30 AM ET" (international), "Sunday · 1:00 PM ET", "Sunday · 4:05 / 4:25 PM ET" (label built from the distinct times present), "Sunday Night", "Monday Night", otherwise "<Weekday> · <time> ET". The This Week and game-detail routes attach `away_color`, `home_color`, their text colours and the window label to each game dict (formatting only; no metric); the game card reads them with `|default(...)` fallbacks. Pages that only know a `game_id` (the /bets slips) use the same helper through a Jinja global registered in `api/dependencies.py`.
2. **Headliners on This Week**: the route also builds the /bets context for the shown week through the existing builder and passes the live bets + state to the template.
3. **Result strip on /bets**: a new read-only `DataService.get_graded_bet_outcomes()` returning, per graded `bet_list` row, `(provenance, validation_type, season, week, game_id, target, grading outcome)` in a fixed order. No arithmetic in the request path; the template draws one tick per row.
4. **Season week strip**: `api/season_metrics.py` gains `compute_weekly_records(rows)` (per-week combined wins/losses, pushes excluded, same outcome functions as the existing weekly series); its result is stored inside the existing `season_kpis_{season}` JSON blob under a new `weeks` key at population time. No new table, no schema change.

## 10. Responsive and accessibility

- Breakpoints: slate grid 4 / 3 / 2 / 1 columns; headliners 3 / 1; Track Record grids 3 / 1; bet slips collapse to a two-line layout under 768px; wide tables keep horizontal scroll (never clipped).
- Mobile nav: hamburger opens a full-width panel with the five items; 44px minimum touch targets throughout.
- Skewed elements never skew text; the text is counter-skewed. Condensed display type is never used below 12px.
- Every chart and visual keeps a text equivalent nearby (numbers in tiles/tables); strips and bars have `aria-label`s with their counts.
- `<details>` toggles are native (keyboard and screen-reader accessible for free).
- Print stylesheet switches to black-on-white.

## 11. Testing

Rules: targeted test runs by node id only (no full-suite runs; it is too slow). Ruff + pyright on touched Python. Tailwind recompiled after every template batch.

**Tests updated on purpose** (styling, not meaning):
- Class-string assertions (grid classes, badge/figure/stake cell classes, `text-green-700`/`text-red-600` -> the new green/red shades, `bg-red-50` error class, `min-h-[44px]`).
- The two week-selector snapshot files and the selector's opening-tag constant (re-recorded after the restyle; wiring assertions unchanged).
- `test_export.py` row regex (keeps asserting page order == export order against the new slip markup), its `href="/betting"` cross-link check (now `/track-record#betting-sim`, same link text), and the `_export_buttons.html` byte-identity guard (the partial is restyled; the guard's intent -- same partial reused everywhere -- is kept).
- Heading/marker strings that change: "This Week's Predictions" page heading, "Prediction vs Market" -> "Model vs Market", "Feature Importance" -> "What drives the prediction", "Blended" -> "Published", the merged pages' headings.
- Confidence-badge assertions on `/` (removed from cards; asserted on game detail instead, monochrome).
- Nav-link counts (`/insights`, `/betting` no longer in nav; `/bets` still exactly 2 on /bets).
- `test_page_labels.py` page list + markers + per-page old-rule label counts, and `test_page_labels_routes.py` route/fragment lists, for the 2 new pages and the 4 retired templates.
- Content tests for `/performance`, `/backtest`, `/insights`, `/betting` move to `/track-record` and `/how-it-works`; the old URLs get redirect tests (status, `Location`, query passthrough).
- `test_import_guard_bets.py` backtest-import count for `api/charts/core.py` (2 -> 1).

**Tests unchanged** (they must still pass as-is): every honesty-wording assertion, every id/data-attribute/structure/order assertion, HTMX wiring, Cache-Control, fragment shape, malformed-param handling, chart content/id tests.

**New tests:** team colour helper (dark-primary fallback, text contrast, unknown team); kickoff-window helper (all labels incl. international and Monday); This Week headliner state equals /bets state for each fixture state; result-strip counts equal block counts (and strip omitted when they differ); weekly-records function; redirects.

**Visual check:** Playwright screenshots of every page at 1440, 1024, 768 and 390px wide, plus the /bets states, compared against the mockups and reviewed by the owner before merge.

## 12. Delivery

- All work on `redesign/broadcast-ui` in `C:\Users\jackc\Code\nfl-predict-redesign`. The project folder stays on master throughout. `tools/tailwindcss.exe` (untracked) is copied into the worktree; `uv sync` once.
- Nothing runs from the worktree near 5 PM ET on any day, and nothing at all 4:30-5:45 PM ET on Saturday 2026-10-03. No preview server is ever left running against `data/web_cache.duckdb`.
- Previews use a copy of the main folder's `data/web_cache.duckdb` placed in the worktree's own `data/` (the app resolves the cache relative to its own folder), refreshed by hand when needed -- never the live file.
- Build stages, each committed separately and checked before the next:
  1. Foundation: fonts, tokens, components, base layout + nav + footer, honesty-note pattern, shared components, presentation helper.
  2. This Week + game card + game detail.
  3. Bets.
  4. Season (+ weekly records).
  5. Track Record + How It Works + redirects + label-test machinery.
  6. Chart theme + hand-built visuals.
  7. Responsive/accessibility pass + screenshot review.
- **Merge rule:** merge into master only after Plan 33-18 is closed. Expected overlap: 33-18 Task 9 registers a readout in `tests/unit/test_old_rule_labels.py`, which feeds the page-label tests this redesign reworks; resolve at merge.
- After merge, prerendered charts restyle on the next cache rebuild (the next daily run, or a manual `uv run python scripts/populate_cache.py` outside the 5 PM window).

## 13. Open items and risks

- **Methodology copy** (7.6) is new prose and must be owner-approved before merge.
- **Team colour fallback** picks secondaries for very dark primaries; a few teams may need a hand-tuned override after the screenshot review.
- **CDN dependencies** (HTMX, Plotly) are unchanged; the site still needs internet for them, as today.
- **Merge conflicts** with 33-18's final commits are expected in the page-label test area only.
