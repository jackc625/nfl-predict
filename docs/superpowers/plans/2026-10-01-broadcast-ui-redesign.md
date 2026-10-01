# Broadcast UI Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restyle the whole web dashboard in the dark "Broadcast" look, cut the nav from 7 pages to 5 (Track Record and How It Works replace Performance / Backtest / Insights / Betting, with redirects), and rebuild This Week, Bets, Season and game detail per the approved mockups -- presentation only, every honesty contract intact.

**Architecture:** Server-rendered Jinja2 (Jinja2Blocks) + HTMX pages over a read-only DuckDB cache, styled with Tailwind v4 (vendored CLI) plus a small set of component classes declared in `web/static/input.css`. Three small presentation-side additions: a pure helper module `api/presentation.py` (team colours, kickoff labels, TV-window grouping), one read-only `DataService` getter (per-bet graded outcomes), and a per-week record list added to the existing season KPI blob at population time. Plotly charts keep their ids and data and get one dark theme from a new `api/charts/theme.py`.

**Tech Stack:** Python 3.13 (uv, Ruff, pyright), FastAPI, Jinja2 + jinja2-fragments (Jinja2Blocks), HTMX 2.0.4 (CDN), Plotly 2.35.2 (CDN) + plotly.py, Tailwind CSS v4 (`tools/tailwindcss.exe`), DuckDB, pytest.

**Spec:** `docs/superpowers/specs/2026-10-01-broadcast-ui-redesign-design.md` (approved 2026-10-01). Approved mockups: `docs/superpowers/specs/2026-10-01-broadcast-ui-mockups/*.html` -- open them in a browser; they are the visual source of truth for every page in this plan.

**Where the work happens:** branch `redesign/broadcast-ui`, worktree `C:\Users\jackc\Code\nfl-predict-redesign`. Every command in this plan runs from that folder. The main project folder `C:\Users\jackc\Code\nfl-predict` stays on master and is never edited.

## Global Constraints

Every task's requirements implicitly include this section.

- **Presentation only.** Do not touch ingestion, features, gold, models, artifacts, the blend, the gate, the bet selector or its EV floor, `backtest/**` logic, the daily pipeline steps, the schema of any existing cache table, or export content.
- **Honesty wording is kept word-for-word and stays in the HTML** (behind a "Why?" `<details>` where condensed): the not-wagering-advice banner text including "1 unit = 1% of a notional bankroll"; the old-rule sentence "Built under the old rule on inputs later found defective; not evidence." and the date "2026-09-15"; the 13 suppression labels and help lines; the suppressed-section caption; "(no reason recorded)"; the three provenance labels, defined only in `components/_provenance_badge.html`; tracker headings, captions, push footnote, "A negative return here is the measurement, not a display problem.", "Nothing graded yet", "not measured"; the per-game lock sentence; every /bets state heading and recovery text with `generate_bet_list.py` named before `populate_cache.py`. No "friday" on /bets or /season, no week-level lock claim, no `$` on /bets.
- **Colour meaning:** green/red only for a realised result (bet won/lost, pick correct/incorrect, recorded W/L, realised return, realised CLV). Never for a pre-game edge, confidence, EV band or evidence label. EV band, provenance, confidence and old-rule labels are grey-scale only. Yellow `accent` is brand/emphasis only. Error red stays distinct from the grey empty state. Outcome colours use Tailwind `green-*` / `red-*` class names so the substring hue-guard tests stay meaningful.
- **Structure kept:** ids `game-grid`, `bets-content`, `bets-loading`, `bets-failure-template`, `season-error-template`, `suppressed-candidates`, `export-buttons`, the `*-select` ids, `tab-wp|ats|ou`, `feature-chart`, `performance-content`, `betting-content`, `season-content`; data attributes `data-chart-id`, `data-tracker-block`, `data-figure-group`, `data-provenance`, `data-validation-type`, `data-missing-games`, `data-old-rule-label`, `data-utc`; suppressed candidates a collapsed native `<details>` with the caption in `<summary>`; one `<section>` per tracker block; pushes outside the hit-rate group; declared tracker block order; EV-descending bet order with the existing tie-break and no sort control; page order equals export order; all HTMX wiring (endpoints, targets, `hx-include`, the /bets timeout + `hx-sync` + `hx-disabled-elt` + failure handlers); fragments contain no `<html>`/`<head>`/`<nav>`; Cache-Control exactly as today; malformed params fall back (200, never 500).
- **UIAP-01:** no metric is computed in the request path. Grouping, ordering and formatting are presentation; aggregation is not.
- **Number formatting** stays in `components/_prediction_values.html` (`win_prob`, `margin`, `total`, `market_wp_missing`, `absent`) and the existing `side_line` macro, which Task 8 moves verbatim from `pages/bets.html` into `components/_bet_pick.html` (Decision 8). No new sign logic anywhere.
- `components/_game_card.html` must render with only `game=` in a plain `jinja2.Environment(FileSystemLoader)` (no app globals or filters). New fields are read with `|default(...)`.
- `components/_old_rule_label.html` uses only plain (non-variant) classes; each must appear literally as `.classname` in `web/static/css/tailwind-compiled.css`.
- **Tests:** run targeted node ids or single test files only -- NEVER a bare `uv run pytest` (the full suite is too slow). Ruff (`uv run ruff check <files>` and `uv run ruff format <files>`) and pyright (`uv run pyright <files>`) on every touched Python file.
- **CSS build:** after any template or `input.css` change, run `./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify` and commit the compiled file with the change.
- **Timing:** never run anything from the worktree between 16:30 and 17:45 ET on any day (the daily pipeline runs at 17:00 ET from the main folder), and nothing at all 16:30-17:45 ET on Saturday 2026-10-03. Never leave a preview server running. Read ET time with PowerShell `Get-Date` (Git Bash's TZ handling is wrong on this machine).
- **Previews** use a COPY of `C:\Users\jackc\Code\nfl-predict\data\web_cache.duckdb` in the worktree's own `data\` folder, on port 8001, never the live file.
- No emojis in code. Comments explain why, matching the surrounding density.
- **Commits:** one per task (or per step where the task says so), message `<type>(redesign): <summary>`, ending with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never commit `.planning/`.

## Review Focus

These are the five inputs the spec implies but no happy-path test exercises, most likely first. Each one has a test in the task that owns the code.

1. **A week with no market lines at all (today's real Week 3).** Every card, headliner area and game detail must read cleanly with "No line" in every market slot and no edge chips -- no empty chips, no "None", no "nan", no stray "vs". Pinned in Task 7 (card) and Task 9 (detail).
2. **Teams whose primary colour is near-black (CHI, LV, HOU, NE, SEA, CLE, JAX alternates) and an unknown abbreviation.** Blocks must stay readable on the dark panel, and an unknown team must render a neutral block, not crash. Pinned in Task 2.
3. **Odd kickoff times: an international 9:30 AM ET Sunday game, a Saturday late-season slate, a Christmas-Day weekday, a Monday doubleheader, and a game with no `game_date`.** Each must land in a sensible, labelled group, and a missing date must still render (grouped under "Time TBD"). Pinned in Task 2 and Task 8.
4. **The /bets result strip disagreeing with the stored tracker block** (a cache built by a different population run). The strip must be omitted, never shown beside tiles it contradicts. Pinned in Task 12.
5. **Old bookmarks and in-page links to the retired pages, including query strings** (`/performance?season=2023`, `/betting?scope=all`) and HTMX requests to the old full-page URLs. They must land on the right section with the query preserved. Pinned in Task 15.

## Shared Interface Contract

Every task uses these exact names. A task that needs something not listed here defines it in its own **Interfaces: Produces** block.

### 1. Design tokens (Tailwind v4 `@theme` in `web/static/input.css`)

| Token (CSS var) | Value | Utilities it generates |
|---|---|---|
| `--color-ink` | `#0B0F17` | `bg-ink`, `text-ink` |
| `--color-ink-2` | `#0E1320` | `bg-ink-2` |
| `--color-panel` | `#151B29` | `bg-panel` |
| `--color-panel-2` | `#1D2436` | `bg-panel-2` |
| `--color-line` | `rgb(255 255 255 / 0.08)` | `border-line`, `divide-line` |
| `--color-fg` | `#F3F5F9` | `text-fg` |
| `--color-muted` | `#8A93A8` | `text-muted` |
| `--color-dim` | `#5D667C` | `text-dim` |
| `--color-accent` | `#FFD400` | `bg-accent`, `text-accent`, `border-accent` |
| `--color-target-wp` | `#FFD400` | `text-target-wp`, `border-target-wp` |
| `--color-target-ats` | `#4CC9F0` | `text-target-ats`, `border-target-ats` |
| `--color-target-ou` | `#C77DFF` | `text-target-ou`, `border-target-ou` |
| `--font-display` | `"Barlow Condensed", "Arial Narrow", sans-serif` | `font-display` |
| `--font-sans` | `"Inter", system-ui, sans-serif` | `font-sans` (body default) |
| `--font-mono` | `"JetBrains Mono", ui-monospace, monospace` | `font-mono` |

The old `--color-nfl-*` tokens are removed; any remaining `nfl-*` class is replaced in the task that touches its template.

Outcome / state classes (Tailwind built-ins, used ONLY for their meaning):
- Realised win / correct text: `text-green-400`; fill: `bg-green-500`; soft background: `bg-green-500/15`.
- Realised loss / incorrect text: `text-red-400`; fill: `bg-red-500`; soft background: `bg-red-500/15`.
- Error state container: `bg-red-950 border border-red-800`, heading `text-red-200`, body `text-red-100`.
- Neutral / push / zero: `text-muted`.

### 2. Component classes (`@layer components` in `web/static/input.css`)

| Class | Meaning |
|---|---|
| `skew` / `unskew` | `transform: skewX(-12deg)` / `skewX(12deg)` -- skewed boxes, upright text inside |
| `display` | `font-display`, italic, weight 800, uppercase, tight line-height |
| `label` | `font-display`, weight 700, 12px, letter-spacing .12em, uppercase, `text-muted` |
| `num` | `font-mono`, `tabular-nums` |
| `stat-num` | kept for compatibility: same as `num` |
| `panel` | `bg-panel`, radius 4px, padding 14px 16px |
| `panel-title` | `display` 18px with a 4x16px skewed accent bar before it |
| `section-head` | flex row: a `tag`, a 1px `bg-line` rule filling the middle, optional right-hand meta text in `text-muted text-xs` |
| `tag` / `tag-ghost` | yellow skewed section tag (ink text, `display` 15px) / outlined white variant |
| `skew-control` / `skew-control-active` | skewed control box (`bg-panel-2`, `display` 15px) / active state (`bg-accent text-ink`) -- selectors, tabs, nav |
| `team-block` / `team-block-lg` | skewed team-colour block, background and text colour set inline via `style="--team-bg:#..;--team-fg:#.."`; abbreviation inside a `unskew` span |
| `score-bug` | grid of two `score-row`s |
| `score-row` (+ `is-fav` / `is-dog`) | 3-column row: `team-block`, name strip (`bg-panel-2`), value cell (`bg-ink`, `display` value); `is-fav` value in `text-accent`, `is-dog` name and value in `text-muted` |
| `edge-chip` / `edge-chip-soft` | yellow skewed chip (ink text) / outlined yellow chip |
| `stat-tile` | scoreboard tile: `bg-panel`, 3px top rule in `var(--tile-accent, rgb(255 255 255 / .12))`, `label` + big `display` value |
| `bet-slip` | ranked bet row: `bg-panel`, 4px accent left border, grid layout that collapses to two lines below 1024px (`lg`, Decision 14) |
| `honesty-note` | the condensed one-liner `<details>`: 1px `rgb(255 255 255/.16)` border; `<summary>` holds `.honesty-note-key` (display label), the one-line text, and `.honesty-note-why` ("Why?" in accent, chevron flips when open); `.honesty-note-body` holds the full original text |
| `band` + `band-high` / `band-medium` / `band-low` | monochrome EV-band / confidence label: high `#E6E9F0` fill ink text; medium `#5B6478` fill white text; low 1px `#5B6478` outline `#C4CAD8` text |
| `evidence-chip` / `evidence-chip-strong` | monochrome rounded provenance chip: 1px `rgb(255 255 255/.28)` border, `#D5DAE5` text / strong: `rgb(255 255 255/.12)` fill, white text |

### 3. Jinja macros: `web/templates/components/_broadcast.html`

Import with `{% import "components/_broadcast.html" as bc %}`. Plain-Jinja safe (no globals, no filters beyond Jinja built-ins).

```jinja
{% macro section_head(label, meta=none, ghost=false, anchor=none) %}
{% macro team_block(abbr, bg="#1D2436", fg="#FFFFFF", large=false) %}
{% macro score_row(abbr, name, value, bg="#1D2436", fg="#FFFFFF", fav=false, dog=false) %}
{% macro edge_chip(text, soft=false) %}
{% macro stat_tile(label, value, value_class="", sub=none, accent=none) %}
```

### 4. Python presentation helper: `api/presentation.py` (pure functions, no I/O)

```python
class TeamColors(NamedTuple):
    bg: str   # block background hex, e.g. "#E31837"
    fg: str   # "#FFFFFF" or "#0B0F17", whichever contrasts more with bg

def team_block_colors(abbr: str | None) -> TeamColors
def team_nickname(abbr: str | None) -> str            # "KC" -> "Chiefs"; unknown -> the abbreviation, or "" for None
def kickoff_label(game_date: datetime | str | None) -> str   # "Sun 1:00 PM ET"; None -> "Time TBD"
def kickoff_window(game_date: datetime | str | None) -> str  # group label, see Task 2
def decorate_game(game: dict[str, Any]) -> dict[str, Any]    # NEW dict: adds away_color, away_fg, home_color, home_fg, away_name, home_name, kickoff_label, window_label
def group_games_by_window(games: list[dict[str, Any]]) -> list[dict[str, Any]]  # [{"label": str, "games": [...]}] in first-appearance order; input order kept inside a group
```

`predictions.game_date` is a naive TIMESTAMP already in US Eastern time (e.g. `2026-09-27 13:00:00` for a 1 PM ET kickoff); the helpers treat naive values as ET and convert aware values to `America/New_York`.

Registered in `api/dependencies.py` as Jinja globals: `templates.env.globals["team_colors"] = team_block_colors` and `templates.env.globals["team_nickname"] = team_nickname`.

### 5. Template context keys added

- This Week (`/` and `/fragments/games`): `games` (each passed through `decorate_game`), `slate_groups` (`group_games_by_window(games)` when `current_sort == "time"`, else `None`), `headliner` = `{"state": str, "bets": list[dict], "season": int | None, "week": int | None}` where `state` is one of `"bets"`, `"none_cleared"`, `"not_evaluated"`, `"blocked"`, `"not_built"`, `"no_week"` -- derived from the SAME `_build_bets_context(service, season, week, request)` result /bets uses, never re-derived.
- Game detail: `game` passed through `decorate_game`.
- Bets (`/bets` page and fragment): `graded_outcomes: dict[str, list[str]]`, key `f"{provenance}:{validation_type}"`, value the `grading_status` of each graded live bet (`"win" | "loss" | "push"`) in a fixed order.
- Season: `kpis["weeks"]: list[{"week": int, "wins": int, "losses": int}]` (absent or empty for a cache built before this change -- the strip is then not rendered).

### 6. Data and metrics additions

```python
# api/services.py
def get_graded_bet_outcomes(self) -> list[dict[str, Any]]
    # keys: provenance, validation_type, season, week, game_id, target, grading_status
# api/season_metrics.py
def compute_weekly_records(rows: Sequence[dict]) -> list[dict[str, int]]
    # [{"week": 1, "wins": 27, "losses": 21}, ...] ascending week, pushes/ties excluded
```

### 7. Routes

- `GET /track-record` -> `pages/track_record.html`, `current_path="/track-record"`; blocks `performance_content` and `betting_content` live in this template.
- `GET /how-it-works` -> `pages/how_it_works.html`, `current_path="/how-it-works"`.
- 301 redirects, query string preserved: `/performance` -> `/track-record?<qs>#seasons`, `/backtest` -> `/track-record?<qs>`, `/betting` -> `/track-record?<qs>#betting-sim`, `/insights` -> `/how-it-works?<qs>`.
- `/fragments/performance` and `/fragments/betting` render their blocks from `pages/track_record.html`.
- Section anchors on Track Record: `summary`, `seasons`, `vs-market`, `clv`, `betting-sim`.

### 8. Chart theme: `api/charts/theme.py`

```python
INK = "#0B0F17"; PANEL = "#151B29"; FG = "#F3F5F9"; MUTED = "#8A93A8"
GRID = "rgba(255,255,255,0.06)"; REFERENCE_LINE = "rgba(255,255,255,0.45)"; ACCENT = "#FFD400"
WIN_COLOR = "#22C55E"; LOSS_COLOR = "#EF4444"; NEUTRAL_COLOR = "#5B6478"
FONT_DISPLAY = "Barlow Condensed, Inter, sans-serif"; FONT_BODY = "Inter, system-ui, sans-serif"; FONT_MONO = "JetBrains Mono, monospace"
TARGET_COLORS: dict[str, str] = {"wp": "#FFD400", "ats": "#4CC9F0", "ou": "#C77DFF"}
SEASON_COLORS: dict[int, str]   # 2018-2026, dark-friendly sequence
def apply_dark_theme(fig: go.Figure) -> None
```

`api/charts/core.py` imports `TARGET_COLORS` and `SEASON_COLORS` from `api.charts.theme` (no longer from `backtest.report`) and `_apply_layout_defaults` delegates to `apply_dark_theme`.

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `web/static/fonts/*.woff2` + `OFL-*.txt` | self-hosted Barlow Condensed, Inter, JetBrains Mono | 1 |
| `web/static/input.css` | tokens, `@font-face`, component classes | 1 |
| `api/presentation.py` + `tests/unit/test_presentation.py` | team colours, nicknames, kickoff labels, TV windows, game decoration | 2 |
| `api/dependencies.py` | registers the two Jinja globals | 2 |
| `web/static/css/custom.css` | non-Tailwind bits: scrollbars, focus ring, reduced motion, print, htmx indicator, chart container | 3 |
| `web/templates/base.html` | dark shell, 5-item nav, mobile menu, footer | 3 |
| `web/templates/components/_broadcast.html` | shared macros | 3 |
| `components/_not_advice_banner.html`, `_old_rule_label.html`, `_provenance_badge.html`, `_ev_band_badge.html`, `_confidence_badge.html`, `_status_badge.html` | honesty notes and monochrome badges | 4 |
| `components/_week_selector.html`, `_season_selector.html`, `_season_tracking_selector.html`, `_betting_scope_toggle.html`, `_sort_controls.html`, `_empty_state.html`, `_error_state.html`, `_export_buttons.html`, `_loading_skeleton.html`, `_week_summary.html` + the two selector snapshots | controls and states | 5 |
| `api/routes/pages.py` + `api/routes/fragments.py` (This Week context) | decorated games, TV-window groups, headliner state | 6 |
| `web/templates/components/_game_card.html` | score-bug game card | 7 |
| `web/templates/components/_bet_headliners.html`, `web/templates/pages/this_week.html` | This Week layout | 8 |
| `web/templates/pages/game_detail.html` (+ route decoration) | game detail layout | 9 |
| `api/services.py`, `api/routes/pages.py` (bets context) | `get_graded_bet_outcomes`, `graded_outcomes` | 10 |
| `web/templates/pages/bets.html` (header, slips, notices, states, suppressed) | bets layout | 11 |
| `web/templates/pages/bets.html` (tracker), `components/_result_strip.html` | tracker restyle + tick strip | 12 |
| `api/season_metrics.py`, `api/charts/prerender.py`, `api/charts/season.py`, `tests/api/conftest.py` | weekly records into the season KPI blob | 13 |
| `web/templates/pages/season.html`, `components/_week_strip.html` | season layout | 14 |
| `web/templates/pages/track_record.html`, `pages/how_it_works.html`, `api/routes/pages.py`, `api/routes/fragments.py`; deleted `pages/{performance,backtest,betting,insights}.html` | merged pages, redirects, label-test machinery | 15 |
| `web/templates/pages/how_it_works.html` (methodology section) | owner-reviewed plain-English methodology | 16 |
| `api/charts/theme.py`, `api/charts/core.py`, `tests/api/test_import_guard_bets.py` | dark chart theme foundation | 17 |
| `api/charts/insights.py`, `betting.py`, `season.py`, `core.py` chart bodies | per-chart colour/spacing fixes (the game-detail feature chart is restyled in Task 9, not here) | 18 |
| all templates (responsive + accessibility pass) | breakpoints, focus, motion, aria | 19 |
| -- | pre-merge verification and owner screenshot review | 20 |

---

## Decisions made while planning (deviations from the spec, owner to confirm at plan review)

1. **/bets heading stays "Weekly Bet List"** (not "Bets . Week N"): the heading sits outside the HTMX swap target, so a week number there would go stale after a week change. The week is named by the selector and the in-swap "Live bets -- <season> Week <n>" tag.
2. **No "X.XXu total" on /bets or the headliners:** summing stakes in the template would be a request-path aggregate (UIAP-01). The bet count stays.
3. **Headliner cards show the pick, stake, EV and EV band, not "edge":** a bet row carries EV, not the card's model-vs-market edge.
4. **Season "Record" tile counts Winner picks only** (as the stored KPI does today), so it is labelled that way rather than "all three bet types"; the three hit-rate tiles are "Winner / Spread / Totals hit rate".
5. **Track Record layout:** the season-comparison heatmap sits in the summary section and closing-line value is a panel (`id="clv"`) inside Model vs Market; all five anchors still exist.
6. **The season heatmap uses a single grey scale** instead of red-yellow-green (green/red are reserved for realised results).
7. **TV-window grouping applies to any time order** (including an unrecognised `?sort=`, which the service already treats as time order); only Confidence and Edge sorts show one ungrouped grid.
8. **Shared spread-pick wording:** `side_line`, `picked_team` and `pick_label` move verbatim from `pages/bets.html` into `components/_bet_pick.html` (Task 8) so /bets and the This Week headliners cannot sign a bet differently.
9. **`bet-slip` is a visual container only;** the slip's grid layout lives in `pages/bets.html` (Task 11).
10. **The week strip is a labelled `<ol>`** (each week's record is real text) rather than `role="img"`; the result strip, which has no per-item text, is `role="img"` with its counts in the label.
11. **Suppressed-candidate groups stay restyled tables, not slips** (spec 7.3.4 says "styled like slips"): a suppressed group can hold many rows, and a dense table scans better than a stack of slips. The native collapsed `<details>` and its `<summary>` caption are unchanged (Task 11).
12. **The This Week week-results strip keeps the labels "Win Prob" / "Spread" / "Total"** (spec 7.1 and Part D use "Winner"): `tests/api/test_fragments.py` pins those exact labels, and the strip is restyled only (Task 5's `_week_summary.html`).
13. **The model-vs-market "Gap" on Track Record is monochrome** (Task 15): the number keeps its sign and a muted "favours model" / "favours market" word, read from the stored `gap_favorable`, says which side it is on. A metric gap is a comparison, not a realised result, so it never takes green/red (the same reasoning as Decision 6).
14. **Tags wrap below 640px, and the 7-column bet-slip grid starts at `lg`:** `.tag` / `.tag-ghost` drop `white-space: nowrap` under 640px (Task 1) so the long tracker headings cannot push a 390px phone sideways, and the slip's seven columns need more than the 720px a 768px tablet gives, so slips keep the stacked two-line layout below `lg` (Task 11). Spec 10: no horizontal scroll at any width.

---

# Part A -- Foundation (Tasks 1-5)

## Contract notes (Part A)

These refine the Shared Interface Contract where the real code, or another part's requirements, forced a decision. Later tasks rely on them. They are reconciled with Part C's contract notes (items 3-5) and Part E's conventions (Task 19's accessibility audit).

1. **Legacy `--color-nfl-*` tokens stay until Task 20.** Task 1 keeps the five old tokens in a separate, clearly marked `@theme` block in `web/static/input.css`. `game_detail.html`, `bets.html`, `backtest.html`, `_game_card.html` and the JS tab classes in `game_detail.html` still use `text-nfl-primary` / `bg-nfl-primary` until their own tasks restyle them; deleting the tokens first would leave those pages unstyled mid-branch. Task 20 deletes the block once `grep -rn "nfl-" web/templates api` prints nothing.
2. **Every skewed box transforms the ELEMENT; its text goes in a `.unskew` child.** This applies to `tag`, `tag-ghost`, `skew-control`, `skew-control-active`, `edge-chip`, `edge-chip-soft`, `team-block` and `team-block-lg`. Write `<h2 class="tag"><span class="unskew">Live bets</span></h2>`, `<a class="skew-control"><span class="unskew">CSV</span></a>`, and so on. The macros already do this. Task 19's audit fails any text node that sits directly inside one of these classes.
   - The one exception is `<select class="skew-control">`. A select cannot hold a `.unskew` child, so the CSS does not transform it; it gets its slant from a `clip-path` parallelogram. Its only text is `<option>`s, which the audit ignores.
   - A `skew-control` keeps a 44px touch target and paints a 32px band: its background is a gradient sized `100% calc(100% - 12px)`.
3. **`tag-ghost` is a standalone class, not a modifier.** Write `<h2 class="tag-ghost">`, as Part C does. It carries every `tag` property plus the outline, and never needs `tag` beside it. `section_head(..., ghost=true)` emits `class="tag-ghost"`.
4. **`section-head` is a plain flex row** (`display:flex; align-items:center; gap:.75rem`). It draws no rule of its own and has no margin. The rule is an explicit child, `<span class="flex-1 h-px bg-line" aria-hidden="true">`, and the meta is `<span class="text-xs text-muted whitespace-nowrap">`. The `section_head` macro emits exactly that, adds `mb-3` to the row, and puts the label in a `.unskew` span inside the `<h2>`. That keeps Part C's tracker test true: a badge placed after the `<h2>` sits before the section's first `</div>`.
5. **Classes added beyond contract section 2** (all in `input.css`, all compiled): `nav-link`, `wordmark`, `skip-link`, `score-name`, `score-value`, `stat-tile-value`, `stat-tile-sub` (12px: spec 5 allows 11px only in uppercase), `honesty-note-key`, `honesty-note-line`, `honesty-note-why`, `honesty-note-body`, `grow-in` (the one-shot win-probability bar animation, switched off under `prefers-reduced-motion`). `bet-slip` is a visual container only (panel background, accent left border, hover); Task 11 lays the slip out with grid utilities in `pages/bets.html` (reconciled with Part C, which owns that layout).
6. **Confidence label markup** (Task 4): `<span class="band band-{high|medium|low}" data-confidence-band="{band}">{label}</span>`. The EV-band label shares the `band band-*` classes, keeps `class` BEFORE `title="EV band ..."` (Part C 5a), and has no `data-confidence-band`. **Part B:** any test that counts confidence labels must key on `data-confidence-band`, never on the class.
7. **Provenance chip mapping** (Part C 5b, used exactly). The before-redesign strongest grey keeps the strong chip.

   | Validation type | Chip classes |
   |---|---|
   | `contaminated` | `evidence-chip` |
   | `clean_holdout` | `evidence-chip evidence-chip-strong` |
   | `forward_realized` | `evidence-chip` |

   The outer `<span>` keeps `data-provenance`, `data-validation-type`, and the nested `<span class="sr-only">`.
8. **Error-state reds** (Task 5): `_error_state.html` now uses `bg-red-950 border border-red-800`, a `text-red-200` heading, a `text-red-100` body and a `text-red-300` icon. None of those is the realised-loss `text-red-400`. The instruction text stays in ONE `<p>`: `recovery_text` here, and `body` in `_empty_state.html` (Part C 5f).
   - Task 5 does NOT touch `test_green_and_red_appear_only_inside_the_tracker_sections`. Part C's Task 12 rewrites that test whole.
   - Between the two tasks it still passes. Above the tracker, `text-red-600` and `bg-red-50` both count zero once the error state stops using them.
9. **Week selector root.** Task 5 gives the selector root a unique class, `week-selector`, so `SELECTOR_OPEN` becomes `'<div class="week-selector flex flex-wrap items-center gap-2">'`. **Any later edit to `components/_week_selector.html` must re-record both snapshots with Task 5 Step 6**, and must not change the root tag.
10. **Active nav item.** `base.html` highlights This Week when `current_path` is `"/"` or starts with `"/games"`. **Part B (Task 9):** the game-detail route passes `f"/games/{game_id}"`, which that rule highlights. Today the route passes `""`, which highlights nothing. Every active nav link carries `aria-current="page"`. The hamburger carries `aria-controls="mobile-menu"`, `aria-expanded` and `aria-label`, using Part E's exact markup.
11. **Macro imports inside blocks** (Part C note 4). jinja2-fragments renders only the named block for an `HX-Request`, and a child template's top-level `{% import %}` does not run in a block render. Any page rendered with `block_name=` imports `components/_broadcast.html` INSIDE that block, or inside the included component. `_week_summary.html` imports it itself for that reason.
12. **`@source`.** Task 1 imports Tailwind with `@import "tailwindcss" source(none);`, which turns off Tailwind v4's automatic whole-repo scan, and registers `@source "../../web/templates";` as the only scanned folder. Without `source(none)` Tailwind scans every file in the repo (today's sheet carries `.top-25` only because `SELECTION-CENSUS.md` mentions it), so the compiled CSS would change whenever a .py, .md or test file changes and Task 20's drift and grep checks would report false defects. Part E's Task 17 adds `@source "../../api/charts";` for the class strings in `_empty_chart_div`, which `source(none)` makes required; Part A adds no `@source` for `api/`.
13. **One file not in the contract's file table:** `api/main.py` registers the `font/woff2` MIME type (Task 1). The Windows MIME registry does not know `.woff2`, so Starlette would otherwise serve the fonts as `text/plain`.
14. **Team colour rule** (Task 2). Each team keeps its primary colour unless that colour's contrast against the panel `#151B29` is below `MIN_BLOCK_CONTRAST = 1.15`, or it is pure black or white; then it uses the secondary. The fallback branch reaches no current team.
    - **Changed by the rule** (exact, for Part B's tests):

      | Team | Block colour | Text colour |
      |---|---|---|
      | CHI | `#C83803` | white |
      | HOU | `#A71930` | white |
      | NE | `#C60C30` | white |
      | SEA | `#69BE28` | ink `#0B0F17` |
      | TEN | `#4B92DB` | ink |
      | CLE | `#FF3C00` | ink |
      | LV | `#A5ACAF` | ink |
    - **Primary kept:**

      | Team | Block colour | Text colour |
      |---|---|---|
      | KC | `#E31837` | white |
      | MIA | `#008E97` | ink |
      | LAC | `#0080C6` | ink |
      | BUF | `#00338D` | white |
      | NYG | `#0B2265` | white |
      | BAL | `#241773` | white |
    - An unknown team or `None` gets `TeamColors("#1D2436", "#FFFFFF")`. `api.presentation.contrast_ratio(a, b)` is public, so tests can measure.
15. **Kickoff-window labels** (Task 2). The separator is a middle dot, U+00B7, written `"\u00b7"` in Python source, so `"Sunday \u00b7 1:00 PM ET"`. The label forms are:
    - `"Thursday Night"` / `"Sunday Night"` / `"Monday Night"` -- any kickoff at 7 PM ET or later, any weekday.
    - `"Sunday \u00b7 9:30 AM ET"` -- before noon.
    - `"Sunday \u00b7 1:00 PM ET"` -- noon to 3:59 PM.
    - `"Sunday \u00b7 4:05 / 4:25 PM ET"` -- 4 PM to 6:59 PM; a group label lists every distinct time in the group.
    - The bare weekday (`"Sunday"`) -- a date with no time. A midnight timestamp counts as "no time".
    - `"Time TBD"` -- no date at all.

    `decorate_game` sets `window_label` from that one game alone. `group_games_by_window` builds the merged label for a whole group, so the template should print the **group's** label.

---

### Task 1: Worktree setup, self-hosted fonts, design tokens and component classes

**Files:**
- Create: `web/static/fonts/` (8 `.woff2` files + `OFL-BarlowCondensed.txt`, `OFL-Inter.txt`, `OFL-JetBrainsMono.txt`)
- Modify: `web/static/input.css` (full replacement)
- Modify: `api/main.py:16-17` (import) and `api/main.py:153-154` (MIME registration before the static mount)
- Regenerate: `web/static/css/tailwind-compiled.css`
- Test: `tests/unit/test_broadcast_theme_css.py` (new), `tests/api/test_static_fonts.py` (new)

**Interfaces:**
- Consumes: nothing.
- Produces: every token in contract section 1 (emitted statically, so `var(--color-*)` works anywhere); every component class in contract section 2 plus the extras in Contract note 5, plus the `grow-in` animation class (spec 5's win-probability grow-in, used by Task 9); the font families `"Barlow Condensed"`, `"Inter"`, `"JetBrains Mono"` served from `/static/fonts/*.woff2` as `font/woff2`.

- [ ] **Step 1: Check the clock, then set up the worktree**

Never run anything here between 16:30 and 17:45 ET. Check first:

```bash
powershell -NoProfile -Command "Get-Date -Format 'yyyy-MM-dd HH:mm'"
```

Expected: a time outside 16:30-17:45. If it is inside that window, stop and wait.

```bash
cd /c/Users/jackc/Code/nfl-predict-redesign
mkdir -p tools data
cp /c/Users/jackc/Code/nfl-predict/tools/tailwindcss.exe tools/tailwindcss.exe
cp /c/Users/jackc/Code/nfl-predict/data/web_cache.duckdb data/web_cache.duckdb
uv sync
./tools/tailwindcss.exe --help | head -1
git status --short
```

Expected: the help line names `tailwindcss v4.2.2`. `git status --short` prints nothing, because `tools/`, `data/web_cache.duckdb` and `.venv/` are all git-ignored.

- [ ] **Step 2: Download the fonts and their licences**

The versions are pinned to `@5.3.0` so a re-download is byte-reproducible. Every file is Latin-subset `woff2`. Inter is a single variable file covering weights 100-900.

```bash
cd /c/Users/jackc/Code/nfl-predict-redesign
mkdir -p web/static/fonts
FS="https://cdn.jsdelivr.net/npm"
BC="$FS/@fontsource/barlow-condensed@5.3.0/files"
for f in barlow-condensed-latin-600-normal barlow-condensed-latin-700-normal barlow-condensed-latin-800-normal barlow-condensed-latin-700-italic barlow-condensed-latin-800-italic; do
  curl -fsSL -o "web/static/fonts/$f.woff2" "$BC/$f.woff2"
done
curl -fsSL -o web/static/fonts/inter-latin-wght-normal.woff2 "$FS/@fontsource-variable/inter@5.3.0/files/inter-latin-wght-normal.woff2"
curl -fsSL -o web/static/fonts/jetbrains-mono-latin-500-normal.woff2 "$FS/@fontsource/jetbrains-mono@5.3.0/files/jetbrains-mono-latin-500-normal.woff2"
curl -fsSL -o web/static/fonts/jetbrains-mono-latin-700-normal.woff2 "$FS/@fontsource/jetbrains-mono@5.3.0/files/jetbrains-mono-latin-700-normal.woff2"
curl -fsSL -o web/static/fonts/OFL-BarlowCondensed.txt "$FS/@fontsource/barlow-condensed@5.3.0/LICENSE"
curl -fsSL -o web/static/fonts/OFL-Inter.txt "$FS/@fontsource-variable/inter@5.3.0/LICENSE"
curl -fsSL -o web/static/fonts/OFL-JetBrainsMono.txt "$FS/@fontsource/jetbrains-mono@5.3.0/LICENSE"
ls -la web/static/fonts
for f in web/static/fonts/*.woff2; do head -c 4 "$f"; echo "  $f"; done
```

Expected:
- 11 files: eight `.woff2` of roughly 21-48 KB each, and three licence `.txt` files of roughly 4-6 KB.
- Every `.woff2` starts with the 4-byte signature `wOF2`.

- [ ] **Step 3: Write the failing CSS and font tests**

Create `tests/unit/test_broadcast_theme_css.py`:

```python
"""Guards the Broadcast design system's compiled output (redesign Task 1).

Tailwind v4 emits only what it is told to: a token missing from ``@theme static``, a component
class missing from ``@layer components`` or an ``@font-face`` pointing at a file that was never
committed renders as silently unstyled HTML. Each of those is checked against the COMPILED sheet
the app actually serves, not against ``input.css``.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPILED_CSS = REPO_ROOT / "web" / "static" / "css" / "tailwind-compiled.css"
FONTS_DIR = REPO_ROOT / "web" / "static" / "fonts"

TOKENS = (
    "--color-ink",
    "--color-ink-2",
    "--color-panel",
    "--color-panel-2",
    "--color-line",
    "--color-fg",
    "--color-muted",
    "--color-dim",
    "--color-accent",
    "--color-target-wp",
    "--color-target-ats",
    "--color-target-ou",
    "--font-display",
    "--font-sans",
    "--font-mono",
)

COMPONENT_CLASSES = (
    "skew",
    "unskew",
    "display",
    "label",
    "num",
    "stat-num",
    "panel",
    "panel-title",
    "section-head",
    "tag",
    "tag-ghost",
    "skew-control",
    "skew-control-active",
    "nav-link",
    "wordmark",
    "skip-link",
    "team-block",
    "team-block-lg",
    "score-bug",
    "score-row",
    "score-name",
    "score-value",
    "is-fav",
    "is-dog",
    "edge-chip",
    "edge-chip-soft",
    "stat-tile",
    "stat-tile-value",
    "stat-tile-sub",
    "bet-slip",
    "honesty-note",
    "honesty-note-key",
    "honesty-note-line",
    "honesty-note-why",
    "honesty-note-body",
    "band",
    "band-high",
    "band-medium",
    "band-low",
    "evidence-chip",
    "evidence-chip-strong",
    "grow-in",
)

FONT_FILES = (
    "barlow-condensed-latin-600-normal.woff2",
    "barlow-condensed-latin-700-italic.woff2",
    "barlow-condensed-latin-700-normal.woff2",
    "barlow-condensed-latin-800-italic.woff2",
    "barlow-condensed-latin-800-normal.woff2",
    "inter-latin-wght-normal.woff2",
    "jetbrains-mono-latin-500-normal.woff2",
    "jetbrains-mono-latin-700-normal.woff2",
)

LICENCES = ("OFL-BarlowCondensed.txt", "OFL-Inter.txt", "OFL-JetBrainsMono.txt")


@pytest.fixture(scope="module")
def css() -> str:
    return COMPILED_CSS.read_text(encoding="utf-8")


@pytest.mark.parametrize("token", TOKENS)
def test_every_design_token_is_emitted(css: str, token: str) -> None:
    """``@theme static`` emits each token even when no utility uses it yet."""
    assert f"{token}:" in css, f"{token} is not defined in the compiled stylesheet"


@pytest.mark.parametrize("name", COMPONENT_CLASSES)
def test_every_component_class_is_compiled(css: str, name: str) -> None:
    """The exact class, not merely a longer class that starts with the same letters."""
    assert re.search(rf"\.{re.escape(name)}(?![\w-])", css), (
        f".{name} is missing from tailwind-compiled.css"
    )


def test_each_font_face_points_at_a_committed_file(css: str) -> None:
    urls = re.findall(r"url\([\"']?/static/fonts/([^\"')]+)[\"']?\)", css)
    assert sorted(set(urls)) == sorted(FONT_FILES)
    for name in urls:
        assert (FONTS_DIR / name).is_file(), f"@font-face points at a missing file: {name}"


def test_the_three_families_are_declared(css: str) -> None:
    assert css.count("@font-face") == len(FONT_FILES)
    for family in ("Barlow Condensed", "Inter", "JetBrains Mono"):
        assert family in css


@pytest.mark.parametrize("name", FONT_FILES)
def test_every_font_file_is_real_woff2(name: str) -> None:
    assert (FONTS_DIR / name).read_bytes()[:4] == b"wOF2"


@pytest.mark.parametrize("name", LICENCES)
def test_the_open_font_licences_ship_with_the_fonts(name: str) -> None:
    text = (FONTS_DIR / name).read_text(encoding="utf-8")
    assert "SIL Open Font License" in text
```

Create `tests/api/test_static_fonts.py`:

```python
"""The self-hosted fonts are served with a font MIME type (redesign Task 1).

Windows' MIME registry has no entry for .woff2, so without the registration in api/main.py
Starlette answers text/plain and a browser honouring nosniff refuses the font.
"""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_the_self_hosted_fonts_are_served_as_woff2(test_client: TestClient) -> None:
    response = test_client.get("/static/fonts/barlow-condensed-latin-800-italic.woff2")
    assert response.status_code == 200
    assert response.headers["content-type"] == "font/woff2"
    assert response.content[:4] == b"wOF2"
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_broadcast_theme_css.py -q`, then `uv run pytest tests/api/test_static_fonts.py -q`

Expected:
- **Fail:** the token, component-class and `@font-face` tests, because the sheet has no Broadcast tokens yet.
- **Fail:** the font MIME test, with `text/plain` (or similar) instead of `font/woff2`.
- **Pass:** the `wOF2` and licence tests, because Step 2 already downloaded those files.

- [ ] **Step 5: Replace `web/static/input.css`**

```css
@import "tailwindcss" source(none);

/* source(none) turns off Tailwind v4's automatic scan of the WHOLE repo (which would pull class
   names out of .md, .py and test files and change this sheet whenever they change), so only the
   folders named here are scanned. Task 17 adds @source "../../api/charts"; for the class strings
   that api/charts/core.py emits from Python. */
@source "../../web/templates";

/* ============================================================================================
   Self-hosted fonts (SIL Open Font License; the texts ship in web/static/fonts/OFL-*.txt).
   Absolute /static/ URLs on purpose: the compiled sheet lives in web/static/css/ and Tailwind
   does not rebase relative url()s, so a relative path would resolve against the wrong folder.
   ============================================================================================ */
@font-face {
  font-family: "Barlow Condensed";
  font-style: normal;
  font-weight: 600;
  font-display: swap;
  src: url("/static/fonts/barlow-condensed-latin-600-normal.woff2") format("woff2");
}
@font-face {
  font-family: "Barlow Condensed";
  font-style: normal;
  font-weight: 700;
  font-display: swap;
  src: url("/static/fonts/barlow-condensed-latin-700-normal.woff2") format("woff2");
}
@font-face {
  font-family: "Barlow Condensed";
  font-style: normal;
  font-weight: 800;
  font-display: swap;
  src: url("/static/fonts/barlow-condensed-latin-800-normal.woff2") format("woff2");
}
@font-face {
  font-family: "Barlow Condensed";
  font-style: italic;
  font-weight: 700;
  font-display: swap;
  src: url("/static/fonts/barlow-condensed-latin-700-italic.woff2") format("woff2");
}
@font-face {
  font-family: "Barlow Condensed";
  font-style: italic;
  font-weight: 800;
  font-display: swap;
  src: url("/static/fonts/barlow-condensed-latin-800-italic.woff2") format("woff2");
}
@font-face {
  font-family: "Inter";
  font-style: normal;
  font-weight: 100 900;
  font-display: swap;
  src: url("/static/fonts/inter-latin-wght-normal.woff2") format("woff2");
}
@font-face {
  font-family: "JetBrains Mono";
  font-style: normal;
  font-weight: 500;
  font-display: swap;
  src: url("/static/fonts/jetbrains-mono-latin-500-normal.woff2") format("woff2");
}
@font-face {
  font-family: "JetBrains Mono";
  font-style: normal;
  font-weight: 700;
  font-display: swap;
  src: url("/static/fonts/jetbrains-mono-latin-700-normal.woff2") format("woff2");
}

/* ============================================================================================
   Design tokens (spec section 5). `static` so every variable is emitted even before a utility
   uses it: custom.css, the component classes below and inline styles such as
   style="--tile-accent: var(--color-target-wp)" all read these variables directly.
   ============================================================================================ */
@theme static {
  --color-ink: #0B0F17;
  --color-ink-2: #0E1320;
  --color-panel: #151B29;
  --color-panel-2: #1D2436;
  --color-line: rgb(255 255 255 / 0.08);
  --color-fg: #F3F5F9;
  --color-muted: #8A93A8;
  --color-dim: #5D667C;
  --color-accent: #FFD400;
  --color-target-wp: #FFD400;
  --color-target-ats: #4CC9F0;
  --color-target-ou: #C77DFF;

  --font-display: "Barlow Condensed", "Arial Narrow", sans-serif;
  --font-sans: "Inter", system-ui, sans-serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;
}

/* LEGACY -- the pre-redesign palette, kept ONLY so templates not yet restyled keep rendering
   while the branch is in progress. Not `static`: it emits only while something still uses it.
   Task 20 deletes this block once `grep -rn "nfl-" web/templates api` finds nothing. */
@theme {
  --color-nfl-primary: #013369;
  --color-nfl-secondary: #D50A0A;
  --color-nfl-accent: #FF8C00;
  --color-nfl-dark: #1a1a1a;
  --color-nfl-light: #F8F9FA;
}

@layer base {
  html {
    color-scheme: dark;
    background-color: var(--color-ink);
  }

  body {
    background-image: linear-gradient(180deg, var(--color-ink) 0%, var(--color-ink-2) 100%);
    background-attachment: fixed;
    color: var(--color-fg);
  }

  ::selection {
    background: rgb(255 212 0 / 0.35);
    color: #FFFFFF;
  }
}

/* ============================================================================================
   Component classes (contract section 2). Plain CSS inside @layer components is always emitted,
   and Tailwind utilities (a higher layer) still override it -- so value_class="text-green-400"
   on a .stat-tile-value wins over the tile's default colour.

   SLANT: every skewed box (.tag, .tag-ghost, .skew-control, .edge-chip, .team-block) transforms
   the ELEMENT, and its text sits in a <span class="unskew"> child that counter-skews it upright.
   The one exception is <select class="skew-control">, which cannot hold a child: it is clipped to
   a parallelogram instead of transformed.
   ============================================================================================ */
@layer components {
  /* -- Primitives ------------------------------------------------------------------------- */
  .skew {
    transform: skewX(-12deg);
  }

  .unskew {
    display: inline-block;
    transform: skewX(12deg);
  }

  .display {
    font-family: var(--font-display);
    font-style: italic;
    font-weight: 800;
    line-height: 0.95;
    letter-spacing: 0.01em;
    text-transform: uppercase;
  }

  .label {
    font-family: var(--font-display);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--color-muted);
  }

  .num,
  .stat-num {
    font-family: var(--font-mono);
    font-variant-numeric: tabular-nums;
  }

  .wordmark {
    font-family: var(--font-display);
    font-size: 22px;
    font-style: italic;
    font-weight: 800;
    letter-spacing: 0.01em;
    text-transform: uppercase;
    text-decoration: none;
    color: var(--color-fg);
  }

  .skip-link {
    position: absolute;
    top: -48px;
    left: 16px;
    z-index: 50;
    padding: 8px 12px;
    background: var(--color-accent);
    color: var(--color-ink);
    font-weight: 700;
  }

  .skip-link:focus {
    top: 8px;
  }

  /* -- Surfaces --------------------------------------------------------------------------- */
  .panel {
    background-color: var(--color-panel);
    border-radius: 4px;
    padding: 14px 16px;
  }

  .panel-title {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 0 0 10px;
    font-family: var(--font-display);
    font-size: 18px;
    font-style: italic;
    font-weight: 800;
    letter-spacing: 0.03em;
    text-transform: uppercase;
    color: var(--color-fg);
  }

  .panel-title::before {
    content: "";
    flex: none;
    width: 4px;
    height: 16px;
    background: var(--color-accent);
    transform: skewX(-12deg);
  }

  /* -- Section heads and tags ------------------------------------------------------------- */
  /* A plain flex row: no margin and no rule of its own. The rule is an explicit
     <span class="flex-1 h-px bg-line"> child, so a badge placed after the heading sits inside the
     row's first </div> (tests/api/test_bets_page.py reads the header up to that tag). */
  .section-head {
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }

  .tag,
  .tag-ghost {
    display: inline-flex;
    align-items: center;
    margin: 0;
    padding: 2px 12px;
    transform: skewX(-12deg);
    font-family: var(--font-display);
    font-size: 15px;
    font-style: italic;
    font-weight: 800;
    line-height: 1.4;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .tag {
    background: var(--color-accent);
    color: var(--color-ink);
  }

  .tag-ghost {
    background: transparent;
    border: 1.5px solid rgb(255 255 255 / 0.5);
    color: var(--color-fg);
  }

  /* Under 640px a long tag ("Backtest replay -- reconstructed after the fact") is wider than a
     390px phone, so it wraps onto a second line instead of pushing the page sideways (spec 10). */
  @media (max-width: 639.98px) {
    .tag,
    .tag-ghost {
      white-space: normal;
    }
  }

  /* -- Controls: nav links, tabs, buttons, selects ------------------------------------------ */
  /* A 44px touch target that PAINTS a 32px slanted band: the background is a gradient sized to
     the middle of the box, and the whole element is skewed (its text sits in a .unskew child). */
  .skew-control {
    --skew-bg: var(--color-panel-2);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    min-width: 44px;
    min-height: 44px;
    padding: 0 14px;
    border: 0;
    background: linear-gradient(var(--skew-bg), var(--skew-bg)) center / 100% calc(100% - 12px) no-repeat;
    transform: skewX(-12deg);
    font-family: var(--font-display);
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.02em;
    color: var(--color-fg);
    text-decoration: none;
    cursor: pointer;
  }

  .skew-control:hover {
    --skew-bg: #283149;
  }

  .skew-control:disabled,
  .skew-control[aria-disabled="true"] {
    opacity: 0.4;
    cursor: not-allowed;
  }

  /* A <select> cannot hold a .unskew child, so it is never transformed: it is clipped to the same
     32px parallelogram instead, and its text (the options) stays upright on its own. */
  select.skew-control {
    appearance: none;
    transform: none;
    padding-right: 32px;
    background-color: var(--skew-bg);
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 10 6'%3E%3Cpath d='M1 1l4 4 4-4' fill='none' stroke='%23FFD400' stroke-width='1.6'/%3E%3C/svg%3E");
    background-position: right 14px center;
    background-repeat: no-repeat;
    background-size: 10px 6px;
    clip-path: polygon(7px 6px, 100% 6px, calc(100% - 7px) calc(100% - 6px), 0 calc(100% - 6px));
  }

  /* The clip-path would cut a focus outline off, so focus is shown by colour inside the shape. */
  select.skew-control:focus-visible {
    --skew-bg: #2E3850;
    outline: none;
    color: var(--color-accent);
  }

  select.skew-control option {
    background: var(--color-panel);
    color: var(--color-fg);
  }

  .nav-link {
    --skew-bg: transparent;
    font-family: var(--font-sans);
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0;
    color: var(--color-muted);
  }

  .nav-link:hover {
    --skew-bg: var(--color-panel-2);
    color: var(--color-fg);
  }

  .skew-control-active,
  .skew-control-active:hover,
  .nav-link.skew-control-active {
    --skew-bg: var(--color-accent);
    color: var(--color-ink);
  }

  /* -- Team blocks and score bugs --------------------------------------------------------- */
  .team-block {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 46px;
    height: 26px;
    margin-left: 5px;
    padding: 0 6px;
    background: var(--team-bg, var(--color-panel-2));
    color: var(--team-fg, #FFFFFF);
    transform: skewX(-12deg);
    font-family: var(--font-display);
    font-size: 17px;
    font-style: italic;
    font-weight: 800;
    letter-spacing: 0.02em;
  }

  .team-block-lg {
    min-width: 86px;
    height: 58px;
    font-size: 34px;
  }

  .score-bug {
    display: grid;
    gap: 3px;
  }

  .score-row {
    display: grid;
    grid-template-columns: 56px 1fr 64px;
    align-items: stretch;
    height: 30px;
  }

  .score-row .team-block {
    min-width: 0;
    height: 100%;
  }

  .score-name {
    display: flex;
    align-items: center;
    padding-left: 10px;
    overflow: hidden;
    background: var(--color-panel-2);
    font-family: var(--font-display);
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 0.04em;
    text-overflow: ellipsis;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .score-value {
    display: flex;
    align-items: center;
    justify-content: flex-end;
    padding-right: 8px;
    background: var(--color-ink);
    font-family: var(--font-display);
    font-size: 19px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
  }

  .score-row.is-fav .score-value {
    color: var(--color-accent);
  }

  .score-row.is-dog .score-name,
  .score-row.is-dog .score-value {
    color: var(--color-muted);
  }

  /* -- Edge chips, monochrome bands, evidence chips --------------------------------------- */
  .edge-chip,
  .edge-chip-soft {
    display: inline-flex;
    align-items: center;
    padding: 0 8px;
    transform: skewX(-12deg);
    font-family: var(--font-display);
    font-size: 15px;
    font-style: italic;
    font-weight: 800;
    line-height: 1.35;
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  .edge-chip {
    background: var(--color-accent);
    color: var(--color-ink);
  }

  /* Declared after .edge-chip so `class="edge-chip edge-chip-soft"` resolves to the outline. */
  .edge-chip-soft {
    background: transparent;
    border: 1px solid rgb(255 212 0 / 0.45);
    color: var(--color-accent);
  }

  .band {
    display: inline-flex;
    align-items: center;
    padding: 1px 8px;
    font-family: var(--font-display);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    white-space: nowrap;
  }

  .band-high {
    background: #E6E9F0;
    color: var(--color-ink);
  }

  .band-medium {
    background: #5B6478;
    color: #FFFFFF;
  }

  .band-low {
    border: 1px solid #5B6478;
    color: #C4CAD8;
  }

  /* Wraps rather than clips: the longest provenance label is 30 characters (UI-SPEC E10). */
  .evidence-chip {
    display: inline-flex;
    align-items: center;
    padding: 1px 8px;
    border: 1px solid rgb(255 255 255 / 0.28);
    border-radius: 999px;
    font-size: 11px;
    font-weight: 600;
    color: #D5DAE5;
  }

  .evidence-chip-strong {
    background: rgb(255 255 255 / 0.12);
    color: #FFFFFF;
  }

  /* -- Scoreboard tiles ------------------------------------------------------------------- */
  .stat-tile {
    padding: 10px 14px;
    background: var(--color-panel);
    border-top: 3px solid var(--tile-accent, rgb(255 255 255 / 0.12));
  }

  .stat-tile-value {
    margin-top: 2px;
    font-family: var(--font-display);
    font-size: 34px;
    font-weight: 800;
    line-height: 1.05;
    font-variant-numeric: tabular-nums;
    color: var(--color-fg);
  }

  /* 12px, not 11px: spec 5 allows 11px only for uppercase labels, and this line is lowercase. */
  .stat-tile-sub {
    font-size: 12px;
    color: var(--color-muted);
  }

  /* -- Bet slips: the visual container only. pages/bets.html (Task 11) lays each slip out with
     grid utilities -- seven columns on lg+, rank + stacked content below 1024px -- so the layout
     lives next to the markup it arranges and no component grid competes with it. ---------- */
  .bet-slip {
    min-height: 64px;
    background: var(--color-panel);
    border-left: 4px solid var(--color-accent);
    transition: background-color 150ms ease;
  }

  .bet-slip:hover {
    background: #1A2133;
  }

  /* -- Honesty note: the condensed one-liner <details> (owner decision O4) ------------------ */
  .honesty-note {
    border: 1px solid rgb(255 255 255 / 0.16);
    border-radius: 4px;
    font-size: 12px;
    color: #C4CAD8;
  }

  /* The summary is the note's only control, so it keeps the 44px touch target (spec 10). */
  .honesty-note > summary {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 4px 10px;
    min-height: 44px;
    padding: 7px 12px;
    list-style: none;
    cursor: pointer;
  }

  .honesty-note > summary::-webkit-details-marker {
    display: none;
  }

  .honesty-note-key {
    font-family: var(--font-display);
    font-size: 12px;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--color-fg);
  }

  .honesty-note-line {
    flex: 1 1 18rem;
  }

  .honesty-note-why {
    margin-left: auto;
    font-weight: 600;
    color: var(--color-accent);
  }

  .honesty-note-why::after {
    content: " \25BE";
  }

  .honesty-note[open] .honesty-note-why::after {
    content: " \25B4";
  }

  .honesty-note-body {
    max-width: 48rem;
    padding: 0 12px 10px;
    line-height: 1.55;
    color: var(--color-muted);
  }

  /* -- Motion: the win-probability bar grows in from the left once, on load (spec 5). Readers
     who ask for reduced motion get the final width at once (spec 10). ---------------------- */
  .grow-in {
    transform-origin: left center;
    animation: grow-in 600ms cubic-bezier(0.22, 1, 0.36, 1) both;
  }

  @media (prefers-reduced-motion: reduce) {
    .grow-in {
      animation: none;
    }
  }
}

@keyframes grow-in {
  from {
    transform: scaleX(0);
  }
  to {
    transform: scaleX(1);
  }
}
```

- [ ] **Step 6: Register the font MIME type in `api/main.py`**

Edit 1 -- old:

```python
import threading
from contextlib import asynccontextmanager
```

new:

```python
import mimetypes
import threading
from contextlib import asynccontextmanager
```

Edit 2 -- old:

```python
# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
```

new:

```python
# Mount static files. The self-hosted fonts are .woff2, which the Windows MIME registry does not
# know: Starlette would serve them as text/plain and a browser honouring nosniff would refuse
# them. Registering the type before the mount makes every font response say font/woff2.
mimetypes.add_type("font/woff2", ".woff2")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
```

- [ ] **Step 7: Compile the stylesheet**

```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
```

Expected:
- The run ends with a `Done in ...` line.
- `grep -c "@font-face" web/static/css/tailwind-compiled.css` prints `8`.
- `grep -o '\.top-25' web/static/css/tailwind-compiled.css | wc -l` prints `0`: `source(none)` is in force, so a class that only a Markdown file mentions (`SELECTION-CENSUS.md`) is no longer compiled.

- [ ] **Step 8: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_broadcast_theme_css.py -q
uv run pytest tests/api/test_static_fonts.py -q
uv run pytest tests/unit/test_page_labels.py::TestThePartial -q
uv run ruff check api/main.py tests/unit/test_broadcast_theme_css.py tests/api/test_static_fonts.py
uv run ruff format api/main.py tests/unit/test_broadcast_theme_css.py tests/api/test_static_fonts.py
uv run pyright api/main.py tests/unit/test_broadcast_theme_css.py tests/api/test_static_fonts.py
```

Expected:
- All tests PASS. `TestThePartial` still passes: the old-rule partial's existing classes are still in the recompiled sheet.
- Ruff is clean; `ruff format` reports nothing to change, or reformats only the new test files.
- pyright reports 0 errors.

- [ ] **Step 9: Commit**

```bash
git add web/static/fonts web/static/input.css web/static/css/tailwind-compiled.css api/main.py tests/unit/test_broadcast_theme_css.py tests/api/test_static_fonts.py
git commit -m "$(cat <<'EOF'
feat(redesign): Broadcast design tokens, component classes and self-hosted fonts

Adds the dark Broadcast tokens (@theme static), the shared component classes
(tags, skew controls, team blocks, score bugs, edge chips, monochrome bands,
evidence chips, stat tiles, bet slips, honesty notes) and self-hosted
Barlow Condensed / Inter / JetBrains Mono with their OFL licences. Registers
the font/woff2 MIME type so the fonts are not served as text/plain. The old
nfl-* tokens stay in a marked legacy block until every template is restyled.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Presentation helper -- team colours, nicknames, kickoff labels and TV windows

**Files:**
- Create: `api/presentation.py`
- Modify: `api/dependencies.py` (one import; two Jinja globals after the two filters at lines 168-169)
- Test: `tests/unit/test_presentation.py` (new)

**Interfaces:**
- Consumes: `utils.team_data.get_team_info(abbr) -> dict` (keys `name`, `colors`), `utils.team_data.validate_team_abbreviation(abbr) -> bool` (accepts aliases such as `LAR`, `OAK`, `JAC`) and `utils.date_utils.ET` (the existing `ZoneInfo("America/New_York")` constant).
- Produces (exact, contract section 4 plus the extras):
  - `class TeamColors(NamedTuple): bg: str; fg: str`
  - `team_block_colors(abbr: str | None) -> TeamColors`
  - `team_nickname(abbr: str | None) -> str`
  - `kickoff_label(game_date: datetime | date | str | None) -> str`
  - `kickoff_window(game_date: datetime | date | str | None) -> str`
  - `decorate_game(game: dict[str, Any]) -> dict[str, Any]`
  - `group_games_by_window(games: list[dict[str, Any]]) -> list[dict[str, Any]]`
  - `contrast_ratio(a: str, b: str) -> float`
  - constants `PANEL_HEX`, `MIN_BLOCK_CONTRAST`, `UNKNOWN_BLOCK`, `TEXT_LIGHT`, `TEXT_DARK`, `TIME_TBD`
  - Jinja globals `team_colors` and `team_nickname`

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_presentation.py`:

```python
"""Unit tests for api.presentation (redesign Task 2).

Pins review-focus items 2 and 3: near-black team colours and unknown teams must still render a
readable block, and odd kickoffs (an international morning game, a Saturday slate, a Christmas
weekday, a Monday doubleheader, no date at all) must land in a sensible, labelled group.

ASCII only, no emoji (CLAUDE.md). The window separator is U+00B7, written as an escape.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from api.presentation import (
    MIN_BLOCK_CONTRAST,
    PANEL_HEX,
    TEXT_DARK,
    TEXT_LIGHT,
    TIME_TBD,
    UNKNOWN_BLOCK,
    TeamColors,
    contrast_ratio,
    decorate_game,
    group_games_by_window,
    kickoff_label,
    kickoff_window,
    team_block_colors,
    team_nickname,
)
from utils.team_data import NFL_TEAMS_DATA

DOT = "\u00b7"


class TestTeamBlockColors:
    def test_a_readable_primary_is_kept(self) -> None:
        assert team_block_colors("KC") == TeamColors("#E31837", TEXT_LIGHT)

    def test_dark_but_distinct_primaries_keep_their_identity(self) -> None:
        assert team_block_colors("NYG").bg == "#0B2265"
        assert team_block_colors("BAL").bg == "#241773"

    @pytest.mark.parametrize(
        ("abbr", "expected"),
        [
            ("CHI", TeamColors("#C83803", TEXT_LIGHT)),
            ("HOU", TeamColors("#A71930", TEXT_LIGHT)),
            ("NE", TeamColors("#C60C30", TEXT_LIGHT)),
            ("SEA", TeamColors("#69BE28", TEXT_DARK)),
            ("TEN", TeamColors("#4B92DB", TEXT_DARK)),
            ("CLE", TeamColors("#FF3C00", TEXT_DARK)),
        ],
    )
    def test_a_near_black_primary_falls_back_to_the_secondary(
        self, abbr: str, expected: TeamColors
    ) -> None:
        assert team_block_colors(abbr) == expected

    def test_a_pure_black_primary_is_never_a_block(self) -> None:
        assert team_block_colors("LV") == TeamColors("#A5ACAF", TEXT_DARK)

    def test_aliases_resolve_to_the_same_team(self) -> None:
        assert team_block_colors("LAR") == team_block_colors("LA")
        assert team_block_colors("OAK") == team_block_colors("LV")
        assert team_block_colors("kc") == team_block_colors("KC")

    @pytest.mark.parametrize("abbr", [None, "", "XYZ"])
    def test_an_unknown_team_gets_a_neutral_block(self, abbr: str | None) -> None:
        assert team_block_colors(abbr) == TeamColors(UNKNOWN_BLOCK, TEXT_LIGHT)

    @pytest.mark.parametrize("abbr", sorted(NFL_TEAMS_DATA))
    def test_every_team_block_reads_on_the_panel(self, abbr: str) -> None:
        colors = team_block_colors(abbr)
        assert colors.bg not in {"#000000", "#FFFFFF"}
        assert contrast_ratio(colors.bg, PANEL_HEX) >= MIN_BLOCK_CONTRAST
        # The text on the block: the better of white and ink is never below 4.4 for these colours.
        assert contrast_ratio(colors.fg, colors.bg) >= 4.4


class TestTeamNickname:
    @pytest.mark.parametrize(
        ("abbr", "expected"),
        [
            ("KC", "Chiefs"),
            ("SF", "49ers"),
            ("WAS", "Commanders"),
            ("LAR", "Rams"),
            ("XYZ", "XYZ"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_nickname(self, abbr: str | None, expected: str) -> None:
        assert team_nickname(abbr) == expected


class TestKickoffLabel:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (datetime(2026, 9, 27, 13, 0), "Sun 1:00 PM ET"),
            ("2026-09-27 16:25:00", "Sun 4:25 PM ET"),
            (datetime(2026, 9, 28, 0, 20, tzinfo=UTC), "Sun 8:20 PM ET"),
            ("2026-09-27T17:00:00+00:00", "Sun 1:00 PM ET"),
            (datetime(2026, 9, 27, 9, 30), "Sun 9:30 AM ET"),
            ("2024-09-10", "Tue Sep 10"),
            (datetime(2024, 9, 10), "Tue Sep 10"),
            (date(2024, 9, 10), "Tue Sep 10"),
            (None, TIME_TBD),
            ("", TIME_TBD),
            ("not a date", TIME_TBD),
        ],
    )
    def test_label(self, value: datetime | date | str | None, expected: str) -> None:
        assert kickoff_label(value) == expected

    def test_a_pandas_not_a_time_is_time_tbd(self) -> None:
        assert kickoff_label(pd.NaT) == TIME_TBD  # pyright: ignore[reportArgumentType]


class TestKickoffWindow:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (datetime(2026, 9, 24, 20, 15), "Thursday Night"),
            (datetime(2026, 9, 27, 9, 30), f"Sunday {DOT} 9:30 AM ET"),
            (datetime(2026, 9, 27, 13, 0), f"Sunday {DOT} 1:00 PM ET"),
            (datetime(2026, 9, 27, 16, 25), f"Sunday {DOT} 4:25 PM ET"),
            (datetime(2026, 9, 27, 20, 20), "Sunday Night"),
            (datetime(2026, 9, 28, 19, 15), "Monday Night"),
            ("2026-09-27", "Sunday"),
            (None, TIME_TBD),
        ],
    )
    def test_window(self, value: datetime | str | None, expected: str) -> None:
        assert kickoff_window(value) == expected


def _games(*kickoffs: datetime | None) -> list[dict]:
    return [{"game_id": f"g{i}", "game_date": k} for i, k in enumerate(kickoffs)]


class TestGroupGamesByWindow:
    def test_a_full_week_groups_in_first_appearance_order(self) -> None:
        games = _games(
            datetime(2026, 9, 24, 20, 15),
            datetime(2026, 9, 27, 9, 30),
            datetime(2026, 9, 27, 13, 0),
            datetime(2026, 9, 27, 13, 0),
            datetime(2026, 9, 27, 16, 5),
            datetime(2026, 9, 27, 16, 25),
            datetime(2026, 9, 27, 20, 20),
            datetime(2026, 9, 28, 19, 15),
            datetime(2026, 9, 28, 20, 15),
        )
        groups = group_games_by_window(games)
        assert [g["label"] for g in groups] == [
            "Thursday Night",
            f"Sunday {DOT} 9:30 AM ET",
            f"Sunday {DOT} 1:00 PM ET",
            f"Sunday {DOT} 4:05 / 4:25 PM ET",
            "Sunday Night",
            "Monday Night",
        ]
        assert [len(g["games"]) for g in groups] == [1, 1, 2, 2, 1, 2]
        # Input order is kept inside a group.
        assert [g["game_id"] for g in groups[5]["games"]] == ["g7", "g8"]

    def test_a_saturday_late_season_slate(self) -> None:
        groups = group_games_by_window(
            _games(
                datetime(2025, 12, 20, 16, 30),
                datetime(2025, 12, 20, 20, 0),
                datetime(2025, 12, 21, 13, 0),
            )
        )
        assert [g["label"] for g in groups] == [
            f"Saturday {DOT} 4:30 PM ET",
            "Saturday Night",
            f"Sunday {DOT} 1:00 PM ET",
        ]

    def test_a_christmas_day_weekday(self) -> None:
        groups = group_games_by_window(
            _games(datetime(2024, 12, 25, 13, 0), datetime(2024, 12, 25, 16, 30))
        )
        assert [g["label"] for g in groups] == [
            f"Wednesday {DOT} 1:00 PM ET",
            f"Wednesday {DOT} 4:30 PM ET",
        ]

    def test_a_game_with_no_date_lands_in_time_tbd(self) -> None:
        groups = group_games_by_window(_games(datetime(2026, 9, 27, 13, 0), None))
        assert [g["label"] for g in groups] == [f"Sunday {DOT} 1:00 PM ET", TIME_TBD]
        assert groups[1]["games"][0]["game_id"] == "g1"

    def test_an_empty_week_has_no_groups(self) -> None:
        assert group_games_by_window([]) == []


class TestDecorateGame:
    def test_it_adds_the_presentation_fields_without_mutating_the_input(self) -> None:
        game = {
            "game_id": "2026_W03_KC@MIA",
            "away_team": "KC",
            "home_team": "MIA",
            "game_date": datetime(2026, 9, 27, 13, 0),
            "wp_prob": 0.412,
        }
        decorated = decorate_game(game)
        assert decorated is not game
        assert "away_color" not in game
        assert decorated["wp_prob"] == 0.412
        assert decorated["away_color"] == "#E31837"
        assert decorated["away_fg"] == TEXT_LIGHT
        assert decorated["home_color"] == "#008E97"
        assert decorated["home_fg"] == TEXT_DARK
        assert decorated["away_name"] == "Chiefs"
        assert decorated["home_name"] == "Dolphins"
        assert decorated["kickoff_label"] == "Sun 1:00 PM ET"
        assert decorated["window_label"] == f"Sunday {DOT} 1:00 PM ET"

    def test_an_unknown_team_and_missing_date_still_decorate(self) -> None:
        decorated = decorate_game({"away_team": "XYZ", "home_team": None})
        assert decorated["away_color"] == UNKNOWN_BLOCK
        assert decorated["away_name"] == "XYZ"
        assert decorated["home_name"] == ""
        assert decorated["kickoff_label"] == TIME_TBD
        assert decorated["window_label"] == TIME_TBD


def test_the_helpers_are_jinja_globals() -> None:
    from api.dependencies import templates

    rendered = templates.env.from_string(
        "{{ team_colors('KC').bg }} {{ team_colors('KC').fg }} {{ team_nickname('KC') }}"
    ).render()
    assert rendered == "#E31837 #FFFFFF Chiefs"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_presentation.py -q`
Expected: FAIL at collection, with `ModuleNotFoundError: No module named 'api.presentation'`.

- [ ] **Step 3: Write `api/presentation.py`**

```python
"""Presentation helpers for the Broadcast UI: team colours, nicknames and kickoff windows.

Pure functions over static team data and a game's own kickoff timestamp. Nothing here reads the
cache or computes a metric (UIAP-01): which colour a team's block uses, what a team is called and
which TV window a kickoff falls in are formatting decisions, made identically on every request.

``predictions.game_date`` is a naive TIMESTAMP already in US Eastern time -- a 1 PM ET kickoff is
stored as ``13:00:00`` -- so a naive value is read as Eastern as-is, and an aware one is converted.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, NamedTuple

# The project's one America/New_York constant, not a third copy of it.
from utils.date_utils import ET
from utils.team_data import get_team_info, validate_team_abbreviation

#: The card colour team blocks sit on (``--color-panel`` in web/static/input.css).
PANEL_HEX = "#151B29"
#: Below this contrast against the panel a block stops reading as a shape: the near-black
#: primaries (CHI, HOU, NE, SEA, TEN, CLE's brown) vanish into the card. Dark-but-distinct
#: primaries such as NYG's navy (1.17) and BAL's purple (1.18) stay above it and keep their colour.
MIN_BLOCK_CONTRAST = 1.15
TEXT_LIGHT = "#FFFFFF"
TEXT_DARK = "#0B0F17"
#: An abbreviation utils.team_data does not know: a neutral panel-2 block, never a guess.
UNKNOWN_BLOCK = "#1D2436"
#: A known team whose primary and secondary are both unusable. No current team reaches this.
FALLBACK_BLOCK = "#5B6478"
#: Pure black disappears on the dark panel, and pure white is never a team's identity colour.
_NEVER_A_BLOCK = frozenset({"#000000", "#FFFFFF"})

TIME_TBD = "Time TBD"
_DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
_DAY_ABBR = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")
_MONTH_ABBR = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)
#: TV-window bands by ET kickoff hour. 7 PM ET or later is a national night window on any day.
_NIGHT_FROM_HOUR = 19
_LATE_FROM_HOUR = 16
_EARLY_FROM_HOUR = 12
_DOT = "\u00b7"
_TBD_KEY: tuple[str, ...] = ("tbd",)


class TeamColors(NamedTuple):
    """A team block's background, and the text colour that reads on it."""

    bg: str
    fg: str


def _channel(value: int) -> float:
    c = value / 255
    return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4


def _luminance(hex_color: str) -> float:
    """WCAG relative luminance of a ``#RRGGBB`` colour."""
    digits = hex_color.lstrip("#")
    r, g, b = (int(digits[i : i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * _channel(r) + 0.7152 * _channel(g) + 0.0722 * _channel(b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two ``#RRGGBB`` colours, from 1.0 to 21.0."""
    high, low = sorted((_luminance(a), _luminance(b)), reverse=True)
    return (high + 0.05) / (low + 0.05)


def _usable_block(hex_color: str) -> bool:
    return (
        hex_color.upper() not in _NEVER_A_BLOCK
        and contrast_ratio(hex_color, PANEL_HEX) >= MIN_BLOCK_CONTRAST
    )


def _text_on(bg: str) -> str:
    """White or ink, whichever contrasts more with *bg*."""
    if contrast_ratio(bg, TEXT_LIGHT) >= contrast_ratio(bg, TEXT_DARK):
        return TEXT_LIGHT
    return TEXT_DARK


def team_block_colors(abbr: str | None) -> TeamColors:
    """The block colour for *abbr*: its primary, else its secondary, else a neutral slate.

    A colour is usable when it is neither pure black nor pure white and reads as a shape on the
    dark panel. An unknown abbreviation gets a neutral block rather than a guessed team colour.
    """
    if not abbr or not validate_team_abbreviation(abbr):
        return TeamColors(UNKNOWN_BLOCK, TEXT_LIGHT)
    primary, secondary = (color.upper() for color in get_team_info(abbr)["colors"][:2])
    for candidate in (primary, secondary):
        if _usable_block(candidate):
            return TeamColors(candidate, _text_on(candidate))
    return TeamColors(FALLBACK_BLOCK, _text_on(FALLBACK_BLOCK))


def team_nickname(abbr: str | None) -> str:
    """``"KC"`` -> ``"Chiefs"``. An unknown abbreviation is returned as-is; ``None`` is ``""``."""
    if not abbr:
        return ""
    if not validate_team_abbreviation(abbr):
        return abbr
    # Every current team name ends in its nickname ("San Francisco 49ers" -> "49ers").
    return str(get_team_info(abbr)["name"]).rsplit(" ", 1)[-1]


def _to_eastern(game_date: datetime | date | str | None) -> tuple[date, datetime | None] | None:
    """(ET calendar date, ET kickoff or None when no time is known), or None when unknown."""
    if game_date is None:
        return None
    value: datetime | date | str = game_date
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            value = datetime.fromisoformat(text)
        except ValueError:
            return None
    if isinstance(value, datetime):
        # pandas' NaT is a datetime that is not equal to itself.
        if value != value:  # noqa: PLR0124
            return None
        if value.tzinfo is not None:
            value = value.astimezone(ET)
        # A midnight stamp is a date with no kickoff time: no NFL game kicks off at 00:00 ET.
        if (value.hour, value.minute, value.second) == (0, 0, 0):
            return value.date(), None
        return value.date(), value
    if isinstance(value, date):
        return value, None
    return None


def _clock(moment: datetime) -> str:
    return f"{moment.hour % 12 or 12}:{moment.minute:02d}"


def _meridiem(moment: datetime) -> str:
    return "AM" if moment.hour < 12 else "PM"


def _band(moment: datetime) -> str:
    if moment.hour >= _NIGHT_FROM_HOUR:
        return "night"
    if moment.hour >= _LATE_FROM_HOUR:
        return "late"
    if moment.hour >= _EARLY_FROM_HOUR:
        return "early"
    return "morning"


def _window_label(day: date, band: str, moments: list[datetime]) -> str:
    weekday = _DAY_NAMES[day.weekday()]
    if band == "allday" or not moments:
        return weekday
    if band == "night":
        return f"{weekday} Night"
    ordered = sorted(moments)
    clocks: list[str] = []
    for moment in ordered:
        clock = _clock(moment)
        if clock not in clocks:
            clocks.append(clock)
    # Every time in one band shares its AM/PM (bands split at noon), so one suffix is honest.
    return f"{weekday} {_DOT} {' / '.join(clocks)} {_meridiem(ordered[0])} ET"


def kickoff_label(game_date: datetime | date | str | None) -> str:
    """``"Sun 1:00 PM ET"``; a date with no time ``"Tue Sep 10"``; nothing known ``"Time TBD"``."""
    parsed = _to_eastern(game_date)
    if parsed is None:
        return TIME_TBD
    day, moment = parsed
    if moment is None:
        return f"{_DAY_ABBR[day.weekday()]} {_MONTH_ABBR[day.month - 1]} {day.day}"
    return f"{_DAY_ABBR[day.weekday()]} {_clock(moment)} {_meridiem(moment)} ET"


def kickoff_window(game_date: datetime | date | str | None) -> str:
    """The TV-window label for ONE game: ``"Sunday Night"``, ``"Sunday - 4:25 PM ET"``, ...

    (The separator written ``-`` here is really U+00B7, a middle dot; see ``_DOT``.)
    ``group_games_by_window`` builds the label for a whole group, which lists every distinct
    time in the group ("Sunday - 4:05 / 4:25 PM ET"); pages print the group's label.
    """
    parsed = _to_eastern(game_date)
    if parsed is None:
        return TIME_TBD
    day, moment = parsed
    if moment is None:
        return _window_label(day, "allday", [])
    return _window_label(day, _band(moment), [moment])


def decorate_game(game: dict[str, Any]) -> dict[str, Any]:
    """A NEW dict: *game* plus the presentation fields the Broadcast templates read."""
    away = game.get("away_team")
    home = game.get("home_team")
    away_colors = team_block_colors(away)
    home_colors = team_block_colors(home)
    return {
        **game,
        "away_color": away_colors.bg,
        "away_fg": away_colors.fg,
        "home_color": home_colors.bg,
        "home_fg": home_colors.fg,
        "away_name": team_nickname(away),
        "home_name": team_nickname(home),
        "kickoff_label": kickoff_label(game.get("game_date")),
        "window_label": kickoff_window(game.get("game_date")),
    }


def group_games_by_window(games: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """``[{"label": str, "games": [...]}]`` in first-appearance order; input order kept inside.

    A window is an ET calendar day plus a band (morning, early, late, night). A date with no time
    is its own all-day group, and a game with no date at all lands in ``"Time TBD"``.
    """
    members: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    days: dict[tuple[str, ...], date] = {}
    moments: dict[tuple[str, ...], list[datetime]] = {}
    for game in games:
        parsed = _to_eastern(game.get("game_date"))
        key: tuple[str, ...]
        if parsed is None:
            key = _TBD_KEY
        else:
            day, moment = parsed
            key = (day.isoformat(), "allday" if moment is None else _band(moment))
            days[key] = day
            if moment is not None:
                moments.setdefault(key, []).append(moment)
        members.setdefault(key, []).append(game)
    groups: list[dict[str, Any]] = []
    for key, members_in_window in members.items():
        if key == _TBD_KEY:
            label = TIME_TBD
        else:
            label = _window_label(days[key], key[1], moments.get(key, []))
        groups.append({"label": label, "games": members_in_window})
    return groups
```

- [ ] **Step 4: Register the Jinja globals in `api/dependencies.py`**

Edit 1 -- old:

```python
from .services import DataService, clear_cache
```

new:

```python
from .presentation import team_block_colors, team_nickname
from .services import DataService, clear_cache
```

Edit 2 -- old:

```python
templates.env.filters["format_datetime"] = format_datetime
templates.env.filters["format_currency"] = format_currency
```

new:

```python
templates.env.filters["format_datetime"] = format_datetime
templates.env.filters["format_currency"] = format_currency
# Broadcast presentation helpers (api/presentation.py). Globals rather than filters because a
# page that knows only a game_id -- a /bets slip -- needs a team's colours from its abbreviation.
templates.env.globals["team_colors"] = team_block_colors
templates.env.globals["team_nickname"] = team_nickname
```

- [ ] **Step 5: Run the tests to verify they pass**

```bash
uv run pytest tests/unit/test_presentation.py -q
uv run pytest tests/api/test_import_guard.py -q
uv run ruff check api/presentation.py api/dependencies.py tests/unit/test_presentation.py
uv run ruff format api/presentation.py api/dependencies.py tests/unit/test_presentation.py
uv run pyright api/presentation.py api/dependencies.py tests/unit/test_presentation.py
```

Expected:
- All PASS. The import guard still passes because `api/presentation.py` imports only `utils.team_data`, `utils.date_utils` (already imported by `api/routes/health.py`) and the standard library.
- Ruff is clean. If ruff flags `PLR0124` as an unknown or unused `noqa` code in this repo's configuration, delete the `# noqa: PLR0124`.
- pyright reports 0 errors.

- [ ] **Step 6: Commit**

```bash
git add api/presentation.py api/dependencies.py tests/unit/test_presentation.py
git commit -m "$(cat <<'EOF'
feat(redesign): presentation helper for team colours, nicknames and TV windows

Pure helpers for the Broadcast templates: a readable team block colour per team
(near-black primaries fall back to the secondary, unknown teams get a neutral
block), team nicknames, kickoff labels in ET, TV-window grouping that handles
international, Saturday, holiday and Monday-doubleheader slates and a missing
date, and decorate_game. Registered as the team_colors / team_nickname globals.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Dark shell -- `base.html`, the five-item nav, `custom.css`, shared macros

**Files:**
- Modify: `web/templates/base.html` (full replacement)
- Modify: `web/static/css/custom.css` (full replacement)
- Create: `web/templates/components/_broadcast.html`
- Modify: `tests/api/test_pages.py` (two nav assertions; two new nav tests)
- Modify: `tests/api/test_bets_page.py:429-433` (nav test name and docstring)
- Regenerate: `web/static/css/tailwind-compiled.css`
- Test: `tests/unit/test_broadcast_macros.py` (new)

**Interfaces:**
- Consumes: tokens and component classes (Task 1).
- Produces:
  - The five nav items `("/", "This Week")`, `("/bets", "Bets")`, `("/season", "Season")`, `("/track-record", "Track Record")`, `("/how-it-works", "How It Works")`, each emitted twice: once in the desktop row, once in `#mobile-menu`. Each label sits in a `<span class="unskew">`. The active one's class list starts `nav-link skew-control` and ends `skew-control-active` (`class="nav-link skew-control skew-control-active"` in the desktop row, `class="nav-link skew-control justify-start skew-control-active"` in `#mobile-menu`), followed by `aria-current="page"`.
  - `<main id="main">`.
  - The macros in contract section 3, in `components/_broadcast.html`.
  - The `.game-card` hover rule in `custom.css`.

`/track-record` and `/how-it-works` return 404 until Task 15 adds them. The nav links to them from this task on; that is expected mid-branch.

- [ ] **Step 1: Write the failing tests**

Create `tests/unit/test_broadcast_macros.py`:

```python
"""The shared Broadcast macros render in a BARE Jinja environment (redesign Task 3).

The game card must render with no app globals or filters (tests/unit/
test_current_week_rows_reach_the_site.py renders it through a plain FileSystemLoader), and it
imports these macros -- so they are checked the same way, with autoescape on as the app has it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = Path(__file__).resolve().parents[2] / "web" / "templates"


@pytest.fixture(scope="module")
def bc():
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape())
    return env.get_template("components/_broadcast.html").module


def test_section_head_renders_a_yellow_tag_heading(bc) -> None:
    html = str(bc.section_head("Live bets", meta="3 bets", anchor="live"))
    assert '<div class="section-head mb-3" id="live">' in html
    assert '<h2 class="tag"><span class="unskew">Live bets</span></h2>' in html
    assert '<span class="flex-1 h-px bg-line" aria-hidden="true"></span>' in html
    assert '<span class="text-xs text-muted whitespace-nowrap">3 bets</span>' in html
    # The rule follows the heading, so anything placed after the <h2> sits before the first </div>.
    assert html.index("</h2>") < html.index("flex-1 h-px bg-line") < html.index("</div>")


def test_section_head_ghost_and_no_meta(bc) -> None:
    html = str(bc.section_head("Backtest replay", ghost=True))
    assert '<h2 class="tag-ghost"><span class="unskew">Backtest replay</span></h2>' in html
    assert "text-xs text-muted" not in html
    assert " id=" not in html


def test_team_block_carries_its_colours_and_an_upright_label(bc) -> None:
    html = str(bc.team_block("KC", "#E31837", "#FFFFFF"))
    assert (
        '<span class="team-block" style="--team-bg: #E31837; --team-fg: #FFFFFF;">'
        '<span class="unskew">KC</span></span>'
    ) in html
    assert "team-block-lg" in str(bc.team_block("KC", large=True))


def test_score_row_marks_favourite_and_underdog(bc) -> None:
    fav = str(bc.score_row("KC", "Chiefs", "58.8%", "#E31837", "#FFFFFF", fav=True))
    dog = str(bc.score_row("MIA", "Dolphins", "41.2%", dog=True))
    assert '<div class="score-row is-fav">' in fav
    assert '<span class="score-name">Chiefs</span>' in fav
    assert '<span class="score-value">58.8%</span>' in fav
    assert '<div class="score-row is-dog">' in dog
    assert "--team-bg: #1D2436;" in dog


def test_edge_chip_solid_and_soft(bc) -> None:
    assert (
        str(bc.edge_chip("+2.1"))
        == '<span class="edge-chip"><span class="unskew">+2.1</span></span>'
    )
    assert (
        str(bc.edge_chip("3.1", soft=True))
        == '<span class="edge-chip edge-chip-soft"><span class="unskew">3.1</span></span>'
    )


def test_stat_tile_with_accent_sub_and_value_class(bc) -> None:
    html = str(
        bc.stat_tile(
            "Hit rate",
            "58.0%",
            value_class="text-green-400",
            sub="51 of 88",
            accent="var(--color-target-wp)",
        )
    )
    assert '<div class="stat-tile" style="--tile-accent: var(--color-target-wp);">' in html
    assert '<p class="label">Hit rate</p>' in html
    assert '<p class="stat-tile-value text-green-400">58.0%</p>' in html
    assert '<p class="stat-tile-sub">51 of 88</p>' in html


def test_stat_tile_plain(bc) -> None:
    html = str(bc.stat_tile("Bets graded", 90))
    assert '<div class="stat-tile">' in html
    assert '<p class="stat-tile-value">90</p>' in html
    assert "stat-tile-sub" not in html


def test_macro_arguments_are_escaped(bc) -> None:
    assert "&lt;b&gt;" in str(bc.section_head("<b>"))
```

Add these two tests to `tests/api/test_pages.py`, directly after `test_this_week_page_has_game_cards`:

```python
_NAV_ITEMS = ["/", "/bets", "/season", "/track-record", "/how-it-works"]


def test_the_nav_carries_the_five_broadcast_items_in_order(test_client: TestClient):
    """Spec section 6: five items, in this order, in the desktop row AND the mobile panel."""
    html = test_client.get("/").text
    nav = html[html.index("<nav") : html.index("</nav>")]
    hrefs = re.findall(r'<a href="([^"]+)" class="nav-link', nav)
    assert hrefs == _NAV_ITEMS + _NAV_ITEMS
    assert 'aria-controls="mobile-menu"' in nav
    assert '<main id="main"' in html
    assert 'href="#main" class="skip-link"' in html


def test_the_nav_marks_only_the_current_page_active(test_client: TestClient):
    html = test_client.get("/season").text
    nav = html[html.index("<nav") : html.index("</nav>")]
    # The mobile link carries `justify-start` between the base and the active class, so the active
    # class is matched anywhere in the list rather than at a fixed position.
    active = re.findall(
        r'<a href="([^"]+)" class="nav-link skew-control[^"]*\bskew-control-active\b"', nav
    )
    assert active == ["/season", "/season"]
    assert nav.count('aria-current="page"') == 2
    # Skewed links keep their label upright in a .unskew child (spec 10).
    assert '<span class="unskew">Season</span></a>' in nav
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
uv run pytest tests/unit/test_broadcast_macros.py -q
uv run pytest "tests/api/test_pages.py::test_the_nav_carries_the_five_broadcast_items_in_order" "tests/api/test_pages.py::test_the_nav_marks_only_the_current_page_active" -q
```

Expected:
- `test_broadcast_macros.py` FAILS: `TemplateNotFound: components/_broadcast.html`.
- The two nav tests FAIL: today's nav has seven items and no `nav-link` anchors in that form.

- [ ] **Step 3: Create `web/templates/components/_broadcast.html`**

```jinja
{# Shared Broadcast macros (redesign spec section 5). Import with
     {% import "components/_broadcast.html" as bc %}

   Plain Jinja only -- no app globals and no custom filters -- because the game card imports this
   file and must render in a bare jinja2.Environment (tests/unit/
   test_current_week_rows_reach_the_site.py). Colours arrive as arguments; the caller looks them
   up (decorate_game for a game dict, the team_colors global elsewhere).

   Every skewed box (.tag, .tag-ghost, .edge-chip, .team-block) transforms the element, so its
   text sits in an .unskew span that counter-skews it upright.

   IMPORT INSIDE THE BLOCK. jinja2-fragments renders only the named block for an HX-Request, and a
   child template's top-level {% import %} does not run in a block render. A page rendered with
   block_name= imports this file inside that block (or inside the included component). #}

{% macro section_head(label, meta=none, ghost=false, anchor=none) -%}
<div class="section-head mb-3"{% if anchor %} id="{{ anchor }}"{% endif %}>
  <h2 class="{{ 'tag-ghost' if ghost else 'tag' }}"><span class="unskew">{{ label }}</span></h2>
  <span class="flex-1 h-px bg-line" aria-hidden="true"></span>
  {%- if meta %}
  <span class="text-xs text-muted whitespace-nowrap">{{ meta }}</span>
  {%- endif %}
</div>
{%- endmacro %}

{% macro team_block(abbr, bg="#1D2436", fg="#FFFFFF", large=false) -%}
<span class="team-block{% if large %} team-block-lg{% endif %}" style="--team-bg: {{ bg }}; --team-fg: {{ fg }};"><span class="unskew">{{ abbr }}</span></span>
{%- endmacro %}

{% macro score_row(abbr, name, value, bg="#1D2436", fg="#FFFFFF", fav=false, dog=false) -%}
<div class="score-row{% if fav %} is-fav{% endif %}{% if dog %} is-dog{% endif %}">
  {{ team_block(abbr, bg, fg) }}
  <span class="score-name">{{ name }}</span>
  <span class="score-value">{{ value }}</span>
</div>
{%- endmacro %}

{% macro edge_chip(text, soft=false) -%}
<span class="edge-chip{% if soft %} edge-chip-soft{% endif %}"><span class="unskew">{{ text }}</span></span>
{%- endmacro %}

{% macro stat_tile(label, value, value_class="", sub=none, accent=none) -%}
<div class="stat-tile"{% if accent %} style="--tile-accent: {{ accent }};"{% endif %}>
  <p class="label">{{ label }}</p>
  <p class="stat-tile-value{% if value_class %} {{ value_class }}{% endif %}">{{ value }}</p>
  {%- if sub %}
  <p class="stat-tile-sub">{{ sub }}</p>
  {%- endif %}
</div>
{%- endmacro %}
```

- [ ] **Step 4: Replace `web/templates/base.html`**

```jinja
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <meta name="color-scheme" content="dark">
  <meta name="theme-color" content="#0B0F17">
  <title>{% block title %}NFL Predictions{% endblock %} | NFL Predict</title>
  {# Inline SVG favicon -- the wordmark's yellow slash on ink. A data URI, so there is no file to
     serve and no 404 in the console on every page load. #}
  <link rel="icon" href="data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E%3Crect width='32' height='32' rx='6' fill='%230B0F17'/%3E%3Cpath d='M19 5h5L13 27H8z' fill='%23FFD400'/%3E%3C/svg%3E">
  {# The two faces visible above the fold, preloaded so the header does not flash a fallback. #}
  <link rel="preload" href="{{ url_for('static', path='fonts/barlow-condensed-latin-800-italic.woff2') }}" as="font" type="font/woff2" crossorigin>
  <link rel="preload" href="{{ url_for('static', path='fonts/inter-latin-wght-normal.woff2') }}" as="font" type="font/woff2" crossorigin>
  <link rel="stylesheet" href="{{ url_for('static', path='css/tailwind-compiled.css') }}">
  <link rel="stylesheet" href="{{ url_for('static', path='css/custom.css') }}">
  <script src="https://unpkg.com/htmx.org@2.0.4"></script>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  {% block head_extra %}{% endblock %}
</head>
<body class="min-h-screen flex flex-col bg-ink text-fg font-sans antialiased">
  <a href="#main" class="skip-link">Skip to content</a>
  {# The five nav items (spec section 6). The active-item rule lives here once, for both lists: a
     game-detail page (/games/...) counts as This Week. #}
  {% set _path = current_path if current_path is defined and current_path else "" %}
  {% set _active_href = "/" if _path.startswith("/games") else _path %}
  {% set _nav_items = [
       ("/", "This Week"),
       ("/bets", "Bets"),
       ("/season", "Season"),
       ("/track-record", "Track Record"),
       ("/how-it-works", "How It Works")
     ] %}
  <nav class="border-b border-line" aria-label="Primary">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
      <div class="flex items-center justify-between h-16">
        <a href="/" class="wordmark">NFL<span class="text-accent">/</span>Predict</a>
        <div class="hidden md:flex items-center gap-1">
          {% for href, label in _nav_items %}
          {% set _on = href == _active_href %}
          <a href="{{ href }}" class="nav-link skew-control{% if _on %} skew-control-active{% endif %}"{% if _on %} aria-current="page"{% endif %}><span class="unskew">{{ label }}</span></a>
          {% endfor %}
        </div>
        {# Part E's markup (Task 19 audits it): aria-controls, an aria-expanded the script keeps in
           sync, and an aria-label. #}
        <button type="button"
                class="md:hidden skew-control min-h-[44px] min-w-[44px]"
                aria-label="Open navigation menu"
                aria-controls="mobile-menu"
                aria-expanded="false"
                onclick="var menu = document.getElementById('mobile-menu'); var nowHidden = menu.classList.toggle('hidden'); this.setAttribute('aria-expanded', nowHidden ? 'false' : 'true');">
          <span class="unskew" aria-hidden="true">
            <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg>
          </span>
        </button>
      </div>
    </div>
    <div id="mobile-menu" class="hidden md:hidden border-t border-line">
      <div class="px-4 py-3 flex flex-col items-stretch gap-1">
        {% for href, label in _nav_items %}
        {% set _on = href == _active_href %}
        <a href="{{ href }}" class="nav-link skew-control justify-start{% if _on %} skew-control-active{% endif %}"{% if _on %} aria-current="page"{% endif %}><span class="unskew">{{ label }}</span></a>
        {% endfor %}
      </div>
    </div>
  </nav>

  <main id="main" class="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6">
    {% block content %}{% endblock %}
  </main>

  <footer class="mt-auto border-t border-line">
    <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
      {# DEF-31-16: the guard is on the KEY, not on the dict.
         ``{% if cache_meta %}`` tested whether the DICT was non-empty. An EMPTY cache_meta took
         the else branch and rendered "Unknown", which is why this never fired -- but a
         cache_meta carrying ROWS and no ``last_updated`` took the FIRST branch,
         ``cache_meta.last_updated`` resolved to Jinja Undefined, and
         ``api/dependencies.py::format_datetime`` called ``.strftime`` on it. That is an
         UndefinedError, and because this footer lives in the SHARED base template it 500s
         EVERY page on the site, not just the one being built.

         It was unreachable while ONE writer stamped cache_meta and always wrote its three keys
         together. Plan 31-18 adds a SECOND, INDEPENDENT writer -- the per-week bet-list marker
         -- which is exactly what makes the shape reachable: a partial rebuild, a resumed run or
         a hand-built cache can now carry marker rows without ``last_updated``.

         ``.get`` yields None for a missing key; ``format_datetime`` already renders None as
         "Unknown", so the fallback is unchanged for BOTH absences. The ``or ''`` keeps the data
         attribute empty rather than the string "None", so the client-side localiser skips it. #}
      <p class="text-center text-xs text-muted">
        Data updated:
        <span id="cache-timestamp"
              data-utc="{{ (cache_meta.get('last_updated') if cache_meta else None) or '' }}">
          {{ (cache_meta.get('last_updated') if cache_meta else None)|format_datetime }}
        </span>
        {% if cache_meta and cache_meta.prediction_count %}
          <span class="mx-1">&middot;</span> {{ cache_meta.prediction_count }} predictions loaded
        {% endif %}
        <span class="mx-1">&middot;</span> Research tool, not betting advice
      </p>
    </div>
    <script>
    (function() {
      var el = document.getElementById('cache-timestamp');
      if (el && el.dataset.utc) {
        var d = new Date(el.dataset.utc);
        if (!isNaN(d.getTime())) {
          el.textContent = d.toLocaleString(undefined, {
            year: 'numeric', month: 'short', day: 'numeric',
            hour: 'numeric', minute: '2-digit'
          });
        }
      }
    })();
    </script>
  </footer>
</body>
</html>
```

- [ ] **Step 5: Replace `web/static/css/custom.css`**

```css
/* Non-Tailwind rules for the Broadcast UI. Tokens and component classes live in
   web/static/input.css; this sheet loads AFTER tailwind-compiled.css, so every var(--color-*)
   used here is already defined. */

/* ------------------------------------------------------------------------------------------
   Scrollbars, tuned for the dark background
   ------------------------------------------------------------------------------------------ */
html {
  scrollbar-color: var(--color-panel-2) var(--color-ink);
}

::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}

::-webkit-scrollbar-track {
  background: var(--color-ink);
}

::-webkit-scrollbar-thumb {
  background: var(--color-panel-2);
  border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
  background: #2E3850;
}

/* ------------------------------------------------------------------------------------------
   Loading spinner and the HTMX indicator that shows it while a request is in flight
   ------------------------------------------------------------------------------------------ */
.loading-spinner {
  display: inline-block;
  width: 20px;
  height: 20px;
  border: 2px solid rgb(255 212 0 / 0.2);
  border-top-color: var(--color-accent);
  border-radius: 50%;
  animation: spin 0.6s linear infinite;
}

@keyframes spin {
  to { transform: rotate(360deg); }
}

.htmx-indicator {
  display: none;
}

.htmx-request .htmx-indicator,
.htmx-request.htmx-indicator {
  display: inline-block;
}

/* ------------------------------------------------------------------------------------------
   Game card: lifts on hover and its glow picks up the accent
   ------------------------------------------------------------------------------------------ */
.game-card {
  transition: transform 200ms ease, box-shadow 200ms ease, border-color 200ms ease;
}

.game-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 14px 30px -12px rgb(255 212 0 / 0.35);
}

/* ------------------------------------------------------------------------------------------
   Chart container: the prerendered Plotly divs fill their panel
   ------------------------------------------------------------------------------------------ */
.chart-container {
  min-height: 300px;
  position: relative;
}

.chart-container .js-plotly-plot,
.chart-container .plotly-graph-div {
  width: 100% !important;
}

.chart-container > div {
  max-width: 100%;
}

/* ------------------------------------------------------------------------------------------
   Focus: one visible ring everywhere, in the accent
   ------------------------------------------------------------------------------------------ */
*:focus-visible {
  outline: 2px solid var(--color-accent);
  outline-offset: 2px;
}

/* ------------------------------------------------------------------------------------------
   Reduced motion: no lifts, no growing bars, no spinners spinning
   ------------------------------------------------------------------------------------------ */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }

  .game-card:hover {
    transform: none;
  }
}

/* ------------------------------------------------------------------------------------------
   Higher contrast: hard edges on every surface and control
   ------------------------------------------------------------------------------------------ */
@media (prefers-contrast: more) {
  .panel,
  .game-card,
  .stat-tile,
  .bet-slip {
    outline: 1px solid rgb(255 255 255 / 0.6);
  }
}

/* ------------------------------------------------------------------------------------------
   Print: black on white, no chrome
   ------------------------------------------------------------------------------------------ */
@media print {
  nav,
  footer,
  .skip-link,
  .htmx-indicator {
    display: none !important;
  }

  html,
  body {
    background: #FFFFFF !important;
    color: #000000 !important;
  }

  .panel,
  .stat-tile,
  .bet-slip,
  .game-card {
    background: #FFFFFF !important;
    color: #000000 !important;
    border: 1px solid #999999;
    box-shadow: none;
    break-inside: avoid;
  }

  .text-muted,
  .text-dim {
    color: #333333 !important;
  }
}
```

- [ ] **Step 6: Update the nav assertions that named the retired items**

`tests/api/test_pages.py`, the def line and body of `test_insights_page_has_nav_link`. The test is renamed so its name says what it now checks; Task 15 deletes it under the new name. Old:

```python
def test_insights_page_has_nav_link(test_client: TestClient):
    """D-16: Insights link appears in rendered HTML (desktop + mobile menus)."""
    response = test_client.get("/insights")
    # Exactly two occurrences expected: desktop nav + mobile menu.
    assert response.text.count('href="/insights"') >= 2
```

New:

```python
def test_insights_page_nav_links_to_how_it_works(test_client: TestClient):
    """The insights content moves to How It Works (Task 15); the nav already links there."""
    response = test_client.get("/insights")
    # Desktop nav + mobile menu.
    assert response.text.count('href="/how-it-works"') >= 2
```

`tests/api/test_pages.py`, in `test_betting_page_200`. Old:

```python
    # Nav link present in both desktop nav and mobile menu (D-02).
    assert html.count('href="/betting"') >= 2
```

New:

```python
    # The betting simulation moves to Track Record (Task 15); the nav already links there.
    assert html.count('href="/track-record"') >= 2
```

`tests/api/test_bets_page.py`, lines 429-433. Old (the whole test):

```python
def test_nav_carries_the_seventh_item(bets_client: TestClient) -> None:
    """The seventh nav item reaches /bets from both the desktop and the mobile list."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert body.count('href="/bets"') == 2
    assert ">Bets</a>" in body
```

New:

```python
def test_nav_carries_the_bets_item(bets_client: TestClient) -> None:
    """The Bets nav item reaches /bets from both the desktop row and the mobile panel."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert body.count('href="/bets"') == 2
    # The label sits in the skewed link's .unskew child (spec 10).
    assert '<span class="unskew">Bets</span></a>' in body
```

- [ ] **Step 7: Compile the stylesheet and run the tests**

```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
uv run pytest tests/unit/test_broadcast_macros.py -q
uv run pytest tests/unit/test_broadcast_theme_css.py -q
uv run pytest "tests/api/test_pages.py::test_the_nav_carries_the_five_broadcast_items_in_order" "tests/api/test_pages.py::test_the_nav_marks_only_the_current_page_active" "tests/api/test_pages.py::test_insights_page_nav_links_to_how_it_works" "tests/api/test_pages.py::test_betting_page_200" "tests/api/test_pages.py::test_season_page_200" "tests/api/test_pages.py::test_this_week_page" "tests/api/test_pages.py::test_game_detail_not_found" -q
uv run pytest tests/api/test_bets_page.py -k "nav_carries or not_advice or unknown or data_updated or last_updated or cache_meta or footer" -q
uv run pytest tests/unit/test_page_labels.py -q
uv run pytest tests/api/test_page_labels_routes.py -q
uv run pytest tests/api/test_fragments.py -q
uv run ruff check tests/unit/test_broadcast_macros.py tests/api/test_pages.py tests/api/test_bets_page.py
uv run ruff format tests/unit/test_broadcast_macros.py tests/api/test_pages.py tests/api/test_bets_page.py
uv run pyright tests/unit/test_broadcast_macros.py tests/api/test_pages.py tests/api/test_bets_page.py
```

Expected:
- All PASS.
- The `-k` expression on `test_bets_page.py` also selects the three DEF-31-16 footer guards (`test_a_cache_meta_with_rows_but_no_last_updated_does_not_500_the_bets_page`, `test_the_same_cache_meta_does_not_500_any_other_page_either`, `test_the_footer_still_renders_the_timestamp_when_it_is_present`): this task rewrites the shared footer, and a footer that 500s takes every page down with it.
- The page-label suites still pass: every page renders through the new `base.html`, `_StubRequest.url_for` serves the font preload URLs, and the footer keeps `Data updated:` and `Unknown`.
- `test_fragments.py` still finds no `<nav` in any fragment.
- Ruff is clean (`ruff format` reformats only the new test file, if anything). pyright reports no error on a line this task wrote; an error it reports elsewhere in an existing test file predates the branch -- note it in the task report and leave it.

- [ ] **Step 8: Visual check against the mockups**

```bash
powershell -NoProfile -Command "Get-Date -Format 'HH:mm'"
```

Expected: a time outside 16:30-17:45. Then start the preview server in the background (Bash `run_in_background: true`):

```bash
cd /c/Users/jackc/Code/nfl-predict-redesign && uv run uvicorn api.main:app --host 127.0.0.1 --port 8001
```

1. With the Playwright MCP, open `http://127.0.0.1:8001/` and `http://127.0.0.1:8001/bets`.
2. Screenshot each at 1440x900 and at 390x844. On mobile, open the menu.
3. Compare against the nav, wordmark and footer in `docs/superpowers/specs/2026-10-01-broadcast-ui-mockups/02-this-week-layouts.html`. Look for:
   - a dark gradient page with the italic `NFL/PREDICT` wordmark and its yellow slash;
   - five nav items, with This Week as a yellow skewed tab on `/`;
   - a one-line muted footer;
   - the hamburger opening a vertical list on mobile.

The page bodies are still the old light layouts at this point; that is expected. Stop the server:

```bash
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8001 -State Listen -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }"
```

- [ ] **Step 9: Commit**

```bash
git add web/templates/base.html web/templates/components/_broadcast.html web/static/css/custom.css web/static/css/tailwind-compiled.css tests/unit/test_broadcast_macros.py tests/api/test_pages.py tests/api/test_bets_page.py
git commit -m "$(cat <<'EOF'
feat(redesign): dark Broadcast shell with the five-item nav and shared macros

base.html becomes the dark Broadcast shell: italic NFL/PREDICT wordmark, the
five-item nav (This Week, Bets, Season, Track Record, How It Works) with a
skewed yellow active tab in both the desktop row and the mobile panel, a skip
link, preloaded self-hosted fonts and a one-line footer that keeps the
DEF-31-16 guard and the data-utc localiser. Adds the shared section_head /
team_block / score_row / edge_chip / stat_tile macros and rewrites custom.css
for the dark theme. Track Record and How It Works routes land in Task 15.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Honesty notes and monochrome badges

**Files:**
- Modify (full replacements): `web/templates/components/_not_advice_banner.html`, `_old_rule_label.html`, `_provenance_badge.html`, `_ev_band_badge.html`, `_confidence_badge.html`, `_status_badge.html`
- Modify: `tests/api/test_bets_page.py` (Part C 5a, 5b, 5d, 5e: `test_ev_band_badge_is_monochrome`; the `_VALIDATION_CLASSES` table and the assertion that reads it; `test_zero_suppressed_rows_renders_the_line_and_no_disclosure`; `test_the_caption_is_present_whether_or_not_the_disclosure_is_expanded`)
- Modify: `tests/api/test_pages.py` (`test_this_week_page_confidence_badges`, `test_the_landing_page_still_renders_all_three_edge_band_labels`, `test_the_landing_page_renders_the_band_it_was_served_and_never_rederives_one`)
- Regenerate: `web/static/css/tailwind-compiled.css`
- Test: `tests/unit/test_broadcast_components.py` (new)

**Interfaces:**
- Consumes: `honesty-note*`, `band*`, `evidence-chip*` classes (Task 1).
- Produces:
  - The not-advice banner and the old-rule label: `<aside role="note">` -> `<details class="honesty-note">` -> `<summary>` (key + one-liner + "Why?") -> `<p class="honesty-note-body">` with the full original text. `data-old-rule-label` stays on the old-rule `<aside>`, once per label.
  - Confidence label: `<span class="band band-{b}" data-confidence-band="{b}">{value}</span>`.
  - EV-band label: `<span class="band band-{b}" title="EV band {b}: ...">{b}<span class="sr-only"> -- ...</span></span>`.
  - Provenance chip: `<span class="evidence-chip[ evidence-chip-strong]" data-provenance=".." data-validation-type=".." title="..">{label}<span class="sr-only"> -- ..</span></span>`. `clean_holdout` is strong; `contaminated` and `forward_realized` are plain (Part C 5b).

- [ ] **Step 1: Write the failing component tests**

Create `tests/unit/test_broadcast_components.py`:

```python
"""Rendering guards for the restyled shared components (redesign Tasks 4-5).

Each partial is rendered through the app's own Jinja environment and checked for what the
redesign promises: honesty text kept word for word inside a native, collapsed <details>; badges
that carry no outcome hue; and controls that keep every piece of their htmx wiring.

ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from api.dependencies import templates

NOT_ADVICE_FULL_TEXT = (
    "This is a personal research tool. No wager is placed and no currency amount is shown -- "
    "stakes are expressed in units, where 1 unit = 1% of a notional bankroll. The deployed "
    "models have not demonstrated a positive edge against the closing market: the "
    "win-probability model's pooled closing-line value is negative, and the spread and totals "
    "models were retained after failing their most recent re-fit gate. A bet appearing on this "
    "list means it cleared a pre-registered expected-value floor, not that it is expected to win."
)
OLD_RULE_FULL_TEXT = (
    "Built under the old rule on inputs later found defective; not evidence. These figures come "
    "from before the September 2026 fix to what the models were fed, and they stay here for the "
    "record only. Only the 2026 season, recorded live with each game locked at 6 PM Eastern the "
    "day before kickoff, counts as evidence."
)
OUTCOME_HUES = ("green", "red", "amber", "yellow")


def render(name: str, **context: Any) -> str:
    return templates.env.get_template(f"components/{name}").render(**context)


def flat(html: str) -> str:
    return " ".join(html.split())


def class_attributes(html: str) -> list[str]:
    return re.findall(r'class="([^"]*)"', html)


def summary_of(html: str) -> str:
    return html[html.index("<summary>") : html.index("</summary>")]


class TestNotAdviceBanner:
    def test_it_is_a_collapsed_native_disclosure_inside_a_note(self) -> None:
        html = render("_not_advice_banner.html")
        assert '<aside role="note"' in html
        assert '<details class="honesty-note">' in html
        assert "<details open" not in html
        assert "<button" not in html and "<script" not in html

    def test_the_summary_names_it_and_offers_why(self) -> None:
        summary = summary_of(render("_not_advice_banner.html"))
        assert "Not wagering advice" in summary
        assert "Why?" in summary

    def test_the_full_text_is_kept_word_for_word(self) -> None:
        assert NOT_ADVICE_FULL_TEXT in flat(render("_not_advice_banner.html"))


class TestOldRuleLabel:
    def test_an_unwired_block_labels_once_as_a_collapsed_disclosure(self) -> None:
        html = render("_old_rule_label.html")
        assert html.count("data-old-rule-label") == 1
        assert '<details class="honesty-note">' in html
        assert "<details open" not in html

    def test_the_summary_states_the_date_and_not_evidence(self) -> None:
        summary = summary_of(render("_old_rule_label.html"))
        assert "Old-rule numbers" in summary
        assert "2026-09-15" in summary
        assert "not evidence" in summary

    def test_the_full_sentence_is_kept_word_for_word(self) -> None:
        assert OLD_RULE_FULL_TEXT in flat(render("_old_rule_label.html"))


class TestMonochromeBadges:
    @pytest.mark.parametrize("band", ["high", "medium", "low"])
    def test_ev_band(self, band: str) -> None:
        html = render("_ev_band_badge.html", band=band)
        assert f'<span class="band band-{band}" title="EV band {band}:' in html
        for classes in class_attributes(html):
            for hue in OUTCOME_HUES:
                assert hue not in classes

    def test_an_unknown_ev_band_renders_as_low(self) -> None:
        assert 'class="band band-low"' in render("_ev_band_badge.html", band="weird")

    @pytest.mark.parametrize("level", ["high", "medium", "low"])
    def test_confidence(self, level: str) -> None:
        html = render("_confidence_badge.html", level=level, value=level.title())
        assert (
            f'<span class="band band-{level}" data-confidence-band="{level}">'
            f"{level.title()}</span>"
        ) in html

    def test_an_unknown_confidence_renders_as_low(self) -> None:
        html = render("_confidence_badge.html", level="weird", value="Weird")
        assert 'data-confidence-band="low"' in html

    @pytest.mark.parametrize(
        ("provenance", "validation_type", "classes", "label"),
        [
            ("backtest_replay", "contaminated", "evidence-chip", "Contaminated split"),
            (
                "backtest_replay",
                "clean_holdout",
                "evidence-chip evidence-chip-strong",
                "Old rule -- 2025, not evidence",
            ),
            (
                "forward",
                "forward_realized",
                "evidence-chip",
                "Live forward record",
            ),
        ],
    )
    def test_provenance(
        self, provenance: str, validation_type: str, classes: str, label: str
    ) -> None:
        html = render(
            "_provenance_badge.html", provenance=provenance, validation_type=validation_type
        )
        assert f'<span class="{classes}" data-provenance="{provenance}"' in html
        assert f'data-validation-type="{validation_type}"' in html
        assert f">{label}<span class=\"sr-only\">" in html
        assert html.strip().endswith("</span></span>")

    def test_an_impossible_provenance_pair_renders_its_raw_code(self) -> None:
        html = render(
            "_provenance_badge.html", provenance="forward", validation_type="contaminated"
        )
        assert ">contaminated<span" in html

    @pytest.mark.parametrize(
        ("status", "text"),
        [
            ("completed", "Completed"),
            ("scheduled", "Scheduled"),
            ("in_progress", "In Progress"),
            ("postponed", "Postponed"),
            ("cancelled", "Cancelled"),
            ("delayed", "Delayed"),
        ],
    )
    def test_status_badges_never_use_an_outcome_hue(self, status: str, text: str) -> None:
        html = render("_status_badge.html", status=status)
        assert f">{text}</span>" in html
        for classes in class_attributes(html):
            for hue in ("green", "red", "amber", "blue"):
                assert hue not in classes
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/unit/test_broadcast_components.py -q`

Expected:
- **Fail:** the honesty-note tests, because there is no `<details>` yet.
- **Fail:** the band, confidence, provenance and status tests, because those partials still use the old classes.

- [ ] **Step 3: Replace `web/templates/components/_not_advice_banner.html`**

```jinja
{# Not-advice banner. No parameters -- the copy is fixed authored prose.

   PERSISTENT and NON-DISMISSABLE by construction: no close button, no `hidden` class, no JS, and
   no conditional wrapper. It is server-rendered inside <main>, ABOVE the section _error_state.html
   replaces, so it is present in the first byte of every response and survives every empty state,
   the stale-cache hard-block, and every HTMX swap (UI-SPEC E1 loading/error).

   CONDENSED, NOT HIDDEN (Broadcast redesign, owner decision O4, 2026-10-01). A native <details>:
   the <summary> is always on screen and cannot be dismissed -- it names the banner and gives the
   one-line version -- and "Why?" opens the full original paragraph, which is in the HTML whether
   or not the reader opens it. Expanding is the only interaction; nothing takes it off the page.

   Grey, not red and not amber: a red banner reads as an error and gets dismissed mentally after
   two visits; this statement is permanent, not an alert. #}
<aside role="note" class="mb-4">
  <details class="honesty-note">
    <summary><span class="honesty-note-key">Not wagering advice</span><span class="honesty-note-line">Research tool: no wager is placed, stakes are units, and no edge over the closing market has been demonstrated.</span><span class="honesty-note-why">Why?</span></summary>
    <p class="honesty-note-body">This is a personal research tool. No wager is placed and no currency amount is shown -- stakes are expressed in units, where 1 unit = 1% of a notional bankroll. The deployed models have not demonstrated a positive edge against the closing market: the win-probability model's pooled closing-line value is negative, and the spread and totals models were retained after failing their most recent re-fit gate. A bet appearing on this list means it cleared a pre-registered expected-value floor, not that it is expected to win.</p>
  </details>
</aside>
```

- [ ] **Step 4: Replace `web/templates/components/_old_rule_label.html`**

The comment must not contain any word from `backtest.ev_chain_constants.READOUT_FORBIDDEN_WORDS`. `tests/unit/test_page_labels.py` scans the whole file, comments included. The file must be ASCII and use only plain classes.

```jinja
{# Dated old-rule label (Phase 33.2, R16 / D33.2-07). Pass: scope -- the block's season scope
   from DataService.old_rule_scope, which the ROUTE builds from the block's own data.

   THE DECISION LIVES HERE, not in the calling page. A page includes this partial in every block
   that can show numbers and hands it that block's scope; the partial alone decides whether the
   label renders. So no page can include it and forget the condition, and no page needs a
   hardcoded per-page switch that would go stale the moment it gains a block.

   It renders when the scope says the block shows a season at or before 2025, AND WHEN THE SCOPE IS
   MISSING. That second branch is deliberate and it is the whole safety property: a block someone
   adds later and forgets to wire hands this partial nothing, and the result is a visible label --
   perhaps a wrong one, and therefore a noticed one. The opposite default would make the same
   mistake invisible, which is exactly the failure R16 exists to prevent. A block that renders no
   numbers passes a KNOWN-EMPTY scope, which does not label.

   CONDENSED, NOT HIDDEN (Broadcast redesign, owner decision O4, 2026-10-01). The label is a native
   <details>: its <summary> -- always on screen, never dismissable -- calls these old-rule numbers,
   says they are not evidence and carries the 2026-09-15 date; the full sentence is the body, in
   the HTML whether or not the reader opens it. The data-old-rule-label marker stays on the outer
   <aside>, once per label, because that is what the label tests count.

   MONOCHROME, for the reason the evidence-class badge and _ev_band_badge.html are: on this site
   hue carries outcome meaning (green a won bet, red a lost one), and a coloured label would borrow
   that meaning for a statement about evidence. Every class here must already exist in
   tailwind-compiled.css as a plain, non-variant class; tests/unit/test_page_labels.py checks that.

   The sentence "Built under the old rule on inputs later found defective; not evidence." is the
   same sentence the repo-root readouts carry in their dated addendum. Keep the two identical. #}
{% set _old_rule_scope_unknown = scope is not defined or scope is none or scope.contains_old_rule_results is not defined %}
{% if _old_rule_scope_unknown or scope.contains_old_rule_results %}
<aside role="note" data-old-rule-label class="mb-4">
  <details class="honesty-note">
    <summary><span class="honesty-note-key">Old-rule numbers</span><span class="honesty-note-line">Labelled 2026-09-15: built on inputs later found defective -- not evidence.</span><span class="honesty-note-why">Why?</span></summary>
    <p class="honesty-note-body">Built under the old rule on inputs later found defective; not evidence. These figures come from before the September 2026 fix to what the models were fed, and they stay here for the record only. Only the 2026 season, recorded live with each game locked at 6 PM Eastern the day before kickoff, counts as evidence.</p>
  </details>
</aside>
{% endif %}
```

- [ ] **Step 5: Replace `web/templates/components/_provenance_badge.html`**

```jinja
{# Provenance badge pill. Pass: provenance, validation_type.

   THE SINGLE SOURCE of the honesty-label vocabulary (SPEC R8, D31-22). Both the tracker block
   headers and every displayed row draw their labels from HERE. A second label source is the
   two-lists failure this repository has already paid for once, so nothing else may spell these
   strings -- tests/api/test_bets_page.py asserts no second source exists.

   KEYED ON BOTH COLUMNS, not on validation_type alone. The two D31-22 axes are orthogonal and both
   are required on every row, so the LABEL lookup is nested: a pair the vocabulary does not contain
   -- a forward row carrying a replay validation type, say -- resolves to no label and renders its
   RAW CODE. That is the point of keying on the pair: an impossible combination is shown as the
   defect it is rather than confidently mislabelled by whichever column was consulted alone.

   MONOCHROME, for the reason _ev_band_badge.html is monochrome: hue on /bets carries meaning only
   inside the tracker's realized outcomes, where green means a bet WON and red means it LOST. A
   coloured honesty label would borrow that meaning for a statement about evidence class. Fill
   WEIGHT carries the distinction instead: the clean-holdout class keeps the strong chip it had
   before the Broadcast redesign (it was the heaviest grey), and the other two are the plain chip.

   NO whitespace-nowrap: the longest label is 30 characters and the badge sits on a block header
   where it must WRAP rather than clip (UI-SPEC E10 overflow). Inside a table cell the cell's own
   whitespace-nowrap governs, and the row already scrolls horizontally rather than clipping.

   There is NO null state to design: both columns are required on every row, so a row missing
   either is a data defect caught upstream, not a render branch. #}
{% set _PROVENANCE_LABELS = {
     "backtest_replay": "Backtest replay",
     "forward": "Forward record"
   } %}
{% set _VALIDATION_LABELS = {
     "backtest_replay": {
       "contaminated": "Contaminated split",
       "clean_holdout": "Old rule -- 2025, not evidence"
     },
     "forward": {
       "forward_realized": "Live forward record"
     }
   } %}
{% set _VALIDATION_CLASSES = {
     "contaminated": "evidence-chip",
     "clean_holdout": "evidence-chip evidence-chip-strong",
     "forward_realized": "evidence-chip"
   } %}
{% set _VALIDATION_DESCRIPTIONS = {
     "contaminated": "measured on seasons already used to build and tune the models, so the figure is optimistic by construction",
     "clean_holdout": "measured on the 2025 season under the old rule, on inputs later found defective, so it is not evidence",
     "forward_realized": "recommended before kickoff and graded afterwards"
   } %}
{% set _provenance_label = _PROVENANCE_LABELS.get(provenance, provenance) %}
{% set _validation_label = _VALIDATION_LABELS.get(provenance, {}).get(validation_type, validation_type) %}
{% set _validation_classes = _VALIDATION_CLASSES.get(validation_type, "evidence-chip") %}
{% set _validation_description = _VALIDATION_DESCRIPTIONS.get(validation_type, "an evidence class with no description in the displayed vocabulary; the raw code is shown") %}
<span class="{{ _validation_classes }}" data-provenance="{{ provenance }}" data-validation-type="{{ validation_type }}" title="{{ _provenance_label }}: {{ _validation_description }}">{{ _validation_label }}<span class="sr-only"> -- {{ _provenance_label }}, {{ _validation_description }}</span></span>
```

- [ ] **Step 6: Replace `web/templates/components/_ev_band_badge.html`**

```jinja
{# EV band label. Pass: band ("high" / "medium" / "low"); anything else renders as "low", exactly
   as the pre-redesign partial did.

   MONOCHROME on purpose (UI-SPEC Deviation 1). An EV band is a pre-registered arithmetic band on
   expected value (D31-24) -- not a probability of winning and not a quality claim -- and the
   project has not measured a positive edge to justify a green signal. On /bets green and red
   mean a bet won or lost, inside the tracker only.

   Fill WEIGHT encodes magnitude; hue encodes nothing. It shares the band-* classes with the
   confidence label but never its data-confidence-band attribute -- it carries this title
   instead, which is how a test finds it. #}
{% set _band = band if band in ("high", "medium") else "low" %}
{% set _EV_BAND_TITLES = {
     "high": "EV band high: expected value at or above 5.0% per unit staked",
     "medium": "EV band medium: expected value from 3.0% up to 5.0% per unit staked",
     "low": "EV band low: expected value from the pre-registered floor up to 3.0% per unit staked"
   } %}
<span class="band band-{{ _band }}" title="{{ _EV_BAND_TITLES[_band] }}">{{ _band }}<span class="sr-only"> -- {{ _EV_BAND_TITLES[_band] }}</span></span>
```

- [ ] **Step 7: Replace `web/templates/components/_confidence_badge.html`**

```jinja
{# Confidence (edge band) label. Pass: level ("high"/"medium"/"low"), value (display text).

   MONOCHROME since the Broadcast redesign (2026-10-01). It used to map high/medium/low to
   green/amber/red, but on this site green and red mean a realised result -- a won or lost bet, a
   correct or incorrect pick -- and a pre-game band is neither. Fill weight now encodes the band,
   exactly as _ev_band_badge.html does. data-confidence-band names the band, so a test can tell
   this label from the EV band label that shares the band-* classes. Anything other than "high" or
   "medium" renders as "low", as before. #}
{% set _band = level if level in ("high", "medium") else "low" %}
<span class="band band-{{ _band }}" data-confidence-band="{{ _band }}">{{ value }}</span>
```

- [ ] **Step 8: Replace `web/templates/components/_status_badge.html`**

```jinja
{# Game status badge. Pass: status ("scheduled"/"completed"/"postponed"/etc.)

   Neutral on purpose (Broadcast redesign): a game's status is not a result, so it never borrows
   the green/red outcome hues. Only a game in progress gets the accent, because it is live. #}
{% set _status_base = "inline-flex items-center px-2 py-0.5 font-display text-xs font-bold uppercase tracking-wider" %}
{% if status == "completed" %}
<span class="{{ _status_base }} bg-panel-2 text-fg">Completed</span>
{% elif status == "scheduled" %}
<span class="{{ _status_base }} border border-line text-muted">Scheduled</span>
{% elif status == "in_progress" %}
<span class="{{ _status_base }} bg-accent text-ink">In Progress</span>
{% elif status == "postponed" %}
<span class="{{ _status_base }} border border-line text-muted">Postponed</span>
{% elif status == "cancelled" %}
<span class="{{ _status_base }} border border-line text-dim">Cancelled</span>
{% else %}
<span class="{{ _status_base }} border border-line text-muted">{{ status|title }}</span>
{% endif %}
```

- [ ] **Step 9: Update the tests that pinned the old badge classes**

Part C's contract notes 5a, 5b, 5d and 5e land here, because this task causes the breaks. Apply them verbatim.

**5a.** `tests/api/test_bets_page.py::test_ev_band_badge_is_monochrome`. Old:

```python
    assert "EV band" in body
    assert "bg-gray-200 text-gray-900 border border-gray-300" in body
```

New:

```python
    assert "EV band" in body
    assert re.search(
        r'<span class="[^"]*\bband-high\b[^"]*"[^>]*title="EV band high', body
    ), "the high EV band did not render with the monochrome band-high style"
```

**5b.** `tests/api/test_bets_page.py`, the `_VALIDATION_CLASSES` table. Old:

```python
_VALIDATION_CLASSES: dict[str, str] = {
    "contaminated": "bg-gray-100 text-gray-700 border border-gray-300",
    "clean_holdout": "bg-gray-200 text-gray-900 border border-gray-300",
    "forward_realized": "bg-white text-gray-700 border border-gray-300",
}
```

New:

```python
_VALIDATION_CLASSES: dict[str, str] = {
    "contaminated": "evidence-chip",
    "clean_holdout": "evidence-chip evidence-chip-strong",
    "forward_realized": "evidence-chip",
}
```

The assertion that reads the table, in `test_every_validation_type_renders_its_declared_label_and_classes`, is tightened to the exact class attribute: a bare `"evidence-chip" in section` would also match inside the strong chip's `"evidence-chip evidence-chip-strong"`, so it could not fail. Old:

```python
        assert _VALIDATION_CLASSES[validation_type] in section, (
            f"{validation_type} rendered without its declared class set"
        )
```

New:

```python
        # The WHOLE class attribute, anchored on the data-provenance that follows it: a bare
        # "evidence-chip" would also match inside "evidence-chip evidence-chip-strong".
        assert (
            f'class="{_VALIDATION_CLASSES[validation_type]}" data-provenance="{pair[0]}"'
            in section
        ), f"{validation_type} rendered without its declared class set"
```

**5d.** `tests/api/test_bets_page.py::test_zero_suppressed_rows_renders_the_line_and_no_disclosure`. The condensed honesty notes are `<details>` too. Old:

```python
    assert "<details" not in body
```

New:

```python
    # The condensed honesty notes are <details> too; only the suppressed disclosure is absent.
    assert '<details id="suppressed-candidates"' not in body
```

**5e.** `tests/api/test_bets_page.py::test_the_caption_is_present_whether_or_not_the_disclosure_is_expanded`. The first `<summary>` on the page is now the not-advice note's. Old:

```python
    summary = body[body.index("<summary") : body.index("</summary>")]
```

New:

```python
    start = body.index('<details id="suppressed-candidates"')
    summary = body[body.index("<summary", start) : body.index("</summary>", start)]
```

`tests/api/test_pages.py::test_this_week_page_confidence_badges`. Old:

```python
    """D-08: Confidence badges with color-coded levels."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    # Sample data contains high, medium, and low confidence values
    assert "bg-green-100" in html  # high
    assert "bg-amber-100" in html  # medium
    assert "bg-red-100" in html  # low
```

New:

```python
    """D-08: Confidence labels render one monochrome band each (Broadcast redesign)."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    # Sample data contains high, medium, and low confidence values. The labels are grey-scale:
    # green and red now mean a realised result only, and a pre-game band is not one.
    assert 'class="band band-high" data-confidence-band="high"' in html
    assert 'class="band band-medium" data-confidence-band="medium"' in html
    assert 'class="band band-low" data-confidence-band="low"' in html
    # Only amber is checked here: until Task 7 replaces the card, its Correct/Incorrect result
    # badge still (rightly) renders bg-green-100 / bg-red-100 for the fixture's completed games.
    # Task 7 adds the green/red check once that badge is gone.
    assert "bg-amber-100" not in html
```

`tests/api/test_pages.py::test_the_landing_page_still_renders_all_three_edge_band_labels`. Old:

```python
    # The three colour classes the confidence badge maps the three labels onto, one per band.
    for badge_class in ("bg-green-100", "bg-amber-100", "bg-red-100"):
        assert badge_class in html, (
            f"the {badge_class} badge disappeared from /; an edge band label has moved"
        )
```

New:

```python
    # One confidence label per band, located by the band it declares.
    for band in sorted(EDGE_TIER_LABELS):
        assert f'data-confidence-band="{band}"' in html, (
            f"the {band} confidence label disappeared from /; an edge band label has moved"
        )
```

`tests/api/test_pages.py::test_the_landing_page_renders_the_band_it_was_served_and_never_rederives_one`. Old:

```python
    # The badge partial maps each band onto ONE colour class and renders the label title-cased.
    # Comparing the MULTISET of rendered badges against the multiset of served bands is what makes
    # this a re-banding check rather than a spelling check: a page that turned one served "low"
    # into a "high" would leave the vocabulary intact and the counts different.
    #
    # The class triple below is the CONFIDENCE badge's, not the STATUS badge's. Both live on game
    # cards and both use ``bg-green-100``; only the confidence badge carries the matching
    # ``border border-<colour>-200``. Keying on the bare background colour matches the status
    # badge's "Completed" pill and reports it as a mis-banded row -- which is how this assertion
    # first failed, and why the selector is the full triple.
    band_by_class = {"green": "high", "amber": "medium", "red": "low"}
    rendered: list[str] = []
    for colour, band in band_by_class.items():
        pattern = (
            rf"bg-{colour}-100 text-{colour}-\d+ border border-{colour}-200[^>]*>"
            r"([^<]+)</span>"
        )
        for match in re.finditer(pattern, html):
            assert match.group(1).strip().lower() == band, (
                f"a bg-{colour}-100 confidence badge renders {match.group(1)!r}, which is not "
                f"the {band!r} band that colour is reserved for"
            )
            rendered.append(band)
```

New:

```python
    # The badge partial renders each band as one monochrome class plus a data-confidence-band
    # attribute, and the label title-cased. Comparing the MULTISET of rendered badges against the
    # multiset of served bands is what makes this a re-banding check rather than a spelling check:
    # a page that turned one served "low" into a "high" would leave the vocabulary intact and the
    # counts different.
    #
    # The selector keys on data-confidence-band, not on the band-* class alone: the EV band label
    # shares those classes, and keying on a bare class is how this assertion first failed (it
    # matched the status badge when the two shared a colour).
    rendered: list[str] = []
    pattern = (
        r'<span class="band band-(high|medium|low)" data-confidence-band="(high|medium|low)">'
        r"([^<]+)</span>"
    )
    for match in re.finditer(pattern, html):
        css_band, attr_band, label = match.groups()
        assert css_band == attr_band == label.strip().lower(), (
            f"a confidence badge renders {label!r} with class band-{css_band} and "
            f"data-confidence-band={attr_band!r}; the three must name the same band"
        )
        rendered.append(attr_band)
```

- [ ] **Step 10: Compile and run every affected test**

```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
uv run pytest tests/unit/test_broadcast_components.py -q
uv run pytest tests/unit/test_page_labels.py -q
uv run pytest tests/api/test_page_labels_routes.py -q
uv run pytest "tests/api/test_pages.py::test_this_week_page_confidence_badges" "tests/api/test_pages.py::test_the_landing_page_still_renders_all_three_edge_band_labels" "tests/api/test_pages.py::test_the_landing_page_renders_the_band_it_was_served_and_never_rederives_one" "tests/api/test_pages.py::test_the_landing_page_sort_by_band_is_unchanged" -q
uv run pytest tests/api/test_bets_page.py -k "monochrome or validation_type or badge or not_advice or banner or provenance or label or suppressed or caption or disclosure" -q
uv run pytest tests/unit/test_current_week_rows_reach_the_site.py -q
uv run ruff check tests/unit/test_broadcast_components.py tests/api/test_bets_page.py tests/api/test_pages.py
uv run ruff format tests/unit/test_broadcast_components.py tests/api/test_bets_page.py tests/api/test_pages.py
uv run pyright tests/unit/test_broadcast_components.py tests/api/test_bets_page.py tests/api/test_pages.py
```

Expected:
- All PASS. `test_page_labels.py::TestThePartial` confirms three things: the partial is ASCII, it carries the label phrase and no over-claim word, and its six plain classes (`mb-4`, `honesty-note`, `honesty-note-key`, `honesty-note-line`, `honesty-note-why`, `honesty-note-body`) are in the compiled sheet.
- `test_current_week_rows_reach_the_site.py` still renders the card in a bare environment. Its "Low" check passes because a game with no band renders no confidence label.
- Ruff is clean. pyright reports no error on a line this task wrote; an error elsewhere in an existing test file predates the branch -- note it in the task report and leave it.

- [ ] **Step 11: Commit**

```bash
git add web/templates/components/_not_advice_banner.html web/templates/components/_old_rule_label.html web/templates/components/_provenance_badge.html web/templates/components/_ev_band_badge.html web/templates/components/_confidence_badge.html web/templates/components/_status_badge.html web/static/css/tailwind-compiled.css tests/unit/test_broadcast_components.py tests/api/test_bets_page.py tests/api/test_pages.py
git commit -m "$(cat <<'EOF'
feat(redesign): condensed honesty notes and monochrome badges

The not-advice banner and the dated old-rule label become one-line native
<details> notes: the summary is always on screen and names the note, and "Why?"
opens the full original wording, kept word for word in the HTML. The confidence,
EV band and provenance labels and the status badge are grey-scale; green and red
are now reserved for realised results. Confidence labels carry
data-confidence-band so they stay distinguishable from EV band labels.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Controls and states -- selectors, toggles, sort, empty/error, exports, skeleton, week summary

**Files:**
- Modify: `web/templates/components/_week_selector.html` (nine edits listed in Step 4; logic untouched)
- Modify (full replacements): `_season_selector.html`, `_season_tracking_selector.html`, `_betting_scope_toggle.html`, `_sort_controls.html`, `_empty_state.html`, `_error_state.html`, `_export_buttons.html`, `_loading_skeleton.html`, `_week_summary.html`
- Modify: `tests/api/week_selector_snapshot.py:23-25` (`SELECTOR_OPEN`)
- Re-record: `tests/api/snapshots/week_selector_this_week.html`, `tests/api/snapshots/week_selector_prev_next.html`
- Modify: `tests/api/test_pages.py` (two `bg-red-50` assertions)
- Modify: `tests/api/test_bets_page.py` (three `bg-red-50` assertions; Part C 5c)
- Modify: `tests/api/test_export.py`:
  - replace the git-diff byte guard with a reuse guard;
  - drop `import subprocess`.
- Regenerate: `web/static/css/tailwind-compiled.css`
- Test: `tests/unit/test_broadcast_components.py` (extend)

**Interfaces:**
- Consumes: `skew-control`, `skew-control-active`, `label`, `stat-tile` via `bc.stat_tile` (Tasks 1 and 3).
- Produces:
  - Selector root `<div class="week-selector flex flex-wrap items-center gap-2">`.
  - Every control is a `skew-control`, the active toggle a `skew-control-active`.
  - Error state classes `bg-red-950 border border-red-800`, `text-red-200`, `text-red-100`, `text-red-300`.
  - Empty state: a dashed grey box whose heading is an `<h3 class="display ...">` and whose body is a `<p>`.
  - Export links keep `id`s, `data-base-url` and `min-h-[44px]`.
  - The week summary is three `stat-tile`s keyed to the bet-type colours.

- [ ] **Step 1: Extend the component tests (failing)**

Append to `tests/unit/test_broadcast_components.py`:

```python
class TestStates:
    def test_empty_state_is_a_grey_box_with_a_heading_and_a_paragraph(self) -> None:
        html = render(
            "_empty_state.html",
            heading="Nothing graded yet",
            body="Grades appear after the games are played.",
            action_text=None,
            action_url=None,
        )
        assert "border-dashed" in html
        assert '<h3 class="display text-xl text-fg">Nothing graded yet</h3>' in html
        assert "<p " in html and "Grades appear after the games are played." in html
        assert "<a " not in html
        for classes in class_attributes(html):
            assert "red" not in classes

    def test_empty_state_action_is_the_accent_control(self) -> None:
        html = render(
            "_empty_state.html",
            heading="Game not found",
            body="This game does not exist in the prediction database.",
            action_text="Back to This Week",
            action_url="/",
        )
        assert (
            '<a href="/" class="skew-control skew-control-active">'
            '<span class="unskew">Back to This Week</span></a>'
        ) in html

    def test_error_state_uses_the_reserved_error_reds(self) -> None:
        html = render(
            "_error_state.html", message="Could not load season data", recovery_text="Refresh."
        )
        assert 'class="rounded bg-red-950 border border-red-800 p-6 text-center"' in html
        assert "text-red-200" in html and "Could not load season data" in html
        assert '<p class="mt-1 text-sm text-red-100">Refresh.</p>' in html
        assert "text-red-400" not in html, "the realised-loss red is not an error colour"


class TestControls:
    def test_export_buttons_keep_ids_urls_and_touch_height(self) -> None:
        html = render(
            "_export_buttons.html",
            csv_url="/api/export/csv?season=2026&week=3",
            json_url="/api/export/json?season=2026&week=3",
            season_csv_url="/api/export/csv?season=2026",
        )
        assert 'id="export-buttons"' in html
        for element_id in ("export-csv", "export-json", "export-season-csv"):
            assert f'id="{element_id}"' in html
        assert html.count('data-base-url="/api/export/') == 3
        assert html.count("min-h-[44px]") == 3
        assert html.count('class="skew-control min-h-[44px]"') == 3
        assert html.count('<span class="unskew inline-flex items-center gap-1.5">') == 3
        assert "nfl-" not in html

    def test_export_buttons_without_a_season_link(self) -> None:
        html = render("_export_buttons.html", csv_url="/c", json_url="/j")
        assert "export-season-csv" not in html

    def test_scope_toggle_marks_the_current_scope_only(self) -> None:
        html = render("_betting_scope_toggle.html", current_scope="all")
        assert 'hx-get="/fragments/betting?scope=recommended"' in html
        assert 'hx-get="/fragments/betting?scope=all"' in html
        assert html.count('hx-target="#betting-content"') == 2
        assert html.count('hx-indicator="#betting-loading"') == 2
        assert html.count('aria-pressed="true"') == 1
        assert html.count("skew-control-active") == 1
        pressed = html[html.index('aria-pressed="true"') :]
        assert "skew-control-active" in pressed[: pressed.index(">")]
        assert '<span class="unskew">All bets</span></button>' in html

    def test_sort_controls_keep_their_wiring(self) -> None:
        html = render("_sort_controls.html", current_sort="edge")
        assert 'id="sort-select"' in html and 'for="sort-select"' in html
        assert 'hx-get="/fragments/games"' in html
        assert 'hx-target="#game-grid"' in html
        assert "hx-include=\"[name='week'],[name='season']\"" in html
        assert '<option value="edge" selected>Edge</option>' in html
        assert 'class="skew-control"' in html

    def test_season_selector_keeps_its_wiring(self) -> None:
        html = render(
            "_season_selector.html", available_seasons=[2024, 2023], current_season=2024
        )
        assert 'id="season-select"' in html
        assert 'hx-get="/fragments/performance"' in html
        assert 'hx-target="#performance-content"' in html
        assert 'id="perf-loading"' in html
        assert ">All Seasons</option>" in html
        assert '<option value="2024" selected>2024 Season</option>' in html

    def test_season_tracking_selector_keeps_its_error_handler(self) -> None:
        html = render(
            "_season_tracking_selector.html", available_seasons=[2026, 2025], current_season=2026
        )
        assert 'id="season-track-select"' in html
        assert 'hx-get="/fragments/season"' in html
        assert 'hx-target="#season-content"' in html
        assert "hx-on::response-error=" in html
        assert "season-error-template" in html
        assert "All Seasons" not in html

    @pytest.mark.parametrize("variant", ["cards", "chart", "table"])
    def test_loading_skeleton_is_dark(self, variant: str) -> None:
        html = render("_loading_skeleton.html", variant=variant)
        assert "animate-pulse" in html
        assert "bg-white" not in html and "bg-gray-" not in html


class TestWeekSummary:
    SUMMARY = {
        "total_games": 13,
        "wp_correct": 9,
        "wp_total": 13,
        "wp_pct": 69,
        "ats_correct": 0,
        "ats_total": 0,
        "ats_pct": 0,
        "ou_correct": 7,
        "ou_total": 12,
        "ou_pct": 58,
    }

    def test_three_tiles_keyed_to_the_bet_type_colours(self) -> None:
        html = render("_week_summary.html", week_summary=self.SUMMARY)
        assert 'aria-label="Weekly prediction accuracy summary"' in html
        assert html.count('class="stat-tile"') == 3
        for label in ("Win Prob", "Spread", "Total"):
            assert f'<p class="label">{label}</p>' in html
        assert "--tile-accent: var(--color-target-wp);" in html
        assert "--tile-accent: var(--color-target-ats);" in html
        assert "--tile-accent: var(--color-target-ou);" in html
        assert ">9/13</p>" in html and "(69%)" in html
        assert ">7/12</p>" in html and "(58%)" in html

    def test_a_target_without_odds_reads_n_a(self) -> None:
        html = render("_week_summary.html", week_summary=self.SUMMARY)
        assert '<p class="stat-tile-value text-dim">N/A</p>' in html
        assert "No odds data" in html

    def test_nothing_renders_for_a_week_with_no_completed_game(self) -> None:
        assert render("_week_summary.html", week_summary={}).strip() == ""
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/unit/test_broadcast_components.py -q`
Expected: the new `TestStates`, `TestControls` and `TestWeekSummary` tests fail; the Task 4 tests still pass.

- [ ] **Step 3: Replace the nine small component files**

`web/templates/components/_empty_state.html`:

```jinja
{# Empty state. Pass: heading, body, action_text (optional), action_url (optional).

   A "no data" state, never a failure: grey and dashed, distinct from _error_state.html's red.
   The body stays in a <p> -- tests/api/test_cold_start_bet_list_recovery.py treats paragraphs as
   the instruction blocks a page gives. #}
<div class="border border-dashed border-white/20 px-6 py-10 text-center">
  <h3 class="display text-xl text-fg">{{ heading }}</h3>
  <p class="mt-2 text-sm text-muted max-w-md mx-auto">{{ body }}</p>
  {% if action_text and action_url %}
  <div class="mt-5">
    <a href="{{ action_url }}" class="skew-control skew-control-active"><span class="unskew">{{ action_text }}</span></a>
  </div>
  {% endif %}
</div>
```

`web/templates/components/_error_state.html`:

```jinja
{# Error state. Pass: message, recovery_text.

   A FAILURE, never a "no data" state. These reds -- bg-red-950 / border-red-800, a red-200
   heading, a red-100 body, a red-300 icon -- are reserved for this partial and are deliberately
   NOT the realised-loss red-400 the bets tracker uses, so "a request failed" can never read as
   "a bet lost". The recovery text stays in a <p> (see _empty_state.html). #}
<div class="rounded bg-red-950 border border-red-800 p-6 text-center">
  <svg class="mx-auto h-9 w-9 text-red-300" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true">
    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="1.5" d="M12 9v3.75m9-.75a9 9 0 11-18 0 9 9 0 0118 0zm-9 3.75h.008v.008H12v-.008z"/>
  </svg>
  <h3 class="mt-3 font-display text-lg font-bold tracking-wide text-red-200">{{ message }}</h3>
  <p class="mt-1 text-sm text-red-100">{{ recovery_text }}</p>
</div>
```

`web/templates/components/_export_buttons.html`:

```jinja
{# Export buttons for CSV and JSON downloads.
   When used on the predictions page, JS updates hrefs dynamically based on current dropdown values.
   Pass: csv_url, json_url, season_csv_url (optional), season_json_url (optional)

   ONE partial for every page that offers exports (D31-32); tests/api/test_export.py asserts no
   page builds export links without it. Broadcast restyle: skew controls with an accent download
   icon. min-h-[44px] stays spelled out so the 44-pixel touch height is visible in the markup. #}
<div id="export-buttons" class="flex flex-wrap items-center gap-2">
  <a id="export-csv" href="{{ csv_url }}" data-base-url="/api/export/csv" class="skew-control min-h-[44px]">
    <span class="unskew inline-flex items-center gap-1.5"><svg class="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>CSV</span>
  </a>
  <a id="export-json" href="{{ json_url }}" data-base-url="/api/export/json" class="skew-control min-h-[44px]">
    <span class="unskew inline-flex items-center gap-1.5"><svg class="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>JSON</span>
  </a>
  {% if season_csv_url is defined and season_csv_url %}
  <a id="export-season-csv" href="{{ season_csv_url }}" data-base-url="/api/export/csv" class="skew-control min-h-[44px]">
    <span class="unskew inline-flex items-center gap-1.5"><svg class="w-4 h-4 text-accent" fill="none" stroke="currentColor" viewBox="0 0 24 24" aria-hidden="true"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"/></svg>Export Full Season</span>
  </a>
  {% endif %}
</div>
```

`web/templates/components/_loading_skeleton.html`:

```jinja
{# Loading skeleton component. Pass variant: "cards", "table", or "chart". Dark panels with
   lighter bars, pulsing (the pulse stops under prefers-reduced-motion via custom.css). #}
{% if variant == "cards" %}
<div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 md:gap-6">
  {% for _ in range(6) %}
  <div class="bg-panel rounded p-4 md:p-6 animate-pulse">
    <div class="h-4 bg-panel-2 rounded w-3/4 mb-3"></div>
    <div class="h-3 bg-panel-2 rounded w-1/2 mb-4"></div>
    <div class="grid grid-cols-3 gap-3">
      <div class="h-8 bg-panel-2 rounded"></div>
      <div class="h-8 bg-panel-2 rounded"></div>
      <div class="h-8 bg-panel-2 rounded"></div>
    </div>
  </div>
  {% endfor %}
</div>
{% elif variant == "chart" %}
<div class="bg-panel rounded p-6 animate-pulse">
  <div class="h-4 bg-panel-2 rounded w-1/3 mb-4"></div>
  <div class="h-64 bg-ink rounded"></div>
</div>
{% elif variant == "table" %}
<div class="bg-panel rounded p-6 animate-pulse">
  <div class="h-4 bg-panel-2 rounded w-1/4 mb-4"></div>
  {% for _ in range(5) %}
  <div class="h-3 bg-panel-2 rounded w-full mb-2"></div>
  {% endfor %}
</div>
{% endif %}
```

`web/templates/components/_week_summary.html`:

```jinja
{# Week summary scoreboard: per-target accuracy over the week's completed games.
   Pass: week_summary (dict with wp/ats/ou counts, from pages._compute_week_summary).

   The numbers are exactly what the route computed; this partial only lays them out as three
   scoreboard tiles, each keyed to its bet type's colour (the same keys every chart uses). A
   target with no decided game reads N/A with "No odds data", as before. #}
{% import "components/_broadcast.html" as bc %}
{% if week_summary and week_summary.total_games is defined and week_summary.total_games > 0 %}
<div class="grid grid-cols-3 gap-2 mb-4" role="group" aria-label="Weekly prediction accuracy summary">
  {{ bc.stat_tile("Win Prob", week_summary.wp_correct ~ "/" ~ week_summary.wp_total, sub="(" ~ week_summary.wp_pct ~ "%)", accent="var(--color-target-wp)") }}
  {% if week_summary.ats_total > 0 %}
  {{ bc.stat_tile("Spread", week_summary.ats_correct ~ "/" ~ week_summary.ats_total, sub="(" ~ week_summary.ats_pct ~ "%)", accent="var(--color-target-ats)") }}
  {% else %}
  {{ bc.stat_tile("Spread", "N/A", value_class="text-dim", sub="No odds data", accent="var(--color-target-ats)") }}
  {% endif %}
  {% if week_summary.ou_total > 0 %}
  {{ bc.stat_tile("Total", week_summary.ou_correct ~ "/" ~ week_summary.ou_total, sub="(" ~ week_summary.ou_pct ~ "%)", accent="var(--color-target-ou)") }}
  {% else %}
  {{ bc.stat_tile("Total", "N/A", value_class="text-dim", sub="No odds data", accent="var(--color-target-ou)") }}
  {% endif %}
</div>
{% endif %}
```

`web/templates/components/_season_selector.html`:

```jinja
{# Season selector for the season-metrics table. Pass: available_seasons (list of ints),
   current_season. Wiring unchanged by the Broadcast restyle; only the classes moved. #}
<div class="flex items-center gap-2">
  <label for="season-select" class="label">Season</label>
  <select id="season-select" name="season"
          hx-get="/fragments/performance"
          hx-target="#performance-content"
          hx-swap="innerHTML"
          hx-indicator="#perf-loading"
          class="skew-control">
    <option value="" {% if not current_season %}selected{% endif %}>All Seasons</option>
    {% for s in available_seasons %}
    <option value="{{ s }}" {% if s == current_season %}selected{% endif %}>{{ s }} Season</option>
    {% endfor %}
  </select>
  <div id="perf-loading" class="htmx-indicator">
    <div class="loading-spinner"></div>
  </div>
</div>
```

`web/templates/components/_season_tracking_selector.html`:

```jinja
{# Season Tracking selector. Pass: available_seasons (list of ints), current_season.
   Forked from _season_selector.html for the /season page (D-04 / D-11):
   - retargets the HTMX swap at the season fragment route + #season-content
   - DROPS the aggregate all-seasons option (D-04): the page always tracks one
     season, defaulting to the latest (D-01).

   Error handling (D-17 — the first HTMX error wiring in the app):
   On a failed swap (4xx/5xx) HTMX fires `htmx:responseError` but, by default,
   does NOT swap content into the target (detail.shouldSwap is false for non-200
   responses -- verified against htmx.org/events for htmx 2.0.4). Without this
   handler #season-content would keep its stale/partial markup. The inline
   `hx-on::response-error` handler (HTMX-native, no extension, no new JS bundle)
   copies the server-rendered `_error_state.html` markup from the hidden
   <template id="season-error-template"> in season.html into #season-content, so
   a failed load degrades to the red "Could not load season data" error card.
   The `hx-on::` double-colon shorthand + kebab-case `response-error` is the
   correct htmx 2.0 form (DOM attributes are lowercased, so `responseError` would
   not bind). This error path (red) is DISTINCT from the gray "No completed games
   yet" empty state, which is a no-data case, not a failure.

   Broadcast restyle: only the classes moved; every attribute above is unchanged. #}
<div class="flex items-center gap-2">
  <label for="season-track-select" class="label">Season</label>
  <select id="season-track-select" name="season"
          hx-get="/fragments/season"
          hx-target="#season-content"
          hx-swap="innerHTML"
          hx-indicator="#season-loading"
          hx-on::response-error="var t = document.getElementById('season-error-template'); var c = document.getElementById('season-content'); if (t && c) { c.innerHTML = t.innerHTML; }"
          class="skew-control">
    {% for s in available_seasons %}
    <option value="{{ s }}" {% if s == current_season %}selected{% endif %}>{{ s }} Season</option>
    {% endfor %}
  </select>
  <div id="season-loading" class="htmx-indicator">
    <div class="loading-spinner"></div>
  </div>
</div>
```

The original comment contains one non-ASCII em dash, in `(D-17 — the first ...)`. Keep it exactly as it is in the file today; nothing requires this partial to be ASCII.

`web/templates/components/_betting_scope_toggle.html`:

```jinja
{# Betting scope toggle. Pass: current_scope ("all" | "recommended").

   Two-segment button group (Recommended / All bets) that swaps the
   #betting-content block via HTMX. Both scope variants are pre-rendered into
   the cache, so each click reads cached HTML/JSON only (zero recompute). The
   active segment is driven by current_scope; default selected is Recommended
   (D-17). Mirrors the _season_selector HTMX attribute set (hx-get, hx-target,
   hx-swap, hx-indicator) per RESEARCH Pattern 3.

   Broadcast restyle: two skew controls; the active one is the yellow skew-control-active, and
   aria-pressed still says which is pressed for assistive technology. #}
<div class="flex flex-wrap items-center gap-3">
  <div class="inline-flex items-center gap-1" role="group" aria-label="Bet scope">
    <button type="button"
            hx-get="/fragments/betting?scope=recommended"
            hx-target="#betting-content"
            hx-swap="innerHTML"
            hx-indicator="#betting-loading"
            aria-pressed="{{ 'true' if current_scope == 'recommended' else 'false' }}"
            class="skew-control{% if current_scope == 'recommended' %} skew-control-active{% endif %}"><span class="unskew">Recommended</span></button>
    <button type="button"
            hx-get="/fragments/betting?scope=all"
            hx-target="#betting-content"
            hx-swap="innerHTML"
            hx-indicator="#betting-loading"
            aria-pressed="{{ 'true' if current_scope == 'all' else 'false' }}"
            class="skew-control{% if current_scope == 'all' %} skew-control-active{% endif %}"><span class="unskew">All bets</span></button>
  </div>
  <div id="betting-loading" class="htmx-indicator">
    <div class="loading-spinner"></div>
  </div>
</div>
<p class="text-xs text-muted mt-2 max-w-3xl">"Recommended" = bets the simulation sized with real money under Kelly (genuine model edge). "All bets" includes every directional bet the simulation placed.</p>
```

`web/templates/components/_sort_controls.html`:

```jinja
{# Sort controls dropdown. HTMX-powered.
   Pass: current_sort, current_week, current_season #}
<div class="flex items-center gap-2">
  <label for="sort-select" class="label">Sort</label>
  <select id="sort-select" name="sort"
          hx-get="/fragments/games"
          hx-target="#game-grid"
          hx-swap="innerHTML"
          hx-include="[name='week'],[name='season']"
          class="skew-control">
    <option value="time" {% if current_sort == "time" %}selected{% endif %}>Game Time</option>
    <option value="confidence" {% if current_sort == "confidence" %}selected{% endif %}>Confidence</option>
    <option value="edge" {% if current_sort == "edge" %}selected{% endif %}>Edge</option>
  </select>
</div>
```

- [ ] **Step 4: Restyle `_week_selector.html` without touching its logic**

Make exactly these edits. Nothing else in the file changes.

Edit 1 -- old (lines 5-10 of the comment):

```
   PARAMETERISED IN PLACE for D31-26 so ONE selector serves both / and /bets. It is deliberately
   NOT forked: a second selector file is a duplicated definition that drifts, and this repo has a
   documented incident of exactly that failure mode (29-06). Every parameter below DEFAULTS to the
   value this component shipped with, so the This Week page renders BYTE-IDENTICALLY -- pinned by
   two recorded snapshots in tests/api/test_bets_page.py. No layout class and no type class was
   touched; the diff is the attribute values, the failure handlers and the id derivation.
```

new:

```
   PARAMETERISED IN PLACE for D31-26 so ONE selector serves both / and /bets. It is deliberately
   NOT forked: a second selector file is a duplicated definition that drifts, and this repo has a
   documented incident of exactly that failure mode (29-06). Every parameter below DEFAULTS to the
   value this component shipped with, so the This Week page renders exactly the recorded markup --
   pinned by two snapshots in tests/api/snapshots/, compared in tests/api/test_pages.py. The
   Broadcast redesign (2026-10) restyled the classes and RE-RECORDED both snapshots; every hx-*
   attribute, id, label and option was left byte-identical, and the re-recording step checks that.
```

Edit 2 -- old:

```
<div class="flex flex-wrap items-center gap-3">
```

new:

```
<div class="week-selector flex flex-wrap items-center gap-2">
```

Edit 3 -- old:

```
    <label for="{{ _ws_season_id }}" class="text-sm font-semibold text-gray-700">Season</label>
```

new:

```
    <label for="{{ _ws_season_id }}" class="label">Season</label>
```

Edit 4 -- old (the season select's class line; it is preceded by the `hx-vals` line, which makes it unique):

```
{% endif %}            hx-vals='{"week": ""}'
            class="rounded-md border-gray-300 text-sm font-medium py-2 px-3 min-h-[44px]">
```

new:

```
{% endif %}            hx-vals='{"week": ""}'
            class="skew-control">
```

Edit 5 -- old:

```
  <label for="{{ _ws_week_id }}" class="text-sm font-semibold text-gray-700">Week</label>
```

new:

```
  <label for="{{ _ws_week_id }}" class="label">Week</label>
```

Edit 6 -- old (the week select's class line; it starts with `{% endif %}`, which makes it unique):

```
{% endif %}            class="rounded-md border-gray-300 text-sm font-medium py-2 px-3 min-h-[44px]">
```

new:

```
{% endif %}            class="skew-control">
```

Edit 7 -- old (two occurrences, the previous and next buttons -- replace both):

```
            class="p-2 rounded-md border border-gray-300 hover:bg-gray-50 disabled:opacity-40 disabled:cursor-not-allowed min-h-[44px] min-w-[44px] flex items-center justify-center"
```

new:

```
            class="skew-control"
```

Edit 8 -- the previous-week chevron moves into a counter-skewed span (the button itself is now skewed). Old:

```
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg>
```

new:

```
      <span class="unskew"><svg class="w-4 h-4 text-accent" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 19l-7-7 7-7"/></svg></span>
```

Edit 9 -- the next-week chevron, the same way. Old:

```
      <svg class="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg>
```

new:

```
      <span class="unskew"><svg class="w-4 h-4 text-accent" aria-hidden="true" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 5l7 7-7 7"/></svg></span>
```

Then update `tests/api/week_selector_snapshot.py`. Old:

```python
#: The selector's outermost element. Its class string is pinned by the snapshots too, so a change
#: here is caught rather than silently retargeting the extraction.
SELECTOR_OPEN = '<div class="flex flex-wrap items-center gap-3">'
```

New:

```python
#: The selector's outermost element. Its class string is pinned by the snapshots too, so a change
#: here is caught rather than silently retargeting the extraction. The ``week-selector`` class
#: (Broadcast redesign, 2026-10) keeps the tag unique on every page that renders the selector.
SELECTOR_OPEN = '<div class="week-selector flex flex-wrap items-center gap-2">'
```

- [ ] **Step 5: Compile the stylesheet**

```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
```

- [ ] **Step 6: Re-record the two selector snapshots, and prove the wiring did not move**

Create a TEMPORARY file `tests/api/test_zz_regenerate_selector_snapshots.py`. Never commit it.

```python
"""TEMPORARY -- re-records the two week-selector snapshots. Delete after running; never commit."""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.dependencies import templates
from tests.api.week_selector_snapshot import PRE_PARAM_CONTEXT, SNAPSHOT_DIR, extract_selector


def test_regenerate(test_client: TestClient) -> None:
    this_week = extract_selector(test_client.get("/").text).replace("\r\n", "\n")
    (SNAPSHOT_DIR / "week_selector_this_week.html").write_text(
        this_week, encoding="utf-8", newline="\n"
    )
    prev_next = (
        templates.env.get_template("components/_week_selector.html")
        .render(**PRE_PARAM_CONTEXT)
        .replace("\r\n", "\n")
    )
    (SNAPSHOT_DIR / "week_selector_prev_next.html").write_text(
        prev_next, encoding="utf-8", newline="\n"
    )
```

The wiring check compares the old and new snapshots with every ` class="..."` attribute (and any carriage return) stripped from BOTH sides first. Edits 3 and 5 change only the class on the two `<label for=...>` lines; diffing the raw files would print those lines on every run, because they contain `for=`, and the check could never pass.

```bash
uv run pytest tests/api/test_zz_regenerate_selector_snapshots.py -q
rm tests/api/test_zz_regenerate_selector_snapshots.py
git diff --stat tests/api/snapshots/
for snap in week_selector_this_week.html week_selector_prev_next.html; do
  diff <(git show "HEAD:tests/api/snapshots/$snap" | sed -E 's/\r$//; s/ class="[^"]*"//g') \
       <(sed -E 's/\r$//; s/ class="[^"]*"//g' "tests/api/snapshots/$snap")
done | grep -E "^[<>]" | grep -E "hx-|id=|for=|aria-label|<option|value=|Week [0-9]|data-season|disabled" || echo "WIRING UNCHANGED"
```

Expected:
- The first command passes and writes the two files.
- `git diff --stat` shows both snapshot files changed.
- The final pipeline prints `WIRING UNCHANGED`. With classes stripped, the only lines left that differ are the two chevrons (Edits 8 and 9: the new `<span>` wrapper and `aria-hidden="true"`), and neither matches the wiring pattern.

If the pipeline prints any line, the restyle altered wiring. Revert that edit (`git checkout -- web/templates/components/_week_selector.html`) and redo Step 4.

- [ ] **Step 7: Update the tests that pinned the old error classes and the export byte guard**

`tests/api/test_pages.py` -- both occurrences, in `test_season_error_state_wired_and_distinct_from_empty` and `test_season_error_copy_renders_via_error_component`. Old (replace both):

```python
    assert "bg-red-50" in html
```

New:

```python
    assert "bg-red-950" in html
```

`tests/api/test_bets_page.py::test_state_four_a_week_with_no_list_for_its_locked_games_is_refused`. Old:

```python
    assert "bg-red-50" in body, "the refusal did not render through _error_state.html"
```

New:

```python
    assert "bg-red-950" in body, "the refusal did not render through _error_state.html"
```

`tests/api/test_bets_page.py::test_the_failure_template_carries_the_message_and_a_retry`. Old:

```python
    assert "bg-red-50" in template, (
```

New:

```python
    assert "bg-red-950" in template, (
```

`tests/api/test_bets_page.py::test_the_forward_block_is_withheld_under_the_hard_block_while_replay_stays_readable`. Old:

```python
    assert "bg-red-50" in forward, (
```

New:

```python
    assert "bg-red-950" in forward, (
```

`tests/api/test_bets_page.py::test_green_and_red_appear_only_inside_the_tracker_sections` is deliberately NOT edited here. Part C's Task 12 rewrites it whole. It keeps passing in the meantime: above the tracker, `text-red-600` and `bg-red-50` now both count zero.

`tests/api/test_export.py` -- the byte guard pinned the partial's LOOK, which the redesign changes on purpose (spec section 11). D31-32's actual requirement is that one partial is reused everywhere, so assert that instead. Old (the whole test):

```python
def test_the_export_buttons_partial_is_reused_byte_for_byte() -> None:
    """D31-32 reuses ``_export_buttons.html`` VERBATIM; no class inside it is edited."""
    repo_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            "git",
            "diff",
            "--exit-code",
            "HEAD",
            "--",
            "web/templates/components/_export_buttons.html",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"the export buttons partial was modified:\n{result.stdout}"
    )
```

New:

```python
def test_exactly_one_export_buttons_partial_serves_every_page() -> None:
    """D31-32: ONE export partial, reused by every page that offers exports -- never a fork.

    This used to compare the partial byte-for-byte against HEAD, which pinned its LOOK rather than
    its reuse; the Broadcast redesign restyled it on purpose (spec section 11). What D31-32 needs
    is that /bets builds its export links through the same partial as every other page, so that
    is what is asserted: one partial on disk, included by every page that renders export links.
    """
    templates_dir = Path(__file__).resolve().parents[2] / "web" / "templates"
    partials = sorted(p.name for p in (templates_dir / "components").glob("*export*.html"))
    assert partials == ["_export_buttons.html"], f"a second export partial appeared: {partials}"
    for page in sorted((templates_dir / "pages").glob("*.html")):
        source = page.read_text(encoding="utf-8")
        if "/api/export/" in source:
            assert "components/_export_buttons.html" in source, (
                f"{page.name} builds export links without the shared partial"
            )
```

Then delete the now-unused import. Old:

```python
import re
import subprocess
import threading
```

New:

```python
import re
import threading
```

- [ ] **Step 8: Run every affected test**

```bash
uv run pytest tests/unit/test_broadcast_components.py -q
uv run pytest tests/unit/test_broadcast_macros.py -q
uv run pytest "tests/api/test_pages.py::test_the_this_week_selector_renders_byte_identically" "tests/api/test_pages.py::test_the_prev_next_branch_renders_byte_identically" "tests/api/test_pages.py::test_the_this_week_page_still_targets_the_games_grid" "tests/api/test_pages.py::test_season_error_state_wired_and_distinct_from_empty" "tests/api/test_pages.py::test_season_error_copy_renders_via_error_component" "tests/api/test_pages.py::test_game_detail_not_found" "tests/api/test_pages.py::test_this_week_page_with_sort" "tests/api/test_pages.py::test_betting_fragment" "tests/api/test_pages.py::test_betting_fragment_scope_whitelist" "tests/api/test_pages.py::test_performance_fragment" -q
uv run pytest tests/api/test_bets_page.py -k "selector or failure or timeout or sync or element_ids or layout_and_type or state_four or withheld or green_and_red or empty or nothing_graded" -q
uv run pytest tests/api/test_export.py -q
uv run pytest tests/api/test_cold_start_bet_list_recovery.py -q
uv run pytest tests/api/test_fragments.py -q
uv run pytest tests/unit/test_page_labels.py -q
uv run pytest tests/api/test_page_labels_routes.py -q
uv run ruff check tests/api/test_export.py tests/api/week_selector_snapshot.py tests/unit/test_broadcast_components.py tests/api/test_pages.py tests/api/test_bets_page.py
uv run ruff format tests/api/test_export.py tests/api/week_selector_snapshot.py tests/unit/test_broadcast_components.py tests/api/test_pages.py tests/api/test_bets_page.py
uv run pyright tests/api/test_export.py tests/api/week_selector_snapshot.py tests/unit/test_broadcast_components.py tests/api/test_pages.py tests/api/test_bets_page.py
git status --short
```

Expected:
- All PASS, and ruff is clean. pyright reports no error on a line this task wrote; an error elsewhere in an existing test file predates the branch -- note it in the task report and leave it.
- `test_cold_start_bet_list_recovery.py` still finds every instruction inside a `<p>`.
- `git status --short` lists no `test_zz_regenerate_selector_snapshots.py`.

- [ ] **Step 9: Commit**

```bash
git add web/templates/components/_week_selector.html web/templates/components/_season_selector.html web/templates/components/_season_tracking_selector.html web/templates/components/_betting_scope_toggle.html web/templates/components/_sort_controls.html web/templates/components/_empty_state.html web/templates/components/_error_state.html web/templates/components/_export_buttons.html web/templates/components/_loading_skeleton.html web/templates/components/_week_summary.html web/static/css/tailwind-compiled.css tests/api/week_selector_snapshot.py tests/api/snapshots/week_selector_this_week.html tests/api/snapshots/week_selector_prev_next.html tests/api/test_pages.py tests/api/test_bets_page.py tests/api/test_export.py tests/unit/test_broadcast_components.py
git commit -m "$(cat <<'EOF'
feat(redesign): restyle selectors, toggles, states and exports; re-record selector snapshots

Every shared control becomes a Broadcast skew control and every state a dark
panel: the week, season and season-tracking selectors, the scope toggle, the
sort control, the export buttons, the loading skeleton and the week summary
(now three scoreboard tiles keyed to the bet-type colours). Text in every skewed
control sits in an upright .unskew span. The empty state is a grey dashed box;
the error state uses reserved reds (red-950/800/200/100) that can never be read
as a lost bet. hx-* wiring, ids, labels and wording are
unchanged; the two week-selector snapshots are re-recorded after a check that
only classes moved. The export byte guard now asserts what D31-32 needs: one
partial, reused by every page that offers exports.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

# Part B -- This Week and game detail (Tasks 6-9)

## Contract notes (Part B)

These refine the Shared Interface Contract where the real code forced a decision. Later parts must use them as written.

1. **`slate_groups` is built for every TIME order, not only `sort == "time"`.** `DataService.get_predictions` orders by kickoff for any sort it does not recognise, so a garbage `?sort=` is still a time order. `slate_groups` is `None` only when `current_sort` is `"confidence"` or `"edge"` (`pages._NON_TIME_SORTS`).
2. **The card learns about bets through `game["bet_targets"]: list[str]`** (the live bet types the game carries this week, e.g. `["ou"]`), not a boolean. The card treats a non-empty list as "on the bet list" and draws the matching row's edge chip solid. Absent key = `[]`.
3. **Each headliner bet carries `bet["game"]`**: the decorated prediction row for its game, or `None` when the week's predictions do not include it. The bet row itself is otherwise exactly what `DataService.get_bet_list` returned.
4. **The headliner cards show stake, EV and EV band only -- no "edge".** Spec 7.1 lists an edge, but the only stored edge is the prediction row's, measured against a different line than the bet's. Showing it beside the bet would put two numbers on the card that need not agree. Every number on a headliner card is a stored column of the bet row.
5. **`_build_headliner` reports `"no_week"` when /bets would resolve the requested week to a different one.** This Week can show a predictions week that has no schedule-derived bet week. `/bets?season=S&week=W` would quietly show another week, so that week's bets are not this week's. The one exception is `"not_built"`, which does not depend on the week.
6. **This Week's `old_rule_scope` covers the headliner bets too:** `_rows_seasons(games) + _rows_seasons(headliner["bets"])`. The headliners live in the same `game_grid` block, so the block still carries exactly ONE old-rule label.
7. **New shared partial `web/templates/components/_bet_pick.html`** (Task 8). It holds `side_line(bet, fmt)`, `picked_team(bet)` and `pick_label(bet)`, moved VERBATIM out of `pages/bets.html`. Task 8 makes `bets.html` import it. Tasks 11 and 12 (Part C) must keep `{% import "components/_bet_pick.html" as bp %}` inside the `bets_content` block and call `bp.pick_label(bet)` / `bp.side_line(bet, fmt)`. They must never redefine the spread sign logic.
8. **Every `{% import %}` a block needs must sit INSIDE that block.** `jinja2_fragments.render_block` renders a block through `template.blocks[name]` without running the template's top-level statements. A top-level import is therefore undefined in every HTMX fragment. This applies to every page with a fragment block: `game_grid`, `bets_content`, `season_tracking_content`, `performance_content`, `betting_content`.
9. **`components/_prediction_values.html` changes in Task 7**:
   - It gains `team_pct(prob, side)`, one team's own win chance for the score bug.
   - `absent()` changes from `text-gray-400 italic` to `text-dim italic`.
10. **Markers kept for Part D's label-test machinery:**
    - `pages/this_week.html` keeps `<title>` block text `This Week's Predictions`.
    - `pages/game_detail.html` keeps `<title>` block text `{{ away }} @ {{ home }}` and adds a visually hidden `<h1>` with the same text.
    - Each still renders exactly ONE `data-old-rule-label` when unwired.
11. **Data hooks added for tests:**
    - `data-edge="wp|ats|ou"` wraps every edge chip on the card and on the detail page.
    - `data-band="<level>"` wraps each confidence band on the detail page.
    - `data-on-bet-list` marks a card with a live bet.
    - `data-window="<label>"` marks each TV-window section.
    - `data-headliner-state="<state>"`, `data-headliner-bet="<game_id>:<target>"` and `data-headliner-note` mark the headliner area.

---

### Task 6: This Week context -- decorated games, TV-window groups, headliner state

**Files:**
- Modify: `api/routes/pages.py` -- imports (lines 23-38), new helpers after `_build_bets_context` (after line 617), `this_week_page` (lines 735-797)
- Modify: `api/routes/fragments.py` -- imports (lines 18-30), `games_fragment` (lines 36-84)
- Create: `tests/api/test_this_week_headliner.py`

**Interfaces:**
- Consumes (Task 2): `api.presentation.decorate_game(game) -> dict`, `api.presentation.group_games_by_window(games) -> list[{"label": str, "games": list}]`.
- Consumes (existing): `pages._normalize_week`, `pages._build_bets_context`, `pages._rows_seasons`.
- Produces:
  - `pages.HEADLINER_STATES: tuple[str, ...] = ("not_built", "blocked", "not_evaluated", "no_week", "none_cleared", "bets")`
  - `pages._headliner_state(bets_context: dict[str, Any]) -> str`
  - `pages._build_headliner(service: DataService, season: int | None, week: int | None, request: Request) -> dict[str, Any]` returning `{"state", "bets", "season", "week"}`
  - `pages._this_week_grid_context(service, games, season, week, sort, request) -> {"games": list[dict], "slate_groups": list[dict] | None, "headliner": dict}`
  - each game dict gains `bet_targets: list[str]`; each headliner bet gains `game: dict | None`
  - template context keys `games`, `slate_groups`, `headliner` on `/` and `/fragments/games`

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_this_week_headliner.py`:

```python
"""This Week's headliner area agrees with /bets about a week (Broadcast redesign, Task 6).

The landing page now leads with the week's bets. It must never describe a week differently from
/bets, so the headliner's state is not decided a second time: it is read off the same
``_build_bets_context`` /bets renders from, in pages/bets.html's own precedence. These tests build
one REAL cache per /bets state, render /bets for it, and check the headliner names the same state.
They then pin the decorated slate the game grid renders from.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import duckdb
import pandas as pd
import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from api.cache import (
    BET_LIST_COLUMNS,
    CACHE_SCHEMA,
    GRADING_STATUS_PENDING,
    PREDICTIONS_TABLE_COLUMNS,
    bet_list_populated_at_key,
    materialize_available_bet_weeks,
    materialize_bet_list,
    materialize_bet_week_freeze,
)
from api.routes.pages import (
    HEADLINER_STATES,
    _build_headliner,
    _headliner_state,
    _this_week_grid_context,
)
from api.services import DataService, clear_cache

_SEASON = 2023
_WEEK = 1
_GAME = "2023_W01_DET@KC"
_SECOND_GAME = "2023_W01_CAR@ATL"
_PAST_LOCK = datetime(2023, 9, 7, 22, 0, 0, tzinfo=UTC)
_FUTURE_LOCK = datetime(2099, 9, 7, 22, 0, 0, tzinfo=UTC)

# The text only one /bets render carries, per state. "bets" is the render that carries none.
_BETS_PAGE_MARKERS: dict[str, str] = {
    "not_built": "Bet list not built yet",
    "blocked": "its locked games have no list in the cache",
    "not_evaluated": "No game of this week has reached its lock.",
    "no_week": "No current week",
    "none_cleared": "No bets cleared the floor this week",
}


class _StubRequest:
    """The one attribute a full-page render needs from a request: ``url_for`` for static files."""

    def url_for(self, name: str, **path_params: Any) -> str:
        return f"/{name}/{path_params.get('path', '')}"


_REQUEST = cast(Request, _StubRequest())


def _rendered_state(html: str) -> str:
    """Which of the six renders a /bets page made, read off the text only that render carries."""
    present = [state for state, marker in _BETS_PAGE_MARKERS.items() if marker in html]
    assert len(present) <= 1, f"/bets rendered more than one state at once: {present}"
    return present[0] if present else "bets"


def _bet_row(status: str) -> dict[str, Any]:
    """One bet_list row for _GAME carrying the minimum /bets renders, live or suppressed."""
    row = dict.fromkeys(BET_LIST_COLUMNS)
    row.update(
        {
            "game_id": _GAME,
            "season": _SEASON,
            "week": _WEEK,
            "target": "ou",
            "bet_side": "under",
            "line": 47.5,
            "status": status,
            "snapshot_ts": "2023-09-07T18:00:00-04:00",
            "freeze_ts": "2023-09-07T18:00:00-04:00",
            "provenance": "backtest_replay",
            "validation_type": "contaminated",
            "grading_status": GRADING_STATUS_PENDING,
            "outcome": None,
        }
    )
    if status == "live":
        row.update(
            {
                "per_bet_ev": 0.0625,
                "stake_units": 1.5,
                "ev_tier": "high",
                "selected_odds": -110.0,
                "flat_stake": 1.0,
            }
        )
    else:
        row["rejection_reason"] = "ev_below_floor"
    return row


def _prediction(game_id: str, game_date: datetime | None) -> dict[str, Any]:
    """One scheduled prediction row for *game_id*, shaped like the predictions table."""
    away, home = game_id.split("_")[-1].split("@")
    row = dict.fromkeys(PREDICTIONS_TABLE_COLUMNS)
    row.update(
        game_id=game_id,
        season=_SEASON,
        week=_WEEK,
        game_date=game_date,
        home_team=home,
        away_team=away,
        status="scheduled",
        wp_prob=0.6,
        ats_prediction=3.0,
        ou_prediction=44.0,
    )
    return row


def _build_state_cache(db_path: Path, state: str) -> None:
    """Build a cache in which /bets?season=2023&week=1 makes the render named *state*.

    Each state is built the way tests/api/test_bets_page.py builds it, from the same
    materializers production calls, so the parity below is checked against real cache shapes.
    """
    conn = duckdb.connect(str(db_path))
    try:
        for statement in CACHE_SCHEMA.strip().split(";"):
            if statement.strip():
                conn.execute(statement)
        stamped = datetime(2023, 9, 7, 21, 0, 0)
        conn.executemany(
            "INSERT OR REPLACE INTO cache_meta VALUES (?, ?, ?)",
            [
                [
                    bet_list_populated_at_key(_SEASON, _WEEK),
                    "2023-09-07T21:00:00+00:00",
                    stamped,
                ],
                ["last_updated", stamped.isoformat(), stamped],
            ],
        )
        if state == "no_week":
            return
        materialize_available_bet_weeks(
            conn, pd.DataFrame([{"game_id": _GAME, "season": _SEASON, "week": _WEEK}])
        )
        lock = _FUTURE_LOCK if state == "not_evaluated" else _PAST_LOCK
        materialize_bet_week_freeze(
            conn,
            pd.DataFrame(
                [
                    {
                        "game_id": _GAME,
                        "season": _SEASON,
                        "week": _WEEK,
                        "game_freeze_ts": lock,
                    }
                ]
            ),
        )
        if state == "not_built":
            conn.execute("DROP TABLE bet_list")
        elif state == "none_cleared":
            materialize_bet_list(conn, pd.DataFrame([_bet_row("suppressed")]))
        elif state == "bets":
            materialize_bet_list(conn, pd.DataFrame([_bet_row("live")]))
    finally:
        conn.close()


@contextmanager
def _serving(db_path: Path) -> Iterator[tuple[TestClient, duckdb.DuckDBPyConnection]]:
    """A TestClient serving *db_path*, and the read-only connection it serves through."""
    from api.dependencies import get_db
    from api.main import app

    clear_cache()
    conn = duckdb.connect(str(db_path), read_only=True)
    app.state.db_lock = threading.RLock()
    app.dependency_overrides[get_db] = lambda: conn
    try:
        yield TestClient(app, raise_server_exceptions=False), conn
    finally:
        app.dependency_overrides.clear()
        conn.close()


# ---------------------------------------------------------------------------
# The headliner state is the /bets render, never a second decision
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("state", HEADLINER_STATES)
def test_the_headliner_names_the_render_bets_makes_for_the_same_week(
    tmp_path: Path, state: str
) -> None:
    db_path = tmp_path / f"{state}.duckdb"
    _build_state_cache(db_path, state)

    with _serving(db_path) as (client, conn):
        bets_html = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
        clear_cache()
        headliner = _build_headliner(DataService(conn), _SEASON, _WEEK, _REQUEST)

    assert _rendered_state(bets_html) == state, (
        f"the fixture for {state!r} did not make /bets render that state; the parity below "
        "would compare the wrong things"
    )
    assert headliner["state"] == state, (
        f"/bets renders {state!r} for this week but the This Week headliner says "
        f"{headliner['state']!r}"
    )
    assert headliner["season"] == _SEASON and headliner["week"] == _WEEK


_QUIET: dict[str, Any] = {
    "bet_list_available": True,
    "bets_blocked": False,
    "week_not_evaluated": False,
    "current_week": _WEEK,
    "bets": [],
}


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        # Every flag at once: the missing TABLE wins, as in pages/bets.html.
        (
            {"bet_list_available": False, "bets_blocked": True, "week_not_evaluated": True},
            "not_built",
        ),
        ({"bets_blocked": True, "week_not_evaluated": True}, "blocked"),
        ({"week_not_evaluated": True, "current_week": None}, "not_evaluated"),
        ({"current_week": None, "bets": [{"game_id": _GAME}]}, "no_week"),
        ({}, "none_cleared"),
        ({"bets": [{"game_id": _GAME}]}, "bets"),
    ],
)
def test_the_state_follows_the_bets_page_precedence(
    flags: dict[str, Any], expected: str
) -> None:
    assert _headliner_state({**_QUIET, **flags}) == expected


def test_a_week_bets_would_not_show_has_no_list_of_its_own(tmp_path: Path) -> None:
    """A predictions week with no schedule row resolves to ANOTHER week on /bets.

    Its bets are not this week's, so the headliner says there is no list rather than borrowing
    the other week's.
    """
    db_path = tmp_path / "mismatch.duckdb"
    _build_state_cache(db_path, "bets")

    with _serving(db_path) as (_client, conn):
        headliner = _build_headliner(DataService(conn), _SEASON, _WEEK + 1, _REQUEST)

    assert headliner == {
        "state": "no_week",
        "bets": [],
        "season": _SEASON,
        "week": _WEEK + 1,
    }


# ---------------------------------------------------------------------------
# The decorated slate the game grid renders from
# ---------------------------------------------------------------------------


def test_the_grid_context_decorates_marks_bets_and_groups_only_a_time_order(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "grid.duckdb"
    _build_state_cache(db_path, "bets")
    games = [
        _prediction(_GAME, datetime(2023, 9, 7, 20, 20)),  # Thursday night
        _prediction(_SECOND_GAME, datetime(2023, 9, 10, 13, 0)),  # Sunday 1:00
    ]

    with _serving(db_path) as (_client, conn):
        service = DataService(conn)
        by_time = _this_week_grid_context(service, games, _SEASON, _WEEK, "time", _REQUEST)
        by_edge = _this_week_grid_context(service, games, _SEASON, _WEEK, "edge", _REQUEST)
        by_band = _this_week_grid_context(
            service, games, _SEASON, _WEEK, "confidence", _REQUEST
        )
        unknown = _this_week_grid_context(service, games, _SEASON, _WEEK, "zzz", _REQUEST)

    assert [g["bet_targets"] for g in by_time["games"]] == [["ou"], []]
    assert all("home_color" in g and "kickoff_label" in g for g in by_time["games"])
    assert "home_color" not in games[0], "the source rows were mutated"

    assert by_time["slate_groups"] is not None
    assert [len(group["games"]) for group in by_time["slate_groups"]] == [1, 1]
    assert by_edge["slate_groups"] is None
    assert by_band["slate_groups"] is None
    # get_predictions orders an unknown sort by kickoff, so it is a time order and keeps groups.
    assert unknown["slate_groups"] is not None

    assert by_time["headliner"]["state"] == "bets"
    assert by_time["headliner"]["bets"][0]["game"]["game_id"] == _GAME
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run pytest tests/api/test_this_week_headliner.py -v`
Expected: collection ERROR -- `ImportError: cannot import name 'HEADLINER_STATES' from 'api.routes.pages'`.

- [ ] **Step 3: Add the helpers to `api/routes/pages.py`**

Edit the import block (lines 23-38). Old:

```python
from api.charts import BETTING_CHART_IDS, INSIGHTS_CHART_IDS
from api.dependencies import get_data_service, templates
```

New:

```python
from api.charts import BETTING_CHART_IDS, INSIGHTS_CHART_IDS
from api.dependencies import get_data_service, templates
from api.presentation import decorate_game, group_games_by_window
```

Insert this block immediately AFTER `_build_bets_context` (after its closing `}` at line 617, before `def _annotate_wp_correct`):

```python
# ---------------------------------------------------------------------------
# This Week: the headliner bets and the decorated slate (Broadcast redesign)
# ---------------------------------------------------------------------------

# The renders /bets chooses between for one week, in the order pages/bets.html tests them. The
# This Week headliner area names one of them, so the two pages can be checked to agree on a week.
HEADLINER_STATES: tuple[str, ...] = (
    "not_built",
    "blocked",
    "not_evaluated",
    "no_week",
    "none_cleared",
    "bets",
)

# get_predictions orders by kickoff for every sort it does not recognise, so these two are the
# only NON-time orders -- and only a time order may be cut into TV windows without the groups
# contradicting the order the reader asked for.
_NON_TIME_SORTS: frozenset[str] = frozenset({"confidence", "edge"})


def _headliner_state(bets_context: dict[str, Any]) -> str:
    """Which render /bets makes for a week, decided in pages/bets.html's own precedence.

    The order is the template's: a missing TABLE first (the only state that names an action the
    reader can take), then the per-game refusal, then a week whose locks are all ahead, then no
    current week, then a week that admitted nothing. Deciding it here in any other order would let
    / and /bets describe the same week two ways; tests/api/test_this_week_headliner.py renders
    /bets for each state and checks the two agree.
    """
    if not bets_context["bet_list_available"]:
        return "not_built"
    if bets_context["bets_blocked"]:
        return "blocked"
    if bets_context["week_not_evaluated"]:
        return "not_evaluated"
    if bets_context["current_week"] is None:
        return "no_week"
    if not bets_context["bets"]:
        return "none_cleared"
    return "bets"


def _build_headliner(
    service: DataService,
    season: int | None,
    week: int | None,
    request: Request,
) -> dict[str, Any]:
    """The week's live bets and their /bets state, for the This Week headliner area.

    Built from THE /bets context (``_build_bets_context``) for the week /bets itself would serve
    for this season and week, so no state is re-derived. When /bets would resolve the request to
    a DIFFERENT week -- This Week can show a predictions week with no schedule-derived bet week --
    the list it would show is not this week's, so the state is "no_week" and no bet is borrowed.
    A missing bet-list table does not depend on the week and keeps its own state.

    Reads cached rows only; nothing here is a metric (UIAP-01).
    """
    resolved = _normalize_week(service, season, week)
    bets_context = _build_bets_context(service, resolved[0], resolved[1], request)
    state = _headliner_state(bets_context)
    if state != "not_built" and resolved != (season, week):
        state = "no_week"
    return {
        "state": state,
        "bets": bets_context["bets"] if state == "bets" else [],
        "season": season,
        "week": week,
    }


def _this_week_grid_context(
    service: DataService,
    games: list[dict[str, Any]],
    season: int | None,
    week: int | None,
    sort: str,
    request: Request,
) -> dict[str, Any]:
    """The three ``game_grid`` keys both This Week routes add: games, slate_groups, headliner.

    Shared by ``this_week_page`` and ``fragments.games_fragment`` so a week change through HTMX
    renders exactly what a full navigation renders. Every game is passed through
    ``api.presentation.decorate_game`` (team colours, nicknames, kickoff and window labels) as a
    NEW dict -- the source rows may be the DataService TTLCache's own -- and carries
    ``bet_targets``, the bet types it has a live bet for this week, so its card can say so. Each
    headliner bet carries ``game``, its decorated game row (or None when the week's predictions
    do not include it), for the kickoff label and team colours.
    """
    headliner = _build_headliner(service, season, week, request)
    targets_by_game: dict[str, list[str]] = {}
    for bet in headliner["bets"]:
        targets_by_game.setdefault(bet["game_id"], []).append(bet["target"])

    decorated: list[dict[str, Any]] = []
    for game in games:
        new_game = decorate_game(game)
        new_game["bet_targets"] = targets_by_game.get(new_game["game_id"], [])
        decorated.append(new_game)

    games_by_id = {game["game_id"]: game for game in decorated}
    headliner["bets"] = [
        {**bet, "game": games_by_id.get(bet["game_id"])} for bet in headliner["bets"]
    ]
    return {
        "games": decorated,
        "slate_groups": None
        if sort in _NON_TIME_SORTS
        else group_games_by_window(decorated),
        "headliner": headliner,
    }
```

- [ ] **Step 4: Wire `this_week_page` to the shared context**

In `this_week_page`, old (lines 767-786):

```python
    games = service.get_predictions(season=season, week=week, sort=sort)

    # Compute wp_correct on a fresh list so the DataService TTLCache source
    # is never mutated (plan 15-02 review item #4).
    games = _annotate_wp_correct(games)

    context = {
        "request": request,
        "games": games,
        "available_weeks": available_weeks,
        "available_seasons": available_seasons,
        "current_week": week,
        "current_season": season,
        "current_sort": sort,
        "current_path": "/",
        "cache_meta": cache_meta,
        "week_summary": _compute_week_summary(games),
        # One block: the week summary and the game grid, scoped to the games actually shown.
        "old_rule_scope": DataService.old_rule_scope(_rows_seasons(games)),
    }
```

New:

```python
    games = service.get_predictions(season=season, week=week, sort=sort)

    # Compute wp_correct on a fresh list so the DataService TTLCache source
    # is never mutated (plan 15-02 review item #4).
    games = _annotate_wp_correct(games)
    grid = _this_week_grid_context(service, games, season, week, sort, request)

    context = {
        "request": request,
        "games": grid["games"],
        "slate_groups": grid["slate_groups"],
        "headliner": grid["headliner"],
        "available_weeks": available_weeks,
        "available_seasons": available_seasons,
        "current_week": week,
        "current_season": season,
        "current_sort": sort,
        "current_path": "/",
        "cache_meta": cache_meta,
        "week_summary": _compute_week_summary(games),
        # One block: the week summary, the headliner bets and the game grid, scoped to the games
        # and the bets actually shown -- a week whose predictions are absent can still show bets.
        "old_rule_scope": DataService.old_rule_scope(
            _rows_seasons(games) + _rows_seasons(grid["headliner"]["bets"])
        ),
    }
```

- [ ] **Step 5: Wire `games_fragment` to the same context**

In `api/routes/fragments.py`, old import list (lines 18-30):

```python
from api.routes.pages import (
    _DEFAULT_BETTING_SCOPE,
    PAGE_CACHE_CONTROL,
    _annotate_wp_correct,
    _build_betting_context,
    _build_season_context,
    _compute_week_summary,
    _normalize_betting_scope,
    _normalize_season,
    _parse_int_param,
    _pivot_season_metrics,
    _rows_seasons,
)
```

New:

```python
from api.routes.pages import (
    _DEFAULT_BETTING_SCOPE,
    PAGE_CACHE_CONTROL,
    _annotate_wp_correct,
    _build_betting_context,
    _build_season_context,
    _compute_week_summary,
    _normalize_betting_scope,
    _normalize_season,
    _parse_int_param,
    _pivot_season_metrics,
    _rows_seasons,
    _this_week_grid_context,
)
```

Old body (lines 65-79):

```python
    games = service.get_predictions(season=season_int, week=week_int, sort=sort)

    # Compute wp_correct on a fresh list so the DataService TTLCache source
    # is never mutated (plan 15-02 review item #4).
    games = _annotate_wp_correct(games)

    context = {
        "games": games,
        "current_week": week_int,
        "current_season": season_int,
        "current_sort": sort,
        "week_summary": _compute_week_summary(games),
        # The same scope this_week_page builds for the game_grid block (R16 / D33.2-07).
        "old_rule_scope": DataService.old_rule_scope(_rows_seasons(games)),
    }
```

New:

```python
    games = service.get_predictions(season=season_int, week=week_int, sort=sort)

    # Compute wp_correct on a fresh list so the DataService TTLCache source
    # is never mutated (plan 15-02 review item #4).
    games = _annotate_wp_correct(games)
    # The SAME games / groups / headliner this_week_page builds, so a week change through HTMX
    # renders exactly what a full navigation renders.
    grid = _this_week_grid_context(service, games, season_int, week_int, sort, request)

    context = {
        "games": grid["games"],
        "slate_groups": grid["slate_groups"],
        "headliner": grid["headliner"],
        "current_week": week_int,
        "current_season": season_int,
        "current_sort": sort,
        "week_summary": _compute_week_summary(games),
        # The same scope this_week_page builds for the game_grid block (R16 / D33.2-07).
        "old_rule_scope": DataService.old_rule_scope(
            _rows_seasons(games) + _rows_seasons(grid["headliner"]["bets"])
        ),
    }
```

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `uv run pytest tests/api/test_this_week_headliner.py -v`
Expected: 14 passed (6 parity + 6 precedence + 1 mismatch + 1 grid).

- [ ] **Step 7: Run the This Week regressions (the templates have not changed yet, so they must still pass)**

```bash
uv run pytest tests/api/test_pages.py -q
uv run pytest tests/api/test_fragments.py -q
uv run pytest tests/api/test_cache_headers.py -q
uv run pytest tests/api/test_page_labels_routes.py -q
```

Expected: every test that passed before this task still passes. Task 3's nav change may already have deliberately updated some of these.

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check api/routes/pages.py api/routes/fragments.py tests/api/test_this_week_headliner.py && uv run ruff format api/routes/pages.py api/routes/fragments.py tests/api/test_this_week_headliner.py && uv run pyright api/routes/pages.py api/routes/fragments.py tests/api/test_this_week_headliner.py`
Expected: no errors.

- [ ] **Step 9: Commit**

```bash
git add api/routes/pages.py api/routes/fragments.py tests/api/test_this_week_headliner.py
git commit -m "$(cat <<'EOF'
feat(redesign): This Week context carries decorated games, TV-window groups and the /bets headliner state

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: The Broadcast game card

**Files:**
- Modify: `web/templates/components/_prediction_values.html` (add `team_pct`; restyle `absent`)
- Replace: `web/templates/components/_game_card.html` (whole file)
- Create: `tests/unit/test_game_card_broadcast.py`
- Modify: `tests/unit/test_current_week_rows_reach_the_site.py:215` (and add `import re`)
- Modify: `tests/api/test_pages.py` -- the function `test_this_week_page_confidence_badges` (as Task 4 left it), and the two functions `test_the_landing_page_still_renders_all_three_edge_band_labels` and `test_the_landing_page_renders_the_band_it_was_served_and_never_rederives_one`
- Rebuild: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes (Task 1): classes `panel`, `label`, `num`, `border-accent`, `bg-accent`, `text-ink`, `text-fg`, `text-muted`, `text-dim`, `bg-panel-2`, `divide-line`, `border-line`, `edge-chip-soft`.
- Consumes (Task 3): `bc.score_row(abbr, name, value, bg, fg, fav, dog)`, `bc.edge_chip(text, soft)`.
- Consumes (Task 6): `game.bet_targets`, `game.away_color|away_fg|home_color|home_fg|away_name|home_name|kickoff_label`, `game.wp_correct`.
- Produces:
  - `pv.team_pct(prob, side)`, where `side` is `"home"` or `"away"`.
  - Card hooks `data-edge="wp|ats|ou"` and `data-on-bet-list`.
  - Card text kept for tests: "Win Prob", "Spread", "Total", "View Details", "Predicted: X NN% WP", "No line", "No pick".

- [ ] **Step 1: Write the failing card tests**

Create `tests/unit/test_game_card_broadcast.py`:

```python
"""The Broadcast game card reads cleanly in every state (Broadcast redesign, Task 7).

The card is rendered here exactly as tests/unit/test_current_week_rows_reach_the_site.py renders
it: a plain jinja2 Environment with no app globals or filters, handed only ``game=``. That is the
contract the card keeps so a bare prediction row always renders.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from api.cache import PREDICTIONS_TABLE_COLUMNS
from api.presentation import decorate_game

TEMPLATES = Path(__file__).resolve().parents[2] / "web" / "templates"

# A green or red Tailwind colour utility in a class attribute. On this site those two hues mean a
# REALISED result only; an edge, a band or a flag must never carry one.
_HUE = re.compile(r"\b(?:text|bg|border)-(green|red)-\d{2,3}")


def _render(game: dict[str, Any]) -> str:
    return (
        Environment(loader=FileSystemLoader(TEMPLATES))
        .get_template("components/_game_card.html")
        .render(game=game)
    )


def _hues(html: str) -> list[str]:
    return [
        match.group(1)
        for classes in re.findall(r'class="([^"]*)"', html)
        for match in _HUE.finditer(classes)
    ]


def _row(**overrides: Any) -> dict[str, Any]:
    """KC at MIA, Week 3 2026, model numbers only unless *overrides* add more."""
    row = dict.fromkeys(PREDICTIONS_TABLE_COLUMNS)
    row.update(
        game_id="2026_W03_KC@MIA",
        season=2026,
        week=3,
        game_date=datetime(2026, 9, 27, 13, 0),
        home_team="MIA",
        away_team="KC",
        status="scheduled",
        wp_prob=0.412,
        ats_prediction=-2.5,
        ou_prediction=41.3,
    )
    row.update(overrides)
    return row


_MARKET = {
    "market_wp": 0.449,
    "market_spread": -1.5,
    "market_total": 44.5,
    "wp_edge": -0.037,
    "ats_edge": -1.0,
    "ou_edge": -0.0719,
}


def test_a_week_with_no_market_lines_reads_cleanly() -> None:
    """Review Focus 1: today's real Week 3 -- no line anywhere -- shows "No line" and nothing else.

    No empty chip, no "None", no "nan", no pick words without a line to pick against.
    """
    card = _render(decorate_game(_row()))

    assert card.count("No line") == 3, "each of the three market slots must say No line"
    assert "data-edge" not in card, "an edge chip rendered for a game with no edge"
    assert "None" not in card
    assert "nan" not in card.lower()
    for pick_word in ("covers", "Over", "Under", "No pick"):
        assert pick_word not in card, f"{pick_word!r} rendered with no line to pick against"
    assert " vs " not in card
    assert "data-on-bet-list" not in card
    assert "KC 58.8%" in card, "the model's own win probability is missing"
    assert "KC by 2.5" in card and "41.3" in card


def test_model_and_market_numbers_pick_words_and_edges_render() -> None:
    card = _render(decorate_game(_row(**_MARKET)))

    assert "KC 58.8%" in card and "KC 55.1%" in card
    assert "KC by 2.5" in card and "KC by 1.5" in card
    assert "KC covers" in card, "model margin below the home-margin line backs the away side"
    assert "Under" in card and "44.5" in card
    for target, text in (("wp", r"-3\.7%"), ("ats", r"-1\.0 pts"), ("ou", r"-7\.2%")):
        assert re.search(rf'data-edge="{target}".*?{text}', card, re.DOTALL), (
            f"the {target} edge chip does not show {text}"
        )
    assert _hues(card) == [], "a pre-game number was drawn in green or red"


def test_a_bet_game_carries_the_flag_and_a_solid_chip_for_its_bet_type() -> None:
    game = {**decorate_game(_row(**_MARKET)), "bet_targets": ["ou"]}
    card = _render(game)
    plain_card = _render(decorate_game(_row(**_MARKET)))

    assert "data-on-bet-list" in card
    # The TOP RULE's colour, not any "border-accent": every card also carries hover:border-accent,
    # so a bare substring check would pass for a card with no bet. The plain card is the control.
    top_rule = "border-t-[3px] border-accent "
    assert top_rule in card.split(">", 1)[0], "the bet card's top edge is not the accent"
    assert top_rule not in plain_card.split(">", 1)[0], "a card with no bet has the accent top edge"
    # Three edges, one of them the bet's own: two outlined chips and one solid.
    assert card.count("edge-chip-soft") == 2


def test_a_final_game_shows_the_score_the_grade_and_the_prediction() -> None:
    game = {
        **decorate_game(
            _row(
                status="completed",
                away_score=20,
                home_score=27,
                wp_prob=0.62,
                **_MARKET,
            )
        ),
        "wp_correct": True,
    }
    card = _render(game)
    flat = " ".join(card.split())

    assert "Final" in card
    assert "Correct" in card and "Incorrect" not in card
    assert "Predicted: MIA 62% WP" in flat
    assert set(_hues(card)) == {"green"}, "only the realised result may be green"


def test_a_tie_is_graded_neither_correct_nor_incorrect() -> None:
    game = {
        **decorate_game(
            _row(status="completed", away_score=20, home_score=20, wp_prob=0.40)
        ),
        "wp_correct": None,
    }
    card = _render(game)

    assert "Tie" in card
    assert "Correct" not in card and "Incorrect" not in card
    assert _hues(card) == []


def test_the_card_renders_from_a_bare_row_with_no_decoration() -> None:
    """A row nobody decorated still renders: every new field has a default."""
    card = _render(_row())

    assert "game-card" in card
    assert "KC 58.8%" in card
    assert "View Details" in card
```

- [ ] **Step 2: Run the card tests to verify they fail**

Run: `uv run pytest tests/unit/test_game_card_broadcast.py -v`
Expected: FAIL. `test_a_week_with_no_market_lines_reads_cleanly` fails on `"KC 58.8%" in card` or on the `No line` count, and `test_a_bet_game_...` fails on `data-on-bet-list`, because the old card has neither the score bug nor the hooks.

- [ ] **Step 3: Add `team_pct` and restyle `absent` in `components/_prediction_values.html`**

Old:

```jinja
{% macro absent(text) -%}
<span class="text-gray-400 italic">{{ text }}</span>
{%- endmacro %}
```

New:

```jinja
{% macro absent(text) -%}
<span class="text-dim italic">{{ text }}</span>
{%- endmacro %}
```

Then append at the end of the file:

```jinja

{# "58.8%": ONE team's own chance of winning, for the score bug that lists both teams. The stored
   probability is the HOME team's (DEF-31-01), so the away side's is its complement. A missing
   probability is "-", never a zero. #}
{% macro team_pct(prob, side) -%}
{%- if prob is none -%}{{ absent("-") }}
{%- elif side == "home" -%}{{ "%.1f"|format(prob * 100) }}%
{%- else -%}{{ "%.1f"|format((1 - prob) * 100) }}%
{%- endif -%}
{%- endmacro %}
```

- [ ] **Step 4: Replace `components/_game_card.html` with the score-bug card**

Whole file:

```jinja
{# One game card, drawn as a broadcast score bug (Broadcast redesign, spec 7.1). Pass: game.

   The routes hand the row through api.presentation.decorate_game (team colours, nicknames, the
   kickoff label) and add ``wp_correct`` (pages._annotate_wp_correct) and ``bet_targets`` (the
   live bet types this game carries, pages._this_week_grid_context). EVERY one of those is read
   with a default: the card must also render from a bare prediction row in a plain Jinja
   environment with no app globals or filters, which is exactly how
   tests/unit/test_current_week_rows_reach_the_site.py renders it.

   Each row shows OUR MODEL'S own number with the market's beneath it, for every game -- a bet or
   not, a market line or not (owner requirement, 2026-09-23). An edge chip appears only where an
   edge exists: the edge is stored NULL where none exists (utils.edge_tier, 33.2 review C2
   CR-04), so a game with no line shows no chip. Chips are yellow emphasis, never green or red:
   an edge is a pre-game number, and on this site green and red mean a realised result only. The
   edge band (confidence) is not on the card -- the chip already shows the edge the band
   summarises; the band is on game detail, monochrome. #}
{% import "components/_prediction_values.html" as pv %}
{% import "components/_broadcast.html" as bc %}
{%- set home = game.home_team -%}
{%- set away = game.away_team -%}
{%- set bet_targets = game.bet_targets|default([]) -%}
{%- set final = game.status == "completed" and game.home_score is not none and game.away_score is not none -%}
<article class="game-card panel relative border-t-[3px] {{ 'border-accent' if bet_targets else 'border-white/10' }} transition duration-200 hover:-translate-y-0.5 hover:border-accent motion-reduce:transition-none motion-reduce:hover:translate-y-0"{% if bet_targets %} data-on-bet-list{% endif %}>
  <a href="/games/{{ game.game_id }}" class="block">
    {# Header: the kickoff (or FINAL with the graded result) and the flags. #}
    <div class="flex items-center justify-between gap-2">
      <span class="label">{% if final %}Final{% else %}{{ game.kickoff_label|default(game.game_date if game.game_date is not none else "Time TBD") }}{% endif %}</span>
      <span class="flex items-center gap-1.5">
        {%- if final and game.wp_correct is defined -%}
          {# Graded by THE classifier the week banner and /season use (_wp_outcome through
             _annotate_wp_correct): a tie or a game with no WP pick is neither correct nor
             incorrect (33.2 review C2 WR-04). Shown only when the route annotated the grade. #}
          {%- if game.wp_correct is sameas true -%}
          <span class="label px-1.5 bg-green-500/15 text-green-400">Correct</span>
          {%- elif game.wp_correct is sameas false -%}
          <span class="label px-1.5 bg-red-500/15 text-red-400">Incorrect</span>
          {%- else -%}
          <span class="label px-1.5 bg-panel-2 text-fg">{{ "Tie" if game.home_score == game.away_score else "No pick" }}</span>
          {%- endif -%}
        {%- endif -%}
        {%- if bet_targets -%}
          <span class="label px-1.5 bg-accent text-ink">Bet</span>
        {%- endif -%}
        {%- if game.status not in ("scheduled", "completed") -%}
          {% with status=game.status %}{% include "components/_status_badge.html" %}{% endwith %}
        {%- endif -%}
      </span>
    </div>

    {# The score bug: away over home. Before kickoff each row shows that team's own win chance,
       the favourite bright and the underdog dimmed; after the game, the final score with the
       winner bright. An even 0.5 favours neither. #}
    <div class="score-bug mt-2">
      {%- if final -%}
        {{ bc.score_row(away, game.away_name|default(away), game.away_score|string, game.away_color|default("#1D2436"), game.away_fg|default("#FFFFFF"), fav=game.away_score > game.home_score, dog=game.away_score < game.home_score) }}
        {{ bc.score_row(home, game.home_name|default(home), game.home_score|string, game.home_color|default("#1D2436"), game.home_fg|default("#FFFFFF"), fav=game.home_score > game.away_score, dog=game.home_score < game.away_score) }}
      {%- else -%}
        {%- set wp = game.wp_prob -%}
        {{ bc.score_row(away, game.away_name|default(away), pv.team_pct(wp, "away"), game.away_color|default("#1D2436"), game.away_fg|default("#FFFFFF"), fav=wp is not none and wp < 0.5, dog=wp is not none and wp > 0.5) }}
        {{ bc.score_row(home, game.home_name|default(home), pv.team_pct(wp, "home"), game.home_color|default("#1D2436"), game.home_fg|default("#FFFFFF"), fav=wp is not none and wp > 0.5, dog=wp is not none and wp < 0.5) }}
      {%- endif -%}
    </div>

    {# Win / Spread / Total. Both margins are home margins, POSITIVE when the home team is
       favoured (DEF-31-01), so a model margin ABOVE the line means the home team covers -- the
       side backtest/ats_ev_chain.py prices. Equal numbers are NO PICK, exactly as
       season_metrics._ats_outcome / _ou_outcome grade them (33.2 review C2 IN-05). #}
    <dl class="mt-3 divide-y divide-line text-xs">
      <div class="grid grid-cols-[4.25rem_1fr_auto] items-center gap-2 py-1.5">
        <dt class="label">Win Prob</dt>
        <dd>
          <span class="num text-fg">{{ pv.win_prob(game.wp_prob, home, away, "-") }}</span>
          <span class="block text-muted">Market {{ pv.win_prob(game.market_wp, home, away, pv.market_wp_missing(game)) }}</span>
        </dd>
        <dd>{% if game.wp_edge is not none %}<span data-edge="wp">{{ bc.edge_chip("%+.1f"|format(game.wp_edge * 100) ~ "%", soft="wp" not in bet_targets) }}</span>{% endif %}</dd>
      </div>
      <div class="grid grid-cols-[4.25rem_1fr_auto] items-center gap-2 py-1.5">
        <dt class="label">Spread</dt>
        <dd>
          <span class="num text-fg">{{ pv.margin(game.ats_prediction, home, away, "-") }}</span>
          <span class="block text-muted">Market {{ pv.margin(game.market_spread, home, away) }}{% if game.ats_prediction is not none and game.market_spread is not none %} &middot; <span class="font-semibold text-accent">{% if game.ats_prediction > game.market_spread %}{{ home }} covers{% elif game.ats_prediction < game.market_spread %}{{ away }} covers{% else %}No pick{% endif %}</span>{% endif %}</span>
        </dd>
        <dd>{% if game.ats_edge is not none %}<span data-edge="ats">{{ bc.edge_chip("%+.1f"|format(game.ats_edge) ~ " pts", soft="ats" not in bet_targets) }}</span>{% endif %}</dd>
      </div>
      <div class="grid grid-cols-[4.25rem_1fr_auto] items-center gap-2 py-1.5">
        <dt class="label">Total</dt>
        <dd>
          <span class="num text-fg">{{ pv.total(game.ou_prediction, "-") }}</span>
          <span class="block text-muted">Market {{ pv.total(game.market_total) }}{% if game.ou_prediction is not none and game.market_total is not none %} &middot; <span class="font-semibold text-accent">{% if game.ou_prediction > game.market_total %}Over{% elif game.ou_prediction < game.market_total %}Under{% else %}No pick{% endif %}</span>{% endif %}</span>
        </dd>
        <dd>{% if game.ou_edge is not none %}<span data-edge="ou">{{ bc.edge_chip("%+.1f"|format(game.ou_edge * 100) ~ "%", soft="ou" not in bet_targets) }}</span>{% endif %}</dd>
      </div>
    </dl>

    {# Footer: what the model picked (once the game is graded) and the drill-down affordance. The
       "Predicted" team is named only ABOVE 0.5 for the home side -- the comparison pv.win_prob
       and season_metrics._wp_outcome use, so an exact 0.5 names the same team everywhere. #}
    <div class="mt-2 flex items-center justify-between gap-2 border-t border-line pt-2 text-xs">
      <span class="text-muted">
        {%- if final and game.wp_prob is not none -%}
        Predicted: {{ home if game.wp_prob > 0.5 else away }} {{ "%.0f"|format((game.wp_prob if game.wp_prob > 0.5 else 1 - game.wp_prob) * 100) }}% WP
        {%- endif -%}
      </span>
      <span class="font-display font-bold uppercase tracking-wide text-accent">View Details &rsaquo;</span>
    </div>
  </a>
</article>
```

- [ ] **Step 5: Update the tests the card change deliberately breaks**

In `tests/unit/test_current_week_rows_reach_the_site.py`, add `import re` after `from __future__ import annotations` (with the stdlib imports), then old (line 215):

```python
    assert "Edge: +3.0%" in card
```

New:

```python
    # The edge now renders as a chip beside the two numbers (Broadcast redesign); the claim is
    # unchanged: the chip shows the +3.0 points of win probability between them.
    assert re.search(r'data-edge="wp".*?\+3\.0%', card, re.DOTALL), (
        "the WP edge chip does not show the +3.0% the model and market numbers beside it imply"
    )
```

In `tests/api/test_pages.py`, replace the whole function `test_this_week_page_confidence_badges` (from its `def` line through its last line) as Task 4 Step 9 left it. Old:

```python
def test_this_week_page_confidence_badges(test_client: TestClient):
    """D-08: Confidence labels render one monochrome band each (Broadcast redesign)."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    # Sample data contains high, medium, and low confidence values. The labels are grey-scale:
    # green and red now mean a realised result only, and a pre-game band is not one.
    assert 'class="band band-high" data-confidence-band="high"' in html
    assert 'class="band band-medium" data-confidence-band="medium"' in html
    assert 'class="band band-low" data-confidence-band="low"' in html
    # Only amber is checked here: until Task 7 replaces the card, its Correct/Incorrect result
    # badge still (rightly) renders bg-green-100 / bg-red-100 for the fixture's completed games.
    # Task 7 adds the green/red check once that badge is gone.
    assert "bg-amber-100" not in html
```

If `ruff format` in Task 4 rewrapped any of those lines, match the function by name and replace it whole. New (this is where the green/red check Task 4 deferred lands, now that the card's result badge uses the `-400`/`-500` outcome classes):

```python
def test_this_week_cards_carry_no_confidence_pill(test_client: TestClient):
    """Broadcast redesign: the card's edge chip already shows the edge the band summarises.

    The band moved to game detail, monochrome. Its three old colour classes must not survive on
    /, where green and red now mean a realised result only.
    """
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    for badge_class in ("bg-green-100", "bg-amber-100", "bg-red-100"):
        assert badge_class not in html
    assert "data-band" not in html
```

Replace the two band tests -- old: the whole of `test_the_landing_page_still_renders_all_three_edge_band_labels` and `test_the_landing_page_renders_the_band_it_was_served_and_never_rederives_one`, from `def test_the_landing_page_still_renders_all_three_edge_band_labels(` through the closing `)` of the final `assert Counter(rendered) == Counter(served), (...)`. New:

```python
def test_the_landing_page_renders_no_edge_band_and_so_cannot_rederive_one(
    test_client: TestClient,
) -> None:
    """The band left the cards in the Broadcast redesign (31-17 / D31-23 history above).

    The band now renders only on game detail, and the served-equals-rendered guard belongs there
    with it. On / the claim is now the stronger one: no band label is rendered at all, so none can
    be re-derived.
    """
    from utils.edge_tier import EDGE_TIER_LABELS

    html = test_client.get("/").text
    grid = html[html.index('id="game-grid"') :]
    for label in EDGE_TIER_LABELS:
        assert f">{label.title()}<" not in grid, (
            f"a {label!r} band label rendered on /, which no longer shows bands"
        )
```

Then remove `from collections import Counter` from the imports at the top of `tests/api/test_pages.py` if nothing else in the file uses it (run `grep -n "Counter" tests/api/test_pages.py`. If the only remaining hit is the import line, delete it).

- [ ] **Step 6: Rebuild the CSS**

Run: `./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify`
Expected: `Done in ...` with no error.

- [ ] **Step 7: Run the card tests and every test that renders the card**

```bash
uv run pytest tests/unit/test_game_card_broadcast.py -v
uv run pytest tests/unit/test_current_week_rows_reach_the_site.py -v
uv run pytest tests/api/test_fragments.py -v
```

Expected: all pass.

Run: `uv run pytest "tests/api/test_pages.py::test_this_week_page" "tests/api/test_pages.py::test_this_week_page_has_game_cards" "tests/api/test_pages.py::test_this_week_page_responsive_grid" "tests/api/test_pages.py::test_this_week_cards_carry_no_confidence_pill" "tests/api/test_pages.py::test_the_landing_page_renders_no_edge_band_and_so_cannot_rederive_one" "tests/api/test_pages.py::test_the_landing_page_sort_by_band_is_unchanged" "tests/api/test_pages.py::test_this_week_htmx_returns_fragment" -v`
Expected: all pass.

- [ ] **Step 8: Lint the touched Python**

Run: `uv run ruff check tests/unit/test_game_card_broadcast.py tests/unit/test_current_week_rows_reach_the_site.py tests/api/test_pages.py && uv run ruff format tests/unit/test_game_card_broadcast.py tests/unit/test_current_week_rows_reach_the_site.py tests/api/test_pages.py && uv run pyright tests/unit/test_game_card_broadcast.py tests/unit/test_current_week_rows_reach_the_site.py tests/api/test_pages.py`
Expected: no ruff error; pyright reports no error on a line this task wrote (an error elsewhere in an existing test file predates the branch -- note it in the task report and leave it).

- [ ] **Step 9: Commit**

```bash
git add web/templates/components/_prediction_values.html web/templates/components/_game_card.html web/static/css/tailwind-compiled.css tests/unit/test_game_card_broadcast.py tests/unit/test_current_week_rows_reach_the_site.py tests/api/test_pages.py
git commit -m "$(cat <<'EOF'
feat(redesign): Broadcast score-bug game card

Model beside market for win, spread and total; yellow edge chips only where an edge exists; the
bet flag from the week's live list; the realised grade alone in green/red. The edge band moves off
the card (game detail carries it, monochrome).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: This Week layout -- headliner bets and the TV-window slate

**Files:**
- Create: `web/templates/components/_bet_pick.html`
- Modify: `web/templates/pages/bets.html` (lines 202-210, 247-260, 263) -- the pick macros move out, verbatim
- Create: `web/templates/components/_bet_headliners.html`
- Replace: `web/templates/pages/this_week.html` (whole file)
- Modify: `tests/api/test_this_week_headliner.py` (append the `_insert_predictions` helper, which first has a caller here, and the layout tests)
- Modify: `tests/api/test_pages.py` (`test_this_week_page_responsive_grid`: the slate's grid classes change on purpose, spec 11)
- Rebuild: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes (Task 2): `decorate_game`, `group_games_by_window`, the Jinja global `team_colors(abbr) -> TeamColors(bg, fg)`.
- Consumes (Task 3): `bc.section_head(label, meta, ghost, anchor)`.
- Consumes (Task 4): `components/_not_advice_banner.html` (condensed), `components/_ev_band_badge.html` (monochrome, takes `band`).
- Consumes (Task 5): `components/_week_selector.html`, `_sort_controls.html`, `_export_buttons.html`, `_week_summary.html`, `_empty_state.html`, all included unchanged.
- Consumes (Task 6): `headliner`, `slate_groups`, `games[*].bet_targets`, `headliner.bets[*].game`.
- Produces:
  - `components/_bet_pick.html` with `side_line(bet, fmt)`, `picked_team(bet)` and `pick_label(bet)`. Part C's Tasks 11-12 import it as `bp`.
  - `components/_bet_headliners.html`, with root `<section id="headliners" data-headliner-state="...">`.
  - The `data-window="<label>"` section wrapper on the slate.

- [ ] **Step 1: Write the failing layout tests**

Append to `tests/api/test_this_week_headliner.py`:

```python
# ---------------------------------------------------------------------------
# The rendered page: headliners first, then the slate by TV window (Task 8)
# ---------------------------------------------------------------------------

_WEEK_GAMES = [
    _prediction(_GAME, datetime(2023, 9, 7, 20, 20)),  # Thursday night
    _prediction(_SECOND_GAME, datetime(2023, 9, 10, 13, 0)),  # Sunday 1:00
]


def _insert_predictions(db_path: Path, rows: list[dict[str, Any]]) -> None:
    """Add prediction rows to a cache built by _build_state_cache, so / has games to show."""
    conn = duckdb.connect(str(db_path))
    try:
        columns = ", ".join(PREDICTIONS_TABLE_COLUMNS)
        placeholders = ", ".join("?" for _ in PREDICTIONS_TABLE_COLUMNS)
        conn.executemany(
            f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
            [[row[c] for c in PREDICTIONS_TABLE_COLUMNS] for row in rows],
        )
    finally:
        conn.close()


def _page(tmp_path: Path, state: str, path: str, *, name: str = "page") -> str:
    """Build a fresh cache for *state*, serve it, and return the body of GET *path*.

    A test that renders twice passes a distinct *name*: building onto a file that already holds
    the week's predictions would insert the same game ids again and hit the predictions table's
    primary key.
    """
    db_path = tmp_path / f"{name}_{state}.duckdb"
    _build_state_cache(db_path, state)
    _insert_predictions(db_path, _WEEK_GAMES)
    with _serving(db_path) as (client, _conn):
        response = client.get(path)
    assert response.status_code == 200
    return response.text


def test_this_week_leads_with_the_weeks_bets(tmp_path: Path) -> None:
    html = _page(tmp_path, "bets", f"/?season={_SEASON}&week={_WEEK}")

    assert 'data-headliner-state="bets"' in html
    headliners = html[html.index('id="headliners"') : html.index("data-window=")]
    assert f'data-headliner-bet="{_GAME}:ou"' in headliners
    assert "Under 47.5" in headliners, "the pick is not worded the way /bets words it"
    assert "+6.25%" in headliners and "1.50u" in headliners
    assert html.index('id="headliners"') < html.index("game-card"), (
        "the bets must come before the slate"
    )
    assert html.count("data-on-bet-list") == 1, "only the bet's own game carries the flag"


@pytest.mark.parametrize(
    "state", [s for s in HEADLINER_STATES if s != "bets"]
)
def test_every_other_state_is_one_line_pointing_to_bets(
    tmp_path: Path, state: str
) -> None:
    html = _page(tmp_path, state, f"/?season={_SEASON}&week={_WEEK}")

    assert f'data-headliner-state="{state}"' in html
    assert "data-headliner-note" in html
    assert "data-headliner-bet" not in html
    assert 'href="/bets' in html[html.index('id="headliners"') :]


def test_a_week_change_re_renders_the_headliners_with_the_slate(tmp_path: Path) -> None:
    html = _page(tmp_path, "bets", f"/fragments/games?season={_SEASON}&week={_WEEK}")

    assert 'data-headliner-state="bets"' in html
    assert "<nav" not in html and "<html" not in html


def test_the_default_sort_groups_by_tv_window_and_a_non_time_sort_does_not(
    tmp_path: Path,
) -> None:
    grouped = _page(tmp_path, "bets", f"/?season={_SEASON}&week={_WEEK}", name="grouped")
    assert grouped.count("data-window=") == 2, "Thursday night and Sunday 1:00 are two windows"

    by_edge = _page(
        tmp_path, "bets", f"/?season={_SEASON}&week={_WEEK}&sort=edge", name="by_edge"
    )
    assert "data-window=" not in by_edge
    assert by_edge.count("game-card") == grouped.count("game-card")


def test_the_page_keeps_one_old_rule_label_and_the_not_advice_note(
    tmp_path: Path,
) -> None:
    """2023 is an old-rule season: the headliners sit in the game_grid block's ONE label."""
    html = _page(tmp_path, "bets", f"/?season={_SEASON}&week={_WEEK}")

    assert html.count("data-old-rule-label") == 1
    assert "Not wagering advice" in html


def test_a_game_with_no_kickoff_time_renders_under_its_own_tag() -> None:
    """Review Focus 3: a game whose kickoff is unknown still renders, under "Time TBD"."""
    from api.dependencies import templates
    from api.presentation import decorate_game, group_games_by_window

    game = {**decorate_game(_prediction(_GAME, None)), "bet_targets": []}
    context = {
        "request": _StubRequest(),
        "cache_meta": {},
        "games": [game],
        "slate_groups": group_games_by_window([game]),
        "available_weeks": [{"season": _SEASON, "week": _WEEK}],
        "available_seasons": [_SEASON],
        "current_week": _WEEK,
        "current_season": _SEASON,
        "current_sort": "time",
        "current_path": "/",
        "week_summary": {},
        "old_rule_scope": DataService.old_rule_scope([]),
    }
    html = templates.env.get_template("pages/this_week.html").render(context)

    assert 'data-window="Time TBD"' in html
    assert html.count("game-card") == 1
    assert "Time TBD" in html.split('data-window="Time TBD"', 1)[1]
```

- [ ] **Step 2: Run the new tests to verify they fail**

Run: `uv run pytest tests/api/test_this_week_headliner.py -v -k "leads or other_state or week_change or default_sort or old_rule or kickoff"`
Expected: FAIL. The old template has no `data-headliner-state` and no `data-window`.

- [ ] **Step 3: Create `components/_bet_pick.html` (moved verbatim from `pages/bets.html`)**

```jinja
{# The pick a bet names, in the words and the sign a sportsbook would use. ONE source, shared by
   pages/bets.html and the This Week headliners (components/_bet_headliners.html), so the two
   pages cannot word -- or sign -- the same bet differently. Moved here VERBATIM from
   pages/bets.html (Broadcast redesign, Task 8); nothing about the rule changed.

   A spread bet's line FROM THE PICKED SIDE, the way a sportsbook quotes it (33.2 review C2
   CR-03). The stored ``line`` is the HOME MARGIN, positive when the home team is favoured
   (DEF-31-01), so the home side's own line is its negation and the away side's is the margin as
   stored: 2025_W01_CIN@CLE home_cover at -5.5 is CLE +5.5, and 2025_W01_MIA@IND home_cover at
   +1.5 is IND -1.5. Printing the margin raw put the opposite sign on every home pick.

   Every macro here is whitespace-trimmed so a call inside a table cell renders exactly the text
   the inline code it replaced rendered. #}

{% macro side_line(bet, fmt) -%}
  {%- set value = -bet.line if bet.bet_side == "home_cover" else bet.line -%}
  {%- if value == 0 -%}PK{%- else -%}{{ fmt|format(value) }}{%- endif -%}
{%- endmacro %}

{# The team a bet backs, never the raw side code. The two teams come from the game id's
   AWAY@HOME matchup; an id without one falls back to the words. #}
{% macro picked_team(bet) -%}
  {%- set matchup = bet.game_id.split('_')[-1] -%}
  {%- set away_team, home_team = matchup.split('@') if '@' in matchup else ("Away", "Home") -%}
  {{- home_team if bet.bet_side in ("home", "home_cover") else away_team -}}
{%- endmacro %}

{# "Under 47.5", "CLE +5.5", "KC": the whole pick as one phrase. #}
{% macro pick_label(bet) -%}
  {%- if bet.target == "ou" -%}
    {{ bet.bet_side|capitalize }} {{ "%g"|format(bet.line) if bet.line is not none else "" }}
  {%- elif bet.target == "ats" and bet.line is not none -%}
    {{ picked_team(bet) }} {{ side_line(bet, "%+g") }}
  {%- else -%}
    {{ picked_team(bet) }}
  {%- endif -%}
{%- endmacro %}
```

- [ ] **Step 4: Point `pages/bets.html` at the shared partial**

Old (lines 202-210):

```jinja
  {# A spread bet's line FROM THE PICKED SIDE, the way a sportsbook quotes it (33.2 review C2
     CR-03). The stored ``line`` is the HOME MARGIN, positive when the home team is favoured
     (DEF-31-01), so the home side's own line is its negation and the away side's is the margin
     as stored: 2025_W01_CIN@CLE home_cover at -5.5 is CLE +5.5, and 2025_W01_MIA@IND home_cover
     at +1.5 is IND -1.5. Printing the margin raw put the opposite sign on every home pick. #}
  {% macro side_line(bet, fmt) -%}
    {%- set value = -bet.line if bet.bet_side == "home_cover" else bet.line -%}
    {%- if value == 0 -%}PK{%- else -%}{{ fmt|format(value) }}{%- endif -%}
  {%- endmacro %}
```

New:

```jinja
  {# The pick, and a spread bet's line FROM THE PICKED SIDE the way a sportsbook quotes it (33.2
     review C2 CR-03), from the ONE shared partial: the This Week headliners name the same bets
     and must word and sign them identically. Imported inside the block, because this block is
     also rendered alone as the /bets fragment. #}
  {% import "components/_bet_pick.html" as bp %}
```

Old (lines 247-260):

```jinja
            {# The pick names the TEAM, never the raw side code. The two teams come from the
               game id's AWAY@HOME matchup; an id without one falls back to the words. #}
            {%- set matchup = bet.game_id.split('_')[-1] -%}
            {%- set away_team, home_team = matchup.split('@') if '@' in matchup else ("Away", "Home") -%}
            {%- set picked_team = home_team if bet.bet_side in ("home", "home_cover") else away_team -%}
            <td class="px-4 py-3 text-left text-gray-700 whitespace-nowrap">
              {%- if bet.target == "ou" -%}
                {{ bet.bet_side|capitalize }} {{ "%g"|format(bet.line) if bet.line is not none else "" }}
              {%- elif bet.target == "ats" and bet.line is not none -%}
                {{ picked_team }} {{ side_line(bet, "%+g") }}
              {%- else -%}
                {{ picked_team }}
              {%- endif -%}
            </td>
```

New:

```jinja
            {# The pick names the TEAM, never the raw side code (components/_bet_pick.html). #}
            <td class="px-4 py-3 text-left text-gray-700 whitespace-nowrap">{{ bp.pick_label(bet) }}</td>
```

Old (line 263):

```jinja
              {%- elif bet.target == "ats" -%}{{ side_line(bet, "%+.1f") }}
```

New:

```jinja
              {%- elif bet.target == "ats" -%}{{ bp.side_line(bet, "%+.1f") }}
```

- [ ] **Step 5: Prove the move changed nothing on /bets**

Run: `uv run pytest tests/api/test_bets_page.py -v -k "served or tie_break or raw_target or spread_pick or moneyline" && uv run pytest tests/api/test_export.py -v -k "bets"`
Expected: all pass. The pick and line cells render the same text as before.

- [ ] **Step 6: Create `components/_bet_headliners.html`**

```jinja
{# The week's bets, first on This Week (Broadcast redesign, spec 7.1, mockup 02 layout L3).

   Pass: headliner -- {"state", "bets", "season", "week"} from pages._build_headliner. The state is
   NOT decided here: it is the render /bets makes for the same season and week, read off the same
   context in pages/bets.html's own precedence, so the two pages cannot describe one week two
   ways. Only the "bets" state draws cards; every other state is one line pointing to /bets,
   where the full explanation lives.

   Each card shows the bet's OWN stored numbers -- the pick, the stake in units, the EV and its
   pre-registered EV band -- and nothing derived. No "edge": the only stored edge is the
   prediction row's, measured against a different line than the bet's, so it would be a second
   number that need not agree with the first. No hue: the EV band stays monochrome
   (_ev_band_badge.html) for the reason it is monochrome on /bets, and the team-colour wash is
   identity, not a signal. Rendered only through the app's environment (it reads the team_colors
   global), never by the plain-Jinja card tests. #}
{% import "components/_broadcast.html" as bc %}
{% import "components/_bet_pick.html" as bp %}
{% set BET_TYPE_LABELS = {"wp": "Winner", "ats": "Spread", "ou": "Totals"} %}
{% set HEADLINER_NOTES = {
  "none_cleared": "No bets cleared the floor this week. Every candidate that was evaluated, and why each was declined, is on Bets.",
  "not_evaluated": "Not evaluated yet. Each game's list is built by the daily run the day before its kickoff, before its 6:00 PM Eastern lock.",
  "blocked": "This week's list is missing for games past their lock. Bets names them and says how to rebuild it.",
  "not_built": "The bet list has not been built in this cache yet. Bets says how to build it.",
  "no_week": "There is no bet list for this week."
} %}
{% set bets_url = "/bets?season=" ~ (headliner.season if headliner.season is not none else "") ~ "&week=" ~ (headliner.week if headliner.week is not none else "") %}
<section id="headliners" class="mb-8" data-headliner-state="{{ headliner.state }}">
  {{ bc.section_head("This week's bets") }}
  {% if headliner.state == "bets" %}
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-3">
    {% for bet in headliner.bets %}
    {%- set picked = bp.picked_team(bet)|trim -%}
    {%- set drawn = (bet.game.home_team if bet.game else picked) if bet.target == "ou" else picked -%}
    {%- set colors = team_colors(drawn) -%}
    <article class="panel relative overflow-hidden border-t-[3px] border-accent" style="background-image: linear-gradient(135deg, color-mix(in srgb, {{ colors.bg }} 35%, transparent) 0%, transparent 55%);" data-headliner-bet="{{ bet.game_id }}:{{ bet.target }}">
      <a href="/games/{{ bet.game_id }}" class="block">
        <p class="label text-[#C4CAD8]">{{ bet.game_id.split('_')[-1].replace('@', ' @ ') }}{% if bet.game %} &middot; {{ bet.game.kickoff_label }}{% endif %}</p>
        <p class="label mt-1">{{ BET_TYPE_LABELS.get(bet.target, bet.target) }}</p>
        <p class="display text-4xl leading-none text-accent">{{ bp.pick_label(bet) }}</p>
        {# stake_units and per_bet_ev are stored by the selector; the x100 is a unit render
           (fraction -> percent), the same one /bets applies under its "EV %" header (UIAP-01). #}
        <p class="mt-3 flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-muted">
          <span>Stake <span class="num text-fg">{{ "%.2f"|format(bet.stake_units) }}u</span></span>
          <span>EV <span class="num text-fg">{{ "%+.2f"|format(bet.per_bet_ev * 100) }}%</span></span>
          {% with band=bet.ev_tier %}{% include "components/_ev_band_badge.html" %}{% endwith %}
        </p>
        <span aria-hidden="true" class="display pointer-events-none absolute -right-2 -bottom-6 text-8xl text-white/5">{{ drawn }}</span>
      </a>
    </article>
    {% endfor %}
  </div>
  <p class="mt-2 text-right text-xs">
    <a href="{{ bets_url }}" class="font-semibold text-accent hover:underline">Full list, every declined candidate and the record on Bets &rsaquo;</a>
  </p>
  {% else %}
  <p class="panel text-sm text-fg" data-headliner-note>
    {{ HEADLINER_NOTES.get(headliner.state, headliner.state) }}
    <a href="{{ bets_url if headliner.state != 'no_week' else '/bets' }}" class="ml-1 whitespace-nowrap font-semibold text-accent hover:underline">Open Bets &rsaquo;</a>
  </p>
  {% endif %}
</section>
```

- [ ] **Step 7: Replace `pages/this_week.html`**

Whole file:

```jinja
{% extends "base.html" %}
{% block title %}This Week's Predictions{% endblock %}
{% block content %}
{# This Week (Broadcast redesign, spec 7.1, mockup 02: layout L3 with L1's TV-window tags): the
   week's bets first, then every game grouped by TV window.

   The CONTROLS sit outside #game-grid, so a week, season or sort change swaps only the content
   below them. The HEADING sits inside it, so it can never name a week other than the one the
   grid shows -- a "WEEK 3" heading outside the swap target would go stale on the first change. #}
<div class="mb-4 flex flex-wrap items-end justify-between gap-4">
  <p class="label">This Week</p>
  <div class="flex flex-wrap items-center gap-3">
    {% include "components/_week_selector.html" %}
    {% include "components/_sort_controls.html" %}
    {% with csv_url="/api/export/csv?season=" ~ (current_season or '') ~ "&week=" ~ (current_week or ''),
            json_url="/api/export/json?season=" ~ (current_season or '') ~ "&week=" ~ (current_week or ''),
            season_csv_url="/api/export/csv?season=" ~ (current_season or ''),
            season_json_url="/api/export/json?season=" ~ (current_season or '') %}
      {% include "components/_export_buttons.html" %}
    {% endwith %}
  </div>
</div>

{# The page now leads with bets, so it carries the same condensed not-advice note /bets does:
   one line, with the full original text behind "Why?". #}
{% include "components/_not_advice_banner.html" %}

{# Game grid -- HTMX swap target #}
<div id="game-grid">
  {% block game_grid %}
  {# Imported INSIDE the block: jinja2-fragments renders this block alone for /fragments/games and
     for an HX-Request to /, and a block rendered alone never runs the template's top-level
     statements, so a top-level import would be undefined in every fragment. #}
  {% import "components/_broadcast.html" as bc %}
  {% set SORTED_BY = {"confidence": "sorted by confidence", "edge": "sorted by edge"} %}
  <div class="mb-5">
    <h1 class="display text-5xl leading-none text-fg">{% if current_week %}Week {{ current_week }}{% else %}This Week{% endif %}</h1>
    <p class="mt-2 text-sm text-muted">{% if current_season %}{{ current_season }} season &middot; {% endif %}{{ games|length }} game{{ "" if games|length == 1 else "s" }}{% if headliner is defined and headliner.state == "bets" %} &middot; <span class="font-semibold text-accent">{{ headliner.bets|length }} bet{{ "" if headliner.bets|length == 1 else "s" }}</span> on the list{% endif %} &middot; each game locks at 6 PM ET the day before its kickoff</p>
  </div>

  {# Old-rule label for the selected week (R16 / D33.2-07). Inside game_grid so an HTMX swap
     re-renders it for the newly selected season rather than leaving a stale one behind. Its scope
     covers the headliner bets too (pages.this_week_page), which sit in this same block. #}
  {% with scope=old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}
  {% include "components/_week_summary.html" %}

  {% if headliner is defined and headliner %}
    {% include "components/_bet_headliners.html" %}
  {% endif %}

  {% if games %}
    {% if slate_groups %}
      {# A time order, cut into TV windows (api.presentation.group_games_by_window). #}
      {% for group in slate_groups %}
      <section class="mb-6" data-window="{{ group.label }}">
        {{ bc.section_head(group.label, meta=(group.games|length) ~ (" game" if group.games|length == 1 else " games")) }}
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {% for game in group.games %}
            {% include "components/_game_card.html" %}
          {% endfor %}
        </div>
      </section>
      {% endfor %}
    {% else %}
      {# A confidence or edge order: one grid in that order, because time groups would
         contradict the order the reader asked for. #}
      <section class="mb-6">
        {{ bc.section_head("Full slate", meta=SORTED_BY.get(current_sort)) }}
        <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3">
          {% for game in games %}
            {% include "components/_game_card.html" %}
          {% endfor %}
        </div>
      </section>
    {% endif %}
  {% else %}
    {% with heading="No predictions available",
            body="Predictions for this week have not been generated yet. Run the prediction pipeline to see this week's games.",
            action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% endif %}
  {% endblock %}
</div>

<script>
(function() {
  function updateExportUrls() {
    var seasonEl = document.getElementById('season-select');
    var weekEl = document.getElementById('week-select');
    var season = seasonEl ? seasonEl.value : '{{ current_season or "" }}';
    var week = weekEl ? weekEl.value : '{{ current_week or "" }}';

    var csvBtn = document.getElementById('export-csv');
    var jsonBtn = document.getElementById('export-json');
    var seasonCsvBtn = document.getElementById('export-season-csv');

    if (csvBtn) csvBtn.href = '/api/export/csv?season=' + season + '&week=' + week;
    if (jsonBtn) jsonBtn.href = '/api/export/json?season=' + season + '&week=' + week;
    if (seasonCsvBtn) seasonCsvBtn.href = '/api/export/csv?season=' + season;
  }

  // Update on dropdown change
  document.addEventListener('change', function(e) {
    if (e.target && (e.target.id === 'season-select' || e.target.id === 'week-select')) {
      updateExportUrls();
    }
  });

  // Update after HTMX swaps (week selector updates available_weeks which changes week dropdown)
  document.addEventListener('htmx:afterSwap', function() {
    updateExportUrls();
  });
})();
</script>
{% endblock %}
```

- [ ] **Step 8: Rebuild the CSS**

Run: `./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify`
Expected: `Done in ...` with no error.

- [ ] **Step 9: Update the grid assertion the slate changes on purpose**

The slate's grid is `grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4`; `md:grid-cols-2` no longer appears on `/`, so the D-07 test is updated (spec 11 lists grid-class assertions as updated on purpose). In `tests/api/test_pages.py`, old:

```python
def test_this_week_page_responsive_grid(test_client: TestClient):
    """D-07: Grid uses responsive column classes."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "grid-cols-1" in html
    assert "md:grid-cols-2" in html
    assert "lg:grid-cols-3" in html
```

New:

```python
def test_this_week_page_responsive_grid(test_client: TestClient):
    """D-07: Grid uses responsive column classes (Broadcast slate: 1, 2, 3, then 4 columns)."""
    response = test_client.get("/")
    assert response.status_code == 200
    html = response.text
    assert "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4" in html
```

- [ ] **Step 10: Run the layout tests and the This Week / label regressions**

Run: `uv run pytest tests/api/test_this_week_headliner.py -v`
Expected: all pass. That is 24 items: 14 from Task 6, plus 10 layout items (5 from the parametrised state test and 5 single tests).

```bash
uv run pytest tests/api/test_fragments.py -q
uv run pytest tests/api/test_cache_headers.py -q
uv run pytest tests/unit/test_page_labels.py -q
uv run pytest tests/api/test_page_labels_routes.py -q
```

Expected: pass. `this_week.html` still renders exactly one `data-old-rule-label` and the marker `This Week's Predictions`.

Run: `uv run pytest tests/api/test_pages.py -q -k "this_week or landing or selector"`
Expected: pass, including `test_this_week_page_responsive_grid` as updated in Step 9. The selector snapshot is unchanged, because the include and its parameters are identical.

- [ ] **Step 11: Lint and type-check the touched tests**

```bash
uv run ruff check tests/api/test_this_week_headliner.py tests/api/test_pages.py
uv run ruff format tests/api/test_this_week_headliner.py tests/api/test_pages.py
uv run pyright tests/api/test_this_week_headliner.py tests/api/test_pages.py
```

Expected: no ruff error. pyright reports no error on a line this task wrote; an error elsewhere in `test_pages.py` predates the branch -- note it in the task report and leave it.

- [ ] **Step 12: Commit**

```bash
git add web/templates/components/_bet_pick.html web/templates/pages/bets.html web/templates/components/_bet_headliners.html web/templates/pages/this_week.html web/static/css/tailwind-compiled.css tests/api/test_this_week_headliner.py tests/api/test_pages.py
git commit -m "$(cat <<'EOF'
feat(redesign): This Week layout with headliner bets and the TV-window slate

The week's bets lead the page (the /bets state for the same week, never re-decided); the slate
follows grouped by TV window, or as one grid for a confidence or edge sort. The pick wording and
spread sign move into components/_bet_pick.html, shared verbatim with /bets.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 9: The Broadcast game detail page

**Files:**
- Replace: `web/templates/pages/game_detail.html` (whole file)
- Modify: `api/routes/pages.py` -- the `game_detail_page` handler
- Create: `tests/api/test_game_detail_page.py`
- Modify: `tests/api/test_pages.py` -- the functions `test_game_detail_page` and `test_game_detail_team_context`
- Rebuild: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes (Task 2): `decorate_game`.
- Consumes (Task 3): `bc.team_block(abbr, bg, fg, large)`, `bc.edge_chip(text, soft)`, and the classes `panel`, `panel-title`, `skew`, `unskew`, `skew-control`, `skew-control-active`, `display`, `label`, `num`.
- Consumes (Task 4): `components/_confidence_badge.html` (monochrome, takes `level` and `value`), `components/_status_badge.html`, `components/_old_rule_label.html`.
- Consumes (Task 7): `pv.team_pct`.
- Consumes (Task 1): the `grow-in` class, on the win-probability bar.
- The feature-chart script mirrors the chart colours of contract section 8 as literals, because it runs client-side; it imports nothing from `api/charts/theme.py`, which Task 17 creates later.
- Produces: `data-band="<level>"` and `data-edge="<target>"` hooks on the detail page. The kept ids are `tab-wp|ats|ou`, `feature-chart`, `role="tablist"` and `showFeatureChart`. The kept text is "Model vs Market", "What drives the prediction", "Tale of the tape", "Elo rating", "Venue &amp; Weather", "Published", the result badges and "20 - 27"-style scores.

- [ ] **Step 1: Write the failing detail tests**

Create `tests/api/test_game_detail_page.py`:

```python
"""The Broadcast game detail page (Broadcast redesign, Task 9).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any

import duckdb
from fastapi.testclient import TestClient

from api.cache import CACHE_SCHEMA, PREDICTIONS_TABLE_COLUMNS
from api.dependencies import get_db, templates
from api.presentation import decorate_game
from api.services import DataService, clear_cache

_GAME_ID = "2026_W03_KC@MIA"
_HUE = re.compile(r"\b(?:text|bg|border)-(green|red)-\d{2,3}")
# One rendered band: the data-band hook, the monochrome label inside it, the band that label
# declares, and its visible text. Whitespace between the tags is the include's own newlines.
_BAND = re.compile(
    r'<span data-band="([a-z]+)">\s*'
    r'<span class="band band-([a-z]+)" data-confidence-band="([a-z]+)">([^<]+)</span>\s*'
    r"</span>"
)


class _StubRequest:
    """The one attribute the base template needs from a request: ``url_for`` for static files."""

    def url_for(self, name: str, **path_params: Any) -> str:
        return f"/{name}/{path_params.get('path', '')}"


def _main(html: str) -> str:
    """The page's <main> region, so the site chrome cannot satisfy or spoil an assertion."""
    match = re.search(r"<main[^>]*>(.*)</main>", html, re.DOTALL)
    assert match, "the page has no <main> region"
    return match.group(1)


def _hues(html: str) -> list[str]:
    return [
        match.group(1)
        for classes in re.findall(r'class="([^"]*)"', html)
        for match in _HUE.finditer(classes)
    ]


def _detail_html(**overrides: Any) -> str:
    """Render the detail page for KC at MIA stored in an in-memory cache, decorated as the route does."""
    clear_cache()
    conn = duckdb.connect(":memory:")
    for statement in CACHE_SCHEMA.strip().split(";"):
        if statement.strip():
            conn.execute(statement)
    row = dict.fromkeys(PREDICTIONS_TABLE_COLUMNS)
    row.update(
        game_id=_GAME_ID,
        season=2026,
        week=3,
        game_date=datetime(2026, 9, 27, 13, 0),
        home_team="MIA",
        away_team="KC",
        status="scheduled",
        wp_prob=0.412,
        ats_prediction=-2.5,
        ou_prediction=41.3,
    )
    row.update(overrides)
    columns = ", ".join(PREDICTIONS_TABLE_COLUMNS)
    placeholders = ", ".join("?" for _ in PREDICTIONS_TABLE_COLUMNS)
    conn.execute(
        f"INSERT INTO predictions ({columns}) VALUES ({placeholders})",
        [row[c] for c in PREDICTIONS_TABLE_COLUMNS],
    )
    game = DataService(conn).get_game_detail(_GAME_ID)
    conn.close()
    assert game is not None
    return templates.env.get_template("pages/game_detail.html").render(
        request=_StubRequest(),
        game=decorate_game(game),
        current_path="",
        cache_meta={},
        old_rule_scope=DataService.old_rule_scope([]),
    )


def test_the_detail_page_renders_the_band_it_was_served(test_client: TestClient) -> None:
    """Moved from / (31-17 / D31-23): the page renders the STORED band, never a re-derived one.

    Every stored band must appear once, for its own target, as a monochrome label -- a page that
    turned a served "low" into a "high" keeps the vocabulary intact and changes these counts. The
    check reads the LABEL the reader sees, not only the data-band hook that echoes the stored
    value: the hook, the label's class, its data-confidence-band and its text must all agree.
    """
    from api.main import app
    from utils.edge_tier import EDGE_TIER_LABELS

    html = test_client.get("/games/2024_W01_BUF@KC").text
    # The app object, not test_client.app: TestClient types .app as a bare ASGI callable.
    game = DataService(app.dependency_overrides[get_db]()).get_game_detail("2024_W01_BUF@KC")
    assert game is not None
    served = [
        game[f"{target}_confidence"]
        for target in ("wp", "ats", "ou")
        if game.get(f"{target}_confidence") is not None
    ]
    assert served, "the fixture served no band; the check would pass vacuously"
    assert set(served) <= set(EDGE_TIER_LABELS)

    hooks = re.findall(r'data-band="([a-z]+)"', html)
    labels = _BAND.findall(html)
    assert len(labels) == len(hooks), "a data-band hook does not wrap exactly one monochrome label"
    assert Counter(hook for hook, _css, _attr, _text in labels) == Counter(served)
    for hook, css_band, attr_band, text in labels:
        # The class is exactly "band band-<level>", so no hue can ride on it.
        assert hook == css_band == attr_band == text.strip().lower(), (
            f"the {hook!r} slot renders the label {text!r} (class band-{css_band}, "
            f"data-confidence-band={attr_band!r}); all four must name the stored band"
        )


def test_a_detail_page_with_no_market_lines_reads_cleanly() -> None:
    """Review Focus 1 on the detail page: every market slot says No line, and nothing else leaks."""
    main = _main(_detail_html())

    assert main.count("No line") == 3, "each of the three market cells must say No line"
    assert "data-edge" not in main
    assert "data-band" not in main
    assert "None" not in main
    assert "nan" not in main.lower()
    for pick_word in ("covers", "Over ", "Under "):
        assert pick_word not in main
    assert "KC 58.8%" in main and "KC by 2.5" in main and "41.3" in main


def test_only_a_realised_result_is_green_or_red() -> None:
    before = _main(
        _detail_html(
            market_wp=0.449,
            market_spread=-1.5,
            market_total=44.5,
            wp_edge=-0.037,
            ats_edge=-1.0,
            ou_edge=-0.0719,
            wp_confidence="medium",
            ats_confidence="low",
            ou_confidence="high",
        )
    )
    assert _hues(before) == [], "a pre-game edge or band was drawn in green or red"
    assert re.search(r'data-edge="ats".*?-1\.0 pts', before, re.DOTALL)
    assert "KC covers" in before and "Under 44.5" in before

    after = _main(
        _detail_html(status="completed", away_score=20, home_score=27, wp_prob=0.70)
    )
    assert "20 - 27" in after and "Correct" in after
    assert set(_hues(after)) == {"green"}


def test_the_page_is_decorated_and_links_back_to_its_week(test_client: TestClient) -> None:
    html = test_client.get("/games/2024_W01_BUF@KC").text

    assert "Chiefs" in html, "the team nickname from api.presentation is missing"
    assert "#E31837" in html, "the KC team colour from utils/team_data.py is missing"
    assert 'href="/?season=2024&amp;week=1"' in html
    assert "Published" in html and "Blended" not in html
    # Spec 5: the win-probability bar grows in once (Task 1's .grow-in, off under reduced motion).
    assert re.search(
        r'<div class="[^"]*\bgrow-in\b[^"]*" role="img" aria-label="Win probability:', html
    ), "the win-probability bar does not carry the grow-in animation"
```

- [ ] **Step 2: Run the detail tests to verify they fail**

Run: `uv run pytest tests/api/test_game_detail_page.py -v`
Expected: FAIL. The old page has no `data-band`, no "Published" and no nickname.

- [ ] **Step 3: Decorate the game in the route**

In `api/routes/pages.py` `game_detail_page`, old:

```python
    game = service.get_game_detail(game_id)
    cache_meta = service.get_cache_meta()
```

New:

```python
    game = service.get_game_detail(game_id)
    if game is not None:
        # Team colours, nicknames and the kickoff label for the matchup header (presentation
        # only, api.presentation). A NEW dict; the service row is not mutated.
        game = decorate_game(game)
    cache_meta = service.get_cache_meta()
```

And in the same handler's context dict, old:

```python
        "current_path": "",
```

New (base.html marks This Week active for any `/games/...` path, with `aria-current="page"`, so the reader keeps their place in the nav -- Task 3 / Part A contract note 10):

```python
        "current_path": f"/games/{game_id}",
```

- [ ] **Step 4: Replace `pages/game_detail.html`**

Whole file:

```jinja
{% extends "base.html" %}
{% block title %}{% if game %}{{ game.away_team }} @ {{ game.home_team }}{% else %}Game Not Found{% endif %}{% endblock %}
{% block content %}
{% import "components/_prediction_values.html" as pv %}
{% import "components/_broadcast.html" as bc %}

{% if not game %}
  {% with heading="Game not found",
          body="This game does not exist in the prediction database.",
          action_text="Back to This Week", action_url="/" %}
    {% include "components/_empty_state.html" %}
  {% endwith %}
{% else %}
{# Game detail (Broadcast redesign, spec 7.2, mockup 04). The route passes the row through
   api.presentation.decorate_game; every decorated field is read with a default so a bare row
   (tests render one) still draws. #}
{%- set home = game.home_team -%}
{%- set away = game.away_team -%}
{%- set home_bg = game.home_color|default("#1D2436") -%}
{%- set away_bg = game.away_color|default("#1D2436") -%}
{%- set ctx = game.context -%}

<a href="/?season={{ game.season }}&amp;week={{ game.week }}" class="mb-3 inline-flex items-center gap-1 text-sm font-semibold text-accent hover:underline">&lsaquo; Week {{ game.week }}</a>

{# Old-rule label for this game (R16 / D33.2-07): a 2021-2025 game's prediction, result badge
   and CLV are all pre-fix. The route scopes it to the game's own season. #}
{% with scope=old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}

{# The matchup header: team blocks and win chances either side, kickoff and stadium between. #}
<h1 class="sr-only">{{ away }} @ {{ home }}</h1>
<section class="mb-4 overflow-hidden rounded-sm bg-panel p-5" style="background-image: linear-gradient(90deg, color-mix(in srgb, {{ away_bg }} 40%, transparent) 0%, transparent 42%, transparent 58%, color-mix(in srgb, {{ home_bg }} 45%, transparent) 100%);">
  <div class="grid grid-cols-1 items-center gap-4 md:grid-cols-[1fr_auto_1fr]">
    <div class="flex items-center gap-4">
      {{ bc.team_block(away, away_bg, game.away_fg|default("#FFFFFF"), large=true) }}
      <div>
        <p class="display text-2xl leading-none">{{ game.away_name|default(away) }}</p>
        {% if ctx and ctx.away_elo is not none %}<p class="mt-1 text-xs text-[#C4CAD8]">Elo {{ "%.0f"|format(ctx.away_elo) }}</p>{% endif %}
      </div>
      <p class="display ml-auto text-5xl tabular-nums {{ 'text-accent' if game.wp_prob is not none and game.wp_prob < 0.5 else 'text-[#C4CAD8]' }}">{{ pv.team_pct(game.wp_prob, "away") }}</p>
    </div>
    <div class="text-center">
      <p class="display text-2xl text-white/40">@</p>
      <p class="label text-[#C4CAD8]">{{ game.kickoff_label|default(game.game_date if game.game_date is not none else "Time TBD") }}</p>
      {% if ctx and ctx.venue_name %}<p class="label">{{ ctx.venue_name }}</p>{% endif %}
      <p class="mt-1">{% with status=game.status %}{% include "components/_status_badge.html" %}{% endwith %}</p>
    </div>
    <div class="flex items-center gap-4">
      <p class="display mr-auto text-5xl tabular-nums {{ 'text-accent' if game.wp_prob is not none and game.wp_prob > 0.5 else 'text-[#C4CAD8]' }}">{{ pv.team_pct(game.wp_prob, "home") }}</p>
      <div class="text-right">
        <p class="display text-2xl leading-none">{{ game.home_name|default(home) }}</p>
        {% if ctx and ctx.home_elo is not none %}<p class="mt-1 text-xs text-[#C4CAD8]">Elo {{ "%.0f"|format(ctx.home_elo) }}</p>{% endif %}
      </div>
      {{ bc.team_block(home, home_bg, game.home_fg|default("#FFFFFF"), large=true) }}
    </div>
  </div>
  {% if game.wp_prob is not none %}
  {# grow-in (Task 1): the bar sweeps in from the left once on load (spec 5); a reader who asks for
     reduced motion sees it at its final width at once. #}
  <div class="mt-4 flex h-1.5 gap-1 grow-in" role="img" aria-label="Win probability: {{ away }} {{ pv.team_pct(game.wp_prob, 'away') }}, {{ home }} {{ pv.team_pct(game.wp_prob, 'home') }}">
    <span class="skew" style="width: {{ ((1 - game.wp_prob) * 100)|round(1) }}%; background: {{ away_bg }};"></span>
    <span class="skew" style="width: {{ (game.wp_prob * 100)|round(1) }}%; background: {{ home_bg }};"></span>
  </div>
  {% endif %}

  {# Result overlay for completed games (D-15). Three states, matching the card's grading
     (33.2 review C2 WR-04): correct, incorrect, and NEITHER -- a tie or a game with no WP pick,
     which used to render a red "Incorrect". The result and the CLV are REALISED numbers, so they
     alone may be green or red. #}
  {% if game.status == "completed" and game.home_score is not none %}
  <div class="mt-4 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-3">
    <p class="flex items-center gap-3">
      <span class="label">Final</span>
      <span class="display text-2xl tabular-nums">{{ game.away_score }} - {{ game.home_score }}</span>
      <span class="label px-2 py-0.5 {% if game.wp_correct is sameas true %}bg-green-500/15 text-green-400{% elif game.wp_correct is sameas false %}bg-red-500/15 text-red-400{% else %}bg-panel-2 text-fg{% endif %}">
        {% if game.wp_correct is sameas true %}Correct{% elif game.wp_correct is sameas false %}Incorrect{% elif game.home_score == game.away_score %}Tie{% else %}No pick{% endif %}
      </span>
    </p>
    {% if game.wp_clv is not none %}
    <span class="num text-sm font-semibold {% if game.wp_clv > 0 %}text-green-400{% else %}text-red-400{% endif %}">CLV: {{ "%+.2f"|format(game.wp_clv) }}%</span>
    {% endif %}
  </div>
  {% endif %}
</section>

<div class="grid grid-cols-1 gap-4 lg:grid-cols-[1.25fr_1fr]">
  <div class="space-y-4">

    {# Model vs Market (D-14). Model = OUR MODEL'S own number; Market = the pre-lock line we own
       ("No line" where we own none -- never a zero); Published = the blended number the site
       publishes. An edge is a neutral yellow chip, never green or red, with its stored band
       beside it, monochrome (this is where the band lives since it left the cards). An absent
       edge is "-". #}
    <section class="panel">
      <h2 class="panel-title">Model vs Market</h2>
      <div class="overflow-x-auto">
        <table class="w-full text-sm">
          <thead>
            <tr class="label">
              <th scope="col" class="pb-2 text-left"><span class="sr-only">Metric</span></th>
              <th scope="col" class="pb-2 text-right">Model</th>
              <th scope="col" class="pb-2 text-right">Market</th>
              <th scope="col" class="pb-2 text-right">Published</th>
              <th scope="col" class="pb-2 text-right">Edge</th>
            </tr>
          </thead>
          <tbody class="divide-y divide-line">
            <tr>
              <td class="py-2.5">Win Probability</td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.win_prob(game.wp_prob, home, away, "-") }}</td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.win_prob(game.market_wp, home, away, pv.market_wp_missing(game)) }}</td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.win_prob(game.blended_wp, home, away, "-") }}</td>
              <td class="whitespace-nowrap py-2.5 text-right">
                {%- if game.wp_edge is not none -%}
                <span data-edge="wp">{{ bc.edge_chip("%+.1f"|format(game.wp_edge * 100) ~ "%") }}</span>
                {%- else -%}{{ pv.absent("-") }}{%- endif -%}
                {%- if game.wp_confidence %} <span data-band="{{ game.wp_confidence }}">{% with level=game.wp_confidence, value=game.wp_confidence|title %}{% include "components/_confidence_badge.html" %}{% endwith %}</span>{% endif -%}
              </td>
            </tr>
            <tr>
              <td class="py-2.5">Spread
                {% if game.ats_prediction is not none and game.market_spread is not none %}
                <span class="block text-xs font-semibold text-accent">
                  {# Home margins, POSITIVE when the home team is favoured (DEF-31-01): a model
                     margin above the line = HOME covers, the side backtest/ats_ev_chain.py prices.
                     Equal numbers are NO PICK. The same label as components/_game_card.html. #}
                  {% if game.ats_prediction > game.market_spread %}{{ home }} covers{% elif game.ats_prediction < game.market_spread %}{{ away }} covers{% else %}No pick{% endif %}
                </span>
                {% endif %}
              </td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.margin(game.ats_prediction, home, away, "-") }}</td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.margin(game.market_spread, home, away) }}</td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.margin(game.blended_ats, home, away, "-") }}</td>
              {# R13 / D33-05 (Plan 33-10): the ATS edge is a POINT DIFFERENCE, not a ratio, so it
                 renders in points; the WP edge (a probability delta) and the O/U edge (a fraction
                 of the total) keep their percentage rendering. #}
              <td class="whitespace-nowrap py-2.5 text-right">
                {%- if game.ats_edge is not none -%}
                <span data-edge="ats">{{ bc.edge_chip("%+.1f"|format(game.ats_edge) ~ " pts") }}</span>
                {%- else -%}{{ pv.absent("-") }}{%- endif -%}
                {%- if game.ats_confidence %} <span data-band="{{ game.ats_confidence }}">{% with level=game.ats_confidence, value=game.ats_confidence|title %}{% include "components/_confidence_badge.html" %}{% endwith %}</span>{% endif -%}
              </td>
            </tr>
            <tr>
              <td class="py-2.5">Total
                {% if game.ou_prediction is not none and game.market_total is not none %}
                <span class="block text-xs font-semibold text-accent">
                  {% if game.ou_prediction > game.market_total %}Over {{ "%.1f"|format(game.market_total) }}{% elif game.ou_prediction < game.market_total %}Under {{ "%.1f"|format(game.market_total) }}{% else %}No pick{% endif %}
                </span>
                {% endif %}
              </td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.total(game.ou_prediction, "-") }}</td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.total(game.market_total) }}</td>
              <td class="num whitespace-nowrap py-2.5 text-right">{{ pv.total(game.blended_ou, "-") }}</td>
              <td class="whitespace-nowrap py-2.5 text-right">
                {%- if game.ou_edge is not none -%}
                <span data-edge="ou">{{ bc.edge_chip("%+.1f"|format(game.ou_edge * 100) ~ "%") }}</span>
                {%- else -%}{{ pv.absent("-") }}{%- endif -%}
                {%- if game.ou_confidence %} <span data-band="{{ game.ou_confidence }}">{% with level=game.ou_confidence, value=game.ou_confidence|title %}{% include "components/_confidence_badge.html" %}{% endwith %}</span>{% endif -%}
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    {# What drives the prediction (D-13 + UIAP-04: multi-target tabs). Same ids, same client-side
       Plotly call; dark styling mirrors api/charts/theme.py, as literals because it runs in the
       browser. Positive importances are the accent yellow and negative ones neutral grey --
       never red, which on this site means a realised loss. #}
    <section class="panel">
      <h2 class="panel-title">What drives the prediction</h2>
      {% if game.feature_importances %}
      {% set available_targets = [] %}
      {% for t in ["wp", "ats", "ou"] %}
        {% if t in game.feature_importances %}
          {% if available_targets.append(t) %}{% endif %}
        {% endif %}
      {% endfor %}
      {% set TAB_LABELS = {"wp": "Win", "ats": "Spread", "ou": "Total"} %}

      {% if available_targets | length > 1 %}
      <div class="mb-3 flex gap-1.5" role="tablist" aria-label="Model">
        {% for target in available_targets %}
        <button type="button"
                onclick="showFeatureChart('{{ target }}')"
                id="tab-{{ target }}"
                role="tab"
                aria-selected="{{ 'true' if loop.first else 'false' }}"
                aria-controls="feature-chart"
                class="skew-control min-h-[44px] px-3{% if loop.first %} skew-control-active{% endif %}">
          <span class="unskew inline-block">{{ TAB_LABELS[target] }}</span>
        </button>
        {% endfor %}
      </div>
      {% endif %}

      <div id="feature-chart" role="tabpanel" style="width: 100%; min-height: 300px;"></div>
      <script>
        (function() {
          var allFeatures = {{ game.feature_importances | tojson }};
          var availableTargets = {{ available_targets | tojson }};
          var POSITIVE = "#FFD400";
          var NEGATIVE = "#5B6478";

          function showFeatureChart(target) {
            // Move the active state to the chosen tab.
            availableTargets.forEach(function(t) {
              var tab = document.getElementById("tab-" + t);
              if (!tab) return;
              var active = t === target;
              tab.classList.toggle("skew-control-active", active);
              tab.setAttribute("aria-selected", active ? "true" : "false");
            });

            // Top ten features by absolute importance, largest at the top.
            var data = allFeatures[target] || {};
            var sorted = Object.entries(data)
              .sort(function(a, b) { return Math.abs(b[1]) - Math.abs(a[1]); })
              .slice(0, 10);
            var names = sorted.map(function(d) { return d[0]; }).reverse();
            var values = sorted.map(function(d) { return d[1]; }).reverse();
            var colors = values.map(function(v) { return v >= 0 ? POSITIVE : NEGATIVE; });

            Plotly.newPlot("feature-chart", [{
              type: "bar",
              orientation: "h",
              x: values,
              y: names,
              marker: { color: colors },
              hovertemplate: "%{y}: %{x:.3f}<extra></extra>"
            }], {
              font: { family: "Barlow Condensed, Inter, sans-serif", size: 13, color: "#C4CAD8" },
              plot_bgcolor: "rgba(0,0,0,0)",
              paper_bgcolor: "rgba(0,0,0,0)",
              margin: { l: 160, r: 16, t: 8, b: 40 },
              xaxis: { title: { text: "IMPORTANCE" }, gridcolor: "rgba(255,255,255,0.06)", zeroline: false, color: "#8A93A8" },
              yaxis: { color: "#C4CAD8", tickfont: { family: "JetBrains Mono, monospace", size: 11 } },
              hoverlabel: { bgcolor: "#0B0F17", bordercolor: "#FFD400", font: { family: "JetBrains Mono, monospace", color: "#FFFFFF" } }
            }, { responsive: true, displayModeBar: false });
          }

          // Make function globally accessible for tab onclick handlers
          window.showFeatureChart = showFeatureChart;

          // Show default target (first available, preferring WP)
          showFeatureChart(availableTargets[0]);
        })();
      </script>
      {% else %}
        <p class="text-sm text-muted">Feature importance data not available for this game.</p>
      {% endif %}
    </section>
  </div>

  <div class="space-y-4">
    {# Tale of the tape (D-14: Elo, last 5, H2H), away on the left as in the header. Recent form
       is a set of REALISED results, so W and L alone are green and red. #}
    <section class="panel">
      <h2 class="panel-title">Tale of the tape</h2>
      {% if ctx %}
        {% if ctx.home_elo is not none or ctx.away_elo is not none %}
        <p class="label">Elo rating</p>
        <div class="mb-4 mt-1 grid grid-cols-[4rem_1fr_4rem] items-center gap-3">
          <p><span class="label block">{{ away }}</span><span class="display text-2xl">{% if ctx.away_elo is not none %}{{ "%.0f"|format(ctx.away_elo) }}{% else %}<span class="text-dim">N/A</span>{% endif %}</span></p>
          {% if ctx.away_elo is not none and ctx.home_elo is not none %}
          {# NOTHING IS COMPUTED HERE (UIAP-01): each half's CSS flex-grow is that team's stored
             Elo, so the browser sizes the split -- the same technique as the Season week strip. #}
          <div class="flex h-2.5 gap-1" role="img" aria-label="Elo rating: {{ away }} {{ "%.0f"|format(ctx.away_elo) }}, {{ home }} {{ "%.0f"|format(ctx.home_elo) }}">
            <span class="skew" aria-hidden="true" style="flex: {{ "%.0f"|format(ctx.away_elo) }} 1 0%; background: {{ away_bg }};"></span>
            <span class="skew" aria-hidden="true" style="flex: {{ "%.0f"|format(ctx.home_elo) }} 1 0%; background: {{ home_bg }};"></span>
          </div>
          {% else %}
          <span></span>
          {% endif %}
          <p class="text-right"><span class="label block">{{ home }}</span><span class="display text-2xl">{% if ctx.home_elo is not none %}{{ "%.0f"|format(ctx.home_elo) }}{% else %}<span class="text-dim">N/A</span>{% endif %}</span></p>
        </div>
        {% endif %}

        {% if ctx.home_last5_list or ctx.away_last5_list %}
        <p class="label mb-1.5">Recent form (last 5)</p>
        <div class="mb-4 flex justify-between gap-4">
          {% for team, results in [(away, ctx.away_last5_list), (home, ctx.home_last5_list)] %}
          <div>
            <p class="label mb-1">{{ team }}</p>
            <div class="flex gap-1">
              {% for result in results %}
              <span class="skew inline-flex h-6 w-6 items-center justify-center font-display text-sm font-extrabold {% if result == 'W' %}bg-green-500/15 text-green-400{% elif result == 'L' %}bg-red-500/15 text-red-400{% else %}bg-panel-2 text-muted{% endif %}"><span class="unskew">{{ result }}</span></span>
              {% endfor %}
              {% for _ in range(5 - results|length) %}<span class="skew h-6 w-6 bg-panel-2" aria-hidden="true"></span>{% endfor %}
            </div>
          </div>
          {% endfor %}
        </div>
        {% endif %}

        {% if ctx.h2h_record %}
        <p class="label mb-1">Head-to-head (last 5 seasons)</p>
        <p class="display text-xl">
          {%- if ctx.h2h_home_wins > ctx.h2h_away_wins -%}{{ home }} leads {{ ctx.h2h_home_wins }}-{{ ctx.h2h_away_wins }}
          {%- elif ctx.h2h_away_wins > ctx.h2h_home_wins -%}{{ away }} leads {{ ctx.h2h_away_wins }}-{{ ctx.h2h_home_wins }}
          {%- else -%}Series even {{ ctx.h2h_home_wins }}-{{ ctx.h2h_away_wins }}
          {%- endif -%}
        </p>
        {% endif %}
      {% else %}
        <p class="text-sm text-muted">Team context data not available for this game.</p>
      {% endif %}
    </section>

    {# Venue and Weather (D-14) #}
    <section class="panel">
      <h2 class="panel-title">Venue &amp; Weather</h2>
      {% if ctx %}
      <dl class="space-y-2.5 text-sm">
        {% if ctx.venue_name %}
        <div class="flex justify-between gap-4"><dt class="text-muted">Stadium</dt><dd class="text-right font-semibold text-fg">{{ ctx.venue_name }}</dd></div>
        {% endif %}
        {% if ctx.surface %}
        <div class="flex justify-between gap-4"><dt class="text-muted">Surface</dt><dd class="text-right font-semibold text-fg">{{ ctx.surface }}</dd></div>
        {% endif %}
        {% if ctx.roof_type %}
        <div class="flex justify-between gap-4"><dt class="text-muted">Roof</dt><dd class="text-right font-semibold text-fg">{{ ctx.roof_type|title }}</dd></div>
        {% endif %}
        {% if ctx.is_outdoor and ctx.weather_severity is not none %}
        <div class="flex justify-between gap-4"><dt class="text-muted">Weather Severity</dt><dd class="text-right font-semibold text-fg" title="severity {{ "%.2f"|format(ctx.weather_severity) }}">{{ ctx.weather_severity_band }}</dd></div>
        {% endif %}
        {% if ctx.is_outdoor and ctx.wind_mph is not none %}
        <div class="flex justify-between gap-4"><dt class="text-muted">Wind</dt><dd class="num text-right font-semibold text-fg">{{ "%.0f"|format(ctx.wind_mph) }} mph</dd></div>
        {% endif %}
        {% if ctx.is_divisional %}
        <div class="flex justify-between gap-4"><dt class="text-muted">Type</dt><dd class="text-right font-semibold text-fg">Divisional Game</dd></div>
        {% endif %}
      </dl>
      {% else %}
      <p class="text-sm text-muted">Venue and weather data not available.</p>
      {% endif %}
    </section>

    {# Export for this game #}
    <div class="flex justify-end">
      {% with csv_url="/api/export/csv?game_id=" ~ game.game_id,
              json_url="/api/export/json?game_id=" ~ game.game_id %}
        {% include "components/_export_buttons.html" %}
      {% endwith %}
    </div>
  </div>
</div>

{% endif %}
{% endblock %}
```

- [ ] **Step 5: Update the two detail tests the renames deliberately break**

In `tests/api/test_pages.py`, in `test_game_detail_page`, old:

```python
    assert "Feature Importance" in html
    assert "Prediction vs Market" in html
```

New:

```python
    # Panel titles renamed in the Broadcast redesign (spec 7.2).
    assert "What drives the prediction" in html
    assert "Model vs Market" in html
```

Old (the whole function `test_game_detail_team_context`):

```python
def test_game_detail_team_context(test_client: TestClient):
    """Game detail shows team context (Elo, form, H2H)."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    assert "Team Context" in html
    assert "Elo Ratings" in html
    assert "1550" in html  # home Elo
```

New:

```python
def test_game_detail_team_context(test_client: TestClient):
    """Game detail shows team context (Elo, form, H2H) in the Tale of the tape panel."""
    response = test_client.get("/games/2024_W01_BUF@KC")
    assert response.status_code == 200
    html = response.text
    assert "Tale of the tape" in html
    assert "Elo rating" in html
    assert "1550" in html  # home Elo
```

- [ ] **Step 6: Rebuild the CSS**

Run: `./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify`
Expected: `Done in ...` with no error.

- [ ] **Step 7: Run the detail tests and every test that renders or reads the detail page**

Run: `uv run pytest tests/api/test_game_detail_page.py -v`
Expected: 4 passed.

Run: `uv run pytest tests/api/test_pages.py -v -k "game_detail or detail_badge"`
Expected: all pass:
- `test_game_detail_page`, `_team_context`, `_venue_weather`, `_result_overlay` ("20 - 27", "Correct"), `_feature_chart` (`Plotly.newPlot("feature-chart"`), `_multi_target_importances`, `_not_found`, `_export_buttons`;
- `test_the_detail_badge_grades_a_tie_and_a_missing_pick_as_neither`.

```bash
uv run pytest "tests/unit/test_cache_ats_edge.py::test_the_game_detail_page_renders_the_ats_edge_in_points_not_as_a_percentage" -q
uv run pytest tests/unit/test_page_labels.py -q
uv run pytest tests/api/test_page_labels_routes.py -q
uv run pytest tests/api/test_cache_headers.py -q
```

Expected: pass. The ATS edge line still contains `game.ats_edge` and "pts", there is exactly one old-rule label, and the marker `BUF @ KC` is in `<title>` and the sr-only `<h1>`.

- [ ] **Step 8: Lint and type-check**

Run: `uv run ruff check api/routes/pages.py tests/api/test_game_detail_page.py tests/api/test_pages.py && uv run ruff format api/routes/pages.py tests/api/test_game_detail_page.py tests/api/test_pages.py && uv run pyright api/routes/pages.py tests/api/test_game_detail_page.py tests/api/test_pages.py`
Expected: no ruff error and 0 pyright errors in `api/routes/pages.py` and `tests/api/test_game_detail_page.py` (the band test reads the override map off `api.main.app`, not `test_client.app`, which TestClient types as a bare ASGI callable). In `test_pages.py`, no error on a line this task wrote; an older one predates the branch -- note it in the task report and leave it.

- [ ] **Step 9: Commit**

```bash
git add web/templates/pages/game_detail.html api/routes/pages.py web/static/css/tailwind-compiled.css tests/api/test_game_detail_page.py tests/api/test_pages.py
git commit -m "$(cat <<'EOF'
feat(redesign): Broadcast game detail page

Matchup header with team blocks and win chances; Model vs Market with neutral edge chips and the
stored band beside each, monochrome; the feature chart restyled dark (same ids and Plotly call);
the tale of the tape; venue and weather. Only the realised result, CLV and recent form are
green/red. "Blended" is now labelled "Published".

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

# Part C -- Bets (Tasks 10-12)

## Contract notes (Part C)

These are places where the real code forced a decision the contract did not spell out. Tasks 10-12 below rely on them.

1. **The /bets `<h1>` stays the static text "Weekly Bet List", not "Bets · Week N".** The page header sits OUTSIDE the `#bets-content` swap target. A week number in it would go stale on the first HTMX week change. The week is named by the selector beside it and by the "Live bets -- <season> Week <n>" tag, which IS inside the swap target. (The tests also pin that tag text.)
2. **The live-bets header meta shows "ranked by expected value · N bets" and drops the mockup's "X.XXu total".** Summing the stakes in the template would be a request-path aggregate (UIAP-01). The count is a row count of what is on the page, the same kind of count the existing "Suppressed candidates (N)" already shows.
3. **Section heads on /bets are written inline, not through `bc.section_head`.** The markup is:
   - `<div class="section-head">`
   - `<h2 class="tag">` or `<h2 class="tag-ghost">`, holding a `<span class="unskew">` with the text
   - optional badge
   - an explicit `<span class="flex-1 h-px bg-line">` rule
   - optional meta, then `</div>`.

   The tracker test (`test_the_badge_appears_on_both_tracker_block_headers`) requires the evidence badge to sit before the section's first `</div>`. **Task 1 must therefore make `.section-head` a plain flex row** (`display:flex; align-items:center; gap:.75rem`) that draws NO rule of its own (no `::before`/`::after`). `.tag` and `.tag-ghost` skew the box and expect a `.unskew` child. Task 3's `section_head` macro should emit this same structure.
4. **Macro imports go INSIDE `{% block bets_content %}`.** jinja2-fragments renders only the named block for an `HX-Request`, and top-level `{% import %}` statements of a child template are not executed in a block render. Any page Part B or D renders with `block_name=` needs the same treatment: import inside the block, or inside the included component.
5. **Tests in `tests/api/test_bets_page.py` that FIRST break at Task 4 or Task 5 (component restyles), not in Part C.** They should be applied in the task that causes the break. Exact replacements:
   - **5a. Task 4 (`_ev_band_badge.html`), `test_ev_band_badge_is_monochrome`.**
     Old:
     ```python
         assert "EV band" in body
         assert "bg-gray-200 text-gray-900 border border-gray-300" in body
     ```
     New:
     ```python
         assert "EV band" in body
         assert re.search(
             r'<span class="[^"]*\bband-high\b[^"]*"[^>]*title="EV band high', body
         ), "the high EV band did not render with the monochrome band-high style"
     ```
     This requires `_ev_band_badge.html` to keep `class` before `title="EV band ..."` on its outer `<span>`. The colour-guard regex in `test_green_and_red_appear_only_inside_the_tracker_sections` relies on the same order.
   - **5b. Task 4 (`_provenance_badge.html`), the `_VALIDATION_CLASSES` literal.**
     Old:
     ```python
     _VALIDATION_CLASSES: dict[str, str] = {
         "contaminated": "bg-gray-100 text-gray-700 border border-gray-300",
         "clean_holdout": "bg-gray-200 text-gray-900 border border-gray-300",
         "forward_realized": "bg-white text-gray-700 border border-gray-300",
     }
     ```
     New. Task 4 must use exactly this mapping in `_VALIDATION_CLASSES` of the partial; the strongest grey today, `clean_holdout`, maps to the strong chip:
     ```python
     _VALIDATION_CLASSES: dict[str, str] = {
         "contaminated": "evidence-chip",
         "clean_holdout": "evidence-chip evidence-chip-strong",
         "forward_realized": "evidence-chip",
     }
     ```
     The badge must also keep `data-provenance` and `data-validation-type` on its outer `<span>`, and keep the nested `<span class="sr-only">`, so that `_badges()` (which matches `...</span></span>`) still finds it. The assertion that reads the table is tightened in the same step to the whole attribute, `class="{classes}" data-provenance="{provenance}"`, because a bare `"evidence-chip" in section` also matches inside the strong chip (Task 4 Step 9 gives the exact edit).
   - **5c. Task 5 (`_error_state.html` -> `bg-red-950 border border-red-800`).** In these three tests, replace the literal `"bg-red-50"` with `"bg-red-950"`. Nothing else in each assertion changes:
     - `test_state_four_a_week_with_no_list_for_its_locked_games_is_refused`
     - `test_the_failure_template_carries_the_message_and_a_retry`
     - `test_the_forward_block_is_withheld_under_the_hard_block_while_replay_stays_readable`
   - **5d. Task 4 (the condensed not-advice and old-rule notes are now `<details>`), `test_zero_suppressed_rows_renders_the_line_and_no_disclosure`.**
     Old:
     ```python
         assert "<details" not in body
     ```
     New:
     ```python
         # The condensed honesty notes are <details> too; only the suppressed disclosure is absent.
         assert '<details id="suppressed-candidates"' not in body
     ```
   - **5e. Task 4, same cause, `test_the_caption_is_present_whether_or_not_the_disclosure_is_expanded`.**
     Old:
     ```python
         summary = body[body.index("<summary") : body.index("</summary>")]
     ```
     New:
     ```python
         start = body.index('<details id="suppressed-candidates"')
         summary = body[body.index("<summary", start) : body.index("</summary>", start)]
     ```
   - **5f. Task 5 must keep the instruction text inside ONE `<p>`.** In `_error_state.html` that is `recovery_text`; in `_empty_state.html` it is `body`. `tests/api/test_cold_start_bet_list_recovery.py::_instructional_paragraphs` reads paragraphs as the block boundary.
6. **Owned by Part D (Task 15), not Part C.** `test_bets_page.py::test_the_same_cache_meta_does_not_500_any_other_page_either` iterates `/performance`, `/backtest`, `/insights`, `/betting`. After the merge, those redirect and the test client follows them. The route list should become `("/", "/season", "/track-record", "/how-it-works")`.
7. **Temporary broken link.** The /bets cross-link points at `/track-record#betting-sim` from Task 11 onward. That page only exists from Task 15, so the link 404s on the branch between those two tasks. No test follows it.
8. **The tracker's record line ("18-20-2") and the result strip restate stored counts; they add no figure.** The six figure labels and the three `data-figure-group` wrappers are unchanged. Tracker figure labels and values gain `data-figure-label` / `data-figure-value` hooks, so the tests locate them by markup rather than by a class string other page regions may share.

---

### Task 10: Graded bet outcomes for the result strip

**Files:**
- Modify: `api/services.py` (the `from api.cache import (...)` list near line 44; new methods after `_get_bet_tracker_blocks_uncached`, before `get_cache_meta`)
- Modify: `api/routes/pages.py` (new helper before `_build_bets_context`; one new context key inside it)
- Test: `tests/api/test_bets_page.py` (new tests after `_tracker_sections`; import line 67)

**Interfaces:**
- Consumes: `api.cache.BET_STATUS_LIVE`, `GRADING_STATUS_WIN`, `GRADING_STATUS_LOSS`, `GRADING_STATUS_PUSH` (existing constants); the existing `_cache_get` / `_cache_set` TTL-cache helpers in `api/services.py`.
- Produces:
  - `DataService.get_graded_bet_outcomes(self) -> list[dict[str, Any]]`. Keys are `provenance`, `validation_type`, `season`, `week`, `game_id`, `target`, `grading_status`, one dict per graded LIVE row. Rows are ordered by `provenance, validation_type, season, week, game_id, target`.
  - `api.routes.pages._graded_outcomes_by_class(rows: list[dict[str, Any]]) -> dict[str, list[str]]`. The key is `f"{provenance}:{validation_type}"`; the value is the `grading_status` list in getter order.
  - Context key `graded_outcomes` on `/bets` (page and HX fragment, which share `_build_bets_context`).

- [ ] **Step 1: Write the failing tests**

In `tests/api/test_bets_page.py`, change the import on line 67.

Old:
```python
from backtest.bet_tracker import EmptyTrackerBlock, TrackerBlock, to_tracker_frame
```
New:
```python
from backtest.bet_tracker import (
    EmptyTrackerBlock,
    TrackerBlock,
    aggregate_all_blocks,
    to_tracker_frame,
)
```

Then insert this block immediately AFTER the `_tracker_sections` function (the function whose last line is `    return sections`, around line 2021) and BEFORE `def test_the_tracker_renders_one_section_per_honesty_class_and_never_pools_them`:

```python
# ---------------------------------------------------------------------------
# The result strip's source: the graded outcomes behind each tracker block (redesign Task 10)
# ---------------------------------------------------------------------------
#
# The strip draws one mark per graded bet. It must be drawn from EXACTLY the rows
# backtest.bet_tracker counts into a block -- status live, grading_status win/loss/push -- or it
# would show a different record from the tiles beside it. These tests pin that row set against the
# real aggregator rather than against a transcription of its filter.


def _graded_row(
    game_id: str,
    grading_status: str,
    *,
    pair: tuple[str, str] = _CONTAMINATED,
    week: int = _WEEK,
    target: str = "ou",
) -> dict[str, Any]:
    """One LIVE bet_list row carrying a stored grade, in the shape the grader writes it."""
    row = _live_row(game_id, target, week=week)
    row.update(
        {
            "provenance": pair[0],
            "validation_type": pair[1],
            "grading_status": grading_status,
            "outcome": {"win": True, "loss": False}.get(grading_status),
            "payout_flat": {"win": 0.909, "loss": -1.0, "push": 0.0}.get(
                grading_status
            ),
        }
    )
    return row


def _graded_fixture_rows() -> list[dict[str, Any]]:
    """Graded, ungraded and never-bet rows across two classes and two weeks."""
    return [
        _graded_row("2023_W01_DET@KC", "win"),
        _graded_row("2023_W01_CAR@ATL", "loss"),
        _graded_row("2023_W01_CIN@CLE", "push"),
        # Ungraded: a live bet whose result is not known yet. Not part of any record.
        _graded_row("2023_W01_JAX@IND", "pending"),
        # Never bet: a suppressed candidate. Not part of any record.
        _suppressed_row("2023_W01_SEA@SFO", "ats", "ev_below_floor"),
        _graded_row("2023_W02_AAA@BBB", "win", week=_EMPTY_WEEK),
        _graded_row("2023_W01_DEN@LVR", "win", pair=_FORWARD_CLASS),
    ]


def _graded_cache(tmp_path: Path, name: str) -> Path:
    """A cache whose bet_list carries the graded fixture rows and their schedule."""
    clear_cache()
    rows = _graded_fixture_rows()
    db_path = tmp_path / f"{name}.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        materialize_bet_list(conn, pd.DataFrame(rows))
        materialize_available_bet_weeks(
            conn,
            pd.DataFrame(
                [
                    {"game_id": r["game_id"], "season": _SEASON, "week": r["week"]}
                    for r in rows
                ]
            ),
        )
    finally:
        conn.close()
    return db_path


def test_the_graded_outcomes_are_the_live_graded_rows_in_a_fixed_order(
    tmp_path: Path,
) -> None:
    """Live AND graded only, ordered by class then the four-key tie-break, as stored."""
    db_path = _graded_cache(tmp_path, "graded_order")
    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        rows = DataService(probe).get_graded_bet_outcomes()
    finally:
        probe.close()

    assert [
        (r["provenance"], r["validation_type"], r["game_id"], r["grading_status"])
        for r in rows
    ] == [
        ("backtest_replay", "contaminated", "2023_W01_CAR@ATL", "loss"),
        ("backtest_replay", "contaminated", "2023_W01_CIN@CLE", "push"),
        ("backtest_replay", "contaminated", "2023_W01_DET@KC", "win"),
        ("backtest_replay", "contaminated", "2023_W02_AAA@BBB", "win"),
        ("forward", "forward_realized", "2023_W01_DEN@LVR", "win"),
    ]
    assert set(rows[0]) == {
        "provenance",
        "validation_type",
        "season",
        "week",
        "game_id",
        "target",
        "grading_status",
    }


def test_the_graded_outcomes_count_exactly_what_the_tracker_aggregates(
    tmp_path: Path,
) -> None:
    """Per class, the outcome marks equal the REAL aggregator's wins, losses and pushes."""
    db_path = _graded_cache(tmp_path, "graded_vs_tracker")
    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        outcomes = DataService(probe).get_graded_bet_outcomes()
    finally:
        probe.close()

    blocks = aggregate_all_blocks(pd.DataFrame(_graded_fixture_rows()))
    assert blocks, "the fixture produced no tracker block, so this proves nothing"
    for block in blocks:
        assert isinstance(block, TrackerBlock)
        statuses = [
            r["grading_status"]
            for r in outcomes
            if (r["provenance"], r["validation_type"])
            == (block.provenance, block.validation_type)
        ]
        assert statuses.count("win") == block.wins
        assert statuses.count("loss") == block.losses
        assert statuses.count("push") == block.pushes
        assert len(statuses) == block.bets_graded


def test_the_graded_outcomes_tolerate_a_cache_without_a_bet_list(
    tmp_path: Path,
) -> None:
    """A cache that predates the bet list returns no outcomes rather than raising."""
    clear_cache()
    db_path = tmp_path / "graded_no_table.duckdb"
    conn = _build_bare_cache(db_path)
    try:
        conn.execute("DROP TABLE bet_list")
    finally:
        conn.close()
    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        assert DataService(probe).get_graded_bet_outcomes() == []
    finally:
        probe.close()


def test_the_bets_context_partitions_the_outcomes_by_honesty_class(
    tmp_path: Path,
) -> None:
    """The context keys each class by the same string its tracker section renders."""
    from api.routes.pages import _build_bets_context

    db_path = _graded_cache(tmp_path, "graded_context")
    probe = duckdb.connect(str(db_path), read_only=True)
    try:
        context = _build_bets_context(DataService(probe), _SEASON, _WEEK, None)  # pyright: ignore[reportArgumentType]
    finally:
        probe.close()

    assert context["graded_outcomes"] == {
        "backtest_replay:contaminated": ["loss", "push", "win", "win"],
        "forward:forward_realized": ["win"],
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:
```bash
uv run pytest "tests/api/test_bets_page.py::test_the_graded_outcomes_are_the_live_graded_rows_in_a_fixed_order" "tests/api/test_bets_page.py::test_the_graded_outcomes_count_exactly_what_the_tracker_aggregates" "tests/api/test_bets_page.py::test_the_graded_outcomes_tolerate_a_cache_without_a_bet_list" "tests/api/test_bets_page.py::test_the_bets_context_partitions_the_outcomes_by_honesty_class" -v
```
Expected: 4 FAIL. The first three fail with `AttributeError: 'DataService' object has no attribute 'get_graded_bet_outcomes'`; the fourth fails with `KeyError: 'graded_outcomes'`.

- [ ] **Step 3: Add the getter**

In `api/services.py`, extend the `api.cache` import.

Old:
```python
    CURRENT_SLATE_KEY,
    PREDICTIONS_TABLE_COLUMNS,
```
New:
```python
    CURRENT_SLATE_KEY,
    GRADING_STATUS_LOSS,
    GRADING_STATUS_PUSH,
    GRADING_STATUS_WIN,
    PREDICTIONS_TABLE_COLUMNS,
```

Then insert the two methods immediately before `get_cache_meta`.

Old:
```python
    def get_cache_meta(self) -> dict[str, Any]:
```
New:
```python
    def get_graded_bet_outcomes(self) -> list[dict[str, Any]]:
        """Return every GRADED live bet's stored outcome, for the /bets result strip.

        The row set is exactly the one ``backtest.bet_tracker.aggregate_by_provenance`` counts
        into ``bet_tracker_blocks``: ``status`` live and ``grading_status`` one of win / loss /
        push. A pending bet is ungraded and a suppressed candidate was never bet, so neither is
        here. A read, not a computation: each stored ``grading_status`` is returned as written,
        with no count, rate or SQL aggregate (UIAP-01); the template draws one mark per row.

        Ordered by class and then by the bet list's four-key tie-break, so the strip reads in
        week order (game id order within a week) and two requests render it identically.
        """
        key = ("graded_bet_outcomes",)
        cached = _cache_get(key)
        if cached is not None:
            return cached
        result = self._get_graded_bet_outcomes_uncached()
        _cache_set(key, result)
        return copy.deepcopy(result)

    def _get_graded_bet_outcomes_uncached(self) -> list[dict[str, Any]]:
        try:
            result = self._conn.execute(
                "SELECT provenance, validation_type, season, week, game_id, target, "
                "grading_status FROM bet_list "
                "WHERE status = ? AND grading_status IN (?, ?, ?) "
                "ORDER BY provenance, validation_type, season, week, game_id, target",
                [
                    BET_STATUS_LIVE,
                    GRADING_STATUS_WIN,
                    GRADING_STATUS_LOSS,
                    GRADING_STATUS_PUSH,
                ],
            )
        except duckdb.Error:
            # The cache predates Phase 31 (no bet_list table): no strip, never a 500.
            logger.warning("bet_list table not available in cache")
            return []
        columns = [desc[0] for desc in result.description]
        return [dict(zip(columns, row)) for row in result.fetchall()]

    def get_cache_meta(self) -> dict[str, Any]:
```

- [ ] **Step 4: Add the partition helper and the context key**

In `api/routes/pages.py`, insert the helper immediately before `_build_bets_context`.

Old:
```python
def _build_bets_context(
    service: DataService,
```
New:
```python
def _graded_outcomes_by_class(rows: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Partition the stored graded outcomes by honesty class, for the /bets result strip.

    A partition, not an aggregate: every row lands in exactly one list, in the getter's order,
    and nothing is counted, summed or derived (UIAP-01). The key is the tracker section's own
    ``data-tracker-block`` value, so the template finds a class by the string it already renders.
    """
    by_class: dict[str, list[str]] = {}
    for row in rows:
        key = f"{row['provenance']}:{row['validation_type']}"
        by_class.setdefault(key, []).append(row["grading_status"])
    return by_class


def _build_bets_context(
    service: DataService,
```

Then add the context key right after `tracker_blocks`.

Old:
```python
        # its sections by matching the two stored labels; it never pools two classes into one
        # figure, because the pooled figure does not exist to render.
        "tracker_blocks": tracker_blocks,
```
New:
```python
        # its sections by matching the two stored labels; it never pools two classes into one
        # figure, because the pooled figure does not exist to render.
        "tracker_blocks": tracker_blocks,
        # The graded rows behind those blocks, partitioned by the same class key the tracker
        # sections render, for the result strip (redesign spec 7.3). A read and a partition, not
        # a count: the template draws the strip only when it agrees with the stored block.
        "graded_outcomes": _graded_outcomes_by_class(
            service.get_graded_bet_outcomes()
        ),
```

- [ ] **Step 5: Run the new tests and the structural guards**

Run:
```bash
uv run pytest "tests/api/test_bets_page.py::test_the_graded_outcomes_are_the_live_graded_rows_in_a_fixed_order" "tests/api/test_bets_page.py::test_the_graded_outcomes_count_exactly_what_the_tracker_aggregates" "tests/api/test_bets_page.py::test_the_graded_outcomes_tolerate_a_cache_without_a_bet_list" "tests/api/test_bets_page.py::test_the_bets_context_partitions_the_outcomes_by_honesty_class" "tests/api/test_bets_page.py::test_bets_navigation_reads_the_schedule_getters_and_not_the_predictions_ones" "tests/api/test_bets_page.py::test_the_page_renders_the_stored_aggregate_and_computes_nothing" "tests/api/test_bets_page.py::test_the_refusal_branch_reads_the_schedule_locks_and_the_week_coverage" "tests/api/test_bets_page.py::test_the_structural_guard_is_not_vacuous" -v
uv run pytest tests/api/test_import_guard_bets.py -v
```
Expected: all PASS. The last four tests and the import guard prove that the new read adds no `backtest` import and does not touch the refusal branch's getter set.

- [ ] **Step 6: Lint and type-check**

Run:
```bash
uv run ruff check api/services.py api/routes/pages.py tests/api/test_bets_page.py
uv run ruff format api/services.py api/routes/pages.py tests/api/test_bets_page.py
uv run pyright api/services.py api/routes/pages.py tests/api/test_bets_page.py
```
Expected: `All checks passed!`, files left formatted, and `0 errors` in `api/services.py` and `api/routes/pages.py`. In `tests/api/test_bets_page.py`, no pyright error on a line this task wrote; an older one predates the branch -- note it in the task report and leave it.

- [ ] **Step 7: Commit**

```bash
git add api/services.py api/routes/pages.py tests/api/test_bets_page.py
git commit -m "$(cat <<'EOF'
feat(redesign): read the graded outcomes behind each /bets tracker block

DataService.get_graded_bet_outcomes returns the live graded rows the tracker
aggregates (status live, grading_status win/loss/push) in a fixed order, and
/bets partitions them by honesty class for the result strip. A read and a
partition; no count or rate in the request path.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 11: Bets page restyle -- header, bet slips, notices, states, suppressed disclosure

**Files:**
- Modify (full rewrite): `web/templates/pages/bets.html`. The tracker region is carried over verbatim in this task and restyled in Task 12.
- Modify: `tests/api/test_bets_page.py`
- Modify: `tests/api/test_export.py`
- Verify only (no edits needed): `tests/api/test_cold_start_bet_list_recovery.py`, `tests/api/test_cache_swap_recovery.py`. Both key on the string "Stake (units)", which each bet slip keeps as its stake label.
- Regenerate: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes:
  - Task 1 classes: `display`, `label`, `num`, `tag`, `section-head`, `unskew`, `bet-slip`, `skew-control`, `skew-control-active`, tokens `bg-panel`, `bg-panel-2`, `text-fg`, `text-muted`, `text-dim`, `text-accent`, `bg-accent`, `text-ink`, `border-line`, `bg-line`, `divide-line`.
  - Task 2 Jinja globals: `team_colors(abbr) -> TeamColors(bg, fg)`, `team_nickname(abbr) -> str`.
  - Task 3 macro: `bc.team_block(abbr, bg, fg, large=false)` from `components/_broadcast.html`.
  - Task 8 partial `components/_bet_pick.html` (imported as `bp` INSIDE `{% block bets_content %}`): `bp.pick_label(bet)` and `bp.side_line(bet, fmt)` -- the ONLY place the spread sign rule lives; this template never redefines it.
  - Tasks 4/5 partials, included with their existing parameters: `_not_advice_banner.html`, `_old_rule_label.html`, `_ev_band_badge.html`, `_provenance_badge.html`, `_empty_state.html`, `_error_state.html`, `_export_buttons.html`, `_loading_skeleton.html`, `_week_selector.html`.
- Produces: markup hooks used by tests and by Task 12:
  - `<li class="bet-slip ...">` per live bet
  - `data-bet-type` on every bet-type label (slip `<span>` and suppressed-table `<td>`)
  - `data-slip-line`, `data-slip-stake`, `data-slip-provenance` on the slip
  - `data-reason-cell` on the suppressed reason `<td>`
  - `data-suppressed-reason-counts` on the summary's count row
  - the cross-link `href="/track-record#betting-sim"`.

- [ ] **Step 1: Update the existing assertions that pin the old table markup, and add the new tests**

These edits describe the NEW markup, so they fail until Step 3. Apply each old -> new edit to `tests/api/test_bets_page.py`:

(a) In `test_served_equals_selector_top_row`.

Old:
```python
    assert (
        f'class="px-4 py-3 text-right stat-num whitespace-nowrap">{expected_stake}<'
        in body
    )
```
New:
```python
    assert f"data-slip-stake>{expected_stake}<" in body
```

(b) In `test_raw_target_codes_are_never_rendered`.

Old:
```python
    """Bet type renders Winner / Spread / Totals, never the raw wp / ats / ou codes."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert ">Totals</td>" in body
```
New:
```python
    """Bet type renders Winner / Spread / Totals, never the raw wp / ats / ou codes."""
    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    assert "data-bet-type>Totals<" in body
    assert "data-bet-type>ou<" not in body
```

(c) In `test_total_but_no_moneyline_yields_one_live_and_one_suppressed_row`.

Old:
```python
    assert ">Totals</td>" in body
    assert ">Winner</td>" in body
```
New:
```python
    assert "data-bet-type>Totals<" in body
    assert "data-bet-type>Winner<" in body
```

(d) In `test_a_spread_pick_shows_the_picked_teams_own_line`.

Old:
```python
    # The Line column carries the same side-perspective number as the pick.
    assert 'whitespace-nowrap">+5.5</td>' in body
    assert 'whitespace-nowrap">-1.5</td>' in body
```
New:
```python
    # The slip's Line figure carries the same side-perspective number as the pick.
    assert "data-slip-line>+5.5<" in body
    assert "data-slip-line>-1.5<" in body
```

(e) The blank-reason constant.

Old:
```python
_BLANK_REASON_CELL = (
    '<td class="px-4 py-3 text-left text-gray-700 whitespace-nowrap"></td>'
)
```
New:
```python
# The reason cell is located by its data hook rather than by a class string, so a restyle cannot
# turn this "must not appear" check vacuous by changing the classes it spelled.
_BLANK_REASON_CELL = "data-reason-cell></td>"
```

(f) In `test_the_badge_renders_on_every_displayed_row`.

Old:
```python
    assert body.count("Provenance</th>") == 2, (
        "the live table and the suppressed table do not both carry the provenance column"
    )
```
New:
```python
    # The live list is a column of bet slips and the suppressed list a table: each carries its
    # own provenance slot, so both halves still label every displayed row.
    assert body.count("Provenance</th>") == 1, (
        "the suppressed table does not carry its provenance column"
    )
    assert body.count("data-slip-provenance") == 1, (
        "the live bet slip does not carry its provenance slot"
    )
```

(g) Add these two tests immediately after `test_a_spread_pick_shows_the_picked_teams_own_line`:

```python
def test_each_live_bet_renders_one_ranked_slip(
    bets_client: TestClient, selected_records: list[dict[str, Any]]
) -> None:
    """Redesign: one bet slip per live bet, carrying both teams and its line capture time.

    The slip replaces the ten-column table. The per-row "Line as of" the table showed in its own
    column is now visible text on the slip, so the reader can still check each row's capture time
    against the per-game lock rule stated at the foot of the page.
    """
    from api.presentation import team_nickname

    body = bets_client.get(f"/bets?season={_SEASON}&week={_WEEK}").text
    count = len(selected_records)

    assert body.count('<li class="bet-slip') == count
    assert f"Line as of {_SNAPSHOT_TS}" in body
    # The header meta, with its row count -- not the <ol>'s aria-label, which also says
    # "ranked by expected value" and would satisfy a bare substring check on its own.
    noun = "bet" if count == 1 else "bets"
    assert f"ranked by expected value &middot; {count} {noun}</span>" in body
    for record in selected_records:
        away, home = record["game_id"].split("_")[-1].split("@")
        assert team_nickname(away) in body
        assert team_nickname(home) in body


def test_the_suppressed_summary_counts_each_reason_while_collapsed(
    tmp_path: Path,
) -> None:
    """Redesign: the closed disclosure's summary names each reason with its row count.

    Each count is the length of the SAME group the disclosure body renders under that reason, so
    the summary and the body cannot disagree. It is a count of rows on the page, not a metric.
    """
    rows = [
        _suppressed_row("2023_W01_AAA@BBB", "ou", "ev_below_floor"),
        _suppressed_row("2023_W01_CCC@DDD", "ou", "ev_below_floor"),
        _suppressed_row("2023_W01_EEE@FFF", "ats", "stale_line"),
    ]
    with _client_for(tmp_path, rows, "summary_counts") as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    start = body.index('<details id="suppressed-candidates"')
    summary = body[body.index("<summary", start) : body.index("</summary>", start)]
    assert "data-suppressed-reason-counts" in summary
    assert 'Expected value below the floor <b class="num">2</b>' in summary
    assert 'Line captured after the lock <b class="num">1</b>' in summary
    # The body's per-reason headings carry the same counts.
    assert "Expected value below the floor (2)" in body
    assert "Line captured after the lock (1)" in body
```

Then update `tests/api/test_export.py`.

(h) Replace the whole `_page_row_identifiers` function.

Old:
```python
def _page_row_identifiers(body: str) -> list[tuple[str, str]]:
    """Every (game_id, bet-type label) the page renders, in document order.

    Both tables are read: the live list and the collapsed suppressed disclosure, whose rows are in
    the document even while closed. That is the set an export must equal.
    """
    return [
        (match.group(1), match.group(2))
        for match in re.finditer(
            r'<a href="/games/([^"]+)"[^>]*>[^<]*</a>\s*</td>\s*'
            r'<td class="px-4 py-3 text-left text-gray-700 whitespace-nowrap">([^<]+)</td>',
            body,
        )
    ]
```
New:
```python
def _page_row_identifiers(body: str) -> list[tuple[str, str]]:
    """Every (game_id, bet-type label) the page renders, in document order.

    Both halves are read: the live bet slips and the collapsed suppressed disclosure's table, whose
    rows are in the document even while closed. That is the set an export must equal. Each row's
    matchup link is paired with the first ``data-bet-type`` label after it and before the next
    matchup link -- a data hook both halves carry, so a restyle cannot make this read zero rows.
    """
    return [
        (match.group(1), match.group(2))
        for match in re.finditer(
            r'<a href="/games/([^"]+)"[^>]*>[^<]*</a>'
            r'(?:(?!<a href="/games/).)*?'
            r"data-bet-type[^>]*>([^<]+)<",
            body,
            flags=re.S,
        )
    ]
```

(i) In `test_the_page_offers_both_exports_and_the_cross_link`.

Old:
```python
    assert _BETS_CROSS_LINK in body
    assert 'href="/betting"' in body
```
New:
```python
    assert _BETS_CROSS_LINK in body
    # /betting was merged into Track Record's betting-simulation section (redesign spec 7.5).
    assert 'href="/track-record#betting-sim"' in body
```

- [ ] **Step 2: Run the changed tests to verify they fail**

Run:
```bash
uv run pytest "tests/api/test_bets_page.py::test_served_equals_selector_top_row" "tests/api/test_bets_page.py::test_raw_target_codes_are_never_rendered" "tests/api/test_bets_page.py::test_total_but_no_moneyline_yields_one_live_and_one_suppressed_row" "tests/api/test_bets_page.py::test_a_spread_pick_shows_the_picked_teams_own_line" "tests/api/test_bets_page.py::test_the_badge_renders_on_every_displayed_row" "tests/api/test_bets_page.py::test_each_live_bet_renders_one_ranked_slip" "tests/api/test_bets_page.py::test_the_suppressed_summary_counts_each_reason_while_collapsed" -v
uv run pytest "tests/api/test_export.py::test_the_exported_row_set_equals_the_pages_live_and_suppressed_rows" "tests/api/test_export.py::test_exported_row_order_equals_the_pages_order_position_by_position" "tests/api/test_export.py::test_the_page_offers_both_exports_and_the_cross_link" -v
```
Expected: FAIL. The new hooks (`data-slip-stake`, `data-bet-type`, `bet-slip`, `data-suppressed-reason-counts`, the new cross-link href) are not in the page yet. The export row-set and row-order tests fail too, though not necessarily with the same message: the row-set test can report that the page rendered no identifiable rows, while the order test fails its position-by-position comparison. Either failure is the expected red.

- [ ] **Step 3: Rewrite `web/templates/pages/bets.html`**

Replace the entire file with the following. Everything from the tracker banner comment (`Rank 3: the realized-versus-expected tracker`) down to the tracker's closing `{% endif %}` is byte-for-byte the current text; Task 12 restyles it.

```jinja
{% extends "base.html" %}
{% block title %}Weekly Bet List{% endblock %}
{% block content %}

{# ------------------------------------------------------------------------ #}
{# Rank 0: the not-advice banner.                                            #}
{# It is included HERE -- above every conditional branch and OUTSIDE the      #}
{# bets_content swap target -- so it is present in the first byte of every    #}
{# response, in every empty state, and in the stale-cache hard-block          #}
{# (UI-SPEC E1). Do not move it inside the block. The redesign condenses it   #}
{# to a one-line note whose "Why?" expands to the full original text; the     #}
{# text stays in the HTML either way.                                         #}
{# ------------------------------------------------------------------------ #}
{% include "components/_not_advice_banner.html" %}

{# The heading names the PAGE, not the week: it sits OUTSIDE the #bets-content swap target, so a
   week in it would go stale on the first week change. The week is named by the selector beside it
   and by the "Live bets -- <season> Week <n>" tag inside the swap target. #}
<header class="mt-2 mb-5 flex flex-wrap items-end gap-x-6 gap-y-2">
  <h1 class="display text-4xl md:text-5xl leading-none text-fg">Weekly Bet List</h1>
  <p class="text-sm text-muted max-w-2xl pb-1">Every scheduled game is evaluated for all three bet types. Bets that clear the pre-registered expected-value floor are listed here; every candidate that did not is listed below with its reason.</p>
</header>

{# ------------------------------------------------------------------------ #}
{# The week selector, OUTSIDE the swap target so it stays in place and       #}
{# interactive while the skeleton fills the target (UI-SPEC E5 loading).     #}
{#                                                                          #}
{# It is the SAME partial the This Week page uses, parameterised in place    #}
{# (D31-26) -- not a fork. Its options come from the SCHEDULE-derived        #}
{# available_bet_weeks / bet_seasons, never from get_available_weeks (which  #}
{# reads predictions) or get_available_seasons (which reads backtest_        #}
{# metrics): a scheduled week with no prediction row is exactly the week a   #}
{# reader most wants to check, and it would be ABSENT from a predictions-    #}
{# derived control rather than selectable and resolving to an explanatory    #}
{# empty state (REVIEW-NAV, T-31-74c).                                        #}
{#                                                                          #}
{# ws_extra_params={} because /bets has no sort control; the sort parameter  #}
{# is OMITTED rather than emitted empty.                                     #}
{#                                                                          #}
{# ws_timeout=10000 is what makes the selector's hx-on::timeout handler      #}
{# REACHABLE (G-31-127). htmx gives its XMLHttpRequest a timeout of zero     #}
{# unless one is supplied, and a zero timeout means "no timeout", so the     #}
{# event the handler is bound to never fires and a wedged request leaves the #}
{# previous week's rows under the new week's selection forever. /bets is the #}
{# only page that passes ws_failure_template, so it is the only page that    #}
{# passes a timeout -- see the partial's own comment for why that pairing is #}
{# deliberate rather than a global htmx.config.timeout.                      #}
{#                                                                          #}
{# WHY TEN SECONDS. Every htmx endpoint this app serves answers in under     #}
{# 40 ms, because nothing is computed in the request path (UIAP-01) -- the   #}
{# /bets fragment itself measured 7.5-15.5 ms. Ten seconds is roughly 250x   #}
{# the measured worst case, so it cannot plausibly abort a real request. It  #}
{# is deliberately not tighter: the failure state REPLACES the whole target  #}
{# and its only retry is a full navigation, so the cost of giving up early   #}
{# on a request that would have arrived is higher than the cost of waiting.  #}
{#                                                                          #}
{# ws_sync=true closes the SECOND defect the timeout diagnosis exposed but   #}
{# did not cause (E10): while a request is in flight htmx QUEUES a second    #}
{# one from the same element and leaves the element interactive, so the      #}
{# selector could be moved to a week the content does not describe -- the    #}
{# mismatch widened with every further interaction and only a full page      #}
{# reload recovered. The timeout above bounds how LONG that window lasts;    #}
{# this stops the page lying DURING it. It emits hx-disabled-elt on the      #}
{# control that is asking and hx-sync on the swap target with the replace    #}
{# strategy, so a request from any of the four aborts the in-flight one      #}
{# instead of queueing behind it.                                            #}
{#                                                                          #}
{# OWNER-RULED 2026-09-10 (plan 31-24 Task 1, option "ship"): ship both      #}
{# halves. The debugger flagged this as a separate decision and declined to  #}
{# bundle it, so it was ruled on rather than absorbed. The accepted cost is  #}
{# on the happy path -- a rapid second week change now cancels the first     #}
{# request rather than queueing it -- which is safe here only because every  #}
{# htmx endpoint this app serves answers in under 40 ms (E11), so the        #}
{# cancelled request had almost certainly already completed. /bets is the    #}
{# ONLY page that opts in; the parameter defaults to omitted and the six     #}
{# other pages keep htmx's shipped queueing behaviour.                       #}
{# ------------------------------------------------------------------------ #}
<div class="flex flex-wrap items-center justify-between gap-4 mb-6">
  {% with available_weeks=available_bet_weeks,
          available_seasons=bet_seasons,
          ws_endpoint="/bets",
          ws_target="#bets-content",
          ws_include_season="",
          ws_include_week="[name='season']",
          ws_extra_params={},
          ws_indicator="#bets-loading",
          ws_failure_template="bets-failure-template",
          ws_timeout=10000,
          ws_sync=true %}
    {% include "components/_week_selector.html" %}
  {% endwith %}
</div>

{# ------------------------------------------------------------------------ #}
{# The failed-fragment markup (owner decision, UI-SPEC E5 error).            #}
{#                                                                          #}
{# htmx's default on a failed request is to leave the target untouched and   #}
{# hide the indicator -- which on this page means one week's rows sitting    #}
{# under another week's heading. The selector's three failure handlers copy  #}
{# this template into the swap target instead and revert both controls to    #}
{# the week actually being shown.                                            #}
{#                                                                          #}
{# This is DISTINCT from the stale-cache hard-block above, which is a        #}
{# server-rendered 200 that refuses a list it HAS. This one is a request     #}
{# that never delivered a list at all.                                       #}
{#                                                                          #}
{# The retry affordance is a plain link, not an hx-get: markup inserted by   #}
{# hand into innerHTML is not processed by htmx, so an htmx control here     #}
{# would render as a dead button. A full navigation genuinely retries.       #}
{# ------------------------------------------------------------------------ #}
<template id="bets-failure-template">
  {% with message="This week's list could not be loaded",
          recovery_text="The request failed before any rows arrived. Nothing is shown rather than the previous week's rows under this week's heading. The week selector has been reset to the week actually displayed." %}
    {% include "components/_error_state.html" %}
  {% endwith %}
  <div class="mt-4 text-center">
    <a href="/bets?season={{ current_season if current_season is not none else '' }}&amp;week={{ current_week if current_week is not none else '' }}" class="skew-control skew-control-active inline-flex items-center px-4 min-h-[44px]"><span class="unskew">Retry this week</span></a>
  </div>
</template>

{# HTMX swap target: the fragment route returns exactly this block. #}
<div id="bets-content">
{% block bets_content %}
{# Imported INSIDE the block, not at the top of the file: an HX-Request renders ONLY this block
   (jinja2-fragments), and a child template's top-level imports do not run in a block render. #}
{% import "components/_broadcast.html" as bc %}
{% import "components/_bet_pick.html" as bp %}

{# The swap indicator. It occupies the swap target during a week change so the
   swap causes no layout shift, mirroring the shipped _betting_scope_toggle.html
   spinner pattern rather than inventing a second one (UI-SPEC E2 loading). #}
<div id="bets-loading" class="htmx-indicator">
  {% with variant="table" %}
    {% include "components/_loading_skeleton.html" %}
  {% endwith %}
</div>

<section class="mb-8" hx-indicator="#bets-loading">
  <div class="section-head mb-3">
    <h2 class="tag"><span class="unskew">Live bets{% if current_season and current_week %} -- {{ current_season }} Week {{ current_week }}{% endif %}</span></h2>
    <span class="flex-1 h-px bg-line" aria-hidden="true"></span>
    {# A count of the rows on this page, like "Suppressed candidates (N)". No stake total: summing
       the stakes here would be a request-path aggregate (UIAP-01). #}
    {% if bet_list_available and not bets_blocked and bets %}
    <span class="text-xs text-muted whitespace-nowrap">ranked by expected value &middot; {{ bets|length }} {{ "bet" if bets|length == 1 else "bets" }}</span>
    {% endif %}
  </div>

  {# Old-rule label for the selected week (R16 / D33.2-07): a 2021-2025 week is a replay
     reconstructed by the old models. The route scopes it to the selected season. #}
  {% with scope=week_old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}

  {# The week's games judged against their own locks (33.2 review C2 CR-02), as matchup labels. #}
  {% set missing_labels = [] %}
  {% for _game_id in missing_games or [] %}{% set _ = missing_labels.append(_game_id.split('_')[-1].replace('@', ' @ ')) %}{% endfor %}
  {% set pending_labels = [] %}
  {% for _game_id in pending_games or [] %}{% set _ = pending_labels.append(_game_id.split('_')[-1].replace('@', ' @ ')) %}{% endfor %}

  {# A week built in part: the games that passed their lock with NO list are named above whatever
     was built, never silently left out (33.2 review C2 CR-02). Neutral, like every other
     non-tracker surface on this page -- colour is tracker-only here. Paragraphs only inside this
     box: a test reads the missing games up to its FIRST closing div. #}
  {% if bet_list_available and not bets_blocked and missing_labels %}
  <div class="mb-4 border border-line bg-panel px-4 py-3 text-sm max-w-3xl" data-missing-games>
    <p class="font-semibold text-fg">No list for {{ missing_labels | length }} game{{ "" if missing_labels | length == 1 else "s" }} past {{ "its" if missing_labels | length == 1 else "their" }} lock</p>
    <p class="mt-1 text-muted">{{ missing_labels | join(", ") }}. Each game's list is built the day before its kickoff, before its 6:00 PM Eastern lock; these were not, so they carry no recommendation. The games below are unaffected.</p>
  </div>
  {% endif %}

  {# The four distinct non-happy renders (SPEC R5, UI-SPEC E2 empty / E7). Order matters:
     a missing TABLE is checked first because it is the only one of the four that names an action
     the reader can take, and reporting it as "no bets cleared the floor" would be a claim about
     the models made from the absence of a table. Never a blank page and never a 500. #}
  {% if not bet_list_available %}
    {% with heading="Bet list not built yet",
            body='The bet-list cache table does not exist. Build the list for this week with "uv run python scripts/generate_bet_list.py", then load it into the cache with "uv run python scripts/populate_cache.py", then reload this page. The second command only COPIES what the first produces, so running it on its own creates the table with no rows in it.',
            action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% elif bets_blocked %}
    {# The hard-block (D31-27), now judged game by game (33.2 review C2 CR-02): games of this week
       have passed their own lock and NONE has a list in the cache -- the failed-insertion state.
       Rank 1 of the section the list would have held, so the absence is STRUCTURAL. The recovery
       text names the games and repeats both timestamps so the claim can be checked rather than
       taken. The suppressed section is withheld with it (UI-SPEC E3 error, owner decision). #}
    {% with message="This week's list is missing -- its locked games have no list in the cache",
            recovery_text="Each game's list is built the day before its kickoff, before its 6:00 PM Eastern lock. These games have passed that lock with no list in the cache: " ~ (missing_labels | join(", ")) ~ '. Past weeks below are unaffected and remain readable. To restore this section, rebuild the list with "uv run python scripts/generate_bet_list.py" and then load it into the cache with "uv run python scripts/populate_cache.py". The second command only COPIES what the first produces, so running it on its own leaves this section withheld. The running server picks the rebuilt cache up on the next request, so no restart is needed. Cache last populated: ' ~ (bet_list_populated_at or "not recorded") ~ "; latest game lock this week: " ~ (bet_week_freeze or "not recorded") ~ "." %}
      {% include "components/_error_state.html" %}
    {% endwith %}
  {% elif week_not_evaluated %}
    {# Every game of this week is still ahead of its lock and none has a list yet: NOT EVALUATED,
       which is neither a refusal nor a result (33.2 review C2 WR-01). #}
    {% with heading="Not evaluated yet",
            body="No game of this week has reached its lock. Each game's list is built by the daily run the day before its kickoff, before its 6:00 PM Eastern lock, and appears here once it is.",
            action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% elif current_week is none %}
    {% with heading="No current week",
            body="The season has not started or has finished, so there is no forward list. Use the week selector to read the list for any completed week.",
            action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% elif not bets %}
    {# The claim is scoped to what was evaluated: with a game missing or still ahead of its lock,
       "every scheduled game" would be false (33.2 review C2 CR-02). #}
    {% with heading="No bets cleared the floor this week",
            body=("Every game with a list so far was checked" if (missing_labels or pending_labels) else "Every scheduled game was evaluated") ~ " for all three bet types and none cleared the pre-registered expected-value floor. An empty list is a result, not a failure or an outage. The full record of what was evaluated and why each candidate was declined is in the suppressed section below.",
            action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% else %}
  {# Bet-type label vocabulary. The raw wp / ats / ou codes are NEVER rendered --
     this is the exact wording betting.html already uses. #}
  {% set BET_TYPE_LABELS = {"wp": "Winner", "ats": "Spread", "ou": "Totals"} %}
  {# The spread sign rule (a pick's own line, as a sportsbook quotes it) lives ONLY in
     components/_bet_pick.html (Task 8), imported as bp at the top of this block. #}
  {# One RANKED bet slip per live bet (redesign spec 7.3). Order is fixed by the cache query
     (per-bet EV descending, ties broken season, week, game_id, target). No sort control is
     offered, so a reader cannot re-rank away from the pre-registered order (D31-28).

     R8 requires the two honesty labels on every DISPLAYED and exported row, so every slip carries
     its own evidence chip; a tracker-level label alone would leave an individual row unattributed.

     LAYOUT: two columns below lg (rank, then the slip's content stacked), seven on lg+. The seven
     tracks and their gaps need ~820px, more than the 720px a 768px tablet leaves, so the stacked
     layout holds until lg (Decision 14; spec 10: no sideways scroll). The figure group is
     display:contents on lg+ so its three figures become grid columns there and a three-up row
     below it. Tailwind utilities sit in the utilities layer, so they win over the
     .bet-slip component defaults from Task 1. #}
  <ol class="space-y-2" aria-label="Live bets, ranked by expected value">
    {% for bet in bets %}
    {%- set matchup = bet.game_id.split('_')[-1] -%}
    {%- set away_team, home_team = matchup.split('@') if '@' in matchup else ("Away", "Home") -%}
    {%- set away_colors = team_colors(away_team) -%}
    {%- set home_colors = team_colors(home_team) -%}
    <li class="bet-slip grid grid-cols-[2.5rem_minmax(0,1fr)] lg:grid-cols-[3rem_12rem_minmax(0,1fr)_5rem_6rem_6rem_12rem] items-center gap-x-4 gap-y-2 py-3 pr-4">
      <span class="display text-3xl text-dim text-center row-span-4 lg:row-span-1">{{ loop.index }}</span>
      <div class="flex flex-col gap-1 min-w-0">
        <div class="flex items-center gap-2">{{ bc.team_block(away_team, away_colors.bg, away_colors.fg) }}<span class="label text-fg truncate">{{ team_nickname(away_team) }}</span></div>
        <div class="flex items-center gap-2">{{ bc.team_block(home_team, home_colors.bg, home_colors.fg) }}<span class="label text-fg truncate">{{ team_nickname(home_team) }}</span></div>
      </div>
      <div class="min-w-0">
        <a href="/games/{{ bet.game_id }}" class="text-xs text-muted hover:text-accent">{{ bet.game_id.split('_')[-1].replace('@', ' @ ') }}</a>
        <span class="label block" data-bet-type>{{ BET_TYPE_LABELS.get(bet.target, bet.target) }}</span>
        {# The pick names the TEAM, never the raw side code. The two teams come from the game
           id's AWAY@HOME matchup; an id without one falls back to the words. #}
        <p class="display text-2xl md:text-3xl leading-none text-accent mt-1 whitespace-nowrap">{{ bp.pick_label(bet) }}</p>
      </div>
      <div class="col-start-2 lg:col-start-auto grid grid-cols-3 gap-3 lg:contents">
        <div class="lg:text-right">
          <span class="label block" title="A spread line is the picked team's own line, as a sportsbook quotes it">Line</span>
          <span class="num text-fg whitespace-nowrap" data-slip-line>
            {%- if bet.line is none -%}--
            {%- elif bet.target == "ats" -%}{{ bp.side_line(bet, "%+.1f") }}
            {%- else -%}{{ "%.1f"|format(bet.line) }}
            {%- endif -%}
          </span>
        </div>
        {# per_bet_ev is stored as a FRACTION of stake; the x100 below is a unit
           render (fraction -> percent) under an "EV %" label, not a derivation.
           No metric is recomputed on the request path (UIAP-01). #}
        <div class="lg:text-right">
          <span class="label block">EV %</span>
          <span class="num font-bold text-fg whitespace-nowrap">{{ "%+.2f"|format(bet.per_bet_ev * 100) }}%</span>
        </div>
        <div class="lg:text-right">
          <span class="label block" title="1 unit = 1% of a notional bankroll">Stake (units)</span>
          <span class="num font-bold text-fg whitespace-nowrap" data-slip-stake>{{ "%.2f"|format(bet.stake_units) }}</span>
        </div>
      </div>
      <div class="col-start-2 lg:col-start-auto flex flex-wrap lg:flex-col items-start lg:items-end gap-1.5">
        {% with band=bet.ev_tier %}
          {% include "components/_ev_band_badge.html" %}
        {% endwith %}
        <span data-slip-provenance>
          {%- with provenance=bet.provenance, validation_type=bet.validation_type -%}
            {% include "components/_provenance_badge.html" %}
          {%- endwith -%}
        </span>
        {# The per-row capture time, visible, so the per-game lock rule stated at the foot of the
           page can be checked row by row rather than taken. #}
        <span class="num text-xs text-muted whitespace-nowrap" title="Line as of {{ bet.snapshot_ts or '--' }}">Line as of {{ bet.snapshot_ts or "--" }}</span>
      </div>
    </li>
    {% endfor %}
  </ol>
  <p class="text-xs text-muted mt-3 max-w-2xl">Stakes are quarter-Kelly, capped at 5% of bankroll per bet and 10% across all bets in a week pooled over all three bet types, then de-weighted where bets on the same game or the same side are correlated. Order is by expected value, not by stake -- a capped bet can outrank a larger one.</p>
  {% endif %}
  {# Games still ahead of their lock, in a week that has something to show: not evaluated YET. #}
  {% if bet_list_available and not bets_blocked and not week_not_evaluated and pending_labels %}
  <p class="text-xs text-muted mt-3 max-w-2xl" data-pending-games>Not evaluated yet: {{ pending_labels | join(", ") }}. Each game's list is built by the daily run the day before its kickoff, before its 6:00 PM Eastern lock.</p>
  {% endif %}
</section>

{# The suppressed section is WITHHELD in three states, each for its own reason:
   - no bet_list table: there is no record to show, only an action to take;
   - the stale-cache hard-block: every suppression reason is a verdict computed from the same
     cache the block just declared out of date, so showing them would assert a current record of
     what was declined that is exactly as stale as the list being refused (UI-SPEC E3 error);
   - no current week: season and week are both None, so the getter carries NO week filter and
     would dump every suppressed row in the cache under a single week's heading;
   - a week not evaluated yet: nothing was evaluated, so "no candidate was suppressed" would be
     a claim about an evaluation that has not happened (33.2 review C2 WR-01). #}
{% if bet_list_available and not bets_blocked and not week_not_evaluated and current_week is not none %}
{# ------------------------------------------------------------------------ #}
{# Rank 3: the suppressed-candidates disclosure (SPEC R6, UI-SPEC E3).       #}
{#                                                                          #}
{# Every scheduled game times every registered target yields exactly one     #}
{# record, live or suppressed -- the suppressed half is READ from the cache  #}
{# (DataService.get_suppressed_bets), never derived from the live list.      #}
{#                                                                          #}
{# It is a NATIVE <details> disclosure and deliberately not a scripted       #}
{# toggle: native disclosure is keyboard-accessible and screen-reader-       #}
{# announced for free, needs no script, and -- the load-bearing reason --    #}
{# keeps the suppressed rows IN THE DOCUMENT and in the HTML export even     #}
{# while collapsed, so the "never silently dropped" guarantee holds for a    #}
{# reader who never expands it (D31-28).                                     #}
{# ------------------------------------------------------------------------ #}
<section class="mb-8" hx-indicator="#bets-loading">
  {# The FIXED label vocabulary, keyed on backtest.bet_selector.REJECTION_REASONS. The selector
     emits the code; this map turns it into a label. No free prose is minted here, and an
     UNRECOGNISED code renders the RAW CODE rather than a blank cell -- so a newly added taxonomy
     member shows up as an unexplained-but-visible row instead of disappearing (UI-SPEC E3
     long-text). tests/api/test_bets_page.py iterates the exported tuple and asserts every live
     member has an entry here, so the map cannot silently fall behind the taxonomy. #}
  {% set SUPPRESSION_LABELS = {
    "ev_below_floor":     "Expected value below the floor",
    "not_subpop":         "Outside the eligible sub-population",
    "stale_line":         "Line captured after the lock",
    "missing_snapshot":   "No market line for this bet type",
    "missing_prediction": "No model prediction for this game",
    "real_odds_failed":   "Odds failed the real-market check",
    "zero_kelly_stake":   "Sizing returned no stake",
    "ev_not_finite":      "Expected value could not be computed",
    "no_bet_side":        "Model agrees with the market",
    "no_honest_ev_floor": "No honest threshold for this bet type",
    "edge_below_threshold": "Win edge over the spread line below the threshold",
    "no_honest_edge_threshold": "No honest edge threshold for this bet type",
    "no_bound_converter": "No spread converter bound to the blend"
  } %}
  {% set SUPPRESSION_HELP = {
    "ev_below_floor":     "Evaluated and priced; the edge was not large enough to bet.",
    "not_subpop":         "This bet type only bets a pre-registered slice; this game is not in it.",
    "stale_line":         "The stored line was captured after this game's lock, 6:00 PM Eastern the day before its kickoff, so the decision could not have had it.",
    "missing_snapshot":   "The game has odds for other bet types but not this one.",
    "missing_prediction": "The model produced no output for this game -- a pipeline gap, not a market gap.",
    "real_odds_failed":   "The stored price did not come from an accepted sportsbook source.",
    "zero_kelly_stake":   "Admitted on expected value but sized to zero after the caps.",
    "ev_not_finite":      "A non-finite value reached the expected-value calculation; the row is suppressed rather than tiered.",
    "no_bet_side":        "The model's number sits inside the no-bet band around the market's, so there is no side to bet and nothing was priced.",
    "no_honest_ev_floor": "No betting threshold could be set honestly for this bet type, so it places no bets; its predictions are still published.",
    "edge_below_threshold": "The price cleared the floor, but the model's edge over the market's spread-based win chance did not clear the threshold; a win bet needs both.",
    "no_honest_edge_threshold": "There was too little honest pre-lock data to set this bet type's edge threshold, so it places no bets; its predictions are still published.",
    "no_bound_converter": "The deployed blend binds no spread-to-win-probability converter, so a win bet's second test cannot run. An artifact fault, not missing odds."
  } %}
  {% set SUPPRESSION_CAPTION = "Every scheduled game is evaluated for all three bet types. This section is the complete record of what was not bet, and why." %}
  {% set BET_TYPE_LABELS = {"wp": "Winner", "ats": "Spread", "ou": "Totals"} %}

  {% if not suppressed_bets %}
  {# Zero suppressed rows renders the HEADER plus one line -- never an empty <details>, whose
     collapsed summary would promise a record that has no body (UI-SPEC E3 empty). #}
  <div class="min-h-[44px] flex flex-col justify-center gap-1">
    <h2 class="display text-lg text-fg">Suppressed candidates (0)</h2>
    <p class="text-xs text-muted max-w-2xl">{{ SUPPRESSION_CAPTION }}</p>
  </div>
  <p class="text-sm text-muted mt-2">No candidate was suppressed this week.</p>
  {% else %}
  {# Grouped by reason with a count per group. A row whose rejection_reason is SQL NULL is keyed
     to an explicit marker FIRST: groupby's own default fires only on a MISSING attribute, not on
     a present-but-None one, so a raw group would sort None against strings (a TypeError, a 500)
     and would render the word None as the reason. A suppressed row with no reason is a data
     defect and is shown as one -- never as a blank and never as a minted taxonomy member.
     Keyed BEFORE the disclosure so the summary and the body read the same groups. #}
  {% set _keyed_rows = [] %}
  {% for _row in suppressed_bets %}
    {% set _ = _keyed_rows.append(dict(_row, reason_key=_row.rejection_reason or "(no reason recorded)")) %}
  {% endfor %}
  {# The opening tag stays exactly <details id="suppressed-candidates"> (a test pins it); the
     panel styling sits on the summary and the body instead. #}
  <details id="suppressed-candidates">
    {# The caption lives INSIDE <summary> so it is visible while collapsed: anything after the
       summary is hidden until the disclosure is opened, and a reader who never opens it must
       still be told this section is the complete record. The per-reason counts sit in the summary
       too -- each is the length of the SAME group the body renders, so the two cannot disagree. #}
    <summary class="bg-panel cursor-pointer min-h-[44px] flex flex-col gap-2 px-4 py-3">
      <span class="flex flex-wrap items-center gap-x-3 gap-y-1">
        <span class="display text-lg text-fg">Suppressed candidates ({{ suppressed_bets|length }})</span>
        <span class="text-xs text-muted max-w-2xl">{{ SUPPRESSION_CAPTION }}</span>
        <span class="ml-auto text-sm font-semibold text-accent" aria-hidden="true">Show / hide</span>
      </span>
      <span class="flex flex-wrap gap-1.5" data-suppressed-reason-counts>
        {%- for reason, rows in _keyed_rows|groupby("reason_key") %}
        <span class="bg-panel-2 px-2 py-0.5 text-xs text-fg">{{ SUPPRESSION_LABELS.get(reason, reason) }} <b class="num">{{ rows|length }}</b></span>
        {%- endfor %}
      </span>
    </summary>

    <div class="bg-panel px-4 pb-4">
    {% for reason, rows in _keyed_rows|groupby("reason_key") %}
    <div class="mt-4">
      <h3 class="display text-base text-fg">{{ SUPPRESSION_LABELS.get(reason, reason) }} ({{ rows|length }})</h3>
      <p class="text-xs text-muted max-w-2xl mt-1">{{ SUPPRESSION_HELP.get(reason, "This reason code is not in the displayed vocabulary; the raw code from the selector is shown above.") }}</p>
      <div class="border border-line overflow-hidden mt-2">
        {# overflow-x-auto + whitespace-nowrap: horizontal scroll on narrow viewports, never a
           squeezed or clipped column (UI-SPEC E3 overflow). #}
        <div class="overflow-x-auto">
          <table class="w-full text-sm">
            <thead class="bg-panel-2">
              <tr>
                <th scope="col" class="px-4 py-2 text-left label whitespace-nowrap">Matchup</th>
                <th scope="col" class="px-4 py-2 text-left label whitespace-nowrap">Bet type</th>
                <th scope="col" class="px-4 py-2 text-left label whitespace-nowrap">Reason</th>
                <th scope="col" class="px-4 py-2 text-right label whitespace-nowrap">Line as of</th>
                <th scope="col" class="px-4 py-2 text-left label whitespace-nowrap">Provenance</th>
              </tr>
            </thead>
            <tbody class="divide-y divide-line">
              {% for bet in rows %}
              <tr class="hover:bg-panel-2 transition-colors">
                <td class="px-4 py-3 text-left whitespace-nowrap">
                  <a href="/games/{{ bet.game_id }}" class="text-accent hover:underline">{{ bet.game_id.split('_')[-1].replace('@', ' @ ') }}</a>
                </td>
                <td class="px-4 py-3 text-left text-fg whitespace-nowrap" data-bet-type>{{ BET_TYPE_LABELS.get(bet.target, bet.target) }}</td>
                <td class="px-4 py-3 text-left text-fg whitespace-nowrap" data-reason-cell>{{ SUPPRESSION_LABELS.get(reason, reason) }}</td>
                <td class="px-4 py-3 text-right num text-xs whitespace-nowrap text-muted">{{ bet.snapshot_ts or "--" }}</td>
                <td class="px-4 py-3 text-left whitespace-nowrap">
                  {% with provenance=bet.provenance, validation_type=bet.validation_type %}
                    {% include "components/_provenance_badge.html" %}
                  {% endwith %}
                </td>
              </tr>
              {% endfor %}
            </tbody>
          </table>
        </div>
      </div>
    </div>
    {% endfor %}
    </div>
  </details>
  {% endif %}
</section>
{% endif %}

{# ------------------------------------------------------------------------ #}
{# Rank 3: the realized-versus-expected tracker (SPEC R8, D31-22, UI-SPEC E4)#}
{#                                                                          #}
{# TWO PROVENANCE CLASSES, and THREE sections when three exist. The stored   #}
{# blocks are partitioned on the (provenance, validation_type) PAIR, not on  #}
{# provenance alone (plan 31-13): the replay class holds BOTH the burned     #}
{# 2021-2024 rows and the single clean 2025 holdout, and pooling those would #}
{# publish the one unspent split inside a contaminated figure -- exactly     #}
{# what D31-22's second column exists to prevent. So a cache carrying both   #}
{# renders TWO replay sections under the replay heading, told apart by their #}
{# provenance badges and carrying separate totals, plus the forward section. #}
{# The design contract names two HEADINGS; it does not cap the section count.#}
{#                                                                          #}
{# NO FIGURE IS COMPUTED ACROSS TWO SECTIONS, and none is computed here at   #}
{# all: every number below was aggregated by backtest.bet_tracker at         #}
{# population time and is rendered as stored (UIAP-01). The only transform   #}
{# applied is a UNIT render -- a stored fraction shown under a per-cent      #}
{# header -- which is formatting, not derivation.                            #}
{#                                                                          #}
{# It sits INSIDE the swap target so both blocks share the single week       #}
{# skeleton indicator (UI-SPEC E4 loading).                                  #}
{# ------------------------------------------------------------------------ #}
{% if bet_list_available %}
{% set TRACKER_ORDER = [
     ("backtest_replay", "contaminated"),
     ("backtest_replay", "clean_holdout"),
     ("forward", "forward_realized")
   ] %}
{% set TRACKER_HEADINGS = {
     "backtest_replay": "Backtest replay -- reconstructed after the fact",
     "forward": "Forward record -- recommended before kickoff"
   } %}
{% set TRACKER_CAPTIONS = {
     "backtest_replay": "These bets were never recommended in advance. They were reconstructed from historical data to show how the selection rule would have behaved. Do not read them as a track record.",
     "forward": "Each of these was decided before its game locked at 6:00 PM Eastern the day before kickoff, before the result existed. This is the only block that is a track record."
   } %}
{% set TRACKER_PUSH_FOOTNOTE = "A push returns the stake. Pushes are excluded from the hit-rate denominator and are never counted as a win or a loss." %}
{% set TRACKER_ZERO_RESULT_LINE = "A negative return here is the measurement, not a display problem." %}
{% set TRACKER_EMPTY_HEADING = "Nothing graded yet" %}
{% set TRACKER_EMPTY_BODY = "No recommendation in this class has a final result yet. Grades appear after the games are played and the cache is rebuilt." %}

{# The DECLARED display order, weakest evidence to strongest, mirroring
   backtest.bet_tracker.TRACKER_BLOCK_ORDER. The getter issues no ORDER BY, so the order is
   fixed HERE rather than assumed from the insert order of a table nobody sorts. #}
{% set _ordered_blocks = [] %}
{% for _pair in TRACKER_ORDER %}
  {% for _block in tracker_blocks %}
    {% if _block.provenance == _pair[0] and _block.validation_type == _pair[1] %}
      {% set _ = _ordered_blocks.append(_block) %}
    {% endif %}
  {% endfor %}
{% endfor %}
{% set replay_blocks = _ordered_blocks | selectattr("provenance", "equalto", "backtest_replay") | list %}
{% set forward_blocks = _ordered_blocks | selectattr("provenance", "equalto", "forward") | list %}

{# One figure card. The geometry is the shipped betting.html KPI card, so the tracker looks like
   the rest of the app; the tone class is the ONLY thing that varies. #}
{% macro tracker_figure(label, value, tone) -%}
<div class="bg-white rounded-lg border border-gray-200 p-4 text-center flex-1">
  <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide">{{ label }}</p>
  <p class="text-2xl font-semibold mt-1 stat-num whitespace-nowrap {{ tone }}">{{ value }}</p>
</div>
{%- endmacro %}

{# The six figures of ONE class, in three groups.

   PUSHES SITS OUTSIDE THE GROUP CONTAINING THE HIT RATE, structurally and not merely visually: a
   push settled without either side winning, so it is neither a win nor a loss and it is not in the
   hit-rate denominator. The three data-figure-group wrappers are what make that assertable.

   COLOUR IS TRACKER-ONLY on this page: green for a won bet, red for a lost bet, gray for a push or
   a zero. The EV band badge above stays monochrome for exactly this reason -- a green band on the
   same page as a green won-bet would read as a prediction of winning (UI-SPEC Deviation 1).

   A NULL return renders as "not measured", never as 0.000. The aggregator returns None (not 0.0)
   when the graded rows carry no stake, mirroring BetSelector._clv_report defaulting clv to None:
   an unmeasured return published as zero reads as a break-even result, which is a claim nobody
   made. A blank and a zero must not render identically. #}
{% macro tracker_figures(block) -%}
{% set _return = block.flat_return_units %}
{# MEASURED versus NOT MEASURED, kept apart at the render. A stored NULL means the aggregator did
   not compute the figure; a stored NaN means something upstream produced a non-finite value. Both
   are "not measured" and NEITHER may render as a number -- "+0.000" and "+nan" would each put a
   figure on the page that nothing established. The self-inequality is the NaN test: NaN is the one
   value not equal to itself. #}
{% set _return_measured = _return is not none and _return == _return %}
{% set _hit_rate_measured = block.hit_rate is not none and block.hit_rate == block.hit_rate %}
<div class="overflow-x-auto">
  <div class="flex flex-wrap gap-4">
    <div data-figure-group="counts" class="flex flex-wrap gap-4 flex-1">
      {{ tracker_figure("Bets graded", block.bets_graded, "text-gray-900") }}
      {{ tracker_figure("Wins", block.wins, "text-green-700") }}
      {{ tracker_figure("Losses", block.losses, "text-red-600") }}
    </div>
    <div data-figure-group="pushes" class="flex flex-wrap gap-4 flex-1">
      {{ tracker_figure("Pushes", block.pushes, "text-gray-500") }}
    </div>
    <div data-figure-group="rates" class="flex flex-wrap gap-4 flex-1">
      {# hit_rate is stored as a FRACTION of the contested rows; the x100 is a unit render under a
         per-cent label, the same formatting transform the EV column applies (UIAP-01). #}
      {% if _hit_rate_measured %}
        {{ tracker_figure("Hit rate", "%.1f"|format(block.hit_rate * 100) ~ "%", "text-gray-900") }}
      {% else %}
        {{ tracker_figure("Hit rate", "not measured", "text-gray-500") }}
      {% endif %}
      {% if not _return_measured %}
        {{ tracker_figure("Return (flat, units)", "not measured", "text-gray-500") }}
      {% elif _return > 0 %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-green-700") }}
      {% elif _return < 0 %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-red-600") }}
      {% else %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-gray-500") }}
      {% endif %}
    </div>
  </div>
</div>
<p class="text-xs text-gray-500 mt-3 max-w-2xl">{{ TRACKER_PUSH_FOOTNOTE }}</p>
{% if _return_measured and _return <= 0 %}
<p class="text-xs text-gray-500 mt-2 max-w-2xl">{{ TRACKER_ZERO_RESULT_LINE }}</p>
{% endif %}
{%- endmacro %}

{# One SECTION per honesty class. Own heading, own caption, own totals, no shared row. #}
{% macro tracker_section(provenance, validation_type, block) -%}
<section class="mb-8" hx-indicator="#bets-loading" data-tracker-block="{{ provenance }}{% if validation_type %}:{{ validation_type }}{% endif %}">
  <div class="mb-4">
    <div class="flex flex-wrap items-center gap-2">
      <h2 class="text-base font-semibold text-gray-900">{{ TRACKER_HEADINGS[provenance] }}</h2>
      {# The honesty label for THIS class, from the single vocabulary source. It is what tells the
         two replay sections apart: they share a heading and differ only in evidence class, so the
         badge is load-bearing here rather than decorative. #}
      {% if validation_type %}
        {% with provenance=provenance, validation_type=validation_type %}
          {% include "components/_provenance_badge.html" %}
        {% endwith %}
      {% endif %}
    </div>
    <p class="text-sm text-gray-500 max-w-2xl mt-1">{{ TRACKER_CAPTIONS[provenance] }}</p>
  </div>
  {% if block is none or not block.bets_graded %}
    {# Zero graded rows renders the empty state instead of a row of zeros. The hit rate is not
       computed at all in this branch -- never a 0 / 0 display and never a divide-by-zero
       (UI-SPEC E4 empty). #}
    {% with heading=TRACKER_EMPTY_HEADING, body=TRACKER_EMPTY_BODY, action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% else %}
    {{ tracker_figures(block) }}
  {% endif %}
</section>
{%- endmacro %}

{# Old-rule label for the backtest-replay sections below (R16 / D33.2-07). The route scopes it
   to the replay rows' seasons, and to a known-empty scope when no replay class has graded bets. #}
{% with scope=replay_old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}

{# The replay sections. One per replay class PRESENT in the ledger; when the ledger holds none, a
   single section still renders so the reader learns that nothing is graded rather than finding
   the block missing. #}
{% if replay_blocks %}
  {% for block in replay_blocks %}
    {{ tracker_section("backtest_replay", block.validation_type, block) }}
  {% endfor %}
{% else %}
  {{ tracker_section("backtest_replay", none, none) }}
{% endif %}

{# The forward section. Under the hard-block it is WITHHELD while the replay sections above stay
   readable in the SAME response -- a replay figure does not depend on the current week's locks,
   so refusing it would be a refusal nothing justified (UI-SPEC E4 error). #}
{% if bets_blocked %}
<section class="mb-8" hx-indicator="#bets-loading" data-tracker-block="forward">
  <div class="mb-4">
    <div class="flex flex-wrap items-center gap-2">
      <h2 class="text-base font-semibold text-gray-900">{{ TRACKER_HEADINGS["forward"] }}</h2>
      {# The evidence class is known by construction for the forward provenance, and it describes
         the class rather than any figure -- so the label stays visible while the totals are
         withheld. #}
      {% with provenance="forward", validation_type="forward_realized" %}
        {% include "components/_provenance_badge.html" %}
      {% endwith %}
    </div>
    <p class="text-sm text-gray-500 max-w-2xl mt-1">{{ TRACKER_CAPTIONS["forward"] }}</p>
  </div>
  {% with message="The forward record is withheld -- this week's locked games have no list in the cache",
          recovery_text='The forward record\'s totals are computed from the same cache the list above is missing from, so they would be incomplete. The backtest replay above is unaffected and remains readable, because a replay figure does not depend on the current week\'s locks. To restore this section, rebuild the list with "uv run python scripts/generate_bet_list.py" and then load it into the cache with "uv run python scripts/populate_cache.py". The second command only COPIES what the first produces, so running it on its own leaves this section withheld.' %}
    {% include "components/_error_state.html" %}
  {% endwith %}
</section>
{% elif forward_blocks %}
  {% for block in forward_blocks %}
    {{ tracker_section("forward", block.validation_type, block) }}
  {% endfor %}
{% else %}
  {{ tracker_section("forward", none, none) }}
{% endif %}
{% endif %}

{# ------------------------------------------------------------------------ #}
{# Rank 4: the trailing utilities -- the export pair and the cross-link      #}
{# (D31-32, UI-SPEC E8).                                                     #}
{#                                                                          #}
{# INSIDE the swap target on purpose. The week selector swaps only           #}
{# #bets-content, so an export link rendered outside it would keep pointing  #}
{# at the week the page first loaded with -- handing the reader a file for a #}
{# week other than the one on screen, which is the export-disagrees-with-the #}
{# -page failure this whole task exists to prevent.                           #}
{#                                                                          #}
{# _export_buttons.html is the one shared export partial, reused as-is: its  #}
{# in-flight and failure behaviour is not re-decided here.                   #}
{#                                                                          #}
{# The export routes take type=bets and return BOTH halves of the week --    #}
{# the live rows and the suppressed ones -- through the same two getters     #}
{# this page reads, in this page's order. They are withheld under the        #}
{# hard-block for the same reason the list is: a download of rows the page   #}
{# just refused to show would be the refusal undone by a link beneath it.    #}
{# ------------------------------------------------------------------------ #}
{% if bet_list_available %}
<div class="mt-8 pt-4 border-t border-line flex flex-wrap items-center justify-between gap-4">
  {% if not bets_blocked and current_season is not none and current_week is not none %}
    {# Raw ampersands, matching the shipped this_week.html call site: autoescape turns them into
       &amp; in the rendered href, and pre-escaping here would emit &amp;amp; and break the URL. #}
    {% with csv_url="/api/export/csv?type=bets&season=" ~ current_season ~ "&week=" ~ current_week,
            json_url="/api/export/json?type=bets&season=" ~ current_season ~ "&week=" ~ current_week %}
      {% include "components/_export_buttons.html" %}
    {% endwith %}
  {% endif %}
  {# A plain navigation to the betting-simulation evidence. /betting was merged into Track
     Record's betting-simulation section (redesign spec 7.5), so the link points there directly
     rather than through the redirect. #}
  <a href="/track-record#betting-sim" class="text-sm font-semibold text-accent hover:underline">See the backtest evidence behind these bets</a>
</div>
{% endif %}

{% endblock %}
</div>

{# ------------------------------------------------------------------------ #}
{# The trailing utilities row: the cache stamp (UI-SPEC E12).                #}
{#                                                                          #}
{# OUTSIDE the swap target, so it is server-rendered with the page frame and #}
{# has no in-flight state. It renders in the hard-block too, deliberately:   #}
{# the refusal's recovery text repeats both timestamps, so the two claims    #}
{# are visible together and can be compared.                                 #}
{#                                                                          #}
{# The freeze sentence makes NO week-level claim. A week-level "lines were   #}
{# locked Saturday 6 PM ET" would be FALSE for every Thursday game (D31-18).#}
{# The per-game rule is stated once here and each slip carries its own       #}
{# Line as of value, so the reader can check the claim rather than take it.  #}
{# ------------------------------------------------------------------------ #}
{% if bet_list_available %}
<div class="mt-8 pt-4 border-t border-line">
  <p class="text-xs text-muted max-w-2xl">Bet list last populated: {% if bet_list_populated_at %}{{ bet_list_populated_at }}{% else %}not recorded{% endif %}. Each game's line locks at 6:00 PM Eastern the day before its own kickoff -- a Thursday game locks on the Wednesday, that week's Sunday games on the Saturday.</p>
</div>
{% endif %}
{% endblock %}
```

- [ ] **Step 4: Recompile the stylesheet**

Run:
```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
```
Expected: `Done in ...`. Then confirm the arbitrary grid class compiled:
```bash
grep -c "3rem_12rem" web/static/css/tailwind-compiled.css
```
Expected: `1` (the minified file is one line; any non-zero count means the class is present).

- [ ] **Step 5: Run every test file that reads /bets or its export**

Run one file per command:
```bash
uv run pytest tests/api/test_bets_page.py -v
uv run pytest tests/api/test_export.py -v
uv run pytest tests/api/test_cold_start_bet_list_recovery.py -v
uv run pytest tests/api/test_cache_swap_recovery.py -v
uv run pytest tests/unit/test_page_labels.py -v
uv run pytest tests/api/test_page_labels_routes.py -v
```
Expected: all PASS, on condition that the Task 4/5 replacements in Contract notes 5a-5e are already applied. Pay particular attention to the tracker tests (unchanged markup), the four-state tests, the hard-block tests, `test_tie_break_is_four_key` (matchup anchors) and `test_a_partly_built_week_shows_its_rows_and_names_the_rest` (the missing-games box contains no inner `<div>`).

- [ ] **Step 6: Lint and type-check the touched tests**

```bash
uv run ruff check tests/api/test_bets_page.py tests/api/test_export.py
uv run ruff format tests/api/test_bets_page.py tests/api/test_export.py
uv run pyright tests/api/test_bets_page.py tests/api/test_export.py
```

Expected: no ruff error. pyright reports no error on a line this task wrote; an older one predates the branch -- note it in the task report and leave it.

- [ ] **Step 7: Commit**

```bash
git add web/templates/pages/bets.html web/static/css/tailwind-compiled.css tests/api/test_bets_page.py tests/api/test_export.py
git commit -m "$(cat <<'EOF'
feat(redesign): /bets live list as ranked bet slips, restyled states and disclosure

Each live bet is a slip (rank, team blocks, bet type, the pick, line, EV %,
stake, EV band, evidence chip, visible line-capture time). The suppressed
disclosure stays a collapsed native details element and now shows a per-reason
count row in its summary. Every honesty string, id, state and order is kept;
tests locate rows by data hooks instead of class strings. The evidence
cross-link points at Track Record's betting-simulation section.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 12: Tracker restyle and the result strip

**Files:**
- Create: `web/templates/components/_result_strip.html`
- Modify: `web/templates/pages/bets.html` (the tracker macros and the section calls carried over verbatim in Task 11)
- Modify: `tests/api/test_bets_page.py`
- Regenerate: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes:
  - Task 10's `graded_outcomes` context key.
  - Task 1 classes: `stat-tile`, `stat-tile-value`, `label`, `display`, `tag`, `tag-ghost`, `section-head`, `unskew`, tokens `text-fg`, `text-muted`, `bg-line`.
  - Contract outcome classes: `text-green-400`, `text-red-400`, `bg-green-500`, `bg-red-500`.
- Produces:
  - The macro `result_strip(outcomes: list[str], block) -> markup`, in `components/_result_strip.html`. It renders only when the counts agree with the block.
  - Tracker hooks: `data-figure-label`, `data-figure-value`, `data-tracker-record`, `data-result-strip`, `data-result-mark`.

- [ ] **Step 1: Update the tracker assertions and add the strip tests**

Apply each old -> new edit to `tests/api/test_bets_page.py`.

(a) In `test_the_tracker_renders_one_section_per_honesty_class_and_never_pools_them`.

Old:
```python
    # Every rendered figure card lives inside exactly one section.
    inside = sum(
        section.count(
            'class="text-xs font-semibold text-gray-500 uppercase tracking-wide"'
        )
        for section in sections.values()
    )
    tracker_start = min(body.index(s) for s in sections.values())
    tracker_region = body[tracker_start:]
    assert inside == tracker_region.count(
        'class="text-xs font-semibold text-gray-500 uppercase tracking-wide"'
    ), "a tracker figure rendered outside one of the sections"
```
New:
```python
    # Every rendered figure tile lives inside exactly one section, located by its data hook.
    inside = sum(section.count("data-figure-label") for section in sections.values())
    tracker_start = min(body.index(s) for s in sections.values())
    tracker_region = body[tracker_start:]
    assert inside == tracker_region.count("data-figure-label"), (
        "a tracker figure rendered outside one of the sections"
    )
    assert inside == 18, "three sections of six figures each did not all render"
```

(b) In `test_each_block_renders_exactly_the_six_named_figures`.

Old:
```python
    labels = re.findall(
        r'<p class="text-xs font-semibold text-gray-500 uppercase tracking-wide">([^<]+)</p>',
        section,
    )
```
New:
```python
    labels = re.findall(r'<p class="label" data-figure-label>([^<]+)</p>', section)
```

(c) In `test_a_negative_return_states_the_measurement_and_a_positive_one_does_not`.

Old:
```python
    assert "-0.053" in negative
    assert "text-red-600" in negative
```
New:
```python
    # The RETURN tile itself carries the loss red. The Losses tile is red in every section, so a
    # bare "text-red-400" substring check would pass whatever colour the return took.
    assert 'text-red-400" data-figure-value>-0.053<' in negative
```

(d) In `test_an_unmeasured_return_never_renders_as_a_zero`.

Old:
```python
    values = re.findall(r'<p class="text-2xl[^"]*">([^<]*)</p>', section)
```
New:
```python
    values = re.findall(
        r'<p class="stat-tile-value[^"]*" data-figure-value>([^<]*)</p>', section
    )
```

(e) In `test_green_and_red_appear_only_inside_the_tracker_sections`, replace the first two tracker assertions.

Old:
```python
    sections = _tracker_sections(body)
    assert any("text-green-700" in s for s in sections.values())
    assert any("text-red-600" in s for s in sections.values())
```
New:
```python
    sections = _tracker_sections(body)
    replay = sections["backtest_replay:contaminated"]
    # Each realized-outcome hue is pinned to the tile that carries it. The Wins tile is always
    # green and the Losses tile always red, so a bare substring check anywhere in the section
    # could not tell whether the RETURN (-0.01 here) took its colour from its sign.
    assert 'text-green-400" data-figure-value>6<' in replay
    assert 'text-red-400" data-figure-value>-0.010<' in replay
```
Then replace the above-the-tracker guard at the end of the same test.

Old:
```python
    # Outside the tracker region the realized-outcome colours do not appear at all. The ONE red
    # above the tracker is the REFUSAL role, not the realized-outcome role: _error_state.html
    # pairs exactly one bg-red-50 container with exactly one text-red-600 recovery line, and the
    # UI-SPEC's colour table lists those as separate roles. Counting the pair is what keeps this
    # assertion honest without pretending the shipped refusal partial is a tracker colour.
    tracker_start = min(body.index(s) for s in sections.values())
    above = body[:tracker_start]
    assert "text-green-700" not in above, (
        "a realized-outcome green rendered above the tracker"
    )
    assert above.count("text-red-600") == above.count("bg-red-50"), (
        "a red above the tracker is not accounted for by a refusal block"
    )
```
New:
```python
    # Outside the tracker region the realized-outcome colours do not appear at all. The refusal
    # role uses its own red family (the error state's red-950 / 800 / 200 / 100), so the OUTCOME
    # shades -- green and red 400 text, 500 fill -- must be absent above the tracker outright.
    tracker_start = min(body.index(s) for s in sections.values())
    above = body[:tracker_start]
    for outcome_class in ("text-green-400", "bg-green-500", "text-red-400", "bg-red-500"):
        assert outcome_class not in above, (
            f"the realized-outcome colour {outcome_class} rendered above the tracker"
        )
```

(f) Append these tests at the end of the tracker test group, immediately after `test_the_tracker_shares_the_single_week_swap_indicator`:

```python
def _render_strip(outcomes: list[str], block: dict[str, int]) -> str:
    """Render the result-strip macro on its own, through the app's template environment."""
    module = app_templates.env.get_template("components/_result_strip.html").module
    return str(module.result_strip(outcomes, block))  # pyright: ignore[reportAttributeAccessIssue]


def test_the_result_strip_renders_only_when_it_agrees_with_its_block() -> None:
    """Review Focus 4: a strip that disagrees with the stored block renders NOTHING.

    The tiles read the precomputed block; the marks read the graded rows. A population run
    interrupted between the two writes leaves them out of step, and two disagreeing records on one
    page would each contradict the other. The stored tiles stay the authority.
    """
    block = {"bets_graded": 3, "wins": 2, "losses": 1, "pushes": 0}

    assert _render_strip([], block).strip() == "", "an empty class drew a strip"
    assert _render_strip(["win", "win", "win"], block).strip() == "", (
        "a strip whose wins disagree with the stored block was drawn"
    )
    # Every named count agrees but the length does not: a value outside the grading vocabulary
    # slipped in. Still a disagreement, still no strip.
    assert _render_strip(["win", "win", "loss", "pending"], block).strip() == ""

    agreeing = _render_strip(["win", "loss", "win"], block)
    assert agreeing.count("data-result-mark") == 3
    assert "wins 2, losses 1, pushes 0" in agreeing
    assert agreeing.count("bg-green-500") == 2
    assert agreeing.count("bg-red-500") == 1


def test_the_result_strip_draws_one_mark_per_graded_bet_in_its_own_section(
    tmp_path: Path,
) -> None:
    """Each class's strip draws that class's graded bets and no other class's."""
    rows = [
        _graded_row("2023_W01_DET@KC", "win"),
        _graded_row("2023_W01_CAR@ATL", "loss"),
        _graded_row("2023_W01_CIN@CLE", "win"),
        _graded_row("2023_W01_JAX@IND", "push"),
        _graded_row("2023_W01_DEN@LVR", "win", pair=_FORWARD_CLASS),
    ]
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=2,
            losses=1,
            pushes=1,
            hit_rate=2 / 3,
            flat_return_units=0.1,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=1,
            wins=1,
            losses=0,
            pushes=0,
            hit_rate=1.0,
            flat_return_units=0.909,
        ),
    ]
    with _client_with_tracker(tmp_path, blocks, "strip_agrees", rows=rows) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    sections = _tracker_sections(body)
    replay = sections["backtest_replay:contaminated"]
    forward = sections["forward:forward_realized"]
    assert replay.count("data-result-mark") == 4
    assert "wins 2, losses 1, pushes 1" in replay
    assert ">2-1-1<" in replay, "the record line does not restate the stored counts"
    assert forward.count("data-result-mark") == 1
    assert "wins 1, losses 0, pushes 0" in forward


def test_a_result_strip_that_disagrees_with_its_stored_block_is_omitted(
    tmp_path: Path,
) -> None:
    """The page-level half of Review Focus 4: no strip, and the stored tiles still render."""
    rows = [
        _graded_row("2023_W01_DET@KC", "win"),
        _graded_row("2023_W01_CAR@ATL", "win"),
        _graded_row("2023_W01_CIN@CLE", "win"),
        _graded_row("2023_W01_JAX@IND", "loss"),
    ]
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=4,
            wins=2,
            losses=1,
            pushes=1,
            hit_rate=2 / 3,
            flat_return_units=0.1,
        )
    ]
    with _client_with_tracker(tmp_path, blocks, "strip_disagrees", rows=rows) as client:
        body = client.get(f"/bets?season={_SEASON}&week={_WEEK}").text

    replay = _tracker_sections(body)["backtest_replay:contaminated"]
    assert "data-result-strip" not in replay
    assert "data-result-mark" not in replay
    # The STORED block stays the authority: its tiles and its record line still render.
    assert "data-figure-value>4<" in replay
    assert "Hit rate" in replay
    assert ">2-1-1<" in replay
```

- [ ] **Step 2: Run the changed and new tests to verify they fail**

Run:
```bash
uv run pytest "tests/api/test_bets_page.py::test_the_tracker_renders_one_section_per_honesty_class_and_never_pools_them" "tests/api/test_bets_page.py::test_each_block_renders_exactly_the_six_named_figures" "tests/api/test_bets_page.py::test_a_negative_return_states_the_measurement_and_a_positive_one_does_not" "tests/api/test_bets_page.py::test_an_unmeasured_return_never_renders_as_a_zero" "tests/api/test_bets_page.py::test_green_and_red_appear_only_inside_the_tracker_sections" "tests/api/test_bets_page.py::test_the_result_strip_renders_only_when_it_agrees_with_its_block" "tests/api/test_bets_page.py::test_the_result_strip_draws_one_mark_per_graded_bet_in_its_own_section" "tests/api/test_bets_page.py::test_a_result_strip_that_disagrees_with_its_stored_block_is_omitted" -v
```
Expected: all FAIL. Either `data-figure-label`, `text-green-400` and `data-result-mark` are absent, or the macro test fails with `TemplateNotFound: components/_result_strip.html`.

- [ ] **Step 3: Create `web/templates/components/_result_strip.html`**

```jinja
{# The /bets result strip: one mark per graded live bet of ONE honesty class (redesign spec 7.3).

   A SECOND RENDERING OF FIGURES THE TILES ALREADY SHOW, so it is drawn only when it agrees with
   them. The tiles read the PRECOMPUTED tracker block (backtest.bet_tracker, UIAP-01); the marks
   read the graded rows that block was aggregated from (DataService.get_graded_bet_outcomes). The
   two arrive through separate population writes -- a run interrupted between the bet-list write
   and the tracker write leaves them out of step -- and two disagreeing records on one page would
   each contradict the other. So the strip renders only when its win, loss and push marks number
   exactly the block's stored wins, losses and pushes AND its length equals bets_graded; otherwise
   it renders nothing and the stored tiles stand alone as the authority.

   The counts below are a CONSISTENCY CHECK, never a published figure: nothing is displayed here
   that the stored block does not already carry.

   Colour is the tracker's outcome colour -- green won, red lost, grey push. This strip only ever
   renders inside a tracker section, the one place on /bets where hue may mean a result.

   Usage: {% from "components/_result_strip.html" import result_strip %}
          {{ result_strip(outcomes, block) }}
   outcomes: list of "win" / "loss" / "push" in week order (game id order within a week);
   block: the stored tracker block. #}
{% macro result_strip(outcomes, block) -%}
{%- set _outcomes = outcomes or [] -%}
{%- set _wins = _outcomes | select("equalto", "win") | list | length -%}
{%- set _losses = _outcomes | select("equalto", "loss") | list | length -%}
{%- set _pushes = _outcomes | select("equalto", "push") | list | length -%}
{%- if _outcomes
      and _wins == block.wins
      and _losses == block.losses
      and _pushes == block.pushes
      and (_outcomes | length) == block.bets_graded -%}
{# The label prints the STORED block's counts: the strip only renders when its own marks agree
   with them, and the stored block stays the one source of every published figure. #}
<div class="flex flex-wrap items-center gap-0.5" role="img" data-result-strip aria-label="Graded results in week order: wins {{ block.wins }}, losses {{ block.losses }}, pushes {{ block.pushes }}">
  {%- for outcome in _outcomes %}
  <span data-result-mark class="block w-1.5 h-4 {% if outcome == 'win' %}bg-green-500{% elif outcome == 'loss' %}bg-red-500{% else %}bg-[#5B6478]{% endif %}"></span>
  {%- endfor %}
</div>
{%- endif -%}
{%- endmacro %}
```

- [ ] **Step 4: Restyle the tracker macros in `web/templates/pages/bets.html`**

Old: the span from the `One figure card` comment through the `{%- endmacro %}` that closes `tracker_section`. It is exactly this text, as carried over verbatim in Task 11:
```jinja
{# One figure card. The geometry is the shipped betting.html KPI card, so the tracker looks like
   the rest of the app; the tone class is the ONLY thing that varies. #}
{% macro tracker_figure(label, value, tone) -%}
<div class="bg-white rounded-lg border border-gray-200 p-4 text-center flex-1">
  <p class="text-xs font-semibold text-gray-500 uppercase tracking-wide">{{ label }}</p>
  <p class="text-2xl font-semibold mt-1 stat-num whitespace-nowrap {{ tone }}">{{ value }}</p>
</div>
{%- endmacro %}

{# The six figures of ONE class, in three groups.

   PUSHES SITS OUTSIDE THE GROUP CONTAINING THE HIT RATE, structurally and not merely visually: a
   push settled without either side winning, so it is neither a win nor a loss and it is not in the
   hit-rate denominator. The three data-figure-group wrappers are what make that assertable.

   COLOUR IS TRACKER-ONLY on this page: green for a won bet, red for a lost bet, gray for a push or
   a zero. The EV band badge above stays monochrome for exactly this reason -- a green band on the
   same page as a green won-bet would read as a prediction of winning (UI-SPEC Deviation 1).

   A NULL return renders as "not measured", never as 0.000. The aggregator returns None (not 0.0)
   when the graded rows carry no stake, mirroring BetSelector._clv_report defaulting clv to None:
   an unmeasured return published as zero reads as a break-even result, which is a claim nobody
   made. A blank and a zero must not render identically. #}
{% macro tracker_figures(block) -%}
{% set _return = block.flat_return_units %}
{# MEASURED versus NOT MEASURED, kept apart at the render. A stored NULL means the aggregator did
   not compute the figure; a stored NaN means something upstream produced a non-finite value. Both
   are "not measured" and NEITHER may render as a number -- "+0.000" and "+nan" would each put a
   figure on the page that nothing established. The self-inequality is the NaN test: NaN is the one
   value not equal to itself. #}
{% set _return_measured = _return is not none and _return == _return %}
{% set _hit_rate_measured = block.hit_rate is not none and block.hit_rate == block.hit_rate %}
<div class="overflow-x-auto">
  <div class="flex flex-wrap gap-4">
    <div data-figure-group="counts" class="flex flex-wrap gap-4 flex-1">
      {{ tracker_figure("Bets graded", block.bets_graded, "text-gray-900") }}
      {{ tracker_figure("Wins", block.wins, "text-green-700") }}
      {{ tracker_figure("Losses", block.losses, "text-red-600") }}
    </div>
    <div data-figure-group="pushes" class="flex flex-wrap gap-4 flex-1">
      {{ tracker_figure("Pushes", block.pushes, "text-gray-500") }}
    </div>
    <div data-figure-group="rates" class="flex flex-wrap gap-4 flex-1">
      {# hit_rate is stored as a FRACTION of the contested rows; the x100 is a unit render under a
         per-cent label, the same formatting transform the EV column applies (UIAP-01). #}
      {% if _hit_rate_measured %}
        {{ tracker_figure("Hit rate", "%.1f"|format(block.hit_rate * 100) ~ "%", "text-gray-900") }}
      {% else %}
        {{ tracker_figure("Hit rate", "not measured", "text-gray-500") }}
      {% endif %}
      {% if not _return_measured %}
        {{ tracker_figure("Return (flat, units)", "not measured", "text-gray-500") }}
      {% elif _return > 0 %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-green-700") }}
      {% elif _return < 0 %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-red-600") }}
      {% else %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-gray-500") }}
      {% endif %}
    </div>
  </div>
</div>
<p class="text-xs text-gray-500 mt-3 max-w-2xl">{{ TRACKER_PUSH_FOOTNOTE }}</p>
{% if _return_measured and _return <= 0 %}
<p class="text-xs text-gray-500 mt-2 max-w-2xl">{{ TRACKER_ZERO_RESULT_LINE }}</p>
{% endif %}
{%- endmacro %}

{# One SECTION per honesty class. Own heading, own caption, own totals, no shared row. #}
{% macro tracker_section(provenance, validation_type, block) -%}
<section class="mb-8" hx-indicator="#bets-loading" data-tracker-block="{{ provenance }}{% if validation_type %}:{{ validation_type }}{% endif %}">
  <div class="mb-4">
    <div class="flex flex-wrap items-center gap-2">
      <h2 class="text-base font-semibold text-gray-900">{{ TRACKER_HEADINGS[provenance] }}</h2>
      {# The honesty label for THIS class, from the single vocabulary source. It is what tells the
         two replay sections apart: they share a heading and differ only in evidence class, so the
         badge is load-bearing here rather than decorative. #}
      {% if validation_type %}
        {% with provenance=provenance, validation_type=validation_type %}
          {% include "components/_provenance_badge.html" %}
        {% endwith %}
      {% endif %}
    </div>
    <p class="text-sm text-gray-500 max-w-2xl mt-1">{{ TRACKER_CAPTIONS[provenance] }}</p>
  </div>
  {% if block is none or not block.bets_graded %}
    {# Zero graded rows renders the empty state instead of a row of zeros. The hit rate is not
       computed at all in this branch -- never a 0 / 0 display and never a divide-by-zero
       (UI-SPEC E4 empty). #}
    {% with heading=TRACKER_EMPTY_HEADING, body=TRACKER_EMPTY_BODY, action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% else %}
    {{ tracker_figures(block) }}
  {% endif %}
</section>
{%- endmacro %}
```
New (the whole replacement for that span):
```jinja
{# The graded outcomes behind the blocks, keyed by the section's own data-tracker-block string
   (Task 10). A cache or a hand-built context without them gets an empty map: no strip, and the
   stored tiles render exactly as before. #}
{% from "components/_result_strip.html" import result_strip %}
{% set _outcomes_by_class = graded_outcomes if graded_outcomes is defined and graded_outcomes else {} %}

{# One figure tile. The tone class is the ONLY thing that varies. data-figure-label and
   data-figure-value are the structural hooks the tests read, so a figure is attributed to the
   tracker by markup rather than by a class string other regions of the page may share. #}
{% macro tracker_figure(label, value, tone) -%}
<div class="stat-tile flex-1 min-w-[7.5rem]">
  <p class="label" data-figure-label>{{ label }}</p>
  <p class="stat-tile-value display text-3xl leading-tight tabular-nums whitespace-nowrap {{ tone }}" data-figure-value>{{ value }}</p>
</div>
{%- endmacro %}

{# The six figures of ONE class, in three groups, under the class's record line and result strip.

   PUSHES SITS OUTSIDE THE GROUP CONTAINING THE HIT RATE, structurally and not merely visually: a
   push settled without either side winning, so it is neither a win nor a loss and it is not in the
   hit-rate denominator. The three data-figure-group wrappers are what make that assertable.

   THE RECORD LINE AND THE STRIP ADD NO FIGURE. "W-L-P" restates the stored wins, losses and
   pushes in the order a sports record is read; the strip draws the same graded bets one mark each
   and only when they agree with the block (components/_result_strip.html). Neither is pooled
   across classes and neither is computed from anything the block does not carry.

   COLOUR IS TRACKER-ONLY on this page: green for a won bet, red for a lost bet, grey for a push or
   a zero. The EV band badge above stays monochrome for exactly this reason -- a green band on the
   same page as a green won-bet would read as a prediction of winning (UI-SPEC Deviation 1).

   A NULL return renders as "not measured", never as 0.000. The aggregator returns None (not 0.0)
   when the graded rows carry no stake, mirroring BetSelector._clv_report defaulting clv to None:
   an unmeasured return published as zero reads as a break-even result, which is a claim nobody
   made. A blank and a zero must not render identically. #}
{% macro tracker_figures(block, outcomes) -%}
{% set _return = block.flat_return_units %}
{# MEASURED versus NOT MEASURED, kept apart at the render. A stored NULL means the aggregator did
   not compute the figure; a stored NaN means something upstream produced a non-finite value. Both
   are "not measured" and NEITHER may render as a number -- "+0.000" and "+nan" would each put a
   figure on the page that nothing established. The self-inequality is the NaN test: NaN is the one
   value not equal to itself. #}
{% set _return_measured = _return is not none and _return == _return %}
{% set _hit_rate_measured = block.hit_rate is not none and block.hit_rate == block.hit_rate %}
<div class="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2">
  <span class="display text-2xl text-fg whitespace-nowrap" data-tracker-record>{{ block.wins }}-{{ block.losses }}-{{ block.pushes }}</span>
  {{ result_strip(outcomes, block) }}
</div>
<div class="overflow-x-auto">
  <div class="flex flex-wrap gap-2">
    <div data-figure-group="counts" class="flex flex-wrap gap-2 flex-1">
      {{ tracker_figure("Bets graded", block.bets_graded, "text-fg") }}
      {{ tracker_figure("Wins", block.wins, "text-green-400") }}
      {{ tracker_figure("Losses", block.losses, "text-red-400") }}
    </div>
    <div data-figure-group="pushes" class="flex flex-wrap gap-2 flex-1">
      {{ tracker_figure("Pushes", block.pushes, "text-muted") }}
    </div>
    <div data-figure-group="rates" class="flex flex-wrap gap-2 flex-1">
      {# hit_rate is stored as a FRACTION of the contested rows; the x100 is a unit render under a
         per-cent label, the same formatting transform the EV column applies (UIAP-01). #}
      {% if _hit_rate_measured %}
        {{ tracker_figure("Hit rate", "%.1f"|format(block.hit_rate * 100) ~ "%", "text-fg") }}
      {% else %}
        {{ tracker_figure("Hit rate", "not measured", "text-muted") }}
      {% endif %}
      {% if not _return_measured %}
        {{ tracker_figure("Return (flat, units)", "not measured", "text-muted") }}
      {% elif _return > 0 %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-green-400") }}
      {% elif _return < 0 %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-red-400") }}
      {% else %}
        {{ tracker_figure("Return (flat, units)", "%+.3f"|format(_return), "text-muted") }}
      {% endif %}
    </div>
  </div>
</div>
<p class="text-xs text-muted mt-3 max-w-2xl">{{ TRACKER_PUSH_FOOTNOTE }}</p>
{% if _return_measured and _return <= 0 %}
<p class="text-xs text-muted mt-2 max-w-2xl">{{ TRACKER_ZERO_RESULT_LINE }}</p>
{% endif %}
{%- endmacro %}

{# One SECTION per honesty class. Own heading, own caption, own totals, no shared row.

   The heading row is ONE div holding the tag, the evidence chip and the rule, in that order: a
   test reads the section up to its FIRST closing div and requires the chip inside it. The live
   forward record takes the solid yellow tag; the replay classes take the outlined one, so the one
   block that is a track record is the one that looks like the headline. #}
{% macro tracker_section(provenance, validation_type, block, outcomes) -%}
<section class="mb-8" hx-indicator="#bets-loading" data-tracker-block="{{ provenance }}{% if validation_type %}:{{ validation_type }}{% endif %}">
  <div class="mb-4">
    <div class="section-head flex-wrap">
      <h2 class="{{ 'tag' if provenance == 'forward' else 'tag-ghost' }}"><span class="unskew">{{ TRACKER_HEADINGS[provenance] }}</span></h2>
      {# The honesty label for THIS class, from the single vocabulary source. It is what tells the
         two replay sections apart: they share a heading and differ only in evidence class, so the
         badge is load-bearing here rather than decorative. #}
      {% if validation_type %}
        {% with provenance=provenance, validation_type=validation_type %}
          {% include "components/_provenance_badge.html" %}
        {% endwith %}
      {% endif %}
      <span class="flex-1 h-px bg-line min-w-[2rem]" aria-hidden="true"></span>
    </div>
    <p class="text-sm text-muted max-w-2xl mt-2">{{ TRACKER_CAPTIONS[provenance] }}</p>
  </div>
  {% if block is none or not block.bets_graded %}
    {# Zero graded rows renders the empty state instead of a row of zeros. The hit rate is not
       computed at all in this branch -- never a 0 / 0 display and never a divide-by-zero
       (UI-SPEC E4 empty). #}
    {% with heading=TRACKER_EMPTY_HEADING, body=TRACKER_EMPTY_BODY, action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% else %}
    {{ tracker_figures(block, outcomes) }}
  {% endif %}
</section>
{%- endmacro %}
```

- [ ] **Step 5: Pass each class its outcomes, and restyle the withheld forward header**

Old (the replay calls):
```jinja
{% if replay_blocks %}
  {% for block in replay_blocks %}
    {{ tracker_section("backtest_replay", block.validation_type, block) }}
  {% endfor %}
{% else %}
  {{ tracker_section("backtest_replay", none, none) }}
{% endif %}
```
New:
```jinja
{% if replay_blocks %}
  {% for block in replay_blocks %}
    {{ tracker_section("backtest_replay", block.validation_type, block, _outcomes_by_class.get("backtest_replay:" ~ block.validation_type, [])) }}
  {% endfor %}
{% else %}
  {{ tracker_section("backtest_replay", none, none, []) }}
{% endif %}
```

Old (the withheld forward section's heading row):
```jinja
<section class="mb-8" hx-indicator="#bets-loading" data-tracker-block="forward">
  <div class="mb-4">
    <div class="flex flex-wrap items-center gap-2">
      <h2 class="text-base font-semibold text-gray-900">{{ TRACKER_HEADINGS["forward"] }}</h2>
      {# The evidence class is known by construction for the forward provenance, and it describes
         the class rather than any figure -- so the label stays visible while the totals are
         withheld. #}
      {% with provenance="forward", validation_type="forward_realized" %}
        {% include "components/_provenance_badge.html" %}
      {% endwith %}
    </div>
    <p class="text-sm text-gray-500 max-w-2xl mt-1">{{ TRACKER_CAPTIONS["forward"] }}</p>
  </div>
```
New:
```jinja
<section class="mb-8" hx-indicator="#bets-loading" data-tracker-block="forward">
  <div class="mb-4">
    <div class="section-head flex-wrap">
      <h2 class="tag"><span class="unskew">{{ TRACKER_HEADINGS["forward"] }}</span></h2>
      {# The evidence class is known by construction for the forward provenance, and it describes
         the class rather than any figure -- so the label stays visible while the totals are
         withheld. #}
      {% with provenance="forward", validation_type="forward_realized" %}
        {% include "components/_provenance_badge.html" %}
      {% endwith %}
      <span class="flex-1 h-px bg-line min-w-[2rem]" aria-hidden="true"></span>
    </div>
    <p class="text-sm text-muted max-w-2xl mt-2">{{ TRACKER_CAPTIONS["forward"] }}</p>
  </div>
```

Old (the forward calls):
```jinja
{% elif forward_blocks %}
  {% for block in forward_blocks %}
    {{ tracker_section("forward", block.validation_type, block) }}
  {% endfor %}
{% else %}
  {{ tracker_section("forward", none, none) }}
{% endif %}
```
New:
```jinja
{% elif forward_blocks %}
  {% for block in forward_blocks %}
    {{ tracker_section("forward", block.validation_type, block, _outcomes_by_class.get("forward:" ~ block.validation_type, [])) }}
  {% endfor %}
{% else %}
  {{ tracker_section("forward", none, none, []) }}
{% endif %}
```

- [ ] **Step 6: Recompile the stylesheet**

Run:
```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
```
Expected: `Done in ...`. Then check each outcome class individually:
```bash
for c in text-green-400 bg-green-500 text-red-400 bg-red-500; do grep -q "\.$c" web/static/css/tailwind-compiled.css && echo "$c ok" || echo "$c MISSING"; done
```
Expected: four lines, each ending `ok`.

- [ ] **Step 7: Run every /bets test file**

Run one file per command:
```bash
uv run pytest tests/api/test_bets_page.py -v
uv run pytest tests/api/test_export.py -v
uv run pytest tests/api/test_cold_start_bet_list_recovery.py -v
uv run pytest tests/api/test_cache_swap_recovery.py -v
uv run pytest tests/unit/test_page_labels.py -v
uv run pytest tests/api/test_page_labels_routes.py -v
```
Expected: all PASS. These must pass unchanged:
- `test_pushes_sits_outside_the_group_that_holds_the_hit_rate` (group regex; tiles hold no nested div)
- `test_the_badge_appears_on_both_tracker_block_headers` (chip before the first `</div>`)
- `test_every_tracker_section_closes_every_div_it_opens`
- `test_the_page_renders_the_stored_aggregate_and_computes_nothing` (`>12.5%<` and `>-0.750<` from the stored block)
- `test_the_forward_block_is_withheld_under_the_hard_block_while_replay_stays_readable`

- [ ] **Step 8: Lint and type-check the touched test file**

```bash
uv run ruff check tests/api/test_bets_page.py
uv run ruff format tests/api/test_bets_page.py
uv run pyright tests/api/test_bets_page.py
```

Expected: no ruff error. pyright reports no error on a line this task wrote; an older one predates the branch -- note it in the task report and leave it.

- [ ] **Step 9: Commit**

```bash
git add web/templates/components/_result_strip.html web/templates/pages/bets.html web/static/css/tailwind-compiled.css tests/api/test_bets_page.py
git commit -m "$(cat <<'EOF'
feat(redesign): /bets tracker as scoreboard tiles with a W-L-P record and result strip

Tracker sections take the yellow (forward) or outlined (replay) tag with the
evidence chip in the heading row, a W-L-P record line, stat tiles in the
existing three figure groups, and outcome colours green/red-400. The result
strip draws one mark per graded bet and renders only when its counts equal the
stored block, so two disagreeing records never share the page.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

# Part D -- Season, Track Record, How It Works (Tasks 13-16)

## Contract notes (Part D)

These are places where the real code forced a choice the contract does not spell out. Every task in this part already follows them.

1. **jinja2-fragments renders a block without running the template's top-level statements.** `render_block` calls `template.blocks[name](template.new_context(vars))`, so a `{% import %}` written at the top of a page is UNDEFINED inside a block served on its own (an HTMX swap or a `/fragments/*` route). Any block that is rendered standalone (`season_tracking_content`, `performance_content`, `betting_content`, and in other parts `game_grid` and `bets_content`) must import the macros it uses INSIDE the block. Parts B and C must do the same for `game_grid` and `bets_content`.
2. **`_build_betting_context` keys change:** `old_rule_scope` is renamed `betting_old_rule_scope` (Track Record carries four scopes side by side, so each is named for its block) and `current_path` becomes `"/track-record"`. Nothing outside Task 15 reads either key.
3. **Track Record carries 4 old-rule labels, not 5.** Spec 7.5 lists the season-comparison heatmap under "Season by season" and closing-line value as its own section. Each block that shows numbers must carry a label from its own scope (R16), so placing them literally would put two labels in one section and a fifth on the page. Instead: the heatmap sits in **All-time summary** (it spans the same whole backtest corpus as the summary tiles, so it shares their label), and the cumulative-CLV chart is a panel with `id="clv"` inside **Model vs Market** (same corpus, same label). The `clv` anchor in contract section 7 still works. Labels: summary, season table, model-vs-market, betting simulation.
4. **Season KPI tile labels move to the bet-type vocabulary** ("Winner / Spread / Totals hit rate"), matching /bets. The fourth tile keeps the label `Record (W-L)` and gains the sub-line "straight-up winner picks", because `kpis.record` is the Winner record only, while the new week strip combines all three bet types. The two existing `"WP Hit Rate"` assertions are updated in Task 14.
5. **New context key `current_slate_week`** (Season page): the live slate's week when the page shows the live slate's season, else `None`. It comes from the existing `DataService.get_current_slate()`.
6. **New shared component `components/_chart_panel.html`** with macro `chart_panel(title, chart_html, empty_body, min_height=380, panel_id=none)`. It is created in Task 14 and reused by Task 15, so the "chart or 'Chart unavailable'" pattern lives in one place.
7. **Page `<title>` blocks are kept** ("Season Tracking", and in other parts "This Week's Predictions", "Weekly Bet List"), because `tests/unit/test_page_labels.py::PAGE_MARKERS` and `tests/api/test_page_labels_routes.py::PAGE_ROUTES` find each page by a string that the `<title>` supplies. The new pages' markers are section headings unique to them: `track_record.html` -> "Season by season", `how_it_works.html` -> "What the models rely on".
8. **HTMX requests to the retired `/performance` and `/betting` URLs are answered with the block itself, not a redirect.** htmx would otherwise swap a whole page into a table-sized target. So the two `FRAGMENT_REQUESTS` entries for those URLs in `test_page_labels_routes.py` stay valid and now pin the legacy branch.

---

### Task 13: Per-week combined record in the season KPI blob

**Files:**
- Modify: `api/season_metrics.py` (module docstring "Contents" list; new function after `compute_weekly_series`)
- Modify: `api/charts/season.py:58-82` (re-export)
- Modify: `api/charts/prerender.py:386-389` (the season KPI blob)
- Modify: `tests/api/conftest.py:1199-1216` (fixture KPI payload)
- Test: `tests/test_season_metrics.py` (new tests at the end), `tests/test_charts.py` (new test after `test_season_prerender_covers_every_fixture_season`)

**Interfaces:**
- Consumes: `api.season_metrics._completed_sorted`, `_OUTCOME_FNS`, `_TARGETS`, `_num` (existing private helpers in the same module).
- Produces: `api.season_metrics.compute_weekly_records(rows: Sequence[dict]) -> list[dict[str, int]]`, returning `[{"week": int, "wins": int, "losses": int}, ...]` in ascending week order, weeks with no decided pick omitted. Re-exported as `api.charts.season.compute_weekly_records`. Every `season_kpis_<season>` JSON blob gains the key `"weeks"` holding that list. The fixture blob in `tests/api/conftest.py` carries `"weeks": [{"week": 1, "wins": 14, "losses": 9}, {"week": 2, "wins": 12, "losses": 11}]`.

- [ ] **Step 1: Write the failing metric tests**

Append to the end of `tests/test_season_metrics.py`:

```python
# ---------------------------------------------------------------------------
# compute_weekly_records -- the Season page's week-by-week strip
# ---------------------------------------------------------------------------


def test_weekly_records_combine_all_three_targets_per_week() -> None:
    """One week's record is the decided picks of all three bet types together.

    Week 1 (home wins 24-20, margin +4, total 44): WP 0.60 picks home -> HIT. ATS -2.0 vs -3.0 is
    above the line -> home_cover, slipped -2.5, margin 4 > -2.5 -> HIT. OU 45 vs 44 -> over,
    slipped 44.5, total 44 -> MISS. Week 2 (home loses 14-28, total 42): all three MISS.
    """
    from api.season_metrics import compute_weekly_records

    rows = [
        _game(week=1),
        _game(week=2, wp_prob=0.70, home_score=14, away_score=28),
    ]
    assert compute_weekly_records(rows) == [
        {"week": 1, "wins": 2, "losses": 1},
        {"week": 2, "wins": 0, "losses": 3},
    ]


def test_weekly_records_exclude_pushes_ties_and_unplayed_games() -> None:
    """Excluded picks are neither wins nor losses, and a week with none decided is omitted.

    Week 3 is a 21-21 tie: WP is excluded (tie); ATS home_cover slipped -2.5, margin 0 > -2.5 ->
    HIT; OU over slipped 44.5, total 42 -> MISS. Week 4 is not completed. Week 5 has no WP and no
    lines, so every target is excluded. Weeks 4 and 5 must not appear as 0-0.
    """
    from api.season_metrics import compute_weekly_records

    rows = [
        _game(week=3, home_score=21, away_score=21),
        _game(week=4, status="scheduled", home_score=None, away_score=None),
        _game(week=5, wp_prob=None, market_spread=None, market_total=None),
    ]
    assert compute_weekly_records(rows) == [{"week": 3, "wins": 1, "losses": 1}]


def test_weekly_records_empty_rows_returns_empty() -> None:
    from api.season_metrics import compute_weekly_records

    assert compute_weekly_records([]) == []


def test_weekly_records_sum_to_the_season_kpi_counts() -> None:
    """The strip and the KPI tiles read the same classifiers, so their totals agree."""
    from api.season_metrics import compute_season_kpis, compute_weekly_records

    rows = [
        _game(week=1),
        _game(week=2, wp_prob=0.70, home_score=14, away_score=28),
        _game(week=3, home_score=21, away_score=21),
    ]
    weeks = compute_weekly_records(rows)
    kpis = compute_season_kpis(rows)
    hits = sum(kpis[f"{t}_hits"] for t in ("wp", "ats", "ou"))
    decided = sum(kpis[f"{t}_decided"] for t in ("wp", "ats", "ou"))
    assert sum(w["wins"] for w in weeks) == hits
    assert sum(w["wins"] + w["losses"] for w in weeks) == decided
```

- [ ] **Step 2: Write the failing prerender test**

In `tests/test_charts.py`, insert this function directly after `test_season_prerender_covers_every_fixture_season` (before `def test_season_prerender_failure_isolation(`):

```python
def test_season_kpi_blob_carries_the_weekly_records() -> None:
    """season_kpis_<s> carries a ``weeks`` list equal to compute_weekly_records over that season.

    The strip and the KPI tiles are written into ONE blob by ONE population run, so they cannot
    describe two different caches.
    """
    import json

    from api.charts.prerender import prerender_charts_for_cache
    from api.season_metrics import compute_weekly_records

    preds, expected_seasons = _season_prediction_bundle()
    result = prerender_charts_for_cache({"predictions": preds})

    non_empty = 0
    for season in expected_seasons:
        decoded = json.loads(result[f"season_kpis_{season}"])
        season_rows = [
            r for r in preds if r.get("season") is not None and int(r["season"]) == season
        ]
        assert decoded["weeks"] == compute_weekly_records(season_rows)
        non_empty += bool(decoded["weeks"])
    assert non_empty, "no fixture season produced a weekly record; the check proves nothing"
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `uv run pytest tests/test_season_metrics.py -k weekly_records -v`
Expected: 4 FAIL with `ImportError: cannot import name 'compute_weekly_records' from 'api.season_metrics'`.

Run: `uv run pytest tests/test_charts.py::test_season_kpi_blob_carries_the_weekly_records -v`
Expected: FAIL with the same `ImportError`.

- [ ] **Step 4: Implement `compute_weekly_records`**

In `api/season_metrics.py`, in the module docstring's "Contents" list, replace:

```
* Weekly series — :func:`compute_weekly_series` (per-week hit-rate + a
  rolling-average overlay).
```

with:

```
* Weekly series — :func:`compute_weekly_series` (per-week hit-rate + a
  rolling-average overlay).
* Weekly records — :func:`compute_weekly_records` (per-week combined W-L across
  all three targets, for the Season page's week strip).
```

Then append to the end of `api/season_metrics.py`:

```python


# ---------------------------------------------------------------------------
# Week-by-week combined record (the Season page's week strip)
# ---------------------------------------------------------------------------


def compute_weekly_records(rows: Sequence[dict]) -> list[dict[str, int]]:
    """Per-week combined W-L across all three targets, for the Season page's week strip.

    A week's ``wins`` and ``losses`` count every DECIDED pick of that week -- Winner, Spread and
    Totals together -- through the same classifiers and the same completed-row filter as every
    other figure in this module, so the strip's totals equal the KPI tiles' hit and decided counts
    by construction. Pushes, ties and rows with no line or no prediction are excluded and never
    counted as a loss. A week with no decided pick is omitted rather than reported as 0-0, which
    would read as a measured record.

    Returns ``[{"week": int, "wins": int, "losses": int}, ...]`` in ascending week order.
    """
    per_week: dict[int, list[int]] = {}
    for r in _completed_sorted(rows):
        week = int(_num(r.get("week")))
        for target in _TARGETS:
            outcome = _OUTCOME_FNS[target](r)
            if outcome is None:
                continue  # push / tie / no line -> neither a win nor a loss
            bucket = per_week.setdefault(week, [0, 0])
            bucket[0 if outcome else 1] += 1
    return [
        {"week": week, "wins": per_week[week][0], "losses": per_week[week][1]}
        for week in sorted(per_week)
    ]
```

- [ ] **Step 5: Re-export it from the season chart module**

In `api/charts/season.py`, in the module docstring replace:

```
* :func:`compute_season_kpis`, :func:`compute_cumulative_series`,
  :func:`compute_weekly_series` — thin re-exports of the
```

with:

```
* :func:`compute_season_kpis`, :func:`compute_cumulative_series`,
  :func:`compute_weekly_series`, :func:`compute_weekly_records` — thin re-exports of the
```

Replace the import block:

```python
from api.season_metrics import (
    BREAKEVEN_WIN_RATE,
    compute_cumulative_series,
    compute_season_kpis,
    compute_weekly_series,
)
```

with:

```python
from api.season_metrics import (
    BREAKEVEN_WIN_RATE,
    compute_cumulative_series,
    compute_season_kpis,
    compute_weekly_records,
    compute_weekly_series,
)
```

and in `__all__` replace:

```python
    "compute_season_kpis",
    "compute_weekly_series",
```

with:

```python
    "compute_season_kpis",
    "compute_weekly_records",
    "compute_weekly_series",
```

- [ ] **Step 6: Write the weekly records into the season KPI blob**

In `api/charts/prerender.py`, replace:

```python
        try:
            charts[f"season_kpis_{season}"] = json.dumps(
                _season_charts.compute_season_kpis(rows_s),
            )
```

with:

```python
        try:
            kpis = _season_charts.compute_season_kpis(rows_s)
            # The week strip's per-week combined record rides inside the SAME blob, so the strip
            # and the KPI tiles are always written by one population run and cannot disagree.
            kpis["weeks"] = _season_charts.compute_weekly_records(rows_s)
            charts[f"season_kpis_{season}"] = json.dumps(kpis)
```

(The `except` branch below it is unchanged: a failure still stores `{}`.)

- [ ] **Step 7: Give the fixture KPI blob a `weeks` list**

In `tests/api/conftest.py`, replace:

```python
    # Phase 18: marker rows for every season chart_id (one set per fixture
    # season). The season_kpis_<s> family carries a decodable JSON DICT (the
    # per-target hit-rate + W-L record shape compute_season_kpis returns);
```

with:

```python
    # Phase 18: marker rows for every season chart_id (one set per fixture
    # season). The season_kpis_<s> family carries a decodable JSON DICT (the
    # per-target hit-rate + W-L record shape compute_season_kpis returns, plus
    # the per-week "weeks" list population adds for the Season week strip);
```

and replace:

```python
                    "record_wins": 11,
                    "record_losses": 5,
                    "record": "11-5",
                },
```

with:

```python
                    "record_wins": 11,
                    "record_losses": 5,
                    "record": "11-5",
                    # 26 wins / 20 losses in total = the 26 hits and 46 decided picks above.
                    "weeks": [
                        {"week": 1, "wins": 14, "losses": 9},
                        {"week": 2, "wins": 12, "losses": 11},
                    ],
                },
```

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run pytest tests/test_season_metrics.py -v`
Expected: all PASS (the 4 new tests included).

Run: `uv run pytest tests/test_charts.py -k "season" -v`
Expected: all PASS, including `test_season_kpi_blob_carries_the_weekly_records`, `test_season_prerender_covers_every_fixture_season` and `test_season_prerender_failure_isolation`.

Run: `uv run pytest tests/api/test_pages.py -k season -v`
Expected: all PASS (the extra blob key is ignored by the current template).

- [ ] **Step 9: Lint and type-check**

Run: `uv run ruff check api/season_metrics.py api/charts/season.py api/charts/prerender.py tests/test_season_metrics.py tests/test_charts.py tests/api/conftest.py`
Run: `uv run ruff format api/season_metrics.py api/charts/season.py api/charts/prerender.py tests/test_season_metrics.py tests/test_charts.py tests/api/conftest.py`
Run: `uv run pyright api/season_metrics.py api/charts/season.py api/charts/prerender.py tests/test_season_metrics.py tests/test_charts.py tests/api/conftest.py`
Expected: no ruff error, and 0 pyright errors in the three `api/` files. In the three test files, no pyright error on a line this task wrote; an older one predates the branch -- note it in the task report and leave it.

- [ ] **Step 10: Commit**

```bash
git add api/season_metrics.py api/charts/season.py api/charts/prerender.py tests/test_season_metrics.py tests/test_charts.py tests/api/conftest.py
git commit -m "$(cat <<'EOF'
feat(redesign): per-week combined record in the season KPI blob

compute_weekly_records counts each week's decided Winner, Spread and Totals
picks through the same classifiers as the KPI tiles, and population stores it
under "weeks" in the existing season_kpis_<season> blob. No new table.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 14: Season page restyle and the week strip

**Files:**
- Create: `web/templates/components/_chart_panel.html`
- Create: `web/templates/components/_week_strip.html`
- Modify (full rewrite): `web/templates/pages/season.html`
- Modify: `api/routes/pages.py` (the return statement of `_build_season_context`)
- Modify: `tests/api/test_pages.py` (the `_Service` stub in `test_a_season_with_nothing_graded_renders_the_empty_state_not_zeroes`; the `"WP Hit Rate"` assertion in `test_season_error_state_wired_and_distinct_from_empty`)
- Modify: `tests/api/test_fragments.py` (the `"WP Hit Rate"` assertion in `test_season_fragment_returns_block_only`)
- Create: `tests/api/test_season_week_strip.py`
- Regenerate: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes: Task 13's `kpis["weeks"]`; the contract's classes `label`, `display`, `panel`, `panel-title`, `chart-container` and macros `bc.section_head`, `bc.stat_tile`; `DataService.get_current_slate() -> tuple[int, int] | None` (existing); Task 5's restyled `_season_tracking_selector.html`, `_empty_state.html` and `_error_state.html`, whose include parameters are unchanged.
- Produces:
  - `components/_chart_panel.html`, macro `chart_panel(title, chart_html, empty_body, min_height=380, panel_id=none)`: a `panel` holding one pre-rendered chart, or the shared "Chart unavailable" empty state with `empty_body` as its text. Reused by Task 15.
  - `components/_week_strip.html`, included with `weeks` (the `kpis["weeks"]` list) and `current_week` (`int | None`).
  - Season context key `current_slate_week: int | None`.

- [ ] **Step 1: Write the failing tests**

Create `tests/api/test_season_week_strip.py`:

```python
"""The Season page's week-by-week strip, its KPI tiles and its context (redesign Task 14).

The strip is drawn from the per-week records population stores in the season KPI blob
(``api.season_metrics.compute_weekly_records``). Nothing is computed on the request path: the
win/loss bar is sized by CSS flex-grow set to the stored counts.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi.testclient import TestClient

from api.dependencies import templates


def _render_strip(weeks: list[dict[str, int]], current_week: int | None) -> str:
    return templates.env.get_template("components/_week_strip.html").render(
        weeks=weeks, current_week=current_week
    )


def _tile(html: str, week: int) -> str:
    match = re.search(rf'<li data-week="{week}"[^>]*>.*?</li>', html, re.S)
    assert match, f"no tile for week {week}"
    return match.group(0)


class TestTheWeekStrip:
    def test_renders_the_eighteen_regular_season_weeks(self) -> None:
        html = _render_strip([{"week": 1, "wins": 27, "losses": 21}], None)
        assert html.count("<li data-week=") == 18

    def test_a_graded_week_shows_its_record_and_a_win_loss_bar(self) -> None:
        tile = _tile(_render_strip([{"week": 2, "wins": 26, "losses": 22}], None), 2)
        assert "26-22" in tile
        assert "flex: 26 1 0%" in tile
        assert "flex: 22 1 0%" in tile
        assert "bg-green-500" in tile
        assert "bg-red-500" in tile
        assert 'aria-label="Week 2: 26 wins, 22 losses"' in tile
        assert "opacity-40" not in tile

    def test_a_week_with_no_graded_pick_is_dimmed_and_never_reads_0_0(self) -> None:
        tile = _tile(_render_strip([{"week": 1, "wins": 27, "losses": 21}], None), 5)
        assert "opacity-40" in tile
        assert "0-0" not in tile
        assert "bg-green-500" not in tile
        assert 'aria-label="Week 5: no graded picks"' in tile

    def test_the_current_week_is_outlined_and_marked_once(self) -> None:
        html = _render_strip([{"week": 3, "wins": 26, "losses": 22}], 3)
        tile = _tile(html, 3)
        assert "outline-accent" in tile
        assert 'aria-current="true"' in tile
        assert html.count("outline-accent") == 1
        assert html.count('aria-current="true"') == 1

    def test_no_current_week_outlines_nothing(self) -> None:
        html = _render_strip([{"week": 3, "wins": 26, "losses": 22}], None)
        assert "outline-accent" not in html
        assert "aria-current" not in html

    def test_postseason_weeks_extend_the_strip(self) -> None:
        html = _render_strip([{"week": 20, "wins": 2, "losses": 1}], None)
        assert html.count("<li data-week=") == 20
        assert "2-1" in _tile(html, 20)


class _Request:
    def url_for(self, name: str, **path_params: object) -> str:
        return f"/{name}/{path_params.get('path', '')}"


class _Service:
    """The five reads ``_build_season_context`` makes, with a chosen current slate."""

    def __init__(self, slate: tuple[int, int] | None, kpis: dict[str, Any]) -> None:
        self._slate = slate
        self._kpis = kpis

    def get_chart_html(self, chart_id: str) -> str:
        return f'<div data-chart-id="{chart_id}">chart</div>'

    def get_season_kpis(self, season: int) -> dict[str, Any]:
        return self._kpis

    def get_prediction_seasons(self) -> list[int]:
        return [2026, 2025]

    def get_cache_meta(self) -> dict[str, Any]:
        return {}

    def get_current_slate(self) -> tuple[int, int] | None:
        return self._slate


_GRADED_KPIS: dict[str, Any] = {
    "wp_hit_rate": 50.0,
    "wp_decided": 2,
    "wp_hits": 1,
    "ats_hit_rate": None,
    "ats_decided": 0,
    "ats_hits": 0,
    "ou_hit_rate": None,
    "ou_decided": 0,
    "ou_hits": 0,
    "record": "1-1",
}


class TestTheSeasonContext:
    def test_the_current_slate_week_is_passed_only_for_its_own_season(self) -> None:
        from api.routes.pages import _build_season_context

        kpis = _GRADED_KPIS | {"weeks": [{"week": 1, "wins": 1, "losses": 1}]}
        live = _build_season_context(_Service((2026, 4), kpis), 2026, _Request())  # type: ignore[arg-type]
        past = _build_season_context(_Service((2026, 4), kpis), 2025, _Request())  # type: ignore[arg-type]
        offseason = _build_season_context(_Service(None, kpis), 2026, _Request())  # type: ignore[arg-type]
        assert live["current_slate_week"] == 4
        assert past["current_slate_week"] is None
        assert offseason["current_slate_week"] is None

    def test_a_blob_without_weeks_renders_no_strip(self) -> None:
        """A cache built before the strip existed has no ``weeks`` key: no strip, and no error."""
        from api.routes.pages import _build_season_context

        context = _build_season_context(_Service(None, dict(_GRADED_KPIS)), 2026, _Request())  # type: ignore[arg-type]
        html = templates.env.get_template("pages/season.html").render(context)
        assert "Winner hit rate" in html
        assert "Week by week" not in html
        assert "<li data-week=" not in html


class TestTheSeasonPage:
    def test_renders_the_week_strip_from_the_kpi_blob(self, test_client: TestClient) -> None:
        from tests.api.conftest import _FIXTURE_SEASONS

        html = test_client.get(f"/season?season={max(_FIXTURE_SEASONS)}").text
        assert "Week by week" in html
        strip = html.split("Week by week", 1)[1]
        assert "14-9" in _tile(strip, 1)
        assert "12-11" in _tile(strip, 2)

    def test_the_kpi_tiles_use_the_bet_type_names_and_counts(
        self, test_client: TestClient
    ) -> None:
        from tests.api.conftest import _FIXTURE_SEASONS

        html = test_client.get(f"/season?season={max(_FIXTURE_SEASONS)}").text
        for label in ("Winner hit rate", "Spread hit rate", "Totals hit rate", "Record (W-L)"):
            assert label in html, label
        assert "67.7%" in html
        assert "11 of 16" in html  # wp_hits of wp_decided, both stored in the blob
        assert "straight-up winner picks" in html
        assert "WP Hit Rate" not in html

    def test_the_season_heading_is_inside_the_swapped_block(
        self, test_client: TestClient
    ) -> None:
        """A season swap must never leave last season's name above this season's numbers."""
        from tests.api.conftest import _FIXTURE_SEASONS

        older = min(_FIXTURE_SEASONS)
        response = test_client.get(
            f"/fragments/season?season={older}", headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
        assert f"{older} Season" in response.text
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/api/test_season_week_strip.py -v`
Expected: FAIL -- `jinja2.exceptions.TemplateNotFound: components/_week_strip.html` for the strip tests, `KeyError: 'current_slate_week'` for the context test, and the page tests fail on the missing labels.

- [ ] **Step 3: Pass the live slate's week into the season context**

In `api/routes/pages.py`, inside `_build_season_context`, replace:

```python
    if not season_has_graded_games(kpis):
        kpis = {}
    shows_numbers = season is not None and bool(kpis)
    return {
        "request": request,
        "charts": charts,
        "kpis": kpis,
        "available_seasons": service.get_prediction_seasons(),
        "current_season": season,
        "current_path": "/season",
        "cache_meta": service.get_cache_meta(),
        "old_rule_scope": DataService.old_rule_scope([season] if shows_numbers else []),
    }
```

with:

```python
    if not season_has_graded_games(kpis):
        kpis = {}
    shows_numbers = season is not None and bool(kpis)
    slate = service.get_current_slate()
    return {
        "request": request,
        "charts": charts,
        "kpis": kpis,
        "available_seasons": service.get_prediction_seasons(),
        "current_season": season,
        "current_path": "/season",
        "cache_meta": service.get_cache_meta(),
        "old_rule_scope": DataService.old_rule_scope([season] if shows_numbers else []),
        # The live slate's week, outlined on the week strip -- only while the page shows that
        # slate's season. A past season has no "current" week to point at.
        "current_slate_week": (
            slate[1]
            if slate is not None and season is not None and slate[0] == season
            else None
        ),
    }
```

- [ ] **Step 4: Create the shared chart panel**

Create `web/templates/components/_chart_panel.html`:

```jinja
{# One pre-rendered Plotly chart in a titled panel, or the shared "Chart unavailable" empty state
   when the cache holds no HTML for it. Import with
   {% import "components/_chart_panel.html" as cp %} -- inside the block when the block is served
   on its own by an HTMX swap (jinja2-fragments does not run a page's top-level imports).

   chart_html is the cached HTML itself (or none); empty_body is the empty state's explanation,
   which differs per chart family. The chart HTML is trusted: it was generated by api/charts at
   population time, never taken from a request. panel_id makes the panel an in-page anchor
   (Track Record's #clv), with the same scroll margin the page's section anchors use. #}
{% macro chart_panel(title, chart_html, empty_body, min_height=380, panel_id=none) -%}
<div class="panel{% if panel_id %} scroll-mt-6{% endif %}"{% if panel_id %} id="{{ panel_id }}"{% endif %}>
  {% if title %}<h3 class="panel-title mb-3">{{ title }}</h3>{% endif %}
  <div class="chart-container" style="min-height: {{ min_height }}px;">
    {% if chart_html %}
      {{ chart_html|safe }}
    {% else %}
      {% with heading="Chart unavailable", body=empty_body, action_text=none, action_url=none %}
        {% include "components/_empty_state.html" %}
      {% endwith %}
    {% endif %}
  </div>
</div>
{%- endmacro %}
```

- [ ] **Step 5: Create the week strip**

Create `web/templates/components/_week_strip.html`:

```jinja
{# Week-by-week strip for the Season page. Pass: weeks -- the season KPI blob's "weeks" list of
   {"week", "wins", "losses"}, written at population time by
   api.season_metrics.compute_weekly_records -- and current_week, the live slate's week while this
   is the live season, else none.

   NOTHING IS COMPUTED HERE (UIAP-01). The bar's two halves are sized by CSS flex-grow set to the
   stored win and loss COUNTS, so the proportion is drawn without a division in the template.
   Green and red are allowed because every tick is a realised result. A week with no decided pick
   shows a dot and is dimmed -- never "0-0", which would read as a measured record.

   18 tiles for the regular season; a postseason week in the data extends the strip. #}
{% set by_week = {} %}
{% for w in weeks %}{% set _ = by_week.update({w.week: w}) %}{% endfor %}
{% set last_week = ([18] + (weeks | map(attribute="week") | list)) | max %}
<ol class="grid grid-cols-6 sm:grid-cols-9 lg:grid-cols-18 gap-1" aria-label="Week-by-week record, all three bet types combined">
  {% for n in range(1, last_week + 1) %}
  {% set rec = by_week.get(n) %}
  {% set is_current = current_week is not none and n == current_week %}
  <li data-week="{{ n }}" class="bg-panel px-1 pt-1.5 pb-1.5 text-center{% if is_current %} outline-2 outline-offset-1 outline-accent{% endif %}{% if not rec %} opacity-40{% endif %}"{% if is_current %} aria-current="true"{% endif %} aria-label="Week {{ n }}: {% if rec %}{{ rec.wins }} wins, {{ rec.losses }} losses{% else %}no graded picks{% endif %}">
    <span class="label block">Wk {{ n }}</span>
    <span class="block font-display font-extrabold text-[15px] text-fg num">{% if rec %}{{ rec.wins }}-{{ rec.losses }}{% else %}&middot;{% endif %}</span>
    <span class="flex h-1 mt-1 bg-panel-2" aria-hidden="true">
      {% if rec %}
      <span class="bg-green-500" style="flex: {{ rec.wins }} 1 0%"></span>
      <span class="bg-red-500" style="flex: {{ rec.losses }} 1 0%"></span>
      {% endif %}
    </span>
  </li>
  {% endfor %}
</ol>
```

- [ ] **Step 6: Rewrite the Season page**

Replace the whole of `web/templates/pages/season.html` with:

```jinja
{% extends "base.html" %}
{% block title %}Season Tracking{% endblock %}
{% block content %}

{# Season selector (no "All Seasons" option -- D-04), OUTSIDE the swap target so it stays in place
   while #season-content swaps. The big season heading lives INSIDE the block, so a swap can never
   leave last season's name above this season's numbers. #}
<div class="flex flex-wrap items-center justify-between gap-3 mb-3">
  <p class="label">Season tracking</p>
  {% include "components/_season_tracking_selector.html" %}
</div>

{# Error-state source for the HTMX failed-swap path (D-17). Kept OUTSIDE
   #season-content (inside an inert <template>) so it survives innerHTML swaps and
   is never rendered on the happy path. The selector's `hx-on::response-error`
   handler copies this markup into #season-content when a swap returns 4xx/5xx --
   HTMX leaves the target untouched on a non-200 response (detail.shouldSwap is
   false; verified against htmx.org/events for htmx 2.0.4), so without this the
   content area would go blank/stale. Uses the LOCKED /season error copy and the
   shared _error_state.html (red) -- DISTINCT from the gray "No completed games
   yet" empty state below, which is a no-data case, not a failure. #}
<template id="season-error-template">
  {% with message="Could not load season data",
          recovery_text="Refresh the page or rebuild the cache. If the problem persists, re-run the daily pipeline." %}
    {% include "components/_error_state.html" %}
  {% endwith %}
</template>

{# HTMX swap target: the fragment route returns exactly this block (D-11). #}
<div id="season-content">
{% block season_tracking_content %}
{# Imported INSIDE the block: jinja2-fragments renders this block on its own for the HTMX swap and
   does not run the template's top-level statements, so a top-level import would be undefined. #}
{% import "components/_broadcast.html" as bc %}
{% import "components/_chart_panel.html" as cp %}

{% set cumulative_id = "season_cumulative_" ~ current_season %}
{% set weekly_id = "season_weekly_" ~ current_season %}

<header class="mb-4">
  <h1 class="display text-4xl md:text-5xl leading-none text-fg">{% if current_season is not none %}{{ current_season }} Season{% else %}Season{% endif %}</h1>
  <p class="text-sm text-muted mt-2 max-w-2xl">How the picks are grading, week by week -- Winner, Spread and Totals, graded on the same rules as the rest of the site.</p>
</header>

{# Old-rule label for the selected season (R16 / D33.2-07). The route passes a known-empty
   scope when the season has nothing to show, so the empty state below carries no label. #}
{% with scope=old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}

{# Whole-season empty guard FIRST (D-02): a season with no GRADED game renders a
   single empty state instead of the KPI strip + charts. The route passes an EMPTY
   kpis for such a season (33.2 review C2 WR-02): its KPI blob exists, since
   population builds one for every season in predictions, but its counts are all
   zero, and rendering it showed "0.0%" and "0-0" as if they were measurements.
   This is the offseason reality for a not-yet-played season -- it is "no data"
   (gray), NOT an error.
   WR-04: gate on `current_season is none` explicitly first (empty DB ->
   _normalize_season returns None). #}
{% if current_season is none or not kpis %}
  {% with heading="No completed games yet",
          body="This season has no completed game with a pick to grade in the cache yet. Once results arrive from the daily run, weekly and cumulative accuracy will appear here.",
          action_text=none, action_url=none %}
    {% include "components/_empty_state.html" %}
  {% endwith %}
{% else %}

{# Section 1: the season-to-date scoreboard (D-08). Every figure is read from the KPI blob as
   stored; the strings below only format it. Tile accents are the bet-type colour tokens every
   chart on the site uses (Winner yellow, Spread cyan, Totals violet), read from the Task 1
   variables as Task 5's week summary does, so a token change cannot leave a stale hex here. The Record tile is the WINNER record
   only, so its sub-line says so -- the week strip below combines all three bet types. #}
{% set wp_rate = ("%.1f"|format(kpis.wp_hit_rate) ~ "%") if kpis.wp_hit_rate is not none else "--" %}
{% set ats_rate = ("%.1f"|format(kpis.ats_hit_rate) ~ "%") if kpis.ats_hit_rate is not none else "--" %}
{% set ou_rate = ("%.1f"|format(kpis.ou_hit_rate) ~ "%") if kpis.ou_hit_rate is not none else "--" %}
{% set wp_sub = (kpis.wp_hits|default(0)|string ~ " of " ~ kpis.wp_decided|default(0)|string) if kpis.wp_decided|default(0) else "no graded picks" %}
{% set ats_sub = (kpis.ats_hits|default(0)|string ~ " of " ~ kpis.ats_decided|default(0)|string) if kpis.ats_decided|default(0) else "no graded picks" %}
{% set ou_sub = (kpis.ou_hits|default(0)|string ~ " of " ~ kpis.ou_decided|default(0)|string) if kpis.ou_decided|default(0) else "no graded picks" %}
<section class="mb-8" aria-label="Season-to-date scoreboard">
  <div class="grid grid-cols-2 md:grid-cols-4 gap-2">
    {{ bc.stat_tile("Winner hit rate", wp_rate, sub=wp_sub, accent="var(--color-target-wp)") }}
    {{ bc.stat_tile("Spread hit rate", ats_rate, sub=ats_sub, accent="var(--color-target-ats)") }}
    {{ bc.stat_tile("Totals hit rate", ou_rate, sub=ou_sub, accent="var(--color-target-ou)") }}
    {{ bc.stat_tile("Record (W-L)", kpis.record if kpis.record else "--", sub="straight-up winner picks", accent="var(--color-fg)") }}
  </div>
  <p class="text-xs text-muted mt-3 max-w-3xl">Pushes are excluded from the denominator.</p>
</section>

{# Section 2: the week strip. A cache built before population wrote "weeks" has none, and then
   the strip is simply not shown -- never an empty or zeroed strip. #}
{% if kpis.weeks %}
<section class="mb-8" aria-label="Week by week">
  {{ bc.section_head("Week by week", meta="all three bet types combined") }}
  {% with weeks=kpis.weeks, current_week=current_slate_week|default(none) %}
    {% include "components/_week_strip.html" %}
  {% endwith %}
</section>
{% endif %}

{# Section 3: Cumulative Accuracy (DASH-07) #}
<section class="mb-8">
  {{ bc.section_head("Cumulative Accuracy", meta="running season-to-date hit rate") }}
  <p class="text-sm text-muted mb-3 max-w-3xl">Running season-to-date hit rate for each target. The line is the trend -- where the season has settled so far.</p>
  {{ cp.chart_panel(none, charts[cumulative_id], "Season-tracking data has not been generated. Run the cache rebuild to populate this view.", min_height=320) }}
</section>

{# Section 4: Weekly Performance (DASH-08) #}
<section class="mb-8">
  {{ bc.section_head("Weekly Performance", meta="per-week hit rate with a rolling average") }}
  <p class="text-sm text-muted mb-3 max-w-3xl">Hit rate week by week, with a rolling-average overlay to smooth small-sample weeks. This is the model's recent form.</p>
  {{ cp.chart_panel(none, charts[weekly_id], "Season-tracking data has not been generated. Run the cache rebuild to populate this view.", min_height=320) }}
  <p class="text-xs text-muted mt-4 max-w-3xl">52.4% is the standard -110 break-even, exact for spread and totals. Moneyline (Winner) odds vary, so per-game breakeven differs -- treat this as a reference, not a per-game truth.</p>
</section>

{% endif %}

{% endblock %}
</div>
{% endblock %}
```

- [ ] **Step 7: Update the existing season tests that pin the old wording or the old stub**

In `tests/api/test_pages.py`, inside `test_season_error_state_wired_and_distinct_from_empty`, replace:

```python
    assert "WP Hit Rate" in html
```

with:

```python
    assert "Winner hit rate" in html
```

In `tests/api/test_pages.py`, inside `test_a_season_with_nothing_graded_renders_the_empty_state_not_zeroes`, replace:

```python
        def get_cache_meta(self) -> dict:
            return {}
```

with:

```python
        def get_cache_meta(self) -> dict:
            return {}

        def get_current_slate(self) -> None:
            return None
```

In `tests/api/test_fragments.py`, inside `test_season_fragment_returns_block_only`, replace:

```python
    assert "WP Hit Rate" in html
```

with:

```python
    assert "Winner hit rate" in html
```

- [ ] **Step 8: Recompile the stylesheet**

Run: `./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify`
Expected: exits 0. Then confirm the new utilities were emitted:
Run: `grep -oE "grid-cols-18|outline-accent|opacity-40" web/static/css/tailwind-compiled.css | sort -u`
Expected: exactly three lines, `grid-cols-18`, `opacity-40` and `outline-accent`. (`grep -c` cannot prove this: the minified sheet is one line, so it prints 1 as soon as any ONE of the three exists.)

- [ ] **Step 9: Run the tests to verify they pass**

Run: `uv run pytest tests/api/test_season_week_strip.py -v`
Expected: all PASS.

Run: `uv run pytest tests/api/test_pages.py -k "season" -v`
Expected: all PASS.

Run: `uv run pytest tests/api/test_fragments.py -k "season" -v`
Expected: all PASS.

Run: `uv run pytest tests/unit/test_page_labels.py -k "season" -v`
Expected: all PASS (the `<title>` still carries "Season Tracking"; one label for a past season, none for 2026, one when unwired).

Run: `uv run pytest tests/api/test_page_labels_routes.py -k "season" -v`
Expected: all PASS.

- [ ] **Step 10: Lint and type-check**

Run: `uv run ruff check api/routes/pages.py tests/api/test_season_week_strip.py tests/api/test_pages.py tests/api/test_fragments.py`
Run: `uv run ruff format api/routes/pages.py tests/api/test_season_week_strip.py tests/api/test_pages.py tests/api/test_fragments.py`
Run: `uv run pyright api/routes/pages.py tests/api/test_season_week_strip.py tests/api/test_pages.py tests/api/test_fragments.py`
Expected: no ruff error; 0 pyright errors in `api/routes/pages.py` and `tests/api/test_season_week_strip.py`. In `test_pages.py` and `test_fragments.py`, no pyright error on a line this task wrote; an older one predates the branch -- note it in the task report and leave it.

- [ ] **Step 11: Commit**

```bash
git add api/routes/pages.py web/templates/components/_chart_panel.html web/templates/components/_week_strip.html web/templates/pages/season.html web/static/css/tailwind-compiled.css tests/api/test_season_week_strip.py tests/api/test_pages.py tests/api/test_fragments.py
git commit -m "$(cat <<'EOF'
feat(redesign): Season page in the Broadcast style with a week-by-week strip

KPI tiles keyed to the bet-type colours (Winner / Spread / Totals hit rate,
plus the straight-up Record), an 18-week strip drawn from the stored weekly
records with the live slate's week outlined, and the two season charts in the
shared chart panel. The season heading moved inside the swapped block so a
season change can never leave a stale title above new numbers.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 15: Track Record, How It Works, and the retired-page redirects

**Files:**
- Create: `web/templates/pages/track_record.html`
- Create: `web/templates/pages/how_it_works.html`
- Delete: `web/templates/pages/performance.html`, `web/templates/pages/backtest.html`, `web/templates/pages/betting.html`, `web/templates/pages/insights.html`
- Modify: `api/routes/pages.py` (module docstring; imports; `_build_betting_context`; new builders and block responders above the `# Routes` banner; the four retired handlers replaced)
- Modify: `api/routes/fragments.py` (docstring; imports; `performance_fragment`; `betting_fragment`)
- Create: `tests/api/test_merged_pages.py`
- Modify: `tests/unit/test_page_labels.py`, `tests/api/test_page_labels_routes.py`, `tests/api/test_pages.py`, `tests/api/test_cache_headers.py`, `tests/api/test_bets_page.py` (one route tuple)
- Modify: `PIPELINE.md` (Stage 7 page list) and `tests/unit/test_pipeline_md.py` (its pin)
- Regenerate: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes: `components/_chart_panel.html` (Task 14); `components/_broadcast.html` macros `section_head` and `stat_tile`; classes `skew-control`, `unskew`, `panel`, `panel-title`, `label`, `num`, `display`; Task 3's nav, which already links `/track-record` and `/how-it-works` in both the desktop row and the mobile menu; Task 5's restyled `_season_selector.html`, `_betting_scope_toggle.html`, `_export_buttons.html`, `_empty_state.html` with unchanged include parameters and HTMX wiring; Task 11 already pointed the /bets cross-link at `/track-record#betting-sim`.
- Produces (all in `api/routes/pages.py`):
  - `_season_metrics_block(service: DataService, season: int | None) -> dict[str, Any]` (keys `season_metrics`, `season_metrics_old_rule_scope`; shared by the page builder and the season block responder)
  - `_build_track_record_context(service: DataService, season: int | None, scope: str, request: Request) -> dict[str, Any]`
  - `_build_how_it_works_context(service: DataService, request: Request) -> dict[str, Any]`
  - `_performance_block_response(request: Request, service: DataService, season: int | None) -> Response`
  - `_betting_block_response(request: Request, service: DataService, scope: str) -> Response`
  - `_legacy_redirect(request: Request, path: str, anchor: str | None = None) -> RedirectResponse`
  - Routes `GET /track-record` (`season: str | None`, `scope: str`), `GET /how-it-works`, and the 301s for `/performance`, `/backtest`, `/insights`, `/betting`.
  - Context keys: `summary_old_rule_scope`, `season_metrics_old_rule_scope`, `backtest_old_rule_scope`, `betting_old_rule_scope` on Track Record; `old_rule_scope` on How It Works.

- [ ] **Step 1: Write the failing route tests**

Create `tests/api/test_merged_pages.py`:

```python
"""Track Record, How It Works and the retired-page redirects (redesign Task 15).

The four retired pages -- /performance, /backtest, /insights, /betting -- were merged into two:
Track Record (how the model has done) and How It Works (why it predicts what it does). These
tests pin where every chart landed and that each appears once, that every retired URL still works
(a 301 carrying its query string to the right section, or the block itself for an HTMX request),
and that both new pages keep the read-only cache contract. Their Cache-Control (pages and 301s) is
pinned once, in tests/api/test_cache_headers.py, and the nav in tests/api/test_pages.py.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from api.charts import INSIGHTS_CHART_IDS

PAGES_DIR = Path(__file__).resolve().parents[2] / "web" / "templates" / "pages"

_VS_MARKET = tuple(c for c in INSIGHTS_CHART_IDS if c.startswith("insights_model_vs_market_"))
_HOW_IT_WORKS_INSIGHTS = tuple(c for c in INSIGHTS_CHART_IDS if c not in _VS_MARKET)


class TestTrackRecord:
    def test_renders_every_section_with_its_anchor(self, test_client: TestClient) -> None:
        response = test_client.get("/track-record")
        assert response.status_code == 200
        html = response.text
        for anchor in ("summary", "seasons", "vs-market", "clv", "betting-sim"):
            assert f'id="{anchor}"' in html, f"missing section anchor #{anchor}"
        # The page's own <h1>: "Track Record" alone would also be found in the nav and <title>.
        assert re.search(r"<h1[^>]*>Track Record</h1>", html), "the page heading is missing"
        for heading in (
            "All-time summary",
            "Total Games",
            "Season by season",
            "Model vs Market",
            "Cumulative CLV",
            "Betting Simulation",
            "Equity Curve",
            "ROI Breakdown",
            "Edge Distribution",
        ):
            assert heading in html, heading

    def test_places_each_backtest_chart_once(self, test_client: TestClient) -> None:
        html = test_client.get("/track-record").text
        assert html.count("test heatmap") == 1
        assert html.count("test clv") == 1
        for chart_id in _VS_MARKET:
            assert html.count(f'data-chart-id="{chart_id}"') == 1, chart_id
        # The duplicates the merge removed: the backtest equity curve repeats the betting equity
        # curve, and WP calibration (with the rest of /insights' model half) lives on How It Works.
        assert "test equity" not in html
        assert 'data-chart-id="calibration"' not in html
        for chart_id in _HOW_IT_WORKS_INSIGHTS:
            assert f'data-chart-id="{chart_id}"' not in html, chart_id

    def test_renders_the_recommended_betting_charts_by_default(
        self, test_client: TestClient
    ) -> None:
        from tests.api.conftest import _BETTING_CHART_BASES

        html = test_client.get("/track-record").text
        for base in _BETTING_CHART_BASES:
            if base in ("betting_kpis", "betting_roi_table"):
                continue
            assert f'data-chart-id="{base}_recommended"' in html, base
            assert f'data-chart-id="{base}_all"' not in html, base

    def test_the_scope_parameter_selects_the_betting_scope(
        self, test_client: TestClient
    ) -> None:
        html = test_client.get("/track-record?scope=all").text
        assert 'data-chart-id="betting_equity_all"' in html
        assert 'data-chart-id="betting_equity_recommended"' not in html

    def test_an_unknown_scope_falls_back_to_recommended(self, test_client: TestClient) -> None:
        response = test_client.get("/track-record?scope=bogus")
        assert response.status_code == 200
        assert 'data-chart-id="betting_equity_recommended"' in response.text

    def test_the_season_parameter_selects_the_season(self, test_client: TestClient) -> None:
        html = test_client.get("/track-record?season=2023").text
        assert re.search(r'<option value="2023"\s+selected', html)

    @pytest.mark.parametrize("raw", ["abc", "--5", "", "1.5"])
    def test_a_malformed_season_degrades_to_all_seasons(
        self, test_client: TestClient, raw: str
    ) -> None:
        response = test_client.get(f"/track-record?season={raw}")
        assert response.status_code == 200
        assert re.search(r'<option value=""\s+selected', response.text)

    def test_an_empty_cache_renders_empty_states_not_a_500(
        self, empty_test_client: TestClient
    ) -> None:
        response = empty_test_client.get("/track-record")
        assert response.status_code == 200
        assert "Chart unavailable" in response.text
        assert "No backtest data" in response.text


class TestHowItWorks:
    def test_renders_its_sections(self, test_client: TestClient) -> None:
        response = test_client.get("/how-it-works")
        assert response.status_code == 200
        html = response.text
        # The page's own <h1>: "How It Works" alone would also be found in the nav and <title>.
        assert re.search(r"<h1[^>]*>How It Works</h1>", html), "the page heading is missing"
        for heading in ("Calibration", "What the models rely on", "Accuracy over time"):
            assert heading in html, heading
        for anchor in ("calibration", "features", "accuracy"):
            assert f'id="{anchor}"' in html, anchor

    def test_renders_the_model_charts_once_and_not_the_market_ones(
        self, test_client: TestClient
    ) -> None:
        html = test_client.get("/how-it-works").text
        assert html.count('data-chart-id="calibration"') == 1
        for chart_id in _HOW_IT_WORKS_INSIGHTS:
            assert html.count(f'data-chart-id="{chart_id}"') == 1, chart_id
        for chart_id in _VS_MARKET:
            assert f'data-chart-id="{chart_id}"' not in html, chart_id

    def test_an_empty_cache_renders_empty_states_not_a_500(
        self, empty_test_client: TestClient
    ) -> None:
        response = empty_test_client.get("/how-it-works")
        assert response.status_code == 200
        assert "Chart unavailable" in response.text


class TestRetiredUrls:
    @pytest.mark.parametrize(
        ("url", "location"),
        [
            ("/performance", "/track-record#seasons"),
            ("/performance?season=2023", "/track-record?season=2023#seasons"),
            ("/backtest", "/track-record"),
            ("/betting", "/track-record#betting-sim"),
            ("/betting?scope=all", "/track-record?scope=all#betting-sim"),
            ("/insights", "/how-it-works"),
        ],
    )
    def test_a_retired_url_redirects_permanently_with_its_query(
        self, test_client: TestClient, url: str, location: str
    ) -> None:
        response = test_client.get(url, follow_redirects=False)
        assert response.status_code == 301
        assert response.headers["location"] == location

    def test_an_old_season_bookmark_opens_that_season(self, test_client: TestClient) -> None:
        response = test_client.get("/performance?season=2023")
        assert response.status_code == 200
        assert response.url.path == "/track-record"
        assert re.search(r'<option value="2023"\s+selected', response.text)

    def test_an_old_scope_bookmark_opens_that_scope(self, test_client: TestClient) -> None:
        response = test_client.get("/betting?scope=all")
        assert response.status_code == 200
        assert response.url.path == "/track-record"
        assert 'data-chart-id="betting_equity_all"' in response.text

    def test_an_htmx_request_to_the_old_performance_url_gets_the_season_block(
        self, test_client: TestClient
    ) -> None:
        response = test_client.get(
            "/performance?season=2023", headers={"HX-Request": "true"}
        )
        assert response.status_code == 200
        html = response.text
        assert "<html" not in html
        assert "<nav" not in html
        assert "Season Metrics" in html

    def test_an_htmx_request_to_the_old_betting_url_gets_the_betting_block(
        self, test_client: TestClient
    ) -> None:
        response = test_client.get("/betting?scope=all", headers={"HX-Request": "true"})
        assert response.status_code == 200
        html = response.text
        assert "<html" not in html
        assert "<nav" not in html
        assert 'data-chart-id="betting_equity_all"' in html

    @pytest.mark.parametrize(
        "name", ["performance.html", "backtest.html", "betting.html", "insights.html"]
    )
    def test_the_retired_templates_are_gone(self, name: str) -> None:
        assert not (PAGES_DIR / name).exists()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/api/test_merged_pages.py -v`
Expected: FAIL -- `/track-record` and `/how-it-works` return 404, the retired URLs return 200 instead of 301, and the retired templates still exist.

- [ ] **Step 3: Rename the betting block's keys**

In `api/routes/pages.py`, inside `_build_betting_context`, replace the docstring paragraph:

```python
    Shared by both ``betting_page`` and ``betting_fragment`` so the cached-read
    contract lives in one place and cannot drift between the two handlers.
    """
```

with:

```python
    Shared by the Track Record page (``_build_track_record_context``) and
    ``_betting_block_response`` (the betting fragment and the retired ``/betting``
    URL's HTMX branch), so the cached-read contract lives in one place and cannot
    drift between them.
    """
```

and replace:

```python
        "current_scope": scope,
        "current_path": "/betting",
        "cache_meta": service.get_cache_meta(),
        # One block: every KPI, chart and ROI row is drawn from the betting simulation ledger.
        # Built here so betting_page and betting_fragment receive it from one place.
        "old_rule_scope": service.cached_span_old_rule_scope(BETTING_SEASON_RANGE_KEY),
    }
```

with:

```python
        "current_scope": scope,
        "current_path": "/track-record",
        "cache_meta": service.get_cache_meta(),
        # One block: every KPI, chart and ROI row is drawn from the betting simulation ledger.
        # Named for its block because Track Record carries four scopes side by side.
        "betting_old_rule_scope": service.cached_span_old_rule_scope(
            BETTING_SEASON_RANGE_KEY
        ),
    }
```

- [ ] **Step 4: Update the module docstring and imports**

In `api/routes/pages.py`, replace the module docstring's route list:

```python
Routes:
    GET /            -- This Week's predictions dashboard (landing page)
    GET /performance -- Historical performance view
    GET /backtest    -- Backtest analysis with Plotly charts
    GET /insights    -- Model insights (calibration, feature importance, vs market)
    GET /betting     -- Betting dashboard (KPI strip, equity, ROI, edge; scope toggle)
    GET /bets        -- Weekly bet list (ranked +EV bets, units, EV band; week selector)
    GET /games/{id}  -- Game detail drill-down (feature importance, market comparison)
"""
```

with:

```python
Routes:
    GET /             -- This Week's predictions dashboard (landing page)
    GET /bets         -- Weekly bet list (ranked +EV bets, units, EV band; week selector)
    GET /season       -- Season tracking (KPI strip, week strip, cumulative + weekly charts)
    GET /track-record -- Track Record: all-time summary, season metrics, model vs market,
                         closing-line value and the betting simulation (the merged /performance,
                         /backtest and /betting pages plus the market half of /insights)
    GET /how-it-works -- How It Works: methodology, calibration, feature importance, accuracy
    GET /games/{id}   -- Game detail drill-down (feature importance, market comparison)

Retired URLs answer with a 301 to the merged page, query string kept: /performance, /backtest,
/insights, /betting. An HTMX request to /performance or /betting gets the block it asked for.
"""
```

Replace:

```python
from fastapi import APIRouter, Depends, Query, Request
```

with:

```python
from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import RedirectResponse, Response
```

- [ ] **Step 5: Add the builders, the block responders and the redirect helper**

In `api/routes/pages.py`, insert this code immediately ABOVE the three-line banner

```python
# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
```

(i.e. directly after the end of `_compute_summary`). `_season_metrics_block` is the one builder for the season table and its scope, used by both the page and the block responder. The `"summary"` line in `_build_track_record_context` carries a comment naming `_compute_summary` a known, pre-existing UIAP-01 exception (a request-path average carried over from the retired /performance page); this is a comment only, with no behaviour change:

```python
# The insights chart family that moved to Track Record; the rest of INSIGHTS_CHART_IDS is
# How It Works'. Matched by prefix so the split cannot drift from the id tuple itself.
_MODEL_VS_MARKET_PREFIX = "insights_model_vs_market_"


def _season_metrics_block(service: DataService, season: int | None) -> dict[str, Any]:
    """The season-metrics table rows and their own old-rule scope (Track Record's season block).

    ONE builder for the full Track Record page and ``_performance_block_response`` (the season
    swap and the retired ``/performance`` URL's HTMX branch), so the table and its label are read
    one way. The scope comes from the seasons the table shows (R16 / D33.2-07).
    """
    season_metrics = _pivot_season_metrics(service.get_backtest_metrics(season=season))
    return {
        "season_metrics": season_metrics,
        "season_metrics_old_rule_scope": DataService.old_rule_scope(
            _rows_seasons(season_metrics)
        ),
    }


def _build_track_record_context(
    service: DataService,
    season: int | None,
    scope: str,
    request: Request,
) -> dict[str, Any]:
    """Assemble the Track Record page context from cached data only.

    Track Record merges the retired /performance, /backtest and /betting pages and the market half
    of /insights, so it reuses each one's cached reads instead of re-deriving any of them: the
    betting block comes whole from ``_build_betting_context`` (the builder the betting fragment
    uses), and the summary, season table and chart reads are the ones the retired handlers made.
    The backtest ``equity`` chart and the WP ``calibration`` chart are deliberately NOT read: the
    first repeats the betting equity curve, the second lives on How It Works. Their ids stay in the
    cache untouched.

    Four blocks show numbers, so four old-rule scopes, each from its block's own data (R16 /
    D33.2-07): the all-history summary (tiles + season heatmap) and the model-vs-market section
    (charts, aggregate table, cumulative CLV) span the whole backtest corpus whatever season is
    selected, the season table spans the rows it shows, and the betting block spans the simulation
    ledger.
    """
    betting = _build_betting_context(service, scope, request)
    backtest_scope = service.cached_span_old_rule_scope(BACKTEST_SEASON_RANGE_KEY)
    charts: dict[str, str | None] = {
        **betting["charts"],
        "clv": service.get_chart_html("clv"),
        "heatmap": service.get_chart_html("heatmap"),
    }
    for chart_id in INSIGHTS_CHART_IDS:
        if chart_id.startswith(_MODEL_VS_MARKET_PREFIX):
            charts[chart_id] = service.get_chart_html(chart_id)
    return {
        **betting,
        "charts": charts,
        "available_seasons": service.get_available_seasons(),
        "current_season": season,
        **_season_metrics_block(service, season),
        # Known, pre-existing UIAP-01 exception: _compute_summary averages the cached backtest
        # metric rows in the request path. It is carried over unchanged from the retired
        # /performance page; the redesign moves where it renders, not what it computes.
        "summary": _compute_summary(service),
        "aggregate_table": service.get_insights_aggregate_table(),
        "current_path": "/track-record",
        "summary_old_rule_scope": backtest_scope,
        "backtest_old_rule_scope": backtest_scope,
    }


def _build_how_it_works_context(service: DataService, request: Request) -> dict[str, Any]:
    """Assemble the How It Works page context from cached data only.

    The calibration, feature-importance and accuracy-trend half of the retired /insights page,
    plus the WP reliability chart (``calibration``) that /backtest and /insights both used to
    show. One block -- every chart is drawn from the backtest corpus -- so one old-rule scope.
    """
    charts: dict[str, str | None] = {
        chart_id: service.get_chart_html(chart_id)
        for chart_id in INSIGHTS_CHART_IDS
        if not chart_id.startswith(_MODEL_VS_MARKET_PREFIX)
    }
    charts["calibration"] = service.get_chart_html("calibration")
    return {
        "request": request,
        "charts": charts,
        "current_path": "/how-it-works",
        "cache_meta": service.get_cache_meta(),
        "old_rule_scope": service.cached_span_old_rule_scope(BACKTEST_SEASON_RANGE_KEY),
    }


def _performance_block_response(
    request: Request, service: DataService, season: int | None
) -> Response:
    """Render Track Record's ``performance_content`` block: the season-metrics table alone.

    Shared by ``/fragments/performance`` (the season selector's swap) and an HTMX request to the
    retired ``/performance`` URL, so the two cannot drift. The context is exactly what the block
    reads: the rows, the selected season, and the rows' own old-rule scope.
    """
    context = {
        "request": request,
        "current_season": season,
        **_season_metrics_block(service, season),
    }
    template_response = templates.TemplateResponse(
        request, "pages/track_record.html", context, block_name="performance_content"
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


def _betting_block_response(request: Request, service: DataService, scope: str) -> Response:
    """Render Track Record's ``betting_content`` block for an already-whitelisted *scope*.

    Shared by ``/fragments/betting`` (the scope toggle's swap) and an HTMX request to the retired
    ``/betting`` URL. Reads cached HTML/JSON only -- zero metric logic on the request path (D-20).
    """
    context = _build_betting_context(service, scope, request)
    template_response = templates.TemplateResponse(
        request, "pages/track_record.html", context, block_name="betting_content"
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


def _legacy_redirect(
    request: Request, path: str, anchor: str | None = None
) -> RedirectResponse:
    """301 a retired page URL to its merged page, keeping the query string and adding the anchor.

    The query string is carried across verbatim, so an old bookmark such as
    ``/performance?season=2023`` opens the same season on Track Record. *path* is always a fixed
    literal from this module, never built from the request, so the redirect cannot be pointed
    off-site. The response carries the page Cache-Control: a browser caches the move for a minute
    instead of permanently, so the target can still be changed later.
    """
    query = request.url.query
    target = f"{path}?{query}" if query else path
    if anchor:
        target = f"{target}#{anchor}"
    response = RedirectResponse(target, status_code=301)
    response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return response


```

- [ ] **Step 6: Replace the four retired handlers**

In `api/routes/pages.py`, delete the whole region from the line `@router.get("/performance")` down to, but NOT including, the line `@router.get("/bets")` -- that is the four handlers `performance_page`, `backtest_page`, `insights_page` and `betting_page` -- and put this code in its place:

```python
@router.get("/track-record")
def track_record_page(
    request: Request,
    season: str | None = Query(None),
    scope: str = Query(_DEFAULT_BETTING_SCOPE),
    service: DataService = Depends(get_data_service),
):
    """Serve the Track Record page (the merged /performance, /backtest and /betting pages).

    ``season`` selects the season-metrics table and is parsed defensively (an unparseable value
    means "all seasons", never a 422); ``scope`` selects the betting block and is whitelisted to
    all / recommended. They are the parameters the retired pages took, so a redirected bookmark
    keeps its meaning. Always a full page: the two swappable blocks are served by
    ``/fragments/performance`` and ``/fragments/betting``.
    """
    context = _build_track_record_context(
        service, _parse_int_param(season), _normalize_betting_scope(scope), request
    )
    template_response = templates.TemplateResponse(
        request, "pages/track_record.html", context
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/how-it-works")
def how_it_works_page(
    request: Request,
    service: DataService = Depends(get_data_service),
):
    """Serve the How It Works page: methodology, calibration, feature importance, accuracy.

    Static, with no filters. Every chart is pre-rendered during cache population and read here
    as cached HTML (no statistical logic on the request path).
    """
    context = _build_how_it_works_context(service, request)
    template_response = templates.TemplateResponse(
        request, "pages/how_it_works.html", context
    )
    template_response.headers["Cache-Control"] = PAGE_CACHE_CONTROL
    return template_response


@router.get("/performance")
def performance_retired(
    request: Request,
    season: str | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Retired: a full navigation is sent to Track Record's season section, query kept.

    An HTMX request (a page still holding the old selector markup) gets the season block it
    asked for instead, because htmx would otherwise swap a whole page into a table-sized target.
    """
    if request.headers.get("HX-Request"):
        return _performance_block_response(request, service, _parse_int_param(season))
    return _legacy_redirect(request, "/track-record", anchor="seasons")


@router.get("/backtest")
def backtest_retired(request: Request):
    """Retired: its four charts now live on Track Record (and WP calibration on How It Works)."""
    return _legacy_redirect(request, "/track-record")


@router.get("/insights")
def insights_retired(request: Request):
    """Retired: its model half is How It Works (the market half is on Track Record)."""
    return _legacy_redirect(request, "/how-it-works")


@router.get("/betting")
def betting_retired(
    request: Request,
    scope: str = Query(_DEFAULT_BETTING_SCOPE),
    service: DataService = Depends(get_data_service),
):
    """Retired: a full navigation is sent to Track Record's betting section, query kept.

    An HTMX request gets the betting block for the whitelisted scope, as the old page's own
    HX-Request branch did.
    """
    if request.headers.get("HX-Request"):
        return _betting_block_response(request, service, _normalize_betting_scope(scope))
    return _legacy_redirect(request, "/track-record", anchor="betting-sim")


```

- [ ] **Step 7: Point the two fragments at the shared block responders**

In `api/routes/fragments.py`, replace the docstring lines:

```python
    GET /fragments/performance -- Performance content fragment (season swap)
    GET /fragments/betting     -- Betting content fragment (All/Recommended scope swap)
```

with:

```python
    GET /fragments/performance -- Track Record's season-metrics block (season swap)
    GET /fragments/betting     -- Track Record's betting block (All/Recommended scope swap)
```

In the `from api.routes.pages import (...)` list, delete the two lines `    _build_betting_context,` and `    _pivot_season_metrics,`, and add the two lines `    _betting_block_response,` and `    _performance_block_response,` (keep the list alphabetical after the leading constants; leave every other entry -- including any Task 6 added -- as it is).

Replace the whole `performance_fragment` function with:

```python
@router.get("/performance")
def performance_fragment(
    request: Request,
    season: str | None = Query(None),
    service: DataService = Depends(get_data_service),
):
    """Return Track Record's ``performance_content`` block for the HTMX season swap.

    Renders only the season-metrics table, used when the season selector changes. The render is
    shared with the retired ``/performance`` URL's HTMX branch (``_performance_block_response``).
    """
    return _performance_block_response(request, service, _parse_int_param(season))
```

Replace the whole `betting_fragment` function with:

```python
@router.get("/betting")
def betting_fragment(
    request: Request,
    scope: str = Query(_DEFAULT_BETTING_SCOPE),
    service: DataService = Depends(get_data_service),
):
    """Return Track Record's ``betting_content`` block for the All/Recommended swap.

    ``scope`` is whitelisted to {"all", "recommended"} (default ``"recommended"``) through the
    same chokepoint the page uses, and the render is shared with the retired ``/betting`` URL's
    HTMX branch (``_betting_block_response``). Reads cached HTML/JSON only (D-20).
    """
    return _betting_block_response(request, service, _normalize_betting_scope(scope))
```

- [ ] **Step 8: Create the Track Record page**

Create `web/templates/pages/track_record.html`:

```jinja
{% extends "base.html" %}
{% import "components/_broadcast.html" as bc %}
{% import "components/_chart_panel.html" as cp %}
{% block title %}Track Record{% endblock %}
{% block content %}

<header class="mb-8">
  <h1 class="display text-4xl md:text-5xl leading-none text-fg">Track Record</h1>
  <p class="text-sm text-muted mt-2 max-w-2xl">How the model has done on past seasons -- season by season, against the betting market, and in a simulated betting account. Every figure on this page comes from a walk-forward backtest.</p>
  <nav aria-label="On this page" class="flex flex-wrap gap-2 mt-4">
    {% for anchor, text in [("summary", "Summary"), ("seasons", "Season by season"), ("vs-market", "Model vs market"), ("clv", "Closing-line value"), ("betting-sim", "Betting simulation")] %}
    <a href="#{{ anchor }}" class="skew-control"><span class="unskew">{{ text }}</span></a>
    {% endfor %}
  </nav>
</header>

{# ---------------------------------------------------------------------- #}
{# 1. All-time summary: the four headline tiles and the season heatmap.    #}
{# Both are drawn from the WHOLE backtest corpus whatever season is chosen #}
{# below, so they share one scope and one label (R16 / D33.2-07) -- never  #}
{# the season query parameter.                                             #}
{# ---------------------------------------------------------------------- #}
<section id="summary" class="mb-12 scroll-mt-6">
  {{ bc.section_head("All-time summary", meta="every backtest season") }}
  {% with scope=summary_old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}
  {% set clv = summary.overall_clv|default(0) %}
  <div class="grid grid-cols-2 md:grid-cols-4 gap-2 mb-4">
    {{ bc.stat_tile("Total Games", summary.total_games|default(0)) }}
    {# CLV is a realised measurement against the closing line, so it may carry green/red. #}
    {{ bc.stat_tile("Overall CLV", "%.2f"|format(clv) ~ "%", value_class=("text-green-400" if clv > 0 else ("text-red-400" if clv < 0 else "text-muted"))) }}
    {{ bc.stat_tile("WP Accuracy", "%.1f"|format(summary.wp_accuracy|default(0)) ~ "%") }}
    {{ bc.stat_tile("WP Brier Score", ("%.4f"|format(summary.brier_score)) if summary.brier_score is defined and summary.brier_score is not none else "--") }}
  </div>
  {{ cp.chart_panel("Season Comparison", charts.heatmap, "Season heatmap has not been generated.") }}
</section>

{# ---------------------------------------------------------------------- #}
{# 2. Season by season: selector + export, and the swappable table.       #}
{# ---------------------------------------------------------------------- #}
<section id="seasons" class="mb-12 scroll-mt-6">
  {{ bc.section_head("Season by season", meta="backtest metrics for each season and target") }}
  <div class="flex flex-wrap items-center justify-between gap-4 mb-4">
    {% include "components/_season_selector.html" %}
    {% with csv_url="/api/export/csv?type=backtest",
            json_url="/api/export/json?type=backtest" %}
      {% include "components/_export_buttons.html" %}
    {% endwith %}
  </div>

  {# Season-specific content (HTMX swap target: /fragments/performance). The block reads only
     season_metrics, current_season and its own scope -- the fragment passes nothing else. #}
  <div id="performance-content">
  {% block performance_content %}
  {# Old-rule label for the season-scoped table, from the seasons the table shows. #}
  {% with scope=season_metrics_old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}
  {% if season_metrics %}
  <div class="panel p-0 overflow-hidden">
    <h3 class="panel-title px-4 pt-4 pb-3">Season Metrics</h3>
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line">
            <th scope="col" class="label px-4 py-2 text-left">Season</th>
            <th scope="col" class="label px-4 py-2 text-left">Target</th>
            <th scope="col" class="label px-4 py-2 text-right">Games</th>
            <th scope="col" class="label px-4 py-2 text-right">Accuracy</th>
            <th scope="col" class="label px-4 py-2 text-right">MAE</th>
            <th scope="col" class="label px-4 py-2 text-right">RMSE</th>
            <th scope="col" class="label px-4 py-2 text-right">R&sup2;</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-line">
          {% for m in season_metrics %}
          <tr class="hover:bg-panel-2 transition-colors">
            <td class="px-4 py-2.5 text-fg font-semibold num">{{ m.season }}</td>
            <td class="px-4 py-2.5 text-fg font-semibold uppercase">{{ m.target }}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{{ m.games if m.games is not none else '-' }}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{% if m.accuracy is not none %}{{ "%.1f"|format(m.accuracy) }}%{% else %}-{% endif %}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{% if m.mae is not none %}{{ "%.4f"|format(m.mae) }}{% else %}-{% endif %}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{% if m.rmse is not none %}{{ "%.2f"|format(m.rmse) }}{% else %}-{% endif %}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{% if m.r2 is not none %}{{ "%.3f"|format(m.r2) }}{% else %}-{% endif %}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
  </div>
  {% else %}
    {% with heading="No backtest data",
            body="Run the backtest pipeline to generate historical performance data.",
            action_text=none, action_url=none %}
      {% include "components/_empty_state.html" %}
    {% endwith %}
  {% endif %}
  {% endblock %}
  </div>
</section>

{# ---------------------------------------------------------------------- #}
{# 3. Model vs market, with the cumulative closing-line value chart. All   #}
{# of it is drawn from the same backtest corpus, so one label covers the   #}
{# section; #clv is the CLV panel's own anchor.                            #}
{# ---------------------------------------------------------------------- #}
<section id="vs-market" class="mb-12 scroll-mt-6">
  {{ bc.section_head("Model vs Market", meta="against the closing line") }}
  {% with scope=backtest_old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}
  <p class="text-sm text-muted mb-4 max-w-3xl">How the model stacks up against the closing market line -- by season and in aggregate.</p>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-4">
    {{ cp.chart_panel("Win Probability -- Model vs Market", charts.insights_model_vs_market_wp, "No backtest predictions or market data available for comparison.") }}
    {{ cp.chart_panel("Against the Spread -- Model vs Market MAE", charts.insights_model_vs_market_ats, "No backtest predictions or market data available for comparison.") }}
    {{ cp.chart_panel("Over / Under -- Model vs Market MAE", charts.insights_model_vs_market_ou, "No backtest predictions or market data available for comparison.") }}
  </div>

  <div class="panel p-0 overflow-hidden mb-4">
    <h3 class="panel-title px-4 pt-4 pb-3">Model vs Market -- Aggregate</h3>
    {% if aggregate_table %}
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line">
            <th scope="col" class="label px-4 py-2 text-left">Target</th>
            <th scope="col" class="label px-4 py-2 text-left">Metric</th>
            <th scope="col" class="label px-4 py-2 text-right">Model</th>
            <th scope="col" class="label px-4 py-2 text-right">Market</th>
            <th scope="col" class="label px-4 py-2 text-right">Gap</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-line">
          {% for row in aggregate_table %}
          <tr class="hover:bg-panel-2 transition-colors">
            <td class="px-4 py-2.5 text-fg font-semibold uppercase">{{ row.target }}</td>
            <td class="px-4 py-2.5 text-fg">{{ row.metric }}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{{ row.model_fmt }}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{{ row.market_fmt }}</td>
            {# MONOCHROME (Decision 13): a gap compares two metrics, it is not a realised result, so
               it never takes green or red. The stored gap_favorable tri-state picks the word. #}
            <td class="px-4 py-2.5 text-right font-semibold text-fg num whitespace-nowrap">{{ row.gap_fmt }}{% if row.gap_favorable is sameas true %} <span class="font-sans text-xs font-normal text-muted">favours model</span>{% elif row.gap_favorable is sameas false %} <span class="font-sans text-xs font-normal text-muted">favours market</span>{% endif %}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    <p class="text-xs text-muted px-4 py-3 border-t border-line">Gap = Model - Market. "Favours model" means the model's number is on the better side of the metric (higher accuracy, lower error); "favours market" means the market's is. No word means a neutral gap or missing data. A gap compares two measurements, so it is not coloured as a win or a loss.</p>
    {% else %}
      <div class="px-4 pb-4">
        {% with heading="Chart unavailable", body="Aggregate metrics could not be computed. Backtest predictions or market data are missing.", action_text=none, action_url=none %}
          {% include "components/_empty_state.html" %}
        {% endwith %}
      </div>
    {% endif %}
  </div>

  {# The panel carries the #clv anchor itself (chart_panel's panel_id). #}
  {{ cp.chart_panel("Cumulative CLV", charts.clv, "CLV chart data has not been generated.", panel_id="clv") }}
</section>

{# ---------------------------------------------------------------------- #}
{# 4. Betting simulation (HTMX swap target: /fragments/betting).          #}
{# ---------------------------------------------------------------------- #}
<section id="betting-sim" class="mb-12 scroll-mt-6">
  {{ bc.section_head("Betting Simulation", meta="walk-forward backtest, pretend bankroll") }}
  <p class="text-sm text-muted mb-4 max-w-3xl">Would betting this model have made money -- and is its edge real? A walk-forward simulation over the backtest seasons, from a pretend $10,000 bankroll.</p>

  <div id="betting-content">
  {% block betting_content %}
  {# Imported INSIDE the block: jinja2-fragments renders this block on its own for the scope swap
     and does not run the template's top-level statements. #}
  {% import "components/_broadcast.html" as bc %}
  {% import "components/_chart_panel.html" as cp %}

  {# Scope toggle (All / Recommended). Lives INSIDE the swap target so every swap re-renders it
     with the correct current_scope-driven active highlight + aria-pressed (WR-01). #}
  <div class="mb-6">
    {% include "components/_betting_scope_toggle.html" %}
  </div>

  {# Old-rule label for every part of the simulation below (R16 / D33.2-07). Inside the block so
     the fragment swap re-renders it with the figures. #}
  {% with scope=betting_old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}

  {# KPI scoreboard (D-06). Simulated profit and loss are realised outcomes of the simulation,
     so the ROI and profit tiles may carry green/red. #}
  {% set roi_flat = kpis.roi_flat|default(0) %}
  {% set roi_kelly = kpis.roi_kelly|default(0) %}
  {% set net_profit = kpis.net_profit_flat|default(0) %}
  <div class="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-2">
    {{ bc.stat_tile("Total Bets", kpis.total_bets|default(0)) }}
    {{ bc.stat_tile("Win Rate", "%.1f"|format(kpis.win_rate|default(0)) ~ "%") }}
    {{ bc.stat_tile("ROI (Flat)", "%+.2f"|format(roi_flat) ~ "%", value_class=("text-green-400" if roi_flat > 0 else ("text-red-400" if roi_flat < 0 else "text-muted"))) }}
    {{ bc.stat_tile("ROI (Kelly)", "%+.2f"|format(roi_kelly) ~ "%", value_class=("text-green-400" if roi_kelly > 0 else ("text-red-400" if roi_kelly < 0 else "text-muted"))) }}
    {{ bc.stat_tile("Net Profit", net_profit|format_currency, value_class=("text-green-400" if net_profit > 0 else ("text-red-400" if net_profit < 0 else "text-muted"))) }}
    {{ bc.stat_tile("Final Bankroll", kpis.final_bankroll_flat|default(10000)|format_currency) }}
    {{ bc.stat_tile("Max Drawdown", kpis.max_drawdown_flat|default(0)|format_currency) }}
  </div>
  <p class="text-xs text-muted mt-3 mb-8 max-w-3xl">Kelly figures are identical across scopes -- Kelly already zeroes non-edge bets, so only flat-stake and bet counts change when you switch scope.</p>

  {# Equity Curve (D-07 / D-08) #}
  <h3 class="panel-title mb-1">Equity Curve</h3>
  <p class="text-sm text-muted mb-3 max-w-3xl">Cumulative pretend bankroll from a $10,000 start, flat-stake vs Kelly, ordered by season and week.</p>
  <div class="mb-6">
    {{ cp.chart_panel(none, charts['betting_equity_' ~ current_scope], "Betting simulation data has not been generated. Run the cache rebuild to populate this view.", min_height=320) }}
  </div>

  <h3 class="panel-title mb-1">Equity by Bet Type</h3>
  <p class="text-sm text-muted mb-3 max-w-3xl">Which bet type drove returns -- Winner, Spread, Totals.</p>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-8">
    {% for tgt, label in [("wp", "Winner"), ("ats", "Spread"), ("ou", "Totals")] %}
      {{ cp.chart_panel(label, charts["betting_equity_mini_" ~ tgt ~ "_" ~ current_scope], "Betting simulation data has not been generated. Run the cache rebuild to populate this view.") }}
    {% endfor %}
  </div>

  {# ROI Breakdown (D-10 / D-11) #}
  <h3 class="panel-title mb-1">ROI Breakdown</h3>
  <p class="text-sm text-muted mb-3 max-w-3xl">Return on investment by bet type, season, and edge bucket. Flat-stake isolates pick quality; Kelly mixes in sizing.</p>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-3">
    {% for slice_kind, label in [("type", "ROI by Bet Type"), ("season", "ROI by Season"), ("bucket", "ROI by Edge Bucket")] %}
      {{ cp.chart_panel(label, charts["betting_roi_" ~ slice_kind ~ "_" ~ current_scope], "Betting simulation data has not been generated. Run the cache rebuild to populate this view.") }}
    {% endfor %}
  </div>
  <p class="text-xs text-muted mb-6 max-w-3xl">Bars above 0% are profitable; bars below are losses.</p>

  <div class="panel p-0 overflow-hidden mb-8">
    <h3 class="panel-title px-4 pt-4 pb-3">ROI Summary</h3>
    {% if roi_table %}
    <div class="overflow-x-auto">
      <table class="w-full text-sm">
        <thead>
          <tr class="border-b border-line">
            <th scope="col" class="label px-4 py-2 text-left">Slice</th>
            <th scope="col" class="label px-4 py-2 text-right">Bets</th>
            <th scope="col" class="label px-4 py-2 text-right">Win Rate</th>
            <th scope="col" class="label px-4 py-2 text-right">ROI (Flat)</th>
            <th scope="col" class="label px-4 py-2 text-right">ROI (Kelly)</th>
          </tr>
        </thead>
        <tbody class="divide-y divide-line">
          {% for row in roi_table %}
          <tr class="hover:bg-panel-2 transition-colors">
            <td class="px-4 py-2.5 text-fg font-semibold uppercase">{{ row.label|default(row.slice|default('')) }}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{{ row.bet_count|default(0) }}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{{ row.win_rate_fmt|default('-') }}</td>
            <td class="px-4 py-2.5 text-right font-semibold num {% if row.roi_favorable is sameas true %}text-green-400{% elif row.roi_favorable is sameas false %}text-red-400{% else %}text-muted{% endif %}">{{ row.roi_flat_fmt|default('-') }}</td>
            <td class="px-4 py-2.5 text-right text-fg num">{{ row.roi_kelly_fmt|default('-') }}</td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
    </div>
    <p class="text-xs text-muted px-4 py-3 border-t border-line">52.4% is the standard -110 break-even, exact for spread and totals. Moneyline (Winner) odds vary, so per-bet breakeven differs -- treat this as a reference, not a per-bet truth.</p>
    {% else %}
      <div class="px-4 pb-4">
        {% with heading="Chart unavailable", body="Betting simulation results could not be computed. Run the backtest betting simulation and rebuild the cache.", action_text=none, action_url=none %}
          {% include "components/_empty_state.html" %}
        {% endwith %}
      </div>
    {% endif %}
  </div>

  {# Edge Distribution (D-13 / D-14) #}
  <h3 class="panel-title mb-1">Edge Distribution</h3>
  <p class="text-sm text-muted mb-3 max-w-3xl">How often big-edge bets won. Each bet type is shown in its own units -- probability for Winner, points for Spread and Totals.</p>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
    {% for tgt, label in [("wp", "Winner"), ("ats", "Spread"), ("ou", "Totals")] %}
      {{ cp.chart_panel(label, charts["betting_edge_hist_" ~ tgt ~ "_" ~ current_scope], "Betting simulation data has not been generated. Run the cache rebuild to populate this view.") }}
    {% endfor %}
  </div>
  {% endblock %}
  </div>
</section>

{% endblock %}
```

- [ ] **Step 9: Create the How It Works page**

Create `web/templates/pages/how_it_works.html`:

```jinja
{% extends "base.html" %}
{% import "components/_broadcast.html" as bc %}
{% import "components/_chart_panel.html" as cp %}
{% block title %}How It Works{% endblock %}
{% block content %}

<header class="mb-8">
  <h1 class="display text-4xl md:text-5xl leading-none text-fg">How It Works</h1>
  <p class="text-sm text-muted mt-2 max-w-2xl">Why the model predicts what it does: how well its probabilities match what actually happened, and which inputs each model leans on most.</p>
</header>

{# One block: calibration, feature importance and the accuracy trend are all drawn from the same
   backtest corpus, so one label covers the three sections below (R16 / D33.2-07). #}
{% with scope=old_rule_scope %}{% include "components/_old_rule_label.html" %}{% endwith %}

<section id="calibration" class="mb-12 scroll-mt-6">
  {{ bc.section_head("Calibration", meta="are the predictions honest?") }}
  <p class="text-sm text-muted mb-4 max-w-3xl">Are predicted probabilities and predicted spreads honest? Each chart compares predictions against actual outcomes across the backtest.</p>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
    {{ cp.chart_panel("Win Probability -- Reliability", charts.calibration, "Calibration data has not been generated. Run the cache rebuild to populate.") }}
    {{ cp.chart_panel("Against the Spread -- Predicted vs Actual Margin", charts.insights_calibration_ats, "Calibration data has not been generated. Run the cache rebuild to populate.") }}
    {{ cp.chart_panel("Over / Under -- Predicted vs Actual Total", charts.insights_calibration_ou, "Calibration data has not been generated. Run the cache rebuild to populate.") }}
  </div>
</section>

<section id="features" class="mb-12 scroll-mt-6">
  {{ bc.section_head("What the models rely on", meta="top features of the current models") }}
  <p class="text-sm text-muted mb-4 max-w-3xl">The inputs each production model leans on most.</p>
  <div class="grid grid-cols-1 lg:grid-cols-3 gap-4">
    {{ cp.chart_panel("Top Features -- Win Probability", charts.insights_feature_importance_wp, "No feature importance snapshot was found for this target. Train the model and rebuild the cache.") }}
    {{ cp.chart_panel("Top Features -- Against the Spread", charts.insights_feature_importance_ats, "No feature importance snapshot was found for this target. Train the model and rebuild the cache.") }}
    {{ cp.chart_panel("Top Features -- Over / Under", charts.insights_feature_importance_ou, "No feature importance snapshot was found for this target. Train the model and rebuild the cache.") }}
  </div>
</section>

<section id="accuracy" class="mb-12 scroll-mt-6">
  {{ bc.section_head("Accuracy over time", meta="per backtest season") }}
  {{ cp.chart_panel("Per-Season Accuracy & Error Trend", charts.insights_accuracy_trend, "No per-season metrics found. Run the backtest pipeline.", min_height=320) }}
</section>

{% endblock %}
```

- [ ] **Step 10: Delete the four retired page templates**

Run: `git rm web/templates/pages/performance.html web/templates/pages/backtest.html web/templates/pages/betting.html web/templates/pages/insights.html`
Expected: four `rm` lines.

- [ ] **Step 11: Run the new route tests**

Run: `uv run pytest tests/api/test_merged_pages.py -v`
Expected: all PASS.

- [ ] **Step 12: Migrate the template-level label guard**

`tests/unit/test_page_labels.py` raises at import until its page list matches the folder, so it must be updated now.

Replace the `EXPECTED_PREFIX_BLOCKS` dict:

```python
EXPECTED_PREFIX_BLOCKS: dict[str, int] = {
    # The four pre-rendered charts over the whole backtest corpus, as one block.
    "backtest.html": 1,
    # The selected week's live list (a past week is a replay), and the replay tracker sections.
    "bets.html": 2,
    # KPI strip, equity, ROI and edge charts: one simulation over the backtest window.
    "betting.html": 1,
    # The game header: its season line and the completed-game result overlay (score, badge, CLV).
    "game_detail.html": 1,
    # Calibration, feature importance and model-vs-market sections: one backtest corpus.
    "insights.html": 1,
    # The all-history summary strip, and the season-scoped metrics table.
    "performance.html": 2,
    # The selected season's KPI strip and its cumulative and weekly charts.
    "season.html": 1,
    # The selected week's summary banner and game grid.
    "this_week.html": 1,
}
```

with:

```python
EXPECTED_PREFIX_BLOCKS: dict[str, int] = {
    # The selected week's live list (a past week is a replay), and the replay tracker sections.
    "bets.html": 2,
    # The game header: its season line and the completed-game result overlay (score, badge, CLV).
    "game_detail.html": 1,
    # Calibration, feature importance and the accuracy trend: one backtest corpus.
    "how_it_works.html": 1,
    # The selected season's KPI strip, week strip and its cumulative and weekly charts.
    "season.html": 1,
    # The selected week's summary banner and game grid.
    "this_week.html": 1,
    # The all-history summary (tiles + season heatmap), the season-scoped metrics table, the
    # model-vs-market section (charts, aggregate table, cumulative CLV), and the betting
    # simulation: four blocks, four scopes.
    "track_record.html": 4,
}
```

In `page_context`, replace the four consecutive branches -- from the line `    if page == "backtest.html":` down to and including the performance branch's closing lines

```python
            "summary_old_rule_scope": scope,
            "season_metrics_old_rule_scope": scope,
        }
```

-- with:

```python
    if page == "track_record.html":
        metrics = [
            {"season": season, "target": "wp", "games": 256, "accuracy": 64.0}
            | {"mae": None, "rmse": None, "r2": None}
        ]
        return {
            **common,
            "charts": {"heatmap": "<div>heatmap</div>", "clv": "<div>clv</div>"},
            "available_seasons": [season],
            "current_season": None,
            "season_metrics": metrics,
            "summary": {"total_games": 256, "overall_clv": -1.2, "wp_accuracy": 64.0},
            "aggregate_table": [
                {
                    "target": "wp",
                    "metric": "Brier",
                    "model_fmt": "0.220",
                    "market_fmt": "0.210",
                    "gap_fmt": "+0.010",
                    "gap_favorable": False,
                }
            ],
            "kpis": {"total_bets": 5, "win_rate": 60.0, "roi_flat": 1.2},
            "roi_table": [],
            "current_scope": "recommended",
            "current_path": "/track-record",
            "summary_old_rule_scope": scope,
            "season_metrics_old_rule_scope": scope,
            "backtest_old_rule_scope": scope,
            "betting_old_rule_scope": scope,
        }
    if page == "how_it_works.html":
        return {
            **common,
            "charts": {"calibration": "<div>calibration</div>"},
            "current_path": "/how-it-works",
            "old_rule_scope": scope,
        }
```

Replace the `PAGE_MARKERS` dict:

```python
PAGE_MARKERS: dict[str, str] = {
    "backtest.html": "Backtest Results",
    "bets.html": "Weekly Bet List",
    "betting.html": "Betting Dashboard",
    "game_detail.html": "BUF @ KC",
    "insights.html": "Model Insights",
    "performance.html": "Historical Performance",
    "season.html": "Season Tracking",
    "this_week.html": "This Week's Predictions",
}
```

with:

```python
PAGE_MARKERS: dict[str, str] = {
    "bets.html": "Weekly Bet List",
    "game_detail.html": "BUF @ KC",
    "how_it_works.html": "What the models rely on",
    "season.html": "Season Tracking",
    "this_week.html": "This Week's Predictions",
    "track_record.html": "Season by season",
}
```

Replace:

```python
    def test_a_page_with_an_empty_context_labels_its_block_once(self) -> None:
        """The backtest page with no scope at all: its one block labels once, never zero times."""
        context = _without_scopes(page_context("backtest.html", NEW_RULE_SEASON))
        html = render_page("backtest.html", context)
        assert label_count(html) == 1

    def test_the_rendered_label_is_dated(self) -> None:
        html = render_page("backtest.html", page_context("backtest.html", PAST_SEASON))
        assert "2026-09-15" in html
```

with:

```python
    def test_a_page_with_an_empty_context_labels_its_block_once(self) -> None:
        """How It Works with no scope at all: its one block labels once, never zero times."""
        context = _without_scopes(page_context("how_it_works.html", NEW_RULE_SEASON))
        html = render_page("how_it_works.html", context)
        assert label_count(html) == 1

    def test_the_rendered_label_is_dated(self) -> None:
        html = render_page("how_it_works.html", page_context("how_it_works.html", PAST_SEASON))
        assert "2026-09-15" in html
```

- [ ] **Step 13: Migrate the route-level label guard**

In `tests/api/test_page_labels_routes.py`, replace the `PAGE_ROUTES` dict:

```python
PAGE_ROUTES: dict[str, tuple[str, str]] = {
    "/": ("this_week.html", "This Week's Predictions"),
    "/bets": ("bets.html", "Weekly Bet List"),
    "/betting": ("betting.html", "Betting Dashboard"),
    "/backtest": ("backtest.html", "Backtest Results"),
    "/insights": ("insights.html", "Model Insights"),
    "/performance": ("performance.html", "Historical Performance"),
    "/season": ("season.html", "Season Tracking"),
    f"/games/{_GAME_ID}": ("game_detail.html", "BUF @ KC"),
}
```

with:

```python
PAGE_ROUTES: dict[str, tuple[str, str]] = {
    "/": ("this_week.html", "This Week's Predictions"),
    "/bets": ("bets.html", "Weekly Bet List"),
    "/how-it-works": ("how_it_works.html", "What the models rely on"),
    "/season": ("season.html", "Season Tracking"),
    "/track-record": ("track_record.html", "Season by season"),
    f"/games/{_GAME_ID}": ("game_detail.html", "BUF @ KC"),
}
```

Replace the comment above `FRAGMENT_REQUESTS`:

```python
# HTMX fragment requests and the labelled blocks each swaps in. The performance and betting page
# handlers branch on the HX-Request header; the /fragments/* routes are the dedicated swap paths.
```

with:

```python
# HTMX fragment requests and the labelled blocks each swaps in. The retired /performance and
# /betting URLs answer an HX request with their Track Record block instead of a redirect; the
# /fragments/* routes are the dedicated swap paths.
```

Replace everything from the line `class TestTheSeasonlessHandlersLabel:` to the end of the file with:

```python
class TestTheSeasonlessPagesLabel:
    """The merged pages pass no season, yet label the pre-fix corpus each block draws from."""

    @pytest.mark.parametrize(("url", "expected"), [("/how-it-works", 1), ("/track-record", 4)])
    def test_a_seasonless_page_labels_its_pre_fix_corpus(
        self, pre_2026_client: TestClient, url: str, expected: int
    ) -> None:
        response = pre_2026_client.get(url)
        assert response.status_code == 200
        assert _labels(response.text) == expected

    def test_the_betting_fragment_carries_the_scope_its_section_does(
        self, pre_2026_client: TestClient
    ) -> None:
        """_build_betting_context feeds both the page's betting section and the fragment."""
        page = pre_2026_client.get("/track-record")
        fragment = pre_2026_client.get("/fragments/betting")
        assert page.status_code == 200
        assert fragment.status_code == 200
        betting_section = page.text.split('id="betting-content"', 1)[1]
        assert _labels(betting_section) == _labels(fragment.text) == 1


class TestTheSummaryIgnoresTheSelectedSeason:
    """The all-history summary aggregates every season, so selecting 2026 does not unlabel it."""

    def test_selecting_2026_still_labels_the_all_history_summary(
        self, pre_2026_client: TestClient
    ) -> None:
        response = pre_2026_client.get("/track-record?season=2026")
        assert response.status_code == 200
        body = response.text
        # The season-scoped table has no 2026 rows, so it shows its empty state and no label; the
        # summary, the model-vs-market section and the betting simulation keep theirs.
        assert _labels(body) == 3
        assert "No backtest data" in body
        assert body.index(LABEL_MARKER) < body.index("Total Games"), (
            "the first label is not the one above the all-history summary"
        )

    def test_a_past_season_labels_every_block(self, pre_2026_client: TestClient) -> None:
        response = pre_2026_client.get("/track-record?season=2023")
        assert response.status_code == 200
        assert _labels(response.text) == 4
```

- [ ] **Step 14: Migrate the page, cache-header and footer tests**

In `tests/api/test_pages.py`:

Replace the module docstring's first paragraph:

```python
"""Tests for HTML page routes.

Validates the predictions dashboard (landing page), performance page,
backtest page, and HTMX block rendering.
```

with:

```python
"""Tests for HTML page routes.

Validates the predictions dashboard (landing page), the game detail page, the season page and
HTMX block rendering. The merged Track Record and How It Works pages and the retired-URL
redirects are covered in ``tests/api/test_merged_pages.py``.
```

Delete these test functions entirely (each from its `def` line through its last line; they are replaced by `tests/api/test_merged_pages.py`): `test_performance_page`, `test_performance_season_filter`, `test_backtest_page`, `test_backtest_page_has_chart_containers`, `test_backtest_page_responsive_grid`, `test_insights_page_200`, `test_insights_page_cache_control`, `test_insights_page_nav_links_to_how_it_works` (Task 3 renamed it from `test_insights_page_has_nav_link`), `test_insights_page_renders_expected_chart_ids`, `test_insights_page_empty_db`, `test_betting_page_200`, `test_betting_page_renders_recommended_chart_ids`, `test_betting_empty_db`.

Keep `test_performance_fragment`, `test_performance_fragment_all_seasons`, `test_performance_page_htmx_returns_block` (it now exercises the retired URL's HTMX branch), `test_betting_fragment` and `test_betting_fragment_scope_whitelist` unchanged.

Delete the now-empty Phase 16 banner:

```python
# ---------------------------------------------------------------------------
# Phase 16: /insights page (DASH-10) route tests
# ---------------------------------------------------------------------------
# Activated by Plan 16-03 (route, template, and nav link are now wired).
```

and replace the Phase 17 banner:

```python
# ---------------------------------------------------------------------------
# Phase 17: /betting page + /fragments/betting route scaffolds (skip-gated)
# ---------------------------------------------------------------------------
# Full assertion bodies now; gated by ``@pytest.mark.skip(reason="activated in
# 17-04")`` until Plan 17-04 lands the ``betting_page`` handler, the
# ``/fragments/betting`` route, ``web/templates/pages/betting.html``, and the
# ``base.html`` nav link. Plan 17-04 activates them by deleting the skip marker
# (Phase 16 skip-gated pattern). Default scope = "recommended" (D-17).
```

with:

```python
# ---------------------------------------------------------------------------
# /fragments/betting: Track Record's All/Recommended swap (default "recommended", D-17)
# ---------------------------------------------------------------------------
```

In `tests/api/test_cache_headers.py`, replace:

```python
def test_performance_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/performance")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_backtest_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/backtest")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"
```

with:

```python
def test_track_record_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/track-record")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


def test_how_it_works_page_has_cache_control(test_client: TestClient) -> None:
    response = test_client.get("/how-it-works")
    assert response.status_code == 200
    assert response.headers.get("cache-control") == "public, max-age=60"


@pytest.mark.parametrize("url", ["/performance", "/backtest", "/insights", "/betting"])
def test_a_retired_page_redirect_has_cache_control(test_client: TestClient, url: str) -> None:
    """A 301 is cacheable by default; the header bounds it to a minute so it can be re-pointed."""
    response = test_client.get(url, follow_redirects=False)
    assert response.status_code == 301
    assert response.headers.get("cache-control") == "public, max-age=60"
```

In `tests/api/test_bets_page.py`, inside `test_the_same_cache_meta_does_not_500_any_other_page_either`, replace:

```python
        for route in (
            "/",
            "/performance",
            "/backtest",
            "/insights",
            "/betting",
            "/season",
        ):
```

with:

```python
        for route in (
            "/",
            "/season",
            "/track-record",
            "/how-it-works",
        ):
```

- [ ] **Step 15: Update PIPELINE.md's Stage 7 page list and its pin**

`PIPELINE.md` (Stage 7, "Produces") still lists the seven pre-redesign pages, and `tests/unit/test_pipeline_md.py::TestPipelineMdAnchors::test_stage7_betting_and_season_pages_present` pins `/betting` in that list. Both change together. The em dash on the Stage 7 line is a pre-existing non-ASCII character the pin test's docstring says to keep: edit around it, do not normalise it.

In `PIPELINE.md`, old:

```
- **Produces:** FastAPI at http://localhost:8000 — the seven top-nav pages `/`
  (This Week), `/performance`, `/backtest`, `/insights`, `/betting`,
  `/season` and `/bets`, plus the `/games/{id}` game-detail drill-down and a
  `/health` endpoint.
```

New:

```
- **Produces:** FastAPI at http://localhost:8000 — the five top-nav pages `/`
  (This Week), `/bets`, `/season`, `/track-record` and `/how-it-works`, plus the
  `/games/{id}` game-detail drill-down and a `/health` endpoint. The retired URLs
  `/performance`, `/backtest`, `/insights` and `/betting` answer with a 301 to the merged
  page, query string kept.
```

In `tests/unit/test_pipeline_md.py`, replace the whole method `test_stage7_betting_and_season_pages_present` (inside `class TestPipelineMdAnchors`). Old:

```python
    def test_stage7_betting_and_season_pages_present(self):
        """D-06.2 — Stage 7 page list includes /betting and /season.

        These two pages were added in Phase 23 (WR-03); README and RUNBOOK were
        corrected then but PIPELINE.md was missed.  Phase 23.1 closes the gap.
        """
        content = _read_pipeline_md()
        assert "/betting" in content, "PIPELINE.md Stage 7 missing page: /betting"
        assert "/season" in content, "PIPELINE.md Stage 7 missing page: /season"
```

New:

```python
    def test_stage7_lists_the_five_broadcast_pages(self):
        """D-06.2, updated by the Broadcast redesign (2026-10): Stage 7 lists the five nav pages.

        Phase 23.1 added /betting and /season to this list. The Broadcast redesign then merged
        /performance, /backtest, /insights and /betting into /track-record and /how-it-works, so
        the list names the five pages the nav serves; the retired URLs appear only as redirects.
        """
        content = _read_pipeline_md()
        for page in ("/bets", "/season", "/track-record", "/how-it-works"):
            assert f"`{page}`" in content, f"PIPELINE.md Stage 7 missing page: {page}"
        assert "the five top-nav pages" in content, (
            "PIPELINE.md Stage 7 does not say the nav has five pages"
        )
        assert "the seven top-nav pages" not in content, (
            "PIPELINE.md Stage 7 still lists the seven pre-redesign pages"
        )
```

The em dash shown in both Old blocks is the U+2014 character that is in both files today; keep it exactly as it is. Then run:

```bash
uv run pytest tests/unit/test_pipeline_md.py -v
```

Expected: all PASS, including `test_stage7_lists_the_five_broadcast_pages`. `CLAUDE.md`'s "SEVEN pages" sentence is the historical Phase-31 record and is left unchanged.

- [ ] **Step 16: Confirm nothing still links to a retired page**

Run: `grep -rn 'href="/performance\|href="/backtest\|href="/insights\|href="/betting' web/templates`
Expected: no output. (Task 3 replaced the nav and Task 11 the /bets cross-link. If a line prints, point it at `/track-record` / `/how-it-works` with the matching anchor before continuing.)

- [ ] **Step 17: Recompile the stylesheet**

Run: `./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify`
Expected: exits 0.

- [ ] **Step 18: Lint and type-check**

Run: `uv run ruff check api/routes/pages.py api/routes/fragments.py tests/api/test_merged_pages.py tests/unit/test_page_labels.py tests/api/test_page_labels_routes.py tests/api/test_pages.py tests/api/test_cache_headers.py tests/api/test_bets_page.py tests/unit/test_pipeline_md.py`
Run: `uv run ruff format api/routes/pages.py api/routes/fragments.py tests/api/test_merged_pages.py tests/unit/test_page_labels.py tests/api/test_page_labels_routes.py tests/api/test_pages.py tests/api/test_cache_headers.py tests/api/test_bets_page.py tests/unit/test_pipeline_md.py`
Run: `uv run pyright api/routes/pages.py api/routes/fragments.py tests/api/test_merged_pages.py tests/unit/test_page_labels.py tests/api/test_page_labels_routes.py tests/api/test_pages.py tests/api/test_cache_headers.py tests/api/test_bets_page.py tests/unit/test_pipeline_md.py`
Expected: no ruff error; 0 pyright errors in the two route files and `tests/api/test_merged_pages.py`. In the existing test files, no pyright error on a line this task wrote; an older one predates the branch -- note it in the task report and leave it. If `ruff check` reports `I001` (import order) on `api/routes/fragments.py` or `api/routes/pages.py`, run `uv run ruff check --fix api/routes/fragments.py api/routes/pages.py` and re-run the check.

- [ ] **Step 19: Run every affected test file**

One file per command:

```bash
uv run pytest tests/api/test_merged_pages.py -v
uv run pytest tests/api/test_pages.py -v
uv run pytest tests/api/test_fragments.py -v
uv run pytest tests/api/test_cache_headers.py -v
uv run pytest tests/unit/test_pipeline_md.py -v
```

Expected: all PASS.

```bash
uv run pytest tests/unit/test_page_labels.py -v
uv run pytest tests/api/test_page_labels_routes.py -v
```

Expected: all PASS -- track_record.html renders 4 labels for a past season, 0 for 2026, 4 unwired; how_it_works.html 1 / 0 / 1; the route counts match on both corpora; the `/performance` and `/betting` HX entries in `FRAGMENT_REQUESTS` still return 1-label fragments.

```bash
uv run pytest "tests/api/test_bets_page.py::test_the_same_cache_meta_does_not_500_any_other_page_either" -v
uv run pytest tests/api/test_import_guard.py -v
uv run pytest tests/api/test_import_guard_bets.py -v
uv run pytest tests/api/test_caching.py -v
```

Expected: all PASS (no new `backtest` import under `api/`).

- [ ] **Step 20: Commit**

```bash
git add api/routes/pages.py api/routes/fragments.py web/templates/pages/track_record.html web/templates/pages/how_it_works.html web/static/css/tailwind-compiled.css tests/api/test_merged_pages.py tests/unit/test_page_labels.py tests/api/test_page_labels_routes.py tests/api/test_pages.py tests/api/test_cache_headers.py tests/api/test_bets_page.py PIPELINE.md tests/unit/test_pipeline_md.py
git commit -m "$(cat <<'EOF'
feat(redesign): merge four analysis pages into Track Record and How It Works

/performance, /backtest, /betting and the market half of /insights become
Track Record (summary + season heatmap, season table, model vs market with
cumulative CLV, betting simulation); calibration, features and the accuracy
trend become How It Works. Duplicate charts are no longer displayed (their
cache ids are untouched). The retired URLs answer 301 with the query string
kept and a section anchor; an HTMX request to /performance or /betting still
gets its block. Four old-rule labels on Track Record, one per scoped block.
The model-vs-market gap is monochrome with a "favours model / market" word.
PIPELINE.md's Stage 7 page list and its pin test name the five pages.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

(`git rm` in Step 10 already staged the four deletions; they land in this commit.)

---

### Task 16: The How It Works methodology section (owner-reviewed)

**Files:**
- Modify: `web/templates/pages/how_it_works.html` (insert one section between `</header>` and the old-rule label)
- Modify: `tests/api/test_merged_pages.py` (new test class)
- Regenerate: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes: `how_it_works.html` from Task 15; `bc.section_head`; `READOUT_FORBIDDEN_WORDS` from `backtest.ev_chain_constants` (tests only -- no `api/` import).
- Produces: a `<section id="methodology">` above `#calibration`, carrying the owner-approved text.

- [ ] **Step 1: Write the failing tests**

Append to `tests/api/test_merged_pages.py`:

```python
def _methodology_section(html: str) -> str:
    """The methodology section's own markup, ending at its closing tag -- so the old-rule label
    that follows it can never satisfy an assertion about the methodology text."""
    assert 'id="methodology"' in html, "How It Works has no methodology section"
    return html.split('id="methodology"', 1)[1].split("</section>", 1)[0]


class TestTheMethodologySection:
    def test_sits_above_the_charts(self, test_client: TestClient) -> None:
        html = test_client.get("/how-it-works").text
        assert html.index('id="methodology"') < html.index('id="calibration"')

    def test_states_the_lock_the_evidence_rule_and_the_advice_disclaimer(
        self, test_client: TestClient
    ) -> None:
        section = _methodology_section(test_client.get("/how-it-works").text)
        assert "6 PM Eastern on the day before kickoff" in section
        assert "not evidence" in section
        assert "not betting advice" in section
        assert "walk-forward" in section

    def test_makes_no_over_claim(self, test_client: TestClient) -> None:
        from backtest.ev_chain_constants import READOUT_FORBIDDEN_WORDS

        section = _methodology_section(test_client.get("/how-it-works").text).lower()
        assert [word for word in READOUT_FORBIDDEN_WORDS if word in section] == []

    def test_is_ascii(self, test_client: TestClient) -> None:
        section = _methodology_section(test_client.get("/how-it-works").text)
        assert section.isascii()
```

- [ ] **Step 2: Run them to verify they fail**

Run: `uv run pytest tests/api/test_merged_pages.py::TestTheMethodologySection -v`
Expected: FAIL with `AssertionError: How It Works has no methodology section`.

- [ ] **Step 3: Insert the methodology section**

In `web/templates/pages/how_it_works.html`, insert this block immediately BEFORE the comment line that begins `{# One block: calibration, feature importance and the accuracy trend` (i.e. directly after `</header>` and its blank line):

```jinja
{# How the model works -- plain-English methodology, approved by the owner. It reports no result,
   so it carries no old-rule label; the label below covers the charts. Sources: METHODOLOGY.md,
   PIPELINE.md, PROFITABILITY-READOUT.md and the Phase 33.2 old-rule addendum. Words such as
   "proven" or "validated" are banned here exactly as they are in the readouts. #}
<section id="methodology" class="mb-12 scroll-mt-6">
  {{ bc.section_head("How the model works") }}
  <div class="panel max-w-3xl space-y-4 text-sm leading-relaxed text-fg">
    <p><strong class="font-display font-bold uppercase tracking-wide text-accent">What it predicts.</strong> For every game the site publishes three numbers: each team's chance of winning, the expected winning margin (set against the point spread), and the expected total points (set against the over/under). Each one is shown next to the betting market's own number for the same game, so you can see where the model and the market disagree.</p>
    <p><strong class="font-display font-bold uppercase tracking-wide text-accent">What it knows, and when.</strong> Each game's information locks at 6 PM Eastern on the day before kickoff. Anything known by then can be used -- team strength ratings that update after every game (an Elo system running since 2002), recent form, rest and schedule, the venue, the weather forecast, and the betting lines captured before the lock. Anything that arrives later cannot. The models are trained and tested walk-forward: a model is only ever tested on games played after everything it learned from.</p>
    <p><strong class="font-display font-bold uppercase tracking-wide text-accent">Three models, then the market.</strong> Win probability comes from a logistic regression whose output is calibrated, which aims to make a 70% call come true about 70% of the time. The margin and the total each come from a gradient-boosted tree model (XGBoost). Each model's number is then blended with the market's -- win probability in log-odds, the margin and total on the line itself -- using weights tuned on older seasons that are kept apart from every test. The blended number is the one this site publishes.</p>
    <p><strong class="font-display font-bold uppercase tracking-wide text-accent">How a new model gets in.</strong> A retrained model replaces the running one only if it passes a gate set before it is measured: on games it never trained on, it must do at least as well as the model it would replace at beating the closing betting line. The gate protects against getting worse. It does not show that a model beats the market, and none of the models has shown that it does.</p>
    <p><strong class="font-display font-bold uppercase tracking-wide text-accent">What the numbers on this site mean.</strong> In September 2026 the inputs behind every earlier result were found to be defective -- for example, weather recorded after a game stood in for the forecast, and closing lines known only at kickoff were used as inputs. Every past-season figure on this site was produced under that old rule. Those figures are labelled, kept for the record, and are not evidence. Only the 2026 season, recorded live with each game locked at 6 PM Eastern on the day before kickoff, counts as evidence.</p>
    <p><strong class="font-display font-bold uppercase tracking-wide text-accent">Not betting advice.</strong> This is a personal research tool and not betting advice. A bet on the Bets page cleared an expected-value threshold that was fixed in advance; that is not a forecast that it will win.</p>
  </div>
</section>

```

- [ ] **Step 4: Recompile the stylesheet and run the tests**

Run: `./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify`
Expected: exits 0.

Run: `uv run pytest tests/api/test_merged_pages.py -v`
Expected: all PASS.

Two commands, one per file, each with its own `-k` (pytest keeps only the LAST `-k` of one invocation, which would silently deselect every `test_page_labels.py` test, whose ids say `how_it_works.html`):

```bash
uv run pytest tests/unit/test_page_labels.py -k how_it_works -v
uv run pytest tests/api/test_page_labels_routes.py -k how-it-works -v
```

Expected: all PASS (still exactly 1 label on How It Works; the methodology section adds none), and each command selects at least one test.

Run: `uv run ruff check tests/api/test_merged_pages.py`, `uv run ruff format tests/api/test_merged_pages.py` and `uv run pyright tests/api/test_merged_pages.py`
Expected: no errors.

- [ ] **Step 5: BLOCKING owner review of the wording -- do not commit before this**

Stop and show the owner the six paragraphs exactly as written in Step 3 (paste them into the conversation as plain text, one paragraph per bullet, with their bold lead-ins), and ask, in plain words, whether each one is accurate and reads the way they want. Point out the two places most likely to need their judgement: the list of inputs in "What it knows, and when" (it deliberately names no input group that a gate ruled out of training), and the sentence "none of the models has shown that it does" in "How a new model gets in".

- If the owner approves as written, continue to Step 6.
- If the owner edits any wording, apply their text verbatim in `how_it_works.html`, keep it ASCII (use `--` for dashes), then re-run `uv run pytest tests/api/test_merged_pages.py::TestTheMethodologySection -v`. If one of their edits trips a test (a banned word, or a removed required phrase), show them which and ask how to reword it -- never change their meaning to make a test pass.

Wait for an explicit "approved" (or equivalent) before Step 6.

- [ ] **Step 6: Commit**

```bash
git add web/templates/pages/how_it_works.html web/static/css/tailwind-compiled.css tests/api/test_merged_pages.py
git commit -m "$(cat <<'EOF'
feat(redesign): plain-English methodology on How It Works (owner-approved)

Six short paragraphs above the charts: what the site predicts, the day-before
6 PM ET lock and walk-forward testing, the three models and the market blend,
what the deploy gate does and does not show, why past-season figures are not
evidence, and that this is not betting advice. Guarded against the readouts'
banned over-claim words.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

# Part E -- Charts, polish, verification and merge (Tasks 17-20)

## Contract notes (Part E)

These extend the Shared Interface Contract (section 8, Chart theme) without changing any name it already defines. Every existing call keeps working.

1. **`apply_dark_theme` gains one keyword-only argument:** `apply_dark_theme(fig: go.Figure, *, legend_position: LegendPosition = "top") -> None`, with `LegendPosition = Literal["top", "bottom"]` exported from `api/charts/theme.py`. `api/charts/core.py`'s `_apply_layout_defaults(fig, *, legend_position="top")` passes it through. Why: spec 8 puts legends on top, but four charts already carry subplot titles or a six-entry legend along their top edge (WP calibration, accuracy trend, WP model-vs-market, season weekly). A top legend there would collide with the titles. Those four pass `legend_position="bottom"`. A bare `apply_dark_theme(fig)` still means "top".
2. **Extra constants in `api/charts/theme.py`** beyond contract section 8:
   - `TRANSPARENT = "rgba(0,0,0,0)"`
   - `BOUNDARY_LINE = "rgba(255,255,255,0.14)"` for season-boundary verticals
   - `LEGEND_TEXT = "#C4CAD8"`
   - `STRATEGY_COLORS = {"flat_stake": "#E6E9F0", "kelly": "#FF8A3D"}`. Flat and Kelly are staking strategies, not outcomes, so they may not be green/red. They were indigo/pink before.
   - `HEATMAP_SCALE = [[0.0, "#1D2436"], [1.0, "#7C869C"]]`
   - `MARGINS = {"top": {...}, "bottom": {...}}`
3. **The season heatmap leaves `RdYlGn`.** Green/red are reserved for realised win/loss (spec 4 and 8). A metric heatmap is neither, so it becomes the monochrome `HEATMAP_SCALE`, on which brighter means a higher value. Each heatmap trace keeps its pre-existing direction: the scale is reversed exactly where `RdYlGn_r` was used (the ATS and O/U error-metric traces) and nowhere else. That is not "brighter = better" everywhere -- three of WP's four columns are error metrics on the non-reversed trace -- so no comment, test or caption may claim it is.
4. **`web/static/input.css` gains `@source "../../api/charts";`** (Task 17). `_empty_chart_div` now emits dark-token classes (`text-muted`, `border-line`, `min-h-[220px]`). Those strings live in Python, and Task 1's `@import "tailwindcss" source(none);` limits scanning to the folders named by `@source`, so the line is REQUIRED: without it `min-h-[220px]`, which no template uses, would not compile.
5. **`tests/api/test_import_guard_bets.py`: the allow-list count for `api/charts/core.py` goes from 2 to 1.** The module-level `from backtest.report import SEASON_COLORS, TARGET_COLORS` is removed. The lazy `from backtest.metrics import _compute_ece` stays. The guard's exactness test forces this edit in the same commit, which is the intended behaviour of that guard.
6. **What Task 19's audit expects from earlier tasks.** Tasks 7, 8, 9, 12 and 14 should build these up front so Task 19 finds nothing:
   - Text inside any skewed component sits in a `.unskew` descendant.
   - Every proportion bar drawn with an inline `width:N%` sits inside an element with `role="img"` and an `aria-label` that carries the numbers.
   - The root element of `components/_result_strip.html` carries `role="img"` and an `aria-label` with the counts; `components/_week_strip.html` is a labelled `<ol>` whose items carry their own record text.
   - Every `<table>` sits inside an `overflow-x-auto` wrapper.
   - The slate grid uses `grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4`.
   - The headliner grid uses `grid-cols-1 lg:grid-cols-3`.
   - Nav links carry `aria-current="page"` on the active item, and game detail marks "This Week" (`/`) active.
   - The hamburger button carries `aria-controls="mobile-menu"`, `aria-expanded` and an `aria-label`.
   - Every control keeps a 44px touch target, including the condensed honesty note's `<summary>`: Task 1 gives `.honesty-note > summary` `min-height: 44px` alongside `.skew-control`'s.

---

### Task 17: Dark chart theme foundation (`api/charts/theme.py`)

**Files:**
- Create: `api/charts/theme.py`
- Create: `tests/unit/test_chart_theme.py`
- Modify: `api/charts/core.py:1-86` (module docstring, the `typing.Any` import, the `backtest.report` import, `CHART_LAYOUT_DEFAULTS` (deleted), `DEFAULT_COLOR`, `_apply_layout_defaults`, `_empty_chart_div`)
- Modify: `api/charts/__init__.py` (drop the `CHART_LAYOUT_DEFAULTS` re-export; nothing in `api/`, `tests/`, `scripts/` or `pipeline/` reads it)
- Modify: `tests/api/test_import_guard_bets.py:11-15, 40-53, 117-123, 146-153, 168-171`
- Modify: `web/static/input.css` (one `@source` line)
- Regenerate: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes: the Tailwind tokens `text-muted`, `border-line` from Task 1 (`web/static/input.css` `@theme`).
- Produces (later tasks and Task 18 rely on these exact names):
  - `api.charts.theme`: `INK`, `PANEL`, `FG`, `MUTED`, `ACCENT`, `TRANSPARENT`, `GRID`, `REFERENCE_LINE`, `BOUNDARY_LINE`, `LEGEND_TEXT`, `WIN_COLOR`, `LOSS_COLOR`, `NEUTRAL_COLOR`, `FONT_DISPLAY`, `FONT_BODY`, `FONT_MONO`, `TARGET_COLORS: dict[str, str]`, `SEASON_COLORS: dict[int, str]`, `STRATEGY_COLORS: dict[str, str]`, `HEATMAP_SCALE: list[list[float | str]]`, `MARGINS: dict[str, dict[str, int]]`, `LegendPosition`, `apply_dark_theme(fig, *, legend_position="top") -> None`.
  - `api.charts.core._apply_layout_defaults(fig: go.Figure, *, legend_position: LegendPosition = "top") -> None`.
  - `api.charts.core.DEFAULT_COLOR == theme.MUTED`.
  - `api.charts.core.TARGET_COLORS is theme.TARGET_COLORS` and `api.charts.core.SEASON_COLORS is theme.SEASON_COLORS`.

- [ ] **Step 1: Check the clock**

Run (PowerShell):
```powershell
$et = [System.TimeZoneInfo]::ConvertTimeBySystemTimeZoneId([DateTime]::UtcNow, 'Eastern Standard Time'); $et.ToString('yyyy-MM-dd HH:mm'); if ($et.TimeOfDay -ge [TimeSpan]'16:30' -and $et.TimeOfDay -lt [TimeSpan]'17:45') { 'STOP: inside the 16:30-17:45 ET window' } else { 'OK to proceed' }
```
Expected: `OK to proceed`. If it prints `STOP`, wait until 17:45 ET.

- [ ] **Step 2: Write the failing theme tests**

Create `tests/unit/test_chart_theme.py`:

```python
"""The dark Broadcast chart theme every dashboard chart shares (redesign Task 17).

Layouts are read back through ``fig.to_dict()`` so the assertions see exactly what Plotly
serialises into the prerendered HTML.
"""

from __future__ import annotations

from typing import Any, cast

import plotly.graph_objects as go
import plotly.io as pio
import pytest
from plotly.subplots import make_subplots

from api.charts import theme


def _figure() -> go.Figure:
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[2, 1, 3], name="Model"))
    fig.update_layout(title={"text": "Example", "x": 0.5}, legend={"x": 0.02, "y": 0.98})
    return fig


def _layout(fig: go.Figure) -> dict[str, Any]:
    return fig.to_dict()["layout"]


def test_backgrounds_are_transparent_so_the_panel_shows_through() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    layout = _layout(fig)
    assert layout["paper_bgcolor"] == theme.TRANSPARENT
    assert layout["plot_bgcolor"] == theme.TRANSPARENT


def test_the_default_light_template_is_dropped() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    assert _layout(fig)["template"] == pio.templates["none"].to_plotly_json()


def test_axis_text_uses_the_display_face_and_hover_uses_the_body_face() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    layout = _layout(fig)
    assert layout["font"]["family"] == theme.FONT_DISPLAY
    assert layout["font"]["color"] == theme.MUTED
    assert layout["title"]["font"]["family"] == theme.FONT_DISPLAY
    assert layout["title"]["font"]["color"] == theme.FG
    assert layout["hoverlabel"]["font"]["family"] == theme.FONT_BODY


def test_gridlines_are_faint_and_zero_lines_are_off() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    layout = _layout(fig)
    for axis in ("xaxis", "yaxis"):
        assert layout[axis]["gridcolor"] == theme.GRID
        assert layout[axis]["showgrid"] is True
        assert layout[axis]["zeroline"] is False


def test_every_subplot_axis_and_subplot_title_is_themed() -> None:
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Accuracy", "MAE"))
    fig.add_trace(go.Scatter(x=[1], y=[1]), row=1, col=1)
    fig.add_trace(go.Scatter(x=[1], y=[1]), row=1, col=2)
    theme.apply_dark_theme(fig, legend_position="bottom")
    layout = _layout(fig)
    assert layout["xaxis2"]["gridcolor"] == theme.GRID
    assert layout["yaxis2"]["zeroline"] is False
    assert layout["annotations"], "make_subplots should have produced subplot titles"
    assert all(a["font"]["family"] == theme.FONT_DISPLAY for a in layout["annotations"])


def test_hover_label_is_dark_with_the_accent_border() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    hover = _layout(fig)["hoverlabel"]
    assert hover["bgcolor"] == theme.INK
    assert hover["bordercolor"] == theme.ACCENT


def test_legend_sits_on_top_by_default_and_overrides_a_leftover_position() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig)
    legend = _layout(fig)["legend"]
    assert legend["orientation"] == "h"
    assert legend["yanchor"] == "bottom"
    assert legend["y"] >= 1.0
    assert legend["x"] == 0
    assert legend["bgcolor"] == theme.TRANSPARENT


def test_legend_can_sit_below_the_plot() -> None:
    fig = _figure()
    theme.apply_dark_theme(fig, legend_position="bottom")
    layout = _layout(fig)
    assert layout["legend"]["yanchor"] == "top"
    assert layout["legend"]["y"] < 0
    assert layout["margin"]["b"] >= theme.MARGINS["top"]["b"]


def test_an_unknown_legend_position_is_refused() -> None:
    with pytest.raises(ValueError, match="legend_position"):
        theme.apply_dark_theme(_figure(), legend_position=cast(Any, "left"))


def test_bet_type_colours_are_the_spec_keys() -> None:
    assert theme.TARGET_COLORS == {"wp": "#FFD400", "ats": "#4CC9F0", "ou": "#C77DFF"}


def test_season_colours_cover_2018_to_2026_distinctly_and_avoid_reserved_colours() -> None:
    assert set(range(2018, 2027)) <= set(theme.SEASON_COLORS)
    colours = {colour.upper() for colour in theme.SEASON_COLORS.values()}
    assert len(colours) == len(theme.SEASON_COLORS)
    # Green/red mean a realised result, and the bet-type colours and the accent mean
    # Winner / Spread / Totals and emphasis on every chart (spec 7.4), so no season takes one.
    reserved = {
        theme.WIN_COLOR,
        theme.LOSS_COLOR,
        theme.ACCENT,
        *theme.TARGET_COLORS.values(),
    }
    assert not colours & {colour.upper() for colour in reserved}


def test_strategy_colours_are_not_outcome_colours() -> None:
    assert not {theme.WIN_COLOR, theme.LOSS_COLOR} & set(theme.STRATEGY_COLORS.values())


def test_core_takes_its_palette_from_the_theme() -> None:
    from api.charts import core

    assert core.TARGET_COLORS is theme.TARGET_COLORS
    assert core.SEASON_COLORS is theme.SEASON_COLORS
    assert core.DEFAULT_COLOR == theme.MUTED
    assert core._get_target_color("ats") == "#4CC9F0"


def test_layout_defaults_delegate_to_the_theme() -> None:
    from api.charts.core import _apply_layout_defaults

    fig = _figure()
    _apply_layout_defaults(fig)
    layout = _layout(fig)
    assert layout["paper_bgcolor"] == theme.TRANSPARENT
    assert layout["hoverlabel"]["bordercolor"] == theme.ACCENT

    bottom = _figure()
    _apply_layout_defaults(bottom, legend_position="bottom")
    assert _layout(bottom)["legend"]["y"] < 0


def test_empty_chart_div_uses_dark_tokens_and_keeps_its_message() -> None:
    from api.charts.core import _empty_chart_div

    html = _empty_chart_div("Chart unavailable")
    assert "Chart unavailable" in html
    assert "text-muted" in html
    assert "text-gray-500" not in html
```

- [ ] **Step 3: Run the tests and watch them fail**

Run: `uv run pytest tests/unit/test_chart_theme.py -v`
Expected: FAIL at collection with `ImportError: cannot import name 'theme' from 'api.charts'`.

- [ ] **Step 4: Create the theme module**

Create `api/charts/theme.py`:

```python
"""Dark "Broadcast" chart theme shared by every prerendered dashboard chart.

One module owns the chart look, so the generators in ``api.charts`` cannot drift apart:
- the figure is transparent, so the page's panel shows through;
- axis, legend and subplot-title text uses the condensed display face the HTML uses;
- the hover label matches the site's dark tooltip, with the yellow accent border.

Colour meaning follows the site-wide rule (spec section 4). ``WIN_COLOR`` and
``LOSS_COLOR`` are for realised win/loss and profit/loss series only. They are never used
for a model, market, season or strategy series. Bet types always take ``TARGET_COLORS``.
A market series is ``MUTED`` and dashed, beside its solid bet-type colour.

This module imports nothing from ``backtest/`` (tests/api/test_import_guard_bets.py). The
web palette is therefore independent of the white-background backtest HTML reports, which
keep ``backtest.report``'s own palette.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import plotly.graph_objects as go

# Base colours: the same values as the Tailwind tokens in web/static/input.css.
INK: Final = "#0B0F17"
PANEL: Final = "#151B29"
FG: Final = "#F3F5F9"
MUTED: Final = "#8A93A8"
ACCENT: Final = "#FFD400"
TRANSPARENT: Final = "rgba(0,0,0,0)"
GRID: Final = "rgba(255,255,255,0.06)"
REFERENCE_LINE: Final = "rgba(255,255,255,0.45)"
# Season-boundary verticals: visible, but quieter than a reference line.
BOUNDARY_LINE: Final = "rgba(255,255,255,0.14)"
LEGEND_TEXT: Final = "#C4CAD8"

# Realised outcomes only: Tailwind green-500 / red-500, the same hues the HTML uses.
WIN_COLOR: Final = "#22C55E"
LOSS_COLOR: Final = "#EF4444"
NEUTRAL_COLOR: Final = "#5B6478"

FONT_DISPLAY: Final = "Barlow Condensed, Inter, sans-serif"
FONT_BODY: Final = "Inter, system-ui, sans-serif"
FONT_MONO: Final = "JetBrains Mono, monospace"

TARGET_COLORS: dict[str, str] = {"wp": "#FFD400", "ats": "#4CC9F0", "ou": "#C77DFF"}

# Distinct on the dark panel and free of every reserved colour: green/red (realised
# outcomes), the three bet-type colours and the accent (Winner / Spread / Totals and emphasis
# on every chart, spec 7.4). 2026, the live season, is the brightest line (near-white; not
# #FFFFFF, which the chart-layer literal checks treat as a leftover light-theme colour).
SEASON_COLORS: dict[int, str] = {
    2018: "#94A3B8",
    2019: "#A78BFA",
    2020: "#F472B6",
    2021: "#60A5FA",
    2022: "#818CF8",
    2023: "#FDBA74",
    2024: "#E879F9",
    2025: "#D6D3D1",
    2026: "#F8FAFC",
}

# Flat and Kelly are staking strategies, not outcomes, so they never take green/red.
STRATEGY_COLORS: dict[str, str] = {"flat_stake": "#E6E9F0", "kelly": "#FF8A3D"}

# Monochrome metric heatmap: brighter = a higher value. Each heatmap trace keeps its
# pre-existing direction (reversed exactly where the old red-yellow-green scale was), so
# brighter is not "better" on every column.
HEATMAP_SCALE: list[list[float | str]] = [[0.0, "#1D2436"], [1.0, "#7C869C"]]

LegendPosition = Literal["top", "bottom"]

# "top": just above the plot area, under the title.
# "bottom": under the x-axis title, for charts whose top edge already carries subplot
# titles or a long legend.
_LEGENDS: dict[str, dict[str, Any]] = {
    "top": {"orientation": "h", "x": 0, "xanchor": "left", "y": 1.02, "yanchor": "bottom"},
    "bottom": {
        "orientation": "h",
        "x": 0,
        "xanchor": "left",
        "y": -0.22,
        "yanchor": "top",
    },
}
MARGINS: dict[str, dict[str, int]] = {
    "top": {"l": 52, "r": 16, "t": 76, "b": 48},
    "bottom": {"l": 52, "r": 16, "t": 64, "b": 96},
}


def apply_dark_theme(fig: go.Figure, *, legend_position: LegendPosition = "top") -> None:
    """Apply the dark Broadcast look to *fig*, in place.

    Call it LAST, after the generator's own layout. It drops Plotly's default light
    template, then sets backgrounds, fonts, gridlines, the hover label and the legend
    placement. That overrides any per-chart leftovers, so every chart reads as one system.
    """
    if legend_position not in _LEGENDS:
        msg = f"legend_position must be 'top' or 'bottom', got {legend_position!r}"
        raise ValueError(msg)
    fig.update_layout(
        template="none",
        paper_bgcolor=TRANSPARENT,
        plot_bgcolor=TRANSPARENT,
        font={"family": FONT_DISPLAY, "size": 13, "color": MUTED},
        title_font={"family": FONT_DISPLAY, "size": 17, "color": FG},
        legend={
            **_LEGENDS[legend_position],
            "bgcolor": TRANSPARENT,
            "font": {"family": FONT_DISPLAY, "size": 13, "color": LEGEND_TEXT},
        },
        hoverlabel={
            "bgcolor": INK,
            "bordercolor": ACCENT,
            "font": {"family": FONT_BODY, "size": 13, "color": FG},
        },
        margin=MARGINS[legend_position],
    )
    fig.update_xaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, showline=False, automargin=True
    )
    fig.update_yaxes(
        showgrid=True, gridcolor=GRID, zeroline=False, showline=False, automargin=True
    )
    # Subplot titles and reference-line labels are annotations, so they get the same
    # styling as axis text.
    fig.update_annotations(font={"family": FONT_DISPLAY, "size": 13, "color": MUTED})
```

- [ ] **Step 5: Point `api/charts/core.py` at the theme**

Edit `api/charts/core.py`. In the module docstring, replace

```python
Each function returns an HTML div string via plotly.io.to_html with
include_plotlyjs=False (Plotly CDN loaded in base.html).
"""
```

with

```python
Each function returns an HTML div string via plotly.io.to_html with
include_plotlyjs=False (Plotly CDN loaded in base.html). Every figure gets the dark
dashboard theme from api.charts.theme through _apply_layout_defaults.
"""
```

Replace

```python
from backtest.report import SEASON_COLORS, TARGET_COLORS

# ---------------------------------------------------------------------------
# Layout defaults (per UI-SPEC CHART_LAYOUT_DEFAULTS)
# ---------------------------------------------------------------------------

CHART_LAYOUT_DEFAULTS: dict[str, Any] = {
    "font_family": "system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    "font_size": 13,
    "plot_bgcolor": "#FFFFFF",
    "paper_bgcolor": "#FFFFFF",
    "margin": {"l": 48, "r": 16, "t": 48, "b": 48},
    "gridcolor": "#E5E7EB",
}

PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False}

DEFAULT_COLOR = "#95a5a6"


def _apply_layout_defaults(fig: go.Figure) -> None:
    """Apply the UI-SPEC layout defaults to a Plotly figure."""
    defaults = CHART_LAYOUT_DEFAULTS
    fig.update_layout(
        font_family=defaults["font_family"],
        font_size=defaults["font_size"],
        plot_bgcolor=defaults["plot_bgcolor"],
        paper_bgcolor=defaults["paper_bgcolor"],
        margin=defaults["margin"],
    )
    fig.update_xaxes(gridcolor=defaults["gridcolor"])
    fig.update_yaxes(gridcolor=defaults["gridcolor"])
```

with the block below. `CHART_LAYOUT_DEFAULTS` is deleted rather than kept: no figure reads it once `_apply_layout_defaults` delegates to the theme, and nothing in `api/`, `tests/`, `scripts/` or `pipeline/` imports it (only the `api/charts/__init__.py` re-export, removed in the next edit).

```python
from api.charts.theme import (
    MUTED,
    SEASON_COLORS,
    TARGET_COLORS,
    LegendPosition,
    apply_dark_theme,
)

# ---------------------------------------------------------------------------
# Layout defaults
# ---------------------------------------------------------------------------

PLOTLY_CONFIG = {"responsive": True, "displayModeBar": False}

# A series with no palette entry falls back to the theme's muted grey, never a light grey
# that disappears on the dark panel.
DEFAULT_COLOR = MUTED


def _apply_layout_defaults(
    fig: go.Figure, *, legend_position: LegendPosition = "top"
) -> None:
    """Apply the shared dark chart theme (api.charts.theme) to a Plotly figure.

    Every generator calls this LAST, after its own layout. The theme's backgrounds, fonts,
    gridlines, hover label and legend placement therefore win over any per-chart leftovers.
    Pass legend_position="bottom" when subplot titles or a long legend already occupy the
    top edge.
    """
    apply_dark_theme(fig, legend_position=legend_position)
```

`Any` was used only by the deleted dict, so drop its import. Replace

```python
from typing import Any

import numpy as np
```

with

```python
import numpy as np
```

In `api/charts/__init__.py`, drop the dead re-export. Replace

```python
from api.charts.core import (  # noqa: F401
    CHART_LAYOUT_DEFAULTS,
    DEFAULT_COLOR,
```

with

```python
from api.charts.core import (  # noqa: F401
    DEFAULT_COLOR,
```

Replace

```python
def _empty_chart_div(message: str) -> str:
    """Return an HTML div indicating no chart data is available."""
    return (
        '<div class="flex items-center justify-center h-full text-sm text-gray-500">'
        f"<p>{message}</p></div>"
    )
```

with

```python
def _empty_chart_div(message: str) -> str:
    """Return an HTML div indicating no chart data is available.

    The classes are dark-theme tokens. web/static/input.css scans api/charts, so they are
    compiled even though they appear only in this Python string.
    """
    return (
        '<div class="flex min-h-[220px] items-center justify-center rounded-sm '
        'border border-dashed border-line px-4 text-center text-sm text-muted">'
        f"<p>{message}</p></div>"
    )
```

- [ ] **Step 6: Update the import guard's exact count on purpose**

Edit `tests/api/test_import_guard_bets.py`.

Replace

```python
Inherited debt is NAMED, not silenced. ``api/charts/core.py`` already imports ``backtest`` twice --
a module-level import of two colour constants from the report module, and a lazy in-function import
of an ECE computation helper (the second is a metric COMPUTATION reachable from a chart-render
path). Both predate Phase 31. They are carried in an EXPLICIT allow-list constant with an EXACT
count, so removing one of them or adding a third both fail.
```

with

```python
Inherited debt is NAMED, not silenced. ``api/charts/core.py`` imports ``backtest`` once: a lazy
in-function import of an ECE computation helper, which is a metric COMPUTATION reachable from a
chart-render path. It predates Phase 31. The Broadcast redesign (2026-10) paid off the second
inherited site, a module-level import of two colour constants from the report module, by moving
the dashboard palette to ``api/charts/theme.py``. The remaining site is carried in an EXPLICIT
allow-list constant with an EXACT count, so removing it or adding a second both fail.
```

Replace

```python
# Maps a POSIX-normalized path under api/ to the EXACT number of permitted top-level-``backtest``
# import nodes in that file. Both entries predate Phase 31:
#
#   api/charts/core.py  module-level  from backtest.report import SEASON_COLORS, TARGET_COLORS
#   api/charts/core.py  lazy, in-fn   from backtest.metrics import _compute_ece
#
# Phase 31 adds NOTHING to this mapping -- see the module docstring for why the tracker seam is a
# pure-persistence handoff instead of an allow-list entry.
_BACKTEST_IMPORT_ALLOW_LIST: dict[str, int] = {
    "api/charts/core.py": 2,
}
```

with

```python
# Maps a POSIX-normalized path under api/ to the EXACT number of permitted top-level-``backtest``
# import nodes in that file. The one entry predates Phase 31:
#
#   api/charts/core.py  lazy, in-fn   from backtest.metrics import _compute_ece
#
# The Broadcast redesign removed the module-level colour import from backtest.report. The
# dashboard palette now lives in api/charts/theme.py, so the count dropped from 2 to 1. The
# exactness test below is what forced this edit.
#
# Phase 31 adds NOTHING to this mapping -- see the module docstring for why the tracker seam is a
# pure-persistence handoff instead of an allow-list entry.
_BACKTEST_IMPORT_ALLOW_LIST: dict[str, int] = {
    "api/charts/core.py": 1,
}
```

Replace

```python
    """No ``backtest`` import exists under ``api/`` outside the two allow-listed sites.
```

with

```python
    """No ``backtest`` import exists under ``api/`` outside the allow-listed site.
```

Replace

```python
    Exactness matters in both directions. A third site in ``api/charts/core.py`` is new surface
    and must fail. But REMOVING one of the two inherited sites must also fail, because the debt is
    then paid and the allow-list entry has become a licence nobody is using -- a stale allowance
    is how a guard quietly stops guarding.
```

with

```python
    Exactness matters in both directions. A second site in ``api/charts/core.py`` is new surface
    and must fail. But REMOVING the inherited site must also fail, because the debt is then paid
    and the allow-list entry has become a licence nobody is using -- a stale allowance is how a
    guard quietly stops guarding.
```

Replace

```python
    assert _BACKTEST_IMPORT_ALLOW_LIST["api/charts/core.py"] == 2
```

with

```python
    assert _BACKTEST_IMPORT_ALLOW_LIST["api/charts/core.py"] == 1
```

- [ ] **Step 7: Let Tailwind see the chart classes, then recompile**

This line is REQUIRED, not belt-and-braces: Task 1 imports Tailwind with `source(none)`, so only the folders named by `@source` are scanned, and `_empty_chart_div`'s classes live in a Python string under `api/charts/`.

Edit `web/static/input.css`. Replace

```css
@source "../../web/templates";
```

with

```css
@source "../../web/templates";
@source "../../api/charts";
```

Run (Git Bash, from the worktree):
```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
for cls in '.text-muted' '.border-line' '.border-dashed' '.min-h-\[220px\]'; do printf '%s ' "$cls"; grep -c -F "$cls" web/static/css/tailwind-compiled.css; done
```
Expected: each class prints a count of 1 or more. `.min-h-\[220px\]` is the one that proves the new line works: no template uses it, so before this edit it printed 0.

- [ ] **Step 8: Run the theme, guard and chart tests**

Run:
```bash
uv run pytest tests/unit/test_chart_theme.py -v
uv run pytest tests/api/test_import_guard_bets.py -v
uv run pytest tests/api/test_import_guard.py -v
uv run pytest tests/test_charts.py -q
```
Expected: all PASS. `tests/test_charts.py` passes unchanged because its assertions are on content (titles, labels, ids, empty fallbacks), not colour.

- [ ] **Step 9: Lint and type-check**

Run:
```bash
uv run ruff check api/charts/theme.py api/charts/core.py api/charts/__init__.py tests/unit/test_chart_theme.py tests/api/test_import_guard_bets.py
uv run ruff format api/charts/theme.py api/charts/core.py api/charts/__init__.py tests/unit/test_chart_theme.py tests/api/test_import_guard_bets.py
uv run pyright api/charts/theme.py api/charts/core.py api/charts/__init__.py tests/unit/test_chart_theme.py tests/api/test_import_guard_bets.py
```
Expected: `All checks passed!`, files formatted, `0 errors`.

- [ ] **Step 10: Commit**

```bash
git add api/charts/theme.py api/charts/core.py api/charts/__init__.py tests/unit/test_chart_theme.py tests/api/test_import_guard_bets.py web/static/input.css web/static/css/tailwind-compiled.css
git commit -m "$(cat <<'EOF'
feat(redesign): dark Broadcast chart theme shared by every dashboard chart

api/charts/theme.py owns the chart look: transparent figure, display-face axis and
legend text, dark hover label with the accent border, legend on top (or below for
subplot charts). core.py takes its palette from the theme instead of backtest.report,
so the import guard's exact allow-list count for core.py drops from 2 to 1.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 18: Per-chart colour and spacing fixes

**Files:**
- Create: `tests/unit/test_chart_colour_rules.py`
- Modify: `api/charts/core.py` (calibration, CLV, heatmap, equity bodies; theme import list)
- Modify: `api/charts/insights.py` (imports, docstring, both calibration charts, accuracy trend, the three model-vs-market charts)
- Modify: `api/charts/betting.py` (docstrings; the local palette aliases deleted in favour of the theme names at their six call sites; boundary/reference lines; legends)
- Modify: `api/charts/season.py` (docstrings, reference lines, legends)
- Test (unchanged, must still pass): `tests/test_charts.py`

**Interfaces:**
- Consumes: everything Task 17 produces in `api.charts.theme`; `_apply_layout_defaults(fig, *, legend_position=...)`.
- Produces: no new names. Every generator in `api.charts` keeps its name, signature, chart id, title text, trace names and empty-state message.

- [ ] **Step 1: Write the failing colour-rule tests**

Create `tests/unit/test_chart_colour_rules.py`:

```python
"""Colour and layout rules every dashboard chart must follow under the dark theme (Task 18).

Each generator is fed small, non-empty inputs, so every one draws a real figure. The data
and layout are then read back out of the Plotly.newPlot(...) call in the HTML -- exactly
what the browser receives from the cache.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from api.charts import theme

# ---------------------------------------------------------------------------
# Small deterministic inputs (shapes mirror the cache tables)
# ---------------------------------------------------------------------------

_GAME_IDS = [f"2023_W{i:02d}_A@B" for i in range(1, 21)]

_WP_ROWS: list[dict[str, Any]] = [
    {
        "game_id": gid,
        "season": 2023,
        "week": i + 1,
        "target": "wp",
        "model_prob": round(0.05 + i * 0.045, 3),
        "actual": 1.0 if i % 3 else 0.0,
        "probability_clv": 0.01 * ((i % 5) - 2),
        "has_closing_odds": True,
    }
    for i, gid in enumerate(_GAME_IDS)
]

_ATS_ROWS: list[dict[str, Any]] = [
    {
        "game_id": gid,
        "season": 2023,
        "week": i + 1,
        "target": "ats",
        "model_prob": -10.0 + i,
        "actual": -10.0 + i + (1.5 if i % 2 else -1.5),
        "probability_clv": 0.0,
        "has_closing_odds": True,
    }
    for i, gid in enumerate(_GAME_IDS)
] + [
    {"game_id": "2023_W21_C@D", "season": 2023, "week": 21, "target": "ats",
     "model_prob": -25.0, "actual": 5.0, "probability_clv": 0.0, "has_closing_odds": True},
    {"game_id": "2023_W22_E@F", "season": 2023, "week": 22, "target": "ats",
     "model_prob": 25.0, "actual": -5.0, "probability_clv": 0.0, "has_closing_odds": True},
]

_OU_ROWS: list[dict[str, Any]] = [
    {
        "game_id": gid,
        "season": 2023,
        "week": i + 1,
        "target": "ou",
        "model_prob": 38.0 + i,
        "actual": 38.0 + i + (2.5 if i % 2 else -2.5),
        "probability_clv": 0.0,
        "has_closing_odds": True,
    }
    for i, gid in enumerate(_GAME_IDS)
]

_MARKET: list[dict[str, Any]] = [
    {
        "game_id": gid,
        "season": 2023,
        "week": i + 1,
        "market_spread": -3.0 + (i % 4),
        "market_total": 44.5 + (i % 3),
        "market_ml_home": -150 if i % 2 else 130,
        "market_ml_away": 130 if i % 2 else -150,
    }
    for i, gid in enumerate(_GAME_IDS)
]

_METRICS: list[dict[str, Any]] = [
    row
    for season in (2021, 2022, 2023)
    for row in (
        {"season": season, "target": "wp", "metric_name": "accuracy",
         "metric_value": 0.62 + 0.01 * (season - 2021)},
        {"season": season, "target": "wp", "metric_name": "brier_score",
         "metric_value": 0.22 - 0.002 * (season - 2021)},
        {"season": season, "target": "ats", "metric_name": "mae",
         "metric_value": 10.4 - 0.1 * (season - 2021)},
        {"season": season, "target": "ou", "metric_name": "mae",
         "metric_value": 10.8 - 0.2 * (season - 2021)},
    )
]

_IMPORTANCES: list[dict[str, Any]] = [
    {"game_id": "_model_", "target": target, "feature_name": name, "importance": weight}
    for target in ("wp", "ats", "ou")
    for name, weight in (("elo_diff", 0.4), ("hfa_used", 0.3), ("is_divisional", 0.2))
]

_EQUITY: list[dict[str, Any]] = [
    {"strategy": strategy, "bet_index": i, "bankroll": 10000.0 + step * i}
    for strategy, step in (("flat_stake", 40.0), ("kelly", -25.0))
    for i in range(5)
]


def _bet(game_id: str, target: str, edge: float, outcome: bool | None, payout: float) -> dict[str, Any]:
    return {
        "game_id": game_id, "season": int(game_id[:4]), "week": int(game_id[6:8]),
        "target": target, "bet_side": "home", "model_value": 0.5, "market_value": 0.5,
        "edge": edge, "slipped_line": None if target == "wp" else -3.0, "odds": -110.0,
        "flat_stake": 100.0, "kelly_stake": 80.0, "outcome": outcome,
        "payout_flat": payout, "payout_kelly": payout * 0.8,
    }


_BETS: list[dict[str, Any]] = [
    _bet("2022_W01_A@B", "wp", 0.07, True, 66.7),
    _bet("2022_W02_C@D", "wp", 0.06, False, -100.0),
    _bet("2023_W03_E@F", "ats", 2.5, True, 90.9),
    _bet("2023_W04_G@H", "ats", 1.5, False, -100.0),
    _bet("2023_W05_I@J", "ats", 1.2, None, 0.0),
    _bet("2024_W06_K@L", "ou", 2.5, True, 90.9),
    _bet("2024_W07_M@N", "ou", 3.0, False, -100.0),
]

_SEASON_ROWS: list[dict[str, Any]] = [
    {
        "game_id": f"2024_W{wk:02d}_A@B", "season": 2024, "week": wk, "status": "completed",
        "home_score": 27 if wk % 2 else 17, "away_score": 20 if wk % 2 else 24,
        "wp_prob": 0.62 if wk % 2 else 0.41, "ats_prediction": -4.0 if wk % 2 else 2.0,
        "ou_prediction": 50.0 if wk % 2 else 40.0, "market_spread": -3.0, "market_total": 45.0,
    }
    for wk in range(1, 6)
]

# Charts whose series ARE realised outcomes; only these may use WIN_COLOR / LOSS_COLOR.
_OUTCOME_CHARTS = frozenset({"betting_edge_hist_ats"})

# Colours the light theme and the old palettes used. None may survive on the dark panel.
_LEGACY_LITERALS = (
    '#999"', '#ccc"', '#333"', '#666"', "#667eea", "#f093fb", "#16a34a", "#dc2626",
    "#95a5a6", "#4fd1c5", "#E5ECF6", "#E5E7EB", "#FFFFFF",
)


def _render_all() -> dict[str, str]:
    from api.charts import (
        generate_betting_edge_hist_ats,
        generate_betting_equity_chart,
        generate_betting_equity_mini_ats,
        generate_betting_roi_type,
        generate_dashboard_calibration_chart,
        generate_dashboard_clv_chart,
        generate_dashboard_equity_chart,
        generate_dashboard_heatmap,
        generate_insights_accuracy_trend,
        generate_insights_ats_calibration,
        generate_insights_feature_importance_wp,
        generate_insights_model_vs_market_ats,
        generate_insights_model_vs_market_ou,
        generate_insights_model_vs_market_wp,
        generate_insights_ou_calibration,
        generate_season_cumulative,
        generate_season_weekly,
    )

    return {
        "calibration": generate_dashboard_calibration_chart(_WP_ROWS),
        "clv": generate_dashboard_clv_chart(_WP_ROWS),
        "heatmap": generate_dashboard_heatmap(_METRICS),
        "equity": generate_dashboard_equity_chart(_EQUITY),
        "ats_calibration": generate_insights_ats_calibration(_ATS_ROWS),
        "ou_calibration": generate_insights_ou_calibration(_OU_ROWS),
        "feature_importance_wp": generate_insights_feature_importance_wp(_IMPORTANCES),
        "accuracy_trend": generate_insights_accuracy_trend(_METRICS),
        "mvm_wp": generate_insights_model_vs_market_wp(_WP_ROWS, _MARKET),
        "mvm_ats": generate_insights_model_vs_market_ats(_ATS_ROWS, _MARKET),
        "mvm_ou": generate_insights_model_vs_market_ou(_OU_ROWS, _MARKET),
        "betting_equity": generate_betting_equity_chart(_BETS),
        "betting_equity_mini_ats": generate_betting_equity_mini_ats(_BETS),
        "betting_roi_type": generate_betting_roi_type(_BETS),
        "betting_edge_hist_ats": generate_betting_edge_hist_ats(_BETS),
        "season_cumulative": generate_season_cumulative(_SEASON_ROWS),
        "season_weekly": generate_season_weekly(_SEASON_ROWS),
    }


@pytest.fixture(scope="module")
def charts() -> dict[str, str]:
    rendered = _render_all()
    # Every input above is non-empty, so every generator must draw a real figure. An
    # empty-state div here would make the colour checks below pass vacuously.
    empty = [name for name, html in rendered.items() if "Plotly.newPlot(" not in html]
    assert not empty, f"generators fell back to the empty state: {empty}"
    return rendered


def _plot_args(html: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return (data, layout) from the Plotly.newPlot(id, data, layout, config) call."""
    decoder = json.JSONDecoder()
    body = html[html.index("Plotly.newPlot(") + len("Plotly.newPlot(") :]
    _div_id, end = decoder.raw_decode(body, body.index('"'))
    data, end = decoder.raw_decode(body, body.index("[", end))
    layout, _ = decoder.raw_decode(body, body.index("{", end))
    return data, layout


def _layout(html: str) -> dict[str, Any]:
    return _plot_args(html)[1]


def _data(html: str) -> list[dict[str, Any]]:
    return _plot_args(html)[0]


def test_every_chart_is_transparent_on_the_panel(charts: dict[str, str]) -> None:
    for name, html in charts.items():
        layout = _layout(html)
        assert layout["paper_bgcolor"] == theme.TRANSPARENT, name
        assert layout["plot_bgcolor"] == theme.TRANSPARENT, name


def test_no_light_theme_or_legacy_palette_literal_survives(charts: dict[str, str]) -> None:
    leaks = {
        name: [lit for lit in _LEGACY_LITERALS if lit.lower() in html.lower()]
        for name, html in charts.items()
    }
    assert not {name: found for name, found in leaks.items() if found}


def test_green_and_red_appear_only_on_realised_win_loss_charts(
    charts: dict[str, str],
) -> None:
    for name, html in charts.items():
        if name in _OUTCOME_CHARTS:
            assert theme.WIN_COLOR in html, name
            assert theme.LOSS_COLOR in html, name
        else:
            assert theme.WIN_COLOR not in html, f"{name} paints a non-outcome series green"
            assert theme.LOSS_COLOR not in html, f"{name} paints a non-outcome series red"


def test_market_is_dashed_muted_beside_the_solid_bet_type_colour(
    charts: dict[str, str],
) -> None:
    for name, target in (("mvm_ats", "ats"), ("mvm_ou", "ou"), ("mvm_wp", "wp")):
        traces = _data(charts[name])
        market = next(t for t in traces if t.get("name") == "Market")
        model = next(t for t in traces if t.get("name") == "Model")
        assert market["line"]["color"] == theme.MUTED, name
        assert market["line"]["dash"] == "dash", name
        assert model["line"]["color"] == theme.TARGET_COLORS[target], name
        assert "dash" not in model["line"], name


def test_reference_lines_use_the_theme_reference_colour(charts: dict[str, str]) -> None:
    for name in ("clv", "equity", "betting_equity", "betting_roi_type", "season_cumulative"):
        shapes = _layout(charts[name]).get("shapes", [])
        assert any(s["line"]["color"] == theme.REFERENCE_LINE for s in shapes), name


def test_stacked_calibration_rows_have_room_for_titles(charts: dict[str, str]) -> None:
    for name in ("ats_calibration", "ou_calibration"):
        layout = _layout(charts[name])
        gap = layout["yaxis"]["domain"][0] - layout["yaxis2"]["domain"][1]
        assert gap >= 0.25, f"{name}: rows only {gap:.2f} apart"
        assert layout["height"] == 480, name


def test_legends_never_sit_on_a_title(charts: dict[str, str]) -> None:
    for name in ("clv", "equity", "betting_equity", "betting_roi_type", "mvm_ats",
                 "season_cumulative"):
        assert _layout(charts[name])["legend"]["y"] >= 1.0, name
    for name in ("calibration", "accuracy_trend", "mvm_wp", "season_weekly"):
        assert _layout(charts[name])["legend"]["y"] < 0, name


def test_titles_keep_their_meaning(charts: dict[str, str]) -> None:
    assert _layout(charts["ats_calibration"])["title"]["text"] == (
        "ATS \u2014 Predicted margin vs actual"
    )
    assert "ECE = " in _layout(charts["calibration"])["title"]["text"]


def test_heatmap_is_monochrome_and_keeps_each_traces_pre_existing_direction(
    charts: dict[str, str],
) -> None:
    heatmaps = [t for t in _data(charts["heatmap"]) if t.get("type") == "heatmap"]
    # Targets render in sorted order: ats, ou, wp. The scale is reversed exactly where
    # RdYlGn_r was (the ats and ou traces) -- the pre-existing per-trace direction. This does
    # not make brighter "better" on every column: the wp trace's error columns (Brier, for
    # one) are not reversed.
    assert [t.get("reversescale", False) for t in heatmaps] == [True, True, False]
    for trace in heatmaps:
        assert trace["colorscale"] == theme.HEATMAP_SCALE
```

- [ ] **Step 2: Run it and watch it fail**

Run: `uv run pytest tests/unit/test_chart_colour_rules.py -v`
Expected: these FAIL:
- `test_no_light_theme_or_legacy_palette_literal_survives`, on `#999"`, `#667eea`, `#16a34a` and others;
- `test_green_and_red_appear_only_on_realised_win_loss_charts`, because the ATS edge histogram still paints its won bets `#16a34a` rather than `theme.WIN_COLOR`;
- `test_reference_lines_use_the_theme_reference_colour`;
- `test_stacked_calibration_rows_have_room_for_titles`;
- `test_legends_never_sit_on_a_title`, which fails for the bottom group only;
- `test_heatmap_is_monochrome_and_keeps_each_traces_pre_existing_direction`.

`test_every_chart_is_transparent_on_the_panel`, `test_titles_keep_their_meaning` and `test_market_is_dashed_muted_beside_the_solid_bet_type_colour` already PASS after Task 17: the market line falls back to `DEFAULT_COLOR`, which Task 17 made `theme.MUTED`.

- [ ] **Step 3: Fix the four charts in `api/charts/core.py`**

Replace the theme import block written in Task 17

```python
from api.charts.theme import (
    MUTED,
    SEASON_COLORS,
    TARGET_COLORS,
    LegendPosition,
    apply_dark_theme,
)
```

with

```python
from api.charts.theme import (
    BOUNDARY_LINE,
    FG,
    HEATMAP_SCALE,
    MUTED,
    REFERENCE_LINE,
    SEASON_COLORS,
    STRATEGY_COLORS,
    TARGET_COLORS,
    LegendPosition,
    apply_dark_theme,
)
```

Calibration chart -- replace

```python
            line={"dash": "dash", "color": "#999", "width": 1},
            name="Perfect Calibration",
```

with

```python
            line={"dash": "dash", "color": REFERENCE_LINE, "width": 1},
            name="Perfect Calibration",
```

replace

```python
                name="Overall",
                line={"color": "#333", "width": 3},
```

with

```python
                name="Overall",
                line={"color": FG, "width": 3},
```

and replace

```python
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend={"x": 0.02, "y": 0.98},
    )
    _apply_layout_defaults(fig)
```

with

```python
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
    )
    # Up to seven entries (reference, seasons, overall) would wrap into the title on top.
    _apply_layout_defaults(fig, legend_position="bottom")
```

CLV chart -- replace

```python
                fig.add_vline(
                    x=i,
                    line_dash="dot",
                    line_color="#ccc",
                    line_width=1,
                )
```

with

```python
                fig.add_vline(
                    x=i,
                    line_dash="dot",
                    line_color=BOUNDARY_LINE,
                    line_width=1,
                )
```

and replace

```python
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1)

    fig.update_layout(
        title={"text": "Cumulative CLV Over Time", "x": 0.5},
        xaxis_title="Game Index (Chronological)",
        yaxis_title="Cumulative Mean CLV",
        legend={"x": 0.02, "y": 0.98},
    )
```

with

```python
    # Zero line
    fig.add_hline(y=0, line_dash="dash", line_color=REFERENCE_LINE, line_width=1)

    fig.update_layout(
        title={"text": "Cumulative CLV Over Time", "x": 0.5},
        xaxis_title="Game Index (Chronological)",
        yaxis_title="Cumulative Mean CLV",
    )
```

Heatmap -- replace

```python
        # Color scale: for error metrics, reverse (green = low)
        colorscale = "RdYlGn"
        if target in ("ats", "ou"):
            colorscale = "RdYlGn_r"

        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=metric_names,
                y=[str(s) for s in all_seasons],
                text=text_values,
                texttemplate="%{text}",
                colorscale=colorscale,
                showscale=(col_idx == n_targets),
```

with

```python
        # One monochrome scale: brighter = a higher value. On this site green/red mean a
        # realised win/loss, and a metric heatmap is neither. Each trace keeps the direction
        # it already had: the ats and ou traces are reversed, exactly where the old
        # red-yellow-green scale was reversed, and the wp trace is not.
        fig.add_trace(
            go.Heatmap(
                z=z_values,
                x=metric_names,
                y=[str(s) for s in all_seasons],
                text=text_values,
                texttemplate="%{text}",
                textfont={"color": FG},
                colorscale=HEATMAP_SCALE,
                reversescale=target in ("ats", "ou"),
                xgap=2,
                ygap=2,
                colorbar={"outlinewidth": 0},
                showscale=(col_idx == n_targets),
```

Equity chart -- replace

```python
    strategy_colors = {
        "flat_stake": "#667eea",
        "kelly": "#f093fb",
    }

    for strategy in sorted(strategies):
        rows = sorted(strategies[strategy], key=lambda r: r.get("bet_index", 0))
        x_vals = [r.get("bet_index", i) for i, r in enumerate(rows)]
        y_vals = [float(r.get("bankroll", 0)) for r in rows]

        color = strategy_colors.get(strategy, DEFAULT_COLOR)
```

with

```python
    for strategy in sorted(strategies):
        rows = sorted(strategies[strategy], key=lambda r: r.get("bet_index", 0))
        x_vals = [r.get("bet_index", i) for i, r in enumerate(rows)]
        y_vals = [float(r.get("bankroll", 0)) for r in rows]

        color = STRATEGY_COLORS.get(strategy, DEFAULT_COLOR)
```

replace

```python
        fig.add_hline(
            y=first_bankroll,
            line_dash="dash",
            line_color="#999",
```

with

```python
        fig.add_hline(
            y=first_bankroll,
            line_dash="dash",
            line_color=REFERENCE_LINE,
```

and replace

```python
        yaxis_title="Bankroll ($)",
        legend={"x": 0.02, "y": 0.98},
        yaxis_tickformat="$,.0f",
    )
```

with

```python
        yaxis_title="Bankroll ($)",
        yaxis_tickformat="$,.0f",
    )
```

- [ ] **Step 4: Fix `api/charts/insights.py`**

In the module docstring, replace

```python
Imports only from :mod:`api.charts.core`, :mod:`api.insights_metrics`, stdlib,
```

with

```python
Imports only from :mod:`api.charts.core`, :mod:`api.charts.theme`,
:mod:`api.insights_metrics`, stdlib,
```

Replace the core import

```python
from api.charts.core import (
    DEFAULT_COLOR,
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)
```

with

```python
from api.charts.core import (
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)
from api.charts.theme import MUTED, REFERENCE_LINE
```

Replace

```python
_MIN_WP_ROWS_PER_SEASON = 10
```

with

```python
_MIN_WP_ROWS_PER_SEASON = 10

# The two stacked calibration rows need room between them for row 1's tick labels and axis
# title AND row 2's subplot title. At 0.15 of a ~380px chart, the "Predicted margin" axis
# title sat on top of "Residuals". A fixed height keeps the pixel gap the same at every
# card width.
_CALIBRATION_ROW_GAP = 0.26
_CALIBRATION_HEIGHT = 480
```

ATS calibration -- replace

```python
        vertical_spacing=0.15,
        subplot_titles=("Predicted margin vs actual", "Residuals"),
```

with

```python
        vertical_spacing=_CALIBRATION_ROW_GAP,
        subplot_titles=("Predicted margin vs actual", "Residuals"),
```

replace

```python
            x=[-24, 24],
            y=[-24, 24],
            mode="lines",
            line={"dash": "dash", "color": "#999", "width": 1},
```

with

```python
            x=[-24, 24],
            y=[-24, 24],
            mode="lines",
            line={"dash": "dash", "color": REFERENCE_LINE, "width": 1},
```

replace

```python
            "text": ATS_UNDERFLOW_LABEL,
            "showarrow": False,
            "font": {"size": 10, "color": "#666"},
        },
```

with

```python
            "text": ATS_UNDERFLOW_LABEL,
            "showarrow": False,
        },
```

replace

```python
            "text": ATS_OVERFLOW_LABEL,
            "showarrow": False,
            "font": {"size": 10, "color": "#666"},
        },
```

with

```python
            "text": ATS_OVERFLOW_LABEL,
            "showarrow": False,
        },
```

replace

```python
        title={"text": "ATS — Predicted margin vs actual", "x": 0.5},
        showlegend=False,
        annotations=[
```

with

```python
        title={"text": "ATS — Predicted margin vs actual", "x": 0.5},
        showlegend=False,
        height=_CALIBRATION_HEIGHT,
        annotations=[
```

and replace

```python
    fig.update_xaxes(title_text="Predicted margin", row=1, col=1)
```

with

```python
    fig.update_xaxes(title_text="Predicted margin", title_standoff=6, row=1, col=1)
```

O/U calibration -- replace

```python
        vertical_spacing=0.15,
        subplot_titles=("Predicted total vs actual", "Residuals"),
```

with

```python
        vertical_spacing=_CALIBRATION_ROW_GAP,
        subplot_titles=("Predicted total vs actual", "Residuals"),
```

replace

```python
            x=[30, 72],
            y=[30, 72],
            mode="lines",
            line={"dash": "dash", "color": "#999", "width": 1},
```

with

```python
            x=[30, 72],
            y=[30, 72],
            mode="lines",
            line={"dash": "dash", "color": REFERENCE_LINE, "width": 1},
```

and replace

```python
        title={"text": "OU — Predicted total vs actual", "x": 0.5},
        showlegend=False,
    )
    fig.update_xaxes(title_text="Predicted total", row=1, col=1)
```

with

```python
        title={"text": "OU — Predicted total vs actual", "x": 0.5},
        showlegend=False,
        height=_CALIBRATION_HEIGHT,
    )
    fig.update_xaxes(title_text="Predicted total", title_standoff=6, row=1, col=1)
```

Accuracy trend -- replace

```python
        title={"text": "Per-season accuracy and MAE", "x": 0.5},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
    _apply_layout_defaults(fig)
```

with

```python
        title={"text": "Per-season accuracy and MAE", "x": 0.5},
    )
    # Subplot titles occupy the top edge, so the legend goes underneath.
    _apply_layout_defaults(fig, legend_position="bottom")
```

Model vs market, all three -- replace every occurrence (3) of

```python
"color": DEFAULT_COLOR,
```

with

```python
"color": MUTED,
```

(Use Edit with `replace_all: true`. Each occurrence is the dashed "Market" line.) Then replace

```python
        title={"text": "WP — Model vs Market", "x": 0.5},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
    )
    _apply_layout_defaults(fig)
```

with

```python
        title={"text": "WP — Model vs Market", "x": 0.5},
    )
    # Three subplot titles occupy the top edge, so the legend goes underneath.
    _apply_layout_defaults(fig, legend_position="bottom")
```

replace

```python
        title={"text": "ATS — Model vs Market MAE", "x": 0.5},
        xaxis_title="Season",
        yaxis_title="Mean Absolute Error",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
```

with

```python
        title={"text": "ATS — Model vs Market MAE", "x": 0.5},
        xaxis_title="Season",
        yaxis_title="Mean Absolute Error",
    )
```

and replace

```python
        title={"text": "OU — Model vs Market MAE", "x": 0.5},
        xaxis_title="Season",
        yaxis_title="Mean Absolute Error",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
    )
```

with

```python
        title={"text": "OU — Model vs Market MAE", "x": 0.5},
        xaxis_title="Season",
        yaxis_title="Mean Absolute Error",
    )
```

- [ ] **Step 5: Fix `api/charts/betting.py`**

In the module docstring, replace

```python
``ratings``, and the ``BettingSimulator`` class is never named. Palettes come
through :mod:`api.charts.core` (which sources ``backtest.report`` — the only
permitted ``backtest.*`` import in the chart layer).
```

with

```python
``ratings``, and the ``BettingSimulator`` class is never named. Palettes come
from :mod:`api.charts.theme` (the dark dashboard theme), so no ``backtest.*``
module is imported here.
```

Replace the core import

```python
from api.charts.core import (
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)

logger = logging.getLogger(__name__)
```

with

```python
from api.charts.core import (
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)
# Palette: the dark dashboard theme's names, used directly (no local alias to drift). Flat and
# Kelly are staking strategies, not outcomes, so they take STRATEGY_COLORS; WIN_COLOR and
# LOSS_COLOR appear only on the edge histograms' realised win/loss series.
from api.charts.theme import (
    BOUNDARY_LINE,
    LOSS_COLOR,
    REFERENCE_LINE,
    STRATEGY_COLORS,
    WIN_COLOR,
)

logger = logging.getLogger(__name__)
```

Delete the local palette block. Replace

```python
# ---------------------------------------------------------------------------
# Strategy palette (UI-SPEC Chart series palette; core.py:441-442)
# ---------------------------------------------------------------------------

_FLAT_COLOR = "#667eea"  # flat-stake series (indigo)
_KELLY_COLOR = "#f093fb"  # Kelly series (pink)
_WIN_COLOR = "#16a34a"  # win / favorable (green-600)
_LOSS_COLOR = "#dc2626"  # loss / unfavorable (red-600)

# Empty-state copy shared by every generator (matches _safe_render fallback +
```

with

```python
# Empty-state copy shared by every generator (matches _safe_render fallback +
```

Then point the six call sites at the theme names. Replace

```python
            line={"color": _FLAT_COLOR, "width": 2},
```

with

```python
            line={"color": STRATEGY_COLORS["flat_stake"], "width": 2},
```

replace

```python
            line={"color": _KELLY_COLOR, "width": 2},
```

with

```python
            line={"color": STRATEGY_COLORS["kelly"], "width": 2},
```

replace

```python
        go.Bar(name="Flat", x=categories, y=flat_rois, marker_color=_FLAT_COLOR),
```

with

```python
        go.Bar(
            name="Flat",
            x=categories,
            y=flat_rois,
            marker_color=STRATEGY_COLORS["flat_stake"],
        ),
```

replace

```python
        go.Bar(name="Kelly", x=categories, y=kelly_rois, marker_color=_KELLY_COLOR),
```

with

```python
        go.Bar(
            name="Kelly",
            x=categories,
            y=kelly_rois,
            marker_color=STRATEGY_COLORS["kelly"],
        ),
```

replace

```python
            marker_color=_WIN_COLOR,
```

with

```python
            marker_color=WIN_COLOR,
```

and replace

```python
            marker_color=_LOSS_COLOR,
```

with

```python
            marker_color=LOSS_COLOR,
```

After these, `grep -n "_FLAT_COLOR\|_KELLY_COLOR\|_WIN_COLOR\|_LOSS_COLOR" api/charts/betting.py` prints nothing.

Replace

```python
            fig.add_vline(x=i, line_dash="dot", line_color="#ccc", line_width=1)
```

with

```python
            fig.add_vline(x=i, line_dash="dot", line_color=BOUNDARY_LINE, line_width=1)
```

replace

```python
        y=STARTING_BANKROLL,
        line_dash="dash",
        line_color="#999",
```

with

```python
        y=STARTING_BANKROLL,
        line_dash="dash",
        line_color=REFERENCE_LINE,
```

replace

```python
    Two ``go.Scatter`` lines (Flat #667eea / Kelly #f093fb) over a chronological
```

with

```python
    Two ``go.Scatter`` lines (Flat / Kelly in the theme's strategy colours) over a chronological
```

replace

```python
        yaxis_tickformat="$,.0f",
        legend={"x": 0.02, "y": 0.98},
    )
```

with

```python
        yaxis_tickformat="$,.0f",
    )
```

replace

```python
    Two ``go.Bar`` traces (Flat #667eea / Kelly #f093fb), ``barmode="group"``,
    and a dashed #999 ``add_hline`` at y=0 (the 0% ROI baseline, D-18).
```

with

```python
    Two ``go.Bar`` traces (Flat / Kelly in the theme's strategy colours),
    ``barmode="group"``, and a dashed ``REFERENCE_LINE`` ``add_hline`` at y=0 (the 0%
    ROI baseline, D-18).
```

replace

```python
    fig.add_hline(y=0, line_dash="dash", line_color="#999", line_width=1)
```

with

```python
    fig.add_hline(y=0, line_dash="dash", line_color=REFERENCE_LINE, line_width=1)
```

replace

```python
        yaxis_title="ROI (%)",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
```

with

```python
        yaxis_title="ROI (%)",
```

and replace

```python
        yaxis_title="Count",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.2},
```

with

```python
        yaxis_title="Count",
```

- [ ] **Step 6: Fix `api/charts/season.py`**

In the module docstring, replace

```python
Imports only from :mod:`api.charts.core`, :mod:`api.season_metrics`, stdlib, and
``plotly``. No imports from ``models``, ``features``, or ``ratings``. Palettes
come through :mod:`api.charts.core` (which sources ``backtest.report`` — the
only permitted ``backtest.*`` import in the chart layer), so the three target
lines use ``_get_target_color`` rather than fresh hex literals (D-13).
```

with

```python
Imports only from :mod:`api.charts.core`, :mod:`api.charts.theme`,
:mod:`api.season_metrics`, stdlib, and ``plotly``. No imports from ``models``,
``features``, or ``ratings``. Palettes come from :mod:`api.charts.theme`
(through ``_get_target_color`` and the reference-line constant), so the three
target lines use the dashboard's bet-type colours rather than fresh hex literals
(D-13).
```

Replace

```python
from api.charts.core import (
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)
from api.season_metrics import (
```

with

```python
from api.charts.core import (
    _apply_layout_defaults,
    _empty_chart_div,
    _get_target_color,
    _to_html,
)
from api.charts.theme import REFERENCE_LINE
from api.season_metrics import (
```

Replace

```python
    The idiom is copied from ``api/charts/core.py`` /
    ``api/charts/betting.py`` (dashed ``#999`` ``add_hline`` with an annotation);
    ``#999`` is the only inline color literal, matching the core reference-line
    contract (D-06). The 52.4% line is exact for spread/totals; moneyline (WP)
```

with

```python
    The idiom is copied from ``api/charts/core.py`` /
    ``api/charts/betting.py``: a dashed ``REFERENCE_LINE`` ``add_hline`` with an
    annotation, the theme's one reference-line colour (D-06). The 52.4% line is
    exact for spread/totals; moneyline (WP)
```

Replace every occurrence (2) of

```python
        line_color="#999",
```

with

```python
        line_color=REFERENCE_LINE,
```

(Use Edit with `replace_all: true`. Both are in `_add_reference_lines`.) Then replace

```python
        yaxis={"range": [0, 100]},
        legend={"x": 0.02, "y": 0.98},
    )
```

with

```python
        yaxis={"range": [0, 100]},
    )
```

and replace

```python
        yaxis={"range": [0, 100]},
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.25},
    )
    _apply_layout_defaults(fig)
```

with

```python
        yaxis={"range": [0, 100]},
    )
    # Six entries (three targets plus their rolling averages) wrap; below the plot they
    # never meet the title.
    _apply_layout_defaults(fig, legend_position="bottom")
```

- [ ] **Step 7: Confirm no stray literal is left in the chart layer**

Run (Git Bash):
```bash
git grep -n -E '"#(999|ccc|333|666)"|#667eea|#f093fb|#16a34a|#dc2626|#95a5a6|RdYlGn|#FFFFFF|#E5E7EB|DEFAULT_COLOR' -- api/charts/
```
Expected: only the `DEFAULT_COLOR` lines:
- in `api/charts/core.py`: its definition, `_get_season_color`, `_get_target_color` and the equity fallback;
- in `api/charts/__init__.py`: the back-compat re-export.

No colour literal lines, and no `RdYlGn` line: the heatmap and theme comments describe the old scale in words for exactly this reason.

- [ ] **Step 8: Run the new rules, the theme tests and the unchanged chart contract**

Run:
```bash
uv run pytest tests/unit/test_chart_colour_rules.py -v
uv run pytest tests/unit/test_chart_theme.py -q
uv run pytest tests/test_charts.py -v
uv run pytest tests/api/test_import_guard_bets.py -q
uv run pytest tests/api/test_import_guard.py -q
```
Expected: all PASS. These `tests/test_charts.py` assertions in particular must stay green with no edit to that file:
- `test_insights_ats_calibration_happy_path` ("Predicted margin")
- `test_insights_ats_calibration_overflow_bins` ("<=-21" / ">=21")
- `test_insights_ou_calibration_happy_path` ("Predicted total")
- `test_insights_model_vs_market_wp_happy_path` ("Accuracy" / "Brier" / "Log")
- `test_betting_equity_happy_path` ("Starting Bankroll")
- `test_betting_edge_hist_per_type` ("Win" / "Loss")
- `test_season_cumulative_happy_path` ("50% coin flip" / "breakeven")
- `test_season_weekly_happy_path` ("avg")
- the three `*_failure_isolation` tests ("Chart unavailable")

- [ ] **Step 9: Lint and type-check**

Run:
```bash
uv run ruff check api/charts/core.py api/charts/insights.py api/charts/betting.py api/charts/season.py tests/unit/test_chart_colour_rules.py
uv run ruff format api/charts/core.py api/charts/insights.py api/charts/betting.py api/charts/season.py tests/unit/test_chart_colour_rules.py
uv run pyright api/charts/core.py api/charts/insights.py api/charts/betting.py api/charts/season.py tests/unit/test_chart_colour_rules.py
git status --short
```
Expected: `All checks passed!`, `0 errors`. The files are named one by one, never `api/charts/`: this repo's ruff runs with `fix = true`, so a directory argument could rewrite a file this task does not commit. `git status --short` lists only the five files in the Step 10 `git add`.

- [ ] **Step 10: Commit**

```bash
git add api/charts/core.py api/charts/insights.py api/charts/betting.py api/charts/season.py tests/unit/test_chart_colour_rules.py
git commit -m "$(cat <<'EOF'
fix(redesign): every chart uses the dark theme's colours, lines and legend placement

Bet types use the target colours, the market is dashed muted, and reference and season
lines use the theme's line colours. Flat/Kelly get neutral strategy colours, and green/red
stay only on the realised win/loss histograms, named straight from the theme. The metric
heatmap leaves RdYlGn for a monochrome scale that keeps each trace's existing direction. The stacked ATS/O-U calibration charts get
room between rows, so "Predicted margin" no longer sits on "Residuals".

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 11: Re-render the charts into the worktree's COPY of the cache**

The prerendered charts live in the cache. The preview only shows the new theme after they are re-rendered. This step only ever touches the worktree's own copy, never `C:\Users\jackc\Code\nfl-predict\data\web_cache.duckdb`.

First run the Step 1 clock check from Task 17 (PowerShell). Expected: `OK to proceed`. Then refresh the copy (PowerShell):
```powershell
Copy-Item -LiteralPath 'C:\Users\jackc\Code\nfl-predict\data\web_cache.duckdb' -Destination 'C:\Users\jackc\Code\nfl-predict-redesign\data\web_cache.duckdb' -Force
```
Then re-render in place (Git Bash, from the worktree). The assertion refuses to run against anything but the worktree's copy:
```bash
uv run python -c "from pathlib import Path; import duckdb; from api.cache import _prerender_charts; p = Path('data/web_cache.duckdb').resolve(); assert 'nfl-predict-redesign' in str(p), p; conn = duckdb.connect(str(p)); n = _prerender_charts(conn); conn.close(); print('re-rendered', n, 'charts into', p)"
```
Expected: `re-rendered <N> charts into C:\Users\jackc\Code\nfl-predict-redesign\data\web_cache.duckdb`, with N over 60. Nothing to commit: `data/` is gitignored.

---

### Task 19: Responsive and accessibility pass

**Files:**
- Create: `tests/api/test_broadcast_a11y.py`
- Modify (only where the audit or the screenshots find a violation): `web/templates/base.html`, `web/templates/components/_broadcast.html`, `web/templates/components/*.html`, `web/templates/pages/*.html`, `web/static/input.css`, `web/static/css/custom.css`, and on the Python side `api/routes/pages.py` or an `api/charts/*.py` module when a fix belongs there
- Re-record (only if `components/_week_selector.html` changes): `tests/api/snapshots/week_selector_this_week.html`, `tests/api/snapshots/week_selector_prev_next.html`, with Task 5 Step 6
- Regenerate: `web/static/css/tailwind-compiled.css`

**Interfaces:**
- Consumes: every page and component from Tasks 1-16, plus the shared component classes in contract section 2.
- Produces: no new names. The audit test is the standing guard for spec section 10.

- [ ] **Step 1: Write the audit test**

Create `tests/api/test_broadcast_a11y.py`:

```python
"""Responsive and accessibility audit for the Broadcast redesign (redesign Task 19).

Rendered-HTML rules run against every page the test client serves, AND against /bets and / built
from caches that hold live, suppressed and graded bets -- the shared test_client cache has no bet
tables, so without those the bet slips, tracker, result strip, suppressed disclosure and headliner
cards would never be audited. Source rules read the templates and stylesheets directly. Each rule
pins one line of spec section 10.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Page -> the nav href that must carry aria-current="page" while it is open.
PAGES: dict[str, str] = {
    "/": "/",
    "/bets": "/bets",
    "/season": "/season",
    "/track-record": "/track-record",
    "/how-it-works": "/how-it-works",
    "/games/2024_W01_BUF@KC": "/",
}

# Classes whose element is itself transformed with skewX. Text inside them must sit in a
# descendant .unskew so it reads upright (spec 10). A component that paints its skew on a
# ::before layer instead (its text is never transformed) does not belong in this set.
SKEWED_CLASSES = frozenset(
    {
        "skew",
        "tag",
        "tag-ghost",
        "team-block",
        "team-block-lg",
        "edge-chip",
        "edge-chip-soft",
        "skew-control",
        "skew-control-active",
    }
)

_TEXTLESS_TAGS = frozenset({"script", "style", "option", "title"})
_VOID_TAGS = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source",
     "track", "wbr"}
)
# A proportion bar is drawn with an inline percentage width; 100% is a layout width, not a bar.
_BAR_WIDTH = re.compile(r"(?:^|;)\s*width:\s*(?!100%)\d+(?:\.\d+)?%")
_DIGIT = re.compile(r"\d")
_TOUCH_CLASSES = frozenset({"skew-control", "skew-control-active", "min-h-[44px]", "min-h-11"})


class _Element:
    def __init__(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tag = tag
        self.attrs = {name: value or "" for name, value in attrs}
        self.classes = frozenset(self.attrs.get("class", "").split())

    def is_labelled_image(self) -> bool:
        return self.attrs.get("role") == "img" and bool(
            _DIGIT.search(self.attrs.get("aria-label", ""))
        )


class PageAudit(HTMLParser):
    """Walks one rendered page and records every accessibility-rule violation."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[_Element] = []
        self.skewed_text: list[str] = []
        self.unlabelled_bars: list[str] = []
        self.unlabelled_images: list[str] = []
        self.unscrollable_tables = 0
        self.current_nav_hrefs: list[str] = []
        self.mobile_links_without_target: list[str] = []
        self.menu_buttons: list[dict[str, str]] = []
        self.class_sets: list[frozenset[str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = _Element(tag, attrs)
        self._check(element)
        if tag not in _VOID_TAGS:
            self.stack.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._check(_Element(tag, attrs))

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index].tag == tag:
                del self.stack[index:]
                return

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or any(el.tag in _TEXTLESS_TAGS for el in self.stack):
            return
        for element in reversed(self.stack):
            if "unskew" in element.classes:
                return
            skewed = element.classes & SKEWED_CLASSES
            if skewed:
                self.skewed_text.append(f"{sorted(skewed)} <{element.tag}>: {text[:40]!r}")
                return

    def _check(self, element: _Element) -> None:
        self.class_sets.append(element.classes)
        attrs = element.attrs
        labelled = element.is_labelled_image() or any(
            el.is_labelled_image() for el in self.stack
        )
        if _BAR_WIDTH.search(attrs.get("style", "")) and not labelled:
            self.unlabelled_bars.append(f"<{element.tag} style={attrs['style']!r}>")
        if attrs.get("role") == "img" and not element.is_labelled_image():
            self.unlabelled_images.append(f"<{element.tag} class={attrs.get('class', '')!r}>")
        if element.tag == "table" and not any(
            "overflow-x-auto" in el.classes for el in self.stack
        ):
            self.unscrollable_tables += 1
        in_nav = any(el.tag == "nav" for el in self.stack)
        if element.tag == "a" and in_nav and attrs.get("aria-current") == "page":
            self.current_nav_hrefs.append(attrs.get("href", ""))
        in_mobile_menu = any(el.attrs.get("id") == "mobile-menu" for el in self.stack)
        if element.tag == "a" and in_mobile_menu and not element.classes & _TOUCH_CLASSES:
            self.mobile_links_without_target.append(attrs.get("href", ""))
        if element.tag == "button" and attrs.get("aria-controls") == "mobile-menu":
            self.menu_buttons.append(attrs)


def _audit_html(html: str) -> PageAudit:
    audit = PageAudit()
    audit.feed(html)
    audit.close()
    return audit


def _get(client: TestClient, path: str) -> str:
    response = client.get(path)
    assert response.status_code == 200, path
    return response.text


def _audit(client: TestClient, path: str) -> PageAudit:
    return _audit_html(_get(client, path))


# ---------------------------------------------------------------------------
# Rendered-page rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", list(PAGES))
def test_skewed_boxes_never_skew_their_text(test_client: TestClient, path: str) -> None:
    assert not _audit(test_client, path).skewed_text


@pytest.mark.parametrize("path", list(PAGES))
def test_bars_and_strips_carry_their_numbers_for_screen_readers(
    test_client: TestClient, path: str
) -> None:
    audit = _audit(test_client, path)
    assert not audit.unlabelled_bars
    assert not audit.unlabelled_images


@pytest.mark.parametrize("path", list(PAGES))
def test_wide_tables_scroll_instead_of_clipping(test_client: TestClient, path: str) -> None:
    assert _audit(test_client, path).unscrollable_tables == 0


@pytest.mark.parametrize("path", list(PAGES))
def test_the_active_nav_item_is_announced(test_client: TestClient, path: str) -> None:
    assert set(_audit(test_client, path).current_nav_hrefs) == {PAGES[path]}


@pytest.mark.parametrize("path", list(PAGES))
def test_the_mobile_menu_is_wired_for_assistive_tech(
    test_client: TestClient, path: str
) -> None:
    audit = _audit(test_client, path)
    assert len(audit.menu_buttons) == 1
    button = audit.menu_buttons[0]
    assert button.get("aria-expanded") in {"true", "false"}
    assert button.get("aria-label")
    assert not audit.mobile_links_without_target


def test_this_week_slate_steps_4_3_2_1(test_client: TestClient) -> None:
    audit = _audit(test_client, "/")
    assert any(
        {"grid-cols-1", "lg:grid-cols-3", "xl:grid-cols-4"} <= classes
        and bool({"sm:grid-cols-2", "md:grid-cols-2"} & classes)
        for classes in audit.class_sets
    )


# ---------------------------------------------------------------------------
# The same rendered-page rules on the markup only a cache WITH bets produces
# ---------------------------------------------------------------------------

# Audited page -> the nav href that must carry aria-current="page" on it.
_BET_PAGES: dict[str, str] = {"/bets": "/bets", "/": "/"}
_BETS_SEASON = 2023
_BETS_WEEK = 1


@pytest.fixture(scope="module")
def bet_page_audits(tmp_path_factory: pytest.TempPathFactory) -> dict[str, PageAudit]:
    """Audits of /bets and / rendered from caches holding live, suppressed and graded bets.

    Built with the bets and headliner suites' own builders (the production materializers), so
    the audit reads the same markup those suites pin. Each page is checked for the markup that
    makes the audit meaningful before it is audited, so the rules can never pass vacuously.
    """
    from tests.api.test_bets_page import (
        _CONTAMINATED,
        _FORWARD_CLASS,
        _block,
        _client_with_tracker,
        _graded_row,
        _live_row,
        _suppressed_row,
    )
    from tests.api.test_this_week_headliner import (
        _WEEK_GAMES,
        _build_state_cache,
        _insert_predictions,
        _serving,
    )

    tmp_path = tmp_path_factory.mktemp("a11y_bets")
    # Five live slips -- one ungraded, three graded replay bets and one graded forward bet --
    # and two suppressed candidates, one with no recorded reason, so the disclosure renders
    # its tables.
    rows = [
        _live_row("2023_W01_BUF@MIA", "ou"),
        _graded_row("2023_W01_DET@KC", "win"),
        _graded_row("2023_W01_CAR@ATL", "loss"),
        _graded_row("2023_W01_CIN@CLE", "push"),
        _graded_row("2023_W01_DEN@LVR", "win", pair=_FORWARD_CLASS),
        _suppressed_row("2023_W01_SEA@SFO", "ats", "ev_below_floor"),
        _suppressed_row("2023_W01_NYJ@NE", "wp", None),
    ]
    # Stored blocks that AGREE with the graded rows, so each result strip renders.
    blocks = [
        _block(
            _CONTAMINATED,
            bets_graded=3,
            wins=1,
            losses=1,
            pushes=1,
            hit_rate=0.5,
            flat_return_units=-0.091,
        ),
        _block(
            _FORWARD_CLASS,
            bets_graded=1,
            wins=1,
            losses=0,
            pushes=0,
            hit_rate=1.0,
            flat_return_units=0.909,
        ),
    ]
    audits: dict[str, PageAudit] = {}
    with _client_with_tracker(tmp_path, blocks, "a11y_bets", rows=rows) as client:
        bets_html = _get(client, f"/bets?season={_BETS_SEASON}&week={_BETS_WEEK}")
    for marker in (
        '<li class="bet-slip',
        "data-tracker-block=",
        "data-result-strip",
        '<details id="suppressed-candidates"',
    ):
        assert marker in bets_html, f"/bets rendered no {marker}; its audit would be vacuous"
    audits["/bets"] = _audit_html(bets_html)

    home_db = tmp_path / "a11y_home.duckdb"
    _build_state_cache(home_db, "bets")
    _insert_predictions(home_db, _WEEK_GAMES)
    with _serving(home_db) as (client, _conn):
        home_html = _get(client, f"/?season={_BETS_SEASON}&week={_BETS_WEEK}")
    assert "data-headliner-bet=" in home_html, "/ rendered no headliner card; vacuous audit"
    audits["/"] = _audit_html(home_html)
    return audits


@pytest.mark.parametrize("page", list(_BET_PAGES))
def test_bet_carrying_pages_pass_every_rendered_rule(
    bet_page_audits: dict[str, PageAudit], page: str
) -> None:
    """Slips and their team blocks, tracker tiles and result strips, the suppressed tables and
    the headliner cards: upright text, labelled bars and strips, scrolling tables, the
    announced nav item and the wired mobile menu."""
    audit = bet_page_audits[page]
    assert not audit.skewed_text
    assert not audit.unlabelled_bars
    assert not audit.unlabelled_images
    assert audit.unscrollable_tables == 0
    assert set(audit.current_nav_hrefs) == {_BET_PAGES[page]}
    assert len(audit.menu_buttons) == 1
    assert not audit.mobile_links_without_target


# ---------------------------------------------------------------------------
# Source rules (templates and stylesheets)
# ---------------------------------------------------------------------------

_TEMPLATES = Path("web/templates")
_INPUT_CSS = Path("web/static/input.css")
_CUSTOM_CSS = Path("web/static/css/custom.css")
_CLASS_ATTR = re.compile(r'class="([^"]*)"')
_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_DISPLAY_TOKENS = frozenset(
    {"font-display", "display", "label", "tag", "tag-ghost", "edge-chip", "edge-chip-soft",
     "skew-control", "panel-title"}
)
_ARBITRARY_SIZE = re.compile(r"^text-\[(\d+(?:\.\d+)?)(px|rem)\]$")
_MIN_44 = re.compile(r"min-height:\s*(?:44px|2\.75rem)|min-h-(?:\[44px\]|11)\b")


def _class_lists(path: Path) -> list[list[str]]:
    return [m.group(1).split() for m in _CLASS_ATTR.finditer(path.read_text(encoding="utf-8"))]


def _rules(css: str) -> list[tuple[str, str]]:
    return [(selector.strip(), body) for selector, body in _RULE.findall(css)]


def _px(value: str, unit: str) -> float:
    return float(value) * (16 if unit == "rem" else 1)


def _at_rule(css: str, header: str) -> str:
    """Return the full text of the first ``header { ... }`` block, nested braces included."""
    start = css.index(header)
    depth = 0
    for index in range(css.index("{", start), len(css)):
        if css[index] == "{":
            depth += 1
        elif css[index] == "}":
            depth -= 1
            if depth == 0:
                return css[start : index + 1]
    msg = f"unterminated {header}"
    raise AssertionError(msg)


def _stylesheets() -> str:
    return _CUSTOM_CSS.read_text(encoding="utf-8") + _INPUT_CSS.read_text(encoding="utf-8")


@pytest.mark.parametrize("template", ["pages/track_record.html", "pages/how_it_works.html"])
def test_merged_page_grids_collapse_to_one_column(template: str) -> None:
    for classes in _class_lists(_TEMPLATES / template):
        assert "grid-cols-3" not in classes, classes
        assert "grid-cols-4" not in classes, classes
        if "lg:grid-cols-3" in classes:
            assert "grid-cols-1" in classes, classes


def test_headliner_cards_stack_on_phones() -> None:
    lists = _class_lists(_TEMPLATES / "components/_bet_headliners.html")
    assert any("grid-cols-1" in c and "lg:grid-cols-3" in c for c in lists)


@pytest.mark.parametrize("partial", ["components/_result_strip.html", "components/_week_strip.html"])
def test_strip_partials_label_themselves(partial: str) -> None:
    source = (_TEMPLATES / partial).read_text(encoding="utf-8")
    markup = re.sub(r"\{#.*?#\}|\{%.*?%\}", "", source, flags=re.DOTALL)
    first_tag = re.search(r"<([a-z][a-z0-9]*)\b([^>]*)>", markup)
    assert first_tag, partial
    # A labelled list (the week strip: one <li> per week, each with its own record as text) is
    # better than role="img", which would hide those items from assistive tech; a strip with no
    # per-item text (the result strip's tick marks) must be a labelled image.
    is_labelled_list = first_tag.group(1) in ("ol", "ul")
    assert is_labelled_list or 'role="img"' in first_tag.group(2), partial
    assert "aria-label=" in first_tag.group(2), partial


def test_condensed_display_type_is_never_below_12px_in_templates() -> None:
    offenders: list[str] = []
    for path in sorted(_TEMPLATES.rglob("*.html")):
        for classes in _class_lists(path):
            if not _DISPLAY_TOKENS & set(classes):
                continue
            for token in classes:
                match = _ARBITRARY_SIZE.match(token)
                if match and _px(match.group(1), match.group(2)) < 12:
                    offenders.append(f"{path}: {token}")
    assert not offenders


def test_condensed_display_components_are_never_below_12px() -> None:
    for selector, body in _rules(_INPUT_CSS.read_text(encoding="utf-8")):
        if not set(re.findall(r"\.([\w-]+)", selector)) & _DISPLAY_TOKENS:
            continue
        sizes = [
            _px(v, u) for v, u in re.findall(r"font-size:\s*(\d+(?:\.\d+)?)(px|rem)", body)
        ] + [_px(v, u) for v, u in re.findall(r"text-\[(\d+(?:\.\d+)?)(px|rem)\]", body)]
        assert all(size >= 12 for size in sizes), f"{selector}: {sizes}"


def test_controls_and_disclosures_declare_44px_touch_targets() -> None:
    rules = _rules(_INPUT_CSS.read_text(encoding="utf-8"))
    control = [b for s, b in rules if ".skew-control" in s and "-active" not in s]
    disclosure = [b for s, b in rules if ".honesty-note" in s and "summary" in s]
    assert any(_MIN_44.search(body) for body in control), ".skew-control needs 44px"
    assert any(_MIN_44.search(body) for body in disclosure), ".honesty-note summary needs 44px"


def test_reduced_motion_stops_lifts_and_animations() -> None:
    block = _at_rule(_stylesheets(), "@media (prefers-reduced-motion: reduce)")
    assert "transform: none" in block
    assert "animation" in block


def test_focus_ring_is_the_accent() -> None:
    """One visible ring everywhere: the universal :focus-visible rule draws an accent OUTLINE.

    Keyed on the universal rule, not on any :focus-visible rule: select.skew-control's rule
    colours its text accent and sets outline:none (its clip-path would cut a ring off), so a
    check that accepted any rule mentioning the accent would pass with no ring at all.
    """
    ring_bodies = [
        body
        for selector, body in _rules(_stylesheets())
        if selector.strip().endswith("*:focus-visible")
    ]
    assert ring_bodies, "no universal *:focus-visible rule"
    assert any(
        re.search(
            r"outline:\s*\d+px\s+solid\s+(?:#ffd400|var\(--color-accent\))",
            body,
            re.IGNORECASE,
        )
        for body in ring_bodies
    ), "the universal :focus-visible rule does not draw an accent outline"


def test_print_is_black_on_white() -> None:
    block = _at_rule(_stylesheets(), "@media print")
    assert re.search(r"background(?:-color)?:\s*(?:#fff\b|#ffffff|white)", block, re.IGNORECASE)
    assert re.search(r"(?<![-\w])color:\s*(?:#000\b|#000000|black)", block, re.IGNORECASE)
```

- [ ] **Step 2: Run the audit and list what fails**

Run: `uv run pytest tests/api/test_broadcast_a11y.py -v`
Expected: some FAIL on the first run. Every failure names the page and the element. Each failure category has exactly one fix recipe in Step 3. If everything passes, skip to Step 4.

Before fixing a `test_skewed_boxes_never_skew_their_text` failure, open `web/static/input.css` and read the named class's rule. If that rule paints the skew on a `::before` layer and never transforms the element itself, the text is already upright. In that case, remove the class from `SKEWED_CLASSES`, and replace the set's comment with one line naming the rule, e.g. `# .tag paints its skew on .tag::before, so its text is never transformed.` Otherwise apply recipe (a).

- [ ] **Step 3: Apply the fix recipe for each failing category**

(a) **Skewed text.** Wrap the text in a counter-skewed span inside the skewed element. The macros in `components/_broadcast.html` and Task 3's nav already do this (`<h2 class="tag"><span class="unskew">{{ label }}</span></h2>`), so a failure points at hand-written markup in a page or component. The change looks like this:

```jinja
{# before #}
<h2 class="tag-ghost">{{ heading }}</h2>
{# after #}
<h2 class="tag-ghost"><span class="unskew">{{ heading }}</span></h2>
```

Apply the same pattern to every `team-block`, `edge-chip`, `skew-control` link or button the audit names.

(b) **Unlabelled bar.** Put `role="img"` and an `aria-label` carrying the same numbers printed next to the bar on the bar's wrapper. Mark the decorative segments inside it `aria-hidden="true"`. Game card / game detail win-probability bar:

```jinja
<div class="flex h-1.5 gap-0.5" role="img"
     aria-label="Win probability: {{ game.away_team }} {{ '%.1f'|format((1 - game.wp_prob) * 100) }}%, {{ game.home_team }} {{ '%.1f'|format(game.wp_prob * 100) }}%">
  <span aria-hidden="true" class="skew" style="width:{{ '%.1f'|format((1 - game.wp_prob) * 100) }}%;background:{{ game.away_color|default('#1D2436') }}"></span>
  <span aria-hidden="true" class="skew" style="width:{{ '%.1f'|format(game.wp_prob * 100) }}%;background:{{ game.home_color|default('#1D2436') }}"></span>
</div>
```

The bar renders only inside the template's existing `{% if game.wp_prob is not none %}` branch. If you rewrite the game-detail bar, keep its existing wrapper classes, including `grow-in` (spec 5's grow-in animation, Task 9). For the tale-of-the-tape Elo bar (sized by flex-grow, no width, since Task 9), use `aria-label="Elo rating: {{ game.away_team }} {{ '%.0f'|format(game.context.away_elo) }}, {{ game.home_team }} {{ '%.0f'|format(game.context.home_elo) }}"`.

(c) **`role="img"` without numbers.** Add the counts to the `aria-label`. Result strip root: print the STORED block's counts, as Task 12 does -- `aria-label="Graded results in week order: wins {{ block.wins }}, losses {{ block.losses }}, pushes {{ block.pushes }}"` -- never the counts the partial derives from its marks (those are a consistency check only). Week strip root: `aria-label="Week by week record: {% for w in weeks %}week {{ w.week }} {{ w.wins }}-{{ w.losses }}{% if not loop.last %}, {% endif %}{% endfor %}"`.

(d) **Table without scroll.** Wrap it:

```html
<div class="overflow-x-auto">
  <table class="w-full text-sm"> ... </table>
</div>
```

and give its numeric cells `whitespace-nowrap`.

(e) **Active nav item not announced.** Already met by Task 3: `base.html` computes `_active_href` once (a `/games/...` path counts as This Week, `"/"`), and both the desktop and the mobile link put `skew-control-active` and `aria-current="page"` on `href == _active_href`; Task 9's game-detail route passes `f"/games/{game_id}"`. If this rule fails, a later edit broke that markup: restore Task 3 Step 4's nav loops and Task 9 Step 3's `current_path` line rather than writing a second rule, then re-run Task 3's two nav tests in `tests/api/test_pages.py`.

(f) **Mobile menu button.** Replace the hamburger button in `web/templates/base.html` with:

```html
<button type="button"
        class="md:hidden skew-control min-h-[44px] min-w-[44px]"
        aria-label="Open navigation menu"
        aria-controls="mobile-menu"
        aria-expanded="false"
        onclick="var menu = document.getElementById('mobile-menu'); var nowHidden = menu.classList.toggle('hidden'); this.setAttribute('aria-expanded', nowHidden ? 'false' : 'true');">
  <span class="unskew" aria-hidden="true">
    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16"/></svg>
  </span>
</button>
```

Every `<a>` inside `#mobile-menu` gets `skew-control` or `min-h-[44px]`.

(g) **Slate or headliner grid.**
- The slate grid in `pages/this_week.html` (every TV-window group and the ungrouped sort view) uses `class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-3"`.
- The headliner grid in `components/_bet_headliners.html` uses `class="grid grid-cols-1 lg:grid-cols-3 gap-3"`.
- Track Record and How It Works chart grids use `class="grid grid-cols-1 lg:grid-cols-3 gap-6"`.

(h) **Display type under 12px.** Change the offending size to `text-xs` (12px) in the template, or to `font-size: 12px` in the `input.css` rule.

(i) **Touch targets.** Task 1 already declares `min-height: 44px` in both the `.skew-control` rule and the `.honesty-note > summary` rule of `web/static/input.css`. If this fails, a later edit removed one: put the line back inside that rule.

```css
  min-height: 44px;
```

(j) **Reduced motion / focus ring / print.** If `web/static/css/custom.css` lacks the block the test names, append the matching block below verbatim. If a block of the same kind already exists, merge the missing declarations into it instead of adding a second one.

```css
/* Reduced motion: no card lifts, no bar grow-in (spec 10) */
@media (prefers-reduced-motion: reduce) {
  *,
  *::before,
  *::after {
    animation-duration: 0.01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: 0.01ms !important;
    scroll-behavior: auto !important;
  }

  .game-card:hover,
  .bet-slip:hover,
  .panel:hover {
    transform: none;
  }
}

/* Focus ring: 2px accent on every keyboard-focused control */
*:focus-visible {
  outline: 2px solid #FFD400;
  outline-offset: 2px;
}

/* Print: black on white, chrome hidden */
@media print {
  nav,
  footer,
  #mobile-menu,
  .no-print {
    display: none !important;
  }

  body {
    background: #fff !important;
    color: #000 !important;
  }

  .panel,
  .bet-slip,
  .stat-tile,
  .game-card,
  .score-row {
    background: #fff !important;
    color: #000 !important;
    border: 1px solid #ccc !important;
    box-shadow: none !important;
  }

  .game-card,
  .bet-slip {
    break-inside: avoid;
  }
}
```

After each category's fixes, recompile and re-run (Git Bash):
```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
uv run pytest tests/api/test_broadcast_a11y.py -v
```
Any edit to `components/_week_selector.html` (a recipe (a) or (i) fix, say) must re-record BOTH selector snapshots with Task 5 Step 6, including its class-stripped wiring check, before the commit (Part A contract note 9); otherwise `test_the_this_week_selector_renders_byte_identically` fails. Stage the two snapshot files with the fix.

Commit each category separately once its tests pass, e.g.:
```bash
git add web/ api/ tests/api/test_broadcast_a11y.py tests/api/snapshots/
git commit -m "$(cat <<'EOF'
fix(redesign): counter-skew the text inside tags, chips and nav controls

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```
(Name the category in each message: "label the win-probability and Elo bars", "wrap tables for horizontal scroll", "announce the active nav item", "wire the mobile menu button", "grid breakpoints 4/3/2/1", "12px floor for display type", "44px touch targets", "reduced motion, focus ring and print".)

- [ ] **Step 4: Prove the existing page tests still hold**

Run each file on its own:
```bash
uv run pytest tests/api/test_broadcast_a11y.py -q
uv run pytest tests/api/test_pages.py -q
uv run pytest tests/api/test_bets_page.py -q
uv run pytest tests/api/test_fragments.py -q
uv run pytest tests/unit/test_page_labels.py -q
uv run pytest tests/api/test_page_labels_routes.py -q
uv run pytest tests/unit/test_current_week_rows_reach_the_site.py -q
uv run ruff check tests/api/test_broadcast_a11y.py
uv run ruff format tests/api/test_broadcast_a11y.py
uv run pyright tests/api/test_broadcast_a11y.py
```
Expected: all PASS, ruff clean, pyright 0 errors (also run ruff and pyright on any `api/` file a fix touched). A fix that breaks one of these changed an honesty string, an id or a structure. Revert that fix and redo it without touching the contract (Global Constraints, "Structure kept").

- [ ] **Step 5: Start the preview server against the worktree's copy**

Run the Task 17 Step 1 clock check (PowerShell). Expected: `OK to proceed`, and the time is before 16:00 ET, so the capture finishes well before 16:30 ET. Then confirm the copy exists and find a replay week that has live bets (Git Bash, from the worktree):
```bash
ls -l data/web_cache.duckdb
uv run python -c "import duckdb; c = duckdb.connect('data/web_cache.duckdb', read_only=True); print(c.execute(\"SELECT season, week, count(*) AS n FROM bet_list WHERE status = 'live' GROUP BY season, week ORDER BY n DESC, season DESC, week DESC LIMIT 3\").fetchall()); c.close()"
```
Note the first `(season, week)` pair as REPLAY_SEASON / REPLAY_WEEK. If `data/web_cache.duckdb` is missing, run Task 18 Step 11 first.

Start the server with the Bash tool and `run_in_background: true` (from the worktree):
```bash
uv run uvicorn api.main:app --host 127.0.0.1 --port 8001
```
Wait for it (Git Bash):
```bash
for i in $(seq 1 30); do code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8001/); [ "$code" = "200" ] && break; sleep 1; done; echo "status $code"
```
Expected: `status 200`.

- [ ] **Step 6: Capture every page at four widths with Playwright MCP**

Load the tools once with ToolSearch `select:mcp__playwright__browser_navigate,mcp__playwright__browser_resize,mcp__playwright__browser_take_screenshot,mcp__playwright__browser_evaluate,mcp__playwright__browser_click,mcp__playwright__browser_wait_for,mcp__playwright__browser_snapshot`.

Screenshots must be saved under Playwright's allowed root. Use `C:\Users\jackc\Code\nfl-predict\.playwright-mcp\redesign\task19\`. That folder is local-only scratch (`.git/info/exclude`) and lies outside the five folders the daily run's checks photograph (`data/`, `outputs/`, `artifacts/`, `config/`, `logs/`).

Pages (NAME -> URL; GAME_URL is read on `/` with `browser_evaluate` `() => document.querySelector('a[href^="/games/"]').getAttribute('href')`):

| NAME | URL |
|---|---|
| home | `http://127.0.0.1:8001/` |
| home-edge | `http://127.0.0.1:8001/?sort=edge` |
| game | `http://127.0.0.1:8001` + GAME_URL |
| bets-current | `http://127.0.0.1:8001/bets` |
| bets-replay | `http://127.0.0.1:8001/bets?season=REPLAY_SEASON&week=REPLAY_WEEK` |
| season-current | `http://127.0.0.1:8001/season` |
| season-2025 | `http://127.0.0.1:8001/season?season=2025` |
| track-record | `http://127.0.0.1:8001/track-record` |
| how-it-works | `http://127.0.0.1:8001/how-it-works` |

Widths (resize before each pass): 1440x900, 1024x768, 768x1024, 390x844.

For every NAME at every width:
1. `browser_resize` to the width.
2. `browser_navigate` to the URL.
3. `browser_wait_for` with `time: 1` (lets Plotly draw).
4. `browser_take_screenshot` with `type: "png"`, `scale: "css"`, `fullPage: true`, `filename: "C:\Users\jackc\Code\nfl-predict\.playwright-mcp\redesign\task19\NAME-WIDTH.png"`.
5. `browser_evaluate` with `() => document.documentElement.scrollWidth - window.innerWidth`. Expected: `0` or less. A positive number is a horizontal page scroll, which is a bug to fix.

Then the interactive states, at 1440 and 390 only:
- On bets-replay, `browser_click` the `#suppressed-candidates > summary` element and the first `.honesty-note > summary` element, then screenshot as `bets-replay-open-WIDTH.png`.
- At 390, `browser_click` the button with `aria-controls="mobile-menu"`, then screenshot as `nav-open-390.png`. Check with `browser_evaluate` `() => document.querySelector('[aria-controls="mobile-menu"]').getAttribute('aria-expanded')`. Expected: `"true"`.

Then the redirects:
- `browser_navigate` to `http://127.0.0.1:8001/performance?season=2023`, then `browser_evaluate` `() => location.pathname + location.search + location.hash`. Expected: `/track-record?season=2023#seasons`.
- Same for `/betting?scope=all`. Expected: `/track-record?scope=all#betting-sim`.

- [ ] **Step 7: Review the screenshots against the mockups and fix what is wrong**

Open each screenshot with Read and compare it to the approved mockups:
- `docs/superpowers/specs/2026-10-01-broadcast-ui-mockups/02-this-week-layouts.html` (L3 + L1 tags)
- `03-bets.html`
- `04-game-detail-and-season.html`

Check this list on every screenshot:
- Nothing overlaps. In particular, no chart title meets its legend, and no subplot title meets an axis title.
- No text is clipped, and no "None", "nan" or empty chip appears.
- Skewed blocks read upright.
- The slate shows 4/3/2/1 columns at 1440/1024/768/390.
- Headliners show 3 columns at 1440 and 1024, and 1 at 768 and 390.
- Bet slips are stacked (two-line) at 768 and 390 and seven columns at 1440 and 1024 (Decision 14).
- Tables scroll inside their panel at 390, and the page does not scroll sideways.
- Green/red appear only on realised results.
- Yellow appears only as accent/emphasis.

For each problem, fix the owning template, CSS rule or chart module. Recompile CSS when templates or CSS changed. Re-run Step 4's test files. Re-capture only the affected NAME-WIDTH screenshots, then commit with a message naming the page, width and fix, e.g.:

```bash
git add web/ api/
git commit -m "$(cat <<'EOF'
fix(redesign): the Track Record ROI table scrolls inside its panel at 390px

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 8: Stop the preview server**

Run (PowerShell):
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn' -and $_.CommandLine -match '8001' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped $($_.ProcessId)" }
```
Then confirm (Git Bash): `curl -s -o /dev/null -w "%{http_code}\n" --max-time 2 http://127.0.0.1:8001/ || echo "server down"`
Expected: `000` / `server down`.

- [ ] **Step 9: Commit the audit test**

If it was not already committed with a fix category:
```bash
git add tests/api/test_broadcast_a11y.py
git commit -m "$(cat <<'EOF'
test(redesign): standing responsive and accessibility audit for every page

Pins spec section 10: upright text in skewed boxes, labelled bars and strips, scrolling
tables, the announced active nav item, a wired mobile menu, 4/3/2/1 slate columns, a
12px floor for display type, 44px targets, reduced motion, the accent focus ring and a
black-on-white print sheet.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 20: Pre-merge verification, owner review, and the merge after Plan 33-18 closes

**Files:**
- No new files. This task runs checks, collects the owner's approval, and performs the merge procedure.

**Interfaces:**
- Consumes: the whole branch (Tasks 1-19).
- Produces: a merged master, a live cache rebuilt with the themed charts, and the worktree removed.

- [ ] **Step 1: Check the clock**

Run the Task 17 Step 1 clock check (PowerShell). Expected: `OK to proceed`.

- [ ] **Step 2: Run every touched test file, one file at a time**

The list has three parts, all de-duplicated:
1. every test file the branch added or changed;
2. the fixed list of web-layer test files that assert on pages, charts or labels;
3. every test file that names a non-test module the branch changed.

Run (Git Bash, from the worktree):
```bash
changed_tests=$(git diff --name-only master...HEAD -- tests | grep -E '/test_[^/]*\.py$|^tests/test_[^/]*\.py$')
web_tests="tests/api/test_pages.py tests/api/test_bets_page.py tests/api/test_export.py tests/api/test_fragments.py tests/api/test_cache_headers.py tests/api/test_cold_start_bet_list_recovery.py tests/api/test_cache_swap_recovery.py tests/api/test_page_labels_routes.py tests/api/test_import_guard.py tests/api/test_import_guard_bets.py tests/api/test_caching.py tests/api/test_broadcast_a11y.py tests/unit/test_page_labels.py tests/unit/test_old_rule_labels.py tests/unit/test_current_week_rows_reach_the_site.py tests/unit/test_cache_ats_edge.py tests/unit/test_pipeline_md.py tests/unit/test_runbook_md.py tests/unit/test_chart_theme.py tests/unit/test_chart_colour_rules.py tests/test_charts.py tests/integration/test_phase15_integration.py"
module_tests=$(for f in $(git diff --name-only master...HEAD -- '*.py' | grep -v '^tests/'); do mod=$(echo "${f%.py}" | tr '/' '.'); git grep -l -F "$mod" -- 'tests/*.py' 'tests/**/*.py'; done)
files=$(printf '%s\n' $changed_tests $web_tests $module_tests | grep -v -E '/(conftest|__init__)\.py$' | sort -u)
echo "$files" | wc -l
failed=""
missing=""
for f in $files; do
  if [ -f "$f" ]; then
    uv run pytest "$f" -q -p no:cacheprovider > /dev/null 2>&1 && echo "PASS $f" || { echo "FAIL $f"; failed="$failed $f"; }
  else
    echo "MISSING $f"; missing="$missing $f"
  fi
done
echo "FAILED FILES:${failed:- none}"
echo "MISSING FILES:${missing:- none}"
```
Expected: `FAILED FILES: none` and `MISSING FILES: none`. A MISSING line is never skipped silently: a file in the fixed list that does not exist means the list or the branch is wrong -- find out which (a renamed or deleted test file, or a typo in the list) and fix it before going on.

For any FAIL, re-run that one file with `-v` to see the failing node ids. Then compare them with the five deliberate tripwires:
```bash
uv run python -c "import tests.phase33_state as s; print('\n'.join(s.DELIBERATE_TRIPWIRE_NODE_IDS))"
```
A tripwire node that FAILS is expected and stays red; never touch it. Any other failure is a redesign bug: fix it, commit, and re-run this step. If a tripwire node PASSES, stop and tell the owner. That is not something this branch may change.

- [ ] **Step 3: Lint and type-check every touched Python file**

Run:
```bash
py=$(git diff --name-only --diff-filter=AM master...HEAD -- '*.py')
uv run ruff check $py
uv run ruff format --check $py
uv run pyright $py
```
Expected: `All checks passed!`, `N files already formatted`, `0 errors`.

- [ ] **Step 3b: Retire the legacy colour tokens (Part A contract note 1)**

Run: `git grep -n -E 'nfl-(primary|secondary|accent|dark|light)' -- web/templates api`
Expected: no output (every template and chart module has been restyled by Tasks 3-18). If anything prints, restyle that line with the contract tokens first.

Then delete this whole block from `web/static/input.css` (comment included):

```css
/* LEGACY -- the pre-redesign palette, kept ONLY so templates not yet restyled keep rendering
   while the branch is in progress. Not `static`: it emits only while something still uses it.
   Task 20 deletes this block once `grep -rn "nfl-" web/templates api` finds nothing. */
@theme {
  --color-nfl-primary: #013369;
  --color-nfl-secondary: #D50A0A;
  --color-nfl-accent: #FF8C00;
  --color-nfl-dark: #1a1a1a;
  --color-nfl-light: #F8F9FA;
}
```

Also remove the five `--nfl-*` custom properties from `:root` in `web/static/css/custom.css` if Task 3 left any. Recompile and commit:

```bash
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
git add web/static/input.css web/static/css/custom.css web/static/css/tailwind-compiled.css
git commit -m "chore(redesign): retire the pre-redesign nfl-* colour tokens

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 4: The compiled CSS is current**

Run:
```bash
ls tools/tailwindcss.exe
./tools/tailwindcss.exe -i web/static/input.css -o web/static/css/tailwind-compiled.css --minify
git diff --exit-code --stat web/static/css/tailwind-compiled.css && echo "compiled CSS is current"
```
Expected: `compiled CSS is current`. If there is a diff, a template change was committed without recompiling. Commit the regenerated file as `chore(redesign): recompile Tailwind`, with the Co-Authored-By line.

- [ ] **Step 5: Grep audit**

Run each command (Git Bash). The expected result is stated for each.

```bash
# The compiled sheet is excluded from 1 and 2: it is generated, and Step 4 already proves it
# matches its sources. Grep the sources, never the build output.

# 1. No old navy/red tokens remain anywhere in the web layer or chart code. Expected: no output.
git grep -n -E 'nfl-(primary|secondary|accent|dark|light)' -- web/ api/ ':!web/static/css/tailwind-compiled.css'

# 2. No light-theme outcome classes remain in the templates or the Python layer. Expected: no output.
git grep -n -E 'text-green-700|bg-green-100|bg-amber-100|bg-red-100|text-red-600|bg-red-50\b' -- web/templates api/

# 3. Exactly the six page templates remain. Expected:
#    bets.html game_detail.html how_it_works.html season.html this_week.html track_record.html
ls web/templates/pages

# 4. The provenance labels live in one partial only. Expected: web/templates/components/_provenance_badge.html, three times.
for label in "Contaminated split" "Old rule -- 2025, not evidence" "Live forward record"; do git grep -l -F "$label" -- web/; done

# 5. No "friday" on /bets or /season. Expected: no output.
git grep -n -i friday -- web/templates/pages/bets.html web/templates/pages/season.html

# 6. Every honesty string is still in the templates. Expected: every line ends in a count of 1 or more.
while IFS= read -r s; do printf '%s => ' "$s"; git grep -F -c "$s" -- web/templates | awk -F: '{n+=$NF} END {print n+0}'; done <<'EOF'
Not wagering advice
1 unit = 1% of a notional bankroll
Built under the old rule on inputs later found defective; not evidence.
2026-09-15
Every scheduled game is evaluated for all three bet types. This section is the complete record of what was not bet, and why.
(no reason recorded)
A push returns the stake. Pushes are excluded from the hit-rate denominator and are never counted as a win or a loss.
A negative return here is the measurement, not a display problem.
Nothing graded yet
not measured
Expected value below the floor
Outside the eligible sub-population
Line captured after the lock
No market line for this bet type
No model prediction for this game
Odds failed the real-market check
Sizing returned no stake
Expected value could not be computed
Model agrees with the market
No honest threshold for this bet type
Win edge over the spread line below the threshold
No honest edge threshold for this bet type
No spread converter bound to the blend
Evaluated and priced; the edge was not large enough to bet.
This bet type only bets a pre-registered slice; this game is not in it.
The stored line was captured after this game's lock, 6:00 PM Eastern the day before its kickoff, so the decision could not have had it.
The game has odds for other bet types but not this one.
The model produced no output for this game -- a pipeline gap, not a market gap.
The stored price did not come from an accepted sportsbook source.
Admitted on expected value but sized to zero after the caps.
A non-finite value reached the expected-value calculation; the row is suppressed rather than tiered.
The model's number sits inside the no-bet band around the market's, so there is no side to bet and nothing was priced.
No betting threshold could be set honestly for this bet type, so it places no bets; its predictions are still published.
The price cleared the floor, but the model's edge over the market's spread-based win chance did not clear the threshold; a win bet needs both.
There was too little honest pre-lock data to set this bet type's edge threshold, so it places no bets; its predictions are still published.
The deployed blend binds no spread-to-win-probability converter, so a win bet's second test cannot run. An artifact fault, not missing odds.
Each game's line locks at 6:00 PM Eastern the day before its own kickoff -- a Thursday game locks on the Wednesday, that week's Sunday games on the Saturday.
generate_bet_list.py
populate_cache.py
EOF

# 7. No "$" anywhere /bets renders: stakes are units, never currency. Expected: no output.
git grep -n -F '$' -- web/templates/base.html web/templates/pages/bets.html web/templates/components/_broadcast.html web/templates/components/_bet_pick.html web/templates/components/_result_strip.html web/templates/components/_not_advice_banner.html web/templates/components/_old_rule_label.html web/templates/components/_ev_band_badge.html web/templates/components/_provenance_badge.html web/templates/components/_week_selector.html web/templates/components/_empty_state.html web/templates/components/_error_state.html web/templates/components/_export_buttons.html web/templates/components/_loading_skeleton.html
```

Item 6 now also covers the 13 suppression help lines and the per-game lock sentence (Global Constraints, honesty wording). Any unexpected output is a defect. Fix it, commit, and re-run Steps 2 and 5.

- [ ] **Step 6: OWNER REVIEW CHECKPOINT (blocking)**

1. Start the preview server on port 8001 (Task 19 Step 5).
2. Re-capture every page at all four widths (Task 19 Step 6) into `C:\Users\jackc\Code\nfl-predict\.playwright-mcp\redesign\final\`.
3. Stop the server (Task 19 Step 8).

Then tell the owner, in plain words:
- Where the final screenshots are (`.playwright-mcp\redesign\final\`, one file per page and width).
- That you can start the preview again at http://127.0.0.1:8001 for them to click through. It runs on a copy of the data, never between 16:30 and 17:45 ET, and you stop it afterwards.
- The test summary from Step 2, and that the grep audit in Step 5 came back clean.
- That the methodology text on How It Works was approved in Task 16, or, if it changed since, show the changed paragraphs.

Wait for one of two replies:
- **"approved"**: continue to Step 7.
- **A list of changes**: make each change, recompile CSS, re-run Steps 2-5, re-capture the affected screenshots, and report back. Repeat until approved.

- [ ] **Step 7: STOP until Plan 33-18 is closed**

Do not merge while Plan 33-18 is open. Ask the owner to confirm it is closed. The evidence is that `C:\Users\jackc\Code\nfl-predict\.planning\phases\33-live-cold-start-forward-temporal-integrity\33-18-SUMMARY.md` exists. Check (Git Bash):
```bash
ls "C:/Users/jackc/Code/nfl-predict/.planning/phases/33-live-cold-start-forward-temporal-integrity/33-18-SUMMARY.md"
```
Expected: the path is listed, AND the owner has said "go ahead and merge". Without both, stop here.

- [ ] **Step 8: Merge master INTO the branch (in the worktree)**

Merge rather than rebase. The branch's commits were reviewed and screenshotted as they are, and rebasing would rewrite every one of them. A merge keeps them byte-for-byte and records exactly what came in from master.

First the clock check (Task 17 Step 1). Then:
```bash
git merge --no-ff --no-commit master
git diff --name-only --diff-filter=U
```

This is the ONLY point where master comes into the branch: never merge master mid-run (before this task), because an unfinished 33-18 would land half-applied under the redesign. One larger merge here, after 33-18 closes, is the accepted cost.

Where conflicts can come from: master commits (Plan 33-18 and anything after the branch base) that touch a file this branch rewrote or deleted. Forecast -- check each against `git log --oneline HEAD..master -- <file>` before resolving:
- rewritten templates: `web/templates/base.html`, `pages/this_week.html`, `pages/bets.html`, `pages/game_detail.html`, `pages/season.html`, and the components restyled in Tasks 3-14 (`_game_card.html`, `_provenance_badge.html`, `_old_rule_label.html`, `_not_advice_banner.html`, `_ev_band_badge.html`, `_confidence_badge.html`, `_status_badge.html`, `_week_selector.html`, `_prediction_values.html`, `_empty_state.html`, `_error_state.html`, `_export_buttons.html`, `_week_summary.html`, and the other Task 5 controls);
- deleted templates: `pages/performance.html`, `pages/backtest.html`, `pages/betting.html`, `pages/insights.html`;
- Python the branch changed: `api/routes/pages.py`, `api/routes/fragments.py`, `api/services.py`, `api/dependencies.py`, `api/main.py`, `api/season_metrics.py`, `api/charts/*.py`;
- tests the branch changed: `tests/api/test_bets_page.py`, `tests/api/test_pages.py`, `tests/api/test_fragments.py`, `tests/api/test_export.py`, `tests/api/test_cache_headers.py`, `tests/unit/test_page_labels.py`, `tests/api/test_page_labels_routes.py`, `tests/api/test_import_guard_bets.py`, `tests/api/conftest.py`, `tests/unit/test_pipeline_md.py`;
- `web/static/input.css`, `web/static/css/custom.css`, `web/static/css/tailwind-compiled.css` (never hand-merge the compiled sheet: take either side, then recompile), and `PIPELINE.md`.

`tests/unit/test_old_rule_labels.py` is not on the list: the branch never edits it, so 33-18's changes to it merge without a conflict.

Resolution rule: re-apply master's BEHAVIOUR CHANGE onto the redesigned version, then re-run that file's tests.
- A conflicted template or component: start from the branch's (redesigned) markup and re-apply master's change to it -- the new text, condition, attribute, id or include -- in the redesigned structure. Honesty wording master adds is kept word for word.
- A template the branch deleted: keep the deletion and carry master's change into the merged page that replaced it (`pages/track_record.html` for performance / backtest / betting and the market half of insights, `pages/how_it_works.html` for the rest of insights).
- A conflicted Python or test file: keep both sides. Every line master added stays verbatim (a readout registration, a new constant, a new assertion), and every redesign change stays (the six-page template list, the new markers and label counts, `/track-record` and `/how-it-works` in the route and fragment lists, the restyled markup the tests now read).
- Never delete a master line and never revert a redesign change. If master's change cannot be expressed in the redesigned markup without changing its meaning, or the two sides disagree about behaviour: stop, show the owner both sides in plain words, and ask which to keep.

After resolving, re-run the tests of every resolved file, one file per command, plus the page-label suites:
```bash
git add <each resolved file>
uv run pytest <each resolved test file, and the test file that renders each resolved template> -q
uv run pytest tests/unit/test_old_rule_labels.py -q
uv run pytest tests/unit/test_page_labels.py -q
uv run pytest tests/api/test_page_labels_routes.py -q
git commit -m "$(cat <<'EOF'
merge(redesign): bring master (Plan 33-18 close) into redesign/broadcast-ui

Re-applied master's behaviour changes onto the redesigned markup and kept both
sides in the conflicted Python and test files (33-18's additions verbatim; the
redesign's six-page list, markers, label counts and restyled markup).

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
EOF
)"
```
Then re-run Steps 2-5 on the merged branch. Expected: all clean.

- [ ] **Step 9: Fast-forward master to the branch (in the main folder)**

The owner's uncommitted local changes in the main folder stay exactly as they are. Never stash, commit, reset or discard them. That includes `.planning/*`, `config/upstream_live/2026.json`, `config/upstream_probe_log.jsonl`, `HANDOFF.md` and `MEMPALACE_GSD_SETUP.md`.

Clock check first (Task 17 Step 1). Then (Git Bash):
```bash
git -C "C:/Users/jackc/Code/nfl-predict" status --short
git -C "C:/Users/jackc/Code/nfl-predict" merge --ff-only redesign/broadcast-ui
git -C "C:/Users/jackc/Code/nfl-predict" log --oneline -3
git -C "C:/Users/jackc/Code/nfl-predict" rev-parse master redesign/broadcast-ui
```
Expected: the merge fast-forwards, and master now points at the branch tip: `rev-parse` prints the same SHA twice. (The tip is the Step 8 merge commit, or a later fix commit if re-running Steps 2-5 after the merge produced one.) If git refuses because a local change would be overwritten, or because master moved since Step 8, stop and tell the owner. Do not force anything. Push to origin only if the owner asks.

- [ ] **Step 10: Rebuild the live cache so the prerendered charts pick up the theme**

The charts and the season KPI blob (with its new `weeks` list) live in the cache. Until it is rebuilt, the live site serves the old light charts inside the new dark pages.

1. Run the clock check (Task 17 Step 1). Expected: `OK to proceed`. Never run the rebuild between 16:30 and 17:45 ET.
2. Make sure nothing holds the live cache open. That would block the file swap on Windows. Run (PowerShell):
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn' } | Select-Object ProcessId, CommandLine
```
Expected: no rows. If a server is listed, ask the owner before stopping it, because it may be theirs.
3. From the main folder (Git Bash):
```bash
cd "C:/Users/jackc/Code/nfl-predict" && uv run python scripts/populate_cache.py
```
Expected: the script completes and reports the chart count. Then smoke-test the live pages without leaving a server running. Start the server with the Bash tool and `run_in_background: true`, from the main folder:
```bash
uv run uvicorn api.main:app --host 127.0.0.1 --port 8000
```
Wait for it, as in Task 19 Step 5 (Git Bash):
```bash
for i in $(seq 1 30); do code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 2 http://127.0.0.1:8000/); [ "$code" = "200" ] && break; sleep 1; done; echo "status $code"
```
Expected: `status 200`. Then check them (Git Bash):
```bash
for p in / /bets /season /track-record /how-it-works; do echo "$p $(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8000$p)"; done
curl -s -o /dev/null -w "/performance -> %{http_code} %{redirect_url}\n" "http://127.0.0.1:8000/performance?season=2023"
curl -s http://127.0.0.1:8000/track-record | grep -c 'rgba(0,0,0,0)'
```
Expected:
- five `200`s;
- `/performance` -> `301` with a redirect URL ending in `/track-record?season=2023` (curl may leave out the `#seasons` fragment);
- a count of 1 or more, meaning themed charts are being served.

Then stop that server (PowerShell):
```powershell
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'uvicorn' -and $_.CommandLine -match '8000' } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; "stopped $($_.ProcessId)" }
```

- [ ] **Step 11: Remove the worktree and the merged branch**

`git worktree remove` deletes git-ignored files with the folder, and the SDD ledger for this plan (`.superpowers/sdd/2026-10-01-broadcast-ui-redesign/`: `progress.md`, `preflight.md`, the amendment notes) lives in the worktree and is git-ignored. Copy it out first, so the record survives (Git Bash):
```bash
mkdir -p "C:/Users/jackc/Code/nfl-predict/.superpowers/sdd"
cp -r "C:/Users/jackc/Code/nfl-predict-redesign/.superpowers/sdd/2026-10-01-broadcast-ui-redesign" "C:/Users/jackc/Code/nfl-predict/.superpowers/sdd/"
ls "C:/Users/jackc/Code/nfl-predict/.superpowers/sdd/2026-10-01-broadcast-ui-redesign"
```
Expected: the ledger files are listed in the main folder's `.superpowers/sdd/` (ignored there too, via `.git/info/exclude`). Then run (Git Bash):
```bash
rm -f "C:/Users/jackc/Code/nfl-predict-redesign/data/web_cache.duckdb"
git -C "C:/Users/jackc/Code/nfl-predict" worktree remove "C:/Users/jackc/Code/nfl-predict-redesign"
git -C "C:/Users/jackc/Code/nfl-predict" worktree list
git -C "C:/Users/jackc/Code/nfl-predict" branch -d redesign/broadcast-ui
```
Expected: the worktree list shows only the main folder, and the branch deletes cleanly because it is fully merged.

If `worktree remove` refuses because of untracked files, list them with `git -C "C:/Users/jackc/Code/nfl-predict-redesign" status --short --ignored`. Report them to the owner rather than forcing the removal.

The local scratch folders `.superpowers/` and `.playwright-mcp/` in the main folder are ignored via `.git/info/exclude`. Tell the owner they can delete them; do not delete them unasked.

