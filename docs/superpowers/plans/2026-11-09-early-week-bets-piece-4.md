# Early-Week Bets (piece 4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** From 2026 week 11, the Tuesday daily run prices every regular-season game of the coming NFL week off the recorded early-week board and records over/under and spread decisions under a new `early` arm, starting Tue 2026-11-17.

**Architecture:** One record file fixes every number. A Tuesday side step in the daily run refreshes Elo and team form, builds that week's v2 inputs into a private gold folder, scores them with the installed v2 models, prices each game against the best regulated-book quote through a key-number converter, sizes with quarter-Kelly, and writes through the existing ledger path. No schema change and no new cache table: the chosen book travels in `eligibility_label`, and the arm's record is one JSON blob in `cache_meta`.

**Tech Stack:** Python 3.13, uv, pandas, scipy, scikit-learn (the installed v2 recipe), FastAPI + Jinja2, pytest.

**Spec:** `C:\Users\jackc\code\nfl-predict\.planning\designs\2026-11-05-profitability-fixes-design.md` sections 6.1 (best price, sizing, no-contradiction), 6.2 item 6 (key numbers), 6.3 (early window), 6.4 (page, notifier), narrowed by the rulings below. Resume file: `.planning/designs/2026-11-05-RESUME.md`. Study 1b report: `.planning/research/2026-11-05-why-not-profitable/study-1b-report.md`. Line numbers were read at master `2f3b425`.

**Owner rulings this plan rests on.**
- 2026-11-06, lean switch: piece 4 builds the pricing piece 3 skipped.
- 2026-11-09, option B: the design's rule (section 6.3: same-source slope plus 0.5-point slippage) placed 0 bets on 2,625 replayed game-bets, so the owner chose to loosen it. Early bets price with Study 1b's CLOSE slopes (O/U 0.13624, ATS 0.03688) at the real best regulated-book quote, with no slippage. Replay, flat -110, 2020-2024: O/U 62 bets of 1,310 (12.4 a season), flat return +21.7%, line moved our way 0.89 points; ATS 0 bets. That return is in-sample and may include a source effect. The owner accepted this knowingly: paper only, and live same-book closing lines test it.

**Calls made in this plan (the owner can overrule at review).**
1. **One record file, no tenth root record.** `backtest/early_arm_2026.py` (constants plus a plain-English docstring), committed alone. It answers `MIDSEASON-SWITCH-RECORD.md` section 7 ("for a later record"). Every decision snapshot stores its sha256, so a later edit is visible. Owner ruling A (no new witness or frozen test in 2026) means no state-module witness.
2. **The chosen book is stored in `eligibility_label`** as `book:<key>`, and in the snapshot. That column is unused today (always `no_subpopulation`), is covered by the hash chain, and already reaches the cache. A new column would need a new chain version.
3. **The decision day is a calendar rule:** the ET calendar day after the last kickoff of the season's earlier weeks. That is Tuesday in every remaining 2026 week. A missed day is never back-filled.
4. **Regular season, non-neutral games only,** which is what Study 1b covered. `2026_W11_MIN@SF` is flagged neutral and is declined.
5. **Caps reuse the existing values:** 5% of bankroll per game, 10% per NFL week, scaled proportionally.
6. **The converter treats the market's own line as fair.** A first scratch version without this invented edges at particular line numbers (87 spread bets, -13%).
7. **A declined row still records its side, line and book,** so it gets a same-book closing reading. That measures the real line-move figure on every priced game, not just on bets.
8. **A quote older than 6 hours cannot be the best price.** Best-price selection actively picks outliers, and stale quotes are the likeliest outliers.
9. **No-contradiction:** spread and total bets cannot contradict each other, so with moneyline out the rule cannot fire. It is built as one small check, as asked.
10. **Not built here:** the `market` arm, an early arm for moneyline, the February report, replay of early rows (reported as not replayable), the This Week headliner and the weekly email.

## File structure

| File | New / edit | Responsibility |
|---|---|---|
| `features/supplied_locks.py`, `models/v2_trainer.py` | edit | the two piece-2 review notes |
| `backtest/early_arm_2026.py` | new | the record: every constant, stdlib only |
| `api/cache_schema.py`, `forward_ledger/schema.py` | edit | `ARM_EARLY`, the page's copy of the start week |
| `forward_ledger/declarations.py` | edit | resolve `recipe-2026-v2` and `fill-v2` |
| `models/key_numbers.py`, `scripts/fit_key_numbers.py`, `config/key_number_tables.json` | new | the converter, its one-time fit, the frozen table |
| `backtest/early_pricing.py` | new | best price, EV, Kelly, caps, no-contradiction (pure) |
| `utils/game_lock.py` | edit | `early_decision_date` |
| `backtest/early_board.py` | new | which week is due, read the board, match it to games |
| `backtest/early_decide.py` | new | a week's rows and candidates (pure) |
| `forward_ledger/runner.py`, `closing.py`, `replay.py`, `backtest/bet_tracker.py` | edit | write early rows, own-book close, replay skip, tracker drop |
| `pipeline/early_week.py` | new | the run's sequence, the private gold build, the rehearsal |
| `scripts/daily_lock_pipeline.py`, `notify/report.py`, `notify/messages.py`, `notify/end_of_run.py` | edit | the side step and its pushes |
| `backtest/early_arm_report.py`, `forward_ledger/cache_sources.py`, `api/cache.py`, `api/services.py`, `api/routes/pages.py`, `web/templates/pages/bets.html`, `web/templates/components/_early_week_bets.html` | new/edit | the arm's record and section on /bets |

## Global Constraints

- Work in the worktree `C:\Users\jackc\Code\nfl-predict-v2` on a new branch `early-week` from master `2f3b425`. Its `data/bronze`, `silver`, `gold`, `baselines`, `optuna` are junctions into the LIVE lake. Tests write only under `tmp_path`. Nothing in Tasks 1-10 writes under `data/`, `artifacts/`, `ledger/` or `logs/`.
- Do not merge to master until the Wed 2026-11-11 17:00 ET check in the resume file has passed. It must be on master before Tue 2026-11-17 17:00 ET. Merge with `gh pr merge <n> --merge --admin`, then `git merge --ff-only origin/master` in the main checkout. Do not push, open a PR or merge until the owner says so.
- Nothing touching the live lake, and no test or build, runs 17:00-17:20 ET. Read ET time with PowerShell `Get-Date`, never Git Bash `TZ=`.
- Frozen, never edited: `SEASON-VERDICT-PREREGISTRATION.md`, `backtest/verdict_test_2026.py`, `ROLLING-REFIT-PREREGISTRATION.md`, `backtest/refit_preregistration_2026.py`, `backtest/fill_conventions.py`, `backtest/recipe_registry.py`, `backtest/verdict_scope_2026.py`, `backtest/shadow_pin_2026.py`, `MIDSEASON-SWITCH-RECORD.md`, `backtest/midseason_switch_2026.py`, `refit/cadence.py`, `config/gate.toml`, `models/deploy_gate.py`, `scripts/promote_models.py`, every existing slot in `tests/phase3*_state.py`. Never hand-edit `artifacts/latest.json`.
- Early boards never enter silver `odds_snapshot`. Early gold never goes under `data/`.
- Tests: named files or node ids only, `uv run pytest <file>::<test> -q -p no:cacheprovider`. Never bare `pytest`, a directory or the full suite. No `xfail`, no skip, no loosened assertion. A test pinning old behaviour is re-pointed with a dated docstring line ("RE-POINTED 2026-11-xx for backtest/early_arm_2026.py: ...").
- Lint every touched file: `uv run ruff check --no-fix --no-cache <files>`, `uv run ruff format <files>`, `uv run pyright <files>` (0 errors).
- Repo scan rules: ledger stamp columns (`arm`, `recipe_id`, `model_artifact_id`, `blend_id`, `fill_convention_id`, the repro keys, `verdict_scope`, `regime_label`) are set only by `forward_ledger/stamps.py`; never as a dict key, `.assign`, `.update` or `dict()` keyword elsewhere. `decided_at_utc` is emitted only through `backtest.weekly_bet_list.records_to_bet_list_frame`. Import the lock rule only as `import utils.game_lock as lock_rule`. No function combines `.weekday()` with a fixed hour. `api/` imports no `backtest`, `forward_ledger`, `scripts` or `notify`. No new top-level package.
- ASCII only in code, tests, templates and documents; no emoji. Never commit `.planning/`.
- Commit after each task, ending every message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01LVgorMVdeVKxoPzpgXMo2D`.
- Every subagent runs on Opus. Under 50 changed files means ONE whole-branch code reviewer.

## Review Focus

1. **A stale or outlying book quote becoming the "best price".** Expected: a quote older than 6 hours is ignored. Pinned in Task 4.
2. **The Monday-night score not yet in silver on Tuesday at 17:00.** Expected: the week is skipped by name, no rows, no crash. Pinned in Task 8.
3. **An indoor game with no forecast row on Tuesday.** Study 1b gave indoor games wind 0.0, and a no-observation row would be filled with the median instead. Expected: wind is 0.0. Pinned in Task 8.
4. **The run repeated the same day, or a later day that week.** Expected: a week with early rows is skipped, with no first-pick refusal. Pinned in Task 8.
5. **A Wednesday game (2026 week 12, Tue 11-24) whose lock passes mid-run.** Expected: that game is left out by name and the others are decided. Pinned in Task 8.

---

### Task 1: The two piece-2 review notes

**Files:**
- Modify: `features/supplied_locks.py`, `models/v2_trainer.py:147`
- Test: `tests/unit/test_supplied_locks.py` (new), `tests/unit/test_v2_trainer.py`

**Interfaces:**
- Produces: `check_supplied_locks` also raises `ValueError` on a naive instant; `walk_forward_v2` returns an empty frame for a target with no rows.

- [ ] **Step 1: Create the branch and commit this plan on it**

```powershell
git -C C:\Users\jackc\Code\nfl-predict-v2 switch -c early-week master
Move-Item C:\Users\jackc\code\nfl-predict\docs\superpowers\plans\2026-11-09-early-week-bets-piece-4.md C:\Users\jackc\Code\nfl-predict-v2\docs\superpowers\plans\
git -C C:\Users\jackc\Code\nfl-predict-v2 add docs/superpowers/plans/2026-11-09-early-week-bets-piece-4.md
git -C C:\Users\jackc\Code\nfl-predict-v2 commit -m "docs: implementation plan for early-week bets (piece 4)"
```

Expected: the commit sits on `2f3b425`; the main checkout keeps no copy.

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_supplied_locks.py
def test_a_naive_lock_is_refused():
    locks = pd.Series([pd.Timestamp("2026-11-17 17:00")], index=["g1"])
    with pytest.raises(ValueError, match="no timezone"):
        check_supplied_locks(locks, ["g1"])

def test_a_naive_lock_among_aware_ones_is_named():
    aware = pd.Timestamp("2026-11-17 17:00", tz="America/New_York")
    locks = pd.Series([aware, pd.Timestamp("2026-11-17 17:00")], index=["g1", "g2"], dtype=object)
    with pytest.raises(ValueError, match="g2"):
        check_supplied_locks(locks, ["g1", "g2"])

def test_aware_locks_pass():
    locks = pd.Series([pd.Timestamp("2026-11-17 17:00", tz="UTC")], index=["g1"])
    check_supplied_locks(locks, ["g1"])

# tests/unit/test_v2_trainer.py, built like test_a_fold_with_no_supplied_rows_is_never_fitted
def test_a_target_with_no_supplied_rows_returns_an_empty_frame(tmp_path):
    ...  # rows["ats"] = rows["ats"].iloc[0:0]
    out = walk_forward_v2(gold_dir, games, datetime(2026, 9, 1, tzinfo=UTC), prediction_rows=rows)
    assert out["ats"].empty
    assert list(out["ats"].columns) == ["game_id", "season", "week", "prediction", "actual", "fold", "fold_cutoff_utc"]
    assert not out["ou"].empty
```

- [ ] **Step 3: Run them and confirm they fail**

Run: `uv run pytest tests/unit/test_supplied_locks.py "tests/unit/test_v2_trainer.py::test_a_target_with_no_supplied_rows_returns_an_empty_frame" -q -p no:cacheprovider`
Expected: 2 FAIL for naive locks (no raise); 1 FAIL with `ValueError: No objects to concatenate`.

- [ ] **Step 4: Implement both fixes**

`check_supplied_locks`: after the null check, collect ids whose non-null `pd.Timestamp(value).tzinfo is None` and raise `f"locks carry no timezone for {n} game(s), first ten: {...}"`. `_walk_forward_target`: when `frames` is empty, return an empty frame with those seven columns and dtypes object, int64, int64, float64, float64, object, `datetime64[ns, UTC]`.

- [ ] **Step 5: Run the tests plus the existing callers**

Run: `uv run pytest tests/unit/test_supplied_locks.py tests/unit/test_v2_trainer.py tests/unit/test_market_rating.py tests/unit/test_league_level.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 6: Lint and commit** (`fix: refuse naive supplied locks; empty prediction_rows target returns an empty frame`)

---

### Task 2: The record, the arm, and what the ledger accepts

**Files:**
- Create: `backtest/early_arm_2026.py` (its own commit, nothing else in it), `tests/unit/test_early_arm_record.py`
- Modify: `api/cache_schema.py:188-190`, `api/cache.py:36-38`, `forward_ledger/schema.py`, `forward_ledger/declarations.py:214-257`, `tests/unit/test_bet_list_schema.py:846`, `tests/unit/test_cache_data_format.py:61`, `tests/unit/test_forward_ledger_store.py`

**Interfaces:**
- Produces, in `backtest/early_arm_2026.py` (stdlib only, no logic):
  - `RECORD_PATH: str = "backtest/early_arm_2026.py"`
  - `EARLY_RECIPE: str = "recipe-2026-v2"`, `FILL_V2_ID: str = "fill-v2"`
  - `EARLY_FROM: tuple[int, int] = (2026, 11)`, `EARLY_TARGETS: tuple[str, ...] = ("ats", "ou")`
  - `REGULATED_BOOKS: tuple[str, ...] = ("betmgm", "betrivers", "draftkings", "fanduel")`, `MIN_BOOKS: int = 2`, `QUOTE_MAX_AGE_HOURS: int = 6`
  - `EXPECTED_MOVE_SLOPE: Mapping[str, float]` = read-only `{"ats": 0.03688, "ou": 0.13624}`
  - `SLIPPAGE_POINTS: float = 0.0`, `EV_FLOOR: float = 0.0`
  - `KELLY_FRACTION: float = 0.25`, `PER_GAME_CAP: float = 0.05`, `PER_WEEK_CAP: float = 0.10`
  - `BOOK_LABEL_PREFIX: str = "book:"`
  - `KEY_NUMBER_FIT_SEASONS: tuple[int, int] = (2015, 2025)`, `KEY_NUMBER_TABLES_PATH: str = "config/key_number_tables.json"`
- Produces: `api.cache_schema.ARM_EARLY = "early"`, `ARMS = ("live", "shadow", "early")`, `EARLY_ARM_FROM: tuple[int, int] = (2026, 11)` (the page's own copy), re-exported from `forward_ledger.schema`.
- Produces: `resolve_recipe("recipe-2026-v2")` returns `RefitRecipe(recipe_id="recipe-2026-v2", fill_convention_id="fill-v2")`; `resolve_fill_convention("fill-v2")` returns it.

The record's docstring must state in plain English: what the arm is; that it answers section 7 of `MIDSEASON-SWITCH-RECORD.md`; the 2026-11-09 ruling with its replay figures and the warning that they are in-sample and may include a source effect; that `fill-v2` means a real quote at a named regulated book with no slippage, registered here because `backtest/fill_conventions.py` is frozen; that the chosen book is `eligibility_label`; and that editing the file after week 11's first decision destroys the evidence.

- [ ] **Step 1: Write the record and commit it ALONE**

```bash
git add backtest/early_arm_2026.py
git commit -m "record: the early-week arm from 2026 week 11

Answers MIDSEASON-SWITCH-RECORD.md section 7. Owner ruling 2026-11-09 (option B): price off
Study 1b's close slopes at the real best regulated-book quote, no slippage."
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/unit/test_early_arm_record.py
def test_the_recipe_is_the_installed_one():
    assert record.EARLY_RECIPE == INSTALLED_RECIPE == V2_RECIPE_NAME

def test_moneyline_is_out():
    assert record.EARLY_TARGETS == ("ats", "ou")

def test_the_page_copy_of_the_start_week_agrees():
    assert api.cache_schema.EARLY_ARM_FROM == record.EARLY_FROM

def test_the_record_imports_only_the_standard_library(): ...  # ast scan of its imports

def test_fill_v2_resolves_and_fill_v1_still_does():
    assert resolve_fill_convention("fill-v2") == "fill-v2"
    assert resolve_fill_convention("fill-v1") == "fill-v1"

def test_the_v2_recipe_resolves():
    entry = resolve_recipe("recipe-2026-v2")
    assert isinstance(entry, RefitRecipe) and entry.fill_convention_id == "fill-v2"

def test_an_unknown_recipe_is_still_refused():
    with pytest.raises(UnknownRecipeError):
        resolve_recipe("recipe-nope")

# tests/unit/test_forward_ledger_store.py, with make_row
def test_an_early_row_and_a_live_row_are_two_keys(tmp_path): ...
```

Re-point `tests/unit/test_bet_list_schema.py:846` to `ARMS == ("live", "shadow", "early")`; add `ARM_EARLY` and `EARLY_ARM_FROM` beside `ARMS` in `tests/unit/test_cache_data_format.py:61`.

- [ ] **Step 3: Run them and confirm they fail** (`AttributeError` / `UnknownFillConventionError`)

Run: `uv run pytest tests/unit/test_early_arm_record.py -q -p no:cacheprovider`

- [ ] **Step 4: Implement**

Add the constants. In `resolve_recipe`, after the re-fit branch and with a function-local import, return the `RefitRecipe` for `EARLY_RECIPE`. In `resolve_fill_convention`, accept `REGISTERED_FILL_CONVENTIONS + (FILL_V2_ID,)`. `backtest/fill_conventions.py` stays byte-unchanged.

- [ ] **Step 5: Run the tests**

Run: `uv run pytest tests/unit/test_early_arm_record.py tests/unit/test_bet_list_schema.py tests/unit/test_cache_data_format.py tests/unit/test_forward_ledger_store.py tests/unit/test_fill_conventions.py tests/unit/test_ledger_stamping_site.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 6: Lint and commit** (`feat(ledger): the early arm, the v2 recipe and fill-v2 resolve`)

---

### Task 3: The key-number converter

**Files:**
- Create: `models/key_numbers.py`, `scripts/fit_key_numbers.py`, `config/key_number_tables.json`, `tests/unit/test_key_numbers.py`

**Interfaces:**
- Consumes: `KEY_NUMBER_FIT_SEASONS`, `KEY_NUMBER_TABLES_PATH`.
- Produces:
  - `@dataclass(frozen=True) class KeyNumberTable: sd: float; low: int; factors: tuple[float, ...]`
  - `@dataclass(frozen=True) class OutcomeChance: above: float; at: float; below: float`
  - `fit_key_number_table(actual: ArrayLike, closing: ArrayLike, *, low: int, high: int, mirror: bool) -> KeyNumberTable`
  - `outcome_chance(table: KeyNumberTable, *, market_line: float, center: float, threshold: float) -> OutcomeChance`
  - `load_key_number_tables(path: Path | str = KEY_NUMBER_TABLES_PATH) -> dict[str, KeyNumberTable]` (keys `ats`, `ou`)

The algorithm the signatures do not determine:

```python
# base(c)[k] = norm.cdf((k + .5 - c) / sd) - norm.cdf((k - .5 - c) / sd), for integer k in [low, high]
# fit: sd = std(actual - closing); expected = sum of base(c_i) over games; observed = counts of round(actual)
#      mirror=True pools k with -k in both; factors = (observed + 10) / (expected + 10)
# chance: at_line = base(market_line) * factors
#         scale every k above market_line, and every k below it, so the two sides' total mass is equal
#         mass = base(center) * factors * scale; mass /= mass.sum()
#         above/at/below are the mass with k > threshold, k == threshold, k < threshold
```

- [ ] **Step 1: Write the failing tests**

```python
def test_the_market_line_is_fair():
    chance = outcome_chance(TABLE, market_line=44.5, center=44.5, threshold=44.5)
    assert chance.above == pytest.approx(chance.below, abs=1e-9) and chance.at == 0

def test_a_whole_number_line_has_a_push():
    chance = outcome_chance(TABLE, market_line=3.0, center=3.0, threshold=3.0)
    assert chance.above == pytest.approx(chance.below, abs=1e-9) and chance.at > 0

def test_the_three_chances_sum_to_one(): ...
def test_moving_the_center_up_raises_above(): ...

def test_a_half_point_across_a_key_number_is_worth_more():
    # table with factor 3.0 at 3 and 1.0 elsewhere
    across = chance(2.5).above - chance(3.5).above
    plain = chance(4.5).above - chance(5.5).above
    assert across > plain

def test_a_mirrored_fit_is_symmetric(): ...
def test_a_fit_refuses_unequal_lengths(): ...

def test_the_committed_tables_know_the_key_numbers():
    tables = load_key_number_tables()
    margin = tables["ats"]
    assert margin.factors[3 - margin.low] > 2.0
    assert margin.factors[7 - margin.low] > 1.4
    assert margin.factors[0 - margin.low] < 0.5
    assert 12.0 < margin.sd < 15.0 and 12.0 < tables["ou"].sd < 15.0
```

- [ ] **Step 2: Run them and confirm they fail** (`ModuleNotFoundError`)

Run: `uv run pytest tests/unit/test_key_numbers.py -q -p no:cacheprovider`

- [ ] **Step 3: Implement `models/key_numbers.py` and `scripts/fit_key_numbers.py`**

The script reads `data.upstream_pin.load_schedules(range(2015, 2026))`, keeps `game_type == "REG"` with `result`, `total`, `spread_line`, `total_line` all present, fits `ats` from `(result, spread_line)` on `[-70, 70]` with `mirror=True` and `ou` from `(total, total_line)` on `[0, 120]` with `mirror=False`, and writes `{"fit_seasons": [2015, 2025], "ats": {...}, "ou": {...}}` as ASCII JSON with sorted keys and `indent=2`.

- [ ] **Step 4: Run the fit once, outside 17:00-17:20 ET**

Run: `uv run python -m scripts.fit_key_numbers`
Expected: prints two lines near `ats sd=12.73` and `ou sd=13.19`; writes `config/key_number_tables.json`.

- [ ] **Step 5: Run the tests** (all pass), **lint, commit** (`feat(models): key-number converter anchored at the market line`)

---

### Task 4: Best price, sizing, caps, no-contradiction

**Files:**
- Create: `backtest/early_pricing.py`, `tests/unit/test_early_pricing.py`

**Interfaces:**
- Consumes: Task 2's constants, Task 3's `KeyNumberTable`, `outcome_chance`; `backtest.ou_ev_chain.american_to_payout`; `backtest.ats_ev_chain.ATS_SIDES`; `backtest.selector_strategies.OU_SIDES`.
- Produces:
  - `@dataclass(frozen=True) class Quote: sportsbook: str; line: float; price_up: int; price_down: int; last_update: datetime | None` (`up` is over or home cover; `line` is on the ledger's scale, positive when home is favoured)
  - `@dataclass(frozen=True) class PricedSide: side: str; sportsbook: str; line: float; price: int; consensus: float; expected_close: float; p_win: float; p_push: float; p_lose: float; ev: float`
  - `usable_quotes(quotes: Sequence[Quote], captured_at: datetime) -> list[Quote]`
  - `price_target(target: str, model_value: float, quotes: Sequence[Quote], table: KeyNumberTable, *, captured_at: datetime) -> PricedSide | str` (the `str` is a rejection code: `missing_snapshot` with fewer than `MIN_BOOKS` usable quotes, `no_bet_side` when the model equals the consensus)
  - `quarter_kelly_fraction(p_win: float, p_lose: float, american_odds: int) -> float`
  - `apply_caps(stakes: Sequence[tuple[str, float]]) -> list[float]` (`(game_id, fraction)` pairs)
  - `assert_no_contradiction(bets: Sequence[Mapping[str, Any]]) -> None`
  - `book_label(sportsbook: str) -> str`

Rules: consensus is the median line of usable quotes. The side is up when the model is above consensus. `expected_close = consensus + slope * (model - consensus)`. For each quote, `threshold = line + SLIPPAGE_POINTS` going up and `line - SLIPPAGE_POINTS` going down; `ev = p_win * american_to_payout(price) - p_lose`. The best quote is the highest `ev`, ties to the lower book name. A book missing either price is not usable. Kelly is `KELLY_FRACTION * max(0, (b*p - q) / (b * (p + q)))` with `b = american_to_payout(odds)`, with no other multiplier. Caps apply per game first, then per week, each scaling proportionally.

- [ ] **Step 1: Write the failing tests**

```python
NOW = datetime(2026, 11, 17, 22, 5, tzinfo=UTC)
def q(book, line=44.5, up=-110, down=-110, age_h=1): ...  # builds a Quote

def test_the_better_price_wins():
    priced = price_target("ou", 53.0, [q("draftkings"), q("fanduel", up=-105)], TABLES["ou"], captured_at=NOW)
    assert priced.sportsbook == "fanduel" and priced.price == -105

def test_an_offshore_book_is_never_used():
    priced = price_target("ou", 53.0, [q("draftkings"), q("fanduel"), q("bovada", up=120)], TABLES["ou"], captured_at=NOW)
    assert priced.sportsbook != "bovada"

def test_a_stale_quote_is_ignored():
    priced = price_target("ou", 53.0, [q("draftkings"), q("fanduel"), q("betmgm", up=120, age_h=7)], TABLES["ou"], captured_at=NOW)
    assert priced.sportsbook != "betmgm"

def test_one_book_is_no_market():
    assert price_target("ou", 53.0, [q("draftkings")], TABLES["ou"], captured_at=NOW) == "missing_snapshot"

def test_no_disagreement_is_no_side():
    assert price_target("ou", 44.5, [q("draftkings"), q("fanduel")], TABLES["ou"], captured_at=NOW) == "no_bet_side"

def test_a_two_point_gap_does_not_clear_the_cut():
    assert price_target("ou", 46.5, FOUR_BOOKS, TABLES["ou"], captured_at=NOW).ev < 0

def test_an_eight_point_gap_clears_the_cut():
    priced = price_target("ou", 52.5, FOUR_BOOKS, TABLES["ou"], captured_at=NOW)
    assert priced.side == "over" and priced.ev > 0

def test_a_spread_side_uses_the_home_positive_scale():
    priced = price_target("ats", 6.0, SPREAD_BOOKS_AT_3_5, TABLES["ats"], captured_at=NOW)
    assert priced.side == "home_cover" and priced.line == 3.5

def test_quarter_kelly_value():
    assert quarter_kelly_fraction(0.55, 0.45, -110) == pytest.approx(0.01375)
    assert quarter_kelly_fraction(0.50, 0.50, -110) == 0.0

def test_the_game_cap_scales_a_game():
    assert apply_caps([("g", 0.04), ("g", 0.04)]) == pytest.approx([0.025, 0.025])

def test_the_week_cap_scales_the_week():
    assert apply_caps([(f"g{i}", 0.04) for i in range(4)]) == pytest.approx([0.025] * 4)

def test_stakes_under_both_caps_are_untouched():
    assert apply_caps([("a", 0.01), ("b", 0.02)]) == pytest.approx([0.01, 0.02])

def test_a_spread_and_a_total_never_contradict(): ...
def test_both_sides_of_one_target_are_refused(): ...  # ValueError
def test_a_win_bet_and_a_spread_bet_on_opposite_teams_are_refused(): ...  # ValueError
```

`TABLES` loads the committed file; `FOUR_BOOKS` is one fresh `-110` quote at 44.5 per regulated book.

- [ ] **Step 2: Run them and confirm they fail**, **Step 3: Implement**, **Step 4: Run them and confirm they pass**

Run: `uv run pytest tests/unit/test_early_pricing.py -q -p no:cacheprovider`

- [ ] **Step 5: Lint and commit** (`feat(backtest): early-week best price, quarter-Kelly and caps`)

---

### Task 5: Which week is due, and reading the board

**Files:**
- Modify: `utils/game_lock.py`
- Create: `backtest/early_board.py`, `tests/unit/test_early_board.py`

**Interfaces:**
- Produces:
  - `utils.game_lock.early_decision_date(prior_kickoffs: Iterable[Any]) -> date`: the ET calendar day after the last of the kickoffs, read through `kickoff_wall_clock_et`; naive refused through `_require_aware`, null through `MissingKickoffError`, empty through `ValueError`.
  - `@dataclass(frozen=True) class EarlyBoard: captured_at: datetime; events: list[dict[str, Any]]`
  - `read_board_for_run(path: Path, run_date_et: date) -> EarlyBoard | None`: the last record of `early_lines.jsonl` whose `captured_at_utc` falls on that ET date, else None.
  - `latest_board(path: Path) -> EarlyBoard | None` (rehearsal only).
  - `early_week_due(games: pd.DataFrame, run_date_et: date) -> tuple[int, int] | None`: `(season, week)` for season `EARLY_FROM[0]`, week at or after `EARLY_FROM[1]`, every game of that week `game_type == "REG"`, whose `early_decision_date` of all lower weeks' kickoffs equals `run_date_et`.
  - `board_odds(board: EarlyBoard, week_games: pd.DataFrame, ingester: OddsDataIngester) -> pd.DataFrame`: `ingester.transform_odds_data(board.events, schedule=week_games, locks=<lock_rule.lock_frame as dict>, captured_at=board.captured_at, capture_kind="decision")` then `validate_odds_data`. Never written anywhere.
  - `quotes_for(odds: pd.DataFrame, target: str) -> dict[str, list[Quote]]`: game id to quotes; ATS reads `spread`, `spread_ju_home` (up), `spread_ju_away` (down); O/U reads `total`, `total_over_ju` (up), `total_under_ju` (down).

- [ ] **Step 1: Write the failing tests**

```python
def test_the_day_after_a_monday_night_game():
    assert lock_rule.early_decision_date([pd.Timestamp("2026-11-17T01:15Z")]) == date(2026, 11, 17)

def test_the_day_after_a_sunday_night_game():
    assert lock_rule.early_decision_date([pd.Timestamp("2026-11-16T01:20Z")]) == date(2026, 11, 16)

def test_a_naive_kickoff_is_refused(): ...

def test_week_11_is_due_on_tuesday_11_17():
    assert early_week_due(SCHEDULE, date(2026, 11, 17)) == (2026, 11)

def test_a_missed_tuesday_is_never_back_filled():
    assert early_week_due(SCHEDULE, date(2026, 11, 18)) is None

def test_week_10_is_before_the_arm():
    assert early_week_due(SCHEDULE, date(2026, 11, 10)) is None

def test_a_postseason_week_is_never_due(): ...
def test_only_todays_board_is_read(tmp_path): ...
def test_the_last_of_todays_boards_is_used(tmp_path): ...

def test_a_home_favourite_is_stored_positive():
    # board with Detroit -3.5 at home, -105 home price, -115 away price
    quote = quotes_for(board_odds(BOARD, GAMES, INGESTER), "ats")[GAME][0]
    assert quote.line == 3.5 and quote.price_up == -105 and quote.price_down == -115

def test_a_game_past_its_lock_is_dropped(): ...
```

`INGESTER` is `OddsDataIngester.__new__(OddsDataIngester)` with `sportsbook_priority = []`, the pattern of `test_one_lock_rule_source_scan.py:218`.

- [ ] **Step 2: Run them and confirm they fail**, **Step 3: Implement**, **Step 4: Run them, plus the lock-rule scan, and confirm they pass**

Run: `uv run pytest tests/unit/test_early_board.py tests/unit/test_one_lock_rule_source_scan.py -q -p no:cacheprovider`

- [ ] **Step 5: Lint and commit** (`feat(backtest): the early decision day and the early-board reader`)

---

### Task 6: Writing early rows, the same-book close, and keeping the old record clean

**Files:**
- Modify: `forward_ledger/runner.py`, `forward_ledger/closing.py:156-199,311-371`, `forward_ledger/replay.py:419-424`, `backtest/bet_tracker.py:402-405`
- Test: `tests/unit/test_early_ledger.py` (new), `tests/unit/test_closing_capture.py`, `tests/unit/test_replay_ledger.py`, `tests/unit/test_bet_tracker_verdict.py`

**Interfaces:**
- Consumes: `ARM_EARLY`, `EARLY_RECIPE`, `FILL_V2_ID`, `BOOK_LABEL_PREFIX`.
- Produces: `forward_ledger.runner.record_early_week(bundle: DecisionBundle, *, odds: pd.DataFrame, gold_dir: Path, decided_at: datetime, publish_by: datetime | None, meta_extra: Mapping[str, Any], ledger_dir: Path | str = LEDGER_DIR, artifacts_dir: Path = Path("artifacts"), manifest_dir: Path | str | None = None, log: LogCallable | None = None, repo_dir: Path | str | None = None, clock: Callable[[], datetime] | None = None) -> RecordOutcome`
- Produces: `select_closing_capture(..., *, sportsbook: str | None = None)`; `None` keeps today's behaviour byte for byte.

`record_early_week` follows `record_forward_slate` steps 2-6 and 8 without a shadow arm: `in_force_generation` must name `EARLY_RECIPE`, otherwise raise `UnknownGenerationError`. Stamp with `scope=None`, `ledger_has_verdict_rows=False` (every early row is `pre_verdict`, so it can never count toward the R6 verdict), `recipe_id=EARLY_RECIPE`, `fill_convention_id=FILL_V2_ID`, `arm=ARM_EARLY`, `registered_ids=_registered_ids(generation, ledger)`. Pass `meta_extra` to both `compute_snapshot_digest` and `write_decision_snapshot`.

In `finalize_closing`, an early row's sportsbook is `eligibility_label` with the prefix removed. A label without the prefix on an early row is left unfinalized and logged, never read from another book.

- [ ] **Step 1: Write the failing tests**

```python
# test_early_ledger.py, with a tiny artifacts dir and installed-generation record like tests/unit/test_owner_install.py
def test_early_rows_are_stamped_early_and_never_count(tmp_path):
    outcome = record_early_week(BUNDLE, ...)
    row = outcome.appended_rows[0]
    assert row["arm"] == "early" and row["verdict_scope"] == "pre_verdict"
    assert row["recipe_id"] == "recipe-2026-v2" and row["fill_convention_id"] == "fill-v2"

def test_the_same_decision_again_writes_nothing(tmp_path): ...  # second call: appended == 0, wrote is False
def test_nothing_lands_after_the_deadline(tmp_path): ...  # PublishDeadlinePassedError, no snapshot dir
def test_models_that_are_not_v2_are_refused(tmp_path): ...  # UnknownGenerationError

def test_an_early_bet_grades_at_its_own_price():
    # over 44.5 at -105, total 50: payout_flat == pytest.approx(100 / 105)

# test_closing_capture.py
def test_an_early_row_reads_its_own_book_at_the_close(): ...  # closing_sportsbook == "fanduel" though draftkings ranks first
def test_an_early_row_whose_book_has_no_close_is_no_closing_market(): ...
def test_a_live_row_keeps_the_fill_v1_book_order(): ...

# test_replay_ledger.py
def test_an_early_row_is_reported_not_replayable(tmp_path): ...

# test_bet_tracker_verdict.py
def test_an_early_row_never_enters_a_forward_block():
    rows = [_row("A", GRADING_STATUS_WIN), _row("E", GRADING_STATUS_WIN, arm="early")]
    frame = ...  # built from rows as the neighbouring tests build theirs
    assert aggregate_all_blocks(frame)[0].bets_graded == 1
```

- [ ] **Step 2: Run them and confirm they fail**, **Step 3: Implement**, **Step 4: Run them, plus the guards**

Run: `uv run pytest tests/unit/test_early_ledger.py tests/unit/test_closing_capture.py tests/unit/test_replay_ledger.py tests/unit/test_bet_tracker_verdict.py tests/unit/test_ledger_stamping_site.py tests/unit/test_closing_rows_never_priced.py tests/unit/test_verdict_scope.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Lint and commit** (`feat(ledger): write early rows; close from the bet's own book`)

---

### Task 7: One week's rows

**Files:**
- Create: `backtest/early_decide.py`, `tests/unit/test_early_decide.py`

**Interfaces:**
- Consumes: Tasks 3-5; `backtest.weekly_bet_list.records_to_bet_list_frame`, `backtest.bet_selector.SelectionResult`, `DEFAULT_BANKROLL`.
- Produces:
  - `@dataclass(frozen=True) class EarlyDecision: frame: pd.DataFrame; candidates: pd.DataFrame`
  - `decide_early_week(week_games: pd.DataFrame, *, predictions: Mapping[str, Mapping[str, float]], quotes: Mapping[str, Mapping[str, Sequence[Quote]]], tables: Mapping[str, KeyNumberTable], locks: Mapping[str, datetime], captured_at: datetime, decided_at: datetime) -> EarlyDecision`

Every game gets exactly one record per `EARLY_TARGETS` target, built in the selector's record shape: `game_id, season, week, target, bet_side, model_spread|model_total, closing_spread|closing_total (the chosen book's line), slipped_line (= line), calibrated_p_side (= p_win), per_bet_ev, kelly_stake (fraction x DEFAULT_BANKROLL), subpop_label (book_label), snapshot_ts (captured_at), freeze_ts (the game's lock), selected_odds, clv (None)`. Rejected records add `rejection_reason`: no prediction is `missing_prediction`, `price_target` returning a code is that code, `ev <= EV_FLOOR` is `ev_below_floor`, and a zero capped stake is `zero_kelly_stake`. A rejected record keeps its side, line, book and price when it was priced. The frame comes from `records_to_bet_list_frame(SelectionResult(...), fits={t: ev_floor_t=EV_FLOOR}, run_mode="forward", decided_at=decided_at)`, with the fits built as a `SimpleNamespace` cast to `dict[str, WeeklyChainFit]`, since only `.ev_floor_t` is read. Call `assert_no_contradiction` on the selected records. `candidates` has one row per `(game_id, target)`: consensus, expected_close, model_value, bet_side, sportsbook, line, price, p_win, p_push, p_lose, ev.

- [ ] **Step 1: Write the failing tests**

```python
def test_every_game_gets_one_row_per_early_target():
    d = decide_early_week(TWO_GAMES, **ARGS)
    assert len(d.frame) == 4 and set(d.frame["target"]) == {"ats", "ou"}

def test_a_bet_row_carries_book_price_and_stake():
    row = bet_row(d)
    assert row.status == "live" and row.eligibility_label == "book:fanduel"
    assert row.selected_odds == -105 and row.slipped_line == row.line
    assert row.flat_stake == 1.0 and row.ev_tier in ("low", "medium", "high")
    assert row.stake_units == pytest.approx(expected_fraction * 100)

def test_a_declined_row_keeps_its_side_line_and_book():
    row = declined_row(d)
    assert row.status == "suppressed" and row.rejection_reason == "ev_below_floor"
    assert row.bet_side in ("over", "under") and row.line == 44.5 and row.eligibility_label.startswith("book:")
    assert pd.isna(row.flat_stake)

def test_a_game_with_no_prediction_is_declined_by_name():
    assert reasons(d, "g2") == {"missing_prediction"}

def test_no_moneyline_row_is_ever_written():
    assert "wp" not in set(d.frame["target"])

def test_the_row_is_decided_before_its_lock():
    assert_decided_at_before_freeze(stamped_row(d))
```

- [ ] **Step 2: Run them and confirm they fail**, **Step 3: Implement**, **Step 4: Run them, plus `tests/unit/test_decided_at_utc.py`, and confirm they pass**

Run: `uv run pytest tests/unit/test_early_decide.py tests/unit/test_decided_at_utc.py -q -p no:cacheprovider`

- [ ] **Step 5: Lint and commit** (`feat(backtest): a week's early-arm rows`)

---

### Task 8: The Tuesday run and its rehearsal

**Files:**
- Create: `pipeline/early_week.py`, `tests/unit/test_early_week_run.py`

**Interfaces:**
- Consumes: Tasks 2-7; `backtest.study_1b_inputs.build_early_inputs`, `models.v2_trainer.predict_v2`, `models.artifacts.load_model_artifact`, `resolve_production_artifacts`, `refit.generations.in_force_recipe`, `pipeline.steps.persist_current_season_elo`, `step_build_team_form`, `pipeline.daily_steps.build_weather_features_through`, `scripts.build_gold_v2.FeatureMatrixBuilderV2Live`, `features.elo_features.assert_no_provisional_training_rows`.
- Produces:
  - `@dataclass(frozen=True) class EarlyWeekResult: outcome: str; reason: str | None; season: int | None; week: int | None; games: int; bets: int; declined: int; appended_rows: tuple[dict[str, Any], ...]`
  - `OUTCOME_NOT_DUE = "not_due"`, `OUTCOME_SKIPPED = "skipped"`, `OUTCOME_DECIDED = "decided"`
  - `run_early_week(run_date_et: date, *, board_captured: bool) -> EarlyWeekResult`
  - `rehearse_early_week(week: int) -> None`
  - `main(argv: list[str] | None = None) -> int` for `python -m pipeline.early_week --rehearse-week N`
  - Module seams, replaced in tests: `REFRESH_SILVER`, `BUILD_GOLD`, `BUILD_INPUTS`, `SCORE`, `RECORD`, `NOW`

The order, each skip a named `reason` with no write:

1. `early_week_due(games, run_date_et)`; None is `not_due`.
2. The ledger already holds an `early` row for that week: `already_decided`.
3. Any game of a lower week lacks a score: `earlier_week_unscored`.
4. `board_captured` is False or `read_board_for_run` is None: `no_board`.
5. `in_force_recipe` is not `EARLY_RECIPE`: `recipe_not_v2`.
6. Refresh silver: `persist_current_season_elo(week_ids)`, `step_build_team_form()`, `build_weather_features_through(season, week_ids, {})`.
7. Call `assert_no_provisional_training_rows` on the lower weeks' game ids; this catches a played game still carrying a provisional Elo row.
8. Build matrices with `FeatureMatrixBuilderV2Live().generate_feature_matrices(excluded_game_ids=live_skip.excluded_games(), unplayed_through_lock=<latest lock of the week>)`. Set `wind_mph = 0.0` for that week's `venue_roof == "indoor"` games, and write the three parquet files to `outputs/early_gold/<season>_w<week>/` through `refuse_data_dir`.
9. Games whose lock is at or before the clock are left out and printed as `EARLY_LOCK_PASSED <game_id>`. `early_times` is the board's `captured_at` for the rest.
10. `build_early_inputs(gold_dir, games, load_schedules([season - 1, season]), early_times)`. Excluded games have no prediction.
11. `predict_v2` with each `load_model_artifact(target, version=resolved.model_id_for(target), artifacts_dir=...)["model"]`.
12. `decide_early_week`, `DecisionBundle(frame, candidates, schedule=<game_id, season, week, home_team, away_team, kickoff_et>, gold_inputs=inputs.rows, fits={}, resolved=resolved, chain_fit_path=Path(KEY_NUMBER_TABLES_PATH), run_instant=decided_at)`, then `record_early_week(..., publish_by=<earliest lock kept>, meta_extra={"early_arm_record_sha256": <sha256 of RECORD_PATH>}, repo_dir=Path())`.

Print `EARLY_BETS= <outcome> [reason=...] season=... week=... games=... bets=... declined=...`.

`rehearse_early_week(week)` runs steps 8-12 with no silver refresh and no ledger write, using `latest_board` and `outputs/early_gold_rehearsal/`, and prints one line per game and target: `consensus`, `model`, side, book, price, `ev`, status.

- [ ] **Step 1: Write the failing tests** (all heavy steps replaced through the seams)

```python
def test_not_due_writes_nothing(): ...           # outcome == "not_due"; REFRESH_SILVER never called
def test_a_decided_week_is_not_decided_twice(): ...  # reason == "already_decided"
def test_an_unscored_earlier_game_skips_the_week(): ...  # reason == "earlier_week_unscored"; RECORD never called
def test_no_board_skips_the_week(): ...          # reason == "no_board"
def test_old_models_skip_the_week(): ...         # reason == "recipe_not_v2"
def test_a_played_game_with_a_provisional_elo_row_stops_the_week(): ...

def test_an_indoor_game_keeps_zero_wind(tmp_path):
    written = pd.read_parquet(tmp_path / "features_ou.parquet").set_index("game_id")
    assert written.loc["indoor", "wind_mph"] == 0.0 and pd.isna(written.loc["outdoor", "wind_mph"])

def test_a_game_past_its_lock_is_left_out_and_the_rest_decided(capsys):
    result = run_early_week(TUESDAY, board_captured=True)
    assert "EARLY_LOCK_PASSED g1" in capsys.readouterr().out and result.games == 1

def test_a_neutral_site_game_is_declined_for_no_prediction(): ...
def test_the_steps_run_in_order(): ...  # refresh, build, inputs, score, record
def test_the_record_gets_the_record_files_sha(): ...
def test_the_rehearsal_writes_no_ledger_row(): ...  # RECORD and REFRESH_SILVER never called
```

- [ ] **Step 2: Run them and confirm they fail**, **Step 3: Implement**, **Step 4: Run them and confirm they pass**

Run: `uv run pytest tests/unit/test_early_week_run.py -q -p no:cacheprovider`

- [ ] **Step 5: Lint and commit** (`feat(pipeline): the Tuesday early-week run and its rehearsal`)

---

### Task 9: The daily run and its messages

**Files:**
- Modify: `scripts/daily_lock_pipeline.py:928-953,1083-1086`, `notify/report.py`, `notify/messages.py`, `notify/end_of_run.py`, `tests/conftest.py:585-599`, `docs/guides/AUTOMATION.md`
- Test: `tests/unit/test_daily_lock_run.py`, `tests/unit/test_notify_events.py`

**Interfaces:**
- Consumes: `run_early_week`.
- Produces: `RunReport.early_board_captured: bool = False`, `early_ran: bool = False`, `early_rows: list[dict[str, Any]]` (default empty), `early_error: str | None = None`; the seam `EARLY_BETS: Callable[[RunReport], object]`; `_early_bets_quietly(report)`; `notify.messages.early_bets_message(report) -> PushMessage | None`.

Wiring in `run_daily`, right after `_capture_early_lines_quietly(report)` under the same condition (`not dry_run and report.status != STATUS_OFFSEASON`). `_capture_early_lines_quietly` sets `report.early_board_captured` when the outcome is `captured`. `_early_bets_quietly` is non-fatal and catches `Exception` only (printing `EARLY_BETS= failed <Class>`, setting `report.early_error`). When rows were appended it calls `_sync_ledger(report=report)` and `_rebuild_web_cache(report, contract="WEB_CACHE_AFTER_EARLY=")`, so they are anchored and reach the site before the push. Add `EARLY_BETS` to the inert seams in `tests/conftest.py`.

The push title is `{n} early-week bet(s): {season} week {week}`, one item line per live row: `Totals: NE @ DET under 44.5 -105 at fanduel, 1.40 units`. Nothing is sent when there are no bets; print `NOTIFY_EARLY_BETS= none`. A failed step adds one normal-priority `FAILED: early-week bets` message to `failed_messages`, in the shape of the re-fit's at `notify/messages.py:673`. `AUTOMATION.md` gets a daily-run item 7 describing the step and its `EARLY_BETS=` line.

- [ ] **Step 1: Write the failing tests**

```python
# test_daily_lock_run.py, in the style of the existing EARLY_LINES wiring test
def test_the_early_step_runs_after_the_board_capture_on_a_no_games_day(): ...
def test_a_failed_early_step_never_changes_the_exit_code(): ...   # exit 0, "EARLY_BETS= failed RuntimeError"
def test_new_early_rows_are_synced_and_the_cache_rebuilt(): ...   # the sync runs twice; WEB_CACHE_AFTER_EARLY= printed
def test_a_dry_run_makes_no_early_decision(): ...
def test_a_skipped_board_capture_is_passed_on(): ...              # report.early_board_captured is False

# test_notify_events.py
def test_the_early_push_names_book_line_price_and_stake():
    message = early_bets_message(report_with_one_bet())
    assert message.title == "1 early-week bet: 2026 week 11"
    assert message.item_lines[0] == "Totals: NE @ DET under 44.5 -105 at fanduel, 1.40 units"

def test_a_declined_row_sends_no_early_push(): ...
def test_a_failed_early_step_sends_one_failed_push(): ...
```

- [ ] **Step 2: Run them and confirm they fail**, **Step 3: Implement**, **Step 4: Run them and confirm they pass**

Run: `uv run pytest tests/unit/test_daily_lock_run.py tests/unit/test_notify_events.py tests/unit/test_step_registry_order.py -q -p no:cacheprovider`

- [ ] **Step 5: Lint and commit** (`feat(run): the early-week side step and its pushes`)

---

### Task 10: The bets page

**Files:**
- Create: `backtest/early_arm_report.py`, `web/templates/components/_early_week_bets.html`, `tests/unit/test_early_arm_report.py`
- Modify: `forward_ledger/cache_sources.py`, `api/cache_schema.py`, `api/cache.py:1916,3104`, `pipeline/steps.py:1358`, `scripts/populate_cache.py:128`, `api/services.py`, `api/routes/pages.py:653-723`, `web/templates/pages/bets.html:185-195`
- Test: `tests/api/test_bets_page.py`

**Interfaces:**
- Produces:
  - `early_arm_summary(bet_list: pd.DataFrame, corrections: pd.DataFrame | None) -> dict[str, Any]` with keys `bets, graded, wins, losses, pushes, flat_return, closing_bets, mean_clv, closing_share`. A figure with nothing behind it is `None`, never 0.0.
  - `BetCacheSources.early_arm_summary: dict[str, Any]`
  - `api.cache_schema.EARLY_ARM_SUMMARY_KEY = "early_arm_summary"`
  - `api.cache.stamp_early_arm_summary(conn, summary, now)` (one JSON value in `cache_meta`; no new table, no `DATA_FORMAT_VERSION` bump)
  - `populate_cache(..., early_arm_summary: dict[str, Any] | None = None)`
  - `DataService.get_early_arm_summary() -> dict[str, Any] | None`
  - Bets-page context: `early_arm_open` (`(season, week) >= EARLY_ARM_FROM`), `early_bets` (`get_bet_list(season, week, arm=ARM_EARLY) or []`), `early_declined` (`len(get_suppressed_bets(season, week, arm=ARM_EARLY))`), `early_summary`.

The section renders under the retired message when `early_arm_open` is true. Fixed copy:
- Heading: `Early-week bets (paper)`
- Caption: `Placed on the Tuesday before the games, at the best price among DraftKings, FanDuel, BetMGM and BetRivers when the line was read. Paper only.`
- No bets, week decided: `No early-week bets this week. {early_declined} bets were checked on Tuesday and none cleared the sportsbook's cut.`
- Week not decided yet: `Not decided yet. These bets are decided by the Tuesday run.`
- Record row: closing-line value first (`mean_clv` in points), then flat return, record and closing share. `None` renders as `--`.

Each bet row reuses `bp.pick_label` and shows the book name, line, price, EV % and stake units. The book name comes from `eligibility_label` with the prefix removed.

- [ ] **Step 1: Write the failing tests**

```python
# test_early_arm_report.py
def test_the_summary_counts_only_early_live_rows(): ...
def test_closing_value_is_the_mean_over_rows_that_have_one():
    assert summary["mean_clv"] == pytest.approx(1.0) and summary["closing_share"] == pytest.approx(0.5)
def test_nothing_graded_is_none_not_zero():
    assert summary["flat_return"] is None and summary["mean_clv"] is None
def test_a_correction_changes_the_record(): ...

# tests/api/test_bets_page.py
def test_the_early_section_shows_book_line_price_and_stake(): ...
def test_a_week_with_no_early_bets_says_how_many_were_checked(): ...
def test_week_10_shows_no_early_section(): ...
def test_early_rows_stay_out_of_the_retired_tracker(): ...
def test_the_record_shows_closing_value_beside_the_return(): ...
```

- [ ] **Step 2: Run them and confirm they fail**, **Step 3: Implement**, **Step 4: Run them, plus the cache guards**

Run: `uv run pytest tests/unit/test_early_arm_report.py tests/api/test_bets_page.py tests/api/test_this_week_headliner.py tests/api/test_cache_betting.py tests/unit/test_cache_data_format.py -q -p no:cacheprovider`
Expected: all pass.

- [ ] **Step 5: Lint and commit** (`feat(web): early-week bets on /bets`)

---

### Task 11: Land it and see it run

**Files:** none new. Update `.planning/designs/2026-11-05-RESUME.md` (gitignored, never committed).

- [ ] **Step 1: Whole-branch review.** One reviewer on Opus over `master..early-week`, briefed with this plan and the design. Fix what it finds, each fix with a test.

- [ ] **Step 2: Run the five tripwires and the bounded gate** with the command in `docs/superpowers/plans/2026-11-06-switch-piece-3.md` Task 12 Step 4.
Expected: each tripwire fails by AssertionError, and the gate has no failures.

- [ ] **Step 3: Wait for the Wed 2026-11-11 17:00 ET run**, then confirm the resume file's check (2) passed. If it failed, fixing it comes first.

- [ ] **Step 4: Rehearse on real data, outside 17:00-17:20 ET**

Run, from the worktree: `uv run python -m pipeline.early_week --rehearse-week 10`
Expected: one line per week-10 game and target, no ledger change. Check that lines match a sportsbook, `over`/`under` and home/away are the right way round, and the bets count is in the range of 0-3. A gate refusal naming a game is a stop: fix it before merging.

- [ ] **Step 5: With the owner's go-ahead, merge** before Tue 2026-11-17 17:00 ET and outside 17:00-17:20 ET. Fast-forward the main checkout.

- [ ] **Step 6: After the Tue 2026-11-17 17:00 ET run**

Read the run's console log.
Expected: `EARLY_LINES= captured`, `EARLY_BETS= decided season=2026 week=11 games=<kept> bets=<n> declined=<rest>`, no failed step. `uv run python -m scripts.verify_ledger` reports no failure. /bets for week 11 shows the section. Week 11 has 13 games, so 26 rows, including `2026_W11_MIN@SF` declined as `missing_prediction`.

- [ ] **Step 7: Update the resume file:** piece 4 live, first-week outcome, and what to check at the first closing readings.
