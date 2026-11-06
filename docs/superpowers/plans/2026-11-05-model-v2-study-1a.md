# Model v2 and Study 1a Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a repaired "v2" feature set into a separate gold directory, train simple v2 models walk-forward, and produce the Study 1a report that decides (a) whether v2 replaces the live models at the switch and (b) how much weight each target's model earns against the market.

**Architecture:** Purely additive. The v2 feature pieces are subclasses and new builders in new modules; a new script builds gold into `outputs/gold_v2/` by calling the live builder's `generate_feature_matrices` (which writes nothing) and writing parquet itself. New trainer, statistics and study modules read that gold, reuse the re-fit's fold and corpus helpers read-only, and write a report to `outputs/study_1a/`. No existing tracked file is edited, so the live 17:00 ET run builds and serves exactly what it does today.

**Tech Stack:** Python 3.13, uv, pandas, numpy, scipy, scikit-learn (Pipeline, SimpleImputer, StandardScaler, LogisticRegression, Ridge), pytest.

**Spec:** `C:\Users\jackc\code\nfl-predict\.planning\designs\2026-11-05-profitability-fixes-design.md` (sections 5, 6.2, 7). Evidence: `C:\Users\jackc\code\nfl-predict\.planning\research\2026-11-05-why-not-profitable\` -- `code-map-model-v2.md` (exact signatures and traps, cited below as "map Tn" / "map s.n"), `audit-model-signal.md`, `scratch-scripts\wf_score.py` (the audit's regression code). These files are gitignored: a worktree does not contain them, so read them at these absolute paths.

**Scope of this plan.** This is plan 1 of 4 from the design. Out of scope here: the key-number cover-probability converter, pricing, the bet list and its records (plan 3, the switch); the early-week study (plan 2) and arm (plan 4). Two deliberate narrowings of design 6.2, stated for the owner: (1) no starter-changed flag -- the map measured that 2002-2024 depth charts catch 25-43% of starter changes against 55% live (map T12), so a flag trained on history would misfire live; the backup problem is handled instead by a below-average replacement value for unknown QBs. (2) The input sets are fixed in this plan before any replay runs (Task 6), instead of being chosen on the replay, so the replay cannot be overfitted by input selection.

## Global Constraints

- Work in the git worktree on branch `model-v2` at `C:\Users\jackc\code\nfl-predict-v2` (set up already). `data/` there is the `data` CODE package (tracked .py files); the live data lake is reached through directory junctions `data/bronze`, `data/silver`, `data/gold`, `data/baselines`, `data/optuna` pointing into `C:\Users\jackc\code\nfl-predict\data\`, and the worktree's gitignored `.env` sets `DUCKDB_PATH` to the live `nfl_predictions.duckdb`. Because of the junctions, anything written under those folders lands in the LIVE lake: nothing in this plan may write under `data/`.
- Additive only: do not edit any existing tracked file. New modules go under existing top-level folders (`features/`, `models/`, `backtest/`, `scripts/`, `tests/unit/`); no new top-level package (map 8.2.1).
- Never call `save_dataframe`, `save_feature_matrices`, `save_model_artifact`, `BaseTrainer.save`, `MarketBlender.save_blend_artifacts`, `train_generation(save=True)`, or anything else in map s.6.5's "must not be called" list. Never write `artifacts/` or `artifacts/latest.json`. v2 outputs go to `outputs/gold_v2/` and `outputs/study_1a/` with plain `to_parquet` / `write_text`.
- Do not run the v2 build or the study between 17:00 and 17:20 ET (it can block the live run's DuckDB writer, map T2), and not while a pytest session runs.
- Tests: run only the named files or node ids (`uv run pytest <file>::<test> -p no:cacheprovider`). Never bare `pytest` or a directory. No `xfail`, no skip. Tests write only under `tmp_path`.
- Lint every touched file: `uv run ruff check --no-fix --no-cache <files>`, `uv run ruff format <files>`, `uv run pyright <files>` (0 errors).
- Naming: no column name containing `margin`, `closing`, `result`, `final`, `actual`, `outcome`, `winner`, `loser`, `total_score`, or ending in a market/line-movement suffix (map T4); no name containing `FIRST_SEASON` bound to a year; no dict key or frame column named `arm`, `recipe_id` or the other ledger stamp names (map 8.2.3).
- Import the lock rule as `import utils.game_lock as lock_rule` (never `from utils.game_lock import ...` at module level). Read nflverse data only through `data.upstream_pin` (map s.5).
- ASCII only in code, tests and reports; no emoji.
- Commit after each task on `model-v2`, ending the message with:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` and
  `Claude-Session: https://claude.ai/code/session_01LVgorLUwwGjKecmY3yVBge`.

## Review Focus

1. **Unplayed 2026 games carry a moving line, not a close.** nflverse `spread_line` for a game not yet played is today's line. Expected: a market rating uses only games that kicked off at or before the target's lock. Pinned in Task 3.
2. **Neutral-site games** (London, Germany, Mexico, playoffs). Expected: no home-field term in the rating fit for them; the target's own neutral flag does not shift its rating difference. Pinned in Task 3.
3. **Relocated or renamed teams** (OAK->LV, SD->LAC, STL->LA). Expected: a team keeps its prior-season rating across the code change because ids pass through the canonical mapping. Pinned in Task 3.
4. **Missing inputs** (wind is NaN on 28% of 2026 rows; week-1 ratings can be blank). Expected: every v2 prediction is finite. Pinned in Task 6.
5. **A team with no history at all** (Houston in 2002). Expected: blank rating and coverage 0.0, never a crash or a silent 0.0. Pinned in Task 3.

---

### Task 1: Windowed opponent-adjusted EPA

**Files:**
- Create: `features/opponent_adj_windowed.py`
- Test: `tests/unit/test_opponent_adj_windowed.py`

**Interfaces:**
- Consumes: `features.opponent_adj.OpponentAdjuster` (`opponent_adj.py:143`), its `_rolling_for_targets(self, adjusted, targets) -> pd.DataFrame` (`:531-593`) and the frames `per_game_adjusted` (`:464-525`) and `_target_team_games` (`:595-614`) produce.
- Produces: `class WindowedOpponentAdjuster(OpponentAdjuster)` with constants `DECAY = 0.9`, `PRIOR_GAMES = 3.0`, `PRIOR_REGRESSION = 0.5`. Same constructor, same output schema as the parent's `_rolling_for_targets` (`team, side, target_season, target_week, games_used, rolling_opp_adj_epa_per_play, rolling_opp_adj_pass_epa, rolling_opp_adj_rush_epa, rolling_opp_adj_coverage, latest_information_end`).

The rule, per (team, side, target) and per metric: current-season adjusted games that ENDED at or before the target's lock get weights `DECAY ** k` (k = 0 for the most recent). The prior is `PRIOR_REGRESSION *` the team's previous-season value (the same decayed mean over all of last season's adjusted games). Value = `(sum(w*x) + PRIOR_GAMES * prior) / (sum(w) + PRIOR_GAMES)`; with no current-season game the value is the prior; with neither, NaN and coverage 0.0. Coverage is 1.0 whenever a value is produced. Seasons older than the previous one are never read. `latest_information_end` = max `information_end` over the rows read. Keep the parent's "no history at all -> no row" behaviour.

- [ ] **Step 1: Write the failing tests**

Build `adjusted` and `targets` with exactly the columns the parent methods produce (read `opponent_adj.py:464-614`); one team, side `offense`, all three metrics equal per row.

```python
def test_decayed_current_season_blended_with_regressed_prior():
    # prior season (2023): adjusted values 0.2, 0.2 -> prior 0.5 * 0.2 = 0.1
    # 2024: 0.4 (older), 0.6 (newer), both ended before the lock -> weights 0.9, 1.0
    out = WindowedOpponentAdjuster()._rolling_for_targets(adjusted, targets)
    row = out.loc[(out["team"] == "KC") & (out["side"] == "offense")].iloc[0]
    assert row["rolling_opp_adj_epa_per_play"] == pytest.approx((0.9*0.4 + 1.0*0.6 + 3.0*0.1) / (1.9 + 3.0))
    assert row["rolling_opp_adj_coverage"] == 1.0

def test_games_ending_after_the_lock_and_older_seasons_are_ignored():
    # add a 2024 game ending one second after the lock (value 9.0) and a 2022 game (value -9.0)
    out = WindowedOpponentAdjuster()._rolling_for_targets(adjusted_plus_noise, targets)
    assert out["rolling_opp_adj_epa_per_play"].iloc[0] == pytest.approx(0.25714, abs=1e-4)

def test_week_one_uses_the_regressed_prior_only():
    out = WindowedOpponentAdjuster()._rolling_for_targets(prior_season_only, week1_targets)
    assert out["rolling_opp_adj_epa_per_play"].iloc[0] == pytest.approx(0.1)

def test_history_but_no_adjusted_game_gives_nan_and_flag_zero():
    out = WindowedOpponentAdjuster()._rolling_for_targets(unadjusted_only, targets)
    assert np.isnan(out["rolling_opp_adj_epa_per_play"].iloc[0])
    assert out["rolling_opp_adj_coverage"].iloc[0] == 0.0
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_opponent_adj_windowed.py -p no:cacheprovider`
Expected: FAIL (`ModuleNotFoundError: features.opponent_adj_windowed`).

- [ ] **Step 3: Implement `WindowedOpponentAdjuster._rolling_for_targets(self, adjusted: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame`** in `features/opponent_adj_windowed.py`, overriding only that method. Do not use `.split("_")` on ids.

- [ ] **Step 4: Run the new tests and the parent's tests**

Run: `uv run pytest tests/unit/test_opponent_adj_windowed.py tests/unit/test_opponent_adjustment.py tests/unit/test_opponent_adjustment_runs.py -p no:cacheprovider`
Expected: all PASS (the parent is untouched).

- [ ] **Step 5: Lint and commit**

```bash
git add features/opponent_adj_windowed.py tests/unit/test_opponent_adj_windowed.py
git commit -m "feat(v2): windowed opponent-adjusted EPA with a regressed prior"
```

---

### Task 2: Replacement-level value for unknown quarterbacks

**Files:**
- Create: `features/qb_v2.py`
- Test: `tests/unit/test_qb_v2.py`

**Interfaces:**
- Consumes: `features.qb_tracking.QBTracker` (`qb_tracking.py:177`), its staticmethod `_quality_of(rolling, qb_id) -> float` (`:1026-1034`, called as `self._quality_of` at `:988`) and `no_information_signature(self)` (`:826-833`); `features.injury.REPLACEMENT_LEVEL_QB_QUALITY` (= -0.5).
- Produces: `class QBTrackerV2(QBTracker)` where `_quality_of` returns `REPLACEMENT_LEVEL_QB_QUALITY` instead of 0.0 for no id, no rolling rows, or a QB missing from the window, and `no_information_signature()` returns `{"qb_adjustment": REPLACEMENT_LEVEL_QB_QUALITY}`.

- [ ] **Step 1: Write the failing tests**

```python
def test_unknown_qb_is_below_average_not_league_average():
    rolling = pd.DataFrame({"passer_player_id": ["00-1"], "qb_quality": [1.2]})
    assert QBTrackerV2._quality_of(rolling, None) == -0.5
    assert QBTrackerV2._quality_of(rolling.iloc[0:0], "00-1") == -0.5
    assert QBTrackerV2._quality_of(rolling, "00-9") == -0.5

def test_known_qb_keeps_his_rolling_quality():
    rolling = pd.DataFrame({"passer_player_id": ["00-1"], "qb_quality": [1.2]})
    assert QBTrackerV2._quality_of(rolling, "00-1") == pytest.approx(1.2)

def test_no_information_value_matches_the_replacement_level():
    assert QBTrackerV2().no_information_signature() == {"qb_adjustment": -0.5}
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_qb_v2.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `QBTrackerV2`** in `features/qb_v2.py` (staticmethod override plus the signature override; nothing else).

- [ ] **Step 4: Run the new tests and the parent's**

Run: `uv run pytest tests/unit/test_qb_v2.py tests/unit/test_qb_tracking.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add features/qb_v2.py tests/unit/test_qb_v2.py
git commit -m "feat(v2): unknown quarterbacks get a replacement-level value"
```

---

### Task 3: Market-implied team ratings from past games' lines

**Files:**
- Create: `features/market_rating.py`
- Test: `tests/unit/test_market_rating.py`

**Interfaces:**
- Consumes: `data.upstream_pin.load_schedules(seasons) -> pd.DataFrame` (nflverse: `season, week, home_team, away_team, gameday, gametime, spread_line` home-positive, `total_line`); silver `games` (`game_id, season, week, kickoff_et, home_team, away_team, neutral_site`); `utils.game_id_utils.create_standard_game_id(season, week, away_team, home_team) -> str`; `import utils.game_lock as lock_rule` (`lock_rule.lock_frame(games_df) -> pd.Series`).
- Produces:
  - `RATING_PRIOR_SEASON_WEIGHT = 0.35`, `RATING_RIDGE = 1.0`
  - `fit_spread_ratings(lines: pd.DataFrame) -> tuple[dict[str, float], float]` -- weighted ridge least squares of `spread_line = hfa * (1 - neutral) + r[home] - r[away]` over rows with columns `home_team, away_team, spread_line, neutral, weight`; returns (ratings, hfa); ridge pulls ratings toward 0, not hfa.
  - `fit_total_levels(lines: pd.DataFrame) -> tuple[dict[str, float], float]` -- same weights, `total_line = mu + t[home] + t[away]`; returns (levels, mu).
  - `market_ratings(games: pd.DataFrame, nflverse: pd.DataFrame) -> pd.DataFrame` with columns `game_id, home_mkt_rating, away_mkt_rating, mkt_rating_diff, mkt_total_sum, mkt_rating_coverage, information_time` (one row per silver game). For target G with lock L: read lines of games P with season in {G.season - 1, G.season}, `kickoff_P <= L`, weight 1.0 for current season and `RATING_PRIOR_SEASON_WEIGHT` for the previous one; solve once per distinct lock. `mkt_rating_diff = home - away` (no hfa); `mkt_total_sum = mu + t[home] + t[away]`; `information_time` = max kickoff read; a team with no line read gets NaN and `mkt_rating_coverage` 0.0 (1.0 when both teams are rated).
  - `market_rating_no_information_signature() -> dict[str, float | None]` (NaN-valued columns None, coverage 0.0), for Task 5's separate gate.

- [ ] **Step 1: Write the failing tests**

```python
def test_ratings_recover_known_strengths_and_home_field():
    # 4 teams with true ratings A=+3, B=+1, C=-1, D=-3, hfa=2.0; every pairing home and away
    ratings, hfa = fit_spread_ratings(lines_from_truth)
    assert ratings["A"] - ratings["D"] == pytest.approx(6.0, abs=0.15)
    assert hfa == pytest.approx(2.0, abs=0.15)

def test_neutral_site_games_carry_no_home_field():
    ratings, hfa = fit_spread_ratings(lines_with_neutral_rows)   # neutral rows built without hfa
    assert hfa == pytest.approx(2.0, abs=0.15)

def test_only_lines_kicked_off_by_the_lock_are_read():
    base = market_ratings(games, nflverse)
    moved = market_ratings(games, nflverse_with_later_game_line_set_to_50)  # a game kicking off 1 s after G's lock
    assert base.set_index("game_id").loc[G, "mkt_rating_diff"] == moved.set_index("game_id").loc[G, "mkt_rating_diff"]
    assert base.set_index("game_id").loc[G, "information_time"] <= lock_of_G

def test_relocated_team_keeps_its_prior_season_rating():
    # previous season under SD, current season under LAC (canonical mapping makes them one team)
    out = market_ratings(games_lac_week1, nflverse_sd_then_lac).set_index("game_id")
    assert not np.isnan(out.loc[LAC_WEEK1_GAME, "home_mkt_rating"])

def test_team_with_no_history_is_blank_with_coverage_zero():
    out = market_ratings(games_houston_2002_week1, nflverse_2001_without_hou).set_index("game_id")
    assert np.isnan(out.loc[HOU_GAME, "home_mkt_rating"])
    assert out.loc[HOU_GAME, "mkt_rating_coverage"] == 0.0
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_market_rating.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the four functions** in `features/market_rating.py` with `numpy.linalg.lstsq` on the weighted, ridge-augmented system. Map nflverse rows to silver ids through `create_standard_game_id`; kickoffs come from silver `games.kickoff_et`, never from nflverse `gameday`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_market_rating.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add features/market_rating.py tests/unit/test_market_rating.py
git commit -m "feat(v2): market-implied team ratings from past games' lines"
```

---

### Task 4: League scoring level

**Files:**
- Create: `features/league_level.py`
- Test: `tests/unit/test_league_level.py`

**Interfaces:**
- Consumes: silver `games` (`game_id, season, kickoff_et, home_score, away_score`); `features.provenance.DECLARED_GAME_DURATION` (4 h); `import utils.game_lock as lock_rule`.
- Produces: `LEVEL_PRIOR_GAMES = 32.0`; `league_points_level(games: pd.DataFrame) -> pd.DataFrame` with columns `game_id, league_points_level, information_time`. For target G with lock L: current-season games that ENDED (kickoff + 4 h) at or before L contribute their home+away points; the prior is the mean of the whole previous season. Level = `(sum(points) + LEVEL_PRIOR_GAMES * prior) / (n + LEVEL_PRIOR_GAMES)`; with no previous season, the current-season mean (NaN when n = 0). `information_time` = latest end read (NaN-time when nothing read).

- [ ] **Step 1: Write the failing tests**

```python
def test_level_blends_current_season_with_the_prior_season_mean():
    # prior season mean 44.0; current season ended games: 50, 40 (both ended before L)
    out = league_points_level(games).set_index("game_id")
    assert out.loc[G, "league_points_level"] == pytest.approx((90 + 32 * 44.0) / (2 + 32))

def test_a_game_kicked_off_but_not_ended_by_the_lock_is_not_read():
    # a 16:25 ET Sunday game vs a Monday target locked Sunday 18:00: ends 20:25, after the lock
    out = league_points_level(games_with_late_sunday_game).set_index("game_id")
    assert out.loc[MONDAY_GAME, "league_points_level"] == pytest.approx((90 + 32 * 44.0) / 34)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_league_level.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `league_points_level`** in `features/league_level.py`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_league_level.py -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add features/league_level.py tests/unit/test_league_level.py
git commit -m "feat(v2): league scoring level fenced on game end"
```

---

### Task 5: The v2 gold build

**Files:**
- Create: `scripts/build_gold_v2.py`
- Test: `tests/unit/test_build_gold_v2.py`

**Interfaces:**
- Consumes: `scripts.build_features.FeatureMatrixBuilder` (`:699`), its `generate_feature_matrices(...) -> dict[str, pd.DataFrame]` (`:3310`, writes nothing), `normalize_combined_features(self, processed_features, target_season=None)` (`:2606`), `_merge_opponent_adjusted(self, combined, games, lock_frame, as_of_datetime, target_season, target_week)` (`:1953`, drops the six raw EPA columns at `:2055-2074`); Tasks 1-4; `features.provenance.InformationTimeGate` (`check(...)` `:403`), `build_lock_frame` (`:348`); `features.validation.LeakageGate.validate_combined_matrix(df, as_of)` (`:156`); `data.write_sink.active_sink`, `RecordingSink`.
- Produces:
  - `class FeatureMatrixBuilderV2(FeatureMatrixBuilder)`: after `super().__init__()` sets `self.opponent_adj = WindowedOpponentAdjuster(window=10, min_opponent_games=4)` and `self.qb_tracker = QBTrackerV2()`; `normalize_combined_features` returns its input unchanged (same signature); `_merge_opponent_adjusted` keeps the six `{home,away}_{off,def}_rolling_{epa_per_play,pass_epa_per_play,rush_epa_per_play}` columns (stash, call super, merge back on `game_id`).
  - `V2_DERIVED_COLUMNS = ("success_rate_net", "opp_adj_epa_net", "qb_value_diff", "off_epa_sum", "def_epa_sum", "pace_sum")`
  - `add_v2_derived_columns(frame: pd.DataFrame) -> pd.DataFrame`:
    - `success_rate_net = (home_off_rolling_success_rate - home_def_rolling_success_rate) - (away_off_rolling_success_rate - away_def_rolling_success_rate)`
    - `opp_adj_epa_net = (home_off_rolling_opp_adj_epa_per_play - home_def_rolling_opp_adj_epa_per_play) - (away_off_rolling_opp_adj_epa_per_play - away_def_rolling_opp_adj_epa_per_play)`
    - `qb_value_diff = home_qb_adjustment - away_qb_adjustment`
    - `off_epa_sum = home_off_rolling_opp_adj_epa_per_play + away_off_rolling_opp_adj_epa_per_play`
    - `def_epa_sum = home_def_rolling_opp_adj_epa_per_play + away_def_rolling_opp_adj_epa_per_play`
    - `pace_sum = home_off_rolling_neutral_pace + away_off_rolling_neutral_pace`
  - `add_v2_families(matrices: dict[str, pd.DataFrame], games: pd.DataFrame, nflverse: pd.DataFrame) -> dict[str, pd.DataFrame]`: merges `market_ratings` and `league_points_level` on `game_id` into all three matrices after checking each with its OWN `InformationTimeGate()` against `build_lock_frame(games)` (never the builder's coverage report, map T3), then applies `add_v2_derived_columns`. The `information_time` columns are not merged into gold.
  - `build_gold_v2(out_dir: Path) -> dict[str, pd.DataFrame]`: runs `FeatureMatrixBuilderV2().generate_feature_matrices()` inside `active_sink(RecordingSink())` and refuses (raises `RuntimeError`) if the sink recorded any intended write; adds the families; runs `LeakageGate().validate_combined_matrix` on each matrix; writes `out_dir/features_{wp,ats,ou}.parquet` with `to_parquet` and `out_dir/manifest.json` (`built_at_utc`, `git_commit`, rows and columns per matrix). Never `save_dataframe` (map T1); `out_dir` must not be under `data/` (raise `ValueError`).
  - `main(argv: list[str] | None = None) -> int` with `--out-dir` (default `outputs/gold_v2`).

- [ ] **Step 1: Write the failing tests**

```python
def test_derived_columns_have_the_stated_values():
    out = add_v2_derived_columns(one_row_frame)   # inputs chosen so each result is a distinct round number
    assert out.loc[0, "success_rate_net"] == pytest.approx(0.10)
    assert out.loc[0, "qb_value_diff"] == pytest.approx(1.5)
    assert out.loc[0, "pace_sum"] == pytest.approx(130.0)

def test_v2_builder_swaps_in_the_repaired_pieces_and_skips_normalization():
    b = FeatureMatrixBuilderV2()
    assert isinstance(b.opponent_adj, WindowedOpponentAdjuster)
    assert isinstance(b.qb_tracker, QBTrackerV2)
    frame = pd.DataFrame({"elo_diff": [87.0]})
    assert b.normalize_combined_features(frame).equals(frame)

def test_build_writes_only_three_parquet_files_and_a_manifest(tmp_path, monkeypatch):
    # monkeypatch generate_feature_matrices, market_ratings, league_points_level and the
    # silver/nflverse loads to return tiny frames
    build_gold_v2(tmp_path / "gold_v2")
    assert sorted(p.name for p in (tmp_path / "gold_v2").iterdir()) == [
        "features_ats.parquet", "features_ou.parquet", "features_wp.parquet", "manifest.json"]

def test_build_refuses_an_output_directory_under_data():
    with pytest.raises(ValueError):
        build_gold_v2(Path("data/gold_v2"))
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_build_gold_v2.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement the module.** Load silver `games` with `data.storage.load_dataframe("games", layer="silver")` and nflverse with `data.upstream_pin.load_schedules(list(range(2001, 2027)))`.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_build_gold_v2.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Run the real build (about 20 minutes; not 17:00-17:20 ET)**

Run: `uv run python -m scripts.build_gold_v2`
Expected: exit 0; `outputs/gold_v2/` holds the three parquet files and `manifest.json`; each matrix has the same `game_id` set as `data/gold/features_{t}.parquet` from the same day; every `V2_DERIVED_COLUMNS` name plus `mkt_rating_diff`, `mkt_total_sum`, `league_points_level` is present; `elo_diff` standard deviation is above 50 (raw Elo points, not z-scores); the modification times of `data/gold/features_{wp,ats,ou}.parquet` and `data/nfl_predictions.duckdb` are the same as before the run (record them first).

- [ ] **Step 6: Lint and commit**

```bash
git add scripts/build_gold_v2.py tests/unit/test_build_gold_v2.py
git commit -m "feat(v2): build repaired gold into outputs/gold_v2"
```

---

### Task 6: The v2 recipe and walk-forward trainer

**Files:**
- Create: `models/v2_recipe.py`, `models/v2_trainer.py`
- Test: `tests/unit/test_v2_trainer.py`

**Interfaces:**
- Consumes: `refit.remeasure.fold_specs(games, generation_cutoff) -> tuple[FoldSpec, ...]` (`FoldSpec.label`, `.cutoff_utc`, `.predict_game_ids`); `refit.training.eligible_rows(gold, games, cutoff_utc, target) -> EligibleRows` (`.frame`, sorted, labelled).
- Produces:
  - `models/v2_recipe.py`: `V2_RECIPE_NAME = "recipe-2026-v2"`; `V2_TRAIN_START = 2002`; `V2_LABEL = {"wp": "home_win", "ats": "home_margin", "ou": "total_points"}`; and the fixed inputs:
    - `V2_INPUTS["wp"] = V2_INPUTS["ats"] = ("elo_diff", "success_rate_net", "opp_adj_epa_net", "qb_value_diff", "mkt_rating_diff", "hfa_used")`
    - `V2_INPUTS["ou"] = ("league_points_level", "mkt_total_sum", "off_epa_sum", "def_epa_sum", "pace_sum", "wind_mph")`
    - Estimators: WP `LogisticRegression(C=1.0, max_iter=1000)`; ATS and O/U `Ridge(alpha=1.0)`; each in `Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()), ("model", ...)])`. No extra calibrator.
  - `models/v2_trainer.py`:
    - `fit_v2(target: str, train: pd.DataFrame) -> Pipeline`
    - `predict_v2(target: str, model: Pipeline, rows: pd.DataFrame) -> np.ndarray` (WP: probability of a home win; ATS: home margin; O/U: total)
    - `walk_forward_v2(gold_dir: Path, games: pd.DataFrame, generation_cutoff: datetime) -> dict[str, pd.DataFrame]` returning per target the same columns as `refit.remeasure.walk_forward_predictions`: `game_id, season, week, prediction, actual, fold, fold_cutoff_utc`. Per fold, train on `eligible_rows(..., fold.cutoff_utc, target).frame` restricted to seasons >= `V2_TRAIN_START`, predict the fold's `predict_game_ids` present in gold.

- [ ] **Step 1: Write the failing tests**

```python
def test_fit_recovers_the_direction_of_a_planted_signal():
    model = fit_v2("ats", synthetic_rows)            # home_margin = 2.0 * elo_diff/25 + noise
    hi, lo = predict_v2("ats", model, rows_with_elo_diff([100.0, -100.0]))
    assert hi > lo

def test_missing_inputs_still_give_finite_predictions():
    model = fit_v2("ou", synthetic_ou_rows)
    rows = synthetic_ou_rows.head(3).assign(wind_mph=np.nan, mkt_total_sum=np.nan)
    assert np.isfinite(predict_v2("ou", model, rows)).all()

def test_wp_predictions_are_probabilities():
    p = predict_v2("wp", fit_v2("wp", synthetic_rows), synthetic_rows)
    assert ((p > 0) & (p < 1)).all()

def test_walk_forward_never_trains_on_a_game_it_predicts(tmp_path):
    # tiny gold dir (three seasons) + games; record the training game ids per fold via a spy on fit_v2
    out = walk_forward_v2(gold_dir, games, generation_cutoff)
    for fold, trained_ids in spy.items():
        predicted = set(out["ats"].loc[out["ats"]["fold"] == fold, "game_id"])
        assert predicted.isdisjoint(trained_ids)
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_v2_trainer.py -p no:cacheprovider`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement `models/v2_recipe.py` and `models/v2_trainer.py`.**

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_v2_trainer.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add models/v2_recipe.py models/v2_trainer.py tests/unit/test_v2_trainer.py
git commit -m "feat(v2): fixed-input linear recipe and walk-forward trainer"
```

---

### Task 7: Does the model know anything the line does not?

**Files:**
- Create: `backtest/market_info.py`
- Test: `tests/unit/test_market_info.py`

**Interfaces:**
- Consumes: numpy, `scipy.stats.norm`. Reference implementation of the same regressions: `scratch-scripts\wf_score.py:60-80, 112, 134` (absolute path in the Spec line).
- Produces:
  - `@dataclass(frozen=True) class InfoResult: coef: float; se: float; p_one_sided: float; n: int`
  - `information_beyond_line(outcome: np.ndarray, line: np.ndarray, model: np.ndarray) -> InfoResult` -- OLS of `outcome - line` on an intercept and `model - line`; `coef` is the slope; `p_one_sided = 1 - norm.cdf(coef / se)`.
  - `information_beyond_probability(home_win: np.ndarray, market_prob: np.ndarray, model_prob: np.ndarray) -> InfoResult` -- logistic regression with offset `logit(market_prob)`, an intercept, and slope on `logit(model_prob) - logit(market_prob)`, fitted by Newton-Raphson (probabilities clipped to [1e-6, 1 - 1e-6]); same p rule.

- [ ] **Step 1: Write the failing tests**

```python
def test_line_regression_finds_a_planted_slope():
    rng = np.random.default_rng(7)
    line = rng.normal(0, 6, 4000); edge = rng.normal(0, 2, 4000)
    outcome = line + 0.3 * edge + rng.normal(0, 13, 4000)
    r = information_beyond_line(outcome, line, line + edge)
    assert r.coef == pytest.approx(0.3, abs=0.12) and r.p_one_sided < 0.05 and r.n == 4000

def test_line_regression_sees_nothing_in_noise():
    rng = np.random.default_rng(8)
    line = rng.normal(0, 6, 4000)
    outcome = line + rng.normal(0, 13, 4000)
    r = information_beyond_line(outcome, line, line + rng.normal(0, 2, 4000))
    assert r.p_one_sided > 0.05

def test_probability_regression_finds_a_planted_slope():
    # market logit ~ N(0, 1); model logit = market logit + d; true logit = market + 0.5 d
    r = information_beyond_probability(home_win, market_prob, model_prob)
    assert r.coef == pytest.approx(0.5, abs=0.2) and r.p_one_sided < 0.05
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_market_info.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement both functions** in `backtest/market_info.py`. Do not import `backtest.bet_selector` (map 8.2.5).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_market_info.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Lint and commit**

```bash
git add backtest/market_info.py tests/unit/test_market_info.py
git commit -m "feat(v2): information-beyond-market regressions"
```

---

### Task 8: Study 1a -- the replay and its verdict

**Files:**
- Create: `backtest/study_1a.py`, `scripts/run_study_1a.py`
- Test: `tests/unit/test_study_1a.py`

**Interfaces:**
- Consumes: Task 6 `walk_forward_v2`; `refit.remeasure.walk_forward_predictions(gold_dir=Path("data/gold"), games, generation_cutoff)` for the current-recipe arm (LIVE gold only, map T9); `refit.remeasure.extended_corpus(silver_dir, games, generation_cutoff)` and `extended_slopes` (map s.6.4); `models.blending_data.build_tuning_frames(corpus_frame, predictions, walk_forward_slopes) -> BlendTuningFrames` (`.frames[t]` with `BLEND_TUNING_COLUMNS`: wp `model_prob, market_prob_oof, home_win`; ats `model_spread, market_spread, home_margin`; ou `model_total, market_total, total_points`); `models.blending.MarketBlender().tune_weights(frames) -> TuningResult` (`.weights[t]`); `backtest.metrics.compute_wp_metrics` / `compute_ats_metrics` / `compute_ou_metrics`; Task 7; `data.upstream_pin.load_schedules` for the nflverse close (report only; never pass close columns into market-probability fits, map T7).
- Produces:
  - `INSTALL_SEASONS = tuple(range(2017, 2025))`; `WEIGHT_ALPHA = 0.05`
  - `install_verdict(current: dict[str, float], v2: dict[str, float]) -> bool` -- True only if `v2[t] <= current[t]` for every target (losses: WP log loss, ATS/O/U MAE).
  - `earned_weight(tuned_weight: float, info: InfoResult) -> float` -- `tuned_weight` if `info.coef > 0 and info.p_one_sided < WEIGHT_ALPHA`, else `0.0`.
  - `@dataclass class Study1aResult` holding, per target and per arm (`current`, `v2`): pooled 2017-2024 loss, per-season loss and mean residual (the home lean), 2026-weeks loss; decision-time `InfoResult` and tuned weight; close `InfoResult` (report only); plus `install: bool`, `weights: dict[str, float]` (v2 arm, earned), games counted and excluded.
  - `run_study_1a(v2_gold_dir: Path, silver_dir: Path, generation_cutoff: datetime) -> Study1aResult`
  - `write_report(result: Study1aResult, out_dir: Path) -> None` -- `out_dir/study_1a.json` and `out_dir/report.md`; the report opens with a plain-English summary of at most 8 lines (does v2 go in, which targets earned weight, how big), then the tables.
  - `scripts/run_study_1a.py: main(argv: list[str] | None = None) -> int` with `--v2-gold-dir` (default `outputs/gold_v2`), `--out-dir` (default `outputs/study_1a`); the cutoff is the run instant.

- [ ] **Step 1: Write the failing tests**

```python
def test_v2_installs_only_if_it_is_at_least_as_good_on_every_target():
    cur = {"wp": 0.640, "ats": 10.50, "ou": 11.07}
    assert install_verdict(cur, {"wp": 0.630, "ats": 10.30, "ou": 11.07})
    assert not install_verdict(cur, {"wp": 0.630, "ats": 10.30, "ou": 11.08})

def test_weight_is_zero_unless_the_information_test_passes():
    assert earned_weight(0.4, InfoResult(0.2, 0.08, 0.006, 1300)) == 0.4
    assert earned_weight(0.4, InfoResult(0.2, 0.15, 0.09, 1300)) == 0.0
    assert earned_weight(0.4, InfoResult(-0.1, 0.03, 0.999, 1300)) == 0.0

def test_report_is_ascii_and_states_the_verdict(tmp_path):
    write_report(fake_result(install=False, weights={"wp": 0.0, "ats": 0.0, "ou": 0.12}), tmp_path)
    text = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert text.isascii()
    assert "v2 does NOT go in" in text
    assert json.loads((tmp_path / "study_1a.json").read_text())["weights"]["ou"] == 0.12
```

- [ ] **Step 2: Run them to see them fail**

Run: `uv run pytest tests/unit/test_study_1a.py -p no:cacheprovider`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement `backtest/study_1a.py` and `scripts/run_study_1a.py`.** Per-target predictions go into `build_tuning_frames` as `{target: DataFrame[game_id, season, prediction, actual]}`; WP 2020 rows drop for lack of a prior fold (expected, reported as excluded). The corpus line is already home-positive: never flip it (map T6).

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/test_study_1a.py -p no:cacheprovider`
Expected: all PASS.

- [ ] **Step 5: Run the real study (not 17:00-17:20 ET)**

Run: `uv run python -m scripts.run_study_1a`
Expected: exit 0; `outputs/study_1a/report.md` and `study_1a.json` exist. Sanity: the current-recipe arm's pooled losses land near the audit's (WP log loss about 0.64, ATS MAE about 10.5, O/U MAE about 11.1); a large departure means the comparator is wired wrong -- stop and investigate before reading the v2 numbers.

- [ ] **Step 6: Lint and commit**

```bash
git add backtest/study_1a.py scripts/run_study_1a.py tests/unit/test_study_1a.py
git commit -m "feat(v2): Study 1a replay, install and weight verdicts"
```

---

### Task 9: Guard scans, merge, and the readout to the owner

**Files:**
- No new code. Copy `outputs/study_1a/report.md` to `C:\Users\jackc\code\nfl-predict\.planning\research\2026-11-05-why-not-profitable\study-1a-report.md`.

- [ ] **Step 1: Run the repo scans a new module can trip (map 8.2) and the parents' tests**

Run: `uv run pytest tests/unit/test_season_floor_single_source.py tests/unit/test_promote_models_ledger_guard.py tests/unit/test_ledger_stamping_site.py tests/unit/test_one_lock_rule_source_scan.py tests/unit/test_one_recommendation_path.py tests/unit/test_no_global_cutoff_in_features.py tests/unit/test_upstream_pin.py tests/unit/test_opponent_adjustment.py tests/unit/test_opponent_adjustment_runs.py tests/unit/test_qb_tracking.py tests/unit/test_build_features_gold_write.py -p no:cacheprovider`
Expected: all PASS. A failure here means a new module broke a repo rule: fix the new module, never the scan.

- [ ] **Step 2: Confirm nothing live changed**

Run: `git -C C:\Users\jackc\code\nfl-predict-v2 diff --name-status master...model-v2`
Expected: every line starts with `A` (added). Any `M` or `D` means an existing file changed: undo it.

- [ ] **Step 3: Merge into master outside 17:00-17:20 ET**

```bash
git -C C:\Users\jackc\code\nfl-predict merge --no-ff model-v2 -m "Merge model-v2: repaired features, v2 recipe and Study 1a (additive; live unchanged)"
```

Then remove the worktree's five junctions with `cmd /c rmdir C:\Users\jackc\code\nfl-predict-v2\data\<name>` for `bronze`, `silver`, `gold`, `baselines`, `optuna` (`rmdir` without `/s` removes the link only, never the live lake) before `git worktree remove`.

- [ ] **Step 4: Report the study to the owner** in plain words: whether v2 goes in, which targets earned weight against the market and how much, and what that means for the switch (plan 3). Do not start plan 3 before the owner has seen this.
