"""The cutoff-scoped re-fit training path (Phase 37 Plan 37-07; SPEC R3, R10, D-26, D40-06).

WHAT THIS PINS
--------------
* Eligibility at the cutoff (edges R2/R3): a game kicking off exactly at the cutoff is out, one a
  second earlier with a final score is in; a game that kicked off with no final score is out and
  counted under ``no_final_score``; a finished game with no gold row under ``no_gold_row``.
* Labels come from silver scores by the gold label formula and must agree with gold's label.
* D-26: the calibration season is the latest season wholly before the cutoff, and the calibrator /
  converter is fitted on a helper model that never saw that season.
* The frozen input pool (no excluded group, caps respected) and the trainers' default settings.
* The row count equals the finished games minus the recorded exclusions; a short gold refuses.
* A saved generation loads through the serving loader and scores, with ``calibrator.pkl`` for WP
  and a fitted ``residual_std`` for ATS / O/U, and never touches ``latest.json``.
* R10: a game excluded at the date's first attempt stays excluded on a later attempt.

Every store is the synthetic fixture under ``tmp_path`` (``tests/fixtures/refit_gold.py``); the
provisional-row guard is handed the fixture's snapshots. Nothing here reads or writes ``data/``,
``artifacts/`` or ``ledger/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
import math
from datetime import UTC, date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from backtest.diagnose import score_deployed_artifacts
from backtest.refit_preregistration_2026 import (
    EXCLUDED_FEATURE_GROUPS,
    INPUT_CAPS,
    REFIT_RECIPE_ID,
)
from backtest.signal_lift import group_columns
from models.artifacts import load_model_artifact
from models.train import (
    GOLD_GENERATION_DIGEST_METADATA_KEY,
    GROUP_VERDICT_DIGEST_METADATA_KEY,
    THREAD_LIMIT_METADATA_KEY,
)
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer
from pipeline.daily_steps import DailySlate, slate_lock
from refit import witness
from refit.training import (
    EXCLUSION_NO_FINAL_SCORE,
    EXCLUSION_NO_GOLD_ROW,
    RefitTrainingError,
    calibration_season,
    eligible_rows,
    fit_target_at_cutoff,
    train_generation,
)
from tests.fixtures.refit_gold import RefitFixture, build_refit_fixture

_TRAINERS = {"wp": WPTrainer, "ats": ATSTrainer, "ou": OUTrainer}

# The keys Plan 37-20's metadata check reads, plus the serving loaders' own.
_REQUIRED_METADATA_KEYS = (
    "target",
    "feature_names",
    "refit_recipe",
    "cutoff_utc",
    "training_rows",
    "exclusions",
    "selected_inputs",
    "calibration",
    "best_params",
    "exclude_groups",
    "exclude_groups_provenance",
    "preregistration_sha256",
    GOLD_GENERATION_DIGEST_METADATA_KEY,
    GROUP_VERDICT_DIGEST_METADATA_KEY,
    THREAD_LIMIT_METADATA_KEY,
)


@pytest.fixture
def store(tmp_path) -> RefitFixture:
    return build_refit_fixture(tmp_path)


def _gold(store: RefitFixture, target: str) -> pd.DataFrame:
    return pd.read_parquet(store.gold_dir / f"features_{target}.parquet")


def _games(store: RefitFixture) -> pd.DataFrame:
    return pd.read_parquet(store.silver_dir / "games.parquet")


def _finished_before(games: pd.DataFrame, cutoff: datetime) -> set[str]:
    kicked = games[(games["kickoff_et"] < cutoff) & (games["season"] >= 2002)]
    scored = kicked["home_score"].notna() & kicked["away_score"].notna()
    return set(kicked.loc[scored, "game_id"].astype(str))


# ---------------------------------------------------------------------------
# Eligibility (edges R2 / R3 adjacency)
# ---------------------------------------------------------------------------


def test_cutoff_boundary(store):
    eligible = eligible_rows(_gold(store, "wp"), _games(store), store.cutoff_utc, "wp")
    ids = set(eligible.frame["game_id"])
    assert store.at_cutoff_game_id not in ids
    assert store.before_cutoff_game_id in ids
    # The game one second earlier has no gold label yet: its label came from silver.
    row = eligible.frame.loc[eligible.frame["game_id"] == store.before_cutoff_game_id]
    games = _games(store).set_index("game_id")
    expected = float(
        games.loc[store.before_cutoff_game_id, "home_score"]
        > games.loc[store.before_cutoff_game_id, "away_score"]
    )
    assert float(row["home_win"].iloc[0]) == expected


def test_postponed_game_is_excluded_and_counted(store):
    eligible = eligible_rows(
        _gold(store, "ats"), _games(store), store.cutoff_utc, "ats"
    )
    ids = set(eligible.frame["game_id"])
    assert store.postponed_game_id not in ids
    assert eligible.exclusions[EXCLUSION_NO_FINAL_SCORE] == (store.postponed_game_id,)
    assert store.no_gold_game_id not in ids
    assert eligible.exclusions[EXCLUSION_NO_GOLD_ROW] == (store.no_gold_game_id,)


def test_labels_are_checked_against_silver(store):
    gold = _gold(store, "ou")
    bad_id = str(gold.loc[gold["season"] == 2010, "game_id"].iloc[3])
    gold.loc[gold["game_id"] == bad_id, "total_points"] += 3.0
    with pytest.raises(RefitTrainingError) as refused:
        eligible_rows(gold, _games(store), store.cutoff_utc, "ou")
    assert refused.value.check == "training_scope"
    assert bad_id in str(refused.value)


def test_rows_are_in_canonical_order(store):
    gold = _gold(store, "wp").sample(frac=1.0, random_state=7)
    frame = eligible_rows(gold, _games(store), store.cutoff_utc, "wp").frame
    ordered = frame.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
    assert frame["game_id"].tolist() == ordered["game_id"].tolist()
    assert frame.index.tolist() == list(range(len(frame)))


# ---------------------------------------------------------------------------
# D-26 calibration
# ---------------------------------------------------------------------------


def test_calibration_season_is_the_latest_complete_one(store):
    games = _games(store)
    assert calibration_season(games, store.cutoff_utc) == 2025
    first_2020 = games.loc[games["season"] == 2020, "kickoff_et"].min().to_pydatetime()
    assert calibration_season(games, first_2020) == 2019
    # One second after season 2020's last kickoff, 2020 is the latest complete season.
    last_2020 = games.loc[games["season"] == 2020, "kickoff_et"].max().to_pydatetime()
    assert calibration_season(games, last_2020 + timedelta(seconds=1)) == 2020


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_calibrator_never_sees_its_own_training_rows(store, target):
    fit = fit_target_at_cutoff(
        target,
        _gold(store, target),
        _games(store),
        store.cutoff_utc,
        snapshots=store.snapshots,
    )
    calibration = fit.metadata["calibration"]
    assert calibration["season"] == 2025
    assert calibration["helper_seasons"] == list(range(2002, 2025))
    frame = fit.eligible.frame
    season_2025 = sorted(frame.loc[frame["season"] == 2025, "game_id"])
    assert sorted(fit.calibration_game_ids) == season_2025
    assert sorted(calibration["game_ids"]) == season_2025
    helper_ids = set(frame.loc[frame["season"] < 2025, "game_id"])
    assert helper_ids.isdisjoint(fit.calibration_game_ids)
    # The fitted calibration object is on the trainer, so save() persists it.
    if target == "wp":
        assert fit.trainer.calibrator is not None
    elif target == "ats":
        assert fit.trainer.residual_converter is not None
    else:
        assert fit.trainer.total_converter is not None


# ---------------------------------------------------------------------------
# The frozen pool, the caps and the fixed settings
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_inputs_from_the_frozen_pool_within_caps(store, target):
    gold = _gold(store, target)
    excluded = {
        column
        for group in EXCLUDED_FEATURE_GROUPS
        for column in group_columns(gold, group)
    }
    assert len(excluded) == 3  # non-vacuity: the fixture carries one column per group
    fit = fit_target_at_cutoff(
        target, gold, _games(store), store.cutoff_utc, snapshots=store.snapshots
    )
    selected = fit.metadata["selected_inputs"]
    assert selected
    assert excluded.isdisjoint(selected)
    assert excluded.isdisjoint(fit.metadata["calibration"]["helper_inputs"])
    assert len(selected) <= INPUT_CAPS[target]
    assert fit.trainer.feature_names == selected
    defaults = object.__new__(_TRAINERS[target])._get_default_params()
    assert fit.metadata["best_params"] == defaults
    assert fit.metadata["exclude_groups"] == list(EXCLUDED_FEATURE_GROUPS)
    assert fit.metadata["exclude_groups_provenance"] == "verdict"


# ---------------------------------------------------------------------------
# Row count (edge R3 empty)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("target", ["wp", "ats", "ou"])
def test_row_count_matches_finished_games(store, target):
    games = _games(store)
    fit = fit_target_at_cutoff(
        target, _gold(store, target), games, store.cutoff_utc, snapshots=store.snapshots
    )
    finished = _finished_before(games, store.cutoff_utc)
    exclusions = fit.metadata["exclusions"]
    assert exclusions[EXCLUSION_NO_GOLD_ROW] == [store.no_gold_game_id]
    assert exclusions[EXCLUSION_NO_FINAL_SCORE] == [store.postponed_game_id]
    assert fit.metadata["training_rows"] == len(finished) - len(
        exclusions[EXCLUSION_NO_GOLD_ROW]
    )
    assert fit.metadata["training_rows"] == len(fit.eligible.frame)


def test_a_gold_missing_a_whole_season_is_refused(store):
    gold = _gold(store, "wp")
    short = gold[gold["season"] != 2011]
    with pytest.raises(RefitTrainingError) as refused:
        eligible_rows(short, _games(store), store.cutoff_utc, "wp")
    assert refused.value.check == "training_scope"
    assert "2011" in str(refused.value)


def test_a_gold_missing_some_past_games_is_refused(store):
    """WR-B03: a partial build of a past season is refused, never absorbed as exclusions."""
    gold = _gold(store, "wp")
    dropped = sorted(gold.loc[gold["season"] == 2011, "game_id"].astype(str))[:3]
    short = gold[~gold["game_id"].astype(str).isin(dropped)]
    with pytest.raises(RefitTrainingError) as refused:
        eligible_rows(short, _games(store), store.cutoff_utc, "wp")
    assert refused.value.check == "training_scope"
    assert dropped[0] in str(refused.value)


def test_an_empty_gold_is_refused(store):
    gold = _gold(store, "ats").iloc[0:0]
    with pytest.raises(RefitTrainingError) as refused:
        eligible_rows(gold, _games(store), store.cutoff_utc, "ats")
    assert refused.value.check == "training_scope"


# ---------------------------------------------------------------------------
# The tracer: a saved generation loads and scores
# ---------------------------------------------------------------------------


def test_saved_artifact_loads_and_scores(store):
    latest = store.artifacts_dir / "latest.json"
    assert not latest.exists()
    result = train_generation(
        store.cutoff_utc,
        gold_dir=store.gold_dir,
        silver_dir=store.silver_dir,
        artifacts_dir=store.artifacts_dir,
        snapshots=store.snapshots,
    )
    assert sorted(result.artifact_ids) == ["ats", "ou", "wp"]
    assert result.no_final_score_ids == (store.postponed_game_id,)
    assert not latest.exists()

    for target, version in result.artifact_ids.items():
        loaded = load_model_artifact(
            target, version=version, artifacts_dir=store.artifacts_dir
        )
        metadata = loaded["metadata"]
        for key in _REQUIRED_METADATA_KEYS:
            assert key in metadata, f"{target} metadata lacks {key!r}"
        assert metadata["target"] == target
        assert metadata["refit_recipe"] == REFIT_RECIPE_ID
        assert metadata[GOLD_GENERATION_DIGEST_METADATA_KEY] == (
            result.gold_generation_digest
        )
        assert metadata[THREAD_LIMIT_METADATA_KEY] == 1
        assert metadata["preregistration_sha256"] == witness.PREREGISTRATION_SHA256
        assert datetime.fromisoformat(metadata["cutoff_utc"]) == store.cutoff_utc
        assert metadata["cutoff_utc"].endswith("+00:00")
        assert set(metadata["calibration"]) >= {
            "season",
            "helper_seasons",
            "helper_inputs",
            "game_ids",
        }
        assert loaded["feature_list"] == metadata["feature_names"]

        rows = _gold(store, target)
        rows = rows[rows["season"] == 2026].reset_index(drop=True)
        scored = score_deployed_artifacts(
            target, gold_df=rows, artifacts_dir=store.artifacts_dir, version=version
        )
        values = scored["model_prob"].to_numpy(dtype=float)
        assert len(values) == len(rows)
        assert np.isfinite(values).all()
        if target == "wp":
            assert ((values >= 0.0) & (values <= 1.0)).all()
            assert (store.artifacts_dir / version / "calibrator.pkl").is_file()
        else:
            raw = json.loads(
                (store.artifacts_dir / version / "metadata.json").read_text()
            )
            residual_std = raw["converter_params"]["residual_std"]
            assert isinstance(residual_std, float)
            assert math.isfinite(residual_std)
            assert residual_std > 0.0
    assert not latest.exists()


# ---------------------------------------------------------------------------
# R10: eligibility is read at the date's first attempt
# ---------------------------------------------------------------------------


def test_a_score_arriving_after_the_first_attempt_stays_excluded(store):
    first = train_generation(
        store.cutoff_utc,
        gold_dir=store.gold_dir,
        silver_dir=store.silver_dir,
        artifacts_dir=store.artifacts_dir,
        snapshots=store.snapshots,
        save=False,
    )
    assert first.no_final_score_ids == (store.postponed_game_id,)

    # The postponed game's result lands in silver between the attempts.
    games_path = store.silver_dir / "games.parquet"
    games = pd.read_parquet(games_path)
    landed = games["game_id"] == store.postponed_game_id
    games.loc[landed, ["home_score", "away_score"]] = [24.0, 17.0]
    games.to_parquet(games_path, index=False)

    held = train_generation(
        store.cutoff_utc,
        gold_dir=store.gold_dir,
        silver_dir=store.silver_dir,
        artifacts_dir=store.artifacts_dir,
        snapshots=store.snapshots,
        save=False,
        held_no_score_ids=frozenset({store.postponed_game_id}),
    )
    assert held.no_final_score_ids == (store.postponed_game_id,)
    for target in ("wp", "ats", "ou"):
        assert (
            held.metadata[target]["training_rows"]
            == first.metadata[target]["training_rows"]
        )
        assert held.metadata[target]["exclusions"][EXCLUSION_NO_FINAL_SCORE] == [
            store.postponed_game_id
        ]

    fresh = train_generation(
        store.cutoff_utc,
        gold_dir=store.gold_dir,
        silver_dir=store.silver_dir,
        artifacts_dir=store.artifacts_dir,
        snapshots=store.snapshots,
        save=False,
    )
    assert fresh.no_final_score_ids == ()
    for target in ("wp", "ats", "ou"):
        assert (
            fresh.metadata[target]["training_rows"]
            == first.metadata[target]["training_rows"] + 1
        )
    # The silver file on disk was never altered by the held set.
    assert pd.read_parquet(games_path).loc[landed, "home_score"].iloc[0] == 24.0


def test_cutoff_must_be_tz_aware(store):
    naive = datetime(2026, 11, 3, 22, 0)
    with pytest.raises(RefitTrainingError):
        eligible_rows(_gold(store, "wp"), _games(store), naive, "wp")
    assert store.cutoff_utc.tzinfo is UTC


# ---------------------------------------------------------------------------
# D-37: the data preparation is the nightly build a prediction day runs
# ---------------------------------------------------------------------------


def _record_the_build(monkeypatch) -> list[tuple]:
    from refit import data_prep

    calls: list[tuple] = []

    def weather(season, ids, failures):
        calls.append(("weather", season, ids, dict(failures)))
        return {}

    monkeypatch.setattr(
        data_prep, "persist_current_season_elo", lambda ids: calls.append(("elo", ids))
    )
    monkeypatch.setattr(
        data_prep, "step_build_team_form", lambda: calls.append(("team_form",))
    )
    monkeypatch.setattr(data_prep, "build_weather_features_through", weather)
    monkeypatch.setattr(
        data_prep,
        "build_and_save_gold",
        lambda ids, *, unplayed_through_lock: calls.append(
            ("gold", ids, unplayed_through_lock)
        ),
    )
    monkeypatch.setattr(
        data_prep, "step_validate_features", lambda: calls.append(("leakage",))
    )
    return calls


def test_prepare_gold_runs_the_nightly_build_in_order(monkeypatch):
    from refit.data_prep import prepare_gold

    calls = _record_the_build(monkeypatch)
    prepare_gold(date(2026, 11, 3), slate=None)

    assert [call[0] for call in calls] == [
        "elo",
        "team_form",
        "weather",
        "gold",
        "leakage",
    ]
    assert calls[0] == ("elo", frozenset())
    assert calls[2] == ("weather", 2026, frozenset(), {})
    _, serve_ids, lock = calls[3]
    assert serve_ids == frozenset()
    # 18:00 EST on Tue 2026-11-03 (DST ended Sun 2026-11-01).
    assert lock == slate_lock(date(2026, 11, 3))
    assert lock == datetime(2026, 11, 3, 23, 0, tzinfo=UTC)


def test_prepare_gold_keeps_a_same_day_slate(monkeypatch):
    from refit.data_prep import prepare_gold

    run_date = date(2026, 11, 24)
    ids = ("2026_12_AAA_BBB", "2026_12_CCC_DDD")
    failures = {ids[1]: "WeatherDataError"}
    slate = DailySlate(
        run_date_et=run_date,
        lock=slate_lock(run_date),
        schedule=pd.DataFrame(
            {"game_id": list(ids), "season": [2026, 2026], "week": [12, 12]}
        ),
        weather_failures=dict(failures),
    )
    calls = _record_the_build(monkeypatch)
    prepare_gold(run_date, slate=slate)

    assert [call[0] for call in calls] == [
        "elo",
        "team_form",
        "weather",
        "gold",
        "leakage",
    ]
    assert calls[0] == ("elo", frozenset(ids))
    assert calls[2] == ("weather", 2026, frozenset(ids), failures)
    assert calls[3] == ("gold", frozenset(ids), slate.lock)


def test_slate_weather_delegates_unchanged(monkeypatch):
    from pipeline import daily_steps

    run_date = date(2026, 10, 28)
    slate = DailySlate(
        run_date_et=run_date,
        lock=slate_lock(run_date),
        schedule=pd.DataFrame(
            {"game_id": ["2026_08_EEE_FFF"], "season": [2026], "week": [8]}
        ),
        weather_failures={"2026_08_EEE_FFF": "WeatherDataError: timeout"},
    )
    seen: list[tuple] = []

    def through(season, slate_game_ids, weather_failures, *, weather_unknown):
        seen.append((season, slate_game_ids, weather_failures, weather_unknown))
        weather_unknown["2026_08_EEE_FFF"] = weather_failures["2026_08_EEE_FFF"]
        return weather_unknown

    monkeypatch.setattr(daily_steps, "build_weather_features_through", through)
    daily_steps.build_slate_weather_features(slate)

    assert len(seen) == 1
    season, ids, failures, unknown = seen[0]
    assert (season, ids, failures) == (
        2026,
        frozenset({"2026_08_EEE_FFF"}),
        slate.weather_failures,
    )
    # The slate's own map is filled in place, so its entries exist even if the build raises.
    assert unknown is slate.weather_unknown
    assert slate.weather_unknown == {"2026_08_EEE_FFF": "WeatherDataError: timeout"}
