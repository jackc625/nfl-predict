"""Tests for the v2 fixed-input recipe and its walk-forward trainer."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

import models.v2_trainer as trainer
from models.v2_recipe import V2_INPUTS
from models.v2_trainer import fit_v2, predict_v2, walk_forward_v2


def _synthetic_rows(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cols = sorted({c for inputs in V2_INPUTS.values() for c in inputs})
    frame = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=pd.Index(cols))
    frame["elo_diff"] = rng.normal(scale=150.0, size=n)
    noise = rng.normal(scale=3.0, size=n)
    frame["home_margin"] = 2.0 * frame["elo_diff"] / 25 + noise
    frame["home_win"] = (frame["home_margin"] > 0).astype(int)
    frame["total_points"] = (
        44.0 + 0.5 * frame["mkt_total_sum"] + rng.normal(scale=2.0, size=n)
    )
    return frame


def test_fit_recovers_the_direction_of_a_planted_signal():
    rows = _synthetic_rows()
    model = fit_v2("ats", rows)
    probe = rows.head(2).copy()
    probe["elo_diff"] = [100.0, -100.0]
    hi, lo = predict_v2("ats", model, probe)
    assert hi > lo


def test_missing_inputs_still_give_finite_predictions():
    rows = _synthetic_rows()
    model = fit_v2("ou", rows)
    probe = rows.head(3).assign(wind_mph=np.nan, mkt_total_sum=np.nan)
    assert np.isfinite(predict_v2("ou", model, probe)).all()


def test_wp_predictions_are_probabilities():
    rows = _synthetic_rows()
    p = predict_v2("wp", fit_v2("wp", rows), rows)
    assert ((p > 0) & (p < 1)).all()


def _tiny_world(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    rng = np.random.default_rng(1)
    records = []
    for season in range(2015, 2025):
        for week in (1, 2):
            for slot in range(6):
                home = int(rng.integers(10, 35))
                away = int(rng.integers(10, 35))
                if home == away:
                    home += 1
                kick = pd.Timestamp(
                    year=season, month=9, day=7 + 7 * (week - 1), hour=17, tz="UTC"
                )
                records.append(
                    {
                        "game_id": f"{season}_{week:02d}_{slot}",
                        "season": season,
                        "week": week,
                        "kickoff_et": kick.tz_convert("America/New_York"),
                        "home_score": home,
                        "away_score": away,
                    }
                )
    games = pd.DataFrame(records)
    gold_dir = tmp_path / "gold"
    gold_dir.mkdir()
    cols = sorted({c for inputs in V2_INPUTS.values() for c in inputs})
    base = pd.DataFrame(
        rng.normal(size=(len(games), len(cols))), columns=pd.Index(cols)
    )
    base["game_id"] = games["game_id"]
    base["season"] = games["season"]
    base["week"] = games["week"]
    base["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    base["home_margin"] = (games["home_score"] - games["away_score"]).astype(float)
    base["total_points"] = (games["home_score"] + games["away_score"]).astype(float)
    for target in ("wp", "ats", "ou"):
        base.to_parquet(gold_dir / f"features_{target}.parquet")
    return gold_dir, games


def test_walk_forward_never_trains_on_a_game_it_predicts(tmp_path, monkeypatch):
    gold_dir, games = _tiny_world(tmp_path)
    spy: list[set[str]] = []
    real_fit = trainer.fit_v2

    def recording_fit(target, train):
        if target == "ats":
            spy.append(set(train["game_id"].astype(str)))
        return real_fit(target, train)

    monkeypatch.setattr(trainer, "fit_v2", recording_fit)
    out = walk_forward_v2(gold_dir, games, datetime(2026, 9, 1, tzinfo=UTC))

    folds = list(out["ats"].drop_duplicates("fold")["fold"])
    assert len(folds) == len(spy) == 8
    for fold, trained_ids in zip(folds, spy, strict=True):
        predicted = set(out["ats"].loc[out["ats"]["fold"] == fold, "game_id"])
        assert predicted
        assert predicted.isdisjoint(trained_ids)
    assert list(out["wp"].columns) == [
        "game_id",
        "season",
        "week",
        "prediction",
        "actual",
        "fold",
        "fold_cutoff_utc",
    ]


def test_every_fit_runs_inside_the_pinned_thread_pool(tmp_path, monkeypatch):
    gold_dir, games = _tiny_world(tmp_path)
    state = {"pinned": False, "fits": 0, "unpinned_fits": 0, "limit": None}
    real_fit = trainer.fit_v2

    class _Pin:
        def __init__(self, limit):
            state["limit"] = limit

        def __enter__(self):
            state["pinned"] = True

        def __exit__(self, *exc):
            state["pinned"] = False

    def checking_fit(target, train):
        state["fits"] += 1
        state["unpinned_fits"] += 0 if state["pinned"] else 1
        return real_fit(target, train)

    monkeypatch.setattr(trainer, "pinned_thread_pool", _Pin)
    monkeypatch.setattr(trainer, "fit_v2", checking_fit)
    walk_forward_v2(gold_dir, games, datetime(2026, 9, 1, tzinfo=UTC))

    assert state["fits"] > 0
    assert state["unpinned_fits"] == 0
    assert state["limit"] == trainer.THREAD_LIMIT
