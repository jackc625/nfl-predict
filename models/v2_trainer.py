"""Fit and walk-forward predict the v2 fixed-input linear models. Writes nothing.

``walk_forward_v2`` returns the same per-target frames as
``refit.remeasure.walk_forward_predictions`` (same folds, columns and dtypes), so they feed
``models.blending_data.build_tuning_frames`` unchanged. ASCII only.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from forward_ledger.settle import realized_values_from_scores
from models.v2_recipe import (
    V2_INPUTS,
    V2_LABEL,
    V2_TARGETS,
    V2_TRAIN_START,
    build_v2_estimator,
)
from refit.remeasure import fold_specs
from refit.training import eligible_rows

__all__ = ["fit_v2", "predict_v2", "walk_forward_v2"]

_ROW_ORDER: list[str] = ["season", "week", "game_id"]


def _inputs(target: str, rows: pd.DataFrame) -> pd.DataFrame:
    columns = list(V2_INPUTS[target])
    missing = [name for name in columns if name not in rows.columns]
    if missing:
        msg = f"{target} rows lack the fixed input column(s) {missing}"
        raise ValueError(msg)
    return cast("pd.DataFrame", rows[columns]).astype(float)


def fit_v2(target: str, train: pd.DataFrame) -> Pipeline:
    """Fit the target's fixed recipe on *train* (its inputs plus its label column)."""
    label = V2_LABEL[target]
    if label not in train.columns:
        msg = f"{target} training rows lack the label column {label!r}"
        raise ValueError(msg)
    model = build_v2_estimator(target)
    model.fit(_inputs(target, train), train[label].astype(float))
    return model


def predict_v2(target: str, model: Pipeline, rows: pd.DataFrame) -> np.ndarray:
    """WP: probability of a home win; ATS: home margin; O/U: the total."""
    features = _inputs(target, rows)
    if target == "wp":
        return np.asarray(model.predict_proba(features)[:, 1], dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def walk_forward_v2(
    gold_dir: Path, games: pd.DataFrame, generation_cutoff: datetime
) -> dict[str, pd.DataFrame]:
    """Every fold's out-of-sample v2 predictions, per target.

    Per fold, the model trains on the eligible labelled rows from ``V2_TRAIN_START`` and predicts
    the fold's games that have a gold row. Raises ``ValueError`` if a fold would predict a game it
    trained on, or if one game is predicted twice.
    """
    specs = fold_specs(games, generation_cutoff)
    labels = realized_values_from_scores(games)
    predictions: dict[str, pd.DataFrame] = {}
    for target in V2_TARGETS:
        gold = pd.read_parquet(Path(gold_dir) / f"features_{target}.parquet")
        gold_ids = gold["game_id"].astype(str)
        frames: list[pd.DataFrame] = []
        for fold in specs:
            rows = gold.loc[gold_ids.isin(fold.predict_game_ids)].copy()
            if rows.empty:
                continue
            rows["game_id"] = rows["game_id"].astype(str)
            rows = rows.sort_values(_ROW_ORDER, kind="mergesort").reset_index(drop=True)
            eligible = eligible_rows(gold, games, fold.cutoff_utc, target).frame
            train = eligible.loc[eligible["season"].astype(int) >= V2_TRAIN_START]
            seen = set(train["game_id"].astype(str)) & set(rows["game_id"])
            if seen:
                msg = f"{target} fold {fold.label} would predict {sorted(seen)[:10]}, trained on"
                raise ValueError(msg)
            model = fit_v2(target, train)
            frames.append(
                pd.DataFrame(
                    {
                        "game_id": rows["game_id"],
                        "season": rows["season"].astype(int),
                        "week": rows["week"].astype(int),
                        "prediction": predict_v2(target, model, rows),
                        "actual": [labels[target][gid] for gid in rows["game_id"]],
                        "fold": fold.label,
                        "fold_cutoff_utc": pd.Timestamp(fold.cutoff_utc),
                    }
                )
            )
        frame = pd.concat(frames, ignore_index=True)
        repeated = sorted(frame.loc[frame["game_id"].duplicated(), "game_id"])
        if repeated:
            msg = f"{target} predicted {repeated[:10]} in more than one fold"
            raise ValueError(msg)
        predictions[target] = frame
    return predictions
