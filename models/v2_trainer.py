"""Fit and walk-forward predict the v2 fixed-input linear models. Writes nothing.

``walk_forward_v2`` returns the same per-target frames as
``refit.remeasure.walk_forward_predictions`` (same folds, columns and dtypes), so they feed
``models.blending_data.build_tuning_frames`` unchanged. ASCII only.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from backtest.refit_preregistration_2026 import THREAD_LIMIT
from forward_ledger.settle import realized_values_from_scores
from models.train import pinned_thread_pool
from models.v2_recipe import (
    V2_INPUTS,
    V2_LABEL,
    V2_TARGETS,
    V2_TRAIN_START,
    build_v2_estimator,
)
from refit.remeasure import FoldSpec, fold_specs
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
    gold_dir: Path,
    games: pd.DataFrame,
    generation_cutoff: datetime,
    *,
    prediction_rows: Mapping[str, pd.DataFrame] | None = None,
) -> dict[str, pd.DataFrame]:
    """Every fold's out-of-sample v2 predictions, per target.

    Per fold, the model trains on the gold's eligible labelled rows from ``V2_TRAIN_START`` and
    predicts the fold's games that have a gold row. With *prediction_rows* (a frame per target,
    each carrying ``game_id``, ``season``, ``week`` and the target's ``V2_INPUTS``) training is
    unchanged, but each fold predicts only its games present in that target's frame, scored from
    those rows; a fold with none is skipped before fitting, and a target with none in any fold
    returns an empty frame. Raises ``ValueError`` if *prediction_rows* lacks a target or a column,
    if a fold would predict a game it trained on, or if one game is predicted twice.
    """
    if prediction_rows is not None:
        for target in V2_TARGETS:
            if target not in prediction_rows:
                msg = f"prediction_rows lacks the {target} target"
                raise ValueError(msg)
            needed = ["game_id", "season", "week", *V2_INPUTS[target]]
            missing = [c for c in needed if c not in prediction_rows[target].columns]
            if missing:
                msg = f"{target} prediction_rows lack the column(s) {missing}"
                raise ValueError(msg)
    specs = fold_specs(games, generation_cutoff)
    labels = realized_values_from_scores(games)
    predictions: dict[str, pd.DataFrame] = {}
    with pinned_thread_pool(THREAD_LIMIT):
        for target in V2_TARGETS:
            supplied = None if prediction_rows is None else prediction_rows[target]
            predictions[target] = _walk_forward_target(
                target, gold_dir, games, specs, labels, prediction_rows=supplied
            )
    return predictions


def _empty_predictions() -> pd.DataFrame:
    """The seven prediction columns with no rows, for a target no fold could score."""
    return pd.DataFrame(
        {
            "game_id": pd.Series(dtype="object"),
            "season": pd.Series(dtype="int64"),
            "week": pd.Series(dtype="int64"),
            "prediction": pd.Series(dtype="float64"),
            "actual": pd.Series(dtype="float64"),
            "fold": pd.Series(dtype="object"),
            "fold_cutoff_utc": pd.Series(dtype="datetime64[ns, UTC]"),
        }
    )


def _walk_forward_target(
    target: str,
    gold_dir: Path,
    games: pd.DataFrame,
    specs: tuple[FoldSpec, ...],
    labels: dict[str, dict[str, float]],
    *,
    prediction_rows: pd.DataFrame | None,
) -> pd.DataFrame:
    """One target's out-of-sample predictions over every fold.

    Trains on the gold; scores *prediction_rows* when given, else the gold rows.
    """
    gold = pd.read_parquet(Path(gold_dir) / f"features_{target}.parquet")
    scored = gold if prediction_rows is None else prediction_rows
    scored_ids = scored["game_id"].astype(str)
    frames: list[pd.DataFrame] = []
    for fold in specs:
        rows = scored.loc[scored_ids.isin(fold.predict_game_ids)].copy()
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
    if not frames:
        return _empty_predictions()
    frame = pd.concat(frames, ignore_index=True)
    repeated = sorted(frame.loc[frame["game_id"].duplicated(), "game_id"])
    if repeated:
        msg = f"{target} predicted {repeated[:10]} in more than one fold"
        raise ValueError(msg)
    return frame
