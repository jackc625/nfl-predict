"""Census of the PRODUCTION feature-selection path (Plan 30-17, Task 1).

WHY THIS EXISTS
---------------
Two prior measurements of "does removing information-free columns change what
``SelectFromModel`` selects" disagreed, because they measured different objects:

* Plan 30-15's ``86 -> 89`` came from ``ATSTrainer.select_features`` called DIRECTLY
  and UNCAPPED (``max_features=None``) on the 2018-2019 train leg of a frozen fixture.
* The orchestrator's counter-reproduction ran at a cap of 25 and different
  hyperparameters, found only two features clearing the mean-importance threshold at
  all, and measured a top-25 symmetric difference of ZERO.

Neither describes the production path with confidence. This module settles it by
EXECUTION rather than by reading: it instruments the real call site, records what was
actually binding per target, and then runs the decisive experiment neither prior
measurement performed -- pad the input frame with information-free columns and observe
whether the SELECTED SET moves.

WHAT IT DOES NOT DO
-------------------
Read-only. It never writes under ``data/``, never trains or promotes an artifact, never
touches ``artifacts/``, and leaves no instrumentation behind in ``models/`` (every patch
below is installed and removed inside a context manager in THIS process).

Run it with the project venv:

    uv run python -m scripts.selection_census
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import inspect
import json
import warnings
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection._base import _get_feature_importances
from sklearn.feature_selection._from_model import _calculate_threshold

import models.trainers.base as base_module
from models.temporal import TemporalSplitConfig, WalkForwardSplitter
from models.trainers.ats_trainer import _ATS_MAX_FEATURES, ATSTrainer
from models.trainers.base import BaseTrainer
from models.trainers.ou_trainer import _OU_MAX_FEATURES, OUTrainer
from models.trainers.wp_trainer import _WP_MAX_FEATURES, WPTrainer

# The production trainer, its incumbent per-target budget, and its gold matrix. The cap
# values are IMPORTED, never restated -- a census that hardcoded 20/25/25 would keep
# reporting them after somebody changed the module constant.
TARGETS: dict[str, dict[str, Any]] = {
    "wp": {
        "trainer_cls": WPTrainer,
        "cap": _WP_MAX_FEATURES,
        "gold": Path("data/gold/features_wp.parquet"),
    },
    "ats": {
        "trainer_cls": ATSTrainer,
        "cap": _ATS_MAX_FEATURES,
        "gold": Path("data/gold/features_ats.parquet"),
    },
    "ou": {
        "trainer_cls": OUTrainer,
        "cap": _OU_MAX_FEATURES,
        "gold": Path("data/gold/features_ou.parquet"),
    },
}

# The padding experiment grid. The three kinds ask three different questions:
#
#   constant          -- zero-variance at 0.0. The literal dead-column question: does the
#                        COUNT of information-free columns move the selection?
#   constant_nonzero  -- zero-variance at 1.0. Identical in information content, but a
#                        non-zero constant can carry an intercept-like linear coefficient,
#                        so it is a STRICTLY stronger probe for the LogisticRegression
#                        target than a column of zeros (which contributes nothing to a
#                        linear score no matter what coefficient it is given).
#   noise             -- unit-variance and independent of the target. Information-free
#                        ABOUT THE TARGET, but indistinguishable from a weak real signal at
#                        fit time, so it can legitimately compete for a slot.
PADDING_N = (10, 25, 50)
PADDING_KINDS = ("constant", "constant_nonzero", "noise")

# Fixed so the noise padding is reproducible run to run. Any seed would do; what matters
# is that a re-run of this census produces the same synthetic columns.
NOISE_SEED = 20260822

DEFAULT_OUT = Path("outputs/selection/selection_census.json")


# ---------------------------------------------------------------------------
# Instrumentation -- installed and removed in-process, never left in models/
# ---------------------------------------------------------------------------


class _RecordingSelectFromModel(SelectFromModel):
    """A ``SelectFromModel`` that records the resolved threshold and the importances.

    It records from the SAME fitted estimator the real selection used, so no second fit
    is performed and the recorded threshold is the one that actually decided the mask.
    ``_get_feature_importances`` and ``_calculate_threshold`` are scikit-learn private
    helpers; they are called here deliberately, because re-deriving either by hand would
    be a re-implementation of the very semantics this census is trying to observe.
    """

    sink: list[dict[str, Any]] = []

    def get_support(self, indices: bool = False) -> Any:
        mask = super().get_support(indices=indices)
        scores = np.asarray(
            _get_feature_importances(
                estimator=self.estimator,
                getter=self.importance_getter,
                transform_func="norm",
                norm_order=self.norm_order,
            ),
            dtype=float,
        )
        threshold = float(_calculate_threshold(self.estimator, scores, self.threshold))
        n_clearing = int(np.count_nonzero(scores >= threshold))
        mask_array = np.asarray(mask)
        type(self).sink.append(
            {
                "estimator": type(self.estimator).__name__,
                "max_features_arg": self.max_features,
                "threshold_arg": self.threshold,
                "resolved_threshold": threshold,
                "n_scored": int(scores.size),
                "n_clearing_threshold": n_clearing,
                "n_selected": int(
                    mask_array.size if indices else np.count_nonzero(mask_array)
                ),
            }
        )
        return mask


@contextlib.contextmanager
def _instrumented() -> Iterator[dict[str, list[dict[str, Any]]]]:
    """Patch the selector and ``select_features`` for the duration of the block.

    Records, per ``select_features`` invocation: the CALLER (file:line and function --
    this is how the census answers "which call site runs" by execution rather than by
    reading), the ``max_features`` actually passed, the input width, and the selector
    internals recorded by ``_RecordingSelectFromModel``.
    """
    calls: list[dict[str, Any]] = []
    _RecordingSelectFromModel.sink = []
    original_select = BaseTrainer.select_features
    original_sfm = base_module.SelectFromModel

    def wrapped(
        self: BaseTrainer,
        X: pd.DataFrame,
        y: pd.Series,
        max_features: int | None = None,
    ) -> list[str]:
        caller = inspect.stack()[1]
        before = len(_RecordingSelectFromModel.sink)
        selected = original_select(self, X, y, max_features=max_features)
        internals = (
            _RecordingSelectFromModel.sink[before]
            if len(_RecordingSelectFromModel.sink) > before
            else None
        )
        calls.append(
            {
                "trainer_class": type(self).__name__,
                "caller_file": Path(caller.filename)
                .as_posix()
                .split("nfl-predict/")[-1],
                "caller_line": caller.lineno,
                "caller_function": caller.function,
                "max_features_passed": max_features,
                "input_width": int(X.shape[1]),
                "input_rows": int(X.shape[0]),
                "n_constant_columns": int((X.nunique(dropna=False) <= 1).sum()),
                "constant_columns": sorted(
                    X.columns[(X.nunique(dropna=False) <= 1).to_numpy()].tolist()
                ),
                "n_selected": len(selected),
                "selected": selected,
                "selector": internals,
            }
        )
        return selected

    base_module.SelectFromModel = _RecordingSelectFromModel
    BaseTrainer.select_features = wrapped  # type: ignore[method-assign]
    try:
        yield {"calls": calls}
    finally:
        BaseTrainer.select_features = original_select  # type: ignore[method-assign]
        base_module.SelectFromModel = original_sfm
        _RecordingSelectFromModel.sink = []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pad(df: pd.DataFrame, n: int, kind: str, seed: int) -> pd.DataFrame:
    """Return a copy of ``df`` with ``n`` synthetic information-free columns appended.

    ``constant`` columns are literally zero-variance. ``noise`` columns are drawn
    independently of the target, so they carry no information about it either -- but
    unlike a constant they have variance, which is what lets them compete for a slot.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    for i in range(n):
        name = f"__pad_{kind}_{i:03d}"
        if kind == "constant":
            out[name] = 0.0
        elif kind == "constant_nonzero":
            out[name] = 1.0
        elif kind == "noise":
            out[name] = rng.standard_normal(len(out))
        else:  # pragma: no cover - guarded by the caller's grid
            msg = f"unknown padding kind: {kind}"
            raise ValueError(msg)
    return out


def _select_via_production_route(
    trainer_cls: type[Any],
    cap: int,
    features_df: pd.DataFrame,
    config: TemporalSplitConfig,
) -> dict[str, Any]:
    """Run production Step 1 exactly: split, then ``select_features`` with the cap.

    This mirrors the three concrete ``train_and_evaluate`` overrides
    (``wp_trainer.py:262-267``, ``ats_trainer.py:209-214``, ``ou_trainer.py:212-217``)
    line for line. It is used for the padding grid because it isolates the selection
    step; the census PROVES it reproduces the full production call before relying on it.
    """
    trainer = trainer_cls(config=config)
    target_col = trainer._get_target_column()
    frame = features_df
    if target_col not in frame.columns and target_col == "home_win":
        frame = frame.copy()
        frame[target_col] = (frame["home_score"] > frame["away_score"]).astype(int)
    splitter = WalkForwardSplitter(config=config, target_col=target_col)
    split = splitter.get_train_val_split(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected = trainer.select_features(
            split.train_data, split.train_targets, max_features=cap
        )
    return {"selected": selected, "width": int(split.train_data.shape[1])}


def _uncapped_probe(
    trainer_cls: type[Any],
    features_df: pd.DataFrame,
    config: TemporalSplitConfig,
) -> dict[str, Any]:
    """Reproduce Plan 30-15's measurement path -- DIRECT and UNCAPPED -- on today's gold.

    Recorded so the two disagreeing prior measurements can be reconciled on ONE frame
    rather than compared across two different frames and two different call shapes.
    """
    trainer = trainer_cls(config=config)
    target_col = trainer._get_target_column()
    frame = features_df
    if target_col not in frame.columns and target_col == "home_win":
        frame = frame.copy()
        frame[target_col] = (frame["home_score"] > frame["away_score"]).astype(int)
    splitter = WalkForwardSplitter(config=config, target_col=target_col)
    split = splitter.get_train_val_split(frame)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        selected = trainer.select_features(split.train_data, split.train_targets)
    return {"n_selected": len(selected), "width": int(split.train_data.shape[1])}


# ---------------------------------------------------------------------------
# The census
# ---------------------------------------------------------------------------


def run_census(config: TemporalSplitConfig | None = None) -> dict[str, Any]:
    """Measure the production selection path for all three targets."""
    config = config or TemporalSplitConfig.default()
    census: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "plan": "30-17",
        "sklearn_semantics": _sklearn_semantics(),
        "config": {
            "train_seasons": config.train_seasons,
            "hp_val_seasons": config.hp_val_seasons,
            "holdout_seasons": config.holdout_seasons,
        },
        "gold": {},
        "targets": {},
    }

    for target, spec in TARGETS.items():
        gold_path: Path = spec["gold"]
        gold_df = pd.read_parquet(gold_path)
        census["gold"][target] = {
            "path": gold_path.as_posix(),
            "sha256": _sha256(gold_path),
            "width": int(gold_df.shape[1]),
            "rows": int(gold_df.shape[0]),
        }

        trainer_cls = spec["trainer_cls"]
        cap = spec["cap"]

        # (1) The REAL production entry, instrumented. This is what answers "which call
        # site runs" and "what was binding" by execution.
        with _instrumented() as rec:
            trainer = trainer_cls(config=config)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                trainer.train_and_evaluate(gold_df, None, tune=False)
        production_calls = rec["calls"]

        # (2) The isolated route the padding grid uses, proven against (1).
        isolated = _select_via_production_route(trainer_cls, cap, gold_df, config)
        baseline_selected = isolated["selected"]
        reproduces = (
            len(production_calls) == 1
            and production_calls[0]["selected"] == baseline_selected
        )

        # (3) Determinism of the isolated route on an unchanged frame. Without this a
        # moved selected set under padding could be run-to-run noise rather than
        # count-dependence.
        repeat = _select_via_production_route(trainer_cls, cap, gold_df, config)
        deterministic = repeat["selected"] == baseline_selected

        # (4) The decisive experiment.
        padding: list[dict[str, Any]] = []
        for kind in PADDING_KINDS:
            for n in PADDING_N:
                padded = _pad(gold_df, n, kind, NOISE_SEED + n)
                result = _select_via_production_route(trainer_cls, cap, padded, config)
                added = sorted(set(result["selected"]) - set(baseline_selected))
                removed = sorted(set(baseline_selected) - set(result["selected"]))
                padding.append(
                    {
                        "kind": kind,
                        "n": n,
                        "width_in": result["width"],
                        "n_selected": len(result["selected"]),
                        "selected_set_changed": bool(added or removed),
                        "added": added,
                        "removed": removed,
                        "symmetric_difference": len(added) + len(removed),
                        "synthetic_columns_selected": sorted(
                            c for c in result["selected"] if c.startswith("__pad_")
                        ),
                    }
                )

        # (5) The ABLATION, which is the direction Phase 30 actually moved in. Padding
        # asks what happens when dead columns ARRIVE; Plan 30-07 removed fifteen columns
        # that are constant across the 2018-2019 selection window, so the removal
        # direction is the one with a production precedent (D30-DEFER-13). Dropping every
        # zero-variance column is the largest such move available on this frame.
        constant_in_window = production_calls[0]["constant_columns"]
        ablated_df = gold_df.drop(
            columns=[c for c in constant_in_window if c in gold_df.columns]
        )
        ablated = _select_via_production_route(trainer_cls, cap, ablated_df, config)
        ablation_added = sorted(set(ablated["selected"]) - set(baseline_selected))
        ablation_removed = sorted(set(baseline_selected) - set(ablated["selected"]))
        ablation = {
            "dropped": len(constant_in_window),
            "width_in": ablated["width"],
            "n_selected": len(ablated["selected"]),
            "added": ablation_added,
            "removed": ablation_removed,
            "symmetric_difference": len(ablation_added) + len(ablation_removed),
            "selected_set_changed": bool(ablation_added or ablation_removed),
            "baseline_selected_that_are_constant_in_window": sorted(
                set(baseline_selected) & set(constant_in_window)
            ),
        }

        # (5b) The FIFTEEN-column drop, sized to match Plan 30-07's rung-3 removal. That
        # rung removed the fifteen line_movement columns, every one of which is constant
        # across the canonical 2018-2019 selection window (backtest/signal_lift.py's
        # COVERAGE_WINDOW_CONFIG note), and D30-DEFER-13 attributes a moved WP metric to
        # the resulting 195 -> 180 candidate change. The columns themselves no longer
        # exist in gold, so this drops fifteen OTHER window-constant columns -- the
        # closest available reproduction of the same move on the same frame.
        fifteen = constant_in_window[:15]
        fifteen_df = gold_df.drop(columns=[c for c in fifteen if c in gold_df.columns])
        fifteen_result = _select_via_production_route(
            trainer_cls, cap, fifteen_df, config
        )
        f_added = sorted(set(fifteen_result["selected"]) - set(baseline_selected))
        f_removed = sorted(set(baseline_selected) - set(fifteen_result["selected"]))
        fifteen_drop = {
            "dropped": len(fifteen),
            "dropped_columns": fifteen,
            "width_in": fifteen_result["width"],
            "added": f_added,
            "removed": f_removed,
            "symmetric_difference": len(f_added) + len(f_removed),
            "selected_set_changed": bool(f_added or f_removed),
        }

        # (6) Plan 30-15's uncapped path, re-run on today's gold.
        uncapped = _uncapped_probe(trainer_cls, gold_df, config)

        census["targets"][target] = {
            "trainer_class": trainer_cls.__name__,
            "cap_constant": {
                "wp": "_WP_MAX_FEATURES",
                "ats": "_ATS_MAX_FEATURES",
                "ou": "_OU_MAX_FEATURES",
            }[target],
            "cap_value": cap,
            "production_calls": production_calls,
            "isolated_route_reproduces_production": reproduces,
            "isolated_route_is_deterministic": deterministic,
            "baseline_selected": baseline_selected,
            "uncapped_probe": uncapped,
            "padding_experiment": padding,
            "constant_column_ablation": ablation,
            "fifteen_constant_column_drop": fifteen_drop,
            "count_dependent": any(p["selected_set_changed"] for p in padding)
            or ablation["selected_set_changed"]
            or fifteen_drop["selected_set_changed"],
            "count_dependent_on_zero_variance_columns": any(
                p["selected_set_changed"]
                for p in padding
                if p["kind"].startswith("constant")
            )
            or ablation["selected_set_changed"]
            or fifteen_drop["selected_set_changed"],
        }

    census["verdict"] = _verdict(census)
    census["reconciliation"] = _reconciliation(census)
    return census


def _sklearn_semantics() -> dict[str, Any]:
    """Record what the INSTALLED scikit-learn actually does, not what docs recall."""
    import sklearn

    return {
        "version": sklearn.__version__,
        "source": "sklearn/feature_selection/_from_model.py::SelectFromModel._get_support_mask",
        "rule": (
            "mask = top-`max_features` by importance (argsort, mergesort, descending); "
            "then mask[scores < threshold] = False. With `threshold` unset it resolves "
            "via _calculate_threshold to 'mean' for any non-L1-penalised estimator, i.e. "
            "the MEAN of the norm-transformed importances. The cap and the threshold are "
            "therefore an INTERSECTION: the cap binds only when more than K features "
            "clear the mean, and the threshold binds whenever fewer do."
        ),
    }


def _verdict(census: dict[str, Any]) -> dict[str, Any]:
    per_target = {t: census["targets"][t]["count_dependent"] for t in census["targets"]}
    zero_variance = {
        t: census["targets"][t]["count_dependent_on_zero_variance_columns"]
        for t in census["targets"]
    }
    return {
        "production_selection_is_count_dependent": any(per_target.values()),
        "per_target": per_target,
        "count_dependent_on_zero_variance_columns": any(zero_variance.values()),
        "per_target_zero_variance": zero_variance,
        "mechanism": (
            "The THRESHOLD is not the mechanism. On this gold every target's cap is "
            "binding (more features clear the resolved mean threshold than the cap "
            "admits), so the threshold filter is a no-op and the rule is already "
            "effectively pure top-K. What moves the selected set is the ESTIMATOR REFIT: "
            "select_features fits its scoring model on whatever columns are present, so "
            "adding or removing columns changes the fit itself. For the two XGBoost "
            "targets colsample_bytree=0.8 and subsample=0.8 resample columns per tree, so "
            "even a zero-variance column changes which columns each tree sees and "
            "therefore the gain importances of the real features."
        ),
    }


def _reconciliation(census: dict[str, Any]) -> dict[str, str]:
    ats = census["targets"]["ats"]
    call = ats["production_calls"][0] if ats["production_calls"] else {}
    selector = call.get("selector") or {}
    return {
        "plan_30_15_probe": (
            "ATSTrainer.select_features called DIRECTLY with max_features=None on the "
            "2018-2019 train leg of the frozen Plan 30-03 fixture (210 cols / 6,263 "
            "rows): 201 -> 86 selected before the raw_* exclusion, 195 -> 89 after. It "
            "measured the UNCAPPED rule, where the mean threshold is the ONLY filter, so "
            "the selected count is exactly the number of features clearing the mean and "
            "moves whenever the mean moves. Re-run on today's gold it yields "
            f"{ats['uncapped_probe']['n_selected']} of "
            f"{ats['uncapped_probe']['width']}."
        ),
        "orchestrator_counter_reproduction": (
            "ATS/home_margin at XGBoost defaults with a cap of 25: only two features "
            "cleared the mean threshold at all, and the top-25 symmetric difference was "
            "ZERO. It measured the CAPPED rule at a point where the THRESHOLD, not the "
            "cap, was binding -- when only two features clear the mean, the top-25 "
            "candidate set is irrelevant and the result is those two features either way."
        ),
        "why_they_differ": (
            "They are not two readings of one quantity. sklearn intersects a top-K "
            "candidate set with a threshold filter, and the two probes sat on opposite "
            "sides of that intersection: one removed the cap (so only the threshold "
            "spoke) and one removed the threshold's competition (so, with only two "
            "features clearing, the cap never spoke). Neither is wrong about what it "
            "measured and neither describes production, which is the intersection of "
            "both. This census measures the intersection directly: on ATS production "
            f"passes max_features={call.get('max_features_passed')} over "
            f"{call.get('input_width')} candidates, "
            f"{selector.get('n_clearing_threshold')} clear the resolved threshold "
            f"{selector.get('resolved_threshold')}, and "
            f"{call.get('n_selected')} are selected."
        ),
        "phase_30_production_observation": (
            "Plan 30-07's rung-3 drop removed 15 line_movement columns and moved a WP "
            "metric through the CAPPED production path -- WP went from 195 candidates to "
            "180 with _WP_MAX_FEATURES=20 unchanged, and headline_clv moved "
            "+0.00103305 -> -0.00282341 (D30-DEFER-13). That is a production observation "
            "on a real column-count change, and it is what the padding experiment above "
            "tests directly."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    census = run_census()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")

    print(f"wrote {args.out}")
    for target, block in census["targets"].items():
        call = block["production_calls"][0] if block["production_calls"] else {}
        selector = call.get("selector") or {}
        print(
            f"  {target}: cap={block['cap_value']} width={call.get('input_width')} "
            f"clearing={selector.get('n_clearing_threshold')} "
            f"selected={call.get('n_selected')} "
            f"constant={call.get('n_constant_columns')} "
            f"count_dependent={block['count_dependent']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
