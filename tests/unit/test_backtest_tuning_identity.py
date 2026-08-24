"""Proof the BACKTEST's tuned train genuinely searches, run-scoped and reproducible.

Plan 30-16 (PROD-01), resolving D30-DEFER-01 by owner ruling D30-OWNER-02 (Option 2).

THE DEFECT THIS MODULE CLOSES. ``backtest/engine.py`` calls ``train_and_evaluate(...)`` with
``tune`` defaulting to True, but the trainer it built kept the LEGACY study identity
(``{target}_tuning_v1`` under ``data/optuna/``). Those three studies were written 2026-03-31
and already hold the full 100-trial budget, and ``OptunaTuner.optimize`` computes remaining
trials as ``max(0, n_trials - len(study.trials))`` under ``load_if_exists=True``. So the
search ran ZERO trials, returned the STORED v2.0 parameters, and still reported
``TuningResult.n_trials == 100``. Every "tuned" backtest since March was a straight re-fit on
v2.0 hyperparameters, and nothing failed and nothing logged a warning.

WHY PER-RUN AND NOT PER-PHASE. The Stage-2 identity (``use_phase30_tuning``) is per PHASE,
which is right for a search that happens once per phase and must hard-fail if repeated. The
backtest runs repeatedly -- three engine runs in the canonical PIPELINE sequence alone
(``--blend`` runs the engine twice: blended, then the unblended baseline) plus every execution
of the anchor test. A static tag would make run 1 genuine and every later run a resume at
budget: the same vacuity with a newer date on it. So the identity carries a per-INSTANCE run
id, and the run-scoping test below is what stops the two halves of one ``--blend`` invocation
from sharing a search.

WHY THE FRESHNESS GUARD IS DELIBERATELY DISARMED HERE. Within one engine run the study name
carries the run, not the holdout season, so for target ``wp`` the season-2021 fold fills the
study with the full budget and the 2022/2023/2024 folds resume it and add zero. Arming
``require_fresh_search`` would make every multi-season backtest raise on its second fold. What
replaces the guard is the per-RUN identity itself: no run can inherit ANOTHER run's stored
parameters, which is the property the defect violated.

WHY ASCENDING HOLDOUT ORDER IS LOAD-BEARING, AND THEREFORE GUARDED. The shared study is filled
by the FIRST holdout season processed, so with ascending seasons every later fold trains under
parameters searched on strictly PRIOR data. Reverse or shuffle the list and holdout 2024's
window (train 2018-2022, hp_val 2023) would supply the hyperparameters used to backtest holdout
2021 -- future information reaching a past fold. ``BacktestEngine.run()`` hard-fails a
non-ascending list rather than silently sorting it, because silently reordering what the caller
asked for hides the problem instead of reporting it.

Sibling module: tests/unit/test_promote_models_tuned_path.py, which holds the same proofs for
the Stage-2 identity and the legacy scoping invariant.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import pandas as pd
import pytest

from backtest.engine import BacktestConfig, BacktestEngine
from models.trainers import base as base_trainer
from models.trainers.base import (
    BACKTEST_TUNING_STORAGE_DIR,
    BACKTEST_TUNING_STUDY_TAG_PREFIX,
    LEGACY_TUNING_STUDY_TAG,
    TUNING_STUDY_TAG,
    BaseTrainer,
)
from models.tuning import OptunaTuner

REPO_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Hermetic trainer helpers (the tests/unit/test_promote_models_tuned_path.py shape)
# ---------------------------------------------------------------------------


class _DummyModel:
    """A trivial model whose prediction is its own hyperparameter.

    That makes the objective a deterministic function of the trial's suggested values,
    which is what lets the reproducibility test below mean something.
    """

    def __init__(self, params: dict) -> None:
        self.params = params

    def fit(self, X: pd.DataFrame, y: pd.Series) -> _DummyModel:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), float(self.params.get("x", 0.0)))


class _DummyTrainer(BaseTrainer):
    """Minimal concrete BaseTrainer so the tuning identity can be exercised hermetically."""

    def __init__(self, target: str = "wp") -> None:
        super().__init__(target=target)

    def _create_model(self, params: dict) -> _DummyModel:
        return _DummyModel(params)

    def _get_target_column(self) -> str:
        return "home_win"

    def _predict_raw(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        return model.predict(X)

    def _get_default_params(self) -> dict:
        return {"x": 0.5}

    def _define_search_space(self, trial: optuna.Trial) -> dict:
        return {"x": trial.suggest_float("x", 0.0, 1.0)}


def _prefill_study(storage_dir: Path, study_name: str, n_trials: int) -> None:
    """Create a study under ``storage_dir`` already holding ``n_trials`` completed trials."""
    storage_dir.mkdir(parents=True, exist_ok=True)
    tuner = OptunaTuner(
        study_name=study_name, storage_dir=storage_dir, n_trials=n_trials
    )
    study = optuna.create_study(
        study_name=study_name, storage=tuner.storage_url, direction="minimize"
    )
    study.optimize(lambda trial: trial.suggest_float("x", 0.0, 1.0), n_trials=n_trials)


def _training_frame(n_rows: int = 60) -> tuple[pd.DataFrame, pd.Series]:
    """A frame large enough for make_temporal_cv_splits."""
    rng = np.random.default_rng(7)
    return (
        pd.DataFrame({"f1": np.arange(n_rows, dtype=float)}),
        pd.Series(rng.random(n_rows)),
    )


# ---------------------------------------------------------------------------
# T-30-68: the backtest study storage lives OUTSIDE data/
# ---------------------------------------------------------------------------


def test_backtest_tuning_storage_dir_resolves_outside_data() -> None:
    """``BACKTEST_TUNING_STORAGE_DIR`` is not inside the repository ``data/`` directory.

    ``OptunaTuner`` defaults ``storage_dir`` to ``data/optuna``; this phase forbids writing
    under ``data/`` outside the one sanctioned rebuild, and the prohibition's own before/after
    hash manifest reads the gold parquets and would NOT catch a ``data/optuna/`` write.
    Remediation if this goes red: point BACKTEST_TUNING_STORAGE_DIR back under ``outputs/``.
    """
    storage = Path(BACKTEST_TUNING_STORAGE_DIR).resolve()
    data_root = (REPO_ROOT / "data").resolve()

    assert not storage.is_relative_to(data_root), (
        f"BACKTEST_TUNING_STORAGE_DIR resolves to {storage}, which is inside {data_root}. "
        "A tuned backtest would write under data/, violating the Plan 30-01 prohibition."
    )


def test_a_backtest_trainers_storage_dir_resolves_outside_data() -> None:
    """The INSTANCE storage dir an opted-in trainer ends up with is also outside ``data/``.

    The constant being right is not enough: ``use_backtest_tuning`` derives a per-run
    subdirectory from it, and a derivation bug (an absolute join, say) would escape the
    constant's guarantee.
    """
    trainer = _DummyTrainer()
    trainer.use_backtest_tuning("20260824_120000_abc123")
    resolved = Path(trainer.tuning_storage_dir).resolve()

    assert not resolved.is_relative_to((REPO_ROOT / "data").resolve()), (
        f"An opted-in trainer's storage dir resolves to {resolved}, inside data/."
    )


# ---------------------------------------------------------------------------
# T-30-63 / T-30-66: three identities, three lifetimes, and no default change
# ---------------------------------------------------------------------------


def test_use_backtest_tuning_switches_identity_and_storage_and_leaves_guard_disarmed() -> (
    None
):
    """The opt-in flips tag and storage but deliberately does NOT arm the freshness guard.

    Arming it would raise on the SECOND holdout season of every engine run, because that
    fold legitimately resumes the study this same run created. The property that replaces the
    guard is the per-run identity: no run inherits another run's parameters.
    """
    run_id = "20260824_120000_abc123"
    trainer = _DummyTrainer()

    assert trainer.tuning_study_tag == LEGACY_TUNING_STUDY_TAG
    assert trainer.tuning_storage_dir == base_trainer.LEGACY_TUNING_STORAGE_DIR
    assert trainer.require_fresh_search is False

    trainer.use_backtest_tuning(run_id)

    assert run_id in trainer.tuning_study_tag, (
        f"The backtest study tag {trainer.tuning_study_tag!r} does not carry the run id; "
        "two runs would share one study and the second would resume it at budget."
    )
    assert trainer.tuning_storage_dir != base_trainer.LEGACY_TUNING_STORAGE_DIR
    assert trainer.require_fresh_search is False, (
        "use_backtest_tuning armed the freshness guard; the second holdout season of every "
        "engine run would then raise."
    )


def test_backtest_identity_differs_from_both_the_legacy_and_stage2_identities() -> None:
    """The backtest study name collides with neither the v2.0 nor the Stage-2 identity.

    Colliding with ``{target}_tuning_v1`` reinstates the exact vacuity this plan repairs;
    colliding with the Stage-2 ``p30s2`` identity would let a diagnostic backtest consume the
    binding candidate search's trial budget.
    """
    trainer = _DummyTrainer(target="ats")
    trainer.use_backtest_tuning("20260824_120000_abc123")
    name = f"{trainer.target}_tuning_{trainer.tuning_study_tag}"

    assert name != "ats_tuning_v1", (
        f"Resolved backtest study name {name!r} collides with the v2.0 identity."
    )
    assert name != f"ats_tuning_{TUNING_STUDY_TAG}", (
        f"Resolved backtest study name {name!r} collides with the Stage-2 identity."
    )
    assert BACKTEST_TUNING_STUDY_TAG_PREFIX not in (
        LEGACY_TUNING_STUDY_TAG,
        TUNING_STUDY_TAG,
    )


def test_the_module_default_is_still_the_legacy_identity() -> None:
    """Adding a third identity did NOT change what a non-opted-in caller does.

    ``scripts.retrain_models`` and every other caller that never opts in must keep resuming
    the legacy study, byte-for-byte -- this plan's dispatched scope is the backtest alone.
    """
    trainer = _DummyTrainer()
    assert trainer.tuning_study_tag == LEGACY_TUNING_STUDY_TAG
    assert trainer.tuning_storage_dir == base_trainer.LEGACY_TUNING_STORAGE_DIR
    assert trainer.require_fresh_search is False


# ---------------------------------------------------------------------------
# T-30-63: the identity is per RUN, and every trainer gets it
# ---------------------------------------------------------------------------


def test_two_engines_in_one_process_carry_different_run_ids() -> None:
    """Two ``BacktestEngine`` instances never share a run id.

    ``backtest.run.run_backtest(blend=True)`` constructs and runs the engine TWICE (blended,
    then the unblended baseline), typically within the same second. A timestamp-only run id
    would collide and make the second engine a resume-at-budget of the first.
    """
    ids = {BacktestEngine().run_id for _ in range(8)}
    assert len(ids) == 8, f"BacktestEngine run ids collided: {ids}"


def test_create_trainer_opts_every_trainer_into_this_runs_identity() -> None:
    """Every trainer the engine builds -- all targets, all seasons -- carries the run id.

    The opt-in lives in ``_create_trainer`` rather than at the call site precisely so it
    cannot be forgotten for one target or one holdout season.
    """
    engine = BacktestEngine()
    for season in (2021, 2022, 2023, 2024):
        split = engine._create_split_config(season)
        for target in ("wp", "ats", "ou"):
            trainer = engine._create_trainer(target, split)
            assert engine.run_id in trainer.tuning_study_tag, (
                f"{target}/{season}: trainer tag {trainer.tuning_study_tag!r} does not "
                f"carry the engine run id {engine.run_id!r}."
            )
            assert trainer.tuning_study_tag != LEGACY_TUNING_STUDY_TAG
            assert "data" not in Path(trainer.tuning_storage_dir).parts


def test_create_trainer_source_calls_the_backtest_optin() -> None:
    """Source-level guard: ``_create_trainer`` calls ``use_backtest_tuning``.

    A refactor that returns the trainer before opting it in would restore the legacy
    resume-at-budget silently, and only a full engine run would notice.
    """
    source = inspect.getsource(BacktestEngine._create_trainer)
    assert "use_backtest_tuning" in source, (
        "BacktestEngine._create_trainer no longer calls use_backtest_tuning; the backtest "
        "has silently reverted to resuming the v2.0 studies at budget."
    )


# ---------------------------------------------------------------------------
# T-30-63: a genuine search RECORDS the trials it added; a same-run resume does not raise
# ---------------------------------------------------------------------------


def test_a_fresh_search_records_the_trials_it_added(tmp_path: Path) -> None:
    """Against empty storage the search adds the full budget, and says so on the instance.

    ``trials_added`` is what distinguishes a real search from a resume in the provenance
    record. Before this plan the same call added zero and reported a full count.
    """
    trainer = _DummyTrainer()
    trainer.use_backtest_tuning("20260824_120000_abc123")
    trainer.tuning_storage_dir = tmp_path / "run_a"

    X_train, y_train = _training_frame()
    trainer.tune_hyperparameters(X_train, y_train, n_trials=4)

    assert trainer.last_tuning_trials_before == 0
    assert trainer.last_tuning_trials_added == 4, (
        f"A fresh search added {trainer.last_tuning_trials_added} trials, expected 4."
    )
    assert trainer.last_tuning_study_name == "wp_tuning_" + trainer.tuning_study_tag


def test_resuming_the_study_this_run_created_returns_stored_params_without_raising(
    tmp_path: Path,
) -> None:
    """The second and later holdout seasons of one run resume, add zero, and do NOT raise.

    This is the STATED contract of the backtest identity, not an accident: within a run the
    study is filled by the earliest holdout season and reused by the later ones. Remediation
    if this goes red: do not arm ``require_fresh_search`` on the backtest path.
    """
    run_id = "20260824_120000_abc123"
    storage = tmp_path / "run_a"
    trainer = _DummyTrainer()
    trainer.use_backtest_tuning(run_id)
    trainer.tuning_storage_dir = storage
    study_name = f"wp_tuning_{trainer.tuning_study_tag}"
    _prefill_study(storage, study_name, 4)

    second_fold = _DummyTrainer()
    second_fold.use_backtest_tuning(run_id)
    second_fold.tuning_storage_dir = storage

    X_train, y_train = _training_frame()
    best_params = second_fold.tune_hyperparameters(X_train, y_train, n_trials=4)

    assert "x" in best_params
    assert second_fold.last_tuning_trials_before == 4
    assert second_fold.last_tuning_trials_added == 0, (
        "A same-run resume must report ZERO trials added -- that zero is what makes the "
        "provenance record honest about within-run reuse."
    )


def test_two_independent_fresh_searches_return_identical_best_params(
    tmp_path: Path,
) -> None:
    """Two fresh searches over the same window, in separate storage, agree exactly.

    This is the property the re-ratified v2.1 anchors rest on: the anchors are now produced
    by a search that runs fresh on EVERY invocation, so an unreproducible search would make
    them unpinnable. TPE is seeded 42 with 10 startup trials; the objective here is a
    deterministic function of the suggested parameters.
    """
    X_train, y_train = _training_frame()

    results = []
    for run_id in ("20260824_120000_aaaaaa", "20260824_120001_bbbbbb"):
        trainer = _DummyTrainer()
        trainer.use_backtest_tuning(run_id)
        trainer.tuning_storage_dir = tmp_path / run_id
        results.append(trainer.tune_hyperparameters(X_train, y_train, n_trials=12))

    assert results[0] == results[1], (
        f"Two independent fresh searches disagreed: {results[0]} vs {results[1]}. The "
        "anchors Task 3 re-ratifies cannot be pinned if the search is not reproducible."
    )


# ---------------------------------------------------------------------------
# T-30-65: ascending holdout order is load-bearing, so it is a hard failure
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "seasons", [[2024, 2021, 2022, 2023], [2022, 2021], [2021, 2021, 2022]]
)
def test_non_ascending_holdout_seasons_is_rejected_naming_the_mechanism(
    seasons: list[int],
) -> None:
    """``run()`` refuses a non-ascending holdout list, and the message explains WHY.

    One study per target per run is filled by the FIRST holdout season processed and reused
    by the rest, so a descending or shuffled list applies hyperparameters chosen on a LATER
    window to an EARLIER fold -- future information reaching a past fold. The engine raises
    rather than silently sorting: reordering what the caller asked for hides the problem.
    """
    engine = BacktestEngine(config=BacktestConfig(holdout_seasons=seasons))

    with pytest.raises(ValueError) as excinfo:
        engine.run()

    message = str(excinfo.value)
    assert "ascending" in message.lower(), (
        f"The rejection does not name the rule: {message!r}"
    )
    assert "study" in message.lower(), (
        f"The rejection does not name the shared-study MECHANISM, only the rule: {message!r}"
    )


def test_the_default_holdout_list_is_ascending() -> None:
    """The shipped default satisfies the guard -- the guard cannot break the canonical run."""
    seasons = BacktestConfig().holdout_seasons
    assert seasons == sorted(set(seasons))
