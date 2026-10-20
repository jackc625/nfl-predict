"""A re-fit is bitwise reproducible, pinned to one thread, with no reachable search (Phase 37).

WHY THESE TESTS EXIST (SPEC R15, RFIT-07, research Q7)
-----------------------------------------------------
Research Q7 measured an XGBoost fit run on every core differing from a one-thread fit by up to
3.98 points on the same data, and XGBoost's row subsampling draws by position, so a shuffled
training frame is a different model. A re-fit generation must be reproducible from the gold it
read, so these tests pin the guarantees that make it so:

* two re-fits on one gold copy with one cutoff give bitwise-equal float64 predictions for WP, ATS
  and O/U -- ``numpy.array_equal``, never a tolerance;
* the same holds when the gold rows arrive in a shuffled order (rows are put in canonical order
  before anything else);
* a thread count other than one is refused before any file is read, and the count in force is
  recorded in every saved artifact;
* no hyperparameter search is reachable: with every tuning entry point patched to raise, a whole
  re-fit still completes.

Fixture only (``tests/fixtures/refit_gold.py`` under ``tmp_path``); nothing here reads or writes
``data/``, ``artifacts/`` or ``ledger/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from models.train import THREAD_LIMIT_METADATA_KEY
from models.trainers.base import BaseTrainer
from refit.training import (
    RefitThreadCountError,
    fit_target_at_cutoff,
    predict_rows,
    train_generation,
)
from tests.fixtures.refit_gold import (
    FIXTURE_CUTOFF_UTC,
    RefitFixture,
    build_refit_fixture,
)

_TARGETS = ("wp", "ats", "ou")
_SHUFFLE_SEED = 11


@pytest.fixture
def store(tmp_path) -> RefitFixture:
    return build_refit_fixture(tmp_path)


def _gold(store: RefitFixture, target: str) -> pd.DataFrame:
    return pd.read_parquet(store.gold_dir / f"features_{target}.parquet")


def _games(store: RefitFixture) -> pd.DataFrame:
    return pd.read_parquet(store.silver_dir / "games.parquet")


def _predictions(store: RefitFixture, target: str, gold: pd.DataFrame) -> np.ndarray:
    fit = fit_target_at_cutoff(
        target, gold, _games(store), store.cutoff_utc, snapshots=store.snapshots
    )
    # One fixed frame for every run: the target's whole gold matrix, in file order.
    return predict_rows(fit, _gold(store, target))


@pytest.mark.parametrize("target", _TARGETS)
def test_double_refit_is_bitwise_equal(store, target):
    first = _predictions(store, target, _gold(store, target))
    second = _predictions(store, target, _gold(store, target))
    assert first.dtype == np.float64
    assert second.dtype == np.float64
    assert len(first) == len(_gold(store, target))
    assert np.array_equal(first, second)


@pytest.mark.parametrize("target", _TARGETS)
def test_shuffled_row_order_is_identical(store, target):
    gold = _gold(store, target)
    shuffled = gold.sample(frac=1.0, random_state=_SHUFFLE_SEED)
    assert shuffled["game_id"].tolist() != gold["game_id"].tolist()
    in_order = _predictions(store, target, gold)
    out_of_order = _predictions(store, target, shuffled)
    assert np.array_equal(in_order, out_of_order)


def test_a_thread_count_other_than_one_is_refused(tmp_path):
    missing = tmp_path / "does-not-exist"
    with pytest.raises(RefitThreadCountError) as refused:
        train_generation(
            FIXTURE_CUTOFF_UTC,
            gold_dir=missing / "gold",
            silver_dir=missing / "silver",
            artifacts_dir=missing / "artifacts",
            thread_limit=2,
        )
    assert refused.value.check == "fixed_settings"
    assert not missing.exists()


def test_the_thread_count_is_recorded(store):
    result = train_generation(
        store.cutoff_utc,
        gold_dir=store.gold_dir,
        silver_dir=store.silver_dir,
        artifacts_dir=store.artifacts_dir,
        snapshots=store.snapshots,
    )
    assert sorted(result.artifact_ids) == sorted(_TARGETS)
    for target, version in result.artifact_ids.items():
        assert result.metadata[target][THREAD_LIMIT_METADATA_KEY] == 1
        saved = json.loads(
            (store.artifacts_dir / version / "metadata.json").read_text()
        )
        assert saved[THREAD_LIMIT_METADATA_KEY] == 1


def test_no_search_is_reachable(store, monkeypatch):
    def _refuse(*_args, **_kwargs):
        msg = "a re-fit reached a hyperparameter search entry point"
        raise AssertionError(msg)

    for name in (
        "tune_hyperparameters",
        "use_phase30_tuning",
        "use_phase332_tuning",
        "use_backtest_tuning",
    ):
        monkeypatch.setattr(BaseTrainer, name, _refuse)

    result = train_generation(
        store.cutoff_utc,
        gold_dir=store.gold_dir,
        silver_dir=store.silver_dir,
        artifacts_dir=store.artifacts_dir,
        snapshots=store.snapshots,
        save=False,
    )
    assert sorted(result.metadata) == sorted(_TARGETS)
    for target in _TARGETS:
        assert result.metadata[target]["training_rows"] > 0
