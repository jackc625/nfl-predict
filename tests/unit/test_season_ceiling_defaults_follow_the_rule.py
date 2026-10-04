"""Live season-ceiling defaults follow ``conf.season_partition.LATEST_COMPLETED_SEASON``.

Review fix WR-14 (commit 48d7464). These defaults were the literal 2024, and a ceiling that names
a season is exactly how 2025 stayed invisible to every consumer that loaded data through it. They
now derive from the committed rule, so the next season roll moves them all with one edit.

THE SITES (one pytest node each, so a failure names the one that drifted):

* ``scripts/data_qa.py``: ``GOLD_LAST_COMPLETE_SEASON`` (live via pipeline/steps.py);
* ``backtest/walkforward.py``: ``BacktestConfig.end_season`` and the two factories
  ``create_default_config()`` and ``create_strict_config()``;
* ``models/utils.py``: ``end_season`` of ``WalkForwardValidator.create_seasonal_splits`` and
  ``TrainingPipeline.run_walk_forward_training``;
* ``models/train_wp.py`` and ``models/baseline_elo.py``: ``run_walk_forward_validation``.

Defaults are read through ``inspect.signature`` and by constructing the config, so nothing is
executed beyond importing the modules. The FENCED record literals in ``backtest/signal_lift.py``,
``backtest/diagnose.py`` and ``backtest/ou_divergence.py`` deliberately stay 2021-2024 and are not
covered; neither is the argparse branch of ``scripts/build_team_form.py`` or
``scripts/capture_baseline.py``.

NON-VACUITY: the rule itself is an int at or past 2025, so "equals the rule" cannot be satisfied
by a stale or missing value on both sides.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import pytest

from backtest.walkforward import (
    BacktestConfig,
    create_default_config,
    create_strict_config,
)
from conf.season_partition import LATEST_COMPLETED_SEASON
from models.baseline_elo import BaselineEloModel
from models.train_wp import WinProbabilityModel
from models.utils import TrainingPipeline, WalkForwardValidator
from scripts import data_qa


def _signature_default(function: Callable[..., Any]) -> int:
    return inspect.signature(function).parameters["end_season"].default


#: site name -> callable returning the current default for that site.
CEILING_SITES: dict[str, Callable[[], Any]] = {
    "scripts.data_qa.GOLD_LAST_COMPLETE_SEASON": lambda: (
        data_qa.GOLD_LAST_COMPLETE_SEASON
    ),
    "backtest.walkforward.BacktestConfig.end_season": lambda: (
        BacktestConfig().end_season
    ),
    "backtest.walkforward.create_default_config()": lambda: (
        create_default_config().end_season
    ),
    "backtest.walkforward.create_strict_config()": lambda: (
        create_strict_config().end_season
    ),
    "models.utils.WalkForwardValidator.create_seasonal_splits": lambda: (
        _signature_default(WalkForwardValidator.create_seasonal_splits)
    ),
    "models.utils.TrainingPipeline.run_walk_forward_training": lambda: (
        _signature_default(TrainingPipeline.run_walk_forward_training)
    ),
    "models.train_wp.WinProbabilityModel.run_walk_forward_validation": lambda: (
        _signature_default(WinProbabilityModel.run_walk_forward_validation)
    ),
    "models.baseline_elo.BaselineEloModel.run_walk_forward_validation": lambda: (
        _signature_default(BaselineEloModel.run_walk_forward_validation)
    ),
}


def test_the_rule_is_a_real_season() -> None:
    """NON-VACUITY: the value every site is compared with is an int, 2025 or later."""
    assert type(LATEST_COMPLETED_SEASON) is int
    assert LATEST_COMPLETED_SEASON >= 2025
    assert len(CEILING_SITES) == 8


@pytest.mark.parametrize("site", sorted(CEILING_SITES))
def test_the_ceiling_default_equals_the_latest_completed_season(site: str) -> None:
    actual = CEILING_SITES[site]()
    assert actual == LATEST_COMPLETED_SEASON, (
        f"{site} defaults to {actual!r} but the rule says the latest completed season is "
        f"{LATEST_COMPLETED_SEASON}. It must import conf.season_partition."
        "LATEST_COMPLETED_SEASON rather than carry a literal, or the next season roll leaves "
        "it behind."
    )
