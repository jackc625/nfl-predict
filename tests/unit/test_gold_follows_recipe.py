"""The nightly gold build and its width check follow the recipe in force (MIDSEASON-SWITCH-RECORD.md).

The old models read gold columns BY NAME, so v2 gold served under them would silently corrupt every
prediction. ``pipeline.steps.gold_builder_class`` therefore builds with the v2 builder only while
``refit.generations.in_force_recipe`` answers the installed v2 recipe, and data QA expects that
recipe's widths. Gold and models can then only change together, at the owner install.

The v2 width rung of the ladder (``tests/phase33_state.P332_SWITCH_V2_GOLD_WIDTH_DELTA``) is checked
here against data QA's two width pins and against the column names the v2 build adds.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pandas as pd
import pytest

from features.provenance import InformationTimeViolation
from pipeline import steps
from scripts import build_gold_v2 as v2
from scripts import data_qa, rebuild_gold
from scripts.build_features import FeatureMatrixBuilder
from scripts.build_gold_v2 import (
    RAW_ROLLING_EPA_COLUMNS,
    V2_DERIVED_COLUMNS,
    FeatureMatrixBuilderV2,
    FeatureMatrixBuilderV2Live,
)
from scripts.data_qa import (
    GOLD_FEATURE_MATRICES,
    GOLD_FEATURE_MATRICES_V2,
    expected_gold_widths,
)
from tests import phase33_state
from tests.unit.test_build_gold_v2 import (
    _games,
    _league_family,
    _market_family,
    _three_matrices,
)

V1_RECIPE = "recipe-2026-refit-v1"
V2_RECIPE = "recipe-2026-v2"

# The value columns the two v2 families merge: five market-rating columns and the league level.
_FAMILY_COLUMNS: tuple[str, ...] = (
    "home_mkt_rating",
    "away_mkt_rating",
    "mkt_rating_diff",
    "mkt_total_sum",
    "mkt_rating_coverage",
    "league_points_level",
)


def _in_force(monkeypatch: pytest.MonkeyPatch, recipe: str) -> None:
    """Pin the recipe ``refit.generations.in_force_recipe`` answers."""
    monkeypatch.setattr("refit.generations.in_force_recipe", lambda *_a, **_k: recipe)


@pytest.fixture
def base_build_calls(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Tiny frames in place of the base v2 build, the silver/nflverse loads and the families.

    Patched as ``tests/unit/test_build_gold_v2.py`` patches them; returns the keyword arguments
    each base build was called with.
    """
    calls: list[dict[str, object]] = []

    def fake_generate(self, **kwargs):
        calls.append(kwargs)
        return _three_matrices()

    monkeypatch.setattr(
        FeatureMatrixBuilderV2, "generate_feature_matrices", fake_generate
    )
    monkeypatch.setattr(v2, "load_dataframe", lambda *_a, **_k: _games())
    monkeypatch.setattr(v2, "load_schedules", lambda *_a, **_k: pd.DataFrame())
    monkeypatch.setattr(
        v2,
        "market_ratings",
        lambda *_a: _market_family(pd.Timestamp("2024-02-12 03:30", tz="UTC")),
    )
    monkeypatch.setattr(v2, "league_points_level", lambda *_a: _league_family())
    return calls


def test_the_gold_builder_follows_the_in_force_recipe(monkeypatch):
    _in_force(monkeypatch, V1_RECIPE)
    assert steps.gold_builder_class() is FeatureMatrixBuilder
    _in_force(monkeypatch, V2_RECIPE)
    assert steps.gold_builder_class() is FeatureMatrixBuilderV2Live


def test_the_gold_builder_refuses_without_a_latest_json(monkeypatch, tmp_path):
    """No ``artifacts/latest.json``: the recipe is never guessed, so no builder is picked.

    The lookup is NOT patched: an empty directory (no ``artifacts/``, no ``ledger/``) is what
    the real ``in_force_recipe`` reads, and a fall-back to the v1 builder would return here.
    """
    monkeypatch.chdir(tmp_path)
    with pytest.raises(FileNotFoundError, match=r"latest\.json"):
        steps.gold_builder_class()


def test_the_live_v2_build_adds_the_families_once(base_build_calls):
    out = FeatureMatrixBuilderV2Live().generate_feature_matrices(
        excluded_game_ids=frozenset()
    )

    assert set(V2_DERIVED_COLUMNS) | {"mkt_rating_diff", "league_points_level"} <= set(
        out["ou"].columns
    )
    # Once: a second merge would have been refused (no family overwrites a column), and each
    # matrix is exactly its base width plus the six family columns and the six derived ones.
    for target, base in _three_matrices().items():
        assert len(out[target].columns) == len(base.columns) + len(
            _FAMILY_COLUMNS
        ) + len(V2_DERIVED_COLUMNS)
    assert len(base_build_calls) == 1


def test_the_live_v2_build_passes_the_run_s_exclusions_to_the_base_build(
    base_build_calls,
):
    lock = datetime(2024, 9, 4, 22, 0, tzinfo=UTC)
    FeatureMatrixBuilderV2Live().generate_feature_matrices(
        excluded_game_ids=frozenset({"2024_W01_NYJ@SF"}), unplayed_through_lock=lock
    )

    (call,) = base_build_calls
    assert call["excluded_game_ids"] == frozenset({"2024_W01_NYJ@SF"})
    assert call["unplayed_through_lock"] == lock


def test_an_empty_live_v2_build_is_returned_as_is(monkeypatch):
    def unreachable(*_a, **_k):
        raise AssertionError("an empty build loads no family input")

    monkeypatch.setattr(
        FeatureMatrixBuilderV2, "generate_feature_matrices", lambda self, **_k: {}
    )
    monkeypatch.setattr(v2, "load_dataframe", unreachable)
    monkeypatch.setattr(v2, "load_schedules", unreachable)

    assert FeatureMatrixBuilderV2Live().generate_feature_matrices() == {}


def test_the_live_v2_build_refuses_a_planted_provenance_column(
    monkeypatch, base_build_calls
):
    """``check_v2_matrices`` runs on the nightly path: a provenance-named column is refused."""

    def planted_generate(self, **_kwargs):
        matrices = _three_matrices()
        matrices["ou"] = matrices["ou"].assign(elo_information_time=0.0)
        return matrices

    monkeypatch.setattr(
        FeatureMatrixBuilderV2, "generate_feature_matrices", planted_generate
    )
    with pytest.raises(InformationTimeViolation, match="information_time"):
        FeatureMatrixBuilderV2Live().generate_feature_matrices(
            excluded_game_ids=frozenset()
        )


def test_data_qa_expects_the_in_force_recipe_widths():
    assert expected_gold_widths(V2_RECIPE) == {
        "features_wp": 212,
        "features_ats": 212,
        "features_ou": 211,
    }
    assert expected_gold_widths(V1_RECIPE) == GOLD_FEATURE_MATRICES


def _gold_of_width(width: int) -> pd.DataFrame:
    """A one-row gold matrix *width* columns wide, with no all-null column."""
    columns: dict[str, list[object]] = {
        "game_id": ["2024_W01_BAL@KC"],
        "season": [2024],
        "week": [1],
    }
    columns |= {f"feature_{index}": [0.0] for index in range(width - len(columns))}
    return pd.DataFrame(columns)


def test_the_gold_integrity_check_expects_the_in_force_recipe_widths(monkeypatch):
    monkeypatch.setattr(
        data_qa,
        "load_dataframe",
        lambda name, **_k: _gold_of_width(GOLD_FEATURE_MATRICES_V2[name]),
    )
    monitor = object.__new__(data_qa.DataQualityMonitor)

    _in_force(monkeypatch, V2_RECIPE)
    under_v2 = monitor.check_gold_integrity()["checks"]
    _in_force(monkeypatch, V1_RECIPE)
    under_v1 = monitor.check_gold_integrity()["checks"]

    for name in GOLD_FEATURE_MATRICES_V2:
        assert under_v2[name]["status"] == "pass", under_v2[name]
        # Control: the same v2 gold under a v1 recipe is refused on width.
        assert under_v1[name]["status"] == "fail", under_v1[name]
        assert under_v1[name]["expected_column_count"] == GOLD_FEATURE_MATRICES[name]


def test_the_v2_ladder_slot_matches_data_qa():
    slot = phase33_state.P332_SWITCH_V2_GOLD_WIDTH_DELTA
    assert slot["widths_after"] == tuple(GOLD_FEATURE_MATRICES_V2.values())
    assert slot["widths_before"] == tuple(GOLD_FEATURE_MATRICES.values())
    added = slot["added"]
    assert isinstance(added, tuple)
    assert len(added) == 24
    assert set(added) == {
        *RAW_ROLLING_EPA_COLUMNS,
        *_FAMILY_COLUMNS,
        *V2_DERIVED_COLUMNS,
    }
    assert slot["removed"] == ()


def test_rebuild_gold_runs_the_pipeline_gold_step_once(monkeypatch):
    """The recovery command is the pipeline's own gold step, whose builder follows the recipe."""
    calls: list[str] = []
    monkeypatch.setattr(
        "pipeline.steps.step_build_features", lambda: calls.append("built")
    )
    monkeypatch.setattr(sys, "argv", ["rebuild_gold"])

    assert rebuild_gold.main() == 0
    assert calls == ["built"]


def test_the_gold_recovery_command_is_rebuild_gold():
    """The stale-gold refusal names the recipe-following rebuild, never scripts.build_features."""
    command = steps._BOUNDARY_RECOVERY_COMMAND[steps.ARTIFACT_BOUNDARY_GOLD]
    assert command == "uv run python -m scripts.rebuild_gold"
