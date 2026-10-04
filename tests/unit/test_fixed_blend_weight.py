"""One FIXED blend weight per target; the week-varying blend is retired (Plan 33.2-24 Task 2).

WHAT THIS MODULE PINS (D33.2-10, D33.2-09, SPEC R13)
----------------------------------------------------
* The dynamic shape is REMOVED, not disabled: no ``dynamic_weights`` constructor parameter,
  no ``_dynamic_weights`` identifier anywhere in the parsed ``models/blending.py``, no CLI
  flag reaching the deleted tuner -- proved STRUCTURALLY with a planted-violation control, so
  the comment explaining the retirement can name the retired machinery freely.
* ``save_blend_artifacts`` takes ``update_latest: bool = False`` and by default leaves
  ``latest.json`` BYTE-identical; ``update_latest=True`` rewrites it (the control that makes
  the first case load-bearing).
* ``from_artifacts`` REFUSES a payload carrying a ``dynamic`` section with
  ``RetiredDynamicBlendError`` -- a ``ValueError`` that is NOT a ``KeyError`` or a
  ``FileNotFoundError``, the two types the current-week path and the cache swallow into a
  silent no-blend.
* ``blend_weights.json`` is the blend's ONE payload file and carries its whole provenance:
  the gold generation digest, the three source artifact ids, the converter binding. No
  ``metadata.json`` beside it.
* The WP market side of every HISTORICAL tuning row is OUT OF FOLD (``oof_market_probability``,
  that season's prior-only slope) and NEVER the serving ``slope_beta`` -- asserted by value
  against a planted ``slope_beta`` conversion, and BEHAVIOURALLY with both serving-slope paths
  patched to raise.
* A first-owned-season row is absent from the WP fit, present in the excluded set as
  ``no_prior_fold_converter``, and present in the ATS and O/U fits.
* The fit reads the three corrected source artifacts and the converter -- and no pre-correction
  artifact, no ``latest.json`` and no incumbent blend. The READ SET is asserted under an audit
  hook over a real run, not the intent.
* A weight of exactly 0 or 1 is permitted and REPORTED.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import models.blending as blending_module
from models.blending import BlendConfig, BlendWeights, MarketBlender
from utils.game_lock import game_lock

# The names this plan ADDS to models.blending are read off the module inside each test, so the
# module collects before they exist and a missing one fails as a TEST rather than as an import
# crash that would hide every other case.

REPO_ROOT = Path(__file__).resolve().parents[2]
BLENDING_PATH = REPO_ROOT / "models" / "blending.py"
TUNE_PATH = REPO_ROOT / "backtest" / "tune.py"

_CONVERTER_ID = "market_probability_20990101_000000"
_CONVERTER_SLOPE = 0.15
_DIGEST = "a" * 64
_SOURCE_IDS = {
    "wp": "wp_20990101_000001",
    "ats": "ats_20990101_000002",
    "ou": "ou_20990101_000003",
}
_PRE_CORRECTION_ID = "wp_20260914_221745"
_INCUMBENT_BLEND = "blend_dynamic_20260606_020635"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _converter_dir(
    root: Path,
    *,
    slope: float = _CONVERTER_SLOPE,
    walk_forward: dict[str, float] | None = None,
) -> None:
    _write_json(
        root / _CONVERTER_ID / "metadata.json",
        {
            "version": "1.0",
            "slope_beta": slope,
            "walk_forward_slopes": walk_forward or {"2021": 0.14},
            "training_seasons": [2020, 2021],
            "n_games": 10,
            "input_digest": "0" * 64,
            "fitted_at": "2099-01-01T00:00:00+00:00",
        },
    )


def _source_artifacts(root: Path, digest: str = _DIGEST) -> None:
    for target, artifact_id in _SOURCE_IDS.items():
        _write_json(
            root / artifact_id / "metadata.json",
            {
                "target": target,
                "feature_names": ["f1", "f2"],
                "best_params": {"C": 1.0},
                "exclude_groups": ["injury", "situational", "snap"],
                "gold_generation_digest": digest,
                "thread_limit": 1,
            },
        )


def _incumbents(root: Path) -> None:
    """A PRE-correction model and the retired dynamic incumbent, planted beside the sources."""
    _write_json(
        root / _PRE_CORRECTION_ID / "metadata.json",
        {"target": "wp", "gold_generation_digest": "b" * 64},
    )
    _write_json(
        root / _INCUMBENT_BLEND / "blend_weights.json",
        {
            "weights": {"wp": 0.59, "ats": 0.55, "ou": 0.60},
            "dynamic": {"wp": {"midpoint": 0.2, "steepness": 1.0}},
        },
    )
    _write_json(
        root / "latest.json",
        {
            "wp": _PRE_CORRECTION_ID,
            "ats": "ats_x",
            "ou": "ou_x",
            "blend": _INCUMBENT_BLEND,
        },
    )


def _kickoff(season: int, day: int) -> pd.Timestamp:
    return pd.Timestamp(f"{season}-10-{day:02d} 17:00:00", tz="UTC")


def _silver(root: Path) -> list[str]:
    """Owned timeline + schedule over 2020-2021. Returns the ids that carry a pre-lock line."""
    root.mkdir(parents=True, exist_ok=True)
    games: list[dict[str, Any]] = []
    lines: list[dict[str, Any]] = []
    with_line: list[str] = []
    for season in (2020, 2021):
        for index in range(6):
            game_id = f"{season}_W05_A{index}@H{index}"
            kickoff = _kickoff(season, 4 + index)
            games.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "week": 5,
                    "game_type": "REG",
                    "kickoff_et": kickoff,
                    "home_score": 24 + index,
                    "away_score": 20 if index % 2 else 30,
                }
            )
            if season == 2021 and index == 5:
                continue  # no owned line at all -> EXCLUDED, never filled
            lock = pd.Timestamp(game_lock(kickoff)).tz_convert("UTC")
            lines.append(
                {
                    "game_id": game_id,
                    "snapshot_ts": lock - timedelta(hours=20),
                    "total": 44.0 + index,
                    # the timeline's own inverted sign
                    "spread": -(index - 2.5),
                    "sportsbook": "consensus_median",
                    "region": "us",
                    "created_at": pd.Timestamp("2026-08-16 18:00:00", tz="UTC"),
                }
            )
            with_line.append(game_id)
    pd.DataFrame(games).to_parquet(root / "games.parquet")
    pd.DataFrame(lines).to_parquet(root / "odds_timeline.parquet")
    return with_line


def _fake_predictions(target: str, recipe: Any, seasons: list[int]) -> pd.DataFrame:
    """Deterministic stand-in walk-forward predictions for every fixture game."""
    rows = []
    for season in seasons:
        for index in range(6):
            game_id = f"{season}_W05_A{index}@H{index}"
            home, away = 24 + index, (20 if index % 2 else 30)
            if target == "wp":
                prediction, actual = 0.3 + 0.08 * index, float(home > away)
            elif target == "ats":
                prediction, actual = float(index) - 1.0, float(home - away)
            else:
                prediction, actual = 48.0 + index, float(home + away)
            rows.append(
                {
                    "game_id": game_id,
                    "season": season,
                    "prediction": prediction,
                    "actual": actual,
                }
            )
    return pd.DataFrame(rows)


def _artifacts_tree(root: Path) -> None:
    _converter_dir(root)
    _source_artifacts(root)
    _incumbents(root)


def _run(tmp_path: Path, *, build: bool = True, **overrides: Any):
    """Run the real fit over the fixture tree. ``build=False`` reuses a tree a test edited."""
    from backtest.tune import run_blend_tuning

    artifacts = tmp_path / "artifacts"
    silver = tmp_path / "silver"
    if build:
        _artifacts_tree(artifacts)
        _silver(silver)
    kwargs: dict[str, Any] = {
        "artifacts_dir": artifacts,
        "silver_dir": silver,
        "gold_dir": tmp_path / "gold",
        "source_artifact_ids": dict(_SOURCE_IDS),
        "converter_artifact_id": _CONVERTER_ID,
        "recorded_gold_generation": _DIGEST,
        "predictions_fn": _fake_predictions,
        "gold_generation_fn": lambda: _DIGEST,
    }
    kwargs.update(overrides)
    return run_blend_tuning(**kwargs)


def _provenance() -> Any:
    return blending_module.BlendProvenance(
        gold_generation_digest=_DIGEST,
        source_artifact_ids=dict(_SOURCE_IDS),
        tuning_corpus_rows=11,
        excluded_counts={"no_prelock_line": 1, "no_prior_fold_converter": 6},
        thread_limit=1,
    )


def _tuning_result() -> Any:
    return blending_module.TuningResult(
        weights=BlendWeights(0.4, 0.5, 0.6),
        objective_by_target={
            "wp": "log_loss",
            "ats": "mean_absolute_error",
            "ou": "mean_absolute_error",
        },
        loss_by_target={"wp": 0.6, "ats": 10.0, "ou": 9.9},
        market_only_loss_by_target={"wp": 0.61, "ats": 10.1, "ou": 10.0},
        model_only_loss_by_target={"wp": 0.62, "ats": 10.2, "ou": 10.1},
        grid_by_target={"wp": [], "ats": [], "ou": []},
        seasons_by_target={"wp": [2021], "ats": [2020, 2021], "ou": [2020, 2021]},
        n_games={"wp": 5, "ats": 11, "ou": 11},
        season_best_weight_by_target={"wp": {}, "ats": {}, "ou": {}},
    )


def _bound_blender() -> MarketBlender:
    return MarketBlender(
        config=BlendConfig(weights=BlendWeights(0.4, 0.5, 0.6)),
        market_probability_artifact_id=_CONVERTER_ID,
        market_probability_slope_beta=_CONVERTER_SLOPE,
    )


# ---------------------------------------------------------------------------
# The dynamic shape is removed, not disabled -- structurally
# ---------------------------------------------------------------------------

#: The identifiers whose presence as CODE means the week-varying path survives.
_DYNAMIC_IDENTIFIERS = frozenset({"_dynamic_weights", "dynamic_weights"})
_RETIRED_FLAGS = frozenset(
    {"--dynamic", "--compare", "--baselines-dir", "--rng-seed", "--n-trials"}
)


def dynamic_hook_violations(source: str) -> list[str]:
    """Every CODE construct in *source* that keeps the week-varying blend reachable.

    Identifiers, attributes, parameters and keyword arguments named ``_dynamic_weights`` /
    ``dynamic_weights``, and string constants naming a retired CLI flag inside an
    ``add_argument`` call. Comments and docstrings are not code and are not scanned, so the
    required explanation of the retirement cannot trip the check that the retirement happened.
    """
    tree = ast.parse(source)
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in _DYNAMIC_IDENTIFIERS:
            hits.append(f"name:{node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in _DYNAMIC_IDENTIFIERS:
            hits.append(f"attribute:{node.attr}")
        elif isinstance(node, ast.arg) and node.arg in _DYNAMIC_IDENTIFIERS:
            hits.append(f"parameter:{node.arg}")
        elif isinstance(node, ast.keyword) and node.arg in _DYNAMIC_IDENTIFIERS:
            hits.append(f"keyword:{node.arg}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            for arg in node.args:
                if isinstance(arg, ast.Constant) and arg.value in _RETIRED_FLAGS:
                    hits.append(f"flag:{arg.value}")
    return hits


class TestTheDynamicShapeIsGone:
    def test_the_constructor_takes_no_dynamic_weights(self) -> None:
        params = inspect.signature(MarketBlender.__init__).parameters
        assert "dynamic_weights" not in params
        assert "config" in params  # non-vacuity: the signature was read

    def test_a_blender_carries_no_dynamic_attribute(self) -> None:
        assert not hasattr(MarketBlender(), "_dynamic_weights")

    def test_models_blending_has_no_dynamic_hook_in_code(self) -> None:
        assert dynamic_hook_violations(BLENDING_PATH.read_text(encoding="utf-8")) == []

    def test_backtest_tune_has_no_dynamic_hook_in_code(self) -> None:
        assert dynamic_hook_violations(TUNE_PATH.read_text(encoding="utf-8")) == []

    def test_the_week_season_column_helper_is_gone(self) -> None:
        assert not hasattr(MarketBlender, "_ensure_week_season_columns")

    def test_a_planted_dynamic_branch_is_flagged(self) -> None:
        planted = (
            "class B:\n"
            "    def blend(self):\n"
            "        if self._dynamic_weights:\n"
            "            return 1\n"
        )
        assert dynamic_hook_violations(planted) == ["attribute:_dynamic_weights"]

    def test_a_planted_retired_flag_is_flagged(self) -> None:
        planted = "p.add_argument('--dynamic', action='store_true')\n"
        assert dynamic_hook_violations(planted) == ["flag:--dynamic"]

    def test_prose_naming_the_retired_shape_is_not_a_violation(self) -> None:
        planted = '"""The _dynamic_weights path and --dynamic are retired."""\nx = 1\n'
        assert dynamic_hook_violations(planted) == []

    def test_the_cli_exposes_no_retired_flag(self) -> None:
        from backtest.tune import _build_cli_parser

        options = set(vars(_build_cli_parser()).get("_option_string_actions", {}))
        assert options, "the parser exposed no options at all; the probe proves nothing"
        assert options.isdisjoint(_RETIRED_FLAGS)

    def test_the_six_audited_sites_are_gone(self) -> None:
        from backtest import tune
        from models import blending_data

        for name in (
            "build_dynamic_synthetic_predictions",
            "create_sigmoid_objective",
            "run_dynamic_blend_tuning",
            "run_comparison",
            "_gate_per_target",
            "_generate_comparison_report",
        ):
            assert not hasattr(tune, name), name
        assert not hasattr(blending_data, "extract_noise_profile")
        assert not hasattr(blending_data, "TUNING_SEASONS")


# ---------------------------------------------------------------------------
# One prefix, decided in one place
# ---------------------------------------------------------------------------


class TestOneShapeOnePrefix:
    def test_the_prefix_constant_is_blend(self) -> None:
        assert blending_module.BLEND_ARTIFACT_PREFIX == "blend"

    def test_a_saved_artifact_uses_it_and_never_blend_dynamic(
        self, tmp_path: Path
    ) -> None:
        _converter_dir(tmp_path)
        directory = _bound_blender().save_blend_artifacts(
            _tuning_result(), tmp_path, provenance=_provenance()
        )
        assert directory.name.startswith(f"{blending_module.BLEND_ARTIFACT_PREFIX}_")
        assert not directory.name.startswith("blend_dynamic")


# ---------------------------------------------------------------------------
# update_latest: the default never touches the production swap surface
# ---------------------------------------------------------------------------


class TestUpdateLatest:
    def test_the_default_leaves_latest_json_byte_identical(
        self, tmp_path: Path
    ) -> None:
        _converter_dir(tmp_path)
        _write_json(tmp_path / "latest.json", {"blend": _INCUMBENT_BLEND, "wp": "w"})
        before = _sha256(tmp_path / "latest.json")

        _bound_blender().save_blend_artifacts(
            _tuning_result(), tmp_path, provenance=_provenance()
        )

        assert _sha256(tmp_path / "latest.json") == before

    def test_the_default_creates_no_latest_json_where_none_existed(
        self, tmp_path: Path
    ) -> None:
        _converter_dir(tmp_path)
        _bound_blender().save_blend_artifacts(
            _tuning_result(), tmp_path, provenance=_provenance()
        )
        assert not (tmp_path / "latest.json").exists()

    def test_update_latest_true_rewrites_it(self, tmp_path: Path) -> None:
        """The control that makes the default case load-bearing."""
        _converter_dir(tmp_path)
        _write_json(tmp_path / "latest.json", {"blend": _INCUMBENT_BLEND, "wp": "w"})
        before = _sha256(tmp_path / "latest.json")

        directory = _bound_blender().save_blend_artifacts(
            _tuning_result(), tmp_path, provenance=_provenance(), update_latest=True
        )

        manifest = json.loads((tmp_path / "latest.json").read_text(encoding="utf-8"))
        assert _sha256(tmp_path / "latest.json") != before
        assert manifest["blend"] == directory.name
        assert manifest["wp"] == "w"

    def test_the_signature_default_is_false(self) -> None:
        saver = inspect.signature(MarketBlender.save_blend_artifacts).parameters
        assert saver["update_latest"].default is False


# ---------------------------------------------------------------------------
# A dynamic payload is refused by name, with a type nobody swallows
# ---------------------------------------------------------------------------


class TestADynamicPayloadIsRefused:
    def test_from_artifacts_refuses_a_dynamic_section(self, tmp_path: Path) -> None:
        _incumbents(tmp_path)
        with pytest.raises(
            blending_module.RetiredDynamicBlendError, match=_INCUMBENT_BLEND
        ):
            MarketBlender.from_artifacts(tmp_path)

    def test_the_refusal_is_a_value_error_and_not_a_swallowed_type(self) -> None:
        refusal = blending_module.RetiredDynamicBlendError
        assert issubclass(refusal, ValueError)
        assert not issubclass(refusal, (KeyError, FileNotFoundError))

    def test_the_refusal_escapes_the_current_week_catch_tuple(
        self, tmp_path: Path
    ) -> None:
        """The exact ``except (KeyError, FileNotFoundError)`` the serving paths use."""
        _incumbents(tmp_path)
        with pytest.raises(blending_module.RetiredDynamicBlendError):
            try:
                MarketBlender.from_artifacts(tmp_path)
            except (KeyError, FileNotFoundError):  # pragma: no cover - must not match
                pytest.fail("the refusal was swallowed into a silent no-blend")


# ---------------------------------------------------------------------------
# One payload file, one provenance answer
# ---------------------------------------------------------------------------


class TestTheBlendCarriesItsProvenance:
    def test_blend_weights_json_carries_every_provenance_key(
        self, tmp_path: Path
    ) -> None:
        _converter_dir(tmp_path)
        directory = _bound_blender().save_blend_artifacts(
            _tuning_result(), tmp_path, provenance=_provenance()
        )
        payload = json.loads((directory / "blend_weights.json").read_text("utf-8"))
        assert payload["gold_generation_digest"] == _DIGEST
        assert payload["source_artifact_ids"] == _SOURCE_IDS
        assert payload["market_probability_artifact_id"] == _CONVERTER_ID
        assert payload["market_probability_slope_beta"] == _CONVERTER_SLOPE
        assert payload["excluded_counts"] == {
            "no_prelock_line": 1,
            "no_prior_fold_converter": 6,
        }
        assert payload["thread_limit"] == 1
        assert "dynamic" not in payload

    def test_there_is_no_second_provenance_file(self, tmp_path: Path) -> None:
        _converter_dir(tmp_path)
        directory = _bound_blender().save_blend_artifacts(
            _tuning_result(), tmp_path, provenance=_provenance()
        )
        assert sorted(p.name for p in directory.iterdir()) == ["blend_weights.json"]

    def test_from_artifacts_exposes_the_provenance_on_the_loaded_object(
        self, tmp_path: Path
    ) -> None:
        _converter_dir(tmp_path)
        directory = _bound_blender().save_blend_artifacts(
            _tuning_result(), tmp_path, provenance=_provenance()
        )
        loaded = MarketBlender.from_artifacts(tmp_path, version=directory.name)
        assert loaded.provenance == _provenance()
        assert loaded.market_probability_artifact_id == _CONVERTER_ID
        assert loaded.config.weights == BlendWeights(0.4, 0.5, 0.6)

    def test_a_blender_with_no_converter_binding_refuses_to_save(
        self, tmp_path: Path
    ) -> None:
        """A blend that cannot convert a spread cannot serve WP; it is not written."""
        with pytest.raises(blending_module.MarketProbabilityBindingError):
            MarketBlender().save_blend_artifacts(
                _tuning_result(), tmp_path, provenance=_provenance()
            )

    def test_the_binding_error_class_is_defined_exactly_once(self) -> None:
        tree = ast.parse(BLENDING_PATH.read_text(encoding="utf-8"))
        defs = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
            and node.name == "MarketProbabilityBindingError"
        ]
        assert len(defs) == 1


# ---------------------------------------------------------------------------
# The WP market side is OUT OF FOLD, never the serving slope
# ---------------------------------------------------------------------------


def _corpus_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["2021_a", "2021_b", "2022_a", "2022_b", "2020_a"],
            "season": [2021, 2021, 2022, 2022, 2020],
            "week": [1, 1, 1, 1, 1],
            "market_spread": [7.0, -3.0, 7.0, -3.0, 2.5],
            "market_total": [44.0, 45.0, 46.0, 47.0, 48.0],
        }
    )


def _predictions_for(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    base = frame[["game_id", "season"]]
    return {
        "wp": base.assign(prediction=0.55, actual=[1.0, 0.0, 1.0, 0.0, 1.0]),
        "ats": base.assign(prediction=1.0, actual=[3.0, -4.0, 10.0, 1.0, 2.0]),
        "ou": base.assign(prediction=45.0, actual=[40.0, 50.0, 44.0, 52.0, 47.0]),
    }


class TestTheWpMarketSideIsOutOfFold:
    _WALK_FORWARD = {"2021": 0.05, "2022": 0.30}
    _SERVING = 0.15

    def test_the_market_column_equals_oof_market_probability_row_for_row(self) -> None:
        from models.blending_data import attach_oof_market_probability
        from models.market_probability import oof_market_probability

        frame = _corpus_frame()
        rows, _ = attach_oof_market_probability(frame, self._WALK_FORWARD)
        expected = oof_market_probability(
            rows.assign(home_fav_margin=rows["market_spread"]), self._WALK_FORWARD
        )
        np.testing.assert_array_equal(rows["market_prob_oof"].to_numpy(), expected)

    def test_a_planted_serving_slope_conversion_differs(self) -> None:
        """The control that makes the equality above load-bearing, not coincidental."""
        from models.blending_data import attach_oof_market_probability
        from models.market_probability import market_home_win_probability

        rows, _ = attach_oof_market_probability(_corpus_frame(), self._WALK_FORWARD)
        planted = market_home_win_probability(
            rows["market_spread"].to_numpy(), self._SERVING
        )
        assert not np.allclose(rows["market_prob_oof"].to_numpy(), planted)

    def test_a_first_owned_season_row_is_excluded_from_wp_only(self) -> None:
        from models.blending_data import (
            EXCLUSION_NO_PRIOR_FOLD_CONVERTER,
            build_tuning_frames,
        )

        frame = _corpus_frame()
        frames = build_tuning_frames(frame, _predictions_for(frame), self._WALK_FORWARD)

        assert "2020_a" not in set(frames.frames["wp"]["game_id"])
        excluded = frames.excluded.set_index("game_id")
        assert excluded.loc["2020_a", "reason"] == EXCLUSION_NO_PRIOR_FOLD_CONVERTER
        assert "2020_a" in set(frames.frames["ats"]["game_id"])
        assert "2020_a" in set(frames.frames["ou"]["game_id"])
        # Non-vacuity: every other WP row survived.
        assert len(frames.frames["wp"]) == 4

    def test_the_ats_and_ou_market_is_the_line_itself(self) -> None:
        from models.blending_data import build_tuning_frames

        frame = _corpus_frame()
        frames = build_tuning_frames(frame, _predictions_for(frame), self._WALK_FORWARD)
        ats = frames.frames["ats"].set_index("game_id")
        ou = frames.frames["ou"].set_index("game_id")
        assert ats.loc["2021_a", "market_spread"] == 7.0
        assert ou.loc["2022_b", "market_total"] == 47.0

    def test_a_corpus_game_with_no_prediction_refuses(self) -> None:
        from models.blending_data import TuningCorpusError, build_tuning_frames

        frame = _corpus_frame()
        predictions = _predictions_for(frame)
        predictions["ats"] = predictions["ats"].iloc[1:]
        with pytest.raises(TuningCorpusError, match="2021_a"):
            build_tuning_frames(frame, predictions, self._WALK_FORWARD)

    def test_the_wp_fit_never_touches_a_serving_slope_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BEHAVIOURAL: both serving-slope paths RAISE, and the WP fit still completes."""
        from models import market_probability
        from models.blending_data import build_tuning_frames

        def _explode(*_args: Any, **_kwargs: Any) -> None:
            msg = "a historical row reached a serving-slope conversion"
            raise AssertionError(msg)

        monkeypatch.setattr(market_probability, "market_home_win_probability", _explode)
        monkeypatch.setattr(blending_module, "market_home_win_probability", _explode)
        monkeypatch.setattr(MarketBlender, "_blend_wp_predictions", _explode)

        frame = _corpus_frame()
        frames = build_tuning_frames(frame, _predictions_for(frame), self._WALK_FORWARD)
        result = MarketBlender().tune_weights({"wp": frames.frames["wp"]})

        assert result.n_games["wp"] == 4
        assert 0.0 <= result.weights.wp_model_weight <= 1.0


# ---------------------------------------------------------------------------
# A weight of exactly 0 or 1 is permitted and REPORTED
# ---------------------------------------------------------------------------


class TestABoundaryWeightIsAFinding:
    def test_a_model_that_is_the_outcome_takes_weight_one(self) -> None:
        frame = pd.DataFrame(
            {
                "game_id": [f"g{i}" for i in range(8)],
                "season": [2022] * 8,
                "week": [1] * 8,
                "model_spread": [3.0, -7.0, 10.0, 1.0, -2.0, 4.0, 6.0, -1.0],
                "market_spread": [0.0] * 8,
                "home_margin": [3.0, -7.0, 10.0, 1.0, -2.0, 4.0, 6.0, -1.0],
            }
        )
        result = MarketBlender().tune_weights({"ats": frame})
        assert result.weights.ats_model_weight == 1.0
        assert "ats" in result.boundary_targets

    def test_a_market_that_is_the_outcome_takes_weight_zero(self) -> None:
        frame = pd.DataFrame(
            {
                "game_id": [f"g{i}" for i in range(8)],
                "season": [2022] * 8,
                "week": [1] * 8,
                "model_prob": [0.5] * 8,
                "market_prob_oof": [0.999, 0.001] * 4,
                "home_win": [1.0, 0.0] * 4,
            }
        )
        result = MarketBlender().tune_weights({"wp": frame})
        assert result.weights.wp_model_weight == 0.0
        assert result.boundary_targets == ["wp"]

    def test_an_interior_weight_is_not_reported_as_a_boundary(self) -> None:
        rng = np.random.default_rng(7)
        outcome = rng.normal(0, 10, 400)
        frame = pd.DataFrame(
            {
                "game_id": [f"g{i}" for i in range(400)],
                "season": [2022] * 400,
                "week": [1] * 400,
                "model_total": outcome + rng.normal(0, 6, 400),
                "market_total": outcome + rng.normal(0, 6, 400),
                "total_points": outcome,
            }
        )
        result = MarketBlender().tune_weights({"ou": frame})
        assert 0.0 < result.weights.ou_model_weight < 1.0
        assert result.boundary_targets == []

    def test_an_empty_frame_is_refused_rather_than_fitted(self) -> None:
        """Non-vacuity: a fit over nothing would report a weight that means nothing."""
        empty = pd.DataFrame(
            columns=[
                "game_id",
                "season",
                "week",
                "model_spread",
                "market_spread",
                "home_margin",
            ]
        )
        with pytest.raises(blending_module.BlendTuningError):
            MarketBlender().tune_weights({"ats": empty})


# ---------------------------------------------------------------------------
# The end-to-end fit: the READ SET, the corpus, the written artifact
# ---------------------------------------------------------------------------

_RECORDER: list[tuple[str, bool]] | None = None
_HOOK_INSTALLED = False
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC


def _record_into(sink: list[tuple[str, bool]] | None) -> None:
    """Arm (or disarm) the open-event recorder; audit hooks cannot be uninstalled."""
    global _RECORDER, _HOOK_INSTALLED
    if not _HOOK_INSTALLED:
        sys.addaudithook(_audit)
        _HOOK_INSTALLED = True
    _RECORDER = sink


def _audit(event: str, args: tuple) -> None:
    """Record every ``open`` as (path, is_write).

    ``builtins.open`` reports a mode string; ``os.open`` (which ``tempfile`` uses) reports
    ``None`` for the mode and puts the intent in the flags. Both are classified, so a file the
    fit WRITES is never mistaken for one it READ.
    """
    if _RECORDER is None or event != "open":
        return
    target, mode, flags = args[0], args[1], args[2]
    if not isinstance(target, (str, Path)):
        return
    if isinstance(mode, str):
        is_write = any(ch in mode for ch in "wax+")
    else:
        is_write = bool(int(flags or 0) & _WRITE_FLAGS)
    _RECORDER.append((str(target), is_write))


class TestTheEndToEndFit:
    def test_the_fit_reads_the_three_sources_and_the_converter_and_nothing_else(
        self, tmp_path: Path
    ) -> None:
        opened: list[tuple[str, bool]] = []
        _record_into(opened)
        try:
            run = _run(tmp_path)
        finally:
            _record_into(None)

        artifacts = (tmp_path / "artifacts").resolve()
        read_dirs = set()
        for path, is_write in opened:
            if is_write:
                continue
            resolved = Path(path).resolve()
            if resolved.is_relative_to(artifacts):
                read_dirs.add(resolved.relative_to(artifacts).parts[0])

        assert set(_SOURCE_IDS.values()) <= read_dirs, (
            "a source artifact was never read"
        )
        assert _CONVERTER_ID in read_dirs
        assert read_dirs <= {*_SOURCE_IDS.values(), _CONVERTER_ID}, read_dirs
        assert _PRE_CORRECTION_ID not in read_dirs
        assert _INCUMBENT_BLEND not in read_dirs
        assert "latest.json" not in read_dirs
        assert run.artifact_dir.parent == tmp_path / "artifacts"

    def test_the_fit_leaves_latest_json_byte_identical(self, tmp_path: Path) -> None:
        artifacts = tmp_path / "artifacts"
        _artifacts_tree(artifacts)
        before = _sha256(artifacts / "latest.json")
        _run(tmp_path)
        assert _sha256(artifacts / "latest.json") == before

    def test_the_written_artifact_records_the_run(self, tmp_path: Path) -> None:
        run = _run(tmp_path)
        payload = json.loads(
            (run.artifact_dir / "blend_weights.json").read_text(encoding="utf-8")
        )
        assert payload["source_artifact_ids"] == _SOURCE_IDS
        assert payload["gold_generation_digest"] == _DIGEST
        assert payload["market_probability_artifact_id"] == _CONVERTER_ID
        # 12 scheduled, one with no owned line: 11 tuned, 1 no_prelock_line; the six 2020
        # rows have no prior-fold converter slope, so the WP fit drops them.
        assert payload["tuning_corpus"]["rows"] == 11
        assert payload["excluded_counts"] == {
            "no_prelock_line": 1,
            "no_prior_fold_converter": 6,
        }
        assert payload["n_games"] == {"wp": 5, "ats": 11, "ou": 11}
        assert payload["tuning_seasons"]["wp"] == [2021]
        assert payload["tuning_seasons"]["ats"] == [2020, 2021]
        assert payload["thread_limit"] == 1
        assert "dynamic" not in payload

    def test_a_live_gold_that_is_not_the_recorded_generation_refuses(
        self, tmp_path: Path
    ) -> None:
        from backtest.tune import BlendSourceError

        with pytest.raises(BlendSourceError, match="gold"):
            _run(tmp_path, gold_generation_fn=lambda: "c" * 64)

    def test_source_artifacts_that_disagree_on_their_gold_refuse(
        self, tmp_path: Path
    ) -> None:
        from backtest.tune import BlendSourceError

        artifacts = tmp_path / "artifacts"
        _artifacts_tree(artifacts)
        _write_json(
            artifacts / _SOURCE_IDS["ou"] / "metadata.json",
            {
                "target": "ou",
                "best_params": {},
                "exclude_groups": [],
                "gold_generation_digest": "d" * 64,
            },
        )
        _silver(tmp_path / "silver")
        with pytest.raises(BlendSourceError, match="gold"):
            _run(tmp_path, build=False)

    def test_a_recipe_that_is_not_the_trainer_default_refuses(self) -> None:
        """The walk-forward refits the recipe with tune=False, i.e. the trainer defaults."""
        from backtest.tune import BlendSourceError, SourceRecipe, require_default_recipe

        recipe = SourceRecipe(
            target="wp",
            artifact_id=_SOURCE_IDS["wp"],
            best_params={"C": 0.01, "max_iter": 1000},
            exclude_groups=(),
            gold_generation_digest=_DIGEST,
        )
        with pytest.raises(BlendSourceError, match="default"):
            require_default_recipe(recipe)

    def test_the_production_source_ids_are_the_recorded_refit(self) -> None:
        from backtest.tune import blend_source_artifact_ids

        # Was: P332_23_REFIT_ARTIFACT_IDS, then P332_25B_REFIT_ARTIFACT_IDS. WINDOWS row 19
        # re-fitted the models on the neutral-site Elo gold; the blend is tuned on those.
        from tests.phase33_state import ROW19_REFIT_ARTIFACT_IDS

        assert blend_source_artifact_ids() == dict(ROW19_REFIT_ARTIFACT_IDS)


# ---------------------------------------------------------------------------
# A33.2-review WR-06: the production CLI does not import the tests package
# ---------------------------------------------------------------------------


class TestTheBlendCliDoesNotImportTheTestsPackage:
    def test_backtest_tune_names_no_tests_module_in_any_import(self) -> None:
        tree = ast.parse(TUNE_PATH.read_text(encoding="utf-8"))
        imported: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
            elif isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
        assert imported, "no import was parsed; the probe proves nothing"
        assert [m for m in imported if m == "tests" or m.startswith("tests.")] == []

    def test_the_committed_ids_equal_the_test_manifests_record(self) -> None:
        """The two records of one fact cannot drift apart without this failing."""
        from config import blend_sources
        from tests import phase33_state

        assert (
            blend_sources.BLEND_CONVERTER_ARTIFACT_ID
            == phase33_state.P332_24B_CONVERTER_ARTIFACT_ID
        )
        # The source ids and gold are the WINDOWS row 19 re-fits' (were P332_25B_REFIT_*).
        assert (
            blend_sources.BLEND_SOURCE_ARTIFACT_IDS
            == phase33_state.ROW19_REFIT_ARTIFACT_IDS
        )
        assert (
            blend_sources.BLEND_SOURCE_GOLD_GENERATION
            == phase33_state.ROW19_REFIT_GOLD_GENERATION
        )

    def test_the_fit_refuses_without_a_supplied_live_gold_generation(
        self, tmp_path: Path
    ) -> None:
        from backtest.tune import BlendSourceError

        with pytest.raises(BlendSourceError, match="--gold-generation"):
            _run(tmp_path, gold_generation_fn=None)

    def test_the_cli_requires_the_gold_generation(self) -> None:
        from backtest.tune import _build_cli_parser

        with pytest.raises(SystemExit):
            _build_cli_parser().parse_args([])
        args = _build_cli_parser().parse_args(["--gold-generation", _DIGEST])
        assert args.gold_generation == _DIGEST
