"""The superseding correction of the EV floor and the frozen residual SD (Plan 33.2-29 Task 1).

D33.2-25: both scalars were swept against CLOSING-line outcomes on models this phase replaced,
so they are re-derived on the corrected models and the owned pre-lock lines (2020-2024) and
published as a visible correction that never edits the frozen ``ee20773`` pre-registration.

What these tests pin:

* the frozen pre-registration (``backtest/ev_chain_constants.py`` +
  ``PROFITABILITY-PREREGISTRATION.md``) and the spent 2025 run ledger are untouched;
* the corrected module names ``ee20773`` and re-declares NO frozen symbol (planted control);
* every corrected floor is a frozen grid value or a named refusal (``None``), never a
  numeric stand-in; ATS and O/U carry a frozen SD, WP does not (D31-07);
* a pool too small to sweep REFUSES where the Phase-31 recipe fell back to the grid floor;
* a 2025 row or a closing line reaching the fit is refused;
* the WP market side is converted OUT OF FOLD and never through the serving slope.

The derivation module is imported lazily, so a missing module fails an assertion rather than
crashing collection.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import importlib
import importlib.util
import math
import subprocess
import tomllib
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_MODULE = "scripts.derive_corrected_ev_chain"
CORRECTED_MODULE = "backtest.corrected_ev_chain_constants"
FROZEN_PAIR = ("backtest/ev_chain_constants.py", "PROFITABILITY-PREREGISTRATION.md")
LEDGER = "config/profitability_2025_run_ledger.toml"
SUPERSEDED_SHA = "ee20773"


class ServingSlopeUsedOnHistoryError(AssertionError):
    """A historical row reached a serving-slope conversion path."""


def _load(name: str) -> ModuleType:
    """Import *name*, failing an ASSERTION (not a collection error) when it does not exist."""
    assert importlib.util.find_spec(name) is not None, f"{name} does not exist yet"
    return importlib.import_module(name)


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _uppercase_names(names: list[str] | set[str]) -> set[str]:
    return {name for name in names if name.isupper()}


def _frozen_symbols() -> set[str]:
    import backtest.ev_chain_constants as frozen_constants
    import backtest.ou_ev_chain as frozen_chain

    return _uppercase_names(dir(frozen_constants)) | _uppercase_names(dir(frozen_chain))


# ---------------------------------------------------------------------------
# Fixture: a small synthetic corpus, seed predictions and a converter
# ---------------------------------------------------------------------------

_GAMES_PER_SEASON = 48
_SEED_SEASONS = (2017, 2018, 2019)
_CORPUS_SEASONS = (2020, 2021, 2022, 2023, 2024)
# Prior-only slopes that differ SHARPLY from the serving slope, so a serving-slope conversion
# is visibly different from the out-of-fold one.
_WALK_FORWARD_SLOPES = {2021: 0.10, 2022: 0.12, 2023: 0.20, 2024: 0.25}
_SERVING_SLOPE = 0.40


def _fixture() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """``(corpus_frame, predictions)`` shaped like the real derivation's inputs."""
    rng = np.random.default_rng(29)
    corpus_rows: list[dict[str, object]] = []
    predictions: dict[str, list[dict[str, object]]] = {"wp": [], "ats": [], "ou": []}
    for season in (*_SEED_SEASONS, *_CORPUS_SEASONS):
        for i in range(_GAMES_PER_SEASON):
            game_id = f"{season}_W{1 + i % 16:02d}_A{i}@H{i}"
            week = 1 + i % 16
            market_spread = float(rng.normal(0.0, 6.0))
            market_total = float(rng.normal(45.0, 4.0))
            margin = market_spread + float(rng.normal(0.0, 13.0))
            total = market_total + float(rng.normal(0.0, 13.0))
            p_home = 1.0 / (1.0 + math.exp(-0.15 * market_spread))
            predictions["wp"].append(
                {
                    "game_id": game_id,
                    "season": season,
                    "prediction": float(
                        np.clip(p_home + rng.normal(0, 0.05), 0.02, 0.98)
                    ),
                    "actual": float(margin > 0),
                }
            )
            predictions["ats"].append(
                {
                    "game_id": game_id,
                    "season": season,
                    "prediction": market_spread + float(rng.normal(0.0, 3.0)),
                    "actual": margin,
                }
            )
            predictions["ou"].append(
                {
                    "game_id": game_id,
                    "season": season,
                    "prediction": market_total + float(rng.normal(0.0, 3.0)),
                    "actual": total,
                }
            )
            if season in _CORPUS_SEASONS:
                lock = pd.Timestamp(f"{season}-10-01T22:00:00Z") + pd.Timedelta(days=i)
                corpus_rows.append(
                    {
                        "game_id": game_id,
                        "season": season,
                        "week": week,
                        "game_type": "REG",
                        "snapshot_ts": lock - pd.Timedelta(hours=2),
                        "lock": lock,
                        "created_at": lock,
                        "market_spread": market_spread,
                        "market_total": market_total,
                    }
                )
    frames = {target: pd.DataFrame(rows) for target, rows in predictions.items()}
    return pd.DataFrame(corpus_rows), frames


# ---------------------------------------------------------------------------
# The frozen record and the spent ledger are untouched
# ---------------------------------------------------------------------------


class TestTheFrozenRecordIsUntouched:
    def test_the_preregistration_pair_is_byte_unchanged_since_ee20773(self) -> None:
        assert _git("diff", SUPERSEDED_SHA, "HEAD", "--", *FROZEN_PAIR) == ""
        assert _git("status", "--porcelain", "--", *FROZEN_PAIR) == ""

    def test_the_anchor_still_resolves_to_the_recorded_commit(self) -> None:
        from tests.phase31_state import PRE_REGISTRATION_COMMIT

        anchor = _git("log", "-1", "--format=%H", "--", *FROZEN_PAIR)
        assert anchor == PRE_REGISTRATION_COMMIT
        assert anchor.startswith(SUPERSEDED_SHA)

    def test_the_spent_2025_ledger_is_untouched_and_completed(self) -> None:
        assert _git("status", "--porcelain", "--", LEDGER) == ""
        ledger = tomllib.loads((REPO_ROOT / LEDGER).read_text(encoding="utf-8"))
        assert ledger["state"] == "completed"
        assert ledger["hold_seasons"] == ["2025"]


# ---------------------------------------------------------------------------
# The corrected module
# ---------------------------------------------------------------------------


class TestTheCorrectedModule:
    def test_it_names_the_superseded_commit(self) -> None:
        corrected = _load(CORRECTED_MODULE)
        assert SUPERSEDED_SHA in (corrected.__doc__ or "")
        from tests.phase31_state import PRE_REGISTRATION_COMMIT

        assert corrected.SUPERSEDED_PREREGISTRATION_COMMIT == PRE_REGISTRATION_COMMIT

    def test_it_redeclares_no_frozen_symbol(self) -> None:
        corrected = _load(CORRECTED_MODULE)
        frozen = _frozen_symbols()
        assert len(frozen) > 20, "non-vacuity: the frozen symbol set must be populated"
        assert frozen & _uppercase_names(dir(corrected)) == set()

    def test_a_planted_redeclaration_is_caught(self) -> None:
        """Control: the same comparison DOES catch a copied frozen constant."""
        planted = {"CORRECTED_EV_FLOOR_BY_TARGET", "EV_FLOOR_GRID"}
        assert _frozen_symbols() & _uppercase_names(planted) == {"EV_FLOOR_GRID"}

    def test_the_replay_policy_is_carried_under_the_corrected_name_only(self) -> None:
        corrected = _load(CORRECTED_MODULE)
        assert "REPLAY_SNAPSHOT_TS_POLICY" not in dir(corrected)
        policy = corrected.CORRECTED_REPLAY_SNAPSHOT_TS_POLICY
        assert "day before" in policy.lower()
        assert "18:00" in policy
        assert "supersedes" in policy.lower()

    def test_every_floor_is_a_grid_value_or_a_named_refusal(self) -> None:
        from backtest.ou_ev_chain import EV_FLOOR_GRID

        corrected = _load(CORRECTED_MODULE)
        floors = corrected.CORRECTED_EV_FLOOR_BY_TARGET
        assert set(floors) == {"wp", "ats", "ou"}
        for target, floor in floors.items():
            if floor is None:
                assert target in corrected.CORRECTED_FLOOR_REFUSALS
                continue
            assert isinstance(floor, float) and math.isfinite(floor)
            assert floor in EV_FLOOR_GRID

    def test_the_old_values_are_published_beside_the_new(self) -> None:
        corrected = _load(CORRECTED_MODULE)
        assert set(corrected.SUPERSEDED_EV_FLOOR_BY_TARGET) == {"wp", "ats", "ou"}
        assert set(corrected.SUPERSEDED_FROZEN_SD_BY_TARGET) == {"wp", "ats", "ou"}

    def test_frozen_sd_is_ats_and_ou_only_and_wp_says_why(self) -> None:
        corrected = _load(CORRECTED_MODULE)
        sds = corrected.CORRECTED_FROZEN_SD_BY_TARGET
        assert sorted(t for t, v in sds.items() if v is not None) == ["ats", "ou"]
        assert sds["wp"] is None
        assert "D31-07" in corrected.CORRECTED_WP_NO_FROZEN_SD_REASON
        for target in ("ats", "ou"):
            assert math.isfinite(sds[target]) and sds[target] > 0

    def test_the_windows_are_the_honest_corpus(self) -> None:
        corrected = _load(CORRECTED_MODULE)
        assert corrected.CORRECTED_DERIVATION_SEASONS == (2020, 2021, 2022, 2023, 2024)
        assert corrected.CORRECTED_WP_DERIVATION_SEASONS == (2021, 2022, 2023, 2024)
        assert corrected.CORRECTED_WP_EXCLUDED_NO_PRIOR_FOLD > 0
        assert 2025 not in corrected.CORRECTED_BIAS_SEED_SEASONS
        assert max(corrected.CORRECTED_BIAS_SEED_SEASONS) < 2020


# ---------------------------------------------------------------------------
# The refusals
# ---------------------------------------------------------------------------


class TestTheRefusals:
    def test_a_too_small_pool_refuses_where_the_old_recipe_fell_back(self) -> None:
        """No grid floor admits a bet: Phase 31 froze EV_FLOOR_GRID[0]; this refuses."""
        from backtest.ou_ev_chain import EV_FLOOR_GRID

        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        frames = script.build_candidate_frames(
            corpus, predictions, _WALK_FORWARD_SLOPES
        )
        # A pool where the model agrees with the market exactly: no side, so no bet at any floor.
        thin = frames.frames["ats"].head(3).copy()
        thin["model_spread"] = thin["closing_spread"]
        fit_frame = script.fit_frame_for("ats", predictions, frames)
        chain_fit, gate = script.fit_target("ats", fit_frame)
        sweep = script.sweep_ev_floor("ats", thin, chain_fit, gate)

        assert sweep.best_t is None
        old_recipe_floor = (
            sweep.best_t if sweep.best_t is not None else float(EV_FLOOR_GRID[0])
        )
        assert old_recipe_floor == EV_FLOOR_GRID[0]

        with pytest.raises(script.InsufficientHonestDataForFloorError, match="ats"):
            script.choose_ev_floor("ats", sweep)
        floor, refusal = script.floor_or_refusal("ats", sweep)
        assert floor is None
        assert floor != old_recipe_floor
        assert refusal and "ats" in refusal

    def test_the_refusal_is_not_swallowed_by_the_common_catch_tuples(self) -> None:
        script = _load(SCRIPT_MODULE)
        error = script.InsufficientHonestDataForFloorError
        for swallowing in (ValueError, KeyError, RuntimeError, LookupError):
            assert not issubclass(error, swallowing)

    def test_a_2025_row_in_the_corpus_is_refused(self) -> None:
        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        planted = corpus.head(1).assign(season=2025, game_id="2025_W01_A0@H0")
        with pytest.raises(script.SpentHoldSeasonError, match="2025"):
            script.build_candidate_frames(
                pd.concat([corpus, planted], ignore_index=True),
                predictions,
                _WALK_FORWARD_SLOPES,
            )

    def test_a_2025_row_in_the_bias_seed_is_refused(self) -> None:
        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        frames = script.build_candidate_frames(
            corpus, predictions, _WALK_FORWARD_SLOPES
        )
        planted = predictions["ou"].head(1).assign(season=2025)
        predictions["ou"] = pd.concat([predictions["ou"], planted], ignore_index=True)
        with pytest.raises(script.SpentHoldSeasonError, match="2025"):
            script.fit_frame_for("ou", predictions, frames)

    def test_a_closing_line_column_in_the_fit_inputs_is_refused(self) -> None:
        from models.market_probability import ClosingLineInFitError

        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        planted = corpus.assign(closing_total=corpus["market_total"])
        with pytest.raises(ClosingLineInFitError, match="closing_total"):
            script.build_candidate_frames(planted, predictions, _WALK_FORWARD_SLOPES)

    def test_a_post_lock_line_in_the_fit_inputs_is_refused(self) -> None:
        from models.market_probability import ClosingLineInFitError

        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        late = corpus.copy()
        late.loc[0, "snapshot_ts"] = late.loc[0, "lock"] + pd.Timedelta(seconds=1)
        with pytest.raises(ClosingLineInFitError):
            script.build_candidate_frames(late, predictions, _WALK_FORWARD_SLOPES)


# ---------------------------------------------------------------------------
# The WP market side is out of fold
# ---------------------------------------------------------------------------


class TestTheWpMarketSideIsOutOfFold:
    def test_the_market_column_equals_oof_market_probability_row_for_row(self) -> None:
        from models.market_probability import oof_market_probability

        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        wp = script.build_candidate_frames(
            corpus, predictions, _WALK_FORWARD_SLOPES
        ).frames["wp"]
        expected = oof_market_probability(
            wp.assign(home_fav_margin=wp["market_spread"]), _WALK_FORWARD_SLOPES
        )
        np.testing.assert_array_equal(wp["market_prob"].to_numpy(), expected)

    def test_a_planted_serving_slope_conversion_differs(self) -> None:
        """Control: the equality above is load-bearing because the serving slope differs."""
        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        wp = script.build_candidate_frames(
            corpus, predictions, _WALK_FORWARD_SLOPES
        ).frames["wp"]
        serving = 1.0 / (1.0 + np.exp(-_SERVING_SLOPE * wp["market_spread"].to_numpy()))
        assert not np.allclose(wp["market_prob"].to_numpy(), serving)

    def test_the_first_owned_season_leaves_the_wp_sweep_only(self) -> None:
        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        frames = script.build_candidate_frames(
            corpus, predictions, _WALK_FORWARD_SLOPES
        )
        assert sorted(set(frames.frames["wp"]["season"])) == [2021, 2022, 2023, 2024]
        assert set(frames.wp_excluded["season"]) == {2020}
        assert set(frames.wp_excluded["reason"]) == {"no_prior_fold_converter"}
        for target in ("ats", "ou"):
            assert sorted(set(frames.frames[target]["season"])) == list(_CORPUS_SEASONS)

    def test_the_derivation_never_touches_a_serving_slope_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import models.blending as blending_module
        from models import market_probability
        from models.blending import MarketBlender

        def _explode(*_args: object, **_kwargs: object) -> None:
            msg = "a historical row was converted through a serving-slope path"
            raise ServingSlopeUsedOnHistoryError(msg)

        monkeypatch.setattr(market_probability, "market_home_win_probability", _explode)
        monkeypatch.setattr(blending_module, "market_home_win_probability", _explode)
        monkeypatch.setattr(MarketBlender, "_blend_wp_predictions", _explode)

        script = _load(SCRIPT_MODULE)
        corpus, predictions = _fixture()
        derivation = script.derive_chain(corpus, predictions, _WALK_FORWARD_SLOPES)
        assert set(derivation.floors) == {"wp", "ats", "ou"}
        assert derivation.candidate_seasons["wp"] == (2021, 2022, 2023, 2024)
        assert derivation.candidate_seasons["ats"] == _CORPUS_SEASONS
