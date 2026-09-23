"""Unit tests for the blend's data module (models/blending_data.py).

WHAT MOVED, AND WHY (Plan 33.2-24)
----------------------------------
This module used to test two things the blend re-tune retires, and every retired test is
accounted for here rather than dropped silently.

**Task 1 -- the live nflverse loader (nine tests).** ``load_tuning_period_data`` used to fetch
nflverse schedules live and tune the blend on 2010-2017 CLOSING lines. That loader is RETIRED
by D33.2-03 / D33.2-10 -- a closing line did not exist at a game's lock:

* column mapping / moneyline mapping / abbreviation normalization / pandas type: the new
  loader reads the owned ``odds_timeline``, which has no moneyline and already stores
  canonical game ids; its columns are pinned by
  ``tests/integration/test_blend_tuning_prelock_only.py``;
* NaN dropping: carried forward as "no missing line is ever filled" -- a game with no
  pre-lock line is EXCLUDED and returned, pinned there by the prohibition test;
* the regular-season filter: deliberately NOT carried forward. The owned corpus keeps the
  six postseason games that have a pre-lock line; the models predict postseason games too,
  and dropping them would discard honest evidence (pinned below);
* ``TUNING_SEASONS`` / ``get_tuning_period_games``: the 2010-2017 window and its game-info
  wrapper have no reader left.

**Task 2 -- the noise profile (nine tests).** ``extract_noise_profile`` measured per-week
model-versus-CLOSING-market errors so the retired week-varying tuner could SYNTHESIZE model
predictions for 2010-2017. The blend is now tuned on the corrected models' REAL walk-forward
predictions, joined to the owned pre-lock corpus by ``build_tuning_frames``, so each intent
is carried onto that builder or retired by ruling:

* "returns all three targets" -> ``test_every_target_gets_a_frame``;
* "the error is computed from the right columns" -> ``test_each_target_reads_its_own_columns``
  (model prediction, market opinion and realized outcome land in the right named columns);
* "a missing baselines directory fails fast, no fallback (D-17)" ->
  ``test_a_game_missing_a_prediction_refuses_rather_than_dropping``;
* "the path is configurable" -> the converter slopes are an ARGUMENT, never looked up
  (``test_the_slopes_are_an_argument_not_a_lookup``);
* "playoff weeks are filtered" -> RETIRED, deliberately: the corpus keeps a postseason game
  that has a pre-lock line (``test_a_postseason_game_with_a_line_is_kept``);
* per-week mean/std, sparse-week blending toward the season, and the W01/W1 game-id variants
  -> RETIRED BY RULING (D33.2-10): there is no per-week anything left to compute, and game ids
  come from canonical silver.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from models.blending import BLEND_TUNING_COLUMNS
from models.blending_data import (
    EXCLUSION_NO_PRIOR_FOLD_CONVERTER,
    EXCLUSION_REASONS,
    TuningCorpusError,
    build_tuning_frames,
)

_SLOPES = {"2021": 0.14, "2022": 0.15}
_IDS = ["2021_W01_A@B", "2021_W02_C@D", "2022_W19_E@F"]
_SEASONS = [2021, 2021, 2022]


def _corpus() -> pd.DataFrame:
    """A small owned pre-lock corpus: two regular-season 2021 games, one 2022 wild card."""
    return pd.DataFrame(
        {
            "game_id": _IDS,
            "season": _SEASONS,
            "week": [1, 2, 19],
            "game_type": ["REG", "REG", "WC"],
            "market_spread": [3.0, -7.0, 1.5],
            "market_total": [44.5, 51.0, 47.0],
        }
    )


def _frame(predictions: list[float], actuals: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": _IDS,
            "season": _SEASONS,
            "prediction": predictions,
            "actual": actuals,
        }
    )


def _predictions() -> dict[str, pd.DataFrame]:
    return {
        "wp": _frame([0.6, 0.3, 0.55], [1, 0, 1]),
        "ats": _frame([2.0, -5.0, 0.5], [7.0, -10.0, 3.0]),
        "ou": _frame([45.0, 49.0, 46.0], [41.0, 55.0, 44.0]),
    }


class TestTheTuningFrames:
    def test_every_target_gets_a_frame(self) -> None:
        frames = build_tuning_frames(_corpus(), _predictions(), _SLOPES)
        assert set(frames.frames) == {"wp", "ats", "ou"}
        for frame in frames.frames.values():
            assert len(frame) == 3

    def test_each_target_reads_its_own_columns(self) -> None:
        frames = build_tuning_frames(_corpus(), _predictions(), _SLOPES)
        for target, columns in BLEND_TUNING_COLUMNS.items():
            assert set(columns) <= set(frames.frames[target].columns)
        ats = frames.frames["ats"].set_index("game_id")
        assert ats.loc["2021_W02_C@D", "model_spread"] == -5.0
        assert ats.loc["2021_W02_C@D", "market_spread"] == -7.0
        assert ats.loc["2021_W02_C@D", "home_margin"] == -10.0
        ou = frames.frames["ou"].set_index("game_id")
        assert ou.loc["2021_W01_A@B", "market_total"] == 44.5
        assert ou.loc["2021_W01_A@B", "total_points"] == 41.0
        wp = frames.frames["wp"].set_index("game_id")
        assert wp.loc["2021_W01_A@B", "model_prob"] == 0.6
        assert wp.loc["2021_W01_A@B", "home_win"] == 1

    def test_the_wp_market_is_a_probability_on_the_right_side(self) -> None:
        """A home favourite (positive home-margin spread) reads above 0.5, an underdog below."""
        frames = build_tuning_frames(_corpus(), _predictions(), _SLOPES)
        wp = frames.frames["wp"].set_index("game_id")
        assert wp.loc["2021_W01_A@B", "market_prob_oof"] > 0.5
        assert wp.loc["2021_W02_C@D", "market_prob_oof"] < 0.5
        assert wp.loc["2021_W01_A@B", "market_prob_oof"] == pytest.approx(
            1.0 / (1.0 + np.exp(-0.14 * 3.0))
        )

    def test_a_game_missing_a_prediction_refuses_rather_than_dropping(self) -> None:
        predictions = _predictions()
        predictions["ou"] = predictions["ou"].iloc[:2]
        with pytest.raises(TuningCorpusError, match="2022_W19_E@F"):
            build_tuning_frames(_corpus(), predictions, _SLOPES)

    def test_a_missing_target_refuses(self) -> None:
        predictions = _predictions()
        del predictions["ats"]
        with pytest.raises(TuningCorpusError, match="ats"):
            build_tuning_frames(_corpus(), predictions, _SLOPES)

    def test_a_season_disagreement_refuses(self) -> None:
        predictions = _predictions()
        predictions["wp"].loc[0, "season"] = 2020
        with pytest.raises(TuningCorpusError, match="season"):
            build_tuning_frames(_corpus(), predictions, _SLOPES)

    def test_the_slopes_are_an_argument_not_a_lookup(self) -> None:
        """Different slopes give different WP market columns -- nothing is resolved from disk."""
        low = build_tuning_frames(
            _corpus(), _predictions(), {"2021": 0.05, "2022": 0.05}
        )
        high = build_tuning_frames(
            _corpus(), _predictions(), {"2021": 0.30, "2022": 0.30}
        )
        assert not np.allclose(
            low.frames["wp"]["market_prob_oof"], high.frames["wp"]["market_prob_oof"]
        )

    def test_a_season_with_no_prior_fold_leaves_wp_only(self) -> None:
        frames = build_tuning_frames(_corpus(), _predictions(), {"2022": 0.15})
        assert list(frames.frames["wp"]["game_id"]) == ["2022_W19_E@F"]
        assert set(frames.excluded["reason"]) == {EXCLUSION_NO_PRIOR_FOLD_CONVERTER}
        assert len(frames.excluded) == 2
        assert len(frames.frames["ats"]) == 3
        assert len(frames.frames["ou"]) == 3

    def test_a_postseason_game_with_a_line_is_kept(self) -> None:
        frames = build_tuning_frames(_corpus(), _predictions(), _SLOPES)
        for frame in frames.frames.values():
            assert "2022_W19_E@F" in set(frame["game_id"])

    def test_the_exclusion_vocabulary_is_the_two_published_classes(self) -> None:
        assert EXCLUSION_REASONS == ("no_prelock_line", "no_prior_fold_converter")
