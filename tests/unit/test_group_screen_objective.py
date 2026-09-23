"""The feature-group screen's OUTCOME-LOSS objective (Plan 33.2-22 Task 1, D33.2-15).

WHAT THIS MODULE GUARDS
-----------------------
The Phase-28/29/30 screen chose feature groups by paired CLOSING-LINE CLV:
``_walkforward_clv_series`` keeps only games with closing odds and reads the target's CLV
column, and ``run_signal_lift_screen`` loads ``data/silver/odds_snapshot.parquet`` whenever it
is not handed odds. Reused unchanged on corrected gold, a CLOSING line would choose which
feature families the corrected models are trained on -- a closing line feeding a FIT decision,
which is the post-lock leak D33.2-03 removes.

The replacement is each model's OWN out-of-sample outcome loss under its own primary metric:
WP per-game log loss with the trainer's clip, ATS and O/U per-game absolute error. No market
line of ANY timing enters it.

THE FOUR STRUCTURAL CONTROLS (the ``test_freeze_parse_single_source.py`` shape):

  * NON-VACUITY -- the refusal's column list is non-empty and its length is asserted, and the
    stubbed screen scores a non-zero paired count. An empty list or an empty grid would make
    every assertion below pass while checking nothing.
  * THE ASSERTION -- under ``objective="outcome_loss"`` every trainer call receives
    ``closing_odds_df=None``, no parquet is read at all, and the paired delta equals
    ``baseline_loss - candidate_loss`` recomputed here from the stub's own predictions.
  * THE PLANTED VIOLATION -- a gold frame carrying ``snapshot_spread`` raises; so does one
    carrying ``ml_home``; so does passing a non-None ``closing_odds_df``.
  * NO FALSE POSITIVE -- ``total_points`` (the O/U target) and
    ``home_off_rolling_avg_drive_start_yardline`` (which CONTAINS the word "line") pass.

Everything here runs on synthetic frames with stub trainers: no gold, no odds, no network.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pytest

from backtest import signal_lift
from backtest.diagnose import MIN_CLV_SAMPLE
from backtest.ev_chain_constants import HOLD_SEASONS_P31
from conf.season_partition import LATEST_COMPLETED_SEASON
from models.temporal import TemporalSplitConfig

# The injury column whose presence tells a stub trainer it is the CANDIDATE leg.
_INJURY_COLUMN = "home_qb_out_flag"
_INJURY_COLUMNS = ("home_qb_out_flag", "away_qb_out_flag")

_TARGET_LABEL_COLUMN = {
    "wp": "home_win",
    "ats": "home_margin",
    "ou": "total_points",
}


def _require(name: str) -> Any:
    """Return ``backtest.signal_lift.<name>``, asserting it exists.

    Written as an ASSERTION rather than a bare attribute access so the RED phase of this task
    fails on a message naming the planned behaviour, not on an AttributeError traceback.
    """
    value = getattr(signal_lift, name, None)
    assert value is not None, (
        f"backtest.signal_lift.{name} does not exist. Plan 33.2-22 Task 1 adds the "
        "outcome-loss objective, its named closing-line refusal and the derived "
        "2025-excluding config helper to the ONE screen."
    )
    return value


# ---------------------------------------------------------------------------
# Synthetic gold and the stub trainers
# ---------------------------------------------------------------------------


def _synthetic_gold(
    target: str,
    seasons: tuple[int, ...],
    *,
    games_per_season: int = 14,
    extra_columns: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a small synthetic gold matrix for one target.

    Deterministic: every value is a closed-form function of the row index, so the expected
    losses below are recomputable without a seed.
    """
    rows: list[dict[str, Any]] = []
    index = 0
    for season in seasons:
        for game in range(games_per_season):
            index += 1
            row: dict[str, Any] = {
                "game_id": f"{season}_{game:02d}_HOME@AWAY",
                "season": season,
                "week": (game % 18) + 1,
                "home_team": f"H{game:02d}",
                "away_team": f"A{game:02d}",
                "home_win": index % 2,
                "home_margin": float((index % 21) - 10),
                "total_points": float(30 + (index % 25)),
                "feature_one": float(index % 7),
                "feature_two": float((index * 3) % 11),
                _INJURY_COLUMNS[0]: float(index % 2),
                _INJURY_COLUMNS[1]: float((index + 1) % 2),
            }
            for name, value in (extra_columns or {}).items():
                row[name] = value
            rows.append(row)
    frame = pd.DataFrame(rows)
    # Keep only the label this target's trainer would read, so the frame is target-shaped.
    drop = [
        c for c in _TARGET_LABEL_COLUMN.values() if c != _TARGET_LABEL_COLUMN[target]
    ]
    return frame.drop(columns=drop)


class _StubTrainerBase:
    """A trainer stand-in that RECORDS its call and returns fixed holdout predictions."""

    calls: ClassVar[list[dict[str, Any]]] = []
    frames: ClassVar[dict[tuple[str, str], pd.DataFrame]] = {}
    target: ClassVar[str] = ""

    def __init__(self, config: TemporalSplitConfig) -> None:
        self.config = config
        self.feature_names: list[str] = []

    def train_and_evaluate(
        self,
        features_df: pd.DataFrame,
        closing_odds_df: pd.DataFrame | None = None,
        tune: bool = True,
    ) -> dict[str, Any]:
        leg = "candidate" if _INJURY_COLUMN in features_df.columns else "baseline"
        _StubTrainerBase.calls.append(
            {
                "target": self.target,
                "leg": leg,
                "closing_odds_df": closing_odds_df,
                "tune": tune,
                "seasons": sorted(int(s) for s in features_df["season"].unique()),
                "columns": list(features_df.columns),
            }
        )
        holdout = self._holdout(features_df, leg)
        _StubTrainerBase.frames[(self.target, leg)] = holdout
        selected = ["feature_one", "feature_two"]
        if leg == "candidate":
            selected = [*selected, *_INJURY_COLUMNS]
        self.feature_names = selected
        return {
            "season_results": [],
            "feature_names": selected,
            "best_params": {},
            "clv_results": None,
            "holdout_predictions": holdout,
            "metadata": {},
        }

    def _holdout(self, features_df: pd.DataFrame, leg: str) -> pd.DataFrame:
        holdout_seasons = set(self.config.holdout_seasons)
        rows = features_df[features_df["season"].isin(holdout_seasons)].reset_index(
            drop=True
        )
        actual = rows[_TARGET_LABEL_COLUMN[self.target]].to_numpy(dtype=float)
        position = np.arange(len(rows), dtype=float)
        if self.target == "wp":
            # Two DIFFERENT, varying probability sequences, so the paired delta has variance
            # and the significance primitive has a p-value to return.
            base = 0.30 + 0.02 * (position % 9)
            cand = 0.55 + 0.03 * (position % 7)
            prediction = cand if leg == "candidate" else base
        else:
            offset = 4.0 + (position % 5)
            shrink = 1.0 + (position % 3)
            prediction = actual - shrink if leg == "candidate" else actual + offset
        return pd.DataFrame(
            {
                "game_id": rows["game_id"].to_numpy(),
                "season": rows["season"].to_numpy(),
                "prediction": prediction,
                "actual": actual,
            }
        )


def _stub_trainer_for() -> dict[str, type]:
    """A fresh ``_TRAINER_FOR`` mapping of per-target stub classes."""
    _StubTrainerBase.calls = []
    _StubTrainerBase.frames = {}
    return {
        target: type(
            f"_Stub{target.upper()}Trainer", (_StubTrainerBase,), {"target": target}
        )
        for target in ("wp", "ats", "ou")
    }


def _expected_loss(target: str, frame: pd.DataFrame) -> pd.Series:
    """Recompute the per-game loss HERE, from the stub's own predictions and actuals."""
    prediction = frame["prediction"].to_numpy(dtype=float)
    actual = frame["actual"].to_numpy(dtype=float)
    if target == "wp":
        low, high = 1e-7, 1.0 - 1e-7
        clipped = np.clip(prediction, low, high)
        loss = -(actual * np.log(clipped) + (1.0 - actual) * np.log(1.0 - clipped))
    else:
        loss = np.abs(prediction - actual)
    return pd.Series(loss, index=frame["game_id"].to_numpy())


def _no_parquet(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make ANY parquet read an immediate failure, so 'never read the odds path' is literal."""

    def _forbidden(*args: Any, **kwargs: Any) -> pd.DataFrame:
        message = (
            f"the outcome-loss screen read a parquet file: {args!r}. Under "
            "objective='outcome_loss' the odds loader must never be reached."
        )
        raise AssertionError(message)

    monkeypatch.setattr(pd, "read_parquet", _forbidden)


_CONFIG = TemporalSplitConfig(
    train_seasons=[2018, 2019],
    hp_val_seasons=[2020],
    holdout_seasons=[2021, 2022],
)


def _gold_by_target(**kwargs: Any) -> dict[str, pd.DataFrame]:
    seasons = (2018, 2019, 2020, 2021, 2022)
    return {t: _synthetic_gold(t, seasons, **kwargs) for t in ("wp", "ats", "ou")}


# ---------------------------------------------------------------------------
# NON-VACUITY
# ---------------------------------------------------------------------------


class TestNonVacuity:
    """An empty column list or an empty grid would make every check below pass."""

    def test_the_closing_line_column_list_is_non_empty_and_its_length_is_pinned(
        self,
    ) -> None:
        columns = tuple(_require("CLOSING_LINE_COLUMNS"))
        assert len(columns) == 12, (
            "CLOSING_LINE_COLUMNS must name every closing-line column the refusal knows by "
            f"name; got {len(columns)}: {columns}"
        )
        assert all(isinstance(name, str) and name for name in columns)

    def test_the_two_objectives_are_declared_and_distinct(self) -> None:
        objectives = tuple(_require("SCREEN_OBJECTIVES"))
        assert objectives == ("closing_clv", "outcome_loss"), objectives

    def test_the_stubbed_screen_scores_a_non_zero_paired_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", _stub_trainer_for())
        _no_parquet(monkeypatch)
        result = signal_lift.run_signal_lift_screen(
            gold_by_target=_gold_by_target(),
            config=_CONFIG,
            groups=("injury",),
            baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
            objective="outcome_loss",
        )
        for target in ("wp", "ats", "ou"):
            cell = result["groups"]["injury"]["per_target"][target]
            assert cell["n_paired"] >= MIN_CLV_SAMPLE, (
                f"the {target} cell scored {cell['n_paired']} paired games; a cell below "
                f"MIN_CLV_SAMPLE={MIN_CLV_SAMPLE} measures nothing and cannot carry an "
                "assertion about the objective"
            )


# ---------------------------------------------------------------------------
# THE ASSERTION
# ---------------------------------------------------------------------------


class TestTheOutcomeLossObjective:
    """No market line of any timing, and the delta is the loss improvement."""

    def test_every_trainer_call_receives_no_closing_odds_frame(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", _stub_trainer_for())
        _no_parquet(monkeypatch)
        signal_lift.run_signal_lift_screen(
            gold_by_target=_gold_by_target(),
            config=_CONFIG,
            groups=("injury",),
            baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
            objective="outcome_loss",
        )
        assert _StubTrainerBase.calls, "no trainer was called at all"
        for call in _StubTrainerBase.calls:
            assert call["closing_odds_df"] is None, (
                f"the {call['target']} {call['leg']} leg was handed a closing-odds frame "
                "under objective='outcome_loss'"
            )

    @pytest.mark.parametrize("target", ["wp", "ats"])
    def test_the_paired_delta_is_the_hand_computed_loss_improvement(
        self, target: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """delta == baseline_loss - candidate_loss, recomputed here from the stub's frames."""
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", _stub_trainer_for())
        _no_parquet(monkeypatch)
        result = signal_lift.run_signal_lift_screen(
            gold_by_target=_gold_by_target(),
            config=_CONFIG,
            targets=(target,),
            groups=("injury",),
            baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
            objective="outcome_loss",
        )
        baseline = _expected_loss(target, _StubTrainerBase.frames[(target, "baseline")])
        candidate = _expected_loss(
            target, _StubTrainerBase.frames[(target, "candidate")]
        )
        expected = (baseline - candidate).to_numpy()

        cell = result["groups"]["injury"]["per_target"][target]
        assert cell["n_paired"] == len(expected)
        assert cell["delta_mean"] == pytest.approx(float(np.mean(expected))), (
            "the paired delta must be baseline_loss - candidate_loss, so a POSITIVE delta "
            "still means the group helped and the frozen rule's KEEP direction is unchanged"
        )

    def test_the_result_names_its_objective_and_the_loss_it_measured(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", _stub_trainer_for())
        _no_parquet(monkeypatch)
        result = signal_lift.run_signal_lift_screen(
            gold_by_target=_gold_by_target(),
            config=_CONFIG,
            groups=("injury",),
            baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
            objective="outcome_loss",
        )
        assert result["objective"] == "outcome_loss"
        anchor = result["objective_anchor"]
        assert "loss" in anchor.lower(), anchor
        assert "clv" not in anchor.lower(), (
            f"the objective anchor must not invite a CLV reading: {anchor!r}"
        )
        for target, expected_metric in (
            ("wp", "log_loss"),
            ("ats", "absolute_error"),
            ("ou", "absolute_error"),
        ):
            cell = result["groups"]["injury"]["per_target"][target]
            assert cell["metric"] == expected_metric
            assert cell["clv_column"] == expected_metric, (
                "under outcome_loss the cell's metric field must NAME the loss; leaving a "
                "CLV column name there would render a CLV label onto a loss measurement"
            )


class TestTheDefaultObjectiveIsUnchanged:
    """Every existing caller keeps the closing-CLV path byte-for-byte."""

    def test_the_default_is_closing_clv(self) -> None:
        parameters = inspect.signature(signal_lift.run_signal_lift_screen).parameters
        assert "objective" in parameters, (
            "run_signal_lift_screen has no objective parameter yet"
        )
        assert parameters["objective"].default == "closing_clv"

    def test_a_call_with_no_objective_takes_the_closing_clv_path(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        trainers = _stub_trainer_for()
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", trainers)
        odds = pd.DataFrame({"game_id": ["x"], "ml_home": [-110], "ml_away": [-110]})
        result = signal_lift.run_signal_lift_screen(
            gold_by_target=_gold_by_target(),
            closing_odds_df=odds,
            config=_CONFIG,
            groups=("injury",),
            baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
        )
        assert result["objective"] == "closing_clv"
        assert _StubTrainerBase.calls, "no trainer was called"
        for call in _StubTrainerBase.calls:
            assert call["closing_odds_df"] is odds, (
                "the default path must keep handing the trainer the closing-odds frame"
            )
        for target in ("wp", "ats", "ou"):
            cell = result["groups"]["injury"]["per_target"][target]
            assert cell["clv_column"] == signal_lift.CLV_COLUMN_FOR[target]


# ---------------------------------------------------------------------------
# THE PLANTED VIOLATION
# ---------------------------------------------------------------------------


class TestTheClosingLineRefusal:
    """A closing line reaching the screen is a refusal BY NAME, not a convention."""

    def test_the_refusal_is_not_caught_by_any_degrade_quietly_tuple(self) -> None:
        error = _require("ClosingLineInScreenError")
        assert issubclass(error, Exception)
        assert not issubclass(error, ValueError | KeyError | TypeError), (
            "ClosingLineInScreenError must not inherit ValueError / KeyError / TypeError: "
            "several call sites in this repository catch that tuple and degrade quietly, "
            "and a closing-line refusal degraded into 'carry on' is the leak it exists to "
            "stop (the data/sealed_probe_log.py reasoning)"
        )

    @pytest.mark.parametrize("planted", ["snapshot_spread", "ml_home"])
    def test_a_planted_closing_line_column_in_gold_raises(
        self, planted: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = _require("ClosingLineInScreenError")
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", _stub_trainer_for())
        _no_parquet(monkeypatch)
        with pytest.raises(error) as raised:
            signal_lift.run_signal_lift_screen(
                gold_by_target=_gold_by_target(extra_columns={planted: 1.5}),
                config=_CONFIG,
                groups=("injury",),
                baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
                objective="outcome_loss",
            )
        assert planted in str(raised.value), (
            "the refusal must NAME the offending column, so the reader knows what to remove"
        )

    def test_passing_a_closing_odds_frame_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        error = _require("ClosingLineInScreenError")
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", _stub_trainer_for())
        _no_parquet(monkeypatch)
        with pytest.raises(error):
            signal_lift.run_signal_lift_screen(
                gold_by_target=_gold_by_target(),
                closing_odds_df=pd.DataFrame({"game_id": ["x"], "ml_home": [-110]}),
                config=_CONFIG,
                groups=("injury",),
                baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
                objective="outcome_loss",
            )

    def test_a_closing_line_arriving_through_the_trainer_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same check applies to each leg's per-game frame, not only to gold."""
        error = _require("ClosingLineInScreenError")
        trainers = _stub_trainer_for()

        class _Leaky(trainers["wp"]):  # type: ignore[misc, valid-type]
            def train_and_evaluate(self, features_df, closing_odds_df=None, tune=True):
                out = super().train_and_evaluate(features_df, closing_odds_df, tune)
                leaked = out["holdout_predictions"].copy()
                leaked["line_clv"] = 0.0
                out["holdout_predictions"] = leaked
                return out

        trainers["wp"] = _Leaky
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", trainers)
        _no_parquet(monkeypatch)
        with pytest.raises(error) as raised:
            signal_lift.run_signal_lift_screen(
                gold_by_target=_gold_by_target(),
                config=_CONFIG,
                targets=("wp",),
                groups=("injury",),
                baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
                objective="outcome_loss",
            )
        assert "line_clv" in str(raised.value)


# ---------------------------------------------------------------------------
# NO FALSE POSITIVE
# ---------------------------------------------------------------------------


class TestNoFalsePositive:
    """A column that merely CONTAINS the word is not a closing line."""

    def test_the_target_and_a_drive_start_yardline_column_are_not_flagged(self) -> None:
        detector = _require("closing_line_columns_in")
        frame = pd.DataFrame(
            {
                "game_id": ["g"],
                "total_points": [44.0],
                "home_off_rolling_avg_drive_start_yardline": [27.4],
                "rolling_total_epa": [0.1],
            }
        )
        assert detector(frame) == [], (
            "membership must be EXACT on CLOSING_LINE_COLUMNS and suffix-exact on the market "
            "predicate, never a substring test: 'drive_start_yardline' contains 'line' and "
            "'total_points' contains 'total'"
        )

    def test_the_whole_synthetic_gold_frame_is_not_flagged(self) -> None:
        detector = _require("closing_line_columns_in")
        for target in ("wp", "ats", "ou"):
            frame = _synthetic_gold(target, (2018, 2019))
            assert detector(frame) == [], target

    def test_the_detector_does_flag_a_real_closing_line(self) -> None:
        detector = _require("closing_line_columns_in")
        frame = pd.DataFrame(
            {"game_id": ["g"], "snapshot_total": [44.0], "ml_away": [110.0]}
        )
        assert detector(frame) == ["ml_away", "snapshot_total"]


# ---------------------------------------------------------------------------
# THE SPENT 2025 HOLD
# ---------------------------------------------------------------------------


class TestTheSpentHoldIsNeverScored:
    """The screen's config is DERIVED with HOLD_SEASONS_P31 removed, never typed."""

    def test_the_derived_config_excludes_the_spent_hold_and_any_live_season(
        self,
    ) -> None:
        derive = _require("screen_config_excluding_spent_hold")
        gold = _gold_by_target()
        # A full corpus, including the spent hold and a live season.
        seasons = tuple(range(2002, 2027))
        gold = {t: _synthetic_gold(t, seasons, games_per_season=3) for t in gold}
        config = derive(gold)
        assert isinstance(config, TemporalSplitConfig)
        scored = set(config.all_seasons)
        assert not scored & set(HOLD_SEASONS_P31), (
            f"the derived config scores the spent hold {sorted(scored & set(HOLD_SEASONS_P31))}"
        )
        assert max(scored) <= LATEST_COMPLETED_SEASON, (
            "the derived config reaches past the last completed season"
        )
        assert config.holdout_seasons == [2023, 2024], config.holdout_seasons
        assert config.hp_val_seasons == [2022], config.hp_val_seasons
        assert config.train_seasons[0] == 2002 and config.train_seasons[-1] == 2021

    def test_no_2025_row_reaches_a_trainer(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(signal_lift, "_TRAINER_FOR", _stub_trainer_for())
        _no_parquet(monkeypatch)
        seasons = tuple(range(2002, 2027))
        gold = {
            t: _synthetic_gold(t, seasons, games_per_season=3)
            for t in ("wp", "ats", "ou")
        }
        signal_lift.run_signal_lift_screen(
            gold_by_target=gold,
            groups=("injury",),
            baseline_exclude_groups=signal_lift.ALL_REGISTERED_GROUPS,
            objective="outcome_loss",
        )
        assert _StubTrainerBase.calls
        for call in _StubTrainerBase.calls:
            forbidden = set(call["seasons"]) & set(HOLD_SEASONS_P31)
            assert not forbidden, (
                f"the {call['target']} {call['leg']} leg was handed the spent hold season(s) "
                f"{sorted(forbidden)}; choosing feature groups by their 2025 score would make "
                "2025 a selection criterion"
            )
            assert max(call["seasons"]) <= LATEST_COMPLETED_SEASON, call["seasons"]


# ---------------------------------------------------------------------------
# THE TRAINER SEAM
# ---------------------------------------------------------------------------


def _trainer_frame(target: str) -> pd.DataFrame:
    return _synthetic_gold(target, (2018, 2019, 2020, 2021, 2022), games_per_season=40)


class TestTheTrainersReturnTheirHoldoutPredictions:
    """No caller should have to hand a trainer a closing line to get its own predictions."""

    @pytest.mark.parametrize("target", ["wp", "ats", "ou"])
    def test_holdout_predictions_are_returned_with_no_closing_odds_frame(
        self, target: str
    ) -> None:
        from models.trainers.ats_trainer import ATSTrainer
        from models.trainers.ou_trainer import OUTrainer
        from models.trainers.wp_trainer import WPTrainer

        trainer_cls = {"wp": WPTrainer, "ats": ATSTrainer, "ou": OUTrainer}[target]
        result = trainer_cls(config=_CONFIG).train_and_evaluate(
            _trainer_frame(target), None, tune=False
        )
        assert "holdout_predictions" in result, (
            f"{trainer_cls.__name__}.train_and_evaluate returns no holdout_predictions, so "
            "an outcome-based screen cannot be computed without handing it a closing line"
        )
        frame = result["holdout_predictions"]
        assert list(frame.columns) == ["game_id", "season", "prediction", "actual"]
        assert len(frame) > 0
        assert set(frame["season"]) == set(_CONFIG.holdout_seasons)
        assert frame["prediction"].notna().all()
        assert result["clv_results"] is None

    def test_clv_results_are_unchanged_when_odds_are_passed(self) -> None:
        """The CLV branch keeps consuming the same per-game frame it consumes today."""
        from models.trainers.ats_trainer import ATSTrainer

        frame = _trainer_frame("ats")
        holdout_ids = frame.loc[
            frame["season"].isin(_CONFIG.holdout_seasons), "game_id"
        ]
        odds = pd.DataFrame(
            {
                "game_id": holdout_ids.to_numpy(),
                "ml_home": -110.0,
                "ml_away": -110.0,
                "spread": -2.5,
                "total": 44.5,
            }
        )
        result = ATSTrainer(config=_CONFIG).train_and_evaluate(frame, odds, tune=False)
        clv = result["clv_results"]
        assert clv is not None
        for column in ("game_id", "model_prob", "model_spread", "actual", "season"):
            assert column in clv.columns, column
        assert "prediction" not in clv.columns, (
            "the CLV frame must be exactly what it is today: adding a column to it would "
            "change clv_results for every existing caller"
        )
        assert bool(clv["has_closing_odds"].all())
