"""Live tests for the add-one-in signal-lift screen (SIG-05 / SC5).

Fills the Plan 28-01 Wave-0 scaffold. Covers four invariants:

  1. The D-05 keep/drop rule (``decide_group_keep``) -- a pure-logic test of the keep-if-positive-
     on->=1-target-AND-not-vetoed-on-ANY-target rule, including the per-target veto and the
     no-positive drop.
  2. The feature-group column-selection helper (``select_group_columns``) -- the baseline leg drops
     every Phase-28 new column; each candidate leg re-admits exactly that group's columns; the
     unrelated ``snapshot_*`` odds columns are NEVER mis-classified as snap features; the helper
     never mutates its input (HARD BOUNDARY, review #3).
  3. The IN-PROCESS walk-forward seam (review #2/#3) -- the screen drives
     ``BaseTrainer.train_and_evaluate(tune=False)`` and does NOT call ``score_deployed_artifacts``
     as the lift anchor (a spy + a poisoned ``score_deployed_artifacts``).
  4. Determinism (the ``ou_divergence.py`` anti-rot shape) -- two runs over the same fixture gold
     produce identical structured output (the guard the D-20 SIGNAL-LIFT-READOUT.md doc-drift test
     relies on).

The fixture gold is a small synthetic frame so the real trainers run fast; WP (deterministic
LogReg) anchors the determinism + seam runs. ASCII only, no emoji (CLAUDE.md).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backtest import signal_lift
from backtest.signal_lift import (
    decide_group_keep,
    group_columns,
    phase28_new_columns,
    run_signal_lift_screen,
    select_group_columns,
)
from models.temporal import TemporalSplitConfig

# ---------------------------------------------------------------------------
# Synthetic fixtures
# ---------------------------------------------------------------------------


def _make_fixture_gold(seed: int = 7) -> pd.DataFrame:
    """Build a small synthetic widened-gold frame spanning four seasons.

    Carries the ID/target columns the trainers require, two baseline numeric features, and one
    representative column per Phase-28 group (snap / injury / situational) plus the ``snapshot_*``
    decoy columns that must NOT be mis-classified as snap features.
    """
    rng = np.random.default_rng(seed)
    # The 11 ADDED injury columns (CR-01 fixture widening) draw from an
    # INDEPENDENT generator so they never perturb the main ``rng`` stream that
    # produces the baseline / snap / situational values. This keeps the
    # snap-group determinism run byte-identical to the pre-widening fixture
    # (only ``home_injury_coverage`` stays on the main stream, exactly as before).
    inj_rng = np.random.default_rng(seed + 1000)
    seasons = [2018, 2019, 2020, 2021]
    n_per_season = 40
    rows: list[dict] = []
    gid = 0
    for season in seasons:
        for i in range(n_per_season):
            home_score = int(rng.integers(10, 35))
            away_score = int(rng.integers(10, 35))
            feat_a = float(rng.normal())
            feat_b = float(rng.normal())
            rows.append(
                {
                    "game_id": f"{season}_{gid:04d}",
                    "season": season,
                    "week": (i % 18) + 1,
                    "home_team": "AAA",
                    "away_team": "BBB",
                    "home_score": home_score,
                    "away_score": away_score,
                    "home_win": int(home_score > away_score),
                    "home_margin": float(home_score - away_score),
                    "total_points": float(home_score + away_score),
                    # baseline numeric features
                    "feat_a": feat_a,
                    "feat_b": feat_b,
                    # decoy odds-snapshot columns (carry "snap" substring -- NOT snap features)
                    "snapshot_spread": float(rng.normal()),
                    "snapshot_total": float(rng.normal()),
                    # one column per Phase-28 group (snap / situational) plus the
                    # FULL home_/away_ injury column set (CR-01: the buggy
                    # "injury" substring predicate matched only *_injury_coverage,
                    # so a single-injury-column fixture could not catch it).
                    "home_rolling_snap_share_qb": float(rng.normal()),
                    "away_snap_continuity": float(rng.normal()),
                    # home_injury_coverage stays on the MAIN rng (original draw);
                    # the other 11 injury columns come from inj_rng so the main
                    # stream -- and thus the snap-group determinism run -- is
                    # unchanged from the pre-widening fixture.
                    "home_injury_coverage": float(rng.normal()),
                    "away_injury_coverage": float(inj_rng.integers(0, 2)),
                    "home_qb_out_flag": float(inj_rng.integers(0, 2)),
                    "away_qb_out_flag": float(inj_rng.integers(0, 2)),
                    "home_backup_quality_delta": float(inj_rng.normal()),
                    "away_backup_quality_delta": float(inj_rng.normal()),
                    "home_availability_fraction": float(inj_rng.uniform(0.5, 1.0)),
                    "away_availability_fraction": float(inj_rng.uniform(0.5, 1.0)),
                    "home_availability_coverage": float(inj_rng.integers(0, 2)),
                    "away_availability_coverage": float(inj_rng.integers(0, 2)),
                    "home_date_modified_coverage": float(inj_rng.integers(0, 2)),
                    "away_date_modified_coverage": float(inj_rng.integers(0, 2)),
                    "home_off_bye": int(rng.integers(0, 2)),
                    "away_look_ahead_spot": int(rng.integers(0, 2)),
                }
            )
            gid += 1
    return pd.DataFrame(rows)


def _make_fixture_odds(gold: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    """Build a closing-odds frame keyed to the fixture gold's game_ids (all games have odds)."""
    rng = np.random.default_rng(seed)
    n = len(gold)
    return pd.DataFrame(
        {
            "game_id": gold["game_id"].to_numpy(),
            "ml_home": rng.choice([-150, -120, 110, 130], size=n).astype(float),
            "ml_away": rng.choice([-140, -110, 120, 140], size=n).astype(float),
            "spread": rng.normal(0, 3, size=n),
            "total": rng.normal(44, 4, size=n),
        }
    )


# The Phase-29 line-movement family (D-09), game-level (no home_/away_ prefix). Added by a
# SEPARATE fixture builder on a SEPARATE generator so the Phase-28 fixture above is untouched and
# every pre-existing Phase-28 assertion keeps measuring exactly what it measured before.
_LINE_MOVEMENT_COLUMNS = (
    "line_movement_coverage",
    "opening_total",
    "total_drift",
    "total_drift_dir",
    "total_late_drift",
    "total_abs_travel",
    "total_reversals",
    "total_range",
    "opening_spread",
    "spread_drift",
    "spread_drift_dir",
    "spread_late_drift",
    "spread_abs_travel",
    "spread_reversals",
    "spread_range",
)


def _make_fixture_gold_with_line_movement(seed: int = 7) -> pd.DataFrame:
    """The Phase-28 fixture widened with the Phase-29 line-movement family.

    Mirrors the real Plan 29-06 gold: 15 game-level line-movement columns sitting alongside the
    Phase-28 groups AND the ``snapshot_*`` / ``*_movement`` decoys that must never be swept into
    the line_movement group.
    """
    gold = _make_fixture_gold(seed)
    rng = np.random.default_rng(seed + 5000)
    n = len(gold)
    drift = rng.normal(0, 1.5, size=n)
    spread_drift = rng.normal(0, 1.0, size=n)
    widened = gold.copy()
    widened["line_movement_coverage"] = rng.choice([0.0, 1.0], size=n)
    widened["opening_total"] = rng.normal(44, 4, size=n)
    widened["total_drift"] = drift
    widened["total_drift_dir"] = np.sign(drift)
    widened["total_late_drift"] = rng.normal(0, 0.5, size=n)
    widened["total_abs_travel"] = np.abs(rng.normal(0, 2, size=n))
    widened["total_reversals"] = rng.integers(0, 3, size=n).astype(float)
    widened["total_range"] = np.abs(rng.normal(0, 3, size=n))
    widened["opening_spread"] = rng.normal(0, 3, size=n)
    widened["spread_drift"] = spread_drift
    widened["spread_drift_dir"] = np.sign(spread_drift)
    widened["spread_late_drift"] = rng.normal(0, 0.4, size=n)
    widened["spread_abs_travel"] = np.abs(rng.normal(0, 1.5, size=n))
    widened["spread_reversals"] = rng.integers(0, 3, size=n).astype(float)
    widened["spread_range"] = np.abs(rng.normal(0, 2, size=n))
    # A pre-existing MarketAnchor decoy pair: historically identically 0.0, already in the
    # baseline, and NOT part of the Phase-29 family (they must not be captured by the predicate).
    widened["total_movement"] = 0.0
    widened["spread_movement"] = 0.0
    return widened


_FIXTURE_CONFIG = TemporalSplitConfig(
    train_seasons=[2018],
    hp_val_seasons=[2019],
    holdout_seasons=[2020, 2021],
)


# ---------------------------------------------------------------------------
# (1) D-05 keep/drop rule -- pure logic
# ---------------------------------------------------------------------------


class TestKeepDropRule:
    """The D-05 rule: keep on >=1 positive target AND not vetoed on ANY target."""

    @staticmethod
    def _cell(mean: float | None, p: float | None) -> dict:
        keep_target = mean is not None and mean > 0
        veto = mean is not None and mean < 0 and p is not None and p < 0.05
        return {
            "delta_mean": mean,
            "delta_p": p,
            "keep_target": keep_target,
            "veto": veto,
        }

    def test_keep_when_positive_on_one_target_and_no_veto(self) -> None:
        per_target = {
            "wp": self._cell(0.01, 0.20),  # positive (not significant) -> keep_target
            "ats": self._cell(-0.02, 0.40),  # negative but NOT significant -> no veto
            "ou": self._cell(0.00, 0.90),
        }
        decision = decide_group_keep(per_target)
        assert decision["keep"] is True
        assert decision["any_positive"] is True
        assert decision["any_veto"] is False
        assert "wp" in decision["positive_targets"]
        assert "carried to Phase 30" in decision["reason"]

    def test_drop_when_significantly_negative_on_any_target(self) -> None:
        per_target = {
            "wp": self._cell(0.05, 0.01),  # strongly positive
            "ats": self._cell(-0.10, 0.001),  # significantly NEGATIVE -> veto
            "ou": self._cell(0.02, 0.30),
        }
        decision = decide_group_keep(per_target)
        assert decision["keep"] is False  # veto overrides the positive WP target
        assert decision["any_veto"] is True
        assert "ats" in decision["veto_targets"]
        assert "dropped, not silently retained" in decision["reason"]

    def test_drop_when_no_positive_target(self) -> None:
        per_target = {
            "wp": self._cell(-0.01, 0.60),
            "ats": self._cell(0.00, 0.99),
            "ou": self._cell(-0.03, 0.20),
        }
        decision = decide_group_keep(per_target)
        assert decision["keep"] is False
        assert decision["any_positive"] is False
        assert decision["any_veto"] is False
        assert "no incremental CLV lift" in decision["reason"]


# ---------------------------------------------------------------------------
# (2) Feature-group column selection (review #3)
# ---------------------------------------------------------------------------


class TestSelectGroupColumns:
    """The add-one-in column selector: baseline drops all new cols; a group re-admits only its own."""

    def test_group_columns_partition_the_new_columns(self) -> None:
        gold = _make_fixture_gold()
        snap = group_columns(gold, "snap")
        injury = group_columns(gold, "injury")
        situational = group_columns(gold, "situational")

        assert snap == ["away_snap_continuity", "home_rolling_snap_share_qb"]
        # CR-01: the injury group must return ALL 12 home_/away_ injury columns,
        # not just the coverage flag the old "injury" substring predicate matched.
        assert injury == [
            "away_availability_coverage",
            "away_availability_fraction",
            "away_backup_quality_delta",
            "away_date_modified_coverage",
            "away_injury_coverage",
            "away_qb_out_flag",
            "home_availability_coverage",
            "home_availability_fraction",
            "home_backup_quality_delta",
            "home_date_modified_coverage",
            "home_injury_coverage",
            "home_qb_out_flag",
        ]
        assert situational == ["away_look_ahead_spot", "home_off_bye"]

        # The decoy snapshot_* odds columns are NOT classified as snap features.
        assert "snapshot_spread" not in snap
        assert "snapshot_total" not in snap

        all_new = phase28_new_columns(gold)
        assert set(all_new) == set(snap) | set(injury) | set(situational)

    def test_baseline_leg_drops_every_phase28_column(self) -> None:
        gold = _make_fixture_gold()
        baseline = select_group_columns(gold, group=None)
        for col in phase28_new_columns(gold):
            assert col not in baseline.columns
        # Baseline retains the unrelated odds-snapshot + baseline features.
        for col in ("snapshot_spread", "snapshot_total", "feat_a", "feat_b"):
            assert col in baseline.columns

    def test_candidate_leg_admits_only_its_group(self) -> None:
        gold = _make_fixture_gold()
        snap_df = select_group_columns(gold, group="snap")
        for col in group_columns(gold, "snap"):
            assert col in snap_df.columns
        # No injury/situational columns leak into the snap candidate.
        for col in group_columns(gold, "injury") + group_columns(gold, "situational"):
            assert col not in snap_df.columns

    def test_selector_never_mutates_input(self) -> None:
        gold = _make_fixture_gold()
        before = list(gold.columns)
        _ = select_group_columns(gold, group="injury")
        _ = select_group_columns(gold, group=None)
        assert list(gold.columns) == before  # input frame untouched (HARD BOUNDARY)


# ---------------------------------------------------------------------------
# (3) In-process walk-forward seam -- NOT score_deployed_artifacts (review #2/#3)
# ---------------------------------------------------------------------------


class TestWalkForwardSeam:
    """The lift anchor is train_and_evaluate(tune=False); score_deployed_artifacts is NOT called."""

    def test_screen_uses_train_and_evaluate_tune_false_not_score_deployed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gold = _make_fixture_gold()
        odds = _make_fixture_odds(gold)

        calls: list[dict] = []

        from models.trainers.wp_trainer import WPTrainer

        original = WPTrainer.train_and_evaluate

        def _spy(self, features_df, closing_odds_df=None, tune=True):
            calls.append({"tune": tune, "n_cols": len(features_df.columns)})
            return original(self, features_df, closing_odds_df, tune=tune)

        monkeypatch.setattr(WPTrainer, "train_and_evaluate", _spy)

        # Poison score_deployed_artifacts: if the screen uses it as the anchor, the run fails loudly.
        from backtest import diagnose

        def _poisoned(*args, **kwargs):  # pragma: no cover - must never be reached
            raise AssertionError(
                "score_deployed_artifacts must NOT be the lift anchor (review #2)"
            )

        monkeypatch.setattr(diagnose, "score_deployed_artifacts", _poisoned)
        # Also guard the name if it had been imported into signal_lift's namespace.
        if hasattr(signal_lift, "score_deployed_artifacts"):
            monkeypatch.setattr(signal_lift, "score_deployed_artifacts", _poisoned)

        result = run_signal_lift_screen(
            gold_by_target={"wp": gold},
            closing_odds_df=odds,
            config=_FIXTURE_CONFIG,
            targets=["wp"],
            groups=["snap"],
        )

        # The seam was driven with tune=False on both the baseline and candidate legs.
        assert calls, "train_and_evaluate was never invoked"
        assert all(c["tune"] is False for c in calls)
        # Baseline leg + candidate leg = two re-fits for one (group, target).
        assert len(calls) == 2
        # The candidate leg carries strictly more columns than the baseline (the add-one-in).
        assert calls[1]["n_cols"] > calls[0]["n_cols"]

        cell = result["groups"]["snap"]["per_target"]["wp"]
        assert cell["n_paired"] > 0
        assert result["anchor"] == "BaseTrainer.train_and_evaluate(tune=False)"

    def test_signal_lift_does_not_import_score_deployed_as_anchor(self) -> None:
        """signal_lift must not bind score_deployed_artifacts as a module-level anchor."""
        assert not hasattr(signal_lift, "score_deployed_artifacts")


# ---------------------------------------------------------------------------
# (4) Determinism + structure (the ou_divergence anti-rot guard shape)
# ---------------------------------------------------------------------------


class TestSignalLiftDeterminism:
    """Two runs over the same fixture gold produce identical structured output."""

    def test_run_signal_lift_screen_is_deterministic(self) -> None:
        gold = _make_fixture_gold()
        odds = _make_fixture_odds(gold)

        kwargs = {
            "gold_by_target": {"wp": gold},
            "closing_odds_df": odds,
            "config": _FIXTURE_CONFIG,
            "targets": ["wp"],
            "groups": ["snap"],
        }
        first = run_signal_lift_screen(**kwargs)
        second = run_signal_lift_screen(**kwargs)
        assert first == second  # byte-identical structured dict (anti-rot guard)

    def test_full_screen_structure_and_keep_drop_applied(self) -> None:
        gold = _make_fixture_gold()
        odds = _make_fixture_odds(gold)

        result = run_signal_lift_screen(
            gold_by_target={"wp": gold, "ats": gold, "ou": gold},
            closing_odds_df=odds,
            config=_FIXTURE_CONFIG,
        )

        assert result["anchor"] == "BaseTrainer.train_and_evaluate(tune=False)"
        assert result["alpha"] == 0.05
        assert set(result["groups"]) == {"injury", "snap", "situational"}

        for _group, gdata in result["groups"].items():
            assert "coverage_span" in gdata
            assert set(gdata["per_target"]) == {"wp", "ats", "ou"}
            # The D-05 decision is present and internally consistent.
            decision = gdata["decision"]
            assert isinstance(decision["keep"], bool)
            expected_keep = decision["any_positive"] and not decision["any_veto"]
            assert decision["keep"] == expected_keep
            for target, cell in gdata["per_target"].items():
                assert cell["target"] == target
                assert cell["n_paired"] > 0
                assert cell["clv_column"] in ("probability_clv", "line_clv")


# ---------------------------------------------------------------------------
# (5) Phase 29 (SIG-04): the line_movement group + the baseline that KEEPS Phase 28
# ---------------------------------------------------------------------------


class TestLineMovementGroupSelection:
    """The line_movement suffix predicate selects the family and nothing that merely looks like it."""

    def test_predicate_selects_exactly_the_line_movement_family(self) -> None:
        gold = _make_fixture_gold_with_line_movement()
        selected = group_columns(gold, "line_movement")
        assert selected == sorted(_LINE_MOVEMENT_COLUMNS)

    def test_predicate_does_not_match_the_lookalike_columns(self) -> None:
        """A bare ``total`` / ``spread`` substring would sweep in four baseline columns."""
        gold = _make_fixture_gold_with_line_movement()
        selected = set(group_columns(gold, "line_movement"))
        for decoy in (
            "snapshot_total",  # the freeze anchor -- already a baseline feature
            "snapshot_spread",
            "total_points",  # the OU target
            "total_movement",  # pre-existing MarketAnchor column, historically 0.0
            "spread_movement",
        ):
            assert decoy in gold.columns, f"fixture lost its {decoy!r} decoy"
            assert decoy not in selected, (
                f"{decoy!r} was mis-classified as a line-movement feature"
            )

    def test_line_movement_is_not_in_the_default_groups_tuple(self) -> None:
        """Registered in the predicate map ONLY (review 29-07 HIGH).

        Membership in ``GROUPS`` would put line_movement in the DEFAULT baseline-exclusion
        union, stripping the kept Phase-28 signal out of the Phase-29 baseline leg.
        """
        assert "line_movement" not in signal_lift.GROUPS
        assert "line_movement" in signal_lift._GROUP_PREDICATE
        assert "line_movement" in signal_lift.GROUP_COVERAGE
        assert "2020-06-06" in signal_lift.GROUP_COVERAGE["line_movement"]
        assert "2021-2024" in signal_lift.GROUP_COVERAGE["line_movement"]


class TestBaselineRetainsPhase28:
    """The load-bearing review 29-07 HIGH fixture: the Phase-29 baseline KEEPS the Phase-28 groups."""

    def test_phase29_baseline_keeps_phase28_columns_and_drops_only_line_movement(
        self,
    ) -> None:
        gold = _make_fixture_gold_with_line_movement()
        baseline = select_group_columns(
            gold, group=None, exclude_groups=("line_movement",)
        )

        phase28 = phase28_new_columns(gold)
        assert phase28, (
            "fixture must carry Phase-28 columns for this test to mean anything"
        )
        # Named spot-check so a future predicate regression names the column it lost.
        assert "home_qb_out_flag" in phase28
        for col in phase28:
            assert col in baseline.columns, (
                f"Phase-28 column {col!r} was stripped from the Phase-29 baseline -- the "
                "D-02 'redundant WITH the injury signal?' question is then unanswerable"
            )
        for col in _LINE_MOVEMENT_COLUMNS:
            assert col not in baseline.columns
        assert len(baseline.columns) == len(gold.columns) - len(_LINE_MOVEMENT_COLUMNS)

    def test_phase28_default_baseline_is_unchanged_by_the_new_parameter(self) -> None:
        """Default ``exclude_groups=GROUPS`` reproduces the Phase-28 baseline exactly."""
        gold = _make_fixture_gold_with_line_movement()
        implicit = select_group_columns(gold, group=None)
        explicit = select_group_columns(
            gold, group=None, exclude_groups=signal_lift.GROUPS
        )
        assert list(implicit.columns) == list(explicit.columns)
        # The Phase-28 default drops the Phase-28 columns and RETAINS line-movement.
        for col in phase28_new_columns(gold):
            assert col not in implicit.columns
        for col in _LINE_MOVEMENT_COLUMNS:
            assert col in implicit.columns

    def test_candidate_leg_is_baseline_plus_exactly_the_line_movement_family(
        self,
    ) -> None:
        gold = _make_fixture_gold_with_line_movement()
        baseline = select_group_columns(
            gold, group=None, exclude_groups=("line_movement",)
        )
        candidate = select_group_columns(
            gold, group="line_movement", exclude_groups=("line_movement",)
        )
        added = set(candidate.columns) - set(baseline.columns)
        assert added == set(_LINE_MOVEMENT_COLUMNS)
        assert not set(baseline.columns) - set(candidate.columns)


class TestLineMovementScreen:
    """The Phase-29 screen runs on the walk-forward anchor and applies the keep/drop rule."""

    def test_screen_measures_line_movement_via_walkforward_anchor(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gold = _make_fixture_gold_with_line_movement()
        odds = _make_fixture_odds(gold)

        calls: list[dict] = []
        from models.trainers.wp_trainer import WPTrainer

        original = WPTrainer.train_and_evaluate

        def _spy(self, features_df, closing_odds_df=None, tune=True):
            calls.append({"tune": tune, "cols": list(features_df.columns)})
            return original(self, features_df, closing_odds_df, tune=tune)

        monkeypatch.setattr(WPTrainer, "train_and_evaluate", _spy)

        from backtest import diagnose

        def _poisoned(*args, **kwargs):  # pragma: no cover - must never be reached
            raise AssertionError(
                "score_deployed_artifacts must NOT be the lift anchor (review #2)"
            )

        monkeypatch.setattr(diagnose, "score_deployed_artifacts", _poisoned)

        result = run_signal_lift_screen(
            gold_by_target={"wp": gold},
            closing_odds_df=odds,
            config=_FIXTURE_CONFIG,
            targets=["wp"],
            groups=["line_movement"],
            baseline_exclude_groups=["line_movement"],
        )

        assert len(calls) == 2, "expected one baseline leg + one candidate leg"
        assert all(c["tune"] is False for c in calls)
        baseline_cols, candidate_cols = calls[0]["cols"], calls[1]["cols"]
        # The BASELINE leg that actually reached the trainer keeps the Phase-28 signal.
        assert "home_qb_out_flag" in baseline_cols
        assert "away_look_ahead_spot" in baseline_cols
        assert "total_drift" not in baseline_cols
        assert set(candidate_cols) - set(baseline_cols) == set(_LINE_MOVEMENT_COLUMNS)

        assert result["anchor"] == "BaseTrainer.train_and_evaluate(tune=False)"
        assert result["baseline_excludes"] == ["line_movement"]
        cell = result["groups"]["line_movement"]["per_target"]["wp"]
        assert cell["n_paired"] > 0
        assert cell["clv_column"] == "probability_clv"

    def test_keep_drop_rule_is_applied_to_the_line_movement_group(self) -> None:
        gold = _make_fixture_gold_with_line_movement()
        odds = _make_fixture_odds(gold)

        result = run_signal_lift_screen(
            gold_by_target={"wp": gold},
            closing_odds_df=odds,
            config=_FIXTURE_CONFIG,
            targets=["wp"],
            groups=["line_movement"],
            baseline_exclude_groups=["line_movement"],
        )

        gdata = result["groups"]["line_movement"]
        decision = gdata["decision"]
        assert isinstance(decision["keep"], bool)
        assert decision["keep"] == (
            decision["any_positive"] and not decision["any_veto"]
        )
        assert "2020-06-06" in gdata["coverage_span"]
        # A DROP must read as a drop, never as a silent retention.
        if not decision["keep"]:
            assert "DROP" in decision["reason"]
            assert "dropped, not silently retained" in decision["reason"]

    def test_screen_is_deterministic_and_never_mutates_gold(self) -> None:
        gold = _make_fixture_gold_with_line_movement()
        odds = _make_fixture_odds(gold)
        before = list(gold.columns)

        kwargs = {
            "gold_by_target": {"wp": gold},
            "closing_odds_df": odds,
            "config": _FIXTURE_CONFIG,
            "targets": ["wp"],
            "groups": ["line_movement"],
            "baseline_exclude_groups": ["line_movement"],
        }
        assert run_signal_lift_screen(**kwargs) == run_signal_lift_screen(**kwargs)
        assert list(gold.columns) == before


class TestMeasurabilityAccounting:
    """A screen must be able to say "never used" instead of silently reporting selection churn.

    Every trainer runs its own ``SelectFromModel`` pass on the ``train_seasons`` window and LOCKS
    the result for the whole holdout walk-forward. A group whose columns are constant in that
    window has zero importance by construction and can NEVER be selected -- so the candidate
    model never sees it and the paired delta measures churn among the OTHER features, not the
    group. Without this accounting that delta is indistinguishable from a real lift.
    """

    @staticmethod
    def _screen(gold: pd.DataFrame) -> dict:
        return run_signal_lift_screen(
            gold_by_target={"wp": gold},
            closing_odds_df=_make_fixture_odds(gold),
            config=_FIXTURE_CONFIG,
            targets=["wp"],
            groups=["line_movement"],
            baseline_exclude_groups=["line_movement"],
        )

    def test_cells_report_how_many_group_columns_the_model_actually_used(self) -> None:
        gold = _make_fixture_gold_with_line_movement()
        cell = self._screen(gold)["groups"]["line_movement"]["per_target"]["wp"]
        assert cell["n_group_columns"] == len(_LINE_MOVEMENT_COLUMNS)
        assert 0 <= cell["n_group_columns_selected"] <= cell["n_group_columns"]
        assert cell["measurable"] == (cell["n_group_columns_selected"] > 0)
        assert set(cell["group_columns_selected"]) <= set(_LINE_MOVEMENT_COLUMNS)

    def test_group_constant_in_the_selection_window_is_flagged_not_measured(
        self,
    ) -> None:
        """The real Phase-29 shape: the family is constant across the whole train window."""
        gold = _make_fixture_gold_with_line_movement()
        train_mask = gold["season"].isin(_FIXTURE_CONFIG.train_seasons)
        for col in _LINE_MOVEMENT_COLUMNS:
            gold.loc[train_mask, col] = 0.0
        assert gold.loc[train_mask, "total_drift"].nunique() == 1

        gdata = self._screen(gold)["groups"]["line_movement"]
        assert gdata["per_target"]["wp"]["n_group_columns_selected"] == 0
        assert gdata["per_target"]["wp"]["measurable"] is False
        assert gdata["measurability"]["measurable"] is False
        assert gdata["measurability"]["measurable_targets"] == []
        assert "NOT MEASURED" in gdata["measurability"]["note"]
        assert "selection churn" in gdata["measurability"]["note"]

    def test_not_measured_groups_are_called_out_in_the_printed_report(self) -> None:
        gold = _make_fixture_gold_with_line_movement()
        train_mask = gold["season"].isin(_FIXTURE_CONFIG.train_seasons)
        for col in _LINE_MOVEMENT_COLUMNS:
            gold.loc[train_mask, col] = 0.0

        report = signal_lift._format_screen_report(self._screen(gold))
        assert "NOT MEASURED" in report
        assert "grp_cols_used" in report
        assert f"0/{len(_LINE_MOVEMENT_COLUMNS)}" in report


class TestPhase29CliWiring:
    """``--phase 29`` is the single deterministic command the doc-drift guard re-runs."""

    def test_default_phase_is_28_and_carries_no_overrides(self) -> None:
        args = signal_lift._build_parser().parse_args([])
        assert args.phase == 28
        assert signal_lift.screen_kwargs_for_phase(28) == {}

    def test_phase_29_selects_the_line_movement_screen_on_both_legs(self) -> None:
        args = signal_lift._build_parser().parse_args(["--phase", "29"])
        assert args.phase == 29
        assert args.coverage_window is False
        kwargs = signal_lift.screen_kwargs_for_phase(29)
        assert kwargs == {
            "groups": ("line_movement",),
            "baseline_exclude_groups": ("line_movement",),
        }

    def test_coverage_window_flag_swaps_in_the_diagnostic_config(self) -> None:
        """The diagnostic window must be opt-in and must not touch the canonical run."""
        args = signal_lift._build_parser().parse_args(
            ["--phase", "29", "--coverage-window"]
        )
        assert args.coverage_window is True
        kwargs = signal_lift.screen_kwargs_for_phase(29, coverage_window=True)
        assert kwargs["config"] is signal_lift.COVERAGE_WINDOW_CONFIG
        assert kwargs["groups"] == ("line_movement",)
        # The diagnostic trains on covered seasons and measures ONE season.
        assert signal_lift.COVERAGE_WINDOW_CONFIG.train_seasons == [2021, 2022]
        assert signal_lift.COVERAGE_WINDOW_CONFIG.holdout_seasons == [2024]
        # ...and it is NOT what a default run uses.
        assert "config" not in signal_lift.screen_kwargs_for_phase(29)

    def test_covered_selection_window_flag_swaps_in_its_config(self) -> None:
        """The covered selection window is opt-in and leaves the default run untouched."""
        args = signal_lift._build_parser().parse_args(
            ["--phase", "29", "--covered-selection-window"]
        )
        assert args.covered_selection_window is True
        assert args.coverage_window is False
        kwargs = signal_lift.screen_kwargs_for_phase(29, covered_selection_window=True)
        assert kwargs["config"] is signal_lift.COVERED_SELECTION_WINDOW_CONFIG
        assert kwargs["groups"] == ("line_movement",)

    def test_covered_selection_window_is_absent_from_a_default_run(self) -> None:
        args = signal_lift._build_parser().parse_args(["--phase", "29"])
        assert args.covered_selection_window is False
        assert "config" not in signal_lift.screen_kwargs_for_phase(29)

    def test_covered_selection_window_config_shape(self) -> None:
        """Train 2018-2020 with NO hp-val fold; measure the whole 2021-2024 holdout."""
        config = signal_lift.COVERED_SELECTION_WINDOW_CONFIG
        assert config.train_seasons == [2018, 2019, 2020]
        assert config.hp_val_seasons == []
        assert config.holdout_seasons == [2021, 2022, 2023, 2024]
        config.validate()
        # The canonical window is NOT mutated by adding this sibling (D-Q2).
        assert TemporalSplitConfig.default().train_seasons == [2018, 2019]

    def test_the_two_window_flags_are_mutually_exclusive(self) -> None:
        """Rejected at the parser AND at the helper -- they name different spans."""
        with pytest.raises(SystemExit):
            signal_lift._build_parser().parse_args(
                ["--phase", "29", "--coverage-window", "--covered-selection-window"]
            )
        with pytest.raises(ValueError, match="mutually exclusive"):
            signal_lift.screen_kwargs_for_phase(
                29, coverage_window=True, covered_selection_window=True
            )

    def test_report_labels_the_window_actually_measured(self) -> None:
        """A one-season diagnostic must not be printed under the canonical 2021-2024 label."""
        gold = _make_fixture_gold_with_line_movement()
        result = run_signal_lift_screen(
            gold_by_target={"wp": gold},
            closing_odds_df=_make_fixture_odds(gold),
            config=_FIXTURE_CONFIG,
            targets=["wp"],
            groups=["line_movement"],
            baseline_exclude_groups=["line_movement"],
        )
        assert (
            result["measure_window"] == "2020-2021"
        )  # the fixture's holdout, not the constant
