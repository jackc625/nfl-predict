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
