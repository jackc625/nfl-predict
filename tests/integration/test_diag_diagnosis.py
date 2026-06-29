"""Wave 0 Nyquist harness for the DIAG honest-accuracy diagnosis (Phase 22, plan 22-01).

Covers DIAG-01..05 measurement correctness for ``backtest/diagnose.py``:

  - DIAG-01: BOTH the all-games straight-pick hit-rate (min_edge_threshold=0.0) AND the
    canonical edge-filtered bet rate (default min_edge_threshold=0.02) are computed via the
    BettingSimulator honest sign convention -- never metrics.py cover_accuracy/over_accuracy.
  - DIAG-02: WP ECE + Brier decomposition reproduced (compute_wp_metrics) on the deployed cut.
  - DIAG-03: per-game CLV significance (t-stat, p-value, 95% CI) via scipy.stats.ttest_1samp on
    the per-game CLV arrays -- probability_clv for WP, line_clv for ATS/OU.
  - DIAG-04: the verdict rubric is significance-gated (a non-significant negative CLV reads
    "indistinguishable from zero"); the scored population includes postseason (weeks 19-22).
  - DIAG-05: the three deployed v1.0 pre-Elo artifacts LOAD and SCORE over 2021-2024 canonical
    gold with zero missing features (1139 games/target), mirroring run_predictions EXACTLY; the
    raw + market-blended 2x2 is produced and the double-run is value-identical.

Carried-D-01 HARD BOUNDARY (the milestone's namesake): the scoring path imports NO ``train_*``
module and NEVER writes ``data/gold/`` -- LOAD + predict only, NO re-fit, NO gold rebuild. The
no-train/no-write-gold guard and the postseason-population flag are asserted here directly.

Number anchoring (T-22-02): the regenerated pooled WP accuracy and headline_clv wp must reproduce
the deterministic walk-forward backtest values within float tolerance, guarding against silent
drift. The anchors were the v2.1 AUDIT-REPORT post-rebuild figures (WP accuracy 0.66725,
headline_clv wp -0.00207). Phase 28 (new signal -- injuries/snaps/situational, plan 28-06)
DELIBERATELY widened the gold matrices (156/157/156 -> 194/195/194 columns; +38 columns/matrix),
so the per-fold walk-forward backtest now trains on the wider feature set and the deterministic WP
accuracy moved 0.66725 -> 0.67691 (it IMPROVED -- the new signal helps) and headline_clv wp moved
-0.00207 -> -0.00469. The anchors below are reconciled to those NEW post-widening deterministic
values (IN-03 convention: a deliberate, documented update -- NEVER silenced -- exactly analogous to
the GOLD_FEATURE_MATRICES width tripwire that plan 28-06 already reconciled). The literal v2.1
AUDIT-REPORT.md figures are intentionally NOT edited (that file is a frozen v2.1 forensic record of
the Phase-20 weather rebuild at the 156/157/156 width); the anchors are moved past those literal
figures here so the Phase-28 provenance is honest.

Reproducibility convention: the shared fixture loads gold + normalized closing odds via the
engine loaders (``_load_features`` / ``_load_closing_odds``) rather than re-reading parquet so the
LAR->LA canonical team-abbreviation normalization matches the backtest exactly (CLAUDE.md).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backtest.engine import BacktestEngine

# Gold presence skip-guard: the integration tests need the Phase-20 rebuilt canonical gold.
_GOLD_WP_PATH = Path("data/gold/features_wp.parquet")

# Deterministic walk-forward backtest anchors (n-games-weighted pooled, 2021-2024).
# Reconciled in Phase 28 (plan 28-06 post-merge fix) after the deliberate gold widening
# (+38 snap/injury/situational columns/matrix) moved the per-fold WP backtest:
#   accuracy     0.66725 (v2.1 AUDIT-REPORT) -> 0.67691 (Phase-28 widened gold; improved)
#   headline_clv -0.00207 (v2.1 AUDIT-REPORT) -> -0.00469 (Phase-28 widened gold)
# IN-03 convention: a deliberate, DOCUMENTED anchor update -- never silenced. AUDIT-REPORT.md
# stays frozen at its v2.1 156/157/156-width figures (it is a historical forensic record).
ANCHOR_WP_ACCURACY = 0.67691
ANCHOR_HEADLINE_CLV_WP = -0.00469

# Expected holdout population (verified): 1139 games per target across 2021-2024.
EXPECTED_GAMES_PER_TARGET = 1139

# Postseason weeks that MUST appear in the scored population (D-04 flag).
POSTSEASON_WEEKS = {19, 20, 21, 22}


@pytest.fixture(scope="module")
def gold_and_odds_2021_2024() -> dict[str, object]:
    """Shared fixture: 2021-2024 gold per target + normalized closing odds.

    Reuses ``BacktestEngine._load_features`` (gold, season-filtered to <= 2024) and
    ``BacktestEngine._load_closing_odds`` (LAR->LA normalized game_ids) so the DIAG fixture
    matches the backtest's canonical team-mapping exactly. Does NOT read the parquet raw.

    Returns a dict with ``gold`` (dict[target -> 2021-2024 frame]) and ``odds`` (normalized
    closing-odds frame). Skips the whole module if canonical gold is absent.
    """
    if not _GOLD_WP_PATH.exists():
        pytest.skip(f"Canonical gold not present at {_GOLD_WP_PATH}")

    engine = BacktestEngine()
    gold: dict[str, pd.DataFrame] = {}
    for target in ("wp", "ats", "ou"):
        df = engine._load_features(target)
        gold[target] = df[(df["season"] >= 2021) & (df["season"] <= 2024)].copy()
    odds = engine._load_closing_odds()
    return {"gold": gold, "odds": odds}


@pytest.mark.integration
class TestDiagDiagnosis:
    """DIAG-01..05 measurement + determinism + significance-gate + no-re-fit guards."""

    # -- DIAG-05: deployed-artifact scoring ----------------------------------------------

    def test_deployed_artifacts_score_no_missing_features(
        self, gold_and_odds_2021_2024
    ) -> None:
        """The three deployed artifacts score 2021-2024 with 0 missing features, 1139 games."""
        from backtest.diagnose import score_deployed_artifacts

        gold = gold_and_odds_2021_2024["gold"]
        for target in ("wp", "ats", "ou"):
            preds = score_deployed_artifacts(target, gold_df=gold[target])
            assert len(preds) == EXPECTED_GAMES_PER_TARGET, (
                f"{target}: expected {EXPECTED_GAMES_PER_TARGET} scored games, "
                f"got {len(preds)}"
            )
            assert "model_prob" in preds.columns
            assert "game_id" in preds.columns
            assert "actual" in preds.columns
            assert preds["model_prob"].notna().all(), (
                f"{target}: all games must score (no missing-feature NaNs)"
            )

    def test_deployed_scoring_deterministic(self, gold_and_odds_2021_2024) -> None:
        """Double-run deployed-artifact scoring is value-identical (assert_frame_equal)."""
        from backtest.diagnose import score_deployed_artifacts

        gold = gold_and_odds_2021_2024["gold"]
        for target in ("wp", "ats", "ou"):
            run_a = (
                score_deployed_artifacts(target, gold_df=gold[target])
                .sort_values("game_id")
                .reset_index(drop=True)
            )
            run_b = (
                score_deployed_artifacts(target, gold_df=gold[target])
                .sort_values("game_id")
                .reset_index(drop=True)
            )
            pd.testing.assert_frame_equal(run_a, run_b)

    # -- DIAG-01: hit-rate via simulator convention --------------------------------------

    def test_hit_rate_uses_simulator_convention(self, gold_and_odds_2021_2024) -> None:
        """Reported hit-rate equals the BettingSimulator by_target.win_rate (not metrics.py)."""
        from backtest.diagnose import (
            both_population_hit_rates,
            score_deployed_artifacts,
        )
        from backtest.simulation import BettingSimulator, SimulationConfig

        gold = gold_and_odds_2021_2024["gold"]
        odds = gold_and_odds_2021_2024["odds"]

        preds = {
            t: score_deployed_artifacts(t, gold_df=gold[t]) for t in ("wp", "ats", "ou")
        }
        results_like = _make_results_like(preds)

        hit = both_population_hit_rates(results_like, odds)

        # Independently re-derive the edge-filtered rate via the simulator and require equality.
        sim = BettingSimulator(SimulationConfig(min_edge_threshold=0.02))
        sim_res = sim.simulate(results_like, odds)
        for target in ("wp", "ats", "ou"):
            expected = sim_res.by_target[target]["win_rate"]
            assert abs(hit[target]["edge_filtered"] - expected) < 1e-9, (
                f"{target}: edge-filtered hit-rate must equal simulator by_target.win_rate"
            )

        # The harness source must not use cover_accuracy/over_accuracy for the reported hit-rate.
        import backtest.diagnose as diag_mod

        source = inspect.getsource(diag_mod)
        assert "cover_accuracy" not in source, (
            "diagnose.py must NOT use metrics.cover_accuracy for the hit-rate (D-01)"
        )
        assert "over_accuracy" not in source, (
            "diagnose.py must NOT use metrics.over_accuracy for the hit-rate (D-01)"
        )

        # The fixture must reuse the normalized-odds loader, not the raw odds parquet. Check the
        # genuine anti-pattern (a read_parquet call on the raw odds snapshot); the token is joined
        # at runtime so this assertion does not match its own literal in the file source.
        test_source = Path(__file__).read_text(encoding="utf-8")
        assert "_load_closing_odds" in test_source
        raw_odds_antipattern = "read_parquet(" + chr(34) + "data/silver/odds_snapshot"
        assert raw_odds_antipattern not in test_source

    def test_all_vs_edge_filtered_gap_reported(self, gold_and_odds_2021_2024) -> None:
        """Both populations (straight_pick min_edge=0.0 + edge_filtered 0.02) are computed."""
        from backtest.diagnose import (
            both_population_hit_rates,
            score_deployed_artifacts,
        )

        gold = gold_and_odds_2021_2024["gold"]
        odds = gold_and_odds_2021_2024["odds"]

        preds = {
            t: score_deployed_artifacts(t, gold_df=gold[t]) for t in ("wp", "ats", "ou")
        }
        results_like = _make_results_like(preds)

        hit = both_population_hit_rates(results_like, odds)
        for target in ("wp", "ats", "ou"):
            assert "straight_pick" in hit[target]
            assert "edge_filtered" in hit[target]
            assert "gap" in hit[target]
            sp = hit[target]["straight_pick"]
            ef = hit[target]["edge_filtered"]
            assert 0.0 <= sp <= 1.0
            assert 0.0 <= ef <= 1.0
            assert abs(hit[target]["gap"] - (sp - ef)) < 1e-9

    # -- DIAG-02: WP calibration ---------------------------------------------------------

    def test_wp_calibration_reproducible(self, gold_and_odds_2021_2024) -> None:
        """WP ECE + Brier decomposition reproduced deterministically on the deployed cut."""
        from backtest.diagnose import score_deployed_artifacts
        from backtest.metrics import compute_wp_metrics

        gold = gold_and_odds_2021_2024["gold"]
        preds = score_deployed_artifacts("wp", gold_df=gold["wp"])

        metrics_a = compute_wp_metrics(
            preds["actual"].to_numpy(), preds["model_prob"].to_numpy()
        )
        metrics_b = compute_wp_metrics(
            preds["actual"].to_numpy(), preds["model_prob"].to_numpy()
        )
        for key in (
            "accuracy",
            "ece",
            "brier_score",
            "brier_reliability",
            "brier_resolution",
        ):
            assert key in metrics_a
            assert metrics_a[key] == metrics_b[key]
        assert 0.0 <= metrics_a["ece"] <= 1.0

    # -- DIAG-03: CLV significance -------------------------------------------------------

    def test_clv_significance_and_ci(self, gold_and_odds_2021_2024) -> None:
        """Per-target CLV significance: t/p/ci95 present; ATS/OU use line_clv, WP uses prob_clv."""
        from backtest.diagnose import (
            apply_blended_cut,  # noqa: F401  (ensures harness API is importable)
            clv_significance,
            score_deployed_artifacts,
        )
        from models.clv import compute_clv_for_predictions

        gold = gold_and_odds_2021_2024["gold"]
        odds = gold_and_odds_2021_2024["odds"]

        clv_column_for = {"wp": "probability_clv", "ats": "line_clv", "ou": "line_clv"}
        for target in ("wp", "ats", "ou"):
            preds = score_deployed_artifacts(target, gold_df=gold[target])
            clv_df = compute_clv_for_predictions(preds, odds, target)
            valid = clv_df[clv_df["has_closing_odds"]]
            column = clv_column_for[target]
            assert column in clv_df.columns, f"{target}: CLV frame must carry {column}"
            sig = clv_significance(valid[column].to_numpy())
            assert sig["n"] >= 10
            assert sig["t"] is not None
            assert sig["p"] is not None
            assert sig["ci95"] is not None
            lo, hi = sig["ci95"]
            assert lo <= sig["mean"] <= hi

    # -- DIAG-04: verdict rubric significance gate ---------------------------------------

    def test_verdict_rubric_significance_gated(self) -> None:
        """A synthetic non-significant negative CLV reads 'indistinguishable from zero'."""
        from backtest.diagnose import clv_verdict

        # Tiny negative mean, large noise => not statistically distinguishable from zero.
        rng = np.random.default_rng(42)
        non_sig = rng.normal(loc=-0.001, scale=0.5, size=500)
        verdict = clv_verdict(non_sig)
        assert "indistinguishable from zero" in verdict.lower()

        # A clearly significant negative CLV must NOT read indistinguishable.
        sig_neg = rng.normal(loc=-0.5, scale=0.2, size=500)
        verdict_neg = clv_verdict(sig_neg)
        assert "indistinguishable from zero" not in verdict_neg.lower()

    # -- DIAG-04: postseason population flag (harness-independent) ------------------------

    def test_postseason_population_included(self, gold_and_odds_2021_2024) -> None:
        """The scored 2021-2024 population includes postseason weeks 19-22 (D-04)."""
        gold = gold_and_odds_2021_2024["gold"]
        for target in ("wp", "ats", "ou"):
            weeks = {int(w) for w in gold[target]["week"].unique()}
            assert POSTSEASON_WEEKS.issubset(weeks), (
                f"{target}: scored population must include postseason weeks 19-22, "
                f"got max week {max(weeks)}"
            )

    # -- T-22-02: number anchoring -------------------------------------------------------

    def test_backtest_numbers_match_audit_report(self, gold_and_odds_2021_2024) -> None:
        """Backtest WP pooled accuracy ~0.66725 and headline_clv wp ~ -0.00207 (AUDIT-REPORT)."""
        from backtest.diagnose import run_diagnosis

        diag = run_diagnosis(
            gold=gold_and_odds_2021_2024["gold"],
            odds=gold_and_odds_2021_2024["odds"],
            run_backtest_half=True,
        )

        wp_acc = diag["backtest"]["wp"]["pooled_accuracy"]
        assert abs(wp_acc - ANCHOR_WP_ACCURACY) < 5e-3, (
            f"WP pooled accuracy {wp_acc} drifted from anchor {ANCHOR_WP_ACCURACY}"
        )

        headline_clv_wp = diag["backtest"]["wp"]["headline_clv"]
        assert abs(headline_clv_wp - ANCHOR_HEADLINE_CLV_WP) < 5e-3, (
            f"headline_clv wp {headline_clv_wp} drifted from anchor "
            f"{ANCHOR_HEADLINE_CLV_WP}"
        )

    # -- T-22-03: no-re-fit / no-write-gold guard (harness-independent) ------------------

    def test_scoring_does_not_train_or_write_gold(self) -> None:
        """The scoring path imports no train_* module and never writes data/gold/ (D-01)."""
        import backtest.diagnose as diag_mod

        source = inspect.getsource(diag_mod)

        # No trainer imports on the scoring path (the milestone's namesake hard boundary).
        forbidden_imports = (
            "import train_",
            "from models.train_",
            "from models.trainers",
            "import models.train",
            "scripts.train_models",
            "scripts.retrain_models",
        )
        for token in forbidden_imports:
            assert token not in source, (
                f"diagnose.py must not import a trainer ({token!r}) on the scoring path"
            )

        # The harness must never WRITE any parquet (LOAD + predict only). A single direct,
        # intent-clear check replaces the earlier near-tautological compound assertion (WR-03):
        # the runtime guard in test_run_diagnosis_writes_no_gold is the real behavioral assertion.
        assert ".to_parquet(" not in source, "diagnose.py must not write any parquet"

    def test_run_diagnosis_writes_no_gold(
        self, gold_and_odds_2021_2024, monkeypatch
    ) -> None:
        """RUNTIME guard (WR-02): a real run_diagnosis call writes no data/gold/ parquet.

        The source-grep guard above only inspects diagnose.py's own text; it cannot catch a gold
        write performed through an imported helper, nor a writer reached transitively. This
        monkeypatches ``DataFrame.to_parquet`` to record every write path during a real
        production-half run and asserts none target the gold layer -- a behavioral assertion of
        the milestone's namesake HARD BOUNDARY (LOAD + predict only, never re-fit, never rebuild).
        """
        from backtest.diagnose import run_diagnosis

        calls: list[str] = []
        orig_to_parquet = pd.DataFrame.to_parquet

        def spy_to_parquet(self, path, *args, **kwargs):
            calls.append(str(path))
            return orig_to_parquet(self, path, *args, **kwargs)

        monkeypatch.setattr(pd.DataFrame, "to_parquet", spy_to_parquet)

        # Production half only: LOAD + predict over the deployed artifacts; no engine re-fit.
        run_diagnosis(
            gold=gold_and_odds_2021_2024["gold"],
            odds=gold_and_odds_2021_2024["odds"],
            run_backtest_half=False,
        )

        gold_writes = [c for c in calls if "data/gold" in c.replace("\\", "/")]
        assert not gold_writes, (
            "run_diagnosis must never write data/gold/ (LOAD + predict only); "
            f"observed gold writes: {gold_writes}"
        )


def _make_results_like(preds: dict[str, pd.DataFrame]):
    """Build a lightweight BacktestResults-shaped object for the simulator/CLV consumers.

    The BettingSimulator and the harness only read ``.all_predictions`` (a dict of target ->
    predictions frame), so a thin shim avoids re-running the full engine in convention tests.
    """
    from backtest.engine import BacktestConfig, BacktestResults

    return BacktestResults(
        config=BacktestConfig(),
        season_results=[],
        all_predictions=preds,
        all_clv={},
        headline_clv={},
        odds_coverage={},
        covid_annotation={},
        era_info={},
        is_blended=False,
    )
