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

THIRD ANCHOR STATE (Plan 30-16, 2026-08-24; owner ruling D30-OWNER-02, plus the owner's
2026-08-24 ruling on this plan's Task 3 halt). TWO things moved the anchors at once, and both
are named here because this file's whole convention is that an anchor never moves silently:

  1. The four-rung Phase-30 gold rebuild (Plans 30-07 / 30-08 / 30-10), which this test re-fits
     ``backtest.engine`` on.
  2. For the FIRST TIME, a tuned train that ACTUALLY SEARCHED. ``backtest/engine.py`` called
     ``train_and_evaluate(tune=True)`` but kept the LEGACY Optuna identity ``{target}_tuning_v1``,
     whose three study files were written 2026-03-31 and were already at the full 100-trial
     budget. ``OptunaTuner.optimize`` computes remaining trials as
     ``max(0, n_trials - len(study.trials))`` under ``load_if_exists=True``, so the "search" ran
     ZERO trials, returned the STORED v2.0 parameters, and still reported ``n_trials=100``. Every
     "tuned" backtest between 2026-03-31 and Plan 30-16 was a straight re-fit on frozen v2.0
     hyperparameters (D30-DEFER-01, fixed under D30-OWNER-02).

The measured move, with the arithmetic and the direction stated:

  WP pooled accuracy    0.67691 ->  0.6769095697980685   drift 4.3e-7    -- did NOT move
  headline_clv wp      -0.00469 -> -0.0028010282747504   drift 1.889e-3  -- moved, and IMPROVED

Both readings were already inside the 5e-3 band. The band was NOT widened, re-expressed or made
relative, and the constants are re-ratified to the measured values rather than left pointing at a
superseded figure that happens to still pass. The CLV anchor moved TOWARD zero: the genuinely
searched backtest gives up less to the closing line than the frozen-v2.0-parameter one did.

REPRODUCIBILITY, because this anchor is now produced by a search that runs fresh on EVERY
invocation and a single reading could not pin it. The diagnosis was run TWICE and returned
BYTE-IDENTICAL values both times (0.6769095697980685 / -0.0028010282747504157), matching the
plain ``scripts/run_backtest.py`` headline from the same code; and three independent production
backtest runs returned byte-identical Optuna best parameters for all three targets. That
reproducibility is neither free nor a seeding accident. Optuna's ``HyperbandPruner`` assigns each
trial to a bracket by a CRC32 of the STUDY NAME (``optuna/pruners/_hyperband.py:255-258``), so
Plan 30-16's FIRST attempt -- which put the engine run id in the study NAME -- made two readings
of this very anchor land 3.51e-3 apart, 70% of the band, with five runs choosing three different
WP penalties. The owner ruled the identity split: constant study NAME, per-run study STORAGE. If
this test ever starts oscillating again, check ``models.trainers.base.BACKTEST_TUNING_STUDY_TAG``
before anything else.

``AUDIT-REPORT.md`` and ``METHODOLOGY.md`` are deliberately NOT edited by Plan 30-16. They are
frozen v2.1 forensic records, and D30-OWNER-02 assigned their reconciliation to Plan 30-14 as
additional (doc, guard) pairs.

RETRACTED, recorded here because lifting the quarantine DELETES the text that carried it: the
skip reason removed below asserted, as its point (3), that the rung-3 anchor movement came from
the WP selector choosing its 20 features out of 180 numeric candidates rather than 195. Plan
30-17's census DISPROVED exactly that -- WP's selected set is invariant to dropping the 15
window-constant columns, to dropping all 82, and to padding -- because WP's ``LogisticRegression``
has no column subsampling. No replacement mechanism is offered: two hypotheses remain open, the
census cannot discriminate between them because those columns no longer exist in gold, and an
honest open question beats a tidy wrong answer (D30-DEFER-17).

FOURTH ANCHOR STATE (Plan 33.1-04, 2026-09-13; OWNER RULING of the same date on this plan's
reported halt). ONE thing moved the accuracy anchor, it is named without hedging, and it is a
METHODOLOGY CORRECTION rather than a finding:

  D33.1-R3 changed ``WPTrainer`` into a fold-fitted imputation Pipeline. The fill rule is now
  fitted INSIDE each walk-forward fold instead of across the whole frame, so no imputation
  statistic crosses a fold boundary. ``run_diagnosis(run_backtest_half=True)`` drives
  ``backtest/engine.py`` -> ``WPTrainer.train_and_evaluate``, which is the path that moved.

  WP pooled accuracy    0.67691 -> 0.6821773485513608   drift +5.2673e-3  -- EXCEEDED the band

THE DRIFT EXCEEDED THE 5e-3 BAND, which is why this needed an owner ruling rather than passing
silently, and why the executor reported a HALT instead of editing anything. It was MEASURED TWICE
and returned identical values to every printed digit; that reproducibility is what makes
re-ratification defensible here rather than a guess about noise. The band is STILL 5e-3: it was
not widened, re-expressed or made relative, and the constant is moved to what the code actually
produces. A test that goes green because a band was loosened to cover a drift is a test that has
stopped measuring anything.

WHAT THIS NUMBER IS NOT (SPEC R8, and it is a hard constraint rather than a caveat). The reading
moved UP. That is NOT a result and must not be quoted as one. Phase 33.1 re-fit no model,
promoted no model, and left ``artifacts/latest.json`` byte-unchanged; what changed is that a
temporal-correctness defect was corrected and this number moved as a side effect. The model did
not get better. The real accuracy question is not answerable until a re-fit on real weather has
actually run, and none has.

The superseded reading 0.67691 is NOT deleted -- see the constant block below, where all four
states stay on the page. That is this file's standing convention and this ruling does not
suspend it.

COST: this test now runs a genuine 100-trial search per target and takes about 225 seconds
(measured 225.3s and 228.7s), against roughly 35 seconds when the search was vacuous. It carries
the repository's ``slow`` marker for that reason, and for that reason alone.

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
# FOUR states, all of them on the page. See the module docstring for the cause of each move:
#   accuracy     0.66725 (v2.1) -> 0.67691 (Phase-28 widened gold) -> 0.67691 (Plan 30-16)
#                -> 0.6821773485513608 (Plan 33.1-04, owner ruling 2026-09-13)
#   headline_clv -0.00207 (v2.1) -> -0.00469 (Phase-28 widened gold) -> -0.00280 (Plan 30-16)
#                -> unchanged by Plan 33.1-04 (the accuracy assertion fires first, so the CLV
#                   reading was never reached while the test was red; it is re-confirmed by the
#                   green run and left at its Plan 30-16 value)
# The Plan 30-16 state was measured 2026-08-24 on the four-rung Phase-30 gold through a backtest
# whose tuned train GENUINELY SEARCHES for the first time: 0.6769095697980685 and
# -0.0028010282747504157, byte-identical across two independent runs.
#
# THE FOURTH STATE, measured 2026-09-13 by Plan 33.1-04, TWICE and identical to every printed
# digit. CAUSE, stated without hedging: D33.1-R3 made WPTrainer a fold-fitted imputation
# Pipeline, so the fill rule is fitted inside each walk-forward fold rather than across the whole
# frame. This drift EXCEEDED the 5e-3 band, which is why it went to the owner as a HALT and was
# re-ratified by ruling rather than absorbed.
#
# SPEC R8: the reading moved UP and that is NOT a result. No model was re-fit or promoted in
# Phase 33.1; a methodology defect was corrected and this number moved as a side effect. The
# accuracy question is not answerable until a re-fit on real weather has run, and none has.
#
# IN-03 convention: a deliberate, DOCUMENTED anchor update -- never silenced, and never absorbed
# by widening the 5e-3 band below. The superseded readings above are kept, not overwritten.
# AUDIT-REPORT.md stays frozen at its v2.1 156/157/156-width figures (it is a historical forensic
# record; Plan 30-14 owns its reconciliation).
ANCHOR_WP_ACCURACY = 0.6821773485513608
ANCHOR_HEADLINE_CLV_WP = -0.00280

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

    @pytest.mark.slow
    def test_backtest_numbers_match_audit_report(self, gold_and_odds_2021_2024) -> None:
        """Backtest WP pooled accuracy ~0.68218 and headline_clv wp ~ -0.00280.

        UN-QUARANTINED by Plan 30-16 Task 3 (2026-08-24). The skip marker placed by Plan 30-15
        under D30-OWNER-05, and converted from xfail(strict) by Plan 30-18 under D30-OWNER-09,
        is REMOVED rather than swapped for another marker: D30-OWNER-05 placed it so that a
        re-passing anchor would force a deliberate re-ratification, and this is that
        re-ratification. The four-rung ladder has stopped moving gold (D30-OWNER-11 accepted
        rung 4) and the tuned search this test runs is now genuine and reproducible, so the
        two conditions the quarantine was waiting on are both discharged.

        The full drift trail, all four anchor states and their causes, is in the module
        docstring. Do NOT widen the 5e-3 band. If this reddens, re-measure TWICE and re-ratify
        with the drift recorded, or report a HALT -- never absorb a drift into the tolerance.

        THAT RULE WAS EXERCISED ON 2026-09-13 and it worked, which is worth recording because a
        rule nobody has ever followed is not evidence of anything. Plan 33.1-04 moved the
        accuracy reading past the band, reported a HALT rather than editing this file, measured
        twice, and the OWNER re-ratified the constant on the re-ratify branch. The band was not
        touched. The superseded readings were not deleted.

        SPEC R8: the reading moved UP and that is NOT a result. Nothing was re-fit or promoted.

        GENERATION-GATED on 2026-09-14 by Plan 33.1-08 Task 2, and the rule above is the
        reason WHY rather than an obstacle to it. Plan 33.1-07's rung-3 rebuild fixed three
        defects that had been throwing away real data -- rainfall discarded for every outdoor
        game, the coverage flag z-scored into the value meaning NO OBSERVATION, and season
        2025's team strength never built -- and gold moved. WP pooled accuracy now measures
        0.6769095697980685, a drift of 0.00527 past the 0.005 band. The band is NOT widened
        and the anchor is NOT edited: both are forbidden above, and the third option is the
        one taken here. The comparison is SET ASIDE, because the anchor was measured on gold
        that no longer exists, and a reading measured on other gold is not a wrong reading.

        This is NOT a quarantine and NOT a deletion. Every other assertion in this module
        still runs. The route back is unchanged and is the owner's: re-measure, re-ratify the
        constant against a generation somebody actually captured, and point the constant below
        at it. Until then this half skips and says so in the terminal.

        Marked slow: a genuine 100-trial search per target costs about 225 seconds.
        """
        from backtest.diagnose import run_diagnosis
        from tests.gold_generation import require_gold_generation
        from tests.phase33_state import (
            GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
        )

        require_gold_generation(
            GOLD_GENERATION_BEFORE_WEATHER_RUNG_UNCAPTURED,
            reading=(
                "AUDIT-REPORT.md's WP pooled accuracy anchor of "
                f"{ANCHOR_WP_ACCURACY} and headline_clv anchor of "
                f"{ANCHOR_HEADLINE_CLV_WP}"
            ),
            moved_by="Phase 33.1's weather rung (Plan 33.1-07, rung 3)",
            recorded_in=(
                "tests.phase33_state.GOLD_DERIVED_READINGS and the repo-root "
                "readout Plan 33.1-11 writes"
            ),
        )

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
