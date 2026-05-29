"""End-to-end stage runner + current-week dry-run harness (AUDIT-01, Phase 20 / D-06).

This is the re-runnable verification harness that proves AUDIT-01: every pipeline
stage runs end-to-end on current data, exercising BOTH stage paths per D-06 --

  (a) the historical/full path -- each of PIPELINE.md's 7 canonical stages has an
      importable live entry point whose ``--help``/main is callable exit-0 (we prove
      the stage entry points run; the single full historical gold rebuild is Wave 3's
      job, NOT this harness -- D-10 catalog-then-batch), and

  (b) a simulated current-week dry-run driving the incremental code paths
      (``--current`` / ``--season --week``) against the most-recent fully-completed
      week. This is where the cb61042 silent break and the CR-01 abort lived; proving
      these run end-to-end without a live game is the heart of D-06.

Stand-in "current week" for the dry-run: **season 2024, week 18** -- the latest
fully-completed week with definitely-known answers and 16 games on disk in gold
(verified 2026-05-28). A late 2024 week is preferred over a partial-2025 week per
RESEARCH Open Question 1.

Scope guards (honored here):
  - D-01: NO re-training. The predict dry-run LOADS artifacts via ``load_model_artifact``;
    no training entry point is invoked.
  - D-06: the dry-run STAGES the incremental scripts; it does NOT wrap them in the
    Friday orchestrator (that is Phase 21 / AUTO-01).
  - Re-runnable / idempotent: the predict dry-run writes only to ``tmp_path``; the Elo
    incremental run uses ``--no-save``; the team-form incremental run patches out the
    ``save_dataframe`` side effect so the harness never grows the real silver lake. The
    non-idempotent ``team_game_stats`` append and the deprecated team-form ``--current``
    path are cataloged as findings in AUDIT-REPORT.md (D-10), not fixed here.

Mirrors the import/subprocess style of ``tests/integration/test_training_pipeline.py``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

# The stand-in "current week" for the dry-run (D-06 discretion): latest fully-settled
# week with known answers and games on disk. Verified 2024 wk18 = 16 gold rows.
STANDIN_SEASON = 2024
STANDIN_WEEK = 18

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# The 7 canonical PIPELINE.md stages -> their live entry-point scripts. Each must be
# importable and expose a callable ``--help`` (argparse main) that exits 0.
PIPELINE_STAGE_SCRIPTS = [
    # Stage 1: ingest
    "ingest_games.py",
    "ingest_odds.py",
    "ingest_weather.py",
    # Stage 2: features
    "build_elo.py",
    "build_team_form.py",
    "build_contextual.py",
    "build_weather.py",
    "build_market_anchors.py",
    "build_features.py",
    # Stage 3: train
    "train_models.py",
    # Stage 4: backtest
    "run_backtest.py",
    # Stage 5: predict
    "generate_current_week_predictions.py",
    # Stage 6: build-cache
    "populate_cache.py",
]


def _run_script_help(script_name: str) -> subprocess.CompletedProcess[str]:
    """Invoke ``uv run python scripts/<script> --help`` from the project root."""
    return subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / script_name), "--help"],
        capture_output=True,
        text=True,
        cwd=str(PROJECT_ROOT),
        check=False,
    )


# ---------------------------------------------------------------------------
# (a) Historical / full path -- stage entry points run (exit-0), do NOT rebuild gold
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestHistoricalStageEntryPoints:
    """Each PIPELINE.md stage's live entry point is importable and runs (exit-0)."""

    @pytest.mark.parametrize("script_name", PIPELINE_STAGE_SCRIPTS)
    def test_stage_script_help_exits_zero(self, script_name: str) -> None:
        """Every canonical stage script exposes a callable ``--help`` (exit 0).

        This proves the historical/full-path entry points run without an import or
        argparse error; the actual full historical rebuild is exercised by Wave 3's
        single rebuild (D-10), not here.
        """
        result = _run_script_help(script_name)
        assert result.returncode == 0, (
            f"{script_name} --help exited {result.returncode}\nstderr:\n{result.stderr}"
        )

    def test_serve_stage_app_importable(self) -> None:
        """Stage 7 (serve): ``api.main:app`` is importable (FastAPI app exists)."""
        from api.main import app

        assert app is not None
        # FastAPI app exposes a routable interface
        assert hasattr(app, "routes")


# ---------------------------------------------------------------------------
# (b) Current-week dry-run -- incremental code paths run end-to-end (no live game)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestCurrentWeekPredictDryRun:
    """Predict path against a completed week: artifacts LOADED, predictions written."""

    def test_generate_and_write_produces_nonempty_predictions(
        self, tmp_path: Path
    ) -> None:
        """``generate_and_write`` writes a non-empty predictions CSV for a completed week.

        Drives the prediction stage against the stand-in current week (2024 wk18). The
        models are LOADED via ``load_model_artifact`` inside the prediction path -- no
        training happens (D-01). Output is written only to ``tmp_path`` so the harness
        is re-runnable.
        """
        from scripts.generate_current_week_predictions import generate_and_write

        summary = generate_and_write(
            season=STANDIN_SEASON,
            week=STANDIN_WEEK,
            output_dir=tmp_path,
            artifacts_dir=PROJECT_ROOT / "artifacts",
        )

        # The predict path returns a summary and wrote the prediction CSV + context CSV.
        assert summary["n_games"] > 0, "Expected at least one game for 2024 wk18"

        pred_csv = tmp_path / f"predictions_{STANDIN_SEASON}_week{STANDIN_WEEK}.csv"
        assert pred_csv.exists(), f"Predictions CSV not written: {pred_csv}"
        assert pred_csv.stat().st_size > 0, "Predictions CSV is empty"

        pred_df = pd.read_csv(pred_csv)
        assert len(pred_df) > 0, "Predictions CSV has no rows"
        # Sanity: the three model targets produced columns in the output.
        assert "wp_prob" in pred_df.columns
        assert "ats_prediction" in pred_df.columns
        assert "ou_prediction" in pred_df.columns

        # The predict stage also writes the per-game context CSV alongside it.
        context_csv = tmp_path / f"game_context_{STANDIN_SEASON}_week{STANDIN_WEEK}.csv"
        assert context_csv.exists(), f"Game context CSV not written: {context_csv}"

    def test_predict_dry_run_does_not_train(self, monkeypatch) -> None:
        """The predict dry-run never invokes a trainer (D-01 guard).

        We fail loudly if ``run_predictions`` ever reaches into the trainer modules:
        the prediction path must LOAD artifacts, not fit models.
        """
        import models.trainers.base as base_module

        def _explode(*_args, **_kwargs):  # pragma: no cover - guard
            raise AssertionError(
                "Predict dry-run must not train models (D-01 violated)"
            )

        # Any attempt to instantiate the base trainer during prediction is a violation.
        monkeypatch.setattr(base_module.BaseTrainer, "train_and_evaluate", _explode)

        from scripts.generate_current_week_predictions import run_predictions

        results = run_predictions(
            PROJECT_ROOT / "artifacts", STANDIN_SEASON, STANDIN_WEEK
        )
        # Predictions came back for all three targets without touching a trainer.
        assert set(results.keys()) == {"wp", "ats", "ou"}
        for target, frame in results.items():
            assert len(frame) > 0, f"No {target} predictions for the stand-in week"


@pytest.mark.integration
class TestCurrentWeekIncrementalBuilders:
    """The ``--current`` incremental builder paths run end-to-end without error."""

    def test_build_elo_current_dry_run_exits_zero(self) -> None:
        """``build_elo.py --current --no-save`` runs the incremental Elo path (exit 0).

        ``--no-save`` keeps the harness re-runnable: it exercises the full incremental
        Elo update for the current season but does not mutate the silver lake.
        """
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "build_elo.py"),
                "--current",
                "--no-save",
            ],
            capture_output=True,
            text=True,
            cwd=str(PROJECT_ROOT),
            check=False,
        )
        assert result.returncode == 0, (
            f"build_elo.py --current --no-save exited {result.returncode}\n"
            f"stderr:\n{result.stderr}"
        )

    def test_build_team_form_current_path_runs(self, monkeypatch) -> None:
        """The team-form ``--current`` incremental path runs and returns features.

        We invoke ``TeamFormBuilder.build_for_current_week`` -- the exact code path the
        ``build_team_form.py --current`` CLI runs -- but patch out the ``save_dataframe``
        silver write so the harness stays idempotent.

        WR-01 fix (committed in phase 20): the current-week path passes only a
        3-season subset, so it MUST NOT overwrite the full-history
        ``team_game_stats`` silver table (``replace_mode=True`` on a subset would
        shrink the on-disk 2002-2024 history to 3 seasons -- a data-loss
        regression). The persisted write is skipped on the incremental path; the
        full-rebuild path (``build_for_seasons``) remains the canonical producer of
        the complete table. This test asserts that the current-week path runs
        end-to-end AND does not attempt any persisted silver write.
        """
        import features.team_form as team_form_module
        from scripts.build_team_form import TeamFormBuilder

        saved_tables: list[str] = []

        def _capture_save(_df, table, *_args, **_kwargs):
            # Record the write intent without touching the real lake. The silver write
            # lives in features.team_form (the calculator), so we patch it there.
            saved_tables.append(table)

        monkeypatch.setattr(team_form_module, "save_dataframe", _capture_save)

        builder = TeamFormBuilder(rolling_weeks=4)
        form_df = builder.build_for_current_week()

        # The incremental path ran end-to-end and produced rolling team-form features.
        assert isinstance(form_df, pd.DataFrame)
        assert len(form_df) > 0, "Team-form current-week path produced no features"
        # WR-01: the current-week subset path must NOT persist team_game_stats
        # (doing so would destroy the full-history table). It returns the rolling
        # features in-memory without mutating the silver artifact.
        assert "team_game_stats" not in saved_tables, (
            "WR-01 regression: the current-week path must not overwrite the "
            "full-history team_game_stats silver table"
        )
