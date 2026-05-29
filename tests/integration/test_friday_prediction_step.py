"""Integration tests for the Friday pipeline's prediction-generation steps.

These run the REAL step bodies (load model artifacts, predict against the gold
feature matrices, apply blending, write output) -- the gap that OPS-01 hid,
because the orchestrator unit tests replace the whole step registry with stubs
and therefore never execute the actual step functions.

``test_orchestrator_predictions_phase_e2e`` is the durable AUTO-01 guard against
a cb61042-class silent break at the ORCHESTRATOR boundary: where the sibling
``test_prediction_phase_steps_run_unmocked`` calls the step bodies directly, this
test drives the REAL ``FridayPipeline(mode="predictions-only", force=True).run()``
through the REAL step registry (NO ``build_step_registry`` patch -- that patch IS
the cb61042 aliasing blind spot), proving the whole orchestrator loop actually
produces predictions.

Marked ``slow`` because they load real model artifacts and gold matrices.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

# A season/week guaranteed to exist in the committed gold feature matrices.
_SEASON = 2024
_WEEK = 1


def _gold_has_season_week(season: int, week: int) -> bool:
    """True when the gold WP matrix contains rows for the given season/week."""
    path = Path("data/gold/features_wp.parquet")
    if not path.exists():
        return False
    df = pd.read_parquet(path, columns=["season", "week"])
    return bool(((df["season"] == season) & (df["week"] == week)).any())


@pytest.mark.slow
def test_generate_and_write_produces_real_predictions(tmp_path):
    """generate_and_write loads models, predicts, blends, and persists a file."""
    if not _gold_has_season_week(_SEASON, _WEEK):
        pytest.skip(f"gold matrix lacks {_SEASON} week {_WEEK}")

    from scripts.generate_current_week_predictions import generate_and_write

    summary = generate_and_write(season=_SEASON, week=_WEEK, output_dir=tmp_path)

    assert summary["n_games"] > 0
    pred_path = tmp_path / f"predictions_{_SEASON}_week{_WEEK}.csv"
    assert pred_path.exists()

    df = pd.read_csv(pred_path)
    assert len(df) == summary["n_games"]
    for col in ("game_id", "wp_prob", "ats_prediction", "ou_prediction"):
        assert col in df.columns
    assert df["wp_prob"].between(0.0, 1.0).all()


@pytest.mark.slow
def test_prediction_phase_steps_run_unmocked(tmp_path, monkeypatch):
    """The orchestrator step bodies run end-to-end (no mocked step registry).

    Regression guard for OPS-01: step_generate_predictions previously created a
    pipeline without loading models and never persisted output, so a live run
    raised ValueError. This exercises the real bodies in sequence.
    """
    if not _gold_has_season_week(_SEASON, _WEEK):
        pytest.skip(f"gold matrix lacks {_SEASON} week {_WEEK}")

    from pipeline import steps

    # Redirect output into tmp_path and pin the "current" week deterministically.
    monkeypatch.setattr(steps, "_predictions_output_dir", lambda: tmp_path)
    monkeypatch.setattr(
        "utils.date_utils.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )

    steps.step_generate_predictions()
    steps.step_export_artifacts()
    steps.step_generate_recommendations()
    steps.step_validate_predictions()  # raises on any failure
    steps.step_verify_output_files()

    assert (tmp_path / f"predictions_{_SEASON}_week{_WEEK}.csv").exists()
    assert (tmp_path / f"predictions_{_SEASON}_week{_WEEK}.json").exists()
    assert (tmp_path / f"recommendations_{_SEASON}_week{_WEEK}.json").exists()
    assert (tmp_path / f"game_context_{_SEASON}_week{_WEEK}.csv").exists()


@pytest.mark.slow
def test_orchestrator_predictions_phase_e2e(tmp_path, monkeypatch):
    """The REAL FridayPipeline predictions phase produces predictions end-to-end.

    Durable AUTO-01 guard (D-01/D-02) against a cb61042-class silent break at the
    ORCHESTRATOR boundary. Unlike the sibling
    ``test_prediction_phase_steps_run_unmocked`` (which calls the step bodies
    directly), this drives the REAL ``FridayPipeline(mode="predictions-only",
    force=True).run()`` through the REAL step registry. It intentionally does NOT
    patch ``build_step_registry`` and does NOT use ``make_mock_step`` -- stubbing
    the registry is exactly the aliasing blind spot that hid cb61042, so doing so
    here would void the proof.

    Offseason-safe and no-live-network: the week is pinned to the committed 2024
    gold, and the three network/rebuild-touching predictions-phase steps
    (``step_ingest_odds``, ``step_build_market_anchors``, ``step_build_features``)
    are no-op'd because the committed gold already contains those features
    (carried Phase 20 D-04: no live odds pull). The generate / validate / export /
    verify step bodies still run for real.
    """
    if not _gold_has_season_week(_SEASON, _WEEK):
        pytest.skip(f"gold matrix lacks {_SEASON} week {_WEEK}")

    from pipeline import steps
    from pipeline.orchestrator import FridayPipeline

    # Pin the "current" week at BOTH import sites: the orchestrator resolves
    # (season, week) in __init__ via pipeline.orchestrator.get_current_nfl_week,
    # and step_generate_predictions re-imports it deferred from utils.date_utils.
    monkeypatch.setattr(
        "pipeline.orchestrator.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )
    monkeypatch.setattr(
        "utils.date_utils.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )

    # Redirect every prediction-phase artifact into tmp_path via the one helper.
    monkeypatch.setattr(steps, "_predictions_output_dir", lambda: tmp_path)

    # No-op the network/rebuild-touching steps (D-02): the committed 2024 gold
    # already contains the odds/market/feature-matrix inputs, so we keep the REAL
    # orchestrator loop + REAL generate/validate/export bodies without a live pull.
    monkeypatch.setattr(steps, "step_ingest_odds", lambda: None)
    monkeypatch.setattr(steps, "step_build_market_anchors", lambda: None)
    monkeypatch.setattr(steps, "step_build_features", lambda: None)

    # Drive the REAL orchestrator loop -- no build_step_registry patch.
    pipeline = FridayPipeline(mode="predictions-only", force=True)
    log = pipeline.run()

    assert log.status == "success"

    pred_path = tmp_path / f"predictions_{_SEASON}_week{_WEEK}.csv"
    assert pred_path.exists()

    df = pd.read_csv(pred_path)
    assert "game_id" in df.columns
    assert df["wp_prob"].between(0.0, 1.0).all()

    # The export/recommendation/context artifacts the orchestrator loop produces.
    assert (tmp_path / f"predictions_{_SEASON}_week{_WEEK}.json").exists()
    assert (tmp_path / f"recommendations_{_SEASON}_week{_WEEK}.json").exists()
    assert (tmp_path / f"game_context_{_SEASON}_week{_WEEK}.csv").exists()
