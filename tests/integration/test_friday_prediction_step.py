"""Integration tests for the Friday pipeline's prediction-generation steps.

These run the REAL step bodies (load model artifacts, predict against the gold
feature matrices, apply blending, write output) -- the gap that OPS-01 hid,
because the orchestrator unit tests replace the whole step registry with stubs
and therefore never execute the actual step functions.

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
