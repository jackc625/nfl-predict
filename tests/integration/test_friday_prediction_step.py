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

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

# A season/week guaranteed to exist in the committed gold feature matrices.
_SEASON = 2024
_WEEK = 1


def _pin_bet_list_run_instant(monkeypatch) -> None:
    """Pin the bet-list run clock to a PRE-FREEZE instant for the pinned week.

    Phase 33, Plan 33-05 Task 3. These tests pin the WEEK to a historical one so they can run in
    the offseason against committed gold, but they left the CLOCK real -- and a forward run whose
    observation time falls after its own game freeze is now refused by name (R7), correctly. The
    combination "week 1 of 2024, decided in 2026" is exactly the post-hoc pick the fence exists to
    stop, so the fix is to pin the clock alongside the week rather than to weaken the fence.

    In PRODUCTION nothing here applies: ``get_current_nfl_week`` returns the current week, so the
    run clock is naturally before that week's freeze.

    The instant is one second before the EARLIEST 2024 week-1 freeze (Thursday 2024-09-05's
    kickoff freezes on Friday 2024-08-30 at 18:00 ET = 22:00 UTC), so every game in the week
    satisfies ``decided_at <= freeze``. The real function is called -- only its ``now`` is bound.
    """
    import backtest.weekly_bet_list as wbl

    run_instant = datetime(2024, 8, 30, 21, 59, 59, tzinfo=UTC)
    real = wbl.generate_weekly_bet_list
    monkeypatch.setattr(
        wbl,
        "generate_weekly_bet_list",
        lambda **kwargs: real(now=run_instant, **kwargs),
    )


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

    bet_list_dir = tmp_path / "bet_list"
    monkeypatch.setattr(steps, "_bet_list_output_dir", lambda: bet_list_dir)
    _pin_bet_list_run_instant(monkeypatch)

    steps.step_generate_predictions()
    steps.step_export_artifacts()
    steps.step_generate_recommendations()
    steps.step_validate_predictions()  # raises on any failure
    steps.step_verify_output_files()

    assert (tmp_path / f"predictions_{_SEASON}_week{_WEEK}.csv").exists()
    assert (tmp_path / f"predictions_{_SEASON}_week{_WEEK}.json").exists()
    assert (tmp_path / f"game_context_{_SEASON}_week{_WEEK}.csv").exists()

    # RETIRED by plan 31-17 (D31-32): the legacy weekly JSON had no code consumer and its
    # confidence-tier selection criterion is the helper renamed in the same wave. Asserting its
    # ABSENCE is what makes the retirement a fact this suite re-checks on every run.
    assert not (tmp_path / f"recommendations_{_SEASON}_week{_WEEK}.json").exists()

    # What replaced it: the two durable bet-list artifacts, under their OWN directory.
    assert (bet_list_dir / "bet_list.parquet").exists()
    assert (bet_list_dir / "bet_tracker.json").exists()


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

    from pipeline import orchestrator, steps
    from pipeline.orchestrator import FridayPipeline

    # Redirect the orchestrator's execution log into tmp_path so the test is
    # hermetic. ``LOG_PATH`` is a module global the orchestrator writes on every
    # step + finalize; without this redirect the run clobbers the repo's real
    # ``logs/friday_pipeline.json`` -- the genuine runtime artifact a developer or
    # the scheduler inspects after a live Friday run.
    monkeypatch.setattr(orchestrator, "LOG_PATH", tmp_path / "friday_pipeline.json")

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
    bet_list_dir = tmp_path / "bet_list"
    monkeypatch.setattr(steps, "_bet_list_output_dir", lambda: bet_list_dir)
    _pin_bet_list_run_instant(monkeypatch)

    # No-op the network/rebuild-touching steps (D-02): the committed 2024 gold
    # already contains the odds/market/feature-matrix inputs, so we keep the REAL
    # orchestrator loop + REAL generate/validate/export bodies without a live pull.
    monkeypatch.setattr(steps, "step_ingest_odds", lambda: None)
    monkeypatch.setattr(steps, "step_build_market_anchors", lambda: None)
    monkeypatch.setattr(steps, "step_build_features", lambda: None)

    # And the step plan 31-18 added, for the SAME reason plus a harder one: its real body
    # rebuilds the PRODUCTION `data/web_cache.duckdb` in place. Leaving it live would make this
    # test rewrite a production store as a side effect of asserting the prediction phase -- the
    # exact class of unintended `data/` write `tests/data_boundary.py` exists to catch. Its own
    # registration, ordering and non-critical failure behaviour are proven in
    # `tests/unit/test_step_registry_order.py` and
    # `tests/integration/test_orchestrator_degraded_cache.py`, so nothing is lost by no-op'ing it
    # here. Asserted present first, so a rename cannot turn this into a silent no-op that lets the
    # real body run again.
    assert hasattr(steps, "step_populate_web_cache"), (
        "pipeline.steps no longer defines step_populate_web_cache; this no-op has gone stale and "
        "the real cache rebuild would run against the production cache"
    )
    monkeypatch.setattr(steps, "step_populate_web_cache", lambda: None)

    # Drive the REAL orchestrator loop -- no build_step_registry patch.
    pipeline = FridayPipeline(mode="predictions-only", force=True)
    log = pipeline.run()

    assert log.status == "success"

    pred_path = tmp_path / f"predictions_{_SEASON}_week{_WEEK}.csv"
    assert pred_path.exists()

    df = pd.read_csv(pred_path)
    assert "game_id" in df.columns
    assert df["wp_prob"].between(0.0, 1.0).all()

    # The export/bet-list/context artifacts the orchestrator loop produces.
    assert (tmp_path / f"predictions_{_SEASON}_week{_WEEK}.json").exists()
    assert (tmp_path / f"game_context_{_SEASON}_week{_WEEK}.csv").exists()
    assert (bet_list_dir / "bet_list.parquet").exists()
    assert (bet_list_dir / "bet_tracker.json").exists()

    # The retired weekly JSON (D31-32) is not produced by the REAL orchestrator loop either.
    assert not (tmp_path / f"recommendations_{_SEASON}_week{_WEEK}.json").exists()


@pytest.mark.slow
def test_orchestrator_data_phase_reaches_verify_gate(tmp_path, monkeypatch):
    """The REAL FridayPipeline DATA phase reaches and passes the verify gate (B1).

    This closes the v2.1 milestone-audit BLOCKER (AUTO-01): a full-mode Friday run
    aborted at ``step_verify_data_artifacts`` (DATA step 8/18, ``critical=True``)
    because the gate required two silver filenames -- ``elo_ratings.parquet`` /
    ``team_form.parquet`` -- that NO build script writes. The durable AUTO-01 guard
    above never caught it because it runs ``mode="predictions-only"``, which
    structurally filters OUT every ``PipelinePhase.DATA`` step (the gate included) --
    the exact undersampling blind spot.

    This test samples the failure directly: it drives the REAL step registry through
    the DATA phase so the REAL ``step_verify_data_artifacts`` body executes against
    the actual on-disk silver+gold layout and is asserted to return success.

    Mode note (Open Question 2): the SCHEDULER uses ``mode="full"`` (no flag in
    ``deployment/windows_scheduler.xml`` -> ``friday_pipeline.py`` defaults to full).
    ``mode="data-only"`` is the minimal superset of full that runs the SAME
    ``critical=True`` DATA-phase gate (full = DATA + PREDICTIONS; data-only = DATA),
    so the proof transfers while avoiding a live odds pull. It is deliberately NOT
    ``mode="predictions-only"`` (that filters the gate out -- the bug) and does NOT
    patch ``build_step_registry`` or use ``make_mock_step`` (the cb61042 aliasing
    blind spot that would prevent the real gate body from ever running, D-05).

    HARD BOUNDARY (D-01): all seven non-gate DATA-phase step bodies are no-op'd, so
    the test never re-ingests, never hits the network, never rebuilds Elo/form, and
    never rewrites ``data/silver/`` or ``data/gold/``. The ONLY real body is
    ``step_verify_data_artifacts`` (pure path-existence checks). Skip-guarded on the
    real silver + gold layout so it skips cleanly when the data layer is absent.
    """
    # Skip-guard on the REAL on-disk layout (Pitfall 2: do NOT chdir to tmp_path --
    # the gate is CWD-relative and must check the repo's real data/ tree).
    if not Path("data/silver/elo_game_snapshots.parquet").exists():
        pytest.skip("real silver layout absent")
    if not _gold_has_season_week(_SEASON, _WEEK):
        pytest.skip(f"gold matrix lacks {_SEASON} week {_WEEK}")

    from pipeline import orchestrator, steps
    from pipeline.orchestrator import FridayPipeline

    # Redirect the orchestrator's execution log into tmp_path so the test is
    # hermetic. ``LOG_PATH`` is a module global the orchestrator writes on every
    # step + finalize; without this redirect the run clobbers the repo's real
    # ``logs/friday_pipeline.json`` -- the genuine runtime artifact a developer or
    # the scheduler inspects after a live Friday run.
    monkeypatch.setattr(orchestrator, "LOG_PATH", tmp_path / "friday_pipeline.json")

    # Pin the "current" week at BOTH import sites (Pitfall 4): the orchestrator
    # resolves (season, week) in __init__ via pipeline.orchestrator.get_current_nfl_week,
    # and step bodies re-import it deferred from utils.date_utils. The gate itself
    # does not use the week, but the pre-flight gates + log do.
    monkeypatch.setattr(
        "pipeline.orchestrator.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )
    monkeypatch.setattr(
        "utils.date_utils.get_current_nfl_week", lambda: (_SEASON, _WEEK)
    )

    # No-op ONLY the seven non-gate DATA-phase step bodies (D-01: no rebuild). This
    # is the no-op-the-rebuild idiom the predictions-phase E2E uses, swapped to the
    # DATA-phase steps. step_verify_data_artifacts is DELIBERATELY NOT patched -- it
    # must run for real against the on-disk layout (that is the whole point).
    for name in (
        "step_ingest_games",
        "step_ingest_weather",
        "step_data_qa",
        "step_build_elo",
        "step_build_team_form",
        "step_build_contextual",
        "step_build_weather_features",
    ):
        monkeypatch.setattr(steps, name, lambda: None)

    # Drive the REAL orchestrator loop through the DATA phase -- no build_step_registry
    # patch, no make_mock_step (D-05). force=True makes the pre-flight staleness/health
    # gates advisory so the run reaches the step loop out of season too.
    pipeline = FridayPipeline(mode="data-only", force=True)
    log = pipeline.run()

    assert log.status == "success"

    # The gate step ran (predictions-only would have filtered it out) and passed.
    step_names = [s.name for s in log.steps]
    assert "verify_data_artifacts" in step_names
    gate = next(s for s in log.steps if s.name == "verify_data_artifacts")
    assert gate.status == "success"  # StepStatus.SUCCESS.value as stored in the log
