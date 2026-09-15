"""A weather-features refusal STOPS the Friday pipeline before gold is built.

THE PATH THIS CLOSES. ``build_weather_features`` refuses a game whose weather cannot be
accounted for, and ``build_features`` -- which rebuilds gold -- runs four steps later. While
the step was registered ``critical=False`` the orchestrator recorded the refusal as a warning,
marked the run ``degraded`` and carried straight on, so gold was rebuilt from a weather-features
table that did not cover the week being predicted. That is how a silent no-weather gold was
reachable. Owner ruling W1 of 2026-09-15 made the step critical; this module pins the
consequence at the orchestrator, not only at the registry flag.

It drives the REAL ``FridayPipeline.run`` over the REAL registry's names, phases and critical
flags, with every step body replaced by a recorder and the weather-features step raising the
builder's own refusal. The pre-flight gates, alerting and log writes are stubbed exactly as
``tests/unit/test_pipeline_orchestrator.py`` stubs them. Nothing touches a store.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from features.weather import WeatherObservationError
from pipeline.staleness import StalenessResult
from pipeline.steps import StepDefinition, build_step_registry

pytestmark = pytest.mark.integration


def _recording_registry(calls: list[str]) -> list[StepDefinition]:
    """The live registry with each body replaced: record the name, or raise the refusal."""

    def body_for(name: str):
        def body() -> None:
            calls.append(name)
            if name == "build_weather_features":
                raise WeatherObservationError(
                    "game '2026_W02_CAR@ATL' has NO row in the silver `weather` table."
                )

        return body

    return [
        StepDefinition(
            name=step.name,
            callable=body_for(step.name),
            phase=step.phase,
            critical=step.critical,
            retryable=step.retryable,
            max_retries=step.max_retries,
            description=step.description,
        )
        for step in build_step_registry()
    ]


@pytest.fixture
def pipeline_run():
    staleness = MagicMock()
    staleness.run_all_checks.return_value = StalenessResult(passed=True)
    health = MagicMock()
    health.run_preflight.return_value = {"status": "healthy", "checks": []}
    health.run_postrun.return_value = {"status": "healthy", "checks": []}
    calls: list[str] = []
    with (
        patch("pipeline.orchestrator.get_current_nfl_week", return_value=(2026, 2)),
        patch("pipeline.orchestrator.write_execution_log_atomic"),
        patch("pipeline.orchestrator.StalenessGate", return_value=staleness),
        patch("pipeline.orchestrator.PipelineHealthChecker", return_value=health),
        patch("pipeline.orchestrator.PipelineAlertManager", return_value=MagicMock()),
        patch(
            "pipeline.orchestrator.build_step_registry",
            return_value=_recording_registry(calls),
        ),
    ):
        from pipeline.orchestrator import FridayPipeline

        yield FridayPipeline(force=True), calls


class TestAWeatherFeaturesRefusalStopsTheRun:
    def test_the_run_raises(self, pipeline_run):
        pipeline, _ = pipeline_run
        with pytest.raises(RuntimeError, match="NO row in the silver"):
            pipeline.run()

    def test_build_features_never_runs(self, pipeline_run):
        pipeline, calls = pipeline_run
        with pytest.raises(RuntimeError):
            pipeline.run()
        assert "build_weather_features" in calls, "the refusing step never ran at all"
        assert "build_features" not in calls, (
            f"gold was rebuilt after the weather-features refusal; steps run: {calls}"
        )

    def test_the_run_is_recorded_failed_not_degraded(self, pipeline_run):
        pipeline, _ = pipeline_run
        with pytest.raises(RuntimeError):
            pipeline.run()
        assert pipeline.execution_log.status == "failed"
