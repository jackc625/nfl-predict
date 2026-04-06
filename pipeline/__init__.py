"""Pipeline orchestrator package for Friday NFL prediction pipeline.

This package provides the unified pipeline orchestrator that replaces
both friday_data_update.py and friday_predictions_run.py with a single
process using direct Python imports via adapter functions.
"""

from pipeline.execution_log import (
    ExecutionLog,
    StepLogEntry,
    write_execution_log_atomic,
)
from pipeline.orchestrator import FridayPipeline
from pipeline.steps import (
    PipelinePhase,
    StepDefinition,
    StepResult,
    StepStatus,
    build_step_registry,
)

__all__ = [
    "ExecutionLog",
    "FridayPipeline",
    "PipelinePhase",
    "StepDefinition",
    "StepLogEntry",
    "StepResult",
    "StepStatus",
    "build_step_registry",
    "write_execution_log_atomic",
]
