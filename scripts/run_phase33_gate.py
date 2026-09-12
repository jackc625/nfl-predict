"""INTERFACE STUB for the Phase-33 deploy-gate runner (Plan 33-08 Task 3, RED phase).

Every public name Plan 33-08 declares is present here and NOTHING is implemented. The
stub exists for one mechanical reason: five test modules import this module, and without
it pytest fails at COLLECTION -- which is INVALID_RED under the phase's own TDD gate
(a load failure proves nothing about behaviour and must not authorize GREEN). With the
names declared, the suite collects and the failures are real assertions about behaviour
that does not exist yet.

The implementation, its rationale and the frozen pre-registration land in the GREEN
commit that replaces this file.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

# Deliberately NOT the pre-registered value: the RED phase must not satisfy the
# pre-registration assertion by accident.
PHASE33_FIX_CYCLE_ALLOWANCE: int = -1
VERDICT_STATES: tuple[str, ...] = ()
GATED_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")
VERDICT_RECORD_PATH = Path("outputs/phase33_gate_verdict.json")
COMMITTED_VERDICT_PATH = Path("config/phase33_gate_verdict.toml")
PRODUCTION_ARTIFACTS_DIR = Path("artifacts")
STAGING_ARTIFACTS_DIR = Path("artifacts_staging")
MIN_FREE_DISK_BYTES: int = 0

deploy_gate: Any = None
update_manifest: Any = None


class UntestableRefusalReason(StrEnum):
    """Not yet declared."""

    UNIMPLEMENTED = "unimplemented"


class PreflightFailedError(RuntimeError):
    """Not yet implemented."""


class MissingStageOneVerdictError(RuntimeError):
    """Not yet implemented."""


class FixCycleAllowanceExceededError(RuntimeError):
    """Not yet implemented."""


class MalformedVerdictRecordError(ValueError):
    """Not yet implemented."""


@dataclass(frozen=True)
class TargetScoring:
    """Not yet implemented."""

    target: str = ""
    candidate_version: str = ""
    incumbent_version: str = ""
    candidate_bundle: dict[str, Any] = field(default_factory=dict)
    incumbent_scored: Any = None
    candidate_scored: Any = None
    gold: Any = None
    paired_delta: Any = None


@dataclass(frozen=True)
class GateVerdictRecord:
    """Not yet implemented."""

    judge_version: str = ""
    judge_code_digest: str = ""
    rendered_at: str = ""
    fix_cycle_allowance: int = -1
    verdict_states: tuple[str, ...] = ()
    secondary_scalar_names: tuple[str, ...] = ()
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateVerdictRecord:
        raise NotImplementedError

    def passing_targets(self) -> tuple[str, ...]:
        raise NotImplementedError


def validate_verdict_payload(payload: Mapping[str, Any]) -> None:
    """Not yet implemented."""
    raise NotImplementedError


def preflight_health_check(**kwargs: Any) -> dict[str, Any]:
    """Not yet implemented."""
    raise NotImplementedError


def render_target_verdict(
    target: str, scoring: TargetScoring, cfg: dict[str, Any]
) -> dict[str, Any]:
    """Not yet implemented."""
    raise NotImplementedError


def _refuse_shared_roots(artifacts_dir: Path, staging_dir: Path) -> None:
    """Not yet implemented."""
    raise NotImplementedError


def _refuse_second_candidate(record_path: Path, targets: Sequence[str]) -> None:
    """Not yet implemented."""
    raise NotImplementedError


def stage_one_judge(
    *, scorer: Callable[..., TargetScoring], cfg: dict[str, Any], **kwargs: Any
) -> GateVerdictRecord:
    """Not yet implemented."""
    raise NotImplementedError


def read_verdict_record(
    verdict_record_path: Path = VERDICT_RECORD_PATH,
) -> GateVerdictRecord:
    """Not yet implemented."""
    raise NotImplementedError


def stage_two_promote(**kwargs: Any) -> tuple[str, ...]:
    """Not yet implemented."""
    raise NotImplementedError


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Not yet implemented."""
    raise NotImplementedError


def main(argv: list[str] | None = None) -> int:
    """Not yet implemented."""
    raise NotImplementedError
