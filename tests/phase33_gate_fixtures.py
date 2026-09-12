"""Shared builders for the Phase-33 gate-runner tests. NOT a test module.

Phase 33, Plan 33-08 Task 3 (COLD-04).

WHY THIS FILE EXISTS. Three modules -- ``test_deploy_gate_empty_pairs.py``,
``test_deploy_gate_order_invariance.py`` and ``test_phase33_gate_runner.py`` -- all need
the same two things: a SANDBOXED pair of artifact roots that satisfies the pre-flight,
and a deterministic scorer that produces a ``TargetScoring`` without training anything.
Copying those into three modules would be three places for the sandbox to drift, and the
one drift that matters is a test that reaches the REAL ``artifacts/`` root.

WHY IT LIVES IN ``tests/`` AND NOT ``tests/unit/``. It sits beside ``phase33_state.py``
and ``data_boundary.py``, the repository's existing home for shared test infrastructure,
and its name carries no ``test_`` prefix so pytest never collects it.

THE SANDBOX IS THE POINT, NOT A CONVENIENCE. Plan 33-06 overwrote all three production
gold matrices because a test called a helper that wrote, without the sandbox fixture.
Plan 33-07 came within one name of fetching over the network and rewriting a committed
manifest. ``sandbox_roots`` is therefore FAIL-CLOSED: it refuses to hand back a root that
resolves inside the repository's real ``artifacts/`` tree, so a caller that forgets to
pass ``tmp_path`` gets an exception rather than a production write.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_ARTIFACTS_ROOT = (REPO_ROOT / "artifacts").resolve()

SEASONS: tuple[int, ...] = (2021, 2022, 2023, 2024)

# The pre-phase production manifest shape: FOUR entries, `blend` included, because the
# swap surface is not three independent slots.
INCUMBENT_VERSIONS: dict[str, str] = {
    "wp": "wp_20260824_113325",
    "ats": "ats_20260605_220128",
    "ou": "ou_20260326_163930",
    "blend": "blend_dynamic_20260606_020635",
}
CANDIDATE_VERSIONS: dict[str, str] = {
    "wp": "wp_20990101_000001",
    "ats": "ats_20990101_000002",
    "ou": "ou_20990101_000003",
}

_PREDICTION_COLUMN: dict[str, str] = {
    "wp": "model_prob",
    "ats": "model_spread",
    "ou": "model_total",
}


class SandboxEscapeError(AssertionError):
    """A sandbox root resolved inside the repository's real ``artifacts/`` tree.

    Fail-closed by design (see the module docstring). A stop beats a detector: this
    raises BEFORE anything is written, rather than reporting the write afterwards.
    """


@dataclass(frozen=True)
class SandboxRoots:
    """A production/staging pair that is provably not the real production tree."""

    artifacts: Path
    staging: Path


def sandbox_roots(tmp_path: Path) -> SandboxRoots:
    """Build a pre-flight-satisfying sandbox under *tmp_path*.

    Creates a production root with a four-entry ``latest.json`` and a resolvable
    ``metadata.json`` for every incumbent, plus an empty staging root.

    Raises:
        SandboxEscapeError: If either root would resolve inside the real ``artifacts/``.
    """
    artifacts = (tmp_path / "artifacts").resolve()
    staging = (tmp_path / "artifacts_staging").resolve()
    for root in (artifacts, staging):
        if root == REAL_ARTIFACTS_ROOT or REAL_ARTIFACTS_ROOT in root.parents:
            msg = (
                f"refusing to build a sandbox at '{root}': it resolves inside the "
                f"repository's real artifacts tree at '{REAL_ARTIFACTS_ROOT}'. Pass a "
                "tmp_path-derived root."
            )
            raise SandboxEscapeError(msg)

    artifacts.mkdir(parents=True, exist_ok=True)
    staging.mkdir(parents=True, exist_ok=True)
    for version in INCUMBENT_VERSIONS.values():
        version_dir = artifacts / version
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "metadata.json").write_text(
            json.dumps({"version": version}), encoding="utf-8"
        )
    (artifacts / "latest.json").write_text(
        json.dumps(INCUMBENT_VERSIONS, indent=2), encoding="utf-8"
    )
    return SandboxRoots(artifacts=artifacts, staging=staging)


def stage_candidate_dirs(roots: SandboxRoots, targets: tuple[str, ...]) -> None:
    """Put a gate-scored candidate dir under staging for each of *targets*."""
    for target in targets:
        version_dir = roots.staging / CANDIDATE_VERSIONS[target]
        version_dir.mkdir(parents=True, exist_ok=True)
        (version_dir / "metadata.json").write_text(
            json.dumps({"version": CANDIDATE_VERSIONS[target]}), encoding="utf-8"
        )


def game_ids(n: int) -> list[str]:
    return [f"g{i:04d}" for i in range(n)]


def scored_frame(target: str, ids: list[str], *, value: float) -> pd.DataFrame:
    """A minimal scored frame in the backtest contract."""
    frame = pd.DataFrame(
        {
            "game_id": ids,
            "season": [SEASONS[i % len(SEASONS)] for i in range(len(ids))],
            "actual": [
                float(i % 2) if target == "wp" else 4.0 for i in range(len(ids))
            ],
            "model_prob": [value] * len(ids),
        }
    )
    column = _PREDICTION_COLUMN[target]
    if column != "model_prob":
        frame[column] = value
    return frame


def gold_frame(target: str, ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ids,
            "actual": [
                float(i % 2) if target == "wp" else 4.0 for i in range(len(ids))
            ],
        }
    )


def paired_delta_frame(ids: list[str], *, delta: float) -> pd.DataFrame:
    """A paired candidate-minus-incumbent CLV delta with a tiny deterministic wobble.

    A perfectly constant delta has zero variance and ``ttest_1samp`` returns NaN for it,
    so the wobble is what makes the significance test a real test rather than a
    degenerate one. It is deterministic (index-derived, not random) so a verdict is
    reproducible.
    """
    wobble = [((i % 7) - 3) * 1e-3 for i in range(len(ids))]
    return pd.DataFrame(
        {
            "game_id": ids,
            "season": [SEASONS[i % len(SEASONS)] for i in range(len(ids))],
            "clv_delta": [delta + w for w in wobble],
        }
    )


def candidate_bundle(target: str, **metrics: float) -> dict[str, Any]:
    """A candidate bundle carrying only what the secondary gate reads."""
    bundle: dict[str, Any] = {"clv_values": None, "per_season": {}}
    bundle.update(metrics)
    return bundle


def gate_cfg(**secondary_overrides: float) -> dict[str, Any]:
    """The committed gate config's [gate] section, in memory, tolerances overridable."""
    secondary = {
        "evaluation": "pooled",
        "wp_accuracy_max_drop": 0.01,
        "regression_mae_max_increase": 0.0,
        "wp_ece_max_increase": 0.0125,
        "wp_brier_max_increase": 0.0030,
    }
    secondary.update(secondary_overrides)
    return {
        "gate": {
            "alpha": 0.05,
            "floor_mode": "non_regression",
            "per_season_must_pass": True,
            "calibration_in_gate": True,
            "secondary": secondary,
        }
    }
