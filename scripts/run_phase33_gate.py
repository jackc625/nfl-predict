"""The Phase-33 deploy-gate runner: judge in stage one, promote in stage two.

Phase 33, Plan 33-08 Task 3 (COLD-04, D33-12 / D33-13 / D33-33).

THIS MODULE DOES NOT RUN THE GATE. Plan 33-15 runs it, after Plan 33-14 rebuilds gold.
What ships here is the runner and -- the point of shipping it early -- the FIX-CYCLE
ALLOWANCE, declared in committed source BEFORE any verdict exists.

THE FIX-CYCLE ALLOWANCE IS ZERO, AND IT IS DECLARED NOW (D33-12)
------------------------------------------------------------------
Written in the voice ``backtest/ev_chain_constants.py`` established, and for the same
reason: EDITING THIS DECLARATION AFTER A VERDICT EXISTS DOES NOT FIX A BUG, IT DESTROYS
THE EVIDENCE. There is no honest repair path. If the allowance is wrong it is wrong for
the remainder of the phase, and the only legitimate response is to say so in the readout.

Each target gets exactly ONE candidate: the SAME recipe as its incumbent, re-fit on
corrected gold. An allowance of one would have nothing legitimate to spend, because
every candidate lever is already spoken for:

  * the TRAINING WINDOW belongs to Phase 37's recipe;
  * the FEATURE GROUPS were bindingly ruled in Phase 30 (snap KEEP, situational KEEP,
    injury DROP);
  * HYPERPARAMETER SEARCH is out of scope for this phase.

Phase 30 discovered that by accident -- its single pre-registered fix-cycle lever went
UNSPENT for both failing targets because it had no unspent move, not because anybody
overlooked it. Declaring zero up front says the same thing honestly, and it forecloses
the shape the rule exists to stop: unlimited retries against a live gate, which is
p-hacking with extra steps.

THE NAMED CONSEQUENCE, STATED IN ADVANCE. If ATS fails, production RETAINS
``ats_20260605_220128``, whose own training rows were 61.2% fabricated Elo. That is the
RIGHT outcome, because both models are compared on CORRECT gold: a gate that promoted a
worse model to escape an embarrassing incumbent would be measuring embarrassment.

TWO STRICTLY SEPARATED STAGES (D33-13)
----------------------------------------
Stage one scores all three candidates AND all three incumbent re-scores against ONE
frozen pre-run state, renders three verdicts, and writes NOTHING into ``artifacts/``.
Stage two applies only the PASS promotions and CANNOT run without stage one's recorded
verdict -- ``MissingStageOneVerdictError`` makes "no promotion without a recorded
verdict carrying its paired statistic and p-value" STRUCTURAL rather than procedural.

Why not interleave: ``scripts/promote_models._promote_artifact_dir`` is the function
whose Rule-1 swap bug was found mid-run in Phase 25. A judge that swaps as it goes has
no state from which to answer "what did we decide before we started changing things".

CANDIDATES ARE STAGED, ALWAYS. Every candidate is fitted and scored under
``artifacts_staging/`` -- the root ``scripts/promote_models.py`` already expects. The
weaker manifest-only bracket is not an available option, so stage one leaves the WHOLE
production ``artifacts/`` tree byte-unchanged rather than merely its manifest, and the
``digest_tree`` bracket in the tests asserts exactly that.

ONE VERDICT RECORD, NOT THREE. Recorded discretion: the three verdicts are rendered
against one frozen state, and splitting them into three files would let a later reader
pair verdicts that were never rendered together. It is emitted in two halves because
``outputs/`` is gitignored and a fresh checkout must still be able to say WHICH state a
verdict was measured against: the full JSON to :data:`VERDICT_RECORD_PATH`, and the
committed generator-output half to :data:`COMMITTED_VERDICT_PATH`.

THE VERDICT VOCABULARY IS CLOSED AND COMPLETE
-----------------------------------------------
``("PASS", "FAIL", "UNTESTABLE_REFUSAL")``. The third member is not decoration.
``backtest.diagnose.clv_significance`` DELIBERATELY returns ``t=None, p=None`` below
``MIN_CLV_SAMPLE`` (``backtest/diagnose.py:249-285``), and this phase's own success
criterion says a zero-eligible-row target is REFUSED. A schema demanding non-null
statistics everywhere would reject that legitimate refusal as MALFORMED -- and the only
way to make such a record valid would be to fabricate a number. So ``PASS`` and ``FAIL``
REQUIRE non-null statistics; ``UNTESTABLE_REFUSAL`` permits nulls and REQUIRES a reason
drawn from :class:`UntestableRefusalReason`. An ``UNTESTABLE_REFUSAL`` always retains
the incumbent and is never a promotion.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

from backtest.diagnose import clv_significance
from models import deploy_gate
from models.artifacts import update_manifest
from scripts import promote_models
from utils import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# (1) The frozen pre-registration
# ---------------------------------------------------------------------------

# DECLARED 2026-09-12, BEFORE ANY PHASE-33 VERDICT EXISTS. See the module docstring for
# why every candidate lever is already spoken for and an allowance of one would have
# nothing legitimate to spend. Editing this after a verdict exists destroys the
# evidence rather than fixing anything.
PHASE33_FIX_CYCLE_ALLOWANCE: int = 0

# The COMPLETE verdict vocabulary. A verdict string outside this tuple is rejected by
# name (see :func:`validate_verdict_payload`).
VERDICT_STATES: tuple[str, ...] = ("PASS", "FAIL", "UNTESTABLE_REFUSAL")

# The three gated targets, in canonical order. `blend` is NOT here: it shares the one
# production swap surface and is passed through byte-identical (T-33-42).
GATED_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

VERDICT_RECORD_PATH = Path("outputs/phase33_gate_verdict.json")
COMMITTED_VERDICT_PATH = Path("config/phase33_gate_verdict.toml")
PRODUCTION_ARTIFACTS_DIR = Path("artifacts")
STAGING_ARTIFACTS_DIR = Path("artifacts_staging")

# The declared free-disk floor the pre-flight requires before any candidate is scored.
# Five gigabytes: a three-target staging re-fit plus its scored frames, with room for
# the copy stage two performs into production. Declared as a constant so the check has
# a number a reader can argue with rather than a magic literal inside a branch.
MIN_FREE_DISK_BYTES: int = 5 * 1024**3

# The pre-flight's writability probe. The suffix is deliberately OUTSIDE
# tests/data_boundary.TRACKED_SUFFIXES, and the file is removed immediately, so the
# probe cannot register as a production-store content change.
_PROBE_NAME = ".phase33_preflight_probe"


class UntestableRefusalReason(StrEnum):
    """Why a target could not be tested. An ``UNTESTABLE_REFUSAL`` must name one."""

    ZERO_ELIGIBLE_ROWS = "zero_eligible_paired_rows"
    BELOW_MIN_CLV_SAMPLE = "paired_sample_below_min_clv_sample"
    NO_PAIRED_DELTA = "no_paired_delta_supplied"


class PreflightFailedError(RuntimeError):
    """The environment failed a pre-flight check BEFORE any candidate was scored."""


class MissingStageOneVerdictError(RuntimeError):
    """Stage two was asked to promote with no stage-one verdict record present."""


class FixCycleAllowanceExceededError(RuntimeError):
    """A second candidate was offered for a target whose verdict is already recorded."""


class MalformedVerdictRecordError(ValueError):
    """A verdict record violates the closed schema (state, statistics or reason)."""


# ---------------------------------------------------------------------------
# (2) The record
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TargetScoring:
    """One target's frozen pre-run scoring inputs, as the scorer hands them over.

    The scorer is INJECTED rather than hardcoded so stage one can be driven -- by the
    tests here and by Plan 33-15's wiring -- without this module owning the training
    pipeline. What it must guarantee is the part that matters: the candidate was fitted
    and scored under ``artifacts_staging/``, never under production ``artifacts/``.
    """

    target: str
    candidate_version: str
    incumbent_version: str
    candidate_bundle: dict[str, Any]
    incumbent_scored: Any
    candidate_scored: Any
    gold: Any = None
    # game_id + season + clv_delta, the paired per-game candidate-minus-incumbent CLV.
    # Carried as a FRAME rather than an array so the delta can be restricted to the
    # shared eligibility index instead of being trusted to already line up.
    paired_delta: Any = None


@dataclass(frozen=True)
class GateVerdictRecord:
    """The ONE record stage one writes and stage two reads."""

    judge_version: str
    judge_code_digest: str
    rendered_at: str
    fix_cycle_allowance: int
    verdict_states: tuple[str, ...]
    secondary_scalar_names: tuple[str, ...]
    targets: dict[str, dict[str, Any]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """The JSON-serializable view, with targets in canonical order."""
        return {
            "judge_version": self.judge_version,
            "judge_code_digest": self.judge_code_digest,
            "rendered_at": self.rendered_at,
            "fix_cycle_allowance": self.fix_cycle_allowance,
            "verdict_states": list(self.verdict_states),
            "secondary_scalar_names": list(self.secondary_scalar_names),
            "targets": {
                target: self.targets[target]
                for target in GATED_TARGETS
                if target in self.targets
            },
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> GateVerdictRecord:
        """Rebuild a record from its serialized form, validating the schema first."""
        validate_verdict_payload(payload)
        return cls(
            judge_version=str(payload["judge_version"]),
            judge_code_digest=str(payload["judge_code_digest"]),
            rendered_at=str(payload["rendered_at"]),
            fix_cycle_allowance=int(payload["fix_cycle_allowance"]),
            verdict_states=tuple(payload["verdict_states"]),
            secondary_scalar_names=tuple(payload["secondary_scalar_names"]),
            targets=dict(payload["targets"]),
        )

    def passing_targets(self) -> tuple[str, ...]:
        """The PASS targets, in canonical order. Nothing else is ever promoted."""
        return tuple(
            target
            for target in GATED_TARGETS
            if self.targets.get(target, {}).get("verdict") == "PASS"
        )


def validate_verdict_payload(payload: Mapping[str, Any]) -> None:
    """Enforce the closed verdict schema; raise :class:`MalformedVerdictRecordError`.

    The asymmetry is the whole design (T-33-41b). ``PASS`` and ``FAIL`` are claims about
    a measurement, so they REQUIRE a non-null paired statistic and p-value.
    ``UNTESTABLE_REFUSAL`` is a claim that no measurement was possible, so it PERMITS
    nulls and REQUIRES a stated reason. Demanding statistics everywhere would force a
    fabricated number into a record to make a legitimate refusal valid.
    """
    for key in (
        "judge_version",
        "judge_code_digest",
        "rendered_at",
        "fix_cycle_allowance",
        "verdict_states",
        "secondary_scalar_names",
        "targets",
    ):
        if key not in payload:
            msg = f"verdict record missing required key '{key}'"
            raise MalformedVerdictRecordError(msg)

    if tuple(payload["verdict_states"]) != VERDICT_STATES:
        msg = (
            f"verdict record declares states {tuple(payload['verdict_states'])}, but the "
            f"closed vocabulary is {VERDICT_STATES}"
        )
        raise MalformedVerdictRecordError(msg)

    for target, row in dict(payload["targets"]).items():
        verdict = row.get("verdict")
        if verdict not in VERDICT_STATES:
            msg = (
                f"target '{target}' carries verdict {verdict!r}, which is outside the "
                f"closed vocabulary {VERDICT_STATES}"
            )
            raise MalformedVerdictRecordError(msg)
        if verdict == "UNTESTABLE_REFUSAL":
            if not str(row.get("reason") or "").strip():
                msg = (
                    f"target '{target}' is UNTESTABLE_REFUSAL with no reason. A refusal "
                    "that does not say why it could not be tested is indistinguishable "
                    "from a result that was never computed."
                )
                raise MalformedVerdictRecordError(msg)
            continue
        if row.get("paired_statistic") is None or row.get("p_value") is None:
            msg = (
                f"target '{target}' carries verdict {verdict} with a null paired "
                "statistic or p-value. PASS and FAIL are claims about a measurement and "
                "require one; an untestable target must be UNTESTABLE_REFUSAL instead."
            )
            raise MalformedVerdictRecordError(msg)


# ---------------------------------------------------------------------------
# (3) The pre-flight
# ---------------------------------------------------------------------------


def _probe_writable(root: Path, check: str) -> None:
    """Create and remove a probe file under *root*; raise naming *check* on failure."""
    try:
        root.mkdir(parents=True, exist_ok=True)
        probe = root / _PROBE_NAME
        probe.write_text("probe", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        msg = (
            f"pre-flight check '{check}' FAILED for '{root}': {type(exc).__name__}: {exc}. "
            "The gate refuses to start rather than discover this after scoring, because "
            "there is no legitimate second run to discover it in."
        )
        raise PreflightFailedError(msg) from exc


def preflight_health_check(
    *,
    artifacts_dir: Path = PRODUCTION_ARTIFACTS_DIR,
    staging_dir: Path = STAGING_ARTIFACTS_DIR,
    targets: Sequence[str] = GATED_TARGETS,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
) -> dict[str, Any]:
    """Fail the ENVIRONMENT before any number exists. Runs BEFORE any candidate is scored.

    THIS CREATES NO RETRY STATE (D33-33). The zero fix-cycle rule is UNCHANGED and
    ABSOLUTE: no re-runnable failure category, no post-scoring retry, and no
    environmental-abort escape hatch. Once scoring starts, one run is the run, whatever
    kills it. This function exists so an unwritable staging root or a full disk is
    caught while nothing has been measured -- the answer to "what if the environment
    breaks mid-run" is to make the environment fail FIRST, not to allow a second look
    after a number exists. Nothing below may be read as a licence to retry a scored run.

    The four things it checks:
      1. the staging root is WRITABLE (probe write, then removed);
      2. the staging root is CLEAR (no stale candidate dir, no stale manifest -- a stale
         dir is a candidate the newest-by-name resolution could silently select);
      3. the production artifacts root is WRITABLE and every incumbent named in its
         ``latest.json`` RESOLVES on disk with its metadata;
      4. free disk exceeds :data:`MIN_FREE_DISK_BYTES`.

    Args:
        artifacts_dir: The production artifacts root (stage two's swap target).
        staging_dir: The staging root every candidate is fitted and scored under.
        targets: The gated targets whose incumbents must resolve.
        min_free_bytes: The declared free-disk floor.

    Returns:
        ``{check_name: detail}`` for every check that passed -- the positive arm's
        evidence, so a healthy run records WHAT was checked and not merely that
        something was.

    Raises:
        PreflightFailedError: Naming which check failed and what it found.
    """
    report: dict[str, Any] = {}

    _probe_writable(Path(staging_dir), "staging_writable")
    report["staging_writable"] = str(staging_dir)

    stale = sorted(
        p.name
        for p in Path(staging_dir).iterdir()
        if p.name == "latest.json" or any(p.name.startswith(f"{t}_") for t in targets)
    )
    if stale:
        msg = (
            f"pre-flight check 'staging_clear' FAILED: '{staging_dir}' still holds {stale}. "
            "A stale candidate dir can be selected by the newest-by-name resolution and "
            "scored as if it were this run's. Clear the staging root and start again -- "
            "nothing has been measured yet, so this is not a retry."
        )
        raise PreflightFailedError(msg)
    report["staging_clear"] = True

    _probe_writable(Path(artifacts_dir), "artifacts_writable")
    report["artifacts_writable"] = str(artifacts_dir)

    manifest_path = Path(artifacts_dir) / "latest.json"
    if not manifest_path.exists():
        msg = (
            f"pre-flight check 'incumbents_resolve' FAILED: no production manifest at "
            f"'{manifest_path}'. The paired comparison re-scores the DEPLOYED incumbent, "
            "so a missing manifest means there is nothing to compare against."
        )
        raise PreflightFailedError(msg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    resolved: dict[str, str] = {}
    for target in targets:
        version = manifest.get(target)
        metadata = Path(artifacts_dir) / str(version) / "metadata.json"
        if not version or not metadata.exists():
            msg = (
                f"pre-flight check 'incumbents_resolve' FAILED for target '{target}': "
                f"manifest points at {version!r} and '{metadata}' does not exist. The "
                "incumbent is BOTH the paired comparator and the rollback target; never "
                "delete a deployed artifact dir."
            )
            raise PreflightFailedError(msg)
        resolved[target] = str(version)
    report["incumbents_resolve"] = resolved

    free = shutil.disk_usage(Path(artifacts_dir)).free
    if free < min_free_bytes:
        msg = (
            f"pre-flight check 'free_disk' FAILED: {free} bytes free under "
            f"'{artifacts_dir}', below the declared floor of {min_free_bytes}. A run that "
            "fills the disk halfway through scoring cannot be repeated under the zero "
            "fix-cycle rule, so it must not be started."
        )
        raise PreflightFailedError(msg)
    report["free_disk"] = free

    return report


# ---------------------------------------------------------------------------
# (4) Rendering one target's verdict
# ---------------------------------------------------------------------------


def _delta_on_index(
    paired_delta: Any, game_ids: Sequence[str]
) -> tuple[np.ndarray | None, dict[int, np.ndarray]]:
    """Restrict the paired per-game CLV delta to the shared eligibility index.

    Reindexing rather than trusting the array to already line up is what makes the
    pooled and per-season deltas paired AND order-invariant: the same games in the same
    order on both sides, chosen once.
    """
    if paired_delta is None or not len(game_ids):
        return None, {}
    indexed = paired_delta.set_index(paired_delta["game_id"].astype(str)).reindex(
        list(game_ids)
    )
    pooled = indexed["clv_delta"].to_numpy(dtype=float)
    per_season: dict[int, np.ndarray] = {}
    if "season" in indexed.columns:
        for season in sorted({int(s) for s in indexed["season"].dropna()}):
            slice_ = indexed.loc[indexed["season"] == season, "clv_delta"]
            per_season[season] = slice_.to_numpy(dtype=float)
    return pooled, per_season


def render_target_verdict(
    target: str,
    scoring: TargetScoring,
    cfg: dict[str, Any],
) -> dict[str, Any]:
    """Render ONE target's verdict against the frozen pre-run state. Writes nothing.

    Every path through here produces a row carrying the judge version and the
    scorer-code digest, because a verdict that cannot say which judge rendered it
    cannot be compared against a later one.
    """
    index = deploy_gate.build_eligibility_index(
        target, scoring.gold, scoring.incumbent_scored, scoring.candidate_scored
    )
    comparator = deploy_gate.live_secondary_metrics(
        target, scoring.incumbent_scored, scoring.gold, index
    )
    pooled_delta, per_season_delta = _delta_on_index(
        scoring.paired_delta, index.game_ids
    )

    row: dict[str, Any] = {
        "target": target,
        "candidate_version": scoring.candidate_version,
        "incumbent_version": scoring.incumbent_version,
        "eligibility": index.as_record(),
        "comparator_metrics": {
            metric: comparator.get(metric)
            for metric in deploy_gate.SECONDARY_METRICS_FOR[target]
        },
        "candidate_metrics": {
            metric: scoring.candidate_bundle.get(metric)
            for metric in deploy_gate.SECONDARY_METRICS_FOR[target]
        },
        "comparator_provenance": deploy_gate.comparator_provenance(comparator),
        "judge_version": deploy_gate.JUDGE_VERSION,
        "judge_code_digest": deploy_gate.judge_code_digest(),
    }

    if index.n == 0:
        return {
            **row,
            "verdict": "UNTESTABLE_REFUSAL",
            "reason": UntestableRefusalReason.ZERO_ELIGIBLE_ROWS.value,
            "reasons": [],
            "paired_statistic": None,
            "p_value": None,
            "paired_mean": None,
            "n_paired": 0,
        }
    if pooled_delta is None:
        return {
            **row,
            "verdict": "UNTESTABLE_REFUSAL",
            "reason": UntestableRefusalReason.NO_PAIRED_DELTA.value,
            "reasons": [],
            "paired_statistic": None,
            "p_value": None,
            "paired_mean": None,
            "n_paired": 0,
        }

    sig = clv_significance(pooled_delta)
    if sig["t"] is None:
        return {
            **row,
            "verdict": "UNTESTABLE_REFUSAL",
            "reason": UntestableRefusalReason.BELOW_MIN_CLV_SAMPLE.value,
            "reasons": [],
            "paired_statistic": None,
            "p_value": None,
            "paired_mean": sig["mean"],
            "n_paired": sig["n"],
        }

    bundle = dict(scoring.candidate_bundle)
    bundle["clv_delta_values"] = pooled_delta
    bundle["per_season_clv_delta_values"] = per_season_delta
    result = deploy_gate.evaluate_target(target, bundle, comparator, cfg)

    return {
        **row,
        "verdict": "PASS" if result["passed"] else "FAIL",
        "reason": "" if result["passed"] else "; ".join(result["reasons"]),
        "reasons": list(result["reasons"]),
        "paired_statistic": sig["t"],
        "p_value": sig["p"],
        "paired_mean": sig["mean"],
        "n_paired": sig["n"],
    }


# ---------------------------------------------------------------------------
# (5) Stage one -- judge, write nothing into production
# ---------------------------------------------------------------------------


def _refuse_shared_roots(artifacts_dir: Path, staging_dir: Path) -> None:
    """Staging must be a DIFFERENT root from production, or 'staged' means nothing."""
    if Path(artifacts_dir).resolve() == Path(staging_dir).resolve():
        msg = (
            f"pre-flight check 'staging_is_separate' FAILED: staging and production both "
            f"resolve to '{Path(artifacts_dir).resolve()}'. Every candidate must be fitted "
            "and scored under a staging root so stage one leaves the whole production "
            "artifacts tree byte-unchanged; a shared root makes that claim untestable."
        )
        raise PreflightFailedError(msg)


def _refuse_second_candidate(record_path: Path, targets: Sequence[str]) -> None:
    """Enforce :data:`PHASE33_FIX_CYCLE_ALLOWANCE` against an existing verdict record."""
    if not Path(record_path).exists():
        return
    existing = json.loads(Path(record_path).read_text(encoding="utf-8"))
    already = sorted(set(existing.get("targets", {})) & set(targets))
    if already and PHASE33_FIX_CYCLE_ALLOWANCE <= 0:
        msg = (
            f"a verdict is already recorded at '{record_path}' for {already}, and the "
            f"pre-registered fix-cycle allowance is {PHASE33_FIX_CYCLE_ALLOWANCE}. Offering "
            "a second candidate for a judged target is unlimited retries against a live "
            "gate, which is p-hacking with extra steps. The allowance was declared before "
            "any verdict existed and is not editable now."
        )
        raise FixCycleAllowanceExceededError(msg)


def _render_committed_verdict_toml(record: GateVerdictRecord) -> str:
    """The committed generator-output half, in ``config/group_gate_verdict.toml``'s shape."""
    lines = [
        "# " + "=" * 75,
        "# config/phase33_gate_verdict.toml -- the Phase-33 per-target gate verdict",
        "# Phase 33 (live cold-start and forward temporal integrity) -- COLD-04.",
        "#",
        "# GENERATOR OUTPUT. Produced by `python -m scripts.run_phase33_gate` and written",
        "# in ONE operation (D24-07). Do NOT hand-edit any value below: re-running the",
        "# gate is not available either, because the pre-registered fix-cycle allowance",
        f"# is {PHASE33_FIX_CYCLE_ALLOWANCE}. A hand-edited verdict is indistinguishable from a",
        "# tampered one.",
        "#",
        "# The committed half exists because outputs/ is gitignored and a fresh checkout",
        "# must still be able to say WHICH state a verdict was measured against.",
        "#",
        "# ASCII only, no emoji (CLAUDE.md hard constraint).",
        "# " + "=" * 75,
        "",
        f'judge_version = "{record.judge_version}"',
        f'judge_code_digest = "{record.judge_code_digest}"',
        f'rendered_at = "{record.rendered_at}"',
        f"fix_cycle_allowance = {record.fix_cycle_allowance}",
        "verdict_states = ["
        + ", ".join(f'"{state}"' for state in record.verdict_states)
        + "]",
        "",
    ]
    for target in GATED_TARGETS:
        row = record.targets.get(target)
        if row is None:
            continue
        lines.extend(
            [
                f"[verdict.{target}]",
                f'verdict = "{row["verdict"]}"',
                f'reason = "{str(row.get("reason", "")).replace(chr(34), chr(39))}"',
                f'candidate_version = "{row["candidate_version"]}"',
                f'incumbent_version = "{row["incumbent_version"]}"',
                f"paired_statistic = {_toml_scalar(row['paired_statistic'])}",
                f"p_value = {_toml_scalar(row['p_value'])}",
                f"paired_mean = {_toml_scalar(row['paired_mean'])}",
                f"n_paired = {int(row['n_paired'])}",
                f"n_eligible = {int(row['eligibility']['n_eligible'])}",
                "excluded_incumbent_only = "
                f"{int(row['eligibility']['excluded_incumbent_only'])}",
                "excluded_candidate_only = "
                f"{int(row['eligibility']['excluded_candidate_only'])}",
                f"excluded_not_in_gold = {int(row['eligibility']['excluded_not_in_gold'])}",
                f'comparator_provenance = "{row["comparator_provenance"]}"',
                "",
            ]
        )
    return "\n".join(lines)


def _toml_scalar(value: Any) -> str:
    """A TOML literal for a float that may legitimately be null.

    TOML has no null, so an untestable statistic is emitted as ``nan`` rather than as a
    zero. A zero would read as a measurement.
    """
    return "nan" if value is None else repr(float(value))


def _write_verdict_record(
    record: GateVerdictRecord,
    verdict_record_path: Path,
    committed_verdict_path: Path,
) -> None:
    """Emit both halves of the record. Kept OUT of ``stage_one_judge`` deliberately.

    Stage one must be provably free of writes into the production swap surface, and the
    cheapest way to make that checkable is for its own body to contain no write call at
    all -- an AST scan over ``stage_one_judge`` sees no manifest or file-write attribute
    because the writing lives here.
    """
    for path, payload in (
        (
            Path(verdict_record_path),
            json.dumps(record.as_dict(), indent=2, default=str),
        ),
        (Path(committed_verdict_path), _render_committed_verdict_toml(record)),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload + "\n", encoding="utf-8")
        logger.info("Wrote Phase-33 gate verdict half", path=str(path))


def stage_one_judge(
    *,
    scorer: Callable[[str, Path, Path], TargetScoring],
    cfg: dict[str, Any],
    targets: Sequence[str] = GATED_TARGETS,
    artifacts_dir: Path = PRODUCTION_ARTIFACTS_DIR,
    staging_dir: Path = STAGING_ARTIFACTS_DIR,
    verdict_record_path: Path = VERDICT_RECORD_PATH,
    committed_verdict_path: Path = COMMITTED_VERDICT_PATH,
    min_free_bytes: int = MIN_FREE_DISK_BYTES,
    now: str | None = None,
) -> GateVerdictRecord:
    """Judge all three targets against ONE frozen pre-run state. Promotes NOTHING.

    The production ``artifacts/`` tree is byte-unchanged when this returns -- not merely
    its manifest, the whole tree -- because every candidate is fitted and scored under
    *staging_dir*. The pre-flight runs FIRST and the scorer is not called at all if it
    fails, so an environmental fault costs nothing measured.

    The verdicts are INVARIANT to the order *targets* is given in: each is rendered
    against the same frozen state, and the record serializes them in canonical order.

    Args:
        scorer: ``(target, staging_dir, artifacts_dir) -> TargetScoring``. Injected so
            this module does not own the training pipeline.
        cfg: The loaded gate config.
        targets: Which targets to judge. Order does not affect the outcome.
        artifacts_dir: Production artifacts root. READ ONLY in this stage.
        staging_dir: The staging root every candidate lives under.
        verdict_record_path: Where the full JSON record goes (gitignored ``outputs/``).
        committed_verdict_path: Where the committed generator-output half goes.
        min_free_bytes: The pre-flight's declared free-disk floor.
        now: Override the render timestamp (tests pin it to compare whole records).

    Returns:
        The :class:`GateVerdictRecord` that was written.

    Raises:
        PreflightFailedError: The environment failed a check before anything was scored.
        FixCycleAllowanceExceededError: A verdict already exists for one of *targets*.
    """
    _refuse_shared_roots(artifacts_dir, staging_dir)
    preflight = preflight_health_check(
        artifacts_dir=artifacts_dir,
        staging_dir=staging_dir,
        targets=targets,
        min_free_bytes=min_free_bytes,
    )
    logger.info("Phase-33 gate pre-flight PASSED", checks=sorted(preflight))
    _refuse_second_candidate(verdict_record_path, targets)

    rows: dict[str, dict[str, Any]] = {}
    for target in targets:
        rows[target] = render_target_verdict(
            target, scorer(target, staging_dir, artifacts_dir), cfg
        )
        logger.info(
            "Phase-33 gate verdict rendered",
            target=target,
            verdict=rows[target]["verdict"],
        )

    record = GateVerdictRecord(
        judge_version=deploy_gate.JUDGE_VERSION,
        judge_code_digest=deploy_gate.judge_code_digest(),
        rendered_at=now or datetime.now(UTC).isoformat(),
        fix_cycle_allowance=PHASE33_FIX_CYCLE_ALLOWANCE,
        verdict_states=VERDICT_STATES,
        secondary_scalar_names=deploy_gate.SECONDARY_SCALAR_NAMES,
        targets=rows,
    )
    validate_verdict_payload(record.as_dict())
    _write_verdict_record(record, verdict_record_path, committed_verdict_path)
    return record


# ---------------------------------------------------------------------------
# (6) Stage two -- apply only the PASS promotions
# ---------------------------------------------------------------------------


def read_verdict_record(
    verdict_record_path: Path = VERDICT_RECORD_PATH,
) -> GateVerdictRecord:
    """Load stage one's record, or refuse by name if it is not there."""
    path = Path(verdict_record_path)
    if not path.exists():
        msg = (
            f"no stage-one verdict record at '{path}'. Stage two cannot promote without "
            "one: 'no promotion without a recorded verdict carrying its paired statistic "
            "and p-value' is structural here, not a convention somebody remembers."
        )
        raise MissingStageOneVerdictError(msg)
    return GateVerdictRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))


def stage_two_promote(
    *,
    verdict_record_path: Path = VERDICT_RECORD_PATH,
    artifacts_dir: Path = PRODUCTION_ARTIFACTS_DIR,
    staging_dir: Path = STAGING_ARTIFACTS_DIR,
) -> tuple[str, ...]:
    """Apply ONLY the PASS promotions from stage one's record. Recomputes no verdict.

    COPY BEFORE SWAP, and the order is load-bearing. Every PASS target's staged artifact
    directory is copied into production with
    ``scripts.promote_models._promote_artifact_dir`` FIRST -- all of them, before any
    pointer moves. That function's own docstring
    (``scripts/promote_models.py:575-580``) states the manifest writer rewrites only the
    per-target POINTER, so aiming production ``latest.json`` at a staging-only version
    leaves ``load_model_artifact(target)`` raising ``FileNotFoundError``. Copying first
    also means a crash between the two halves leaves the OLD manifest fully usable:
    every version it names still resolves.

    THE PER-TARGET LOOP IS KEPT DELIBERATELY. Rewriting the whole four-entry manifest in
    one shot was proposed and REJECTED: ``models.artifacts.update_manifest`` is the ONLY
    sanctioned writer of production ``artifacts/latest.json`` (D24-08), performs a
    per-key update so the ``blend`` pointer and every non-promoted target survive
    untouched, and already writes atomically. Its docstring names the exact failure the
    shortcut would cause: "never rewrite the whole manifest from a subset of passing
    targets."

    Args:
        verdict_record_path: Stage one's record. Its absence is a refusal, not a no-op.
        artifacts_dir: The production artifacts root.
        staging_dir: The staging root the gate-scored candidates live under.

    Returns:
        The promoted targets, in canonical order.

    Raises:
        MissingStageOneVerdictError: No stage-one record is present.
        MalformedVerdictRecordError: The record violates the closed schema.
    """
    record = read_verdict_record(verdict_record_path)
    promoted = [
        (target, record.targets[target]["candidate_version"])
        for target in record.passing_targets()
    ]

    for target, version in promoted:
        promote_models._promote_artifact_dir(
            target, version, Path(staging_dir), Path(artifacts_dir)
        )
        logger.info("Copied gate-scored artifact into production", target=target)

    for target, version in promoted:
        update_manifest(target, version, Path(artifacts_dir))
        logger.info("Moved production pointer", target=target, version=version)

    return tuple(target for target, _ in promoted)


# ---------------------------------------------------------------------------
# (7) CLI -- stage one by DEFAULT, matching promote_models' dry-run-by-default posture
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the runner's arguments. Stage two is behind an explicit flag."""
    parser = argparse.ArgumentParser(
        description=(
            "Phase-33 deploy gate. STAGE ONE BY DEFAULT: judges every target against one "
            "frozen pre-run state and writes NOTHING into production artifacts/. "
            "--promote runs stage two, which applies only the recorded PASS promotions "
            "and refuses without a stage-one verdict record."
        )
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help="Run STAGE TWO: apply the recorded PASS promotions. Requires a stage-one record.",
    )
    parser.add_argument("--artifacts-dir", type=Path, default=PRODUCTION_ARTIFACTS_DIR)
    parser.add_argument("--staging-dir", type=Path, default=STAGING_ARTIFACTS_DIR)
    parser.add_argument("--verdict-record", type=Path, default=VERDICT_RECORD_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Stage one has no wired scorer here -- Plan 33-15 supplies it."""
    args = parse_args(argv)
    if not args.promote:
        print(
            "Stage one needs a scorer. Plan 33-15 wires the staging re-fit and calls "
            "stage_one_judge(scorer=..., cfg=...) directly; this CLI exposes stage two "
            "and the pre-flight so the environment can be checked on its own."
        )
        report = preflight_health_check(
            artifacts_dir=args.artifacts_dir, staging_dir=args.staging_dir
        )
        print(f"PRE-FLIGHT PASSED: {sorted(report)}")
        return 0
    promoted = stage_two_promote(
        verdict_record_path=args.verdict_record,
        artifacts_dir=args.artifacts_dir,
        staging_dir=args.staging_dir,
    )
    print(f"PROMOTED: {list(promoted)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
