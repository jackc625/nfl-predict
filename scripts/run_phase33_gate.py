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
import hashlib
import inspect
import json
import shutil
import subprocess
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


# ---------------------------------------------------------------------------
# (1b) THE IN-SAMPLE LABEL, WRITTEN BEFORE ANY VERDICT VALUE EXISTS
# ---------------------------------------------------------------------------

#: The label every verdict row and the record header carry, in words a reader cannot skip.
#:
#: IT IS NOT A CAVEAT ADDED AFTERWARDS. It is a module CONSTANT, so it is in committed
#: source before any candidate is fitted, and `render_target_verdict` writes it onto every
#: row it builds -- including the refusal rows, which return early. The owner was told and
#: ACCEPTED this twice, most recently on 2026-09-14 once it had become mechanical rather
#: than hypothetical, and `conf/season_partition.py`'s own closing section records it.
#:
#: WHY IT IS TRUE, mechanically: under D33.1-01 the shipped artifact is fitted on EVERY
#: completed season through `models/trainers/final_fit.py`, and
#: `models.deploy_gate.HOLDOUT_SEASONS` is the same `partition.holdout` those seasons
#: include. So the candidate is trained over exactly the window the gate then re-scores it
#: on. `scripts.promote_models._incumbent_window`'s per-target window report is NOT
#: switched off and no artifact metadata is edited to make a refusal pass.
IN_SAMPLE_VERDICT_LABEL: str = (
    "IN-SAMPLE. The shipped artifact was fitted on the holdout seasons through the "
    "final-fit entry point (D33.1-01 / D33.1-02), and this gate re-scores that same "
    "artifact on holdout gold. The verdict is therefore NOT an out-of-sample "
    "generalisation estimate and MUST NEVER be presented as a clean gate pass. A "
    "confident-looking number here says the model reproduces rows it was fitted on. "
    "This label was written into committed source BEFORE any candidate was fitted, and "
    "into every verdict row before any verdict value was computed."
)

#: WHY THIS PHASE BUILDS ITS OWN argv INSTEAD OF REUSING
#: `scripts.promote_models._build_train_argv`. Recorded as a DELIBERATE, NAMED difference
#: rather than left for a later reader to discover two argv builders that disagree.
NO_TUNE_DIVERGENCE_REASON: str = (
    "scripts.promote_models._build_train_argv deliberately OMITS --no-tune, under Phase "
    "30's SPEC R5 which required a TUNED Stage-2 candidate. Phase 33's SPEC puts "
    "hyperparameter search of ANY kind out of scope, so this phase builds its own argv "
    "with --no-tune PRESENT. Everything else follows _build_train_argv's shape, "
    "including --exclude-groups and --exclude-groups-provenance so the Phase-30 group "
    "verdict travels with the artifact. The divergence is recorded, not discovered."
)


class UntestableRefusalReason(StrEnum):
    """Why a target could not be tested. An ``UNTESTABLE_REFUSAL`` must name one."""

    ZERO_ELIGIBLE_ROWS = "zero_eligible_paired_rows"
    BELOW_MIN_CLV_SAMPLE = "paired_sample_below_min_clv_sample"
    NO_PAIRED_DELTA = "no_paired_delta_supplied"


class PreflightFailedError(RuntimeError):
    """The environment failed a pre-flight check BEFORE any candidate was scored."""


class MissingStageOneVerdictError(RuntimeError):
    """Stage two was asked to promote with no stage-one verdict record present."""


class StaleStageOneVerdictError(RuntimeError):
    """Production no longer serves the incumbent a stage-one verdict was rendered against.

    Raised BEFORE anything is copied or swapped. A verdict compares ONE candidate with ONE
    incumbent; once production serves some third model, promoting the old candidate would
    replace whatever is live now with a model the verdict never compared against it.
    """


class FixCycleAllowanceExceededError(RuntimeError):
    """A second candidate was offered for a target whose verdict is already recorded."""


class MalformedVerdictRecordError(ValueError):
    """A verdict record violates the closed schema (state, statistics or reason)."""


class NonPassPromotionWithoutOverrideError(RuntimeError):
    """A promotion was requested for a target whose verdict is not PASS, with no override.

    Raised BEFORE anything is copied or swapped, so a refusal leaves the production swap
    surface untouched. The override it demands carries the owner's ruling AND the date;
    an undated ruling is an assertion nobody can place in time.
    """


class VirtualManifestUnresolvableError(RuntimeError):
    """A virtual-manifest entry names a version neither root can serve (Plan 33-15).

    Named separately rather than reusing a stage-two error: this is a BLEND MEASUREMENT
    failure at stage-one time, and mislabelling it as a missing verdict would send the
    reader looking for the wrong thing.
    """


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
    # EXTRA verdict-row fields the scorer measured and this module does not compute:
    # the three windows and their source, the window report, the resolved
    # hyperparameters, the selected-feature and weather-column counts before and after.
    # Added by Plan 33-15 Task 2 and DEFAULTED, so every Plan 33-08 caller that omits it
    # is byte-for-byte unaffected. `render_target_verdict` merges it into the row it
    # builds -- a scorer that measured something is the only thing that can report it,
    # and re-deriving it here would be a second derivation that could drift.
    extra: dict[str, Any] = field(default_factory=dict)


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
    # The blend's VIRTUAL-MANIFEST re-score (D33-14), added by Plan 33-15 Task 2 and
    # DEFAULTED so every Plan 33-08 caller is unaffected. It rides on the SAME record as
    # the three verdicts because it was measured against a manifest BUILT FROM THOSE
    # VERDICTS: splitting them would let a later reader pair a blend measurement with a
    # verdict set that never produced it.
    blend: dict[str, Any] = field(default_factory=dict)
    # The IN-SAMPLE label, at the record's head as well as on every row.
    in_sample_label: str = IN_SAMPLE_VERDICT_LABEL

    def as_dict(self) -> dict[str, Any]:
        """The JSON-serializable view, with targets in canonical order."""
        return {
            "judge_version": self.judge_version,
            "judge_code_digest": self.judge_code_digest,
            "rendered_at": self.rendered_at,
            "fix_cycle_allowance": self.fix_cycle_allowance,
            "verdict_states": list(self.verdict_states),
            "secondary_scalar_names": list(self.secondary_scalar_names),
            "in_sample_label": self.in_sample_label,
            "blend": dict(self.blend),
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
            blend=dict(payload.get("blend") or {}),
            in_sample_label=str(
                payload.get("in_sample_label") or IN_SAMPLE_VERDICT_LABEL
            ),
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


def _secondary_scalar_table(
    target: str,
    bundle: Mapping[str, Any],
    comparator: Mapping[str, Any],
    side: str = "candidate",
) -> dict[str, Any]:
    """All FIVE `SECONDARY_SCALAR_NAMES`, with the ones this target does not own None.

    THE COUNT IS READ FROM THE CONSTANT, NEVER FROM A LITERAL. An earlier draft of Plan
    33-08 said four secondary scalars, and a completeness check written against four
    would have PASSED with one scalar unchecked -- which is why every count assertion in
    this phase reads `deploy_gate.SECONDARY_SCALAR_NAMES`.

    A scalar this target does not own is None rather than 0.0. A zero MAE would read as a
    perfect model, which is the same "absent is not zero" rule the eligibility index and
    `live_secondary_metrics` already follow.

    Args:
        target: One of "wp", "ats", "ou".
        bundle: The side to read this target's own metrics from.
        comparator: Unused for the candidate side; kept in the signature so both sides
            call one function and no second flattening exists.
        side: "candidate" or "comparator" -- recorded only, for the caller's clarity.

    Returns:
        ``{flat_name: value_or_None}`` with exactly ``len(SECONDARY_SCALAR_NAMES)`` keys.
    """
    del comparator, side  # See the docstring: one flattening, two callers.
    owned = set(deploy_gate.SECONDARY_METRICS_FOR[target])
    table: dict[str, Any] = {}
    for flat_name in deploy_gate.SECONDARY_SCALAR_NAMES:
        scalar_target, _, metric = flat_name.partition(".")
        value = (
            bundle.get(metric) if scalar_target == target and metric in owned else None
        )
        table[flat_name] = None if value is None else float(value)
    return table


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
    # The candidate's secondary scalars are re-scored by the SAME function over the SAME
    # index. `build_candidate_bundle` computes them over the games with a closing line
    # only (557 in the Phase-33 run) while the comparator covers the whole index (570),
    # so passing the bundle's scalars through compared the two models on different games.
    candidate_secondary = deploy_gate.live_secondary_metrics(
        target, scoring.candidate_scored, scoring.gold, index
    )
    candidate_bundle = {
        **scoring.candidate_bundle,
        **{
            metric: candidate_secondary[metric]
            for metric in deploy_gate.SECONDARY_METRICS_FOR[target]
        },
    }
    pooled_delta, per_season_delta = _delta_on_index(
        scoring.paired_delta, index.game_ids
    )

    row: dict[str, Any] = {
        "target": target,
        "candidate_version": scoring.candidate_version,
        "incumbent_version": scoring.incumbent_version,
        # Plan 33-15 ALIASES of the two fields above, carried DELIBERATELY rather than
        # renamed. `stage_two_promote` reads `candidate_version`; this plan's record,
        # its committed TOML half and `tests.phase33_state.GATE_VERDICTS` speak of
        # ARTIFACTS. Duplicating one string is cheaper than making either consumer
        # translate, and a rename would have moved a field stage two depends on.
        "candidate_artifact": scoring.candidate_version,
        "incumbent_artifact": scoring.incumbent_version,
        "eligibility": index.as_record(),
        "eligible_pairs": index.n,
        "comparator_metrics": {
            metric: comparator.get(metric)
            for metric in deploy_gate.SECONDARY_METRICS_FOR[target]
        },
        "candidate_metrics": {
            metric: candidate_bundle.get(metric)
            for metric in deploy_gate.SECONDARY_METRICS_FOR[target]
        },
        # ALL FIVE secondary scalars on EVERY row, keyed by
        # `deploy_gate.SECONDARY_SCALAR_NAMES` verbatim. The count is read from that
        # constant and never from a literal: an earlier draft of Plan 33-08 said FOUR,
        # and a completeness check written against four would have PASSED with one
        # scalar unchecked. A scalar this target does not own is None, never 0.0.
        "secondary_scalars": _secondary_scalar_table(
            target, candidate_bundle, comparator
        ),
        "comparator_secondary_scalars": _secondary_scalar_table(
            target, comparator, comparator, side="comparator"
        ),
        "comparator_provenance": deploy_gate.comparator_provenance(comparator),
        "judge_version": deploy_gate.JUDGE_VERSION,
        "judge_code_digest": deploy_gate.judge_code_digest(),
        # WRITTEN BEFORE ANY VERDICT VALUE IS COMPUTED. Every return below spreads this
        # dict, including the three refusal returns, so no row can reach the record
        # without it.
        "in_sample_label": IN_SAMPLE_VERDICT_LABEL,
        # What the scorer MEASURED and this module cannot: the three windows and their
        # source, the window report, the resolved hyperparameters and the
        # selected-feature / weather-column counts before and after.
        **dict(scoring.extra),
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

    bundle = dict(candidate_bundle)
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
        f"in_sample_label = {_toml_string(record.in_sample_label)}",
        "",
    ]
    for target in GATED_TARGETS:
        row = record.targets.get(target)
        if row is None:
            continue
        windows = dict(row.get("windows") or {})
        lines.extend(
            [
                f"[verdicts.{target}]",
                f'verdict = "{row["verdict"]}"',
                f"reason = {_toml_string(str(row.get('reason', '')))}",
                f"reasons = {_toml_string_array(row.get('reasons') or [])}",
                f'candidate_artifact = "{row["candidate_artifact"]}"',
                f'incumbent_artifact = "{row["incumbent_artifact"]}"',
                # Plan 33-08's own names kept beside the Plan 33-15 aliases, because
                # stage two reads them and a committed record should not force a reader
                # to know which plan named which field.
                f'candidate_version = "{row["candidate_version"]}"',
                f'incumbent_version = "{row["incumbent_version"]}"',
                f"paired_statistic = {_toml_scalar(row['paired_statistic'])}",
                f"p_value = {_toml_scalar(row['p_value'])}",
                f"paired_mean = {_toml_scalar(row['paired_mean'])}",
                f"n_paired = {int(row['n_paired'])}",
                f"eligible_pairs = {int(row['eligibility']['n_eligible'])}",
                f"n_eligible = {int(row['eligibility']['n_eligible'])}",
                "excluded_incumbent_only = "
                f"{int(row['eligibility']['excluded_incumbent_only'])}",
                "excluded_candidate_only = "
                f"{int(row['eligibility']['excluded_candidate_only'])}",
                f"excluded_not_in_gold = {int(row['eligibility']['excluded_not_in_gold'])}",
                f'comparator_provenance = "{row["comparator_provenance"]}"',
                f'judge_version = "{row["judge_version"]}"',
                f'judge_code_digest = "{row["judge_code_digest"]}"',
                f"scorer_code_digest = {_toml_string(str(row.get('scorer_code_digest', '')))}",
                f"hyperparameter_search = {_toml_string(str(row.get('hyperparameter_search', 'none')))}",
                f"selected_feature_count = {_toml_int(row.get('selected_feature_count'))}",
                "selected_feature_count_before = "
                f"{_toml_int(row.get('selected_feature_count_before'))}",
                "selected_weather_count_after = "
                f"{_toml_int(row.get('selected_weather_count_after'))}",
                "selected_weather_count_before = "
                f"{_toml_int(row.get('selected_weather_count_before'))}",
                f"windows_source = {_toml_string(str(row.get('windows_source', '')))}",
                f"window_report = {_toml_string(str(row.get('window_report', '')))}",
                f"in_sample_label = {_toml_string(str(row.get('in_sample_label', '')))}",
                "",
                f"[verdicts.{target}.windows]",
                f"train = {_toml_int_array(windows.get('train') or [])}",
                f"hp_val = {_toml_int_array(windows.get('hp_val') or [])}",
                f"holdout = {_toml_int_array(windows.get('holdout') or [])}",
                "",
                f"[verdicts.{target}.incumbent_recorded_windows]",
                f"train = {_toml_int_array(_recorded_window(row, 'train'))}",
                f"hp_val = {_toml_int_array(_recorded_window(row, 'hp_val'))}",
                f"holdout = {_toml_int_array(_recorded_window(row, 'holdout'))}",
                "",
                f"[verdicts.{target}.secondary_scalars]",
                *_toml_table_lines(row.get("secondary_scalars") or {}),
                "",
                f"[verdicts.{target}.comparator_secondary_scalars]",
                *_toml_table_lines(row.get("comparator_secondary_scalars") or {}),
                "",
                f"[verdicts.{target}.resolved_hyperparameters]",
                *_toml_table_lines(row.get("resolved_hyperparameters") or {}),
                "",
            ]
        )
    lines.extend(_blend_toml_lines(record.blend))
    return "\n".join(lines)


def _recorded_window(row: Mapping[str, Any], key: str) -> list[int]:
    """The INCUMBENT's own recorded window for *key*, or an empty list."""
    recorded = dict(row.get("incumbent_recorded_windows") or {})
    return [int(season) for season in (recorded.get(key) or [])]


def _blend_toml_lines(blend: Mapping[str, Any]) -> list[str]:
    """The ``[blend]`` section: the virtual manifest FIRST, then before and after.

    THE VIRTUAL MANIFEST IS EMITTED BEFORE THE MEASUREMENTS, in the same order it was
    built: constructed from the verdicts, recorded, and only then scored against. A
    manifest written after the numbers could have been reconstructed to match them.
    """
    if not blend:
        return []
    lines = [
        "[blend]",
        f"artifact = {_toml_string(str(blend.get('artifact', '')))}",
        f"retuned = {'true' if blend.get('retuned') else 'false'}",
        f"note = {_toml_string(str(blend.get('note', '')))}",
        "",
        "[blend.virtual_manifest]",
        *[
            f'{key} = "{value}"'
            for key, value in sorted(dict(blend.get("virtual_manifest") or {}).items())
        ],
        "",
        "[blend.virtual_manifest_source]",
        *[
            f'{key} = "{value}"'
            for key, value in sorted(
                dict(blend.get("virtual_manifest_source") or {}).items()
            )
        ],
        "",
        "[blend.clv_before]",
        *_toml_table_lines(blend.get("clv_before") or {}),
        "",
        "[blend.clv_after]",
        *_toml_table_lines(blend.get("clv_after") or {}),
        "",
        "[blend.recorded_at_tuning]",
        *_toml_table_lines(blend.get("recorded_at_tuning") or {}),
        "",
        "[blend.weights]",
        *_toml_table_lines(blend.get("weights") or {}),
        "",
    ]
    return lines


def _toml_table_lines(table: Mapping[str, Any]) -> list[str]:
    """Render a flat mapping as TOML key/value lines, keys quoted so dots survive."""
    return [
        f"{_toml_string(str(key))} = {_toml_value(value)}"
        for key, value in sorted(table.items(), key=lambda item: str(item[0]))
    ]


def _toml_value(value: Any) -> str:
    """A TOML literal for a scalar of unknown type, with None emitted as ``nan``.

    TOML HAS NO NULL. A numeric absence becomes ``nan`` -- never 0.0, which would read as
    a measurement -- and a non-numeric absence becomes the empty string.
    """
    if value is None:
        return "nan"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return _toml_string(str(value))


def _toml_string(value: str) -> str:
    """A TOML basic string. ``json.dumps`` produces exactly TOML's escape set for ASCII."""
    return json.dumps(str(value))


def _toml_string_array(values: Sequence[Any]) -> str:
    """A TOML array of basic strings."""
    return "[" + ", ".join(_toml_string(str(value)) for value in values) + "]"


def _toml_int_array(values: Sequence[Any]) -> str:
    """A TOML array of integers."""
    return "[" + ", ".join(str(int(value)) for value in values) + "]"


def _toml_int(value: Any) -> str:
    """A TOML integer, or ``-1`` for an absent count.

    ``-1`` rather than ``nan``: TOML has no null and these are COUNTS, so a float NaN
    would change the field's type between runs. A negative count is impossible by
    construction, so it reads unambiguously as "not measured".
    """
    return "-1" if value is None else str(int(value))


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
    blend_scorer: Callable[[dict[str, dict[str, Any]]], dict[str, Any]] | None = None,
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
        blend_scorer: Optional ``{target: verdict_row} -> blend section`` (D33-14, Plan
            33-15 Task 2). Called AFTER the three rows are rendered and BEFORE the record
            is written, which is the only order that works: the blend is re-scored
            against a VIRTUAL MANIFEST built FROM the verdicts, and both halves must land
            in ONE write so a later reader cannot pair a blend measurement with a verdict
            set that never produced it. It writes nothing into production and is passed
            the rows rather than the record, because the record does not exist yet.

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

    blend_section: dict[str, Any] = {}
    if blend_scorer is not None:
        blend_section = dict(blend_scorer(rows))
        logger.info(
            "Phase-33 blend re-scored against the virtual manifest",
            virtual_manifest=blend_section.get("virtual_manifest"),
        )

    record = GateVerdictRecord(
        judge_version=deploy_gate.JUDGE_VERSION,
        judge_code_digest=deploy_gate.judge_code_digest(),
        rendered_at=now or datetime.now(UTC).isoformat(),
        fix_cycle_allowance=PHASE33_FIX_CYCLE_ALLOWANCE,
        verdict_states=VERDICT_STATES,
        secondary_scalar_names=deploy_gate.SECONDARY_SCALAR_NAMES,
        targets=rows,
        blend=blend_section,
        in_sample_label=IN_SAMPLE_VERDICT_LABEL,
    )
    validate_verdict_payload(record.as_dict())
    _write_verdict_record(record, verdict_record_path, committed_verdict_path)
    return record


# ---------------------------------------------------------------------------
# (5b) THE PLAN 33-15 SCORER -- the injected `scorer` stage one already expects
#
# `stage_one_judge` takes `(target, staging_dir, artifacts_dir) -> TargetScoring` so this
# module does not own the training pipeline. This is that function. What it guarantees is
# the part that matters: every candidate is fitted and scored under `staging_dir`, never
# under production `artifacts/`.
#
# WHAT IS CARRIED FORWARD IS THE SELECTION PROCEDURE, NEVER A FEATURE LIST. This is the
# instruction that would have wasted the phase if it were misread, so it is stated where
# the code is rather than only in a plan. NO incumbent `feature_list.json` is read as an
# INPUT to a fit, and no feature list is passed to anything:
# `BaseTrainer.train_and_evaluate` calls `select_features` as its FIRST step on every run
# and `models.train` exposes no feature-list flag, so a re-fit DOES re-select. WP and ATS
# selected ZERO weather features before, not because they rejected weather but because
# every weather column was frozen at a fabricated constant and a constant column has no
# importance and cannot be selected by any procedure -- THEY WERE NEVER OFFERED ANY.
# Pinning their old lists would ship two weatherless models out of the phase whose entire
# purpose was to give them weather to see.
#
# The incumbent's feature list IS read, for exactly one thing: the BEFORE half of the
# selected-feature / weather-column measurement, so the re-selection is a measured fact
# rather than an assertion. It is read as the record of a past run and reaches no fit.
# ---------------------------------------------------------------------------

#: The functions whose source the SCORER digest covers. A change to how a candidate is
#: built or paired is a change no verdict record could otherwise see. Mirrors
#: `deploy_gate.JUDGE_DIGEST_FUNCTIONS`, which covers the JUDGE rather than the scorer.
SCORER_DIGEST_FUNCTIONS: tuple[str, ...] = (
    "_build_phase33_train_argv",
    "phase33_scorer",
    "build_virtual_manifest",
    "score_blend_against_virtual_manifest",
)

#: Where the three windows come from, recorded in every verdict row so a reader never has
#: to infer it. Since review CR-01 `scripts.promote_models._incumbent_window` takes ALL
#: THREE from this rule rather than reading train and hp_val out of a VOID pre-correction
#: artifact's metadata.
WINDOWS_SOURCE: str = (
    "conf.season_partition.default_season_partition(), read through "
    "scripts.promote_models._incumbent_window (all three windows, since review CR-01)"
)


def scorer_code_digest(sources: list[str] | None = None) -> str:
    """A newline-normalized sha256 over the source of every scoring function.

    Newline normalization follows the idiom at
    ``tests/unit/test_preregistration_ancestry.py:32-39``: this repository has
    ``core.autocrlf=true`` and no ``.gitattributes``, so a digest over raw bytes would
    pin a value that holds only on the machine that measured it.
    """
    if sources is None:
        sources = [
            inspect.getsource(globals()[name]) for name in SCORER_DIGEST_FUNCTIONS
        ]
    normalized = "\n".join(s.replace("\r\n", "\n").replace("\r", "\n") for s in sources)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _build_phase33_train_argv(
    target: str,
    staging_dir: Path,
    window: Mapping[str, str],
    exclude_groups: Sequence[str],
    exclusion_provenance: str,
    gold_generation: str,
) -> list[str]:
    """Build the ``models.train`` argv for ONE Phase-33 candidate. Pure.

    TWO PROPERTIES ARE LOAD-BEARING, and the first is a DELIBERATE DIVERGENCE:

      * ``--no-tune`` is PRESENT. See :data:`NO_TUNE_DIVERGENCE_REASON`:
        ``promote_models._build_train_argv`` omits it on purpose under Phase 30's SPEC R5,
        and this phase puts hyperparameter search of any kind out of scope. That is why
        this function exists at all rather than reusing that one.
      * the three ``--config-*-seasons`` values come from *window*, which
        ``_incumbent_window`` fills from the committed partition rule. NO season list is
        typed here and none is read from an artifact's metadata.

    ``--gold-generation`` carries the measured generation key, so every candidate's own
    metadata declares the gold it was fitted on rather than leaving it to be inferred.
    """
    return [
        sys.executable,
        "-m",
        "models.train",
        "--target",
        target,
        "--artifacts-dir",
        str(staging_dir),
        "--no-tune",
        "--config-train-seasons",
        window["train"],
        "--config-hp-val-seasons",
        window["hp_val"],
        "--config-holdout-seasons",
        window["holdout"],
        "--exclude-groups",
        ",".join(exclude_groups),
        "--exclude-groups-provenance",
        exclusion_provenance,
        "--gold-generation",
        gold_generation,
    ]


def _weather_column_count(feature_list: Sequence[str]) -> int:
    """How many of *feature_list* are in the weather module's own declared family.

    DERIVED from ``features.weather.WEATHER_FEATURE_COLUMNS`` rather than from a list
    written here, so a column added to the family is counted without a second edit. The
    SAME definition is applied to the incumbent's list and to the candidate's, which is
    what makes the before/after pair commensurable.
    """
    from features.weather import WEATHER_FEATURE_COLUMNS

    return len(set(feature_list) & set(WEATHER_FEATURE_COLUMNS))


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON document, or an empty mapping when it is absent."""
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload) if isinstance(payload, Mapping) else {}


def _read_feature_list(path: Path) -> list[str]:
    """Read a ``feature_list.json``, or an empty list when it is absent."""
    if not path.is_file():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("features", [])
    return [str(name) for name in payload]


def phase33_scorer(
    target: str,
    staging_dir: Path,
    artifacts_dir: Path,
    *,
    cfg: dict[str, Any],
    gold_generation: str,
    exclude_groups: Sequence[str] = (),
    exclusion_provenance: str = "none",
    engine: Any = None,
    odds_df: Any = None,
) -> TargetScoring:
    """Fit ONE candidate under *staging_dir* and hand stage one its frozen scoring inputs.

    The signature ``stage_one_judge`` expects is the first three parameters; the rest are
    keyword-only and are bound by the caller (see :func:`run_stage_one`), so the injected
    scorer stays a plain ``(target, staging_dir, artifacts_dir) -> TargetScoring``.

    Args:
        target: One of "wp", "ats", "ou".
        staging_dir: The staging artifacts root. EVERY write this function makes lands
            here; production ``artifacts/`` is opened READ-ONLY.
        artifacts_dir: The production artifacts root -- the incumbent side of the pair.
        cfg: The loaded gate config.
        gold_generation: The measured gold generation key, passed to the candidate's own
            metadata through ``--gold-generation``.
        exclude_groups: The Phase-30 group exclusion, passed through so the ratified
            verdict travels with the artifact.
        exclusion_provenance: "verdict", "override" or "none".
        engine: A constructed ``BacktestEngine`` (loader-only). Built here when omitted.
        odds_df: Normalized closing odds. Loaded from *engine* when omitted.

    Returns:
        The :class:`TargetScoring`, with the measured recipe in ``extra``.
    """
    from backtest.diagnose import CLV_COLUMN_FOR, score_deployed_artifacts
    from backtest.engine import BacktestEngine

    if engine is None:
        engine = BacktestEngine()
    if odds_df is None:
        odds_df = engine._load_closing_odds()

    # (1) THE WINDOW, from the committed rule, with the difference against the incumbent's
    # own record reported rather than silently honoured.
    window = promote_models._incumbent_window(target, Path(artifacts_dir))
    incumbent_version = str(_read_json(Path(artifacts_dir) / "latest.json")[target])
    incumbent_meta = _read_json(
        Path(artifacts_dir) / incumbent_version / "metadata.json"
    )
    incumbent_config = dict(incumbent_meta.get("config") or {})
    incumbent_features = _read_feature_list(
        Path(artifacts_dir) / incumbent_version / "feature_list.json"
    )

    # (2) THE FIT, as a subprocess into staging. One target, one candidate.
    argv = _build_phase33_train_argv(
        target,
        Path(staging_dir),
        window,
        exclude_groups,
        exclusion_provenance,
        gold_generation,
    )
    logger.info("Fitting Phase-33 candidate", target=target, argv=argv[2:])
    subprocess.run(argv, check=True)

    candidate_version = promote_models._resolve_staged_version(
        target, Path(staging_dir), skip_train=False
    )
    if candidate_version is None:  # pragma: no cover - _resolve_staged_version raises
        msg = f"no staged candidate dir for '{target}' under '{staging_dir}'"
        raise RuntimeError(msg)

    # The STAGING manifest, so `score_deployed_artifacts` can resolve the candidate.
    # artifacts_dir=staging_dir: this never touches production (D24-08 / D24-09).
    update_manifest(target, candidate_version, artifacts_dir=Path(staging_dir))
    # The in-sample window report rides in the candidate's OWN metadata, written into
    # STAGING before any promotion copy, so a consumer of a promoted artifact inherits it
    # instead of relying on somebody having read the console (WR-12).
    promote_models._record_window_report(
        target, candidate_version, Path(staging_dir), window.get("window_report", "")
    )

    candidate_meta = _read_json(Path(staging_dir) / candidate_version / "metadata.json")
    candidate_features = _read_feature_list(
        Path(staging_dir) / candidate_version / "feature_list.json"
    )

    # (3) THE SCORE, both sides on the SAME gold frame.
    gold = promote_models._load_gold_holdout(target, engine)
    candidate_scored = score_deployed_artifacts(
        target, gold_df=gold, artifacts_dir=Path(staging_dir)
    )
    incumbent_scored = score_deployed_artifacts(
        target, gold_df=gold, artifacts_dir=Path(artifacts_dir)
    )
    candidate_bundle = deploy_gate.build_candidate_bundle(
        target, candidate_scored, odds_df, cfg
    )

    # (4) THE PAIRED PER-GAME DELTA, through the SAME two helpers the legacy promotion
    # path uses, so the pairing cannot drift into a third implementation.
    candidate_valid = promote_models._candidate_clv_frame(
        target, candidate_scored, odds_df
    )
    incumbent_valid = promote_models._score_baseline_clv(
        target, gold, odds_df, Path(artifacts_dir)
    )
    clv_column = CLV_COLUMN_FOR[target]
    paired = candidate_valid.merge(
        incumbent_valid, on="game_id", how="inner", suffixes=("_cand", "_base")
    )
    paired_delta = paired[["game_id"]].copy()
    paired_delta["season"] = paired["season_cand"]
    paired_delta["clv_delta"] = (
        paired[f"{clv_column}_cand"].to_numpy()
        - paired[f"{clv_column}_base"].to_numpy()
    )

    extra: dict[str, Any] = {
        "windows": {
            key: [int(season) for season in window[key].split(",") if season]
            for key in ("train", "hp_val", "holdout")
        },
        "windows_source": WINDOWS_SOURCE,
        "incumbent_recorded_windows": {
            "train": [int(s) for s in (incumbent_config.get("train_seasons") or [])],
            "hp_val": [int(s) for s in (incumbent_config.get("hp_val_seasons") or [])],
            "holdout": [
                int(s) for s in (incumbent_config.get("holdout_seasons") or [])
            ],
        },
        "window_report": window.get("window_report", ""),
        # "none" rather than False: this field answers "which search ran", and the answer
        # is that none did. Every candidate is trained with --no-tune and no Optuna study
        # is created.
        "hyperparameter_search": "none",
        "no_tune_divergence": NO_TUNE_DIVERGENCE_REASON,
        "resolved_hyperparameters": dict(candidate_meta.get("best_params") or {}),
        "selected_feature_count": len(candidate_features),
        "selected_feature_count_before": len(incumbent_features),
        "selected_weather_count_after": _weather_column_count(candidate_features),
        "selected_weather_count_before": _weather_column_count(incumbent_features),
        "gold_generation": gold_generation,
        "candidate_gold_generation_marker": candidate_meta.get(
            "trained_on_real_weather_generation"
        ),
        "final_fit_seasons": candidate_meta.get("final_fit_seasons"),
        "exclude_groups": list(exclude_groups),
        "exclude_groups_provenance": exclusion_provenance,
        "scorer_code_digest": scorer_code_digest(),
        # DELIBERATELY ABSENT: no `feature_list` field. A pinned list in the record is the
        # shape the selection-procedure rule forbids, and a verify command fails when one
        # is present.
    }

    return TargetScoring(
        target=target,
        candidate_version=candidate_version,
        incumbent_version=incumbent_version,
        candidate_bundle=candidate_bundle,
        incumbent_scored=incumbent_scored,
        candidate_scored=candidate_scored,
        gold=gold,
        paired_delta=paired_delta,
        extra=extra,
    )


# ---------------------------------------------------------------------------
# (5c) THE BLEND, RE-SCORED AGAINST A VIRTUAL MANIFEST (D33-14)
#
# `blend_dynamic_20260606_020635` is NOT re-tuned -- a weight sweep is a SEARCH and search
# is out of scope for this phase. It IS re-scored on the rebuilt gold against the
# POST-GATE END STATE, which at stage-one time does not yet exist in
# `artifacts/latest.json` because stage two has not run. Scoring against the unchanged
# production manifest would measure the OLD ensemble and label it the new one.
# ---------------------------------------------------------------------------


def build_virtual_manifest(
    rows: Mapping[str, Mapping[str, Any]],
    incumbents: Mapping[str, str],
) -> dict[str, str]:
    """The manifest the post-gate end state WOULD have, built FROM the verdicts.

    For each target: the candidate id where its verdict authorises a promotion, or the
    incumbent id where it does not. Plus ``blend``, carried through unchanged.

    IT IS BUILT FROM THE VERDICTS, NOT FROM A GUESS AT THE OWNER'S RULING. If the Task-3
    ruling changes which targets ship, the blend is re-scored against the RULED manifest
    and BOTH measurements are recorded with which manifest produced each -- never one
    silently replaced by the other.

    Args:
        rows: ``{target: verdict_row}`` as stage one rendered them.
        incumbents: ``{target: version}`` from the production manifest, including
            ``blend``.

    Returns:
        ``{target: version}`` with four entries.
    """
    virtual = {
        target: (
            str(rows[target]["candidate_version"])
            if str(rows.get(target, {}).get("verdict")) == "PASS"
            else str(incumbents[target])
        )
        for target in GATED_TARGETS
    }
    virtual["blend"] = str(incumbents["blend"])
    return virtual


def _resolve_version_root(
    target: str,
    version: str,
    artifacts_dir: Path,
    staging_dir: Path,
) -> Path:
    """Which root holds *version* for *target*, ASSERTED rather than assumed.

    ``score_deployed_artifacts`` resolves a target through the root's own ``latest.json``
    and takes no version argument, so "score this version" has to be expressed as "score
    from the root whose manifest names this version". That identity is CHECKED here
    instead of being trusted, because trusting it is how a blend measurement ends up
    describing an ensemble nobody selected.

    PRODUCTION IS CHECKED FIRST, AND THE ORDER IS NOT ARBITRARY. During stage one an
    INCUMBENT version is always present in production, and a CANDIDATE version is never
    present there -- which is exactly what the content-digest bracket over the whole
    production tree proves. So production-first resolves every incumbent unambiguously and
    every candidate falls through to staging. A staging-first order was measured to be
    wrong on this tree: a stale Phase-30 staging manifest still named
    ``wp_20260824_113325``, the deployed WP incumbent, so the incumbent would have been
    served out of staging. It was byte-identical there (a promotion is a copy) so nothing
    was misreported, but "it happened to be the same bytes" is not a property to rely on.
    """
    for root in (Path(artifacts_dir), Path(staging_dir)):
        manifest = _read_json(root / "latest.json")
        if str(manifest.get(target) or "") == version and (root / version).is_dir():
            return root
    msg = (
        f"the virtual manifest names '{version}' for target '{target}', and neither "
        f"'{artifacts_dir}' nor '{staging_dir}' has a latest.json naming it with the "
        "directory present. The blend cannot be scored against an ensemble that cannot "
        "be resolved, and mutating a manifest to make it resolvable is exactly what this "
        "function exists to avoid."
    )
    raise VirtualManifestUnresolvableError(msg)


def score_blend_against_virtual_manifest(
    virtual_manifest: Mapping[str, str],
    *,
    artifacts_dir: Path,
    staging_dir: Path,
    engine: Any = None,
    odds_df: Any = None,
) -> tuple[dict[str, float | None], dict[str, str]]:
    """Per-target mean CLV of the BLENDED predictions under *virtual_manifest*.

    Resolves each target's artifact through :func:`_resolve_version_root` and NEVER by
    mutating ``artifacts/latest.json``. The blend itself is loaded from the production
    root at the manifest's own ``blend`` id and is not re-tuned.

    Returns:
        ``({target: mean_clv_or_None}, {target: root_that_served_it})``.
    """
    from backtest.diagnose import CLV_COLUMN_FOR, score_deployed_artifacts
    from backtest.engine import BacktestEngine
    from models.blending import MarketBlender
    from models.blending_data import blend_historical_predictions
    from models.clv import compute_clv_for_predictions

    if engine is None:
        engine = BacktestEngine()
    if odds_df is None:
        odds_df = engine._load_closing_odds()

    blender = MarketBlender.from_artifacts(
        artifacts_dir=Path(artifacts_dir), version=str(virtual_manifest["blend"])
    )

    clv: dict[str, float | None] = {}
    sources: dict[str, str] = {}
    for target in GATED_TARGETS:
        version = str(virtual_manifest[target])
        root = _resolve_version_root(target, version, artifacts_dir, staging_dir)
        sources[target] = str(root)
        gold = promote_models._load_gold_holdout(target, engine)
        scored = score_deployed_artifacts(target, gold_df=gold, artifacts_dir=root)
        # A33.2-review WR-04: these are HISTORICAL games, so WP is blended against each
        # game's owned pre-lock spread at its own season's PRIOR-ONLY converter slope, not
        # a closing spread at the in-sample serving slope. A game with no line is left out.
        blended = blend_historical_predictions(
            blender, scored, odds_df, target, artifacts_dir=Path(artifacts_dir)
        )
        clv_df = compute_clv_for_predictions(blended, odds_df, target)
        valid = clv_df.loc[clv_df["has_closing_odds"]]
        column = CLV_COLUMN_FOR[target]
        clv[target] = float(valid[column].mean()) if len(valid) else None
    sources["blend"] = str(Path(artifacts_dir))
    return clv, sources


# ---------------------------------------------------------------------------
# (5d) THE DRIVER -- what the operator actually runs for stage one
# ---------------------------------------------------------------------------


def run_stage_one(
    *,
    gold_generation: str,
    artifacts_dir: Path = PRODUCTION_ARTIFACTS_DIR,
    staging_dir: Path = STAGING_ARTIFACTS_DIR,
    verdict_record_path: Path = VERDICT_RECORD_PATH,
    committed_verdict_path: Path = COMMITTED_VERDICT_PATH,
    targets: Sequence[str] = GATED_TARGETS,
) -> dict[str, Any]:
    """Wire the scorer and the blend re-score, and run stage one ONCE.

    THE PRE-FLIGHT RUNS FIRST AND EXPLICITLY, and its result is returned so the operator
    can record it. ``stage_one_judge`` runs it again as its own gate; running it twice is
    free, because it probes and removes and asserts nothing about history. D33-33 is
    unchanged by its presence: it creates NO retry state, the zero fix-cycle rule is
    ABSOLUTE, and this is the ONLY point at which a re-run is available -- once scoring
    starts, one run is the run, whatever kills it.

    Args:
        gold_generation: The measured gold generation key every candidate declares.
        artifacts_dir: The production artifacts root. READ ONLY throughout stage one.
        staging_dir: The staging root every candidate is fitted and scored under.
        verdict_record_path: Where the full JSON record goes.
        committed_verdict_path: Where the committed generator-output half goes.
        targets: Which targets to judge. Order does not affect the outcome.

    Returns:
        ``{"preflight": ..., "record": GateVerdictRecord, "exclude_groups": ...}``.
    """
    from backtest.engine import BacktestEngine

    preflight = preflight_health_check(
        artifacts_dir=Path(artifacts_dir),
        staging_dir=Path(staging_dir),
        targets=targets,
    )
    print(f"PRE-FLIGHT PASSED: {sorted(preflight)}")

    cfg = deploy_gate.load_gate_config()
    deploy_gate.validate_gate_config(cfg)

    # DERIVED from the ratified Stage-1 verdict, never transcribed (D24-07). `promote=True`
    # selects the ARMED precedence: an absent verdict file is a STOP rather than a silent
    # "train every group", because this run's candidates are the ones a promotion would
    # ship.
    exclude_groups, exclusion_provenance = promote_models._resolve_exclude_groups(
        argparse.Namespace(exclude_groups=None, promote=True)
    )

    engine = BacktestEngine()
    odds_df = engine._load_closing_odds()
    incumbents = _read_json(Path(artifacts_dir) / "latest.json")

    def _scorer(target: str, staging: Path, artifacts: Path) -> TargetScoring:
        return phase33_scorer(
            target,
            staging,
            artifacts,
            cfg=cfg,
            gold_generation=gold_generation,
            exclude_groups=exclude_groups,
            exclusion_provenance=exclusion_provenance,
            engine=engine,
            odds_df=odds_df,
        )

    def _blend_scorer(rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
        virtual = build_virtual_manifest(rows, incumbents)
        # RECORDED BEFORE IT IS SCORED AGAINST. Logged here and emitted FIRST in the
        # committed record's [blend] section, so the manifest the measurement was taken
        # against is on the record and cannot be reconstructed afterwards to match it.
        logger.info("Phase-33 blend virtual manifest", virtual_manifest=virtual)
        print(f"BLEND VIRTUAL MANIFEST (recorded before scoring): {virtual}")
        incumbent_manifest = {
            key: str(incumbents[key]) for key in (*GATED_TARGETS, "blend")
        }
        clv_before, _ = score_blend_against_virtual_manifest(
            incumbent_manifest,
            artifacts_dir=Path(artifacts_dir),
            staging_dir=Path(staging_dir),
            engine=engine,
            odds_df=odds_df,
        )
        clv_after, sources = score_blend_against_virtual_manifest(
            virtual,
            artifacts_dir=Path(artifacts_dir),
            staging_dir=Path(staging_dir),
            engine=engine,
            odds_df=odds_df,
        )
        weights_doc = _read_json(
            Path(artifacts_dir) / str(incumbents["blend"]) / "blend_weights.json"
        )
        return {
            "artifact": str(incumbents["blend"]),
            "retuned": False,
            "note": (
                "NOT re-tuned: a weight_range 0.5-0.7 / weight_step 0.01 sweep is a "
                "SEARCH and search is out of scope for this phase (D33-14). clv_before "
                "and clv_after are BOTH measured on today's rebuilt gold over the live "
                "holdout, so they are commensurable with each other: before is the blend "
                "over the INCUMBENT ensemble, after is the blend over the virtual "
                "manifest. recorded_at_tuning is the artifact's own pre-rebuild reference "
                "and is NOT commensurable with either -- it was measured on different "
                "gold over a different season span. A materially worse post-rebuild value "
                "is a FINDING handed to Phase 37, not a licence to re-tune here."
            ),
            "virtual_manifest": virtual,
            "virtual_manifest_source": sources,
            "incumbent_manifest": incumbent_manifest,
            "clv_before": clv_before,
            "clv_after": clv_after,
            "recorded_at_tuning": dict(weights_doc.get("per_target_clv") or {}),
            "weights": dict(weights_doc.get("weights") or {}),
            "recorded_at_tuning_n_games": dict(weights_doc.get("n_games") or {}),
            "recorded_at_tuning_tuned_at": str(weights_doc.get("tuned_at") or ""),
        }

    record = stage_one_judge(
        scorer=_scorer,
        cfg=cfg,
        targets=targets,
        artifacts_dir=Path(artifacts_dir),
        staging_dir=Path(staging_dir),
        verdict_record_path=Path(verdict_record_path),
        committed_verdict_path=Path(committed_verdict_path),
        blend_scorer=_blend_scorer,
    )
    return {
        "preflight": preflight,
        "record": record,
        "exclude_groups": list(exclude_groups),
        "exclude_groups_provenance": exclusion_provenance,
    }


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


def _authorised_promotions(
    record: GateVerdictRecord,
    authorised_targets: Sequence[str] | None,
    overrides: Mapping[str, Mapping[str, str]] | None,
) -> list[tuple[str, str]]:
    """Resolve which targets ship, refusing any non-PASS promotion with no override.

    THE DEFAULT IS UNCHANGED. ``authorised_targets is None`` means exactly
    ``record.passing_targets()``, which is what Plan 33-08 shipped and what every caller
    written against it still gets.

    WHY AN OVERRIDE EXISTS AT ALL. The owner's standing ruling of 2026-09-14 voids the
    pre-correction incumbents, so RETAINING one is not a safe default; the anti-p-hacking
    prohibition still forbids PROMOTING a target with no PASS verdict. Those two cannot
    both be satisfied by a default, so the disposition of a non-PASS target is an owner
    RULING, and a ruling that is not recorded beside the verdict it overrides is
    indistinguishable from a verdict somebody re-read.

    THE VERDICT IS NEVER TOUCHED. An override rides BESIDE the row and changes no field of
    it: a FAIL stays a FAIL in the record, in the committed TOML and in the state manifest,
    whatever was done about it. An override recorded beside a verdict is a disclosure; a
    verdict rewritten to look like a pass is the defect class this milestone exists to
    detect.

    Args:
        record: Stage one's record.
        authorised_targets: The targets the owner authorised, or None for PASS-only.
        overrides: ``{target: {"ruling": ..., "ruled_on": ...}}``. Required for every
            authorised target whose verdict is not PASS.

    Returns:
        ``[(target, candidate_version), ...]`` in canonical order.

    Raises:
        NonPassPromotionWithoutOverrideError: An authorised target is not PASS and carries
            no override with BOTH a ruling and a date, or names a target absent from the
            record.
    """
    if authorised_targets is None:
        return [
            (target, record.targets[target]["candidate_version"])
            for target in record.passing_targets()
        ]

    requested = [
        target for target in GATED_TARGETS if target in set(authorised_targets)
    ]
    unknown = sorted(set(authorised_targets) - set(record.targets))
    if unknown:
        msg = (
            f"authorised targets {unknown} carry no verdict in this record, whose targets "
            f"are {sorted(record.targets)}. A promotion for a target the gate never judged "
            "is a production change with no measurement behind it at all."
        )
        raise NonPassPromotionWithoutOverrideError(msg)

    supplied = dict(overrides or {})
    unbacked = []
    for target in requested:
        if record.targets[target].get("verdict") == "PASS":
            continue
        override = dict(supplied.get(target) or {})
        if not str(override.get("ruling") or "").strip():
            unbacked.append(f"{target} (no ruling)")
        elif not str(override.get("ruled_on") or "").strip():
            unbacked.append(f"{target} (ruling with no date)")
    if unbacked:
        msg = (
            f"refusing to promote {unbacked}: each carries a verdict that is not PASS and "
            "no complete owner override. A non-PASS promotion is possible ONLY through an "
            "explicit override recorded BESIDE that target's verdict, carrying the owner's "
            "words AND the date they ruled. Without both, the promotion is a production "
            "change with no recorded authority, which is the shape the anti-p-hacking "
            "prohibition exists to prevent. The verdict itself is never edited to make "
            "this pass."
        )
        raise NonPassPromotionWithoutOverrideError(msg)

    return [
        (target, record.targets[target]["candidate_version"]) for target in requested
    ]


def stage_two_promote(
    *,
    verdict_record_path: Path = VERDICT_RECORD_PATH,
    artifacts_dir: Path = PRODUCTION_ARTIFACTS_DIR,
    staging_dir: Path = STAGING_ARTIFACTS_DIR,
    authorised_targets: Sequence[str] | None = None,
    overrides: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[str, ...]:
    """Apply the AUTHORISED promotions from stage one's record. Recomputes no verdict.

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
        authorised_targets: The targets the OWNER authorised. ``None`` -- the default, and
            Plan 33-08's behaviour byte for byte -- promotes the PASS targets and nothing
            else.
        overrides: ``{target: {"ruling": ..., "ruled_on": ...}}``, required for every
            authorised target whose verdict is not PASS. See
            :func:`_authorised_promotions` for why this exists and what it deliberately
            does NOT do, which is touch the verdict.

    Returns:
        The promoted targets, in canonical order.

    Raises:
        MissingStageOneVerdictError: No stage-one record is present.
        MalformedVerdictRecordError: The record violates the closed schema.
        NonPassPromotionWithoutOverrideError: An authorised target is not PASS and carries
            no complete override. Raised BEFORE anything is copied or swapped, so a
            refusal leaves the production swap surface untouched rather than half-moved.
        StaleStageOneVerdictError: Production serves neither the verdict's incumbent nor
            its candidate for a target about to be promoted. Also raised before anything
            is copied or swapped.
    """
    record = read_verdict_record(verdict_record_path)
    promoted = _authorised_promotions(record, authorised_targets, overrides)

    # A verdict is a comparison against the incumbent production served WHEN IT WAS
    # RENDERED. If production has since moved to some third model, replaying the verdict
    # would roll that model back to the old candidate (the 2026-09-14 record would put the
    # voided wp_20260914_221745 back in place of the live WP). A manifest that already
    # names the candidate is this same promotion having run, so a repeat is a no-op.
    serving = _read_json(Path(artifacts_dir) / "latest.json")
    stale = {
        target: {
            "production_serves": serving.get(target),
            "verdict_incumbent": record.targets[target].get("incumbent_version"),
            "verdict_candidate": version,
        }
        for target, version in promoted
        if serving.get(target)
        not in (record.targets[target].get("incumbent_version"), version)
    }
    if stale:
        msg = (
            f"refusing to promote from the stage-one record at '{verdict_record_path}': "
            f"production has moved since that verdict was rendered: {stale}. The verdict "
            "compared its candidate with the incumbent it names, not with the model "
            "production serves now, so applying it would replace the live model with one "
            "nothing has compared against it. Nothing was copied or swapped."
        )
        raise StaleStageOneVerdictError(msg)

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
    parser.add_argument(
        "--gold-generation",
        type=str,
        default=None,
        help=(
            "RUN STAGE ONE, fitting one candidate per target under the staging root and "
            "declaring this gold generation key in every candidate's metadata. Supplying "
            "it is what arms stage one: a BARE invocation still runs the pre-flight only, "
            "which is the posture Plan 33-08 shipped and the one the tests pin. The value "
            "is measured with tests.gold_generation.gold_generation_key() and recorded in "
            "tests.phase33_state.GOLD_GENERATION_AT_REFIT. THE FIX-CYCLE ALLOWANCE IS "
            f"{PHASE33_FIX_CYCLE_ALLOWANCE}: a second invocation against an existing "
            "verdict record raises FixCycleAllowanceExceededError."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Stage one has no wired scorer here -- Plan 33-15 supplies it."""
    args = parse_args(argv)
    if not args.promote and args.gold_generation:
        # STAGE ONE, ARMED. Plan 33-15 wires the scorer; this is the reproducible
        # invocation, so the run that produced the committed verdict is a command in
        # committed source rather than a shell history nobody kept.
        result = run_stage_one(
            gold_generation=args.gold_generation,
            artifacts_dir=args.artifacts_dir,
            staging_dir=args.staging_dir,
            verdict_record_path=args.verdict_record,
        )
        record = result["record"]
        for target in GATED_TARGETS:
            row = record.targets.get(target)
            if row is None:
                continue
            print(
                f"  {target.upper():4s} {row['verdict']:<20s} "
                f"t={row['paired_statistic']} p={row['p_value']} "
                f"n_paired={row['n_paired']} eligible={row['eligible_pairs']}"
            )
        print(f"VERDICT RECORD: {args.verdict_record}")
        return 0
    if not args.promote:
        print(
            "Stage one needs a scorer AND a measured gold generation key. Pass "
            "--gold-generation to arm it (Plan 33-15). A bare invocation runs the "
            "pre-flight only, so the environment can be checked on its own."
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
