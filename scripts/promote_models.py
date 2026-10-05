"""The SOLE production-swap entry point for per-target model promotion (Phase 24).

This orchestrator is the only sanctioned path that may swap a candidate model into
production ``artifacts/latest.json``. It is dry-run by DEFAULT: a bare
``python -m scripts.promote_models`` trains candidates to ``artifacts_staging/``, scores
them, runs the frozen significance-tested deploy gate, prints the per-target 2x2 readout
(candidate vs frozen v1.0), and swaps NOTHING in production. ``--promote`` is REQUIRED for
any production write, and even then only PASSING targets are swapped (per-key, via
``models.artifacts.update_manifest``, preserving the ``blend`` pointer and any non-promoted
target). A gate FAIL produces zero production swaps AND a non-zero exit code so automation/CI
observes the hard block.

Why a hard block (the D-17 precedent): in v2.0 an ungated swap shipped a regression -- the
deployed v1.0 pre-Elo WP/ATS models were retained only because per-target gating caught the
regression after the fact. v3.0 deliberately re-fits (lifting the v2.1 D-01 boundary), so this
gate is the safety rail: nothing ships that regresses. The frozen thresholds + baseline live in
the git-tracked ``config/gate.toml``; this script never loosens them.

The flow (D24-08/09/10/11):
  STEP 1   staging straight re-fit (subprocess: python -m models.train --no-tune, no Optuna)
           into ``artifacts_staging/`` -- never production.
  STEP 1b  write the STAGING manifest ``artifacts_staging/latest.json`` (the subprocess train
           no longer auto-writes it; Plan 24-01 made BaseTrainer.save() pass update_latest=False)
           via update_manifest(artifacts_dir=staging_dir). D24-09 ALLOWS this staging manifest;
           D24-08 protects only production ``artifacts/latest.json``. The staged_version map is
           resolved DETERMINISTICALLY here (newest by the dir-name timestamp, not filesystem
           mtime) and reused in STEP 4 -- the staging manifest must exist before STEP 2 because
           score_deployed_artifacts resolves the staged artifact via that manifest.
  STEP 2   score the staged candidates on canonical 2021-2024 gold
           (diagnose.score_deployed_artifacts, artifacts_dir=staging_dir).
  STEP 3   build candidate bundles through the SHARED deploy_gate.build_candidate_bundle (the one
           bundle shape promote/retrain/tests all use) and run deploy_gate.evaluate_target against
           the frozen gate.toml baseline; print the 2x2 readout.
  STEP 4   on --promote, swap ONLY passing targets into production via update_manifest
           (artifacts_dir=artifacts_dir); a FAIL never swaps and exits non-zero.

This script deliberately does NOT wire into the Friday orchestrator (out of scope).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from backtest.diagnose import CLV_COLUMN_FOR, clv_significance, score_deployed_artifacts
from backtest.engine import BacktestEngine
from conf.season_partition import default_season_partition
from models import deploy_gate
from models.artifacts import update_manifest
from models.clv import compute_clv_for_predictions
from utils import get_logger

if TYPE_CHECKING:
    import pandas as pd

logger = get_logger(__name__)

# Targets promoted by this orchestrator (the production swap surface; "blend" is never
# touched by a re-fit -- update_manifest's per-key write preserves it).
_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

# SITE 9 of the season partition (RESEARCH 11.1): the LIVE holdout bounds this module
# slices gold on. DERIVED from deploy_gate.HOLDOUT_SEASONS, which is itself derived from
# conf.season_partition (SPEC R6, D33.1-03). They used to be two more literals reading
# 2021 and 2024, and the research table records them as the site that could let the legacy
# path silently disagree with the gate.
#
# THESE ARE THE *LIVE* BOUNDS AND THEY MOVE. The frozen 2021-2024 window the
# config/gate.toml [baseline.*] block was frozen over is a DIFFERENT set with a different
# name: deploy_gate.FROZEN_BASELINE_SEASONS. `_load_gold_holdout` slices on the pair below;
# `_load_drift_reproduction_frame` slices on the frozen set. Do not collapse them.
_HOLDOUT_FIRST_SEASON = deploy_gate.HOLDOUT_SEASONS[0]
_HOLDOUT_LAST_SEASON = deploy_gate.HOLDOUT_SEASONS[-1]

# D25-15 gate-time drift tripwire: RECOMPUTATION tolerance for the re-scored v1.0 aggregates
# vs the frozen config/gate.toml baseline. This is NOT a loosening of D25-15's "exact match"
# SEMANTIC intent -- the baseline MUST be the same artifacts, the same gold, and the same CLV
# column; drift in ANY of those is a hard abort, not a tolerance question. The tolerance covers
# only float/library recomputation jitter: a deterministic re-score on the same artifacts should
# reproduce the frozen numbers to within this band. The value mirrors the freshness band used by
# tests/integration/test_promote_models.py::test_frozen_baseline_matches_rescore (_FRESHNESS_TOL)
# and the test_diag_diagnosis anchor tolerance, so the gate-time abort uses the same recomputation
# noise band the committed freshness test already trusts.
_FRESHNESS_TOL = 5e-3

# The ratified Stage-1 group verdict (D24-07 generator-output-over-transcription: Plan 30-05
# writes the emitter, Plan 30-10 writes this file). When it exists and no explicit
# --exclude-groups was given, the Stage-2 exclusion list is DERIVED from it rather than typed.
#
# Anchored to the REPOSITORY, not to the process working directory. A CWD-relative path made
# the single most consequential input to WHAT gets trained depend on where the operator
# happened to be standing, and its not-found branch does not stop -- it excludes nothing and
# carries on (WR-01). The verdict is a git-tracked file at a fixed location in this repo, so
# resolving it from __file__ is both correct and un-spoofable by a chdir.
_GROUP_GATE_VERDICT_PATH = (
    Path(__file__).resolve().parent.parent / "config" / "group_gate_verdict.toml"
)

# The season lists ``_incumbent_window`` READS OUT OF THE INCUMBENT'S RECORD, mapped from the
# metadata config key to the ``models.train --config-*-seasons`` flag stem they are compared
# against. Since review CR-01 the values handed to models.train come from the committed rule,
# not from here; this mapping is what lets the drift report name all three fields.
_WINDOW_KEYS: tuple[tuple[str, str], ...] = (
    ("train_seasons", "train"),
    ("hp_val_seasons", "hp_val"),
    ("holdout_seasons", "holdout"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Parsed arguments namespace with promote, artifacts_dir, staging_dir, skip_train.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Staged per-target model promotion. Dry-run by default (swaps nothing); "
            "--promote performs the conditional per-target production swap."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--promote",
        action="store_true",
        help=(
            "Perform the conditional per-target production swap "
            "(default: dry-run, swaps nothing)"
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts"),
        help="Production artifacts dir -- the swap target (default: artifacts)",
    )
    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=Path("artifacts_staging"),
        help="Staging artifacts dir for the candidate re-fit (default: artifacts_staging)",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help=(
            "Reuse existing staging artifacts instead of re-training "
            "(keeps the gate path testable without a long train)"
        ),
    )
    parser.add_argument(
        "--exclude-groups",
        type=str,
        default=None,
        help=(
            "Comma-separated feature GROUPS to exclude from the Stage-2 candidate train, "
            "passed through to models.train --exclude-groups (D30-01). Omitting the flag "
            "derives the list from the ratified Stage-1 verdict "
            f"({_GROUP_GATE_VERDICT_PATH}) when that file exists, and otherwise excludes "
            "nothing with a loud warning. Passing the flag explicitly is an OVERRIDE and is "
            "announced as such -- an empty value ('') is a valid override meaning 'exclude "
            "nothing' and does NOT fall through to the verdict file."
        ),
    )
    return parser.parse_args(argv)


def _resolve_exclude_groups(args: argparse.Namespace) -> tuple[list[str], str]:
    """Resolve the Stage-2 feature-group exclusion list and where it came from.

    Precedence, in order:

      1. An explicit ``--exclude-groups`` value WINS and is announced as an OVERRIDE -- the
         list came from the command line and NOT from the ratified Stage-1 verdict. The branch
         tests ``is not None``, not truthiness, so ``--exclude-groups ""`` is an explicit
         "exclude nothing" rather than a silent fall-through.
      2. The ratified verdict file's ``excluded_groups`` key, DERIVED not transcribed (D24-07).
      3. An empty list, with a loud warning that no Stage-1 verdict has been ratified --
         but ONLY on an unarmed run. On the ARMED path (``--promote``) an absent verdict is
         a STOP.

    WR-01: this used to fail OPEN unconditionally. A missing verdict file printed a
    ``[WARNING]``, returned ``([], "none")``, and the run then trained three candidates on
    EVERY feature group -- including the group the frozen rule DROPPED -- gated them, and
    under ``--promote`` swapped any that passed. That silently reverses a ratified owner
    decision, with ``provenance: none`` in the banner as the only trace and nothing
    machine-checking it.

    The sibling input in this same module already insists on the opposite discipline:
    ``_incumbent_window``'s docstring says "an unresolvable window must be a stop, never a
    guess ... and it would still print a confident 2x2". The exclusion list is the MORE
    consequential of the two -- it decides what gets trained at all -- so it gets the same
    rule on the path that can mutate production. A dry run still warns and continues, because
    a dry run swaps nothing and its whole purpose is to be runnable on a checkout that has
    not ratified anything.

    Stating "exclude nothing" deliberately remains available at all times, on both paths:
    ``--exclude-groups ''`` is an explicit override.

    Args:
        args: The parsed promote arguments.

    Returns:
        ``(groups, provenance)`` where provenance is ``"override"``, ``"verdict"`` or
        ``"none"``. The provenance rides into the banner so checkpoint 4 can see whether the
        single most consequential input to what gets trained was ratified or hand-typed.

    Raises:
        FileNotFoundError: On an ARMED run (``--promote``) with no verdict file and no
            explicit ``--exclude-groups``.
    """
    if args.exclude_groups is not None:
        groups = [g.strip() for g in args.exclude_groups.split(",") if g.strip()]
        print(
            "  [OVERRIDE] --exclude-groups was supplied on the COMMAND LINE; this list is "
            "NOT the ratified Stage-1 verdict."
        )
        return groups, "override"

    if _GROUP_GATE_VERDICT_PATH.exists():
        with _GROUP_GATE_VERDICT_PATH.open("rb") as handle:
            verdict = tomllib.load(handle)
        groups = [str(g) for g in verdict.get("excluded_groups", [])]
        print(
            f"  Exclusion list DERIVED from the ratified Stage-1 verdict "
            f"({_GROUP_GATE_VERDICT_PATH})."
        )
        return groups, "verdict"

    if args.promote:
        msg = (
            f"No ratified Stage-1 verdict at '{_GROUP_GATE_VERDICT_PATH}' and no explicit "
            "--exclude-groups. Refusing an ARMED promotion whose candidate feature set was "
            "not Stage-1 selected: without the verdict every feature group is trained, "
            "INCLUDING any the frozen rule dropped, and a passing candidate would then be "
            "swapped into production -- silently reversing a ratified decision. Pass "
            "--exclude-groups '' to state 'exclude nothing' deliberately, or restore the "
            "verdict file."
        )
        raise FileNotFoundError(msg)

    print(
        f"  [WARNING] No ratified Stage-1 verdict at {_GROUP_GATE_VERDICT_PATH} and no "
        "--exclude-groups override: the candidate will be trained on EVERY feature group. "
        "That is a legitimate configuration, but it is not a Stage-1-selected feature set. "
        "This run is a DRY RUN and swaps nothing; --promote would REFUSE here."
    )
    return [], "none"


def _incumbent_window(target: str, artifacts_dir: Path) -> dict[str, str]:
    """Resolve one target's training window from THE COMMITTED RULE, and report the drift.

    WHAT THIS FUNCTION DOES NOW, AND WHY THE NAME IS KEPT ANYWAY. All three windows --
    train, hp_val and holdout -- come from ``conf.season_partition.default_season_partition()``
    (SPEC R6, D33.1-03). The deployed incumbent's ``metadata.json`` is still READ, and an
    unresolvable one is still a STOP, but it is read as the RECORD of a past training run
    rather than as an input to the next one: it supplies the left-hand side of the difference
    report below and nothing else. The name is kept because every caller, test and readout
    refers to it, and renaming it would be a refactor beyond this fix.

    WHY ALL THREE, AND NOT JUST THE HOLDOUT (Phase 33.1 review CR-01). Until this change the
    holdout came from the rule while train and hp_val were passed through from the incumbent's
    metadata verbatim. Three things were wrong with that, and none of them was cosmetic:

      1. Measured, a Wave-15 WP candidate would have run as ``--config-train-seasons 2018,2019``
         -- the 534-row window ``conf/season_partition.py``'s own evidence block records as
         admitting SIX of ATS's 25 and NINE of O/U's 25 selected features from pure synthetic
         noise, with 45 new live weather candidates now entering that same selector.
      2. The calibrator is fitted on the hp-val fold while the shipped model trains on
         everything before the last holdout season. An hp_val of 2020 against a 2024-2025
         holdout widened that in-sample overlap from one season to five.
      3. ``HISTORICAL-WEATHER-READOUT.md`` sections 0 and 3 tell Wave 15 that it consumes the
         committed rule "or it disagrees with the rest of the repository". Only the readout
         was true; this path was not.

    WHAT WAS DELIBERATELY NOT DISCARDED. D30-12's finding stands: the deployed ATS incumbent
    records a five-season train window that ``promote_models`` did not match, and a candidate
    facing its own incumbent from a worse configuration is a handicap nobody chose. The fix for
    that asymmetry is that BOTH sides are now stated by one committed rule instead of by three
    artifacts' metadata -- which is the same asymmetry closed one level up, not the asymmetry
    reintroduced. The per-target difference is now REPORTED rather than silently honoured.

    THE OWNER'S STANDING RULING is what makes this the only defensible shape: the pre-correction
    artifacts were fitted on data since found to be wrong, so they are void. Deriving the NEXT
    model's training window from a void model's metadata is the defect, not the safeguard.

    Args:
        target: One of "wp", "ats", "ou".
        artifacts_dir: The PRODUCTION artifacts dir holding ``latest.json`` and the version dirs.

    Returns:
        ``{"train": "2018,...,2022", "hp_val": "2023", "holdout": "2024,2025",
        "window_report": "..."}``. The first three are comma-joined season strings ready to
        hand to ``models.train --config-*-seasons``, and ALL THREE come from the committed
        partition rule. ``window_report`` is the DIFFERENCE between the incumbent's recorded
        windows and the live ones -- empty when all three agree, and otherwise a sentence
        naming every field that moved, both values, and the in-sample consequence. It used to
        be a raise, and then a holdout-only report; see the block comment below.

    Raises:
        FileNotFoundError: If ``latest.json`` or the resolved ``{version}/metadata.json`` is
            absent, naming the exact missing path.
        KeyError: If the manifest has no pointer for the target, or the metadata has no
            ``config`` block or is missing one of the season lists, naming the missing key.
    """
    manifest_path = artifacts_dir / "latest.json"
    if not manifest_path.exists():
        msg = (
            f"Cannot derive the '{target}' selection window: production manifest not found at "
            f"'{manifest_path}'. The window must be DERIVED from the deployed incumbent's own "
            "metadata (D30-12), never typed and never defaulted. Inspect that path; on a fresh "
            "checkout, bootstrap it per the clean-checkout step (RUNBOOK, Plan 25-05)."
        )
        raise FileNotFoundError(msg)

    with manifest_path.open(encoding="utf-8") as handle:
        versions = json.load(handle)

    version = versions.get(target)
    if version is None:
        msg = (
            f"Cannot derive the '{target}' selection window: manifest '{manifest_path}' has no "
            f"'{target}' pointer (it holds {sorted(versions)}). The window must be DERIVED from "
            "the deployed incumbent for THIS target; restore the pointer rather than supplying a "
            "window by hand."
        )
        raise KeyError(msg)

    metadata_path = artifacts_dir / version / "metadata.json"
    if not metadata_path.exists():
        msg = (
            f"Cannot derive the '{target}' selection window: metadata.json not found at "
            f"'{metadata_path}' (manifest points '{target}' at '{version}'). Do NOT delete the "
            "deployed incumbent artifact dirs -- they are the paired baseline AND the rollback "
            "target (D25-17). Restore that dir; the window is derived from it, not typed."
        )
        raise FileNotFoundError(msg)

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)

    config = metadata.get("config")
    if not config:
        msg = (
            f"Cannot derive the '{target}' selection window: '{metadata_path}' has no 'config' "
            "block, so it cannot supply a window. Inspect that file. Inventing a window here "
            "would reintroduce the D30-12 asymmetry the derivation exists to eliminate."
        )
        raise KeyError(msg)

    # The incumbent's recorded windows. Still read, and an unresolvable one is still a STOP --
    # but as the RECORD of a past run, which is the left-hand side of the difference report
    # below. Nothing here reaches models.train any more (review CR-01).
    recorded: dict[str, list[int]] = {}
    for metadata_key, flag_stem in _WINDOW_KEYS:
        seasons = config.get(metadata_key)
        if not seasons:
            msg = (
                f"Cannot derive the '{target}' selection window: '{metadata_path}' config block "
                f"has no '{metadata_key}' season list. A metadata file that cannot state what "
                "its own run was trained on cannot support the difference report this function "
                "returns, and a silently absent record is how an undisclosed window change "
                "happens. Inspect that file; do not substitute a default."
            )
            raise KeyError(msg)
        recorded[flag_stem] = [int(season) for season in seasons]

    # WR-02: the derived holdout is passed straight to ``models.train
    # --config-holdout-seasons``, while ``_load_gold_holdout`` and every gate call read the
    # FROZEN ``deploy_gate.HOLDOUT_SEASONS``. Nothing reconciled the two.
    #
    # If an incumbent's metadata ever recorded a NARROWER holdout, the candidate would be
    # trained over seasons the gate then scores it on -- ``BaseTrainer.train_and_evaluate``
    # keeps the model from the LAST split, so the saved artifact would have SEEN them -- and
    # an in-sample candidate would be compared against an out-of-sample baseline. A confident
    # 2x2 would print with no warning at all.
    #
    # All three incumbents currently record [2021, 2022, 2023, 2024], so this is latent. It
    # is checked HERE, before any train can start, because that is the only place it is
    # cheap: after twelve walk-forward re-fits it would be a refusal nobody could afford to
    # trust.
    #
    # ---------------------------------------------------------------------------
    # RETARGETED FROM A REFUSAL TO A REPORT (D33.1-04, Plan 33.1-09 Task 2).
    #
    # The comment above is kept VERBATIM because it is the clearest statement in this
    # repository of what D33.1-01 costs, and that cost is now REAL rather than latent. Under
    # D33.1-03 the partition comes from the committed rule in conf/season_partition.py, so
    # the per-target derivation from incumbent metadata (D30-12) is SUPERSEDED for the
    # holdout: all three incumbents record [2021..2024] and the live partition is (2024,
    # 2025), so this raise would fire on every target, every run, for a reason that is not a
    # defect.
    #
    # THE OWNER HAS ACCEPTED THE COST, AND ACCEPTING IT IS NOT THE SAME AS HIDING IT. Once
    # the shipped artifact has been fitted on the holdout seasons, promote_models' re-score of
    # it on those seasons is IN-SAMPLE and is no longer an out-of-sample generalisation
    # estimate. Wave 15's verdict must be LABELLED in-sample rather than presented as a clean
    # gate pass, and the readout must say so. That is what this report is for.
    #
    # EDITING AN ARTIFACT'S metadata.json TO MAKE THIS PASS IS PROHIBITED (D33.1-04). Those
    # files are the RECORD of a past training run; falsifying a record to unblock a gate is
    # the defect class this milestone exists to detect. The sentence this function used to
    # raise with -- "never widen a window to make this pass" -- was right, and it applies to
    # the record just as much as to the window.
    #
    # ---------------------------------------------------------------------------
    # RETARGETED AGAIN, FROM A HOLDOUT-ONLY REPORT TO A WHOLE-WINDOW ONE (review CR-01).
    #
    # Both comments above are kept VERBATIM: they are the clearest statement in this
    # repository of what D33.1-01 costs, and every word of them still holds. What changed is
    # their SCOPE. The report covered the holdout alone because the holdout alone came from
    # the rule; all three windows come from the rule now, so all three are compared. A field
    # that moved and was not named is precisely the failure this block exists to prevent, and
    # under the previous shape a train window inherited from a VOID pre-correction artifact
    # moved the next model's feature selection without appearing in any report at all.
    # ---------------------------------------------------------------------------
    # THE WINDOW HANDED TO models.train, ALL THREE FIELDS FROM THE COMMITTED RULE.
    # ``deploy_gate.HOLDOUT_SEASONS`` is this same ``partition.holdout`` (deploy_gate.py:163),
    # so the candidate is trained on exactly the window the gate then scores it over -- one
    # rule, read once, rather than two derivations that could drift apart.
    partition = default_season_partition()
    live: dict[str, tuple[int, ...]] = {
        "train": partition.selection,
        "hp_val": partition.hp_val,
        "holdout": partition.holdout,
    }
    window: dict[str, str] = {
        flag_stem: ",".join(str(season) for season in seasons)
        for flag_stem, seasons in live.items()
    }

    moved = [
        (flag_stem, recorded[flag_stem], list(live[flag_stem]))
        for flag_stem in ("train", "hp_val", "holdout")
        if recorded[flag_stem] != list(live[flag_stem])
    ]
    if not moved:
        window["window_report"] = ""
    else:
        differences = "; ".join(
            f"{flag_stem}: recorded {was} -> live {now}"
            for flag_stem, was, now in moved
        )
        window["window_report"] = (
            f"'{target}' incumbent metadata ('{metadata_path}') records windows that differ "
            f"from the LIVE partition in conf/season_partition.py -- {differences}. This is a "
            "REPORTED DIFFERENCE, not a refusal (D33.1-04): every window now comes from the "
            "committed rule, so the incumbent's recorded windows are historical facts about a "
            "past run rather than inputs to the next one. CONSEQUENCE, stated plainly: the "
            "candidate is trained over seasons the gate then re-scores it on, so that verdict "
            "is IN-SAMPLE and must be labelled in-sample rather than read as a clean gate "
            "pass. The metadata.json files are the record of past training runs and MUST NOT "
            "be edited to make this agree."
        )
        logger.warning(
            "Incumbent windows differ from the live partition",
            target=target,
            recorded=recorded,
            live={key: list(value) for key, value in live.items()},
            metadata_path=str(metadata_path),
        )

    return window


def _build_train_argv(
    target: str,
    staging_dir: Path,
    window: dict[str, str],
    exclude_groups: list[str],
    exclusion_provenance: str = "none",
) -> list[str]:
    """Build the STEP 1 ``models.train`` argv for ONE target.

    Pure and side-effect free so the argv contract is testable without spawning a subprocess.

    Two properties are load-bearing:

      * ``--no-tune`` is ABSENT. SPEC R5 requires the Stage-2 candidate to be trained WITH Optuna
        tuning; the flag that STEP 1 carried through Phases 24-25 would silently downgrade every
        candidate to a straight re-fit.
      * the three ``--config-*-seasons`` values come from ``_incumbent_window``, which since
        review CR-01 takes ALL THREE from the committed partition rule. STEP 1 still issues one
        subprocess PER TARGET: that is left UNCHANGED by the fix, because the argv is built per
        target regardless and because a future per-target window would need exactly this shape
        again (D30-12). What is no longer true is that the windows must differ between targets
        -- under the committed rule they are identical, and the per-target DIFFERENCE now shows
        up in ``window_report`` instead of in the argv.

    Args:
        target: One of "wp", "ats", "ou".
        staging_dir: The staging artifacts root -- never production (D24-08).
        window: The derived window from ``_incumbent_window``.
        exclude_groups: The resolved Stage-2 exclusion list (may be empty).
        exclusion_provenance: ``"verdict"``, ``"override"`` or ``"none"`` -- passed through
            so the trained artifact's own metadata records whether the exclusion was
            ratified or hand-typed (WR-04). This function already knew it; it was
            previously dropped on the floor at the argv boundary.

    Returns:
        The argv list for ``subprocess.run``.
    """
    return [
        sys.executable,
        "-m",
        "models.train",
        "--target",
        target,
        "--artifacts-dir",
        str(staging_dir),
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
    ]


#: The metadata key the in-sample window report is persisted under. Named once so the
#: writer here and any consumer of a promoted artifact spell it the same way.
WINDOW_REPORT_METADATA_KEY = "holdout_in_sample_report"


def _record_window_report(
    target: str,
    version: str,
    staging_dir: Path,
    report: str,
) -> bool:
    """Persist the in-sample window report into the STAGED candidate's metadata.

    WHY THIS EXISTS (code review WR-12). ``holdout_report`` -- the sentence saying the
    verdict is in-sample and must be LABELLED so -- was built into a dict, logged at
    WARNING and printed to stdout, and that was all. Nothing wrote it into the
    candidate's ``metadata.json``, into the gate verdict record, or into any other
    machine-readable place. ``HISTORICAL-WEATHER-READOUT.md`` (7e) makes the label
    BINDING: "Wave 15's verdict must be LABELLED in-sample, never presented as a clean
    gate pass." A stdout line in a long promotion run is the weakest possible carrier
    for a binding label, and it is exactly the "prints with no warning at all" outcome
    the function's own comment argues against.

    WHY THE STAGED COPY AND NOT THE PROMOTED ONE. D33.1-04 prohibits editing an
    artifact's ``metadata.json``: those files are the RECORD of a past training run.
    This writes into the STAGING dir, which D24-09 allows and which holds the run that
    has just happened -- so the label is part of the record from birth and travels into
    production with ``_promote_artifact_dir``'s copy. No existing record is edited.

    Args:
        target: One of "wp", "ats", "ou".
        version: The staged artifact dir name.
        staging_dir: The staging artifacts root -- never production (D24-08).
        report: The window report; an empty string writes nothing.

    Returns:
        True when the report was written, False when there was nothing to write or the
        staged metadata could not be read.
    """
    if not report:
        return False

    metadata_path = staging_dir / version / "metadata.json"
    if not metadata_path.exists():
        # NOT fatal, and deliberately so: the label is a disclosure ABOUT a verdict, and
        # losing the disclosure must not destroy the run that produced it. It is loud
        # instead, and the stdout line still prints.
        logger.warning(
            "Cannot persist the in-sample window report: staged metadata is absent",
            target=target,
            metadata_path=str(metadata_path),
        )
        return False

    with metadata_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    metadata[WINDOW_REPORT_METADATA_KEY] = report
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return True


def _parse_version_timestamp(version_dir_name: str) -> datetime:
    """Parse the trailing ``{YYYYMMDD}_{HHMMSS}`` timestamp from an artifact dir name.

    Versioned artifact dirs are named ``{target}_{YYYYMMDD}_{HHMMSS}`` (e.g.
    ``wp_20260327_114739``). Staged-version selection sorts by THIS embedded timestamp, never
    by filesystem modification time: filesystem mtimes can tie (a fast re-fit writes all three
    targets in the same second) and a stale dir's mtime can mislead. Parsing the name is
    deterministic (review concern #10 / T-24-17).

    Args:
        version_dir_name: An artifact directory name like ``wp_20260327_114739``.

    Returns:
        The parsed ``datetime`` (used only as a sort key).

    Raises:
        ValueError: If the name does not end in a parseable ``YYYYMMDD_HHMMSS`` stamp.
    """
    # The last two underscore-separated fields are the date and time stamp; the target prefix
    # itself may contain no underscore, but splitting from the right is robust regardless.
    parts = version_dir_name.rsplit("_", 2)
    if len(parts) < 3:
        msg = (
            f"Cannot parse timestamp from artifact dir name '{version_dir_name}': "
            "expected '{target}_{YYYYMMDD}_{HHMMSS}'."
        )
        raise ValueError(msg)
    stamp = f"{parts[-2]}_{parts[-1]}"
    return datetime.strptime(stamp, "%Y%m%d_%H%M%S")  # noqa: DTZ007 (naive sort key only)


def _resolve_staged_version(
    target: str,
    staging_dir: Path,
    *,
    skip_train: bool,
) -> str | None:
    """Pick the newest staged candidate dir for a target by its DIR-NAME timestamp.

    Globs ``{staging_dir}/{target}_*`` and returns the dir whose embedded
    ``{YYYYMMDD}_{HHMMSS}`` stamp is the max (deterministic; not filesystem mtime -- review
    concern #10). The empty-glob CONTRACT is two-branch (review concern #2):

      * ``skip_train=False`` (the real path just ran a staging train): a target with no
        matching dir is a HARD ERROR -- the train was supposed to produce it.
      * ``skip_train=True`` (the test/reuse path): a missing dir is a documented SKIP -- the
        target gets no staged_version entry, is excluded from the gate loop, and can never be
        indexed in STEP 4 (so no KeyError).

    Args:
        target: One of "wp", "ats", "ou".
        staging_dir: The staging artifacts root.
        skip_train: Whether the reuse path is active (changes the empty-glob disposition).

    Returns:
        The chosen version dir name (e.g. ``wp_20260327_114739``), or None when skip_train is
        set and no candidate dir exists for the target.

    Raises:
        RuntimeError: When skip_train is False and the staging train produced no candidate dir
            for the target.
    """
    candidates = [p for p in staging_dir.glob(f"{target}_*") if p.is_dir()]
    if not candidates:
        if skip_train:
            logger.warning(
                "No staged candidate dir for target under --skip-train; excluding it",
                target=target,
                staging_dir=str(staging_dir),
            )
            return None
        msg = f"staging train did not produce a {target} candidate dir under {staging_dir}"
        raise RuntimeError(msg)
    newest = max(candidates, key=lambda p: _parse_version_timestamp(p.name))
    return newest.name


def _clear_staging_dir(staging_dir: Path) -> None:
    """Delete stale candidate dirs and any stale latest.json before a fresh staging train.

    Clearing guarantees only the freshly-trained candidates exist, so the deterministic
    newest-by-name resolution can never select a stale dir (review concern #10 / T-24-17).
    Called ONLY on the real (non --skip-train) path; --skip-train must keep the supplied
    staging dirs intact.

    Removal failures are re-raised with the offending path NAMED (F8). ``shutil.rmtree`` raises
    ``PermissionError`` on Windows whenever a file underneath is read-only or is held open -- a
    live DuckDB/SQLite handle from an earlier run, or a file browser sitting in the directory.
    This host is Windows 11, so it is a real failure mode, and an unhandled traceback here is the
    worst available outcome: it lands BETWEEN clearing one staging dir and training the next,
    leaving the staging tree half-cleared with no message saying what to close.

    A directory the forward ledger references (a row's model or blend id, or an artifact the
    ledger keeps a copy of) is NEVER deleted (Phase 34 D-11): the ledger's replay depends on it,
    so a misconfigured ``--staging-dir artifacts`` cannot prune it.

    Args:
        staging_dir: The staging artifacts root to clear (created if absent).

    Raises:
        RuntimeError: If a stale dir or the stale manifest cannot be removed, naming the exact
            locked path and the two likely holders; or if a stale dir is referenced by the
            forward ledger, naming it.
    """
    if not staging_dir.exists():
        staging_dir.mkdir(parents=True, exist_ok=True)
        return

    import shutil

    from forward_ledger.retention import referenced_artifact_ids

    stale_dirs = [
        stale
        for target in _TARGETS
        for stale in staging_dir.glob(f"{target}_*")
        if stale.is_dir()
    ]
    # Checked for every dir BEFORE any is removed, so a refusal leaves staging untouched.
    referenced = referenced_artifact_ids()
    protected = sorted(stale.name for stale in stale_dirs if stale.name in referenced)
    if protected:
        msg = (
            f"Refusing to clear '{staging_dir}': the forward ledger references {protected}, "
            "and its replay depends on those artifacts (Phase 34 D-11). Nothing was removed; "
            "point --staging-dir at a staging root, never at a directory holding them."
        )
        raise RuntimeError(msg)
    for stale in stale_dirs:
        try:
            shutil.rmtree(stale)
        except OSError as exc:
            msg = _locked_path_message(stale, exc)
            raise RuntimeError(msg) from exc
    stale_manifest = staging_dir / "latest.json"
    if stale_manifest.exists():
        try:
            stale_manifest.unlink()
        except OSError as exc:
            msg = _locked_path_message(stale_manifest, exc)
            raise RuntimeError(msg) from exc


def _locked_path_message(path: Path, exc: OSError) -> str:
    """Build the actionable message for a staging path that could not be removed.

    Args:
        path: The exact path whose removal failed.
        exc: The underlying OSError (typically PermissionError / WinError 32).

    Returns:
        A runbook-style message naming the path, the underlying error and the remediation.
    """
    return (
        f"Cannot clear the staging path '{path}': {type(exc).__name__}: {exc}. On Windows this "
        "is almost always a HELD HANDLE or a read-only file. The two likely holders are (1) an "
        "open DuckDB/SQLite connection left behind by an earlier run, and (2) a file browser or "
        "editor sitting in that directory. Close the holder and re-run. Staging was NOT fully "
        "cleared, so do not re-run with --skip-train -- a stale candidate could be scored."
    )


def _promote_artifact_dir(
    target: str,
    version: str,
    staging_dir: Path,
    artifacts_dir: Path,
) -> None:
    """Copy a passing target's staged artifact dir into production (byte-identical, D25-06).

    ``update_manifest`` only rewrites the per-target POINTER in production ``latest.json``; it
    does NOT relocate the artifact files. The candidate was trained into -- and gate-scored from
    -- ``staging_dir/{version}`` (STEP 1/STEP 2 both use ``artifacts_dir=staging_dir``). When the
    staging dir differs from the production dir (the real run uses ``artifacts_staging`` vs
    ``artifacts``), pointing production ``latest.json`` at ``{version}`` without copying the dir
    leaves the deployed artifact UNRESOLVABLE: ``load_model_artifact(target)`` resolves
    ``artifacts/{version}`` and raises ``FileNotFoundError`` because the files only exist under
    staging. This copies the GATE-SCORED staging dir verbatim into production BEFORE the manifest
    swap, so the deployed artifact is byte-identical to the one the gate scored (D25-06) and is
    resolvable by every consumer.

    The copy is verbatim (``shutil.copytree``) -- no re-train, no re-serialize, so byte-identity is
    preserved. When staging and production are the SAME dir (e.g. ``--skip-train`` against the
    production dir, or a test that stages directly into production) the artifact is already in
    place and this is a no-op. A pre-existing production dir at the same version (re-promote of an
    identical version) is left untouched -- the version stamp makes a collision astronomically
    unlikely, and overwriting an in-place artifact would be the no-op staging==production case.

    Args:
        target: One of "wp", "ats", "ou" (for the log line).
        version: The staged artifact dir name to promote (e.g. ``wp_20260605_215552``).
        staging_dir: The staging artifacts root the candidate was trained/scored from.
        artifacts_dir: The production artifacts root (the swap target).

    Raises:
        FileNotFoundError: If the staged artifact dir does not exist (the gate scored it, so its
            absence here is an integrity failure, not a routine miss).
    """
    src = staging_dir / version
    dst = artifacts_dir / version
    if src.resolve() == dst.resolve():
        # staging == production: the gate-scored artifact is already in place (no-op).
        return
    if not src.is_dir():
        msg = (
            f"Staged artifact dir for '{target}' not found at '{src}'; cannot promote it into "
            f"production. The gate scored this exact dir, so its absence is an integrity failure."
        )
        raise FileNotFoundError(msg)
    if dst.exists():
        # Same version already present in production (re-promote of an identical stamp) -- the
        # gate-scored artifact is effectively in place; leave it untouched.
        return
    import shutil

    # Verbatim copy preserves byte-identity with the gate-scored staging artifact (D25-06).
    shutil.copytree(src, dst)


def _warn_skip_train_staleness(staging_dir: Path, *, promote: bool) -> None:
    """Print a loud staleness warning for each staged candidate under --skip-train.

    --skip-train reuses whatever already exists in ``staging_dir``; there is no guarantee those
    artifacts correspond to the current code/gold (review concern WR-06). _clear_staging_dir
    only runs on the real train path, so a stale dir from a previous (possibly buggy or
    pre-gold-rebuild) run can be scored and -- under ``--promote`` -- swapped into production.

    This makes the risk LOUD and visible: it reports each resolvable staged dir with its
    embedded ``{YYYYMMDD}_{HHMMSS}`` timestamp so the operator can see exactly how old the
    artifacts are. The strongest emphasis is reserved for ``--promote --skip-train`` (the
    production foot-gun). A full freshness sentinel (gold hash + git SHA) is intentionally out
    of scope here; this is the documented "loud warning + staged timestamp" mitigation.

    Args:
        staging_dir: The staging artifacts root being reused.
        promote: Whether the production swap is armed (raises the warning severity).
    """
    severity = "WARNING (production swap armed)" if promote else "NOTICE"
    print(
        f"  [{severity}] --skip-train reuses existing staging artifacts WITHOUT verifying "
        "they match the current code/gold."
    )
    for target in _TARGETS:
        candidates = [p for p in staging_dir.glob(f"{target}_*") if p.is_dir()]
        if not candidates:
            print(f"    {target}: (no staged dir found)")
            continue
        for cand in sorted(candidates, key=lambda p: p.name):
            try:
                stamp = _parse_version_timestamp(cand.name).isoformat(sep=" ")
            except ValueError:
                stamp = "unparseable timestamp"
            print(f"    {target}: {cand.name} (staged {stamp})")
    if promote:
        print(
            "    Confirm these staged artifacts are fresh before trusting the swap; re-run "
            "without --skip-train to train fresh candidates."
        )


def _load_gold_holdout(target: str, engine: BacktestEngine) -> pd.DataFrame:
    """Load the LIVE gold holdout for a target via the canonical engine loader.

    THIS IS THE LIVE FRAME: what the CANDIDATE and the INCUMBENT are scored on. It follows
    ``deploy_gate.HOLDOUT_SEASONS``, so it moves as the committed partition rule rolls
    forward -- on today's data, 2024-2025.

    IT IS NOT THE FRAME THE DRIFT TRIPWIRE WANTS, and the next reader's instinct will be to
    reuse it because it is already loaded. ``_drift_tripwire`` asks a question about a
    HISTORICAL record -- "does the frozen [baseline.*] block still reproduce on the seasons it
    was frozen over" -- and it can only ask that on THOSE seasons' rows. Handing it this frame
    is a deterministic abort, not a subtle inaccuracy: with the live holdout at 2024-2025,
    seasons 2021-2023 arrive with zero rows against a non-None frozen ``n`` and the exact
    integer sample-size comparison raises every time. Use
    :func:`_load_drift_reproduction_frame` for that, and keep the two frames apart.

    Reuses ``BacktestEngine._load_features`` (season-filtered, canonical team mapping) then
    restricts to the live holdout bounds.

    Args:
        target: One of "wp", "ats", "ou".
        engine: A constructed BacktestEngine (loader-only use here).

    Returns:
        The LIVE holdout gold frame for the target.
    """
    df = engine._load_features(target)
    in_holdout = (df["season"] >= _HOLDOUT_FIRST_SEASON) & (
        df["season"] <= _HOLDOUT_LAST_SEASON
    )
    holdout: pd.DataFrame = df.loc[in_holdout].copy()
    return holdout


def _load_drift_reproduction_frame(target: str, engine: BacktestEngine) -> pd.DataFrame:
    """Load the FROZEN-baseline gold frame the drift tripwire re-derives its record on.

    THIS IS THE FROZEN FRAME, and it exists because the live one is the wrong population for
    the question ``_drift_tripwire`` asks (Plan 33.1-09 Ruling S1). The tripwire compares the
    re-scored incumbent's pooled and per-season CLV aggregates -- including per-season sample
    sizes, as EXACT integers -- against ``config/gate.toml``'s ``[baseline.*]`` block. That
    block was frozen over ``deploy_gate.FROZEN_BASELINE_SEASONS`` (2021-2024) and never moves.
    So the only population on which the question "does the frozen record still reproduce" has
    an answer is the rows of those seasons.

    THIS PRESERVES THE CHECK RATHER THAN RELAXING IT. Nothing here loosens a tolerance or
    drops a field. The alternative that WOULD have relaxed it -- retargeting the tripwire's
    season LIST while leaving it fed from the live frame -- is what produced the deterministic
    abort; the alternative that would have REMOVED it -- retiring the legacy runner -- was
    rejected, because this tripwire is the only instrument asserting that the deployed
    artifacts still match ``config/gate.toml``, whose 47-of-68-field divergence is an
    undischarged disclosure (D33.1-05), and because ``scripts/run_phase33_gate.stage_two_promote``
    calls into this module's ``_promote_artifact_dir`` so the two runners are not separable
    anyway.

    RESEARCH P-7 records that ``_drift_tripwire`` is absent from CONTEXT's refusal inventory:
    it is the third runtime refusal, and the one nobody had named.

    Loads through the SAME canonical ``BacktestEngine._load_features`` the live frame uses, so
    the two differ only in which seasons they carry.

    Args:
        target: One of "wp", "ats", "ou".
        engine: A constructed BacktestEngine (loader-only use here).

    Returns:
        The gold frame restricted to ``deploy_gate.FROZEN_BASELINE_SEASONS``.
    """
    frozen = deploy_gate.FROZEN_BASELINE_SEASONS
    df = engine._load_features(target)
    in_frozen = (df["season"] >= min(frozen)) & (df["season"] <= max(frozen))
    reproduction: pd.DataFrame = df.loc[in_frozen].copy()
    return reproduction


def _baseline_bundle(target: str, cfg: dict[str, Any]) -> dict[str, Any]:
    """Flatten the frozen gate.toml baseline for a target into the evaluate_target shape.

    ``config/gate.toml`` stores the baseline NESTED (``[baseline.<t>.pooled]`` +
    ``[baseline.<t>.season.YYYY]``), but ``deploy_gate.evaluate_target`` reads a FLAT baseline
    bundle (``accuracy``/``ece``/``brier_score`` for WP, ``mae`` for ATS/OU). This adapts the
    nested config table to that flat shape, mapping the TOML ``brier`` key to the
    ``brier_score`` key evaluate_target reads. This does NOT build a candidate bundle -- the
    candidate side always flows through ``build_candidate_bundle`` (the single bundle seam);
    this only shapes the FROZEN baseline side for the comparison.

    Args:
        target: One of "wp", "ats", "ou".
        cfg: The loaded gate config (from ``deploy_gate.load_gate_config``).

    Returns:
        The flat baseline bundle evaluate_target consumes. Empty when the baseline table is
        unpopulated (Wave-1-tolerated shape); evaluate_target then records the secondary
        comparison as skipped.
    """
    target_cfg = cfg.get("baseline", {}).get(target, {})
    pooled = target_cfg.get("pooled", {})
    bundle: dict[str, Any] = {
        "mean": pooled.get("mean"),
        "t": pooled.get("t"),
        "p": pooled.get("p"),
    }
    if target == "wp":
        bundle["accuracy"] = pooled.get("accuracy")
        bundle["ece"] = pooled.get("ece")
        # gate.toml stores the calibration metric as "brier"; evaluate_target reads
        # "brier_score" (matching build_candidate_bundle's WP key).
        bundle["brier_score"] = pooled.get("brier")
    else:
        bundle["mae"] = pooled.get("mae")
    return bundle


def _assert_artifacts_dir_present(
    target: str, artifacts_dir: Path, version: str
) -> None:
    """Raise a clear, actionable error if a target's production artifact dir is missing.

    The paired baseline re-score (D25-15) loads the DEPLOYED INCUMBENT artifacts from the
    production artifacts dir (post-Phase-25/D25-11: the WP/ATS re-fits, OU retained v1.0);
    ``score_deployed_artifacts`` -> ``load_model_artifact`` would otherwise raise an OPAQUE
    failure if the dir (or its metadata) is absent. The deployed incumbent dirs are required
    BOTH for the paired re-score AND as the rollback target (D25-17), so a missing dir is a
    deploy-blocking integrity problem, not a transient: it must surface a named-path error that
    points at the clean-checkout bootstrap remedy (documented in DIAGNOSIS-NOTES.md / RUNBOOK by
    Plan 25-05) rather than an opaque load failure (Gemini consensus concern #2).

    Codex no-artifact-deletion guard: this check intentionally runs BEFORE the re-score so a
    missing production dir cannot be silently treated as "nothing to re-score". NEVER delete the
    deployed incumbent artifact dirs -- they are the paired baseline AND the rollback target.

    Args:
        target: One of "wp", "ats", "ou".
        artifacts_dir: The production artifacts dir (the swap target + baseline re-score source).
        version: The production version dir name for this target (from latest.json).

    Raises:
        FileNotFoundError: If the production artifacts dir, the target's version subdir, or its
            metadata.json is missing/empty -- with a named-path message and the bootstrap remedy.
    """
    if not artifacts_dir.exists():
        msg = (
            f"Production artifacts dir not found at '{artifacts_dir}'. The paired baseline "
            f"re-score (and rollback) needs the DEPLOYED v1.0 artifacts here. Do NOT delete the "
            "v1.0 artifact dirs. On a fresh checkout, bootstrap the first artifacts/latest.json "
            "per the clean-checkout bootstrap step (see DIAGNOSIS-NOTES.md / RUNBOOK, Plan 25-05)."
        )
        raise FileNotFoundError(msg)

    version_dir = artifacts_dir / version
    metadata = version_dir / "metadata.json"
    if not version_dir.is_dir() or not metadata.exists():
        msg = (
            f"Production artifact for target '{target}' missing at '{version_dir}' "
            f"(expected metadata at '{metadata}'). The paired baseline re-score (and rollback) "
            "needs the DEPLOYED v1.0 artifact for this target. Do NOT delete the v1.0 artifact "
            "dirs. On a fresh checkout, bootstrap the first artifacts/latest.json per the "
            "clean-checkout bootstrap step (see DIAGNOSIS-NOTES.md / RUNBOOK, Plan 25-05)."
        )
        raise FileNotFoundError(msg)


def _production_versions(artifacts_dir: Path) -> dict[str, str]:
    """Read the per-target version pointers from the production ``latest.json`` manifest.

    Used by the missing-dir guard to resolve which v1.0 artifact subdir each target points at,
    so the guard can name the exact expected path.

    Args:
        artifacts_dir: The production artifacts dir containing ``latest.json``.

    Returns:
        ``{target: version_dir_name}`` for the gated targets present in the manifest.

    Raises:
        FileNotFoundError: If ``latest.json`` itself is absent (a missing production manifest is
            the clean-checkout bootstrap gap; surface it with the bootstrap remedy).
    """
    manifest = artifacts_dir / "latest.json"
    if not manifest.exists():
        msg = (
            f"Production manifest not found at '{manifest}'. The paired baseline re-score needs "
            "the DEPLOYED v1.0 artifacts the manifest points at. On a fresh checkout, bootstrap "
            "the first artifacts/latest.json per the clean-checkout bootstrap step (see "
            "DIAGNOSIS-NOTES.md / RUNBOOK, Plan 25-05)."
        )
        raise FileNotFoundError(msg)
    with manifest.open(encoding="utf-8") as f:
        return json.load(f)


def _score_baseline_clv(
    target: str,
    gold_df: pd.DataFrame,
    odds_df: pd.DataFrame,
    artifacts_dir: Path,
) -> pd.DataFrame:
    """Re-score the DEPLOYED INCUMBENT artifacts and compute per-game baseline CLV (paired side).

    Mirrors the candidate-side scoring (STEP 2) but against the PRODUCTION artifacts dir: scores
    the deployed incumbent artifact (post-Phase-25/D25-11: the WP/ATS re-fits, OU retained v1.0)
    on the SAME gold the candidate was scored on, then computes the per-game CLV
    (``probability_clv`` for WP, ``line_clv`` for ATS/OU) via the same
    ``compute_clv_for_predictions`` path ``build_candidate_bundle`` uses internally. The returned
    frame carries ``game_id`` + ``season`` + the target's CLV column, restricted to games with
    closing odds -- the per-game baseline the candidate is paired against (D25-15).

    Args:
        target: One of "wp", "ats", "ou".
        gold_df: The SAME 2021-2024 gold holdout frame the candidate was scored on.
        odds_df: Normalized closing odds.
        artifacts_dir: The PRODUCTION artifacts dir (the deployed incumbent swap surface).

    Returns:
        A ``has_closing_odds``-filtered frame with ``game_id``, ``season``, and the target's CLV
        column.
    """
    scored = score_deployed_artifacts(
        target, gold_df=gold_df, artifacts_dir=artifacts_dir
    )
    clv_df = compute_clv_for_predictions(scored, odds_df, target)
    valid = clv_df.loc[clv_df["has_closing_odds"]]
    col = CLV_COLUMN_FOR[target]
    return valid[["game_id", "season", col]].copy()


def _candidate_clv_frame(
    target: str,
    scored_df: pd.DataFrame,
    odds_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the candidate per-game CLV frame the SAME way ``build_candidate_bundle`` does.

    ``build_candidate_bundle`` internally drops pre-existing CLV/odds columns, recomputes CLV via
    ``compute_clv_for_predictions``, and filters to ``has_closing_odds``; it does NOT expose the
    per-game frame. This re-derives that exact frame (``game_id`` + ``season`` + the target's CLV
    column) so the paired delta is built on the SAME population the bundle's pooled metrics used --
    no second scoring pass, just the same recompute on the already-scored frame.

    Args:
        target: One of "wp", "ats", "ou".
        scored_df: The candidate scored frame from ``score_deployed_artifacts`` (staging dir).
        odds_df: Normalized closing odds.

    Returns:
        A ``has_closing_odds``-filtered frame with ``game_id``, ``season``, and the target's CLV
        column.
    """
    pre_drop = [c for c in deploy_gate._CLV_ODDS_COLS if c in scored_df.columns]
    base = scored_df.drop(columns=pre_drop) if pre_drop else scored_df
    clv_df = compute_clv_for_predictions(base, odds_df, target)
    valid = clv_df.loc[clv_df["has_closing_odds"]]
    col = CLV_COLUMN_FOR[target]
    return valid[["game_id", "season", col]].copy()


def _drift_tripwire(
    target: str,
    baseline_valid: pd.DataFrame,
    cfg: dict[str, Any],
) -> None:
    """HARD-assert the re-scored v1.0 aggregates equal config/gate.toml BEFORE the paired test.

    THE FRAME THIS TAKES IS THE FROZEN ONE, NOT THE LIVE ONE (Plan 33.1-09 Ruling S1). Feed it
    ``_load_drift_reproduction_frame``'s output -- gold restricted to
    ``deploy_gate.FROZEN_BASELINE_SEASONS`` -- and NEVER ``_load_gold_holdout``'s, even though
    that one is already loaded two lines earlier at the call site. This function asks whether
    the frozen ``[baseline.*]`` record still reproduces on the seasons it was frozen over, and
    it can only ask that on those seasons' rows. Handing it the live frame is a deterministic
    abort: with the live holdout at 2024-2025 every frozen season yields zero rows against a
    non-None frozen ``n``, and the exact-integer sample-size comparison below raises every
    time. Retargeting the season LIST alone is NOT enough, and believing it was is the defect
    this ruling corrects.

    Nothing here is relaxed to accommodate the moved holdout. The tolerance, the field list and
    the exact-integer sample-size rule are unchanged; only the POPULATION the question is asked
    on is corrected to the one the question is about.

    D25-15 anchor (T-25-02-drift): the paired non-regression delta is only meaningful if the
    baseline side is the SAME frozen judge ``config/gate.toml`` describes. This re-derives the
    re-scored v1.0 pooled AND per-season CLV aggregates from ``baseline_valid`` and HARD-asserts
    they match the committed ``[baseline.<target>.*]`` values on ALL frozen fields, raising on ANY
    mismatch so the gate aborts before forming a delta against a drifted baseline (Codex MEDIUM:
    compare all fields, not only the means).

    Fields compared:
      * the CLV column identity -- ``CLV_COLUMN_FOR[target]`` is the column the re-score and the
        frozen baseline were both measured on; a different column means a different metric;
      * the pooled CLV mean;
      * each per-season CLV mean AND its per-season sample size ``n`` (the frozen block carries
        per-season ``n``; a different population is a hard drift signal).

    The numeric comparison uses ``_FRESHNESS_TOL`` (5e-3), DOCUMENTED at its definition as a
    RECOMPUTATION-noise tolerance -- NOT a loosening of D25-15's "exact match" SEMANTIC intent
    (same artifacts, same gold, same column). The sample-size comparison is EXACT (an integer
    population count cannot drift by float noise).

    Args:
        target: One of "wp", "ats", "ou".
        baseline_valid: The re-scored v1.0 per-game frame (``game_id``, ``season``, CLV column),
            ``has_closing_odds``-filtered, from ``_score_baseline_clv`` applied to
            ``_load_drift_reproduction_frame``'s FROZEN frame -- never to the live holdout.
        cfg: The loaded gate config (with int-normalized baseline season keys).

    Raises:
        ValueError: If the re-scored v1.0 frame is degenerate (zero has_closing_odds rows, the
            most extreme drift -- WR-01) or its aggregates drift from the frozen config on ANY
            field (CLV column identity, pooled mean, a per-season mean, or a per-season sample
            size).
    """
    frozen = cfg.get("baseline", {}).get(target, {})
    pooled_frozen = frozen.get("pooled", {})
    col = CLV_COLUMN_FOR[target]

    # Column-identity check: the frozen baseline and this re-score must be on the SAME CLV column.
    if col not in baseline_valid.columns:
        msg = (
            f"Drift tripwire ABORT for '{target}': the re-scored v1.0 frame lacks the frozen CLV "
            f"column '{col}' (CLV_COLUMN_FOR[{target!r}]). The baseline was measured on a "
            "different column than the gate now reads -- the paired test would compare two metrics."
        )
        raise ValueError(msg)

    # Pooled mean: re-derive from the re-scored v1.0 CLV and compare to the frozen pooled mean.
    # Empty-frame guard (WR-01): a zero-row re-score is the MOST extreme drift (the baseline the
    # gate pairs against does not exist for this target), so it must be the LOUDEST abort, not a
    # silent pass. np.mean([]) is nan and `abs(nan - frozen) > tol` evaluates False, so without
    # this guard the pooled check would no-op on a degenerate baseline and rely solely on the
    # downstream per-season exact-n check to catch it.
    clv_arr = baseline_valid[col].to_numpy()
    if clv_arr.size == 0:
        msg = (
            f"Drift tripwire ABORT for '{target}': re-scored v1.0 frame has ZERO "
            f"has_closing_odds rows on column '{col}'; the baseline is degenerate and "
            "cannot be compared to the frozen config. Re-freeze the baseline (Plan 25-05) "
            "or restore the deployed artifacts."
        )
        raise ValueError(msg)
    pooled_mean = float(np.mean(clv_arr))
    frozen_pooled_mean = pooled_frozen.get("mean")
    if frozen_pooled_mean is not None:
        drift = abs(pooled_mean - float(frozen_pooled_mean))
        if drift > _FRESHNESS_TOL:
            msg = (
                f"Drift tripwire ABORT for '{target}': re-scored v1.0 pooled CLV mean "
                f"{pooled_mean:.8f} drifted from the frozen config {float(frozen_pooled_mean):.8f} "
                f"by {drift:.3e} (> {_FRESHNESS_TOL:.0e} recomputation tolerance). The deployed "
                "artifacts no longer match config/gate.toml; the paired test would use a wrong "
                "baseline. Re-freeze the baseline (Plan 25-05) or restore the deployed artifacts."
            )
            raise ValueError(msg)

    # Per-season means AND sample sizes (Codex MEDIUM: compare per-season fields, not only pooled).
    #
    # The season list is FROZEN_BASELINE_SEASONS, not the live HOLDOUT_SEASONS (Plan 33.1-09,
    # D33.1-05). These are the seasons the [baseline.*] block was frozen over; the live
    # holdout has moved past them and is a different question, asked elsewhere.
    season_frozen = frozen.get("season", {})
    for season in deploy_gate.FROZEN_BASELINE_SEASONS:
        season_block = season_frozen.get(int(season))
        if not season_block:
            continue
        slice_clv = baseline_valid.loc[
            baseline_valid["season"] == season, col
        ].to_numpy()
        # Sample size is an EXACT integer comparison -- a population-count change is a hard drift
        # signal that no float-noise tolerance should absorb.
        frozen_n = season_block.get("n")
        if frozen_n is not None and len(slice_clv) != int(frozen_n):
            msg = (
                f"Drift tripwire ABORT for '{target}' season {season}: re-scored v1.0 sample size "
                f"{len(slice_clv)} != frozen config n={int(frozen_n)}. The re-score covers a "
                "different game population than the frozen baseline; the paired test would compare "
                "mismatched populations. Re-freeze the baseline (Plan 25-05) or restore artifacts."
            )
            raise ValueError(msg)
        frozen_season_mean = season_block.get("mean")
        if frozen_season_mean is not None and len(slice_clv):
            season_mean = float(np.mean(slice_clv))
            drift = abs(season_mean - float(frozen_season_mean))
            if drift > _FRESHNESS_TOL:
                msg = (
                    f"Drift tripwire ABORT for '{target}' season {season}: re-scored v1.0 CLV mean "
                    f"{season_mean:.8f} drifted from the frozen config "
                    f"{float(frozen_season_mean):.8f} by {drift:.3e} "
                    f"(> {_FRESHNESS_TOL:.0e} recomputation tolerance). The deployed artifacts no "
                    "longer match config/gate.toml; re-freeze (Plan 25-05) or restore artifacts."
                )
                raise ValueError(msg)


def _populate_paired_delta_keys(
    target: str,
    candidate: dict[str, Any],
    candidate_valid: pd.DataFrame,
    baseline_valid: pd.DataFrame,
) -> None:
    """Populate the Plan 25-01 PINNED non-regression delta keys via merge-on-game_id pairing.

    The Plan 25-01 ``build_candidate_bundle`` ships ``baseline_clv_values`` / ``clv_delta_values``
    (and the per-season equivalents) as ``None`` placeholders; this populates them IN PLACE from a
    genuine per-game pairing (D25-15). The candidate per-game CLV and the re-scored v1.0 per-game
    CLV are merged on ``game_id`` so the delta is paired on the INTERSECTION of games both sides
    scored with closing odds (T-25-02-pairing: equal n on the merged set, not a flattened
    aggregate). The internal-consistency invariant Plan 25-01 asserts is preserved by construction:
    ``clv_delta_values == clv_values - baseline_clv_values`` element-wise on the merged order.

    Args:
        target: One of "wp", "ats", "ou".
        candidate: The candidate bundle from ``build_candidate_bundle`` (mutated in place).
        candidate_valid: The candidate per-game frame (``game_id``, ``season``, CLV column),
            ``has_closing_odds``-filtered.
        baseline_valid: The re-scored v1.0 per-game frame, ``has_closing_odds``-filtered.
    """
    col = CLV_COLUMN_FOR[target]
    paired = candidate_valid.merge(
        baseline_valid,
        on="game_id",
        how="inner",
        suffixes=("_cand", "_base"),
    )
    cand_clv = paired[f"{col}_cand"].to_numpy()
    base_clv = paired[f"{col}_base"].to_numpy()
    delta = cand_clv - base_clv

    # Pooled paired arrays (aligned by game_id on the intersection both sides scored).
    candidate["clv_values"] = cand_clv
    candidate["baseline_clv_values"] = base_clv
    candidate["clv_delta_values"] = delta
    pooled = clv_significance(cand_clv)
    candidate["mean"] = pooled["mean"]
    candidate["t"] = pooled["t"]
    candidate["p"] = pooled["p"]
    candidate["n"] = pooled["n"]

    # Per-season paired arrays (the merged frame carries season from BOTH sides; they are equal on
    # the intersection, so the candidate-side season suffix is the per-season slice key).
    season_col = "season_cand"
    per_season_baseline: dict[int, Any] = {}
    per_season_delta: dict[int, Any] = {}
    per_season_cand: dict[int, Any] = {}
    for season in deploy_gate.HOLDOUT_SEASONS:
        mask = paired[season_col] == season
        per_season_cand[int(season)] = paired.loc[mask, f"{col}_cand"].to_numpy()
        per_season_baseline[int(season)] = paired.loc[mask, f"{col}_base"].to_numpy()
        per_season_delta[int(season)] = (
            paired.loc[mask, f"{col}_cand"].to_numpy()
            - paired.loc[mask, f"{col}_base"].to_numpy()
        )
    candidate["per_season_clv_values"] = per_season_cand
    candidate["per_season_baseline_clv_values"] = per_season_baseline
    candidate["per_season_clv_delta_values"] = per_season_delta
    # Refresh the per-season raw-candidate significance bundles to the paired (intersection)
    # population so the absolute-vs-zero per-season verdict matches the paired set.
    candidate["per_season"] = {
        int(season): clv_significance(per_season_cand[int(season)])
        for season in deploy_gate.HOLDOUT_SEASONS
    }


def _fmt(value: Any) -> str:
    """Format a metric for the readout (4dp float, or 'N/A' for None)."""
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _print_target_readout(
    target: str,
    result: dict[str, Any],
) -> None:
    """Print the per-target 2x2 readout (candidate vs frozen v1.0) for one target.

    Reuses the ``[PASS]``/``[FAIL]`` idiom from ``scripts.retrain_models.print_gating_summary``.
    Prints the pooled CLV mean/p, the secondary metric (WP accuracy + ECE/Brier, ATS/OU MAE),
    the per-season CLV deltas as an INFORMATIONAL readout (Open-Q-A: per-season secondary is a
    readout, not a blocker), and every gate reason.

    Args:
        target: One of "wp", "ats", "ou".
        result: The evaluate_target result dict for the target.
    """
    candidate = result.get("candidate", {})
    baseline = result.get("baseline", {})
    status = "[PASS]" if result["passed"] else "[FAIL]"

    print(f"\n{target.upper()} -- {status}")
    cand_mean = candidate.get("mean")
    cand_p = candidate.get("p")
    base_mean = baseline.get("mean")
    print(
        f"  Pooled CLV: candidate mean={_fmt(cand_mean)} p={_fmt(cand_p)} "
        f"| frozen v1.0 mean={_fmt(base_mean)}"
    )

    if target == "wp":
        print(
            f"  Accuracy: candidate={_fmt(candidate.get('accuracy'))} "
            f"| v1.0={_fmt(baseline.get('accuracy'))}"
        )
        print(
            f"  ECE: candidate={_fmt(candidate.get('ece'))} "
            f"| v1.0={_fmt(baseline.get('ece'))}    "
            f"Brier: candidate={_fmt(candidate.get('brier_score'))} "
            f"| v1.0={_fmt(baseline.get('brier_score'))}"
        )
    else:
        print(
            f"  MAE: candidate={_fmt(candidate.get('mae'))} "
            f"| v1.0={_fmt(baseline.get('mae'))}"
        )

    # Per-season CLV deltas: informational readout only (Open-Q-A; never blocks here).
    per_season = result.get("per_season", {})
    if per_season:
        print("  Per-season CLV (readout only, not a secondary blocker):")
        for season in sorted(per_season):
            season_sig = per_season[season]
            print(
                f"    {season}: mean={_fmt(season_sig.get('mean'))} "
                f"p={_fmt(season_sig.get('p'))} n={season_sig.get('n')}"
            )

    for reason in result["reasons"]:
        print(f"  - {reason}")

    if result["passed"]:
        print(f"  >> Ship candidate {target.upper()}")
    else:
        print(f"  >> Keep v1.0 {target.upper()} (gate FAIL)")


def main(argv: list[str] | None = None) -> int:
    """Run the staged-promotion orchestrator and return a gate-driven exit code.

    Dry-run by default (swaps nothing); --promote performs the conditional per-target
    production swap. Returns 1 if ANY gated target fails the deploy gate, else 0 -- the hard
    block is observable to CI even in dry-run (D24-10).

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Exit code: 1 if any target failed the gate, else 0.
    """
    args = parse_args(argv)

    print("=" * 70)
    print("NFL Prediction System -- Staged Model Promotion")
    print("=" * 70)
    mode = (
        "PROMOTE (will swap passing targets)" if args.promote else "DRY-RUN (no swap)"
    )
    print(f"  Mode: {mode}")
    print(f"  Production artifacts dir: {args.artifacts_dir}")
    print(f"  Staging artifacts dir:    {args.staging_dir}")
    print(f"  Skip train (reuse staging): {args.skip_train}")

    # The exclusion list is the single most consequential input to WHAT gets trained, and the
    # per-target selection window is what each candidate is compared against. Both are printed
    # here, with the exclusion's provenance, because checkpoint 4 reviews this output.
    exclude_groups, exclusion_provenance = _resolve_exclude_groups(args)
    print(
        f"  Exclude groups: {exclude_groups or '(none)'} "
        f"[provenance: {exclusion_provenance}]"
    )

    windows: dict[str, dict[str, str]] = {}
    try:
        for target in _TARGETS:
            windows[target] = _incumbent_window(target, args.artifacts_dir)
    except (FileNotFoundError, KeyError, ValueError) as exc:
        if not args.skip_train:
            # A window that cannot be resolved is a STOP, never a default (D30-12). Since
            # review CR-01 the windows themselves come from the committed rule, so what stops
            # the run here is an incumbent whose RECORD cannot be read -- which would leave
            # the in-sample window difference unstatable.
            raise
        # --skip-train trains nothing, so an underivable window is not fatal here; say so.
        windows = {}
        print(f"  Selection windows: UNAVAILABLE ({exc}) -- no train will run.")
    for target, window in windows.items():
        print(
            f"  Selection window [{target}]: train={window['train']} "
            f"hp_val={window['hp_val']} holdout={window['holdout']}"
        )
        # D33.1-04: a window that differs from the incumbent's record is REPORTED here rather
        # than raised in _incumbent_window. Printed at the same place the window is, because
        # checkpoint 4 reviews this output and an in-sample verdict is exactly the thing that
        # must not print with no warning at all. Since review CR-01 this covers all three
        # fields, not the holdout alone.
        if window.get("window_report"):
            print(f"  WINDOW DIFFERENCE [{target}]: {window['window_report']}")
    print("=" * 70)

    # -- STEP 1: staging TUNED re-fit (Optuna ON, SPEC R5), into the staging dir only --
    print("\n[Step 1/4] Staging re-fit (TUNED, Optuna ON) -> staging dir...")
    if args.skip_train:
        print("  --skip-train set: reusing existing staging artifacts.")
        # WR-06: --skip-train scores whatever already exists with no freshness check; make the
        # staleness LOUD and print each staged dir's embedded timestamp (strongest emphasis when
        # --promote is also set, the production foot-gun).
        _warn_skip_train_staleness(args.staging_dir, promote=args.promote)
    else:
        # Clear stale dirs first so newest-by-name resolution cannot pick a stale candidate.
        # This stays OUTSIDE the per-target loop on purpose: its glob is per-target
        # (``{target}_*``), so moving it inside would delete a sibling candidate that a
        # previous iteration of this same run had just trained.
        _clear_staging_dir(args.staging_dir)
        # One subprocess PER TARGET (D30-12). A single ``--target all`` invocation cannot carry
        # three different per-target selection windows, and the untuned flag is gone so Optuna
        # actually runs (SPEC R5).
        for target in _TARGETS:
            argv_train = _build_train_argv(
                target,
                args.staging_dir,
                windows[target],
                exclude_groups,
                exclusion_provenance,
            )
            print(f"  Training {target.upper()} candidate: {' '.join(argv_train[2:])}")
            subprocess.run(argv_train, check=True)
        print("  Staging re-fit complete.")

    # -- STEP 1b: write the STAGING manifest + deterministic staged_version resolution --
    # The subprocess train wrote NO artifacts_staging/latest.json (BaseTrainer.save() passes
    # update_latest=False, Plan 24-01). score_deployed_artifacts in STEP 2 resolves the staged
    # artifact via that manifest, so it MUST exist first. This writes the STAGING manifest ONLY
    # (artifacts_dir=args.staging_dir) -- it NEVER touches production (D24-08/09).
    print("\n[Step 1b/4] Writing staging manifest + resolving staged versions...")
    staged_version: dict[str, str] = {}
    for target in _TARGETS:
        chosen = _resolve_staged_version(
            target, args.staging_dir, skip_train=args.skip_train
        )
        if chosen is None:
            continue  # --skip-train: target excluded from the gate loop and any swap.
        staged_version[target] = chosen
        update_manifest(target, chosen, artifacts_dir=args.staging_dir)
        # WR-12: the in-sample label rides in the candidate's own metadata, so any
        # consumer of the promoted artifact inherits it instead of relying on somebody
        # having read the console. Written into STAGING, before the promotion copy.
        report = windows.get(target, {}).get("window_report", "")
        if _record_window_report(target, chosen, args.staging_dir, report):
            print(f"  {target}: in-sample window report recorded in metadata.json")
        print(f"  {target}: {chosen}")
    if not staged_version:
        print("  No staged candidates resolved; nothing to gate.")
        # WR-06: under an armed --promote, finding zero staged candidates is an operator
        # error (e.g. --skip-train against an empty/wrong staging dir), not a clean no-op --
        # exit non-zero so automation/CI observes that the requested promotion did nothing.
        if args.promote:
            print(
                "  --promote was requested but no staged candidates exist to promote; "
                "exiting non-zero."
            )
            return 1
        return 0

    # -- STEP 2: score the staged candidates on canonical 2021-2024 gold --
    print("\n[Step 2/4] Scoring staged candidates on 2021-2024 gold...")
    engine = BacktestEngine()
    odds_df = engine._load_closing_odds()
    scored: dict[str, pd.DataFrame] = {}
    gold_holdout: dict[str, pd.DataFrame] = {}
    for target in staged_version:
        gold_holdout[target] = _load_gold_holdout(target, engine)
        scored[target] = score_deployed_artifacts(
            target, gold_df=gold_holdout[target], artifacts_dir=args.staging_dir
        )
        print(f"  {target}: scored {len(scored[target])} games")

    # -- STEP 3: build bundles via the SHARED builder, pair vs the re-scored v1.0 baseline, --
    # -- then run the gate; print the 2x2 readout. --
    # D25-15: the non-regression floor needs a PAIRED per-game CLV delta, so the deployed v1.0
    # artifacts are re-scored on the SAME gold as each candidate (the baseline side), guarded by
    # a missing-artifacts-dir check and a gate-time drift tripwire, then merged on game_id with the
    # candidate per-game CLV to populate the Plan 25-01 pinned delta keys. The whole baseline
    # pipeline runs ONLY when build_candidate_bundle shipped clv_delta_values as the None
    # placeholder (the real path); a forced-verdict bundle (the hermetic integration seam) ships
    # the delta keys pre-populated and is left untouched, so the gate's downstream consumers stay
    # testable without canonical gold or the deployed v1.0 dirs.
    print(
        "\n[Step 3/4] Building candidate bundles, pairing vs the re-scored v1.0 baseline, "
        "running the deploy gate..."
    )
    cfg = deploy_gate.load_gate_config()
    deploy_gate.validate_gate_config(cfg)

    gate_results: dict[str, dict[str, Any]] = {}
    for target in staged_version:
        candidate = deploy_gate.build_candidate_bundle(
            target, scored[target], odds_df, cfg
        )
        if candidate.get("clv_delta_values") is None:
            # Real path: re-score the deployed v1.0 baseline, guard a missing dir, abort on drift,
            # then pair on game_id to populate the pinned non-regression delta keys (D25-15).
            prod_versions = _production_versions(args.artifacts_dir)
            prod_version = prod_versions.get(target)
            if prod_version is None:
                msg = (
                    f"Production manifest has no '{target}' pointer; cannot re-score the v1.0 "
                    "baseline for the paired non-regression delta. Restore the manifest or "
                    "bootstrap it (see DIAGNOSIS-NOTES.md / RUNBOOK, Plan 25-05)."
                )
                raise KeyError(msg)
            _assert_artifacts_dir_present(target, args.artifacts_dir, prod_version)

            # TWO FRAMES, TWO PURPOSES (Plan 33.1-09 Ruling S1). Each is named for what it
            # is for, because the one that is already loaded is the wrong one for the
            # tripwire and reusing it is the obvious mistake.
            #
            #   gold_holdout[target] -- the LIVE partition. What the candidate is scored on,
            #                           and therefore what the incumbent must be paired
            #                           against for the non-regression delta (D25-15).
            #   drift_frame          -- the FROZEN partition. The only population on which
            #                           "does config/gate.toml's [baseline.*] record still
            #                           reproduce" has an answer.
            #
            # Before this split the live frame was handed to both, which was correct only
            # while the two windows happened to coincide. They no longer do.
            drift_frame = _load_drift_reproduction_frame(target, engine)
            drift_valid = _score_baseline_clv(
                target, drift_frame, odds_df, args.artifacts_dir
            )
            # Gate-time drift tripwire: HARD-abort if the re-scored v1.0 drifted from gate.toml on
            # ANY frozen field (CLV column, pooled mean, per-season means + sample sizes).
            _drift_tripwire(target, drift_valid, cfg)
            print(
                f"  {target}: frozen-baseline re-scored {len(drift_valid)} games over "
                f"{sorted(deploy_gate.FROZEN_BASELINE_SEASONS)} (drift tripwire PASS)"
            )

            # Reuse the gold frame loaded for the candidate side (same window, same target) so the
            # baseline is paired on the SAME gold without a redundant parquet read.
            baseline_valid = _score_baseline_clv(
                target, gold_holdout[target], odds_df, args.artifacts_dir
            )
            print(
                f"  {target}: live-partition baseline re-scored "
                f"{len(baseline_valid)} games over "
                f"{sorted(deploy_gate.HOLDOUT_SEASONS)}"
            )
            candidate_valid = _candidate_clv_frame(target, scored[target], odds_df)
            _populate_paired_delta_keys(
                target, candidate, candidate_valid, baseline_valid
            )
        baseline = _baseline_bundle(target, cfg)
        gate_results[target] = deploy_gate.evaluate_target(
            target, candidate, baseline, cfg
        )
        # WR-12: the verdict carries its own label. A consumer reading gate_results and
        # nothing else would otherwise have no way to know the verdict is in-sample.
        gate_results[target][WINDOW_REPORT_METADATA_KEY] = windows.get(target, {}).get(
            "window_report", ""
        )

    print("\n" + "=" * 70)
    print("PER-TARGET DEPLOY GATE -- candidate vs frozen v1.0 (2x2 readout)")
    print("=" * 70)
    for target in staged_version:
        _print_target_readout(target, gate_results[target])
    print("\n" + "=" * 70)

    passing = [t for t in staged_version if gate_results[t]["passed"]]
    failing = [t for t in staged_version if not gate_results[t]["passed"]]
    print(f"SUMMARY: {len(passing)}/{len(staged_version)} gated targets pass")
    print(f"  Pass: {', '.join(t.upper() for t in passing) or 'none'}")
    print(f"  Fail: {', '.join(t.upper() for t in failing) or 'none'}")
    print("=" * 70)

    # -- STEP 4: conditional per-target production swap (only on --promote, only passing) --
    print("\n[Step 4/4] Production swap...")
    any_fail = bool(failing)
    if args.promote:
        for target in passing:
            # Copy the GATE-SCORED staged artifact dir into production FIRST so the manifest
            # pointer resolves to a real, byte-identical artifact (D25-06). update_manifest only
            # rewrites the pointer; without this copy the deployed version would be unresolvable
            # when staging_dir != artifacts_dir (the real run uses artifacts_staging vs artifacts).
            _promote_artifact_dir(
                target,
                staged_version[target],
                args.staging_dir,
                args.artifacts_dir,
            )
            # The SOLE production swapper: per-key update preserves blend + failing targets.
            update_manifest(
                target, staged_version[target], artifacts_dir=args.artifacts_dir
            )
            print(f"  Swapped production {target.upper()} -> {staged_version[target]}")
        if not passing:
            print("  No passing targets; production unchanged.")
        if any_fail:
            print(
                "  FAIL -> zero swaps for failing targets; non-zero exit "
                f"({', '.join(t.upper() for t in failing)})."
            )
    else:
        print("  Dry-run: no production swap performed (--promote required).")

    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
