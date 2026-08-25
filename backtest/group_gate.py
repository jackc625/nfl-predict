"""The BINDING Stage-1 feature-group judge for Phase 30 (SPEC R4, PROD-01).

This module decides; it measures nothing. ``backtest/signal_lift.py`` produces the paired
add-one-in CLV deltas and ``backtest/diagnose.py`` owns the significance primitive, the per-target
CLV column map and alpha. The judge imports both and adds ONLY the multiplicity correction and the
verdict -- exactly the relationship ``models/deploy_gate.py`` has to ``backtest/diagnose.py``
(D24-13), applied one level down at the feature-group grain (D30-04).

The SCREEN and the JUDGE are deliberately separate modules. ``signal_lift.decide_group_keep`` is
the PERMISSIVE Phase-28 D-05 screen rule (keep unless significantly-negative, grid reported raw,
correction explicitly deferred to this phase). ``decide_group_verdicts`` below REPLACES it for the
binding decision -- it does not extend it. Keeping the two rules in one module is the
screen-versus-deploy conflation this phase exists to end.

The frozen rule itself lives in ``backtest/group_gate_constants.py`` and is one-way after the
pre-registration commit. The mechanics HERE stay editable by design: that split is the whole
reason the constants module can carry a stable git-ancestry assertion (D30-05).

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from scipy.stats import false_discovery_control

# Import-the-primitive parity seam (D24-13, retargeted from models/deploy_gate.py:95-102). These
# symbols are IMPORTED, never re-declared here: this module re-derives NO metric. It must judge on
# the exact same CLV column, sample-size floor and significance level the honest diagnosis uses,
# or the screen and the gate could silently diverge (Pitfall 2, T-30-15).
from backtest.diagnose import (
    CLV_COLUMN_FOR,
    MIN_CLV_SAMPLE,
    SIGNIFICANCE_ALPHA,
    clv_significance,
)
from backtest.group_gate_constants import (
    ALPHA,
    BH_DENOMINATOR,
    CORRECTION_METHOD,
    FIX_CYCLE_LEVERS,
    GRID_GROUPS,
    GRID_TARGETS,
    MDE_POWER,
    VERDICT_DROP,
    VERDICT_KEEP,
    VERDICT_NOT_MEASURED,
    VERDICT_UNDETERMINED,
    bh_family_exclusion_reason,
    bh_rank_order,
    bh_reject,
    mde_for_cell,
)

# The measurement side. ``ALL_REGISTERED_GROUPS`` is the load-bearing baseline pin ``run_group_gate``
# passes EXPLICITLY (the module default is the three-name Phase-28 deny-list, which would silently
# measure something else); the other three are the screen entry points the orchestrator calls.
# ``group_columns`` and ``select_group_columns`` are re-exported rather than called here: they are
# the seam the readout and the Stage-2 exclusion path reach for, and fixing them on this module
# makes the judge's dependency surface part of what the owner ratified.
from backtest.signal_lift import (
    ALL_REGISTERED_GROUPS,
    group_columns,
    run_signal_lift_screen,
    select_group_columns,
)
from utils.paths import reject_data_path

__all__ = [
    "ALL_REGISTERED_GROUPS",
    "ALPHA",
    "BH_DENOMINATOR",
    "CLV_COLUMN_FOR",
    "CORRECTION_METHOD",
    "FIX_CYCLE_LEVERS",
    "GRID_GROUPS",
    "GRID_TARGETS",
    "MDE_POWER",
    "SIGNIFICANCE_ALPHA",
    "VERDICT_DROP",
    "VERDICT_KEEP",
    "VERDICT_NOT_MEASURED",
    "VERDICT_UNDETERMINED",
    "build_parser",
    "clv_significance",
    "decide_group_verdicts",
    "group_columns",
    "main",
    "preregistration_commit",
    "render_verdict_toml",
    "run_group_gate",
    "run_signal_lift_screen",
    "select_group_columns",
]


def _usable_pvalue(cell: dict[str, Any]) -> float | None:
    """The raw two-sided p, or None when it is absent or NaN.

    NaN is DETECTED, never coerced. A zero-variance cell (both legs identical because the group's
    columns were all discarded by feature selection) returns ``t=nan, p=nan`` from
    ``ttest_1samp`` with no warning emitted, and both ``nan <= alpha`` and
    ``false_discovery_control`` handle that badly -- the first silently never rejects, the second
    raises. See T-30-18.
    """
    pvalue = cell.get("delta_p")
    if pvalue is None:
        return None
    pvalue = float(pvalue)
    if math.isnan(pvalue):
        return None
    return pvalue


def decide_group_verdicts(screen_result: dict[str, Any]) -> dict[str, Any]:
    """Apply the FROZEN Stage-1 rule to a signal-lift screen output. PURE.

    A pure function of the screen output, so every arm and edge case is unit-testable on a
    synthetic grid without running two walk-forward trainers per cell.

    Three passes:

    1. **Measurability.** Each (group, target) cell is MEASURED or NOT MEASURED under the frozen
       ``bh_family_exclusion_reason`` predicate, whose check order is itself pre-registered --
       ``n_group_columns == 0`` is tested before the p-value is inspected at all (SPEC R4). The
       cell's own ``paired_sufficient`` and ``measurable`` accounting is CONSUMED, never
       re-derived: ``signal_lift`` already models the small-sample refusal and
       ``clv_significance`` already owns it.
    2. **Correction.** The BH family is assembled from the MEASURED cells ONLY, with ``m`` equal
       to that count, keys ``(group, target)`` and p-values the raw two-sided p taken AS-IS
       (D30-09 -- a two-sided p is roughly twice the one-sided p, so this is strictly
       conservative, and it honours SPEC R4's "no bespoke re-derivation" literally). scipy's
       adjusted q-values are emitted alongside, labelled display-only; the binding decision is the
       vendored inclusive step-up.
    3. **Verdict.** Per group, the four arms are applied in the pre-registered order KEEP, DROP,
       UNDETERMINED, NOT MEASURED. Direction is read from the sign of ``delta_mean`` exactly as
       ``backtest/diagnose.py`` lines 317-322 do, using a STRICT ``> 0.0`` comparison with no
       epsilon anywhere. NOTE the arm order is load-bearing where a group is BH-rejected in both
       directions: KEEP wins, which REVERSES the permissive Phase-28 veto. The DROP arm is the
       simpler pre-registered form -- BH-rejected negative on any target, OR no positive point
       estimate on any target -- so the rule contains no judgement call about which effect sizes
       are "of interest"; every positive-but-unrejected case routes to UNDETERMINED, including one
       exactly equal to its cell's MDE.

    Args:
        screen_result: A ``run_signal_lift_screen`` return dict.

    Returns:
        ``{"screen": <the untouched input>, "verdicts": {group: {...}}, ...}``. The raw screen is
        passed through unmodified so the readout can publish the raw grid beside the corrected
        one, mirroring ``evaluate_target``'s discipline that every metric field was produced
        upstream and the judge adds only the decision.
    """
    groups_in: dict[str, Any] = screen_result.get("groups", {})

    # --- pass 1: measurability -------------------------------------------------------------
    exclusions: dict[tuple[str, str], str | None] = {}
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    for group, group_out in groups_in.items():
        for target, cell in group_out.get("per_target", {}).items():
            key = (group, target)
            cells[key] = cell
            exclusions[key] = bh_family_exclusion_reason(cell)

    # --- pass 2: correction ----------------------------------------------------------------
    measured_keys = sorted(key for key, reason in exclusions.items() if reason is None)
    pvalues = [float(cells[key]["delta_p"]) for key in measured_keys]
    denominator = len(measured_keys)
    rejected = bh_reject(pvalues, measured_keys)
    rank_of = {
        key: rank
        for rank, key in enumerate(bh_rank_order(pvalues, measured_keys), start=1)
    }
    if pvalues:
        q_display = {
            key: float(q)
            for key, q in zip(
                measured_keys, false_discovery_control(pvalues, method="bh")
            )
        }
    else:
        q_display = {}

    # --- pass 3: verdict -------------------------------------------------------------------
    verdicts: dict[str, Any] = {}
    for group, group_out in groups_in.items():
        group_cells: dict[str, Any] = {}
        for target, cell in group_out.get("per_target", {}).items():
            key = (group, target)
            reason = exclusions[key]
            measured = reason is None
            group_cells[target] = {
                "target": target,
                "clv_column": cell.get("clv_column", CLV_COLUMN_FOR.get(target)),
                "measured": measured,
                "exclusion_reason": reason,
                "n_paired": cell.get("n_paired"),
                "n_group_columns": cell.get("n_group_columns"),
                "n_group_columns_selected": cell.get("n_group_columns_selected"),
                "delta_mean": cell.get("delta_mean"),
                "delta_p": _usable_pvalue(cell),
                "delta_ci95": cell.get("delta_ci95"),
                "bh_rejected": bool(rejected.get(key, False)),
                "bh_rank": rank_of.get(key),
                "q_display": q_display.get(key),
                "mde": mde_for_cell(
                    {"n": cell.get("n_paired", 0), "ci95": cell.get("delta_ci95")}
                ),
                "selection_churn": bool(
                    int(cell.get("n_group_columns", 0)) > 0
                    and int(cell.get("n_group_columns_selected", 0)) == 0
                ),
            }

        measured_targets = sorted(t for t, c in group_cells.items() if c["measured"])
        positive_targets = sorted(
            t
            for t in measured_targets
            if group_cells[t]["delta_mean"] is not None
            and group_cells[t]["delta_mean"] > 0.0
        )
        rejected_positive = [
            t for t in positive_targets if group_cells[t]["bh_rejected"]
        ]
        rejected_negative = sorted(
            t
            for t in measured_targets
            if group_cells[t]["bh_rejected"]
            and group_cells[t]["delta_mean"] is not None
            and group_cells[t]["delta_mean"] < 0.0
        )

        largest_positive: dict[str, Any] | None = None
        if positive_targets:
            best = max(positive_targets, key=lambda t: group_cells[t]["delta_mean"])
            mde = group_cells[best]["mde"]
            largest_positive = {
                "target": best,
                "delta_mean": group_cells[best]["delta_mean"],
                "mde": mde,
                # Inclusive at the MDE: an effect exactly equal to its cell's MDE lands INSIDE the
                # UNDETERMINED band (SPEC R4 boundary).
                "at_or_below_mde": (
                    None
                    if mde is None
                    else bool(group_cells[best]["delta_mean"] <= mde)
                ),
            }

        if not measured_targets:
            verdict = VERDICT_NOT_MEASURED
            reasons = sorted(
                {
                    r
                    for t, c in group_cells.items()
                    if (r := c["exclusion_reason"]) is not None
                }
            )
            reason_text = (
                f"NOT MEASURED: no cell of this group entered the BH family ({'; '.join(reasons)}). "
                "This is a refusal to rule, NOT a DROP -- no evidence about this group was "
                f"produced. The CLV sample-size floor is MIN_CLV_SAMPLE={MIN_CLV_SAMPLE}."
            )
        elif rejected_positive:
            verdict = VERDICT_KEEP
            reason_text = (
                f"KEEP: significantly positive after Benjamini-Hochberg correction on "
                f"{rejected_positive} at alpha={ALPHA} with m={denominator} "
                f"({BH_DENOMINATOR} family). Carried into the Stage-2 candidate feature set."
            )
        elif rejected_negative:
            verdict = VERDICT_DROP
            reason_text = (
                f"DROP: significantly NEGATIVE after correction on {rejected_negative}; "
                "this group measurably hurt a target and is dropped, not silently retained."
            )
        elif not positive_targets:
            verdict = VERDICT_DROP
            reason_text = (
                "DROP: no positive point estimate on any measured target (a delta of exactly 0.0 "
                f"is not positive). Measured targets: {measured_targets}."
            )
        else:
            verdict = VERDICT_UNDETERMINED
            reason_text = (
                f"UNDETERMINED: positive point estimate on {positive_targets} but no cell "
                f"survived Benjamini-Hochberg correction at alpha={ALPHA} with m={denominator}. "
                "Resolves to DROP for the deploy decision and is REPORTED as UNDETERMINED -- "
                "'we could not tell', not 'it did not help'."
            )

        verdicts[group] = {
            "verdict": verdict,
            "reason": reason_text,
            "cells": group_cells,
            "measured_targets": measured_targets,
            "positive_targets": positive_targets,
            "rejected_positive_targets": rejected_positive,
            "rejected_negative_targets": rejected_negative,
            "largest_positive_effect": largest_positive,
            "measurability_note": group_out.get("measurability", {}).get("note", ""),
            "coverage_span": group_out.get("coverage_span"),
            "bh_denominator": denominator,
        }

    return {
        "screen": screen_result,
        "verdicts": verdicts,
        "alpha": ALPHA,
        "correction_method": CORRECTION_METHOD,
        "bh_denominator_rule": BH_DENOMINATOR,
        "bh_denominator": denominator,
        "bh_family": list(measured_keys),
        "bh_rank_order": bh_rank_order(pvalues, measured_keys),
        "excluded_cells": {
            f"{group}/{target}": reason
            for (group, target), reason in sorted(exclusions.items())
            if reason is not None
        },
        "mde_power": MDE_POWER,
        "fix_cycle_levers": list(FIX_CYCLE_LEVERS),
        "q_values_are_display_only": True,
    }


# ---------------------------------------------------------------------------
# The Stage-1 ORCHESTRATOR (Plan 30-05)
#
# Everything below is MECHANICS, deliberately outside backtest/group_gate_constants.py. The
# frozen module's last-modifying commit is the SPEC R4 ancestry anchor, so it must change only
# when the RULE changes; a bug in the orchestration, the rendering or the CLI is fixed HERE
# without disturbing that proof.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

# The frozen pre-registration, addressed as a path because its LAST-MODIFYING COMMIT -- not its
# contents -- is what SPEC R4 asserts against.
_FROZEN_RULE_RELPATH = "backtest/group_gate_constants.py"
_UNRESOLVED_COMMIT = "UNRESOLVED"

# Where the owner block-pastes the rendered verdict (Plan 30-10 writes it; this module only
# prints it). config/*.toml is git-tracked; config/*.json would be silently gitignored, which is
# the D24-06/D24-07 anti-gitignore-landmine control that put the frozen baseline inside
# config/gate.toml rather than beside it.
_VERDICT_CONFIG_PATH = "config/group_gate_verdict.toml"

# The CLI's default result path. Under outputs/ (gitignored runtime evidence), NEVER under data/.
_DEFAULT_OUTPUT_PATH = Path("outputs") / "group_gate" / "group_gate_result.json"

# The verdicts whose groups are EXCLUDED from the Stage-2 candidate feature set. Enumerated
# explicitly rather than derived as "not KEEP" so a future fifth verdict word cannot be swept in
# silently -- it would raise instead.
_EXCLUDED_VERDICTS: tuple[str, ...] = (
    VERDICT_DROP,
    VERDICT_UNDETERMINED,
    VERDICT_NOT_MEASURED,
)
_KNOWN_VERDICTS: tuple[str, ...] = (VERDICT_KEEP, *_EXCLUDED_VERDICTS)

# THE FLOAT PRECISION, chosen once and applied to every float this module emits (T-30-54).
#
# ``.17g`` is 17 SIGNIFICANT digits, which is the precision at which no two distinct IEEE-754
# doubles can render identically -- so no two distinct measured values can collide. A fixed
# number of DECIMAL places was rejected: a p-value of 1e-30 and one of 1e-20 would both flatten
# to 0.000000000000 at twelve places, erasing a real difference in the very column the verdict
# turns on.
#
# What must NOT be used is a bare interpolation of the float, which falls through to Python's
# default float formatting. That is shortest-round-trip and is not a contractual, stable
# rendering across platforms or patch releases. Plan 30-10 asserts that the committed
# config/group_gate_verdict.toml is byte-identical to this generator's output, so a formatting
# difference there would read as tampering with a committed measurement artifact rather than as
# the accident it would be.
_FLOAT_FORMAT = ".17g"


def preregistration_commit(repo_root: Path | None = None) -> str:
    """Resolve the FROZEN rule module's last-modifying commit SHA from git.

    RESOLVED, never transcribed. A SHA typed into this source would be a second, silently
    divergable copy of the one fact SPEC R4's ancestry assertion rests on.

    Args:
        repo_root: Repository root to ask git about. Defaults to this file's repository.

    Returns:
        The full 40-character SHA, or ``"UNRESOLVED"`` when git cannot answer (a source
        checkout without git history, for instance). The sentinel is deliberately not an empty
        string: an empty value in the rendered block would read as "no anchor" rather than as
        "the anchor could not be resolved here".
    """
    root = repo_root or _REPO_ROOT
    try:
        completed = subprocess.run(
            ["git", "log", "-1", "--format=%H", "--", _FROZEN_RULE_RELPATH],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return _UNRESOLVED_COMMIT
    return completed.stdout.strip() or _UNRESOLVED_COMMIT


def run_group_gate(
    gold_by_target: dict[str, Any] | None = None,
    closing_odds_df: Any | None = None,
) -> dict[str, Any]:
    """Run the BINDING Stage-1 measurement, correction and verdict in ONE re-runnable call.

    A thin composition and nothing more: the measurement belongs to
    ``backtest/signal_lift.py``, the rule to ``backtest/group_gate_constants.py``, and the
    verdict to ``decide_group_verdicts`` above. This function adds no statistic of its own.

    Args:
        gold_by_target: Optional {target -> widened gold frame}. When None, ``signal_lift``
            loads each target's gold read-only from ``data/gold/features_{target}.parquet``.
        closing_odds_df: Optional normalized closing odds. When None, loaded read-only from
            ``data/silver/odds_snapshot.parquet``.

    Returns:
        The ``decide_group_verdicts`` structure -- ``screen`` (the untouched screen output),
        ``verdicts``, the correction metadata and the frozen scalars -- plus
        ``preregistration_commit``, so the written JSON record names its own rule anchor.
    """
    screen = run_signal_lift_screen(
        gold_by_target=gold_by_target,
        closing_odds_df=closing_odds_df,
        targets=GRID_TARGETS,
        groups=GRID_GROUPS,
        # THE PIN, passed EXPLICITLY (T-30-15). The module default is ``GROUPS``, a deny-list of
        # three names, and a deny-list cannot name a group that does not exist yet: when Phase 29
        # widened gold by fifteen line_movement columns they matched no entry in GROUPS and fell
        # straight through into the BASELINE leg. Nothing failed -- the recorded Phase-28 numbers
        # simply stopped being reproducible, and the situational-OU cell moved from +0.177334 to
        # -0.195371, flipping a published ruling entirely from baseline composition
        # (backtest/signal_lift.py:345-367). Phase 30's binding grid measures each group
        # incremental to the NON-SIGNAL CORE, which is what ALL_REGISTERED_GROUPS names. It is
        # DERIVED from _GROUP_PREDICATE, so it keeps naming line_movement after Plan 30-07 drops
        # those columns -- group_columns then returns [] and the family contributes nothing,
        # which is the correct behaviour and needs no registry change.
        baseline_exclude_groups=ALL_REGISTERED_GROUPS,
    )
    result = decide_group_verdicts(screen)
    result["preregistration_commit"] = preregistration_commit()
    return result


# ---------------------------------------------------------------------------
# The ratified-verdict block (D24-07 block-paste discipline)
# ---------------------------------------------------------------------------


def _fmt_float(value: float | None) -> str:
    """Render one float as a deterministic TOML float literal at the fixed precision.

    A None value renders as the TOML float ``nan``, so the key is always present and the block's
    shape does not change with the data -- a NOT MEASURED cell has no p-value, and an absent key
    would be indistinguishable from a rendering that forgot it.

    Args:
        value: The number, or None for an unmeasurable field.

    Returns:
        A TOML float literal. ``.17g`` can produce an exponent-free, dot-free string for a whole
        number (``2``), which TOML would parse as an INTEGER, so a ``.0`` suffix is appended in
        that one case to keep every emitted numeric field a float.
    """
    if value is None:
        return "nan"
    text = format(float(value), _FLOAT_FORMAT)
    if text in ("nan", "inf", "-inf"):
        return text
    if "." not in text and "e" not in text and "E" not in text:
        text = f"{text}.0"
    return text


def _fmt_str(value: object) -> str:
    """Render one string as a TOML basic string, escaped deterministically."""
    return json.dumps("" if value is None else str(value), ensure_ascii=True)


def _fmt_str_list(values: Iterable[object]) -> str:
    """Render a sequence of strings as a TOML inline array."""
    return "[" + ", ".join(_fmt_str(value) for value in values) + "]"


def _ordered(names: Iterable[str], canonical: tuple[str, ...]) -> list[str]:
    """Order ``names`` by the frozen ``canonical`` sequence, with any extra name sorted after."""
    present = list(names)
    return [name for name in canonical if name in present] + sorted(
        name for name in present if name not in canonical
    )


def render_verdict_toml(result: dict[str, Any]) -> str:
    """Render the COMPLETE ratified Stage-1 verdict as a ready-to-paste TOML block.

    The generator PRINTS; the owner block-pastes the whole thing into
    ``config/group_gate_verdict.toml`` in one operation, so a transcription typo is impossible.
    This mirrors ``scripts/freeze_gate_baseline.render_baseline_toml`` exactly, for the same two
    reasons: there is no TOML *writer* in the standard library, and the block-paste IS the
    discipline (D24-07) rather than a workaround for its absence. Nothing is written here.

    The top-level ``excluded_groups`` key is the seam Stage 2 consumes:
    ``scripts/promote_models._resolve_exclude_groups`` DERIVES ``--exclude-groups`` from it
    instead of an operator typing the list (T-30-26). It is emitted at the document root because
    that is where the consumer reads it, and it is preceded by a comment naming each excluded
    group with its OWN verdict word -- so the exclusion list can never be misread as a collapse
    of the three-valued vocabulary (SPEC R4/R8, T-30-19).

    Args:
        result: A ``run_group_gate`` (or ``decide_group_verdicts``) structure. When it carries
            no ``preregistration_commit`` the SHA is resolved from git.

    Returns:
        The complete block as a string, terminated by a newline. Deterministic: the same
        structure renders byte-identically every time.

    Raises:
        ValueError: If a group carries a verdict word outside the frozen four-value vocabulary.
    """
    verdicts: dict[str, Any] = result.get("verdicts", {})
    screen: dict[str, Any] = result.get("screen", {})
    commit = result.get("preregistration_commit") or preregistration_commit()

    for group, entry in verdicts.items():
        if entry.get("verdict") not in _KNOWN_VERDICTS:
            msg = (
                f"Group '{group}' carries the verdict {entry.get('verdict')!s}, which is not in "
                f"the frozen four-value vocabulary {list(_KNOWN_VERDICTS)}. The exclusion list "
                "must not be derived from a vocabulary this generator does not recognise."
            )
            raise ValueError(msg)

    excluded = sorted(
        group
        for group, entry in verdicts.items()
        if entry["verdict"] in _EXCLUDED_VERDICTS
    )
    ordered_groups = _ordered(verdicts, GRID_GROUPS)
    width = max((len(group) for group in excluded), default=1)

    lines: list[str] = [
        "# =============================================================================",
        f"# {_VERDICT_CONFIG_PATH} -- the RATIFIED Stage-1 feature-group verdict",
        "# Phase 30 (re-fit on widened gold, gated) -- requirement PROD-01, SPEC R4 / R8.",
        "#",
        "# GENERATOR OUTPUT. Produced by `python -m backtest.group_gate` and block-pasted in",
        "# ONE operation (D24-07). Do NOT hand-edit any value below: re-run the generator and",
        "# paste its complete block. A hand-edited value is indistinguishable from a tampered",
        "# one, and Plan 30-10 asserts this file is byte-identical to the generator's output.",
        "#",
        "# The rule that produced these verdicts was FROZEN before any number existed:",
        f"#   frozen rule module : {_FROZEN_RULE_RELPATH}",
        f"#   pre-registration   : {commit}",
        "# SPEC R4 requires that commit to be a STRICT git ancestor of -- and not equal to --",
        "# the commit recording the Stage-1 measurement output.",
        "#",
        "# ASCII only, no emoji (CLAUDE.md hard constraint).",
        "# =============================================================================",
        "",
        "# The Stage-2 exclusion seam, DERIVED and never transcribed:",
        "# scripts/promote_models.py reads this key to build the candidate's --exclude-groups",
        "# list, so what the rule decided and what actually gets trained cannot diverge by a",
        "# typo (T-30-26). Each excluded group with the verdict it actually carries:",
    ]
    if excluded:
        lines.extend(
            f"#   {group.ljust(width)}  {verdicts[group]['verdict']}"
            for group in excluded
        )
    else:
        lines.append("#   (none -- every screened group was KEEP)")
    lines.extend(
        [
            "#",
            "# UNDETERMINED resolves to DROP for the DEPLOY decision and is nonetheless REPORTED",
            "# as UNDETERMINED, never collapsed into DROP (SPEC R4/R8): 'we could not tell' and",
            "# 'we measured it and it did not help' are different findings, and this file must be",
            "# able to say which one happened. NOT MEASURED is not a verdict about the group at",
            "# all -- it is a refusal to rule, and such a group is excluded because no evidence",
            "# for carrying it was produced, not because it was found wanting.",
            f"excluded_groups = {_fmt_str_list(excluded)}",
            "",
            "[stage1]",
            f"requirement = {_fmt_str(screen.get('requirement', 'PROD-01'))}",
            f"frozen_rule_module = {_fmt_str(_FROZEN_RULE_RELPATH)}",
            f"preregistration_commit = {_fmt_str(commit)}",
            f"alpha = {_fmt_float(result.get('alpha'))}",
            f"mde_power = {_fmt_float(result.get('mde_power'))}",
            f"correction_method = {_fmt_str(result.get('correction_method'))}",
            f"bh_denominator_rule = {_fmt_str(result.get('bh_denominator_rule'))}",
            f"bh_denominator = {int(result.get('bh_denominator', 0))}",
            f"measure_window = {_fmt_str(screen.get('measure_window', ''))}",
            f"anchor = {_fmt_str(screen.get('anchor', ''))}",
            f"baseline_excludes = {_fmt_str_list(screen.get('baseline_excludes', []))}",
            f"bh_family = {_fmt_str_list(_cell_ids(result.get('bh_family', [])))}",
            f"bh_rank_order = {_fmt_str_list(_cell_ids(result.get('bh_rank_order', [])))}",
            "# scipy's adjusted q-values are a DISPLAY column only; the binding decision is the",
            "# vendored inclusive step-up in the frozen rule module.",
            "q_values_are_display_only = "
            f"{str(bool(result.get('q_values_are_display_only', True))).lower()}",
            f"fix_cycle_levers = {_fmt_str_list(result.get('fix_cycle_levers', []))}",
            "",
            "# Cells EXCLUDED from the BH family, with the pre-registered reason. Every exclusion",
            "# reduces m, so this table is what makes the denominator above auditable.",
            "[stage1.excluded_cells]",
        ]
    )
    for cell_id, reason in sorted(result.get("excluded_cells", {}).items()):
        lines.append(f"{_fmt_str(cell_id)} = {_fmt_str(reason)}")

    for group in ordered_groups:
        entry = verdicts[group]
        lines.extend(
            [
                "",
                f"[stage1.verdicts.{group}]",
                f"verdict = {_fmt_str(entry['verdict'])}",
                f"reason = {_fmt_str(entry.get('reason', ''))}",
                f"coverage_span = {_fmt_str(entry.get('coverage_span', ''))}",
                f"measured_targets = {_fmt_str_list(entry.get('measured_targets', []))}",
                f"positive_targets = {_fmt_str_list(entry.get('positive_targets', []))}",
                "rejected_positive_targets = "
                f"{_fmt_str_list(entry.get('rejected_positive_targets', []))}",
                "rejected_negative_targets = "
                f"{_fmt_str_list(entry.get('rejected_negative_targets', []))}",
            ]
        )

    for group in ordered_groups:
        cells: dict[str, Any] = verdicts[group].get("cells", {})
        for target in _ordered(cells, GRID_TARGETS):
            cell = cells[target]
            lines.extend(
                [
                    "",
                    f"[stage1.cells.{group}.{target}]",
                    f"clv_column = {_fmt_str(cell.get('clv_column', ''))}",
                    f"measured = {str(bool(cell.get('measured', False))).lower()}",
                    f"exclusion_reason = {_fmt_str(cell.get('exclusion_reason'))}",
                    f"n_paired = {int(cell.get('n_paired') or 0)}",
                    f"n_group_columns = {int(cell.get('n_group_columns') or 0)}",
                    "n_group_columns_selected = "
                    f"{int(cell.get('n_group_columns_selected') or 0)}",
                    f"delta_mean = {_fmt_float(cell.get('delta_mean'))}",
                    f"delta_p = {_fmt_float(cell.get('delta_p'))}",
                    f"delta_ci95_lo = {_fmt_float(_ci_bound(cell.get('delta_ci95'), 0))}",
                    f"delta_ci95_hi = {_fmt_float(_ci_bound(cell.get('delta_ci95'), 1))}",
                    f"bh_rejected = {str(bool(cell.get('bh_rejected', False))).lower()}",
                    # 0 means "not ranked": the cell was excluded from the family, so it has no
                    # rank at all. TOML has no null, and an absent key would be ambiguous.
                    f"bh_rank = {int(cell.get('bh_rank') or 0)}",
                    f"q_display = {_fmt_float(cell.get('q_display'))}",
                    f"mde = {_fmt_float(cell.get('mde'))}",
                    f"selection_churn = {str(bool(cell.get('selection_churn', False))).lower()}",
                ]
            )

    return "\n".join(lines) + "\n"


def _cell_ids(keys: Iterable[Any]) -> list[str]:
    """Render ``(group, target)`` family keys as stable ``group/target`` strings."""
    ids: list[str] = []
    for key in keys:
        group, target = key
        ids.append(f"{group}/{target}")
    return ids


def _ci_bound(ci95: Any, index: int) -> float | None:
    """One end of a 95% CI tuple, or None when the cell has no interval."""
    if ci95 is None:
        return None
    return float(ci95[index])


# ---------------------------------------------------------------------------
# CLI: run the gate, print the grid + the block, write the JSON record
# ---------------------------------------------------------------------------


def _format_gate_report(result: dict[str, Any]) -> str:
    """Render the CORRECTED 9-cell grid and the per-group verdicts as an ASCII report."""
    screen: dict[str, Any] = result.get("screen", {})
    verdicts: dict[str, Any] = result.get("verdicts", {})
    lines: list[str] = [
        "=" * 78,
        "  STAGE 1 -- BINDING feature-group gate (PROD-01, SPEC R4)",
        "=" * 78,
        f"  Anchor          : {screen.get('anchor', 'n/a')}",
        f"  Measure window  : {screen.get('measure_window', 'n/a')}",
        f"  Baseline drops  : {screen.get('baseline_excludes', [])}",
        f"  Alpha           : {result.get('alpha')}  "
        f"({result.get('correction_method')}, m={result.get('bh_denominator')} "
        f"[{result.get('bh_denominator_rule')}])",
        f"  MDE power       : {result.get('mde_power')}",
        f"  Pre-registration: {result.get('preregistration_commit', 'n/a')}",
        "=" * 78,
        "",
    ]
    for group in _ordered(verdicts, GRID_GROUPS):
        entry = verdicts[group]
        lines.append("-" * 78)
        lines.append(f"  GROUP: {group}   VERDICT: {entry['verdict']}")
        lines.append("-" * 78)
        lines.append(
            f"    {'target':<8}{'n_paired':<10}{'delta_mean':<16}{'raw_p':<14}"
            f"{'rank':<6}{'q_disp':<14}{'rejected':<10}{'mde':<16}"
        )
        cells: dict[str, Any] = entry.get("cells", {})
        for target in _ordered(cells, GRID_TARGETS):
            cell = cells[target]
            mean = cell.get("delta_mean")
            pvalue = cell.get("delta_p")
            q_display = cell.get("q_display")
            mde = cell.get("mde")
            lines.append(
                f"    {target:<8}{int(cell.get('n_paired') or 0):<10}"
                f"{('n/a' if mean is None else f'{mean:+.6f}'):<16}"
                f"{('n/a' if pvalue is None else f'{pvalue:.6g}'):<14}"
                f"{(cell.get('bh_rank') or '-')!s:<6}"
                f"{('n/a' if q_display is None else f'{q_display:.6g}'):<14}"
                f"{bool(cell.get('bh_rejected', False))!s:<10}"
                f"{('n/a' if mde is None else f'{mde:.6f}'):<16}"
            )
        lines.append(f"    {entry.get('reason', '')}")
        lines.append("")
    excluded_cells = result.get("excluded_cells", {})
    if excluded_cells:
        lines.append("-" * 78)
        lines.append("  Cells EXCLUDED from the BH family (each one reduces m):")
        for cell_id, reason in sorted(excluded_cells.items()):
            lines.append(f"    {cell_id}: {reason}")
        lines.append("")
    lines.append("=" * 78)
    return "\n".join(lines)


def _reject_data_path(path: Path) -> Path:
    """Resolve an output path, REFUSING anything under ``data/`` (T-30-14, SPEC R1).

    Args:
        path: The requested output path.

    Returns:
        The resolved absolute path.

    Raises:
        ValueError: If the path lands under this repository's (or the current working
            directory's) ``data/`` tree.

    Note:
        WR-07: the implementation now lives in ``utils.paths.reject_data_path``, so the
        two tools that operate directly on the gold tree
        (``scripts/fingerprint_gold.py``, ``scripts/resync_games_duckdb.py``) enforce the
        SAME rule rather than restating the prohibition in prose and not checking it.
        This wrapper keeps the group-gate-specific wording and call signature.
    """
    return reject_data_path(
        path,
        what="the group-gate result",
        suggestion=_DEFAULT_OUTPUT_PATH.as_posix(),
    )


def _jsonable(result: dict[str, Any]) -> dict[str, Any]:
    """Shallow-copy the result with its ``(group, target)`` family keys flattened for JSON."""
    out = dict(result)
    out["bh_family"] = _cell_ids(result.get("bh_family", []))
    out["bh_rank_order"] = _cell_ids(result.get("bh_rank_order", []))
    return out


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so the argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "Run the BINDING Stage-1 feature-group gate: the add-one-in signal-lift screen "
            "over the frozen 3x3 grid with the baseline leg pinned to EVERY registered signal "
            "group, then the frozen Benjamini-Hochberg correction and verdict rule. Prints the "
            "corrected grid and the ready-to-paste ratified-verdict block, and writes the full "
            "result as JSON. READ-ONLY with respect to data/."
        )
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=_DEFAULT_OUTPUT_PATH,
        help=(
            "Where to write the full result JSON. Must NOT be under data/ (SPEC R1). "
            f"Default: {_DEFAULT_OUTPUT_PATH.as_posix()}"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the gate, print the grid and the verdict block, and write the JSON record.

    The output path is validated BEFORE the gate runs: a refusal that arrived after twelve
    walk-forward re-fits would be a refusal nobody could afford to trust.

    Args:
        argv: Command-line arguments. None for sys.argv.

    Returns:
        Exit code 0.

    Raises:
        ValueError: If the requested output path is under ``data/``.
    """
    args = build_parser().parse_args(argv)
    output_path = _reject_data_path(args.output)

    result = run_group_gate()

    print(_format_gate_report(result))
    print()
    print(
        f"Ready-to-paste ratified-verdict block (paste into {_VERDICT_CONFIG_PATH} "
        "in ONE operation; never hand-edit a value):"
    )
    print()
    print(render_verdict_toml(result))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(_jsonable(result), indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Full result written to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
