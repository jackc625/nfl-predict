"""Re-run the FROZEN Stage-1 feature-group rule on corrected gold (Plan 33.2-22, D33.2-15).

WHAT THIS IS, AND WHAT IT IS NOT
--------------------------------
It is the SAME rule, on a DIFFERENT input, under a DIFFERENT objective. Nothing in
``backtest/group_gate_constants.py`` is edited -- not one byte -- and
``tests/unit/test_group_gate_constants_unedited.py`` proves it against the module's own
Phase-30 witness. What changed is what the rule was handed:

  * THE INPUT. The Phase-30 verdict was measured on 2021-2024 gold carrying fabricated Elo,
    hindsight weather and an inverted coverage flag. That gold has been replaced nine times
    over by this phase's ladder; the verdict has to be re-taken before it excludes anything
    from the re-fit. ``config/group_gate_verdict.toml``'s ``excluded_groups`` is read VERBATIM
    at every re-fit (``scripts/promote_models.py``), so a stale verdict shapes the corrected
    models invisibly.
  * THE OBJECTIVE. Phase 30 chose groups by paired CLOSING-LINE CLV lift. A closing line may
    not feed a fit decision after D33.2-03, and choosing which feature families a model is
    FITTED on is a fit decision. This run uses ``objective="outcome_loss"`` -- each model's own
    out-of-sample loss under its own primary metric, with no market line of any timing.
  * THE SEASONS. The screen's split config is DERIVED from the committed partition rule over
    the gold's completed seasons MINUS ``backtest.ev_chain_constants.HOLD_SEASONS_P31``, the
    named spent single-use 2025 hold. Choosing feature groups by their 2025 score would make
    2025 a selection criterion; Plan 33.2-23 excludes it from its own adjudication for exactly
    the same reason.

REUSE, NEVER RE-IMPLEMENT
------------------------
This script composes EXACTLY what ``backtest.group_gate.run_group_gate`` composes --
``run_signal_lift_screen(targets=GRID_TARGETS, groups=GRID_GROUPS,
baseline_exclude_groups=ALL_REGISTERED_GROUPS, ...)`` then ``decide_group_verdicts`` -- with
the two differences above and nothing else. The add-one-in walk-forward, ``select_group_columns``,
the significance primitive, the multiplicity correction and the verdict thresholds are all the
committed ones. A second implementation of the screen would be a second answer, and the whole
point of a pre-registered rule is that nobody re-derives it.

``ALL_REGISTERED_GROUPS`` now also names ``weather_unsupplied`` (Plan 33.2-12) and ``market``
(Plan 33.2-19). Both have ZERO columns in corrected gold, so they contribute nothing to either
leg. That is the correct behaviour and needs no change: the baseline pin is derived from the
registry precisely so a later-registered group is excluded automatically.

THE THREAD PIN, AND WHY A MEASUREMENT NEEDS ONE
-----------------------------------------------
MEASURED 2026-09-22, while building this script's anti-rot guard: the screen's ATS and O/U
legs return DIFFERENT answers at different OpenMP thread counts, and the difference is large
enough to move a VERDICT. At 12 threads all three groups came back DROP; at 8 threads -- the
cap the repo-root ``conftest.py`` applies to every pytest session -- all three came back
UNDETERMINED. The WP leg, which is a LogisticRegression, was bit-identical throughout; the two
XGBoost legs were not. The mechanism is the feature SELECTION: a fold's fitted importances
shift by a hair with the reduction order, which moves which columns clear
``SelectFromModel``'s threshold, which flips a cell between MEASURED and EXCLUDED.

This repository has met this before. Four harness-reproduction test classes were deleted on
2026-09-12 because "a situational-OU delta measured 0.0074 / 0.4424 / 0.4784 at 4 / 1 / 8 BLAS
threads" -- the same defect, then handled by deleting the check. That is not available here:
this run WRITES a verdict that shapes the corrected re-fit, so "it depends on the machine" is
not a limitation to note, it is a defect to close.

So the re-measurement PINS its thread count at :data:`REMEASUREMENT_THREAD_LIMIT` = 1 and
RECORDS it in the written document. One thread is the only setting reproducible on ANY machine:
a pin of 8 is reproducible only where there are eight cores to pin. The cost is small and
measured -- a single leg goes from about 3 s to about 5 s -- and the pin is applied through
``threadpoolctl``, at RUNTIME, so it holds whether or not numerical libraries were already
imported (an environment variable set at module import is too late inside a test process).

The pin is NOT a production change. It wraps this measurement only; no trainer's ``n_jobs``
moves, and the re-fit in Plan 33.2-23 is untouched by it.

THE DOCUMENT THIS WRITES
------------------------
``config/group_gate_verdict.toml``, in ONE write, from ONE generator. The VALUES come from
``backtest.group_gate.render_verdict_toml`` unchanged -- so the rendered block is the same
generator output the Phase-30 document was, float-for-float at the same fixed precision -- and
this script replaces only the leading COMMENT banner (which described a Phase-30 run) and
appends a ``[remeasurement]`` provenance table carrying the objective, the seasons, the gold
generation digest and the frozen-rule commit. Nothing below the banner is hand-edited; a value
that is wrong here is wrong because the measurement was, and the answer is to re-run.

This script writes NOTHING under ``data/``, ``outputs/`` or ``artifacts/``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from backtest.ev_chain_constants import HOLD_SEASONS_P31
from backtest.group_gate import decide_group_verdicts, render_verdict_toml
from backtest.group_gate_constants import GRID_GROUPS, GRID_TARGETS
from backtest.signal_lift import (
    ALL_REGISTERED_GROUPS,
    OBJECTIVE_OUTCOME_LOSS,
    OUTCOME_LOSS_ANCHOR,
    run_signal_lift_screen,
    screen_config_excluding_spent_hold,
)
from tests.phase30_state import MEASUREMENT_COMMIT, PRE_REGISTRATION_COMMIT
from tests.phase33_state import P332_20_CLEAN_BUILD_GOLD_GENERATION

REPO_ROOT = Path(__file__).resolve().parents[1]

#: The ONE file this script writes. Declared here, and nowhere else.
VERDICT_PATH = REPO_ROOT / "config" / "group_gate_verdict.toml"

#: Read-only gold, per target.
GOLD_PATH_FOR = {
    target: REPO_ROOT / "data" / "gold" / f"features_{target}.parquet"
    for target in GRID_TARGETS
}

#: The plan and the owner decision this re-measurement executes.
PLAN_ID = "33.2-22"
DECISION_ID = "D33.2-15"

#: THE THREAD PIN. See the module docstring: the screen's XGBoost legs return different answers
#: at different OpenMP thread counts, by enough to move a verdict, so a measurement that does
#: not pin this is a measurement nobody else can reproduce. ONE is the only value reproducible
#: on any machine -- a pin of 8 needs eight cores to honour it.
REMEASUREMENT_THREAD_LIMIT: int = 1


def load_gold() -> dict[str, pd.DataFrame]:
    """Load the three gold matrices read-only."""
    frames: dict[str, pd.DataFrame] = {}
    for target, path in GOLD_PATH_FOR.items():
        if not path.is_file():
            msg = (
                f"gold matrix {path} is absent. The re-measurement reads gold and writes "
                "nothing under data/; a partial corpus would produce a verdict about a "
                "population that does not exist."
            )
            raise FileNotFoundError(msg)
        frames[target] = pd.read_parquet(path)
    return frames


def remeasure(gold_by_target: dict[str, pd.DataFrame] | None = None) -> dict[str, Any]:
    """Run the frozen rule on corrected gold under the outcome-loss objective.

    Args:
        gold_by_target: Optional pre-loaded gold, for tests. Loaded read-only when None.

    Returns:
        The ``decide_group_verdicts`` structure, plus ``preregistration_commit`` and the
        derived config, so the caller can render and record without re-deriving anything.
    """
    gold_by_target = gold_by_target or load_gold()
    config = screen_config_excluding_spent_hold(gold_by_target)

    # THE THREAD PIN, applied at RUNTIME so it holds inside an already-warm process too.
    # Without it this function returns a different verdict on a 12-core machine than it does
    # under the pytest thread cap -- see the module docstring's measurement.
    with threadpool_limits(limits=REMEASUREMENT_THREAD_LIMIT):
        screen = run_signal_lift_screen(
            gold_by_target=gold_by_target,
            # No closing-odds frame is passed, and under the outcome objective one would RAISE.
            closing_odds_df=None,
            config=config,
            targets=GRID_TARGETS,
            groups=GRID_GROUPS,
            # THE PIN, passed explicitly, exactly as run_group_gate passes it: each group is
            # measured incremental to the NON-SIGNAL CORE, not to a deny-list of three names.
            baseline_exclude_groups=ALL_REGISTERED_GROUPS,
            objective=OBJECTIVE_OUTCOME_LOSS,
        )
    result = decide_group_verdicts(screen)
    # RESOLVED from git by the same function run_group_gate uses, never transcribed. It is
    # cross-checked against the Phase-30 witness below, so a divergence is loud rather than
    # silently rendered into the document.
    from backtest.group_gate import preregistration_commit

    resolved = preregistration_commit()
    if resolved != PRE_REGISTRATION_COMMIT:
        msg = (
            f"the frozen rule's last-modifying commit is {resolved}, not the recorded "
            f"pre-registration {PRE_REGISTRATION_COMMIT}. The rule was EDITED, so this is not "
            "a re-measurement of the pre-registered rule and must not be written."
        )
        raise RuntimeError(msg)
    result["preregistration_commit"] = resolved
    result["remeasurement_config"] = config
    return result


def _toml_int_list(values: object) -> str:
    """Render a sequence of ints as a TOML inline array."""
    return "[" + ", ".join(str(int(value)) for value in values) + "]"  # type: ignore[union-attr]


def _toml_string(value: object) -> str:
    """Render a string as a TOML basic string, escaped deterministically."""
    import json

    return json.dumps("" if value is None else str(value), ensure_ascii=True)


def _banner(result: dict[str, Any], measured_at: str) -> list[str]:
    """The re-measurement's own comment banner, replacing the Phase-30 one."""
    config = result["remeasurement_config"]
    return [
        "# =============================================================================",
        "# config/group_gate_verdict.toml -- the RE-MEASURED Stage-1 feature-group verdict",
        f"# Phase 33.2 Plan {PLAN_ID} ({DECISION_ID}) -- the SAME frozen rule, re-run on",
        "# CORRECTED gold under an objective no market line enters.",
        "#",
        "# GENERATOR OUTPUT. Produced by `python -m scripts.remeasure_group_verdict --apply`,",
        "# which composes exactly what `python -m backtest.group_gate` composes and renders the",
        "# block below with the same generator. Do NOT hand-edit any value: re-run the script.",
        "# A hand-edited value is indistinguishable from a tampered one.",
        "#",
        "# WHAT SUPERSEDED THE PHASE-30 DOCUMENT, AND WHAT DID NOT:",
        f"#   the RULE      : UNCHANGED. {PRE_REGISTRATION_COMMIT}",
        "#                   (backtest/group_gate_constants.py, not one byte edited)",
        "#   the INPUT     : corrected gold, generation",
        f"#                   {P332_20_CLEAN_BUILD_GOLD_GENERATION}",
        "#   the OBJECTIVE : outcome_loss -- each model's OWN out-of-sample loss (WP log loss;",
        "#                   ATS and O/U absolute error). Phase 30 used paired CLOSING-LINE CLV",
        "#                   lift, and a closing line may not feed a fit decision (D33.2-03).",
        f"#   the SEASONS   : {list(config.holdout_seasons)}, derived from the committed",
        f"#                   partition rule minus the spent hold {list(HOLD_SEASONS_P31)}",
        f"#   the THREADS   : pinned at {REMEASUREMENT_THREAD_LIMIT}. The XGBoost legs return",
        "#                   different answers at different thread counts, by enough to move a",
        "#                   verdict, so an unpinned run is one nobody else can reproduce.",
        "#",
        f"# The Phase-30 document it supersedes is recoverable byte-for-byte at {MEASUREMENT_COMMIT}.",
        "# The two statistics are in DIFFERENT UNITS and are compared only through their",
        "# VERDICTS; see GROUP-VERDICT-READOUT.md for the before/after.",
        "#",
        f"# Measured {measured_at}.",
        "#",
        "# ASCII only, no emoji (CLAUDE.md hard constraint).",
        "# =============================================================================",
    ]


def _remeasurement_table(result: dict[str, Any], measured_at: str) -> list[str]:
    """The ``[remeasurement]`` provenance table appended below the rendered block."""
    config = result["remeasurement_config"]
    screen = result.get("screen", {})
    return [
        "",
        "# THE PROVENANCE OF THIS RE-MEASUREMENT, so the verdict above can be reproduced and,",
        "# when the time comes, superseded honestly. A verdict with no gold digest is a verdict",
        "# nobody can reproduce; one with no objective invites being read as a CLV lift.",
        "[remeasurement]",
        f"plan = {_toml_string(PLAN_ID)}",
        f"decision = {_toml_string(DECISION_ID)}",
        f"generator = {_toml_string('python -m scripts.remeasure_group_verdict --apply')}",
        f"objective = {_toml_string(screen.get('objective', OBJECTIVE_OUTCOME_LOSS))}",
        f"objective_anchor = {_toml_string(screen.get('objective_anchor', OUTCOME_LOSS_ANCHOR))}",
        f"measured_seasons = {_toml_int_list(sorted(config.holdout_seasons))}",
        f"selection_seasons = {_toml_int_list(sorted(config.train_seasons))}",
        f"hp_val_seasons = {_toml_int_list(sorted(config.hp_val_seasons))}",
        f"holdout_seasons = {_toml_int_list(sorted(config.holdout_seasons))}",
        f"excluded_hold_seasons = {_toml_int_list(sorted(HOLD_SEASONS_P31))}",
        "# The OpenMP thread count this measurement was pinned to. It is recorded because the",
        "# screen's XGBoost legs return different answers at different thread counts -- by",
        "# enough to move a verdict -- so a run that did not pin this could not be reproduced.",
        f"thread_limit = {int(REMEASUREMENT_THREAD_LIMIT)}",
        f"gold_generation_digest = {_toml_string(P332_20_CLEAN_BUILD_GOLD_GENERATION)}",
        f"frozen_rule_commit = {_toml_string(PRE_REGISTRATION_COMMIT)}",
        f"supersedes_commit = {_toml_string(MEASUREMENT_COMMIT)}",
        f"measured_at = {_toml_string(measured_at)}",
        "# No closing-odds frame was loaded or passed anywhere in this run; under the outcome",
        "# objective one would have raised ClosingLineInScreenError before any leg ran.",
        "closing_line_used = false",
    ]


def render_remeasured_verdict(result: dict[str, Any], measured_at: str) -> str:
    """Render the COMPLETE re-measured verdict document.

    The VALUES are ``render_verdict_toml``'s, unchanged. Only the leading comment banner --
    which described a Phase-30 run -- is replaced, and the ``[remeasurement]`` table appended.
    The banner is identified structurally, as the run of leading ``#`` lines, so this is a
    deterministic operation on COMMENTS and never touches a rendered value.

    Args:
        result: A :func:`remeasure` structure.
        measured_at: The measurement instant, ISO-8601.

    Returns:
        The complete document, terminated by a newline.
    """
    block = render_verdict_toml(result)
    lines = block.split("\n")
    body_start = 0
    while body_start < len(lines) and lines[body_start].startswith("#"):
        body_start += 1
    body = lines[body_start:]
    rendered = [
        *_banner(result, measured_at),
        *body,
        *_remeasurement_table(result, measured_at),
    ]
    return "\n".join(rendered).rstrip("\n") + "\n"


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser (extracted so the argument wiring is unit-testable)."""
    parser = argparse.ArgumentParser(
        description=(
            "Re-run the FROZEN Stage-1 feature-group rule on corrected gold under the "
            "outcome-loss objective, over seasons that exclude the spent 2025 hold, and "
            "re-derive config/group_gate_verdict.toml. Reads gold; writes nothing under "
            "data/, outputs/ or artifacts/."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help=(
            "Write the re-derived verdict to config/group_gate_verdict.toml. Without it the "
            "document is printed and nothing is written."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the re-measurement, print its machine lines, and optionally write the verdict."""
    args = build_parser().parse_args(argv)
    measured_at = datetime.now(UTC).replace(microsecond=0).isoformat()

    result = remeasure()
    document = render_remeasured_verdict(result, measured_at)
    config = result["remeasurement_config"]

    if args.apply:
        VERDICT_PATH.write_text(document, encoding="utf-8")
    else:
        print(document)

    print(f"OBJECTIVE= {OBJECTIVE_OUTCOME_LOSS}")
    print(f"MEASURED_SEASONS= {sorted(config.holdout_seasons)}")
    print(f"SELECTION_SEASONS= {sorted(config.train_seasons)}")
    print(f"HP_VAL_SEASONS= {sorted(config.hp_val_seasons)}")
    print(f"EXCLUDED_HOLD_SEASONS= {sorted(HOLD_SEASONS_P31)}")
    print(f"GOLD_DIGEST= {P332_20_CLEAN_BUILD_GOLD_GENERATION}")
    print(f"FROZEN_RULE_COMMIT= {result['preregistration_commit']}")
    for group in GRID_GROUPS:
        entry = result["verdicts"][group]
        print(f"VERDICT= {group} {entry['verdict']}")
    excluded = sorted(
        group
        for group, entry in result["verdicts"].items()
        if entry["verdict"] != "KEEP"
    )
    print(f"EXCLUDED_GROUPS= {excluded}")
    print(f"VERDICT_WRITTEN= {bool(args.apply)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
