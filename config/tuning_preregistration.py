"""THE FROZEN pre-registration for the Phase-33.2 hyperparameter search (D33.2-17, R13).

ONE-WAY BY CONSTRUCTION, in the shape of ``backtest/group_gate_constants.py``,
``backtest/ev_chain_constants.py`` and ``conf/season_partition.py``. This module IS the
rule, not a description of one. It is READ by the training path and NEVER EDITED once the
search has run: its LAST-MODIFYING COMMIT is the git-ancestry anchor that
``tests/unit/test_tuning_preregistration.py`` asserts must be a STRICT ancestor of the
commit that records the search's output.

Stated plainly because it is easy to forget a plan later: EDITING THIS FILE AFTER A SEARCH
HAS RUN DOES NOT FIX A BUG -- IT DESTROYS THE EVIDENCE. A value that is wrong here is wrong
for the remainder of the phase, and the only legitimate response is to say so in the
readout, not to amend the file. The mechanics that may later need a fix -- the two-arm
runner, the trainers, the CLI -- live OUTSIDE this module, in ``models/trainers/base.py``,
``models/tuning.py`` and ``models/train.py``.

WHAT THE GUARD IS FOR, IN PLAIN ENGLISH
---------------------------------------
Searching a thousand settings and keeping the best one sounds like it can only help. It
cannot only help: with a thousand tries and a small validation fold, the winner is partly
the setting that happened to suit that fold's NOISE. The way to tell a real improvement
from a lucky one is to run the SAME number of tries with settings picked at RANDOM and
require the searched winner to beat the random winner by enough that luck is an unlikely
explanation.

This project has already MEASURED the failure this guards against:
``GATED-REFIT-READOUT.md:401-427`` records its own feature selector picking 6 of 25 (ATS)
and 9 of 25 (O/U) PURE NOISE columns at a 50-column synthetic injection.

WHY THE MARGIN IS READ OFF A SEASON NEITHER ARM SAW
---------------------------------------------------
Both arms optimise the same three temporal CV folds over train+hp_val
(``models/trainers/base.py`` ``_make_objective`` over ``make_temporal_cv_splits``), so the
gap between their in-search best values is a gap between two winners selected on the SAME
noise. It cannot show that the searched winner generalises. See ``OUTER_COMPARISON_RULE``.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from conf.season_partition import derive_season_partition

__all__ = [
    "BEAT_RANDOM_MARGIN_BY_TARGET",
    "EXCLUDED_OUTER_SEASONS",
    "MARGIN_EVIDENCE_BY_TARGET",
    "METRIC_BY_TARGET",
    "NOT_CLEARED_ARM",
    "NOT_CLEARED_RULE",
    "OUTER_COMPARISON_RULE",
    "OWNER_RULING_DATE",
    "OWNER_RULING_Q1",
    "OWNER_RULING_Q2",
    "PINNED_THREAD_COUNT",
    "PRE_REGISTRATION_PATH",
    "PRUNER_CONFIG",
    "RANDOM_COMPARATOR",
    "RANDOM_SAMPLER_SEED",
    "SEARCH_SPACE_BY_TARGET",
    "STUDY_ARM_RANDOM",
    "STUDY_ARM_TPE",
    "TARGETS",
    "TPE_SAMPLER_SEED",
    "TPE_SAMPLER_STARTUP_TRIALS",
    "TRIAL_BUDGET_BY_TARGET",
    "ParamSpec",
    "margin_cleared",
    "outer_comparison_season",
    "search_space_digest",
    "study_name",
]

#: This module's own path, so a consumer or a test names it from ONE place.
PRE_REGISTRATION_PATH: str = "config/tuning_preregistration.py"

#: The three targets this pre-registration covers, in a fixed order.
TARGETS: tuple[str, ...] = ("ats", "ou", "wp")


# ---------------------------------------------------------------------------
# THE OWNER'S TWO RULINGS, RECORDED VERBATIM (2026-09-23).
#
# Both were asked ONE AT A TIME and answered BEFORE any search number existed. They are
# quoted rather than paraphrased because a paraphrase of a pre-registration is not the
# pre-registration.
# ---------------------------------------------------------------------------

#: The date both questions were put and answered.
OWNER_RULING_DATE: str = "2026-09-23"

OWNER_RULING_Q1: str = (
    "Q1 -- how much better must the searched winner be than the random-setting winner? "
    "Owner chose `margin-from-fold-variation`, at TWO noise-widths. Per-target bars, each "
    "twice the measured noise in a head-to-head gap between two settings: WP 0.0067 log "
    "loss; ATS 0.49 points of margin (absolute error); O/U 0.56 points of total (absolute "
    "error). Chance explains a gap that size about 1 time in 40. The supporting "
    "measurements, taken on the corrected gold with threads pinned at 1 and the three "
    "dropped groups excluded: head-to-head noise 0.0033 / 0.245 / 0.277; season-to-season "
    "swing 0.017 / 0.64 / 0.46 over 11 seasons (2014-2024); whole best-to-worst spread "
    "across 8 random settings scored on 2023 (285 games) 0.0055 / 1.63 / 1.47. Rejected by "
    "the owner: one noise-width (a cleared bar would mean little -- chance explains it 1 "
    "time in 6), and a single round number for all three (different units, different "
    "scales). The owner was told BEFORE setting it, in these terms, that the WP bar "
    "(0.0067) is LARGER than the entire best-to-worst range measured across 8 random WP "
    "settings (0.0055), so the WP search will probably NOT clear it and will fall to the "
    "Q2 rule. That consequence is accepted, not overlooked."
)

OWNER_RULING_Q2: str = (
    "Q2 -- what happens when the bar is NOT cleared? Owner chose `fall-back-to-defaults`. "
    "A target whose search fails its bar ships on the library's standard default settings, "
    "and the failure is published per target as a finding. This DEPARTS from the plan's "
    "text, which proposed `adopt-random-winner`. The reasoning the owner accepted: the "
    "random arm's winner is still the best of ~1,000 tries judged on the same games, so it "
    "carries the same luck-fitting the guard exists to catch, and the setting it lands on "
    "is arbitrary within the search space. The plan's objection to defaults -- that today's "
    "live models already run them, so 'no improvement' would read as 'no change' -- does "
    "not hold, because those models are dead by standing ruling and a model re-fit on "
    "corrected gold with a different feature set is a new model whatever its settings."
)


# ---------------------------------------------------------------------------
# THE COMPARATOR (fixed by D33.2-17, not by the owner).
# ---------------------------------------------------------------------------

RANDOM_COMPARATOR: str = (
    "Optuna's RandomSampler, over the SAME search space object and the SAME trial budget "
    "as the tuned (TPE) arm, run FIRST, as a SEPARATE study whose name ends in "
    "'_random' so it can never resume the '_tpe' study (models/tuning.py opens every "
    "study with load_if_exists=True). Same space and same budget are what make the "
    "comparison mean anything: a smaller random budget would rig it in the searched "
    "winner's favour. KNOWN NON-EQUIVALENCE, DECLARED RATHER THAN HIDDEN: "
    "optuna.pruners.HyperbandPruner assigns each trial to a bracket by a crc32 of the "
    "STUDY NAME, so two differently-named arms prune on DIFFERENT schedules even under "
    "one PRUNER_CONFIG. That is harmless here by construction, because NO adoption "
    "decision reads an in-search value -- the decision reads only OUTER_COMPARISON_RULE's "
    "held-out season score."
)

#: The two arm labels. They are the suffix of the study name and the key of the per-arm
#: record written into each artifact's ``{target}_params.json`` ``tuning_metadata``.
STUDY_ARM_TPE: str = "tpe"
STUDY_ARM_RANDOM: str = "random"

#: The RandomSampler seed. Fixed so the baseline reproduces; it is NOT a tuned value.
RANDOM_SAMPLER_SEED: int = 42

#: The TPE sampler's settings, unchanged from what ``models/tuning.py`` has always used.
#: Named here so BOTH arms are described in one place rather than one being a library
#: default nobody wrote down.
TPE_SAMPLER_SEED: int = 42
TPE_SAMPLER_STARTUP_TRIALS: int = 10

#: The ONE pruner configuration BOTH arms use. Identical to what every study in this
#: repository has always run under (``models/tuning.py``), stated here so the two arms
#: cannot be given different pruners by a later edit.
#:
#: Disabling the pruner to make a completed-trial floor reachable was considered and
#: REJECTED: ``models/trainers/base.py`` records the same choice already made for the
#: backtest diagnostic -- production searches under HyperbandPruner, so a search without
#: it would measure something production does not do.
PRUNER_CONFIG: Mapping[str, int] = {
    "min_resource": 1,
    "max_resource": 3,
    "reduction_factor": 3,
}


# ---------------------------------------------------------------------------
# THE METRIC, PER TARGET. The SAME walk-forward validation metric the tuner already
# minimises (``BaseTrainer._compute_cv_score``, overridden by WP for log loss), named
# explicitly so the margin's units are not left to be inferred.
# ---------------------------------------------------------------------------

METRIC_BY_TARGET: Mapping[str, str] = {
    "wp": "mean log loss, probabilities clipped to WP_LOG_LOSS_CLIP (lower is better)",
    "ats": "mean absolute error, points of margin (lower is better)",
    "ou": "mean absolute error, points of total (lower is better)",
}


# ---------------------------------------------------------------------------
# THE MARGIN (owner ruling Q1, 2026-09-23). Per target, in that target's own metric
# units, scaled to the measured noise in a HEAD-TO-HEAD gap between two settings scored
# on the same games -- which is the quantity the adoption gate actually reads.
#
# TWO noise-widths. The owner rejected one width because a bar chance clears about 1 time
# in 6 would mean little, and rejected a single round number for all three because the
# three metrics are on different scales in different units.
# ---------------------------------------------------------------------------

BEAT_RANDOM_MARGIN_BY_TARGET: Mapping[str, float] = {
    "wp": 0.0067,
    "ats": 0.49,
    "ou": 0.56,
}

#: The measurements the bars were scaled to, recorded so the number is justified rather
#: than asserted. Taken READ-ONLY on 2026-09-23 on the corrected gold (generation
#: 484397642530db5b28c49d9234ecfb90e3860783f1b1b41766ab6151a1522597), with OpenMP threads
#: pinned at 1 and the three DROPped feature groups (injury, situational, snap) excluded,
#: by two throwaway probes -- ``measure_season_variation.py`` (11 seasons, 2014-2024, each
#: refit on everything before it and scored once) and ``measure_setting_spread.py`` (8
#: settings drawn from today's space with a fixed seed, each refit on 2002-2022 and scored
#: once on 2023, 285 games). NEITHER probe touched the outer comparison season (2024's
#: score enters only through the season-to-season figure, which is a SCALE and not a
#: selection) and neither touched the spent 2025 hold.
MARGIN_EVIDENCE_BY_TARGET: Mapping[str, Mapping[str, float]] = {
    "wp": {
        "head_to_head_noise": 0.0033,
        "margin_in_noise_widths": 2.0,
        "season_to_season_sd": 0.017,
        "best_to_worst_spread_across_8_settings": 0.0055,
    },
    "ats": {
        "head_to_head_noise": 0.245,
        "margin_in_noise_widths": 2.0,
        "season_to_season_sd": 0.64,
        "best_to_worst_spread_across_8_settings": 1.63,
    },
    "ou": {
        "head_to_head_noise": 0.277,
        "margin_in_noise_widths": 2.0,
        "season_to_season_sd": 0.46,
        "best_to_worst_spread_across_8_settings": 1.47,
    },
}


# ---------------------------------------------------------------------------
# WHAT HAPPENS WHEN THE BAR IS NOT CLEARED (owner ruling Q2, 2026-09-23).
# ---------------------------------------------------------------------------

#: The label the artifact records in ``adopted_arm`` when the bar is missed. It is
#: deliberately NEITHER arm's name: nothing either search found was adopted.
NOT_CLEARED_ARM: str = "defaults"

NOT_CLEARED_RULE: str = (
    "FALL BACK TO DEFAULTS. A target whose searched winner does not beat the random "
    "winner by at least BEAT_RANDOM_MARGIN_BY_TARGET on the OUTER_COMPARISON_RULE season "
    "ships on the trainer's STANDARD DEFAULT parameters -- `_get_default_params()`, the "
    "same settings a `--no-tune` fit uses and the settings the plan's own option text "
    "meant by 'library defaults'. The per-target failure is PUBLISHED as a finding, in "
    "the artifact's tuning record (`margin_cleared = false`, `adopted_arm = "
    "'defaults'`), in the run's stdout and in the plan's SUMMARY. It is never absorbed "
    "silently. THE READING OF 'LIBRARY DEFAULTS' IS PRE-REGISTERED HERE, BEFORE ANY "
    "SEARCH NUMBER EXISTS, because the phrase is ambiguous: it means this repository's "
    "own `_get_default_params()` and NOT the estimator class's bare constructor "
    "defaults. The owner's ruling was given against the plan's option text, which "
    "equated 'library defaults' with what today's deployed artifacts run, and those "
    "artifacts were fitted through `_get_default_params()`. The random arm's winner is "
    "NOT adopted: it is still the best of ~1,000 tries judged on the same games, so it "
    "carries the same luck-fitting the guard exists to catch, and where it lands is "
    "arbitrary within the search space."
)


# ---------------------------------------------------------------------------
# WHERE THE MARGIN IS MEASURED.
# ---------------------------------------------------------------------------

#: Seasons that may NEVER be the outer comparison season, with the reason.
EXCLUDED_OUTER_SEASONS: tuple[int, ...] = (2025,)

OUTER_COMPARISON_RULE: str = (
    "The margin is read off ONE season NEITHER arm ever saw, never off the three folds "
    "both arms optimised. Each arm's best setting is REFIT on the same train+hp_val data "
    "and scored ONCE on the FIRST HOLDOUT SEASON as "
    "conf.season_partition.derive_season_partition defines it over the gold's own "
    "completed seasons -- DERIVED, never typed (2024 on today's corpus). The in-search "
    "best values are recorded and decide nothing: two winners selected on the same noise "
    "can differ by noise alone, so their gap cannot show generalisation. 2025 IS EXCLUDED "
    "BY NAME: it is the spent single-use profitability hold "
    "(config/profitability_2025_run_ledger.toml, state 'completed'), and choosing a "
    "hyperparameter setting by its 2025 score would make 2025 a selection criterion. If "
    "the derived first holdout season is ever 2025, outer_comparison_season RAISES rather "
    "than quietly picking another season. A nested walk-forward over several outer "
    "seasons was considered and NOT taken: it multiplies a 1,000-trial search per arm per "
    "target by the outer-fold count for a comparison one clean unseen season already "
    "answers. THE COST, STATED RATHER THAN HIDDEN: the outer season is also the first "
    "season of the reported walk-forward holdout, so that season's reported metrics are "
    "no longer fully clean for the adopted model -- the hyperparameters saw it once. It "
    "was preferred to the alternatives because every other candidate season is either "
    "inside the search (train+hp_val) or is the spent 2025 hold."
)


# ---------------------------------------------------------------------------
# THE TRIAL BUDGET, IN TRIALS **STARTED**.
#
# WHY STARTED AND NOT COMPLETED. ``OptunaTuner.optimize`` computes remaining trials as
# ``max(0, n_trials - len(study.trials))``, i.e. in trials STARTED, while
# ``count_completed_trials`` counts ``COMPLETE`` only, deliberately (WR-08). Under
# ``HyperbandPruner``, whose whole purpose is to prune most trials, a 1,000-started search
# can legitimately complete far fewer than 900 -- so a completed-trial floor is either
# unreachable or meaningless. STARTED is the only count both arms can be held EQUAL on.
# Completed, pruned and failed counts are RECORDED and PUBLISHED, never a pass bar.
#
# WHY IT IS PRE-REGISTERED AT ALL. ``models/tuning.py`` and ``models/trainers/base.py``
# BOTH default ``n_trials`` to 100. A plan that says "deep search" without setting the
# budget ships a hundred-trial search and nothing in the run says otherwise.
# ---------------------------------------------------------------------------

TRIAL_BUDGET_BY_TARGET: Mapping[str, int] = {
    "wp": 1000,
    "ats": 1000,
    "ou": 1000,
}


# ---------------------------------------------------------------------------
# THE THREAD PIN.
#
# MEASURED by Plan 33.2-22: the XGBoost legs return DIFFERENT answers at different OpenMP
# thread counts, by enough to move a verdict (12 threads and the 8-thread pytest cap
# disagreed). A result nobody else can reproduce is not a result, so every number this
# pre-registration governs -- both search arms, the outer comparison and the final fits --
# runs under this pin, and the value is recorded in each artifact's metadata.
# ---------------------------------------------------------------------------

PINNED_THREAD_COUNT: int = 1


# ---------------------------------------------------------------------------
# THE SEARCH SPACE.
#
# It lives HERE and not inline in the three trainers' ``suggest_*`` calls for two reasons
# that cannot both be had while the bounds are literals: the widening D33.2-17 requires
# happens ONCE, visibly, in an ancestry-tested file; and the RandomSampler baseline
# searches the IDENTICAL space BY CONSTRUCTION rather than by a second declaration that
# can drift.
#
# EVERY BOUNDED PARAMETER RECORDS TODAY'S BOUND BESIDE THE NEW ONE, so the widening is
# VISIBLE rather than asserted. The one parameter that could not be widened says so and
# says why, rather than being quietly omitted.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamSpec:
    """One searched hyperparameter: its new bound, today's bound, and whether it widened.

    Attributes:
        kind: ``"float"``, ``"int"`` or ``"categorical"``.
        low: Lower bound (numeric kinds only).
        high: Upper bound (numeric kinds only).
        log: Whether the numeric range is searched on a log scale.
        choices: The categorical options (categorical kind only).
        current: TODAY'S bound, verbatim, as the three trainers carried it inline before
            this module existed. ``None`` means the parameter was NOT searched at all
            before, so its presence here is itself a widening.
        widened: Whether this parameter's admissible set is strictly LARGER than
            ``current``. False is legal only with a ``note`` saying why.
        note: Why a parameter did not widen, when it did not.
    """

    kind: str
    low: float | int | None = None
    high: float | int | None = None
    log: bool = False
    choices: tuple[Any, ...] | None = None
    current: str | None = None
    widened: bool = True
    note: str = ""


# The XGBoost space ATS and O/U share (they shared it before this module too -- D-08:
# one space, Optuna finds different optima per target).
_XGB_SPACE: Mapping[str, ParamSpec] = {
    "learning_rate": ParamSpec(
        kind="float", low=0.001, high=0.5, log=True, current="0.005 - 0.3 (log)"
    ),
    "max_depth": ParamSpec(kind="int", low=1, high=9, current="2 - 8"),
    "n_estimators": ParamSpec(kind="int", low=25, high=650, current="50 - 500"),
    "subsample": ParamSpec(kind="float", low=0.4, high=1.0, current="0.5 - 1.0"),
    "colsample_bytree": ParamSpec(kind="float", low=0.3, high=1.0, current="0.5 - 1.0"),
    "min_child_weight": ParamSpec(kind="int", low=0, high=40, current="1 - 10"),
    "reg_alpha": ParamSpec(
        kind="float", low=1e-6, high=100.0, log=True, current="1e-4 - 10.0 (log)"
    ),
    "reg_lambda": ParamSpec(
        kind="float", low=1e-6, high=100.0, log=True, current="1e-4 - 10.0 (log)"
    ),
    # NEW in this pre-registration: gamma was fixed at XGBoost's 0.0 and never searched,
    # so adding it widens the space along an axis that did not exist before.
    "gamma": ParamSpec(
        kind="float", low=0.0, high=5.0, current=None, note="not searched before"
    ),
}

SEARCH_SPACE_BY_TARGET: Mapping[str, Mapping[str, ParamSpec]] = {
    "ats": _XGB_SPACE,
    "ou": _XGB_SPACE,
    "wp": {
        "C": ParamSpec(
            kind="float", low=1e-5, high=1000.0, log=True, current="0.001 - 100.0 (log)"
        ),
        "penalty": ParamSpec(
            kind="categorical",
            choices=("l1", "l2", "elasticnet"),
            current="['l1', 'l2', 'elasticnet']",
            widened=False,
            note=(
                "the three penalties this estimator supports alongside a searchable "
                "solver ARE the whole admissible set; there is no fourth value to add. "
                "sklearn's 'no penalty' option is spelled penalty=None and is not a "
                "member of the categorical domain. Recorded as NOT widened rather than "
                "quietly omitted."
            ),
        ),
        "l1_ratio": ParamSpec(
            kind="float",
            low=0.0,
            high=1.0,
            current="0.0 - 1.0",
            widened=False,
            note=(
                "l1_ratio's OWN domain is the closed interval [0, 1]; the existing bound "
                "already IS that domain, so there is nothing to widen. Recorded rather "
                "than dropped, because a silently-absent parameter and an un-widenable "
                "one look identical in a diff."
            ),
        ),
        "solver_l2": ParamSpec(
            kind="categorical",
            choices=("lbfgs", "saga", "newton-cg"),
            current="['lbfgs', 'saga']",
        ),
    },
}


def search_space_digest(target: str) -> str:
    """Return a sha256 over the CANONICAL render of *target*'s search space.

    Written into each artifact's tuning record so a reader can prove both arms searched
    the space this committed file declares, without diffing source.

    Args:
        target: ``"wp"``, ``"ats"`` or ``"ou"``.

    Returns:
        64-character hex digest.
    """
    space = SEARCH_SPACE_BY_TARGET[target]
    canonical = {
        name: {
            "kind": spec.kind,
            "low": spec.low,
            "high": spec.high,
            "log": spec.log,
            "choices": list(spec.choices) if spec.choices is not None else None,
        }
        for name, spec in sorted(space.items())
    }
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def study_name(target: str, tag: str, arm: str) -> str:
    """Return the Optuna study name for one arm of one target's search.

    The two arms MUST be two distinct names: ``OptunaTuner.optimize`` opens every study
    with ``load_if_exists=True``, so a random arm sharing the TPE arm's name would resume
    it and measure nothing.

    Args:
        target: ``"wp"``, ``"ats"`` or ``"ou"``.
        tag: The per-phase ``TUNING_STUDY_TAG``.
        arm: :data:`STUDY_ARM_TPE` or :data:`STUDY_ARM_RANDOM`.

    Returns:
        ``"{target}_tuning_{tag}_{arm}"``.

    Raises:
        ValueError: If *arm* is not one of the two declared arms.
    """
    if arm not in (STUDY_ARM_TPE, STUDY_ARM_RANDOM):
        msg = (
            f"arm must be {STUDY_ARM_TPE!r} or {STUDY_ARM_RANDOM!r}, got {arm!r}. A third "
            "arm would be a search this pre-registration does not describe."
        )
        raise ValueError(msg)
    return f"{target}_tuning_{tag}_{arm}"


def outer_comparison_season(completed_seasons: Iterable[object]) -> int:
    """Return the season the adoption gate scores on: the FIRST holdout season.

    DERIVED from the committed season-partition rule over the gold's own completed
    seasons, never typed. See :data:`OUTER_COMPARISON_RULE`.

    Args:
        completed_seasons: Any season-bearing iterable -- a gold frame's ``season``
            column is accepted directly.

    Returns:
        The first holdout season.

    Raises:
        ValueError: If the derived season is one of :data:`EXCLUDED_OUTER_SEASONS`.
            Refusing is the point: quietly substituting another season would make the
            measurement site a thing chosen after the fact.
    """
    season = int(derive_season_partition(completed_seasons).holdout[0])
    if season in EXCLUDED_OUTER_SEASONS:
        msg = (
            f"the derived first holdout season is {season}, which is EXCLUDED from the "
            f"outer comparison ({list(EXCLUDED_OUTER_SEASONS)}): it is the spent "
            "single-use profitability hold, and choosing a hyperparameter setting by its "
            "score would make it a selection criterion. This is a REFUSAL, not a "
            "fallback -- silently picking a different season would move the measurement "
            "site after the rule was frozen."
        )
        raise ValueError(msg)
    return season


def margin_cleared(
    target: str, tpe_outer: float, random_outer: float
) -> tuple[bool, float]:
    """Decide whether *target*'s searched winner beat the random winner by the bar.

    Both scores are LOWER-IS-BETTER in every target's metric (see
    :data:`METRIC_BY_TARGET`), so the gap in the searched winner's favour is
    ``random_outer - tpe_outer``.

    Args:
        target: ``"wp"``, ``"ats"`` or ``"ou"``.
        tpe_outer: The TPE arm's best setting, refit and scored ONCE on the outer season.
        random_outer: The random arm's best setting, scored the same way.

    Returns:
        ``(cleared, gap)`` -- whether the gap reached the pre-registered bar, and the gap.
    """
    gap = float(random_outer) - float(tpe_outer)
    return gap >= BEAT_RANDOM_MARGIN_BY_TARGET[target], gap
