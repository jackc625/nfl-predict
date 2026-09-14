"""The explicit FINAL FIT: the shipped model, fitted on every completed season.

WHY THIS MODULE EXISTS (D33.1-01, D33.1-02)
-------------------------------------------
``models.temporal.WalkForwardSplitter.generate_splits`` builds every fold as
``season < holdout_season`` over the full gold frame, and each concrete trainer keeps the
LAST fold's model. So the shipped artifact is fitted on everything strictly before the
newest holdout season, and NO choice of the three existing season lists can change that:
widening ``train_seasons`` does not touch the fold mask, and moving the holdout only moves
where the gap sits. D33.1-01 -- the shipped model is fitted on every completed season --
therefore needs a SEPARATE final fit rather than a different list. This is it.

WHERE IT LIVES, AND WHY NOT ON ``BaseTrainer`` (Ruling T)
---------------------------------------------------------
``models/trainers/base.py`` documents ``BaseTrainer.train_and_evaluate`` as unreachable on
every production path -- measured with instrumented production runs, not assumed -- and
``tests/unit/test_feature_selection_stability.py`` fails if a concrete trainer ever stops
overriding it. Adding a REACHABLE method beside it would make that documentation false. So
the entry point is a module BESIDE the three concrete trainers, each of which exposes a
thin ``final_fit`` wrapper. ``BaseTrainer`` gains only a ``preprocessing = None`` class
DEFAULT, which is a default and not reachable behaviour.

THE FOUR PERSISTED COMPONENTS, DECIDED ONE AT A TIME (Ruling S)
---------------------------------------------------------------
"Refit everything on everything" is the wrong default, and the reason is the calibrator.
Each component is decided separately, and the machine-readable form of this table is
:data:`FINAL_FIT_COMPONENT_POLICY` below. The prose is here because the REASON each
component was decided the way it was is the load-bearing part, and a dict alone does not
carry it.

* ``model`` -- REFIT on every completed season. This is the point of D33.1-02 and the only
  reason the entry point exists.

* the WP PREPROCESSING (imputer, missing-indicator, scaler) -- REFIT on the SAME rows as
  the final model, and PERSISTED WITH IT as one inseparable ``Pipeline``. A scaler fitted
  on 2018-2022 applied to a model fitted on 2002-2025 is a mismatch. The pre-D33.1-R3 code
  ALREADY had that mismatch -- the scaler was fitted on ``train_val_split.train_data``
  while the model came from the last fold over ``season < max(holdout)``, so the two never
  shared a row set. Under D33.1-R3 the scaler is the third step of the Pipeline the
  estimator is the fourth step of, so the two CANNOT be given different row sets even
  deliberately. Refitting "the scaler" and refitting "the preprocessing" are the same act
  now, and they are one call: ``Pipeline.fit`` fits every step including the estimator.

* the calibrator / residual converter -- CARRIED OVER unchanged from the walk-forward
  stage. It is fitted on hp-val PREDICTIONS from a model trained on the selection window
  only, which is what makes it OUT-OF-SAMPLE. A final fit that includes the hp_val rows and
  then refits the calibrator on those same rows makes it IN-SAMPLE: the reliability curve
  would look excellent and mean nothing, which is precisely the defect class this milestone
  exists to detect, and WP's entire stated value is calibration. Carrying it over pairs an
  out-of-sample-fitted calibrator with a model whose training set is a SUPERSET, which
  biases mildly toward UNDER-confidence -- a conservative, statable error rather than a
  flattering, unstatable one.

* ``feature_names`` -- CARRIED OVER from the walk-forward stage's selection on the
  selection window. Selection stays where Ruling Q put it, which is what keeps a candidate
  comparable to its incumbent on the axis the deploy gate cares about.

THE REJECTED ALTERNATIVE, AND ITS COST
--------------------------------------
The alternative for the calibrator is HOLDING hp_val OUT of the final fit. It keeps the
calibrator honestly refittable -- but it silently defeats D33.1-01 for one season, because
the shipped model would then be one season short of "every completed season". D33.1-01 is a
LOCKED owner decision and not a default to trade away in an implementer's judgement, so it
was not taken.

The cost of the choice that WAS made is recorded rather than hidden: the shipped model's
calibrator was fitted against a NARROWER model than the one that ships. Plan 33.1-11's
readout carries this, and ``tests.phase33_state.FINAL_FIT_COMPONENT_DECISIONS`` carries it
in committed source so a readout can quote the record rather than re-argue it.

WHAT THIS MODULE DOES NOT DO
----------------------------
It WRITES NOTHING. No file, no artifact directory, no ``artifacts/latest.json`` entry.
Phase 33.1 AUTHORS this mechanism and never runs it on production data; Phase 33's Wave 15
owns the actual re-fit. ``tests/unit/test_final_fit_entry_point.py`` walks the AST of the
four production entry modules and asserts none of them calls any of the three public
symbols here, with a planted-call control proving the scan fires.

ASCII only, no emoji (CLAUDE.md hard constraint).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd
from sklearn.pipeline import Pipeline

from models.trainers.wp_trainer import WP_PIPELINE_STEP_NAMES
from utils import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from conf.season_partition import SeasonPartition

logger = get_logger(__name__)

__all__ = [
    "FINAL_FIT_COMPONENTS_METADATA_KEY",
    "FINAL_FIT_COMPONENT_POLICY",
    "FINAL_FIT_SEASONS_METADATA_KEY",
    "FinalFitResult",
    "apply_final_fit_to_trainer",
    "final_fit_over_completed_seasons",
]


#: Ruling S's table AS DATA. The prose form is this module's docstring; neither is a
#: substitute for the other. A test asserts the four keys, and
#: ``tests.phase33_state.FINAL_FIT_COMPONENT_DECISIONS`` records the same four decisions
#: with the rejected alternative and its cost.
FINAL_FIT_COMPONENT_POLICY: dict[str, str] = {
    "model": (
        "REFIT on every completed season in SeasonPartition.final_fit. This is the point "
        "of D33.1-02 and the only reason this entry point exists: the last walk-forward "
        "fold trains on season < max(holdout), so the newest completed season never "
        "enters the shipped model under any choice of the three season lists."
    ),
    "preprocessing": (
        "REFIT on the SAME rows as the final model, and persisted WITH it as one "
        "inseparable sklearn Pipeline (D33.1-R1). A scaler fitted on the selection window "
        "and applied to a model fitted on the whole corpus is a mismatch; under D33.1-R3 "
        "the scaler and the estimator are steps of one object, so they cannot be given "
        "different row sets even deliberately."
    ),
    "calibrator": (
        "CARRIED OVER unchanged from the walk-forward stage -- it is NOT refitted. It was "
        "fitted on hp-val PREDICTIONS from a model trained on the selection window only, "
        "which is what makes it out-of-sample. Refitting it on rows the final model was "
        "fitted on would make it in-sample: the reliability curve would look excellent and "
        "mean nothing. The carried-over pairing biases mildly toward under-confidence, "
        "which is a conservative and statable error."
    ),
    "feature_names": (
        "CARRIED OVER from the walk-forward stage's selection on the selection window. "
        "Selection stays where Ruling Q put it, which is what keeps a candidate comparable "
        "to its incumbent on the axis the deploy gate measures."
    ),
}

#: The metadata key recording WHICH seasons the shipped object was finally fitted on.
FINAL_FIT_SEASONS_METADATA_KEY: str = "final_fit_seasons"

#: The metadata key recording the PER-COMPONENT provenance of the shipped object.
FINAL_FIT_COMPONENTS_METADATA_KEY: str = "final_fit_components"

# The reason the calibrator is not refitted, as one string, so the provenance dict that
# ships inside metadata.json carries the argument and not just the verdict.
_CALIBRATOR_CARRY_OVER_REASON: str = (
    "The calibration component was fitted on hp-val predictions from a model trained on "
    "the selection window only, which makes it OUT-OF-SAMPLE. The final fit includes those "
    "same hp-val rows, so refitting the calibrator here would make it IN-SAMPLE: it would "
    "look perfectly calibrated while being useless. Carried over by reference instead "
    "(Ruling S, Plan 33.1-10)."
)

# Which attribute holds the trainer's calibration/conversion component, checked in this
# order. ATS and O/U inherit `calibrator` from BaseTrainer and leave it None, so their own
# converter attribute has to be found FIRST or the refusal below would fire on a trainer
# that is perfectly well prepared.
_CALIBRATION_ATTRIBUTES: tuple[str, ...] = (
    "residual_converter",
    "total_converter",
    "calibrator",
)


@dataclass(frozen=True)
class FinalFitResult:
    """The record of one final fit. Purely a record: producing it writes nothing.

    Attributes:
        model: The estimator (or Pipeline) fitted on the final-fit rows.
        preprocessing: The fitted preprocessing object persisted alongside the model, or
            None for a trainer that has none. For WP this IS ``model`` -- the four-step
            Pipeline whose final step is the estimator -- and that identity is the
            structural guarantee D33.1-R1 exists to provide.
        calibrator: The calibration/conversion component, CARRIED OVER BY REFERENCE from
            the walk-forward stage. Never refitted here.
        converter_params: The ATS/O/U converter parameters as plain JSON-able values, or
            None for WP.
        feature_names: The walk-forward stage's selected features, unchanged.
        seasons: The seasons the model was fitted on -- ``partition.final_fit``.
        n_rows: How many rows those seasons carried in the frame handed in.
        components: The per-component provenance dict written into metadata under
            :data:`FINAL_FIT_COMPONENTS_METADATA_KEY`.
    """

    model: Any
    preprocessing: Any | None
    calibrator: Any
    converter_params: dict[str, Any] | None
    feature_names: list[str]
    seasons: tuple[int, ...]
    n_rows: int
    components: dict[str, Any] = field(default_factory=dict)


def _calibration_component(trainer: Any) -> tuple[str, Any]:
    """Return ``(attribute_name, component)`` for *trainer*'s calibration component."""
    for name in _CALIBRATION_ATTRIBUTES:
        if hasattr(trainer, name):
            return name, getattr(trainer, name)
    msg = (
        f"trainer {type(trainer).__name__} exposes none of "
        f"{list(_CALIBRATION_ATTRIBUTES)}, so there is no calibration component to carry "
        "over into the final fit."
    )
    raise ValueError(msg)


def final_fit_over_completed_seasons(
    trainer: Any,
    features_df: pd.DataFrame,
    partition: SeasonPartition,
    *,
    closing_odds_df: pd.DataFrame | None = None,
) -> FinalFitResult:
    """Fit *trainer*'s model on every season in ``partition.final_fit``.

    The final fit differs from the last walk-forward fold in its ROW SET and in nothing
    else: the same construction path, the same recorded best parameters, the same selected
    features, the same calibration component.

    Args:
        trainer: A concrete trainer whose walk-forward stage has ALREADY run. Its selected
            features and its fitted calibration component are both inputs here.
        features_df: The full feature frame, carrying a ``season`` column.
        partition: The season partition from ``conf.season_partition``. Only
            ``final_fit`` is read.
        closing_odds_df: Accepted for signature symmetry with ``train_and_evaluate`` and
            deliberately UNUSED: CLV is a property of walk-forward holdout predictions, and
            a final fit has no out-of-sample fold to compute it over. Computing one here
            would be an in-sample CLV, which is worse than no CLV.

    Returns:
        The :class:`FinalFitResult`. Nothing is written to disk.

    Raises:
        ValueError: If the walk-forward stage has not run (no ``feature_names``, or no
            fitted calibration component), if the target column is absent, or if the
            requested seasons are absent from *features_df*. Each names what was missing.
    """
    del closing_odds_df  # See the docstring: deliberately unused, not forgotten.

    feature_names = list(getattr(trainer, "feature_names", []) or [])
    if not feature_names:
        msg = (
            "final_fit_over_completed_seasons requires trainer.feature_names, and it is "
            "unset or empty. The walk-forward stage must run FIRST: its selection IS the "
            "feature set the final fit uses, and fitting without one would produce an "
            "object that looks like a model and is not one."
        )
        raise ValueError(msg)

    calibration_attribute, calibration_component = _calibration_component(trainer)
    if calibration_component is None:
        msg = (
            f"final_fit_over_completed_seasons requires trainer.{calibration_attribute}, "
            "and it is unset. The walk-forward stage must run FIRST: that component is "
            "fitted on the hp-val fold and is CARRIED OVER unrefitted into the final fit "
            "(Ruling S), so there is nothing to carry over until it exists."
        )
        raise ValueError(msg)

    target_col = trainer._get_target_column()
    if target_col not in features_df.columns:
        msg = (
            f"final_fit_over_completed_seasons needs the target column '{target_col}', "
            f"which is absent from the frame handed in. Present columns include: "
            f"{sorted(features_df.columns)[:12]}."
        )
        raise ValueError(msg)

    seasons = tuple(int(season) for season in partition.final_fit)
    final_fit_rows = features_df[features_df["season"].isin(seasons)]
    if final_fit_rows.empty:
        present = sorted({int(s) for s in features_df["season"].unique()})
        msg = (
            f"final_fit_over_completed_seasons was asked to fit on seasons {list(seasons)}"
            f", and NONE of them is present in the frame, whose seasons are {present}. "
            "Either the frame is the wrong one or the partition's corpus floor was raised "
            "past the data."
        )
        raise ValueError(msg)

    # THE PARTIAL CASE IS REFUSED BY NAME (code review WR-04).
    #
    # The guard above fires only when NONE of the requested seasons is present. The gap it
    # left is the partial one: gold holding 2002-2024 while the rule says 2002-2025 fits on
    # 23 seasons and then records 24 -- into components["final_fit_seasons"], into the
    # model's own description string ("refit on N completed seasons (2002-2025)") and into
    # the persisted FINAL_FIT_SEASONS_METADATA_KEY. A metadata key that names a season the
    # model never saw is the wrong shape in any milestone; in one whose stated purpose is
    # that records must not overstate, it is the defect class itself.
    #
    # Refusing rather than silently narrowing `seasons` is deliberate. A silent narrowing
    # would ship a model fitted on less than the locked D33.1-01 corpus with nothing to
    # notice, which is the same overstatement moved one level down.
    present_seasons = {int(season) for season in final_fit_rows["season"].unique()}
    missing_seasons = [season for season in seasons if season not in present_seasons]
    if missing_seasons:
        msg = (
            f"final_fit_over_completed_seasons was asked for seasons {list(seasons)} but "
            f"{missing_seasons} carry no rows in the frame handed in. Recording them anyway "
            "would make final_fit_seasons -- and the saved artifact's metadata -- name a "
            "season the model never saw. Either hand in a frame that covers the partition, "
            "or change the partition in its own commit; do not fit on less and record more."
        )
        raise ValueError(msg)

    X = final_fit_rows[feature_names]
    y = final_fit_rows[target_col]

    best_params = trainer.metadata.get("best_params")
    best_params_source = "trainer.metadata['best_params']"
    if best_params is None:
        best_params = trainer._get_default_params()
        best_params_source = "trainer._get_default_params()"

    # THE ONE CALL THAT REFITS THE PREPROCESSING TOO.
    #
    # For WP `_create_model` returns the four-step Pipeline (D33.1-R3), and `Pipeline.fit`
    # fits EVERY step on the rows it is handed -- imputer, missing-indicator, scaler and
    # estimator alike. So the preprocessing is refitted on exactly the final-fit rows,
    # which is a DELIBERATE BEHAVIOUR CHANGE from the pre-D33.1-R3 code: there the scaler
    # was fitted on `train_val_split.train_data` while the model came from the last
    # walk-forward fold, so the two had never shared a row set. After D33.1-R1 they cannot
    # be given different row sets even deliberately, because they are one object.
    #
    # For ATS and O/U `_create_model` returns a bare XGBRegressor, there is no
    # preprocessing, and this is an ordinary fit.
    model = trainer._create_model(best_params)
    model.fit(X, y)

    preprocessing = model if isinstance(model, Pipeline) else None
    if preprocessing is not None and trainer.target == "wp":
        observed_steps = tuple(preprocessing.named_steps)
        if observed_steps != WP_PIPELINE_STEP_NAMES:
            msg = (
                f"the WP final fit produced a Pipeline whose steps are {observed_steps}, "
                f"not {WP_PIPELINE_STEP_NAMES}. The persisted preprocessing contract "
                "(D33.1-R1) is declared once in models.trainers.wp_trainer; a divergence "
                "here means the two have drifted apart."
            )
            raise ValueError(msg)

    converter_params_hook = getattr(trainer, "converter_params", None)
    converter_params = (
        converter_params_hook() if callable(converter_params_hook) else None
    )

    components: dict[str, Any] = {
        "model": (
            f"refit on {len(seasons)} completed seasons "
            f"({seasons[0]}-{seasons[-1]}), {len(final_fit_rows)} rows"
        ),
        "preprocessing": (
            "refit with the model as ONE inseparable Pipeline"
            if preprocessing is not None
            else "none -- this trainer has no preprocessing step"
        ),
        "calibrator": (
            f"carried over by reference from trainer.{calibration_attribute}; NOT refitted"
        ),
        "feature_names": (
            f"carried over unchanged from the walk-forward stage's selection "
            f"({len(feature_names)} features)"
        ),
        "calibrator_refitted": False,
        "calibrator_refit_reason": _CALIBRATOR_CARRY_OVER_REASON,
        "calibrator_attribute": calibration_attribute,
        "best_params_source": best_params_source,
        "final_fit_seasons": list(seasons),
        "final_fit_rows": len(final_fit_rows),
        "rejected_alternative": (
            "Holding hp_val OUT of the final fit keeps the calibrator honestly refittable, "
            "but leaves the shipped model one season short of every completed season, "
            "which silently defeats the locked owner decision D33.1-01."
        ),
    }

    logger.info(
        "Final fit completed",
        target=trainer.target,
        seasons=list(seasons),
        rows=len(final_fit_rows),
        n_features=len(feature_names),
        has_preprocessing=preprocessing is not None,
        has_converter_params=converter_params is not None,
        calibrator_refitted=False,
    )

    return FinalFitResult(
        model=model,
        preprocessing=preprocessing,
        calibrator=calibration_component,
        converter_params=converter_params,
        feature_names=feature_names,
        seasons=seasons,
        n_rows=len(final_fit_rows),
        components=components,
    )


def apply_final_fit_to_trainer(trainer: Any, result: FinalFitResult) -> None:
    """Move *trainer*'s state onto *result*, so the EXISTING save path persists it.

    Kept SEPARATE from the fit rather than folded into it: the fit is pure and testable,
    this is the mutation, and a caller that wants the record without the mutation should be
    able to have it.

    ``BaseTrainer.save`` reads ``self.model``, ``self.preprocessing`` and ``self.metadata``.
    Returning a :class:`FinalFitResult` alone would therefore leave the save path
    persisting the LAST FOLD's object while the record described the final fit -- which is
    exactly threat T-33.1-65d. This closes it.

    Args:
        trainer: The trainer the result was produced from.
        result: The record to move onto it.
    """
    trainer.model = result.model
    trainer.preprocessing = result.preprocessing
    trainer.metadata[FINAL_FIT_SEASONS_METADATA_KEY] = result.seasons
    trainer.metadata[FINAL_FIT_COMPONENTS_METADATA_KEY] = result.components

    logger.info(
        "Final-fit state applied to trainer",
        target=trainer.target,
        seasons=list(result.seasons),
        has_preprocessing=result.preprocessing is not None,
    )
