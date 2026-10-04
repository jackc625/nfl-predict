"""Blend weight tuning CLI: ONE fixed weight per target, on lines we owned before the lock.

Run via: python -m backtest.tune

WHAT IT DOES (Plan 33.2-24, D33.2-10)
-------------------------------------
1. Reads the three CORRECTED model artifacts (step 25b's re-fit with the snap coverage flag
   left out, ids recorded in ``config.blend_sources.BLEND_SOURCE_ARTIFACT_IDS``; was Plan
   33.2-23's ``P332_23_REFIT_ARTIFACT_IDS``) for their recipe: parameters, excluded
   feature groups and the gold generation they were fitted on. It reads NO pre-correction
   artifact, NOT ``artifacts/latest.json`` and NOT the incumbent blend.
2. Loads the owned pre-lock tuning corpus (``models.blending_data.load_tuning_period_data``):
   for each 2020-2024 game, the latest owned line timed at or before its lock. A game with
   no such line is EXCLUDED and counted, never filled.
3. Produces each model's REAL walk-forward predictions for every corpus season: for season
   ``S`` the recipe is refit exactly as the trainer fits it, on the partition
   ``conf.season_partition.derive_season_partition`` gives when ``S`` is the latest completed
   season -- feature selection, calibration and the model fit all on seasons before ``S`` --
   and ``S`` is predicted. The OpenMP pool is pinned for every fit and the value recorded.
4. Joins them with the market side: the WP market probability OUT OF FOLD through the bound
   converter's prior-only ``walk_forward_slopes``; the ATS and O/U market as the line itself.
5. Fits ONE weight per target on each model's own outcome loss and writes ONE new
   ``blend_*`` artifact. ``artifacts/latest.json`` is NOT touched: the production swap is
   Plan 33.2-25's.

WHAT IT NO LONGER DOES, AND WHY THERE IS NO FLAG FOR IT
-------------------------------------------------------
This module used to tune a week-varying sigmoid (six parameters) on SYNTHETIC predictions over
2010-2017 nflverse CLOSING lines, and to gate it against a fixed blend on a closing-line CLV
comparison (``--dynamic``, ``--compare``, ``--baselines-dir``, ``--rng-seed``, ``--n-trials``).
Every one of those inputs is dead under D33.2-03, D33.2-07 and the standing ruling that old
baselines are dead, so the tuner, the comparator, its per-target mode gate, its report and
their flags are all DELETED. A week-varying shape may return later only with evidence
measured under the new rule; a flag that reached a deleted function would be the dormant hook
that invites restoring it without the evidence.

``python -m backtest.tune`` therefore has ONE behaviour, and prints ``TUNED_TARGETS= 3`` when
it succeeds.

NO IMPORT FROM THE ``tests`` PACKAGE (A33.2-review WR-06). The recorded ids come from the
committed ``config.blend_sources``, and the live gold generation is passed in on the command
line (``--gold-generation``), exactly as ``models.train --gold-generation`` takes it: measure it
with ``tests.gold_generation.gold_generation_key()`` and pass it. The fit refuses when it
disagrees with the generation the source models were trained on.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from threadpoolctl import threadpool_limits

from backtest.signal_lift import select_group_columns
from conf.season_partition import completed_seasons_from, derive_season_partition
from config.blend_sources import (
    BLEND_CONVERTER_ARTIFACT_ID,
    BLEND_SOURCE_ARTIFACT_IDS,
    BLEND_SOURCE_GOLD_GENERATION,
)
from config.tuning_preregistration import PINNED_THREAD_COUNT
from models.blending import BlendProvenance, MarketBlender, TuningResult
from models.blending_data import (
    EXCLUSION_NO_PRELOCK_LINE,
    EXCLUSION_NO_PRIOR_FOLD_CONVERTER,
    OWNED_TIMELINE_TABLE,
    BlendTuningFrames,
    PrelockTuningCorpus,
    build_tuning_frames,
    load_tuning_period_data,
)
from models.market_probability import load_market_probability_artifact
from models.temporal import TemporalSplitConfig
from models.trainers.ats_trainer import ATSTrainer
from models.trainers.ou_trainer import OUTrainer
from models.trainers.wp_trainer import WPTrainer
from utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "BLEND_CONVERTER_ARTIFACT_ID",
    "BlendSourceError",
    "BlendTuningRun",
    "SourceRecipe",
    "blend_source_artifact_ids",
    "main",
    "read_source_recipes",
    "require_default_recipe",
    "run_blend_tuning",
    "walk_forward_predictions",
]

# BLEND_CONVERTER_ARTIFACT_ID (imported above from ``config.blend_sources``, where it is
# committed -- A33.2-review WR-06) is the converter the blend binds, READ from its recorded
# slot -- never re-derived from a directory listing. Plan 33.2-24 step 24b re-fitted it on the
# repaired owned corpus (1,344 graded games, two 2024 Christmas games the ingest had filed a
# week early now included). Was: ``market_probability_20260923_025709`` (Plan 33.2-21, 1,342
# games), which stays on disk untouched. The blend records it and
# ``MarketBlender.from_artifacts`` cross-checks it against the directory.

#: The targets, in the order everything here is reported.
BLEND_TARGETS: tuple[str, ...] = ("wp", "ats", "ou")

_TRAINER_BY_TARGET: dict[str, type] = {
    "wp": WPTrainer,
    "ats": ATSTrainer,
    "ou": OUTrainer,
}


class BlendSourceError(Exception):
    """A source artifact, its recipe or the gold it names cannot support an honest fit.

    Inherits ``Exception`` rather than ``ValueError`` / ``KeyError`` so no degrade-quietly
    catch tuple turns "the inputs are not what the blend claims" into "tune anyway".
    """


@dataclass(frozen=True)
class SourceRecipe:
    """What one corrected model artifact says about how it was fitted.

    Attributes:
        target: ``wp`` / ``ats`` / ``ou``.
        artifact_id: The artifact directory the recipe was read from.
        best_params: The hyperparameters the artifact was fitted with.
        exclude_groups: The feature groups withheld from its candidate pool.
        gold_generation_digest: The gold generation it was fitted on.
    """

    target: str
    artifact_id: str
    best_params: dict[str, Any]
    exclude_groups: tuple[str, ...]
    gold_generation_digest: str


@dataclass(frozen=True)
class BlendTuningRun:
    """Everything one fit produced, for the CLI and the readout to report."""

    artifact_dir: Path
    result: TuningResult
    provenance: BlendProvenance
    corpus: PrelockTuningCorpus
    frames: BlendTuningFrames
    converter_artifact_id: str
    converter_slope_beta: float


def blend_source_artifact_ids() -> dict[str, str]:
    """The three corrected model artifacts, read from the recorded slot -- never re-derived.

    ``artifacts/latest.json`` still names the dead pre-correction models until Plan
    33.2-25's swap, and a directory listing would silently pick whatever was written last.
    """
    return dict(BLEND_SOURCE_ARTIFACT_IDS)


def read_source_recipes(
    artifact_ids: dict[str, str],
    artifacts_dir: Path,
) -> dict[str, SourceRecipe]:
    """Read EXACTLY the named artifacts' ``metadata.json`` -- nothing else under artifacts/.

    Raises:
        BlendSourceError: when an artifact is absent, names a different target, or lacks
            the recipe fields the walk-forward needs.
    """
    recipes: dict[str, SourceRecipe] = {}
    for target in BLEND_TARGETS:
        if target not in artifact_ids:
            msg = f"no source artifact id was given for {target!r}"
            raise BlendSourceError(msg)
        artifact_id = artifact_ids[target]
        path = Path(artifacts_dir) / artifact_id / "metadata.json"
        if not path.exists():
            msg = (
                f"source artifact {artifact_id!r} for {target!r} is absent ({path}); the "
                "blend is tuned on the corrected models' predictions and cannot stand in "
                "another model's."
            )
            raise BlendSourceError(msg)
        metadata = json.loads(path.read_text(encoding="utf-8"))
        if metadata.get("target") != target:
            msg = f"{artifact_id!r} records target {metadata.get('target')!r}, not {target!r}"
            raise BlendSourceError(msg)
        missing = [
            key
            for key in ("best_params", "exclude_groups", "gold_generation_digest")
            if key not in metadata
        ]
        if missing:
            msg = f"{artifact_id!r} metadata lacks {missing}"
            raise BlendSourceError(msg)
        recipes[target] = SourceRecipe(
            target=target,
            artifact_id=artifact_id,
            best_params=dict(metadata["best_params"]),
            exclude_groups=tuple(str(g) for g in metadata["exclude_groups"]),
            gold_generation_digest=str(metadata["gold_generation_digest"]),
        )
    return recipes


def common_gold_generation(
    recipes: dict[str, SourceRecipe],
    recorded_gold_generation: str,
) -> str:
    """The ONE gold generation all three source models were fitted on.

    Raises:
        BlendSourceError: when the three disagree, or when they disagree with the recorded
            re-fit generation.
    """
    digests = {recipe.gold_generation_digest for recipe in recipes.values()}
    if len(digests) != 1:
        msg = (
            "the three source artifacts record DIFFERENT gold generations "
            f"{sorted(digests)}; a blend fitted across them would name no one gold."
        )
        raise BlendSourceError(msg)
    (digest,) = digests
    if digest != recorded_gold_generation:
        msg = (
            f"the source artifacts' gold generation {digest} is not the recorded re-fit "
            f"generation {recorded_gold_generation}."
        )
        raise BlendSourceError(msg)
    return digest


def require_default_recipe(recipe: SourceRecipe) -> None:
    """Refuse a recipe whose parameters are not the trainer's standard defaults.

    The walk-forward refits each recipe with ``tune=False``, i.e. on the trainer's own
    ``_get_default_params()``. All three corrected artifacts ship on exactly those (Plan
    33.2-23: no search beat its random baseline, so the pre-registered rule fell back to
    defaults). If a recipe ever carries tuned parameters, refitting it on defaults would
    produce predictions from a DIFFERENT model than the one the blend claims -- so it is
    refused by name rather than silently substituted.

    Raises:
        BlendSourceError: when ``recipe.best_params`` differs from the trainer defaults.
    """
    defaults = _TRAINER_BY_TARGET[recipe.target]()._get_default_params()
    if recipe.best_params != defaults:
        msg = (
            f"{recipe.artifact_id!r} was fitted with {recipe.best_params}, not the "
            f"{recipe.target} trainer's default parameters {defaults}. The walk-forward "
            "refits the recipe on the defaults, so it would predict with a different model "
            "than the one this blend names."
        )
        raise BlendSourceError(msg)


def walk_forward_predictions(
    target: str,
    gold_df: pd.DataFrame,
    recipe: SourceRecipe,
    seasons: list[int],
) -> pd.DataFrame:
    """The corrected recipe's REAL out-of-sample predictions, one season at a time.

    For each season ``S`` the committed partition rule is applied to the completed seasons up
    to and including ``S`` (``derive_season_partition``), so ``S`` is its latest holdout
    season. The trainer then runs exactly as it does for the shipped model -- feature
    SELECTION on the selection window, calibration / residual conversion on the hp-val fold,
    the model fitted on every season before ``S`` -- and predicts ``S``. Nothing that informs
    a season's prediction saw that season.

    The shipped artifact's own feature LIST is deliberately NOT reused: it was selected on
    2002-2022, so reusing it for a 2020 prediction would let the selection see the season it
    is predicting. What is reused is the RECIPE -- parameters, excluded groups, selection
    rule -- which is what the corrected model is.

    Args:
        target: ``wp`` / ``ats`` / ``ou``.
        gold_df: That target's gold matrix.
        recipe: The source artifact's recipe (:func:`read_source_recipes`).
        seasons: The seasons to predict.

    Returns:
        ``game_id``, ``season``, ``prediction``, ``actual`` -- one row per game.
    """
    require_default_recipe(recipe)
    features = (
        select_group_columns(gold_df, None, exclude_groups=recipe.exclude_groups)
        if recipe.exclude_groups
        else gold_df
    )
    completed = completed_seasons_from(features["season"])

    frames: list[pd.DataFrame] = []
    for season in sorted(seasons):
        partition = derive_season_partition([s for s in completed if s <= season])
        if partition.holdout[-1] != season:
            msg = (
                f"season {season} is not a completed season of the {target} gold, so it "
                "has no honest walk-forward prediction"
            )
            raise BlendSourceError(msg)
        config = TemporalSplitConfig(
            train_seasons=list(partition.selection),
            hp_val_seasons=list(partition.hp_val),
            holdout_seasons=[season],
        )
        trainer = _TRAINER_BY_TARGET[target](config=config)
        fold = features[features["season"] <= season]
        result = trainer.train_and_evaluate(fold, closing_odds_df=None, tune=False)
        predicted = result["holdout_predictions"]
        frames.append(predicted[predicted["season"] == season])
        logger.info(
            "Walk-forward season predicted",
            target=target,
            season=season,
            n_games=int((predicted["season"] == season).sum()),
            train_seasons=f"{partition.selection[0]}-{season - 1}",
        )

    return pd.concat(frames, ignore_index=True)


def _gold_predictions_fn(
    gold_dir: Path,
) -> Callable[[str, SourceRecipe, list[int]], pd.DataFrame]:
    """The production prediction source: gold on disk, refused if it carries a provisional row."""

    def predict(target: str, recipe: SourceRecipe, seasons: list[int]) -> pd.DataFrame:
        from features.elo_features import assert_no_provisional_training_rows

        gold = pd.read_parquet(Path(gold_dir) / f"features_{target}.parquet")
        # Judged over the rows the walk-forward actually fits: every fold trains and
        # predicts on ``season <= holdout``, so no row past ``max(seasons)`` is used. During
        # the season gold always carries the next slate's provisional rows; judging the
        # whole file refused every re-tune (the 1cfea0a trainer fix, applied here too).
        fitted = gold[gold["season"] <= max(seasons)]
        assert_no_provisional_training_rows(fitted, f"blend-tune:{target}")
        return walk_forward_predictions(target, gold, recipe, seasons)

    return predict


def run_blend_tuning(
    artifacts_dir: Path | str = Path("artifacts"),
    silver_dir: Path | str = Path("data/silver"),
    gold_dir: Path | str = Path("data/gold"),
    *,
    source_artifact_ids: dict[str, str] | None = None,
    converter_artifact_id: str = BLEND_CONVERTER_ARTIFACT_ID,
    recorded_gold_generation: str = BLEND_SOURCE_GOLD_GENERATION,
    predictions_fn: Callable[[str, SourceRecipe, list[int]], pd.DataFrame]
    | None = None,
    gold_generation_fn: Callable[[], str] | None = None,
) -> BlendTuningRun:
    """Fit one fixed blend weight per target and write ONE new blend artifact.

    ``artifacts/latest.json`` is not touched (``save_blend_artifacts``'s default).

    Args:
        artifacts_dir: The artifacts root: source models and the converter are READ from it,
            and the new ``blend_*`` directory is written into it.
        silver_dir: Where the owned timeline and the schedule live.
        gold_dir: Where the gold matrices live.
        source_artifact_ids: ``{wp, ats, ou}`` -> artifact id. Defaults to the recorded
            re-fit (:func:`blend_source_artifact_ids`).
        converter_artifact_id: The converter the blend binds.
        recorded_gold_generation: The gold generation the source models must name.
        predictions_fn: ``(target, recipe, seasons) -> predictions``. Defaults to the real
            walk-forward over ``gold_dir``.
        gold_generation_fn: Returns the live gold's generation. REQUIRED: the CLI passes
            the operator-measured ``--gold-generation``. There is no default, because the
            one producer of the key lives in the ``tests`` package and a production fit
            must not import it (A33.2-review WR-06).

    Returns:
        The :class:`BlendTuningRun`.

    Raises:
        BlendSourceError: when the sources, their recipes or the live gold disagree with
            what the blend would record.
    """
    artifacts_path = Path(artifacts_dir)
    ids = source_artifact_ids or blend_source_artifact_ids()
    recipes = read_source_recipes(ids, artifacts_path)
    gold_digest = common_gold_generation(recipes, recorded_gold_generation)

    if gold_generation_fn is None:
        msg = (
            "no live gold generation was supplied. Measure it with "
            "tests.gold_generation.gold_generation_key() and pass --gold-generation; this "
            "production fit does not import the tests package to measure it for itself."
        )
        raise BlendSourceError(msg)
    live = gold_generation_fn()
    if live != gold_digest:
        msg = (
            f"the gold on disk is generation {live}, but the source models were fitted on "
            f"{gold_digest}. Walk-forward predictions computed from this gold would not be "
            "the corrected models' predictions, and the blend would record a gold "
            "generation it was not fitted on."
        )
        raise BlendSourceError(msg)

    converter = load_market_probability_artifact(converter_artifact_id, artifacts_path)
    corpus = load_tuning_period_data(silver_dir)
    seasons = sorted({int(season) for season in corpus.frame["season"]})

    predict = predictions_fn or _gold_predictions_fn(Path(gold_dir))
    # THE THREAD PIN (Plan 33.2-22): the XGBoost legs answer differently at different OpenMP
    # thread counts, by enough to move a verdict. Applied at RUNTIME so it holds in a warm
    # process, and recorded in the artifact.
    with threadpool_limits(limits=PINNED_THREAD_COUNT):
        predictions = {
            target: predict(target, recipes[target], seasons)
            for target in BLEND_TARGETS
        }

    frames = build_tuning_frames(
        corpus.frame, predictions, converter["walk_forward_slopes"]
    )

    blender = MarketBlender(
        market_probability_artifact_id=converter_artifact_id,
        market_probability_slope_beta=float(converter["slope_beta"]),
    )
    result = blender.tune_weights(frames.frames)

    provenance = BlendProvenance(
        gold_generation_digest=gold_digest,
        source_artifact_ids={t: recipes[t].artifact_id for t in BLEND_TARGETS},
        tuning_corpus_rows=len(corpus.frame),
        excluded_counts={
            EXCLUSION_NO_PRELOCK_LINE: len(corpus.excluded),
            EXCLUSION_NO_PRIOR_FOLD_CONVERTER: len(frames.excluded),
        },
        thread_limit=PINNED_THREAD_COUNT,
    )
    artifact_dir = blender.save_blend_artifacts(
        result, artifacts_path, provenance=provenance
    )

    return BlendTuningRun(
        artifact_dir=artifact_dir,
        result=result,
        provenance=provenance,
        corpus=corpus,
        frames=frames,
        converter_artifact_id=converter_artifact_id,
        converter_slope_beta=float(converter["slope_beta"]),
    )


def _build_cli_parser() -> argparse.ArgumentParser:
    """The CLI: ONE behaviour, one option. Separated from main() so its surface is testable."""
    parser = argparse.ArgumentParser(
        prog="python -m backtest.tune",
        description=(
            "Fit ONE fixed blend weight per target on owned pre-lock lines and write a new "
            "blend artifact. artifacts/latest.json is not touched."
        ),
    )
    parser.add_argument(
        "--artifacts-dir",
        type=str,
        default="artifacts",
        help="Artifacts root the sources are read from and the blend written to",
    )
    parser.add_argument(
        "--gold-generation",
        type=str,
        required=True,
        help=(
            "The live gold generation key, measured with "
            "tests.gold_generation.gold_generation_key(). The fit refuses when it is not "
            "the generation the source models were trained on."
        ),
    )
    return parser


def _print_run(run: BlendTuningRun) -> None:
    """The machine-readable record of what the fit did."""
    result = run.result
    print(f"BLEND_ARTIFACT= {run.artifact_dir.name}")
    print(
        f"CONVERTER= {run.converter_artifact_id} slope_beta={run.converter_slope_beta}"
    )
    print(f"TUNING_CORPUS= {OWNED_TIMELINE_TABLE} rows={len(run.corpus.frame)}")
    for reason, count in run.provenance.excluded_counts.items():
        print(f"EXCLUDED= {reason} {count}")
    for target in BLEND_TARGETS:
        weight = getattr(result.weights, f"{target}_model_weight")
        print(
            f"WEIGHT= {target} {weight:.2f} objective={result.objective_by_target[target]}"
            f" loss={result.loss_by_target[target]:.6f}"
            f" market_only={result.market_only_loss_by_target[target]:.6f}"
            f" model_only={result.model_only_loss_by_target[target]:.6f}"
            f" n={result.n_games[target]}"
            f" seasons={result.seasons_by_target[target]}"
        )
        print(f"SEASON_BEST= {target} {result.season_best_weight_by_target[target]}")
    print(f"BOUNDARY_WEIGHTS= {result.boundary_targets}")
    print(f"THREAD_LIMIT= {run.provenance.thread_limit}")
    print(f"GOLD_GENERATION= {run.provenance.gold_generation_digest}")
    print(f"TUNED_TARGETS= {len(result.objective_by_target)}")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: fit, write, print the record."""
    args = _build_cli_parser().parse_args(argv)
    live_generation = str(args.gold_generation)
    run = run_blend_tuning(
        artifacts_dir=Path(args.artifacts_dir),
        gold_generation_fn=lambda: live_generation,
    )
    _print_run(run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
